"""Train Qwen3 8B with GRPO for budget-aware interaction scaling.

Uses trl's GRPOTrainer to fine-tune Qwen3 8B so the model learns:
1. When to request environment feedback vs continue reasoning
2. Budget awareness — different behavior under different step limits
3. When to stop iterating (calibrated confidence from grounded signals)
4. The think-do-review pattern as an internalized capability

The reward signal comes from grounded feedback:
- Code tasks: test pass/fail (binary)
- Visual tasks: VLM quality score (continuous)
- Research tasks: factual accuracy (continuous)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def _load_model_for_training(model_name: str):
    """Load Qwen3-8B in 4-bit with LoRA adapters (QLoRA) to coexist with other GPU jobs.

    If `model_name` points to a saved PEFT adapter directory (has adapter_config.json),
    load the base Qwen3-8B in 4-bit and apply that adapter as the starting point,
    keeping the LoRA params trainable. Otherwise train a fresh adapter.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    adapter_path = Path(model_name)
    is_adapter = (adapter_path / "adapter_config.json").exists()

    if is_adapter:
        import json as _json
        with open(adapter_path / "adapter_config.json") as f:
            adapter_cfg = _json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-8B")
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            attn_implementation="kernels-community/flash-attn",
            device_map={"": 0},
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation="kernels-community/flash-attn",
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""

    model_name: str = "Qwen/Qwen3-8B"
    output_dir: str = "models/qwen3-8b-interaction-scaling"
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 6144
    max_prompt_length: int = 2048
    max_completion_length: int = 4096
    num_generations: int = 8  # Group size for GRPO (typical papers: 8-16)
    beta: float = 0.1  # KL penalty coefficient
    seed: int = 42


def build_reward_function(task_type: str):
    """Build a reward function for GRPO based on task type.

    The reward function executes the model's generated trajectory
    and returns a reward based on grounded feedback.
    """
    if task_type == "code":
        return _code_reward
    elif task_type in ("slide", "webpage", "animation"):
        return _visual_reward
    elif task_type == "video":
        return _video_reward
    elif task_type == "research":
        return _factual_reward
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _code_reward(completion: str, task: dict) -> float:
    """Reward for code tasks: execute and check test results."""
    from src.evaluation.code_eval import CodeEvaluator
    from src.utils.code_utils import extract_code

    code = extract_code(completion, "python")
    if not code.strip():
        return 0.0

    evaluator = CodeEvaluator()
    test_code = task.get("test_code", "")
    result = evaluator.evaluate(code, test_code)
    return 1.0 if result.passed else 0.0


def _visual_reward(completion: str, task: dict) -> float:
    """Reward for visual tasks: render HTML and get VLM quality score."""
    from src.utils.code_utils import extract_code

    html = extract_code(completion, "html")
    if not html.strip():
        return 0.0

    try:
        from src.rendering.browser import BrowserRenderer
        renderer = BrowserRenderer()
        screenshot_bytes = renderer.render_html(html)
    except Exception:
        return 0.0

    # VLM review of the rendered screenshot
    import base64
    from src.config import ModelConfig
    from src.utils.llm_client import get_client

    screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
    requirements = "\n".join(f"- {r}" for r in task.get("requirements", []))

    try:
        client = get_client()
        response = client.generate(
            config=ModelConfig.claude_sonnet(),
            system="You are a visual quality reviewer. Respond with ONLY valid JSON.",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Rate the visual quality of this rendered output.\n"
                            f"Requirements:\n{requirements}\n\n"
                            f'Respond with ONLY JSON: {{"quality_score": 0.0-1.0}}'
                        ),
                    },
                ],
            }],
        )
        import json as _json
        data = _json.loads(response.content.strip())
        return float(data.get("quality_score", 0.0))
    except Exception:
        return 0.0


def _video_reward(completion: str, task: dict) -> float:
    """Reward for video tasks: execute script, extract frames, VLM review."""
    from src.utils.code_utils import extract_code
    from src.feedback.type3c_video import VideoFeedback

    code = extract_code(completion, "python")
    if not code.strip():
        return 0.0

    try:
        video_fb = VideoFeedback()
        problem = {
            "source_video": task.get("source_video", ""),
            "output_path": "/tmp/output.mp4",
            "frame_check_times_s": task.get("frame_check_times_s", [0, 1, 2, 3]),
            "requirements": "\n".join(task.get("requirements", [])),
        }
        result = video_fb.get_feedback(code, problem)
        return float(result.structured_data.get("quality_score", 0.0))
    except Exception:
        return 0.0


def _factual_reward(completion: str, task: dict) -> float:
    """Reward for research tasks: claim decomposition + verification."""
    from src.feedback.type3d_factual import FactualVerificationFeedback

    if not completion.strip():
        return 0.0

    try:
        fact_fb = FactualVerificationFeedback()
        problem = {"requirements": task.get("requirements", [])}
        result = fact_fb.get_feedback(completion, problem)
        return float(result.structured_data.get("accuracy", 0.0))
    except Exception:
        return 0.0


def load_training_data(data_path: Path) -> list[dict]:
    """Load GRPO training data."""
    with open(data_path) as f:
        return json.load(f)


def prepare_dataset(examples: list[dict], tokenizer) -> list[dict]:
    """Prepare dataset for GRPO training.

    Each example has a prompt that includes task description + budget info.
    The model generates completions (interaction trajectories).
    The reward function scores each trajectory.
    """
    dataset = []
    for ex in examples:
        prompt = ex["prompt"]
        # Tokenize and check length
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > 2048:
            prompt = tokenizer.decode(tokens[:2048])

        dataset.append({
            "prompt": prompt,
            "task_id": ex["task_id"],
            "reward": ex.get("reward", 0.0),
        })
    return dataset


def train_sft(config: TrainingConfig, data_path: Path):
    """Stage 1: Supervised fine-tuning on successful trajectories.

    Teaches the model the basic interaction pattern before RL.
    """
    from trl import SFTConfig, SFTTrainer

    logger.info("Loading model %s for SFT...", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_for_training(config.model_name)

    with open(data_path) as f:
        sft_data = json.load(f)

    # Convert to the format SFTTrainer expects
    from datasets import Dataset

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = Dataset.from_list(sft_data)
    dataset = dataset.map(format_example)

    training_args = SFTConfig(
        output_dir=config.output_dir + "-sft",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=2e-4,
        max_length=config.max_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=config.seed,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config.output_dir + "-sft")
    logger.info("SFT training complete. Model saved to %s", config.output_dir + "-sft")


def train_grpo(config: TrainingConfig, data_path: Path, tasks_dir: Path | None = None):
    """Stage 2: GRPO training with grounded reward.

    The model generates multiple trajectories for each task.
    Each trajectory is scored by the grounded reward function.
    GRPO optimizes the policy using relative rewards within each group.

    Supports multi-task training: the reward function is selected per-example
    based on the task_type field in the training data.
    """
    from trl import GRPOConfig, GRPOTrainer

    logger.info("Loading model %s for GRPO...", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_for_training(config.model_name)

    with open(data_path) as f:
        grpo_data = json.load(f)

    from datasets import Dataset
    dataset = Dataset.from_list([
        {"prompt": ex["prompt"], "task_id": ex["task_id"], "task_type": ex["task_type"]}
        for ex in grpo_data
    ])

    # Build task info map for reward computation
    # Load actual task definitions if available (for test_code, requirements, etc.)
    task_info = {}
    if tasks_dir:
        tasks_dir = Path(tasks_dir)
        for task_file in tasks_dir.glob("*/*_tasks.json"):
            with open(task_file) as f:
                for t in json.load(f):
                    task_info[t["task_id"]] = t

    # Map task_type -> reward function
    reward_fns = {
        "code": _code_reward,
        "slide": _visual_reward,
        "webpage": _visual_reward,
        "animation": _visual_reward,
        "video": _video_reward,
        "research": _factual_reward,
    }

    # Build a task_id -> task_type mapping from training data
    task_type_map = {ex["task_id"]: ex["task_type"] for ex in grpo_data}

    def reward_function(completions, prompts, **kwargs):
        """Score each generated trajectory using grounded feedback.

        Selects the appropriate reward function based on the task type.
        """
        rewards = []
        for completion, prompt in zip(completions, prompts):
            # Extract task_id from prompt to look up task type and info
            task_id = _extract_task_id_from_prompt(prompt, task_type_map)
            task_type = task_type_map.get(task_id, "code")
            reward_fn = reward_fns.get(task_type, _code_reward)
            task = task_info.get(task_id, {})

            try:
                score = reward_fn(completion, task)
            except Exception:
                score = 0.0
            rewards.append(score)
        return rewards

    training_args = GRPOConfig(
        output_dir=config.output_dir + "-grpo",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        beta=config.beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=config.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_function,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model(config.output_dir + "-grpo")
    logger.info("GRPO training complete. Model saved to %s", config.output_dir + "-grpo")


def _extract_task_id_from_prompt(prompt: str, task_type_map: dict) -> str:
    """Extract task_id from a GRPO prompt by matching against known task IDs."""
    for task_id in task_type_map:
        if task_id in prompt:
            return task_id
    return ""


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = TrainingConfig()
    tasks_dir = Path("data/hard_benchmarks")
    stage = os.environ.get("STAGE", sys.argv[1] if len(sys.argv) > 1 else "all")

    if stage in ("sft", "all"):
        sft_data = Path("data/training/sft_data.json")
        if sft_data.exists():
            train_sft(config, sft_data)

    if stage in ("grpo", "all"):
        grpo_data = Path("data/training/grpo_data.json")
        if grpo_data.exists():
            # Point GRPO at the SFT LoRA adapter dir (base model is loaded inside _load_model_for_training
            # with the same Qwen3-8B weights; adapter loading is handled by peft.PeftModel.from_pretrained).
            config.model_name = config.output_dir + "-sft"
            train_grpo(config, grpo_data, tasks_dir=tasks_dir)
