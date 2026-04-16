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
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""

    model_name: str = "Qwen/Qwen3-8B"
    output_dir: str = "models/qwen3-8b-interaction-scaling"
    learning_rate: float = 1e-6
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    num_generations: int = 4  # Group size for GRPO
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
    """Reward for visual tasks: render and VLM quality score."""
    # Placeholder — requires browser rendering + VLM review
    # In production, this would render the HTML and score it
    return 0.5


def _video_reward(completion: str, task: dict) -> float:
    """Reward for video tasks: execute, extract frames, VLM review."""
    # Placeholder — requires ffmpeg + frame extraction + VLM
    return 0.5


def _factual_reward(completion: str, task: dict) -> float:
    """Reward for research tasks: claim decomposition + verification."""
    # Placeholder — requires claim extraction + search verification
    return 0.5


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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

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
        learning_rate=config.learning_rate,
        max_seq_length=config.max_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config.output_dir + "-sft")
    logger.info("SFT training complete. Model saved to %s", config.output_dir + "-sft")


def train_grpo(config: TrainingConfig, data_path: Path, task_type: str = "code"):
    """Stage 2: GRPO training with grounded reward.

    The model generates multiple trajectories for each task.
    Each trajectory is scored by the grounded reward function.
    GRPO optimizes the policy using relative rewards within each group.
    """
    from trl import GRPOConfig, GRPOTrainer

    logger.info("Loading model %s for GRPO...", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    with open(data_path) as f:
        grpo_data = json.load(f)

    from datasets import Dataset
    dataset = Dataset.from_list([
        {"prompt": ex["prompt"], "task_id": ex["task_id"]}
        for ex in grpo_data
    ])

    # Build reward function
    task_map = {ex["task_id"]: ex for ex in grpo_data}
    reward_fn = build_reward_function(task_type)

    def reward_function(completions, prompts, **kwargs):
        """Score each generated trajectory using grounded feedback."""
        rewards = []
        for completion, prompt in zip(completions, prompts):
            # Extract task info from prompt
            # In production, this would look up the task and run execution
            try:
                score = reward_fn(completion, {})
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
        max_prompt_length=config.max_prompt_length,
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
        tokenizer=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model(config.output_dir + "-grpo")
    logger.info("GRPO training complete. Model saved to %s", config.output_dir + "-grpo")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig()

    # Stage 1: SFT
    sft_data = Path("data/training/sft_data.json")
    if sft_data.exists():
        train_sft(config, sft_data)

    # Stage 2: GRPO
    grpo_data = Path("data/training/grpo_data.json")
    if grpo_data.exists():
        # Update model path to use SFT checkpoint
        config.model_name = config.output_dir + "-sft"
        train_grpo(config, grpo_data, task_type="code")
