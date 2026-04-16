"""Evaluate base vs fine-tuned Qwen3 8B on interaction scaling tasks.

Compares three conditions:
1. Qwen3 8B base (single-shot, no interaction)
2. Qwen3 8B base + external proposer-reviewer scaffold
3. Qwen3 8B fine-tuned (internalized interaction scaling)

The fine-tuned model should autonomously decide when to seek feedback,
how to use it, and when to stop — without external orchestration.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.code_eval import CodeEvaluator
from src.utils.code_utils import extract_code

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    base_model: str = "Qwen/Qwen3-8B"
    finetuned_model: str = "models/qwen3-8b-interaction-scaling-grpo"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    device: str = "auto"


def evaluate_single_shot(
    model,
    tokenizer,
    tasks: list[dict],
    model_name: str = "base",
) -> list[dict]:
    """Evaluate model in single-shot mode (no interaction)."""
    evaluator = CodeEvaluator()
    results = []

    for task in tasks:
        prompt = (
            f"Fix the bug in the following code. Output only the corrected "
            f"complete code in a ```python block.\n\n{task['description']}"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_code(response, "python")

        eval_result = evaluator.evaluate(code, task["test_code"])

        results.append({
            "task_id": task["task_id"],
            "model": model_name,
            "mode": "single_shot",
            "passed": eval_result.passed,
            "code": code[:500],
        })

        logger.info(
            "%s single-shot %s: %s",
            model_name, task["task_id"],
            "PASS" if eval_result.passed else "FAIL",
        )

    return results


def evaluate_interaction_aware(
    model,
    tokenizer,
    tasks: list[dict],
    model_name: str = "finetuned",
    max_steps: int = 5,
) -> list[dict]:
    """Evaluate fine-tuned model with internalized interaction scaling.

    The model autonomously generates [EXECUTE], [REVIEW], [SUBMIT] tokens
    to control its interaction with the environment.
    """
    evaluator = CodeEvaluator()
    results = []

    for task in tasks:
        prompt = (
            f"You are an AI agent solving a task with interaction scaling.\n"
            f"Budget: {max_steps} steps.\n"
            f"Task type: code\n\n"
            f"Task:\n{task['description'][:2000]}\n\n"
            f"Solve this task. Use [EXECUTE] to test your solution, "
            f"[REVIEW] to analyze feedback, and [SUBMIT] when done."
        )

        current_code = ""
        passed = False
        iterations = 0

        for step in range(max_steps):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.0,
                    do_sample=False,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            iterations = step + 1

            # Check if model wants to execute
            if "[EXECUTE]" in response:
                code = extract_code(response, "python")
                if code.strip():
                    current_code = code
                eval_result = evaluator.evaluate(current_code, task["test_code"])

                if eval_result.passed:
                    passed = True
                    break

                # Feed execution result back as context
                feedback = f"FAILED: {eval_result.error_message}"
                if eval_result.stderr:
                    feedback += f"\n{eval_result.stderr[:1000]}"
                prompt += f"\n\nAssistant: {response}\n\n[FEEDBACK]: {feedback}\n\nContinue:"

            elif "[SUBMIT]" in response:
                # Model decided to submit
                code = extract_code(response, "python")
                if code.strip():
                    current_code = code
                eval_result = evaluator.evaluate(current_code, task["test_code"])
                passed = eval_result.passed
                break
            else:
                # Model generated code without explicit action token
                code = extract_code(response, "python")
                if code.strip():
                    current_code = code
                break

        results.append({
            "task_id": task["task_id"],
            "model": model_name,
            "mode": "interaction_aware",
            "passed": passed,
            "iterations": iterations,
            "code": current_code[:500],
        })

        logger.info(
            "%s interaction %s: %s (iters=%d)",
            model_name, task["task_id"],
            "PASS" if passed else "FAIL",
            iterations,
        )

    return results


def run_evaluation(config: EvalConfig, tasks_path: Path, output_dir: Path):
    """Run full evaluation comparing base vs fine-tuned."""
    with open(tasks_path) as f:
        tasks = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # 1. Base model single-shot
    logger.info("Loading base model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
    )

    base_ss = evaluate_single_shot(model, tokenizer, tasks, "qwen3_8b_base")
    all_results.extend(base_ss)

    # 2. Base model with interaction (external scaffold)
    base_ia = evaluate_interaction_aware(
        model, tokenizer, tasks, "qwen3_8b_base_scaffold", max_steps=5
    )
    all_results.extend(base_ia)

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Fine-tuned model with internalized interaction
    ft_path = Path(config.finetuned_model)
    if ft_path.exists():
        logger.info("Loading fine-tuned model: %s", config.finetuned_model)
        model = AutoModelForCausalLM.from_pretrained(
            config.finetuned_model,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
        )
        ft_ia = evaluate_interaction_aware(
            model, tokenizer, tasks, "qwen3_8b_finetuned", max_steps=5
        )
        all_results.extend(ft_ia)
    else:
        logger.warning("Fine-tuned model not found at %s, skipping", ft_path)

    # Save results
    with open(output_dir / "training_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(" TRAINING EVALUATION RESULTS")
    print("=" * 70)

    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[f"{r['model']}_{r['mode']}"].append(r)

    for key, items in sorted(by_model.items()):
        n = len(items)
        passed = sum(1 for r in items if r["passed"])
        print(f"{key:<45} {passed}/{n} ({passed/n:.0%})")

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = EvalConfig()
    run_evaluation(
        config,
        tasks_path=Path("data/hard_benchmarks/code/code_tasks.json"),
        output_dir=Path("results/training"),
    )
