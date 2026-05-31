"""GRPO on Phase 4 autonomous review — reward first-shot tool-call code quality.

Starts from the Phase 4 SFT v2 adapter (`models/qwen3-8b-autonomous-review-sft-v2/`).
Each rollout is a single assistant turn containing a <tool_call> block; we parse
the `code` argument, run it against the task's ground-truth test_code, and score:

  +1.0  tool_pass (code passes tests)
  +0.1  valid tool_call emitted but code failed
   0.0  no parseable tool_call

This trains the model to produce correct code *on the first tool call*, which
was the main remaining regression in SFT v2 (tool_pass@1 only 13.3% on held-out).
We don't train the revise step here — v2 already iterates correctly when it
sees a failure signal. The bottleneck is first-shot quality.

Usage:
    PYTHONPATH=. python -m src.training.train_grpo_review \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --adapter models/qwen3-8b-autonomous-review-sft-v2 \\
        --output models/qwen3-8b-autonomous-review-grpo
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.evaluation.code_eval import CodeEvaluator
from src.training.train_grpo import _load_model_for_training

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an AI coding assistant. When fixing a bug, you should verify your "
    "solution by calling the `execute_code` tool with your candidate code and "
    "the test harness provided in the task description. Review the tool's "
    "output and revise if the tests fail. When you are confident the solution "
    "is correct, respond with the final code in a ```python block and do not "
    "call any more tools."
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code combined with a test harness. "
            "Returns a JSON object {passed, stdout, stderr, error}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "test_code": {"type": "string"},
            },
            "required": ["code", "test_code"],
        },
    },
}


def extract_tool_call_code(completion: str) -> str | None:
    """Parse the first <tool_call> in a completion and return its `code` arg."""
    m = TOOL_CALL_RE.search(completion)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    args = obj.get("arguments") or obj.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args.get("code")


def build_reward_fn():
    evaluator = CodeEvaluator()

    def reward_fn(completions, test_code, **kwargs):
        rewards = []
        for comp, tc in zip(completions, test_code):
            code = extract_tool_call_code(comp)
            if code is None:
                rewards.append(0.0)
                continue
            try:
                res = evaluator.evaluate(code, tc, timeout=12)
            except Exception:
                rewards.append(0.05)
                continue
            rewards.append(1.0 if res.passed else 0.1)
        return rewards

    return reward_fn


def build_dataset(tokenizer, tasks_path: str) -> Dataset:
    tasks = json.loads(Path(tasks_path).read_text())
    rows = []
    for t in tasks:
        user_msg = (
            f"Fix the bug in this code.\n\n{t['description']}\n\n"
            f"You can verify your fix against these tests via the execute_code tool:\n"
            f"```python\n{t['test_code']}\n```"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tools=[TOOL_SCHEMA], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tools=[TOOL_SCHEMA], tokenize=False,
                add_generation_prompt=True,
            )
        rows.append({
            "prompt": prompt,
            "test_code": t["test_code"],
            "task_id": t["task_id"],
        })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/training/code_tasks_v2_scrubbed.json")
    ap.add_argument("--adapter", default="models/qwen3-8b-autonomous-review-sft-v2")
    ap.add_argument("--output", default="models/qwen3-8b-autonomous-review-grpo")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--log", default="logs/grpo_review_training.log")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model (warm-start from %s)...", args.adapter)
    model = _load_model_for_training(args.adapter)

    dataset = build_dataset(tokenizer, args.tasks)
    logger.info("Loaded %d training tasks", len(dataset))

    reward_fn = build_reward_fn()

    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO on Phase 4 autonomous review...")
    trainer.train()
    trainer.save_model(args.output)
    logger.info("GRPO complete. Adapter saved to %s", args.output)


if __name__ == "__main__":
    main()
