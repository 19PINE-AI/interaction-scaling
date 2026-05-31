"""GRPO v2 from SFT v4 with retry-aware reward.

Warm-starts from `models/qwen3-8b-autonomous-review-sft-v4/`. Each rollout is a
single completion that may contain multiple `<tool_call>` blocks — we parse all
of them, execute the last one against the task's test_code, and reward so that
multi-call trajectories that ultimately succeed outrank single-call successes:

  +1.5  >=2 tool_calls emitted AND last tool_call's code passes tests
  +1.0  exactly 1 tool_call AND it passes tests
  +0.1  any parseable tool_call, none pass
   0.0  no parseable tool_call

Rationale: SFT v4 still collapses to calls=1 on 84% of held-out PASS samples.
Prior GRPO v1 reward (+1.0 first-call pass) sharpened that collapse. This
reward gives a 0.5-reward margin for trajectories that retry, which should
push the policy toward multi-call behavior when it's uncertain. Note: this is
a single-completion approximation of multi-turn RL — the model has no real
tool feedback between its own emitted calls; the retry-emit behavior is
learned from SFT and merely preserved/reinforced here.

Usage:
    PYTHONPATH=. python -m src.training.train_grpo_review_v2 \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --adapter models/qwen3-8b-autonomous-review-sft-v4 \\
        --output models/qwen3-8b-autonomous-review-grpo-v2
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
from src.training.train_grpo_review import (
    SYSTEM_PROMPT,
    TOOL_SCHEMA,
    TOOL_CALL_RE,
)

logger = logging.getLogger(__name__)


def extract_all_tool_call_codes(completion: str) -> list[str]:
    """Parse every <tool_call> block and return the list of `code` args."""
    codes: list[str] = []
    for m in TOOL_CALL_RE.finditer(completion):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments") or obj.get("function", {}).get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        code = args.get("code")
        if code:
            codes.append(code)
    return codes


def build_reward_fn():
    evaluator = CodeEvaluator()

    def reward_fn(completions, test_code, **kwargs):
        rewards = []
        for comp, tc in zip(completions, test_code):
            codes = extract_all_tool_call_codes(comp)
            if not codes:
                rewards.append(0.0)
                continue
            last = codes[-1]
            try:
                res = evaluator.evaluate(last, tc, timeout=12)
            except Exception:
                rewards.append(0.05)
                continue
            if res.passed:
                rewards.append(1.5 if len(codes) >= 2 else 1.0)
            else:
                rewards.append(0.1)
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
    ap.add_argument("--adapter", default="models/qwen3-8b-autonomous-review-sft-v4")
    ap.add_argument("--output", default="models/qwen3-8b-autonomous-review-grpo-v2")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--log", default="logs/grpo_review_v2_training.log")
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

    logger.info("Starting GRPO v2 (retry-aware reward) on Phase 4 autonomous review...")
    trainer.train()
    trainer.save_model(args.output)
    logger.info("GRPO v2 complete. Adapter saved to %s", args.output)


if __name__ == "__main__":
    main()
