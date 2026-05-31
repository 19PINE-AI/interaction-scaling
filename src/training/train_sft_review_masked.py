"""SFT on stitched review trajectories with draft-turn loss masking (v4).

Differences from train_sft_review.py:
  - Pre-tokenizes each example with `labels` that mask (set to -100):
      * The user turn (task description)
      * The first assistant turn (the draft — which is literal buggy_code)
      * The first tool response
      * Any `<|im_start|>assistant\\n` / `<|im_start|>user\\n` role scaffolding
        prior to the first REVISION assistant turn
  - Trains loss only on: revision turns + final confirm assistant turn.
  - Uses plain Trainer (not SFTTrainer) since we supply pre-tokenized data.

Motivation: v3 training left draft-turn tokens unmasked, so the student learned
to reproduce literal buggy_code patterns on its first tool call. Masking the
draft turn forces the student to only learn the successful-recovery pattern.

Usage:
    PYTHONPATH=. python -m src.training.train_sft_review_masked \\
        --data data/training/sft_review_v4.json \\
        --output models/qwen3-8b-autonomous-review-sft-v4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.training.train_grpo import _load_model_for_training

logger = logging.getLogger(__name__)


def build_prefix_messages(example: dict) -> list[dict]:
    """Return the sub-prefix of messages up through the first tool response.

    This is the region we will mask in labels. Everything after (revision
    assistant turns + tool responses + final confirm) is trained on.
    """
    msgs = example["messages"]
    # Standard stitched shape: [user, assistant(draft)+tool_calls, tool(draft_result), ...]
    # We mask through (and including) the first tool response.
    if len(msgs) < 3:
        return msgs  # degenerate; mask everything
    return msgs[:3]


def tokenize_with_mask(example: dict, tokenizer, max_length: int) -> dict:
    messages = example["messages"]
    tools = example.get("tools")

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        prefix_text = tokenizer.apply_chat_template(
            build_prefix_messages(example), tools=tools, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
    except TypeError:
        full_text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=False,
        )
        prefix_text = tokenizer.apply_chat_template(
            build_prefix_messages(example), tools=tools, tokenize=False,
            add_generation_prompt=False,
        )

    full_ids = tokenizer(full_text, add_special_tokens=False,
                         truncation=True, max_length=max_length)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                           truncation=True, max_length=max_length)["input_ids"]

    # Prefix should be a strict prefix of full (chat template is deterministic
    # given message order). Guard anyway.
    if prefix_ids != full_ids[: len(prefix_ids)]:
        # Fall back: find max k s.t. prefix_ids[:k] == full_ids[:k]
        k = 0
        while k < min(len(prefix_ids), len(full_ids)) and prefix_ids[k] == full_ids[k]:
            k += 1
        prefix_len = k
    else:
        prefix_len = len(prefix_ids)

    labels = list(full_ids)
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "_prefix_len": prefix_len,
        "_total_len": len(full_ids),
    }


class PadCollator:
    """Pad input_ids/labels/attention_mask to the longest in batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_n)
            batch["labels"].append(f["labels"] + [-100] * pad_n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/sft_review_v4.json")
    ap.add_argument("--output", default="models/qwen3-8b-autonomous-review-sft-v4")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--log", default="logs/sft_review_training_v4.log")
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

    model = _load_model_for_training(args.model_name)

    raw = json.loads(Path(args.data).read_text())
    logger.info("Loaded %d stitched examples from %s", len(raw), args.data)

    dataset = Dataset.from_list(raw)
    dataset = dataset.map(
        lambda ex: tokenize_with_mask(ex, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
    )

    lens = [(d["_total_len"], d["_prefix_len"]) for d in dataset]
    tot_mean = sum(l[0] for l in lens) / len(lens)
    pref_mean = sum(l[1] for l in lens) / len(lens)
    trained_frac = 1 - pref_mean / tot_mean
    logger.info("Token stats: total mean=%.0f, masked-prefix mean=%.0f, trained-frac=%.1f%%",
                tot_mean, pref_mean, trained_frac * 100)

    # Drop inspection fields before training
    dataset = dataset.remove_columns(["_prefix_len", "_total_len"])

    collator = PadCollator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=args.seed,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting masked SFT (draft-turn loss masked)...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("SFT complete. Adapter saved to %s", args.output)


if __name__ == "__main__":
    main()
