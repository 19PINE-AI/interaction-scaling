"""SFT on stitched autonomous-review trajectories (Phase 4).

Trains Qwen3-8B (QLoRA, 4-bit) on `data/training/sft_review_v1.json`, a list
of chat-template examples with `tools` + `messages` (assistant turns may
contain `tool_calls`; tool responses come back on the `tool` role). Qwen3's
chat template understands this natively.

Output adapter: `models/qwen3-8b-autonomous-review-sft/`.

Usage:
    PYTHONPATH=. python -m src.training.train_sft_review \
        --data data/training/sft_review_v1.json \
        --output models/qwen3-8b-autonomous-review-sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.training.train_grpo import _load_model_for_training

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/sft_review_v1.json")
    ap.add_argument("--output", default="models/qwen3-8b-autonomous-review-sft")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--log", default="logs/sft_review_training.log")
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

    def format_example(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tools=example.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tools=example.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    dataset = Dataset.from_list(raw)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Sanity: print length distribution
    lens = [len(tokenizer.encode(x["text"])) for x in dataset.select(range(min(20, len(dataset))))]
    logger.info("Token length (first 20): min=%d max=%d mean=%.0f",
                min(lens), max(lens), sum(lens) / len(lens))

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=args.seed,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT on stitched review trajectories...")
    trainer.train()
    trainer.save_model(args.output)
    logger.info("SFT complete. Adapter saved to %s", args.output)


if __name__ == "__main__":
    main()
