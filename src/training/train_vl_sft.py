"""Multimodal SFT for Qwen3-VL-8B-Thinking on filtered teacher traces.

Input: JSONL produced by `prepare_vl_sft.py` (one example per line, with
`messages` in Qwen3-VL chat format and image references as file paths).

Loss: standard causal LM loss, but masked to assistant tokens only. System /
user / tool / image tokens are -100 in labels so the student only learns to
generate assistant content (critique paragraph + revised artifact).

Target modules: attention + MLP linears of the LANGUAGE MODEL only. Vision
tower remains frozen — we are teaching the model what to SAY given a visual
input, not how to see. This matches the Phase 2 recipe (LoRA on text
modules) and keeps the trainable-parameter count manageable on a single
80GB GPU alongside the frozen visual backbone.

Usage:
    PYTHONPATH=. python -m src.training.train_vl_sft \\
        --data data/training/vl_sft.jsonl \\
        --model Qwen/Qwen3-VL-8B-Thinking \\
        --output models/qwen3-vl-8b-vl-sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def _get_model_class(model_id: str):
    """Pick Qwen3VL or Qwen3VLMoe class based on config.model_type."""
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if cfg.model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
    return Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

ASSISTANT_START = "<|im_start|>assistant"
ASSISTANT_END = "<|im_end|>"


def load_messages_with_images(example: dict) -> tuple[list[dict], list[Image.Image]]:
    """Return (messages-for-chat-template, ordered PIL images).

    In Qwen3-VL chat format, each user message may contain
    {"type": "image", "image": "<path>"} parts. The processor's
    apply_chat_template needs the message structure preserved AND the
    actual images loaded for pixel_values computation.

    Preserves `tool_calls` on assistant messages — without these the chat
    template renders assistant turns as plain text, dropping the actual
    `<tool_call>{json}</tool_call>` block. Training without that block
    teaches the model to emit `<|im_end|>` immediately after introductory
    prose, which makes it useless for tool-using inference.
    """
    images: list[Image.Image] = []
    messages = []
    for m in example["messages"]:
        new_m: dict = {"role": m["role"]}
        if "tool_calls" in m and m["tool_calls"]:
            new_m["tool_calls"] = m["tool_calls"]
        if isinstance(m["content"], str) or m["content"] is None:
            new_m["content"] = m["content"] or ""
            messages.append(new_m)
            continue
        parts = []
        for p in m["content"]:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                img = Image.open(p["image"]).convert("RGB")
                images.append(img)
                parts.append({"type": "image"})
        new_m["content"] = parts
        messages.append(new_m)
    return messages, images


class VLSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, processor):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.rows[i]
        messages, images = load_messages_with_images(ex)
        # Agentic SFT examples carry a `tools` field that must be injected
        # into the chat template under the `# Tools` section.
        tools = ex.get("tools")
        text = self.processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=False,
        )
        input_ids = inputs["input_ids"][0]
        # Build labels: -100 everywhere except inside assistant turns.
        labels = torch.full_like(input_ids, -100)
        tokenizer = self.processor.tokenizer
        start_ids = tokenizer(ASSISTANT_START, add_special_tokens=False)["input_ids"]
        end_ids = tokenizer(ASSISTANT_END, add_special_tokens=False)["input_ids"]
        ids = input_ids.tolist()
        n = len(ids)
        # Find contiguous regions [start_ids ... end_ids] and unmask the content after start_ids through end_ids
        i2 = 0
        while i2 < n - len(start_ids):
            if ids[i2:i2 + len(start_ids)] == start_ids:
                content_begin = i2 + len(start_ids)
                # find next end_ids
                j = content_begin
                while j < n - len(end_ids):
                    if ids[j:j + len(end_ids)] == end_ids:
                        labels[content_begin:j + len(end_ids)] = input_ids[
                            content_begin:j + len(end_ids)
                        ]
                        i2 = j + len(end_ids)
                        break
                    j += 1
                else:
                    break
            else:
                i2 += 1

        out = {"input_ids": input_ids, "labels": labels,
               "attention_mask": inputs["attention_mask"][0]}
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            out["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            out["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]
        return out


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    mm_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    has_mm = any("mm_token_type_ids" in ex for ex in batch)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].size(0)
        input_ids[i, :L] = ex["input_ids"]
        labels[i, :L] = ex["labels"]
        mask[i, :L] = ex["attention_mask"]
        if "mm_token_type_ids" in ex:
            mm_ids[i, :L] = ex["mm_token_type_ids"]
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": mask}
    if has_mm:
        out["mm_token_type_ids"] = mm_ids
    if all("pixel_values" in ex for ex in batch):
        out["pixel_values"] = torch.cat([ex["pixel_values"] for ex in batch], dim=0)
    if all("image_grid_thw" in ex for ex in batch):
        out["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in batch], dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--output", default="models/qwen3-vl-8b-vl-sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--log", default="logs/vl_sft_train.log")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA 4-bit base for fitting larger models on a single GPU")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="LoRA over 8-bit base — more precision than 4-bit, ~2× memory")
    ap.add_argument("--lora-attn-only", action="store_true",
                    help="LoRA on attention modules only (q/k/v/o), skip MLP — halves trainable params")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    logger.info("Loading processor from %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model_cls = _get_model_class(args.model)
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        if args.load_in_4bit:
            logger.info("Loading model in 4-bit (QLoRA, NF4 + double-quant)")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            logger.info("Loading model in 8-bit")
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = model_cls.from_pretrained(
            args.model,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        # Minimal kbit prep — skip peft's prepare_model_for_kbit_training because
        # its fp32 promotion of all non-quantized params doubles memory for the
        # many layernorms/embeddings in a 30B MoE. We just need:
        #   - gradient checkpointing enabled
        #   - input embeddings require_grad (for grad ckpt through frozen base)
        # LoRA init handles trainable param dtypes correctly.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        logger.info("Loading model (bf16)")
        model = model_cls.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    # Freeze vision tower
    for n, p in model.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad_(False)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not args.lora_attn_only:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    peft_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    dataset = VLSFTDataset(args.data, processor)
    logger.info("Loaded %d SFT examples", len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=2,
        save_strategy="epoch",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )

    logger.info("Starting multimodal SFT on Qwen3-VL-8B-Thinking")
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    logger.info("SFT complete -> %s", args.output)


if __name__ == "__main__":
    main()
