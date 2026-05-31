"""Run the Phase-5 student (Qwen3-VL-8B-Thinking ± LoRA SFT adapter) on held-out
tasks and dump trajectories for behavioral analysis vs the teacher.

The inference loop matches `collect_vl_traces.py`:
- combined gen+review system prompt
- model produces ```python or ```html block, harness executes/renders it
- harness returns text or images as a synthetic user turn
- repeat until <final>OK</final> or max_turns

Trajectories are saved in the same JSON schema as collect_vl_traces output so
they can be passed straight to `judge_vl_traces.py`.

Usage:
    PYTHONPATH=. python -m src.training.run_vl_student \\
        --adapter models/qwen3-vl-8b-vl-sft \\
        --tasks data/training/heldout_phase5.json \\
        --output results/phase5/student_traces.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from src.evaluation.code_eval import CodeEvaluator
from src.training.collect_vl_traces import (
    SYSTEM_PROMPTS, build_user_turn, extract_artifact, FINAL_RE,
    run_code, run_webpage, run_slide,
)

logger = logging.getLogger(__name__)


def load_model(model_id: str, adapter: str | None,
               load_in_4bit: bool = False, load_in_8bit: bool = False):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    logger.info("Loading processor + base model %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if cfg.model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model_cls = Qwen3VLMoeForConditionalGeneration
    else:
        model_cls = Qwen3VLForConditionalGeneration
    kwargs: dict = {"device_map": "auto"}
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = model_cls.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)
        # merge_and_unload only works with non-quantized base
        if not (load_in_4bit or load_in_8bit):
            model = model.merge_and_unload()
    model.eval()
    return processor, model


def messages_to_inputs(processor, messages: list[dict]):
    """Build model inputs from chat messages with image content parts."""
    images: list[Image.Image] = []
    msgs_flat = []
    for m in messages:
        if isinstance(m["content"], str):
            msgs_flat.append({"role": m["role"], "content": m["content"]})
            continue
        parts = []
        for p in m["content"]:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                # Already a PIL.Image (we pass them as PIL not paths here)
                images.append(p["image"])
                parts.append({"type": "image"})
        msgs_flat.append({"role": m["role"], "content": parts})
    text = processor.apply_chat_template(
        msgs_flat, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=images if images else None,
        return_tensors="pt", padding=False,
    )
    return inputs


CLOSE_THINK_ID = 151668  # Qwen3-VL </think>


class ForceCloseThinking:
    """LogitsProcessor that boosts </think> after `force_after` new tokens
    if the model hasn't closed thinking yet. Runs once per generation."""
    def __init__(self, prompt_len: int, force_after: int = 1500):
        self.prompt_len = prompt_len
        self.force_after = force_after
        self.fired = False

    def __call__(self, input_ids, scores):
        if self.fired:
            return scores
        new_tokens = input_ids.shape[1] - self.prompt_len
        if new_tokens < self.force_after:
            return scores
        # Have we already produced </think>?
        produced = input_ids[0, self.prompt_len:].tolist()
        if CLOSE_THINK_ID in produced:
            self.fired = True
            return scores
        # Force </think> as the very next token
        scores[:, :] = -1e9
        scores[:, CLOSE_THINK_ID] = 1e9
        self.fired = True
        return scores


@torch.no_grad()
def generate(processor, model, messages: list[dict], max_new_tokens: int = 12000,
             temperature: float = 0.0, repetition_penalty: float = 1.0,
             force_close_think_after: int = 0) -> str:
    inputs = messages_to_inputs(processor, messages).to(model.device)
    do_sample = temperature > 0
    logits_processors = []
    if force_close_think_after > 0:
        logits_processors.append(ForceCloseThinking(
            prompt_len=inputs["input_ids"].shape[1],
            force_after=force_close_think_after,
        ))
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        repetition_penalty=repetition_penalty,
        pad_token_id=processor.tokenizer.pad_token_id,
        logits_processor=logits_processors if logits_processors else None,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
    # Strip trailing im_end and any leftover special tokens
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def run_task(processor, model, task: dict, category: str, max_turns: int,
             evaluator: CodeEvaluator, temperature: float = 0.0,
             repetition_penalty: float = 1.0,
             force_close_think_after: int = 0) -> dict:
    system = SYSTEM_PROMPTS[category]
    user0 = build_user_turn(task, category)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user0}]},
    ]
    turns = []
    last_feedback = None
    for turn_idx in range(max_turns):
        t0 = time.time()
        try:
            content = generate(processor, model, messages, temperature=temperature,
                               repetition_penalty=repetition_penalty,
                               force_close_think_after=force_close_think_after)
        except Exception as e:
            logger.warning("generate failed on %s turn %d: %s",
                           task["task_id"], turn_idx, e)
            return {"task_id": task["task_id"], "category": category,
                    "status": "error", "error": str(e), "turns": turns}
        gen_s = round(time.time() - t0, 2)
        turns.append({"assistant": content, "gen_s": gen_s, "feedback": last_feedback})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

        if FINAL_RE.search(content):
            art = None
            for past in reversed(turns):
                a = extract_artifact(past.get("assistant") or "", category)
                if a: art = a; break
            if art and category == "code":
                fb = run_code(art, task["test_code"], evaluator)
                final_passed = fb["passed"]
            else:
                final_passed = None
            return {"task_id": task["task_id"], "category": category,
                    "status": "final", "final_artifact": art,
                    "final_passed": final_passed, "turns": turns}

        art = extract_artifact(content, category)
        if art is None:
            # Walk-back: if this turn collapsed but earlier turns produced
            # valid artifacts, treat the most recent one as the final.
            fallback = None
            for past in reversed(turns[:-1]):
                a = extract_artifact(past.get("assistant") or "", category)
                if a:
                    fallback = a
                    break
            if fallback is not None:
                if category == "code":
                    fb = run_code(fallback, task["test_code"], evaluator)
                    final_passed = fb["passed"]
                else:
                    final_passed = None
                return {"task_id": task["task_id"], "category": category,
                        "status": "no_artifact_walkback", "final_artifact": fallback,
                        "final_passed": final_passed, "turns": turns}
            return {"task_id": task["task_id"], "category": category,
                    "status": "no_artifact", "turns": turns}

        if category == "code":
            fb = run_code(art, task["test_code"], evaluator)
        elif category == "webpages":
            fb = run_webpage(art, task.get("viewport_sizes", [1920, 375]))
        else:
            fb = run_slide(art)

        if fb["type"] == "text":
            new_user = {"role": "user", "content": [{"type": "text", "text": fb["payload"]}]}
        else:
            parts = [{"type": "text", "text": "rendered screenshot(s):"}]
            for w, h, png in fb["payload"]:
                # Resize to match SFT condition
                from io import BytesIO
                img = Image.open(BytesIO(png)).convert("RGB")
                if max(img.size) > 1280:
                    s = 1280 / max(img.size)
                    img = img.resize((int(img.size[0]*s), int(img.size[1]*s)), Image.LANCZOS)
                parts.append({"type": "text", "text": f"viewport {w}x{h}:"})
                parts.append({"type": "image", "image": img})
            if not fb["payload"]:
                parts = [{"type": "text", "text": f"render failed: {fb.get('error','')}"}]
            new_user = {"role": "user", "content": parts}
        messages.append(new_user)

        last_feedback = {
            "type": fb["type"],
            "passed": fb.get("passed"),
            "text_payload": fb["payload"] if fb["type"] == "text" else None,
            "image_count": len(fb["payload"]) if fb["type"] == "image" else 0,
        }

    # Walk-back: even at max_turns, use the last valid artifact for downstream eval.
    art = None
    for past in reversed(turns):
        a = extract_artifact(past.get("assistant") or "", category)
        if a:
            art = a
            break
    final_passed = None
    if art and category == "code":
        try:
            fb = run_code(art, task["test_code"], evaluator)
            final_passed = fb["passed"]
        except Exception:
            final_passed = False
    return {"task_id": task["task_id"], "category": category,
            "status": "max_turns", "final_artifact": art,
            "final_passed": final_passed, "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; omit for base model eval")
    ap.add_argument("--tasks", required=True,
                    help="JSON file: list of {task_id, category, ...task fields}")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--force-close-think-after", type=int, default=0,
                    help="Force </think> token after N generated tokens (0=disable)")
    ap.add_argument("--seed", type=int, default=42, help="generation seed")
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", default="logs/run_vl_student.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    torch.manual_seed(args.seed)
    processor, model = load_model(args.model, args.adapter)
    evaluator = CodeEvaluator()
    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks", len(tasks))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done = {(r["category"], r["task_id"]) for r in results}
        tasks = [t for t in tasks if (t["category"], t["task_id"]) not in done]
        logger.info("Resuming. %d tasks remaining.", len(tasks))

    for i, t in enumerate(tasks):
        cat = t["category"]
        t0 = time.time()
        trace = run_task(processor, model, t, cat, args.max_turns, evaluator,
                         temperature=args.temperature,
                         repetition_penalty=args.repetition_penalty,
                         force_close_think_after=args.force_close_think_after)
        trace["elapsed_s"] = round(time.time() - t0, 2)
        results.append(trace)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info(
            "[%d/%d] %s/%s status=%s turns=%d elapsed=%.1fs",
            len(results), len(results) + len(tasks) - i - 1,
            cat, t["task_id"], trace["status"],
            len(trace.get("turns", [])), trace["elapsed_s"],
        )

    logger.info("Done. %d traces -> %s", len(results), out_path)


if __name__ == "__main__":
    main()
