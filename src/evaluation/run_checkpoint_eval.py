"""Evaluate base/SFT/GRPO Qwen3-8B checkpoints on the code benchmark.

Single-turn generation: prompt the model with a coding task, extract the
first ```python code block, run it against the task's test_code, record
pass/fail. Reports pass@1.

Note: the 15 code tasks all appear in the SFT/GRPO training data — this
is an in-distribution comparison, not held-out generalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI agent that solves tasks through interaction scaling. "
    "You have a budget of 5 steps to solve the task. At each step, you can:\n"
    "- [GENERATE]: Write code/content to solve the task\n"
    "- [EXECUTE]: Run your solution and observe the results\n"
    "- [REVIEW]: Analyze feedback and plan improvements\n"
    "- [SUBMIT]: Submit your final solution\n\n"
    "Be budget-aware: use your steps efficiently. If execution succeeds on the first try, "
    "submit immediately. If execution fails, review the feedback, revise, and re-test.\n\n"
    "IMPORTANT: Always output complete, working code/content — not descriptions of what you "
    "would do. Every [GENERATE] response must contain the full artifact.\n\n"
    "Task type: code\n"
    "Output ONLY the function definition with any needed imports.\n"
    "Handle ALL edge cases carefully.\n"
    "Wrap your code in a ```python code block."
)

USER_PROMPT_TEMPLATE = "[BUDGET: 5 steps remaining]\n[TASK]: Fix the bug in this code.\n\n{description}"

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def load_model(model_id: str, adapter_path: Optional[str]):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        attn_implementation="kernels-community/flash-attn",
        device_map={"": 0},
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def generate(model, tok, description: str, max_new_tokens: int = 1536) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def extract_code(response: str) -> str:
    # Find the LAST python code block (model may emit multiple in multi-turn-style output)
    matches = CODE_BLOCK_RE.findall(response)
    if matches:
        return matches[-1].strip()
    return ""


def evaluate_checkpoint(label: str, base: str, adapter: Optional[str], tasks: list) -> dict:
    logger.info("Loading %s: base=%s adapter=%s", label, base, adapter)
    model, tok = load_model(base, adapter)
    evaluator = CodeEvaluator()
    results = []
    passed = 0
    for task in tasks:
        t0 = time.perf_counter()
        try:
            response = generate(model, tok, task["description"])
        except Exception as e:
            response = ""
            err = f"generation_error: {e}"
        else:
            err = None
        code = extract_code(response) if response else ""
        if not code:
            results.append({
                "task_id": task["task_id"],
                "passed": False,
                "error": err or "no_code_block",
                "response_len": len(response),
                "gen_s": round(time.perf_counter() - t0, 2),
            })
            continue
        exec_result = evaluator.evaluate(code, task["test_code"], timeout=15)
        if exec_result.passed:
            passed += 1
        results.append({
            "task_id": task["task_id"],
            "passed": exec_result.passed,
            "error": exec_result.error_message,
            "response_len": len(response),
            "code_len": len(code),
            "gen_s": round(time.perf_counter() - t0, 2),
        })
        logger.info("%s %s: %s (gen %.1fs)", label, task["task_id"],
                    "PASS" if exec_result.passed else "FAIL", results[-1]["gen_s"])
    del model, tok
    torch.cuda.empty_cache()
    return {
        "label": label,
        "passed": passed,
        "total": len(tasks),
        "pass_rate": passed / len(tasks),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hard_benchmarks/code/code_tasks.json")
    parser.add_argument("--output", default="results/hard_benchmarks/checkpoint_eval.json")
    parser.add_argument("--checkpoints", nargs="+",
                        default=["base", "sft", "grpo"],
                        help="Which checkpoints to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks", len(tasks))

    base_id = "Qwen/Qwen3-8B"
    adapters = {
        "base": (base_id, None),
        "sft": (base_id, "models/qwen3-8b-interaction-scaling-sft"),
        "grpo": (base_id, "models/qwen3-8b-interaction-scaling-grpo"),
    }

    all_results = {}
    for label in args.checkpoints:
        base, adapter = adapters[label]
        try:
            all_results[label] = evaluate_checkpoint(label, base, adapter, tasks)
        except Exception as e:
            logger.exception("Eval failed for %s", label)
            all_results[label] = {"label": label, "error": str(e)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary ===")
    for label, r in all_results.items():
        if "pass_rate" in r:
            print(f"{label:6s}: {r['passed']}/{r['total']} = {r['pass_rate']:.1%}")
        else:
            print(f"{label:6s}: ERROR {r.get('error', '?')}")


if __name__ == "__main__":
    main()
