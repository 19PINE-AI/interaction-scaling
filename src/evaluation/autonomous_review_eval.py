"""No-scaffold autonomous-review evaluation.

The student is given ONE user turn ("Fix the bug in this code: {description}")
plus access to a single `execute_code` tool. It may emit as many tool_call
blocks as it wants across successive assistant turns. We terminate when:
  - the model produces an assistant turn with NO tool_call (natural stop), OR
  - the total tool-call budget is exhausted, OR
  - the total new-token budget is exhausted.

Metrics per (model, task):
  - autonomous_tool_called: any tool_call emitted
  - tool_pass: any execute_code tool_call returned passed=True
  - final_pass: the last ```python code block the model emitted passes tests post-hoc

A task "passes" if tool_pass OR final_pass. This lets us credit both (a) models
that sanity-check and accept a passing draft and (b) models that revise after
failed execution.

Usage:
    python -m src.evaluation.autonomous_review_eval \
        --tasks data/hard_benchmarks/code/code_tasks_heldout.json \
        --output results/autonomous_review/heldout.json \
        --checkpoints base sft_review \
        --max-tool-calls 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

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

SYSTEM_PROMPT = (
    "You are an AI coding assistant. When fixing a bug, you should verify your "
    "solution by calling the `execute_code` tool with your candidate code and "
    "the test harness provided in the task description. Review the tool's "
    "output and revise if the tests fail. When you are confident the solution "
    "is correct, respond with the final code in a ```python block and do not "
    "call any more tools."
)


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


def _extract_test_code(description: str) -> Optional[str]:
    """Pull the test_code out of the task's description if present; else None."""
    # descriptions don't carry test_code — test_code is passed separately
    return None


def generate(model, tok, messages, tools, max_new_tokens: int,
             temperature: float, top_p: float) -> str:
    try:
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
    inputs = tok(text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def parse_tool_calls(response: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(response):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_calls(response: str) -> str:
    return TOOL_CALL_RE.sub("", response).strip()


def extract_last_code(response: str) -> str:
    matches = CODE_BLOCK_RE.findall(response)
    return matches[-1].strip() if matches else ""


def run_task(model, tok, evaluator: CodeEvaluator, task: dict,
             max_tool_calls: int, max_new_tokens: int,
             temperature: float, top_p: float) -> dict:
    user_msg = (
        f"Fix the bug in this code.\n\n{task['description']}\n\n"
        f"You can verify your fix against these tests via the execute_code tool:\n"
        f"```python\n{task['test_code']}\n```"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    tools = [TOOL_SCHEMA]

    tool_calls_made = 0
    any_tool_pass = False
    last_code = ""
    turn_log = []

    for _ in range(max_tool_calls + 1):
        t0 = time.perf_counter()
        response = generate(model, tok, messages, tools,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p)
        gen_s = round(time.perf_counter() - t0, 2)
        calls = parse_tool_calls(response)
        visible_text = strip_tool_calls(response)
        code_in_text = extract_last_code(visible_text)
        if code_in_text:
            last_code = code_in_text

        turn_log.append({
            "response_len": len(response),
            "gen_s": gen_s,
            "num_tool_calls": len(calls),
            "code_in_text_len": len(code_in_text),
        })

        if not calls:
            break  # natural stop

        # Execute the first execute_code call in this turn
        executed_any = False
        tool_results = []
        for call in calls:
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name != "execute_code":
                tool_results.append({"error": f"unknown tool: {name}"})
                continue
            tool_calls_made += 1
            code = args.get("code", "")
            test_code = args.get("test_code", task["test_code"])
            exec_result = evaluator.evaluate(code, test_code, timeout=15)
            result = {
                "passed": exec_result.passed,
                "error": exec_result.error_message,
                "stdout": (exec_result.stdout or "")[-600:],
                "stderr": (exec_result.stderr or "")[-600:],
            }
            if exec_result.passed:
                any_tool_pass = True
                last_code = code  # a passing tool call is a valid final answer
            tool_results.append(result)
            executed_any = True

        # Reconstruct the assistant turn with tool_calls structured.
        tool_call_objs = []
        for call in calls:
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args)
            tool_call_objs.append({
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
        messages.append({
            "role": "assistant",
            "content": visible_text,
            "tool_calls": tool_call_objs,
        })
        # Feed one combined tool response back
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_results[0] if tool_results else {}),
        })

        if not executed_any:
            break
        if tool_calls_made >= max_tool_calls:
            # Allow one more assistant turn (no tools) to let the model finalize
            calls_probe = generate(model, tok, messages, tools,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p)
            final_code = extract_last_code(strip_tool_calls(calls_probe))
            if final_code:
                last_code = final_code
            turn_log.append({
                "response_len": len(calls_probe),
                "note": "final after tool budget",
                "num_tool_calls": 0,
                "code_in_text_len": len(final_code),
            })
            break

    # Post-hoc final-code check
    final_pass = False
    if last_code:
        final_exec = evaluator.evaluate(last_code, task["test_code"], timeout=15)
        final_pass = final_exec.passed

    passed = any_tool_pass or final_pass
    return {
        "task_id": task["task_id"],
        "passed": passed,
        "any_tool_pass": any_tool_pass,
        "final_pass": final_pass,
        "tool_calls_made": tool_calls_made,
        "autonomous_tool_called": tool_calls_made > 0,
        "turns": turn_log,
    }


def evaluate_checkpoint(label: str, base: str, adapter: Optional[str],
                        tasks: list, max_tool_calls: int, max_new_tokens: int,
                        temperature: float, top_p: float,
                        num_samples: int) -> dict:
    logger.info("Loading %s: base=%s adapter=%s T=%.2f k=%d",
                label, base, adapter, temperature, num_samples)
    model, tok = load_model(base, adapter)
    evaluator = CodeEvaluator()
    results = []
    for task in tasks:
        samples = []
        for s in range(num_samples):
            rec = run_task(model, tok, evaluator, task,
                           max_tool_calls=max_tool_calls,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature, top_p=top_p)
            rec["sample"] = s
            samples.append(rec)
            logger.info("  %s %s sample=%d: %s tool_called=%s tool_pass=%s final_pass=%s calls=%d",
                        label, task["task_id"], s,
                        "PASS" if rec["passed"] else "FAIL",
                        rec["autonomous_tool_called"],
                        rec["any_tool_pass"], rec["final_pass"],
                        rec["tool_calls_made"])
        results.append({
            "task_id": task["task_id"],
            "any_pass": any(x["passed"] for x in samples),
            "any_tool_call": any(x["autonomous_tool_called"] for x in samples),
            "samples": samples,
        })
    del model, tok
    torch.cuda.empty_cache()

    n = len(tasks)
    trials = n * num_samples
    sample_pass = sum(1 for r in results for s in r["samples"] if s["passed"])
    sample_tool_calls = sum(1 for r in results for s in r["samples"] if s["autonomous_tool_called"])
    any_pass = sum(1 for r in results if r["any_pass"])
    any_tool = sum(1 for r in results if r["any_tool_call"])
    return {
        "label": label,
        "num_tasks": n,
        "num_samples": num_samples,
        "mean_pass_rate": sample_pass / trials,
        "pass_at_k_rate": any_pass / n,
        "mean_tool_call_rate": sample_tool_calls / trials,
        "tool_call_at_k_rate": any_tool / n,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/hard_benchmarks/code/code_tasks_heldout.json")
    ap.add_argument("--output", default="results/autonomous_review/heldout.json")
    ap.add_argument("--checkpoints", nargs="+", default=["base", "sft_review"])
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft-review-adapter", default="models/qwen3-8b-autonomous-review-sft")
    ap.add_argument("--grpo-review-adapter", default="models/qwen3-8b-autonomous-review-grpo")
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    adapters = {
        "base": (args.base_model, None),
        "sft_review": (args.base_model, args.sft_review_adapter),
        "grpo_review": (args.base_model, args.grpo_review_adapter),
    }

    all_results = {}
    for label in args.checkpoints:
        base, adapter = adapters[label]
        try:
            all_results[label] = evaluate_checkpoint(
                label, base, adapter, tasks,
                max_tool_calls=args.max_tool_calls,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                num_samples=args.num_samples,
            )
        except Exception as e:
            logger.exception("Eval failed for %s", label)
            all_results[label] = {"label": label, "error": str(e)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))

    print("\n=== Autonomous-Review Summary ===")
    print(f"{'Model':<14} | {'pass@1':>6} | {'pass@k':>6} | {'tool@1':>6} | {'tool@k':>6}")
    print("-" * 56)
    for label in args.checkpoints:
        r = all_results.get(label, {})
        if "mean_pass_rate" not in r:
            print(f"{label:<14} | ERROR")
            continue
        print(f"{label:<14} | {r['mean_pass_rate']:6.1%} | {r['pass_at_k_rate']:6.1%} | "
              f"{r['mean_tool_call_rate']:6.1%} | {r['tool_call_at_k_rate']:6.1%}")


if __name__ == "__main__":
    main()
