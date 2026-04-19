"""Multi-turn interaction-scaling evaluation for Qwen3-8B checkpoints.

For each (model, N, task) triple, runs a GENERATE→EXECUTE→REVIEW loop for up
to N turns. If execution passes any turn, the task is counted as solved in
<=N turns. This measures the interaction-scaling curve: pass@1(N) vs N per
model.

Headline claim the loop is designed to test: SFT/GRPO at N=1 should match or
exceed the base model at N=5 — i.e. the student has internalised what the
teacher needed multi-turn loops to achieve.

Usage:
    python -m src.evaluation.interaction_loop_eval \
        --tasks data/hard_benchmarks/code/code_tasks_heldout.json \
        --output results/interaction_scaling/heldout.json \
        --checkpoints base sft grpo \
        --budgets 1 2 3 5
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

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def system_prompt(budget: int) -> str:
    return (
        "You are an AI agent that solves tasks through interaction scaling. "
        f"You have a budget of {budget} steps to solve the task. At each step, you can:\n"
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


def initial_user(description: str, budget: int) -> str:
    return f"[BUDGET: {budget} steps remaining]\n[TASK]: Fix the bug in this code.\n\n{description}"


def feedback_user(budget_remaining: int, prev_error: str, prev_stderr: str) -> str:
    stderr_tail = "\n".join(prev_stderr.strip().splitlines()[-20:])
    return (
        f"[BUDGET: {budget_remaining} steps remaining]\n"
        f"[EXECUTE] Your previous solution failed the tests.\n"
        f"Error: {prev_error}\n"
        f"Stderr tail:\n{stderr_tail}\n\n"
        f"[REVIEW] Please analyze the failure and [GENERATE] a corrected implementation. "
        f"Output the full function in a ```python code block."
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


def generate(model, tok, messages: list, max_new_tokens: int = 1536,
             temperature: float = 0.0, top_p: float = 1.0) -> str:
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def extract_code(response: str) -> str:
    matches = CODE_BLOCK_RE.findall(response)
    return matches[-1].strip() if matches else ""


def run_task(model, tok, evaluator: CodeEvaluator, task: dict, budget: int,
             temperature: float = 0.0, top_p: float = 1.0) -> dict:
    """Run one task for up to `budget` turns."""
    messages = [
        {"role": "system", "content": system_prompt(budget)},
        {"role": "user", "content": initial_user(task["description"], budget)},
    ]
    turns = []
    passed = False
    for turn_idx in range(budget):
        t0 = time.perf_counter()
        response = generate(model, tok, messages, temperature=temperature, top_p=top_p)
        code = extract_code(response)
        if not code:
            turns.append({
                "turn": turn_idx + 1,
                "passed": False,
                "error": "no_code_block",
                "response_len": len(response),
                "gen_s": round(time.perf_counter() - t0, 2),
            })
            break
        exec_result = evaluator.evaluate(code, task["test_code"], timeout=15)
        turn_record = {
            "turn": turn_idx + 1,
            "passed": exec_result.passed,
            "error": exec_result.error_message,
            "response_len": len(response),
            "code_len": len(code),
            "gen_s": round(time.perf_counter() - t0, 2),
        }
        turns.append(turn_record)
        if exec_result.passed:
            passed = True
            break
        # Append assistant's response and feedback user message for next turn.
        if turn_idx + 1 < budget:
            messages.append({"role": "assistant", "content": response})
            remaining = budget - (turn_idx + 1)
            messages.append({
                "role": "user",
                "content": feedback_user(remaining, exec_result.error_message or "unknown",
                                         exec_result.stderr),
            })
    return {
        "task_id": task["task_id"],
        "budget": budget,
        "passed": passed,
        "turns_used": len(turns),
        "turns": turns,
    }


def evaluate_checkpoint(label: str, base: str, adapter: Optional[str],
                        tasks: list, budgets: list,
                        temperature: float = 0.0, top_p: float = 1.0,
                        num_samples: int = 1) -> dict:
    logger.info("Loading %s: base=%s adapter=%s T=%.2f k=%d",
                label, base, adapter, temperature, num_samples)
    model, tok = load_model(base, adapter)
    evaluator = CodeEvaluator()
    per_budget = {}
    for N in budgets:
        logger.info("  Running %s @ budget=%d", label, N)
        results = []
        sample_pass_total = 0
        any_pass_tasks = 0
        for task in tasks:
            samples = []
            task_any_pass = False
            for s in range(num_samples):
                rec = run_task(model, tok, evaluator, task, N,
                               temperature=temperature, top_p=top_p)
                rec["sample"] = s
                samples.append(rec)
                if rec["passed"]:
                    sample_pass_total += 1
                    task_any_pass = True
                logger.info("    %s N=%d %s sample=%d: %s (turns=%d)",
                            label, N, task["task_id"], s,
                            "PASS" if rec["passed"] else "FAIL", rec["turns_used"])
            if task_any_pass:
                any_pass_tasks += 1
            results.append({"task_id": task["task_id"], "samples": samples,
                            "any_pass": task_any_pass})
        total_trials = len(tasks) * num_samples
        per_budget[str(N)] = {
            "passed": sample_pass_total,
            "total_trials": total_trials,
            "num_tasks": len(tasks),
            "num_samples": num_samples,
            "mean_pass_rate": sample_pass_total / total_trials,
            "pass_at_k_rate": any_pass_tasks / len(tasks),
            "results": results,
        }
    del model, tok
    torch.cuda.empty_cache()
    return {"label": label, "per_budget": per_budget}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hard_benchmarks/code/code_tasks_heldout.json")
    parser.add_argument("--output", default="results/interaction_scaling/heldout.json")
    parser.add_argument("--checkpoints", nargs="+", default=["base", "sft", "grpo"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--sft-adapter", default="models/qwen3-8b-interaction-scaling-sft")
    parser.add_argument("--grpo-adapter", default="models/qwen3-8b-interaction-scaling-grpo")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0 = greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Independent samples per (task, budget). Only meaningful with temperature>0.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    adapters = {
        "base": (args.base_model, None),
        "sft": (args.base_model, args.sft_adapter),
        "grpo": (args.base_model, args.grpo_adapter),
    }

    all_results = {}
    for label in args.checkpoints:
        base, adapter = adapters[label]
        try:
            all_results[label] = evaluate_checkpoint(
                label, base, adapter, tasks, args.budgets,
                temperature=args.temperature, top_p=args.top_p,
                num_samples=args.num_samples,
            )
        except Exception as e:
            logger.exception("Eval failed for %s", label)
            all_results[label] = {"label": label, "error": str(e)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))

    print("\n=== Interaction-Scaling Summary ===")
    print(f"{'Model':<8} | " + " | ".join(f"N={N:>2}" for N in args.budgets))
    print("-" * (10 + 8 * len(args.budgets)))
    for label in args.checkpoints:
        r = all_results.get(label, {})
        if "per_budget" not in r:
            print(f"{label:<8} | ERROR")
            continue
        row = [f"{r['per_budget'][str(N)]['mean_pass_rate']:.1%}" for N in args.budgets]
        print(f"{label:<8} | " + " | ".join(f"{c:>4}" for c in row))


if __name__ == "__main__":
    main()
