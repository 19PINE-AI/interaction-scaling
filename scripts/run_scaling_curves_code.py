#!/usr/bin/env python3
"""Performance vs token-budget curves for four interaction strategies.

Strategies:
  R - Reasoning-only (extended thinking, no tools/execution/review)
  S - Best-of-N sampling (independent generations, scored by oracle test)
  L - Single-agent loop (one agent context grows with execute -> stderr -> revise)
  H - Proposer-reviewer harness (separate proposer + reviewer; reviewer sees
      only latest artifact, not full history)

Sweeps budget B in {1K, 5K, 20K} total output tokens per task.
Runs on the 15 hard code tasks (data/hard_benchmarks/code/code_tasks.json).
Sonnet 4 (claude-sonnet-4-20250514). Temp=0 for R/L/H, temp=1.0 for S.

Outputs:
  results/scaling_curves/code_4strategy.json   - per-cell records
  results/scaling_curves/code_4strategy_summary.csv  - aggregate table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anthropic

from src.evaluation.code_eval import CodeEvaluator
from src.utils.code_utils import extract_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scaling_curves")

MODEL_ID = "claude-sonnet-4-20250514"
TASKS_PATH = REPO_ROOT / "data" / "hard_benchmarks" / "code" / "code_tasks.json"
OUT_JSON = REPO_ROOT / "results" / "scaling_curves" / "code_4strategy.json"
OUT_CSV = REPO_ROOT / "results" / "scaling_curves" / "code_4strategy_summary.csv"

# --------------------------------------------------------------------------- #
# Prompts (kept identical to Phase 1 wherever possible)
# --------------------------------------------------------------------------- #

CODE_SYSTEM_PROMPT = """\
You are an expert Python programmer. Write correct, efficient solutions.

Rules:
- Output ONLY the function definition with any needed imports
- Handle ALL edge cases carefully
- Pay attention to boundary conditions, empty inputs, and special values
- Wrap your code in a ```python code block"""

REVIEWER_SYSTEM_PROMPT = """\
You are a meticulous Python code reviewer. You will be shown a function and
the output of running it against a hidden test suite (stderr / error message).

Produce a JSON object:
{
  "issues": [{"description": "...", "severity": "critical|major|minor"}],
  "suggestions": ["..."],
  "confidence": 0.0-1.0
}

Rules:
- Diagnose the failure cause precisely. Do not restate the error.
- Each suggestion must be a concrete code change.
- Output ONLY the JSON object."""

# --------------------------------------------------------------------------- #
# Budgets and per-strategy iteration counts
# --------------------------------------------------------------------------- #

BUDGETS = [1000, 5000, 20000]

# Best-of-N samples per budget. ~2-3K output / sample so:
#   B=1K -> N=1 (single sample); B=5K -> N=3; B=20K -> N=10
N_SAMPLES = {1000: 1, 5000: 3, 20000: 10}

# Max turns for L and H per budget (one turn = propose [+ review for H] + revise)
#   each propose ~ 1-2K, exec is free, review ~ 0.5K
MAX_TURNS_L = {1000: 1, 5000: 3, 20000: 10}
MAX_TURNS_H = {1000: 1, 5000: 2, 20000: 6}  # H burns more tokens per turn (reviewer)

# Thinking budget for R. Anthropic requires thinking >= 1024 if enabled.
# B=1K -> disable thinking; B=5K -> thinking=2048, max_tokens=5000;
# B=20K -> thinking=16000, max_tokens=20000.
R_PARAMS = {
    1000: {"thinking": None, "max_tokens": 1000},
    5000: {"thinking": 2048, "max_tokens": 5000},
    20000: {"thinking": 16000, "max_tokens": 20000},
}

# --------------------------------------------------------------------------- #
# Strategy implementations
# --------------------------------------------------------------------------- #


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (we don't include these in the SFT-style
    output, but extract_code already handles fenced blocks fine)."""
    return text


def strategy_R(client, task: dict, budget: int, evaluator: CodeEvaluator) -> dict:
    """Reasoning-only: single API call with extended thinking, no tools."""
    p = R_PARAMS[budget]
    kwargs = dict(
        model=MODEL_ID,
        max_tokens=p["max_tokens"],
        system=CODE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": task["description"]}],
    )
    if p["thinking"] is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": p["thinking"]}
        kwargs["temperature"] = 1.0  # required by thinking
    else:
        kwargs["temperature"] = 0.0

    t0 = time.time()
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.APIError as e:
        logger.warning("R API error: %s", e)
        return _failed_record("R", budget, task, str(e), 0, 0, 0, 0)

    elapsed = time.time() - t0
    thinking_text = ""
    answer_text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "thinking":
            thinking_text += block.thinking
        elif getattr(block, "type", None) == "text":
            answer_text += block.text

    code = extract_code(answer_text, "python")
    eval_result = evaluator.evaluate(code, task["test_code"])
    return {
        "strategy": "R",
        "budget": budget,
        "task_id": task["task_id"],
        "passed": bool(eval_result.passed),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,  # includes thinking
        "tokens_used": resp.usage.output_tokens,
        "wall_time_seconds": round(elapsed, 2),
        "num_turns": 1,
        "thinking_chars": len(thinking_text),
        "stop_reason": resp.stop_reason,
        "code_emitted": code,
        "error_message": eval_result.error_message,
    }


def strategy_S(client, task: dict, budget: int, evaluator: CodeEvaluator) -> dict:
    """Best-of-N: N independent single-shot generations at temp=1.0.
    Scoring: pass@N (passes if ANY sample passes). Reports per-sample results."""
    n = N_SAMPLES[budget]
    # per-sample max_tokens: room for ~2K output, cap so total ~ budget
    per_max = max(800, min(2500, budget // max(1, n) + 500))

    samples = []
    cum_out = 0
    cum_in = 0
    passed_any = False
    first_passing_idx = None
    t0 = time.time()

    for i in range(n):
        if cum_out > budget:
            break
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=per_max,
                temperature=1.0,
                system=CODE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": task["description"]}],
            )
        except anthropic.APIError as e:
            logger.warning("S sample %d API error: %s", i, e)
            samples.append({"idx": i, "error": str(e)})
            continue

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        code = extract_code(text, "python")
        ev = evaluator.evaluate(code, task["test_code"])
        cum_in += resp.usage.input_tokens
        cum_out += resp.usage.output_tokens
        s_passed = bool(ev.passed)
        if s_passed and not passed_any:
            passed_any = True
            first_passing_idx = i
        samples.append(
            {
                "idx": i,
                "passed": s_passed,
                "output_tokens": resp.usage.output_tokens,
                "stop_reason": resp.stop_reason,
                "code": code if s_passed else code,
                "error_message": ev.error_message,
            }
        )

    elapsed = time.time() - t0
    code_emitted = ""
    if first_passing_idx is not None:
        code_emitted = samples[first_passing_idx]["code"]
    elif samples and "code" in samples[-1]:
        code_emitted = samples[-1]["code"]

    return {
        "strategy": "S",
        "budget": budget,
        "task_id": task["task_id"],
        "passed": passed_any,
        "input_tokens": cum_in,
        "output_tokens": cum_out,
        "tokens_used": cum_out,
        "wall_time_seconds": round(elapsed, 2),
        "num_turns": len(samples),
        "n_samples": n,
        "n_passed_samples": sum(1 for s in samples if s.get("passed")),
        "first_passing_idx": first_passing_idx,
        "code_emitted": code_emitted,
        "samples": samples,
    }


def strategy_L(client, task: dict, budget: int, evaluator: CodeEvaluator) -> dict:
    """Single-agent loop: ONE agent's context grows. After each generation,
    run tests; on failure, append stderr + 'please fix' to the SAME conversation
    and continue. No separate reviewer."""
    max_turns = MAX_TURNS_L[budget]
    history: list[dict] = [{"role": "user", "content": task["description"]}]
    cum_out = 0
    cum_in = 0
    passed = False
    current_code = ""
    t0 = time.time()
    turn_records = []

    for turn in range(max_turns):
        remaining = max(500, budget - cum_out)
        per_max = min(2500, remaining)
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=per_max,
                temperature=0.0,
                system=CODE_SYSTEM_PROMPT,
                messages=history,
            )
        except anthropic.APIError as e:
            logger.warning("L turn %d API error: %s", turn, e)
            turn_records.append({"turn": turn, "error": str(e)})
            break

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        current_code = extract_code(text, "python")
        cum_in += resp.usage.input_tokens
        cum_out += resp.usage.output_tokens

        ev = evaluator.evaluate(current_code, task["test_code"])
        passed = bool(ev.passed)
        turn_records.append({
            "turn": turn,
            "output_tokens": resp.usage.output_tokens,
            "passed": passed,
            "stop_reason": resp.stop_reason,
            "error_message": ev.error_message,
        })

        if passed:
            break
        if cum_out >= budget:
            break
        if turn == max_turns - 1:
            break

        # Append assistant turn + execution feedback to context
        history.append({"role": "assistant", "content": text})
        fb = f"Your code failed the tests.\nerror: {ev.error_message}\n"
        if ev.stderr:
            fb += f"stderr:\n{ev.stderr[:1500]}\n"
        if ev.stdout:
            fb += f"stdout:\n{ev.stdout[:500]}\n"
        fb += "\nPlease fix the issue and output the complete corrected function."
        history.append({"role": "user", "content": fb})

    elapsed = time.time() - t0
    return {
        "strategy": "L",
        "budget": budget,
        "task_id": task["task_id"],
        "passed": passed,
        "input_tokens": cum_in,
        "output_tokens": cum_out,
        "tokens_used": cum_out,
        "wall_time_seconds": round(elapsed, 2),
        "num_turns": len(turn_records),
        "code_emitted": current_code,
        "turns": turn_records,
    }


def strategy_H(client, task: dict, budget: int, evaluator: CodeEvaluator) -> dict:
    """Proposer-reviewer harness: SEPARATE reviewer agent that sees only the
    latest artifact + execution output (NOT the full history). Reviewer emits
    structured JSON; proposer revises against that JSON in a fresh context."""
    max_turns = MAX_TURNS_H[budget]
    cum_out = 0
    cum_in = 0
    passed = False
    current_code = ""
    t0 = time.time()
    turn_records = []
    last_feedback_text = ""  # accumulated only for the proposer's next turn

    for turn in range(max_turns):
        remaining = max(500, budget - cum_out)
        per_max = min(2500, remaining)

        # ---- Proposer call ----
        if turn == 0:
            prop_messages = [{"role": "user", "content": task["description"]}]
        else:
            user_content = (
                f"Problem:\n{task['description']}\n\n"
                f"--- Your previous code ---\n```python\n{current_code}\n```\n\n"
                f"--- Reviewer feedback ---\n{last_feedback_text}\n\n"
                "Output the complete revised function in a ```python code block."
            )
            prop_messages = [{"role": "user", "content": user_content}]

        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=per_max,
                temperature=0.0,
                system=CODE_SYSTEM_PROMPT,
                messages=prop_messages,
            )
        except anthropic.APIError as e:
            logger.warning("H turn %d proposer API error: %s", turn, e)
            turn_records.append({"turn": turn, "error": str(e), "phase": "propose"})
            break

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        current_code = extract_code(text, "python")
        cum_in += resp.usage.input_tokens
        propose_out = resp.usage.output_tokens
        cum_out += propose_out

        ev = evaluator.evaluate(current_code, task["test_code"])
        passed = bool(ev.passed)
        turn_rec = {
            "turn": turn,
            "propose_tokens": propose_out,
            "passed": passed,
            "stop_reason": resp.stop_reason,
            "error_message": ev.error_message,
        }

        if passed or cum_out >= budget or turn == max_turns - 1:
            turn_records.append(turn_rec)
            break

        # ---- Reviewer call (sees ONLY latest code + execution result) ----
        review_remaining = max(300, budget - cum_out)
        review_max = min(800, review_remaining)
        if review_max < 200:
            turn_records.append(turn_rec)
            break

        exec_summary = f"error: {ev.error_message}\n"
        if ev.stderr:
            exec_summary += f"stderr:\n{ev.stderr[:1500]}\n"
        if ev.stdout:
            exec_summary += f"stdout:\n{ev.stdout[:400]}\n"

        rev_user = (
            f"Problem:\n{task['description']}\n\n"
            f"Code:\n```python\n{current_code}\n```\n\n"
            f"Execution result:\n```\n{exec_summary}\n```\n\n"
            "Respond with ONLY the JSON review object."
        )

        try:
            rev_resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=review_max,
                temperature=0.0,
                system=REVIEWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": rev_user}],
            )
        except anthropic.APIError as e:
            logger.warning("H turn %d reviewer API error: %s", turn, e)
            turn_records.append({**turn_rec, "review_error": str(e)})
            break

        rev_text = "".join(b.text for b in rev_resp.content if getattr(b, "type", None) == "text")
        cum_in += rev_resp.usage.input_tokens
        cum_out += rev_resp.usage.output_tokens
        last_feedback_text = rev_text
        turn_rec["review_tokens"] = rev_resp.usage.output_tokens
        turn_records.append(turn_rec)

        if cum_out >= budget:
            break

    elapsed = time.time() - t0
    return {
        "strategy": "H",
        "budget": budget,
        "task_id": task["task_id"],
        "passed": passed,
        "input_tokens": cum_in,
        "output_tokens": cum_out,
        "tokens_used": cum_out,
        "wall_time_seconds": round(elapsed, 2),
        "num_turns": len(turn_records),
        "code_emitted": current_code,
        "turns": turn_records,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failed_record(strategy, budget, task, err, in_tok, out_tok, turns, elapsed):
    return {
        "strategy": strategy,
        "budget": budget,
        "task_id": task["task_id"],
        "passed": False,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "tokens_used": out_tok,
        "wall_time_seconds": elapsed,
        "num_turns": turns,
        "code_emitted": "",
        "error_message": err,
    }


STRATEGY_FN = {
    "R": strategy_R,
    "S": strategy_S,
    "L": strategy_L,
    "H": strategy_H,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default="R,S,L,H", help="comma-separated")
    parser.add_argument("--budgets", default="1000,5000,20000", help="comma-separated")
    parser.add_argument("--tasks", default=None, help="comma-separated task IDs to limit")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    with TASKS_PATH.open() as f:
        all_tasks = json.load(f)
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = [t for t in all_tasks if t["task_id"] in keep]
    else:
        tasks = all_tasks
    logger.info("%d tasks; strategies=%s; budgets=%s", len(tasks), strategies, budgets)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to support resumption
    results: dict = {"R": {}, "S": {}, "L": {}, "H": {}}
    if out_path.exists():
        try:
            results = json.load(out_path.open())
            for s in ["R", "S", "L", "H"]:
                results.setdefault(s, {})
        except Exception:
            pass

    client = _client()
    evaluator = CodeEvaluator()

    total_cells = len(strategies) * len(budgets) * len(tasks)
    cell_idx = 0
    t_start = time.time()

    for strategy in strategies:
        results.setdefault(strategy, {})
        for budget in budgets:
            b_key = str(budget)
            results[strategy].setdefault(b_key, {})
            for task in tasks:
                cell_idx += 1
                tid = task["task_id"]
                # Resume: skip if already done
                if tid in results[strategy][b_key] and results[strategy][b_key][tid].get("tokens_used", 0) > 0:
                    logger.info("[%d/%d] skip %s B=%d %s (cached)", cell_idx, total_cells, strategy, budget, tid)
                    continue

                logger.info("[%d/%d] %s B=%d %s", cell_idx, total_cells, strategy, budget, tid)
                try:
                    rec = STRATEGY_FN[strategy](client, task, budget, evaluator)
                except Exception as e:  # noqa: BLE001
                    logger.exception("cell crashed")
                    rec = _failed_record(strategy, budget, task, f"crashed: {e}", 0, 0, 0, 0)
                results[strategy][b_key][tid] = rec
                logger.info(
                    "  -> passed=%s out_tok=%d turns=%s time=%.1fs",
                    rec.get("passed"), rec.get("output_tokens", 0),
                    rec.get("num_turns"), rec.get("wall_time_seconds", 0),
                )
                # Incremental save
                with out_path.open("w") as f:
                    json.dump(results, f, indent=2)

    elapsed_total = time.time() - t_start
    logger.info("All cells done in %.1f min", elapsed_total / 60)

    # Build CSV summary
    rows = []
    for strategy in ["R", "S", "L", "H"]:
        if strategy not in results:
            continue
        for b_key, cells in sorted(results[strategy].items(), key=lambda x: int(x[0])):
            passes = sum(1 for r in cells.values() if r.get("passed"))
            tot = len(cells)
            toks = [r.get("tokens_used", 0) for r in cells.values()]
            mean_tok = sum(toks) / max(1, len(toks))
            med_tok = sorted(toks)[len(toks) // 2] if toks else 0
            rows.append({
                "strategy": strategy,
                "budget": int(b_key),
                "n_pass": passes,
                "n_total": tot,
                "pass_rate": round(passes / max(1, tot), 4),
                "mean_tokens": round(mean_tok, 1),
                "median_tokens": med_tok,
            })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "budget", "n_pass", "n_total",
                                           "pass_rate", "mean_tokens", "median_tokens"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s and %s", out_path, OUT_CSV)

    # Print table
    print("\n=== Pass rate vs budget ===")
    print(f"{'strategy':<10}{'budget':<10}{'pass_rate':<12}{'mean_tokens':<14}{'n_pass/n_total'}")
    for r in rows:
        print(f"{r['strategy']:<10}{r['budget']:<10}{r['pass_rate']:<12}{r['mean_tokens']:<14}{r['n_pass']}/{r['n_total']}")


if __name__ == "__main__":
    main()
