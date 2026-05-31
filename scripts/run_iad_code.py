#!/usr/bin/env python3
"""IAD (Iterative Agent Decoding) baseline on the 15 Phase 1 hard code tasks.

Faithful approximation of Ruan et al. 2025 (arXiv:2504.01931):
  - Single agent (no separate reviewer)
  - At each iteration, sample K candidates at temperature T
  - Select the best candidate using a verifier R(x, y)
  - Feed the selected candidate + its execution feedback into the next iteration
  - Repeat for up to N iterations or until a candidate passes all tests

Simplifications vs. the paper:
  1. K = 3 candidates per iteration (paper varies 2-6; our task spec says 3-5).
  2. N = 3 max iterations (paper uses 3-4).
  3. Temperature = 0.7 (paper's Sketch2Code uses 0.6; need diversity for K>1).
  4. Verifier R: oracle test scoring. We split task['test_code'] into top-level
     `assert` statements, wrap each in a per-assertion try/except, and count
     pass / fail / runtime-error. Best = max(passes), tiebreak min(errors).
     This is a "near-optimal verifier" -- exactly what the IAD paper argues
     for as the limiting factor. The paper trades verifier quality off for
     coverage; we hold verifier quality at the oracle to give IAD its best
     chance.
  5. Budget: matched to H curve's B=20K cell (per-task cap on output tokens).

Outputs:
  results/iad_baseline/code_iad.json -- per-task records
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
import tempfile
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
logger = logging.getLogger("iad")

MODEL_ID = "claude-sonnet-4-20250514"
TASKS_PATH = REPO_ROOT / "data" / "hard_benchmarks" / "code" / "code_tasks.json"
OUT_JSON = REPO_ROOT / "results" / "iad_baseline" / "code_iad.json"

# IAD hyperparameters (matched to budget B = 20K)
K_CANDIDATES = 3            # samples per iteration
N_ITERATIONS = 3            # max outer iterations
TEMPERATURE = 0.7           # sampling temperature for diversity
PER_SAMPLE_MAX_TOKENS = 2200
BUDGET = 20000              # total output token cap per task

CODE_SYSTEM_PROMPT = """\
You are an expert Python programmer. Write correct, efficient solutions.

Rules:
- Output ONLY the function definition with any needed imports
- Handle ALL edge cases carefully
- Pay attention to boundary conditions, empty inputs, and special values
- Wrap your code in a ```python code block"""


# --------------------------------------------------------------------------- #
# Verifier: partial pass counting
# --------------------------------------------------------------------------- #


def _split_asserts(test_code: str) -> list[str]:
    """Split a test_code blob into individual top-level assert statements.

    Uses Python's AST so multi-line asserts (with parens) and asserts followed
    by other statements (e.g. variable assignments) are handled correctly.
    Returns a list of (still-top-level) statements; non-assert statements
    (setup code, etc.) are kept in line so per-test execution works.

    We return a list of statement source strings preserving order.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        # Fall back: line-based
        return [ln for ln in test_code.splitlines() if ln.strip().startswith("assert")]

    src_lines = test_code.splitlines(keepends=True)
    stmts: list[tuple[str, bool]] = []  # (src, is_assert)
    for node in tree.body:
        end_lineno = getattr(node, "end_lineno", node.lineno)
        lineno = node.lineno
        # ast lineno is 1-indexed; slice line range
        chunk = "".join(src_lines[lineno - 1 : end_lineno])
        is_assert = isinstance(node, ast.Assert)
        stmts.append((chunk, is_assert))
    return stmts  # type: ignore[return-value]


def score_candidate(code: str, test_code: str, timeout: int = 8) -> dict:
    """Run candidate against per-assertion harness and return a score dict.

    Returns:
        {
            'all_pass': bool,
            'n_pass': int,     # number of asserts that passed
            'n_fail': int,     # number of asserts that AssertionError'd
            'n_error': int,    # number of asserts that raised other exception
            'n_total': int,
            'first_error': str | None,  # first failing assert's error msg
            'stderr_full': str,         # full stderr if any
        }
    """
    stmts = _split_asserts(test_code)
    if not stmts:
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_fail": 0,
            "n_error": 1,
            "n_total": 0,
            "first_error": "no asserts found",
            "stderr_full": "",
        }

    # Build a harness that runs each statement in its own try/except.
    # Setup statements (non-asserts) are also wrapped to surface errors but
    # not counted as test fails.
    harness_lines: list[str] = [
        "import sys, traceback, json",
        "_iad_results = []",  # list of {"kind", "ok", "err"}
        "",
    ]
    for stmt_src, is_assert in stmts:
        kind = "assert" if is_assert else "setup"
        # Indent the statement by 4 spaces for the try-block.
        indented = "\n".join("    " + ln if ln.strip() else ln for ln in stmt_src.splitlines())
        harness_lines.append("try:")
        harness_lines.append(indented)
        harness_lines.append(
            f"    _iad_results.append({{'kind': {kind!r}, 'ok': True, 'err': None}})"
        )
        harness_lines.append("except AssertionError as _e:")
        harness_lines.append(
            f"    _iad_results.append({{'kind': {kind!r}, 'ok': False, 'err': 'AssertionError: ' + str(_e)}})"
        )
        harness_lines.append("except Exception as _e:")
        harness_lines.append(
            f"    _iad_results.append({{'kind': {kind!r}, 'ok': False, 'err': type(_e).__name__ + ': ' + str(_e)}})"
        )
        harness_lines.append("")
    harness_lines.append("print('===IAD_RESULTS===')")
    harness_lines.append("print(json.dumps(_iad_results))")
    harness = "\n".join(harness_lines)

    combined = f"{code}\n\n{harness}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(combined)
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        marker = "===IAD_RESULTS==="
        if marker in stdout:
            payload = stdout.split(marker, 1)[1].strip()
            try:
                results = json.loads(payload.splitlines()[0])
            except (json.JSONDecodeError, IndexError):
                results = []
        else:
            results = []

        if not results:
            # Module-level failure (import error, syntax error, etc.).
            return {
                "all_pass": False,
                "n_pass": 0,
                "n_fail": 0,
                "n_error": sum(1 for _, is_a in stmts if is_a),
                "n_total": sum(1 for _, is_a in stmts if is_a),
                "first_error": _first_stderr_line(stderr),
                "stderr_full": stderr[:2000],
            }

        n_pass = sum(1 for r in results if r["kind"] == "assert" and r["ok"])
        n_fail = sum(
            1 for r in results if r["kind"] == "assert" and not r["ok"]
            and r["err"].startswith("AssertionError")
        )
        n_error = sum(
            1 for r in results if r["kind"] == "assert" and not r["ok"]
            and not r["err"].startswith("AssertionError")
        )
        # Also count setup-failures as errors -- a broken setup means no tests
        # could even run, which is strictly worse than a failed assert.
        setup_failures = sum(
            1 for r in results if r["kind"] == "setup" and not r["ok"]
        )
        n_error += setup_failures
        n_total = n_pass + n_fail + n_error - setup_failures
        first_error = None
        for r in results:
            if not r["ok"]:
                first_error = r["err"]
                break

        return {
            "all_pass": (n_pass == n_total) and (n_total > 0),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "n_total": n_total,
            "first_error": first_error,
            "stderr_full": stderr[:2000],
        }

    except subprocess.TimeoutExpired:
        return {
            "all_pass": False,
            "n_pass": 0,
            "n_fail": 0,
            "n_error": sum(1 for _, is_a in stmts if is_a),
            "n_total": sum(1 for _, is_a in stmts if is_a),
            "first_error": f"Timeout after {timeout}s",
            "stderr_full": "",
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def _first_stderr_line(stderr: str) -> str:
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if line:
            return line[:200]
    return "Unknown error"


def candidate_rank_key(score: dict) -> tuple:
    """Higher is better. Use as max(..., key=candidate_rank_key)."""
    # 1) all_pass first  2) max n_pass  3) min n_error  4) min n_fail
    return (
        1 if score["all_pass"] else 0,
        score["n_pass"],
        -score["n_error"],
        -score["n_fail"],
    )


# --------------------------------------------------------------------------- #
# IAD main loop
# --------------------------------------------------------------------------- #


def run_iad(
    client: anthropic.Anthropic,
    task: dict,
    evaluator: CodeEvaluator,
) -> dict:
    """Run IAD on one task. Returns per-task record."""
    tid = task["task_id"]
    desc = task["description"]
    test_code = task["test_code"]

    cum_out = 0
    cum_in = 0
    t0 = time.time()
    iter_records: list[dict] = []
    best_code: str = ""
    best_score: dict | None = None
    passed = False
    final_code = ""

    # The "carried context" -- after each iter we feed best_code + its
    # execution feedback into the user prompt for the next iter.
    for it in range(N_ITERATIONS):
        if cum_out >= BUDGET:
            logger.info("  [%s] iter %d: budget exhausted (%d), stopping", tid, it, cum_out)
            break

        # Build user prompt for this iter
        if it == 0:
            user_content = desc
        else:
            fb = best_score.get("first_error", "unknown") if best_score else "unknown"
            stderr_snip = (best_score or {}).get("stderr_full", "")[:1200]
            pass_summary = (
                f"{best_score['n_pass']}/{best_score['n_total']} tests passed"
                if best_score
                else "no info"
            )
            user_content = (
                f"Problem:\n{desc}\n\n"
                f"--- Best candidate so far ({pass_summary}) ---\n"
                f"```python\n{best_code}\n```\n\n"
                f"--- Failure ---\n{fb}\n"
            )
            if stderr_snip:
                user_content += f"\nstderr:\n{stderr_snip}\n"
            user_content += (
                "\nFix the failure and output the complete corrected function "
                "in a ```python code block."
            )

        # Sample K candidates
        candidates: list[dict] = []
        for k in range(K_CANDIDATES):
            if cum_out >= BUDGET:
                logger.info(
                    "  [%s] iter %d cand %d: budget exhausted, stopping sampling",
                    tid, it, k,
                )
                break
            remaining = max(500, BUDGET - cum_out)
            per_max = min(PER_SAMPLE_MAX_TOKENS, remaining)
            try:
                resp = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=per_max,
                    temperature=TEMPERATURE,
                    system=CODE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
            except anthropic.APIError as e:
                logger.warning("  [%s] iter %d cand %d API error: %s", tid, it, k, e)
                candidates.append({
                    "k": k, "error": str(e), "code": "", "score": None,
                    "output_tokens": 0, "input_tokens": 0,
                })
                continue
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            code = extract_code(text, "python")
            score = score_candidate(code, test_code)
            cum_in += resp.usage.input_tokens
            cum_out += resp.usage.output_tokens
            candidates.append({
                "k": k,
                "output_tokens": resp.usage.output_tokens,
                "input_tokens": resp.usage.input_tokens,
                "stop_reason": resp.stop_reason,
                "score": score,
                "code": code,
            })
            logger.info(
                "  [%s] iter %d cand %d: pass=%d/%d err=%d tok=%d",
                tid, it, k, score["n_pass"], score["n_total"],
                score["n_error"], resp.usage.output_tokens,
            )

            # Early stop within the iter if a candidate passes all tests
            if score["all_pass"]:
                break

        # Select best candidate from this iter; carry the global best forward
        scored = [c for c in candidates if c.get("score") is not None]
        if not scored:
            # All API errors -- abort
            iter_records.append({
                "iter": it, "candidates": candidates, "selected_k": None,
                "no_valid_candidate": True,
            })
            break

        best_in_iter = max(scored, key=lambda c: candidate_rank_key(c["score"]))
        # Compare to global best
        if best_score is None or candidate_rank_key(best_in_iter["score"]) > candidate_rank_key(best_score):
            best_score = best_in_iter["score"]
            best_code = best_in_iter["code"]

        iter_records.append({
            "iter": it,
            "candidates": [
                {
                    "k": c["k"],
                    "output_tokens": c.get("output_tokens", 0),
                    "score": c.get("score"),
                    "error": c.get("error"),
                }
                for c in candidates
            ],
            "selected_k": best_in_iter["k"],
            "selected_score": best_in_iter["score"],
        })

        final_code = best_code
        if best_score and best_score["all_pass"]:
            passed = True
            break

    # Final verification with the standard evaluator for consistency with L/H
    if final_code:
        ev = evaluator.evaluate(final_code, test_code)
        passed = bool(ev.passed)
        final_error = ev.error_message
    else:
        final_error = "no code produced"

    elapsed = time.time() - t0
    return {
        "strategy": "IAD",
        "budget": BUDGET,
        "task_id": tid,
        "passed": passed,
        "input_tokens": cum_in,
        "output_tokens": cum_out,
        "tokens_used": cum_out,
        "wall_time_seconds": round(elapsed, 2),
        "num_iterations": len(iter_records),
        "num_candidates_total": sum(len(r["candidates"]) for r in iter_records),
        "iters": iter_records,
        "code_emitted": final_code,
        "error_message": final_error,
        "hyperparams": {
            "K": K_CANDIDATES,
            "N": N_ITERATIONS,
            "temperature": TEMPERATURE,
            "per_sample_max_tokens": PER_SAMPLE_MAX_TOKENS,
            "budget": BUDGET,
        },
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=None, help="comma-separated task IDs to limit")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    with TASKS_PATH.open() as f:
        all_tasks = json.load(f)
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = [t for t in all_tasks if t["task_id"] in keep]
    else:
        tasks = all_tasks

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    results: dict = {}
    if out_path.exists():
        try:
            results = json.load(out_path.open())
        except Exception:
            results = {}
    results.setdefault("records", {})
    results.setdefault("hyperparams", {
        "K": K_CANDIDATES, "N": N_ITERATIONS, "temperature": TEMPERATURE,
        "per_sample_max_tokens": PER_SAMPLE_MAX_TOKENS, "budget": BUDGET,
        "model": MODEL_ID,
    })

    client = anthropic.Anthropic()
    evaluator = CodeEvaluator()

    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        tid = task["task_id"]
        if tid in results["records"] and results["records"][tid].get("tokens_used", 0) > 0:
            logger.info("[%d/%d] skip %s (cached)", i, len(tasks), tid)
            continue
        logger.info("[%d/%d] %s", i, len(tasks), tid)
        try:
            rec = run_iad(client, task, evaluator)
        except Exception as e:  # noqa: BLE001
            logger.exception("task crashed")
            rec = {
                "strategy": "IAD", "task_id": tid, "passed": False,
                "tokens_used": 0, "error_message": f"crashed: {e}",
            }
        results["records"][tid] = rec
        logger.info(
            "  -> passed=%s out_tok=%d iters=%s time=%.1fs",
            rec.get("passed"), rec.get("tokens_used", 0),
            rec.get("num_iterations"), rec.get("wall_time_seconds", 0),
        )
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("Done in %.1f min", elapsed / 60)

    # Summary
    records = list(results["records"].values())
    n_pass = sum(1 for r in records if r.get("passed"))
    tot = len(records)
    toks = [r.get("tokens_used", 0) for r in records]
    mean_tok = sum(toks) / max(1, len(toks))
    print("\n=== IAD on 15 hard code tasks ===")
    print(f"K={K_CANDIDATES} N={N_ITERATIONS} T={TEMPERATURE} budget={BUDGET}")
    print(f"pass_rate = {n_pass}/{tot} = {n_pass / max(1, tot):.1%}")
    print(f"mean tokens used = {mean_tok:.0f}")
    failed = [r["task_id"] for r in records if not r.get("passed")]
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()
