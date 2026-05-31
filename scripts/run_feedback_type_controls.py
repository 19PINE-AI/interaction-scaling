"""Run Type 1 and Type 2 feedback-channel controls on the 15 hard code tasks.

Type 1 (LLM cross-review, no execution): a "blind" reviewer LLM critiques the
code based on the code itself + LLM priors only. NO test results, NO execution
output, NO static analysis. Same proposer model (Sonnet 4) as Phase 1.

Type 2 (Static analysis + LLM review, no execution): the reviewer sees the code
PLUS the output of `ruff check` run on the code. Still no execution, no test
results.

Both run with the same max_iterations=5 budget as Phase 1's Type 3a run. Final
pass is graded by running the held-out test_code (ground truth) — but the test
output is NEVER shown to the reviewer, only used for end-of-loop scoring.

Phase 1 baseline (Type 3a):
    Single-shot: 11/15 = 73.3%
    Reviewed   : 14/15 = 93.3%   (+20.0pp)

Usage:
    python scripts/run_feedback_type_controls.py --feedback-type 1
    python scripts/run_feedback_type_controls.py --feedback-type 2
    python scripts/run_feedback_type_controls.py --feedback-type all
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig  # noqa: E402
from src.evaluation.code_eval import CodeEvaluator  # noqa: E402
from src.utils.code_utils import extract_code  # noqa: E402
from src.utils.llm_client import get_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feedback_type_controls")

TASK_FILE = ROOT / "data" / "hard_benchmarks" / "code" / "code_tasks.json"
RESULTS_DIR = ROOT / "results" / "feedback_types"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompts (matched to Phase 1's run_code_task style)
# ---------------------------------------------------------------------------

CODE_SYSTEM_PROMPT = """\
You are an expert Python programmer. Write correct, efficient solutions.

Rules:
- Output ONLY the function definition with any needed imports
- Handle ALL edge cases carefully
- Pay attention to boundary conditions, empty inputs, and special values
- Wrap your code in a ```python code block"""


# --- Type 1: blind LLM reviewer (code only, no execution) -------------------

TYPE1_REVIEWER_SYSTEM = """\
You are an expert Python code reviewer performing a STATIC code review.

CRITICAL CONSTRAINTS:
- You have NOT executed the code and you do NOT have access to any test results,
  execution output, stack traces, or runtime values.
- You are evaluating the code purely by reading it, using your knowledge of
  Python semantics and common bugs.
- If the code looks correct, say so briefly and recommend submission.

Your job is to identify likely bugs, missed edge cases, logic errors, off-by-one
mistakes, and incorrect handling of the stated requirements. Be specific:
quote the offending line(s) and propose a concrete fix.
"""

TYPE1_REVIEWER_USER = """\
## Problem
{description}

## Candidate solution (iteration {iteration})
```python
{code}
```

Perform a static code review. Output:
1. **Issues** (numbered list, with severity: critical | major | minor and the
   relevant code fragment).
2. **Concrete fixes** for each issue.
3. **Verdict**: PASS (submit as-is) or REVISE (revise per the issues above).

Remember: you have no execution output. Reason purely from the code.
"""


# --- Type 2: static analysis + LLM reviewer (code + ruff, no execution) -----

TYPE2_REVIEWER_SYSTEM = """\
You are an expert Python code reviewer.

You will receive:
1. The problem description.
2. The candidate Python solution.
3. Output from a STATIC ANALYZER (ruff) run on the code. This is the ONLY
   automated signal you have — no execution, no test results, no runtime trace.

CRITICAL CONSTRAINTS:
- You have NOT executed the code and you do NOT see any test results or runtime
  output. Treat any inference about runtime behaviour as a guess.
- Use the ruff output as a grounding signal where applicable (it flags syntax
  errors, undefined names, unused variables, common bugs, style problems).
- Beyond ruff, you may reason about logic from the code itself.

Identify likely bugs, missed edge cases, and incorrect requirement handling.
Be specific: quote the offending line(s) and propose a concrete fix.
"""

TYPE2_REVIEWER_USER = """\
## Problem
{description}

## Candidate solution (iteration {iteration})
```python
{code}
```

## Static analysis output (ruff check, rules E/F/W/B)
```
{ruff_output}
```

Perform a code review using the above. Output:
1. **Issues** (numbered list, with severity: critical | major | minor and the
   relevant code fragment). Cite ruff findings where relevant.
2. **Concrete fixes** for each issue.
3. **Verdict**: PASS (submit as-is) or REVISE (revise per the issues above).

Remember: you have no execution output. Use ruff + code reading only.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_ruff(code: str) -> str:
    """Run ruff check on *code* with a focused ruleset and return its text output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(code)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        # Use E (pycodestyle errors), F (Pyflakes: undefined/unused/etc.),
        # W (warnings), B (bugbear: likely bugs).
        proc = subprocess.run(
            ["ruff", "check", "--select=E,F,W,B", "--no-cache", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if not out and not err:
            return "ruff: no issues found."
        combined = out
        if err:
            combined = (combined + "\n" + err).strip()
        # Replace temp filename in output for readability
        combined = combined.replace(str(tmp_path), "solution.py")
        return combined
    except FileNotFoundError:
        return "ruff: not installed (skipped)."
    except subprocess.TimeoutExpired:
        return "ruff: timed out after 15s."
    finally:
        tmp_path.unlink(missing_ok=True)


def looks_like_pass_verdict(text: str) -> bool:
    """Heuristic: does the reviewer's verdict say PASS?"""
    upper = text.upper()
    # Look for "VERDICT" line followed by PASS, or a final-line PASS.
    # Avoid being tricked by "verdict: revise" or "this would fail PASS test".
    if "VERDICT: PASS" in upper or "VERDICT** PASS" in upper or "VERDICT**: PASS" in upper:
        return True
    if "**VERDICT**: PASS" in upper or "**VERDICT:** PASS" in upper:
        return True
    # Permissive fallback
    if "VERDICT" in upper and "PASS" in upper.split("VERDICT", 1)[1][:60] and \
       "REVISE" not in upper.split("VERDICT", 1)[1][:60]:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    feedback_type: str  # "type1" or "type2"
    ss_passed: bool
    ss_tokens: int
    rv_passed: bool
    rv_iterations: int
    rv_total_tokens: int  # propose + review tokens combined
    rv_propose_tokens: int
    rv_review_tokens: int
    submitted_early: bool = False
    stopped_reason: str = ""
    final_code: str = ""
    review_trace: list[dict] = field(default_factory=list)


def run_task(
    task: dict,
    feedback_type: str,
    proposer_cfg: ModelConfig,
    reviewer_cfg: ModelConfig,
    max_iterations: int = 5,
) -> TaskResult:
    """Run a single code task with Type 1 or Type 2 feedback."""
    client = get_client()
    evaluator = CodeEvaluator()

    description = task["description"]
    test_code = task["test_code"]
    task_id = task["task_id"]

    client.reset_counters()

    # ---- Initial generation ------------------------------------------------
    init_messages = [{"role": "user", "content": description}]
    response = client.generate(
        config=proposer_cfg,
        system=CODE_SYSTEM_PROMPT,
        messages=init_messages,
    )
    current_code = extract_code(response.content, "python")
    ss_tokens = response.input_tokens + response.output_tokens

    # Hidden ground-truth eval (used only to score; NOT fed to the reviewer)
    eval_result = evaluator.evaluate(current_code, test_code)
    ss_passed = eval_result.passed
    rv_passed = ss_passed

    review_trace: list[dict] = []
    propose_tokens_rv = ss_tokens  # initial generation counts as part of rv too
    review_tokens_rv = 0
    iteration = 1
    submitted_early = False
    stopped_reason = "passed_initial" if ss_passed else ""

    if ss_passed:
        return TaskResult(
            task_id=task_id,
            feedback_type=feedback_type,
            ss_passed=True,
            ss_tokens=ss_tokens,
            rv_passed=True,
            rv_iterations=1,
            rv_total_tokens=ss_tokens,
            rv_propose_tokens=propose_tokens_rv,
            rv_review_tokens=0,
            submitted_early=False,
            stopped_reason="passed_initial",
            final_code=current_code,
            review_trace=[],
        )

    # ---- Interaction loop --------------------------------------------------
    # The reviewer is *blind* to test_code / execution. We never give it
    # eval_result. We just ask it to critique. If it says PASS we submit
    # (and re-grade with ground-truth). Otherwise we revise.
    last_eval_passed = ss_passed
    for it in range(max_iterations - 1):
        # Build review prompt
        if feedback_type == "type1":
            reviewer_system = TYPE1_REVIEWER_SYSTEM
            reviewer_user = TYPE1_REVIEWER_USER.format(
                description=description,
                code=current_code,
                iteration=iteration,
            )
            ruff_output = None
        elif feedback_type == "type2":
            ruff_output = run_ruff(current_code)
            reviewer_system = TYPE2_REVIEWER_SYSTEM
            reviewer_user = TYPE2_REVIEWER_USER.format(
                description=description,
                code=current_code,
                iteration=iteration,
                ruff_output=ruff_output,
            )
        else:
            raise ValueError(f"Unknown feedback_type: {feedback_type}")

        review_response = client.generate(
            config=reviewer_cfg,
            system=reviewer_system,
            messages=[{"role": "user", "content": reviewer_user}],
        )
        review_text = review_response.content
        review_in = review_response.input_tokens
        review_out = review_response.output_tokens
        review_tokens_rv += review_in + review_out

        review_trace.append({
            "iteration": iteration,
            "ruff_output": ruff_output,
            "review_text": review_text,
            "review_tokens": review_in + review_out,
        })

        # If the reviewer's verdict is PASS, accept and stop.
        if looks_like_pass_verdict(review_text):
            submitted_early = True
            stopped_reason = "reviewer_verdict_pass"
            break

        # Revise: feed the reviewer's critique back to the proposer.
        # The proposer also does NOT see test results (the feedback is just
        # the reviewer's text). This is by design — the Type 1/2 condition
        # forbids execution info.
        revision_messages = [
            {"role": "user", "content": description},
            {"role": "assistant", "content": f"```python\n{current_code}\n```"},
            {"role": "user", "content": (
                f"A code reviewer analyzed your code statically (no execution)"
                f" and provided this feedback:\n\n{review_text}\n\n"
                "Apply the changes you think are warranted. Preserve any code "
                "paths the reviewer did not flag."
            )},
        ]
        propose_response = client.generate(
            config=proposer_cfg,
            system=CODE_SYSTEM_PROMPT,
            messages=revision_messages,
        )
        propose_tokens_rv += propose_response.input_tokens + propose_response.output_tokens
        current_code = extract_code(propose_response.content, "python")
        iteration += 1

        # Hidden ground-truth eval for tracking final state (not exposed).
        eval_result = evaluator.evaluate(current_code, test_code)
        last_eval_passed = eval_result.passed
        review_trace[-1]["hidden_eval_passed"] = last_eval_passed

    rv_passed = last_eval_passed
    if not stopped_reason:
        stopped_reason = "max_iterations"

    return TaskResult(
        task_id=task_id,
        feedback_type=feedback_type,
        ss_passed=ss_passed,
        ss_tokens=ss_tokens,
        rv_passed=rv_passed,
        rv_iterations=iteration,
        rv_total_tokens=propose_tokens_rv + review_tokens_rv,
        rv_propose_tokens=propose_tokens_rv,
        rv_review_tokens=review_tokens_rv,
        submitted_early=submitted_early,
        stopped_reason=stopped_reason,
        final_code=current_code,
        review_trace=review_trace,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(feedback_type: str, max_iterations: int) -> dict:
    """Run all 15 tasks under one feedback regime."""
    with open(TASK_FILE) as f:
        tasks = json.load(f)

    proposer_cfg = ModelConfig.claude_sonnet()  # Sonnet 4, no thinking
    reviewer_cfg = ModelConfig.claude_sonnet()  # same family; persona prompt blocks exec info

    results: list[TaskResult] = []
    start = time.time()
    for task in tasks:
        t0 = time.time()
        logger.info("=== %s / %s ===", feedback_type, task["task_id"])
        r = run_task(task, feedback_type, proposer_cfg, reviewer_cfg, max_iterations)
        results.append(r)
        logger.info(
            "  ss=%s rv=%s iters=%d propose_tok=%d review_tok=%d stop=%s (%.1fs)",
            r.ss_passed, r.rv_passed, r.rv_iterations,
            r.rv_propose_tokens, r.rv_review_tokens, r.stopped_reason,
            time.time() - t0,
        )

    elapsed = time.time() - start

    # Summarize
    n = len(results)
    ss_pass = sum(1 for r in results if r.ss_passed)
    rv_pass = sum(1 for r in results if r.rv_passed)
    avg_iters = sum(r.rv_iterations for r in results) / n
    avg_propose = sum(r.rv_propose_tokens for r in results) / n
    avg_review = sum(r.rv_review_tokens for r in results) / n

    summary = {
        "feedback_type": feedback_type,
        "n": n,
        "ss_pass": ss_pass,
        "ss_pass_rate": round(ss_pass / n, 3),
        "rv_pass": rv_pass,
        "rv_pass_rate": round(rv_pass / n, 3),
        "delta_pp": round((rv_pass - ss_pass) / n * 100, 1),
        "avg_iterations": round(avg_iters, 2),
        "avg_propose_tokens": round(avg_propose, 0),
        "avg_review_tokens": round(avg_review, 0),
        "avg_total_rv_tokens": round(avg_propose + avg_review, 0),
        "wall_time_seconds": round(elapsed, 1),
        "max_iterations": max_iterations,
        "proposer_model": proposer_cfg.model_id,
        "reviewer_model": reviewer_cfg.model_id,
    }

    out = {
        "summary": summary,
        "results": [
            {
                "task_id": r.task_id,
                "ss_passed": r.ss_passed,
                "ss_tokens": r.ss_tokens,
                "rv_passed": r.rv_passed,
                "rv_iterations": r.rv_iterations,
                "rv_propose_tokens": r.rv_propose_tokens,
                "rv_review_tokens": r.rv_review_tokens,
                "rv_total_tokens": r.rv_total_tokens,
                "submitted_early": r.submitted_early,
                "stopped_reason": r.stopped_reason,
                "final_code": r.final_code,
                "review_trace": r.review_trace,
            }
            for r in results
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feedback-type",
        choices=["1", "2", "all"],
        default="all",
        help="Which feedback type to run (1 = LLM only, 2 = LLM + ruff, all = both).",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Same budget as Phase 1 Type 3a (default: 5).",
    )
    args = ap.parse_args()

    todo = ["type1", "type2"] if args.feedback_type == "all" else [f"type{args.feedback_type}"]

    for ft in todo:
        logger.info("\n\n########## Running %s ##########\n", ft.upper())
        out = run_all(ft, args.max_iterations)
        out_path = RESULTS_DIR / f"code_{ft}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Saved %s results to %s", ft, out_path)
        print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
