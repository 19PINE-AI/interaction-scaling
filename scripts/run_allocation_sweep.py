"""Budget allocation sweep over the proposer-reviewer harness on hard code tasks.

Contribution 3: given a fixed total token budget B, how should it be split
across (b1) proposal, (b2) execution/environment, (b3) review?

We hold B = 10K tokens total per task and sweep 9 points on the allocation
simplex (b1, b2, b3) summing to 1. Since the execute phase consumes no LLM
tokens (it's pytest in a subprocess), b1 and b3 are realized as per-call
`max_tokens` caps on the proposer and reviewer respectively, scaled to
B / max_iterations. b2 acts as reserved budget — slack the harness can use
for additional iterations when proposer/reviewer run lean.

Reference (section 4): code single-shot 66.7%, reviewed 100% over the same
15 hand-curated tasks at 500K-token cap (rarely binding; mean reviewed
tokens 4,575). At B=10K we are in the regime where the cap can bite.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

# Allow `python scripts/run_allocation_sweep.py` from project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.meta_controller import MetaController
from src.agents.proposer import ProposerAgent
from src.agents.reviewer import ReviewerAgent
from src.budget.allocator import AllocationStrategy, BudgetAllocator
from src.config import ModelConfig
from src.evaluation.code_eval import CodeEvaluator
from src.feedback.type3a_execution import ExecutionFeedback
from src.utils.llm_client import get_client

logging.basicConfig(
    level=logging.WARNING,  # quiet for batch run
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = PROJECT_ROOT / "data" / "hard_benchmarks" / "code" / "code_tasks.json"
OUT_PATH = PROJECT_ROOT / "results" / "allocation_sweep" / "code_allocation.json"

TOTAL_BUDGET = 10_000
MAX_ITERATIONS = 5

# 9 allocation points on the simplex (b1, b2, b3) with b1+b2+b3=1.
ALLOCATIONS = [
    ("A_propose_heavy",   (0.80, 0.10, 0.10)),
    ("B_execute_heavy",   (0.10, 0.80, 0.10)),
    ("C_review_heavy",    (0.10, 0.10, 0.80)),
    ("D_prop_exec",       (0.40, 0.40, 0.20)),
    ("E_prop_review",     (0.40, 0.20, 0.40)),
    ("F_exec_review",     (0.20, 0.40, 0.40)),
    ("G_prop_dominant",   (0.50, 0.25, 0.25)),
    ("H_review_dominant", (0.25, 0.25, 0.50)),
    ("I_equal",           (0.33, 0.34, 0.33)),  # last gets 0.34 to sum to 1
]

# Floors for per-call max_tokens. The Anthropic API requires max_tokens >= 1.
# We use small but non-zero floors to keep the API call alive even on
# starved phases — under tight caps the model just truncates output, which
# is the experimental signal we want to measure.
PROPOSE_FLOOR = 256
REVIEW_FLOOR = 256


def make_capped_proposer_reviewer(b1_frac: float, b3_frac: float):
    """Build proposer and reviewer agents with per-call max_tokens scaled
    to the allocation."""
    propose_cap = max(PROPOSE_FLOOR, int(b1_frac * TOTAL_BUDGET / MAX_ITERATIONS))
    review_cap = max(REVIEW_FLOOR, int(b3_frac * TOTAL_BUDGET / MAX_ITERATIONS))

    # Cap at the base Sonnet limit
    propose_cap = min(propose_cap, 8192)
    review_cap = min(review_cap, 8192)

    base = ModelConfig.claude_sonnet()
    proposer_cfg = ModelConfig(
        provider=base.provider,
        model_id=base.model_id,
        max_tokens=propose_cap,
        temperature=0.0,
        use_thinking=False,
    )
    reviewer_cfg = ModelConfig(
        provider=base.provider,
        model_id=base.model_id,
        max_tokens=review_cap,
        temperature=0.0,
        use_thinking=False,
    )
    return ProposerAgent(proposer_cfg), ReviewerAgent(reviewer_cfg), propose_cap, review_cap


def run_task(task: dict, b1: float, b2: float, b3: float) -> dict:
    """Run one (task, allocation) cell. Returns a result record."""
    proposer, reviewer, propose_cap, review_cap = make_capped_proposer_reviewer(b1, b3)
    feedback_providers = [ExecutionFeedback()]

    # Use a custom allocator with fixed ratios matching this sweep point,
    # purely for the iteration-level logging (the actual per-call caps are
    # already enforced via max_tokens above).
    allocator = BudgetAllocator(fixed_ratios=(b1, b2, b3))

    controller = MetaController(
        proposer=proposer,
        reviewer=reviewer,
        feedback_providers=feedback_providers,
        budget_tokens=TOTAL_BUDGET,
        max_iterations=MAX_ITERATIONS,
        allocation_strategy=AllocationStrategy.FIXED,
    )
    controller.allocator = allocator

    problem_dict = {
        "test_code": task["test_code"],
        "entry_point": "",
        "prompt": "",
    }

    client = get_client()
    client.reset_counters()
    t0 = time.time()
    try:
        run_result = controller.run(task["task_id"], task["description"], problem_dict)
    except Exception as e:
        logger.error("Task %s crashed: %s", task["task_id"], e)
        return {
            "task_id": task["task_id"],
            "passed": False,
            "tokens": 0,
            "iterations": 0,
            "error": str(e),
        }
    elapsed = time.time() - t0

    # Final correctness via CodeEvaluator (binary pass/fail).
    evaluator = CodeEvaluator()
    eval_result = evaluator.evaluate(run_result.final_code, task["test_code"])

    return {
        "task_id": task["task_id"],
        "passed": bool(eval_result.passed),
        "tokens": run_result.total_tokens,
        "iterations": run_result.num_iterations,
        "stopped_reason": run_result.stopped_reason,
        "propose_cap": propose_cap,
        "review_cap": review_cap,
        "wall_time_s": round(elapsed, 1),
    }


def run_sweep(tasks: list[dict]) -> list[dict]:
    cells = []
    for label, (b1, b2, b3) in ALLOCATIONS:
        logger.info("=== ALLOCATION %s  (b1=%.2f, b2=%.2f, b3=%.2f) ===", label, b1, b2, b3)
        per_task_results = []
        for i, task in enumerate(tasks):
            logger.info("  [%s] task %d/%d %s", label, i + 1, len(tasks), task["task_id"])
            res = run_task(task, b1, b2, b3)
            logger.info(
                "    -> %s  tokens=%d  iters=%d  reason=%s",
                "PASS" if res["passed"] else "FAIL",
                res["tokens"],
                res["iterations"],
                res.get("stopped_reason", "?"),
            )
            per_task_results.append(res)

        passes = sum(1 for r in per_task_results if r["passed"])
        mean_tokens = sum(r["tokens"] for r in per_task_results) / max(1, len(per_task_results))
        mean_iters = sum(r["iterations"] for r in per_task_results) / max(1, len(per_task_results))

        cells.append({
            "allocation_label": label,
            "b1_propose": b1,
            "b2_execute": b2,
            "b3_review": b3,
            "n_tasks": len(per_task_results),
            "n_passed": passes,
            "pass_rate": passes / max(1, len(per_task_results)),
            "mean_tokens": round(mean_tokens, 1),
            "mean_iterations": round(mean_iters, 2),
            "propose_cap": per_task_results[0]["propose_cap"] if per_task_results else None,
            "review_cap": per_task_results[0]["review_cap"] if per_task_results else None,
            "per_task": per_task_results,
        })
        logger.info(
            "  ==> %s pass=%d/%d (%.0f%%) mean_tokens=%.0f mean_iters=%.1f",
            label, passes, len(per_task_results),
            100 * passes / max(1, len(per_task_results)),
            mean_tokens, mean_iters,
        )

        # Save partial results after each allocation in case of interruption.
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "config": {
                    "total_budget": TOTAL_BUDGET,
                    "max_iterations": MAX_ITERATIONS,
                    "model": "claude-sonnet-4-20250514",
                    "temperature": 0.0,
                    "use_thinking": False,
                    "n_tasks": len(tasks),
                    "note": (
                        "b1 and b3 control per-call max_tokens caps on "
                        "proposer/reviewer (scaled by B/max_iter). b2 is "
                        "reserved budget — slack for additional iterations "
                        "since the execute phase consumes 0 LLM tokens "
                        "(pytest subprocess). Total cumulative tokens "
                        "capped at B by BudgetTracker."
                    ),
                },
                "cells": cells,
            }, f, indent=2)
    return cells


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)
    logger.info("Loaded %d tasks from %s", len(tasks), TASKS_PATH)

    cells = run_sweep(tasks)

    # Final summary table to stdout
    print()
    print("=" * 88)
    print(f"ALLOCATION SWEEP  B={TOTAL_BUDGET} tokens, max_iter={MAX_ITERATIONS}, n_tasks={len(tasks)}")
    print("=" * 88)
    print(f"{'Label':<22} {'b1':>5} {'b2':>5} {'b3':>5} {'Pass':>9} {'MeanTok':>9} {'Iters':>6}")
    print("-" * 88)
    for c in cells:
        print(
            f"{c['allocation_label']:<22} "
            f"{c['b1_propose']:>5.2f} {c['b2_execute']:>5.2f} {c['b3_review']:>5.2f} "
            f"{c['n_passed']:>2}/{c['n_tasks']:<2} ({100*c['pass_rate']:>4.1f}%) "
            f"{c['mean_tokens']:>9.0f} {c['mean_iterations']:>6.2f}"
        )
    print("=" * 88)
    print(f"\nResults saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
