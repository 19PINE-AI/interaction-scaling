#!/usr/bin/env python3
"""Reasoning-only baseline for the 15 Phase 1 code tasks.

Pivot context: the paper wants to show that wrapping a frontier model in a
proposer-reviewer harness is a *new* compute axis distinct from
"just-think-longer" reasoning scaling. To support that claim we need a baseline
where the *same* model is allowed to spend a comparable token budget on
extended thinking (no tools, no execution, no review) before emitting code.

This script:
  - Loads the 15 hard code tasks from data/hard_benchmarks/code/code_tasks.json
  - Calls claude-sonnet-4-20250514 with extended thinking enabled
    (thinking.budget_tokens=8000, max_tokens=12000) — generous so the model is
    not artificially throttled. Actual tokens are reported per task for the
    matched-budget analysis.
  - Extracts the Python code, evaluates with the existing CodeEvaluator
    (same harness as Phase 1).
  - Writes results/hard_benchmarks/code_reasoning_only.json.

Run:
    python scripts/run_reasoning_only_code.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Make repo root importable when run as a script
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
logger = logging.getLogger("reasoning_only")

MODEL_ID = "claude-sonnet-4-20250514"
THINKING_BUDGET = 8000
MAX_TOKENS = 12000  # must be > THINKING_BUDGET; leaves ~4k for emitted answer

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct, efficient solutions.\n\n"
    "Rules:\n"
    "- Output ONLY the function definition with any needed imports\n"
    "- Handle ALL edge cases carefully\n"
    "- Pay attention to boundary conditions, empty inputs, and special values\n"
    "- Wrap your code in a ```python code block"
)

TASKS_PATH = REPO_ROOT / "data" / "hard_benchmarks" / "code" / "code_tasks.json"
OUT_PATH = REPO_ROOT / "results" / "hard_benchmarks" / "code_reasoning_only.json"


def run_task(client: anthropic.Anthropic, task: dict, evaluator: CodeEvaluator) -> dict:
    task_id = task["task_id"]
    description = task["description"]
    test_code = task["test_code"]

    logger.info("[%s] calling %s with extended thinking", task_id, MODEL_ID)
    t0 = time.time()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
        temperature=1.0,  # required by extended thinking
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}],
    )
    elapsed = time.time() - t0

    thinking_text = ""
    answer_text = ""
    for block in resp.content:
        if block.type == "thinking":
            thinking_text += block.thinking
        elif block.type == "text":
            answer_text += block.text

    code_emitted = extract_code(answer_text, "python")

    # Token accounting. Anthropic reports `output_tokens` as the *total* generated
    # tokens (thinking + visible). We approximate thinking_tokens by counting
    # characters in the thinking blocks and dividing by ~4 chars/token; the
    # ground-truth split is not directly exposed for sonnet-4-20250514. The
    # combined `total_tokens` figure matches Phase 1's accounting.
    usage = resp.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens  # includes thinking
    total_tokens = input_tokens + output_tokens

    # Best-effort split for reporting
    approx_thinking_tokens = max(0, len(thinking_text) // 4)
    approx_visible_tokens = max(0, output_tokens - approx_thinking_tokens)

    # Score with the same harness as Phase 1
    eval_result = evaluator.evaluate(code_emitted, test_code)
    passed = bool(eval_result.passed)

    record = {
        "task_id": task_id,
        "model": "claude-sonnet-4-thinking-reasoning-only",
        "model_id": MODEL_ID,
        "passed": passed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "approx_thinking_tokens": approx_thinking_tokens,
        "approx_visible_output_tokens": approx_visible_tokens,
        "thinking_budget": THINKING_BUDGET,
        "max_tokens": MAX_TOKENS,
        "stop_reason": resp.stop_reason,
        "wall_time_seconds": round(elapsed, 2),
        "code_emitted": code_emitted,
        "thinking_text": thinking_text,
        "answer_text": answer_text,
        "error_message": eval_result.error_message,
        "stderr_excerpt": eval_result.stderr[:2000] if eval_result.stderr else "",
    }
    logger.info(
        "[%s] passed=%s total_tokens=%d (in=%d, out=%d, ~think=%d) stop=%s elapsed=%.1fs",
        task_id, passed, total_tokens, input_tokens, output_tokens,
        approx_thinking_tokens, resp.stop_reason, elapsed,
    )
    if not passed and eval_result.error_message:
        logger.info("[%s]   error: %s", task_id, eval_result.error_message)
    return record


def main():
    if not TASKS_PATH.exists():
        logger.error("Tasks file not found: %s", TASKS_PATH)
        sys.exit(1)

    with TASKS_PATH.open() as f:
        tasks = json.load(f)
    logger.info("Loaded %d tasks", len(tasks))

    client = anthropic.Anthropic()
    evaluator = CodeEvaluator()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for task in tasks:
        try:
            rec = run_task(client, task, evaluator)
        except anthropic.APIError as e:
            logger.exception("[%s] API error", task["task_id"])
            rec = {
                "task_id": task["task_id"],
                "model": "claude-sonnet-4-thinking-reasoning-only",
                "model_id": MODEL_ID,
                "passed": False,
                "error": f"APIError: {e}",
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] unexpected error", task["task_id"])
            rec = {
                "task_id": task["task_id"],
                "model": "claude-sonnet-4-thinking-reasoning-only",
                "model_id": MODEL_ID,
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(rec)
        # Incremental save so a crash doesn't lose everything
        with OUT_PATH.open("w") as f:
            json.dump(results, f, indent=2)

    # Summary
    n_pass = sum(1 for r in results if r.get("passed"))
    avg_tokens = sum(r.get("total_tokens", 0) for r in results) / max(1, len(results))
    logger.info("=" * 60)
    logger.info("Reasoning-only baseline: %d/%d passed (%.1f%%)",
                n_pass, len(results), 100 * n_pass / max(1, len(results)))
    logger.info("Avg total tokens: %.0f", avg_tokens)
    logger.info("Wrote: %s", OUT_PATH)


if __name__ == "__main__":
    main()
