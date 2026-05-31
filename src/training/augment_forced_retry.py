"""Augment v3 traces with forced-retry trajectories (two_revise stratum).

Sonnet is too capable to produce natural multi-step retry trajectories — in a
204-task run, essentially every fix succeeds on the first try. But the student
needs to see >=3-call trajectories to generalize "fix failed → diagnose again
→ try differently" behavior. We synthesize these by asking Sonnet FIRST for
a plausible-but-partial fix (attends to one visible symptom of the bug but
misses a subtler aspect), then after execution shows it fails, ask for the
real fix. Both calls still operate WITHOUT knowledge of the reference fix —
Sonnet derives from stderr at each step.

Output: new trace records (stratum=two_revise) to be appended to the natural
v3 traces. Uses buggy_code tasks not already covered by an existing two_revise.

Usage:
    python -m src.training.augment_forced_retry \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --existing data/training/review_traces_v3.json \\
        --output data/training/review_traces_v3_forced.json \\
        --n 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.evaluation.code_eval import CodeEvaluator
from src.training.collect_review_traces_v3 import (
    MODEL, MAX_TOKENS_VOICE, MAX_TOKENS_FIX,
    VOICE_SYSTEM, FIX_SYSTEM,
    build_voice_user_turn, build_fix_user_turn,
    call_json, _pack_exec,
)

logger = logging.getLogger(__name__)


PARTIAL_FIX_SYSTEM = """You are producing training data that simulates a realistic multi-step debugging trajectory. Your task: write a first-attempt "fix" that IS PLAUSIBLE but DOES NOT actually pass the test harness — because in the next step the engineer will see it still fails, and produce the real fix.

This is essential for the training signal: the student must learn what to do when a fix fails. If every fix passes on the first try, the student never sees that pattern.

You have:
  - the task description
  - the test harness
  - the current (buggy) implementation
  - the traceback from running the tests

You must produce a first-attempt fix that:
  1. ADDRESSES a visible symptom from the traceback (it must look plausible, not random).
  2. STILL FAILS the test harness — it must miss another aspect of the bug, or introduce a new related issue, or partially fix the wrong line, etc.
  3. Is a realistic mistake an engineer could make — NOT deliberate sabotage or obviously-wrong code.

Common realistic-partial-fix patterns:
  - Fix the boundary condition in one branch but not another symmetric branch
  - Handle the specific example in the test but not the general case
  - Fix the primary symptom but leave a data-flow bug that manifests on different input
  - Swap the wrong pair of variables / change the wrong operator
  - Over-correct: introduce a NEW off-by-one while fixing the original

Reply with ONE JSON object (no prose, no fences):

{
  "diagnosis": "<2-4 sentences, first-person, a plausible but INCOMPLETE reading of the bug from the traceback.>",
  "revision_preamble": "<1 sentence lead-in, e.g. 'Let me try this:'.>",
  "fixed_code": "<complete Python source of your first-attempt fix. The tests MUST fail when this code is run. Critical: do NOT produce the correct complete fix. If you cannot think of a plausible partial fix, it is acceptable to make an off-by-one or swap-operator mistake.>"
}

Return only the JSON object, no code fences. `fixed_code` must fail at least one test assertion.
"""


def collect_forced(task: dict, evaluator: CodeEvaluator,
                   client=None) -> dict | None:
    """Build a forced two_revise trace.

    Stages:
      1. Draft = buggy_code. Execute (fail expected).
      2. Voice call for preamble/before_exec/confirm.
      3. First fix call with PARTIAL_FIX_SYSTEM — expect failure.
      4. Second fix call with normal FIX_SYSTEM given the new stderr.
      5. If the second fix also fails, retry once more; drop on triple failure.
    """
    task_id = task["task_id"]
    draft = task["buggy_code"]
    draft_exec = evaluator.evaluate(draft, task["test_code"], timeout=10)
    if draft_exec.passed:
        # buggy_code accidentally passes the tests? Skip.
        logger.warning("%s: buggy_code passes tests, unsuitable for forced retry", task_id)
        return None

    voice = call_json(
        client, VOICE_SYSTEM,
        build_voice_user_turn(task, draft),
        MAX_TOKENS_VOICE, task_id, kind="voice",
    )
    if voice is None or not {"preamble", "before_exec", "confirm"}.issubset(voice.keys()):
        return None

    attempts: list[dict] = []

    # Attempt 1: partial fix. Retry up to 2 times if it accidentally passes.
    partial = None
    code_1 = None
    exec_1 = None
    for retry in range(3):
        boost = ""
        if retry > 0:
            boost = (
                f"\n\nPREVIOUS ATTEMPT (retry {retry}): your 'partial fix' "
                f"actually passed the tests, which makes it unsuitable for "
                f"training data. You MUST introduce a subtle failure. "
                f"Try a different approach: fix only one of two related issues, "
                f"or introduce an off-by-one, or swap operands incorrectly. "
                f"The code must FAIL at least one assertion in the test harness."
            )
        partial = call_json(
            client, PARTIAL_FIX_SYSTEM + boost,
            build_fix_user_turn(task, draft, draft_exec.stderr or "", []),
            MAX_TOKENS_FIX, task_id, kind=f"partial-fix#{retry+1}",
        )
        if partial is None or not {"diagnosis","revision_preamble","fixed_code"}.issubset(partial.keys()):
            return None
        code_1 = _clean_code(str(partial["fixed_code"]))
        exec_1 = evaluator.evaluate(code_1, task["test_code"], timeout=10)
        if not exec_1.passed:
            break

    attempts.append({
        "diagnosis": partial["diagnosis"].strip(),
        "revision_preamble": partial["revision_preamble"].strip(),
        "code": code_1,
        "exec": _pack_exec(exec_1),
    })

    if exec_1.passed:
        # Sonnet refused to produce a failing fix after 3 tries. Save as
        # one_revise (already a valid trajectory) rather than drop.
        logger.info("%s: partial fix passed after 3 retries — saving as one_revise",
                    task_id)
        return _build_trace(task, draft, draft_exec, voice, attempts, "one_revise")

    # Attempt 2: real fix
    prior = [{"code": attempts[0]["code"],
              "stderr_tail": "\n".join((exec_1.stderr or "").strip().splitlines()[-25:])}]
    real = call_json(
        client, FIX_SYSTEM.replace(
            "{PRIOR_ATTEMPTS_CLAUSE}",
            "You have already made a previous attempt that also failed. "
            "Study it carefully — do NOT repeat the same mistake. The correct "
            "fix must address both the original bug and avoid the pitfalls of "
            "your prior attempt.",
        ),
        build_fix_user_turn(task, code_1, exec_1.stderr or "", prior),
        MAX_TOKENS_FIX, task_id, kind="real-fix",
    )
    if real is None or not {"diagnosis","revision_preamble","fixed_code"}.issubset(real.keys()):
        return None

    code_2 = _clean_code(str(real["fixed_code"]))
    exec_2 = evaluator.evaluate(code_2, task["test_code"], timeout=10)
    attempts.append({
        "diagnosis": real["diagnosis"].strip(),
        "revision_preamble": real["revision_preamble"].strip(),
        "code": code_2,
        "exec": _pack_exec(exec_2),
    })

    if exec_2.passed:
        return _build_trace(task, draft, draft_exec, voice, attempts, "two_revise")

    # Attempt 3 (rare): try one more time
    prior.append({"code": attempts[1]["code"],
                  "stderr_tail": "\n".join((exec_2.stderr or "").strip().splitlines()[-25:])})
    third = call_json(
        client, FIX_SYSTEM.replace(
            "{PRIOR_ATTEMPTS_CLAUSE}",
            "You have already made two previous attempts that both failed. "
            "Study them carefully. The correct fix must avoid both pitfalls.",
        ),
        build_fix_user_turn(task, code_2, exec_2.stderr or "", prior),
        MAX_TOKENS_FIX, task_id, kind="third-fix",
    )
    if third is None or not {"diagnosis","revision_preamble","fixed_code"}.issubset(third.keys()):
        return None

    code_3 = _clean_code(str(third["fixed_code"]))
    exec_3 = evaluator.evaluate(code_3, task["test_code"], timeout=10)
    attempts.append({
        "diagnosis": third["diagnosis"].strip(),
        "revision_preamble": third["revision_preamble"].strip(),
        "code": code_3,
        "exec": _pack_exec(exec_3),
    })

    if exec_3.passed:
        return _build_trace(task, draft, draft_exec, voice, attempts, "three_revise")

    logger.warning("%s: all 3 forced attempts failed — dropping", task_id)
    return None


def _clean_code(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


def _build_trace(task, draft, draft_exec, voice, attempts, stratum):
    return {
        "task_id": task["task_id"],
        "bug_class": task.get("bug_class", "?"),
        "description": task["description"],
        "test_code": task["test_code"],
        "stratum": stratum,
        "draft_code": draft,
        "draft_exec": _pack_exec(draft_exec),
        "attempts": attempts,
        "voice": voice,
        "source": "forced",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/training/code_tasks_v2_scrubbed.json")
    ap.add_argument("--existing", default="data/training/review_traces_v3.json")
    ap.add_argument("--output", default="data/training/review_traces_v3_forced.json")
    ap.add_argument("--log", default="logs/augment_forced_retry.log")
    ap.add_argument("--n", type=int, default=40, help="how many forced tasks to attempt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    all_tasks = json.loads(Path(args.tasks).read_text())
    existing = json.loads(Path(args.existing).read_text()) if Path(args.existing).exists() else []
    existing_ids = {t["task_id"] for t in existing}

    # Candidates: tasks where the natural trace was one_revise (the model
    # was still able to produce the fix, so the task is viable) — we pick
    # from this pool since non-buggy-draft tasks (no_revise) aren't useful.
    good_ids = {t["task_id"] for t in existing if t["stratum"] == "one_revise"}
    candidate_tasks = [t for t in all_tasks if t["task_id"] in good_ids]

    rng = random.Random(args.seed)
    rng.shuffle(candidate_tasks)
    candidate_tasks = candidate_tasks[: args.n]
    logger.info("Selected %d candidate tasks for forced retry", len(candidate_tasks))

    out_path = Path(args.output)
    if out_path.exists():
        collected = json.loads(out_path.read_text())
        done_ids = {t["task_id"] for t in collected}
        logger.info("Resuming: %d already forced", len(collected))
    else:
        collected = []
        done_ids = set()

    client = None
    evaluator = CodeEvaluator()
    lock = threading.Lock()

    def work(task):
        if task["task_id"] in done_ids:
            return None
        trace = collect_forced(task, evaluator, client)
        if trace is None:
            return None
        with lock:
            collected.append(trace)
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("FORCED [%d] %s stratum=%s attempts=%d",
                        len(collected), trace["task_id"],
                        trace["stratum"], len(trace["attempts"]))
        return trace

    # Warm cache sequentially
    first_idx = 0
    while first_idx < len(candidate_tasks) and candidate_tasks[first_idx]["task_id"] in done_ids:
        first_idx += 1
    if first_idx < len(candidate_tasks):
        logger.info("Warming cache with one sequential call...")
        work(candidate_tasks[first_idx])
        first_idx += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, t) for t in candidate_tasks[first_idx:]]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                logger.exception("worker failed")

    out_path.write_text(json.dumps(collected, indent=2))
    from collections import Counter
    strata = Counter(t["stratum"] for t in collected)
    logger.info("Done: %d forced traces. Strata: %s", len(collected), dict(strata))


if __name__ == "__main__":
    main()
