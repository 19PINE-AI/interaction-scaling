"""Augment training data with forced three_revise (4-call) trajectories.

Mirrors augment_forced_retry.py but forces BOTH attempts 1 and 2 to be
deliberately-partial fixes. Attempt 3 is the real fix produced by FIX_SYSTEM
with history of the two failed attempts.

Sonnet is asked for two DIFFERENT partial-fix strategies across attempt 1 and
attempt 2 to avoid the "same mistake twice" failure mode that makes the
trajectory look unrealistic.

Usage:
    python -m src.training.augment_three_revise \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --existing data/training/review_traces_v3.json \\
        --output data/training/review_traces_v3_three.json \\
        --n 120
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
    MAX_TOKENS_VOICE, MAX_TOKENS_FIX,
    VOICE_SYSTEM, FIX_SYSTEM,
    build_voice_user_turn, build_fix_user_turn,
    call_json, _pack_exec,
)

logger = logging.getLogger(__name__)


PARTIAL_FIX_A = """You are producing training data simulating a multi-step debugging trajectory. Your task: write a first-attempt "fix" that IS PLAUSIBLE but DOES NOT actually pass the test harness.

This is attempt 1 of an eventual 3-attempt sequence. Your fix must address a visible symptom from the traceback (look plausible, not random), but leave the full bug unresolved — so a second attempt is needed.

You have:
  - the task description
  - the test harness
  - the current (buggy) implementation
  - the traceback from running the tests

Common realistic-partial-fix patterns (attempt 1 style: attack the most obvious symptom):
  - Fix the boundary condition in one branch but not another symmetric branch
  - Handle the specific example in the test but not the general case
  - Swap the wrong pair of variables / change the wrong operator
  - Add a guard that handles the literal traceback value but not related values

Reply with ONE JSON object (no prose, no fences):

{
  "diagnosis": "<2-4 sentences, first-person, a PLAUSIBLE but INCOMPLETE reading of the bug.>",
  "revision_preamble": "<1 sentence lead-in, e.g. 'Let me try this:'.>",
  "fixed_code": "<complete Python source of your first-attempt fix. The tests MUST FAIL when this code is run. Do NOT produce the correct complete fix.>"
}

Return only the JSON object, no code fences. `fixed_code` MUST fail at least one test assertion.
"""


PARTIAL_FIX_B = """You are producing training data simulating a multi-step debugging trajectory. This is attempt 2 of 3. Your previous attempt (attempt 1) was a partial fix that didn't fully resolve the bug.

Produce a SECOND-attempt fix that:
  1. Recognizes that attempt 1 was insufficient and reacts to the new traceback.
  2. Takes a DIFFERENT approach from attempt 1 — don't make the same kind of mistake twice.
  3. STILL DOES NOT fully pass the test harness — introduces a new plausible-but-incomplete correction so that attempt 3 is still needed.
  4. Is a realistic second-pass mistake (e.g., over-corrected, or fixed the main branch but broke a related one, or mixed up a different pair of operators).

You have:
  - the task description
  - the test harness
  - your attempt 1 code (failed)
  - attempt 1 traceback
  - the CURRENT implementation (= attempt 1) and its traceback

Reply with ONE JSON object (no prose, no fences):

{
  "diagnosis": "<2-4 sentences, first-person, partial but honest reading — note attempt 1 was incomplete, propose a new angle.>",
  "revision_preamble": "<1 sentence lead-in.>",
  "fixed_code": "<complete Python source of your second-attempt fix. It MUST FAIL when run. Take a different angle from attempt 1 — but still incomplete. Avoid accidentally producing the correct answer.>"
}

Return only the JSON object, no code fences. `fixed_code` MUST fail at least one test assertion and MUST differ materially from attempt 1.
"""


def _clean_code(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


def _try_partial(client, system_prompt: str, task: dict, current_code: str,
                 stderr: str, prior: list[dict], evaluator: CodeEvaluator,
                 task_id: str, kind: str, max_retries: int = 3):
    """Call Sonnet for a partial fix. Retry with boost if it accidentally passes.

    Returns (partial_json, code, exec_result) or (None, None, None) on give-up.
    """
    last = (None, None, None)
    for retry in range(max_retries):
        boost = ""
        if retry > 0:
            boost = (
                f"\n\nPREVIOUS ATTEMPT (retry {retry}): your output accidentally "
                f"passed the tests, which is not usable for this training step. "
                f"You MUST introduce a subtle failure. Fix only ONE of several "
                f"issues, or introduce a related off-by-one, or swap operands. "
                f"The code MUST FAIL at least one assertion when run."
            )
        partial = call_json(
            client, system_prompt + boost,
            build_fix_user_turn(task, current_code, stderr, prior),
            MAX_TOKENS_FIX, task_id, kind=f"{kind}#{retry+1}",
        )
        if partial is None:
            continue
        if not {"diagnosis", "revision_preamble", "fixed_code"}.issubset(partial.keys()):
            continue
        code = _clean_code(str(partial["fixed_code"]))
        exec_r = evaluator.evaluate(code, task["test_code"], timeout=10)
        last = (partial, code, exec_r)
        if not exec_r.passed:
            return partial, code, exec_r
    return last  # give up and return whatever last was (possibly passing)


def collect_three(task: dict, evaluator: CodeEvaluator,
                  client=None) -> dict | None:
    task_id = task["task_id"]
    draft = task["buggy_code"]
    draft_exec = evaluator.evaluate(draft, task["test_code"], timeout=10)
    if draft_exec.passed:
        logger.warning("%s: buggy_code passes tests, unsuitable", task_id)
        return None

    voice = call_json(
        client, VOICE_SYSTEM,
        build_voice_user_turn(task, draft),
        MAX_TOKENS_VOICE, task_id, kind="voice",
    )
    if voice is None or not {"preamble", "before_exec", "confirm"}.issubset(voice.keys()):
        return None

    attempts: list[dict] = []

    # --- Attempt 1: partial fix (style A) ---
    partial_1, code_1, exec_1 = _try_partial(
        client, PARTIAL_FIX_A, task, draft, draft_exec.stderr or "",
        [], evaluator, task_id, kind="partial-A",
    )
    if partial_1 is None or code_1 is None:
        return None
    attempts.append({
        "diagnosis": partial_1["diagnosis"].strip(),
        "revision_preamble": partial_1["revision_preamble"].strip(),
        "code": code_1,
        "exec": _pack_exec(exec_1),
    })
    if exec_1.passed:
        # Couldn't force a failure — salvage as one_revise.
        logger.info("%s: partial-A passed; saving as one_revise", task_id)
        return _build_trace(task, draft, draft_exec, voice, attempts, "one_revise")

    # --- Attempt 2: partial fix (style B) ---
    prior_2 = [{"code": code_1,
                "stderr_tail": "\n".join((exec_1.stderr or "").strip().splitlines()[-25:])}]
    partial_2, code_2, exec_2 = _try_partial(
        client, PARTIAL_FIX_B, task, code_1, exec_1.stderr or "",
        prior_2, evaluator, task_id, kind="partial-B",
    )
    if partial_2 is None or code_2 is None:
        return None
    attempts.append({
        "diagnosis": partial_2["diagnosis"].strip(),
        "revision_preamble": partial_2["revision_preamble"].strip(),
        "code": code_2,
        "exec": _pack_exec(exec_2),
    })
    if exec_2.passed:
        # Attempt 2 accidentally passed — still a valid two_revise trace.
        logger.info("%s: partial-B passed; saving as two_revise", task_id)
        return _build_trace(task, draft, draft_exec, voice, attempts, "two_revise")

    # --- Attempt 3: real fix ---
    prior_3 = [
        {"code": code_1,
         "stderr_tail": "\n".join((exec_1.stderr or "").strip().splitlines()[-25:])},
        {"code": code_2,
         "stderr_tail": "\n".join((exec_2.stderr or "").strip().splitlines()[-25:])},
    ]
    real = call_json(
        client,
        FIX_SYSTEM.replace(
            "{PRIOR_ATTEMPTS_CLAUSE}",
            "You have already made two previous attempts that both failed. "
            "Study them carefully — do NOT repeat either mistake. The correct "
            "fix must address the original bug and avoid the pitfalls of both "
            "prior attempts.",
        ),
        build_fix_user_turn(task, code_2, exec_2.stderr or "", prior_3),
        MAX_TOKENS_FIX, task_id, kind="real-fix",
    )
    if real is None or not {"diagnosis", "revision_preamble", "fixed_code"}.issubset(real.keys()):
        return None

    code_3 = _clean_code(str(real["fixed_code"]))
    exec_3 = evaluator.evaluate(code_3, task["test_code"], timeout=10)
    attempts.append({
        "diagnosis": real["diagnosis"].strip(),
        "revision_preamble": real["revision_preamble"].strip(),
        "code": code_3,
        "exec": _pack_exec(exec_3),
    })
    if not exec_3.passed:
        logger.warning("%s: real fix still failed — dropping trace", task_id)
        return None

    return _build_trace(task, draft, draft_exec, voice, attempts, "three_revise")


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
        "source": "forced-three",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/training/code_tasks_v2_scrubbed.json")
    ap.add_argument("--existing", default="data/training/review_traces_v3.json")
    ap.add_argument("--output", default="data/training/review_traces_v3_three.json")
    ap.add_argument("--log", default="logs/augment_three_revise.log")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=91)
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

    good_ids = {t["task_id"] for t in existing if t["stratum"] in ("one_revise", "two_revise")}
    candidate_tasks = [t for t in all_tasks if t["task_id"] in good_ids]

    rng = random.Random(args.seed)
    rng.shuffle(candidate_tasks)
    candidate_tasks = candidate_tasks[: args.n]
    logger.info("Selected %d candidate tasks for three_revise augmentation", len(candidate_tasks))

    out_path = Path(args.output)
    if out_path.exists():
        collected = json.loads(out_path.read_text())
        done_ids = {t["task_id"] for t in collected}
        logger.info("Resuming: %d already collected", len(collected))
    else:
        collected = []
        done_ids = set()

    client = None
    evaluator = CodeEvaluator()
    lock = threading.Lock()

    def work(task):
        if task["task_id"] in done_ids:
            return None
        trace = collect_three(task, evaluator, client)
        if trace is None:
            return None
        with lock:
            collected.append(trace)
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("THREE [%d] %s stratum=%s attempts=%d",
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
    logger.info("Done: %d traces. Strata: %s", len(collected), dict(strata))


if __name__ == "__main__":
    main()
