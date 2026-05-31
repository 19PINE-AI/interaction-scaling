"""Collect Stage A/B/C raw traces for Phase 4 autonomous-review distillation.

For each task in `code_tasks_v2.json`, this builds a raw trace of:
  Stage A — a first-pass draft (buggy_code or fixed_code depending on stratum)
  Stage B — real execution of that draft against the test_code (ground-truth obs)
  Stage C — a revision (fixed_code) with a second real execution, if Stage A failed

Sonnet 4.6 is called once per task to produce the assistant-voice glue text
(preambles, review/diagnosis sentences, confirm phrases) as a structured JSON.
The actual code and tool_result content is deterministic and executed locally.

Stratum distribution (configurable): 30% "no_revise" (draft passes on first try
and Stage B is just a verification) / 70% "one_revise" (draft fails, Stage C fixes).

Output: `data/training/review_traces_v2.json` — a list of dicts with fields:
    task_id, stratum, draft_code, draft_exec, fixed_code, fixed_exec,
    voice: {preamble, before_exec, diagnosis, revision_preamble, confirm}

Usage:
    python -m src.training.collect_review_traces_v2 \
        --tasks data/training/code_tasks_v2.json \
        --output data/training/review_traces_v2.json
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2500

VOICE_SYSTEM = """You are producing five short assistant-voice text pieces that represent five successive moments in an engineer's internal monologue while solving a coding task. Each piece captures a DIFFERENT point in time — the engineer's knowledge state shifts between them.

Timeline of moments:

  [1] PREAMBLE — Engineer has just read the task and is about to write their first attempt. They are CONFIDENT. They DO NOT yet know that any bug exists. They have not executed anything. They must NOT describe this as a 'fix' or reference any bug, because from their perspective this is just a normal implementation. Phrases like 'the fix is straightforward', 'the bug is in X', 'the issue is Y' are FORBIDDEN here — the engineer does not yet know of any problem.

  [2] BEFORE_EXEC — They have just written the code. Still confident. Preparing to run the tests as a sanity check before submitting.

  [3] DIAGNOSIS — Only populated if the execution FAILED. They have just seen the test failure. They read the stderr, identify the specific flaw in their draft, and articulate it. First-person reaction ('I see that...', 'The issue is...'). If the draft passed, write an empty string.

  [4] REVISION_PREAMBLE — Only populated if draft failed. They are about to write the corrected version. A brief lead-in ('Here is the fix:', 'Let me apply the correction:'). Do not include code. If draft passed, write an empty string.

  [5] CONFIRM — After the final (passing) tests, a short submit line.

Reply with ONE JSON object (no prose, no fences), with these fields — all short, first-person, natural assistant voice, NO headings, NO lists, NO markdown:

{
  "preamble": "<1-3 sentences describing the engineer's approach to the task as if they are implementing it fresh — no awareness of any bug. End with a colon or phrase like 'Here's my implementation:'. Do NOT include the code block.>",
  "before_exec": "<1-2 sentences preparing to run the tests to verify correctness.>",
  "diagnosis": "<2-4 sentences diagnosing the specific bug after seeing the failure, or empty string if tests passed.>",
  "revision_preamble": "<1-2 sentences introducing the corrected version, or empty string if tests passed.>",
  "confirm": "<1-2 sentences submitting the final passing solution.>"
}

Hard rules:
- All text is first-person assistant voice, no headings, no bullet lists, no markdown.
- Never quote code blocks inside these fields — the stitcher inserts code separately.
- Never reference 'stage A' / 'stage B' / 'the reviewer' / 'the draft' / 'the revision' — the student reads this as one continuous voice.
- If draft_passed=true: set diagnosis=\"\" AND revision_preamble=\"\". Confirm can be 'No issues — submitting' or similar.
- The PREAMBLE must NEVER reference a bug, fix, issue, correction, or mistake. Those words are only allowed in diagnosis/revision_preamble/confirm-after-fix.
- Return only the JSON object."""


def build_user_turn(task: dict, draft_code: str, draft_passed: bool,
                    draft_stderr: str, draft_error: str | None) -> list[dict]:
    # Keep stderr short enough to be informative but not overflow context
    stderr_tail = "\n".join((draft_stderr or "").strip().splitlines()[-25:])
    parts = [
        f"TASK DESCRIPTION:\n{task['description']}",
        f"DRAFT SOLUTION:\n```python\n{draft_code}\n```",
        f"DRAFT EXECUTION RESULT: {'PASSED' if draft_passed else 'FAILED'}",
    ]
    if not draft_passed:
        parts.append(f"DRAFT ERROR: {draft_error}")
        parts.append(f"DRAFT STDERR TAIL:\n{stderr_tail}")
        parts.append(f"REFERENCE FIX (what the revision will contain):\n```python\n{task['fixed_code']}\n```")
    else:
        parts.append("No revision needed — tests passed on the draft.")

    text = "\n\n".join(parts) + "\n\nReturn the JSON object."
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def call_voice(client: anthropic.Anthropic, task: dict, draft_code: str,
               draft_passed: bool, draft_stderr: str,
               draft_error: str | None) -> dict | None:
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": VOICE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=build_user_turn(task, draft_code, draft_passed, draft_stderr, draft_error),
        )
    except Exception as e:
        logger.warning("voice call failed for %s: %s", task["task_id"], e)
        return None

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        voice = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("voice JSON decode failed for %s: %s", task["task_id"], e)
        return None

    required = {"preamble", "before_exec", "diagnosis", "revision_preamble", "confirm"}
    if not required.issubset(voice.keys()):
        logger.warning("voice missing fields for %s", task["task_id"])
        return None
    return voice


def collect_one(task: dict, stratum: str, evaluator: CodeEvaluator,
                client: anthropic.Anthropic) -> dict | None:
    """Build a raw trace for one task under the chosen stratum."""
    if stratum == "no_revise":
        draft = task["fixed_code"]
    elif stratum == "one_revise":
        draft = task["buggy_code"]
    else:
        raise ValueError(stratum)

    # Stage B real execution
    draft_exec = evaluator.evaluate(draft, task["test_code"], timeout=10)
    draft_passed = draft_exec.passed

    # Stratum sanity — if it doesn't match expected outcome, flip stratum to match reality.
    effective_stratum = "no_revise" if draft_passed else "one_revise"

    fixed_exec = None
    if not draft_passed:
        fixed_exec = evaluator.evaluate(task["fixed_code"], task["test_code"], timeout=10)
        if not fixed_exec.passed:
            logger.warning("fixed_code also fails for %s — skipping", task["task_id"])
            return None

    voice = call_voice(client, task, draft, draft_passed,
                       draft_exec.stderr, draft_exec.error_message)
    if voice is None:
        return None

    return {
        "task_id": task["task_id"],
        "bug_class": task.get("bug_class", "?"),
        "stratum": effective_stratum,
        "description": task["description"],
        "test_code": task["test_code"],
        "draft_code": draft,
        "draft_exec": {
            "passed": draft_exec.passed,
            "error": draft_exec.error_message,
            "stdout": (draft_exec.stdout or "")[-1500:],
            "stderr": (draft_exec.stderr or "")[-1500:],
        },
        "fixed_code": task["fixed_code"],
        "fixed_exec": None if fixed_exec is None else {
            "passed": fixed_exec.passed,
            "error": fixed_exec.error_message,
            "stdout": (fixed_exec.stdout or "")[-1500:],
            "stderr": (fixed_exec.stderr or "")[-1500:],
        },
        "voice": voice,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/training/code_tasks_v2.json")
    ap.add_argument("--output", default="data/training/review_traces_v2.json")
    ap.add_argument("--log", default="logs/collect_review_traces_v2.log")
    ap.add_argument("--no-revise-frac", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    rng = random.Random(args.seed)
    # Assign strata up-front so the overall mix is controlled.
    strata = []
    n_no_revise = int(round(len(tasks) * args.no_revise_frac))
    strata = ["no_revise"] * n_no_revise + ["one_revise"] * (len(tasks) - n_no_revise)
    rng.shuffle(strata)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        collected = json.loads(out_path.read_text())
        done_ids = {t["task_id"] for t in collected}
        logger.info("Resuming: %d traces already collected", len(collected))
    else:
        collected = []
        done_ids = set()

    client = anthropic.Anthropic()
    evaluator = CodeEvaluator()
    lock = threading.Lock()

    def work(task, stratum):
        if task["task_id"] in done_ids:
            return None
        trace = collect_one(task, stratum, evaluator, client)
        if trace is None:
            return None
        with lock:
            collected.append(trace)
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("COLLECTED [%d/%d] %s stratum=%s",
                        len(collected), len(tasks), trace["task_id"], trace["stratum"])
        return trace

    # Warm cache
    first_idx = 0
    while first_idx < len(tasks) and tasks[first_idx]["task_id"] in done_ids:
        first_idx += 1
    if first_idx < len(tasks):
        logger.info("Warming cache with one sequential call...")
        work(tasks[first_idx], strata[first_idx])
        first_idx += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i in range(first_idx, len(tasks)):
            futures.append(ex.submit(work, tasks[i], strata[i]))
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                logger.exception("worker failed")

    out_path.write_text(json.dumps(collected, indent=2))
    actual_no_revise = sum(1 for t in collected if t["stratum"] == "no_revise")
    logger.info("Done: %d traces (%d no_revise / %d one_revise) -> %s",
                len(collected), actual_no_revise,
                len(collected) - actual_no_revise, out_path)


if __name__ == "__main__":
    main()
