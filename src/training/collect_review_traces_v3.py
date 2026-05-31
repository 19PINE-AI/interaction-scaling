"""Collect variable-length review traces — v3 pipeline.

Fixes two structural flaws in v2:
  1. v2 gave Sonnet the reference fix when asking for the diagnosis voice text.
     v3 hides the fix entirely — Sonnet sees only the task, the buggy draft,
     and the stderr, and must derive BOTH the diagnosis AND the corrected code.
  2. v2 only produced 1-call (no_revise) or 2-call (one_revise) trajectories,
     so the student never saw a "fix failed → try again" pattern. v3 lets
     Sonnet retry up to 2 more times on failure, producing natural 3- and
     4-call trajectories (two_revise, three_revise) when the first fix fails.

Output schema per trace:
    task_id, bug_class, description, test_code, stratum,
    draft_code, draft_exec,
    attempts: [
      {"diagnosis": ..., "revision_preamble": ..., "code": ..., "exec": {...}},
      ...
    ],
    voice: {preamble, before_exec, confirm}

The draft is always task["buggy_code"] for revise strata. A small fraction of
tasks use task["fixed_code"] as the draft (no_revise stratum) to keep the
"sometimes my draft works and I just verify it" pattern in distribution.

Usage:
    python -m src.training.collect_review_traces_v3 \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --output data/training/review_traces_v3.json
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

import httpx

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

# Teacher: Qwen3-235B via OpenRouter — same model family as student (Qwen3-8B),
# so distillation has minimal distribution mismatch.
MODEL = "qwen/qwen3-235b-a22b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEACHER_TEMPERATURE = 0.7
TEACHER_TIMEOUT = 300.0
MAX_TOKENS_VOICE = 1500
MAX_TOKENS_FIX = 3000
MAX_ATTEMPTS = 3  # up to 3 revision attempts (total 4 tool_calls including draft)

VOICE_SYSTEM = """You are producing three short assistant-voice text pieces.

The engineer has just written an implementation and is about to submit it. They DO NOT yet know whether it has any bug — they're going to verify with the test harness first.

Reply with ONE JSON object (no prose, no fences):

{
  "preamble": "<1-3 sentences: engineer states their approach and presents their implementation. No awareness of any bug. Words like 'fix', 'bug', 'issue', 'mistake', 'correction' are FORBIDDEN here — the engineer has not yet executed anything.>",
  "before_exec": "<1-2 sentences preparing to run the test harness.>",
  "confirm": "<1-2 sentences submitting the final passing solution. Short. E.g. 'All green — submitting.'>"
}

Hard rules:
- First-person assistant voice, no headings, no bullet lists, no markdown.
- Never quote code blocks in these fields.
- Never reference 'stage A' / 'the draft' / 'the reviewer'.
- Return only the JSON object."""


FIX_SYSTEM = """You are an engineer debugging. You just ran your tests and saw a failure. Identify what went wrong in YOUR code and produce a corrected version.

You have:
  - the task description
  - the test harness
  - your most recent implementation (which just failed)
  - the traceback from running the tests

{PRIOR_ATTEMPTS_CLAUSE}

Reply with ONE JSON object (no prose, no fences):

{
  "diagnosis": "<2-4 sentences, first-person, diagnosing the specific flaw in the current implementation based on the traceback. Be concrete — name the variable, line, or control-flow error. Don't just say 'the test failed'.>",
  "revision_preamble": "<1 sentence lead-in to the corrected version, like 'Here is the fix:' or 'Let me apply the correction:'.>",
  "fixed_code": "<the complete corrected Python source. Full function/class definitions, not a diff. Must include everything needed to make the test harness pass. Do NOT wrap in code fences.>"
}

Hard rules:
- `diagnosis` must only reference information visible in the buggy code + stderr. Do not claim certainty about things you can't see.
- `fixed_code` is raw Python source — no ```python fences, no commentary.
- Return only the JSON object."""


def build_voice_user_turn(task: dict, draft_code: str) -> list[dict]:
    text = (
        f"TASK DESCRIPTION:\n{task['description']}\n\n"
        f"MY IMPLEMENTATION:\n```python\n{draft_code}\n```\n\n"
        f"Return the JSON object with preamble, before_exec, confirm."
    )
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def build_fix_user_turn(task: dict, current_code: str, stderr: str,
                        prior_attempts: list[dict]) -> list[dict]:
    stderr_tail = "\n".join((stderr or "").strip().splitlines()[-25:])
    parts = [
        f"TASK DESCRIPTION:\n{task['description']}",
        f"TEST HARNESS (this is what runs after my code):\n```python\n{task['test_code']}\n```",
    ]
    if prior_attempts:
        hist = []
        for i, a in enumerate(prior_attempts, start=1):
            hist.append(
                f"ATTEMPT {i} CODE:\n```python\n{a['code']}\n```\n"
                f"ATTEMPT {i} TRACEBACK:\n{a['stderr_tail']}"
            )
        parts.append("\n\n".join(hist))
    parts.append(f"CURRENT IMPLEMENTATION (just failed):\n```python\n{current_code}\n```")
    parts.append(f"CURRENT TRACEBACK:\n{stderr_tail}")
    parts.append("Return the JSON object with diagnosis, revision_preamble, fixed_code.")
    text = "\n\n".join(parts)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    # Strip <think>...</think> blocks (Qwen3 reasoning mode).
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def _extract_json_object(text: str) -> str:
    """Extract the first balanced {...} JSON object from text, tolerating prose
    before/after. Qwen3 sometimes emits preamble or trailing commentary."""
    text = _strip_json_fences(text)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text[start:]


def _flatten_messages(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style content blocks to OpenAI-style flat strings."""
    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [p["text"] for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(parts)
        out.append({"role": msg["role"], "content": content})
    return out


def call_json(client, system_text: str,
              messages: list[dict], max_tokens: int, task_id: str,
              kind: str) -> dict | None:
    """Call Qwen3-235B via OpenRouter. `client` is ignored (kept for signature
    compatibility with prior Anthropic-based code)."""
    flat = _flatten_messages(messages)
    oai_messages = [{"role": "system", "content": system_text}] + flat
    payload = {
        "model": MODEL,
        "temperature": TEACHER_TEMPERATURE,
        "max_tokens": max_tokens,
        "messages": oai_messages,
    }
    try:
        resp = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TEACHER_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("%s call failed for %s: %s", kind, task_id, e)
        return None

    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError) as e:
        logger.warning("%s unexpected response shape for %s: %s / %r",
                       kind, task_id, e, str(data)[:200])
        return None

    obj_text = _extract_json_object(text)
    try:
        return json.loads(obj_text)
    except json.JSONDecodeError as e:
        logger.warning("%s JSON decode failed for %s: %s (head: %r)",
                       kind, task_id, e, obj_text[:200])
        return None


def collect_one(task: dict, stratum_target: str, evaluator: CodeEvaluator,
                client=None) -> dict | None:
    """Build a variable-length trace.

    stratum_target guides the DRAFT selection:
      - "no_revise": draft = fixed_code. Expect it to pass. Voice only, no fix call.
      - "revise":    draft = buggy_code. Sonnet derives diagnosis + fix. Retries
                     up to MAX_ATTEMPTS-1 times. Actual stratum depends on how
                     many attempts were needed.
    """
    if stratum_target == "no_revise":
        draft = task["fixed_code"]
    else:
        draft = task["buggy_code"]

    draft_exec = evaluator.evaluate(draft, task["test_code"], timeout=10)

    # If the no_revise draft unexpectedly fails, flip to revise to salvage.
    if stratum_target == "no_revise" and not draft_exec.passed:
        stratum_target = "revise"

    # Voice call (always needed — gives us preamble, before_exec, confirm)
    voice = call_json(
        client, VOICE_SYSTEM,
        build_voice_user_turn(task, draft),
        MAX_TOKENS_VOICE, task["task_id"], kind="voice",
    )
    if voice is None:
        return None
    required_voice = {"preamble", "before_exec", "confirm"}
    if not required_voice.issubset(voice.keys()):
        logger.warning("voice missing fields for %s", task["task_id"])
        return None

    if stratum_target == "no_revise":
        if not draft_exec.passed:
            logger.warning("no_revise draft failed for %s after flip — skipping", task["task_id"])
            return None
        return {
            "task_id": task["task_id"],
            "bug_class": task.get("bug_class", "?"),
            "description": task["description"],
            "test_code": task["test_code"],
            "stratum": "no_revise",
            "draft_code": draft,
            "draft_exec": _pack_exec(draft_exec),
            "attempts": [],
            "voice": voice,
        }

    # Revise path. Teacher must derive diagnosis + fix, possibly across retries.
    attempts: list[dict] = []
    current_code = draft
    current_stderr = draft_exec.stderr or ""

    for attempt_idx in range(MAX_ATTEMPTS):
        # Build prior attempts clause. We feed previous FAILED attempts back to
        # Sonnet so it doesn't repeat the same mistake.
        prior_attempts_arg = [
            {
                "code": a["code"],
                "stderr_tail": "\n".join((a.get("exec_stderr") or "").strip().splitlines()[-25:]),
            }
            for a in attempts
        ]
        if attempts:
            prior_clause = (
                "You have already made previous attempts that also failed. "
                "Study them carefully — do NOT repeat the same mistake. "
                "The correct fix must address both the original bug and avoid "
                "the pitfalls of your prior attempts."
            )
        else:
            prior_clause = ""
        fix_system = FIX_SYSTEM.replace("{PRIOR_ATTEMPTS_CLAUSE}", prior_clause)

        fix_json = call_json(
            client, fix_system,
            build_fix_user_turn(task, current_code, current_stderr, prior_attempts_arg),
            MAX_TOKENS_FIX, task["task_id"], kind=f"fix#{attempt_idx+1}",
        )
        if fix_json is None:
            return None
        required_fix = {"diagnosis", "revision_preamble", "fixed_code"}
        if not required_fix.issubset(fix_json.keys()):
            logger.warning("fix#%d missing fields for %s", attempt_idx + 1, task["task_id"])
            return None

        proposed = str(fix_json["fixed_code"]).strip()
        # Strip code fences if Sonnet accidentally added them
        if proposed.startswith("```"):
            proposed = re.sub(r"^```(?:python)?\s*\n?", "", proposed)
            proposed = re.sub(r"\n?```\s*$", "", proposed)

        exec_result = evaluator.evaluate(proposed, task["test_code"], timeout=10)
        attempts.append({
            "diagnosis": fix_json["diagnosis"].strip(),
            "revision_preamble": fix_json["revision_preamble"].strip(),
            "code": proposed,
            "exec": _pack_exec(exec_result),
            "exec_stderr": exec_result.stderr or "",
        })

        if exec_result.passed:
            break

        # Prepare next iteration
        current_code = proposed
        current_stderr = exec_result.stderr or ""

    # Determine stratum from the number of attempts used.
    if not attempts or not attempts[-1]["exec"]["passed"]:
        logger.warning("all %d attempts failed for %s — skipping",
                       len(attempts), task["task_id"])
        return None

    stratum_map = {1: "one_revise", 2: "two_revise", 3: "three_revise"}
    stratum = stratum_map.get(len(attempts))

    return {
        "task_id": task["task_id"],
        "bug_class": task.get("bug_class", "?"),
        "description": task["description"],
        "test_code": task["test_code"],
        "stratum": stratum,
        "draft_code": draft,
        "draft_exec": _pack_exec(draft_exec),
        # Trim exec_stderr from attempts before saving (kept only for retry context)
        "attempts": [
            {k: v for k, v in a.items() if k != "exec_stderr"}
            for a in attempts
        ],
        "voice": voice,
    }


def _pack_exec(r) -> dict:
    return {
        "passed": r.passed,
        "error": r.error_message,
        "stdout": (r.stdout or "")[-1500:],
        "stderr": (r.stderr or "")[-1500:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/training/code_tasks_v2_scrubbed.json")
    ap.add_argument("--output", default="data/training/review_traces_v3.json")
    ap.add_argument("--log", default="logs/collect_review_traces_v3.log")
    ap.add_argument("--no-revise-frac", type=float, default=0.20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
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

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    rng = random.Random(args.seed)
    n_no_revise = int(round(len(tasks) * args.no_revise_frac))
    targets = ["no_revise"] * n_no_revise + ["revise"] * (len(tasks) - n_no_revise)
    rng.shuffle(targets)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        collected = json.loads(out_path.read_text())
        done_ids = {t["task_id"] for t in collected}
        logger.info("Resuming: %d traces already collected", len(collected))
    else:
        collected = []
        done_ids = set()

    client = None  # no longer used (OpenRouter via httpx inside call_json)
    evaluator = CodeEvaluator()
    lock = threading.Lock()

    def work(task, target):
        if task["task_id"] in done_ids:
            return None
        trace = collect_one(task, target, evaluator, client)
        if trace is None:
            return None
        with lock:
            collected.append(trace)
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("COLLECTED [%d/%d] %s stratum=%s attempts=%d",
                        len(collected), len(tasks), trace["task_id"],
                        trace["stratum"], len(trace["attempts"]))
        return trace

    # Warm cache with one sequential call
    first_idx = 0
    while first_idx < len(tasks) and tasks[first_idx]["task_id"] in done_ids:
        first_idx += 1
    if first_idx < len(tasks):
        logger.info("Warming cache with one sequential call...")
        work(tasks[first_idx], targets[first_idx])
        first_idx += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i in range(first_idx, len(tasks)):
            futures.append(ex.submit(work, tasks[i], targets[i]))
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
