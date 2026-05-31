"""Generate an expanded held-out bug-fix benchmark using OpenRouter/Qwen3-235B.

Ported from `generate_tasks_v2.py` (Anthropic) — same schema, same validation,
different task-ids and output path so we can compare all adapters on a wider
test set than the original 15 held-out tasks.

Each task is self-validated: buggy_code must FAIL the test_code, fixed_code
must PASS it. Dedupes by description signature against train + original
held-out.

Usage:
    OPENROUTER_API_KEY=... python -m src.training.generate_heldout_v2 \\
        --target 30 --workers 6 \\
        --output data/hard_benchmarks/code/code_tasks_heldout_v2.json
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

import httpx

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

MODEL = "qwen/qwen3-235b-a22b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.7
TIMEOUT = 300.0
MAX_TOKENS = 8000

BUG_CATEGORIES = [
    "string parsing (CSV/TSV/INI/JSON/TOML/YAML fragment)",
    "URL/URI parsing or normalization",
    "percent/base64/hex/UTF-8 encoding edge case",
    "date/time arithmetic (timezone, DST, leap year, ISO8601)",
    "regex escaping or boundary bug",
    "state machine / finite automaton transition",
    "tree/graph traversal boundary case",
    "heap/priority-queue comparator misuse",
    "hash table / bloom / skip list remove semantics",
    "HTTP header or request-line parsing",
    "IP address / CIDR / subnet arithmetic",
    "numeric precision / rounding / overflow",
    "sorting with custom key / stability / tie-break",
    "deduplication / set / multiset invariant",
    "concurrency-free counter / accumulator edge case",
    "line-ending / framing / delimiter handling",
    "pagination / cursor / window sliding",
    "bit manipulation / bitmask / bitfield",
    "escape / quoting rules (shell, SQL, HTML)",
    "interval / range / segment overlap",
    "version comparison (semver, natural sort)",
    "checksum / CRC / hash boundary",
    "canonical form normalization (paths, hostnames)",
    "priority / scheduling / deadline tie-break",
    "streaming / windowed aggregation",
]

SYSTEM_PROMPT = """You are generating evaluation tasks for a coding agent. Each task presents a subtly buggy Python function and tests that expose the bug. The agent must fix the bug.

Strict output format — reply with ONE JSON object, no prose, no markdown fences:

{
  "task_id": "code_hv2_XXX",
  "bug_class": "<short category>",
  "description": "<user-facing bug report + the buggy code inline in a ```python block + 'Please fix the bug in `func_name`.'>",
  "buggy_code": "<the exact buggy implementation in plain Python, runnable>",
  "fixed_code": "<a reference correct implementation, runnable, same function signature>",
  "test_code": "<a series of `assert` statements exercising normal and edge cases, ending with a `print('All tests passed!')`>",
  "common_bugs": ["<bug mode 1>", "<bug mode 2>", "<bug mode 3>", "<bug mode 4>"]
}

Rules:
1. buggy_code must be self-contained (all imports inside or standard library). It must DEFINE the function(s) but they must behave incorrectly on at least one test case.
2. fixed_code must have the identical function signature(s) as buggy_code and must pass ALL test assertions.
3. test_code calls the function(s) by name; it does not redefine them. It contains 8–14 assertions covering happy path, edge cases, and the specific bug.
4. The bug must be subtle and motivated — a realistic mistake a senior engineer might make (off-by-one, wrong quote-state handling, missing lazy-propagation pushdown, comparator that ignores tiebreak). Not a typo.
5. Use only the standard library. No network calls, no file I/O, no random seed non-determinism.
6. Keep each test runtime under 1 second.
7. The description must be concrete: name the input, name the expected output, and quote the buggy code block. No vague "fix the bug" without specifics.
8. Never re-use a task_id or bug that duplicates the exemplars or prior tasks shown in the user turn.
9. The bug must be in logic, not in imports, not in syntax.
10. Do NOT include a main block, CLI parser, or `if __name__` guard in any of the three code fields.

Return only the JSON object. No explanation, no extra text before or after."""


def load_existing_descriptions() -> list[str]:
    descs: list[str] = []
    for path in [
        Path("data/hard_benchmarks/code/code_tasks.json"),
        Path("data/hard_benchmarks/code/code_tasks_heldout.json"),
        Path("data/training/code_tasks_v2_scrubbed.json"),
    ]:
        if path.exists():
            for t in json.loads(path.read_text()):
                descs.append(t["description"])
    return descs


def desc_signature(desc: str) -> str:
    cleaned = re.sub(r"\s+", " ", desc.lower())
    return cleaned[:200]


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def _extract_json_object(text: str) -> str:
    text = _strip_fences(text)
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


def call_qwen(category: str, exemplar_desc: str, prior_signatures: list[str]) -> dict | None:
    sample_prior = random.sample(prior_signatures, min(20, len(prior_signatures))) \
        if prior_signatures else []
    avoid_block = ""
    if sample_prior:
        avoid_lines = "\n".join(f"- {s[:120]}" for s in sample_prior)
        avoid_block = f"\n\nAvoid duplicating the FUNCTION or BUG of any of these existing tasks:\n{avoid_lines}"
    exemplar_block = f"\n\nOne exemplar in the same spirit (do NOT copy — different function and different bug):\n{exemplar_desc[:1500]}" if exemplar_desc else ""

    user_text = (
        f"Generate ONE task in the bug category: {category}.{exemplar_block}{avoid_block}\n\n"
        f"Return exactly one JSON object following the schema."
    )
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    try:
        resp = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Qwen call failed for %s: %s", category, e)
        return None
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError):
        return None
    obj_text = _extract_json_object(text)
    try:
        return json.loads(obj_text)
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed for %s: %s (head: %r)", category, e, obj_text[:200])
        return None


def validate_task(task: dict, evaluator: CodeEvaluator) -> tuple[bool, str]:
    required = {"task_id", "bug_class", "description", "buggy_code",
                "fixed_code", "test_code", "common_bugs"}
    missing = required - set(task.keys())
    if missing:
        return False, f"missing fields: {missing}"
    for field in ("buggy_code", "fixed_code", "test_code"):
        if not isinstance(task[field], str) or not task[field].strip():
            return False, f"empty or non-string {field}"
    fixed = evaluator.evaluate(task["fixed_code"], task["test_code"], timeout=10)
    if not fixed.passed:
        return False, f"fixed_code failed tests: {fixed.error_message}"
    buggy = evaluator.evaluate(task["buggy_code"], task["test_code"], timeout=10)
    if buggy.passed:
        return False, "buggy_code passed tests (bug is not actually a bug)"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--output", default="data/hard_benchmarks/code/code_tasks_heldout_v2.json")
    ap.add_argument("--log", default="logs/generate_heldout_v2.log")
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

    evaluator = CodeEvaluator()

    existing_descs = load_existing_descriptions()
    existing_sigs = {desc_signature(d) for d in existing_descs}
    logger.info("Loaded %d existing task descriptions for dedup", len(existing_descs))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        collected = json.loads(out_path.read_text())
        logger.info("Resuming from existing output with %d tasks", len(collected))
    else:
        collected = []
    collected_sigs = {desc_signature(t["description"]) for t in collected}

    lock = threading.Lock()
    attempts = [0]
    next_id = [len(collected) + 1]

    def submit_one(category: str):
        with lock:
            if len(collected) >= args.target:
                return None
            if attempts[0] >= args.max_attempts:
                return None
            attempts[0] += 1
            exemplar = random.choice(existing_descs) if existing_descs else ""
            prior_sigs = list(existing_sigs | collected_sigs)
        task = call_qwen(category, exemplar, prior_sigs)
        if task is None:
            return None
        ok, reason = validate_task(task, evaluator)
        if not ok:
            logger.info("REJECT (%s): %s", category, reason)
            return None
        sig = desc_signature(task["description"])
        with lock:
            if sig in existing_sigs or sig in collected_sigs:
                logger.info("REJECT (%s): duplicate signature", category)
                return None
            task["task_id"] = f"code_hv2_{next_id[0]:03d}"
            task["difficulty"] = "hard"
            next_id[0] += 1
            collected.append(task)
            collected_sigs.add(sig)
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("ACCEPT [%d/%d] %s (%s) attempts=%d",
                        len(collected), args.target, task["task_id"],
                        task["bug_class"], attempts[0])
        return task

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        while True:
            with lock:
                if len(collected) >= args.target or attempts[0] >= args.max_attempts:
                    break
                in_flight = sum(1 for f in futures if not f.done())
                capacity = args.workers * 2 - in_flight
            if capacity <= 0:
                time.sleep(0.5)
                continue
            for _ in range(capacity):
                cat = random.choice(BUG_CATEGORIES)
                futures.append(ex.submit(submit_one, cat))
            for f in list(futures):
                if f.done():
                    futures.remove(f)
            time.sleep(0.2)
        for f in as_completed(futures):
            pass

    out_path.write_text(json.dumps(collected, indent=2))
    logger.info("Done: %d tasks collected in %d attempts -> %s",
                len(collected), attempts[0], out_path)


if __name__ == "__main__":
    main()
