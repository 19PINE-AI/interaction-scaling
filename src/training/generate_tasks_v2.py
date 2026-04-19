"""Generate fresh code bug-fix tasks for Phase 4 autonomous-review distillation.

Calls Claude Sonnet 4.6 with prompt-cached system prompt describing the task
schema. Each task is self-validated: the buggy implementation must fail the
tests, and the reference fix must pass.

Output: `data/training/code_tasks_v2.json` — a list of dicts with the same
shape as `data/hard_benchmarks/code/code_tasks.json` plus `buggy_code` and
`fixed_code` for Stage A/C downstream.

Usage:
    python -m src.training.generate_tasks_v2 --target 200 --workers 8
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

SYSTEM_PROMPT = """You are generating training tasks for a coding agent. Each task presents a subtly buggy Python function and tests that expose the bug. The agent must fix the bug.

Strict output format — reply with ONE JSON object, no prose, no markdown fences:

{
  "task_id": "code_v2_XXX",
  "bug_class": "<short category>",
  "description": "<user-facing bug report + the buggy code inline in a ```python block + 'Please fix the bug in `func_name`.'>",
  "buggy_code": "<the exact buggy implementation in plain Python, runnable>",
  "fixed_code": "<a reference correct implementation, runnable, same function signature>",
  "test_code": "<a series of `assert` statements exercising normal and edge cases, ending with a `print('All tests passed!')`>",
  "common_bugs": ["<bug mode 1>", "<bug mode 2>", "<bug mode 3>", "<bug mode 4>"]
}

Rules:
1. The buggy_code must be self-contained (all imports inside or standard library). It must DEFINE the function(s) but they must behave incorrectly on at least one test case.
2. The fixed_code must have the identical function signature(s) as buggy_code and must pass ALL test assertions.
3. test_code calls the function(s) by name; it does not redefine them. It contains 8–14 assertions covering happy path, edge cases, and the specific bug.
4. The bug must be subtle and motivated — a realistic mistake a senior engineer might actually make (e.g. off-by-one, wrong quote-state handling, missing lazy-propagation pushdown, comparator that ignores tiebreak). Not a typo.
5. Use only the standard library. No network calls, no file I/O, no random seed non-determinism.
6. Keep each test runtime under 1 second.
7. The description must be concrete: name the input, name the expected output, and quote the buggy code block. No vague "fix the bug" without specifics.
8. Never re-use a task_id or bug that duplicates the exemplars or prior tasks shown in the user turn.
9. The bug must be in logic, not in imports, not in syntax.
10. Do NOT include a main block, CLI parser, or `if __name__` guard in any of the three code fields.

Return only the JSON object. No explanation, no extra text before or after."""


def load_existing_descriptions() -> list[str]:
    """Collect task descriptions from the training + held-out sets for dedup."""
    descs: list[str] = []
    for path in [
        Path("data/hard_benchmarks/code/code_tasks.json"),
        Path("data/hard_benchmarks/code/code_tasks_heldout.json"),
    ]:
        if path.exists():
            for t in json.loads(path.read_text()):
                descs.append(t["description"])
    return descs


def desc_signature(desc: str) -> str:
    """Cheap dedup signature: first 200 non-space chars lowercased."""
    cleaned = re.sub(r"\s+", " ", desc.lower())
    return cleaned[:200]


def build_user_turn(category: str, exemplar_desc: str, prior_signatures: list[str]) -> list[dict]:
    """Build the (non-cacheable) user turn."""
    # show up to 20 prior task signatures (first 80 chars each) to push novelty
    rng = random.Random()
    rng.seed(hash(category) ^ int(time.time() * 1000) & 0xFFFF)
    sample_prior = rng.sample(prior_signatures, min(20, len(prior_signatures))) if prior_signatures else []

    avoid_block = ""
    if sample_prior:
        avoid_lines = "\n".join(f"- {s[:120]}" for s in sample_prior)
        avoid_block = f"\n\nAvoid duplicating the FUNCTION or BUG of any of these existing tasks:\n{avoid_lines}"

    exemplar_block = f"\n\nOne exemplar in the same spirit (do NOT copy it — use a different function and a different bug):\n{exemplar_desc[:1500]}" if exemplar_desc else ""

    text = (
        f"Generate ONE task in the bug category: {category}.{exemplar_block}{avoid_block}\n\n"
        f"Return exactly one JSON object following the schema."
    )
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def call_claude(client: anthropic.Anthropic, category: str, exemplar_desc: str,
                prior_signatures: list[str]) -> dict | None:
    """One generation attempt. Returns parsed JSON dict or None on failure."""
    messages = build_user_turn(category, exemplar_desc, prior_signatures)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
    except Exception as e:
        logger.warning("Claude call failed for %s: %s", category, e)
        return None

    # Log cache stats occasionally
    u = resp.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(u, "cache_creation_input_tokens", 0) or 0
    logger.debug("usage: in=%d out=%d cache_read=%d cache_create=%d",
                 u.input_tokens, u.output_tokens, cache_read, cache_create)

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Strip accidental code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        task = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed for %s: %s", category, e)
        return None
    return task


def validate_task(task: dict, evaluator: CodeEvaluator) -> tuple[bool, str]:
    """Check schema, then execute buggy (must fail) and fixed (must pass)."""
    required = {"task_id", "bug_class", "description", "buggy_code",
                "fixed_code", "test_code", "common_bugs"}
    missing = required - set(task.keys())
    if missing:
        return False, f"missing fields: {missing}"

    for field in ("buggy_code", "fixed_code", "test_code"):
        if not isinstance(task[field], str) or not task[field].strip():
            return False, f"empty or non-string {field}"

    # Fixed impl must pass.
    fixed = evaluator.evaluate(task["fixed_code"], task["test_code"], timeout=10)
    if not fixed.passed:
        return False, f"fixed_code failed tests: {fixed.error_message}"

    # Buggy impl must fail.
    buggy = evaluator.evaluate(task["buggy_code"], task["test_code"], timeout=10)
    if buggy.passed:
        return False, "buggy_code passed tests (bug is not actually a bug)"

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200, help="Number of validated tasks to produce.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=600,
                    help="Hard cap on generation attempts.")
    ap.add_argument("--output", default="data/training/code_tasks_v2.json")
    ap.add_argument("--log", default="logs/generate_tasks_v2.log")
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

    client = anthropic.Anthropic()
    evaluator = CodeEvaluator()

    existing_descs = load_existing_descriptions()
    existing_sigs = {desc_signature(d) for d in existing_descs}
    logger.info("Loaded %d existing task descriptions (train + heldout) for dedup", len(existing_descs))

    # For seed exemplars: use a random existing task as few-shot inspiration.
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

    def submit_one(category: str) -> dict | None:
        with lock:
            if len(collected) >= args.target:
                return None
            if attempts[0] >= args.max_attempts:
                return None
            attempts[0] += 1
            exemplar = random.choice(existing_descs) if existing_descs else ""
            prior_sigs = list(existing_sigs | collected_sigs)
        task = call_claude(client, category, exemplar, prior_sigs)
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
            task["task_id"] = f"code_v2_{next_id[0]:03d}"
            next_id[0] += 1
            collected.append(task)
            collected_sigs.add(sig)
            # Incremental checkpoint
            out_path.write_text(json.dumps(collected, indent=2))
            logger.info("ACCEPT [%d/%d] %s (%s) attempts=%d",
                        len(collected), args.target, task["task_id"],
                        task["bug_class"], attempts[0])
        return task

    # Warm the cache with one sequential call first so parallel workers get cache-reads.
    if not collected:
        logger.info("Warming prompt cache with one sequential call...")
        submit_one(random.choice(BUG_CATEGORIES))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        # Keep submitting until target or attempt cap
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
            # drain any ready
            for f in list(futures):
                if f.done():
                    futures.remove(f)
            time.sleep(0.2)
        # final drain
        for f in as_completed(futures):
            pass

    out_path.write_text(json.dumps(collected, indent=2))
    logger.info("Done: %d tasks collected in %d attempts -> %s",
                len(collected), attempts[0], out_path)


if __name__ == "__main__":
    main()
