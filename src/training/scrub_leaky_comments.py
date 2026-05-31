"""Strip bug-label comments from task buggy_code / fixed_code.

The Sonnet 4.6 task generator frequently annotates buggy_code with comments like
`# BUG: heap[0] is negative` or `# Missing pushdown`. When these are fed into
SFT training via the first-draft tool_call, the student learns to emit its own
buggy code WITH self-labeled bug comments — a bizarre pattern that wastes tool
budget and has no real-world analog.

This scrubber removes those comments from a tasks file in place (well, to a new
output path) and validates that every bug still triggers a failure and every
fix still passes.

Usage:
    python -m src.training.scrub_leaky_comments \\
        --input data/training/code_tasks_v2.json \\
        --output data/training/code_tasks_v2_scrubbed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

BUG_WORDS = (
    r"(?:BUG|FIXME|TODO|XXX|bug|Bug|[Mm]issing|[Ii]ncorrect|[Ww]rong|"
    r"[Oo]ff.?by|[Ss]hould be|\b[Ff]ix\b|[Bb]uggy)"
)
FULL_LINE_RE = re.compile(rf"^\s*#[^\n]*{BUG_WORDS}[^\n]*$\n?", re.MULTILINE)
INLINE_RE = re.compile(rf"(\S.*?)\s*#[^\n]*{BUG_WORDS}[^\n]*")


def scrub(code: str) -> str:
    code = FULL_LINE_RE.sub("", code)
    code = INLINE_RE.sub(r"\1", code)
    code = re.sub(r"\n\n\n+", "\n\n", code)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/training/code_tasks_v2.json")
    ap.add_argument("--output", default="data/training/code_tasks_v2_scrubbed.json")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Skip the evaluator sweep (faster dry-run).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = json.loads(Path(args.input).read_text())
    logger.info("Loaded %d tasks from %s", len(tasks), args.input)

    scrubbed = []
    for t in tasks:
        t2 = dict(t)
        t2["buggy_code"] = scrub(t["buggy_code"])
        t2["fixed_code"] = scrub(t["fixed_code"])
        scrubbed.append(t2)

    if not args.skip_validate:
        ev = CodeEvaluator()
        broken = []
        for t in scrubbed:
            r_bug = ev.evaluate(t["buggy_code"], t["test_code"], timeout=10)
            r_fix = ev.evaluate(t["fixed_code"], t["test_code"], timeout=10)
            if r_bug.passed or not r_fix.passed:
                broken.append((t["task_id"], r_bug.passed, r_fix.passed))
        if broken:
            logger.error("Scrub broke %d tasks — aborting write:", len(broken))
            for b in broken[:20]:
                logger.error("  %s bug_passed=%s fix_passed=%s", *b)
            sys.exit(1)
        logger.info("All %d tasks validate: bug fails, fix passes.", len(scrubbed))

    Path(args.output).write_text(json.dumps(scrubbed, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
