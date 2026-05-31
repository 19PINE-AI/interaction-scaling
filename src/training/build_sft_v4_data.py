"""Assemble SFT v4 training data with target stratum ratios.

Combines:
  - data/training/review_traces_v3.json        (natural traces, various strata)
  - data/training/review_traces_v3_forced.json (forced two_revise, seed 77)
  - data/training/review_traces_v3_forced2.json (optional second pass)
  - data/training/review_traces_v3_three.json  (forced three_revise)

Policy:
  - Drop no_revise entirely.
  - Stratum target ratios: 30% one_revise / 40% two_revise / 30% three_revise.
  - Dedupe by (task_id, stratum) preferring natural > forced > forced2 > three.
    A given task may appear in multiple strata — that's fine and good for
    cross-stratum coverage.

Then stitches with stitch_trajectory_v3.stitch() and writes sft_review_v4.json.

Usage:
    python -m src.training.build_sft_v4_data \\
        --output data/training/sft_review_v4.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.training.stitch_trajectory_v3 import stitch

logger = logging.getLogger(__name__)


def load_if_exists(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning("Not found: %s", path)
        return []
    data = json.loads(p.read_text())
    logger.info("Loaded %d from %s", len(data), path)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural", default="data/training/review_traces_v3.json")
    ap.add_argument("--forced", default="data/training/review_traces_v3_forced.json")
    ap.add_argument("--forced2", default="data/training/review_traces_v3_forced2.json")
    ap.add_argument("--three", default="data/training/review_traces_v3_three.json")
    ap.add_argument("--output", default="data/training/sft_review_v4.json")
    ap.add_argument("--ratio-one", type=float, default=0.30)
    ap.add_argument("--ratio-two", type=float, default=0.40)
    ap.add_argument("--ratio-three", type=float, default=0.30)
    ap.add_argument("--target-total", type=int, default=0,
                    help="0 = use as much as available respecting ratios")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    natural = load_if_exists(args.natural)
    forced = load_if_exists(args.forced)
    forced2 = load_if_exists(args.forced2)
    three = load_if_exists(args.three)

    # Bucket by stratum. Source priority: natural > forced > forced2 > three.
    # Use (task_id, stratum) as dedup key — the same task across strata is OK.
    buckets: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for source in (natural, forced, forced2, three):
        for t in source:
            stratum = t.get("stratum")
            if stratum in (None, "no_revise"):
                continue  # drop no_revise entirely
            key = (t["task_id"], stratum)
            if key in seen:
                continue
            seen.add(key)
            buckets[stratum].append(t)

    for s in ("one_revise", "two_revise", "three_revise"):
        logger.info("Pool %s: %d", s, len(buckets.get(s, [])))

    # Determine how many of each stratum to use.
    ratios = {
        "one_revise": args.ratio_one,
        "two_revise": args.ratio_two,
        "three_revise": args.ratio_three,
    }
    # Find limiting stratum to compute max N if target_total=0.
    if args.target_total == 0:
        # total N s.t. buckets[s] >= ratios[s] * N for all s.
        # N <= len(buckets[s]) / ratios[s] for each s
        caps = []
        for s, r in ratios.items():
            n = len(buckets.get(s, []))
            if r > 0 and n > 0:
                caps.append(int(n / r))
        target_total = min(caps) if caps else 0
        logger.info("Auto target_total = %d (limited by smallest bucket relative to ratio)",
                    target_total)
    else:
        target_total = args.target_total

    rng = random.Random(args.seed)
    selected: list[dict] = []
    final_counts = Counter()
    for s, r in ratios.items():
        want = int(round(r * target_total))
        pool = list(buckets.get(s, []))
        rng.shuffle(pool)
        take = pool[:want]
        selected.extend(take)
        final_counts[s] = len(take)

    logger.info("Selected stratum counts: %s (total=%d)", dict(final_counts), len(selected))

    rng.shuffle(selected)
    stitched = [stitch(t, rng) for t in selected]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(stitched, indent=2))

    out_strata = Counter(s["stratum"] for s in stitched)
    logger.info("Wrote %d examples to %s. Strata: %s",
                len(stitched), args.output, dict(out_strata))


if __name__ == "__main__":
    main()
