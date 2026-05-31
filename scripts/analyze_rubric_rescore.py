"""Summarize a rubric rescore file: single-shot vs reviewed lift + sign test.

Works on the rescore JSONs produced by rescore_slides_hard / rescore_video_rubric
/ rescore_modality_rubric (rows with ss_rubric / rv_rubric, optionally
ss_defects / rv_defects). Aggregates per task-run pair (the unit of comparison),
reports mean single-shot, mean reviewed, lift, the two-sided sign test p-value,
and a saturation count (perfect-scoring single-shot artifacts).

Usage:
  python -m scripts.analyze_rubric_rescore <rescore.json> [--metric rubric|defects]
"""

import argparse
import json
from math import comb


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test over decisive (non-tied) pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--metric", choices=["rubric", "defects"], default="rubric")
    args = ap.parse_args()

    rows = json.load(open(args.path))
    m = args.metric
    lower_is_better = (m == "defects")

    pairs = []  # (ss, rv) for rows where both scored
    for r in rows:
        ss, rv = r.get(f"ss_{m}"), r.get(f"rv_{m}")
        if ss is not None and rv is not None:
            pairs.append((ss, rv))

    n = len(pairs)
    if n == 0:
        print(f"no scored pairs for metric={m} in {args.path}")
        return
    mean_ss = sum(p[0] for p in pairs) / n
    mean_rv = sum(p[1] for p in pairs) / n

    # "improvement" direction depends on the metric
    def improved(ss, rv):
        return rv < ss if lower_is_better else rv > ss
    def regressed(ss, rv):
        return rv > ss if lower_is_better else rv < ss

    wins = sum(1 for ss, rv in pairs if improved(ss, rv))
    losses = sum(1 for ss, rv in pairs if regressed(ss, rv))
    ties = n - wins - losses
    p = sign_test_p(wins, losses)

    print(f"=== {args.path}  (metric={m}, n={n} task-run pairs) ===")
    print(f"mean single-shot : {mean_ss:.3f}")
    print(f"mean reviewed    : {mean_rv:.3f}")
    sign = "-" if lower_is_better else "+"
    print(f"lift (rv-ss)     : {mean_rv - mean_ss:+.3f}")
    print(f"improved / regressed / tied : {wins} / {losses} / {ties}")
    print(f"two-sided sign-test p       : {p:.4f}")

    if m == "rubric":
        perfect_ss = sum(1 for ss, _ in pairs if ss >= 0.999)
        print(f"single-shot perfect (=1.0)  : {perfect_ss}/{n} "
              f"({'SATURATED' if perfect_ss / n > 0.6 else 'has headroom'})")
    if m == "defects":
        clean_ss = sum(1 for ss, _ in pairs if ss == 0)
        clean_rv = sum(1 for _, rv in pairs if rv == 0)
        print(f"geometrically clean SS / RV : {clean_ss}/{n}  ->  {clean_rv}/{n}")


if __name__ == "__main__":
    main()
