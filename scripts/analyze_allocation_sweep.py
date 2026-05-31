"""Analyze results/allocation_sweep/code_allocation.json and emit the
9-row summary table + findings."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "results" / "allocation_sweep" / "code_allocation.json"


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    cells = data["cells"]
    cfg = data["config"]

    rows = sorted(cells, key=lambda c: -c["pass_rate"])

    print()
    print(f"# Allocation sweep — code (B={cfg['total_budget']} tokens, max_iter={cfg['max_iterations']}, n={cfg['n_tasks']})")
    print()
    print(f"| Label | b1 (prop) | b2 (exec) | b3 (rev) | Pass-rate | Mean tokens | Mean iters | Prop cap | Rev cap |")
    print(f"|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        print(
            f"| {c['allocation_label']} "
            f"| {c['b1_propose']:.2f} | {c['b2_execute']:.2f} | {c['b3_review']:.2f} "
            f"| {c['n_passed']}/{c['n_tasks']} ({100*c['pass_rate']:.1f}%) "
            f"| {c['mean_tokens']:.0f} | {c['mean_iterations']:.2f} "
            f"| {c['propose_cap']} | {c['review_cap']} |"
        )

    rates = [c["pass_rate"] for c in cells]
    best = max(cells, key=lambda c: c["pass_rate"])
    worst = min(cells, key=lambda c: c["pass_rate"])
    spread_pp = (max(rates) - min(rates)) * 100
    mean_rate = sum(rates) / len(rates)
    stdev = (sum((r - mean_rate) ** 2 for r in rates) / len(rates)) ** 0.5

    print()
    print(f"**Best:** {best['allocation_label']} = {best['pass_rate']*100:.1f}%")
    print(f"**Worst:** {worst['allocation_label']} = {worst['pass_rate']*100:.1f}%")
    print(f"**Spread:** {spread_pp:.1f} pp")
    print(f"**Mean:** {mean_rate*100:.1f}% +/- {stdev*100:.1f} pp")


if __name__ == "__main__":
    main()
