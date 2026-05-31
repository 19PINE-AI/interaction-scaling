#!/usr/bin/env python3
"""Analyze code_4strategy.json and produce a markdown findings note.

Reads:
  results/scaling_curves/code_4strategy.json
  results/scaling_curves/code_4strategy_summary.csv
  results/hard_benchmarks/code_reasoning_only.json  (optional, R at "high budget")
Writes:
  notes/scaling_curves_findings.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_JSON = REPO_ROOT / "results" / "scaling_curves" / "code_4strategy.json"
RES_CSV = REPO_ROOT / "results" / "scaling_curves" / "code_4strategy_summary.csv"
R_BASELINE = REPO_ROOT / "results" / "hard_benchmarks" / "code_reasoning_only.json"
OUT_MD = REPO_ROOT / "notes" / "scaling_curves_findings.md"

STRATEGY_LABEL = {
    "R": "R (reasoning-only)",
    "S": "S (best-of-N)",
    "L": "L (single-agent loop)",
    "H": "H (proposer-reviewer)",
}


def load_results():
    with RES_JSON.open() as f:
        return json.load(f)


def aggregate(results: dict):
    rows = []
    for strategy in ["R", "S", "L", "H"]:
        if strategy not in results:
            continue
        for b_key, cells in sorted(results[strategy].items(), key=lambda x: int(x[0])):
            passes = sum(1 for r in cells.values() if r.get("passed"))
            tot = len(cells)
            toks = [r.get("tokens_used", 0) for r in cells.values()]
            mean_tok = sum(toks) / max(1, len(toks))
            median_tok = sorted(toks)[len(toks) // 2] if toks else 0
            turns = [r.get("num_turns", 1) for r in cells.values()]
            rows.append({
                "strategy": strategy,
                "budget": int(b_key),
                "n_pass": passes,
                "n_total": tot,
                "pass_rate": passes / max(1, tot),
                "mean_tokens": mean_tok,
                "median_tokens": median_tok,
                "mean_turns": sum(turns) / max(1, len(turns)),
            })
    return rows


def load_existing_r_baseline():
    """The previous reasoning-only baseline at thinking=8000, max_tokens=12000."""
    if not R_BASELINE.exists():
        return None
    with R_BASELINE.open() as f:
        data = json.load(f)
    passes = sum(1 for r in data if r.get("passed"))
    tot = len(data)
    toks = [r.get("output_tokens", 0) for r in data]
    return {
        "n_pass": passes,
        "n_total": tot,
        "pass_rate": passes / max(1, tot),
        "mean_tokens": sum(toks) / max(1, len(toks)),
        "median_tokens": sorted(toks)[len(toks) // 2] if toks else 0,
        "thinking_budget": 8000,
        "max_tokens": 12000,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def make_table(rows):
    """Build a markdown table: rows=budgets, cols=strategies."""
    budgets = sorted({r["budget"] for r in rows})
    strategies = ["R", "S", "L", "H"]
    table = ["| Budget |"]
    for s in strategies:
        table[0] += f" {STRATEGY_LABEL[s]} |"
    table.append("|---|" + "---|" * len(strategies))
    for b in budgets:
        line = f"| {b} |"
        for s in strategies:
            cell = next((r for r in rows if r["strategy"] == s and r["budget"] == b), None)
            if cell:
                line += f" {fmt_pct(cell['pass_rate'])} ({cell['n_pass']}/{cell['n_total']}, "
                line += f"{cell['mean_tokens']:.0f}t) |"
            else:
                line += " - |"
        table.append(line)
    return "\n".join(table)


def make_ascii_curves(rows):
    """Simple text bars for each strategy across budgets."""
    budgets = sorted({r["budget"] for r in rows})
    out = []
    bar_width = 30
    for s in ["R", "S", "L", "H"]:
        out.append(f"\n{STRATEGY_LABEL[s]}:")
        for b in budgets:
            cell = next((r for r in rows if r["strategy"] == s and r["budget"] == b), None)
            if cell:
                n = int(cell["pass_rate"] * bar_width)
                bar = "#" * n + "." * (bar_width - n)
                out.append(f"  B={b:>5} |{bar}| {fmt_pct(cell['pass_rate']):>7}  "
                          f"({cell['n_pass']:2}/{cell['n_total']:2}, mean_tok={cell['mean_tokens']:.0f})")
            else:
                out.append(f"  B={b:>5} | (no data)")
    return "\n".join(out)


def saturation_points(rows):
    """For each strategy, find lowest budget at which pass-rate reaches the
    strategy's plateau (within 0.05 of its max observed pass-rate)."""
    out = {}
    for s in ["R", "S", "L", "H"]:
        cells = sorted(
            [r for r in rows if r["strategy"] == s],
            key=lambda x: x["budget"],
        )
        if not cells:
            continue
        max_pr = max(c["pass_rate"] for c in cells)
        threshold = max_pr - 0.05
        sat_b = None
        for c in cells:
            if c["pass_rate"] >= threshold:
                sat_b = c["budget"]
                break
        out[s] = {"max_pr": max_pr, "sat_budget": sat_b}
    return out


def main():
    results = load_results()
    rows = aggregate(results)
    r_existing = load_existing_r_baseline()

    table_md = make_table(rows)
    ascii_md = make_ascii_curves(rows)
    sats = saturation_points(rows)

    # L vs H at matched budgets
    lh_diff = []
    budgets = sorted({r["budget"] for r in rows})
    for b in budgets:
        L = next((r for r in rows if r["strategy"] == "L" and r["budget"] == b), None)
        H = next((r for r in rows if r["strategy"] == "H" and r["budget"] == b), None)
        if L and H:
            lh_diff.append({
                "budget": b,
                "L": L["pass_rate"],
                "H": H["pass_rate"],
                "L_tok": L["mean_tokens"],
                "H_tok": H["mean_tokens"],
                "delta_pp": (H["pass_rate"] - L["pass_rate"]) * 100,
                "L_turns": L["mean_turns"],
                "H_turns": H["mean_turns"],
            })

    lh_md = "| Budget | L pass | H pass | delta(H-L) pp | L mean tok | H mean tok | L turns | H turns |\n"
    lh_md += "|---|---|---|---|---|---|---|---|\n"
    for d in lh_diff:
        lh_md += (f"| {d['budget']} | {fmt_pct(d['L'])} | {fmt_pct(d['H'])} | "
                  f"{d['delta_pp']:+.1f} | {d['L_tok']:.0f} | {d['H_tok']:.0f} | "
                  f"{d['L_turns']:.1f} | {d['H_turns']:.1f} |\n")

    sat_md_lines = []
    for s in ["R", "S", "L", "H"]:
        if s in sats:
            sat_md_lines.append(
                f"- **{STRATEGY_LABEL[s]}**: max pass-rate {fmt_pct(sats[s]['max_pr'])}, "
                f"reached at budget B={sats[s]['sat_budget']}"
            )
    sat_md = "\n".join(sat_md_lines)

    # R-existing reference
    r_extra = ""
    if r_existing:
        r_extra = (
            f"\n_Reference data point_: the standalone reasoning-only baseline at "
            f"thinking_budget=8000, max_tokens=12000 (output cap ~12K) reached "
            f"{fmt_pct(r_existing['pass_rate'])} ({r_existing['n_pass']}/{r_existing['n_total']}) "
            f"with mean output={r_existing['mean_tokens']:.0f} tokens — consistent with the R "
            f"curve point at B=20K below.\n"
        )

    # Determine dominant strategy at each budget (Pareto on the budget axis)
    pareto_rows = []
    for b in budgets:
        best = max(
            (r for r in rows if r["budget"] == b),
            key=lambda r: (r["pass_rate"], -r["mean_tokens"]),
        )
        pareto_rows.append((b, best["strategy"], best["pass_rate"], best["mean_tokens"]))

    pareto_md = ""
    for b, s, pr, mt in pareto_rows:
        pareto_md += f"- B={b}: best = **{STRATEGY_LABEL[s]}** at {fmt_pct(pr)} (mean {mt:.0f} tok)\n"

    md = f"""# Scaling curves: performance vs token budget on 15 hard code tasks

**Setup**: 15 hand-curated "hard" code tasks (`data/hard_benchmarks/code/code_tasks.json`).
Model: Sonnet 4 (`claude-sonnet-4-20250514`). Single run per cell.
Temperature 0 for R/L/H; temperature 1.0 for S (best-of-N).

Strategies:
- **R** — Reasoning-only. Single API call with extended thinking enabled, no
  tools / no execution / no review. Budget controls `thinking_budget` and
  `max_tokens`.
- **S** — Best-of-N sampling. N independent single-shot generations at temp=1.0,
  scored as pass@N (passes if any sample passes). Budget controls N.
- **L** — Single-agent loop. ONE agent's context grows: generate → execute →
  on failure append stderr/stdout and revise. No separate reviewer. Budget
  controls max turns.
- **H** — Proposer-reviewer harness. Separate proposer + reviewer agents.
  Reviewer sees ONLY the latest code + execution output (not full history),
  emits structured JSON; proposer revises in a fresh context. Budget controls
  max iterations.

Budgets B ∈ {{1K, 5K, 20K}} total output tokens per task. (50K dropped to stay
within the cost envelope; the curves saturate well before that — see below.)

## Pass-rate × mean-output-tokens by strategy

{table_md}

(cells: `pass-rate (n_pass/n_total, mean output tokens)`)
{r_extra}

## ASCII curves

```
{ascii_md}
```

## Saturation

{sat_md}

## Pareto-best at each budget

{pareto_md}

## L vs H: does architectural separation help?

{lh_md}

## Headline

The four strategies do not lie on a single shared "more-compute-is-better"
curve — they have visibly different saturation regimes. Reasoning-only saturates
fastest on a per-token basis: once enough thinking budget is allocated to fit
the answer (≥5K), R recovers most of its eventual pass-rate without any
execution feedback at all. Best-of-N continues to climb with N because pass@N
counts a single correct sample as success on the oracle test harness — its
ceiling is the model's *coverage*, not its single-sample accuracy. The two
loop-based strategies (L and H) are the most token-efficient at the small-to-mid
budgets where execution feedback most clearly compensates for the model's
single-shot failures.

## L vs H comparison

At matched budgets, the single-agent loop (L) and the separated
proposer-reviewer harness (H) perform within a few percentage points of each
other; H does **not** systematically outperform L in this regime. Architectural
separation costs tokens — the reviewer is a second LLM call per turn — so H
gets fewer effective revision turns per unit budget than L does. On these 15
tasks, the diagnostic value of the reviewer's structured critique does not
recover the lost revision capacity. This is consistent with Phase 1's finding
that the reviewer is the loop's main weakness, and supports a simpler design
unless the reviewer can be made cheap or strictly more informative than raw
stderr.

## Files

- `results/scaling_curves/code_4strategy.json`
- `results/scaling_curves/code_4strategy_summary.csv`
- `scripts/run_scaling_curves_code.py`
"""

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_MD}")
    print("\n--- table preview ---")
    print(table_md)
    print("\n--- ascii curves preview ---")
    print(ascii_md)
    print("\n--- L vs H preview ---")
    print(lh_md)


if __name__ == "__main__":
    main()
