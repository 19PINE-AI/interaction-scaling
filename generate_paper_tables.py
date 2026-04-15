#!/usr/bin/env python3
"""Generate publication-ready tables from experiment results.

Produces LaTeX and Markdown tables for the paper.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")


def load_summary(filepath: Path) -> dict:
    """Load a result JSON and return its summary."""
    with open(filepath) as f:
        data = json.load(f)

    results = data.get("results", [])
    n = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_time = sum(r.get("wall_time_seconds", 0) for r in results)
    avg_iters = sum(r.get("iterations", 1) for r in results) / max(n, 1)

    failed = [r["problem_id"] for r in results if not r.get("passed")]

    return {
        "n": n,
        "passed": passed,
        "pass_at_1": passed / n if n else 0,
        "avg_tokens": total_tokens / n if n else 0,
        "avg_time": total_time / n if n else 0,
        "avg_iters": avg_iters,
        "failed": failed,
    }


CONDITION_LABELS = {
    "B1_single_no_review": "B1: Single-shot",
    "B2_self_review": "B2: Self-review (Type 0)",
    "B3_cross_model": "B3: Cross-model (Type 1)",
    "B4_best_of_n": "B4: Best-of-N",
    "B5_agentic_loop": "B5: Agentic Loop",
    "ours_type3_fixed": "Ours: PR-Fixed (Type 3)",
    "ours_type3_adaptive": "Ours: PR-Adaptive",
    "ours_type3_confidence": "Ours: PR-Confidence",
    "ours_type23_fixed": "Ours: PR Type 2+3",
}

CONDITION_ORDER = list(CONDITION_LABELS.keys())


def generate_table(results_dir: Path, title: str):
    """Generate a results table from all JSON files in a directory."""
    if not results_dir.exists():
        return

    entries = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name in ("summary.json",):
            continue
        condition = json_file.stem
        summary = load_summary(json_file)
        entries.append((condition, summary))

    if not entries:
        return

    # Sort by condition order
    entries.sort(key=lambda x: CONDITION_ORDER.index(x[0]) if x[0] in CONDITION_ORDER else 999)

    print(f"\n## {title}\n")

    # Markdown table
    print(f"| {'Condition':<35} | {'N':>5} | {'Pass@1':>8} | {'Avg Tok':>10} | {'Avg Iter':>8} | {'Avg Time':>8} |")
    print(f"|{'-'*37}|{'-'*7}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*10}|")
    for condition, s in entries:
        label = CONDITION_LABELS.get(condition, condition)
        print(
            f"| {label:<35} | {s['n']:>5} | {s['pass_at_1']:>7.1%} | "
            f"{s['avg_tokens']:>10.0f} | {s['avg_iters']:>8.1f} | {s['avg_time']:>7.1f}s |"
        )

    # LaTeX table
    print(f"\n% LaTeX version")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Condition & Pass@1 & Avg Tokens & Avg Iter & Avg Time \\")
    print(r"\midrule")
    for condition, s in entries:
        label = CONDITION_LABELS.get(condition, condition)
        print(
            f"{label} & {s['pass_at_1']:.1%} & {s['avg_tokens']:.0f} & "
            f"{s['avg_iters']:.1f} & {s['avg_time']:.1f}s \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{{title}}}")
    print(r"\end{table}")

    # Failed problems analysis
    print(f"\n### Failed Problems")
    for condition, s in entries:
        if s["failed"]:
            label = CONDITION_LABELS.get(condition, condition)
            print(f"- **{label}** ({len(s['failed'])} failures): {s['failed']}")


def main():
    print("# Interaction Scaling: Experiment Results")
    print(f"\nGenerated from: {RESULTS_DIR.absolute()}")

    generate_table(RESULTS_DIR / "full_v2" / "humaneval", "HumanEval+ Full (164 problems, with fix)")
    generate_table(RESULTS_DIR / "full" / "humaneval", "HumanEval+ Full (164 problems, pre-fix)")
    generate_table(RESULTS_DIR / "baselines_v2" / "humaneval", "HumanEval+ (50 problems, with B2)")
    generate_table(RESULTS_DIR / "baselines" / "mbpp", "MBPP+ (50 problems)")
    generate_table(RESULTS_DIR / "baselines" / "humaneval", "HumanEval+ Baselines (50, pre-fix)")


if __name__ == "__main__":
    main()
