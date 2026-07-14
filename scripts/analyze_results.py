#!/usr/bin/env python3
"""Analyze and summarize all experiment results.

Usage:
    python scripts/analyze_results.py                    # Print summary tables
    python scripts/analyze_results.py --plots            # Also generate plots
    python scripts/analyze_results.py --save             # Save analysis to files
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_ORDER = [
    "B1_single_no_review",
    "B2_self_review",
    "B3_cross_model",
    "B4_best_of_n",
    "B5_agentic_loop",
    "ours_type3_fixed",
    "ours_type3_adaptive",
    "ours_type3_confidence",
    "ours_type23_fixed",
]


def load_all_results(base_dir: Path) -> pd.DataFrame:
    """Recursively load all JSON result files."""
    records = []
    for json_file in sorted(base_dir.rglob("*.json")):
        if json_file.name in ("summary.json", "scaling_data.json"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                for r in data["results"]:
                    r["source_file"] = str(json_file.relative_to(base_dir))
                    records.append(r)
        except (json.JSONDecodeError, KeyError):
            continue

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame, title: str = "Results"):
    """Print a formatted summary table."""
    if df.empty:
        print(f"No results for: {title}")
        return

    summary = df.groupby("baseline").agg(
        n=("problem_id", "count"),
        pass_at_1=("passed", "mean"),
        avg_tokens=("total_tokens", "mean"),
        avg_iters=("iterations", "mean"),
        avg_time=("wall_time_seconds", "mean"),
    ).round(3)

    # Sort by label order
    order = [b for b in LABEL_ORDER if b in summary.index]
    other = [b for b in summary.index if b not in order]
    summary = summary.reindex(order + other)

    print(f"\n{'=' * 95}")
    print(f" {title}")
    print(f"{'=' * 95}")
    print(f"{'Baseline':<35} {'N':>5} {'Pass@1':>8} {'Avg Tok':>10} {'Avg Iter':>10} {'Avg Time':>10}")
    print(f"{'-' * 95}")
    for name, row in summary.iterrows():
        print(
            f"{name:<35} {row['n']:>5.0f} {row['pass_at_1']:>8.3f} "
            f"{row['avg_tokens']:>10.0f} {row['avg_iters']:>10.1f} {row['avg_time']:>9.1f}s"
        )
    print(f"{'=' * 95}")
    return summary


def analyze_improvement(df: pd.DataFrame):
    """Analyze which problems are fixed by interaction scaling."""
    if df.empty:
        return

    baselines = df["baseline"].unique()
    if "B1_single_no_review" not in baselines:
        return

    b1_results = df[df["baseline"] == "B1_single_no_review"].set_index("problem_id")

    print(f"\n{'=' * 70}")
    print(" Interaction Scaling Analysis")
    print(f"{'=' * 70}")

    for baseline in sorted(baselines):
        if baseline == "B1_single_no_review":
            continue
        other = df[df["baseline"] == baseline].set_index("problem_id")

        # Find common problems
        common = b1_results.index.intersection(other.index)
        if len(common) == 0:
            continue

        b1_pass = b1_results.loc[common, "passed"]
        other_pass = other.loc[common, "passed"]

        fixed = common[(~b1_pass) & other_pass]
        broken = common[b1_pass & (~other_pass)]

        print(f"\n{baseline} vs B1 (on {len(common)} common problems):")
        print(f"  B1 pass: {b1_pass.sum()}/{len(common)} ({b1_pass.mean():.1%})")
        print(f"  {baseline} pass: {other_pass.sum()}/{len(common)} ({other_pass.mean():.1%})")
        print(f"  Fixed by {baseline}: {len(fixed)}")
        if len(fixed) > 0:
            print(f"    {list(fixed[:10])}")
        print(f"  Broken by {baseline}: {len(broken)}")
        if len(broken) > 0:
            print(f"    {list(broken[:10])}")

    # Token efficiency analysis
    print(f"\n{'=' * 70}")
    print(" Token Efficiency Analysis")
    print(f"{'=' * 70}")

    for baseline in sorted(baselines):
        if baseline == "B1_single_no_review":
            continue
        other = df[df["baseline"] == baseline]
        b1 = df[df["baseline"] == "B1_single_no_review"]

        pass_gain = other["passed"].mean() - b1["passed"].mean()
        token_ratio = other["total_tokens"].mean() / max(b1["total_tokens"].mean(), 1)

        print(f"\n{baseline}:")
        print(f"  Pass@1 gain over B1: {pass_gain:+.3f}")
        print(f"  Token ratio vs B1:   {token_ratio:.1f}x")
        if pass_gain > 0:
            tokens_per_point = (
                other["total_tokens"].mean() - b1["total_tokens"].mean()
            ) / (pass_gain * 100)
            print(f"  Extra tokens per 1pp gain: {tokens_per_point:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--save", action="store_true", help="Save analysis files")
    args = parser.parse_args()

    # Load all results
    print("Loading results...")

    # HumanEval baselines (50 problems)
    he50_dir = RESULTS_DIR / "baselines" / "humaneval"
    if he50_dir.exists():
        df_he50 = load_all_results(he50_dir)
        print_summary(df_he50, "HumanEval+ Baselines (50 problems)")
        analyze_improvement(df_he50)

    # HumanEval full v2 (164 problems, with prompt prefix fix)
    he_full_v2_dir = RESULTS_DIR / "full_v2" / "humaneval"
    if he_full_v2_dir.exists():
        df_he_full = load_all_results(he_full_v2_dir)
        print_summary(df_he_full, "HumanEval+ Full (164 problems)")
        analyze_improvement(df_he_full)

    # HumanEval baselines v2 (50 problems, with fix)
    he50_v2_dir = RESULTS_DIR / "baselines_v2" / "humaneval"
    if he50_v2_dir.exists():
        df_he50_v2 = load_all_results(he50_v2_dir)
        print_summary(df_he50_v2, "HumanEval+ Baselines v2 (50 problems)")
        analyze_improvement(df_he50_v2)

    # MBPP baselines
    mbpp_dir = RESULTS_DIR / "baselines" / "mbpp"
    if mbpp_dir.exists():
        df_mbpp = load_all_results(mbpp_dir)
        print_summary(df_mbpp, "MBPP+ Baselines (50 problems)")
        analyze_improvement(df_mbpp)

    if args.plots:
        from src.analysis.plot_results import (
            plot_baseline_comparison,
            plot_tokens_vs_performance,
        )
        if he50_dir.exists():
            plot_baseline_comparison(he50_dir)
            plot_tokens_vs_performance(he50_dir)

    if args.save:
        output_dir = RESULTS_DIR / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving analysis to %s", output_dir)


if __name__ == "__main__":
    main()
