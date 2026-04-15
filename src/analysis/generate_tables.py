"""Generate result tables and analysis from experiment data."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def load_baseline_results(results_dir: Path | None = None) -> pd.DataFrame:
    """Load all baseline results from JSON files into a DataFrame."""
    results_dir = results_dir or RESULTS_DIR / "baselines" / "humaneval"

    records = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)

        if "results" in data:
            for r in data["results"]:
                records.append(r)

    if not records:
        logger.warning("No results found in %s", results_dir)
        return pd.DataFrame()

    return pd.DataFrame(records)


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary table of pass@1, tokens, and time per baseline."""
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("baseline").agg(
        num_problems=("problem_id", "count"),
        pass_at_1=("passed", "mean"),
        avg_tokens=("total_tokens", "mean"),
        total_tokens=("total_tokens", "sum"),
        avg_iterations=("iterations", "mean"),
        avg_time_s=("wall_time_seconds", "mean"),
        total_time_s=("wall_time_seconds", "sum"),
    ).round(3)

    return summary.sort_values("pass_at_1", ascending=False)


def generate_per_problem_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-problem pass/fail comparison across baselines."""
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index="problem_id",
        columns="baseline",
        values="passed",
        aggfunc="first",
    ).fillna(False).astype(int)

    return pivot


def print_results_table(results_dir: Path | None = None):
    """Print a formatted results table."""
    df = load_baseline_results(results_dir)
    if df.empty:
        print("No results found.")
        return

    summary = generate_summary_table(df)
    print("\n" + "=" * 90)
    print("Results Summary: HumanEval+")
    print("=" * 90)
    print(f"{'Baseline':<35} {'Pass@1':>8} {'Avg Tok':>10} {'Avg Iter':>10} {'Avg Time':>10}")
    print("-" * 90)
    for name, row in summary.iterrows():
        print(
            f"{name:<35} {row['pass_at_1']:>8.3f} "
            f"{row['avg_tokens']:>10.0f} "
            f"{row['avg_iterations']:>10.1f} "
            f"{row['avg_time_s']:>10.1f}s"
        )
    print("=" * 90)

    # Per-problem analysis
    comparison = generate_per_problem_comparison(df)
    baselines = list(comparison.columns)

    if len(baselines) >= 2:
        print(f"\nPer-problem comparison ({len(comparison)} problems):")
        for b in baselines:
            print(f"  {b}: {comparison[b].sum()}/{len(comparison)} passed")

        # Find problems where interaction scaling helps
        if "B1_single_no_review" in baselines and "B5_agentic_loop" in baselines:
            b1_fails = comparison[comparison["B1_single_no_review"] == 0]
            b5_fixes = b1_fails[b1_fails["B5_agentic_loop"] == 1]
            print(f"\nProblems fixed by agentic loop (B5) that B1 missed: {len(b5_fixes)}")
            if not b5_fixes.empty:
                print(f"  {list(b5_fixes.index[:10])}")

        if "B5_agentic_loop" in baselines and "ours_type3_fixed" in baselines:
            b5_fails = comparison[comparison["B5_agentic_loop"] == 0]
            ours_fixes = b5_fails[b5_fails["ours_type3_fixed"] == 1]
            print(f"\nProblems fixed by proposer-reviewer that B5 missed: {len(ours_fixes)}")
            if not ours_fixes.empty:
                print(f"  {list(ours_fixes.index[:10])}")


def save_analysis(results_dir: Path | None = None, output_dir: Path | None = None):
    """Save analysis outputs."""
    results_dir = results_dir or RESULTS_DIR / "baselines" / "humaneval"
    output_dir = output_dir or RESULTS_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_baseline_results(results_dir)
    if df.empty:
        return

    summary = generate_summary_table(df)
    summary.to_csv(output_dir / "summary_table.csv")

    comparison = generate_per_problem_comparison(df)
    comparison.to_csv(output_dir / "per_problem_comparison.csv")

    # Save as formatted markdown
    with open(output_dir / "results.md", "w") as f:
        f.write("# HumanEval+ Results\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary.to_markdown() + "\n\n")
        f.write(f"## Per-Problem Results ({len(comparison)} problems)\n\n")
        f.write(comparison.to_markdown() + "\n\n")

    logger.info("Analysis saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_results_table()
