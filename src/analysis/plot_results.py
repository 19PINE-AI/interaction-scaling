"""Generate plots for the interaction scaling paper."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)

# Paper-quality figure settings
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent colors for each condition
COLORS = {
    "B1_single_no_review": "#888888",
    "B2_self_review": "#c44e52",
    "B3_cross_model": "#dd8452",
    "B4_best_of_n": "#937860",
    "B5_agentic_loop": "#4c72b0",
    "ours_type3_fixed": "#55a868",
    "ours_type3_adaptive": "#2ca02c",
    "ours_type3_confidence": "#1f8b4c",
    "ours_type23_fixed": "#17becf",
    # Feedback ablation
    "type0_self_review": "#c44e52",
    "type1_cross_model": "#dd8452",
    "type2_static": "#ccb974",
    "type3_execution": "#55a868",
    "type23_combined": "#17becf",
}

LABELS = {
    "B1_single_no_review": "B1: Single-shot",
    "B2_self_review": "B2: Self-review (Type 0)",
    "B3_cross_model": "B3: Cross-model (Type 1)",
    "B4_best_of_n": "B4: Best-of-N",
    "B5_agentic_loop": "B5: Agentic loop",
    "ours_type3_fixed": "Ours: P-R Fixed",
    "ours_type3_adaptive": "Ours: P-R Adaptive",
    "ours_type3_confidence": "Ours: P-R Confidence",
    "ours_type23_fixed": "Ours: P-R Type 2+3",
    "type0_self_review": "Type 0: Self-review",
    "type1_cross_model": "Type 1: Cross-model",
    "type2_static": "Type 2: Static analysis",
    "type3_execution": "Type 3: Execution",
    "type23_combined": "Type 2+3: Combined",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame."""
    records = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)
        if "results" in data:
            for r in data["results"]:
                records.append(r)
    return pd.DataFrame(records)


def plot_baseline_comparison(results_dir: Path | None = None, output_dir: Path | None = None):
    """Plot bar chart comparing pass@1 across baselines."""
    results_dir = results_dir or RESULTS_DIR / "baselines" / "humaneval"
    output_dir = output_dir or RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        return

    summary = df.groupby("baseline").agg(
        pass_at_1=("passed", "mean"),
        avg_tokens=("total_tokens", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(summary)),
        summary["pass_at_1"],
        color=[COLORS.get(b, "#888888") for b in summary["baseline"]],
    )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(
        [LABELS.get(b, b) for b in summary["baseline"]],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Pass@1")
    ax.set_title("HumanEval+ Pass@1 by Approach")
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, val in zip(bars, summary["pass_at_1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png")
    logger.info("Saved baseline comparison plot to %s", output_dir)
    plt.close(fig)


def plot_tokens_vs_performance(results_dir: Path | None = None, output_dir: Path | None = None):
    """Plot scatter of average tokens vs pass@1 (efficiency frontier)."""
    results_dir = results_dir or RESULTS_DIR / "baselines" / "humaneval"
    output_dir = output_dir or RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        return

    summary = df.groupby("baseline").agg(
        pass_at_1=("passed", "mean"),
        avg_tokens=("total_tokens", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        b = row["baseline"]
        ax.scatter(
            row["avg_tokens"],
            row["pass_at_1"],
            color=COLORS.get(b, "#888888"),
            label=LABELS.get(b, b),
            s=120,
            zorder=5,
        )

    ax.set_xlabel("Average Tokens per Problem")
    ax.set_ylabel("Pass@1")
    ax.set_title("Token Efficiency: Pass@1 vs Average Tokens")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "tokens_vs_performance.png")
    logger.info("Saved tokens vs performance plot to %s", output_dir)
    plt.close(fig)


def plot_feedback_ablation(results_dir: Path | None = None, output_dir: Path | None = None):
    """Plot feedback type ablation results."""
    results_dir = results_dir or RESULTS_DIR / "exp1_feedback_ablation"
    output_dir = output_dir or RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        return

    summary = df.groupby("baseline").agg(
        pass_at_1=("passed", "mean"),
    ).reset_index()

    # Order by feedback type
    order = ["type0_self_review", "type1_cross_model", "type2_static",
             "type3_execution", "type23_combined"]
    summary = summary.set_index("baseline").reindex(order).reset_index()
    summary = summary.dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(summary)),
        summary["pass_at_1"],
        color=[COLORS.get(b, "#888888") for b in summary["baseline"]],
    )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(
        [LABELS.get(b, b) for b in summary["baseline"]],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Pass@1")
    ax.set_title("Feedback Type Ablation (HumanEval+)")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, summary["pass_at_1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "feedback_ablation.png")
    logger.info("Saved feedback ablation plot to %s", output_dir)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_baseline_comparison()
    plot_tokens_vs_performance()
    plot_feedback_ablation()
