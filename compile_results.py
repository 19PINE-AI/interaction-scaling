#!/usr/bin/env python3
"""Compile all hard benchmark results into a unified analysis.

Merges batch results, generates summary tables, and produces
publication-ready output.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
RESULTS_DIR = Path("results/hard_benchmarks")


def load_all_results() -> dict[str, list[dict]]:
    """Load and merge all result files by category."""
    categories = defaultdict(list)
    seen_ids = defaultdict(set)

    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            continue

        for entry in data:
            tid = entry.get("task_id", "")

            # Determine category from task_id or filename
            if tid.startswith("code_"):
                cat = "code_swe" if "swe" in f.name else "code_old"
            elif tid.startswith("slide_"):
                cat = "slides"
            elif tid.startswith("web_"):
                cat = "webpages"
            elif tid.startswith("anim_"):
                cat = "animations"
            elif tid.startswith("video_"):
                cat = "video"
            elif tid.startswith("research_"):
                cat = "research"
            else:
                continue

            # Deduplicate (keep latest)
            if tid not in seen_ids[cat]:
                seen_ids[cat].add(tid)
                categories[cat].append(entry)

    return dict(categories)


def compute_summary(entries: list[dict], category: str) -> dict:
    """Compute summary statistics for a category."""
    n = len(entries)
    if n == 0:
        return {}

    # Normalize key names across batches
    normalized = []
    for e in entries:
        ne = dict(e)
        # Handle old-format keys
        if "single_shot_quality" in ne and "ss_quality" not in ne:
            ne["ss_quality"] = ne["single_shot_quality"]
            ne["ss_meets"] = ne.get("single_shot_meets", False)
            ne["rv_quality"] = ne.get("reviewed_quality", 0)
            ne["rv_meets"] = ne.get("reviewed_meets", False)
        if "single_shot_passed" in ne and "reviewed_passed" not in ne:
            ne["reviewed_passed"] = ne.get("reviewed_passed", False)
        normalized.append(ne)
    entries = normalized

    # Detect metric type from keys
    if "single_shot_passed" in entries[0] or "reviewed_passed" in entries[0]:
        ss_pass = sum(1 for e in entries if e.get("single_shot_passed", False))
        rv_pass = sum(1 for e in entries if e.get("reviewed_passed", False))
        fixed = sum(
            1 for e in entries
            if not e["single_shot_passed"] and e["reviewed_passed"]
        )
        return {
            "category": category,
            "n": n,
            "metric": "pass_rate",
            "single_shot": f"{ss_pass}/{n} ({ss_pass/n:.0%})",
            "reviewed": f"{rv_pass}/{n} ({rv_pass/n:.0%})",
            "ss_value": ss_pass / n,
            "rv_value": rv_pass / n,
            "delta": f"+{(rv_pass-ss_pass)/n*100:.0f}pp",
            "delta_value": (rv_pass - ss_pass) / n,
            "fixed": fixed,
        }
    elif "ss_quality" in entries[0]:
        ss_avg = sum(e["ss_quality"] for e in entries) / n
        rv_vals = [e["rv_quality"] for e in entries if e.get("rv_quality") is not None]
        rv_avg = sum(rv_vals) / len(rv_vals) if rv_vals else 0
        ss_meets = sum(1 for e in entries if e.get("ss_meets"))
        rv_meets = sum(1 for e in entries if e.get("rv_meets"))
        return {
            "category": category,
            "n": n,
            "metric": "quality",
            "single_shot": f"{ss_avg:.2f} ({ss_meets}/{n} meets)",
            "reviewed": f"{rv_avg:.2f} ({rv_meets}/{n} meets)",
            "ss_value": ss_avg,
            "rv_value": rv_avg,
            "delta": f"+{rv_avg-ss_avg:.2f}",
            "delta_value": rv_avg - ss_avg,
        }
    elif "ss_accuracy" in entries[0]:
        ss_avg = sum(e["ss_accuracy"] for e in entries) / n
        rv_avg = sum(e["rv_accuracy"] for e in entries) / n
        return {
            "category": category,
            "n": n,
            "metric": "accuracy",
            "single_shot": f"{ss_avg:.2f}",
            "reviewed": f"{rv_avg:.2f}",
            "ss_value": ss_avg,
            "rv_value": rv_avg,
            "delta": f"+{rv_avg-ss_avg:.2f}",
            "delta_value": rv_avg - ss_avg,
        }
    return {}


def print_results():
    """Print comprehensive results table."""
    categories = load_all_results()

    # Category display order and labels
    order = [
        ("code_swe", "Code (SWE-style)", "Type 3a: Execution"),
        ("research", "Deep Research", "Type 3d: Factual"),
        ("webpages", "Web Pages", "Type 3b: Visual"),
        ("slides", "Slides", "Type 3b: Visual"),
        ("animations", "Animations", "Type 3c: Temporal"),
        ("video", "Video Editing", "Type 3c: Temporal"),
    ]

    print("\n" + "=" * 100)
    print(" INTERACTION SCALING — HARD BENCHMARK RESULTS")
    print("=" * 100)
    print(
        f"{'Category':<22} {'Feedback':<22} {'N':>4} "
        f"{'Single-shot':>16} {'Reviewed':>16} {'Delta':>10}"
    )
    print("-" * 100)

    summaries = []
    for cat_key, cat_label, feedback in order:
        if cat_key not in categories:
            print(f"{cat_label:<22} {feedback:<22} {'—':>4} {'—':>16} {'—':>16} {'—':>10}")
            continue

        summary = compute_summary(categories[cat_key], cat_label)
        summaries.append(summary)

        print(
            f"{cat_label:<22} {feedback:<22} {summary['n']:>4} "
            f"{summary['single_shot']:>16} {summary['reviewed']:>16} "
            f"{summary['delta']:>10}"
        )

    print("=" * 100)

    # Per-task breakdown
    print("\n" + "=" * 100)
    print(" PER-TASK BREAKDOWN")
    print("=" * 100)

    for cat_key, cat_label, _ in order:
        if cat_key not in categories:
            continue
        entries = categories[cat_key]
        print(f"\n--- {cat_label} ({len(entries)} tasks) ---")

        for e in entries:
            tid = e.get("task_id", "?")
            if "single_shot_passed" in e:
                ss = "PASS" if e["single_shot_passed"] else "FAIL"
                rv = "PASS" if e["reviewed_passed"] else "FAIL"
                status = "FIXED" if ss == "FAIL" and rv == "PASS" else (
                    "BOTH_PASS" if ss == "PASS" else "BOTH_FAIL"
                )
                print(f"  {tid}: {status} (ss={ss}, rv={rv}, iters={e.get('reviewed_iters', '?')})")
            elif "ss_quality" in e:
                ss = e["ss_quality"]
                rv = e.get("rv_quality", "?")
                delta = f"+{rv-ss:.2f}" if isinstance(rv, (int, float)) else "?"
                print(f"  {tid}: ss={ss:.2f} rv={rv} ({delta})")
            elif "ss_accuracy" in e:
                ss = e["ss_accuracy"]
                rv = e["rv_accuracy"]
                print(f"  {tid}: ss={ss:.2f} rv={rv:.2f} (+{rv-ss:.2f})")

    # Save compiled results
    output = {
        "summaries": summaries,
        "categories": {k: v for k, v in categories.items()},
    }
    with open(RESULTS_DIR / "compiled_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nCompiled results saved to {RESULTS_DIR / 'compiled_results.json'}")


if __name__ == "__main__":
    print_results()
