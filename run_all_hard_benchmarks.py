"""Run ALL hard benchmarks across all 6 categories.

Runs each task in both single-shot and reviewed (proposer-reviewer) modes,
saves per-category results, and compiles a summary table.

Usage:
    python run_all_hard_benchmarks.py [--categories code,slides,...] [--model sonnet|opus]
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/hard_benchmarks")
RESULTS_DIR = Path("results/hard_benchmarks")

# Category configs: (task_file, runner_method, max_iters_reviewed)
CATEGORIES = {
    "code": {
        "task_file": DATA_DIR / "code" / "code_tasks.json",
        "method": "run_code_task",
        "max_iters": 5,
    },
    "slides": {
        "task_file": DATA_DIR / "slides" / "slide_tasks.json",
        "method": "run_slide_task",
        "max_iters": 3,
    },
    "webpages": {
        "task_file": DATA_DIR / "webpages" / "webpage_tasks.json",
        "method": "run_slide_task",  # shared HTML pipeline
        "max_iters": 3,
    },
    "animations": {
        "task_file": DATA_DIR / "animations" / "animation_tasks.json",
        "method": "run_animation_task",
        "max_iters": 3,
    },
    "video": {
        "task_file": DATA_DIR / "video" / "video_tasks.json",
        "method": "run_video_task",
        "max_iters": 3,
    },
    "research": {
        "task_file": DATA_DIR / "research" / "research_tasks.json",
        "method": "run_research_task",
        "max_iters": 2,
    },
}


def run_category(category_name: str, cat_config: dict, runner, timeout_s: int = 300):
    """Run all tasks in a single category."""
    task_file = cat_config["task_file"]
    method_name = cat_config["method"]
    max_iters = cat_config["max_iters"]

    with open(task_file) as f:
        tasks = json.load(f)

    run_fn = getattr(runner, method_name)
    results = []

    for task in tasks:
        task_id = task["task_id"]
        logger.info("=== %s / %s ===", category_name, task_id)

        # Single-shot
        ss_result = None
        rv_result = None
        try:
            ss_result = run_fn(task, use_review=False, max_iterations=1)
            ss_quality = ss_result.quality_score
            ss_meets = ss_result.meets_requirements
            ss_tokens = ss_result.total_tokens
            logger.info(
                "  Single-shot: quality=%.2f, meets=%s, tokens=%d",
                ss_quality, ss_meets, ss_tokens,
            )
        except Exception as e:
            logger.error("  Single-shot FAILED: %s", e)
            traceback.print_exc()
            ss_quality = None
            ss_meets = None
            ss_tokens = 0

        # Reviewed
        try:
            rv_result = run_fn(task, use_review=True, max_iterations=max_iters)
            rv_quality = rv_result.quality_score
            rv_meets = rv_result.meets_requirements
            rv_iters = rv_result.iterations
            rv_tokens = rv_result.total_tokens
            logger.info(
                "  Reviewed: quality=%.2f, meets=%s, iters=%d, tokens=%d",
                rv_quality, rv_meets, rv_iters, rv_tokens,
            )
        except Exception as e:
            logger.error("  Reviewed FAILED: %s", e)
            traceback.print_exc()
            rv_quality = None
            rv_meets = None
            rv_iters = None
            rv_tokens = 0

        entry = {
            "task_id": task_id,
            "category": category_name,
            "ss_quality": ss_quality,
            "ss_meets": ss_meets,
            "ss_tokens": ss_tokens,
            "rv_quality": rv_quality,
            "rv_meets": rv_meets,
            "rv_iters": rv_iters,
            "rv_tokens": rv_tokens,
        }

        # Save actual generated artifacts for training data collection
        if ss_result is not None:
            entry["ss_final_code"] = getattr(ss_result, "final_code", "")
        if rv_result is not None:
            entry["rv_final_code"] = getattr(rv_result, "final_code", "")

        results.append(entry)

    return results


def summarize_category(name: str, results: list[dict]) -> dict:
    """Compute summary statistics for a category."""
    valid = [
        r for r in results
        if r.get("ss_quality") is not None and r.get("rv_quality") is not None
    ]
    n = len(valid)
    if n == 0:
        return {"category": name, "n": 0, "avg_ss": None, "avg_rv": None, "delta": None}

    avg_ss = sum(r["ss_quality"] for r in valid) / n
    avg_rv = sum(r["rv_quality"] for r in valid) / n
    return {
        "category": name,
        "n": n,
        "total_tasks": len(results),
        "avg_ss": round(avg_ss, 3),
        "avg_rv": round(avg_rv, 3),
        "delta": round(avg_rv - avg_ss, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Run all hard benchmarks")
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to run (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["sonnet", "opus"],
        default="sonnet",
        help="Model to use for proposer/reviewer",
    )
    args = parser.parse_args()

    # Setup
    from src.config import ExperimentConfig, ModelConfig
    from src.experiments.hard_benchmark_runner import HardBenchmarkRunner

    if args.model == "opus":
        model = ModelConfig.claude_opus()
    else:
        model = ModelConfig.claude_sonnet()

    config = ExperimentConfig(
        name="hard_all",
        benchmark="hard",
        budget_tokens=500_000,
        proposer_model=model,
        reviewer_model=model,
    )
    runner = HardBenchmarkRunner(config)

    categories_to_run = list(CATEGORIES.keys())
    if args.categories:
        categories_to_run = [c.strip() for c in args.categories.split(",")]
        for c in categories_to_run:
            if c not in CATEGORIES:
                logger.error("Unknown category: %s. Valid: %s", c, list(CATEGORIES.keys()))
                sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    summaries = []

    start_total = time.time()

    for cat_name in categories_to_run:
        cat_config = CATEGORIES[cat_name]
        logger.info("\n" + "=" * 60)
        logger.info("CATEGORY: %s", cat_name.upper())
        logger.info("=" * 60)

        cat_results = run_category(cat_name, cat_config, runner)
        all_results[cat_name] = cat_results

        # Save per-category results
        out_file = RESULTS_DIR / f"{cat_name}_results.json"
        with open(out_file, "w") as f:
            json.dump(cat_results, f, indent=2)
        logger.info("Saved %s results to %s", cat_name, out_file)

        summary = summarize_category(cat_name, cat_results)
        summaries.append(summary)

    elapsed_total = time.time() - start_total

    # Print summary table
    print(f"\n{'=' * 70}")
    print("HARD BENCHMARK RESULTS — ALL CATEGORIES")
    print(f"{'=' * 70}")
    print(f"Model: {model.model_id}")
    print(f"Total time: {elapsed_total:.0f}s")
    print()
    print(f"{'Category':<15} {'N':>5} {'Single-shot':>12} {'Reviewed':>12} {'Delta':>8}")
    print("-" * 55)
    for s in summaries:
        if s["avg_ss"] is not None:
            print(
                f"{s['category']:<15} {s['n']:>5} "
                f"{s['avg_ss']:>12.3f} {s['avg_rv']:>12.3f} {s['delta']:>+8.3f}"
            )
        else:
            print(f"{s['category']:<15} {'FAILED':>5}")
    print("-" * 55)

    # Compile combined results
    compiled = {
        "model": model.model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_seconds": round(elapsed_total, 1),
        "summaries": summaries,
        "per_category": all_results,
    }
    compiled_path = RESULTS_DIR / "compiled_results.json"
    with open(compiled_path, "w") as f:
        json.dump(compiled, f, indent=2)
    logger.info("Compiled results saved to %s", compiled_path)


if __name__ == "__main__":
    main()
