"""Generate on-policy training traces using local Qwen3-8B.

Uses Qwen3-8B as the proposer (preserving <think> tokens) and
grounded feedback (execution, VLM, factual) as the reviewer.

This produces on-policy SFT data where the model's own thinking
traces are captured, unlike off-policy distillation from Claude.

Usage:
    python3 run_onpolicy_augmentation.py --runs 3 --temperature 0.7
    python3 run_onpolicy_augmentation.py --runs 3 --temperature 0.7 --categories code,video
    python3 run_onpolicy_augmentation.py --model-path Qwen/Qwen3-8B --runs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/hard_benchmarks")
RESULTS_DIR = Path("results/hard_benchmarks")

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
        "method": "run_slide_task",
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


def run_onpolicy_pass(
    run_id: int,
    temperature: float,
    categories_to_run: list[str],
    model_path: str,
):
    """Run one on-policy augmentation pass with local Qwen3-8B."""
    from src.config import ExperimentConfig, ModelConfig, ModelProvider
    from src.experiments.hard_benchmark_runner import HardBenchmarkRunner

    # Use local Qwen3-8B as the proposer
    proposer = ModelConfig.qwen3_8b(model_path)
    proposer.temperature = temperature

    # Use Claude Sonnet as the reviewer (for VLM feedback etc.)
    # For code tasks, the reviewer is execution-based (no LLM needed)
    reviewer = ModelConfig.claude_sonnet()

    config = ExperimentConfig(
        name=f"onpolicy_run{run_id}",
        benchmark="hard",
        budget_tokens=500_000,
        proposer_model=proposer,
        reviewer_model=reviewer,
    )
    runner = HardBenchmarkRunner(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cat_name in categories_to_run:
        cat_config = CATEGORIES[cat_name]
        task_file = cat_config["task_file"]
        method_name = cat_config["method"]
        max_iters = cat_config["max_iters"]

        if not task_file.exists():
            logger.warning("Task file not found: %s", task_file)
            continue

        with open(task_file) as f:
            tasks = json.load(f)

        run_fn = getattr(runner, method_name)
        results = []

        logger.info("=== On-policy run %d: %s (model=%s, temp=%.1f, %d tasks) ===",
                     run_id, cat_name, model_path, temperature, len(tasks))

        for task in tasks:
            task_id = task["task_id"]
            logger.info("  %s / %s", cat_name, task_id)

            ss_result = None
            rv_result = None

            # Single-shot
            try:
                ss_result = run_fn(task, use_review=False, max_iterations=1)
            except Exception as e:
                logger.error("  SS failed: %s", e)

            # Reviewed
            try:
                rv_result = run_fn(task, use_review=True, max_iterations=max_iters)
            except Exception as e:
                logger.error("  RV failed: %s", e)

            entry = {
                "task_id": task_id,
                "category": cat_name,
                "model": model_path,
                "ss_quality": getattr(ss_result, "quality_score", None) if ss_result else None,
                "ss_meets": getattr(ss_result, "meets_requirements", None) if ss_result else None,
                "ss_tokens": getattr(ss_result, "total_tokens", 0) if ss_result else 0,
                "rv_quality": getattr(rv_result, "quality_score", None) if rv_result else None,
                "rv_meets": getattr(rv_result, "meets_requirements", None) if rv_result else None,
                "rv_iters": getattr(rv_result, "iterations", None) if rv_result else None,
                "rv_tokens": getattr(rv_result, "total_tokens", 0) if rv_result else 0,
            }

            # Save actual artifacts (including <think> traces) for training data
            if ss_result:
                entry["ss_final_code"] = getattr(ss_result, "final_code", "")
                # Capture full generation including thinking
                entry["ss_full_output"] = getattr(ss_result, "full_output", "")
            if rv_result:
                entry["rv_final_code"] = getattr(rv_result, "final_code", "")
                entry["rv_full_output"] = getattr(rv_result, "full_output", "")
                # Capture all intermediate outputs for multi-step traces
                entry["rv_intermediate_outputs"] = getattr(rv_result, "intermediate_outputs", [])

            results.append(entry)

        # Save results for this run
        out_file = RESULTS_DIR / f"{cat_name}_onpolicy_run{run_id}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %s on-policy run %d results to %s", cat_name, run_id, out_file)


def main():
    parser = argparse.ArgumentParser(description="Generate on-policy traces with Qwen3-8B")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (default: all)")
    parser.add_argument("--start-run", type=int, default=1, help="Starting run ID")
    parser.add_argument("--model-path", type=str, default="google/gemma-4-31B-it",
                        help="Path to local model (default: Gemma 4 31B for SFT traces)")
    args = parser.parse_args()

    categories_to_run = list(CATEGORIES.keys())
    if args.categories:
        categories_to_run = [c.strip() for c in args.categories.split(",")]

    for run_id in range(args.start_run, args.start_run + args.runs):
        logger.info("\n" + "=" * 60)
        logger.info("ON-POLICY RUN %d / %d (model=%s, temp=%.1f)",
                     run_id, args.runs, args.model_path, args.temperature)
        logger.info("=" * 60)
        run_onpolicy_pass(run_id, args.temperature, categories_to_run, args.model_path)


if __name__ == "__main__":
    main()
