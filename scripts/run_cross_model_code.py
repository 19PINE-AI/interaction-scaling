"""Cross-model replication of the Phase-1 code modality.

Same harness (HardBenchmarkRunner.run_code_task), same 15 Phase-1 code
tasks (data/hard_benchmarks/code/code_tasks.json), single-shot vs
reviewed. Only the proposer/reviewer model is swapped from Claude
Sonnet 4 to a non-Anthropic model (default: Qwen3-235B-Instruct-2507
via OpenRouter).

Outputs results/cross_model/code_<modelname>.json with per-task
records: task_id, single_shot_passed, reviewed_passed, model, tokens,
iterations.

Usage:
    python3 scripts/run_cross_model_code.py --model qwen3-235b --workers 8
"""
import argparse
import concurrent.futures as futures
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_FILE = Path("data/hard_benchmarks/code/code_tasks.json")
OUT_DIR = Path("results/cross_model")
MAX_ITERS = 5  # matches the Claude reference config


def make_proposer(model_name: str):
    from src.config import ModelConfig
    mapping = {
        "qwen3-235b": ModelConfig.qwen3_235b,
        "deepseek-r1": ModelConfig.deepseek_r1,
        "gpt-5": ModelConfig.gpt5,
        "claude-sonnet-thinking": ModelConfig.claude_sonnet_thinking,
        "claude-sonnet": ModelConfig.claude_sonnet,
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model {model_name}; choose from {sorted(mapping)}")
    return mapping[model_name]()


def run_one(task: dict, model_name: str) -> dict:
    """Run single-shot and reviewed on one task; return summary record."""
    from src.config import ExperimentConfig
    from src.experiments.hard_benchmark_runner import HardBenchmarkRunner

    proposer = make_proposer(model_name)
    reviewer = make_proposer(model_name)  # same model for review

    config = ExperimentConfig(
        name=f"cross_model_{model_name}",
        benchmark="hard",
        budget_tokens=500_000,
        proposer_model=proposer,
        reviewer_model=reviewer,
    )
    runner = HardBenchmarkRunner(config)

    task_id = task["task_id"]
    record = {
        "task_id": task_id,
        "model": model_name,
        "category": "code",
    }

    # Single-shot
    t0 = time.time()
    try:
        ss = runner.run_code_task(task, use_review=False, max_iterations=1)
        record["single_shot_passed"] = bool(ss.meets_requirements)
        record["ss_tokens"] = int(ss.total_tokens)
        record["ss_iterations"] = int(ss.iterations)
        record["ss_wall_seconds"] = float(ss.wall_time_seconds)
        record["ss_final_code"] = ss.final_code
    except Exception as e:
        logger.exception("SS failed for %s: %s", task_id, e)
        record["single_shot_passed"] = None
        record["ss_error"] = str(e)

    # Reviewed (execution feedback up to MAX_ITERS)
    try:
        rv = runner.run_code_task(task, use_review=True, max_iterations=MAX_ITERS)
        record["reviewed_passed"] = bool(rv.meets_requirements)
        record["rv_tokens"] = int(rv.total_tokens)
        record["rv_iterations"] = int(rv.iterations)
        record["rv_wall_seconds"] = float(rv.wall_time_seconds)
        record["rv_final_code"] = rv.final_code
    except Exception as e:
        logger.exception("RV failed for %s: %s", task_id, e)
        record["reviewed_passed"] = None
        record["rv_error"] = str(e)

    record["total_wall_seconds"] = time.time() - t0
    logger.info(
        "[%s] SS=%s RV=%s rv_iters=%s rv_tok=%s",
        task_id,
        record.get("single_shot_passed"),
        record.get("reviewed_passed"),
        record.get("rv_iterations"),
        record.get("rv_tokens"),
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-235b",
                        help="qwen3-235b | deepseek-r1 | claude-sonnet | claude-sonnet-thinking")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads (OpenRouter handles concurrency well)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N tasks (for smoke tests)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"code_{args.model}.json"

    with open(DATA_FILE) as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[: args.limit]

    logger.info("Running %d code tasks with model=%s workers=%d",
                len(tasks), args.model, args.workers)
    t0 = time.time()

    # Use threads, not processes: HardBenchmarkRunner's only real cost is the
    # HTTP call to OpenRouter, which releases the GIL.
    results: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_tid = {ex.submit(run_one, t, args.model): t["task_id"] for t in tasks}
        for fut in futures.as_completed(fut_to_tid):
            tid = fut_to_tid[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logger.exception("worker for %s crashed: %s", tid, e)
                results.append({"task_id": tid, "model": args.model, "error": str(e)})
            # Incremental save in case the run is interrupted
            results_sorted = sorted(results, key=lambda r: r["task_id"])
            with open(out_file, "w") as f:
                json.dump(results_sorted, f, indent=2)

    elapsed = time.time() - t0

    # Final aggregate
    ss_pass = sum(1 for r in results if r.get("single_shot_passed"))
    rv_pass = sum(1 for r in results if r.get("reviewed_passed"))
    n = len(results)
    logger.info(
        "DONE %d tasks in %.1fs. SS pass=%d/%d (%.1f%%)  RV pass=%d/%d (%.1f%%)  Wrote %s",
        n, elapsed,
        ss_pass, n, 100 * ss_pass / max(n, 1),
        rv_pass, n, 100 * rv_pass / max(n, 1),
        out_file,
    )


if __name__ == "__main__":
    main()
