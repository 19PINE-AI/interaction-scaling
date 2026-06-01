"""Generate SFT training traces with thinking/reasoning.

Each task runs as a completely separate OS process to avoid
Playwright async conflicts and maximize parallelism.

Usage:
    python3 run_onpolicy_augmentation.py --runs 3 --workers 16
    python3 run_onpolicy_augmentation.py --runs 3 --categories code,slides
    python3 run_onpolicy_augmentation.py --model claude-sonnet-thinking --runs 3
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
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
    "code_hard": {
        "task_file": DATA_DIR / "code" / "code_tasks_hard.json",
        "method": "run_code_task",
        "max_iters": 5,
    },
    "slides": {
        "task_file": DATA_DIR / "slides" / "slide_tasks.json",
        "method": "run_slide_task",
        "max_iters": 3,
    },
    "slides_hard": {
        "task_file": DATA_DIR / "slides" / "slide_tasks_hard.json",
        "method": "run_slide_task",
        "max_iters": 3,
    },
    "slides_hard2": {
        # real-paper-grounded dense slides (text+figure, aligned pillars)
        "task_file": DATA_DIR / "slides" / "slide_tasks_hard2.json",
        "method": "run_slide_task",
        "max_iters": 3,
    },
    "webpages": {
        "task_file": DATA_DIR / "webpages" / "webpage_tasks.json",
        "method": "run_webpage_task",
        "max_iters": 3,
    },
    "webpages_hard": {
        "task_file": DATA_DIR / "webpages" / "webpage_tasks_hard.json",
        "method": "run_webpage_task",
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
    "video_hard": {
        "task_file": DATA_DIR / "video" / "video_tasks_hard.json",
        "method": "run_video_task",
        "max_iters": 3,
    },
    "video_fixed": {
        # original video suite, regenerated in the fixed env (moviepy present)
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

# Worker script that runs in a completely separate process
WORKER_SCRIPT = '''
import json, sys, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

args = json.loads(sys.argv[1])
task = args["task"]
cat_name = args["cat_name"]
method_name = args["method_name"]
max_iters = args["max_iters"]
model_name = args["model_name"]
temperature = args["temperature"]
run_id = args["run_id"]
output_path = args["output_path"]

from src.config import ExperimentConfig, ModelConfig

model_map = {
    "claude-sonnet-thinking": ModelConfig.claude_sonnet_thinking,
    "claude-sonnet": ModelConfig.claude_sonnet,
    "qwen3-235b": ModelConfig.qwen3_235b,
    "deepseek-r1": ModelConfig.deepseek_r1,
}
proposer = model_map[model_name]()
proposer.temperature = temperature

reviewer = ModelConfig.claude_sonnet()

config = ExperimentConfig(
    name=f"onpolicy_run{run_id}",
    benchmark="hard",
    budget_tokens=500_000,
    proposer_model=proposer,
    reviewer_model=reviewer,
)

from src.experiments.hard_benchmark_runner import HardBenchmarkRunner
runner = HardBenchmarkRunner(config)
run_fn = getattr(runner, method_name)

task_id = task["task_id"]
ss_result = None
rv_result = None

try:
    ss_result = run_fn(task, use_review=False, max_iterations=1)
except Exception as e:
    logging.error("SS failed %s/%s: %s", cat_name, task_id, e)

try:
    rv_result = run_fn(task, use_review=True, max_iterations=max_iters)
except Exception as e:
    logging.error("RV failed %s/%s: %s", cat_name, task_id, e)

entry = {
    "task_id": task_id,
    "category": cat_name,
    "model": model_name,
    "run_id": run_id,
    "ss_quality": getattr(ss_result, "quality_score", None) if ss_result else None,
    "ss_meets": getattr(ss_result, "meets_requirements", None) if ss_result else None,
    "ss_tokens": getattr(ss_result, "total_tokens", 0) if ss_result else 0,
    "rv_quality": getattr(rv_result, "quality_score", None) if rv_result else None,
    "rv_meets": getattr(rv_result, "meets_requirements", None) if rv_result else None,
    "rv_iters": getattr(rv_result, "iterations", None) if rv_result else None,
    "rv_tokens": getattr(rv_result, "total_tokens", 0) if rv_result else 0,
}

if ss_result:
    entry["ss_final_code"] = getattr(ss_result, "final_code", "")
    entry["ss_full_output"] = getattr(ss_result, "full_output", "")
    entry["ss_interaction_trace"] = getattr(ss_result, "interaction_trace", [])
if rv_result:
    entry["rv_final_code"] = getattr(rv_result, "final_code", "")
    entry["rv_full_output"] = getattr(rv_result, "full_output", "")
    entry["rv_intermediate_outputs"] = getattr(rv_result, "intermediate_outputs", [])
    entry["rv_interaction_trace"] = getattr(rv_result, "interaction_trace", [])

with open(output_path, "w") as f:
    json.dump(entry, f)
'''


def main():
    parser = argparse.ArgumentParser(description="Generate SFT traces with parallel OS processes")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (default: all)")
    parser.add_argument("--start-run", type=int, default=1, help="Starting run ID")
    parser.add_argument("--model", type=str, default="claude-sonnet-thinking",
                        help="Model: claude-sonnet-thinking, claude-sonnet, qwen3-235b, deepseek-r1")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    args = parser.parse_args()

    categories_to_run = list(CATEGORIES.keys())
    if args.categories:
        categories_to_run = [c.strip() for c in args.categories.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write worker script to temp file
    worker_file = Path("_worker_task.py")
    worker_file.write_text(WORKER_SCRIPT)

    # Build all work items
    work_items = []
    for run_id in range(args.start_run, args.start_run + args.runs):
        for cat_name in categories_to_run:
            cat_config = CATEGORIES[cat_name]
            task_file = cat_config["task_file"]
            if not task_file.exists():
                logger.warning("Task file not found: %s", task_file)
                continue
            with open(task_file) as f:
                tasks = json.load(f)
            for task in tasks:
                work_items.append({
                    "task": task,
                    "cat_name": cat_name,
                    "method_name": cat_config["method"],
                    "max_iters": cat_config["max_iters"],
                    "model_name": args.model,
                    "temperature": args.temperature,
                    "run_id": run_id,
                })

    total = len(work_items)
    logger.info("Total work items: %d (%d runs × %d categories), workers: %d, model: %s",
                total, args.runs, len(categories_to_run), args.workers, args.model)

    # Load existing results for resume capability
    results_by_key = {}  # (run_id, cat_name) -> [entries]
    completed_keys = set()  # (run_id, cat_name, task_id) already done
    for rf in RESULTS_DIR.glob("*_onpolicy_run*.json"):
        try:
            with open(rf) as f:
                existing = json.load(f)
            for e in existing:
                if e.get("ss_quality") is not None or e.get("rv_quality") is not None:
                    key = (e["run_id"], e["category"])
                    results_by_key.setdefault(key, []).append(e)
                    completed_keys.add((e["run_id"], e["category"], e["task_id"]))
        except (json.JSONDecodeError, KeyError):
            pass

    # Filter out already-completed work items
    original_total = len(work_items)
    work_items = [
        item for item in work_items
        if (item["run_id"], item["cat_name"], item["task"]["task_id"]) not in completed_keys
    ]
    skipped = original_total - len(work_items)
    if skipped:
        logger.info("Resuming: %d tasks already completed, %d remaining", skipped, len(work_items))

    # Create temp dir for per-task output files
    tmp_dir = Path(tempfile.mkdtemp(prefix="augmentation_"))

    # Launch workers as separate OS processes
    active = {}  # proc -> (item, output_path)
    pending = list(work_items)
    total = len(work_items)
    completed = 0
    failed = 0
    t0 = time.time()

    def _save_results_incremental(key):
        """Save results for a (run_id, cat_name) group to disk."""
        run_id, cat_name = key
        entries = results_by_key.get(key, [])
        if not entries:
            return
        out_file = RESULTS_DIR / f"{cat_name}_onpolicy_run{run_id}.json"
        with open(out_file, "w") as f:
            json.dump(entries, f, indent=2)

    while pending or active:
        # Launch new workers up to limit
        while pending and len(active) < args.workers:
            item = pending.pop(0)
            output_path = str(tmp_dir / f"task_{completed + len(active)}_{item['task']['task_id']}.json")
            item_with_output = {**item, "output_path": output_path}
            proc = subprocess.Popen(
                [sys.executable, str(worker_file), json.dumps(item_with_output)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            active[proc] = (item, output_path)

        # Poll for completed processes
        done = []
        for proc in list(active):
            ret = proc.poll()
            if ret is not None:
                done.append(proc)

        for proc in done:
            item, output_path = active.pop(proc)
            completed += 1
            task_id = item["task"]["task_id"]
            cat_name = item["cat_name"]
            run_id = item["run_id"]

            if proc.returncode == 0 and Path(output_path).exists():
                with open(output_path) as f:
                    entry = json.load(f)
                key = (run_id, cat_name)
                results_by_key.setdefault(key, []).append(entry)
                # Save incrementally after each task
                _save_results_incremental(key)
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %s/%s run%d: SS=%.2f RV=%.2f (%.1f/min, ETA %.0fs)",
                    completed, total, cat_name, task_id, run_id,
                    entry.get("ss_quality") or 0,
                    entry.get("rv_quality") or 0,
                    rate * 60, eta,
                )
                Path(output_path).unlink(missing_ok=True)
            else:
                failed += 1
                stderr = proc.stderr.read().decode()[-500:] if proc.stderr else ""
                logger.error("[%d/%d] FAILED %s/%s run%d (rc=%d): %s",
                             completed, total, cat_name, task_id, run_id,
                             proc.returncode, stderr)

        if not done:
            time.sleep(0.5)  # avoid busy-wait

    # Final save (redundant but ensures completeness)
    for key in results_by_key:
        _save_results_incremental(key)

    # Cleanup
    worker_file.unlink(missing_ok=True)
    tmp_dir.rmdir()

    elapsed = time.time() - t0
    logger.info("All done: %d completed, %d failed, %.1f min (%.1f tasks/min)",
                completed, failed, elapsed / 60, completed / max(elapsed, 1) * 60)


if __name__ == "__main__":
    main()
