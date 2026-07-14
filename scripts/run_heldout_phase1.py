"""Run Phase 1 code harness (single-shot vs proposer-reviewer) on held-out
v2 code tasks. Uses the same model / harness as the in-distribution Phase 1
runs (`run_onpolicy_augmentation.py`), pointed at a different task file.

Usage:
    python3 scripts/run_heldout_phase1.py --runs 1 --workers 16

Writes per-task JSON list to `results/heldout_phase1/code_heldout_harness.json`.
"""

import argparse
import json
import logging
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

TASK_FILE = Path("data/hard_benchmarks/code/code_tasks_heldout_v2.json")
RESULTS_DIR = Path("results/heldout_phase1")
CAT = "code"
METHOD = "run_code_task"
MAX_ITERS = 5

# Same worker body as run_onpolicy_augmentation.py, but tagged differently.
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
    name=f"heldout_phase1_run{run_id}",
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
if rv_result:
    entry["rv_final_code"] = getattr(rv_result, "final_code", "")
    entry["rv_intermediate_outputs"] = getattr(rv_result, "intermediate_outputs", [])

with open(output_path, "w") as f:
    json.dump(entry, f)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="claude-sonnet-thinking")
    parser.add_argument("--task-file", type=str, default=str(TASK_FILE))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N tasks (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    worker_file = Path("_worker_heldout_phase1.py")
    worker_file.write_text(WORKER_SCRIPT)

    with open(args.task_file) as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[:args.limit]
    logger.info("Loaded %d held-out code tasks from %s", len(tasks), args.task_file)

    # Build work items
    work_items = []
    for run_id in range(args.start_run, args.start_run + args.runs):
        for task in tasks:
            work_items.append({
                "task": task,
                "cat_name": CAT,
                "method_name": METHOD,
                "max_iters": MAX_ITERS,
                "model_name": args.model,
                "temperature": args.temperature,
                "run_id": run_id,
            })

    # Resume: load existing results
    results_by_run = {}  # run_id -> list of entries
    completed = set()    # (run_id, task_id)
    for rf in RESULTS_DIR.glob(f"{CAT}_heldout_harness*.json"):
        try:
            with open(rf) as f:
                existing = json.load(f)
            for e in existing:
                rid = e.get("run_id", 1)
                results_by_run.setdefault(rid, []).append(e)
                completed.add((rid, e["task_id"]))
        except (json.JSONDecodeError, KeyError):
            pass

    original_total = len(work_items)
    work_items = [
        it for it in work_items
        if (it["run_id"], it["task"]["task_id"]) not in completed
    ]
    if original_total - len(work_items):
        logger.info("Resuming: %d done, %d remaining",
                    original_total - len(work_items), len(work_items))

    tmp_dir = Path(tempfile.mkdtemp(prefix="heldout_phase1_"))
    active = {}
    pending = list(work_items)
    total = len(work_items)
    done_count = 0
    failed = 0
    t0 = time.time()

    def save_all():
        # Single canonical file
        all_entries = []
        for rid in sorted(results_by_run):
            all_entries.extend(results_by_run[rid])
        out_file = RESULTS_DIR / f"{CAT}_heldout_harness.json"
        with open(out_file, "w") as f:
            json.dump(all_entries, f, indent=2)

    while pending or active:
        while pending and len(active) < args.workers:
            item = pending.pop(0)
            output_path = str(tmp_dir / f"task_{done_count + len(active)}_{item['task']['task_id']}_r{item['run_id']}.json")
            item_with_output = {**item, "output_path": output_path}
            proc = subprocess.Popen(
                [sys.executable, str(worker_file), json.dumps(item_with_output)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            active[proc] = (item, output_path)

        finished = []
        for proc in list(active):
            if proc.poll() is not None:
                finished.append(proc)

        for proc in finished:
            item, output_path = active.pop(proc)
            done_count += 1
            tid = item["task"]["task_id"]
            rid = item["run_id"]
            if proc.returncode == 0 and Path(output_path).exists():
                with open(output_path) as f:
                    entry = json.load(f)
                results_by_run.setdefault(rid, []).append(entry)
                save_all()
                elapsed = time.time() - t0
                rate = done_count / elapsed
                eta = (total - done_count) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %s r%d: SS=%.2f RV=%.2f rv_iters=%s (%.1f/min, ETA %.0fs)",
                    done_count, total, tid, rid,
                    entry.get("ss_quality") or 0,
                    entry.get("rv_quality") or 0,
                    entry.get("rv_iters"),
                    rate * 60, eta,
                )
                Path(output_path).unlink(missing_ok=True)
            else:
                failed += 1
                stderr = proc.stderr.read().decode()[-500:] if proc.stderr else ""
                logger.error("[%d/%d] FAILED %s r%d (rc=%d): %s",
                             done_count, total, tid, rid, proc.returncode, stderr)

        if not finished:
            time.sleep(0.5)

    save_all()
    worker_file.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    elapsed = time.time() - t0
    logger.info("DONE: %d completed, %d failed, %.1f min",
                done_count, failed, elapsed / 60)


if __name__ == "__main__":
    main()
