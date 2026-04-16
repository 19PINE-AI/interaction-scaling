"""Run webpage generation benchmark with robust timeout handling via multiprocessing."""
import json
import logging
import multiprocessing
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_single_task(task, use_review, max_iterations, result_queue):
    """Worker function to run a single task in a subprocess."""
    import anthropic
    from src.config import ExperimentConfig
    from src.experiments.hard_benchmark_runner import HardBenchmarkRunner
    from src.utils import llm_client

    # Fresh client in subprocess
    api_client = anthropic.Anthropic(timeout=90.0, max_retries=2)
    llm_client._client = llm_client.LLMClient()
    llm_client._client._anthropic = api_client

    config = ExperimentConfig(
        name="hard_web", benchmark="hard", budget_tokens=500000
    )
    runner = HardBenchmarkRunner(config)
    runner.client = llm_client._client

    try:
        result = runner.run_slide_task(
            task, use_review=use_review, max_iterations=max_iterations
        )
        result_queue.put({
            "quality_score": result.quality_score,
            "meets_requirements": result.meets_requirements,
            "iterations": result.iterations,
        })
    except Exception as e:
        result_queue.put({"error": f"{type(e).__name__}: {str(e)[:200]}"})


def run_task_with_timeout(task, use_review, max_iterations, timeout_s):
    """Run a task in a subprocess with a hard timeout."""
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_single_task,
        args=(task, use_review, max_iterations, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        return None, "TIMED OUT"

    if not result_queue.empty():
        data = result_queue.get_nowait()
        if "error" in data:
            return None, data["error"]
        return data, None
    return None, "No result returned"


def main():
    with open("data/hard_benchmarks/webpages/webpage_tasks.json") as f:
        tasks = json.load(f)

    tasks = tasks[:5]
    results = []

    for task in tasks:
        task_id = task["task_id"]
        print(f"\n=== {task_id} ===", flush=True)

        # Single-shot (120s timeout)
        print("  Running single-shot...", flush=True)
        start = time.time()
        r1, err1 = run_task_with_timeout(
            task, use_review=False, max_iterations=1, timeout_s=120
        )
        elapsed1 = time.time() - start
        if r1:
            print(
                f"  Single-shot: quality={r1['quality_score']:.2f}, "
                f"meets_reqs={r1['meets_requirements']} ({elapsed1:.0f}s)",
                flush=True,
            )
        else:
            print(f"  Single-shot: FAILED ({err1}) ({elapsed1:.0f}s)", flush=True)

        # Reviewed (300s timeout for 3 iterations)
        print("  Running reviewed (3 iters)...", flush=True)
        start = time.time()
        r2, err2 = run_task_with_timeout(
            task, use_review=True, max_iterations=3, timeout_s=300
        )
        elapsed2 = time.time() - start
        if r2:
            print(
                f"  Reviewed:    quality={r2['quality_score']:.2f}, "
                f"meets_reqs={r2['meets_requirements']}, "
                f"iters={r2['iterations']} ({elapsed2:.0f}s)",
                flush=True,
            )
        else:
            print(f"  Reviewed:    FAILED ({err2}) ({elapsed2:.0f}s)", flush=True)

        entry = {"task_id": task_id}
        if r1:
            entry["ss_quality"] = r1["quality_score"]
            entry["ss_meets"] = r1["meets_requirements"]
        else:
            entry["ss_quality"] = None
            entry["ss_meets"] = None

        if r2:
            entry["rv_quality"] = r2["quality_score"]
            entry["rv_meets"] = r2["meets_requirements"]
            entry["rv_iters"] = r2["iterations"]
        else:
            entry["rv_quality"] = None
            entry["rv_meets"] = None
            entry["rv_iters"] = None

        results.append(entry)

    # Save results
    out_dir = Path("results/hard_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "webpage_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = [
        r for r in results
        if r.get("ss_quality") is not None and r.get("rv_quality") is not None
    ]
    n = len(valid)

    print(f"\n{'='*60}")
    print(f"WEBPAGE BENCHMARK RESULTS")
    print(f"{'='*60}")

    if n > 0:
        avg_ss = sum(r["ss_quality"] for r in valid) / n
        avg_rv = sum(r["rv_quality"] for r in valid) / n
        ss_meets = sum(1 for r in valid if r["ss_meets"]) / n
        rv_meets = sum(1 for r in valid if r["rv_meets"]) / n
        print(f"Completed tasks: {n} / {len(tasks)}")
        print(f"Single-shot: avg_quality={avg_ss:.2f}, meets_reqs={ss_meets:.0%}")
        print(f"Reviewed:    avg_quality={avg_rv:.2f}, meets_reqs={rv_meets:.0%}")
        print(
            f"Delta:       +{avg_rv - avg_ss:.2f} quality, "
            f"+{(rv_meets - ss_meets)*100:.0f}pp meets_reqs"
        )
        print(f"\nPer-task breakdown:")
        for r in valid:
            delta = r["rv_quality"] - r["ss_quality"]
            print(
                f"  {r['task_id']}: SS={r['ss_quality']:.2f} -> "
                f"RV={r['rv_quality']:.2f} (delta={delta:+.2f}, iters={r['rv_iters']})"
            )

    # Report failed tasks
    failed = [r for r in results if r.get("ss_quality") is None or r.get("rv_quality") is None]
    if failed:
        print(f"\nFailed/partial tasks ({len(failed)}):")
        for r in failed:
            print(f"  {r['task_id']}: ss={r.get('ss_quality')}, rv={r.get('rv_quality')}")

    # Partial single-shot results
    ss_only = [r for r in results if r.get("ss_quality") is not None]
    if len(ss_only) > len(valid):
        avg_ss_all = sum(r["ss_quality"] for r in ss_only) / len(ss_only)
        print(f"\nSingle-shot only avg (all {len(ss_only)} with results): {avg_ss_all:.2f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
