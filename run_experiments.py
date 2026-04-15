#!/usr/bin/env python3
"""Main entry point for running interaction scaling experiments.

Usage:
    # Run all baselines on HumanEval+ (small test)
    python run_experiments.py --experiment baselines --benchmark humaneval --num-problems 10

    # Run Experiment 1: Feedback ablation
    python run_experiments.py --experiment exp1 --benchmark humaneval

    # Run Experiment 2: Scaling curves
    python run_experiments.py --experiment exp2 --benchmark humaneval --num-problems 20

    # Run Experiment 5: Verification gap
    python run_experiments.py --experiment exp5 --benchmark humaneval --num-problems 20

    # Run all experiments
    python run_experiments.py --experiment all --benchmark humaneval --num-problems 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.benchmarks.humaneval import HumanEvalBenchmark
from src.benchmarks.mbpp import MBPPBenchmark
from src.config import ExperimentConfig, RESULTS_DIR
from src.experiments.runner import ExperimentRunner, BASELINES


def run_baselines(args):
    """Run core baselines (B1-B5 + ours) on the specified benchmark."""
    benchmark_name = args.benchmark
    if benchmark_name == "humaneval":
        benchmark = HumanEvalBenchmark()
    elif benchmark_name == "mbpp":
        benchmark = MBPPBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    problems = benchmark.load()
    if args.num_problems:
        problems = problems[: args.num_problems]

    output_dir = RESULTS_DIR / "baselines" / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        name=f"baselines_{benchmark_name}",
        benchmark=benchmark_name,
        budget_tokens=args.budget,
        output_dir=output_dir,
    )
    runner = ExperimentRunner(config)

    # Select which baselines to run
    if args.conditions:
        condition_names = args.conditions.split(",")
        conditions = {k: v for k, v in BASELINES.items() if k in condition_names}
    else:
        conditions = BASELINES

    all_results = {}
    for name, baseline_config in conditions.items():
        logging.info("=" * 60)
        logging.info("Running: %s", name)
        logging.info("=" * 60)

        start = time.time()
        result = runner.run_baseline(baseline_config, problems)
        elapsed = time.time() - start

        all_results[name] = result
        result["wall_time_total"] = elapsed

        # Save per-condition results
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save combined summary
    summary = {}
    for name, r in all_results.items():
        summary[name] = r["summary"]
        summary[name]["wall_time_total"] = r.get("wall_time_total", 0)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Results Summary: {benchmark_name} ({len(problems)} problems)")
    print("=" * 80)
    print(f"{'Condition':<35} {'Pass@1':>8} {'Avg Tokens':>12} {'Time(s)':>10}")
    print("-" * 80)
    for name, s in summary.items():
        print(
            f"{name:<35} {s['pass_at_1']:>8.3f} "
            f"{s['avg_tokens']:>12.0f} "
            f"{s.get('wall_time_total', 0):>10.1f}"
        )
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run interaction scaling experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=["baselines", "exp1", "exp2", "exp5", "all"],
        default="baselines",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp"],
        default="humaneval",
        help="Which benchmark to use",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of problems to run (default: all)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=200_000,
        help="Token budget per problem",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions to run",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start = time.time()

    if args.experiment in ("baselines", "all"):
        run_baselines(args)

    if args.experiment in ("exp1", "all"):
        from src.experiments.exp1_feedback_ablation import run_exp1

        run_exp1(
            benchmark_name=args.benchmark,
            num_problems=args.num_problems,
            budget_tokens=args.budget,
        )

    if args.experiment in ("exp2", "all"):
        from src.experiments.exp2_scaling_curves import run_exp2

        run_exp2(
            benchmark_name=args.benchmark,
            num_problems=args.num_problems,
        )

    if args.experiment in ("exp5", "all"):
        from src.experiments.exp5_verification_gap import run_exp5

        run_exp5(
            benchmark_name=args.benchmark,
            num_problems=args.num_problems,
        )

    total_time = time.time() - start
    logging.info("Total experiment time: %.1fs", total_time)


if __name__ == "__main__":
    main()
