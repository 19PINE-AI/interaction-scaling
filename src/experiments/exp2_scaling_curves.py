"""Experiment 2: Scaling Curves by Dimension.

Measures performance at different budget levels for different scaling strategies:
- Reasoning-only (single shot with extended thinking)
- Sampling (best-of-N)
- Single-agent interaction (agentic loop)
- Proposer-reviewer interaction (ours)
"""

import json
import logging
from pathlib import Path

from src.benchmarks.humaneval import HumanEvalBenchmark
from src.benchmarks.mbpp import MBPPBenchmark
from src.config import ExperimentConfig, RESULTS_DIR
from src.experiments.runner import BaselineConfig, ExperimentRunner, BASELINES
from src.feedback.base import FeedbackType

logger = logging.getLogger(__name__)

# Budget levels to test (in tokens)
BUDGET_LEVELS = [50_000, 100_000, 200_000, 500_000]

# Map budget to max_iterations (rough heuristic: ~20K tokens per iteration)
def budget_to_iterations(budget: int) -> int:
    return max(1, budget // 40_000)


def get_scaling_conditions(budget: int) -> dict[str, BaselineConfig]:
    """Create conditions for a specific budget level."""
    max_iter = budget_to_iterations(budget)

    return {
        f"reasoning_only_B{budget // 1000}K": BaselineConfig(
            name=f"reasoning_only_B{budget // 1000}K",
            description=f"Single-shot reasoning at {budget}T budget",
            max_iterations=1,
        ),
        f"best_of_n_B{budget // 1000}K": BaselineConfig(
            name=f"best_of_n_B{budget // 1000}K",
            description=f"Best-of-N sampling at {budget}T budget",
            num_samples=max(2, budget // 50_000),
            max_iterations=1,
        ),
        f"agentic_loop_B{budget // 1000}K": BaselineConfig(
            name=f"agentic_loop_B{budget // 1000}K",
            description=f"Single-agent agentic loop at {budget}T budget",
            feedback_types=[FeedbackType.EXECUTION],
            use_proposer_revision=True,
            max_iterations=max_iter,
        ),
        f"proposer_reviewer_B{budget // 1000}K": BaselineConfig(
            name=f"proposer_reviewer_B{budget // 1000}K",
            description=f"Proposer-reviewer at {budget}T budget",
            feedback_types=[FeedbackType.EXECUTION],
            use_reviewer=True,
            use_proposer_revision=True,
            max_iterations=max_iter,
        ),
    }


def run_exp2(
    benchmark_name: str = "humaneval",
    num_problems: int | None = None,
    budget_levels: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run Experiment 2: Scaling Curves."""
    budget_levels = budget_levels or BUDGET_LEVELS
    output_dir = output_dir or RESULTS_DIR / "exp2_scaling_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    if benchmark_name == "humaneval":
        benchmark = HumanEvalBenchmark()
    elif benchmark_name == "mbpp":
        benchmark = MBPPBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    problems = benchmark.load()
    if num_problems:
        problems = problems[:num_problems]

    logger.info(
        "Exp2: Running scaling curves on %d %s problems at %d budget levels",
        len(problems),
        benchmark_name,
        len(budget_levels),
    )

    all_results = {}

    for budget in budget_levels:
        logger.info("Budget level: %dK tokens", budget // 1000)
        conditions = get_scaling_conditions(budget)

        config = ExperimentConfig(
            name=f"exp2_{benchmark_name}_B{budget // 1000}K",
            benchmark=benchmark_name,
            budget_tokens=budget,
            output_dir=output_dir,
        )
        runner = ExperimentRunner(config)

        for cond_name, cond_config in conditions.items():
            logger.info("  Running: %s", cond_name)
            result = runner.run_baseline(cond_config, problems)
            all_results[cond_name] = result

            # Save incremental
            result_path = output_dir / f"{cond_name}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    # Save scaling curve data
    scaling_data = {}
    for name, r in all_results.items():
        scaling_data[name] = {
            "pass_at_1": r["summary"]["pass_at_1"],
            "avg_tokens": r["summary"]["avg_tokens"],
            "num_problems": r["summary"]["num_problems"],
        }

    summary_path = output_dir / "scaling_data.json"
    with open(summary_path, "w") as f:
        json.dump(scaling_data, f, indent=2)

    logger.info("Exp2 complete. Results saved to %s", output_dir)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_exp2(num_problems=10, budget_levels=[50_000, 100_000])
