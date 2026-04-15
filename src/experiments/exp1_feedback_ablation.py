"""Experiment 1: Feedback Type Ablation.

Tests the grounded feedback framework by holding architecture constant
(proposer-reviewer) and varying feedback type:
- Type 0: Self-review (no grounding)
- Type 1: Cross-model review (no grounding)
- Type 2: Static analysis (static grounding)
- Type 3a: Execution feedback (dynamic grounding)
- Type 2+3: Static + execution (both)
"""

import json
import logging
from pathlib import Path

from src.benchmarks.humaneval import HumanEvalBenchmark
from src.benchmarks.mbpp import MBPPBenchmark
from src.budget.allocator import AllocationStrategy
from src.config import ExperimentConfig, ModelConfig, RESULTS_DIR
from src.evaluation.metrics import MetricsCollector
from src.experiments.runner import BaselineConfig, ExperimentRunner, BASELINES
from src.feedback.base import FeedbackType

logger = logging.getLogger(__name__)

# Conditions for Experiment 1
EXP1_CONDITIONS = {
    "type0_self_review": BaselineConfig(
        name="type0_self_review",
        description="Proposer-Reviewer with self-review only (Type 0)",
        feedback_types=[FeedbackType.SELF_REVIEW],
        use_reviewer=True,
        use_proposer_revision=True,
        max_iterations=5,
    ),
    "type1_cross_model": BaselineConfig(
        name="type1_cross_model",
        description="Proposer-Reviewer with cross-model review (Type 1)",
        feedback_types=[FeedbackType.CROSS_MODEL],
        use_reviewer=True,
        use_proposer_revision=True,
        max_iterations=5,
    ),
    "type2_static": BaselineConfig(
        name="type2_static",
        description="Proposer-Reviewer with static analysis (Type 2)",
        feedback_types=[FeedbackType.STATIC_ANALYSIS],
        use_reviewer=True,
        use_proposer_revision=True,
        max_iterations=5,
    ),
    "type3_execution": BaselineConfig(
        name="type3_execution",
        description="Proposer-Reviewer with execution feedback (Type 3a)",
        feedback_types=[FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        max_iterations=5,
    ),
    "type23_combined": BaselineConfig(
        name="type23_combined",
        description="Proposer-Reviewer with static + execution (Type 2+3)",
        feedback_types=[FeedbackType.STATIC_ANALYSIS, FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        max_iterations=5,
    ),
}


def run_exp1(
    benchmark_name: str = "humaneval",
    num_problems: int | None = None,
    budget_tokens: int = 200_000,
    output_dir: Path | None = None,
) -> dict:
    """Run Experiment 1: Feedback Type Ablation."""
    output_dir = output_dir or RESULTS_DIR / "exp1_feedback_ablation"
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
        "Exp1: Running feedback ablation on %d %s problems",
        len(problems),
        benchmark_name,
    )

    config = ExperimentConfig(
        name=f"exp1_{benchmark_name}",
        benchmark=benchmark_name,
        budget_tokens=budget_tokens,
        output_dir=output_dir,
    )
    runner = ExperimentRunner(config)

    all_results = {}
    for condition_name, condition_config in EXP1_CONDITIONS.items():
        logger.info("Running condition: %s", condition_name)
        result = runner.run_baseline(condition_config, problems)
        all_results[condition_name] = result

        # Save incremental results
        result_path = output_dir / f"{condition_name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save combined summary
    summary = {
        name: r["summary"] for name, r in all_results.items()
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Exp1 complete. Results saved to %s", output_dir)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_exp1(num_problems=10)  # Small test run
