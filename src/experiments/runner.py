"""Experiment runner framework for interaction scaling experiments.

Provides a unified interface for running different experimental conditions
(baselines and our approach) across benchmark problems.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.agents.meta_controller import MetaController, RunResult
from src.agents.proposer import ProposerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.single_agent import SingleAgentLoop
from src.benchmarks.base import Benchmark, BenchmarkProblem
from src.budget.allocator import AllocationStrategy
from src.budget.tracker import BudgetTracker
from src.config import ExperimentConfig, ModelConfig
from src.evaluation.code_eval import CodeEvaluator
from src.evaluation.metrics import MetricsCollector
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.feedback.type0_self import SelfReviewFeedback
from src.feedback.type1_cross import CrossModelFeedback
from src.feedback.type2_static import StaticAnalysisFeedback
from src.feedback.type3a_execution import ExecutionFeedback
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for a single baseline or experimental condition."""

    name: str
    description: str
    feedback_types: list[FeedbackType] = field(default_factory=list)
    use_reviewer: bool = False
    use_proposer_revision: bool = False
    num_samples: int = 1  # For best-of-N
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIXED
    max_iterations: int = 1
    proposer_model: ModelConfig = field(default_factory=ModelConfig.claude_sonnet)
    reviewer_model: ModelConfig = field(default_factory=ModelConfig.claude_sonnet)


# Pre-defined baseline configurations
BASELINES = {
    "B1_single_no_review": BaselineConfig(
        name="B1_single_no_review",
        description="Single-agent, no review (Think only)",
        max_iterations=1,
    ),
    "B2_self_review": BaselineConfig(
        name="B2_self_review",
        description="Single-agent, self-review (Think + Type 0)",
        feedback_types=[FeedbackType.SELF_REVIEW],
        use_proposer_revision=True,
        max_iterations=3,
    ),
    "B3_cross_model": BaselineConfig(
        name="B3_cross_model",
        description="Cross-model review (Think + Type 1)",
        feedback_types=[FeedbackType.CROSS_MODEL],
        use_proposer_revision=True,
        max_iterations=3,
    ),
    "B4_best_of_n": BaselineConfig(
        name="B4_best_of_n",
        description="Best-of-N sampling",
        num_samples=5,
        max_iterations=1,
    ),
    "B5_agentic_loop": BaselineConfig(
        name="B5_agentic_loop",
        description="Single-agent agentic loop (Do, no separate reviewer)",
        feedback_types=[FeedbackType.EXECUTION],
        use_proposer_revision=True,
        max_iterations=5,
    ),
    "ours_type3_fixed": BaselineConfig(
        name="ours_type3_fixed",
        description="Proposer-Reviewer with execution feedback, fixed allocation",
        feedback_types=[FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        allocation_strategy=AllocationStrategy.FIXED,
        max_iterations=5,
    ),
    "ours_type3_adaptive": BaselineConfig(
        name="ours_type3_adaptive",
        description="Proposer-Reviewer with execution feedback, adaptive allocation",
        feedback_types=[FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        allocation_strategy=AllocationStrategy.PHASE_ADAPTIVE,
        max_iterations=5,
    ),
    "ours_type3_confidence": BaselineConfig(
        name="ours_type3_confidence",
        description="Proposer-Reviewer with execution feedback, confidence-conditioned",
        feedback_types=[FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        allocation_strategy=AllocationStrategy.CONFIDENCE_CONDITIONED,
        max_iterations=5,
    ),
    "ours_type23_fixed": BaselineConfig(
        name="ours_type23_fixed",
        description="Proposer-Reviewer with static + execution feedback",
        feedback_types=[FeedbackType.STATIC_ANALYSIS, FeedbackType.EXECUTION],
        use_reviewer=True,
        use_proposer_revision=True,
        allocation_strategy=AllocationStrategy.FIXED,
        max_iterations=5,
    ),
}


class ExperimentRunner:
    """Runs experimental conditions on benchmark problems."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = CodeEvaluator()
        self.metrics = MetricsCollector()
        self.client = get_client()

    def run_baseline(
        self,
        baseline_config: BaselineConfig,
        problems: list[BenchmarkProblem],
    ) -> dict:
        """Run a single baseline condition on a list of problems."""
        logger.info(
            "Running %s on %d problems", baseline_config.name, len(problems)
        )

        results = []
        for i, problem in enumerate(problems):
            logger.info(
                "[%d/%d] %s: %s",
                i + 1,
                len(problems),
                baseline_config.name,
                problem.task_id,
            )

            start_time = time.time()
            self.client.reset_counters()

            if baseline_config.num_samples > 1:
                result = self._run_best_of_n(baseline_config, problem)
            elif baseline_config.use_reviewer:
                result = self._run_proposer_reviewer(baseline_config, problem)
            elif baseline_config.use_proposer_revision:
                result = self._run_single_agent_loop(baseline_config, problem)
            else:
                result = self._run_single_shot(baseline_config, problem)

            elapsed = time.time() - start_time
            usage = self.client.get_usage_summary()

            # Final evaluation — include prompt prefix for helper definitions
            # (e.g., HumanEval/38 defines encode_cyclic in the prompt)
            full_code = problem.prompt + "\n" + result["code"]
            eval_result = self.evaluator.evaluate(
                full_code, problem.test_code
            )

            record = {
                "problem_id": problem.task_id,
                "baseline": baseline_config.name,
                "passed": eval_result.passed,
                "code": result["code"],
                "iterations": result.get("iterations", 1),
                "total_tokens": usage["total_tokens"],
                "input_tokens": usage["total_input_tokens"],
                "output_tokens": usage["total_output_tokens"],
                "api_calls": usage["call_count"],
                "wall_time_seconds": elapsed,
                "error": eval_result.error_message,
                "stopped_reason": result.get("stopped_reason", "complete"),
            }
            results.append(record)

            self.metrics.add_result(
                problem.task_id, baseline_config.name, record
            )

            logger.info(
                "  %s: %s (%.1fs, %d tokens)",
                problem.task_id,
                "PASS" if eval_result.passed else "FAIL",
                elapsed,
                usage["total_tokens"],
            )

        summary = self.metrics.get_summary(baseline_config.name)
        logger.info(
            "%s complete: pass@1=%.3f, avg_tokens=%d",
            baseline_config.name,
            summary["pass_at_1"],
            summary["avg_tokens"],
        )
        return {"results": results, "summary": summary}

    def _run_single_shot(
        self, config: BaselineConfig, problem: BenchmarkProblem
    ) -> dict:
        """B1: Single generation, no feedback."""
        proposer = ProposerAgent(config.proposer_model)
        response = proposer.generate(problem.prompt)
        return {"code": response.code, "iterations": 1}

    def _run_single_agent_loop(
        self, config: BaselineConfig, problem: BenchmarkProblem
    ) -> dict:
        """B2/B3/B5: Single agent with feedback loop."""
        proposer = ProposerAgent(config.proposer_model)
        feedback_providers = self._create_feedback_providers(config, problem)

        current_code = ""
        last_feedback_content = ""
        all_passed = False
        budget = BudgetTracker(self.config.budget_tokens)
        iteration = 0

        for iteration in range(config.max_iterations):
            if budget.is_exhausted():
                break

            # Generate or revise
            context = None
            if current_code:
                context = {
                    "previous_code": current_code,
                    "feedback": last_feedback_content,
                    "iteration": iteration,
                }
            response = proposer.generate(problem.prompt, context)
            current_code = response.code
            budget.consume("propose", response.input_tokens + response.output_tokens)

            if budget.is_exhausted() or not feedback_providers:
                break

            # Get feedback
            last_feedback_content = ""
            all_passed = True
            for provider in feedback_providers:
                fb = provider.get_feedback(current_code, {
                    "test_code": problem.test_code,
                    "entry_point": problem.entry_point,
                    "prompt": problem.prompt,
                })
                last_feedback_content += f"\n{fb.content}"
                budget.consume("execute", fb.tokens_used)
                if not fb.structured_data.get("passed", False):
                    all_passed = False

            if all_passed:
                break

        return {
            "code": current_code,
            "iterations": iteration + 1,
            "stopped_reason": "tests_passed" if all_passed else "max_iterations",
        }

    def _run_proposer_reviewer(
        self, config: BaselineConfig, problem: BenchmarkProblem
    ) -> dict:
        """Our approach: Proposer-Reviewer with grounded feedback."""
        proposer = ProposerAgent(config.proposer_model)
        reviewer = ReviewerAgent(config.reviewer_model)
        feedback_providers = self._create_feedback_providers(config, problem)

        controller = MetaController(
            proposer=proposer,
            reviewer=reviewer,
            feedback_providers=feedback_providers,
            budget_tokens=self.config.budget_tokens,
            max_iterations=config.max_iterations,
            allocation_strategy=config.allocation_strategy,
        )

        problem_dict = {
            "test_code": problem.test_code,
            "entry_point": problem.entry_point,
            "prompt": problem.prompt,
        }

        run_result = controller.run(
            problem.task_id, problem.prompt, problem_dict
        )

        return {
            "code": run_result.final_code,
            "iterations": run_result.num_iterations,
            "stopped_reason": run_result.stopped_reason,
        }

    def _run_best_of_n(
        self, config: BaselineConfig, problem: BenchmarkProblem
    ) -> dict:
        """B4: Generate N candidates, pick the best."""
        # Use higher temperature for diversity in sampling
        sampling_model = ModelConfig(
            provider=config.proposer_model.provider,
            model_id=config.proposer_model.model_id,
            max_tokens=config.proposer_model.max_tokens,
            temperature=0.8,
        )
        proposer = ProposerAgent(sampling_model)
        candidates = []

        for _ in range(config.num_samples):
            response = proposer.generate(problem.prompt)
            eval_result = self.evaluator.evaluate(response.code, problem.test_code)
            candidates.append({
                "code": response.code,
                "passed": eval_result.passed,
                "error": eval_result.error_message,
            })

        # Select best: first passing, or first candidate
        best = next(
            (c for c in candidates if c["passed"]),
            candidates[0] if candidates else {"code": "", "passed": False},
        )
        return {
            "code": best["code"],
            "iterations": config.num_samples,
            "stopped_reason": "best_of_n",
        }

    def _create_feedback_providers(
        self, config: BaselineConfig, problem: BenchmarkProblem
    ) -> list[FeedbackProvider]:
        """Create feedback providers based on the baseline configuration."""
        providers: list[FeedbackProvider] = []
        for ft in config.feedback_types:
            if ft == FeedbackType.SELF_REVIEW:
                providers.append(SelfReviewFeedback(config.proposer_model))
            elif ft == FeedbackType.CROSS_MODEL:
                # Use a different model for cross-review
                cross_model = (
                    ModelConfig.gpt4()
                    if config.proposer_model.provider.value == "anthropic"
                    else ModelConfig.claude_sonnet()
                )
                providers.append(CrossModelFeedback(cross_model))
            elif ft == FeedbackType.STATIC_ANALYSIS:
                providers.append(StaticAnalysisFeedback())
            elif ft == FeedbackType.EXECUTION:
                providers.append(ExecutionFeedback())
        return providers

    def save_results(self, output_path: Path | None = None):
        """Save all collected metrics to disk."""
        path = output_path or self.config.output_dir / f"{self.config.name}.json"
        self.metrics.save(path)
        logger.info("Results saved to %s", path)
