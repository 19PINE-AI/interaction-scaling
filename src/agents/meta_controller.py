"""Meta-controller for budget-aware proposer-reviewer orchestration.

Orchestrates the interaction loop: proposer generates -> environment provides
grounded feedback -> reviewer analyzes -> proposer revises. Manages budget
allocation across phases.
"""

import logging
from dataclasses import dataclass, field

from src.agents.base import AgentResponse
from src.agents.proposer import ProposerAgent
from src.agents.reviewer import ReviewerAgent, ReviewResult
from src.budget.allocator import AllocationStrategy, BudgetAllocator
from src.budget.tracker import BudgetTracker
from src.config import ExperimentConfig, ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Record of a single propose-execute-review iteration."""

    iteration: int
    code: str
    feedback: FeedbackResult | None = None
    review: ReviewResult | None = None
    tokens_propose: int = 0
    tokens_execute: int = 0
    tokens_review: int = 0
    passed: bool = False


@dataclass
class RunResult:
    """Complete result of a proposer-reviewer run on a single problem."""

    problem_id: str
    final_code: str
    iterations: list[IterationRecord] = field(default_factory=list)
    total_tokens: int = 0
    passed: bool = False
    stopped_reason: str = ""

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)


class MetaController:
    """Orchestrates budget-aware proposer-reviewer interaction loops."""

    def __init__(
        self,
        proposer: ProposerAgent,
        reviewer: ReviewerAgent | None,
        feedback_providers: list[FeedbackProvider],
        budget_tokens: int,
        max_iterations: int = 10,
        allocation_strategy: AllocationStrategy = AllocationStrategy.FIXED,
    ):
        self.proposer = proposer
        self.reviewer = reviewer
        self.feedback_providers = feedback_providers
        self.budget = BudgetTracker(budget_tokens)
        self.allocator = BudgetAllocator()
        self.max_iterations = max_iterations
        self.allocation_strategy = allocation_strategy

    def run(self, problem_id: str, problem_description: str, problem: dict) -> RunResult:
        """Run the full proposer-reviewer loop for a single problem."""
        result = RunResult(problem_id=problem_id, final_code="")
        current_code = ""

        for iteration in range(self.max_iterations):
            if self.budget.is_exhausted():
                result.stopped_reason = "budget_exhausted"
                break

            # Determine budget allocation for this iteration (allocator uses 1-based)
            confidence = self._estimate_confidence(result.iterations)
            allocation = self.allocator.allocate(
                strategy=self.allocation_strategy,
                iteration=iteration + 1,
                total_iterations=self.max_iterations,
                confidence=confidence,
                remaining_budget=self.budget.remaining(),
            )

            logger.info(
                "Iteration %d allocation: propose=%d, execute=%d, review=%d",
                iteration,
                allocation["propose"],
                allocation["execute"],
                allocation["review"],
            )

            record = IterationRecord(iteration=iteration, code="")

            # Phase 1: Propose (respect allocation limit)
            context = None
            if current_code:
                last_feedback = (
                    result.iterations[-1].feedback.content
                    if result.iterations[-1].feedback
                    else ""
                )
                last_review = (
                    result.iterations[-1].review.raw_review
                    if result.iterations[-1].review
                    else ""
                )
                context = {
                    "previous_code": current_code,
                    "feedback": last_feedback,
                    "review": last_review,
                    "iteration": iteration,
                }

            propose_response = self.proposer.generate(problem_description, context)
            current_code = propose_response.code
            record.code = current_code
            propose_tokens = propose_response.input_tokens + propose_response.output_tokens
            record.tokens_propose = propose_tokens
            self.budget.consume("propose", propose_tokens)

            if propose_tokens > allocation["propose"]:
                logger.warning(
                    "Propose phase exceeded allocation: %d > %d",
                    propose_tokens,
                    allocation["propose"],
                )

            if self.budget.is_exhausted():
                result.iterations.append(record)
                result.stopped_reason = "budget_exhausted_after_propose"
                break

            # Phase 2: Execute / Get Feedback
            combined_feedback = []
            total_feedback_tokens = 0
            for provider in self.feedback_providers:
                fb = provider.get_feedback(current_code, problem)
                combined_feedback.append(fb)
                total_feedback_tokens += fb.tokens_used

            record.tokens_execute = total_feedback_tokens
            self.budget.consume("execute", total_feedback_tokens)

            # Merge feedback results
            if combined_feedback:
                merged_content = "\n\n".join(
                    f"[{fb.feedback_type.name}]\n{fb.content}"
                    for fb in combined_feedback
                )
                merged_data = {}
                for fb in combined_feedback:
                    merged_data.update(fb.structured_data)
                record.feedback = FeedbackResult(
                    feedback_type=combined_feedback[0].feedback_type,
                    content=merged_content,
                    structured_data=merged_data,
                    tokens_used=total_feedback_tokens,
                )

                # Check if all tests passed
                record.passed = merged_data.get("passed", False)
                if record.passed:
                    result.iterations.append(record)
                    result.passed = True
                    result.stopped_reason = "tests_passed"
                    break

            if self.budget.is_exhausted():
                result.iterations.append(record)
                result.stopped_reason = "budget_exhausted_after_execute"
                break

            # Phase 3: Review (if reviewer is provided)
            if self.reviewer and record.feedback:
                review_context = {
                    "code": current_code,
                    "execution_result": record.feedback.content,
                    "iteration": iteration,
                }
                review = self.reviewer.review(problem_description, review_context)
                record.review = review
                review_tokens = review.input_tokens + review.output_tokens
                record.tokens_review = review_tokens
                self.budget.consume("review", review_tokens)

                # Check confidence for early stopping
                if review.confidence > 0.95 and record.passed:
                    result.iterations.append(record)
                    result.stopped_reason = "high_confidence_pass"
                    break

            result.iterations.append(record)

        # Finalize
        result.final_code = current_code
        result.total_tokens = self.budget.total_budget - self.budget.remaining()

        # Check final pass status from last iteration
        if result.iterations and result.iterations[-1].passed:
            result.passed = True

        if not result.stopped_reason:
            result.stopped_reason = "max_iterations"

        logger.info(
            "Problem %s: %s after %d iterations (%d tokens). Reason: %s",
            problem_id,
            "PASSED" if result.passed else "FAILED",
            result.num_iterations,
            result.total_tokens,
            result.stopped_reason,
        )
        return result

    def _estimate_confidence(self, iterations: list[IterationRecord]) -> float:
        """Estimate confidence based on iteration history."""
        if not iterations:
            return 0.0

        last = iterations[-1]
        if last.passed:
            return 0.95

        # Use review confidence if available
        if last.review and last.review.confidence > 0:
            return last.review.confidence

        # Heuristic: confidence grows with iterations if we're making progress
        # (fewer errors in later iterations)
        return min(0.3 + 0.1 * len(iterations), 0.8)

    def reset(self, budget_tokens: int | None = None):
        """Reset the controller for a new problem."""
        self.budget = BudgetTracker(budget_tokens or self.budget.total_budget)
