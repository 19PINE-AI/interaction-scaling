"""Budget allocation strategies for proposer-reviewer iterations.

Each strategy decides how to split a remaining token budget across the
three phases (propose, execute, review) for the current iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Available strategies for splitting tokens among phases."""

    FIXED = "fixed"
    PHASE_ADAPTIVE = "phase_adaptive"
    CONFIDENCE_CONDITIONED = "confidence_conditioned"


@dataclass(frozen=True)
class PhaseAllocation:
    """Token budgets assigned to each phase for one iteration.

    All values are non-negative integers representing token counts.
    """

    propose: int
    execute: int
    review: int

    @property
    def total(self) -> int:
        return self.propose + self.execute + self.review

    def as_dict(self) -> dict[str, int]:
        return {"propose": self.propose, "execute": self.execute, "review": self.review}


@dataclass
class BudgetAllocator:
    """Decides how to distribute remaining tokens across phases.

    Args:
        fixed_ratios: Proportional split used by :attr:`AllocationStrategy.FIXED`.
            Defaults to ``(0.4, 0.3, 0.3)`` (propose, execute, review).
    """

    fixed_ratios: tuple[float, float, float] = (0.4, 0.3, 0.3)

    def __post_init__(self) -> None:
        self._validate_ratios(self.fixed_ratios, "fixed_ratios")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def allocate(
        self,
        strategy: AllocationStrategy | str,
        iteration: int,
        total_iterations: int,
        confidence: float | None,
        remaining_budget: int,
    ) -> dict[str, int]:
        """Compute per-phase token budgets for the current iteration.

        Args:
            strategy: Which allocation strategy to use.
            iteration: Current iteration index (1-based).
            total_iterations: Total planned iterations for the run.
            confidence: Model-reported confidence in ``[0.0, 1.0]``, or
                ``None`` when unavailable.  Required for
                :attr:`AllocationStrategy.CONFIDENCE_CONDITIONED`.
            remaining_budget: Tokens still available from the total budget.

        Returns:
            A dict with keys ``"propose"``, ``"execute"``, ``"review"``
            whose values are non-negative integer token counts summing
            to at most *remaining_budget*.

        Raises:
            ValueError: On invalid arguments or missing confidence.
        """
        strategy = self._resolve_strategy(strategy)
        self._validate_inputs(iteration, total_iterations, remaining_budget)

        if remaining_budget == 0:
            return PhaseAllocation(0, 0, 0).as_dict()

        # Each strategy returns a (propose, execute, review) ratio tuple.
        if strategy is AllocationStrategy.FIXED:
            ratios = self._fixed_ratios()
        elif strategy is AllocationStrategy.PHASE_ADAPTIVE:
            ratios = self._phase_adaptive_ratios(iteration, total_iterations)
        elif strategy is AllocationStrategy.CONFIDENCE_CONDITIONED:
            ratios = self._confidence_conditioned_ratios(
                iteration, total_iterations, confidence
            )
        else:
            raise ValueError(f"Unhandled strategy: {strategy}")

        allocation = self._ratios_to_tokens(ratios, remaining_budget)

        logger.debug(
            "Allocation (strategy=%s, iter=%d/%d, conf=%s, budget=%d): %s",
            strategy.value,
            iteration,
            total_iterations,
            confidence,
            remaining_budget,
            allocation.as_dict(),
        )
        return allocation.as_dict()

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fixed_ratios(self) -> tuple[float, float, float]:
        """Return the user-configured fixed ratios."""
        return self.fixed_ratios

    @staticmethod
    def _phase_adaptive_ratios(
        iteration: int,
        total_iterations: int,
    ) -> tuple[float, float, float]:
        """Shift emphasis as the run progresses.

        * Round 1: heavy on propose (0.60 / 0.20 / 0.20) -- invest in a
          strong initial solution.
        * Final round: heavy on execute (0.20 / 0.60 / 0.20) -- polish and
          verify the solution before returning.
        * Intermediate rounds: heavy on review (0.20 / 0.40 / 0.40) --
          maximise feedback to guide refinement.
        """
        if iteration == 1:
            return (0.60, 0.20, 0.20)
        if iteration == total_iterations:
            return (0.20, 0.60, 0.20)
        return (0.20, 0.40, 0.40)

    @staticmethod
    def _confidence_conditioned_ratios(
        iteration: int,
        total_iterations: int,
        confidence: float | None,
    ) -> tuple[float, float, float]:
        """Allocate based on the model's self-assessed confidence.

        Confidence thresholds:
        * **High (>= 0.85):** The solution is likely correct.  Return a
          minimal allocation -- just enough to execute final verification.
          The caller should treat this as an early-stop signal.
        * **Medium ([0.50, 0.85)):** The solution is plausible but needs
          more work.  Shift budget towards execution to iterate on fixes.
        * **Low (< 0.50):** The approach may be wrong.  Invest heavily in
          review to surface what needs to change before re-proposing.

        Falls back to the phase-adaptive schedule when confidence is
        unavailable.
        """
        if confidence is None:
            logger.warning(
                "confidence_conditioned strategy requires confidence; "
                "falling back to phase_adaptive for iteration %d/%d",
                iteration,
                total_iterations,
            )
            return BudgetAllocator._phase_adaptive_ratios(iteration, total_iterations)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {confidence}"
            )

        if confidence >= 0.85:
            # High confidence: minimal allocation for final verification.
            return (0.10, 0.70, 0.20)
        if confidence >= 0.50:
            # Medium confidence: lean into execution to refine the solution.
            return (0.20, 0.50, 0.30)
        # Low confidence: heavy review to diagnose problems.
        return (0.30, 0.20, 0.50)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ratios_to_tokens(
        ratios: tuple[float, float, float],
        budget: int,
    ) -> PhaseAllocation:
        """Convert proportional ratios to integer token counts.

        Uses largest-remainder allocation to ensure the tokens sum to
        exactly *budget* without rounding errors.
        """
        raw = [r * budget for r in ratios]
        floored = [int(v) for v in raw]
        remainders = [v - f for v, f in zip(raw, floored)]
        shortfall = budget - sum(floored)

        # Distribute leftover tokens to the phases with the largest
        # fractional remainders.
        indices = sorted(range(3), key=lambda i: remainders[i], reverse=True)
        for i in indices[:shortfall]:
            floored[i] += 1

        return PhaseAllocation(propose=floored[0], execute=floored[1], review=floored[2])

    @staticmethod
    def _validate_ratios(
        ratios: tuple[float, float, float],
        name: str,
    ) -> None:
        if len(ratios) != 3:
            raise ValueError(f"{name} must have exactly 3 elements, got {len(ratios)}")
        if any(r < 0 for r in ratios):
            raise ValueError(f"{name} elements must be non-negative: {ratios}")
        total = sum(ratios)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"{name} must sum to ~1.0, got {total}")

    @staticmethod
    def _validate_inputs(
        iteration: int,
        total_iterations: int,
        remaining_budget: int,
    ) -> None:
        if iteration < 1:
            raise ValueError(f"iteration must be >= 1, got {iteration}")
        if total_iterations < 1:
            raise ValueError(f"total_iterations must be >= 1, got {total_iterations}")
        if iteration > total_iterations:
            raise ValueError(
                f"iteration ({iteration}) must be <= total_iterations ({total_iterations})"
            )
        if remaining_budget < 0:
            raise ValueError(f"remaining_budget must be non-negative, got {remaining_budget}")

    @staticmethod
    def _resolve_strategy(strategy: AllocationStrategy | str) -> AllocationStrategy:
        """Normalise *strategy* to an :class:`AllocationStrategy` enum member."""
        if isinstance(strategy, AllocationStrategy):
            return strategy
        try:
            return AllocationStrategy(strategy)
        except ValueError:
            valid = ", ".join(s.value for s in AllocationStrategy)
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of: {valid}"
            ) from None
