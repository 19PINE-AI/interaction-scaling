"""Budget tracker for token consumption across propose/execute/review phases."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Phase(Enum):
    """The three phases in a proposer-reviewer iteration."""

    PROPOSE = "propose"
    EXECUTE = "execute"
    REVIEW = "review"


@dataclass
class PhaseUsage:
    """Token usage counters for a single phase."""

    tokens: int = 0
    call_count: int = 0

    def record(self, tokens: int) -> None:
        self.tokens += tokens
        self.call_count += 1


@dataclass
class BudgetTracker:
    """Tracks token consumption against a fixed total budget.

    The budget is shared across three phases (propose, execute, review).
    Callers record usage via ``consume`` and query remaining capacity
    before issuing further LLM calls.

    Args:
        total_budget: Maximum number of tokens that may be consumed across
            all phases combined.
    """

    total_budget: int
    _usage: dict[Phase, PhaseUsage] = field(default_factory=dict, init=False, repr=False)
    _total_consumed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {self.total_budget}")
        self._usage = {phase: PhaseUsage() for phase in Phase}

    # ------------------------------------------------------------------
    # Core accounting
    # ------------------------------------------------------------------

    def consume(self, phase: Phase | str, tokens: int) -> None:
        """Record *tokens* consumed during *phase*.

        Args:
            phase: One of the three iteration phases.  Accepts either a
                :class:`Phase` enum member or its string value
                (``"propose"``, ``"execute"``, ``"review"``).
            tokens: Number of tokens consumed (must be non-negative).

        Raises:
            ValueError: If *tokens* is negative or *phase* is unrecognised.
        """
        phase = self._resolve_phase(phase)
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")

        self._usage[phase].record(tokens)
        self._total_consumed += tokens

        if self._total_consumed > self.total_budget:
            logger.warning(
                "Budget exceeded: %d / %d tokens used (overshoot by %d)",
                self._total_consumed,
                self.total_budget,
                self._total_consumed - self.total_budget,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def remaining(self) -> int:
        """Return the number of tokens still available (floored at zero)."""
        return max(0, self.total_budget - self._total_consumed)

    def is_exhausted(self) -> bool:
        """Return ``True`` when the entire budget has been consumed."""
        return self._total_consumed >= self.total_budget

    def fraction_used(self) -> float:
        """Return the fraction of the budget consumed, in ``[0.0, 1.0+]``.

        Values above 1.0 indicate the budget was exceeded.
        """
        return self._total_consumed / self.total_budget

    def get_breakdown(self) -> dict[str, object]:
        """Return a detailed breakdown of token usage.

        Returns:
            A dictionary with per-phase and aggregate statistics::

                {
                    "total_budget": 200000,
                    "total_consumed": 42000,
                    "remaining": 158000,
                    "fraction_used": 0.21,
                    "phases": {
                        "propose": {"tokens": 20000, "call_count": 3},
                        "execute": {"tokens": 12000, "call_count": 2},
                        "review":  {"tokens": 10000, "call_count": 2},
                    },
                }
        """
        return {
            "total_budget": self.total_budget,
            "total_consumed": self._total_consumed,
            "remaining": self.remaining(),
            "fraction_used": round(self.fraction_used(), 4),
            "phases": {
                phase.value: {
                    "tokens": usage.tokens,
                    "call_count": usage.call_count,
                }
                for phase, usage in self._usage.items()
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_phase(phase: Phase | str) -> Phase:
        """Normalise *phase* to a :class:`Phase` enum member."""
        if isinstance(phase, Phase):
            return phase
        try:
            return Phase(phase)
        except ValueError:
            valid = ", ".join(p.value for p in Phase)
            raise ValueError(
                f"Unknown phase {phase!r}; expected one of: {valid}"
            ) from None
