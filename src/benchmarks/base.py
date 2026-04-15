"""Base classes for benchmark loading."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProblem:
    """A single benchmark problem for code generation evaluation."""

    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_code: str
    difficulty: str | None = None


class Benchmark(ABC):
    """Abstract base class for code-generation benchmarks."""

    @abstractmethod
    def load(self) -> list[BenchmarkProblem]:
        """Load all problems from this benchmark.

        Returns:
            A list of BenchmarkProblem instances.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this benchmark."""
