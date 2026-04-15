"""MBPP+ benchmark loader using the evalplus package."""

from __future__ import annotations

import logging

from evalplus.data import get_mbpp_plus

from src.benchmarks.base import Benchmark, BenchmarkProblem

logger = logging.getLogger(__name__)


class MBPPBenchmark(Benchmark):
    """Loads MBPP+ problems via the evalplus API."""

    def name(self) -> str:
        return "MBPP+"

    def load(self) -> list[BenchmarkProblem]:
        """Load all MBPP+ problems.

        The evalplus package handles downloading and caching automatically.
        """
        logger.info("Loading MBPP+ problems via evalplus...")
        data = get_mbpp_plus()

        problems: list[BenchmarkProblem] = []
        for task_id, entry in data.items():
            # Build test code
            test_str = entry["test"]
            entry_point = entry["entry_point"]
            test_code = f"{test_str}\ncheck({entry_point})\n"

            problems.append(
                BenchmarkProblem(
                    task_id=str(task_id),
                    prompt=entry["prompt"],
                    entry_point=entry_point,
                    canonical_solution=entry["canonical_solution"],
                    test_code=test_code,
                )
            )

        logger.info("Loaded %d MBPP+ problems", len(problems))
        return problems
