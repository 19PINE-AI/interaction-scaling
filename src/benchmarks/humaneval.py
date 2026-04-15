"""HumanEval+ benchmark loader using the evalplus package."""

from __future__ import annotations

import logging

from evalplus.data import get_human_eval_plus

from src.benchmarks.base import Benchmark, BenchmarkProblem

logger = logging.getLogger(__name__)


class HumanEvalBenchmark(Benchmark):
    """Loads HumanEval+ problems via the evalplus API."""

    def name(self) -> str:
        return "HumanEval+"

    def load(self) -> list[BenchmarkProblem]:
        """Load all 164 HumanEval+ problems.

        The evalplus package handles downloading and caching automatically.
        """
        logger.info("Loading HumanEval+ problems via evalplus...")
        data = get_human_eval_plus()

        problems: list[BenchmarkProblem] = []
        for task_id, entry in data.items():
            # Build test code that calls check(entry_point_function)
            test_str = entry["test"]
            entry_point = entry["entry_point"]
            # The test string defines check(candidate), we need to call it
            test_code = f"{test_str}\ncheck({entry_point})\n"

            problems.append(
                BenchmarkProblem(
                    task_id=task_id,
                    prompt=entry["prompt"],
                    entry_point=entry_point,
                    canonical_solution=entry["canonical_solution"],
                    test_code=test_code,
                )
            )

        logger.info("Loaded %d HumanEval+ problems", len(problems))
        return problems
