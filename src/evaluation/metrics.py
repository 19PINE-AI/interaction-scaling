"""Metrics collection and computation for interaction scaling experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _ProblemRecord:
    """Internal bookkeeping for a single problem across multiple samples."""

    problem_id: str
    n_samples: int = 0
    n_correct: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    results: list[dict] = field(default_factory=list)


class MetricsCollector:
    """Collect per-problem results and derive aggregate metrics.

    Results are organized by *baseline* (e.g. ``"zero_shot"``,
    ``"proposer_reviewer_3iter"``).  Call :meth:`add_result` for every
    evaluated sample, then query aggregate statistics with
    :meth:`compute_pass_at_1`, :meth:`compute_pass_at_k`, or
    :meth:`get_summary`.
    """

    def __init__(self) -> None:
        # baseline -> problem_id -> _ProblemRecord
        self._data: dict[str, dict[str, _ProblemRecord]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_result(self, problem_id: str, baseline: str, result: dict) -> None:
        """Register one evaluated sample.

        Args:
            problem_id: Unique identifier for the problem (e.g.
                ``"HumanEval/0"``).
            baseline: Name of the experimental condition.
            result: Dictionary that **must** contain at least:

                - ``passed`` (bool) -- whether the sample passed all tests.

                and **may** contain:

                - ``tokens`` (int) -- total tokens consumed for this sample.
                - ``cost`` (float) -- monetary cost for this sample.
                - Any additional keys (they are stored verbatim).
        """
        problems = self._data.setdefault(baseline, {})
        rec = problems.get(problem_id)
        if rec is None:
            rec = _ProblemRecord(problem_id=problem_id)
            problems[problem_id] = rec

        rec.n_samples += 1
        if result.get("passed", False):
            rec.n_correct += 1
        rec.total_tokens += result.get("total_tokens", result.get("tokens", 0))
        rec.total_cost += result.get("cost", 0.0)
        rec.results.append(result)

    # ------------------------------------------------------------------
    # Pass-rate metrics
    # ------------------------------------------------------------------

    def compute_pass_at_1(self, baseline: str) -> float:
        """Fraction of problems where *at least one* sample passed (k=1).

        Equivalent to ``compute_pass_at_k(baseline, k=1)`` but avoids the
        combinatorial computation for the common single-sample case.
        """
        problems = self._get_problems(baseline)
        if not problems:
            return 0.0
        passed = sum(1 for rec in problems.values() if rec.n_correct > 0)
        return passed / len(problems)

    def compute_pass_at_k(self, baseline: str, k: int) -> float:
        """Unbiased pass@k estimator averaged over all problems.

        Uses the estimator from *Evaluating Large Language Models Trained on
        Code* (Chen et al., 2021)::

            pass@k = 1 - C(n - c, k) / C(n, k)

        where *n* is the total number of samples for a problem and *c* is the
        number that passed.  Problems with ``n < k`` are skipped with a
        warning.

        Args:
            baseline: Experimental condition name.
            k: Number of attempts budget.

        Returns:
            The mean pass@k across all eligible problems.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        problems = self._get_problems(baseline)
        if not problems:
            return 0.0

        estimates: list[float] = []
        for rec in problems.values():
            n = rec.n_samples
            c = rec.n_correct
            if n < k:
                logger.warning(
                    "Problem %s has only %d samples (< k=%d); skipping",
                    rec.problem_id,
                    n,
                    k,
                )
                continue
            if c == n:
                # All samples passed -- pass@k = 1.0 regardless of k
                estimates.append(1.0)
            elif c == 0:
                estimates.append(0.0)
            else:
                estimates.append(1.0 - comb(n - c, k) / comb(n, k))
        if not estimates:
            return 0.0
        return sum(estimates) / len(estimates)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self, baseline: str) -> dict:
        """Return aggregate statistics for *baseline*.

        Returns:
            A dictionary with keys:

            - ``baseline`` (str)
            - ``num_problems`` (int)
            - ``total_samples`` (int)
            - ``total_tokens`` (int)
            - ``total_cost`` (float)
            - ``pass_at_1`` (float)
        """
        problems = self._get_problems(baseline)
        total_samples = sum(r.n_samples for r in problems.values())
        total_tokens = sum(r.total_tokens for r in problems.values())
        total_cost = sum(r.total_cost for r in problems.values())

        avg_tokens = total_tokens / total_samples if total_samples > 0 else 0
        return {
            "baseline": baseline,
            "num_problems": len(problems),
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens": round(avg_tokens, 1),
            "total_cost": round(total_cost, 6),
            "pass_at_1": round(self.compute_pass_at_1(baseline), 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize all collected results to a JSON file at *path*.

        Parent directories are created automatically if they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, dict[str, dict]] = {}
        for baseline, problems in self._data.items():
            payload[baseline] = {}
            for pid, rec in problems.items():
                payload[baseline][pid] = {
                    "n_samples": rec.n_samples,
                    "n_correct": rec.n_correct,
                    "total_tokens": rec.total_tokens,
                    "total_cost": rec.total_cost,
                    "results": rec.results,
                }

        path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Saved metrics to %s", path)

    @classmethod
    def load(cls, path: Path) -> MetricsCollector:
        """Deserialize a :class:`MetricsCollector` from a JSON file.

        Args:
            path: Path to a JSON file previously written by :meth:`save`.

        Returns:
            A new :class:`MetricsCollector` instance with the loaded data.
        """
        path = Path(path)
        raw = json.loads(path.read_text())

        collector = cls()
        for baseline, problems in raw.items():
            for pid, record in problems.items():
                for result in record["results"]:
                    collector.add_result(pid, baseline, result)

        logger.info("Loaded metrics from %s", path)
        return collector

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_problems(self, baseline: str) -> dict[str, _ProblemRecord]:
        """Return the problems dict for *baseline*, warning if absent."""
        problems = self._data.get(baseline, {})
        if not problems:
            logger.warning("No results recorded for baseline %r", baseline)
        return problems
