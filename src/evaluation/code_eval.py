"""Safe code execution and evaluation against test cases."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import EXECUTION_TIMEOUT_SECONDS, MAX_OUTPUT_LENGTH
from src.utils.code_utils import truncate_output

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of running generated code against test cases."""

    passed: bool
    error_message: str | None
    stdout: str
    stderr: str
    execution_time_ms: float
    return_code: int


class CodeEvaluator:
    """Evaluate generated code by executing it in a subprocess.

    Each evaluation writes the combined solution + test code to a temporary
    file and runs it via ``python`` in a subprocess with a wall-clock timeout.
    stdout and stderr are captured and truncated to
    :data:`~src.config.MAX_OUTPUT_LENGTH` characters.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        code: str,
        test_code: str,
        timeout: int = EXECUTION_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        """Run *code* followed by *test_code* and report the outcome.

        The combined source is written to a temporary ``.py`` file which is
        executed in an isolated subprocess.  The subprocess is killed if it
        exceeds *timeout* seconds.

        Args:
            code: The solution source code.
            test_code: Assertions / test harness that exercises the solution.
            timeout: Maximum wall-clock seconds to allow.

        Returns:
            An :class:`ExecutionResult` describing whether the tests passed,
            captured output, and timing information.
        """
        combined = f"{code}\n\n{test_code}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as tmp:
            tmp.write(combined)
            tmp_path = Path(tmp.name)

        try:
            return self._run(tmp_path, timeout)
        finally:
            tmp_path.unlink(missing_ok=True)

    def evaluate_batch(
        self,
        solutions: list[tuple[str, str]],
        timeout: int = EXECUTION_TIMEOUT_SECONDS,
    ) -> list[ExecutionResult]:
        """Evaluate multiple ``(code, test_code)`` pairs sequentially.

        Args:
            solutions: Pairs of ``(solution_code, test_code)`` to evaluate.
            timeout: Per-problem timeout in seconds.

        Returns:
            A list of :class:`ExecutionResult` objects, one per input pair.
        """
        return [
            self.evaluate(code, test_code, timeout=timeout)
            for code, test_code in solutions
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _run(script_path: Path, timeout: int) -> ExecutionResult:
        """Execute *script_path* in a subprocess and interpret the result."""
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            stdout = truncate_output(proc.stdout, MAX_OUTPUT_LENGTH)
            stderr = truncate_output(proc.stderr, MAX_OUTPUT_LENGTH)
            passed = proc.returncode == 0

            error_message: str | None = None
            if not passed:
                error_message = _classify_error(stderr)

            return ExecutionResult(
                passed=passed,
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=round(elapsed_ms, 2),
                return_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Execution timed out after %d s for %s", timeout, script_path
            )
            return ExecutionResult(
                passed=False,
                error_message=f"Execution timed out after {timeout} seconds",
                stdout="",
                stderr="",
                execution_time_ms=round(elapsed_ms, 2),
                return_code=-1,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Unexpected error executing %s", script_path)
            return ExecutionResult(
                passed=False,
                error_message=f"Unexpected error: {exc}",
                stdout="",
                stderr="",
                execution_time_ms=round(elapsed_ms, 2),
                return_code=-1,
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _classify_error(stderr: str) -> str:
    """Return a human-friendly label for the dominant error in *stderr*."""
    if "SyntaxError" in stderr:
        return "SyntaxError: code could not be parsed"
    if "AssertionError" in stderr:
        return "AssertionError: one or more test assertions failed"
    if "TimeoutError" in stderr:
        return "TimeoutError: code raised a timeout internally"
    if "NameError" in stderr:
        return "NameError: undefined name referenced"
    if "TypeError" in stderr:
        return "TypeError: wrong type or argument count"
    # Fallback: grab the last non-empty line (usually the traceback summary)
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return "Unknown runtime error"
