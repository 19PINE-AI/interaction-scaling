"""Type 3a feedback: execution with tests (dynamic grounding).

Runs the candidate solution together with test assertions in a subprocess.
This is the strongest grounding signal: it checks actual runtime behaviour
rather than relying on opinions or static properties.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

from src.config import EXECUTION_TIMEOUT_SECONDS, MAX_OUTPUT_LENGTH
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.utils.code_utils import combine_code_and_tests, truncate_output

logger = logging.getLogger(__name__)


class ExecutionFeedback(FeedbackProvider):
    """Type 3a -- execute the solution against provided tests.

    The *problem* dict must contain a ``'test_code'`` key whose value is a
    string of Python test assertions (e.g. ``assert candidate(2) == 4``).

    The solution and test code are combined into a single script, written to
    a temporary file, and executed in a subprocess with a configurable
    timeout.

    Args:
        timeout: Maximum wall-clock seconds for execution.  Defaults to the
            project-wide ``EXECUTION_TIMEOUT_SECONDS``.
        max_output: Maximum characters of captured stdout/stderr.  Defaults to
            the project-wide ``MAX_OUTPUT_LENGTH``.
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_output: int | None = None,
    ) -> None:
        self.timeout = timeout if timeout is not None else EXECUTION_TIMEOUT_SECONDS
        self.max_output = max_output if max_output is not None else MAX_OUTPUT_LENGTH

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Execute *code* with tests from *problem* and report results."""
        test_code = problem.get("test_code")
        if test_code is None:
            logger.warning(
                "Type 3a execution: problem dict has no 'test_code' key"
            )
            return FeedbackResult(
                feedback_type=FeedbackType.EXECUTION,
                content="SKIP: no test_code provided in problem dict.",
                structured_data={"passed": False, "reason": "no_test_code"},
                tokens_used=0,
            )

        combined = combine_code_and_tests(code, test_code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(combined)
            tmp.flush()
            tmp_path = Path(tmp.name)

        try:
            result = self._execute(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        passed = result["returncode"] == 0
        content = self._format_result(passed, result)

        logger.info(
            "Type 3a execution: %s (returncode=%d)",
            "PASSED" if passed else "FAILED",
            result["returncode"],
        )

        return FeedbackResult(
            feedback_type=FeedbackType.EXECUTION,
            content=content,
            structured_data={
                "passed": passed,
                "returncode": result["returncode"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "timed_out": result["timed_out"],
            },
            tokens_used=0,
        )

    def _execute(self, script_path: Path) -> dict:
        """Run *script_path* in a subprocess and capture output."""
        try:
            proc = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "returncode": proc.returncode,
                "stdout": truncate_output(proc.stdout, self.max_output),
                "stderr": truncate_output(proc.stderr, self.max_output),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            logger.warning(
                "Type 3a execution: timed out after %d s", self.timeout
            )
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout} seconds.",
                "timed_out": True,
            }

    @staticmethod
    def _format_result(passed: bool, result: dict) -> str:
        """Build a human-readable execution report."""
        parts: list[str] = []
        if result["timed_out"]:
            parts.append(
                f"TIMEOUT: execution exceeded {result.get('timeout', '?')}s limit."
            )
        elif passed:
            parts.append("PASSED: all test assertions succeeded.")
        else:
            parts.append("FAILED: test assertions did not pass.")

        if result["stdout"].strip():
            parts.append(f"\n--- stdout ---\n{result['stdout'].strip()}")
        if result["stderr"].strip():
            parts.append(f"\n--- stderr ---\n{result['stderr'].strip()}")

        return "\n".join(parts)
