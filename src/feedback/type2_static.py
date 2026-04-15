"""Type 2 feedback: static analysis (static grounding).

Runs pylint and Python AST parsing against the candidate solution.  This is
the first tier that provides *grounded* feedback -- the signals come from
deterministic program analysis rather than another model's opinion.
"""

import ast
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType

logger = logging.getLogger(__name__)


def _run_pylint(code: str) -> list[dict]:
    """Run pylint on *code* and return structured diagnostics.

    Returns a list of dicts, each with keys: ``type``, ``module``, ``line``,
    ``column``, ``message``, ``message-id``, ``symbol``.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(code)
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "pylint",
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # skip missing-docstring noise
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # pylint exits non-zero when it finds issues -- that is expected.
        if result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except FileNotFoundError:
        logger.warning("pylint not found on PATH; skipping lint analysis")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("pylint timed out after 30 s")
        return []
    except json.JSONDecodeError:
        logger.warning("Failed to parse pylint JSON output")
        return []
    finally:
        tmp_path.unlink(missing_ok=True)


def _check_ast(code: str) -> dict:
    """Parse *code* with the ``ast`` module and report syntax issues.

    Returns a dict with:
        ``valid`` (bool): whether the code parses successfully.
        ``error`` (str | None): the parse error message, if any.
        ``num_functions`` (int): number of top-level function definitions.
        ``num_classes`` (int): number of top-level class definitions.
    """
    try:
        tree = ast.parse(code)
        num_functions = sum(
            1 for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        num_classes = sum(
            1 for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.ClassDef)
        )
        return {
            "valid": True,
            "error": None,
            "num_functions": num_functions,
            "num_classes": num_classes,
        }
    except SyntaxError as exc:
        return {
            "valid": False,
            "error": f"SyntaxError at line {exc.lineno}: {exc.msg}",
            "num_functions": 0,
            "num_classes": 0,
        }


def _format_lint_summary(lint_results: list[dict]) -> str:
    """Produce a human-readable summary of pylint diagnostics."""
    if not lint_results:
        return "pylint: no issues found."
    lines = [f"pylint: {len(lint_results)} issue(s) found:"]
    for item in lint_results:
        lines.append(
            f"  L{item.get('line', '?')}:{item.get('column', '?')} "
            f"[{item.get('symbol', item.get('message-id', '?'))}] "
            f"{item.get('message', '')}"
        )
    return "\n".join(lines)


class StaticAnalysisFeedback(FeedbackProvider):
    """Type 2 -- static analysis via pylint and AST parsing.

    No LLM calls are made; all signals are deterministic and grounded in the
    code itself.
    """

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Run static analysis on *code* and return structured results."""
        logger.info("Type 2 static analysis: running pylint + AST check")

        ast_result = _check_ast(code)
        lint_results = _run_pylint(code) if ast_result["valid"] else []

        # Build human-readable summary
        parts: list[str] = []
        if not ast_result["valid"]:
            parts.append(f"AST parse FAILED: {ast_result['error']}")
        else:
            parts.append(
                f"AST parse OK ({ast_result['num_functions']} function(s), "
                f"{ast_result['num_classes']} class(es))."
            )
        parts.append(_format_lint_summary(lint_results))

        content = "\n".join(parts)

        logger.debug(
            "Type 2 static analysis: ast_valid=%s, lint_issues=%d",
            ast_result["valid"],
            len(lint_results),
        )

        return FeedbackResult(
            feedback_type=FeedbackType.STATIC_ANALYSIS,
            content=content,
            structured_data={
                "ast": ast_result,
                "lint": lint_results,
                "lint_count": len(lint_results),
            },
            tokens_used=0,
        )
