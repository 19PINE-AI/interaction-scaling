"""Reviewer agent: analyzes code and produces structured improvement suggestions."""

import json
import logging
from dataclasses import dataclass, field

from src.agents.base import Agent, AgentResponse
from src.config import ModelConfig
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a meticulous code reviewer. You will receive a Python function along \
with execution results (test outcomes, lint output, error tracebacks, etc.).

Your task is to produce a JSON object with the following structure:

{
  "issues": [
    {
      "description": "Brief description of the issue",
      "severity": "critical" | "major" | "minor",
      "line_hint": "optional: relevant code fragment or line"
    }
  ],
  "suggestions": [
    "Actionable suggestion 1",
    "Actionable suggestion 2"
  ],
  "confidence": 0.85
}

Rules:
- "issues" lists every problem you find, each with a severity level.
  - critical: causes incorrect output or crashes.
  - major: significant logic error or missing edge-case handling.
  - minor: style, naming, or minor inefficiency.
- "suggestions" gives concrete, actionable improvement advice.
- "confidence" is a float between 0 and 1 reflecting how confident you are \
that your review is complete and correct.
- Output ONLY the JSON object, nothing else.
"""


@dataclass
class ReviewResult:
    """Structured result produced by the ReviewerAgent."""

    issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_review: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_critical_issues(self) -> bool:
        return any(
            issue.get("severity") == "critical" for issue in self.issues
        )

    def as_feedback_str(self) -> str:
        """Format the review as a human-readable feedback string.

        This is suitable for passing back into a ProposerAgent's context
        as the ``feedback`` value.
        """
        parts: list[str] = []
        if self.issues:
            parts.append("Issues found:")
            for i, issue in enumerate(self.issues, 1):
                severity = issue.get("severity", "unknown").upper()
                desc = issue.get("description", "")
                hint = issue.get("line_hint", "")
                entry = f"  {i}. [{severity}] {desc}"
                if hint:
                    entry += f"  (near: {hint})"
                parts.append(entry)
        if self.suggestions:
            parts.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        parts.append(f"\nReviewer confidence: {self.confidence:.0%}")
        return "\n".join(parts)


class ReviewerAgent(Agent):
    """Agent that reviews code and produces structured feedback.

    The reviewer takes execution results, lint output, and/or test results
    through the ``context`` dictionary and returns both a standard
    ``AgentResponse`` and a richer ``ReviewResult``.
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.claude_sonnet()
        self.client = get_client()

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #

    def generate(
        self,
        problem_description: str,
        context: dict | None = None,
    ) -> AgentResponse:
        """Produce a code review and return it as an AgentResponse.

        The extracted ``code`` field contains the raw JSON review.  For
        a richer structured result, call :meth:`review` instead.
        """
        result = self.review(problem_description, context)
        return AgentResponse(
            code=result.raw_review,
            reasoning=result.as_feedback_str(),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def review(
        self,
        problem_description: str,
        context: dict | None = None,
    ) -> ReviewResult:
        """Analyze code and feedback, returning a structured ReviewResult.

        Args:
            problem_description: The original problem statement.
            context: Dictionary that should contain some combination of:
                - code: str -- the current solution to review.
                - execution_result: str -- stdout / stderr from running the code.
                - test_results: str -- test pass/fail output.
                - lint_output: str -- linter warnings and errors.
                - iteration: int -- current iteration number.

        Returns:
            A ReviewResult with parsed issues, suggestions, and confidence.
        """
        context = context or {}
        code = context.get("code", "")
        execution_result = context.get("execution_result", "")
        test_results = context.get("test_results", "")
        lint_output = context.get("lint_output", "")
        iteration = context.get("iteration", 0)

        user_content = self._build_user_message(
            problem_description=problem_description,
            code=code,
            execution_result=execution_result,
            test_results=test_results,
            lint_output=lint_output,
            iteration=iteration,
        )

        messages = [{"role": "user", "content": user_content}]

        response = self.client.generate(
            config=self.model_config,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        result = self._parse_review(response.content)
        result.input_tokens = response.input_tokens
        result.output_tokens = response.output_tokens

        logger.info(
            "Reviewer (iter %d): %d issues, confidence %.0f%%, %d tokens",
            iteration,
            len(result.issues),
            result.confidence * 100,
            response.total_tokens,
        )
        return result

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_user_message(
        *,
        problem_description: str,
        code: str,
        execution_result: str,
        test_results: str,
        lint_output: str,
        iteration: int,
    ) -> str:
        parts = [f"Problem:\n{problem_description}\n"]

        if code:
            parts.append(f"Code (iteration {iteration}):\n```python\n{code}\n```\n")

        if execution_result:
            parts.append(f"Execution result:\n```\n{execution_result}\n```\n")

        if test_results:
            parts.append(f"Test results:\n```\n{test_results}\n```\n")

        if lint_output:
            parts.append(f"Lint output:\n```\n{lint_output}\n```\n")

        if not any([code, execution_result, test_results, lint_output]):
            parts.append(
                "(No code or feedback provided -- please note that there is "
                "nothing to review yet.)\n"
            )

        parts.append(
            "Analyze the above and respond with ONLY the JSON review object."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_review(raw: str) -> ReviewResult:
        """Parse the LLM's JSON response into a ReviewResult.

        Handles cases where the model wraps the JSON in markdown code fences
        or adds preamble text.
        """
        raw_stripped = raw.strip()

        # Strip optional markdown code fences
        if raw_stripped.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = raw_stripped.find("\n")
            if first_newline == -1:
                # Malformed fence with no newline — skip stripping
                raw_stripped = raw_stripped[3:]
            else:
                raw_stripped = raw_stripped[first_newline + 1 :]
            if raw_stripped.endswith("```"):
                raw_stripped = raw_stripped[: -len("```")].strip()

        try:
            data = json.loads(raw_stripped)
        except json.JSONDecodeError:
            # Last resort: try to find a JSON object in the text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse reviewer JSON output")
                    return ReviewResult(
                        issues=[],
                        suggestions=["(Review output could not be parsed.)"],
                        confidence=0.0,
                        raw_review=raw,
                    )
            else:
                logger.warning("No JSON found in reviewer output")
                return ReviewResult(
                    issues=[],
                    suggestions=["(Review output could not be parsed.)"],
                    confidence=0.0,
                    raw_review=raw,
                )

        return ReviewResult(
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            confidence=float(data.get("confidence", 0.0)),
            raw_review=raw,
        )
