"""Base types for the feedback hierarchy (Type 0-3b).

The feedback type hierarchy captures increasing levels of grounding:
    Type 0 (Self-Review):      No grounding -- same model re-reads its output.
    Type 1 (Cross-Model):      No grounding -- different model critiques.
    Type 2 (Static Analysis):  Static grounding -- linters, AST checks.
    Type 3a (Execution):       Dynamic grounding -- run code against tests.
    Type 3b (Visual):          Visual grounding -- VLM analyses rendered output.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum


class FeedbackType(IntEnum):
    """Feedback type ordered by grounding strength."""

    SELF_REVIEW = 0
    CROSS_MODEL = 1
    STATIC_ANALYSIS = 2
    EXECUTION = 3
    VISUAL = 4


@dataclass
class FeedbackResult:
    """Structured result returned by every feedback provider.

    Attributes:
        feedback_type: Which feedback tier produced this result.
        content: Human-readable summary of the feedback.
        structured_data: Machine-readable details (lint warnings, test results, etc.).
        tokens_used: Total LLM tokens consumed to produce this feedback (0 for
            non-LLM providers like static analysis or execution).
    """

    feedback_type: FeedbackType
    content: str
    structured_data: dict = field(default_factory=dict)
    tokens_used: int = 0


class FeedbackProvider(ABC):
    """Abstract base class for all feedback providers.

    Subclasses must implement ``get_feedback`` which examines a candidate
    solution and returns a :class:`FeedbackResult`.
    """

    @abstractmethod
    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Evaluate *code* in the context of *problem* and return feedback.

        Args:
            code: The candidate solution source code.
            problem: Problem specification dict (benchmark-dependent).  May
                contain keys like ``'prompt'``, ``'entry_point'``,
                ``'test_code'``, etc.

        Returns:
            A :class:`FeedbackResult` with the evaluation outcome.
        """
        ...
