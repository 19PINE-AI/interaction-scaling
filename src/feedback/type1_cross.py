"""Type 1 feedback: cross-model review (no grounding).

A *different* model critiques the proposer's code.  This adds diversity of
reasoning but still lacks grounding -- the reviewer has no way to verify
claims against concrete evidence (tests, execution, linting).
"""

import logging

from src.config import ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

CROSS_REVIEW_SYSTEM = (
    "You are an expert code reviewer.  Your job is to find bugs, missed edge "
    "cases, and correctness issues in the solution provided by the user.  Be "
    "specific: cite line-level issues and suggest concrete fixes.  If the code "
    "looks correct, say so briefly."
)

CROSS_REVIEW_PROMPT = """\
Review this code for correctness, edge cases, and bugs.

## Problem description
{problem_prompt}

## Candidate solution
```python
{code}
```

Provide a structured review covering:
1. **Correctness** -- does the solution handle the stated requirements?
2. **Edge cases** -- are boundary conditions addressed?
3. **Bugs** -- are there any logic errors, off-by-one mistakes, or crashes?
4. **Verdict** -- PASS (no issues) or FAIL (issues found).
"""


class CrossModelFeedback(FeedbackProvider):
    """Type 1 -- a different model reviews the proposer's code.

    The reviewer model should differ from the proposer model so that the
    review benefits from a genuinely different reasoning process.

    Args:
        reviewer_config: Configuration for the reviewing model.  Defaults to
            GPT-4 (a common cross-model choice when the proposer is Claude).
    """

    def __init__(self, reviewer_config: ModelConfig | None = None) -> None:
        self.reviewer_config = reviewer_config or ModelConfig.gpt4()

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Ask a different model to critique *code* for *problem*."""
        client = get_client()
        problem_prompt = problem.get("prompt", problem.get("description", ""))

        user_message = CROSS_REVIEW_PROMPT.format(
            problem_prompt=problem_prompt,
            code=code,
        )

        logger.info(
            "Type 1 cross-model review: requesting review from %s",
            self.reviewer_config.model_id,
        )

        response = client.generate(
            config=self.reviewer_config,
            system=CROSS_REVIEW_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        logger.debug(
            "Type 1 cross-model review: %d tokens used", response.total_tokens
        )

        return FeedbackResult(
            feedback_type=FeedbackType.CROSS_MODEL,
            content=response.content,
            structured_data={
                "reviewer_model": response.model,
                "stop_reason": response.stop_reason,
            },
            tokens_used=response.total_tokens,
        )
