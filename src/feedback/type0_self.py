"""Type 0 feedback: self-review (no grounding).

The same model (or an identically-configured model) re-reads its own output
and critiques it.  This is the weakest feedback signal because the reviewer
shares all the same blind spots as the proposer.
"""

import logging

from src.config import ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

SELF_REVIEW_SYSTEM = (
    "You are an expert code reviewer.  Your job is to find bugs, missed edge "
    "cases, and correctness issues in the solution provided by the user.  Be "
    "specific: cite line-level issues and suggest concrete fixes.  If the code "
    "looks correct, say so briefly."
)

SELF_REVIEW_PROMPT = """\
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


class SelfReviewFeedback(FeedbackProvider):
    """Type 0 -- the same model reviews its own code with no external signal.

    Args:
        model_config: Configuration for the reviewing model.  Defaults to
            Claude Sonnet (matching a typical proposer).
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.claude_sonnet()

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Ask the model to critique *code* for *problem*."""
        client = get_client()
        problem_prompt = problem.get("prompt", problem.get("description", ""))

        user_message = SELF_REVIEW_PROMPT.format(
            problem_prompt=problem_prompt,
            code=code,
        )

        logger.info(
            "Type 0 self-review: requesting review from %s",
            self.model_config.model_id,
        )

        response = client.generate(
            config=self.model_config,
            system=SELF_REVIEW_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        logger.debug(
            "Type 0 self-review: %d tokens used", response.total_tokens
        )

        return FeedbackResult(
            feedback_type=FeedbackType.SELF_REVIEW,
            content=response.content,
            structured_data={
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
            tokens_used=response.total_tokens,
        )
