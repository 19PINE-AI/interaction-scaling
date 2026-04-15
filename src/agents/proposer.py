"""Proposer agent: generates and revises code solutions."""

import logging

from src.agents.base import Agent, AgentResponse
from src.config import ModelConfig
from src.utils.code_utils import extract_code
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_INITIAL = """\
You are an expert Python programmer. Given a problem description, write a \
correct and efficient Python function that solves it.

Rules:
- Output ONLY the function definition (with any necessary imports at the top).
- Use clear variable names and add a brief docstring.
- Do not include example usage, tests, or a main block.
- Wrap your code in a ```python code block.
"""

SYSTEM_PROMPT_REVISION = """\
You are an expert Python programmer revising a previous solution based on \
feedback.

You will receive:
1. The original problem description.
2. Your previous code.
3. Structured feedback describing what went wrong (test failures, lint \
issues, reviewer suggestions, etc.).

Rules:
- Carefully analyze every piece of feedback before writing code.
- Output the COMPLETE revised function, not just a diff.
- Use clear variable names and add a brief docstring.
- Do not include example usage, tests, or a main block.
- Wrap your code in a ```python code block.
"""


class ProposerAgent(Agent):
    """Agent that generates or revises Python code solutions.

    Operates in two modes:
      * **Initial generation** -- when no context is provided, produces a
        first-pass solution from the problem description alone.
      * **Revision** -- when context contains ``feedback`` and
        ``previous_code``, generates an improved version informed by the
        feedback.
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
        """Generate or revise a code solution.

        Args:
            problem_description: Natural-language description of the problem.
            context: Optional dict with keys such as ``feedback``,
                ``previous_code``, and ``iteration``.

        Returns:
            An AgentResponse containing the extracted code, the raw LLM
            reasoning / response, and token counts.
        """
        if context and "previous_code" in context:
            return self._revise(problem_description, context)
        return self._initial_generate(problem_description)

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    def _initial_generate(self, problem_description: str) -> AgentResponse:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_description}\n\n"
                    "Write a Python function that solves this problem."
                ),
            },
        ]

        response = self.client.generate(
            config=self.model_config,
            system=SYSTEM_PROMPT_INITIAL,
            messages=messages,
        )

        code = extract_code(response.content)
        logger.info(
            "Proposer initial generation: %d tokens (%d in + %d out)",
            response.total_tokens,
            response.input_tokens,
            response.output_tokens,
        )
        return AgentResponse(
            code=code,
            reasoning=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def _revise(
        self,
        problem_description: str,
        context: dict,
    ) -> AgentResponse:
        previous_code: str = context["previous_code"]
        feedback: str = context.get("feedback", "No specific feedback provided.")
        iteration: int = context.get("iteration", 0)

        user_content = (
            f"Problem:\n{problem_description}\n\n"
            f"--- Previous code (iteration {iteration}) ---\n"
            f"```python\n{previous_code}\n```\n\n"
            f"--- Feedback ---\n{feedback}\n\n"
            "Please provide a corrected and improved solution."
        )

        messages = [{"role": "user", "content": user_content}]

        response = self.client.generate(
            config=self.model_config,
            system=SYSTEM_PROMPT_REVISION,
            messages=messages,
        )

        code = extract_code(response.content)
        logger.info(
            "Proposer revision (iter %d): %d tokens (%d in + %d out)",
            iteration,
            response.total_tokens,
            response.input_tokens,
            response.output_tokens,
        )
        return AgentResponse(
            code=code,
            reasoning=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
