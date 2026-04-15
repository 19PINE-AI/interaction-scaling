"""Single-agent agentic loop (Baseline B5).

A single agent generates code, executes it, reads the output, and revises
in a natural loop — the pattern used by most existing coding agents.
No separate reviewer; the same agent processes execution feedback.
"""

import logging

from src.agents.base import Agent, AgentResponse
from src.config import ModelConfig
from src.utils.code_utils import extract_code
from src.utils.llm_client import LLMResponse, get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Python programmer. You solve coding problems by writing code, \
executing it, observing the results, and iterating until the solution is correct.

When you receive execution feedback (test results, errors), analyze the output \
carefully and fix the issues in your next attempt.

Always output your complete solution inside a ```python code block."""

INITIAL_PROMPT = """\
Solve the following problem by writing a Python function.

{prompt}

Write your solution inside a ```python code block."""

REVISION_PROMPT = """\
Your previous solution had issues. Here is the feedback:

## Previous Code
```python
{previous_code}
```

## Execution Result
{execution_result}

Analyze the errors carefully and write a corrected solution. \
Output your complete fixed solution inside a ```python code block."""


class SingleAgentLoop(Agent):
    """Single-agent loop that generates, executes, and revises code.

    This is Baseline B5: the agent processes its own execution feedback
    without a separate reviewer agent.
    """

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig.claude_sonnet()
        self.client = get_client()

    def generate(
        self, problem_description: str, context: dict | None = None
    ) -> AgentResponse:
        if context and context.get("previous_code"):
            return self._revise(problem_description, context)
        return self._initial_generate(problem_description)

    def _initial_generate(self, problem_description: str) -> AgentResponse:
        prompt = INITIAL_PROMPT.format(prompt=problem_description)
        response = self.client.generate(
            config=self.model_config,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_response(response)

    def _revise(self, problem_description: str, context: dict) -> AgentResponse:
        prompt = REVISION_PROMPT.format(
            previous_code=context["previous_code"],
            execution_result=context.get("execution_result", "No output"),
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(
            config=self.model_config,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return self._parse_response(response)

    def _parse_response(self, response: LLMResponse) -> AgentResponse:
        code = extract_code(response.content)
        return AgentResponse(
            code=code,
            reasoning=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
