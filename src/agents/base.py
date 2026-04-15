"""Base classes for agents in the interaction scaling framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Response from an agent, wrapping generated content with token accounting."""

    code: str
    reasoning: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Agent(ABC):
    """Abstract base class for all agents.

    Agents consume a problem description and optional context (feedback,
    previous code, iteration number, etc.) and produce an AgentResponse.
    """

    @abstractmethod
    def generate(
        self,
        problem_description: str,
        context: dict | None = None,
    ) -> AgentResponse:
        """Generate a response for the given problem.

        Args:
            problem_description: Natural-language description of the coding
                problem to solve.
            context: Optional dictionary that may contain:
                - feedback: str or dict with execution / review feedback
                - previous_code: str of the last attempted solution
                - iteration: int indicating the current refinement round
                - Any other agent-specific keys.

        Returns:
            An AgentResponse with the generated code, reasoning, and token
            usage information.
        """
