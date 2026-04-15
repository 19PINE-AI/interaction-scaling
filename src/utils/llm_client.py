"""Unified LLM client supporting Anthropic and OpenAI APIs with token tracking."""

import json
import logging
from dataclasses import dataclass

import anthropic
import openai

from src.config import ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call with token accounting."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    """Unified client for Anthropic and OpenAI models with built-in token tracking."""

    def __init__(self):
        self._anthropic: anthropic.Anthropic | None = None
        self._openai: openai.OpenAI | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    @property
    def anthropic_client(self) -> anthropic.Anthropic:
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic()
        return self._anthropic

    @property
    def openai_client(self) -> openai.OpenAI:
        if self._openai is None:
            self._openai = openai.OpenAI()
        return self._openai

    def generate(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a response from the specified model."""
        temp = temperature if temperature is not None else config.temperature

        if config.provider == ModelProvider.ANTHROPIC:
            response = self._call_anthropic(config, system, messages, temp)
        elif config.provider == ModelProvider.OPENAI:
            response = self._call_openai(config, system, messages, temp)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.call_count += 1

        logger.debug(
            "LLM call #%d: %d in + %d out = %d tokens (%s)",
            self.call_count,
            response.input_tokens,
            response.output_tokens,
            response.total_tokens,
            config.model_id,
        )
        return response

    def _call_anthropic(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        response = self.anthropic_client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=config.model_id,
            stop_reason=response.stop_reason,
        )

    def _call_openai(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.openai_client.chat.completions.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=temperature,
            messages=oai_messages,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=config.model_id,
            stop_reason=choice.finish_reason,
        )

    def reset_counters(self):
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def get_usage_summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": self.call_count,
        }


# Singleton client instance
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Get the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
