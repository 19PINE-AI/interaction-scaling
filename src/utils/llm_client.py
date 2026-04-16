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
    """Unified client for Anthropic, OpenAI, and local models with built-in token tracking."""

    def __init__(self):
        self._anthropic: anthropic.Anthropic | None = None
        self._openai: openai.OpenAI | None = None
        self._local_model = None
        self._local_tokenizer = None
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
        elif config.provider == ModelProvider.LOCAL:
            response = self._call_local(config, system, messages, temp)
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

    def _load_local_model(self, model_id: str):
        """Load a local model with transformers (lazy init)."""
        if self._local_model is not None and self._local_tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading local model: %s", model_id)
        self._local_tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._local_tokenizer.pad_token is None:
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token
        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._local_model.eval()
        logger.info("Local model loaded on %s", self._local_model.device)

    def _call_local(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """Generate using a local transformers model (e.g. Qwen3-8B).

        Preserves <think>...</think> tokens in the output so SFT data
        includes the model's reasoning traces.
        """
        import torch
        self._load_local_model(config.model_id)

        # Build chat messages in Qwen format
        chat_messages = [{"role": "system", "content": system}]
        for msg in messages:
            # Handle multimodal messages (skip image content for local model)
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts) if text_parts else ""
            chat_messages.append({"role": msg["role"], "content": content})

        # Chat template with thinking enabled (works for both Qwen3 and Gemma 4)
        try:
            text = self._local_tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            text = self._local_tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self._local_tokenizer(text, return_tensors="pt").to(self._local_model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=max(temperature, 0.01),  # avoid 0.0 for sampling
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self._local_tokenizer.pad_token_id,
            )

        output_ids = outputs[0][input_len:]
        # Decode full output including <think> tags
        content = self._local_tokenizer.decode(output_ids, skip_special_tokens=False)
        # Clean up EOS tokens but keep <think> tags
        for special in [self._local_tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"]:
            if special:
                content = content.replace(special, "")
        content = content.strip()

        output_len = len(output_ids)
        return LLMResponse(
            content=content,
            input_tokens=input_len,
            output_tokens=output_len,
            model=config.model_id,
            stop_reason="stop",
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
