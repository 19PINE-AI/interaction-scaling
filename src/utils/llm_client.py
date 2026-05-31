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
        self._openrouter: openai.OpenAI | None = None
        self._local_model = None
        self._local_tokenizer = None
        self._vllm_model = None
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

    @property
    def openrouter_client(self) -> openai.OpenAI:
        if self._openrouter is None:
            import os
            self._openrouter = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._openrouter

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
        elif config.provider == ModelProvider.OPENROUTER:
            response = self._call_openrouter(config, system, messages, temp)
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
        kwargs = dict(
            model=config.model_id,
            max_tokens=config.max_tokens,
            system=system,
            messages=messages,
        )

        # Enable extended thinking for distillation
        if config.use_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget,
            }
            kwargs["temperature"] = 1.0  # required for extended thinking
        else:
            kwargs["temperature"] = temperature

        response = self.anthropic_client.messages.create(**kwargs)

        thinking = ""
        content = ""
        for block in response.content:
            if block.type == "thinking":
                thinking += block.thinking
            elif block.type == "text":
                content += block.text

        # Wrap thinking in <think> tags for SFT training data
        if thinking:
            content = f"<think>\n{thinking}\n</think>\n{content}"

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

    def _call_openrouter(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """Call OpenRouter API using raw httpx (bypasses slow Pydantic parsing)."""
        import os
        import httpx

        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p["text"] for p in content
                             if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts) if text_parts else ""
            oai_messages.append({"role": msg["role"], "content": content})

        payload = {
            "model": config.model_id,
            "temperature": temperature,
            "messages": oai_messages,
        }
        if config.max_tokens > 0:
            payload["max_tokens"] = config.max_tokens

        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        reasoning = choice["message"].get("reasoning") or ""
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}"

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=config.model_id,
            stop_reason=choice.get("finish_reason"),
        )

    def _load_local_model(self, model_id: str):
        """Load a local model with vLLM for fast inference (lazy init)."""
        if self._vllm_model is not None and self._local_tokenizer is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer
        logger.info("Loading local model with vLLM: %s", model_id)
        self._local_tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._local_tokenizer.pad_token is None:
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token
        self._vllm_model = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
        )
        logger.info("Local model loaded with vLLM: %s", model_id)

    def _call_local(
        self,
        config: ModelConfig,
        system: str,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """Generate using vLLM for fast local inference.

        Preserves <think>...</think> tokens in the output so SFT data
        includes the model's reasoning traces.
        """
        from vllm import SamplingParams
        self._load_local_model(config.model_id)

        # Build chat messages
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
            text = self._local_tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = self._local_tokenizer.encode(text)
        input_len = len(input_ids)

        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=max(temperature, 0.01),
            top_p=0.95,
        )

        outputs = self._vllm_model.generate(
            prompts=[text],
            sampling_params=sampling_params,
        )

        generated_text = outputs[0].outputs[0].text
        output_len = len(outputs[0].outputs[0].token_ids)

        # Clean up EOS tokens but keep <think>/<|think|> tags
        for special in [self._local_tokenizer.eos_token, "<|im_end|>", "<|endoftext|>", "<end_of_turn>"]:
            if special:
                generated_text = generated_text.replace(special, "")
        content = generated_text.strip()

        return LLMResponse(
            content=content,
            input_tokens=input_len,
            output_tokens=output_len,
            model=config.model_id,
            stop_reason=outputs[0].outputs[0].finish_reason,
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
