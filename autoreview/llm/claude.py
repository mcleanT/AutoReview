from __future__ import annotations

import os
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from autoreview.llm.provider import LLMResponse, LLMStructuredResponse

logger = structlog.get_logger()
T = TypeVar("T", bound=BaseModel)


class ClaudeLLMProvider:
    """Claude implementation of LLMProvider using Anthropic async SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 3,
        max_tokens_generate: int = 4096,
        max_tokens_structured: int = 4096,
    ) -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=0,  # We handle retries ourselves via tenacity
        )
        self.max_retries = max_retries
        self.max_tokens_generate = max_tokens_generate
        self.max_tokens_structured = max_tokens_structured

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Generate free-form text response."""
        effective_model = model_override or self.model
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens_generate,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        response = await self.client.messages.create(**kwargs)

        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        result = LLMResponse(
            content=response.content[0].text if response.content else "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            model=response.model,
        )

        logger.info(
            "llm.generate",
            model=response.model,
            requested_model=model_override,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

        return result

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.0,
        model_override: str | None = None,
    ) -> LLMStructuredResponse:
        """Generate structured output constrained to a Pydantic model schema."""
        effective_model = model_override or self.model
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens_structured,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        # Use Anthropic's native JSON schema output
        schema = response_model.model_json_schema()
        kwargs["tools"] = [
            {
                "name": "structured_output",
                "description": f"Return a {response_model.__name__} object",
                "input_schema": schema,
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        response = await self.client.messages.create(**kwargs)

        # Extract the tool use block
        tool_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_block = block
                break

        if tool_block is None:
            raise ValueError("No tool use block in response")

        parsed = response_model.model_validate(tool_block.input)

        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        result = LLMStructuredResponse(
            parsed=parsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            model=response.model,
        )

        logger.info(
            "llm.generate_structured",
            model=response.model,
            requested_model=model_override,
            response_model=response_model.__name__,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

        return result
