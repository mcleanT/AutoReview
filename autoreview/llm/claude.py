from __future__ import annotations

import os
from typing import TypeVar

import anthropic
import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from autoreview.llm.provider import LLMProvider, LLMResponse, LLMStructuredResponse

logger = structlog.get_logger()
T = TypeVar("T", bound=BaseModel)


class ClaudeLLMProvider:
    """Claude implementation of LLMProvider using Anthropic async SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=0,  # We handle retries ourselves via tenacity
        )
        self.max_retries = max_retries

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate free-form text response."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        result = LLMResponse(
            content=response.content[0].text if response.content else "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

        logger.info(
            "llm.generate",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMStructuredResponse:
        """Generate structured output constrained to a Pydantic model schema."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

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

        result = LLMStructuredResponse(
            parsed=parsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

        logger.info(
            "llm.generate_structured",
            model=response.model,
            response_model=response_model.__name__,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return result
