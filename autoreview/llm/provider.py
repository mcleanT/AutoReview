from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Wrapper for free-form LLM responses with token usage metadata."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMStructuredResponse(BaseModel):
    """Wrapper for structured LLM responses with token usage metadata."""
    # Generic would be ideal but Pydantic BaseModel + Generic is tricky
    # Callers know the type from context
    parsed: object  # The parsed Pydantic model instance
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers. Implementations must support both free-form and structured output."""

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse: ...

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMStructuredResponse: ...
