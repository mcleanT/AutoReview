from __future__ import annotations

import json
import re
from typing import TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from autoreview.llm.provider import LLMResponse, LLMStructuredResponse

logger = structlog.get_logger()
T = TypeVar("T", bound=BaseModel)

OLLAMA_READ_TIMEOUT = 1200.0  # 20 min — thinking models are slow
OLLAMA_CONNECT_TIMEOUT = 30.0

# Qwen 3.x models use internal thinking that consumes num_predict tokens
# before producing visible output.
_THINKING_MODEL_PATTERNS = (re.compile(r"qwen3", re.IGNORECASE),)

# For free-form output, we keep thinking but strip the <think> tags.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _is_thinking_model(model: str) -> bool:
    return any(p.search(model) for p in _THINKING_MODEL_PATTERNS)


_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json(raw: str, model: str = "") -> dict:
    """Extract a JSON object from raw model output.

    Tries, in order:
    1. Direct ``json.loads``
    2. Strip markdown fences and parse
    3. Regex extract first ``{...}`` block
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    fence_match = _MARKDOWN_FENCE_RE.search(raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Extract first JSON object via greedy regex
    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    # All extraction strategies failed
    logger.warning(
        "llm.structured_no_json",
        provider="ollama",
        model=model,
        content_len=len(raw),
        content_preview=raw[:300],
    )
    raise json.JSONDecodeError(
        f"No valid JSON found in response ({len(raw)} chars)",
        raw[:200],
        0,
    )


class OllamaLLMProvider:
    """Ollama implementation of LLMProvider using the REST API."""

    def __init__(
        self,
        model: str = "qwen3.5:35b",
        base_url: str = "http://localhost:11434",
        max_tokens_generate: int = 4096,
        max_tokens_structured: int = 4096,
        keep_alive: str = "1h",
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens_generate = max_tokens_generate
        self.max_tokens_structured = max_tokens_structured
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._is_thinking = _is_thinking_model(model)
        self._http_timeout = httpx.Timeout(
            connect=OLLAMA_CONNECT_TIMEOUT,
            read=OLLAMA_READ_TIMEOUT,
            write=30.0,
            pool=30.0,
        )
        self._client = httpx.AsyncClient(timeout=self._http_timeout)

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Generate free-form text response via Ollama /api/chat."""
        effective_model = model_override or self.model
        is_thinking = _is_thinking_model(effective_model)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        num_predict = max_tokens if max_tokens is not None else self.max_tokens_generate

        options: dict = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        payload: dict = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

        if is_thinking:
            payload["think"] = False

        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        # Safety fallback: strip <think> tags in case think:false isn't honored
        if is_thinking and content:
            content = _THINK_TAG_RE.sub("", content).strip()

        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        result = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=data.get("model", effective_model),
        )

        logger.info(
            "llm.generate",
            provider="ollama",
            model=result.model,
            requested_model=model_override,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result

    @retry(
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ReadTimeout, json.JSONDecodeError, ValidationError)
        ),
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
        """Generate structured output via Ollama.

        For thinking models (Qwen 3.x), we disable thinking via the API
        (``think: false``) and use prompt-based JSON instruction. Ollama's
        ``format: schema`` is NOT compatible with thinking models — it
        causes empty output because thinking tokens consume the entire
        ``num_predict`` budget.

        For non-thinking models, we use Ollama's native ``format: schema``.
        """
        effective_model = model_override or self.model
        is_thinking = _is_thinking_model(effective_model)
        schema = response_model.model_json_schema()

        schema_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Return ONLY the JSON object, no other text."
        )

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt + schema_instruction})

        num_predict = max_tokens if max_tokens is not None else self.max_tokens_structured

        options: dict = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        payload: dict = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

        if is_thinking:
            # Disable thinking at the API level — thinking models with
            # think=false produce clean JSON from prompt instructions alone.
            # Do NOT use format: schema with thinking models (it's ignored
            # when thinking is active and produces prose when disabled).
            payload["think"] = False
        else:
            # Non-thinking models: use Ollama's native constrained decoding
            payload["format"] = schema

        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        raw_content = data.get("message", {}).get("content", "")

        if not raw_content.strip():
            eval_count = data.get("eval_count", 0) or 0
            logger.warning(
                "llm.structured_empty_content",
                provider="ollama",
                model=effective_model,
                eval_count=eval_count,
                num_predict=num_predict,
            )
            raise json.JSONDecodeError(
                f"Empty response (eval_count={eval_count}, num_predict={num_predict})",
                raw_content or "",
                0,
            )

        parsed_json = _extract_json(raw_content, effective_model)

        parsed = response_model.model_validate(parsed_json)

        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        result = LLMStructuredResponse(
            parsed=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=data.get("model", effective_model),
        )

        logger.info(
            "llm.generate_structured",
            provider="ollama",
            model=result.model,
            requested_model=model_override,
            response_model=response_model.__name__,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result
