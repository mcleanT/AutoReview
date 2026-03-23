"""Claude Code CLI provider for LLMProvider protocol.

Routes LLM calls through ``claude -p`` (non-interactive mode), using the
user's existing Claude Code authentication.  No API key required.

This lets the full Python pipeline run natively — KnowledgeBase in memory,
deterministic steps as Python, critique loops, Pydantic validation — with
only the actual LLM generation calls going through the CLI.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypeVar, cast

import structlog
from pydantic import BaseModel, ValidationError

from autoreview.llm.provider import LLMResponse, LLMStructuredResponse

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

# Model alias mapping — translate pipeline model names to CLI aliases
_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-20250514": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    "claude-opus-4-20250514": "opus",
}

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from raw model output."""
    raw = raw.strip()

    try:
        return cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError:
        pass

    fence_match = _MARKDOWN_FENCE_RE.search(raw)
    if fence_match:
        try:
            return cast(dict[str, Any], json.loads(fence_match.group(1).strip()))
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj_match:
        try:
            return cast(dict[str, Any], json.loads(obj_match.group()))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"No valid JSON found in response ({len(raw)} chars)",
        raw[:200],
        0,
    )


class ClaudeCodeProvider:
    """LLM provider that routes calls through the ``claude`` CLI.

    Uses ``claude -p`` (print mode) with ``--output-format json`` to get
    structured results including token usage.  Model selection via ``--model``.
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_tokens_generate: int = 4096,
        max_tokens_structured: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._model = _MODEL_ALIASES.get(model, model)
        self._max_tokens_generate = max_tokens_generate
        self._max_tokens_structured = max_tokens_structured
        self._max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Generate free-form text via ``claude -p``."""
        model = _MODEL_ALIASES.get(model_override or "", model_override) or self._model

        full_prompt = prompt
        if system:
            full_prompt = f"[System: {system}]\n\n{prompt}"

        result = await self._call_cli(full_prompt, model)

        content = result.get("result", "")
        usage = result.get("usage", {})

        return LLMResponse(
            content=content,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            model=model,
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
        """Generate structured output via ``claude -p`` with JSON schema instruction."""
        model = _MODEL_ALIASES.get(model_override or "", model_override) or self._model
        schema = response_model.model_json_schema()

        schema_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this exact schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Return ONLY the JSON object, no other text or explanation."
        )

        full_prompt = prompt + schema_instruction
        if system:
            full_prompt = f"[System: {system}]\n\n{full_prompt}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            result = await self._call_cli(full_prompt, model)
            content = result.get("result", "")
            usage = result.get("usage", {})

            try:
                parsed_dict = _extract_json(content)
                parsed_obj = response_model.model_validate(parsed_dict)

                return LLMStructuredResponse(
                    parsed=parsed_obj,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    model=model,
                )
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "claude_code.structured_retry",
                    attempt=attempt + 1,
                    error=str(e)[:200],
                    content_preview=content[:200],
                )

        raise ValueError(
            f"Failed to parse structured output after {self._max_retries} attempts: {last_error}"
        )

    async def _call_cli(self, prompt: str, model: str) -> dict[str, Any]:
        """Execute ``claude -p`` and return the parsed JSON result."""
        cmd = [
            "claude",
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--disallowedTools",
            "Read,Write,Edit,Bash,Glob,Grep,Agent,NotebookEdit,WebFetch,WebSearch",
        ]

        logger.debug(
            "claude_code.call",
            model=model,
            prompt_len=len(prompt),
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate(input=prompt.encode())

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "unknown error"
            logger.error(
                "claude_code.cli_error", returncode=proc.returncode, stderr=error_msg[:500]
            )
            raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {error_msg[:500]}")

        try:
            return cast(dict[str, Any], json.loads(stdout.decode()))
        except json.JSONDecodeError:
            raw = stdout.decode()
            return {"result": raw, "usage": {}}
