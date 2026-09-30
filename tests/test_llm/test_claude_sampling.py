"""Sampling-parameter handling in ClaudeLLMProvider.

Requests go through the real Anthropic SDK; only the HTTP transport is replaced,
so these tests assert on the JSON body the API would actually receive.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from autoreview.llm import model_capabilities as mc
from autoreview.llm.claude import ClaudeLLMProvider


class _Answer(BaseModel):
    value: str


def _message_body(model: str, structured: bool) -> dict[str, Any]:
    content: list[dict[str, Any]]
    if structured:
        content = [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "structured_output",
                "input": {"value": "x"},
            }
        ]
    else:
        content = [{"type": "text", "text": "hello"}]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _provider(model: str, captured: list[dict[str, Any]], structured: bool) -> ClaudeLLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json=_message_body(body["model"], structured))

    provider = ClaudeLLMProvider(model=model, api_key="test-key")
    provider.client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


REJECTING = [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-fable-5",
    "claude-fable-5-1",
]
ACCEPTING = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-6",
]


@pytest.mark.parametrize("model", REJECTING)
async def test_generate_omits_temperature_for_rejecting_models(model: str) -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(model, captured, structured=False)
    await provider.generate("hi", temperature=0.3)
    assert len(captured) == 1
    for key in ("temperature", "top_p", "top_k"):
        assert key not in captured[0]


@pytest.mark.parametrize("model", REJECTING)
async def test_structured_omits_temperature_for_rejecting_models(model: str) -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(model, captured, structured=True)
    result = await provider.generate_structured("hi", _Answer, temperature=0.0)
    assert result.parsed.value == "x"
    assert "temperature" not in captured[0]


@pytest.mark.parametrize("model", ACCEPTING)
async def test_generate_keeps_temperature_for_older_models(model: str) -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(model, captured, structured=False)
    await provider.generate("hi", temperature=0.3)
    assert captured[0]["temperature"] == 0.3


async def test_structured_keeps_temperature_zero_for_older_models() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider("claude-sonnet-4-6", captured, structured=True)
    await provider.generate_structured("hi", _Answer, temperature=0.0)
    assert captured[0]["temperature"] == 0.0


async def test_model_override_decides_capability() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider("claude-sonnet-4-6", captured, structured=False)
    await provider.generate("hi", temperature=0.3, model_override="claude-sonnet-5-5")
    assert captured[0]["model"] == "claude-sonnet-5-5"
    assert "temperature" not in captured[0]


async def test_unknown_model_omits_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[dict[str, Any]] = []
    monkeypatch.setattr(mc, "_warned_unknown", set())
    monkeypatch.setattr(
        mc.logger, "warning", lambda event, **kw: warnings.append({"event": event, **kw})
    )
    captured: list[dict[str, Any]] = []
    provider = _provider("claude-future-9", captured, structured=False)
    await provider.generate("hi", temperature=0.3)
    await provider.generate("hi", temperature=0.3)
    assert "temperature" not in captured[0]
    assert "temperature" not in captured[1]
    # One warning per unknown id, not one per call.
    assert [w["event"] for w in warnings] == ["llm.sampling.unknown_model"]
    assert warnings[0]["model"] == "claude-future-9"


def test_capability_lookup_normalizes_ids() -> None:
    assert mc.sampling_support("claude-opus-4-8[1m]") is mc.SamplingSupport.REJECTED
    assert mc.sampling_support("claude-opus-4-5-20251101") is mc.SamplingSupport.ACCEPTED
    assert mc.sampling_support("claude-sonnet-5-5") is mc.SamplingSupport.DEFAULT_ONLY
    assert mc.sampling_support("claude-haiku-4-20250514") is mc.SamplingSupport.UNKNOWN


def test_sampling_params_drops_top_p_and_top_k_on_rejecting_model() -> None:
    assert mc.sampling_params("claude-opus-5-5", temperature=0.7, top_p=0.9, top_k=5) == {}
    assert mc.sampling_params("claude-sonnet-4-6", temperature=0.7, top_k=5) == {
        "temperature": 0.7,
        "top_k": 5,
    }
