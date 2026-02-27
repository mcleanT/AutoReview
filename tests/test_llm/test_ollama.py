"""Tests for OllamaLLMProvider."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from autoreview.llm.ollama import OllamaLLMProvider

BASE_URL = "http://localhost:11434"


class SimpleModel(BaseModel):
    name: str
    items: list[str] = Field(default_factory=list)


def _chat_response(
    content: str, model: str = "qwen3.5:35b", prompt_eval: int = 100, eval_count: int = 50
) -> dict:
    """Build a minimal Ollama /api/chat response body."""
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
    }


class TestGenerate:
    @respx.mock
    async def test_basic_generate(self):
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("Hello world"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        result = await provider.generate("Say hello")

        assert result.content == "Hello world"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model == "qwen3.5:35b"

    @respx.mock
    async def test_generate_with_system_prompt(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("response"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate("user prompt", system="system prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert len(request_body["messages"]) == 2
        assert request_body["messages"][0]["role"] == "system"
        assert request_body["messages"][0]["content"] == "system prompt"
        assert request_body["messages"][1]["role"] == "user"

    @respx.mock
    async def test_generate_model_override(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok", model="llama3:8b"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        result = await provider.generate("prompt", model_override="llama3:8b")

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["model"] == "llama3:8b"
        assert result.model == "llama3:8b"

    @respx.mock
    async def test_generate_passes_temperature_and_max_tokens(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate("prompt", temperature=0.7, max_tokens=2048)

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["options"]["temperature"] == 0.7
        assert request_body["options"]["num_predict"] == 2048

    @respx.mock
    async def test_generate_no_system_prompt(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert len(request_body["messages"]) == 1
        assert request_body["messages"][0]["role"] == "user"

    @respx.mock
    async def test_generate_thinking_model_disables_think(self):
        """Qwen3 models should send think: false in generate() payload."""
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("result"))
        )

        provider = OllamaLLMProvider(model="qwen3:8b", base_url=BASE_URL)
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["think"] is False

    @respx.mock
    async def test_generate_non_thinking_model_no_think_key(self):
        """Non-thinking models should NOT have think key in payload."""
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("result", model="llama3:8b"))
        )

        provider = OllamaLLMProvider(model="llama3:8b", base_url=BASE_URL)
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert "think" not in request_body

    @respx.mock
    async def test_generate_includes_keep_alive(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL, keep_alive="2h")
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["keep_alive"] == "2h"

    @respx.mock
    async def test_generate_includes_num_ctx(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL, num_ctx=32768)
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["options"]["num_ctx"] == 32768

    @respx.mock
    async def test_generate_no_num_ctx_when_none(self):
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate("prompt")

        request_body = json.loads(route.calls[0].request.content)
        assert "num_ctx" not in request_body["options"]


class TestGenerateStructured:
    @respx.mock
    async def test_basic_structured(self):
        parsed_json = json.dumps({"name": "test", "items": ["a", "b"]})
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        result = await provider.generate_structured("Extract data", response_model=SimpleModel)

        assert isinstance(result.parsed, SimpleModel)
        assert result.parsed.name == "test"
        assert result.parsed.items == ["a", "b"]
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @respx.mock
    async def test_structured_non_thinking_sends_schema_in_format(self):
        """Non-thinking models use Ollama's native format: schema."""
        parsed_json = json.dumps({"name": "x", "items": []})
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json, model="llama3:8b"))
        )

        provider = OllamaLLMProvider(model="llama3:8b", base_url=BASE_URL)
        await provider.generate_structured("prompt", response_model=SimpleModel)

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["format"] == SimpleModel.model_json_schema()
        assert request_body["stream"] is False
        assert "think" not in request_body

    @respx.mock
    async def test_structured_thinking_model_sends_think_false(self):
        """Thinking models (Qwen3) get think: false and no format key."""
        parsed_json = json.dumps({"name": "x", "items": []})
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate_structured("prompt", response_model=SimpleModel)

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["think"] is False
        assert "format" not in request_body

    @respx.mock
    async def test_structured_includes_schema_in_prompt(self):
        parsed_json = json.dumps({"name": "x", "items": []})
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        await provider.generate_structured("prompt", response_model=SimpleModel)

        request_body = json.loads(route.calls[0].request.content)
        user_message = request_body["messages"][-1]["content"]
        assert "JSON" in user_message
        assert "schema" in user_message.lower()

    @respx.mock
    async def test_structured_includes_keep_alive(self):
        parsed_json = json.dumps({"name": "x", "items": []})
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL, keep_alive="30m")
        await provider.generate_structured("prompt", response_model=SimpleModel)

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["keep_alive"] == "30m"

    @respx.mock
    async def test_structured_includes_num_ctx(self):
        parsed_json = json.dumps({"name": "x", "items": []})
        route = respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response(parsed_json))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL, num_ctx=16384)
        await provider.generate_structured("prompt", response_model=SimpleModel)

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["options"]["num_ctx"] == 16384


class TestPersistentClient:
    @respx.mock
    async def test_client_reused_across_calls(self):
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=_chat_response("ok"))
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        client_before = provider._client

        await provider.generate("first")
        await provider.generate("second")

        assert provider._client is client_before

    @respx.mock
    async def test_close_closes_client(self):
        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        assert not provider._client.is_closed

        await provider.close()
        assert provider._client.is_closed


class TestErrorHandling:
    @respx.mock
    async def test_http_error_raises(self):
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        with pytest.raises(httpx.HTTPStatusError):
            await provider.generate("prompt")

    @respx.mock
    async def test_missing_token_counts_default_to_zero(self):
        response_data = {
            "model": "qwen3.5:35b",
            "message": {"role": "assistant", "content": "hello"},
            "done": True,
        }
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        provider = OllamaLLMProvider(model="qwen3.5:35b", base_url=BASE_URL)
        result = await provider.generate("prompt")

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.content == "hello"
