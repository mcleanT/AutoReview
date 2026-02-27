"""Tests for the LLM provider factory."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from autoreview.config.models import LLMConfig
from autoreview.llm.claude import ClaudeLLMProvider
from autoreview.llm.factory import _detect_provider, create_llm_provider
from autoreview.llm.ollama import OllamaLLMProvider


class TestDetectProvider:
    def test_colon_in_name_returns_ollama(self):
        assert _detect_provider("qwen3.5:35b") == "ollama"
        assert _detect_provider("llama3:8b") == "ollama"

    def test_no_colon_returns_claude(self):
        assert _detect_provider("claude-sonnet-4-20250514") == "claude"
        assert _detect_provider("claude-haiku-4-20250514") == "claude"

    def test_edge_cases(self):
        assert _detect_provider("my-custom-model") == "claude"
        assert _detect_provider("model:latest") == "ollama"


class TestCreateLLMProvider:
    def test_explicit_ollama(self):
        config = LLMConfig(model="qwen3.5:35b")
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.model == "qwen3.5:35b"

    def test_explicit_claude(self):
        config = LLMConfig(model="claude-sonnet-4-20250514", api_key="test-key")
        provider = create_llm_provider(config, provider="claude")
        assert isinstance(provider, ClaudeLLMProvider)
        assert provider.model == "claude-sonnet-4-20250514"

    def test_auto_detect_ollama(self):
        config = LLMConfig(model="qwen3.5:35b")
        provider = create_llm_provider(config)
        assert isinstance(provider, OllamaLLMProvider)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_auto_detect_claude(self):
        config = LLMConfig(model="claude-sonnet-4-20250514")
        provider = create_llm_provider(config)
        assert isinstance(provider, ClaudeLLMProvider)

    def test_config_provider_field(self):
        config = LLMConfig(model="some-model", provider="ollama")
        provider = create_llm_provider(config)
        assert isinstance(provider, OllamaLLMProvider)

    def test_explicit_arg_overrides_config(self):
        config = LLMConfig(model="qwen3.5:35b", provider="claude", api_key="test-key")
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)

    @patch.dict(os.environ, {}, clear=True)
    def test_claude_missing_api_key_raises(self):
        config = LLMConfig(model="claude-sonnet-4-20250514", api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_llm_provider(config, provider="claude")

    def test_unknown_provider_raises(self):
        config = LLMConfig(model="some-model")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_provider(config, provider="gpt")

    def test_ollama_base_url_passed(self):
        config = LLMConfig(model="qwen3.5:35b", ollama_base_url="http://myhost:11434")
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.base_url == "http://myhost:11434"

    def test_max_tokens_passed(self):
        config = LLMConfig(
            model="qwen3.5:35b", max_tokens_generate=8192, max_tokens_structured=2048
        )
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.max_tokens_generate == 8192
        assert provider.max_tokens_structured == 2048

    def test_ollama_keep_alive_passed(self):
        config = LLMConfig(model="qwen3.5:35b", ollama_keep_alive="2h")
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.keep_alive == "2h"

    def test_ollama_num_ctx_passed(self):
        config = LLMConfig(model="qwen3.5:35b", ollama_num_ctx=32768)
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.num_ctx == 32768

    def test_ollama_defaults(self):
        config = LLMConfig(model="qwen3.5:35b")
        provider = create_llm_provider(config, provider="ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.keep_alive == "1h"
        assert provider.num_ctx is None
