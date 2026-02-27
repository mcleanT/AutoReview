from __future__ import annotations

import os

import structlog

from autoreview.config.models import LLMConfig
from autoreview.llm.provider import LLMProvider

logger = structlog.get_logger()


def _detect_provider(model: str) -> str:
    """Auto-detect provider from model name. Models with ':' are Ollama, otherwise Claude."""
    return "ollama" if ":" in model else "claude"


def create_llm_provider(
    config: LLMConfig,
    provider: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance based on config and optional override.

    Resolution order: explicit arg > config.provider > auto-detect from model name.
    """
    resolved = provider or config.provider or _detect_provider(config.model)

    if resolved == "ollama":
        from autoreview.llm.ollama import OllamaLLMProvider

        logger.info(
            "llm.factory", provider="ollama", model=config.model, base_url=config.ollama_base_url
        )
        return OllamaLLMProvider(
            model=config.model,
            base_url=config.ollama_base_url,
            max_tokens_generate=config.max_tokens_generate,
            max_tokens_structured=config.max_tokens_structured,
            keep_alive=config.ollama_keep_alive,
            num_ctx=config.ollama_num_ctx,
        )

    if resolved == "claude":
        from autoreview.llm.claude import ClaudeLLMProvider

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it in your environment or .env file, "
                "or use --provider ollama with a local model."
            )
        logger.info("llm.factory", provider="claude", model=config.model)
        return ClaudeLLMProvider(
            model=config.model,
            api_key=api_key,
            max_tokens_generate=config.max_tokens_generate,
            max_tokens_structured=config.max_tokens_structured,
        )

    raise ValueError(f"Unknown provider '{resolved}'. Supported: 'claude', 'ollama'")
