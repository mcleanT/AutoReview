#!/usr/bin/env python3
"""Run the AutoReview pipeline locally with real Claude LLM and real literature search.

Uses PubMed, Semantic Scholar, and OpenAlex for paper discovery.
Uses ClaudeLLMProvider for all LLM stages (requires ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from autoreview.config import load_config
from autoreview.llm.claude import ClaudeLLMProvider
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.output.formatter import OutputFormatter
from autoreview.pipeline.runner import run_pipeline


async def main() -> None:
    topic = "Genetic Features of Cellular Senescence Across Different Organs"

    # Each run gets its own timestamped directory
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = f"output/{run_id}"

    print(f"{'=' * 70}")
    print("AutoReview LOCAL RUN (real Claude LLM, live search)")
    print(f"Topic: {topic}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Setup structured logging
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.processors.NAME_TO_LEVEL["info"]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    config = load_config(
        domain="general",
        overrides={
            "databases": {
                "primary": ["pubmed", "semantic_scholar"],
                "secondary": ["openalex"],
                "discovery": [],  # Perplexity needs an API key
            },
            "search": {
                "max_results_per_source": 50,
                "relevance_threshold": 3,
            },
        },
    )

    kb = KnowledgeBase(
        topic=topic,
        domain="general",
        output_dir=output_dir,
    )
    kb.save_snapshot("initialized")

    llm = ClaudeLLMProvider(
        model=config.llm.model,
        max_tokens_generate=config.llm.max_tokens_generate,
        max_tokens_structured=config.llm.max_tokens_structured,
    )

    kb = await run_pipeline(llm=llm, config=config, kb=kb)

    # Format output
    formatter = OutputFormatter(style=config.writing.citation_format)
    created = formatter.save(kb, output_dir, fmt="markdown")

    print(f"\n{'=' * 70}")
    print("Pipeline complete!")
    print(f"Phase: {kb.current_phase}")
    print(f"Papers: {len(kb.candidate_papers)} candidates, {len(kb.screened_papers)} screened")
    print(f"Extractions: {len(kb.extractions)}")
    print(f"Sections: {len(kb.section_drafts)}")
    print(f"Critique reports: {len(kb.critique_history)}")
    tokens = kb.total_tokens()
    print(f"Total tokens: {tokens['input_tokens']:,} input, {tokens['output_tokens']:,} output")
    print("\nOutput files:")
    for path in created:
        print(f"  -> {path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
