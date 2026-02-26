"""Contextual enrichment — broadens review sections with adjacent knowledge."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from autoreview.llm.prompts.enrichment import (
    CONTEXTUAL_EXTRACTION_SYSTEM_PROMPT,
    ENRICHMENT_QUERY_SYSTEM_PROMPT,
    BatchContextualExtractionResult,
    ContextualExtractionResult,
    EnrichmentQueryResult,
    build_batch_contextual_extraction_prompt,
    build_contextual_extraction_prompt,
    build_enrichment_query_prompt,
)
from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.models.enrichment import (
    ContextualExtraction,
    SectionEnrichmentQueries,
)
from autoreview.models.narrative import NarrativePlan
from autoreview.models.paper import CandidatePaper

logger = structlog.get_logger()


class ContextualEnricher:
    """Generates enrichment queries and extracts contextual material."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def generate_queries(
        self,
        outline: ReviewOutline,
        narrative_plan: NarrativePlan | None,
        scope_document: str,
    ) -> dict[str, SectionEnrichmentQueries]:
        """Generate enrichment queries for each top-level section.

        Returns:
            Dict mapping section_id to SectionEnrichmentQueries.
        """
        directive_map: dict[str, str] = {}
        if narrative_plan:
            for d in narrative_plan.section_directives:
                directive_map[d.section_id] = (
                    f"Role: {d.narrative_role}\n"
                    f"Central claim: {d.central_claim}\n"
                    f"Structure: {d.structural_suggestion}"
                )

        sem = asyncio.Semaphore(3)

        async def _generate_for_section(section):
            async with sem:
                prompt = build_enrichment_query_prompt(
                    section_id=section.id,
                    section_title=section.title,
                    section_description=section.description,
                    narrative_directive=directive_map.get(section.id, ""),
                    scope_document=scope_document,
                )
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=EnrichmentQueryResult,
                    system=ENRICHMENT_QUERY_SYSTEM_PROMPT,
                    max_tokens=1024,
                )
                result: EnrichmentQueryResult = response.parsed

                logger.info(
                    "contextual_enricher.queries_generated",
                    section_id=section.id,
                    query_count=len(result.queries),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )

                return SectionEnrichmentQueries(
                    section_id=section.id,
                    section_title=section.title,
                    queries=result.queries,
                )

        tasks = [_generate_for_section(s) for s in outline.sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        queries_map: dict[str, SectionEnrichmentQueries] = {}
        for i, r in enumerate(results):
            section = outline.sections[i]
            if isinstance(r, Exception):
                logger.warning(
                    "contextual_enricher.query_generation_failed",
                    section_id=section.id,
                    error=str(r),
                )
            else:
                queries_map[section.id] = r

        return queries_map

    async def extract_contextual(
        self,
        paper: CandidatePaper,
        section_title: str,
        section_description: str,
    ) -> ContextualExtraction:
        """Extract contextual information from a single paper."""
        prompt = build_contextual_extraction_prompt(
            paper_title=paper.title,
            paper_abstract=paper.abstract or "",
            section_title=section_title,
            section_description=section_description,
        )

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=ContextualExtractionResult,
            system=CONTEXTUAL_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=1024,
        )
        result: ContextualExtractionResult = response.parsed

        extraction = result.to_contextual_extraction()
        # Ensure paper_id matches the actual paper
        extraction = extraction.model_copy(update={
            "paper_id": paper.id if hasattr(paper, "id") else paper.doi or paper.title,
            "paper_title": paper.title,
        })

        logger.info(
            "contextual_enricher.extraction_complete",
            paper_title=paper.title[:60],
            concepts=len(extraction.key_concepts),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return extraction

    async def extract_contextual_batch(
        self,
        papers: list[CandidatePaper],
        section_title: str,
        section_description: str,
        max_concurrent: int = 5,
        batch_size: int = 5,
    ) -> list[ContextualExtraction]:
        """Extract contextual information from multiple papers.

        Papers are grouped into batches (default 5) and each batch is sent
        as a single LLM call to reduce overhead. Batches run concurrently
        up to ``max_concurrent``.
        """
        if not papers:
            return []

        sem = asyncio.Semaphore(max_concurrent)

        async def _extract_batch(batch: list[CandidatePaper]) -> list[ContextualExtraction]:
            async with sem:
                paper_dicts = [
                    {
                        "id": p.id if hasattr(p, "id") else p.doi or p.title,
                        "title": p.title,
                        "abstract": p.abstract or "",
                    }
                    for p in batch
                ]
                prompt = build_batch_contextual_extraction_prompt(
                    papers=paper_dicts,
                    section_title=section_title,
                    section_description=section_description,
                )
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=BatchContextualExtractionResult,
                    system=CONTEXTUAL_EXTRACTION_SYSTEM_PROMPT,
                    max_tokens=2048,
                )
                result: BatchContextualExtractionResult = response.parsed

                logger.info(
                    "contextual_enricher.batch_extraction_complete",
                    batch_size=len(batch),
                    results=len(result.extractions),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )

                extractions = []
                for ext_result in result.extractions:
                    ext = ext_result.to_contextual_extraction()
                    extractions.append(ext)
                return extractions

        # Split papers into batches
        batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
        tasks = [_extract_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_extractions: list[ContextualExtraction] = []
        failed = 0
        for r in results:
            if isinstance(r, Exception):
                logger.warning(
                    "contextual_enricher.batch_extraction_failed",
                    error=str(r),
                )
                failed += 1
            else:
                all_extractions.extend(r)

        logger.info(
            "contextual_enricher.all_batches_complete",
            total=len(papers),
            successful=len(all_extractions),
            failed_batches=failed,
        )

        return all_extractions
