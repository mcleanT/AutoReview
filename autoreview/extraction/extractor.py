from __future__ import annotations

import asyncio
from typing import Any

import structlog

from autoreview.extraction.models import PaperExtraction
from autoreview.llm.prompts.extraction import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from autoreview.llm.prompts.screening import (
    SCREENING_SYSTEM_PROMPT,
    ScreeningBatchResult,
    build_screening_prompt,
)
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()


class PaperScreener:
    """Screens papers for relevance using LLM-based scoring."""

    def __init__(self, llm: Any, batch_size: int = 20) -> None:
        self.llm = llm
        self.batch_size = batch_size

    async def screen(
        self,
        papers: list[CandidatePaper],
        scope_document: str,
        threshold: int = 3,
    ) -> list[ScreenedPaper]:
        all_screened: list[ScreenedPaper] = []

        for batch_start in range(0, len(papers), self.batch_size):
            batch = papers[batch_start:batch_start + self.batch_size]
            batch_papers = [
                {"index": str(batch_start + i), "title": p.title, "abstract": p.abstract or ""}
                for i, p in enumerate(batch)
            ]

            prompt = build_screening_prompt(scope_document, batch_papers)
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=ScreeningBatchResult,
                system=SCREENING_SYSTEM_PROMPT,
            )
            result: ScreeningBatchResult = response.parsed

            logger.info(
                "screening.batch",
                batch_start=batch_start,
                batch_size=len(batch),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            for decision in result.decisions:
                paper_idx = decision.paper_index - batch_start
                if 0 <= paper_idx < len(batch):
                    paper = batch[paper_idx]
                    score = max(1, min(5, decision.relevance_score))
                    screened = ScreenedPaper(
                        paper=paper,
                        relevance_score=score,
                        rationale=decision.rationale,
                        include=score >= threshold,
                    )
                    if screened.include:
                        all_screened.append(screened)

        logger.info("screening.complete", total_papers=len(papers), included=len(all_screened))
        return all_screened


class PaperExtractor:
    """Extracts structured information from papers using LLM."""

    def __init__(
        self,
        llm: Any,
        domain_fields: dict[str, bool] | None = None,
        max_concurrent: int = 10,
    ) -> None:
        self.llm = llm
        self.domain_fields = domain_fields
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_text_and_source(self, paper: CandidatePaper) -> tuple[str, str]:
        if paper.full_text:
            return paper.full_text[:200_000], "full_text"
        elif paper.abstract:
            return paper.abstract, "abstract"
        else:
            return paper.title, "title_only"

    async def extract_one(self, paper: CandidatePaper) -> PaperExtraction:
        async with self._semaphore:
            text, source = self._get_text_and_source(paper)
            prompt = build_extraction_prompt(
                title=paper.title, text=text, text_source=source,
                domain_fields=self.domain_fields,
            )

            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=PaperExtraction,
                system=EXTRACTION_SYSTEM_PROMPT,
            )
            extraction: PaperExtraction = response.parsed
            extraction.paper_id = paper.id

            logger.info(
                "extraction.complete",
                paper_id=paper.id, title=paper.title[:60],
                findings=len(extraction.key_findings), text_source=source,
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            )
            return extraction

    async def extract_batch(self, papers: list[CandidatePaper]) -> dict[str, PaperExtraction]:
        tasks = [self.extract_one(p) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: dict[str, PaperExtraction] = {}
        for paper, result in zip(papers, results):
            if isinstance(result, Exception):
                logger.error("extraction.error", paper_id=paper.id, error=str(result))
                continue
            extractions[paper.id] = result

        logger.info(
            "extraction.batch_complete",
            total=len(papers), successful=len(extractions),
            failed=len(papers) - len(extractions),
        )
        return extractions
