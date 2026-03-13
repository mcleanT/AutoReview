from __future__ import annotations

import asyncio
from typing import Any

import structlog

from autoreview.config.models import SectionTruncationConfig, TieredModelConfig
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.truncation import (
    _head_tail_truncate,
    section_aware_truncate,
)
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
        self.borderline_papers: list[CandidatePaper] = []

        for batch_start in range(0, len(papers), self.batch_size):
            batch = papers[batch_start : batch_start + self.batch_size]
            batch_papers = [
                {"index": str(i), "title": p.title, "abstract": p.abstract or ""}
                for i, p in enumerate(batch)
            ]

            prompt = build_screening_prompt(scope_document, batch_papers)
            # 2048 tokens: ~100 tokens/decision × 20 papers + JSON overhead
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=ScreeningBatchResult,
                system=SCREENING_SYSTEM_PROMPT,
                max_tokens=2048,
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
                paper_idx = decision.paper_index
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
                    elif score == threshold - 1:
                        self.borderline_papers.append(paper)

        logger.info(
            "screening.complete",
            total_papers=len(papers),
            included=len(all_screened),
            borderline=len(self.borderline_papers),
        )
        return all_screened


class PaperExtractor:
    """Extracts structured information from papers using LLM."""

    def __init__(
        self,
        llm: Any,
        domain_fields: dict[str, bool] | None = None,
        max_concurrent: int = 10,
        full_text_max_chars: int = 80_000,
        tiered_models: TieredModelConfig | None = None,
        section_truncation: SectionTruncationConfig | None = None,
    ) -> None:
        self.llm = llm
        self.domain_fields = domain_fields
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.full_text_max_chars = full_text_max_chars
        self.tiered_models = tiered_models
        self.section_truncation = section_truncation

    def _get_text_and_source(self, paper: CandidatePaper) -> tuple[str, str]:
        if paper.full_text:
            truncated = self._truncate_full_text(paper.full_text)
            return truncated, "full_text"
        elif paper.abstract:
            return paper.abstract, "abstract"
        else:
            return paper.title, "title_only"

    def _truncate_full_text(self, text: str) -> str:
        """Truncate full text using section-aware or head/tail strategy."""
        if self.section_truncation and self.section_truncation.enabled:
            return section_aware_truncate(text, self.full_text_max_chars, self.section_truncation)
        return _head_tail_truncate(text, self.full_text_max_chars)

    def _select_model(self, relevance_score: int, text_source: str) -> str | None:
        """Select the model to use based on relevance score.

        Returns None to use the default model, or a model name string for override.
        """
        if not self.tiered_models or not self.tiered_models.enabled:
            return None

        if relevance_score in self.tiered_models.high_relevance_scores:
            return self.tiered_models.high_relevance_model
        if relevance_score in self.tiered_models.moderate_relevance_scores:
            return self.tiered_models.moderate_relevance_model

        # Fallback: use high-relevance model for unexpected scores
        return self.tiered_models.high_relevance_model

    async def extract_one(
        self,
        paper: CandidatePaper,
        relevance_score: int = 5,
    ) -> PaperExtraction:
        async with self._semaphore:
            text, source = self._get_text_and_source(paper)
            model_override = self._select_model(relevance_score, source)

            prompt = build_extraction_prompt(
                title=paper.title,
                text=text,
                text_source=source,
                domain_fields=self.domain_fields,
            )

            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=PaperExtraction,
                system=EXTRACTION_SYSTEM_PROMPT,
                model_override=model_override,
            )
            extraction: PaperExtraction = response.parsed
            extraction.paper_id = paper.id

            logger.info(
                "extraction.complete",
                paper_id=paper.id,
                title=paper.title[:60],
                findings=len(extraction.key_findings),
                text_source=source,
                model_used=getattr(response, "model", None),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            return extraction

    async def extract_from_screened(self, sp: ScreenedPaper) -> PaperExtraction:
        """Extract from a ScreenedPaper, using its relevance score for model routing."""
        return await self.extract_one(sp.paper, relevance_score=sp.relevance_score)

    async def extract_batch(
        self,
        papers: list[CandidatePaper] | list[ScreenedPaper],
    ) -> dict[str, PaperExtraction]:
        """Extract from a batch of papers (CandidatePaper or ScreenedPaper)."""
        tasks = []
        paper_ids = []
        for item in papers:
            if isinstance(item, ScreenedPaper):
                tasks.append(self.extract_from_screened(item))
                paper_ids.append(item.paper.id)
            else:
                tasks.append(self.extract_one(item))
                paper_ids.append(item.id)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: dict[str, PaperExtraction] = {}
        for pid, result in zip(paper_ids, results, strict=False):
            if isinstance(result, BaseException):
                logger.error("extraction.error", paper_id=pid, error=str(result))
                continue
            extractions[pid] = result

        logger.info(
            "extraction.batch_complete",
            total=len(papers),
            successful=len(extractions),
            failed=len(papers) - len(extractions),
        )
        return extractions
