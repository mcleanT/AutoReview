"""Tests for partial extraction failure handling in PaperExtractor.extract_batch_safe."""

from __future__ import annotations

import pytest

from autoreview.extraction.extractor import ExtractionFailure, PaperExtractor
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.paper import CandidatePaper


def _make_paper(idx: int) -> CandidatePaper:
    return CandidatePaper(
        title=f"Paper {idx}",
        authors=["Author A"],
        source_database="test",
        abstract=f"Abstract for paper {idx}.",
    )


def _make_extraction(paper_id: str) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[
            Finding(
                claim="Test claim",
                evidence_strength=EvidenceStrength.MODERATE,
                paper_id=paper_id,
            )
        ],
        methods_summary="Test methods",
        limitations="Test limitations",
    )


class MockLLMAllSucceed:
    """Mock LLM that always returns a successful extraction."""

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ) -> LLMStructuredResponse:
        return LLMStructuredResponse(
            parsed=PaperExtraction(
                paper_id="placeholder",
                key_findings=[
                    Finding(
                        claim="Test claim",
                        evidence_strength=EvidenceStrength.MODERATE,
                        paper_id="placeholder",
                    )
                ],
                methods_summary="Test methods",
                limitations="Test limitations",
            ),
            input_tokens=100,
            output_tokens=50,
        )


class MockLLMFailOnSecond:
    """Mock LLM that raises on the second call to simulate a partial failure."""

    def __init__(self) -> None:
        self._call_count = 0

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ) -> LLMStructuredResponse:
        self._call_count += 1
        if self._call_count == 2:
            raise RuntimeError("LLM timeout on paper 2")
        return LLMStructuredResponse(
            parsed=PaperExtraction(
                paper_id="placeholder",
                key_findings=[
                    Finding(
                        claim="Test claim",
                        evidence_strength=EvidenceStrength.MODERATE,
                        paper_id="placeholder",
                    )
                ],
                methods_summary="Test methods",
                limitations="Test limitations",
            ),
            input_tokens=100,
            output_tokens=50,
        )


class TestExtractBatchSafe:
    @pytest.mark.asyncio
    async def test_single_failure_doesnt_crash_batch(self) -> None:
        """One paper failing should not prevent others from being extracted."""
        papers = [_make_paper(i) for i in range(3)]
        extractor = PaperExtractor(MockLLMFailOnSecond(), max_concurrent=1)

        extractions, failures = await extractor.extract_batch_safe(papers)

        # Two out of three papers should succeed
        assert len(extractions) == 2
        # Exactly one failure should be recorded
        assert len(failures) == 1
        assert isinstance(failures[0], ExtractionFailure)
        assert failures[0].paper_id == papers[1].id
        assert "LLM timeout" in failures[0].error

    @pytest.mark.asyncio
    async def test_all_succeed_returns_empty_failures(self) -> None:
        """When all papers extract successfully, the failures list is empty."""
        papers = [_make_paper(i) for i in range(3)]
        extractor = PaperExtractor(MockLLMAllSucceed(), max_concurrent=5)

        extractions, failures = await extractor.extract_batch_safe(papers)

        assert len(extractions) == 3
        assert failures == []
        # Each extraction should have the paper_id set to the actual paper id
        extracted_ids = {e.paper_id for e in extractions}
        expected_ids = {p.id for p in papers}
        assert extracted_ids == expected_ids
