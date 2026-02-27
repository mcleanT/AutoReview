from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.extraction.extractor import PaperScreener, PaperExtractor
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.llm.prompts.screening import ScreeningBatchResult, ScreeningDecision


class MockLLM:
    """Mock LLM that returns predefined structured responses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0, model_override=None):
        self.calls.append({"prompt": prompt, "response_model": response_model, "model_override": model_override})

        if response_model == ScreeningBatchResult:
            return LLMStructuredResponse(
                parsed=ScreeningBatchResult(decisions=[
                    ScreeningDecision(paper_index=0, relevance_score=4, rationale="Relevant"),
                    ScreeningDecision(paper_index=1, relevance_score=2, rationale="Marginal"),
                    ScreeningDecision(paper_index=2, relevance_score=5, rationale="Highly relevant"),
                ]),
                input_tokens=500, output_tokens=200,
            )
        elif response_model == PaperExtraction:
            return LLMStructuredResponse(
                parsed=PaperExtraction(
                    paper_id="placeholder",
                    key_findings=[
                        Finding(claim="Test finding", evidence_strength=EvidenceStrength.MODERATE, paper_id="placeholder")
                    ],
                    methods_summary="Test methods",
                    limitations="Test limitations",
                ),
                input_tokens=1000, output_tokens=500,
            )

        raise ValueError(f"Unexpected model: {response_model}")


class TestPaperScreener:
    async def test_screens_papers(self):
        llm = MockLLM()
        screener = PaperScreener(llm, batch_size=10)
        papers = [
            CandidatePaper(title=f"Paper {i}", authors=["A"], source_database="test", abstract=f"Abstract {i}")
            for i in range(3)
        ]
        result = await screener.screen(papers, scope_document="Test scope", threshold=3)
        assert len(result) == 2
        assert all(sp.include for sp in result)

    async def test_batching(self):
        llm = MockLLM()
        screener = PaperScreener(llm, batch_size=2)
        papers = [
            CandidatePaper(title=f"Paper {i}", authors=["A"], source_database="test")
            for i in range(5)
        ]
        await screener.screen(papers, scope_document="Test scope")
        assert len(llm.calls) == 3


class TestPaperExtractor:
    async def test_extract_one(self):
        llm = MockLLM()
        extractor = PaperExtractor(llm, max_concurrent=5)
        paper = CandidatePaper(
            title="Test Paper", authors=["Author A"],
            source_database="test", abstract="This paper studies X.",
        )
        result = await extractor.extract_one(paper)
        assert result.paper_id == paper.id
        assert len(result.key_findings) == 1

    async def test_extract_batch(self):
        llm = MockLLM()
        extractor = PaperExtractor(llm, max_concurrent=5)
        papers = [
            CandidatePaper(title=f"Paper {i}", authors=["A"], source_database="test", abstract=f"Abstract {i}")
            for i in range(3)
        ]
        results = await extractor.extract_batch(papers)
        assert len(results) == 3
        for paper in papers:
            assert paper.id in results

    async def test_text_source_hierarchy(self):
        llm = MockLLM()
        extractor = PaperExtractor(llm)

        p1 = CandidatePaper(title="T", authors=["A"], source_database="test", abstract="Ab", full_text="Full")
        assert extractor._get_text_and_source(p1) == ("Full", "full_text")

        p2 = CandidatePaper(title="T", authors=["A"], source_database="test", abstract="Ab only")
        assert extractor._get_text_and_source(p2)[1] == "abstract"

        p3 = CandidatePaper(title="Title Only", authors=["A"], source_database="test")
        assert extractor._get_text_and_source(p3)[1] == "title_only"
