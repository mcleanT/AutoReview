"""Tests for tiered model selection in extraction."""

from __future__ import annotations

import pytest

from autoreview.config.models import TieredModelConfig
from autoreview.extraction.extractor import PaperExtractor
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.paper import CandidatePaper, ScreenedPaper


class MockLLM:
    """Mock LLM that records model_override for each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096,
        temperature=0.0, model_override=None,
    ):
        self.calls.append({
            "prompt": prompt,
            "response_model": response_model,
            "model_override": model_override,
        })
        return LLMStructuredResponse(
            parsed=PaperExtraction(
                paper_id="placeholder",
                key_findings=[
                    Finding(
                        claim="Test finding",
                        evidence_strength=EvidenceStrength.MODERATE,
                        paper_id="placeholder",
                    )
                ],
                methods_summary="Test methods",
                limitations="Test limitations",
            ),
            input_tokens=1000,
            output_tokens=500,
            model=model_override or "claude-sonnet-4-20250514",
        )


def _make_paper(title: str = "Test Paper") -> CandidatePaper:
    return CandidatePaper(
        title=title, authors=["Author A"],
        source_database="test", abstract="Test abstract.",
    )


def _make_screened(score: int, title: str = "Test Paper") -> ScreenedPaper:
    return ScreenedPaper(
        paper=_make_paper(title),
        relevance_score=score,
        rationale="Test rationale",
        include=True,
    )


class TestModelSelection:
    def test_no_tiered_config_returns_none(self):
        extractor = PaperExtractor(MockLLM())
        assert extractor._select_model(5, "full_text") is None

    def test_disabled_tiered_config_returns_none(self):
        config = TieredModelConfig(enabled=False)
        extractor = PaperExtractor(MockLLM(), tiered_models=config)
        assert extractor._select_model(5, "full_text") is None

    def test_high_relevance_returns_sonnet(self):
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(MockLLM(), tiered_models=config)
        assert extractor._select_model(5, "full_text") == "claude-sonnet-4-20250514"
        assert extractor._select_model(4, "full_text") == "claude-sonnet-4-20250514"

    def test_moderate_relevance_returns_haiku(self):
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(MockLLM(), tiered_models=config)
        assert extractor._select_model(3, "full_text") == "claude-haiku-4-20250514"

    def test_unexpected_score_falls_back_to_high(self):
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(MockLLM(), tiered_models=config)
        # Score 1 or 2 not in either list — falls back to high relevance model
        assert extractor._select_model(1, "abstract") == "claude-sonnet-4-20250514"

    def test_custom_model_names(self):
        config = TieredModelConfig(
            enabled=True,
            high_relevance_model="custom-sonnet",
            moderate_relevance_model="custom-haiku",
        )
        extractor = PaperExtractor(MockLLM(), tiered_models=config)
        assert extractor._select_model(5, "full_text") == "custom-sonnet"
        assert extractor._select_model(3, "full_text") == "custom-haiku"


class TestExtractWithTieredModels:
    async def test_extract_one_passes_model_override(self):
        llm = MockLLM()
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(llm, tiered_models=config)
        paper = _make_paper()
        await extractor.extract_one(paper, relevance_score=3)
        assert llm.calls[0]["model_override"] == "claude-haiku-4-20250514"

    async def test_extract_one_default_score_uses_sonnet(self):
        llm = MockLLM()
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(llm, tiered_models=config)
        paper = _make_paper()
        await extractor.extract_one(paper)  # default relevance_score=5
        assert llm.calls[0]["model_override"] == "claude-sonnet-4-20250514"

    async def test_extract_from_screened(self):
        llm = MockLLM()
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(llm, tiered_models=config)
        sp = _make_screened(score=3)
        await extractor.extract_from_screened(sp)
        assert llm.calls[0]["model_override"] == "claude-haiku-4-20250514"

    async def test_extract_batch_with_screened_papers(self):
        llm = MockLLM()
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(llm, tiered_models=config)

        papers = [
            _make_screened(score=5, title="High relevance"),
            _make_screened(score=3, title="Moderate relevance"),
            _make_screened(score=4, title="Also high"),
        ]
        results = await extractor.extract_batch(papers)
        assert len(results) == 3

        # Check that the right models were used
        assert llm.calls[0]["model_override"] == "claude-sonnet-4-20250514"
        assert llm.calls[1]["model_override"] == "claude-haiku-4-20250514"
        assert llm.calls[2]["model_override"] == "claude-sonnet-4-20250514"

    async def test_extract_batch_with_candidate_papers(self):
        """Backward compat: CandidatePaper list still works."""
        llm = MockLLM()
        config = TieredModelConfig(enabled=True)
        extractor = PaperExtractor(llm, tiered_models=config)

        papers = [_make_paper(f"Paper {i}") for i in range(3)]
        results = await extractor.extract_batch(papers)
        assert len(results) == 3
        # CandidatePaper defaults to relevance_score=5 → Sonnet
        for call in llm.calls:
            assert call["model_override"] == "claude-sonnet-4-20250514"

    async def test_extract_batch_without_tiered_models(self):
        """Without tiered models, model_override should be None."""
        llm = MockLLM()
        extractor = PaperExtractor(llm)
        papers = [_make_paper()]
        await extractor.extract_batch(papers)
        assert llm.calls[0]["model_override"] is None
