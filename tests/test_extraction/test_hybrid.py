"""Tests for the hybrid extractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from autoreview.extraction.hybrid import HybridExtractor

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction, StudyDesign
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.models.paper import CandidatePaper, ScreenedPaper


def _make_screened_paper(title="Test Paper", abstract="Test abstract about ML.", full_text=None):
    paper = CandidatePaper(
        title=title,
        authors=["A"],
        abstract=abstract,
        full_text=full_text,
        source_database="test",
    )
    return ScreenedPaper(paper=paper, relevance_score=4, rationale="test", include=True)


@pytest.mark.asyncio
class TestHybridExtractor:
    async def test_abstract_only_short_circuits(self):
        """Abstract-only papers skip LLM."""
        sp = _make_screened_paper(full_text=None)
        mock_llm = AsyncMock()
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        result = await hybrid.extract(sp)
        mock_llm.generate_structured.assert_not_called()
        assert result.paper_id == sp.paper.id
        assert len(result.key_findings) > 0  # programmatic still extracts from abstract

    async def test_refine_with_structured_llm(self):
        """API mode uses generate_structured."""
        sp = _make_screened_paper(full_text="Full text with methods and results sections.")
        pid = sp.paper.id
        mock_response = MagicMock()
        mock_response.parsed = PaperExtraction(
            paper_id=pid,
            key_findings=[
                Finding(claim="Refined", evidence_strength=EvidenceStrength.STRONG, paper_id=pid)
            ],
            methods_summary="Refined methods.",
            limitations="Refined limitations.",
            study_design=StudyDesign.META_ANALYSIS,  # Intentionally different
            quality_score=0.99,  # Intentionally different
        )
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(return_value=mock_response)
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        result = await hybrid.extract(sp)
        assert result.key_findings[0].claim == "Refined"
        # study_design and quality_score come from PROGRAMMATIC, not LLM
        assert result.study_design != StudyDesign.META_ANALYSIS
        assert result.quality_score != 0.99
        mock_llm.generate_structured.assert_called_once()

    async def test_fallback_on_parse_failure(self):
        """When LLM returns garbage, falls back to programmatic draft."""
        sp = _make_screened_paper(full_text="Some full text content here.")
        mock_llm = MagicMock()
        # Remove generate_structured to force generate path
        mock_llm.generate_structured = None  # Not callable
        mock_llm.generate = AsyncMock(return_value=MagicMock(content="not valid json"))
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        result = await hybrid.extract(sp)
        assert len(result.key_findings) > 0  # Falls back to programmatic
        assert result.paper_id == sp.paper.id

    async def test_claude_p_mode_parse_json(self):
        """When generate_structured is absent, parses JSON from generate() response."""
        sp = _make_screened_paper(full_text="Full text with results showing 95% accuracy.")
        pid = sp.paper.id
        refined = PaperExtraction(
            paper_id=pid,
            key_findings=[
                Finding(
                    claim="95% accuracy achieved",
                    evidence_strength=EvidenceStrength.MODERATE,
                    quantitative_result="95%",
                    paper_id=pid,
                )
            ],
            methods_summary="Evaluated on benchmark dataset.",
            limitations="Limited to single domain.",
            study_design=StudyDesign.COMPUTATIONAL,
            quality_score=0.75,
        )
        mock_llm = MagicMock()
        mock_llm.generate_structured = None
        mock_llm.generate = AsyncMock(return_value=MagicMock(content=refined.model_dump_json()))
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        result = await hybrid.extract(sp)
        assert result.key_findings[0].claim == "95% accuracy achieved"
        assert result.methods_summary == "Evaluated on benchmark dataset."
        # study_design and quality_score from programmatic, not LLM
        assert result.study_design != StudyDesign.META_ANALYSIS

    async def test_study_design_overridden_from_programmatic(self):
        """study_design is always taken from programmatic draft regardless of LLM output."""
        sp = _make_screened_paper(
            full_text="We conducted a randomized controlled trial (RCT). Results showed improvement."
        )
        pid = sp.paper.id
        programmatic = ProgrammaticExtractor(ExtractionConfig())
        draft = programmatic.extract(sp)
        programmatic_study_design = draft.study_design

        mock_response = MagicMock()
        mock_response.parsed = PaperExtraction(
            paper_id=pid,
            key_findings=[
                Finding(
                    claim="LLM finding", evidence_strength=EvidenceStrength.STRONG, paper_id=pid
                )
            ],
            methods_summary="LLM methods.",
            limitations="LLM limitations.",
            study_design=StudyDesign.META_ANALYSIS,  # LLM says meta-analysis
            quality_score=0.50,
        )
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(return_value=mock_response)
        hybrid = HybridExtractor(programmatic, mock_llm)
        result = await hybrid.extract(sp)
        # Must match programmatic, not LLM's META_ANALYSIS
        assert result.study_design == programmatic_study_design

    async def test_extract_batch_returns_results_and_failures(self):
        """extract_batch processes a list of papers and returns extractions."""
        sp1 = _make_screened_paper(title="Paper 1", full_text=None)
        sp2 = _make_screened_paper(title="Paper 2", full_text=None)
        mock_llm = AsyncMock()
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        extractions, failures = await hybrid.extract_batch([sp1, sp2])
        assert len(extractions) == 2
        assert len(failures) == 0
        assert extractions[0].paper_id == sp1.paper.id
        assert extractions[1].paper_id == sp2.paper.id

    async def test_json_in_markdown_fence_extracted(self):
        """JSON wrapped in markdown fences is extracted correctly."""
        sp = _make_screened_paper(full_text="Study demonstrates 80% improvement in performance.")
        pid = sp.paper.id
        refined = PaperExtraction(
            paper_id=pid,
            key_findings=[
                Finding(
                    claim="80% improvement",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id=pid,
                )
            ],
            methods_summary="Comparative study.",
            limitations="Small sample.",
            study_design=StudyDesign.COHORT,
            quality_score=0.6,
        )
        # Wrap JSON in markdown fence — common LLM output pattern
        fenced_json = f"```json\n{refined.model_dump_json()}\n```"
        mock_llm = MagicMock()
        mock_llm.generate_structured = None
        mock_llm.generate = AsyncMock(return_value=MagicMock(content=fenced_json))
        hybrid = HybridExtractor(ProgrammaticExtractor(ExtractionConfig()), mock_llm)
        result = await hybrid.extract(sp)
        assert result.key_findings[0].claim == "80% improvement"
