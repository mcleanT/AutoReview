"""Tests for contextual enrichment integration with section writer."""

from __future__ import annotations

import pytest

from autoreview.analysis.evidence_map import EvidenceMap, Theme
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.prompts.writing import build_section_writing_prompt
from autoreview.llm.provider import LLMResponse
from autoreview.models.enrichment import (
    ContextualExtraction,
    EnrichmentQuery,
    SectionEnrichment,
)
from autoreview.writing.section_writer import (
    SectionWriter,
    _format_contextual_enrichment,
)


def _make_enrichment() -> SectionEnrichment:
    return SectionEnrichment(
        section_id="1",
        section_title="Introduction",
        queries_generated=[
            EnrichmentQuery(
                query="background neuroinflammation",
                rationale="Foundational context",
                enrichment_type="background",
            ),
        ],
        papers_found=10,
        papers_screened=5,
        contextual_extractions=[
            ContextualExtraction(
                paper_id="enrich_1",
                paper_title="Neuroinflammation Basics",
                background_summary="Microglia are the primary immune cells of the CNS.",
                key_concepts=["microglia", "astrocytes", "cytokines"],
                methodological_context="Standard immunohistochemistry protocols",
                cross_field_connections=["immunology", "gut-brain axis"],
                relevance_to_section="Provides mechanistic background for the introduction.",
            ),
            ContextualExtraction(
                paper_id="enrich_2",
                paper_title="Historical Perspective on Neurodegeneration",
                background_summary="The concept of neurodegeneration evolved over decades.",
                key_concepts=["protein misfolding", "amyloid hypothesis"],
                methodological_context="",
                cross_field_connections=[],
                relevance_to_section="Historical context for the field.",
            ),
        ],
    )


class TestFormatContextualEnrichment:
    def test_formats_extractions(self):
        enrichment = _make_enrichment()
        result = _format_contextual_enrichment(enrichment)
        assert "Neuroinflammation Basics" in result
        assert "microglia" in result
        assert "immunology" in result
        assert "Provides mechanistic background" in result

    def test_includes_all_extractions(self):
        enrichment = _make_enrichment()
        result = _format_contextual_enrichment(enrichment)
        assert "Neuroinflammation Basics" in result
        assert "Historical Perspective on Neurodegeneration" in result

    def test_empty_enrichment_returns_empty(self):
        enrichment = SectionEnrichment(
            section_id="1",
            section_title="Introduction",
            contextual_extractions=[],
        )
        result = _format_contextual_enrichment(enrichment)
        assert result == ""

    def test_handles_missing_optional_fields(self):
        enrichment = SectionEnrichment(
            section_id="1",
            section_title="Introduction",
            contextual_extractions=[
                ContextualExtraction(
                    paper_id="p1",
                    paper_title="Minimal Paper",
                    background_summary="Some background.",
                    key_concepts=[],
                    methodological_context="",
                    cross_field_connections=[],
                    relevance_to_section="",
                ),
            ],
        )
        result = _format_contextual_enrichment(enrichment)
        assert "Minimal Paper" in result
        assert "(none)" in result


class TestBuildSectionWritingPromptWithEnrichment:
    def test_includes_enrichment_block(self):
        prompt = build_section_writing_prompt(
            section_id="1",
            section_title="Introduction",
            section_description="Overview",
            outline_context="Full outline...",
            relevant_extractions="[@p1] findings...",
            contextual_enrichment="### Background Paper\nMicroglia are immune cells.",
        )
        assert "Contextual Background Material" in prompt
        assert "Microglia are immune cells" in prompt

    def test_no_enrichment_block_when_empty(self):
        prompt = build_section_writing_prompt(
            section_id="1",
            section_title="Introduction",
            section_description="Overview",
            outline_context="Full outline...",
            relevant_extractions="[@p1] findings...",
            contextual_enrichment="",
        )
        assert "Contextual Background Material" not in prompt


def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="1", title="Introduction", description="Overview",
                paper_ids=["p1"], estimated_word_count=500,
            ),
            OutlineSection(
                id="2", title="Results", description="Key findings",
                paper_ids=["p1", "p2"], estimated_word_count=1000,
            ),
        ],
    )


def _make_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        themes=[Theme(name="Theme A", description="First", paper_ids=["p1", "p2"])],
    )


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[Finding(claim="Claim A", evidence_strength=EvidenceStrength.STRONG, paper_id="p1")],
            methods_summary="Methods A",
            limitations="Limits A",
        ),
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[Finding(claim="Claim B", evidence_strength=EvidenceStrength.MODERATE, paper_id="p2")],
            methods_summary="Methods B",
            limitations="Limits B",
        ),
    }


class MockWriterLLM:
    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3):
        return LLMResponse(
            content="This section synthesizes findings [@p1] [@p2] with broader context.",
            input_tokens=500,
            output_tokens=300,
        )


class TestSectionWriterWithEnrichment:
    async def test_write_section_with_enrichment(self):
        writer = SectionWriter(MockWriterLLM())
        outline = _make_outline()
        section = outline.sections[0]
        enrichment = _make_enrichment()

        draft = await writer.write_section(
            section=section,
            outline=outline,
            extractions=_make_extractions(),
            evidence_map=_make_evidence_map(),
            enrichment=enrichment,
        )
        assert draft.section_id == "1"
        assert draft.text

    async def test_write_section_without_enrichment(self):
        writer = SectionWriter(MockWriterLLM())
        outline = _make_outline()
        section = outline.sections[0]

        draft = await writer.write_section(
            section=section,
            outline=outline,
            extractions=_make_extractions(),
            evidence_map=_make_evidence_map(),
            enrichment=None,
        )
        assert draft.section_id == "1"

    async def test_write_all_sections_with_enrichment(self):
        writer = SectionWriter(MockWriterLLM())
        outline = _make_outline()
        enrichment_map = {"1": _make_enrichment()}

        drafts = await writer.write_all_sections(
            outline=outline,
            extractions=_make_extractions(),
            evidence_map=_make_evidence_map(),
            contextual_enrichment=enrichment_map,
        )
        assert "1" in drafts
        assert "2" in drafts

    async def test_write_all_sections_without_enrichment(self):
        writer = SectionWriter(MockWriterLLM())
        outline = _make_outline()

        drafts = await writer.write_all_sections(
            outline=outline,
            extractions=_make_extractions(),
            evidence_map=_make_evidence_map(),
            contextual_enrichment=None,
        )
        assert len(drafts) == 2
