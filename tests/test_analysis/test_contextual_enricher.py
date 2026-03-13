"""Tests for contextual enrichment module."""

from __future__ import annotations

from autoreview.analysis.contextual_enricher import ContextualEnricher
from autoreview.llm.prompts.enrichment import (
    BatchContextualExtractionResult,
    ContextualExtractionResult,
    EnrichmentQueryResult,
)
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.enrichment import EnrichmentQuery
from autoreview.models.narrative import NarrativePlan, SectionNarrativeDirective
from autoreview.models.paper import CandidatePaper


class MockEnrichmentLLM:
    """Mock LLM for enrichment tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0,
    ):
        self.calls.append(response_model.__name__)

        if response_model == EnrichmentQueryResult:
            # Parse section_id from prompt
            section_id = "1"
            for sid in ["1", "2", "3"]:
                if f"**ID:** {sid}" in prompt:
                    section_id = sid
                    break
            return LLMStructuredResponse(
                parsed=EnrichmentQueryResult(
                    section_id=section_id,
                    section_title=f"Section {section_id}",
                    queries=[
                        EnrichmentQuery(
                            query="background mechanisms of neuroinflammation",
                            rationale="Provides foundational context",
                            enrichment_type="background",
                        ),
                        EnrichmentQuery(
                            query="clinical biomarkers for neurodegeneration",
                            rationale="Clinical implications",
                            enrichment_type="clinical_implication",
                        ),
                        EnrichmentQuery(
                            query="gut-brain axis signaling pathways",
                            rationale="Cross-field connection",
                            enrichment_type="cross_field_connection",
                        ),
                    ],
                ),
                input_tokens=500,
                output_tokens=200,
            )

        if response_model == ContextualExtractionResult:
            return LLMStructuredResponse(
                parsed=ContextualExtractionResult(
                    paper_id="enrich_1",
                    paper_title="Background on Neuroinflammation",
                    background_summary="Neuroinflammation involves activation of microglia.",
                    key_concepts=["microglia", "cytokines", "blood-brain barrier"],
                    methodological_context="Immunohistochemistry is standard.",
                    cross_field_connections=["immunology", "gut microbiome"],
                    relevance_to_section="Provides mechanistic context for findings.",
                ),
                input_tokens=300,
                output_tokens=150,
            )

        if response_model == BatchContextualExtractionResult:
            # Parse how many papers are in the prompt
            paper_count = prompt.count("### Paper ")
            extractions = [
                ContextualExtractionResult(
                    paper_id=f"enrich_{i+1}",
                    paper_title="Background on Neuroinflammation",
                    background_summary="Neuroinflammation involves activation of microglia.",
                    key_concepts=["microglia", "cytokines", "blood-brain barrier"],
                    methodological_context="Immunohistochemistry is standard.",
                    cross_field_connections=["immunology", "gut microbiome"],
                    relevance_to_section="Provides mechanistic context for findings.",
                )
                for i in range(paper_count)
            ]
            return LLMStructuredResponse(
                parsed=BatchContextualExtractionResult(extractions=extractions),
                input_tokens=500,
                output_tokens=300,
            )

        raise ValueError(f"Unexpected response_model: {response_model}")


def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="1", title="Introduction", description="Overview of the topic",
                paper_ids=["p1"], estimated_word_count=500,
            ),
            OutlineSection(
                id="2", title="Mechanisms", description="Mechanistic evidence",
                paper_ids=["p1", "p2"], estimated_word_count=1000,
            ),
        ],
    )


def _make_narrative_plan() -> NarrativePlan:
    return NarrativePlan(
        central_argument="Gut microbiome modulates neurodegeneration.",
        narrative_arc="From background to mechanisms to clinical implications.",
        section_directives=[
            SectionNarrativeDirective(
                section_id="1",
                narrative_role="Sets the stage",
                central_claim="Neurodegeneration is a growing concern.",
                structural_suggestion="Problem-solution",
                key_insights=["Rising prevalence", "Limited treatments"],
                transition_from_prev="(Opening section)",
                transition_to_next="Leads into mechanistic evidence",
            ),
            SectionNarrativeDirective(
                section_id="2",
                narrative_role="Core evidence",
                central_claim="Multiple pathways link gut to brain.",
                structural_suggestion="Comparative",
                key_insights=["Vagal signaling", "Immune modulation"],
                transition_from_prev="Building on the introduction",
                transition_to_next="Toward clinical implications",
            ),
        ],
    )


def _make_candidate_paper() -> CandidatePaper:
    return CandidatePaper(
        title="Neuroinflammation in Alzheimer's Disease",
        authors=["Smith J", "Doe A"],
        year=2023,
        abstract="This review covers neuroinflammatory mechanisms in AD.",
        source_database="semantic_scholar",
    )


class TestContextualEnricher:
    async def test_generate_queries_returns_per_section(self):
        enricher = ContextualEnricher(MockEnrichmentLLM())
        outline = _make_outline()
        result = await enricher.generate_queries(
            outline=outline,
            narrative_plan=_make_narrative_plan(),
            scope_document="Review of gut-brain axis in neurodegeneration.",
        )
        assert "1" in result
        assert "2" in result
        assert len(result["1"].queries) == 3
        assert result["1"].queries[0].enrichment_type == "background"

    async def test_generate_queries_without_narrative_plan(self):
        enricher = ContextualEnricher(MockEnrichmentLLM())
        outline = _make_outline()
        result = await enricher.generate_queries(
            outline=outline,
            narrative_plan=None,
            scope_document="Review scope.",
        )
        assert len(result) == 2

    async def test_extract_contextual_returns_extraction(self):
        enricher = ContextualEnricher(MockEnrichmentLLM())
        paper = _make_candidate_paper()
        extraction = await enricher.extract_contextual(
            paper=paper,
            section_title="Introduction",
            section_description="Overview of the topic",
        )
        assert extraction.paper_title == "Neuroinflammation in Alzheimer's Disease"
        assert len(extraction.key_concepts) == 3
        assert "microglia" in extraction.key_concepts
        assert len(extraction.cross_field_connections) == 2

    async def test_extract_contextual_batch(self):
        enricher = ContextualEnricher(MockEnrichmentLLM())
        papers = [_make_candidate_paper(), _make_candidate_paper()]
        results = await enricher.extract_contextual_batch(
            papers=papers,
            section_title="Mechanisms",
            section_description="Mechanistic evidence",
            max_concurrent=2,
        )
        assert len(results) == 2
        assert all(r.background_summary for r in results)

    async def test_extract_contextual_batch_empty(self):
        enricher = ContextualEnricher(MockEnrichmentLLM())
        results = await enricher.extract_contextual_batch(
            papers=[],
            section_title="Mechanisms",
            section_description="Mechanistic evidence",
        )
        assert results == []
