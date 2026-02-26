"""Tests for the corpus_expansion pipeline node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from autoreview.config import load_config
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    MethodologyRecord,
    PaperExtraction,
)
from autoreview.llm.prompts.corpus_expansion import (
    CorpusExpansionQuery,
    CorpusExpansionQueryResult,
)
from autoreview.llm.prompts.screening import ScreeningBatchResult, ScreeningDecision
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.enrichment import (
    ContextualExtraction,
    SectionEnrichment,
)
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.pipeline.nodes import PipelineNodes


# --- Mock search source ---

class MockSearchSource:
    """Fake search source that returns a fixed paper."""

    source_name = "mock_source"

    async def search(self, queries, max_results_per_source=500):
        return [
            CandidatePaper(
                title="New Research Paper on Microglia",
                authors=["New Author"],
                year=2024,
                doi="10.9999/new_paper_1",
                abstract="Original research on microglia activation.",
                source_database="mock_source",
            ),
        ]


class MockCorpusExpansionLLM:
    """Mock LLM that handles corpus expansion, screening, and extraction calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0,
    ):
        self.calls.append(response_model.__name__)

        if response_model == CorpusExpansionQueryResult:
            section_id = "1"
            for sid in ["1", "2", "3"]:
                if f"**ID:** {sid}" in prompt:
                    section_id = sid
                    break
            return LLMStructuredResponse(
                parsed=CorpusExpansionQueryResult(
                    section_id=section_id,
                    queries=[
                        CorpusExpansionQuery(
                            query="microglia activation in Alzheimer's disease experimental models",
                            source_section_id=section_id,
                            rationale="Key concept from enrichment lacks direct evidence",
                            target_concepts=["microglia", "neuroinflammation"],
                        ),
                        CorpusExpansionQuery(
                            query="gut microbiome metabolites blood-brain barrier permeability",
                            source_section_id=section_id,
                            rationale="Cross-field connection needs citable data",
                            target_concepts=["metabolites", "blood-brain barrier"],
                        ),
                        CorpusExpansionQuery(
                            query="short-chain fatty acids neuroprotection in vivo studies",
                            source_section_id=section_id,
                            rationale="Mechanistic detail from enrichment",
                            target_concepts=["SCFA", "neuroprotection"],
                        ),
                    ],
                ),
                input_tokens=600,
                output_tokens=300,
            )

        if response_model == ScreeningBatchResult:
            return LLMStructuredResponse(
                parsed=ScreeningBatchResult(
                    decisions=[
                        ScreeningDecision(
                            paper_index=0,
                            relevance_score=4,
                            rationale="Highly relevant primary research",
                        ),
                    ],
                ),
                input_tokens=200,
                output_tokens=100,
            )

        if response_model == PaperExtraction:
            return LLMStructuredResponse(
                parsed=PaperExtraction(
                    paper_id="new_paper_1",
                    key_findings=[
                        Finding(
                            claim="SCFA supplementation reduces neuroinflammation markers",
                            evidence_strength=EvidenceStrength.MODERATE,
                            paper_id="new_paper_1",
                        ),
                    ],
                    methods_summary="In vivo mouse model with SCFA supplementation",
                    limitations="Animal model only",
                    methodology_details=MethodologyRecord(
                        approach="Controlled experiment",
                    ),
                ),
                input_tokens=400,
                output_tokens=300,
            )

        raise ValueError(f"Unexpected response_model: {response_model}")


def _make_kb_with_enrichment() -> KnowledgeBase:
    """Create a KnowledgeBase with contextual enrichment data."""
    kb = KnowledgeBase(
        topic="gut microbiome in neurodegenerative diseases",
        domain="biomedical",
        output_dir="/tmp/autoreview_test",
    )
    kb.scope_document = "Review of gut-brain axis in neurodegeneration."
    kb.outline = {
        "title": "Gut Microbiome and Neurodegeneration",
        "sections": [
            {
                "id": "1",
                "title": "Introduction",
                "description": "Overview of the gut-brain axis",
                "theme_refs": [],
                "paper_ids": ["p1"],
                "estimated_word_count": 500,
                "subsections": [],
            },
            {
                "id": "2",
                "title": "Mechanisms",
                "description": "Mechanistic pathways linking gut to brain",
                "theme_refs": [],
                "paper_ids": ["p1", "p2"],
                "estimated_word_count": 1000,
                "subsections": [],
            },
        ],
    }
    kb.candidate_papers = [
        CandidatePaper(
            title="Existing Paper 1",
            authors=["Author A"],
            year=2023,
            doi="10.1234/existing1",
            abstract="Existing research.",
            source_database="pubmed",
        ),
    ]
    kb.contextual_enrichment = {
        "1": SectionEnrichment(
            section_id="1",
            section_title="Introduction",
            papers_found=5,
            papers_screened=3,
            contextual_extractions=[
                ContextualExtraction(
                    paper_id="enrich_1",
                    paper_title="Background on Neuroinflammation",
                    background_summary="Neuroinflammation involves microglia activation.",
                    key_concepts=["microglia", "cytokines", "blood-brain barrier"],
                    cross_field_connections=["immunology", "gut microbiome"],
                    relevance_to_section="Provides mechanistic context.",
                ),
            ],
        ),
        "2": SectionEnrichment(
            section_id="2",
            section_title="Mechanisms",
            papers_found=4,
            papers_screened=2,
            contextual_extractions=[
                ContextualExtraction(
                    paper_id="enrich_2",
                    paper_title="SCFA and Brain Health",
                    background_summary="Short-chain fatty acids modulate neuroimmunity.",
                    key_concepts=["SCFA", "butyrate", "neuroprotection"],
                    cross_field_connections=["metabolomics", "nutritional neuroscience"],
                    relevance_to_section="Mechanistic link between gut metabolites and brain.",
                ),
            ],
        ),
    }
    return kb


def _patch_search_sources():
    """Patch search source imports to return mock sources."""
    return patch.dict(
        "autoreview.pipeline.nodes.__builtins__",
        {},
    )


def _make_config_with_mock_db():
    """Load config and override databases so only mock_source is used."""
    config = load_config(domain="biomedical")
    # Override databases to avoid real source init
    config.databases = {"primary": [], "secondary": [], "discovery": []}
    return config


@pytest.fixture
def mock_search_aggregator():
    """Patch SearchAggregator.search to return mock papers."""
    mock_paper = CandidatePaper(
        title="New Research Paper on Microglia",
        authors=["New Author"],
        year=2024,
        doi="10.9999/new_paper_1",
        abstract="Original research on microglia activation.",
        source_database="pubmed",
    )
    with patch(
        "autoreview.search.aggregator.SearchAggregator",
        autospec=False,
    ) as mock_cls:
        instance = mock_cls.return_value
        instance.search = AsyncMock(return_value=[mock_paper])
        yield mock_cls


class TestCorpusExpansion:
    @pytest.fixture
    def config(self):
        return load_config(domain="biomedical")

    async def test_sets_phase(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        assert kb.current_phase == PipelinePhase.CORPUS_EXPANSION

    async def test_adds_audit_entry(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        audit_actions = [e.node_name for e in kb.audit_log]
        assert "corpus_expansion" in audit_actions

    async def test_skips_when_no_enrichment(self, config):
        kb = KnowledgeBase(
            topic="test", domain="biomedical", output_dir="/tmp/test",
        )
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        assert kb.current_phase == PipelinePhase.CORPUS_EXPANSION
        assert kb.audit_log[-1].action == "skipped"
        assert "No contextual enrichment" in kb.audit_log[-1].details

    async def test_skips_when_enrichment_has_no_extractions(self, config):
        kb = KnowledgeBase(
            topic="test", domain="biomedical", output_dir="/tmp/test",
        )
        kb.contextual_enrichment = {
            "1": SectionEnrichment(
                section_id="1",
                section_title="Introduction",
                papers_found=0,
                papers_screened=0,
                contextual_extractions=[],
            ),
        }
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        assert kb.current_phase == PipelinePhase.CORPUS_EXPANSION
        assert kb.audit_log[-1].action == "skipped"
        assert "no contextual extractions" in kb.audit_log[-1].details

    async def test_generates_queries_from_enrichment_insights(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        llm = MockCorpusExpansionLLM()
        nodes = PipelineNodes(llm, config)
        await nodes.corpus_expansion(kb)
        # LLM should have been called with CorpusExpansionQueryResult for each section
        query_calls = [c for c in llm.calls if c == "CorpusExpansionQueryResult"]
        assert len(query_calls) == 2  # Two sections with enrichment

    async def test_merges_papers_into_kb_extractions(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        initial_extraction_count = len(kb.extractions)
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        # New extractions should be added
        assert len(kb.extractions) > initial_extraction_count

    async def test_updates_outline_paper_ids(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)

        # Outline sections should have gained new paper IDs
        from autoreview.llm.prompts.outline import ReviewOutline
        outline = ReviewOutline.model_validate(kb.outline)
        section_1 = outline.get_section("1")
        section_2 = outline.get_section("2")
        # Original paper_ids plus new ones
        assert len(section_1.paper_ids) > 1
        assert len(section_2.paper_ids) > 2

    async def test_tracks_corpus_expansion_results(self, config, mock_search_aggregator):
        kb = _make_kb_with_enrichment()
        nodes = PipelineNodes(MockCorpusExpansionLLM(), config)
        await nodes.corpus_expansion(kb)
        assert "1" in kb.corpus_expansion_results
        assert "2" in kb.corpus_expansion_results
        result_1 = kb.corpus_expansion_results["1"]
        assert result_1.section_id == "1"
        assert result_1.queries_generated > 0
