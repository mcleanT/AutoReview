"""Tests for the passage_search pipeline node."""

from __future__ import annotations

from autoreview.analysis.evidence_map import EvidenceMap, Theme
from autoreview.config import load_config
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.pipeline.nodes import PipelineNodes


class MockPassageSearchLLM:
    """Mock LLM for passage_search node tests."""

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        from autoreview.llm.prompts.passage_mining import SectionMiningResult, UndercitedClaim

        if response_model == SectionMiningResult:
            return LLMStructuredResponse(
                parsed=SectionMiningResult(
                    section_id="s1",
                    undercited_claims=[
                        UndercitedClaim(
                            text="Senescent cells accumulate",
                            evidence_weakness="Single source",
                            current_citations=["p1"],
                            suggested_queries=["cellular senescence accumulation aging"],
                            priority="high",
                        )
                    ],
                    topic_expansions=[],
                ),
                input_tokens=300,
                output_tokens=150,
            )
        # Handle screening batch calls
        name = getattr(response_model, "__name__", "")
        if "Screening" in name or "Batch" in name:
            return LLMStructuredResponse(
                parsed=response_model(decisions=[]),
                input_tokens=100,
                output_tokens=50,
            )
        # Handle extraction calls
        if response_model == PaperExtraction:
            return LLMStructuredResponse(
                parsed=PaperExtraction(
                    paper_id="new_p1",
                    key_findings=[
                        Finding(
                            claim="New finding",
                            evidence_strength=EvidenceStrength.MODERATE,
                            paper_id="new_p1",
                        )
                    ],
                    methods_summary="RCT",
                    limitations="Small n",
                ),
                input_tokens=200,
                output_tokens=100,
            )
        raise ValueError(f"Unexpected: {response_model}")

    async def generate(
        self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None
    ):
        return LLMResponse(
            content="Revised section text with [@p1] and [@new_p1].",
            input_tokens=400,
            output_tokens=100,
        )


def _make_kb_with_drafts() -> KnowledgeBase:
    kb = KnowledgeBase(
        topic="cellular senescence",
        domain="biomedical",
        output_dir="/tmp/autoreview_test",
    )
    kb.section_drafts = {
        "s1": "Senescent cells accumulate with age [@p1].",
    }
    kb.extractions = {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(
                    claim="Cells senesce with age",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id="p1",
                )
            ],
            methods_summary="In vitro",
            limitations="Cell lines only",
        )
    }
    kb.evidence_map = EvidenceMap(
        themes=[Theme(name="Senescence", description="Cell senescence", paper_ids=["p1"])],
    )
    kb.scope_document = "Review of cellular senescence across organs."
    return kb


class TestPassageSearchNode:
    async def test_passage_search_sets_phase(self):
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = _make_kb_with_drafts()
        await nodes.passage_search(kb)
        assert kb.current_phase == PipelinePhase.PASSAGE_SEARCH

    async def test_passage_search_adds_audit_entry(self):
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = _make_kb_with_drafts()
        await nodes.passage_search(kb)
        audit_names = [e.node_name for e in kb.audit_log]
        assert "passage_search" in audit_names

    async def test_passage_search_noop_on_empty_drafts(self):
        """If no section drafts exist, passage_search should return without crashing."""
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = KnowledgeBase(topic="test", domain="biomedical", output_dir="/tmp/ar_test")
        # No section_drafts set
        await nodes.passage_search(kb)
        # Should complete without error
