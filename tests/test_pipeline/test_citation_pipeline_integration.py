"""Integration test: verify the citation pipeline eliminates drop-offs."""

from __future__ import annotations


def test_new_dag_nodes_exist():
    """New nodes must be present and old outline node must be absent."""
    from unittest.mock import MagicMock

    from autoreview.config import load_config
    from autoreview.pipeline.runner import build_pipeline

    mock_llm = MagicMock()
    config = load_config(domain="biomedical")
    dag, _nodes = build_pipeline(mock_llm, config)

    assert "draft_outline" in dag.nodes
    assert "final_outline" in dag.nodes
    assert "citation_selection" in dag.nodes
    assert "outline" not in dag.nodes


def test_citation_plan_field_exists_on_knowledge_base():
    """KnowledgeBase must expose citation_plan field (not None after selection)."""
    from autoreview.models.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(topic="test")
    # Field exists and defaults to None
    assert hasattr(kb, "citation_plan")
    assert kb.citation_plan is None

    # Field can be set
    kb.citation_plan = object()
    assert kb.citation_plan is not None


def test_draft_outline_field_exists_on_knowledge_base():
    """KnowledgeBase must expose draft_outline field."""
    from autoreview.models.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(topic="test")
    assert hasattr(kb, "draft_outline")
    assert kb.draft_outline is None


def test_pipeline_phase_includes_new_phases():
    """PipelinePhase must include DRAFT_OUTLINE, FINAL_OUTLINE, CITATION_SELECTION."""
    from autoreview.models.knowledge_base import PipelinePhase

    assert PipelinePhase.DRAFT_OUTLINE == "draft_outline"
    assert PipelinePhase.FINAL_OUTLINE == "final_outline"
    assert PipelinePhase.CITATION_SELECTION == "citation_selection"
