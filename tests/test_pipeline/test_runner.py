"""Tests for pipeline runner and node wiring."""
from __future__ import annotations

import pytest

from autoreview.config import load_config
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.pipeline.runner import build_pipeline


class TestBuildPipeline:
    """Tests for pipeline DAG construction."""

    def test_build_pipeline_returns_dag_and_nodes(self):
        config = load_config(domain="biomedical")
        dag, nodes = build_pipeline(llm=None, config=config)
        assert dag is not None
        assert nodes is not None

    def test_pipeline_has_all_nodes(self):
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        expected_nodes = [
            "query_expansion",
            "search",
            "screening",
            "full_text_retrieval",
            "extraction",
            "clustering",
            "gap_search",
            "outline",
            "narrative_planning",
            "contextual_enrichment",
            "corpus_expansion",
            "section_writing",
            "passage_search",
            "assembly",
            "final_polish",
        ]
        for name in expected_nodes:
            assert name in dag.nodes, f"Missing node: {name}"

    def test_pipeline_topology_is_valid(self):
        """Pipeline should have a valid topological order (no cycles)."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        order = dag._topological_levels()
        assert len(order) > 0
        # All nodes should appear in the sorted order
        flat = [name for level in order for name in level]
        assert len(flat) == 15

    def test_pipeline_dependencies(self):
        """Check key dependency relationships."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)

        # search depends on query_expansion
        assert "query_expansion" in dag.nodes["search"].dependencies
        # screening depends on search
        assert "search" in dag.nodes["screening"].dependencies
        # final_polish depends on assembly
        assert "assembly" in dag.nodes["final_polish"].dependencies

    def test_pipeline_different_domains(self):
        """Pipeline builds correctly for all domains."""
        for domain in ["biomedical", "cs_ai", "chemistry"]:
            config = load_config(domain=domain)
            dag, nodes = build_pipeline(llm=None, config=config)
            assert dag is not None
            assert nodes.config.domain == domain

    def test_corpus_expansion_dependencies(self):
        """corpus_expansion depends on contextual_enrichment; section_writing depends on corpus_expansion."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        assert "contextual_enrichment" in dag.nodes["corpus_expansion"].dependencies
        assert "corpus_expansion" in dag.nodes["section_writing"].dependencies

    def test_full_text_retrieval_dependencies(self):
        """full_text_retrieval depends on screening; extraction depends on full_text_retrieval."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        assert "screening" in dag.nodes["full_text_retrieval"].dependencies
        assert "full_text_retrieval" in dag.nodes["extraction"].dependencies

    def test_passage_search_dependencies(self):
        """passage_search must depend on section_writing; assembly must depend on passage_search."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        assert "section_writing" in dag.nodes["passage_search"].dependencies
        assert "passage_search" in dag.nodes["assembly"].dependencies


class TestPipelineNodesHaveDispatcher:
    def test_nodes_have_dispatcher(self):
        from autoreview.pipeline.nodes import PipelineNodes

        config = load_config(domain="biomedical")
        nodes = PipelineNodes(llm=None, config=config)
        assert hasattr(nodes, "dispatcher")
        assert nodes.dispatcher is not None


class TestSearchConfig:
    def test_remediation_config_defaults(self):
        from autoreview.config.models import SearchConfig
        config = SearchConfig()
        assert config.max_query_expansion_rounds == 2
        assert config.max_gap_search_rounds == 2
        assert config.min_coverage_threshold == 0.75
