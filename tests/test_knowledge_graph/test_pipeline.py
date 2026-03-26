"""Integration tests for the full KG pipeline: ingest → dedup → graph → confidence."""

from __future__ import annotations

from pathlib import Path


class TestBuildGraph:
    def test_full_pipeline(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_extraction_dir)

        # Should have nodes (entities) and edges (merged assertions)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Papers 0 and 1 have same entities + synonymous predicates → should merge into 1 edge
        # Paper 2 has different entities → separate edge
        # So expect: ~4 unique entities, 2 unique edges
        assert graph.number_of_edges() == 2

        # Edges should have confidence scores
        for _u, _v, _k, data in graph.edges(keys=True, data=True):
            assert "confidence_mean" in data
            assert 0.0 <= data["confidence_mean"] <= 1.0

    def test_save_and_load_round_trip(self, sample_extraction_dir: Path, tmp_path: Path):
        from autoreview.knowledge_graph import build_graph, load_graph, save_graph

        graph = build_graph(sample_extraction_dir)
        save_graph(graph, tmp_path / "test_kg")

        assert (tmp_path / "test_kg.pkl").exists()
        assert (tmp_path / "test_kg.graphml").exists()

        graph2 = load_graph(tmp_path / "test_kg.pkl")
        assert graph2.number_of_nodes() == graph.number_of_nodes()
        assert graph2.number_of_edges() == graph.number_of_edges()
