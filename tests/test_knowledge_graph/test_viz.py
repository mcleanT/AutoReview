"""Tests for graph visualization and export."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def _make_viz_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("a", canonical_name="Gene A", entity_type="gene", paper_count=3)
    graph.add_node("b", canonical_name="Process B", entity_type="biological_process", paper_count=2)
    graph.add_edge("a", "b", key="e1", predicate="induces", confidence_mean=0.8, evidence_count=3)
    return graph


class TestGraphMLExport:
    def test_export_creates_file(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        graph = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(graph, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_exported_graphml_readable(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        graph = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(graph, out)
        graph2 = nx.read_graphml(out)
        assert graph2.number_of_nodes() == 2


class TestPlotSubgraph:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_subgraph

        graph = _make_viz_graph()
        out = tmp_path / "subgraph.png"
        plot_subgraph(graph, output_path=out)
        assert out.exists()


class TestConfidenceDistribution:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_confidence_distribution

        graph = _make_viz_graph()
        out = tmp_path / "confidence.png"
        plot_confidence_distribution(graph, output_path=out)
        assert out.exists()


class TestControversyMap:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_controversy_map

        graph = _make_viz_graph()
        graph.edges["a", "b", "e1"]["controversy_score"] = 0.8
        out = tmp_path / "controversy.png"
        plot_controversy_map(graph, output_path=out, threshold=0.5)
        assert out.exists()
