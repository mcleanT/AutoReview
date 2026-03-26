"""Tests for graph visualization and export."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def _make_viz_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("a", canonical_name="Gene A", entity_type="gene", paper_count=3)
    G.add_node("b", canonical_name="Process B", entity_type="biological_process", paper_count=2)
    G.add_edge("a", "b", key="e1", predicate="induces", confidence_mean=0.8, evidence_count=3)
    return G


class TestGraphMLExport:
    def test_export_creates_file(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        G = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(G, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_exported_graphml_readable(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        G = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(G, out)
        G2 = nx.read_graphml(out)
        assert G2.number_of_nodes() == 2


class TestPlotSubgraph:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_subgraph

        G = _make_viz_graph()
        out = tmp_path / "subgraph.png"
        plot_subgraph(G, output_path=out)
        assert out.exists()


class TestConfidenceDistribution:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_confidence_distribution

        G = _make_viz_graph()
        out = tmp_path / "confidence.png"
        plot_confidence_distribution(G, output_path=out)
        assert out.exists()


class TestControversyMap:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_controversy_map

        G = _make_viz_graph()
        G.edges["a", "b", "e1"]["controversy_score"] = 0.8
        out = tmp_path / "controversy.png"
        plot_controversy_map(G, output_path=out, threshold=0.5)
        assert out.exists()
