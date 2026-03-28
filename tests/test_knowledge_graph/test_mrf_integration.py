from __future__ import annotations

from pathlib import Path


class TestMRFIntegration:
    def test_build_graph_with_mrf(self, sample_extraction_dir: Path):
        """Full pipeline with MRF scoring should complete without error."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_extraction_dir, use_mrf=True)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_build_graph_without_mrf_unchanged(self, sample_extraction_dir: Path):
        """Default pipeline (no MRF) should still work exactly as before."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_extraction_dir)
        assert graph.number_of_nodes() > 0
        # No mrf_confidence attribute when use_mrf=False
        for u, v, key, data in graph.edges(data=True, keys=True):
            assert "mrf_confidence" not in data

    def test_mrf_config_passthrough(self, sample_extraction_dir: Path):
        """Custom MRF config should be used."""
        from autoreview.knowledge_graph import MRFConfig, build_graph

        config = MRFConfig(evidence_weight=20.0)
        graph = build_graph(sample_extraction_dir, use_mrf=True, mrf_config=config)
        assert graph.number_of_nodes() > 0
