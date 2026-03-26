"""Tests for NetworkX graph construction and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestBuildGraph:
    def test_entities_become_nodes(self):
        from autoreview.knowledge_graph.graph import build_nx_graph

        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(
                entity_id="ent1",
                canonical_name="Wnt",
                entity_type="pathway",
                ontology_id="GO:0016055",
                ontology_source="GO",
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
            "ent2": KGEntity(
                entity_id="ent2",
                canonical_name="mesoderm",
                entity_type="biological_process",
                ontology_id=None,
                ontology_source=None,
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
        }
        edges = [
            KGEdge(
                edge_id="e1",
                subject_id="ent1",
                object_id="ent2",
                predicate="is_required_for",
                direction="positive",
                assertion_type="mechanistic_causal",
                confidence=BetaPosterior(),
                evidence_links=[],
                source_assertions=["a1"],
                publication_date=None,
            ),
        ]
        G = build_nx_graph(entities, edges)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert G.nodes["ent1"]["canonical_name"] == "Wnt"
        assert G.nodes["ent1"]["entity_type"] == "pathway"

    def test_edge_attributes_stored(self):
        from autoreview.knowledge_graph.graph import build_nx_graph

        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(
                entity_id="ent1",
                canonical_name="A",
                entity_type="gene",
                ontology_id=None,
                ontology_source=None,
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
            "ent2": KGEntity(
                entity_id="ent2",
                canonical_name="B",
                entity_type="gene",
                ontology_id=None,
                ontology_source=None,
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
        }
        edges = [
            KGEdge(
                edge_id="e1",
                subject_id="ent1",
                object_id="ent2",
                predicate="induces",
                direction="positive",
                assertion_type="mechanistic_causal",
                confidence=BetaPosterior(alpha=3.0, beta_param=1.0),
                evidence_links=[],
                source_assertions=["a1"],
                publication_date="2023-01-01",
            ),
        ]
        G = build_nx_graph(entities, edges)
        edge_data = G.edges["ent1", "ent2", "e1"]
        assert edge_data["predicate"] == "induces"
        assert edge_data["confidence_mean"] == pytest.approx(0.75)

    def test_self_loop_allowed(self):
        """Self-loops (autoregulation) are biologically valid."""
        from autoreview.knowledge_graph.graph import build_nx_graph

        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(
                entity_id="ent1",
                canonical_name="SIRT1",
                entity_type="gene",
                ontology_id=None,
                ontology_source=None,
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
        }
        edges = [
            KGEdge(
                edge_id="e1",
                subject_id="ent1",
                object_id="ent1",
                predicate="regulates",
                direction="positive",
                assertion_type="mechanistic_causal",
                confidence=BetaPosterior(),
                evidence_links=[],
                source_assertions=["a1"],
                publication_date=None,
            ),
        ]
        G = build_nx_graph(entities, edges)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 1
        assert G.has_edge("ent1", "ent1")


class TestSerializationRoundTrip:
    def test_pickle_round_trip(self, tmp_path: Path):
        from autoreview.knowledge_graph.graph import build_nx_graph, load_graph, save_graph

        from autoreview.knowledge_graph.models import KGEntity

        entities = {
            "ent1": KGEntity(
                entity_id="ent1",
                canonical_name="A",
                entity_type="gene",
                ontology_id=None,
                ontology_source=None,
                aliases=[],
                paper_count=1,
                source_paper_ids=["p1"],
            ),
        }
        G = build_nx_graph(entities, [])
        save_graph(G, tmp_path / "test_graph")
        G2 = load_graph(tmp_path / "test_graph.pkl")
        assert G2.number_of_nodes() == 1
        assert G2.nodes["ent1"]["canonical_name"] == "A"
