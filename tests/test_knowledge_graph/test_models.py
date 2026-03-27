"""Tests for knowledge graph Pydantic models."""

from __future__ import annotations

import pytest


class TestEntityType:
    def test_all_expected_values(self):
        from autoreview.knowledge_graph.models import EntityType

        expected = {
            "gene",
            "protein",
            "pathway",
            "biological_process",
            "cell_type",
            "chemical",
            "organism",
            "disease",
            "method",
            "other",
            "rna",
            "small_molecule",
            "phenotype",
            "cellular_compartment",
            "tissue",
        }
        assert set(EntityType) == expected


class TestEvidenceStrength:
    def test_all_expected_values(self):
        from autoreview.knowledge_graph.models import EvidenceStrength

        expected = {
            "direct_experimental",
            "indirect_experimental",
            "observational_controlled",
            "observational_uncontrolled",
            "computational_prediction",
            "expert_opinion",
            "review_citation",
        }
        assert set(EvidenceStrength) == expected


class TestBetaPosterior:
    def test_default_uniform_prior(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior()
        assert bp.alpha == 1.0
        assert bp.beta_param == 1.0
        assert bp.mean == pytest.approx(0.5)

    def test_mean_computation(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=3.0, beta_param=1.0)
        assert bp.mean == pytest.approx(0.75)

    def test_ci_95_in_serialization(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=10.0, beta_param=2.0)
        data = bp.model_dump()
        assert "mean" in data
        assert "ci_95" in data
        assert len(data["ci_95"]) == 2
        assert data["ci_95"][0] < data["ci_95"][1]

    def test_json_round_trip(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=5.0, beta_param=3.0)
        json_str = bp.model_dump_json()
        bp2 = BetaPosterior.model_validate_json(json_str)
        assert bp2.alpha == bp.alpha
        assert bp2.beta_param == bp.beta_param


class TestKGEntity:
    def test_creation(self):
        from autoreview.knowledge_graph.models import KGEntity

        entity = KGEntity(
            entity_id="abc123",
            canonical_name="Wnt signaling pathway",
            entity_type="pathway",
            ontology_id="GO:0016055",
            ontology_source="GO",
            aliases=["Wnt pathway", "canonical Wnt"],
            paper_count=5,
            source_paper_ids=["p1", "p2"],
        )
        assert entity.entity_id == "abc123"
        assert entity.entity_type == "pathway"

    def test_extra_fields_ignored(self):
        from autoreview.knowledge_graph.models import KGEntity

        entity = KGEntity(
            entity_id="abc123",
            canonical_name="test",
            entity_type="gene",
            ontology_id=None,
            ontology_source=None,
            aliases=[],
            paper_count=1,
            source_paper_ids=["p1"],
            unknown_field="should be ignored",
        )
        assert not hasattr(entity, "unknown_field")


class TestKGEdge:
    def test_creation_with_evidence(self):
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEvidenceLink

        evidence = KGEvidenceLink(
            evidence_id="e1",
            paper_id="p1",
            evidence_strength="direct_experimental",
            evidence_direction="supports",
            experiment_summary="CHIR treatment of gastruloids",
            model_system="mouse ESC gastruloids",
            sample_size="n=50",
            key_figure="Fig. 2A",
            publication_date="2023-01-15",
        )
        edge = KGEdge(
            edge_id="edge1",
            subject_id="ent1",
            object_id="ent2",
            predicate="is_required_for",
            direction="positive",
            assertion_type="mechanistic_causal",
            confidence=BetaPosterior(alpha=2.0, beta_param=1.0),
            evidence_links=[evidence],
            source_assertions=["a_001"],
            publication_date="2023-01-15",
        )
        assert edge.predicate == "is_required_for"
        assert len(edge.evidence_links) == 1
        assert edge.confidence.mean == pytest.approx(2 / 3)

    def test_null_direction_allowed(self):
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge

        edge = KGEdge(
            edge_id="edge1",
            subject_id="ent1",
            object_id="ent2",
            predicate="related_to",
            direction=None,
            assertion_type="existence",
            confidence=BetaPosterior(),
            evidence_links=[],
            source_assertions=[],
            publication_date=None,
        )
        assert edge.direction is None


class TestKGCitation:
    def test_creation(self):
        from autoreview.knowledge_graph.models import KGCitation

        cit = KGCitation(
            citation_id="c1",
            citing_paper_id="p1",
            cited_source_doi="10.1016/j.cell.2020.0001",
            cited_source_pmid=None,
            citing_sentence="Previous work showed...",
            cited_claim_paraphrase="Wnt is essential",
            relationship="supports",
            linked_assertion_ids=["a_001"],
            section="introduction",
        )
        assert cit.relationship == "supports"
