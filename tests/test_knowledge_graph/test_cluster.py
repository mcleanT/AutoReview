"""Tests for autoreview.knowledge_graph.cluster."""

from __future__ import annotations

import hashlib

import networkx as nx
import pytest

from autoreview.knowledge_graph.cluster import (
    PREDICATE_CLASS_TABLE,
    Finding,
    FindingContradiction,
    TopicCluster,
    _finding_coupling,
    _get_edge_data_by_id,
    build_topic_clusters,
    detect_finding_contradictions,
    form_findings,
    get_organism_class,
    get_predicate_class,
)
from autoreview.knowledge_graph.models import BetaPosterior

# ---------------------------------------------------------------------------
# Shared graph fixture
# ---------------------------------------------------------------------------


def _make_cluster_graph() -> nx.MultiDiGraph:
    """Graph with 3 activating edges on same triple + 1 inhibiting edge + 1 singleton."""
    graph = nx.MultiDiGraph()
    graph.add_node("e1", canonical_name="BMP4")
    graph.add_node("e2", canonical_name="mesoderm")
    graph.add_node("e3", canonical_name="SMAD1")

    # 3 activating edges: BMP4 → mesoderm (same topic cluster)
    graph.add_edge(
        "e1",
        "e2",
        edge_id="a1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.8,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 induces mesoderm",
        _kg_edge=None,
    )
    graph.add_edge(
        "e1",
        "e2",
        edge_id="a2",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.7,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 is sufficient for mesoderm",
        _kg_edge=None,
    )
    graph.add_edge(
        "e1",
        "e2",
        edge_id="a3",
        predicate="induces",
        direction="negative",
        confidence_mean=0.6,
        organism="Homo sapiens",
        in_vitro=False,
        model_system="embryo",
        conditions={},
        section_source="interpretive",
        natural_language="BMP4 does not induce mesoderm in vivo",
        _kg_edge=None,
    )

    # 1 inhibiting edge: BMP4 → mesoderm (different predicate class → different cluster)
    graph.add_edge(
        "e1",
        "e2",
        edge_id="b1",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.5,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 inhibits mesoderm",
        _kg_edge=None,
    )

    # 1 singleton edge: BMP4 → SMAD1 (only 1 edge → should be excluded)
    graph.add_edge(
        "e1",
        "e3",
        edge_id="s1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.9,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 induces SMAD1",
        _kg_edge=None,
    )
    return graph


# ---------------------------------------------------------------------------
# 1. TestPredicateClassTable
# ---------------------------------------------------------------------------


class TestPredicateClassTable:
    def test_all_six_classes_present(self):
        expected = {
            "activating",
            "inhibiting",
            "regulatory",
            "associative",
            "structural",
            "transformative",
        }
        assert set(PREDICATE_CLASS_TABLE.keys()) == expected

    def test_activating_members(self):
        assert set(PREDICATE_CLASS_TABLE["activating"]) == {
            "induces",
            "is_sufficient_for",
            "phosphorylates",
            "stabilizes",
        }

    def test_inhibiting_members(self):
        assert set(PREDICATE_CLASS_TABLE["inhibiting"]) == {"inhibits", "degrades"}

    def test_regulatory_members(self):
        assert set(PREDICATE_CLASS_TABLE["regulatory"]) == {
            "regulates",
            "is_required_for",
            "modifies",
            "maintains",
        }

    def test_associative_members(self):
        assert set(PREDICATE_CLASS_TABLE["associative"]) == {
            "correlates_with",
            "interacts_with",
            "colocalizes_with",
        }

    def test_structural_members(self):
        assert set(PREDICATE_CLASS_TABLE["structural"]) == {
            "is_component_of",
            "is_located_in",
            "is_marker_of",
        }

    def test_transformative_members(self):
        assert set(PREDICATE_CLASS_TABLE["transformative"]) == {
            "differentiates_into",
            "converts",
            "transports",
        }


# ---------------------------------------------------------------------------
# 2. TestGetPredicateClass
# ---------------------------------------------------------------------------


class TestGetPredicateClass:
    def test_known_activating(self):
        assert get_predicate_class("induces") == "activating"

    def test_known_inhibiting(self):
        assert get_predicate_class("inhibits") == "inhibiting"

    def test_known_regulatory(self):
        assert get_predicate_class("regulates") == "regulatory"

    def test_known_associative(self):
        assert get_predicate_class("correlates_with") == "associative"

    def test_known_structural(self):
        assert get_predicate_class("is_component_of") == "structural"

    def test_known_transformative(self):
        assert get_predicate_class("differentiates_into") == "transformative"

    def test_unknown_returns_other(self):
        assert get_predicate_class("some_unknown_predicate") == "other"

    def test_empty_string_returns_other(self):
        assert get_predicate_class("") == "other"

    def test_all_table_predicates_resolve(self):
        for cls, preds in PREDICATE_CLASS_TABLE.items():
            for pred in preds:
                assert get_predicate_class(pred) == cls, f"Predicate '{pred}' should map to '{cls}'"


# ---------------------------------------------------------------------------
# 3. TestTopicCluster
# ---------------------------------------------------------------------------


class TestTopicCluster:
    def test_dataclass_fields(self):
        tc = TopicCluster(
            cluster_id="abc123",
            subject_id="e1",
            object_id="e2",
            predicate_class="activating",
            member_predicates={"induces"},
            edge_ids=["a1", "a2"],
        )
        assert tc.cluster_id == "abc123"
        assert tc.subject_id == "e1"
        assert tc.object_id == "e2"
        assert tc.predicate_class == "activating"
        assert tc.member_predicates == {"induces"}
        assert tc.edge_ids == ["a1", "a2"]
        assert tc.finding_ids == []  # default

    def test_finding_ids_default_empty(self):
        tc = TopicCluster(
            cluster_id="x",
            subject_id="a",
            object_id="b",
            predicate_class="inhibiting",
            member_predicates=set(),
            edge_ids=[],
        )
        assert isinstance(tc.finding_ids, list)
        assert len(tc.finding_ids) == 0


# ---------------------------------------------------------------------------
# 4. TestBuildTopicClusters
# ---------------------------------------------------------------------------


class TestBuildTopicClusters:
    def test_groups_by_predicate_class(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        # e1→e2 activating (a1,a2,a3) → ≥2 edges → cluster present
        # e1→e2 inhibiting (b1 only) → singleton → excluded
        pred_classes = {c.predicate_class for c in clusters}
        assert "activating" in pred_classes
        # inhibiting cluster has only 1 edge so it is excluded (singleton)
        assert "inhibiting" not in pred_classes

    def test_excludes_singletons(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        # e1→e3 has only 1 edge (s1), must be excluded
        all_edge_ids = {eid for c in clusters for eid in c.edge_ids}
        assert "s1" not in all_edge_ids

    def test_activating_cluster_has_three_edges(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        assert len(activating.edge_ids) == 3
        assert set(activating.edge_ids) == {"a1", "a2", "a3"}

    def test_inhibiting_cluster_has_one_edge_excluded(self):
        # b1 is the only inhibiting edge → singleton → excluded
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        # inhibiting cluster from e1→e2: only b1 → singleton → must NOT appear
        inhibiting_clusters = [c for c in clusters if c.predicate_class == "inhibiting"]
        assert len(inhibiting_clusters) == 0

    def test_member_predicates_tracked(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        assert "induces" in activating.member_predicates
        assert "is_sufficient_for" in activating.member_predicates

    def test_deterministic_cluster_id(self):
        expected = hashlib.sha1(b"e1|activating|e2").hexdigest()[:16]
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        assert activating.cluster_id == expected

    def test_sorted_by_cluster_id(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        ids = [c.cluster_id for c in clusters]
        assert ids == sorted(ids)

    def test_empty_graph(self):
        graph = nx.MultiDiGraph()
        clusters = build_topic_clusters(graph)
        assert clusters == []

    def test_subject_object_set_correctly(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        for c in clusters:
            assert c.subject_id == "e1"
            assert c.object_id == "e2"


# ---------------------------------------------------------------------------
# 5. TestGetOrganismClass
# ---------------------------------------------------------------------------


class TestGetOrganismClass:
    def test_mus_musculus(self):
        assert get_organism_class("Mus musculus") == "mouse"

    def test_homo_sapiens(self):
        assert get_organism_class("Homo sapiens") == "human"

    def test_rattus_norvegicus(self):
        assert get_organism_class("Rattus norvegicus") == "rat"

    def test_macaca_fascicularis(self):
        assert get_organism_class("Macaca fascicularis") == "macaque"

    def test_danio_rerio(self):
        assert get_organism_class("Danio rerio") == "zebrafish"

    def test_xenopus_laevis(self):
        assert get_organism_class("Xenopus laevis") == "xenopus"

    def test_xenopus_tropicalis(self):
        assert get_organism_class("Xenopus tropicalis") == "xenopus"

    def test_drosophila(self):
        assert get_organism_class("Drosophila melanogaster") == "drosophila"

    def test_c_elegans(self):
        assert get_organism_class("Caenorhabditis elegans") == "celegans"

    def test_unknown_uses_lowercase_genus(self):
        assert get_organism_class("Gallus gallus") == "gallus"

    def test_single_word_organism(self):
        assert get_organism_class("Unknown") == "unknown"

    def test_none_returns_unknown(self):
        assert get_organism_class(None) == "unknown"


# ---------------------------------------------------------------------------
# 6. TestFormFindings
# ---------------------------------------------------------------------------


class TestFormFindings:
    def test_splits_by_direction(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        # activating cluster: 2 positive (mouse/invitro) + 1 negative (human/invivo)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        cluster_findings = [f for f in findings if f.topic_cluster_id == activating.cluster_id]
        directions = {f.direction for f in cluster_findings}
        assert "positive" in directions
        assert "negative" in directions

    def test_splits_by_organism_class(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        cluster_findings = [f for f in findings if f.topic_cluster_id == activating.cluster_id]
        organism_classes = {f.organism_class for f in cluster_findings}
        assert "mouse" in organism_classes
        assert "human" in organism_classes

    def test_anchor_prefers_interpretive(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        negative_finding = next(
            f
            for f in findings
            if f.topic_cluster_id == activating.cluster_id and f.direction == "negative"
        )
        # a3 is interpretive with direction=negative → should be the anchor
        assert negative_finding.anchor_edge_id == "a3"
        assert "does not induce" in negative_finding.anchor_text

    def test_confidence_aggregates(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        positive_finding = next(
            f
            for f in findings
            if f.topic_cluster_id == activating.cluster_id and f.direction == "positive"
        )
        # 2 positive edges: conf 0.8, 0.7 → alpha=1+0.8+0.7=2.5, beta=1+0.2+0.3=1.5
        assert positive_finding.confidence.alpha == pytest.approx(2.5)
        assert positive_finding.confidence.beta_param == pytest.approx(1.5)

    def test_deterministic_finding_ids(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings1 = form_findings(clusters, graph)

        # Re-build clusters to get fresh objects
        clusters2 = build_topic_clusters(_make_cluster_graph())
        findings2 = form_findings(clusters2, _make_cluster_graph())

        ids1 = sorted(f.finding_id for f in findings1)
        ids2 = sorted(f.finding_id for f in findings2)
        assert ids1 == ids2

    def test_populates_cluster_finding_ids(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        finding_ids = {f.finding_id for f in findings}
        for cluster in clusters:
            for fid in cluster.finding_ids:
                assert fid in finding_ids

    def test_evidence_count_correct(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        positive_finding = next(
            f
            for f in findings
            if f.topic_cluster_id == activating.cluster_id and f.direction == "positive"
        )
        # a1 and a2 are mouse/invitro/positive
        assert positive_finding.evidence_count == 2

    def test_sorted_by_finding_id(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        ids = [f.finding_id for f in findings]
        assert ids == sorted(ids)

    def test_in_vitro_flag_set(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        positive_finding = next(
            f
            for f in findings
            if f.topic_cluster_id == activating.cluster_id and f.direction == "positive"
        )
        assert positive_finding.in_vitro is True


# ---------------------------------------------------------------------------
# 7. TestFindingContradiction
# ---------------------------------------------------------------------------


class TestFindingContradiction:
    def test_dataclass_fields(self):
        fc = FindingContradiction(
            finding_a_id="aaa",
            finding_b_id="bbb",
            topic_cluster_id="ccc",
            contradiction_type="directional",
            severity=0.7,
            condition_coupling=0.7,
            anchor_text_a="text a",
            anchor_text_b="text b",
            paper_ids_a={"p1"},
            paper_ids_b={"p2"},
        )
        assert fc.finding_a_id == "aaa"
        assert fc.finding_b_id == "bbb"
        assert fc.topic_cluster_id == "ccc"
        assert fc.contradiction_type == "directional"
        assert fc.severity == 0.7
        assert fc.condition_coupling == 0.7
        assert fc.anchor_text_a == "text a"
        assert fc.anchor_text_b == "text b"
        assert fc.paper_ids_a == {"p1"}
        assert fc.paper_ids_b == {"p2"}


# ---------------------------------------------------------------------------
# 8. TestDetectFindingContradictions
# ---------------------------------------------------------------------------


class TestDetectFindingContradictions:
    def _get_activating_findings(self, graph=None):
        if graph is None:
            graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        activating = next(c for c in clusters if c.predicate_class == "activating")
        cluster_findings = [f for f in findings if f.topic_cluster_id == activating.cluster_id]
        return cluster_findings, clusters, findings

    def test_directional_contradiction_detected(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters)
        # positive (mouse/invitro) vs negative (human/invivo) — opposite direction
        directional = [
            c for c in contradictions if c.contradiction_type in ("directional", "boundary")
        ]
        assert len(directional) >= 1

    def test_boundary_type_when_coupling_low(self):
        """Low coupling (different organism + different vitro) → boundary contradiction."""
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        # mouse/invitro positive vs human/invivo negative → coupling = 0.7*0.3 + 0.3*0.6 = 0.39
        contradictions = detect_finding_contradictions(findings, clusters, boundary_threshold=0.6)
        boundaries = [c for c in contradictions if c.contradiction_type == "boundary"]
        assert len(boundaries) >= 1

    def test_directional_type_when_coupling_high(self):
        """Same organism, same vitro, opposite direction → directional."""
        graph = nx.MultiDiGraph()
        graph.add_node("s", canonical_name="A")
        graph.add_node("o", canonical_name="B")
        graph.add_edge(
            "s",
            "o",
            edge_id="x1",
            predicate="induces",
            direction="positive",
            confidence_mean=0.8,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="primary_empirical",
            natural_language="A induces B",
            _kg_edge=None,
        )
        graph.add_edge(
            "s",
            "o",
            edge_id="x2",
            predicate="is_sufficient_for",
            direction="negative",
            confidence_mean=0.7,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="primary_empirical",
            natural_language="A does not cause B",
            _kg_edge=None,
        )
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, boundary_threshold=0.6)
        # Same organism + same vitro → coupling = 0.7*1.0 + 0.3*1.0 = 1.0 → directional
        directional = [c for c in contradictions if c.contradiction_type == "directional"]
        assert len(directional) == 1
        assert directional[0].severity == pytest.approx(1.0)

    def test_no_contradiction_same_direction(self):
        """Two findings with same direction and no graph → no contradictions."""
        graph = nx.MultiDiGraph()
        graph.add_node("s")
        graph.add_node("o")
        graph.add_edge(
            "s",
            "o",
            edge_id="p1",
            predicate="induces",
            direction="positive",
            confidence_mean=0.8,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="primary_empirical",
            natural_language="A induces B",
            _kg_edge=None,
        )
        graph.add_edge(
            "s",
            "o",
            edge_id="p2",
            predicate="is_sufficient_for",
            direction="positive",
            confidence_mean=0.7,
            organism="Homo sapiens",
            in_vitro=False,
            model_system="embryo",
            conditions={},
            section_source="primary_empirical",
            natural_language="A sufficient for B",
            _kg_edge=None,
        )
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=None)
        assert len(contradictions) == 0

    def test_deterministic_ordering(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters)
        for c in contradictions:
            assert c.finding_a_id <= c.finding_b_id

    def test_sorted_by_finding_ids(self):
        graph = _make_cluster_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters)
        pairs = [(c.finding_a_id, c.finding_b_id) for c in contradictions]
        assert pairs == sorted(pairs)


# ---------------------------------------------------------------------------
# 9. TestInterpretiveConflictWithGraph
# ---------------------------------------------------------------------------


class TestInterpretiveConflictWithGraph:
    def _make_interpretive_graph(self, same_paper: bool = False) -> nx.MultiDiGraph:
        """Two interpretive edges, same direction but different condition_groups.

        Using different organisms ensures the two edges form separate Findings
        (condition_group = organism_class|vitro_str).  Only separate Findings can
        be compared by detect_finding_contradictions.
        """
        graph = nx.MultiDiGraph()
        graph.add_node("s")
        graph.add_node("o")

        class FakeLink:
            def __init__(self, paper_id):
                self.paper_id = paper_id

        class FakeEdge:
            def __init__(self, paper_id):
                self.evidence_links = [FakeLink(paper_id)]

        paper_b = "paper1" if same_paper else "paper2"

        # mouse/invitro → condition_group "mouse|invitro"
        graph.add_edge(
            "s",
            "o",
            edge_id="i1",
            predicate="induces",
            direction="positive",
            confidence_mean=0.8,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="interpretive",
            natural_language="Interpretation A",
            _kg_edge=FakeEdge("paper1"),
        )
        # human/invivo → condition_group "human|invivo" (separate Finding)
        graph.add_edge(
            "s",
            "o",
            edge_id="i2",
            predicate="is_sufficient_for",
            direction="positive",
            confidence_mean=0.7,
            organism="Homo sapiens",
            in_vitro=False,
            model_system="embryo",
            conditions={},
            section_source="interpretive",
            natural_language="Interpretation B",
            _kg_edge=FakeEdge(paper_b),
        )
        return graph

    def test_interpretive_detected_with_graph(self):
        graph = self._make_interpretive_graph(same_paper=False)
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        interpretive = [c for c in contradictions if c.contradiction_type == "interpretive"]
        assert len(interpretive) == 1
        assert interpretive[0].severity == pytest.approx(0.5)

    def test_interpretive_not_detected_same_paper(self):
        """Same paper → shared paper_ids → no interpretive contradiction."""
        graph = self._make_interpretive_graph(same_paper=True)
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        interpretive = [c for c in contradictions if c.contradiction_type == "interpretive"]
        assert len(interpretive) == 0

    def test_non_interpretive_no_conflict(self):
        """Same direction, primary_empirical anchors → no interpretive contradiction."""
        graph = nx.MultiDiGraph()
        graph.add_node("s")
        graph.add_node("o")
        graph.add_edge(
            "s",
            "o",
            edge_id="e1",
            predicate="induces",
            direction="positive",
            confidence_mean=0.8,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="primary_empirical",
            natural_language="A causes B",
            _kg_edge=None,
        )
        graph.add_edge(
            "s",
            "o",
            edge_id="e2",
            predicate="is_sufficient_for",
            direction="positive",
            confidence_mean=0.7,
            organism="Mus musculus",
            in_vitro=True,
            model_system="mESC",
            conditions={},
            section_source="primary_empirical",
            natural_language="A sufficient for B",
            _kg_edge=None,
        )
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        interpretive = [c for c in contradictions if c.contradiction_type == "interpretive"]
        assert len(interpretive) == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGetEdgeDataById:
    def test_indexes_all_edges(self):
        graph = _make_cluster_graph()
        index = _get_edge_data_by_id(graph)
        assert "a1" in index
        assert "a2" in index
        assert "a3" in index
        assert "b1" in index
        assert "s1" in index

    def test_returns_correct_data(self):
        graph = _make_cluster_graph()
        index = _get_edge_data_by_id(graph)
        assert index["a1"]["predicate"] == "induces"
        assert index["a3"]["section_source"] == "interpretive"


class TestFindingCoupling:
    def test_same_organism_same_vitro(self):
        fa = Finding(
            finding_id="x",
            topic_cluster_id="c",
            direction="positive",
            condition_group="mouse|invitro",
            member_edge_ids=[],
            anchor_edge_id="",
            anchor_text="",
            confidence=BetaPosterior(alpha=1.0, beta_param=1.0),
            paper_ids=set(),
            evidence_count=1,
            organism_class="mouse",
            in_vitro=True,
        )
        fb = Finding(
            finding_id="y",
            topic_cluster_id="c",
            direction="negative",
            condition_group="mouse|invitro",
            member_edge_ids=[],
            anchor_edge_id="",
            anchor_text="",
            confidence=BetaPosterior(alpha=1.0, beta_param=1.0),
            paper_ids=set(),
            evidence_count=1,
            organism_class="mouse",
            in_vitro=True,
        )
        # 0.7*1.0 + 0.3*1.0 = 1.0
        assert _finding_coupling(fa, fb) == pytest.approx(1.0)

    def test_different_organism_different_vitro(self):
        fa = Finding(
            finding_id="x",
            topic_cluster_id="c",
            direction="positive",
            condition_group="mouse|invitro",
            member_edge_ids=[],
            anchor_edge_id="",
            anchor_text="",
            confidence=BetaPosterior(alpha=1.0, beta_param=1.0),
            paper_ids=set(),
            evidence_count=1,
            organism_class="mouse",
            in_vitro=True,
        )
        fb = Finding(
            finding_id="y",
            topic_cluster_id="c",
            direction="negative",
            condition_group="human|invivo",
            member_edge_ids=[],
            anchor_edge_id="",
            anchor_text="",
            confidence=BetaPosterior(alpha=1.0, beta_param=1.0),
            paper_ids=set(),
            evidence_count=1,
            organism_class="human",
            in_vitro=False,
        )
        # 0.7*0.3 + 0.3*0.6 = 0.21 + 0.18 = 0.39
        assert _finding_coupling(fa, fb) == pytest.approx(0.39)
