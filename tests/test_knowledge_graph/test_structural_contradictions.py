"""Tests for autoreview.knowledge_graph.structural_contradictions.

Tests are organized into classes:
- TestDetectContradictions — main function, all core cases
- TestBoundaryConditionClassification — coupling-driven reclassification
- TestEdgeCases — empty graph, single edge, no-contradiction cases
- TestMultipleContradictions — graphs with several contradictions
"""

from __future__ import annotations

import networkx as nx

from autoreview.knowledge_graph.structural_contradictions import (
    ContradictionPair,
    ContradictionType,
    detect_contradictions,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_graph_with_edges(edges: list[dict]) -> nx.MultiDiGraph:
    """Build a test MultiDiGraph from a list of edge dicts."""
    graph = nx.MultiDiGraph()
    for e in edges:
        graph.add_edge(
            e["subject"],
            e["object"],
            predicate=e["predicate"],
            direction=e.get("direction", "positive"),
            organism=e.get("organism"),
            model_system=e.get("model_system"),
            in_vitro=e.get("in_vitro"),
            conditions=e.get("conditions", {}),
            edge_id=e.get("edge_id", f"{e['subject']}_{e['predicate']}_{e['object']}"),
        )
    return graph


# ---------------------------------------------------------------------------
# TestDetectContradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    def test_opposing_predicates_same_conditions_gives_predicate_opposition(self) -> None:
        """induces vs inhibits on same S/O, same species → PREDICATE_OPPOSITION."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "BRAF",
                    "object": "ERK",
                    "predicate": "induces",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "edge_a",
                },
                {
                    "subject": "BRAF",
                    "object": "ERK",
                    "predicate": "inhibits",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "edge_b",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 1
        pair = results[0]
        assert pair.contradiction_type == ContradictionType.PREDICATE_OPPOSITION
        assert pair.subject == "BRAF"
        assert pair.object == "ERK"
        assert pair.edge_a_id == "edge_a"
        assert pair.edge_b_id == "edge_b"

    def test_synonym_opposing_predicates_detected(self) -> None:
        """activates vs suppresses should resolve to induces vs inhibits via canonicalization."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "activates",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "suppresses",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 1
        assert results[0].contradiction_type == ContradictionType.PREDICATE_OPPOSITION

    def test_same_predicate_opposite_direction_gives_direction_conflict(self) -> None:
        """Same predicate but opposite direction → DIRECTION_CONFLICT."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "PI3K",
                    "object": "AKT",
                    "predicate": "regulates",
                    "direction": "positive",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "dir_a",
                },
                {
                    "subject": "PI3K",
                    "object": "AKT",
                    "predicate": "regulates",
                    "direction": "negative",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "dir_b",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 1
        pair = results[0]
        assert pair.contradiction_type == ContradictionType.DIRECTION_CONFLICT
        assert pair.predicate_a == "regulates"
        assert pair.predicate_b == "regulates"
        assert {pair.direction_a, pair.direction_b} == {"positive", "negative"}

    def test_opposing_predicates_different_species_gives_boundary_condition(self) -> None:
        """Opposing predicates but very different conditions → BOUNDARY_CONDITION."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "X",
                    "object": "Y",
                    "predicate": "induces",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "model_system": "Mouse ESC gastruloids",
                    "edge_id": "mouse_edge",
                },
                {
                    "subject": "X",
                    "object": "Y",
                    "predicate": "inhibits",
                    "organism": "Drosophila melanogaster",
                    "in_vitro": False,
                    "model_system": "Wing disc",
                    "edge_id": "fly_edge",
                },
            ]
        )
        # coupling: sp=0.3 (rodent vs insect), sy=0.5 (different), iv=0.6 (different) → 0.42
        results = detect_contradictions(graph, boundary_threshold=0.6)
        assert len(results) == 1
        pair = results[0]
        assert pair.contradiction_type == ContradictionType.BOUNDARY_CONDITION
        assert pair.condition_coupling < 0.6

    def test_different_objects_no_contradiction(self) -> None:
        """Edges with same subject but different objects should not be compared."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "C",
                    "predicate": "inhibits",
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert results == []

    def test_non_opposing_same_subject_object_no_contradiction(self) -> None:
        """Non-opposing predicates on same S/O → no contradiction."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "is_required_for",
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert results == []

    def test_same_predicate_same_direction_no_contradiction(self) -> None:
        """Same predicate and same direction → not a contradiction."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "regulates",
                    "direction": "positive",
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "regulates",
                    "direction": "positive",
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert results == []

    def test_empty_graph_returns_empty_list(self) -> None:
        """Empty graph has no edges → no contradictions."""
        graph = nx.MultiDiGraph()
        results = detect_contradictions(graph)
        assert results == []

    def test_single_edge_no_contradiction(self) -> None:
        """A graph with one edge cannot have any contradiction."""
        graph = _make_graph_with_edges(
            [{"subject": "A", "object": "B", "predicate": "induces", "edge_id": "only_edge"}]
        )
        results = detect_contradictions(graph)
        assert results == []


# ---------------------------------------------------------------------------
# TestBoundaryConditionClassification
# ---------------------------------------------------------------------------


class TestBoundaryConditionClassification:
    def test_high_coupling_gives_predicate_opposition(self) -> None:
        """Same species, same system → coupling high → PREDICATE_OPPOSITION."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "P",
                    "object": "Q",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "model_system": "HeLa",
                    "in_vitro": True,
                    "edge_id": "p1",
                },
                {
                    "subject": "P",
                    "object": "Q",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "model_system": "HeLa",
                    "in_vitro": True,
                    "edge_id": "p2",
                },
            ]
        )
        results = detect_contradictions(graph, boundary_threshold=0.6)
        assert len(results) == 1
        assert results[0].contradiction_type == ContradictionType.PREDICATE_OPPOSITION
        assert results[0].condition_coupling >= 0.6

    def test_low_coupling_gives_boundary_condition(self) -> None:
        """Very different organisms → low coupling → BOUNDARY_CONDITION."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "P",
                    "object": "Q",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "in_vitro": False,
                    "model_system": "Primary neurons",
                    "edge_id": "human_edge",
                },
                {
                    "subject": "P",
                    "object": "Q",
                    "predicate": "inhibits",
                    "organism": "Drosophila melanogaster",
                    "in_vitro": True,
                    "model_system": "S2 cells",
                    "edge_id": "fly_edge",
                },
            ]
        )
        results = detect_contradictions(graph, boundary_threshold=0.6)
        assert len(results) == 1
        assert results[0].contradiction_type == ContradictionType.BOUNDARY_CONDITION

    def test_boundary_threshold_parameter_respected(self) -> None:
        """Adjusting boundary_threshold changes classification."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Rattus norvegicus",
                    "in_vitro": False,
                    "edge_id": "e2",
                },
            ]
        )
        # coupling: sp=0.6 (same rodent), sy=0.5, iv=0.6 → 0.5*0.6 + 0.3*0.5 + 0.2*0.6 = 0.57
        results_low = detect_contradictions(graph, boundary_threshold=0.5)
        results_high = detect_contradictions(graph, boundary_threshold=0.7)

        # With low threshold (0.5), coupling ~0.57 > 0.5 → PREDICATE_OPPOSITION
        assert results_low[0].contradiction_type == ContradictionType.PREDICATE_OPPOSITION
        # With high threshold (0.7), coupling ~0.57 < 0.7 → BOUNDARY_CONDITION
        assert results_high[0].contradiction_type == ContradictionType.BOUNDARY_CONDITION

    def test_direction_conflict_with_low_coupling_gives_boundary(self) -> None:
        """Direction conflict with very different conditions → BOUNDARY_CONDITION."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "regulates",
                    "direction": "positive",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "model_system": "HeLa",
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "regulates",
                    "direction": "negative",
                    "organism": "Caenorhabditis elegans",
                    "in_vitro": False,
                    "model_system": "Whole animal",
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph, boundary_threshold=0.6)
        assert len(results) == 1
        assert results[0].contradiction_type == ContradictionType.BOUNDARY_CONDITION


# ---------------------------------------------------------------------------
# TestMultipleContradictions
# ---------------------------------------------------------------------------


class TestMultipleContradictions:
    def test_multiple_contradictions_in_same_graph(self) -> None:
        """Graph with two independent contradictions both detected."""
        graph = _make_graph_with_edges(
            [
                # Contradiction 1: A→B induces vs inhibits
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "ab1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "ab2",
                },
                # Contradiction 2: C→D regulates positive vs negative
                {
                    "subject": "C",
                    "object": "D",
                    "predicate": "regulates",
                    "direction": "positive",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "cd1",
                },
                {
                    "subject": "C",
                    "object": "D",
                    "predicate": "regulates",
                    "direction": "negative",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "cd2",
                },
                # Non-contradicting edge
                {
                    "subject": "E",
                    "object": "F",
                    "predicate": "is_located_in",
                    "edge_id": "ef1",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 2

        types = {c.contradiction_type for c in results}
        assert ContradictionType.PREDICATE_OPPOSITION in types
        assert ContradictionType.DIRECTION_CONFLICT in types

    def test_three_edges_same_pair_two_contradictions(self) -> None:
        """Three edges on same S/O: first two oppose, second and third also oppose."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e_ind",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e_inh1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e_inh2",
                },
            ]
        )
        results = detect_contradictions(graph)
        # (induces, inhibits1), (induces, inhibits2) → 2 contradictions
        # (inhibits1, inhibits2) → same predicate, same direction → no contradiction
        assert len(results) == 2
        for pair in results:
            assert pair.contradiction_type == ContradictionType.PREDICATE_OPPOSITION

    def test_result_is_sorted_deterministically(self) -> None:
        """Results are sorted by (edge_a_id, edge_b_id) for determinism."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "z_edge",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "a_edge",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 1
        # a_edge < z_edge alphabetically → a_edge should be edge_a_id
        assert results[0].edge_a_id == "a_edge"
        assert results[0].edge_b_id == "z_edge"

    def test_returns_contradiction_pair_instances(self) -> None:
        """All returned items are ContradictionPair instances."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": True,
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        for item in results:
            assert isinstance(item, ContradictionPair)

    def test_condition_coupling_field_in_range(self) -> None:
        """condition_coupling field is always in [0, 1]."""
        graph = _make_graph_with_edges(
            [
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "induces",
                    "organism": "Mus musculus",
                    "in_vitro": True,
                    "edge_id": "e1",
                },
                {
                    "subject": "A",
                    "object": "B",
                    "predicate": "inhibits",
                    "organism": "Homo sapiens",
                    "in_vitro": False,
                    "edge_id": "e2",
                },
            ]
        )
        results = detect_contradictions(graph)
        assert len(results) == 1
        assert 0.0 <= results[0].condition_coupling <= 1.0
