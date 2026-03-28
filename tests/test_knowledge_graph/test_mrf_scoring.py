"""Tests for autoreview.knowledge_graph.mrf_scoring.

Tests verify:
- Transitive composition boosts sparse A→C edges
- Strong well-evidenced edges stay stable
- All edges are present in posteriors
- Empty graph returns empty result
- Contradicting edges — the weaker one is penalised
- Composition rules are not grounded when no A→C edge exists
"""

from __future__ import annotations

import networkx as nx

from autoreview.knowledge_graph.mrf_scoring import MRFConfig, MRFResult, score_graph_mrf

# ---------------------------------------------------------------------------
# Helper: canonical scored graph
# ---------------------------------------------------------------------------


def _make_scored_graph() -> nx.MultiDiGraph:
    """Build a simple three-node MultiDiGraph for testing."""
    G = nx.MultiDiGraph()
    G.add_node("e1", canonical_name="BMP4", entity_type="protein")
    G.add_node("e2", canonical_name="SMAD1", entity_type="protein")
    G.add_node("e3", canonical_name="mesoderm", entity_type="biological_process")

    # BMP4 → SMAD1 (strong)
    G.add_edge(
        "e1",
        "e2",
        predicate="induces",
        direction="positive",
        confidence_mean=0.85,
        edge_id="edge_ab",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    # SMAD1 → mesoderm (strong)
    G.add_edge(
        "e2",
        "e3",
        predicate="induces",
        direction="positive",
        confidence_mean=0.80,
        edge_id="edge_bc",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    # BMP4 → mesoderm (sparse)
    G.add_edge(
        "e1",
        "e3",
        predicate="induces",
        direction="positive",
        confidence_mean=0.35,
        edge_id="edge_ac",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    return G


# ---------------------------------------------------------------------------
# Test 1: Transitive boost
# ---------------------------------------------------------------------------


def test_transitive_boost() -> None:
    """BMP4→SMAD1 + SMAD1→mesoderm should boost BMP4→mesoderm above 0.4."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    assert isinstance(result, MRFResult)
    assert "edge_ac" in result.posteriors, "edge_ac must be in posteriors"

    ac_posterior = result.posteriors["edge_ac"]
    assert ac_posterior > 0.4, (
        f"BMP4→mesoderm posterior should be boosted above 0.4 by the "
        f"BMP4→SMAD1→mesoderm chain, got {ac_posterior:.4f}"
    )
    assert 0.0 <= ac_posterior <= 1.0


# ---------------------------------------------------------------------------
# Test 2: Strong edges remain stable
# ---------------------------------------------------------------------------


def test_strong_edges_stable() -> None:
    """Well-evidenced edges (0.85, 0.80) should stay above 0.75 after inference."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    ab_posterior = result.posteriors["edge_ab"]
    bc_posterior = result.posteriors["edge_bc"]

    assert ab_posterior > 0.75, (
        f"edge_ab (BMP4→SMAD1, conf=0.85) dropped below 0.75: {ab_posterior:.4f}"
    )
    assert bc_posterior > 0.75, (
        f"edge_bc (SMAD1→mesoderm, conf=0.80) dropped below 0.75: {bc_posterior:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: All edges present in result
# ---------------------------------------------------------------------------


def test_all_edges_present() -> None:
    """posteriors must contain all edge_ids from the input graph."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    expected_ids = {"edge_ab", "edge_bc", "edge_ac"}
    assert set(result.posteriors.keys()) == expected_ids, (
        f"Expected {expected_ids}, got {set(result.posteriors.keys())}"
    )
    assert result.n_variables == 3


# ---------------------------------------------------------------------------
# Test 4: Empty graph
# ---------------------------------------------------------------------------


def test_empty_graph() -> None:
    """Empty graph should return MRFResult with empty posteriors and zero counts."""
    G = nx.MultiDiGraph()
    result = score_graph_mrf(G)

    assert isinstance(result, MRFResult)
    assert result.posteriors == {}
    assert result.n_variables == 0
    assert result.n_rules == 0
    assert result.n_contradictions == 0
    assert result.n_compositions == 0
    assert result.converged is True


# ---------------------------------------------------------------------------
# Test 5: Graph with contradiction
# ---------------------------------------------------------------------------


def test_contradiction_weakens_loser() -> None:
    """Opposing claims with similar conditions — the weaker one should decrease."""
    G = nx.MultiDiGraph()
    G.add_node("A", canonical_name="FGF8", entity_type="protein")
    G.add_node("B", canonical_name="MEK", entity_type="protein")

    # Strong inducer (high confidence)
    G.add_edge(
        "A",
        "B",
        predicate="induces",
        direction="positive",
        confidence_mean=0.90,
        edge_id="edge_strong",
        organism="Mus musculus",
        model_system="HEK293",
        in_vitro=True,
        conditions={},
    )
    # Weak inhibitor — opposing claim, same context (coupling → real contradiction)
    G.add_edge(
        "A",
        "B",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.30,
        edge_id="edge_weak",
        organism="Mus musculus",
        model_system="HEK293",
        in_vitro=True,
        conditions={},
    )

    result = score_graph_mrf(G)

    assert result.n_contradictions >= 1, "Should detect at least one contradiction rule"

    weak_posterior = result.posteriors["edge_weak"]
    strong_posterior = result.posteriors["edge_strong"]

    # The weaker claim should have a lower posterior than the stronger one
    assert weak_posterior < strong_posterior, (
        f"Weak claim ({weak_posterior:.4f}) should be lower than "
        f"strong claim ({strong_posterior:.4f})"
    )
    # And the weak claim's posterior should be lower than its prior (0.30)
    assert weak_posterior < 0.30, (
        f"Contradiction penalty should pull weak claim below its prior 0.30, "
        f"got {weak_posterior:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: No composition rule when A→C edge does not exist
# ---------------------------------------------------------------------------


def test_no_hallucinated_composition() -> None:
    """Composition rules must not be grounded if no direct A→C edge exists."""
    G = nx.MultiDiGraph()
    G.add_node("X", canonical_name="X", entity_type="protein")
    G.add_node("Y", canonical_name="Y", entity_type="protein")
    G.add_node("Z", canonical_name="Z", entity_type="biological_process")

    # X→Y and Y→Z exist but no X→Z edge
    G.add_edge(
        "X",
        "Y",
        predicate="induces",
        direction="positive",
        confidence_mean=0.80,
        edge_id="xy",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    G.add_edge(
        "Y",
        "Z",
        predicate="induces",
        direction="positive",
        confidence_mean=0.75,
        edge_id="yz",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )

    result = score_graph_mrf(G)

    # Only the two real edges should be in posteriors — no hallucinated X→Z
    assert set(result.posteriors.keys()) == {"xy", "yz"}, (
        f"Should only have xy and yz, got {set(result.posteriors.keys())}"
    )
    assert result.n_compositions == 0, (
        f"Expected 0 composition rules (no X→Z edge), got {result.n_compositions}"
    )


# ---------------------------------------------------------------------------
# Test 7: MRFConfig defaults
# ---------------------------------------------------------------------------


def test_mrf_config_defaults() -> None:
    """MRFConfig should have the specified default values."""
    config = MRFConfig()
    assert config.evidence_weight == 10.0
    assert config.contradiction_weight == 8.0
    assert config.composition_weight == 5.0
    assert config.max_iter == 200
    assert config.tol == 1e-6
    assert config.boundary_threshold == 0.6


# ---------------------------------------------------------------------------
# Test 8: Custom config changes weights
# ---------------------------------------------------------------------------


def test_custom_config_affects_result() -> None:
    """Providing a custom config with very high composition weight should boost A→C more."""
    G = _make_scored_graph()

    # Very high composition weight should push edge_ac higher
    config_strong = MRFConfig(composition_weight=50.0)
    result_strong = score_graph_mrf(G, config=config_strong)

    # Default config
    config_default = MRFConfig()
    result_default = score_graph_mrf(G, config=config_default)

    ac_strong = result_strong.posteriors["edge_ac"]
    ac_default = result_default.posteriors["edge_ac"]

    assert ac_strong >= ac_default, (
        f"Higher composition weight should produce >= posterior: "
        f"strong={ac_strong:.4f}, default={ac_default:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 9: Diagnostic counts are consistent
# ---------------------------------------------------------------------------


def test_diagnostic_counts() -> None:
    """n_rules should equal n_variables (unary) + n_contradictions + n_compositions."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    expected_rules = result.n_variables + result.n_contradictions + result.n_compositions
    assert result.n_rules == expected_rules, (
        f"n_rules ({result.n_rules}) != n_variables ({result.n_variables}) "
        f"+ n_contradictions ({result.n_contradictions}) "
        f"+ n_compositions ({result.n_compositions}) = {expected_rules}"
    )


# ---------------------------------------------------------------------------
# Test 10: All posterior values are in [0, 1]
# ---------------------------------------------------------------------------


def test_all_posteriors_bounded() -> None:
    """All posterior truth values must be in [0, 1]."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    for edge_id, val in result.posteriors.items():
        assert 0.0 <= val <= 1.0, f"Posterior for {edge_id} = {val:.6f} is outside [0, 1]"
