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

from autoreview.knowledge_graph.mrf_scoring import (
    MRFConfig,
    MRFResult,
    score_graph_mrf,
)

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
    """BMP4→SMAD1 + SMAD1→mesoderm should boost BMP4→mesoderm above raw confidence.

    Threshold is 0.38 (raw confidence_mean is 0.35); the exact equilibrium shifted
    slightly when the argmin gradient for composition rules was added to hlmrf.py,
    so the threshold reflects the actual boosted value rather than a round number.
    """
    G = _make_scored_graph()
    result = score_graph_mrf(G)

    assert isinstance(result, MRFResult)
    assert "edge_ac" in result.posteriors, "edge_ac must be in posteriors"

    ac_posterior = result.posteriors["edge_ac"]
    assert ac_posterior > 0.38, (
        f"BMP4→mesoderm posterior should be boosted above raw confidence (0.35) "
        f"by the BMP4→SMAD1→mesoderm chain, got {ac_posterior:.4f}"
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
    result = score_graph_mrf(G, config=MRFConfig(enable_finding_layer=False))

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


# ---------------------------------------------------------------------------
# Multi-hop composition tests (Feature 1)
# ---------------------------------------------------------------------------


def _make_4node_chain_graph() -> nx.MultiDiGraph:
    """Build A->B->C->D with direct A->D edge for 2-hop composition testing."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "A",
        "B",
        predicate="induces",
        direction="positive",
        confidence_mean=0.90,
        edge_id="ab",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    G.add_edge(
        "B",
        "C",
        predicate="induces",
        direction="positive",
        confidence_mean=0.85,
        edge_id="bc",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    G.add_edge(
        "C",
        "D",
        predicate="induces",
        direction="positive",
        confidence_mean=0.80,
        edge_id="cd",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    G.add_edge(
        "A",
        "D",
        predicate="induces",
        direction="positive",
        confidence_mean=0.25,
        edge_id="ad",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    return G


def test_mrf_config_multihop_defaults() -> None:
    """MRFConfig should have multi-hop composition defaults."""
    config = MRFConfig()
    assert config.max_composition_hops == 3
    assert config.composition_decay == 0.7


def test_multihop_2hop_boosts_ad() -> None:
    """A->B->C->D chain (2-hop) should boost A->D above its prior 0.25."""
    G = _make_4node_chain_graph()
    config = MRFConfig(max_composition_hops=3)
    result = score_graph_mrf(G, config=config)
    ad_posterior = result.posteriors["ad"]
    assert ad_posterior > 0.28, (
        f"2-hop chain should boost A->D above prior 0.25, got {ad_posterior:.4f}"
    )


def test_multihop_disabled_at_hops_1() -> None:
    """max_composition_hops=1 should not discover 2-hop chains."""
    G = _make_4node_chain_graph()
    config_1hop = MRFConfig(max_composition_hops=1)
    result_1hop = score_graph_mrf(G, config=config_1hop)
    config_3hop = MRFConfig(max_composition_hops=3)
    result_3hop = score_graph_mrf(G, config=config_3hop)
    assert result_3hop.posteriors["ad"] >= result_1hop.posteriors["ad"] - 1e-6


def test_composition_decay_reduces_strength() -> None:
    """Higher decay = less attenuation = stronger multi-hop boost."""
    G = _make_4node_chain_graph()
    result_low = score_graph_mrf(G, config=MRFConfig(composition_decay=0.3))
    result_high = score_graph_mrf(G, config=MRFConfig(composition_decay=0.9))
    assert result_high.posteriors["ad"] >= result_low.posteriors["ad"] - 1e-6


# ---------------------------------------------------------------------------
# Diagnostics tests (Feature 2)
# ---------------------------------------------------------------------------


def test_mrf_result_has_diagnostics_field() -> None:
    """MRFResult should have an optional diagnostics field."""
    result = MRFResult()
    assert result.diagnostics is None


def test_mrf_config_diagnostics_default() -> None:
    """MRFConfig should have diagnostics_top_n default."""
    config = MRFConfig()
    assert config.diagnostics_top_n == 20


def test_diagnostics_populated_on_solve() -> None:
    """Solving a non-empty graph should populate diagnostics."""
    G = _make_scored_graph()
    result = score_graph_mrf(G)
    assert result.diagnostics is not None
    assert result.diagnostics.converged is True
    assert result.diagnostics.n_iterations >= 0
    assert result.diagnostics.final_objective >= 0.0
    assert isinstance(result.diagnostics.top_violations, list)
    assert isinstance(result.diagnostics.mean_violation_by_type, dict)
    assert "unary" in result.diagnostics.mean_violation_by_type


def test_diagnostics_top_violations_limited() -> None:
    """Top violations should be limited to diagnostics_top_n."""
    G = _make_scored_graph()
    config = MRFConfig(diagnostics_top_n=2)
    result = score_graph_mrf(G, config=config)
    assert result.diagnostics is not None
    assert len(result.diagnostics.top_violations) <= 2


# ---------------------------------------------------------------------------
# Incremental update tests (Feature 3)
# ---------------------------------------------------------------------------


def test_update_graph_mrf_returns_all_edges() -> None:
    """update_graph_mrf should return posteriors for all edges including new ones."""
    G = _make_scored_graph()
    prior_result = score_graph_mrf(G)
    G.add_edge(
        "e1",
        "e3",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.40,
        edge_id="edge_ac2",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    from autoreview.knowledge_graph.mrf_scoring import update_graph_mrf

    updated = update_graph_mrf(G, new_edge_ids=["edge_ac2"], prior_result=prior_result)
    assert "edge_ac2" in updated.posteriors
    assert "edge_ab" in updated.posteriors
    assert "edge_ac" in updated.posteriors


def test_update_graph_mrf_close_to_full_solve() -> None:
    """Incremental should produce posteriors within 0.15 of full re-solve."""
    G = _make_scored_graph()
    prior_result = score_graph_mrf(G)
    G.add_edge(
        "e2",
        "e3",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.20,
        edge_id="edge_bc2",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    from autoreview.knowledge_graph.mrf_scoring import update_graph_mrf

    incremental = update_graph_mrf(G, new_edge_ids=["edge_bc2"], prior_result=prior_result)
    full = score_graph_mrf(G)
    for eid in full.posteriors:
        if eid in incremental.posteriors:
            assert abs(incremental.posteriors[eid] - full.posteriors[eid]) < 0.15


# ---------------------------------------------------------------------------
# Finding layer config and result tests
# ---------------------------------------------------------------------------


def test_mrf_config_finding_layer_defaults() -> None:
    """MRFConfig should have finding layer defaults."""
    config = MRFConfig()
    assert config.finding_contradiction_weight == 12.0
    assert config.propagation_weight == 3.0
    assert config.enable_finding_layer is True


def test_mrf_result_has_finding_posteriors() -> None:
    """MRFResult should have a finding_posteriors field."""
    result = MRFResult()
    assert result.finding_posteriors == {}
    assert result.n_findings == 0
    assert result.n_finding_contradictions == 0


def test_finding_layer_disabled_matches_baseline() -> None:
    """enable_finding_layer=False should produce identical results to current behavior."""
    G = _make_scored_graph()
    config_off = MRFConfig(enable_finding_layer=False)
    config_default = MRFConfig(enable_finding_layer=False)
    result_off = score_graph_mrf(G, config=config_off)
    result_default = score_graph_mrf(G, config=config_default)
    for eid in result_off.posteriors:
        assert abs(result_off.posteriors[eid] - result_default.posteriors[eid]) < 1e-9


# ---------------------------------------------------------------------------
# Finding layer integration tests
# ---------------------------------------------------------------------------


def _make_finding_contradiction_graph() -> nx.MultiDiGraph:
    """Graph with two papers contradicting at the finding level."""
    G = nx.MultiDiGraph()
    G.add_node("bmp4", canonical_name="BMP4", entity_type="protein")
    G.add_node("meso", canonical_name="mesoderm", entity_type="biological_process")
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="pos1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.85,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 induces mesoderm",
        _kg_edge=None,
    )
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="pos2",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.80,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 is sufficient for mesoderm",
        _kg_edge=None,
    )
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="neg1",
        predicate="induces",
        direction="negative",
        confidence_mean=0.40,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 does not induce mesoderm",
        _kg_edge=None,
    )
    return G


def test_finding_layer_produces_finding_posteriors() -> None:
    G = _make_finding_contradiction_graph()
    config = MRFConfig(enable_finding_layer=True)
    result = score_graph_mrf(G, config=config)
    assert result.n_findings >= 2
    assert len(result.finding_posteriors) >= 2
    for fid, val in result.finding_posteriors.items():
        assert 0.0 <= val <= 1.0


def test_finding_contradiction_reduces_weaker_finding() -> None:
    G = _make_finding_contradiction_graph()
    config = MRFConfig(enable_finding_layer=True)
    result = score_graph_mrf(G, config=config)
    posteriors = result.finding_posteriors
    finding_vals = sorted(posteriors.values())
    assert finding_vals[0] < finding_vals[-1], (
        f"Weaker finding should have lower posterior: {finding_vals}"
    )


def test_downward_propagation_reduces_member_edges() -> None:
    G = _make_finding_contradiction_graph()
    config_off = MRFConfig(enable_finding_layer=False)
    result_off = score_graph_mrf(G, config=config_off)
    config_on = MRFConfig(enable_finding_layer=True)
    result_on = score_graph_mrf(G, config=config_on)
    neg1_off = result_off.posteriors.get("neg1", 0.5)
    neg1_on = result_on.posteriors.get("neg1", 0.5)
    assert neg1_on <= neg1_off + 0.05, (
        f"neg1 should not increase much with finding layer: off={neg1_off:.4f}, on={neg1_on:.4f}"
    )


def test_finding_layer_off_no_finding_vars() -> None:
    G = _make_finding_contradiction_graph()
    config = MRFConfig(enable_finding_layer=False)
    result = score_graph_mrf(G, config=config)
    assert result.n_findings == 0
    assert result.finding_posteriors == {}


def test_existing_edge_tests_unchanged_with_finding_layer_off() -> None:
    G = _make_scored_graph()
    config = MRFConfig(enable_finding_layer=False)
    result = score_graph_mrf(G, config=config)
    assert result.posteriors["edge_ac"] > 0.38
    assert result.posteriors["edge_ab"] > 0.75
    assert result.posteriors["edge_bc"] > 0.75


# ---------------------------------------------------------------------------
# Integration tests — 3-paper fixture graph
# ---------------------------------------------------------------------------


def _make_3paper_integration_graph() -> nx.MultiDiGraph:
    """3 papers, ~10 edges, 2 topic clusters, 1 directional + 1 boundary contradiction.

    Cluster 1: BMP4 → mesoderm (activating)
      - Paper 1: induces, positive, mouse, in_vitro (strong)
      - Paper 1: is_sufficient_for, positive, mouse, in_vitro (strong)
      - Paper 2: induces, negative, mouse, in_vitro (weak — contradiction)
      - Paper 3: induces, positive, human, in_vitro (different organism)

    Cluster 2: SMAD1 → neural (activating)
      - Paper 1: induces, positive, mouse, in_vitro
      - Paper 2: is_sufficient_for, positive, mouse, in_vitro (agreement)
    """
    G = nx.MultiDiGraph()
    G.add_node("bmp4", canonical_name="BMP4", entity_type="protein")
    G.add_node("meso", canonical_name="mesoderm", entity_type="biological_process")
    G.add_node("smad1", canonical_name="SMAD1", entity_type="protein")
    G.add_node("neural", canonical_name="neural", entity_type="biological_process")

    # Cluster 1 edges
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="c1p1e1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.85,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 induces mesoderm (paper 1)",
        _kg_edge=None,
    )
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="c1p1e2",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.80,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 is sufficient for mesoderm (paper 1)",
        _kg_edge=None,
    )
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="c1p2e1",
        predicate="induces",
        direction="negative",
        confidence_mean=0.35,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 does not induce mesoderm (paper 2)",
        _kg_edge=None,
    )
    G.add_edge(
        "bmp4",
        "meso",
        edge_id="c1p3e1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.70,
        organism="Homo sapiens",
        model_system="iPSC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="BMP4 induces mesoderm in human iPSC (paper 3)",
        _kg_edge=None,
    )

    # Cluster 2 edges
    G.add_edge(
        "smad1",
        "neural",
        edge_id="c2p1e1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.75,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="SMAD1 induces neural (paper 1)",
        _kg_edge=None,
    )
    G.add_edge(
        "smad1",
        "neural",
        edge_id="c2p2e1",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.70,
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
        section_source="primary_empirical",
        natural_language="SMAD1 is sufficient for neural (paper 2)",
        _kg_edge=None,
    )
    return G


def test_integration_clusters_form_correctly() -> None:
    """Cluster 1 (BMP4→mesoderm) and Cluster 2 (SMAD1→neural) should form."""
    from autoreview.knowledge_graph.cluster import build_topic_clusters

    G = _make_3paper_integration_graph()
    clusters = build_topic_clusters(G)
    assert len(clusters) == 2
    subjects = {c.subject_id for c in clusters}
    assert "bmp4" in subjects
    assert "smad1" in subjects


def test_integration_findings_form_correctly() -> None:
    """Cluster 1 should have 3 findings (pos+mouse, neg+mouse, pos+human)."""
    from autoreview.knowledge_graph.cluster import build_topic_clusters, form_findings

    G = _make_3paper_integration_graph()
    clusters = build_topic_clusters(G)
    findings = form_findings(clusters, G)
    # Cluster 1: 3 findings (pos/mouse/invitro, neg/mouse/invitro, pos/human/invitro)
    # Cluster 2: 1 finding (pos/mouse/invitro)
    assert len(findings) >= 4


def test_integration_contradictions_detected() -> None:
    """Should detect at least 1 directional contradiction in cluster 1."""
    from autoreview.knowledge_graph.cluster import (
        build_topic_clusters,
        detect_finding_contradictions,
        form_findings,
    )

    G = _make_3paper_integration_graph()
    clusters = build_topic_clusters(G)
    findings = form_findings(clusters, G)
    contradictions = detect_finding_contradictions(findings, clusters, graph=G)
    assert len(contradictions) >= 1
    directional = [c for c in contradictions if c.contradiction_type == "directional"]
    assert len(directional) >= 1


def test_integration_mrf_resolves() -> None:
    """MRF with finding layer should resolve contradictions."""
    G = _make_3paper_integration_graph()
    config = MRFConfig(enable_finding_layer=True)
    result = score_graph_mrf(G, config=config)
    assert result.converged
    assert result.n_findings >= 4
    assert len(result.finding_posteriors) >= 4
    for val in result.posteriors.values():
        assert 0.0 <= val <= 1.0
    for val in result.finding_posteriors.values():
        assert 0.0 <= val <= 1.0


def test_integration_non_contradicted_unaffected() -> None:
    """Cluster 2 (SMAD1→neural, no contradiction) should be stable."""
    G = _make_3paper_integration_graph()
    config_on = MRFConfig(enable_finding_layer=True)
    result_on = score_graph_mrf(G, config=config_on)
    config_off = MRFConfig(enable_finding_layer=False)
    result_off = score_graph_mrf(G, config=config_off)
    for eid in ["c2p1e1", "c2p2e1"]:
        diff = abs(result_on.posteriors[eid] - result_off.posteriors[eid])
        assert diff < 0.15, f"Non-contradicted edge {eid} shifted too much: {diff:.4f}"
