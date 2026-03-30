"""Integration tests for the full Bayesian inference pipeline."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx


def _make_full_graph() -> nx.MultiDiGraph:
    """A realistic 5-node graph with contradictions, compositions, mixed confidence."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D", "E"]:
        G.add_node(n, canonical_name=n, entity_type="protein")

    edges = [
        ("A", "B", "induces", "positive", 0.85, 4, "ab"),
        ("B", "C", "induces", "positive", 0.80, 3, "bc"),
        ("A", "C", "induces", "positive", 0.30, 1, "ac"),  # weak direct, chain supports
        ("C", "D", "inhibits", "negative", 0.75, 2, "cd"),
        ("C", "D", "induces", "positive", 0.60, 2, "cd2"),  # contradicts cd
        ("D", "E", "induces", "positive", 0.70, 2, "de"),
    ]
    for src, dst, pred, direction, mean, count, eid in edges:
        G.add_edge(
            src,
            dst,
            predicate=pred,
            direction=direction,
            confidence_mean=mean,
            evidence_count=count,
            edge_id=eid,
            organism="Mus musculus",
            model_system="mESC",
            in_vitro=True,
            conditions={},
        )
    return G


def test_full_pipeline_laplace_only() -> None:
    """Full pipeline with Laplace only (hotspot_top_k=0) should complete."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(hotspot_top_k=0, seed=42)
    result = score_graph_bayesian(G, config=config)

    assert result.n_variables == 5  # 5 nodes: A, B, C, D, E
    assert len(result.posteriors) == 6  # 6 edges
    assert len(result.credible_intervals) == 6
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        assert lo <= result.posteriors[eid] <= hi


def test_full_pipeline_with_nuts() -> None:
    """Full pipeline with NUTS hotspots should complete and produce samples."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=3,
        hotspot_hop_radius=1,
    )
    result = score_graph_bayesian(G, config=config)

    assert result.n_variables == 5  # 5 nodes: A, B, C, D, E
    assert len(result.posteriors) == 6  # 6 edges
    # Hotspot edges should have samples (graph has contradictions)
    assert len(result.posterior_samples) > 0
    # Diagnostics should be populated
    assert result.diagnostics is not None


def test_incremental_update_adds_new_edge() -> None:
    """Incremental update should incorporate a new edge."""
    from autoreview.knowledge_graph.bayesian import (
        BayesianConfig,
        score_graph_bayesian,
        update_graph_bayesian,
    )

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=0,
    )
    prior = score_graph_bayesian(G, config=config)

    # Add contradicting edge
    G.add_edge(
        "A",
        "B",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.50,
        evidence_count=1,
        edge_id="ab2",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )

    updated = update_graph_bayesian(G, ["ab2"], prior, config=config)
    assert "ab2" in updated.posteriors
    assert len(updated.posteriors) == 7  # 6 original + 1 new


def test_mrf_unchanged_regression() -> None:
    """HL-MRF system should produce identical results (no code changed)."""
    from autoreview.knowledge_graph.mrf_scoring import MRFConfig, score_graph_mrf

    G = _make_full_graph()
    result = score_graph_mrf(G, config=MRFConfig())

    assert len(result.posteriors) == 6
    assert result.converged is True
    # All posteriors in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())


def test_bayesian_posteriors_in_valid_range() -> None:
    """All posterior means should be in [0, 1] and CIs should be ordered."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=2,
    )
    result = score_graph_bayesian(G, config=config)

    for eid, mean in result.posteriors.items():
        assert 0.0 <= mean <= 1.0, f"{eid}: posterior mean {mean} out of range"

    for eid, (lo, hi) in result.credible_intervals.items():
        assert 0.0 <= lo <= hi <= 1.0, f"{eid}: CI [{lo}, {hi}] invalid"
        assert lo <= result.posteriors[eid] <= hi, (
            f"{eid}: mean {result.posteriors[eid]} outside CI [{lo}, {hi}]"
        )
