"""Tests for bayesian/scoring.py."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx  # noqa: E402


def _make_scored_graph() -> nx.MultiDiGraph:
    """3-node graph with composition chain and moderate confidence."""
    graph = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        graph.add_node(n, canonical_name=n, entity_type="protein")
    graph.add_edge(
        "A",
        "B",
        predicate="induces",
        direction="positive",
        confidence_mean=0.85,
        evidence_count=3,
        edge_id="ab",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    graph.add_edge(
        "B",
        "C",
        predicate="induces",
        direction="positive",
        confidence_mean=0.80,
        evidence_count=2,
        edge_id="bc",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    graph.add_edge(
        "A",
        "C",
        predicate="induces",
        direction="positive",
        confidence_mean=0.30,
        evidence_count=1,
        edge_id="ac",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    return graph


def test_bayesian_result_empty() -> None:
    """Empty BayesianResult should have sensible defaults."""
    from autoreview.knowledge_graph.bayesian.scoring import BayesianResult

    result = BayesianResult()
    assert result.posteriors == {}
    assert result.n_variables == 0


def test_score_graph_bayesian_returns_all_edges() -> None:
    """score_graph_bayesian should return posteriors for every edge."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    graph = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=0,  # Laplace only, no NUTS hotspots
    )
    result = score_graph_bayesian(graph, config=config)
    assert "ab" in result.posteriors
    assert "bc" in result.posteriors
    assert "ac" in result.posteriors
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())
    assert result.n_variables == 3


def test_score_graph_bayesian_credible_intervals() -> None:
    """Credible intervals should bracket the posterior mean."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    graph = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=0,
    )
    result = score_graph_bayesian(graph, config=config)
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        mean = result.posteriors[eid]
        assert lo <= mean <= hi, f"{eid}: CI [{lo:.3f}, {hi:.3f}] does not contain mean {mean:.3f}"


def test_score_graph_bayesian_with_hotspots() -> None:
    """With hotspot_top_k > 0 and contradictions present, hotspot edges should have NUTS samples."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    graph = _make_scored_graph()
    # Add contradicting edge so contradiction centrality finds hotspots
    graph.add_edge(
        "A",
        "B",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.60,
        evidence_count=2,
        edge_id="ab_inh",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=3,
        hotspot_hop_radius=1,
    )
    result = score_graph_bayesian(graph, config=config)
    # Should have some edges with posterior samples from NUTS
    assert len(result.posterior_samples) > 0


def test_score_graph_bayesian_empty_graph() -> None:
    """Empty graph should return empty result."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    graph = nx.MultiDiGraph()
    result = score_graph_bayesian(graph, config=BayesianConfig())
    assert result.posteriors == {}
    assert result.n_variables == 0


def test_update_graph_bayesian_returns_all_edges() -> None:
    """update_graph_bayesian should return posteriors for all edges."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import (
        score_graph_bayesian,
        update_graph_bayesian,
    )

    graph = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50,
        n_samples=100,
        n_chains=1,
        seed=42,
        hotspot_top_k=0,
    )
    prior = score_graph_bayesian(graph, config=config)

    # Add a new edge
    graph.add_edge(
        "A",
        "C",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.40,
        evidence_count=1,
        edge_id="ac2",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )

    updated = update_graph_bayesian(graph, ["ac2"], prior, config=config)
    assert "ac2" in updated.posteriors
    assert "ab" in updated.posteriors
    assert "bc" in updated.posteriors
    assert "ac" in updated.posteriors
