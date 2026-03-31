"""Tests for bayesian/inference.py."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx  # noqa: E402


def _make_simple_graph() -> nx.MultiDiGraph:
    """Two independent edges — no contradictions, no compositions."""
    graph = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        graph.add_node(n, canonical_name=n, entity_type="protein")
    graph.add_edge(
        "A",
        "B",
        predicate="induces",
        direction="positive",
        confidence_mean=0.80,
        evidence_count=4,
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
        confidence_mean=0.60,
        evidence_count=2,
        edge_id="bc",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    return graph


def _make_chain_graph() -> nx.MultiDiGraph:
    """A->B->C with A->C direct — composition chain present."""
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


def _make_4node_graph() -> nx.MultiDiGraph:
    """A->B->C->D linear chain with disconnected E->F."""
    graph = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D", "E", "F"]:
        graph.add_node(n, canonical_name=n, entity_type="protein")
    for src, dst, eid in [("A", "B", "ab"), ("B", "C", "bc"), ("C", "D", "cd"), ("E", "F", "ef")]:
        graph.add_edge(
            src,
            dst,
            predicate="induces",
            direction="positive",
            confidence_mean=0.70,
            evidence_count=2,
            edge_id=eid,
            organism="Mus musculus",
            model_system="mESC",
            in_vitro=True,
            conditions={},
        )
    return graph


def test_extract_inference_subgraph_basic() -> None:
    """Subgraph around node B with hop_radius=1 should include A, B, C."""
    from autoreview.knowledge_graph.bayesian.inference import extract_inference_subgraph

    graph = _make_4node_graph()
    sub = extract_inference_subgraph(graph, target_nodes={"B"}, hop_radius=1)
    assert set(sub.nodes()) == {"A", "B", "C"}
    assert sub.number_of_edges() == 2  # ab, bc


def test_extract_inference_subgraph_disconnected() -> None:
    """Subgraph around B should NOT include disconnected E, F."""
    from autoreview.knowledge_graph.bayesian.inference import extract_inference_subgraph

    graph = _make_4node_graph()
    sub = extract_inference_subgraph(graph, target_nodes={"B"}, hop_radius=1)
    assert "E" not in sub.nodes()
    assert "F" not in sub.nodes()


def test_laplace_approximate_returns_result() -> None:
    """Laplace on a simple graph should return means, variances, CIs."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import laplace_approximate
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    graph = _make_simple_graph()
    inputs = prepare_model_inputs(graph, BayesianConfig())
    result = laplace_approximate(inputs, BayesianConfig())
    assert "ab" in result.means
    assert "bc" in result.means
    assert all(0.0 <= v <= 1.0 for v in result.means.values())
    assert all(v >= 0.0 for v in result.variances.values())
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        assert 0.0 <= lo <= hi <= 1.0


def test_laplace_map_near_prior_mean() -> None:
    """With no constraints, Laplace MAP should be near Beta prior mode."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import laplace_approximate
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    graph = _make_simple_graph()
    inputs = prepare_model_inputs(graph, BayesianConfig())
    result = laplace_approximate(inputs, BayesianConfig())
    # ab: confidence_mean=0.80, evidence_count=4 -> kappa=6, alpha=4.8, beta=1.2
    # Beta MAP (mode) = (alpha-1)/(alpha+beta-2) = 3.8/4.0 = 0.95
    # The MAP should be between the prior mean (0.80) and mode (0.95).
    # Tolerance: within 0.20 of the prior mean to capture both possibilities.
    assert abs(result.means["ab"] - 0.80) < 0.20, (
        f"MAP should be near prior mode/mean, got {result.means['ab']:.3f}"
    )


def test_sample_posterior_returns_samples() -> None:
    """NUTS should produce samples with correct structure."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import sample_posterior
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    graph = _make_simple_graph()
    config = BayesianConfig(n_warmup=50, n_samples=100, n_chains=1, seed=42)
    inputs = prepare_model_inputs(graph, config)
    rng_key = jax.random.PRNGKey(42)
    result = sample_posterior(inputs, config, rng_key)
    assert "ab" in result.samples
    assert "bc" in result.samples
    assert len(result.samples["ab"]) == 100  # n_samples
    assert all(0.0 <= s <= 1.0 for s in result.samples["ab"])


def test_sample_posterior_diagnostics() -> None:
    """NUTS diagnostics should report convergence info."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import sample_posterior
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    graph = _make_simple_graph()
    config = BayesianConfig(n_warmup=100, n_samples=200, n_chains=2, seed=42)
    inputs = prepare_model_inputs(graph, config)
    rng_key = jax.random.PRNGKey(42)
    result = sample_posterior(inputs, config, rng_key)
    assert result.diagnostics is not None
    assert isinstance(result.diagnostics.max_rhat, float)
    assert isinstance(result.diagnostics.min_ess, float)
    assert result.diagnostics.sampling_time_seconds > 0.0
