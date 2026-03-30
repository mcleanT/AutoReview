"""Tests for bayesian/model.py."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import jax.numpy as jnp
import networkx as nx
from numpyro.infer.util import log_density


def _make_3node_graph() -> nx.MultiDiGraph:
    """A->B->C with A->C direct edge. Induces 1 composition chain."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
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
    G.add_edge(
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
    G.add_edge(
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
    return G


def _make_contradiction_graph() -> nx.MultiDiGraph:
    """Two opposing edges on same subject-object pair."""
    G = nx.MultiDiGraph()
    for n in ["X", "Y"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "X",
        "Y",
        predicate="induces",
        direction="positive",
        confidence_mean=0.75,
        evidence_count=2,
        edge_id="xy_ind",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    G.add_edge(
        "X",
        "Y",
        predicate="inhibits",
        direction="negative",
        confidence_mean=0.70,
        evidence_count=2,
        edge_id="xy_inh",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    return G


def test_prepare_model_inputs_edge_ids() -> None:
    """prepare_model_inputs should collect all edges with sorted IDs."""
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert set(inputs.edge_ids) == {"ab", "bc", "ac"}
    assert inputs.n_edges == 3


def test_prepare_model_inputs_alpha_beta_derived() -> None:
    """Alpha/beta should be derived from confidence_mean + evidence_count."""
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    idx = inputs.edge_index["ab"]
    # ab: mean=0.85, evidence_count=3 -> kappa=5.0
    # alpha=0.85*5=4.25, beta=0.15*5=0.75
    assert abs(inputs.alpha_priors[idx] - 4.25) < 0.01
    assert abs(inputs.beta_priors[idx] - 0.75) < 0.01


def test_prepare_model_inputs_contradictions() -> None:
    """Should detect opposing predicates as contradictions."""
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_contradiction_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert len(inputs.contra_a_idx) > 0, "Should detect at least one contradiction"


def test_prepare_model_inputs_compositions() -> None:
    """A->B->C chain should produce a composition chain targeting A->C."""
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert len(inputs.composition_chains) > 0, "Should discover A->B->C composition"
    chain = inputs.composition_chains[0]
    target_eid = inputs.edge_ids[chain.target_idx]
    assert target_eid == "ac", f"Target should be ac, got {target_eid}"


def test_kg_flat_model_finite_log_density() -> None:
    """Model should produce finite log-density for reasonable truth values."""
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model, prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    truth_vals = jnp.array([0.5] * inputs.n_edges)
    log_p, _ = log_density(
        kg_flat_model,
        (inputs,),
        {"w_contradiction": 8.0, "w_composition": 5.0},
        {"truth": truth_vals},
    )
    assert jnp.isfinite(log_p), f"Log-density should be finite, got {log_p}"


def test_kg_flat_model_gradient_defined() -> None:
    """Gradient of log-density w.r.t. truth should be finite (no NaN)."""
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model, prepare_model_inputs

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())

    def log_post(truth_vals):
        log_p, _ = log_density(
            kg_flat_model,
            (inputs,),
            {"w_contradiction": 8.0, "w_composition": 5.0},
            {"truth": truth_vals},
        )
        return log_p

    truth_vals = jnp.array([0.5] * inputs.n_edges)
    grad = jax.grad(log_post)(truth_vals)
    assert jnp.all(jnp.isfinite(grad)), f"Gradient should be finite, got {grad}"


def test_kg_flat_model_no_constraints_matches_prior() -> None:
    """With no contradictions or compositions, MAP should equal Beta prior mean."""
    from autoreview.knowledge_graph.bayesian.model import ModelInputs, kg_flat_model

    inputs = ModelInputs(
        edge_ids=["e1"],
        edge_index={"e1": 0},
        n_edges=1,
        alpha_priors=np.array([5.0]),
        beta_priors=np.array([2.0]),
        contra_a_idx=np.array([], dtype=np.intp),
        contra_b_idx=np.array([], dtype=np.intp),
        contra_coupling=np.array([], dtype=np.float64),
        composition_chains=[],
    )

    # Log-density should be maximized near Beta mode = (a-1)/(a+b-2) = 4/5 = 0.8
    def neg_log_post(truth_vals):
        log_p, _ = log_density(
            kg_flat_model,
            (inputs,),
            {"w_contradiction": 8.0, "w_composition": 5.0},
            {"truth": truth_vals},
        )
        return -log_p

    # Evaluate at mode vs away from mode
    val_at_mode = neg_log_post(jnp.array([0.8]))
    val_away = neg_log_post(jnp.array([0.3]))
    assert val_at_mode < val_away, "Log-density should be higher at Beta mode"
