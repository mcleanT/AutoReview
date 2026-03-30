"""Bayesian scoring orchestration for the knowledge graph.

Provides three public callables:

- ``BayesianResult`` — dataclass holding posterior means, credible intervals,
  raw NUTS samples, bimodality flags, and diagnostics.

- ``score_graph_bayesian(graph, config) -> BayesianResult``
  Full pipeline: Laplace approximation over the whole graph, then optional
  targeted NUTS on contradiction-hotspot subgraphs.

- ``update_graph_bayesian(graph, changed_edge_ids, prior_result, config)
  -> BayesianResult``
  Incremental update: runs NUTS on the subgraph around changed edges and
  merges new posteriors into the prior result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.diagnostics import BayesianDiagnostics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BayesianResult:
    """Posterior inference results for a knowledge graph.

    Attributes:
        posteriors: Posterior mean per edge_id, clamped to [0, 1].
        credible_intervals: 95% CI per edge_id as (lower, upper).
        posterior_samples: Raw NUTS samples per edge_id (shape: (n_samples,)).
            Only populated for edges that were sampled via NUTS.
        bimodality_flags: True for edges flagged as bimodal by Hartigan's dip.
        diagnostics: MCMC diagnostics from the most recent NUTS run, or None
            if only Laplace approximation was used.
        n_variables: Number of distinct graph nodes (entity variables).
        n_contradictions: Number of contradiction pairs found in the graph.
        n_compositions: Number of composition chains discovered.
    """

    posteriors: dict[str, float] = field(default_factory=dict)
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    posterior_samples: dict[str, np.ndarray] = field(default_factory=dict)
    bimodality_flags: dict[str, bool] = field(default_factory=dict)
    diagnostics: BayesianDiagnostics | None = None
    n_variables: int = 0
    n_contradictions: int = 0
    n_compositions: int = 0


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def score_graph_bayesian(
    graph: nx.MultiDiGraph,
    config: BayesianConfig | None = None,
) -> BayesianResult:
    """Run Bayesian inference over the full knowledge graph.

    Pipeline:
    1. Prepare :class:`~autoreview.knowledge_graph.bayesian.model.ModelInputs`
       from the graph.
    2. Run Laplace approximation for fast, full-graph credible intervals.
    3. If ``config.hotspot_top_k > 0``:
       a. Score nodes by contradiction centrality.
       b. Extract a subgraph around the top-K hotspot nodes.
       c. Run NUTS on the subgraph.
       d. Merge NUTS posteriors (and CIs, samples) into the Laplace results —
          NUTS values override Laplace for the hotspot edges.
    4. Return a :class:`BayesianResult` with merged posteriors.

    Args:
        graph: Knowledge graph as a NetworkX ``MultiDiGraph``. Edges must
            carry at minimum ``edge_id``, ``predicate``, ``confidence_mean``,
            ``evidence_count`` attributes.
        config: :class:`~autoreview.knowledge_graph.bayesian.config.BayesianConfig`
            controlling inference hyper-parameters. Defaults to
            ``BayesianConfig()`` if not provided.

    Returns:
        :class:`BayesianResult` with posteriors, credible intervals, and
        optional NUTS samples.
    """
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import (
        laplace_approximate,
    )
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    if config is None:
        config = BayesianConfig()

    # ------------------------------------------------------------------
    # Guard: empty graph
    # ------------------------------------------------------------------
    if graph.number_of_edges() == 0:
        return BayesianResult(
            n_variables=graph.number_of_nodes(),
        )

    # ------------------------------------------------------------------
    # Step 1: prepare model inputs
    # ------------------------------------------------------------------
    inputs = prepare_model_inputs(graph, config)

    # ------------------------------------------------------------------
    # Step 2: Laplace approximation over full graph
    # ------------------------------------------------------------------
    laplace = laplace_approximate(inputs, config)

    posteriors: dict[str, float] = dict(laplace.means)
    cis: dict[str, tuple[float, float]] = dict(laplace.credible_intervals)
    samples: dict[str, np.ndarray] = {}
    bimodality_flags: dict[str, bool] = {}
    nuts_diagnostics = None

    # ------------------------------------------------------------------
    # Step 3 (optional): targeted NUTS on hotspot subgraph
    # ------------------------------------------------------------------
    if config.hotspot_top_k > 0:
        nuts_diagnostics = _run_hotspot_nuts(
            graph=graph,
            config=config,
            posteriors=posteriors,
            cis=cis,
            samples=samples,
            bimodality_flags=bimodality_flags,
        )

    log.debug(
        "score_graph_bayesian done",
        n_edges=inputs.n_edges,
        n_nodes=graph.number_of_nodes(),
        n_contradictions=len(inputs.contra_a_idx),
        n_compositions=len(inputs.composition_chains),
        hotspot_top_k=config.hotspot_top_k,
    )

    return BayesianResult(
        posteriors=posteriors,
        credible_intervals=cis,
        posterior_samples=samples,
        bimodality_flags=bimodality_flags,
        diagnostics=nuts_diagnostics,
        n_variables=graph.number_of_nodes(),
        n_contradictions=len(inputs.contra_a_idx),
        n_compositions=len(inputs.composition_chains),
    )


def _run_hotspot_nuts(
    graph: nx.MultiDiGraph,
    config: BayesianConfig,
    posteriors: dict[str, float],
    cis: dict[str, tuple[float, float]],
    samples: dict[str, np.ndarray],
    bimodality_flags: dict[str, bool],
) -> BayesianDiagnostics | None:
    """Run NUTS on hotspot subgraph and mutate result dicts in-place.

    Returns the NUTS diagnostics if NUTS ran, or None if no hotspots found.
    """
    import jax

    from autoreview.knowledge_graph.analysis import score_contradiction_centrality
    from autoreview.knowledge_graph.bayesian.inference import (
        extract_inference_subgraph,
        sample_posterior,
    )
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    # Score nodes by contradiction centrality
    centrality = score_contradiction_centrality(graph, config.boundary_threshold)
    if not centrality:
        log.debug("score_graph_bayesian: no contradiction hotspots found, skipping NUTS")
        return None

    # Extract top-K hotspot nodes
    top_k = min(config.hotspot_top_k, len(centrality))
    hotspot_node_ids: set[str] = {entry["node_id"] for entry in centrality[:top_k]}

    # Build subgraph around hotspot nodes
    subgraph = extract_inference_subgraph(
        graph, target_nodes=hotspot_node_ids, hop_radius=config.hotspot_hop_radius
    )
    if subgraph.number_of_edges() == 0:
        log.debug("score_graph_bayesian: hotspot subgraph has no edges, skipping NUTS")
        return None

    # Prepare inputs for the subgraph
    sub_inputs = prepare_model_inputs(subgraph, config)

    # NUTS sampling
    rng_key = jax.random.PRNGKey(config.seed)
    nuts_result = sample_posterior(sub_inputs, config, rng_key)

    # Merge NUTS results — override Laplace for hotspot edges
    for eid, edge_samples in nuts_result.samples.items():
        mean_val = float(np.mean(edge_samples))
        lo = float(np.percentile(edge_samples, 2.5))
        hi = float(np.percentile(edge_samples, 97.5))
        posteriors[eid] = np.clip(mean_val, 0.0, 1.0)
        cis[eid] = (np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0))
        samples[eid] = edge_samples
        if nuts_result.diagnostics is not None:
            bimodality_flags[eid] = eid in nuts_result.diagnostics.bimodal_edges

    return nuts_result.diagnostics


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


def update_graph_bayesian(
    graph: nx.MultiDiGraph,
    changed_edge_ids: list[str],
    prior_result: BayesianResult,
    config: BayesianConfig | None = None,
) -> BayesianResult:
    """Incrementally update posteriors for a set of changed edges.

    Extracts the subgraph around the nodes touched by ``changed_edge_ids``,
    runs NUTS on that subgraph, and merges the new posteriors into a copy of
    ``prior_result`` (NUTS edges override prior values; unchanged edges keep
    their prior posteriors).

    Args:
        graph: Full knowledge graph (including the changed edges already
            present as graph edges).
        changed_edge_ids: List of ``edge_id`` values that were added or
            modified.
        prior_result: Previous :class:`BayesianResult` to merge into.
        config: :class:`~autoreview.knowledge_graph.bayesian.config.BayesianConfig`.
            Defaults to ``BayesianConfig()`` if not provided.

    Returns:
        New :class:`BayesianResult` with NUTS-updated posteriors for changed
        edges and prior posteriors for all other edges.
    """
    import jax

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import (
        extract_inference_subgraph,
        sample_posterior,
    )
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    if config is None:
        config = BayesianConfig()

    # Start from a shallow copy of the prior result
    posteriors = dict(prior_result.posteriors)
    cis = dict(prior_result.credible_intervals)
    samples = dict(prior_result.posterior_samples)
    bimodality_flags = dict(prior_result.bimodality_flags)

    # Find all nodes involved in changed edges
    changed_nodes: set[str] = set()
    edge_id_to_endpoints: dict[str, tuple[str, str]] = {
        data["edge_id"]: (u, v) for u, v, data in graph.edges(data=True) if "edge_id" in data
    }
    for eid in changed_edge_ids:
        if eid in edge_id_to_endpoints:
            u, v = edge_id_to_endpoints[eid]
            changed_nodes.add(u)
            changed_nodes.add(v)

    if not changed_nodes:
        log.warning(
            "update_graph_bayesian: none of changed_edge_ids found in graph",
            changed_edge_ids=changed_edge_ids,
        )
        return BayesianResult(
            posteriors=posteriors,
            credible_intervals=cis,
            posterior_samples=samples,
            bimodality_flags=bimodality_flags,
            diagnostics=prior_result.diagnostics,
            n_variables=prior_result.n_variables,
            n_contradictions=prior_result.n_contradictions,
            n_compositions=prior_result.n_compositions,
        )

    # Extract subgraph around changed nodes
    subgraph = extract_inference_subgraph(
        graph,
        target_nodes=changed_nodes,
        hop_radius=config.hotspot_hop_radius,
    )

    if subgraph.number_of_edges() == 0:
        return BayesianResult(
            posteriors=posteriors,
            credible_intervals=cis,
            posterior_samples=samples,
            bimodality_flags=bimodality_flags,
            diagnostics=prior_result.diagnostics,
            n_variables=prior_result.n_variables,
            n_contradictions=prior_result.n_contradictions,
            n_compositions=prior_result.n_compositions,
        )

    # Prepare inputs and run NUTS
    sub_inputs = prepare_model_inputs(subgraph, config)
    rng_key = jax.random.PRNGKey(config.seed)
    nuts_result = sample_posterior(sub_inputs, config, rng_key)

    # Merge NUTS results — override prior for subgraph edges
    for eid, edge_samples in nuts_result.samples.items():
        mean_val = float(np.mean(edge_samples))
        lo = float(np.percentile(edge_samples, 2.5))
        hi = float(np.percentile(edge_samples, 97.5))
        posteriors[eid] = float(np.clip(mean_val, 0.0, 1.0))
        cis[eid] = (float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0)))
        samples[eid] = edge_samples
        if nuts_result.diagnostics is not None:
            bimodality_flags[eid] = eid in nuts_result.diagnostics.bimodal_edges

    log.debug(
        "update_graph_bayesian done",
        changed_edges=len(changed_edge_ids),
        subgraph_edges=subgraph.number_of_edges(),
        updated_edges=len(nuts_result.samples),
    )

    return BayesianResult(
        posteriors=posteriors,
        credible_intervals=cis,
        posterior_samples=samples,
        bimodality_flags=bimodality_flags,
        diagnostics=nuts_result.diagnostics,
        n_variables=graph.number_of_nodes(),
        n_contradictions=prior_result.n_contradictions,
        n_compositions=prior_result.n_compositions,
    )
