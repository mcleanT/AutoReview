"""NumPyro factor graph model for Bayesian inference over the knowledge graph.

Provides two public callables:

- ``prepare_model_inputs(graph, config) -> ModelInputs``
  Converts a NetworkX MultiDiGraph into the plain-data struct consumed by the
  NumPyro model.

- ``kg_flat_model(inputs, w_contradiction, w_composition)``
  NumPyro probabilistic program.  Beta priors on each edge's truth value,
  soft contradiction penalties and composition consistency factors.

Design notes
------------
* ``jax.nn.softplus`` replaces the HL-MRF hinge ``max(0, ...)`` for smooth
  gradients throughout — essential for HMC / NUTS.
* Contradiction factors are fully vectorised (single ``numpyro.factor`` call).
* Composition factors are per-chain loops (JAX static-shape requirement means
  we cannot batch over variable-length chain lists in the flat model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompositionChain:
    """A transitive composition chain A→…→Z that implies a direct A→Z link.

    Attributes:
        chain_indices: Ordered list of edge indices (into ModelInputs.edge_ids)
            for the chain edges A→B, B→C, … (excluding the target edge).
        target_idx: Index of the direct target edge (e.g. A→Z).
        strength: Composition strength from the predicate algebra table.
        coupling: Minimum condition coupling along the chain.
        decay: Per-hop decay factor applied to the composition weight.
    """

    chain_indices: list[int]
    target_idx: int
    strength: float
    coupling: float
    decay: float


@dataclass
class ModelInputs:
    """Plain-data input struct consumed by ``kg_flat_model``.

    Attributes:
        edge_ids: Sorted list of edge identifiers.
        edge_index: Mapping from edge_id → positional index in ``edge_ids``.
        n_edges: Total number of edges.
        alpha_priors: Beta distribution α parameter per edge (shape: [n_edges]).
        beta_priors: Beta distribution β parameter per edge (shape: [n_edges]).
        contra_a_idx: Row indices for contradiction pair first edges.
        contra_b_idx: Row indices for contradiction pair second edges.
        contra_coupling: Condition coupling per contradiction pair.
        composition_chains: List of composition chains.
    """

    edge_ids: list[str]
    edge_index: dict[str, int]
    n_edges: int
    alpha_priors: np.ndarray
    beta_priors: np.ndarray
    contra_a_idx: np.ndarray
    contra_b_idx: np.ndarray
    contra_coupling: np.ndarray
    composition_chains: list[CompositionChain] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_beta_params(data: dict[str, Any]) -> tuple[float, float]:
    """Derive Beta(alpha, beta) parameters from edge attributes.

    Uses the mean-kappa parameterisation:
        kappa = 2 + evidence_count   (pseudo-count offset keeps kappa >= 3)
        alpha = mean * kappa
        beta  = (1 - mean) * kappa

    Args:
        data: Edge attribute dict containing optional ``confidence_mean`` and
            ``evidence_count`` keys.

    Returns:
        ``(alpha, beta)`` clamped to a minimum of 0.1 each.
    """
    mean = float(data.get("confidence_mean", 0.5))
    mean = max(0.01, min(0.99, mean))
    evidence_count = max(1, int(data.get("evidence_count", 1)))
    kappa = 2.0 + evidence_count
    alpha = mean * kappa
    beta_param = (1.0 - mean) * kappa
    return max(0.1, alpha), max(0.1, beta_param)


def _discover_composition_chains(
    edge_ids: list[str],
    edge_data: dict[str, dict[str, Any]],
    edge_index: dict[str, int],
    max_hops: int,
    composition_decay: float,
) -> list[CompositionChain]:
    """BFS discovery of transitive composition chains in the edge graph.

    Mirrors the algorithm in ``mrf_scoring._ground_rules`` Step 5 but returns
    ``CompositionChain`` dataclasses instead of ``GroundRule`` objects.

    Args:
        edge_ids: All edge identifiers.
        edge_data: Mapping from edge_id to attribute dict (including ``_u``,
            ``_v`` injected by ``prepare_model_inputs``).
        edge_index: Mapping from edge_id → positional index.
        max_hops: Maximum chain length (BFS depth cap).
        composition_decay: Per-hop decay multiplier.

    Returns:
        List of :class:`CompositionChain` instances.
    """
    from autoreview.knowledge_graph.condition_compat import (
        ConditionVector,
        condition_coupling,
    )
    from autoreview.knowledge_graph.predicate_algebra import compose_predicates

    # outgoing[node] = [(edge_id, next_node, predicate, data), ...]
    outgoing: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
    for eid in edge_ids:
        data = edge_data[eid]
        u = data["_u"]
        v = data["_v"]
        pred = data.get("predicate", "")
        outgoing.setdefault(u, []).append((eid, v, pred, data))

    # direct_edges[(start, end)] = [edge_id, ...]
    direct_edges: dict[tuple[str, str], list[str]] = {}
    for eid in edge_ids:
        data = edge_data[eid]
        pair = (data["_u"], data["_v"])
        direct_edges.setdefault(pair, []).append(eid)

    chains: list[CompositionChain] = []

    for node_a, first_edges in outgoing.items():
        for first_eid, node_b, pred_first, data_first in first_edges:
            # stack entries: (cur_node, visited, chain_eids, chain_datas,
            #                 composed_result, last_pred, depth)
            stack: list[
                tuple[
                    str,
                    set[str],
                    list[str],
                    list[dict[str, Any]],
                    Any,  # CompositionResult | None
                    str,
                    int,
                ]
            ] = [
                (
                    node_b,
                    {node_a, node_b},
                    [first_eid],
                    [data_first],
                    None,
                    pred_first,
                    1,
                )
            ]

            while stack:
                (
                    cur_node,
                    visited,
                    chain_eids,
                    chain_datas,
                    composed_result,
                    last_pred,
                    depth,
                ) = stack.pop()

                if depth >= max_hops:
                    continue

                for next_eid, next_node, next_pred, next_data in outgoing.get(cur_node, []):
                    if next_node in visited:
                        continue

                    if composed_result is None:
                        new_composition = compose_predicates(last_pred, next_pred)
                    else:
                        new_composition = compose_predicates(
                            composed_result.composed_predicate,
                            next_pred,
                        )

                    if new_composition is None:
                        continue

                    new_chain_eids = chain_eids + [next_eid]
                    new_chain_datas = chain_datas + [next_data]
                    new_visited = visited | {next_node}

                    # Is there a direct edge from node_a to next_node?
                    target_candidates = direct_edges.get((node_a, next_node), [])
                    if target_candidates:
                        n_hops = len(new_chain_eids)
                        decay_factor = composition_decay ** (n_hops - 1)

                        # Minimum condition coupling along adjacent chain edges
                        min_coupling = 1.0
                        for i in range(len(new_chain_datas) - 1):
                            cv_i = ConditionVector.from_edge_data(new_chain_datas[i])
                            cv_j = ConditionVector.from_edge_data(new_chain_datas[i + 1])
                            min_coupling = min(min_coupling, condition_coupling(cv_i, cv_j))

                        for target_eid in target_candidates:
                            chains.append(
                                CompositionChain(
                                    chain_indices=[edge_index[e] for e in new_chain_eids],
                                    target_idx=edge_index[target_eid],
                                    strength=new_composition.strength,
                                    coupling=min_coupling,
                                    decay=decay_factor,
                                )
                            )

                    stack.append(
                        (
                            next_node,
                            new_visited,
                            new_chain_eids,
                            new_chain_datas,
                            new_composition,
                            next_pred,
                            depth + 1,
                        )
                    )

    return chains


# ---------------------------------------------------------------------------
# Public: input preparation
# ---------------------------------------------------------------------------


def prepare_model_inputs(
    graph: nx.MultiDiGraph,
    config: BayesianConfig,
) -> ModelInputs:
    """Convert a knowledge graph into ``ModelInputs`` for the NumPyro model.

    Steps:
    1. Collect all edges; enrich with ``_u`` / ``_v`` / ``_key`` fields.
    2. Sort edge_ids and build edge_index.
    3. Derive Beta(alpha, beta) priors per edge from confidence_mean +
       evidence_count.
    4. Detect contradictions (filtering out BOUNDARY_CONDITION type).
    5. Discover composition chains via BFS.

    Args:
        graph: NetworkX MultiDiGraph with edge attributes including at minimum
            ``edge_id``, ``predicate``, ``confidence_mean``, ``evidence_count``.
        config: :class:`BayesianConfig` controlling chain discovery parameters
            and contradiction threshold.

    Returns:
        A fully populated :class:`ModelInputs` struct.
    """
    from autoreview.knowledge_graph.structural_contradictions import (
        ContradictionType,
        detect_contradictions,
    )

    # ------------------------------------------------------------------
    # Step 1 & 2: collect edges, sort, build index
    # ------------------------------------------------------------------
    raw_edge_data: dict[str, dict[str, Any]] = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        eid: str = data.get("edge_id", f"{u}__{key}__{v}")
        enriched = dict(data)
        enriched["_u"] = u
        enriched["_v"] = v
        enriched["_key"] = key
        raw_edge_data[eid] = enriched

    edge_ids: list[str] = sorted(raw_edge_data.keys())
    edge_index: dict[str, int] = {eid: i for i, eid in enumerate(edge_ids)}
    n_edges = len(edge_ids)

    # ------------------------------------------------------------------
    # Step 3: Beta priors
    # ------------------------------------------------------------------
    alpha_priors = np.empty(n_edges, dtype=np.float64)
    beta_priors = np.empty(n_edges, dtype=np.float64)
    for eid in edge_ids:
        idx = edge_index[eid]
        alpha_priors[idx], beta_priors[idx] = _derive_beta_params(raw_edge_data[eid])

    # ------------------------------------------------------------------
    # Step 4: Contradiction detection
    # ------------------------------------------------------------------
    contradictions = detect_contradictions(graph, config.boundary_threshold)

    contra_a: list[int] = []
    contra_b: list[int] = []
    contra_coup: list[float] = []

    for cp in contradictions:
        if cp.contradiction_type == ContradictionType.BOUNDARY_CONDITION:
            continue
        # Both edge IDs must be known; skip if not found (defensive)
        if cp.edge_a_id not in edge_index or cp.edge_b_id not in edge_index:
            log.warning(
                "prepare_model_inputs: contradiction edge not in graph, skipping",
                edge_a=cp.edge_a_id,
                edge_b=cp.edge_b_id,
            )
            continue
        contra_a.append(edge_index[cp.edge_a_id])
        contra_b.append(edge_index[cp.edge_b_id])
        contra_coup.append(cp.condition_coupling)

    contra_a_idx = np.array(contra_a, dtype=np.intp)
    contra_b_idx = np.array(contra_b, dtype=np.intp)
    contra_coupling = np.array(contra_coup, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 5: Composition chain discovery
    # ------------------------------------------------------------------
    composition_chains = _discover_composition_chains(
        edge_ids=edge_ids,
        edge_data=raw_edge_data,
        edge_index=edge_index,
        max_hops=config.max_composition_hops,
        composition_decay=config.composition_decay,
    )

    log.debug(
        "prepare_model_inputs done",
        n_edges=n_edges,
        n_contradictions=len(contra_a),
        n_chains=len(composition_chains),
    )

    return ModelInputs(
        edge_ids=edge_ids,
        edge_index=edge_index,
        n_edges=n_edges,
        alpha_priors=alpha_priors,
        beta_priors=beta_priors,
        contra_a_idx=contra_a_idx,
        contra_b_idx=contra_b_idx,
        contra_coupling=contra_coupling,
        composition_chains=composition_chains,
    )


# ---------------------------------------------------------------------------
# Public: NumPyro model
# ---------------------------------------------------------------------------


def kg_flat_model(
    inputs: ModelInputs,
    w_contradiction: float = 8.0,
    w_composition: float = 5.0,
) -> None:
    """NumPyro flat factor graph model for KG edge truth values.

    Probabilistic program with three layers:

    1. **Beta priors** — per-edge truth values sampled from Beta(alpha, beta)
       distributions derived from confidence_mean and evidence_count.
    2. **Contradiction factors** — vectorised soft penalty when two opposing
       edges are both assigned high truth values.  Uses ``softplus`` for smooth
       gradients (HMC-friendly alternative to the HL-MRF hinge).
    3. **Composition factors** — per-chain soft consistency: if A→B and B→C
       are both true, the implied A→C edge should also be true.

    Args:
        inputs: Pre-computed :class:`ModelInputs` struct.
        w_contradiction: Global contradiction penalty weight.
        w_composition: Global composition consistency weight.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # ------------------------------------------------------------------
    # Layer 1: Beta priors on edge truth values
    # ------------------------------------------------------------------
    truth = numpyro.sample(
        "truth",
        dist.Beta(
            jnp.array(inputs.alpha_priors),
            jnp.array(inputs.beta_priors),
        ),
    )

    # ------------------------------------------------------------------
    # Layer 2: Vectorised contradiction factors
    # ------------------------------------------------------------------
    if len(inputs.contra_a_idx) > 0:
        a_vals = truth[inputs.contra_a_idx]
        b_vals = truth[inputs.contra_b_idx]
        # Penalty is large when both edges are simultaneously "true" (sum > 1)
        penalties = jax.nn.softplus(a_vals + b_vals - 1.0)
        coupling = jnp.array(inputs.contra_coupling)
        total_contra = jnp.sum(w_contradiction * coupling * penalties)
        numpyro.factor("contradictions", -total_contra)

    # ------------------------------------------------------------------
    # Layer 3: Per-chain composition factors
    # ------------------------------------------------------------------
    for i, chain in enumerate(inputs.composition_chains):
        chain_vals = truth[jnp.array(chain.chain_indices)]
        chain_min = jnp.min(chain_vals)
        # Expected truth of target given chain minimum and composition strength
        expected = chain.strength * chain_min
        # Penalise shortfall: target truth below expected
        shortfall = jax.nn.softplus(expected - truth[chain.target_idx])
        numpyro.factor(
            f"comp_{i}",
            -w_composition * chain.coupling * chain.decay * shortfall,
        )
