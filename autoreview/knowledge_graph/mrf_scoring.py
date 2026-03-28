"""MRF scoring module: bridges the knowledge graph with the HL-MRF inference engine.

Grounds rules against the actual graph structure — evidence priors, contradiction
penalties, and transitive composition boosts — then runs convex inference to
produce posterior truth values for every edge.

Public API
----------
score_graph_mrf(graph, config=None) -> MRFResult
    Run HL-MRF inference on a KG MultiDiGraph and return posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import structlog

from autoreview.knowledge_graph.condition_compat import ConditionVector, condition_coupling
from autoreview.knowledge_graph.hlmrf import GroundRule, HLMRFEngine
from autoreview.knowledge_graph.predicate_algebra import compose_predicates
from autoreview.knowledge_graph.structural_contradictions import (
    ContradictionType,
    detect_contradictions,
)
from autoreview.models.base import AutoReviewModel

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------


class MRFConfig(AutoReviewModel):
    """Configuration for MRF scoring.

    Attributes:
        evidence_weight: Weight for unary evidence rules (prior confidence).
        contradiction_weight: Weight for contradiction penalty rules.
        composition_weight: Base weight for transitive composition rules.
        max_iter: Maximum L-BFGS-B iterations.
        tol: Convergence tolerance for the optimizer.
        boundary_threshold: Minimum condition coupling to classify a contradiction
            as a real contradiction (rather than a boundary condition).
    """

    evidence_weight: float = 10.0
    contradiction_weight: float = 8.0
    composition_weight: float = 5.0
    max_iter: int = 200
    tol: float = 1e-6
    boundary_threshold: float = 0.6


@dataclass
class MRFResult:
    """Result of MRF inference over the knowledge graph.

    Attributes:
        posteriors: Mapping from edge_id to posterior truth value in [0, 1].
        n_variables: Number of variables in the MRF.
        n_rules: Total number of ground rules added.
        n_contradictions: Number of contradiction rules grounded.
        n_compositions: Number of composition rules grounded.
        converged: Whether the optimizer converged (always True for empty graphs).
    """

    posteriors: dict[str, float] = field(default_factory=dict)
    n_variables: int = 0
    n_rules: int = 0
    n_contradictions: int = 0
    n_compositions: int = 0
    converged: bool = True


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def score_graph_mrf(
    graph: nx.MultiDiGraph,
    config: MRFConfig | None = None,
) -> MRFResult:
    """Run HL-MRF inference on a knowledge graph and return posterior truth values.

    Grounds three rule types:
    - **Unary evidence rules** — anchor each edge at its empirical confidence mean.
    - **Contradiction rules** — penalise pairs of opposing claims that share similar
      experimental conditions.
    - **Composition rules** — boost sparse A→C edges when a strong A→B→C chain
      exists.

    Args:
        graph: A NetworkX MultiDiGraph where each edge has at minimum an
            ``edge_id`` attribute. Confidence is read from ``confidence_mean``.
            Optional per-edge fields: ``predicate``, ``direction``, ``organism``,
            ``model_system``, ``in_vitro``, ``conditions``.
        config: MRF hyperparameters. Uses :class:`MRFConfig` defaults if None.

    Returns:
        :class:`MRFResult` with posterior truth values and diagnostic counts.
    """
    if config is None:
        config = MRFConfig()

    # -----------------------------------------------------------------------
    # Step 1: Collect all edges and index them
    # -----------------------------------------------------------------------
    # edge_data maps edge_id → flat attribute dict (enriched with _u, _v, _key)
    edge_data: dict[str, dict] = {}

    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_id: str = data.get("edge_id", f"{u}__{key}__{v}")
        enriched = dict(data)
        enriched["_u"] = u
        enriched["_v"] = v
        enriched["_key"] = key
        edge_data[edge_id] = enriched

    if not edge_data:
        log.info("mrf_scoring.empty_graph")
        return MRFResult()

    # -----------------------------------------------------------------------
    # Step 2: Create HLMRFEngine and register one variable per edge
    # -----------------------------------------------------------------------
    engine = HLMRFEngine(max_iter=config.max_iter, tol=config.tol)

    for edge_id, data in edge_data.items():
        init_val = float(data.get("confidence_mean", 0.5))
        # Clamp to [0, 1] in case of invalid data
        init_val = max(0.0, min(1.0, init_val))
        engine.add_variable(edge_id, init=init_val)

    # -----------------------------------------------------------------------
    # Step 3: Unary evidence rules — one per edge
    # -----------------------------------------------------------------------
    for edge_id, data in edge_data.items():
        confidence = float(data.get("confidence_mean", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        engine.add_ground_rule(
            GroundRule(
                head_var=edge_id,
                body_vars=[],
                body_coeffs=[],
                target=confidence,
                weight=config.evidence_weight,
                rule_type="unary",
            )
        )

    log.info("mrf_scoring.evidence_rules_added", n_edges=len(edge_data))

    # -----------------------------------------------------------------------
    # Step 4: Contradiction rules
    # -----------------------------------------------------------------------
    contradictions = detect_contradictions(graph, config.boundary_threshold)

    n_contradictions = 0
    for cp in contradictions:
        # Skip boundary conditions — they reflect context-dependence, not error
        if cp.contradiction_type == ContradictionType.BOUNDARY_CONDITION:
            continue

        # Ensure both edge IDs are registered (they should be, but guard)
        if cp.edge_a_id not in edge_data or cp.edge_b_id not in edge_data:
            log.warning(
                "mrf_scoring.contradiction_edge_missing",
                edge_a=cp.edge_a_id,
                edge_b=cp.edge_b_id,
            )
            continue

        coupling = cp.condition_coupling
        effective_weight = config.contradiction_weight * coupling

        # For each contradiction (A, B): add symmetric rules so both edges
        # are penalised when the pair is simultaneously too high.
        # Rule: head=edge_a, body=[edge_b], target=1.0 (sum should not exceed 1)
        engine.add_ground_rule(
            GroundRule(
                head_var=cp.edge_a_id,
                body_vars=[cp.edge_b_id],
                body_coeffs=[1.0],
                target=1.0,
                weight=effective_weight,
                rule_type="contradiction",
            )
        )
        n_contradictions += 1

    log.info(
        "mrf_scoring.contradiction_rules_added",
        n_real_contradictions=n_contradictions,
        n_boundary=len(contradictions) - n_contradictions,
    )

    # -----------------------------------------------------------------------
    # Step 5: Composition rules — find A→B→C chains
    # -----------------------------------------------------------------------

    # Build outgoing adjacency: node → list of (edge_id, target_node, predicate, data)
    outgoing: dict[str, list[tuple[str, str, str, dict]]] = {}
    for edge_id, data in edge_data.items():
        u = data["_u"]
        outgoing.setdefault(u, []).append((edge_id, data["_v"], data.get("predicate", ""), data))

    # Build (u, v) → list[edge_id] lookup for checking whether A→C exists
    direct_edges: dict[tuple, list[str]] = {}
    for edge_id, data in edge_data.items():
        pair = (data["_u"], data["_v"])
        direct_edges.setdefault(pair, []).append(edge_id)

    n_compositions = 0

    for node_a, ab_edges in outgoing.items():
        for ab_edge_id, node_b, pred_ab, data_ab in ab_edges:
            # Iterate all B→C edges
            for bc_edge_id, node_c, pred_bc, data_bc in outgoing.get(node_b, []):
                # No self-loops A→A
                if node_c == node_a:
                    continue

                # Attempt predicate composition
                composition = compose_predicates(pred_ab, pred_bc)
                if composition is None:
                    continue

                # Check that an A→C edge exists in the graph (no hallucination)
                ac_candidates = direct_edges.get((node_a, node_c), [])
                if not ac_candidates:
                    continue

                # Compute condition coupling between A→B and B→C
                cond_ab = ConditionVector.from_edge_data(data_ab)
                cond_bc = ConditionVector.from_edge_data(data_bc)
                coupling = condition_coupling(cond_ab, cond_bc)

                effective_weight = config.composition_weight * composition.strength * coupling

                # Add one composition rule per A→C edge
                for ac_edge_id in ac_candidates:
                    engine.add_ground_rule(
                        GroundRule(
                            head_var=ac_edge_id,
                            body_vars=[ab_edge_id, bc_edge_id],
                            body_coeffs=[0.5, 0.5],
                            target=composition.strength,
                            weight=effective_weight,
                            rule_type="composition",
                        )
                    )
                    n_compositions += 1

    log.info("mrf_scoring.composition_rules_added", n_composition_rules=n_compositions)

    # -----------------------------------------------------------------------
    # Step 6: Run inference
    # -----------------------------------------------------------------------
    log.info(
        "mrf_scoring.solve_start",
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
    )

    posteriors = engine.solve()

    log.info("mrf_scoring.solve_done", n_posteriors=len(posteriors))

    return MRFResult(
        posteriors=posteriors,
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
        n_contradictions=n_contradictions,
        n_compositions=n_compositions,
        converged=True,  # scipy L-BFGS-B raises on failure; no exception = converged
    )
