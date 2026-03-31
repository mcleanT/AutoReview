"""MRF scoring module: bridges the knowledge graph with the HL-MRF inference engine.

Grounds rules against the actual graph structure — evidence priors, contradiction
penalties, and transitive composition boosts — then runs convex inference to
produce posterior truth values for every edge.

Public API
----------
score_graph_mrf(graph, config=None) -> MRFResult
    Run HL-MRF inference on a KG MultiDiGraph and return posteriors.
update_graph_mrf(graph, new_edge_ids, prior_result, config=None) -> MRFResult
    Incrementally update MRF posteriors after adding new edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import structlog

from autoreview.knowledge_graph.cluster import (
    build_topic_clusters,
    detect_finding_contradictions,
    form_findings,
)
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
    max_composition_hops: int = 3
    composition_decay: float = 0.7
    diagnostics_top_n: int = 20

    # Finding layer
    finding_contradiction_weight: float = 12.0
    propagation_weight: float = 3.0
    enable_finding_layer: bool = True


@dataclass
class RuleViolation:
    """Diagnostic info for a single violated rule."""

    rule_index: int
    rule_type: str
    head_var: str
    violation_magnitude: float
    target: float
    actual: float


@dataclass
class MRFDiagnostics:
    """Convergence diagnostics from MRF inference."""

    converged: bool
    n_iterations: int
    final_objective: float
    top_violations: list[RuleViolation]
    mean_violation_by_type: dict[str, float]


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
        diagnostics: Per-rule violation diagnostics, or None for empty graphs.
    """

    posteriors: dict[str, float] = field(default_factory=dict)
    n_variables: int = 0
    n_rules: int = 0
    n_contradictions: int = 0
    n_compositions: int = 0
    converged: bool = True
    diagnostics: MRFDiagnostics | None = None
    finding_posteriors: dict[str, float] = field(default_factory=dict)
    n_findings: int = 0
    n_finding_contradictions: int = 0


# ---------------------------------------------------------------------------
# Private helper: rule grounding
# ---------------------------------------------------------------------------


def _ground_rules(
    graph: nx.MultiDiGraph,
    config: MRFConfig,
) -> tuple[HLMRFEngine, dict[str, dict], int, int, int, int, list]:
    """Ground all MRF rules from graph structure.

    Performs Steps 1–6 of the MRF pipeline: collect edges, register variables,
    add unary evidence rules, contradiction rules, composition rules, and
    (optionally) finding-layer rules.

    Args:
        graph: A NetworkX MultiDiGraph with edge attributes.
        config: MRF hyperparameters.

    Returns:
        Tuple of (engine, edge_data, n_contradictions, n_compositions,
        n_findings, n_finding_contradictions, all_findings).
        ``edge_data`` maps edge_id → enriched attribute dict.
        The engine is fully grounded and ready for ``solve()`` or
        ``solve_incremental()``.
    """
    # -----------------------------------------------------------------------
    # Step 1: Collect all edges and index them
    # -----------------------------------------------------------------------
    edge_data: dict[str, dict] = {}

    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_id: str = data.get("edge_id", f"{u}__{key}__{v}")
        enriched = dict(data)
        enriched["_u"] = u
        enriched["_v"] = v
        enriched["_key"] = key
        edge_data[edge_id] = enriched

    # -----------------------------------------------------------------------
    # Step 2: Create HLMRFEngine and register one variable per edge
    # -----------------------------------------------------------------------
    engine = HLMRFEngine(max_iter=config.max_iter, tol=config.tol)

    for edge_id, data in edge_data.items():
        init_val = float(data.get("confidence_mean", 0.5))
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
        if cp.contradiction_type == ContradictionType.BOUNDARY_CONDITION:
            continue

        if cp.edge_a_id not in edge_data or cp.edge_b_id not in edge_data:
            log.warning(
                "mrf_scoring.contradiction_edge_missing",
                edge_a=cp.edge_a_id,
                edge_b=cp.edge_b_id,
            )
            continue

        coupling = cp.condition_coupling
        effective_weight = config.contradiction_weight * coupling

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
    # Step 5: Composition rules — multi-hop BFS up to max_composition_hops
    # -----------------------------------------------------------------------
    outgoing: dict[str, list[tuple[str, str, str, dict]]] = {}
    for edge_id, data in edge_data.items():
        u = data["_u"]
        outgoing.setdefault(u, []).append((edge_id, data["_v"], data.get("predicate", ""), data))

    direct_edges: dict[tuple, list[str]] = {}
    for edge_id, data in edge_data.items():
        pair = (data["_u"], data["_v"])
        direct_edges.setdefault(pair, []).append(edge_id)

    n_compositions = 0

    for node_a, first_edges in outgoing.items():
        for first_edge_id, node_b, pred_first, data_first in first_edges:
            stack: list[tuple[str, set[str], list[str], list[dict], object, str, int]] = [
                (
                    node_b,
                    {node_a, node_b},
                    [first_edge_id],
                    [data_first],
                    None,
                    pred_first,
                    1,
                ),
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

                if depth >= config.max_composition_hops:
                    continue

                for next_eid, next_node, next_pred, next_data in outgoing.get(cur_node, []):
                    if next_node in visited:
                        continue

                    if composed_result is None:
                        new_composition = compose_predicates(last_pred, next_pred)
                    else:
                        new_composition = compose_predicates(
                            composed_result.composed_predicate,  # type: ignore[attr-defined]
                            next_pred,
                        )

                    if new_composition is None:
                        continue

                    new_chain_eids = chain_eids + [next_eid]
                    new_chain_datas = chain_datas + [next_data]
                    new_visited = visited | {next_node}

                    target_candidates = direct_edges.get((node_a, next_node), [])
                    if target_candidates:
                        n_hops = len(new_chain_eids)
                        decay_factor = config.composition_decay ** (n_hops - 1)

                        min_coupling = 1.0
                        for i in range(len(new_chain_datas) - 1):
                            cv_i = ConditionVector.from_edge_data(new_chain_datas[i])
                            cv_j = ConditionVector.from_edge_data(new_chain_datas[i + 1])
                            min_coupling = min(min_coupling, condition_coupling(cv_i, cv_j))

                        effective_weight = (
                            config.composition_weight
                            * new_composition.strength
                            * min_coupling
                            * decay_factor
                        )

                        body_coeffs = [1.0 / n_hops] * n_hops

                        for target_eid in target_candidates:
                            engine.add_ground_rule(
                                GroundRule(
                                    head_var=target_eid,
                                    body_vars=list(new_chain_eids),
                                    body_coeffs=body_coeffs,
                                    target=new_composition.strength,
                                    weight=effective_weight,
                                    rule_type="composition",
                                )
                            )
                            n_compositions += 1

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

    log.info("mrf_scoring.composition_rules_added", n_composition_rules=n_compositions)

    # -----------------------------------------------------------------------
    # Step 6: Finding layer (optional)
    # -----------------------------------------------------------------------
    n_findings = 0
    n_finding_contradictions = 0
    all_findings: list = []

    if config.enable_finding_layer:
        clusters = build_topic_clusters(graph)
        all_findings = form_findings(clusters, graph)
        finding_contras = detect_finding_contradictions(
            all_findings,
            clusters,
            graph=graph,
        )

        n_findings = len(all_findings)
        n_finding_contradictions = len(finding_contras)

        # Register finding variables
        for finding in all_findings:
            var_name = f"finding:{finding.finding_id}"
            engine.add_variable(var_name, init=finding.confidence.mean)

        # Rule type 1: Upward aggregation (edge → finding)
        for finding in all_findings:
            member_ids = [eid for eid in finding.member_edge_ids if eid in edge_data]
            if not member_ids:
                continue
            n_members = len(member_ids)
            coeffs = [1.0 / n_members] * n_members
            var_name = f"finding:{finding.finding_id}"
            engine.add_ground_rule(
                GroundRule(
                    head_var=var_name,
                    body_vars=member_ids,
                    body_coeffs=coeffs,
                    target=0.0,
                    weight=config.evidence_weight,
                    rule_type="aggregation",
                )
            )

        # Rule type 2: Finding contradiction (finding ↔ finding)
        for fc in finding_contras:
            if fc.contradiction_type == "boundary":
                continue
            var_a = f"finding:{fc.finding_a_id}"
            var_b = f"finding:{fc.finding_b_id}"
            effective_weight = config.finding_contradiction_weight * fc.severity
            engine.add_ground_rule(
                GroundRule(
                    head_var=var_a,
                    body_vars=[var_b],
                    body_coeffs=[1.0],
                    target=1.0,
                    weight=effective_weight,
                    rule_type="contradiction",
                )
            )

        # Rule type 3: Downward propagation (finding → edge)
        # head_var=eid so the edge is penalised when edge > finding, pulling
        # member edges DOWN when their finding's posterior drops.
        for finding in all_findings:
            var_name = f"finding:{finding.finding_id}"
            for eid in finding.member_edge_ids:
                if eid not in edge_data:
                    continue
                engine.add_ground_rule(
                    GroundRule(
                        head_var=eid,
                        body_vars=[var_name],
                        body_coeffs=[-1.0],
                        target=0.0,
                        weight=config.propagation_weight,
                        rule_type="contradiction",
                    )
                )

        log.info(
            "mrf_scoring.finding_layer_added",
            n_findings=n_findings,
            n_finding_contradictions=n_finding_contradictions,
        )

    return (
        engine,
        edge_data,
        n_contradictions,
        n_compositions,
        n_findings,
        n_finding_contradictions,
        all_findings,
    )


def _build_diagnostics(
    engine: HLMRFEngine,
    posteriors: dict[str, float],
    config: MRFConfig,
) -> MRFDiagnostics:
    """Build MRFDiagnostics from a solved engine and its posteriors.

    Args:
        engine: Solved HLMRFEngine (convergence metadata must be populated).
        posteriors: Variable name -> posterior value mapping from solve().
        config: MRF configuration (for diagnostics_top_n).

    Returns:
        Populated MRFDiagnostics instance.
    """
    raw_diag = engine.compute_diagnostics(posteriors)

    violations = sorted(
        [
            RuleViolation(
                rule_index=d["rule_index"],
                rule_type=d["rule_type"],
                head_var=d["head_var"],
                violation_magnitude=d["violation"],
                target=d["target"],
                actual=d["actual"],
            )
            for d in raw_diag
        ],
        key=lambda v: v.violation_magnitude,
        reverse=True,
    )

    type_violations: dict[str, list[float]] = {}
    for d in raw_diag:
        type_violations.setdefault(d["rule_type"], []).append(d["violation"])
    mean_by_type = {rt: sum(vs) / len(vs) if vs else 0.0 for rt, vs in type_violations.items()}

    return MRFDiagnostics(
        converged=engine.last_converged,
        n_iterations=engine.last_n_iterations,
        final_objective=engine.last_final_objective,
        top_violations=violations[: config.diagnostics_top_n],
        mean_violation_by_type=mean_by_type,
    )


# ---------------------------------------------------------------------------
# Main functions
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

    (
        engine,
        edge_data,
        n_contradictions,
        n_compositions,
        n_findings,
        n_finding_contradictions,
        all_findings,
    ) = _ground_rules(graph, config)

    if not edge_data:
        log.info("mrf_scoring.empty_graph")
        return MRFResult()

    log.info(
        "mrf_scoring.solve_start",
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
    )

    posteriors = engine.solve()
    diagnostics = _build_diagnostics(engine, posteriors, config)

    log.info("mrf_scoring.solve_done", n_posteriors=len(posteriors))

    # Extract finding posteriors (separate from edge posteriors)
    finding_posteriors: dict[str, float] = {}
    for finding in all_findings:
        var_name = f"finding:{finding.finding_id}"
        if var_name in posteriors:
            finding_posteriors[finding.finding_id] = posteriors[var_name]

    return MRFResult(
        posteriors={k: v for k, v in posteriors.items() if not k.startswith("finding:")},
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
        n_contradictions=n_contradictions,
        n_compositions=n_compositions,
        converged=engine.last_converged,
        diagnostics=diagnostics,
        finding_posteriors=finding_posteriors,
        n_findings=n_findings,
        n_finding_contradictions=n_finding_contradictions,
    )


def update_graph_mrf(
    graph: nx.MultiDiGraph,
    new_edge_ids: list[str],
    prior_result: MRFResult,
    config: MRFConfig | None = None,
) -> MRFResult:
    """Incrementally update MRF posteriors after adding new edges.

    Re-grounds all rules (to account for new edges in contradictions and
    compositions) but warm-starts the solver from the prior solution and
    only optimizes variables near the new edges.

    Args:
        graph: Updated graph (with new edges already added).
        new_edge_ids: Edge IDs that are new since the prior solve.
        prior_result: Result from the previous score_graph_mrf call.
        config: MRF configuration.

    Returns:
        Updated MRFResult.
    """
    if config is None:
        config = MRFConfig()

    (
        engine,
        edge_data,
        n_contradictions,
        n_compositions,
        n_findings,
        n_finding_contradictions,
        all_findings,
    ) = _ground_rules(graph, config)

    if not edge_data:
        return MRFResult()

    # Merge edge + finding posteriors so solve_incremental can warm-start
    # finding variables from the prior solution (they are stripped from
    # prior_result.posteriors by score_graph_mrf).
    prior_solution = dict(prior_result.posteriors)
    for fid, val in prior_result.finding_posteriors.items():
        prior_solution[f"finding:{fid}"] = val

    posteriors = engine.solve_incremental(
        prior_solution=prior_solution,
        affected_vars=set(new_edge_ids),
        hop_radius=2,
    )

    diagnostics = _build_diagnostics(engine, posteriors, config)

    log.info(
        "mrf_scoring.incremental_update_done",
        n_new_edges=len(new_edge_ids),
        n_posteriors=len(posteriors),
    )

    # Extract finding posteriors (separate from edge posteriors)
    finding_posteriors: dict[str, float] = {}
    for finding in all_findings:
        var_name = f"finding:{finding.finding_id}"
        if var_name in posteriors:
            finding_posteriors[finding.finding_id] = posteriors[var_name]

    return MRFResult(
        posteriors={k: v for k, v in posteriors.items() if not k.startswith("finding:")},
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
        n_contradictions=n_contradictions,
        n_compositions=n_compositions,
        converged=engine.last_converged,
        diagnostics=diagnostics,
        finding_posteriors=finding_posteriors,
        n_findings=n_findings,
        n_finding_contradictions=n_finding_contradictions,
    )
