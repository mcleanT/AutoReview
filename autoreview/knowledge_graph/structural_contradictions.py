"""Structural contradiction detection for the knowledge graph.

Detects contradictions purely from graph structure — no NLI model needed.
Uses predicate opposition + direction comparison + condition disambiguation.

The key insight: with structured triples (controlled predicates + explicit
direction), contradictions are detectable from graph topology alone.

When conditions differ between contradicting claims, it's a "boundary
condition" — the claim's validity changes based on experimental context.
This is scientifically informative, not an error.

Public API
----------
detect_contradictions(graph, boundary_threshold) -> list[ContradictionPair]
    Detect all structural contradictions in a KG MultiDiGraph.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.condition_compat import (
    ConditionVector,
    condition_coupling,
)
from autoreview.knowledge_graph.predicate_algebra import are_opposing

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ContradictionType(StrEnum):
    PREDICATE_OPPOSITION = "predicate_opposition"
    DIRECTION_CONFLICT = "direction_conflict"
    BOUNDARY_CONDITION = "boundary_condition"


@dataclass
class ContradictionPair:
    """A detected contradiction between two edges in the knowledge graph.

    Attributes:
        edge_a_id: Identifier of the first edge.
        edge_b_id: Identifier of the second edge.
        subject: The subject node shared by both edges.
        object: The object node shared by both edges.
        predicate_a: Predicate of the first edge.
        predicate_b: Predicate of the second edge.
        direction_a: Direction field of the first edge (or None).
        direction_b: Direction field of the second edge (or None).
        contradiction_type: Classification of the contradiction.
        condition_coupling: Condition compatibility score in [0, 1].
        condition_a: Experimental conditions of the first edge.
        condition_b: Experimental conditions of the second edge.
    """

    edge_a_id: str
    edge_b_id: str
    subject: str
    object: str
    predicate_a: str
    predicate_b: str
    direction_a: str | None
    direction_b: str | None
    contradiction_type: ContradictionType
    condition_coupling: float
    condition_a: ConditionVector
    condition_b: ConditionVector


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_contradictions(
    graph: nx.MultiDiGraph,
    boundary_threshold: float = 0.6,
) -> list[ContradictionPair]:
    """Detect contradictions purely from graph structure.

    Algorithm:
    1. Group all edges by (subject_node, object_node) pair.
    2. For each pair of edges in the same group:
       - Check 1: are_opposing(pred_a, pred_b) → predicate opposition
       - Check 2: same predicate + opposite direction field → direction conflict
       - If neither → skip.
    3. Compute condition_coupling(cond_a, cond_b) for detected contradictions.
    4. If coupling < boundary_threshold → BOUNDARY_CONDITION.
    5. Else → PREDICATE_OPPOSITION or DIRECTION_CONFLICT.

    Args:
        graph: A NetworkX MultiDiGraph where each edge has at minimum a
            ``predicate`` and ``edge_id`` attribute. Optional attributes:
            ``direction``, ``organism``, ``model_system``, ``in_vitro``,
            ``conditions``.
        boundary_threshold: Minimum condition coupling for a contradiction to
            be classified as a "true" contradiction rather than a boundary
            condition. Claims with coupling < threshold come from sufficiently
            different experimental contexts that the disagreement reflects
            context-dependency, not error.

    Returns:
        List of :class:`ContradictionPair` instances, one per detected
        contradiction. Order is deterministic (sorted by edge_a_id, edge_b_id).
    """
    # -----------------------------------------------------------------------
    # Step 1: group edges by (subject, object) pair
    # -----------------------------------------------------------------------
    groups: dict[tuple[Any, Any], list[tuple[Any, Any, Any, dict]]] = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        pair = (u, v)
        groups.setdefault(pair, []).append((u, v, key, data))

    contradictions: list[ContradictionPair] = []

    # -----------------------------------------------------------------------
    # Step 2: compare all edge pairs within each group
    # -----------------------------------------------------------------------
    for (subject, obj), edges in groups.items():
        for (u_a, v_a, key_a, data_a), (u_b, v_b, key_b, data_b) in itertools.combinations(
            edges, 2
        ):
            pred_a: str = data_a.get("predicate", "")
            pred_b: str = data_b.get("predicate", "")
            dir_a: str | None = data_a.get("direction")
            dir_b: str | None = data_b.get("direction")
            edge_id_a: str = data_a.get("edge_id", f"{u_a}_{key_a}_{v_a}")
            edge_id_b: str = data_b.get("edge_id", f"{u_b}_{key_b}_{v_b}")

            # Check 1: predicate opposition
            predicate_opposes = are_opposing(pred_a, pred_b)

            # Check 2: same predicate, opposite direction
            direction_conflicts = (
                not predicate_opposes
                and pred_a == pred_b
                and dir_a is not None
                and dir_b is not None
                and dir_a != dir_b
            )

            if not predicate_opposes and not direction_conflicts:
                continue

            # ---------------------------------------------------------------
            # Step 3: compute condition coupling
            # ---------------------------------------------------------------
            cond_a = ConditionVector.from_edge_data(data_a)
            cond_b = ConditionVector.from_edge_data(data_b)
            coupling = condition_coupling(cond_a, cond_b)

            # ---------------------------------------------------------------
            # Step 4 / 5: classify
            # ---------------------------------------------------------------
            if coupling < boundary_threshold:
                c_type = ContradictionType.BOUNDARY_CONDITION
            elif predicate_opposes:
                c_type = ContradictionType.PREDICATE_OPPOSITION
            else:
                c_type = ContradictionType.DIRECTION_CONFLICT

            # Normalize so edge_a_id <= edge_b_id for deterministic ordering
            if edge_id_a <= edge_id_b:
                first_id, second_id = edge_id_a, edge_id_b
                first_pred, second_pred = pred_a, pred_b
                first_dir, second_dir = dir_a, dir_b
                first_cond, second_cond = cond_a, cond_b
            else:
                first_id, second_id = edge_id_b, edge_id_a
                first_pred, second_pred = pred_b, pred_a
                first_dir, second_dir = dir_b, dir_a
                first_cond, second_cond = cond_b, cond_a

            contradictions.append(
                ContradictionPair(
                    edge_a_id=first_id,
                    edge_b_id=second_id,
                    subject=str(subject),
                    object=str(obj),
                    predicate_a=first_pred,
                    predicate_b=second_pred,
                    direction_a=first_dir,
                    direction_b=second_dir,
                    contradiction_type=c_type,
                    condition_coupling=coupling,
                    condition_a=first_cond,
                    condition_b=second_cond,
                )
            )

    # Deterministic ordering
    contradictions.sort(key=lambda c: (c.edge_a_id, c.edge_b_id))

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    n_total = len(contradictions)
    n_boundary = sum(
        1 for c in contradictions if c.contradiction_type == ContradictionType.BOUNDARY_CONDITION
    )
    n_real = n_total - n_boundary
    log.info(
        "structural_contradictions.detected",
        total=n_total,
        real=n_real,
        boundary=n_boundary,
        boundary_threshold=boundary_threshold,
    )

    return contradictions
