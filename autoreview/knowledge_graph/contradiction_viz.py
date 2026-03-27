"""Compute structured contradiction data from NLI-scored graphs for the interactive HTML viewer."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.nli import NLIPairResult
from autoreview.models.base import AutoReviewModel

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ContradictionEdge(AutoReviewModel):
    claim_a_id: str
    claim_b_id: str
    p_contradiction: float
    method: str  # "nli", "predicate_opposition", "parallel_skip"
    is_cross_paper: bool
    shared_entities: list[str]
    papers_a: list[str]
    papers_b: list[str]


class CommunityLabel(AutoReviewModel):
    community_id: int
    auto_label: str
    top_predicates: list[str]
    top_entities: list[str]
    claim_count: int
    subfield: str | None = None  # LLM-assigned or heuristic subfield label
    top_entity_types: list[str] = []  # e.g. ["protein", "pathway", "biological_process"]


class CommunityDisagreement(AutoReviewModel):
    comm_a_id: int
    comm_b_id: int
    comm_a_label: str
    comm_b_label: str
    raw_score: float
    normalized_score: float
    num_contradictions: int
    top_claims: list[dict]  # up to 5: claim_a_text, claim_b_text, p_contradiction


class ContradictionVizData(AutoReviewModel):
    contradiction_edges: list[ContradictionEdge]
    community_labels: dict[int, CommunityLabel]
    community_disagreements: list[CommunityDisagreement]
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Edge index helper
# ---------------------------------------------------------------------------


def _build_edge_index(
    graph: nx.MultiDiGraph,
) -> dict[str, dict[str, Any]]:
    """Build a mapping from claim_id string to edge attribute dict."""
    index: dict[str, dict[str, Any]] = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        predicate = data.get("predicate", "")
        claim_id = f"{u}__{predicate}__{v}__{k}"
        index[claim_id] = {"u": u, "v": v, "k": k, "data": data}
    return index


def _papers_for_claim(edge_entry: dict[str, Any]) -> list[str]:
    """Extract paper IDs from a claim's KGEdge evidence_links."""
    data = edge_entry["data"]
    kg_edge = data.get("_kg_edge")
    if kg_edge is None:
        return []
    return [link.paper_id for link in kg_edge.evidence_links]


def _claim_text(u: str, predicate: str, v: str, graph: nx.MultiDiGraph) -> str:
    """Produce a human-readable label for a claim."""
    u_name = graph.nodes[u].get("canonical_name", u) if graph.has_node(u) else u
    v_name = graph.nodes[v].get("canonical_name", v) if graph.has_node(v) else v
    return f"{u_name} {predicate} {v_name}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_contradiction_edges(
    graph: nx.MultiDiGraph,
    pair_results: list[NLIPairResult],
    threshold: float = 0.8,
) -> list[ContradictionEdge]:
    """Filter NLI pair results to contradictions and enrich with paper provenance."""
    edge_index = _build_edge_index(graph)

    candidates = [
        r for r in pair_results if r.p_contradiction >= threshold and r.method != "parallel_skip"
    ]

    log.debug(
        "contradiction_edges.filtering",
        total_pairs=len(pair_results),
        above_threshold=len(candidates),
        threshold=threshold,
    )

    edges: list[ContradictionEdge] = []
    for result in candidates:
        entry_a = edge_index.get(result.claim_a_id)
        entry_b = edge_index.get(result.claim_b_id)

        papers_a = _papers_for_claim(entry_a) if entry_a else []
        papers_b = _papers_for_claim(entry_b) if entry_b else []

        is_cross_paper = (
            bool(papers_a) and bool(papers_b) and set(papers_a).isdisjoint(set(papers_b))
        )

        edges.append(
            ContradictionEdge(
                claim_a_id=result.claim_a_id,
                claim_b_id=result.claim_b_id,
                p_contradiction=result.p_contradiction,
                method=result.method,
                is_cross_paper=is_cross_paper,
                shared_entities=result.shared_entities,
                papers_a=papers_a,
                papers_b=papers_b,
            )
        )

    edges.sort(key=lambda e: e.p_contradiction, reverse=True)
    return edges


def label_communities(
    graph: nx.MultiDiGraph,
    communities: list[set[str]],
) -> dict[int, CommunityLabel]:
    """Assign human-readable labels to claim communities."""
    # Priority order for meaningful subfield categorization
    _TYPE_PRIORITY: list[str] = [
        "pathway",
        "biological_process",
        "cell_type",
        "protein",
        "gene",
        "disease",
        "chemical",
        "organism",
        "method",
        "other",
    ]

    labels: dict[int, CommunityLabel] = {}

    for community_id, claim_ids in enumerate(communities):
        predicate_counter: Counter[str] = Counter()
        entity_counter: Counter[str] = Counter()
        entity_type_counter: Counter[str] = Counter()
        # Map entity_name -> entity_type for the dominant-type lookup
        entity_type_map: dict[str, str] = {}

        for claim_id in claim_ids:
            parts = claim_id.split("__")
            if len(parts) < 4:
                continue
            u, predicate, v = parts[0], parts[1], parts[2]

            predicate_counter[predicate] += 1

            u_name = graph.nodes[u].get("canonical_name", u) if graph.has_node(u) else u
            v_name = graph.nodes[v].get("canonical_name", v) if graph.has_node(v) else v
            entity_counter[u_name] += 1
            entity_counter[v_name] += 1

            # Collect entity_type from graph node attributes
            if graph.has_node(u):
                u_type: str = graph.nodes[u].get("entity_type", "other") or "other"
                entity_type_counter[u_type] += 1
                entity_type_map.setdefault(u_name, u_type)
            if graph.has_node(v):
                v_type: str = graph.nodes[v].get("entity_type", "other") or "other"
                entity_type_counter[v_type] += 1
                entity_type_map.setdefault(v_name, v_type)

        top_predicates = [p for p, _ in predicate_counter.most_common(3)]
        top_entities = [e for e, _ in entity_counter.most_common(3)]
        top_entity_types = [t for t, _ in entity_type_counter.most_common(3)]

        # Choose a predicate that conveys meaning — skip "related_to"
        label_predicate = "related_to"
        for pred, _ in predicate_counter.most_common():
            if pred != "related_to":
                label_predicate = pred
                break

        entity_parts = " / ".join(top_entities[:2]) if top_entities else "unknown"
        auto_label = f"{label_predicate} of {entity_parts}"

        # --- Heuristic subfield label ---
        # Find the highest-priority entity type with at least 2 entities
        dominant_type: str | None = None
        for candidate_type in _TYPE_PRIORITY:
            if entity_type_counter.get(candidate_type, 0) >= 2:
                dominant_type = candidate_type
                break

        subfield: str | None = None
        if dominant_type:
            # Collect entities that belong to the dominant type, ordered by frequency
            dominant_entities = [
                name
                for name, _ in entity_counter.most_common()
                if entity_type_map.get(name) == dominant_type
            ]
            label_entities = dominant_entities[:2] if dominant_entities else top_entities[:2]
            subfield = " / ".join(label_entities) if label_entities else None
        elif top_entities:
            # Mixed or sparse — use top 2 entities regardless of type
            subfield = " / ".join(top_entities[:2])

        log.debug(
            "label_communities.community",
            community_id=community_id,
            dominant_type=dominant_type,
            subfield=subfield,
            top_entity_types=top_entity_types,
        )

        labels[community_id] = CommunityLabel(
            community_id=community_id,
            auto_label=auto_label,
            top_predicates=top_predicates,
            top_entities=top_entities,
            claim_count=len(claim_ids),
            subfield=subfield,
            top_entity_types=top_entity_types,
        )

    log.debug("label_communities.done", num_communities=len(labels))
    return labels


def compute_community_disagreement(
    communities: list[set[str]],
    contradiction_edges: list[ContradictionEdge],
    labels: dict[int, CommunityLabel],
) -> list[CommunityDisagreement]:
    """Identify cross-community contradiction hotspots, normalized by community size."""
    claim_to_community: dict[str, int] = {}
    for community_id, claim_ids in enumerate(communities):
        for claim_id in claim_ids:
            claim_to_community[claim_id] = community_id

    # Accumulate scores keyed by (min_comm, max_comm) to deduplicate symmetric pairs
    raw_scores: dict[tuple[int, int], float] = defaultdict(float)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    top_pairs_buffer: dict[tuple[int, int], list[tuple[float, dict]]] = defaultdict(list)

    for edge in contradiction_edges:
        comm_a = claim_to_community.get(edge.claim_a_id)
        comm_b = claim_to_community.get(edge.claim_b_id)

        if comm_a is None or comm_b is None or comm_a == comm_b:
            continue

        key = (min(comm_a, comm_b), max(comm_a, comm_b))
        raw_scores[key] += edge.p_contradiction
        pair_counts[key] += 1

        # Parse human-readable text for display
        parts_a = edge.claim_a_id.split("__")
        parts_b = edge.claim_b_id.split("__")
        text_a = f"{parts_a[0]} {parts_a[1]} {parts_a[2]}" if len(parts_a) >= 3 else edge.claim_a_id
        text_b = f"{parts_b[0]} {parts_b[1]} {parts_b[2]}" if len(parts_b) >= 3 else edge.claim_b_id

        top_pairs_buffer[key].append(
            (
                edge.p_contradiction,
                {
                    "claim_a_text": text_a,
                    "claim_b_text": text_b,
                    "p_contradiction": edge.p_contradiction,
                },
            )
        )

    disagreements: list[CommunityDisagreement] = []
    for key, raw_score in raw_scores.items():
        comm_a_id, comm_b_id = key
        label_a = labels.get(comm_a_id)
        label_b = labels.get(comm_b_id)

        size_a = label_a.claim_count if label_a else 1
        size_b = label_b.claim_count if label_b else 1
        normalized_score = raw_score / math.sqrt(size_a * size_b)

        sorted_pairs = sorted(top_pairs_buffer[key], key=lambda x: x[0], reverse=True)
        top_claims = [entry for _, entry in sorted_pairs[:5]]

        disagreements.append(
            CommunityDisagreement(
                comm_a_id=comm_a_id,
                comm_b_id=comm_b_id,
                comm_a_label=label_a.auto_label if label_a else str(comm_a_id),
                comm_b_label=label_b.auto_label if label_b else str(comm_b_id),
                raw_score=raw_score,
                normalized_score=normalized_score,
                num_contradictions=pair_counts[key],
                top_claims=top_claims,
            )
        )

    disagreements.sort(key=lambda d: d.normalized_score, reverse=True)
    log.debug("compute_community_disagreement.done", num_pairs=len(disagreements))
    return disagreements


def build_contradiction_viz_data(
    graph: nx.MultiDiGraph,
    pair_results: list[NLIPairResult],
    communities: list[set[str]],
    threshold: float = 0.8,
) -> ContradictionVizData:
    """Build the full contradiction visualization payload from graph and NLI results."""
    contradiction_edges = compute_contradiction_edges(graph, pair_results, threshold)
    community_labels = label_communities(graph, communities)
    community_disagreements = compute_community_disagreement(
        communities, contradiction_edges, community_labels
    )

    max_disagreement_score = (
        max(d.normalized_score for d in community_disagreements) if community_disagreements else 0.0
    )

    stats: dict[str, Any] = {
        "total_contradiction_edges": len(contradiction_edges),
        "cross_paper_count": sum(1 for e in contradiction_edges if e.is_cross_paper),
        "intra_paper_count": sum(1 for e in contradiction_edges if not e.is_cross_paper),
        "communities_with_labels": len(community_labels),
        "community_disagreement_pairs": len(community_disagreements),
        "max_disagreement_score": max_disagreement_score,
    }

    log.info(
        "build_contradiction_viz_data.done",
        **stats,
    )

    return ContradictionVizData(
        contradiction_edges=contradiction_edges,
        community_labels=community_labels,
        community_disagreements=community_disagreements,
        stats=stats,
    )
