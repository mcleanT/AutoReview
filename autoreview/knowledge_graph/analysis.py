"""Graph analysis utilities: community detection, hub entities, contradiction finding, gaps."""

from __future__ import annotations

from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.cluster import (
    Finding,
    FindingContradiction,
    TopicCluster,
)
from autoreview.knowledge_graph.structural_contradictions import (
    ContradictionType,
    detect_contradictions,
)

log = structlog.get_logger(__name__)

# Weight per contradiction type used in raw_score accumulation
_TYPE_WEIGHTS: dict[ContradictionType, float] = {
    ContradictionType.PREDICATE_OPPOSITION: 1.0,
    ContradictionType.DIRECTION_CONFLICT: 0.8,
    ContradictionType.BOUNDARY_CONDITION: 0.3,
}


def detect_communities(graph: nx.MultiDiGraph) -> list[set[str]]:
    """Detect communities in the knowledge graph using the Louvain algorithm.

    Converts the MultiDiGraph to an undirected graph before detection, since
    Louvain operates on undirected graphs.

    Args:
        graph: Knowledge graph as a MultiDiGraph.

    Returns:
        List of sets, each set containing node IDs belonging to one community.
    """
    graph_undirected = graph.to_undirected()
    communities = list(nx.algorithms.community.louvain_communities(graph_undirected, seed=42))
    log.info("community_detection_complete", num_communities=len(communities))
    return [set(c) for c in communities]


def find_hub_entities(graph: nx.MultiDiGraph, top_n: int = 20) -> list[tuple[str, float]]:
    """Identify hub entities by degree centrality on the undirected projection.

    Args:
        graph: Knowledge graph as a MultiDiGraph.
        top_n: Number of top hubs to return.

    Returns:
        List of (node_id, centrality_score) tuples, sorted by centrality descending.
    """
    graph_undirected = graph.to_undirected()
    centrality: dict[str, float] = nx.degree_centrality(graph_undirected)
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    result = ranked[:top_n]
    log.info("hub_entities_found", top_n=top_n, top_hub=result[0][0] if result else None)
    return result


def find_contradictions(graph: nx.MultiDiGraph, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Find edges with high controversy scores indicating contradictory evidence.

    Args:
        graph: Knowledge graph as a MultiDiGraph.
        threshold: Controversy score above which an edge is flagged (exclusive).

    Returns:
        List of dicts with keys: edge_key, subject, object, predicate,
        controversy_score, confidence_mean.
    """
    contradictions: list[dict[str, Any]] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        score = data.get("controversy_score", 0.0)
        if score > threshold:
            contradictions.append(
                {
                    "edge_key": k,
                    "subject": u,
                    "object": v,
                    "predicate": data.get("predicate", ""),
                    "controversy_score": score,
                    "confidence_mean": data.get("confidence_mean", 0.0),
                }
            )
    log.info(
        "contradictions_found",
        threshold=threshold,
        count=len(contradictions),
    )
    return contradictions


def find_low_evidence_entities(
    graph: nx.MultiDiGraph, min_degree: int = 2, max_evidence: int = 3
) -> list[dict[str, Any]]:
    """Find entities with many connections but sparse evidence — research gaps.

    Identifies nodes that are well-connected in the graph (degree >= min_degree)
    but have low total evidence across their incident edges (total_evidence <= max_evidence).

    Args:
        graph: Knowledge graph as a MultiDiGraph.
        min_degree: Minimum undirected degree to include a node.
        max_evidence: Maximum total evidence count across all incident edges.

    Returns:
        List of dicts with keys: node_id, canonical_name, degree, total_evidence.
    """
    graph_undirected = graph.to_undirected()
    gaps: list[dict[str, Any]] = []

    for node in graph.nodes():
        degree = graph_undirected.degree(node)
        if degree < min_degree:
            continue

        # Sum evidence_count across all incident edges (both in and out)
        total_evidence = 0
        for _, _, data in graph.in_edges(node, data=True):
            total_evidence += data.get("evidence_count", 0)
        for _, _, data in graph.out_edges(node, data=True):
            total_evidence += data.get("evidence_count", 0)

        if total_evidence <= max_evidence:
            gaps.append(
                {
                    "node_id": node,
                    "canonical_name": graph.nodes[node].get("canonical_name", node),
                    "degree": degree,
                    "total_evidence": total_evidence,
                }
            )

    log.info(
        "low_evidence_entities_found",
        min_degree=min_degree,
        max_evidence=max_evidence,
        count=len(gaps),
    )
    return gaps


def find_temporal_gaps(graph: nx.MultiDiGraph, cutoff_year: int = 2020) -> list[dict[str, Any]]:
    """Find edges whose evidence predates a cutoff year — potentially outdated claims.

    Edges with a ``publication_date`` year strictly less than ``cutoff_year`` are
    returned so reviewers can flag claims that have not been revisited recently.

    Args:
        graph: Knowledge graph as a MultiDiGraph. Edges should have a
            ``publication_date`` attribute in ISO-8601 format (``YYYY-MM-DD`` or ``YYYY``).
        cutoff_year: Year threshold; edges published before this year are flagged.

    Returns:
        List of dicts with keys: edge_key, subject, object, predicate, publication_date.
    """
    temporal_gaps: list[dict[str, Any]] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        pub_date: str | None = data.get("publication_date")
        if pub_date is None:
            continue
        try:
            year = int(str(pub_date)[:4])
        except (ValueError, TypeError):
            continue
        if year < cutoff_year:
            temporal_gaps.append(
                {
                    "edge_key": k,
                    "subject": u,
                    "object": v,
                    "predicate": data.get("predicate", ""),
                    "publication_date": pub_date,
                }
            )

    log.info(
        "temporal_gaps_found",
        cutoff_year=cutoff_year,
        count=len(temporal_gaps),
    )
    return temporal_gaps


def extract_subgraph(graph: nx.MultiDiGraph, node_ids: list[str] | set[str]) -> nx.MultiDiGraph:
    """Return the subgraph induced by the given node IDs.

    Args:
        graph: Source knowledge graph.
        node_ids: Collection of node IDs to include.

    Returns:
        A new MultiDiGraph containing only the specified nodes and edges between them.
    """
    subgraph: nx.MultiDiGraph = graph.subgraph(node_ids).copy()
    log.info(
        "subgraph_extracted",
        requested_nodes=len(node_ids),
        actual_nodes=subgraph.number_of_nodes(),
        actual_edges=subgraph.number_of_edges(),
    )
    return subgraph


def score_contradiction_centrality(
    graph: nx.MultiDiGraph,
    boundary_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Score each entity node by its centrality in the contradiction landscape.

    Identifies which entity nodes sit at the center of scientific disagreement
    by aggregating all structural contradiction pairs that involve each node.

    Algorithm:
        1. Detect all ContradictionPair instances via detect_contradictions().
        2. For each pair, both subject and object nodes are credited.
        3. Per-node raw_score = Σ (type_weight × condition_coupling) for all
           contradictions involving that node. Type weights are:
           PREDICATE_OPPOSITION=1.0, DIRECTION_CONFLICT=0.8,
           BOUNDARY_CONDITION=0.3.
        4. normalized_score = raw_score / undirected_degree (controls for
           high-degree hubs that accumulate contradictions incidentally).
        5. Writes ``contradiction_centrality`` (raw_score) and
           ``n_contradictions`` as node attributes on the graph (in-place).

    Args:
        graph: Knowledge graph as a MultiDiGraph. Each edge must have at
            minimum an ``edge_id`` and ``predicate`` attribute. Optional:
            ``direction``, ``organism``, ``model_system``, ``in_vitro``,
            ``conditions``.
        boundary_threshold: Forwarded to detect_contradictions(). Minimum
            condition coupling for a contradiction to be classified as a
            "true" contradiction rather than a boundary condition.

    Returns:
        List of dicts sorted by raw_score descending, one entry per node
        with at least one contradiction. Each dict contains:
        ``node_id``, ``canonical_name``, ``raw_score``, ``normalized_score``,
        ``n_contradictions``, ``degree``.
    """
    pairs = detect_contradictions(graph, boundary_threshold=boundary_threshold)

    # Undirected degree for normalization (degree in the undirected projection)
    graph_undirected = graph.to_undirected()

    # Accumulate per-node scores
    raw_scores: dict[str, float] = {}
    n_contradictions: dict[str, int] = {}

    for pair in pairs:
        weight = _TYPE_WEIGHTS[pair.contradiction_type]
        contribution = weight * pair.condition_coupling
        for node in (pair.subject, pair.object):
            raw_scores[node] = raw_scores.get(node, 0.0) + contribution
            n_contradictions[node] = n_contradictions.get(node, 0) + 1

    # Write node attributes onto the graph (all nodes, zero for non-involved)
    for node in graph.nodes():
        graph.nodes[node]["contradiction_centrality"] = raw_scores.get(node, 0.0)
        graph.nodes[node]["n_contradictions"] = n_contradictions.get(node, 0)

    # Build ranked result list (only nodes with at least one contradiction)
    results: list[dict[str, Any]] = []
    for node, raw_score in raw_scores.items():
        degree = graph_undirected.degree(node)
        normalized_score = raw_score / degree if degree > 0 else 0.0
        results.append(
            {
                "node_id": node,
                "canonical_name": graph.nodes[node].get("canonical_name", node),
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "n_contradictions": n_contradictions[node],
                "degree": degree,
            }
        )

    results.sort(key=lambda x: x["raw_score"], reverse=True)

    top_node = results[0]["node_id"] if results else None
    log.info(
        "contradiction_centrality_scored",
        nodes_with_contradictions=len(results),
        total_contradiction_pairs=len(pairs),
        top_node=top_node,
    )

    return results


def summarize_topic_clusters(
    clusters: list[TopicCluster],
    findings: list[Finding],
) -> list[dict[str, Any]]:
    """Summarize topic clusters for reporting.

    Args:
        clusters: Topic clusters from build_topic_clusters().
        findings: Findings from form_findings().

    Returns:
        List of summary dicts sorted by n_edges descending, each with:
        cluster_id, subject_id, object_id, predicate_class, n_edges, n_findings, member_predicates.
    """
    findings_by_cluster: dict[str, list[Finding]] = {}
    for f in findings:
        findings_by_cluster.setdefault(f.topic_cluster_id, []).append(f)

    summaries: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_findings = findings_by_cluster.get(cluster.cluster_id, [])
        summaries.append(
            {
                "cluster_id": cluster.cluster_id,
                "subject_id": cluster.subject_id,
                "object_id": cluster.object_id,
                "predicate_class": cluster.predicate_class,
                "n_edges": len(cluster.edge_ids),
                "n_findings": len(cluster_findings),
                "member_predicates": sorted(cluster.member_predicates),
            }
        )

    summaries.sort(key=lambda s: s["n_edges"], reverse=True)
    log.info("analysis.topic_cluster_summary", n_clusters=len(summaries))
    return summaries


# Weight per finding contradiction type
_FINDING_TYPE_WEIGHTS: dict[str, float] = {
    "directional": 1.0,
    "boundary": 0.3,
    "interpretive": 0.7,
}


def score_finding_contradiction_centrality(
    graph: nx.MultiDiGraph,
    finding_contradictions: list[FindingContradiction],
) -> list[dict[str, Any]]:
    """Score each entity by its centrality in finding-level contradictions.

    Args:
        graph: Knowledge graph (for node metadata).
        finding_contradictions: Contradictions from detect_finding_contradictions().

    Returns:
        List of dicts sorted by raw_score descending, each with:
        node_id, canonical_name, raw_score, n_finding_contradictions.
    """
    if not finding_contradictions:
        return []

    from autoreview.knowledge_graph.cluster import build_topic_clusters

    raw_scores: dict[str, float] = {}
    n_contras: dict[str, int] = {}

    clusters = build_topic_clusters(graph)
    cluster_map: dict[str, TopicCluster] = {c.cluster_id: c for c in clusters}

    for fc in finding_contradictions:
        tc = cluster_map.get(fc.topic_cluster_id)
        if tc is None:
            continue
        weight = _FINDING_TYPE_WEIGHTS.get(fc.contradiction_type, 0.5)
        contribution = weight * fc.severity
        for node in (tc.subject_id, tc.object_id):
            raw_scores[node] = raw_scores.get(node, 0.0) + contribution
            n_contras[node] = n_contras.get(node, 0) + 1

    results: list[dict[str, Any]] = []
    for node, raw_score in raw_scores.items():
        results.append(
            {
                "node_id": node,
                "canonical_name": graph.nodes[node].get("canonical_name", node)
                if node in graph
                else node,
                "raw_score": raw_score,
                "n_finding_contradictions": n_contras[node],
            }
        )

    results.sort(key=lambda x: x["raw_score"], reverse=True)
    log.info(
        "analysis.finding_contradiction_centrality",
        nodes_with_contradictions=len(results),
    )
    return results
