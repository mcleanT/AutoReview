"""Graph analysis utilities: community detection, hub entities, contradiction finding, gap analysis."""

from __future__ import annotations

from typing import Any

import networkx as nx
import structlog

log = structlog.get_logger(__name__)


def detect_communities(G: nx.MultiDiGraph) -> list[set[str]]:
    """Detect communities in the knowledge graph using the Louvain algorithm.

    Converts the MultiDiGraph to an undirected graph before detection, since
    Louvain operates on undirected graphs.

    Args:
        G: Knowledge graph as a MultiDiGraph.

    Returns:
        List of sets, each set containing node IDs belonging to one community.
    """
    G_undirected = G.to_undirected()
    communities = list(nx.algorithms.community.louvain_communities(G_undirected, seed=42))
    log.info("community_detection_complete", num_communities=len(communities))
    return [set(c) for c in communities]


def find_hub_entities(G: nx.MultiDiGraph, top_n: int = 20) -> list[tuple[str, float]]:
    """Identify hub entities by degree centrality on the undirected projection.

    Args:
        G: Knowledge graph as a MultiDiGraph.
        top_n: Number of top hubs to return.

    Returns:
        List of (node_id, centrality_score) tuples, sorted by centrality descending.
    """
    G_undirected = G.to_undirected()
    centrality: dict[str, float] = nx.degree_centrality(G_undirected)
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    result = ranked[:top_n]
    log.info("hub_entities_found", top_n=top_n, top_hub=result[0][0] if result else None)
    return result


def find_contradictions(G: nx.MultiDiGraph, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Find edges with high controversy scores indicating contradictory evidence.

    Args:
        G: Knowledge graph as a MultiDiGraph.
        threshold: Controversy score above which an edge is flagged (exclusive).

    Returns:
        List of dicts with keys: edge_key, subject, object, predicate,
        controversy_score, confidence_mean.
    """
    contradictions: list[dict[str, Any]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
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
    G: nx.MultiDiGraph, min_degree: int = 2, max_evidence: int = 3
) -> list[dict[str, Any]]:
    """Find entities with many connections but sparse evidence — research gaps.

    Identifies nodes that are well-connected in the graph (degree >= min_degree)
    but have low total evidence across their incident edges (total_evidence <= max_evidence).

    Args:
        G: Knowledge graph as a MultiDiGraph.
        min_degree: Minimum undirected degree to include a node.
        max_evidence: Maximum total evidence count across all incident edges.

    Returns:
        List of dicts with keys: node_id, canonical_name, degree, total_evidence.
    """
    G_undirected = G.to_undirected()
    gaps: list[dict[str, Any]] = []

    for node in G.nodes():
        degree = G_undirected.degree(node)
        if degree < min_degree:
            continue

        # Sum evidence_count across all incident edges (both in and out)
        total_evidence = 0
        for _, _, data in G.in_edges(node, data=True):
            total_evidence += data.get("evidence_count", 0)
        for _, _, data in G.out_edges(node, data=True):
            total_evidence += data.get("evidence_count", 0)

        if total_evidence <= max_evidence:
            gaps.append(
                {
                    "node_id": node,
                    "canonical_name": G.nodes[node].get("canonical_name", node),
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


def find_temporal_gaps(G: nx.MultiDiGraph, cutoff_year: int = 2020) -> list[dict[str, Any]]:
    """Find edges whose evidence predates a cutoff year — potentially outdated claims.

    Edges with a ``publication_date`` year strictly less than ``cutoff_year`` are
    returned so reviewers can flag claims that have not been revisited recently.

    Args:
        G: Knowledge graph as a MultiDiGraph. Edges should have a ``publication_date``
           attribute in ISO-8601 format (``YYYY-MM-DD`` or ``YYYY``).
        cutoff_year: Year threshold; edges published before this year are flagged.

    Returns:
        List of dicts with keys: edge_key, subject, object, predicate, publication_date.
    """
    temporal_gaps: list[dict[str, Any]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
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


def extract_subgraph(G: nx.MultiDiGraph, node_ids: list[str] | set[str]) -> nx.MultiDiGraph:
    """Return the subgraph induced by the given node IDs.

    Args:
        G: Source knowledge graph.
        node_ids: Collection of node IDs to include.

    Returns:
        A new MultiDiGraph containing only the specified nodes and edges between them.
    """
    subgraph: nx.MultiDiGraph = G.subgraph(node_ids).copy()
    log.info(
        "subgraph_extracted",
        requested_nodes=len(node_ids),
        actual_nodes=subgraph.number_of_nodes(),
        actual_edges=subgraph.number_of_edges(),
    )
    return subgraph
