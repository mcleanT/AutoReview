"""Topological analysis of contradiction networks.

Identifies load-bearing contradictions (bridge edges, high betweenness),
simulates resolution scenarios, and ranks contradictions by Value of Information.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class ResolutionResult:
    scenario: str
    removed_node: str | None  # which node was removed (None for dissolved)
    components_before: int
    components_after: int
    delta_components: int  # after - before
    largest_component_before: int
    largest_component_after: int
    orphaned_claims: list[str]  # claims now in singleton components
    fragmented_communities: int  # new components created

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_contradiction_graph(
    kg: nx.MultiDiGraph,
    contradictions: list[dict[str, Any]],
    min_p: float = 0.5,
) -> nx.Graph:
    """Build a claim-centric graph where nodes are KG claims and edges are contradictions.

    Args:
        kg: Knowledge graph MultiDiGraph with entity nodes and claim edges.
        contradictions: List of contradiction dicts with claim_a_id, claim_b_id, p_contradiction, etc.
        min_p: Minimum p_contradiction threshold for including an edge.

    Returns:
        Undirected nx.Graph with claims as nodes and contradictions as edges.
    """
    G: nx.Graph = nx.Graph()

    # Build a lookup: claim_id -> (u, v, key, edge_data) for all KG edges
    claim_lookup: dict[str, tuple[str, str, int, dict[str, Any]]] = {}
    for u, v, key, data in kg.edges(data=True, keys=True):
        claim_id = f"{u}__{data.get('predicate', '')}_{v}__{key}"
        # Also try the canonical format used in contradiction JSON
        # Format: "{subject_entity_id}__{predicate}__{object_entity_id}__{edge_key}"
        predicate = data.get("predicate", "")
        canonical_id = f"{u}__{predicate}__{v}__{key}"
        claim_lookup[canonical_id] = (u, v, key, data)

    # Add nodes for each KG edge (claim)
    for u, v, key, data in kg.edges(data=True, keys=True):
        predicate = data.get("predicate", "")
        claim_id = f"{u}__{predicate}__{v}__{key}"

        # Build human-readable label from canonical names
        subject_name = kg.nodes[u].get("canonical_name", u) if u in kg.nodes else u
        object_name = kg.nodes[v].get("canonical_name", v) if v in kg.nodes else v
        label = f"{subject_name} → {predicate} → {object_name}"

        G.add_node(
            claim_id,
            label=label,
            assertion_type=data.get("assertion_type", ""),
            confidence_mean=data.get("confidence_mean", 0.0),
            evidence_count=data.get("evidence_count", 0),
            controversy_score=data.get("controversy_score", 0.0),
            nli_alpha=data.get("_nli_alpha", 1.0),
            nli_beta=data.get("_nli_beta", 1.0),
        )

    # Add edges for contradictions above threshold
    edges_added = 0
    for contradiction in contradictions:
        p_contradiction = contradiction.get("p_contradiction", 0.0)
        if p_contradiction < min_p:
            continue

        claim_a_id = contradiction["claim_a_id"]
        claim_b_id = contradiction["claim_b_id"]

        # Only add edge if both nodes exist
        if claim_a_id not in G.nodes or claim_b_id not in G.nodes:
            continue

        G.add_edge(
            claim_a_id,
            claim_b_id,
            p_contradiction=p_contradiction,
            p_entailment=contradiction.get("p_entailment", 0.0),
            p_neutral=contradiction.get("p_neutral", 0.0),
            method=contradiction.get("method", ""),
            shared_entities=contradiction.get("shared_entities", []),
        )
        edges_added += 1

    log.info(
        "contradiction_graph_built",
        n_claim_nodes=G.number_of_nodes(),
        n_contradiction_edges=G.number_of_edges(),
        min_p_threshold=min_p,
    )
    return G


def compute_uncertainty(alpha: float, beta_param: float) -> float:
    """Compute normalized uncertainty from Beta distribution variance.

    Uses variance of the Beta distribution normalized by the maximum variance
    (Beta(1,1) = 1/12). High variance = high uncertainty = we'd learn a lot
    by resolving this claim.

    Args:
        alpha: Alpha parameter of the Beta distribution.
        beta_param: Beta parameter of the Beta distribution.

    Returns:
        Normalized uncertainty in [0, 1]. 1.0 = maximally uncertain (uniform prior),
        0.0 = fully resolved (strong evidence in one direction).
    """
    # Variance of Beta(a, b) = ab / ((a+b)^2 * (a+b+1))
    total = alpha + beta_param
    if total == 0:
        return 1.0
    variance = (alpha * beta_param) / (total**2 * (total + 1))
    max_variance = 1.0 / 12.0  # Beta(1,1) variance
    return float(min(1.0, variance / max_variance))


def find_bridges_and_articulation_points(
    G: nx.Graph,
) -> dict[str, Any]:
    """Find bridge edges and articulation points in the contradiction graph.

    Args:
        G: Undirected contradiction graph.

    Returns:
        Dict with keys: bridges, articulation_points, n_bridges, n_articulation_points.
    """
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))

    result = {
        "bridges": bridges,
        "articulation_points": articulation_points,
        "n_bridges": len(bridges),
        "n_articulation_points": len(articulation_points),
    }

    log.info(
        "bridges_and_articulation_points_computed",
        n_bridges=len(bridges),
        n_articulation_points=len(articulation_points),
    )
    return result


def compute_edge_betweenness(G: nx.Graph) -> list[tuple[str, str, float]]:
    """Compute edge betweenness centrality for all contradiction edges.

    Args:
        G: Undirected contradiction graph.

    Returns:
        List of (claim_a_id, claim_b_id, betweenness_score) tuples, sorted descending.
    """
    log.info("computing_edge_betweenness", n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges())
    betweenness: dict[tuple[str, str], float] = nx.edge_betweenness_centrality(G)

    sorted_results = sorted(
        ((u, v, score) for (u, v), score in betweenness.items()),
        key=lambda x: x[2],
        reverse=True,
    )
    return sorted_results


def simulate_resolution(
    G: nx.Graph,
    claim_a: str,
    claim_b: str,
    scenario: str,
) -> ResolutionResult:
    """Simulate the effect of resolving a contradiction between two claims.

    Args:
        G: Undirected contradiction graph.
        claim_a: Node ID of the first claim.
        claim_b: Node ID of the second claim.
        scenario: One of "a_wins", "b_wins", or "dissolved".

    Returns:
        ResolutionResult dataclass describing graph changes.
    """
    # Before state
    components_before = nx.number_connected_components(G)
    component_sizes_before = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    largest_component_before = component_sizes_before[0] if component_sizes_before else 0

    G_copy = G.copy()
    removed_node: str | None = None

    if scenario == "a_wins":
        removed_node = claim_b
        if claim_b in G_copy:
            G_copy.remove_node(claim_b)
    elif scenario == "b_wins":
        removed_node = claim_a
        if claim_a in G_copy:
            G_copy.remove_node(claim_a)
    elif scenario == "dissolved":
        removed_node = None
        if G_copy.has_edge(claim_a, claim_b):
            G_copy.remove_edge(claim_a, claim_b)
    else:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. Must be 'a_wins', 'b_wins', or 'dissolved'."
        )

    # After state
    components_after = nx.number_connected_components(G_copy)
    component_sizes_after = sorted((len(c) for c in nx.connected_components(G_copy)), reverse=True)
    largest_component_after = component_sizes_after[0] if component_sizes_after else 0

    # Orphaned claims = claims now in singleton components
    orphaned_claims = [
        node
        for component in nx.connected_components(G_copy)
        if len(component) == 1
        for node in component
    ]

    # New components created by this resolution
    fragmented_communities = max(0, components_after - components_before)

    return ResolutionResult(
        scenario=scenario,
        removed_node=removed_node,
        components_before=components_before,
        components_after=components_after,
        delta_components=components_after - components_before,
        largest_component_before=largest_component_before,
        largest_component_after=largest_component_after,
        orphaned_claims=orphaned_claims,
        fragmented_communities=fragmented_communities,
    )


def compute_voi(
    G: nx.Graph,
    kg: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    """Compute Value of Information (VOI) for each contradiction edge.

    Higher VOI = resolving this contradiction would most reduce uncertainty and
    fragment (or clarify) the contradiction network.

    Args:
        G: Undirected contradiction graph built by build_contradiction_graph.
        kg: Original knowledge graph (used to look up node labels).

    Returns:
        List of dicts sorted by VOI descending, each containing claim IDs, labels,
        VOI score, component metrics, and resolution predictions.
    """
    log.info("computing_voi", n_edges=G.number_of_edges())

    # Precompute edge betweenness once
    betweenness_raw: dict[tuple[str, str], float] = nx.edge_betweenness_centrality(G)
    # Normalize key order for lookup
    betweenness_lookup: dict[frozenset[str], float] = {
        frozenset([u, v]): score for (u, v), score in betweenness_raw.items()
    }

    # Identify bridge edges
    bridge_set: set[frozenset[str]] = {frozenset([u, v]) for u, v in nx.bridges(G)}

    results: list[dict[str, Any]] = []

    for claim_a, claim_b, edge_data in G.edges(data=True):
        edge_key = frozenset([claim_a, claim_b])

        # Betweenness for this edge
        betweenness = betweenness_lookup.get(edge_key, 0.0)

        # Uncertainty: average of both claims' Beta distribution entropies
        node_a_data = G.nodes[claim_a]
        node_b_data = G.nodes[claim_b]
        alpha_a = node_a_data.get("nli_alpha", 1.0)
        beta_a = node_a_data.get("nli_beta", 1.0)
        alpha_b = node_b_data.get("nli_alpha", 1.0)
        beta_b = node_b_data.get("nli_beta", 1.0)
        uncertainty = (
            compute_uncertainty(alpha_a, beta_a) + compute_uncertainty(alpha_b, beta_b)
        ) / 2.0

        # Simulate all 3 scenarios
        res_a_wins = simulate_resolution(G, claim_a, claim_b, "a_wins")
        res_b_wins = simulate_resolution(G, claim_a, claim_b, "b_wins")
        res_dissolved = simulate_resolution(G, claim_a, claim_b, "dissolved")

        blast_radius = max(
            res_a_wins.delta_components,
            res_b_wins.delta_components,
            res_dissolved.delta_components,
        )

        # VOI formula: betweenness * uncertainty * blast_radius
        # Fallback: if blast_radius == 0, use 1 (still informative)
        effective_blast = blast_radius if blast_radius > 0 else 1
        voi = betweenness * uncertainty * effective_blast

        results.append(
            {
                "claim_a_id": claim_a,
                "claim_b_id": claim_b,
                "claim_a_label": node_a_data.get("label", claim_a),
                "claim_b_label": node_b_data.get("label", claim_b),
                "p_contradiction": edge_data.get("p_contradiction", 0.0),
                "voi": voi,
                "betweenness": betweenness,
                "uncertainty": uncertainty,
                "blast_radius": blast_radius,
                "is_bridge": edge_key in bridge_set,
                "resolution_predictions": {
                    "a_wins": res_a_wins.as_dict(),
                    "b_wins": res_b_wins.as_dict(),
                    "dissolved": res_dissolved.as_dict(),
                },
            }
        )

    results.sort(key=lambda x: x["voi"], reverse=True)
    log.info("voi_computed", n_contradictions=len(results))
    return results


def run_topology_analysis(
    kg_path: Path | str,
    contradictions_path: Path | str,
    output_dir: Path | str,
    min_p: float = 0.5,
    top_n: int = 50,
) -> dict[str, Any]:
    """Main entry point: load data, run full topological analysis, save results.

    Args:
        kg_path: Path to the KG pickle file (nx.MultiDiGraph).
        contradictions_path: Path to the contradiction JSON file.
        output_dir: Directory to write topology_analysis.json.
        min_p: Minimum p_contradiction threshold for contradiction edges.
        top_n: Number of top VOI contradictions to include in the output.

    Returns:
        Full results dict (also written to topology_analysis.json).
    """
    kg_path = Path(kg_path)
    contradictions_path = Path(contradictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KG
    log.info("loading_kg", path=str(kg_path))
    with open(kg_path, "rb") as f:
        kg: nx.MultiDiGraph = pickle.load(f)
    log.info(
        "kg_loaded",
        n_nodes=kg.number_of_nodes(),
        n_edges=kg.number_of_edges(),
    )

    # Load contradictions
    log.info("loading_contradictions", path=str(contradictions_path))
    with open(contradictions_path) as f:
        contradictions: list[dict[str, Any]] = json.load(f)
    log.info("contradictions_loaded", n_contradictions=len(contradictions))

    # Build contradiction graph
    G = build_contradiction_graph(kg, contradictions, min_p=min_p)

    # Structural analysis
    bridge_info = find_bridges_and_articulation_points(G)

    # VOI ranking (includes betweenness + simulation internally)
    voi_results = compute_voi(G, kg)

    # Summary stats
    connected_components = nx.number_connected_components(G)
    summary = {
        "total_claims": G.number_of_nodes(),
        "total_contradictions": G.number_of_edges(),
        "connected_components": connected_components,
        "bridges": bridge_info["n_bridges"],
        "articulation_points": bridge_info["n_articulation_points"],
    }

    log.info("topology_analysis_summary", **summary)

    # Serialize bridge edges — convert tuple pairs to lists for JSON
    bridges_serializable = [list(b) for b in bridge_info["bridges"]]

    full_results: dict[str, Any] = {
        "summary": summary,
        "top_voi_contradictions": voi_results[:top_n],
        "all_bridges": bridges_serializable,
        "all_articulation_points": bridge_info["articulation_points"],
    }

    output_path = output_dir / "topology_analysis.json"
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    log.info("topology_analysis_saved", path=str(output_path))
    return full_results
