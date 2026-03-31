"""Generate 5-tab analytical knowledge graph visualization."""

import asyncio
import contextlib
import json
import pickle
from pathlib import Path

import networkx as nx
import structlog

log = structlog.get_logger(__name__)

OUTPUT = Path("output/knowledge_graph/full_contradictions_tabbed.html")
KG_PATH = Path("output/knowledge_graph/gastruloid_kg_v2_nli.pkl")
NLI_PATH = Path("output/knowledge_graph/cross_claim_nli_v2_contradictions.json")


def load_graph():
    with open(KG_PATH, "rb") as f:
        return pickle.load(f)


def load_nli_pairs(graph):
    """Load NLI pair results, reconstructing claim IDs from the JSON format.

    Handles three JSON formats:
      1. ``claim_a_id`` / ``claim_b_id`` as edge-key strings
         (produced by ``nli.classify_cross_claims`` or the CLI json_report).
      2. ``claim_a`` / ``claim_b`` as dicts with ``subject_id``, ``predicate``,
         ``object_id`` (produced by the KG-extraction scripts).
      3. Flat keys ``claim_a_subject``, ``claim_a_predicate``, ``claim_a_object``
         (legacy format).

    For formats 2 and 3 the function reconstructs the edge-key string by looking
    up the edge in the graph.  When a (subject, predicate, object) triple appears
    as multiple parallel edges (MultiDiGraph), *all* matching keys are tried so
    that the reconstructed claim ID is guaranteed to be present in
    ``_build_edge_index`` — avoiding the silent ``entry_a is None`` path in
    ``compute_contradiction_edges`` that would incorrectly classify the pair as
    intra-paper.
    """
    from collections import defaultdict

    from autoreview.knowledge_graph.nli import NLIPairResult

    with open(NLI_PATH) as f:
        raw = json.load(f)

    # Build a multimap: (subject_id, predicate, object_id) -> [edge_key, ...]
    # A plain dict (last-wins) is wrong for MultiDiGraph — use a list per triple.
    edge_key_multimap: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    edge_index_keys: set[str] = set()
    for u, v, k, data in graph.edges(keys=True, data=True):
        pred = data.get("predicate", "")
        edge_key_multimap[(u, pred, v)].append(k)
        edge_index_keys.add(f"{u}__{pred}__{v}__{k}")

    def _best_key_for_triple(subject_id: str, predicate: str, object_id: str) -> str | None:
        """Return the first edge key for this triple that is present in edge_index."""
        for k in edge_key_multimap.get((subject_id, predicate, object_id), []):
            candidate = f"{subject_id}__{predicate}__{object_id}__{k}"
            if candidate in edge_index_keys:
                return k
        return None

    pairs = []
    for r in raw:
        # ── Format 1: direct edge-key strings ──────────────────────────────────
        if "claim_a_id" in r:
            # Strip unexpected fields so Pydantic validation doesn't fail on
            # records written by other tools.
            with contextlib.suppress(Exception):
                pairs.append(NLIPairResult(**r))

        # ── Format 2: nested claim dicts ───────────────────────────────────────
        elif "claim_a" in r and isinstance(r["claim_a"], dict):
            ca, cb = r["claim_a"], r["claim_b"]
            a_key = _best_key_for_triple(ca["subject_id"], ca["predicate"], ca["object_id"])
            b_key = _best_key_for_triple(cb["subject_id"], cb["predicate"], cb["object_id"])
            if a_key is not None and b_key is not None:
                pairs.append(
                    NLIPairResult(
                        claim_a_id=f"{ca['subject_id']}__{ca['predicate']}__{ca['object_id']}__{a_key}",
                        claim_b_id=f"{cb['subject_id']}__{cb['predicate']}__{cb['object_id']}__{b_key}",
                        p_contradiction=r["p_contradiction"],
                        p_entailment=r.get("p_entailment", 0.0),
                        p_neutral=r.get("p_neutral", 0.0),
                        method=r.get("method", "nli"),
                        shared_entities=r.get("shared_entities", []),
                    )
                )

        # ── Format 3: flat claim_a_subject / claim_a_predicate / claim_a_object ─
        else:
            subj_a = r.get("claim_a_subject", "")
            pred_a = r.get("claim_a_predicate", "")
            obj_a = r.get("claim_a_object", "")
            subj_b = r.get("claim_b_subject", "")
            pred_b = r.get("claim_b_predicate", "")
            obj_b = r.get("claim_b_object", "")
            a_key = _best_key_for_triple(subj_a, pred_a, obj_a)
            b_key = _best_key_for_triple(subj_b, pred_b, obj_b)
            if a_key is not None and b_key is not None:
                pairs.append(
                    NLIPairResult(
                        claim_a_id=f"{subj_a}__{pred_a}__{obj_a}__{a_key}",
                        claim_b_id=f"{subj_b}__{pred_b}__{obj_b}__{b_key}",
                        p_contradiction=r["p_contradiction"],
                        p_entailment=r.get("p_entailment", 0.0),
                        p_neutral=r.get("p_neutral", 0.0),
                        method=r.get("method", "nli"),
                        shared_entities=r.get("shared_entities", []),
                    )
                )

    print(f"Loaded {len(pairs)} NLI pair results")
    return pairs


def compute_communities(graph):
    """Build claim graph and run Louvain for community detection."""
    entity_to_claims = {}
    claim_nodes = set()
    for u, v, k, data in graph.edges(keys=True, data=True):
        pred = data.get("predicate", "")
        cid = f"{u}__{pred}__{v}__{k}"
        claim_nodes.add(cid)
        entity_to_claims.setdefault(u, set()).add(cid)
        entity_to_claims.setdefault(v, set()).add(cid)

    claim_nx = nx.Graph()
    claim_nx.add_nodes_from(claim_nodes)
    for _entity, claims in entity_to_claims.items():
        clist = list(claims)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                if claim_nx.has_edge(clist[i], clist[j]):
                    claim_nx[clist[i]][clist[j]]["weight"] += 1
                else:
                    claim_nx.add_edge(clist[i], clist[j], weight=1)

    communities = [set(c) for c in nx.algorithms.community.louvain_communities(claim_nx, seed=42)]
    print(f"  Communities: {len(communities)}")
    return communities


def filter_graph(graph, edge_filter):
    """Create a subgraph with only edges matching the filter function."""
    sub = nx.MultiDiGraph()
    for u, v, k, data in graph.edges(keys=True, data=True):
        if edge_filter(u, v, k, data):
            if not sub.has_node(u):
                sub.add_node(u, **graph.nodes[u])
            if not sub.has_node(v):
                sub.add_node(v, **graph.nodes[v])
            sub.add_edge(u, v, key=k, **data)
    return sub


def make_cross_paper_subgraph(graph, viz_data_full):
    """Subgraph of claims involved in cross-paper contradictions."""
    cross_claim_ids = set()
    for edge in viz_data_full.contradiction_edges:
        if edge.is_cross_paper:
            cross_claim_ids.add(edge.claim_a_id)
            cross_claim_ids.add(edge.claim_b_id)

    def edge_in_cross(u, v, k, data):
        pred = data.get("predicate", "")
        return f"{u}__{pred}__{v}__{k}" in cross_claim_ids

    sub = filter_graph(graph, edge_in_cross)
    print(f"  Cross-Paper Disputes: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub


def make_mechanistic_subgraph(graph):
    """Subgraph of only mechanistic_causal assertions."""
    sub = filter_graph(graph, lambda u, v, k, d: d.get("assertion_type") == "mechanistic_causal")
    print(f"  Mechanistic Core: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub


def make_controversial_subgraph(graph, percentile=0.75):
    """Subgraph of claims with controversy scores above the percentile threshold."""
    scores = [d.get("controversy_score", 0) for _, _, _, d in graph.edges(keys=True, data=True)]
    scores = [s for s in scores if s > 0]
    if not scores:
        return graph  # fallback to full graph
    scores.sort()
    threshold = scores[int(len(scores) * percentile)] if scores else 0
    print(f"  Controversy threshold (p{int(percentile * 100)}): {threshold:.4f}")
    sub = filter_graph(graph, lambda u, v, k, d: d.get("controversy_score", 0) >= threshold)
    print(f"  Controversial Frontier: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub


async def main():
    from autoreview.knowledge_graph.community_labeling import enrich_community_subfields
    from autoreview.knowledge_graph.contradiction_viz import build_contradiction_viz_data
    from autoreview.knowledge_graph.interactive import generate_multi_graph_html

    from autoreview.llm.claude_code import ClaudeCodeProvider

    # Load data
    print("Loading graph...")
    graph = load_graph()
    print(f"Full graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Loading NLI pairs...")
    pair_results = load_nli_pairs(graph)

    # ── Tab 1 & 2: Full graph (different coloring modes) ──
    print("\nBuilding full graph viz data...")
    full_communities = compute_communities(graph)
    full_viz = build_contradiction_viz_data(graph, pair_results, full_communities, threshold=0.5)
    print(f"  Contradiction edges: {len(full_viz.contradiction_edges)}")
    print(f"  Disagreements: {len(full_viz.community_disagreements)}")

    # ── Tab 3: Cross-Paper Disputes ──
    print("\nBuilding Cross-Paper Disputes subgraph...")
    cross_graph = make_cross_paper_subgraph(graph, full_viz)
    if cross_graph.number_of_edges() > 0:
        cross_communities = compute_communities(cross_graph)
        cross_viz = build_contradiction_viz_data(
            cross_graph, pair_results, cross_communities, threshold=0.5
        )
    else:
        cross_viz = None
        cross_graph = graph  # fallback

    # ── Tab 4: Mechanistic Core ──
    print("\nBuilding Mechanistic Core subgraph...")
    mech_graph = make_mechanistic_subgraph(graph)
    if mech_graph.number_of_edges() > 0:
        mech_communities = compute_communities(mech_graph)
        mech_viz = build_contradiction_viz_data(
            mech_graph, pair_results, mech_communities, threshold=0.5
        )
    else:
        mech_viz = None
        mech_graph = graph

    # ── Tab 5: Controversial Frontier ──
    print("\nBuilding Controversial Frontier subgraph...")
    contr_graph = make_controversial_subgraph(graph, percentile=0.75)
    if contr_graph.number_of_edges() > 0:
        contr_communities = compute_communities(contr_graph)
        contr_viz = build_contradiction_viz_data(
            contr_graph, pair_results, contr_communities, threshold=0.5
        )
    else:
        contr_viz = None
        contr_graph = graph

    # ── LLM enrichment for significant tabs ──
    llm = ClaudeCodeProvider()
    domain = "gastruloid biology, stem cell differentiation, and developmental biology"

    print("\nEnriching full graph communities...")
    full_viz = await enrich_community_subfields(
        full_viz, graph, full_communities, llm, domain, min_claims=5
    )

    if cross_viz and len(cross_viz.community_labels) > 0:
        print("Enriching cross-paper communities...")
        cross_viz = await enrich_community_subfields(
            cross_viz, cross_graph, cross_communities, llm, domain, min_claims=3
        )

    if mech_viz and len(mech_viz.community_labels) > 0:
        print("Enriching mechanistic communities...")
        mech_viz = await enrich_community_subfields(
            mech_viz, mech_graph, mech_communities, llm, domain, min_claims=3
        )

    if contr_viz and len(contr_viz.community_labels) > 0:
        print("Enriching controversial communities...")
        contr_viz = await enrich_community_subfields(
            contr_viz, contr_graph, contr_communities, llm, domain, min_claims=3
        )

    # ── Generate 6-tab HTML ──
    print("\nGenerating 6-tab HTML...")
    tabs = [
        (
            "Playground",
            graph,
            full_viz,
            {"defaultColorMode": "assertion", "defaultEdgeMode": "both"},
        ),
        (
            "Assertion Types",
            graph,
            full_viz,
            {"defaultColorMode": "assertion", "defaultEdgeMode": "shared"},
        ),
        (
            "Community Clusters",
            graph,
            full_viz,
            {"defaultColorMode": "community", "defaultEdgeMode": "shared"},
        ),
        (
            "Cross-Paper Disputes",
            cross_graph,
            cross_viz,
            {"defaultColorMode": "community", "defaultEdgeMode": "contradiction"},
        ),
        (
            "Mechanistic Core",
            mech_graph,
            mech_viz,
            {"defaultColorMode": "assertion", "defaultEdgeMode": "both"},
        ),
        (
            "Controversial Frontier",
            contr_graph,
            contr_viz,
            {"defaultColorMode": "community", "defaultEdgeMode": "contradiction"},
        ),
    ]

    output_path = generate_multi_graph_html(graphs=tabs, output_path=OUTPUT)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nGenerated: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Tabs: {len(tabs)}")

    # Print tab summary
    for label, g, viz, _opts in tabs:
        n_edges = g.number_of_edges()
        n_contra = len(viz.contradiction_edges) if viz else 0
        n_comm = len(viz.community_labels) if viz else 0
        print(f"  {label}: {n_edges} claims, {n_contra} contradictions, {n_comm} communities")


if __name__ == "__main__":
    asyncio.run(main())
