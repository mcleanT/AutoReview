"""Claim-centric interactive HTML visualization for the knowledge graph.

Generates a self-contained HTML file where:
- **Nodes = claims** (subject → predicate → object assertions)
- **Edges = shared entities** between claims (claims referencing the same entity are connected)
- Pan/zoom/drag network exploration
- Assertion-type coloring with toggle to community coloring
- Node size scaled by evidence count
- Edge thickness by number of shared entities
- Hover tooltips with rich claim metadata
- Search box to find claims by entity name or predicate
- Filter by assertion type and minimum confidence
- Physics simulation toggle
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import structlog

from autoreview.knowledge_graph.contradiction_viz import ContradictionVizData

log = structlog.get_logger(__name__)

# Assertion type → color (colorblind-safe)
_ASSERTION_COLORS: dict[str, str] = {
    "mechanistic_causal": "#0072B2",
    "existence": "#009E73",
    "comparative": "#E69F00",
    "methodological": "#984EA3",
    "correlational": "#56B4E9",
    "absence": "#D55E00",
    "conditional": "#CC79A7",
}

# Distinguishable community palette (20 colors, then cycle)
_COMMUNITY_PALETTE: list[str] = [
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#42D4F4",
    "#F032E6",
    "#BFEF45",
    "#FABED4",
    "#469990",
    "#DCBEFF",
    "#9A6324",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
    "#808000",
    "#FFD8B1",
    "#000075",
    "#A9A9A9",
]


def _build_claim_graph(
    kg: nx.MultiDiGraph,
) -> tuple[list[dict], list[dict], list[set[str]], dict]:
    """Build claim-centric graph data from the entity-centric knowledge graph.

    Returns:
        (nodes, edges, communities, stats) for vis.js rendering.
        communities uses NLI-format claim IDs: ``"{u}__{predicate}__{v}__{key}"``.
    """
    # ── Step 1: Extract claims as nodes ──
    entity_to_claims: dict[str, set[str]] = defaultdict(set)
    claims: dict[str, dict] = {}

    for u, v, key, data in kg.edges(keys=True, data=True):
        subj_name = kg.nodes[u].get("canonical_name", u)
        obj_name = kg.nodes[v].get("canonical_name", v)
        predicate = data.get("predicate", "related_to")
        assertion_type = data.get("assertion_type", "existence")
        confidence = data.get("confidence_mean", 0.5)
        evidence_count = data.get("evidence_count", 1)
        controversy = data.get("controversy_score", 0.0)

        # NLI-format claim ID — aligns with contradiction_viz module
        claim_id = f"{u}__{predicate}__{v}__{key}"

        # Get evidence details from rich _kg_edge if available
        evidence_details: list[dict] = []
        kg_edge = data.get("_kg_edge")
        if kg_edge and kg_edge.evidence_links:
            for ev in kg_edge.evidence_links:
                evidence_details.append(
                    {
                        "paper_id": ev.paper_id[:8],
                        "strength": str(ev.evidence_strength),
                        "direction": ev.evidence_direction,
                        "summary": (ev.experiment_summary or "")[:120],
                    }
                )

        claims[claim_id] = {
            "id": claim_id,
            "subject_id": u,
            "object_id": v,
            "subject_name": subj_name,
            "object_name": obj_name,
            "predicate": predicate,
            "assertion_type": assertion_type,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "controversy": controversy,
            "evidence_details": evidence_details,
        }

        entity_to_claims[u].add(claim_id)
        entity_to_claims[v].add(claim_id)

    # ── Step 2: Build claim-claim edges from shared entities ──
    max_claims_per_entity = 20  # Cap: entities with 20+ claims only connect top 20
    edge_weights: dict[tuple[str, str], dict] = {}

    for entity_id, claim_ids in entity_to_claims.items():
        if len(claim_ids) < 2:
            continue
        entity_name = kg.nodes[entity_id].get("canonical_name", entity_id)
        claim_list = sorted(claim_ids)
        if len(claim_list) > max_claims_per_entity:
            log.debug(
                "claim_graph.hub_capped",
                entity=entity_id,
                original=len(claim_list),
                capped=max_claims_per_entity,
            )
            claim_list = sorted(
                claim_list,
                key=lambda cid: claims.get(cid, {}).get("evidence_count", 0),
                reverse=True,
            )[:max_claims_per_entity]
        for i in range(len(claim_list)):
            for j in range(i + 1, len(claim_list)):
                pair = (claim_list[i], claim_list[j])
                if pair not in edge_weights:
                    edge_weights[pair] = {"weight": 0, "shared_entities": []}
                edge_weights[pair]["weight"] += 1
                if len(edge_weights[pair]["shared_entities"]) < 5:
                    edge_weights[pair]["shared_entities"].append(entity_name)

    log.info(
        "claim_graph.edges_computed",
        total_claims=len(claims),
        total_edges=len(edge_weights),
    )

    # ── Step 3: Build claim-centric networkx graph for community detection ──
    claim_nx: nx.Graph = nx.Graph()
    for claim_id in claims:
        claim_nx.add_node(claim_id)
    for (c1, c2), info in edge_weights.items():
        claim_nx.add_edge(c1, c2, weight=info["weight"])

    communities: list[set[str]] = [
        set(c) for c in nx.algorithms.community.louvain_communities(claim_nx, seed=42)
    ]
    log.info("claim_graph.communities", count=len(communities))

    # Map claim -> community
    node_community: dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for cid in comm:
            node_community[cid] = idx

    # Degree in claim graph
    claim_degrees = dict(claim_nx.degree())

    # ── Step 3b: Pre-compute layout positions ──
    n_nodes = claim_nx.number_of_nodes()
    log.info("claim_graph.computing_layout", nodes=n_nodes)
    scale = 2000

    try:
        # igraph Fruchterman-Reingold is 10-100x faster than networkx
        import random as _rng

        import igraph as ig

        g = ig.Graph.from_networkx(claim_nx)
        # igraph seed expects an initial Layout, not an int — generate
        # deterministic random initial positions for reproducibility
        _rng.seed(42)
        initial = ig.Layout([(_rng.gauss(0, 1), _rng.gauss(0, 1)) for _ in range(g.vcount())])
        layout = g.layout_fruchterman_reingold(
            niter=500 if n_nodes < 2000 else 1000,
            seed=initial,
            weights="weight" if g.es and "weight" in g.es.attributes() else None,
        )
        # igraph preserves node ordering from networkx
        node_list = list(claim_nx.nodes())
        positions = {
            node_list[i]: (float(layout[i][0] * scale), float(layout[i][1] * scale))
            for i in range(len(node_list))
        }
        log.info("claim_graph.layout_complete", engine="igraph", nodes=n_nodes)
    except ImportError:
        log.warning("claim_graph.igraph_unavailable", fallback="networkx")
        pos = nx.spring_layout(
            claim_nx,
            k=1.5 / math.sqrt(max(n_nodes, 1)),
            iterations=80,
            seed=42,
            weight="weight",
        )
        positions = {nid: (float(pos[nid][0] * scale), float(pos[nid][1] * scale)) for nid in pos}

    # ── Step 4: Build vis.js node data ──
    max_evidence = max((c["evidence_count"] for c in claims.values()), default=1)

    nodes: list[dict] = []
    for cid, c in claims.items():
        degree = claim_degrees.get(cid, 0)
        comm_id = node_community.get(cid, 0)

        # Node size: scaled by evidence count (min 20, max 70)
        size = 20 + 50 * (math.log1p(c["evidence_count"]) / math.log1p(max_evidence))

        atype = c["assertion_type"]
        assertion_color = _ASSERTION_COLORS.get(atype, "#999999")
        comm_color = _COMMUNITY_PALETTE[comm_id % len(_COMMUNITY_PALETTE)]

        # Short label: truncated predicate
        short_label = c["predicate"]
        if len(short_label) > 30:
            short_label = short_label[:28] + ".."

        # Full label for search
        full_label = f"{c['subject_name']} \u2192 {c['predicate']} \u2192 {c['object_name']}"

        # Evidence summary for tooltip
        ev_html = ""
        for ev in c["evidence_details"][:3]:
            strength_short = ev["strength"].replace("_", " ").title()
            ev_html += (
                f"<div style='margin:2px 0;padding:2px 4px;background:#252833;border-radius:3px'>"
                f"<span style='color:#56B4E9'>{ev['direction']}</span> "
                f"({strength_short})<br>"
                f"<span style='color:#6a7080;font-size:11px'>{_escape_html(ev['summary'])}</span>"
                f"</div>"
            )
        if c["evidence_count"] > 3:
            ev_html += f"<div style='color:#6a7080'>+{c['evidence_count'] - 3} more</div>"

        tooltip = (
            f"<div style='font-family:Arial,sans-serif;font-size:13px;max-width:380px'>"
            f"<b style='font-size:14px'>{_escape_html(c['subject_name'])}</b>"
            f" <span style='color:#56B4E9'>\u2192 {_escape_html(c['predicate'])} \u2192</span> "
            f"<b>{_escape_html(c['object_name'])}</b><br>"
            f"<div style='margin:4px 0;padding:3px 6px;background:{assertion_color}22;"
            f"border-left:3px solid {assertion_color};border-radius:2px'>"
            f"{atype.replace('_', ' ').title()}</div>"
            f"Confidence: <b>{c['confidence']:.2f}</b> &nbsp;|&nbsp; "
            f"Evidence: <b>{c['evidence_count']}</b> &nbsp;|&nbsp; "
            f"Controversy: <b>{c['controversy']:.2f}</b><br>"
            f"Shared-entity neighbors: <b>{degree}</b> &nbsp;|&nbsp; "
            f"Community: {comm_id}"
            + ('<hr style="border-color:#2a2d3a;margin:6px 0">' + ev_html if ev_html else "")
            + "</div>"
        )

        nodes.append(
            {
                "id": cid,
                "label": short_label,
                "fullLabel": full_label,
                "subjectName": c["subject_name"],
                "objectName": c["object_name"],
                "predicate": c["predicate"],
                "size": round(size, 1),
                "title": tooltip,
                "assertionColor": assertion_color,
                "communityColor": comm_color,
                "color": assertion_color,
                "assertionType": atype,
                "communityId": comm_id,
                "confidence": round(c["confidence"], 3),
                "evidenceCount": c["evidence_count"],
                "controversy": round(c["controversy"], 3),
                "degree": degree,
                "x": positions.get(cid, (0, 0))[0],
                "y": positions.get(cid, (0, 0))[1],
                "fixed": {"x": True, "y": True},
                "font": {"size": 0, "color": "#c0c8d8"},
            }
        )

    # ── Step 5: Build vis.js edge data ──
    edges: list[dict] = []
    for idx, ((c1, c2), info) in enumerate(edge_weights.items()):
        weight = info["weight"]
        shared = info["shared_entities"]

        # Width: 0.3 to 4, scaled by shared entity count
        width = max(0.3, min(4, 0.3 + weight * 0.8))

        # Color: more shared entities = more opaque
        alpha = min(0xCC, 0x30 + weight * 0x20)
        edge_color = f"#607080{alpha:02X}"

        shared_text = ", ".join(shared)
        if weight > len(shared):
            shared_text += f" (+{weight - len(shared)} more)"

        tooltip = (
            f"<div style='font-family:Arial,sans-serif;font-size:13px'>"
            f"<b>Shared entities: {weight}</b><br>"
            f"{_escape_html(shared_text)}"
            f"</div>"
        )

        edges.append(
            {
                "from": c1,
                "to": c2,
                "id": f"se_{idx}",
                "title": tooltip,
                "width": round(width, 1),
                "color": {"color": edge_color, "highlight": "#0072B2", "hover": "#56B4E9"},
                "sharedCount": weight,
                "smooth": False,
            }
        )

    # Stats
    degrees_list = list(claim_degrees.values())
    stats = {
        "totalNodes": len(nodes),
        "totalEdges": len(edges),
        "communities": len(communities),
        "maxDegree": max(degrees_list) if degrees_list else 0,
        "avgDegree": round(sum(degrees_list) / len(degrees_list), 1) if degrees_list else 0,
    }

    return nodes, edges, communities, stats


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for tooltip content."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoReview Claim Explorer</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: Arial, Helvetica, sans-serif; background: #0f1117;
    color: #e0e0e0; display: flex; height: 100vh; overflow: hidden;
  }

  /* Sidebar */
  #sidebar {
    width: 320px; min-width: 320px; background: #1a1d27; padding: 16px;
    overflow-y: auto; border-right: 1px solid #2a2d3a;
    display: flex; flex-direction: column; gap: 14px;
  }
  #sidebar h1 {
    font-size: 16px; color: #fff;
    border-bottom: 1px solid #2a2d3a; padding-bottom: 10px;
  }
  #sidebar .subtitle { font-size: 11px; color: #6a7080; margin-top: -8px; }
  #sidebar h2 {
    font-size: 13px; color: #8890a0; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 6px;
  }

  /* Search */
  #search-box {
    width: 100%; padding: 8px 12px; background: #252833;
    border: 1px solid #3a3d4a; border-radius: 6px;
    color: #e0e0e0; font-size: 13px; outline: none;
  }
  #search-box:focus { border-color: #0072B2; }
  #search-results { max-height: 200px; overflow-y: auto; font-size: 12px; }
  .search-result {
    padding: 5px 8px; cursor: pointer; border-radius: 4px;
    line-height: 1.3; border-bottom: 1px solid #252833;
  }
  .search-result:hover { background: #252833; }
  .search-result .sr-predicate { color: #56B4E9; font-weight: bold; }
  .search-result .sr-entities { color: #8890a0; font-size: 11px; }

  /* Controls */
  .control-group { display: flex; flex-direction: column; gap: 6px; }
  .toggle-row {
    display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;
  }
  .toggle-row input[type="checkbox"], .toggle-row input[type="radio"] {
    accent-color: #0072B2; width: 16px; height: 16px;
  }

  /* Assertion type filters */
  .type-filter {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; cursor: pointer; padding: 2px 0;
  }
  .type-filter .swatch {
    width: 12px; height: 12px; border-radius: 3px;
    display: inline-block; flex-shrink: 0;
  }
  .type-filter span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .type-count { color: #6a7080; margin-left: auto; font-size: 11px; flex-shrink: 0; }

  /* Sliders */
  .slider-row { display: flex; flex-direction: column; gap: 4px; }
  .slider-row label { font-size: 12px; display: flex; justify-content: space-between; }
  input[type="range"] { width: 100%; accent-color: #0072B2; }

  /* Stats */
  #stats { font-size: 12px; color: #6a7080; line-height: 1.6; }
  #stats b { color: #a0a8b8; }

  /* Network container */
  #network-container { flex: 1; position: relative; }
  #network { width: 100%; height: 100%; }

  /* Legend overlay */
  #legend {
    position: absolute; bottom: 16px; right: 16px;
    background: rgba(26,29,39,0.92); padding: 12px 16px;
    border-radius: 8px; font-size: 12px; border: 1px solid #2a2d3a; max-width: 220px;
  }
  #legend .legend-item { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
  #legend .swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  #legend h3 {
    font-size: 11px; color: #8890a0;
    text-transform: uppercase; margin-bottom: 4px;
  }

  /* Loading overlay */
  #loading {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15,17,23,0.9); display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-size: 18px; z-index: 100; gap: 12px;
  }
  #loading .progress { font-size: 13px; color: #6a7080; }
  #loading.hidden { display: none; }

  /* Node detail panel */
  #detail-panel {
    position: absolute; top: 16px; right: 16px; width: 380px;
    background: rgba(26,29,39,0.96); padding: 16px;
    border-radius: 8px; border: 1px solid #2a2d3a;
    display: none; max-height: 70vh; overflow-y: auto;
  }
  #detail-panel h3 { font-size: 15px; color: #fff; margin-bottom: 8px; }
  #detail-panel .close-btn {
    position: absolute; top: 8px; right: 12px;
    cursor: pointer; font-size: 18px; color: #6a7080;
  }
  #detail-panel .close-btn:hover { color: #fff; }
  #detail-panel table { width: 100%; font-size: 12px; border-collapse: collapse; }
  #detail-panel td { padding: 3px 0; vertical-align: top; }
  #detail-panel td:first-child { color: #6a7080; width: 110px; }
  .claim-arrow { color: #56B4E9; margin: 0 4px; }
  .neighbor-list { margin-top: 8px; font-size: 12px; max-height: 250px; overflow-y: auto; }
  .neighbor-item {
    padding: 5px 4px; border-bottom: 1px solid #252833;
    line-height: 1.4; cursor: pointer; border-radius: 3px;
  }
  .neighbor-item:hover { background: #252833; }
  .neighbor-item:last-child { border-bottom: none; }
  .neighbor-shared { color: #6a7080; font-size: 11px; }

  /* Contradiction visualization */
  .contra-badge {
    display: inline-block; padding: 1px 6px;
    border-radius: 8px; font-size: 10px; font-weight: bold;
  }
  .contra-cross { background: #D55E0033; color: #D55E00; }
  .contra-intra { background: #CC79A733; color: #CC79A7; }
  .disagreement-item {
    padding: 6px 4px; border-bottom: 1px solid #252833;
    cursor: pointer; border-radius: 3px;
  }
  .disagreement-item:hover { background: #252833; }
  .disagreement-score { float: right; font-weight: bold; color: #D55E00; }
  .deg-btn {
    padding: 4px 10px; background: #252833; border: 1px solid #3a3d4a;
    border-radius: 4px; color: #c0c8d8; font-size: 12px; cursor: pointer;
  }
  .deg-btn:hover { background: #2a2d3a; border-color: #0072B2; }
  .deg-btn.active { background: #0072B2; border-color: #0072B2; color: #fff; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Claim Explorer</h1>
  <div class="subtitle">Nodes = claims &nbsp;|&nbsp; Edges = shared entities + contradictions</div>

  <div class="control-group">
    <h2>Search Claims</h2>
    <input id="search-box" type="text"
      placeholder="Search by entity, predicate..." autocomplete="off">
    <div id="search-results"></div>
  </div>

  <div class="control-group">
    <h2>Coloring</h2>
    <label class="toggle-row">
      <input type="radio" name="coloring" value="assertion" checked> Assertion Type</label>
    <label class="toggle-row">
      <input type="radio" name="coloring" value="community"> Community Cluster</label>
  </div>

  <div class="control-group">
    <h2>Assertion Types</h2>
    <div id="type-filters"></div>
  </div>

  <div class="control-group">
    <h2>Filters</h2>
    <div class="slider-row">
      <label>Min confidence <span id="conf-val">0.00</span></label>
      <input type="range" id="conf-slider" min="0" max="1" step="0.05" value="0">
    </div>
    <div class="slider-row">
      <label>Min evidence <span id="ev-val">0</span></label>
      <input type="range" id="ev-slider" min="0" max="10" step="1" value="0">
    </div>
    <div style="display:flex;flex-direction:column;gap:4px">
      <label style="font-size:12px">Min shared-entity neighbors</label>
      <div id="deg-buttons" style="display:flex;gap:4px;flex-wrap:wrap">
        <button class="deg-btn active" data-val="0">0</button>
        <button class="deg-btn" data-val="1">1</button>
        <button class="deg-btn" data-val="2">2+</button>
        <button class="deg-btn" data-val="5">5+</button>
        <button class="deg-btn" data-val="10">10+</button>
        <button class="deg-btn" data-val="20">20+</button>
        <button class="deg-btn" data-val="50">50+</button>
      </div>
    </div>
  </div>

  <div class="control-group">
    <h2>Display</h2>
    <label class="toggle-row">
      <input type="checkbox" id="toggle-physics" checked> Physics simulation</label>
    <label class="toggle-row"><input type="checkbox" id="toggle-labels"> Show all labels</label>
    <label class="toggle-row"><input type="checkbox" id="toggle-edges" checked> Show edges</label>
  </div>

  <div class="control-group" id="topology-group" style="display:none">
    <h2>Topology Analysis</h2>
    <label class="toggle-row">
      <input type="checkbox" id="toggle-high-impact"> Highlight high-impact nodes</label>
    <div id="topology-stats" style="font-size:11px;color:#6a7080;margin-top:4px"></div>
  </div>

  <div class="control-group" id="edge-mode-group">
    <h2>Edge Mode</h2>
    <label class="toggle-row">
      <input type="radio" name="edgeMode" value="shared" checked> Shared Entity</label>
    <label class="toggle-row">
      <input type="radio" name="edgeMode" value="contradiction"> Contradictions</label>
    <label class="toggle-row"><input type="radio" name="edgeMode" value="both"> Both</label>
    <div class="slider-row" id="contra-threshold-row" style="display:none;margin-top:8px">
      <label>Min contradiction <span id="contra-val">0.90</span></label>
      <input type="range" id="contra-slider" min="0.5" max="1" step="0.05" value="0.9">
    </div>
    <label class="toggle-row" id="cross-paper-row" style="display:none;margin-top:4px">
      <input type="checkbox" id="toggle-cross-paper"> Cross-paper only
    </label>
  </div>

  <div class="control-group">
    <h2>View</h2>
    <label class="toggle-row">
      <input type="radio" name="viewMode" value="claims" checked> Claim Graph</label>
    <label class="toggle-row">
      <input type="radio" name="viewMode" value="communities"> Community Overview</label>
  </div>

  <div class="control-group">
    <h2>Stats</h2>
    <div id="stats"></div>
  </div>

  <div class="control-group" id="disagreements-section">
    <h2>Top Community Disagreements</h2>
    <div id="disagreement-list" style="max-height:300px;overflow-y:auto;font-size:12px"></div>
  </div>
</div>

<div id="network-container">
  <div id="loading">
    Laying out claim graph&hellip;
    <div class="progress" id="loading-progress"></div>
  </div>
  <div id="network"></div>
  <div id="community-network" style="width:100%;height:100%;display:none"></div>
  <div id="legend"></div>
  <div id="detail-panel">
    <span class="close-btn" onclick="closeDetail()">&times;</span>
    <div id="detail-content"></div>
  </div>
</div>

<script>
// ── Data injected by Python ──
const RAW_NODES = __NODES_JSON__;
const RAW_EDGES = __EDGES_JSON__;
const ASSERTION_COLORS = __ASSERTION_COLORS_JSON__;
const GRAPH_STATS = __STATS_JSON__;
const CONTRADICTION_EDGES = __CONTRADICTION_EDGES_JSON__;
const COMMUNITY_LABELS = __COMMUNITY_LABELS_JSON__;
const COMMUNITY_DISAGREEMENTS = __COMMUNITY_DISAGREEMENTS_JSON__;
const TOPOLOGY_DATA = __TOPOLOGY_DATA_JSON__;

// ── State ──
let allNodes = RAW_NODES;
let allEdges = RAW_EDGES;
let network, nodesDataset, edgesDataset;
let colorMode = "assertion";
let activeAssertionTypes = new Set(Object.keys(ASSERTION_COLORS));
let minConfidence = 0;
let minEvidence = 0;
let minDegree = 0;
let showLabels = false;
let edgeMode = "shared";  // "shared", "contradiction", "both"
let contraThreshold = 0.9;
let crossPaperOnly = false;
let viewMode = "claims";  // "claims", "communities"
let communityNetwork = null;  // second vis.js instance for community overview
let showHighImpact = false;

// Index for fast lookups
const nodeIndex = {};
allNodes.forEach(n => { nodeIndex[n.id] = n; });

function initTopology() {
  const hiNodes = TOPOLOGY_DATA && TOPOLOGY_DATA.highImpactNodes;
  if (hiNodes && Object.keys(hiNodes).length > 0) {
    document.getElementById("topology-group").style.display = "";
    const s = TOPOLOGY_DATA.summary || {};
    const nHi = Object.keys(hiNodes).length;
    document.getElementById("topology-stats").innerHTML =
      `<b>${nHi}</b> high-impact nodes<br>` +
      `${s.bridges || 0} bridges &middot; ${s.articulation_points || 0} hubs`;
  }
}

// ── Initialize ──
document.addEventListener("DOMContentLoaded", () => {
  buildTypeFilters();
  buildLegend();
  updateStats();
  buildDisagreementList();
  initTopology();
  initNetwork();
  bindControls();
});

function initNetwork() {
  nodesDataset = new vis.DataSet(prepareNodes(allNodes));
  edgesDataset = new vis.DataSet(prepareEdges(allEdges));

  const container = document.getElementById("network");
  const options = {
    nodes: {
      shape: "dot",
      borderWidth: 1.5,
      borderWidthSelected: 3,
      font: { size: 0, color: "#c0c8d8", strokeWidth: 3, strokeColor: "#1a1d27" },
    },
    edges: {
      smooth: false,
      hoverWidth: 1.5,
      selectionWidth: 2,
    },
    physics: {
      enabled: false,
    },
    interaction: {
      hover: true,
      tooltipDelay: 120,
      hideEdgesOnDrag: true,
      hideEdgesOnZoom: false,
      multiselect: true,
    },
    layout: { improvedLayout: false },
  };

  network = new vis.Network(container, { nodes: nodesDataset, edges: edgesDataset }, options);

  // Layout is pre-computed — hide loading immediately
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("toggle-physics").checked = false;

  network.on("click", (params) => {
    if (params.nodes.length === 1) showDetail(params.nodes[0]);
    else if (params.edges.length === 1) showEdgeDetail(params.edges[0]);
    else closeDetail();
  });

  network.on("doubleClick", (params) => {
    if (params.nodes.length === 1) focusNeighborhood(params.nodes[0]);
  });
}

// ── Node/Edge preparation ──
function prepareNodes(nodes) {
  return nodes.filter(n => {
    if (!activeAssertionTypes.has(n.assertionType)) return false;
    if (n.confidence < minConfidence) return false;
    if (n.evidenceCount < minEvidence) return false;
    if (n.degree < minDegree) return false;
    return true;
  }).map(n => {
    const baseColor = colorMode === "assertion" ? n.assertionColor : n.communityColor;
    const labelVisible = showLabels || n.degree >= 40;
    const _hiMap = showHighImpact && TOPOLOGY_DATA.highImpactNodes;
    const hiNode = _hiMap ? _hiMap[n.id] : null;
    const borderColor = hiNode ? "#FF3333" : darken(baseColor, 30);
    const borderW = hiNode ? 3.5 : undefined;
    return {
      ...n,
      color: {
        background: baseColor,
        border: borderColor,
        highlight: { background: "#ffffff", border: hiNode ? "#FF3333" : baseColor },
        hover: { background: lighten(baseColor, 20), border: hiNode ? "#FF3333" : baseColor },
      },
      borderWidth: borderW,
      font: {
        ...n.font,
        size: labelVisible ? Math.max(9, Math.min(14, 8 + Math.log1p(n.degree) * 1.5)) : 0,
      },
    };
  });
}

function prepareEdges(edges) {
  const visibleIds = new Set(nodesDataset ? nodesDataset.getIds() : allNodes.map(n => n.id));
  return edges.filter(e => visibleIds.has(e.from) && visibleIds.has(e.to));
}

// ── Active edge builder (respects edgeMode) ──
function getActiveEdges() {
  const nodeIds = new Set(nodesDataset ? nodesDataset.getIds() : allNodes.map(n => n.id));
  let edges = [];

  if (edgeMode === "shared" || edgeMode === "both") {
    edges = edges.concat(allEdges.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to)));
  }

  if (edgeMode === "contradiction" || edgeMode === "both") {
    const MAX_CONTRA_EDGES = 500;
    const contraEdges = CONTRADICTION_EDGES
      .filter(e => e.p_contradiction >= contraThreshold)
      .filter(e => !crossPaperOnly || e.is_cross_paper)
      .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
      .slice(0, MAX_CONTRA_EDGES)
      .map((e, i) => ({
        id: `contra_${i}`,
        from: e.from,
        to: e.to,
        width: 2.5 + e.p_contradiction * 3.5,
        color: {
          color: e.is_cross_paper ? "#D55E00" : "#CC79A7",
          highlight: "#FF0000", hover: "#FF4444",
        },
        title: `<div style='font-family:Arial;font-size:13px'>`
          + `<b style='color:${e.is_cross_paper ? "#D55E00" : "#CC79A7"}'>`
          + `Contradiction (p=${e.p_contradiction.toFixed(2)})</b><br>`
          + `${e.method}${e.is_cross_paper ? ' | Cross-paper' : ''}`
          + `<br>Shared: ${e.shared_entities.join(", ")}</div>`,
        smooth: { type: "curvedCW", roundness: 0.15 },
      }));
    edges = edges.concat(contraEdges);
  }

  return edges;
}

// ── Filter & refresh ──
function refreshGraph() {
  const newNodes = prepareNodes(allNodes);
  const nodeIds = new Set(newNodes.map(n => n.id));
  nodesDataset.clear();
  nodesDataset.add(newNodes);

  const newEdges = getActiveEdges();
  edgesDataset.clear();
  edgesDataset.add(newEdges);

  updateStats();
}

// ── Controls ──
function bindControls() {
  document.querySelectorAll('input[name="coloring"]').forEach(r => {
    r.addEventListener("change", (e) => {
      colorMode = e.target.value; refreshGraph(); buildLegend();
    });
  });

  document.getElementById("conf-slider").addEventListener("input", (e) => {
    minConfidence = parseFloat(e.target.value);
    document.getElementById("conf-val").textContent = minConfidence.toFixed(2);
    refreshGraph();
  });

  document.getElementById("ev-slider").addEventListener("input", (e) => {
    minEvidence = parseInt(e.target.value);
    document.getElementById("ev-val").textContent = minEvidence;
    refreshGraph();
  });

  document.querySelectorAll(".deg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".deg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      minDegree = parseInt(btn.dataset.val);
      refreshGraph();
    });
  });

  document.getElementById("toggle-physics").addEventListener("change", (e) => {
    const enable = e.target.checked;
    // Unfix nodes so physics can move them; re-fix when disabled
    const updates = nodesDataset.getIds().map(id => ({
      id,
      fixed: { x: !enable, y: !enable },
    }));
    nodesDataset.update(updates);
    network.setOptions({
      physics: enable ? {
        enabled: true,
        solver: "barnesHut",
        barnesHut: {
          gravitationalConstant: -4000, springLength: 140,
          springConstant: 0.01, damping: 0.7,
        },
        stabilization: false,
      } : { enabled: false },
    });
  });

  document.getElementById("toggle-labels").addEventListener("change", (e) => {
    showLabels = e.target.checked;
    refreshGraph();
  });

  document.getElementById("toggle-high-impact").addEventListener("change", (e) => {
    showHighImpact = e.target.checked;
    refreshGraph();
  });

  document.getElementById("toggle-edges").addEventListener("change", (e) => {
    if (e.target.checked) {
      edgesDataset.clear();
      edgesDataset.add(getActiveEdges());
    } else {
      edgesDataset.clear();
    }
  });

  // Edge mode controls
  document.querySelectorAll('input[name="edgeMode"]').forEach(r => {
    r.addEventListener("change", (e) => {
      edgeMode = e.target.value;
      const showContraControls = edgeMode !== "shared";
      const dispMode = showContraControls ? "flex" : "none";
      document.getElementById("contra-threshold-row").style.display = dispMode;
      document.getElementById("cross-paper-row").style.display = dispMode;
      refreshGraph();
      buildLegend();
    });
  });

  document.getElementById("contra-slider").addEventListener("input", (e) => {
    contraThreshold = parseFloat(e.target.value);
    document.getElementById("contra-val").textContent = contraThreshold.toFixed(2);
    refreshGraph();
  });

  document.getElementById("toggle-cross-paper").addEventListener("change", (e) => {
    crossPaperOnly = e.target.checked;
    refreshGraph();
  });

  // View mode toggle
  document.querySelectorAll('input[name="viewMode"]').forEach(r => {
    r.addEventListener("change", (e) => {
      viewMode = e.target.value;
      if (viewMode === "communities") {
        document.getElementById("network").style.display = "none";
        document.getElementById("community-network").style.display = "block";
        initCommunityNetwork();
      } else {
        document.getElementById("network").style.display = "block";
        document.getElementById("community-network").style.display = "none";
      }
    });
  });

  // Search — matches against full label (subject → predicate → object)
  const searchBox = document.getElementById("search-box");
  searchBox.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const results = document.getElementById("search-results");
    results.innerHTML = "";
    if (q.length < 2) return;
    const matches = allNodes.filter(n =>
      n.fullLabel.toLowerCase().includes(q) ||
      n.predicate.toLowerCase().includes(q)
    ).slice(0, 20);
    matches.forEach(n => {
      const div = document.createElement("div");
      div.className = "search-result";
      div.innerHTML = `
        <span class="sr-predicate">${escapeHtml(n.predicate)}</span><br>
        <span class="sr-entities">${escapeHtml(n.subjectName)}
          &rarr; ${escapeHtml(n.objectName)}</span>
      `;
      div.addEventListener("click", () => {
        network.selectNodes([n.id]);
        network.focus(n.id, { scale: 2.0, animation: { duration: 500 } });
        showDetail(n.id);
        results.innerHTML = "";
        searchBox.value = n.predicate;
      });
      results.appendChild(div);
    });
  });
}

// ── Assertion type filters ──
function buildTypeFilters() {
  const container = document.getElementById("type-filters");
  const typeCounts = {};
  allNodes.forEach(n => { typeCounts[n.assertionType] = (typeCounts[n.assertionType] || 0) + 1; });

  const sorted = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([atype, count]) => {
    const label = document.createElement("label");
    label.className = "type-filter";
    label.innerHTML = `
      <input type="checkbox" checked data-type="${atype}">
      <span class="swatch" style="background:${ASSERTION_COLORS[atype] || '#999'}"></span>
      <span>${atype.replace(/_/g, " ")}</span>
      <span class="type-count">${count}</span>
    `;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) activeAssertionTypes.add(atype);
      else activeAssertionTypes.delete(atype);
      refreshGraph();
    });
    container.appendChild(label);
  });
}

// ── Legend ──
function buildLegend() {
  const legend = document.getElementById("legend");
  if (colorMode === "assertion") {
    const types = Object.entries(ASSERTION_COLORS);
    legend.innerHTML = `<h3>Assertion Types</h3>` + types.map(([t, c]) =>
      `<div class="legend-item">`
      + `<span class="swatch" style="background:${c}"></span>`
      + `${t.replace(/_/g, " ")}</div>`
    ).join("");
  } else {
    legend.innerHTML = `<h3>Communities</h3>`
      + `<div style="color:#6a7080;font-size:11px">`
      + `Colors represent claim<br>community clusters</div>`;
  }
  if (edgeMode === "shared") {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>`
      + `<div style="color:#6a7080;font-size:11px">Width = # shared entities</div>`;
  } else if (edgeMode === "contradiction") {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>
      <div class="legend-item">
        <span style="border-bottom:2px dashed #D55E00;width:20px;display:inline-block">
        </span> Cross-paper contradiction</div>
      <div class="legend-item">
        <span style="border-bottom:2px dashed #CC79A7;width:20px;display:inline-block">
        </span> Intra-paper contradiction</div>
      <div style="color:#6a7080;font-size:11px;margin-top:4px">
        Width = contradiction probability</div>`;
  } else {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>
      <div style="color:#6a7080;font-size:11px">Gray = shared entities<br>
      <span style="color:#D55E00">Orange</span> = cross-paper contradiction<br>
      <span style="color:#CC79A7">Pink</span> = intra-paper contradiction</div>`;
  }
}

// ── Claim detail panel ──
function showDetail(nodeId) {
  const n = nodeIndex[nodeId];
  if (!n) return;

  const panel = document.getElementById("detail-panel");
  const content = document.getElementById("detail-content");

  let html = `
    <div style="margin-bottom:12px">
      <span style="font-size:15px;font-weight:bold">${escapeHtml(n.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-size:15px;color:#56B4E9;font-weight:bold">
        ${escapeHtml(n.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-size:15px;font-weight:bold">${escapeHtml(n.objectName)}</span>
    </div>
    <table>
      <tr><td>Assertion type</td><td>
        <span style="color:${n.assertionColor}">\u25CF</span>
        ${n.assertionType.replace(/_/g, " ")}</td></tr>
      <tr><td>Confidence</td><td>${n.confidence.toFixed(3)}</td></tr>
      <tr><td>Evidence</td><td>${n.evidenceCount} source(s)</td></tr>
      <tr><td>Controversy</td><td>${n.controversy.toFixed(3)}</td></tr>
      <tr><td>Neighbors</td><td>${n.degree} claims share entities</td></tr>
      <tr><td>Community</td><td>${n.communityId}${
        COMMUNITY_LABELS[n.communityId]
          ? ' \u2014 ' + escapeHtml(
              COMMUNITY_LABELS[n.communityId].subfield
              || COMMUNITY_LABELS[n.communityId].auto_label)
          : ''
      }</td></tr>
    </table>
  `;

  // Find neighbor claims
  const connected = allEdges.filter(e => e.from === nodeId || e.to === nodeId);
  if (connected.length > 0) {
    // Sort by shared count descending
    connected.sort((a, b) => (b.sharedCount || 0) - (a.sharedCount || 0));
    html += `<h3 style="margin-top:12px;font-size:13px">`
      + `Related Claims (${connected.length})</h3><div class="neighbor-list">`;
    connected.slice(0, 30).forEach(e => {
      const otherId = e.from === nodeId ? e.to : e.from;
      const other = nodeIndex[otherId];
      if (!other) return;
      html += `<div class="neighbor-item" onclick="navigateTo('${otherId}')">
        <span style="color:${other.assertionColor}">\u25CF</span>
        <b>${escapeHtml(other.predicate)}</b><br>
        <span class="neighbor-shared">
          ${escapeHtml(other.subjectName)} &rarr; ${escapeHtml(other.objectName)}
          &nbsp;|&nbsp;
          ${e.sharedCount || 1} shared entit${(e.sharedCount || 1) === 1 ? 'y' : 'ies'}
        </span>
      </div>`;
    });
    if (connected.length > 30)
      html += `<div style="color:#6a7080;padding:4px">...and ${connected.length - 30} more</div>`;
    html += `</div>`;
  }

  content.innerHTML = html;
  panel.style.display = "block";
}

function showEdgeDetail(edgeId) {
  if (!edgeId || !edgeId.startsWith("contra_")) return;

  // Get edge data from vis.js dataset
  const edge = edgesDataset.get(edgeId);
  if (!edge) return;

  // Find matching contradiction data
  const contra = CONTRADICTION_EDGES.find(
    c => c.from === edge.from && c.to === edge.to
  );
  if (!contra) return;

  const nodeA = nodeIndex[edge.from];
  const nodeB = nodeIndex[edge.to];
  if (!nodeA || !nodeB) return;

  const panel = document.getElementById("detail-panel");
  const content = document.getElementById("detail-content");

  const pctColor = contra.p_contradiction > 0.8
    ? "#D55E00" : contra.p_contradiction > 0.5 ? "#E69F00" : "#56B4E9";
  const typeLabel = contra.is_cross_paper ? "Cross-paper" : "Intra-paper";
  const typeBadgeClass = contra.is_cross_paper ? "contra-cross" : "contra-intra";

  let html = `
    <div style="margin-bottom:12px">
      <span style="font-size:16px;font-weight:bold;color:${pctColor}">Contradiction</span>
      <span style="font-size:14px;color:${pctColor};margin-left:8px">
        p = ${contra.p_contradiction.toFixed(3)}</span>
      <span class="contra-badge ${typeBadgeClass}" style="margin-left:8px">${typeLabel}</span>
    </div>

    <div style="background:#1e2130;border-radius:6px;padding:10px;margin-bottom:8px;
      border-left:3px solid ${nodeA.assertionColor}">
      <div style="font-size:11px;color:#6a7080;margin-bottom:4px">Claim A</div>
      <span style="font-weight:bold">${escapeHtml(nodeA.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="color:#56B4E9;font-weight:bold">${escapeHtml(nodeA.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-weight:bold">${escapeHtml(nodeA.objectName)}</span>
      <div style="margin-top:4px;font-size:11px;color:#6a7080">
        <span style="color:${nodeA.assertionColor}">\u25CF</span>
        ${nodeA.assertionType.replace(/_/g, " ")}
        &nbsp;|&nbsp; Evidence: ${nodeA.evidenceCount}
        &nbsp;|&nbsp; Community: ${nodeA.communityId}
      </div>
      <div style="margin-top:4px"><a href="#"
        onclick="network.selectNodes(['${edge.from.replace(/'/g, "\\'")}']);
          network.focus('${edge.from.replace(/'/g, "\\'")}',
            {scale:2,animation:{duration:500}});
          showDetail('${edge.from.replace(/'/g, "\\'")}'); return false;"
        style="color:#56B4E9;font-size:11px;text-decoration:none">Focus this claim &rarr;</a>
      </div>
    </div>

    <div style="text-align:center;font-size:18px;color:#D55E00;margin:4px 0">&#x26A1;</div>

    <div style="background:#1e2130;border-radius:6px;padding:10px;margin-bottom:12px;
      border-left:3px solid ${nodeB.assertionColor}">
      <div style="font-size:11px;color:#6a7080;margin-bottom:4px">Claim B</div>
      <span style="font-weight:bold">${escapeHtml(nodeB.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="color:#56B4E9;font-weight:bold">${escapeHtml(nodeB.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-weight:bold">${escapeHtml(nodeB.objectName)}</span>
      <div style="margin-top:4px;font-size:11px;color:#6a7080">
        <span style="color:${nodeB.assertionColor}">\u25CF</span>
        ${nodeB.assertionType.replace(/_/g, " ")}
        &nbsp;|&nbsp; Evidence: ${nodeB.evidenceCount}
        &nbsp;|&nbsp; Community: ${nodeB.communityId}
      </div>
      <div style="margin-top:4px"><a href="#"
        onclick="network.selectNodes(['${edge.to.replace(/'/g, "\\'")}']);
          network.focus('${edge.to.replace(/'/g, "\\'")}',
            {scale:2,animation:{duration:500}});
          showDetail('${edge.to.replace(/'/g, "\\'")}'); return false;"
        style="color:#56B4E9;font-size:11px;text-decoration:none">Focus this claim &rarr;</a>
      </div>
    </div>

    <table>
      <tr><td>Method</td><td>${contra.method.replace(/_/g, " ")}</td></tr>
      <tr><td>Shared entities</td><td>${
        contra.shared_entities.length > 0
          ? escapeHtml(contra.shared_entities.join(", ")) : "none"
      }</td></tr>
    </table>
  `;

  content.innerHTML = html;
  panel.style.display = "block";
}

function navigateTo(nodeId) {
  network.selectNodes([nodeId]);
  network.focus(nodeId, { scale: 2.0, animation: { duration: 400 } });
  showDetail(nodeId);
}

function closeDetail() {
  document.getElementById("detail-panel").style.display = "none";
}

function focusNeighborhood(nodeId) {
  const connected = allEdges.filter(e => e.from === nodeId || e.to === nodeId);
  const neighborIds = new Set([nodeId]);
  connected.forEach(e => { neighborIds.add(e.from); neighborIds.add(e.to); });
  network.fit({ nodes: Array.from(neighborIds), animation: { duration: 600 } });
}

// ── Stats ──
function updateStats() {
  const visNodes = nodesDataset ? nodesDataset.length : allNodes.length;
  const visEdges = edgesDataset ? edgesDataset.length : allEdges.length;
  let statsHtml = `
    <b>${visNodes}</b> / ${allNodes.length} claims visible<br>
    <b>${visEdges}</b> / ${allEdges.length} shared-entity links<br>
    <b>${GRAPH_STATS.communities}</b> communities detected<br>
    <b>${GRAPH_STATS.maxDegree}</b> max neighbors<br>
    <b>${GRAPH_STATS.avgDegree}</b> avg neighbors
  `;
  if (CONTRADICTION_EDGES.length > 0) {
    const crossCount = CONTRADICTION_EDGES.filter(e => e.is_cross_paper).length;
    statsHtml += `<hr style="border-color:#2a2d3a;margin:6px 0">
      <b>${CONTRADICTION_EDGES.length}</b> contradictions<br>
      <b>${crossCount}</b> cross-paper<br>
      <b>${COMMUNITY_DISAGREEMENTS.length}</b> community conflicts
    `;
  }
  document.getElementById("stats").innerHTML = statsHtml;
}

// ── Community overview network ──
function initCommunityNetwork() {
  if (communityNetwork) return; // already initialized

  const container = document.getElementById("community-network");

  // ── Controversy coloring: blue (calm) → orange → red (controversial) ──
  const DISAGREE_REF = COMMUNITY_DISAGREEMENTS;

  const controversyMap = {};
  DISAGREE_REF.forEach(d => {
    controversyMap[d.comm_a_id] = (controversyMap[d.comm_a_id] || 0) + d.normalized_score;
    controversyMap[d.comm_b_id] = (controversyMap[d.comm_b_id] || 0) + d.normalized_score;
  });
  const maxControversy = Math.max(...Object.values(controversyMap), 0.001);

  function lerpRGB(a, b, t) {
    return a.map((v, i) => Math.round(v + (b[i] - v) * t));
  }
  const COLD = [86, 180, 233];   // #56B4E9 — calm blue
  const WARM = [230, 159, 0];    // #E69F00 — amber
  const HOT  = [213, 94, 0];     // #D55E00 — controversy red-orange
  function controversyColor(cid) {
    const t = Math.min((controversyMap[cid] || 0) / maxControversy, 1);
    const rgb = t < 0.5 ? lerpRGB(COLD, WARM, t * 2) : lerpRGB(WARM, HOT, (t - 0.5) * 2);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }
  function darkenRGB(color) {
    return color.replace(/\d+/g, (m, i) => i < 3 ? Math.round(Number(m) * 0.7) : m);
  }

  // Build super-nodes from community labels
  const commNodes = Object.values(COMMUNITY_LABELS).map(cl => {
    const bg = controversyColor(cl.community_id);
    const cScore = (controversyMap[cl.community_id] || 0);
    return {
      id: cl.community_id,
      value: cl.claim_count,
      label: cl.subfield || cl.auto_label,
      title: `<div style='font-family:Arial;font-size:13px'>`
        + `<b>${escapeHtml(cl.subfield || cl.auto_label)}</b><br>`
        + `${cl.claim_count} claims`
        + `${cScore > 0 ? '<br>Controversy: ' + cScore.toFixed(2) : ''}`
        + `<br>Top entities: ${cl.top_entities.slice(0, 3).join(", ")}</div>`,
      color: {
        background: bg, border: darkenRGB(bg),
        highlight: { background: "#FFD700", border: "#FFA500" },
      },
      font: { color: "#e0e0e0", strokeWidth: 3, strokeColor: "rgba(26,29,39,0.8)" },
    };
  });

  // Build edges from disagreements — only show top disagreements to avoid clutter
  const maxScore = COMMUNITY_DISAGREEMENTS.length > 0
    ? COMMUNITY_DISAGREEMENTS[0].normalized_score : 1;
  const commEdges = COMMUNITY_DISAGREEMENTS.slice(0, 100).map((d, i) => ({
    id: `cd_${i}`,
    from: d.comm_a_id,
    to: d.comm_b_id,
    width: 1 + (d.normalized_score / maxScore) * 8,
    color: {
      color: `rgba(213,94,0,${0.3 + 0.7 * d.normalized_score / maxScore})`,
      highlight: "#FF0000",
    },
    title: `<div style='font-family:Arial;font-size:13px'>`
      + `<b style='color:#D55E00'>Disagreement Score: ${d.normalized_score.toFixed(2)}</b><br>`
      + `${d.num_contradictions} contradicting claim pairs<br>`
      + `<b>${escapeHtml(COMMUNITY_LABELS[d.comm_a_id]?.subfield || d.comm_a_label)}</b>`
      + ` vs<br>`
      + `<b>${escapeHtml(COMMUNITY_LABELS[d.comm_b_id]?.subfield || d.comm_b_label)}</b></div>`,
    label: d.normalized_score.toFixed(1),
    font: { size: 10, color: "#D55E00", strokeWidth: 3, strokeColor: "#1a1d27" },
  }));

  communityNetwork = new vis.Network(container, {
    nodes: new vis.DataSet(commNodes),
    edges: new vis.DataSet(commEdges),
  }, {
    nodes: {
      shape: "dot",
      borderWidth: 2,
      scaling: {
        min: 5,
        max: 60,
        label: { enabled: true, min: 0, max: 30, drawThreshold: 8, maxVisible: 30 },
      },
    },
    edges: { smooth: { type: "continuous" } },
    physics: {
      solver: "barnesHut",
      barnesHut: {
        gravitationalConstant: -3000,
        centralGravity: 0.3,
        springLength: 150,
        springConstant: 0.04,
        damping: 0.5,
        avoidOverlap: 0.5,
      },
      stabilization: { iterations: 500 },
    },
    interaction: { hover: true, tooltipDelay: 100 },
  });
  // ── Community node click → detail panel ──
  communityNetwork.on("click", (params) => {
    if (params.nodes.length > 0) {
      showCommunityDetail(params.nodes[0]);
    } else {
      document.getElementById("detail-panel").style.display = "none";
    }
  });

  function showCommunityDetail(communityId) {
    const cl = COMMUNITY_LABELS[communityId];
    if (!cl) return;

    const panel = document.getElementById("detail-panel");
    const content = document.getElementById("detail-content");
    const cScore = (controversyMap[communityId] || 0);

    let html = `
      <div style="margin-bottom:12px">
        <span style="font-size:16px;font-weight:bold;color:#fff">
          ${escapeHtml(cl.subfield || cl.auto_label)}</span>
        ${cl.subfield
          ? '<br><span style="color:#6a7080;font-size:12px">'
            + escapeHtml(cl.auto_label) + '</span>'
          : ''}
      </div>
      <table>
        <tr><td>Claims</td><td>${cl.claim_count}</td></tr>
        <tr><td>Controversy</td>
          <td style="color:${cScore > 0 ? '#D55E00' : '#6a7080'}">${cScore.toFixed(2)}</td></tr>
        <tr><td>Top entities</td><td>${cl.top_entities.join(", ")}</td></tr>
        <tr><td>Predicates</td><td>${cl.top_predicates.join(", ")}</td></tr>
        ${cl.top_entity_types && cl.top_entity_types.length
          ? '<tr><td>Entity types</td><td>' + cl.top_entity_types.join(", ") + '</td></tr>'
          : ''}
      </table>
    `;

    // Disagreements involving this community
    const disagreements = COMMUNITY_DISAGREEMENTS.filter(d =>
      d.comm_a_id === communityId || d.comm_b_id === communityId
    ).sort((a, b) => b.normalized_score - a.normalized_score);

    if (disagreements.length > 0) {
      html += '<h3 style="margin-top:14px;font-size:13px;color:#fff">Disagreements ('
        + disagreements.length + ')</h3><div class="neighbor-list">';
      disagreements.slice(0, 15).forEach(d => {
        const otherId = d.comm_a_id === communityId ? d.comm_b_id : d.comm_a_id;
        const other = COMMUNITY_LABELS[otherId];
        const otherName = other ? (other.subfield || other.auto_label) : 'Community ' + otherId;
        html += '<div class="neighbor-item"'
          + ' onclick="event.stopPropagation();'
          + ' window._navComm && window._navComm(' + otherId + ')">'
          + '<span style="color:#D55E00">●</span> '
          + '<b>' + escapeHtml(otherName) + '</b>'
          + '<span style="float:right;color:#D55E00;font-size:11px">'
          + d.normalized_score.toFixed(2) + '</span><br>'
          + '<span class="neighbor-shared">' + d.num_contradictions + ' contradictions</span>'
          + '</div>';

        // Top 2 contradicting claim pairs
        if (d.top_claims && d.top_claims.length > 0) {
          d.top_claims.slice(0, 2).forEach(tc => {
            html += '<div style="margin:2px 0 6px 16px;font-size:11px;'
              + 'color:#6a7080;line-height:1.3">'
              + '“' + escapeHtml(tc.claim_a_text) + '”<br>'
              + '<span style="color:#D55E00;margin-left:8px">vs</span> '
              + '“' + escapeHtml(tc.claim_b_text) + '”'
              + ' <span style="color:#D55E00">(' + tc.p_contradiction.toFixed(2) + ')</span>'
              + '</div>';
          });
        }
      });
      if (disagreements.length > 15) {
        html += '<div style="color:#6a7080;padding:4px">...and '
          + (disagreements.length - 15) + ' more</div>';
      }
      html += '</div>';
    } else {
      html += '<div style="margin-top:12px;color:#6a7080;font-size:12px">'
        + 'No disagreements with other communities</div>';
    }

    content.innerHTML = html;
    panel.style.display = "block";
  }

  // Expose navigation for onclick handlers in innerHTML
  window._navComm = function(communityId) {
    communityNetwork.selectNodes([communityId]);
    communityNetwork.focus(communityId, { scale: 1.5, animation: { duration: 400 } });
    showCommunityDetail(communityId);
  };
}

// ── Disagreement list ──
function buildDisagreementList() {
  const container = document.getElementById("disagreement-list");
  if (!COMMUNITY_DISAGREEMENTS.length) {
    container.innerHTML = '<div style="color:#6a7080">No contradiction data available</div>';
    return;
  }

  COMMUNITY_DISAGREEMENTS.slice(0, 20).forEach(d => {
    const labelA = COMMUNITY_LABELS[d.comm_a_id]?.subfield || d.comm_a_label;
    const labelB = COMMUNITY_LABELS[d.comm_b_id]?.subfield || d.comm_b_label;
    const div = document.createElement("div");
    div.className = "disagreement-item";
    div.innerHTML = `
      <span class="disagreement-score">${d.normalized_score.toFixed(2)}</span>
      <b>${escapeHtml(labelA)}</b><br>
      <span style="color:#D55E00">vs</span> <b>${escapeHtml(labelB)}</b><br>
      <span style="color:#6a7080;font-size:11px">${d.num_contradictions} contradictions</span>
    `;
    div.addEventListener("click", () => {
      // Switch to community view and focus on these two communities
      const vmInput = document.querySelector('input[name="viewMode"][value="communities"]');
      vmInput.checked = true;
      viewMode = "communities";
      document.getElementById("network").style.display = "none";
      document.getElementById("community-network").style.display = "block";
      initCommunityNetwork();
      setTimeout(() => {
        communityNetwork.fit({ nodes: [d.comm_a_id, d.comm_b_id], animation: { duration: 600 } });
      }, 500);
    });
    container.appendChild(div);
  });
}

// ── Utilities ──
function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function darken(hex, pct) {
  const num = parseInt(hex.replace("#",""), 16);
  const r = Math.max(0, (num >> 16) - pct);
  const g = Math.max(0, ((num >> 8) & 0xFF) - pct);
  const b = Math.max(0, (num & 0xFF) - pct);
  return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
}

function lighten(hex, pct) {
  const num = parseInt(hex.replace("#",""), 16);
  const r = Math.min(255, (num >> 16) + pct);
  const g = Math.min(255, ((num >> 8) & 0xFF) + pct);
  const b = Math.min(255, (num & 0xFF) + pct);
  return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
}
</script>
</body>
</html>"""


_MULTI_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoReview Claim Explorer</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: Arial, Helvetica, sans-serif; background: #0f1117;
    color: #e0e0e0; display: flex; height: 100vh; overflow: hidden;
  }

  /* Sidebar */
  #sidebar {
    width: 320px; min-width: 320px; background: #1a1d27; padding: 16px;
    overflow-y: auto; border-right: 1px solid #2a2d3a;
    display: flex; flex-direction: column; gap: 14px;
  }
  #sidebar h1 {
    font-size: 16px; color: #fff;
    border-bottom: 1px solid #2a2d3a; padding-bottom: 10px;
  }
  #sidebar .subtitle { font-size: 11px; color: #6a7080; margin-top: -8px; }
  #sidebar h2 {
    font-size: 13px; color: #8890a0; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 6px;
  }

  /* Search */
  #search-box {
    width: 100%; padding: 8px 12px; background: #252833;
    border: 1px solid #3a3d4a; border-radius: 6px;
    color: #e0e0e0; font-size: 13px; outline: none;
  }
  #search-box:focus { border-color: #0072B2; }
  #search-results { max-height: 200px; overflow-y: auto; font-size: 12px; }
  .search-result {
    padding: 5px 8px; cursor: pointer; border-radius: 4px;
    line-height: 1.3; border-bottom: 1px solid #252833;
  }
  .search-result:hover { background: #252833; }
  .search-result .sr-predicate { color: #56B4E9; font-weight: bold; }
  .search-result .sr-entities { color: #8890a0; font-size: 11px; }

  /* Controls */
  .control-group { display: flex; flex-direction: column; gap: 6px; }
  .toggle-row {
    display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;
  }
  .toggle-row input[type="checkbox"], .toggle-row input[type="radio"] {
    accent-color: #0072B2; width: 16px; height: 16px;
  }

  /* Assertion type filters */
  .type-filter {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; cursor: pointer; padding: 2px 0;
  }
  .type-filter .swatch {
    width: 12px; height: 12px; border-radius: 3px;
    display: inline-block; flex-shrink: 0;
  }
  .type-filter span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .type-count { color: #6a7080; margin-left: auto; font-size: 11px; flex-shrink: 0; }

  /* Sliders */
  .slider-row { display: flex; flex-direction: column; gap: 4px; }
  .slider-row label { font-size: 12px; display: flex; justify-content: space-between; }
  input[type="range"] { width: 100%; accent-color: #0072B2; }

  /* Stats */
  #stats { font-size: 12px; color: #6a7080; line-height: 1.6; }
  #stats b { color: #a0a8b8; }

  /* Network container */
  #network-container { flex: 1; position: relative; }
  #network { width: 100%; height: calc(100% - 44px); }

  /* Legend overlay */
  #legend {
    position: absolute; bottom: 60px; right: 16px;
    background: rgba(26,29,39,0.92); padding: 12px 16px;
    border-radius: 8px; font-size: 12px; border: 1px solid #2a2d3a; max-width: 220px;
  }
  #legend .legend-item { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
  #legend .swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  #legend h3 {
    font-size: 11px; color: #8890a0;
    text-transform: uppercase; margin-bottom: 4px;
  }

  /* Loading overlay */
  #loading {
    position: absolute; top: 0; left: 0; right: 0; bottom: 44px;
    background: rgba(15,17,23,0.9); display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-size: 18px; z-index: 100; gap: 12px;
  }
  #loading .progress { font-size: 13px; color: #6a7080; }
  #loading.hidden { display: none; }

  /* Node detail panel */
  #detail-panel {
    position: absolute; top: 16px; right: 16px; width: 380px;
    background: rgba(26,29,39,0.96); padding: 16px;
    border-radius: 8px; border: 1px solid #2a2d3a;
    display: none; max-height: 70vh; overflow-y: auto;
  }
  #detail-panel h3 { font-size: 15px; color: #fff; margin-bottom: 8px; }
  #detail-panel .close-btn {
    position: absolute; top: 8px; right: 12px;
    cursor: pointer; font-size: 18px; color: #6a7080;
  }
  #detail-panel .close-btn:hover { color: #fff; }
  #detail-panel table { width: 100%; font-size: 12px; border-collapse: collapse; }
  #detail-panel td { padding: 3px 0; vertical-align: top; }
  #detail-panel td:first-child { color: #6a7080; width: 110px; }
  .claim-arrow { color: #56B4E9; margin: 0 4px; }
  .neighbor-list { margin-top: 8px; font-size: 12px; max-height: 250px; overflow-y: auto; }
  .neighbor-item {
    padding: 5px 4px; border-bottom: 1px solid #252833;
    line-height: 1.4; cursor: pointer; border-radius: 3px;
  }
  .neighbor-item:hover { background: #252833; }
  .neighbor-item:last-child { border-bottom: none; }
  .neighbor-shared { color: #6a7080; font-size: 11px; }

  /* Contradiction visualization */
  .contra-badge {
    display: inline-block; padding: 1px 6px;
    border-radius: 8px; font-size: 10px; font-weight: bold;
  }
  .contra-cross { background: #D55E0033; color: #D55E00; }
  .contra-intra { background: #CC79A733; color: #CC79A7; }
  .disagreement-item {
    padding: 6px 4px; border-bottom: 1px solid #252833;
    cursor: pointer; border-radius: 3px;
  }
  .disagreement-item:hover { background: #252833; }
  .disagreement-score { float: right; font-weight: bold; color: #D55E00; }
  .deg-btn {
    padding: 4px 10px; background: #252833; border: 1px solid #3a3d4a;
    border-radius: 4px; color: #c0c8d8; font-size: 12px; cursor: pointer;
  }
  .deg-btn:hover { background: #2a2d3a; border-color: #0072B2; }
  .deg-btn.active { background: #0072B2; border-color: #0072B2; color: #fff; }

  /* Tab bar */
  #tab-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 44px;
    background: #1a1d27;
    border-top: 1px solid #2a2d3a;
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 4px;
    overflow-x: auto;
    z-index: 50;
  }
  #tab-bar::-webkit-scrollbar { height: 3px; }
  #tab-bar::-webkit-scrollbar-thumb { background: #3a3d4a; border-radius: 2px; }
  .tab-btn {
    padding: 8px 18px;
    background: #252833;
    border: 1px solid #3a3d4a;
    border-radius: 6px 6px 0 0;
    color: #8890a0;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: background 0.15s, color 0.15s;
  }
  .tab-btn:hover { background: #2a2d3a; color: #c0c8d8; }
  .tab-btn.active { background: #0072B2; border-color: #0072B2; color: #fff; }

  /* Community network respects tab bar height too */
  #community-network { width: 100%; height: calc(100% - 44px); display: none; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Claim Explorer</h1>
  <div class="subtitle">Nodes = claims &nbsp;|&nbsp; Edges = shared entities + contradictions</div>

  <div class="control-group">
    <h2>Search Claims</h2>
    <input id="search-box" type="text"
      placeholder="Search by entity, predicate..." autocomplete="off">
    <div id="search-results"></div>
  </div>

  <div class="control-group">
    <h2>Coloring</h2>
    <label class="toggle-row">
      <input type="radio" name="coloring" value="assertion" checked> Assertion Type</label>
    <label class="toggle-row">
      <input type="radio" name="coloring" value="community"> Community Cluster</label>
  </div>

  <div class="control-group">
    <h2>Assertion Types</h2>
    <div id="type-filters"></div>
  </div>

  <div class="control-group">
    <h2>Filters</h2>
    <div class="slider-row">
      <label>Min confidence <span id="conf-val">0.00</span></label>
      <input type="range" id="conf-slider" min="0" max="1" step="0.05" value="0">
    </div>
    <div class="slider-row">
      <label>Min evidence <span id="ev-val">0</span></label>
      <input type="range" id="ev-slider" min="0" max="10" step="1" value="0">
    </div>
    <div style="display:flex;flex-direction:column;gap:4px">
      <label style="font-size:12px">Min shared-entity neighbors</label>
      <div id="deg-buttons" style="display:flex;gap:4px;flex-wrap:wrap">
        <button class="deg-btn active" data-val="0">0</button>
        <button class="deg-btn" data-val="1">1</button>
        <button class="deg-btn" data-val="2">2+</button>
        <button class="deg-btn" data-val="5">5+</button>
        <button class="deg-btn" data-val="10">10+</button>
        <button class="deg-btn" data-val="20">20+</button>
        <button class="deg-btn" data-val="50">50+</button>
      </div>
    </div>
  </div>

  <div class="control-group">
    <h2>Display</h2>
    <label class="toggle-row">
      <input type="checkbox" id="toggle-physics" checked> Physics simulation</label>
    <label class="toggle-row"><input type="checkbox" id="toggle-labels"> Show all labels</label>
    <label class="toggle-row"><input type="checkbox" id="toggle-edges" checked> Show edges</label>
  </div>

  <div class="control-group" id="topology-group" style="display:none">
    <h2>Topology Analysis</h2>
    <label class="toggle-row">
      <input type="checkbox" id="toggle-high-impact"> Highlight high-impact nodes</label>
    <div id="topology-stats" style="font-size:11px;color:#6a7080;margin-top:4px"></div>
  </div>

  <div class="control-group" id="edge-mode-group">
    <h2>Edge Mode</h2>
    <label class="toggle-row">
      <input type="radio" name="edgeMode" value="shared" checked> Shared Entity</label>
    <label class="toggle-row">
      <input type="radio" name="edgeMode" value="contradiction"> Contradictions</label>
    <label class="toggle-row"><input type="radio" name="edgeMode" value="both"> Both</label>
    <div class="slider-row" id="contra-threshold-row" style="display:none;margin-top:8px">
      <label>Min contradiction <span id="contra-val">0.90</span></label>
      <input type="range" id="contra-slider" min="0.5" max="1" step="0.05" value="0.9">
    </div>
    <label class="toggle-row" id="cross-paper-row" style="display:none;margin-top:4px">
      <input type="checkbox" id="toggle-cross-paper"> Cross-paper only
    </label>
  </div>

  <div class="control-group">
    <h2>View</h2>
    <label class="toggle-row">
      <input type="radio" name="viewMode" value="claims" checked> Claim Graph</label>
    <label class="toggle-row">
      <input type="radio" name="viewMode" value="communities"> Community Overview</label>
  </div>

  <div class="control-group">
    <h2>Stats</h2>
    <div id="stats"></div>
  </div>

  <div class="control-group" id="disagreements-section">
    <h2>Top Community Disagreements</h2>
    <div id="disagreement-list" style="max-height:300px;overflow-y:auto;font-size:12px"></div>
  </div>
</div>

<div id="network-container">
  <div id="loading">
    Laying out claim graph&hellip;
    <div class="progress" id="loading-progress"></div>
  </div>
  <div id="network"></div>
  <div id="community-network"></div>
  <div id="legend"></div>
  <div id="detail-panel">
    <span class="close-btn" onclick="closeDetail()">&times;</span>
    <div id="detail-content"></div>
  </div>
  <div id="tab-bar"></div>
</div>

<script>
// ── Data injected by Python ──
const GRAPH_TABS = __GRAPH_TABS_JSON__;
const ASSERTION_COLORS = __ASSERTION_COLORS_JSON__;

// ── Tab state ──
let currentTabIndex = 0;

// ── Per-tab mutable globals (populated by loadTab) ──
let allNodes = [];
let allEdges = [];
let GRAPH_STATS_CURRENT = {};
let CONTRADICTION_EDGES_CURRENT = [];
let COMMUNITY_LABELS_CURRENT = {};
let COMMUNITY_DISAGREEMENTS_CURRENT = [];
let TOPOLOGY_DATA_CURRENT = {};

// ── Vis.js state ──
let network, nodesDataset, edgesDataset;
let colorMode = "assertion";
let activeAssertionTypes = new Set(Object.keys(ASSERTION_COLORS));
let minConfidence = 0;
let minEvidence = 0;
let minDegree = 0;
let showLabels = false;
let edgeMode = "shared";  // "shared", "contradiction", "both"
let contraThreshold = 0.9;
let crossPaperOnly = false;
let viewMode = "claims";  // "claims", "communities"
let communityNetwork = null;  // second vis.js instance for community overview
let showHighImpact = false;

// Index for fast lookups
const nodeIndex = {};

// ── Tab management ──
function loadTab(index) {
  currentTabIndex = index;
  const tab = GRAPH_TABS[index];

  allNodes = tab.nodes;
  allEdges = tab.edges;
  GRAPH_STATS_CURRENT = tab.stats;
  CONTRADICTION_EDGES_CURRENT = tab.contradictionEdges;
  COMMUNITY_LABELS_CURRENT = tab.communityLabels;
  COMMUNITY_DISAGREEMENTS_CURRENT = tab.communityDisagreements;
  TOPOLOGY_DATA_CURRENT = tab.topologyData || {};

  // Rebuild node index
  Object.keys(nodeIndex).forEach(k => delete nodeIndex[k]);
  allNodes.forEach(n => { nodeIndex[n.id] = n; });

  // Rebuild assertion type set from new data
  // (keep current checked state for types present in new data)
  const newTypes = new Set(allNodes.map(n => n.assertionType));
  // Preserve active types that exist in new data; add any new types as active
  activeAssertionTypes = new Set(
    [...activeAssertionTypes, ...newTypes].filter(t => newTypes.has(t))
  );

  // Refresh vis.js data
  refreshGraph();

  // Rebuild sidebar sections that depend on data
  document.getElementById("type-filters").innerHTML = "";
  buildTypeFilters();
  buildLegend();
  buildDisagreementList();

  // Reset community network
  if (communityNetwork) {
    communityNetwork.destroy();
    communityNetwork = null;
  }

  // Apply tab-specific defaults (colorMode, edgeMode)
  if (tab.defaultColorMode) {
    colorMode = tab.defaultColorMode;
    document.querySelector(
      `input[name="coloring"][value="${colorMode}"]`).checked = true;
  }
  if (tab.defaultEdgeMode) {
    edgeMode = tab.defaultEdgeMode;
    document.querySelector(`input[name="edgeMode"][value="${edgeMode}"]`).checked = true;
    const showContraControls = edgeMode !== "shared";
    document.getElementById("contra-threshold-row").style.display =
      showContraControls ? "flex" : "none";
    document.getElementById("cross-paper-row").style.display =
      showContraControls ? "flex" : "none";
  }

  // If community view was active, reset to claim view
  if (viewMode === "communities") {
    viewMode = "claims";
    document.querySelector('input[name="viewMode"][value="claims"]').checked = true;
    document.getElementById("network").style.display = "block";
    document.getElementById("community-network").style.display = "none";
  }

  // Update tab button styles
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {
    btn.classList.toggle('active', i === index);
  });

  // Close detail panel — data changed
  closeDetail();

  // Fit to new data
  if (network) {
    network.fit({ animation: { duration: 400 } });
  }

  initTopology();
}

function buildTabBar() {
  const bar = document.getElementById("tab-bar");
  // Only show tab bar when there are 2+ tabs
  if (GRAPH_TABS.length < 2) {
    bar.style.display = "none";
    // Also restore full height for network and community-network
    document.getElementById("network").style.height = "100%";
    document.getElementById("community-network").style.height = "100%";
    document.getElementById("loading").style.bottom = "0";
    return;
  }
  GRAPH_TABS.forEach((tab, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.textContent = tab.label;
    btn.addEventListener("click", () => loadTab(i));
    bar.appendChild(btn);
  });
}

function initTopology() {
  const td = TOPOLOGY_DATA_CURRENT;
  const group = document.getElementById("topology-group");
  if (td && td.highImpactNodes && Object.keys(td.highImpactNodes).length > 0) {
    group.style.display = "";
    const s = td.summary || {};
    const nHi = Object.keys(td.highImpactNodes).length;
    document.getElementById("topology-stats").innerHTML =
      `<b>${nHi}</b> high-impact nodes<br>` +
      `${s.bridges || 0} bridges &middot; ${s.articulation_points || 0} hubs`;
  } else {
    group.style.display = "none";
  }
}

// ── Initialize ──
document.addEventListener("DOMContentLoaded", () => {
  buildTabBar();
  // Load first tab data before initializing network
  const firstTab = GRAPH_TABS[0];
  allNodes = firstTab.nodes;
  allEdges = firstTab.edges;
  GRAPH_STATS_CURRENT = firstTab.stats;
  CONTRADICTION_EDGES_CURRENT = firstTab.contradictionEdges;
  COMMUNITY_LABELS_CURRENT = firstTab.communityLabels;
  COMMUNITY_DISAGREEMENTS_CURRENT = firstTab.communityDisagreements;
  TOPOLOGY_DATA_CURRENT = firstTab.topologyData || {};
  allNodes.forEach(n => { nodeIndex[n.id] = n; });
  activeAssertionTypes = new Set(allNodes.map(n => n.assertionType));

  buildTypeFilters();
  buildLegend();
  updateStats();
  buildDisagreementList();
  initTopology();
  initNetwork();
  bindControls();
});

function initNetwork() {
  nodesDataset = new vis.DataSet(prepareNodes(allNodes));
  edgesDataset = new vis.DataSet(prepareEdges(allEdges));

  const container = document.getElementById("network");
  const options = {
    nodes: {
      shape: "dot",
      borderWidth: 1.5,
      borderWidthSelected: 3,
      font: { size: 0, color: "#c0c8d8", strokeWidth: 3, strokeColor: "#1a1d27" },
    },
    edges: {
      smooth: false,
      hoverWidth: 1.5,
      selectionWidth: 2,
    },
    physics: {
      enabled: false,
    },
    interaction: {
      hover: true,
      tooltipDelay: 120,
      hideEdgesOnDrag: true,
      hideEdgesOnZoom: false,
      multiselect: true,
    },
    layout: { improvedLayout: false },
  };

  network = new vis.Network(container, { nodes: nodesDataset, edges: edgesDataset }, options);

  // Layout is pre-computed — hide loading immediately
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("toggle-physics").checked = false;

  network.on("click", (params) => {
    if (params.nodes.length === 1) showDetail(params.nodes[0]);
    else if (params.edges.length === 1) showEdgeDetail(params.edges[0]);
    else closeDetail();
  });

  network.on("doubleClick", (params) => {
    if (params.nodes.length === 1) focusNeighborhood(params.nodes[0]);
  });
}

// ── Node/Edge preparation ──
function prepareNodes(nodes) {
  return nodes.filter(n => {
    if (!activeAssertionTypes.has(n.assertionType)) return false;
    if (n.confidence < minConfidence) return false;
    if (n.evidenceCount < minEvidence) return false;
    if (n.degree < minDegree) return false;
    return true;
  }).map(n => {
    const baseColor = colorMode === "assertion" ? n.assertionColor : n.communityColor;
    const labelVisible = showLabels || n.degree >= 40;
    const hiNodes = TOPOLOGY_DATA_CURRENT.highImpactNodes;
    const hiNode = showHighImpact && hiNodes ? hiNodes[n.id] : null;
    const borderColor = hiNode ? "#FF3333" : darken(baseColor, 30);
    const borderW = hiNode ? 3.5 : undefined;
    return {
      ...n,
      color: {
        background: baseColor,
        border: borderColor,
        highlight: { background: "#ffffff", border: hiNode ? "#FF3333" : baseColor },
        hover: { background: lighten(baseColor, 20), border: hiNode ? "#FF3333" : baseColor },
      },
      borderWidth: borderW,
      font: {
        ...n.font,
        size: labelVisible ? Math.max(9, Math.min(14, 8 + Math.log1p(n.degree) * 1.5)) : 0,
      },
    };
  });
}

function prepareEdges(edges) {
  const visibleIds = new Set(nodesDataset ? nodesDataset.getIds() : allNodes.map(n => n.id));
  return edges.filter(e => visibleIds.has(e.from) && visibleIds.has(e.to));
}

// ── Active edge builder (respects edgeMode) ──
function getActiveEdges() {
  const nodeIds = new Set(nodesDataset ? nodesDataset.getIds() : allNodes.map(n => n.id));
  let edges = [];

  if (edgeMode === "shared" || edgeMode === "both") {
    edges = edges.concat(allEdges.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to)));
  }

  if (edgeMode === "contradiction" || edgeMode === "both") {
    const MAX_CONTRA_EDGES = 500;
    const contraEdges = CONTRADICTION_EDGES_CURRENT
      .filter(e => e.p_contradiction >= contraThreshold)
      .filter(e => !crossPaperOnly || e.is_cross_paper)
      .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
      .slice(0, MAX_CONTRA_EDGES)
      .map((e, i) => ({
        id: `contra_${i}`,
        from: e.from,
        to: e.to,
        width: 2.5 + e.p_contradiction * 3.5,
        color: {
          color: e.is_cross_paper ? "#D55E00" : "#CC79A7",
          highlight: "#FF0000", hover: "#FF4444",
        },
        title: `<div style='font-family:Arial;font-size:13px'>`
          + `<b style='color:${e.is_cross_paper ? "#D55E00" : "#CC79A7"}'>`
          + `Contradiction (p=${e.p_contradiction.toFixed(2)})</b><br>`
          + `${e.method}${e.is_cross_paper ? ' | Cross-paper' : ''}`
          + `<br>Shared: ${e.shared_entities.join(", ")}</div>`,
        smooth: { type: "curvedCW", roundness: 0.15 },
      }));
    edges = edges.concat(contraEdges);
  }

  return edges;
}

// ── Filter & refresh ──
function refreshGraph() {
  const newNodes = prepareNodes(allNodes);
  const nodeIds = new Set(newNodes.map(n => n.id));
  nodesDataset.clear();
  nodesDataset.add(newNodes);

  const newEdges = getActiveEdges();
  edgesDataset.clear();
  edgesDataset.add(newEdges);

  updateStats();
}

// ── Controls ──
function bindControls() {
  document.querySelectorAll('input[name="coloring"]').forEach(r => {
    r.addEventListener("change", (e) => {
      colorMode = e.target.value; refreshGraph(); buildLegend();
    });
  });

  document.getElementById("conf-slider").addEventListener("input", (e) => {
    minConfidence = parseFloat(e.target.value);
    document.getElementById("conf-val").textContent = minConfidence.toFixed(2);
    refreshGraph();
  });

  document.getElementById("ev-slider").addEventListener("input", (e) => {
    minEvidence = parseInt(e.target.value);
    document.getElementById("ev-val").textContent = minEvidence;
    refreshGraph();
  });

  document.querySelectorAll(".deg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".deg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      minDegree = parseInt(btn.dataset.val);
      refreshGraph();
    });
  });

  document.getElementById("toggle-physics").addEventListener("change", (e) => {
    const enable = e.target.checked;
    // Unfix nodes so physics can move them; re-fix when disabled
    const updates = nodesDataset.getIds().map(id => ({
      id,
      fixed: { x: !enable, y: !enable },
    }));
    nodesDataset.update(updates);
    network.setOptions({
      physics: enable ? {
        enabled: true,
        solver: "barnesHut",
        barnesHut: {
          gravitationalConstant: -4000, springLength: 140,
          springConstant: 0.01, damping: 0.7,
        },
        stabilization: false,
      } : { enabled: false },
    });
  });

  document.getElementById("toggle-labels").addEventListener("change", (e) => {
    showLabels = e.target.checked;
    refreshGraph();
  });

  document.getElementById("toggle-high-impact").addEventListener("change", (e) => {
    showHighImpact = e.target.checked;
    refreshGraph();
  });

  document.getElementById("toggle-edges").addEventListener("change", (e) => {
    if (e.target.checked) {
      edgesDataset.clear();
      edgesDataset.add(getActiveEdges());
    } else {
      edgesDataset.clear();
    }
  });

  // Edge mode controls
  document.querySelectorAll('input[name="edgeMode"]').forEach(r => {
    r.addEventListener("change", (e) => {
      edgeMode = e.target.value;
      const showContraControls = edgeMode !== "shared";
      const dispMode = showContraControls ? "flex" : "none";
      document.getElementById("contra-threshold-row").style.display = dispMode;
      document.getElementById("cross-paper-row").style.display = dispMode;
      refreshGraph();
      buildLegend();
    });
  });

  document.getElementById("contra-slider").addEventListener("input", (e) => {
    contraThreshold = parseFloat(e.target.value);
    document.getElementById("contra-val").textContent = contraThreshold.toFixed(2);
    refreshGraph();
  });

  document.getElementById("toggle-cross-paper").addEventListener("change", (e) => {
    crossPaperOnly = e.target.checked;
    refreshGraph();
  });

  // View mode toggle
  document.querySelectorAll('input[name="viewMode"]').forEach(r => {
    r.addEventListener("change", (e) => {
      viewMode = e.target.value;
      if (viewMode === "communities") {
        document.getElementById("network").style.display = "none";
        document.getElementById("community-network").style.display = "block";
        initCommunityNetwork();
      } else {
        document.getElementById("network").style.display = "block";
        document.getElementById("community-network").style.display = "none";
      }
    });
  });

  // Search — matches against full label (subject → predicate → object)
  const searchBox = document.getElementById("search-box");
  searchBox.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const results = document.getElementById("search-results");
    results.innerHTML = "";
    if (q.length < 2) return;
    const matches = allNodes.filter(n =>
      n.fullLabel.toLowerCase().includes(q) ||
      n.predicate.toLowerCase().includes(q)
    ).slice(0, 20);
    matches.forEach(n => {
      const div = document.createElement("div");
      div.className = "search-result";
      div.innerHTML = `
        <span class="sr-predicate">${escapeHtml(n.predicate)}</span><br>
        <span class="sr-entities">${escapeHtml(n.subjectName)}
          &rarr; ${escapeHtml(n.objectName)}</span>
      `;
      div.addEventListener("click", () => {
        network.selectNodes([n.id]);
        network.focus(n.id, { scale: 2.0, animation: { duration: 500 } });
        showDetail(n.id);
        results.innerHTML = "";
        searchBox.value = n.predicate;
      });
      results.appendChild(div);
    });
  });
}

// ── Assertion type filters ──
function buildTypeFilters() {
  const container = document.getElementById("type-filters");
  const typeCounts = {};
  allNodes.forEach(n => { typeCounts[n.assertionType] = (typeCounts[n.assertionType] || 0) + 1; });

  const sorted = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([atype, count]) => {
    const label = document.createElement("label");
    label.className = "type-filter";
    const isActive = activeAssertionTypes.has(atype);
    label.innerHTML = `
      <input type="checkbox" ${isActive ? "checked" : ""} data-type="${atype}">
      <span class="swatch" style="background:${ASSERTION_COLORS[atype] || '#999'}"></span>
      <span>${atype.replace(/_/g, " ")}</span>
      <span class="type-count">${count}</span>
    `;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) activeAssertionTypes.add(atype);
      else activeAssertionTypes.delete(atype);
      refreshGraph();
    });
    container.appendChild(label);
  });
}

// ── Legend ──
function buildLegend() {
  const legend = document.getElementById("legend");
  if (colorMode === "assertion") {
    const types = Object.entries(ASSERTION_COLORS);
    legend.innerHTML = `<h3>Assertion Types</h3>` + types.map(([t, c]) =>
      `<div class="legend-item">`
      + `<span class="swatch" style="background:${c}"></span>`
      + `${t.replace(/_/g, " ")}</div>`
    ).join("");
  } else {
    legend.innerHTML = `<h3>Communities</h3>`
      + `<div style="color:#6a7080;font-size:11px">`
      + `Colors represent claim<br>community clusters</div>`;
  }
  if (edgeMode === "shared") {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>`
      + `<div style="color:#6a7080;font-size:11px">Width = # shared entities</div>`;
  } else if (edgeMode === "contradiction") {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>
      <div class="legend-item">
        <span style="border-bottom:2px dashed #D55E00;width:20px;display:inline-block">
        </span> Cross-paper contradiction</div>
      <div class="legend-item">
        <span style="border-bottom:2px dashed #CC79A7;width:20px;display:inline-block">
        </span> Intra-paper contradiction</div>
      <div style="color:#6a7080;font-size:11px;margin-top:4px">
        Width = contradiction probability</div>`;
  } else {
    legend.innerHTML += `<h3 style="margin-top:8px">Edges</h3>
      <div style="color:#6a7080;font-size:11px">Gray = shared entities<br>
      <span style="color:#D55E00">Orange</span> = cross-paper contradiction<br>
      <span style="color:#CC79A7">Pink</span> = intra-paper contradiction</div>`;
  }
}

// ── Claim detail panel ──
function showDetail(nodeId) {
  const n = nodeIndex[nodeId];
  if (!n) return;

  const panel = document.getElementById("detail-panel");
  const content = document.getElementById("detail-content");

  let html = `
    <div style="margin-bottom:12px">
      <span style="font-size:15px;font-weight:bold">${escapeHtml(n.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-size:15px;color:#56B4E9;font-weight:bold">
        ${escapeHtml(n.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-size:15px;font-weight:bold">${escapeHtml(n.objectName)}</span>
    </div>
    <table>
      <tr><td>Assertion type</td><td>
        <span style="color:${n.assertionColor}">\u25CF</span>
        ${n.assertionType.replace(/_/g, " ")}</td></tr>
      <tr><td>Confidence</td><td>${n.confidence.toFixed(3)}</td></tr>
      <tr><td>Evidence</td><td>${n.evidenceCount} source(s)</td></tr>
      <tr><td>Controversy</td><td>${n.controversy.toFixed(3)}</td></tr>
      <tr><td>Neighbors</td><td>${n.degree} claims share entities</td></tr>
      <tr><td>Community</td><td>${n.communityId}${COMMUNITY_LABELS_CURRENT[n.communityId]
        ? ' \u2014 ' + escapeHtml(COMMUNITY_LABELS_CURRENT[n.communityId].subfield
          || COMMUNITY_LABELS_CURRENT[n.communityId].auto_label)
        : ''}</td></tr>
    </table>
  `;

  // Find neighbor claims
  const connected = allEdges.filter(e => e.from === nodeId || e.to === nodeId);
  if (connected.length > 0) {
    // Sort by shared count descending
    connected.sort((a, b) => (b.sharedCount || 0) - (a.sharedCount || 0));
    html += `<h3 style="margin-top:12px;font-size:13px">`
      + `Related Claims (${connected.length})</h3><div class="neighbor-list">`;
    connected.slice(0, 30).forEach(e => {
      const otherId = e.from === nodeId ? e.to : e.from;
      const other = nodeIndex[otherId];
      if (!other) return;
      html += `<div class="neighbor-item" onclick="navigateTo('${otherId}')">
        <span style="color:${other.assertionColor}">\u25CF</span>
        <b>${escapeHtml(other.predicate)}</b><br>
        <span class="neighbor-shared">
          ${escapeHtml(other.subjectName)} &rarr; ${escapeHtml(other.objectName)}
          &nbsp;|&nbsp;
          ${e.sharedCount || 1} shared entit${(e.sharedCount || 1) === 1 ? 'y' : 'ies'}
        </span>
      </div>`;
    });
    if (connected.length > 30)
      html += `<div style="color:#6a7080;padding:4px">...and ${connected.length - 30} more</div>`;
    html += `</div>`;
  }

  content.innerHTML = html;
  panel.style.display = "block";
}

function showEdgeDetail(edgeId) {
  if (!edgeId || !edgeId.startsWith("contra_")) return;

  // Get edge data from vis.js dataset
  const edge = edgesDataset.get(edgeId);
  if (!edge) return;

  // Find matching contradiction data
  const contra = CONTRADICTION_EDGES_CURRENT.find(
    c => c.from === edge.from && c.to === edge.to
  );
  if (!contra) return;

  const nodeA = nodeIndex[edge.from];
  const nodeB = nodeIndex[edge.to];
  if (!nodeA || !nodeB) return;

  const panel = document.getElementById("detail-panel");
  const content = document.getElementById("detail-content");

  const pctColor = contra.p_contradiction > 0.8
    ? "#D55E00" : contra.p_contradiction > 0.5 ? "#E69F00" : "#56B4E9";
  const typeLabel = contra.is_cross_paper ? "Cross-paper" : "Intra-paper";
  const typeBadgeClass = contra.is_cross_paper ? "contra-cross" : "contra-intra";

  let html = `
    <div style="margin-bottom:12px">
      <span style="font-size:16px;font-weight:bold;color:${pctColor}">Contradiction</span>
      <span style="font-size:14px;color:${pctColor};margin-left:8px">
        p = ${contra.p_contradiction.toFixed(3)}</span>
      <span class="contra-badge ${typeBadgeClass}" style="margin-left:8px">${typeLabel}</span>
    </div>

    <div style="background:#1e2130;border-radius:6px;padding:10px;margin-bottom:8px;
      border-left:3px solid ${nodeA.assertionColor}">
      <div style="font-size:11px;color:#6a7080;margin-bottom:4px">Claim A</div>
      <span style="font-weight:bold">${escapeHtml(nodeA.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="color:#56B4E9;font-weight:bold">${escapeHtml(nodeA.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-weight:bold">${escapeHtml(nodeA.objectName)}</span>
      <div style="margin-top:4px;font-size:11px;color:#6a7080">
        <span style="color:${nodeA.assertionColor}">\u25CF</span>
        ${nodeA.assertionType.replace(/_/g, " ")}
        &nbsp;|&nbsp; Evidence: ${nodeA.evidenceCount}
        &nbsp;|&nbsp; Community: ${nodeA.communityId}
      </div>
      <div style="margin-top:4px"><a href="#"
        onclick="network.selectNodes(['${edge.from.replace(/'/g, "\\'")}']);
          network.focus('${edge.from.replace(/'/g, "\\'")}',
            {scale:2,animation:{duration:500}});
          showDetail('${edge.from.replace(/'/g, "\\'")}'); return false;"
        style="color:#56B4E9;font-size:11px;text-decoration:none">Focus this claim &rarr;</a>
      </div>
    </div>

    <div style="text-align:center;font-size:18px;color:#D55E00;margin:4px 0">&#x26A1;</div>

    <div style="background:#1e2130;border-radius:6px;padding:10px;margin-bottom:12px;
      border-left:3px solid ${nodeB.assertionColor}">
      <div style="font-size:11px;color:#6a7080;margin-bottom:4px">Claim B</div>
      <span style="font-weight:bold">${escapeHtml(nodeB.subjectName)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="color:#56B4E9;font-weight:bold">${escapeHtml(nodeB.predicate)}</span>
      <span class="claim-arrow">&rarr;</span>
      <span style="font-weight:bold">${escapeHtml(nodeB.objectName)}</span>
      <div style="margin-top:4px;font-size:11px;color:#6a7080">
        <span style="color:${nodeB.assertionColor}">\u25CF</span>
        ${nodeB.assertionType.replace(/_/g, " ")}
        &nbsp;|&nbsp; Evidence: ${nodeB.evidenceCount}
        &nbsp;|&nbsp; Community: ${nodeB.communityId}
      </div>
      <div style="margin-top:4px"><a href="#"
        onclick="network.selectNodes(['${edge.to.replace(/'/g, "\\'")}']);
          network.focus('${edge.to.replace(/'/g, "\\'")}',
            {scale:2,animation:{duration:500}});
          showDetail('${edge.to.replace(/'/g, "\\'")}'); return false;"
        style="color:#56B4E9;font-size:11px;text-decoration:none">Focus this claim &rarr;</a>
      </div>
    </div>

    <table>
      <tr><td>Method</td><td>${contra.method.replace(/_/g, " ")}</td></tr>
      <tr><td>Shared entities</td><td>${
        contra.shared_entities.length > 0
          ? escapeHtml(contra.shared_entities.join(", ")) : "none"
      }</td></tr>
    </table>
  `;

  content.innerHTML = html;
  panel.style.display = "block";
}

function navigateTo(nodeId) {
  network.selectNodes([nodeId]);
  network.focus(nodeId, { scale: 2.0, animation: { duration: 400 } });
  showDetail(nodeId);
}

function closeDetail() {
  document.getElementById("detail-panel").style.display = "none";
}

function focusNeighborhood(nodeId) {
  const connected = allEdges.filter(e => e.from === nodeId || e.to === nodeId);
  const neighborIds = new Set([nodeId]);
  connected.forEach(e => { neighborIds.add(e.from); neighborIds.add(e.to); });
  network.fit({ nodes: Array.from(neighborIds), animation: { duration: 600 } });
}

// ── Stats ──
function updateStats() {
  const visNodes = nodesDataset ? nodesDataset.length : allNodes.length;
  const visEdges = edgesDataset ? edgesDataset.length : allEdges.length;
  let statsHtml = `
    <b>${visNodes}</b> / ${allNodes.length} claims visible<br>
    <b>${visEdges}</b> / ${allEdges.length} shared-entity links<br>
    <b>${GRAPH_STATS_CURRENT.communities}</b> communities detected<br>
    <b>${GRAPH_STATS_CURRENT.maxDegree}</b> max neighbors<br>
    <b>${GRAPH_STATS_CURRENT.avgDegree}</b> avg neighbors
  `;
  if (CONTRADICTION_EDGES_CURRENT.length > 0) {
    const crossCount = CONTRADICTION_EDGES_CURRENT.filter(e => e.is_cross_paper).length;
    statsHtml += `<hr style="border-color:#2a2d3a;margin:6px 0">
      <b>${CONTRADICTION_EDGES_CURRENT.length}</b> contradictions<br>
      <b>${crossCount}</b> cross-paper<br>
      <b>${COMMUNITY_DISAGREEMENTS_CURRENT.length}</b> community conflicts
    `;
  }
  document.getElementById("stats").innerHTML = statsHtml;
}

// ── Community overview network ──
function initCommunityNetwork() {
  if (communityNetwork) return; // already initialized

  const container = document.getElementById("community-network");

  // ── Controversy coloring: blue (calm) → orange → red (controversial) ──
  const DISAGREE_REF = COMMUNITY_DISAGREEMENTS_CURRENT;

  const controversyMap = {};
  DISAGREE_REF.forEach(d => {
    controversyMap[d.comm_a_id] = (controversyMap[d.comm_a_id] || 0) + d.normalized_score;
    controversyMap[d.comm_b_id] = (controversyMap[d.comm_b_id] || 0) + d.normalized_score;
  });
  const maxControversy = Math.max(...Object.values(controversyMap), 0.001);

  function lerpRGB(a, b, t) {
    return a.map((v, i) => Math.round(v + (b[i] - v) * t));
  }
  const COLD = [86, 180, 233];   // #56B4E9 — calm blue
  const WARM = [230, 159, 0];    // #E69F00 — amber
  const HOT  = [213, 94, 0];     // #D55E00 — controversy red-orange
  function controversyColor(cid) {
    const t = Math.min((controversyMap[cid] || 0) / maxControversy, 1);
    const rgb = t < 0.5 ? lerpRGB(COLD, WARM, t * 2) : lerpRGB(WARM, HOT, (t - 0.5) * 2);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }
  function darkenRGB(color) {
    return color.replace(/\d+/g, (m, i) => i < 3 ? Math.round(Number(m) * 0.7) : m);
  }

  // Build super-nodes from community labels
  const commNodes = Object.values(COMMUNITY_LABELS_CURRENT).map(cl => {
    const bg = controversyColor(cl.community_id);
    const cScore = (controversyMap[cl.community_id] || 0);
    return {
      id: cl.community_id,
      value: cl.claim_count,
      label: cl.subfield || cl.auto_label,
      title: `<div style='font-family:Arial;font-size:13px'>`
        + `<b>${escapeHtml(cl.subfield || cl.auto_label)}</b><br>`
        + `${cl.claim_count} claims`
        + `${cScore > 0 ? '<br>Controversy: ' + cScore.toFixed(2) : ''}`
        + `<br>Top entities: ${cl.top_entities.slice(0, 3).join(", ")}</div>`,
      color: {
        background: bg, border: darkenRGB(bg),
        highlight: { background: "#FFD700", border: "#FFA500" },
      },
      font: { color: "#e0e0e0", strokeWidth: 3, strokeColor: "rgba(26,29,39,0.8)" },
    };
  });

  // Build edges from disagreements — only show top disagreements to avoid clutter
  const maxScore = COMMUNITY_DISAGREEMENTS_CURRENT.length > 0
    ? COMMUNITY_DISAGREEMENTS_CURRENT[0].normalized_score : 1;
  const commEdges = COMMUNITY_DISAGREEMENTS_CURRENT.slice(0, 100).map((d, i) => ({
    id: `cd_${i}`,
    from: d.comm_a_id,
    to: d.comm_b_id,
    width: 1 + (d.normalized_score / maxScore) * 8,
    color: {
      color: `rgba(213,94,0,${0.3 + 0.7 * d.normalized_score / maxScore})`,
      highlight: "#FF0000",
    },
    title: `<div style='font-family:Arial;font-size:13px'>`
      + `<b style='color:#D55E00'>Disagreement Score: ${d.normalized_score.toFixed(2)}</b>`
      + `<br>${d.num_contradictions} contradicting claim pairs<br>`
      + `<b>${escapeHtml(COMMUNITY_LABELS_CURRENT[d.comm_a_id]?.subfield || d.comm_a_label)}</b>`
      + ` vs<br>`
      + `<b>${escapeHtml(COMMUNITY_LABELS_CURRENT[d.comm_b_id]?.subfield || d.comm_b_label)}</b>`
      + `</div>`,
    label: d.normalized_score.toFixed(1),
    font: { size: 10, color: "#D55E00", strokeWidth: 3, strokeColor: "#1a1d27" },
  }));

  communityNetwork = new vis.Network(container, {
    nodes: new vis.DataSet(commNodes),
    edges: new vis.DataSet(commEdges),
  }, {
    nodes: {
      shape: "dot",
      borderWidth: 2,
      scaling: {
        min: 5,
        max: 60,
        label: { enabled: true, min: 0, max: 30, drawThreshold: 8, maxVisible: 30 },
      },
    },
    edges: { smooth: { type: "continuous" } },
    physics: {
      solver: "barnesHut",
      barnesHut: {
        gravitationalConstant: -3000,
        centralGravity: 0.3,
        springLength: 150,
        springConstant: 0.04,
        damping: 0.5,
        avoidOverlap: 0.5,
      },
      stabilization: { iterations: 500 },
    },
    interaction: { hover: true, tooltipDelay: 100 },
  });
  // ── Community node click → detail panel ──
  communityNetwork.on("click", (params) => {
    if (params.nodes.length > 0) {
      showCommunityDetail(params.nodes[0]);
    } else {
      document.getElementById("detail-panel").style.display = "none";
    }
  });

  function showCommunityDetail(communityId) {
    const cl = COMMUNITY_LABELS_CURRENT[communityId];
    if (!cl) return;

    const panel = document.getElementById("detail-panel");
    const content = document.getElementById("detail-content");
    const cScore = (controversyMap[communityId] || 0);

    let html = `
      <div style="margin-bottom:12px">
        <span style="font-size:16px;font-weight:bold;color:#fff">
          ${escapeHtml(cl.subfield || cl.auto_label)}</span>
        ${cl.subfield
          ? '<br><span style="color:#6a7080;font-size:12px">'
            + escapeHtml(cl.auto_label) + '</span>'
          : ''}
      </div>
      <table>
        <tr><td>Claims</td><td>${cl.claim_count}</td></tr>
        <tr><td>Controversy</td>
          <td style="color:${cScore > 0 ? '#D55E00' : '#6a7080'}">${cScore.toFixed(2)}</td></tr>
        <tr><td>Top entities</td><td>${cl.top_entities.join(", ")}</td></tr>
        <tr><td>Predicates</td><td>${cl.top_predicates.join(", ")}</td></tr>
        ${cl.top_entity_types && cl.top_entity_types.length
          ? '<tr><td>Entity types</td><td>' + cl.top_entity_types.join(", ") + '</td></tr>'
          : ''}
      </table>
    `;

    // Disagreements involving this community
    const disagreements = COMMUNITY_DISAGREEMENTS_CURRENT.filter(d =>
      d.comm_a_id === communityId || d.comm_b_id === communityId
    ).sort((a, b) => b.normalized_score - a.normalized_score);

    if (disagreements.length > 0) {
      html += '<h3 style="margin-top:14px;font-size:13px;color:#fff">Disagreements ('
        + disagreements.length + ')</h3><div class="neighbor-list">';
      disagreements.slice(0, 15).forEach(d => {
        const otherId = d.comm_a_id === communityId ? d.comm_b_id : d.comm_a_id;
        const other = COMMUNITY_LABELS_CURRENT[otherId];
        const otherName = other ? (other.subfield || other.auto_label) : 'Community ' + otherId;
        html += '<div class="neighbor-item"'
          + ' onclick="event.stopPropagation();'
          + ' window._navComm && window._navComm(' + otherId + ')">'
          + '<span style="color:#D55E00">●</span> '
          + '<b>' + escapeHtml(otherName) + '</b>'
          + '<span style="float:right;color:#D55E00;font-size:11px">'
          + d.normalized_score.toFixed(2) + '</span><br>'
          + '<span class="neighbor-shared">' + d.num_contradictions + ' contradictions</span>'
          + '</div>';

        // Top 2 contradicting claim pairs
        if (d.top_claims && d.top_claims.length > 0) {
          d.top_claims.slice(0, 2).forEach(tc => {
            html += '<div style="margin:2px 0 6px 16px;font-size:11px;'
              + 'color:#6a7080;line-height:1.3">'
              + '“' + escapeHtml(tc.claim_a_text) + '”<br>'
              + '<span style="color:#D55E00;margin-left:8px">vs</span> '
              + '“' + escapeHtml(tc.claim_b_text) + '”'
              + ' <span style="color:#D55E00">(' + tc.p_contradiction.toFixed(2) + ')</span>'
              + '</div>';
          });
        }
      });
      if (disagreements.length > 15) {
        html += '<div style="color:#6a7080;padding:4px">...and '
          + (disagreements.length - 15) + ' more</div>';
      }
      html += '</div>';
    } else {
      html += '<div style="margin-top:12px;color:#6a7080;font-size:12px">'
        + 'No disagreements with other communities</div>';
    }

    content.innerHTML = html;
    panel.style.display = "block";
  }

  // Expose navigation for onclick handlers in innerHTML
  window._navComm = function(communityId) {
    communityNetwork.selectNodes([communityId]);
    communityNetwork.focus(communityId, { scale: 1.5, animation: { duration: 400 } });
    showCommunityDetail(communityId);
  };
}

// ── Disagreement list ──
function buildDisagreementList() {
  const container = document.getElementById("disagreement-list");
  if (!COMMUNITY_DISAGREEMENTS_CURRENT.length) {
    container.innerHTML = '<div style="color:#6a7080">No contradiction data available</div>';
    return;
  }

  container.innerHTML = "";
  COMMUNITY_DISAGREEMENTS_CURRENT.slice(0, 20).forEach(d => {
    const labelA = COMMUNITY_LABELS_CURRENT[d.comm_a_id]?.subfield || d.comm_a_label;
    const labelB = COMMUNITY_LABELS_CURRENT[d.comm_b_id]?.subfield || d.comm_b_label;
    const div = document.createElement("div");
    div.className = "disagreement-item";
    div.innerHTML = `
      <span class="disagreement-score">${d.normalized_score.toFixed(2)}</span>
      <b>${escapeHtml(labelA)}</b><br>
      <span style="color:#D55E00">vs</span> <b>${escapeHtml(labelB)}</b><br>
      <span style="color:#6a7080;font-size:11px">${d.num_contradictions} contradictions</span>
    `;
    div.addEventListener("click", () => {
      // Switch to community view and focus on these two communities
      const vmInput = document.querySelector('input[name="viewMode"][value="communities"]');
      vmInput.checked = true;
      viewMode = "communities";
      document.getElementById("network").style.display = "none";
      document.getElementById("community-network").style.display = "block";
      initCommunityNetwork();
      setTimeout(() => {
        communityNetwork.fit({ nodes: [d.comm_a_id, d.comm_b_id], animation: { duration: 600 } });
      }, 500);
    });
    container.appendChild(div);
  });
}

// ── Utilities ──
function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function darken(hex, pct) {
  const num = parseInt(hex.replace("#",""), 16);
  const r = Math.max(0, (num >> 16) - pct);
  const g = Math.max(0, ((num >> 8) & 0xFF) - pct);
  const b = Math.max(0, (num & 0xFF) - pct);
  return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
}

function lighten(hex, pct) {
  const num = parseInt(hex.replace("#",""), 16);
  const r = Math.min(255, (num >> 16) + pct);
  const g = Math.min(255, ((num >> 8) & 0xFF) + pct);
  const b = Math.min(255, (num & 0xFF) + pct);
  return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
}
</script>
</body>
</html>"""


def generate_multi_graph_html(
    graphs: list[
        tuple[str, nx.MultiDiGraph, ContradictionVizData | None]
        | tuple[str, nx.MultiDiGraph, ContradictionVizData | None, dict]
    ],
    output_path: Path | str,
) -> Path:
    """Generate a multi-graph interactive HTML with a bottom tab bar.

    Each entry in graphs is (tab_label, graph, contradiction_data[, options]).
    The first entry is shown by default. Clicking a tab swaps the vis.js data.

    Options dict supports:
        defaultColorMode: "assertion" | "community" — coloring on tab switch
        defaultEdgeMode: "shared" | "contradiction" | "both" — edge display on tab switch

    Args:
        graphs: List of (tab_label, graph, contradiction_data[, options]) tuples.
        output_path: Destination path for the HTML file.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tabs_data = []
    for entry in graphs:
        label, graph, contradiction_data = entry[0], entry[1], entry[2]
        tab_opts = entry[3] if len(entry) > 3 else {}
        nodes_data, edges_data, communities, stats = _build_claim_graph(graph)

        # Serialize contradiction data (same logic as generate_interactive_html)
        if contradiction_data is not None:
            contra_edges = [
                {
                    "from": e.claim_a_id,
                    "to": e.claim_b_id,
                    "p_contradiction": float(e.p_contradiction),
                    "method": e.method,
                    "is_cross_paper": e.is_cross_paper,
                    "shared_entities": e.shared_entities,
                }
                for e in contradiction_data.contradiction_edges
            ]
            comm_labels = {
                str(cid): {
                    "community_id": cl.community_id,
                    "auto_label": cl.auto_label,
                    "top_predicates": cl.top_predicates,
                    "top_entities": cl.top_entities,
                    "claim_count": cl.claim_count,
                }
                for cid, cl in contradiction_data.community_labels.items()
            }
            comm_disagree = [
                {
                    "comm_a_id": d.comm_a_id,
                    "comm_b_id": d.comm_b_id,
                    "comm_a_label": d.comm_a_label,
                    "comm_b_label": d.comm_b_label,
                    "normalized_score": float(d.normalized_score),
                    "raw_score": float(d.raw_score),
                    "num_contradictions": d.num_contradictions,
                    "top_claims": d.top_claims,
                }
                for d in contradiction_data.community_disagreements
            ]
        else:
            contra_edges = []
            comm_labels = {}
            comm_disagree = []

        tab_entry = {
            "label": label,
            "nodes": nodes_data,
            "edges": edges_data,
            "stats": stats,
            "contradictionEdges": contra_edges,
            "communityLabels": comm_labels,
            "communityDisagreements": comm_disagree,
        }
        if "defaultColorMode" in tab_opts:
            tab_entry["defaultColorMode"] = tab_opts["defaultColorMode"]
        if "defaultEdgeMode" in tab_opts:
            tab_entry["defaultEdgeMode"] = tab_opts["defaultEdgeMode"]
        # Topology data from tab options
        topology_path = tab_opts.get("topology_path")
        if topology_path:
            topo_file = Path(topology_path)
            if topo_file.exists():
                with open(topo_file) as f:
                    topo_raw = json.load(f)
                hi_nodes: dict[str, dict] = {}
                for nid in topo_raw.get("all_articulation_points", []):
                    hi_nodes.setdefault(nid, {"types": [], "voi": 0})
                    hi_nodes[nid]["types"].append("articulation_point")
                for bridge in topo_raw.get("all_bridges", []):
                    for nid in bridge:
                        hi_nodes.setdefault(nid, {"types": [], "voi": 0})
                        if "bridge" not in hi_nodes[nid]["types"]:
                            hi_nodes[nid]["types"].append("bridge")
                for c in topo_raw.get("top_voi_contradictions", []):
                    for key in ("claim_a_id", "claim_b_id"):
                        nid = c[key]
                        hi_nodes.setdefault(nid, {"types": [], "voi": 0})
                        hi_nodes[nid]["voi"] = max(hi_nodes[nid]["voi"], c["voi"])
                        if "high_voi" not in hi_nodes[nid]["types"]:
                            hi_nodes[nid]["types"].append("high_voi")
                tab_entry["topologyData"] = {
                    "highImpactNodes": hi_nodes,
                    "summary": topo_raw.get("summary", {}),
                }
        tabs_data.append(tab_entry)

    html = _MULTI_HTML_TEMPLATE
    html = html.replace("__GRAPH_TABS_JSON__", json.dumps(tabs_data, ensure_ascii=False))
    html = html.replace("__ASSERTION_COLORS_JSON__", json.dumps(_ASSERTION_COLORS))

    output_path.write_text(html, encoding="utf-8")
    log.info(
        "interactive.multi_saved",
        path=str(output_path),
        num_tabs=len(tabs_data),
        size_mb=round(output_path.stat().st_size / 1024 / 1024, 2),
    )
    return output_path


def generate_interactive_html(
    graph: nx.MultiDiGraph,
    output_path: Path | str,
    contradiction_data: ContradictionVizData | None = None,
    topology_path: Path | str | None = None,
) -> Path:
    """Generate a self-contained interactive HTML visualization of the claim graph.

    Transforms the entity-centric knowledge graph into a claim-centric view where:
    - Nodes are assertions (subject → predicate → object)
    - Edges connect claims that reference the same entities
    - Optional contradiction overlay from NLI scoring
    - Optional topology analysis overlay for high-impact node highlighting

    Args:
        graph: The knowledge graph as a NetworkX MultiDiGraph.
        output_path: Destination path for the HTML file.
        contradiction_data: Optional pre-computed contradiction visualization data.
            When provided, adds contradiction edge overlay, community overview, and
            disagreement list to the viewer.
        topology_path: Optional path to a topology analysis JSON file produced by
            the topology analysis stage. When provided, adds a sidebar toggle to
            highlight high-impact nodes (articulation points, bridge endpoints,
            high-VOI contradiction claims) with red borders.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "interactive.generating",
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        has_contradiction_data=contradiction_data is not None,
    )

    nodes_data, edges_data, communities, stats = _build_claim_graph(graph)

    log.info(
        "interactive.claim_graph",
        claims=len(nodes_data),
        shared_entity_edges=len(edges_data),
        communities=len(communities),
    )

    # ── Serialize contradiction data ──
    if contradiction_data is not None:
        contradiction_edges_data = [
            {
                "from": e.claim_a_id,
                "to": e.claim_b_id,
                "p_contradiction": float(e.p_contradiction),
                "method": e.method,
                "is_cross_paper": e.is_cross_paper,
                "shared_entities": e.shared_entities,
            }
            for e in contradiction_data.contradiction_edges
        ]
        community_labels_data = {
            str(cid): {
                "community_id": cl.community_id,
                "auto_label": cl.auto_label,
                "top_predicates": cl.top_predicates,
                "top_entities": cl.top_entities,
                "claim_count": cl.claim_count,
            }
            for cid, cl in contradiction_data.community_labels.items()
        }
        community_disagreements_data = [
            {
                "comm_a_id": d.comm_a_id,
                "comm_b_id": d.comm_b_id,
                "comm_a_label": d.comm_a_label,
                "comm_b_label": d.comm_b_label,
                "normalized_score": float(d.normalized_score),
                "raw_score": float(d.raw_score),
                "num_contradictions": d.num_contradictions,
                "top_claims": d.top_claims,
            }
            for d in contradiction_data.community_disagreements
        ]
    else:
        contradiction_edges_data = []
        community_labels_data = {}
        community_disagreements_data = []

    # ── Serialize topology data ──
    topology_data: dict = {}
    if topology_path is not None:
        topology_file = Path(topology_path)
        if topology_file.exists():
            import json as _json

            with open(topology_file) as f:
                topo_raw = _json.load(f)
            # Build set of high-impact node IDs with metadata
            high_impact_nodes: dict[str, dict] = {}
            # Articulation points
            for nid in topo_raw.get("all_articulation_points", []):
                high_impact_nodes.setdefault(nid, {"types": [], "voi": 0})
                high_impact_nodes[nid]["types"].append("articulation_point")
            # Bridge endpoints
            for bridge in topo_raw.get("all_bridges", []):
                for nid in bridge:
                    high_impact_nodes.setdefault(nid, {"types": [], "voi": 0})
                    if "bridge" not in high_impact_nodes[nid]["types"]:
                        high_impact_nodes[nid]["types"].append("bridge")
            # Top VOI claims
            for c in topo_raw.get("top_voi_contradictions", []):
                for key in ("claim_a_id", "claim_b_id"):
                    nid = c[key]
                    high_impact_nodes.setdefault(nid, {"types": [], "voi": 0})
                    high_impact_nodes[nid]["voi"] = max(high_impact_nodes[nid]["voi"], c["voi"])
                    if "high_voi" not in high_impact_nodes[nid]["types"]:
                        high_impact_nodes[nid]["types"].append("high_voi")
            topology_data = {
                "highImpactNodes": high_impact_nodes,
                "summary": topo_raw.get("summary", {}),
            }
            log.info(
                "topology_data_loaded",
                high_impact_nodes=len(high_impact_nodes),
            )

    # Inject data into template
    html = _HTML_TEMPLATE
    html = html.replace("__NODES_JSON__", json.dumps(nodes_data, ensure_ascii=False))
    html = html.replace("__EDGES_JSON__", json.dumps(edges_data, ensure_ascii=False))
    html = html.replace("__ASSERTION_COLORS_JSON__", json.dumps(_ASSERTION_COLORS))
    html = html.replace("__STATS_JSON__", json.dumps(stats))
    html = html.replace(
        "__CONTRADICTION_EDGES_JSON__",
        json.dumps(contradiction_edges_data, ensure_ascii=False),
    )
    html = html.replace(
        "__COMMUNITY_LABELS_JSON__",
        json.dumps(community_labels_data, ensure_ascii=False),
    )
    html = html.replace(
        "__COMMUNITY_DISAGREEMENTS_JSON__",
        json.dumps(community_disagreements_data, ensure_ascii=False),
    )
    html = html.replace(
        "__TOPOLOGY_DATA_JSON__",
        json.dumps(topology_data, ensure_ascii=False),
    )

    output_path.write_text(html, encoding="utf-8")

    log.info(
        "interactive.saved",
        path=str(output_path),
        size_mb=round(output_path.stat().st_size / 1024 / 1024, 2),
    )
    return output_path


def main() -> None:
    """CLI entry point: load the gastruloid KG and generate interactive claim HTML."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive claim-centric KG visualization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/knowledge_graph/gastruloid_kg.pkl"),
        help="Path to the pickle file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/knowledge_graph/interactive_kg.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    from autoreview.knowledge_graph.graph import load_graph

    graph = load_graph(args.input)
    result = generate_interactive_html(graph, args.output)
    print(f"Interactive claim visualization saved to: {result}")


if __name__ == "__main__":
    main()
