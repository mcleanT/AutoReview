"""Topic cluster, finding, and contradiction detection for the knowledge graph.

Hierarchical view over the flat edge graph:
    Community (Louvain) → TopicCluster → Finding → KGEdge

Cluster = group of edges with same (subject, predicate_class, object) triple.
Finding = cluster partition by (direction, condition_group).
Contradiction = pair of findings within same cluster with opposing evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.models import BetaPosterior

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Predicate class table
# ---------------------------------------------------------------------------

PREDICATE_CLASS_TABLE: dict[str, list[str]] = {
    "activating": ["induces", "is_sufficient_for", "phosphorylates", "stabilizes"],
    "inhibiting": ["inhibits", "degrades"],
    "regulatory": ["regulates", "is_required_for", "modifies", "maintains"],
    "associative": ["correlates_with", "interacts_with", "colocalizes_with"],
    "structural": ["is_component_of", "is_located_in", "is_marker_of"],
    "transformative": ["differentiates_into", "converts", "transports"],
}

# Inverted index: predicate → class name (built once at module load)
_PREDICATE_TO_CLASS: dict[str, str] = {
    pred: cls for cls, preds in PREDICATE_CLASS_TABLE.items() for pred in preds
}


def get_predicate_class(predicate: str) -> str:
    """Return the predicate class for a given predicate string.

    Args:
        predicate: A predicate string (e.g. "induces", "inhibits").

    Returns:
        The class name from PREDICATE_CLASS_TABLE, or "other" if unknown.
    """
    return _PREDICATE_TO_CLASS.get(predicate, "other")


# ---------------------------------------------------------------------------
# Organism class mapping
# ---------------------------------------------------------------------------

_ORGANISM_CLASS_MAP: dict[str, str] = {
    "Mus musculus": "mouse",
    "Rattus norvegicus": "rat",
    "Homo sapiens": "human",
    "Macaca fascicularis": "macaque",
    "Danio rerio": "zebrafish",
    "Xenopus laevis": "xenopus",
    "Xenopus tropicalis": "xenopus",
    "Drosophila melanogaster": "drosophila",
    "Caenorhabditis elegans": "celegans",
}


def get_organism_class(organism: str | None) -> str:
    """Return a normalised organism class label.

    Args:
        organism: Full scientific species name, or None.

    Returns:
        Mapped short label (e.g. "mouse", "human"), lowercase genus as fallback,
        or "unknown" for None.
    """
    if organism is None:
        return "unknown"
    if organism in _ORGANISM_CLASS_MAP:
        return _ORGANISM_CLASS_MAP[organism]
    # Fallback: lowercase genus (first word)
    return organism.split()[0].lower()


# ---------------------------------------------------------------------------
# TopicCluster dataclass
# ---------------------------------------------------------------------------


@dataclass
class TopicCluster:
    """A group of edges sharing (subject, predicate_class, object).

    Attributes:
        cluster_id: SHA1(subject_id|predicate_class|object_id)[:16]
        subject_id: Source entity ID.
        object_id: Target entity ID.
        predicate_class: Normalised predicate class (from PREDICATE_CLASS_TABLE).
        member_predicates: Set of raw predicate strings in this cluster.
        edge_ids: List of edge IDs in this cluster.
        finding_ids: List of finding IDs derived from this cluster (populated by form_findings).
    """

    cluster_id: str
    subject_id: str
    object_id: str
    predicate_class: str
    member_predicates: set[str]
    edge_ids: list[str]
    finding_ids: list[str] = field(default_factory=list)


def _sha1_id(text: str, length: int = 16) -> str:
    """Return first `length` hex characters of the SHA1 of `text`."""
    return hashlib.sha1(text.encode()).hexdigest()[:length]


def build_topic_clusters(graph: nx.MultiDiGraph) -> list[TopicCluster]:
    """Group graph edges into TopicClusters by (subject, predicate_class, object).

    Singleton clusters (exactly 1 edge) are discarded because no within-cluster
    contradiction is possible.

    Args:
        graph: A MultiDiGraph where each edge has at least ``edge_id`` and
               ``predicate`` attributes.

    Returns:
        Sorted list of TopicCluster objects (sorted by cluster_id).
    """
    # Group edges by (subject_id, predicate_class, object_id)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for u, v, data in graph.edges(data=True):
        predicate = data.get("predicate", "")
        pred_class = get_predicate_class(predicate)
        key = (str(u), pred_class, str(v))
        groups.setdefault(key, []).append(data)

    clusters: list[TopicCluster] = []
    for (subject_id, pred_class, object_id), edges in groups.items():
        if len(edges) < 2:
            # Discard singletons
            continue

        cluster_id = _sha1_id(f"{subject_id}|{pred_class}|{object_id}")
        member_predicates = {e.get("predicate", "") for e in edges}
        edge_ids = [e.get("edge_id", "") for e in edges]

        clusters.append(
            TopicCluster(
                cluster_id=cluster_id,
                subject_id=subject_id,
                object_id=object_id,
                predicate_class=pred_class,
                member_predicates=member_predicates,
                edge_ids=edge_ids,
            )
        )

    clusters.sort(key=lambda c: c.cluster_id)
    log.debug(
        "build_topic_clusters",
        n_clusters=len(clusters),
        n_edges=graph.number_of_edges(),
    )
    return clusters


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A coherent directional sub-claim within a TopicCluster.

    Attributes:
        finding_id: SHA1(cluster_id|direction|condition_group_key)[:16]
        topic_cluster_id: ID of the parent TopicCluster.
        direction: "positive" or "negative".
        condition_group: "{organism_class}|{vitro_str}" grouping key.
        member_edge_ids: Edge IDs in this finding.
        anchor_edge_id: The single most representative edge.
        anchor_text: Natural-language text of the anchor edge.
        confidence: Aggregated BetaPosterior over member edges.
        paper_ids: Set of paper IDs contributing to this finding.
        evidence_count: Number of member edges.
        organism_class: Normalised organism label.
        in_vitro: True if in-vitro context.
    """

    finding_id: str
    topic_cluster_id: str
    direction: str
    condition_group: str
    member_edge_ids: list[str]
    anchor_edge_id: str
    anchor_text: str
    confidence: BetaPosterior
    paper_ids: set[str]
    evidence_count: int
    organism_class: str
    in_vitro: bool


def form_findings(
    clusters: list[TopicCluster],
    graph: nx.MultiDiGraph,
) -> list[Finding]:
    """Partition each TopicCluster into Findings by (direction, condition_group).

    Within each cluster, edges are grouped by their direction and experimental
    context (organism class + in_vitro flag). Each group becomes one Finding.
    Aggregated Beta posteriors are computed by accumulating alpha and beta
    contributions from each edge's confidence_mean.

    Side effect: mutates each cluster's ``finding_ids`` list in-place.

    Args:
        clusters: TopicCluster objects (from build_topic_clusters).
        graph: The same MultiDiGraph used to build the clusters.

    Returns:
        Sorted list of Finding objects (sorted by finding_id).
    """
    # Build edge_id → edge data index once
    edge_index = _get_edge_data_by_id(graph)

    findings: list[Finding] = []

    for cluster in clusters:
        # Group edges by (direction, condition_group)
        subgroups: dict[tuple[str, str], list[dict[str, Any]]] = {}

        for eid in cluster.edge_ids:
            edata = edge_index.get(eid)
            if edata is None:
                continue

            direction = edata.get("direction") or "positive"
            organism = edata.get("organism")
            in_vitro_flag: bool = bool(edata.get("in_vitro", False))
            org_class = get_organism_class(organism)
            vitro_str = "invitro" if in_vitro_flag else "invivo"
            condition_group = f"{org_class}|{vitro_str}"

            key = (direction, condition_group)
            subgroups.setdefault(key, []).append(edata)

        for (direction, condition_group), edges in subgroups.items():
            # Select anchor edge: prefer interpretive section_source, then highest confidence_mean
            def _anchor_sort_key(e: dict[str, Any]) -> tuple[int, float]:
                is_interpretive = 1 if e.get("section_source") == "interpretive" else 0
                conf = float(e.get("confidence_mean", 0.0))
                return (is_interpretive, conf)

            sorted_edges = sorted(edges, key=_anchor_sort_key, reverse=True)
            anchor = sorted_edges[0]
            anchor_eid = anchor.get("edge_id", "")
            anchor_text = anchor.get("natural_language", "")

            # Aggregate Beta posterior: start (1.0, 1.0), accumulate per-edge
            alpha = 1.0
            beta_acc = 1.0
            for e in edges:
                conf_mean = float(e.get("confidence_mean", 0.5))
                alpha += conf_mean
                beta_acc += 1.0 - conf_mean

            confidence = BetaPosterior(alpha=alpha, beta_param=beta_acc)

            # Paper IDs from _kg_edge evidence_links
            paper_ids: set[str] = set()
            for e in edges:
                kg_edge = e.get("_kg_edge")
                if kg_edge is not None and hasattr(kg_edge, "evidence_links"):
                    for link in kg_edge.evidence_links:
                        if hasattr(link, "paper_id"):
                            paper_ids.add(link.paper_id)

            # Derive organism_class and in_vitro from condition_group key
            parts = condition_group.split("|")
            org_class = parts[0]
            in_vitro_bool = parts[1] == "invitro" if len(parts) > 1 else False

            member_edge_ids = [e.get("edge_id", "") for e in edges]

            finding_id = _sha1_id(f"{cluster.cluster_id}|{direction}|{condition_group}")

            f = Finding(
                finding_id=finding_id,
                topic_cluster_id=cluster.cluster_id,
                direction=direction,
                condition_group=condition_group,
                member_edge_ids=member_edge_ids,
                anchor_edge_id=anchor_eid,
                anchor_text=anchor_text,
                confidence=confidence,
                paper_ids=paper_ids,
                evidence_count=len(edges),
                organism_class=org_class,
                in_vitro=in_vitro_bool,
            )
            findings.append(f)
            cluster.finding_ids.append(finding_id)

    findings.sort(key=lambda f: f.finding_id)
    log.debug("form_findings", n_findings=len(findings))
    return findings


# ---------------------------------------------------------------------------
# FindingContradiction dataclass
# ---------------------------------------------------------------------------


@dataclass
class FindingContradiction:
    """A detected contradiction between two Findings in the same TopicCluster.

    Attributes:
        finding_a_id: ID of the first finding (lexicographically smaller).
        finding_b_id: ID of the second finding.
        topic_cluster_id: Shared parent cluster ID.
        contradiction_type: One of "directional", "boundary", "interpretive".
        severity: Scalar severity in [0, 1].
        condition_coupling: How similar the experimental conditions are [0, 1].
        anchor_text_a: Anchor text of finding A.
        anchor_text_b: Anchor text of finding B.
        paper_ids_a: Paper IDs from finding A.
        paper_ids_b: Paper IDs from finding B.
    """

    finding_a_id: str
    finding_b_id: str
    topic_cluster_id: str
    contradiction_type: str  # "directional" | "boundary" | "interpretive"
    severity: float
    condition_coupling: float
    anchor_text_a: str
    anchor_text_b: str
    paper_ids_a: set[str]
    paper_ids_b: set[str]


def detect_finding_contradictions(
    findings: list[Finding],
    clusters: list[TopicCluster],
    boundary_threshold: float = 0.6,
    graph: nx.MultiDiGraph | None = None,
) -> list[FindingContradiction]:
    """Detect contradictions between pairs of Findings within the same TopicCluster.

    Three contradiction types:
    - directional: opposite direction + condition_coupling >= boundary_threshold
    - boundary: opposite direction + condition_coupling < boundary_threshold
    - interpretive: same direction, both anchors in interpretive sections,
                    from different papers (requires graph)

    Condition coupling formula:
        0.7 * org_score + 0.3 * vitro_score
        where org_score: 1.0 same organism class, 0.3 different
              vitro_score: 1.0 same in_vitro flag, 0.6 different

    Args:
        findings: List of Finding objects.
        clusters: List of TopicCluster objects (used for cluster ID lookup).
        boundary_threshold: Coupling score below which directional becomes boundary.
        graph: Optional MultiDiGraph; required for interpretive contradiction detection.

    Returns:
        Sorted list of FindingContradiction objects (sorted by finding_a_id, finding_b_id).
    """
    # Build cluster_id → list[Finding] index
    cluster_findings: dict[str, list[Finding]] = {}
    for f in findings:
        cluster_findings.setdefault(f.topic_cluster_id, []).append(f)

    # Pre-compute edge_index once (not inside the inner loop)
    edge_index: dict[str, dict[str, Any]] = {}
    if graph is not None:
        edge_index = _get_edge_data_by_id(graph)

    contradictions: list[FindingContradiction] = []

    for cluster_id, cluster_finding_list in cluster_findings.items():
        n = len(cluster_finding_list)
        for i in range(n):
            for j in range(i + 1, n):
                fa = cluster_finding_list[i]
                fb = cluster_finding_list[j]

                # Condition coupling
                coupling = _finding_coupling(fa, fb)

                # Determine if directions are opposite
                directions_opposite = fa.direction != fb.direction

                if directions_opposite:
                    if coupling >= boundary_threshold:
                        ctype = "directional"
                        severity = coupling
                    else:
                        ctype = "boundary"
                        severity = coupling * 0.5

                    # Normalise ordering
                    aid, bid = (fa.finding_id, fb.finding_id)
                    if aid > bid:
                        aid, bid = bid, aid
                        ta, tb = fb.anchor_text, fa.anchor_text
                        pa, pb = fb.paper_ids, fa.paper_ids
                    else:
                        ta, tb = fa.anchor_text, fb.anchor_text
                        pa, pb = fa.paper_ids, fb.paper_ids

                    contradictions.append(
                        FindingContradiction(
                            finding_a_id=aid,
                            finding_b_id=bid,
                            topic_cluster_id=cluster_id,
                            contradiction_type=ctype,
                            severity=round(severity, 6),
                            condition_coupling=round(coupling, 6),
                            anchor_text_a=ta,
                            anchor_text_b=tb,
                            paper_ids_a=pa,
                            paper_ids_b=pb,
                        )
                    )

                else:
                    # Same direction — check for interpretive conflict
                    if graph is None:
                        continue

                    a_interpretive = _is_anchor_interpretive(fa, edge_index)
                    b_interpretive = _is_anchor_interpretive(fb, edge_index)

                    if not (a_interpretive and b_interpretive):
                        continue

                    # Different paper sources required
                    if fa.paper_ids & fb.paper_ids:
                        # Shared papers — not a pure interpretive conflict
                        continue

                    # Ensure at least some paper provenance for both
                    if not fa.paper_ids or not fb.paper_ids:
                        continue

                    aid, bid = (fa.finding_id, fb.finding_id)
                    if aid > bid:
                        aid, bid = bid, aid
                        ta, tb = fb.anchor_text, fa.anchor_text
                        pa, pb = fb.paper_ids, fa.paper_ids
                    else:
                        ta, tb = fa.anchor_text, fb.anchor_text
                        pa, pb = fa.paper_ids, fb.paper_ids

                    contradictions.append(
                        FindingContradiction(
                            finding_a_id=aid,
                            finding_b_id=bid,
                            topic_cluster_id=cluster_id,
                            contradiction_type="interpretive",
                            severity=0.5,
                            condition_coupling=round(coupling, 6),
                            anchor_text_a=ta,
                            anchor_text_b=tb,
                            paper_ids_a=pa,
                            paper_ids_b=pb,
                        )
                    )

    contradictions.sort(key=lambda c: (c.finding_a_id, c.finding_b_id))
    log.debug("detect_finding_contradictions", n_contradictions=len(contradictions))
    return contradictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_edge_data_by_id(graph: nx.MultiDiGraph) -> dict[str, dict]:
    """Build an edge_id → edge attribute dict index.

    Args:
        graph: The MultiDiGraph to index.

    Returns:
        Mapping from edge_id string to the edge's attribute dict.
    """
    index: dict[str, dict] = {}
    for _u, _v, data in graph.edges(data=True):
        eid = data.get("edge_id")
        if eid is not None:
            index[str(eid)] = data
    return index


def _finding_coupling(fa: Finding, fb: Finding) -> float:
    """Compute condition coupling between two Findings.

    Formula: 0.7 * org_score + 0.3 * vitro_score

    Args:
        fa: First finding.
        fb: Second finding.

    Returns:
        Coupling score in [0, 1].
    """
    org_score = 1.0 if fa.organism_class == fb.organism_class else 0.3
    vitro_score = 1.0 if fa.in_vitro == fb.in_vitro else 0.6
    return 0.7 * org_score + 0.3 * vitro_score


def _is_anchor_interpretive(finding: Finding, edge_index: dict[str, dict[str, Any]]) -> bool:
    """Return True if the anchor edge has section_source == "interpretive".

    Args:
        finding: The Finding whose anchor to check.
        edge_index: Pre-built edge_id → edge data mapping.

    Returns:
        True if anchor section_source is "interpretive", False otherwise.
    """
    edata = edge_index.get(finding.anchor_edge_id)
    if edata is None:
        return False
    return edata.get("section_source") == "interpretive"
