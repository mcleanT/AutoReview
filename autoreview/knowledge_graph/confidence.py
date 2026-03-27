"""Beta-Binomial confidence scoring with evidence strength weights and independence discounting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.models import BetaPosterior

log = structlog.get_logger(__name__)

# Maps EvidenceStrength string values to numeric weights.
EVIDENCE_WEIGHTS: dict[str, float] = {
    "direct_experimental": 1.0,
    "observational_controlled": 0.7,
    "observational_uncontrolled": 0.4,
    "computational_prediction": 0.3,
    "expert_opinion": 0.2,
}

# Independence discount factor: each additional paper from the same (first, last) author
# group is down-weighted by 0.5 per additional paper.
_INDEPENDENCE_DISCOUNT = 0.5

# Maps certainty level to (alpha, beta) prior parameters.
# "high" = narrow prior (author is confident) → Beta(1.0, 1.0) — standard uniform
# "medium" = wider prior (hedged language) → narrower effective range
# "low" = very wide prior (speculative) → even wider
CERTAINTY_PRIORS: dict[str | None, tuple[float, float]] = {
    "high": (1.0, 1.0),
    "medium": (0.8, 0.8),
    "low": (0.6, 0.6),
    None: (1.0, 1.0),  # default for v4 data without certainty
}

# Epistemic weight modifier by section source.
# primary_empirical claims carry full weight — author's own data.
# interpretive = Discussion synthesis — less grounded.
# attributed_prior = cited prior work — secondhand.
# methodological = method claim — low weight for truth assertions.
SECTION_SOURCE_WEIGHTS: dict[str | None, float] = {
    "primary_empirical": 1.0,
    "interpretive": 0.7,
    "attributed_prior": 0.5,
    "methodological": 0.3,
    None: 1.0,  # default for v4 data
}


def score_edge(
    evidence: list[dict[str, Any]],
    certainty: str | None = None,
    section_source: str | None = None,
) -> BetaPosterior:
    """Compute a Beta-Binomial posterior for a single KG edge.

    Starts from a certainty-dependent Beta prior (defaults to Beta(1, 1) for v4
    data without certainty).  For each evidence unit:
    - The independence discount is 0.5^n where n is the number of prior evidence
      units already seen from the same (first_author, last_author) author group.
    - effective_weight = EVIDENCE_WEIGHTS[strength] * independence_discount * source_weight
    - "supports"    → alpha     += effective_weight
    - "contradicts" → beta_param += effective_weight

    Args:
        evidence: List of evidence dicts, each with keys:
            evidence_direction (str): "supports" or "contradicts"
            evidence_strength  (str): one of the EVIDENCE_WEIGHTS keys
            paper_id           (str): unique paper identifier
            first_author       (str): first author surname
            last_author        (str): last / senior author surname
        certainty: v5 certainty level ("high", "medium", "low", or None for v4 data).
            Controls the Beta prior parameters via CERTAINTY_PRIORS.
        section_source: v5 section source ("primary_empirical", "interpretive",
            "attributed_prior", "methodological", or None for v4 data).
            Controls the epistemic weight modifier via SECTION_SOURCE_WEIGHTS.

    Returns:
        BetaPosterior with updated alpha and beta_param.
    """
    prior_alpha, prior_beta = CERTAINTY_PRIORS.get(certainty, (1.0, 1.0))
    alpha: float = prior_alpha
    beta_param: float = prior_beta

    source_weight = SECTION_SOURCE_WEIGHTS.get(section_source, 1.0)

    # Track how many evidence units have been seen per author group (0-indexed count
    # before current unit → discount = 0.5^count_so_far).
    group_counts: dict[tuple[str, str], int] = defaultdict(int)

    for unit in evidence:
        direction: str = unit["evidence_direction"]
        strength: str = unit["evidence_strength"]
        first: str = unit.get("first_author", "")
        last: str = unit.get("last_author", "")

        author_group = (first, last)
        n_seen = group_counts[author_group]
        independence_discount = _INDEPENDENCE_DISCOUNT**n_seen
        group_counts[author_group] += 1

        base_weight = EVIDENCE_WEIGHTS.get(strength, 0.0)
        effective_weight = base_weight * independence_discount * source_weight

        if direction == "supports":
            alpha += effective_weight
        elif direction == "contradicts":
            beta_param += effective_weight

    return BetaPosterior(alpha=alpha, beta_param=beta_param)


def compute_derived_metrics(
    posterior: BetaPosterior,
    evidence_count: int,
    paper_ids: list[str],
    author_groups: int,
) -> dict[str, Any]:
    """Compute derived interpretability metrics from a Beta posterior.

    controversy_score = min(alpha, beta_param) / max(alpha, beta_param)
    A score of 0 means unanimous evidence; 1 means perfectly balanced.

    Args:
        posterior: BetaPosterior from score_edge.
        evidence_count: Total number of raw evidence units.
        paper_ids: List of paper IDs contributing evidence (may contain duplicates).
        author_groups: Number of independent (first, last) author groups observed.

    Returns:
        Dict with keys: controversy_score, evidence_diversity,
        independent_source_count, evidence_count.
    """
    alpha = posterior.alpha
    beta_p = posterior.beta_param
    controversy_score = min(alpha, beta_p) / max(alpha, beta_p)

    return {
        "controversy_score": controversy_score,
        "evidence_diversity": len(set(paper_ids)),
        "independent_source_count": author_groups,
        "evidence_count": evidence_count,
    }


def score_all_edges(
    graph: nx.MultiDiGraph,
    provenance_by_paper: dict[str, dict[str, Any]],
) -> nx.MultiDiGraph:
    """Score every edge in the graph and update its confidence attributes.

    For each edge, reads the ``_kg_edge`` attribute (a KGEdge), builds the
    evidence list by joining with provenance_by_paper, calls score_edge and
    compute_derived_metrics, then writes flat attrs back onto the edge:
    confidence_mean, controversy_score, evidence_diversity,
    independent_source_count, evidence_count.

    Args:
        graph: NetworkX MultiDiGraph produced by build_nx_graph.
        provenance_by_paper: Mapping of paper_hash → paper provenance dict.
            Each dict should have an "authors" list where each author entry has
            a "role" field ("first" / "last" / etc.) and a "name" field.

    Returns:
        The same graph with updated edge attributes (mutated in-place and
        returned for convenience).
    """

    def _get_author(provenance: dict[str, Any], role: str) -> str:
        for author in provenance.get("authors", []):
            if author.get("role") == role:
                return str(author.get("name", ""))
        return ""

    updated = 0
    for u, v, key, attrs in graph.edges(data=True, keys=True):
        kg_edge = attrs.get("_kg_edge")
        if kg_edge is None:
            continue

        evidence_list: list[dict[str, Any]] = []
        paper_ids: list[str] = []
        group_counts: dict[tuple[str, str], int] = defaultdict(int)

        for ev_link in kg_edge.evidence_links:
            prov = provenance_by_paper.get(ev_link.paper_id, {})
            first = _get_author(prov, "first")
            last = _get_author(prov, "last")
            evidence_list.append(
                {
                    "evidence_direction": ev_link.evidence_direction,
                    "evidence_strength": str(ev_link.evidence_strength),
                    "paper_id": ev_link.paper_id,
                    "first_author": first,
                    "last_author": last,
                }
            )
            paper_ids.append(ev_link.paper_id)
            group_counts[(first, last)] += 1  # just to count unique groups

        # v5 fields (None for v4 data)
        certainty = getattr(kg_edge, "certainty", None)
        section_source = getattr(kg_edge, "section_source", None)
        # Convert StrEnum to str for dict lookup
        certainty_str = str(certainty) if certainty is not None else None
        section_source_str = str(section_source) if section_source is not None else None

        posterior = score_edge(
            evidence_list, certainty=certainty_str, section_source=section_source_str
        )
        author_groups = len(group_counts)
        metrics = compute_derived_metrics(
            posterior,
            evidence_count=len(evidence_list),
            paper_ids=paper_ids,
            author_groups=author_groups,
        )

        # Update edge attributes on the live graph
        graph[u][v][key]["confidence_mean"] = posterior.mean
        graph[u][v][key]["controversy_score"] = metrics["controversy_score"]
        graph[u][v][key]["evidence_diversity"] = metrics["evidence_diversity"]
        graph[u][v][key]["independent_source_count"] = metrics["independent_source_count"]
        graph[u][v][key]["evidence_count"] = metrics["evidence_count"]
        updated += 1

    log.info("kg.edges_scored", n_edges=updated)
    return graph
