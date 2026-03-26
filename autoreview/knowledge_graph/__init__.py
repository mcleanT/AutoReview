"""Knowledge graph construction and analysis from extraction data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import structlog

from autoreview.knowledge_graph.confidence import score_all_edges
from autoreview.knowledge_graph.dedup import (
    deduplicate_entities,
    merge_assertions,
    normalize_predicate,
)
from autoreview.knowledge_graph.graph import (
    build_nx_graph,
)
from autoreview.knowledge_graph.graph import (
    load_graph as _load_graph,
)
from autoreview.knowledge_graph.graph import (
    save_graph as _save_graph,
)
from autoreview.knowledge_graph.ingest import ingest_directory
from autoreview.knowledge_graph.models import (
    AssertionType,
    BetaPosterior,
    KGEdge,
    KGEvidenceLink,
)

log = structlog.get_logger(__name__)

__all__ = ["build_graph", "load_graph", "save_graph"]


def build_graph(extraction_dir: Path) -> nx.MultiDiGraph:
    """Full pipeline: ingest → dedup → graph → confidence.

    Steps:
    1. Ingest all extraction JSONs in extraction_dir.
    2. Deduplicate entities across the corpus.
    3. Normalize predicates in all assertions.
    4. Remap subject/object canonical names to deduplicated entity IDs.
    5. Merge assertions sharing the same (subject_id, predicate, object_id) triple.
    6. Build KGEdge model instances from merged assertions + evidence units.
    7. Build the NetworkX MultiDiGraph from entities and edges.
    8. Score edge confidence with Beta-Binomial posteriors.

    Args:
        extraction_dir: Path to a directory of ``*.json`` extraction files.

    Returns:
        A scored NetworkX MultiDiGraph ready for analysis and serialization.
    """
    extraction_dir = Path(extraction_dir)

    # ------------------------------------------------------------------
    # Step 1: Ingest
    # ------------------------------------------------------------------
    corpus = ingest_directory(extraction_dir)
    log.info(
        "kg.pipeline.ingest_done",
        papers=corpus.paper_count,
        entities=len(corpus.all_entities),
        assertions=len(corpus.all_assertions),
        evidence_units=len(corpus.all_evidence_units),
    )

    # ------------------------------------------------------------------
    # Step 2: Entity deduplication
    # ------------------------------------------------------------------
    registry = deduplicate_entities(corpus.all_entities)
    log.info(
        "kg.pipeline.dedup_done",
        unique_entities=len(registry.entities),
        merges=len(registry.merge_log),
    )

    # ------------------------------------------------------------------
    # Step 3 + 4: Normalize predicates and remap entity IDs
    # ------------------------------------------------------------------
    normalized_assertions: list[dict] = []
    skipped = 0
    for assertion in corpus.all_assertions:
        subj_name = (assertion.get("subject_canonical_name") or "").lower()
        obj_name = (assertion.get("object_canonical_name") or "").lower()

        subj_id = registry.surface_to_id.get(subj_name)
        obj_id = registry.surface_to_id.get(obj_name)

        if subj_id is None or obj_id is None:
            log.warning(
                "kg.pipeline.entity_lookup_failed",
                subject=subj_name,
                object=obj_name,
                draft_id=assertion.get("draft_id"),
            )
            skipped += 1
            continue

        norm_predicate = normalize_predicate(assertion["predicate"])
        normalized_assertions.append(
            {
                **assertion,
                "subject_id": subj_id,
                "object_id": obj_id,
                "predicate": norm_predicate,
            }
        )

    log.info(
        "kg.pipeline.normalize_done",
        normalized=len(normalized_assertions),
        skipped=skipped,
    )

    # ------------------------------------------------------------------
    # Step 5: Assertion merging
    # ------------------------------------------------------------------
    merge_result = merge_assertions(normalized_assertions)
    log.info(
        "kg.pipeline.merge_done",
        merged_assertions=len(merge_result.assertions),
        merges=len(merge_result.merge_log),
    )

    # ------------------------------------------------------------------
    # Step 6: Build KGEdge instances
    # ------------------------------------------------------------------
    # Index evidence units by evidence_id for fast lookup
    evidence_index: dict[str, dict] = {ev["evidence_id"]: ev for ev in corpus.all_evidence_units}

    kg_edges: list[KGEdge] = []
    for merged in merge_result.assertions:
        subject_id = merged["subject_id"]
        object_id = merged["object_id"]
        predicate = merged["predicate"]

        # Deterministic edge_id from the canonical triple
        edge_key = f"{subject_id}|{predicate}|{object_id}"
        edge_id = hashlib.sha256(edge_key.encode()).hexdigest()[:16]

        # Build evidence links
        evidence_links: list[KGEvidenceLink] = []
        for ev_id in merged.get("evidence_unit_ids") or []:
            ev_dict = evidence_index.get(ev_id)
            if ev_dict is None:
                log.warning("kg.pipeline.evidence_unit_not_found", evidence_id=ev_id)
                continue
            evidence_links.append(
                KGEvidenceLink(
                    evidence_id=ev_dict["evidence_id"],
                    paper_id=ev_dict["paper_id"],
                    evidence_strength=ev_dict["evidence_strength"],
                    evidence_direction=ev_dict.get("evidence_direction", ""),
                    experiment_summary=ev_dict.get("experiment_summary", ""),
                    model_system=ev_dict.get("model_system"),
                    sample_size=ev_dict.get("sample_size"),
                    key_figure=ev_dict.get("key_figure"),
                    publication_date=ev_dict.get("publication_date"),
                )
            )

        # Resolve assertion_type safely
        raw_assertion_type = merged.get("assertion_type", "mechanistic_causal")
        try:
            assertion_type = AssertionType(raw_assertion_type)
        except ValueError:
            assertion_type = AssertionType.mechanistic_causal

        kg_edges.append(
            KGEdge(
                edge_id=edge_id,
                subject_id=subject_id,
                object_id=object_id,
                predicate=predicate,
                direction=merged.get("direction"),
                assertion_type=assertion_type,
                confidence=BetaPosterior(),
                evidence_links=evidence_links,
                source_assertions=merged.get("source_assertions") or [],
                publication_date=merged.get("publication_date"),
            )
        )

    log.info("kg.pipeline.edges_built", n_edges=len(kg_edges))

    # ------------------------------------------------------------------
    # Step 7: Build NetworkX graph
    # ------------------------------------------------------------------
    G = build_nx_graph(registry.entities, kg_edges)

    # ------------------------------------------------------------------
    # Step 8: Score confidence
    # ------------------------------------------------------------------
    G = score_all_edges(G, corpus.provenance_by_paper)

    log.info(
        "kg.pipeline.complete",
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    return G


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Serialize the graph to pickle and GraphML.

    Args:
        graph: The MultiDiGraph produced by :func:`build_graph`.
        path: Base path without extension. Creates ``{path}.pkl`` and ``{path}.graphml``.
    """
    _save_graph(graph, path)


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a graph from a pickle file.

    Args:
        path: Path to the ``.pkl`` file produced by :func:`save_graph`.

    Returns:
        The deserialized MultiDiGraph.
    """
    return _load_graph(path)
