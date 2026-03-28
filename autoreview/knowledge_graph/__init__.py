"""Knowledge graph construction and analysis from extraction data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import structlog

from autoreview.knowledge_graph.confidence import score_all_edges
from autoreview.knowledge_graph.dedup import (
    EntityRegistry,
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
    Certainty,
    KGEdge,
    KGEvidenceLink,
    QuantitativeContext,
    SectionSource,
)
from autoreview.knowledge_graph.mrf_scoring import MRFConfig
from autoreview.knowledge_graph.nli import (
    NLIConfig,
    classify_cross_claims,
    diagnose_evidence_directions,
)

log = structlog.get_logger(__name__)

__all__ = [
    "build_graph",
    "load_graph",
    "save_graph",
    "classify_cross_claims",
    "diagnose_evidence_directions",
    "NLIConfig",
    "SectionSource",
    "Certainty",
    "QuantitativeContext",
    "MRFConfig",
]


def _build_citation_edges(
    corpus_citations: list[dict],
    registry: EntityRegistry,
    graph: nx.MultiDiGraph,
) -> int:
    """Build cross-paper contradiction/support edges from citation contexts.

    Citation contexts where relationship='contradicts' become explicit contradiction
    signals on the graph, supplementing the NLI-based detection.

    Args:
        corpus_citations: List of citation dicts from ingestion.
        registry: Entity registry for ID lookups.
        graph: The graph to annotate (mutated in-place).

    Returns:
        Number of citation-derived edges added as graph metadata.
    """
    added = 0
    for cit in corpus_citations:
        relationship = cit.get("relationship", "")
        if relationship not in ("contradicts", "supports", "refines", "extends"):
            continue

        # Store citation context on linked assertion edges
        linked_ids = cit.get("linked_assertion_draft_ids") or []
        cited_doi = cit.get("cited_source_doi")
        citing_paper = cit.get("citing_paper_id", "")

        for edge_key_str in graph.edges(keys=True):
            u, v, k = edge_key_str
            edge_data = graph[u][v][k]
            kg_edge = edge_data.get("_kg_edge")
            if kg_edge is None:
                continue

            # Check if this edge's source assertions include any linked draft IDs
            source_assertions = kg_edge.source_assertions or []
            if any(sa in linked_ids for sa in source_assertions):
                # Add citation metadata to the edge
                citation_refs = edge_data.get("_citation_refs", [])
                citation_refs.append(
                    {
                        "relationship": relationship,
                        "cited_doi": cited_doi,
                        "citing_paper": citing_paper,
                        "citing_sentence": cit.get("citing_sentence", ""),
                        "paraphrase": cit.get("cited_claim_paraphrase"),
                    }
                )
                edge_data["_citation_refs"] = citation_refs
                added += 1

    log.info("kg.pipeline.citation_edges", citation_annotations=added)
    return added


def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
) -> nx.MultiDiGraph:
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
    8b. (Optional) MRF confidence propagation — writes ``mrf_confidence`` onto
        each edge when ``use_mrf=True``.
    9. Annotate edges with citation context.

    Args:
        extraction_dir: Path to a directory of ``*.json`` extraction files.
        use_mrf: If ``True``, run HL-MRF inference after Beta-Binomial scoring
            and write ``mrf_confidence`` posteriors back onto each edge.
        mrf_config: Optional :class:`MRFConfig` to control MRF hyperparameters.
            Uses defaults when ``None``.

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
    # Step 5b: Propagate v5 context fields to merged assertions
    # ------------------------------------------------------------------
    # Build index: draft_id → assertion for v5 field lookup
    assertion_by_draft_id: dict[str, dict] = {}
    for assertion in normalized_assertions:
        draft_id = assertion.get("draft_id", "")
        if draft_id:
            assertion_by_draft_id[draft_id] = assertion

    for merged in merge_result.assertions:
        source_drafts = merged.get("source_assertions") or []
        # Use first source assertion's v5 fields (they should be consistent)
        for draft_id in source_drafts:
            source = assertion_by_draft_id.get(draft_id)
            if source:
                for field in (
                    "certainty",
                    "section_source",
                    "model_system",
                    "organism",
                    "in_vitro",
                    "conditions",
                    "hedging",
                    "negatable_form",
                    "natural_language",
                    "quantitative_context",
                ):
                    if field not in merged or merged.get(field) is None:
                        merged[field] = source.get(field)
                break  # use first available source

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

        # Resolve v5 enums safely
        raw_certainty = merged.get("certainty")
        try:
            certainty = Certainty(raw_certainty) if raw_certainty else None
        except ValueError:
            certainty = None

        raw_section_source = merged.get("section_source")
        try:
            section_source = SectionSource(raw_section_source) if raw_section_source else None
        except ValueError:
            section_source = None

        # Resolve quantitative_context
        raw_qc = merged.get("quantitative_context")
        quantitative_context = None
        if isinstance(raw_qc, dict):
            quantitative_context = QuantitativeContext(**raw_qc)

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
                # v5 context fields
                certainty=certainty,
                section_source=section_source,
                model_system=merged.get("model_system"),
                organism=merged.get("organism"),
                in_vitro=merged.get("in_vitro"),
                conditions=merged.get("conditions"),
                hedging=merged.get("hedging"),
                negatable_form=merged.get("negatable_form"),
                natural_language=merged.get("natural_language"),
                quantitative_context=quantitative_context,
            )
        )

    log.info("kg.pipeline.edges_built", n_edges=len(kg_edges))

    # ------------------------------------------------------------------
    # Step 7: Build NetworkX graph
    # ------------------------------------------------------------------
    graph = build_nx_graph(registry.entities, kg_edges)

    # ------------------------------------------------------------------
    # Step 8: Score confidence
    # ------------------------------------------------------------------
    graph = score_all_edges(graph, corpus.provenance_by_paper)

    # ------------------------------------------------------------------
    # Step 8b: MRF confidence propagation (optional)
    # ------------------------------------------------------------------
    if use_mrf:
        from autoreview.knowledge_graph.mrf_scoring import score_graph_mrf

        mrf_cfg = mrf_config or MRFConfig()
        mrf_result = score_graph_mrf(graph, config=mrf_cfg)

        # Write MRF posteriors back to edges
        for u, v, key, data in graph.edges(data=True, keys=True):
            edge_id = data.get("edge_id")
            if edge_id and edge_id in mrf_result.posteriors:
                graph[u][v][key]["mrf_confidence"] = mrf_result.posteriors[edge_id]

        log.info(
            "kg.pipeline.mrf_done",
            n_rules=mrf_result.n_rules,
            n_contradictions=mrf_result.n_contradictions,
            n_compositions=mrf_result.n_compositions,
        )

    # ------------------------------------------------------------------
    # Step 9: Annotate edges with citation context
    # ------------------------------------------------------------------
    _build_citation_edges(corpus.all_citations, registry, graph)

    log.info(
        "kg.pipeline.complete",
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
    )
    return graph


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
