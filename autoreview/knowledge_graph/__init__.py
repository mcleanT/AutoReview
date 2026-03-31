"""Knowledge graph construction and analysis from extraction data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

import networkx as nx
import structlog

from autoreview.knowledge_graph.analysis import (
    score_contradiction_centrality,
    score_finding_contradiction_centrality,
    summarize_topic_clusters,
)
from autoreview.knowledge_graph.cluster import (
    Finding,
    FindingContradiction,
    TopicCluster,
    build_topic_clusters,
    detect_finding_contradictions,
    form_findings,
    get_predicate_class,
)
from autoreview.knowledge_graph.confidence import score_all_edges
from autoreview.knowledge_graph.dedup import (
    EntityRegistry,
    deduplicate_entities,
    merge_assertions,
    merge_assertions_v2,
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
    ConditionContext,
    KGEdge,
    KGEvidenceLink,
    QuantitativeContext,
    SectionSource,
)
from autoreview.knowledge_graph.mrf_scoring import MRFConfig, update_graph_mrf
from autoreview.knowledge_graph.mrf_weight_learning import (
    WeightLearningConfig,
    learn_weights,
)
from autoreview.knowledge_graph.nli import (
    NLIConfig,
    classify_cross_claims,
    diagnose_evidence_directions,
)
from autoreview.knowledge_graph.normalize import ClaimNormalizer, NormalizationReport

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
    "update_graph_mrf",
    "WeightLearningConfig",
    "learn_weights",
    "score_contradiction_centrality",
    "score_graph_bayesian",
    "update_graph_bayesian",
    "BayesianConfig",
    "BayesianResult",
    "TopicCluster",
    "Finding",
    "FindingContradiction",
    "build_topic_clusters",
    "form_findings",
    "detect_finding_contradictions",
    "get_predicate_class",
    "score_finding_contradiction_centrality",
    "summarize_topic_clusters",
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
    version: int = 1,
    normalize: bool = False,
    llm_decompose: bool = True,
    bayesian: bool = False,
    bayesian_config: BayesianConfig | None = None,
) -> nx.MultiDiGraph:
    """Full pipeline: ingest → dedup → graph → confidence.

    Steps:
    1. Ingest all extraction JSONs in extraction_dir.
    2. Deduplicate entities across the corpus.
    3. Normalize predicates in all assertions.
    4. Remap subject/object canonical names to deduplicated entity IDs.
    5. Merge assertions sharing the same (subject_id, predicate, object_id) triple.
       When version >= 2, uses condition-aware merging (merge_assertions_v2) which
       also groups by condition_signature so claims with different experimental
       contexts become separate edges.
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
        version: Pipeline version. ``1`` (default) uses the original
            condition-agnostic merging. ``2`` uses condition-aware merging
            (``merge_assertions_v2``) which creates separate edges for claims
            with distinct experimental contexts (organism, in_vitro, model_system).
        normalize: If ``True``, run post-extraction normalization (text cleaning,
            predicate cleaning, compound decomposition, quantitative backfill)
            to improve cross-paper claim matching. Default ``False``.
        llm_decompose: If ``True`` and ``normalize=True``, use LLM fallback
            for compound objects that rule-based patterns can't handle.
            Default ``True``.
        bayesian: If ``True``, run Bayesian inference (Laplace + targeted NUTS)
            after Beta-Binomial scoring and write ``bayesian_confidence``,
            ``bayesian_ci_low``, ``bayesian_ci_high``, and ``bayesian_bimodal``
            onto each edge. Requires ``pip install autoreview[bayesian]``.
        bayesian_config: Optional :class:`BayesianConfig` to control Bayesian
            inference hyperparameters. Uses defaults when ``None``.

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
    # Step 1b: Pre-dedup normalization (text cleaning + decomposition)
    # ------------------------------------------------------------------
    pre_dedup_report = None
    if normalize:
        import asyncio

        normalizer = ClaimNormalizer(llm_decompose=llm_decompose)
        corpus.all_entities, corpus.all_assertions, pre_dedup_report = asyncio.run(
            normalizer.pre_dedup(corpus.all_entities, corpus.all_assertions)
        )
        log.info(
            "kg.pipeline.pre_dedup_normalize",
            text_cleaned=pre_dedup_report.text_cleaned,
            predicates_cleaned=pre_dedup_report.predicates_cleaned,
            claims_decomposed=pre_dedup_report.claims_decomposed,
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
    # Step 3b: Post-dedup normalization (quantitative backfill)
    # ------------------------------------------------------------------
    post_dedup_report = None
    if normalize:
        import asyncio

        normalizer = ClaimNormalizer(llm_decompose=False)
        normalized_assertions, post_dedup_report = asyncio.run(
            normalizer.post_dedup(normalized_assertions)
        )
        log.info(
            "kg.pipeline.post_dedup_normalize",
            quant_backfilled=post_dedup_report.quant_backfilled,
        )

    # ------------------------------------------------------------------
    # Step 5: Assertion merging
    # ------------------------------------------------------------------
    if version >= 2:
        merge_result = merge_assertions_v2(normalized_assertions)
    else:
        merge_result = merge_assertions(normalized_assertions)
    log.info(
        "kg.pipeline.merge_done",
        merged_assertions=len(merge_result.assertions),
        merges=len(merge_result.merge_log),
        version=version,
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

        # Deterministic edge_id from the canonical triple (+ condition for v2)
        condition_sig = merged.get("condition_signature")
        if version >= 2 and condition_sig:
            edge_key = f"{subject_id}|{predicate}|{object_id}|{condition_sig}"
        else:
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

        # v2 condition fields
        raw_cc = merged.get("condition_context")
        condition_context = None
        if isinstance(raw_cc, dict):
            condition_context = ConditionContext(**raw_cc)

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
                condition_signature=merged.get("condition_signature"),
                condition_context=condition_context,
            )
        )

    log.info("kg.pipeline.edges_built", n_edges=len(kg_edges))

    # ------------------------------------------------------------------
    # Step 7: Build NetworkX graph
    # ------------------------------------------------------------------
    graph = build_nx_graph(registry.entities, kg_edges)

    # Store normalization audit trail on graph
    if normalize:
        combined_report = NormalizationReport(
            text_cleaned=pre_dedup_report.text_cleaned if pre_dedup_report else 0,
            predicates_cleaned=pre_dedup_report.predicates_cleaned if pre_dedup_report else 0,
            claims_decomposed=pre_dedup_report.claims_decomposed if pre_dedup_report else 0,
            claims_produced=pre_dedup_report.claims_produced if pre_dedup_report else 0,
            quant_backfilled=post_dedup_report.quant_backfilled if post_dedup_report else 0,
            llm_calls=pre_dedup_report.llm_calls if pre_dedup_report else 0,
        )
        graph.graph["normalization_report"] = combined_report

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
    # Step 8c: Bayesian inference (optional)
    # ------------------------------------------------------------------
    if bayesian:
        from autoreview.knowledge_graph.bayesian import (
            BayesianConfig,
            score_graph_bayesian,
        )

        b_cfg = bayesian_config or BayesianConfig()
        bayesian_result = score_graph_bayesian(graph, config=b_cfg)

        for u, v, key, data in graph.edges(data=True, keys=True):
            edge_id = data.get("edge_id")
            if edge_id and edge_id in bayesian_result.posteriors:
                graph[u][v][key]["bayesian_confidence"] = bayesian_result.posteriors[edge_id]
                if edge_id in bayesian_result.credible_intervals:
                    ci = bayesian_result.credible_intervals[edge_id]
                    graph[u][v][key]["bayesian_ci_low"] = ci[0]
                    graph[u][v][key]["bayesian_ci_high"] = ci[1]
                if edge_id in bayesian_result.bimodality_flags:
                    graph[u][v][key]["bayesian_bimodal"] = bayesian_result.bimodality_flags[edge_id]

        log.info(
            "kg.pipeline.bayesian_done",
            n_variables=bayesian_result.n_variables,
            n_contradictions=bayesian_result.n_contradictions,
            n_compositions=bayesian_result.n_compositions,
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
