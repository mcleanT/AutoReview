"""Ingest extraction JSON files into raw entity, assertion, evidence, and citation records.

Reads extraction JSONs produced by the LLM extraction stage and returns flat dicts
suitable for downstream deduplication and graph construction. Does NOT perform
dedup or graph construction — only parsing and normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Valid evidence strength values (mirrors EvidenceStrength enum in models.py)
_VALID_EVIDENCE_STRENGTHS = {
    "direct_experimental",
    "indirect_experimental",
    "observational_controlled",
    "observational_uncontrolled",
    "computational_prediction",
    "expert_opinion",
    "review_citation",
}


def _normalize_ontology_source(raw: str | None) -> str | None:
    """Normalize ontology_source: split on ';' and ',', strip, uppercase, take first."""
    if not raw:
        return raw
    # Split on both semicolons and commas
    parts = [p.strip().upper() for p in raw.replace(",", ";").split(";") if p.strip()]
    return parts[0] if parts else None


def _parse_entity(entity_dict: dict, paper_hash: str) -> dict:
    """Convert a raw entity sub-dict from an assertion draft into a flat entity dict.

    Handles both v4 format (canonical_name/entity_type) and v5 format (name/type).
    """
    raw_source = entity_dict.get("ontology_source")
    return {
        # v4 uses canonical_name/entity_type; v5 uses name/type — fall back gracefully
        "canonical_name": entity_dict.get("canonical_name") or entity_dict.get("name", ""),
        "entity_type": entity_dict.get("entity_type") or entity_dict.get("type", "other"),
        "ontology_id": entity_dict.get("ontology_id"),
        "ontology_source": _normalize_ontology_source(raw_source),
        "aliases": entity_dict.get("aliases") or [],
        "paper_ids": [paper_hash],
        "surface_form": entity_dict.get("surface_form", ""),
    }


def _parse_assertion(draft: dict, paper_hash: str, publication_date: str | None) -> dict:
    """Convert a raw assertion_draft (v4) or claim (v5) dict into a flat assertion dict.

    v4 fields: draft_id, subject_entity, object_entity, assertion_type, evidence_unit_ids
    v5 fields: claim_id, subject, object, claim_type, evidence_links, certainty,
               section_source, model_system, organism, quantitative_context
    """
    predicate = (draft.get("predicate") or "").strip()
    if not predicate:
        # v4 uses draft_id; v5 uses claim_id
        logger.warning(
            "empty_predicate_replaced",
            draft_id=draft.get("draft_id") or draft.get("claim_id"),
            paper_hash=paper_hash,
        )
        predicate = "related_to"

    direction = draft.get("direction")
    if direction == "null":
        direction = None

    # v4 uses subject_entity/object_entity; v5 uses subject/object
    subject = draft.get("subject_entity") or draft.get("subject") or {}
    obj = draft.get("object_entity") or draft.get("object") or {}

    # v4 uses draft_id; v5 uses claim_id
    raw_draft_id = draft.get("draft_id") or draft.get("claim_id", "")
    namespaced_draft_id = f"{paper_hash}::{raw_draft_id}" if raw_draft_id else ""

    # v4 uses assertion_type; v5 uses claim_type
    assertion_type = draft.get("assertion_type") or draft.get("claim_type", "")

    # Evidence unit IDs: v4 uses evidence_unit_ids (list of strings),
    # v5 uses evidence_links (list of {"evidence_id": ..., "direction": ...} dicts)
    evidence_unit_ids_raw = draft.get("evidence_unit_ids")
    if evidence_unit_ids_raw is None:
        evidence_links = draft.get("evidence_links") or []
        evidence_unit_ids_raw = [
            link["evidence_id"]
            for link in evidence_links
            if isinstance(link, dict) and "evidence_id" in link
        ]

    # Derive model_system / organism / in_vitro from multiple sources.
    # v5 claim-level fields take priority; fall back to v4 conditions sub-dict.
    model_system = draft.get("model_system")
    organism = draft.get("organism")
    in_vitro: bool | None = None

    conditions = draft.get("conditions") or {}
    if not organism and conditions.get("species"):
        species_list = conditions["species"]
        if isinstance(species_list, list) and species_list:
            organism = species_list[0]
    in_vitro = conditions.get("in_vitro")

    # canonical_name resolution works for both v4 (canonical_name) and v5 (name)
    subject_name = subject.get("canonical_name") or subject.get("name", "")
    object_name = obj.get("canonical_name") or obj.get("name", "")

    return {
        "draft_id": namespaced_draft_id,
        "subject_canonical_name": subject_name,
        "object_canonical_name": object_name,
        "predicate": predicate,
        "direction": direction,
        "assertion_type": assertion_type,
        "evidence_unit_ids": [f"{paper_hash}::{eid}" for eid in evidence_unit_ids_raw],
        "paper_id": paper_hash,
        "publication_date": publication_date,
        # v4 fields (may be None in v5 data without these fields)
        "natural_language": draft.get("natural_language", ""),
        "negatable_form": draft.get("negatable_form"),
        "hedging": draft.get("hedging"),
        # v5 fields (None for v4 data)
        "certainty": draft.get("certainty"),
        "section_source": draft.get("section_source"),
        "causal_type": draft.get("causal_type"),
        "conditions": conditions if conditions else None,
        "model_system": model_system,
        "organism": organism,
        "in_vitro": in_vitro,
        "quantitative_context": draft.get("quantitative_context"),
    }


def _parse_evidence_unit(ev: dict, paper_hash: str, publication_date: str | None) -> dict:
    """Convert a raw evidence_unit (v4) or evidence (v5) dict into a flat evidence dict.

    v4 format: nested experiment/results sub-dicts.
    v5 format: flat top-level fields (description, result_summary, model_system, etc.).
    """
    strength = ev.get("evidence_strength", "")
    if strength not in _VALID_EVIDENCE_STRENGTHS:
        logger.warning(
            "unknown_evidence_strength_replaced",
            original=strength,
            evidence_id=ev.get("evidence_id"),
            paper_hash=paper_hash,
        )
        strength = "expert_opinion"

    # --- Build experiment_summary from v4 nested OR v5 flat format ---
    experiment = ev.get("experiment") or {}
    results = ev.get("results") or {}

    if experiment or results:
        # v4 nested format
        experiment_parts = []
        if experiment.get("description"):
            experiment_parts.append(experiment["description"])
        if experiment.get("readout"):
            experiment_parts.append(f"Readout: {experiment['readout']}")
        if results.get("effect_description"):
            experiment_parts.append(f"Result: {results['effect_description']}")
        experiment_summary = "; ".join(experiment_parts) if experiment_parts else ""
        model_system = experiment.get("model_system")
        organism = experiment.get("organism")
        sample_size = results.get("sample_size")
        key_figure = results.get("key_figure")
    else:
        # v5 flat format — fields at top level
        parts = []
        if ev.get("description"):
            parts.append(ev["description"])
        if ev.get("result_summary"):
            parts.append(ev["result_summary"])
        experiment_summary = "; ".join(parts) if parts else ""
        model_system = ev.get("model_system")
        organism = ev.get("organism")
        sample_size = ev.get("sample_size")
        key_figure = ev.get("key_figure")

    # Namespace evidence_id with paper_hash to avoid collisions across papers
    raw_eid = ev.get("evidence_id", "")
    namespaced_eid = f"{paper_hash}::{raw_eid}" if raw_eid else ""

    return {
        "evidence_id": namespaced_eid,
        "paper_id": paper_hash,
        "evidence_strength": strength,
        "evidence_direction": ev.get("evidence_direction", ""),
        "experiment_summary": experiment_summary,
        "model_system": model_system,
        "organism": organism,
        "sample_size": sample_size,
        "key_figure": key_figure,
        "publication_date": publication_date,
        "assertion_draft_ids": [
            f"{paper_hash}::{aid}" for aid in (ev.get("assertion_draft_ids") or [])
        ],
        # Citation stub fields (None for experimental evidence)
        "citing_sentence": ev.get("citing_sentence"),
        "source_doi": ev.get("source_doi"),
        "section": ev.get("section"),
    }


def _parse_citation(cit: dict, paper_hash: str) -> dict:
    """Convert a raw citation_context dict into a flat citation dict."""
    return {
        "citation_id": cit.get("citation_id", ""),
        "citing_paper_id": paper_hash,
        "cited_source_doi": cit.get("cited_source_doi"),
        "cited_source_pmid": cit.get("cited_source_pmid"),
        "citing_sentence": cit.get("citing_sentence", ""),
        "cited_claim_paraphrase": cit.get("cited_claim_paraphrase"),
        "relationship": cit.get("relationship", ""),
        "linked_assertion_draft_ids": cit.get("linked_assertion_draft_ids") or [],
        "section": cit.get("section"),
    }


@dataclass
class ExtractionRecord:
    """Parsed data from one extraction JSON file."""

    paper_hash: str
    entities: list[dict] = field(default_factory=list)
    assertions: list[dict] = field(default_factory=list)
    evidence_units: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


@dataclass
class CorpusIngestion:
    """Accumulated ingestion results across all papers in a directory."""

    paper_count: int = 0
    all_entities: list[dict] = field(default_factory=list)
    all_assertions: list[dict] = field(default_factory=list)
    all_evidence_units: list[dict] = field(default_factory=list)
    all_citations: list[dict] = field(default_factory=list)
    provenance_by_paper: dict[str, dict] = field(default_factory=dict)
    parse_errors: list[dict] = field(default_factory=list)


def ingest_extraction(data: dict, paper_hash: str) -> ExtractionRecord:
    """Parse and normalize a single extraction JSON dict.

    Args:
        data: Parsed extraction JSON dict.
        paper_hash: Identifier for the source paper (typically the filename stem).

    Returns:
        ExtractionRecord with normalized entities, assertions, evidence_units,
        citations, and provenance.
    """
    # Detect format: v4 uses assertion_drafts/evidence_units/paper_provenance;
    # v5 uses claims/evidence and has provenance fields at the top level.
    drafts = data.get("assertion_drafts") or data.get("claims") or []
    evidence_units_raw = data.get("evidence_units") or data.get("evidence") or []

    provenance = data.get("paper_provenance") or {}
    if not provenance and data.get("doi"):
        # v5 flat format — build a provenance dict from top-level fields
        provenance = {
            "doi": data.get("doi", ""),
            "title": data.get("title", ""),
            "journal": data.get("journal", ""),
            "publication_date": data.get("publication_date"),
        }
    publication_date: str | None = provenance.get("publication_date")

    entities: list[dict] = []
    assertions: list[dict] = []

    for draft in drafts:
        # v4 uses subject_entity/object_entity; v5 uses subject/object
        subject_raw = draft.get("subject_entity") or draft.get("subject") or {}
        object_raw = draft.get("object_entity") or draft.get("object") or {}

        entities.append(_parse_entity(subject_raw, paper_hash))
        entities.append(_parse_entity(object_raw, paper_hash))
        assertions.append(_parse_assertion(draft, paper_hash, publication_date))

    evidence_units: list[dict] = [
        _parse_evidence_unit(ev, paper_hash, publication_date) for ev in evidence_units_raw
    ]

    citations: list[dict] = [
        _parse_citation(cit, paper_hash) for cit in (data.get("citation_contexts") or [])
    ]

    logger.info(
        "ingest_extraction_complete",
        paper_hash=paper_hash,
        entities=len(entities),
        assertions=len(assertions),
        evidence_units=len(evidence_units),
        citations=len(citations),
    )

    return ExtractionRecord(
        paper_hash=paper_hash,
        entities=entities,
        assertions=assertions,
        evidence_units=evidence_units,
        citations=citations,
        provenance=provenance,
    )


def ingest_directory(extraction_dir: Path) -> CorpusIngestion:
    """Ingest all extraction JSON files in a directory.

    Args:
        extraction_dir: Path to directory containing ``*.json`` extraction files.

    Returns:
        CorpusIngestion accumulating all records across valid JSON files.
        Parse errors are collected in ``result.parse_errors`` — processing continues
        even when individual files are malformed.
    """
    result = CorpusIngestion()
    json_files = sorted(extraction_dir.glob("*.json"))

    logger.info("ingest_directory_start", path=str(extraction_dir), file_count=len(json_files))

    for json_path in json_files:
        paper_hash = json_path.stem
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "ingest_parse_error",
                path=str(json_path),
                error=str(exc),
            )
            result.parse_errors.append({"path": str(json_path), "error": str(exc)})
            continue

        record = ingest_extraction(data, paper_hash)
        result.paper_count += 1
        result.all_entities.extend(record.entities)
        result.all_assertions.extend(record.assertions)
        result.all_evidence_units.extend(record.evidence_units)
        result.all_citations.extend(record.citations)
        result.provenance_by_paper[paper_hash] = record.provenance

    logger.info(
        "ingest_directory_complete",
        papers=result.paper_count,
        entities=len(result.all_entities),
        assertions=len(result.all_assertions),
        evidence_units=len(result.all_evidence_units),
        parse_errors=len(result.parse_errors),
    )

    return result
