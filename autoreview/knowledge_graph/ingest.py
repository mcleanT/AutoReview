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
    "observational_controlled",
    "observational_uncontrolled",
    "computational_prediction",
    "expert_opinion",
}


def _normalize_ontology_source(raw: str | None) -> str | None:
    """Normalize ontology_source: split on ';' and ',', strip, uppercase, take first."""
    if not raw:
        return raw
    # Split on both semicolons and commas
    parts = [p.strip().upper() for p in raw.replace(",", ";").split(";") if p.strip()]
    return parts[0] if parts else None


def _parse_entity(entity_dict: dict, paper_hash: str) -> dict:
    """Convert a raw entity sub-dict from an assertion draft into a flat entity dict."""
    raw_source = entity_dict.get("ontology_source")
    return {
        "canonical_name": entity_dict.get("canonical_name", ""),
        "entity_type": entity_dict.get("entity_type", "other"),
        "ontology_id": entity_dict.get("ontology_id"),
        "ontology_source": _normalize_ontology_source(raw_source),
        "aliases": entity_dict.get("aliases") or [],
        "paper_ids": [paper_hash],
        "surface_form": entity_dict.get("surface_form", ""),
    }


def _parse_assertion(draft: dict, paper_hash: str, publication_date: str | None) -> dict:
    """Convert a raw assertion_draft dict into a flat assertion dict."""
    predicate = (draft.get("predicate") or "").strip()
    if not predicate:
        logger.warning(
            "empty_predicate_replaced",
            draft_id=draft.get("draft_id"),
            paper_hash=paper_hash,
        )
        predicate = "related_to"

    direction = draft.get("direction")
    if direction == "null":
        direction = None

    subject = draft.get("subject_entity") or {}
    obj = draft.get("object_entity") or {}

    return {
        "draft_id": draft.get("draft_id", ""),
        "subject_canonical_name": subject.get("canonical_name", ""),
        "object_canonical_name": obj.get("canonical_name", ""),
        "predicate": predicate,
        "direction": direction,
        "assertion_type": draft.get("assertion_type", ""),
        "evidence_unit_ids": draft.get("evidence_unit_ids") or [],
        "paper_id": paper_hash,
        "publication_date": publication_date,
    }


def _parse_evidence_unit(ev: dict, paper_hash: str, publication_date: str | None) -> dict:
    """Convert a raw evidence_unit dict into a flat evidence dict."""
    strength = ev.get("evidence_strength", "")
    if strength not in _VALID_EVIDENCE_STRENGTHS:
        logger.warning(
            "unknown_evidence_strength_replaced",
            original=strength,
            evidence_id=ev.get("evidence_id"),
            paper_hash=paper_hash,
        )
        strength = "expert_opinion"

    experiment = ev.get("experiment") or {}
    results = ev.get("results") or {}

    experiment_parts = []
    if experiment.get("description"):
        experiment_parts.append(experiment["description"])
    if experiment.get("readout"):
        experiment_parts.append(f"Readout: {experiment['readout']}")
    experiment_summary = "; ".join(experiment_parts) if experiment_parts else ""

    return {
        "evidence_id": ev.get("evidence_id", ""),
        "paper_id": paper_hash,
        "evidence_strength": strength,
        "evidence_direction": ev.get("evidence_direction", ""),
        "experiment_summary": experiment_summary,
        "model_system": experiment.get("model_system"),
        "sample_size": results.get("sample_size"),
        "key_figure": results.get("key_figure"),
        "publication_date": publication_date,
        "assertion_draft_ids": ev.get("assertion_draft_ids") or [],
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
    provenance = data.get("paper_provenance") or {}
    publication_date: str | None = provenance.get("publication_date")

    entities: list[dict] = []
    assertions: list[dict] = []

    for draft in data.get("assertion_drafts") or []:
        subject_raw = draft.get("subject_entity") or {}
        object_raw = draft.get("object_entity") or {}

        entities.append(_parse_entity(subject_raw, paper_hash))
        entities.append(_parse_entity(object_raw, paper_hash))
        assertions.append(_parse_assertion(draft, paper_hash, publication_date))

    evidence_units: list[dict] = [
        _parse_evidence_unit(ev, paper_hash, publication_date)
        for ev in (data.get("evidence_units") or [])
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
