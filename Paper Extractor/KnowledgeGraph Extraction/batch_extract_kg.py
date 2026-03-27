#!/usr/bin/env python3
"""Batch KG Extraction — Anthropic Message Batches API.

Submits all full-text papers as a single batch using Haiku with prompt caching.
50% cost discount via batch processing. Uses KGExtraction schema (not ExtractionResult).

Usage:
    ANTHROPIC_API_KEY=sk-... python batch_extract_kg.py
    ANTHROPIC_API_KEY=sk-... python batch_extract_kg.py --poll BATCH_ID   # Resume polling
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]
_AUTONOMOUS_SCI_SRC = _AUTOREVIEW_ROOT.parent / "Autonomous Science" / "src"

sys.path.insert(0, str(_AUTOREVIEW_ROOT))
sys.path.insert(0, str(_AUTONOMOUS_SCI_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))

import anthropic
import structlog
from kg_schema import KGExtraction

from autoreview.config.models import SectionTruncationConfig
from autoreview.extraction.truncation import section_aware_truncate

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 16384
OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
EXTRACTION_CACHE_DIR = OUTPUT_DIR / "extractions_kg"

# SectionTruncationConfig supports intro_max_chars and methods_max_chars only
# (no per-section dict). Use those fields for the desired caps on introduction
# and methods. Results and discussion get no cap (full text kept).
TRUNCATION_CONFIG = SectionTruncationConfig(
    enabled=True,
    keep_sections=[
        "results",
        "methods",
        "discussion",
        "introduction",
    ],
    drop_sections=[
        "references",
        "acknowledgments",
        "acknowledgements",
        "supplementary",
        "abstract",
        "conclusion",
        "funding",
        "conflict of interest",
        "data availability",
        "author contributions",
        "competing interests",
        "ethics",
        "supporting information",
    ],
    intro_max_chars=4000,
    methods_max_chars=5000,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_hash(doi: str | None, title: str | None) -> str:
    raw = (doi or "").lower().strip() if doi else (title or "").lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_system_prompt() -> str:
    prompt_path = _SCRIPT_DIR / "kg_extraction_prompt.md"
    text = prompt_path.read_text()
    marker = "{PAPER_TEXT}"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip()


_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json_str(raw: str) -> str:
    raw = raw.strip()
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    fence_match = _MARKDOWN_FENCE_RE.search(raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    msg = f"No valid JSON found in response ({len(raw)} chars)"
    raise json.JSONDecodeError(msg, raw[:200], 0)


# ---------------------------------------------------------------------------
# Enum coercion maps
# ---------------------------------------------------------------------------

_VALID_CLAIM_TYPES = {
    "mechanistic_causal",
    "correlational",
    "comparative",
    "existence",
    "absence",
    "conditional",
    "methodological",
}
_CLAIM_TYPE_MAP: dict[str, str] = {
    "causal": "mechanistic_causal",
    "observational": "correlational",
}

_VALID_EVIDENCE_STRENGTHS = {
    "direct_experimental",
    "indirect_experimental",
    "observational",
    "computational",
    "review_citation",
}
_EVIDENCE_STRENGTH_MAP: dict[str, str] = {
    "systematic_review_meta_analysis": "review_citation",
    "randomized_controlled_trial": "direct_experimental",
    "observational_controlled": "observational",
    "observational_uncontrolled": "observational",
    "computational_prediction": "computational",
    "case_report": "observational",
    "expert_opinion": "review_citation",
}

_VALID_ENTITY_TYPES = {
    "protein",
    "gene",
    "rna",
    "small_molecule",
    "pathway",
    "biological_process",
    "phenotype",
    "cellular_compartment",
    "organism",
    "cell_type",
    "disease",
    "tissue",
    "method",
    "other",
}

_VALID_APPROACHES = {
    "biochemical_assay",
    "cell_biology",
    "genetics",
    "omics",
    "imaging",
    "computational",
    "clinical",
    "animal_model",
    "in_vitro_model",
    "structural_biology",
    "pharmacology",
}

_VALID_SECTION_SOURCES = {
    "primary_empirical",
    "interpretive",
    "attributed_prior",
    "methodological",
}
_SECTION_SOURCE_MAP: dict[str, str] = {
    "results": "primary_empirical",
    "novel_finding": "primary_empirical",
    "discussion": "interpretive",
    "interpretation": "interpretive",
    "hypothesis": "interpretive",
    "background": "attributed_prior",
    "introduction": "attributed_prior",
    "methods": "methodological",
    "methodological_note": "methodological",
}

_VALID_RESULT_DIRECTIONS = {"positive", "negative", "null", "not_reported"}

_VALID_EVIDENCE_DIRECTIONS = {"supports", "refutes", "mixed", "not_applicable"}

_VALID_CERTAINTIES = {"high", "medium", "low"}

_VALID_DIRECTIONS = {"positive", "negative"}

_VALID_PREDICATES = {
    "activates",
    "inhibits",
    "binds_to",
    "localizes_to",
    "is_required_for",
    "promotes",
    "regulates",
    "colocalizes_with",
    "phosphorylates",
    "is_expressed_in",
    "interacts_with",
    "suppresses",
    "induces",
    "differentiates_into",
    "is_marker_of",
    "correlates_with",
    "is_sufficient_for",
    "is_necessary_for",
    "upregulates",
    "downregulates",
    "is_component_of",
    "degrades",
    "stabilizes",
    "transports",
    "modifies",
    "converts",
    "mediates",
    "blocks",
    "enhances",
    "reduces",
    "maintains",
    "disrupts",
    "enables",
    "prevents",
}

# Maps invalid predicates Haiku commonly generates to valid ones.
# Entries with a tuple value (predicate, direction) also override direction.
# Entries with a plain string value only remap the predicate.
_PREDICATE_COERCION_MAP: dict[str, str | tuple[str, str]] = {
    # "lacks" → X does not maintain Y; direction flipped to negative
    "lacks": ("maintains", "negative"),
    "forms": "induces",
    "contains": "is_component_of",
    "generates": "induces",
    "expresses": "is_expressed_in",
    "represses": "suppresses",
    "models": "correlates_with",
    "recapitulates": "correlates_with",
    "is_active": "is_expressed_in",
    "is_active_in": "is_expressed_in",
    # v4 regressions
    "develops": "induces",
    "exhibits": "maintains",
    "differs": "correlates_with",
    "provides": "enables",
    "controls": "regulates",
}

_VALID_CAUSAL_TYPES = {
    "necessary",
    "sufficient",
    "necessary_and_sufficient",
    "contributory",
    "modulatory",
}


# ---------------------------------------------------------------------------
# Coercion function
# ---------------------------------------------------------------------------


def _coerce_kg_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Pre-process LLM JSON to match KGExtraction schema."""
    # Ensure top-level required strings
    for key in ("doi", "title", "journal", "publication_date"):
        if key not in d or d[key] is None:
            d[key] = ""

    # Ensure lists
    d.setdefault("claims", [])
    d.setdefault("evidence", [])

    # Coerce evidence units
    for ev in d.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        ev.setdefault("evidence_id", "e_000")
        ev.setdefault("description", "")
        ev.setdefault("result_summary", "")
        ev.setdefault("model_system", "")
        ev.setdefault("organism", "")
        ev.setdefault("readout", "")
        ev.setdefault("assay_types", [])

        # Coerce result_direction
        rd = ev.get("result_direction", "not_reported")
        if rd not in _VALID_RESULT_DIRECTIONS:
            ev["result_direction"] = "not_reported"

        # Coerce approach
        approach = ev.get("approach", "cell_biology")
        if approach not in _VALID_APPROACHES:
            ev["approach"] = "cell_biology"  # safe default

    # Coerce claims
    for claim in d.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim.setdefault("claim_id", "c_000")
        claim.setdefault("natural_language", "")
        claim.setdefault("predicate", "unknown")

        # evidence_ids → evidence_links migration
        if "evidence_ids" in claim and "evidence_links" not in claim:
            claim["evidence_links"] = [
                {"evidence_id": eid, "direction": "supports"} for eid in claim.pop("evidence_ids")
            ]
        elif "evidence_ids" in claim and "evidence_links" in claim:
            claim.pop("evidence_ids")  # prefer evidence_links if both present

        claim.setdefault("evidence_links", [])

        # Coerce each evidence_link object
        coerced_links = []
        for link in claim["evidence_links"]:
            if isinstance(link, str):
                # bare string → object
                link = {"evidence_id": link, "direction": "supports"}
            if link.get("direction") not in _VALID_EVIDENCE_DIRECTIONS:
                link["direction"] = "supports"  # safe default
            coerced_links.append(link)
        claim["evidence_links"] = coerced_links

        # Post-processing: auto-set "refutes" for absence claims
        # If a claim is typed as "absence", the evidence demonstrates that
        # something does NOT hold — it refutes the positive form.
        # Note: direction="negative" alone is NOT sufficient — a negative
        # correlation (anti-correlation) is a real finding that evidence
        # supports, not refutes.
        claim_ct = claim.get("claim_type", "existence")
        if claim_ct == "absence":
            for link in claim["evidence_links"]:
                if link.get("direction") == "supports":
                    link["direction"] = "refutes"

        # Coerce direction
        direction = claim.get("direction", "positive")
        if direction not in _VALID_DIRECTIONS:
            claim["direction"] = "positive"

        # Coerce predicate — map invalid values to nearest valid predicate
        predicate = claim.get("predicate", "maintains")
        if predicate not in _VALID_PREDICATES:
            coercion = _PREDICATE_COERCION_MAP.get(predicate)
            if isinstance(coercion, tuple):
                claim["predicate"], claim["direction"] = coercion
            elif isinstance(coercion, str):
                claim["predicate"] = coercion
            # Unknown invalid predicate: leave as-is (schema will reject if truly invalid)

        # Coerce claim_type
        ct = claim.get("claim_type", "existence")
        if ct not in _VALID_CLAIM_TYPES:
            claim["claim_type"] = _CLAIM_TYPE_MAP.get(ct, "existence")

        # Coerce causal_type
        ctype = claim.get("causal_type")
        if ctype is not None and ctype not in _VALID_CAUSAL_TYPES:
            claim["causal_type"] = None

        # Coerce evidence_strength
        es = claim.get("evidence_strength", "direct_experimental")
        if es not in _VALID_EVIDENCE_STRENGTHS:
            claim["evidence_strength"] = _EVIDENCE_STRENGTH_MAP.get(es, "direct_experimental")

        # Coerce certainty
        cert = claim.get("certainty", "medium")
        if cert not in _VALID_CERTAINTIES:
            claim["certainty"] = "medium"

        # Coerce section_source
        ss = claim.get("section_source", "primary_empirical")
        if ss not in _VALID_SECTION_SOURCES:
            claim["section_source"] = _SECTION_SOURCE_MAP.get(ss, "primary_empirical")

        # Coerce entities
        for entity_key in ("subject", "object"):
            ent = claim.get(entity_key)
            if ent is None:
                claim[entity_key] = {"name": "unknown", "type": "other", "ontology_id": None}
            elif isinstance(ent, dict):
                ent.setdefault("name", "unknown")
                ent.setdefault("type", "other")
                etype = ent.get("type", "other")
                if etype not in _VALID_ENTITY_TYPES:
                    ent["type"] = "other"

        # Coerce conditions
        conds = claim.get("conditions")
        if conds is None:
            claim["conditions"] = {}
        elif isinstance(conds, dict):
            # Ensure list fields
            for list_key in ("species", "cell_type", "tissue", "treatment"):
                val = conds.get(list_key)
                if val is None:
                    conds[list_key] = []
                elif isinstance(val, str):
                    conds[list_key] = [val]

    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def submit_batch(api_key: str) -> str:
    """Build and submit the batch extraction request. Returns batch ID."""
    client = anthropic.Anthropic(api_key=api_key)

    # Load corpus
    papers_path = OUTPUT_DIR / "papers.json"
    with open(papers_path) as f:
        papers = json.load(f)
    logger.info("corpus.loaded", total=len(papers))

    # Load system prompt
    system_prompt = _load_system_prompt()
    logger.info("prompt.loaded", chars=len(system_prompt))

    # Build batch requests — skip already-cached extractions
    EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    requests = []
    skipped_cached = 0
    skipped_no_text = 0

    for paper in papers:
        doi = paper.get("doi")
        title = paper.get("title")
        phash = _paper_hash(doi, title)

        # Skip if already extracted
        cache_path = EXTRACTION_CACHE_DIR / f"{phash}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                KGExtraction.model_validate(cached)
                skipped_cached += 1
                continue
            except Exception:
                pass  # Re-extract on cache corruption

        # Get text
        text = paper.get("full_text", "")
        if not text:
            skipped_no_text += 1
            continue

        # Truncate — 100,000 chars to give more room for claims
        if len(text) > 5000:
            text = section_aware_truncate(text, 100_000, TRUNCATION_CONFIG)

        requests.append(
            {
                "custom_id": phash,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": 0.0,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [{"role": "user", "content": text}],
                },
            }
        )

    logger.info(
        "batch.prepared",
        to_extract=len(requests),
        skipped_cached=skipped_cached,
        skipped_no_text=skipped_no_text,
    )

    if not requests:
        logger.info("batch.nothing_to_do")
        return ""

    # Submit batch
    logger.info("batch.submitting", count=len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("batch.submitted", batch_id=batch.id, status=batch.processing_status)

    return batch.id


def poll_batch(api_key: str, batch_id: str) -> None:
    """Poll for batch completion, then download and process results."""
    client = anthropic.Anthropic(api_key=api_key)
    EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Poll
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = (
            counts.processing + counts.succeeded + counts.errored + counts.expired + counts.canceled
        )
        print(
            f"\r  [{batch.processing_status}] "
            f"{counts.succeeded} succeeded, {counts.errored} errored, "
            f"{counts.processing} processing / {total} total",
            end="",
            flush=True,
        )

        if batch.processing_status == "ended":
            print()
            break

        time.sleep(15)

    # Download results
    logger.info("batch.downloading_results", batch_id=batch_id)

    # Build paper lookup for metadata
    papers_path = OUTPUT_DIR / "papers.json"
    with open(papers_path) as f:
        papers = json.load(f)
    paper_lookup: dict[str, dict] = {}
    for p in papers:
        phash = _paper_hash(p.get("doi"), p.get("title"))
        paper_lookup[phash] = p

    successes = 0
    parse_errors = 0
    api_errors = 0
    total_claims = 0
    total_evidence = 0
    total_input_tokens = 0
    total_output_tokens = 0
    section_source_dist: dict[str, int] = {}

    for result in client.messages.batches.results(batch_id):
        phash = result.custom_id

        if result.result.type == "succeeded":
            message = result.result.message
            content = message.content[0].text if message.content else ""

            total_input_tokens += message.usage.input_tokens
            total_output_tokens += message.usage.output_tokens

            try:
                json_str = _extract_json_str(content)
                raw_dict = json.loads(json_str)
                coerced = _coerce_kg_dict(raw_dict)
                validated = KGExtraction.model_validate(coerced)

                # Save to cache
                cache_path = EXTRACTION_CACHE_DIR / f"{phash}.json"
                cache_path.write_text(
                    json.dumps(validated.model_dump(), indent=2, default=str, ensure_ascii=False)
                )

                n_claims = len(validated.claims)
                n_evidence = len(validated.evidence)
                total_claims += n_claims
                total_evidence += n_evidence
                successes += 1

                # Accumulate section_source distribution
                for claim in validated.claims:
                    ss = str(claim.section_source) if claim.section_source else "unknown"
                    section_source_dist[ss] = section_source_dist.get(ss, 0) + 1

            except Exception as e:
                parse_errors += 1
                debug_path = EXTRACTION_CACHE_DIR / f"{phash}_raw.txt"
                debug_path.write_text(content)
                paper = paper_lookup.get(phash, {})
                logger.warning(
                    "result.parse_error",
                    hash=phash,
                    title=(paper.get("title") or "")[:60],
                    error=str(e)[:150],
                )

        elif result.result.type == "errored":
            api_errors += 1
            paper = paper_lookup.get(phash, {})
            logger.warning(
                "result.api_error",
                hash=phash,
                title=(paper.get("title") or "")[:60],
                error=str(result.result.error)[:150],
            )

        elif result.result.type == "expired":
            api_errors += 1
            logger.warning("result.expired", hash=phash)

    # Cost estimate (batch 50% discount)
    input_cost = total_input_tokens * 0.40 / 1_000_000  # 50% batch discount on $0.80
    output_cost = total_output_tokens * 2.00 / 1_000_000  # 50% batch discount on $4.00
    total_cost = input_cost + output_cost

    avg_claims = total_claims / successes if successes else 0
    avg_evidence = total_evidence / successes if successes else 0

    print(f"\n{'=' * 60}")
    print("  Batch KG Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Batch ID:             {batch_id}")
    print(f"  Succeeded:            {successes}")
    print(f"  Parse errors:         {parse_errors}")
    print(f"  API errors:           {api_errors}")
    print(f"  Total claims:         {total_claims}  (avg {avg_claims:.1f}/paper)")
    print(f"  Total evidence:       {total_evidence}  (avg {avg_evidence:.1f}/paper)")
    print(f"  Input tokens:         {total_input_tokens:,}")
    print(f"  Output tokens:        {total_output_tokens:,}")
    print(f"  Est. cost (batch):    ${total_cost:.2f}")
    print(f"    Input:              ${input_cost:.2f}")
    print(f"    Output:             ${output_cost:.2f}")
    if section_source_dist:
        print("  Section source dist:")
        for ss, count in sorted(section_source_dist.items(), key=lambda x: -x[1]):
            print(f"    {ss:<25} {count}")
    print(f"{'=' * 60}")

    # Save run log
    run_log = {
        "batch_id": batch_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "method": "anthropic_message_batches_kg",
        "successes": successes,
        "parse_errors": parse_errors,
        "api_errors": api_errors,
        "total_claims": total_claims,
        "total_evidence": total_evidence,
        "avg_claims_per_paper": round(avg_claims, 1),
        "avg_evidence_per_paper": round(avg_evidence, 1),
        "section_source_distribution": section_source_dist,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 2),
    }
    log_path = OUTPUT_DIR / "batch_run_log_kg.json"
    log_path.write_text(json.dumps(run_log, indent=2, default=str))
    print(f"\n  Run log: {log_path}")


def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Batch KG Extraction via Anthropic Batches API (KGExtraction schema)"
    )
    parser.add_argument("--poll", type=str, help="Resume polling for an existing batch ID")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    if args.poll:
        poll_batch(api_key, args.poll)
    else:
        batch_id = submit_batch(api_key)
        if batch_id:
            poll_batch(api_key, batch_id)


if __name__ == "__main__":
    main()
