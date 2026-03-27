#!/usr/bin/env python3
"""Batch KG Extraction — Anthropic Message Batches API.

Submits all full-text papers as a single batch using Haiku with prompt caching.
50% cost discount via batch processing.

Usage:
    ANTHROPIC_API_KEY=sk-... python batch_extract.py
    ANTHROPIC_API_KEY=sk-... python batch_extract.py --poll BATCH_ID   # Resume polling
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

import anthropic
import structlog
from mycelium.extraction_schema import ExtractionResult

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
EXTRACTION_CACHE_DIR = OUTPUT_DIR / "extractions"

TRUNCATION_CONFIG = SectionTruncationConfig(
    enabled=True,
    keep_sections=[
        "abstract",
        "introduction",
        "results",
        "discussion",
        "conclusion",
        "methods",
    ],
    drop_sections=[
        "references",
        "acknowledgments",
        "acknowledgements",
        "supplementary",
        "funding",
        "conflict of interest",
        "data availability",
        "author contributions",
        "competing interests",
        "ethics",
        "supporting information",
    ],
    intro_max_chars=3000,
    methods_max_chars=2000,
)


# ---------------------------------------------------------------------------
# Helpers (from kg_runner.py)
# ---------------------------------------------------------------------------


def _paper_hash(doi: str | None, title: str | None) -> str:
    raw = (doi or "").lower().strip() if doi else (title or "").lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_system_prompt() -> str:
    prompt_path = _SCRIPT_DIR.parent / "mycelium_extraction_prompt.md"
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


# --- Enum coercion maps (from kg_runner.py) ---
_EVIDENCE_STRENGTH_MAP: dict[str, str] = {
    "observational": "observational_uncontrolled",
    "review_citation": "expert_opinion",
    "computational": "computational_prediction",
    "comparative": "observational_controlled",
    "indirect_experimental": "observational_controlled",
    "experimental": "direct_experimental",
}
_VALID_EVIDENCE_STRENGTHS = {
    "systematic_review_meta_analysis",
    "randomized_controlled_trial",
    "direct_experimental",
    "observational_controlled",
    "observational_uncontrolled",
    "computational_prediction",
    "case_report",
    "expert_opinion",
}
_EVIDENCE_DIRECTION_MAP: dict[str, str] = {
    "positive": "supports",
    "negative": "contradicts",
    "refutes": "contradicts",
    "confirms": "supports",
}
_VALID_EVIDENCE_DIRECTIONS = {"supports", "contradicts", "neutral", "ambiguous"}
_ENTITY_TYPE_MAP: dict[str, str] = {
    "in_vitro_model": "method",
    "biological_structure": "other",
    "phenotype": "biological_process",
    "tissue": "other",
    "signaling_molecule": "protein",
    "receptor": "protein",
    "enzyme": "protein",
    "transcription_factor": "protein",
    "morphogen": "chemical",
    "growth_factor": "protein",
    "structure": "other",
    "cellular_component": "other",
    "developmental_process": "biological_process",
    "signaling_pathway": "pathway",
}
_VALID_ENTITY_TYPES = {
    "gene",
    "protein",
    "chemical",
    "disease",
    "pathway",
    "biological_process",
    "cell_type",
    "organism",
    "method",
    "other",
}
_CAUSALITY_HEDGE_MAP: dict[str, str] = {
    "conditional": "unclear",
    "suggestive": "correlational",
    "probable": "causal",
    "unknown": "unclear",
}
_VALID_CAUSALITY_HEDGES = {"causal", "correlational", "unclear"}


def _coerce_extraction_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Pre-process LLM JSON to fix common validation issues before Pydantic."""
    prov = d.get("paper_provenance")
    if isinstance(prov, dict):
        for key in ("doi", "title", "journal", "publication_date"):
            if prov.get(key) is None:
                prov[key] = ""
        for key in ("authors", "funding_sources"):
            if prov.get(key) is None:
                prov[key] = []

    meta = d.get("extraction_metadata")
    if not isinstance(meta, dict):
        d["extraction_metadata"] = {
            "extraction_model": MODEL,
            "extraction_version": "1.0.0",
            "extraction_timestamp": datetime.now(UTC).isoformat(),
        }
    else:
        for key, default in [
            ("extraction_model", MODEL),
            ("extraction_version", "1.0.0"),
            ("extraction_timestamp", datetime.now(UTC).isoformat()),
        ]:
            if not meta.get(key):
                meta[key] = default

    for eu in d.get("evidence_units", []) or []:
        if not isinstance(eu, dict):
            continue
        strength = eu.get("evidence_strength", "")
        if strength not in _VALID_EVIDENCE_STRENGTHS:
            eu["evidence_strength"] = _EVIDENCE_STRENGTH_MAP.get(strength, "direct_experimental")
        direction = eu.get("evidence_direction", "")
        if direction not in _VALID_EVIDENCE_DIRECTIONS:
            eu["evidence_direction"] = _EVIDENCE_DIRECTION_MAP.get(direction, "supports")
        for key in ("assertion_draft_ids", "limitations_stated_by_authors"):
            if eu.get(key) is None:
                eu[key] = []
        exp = eu.get("experiment")
        if isinstance(exp, dict):
            if exp.get("description") is None:
                exp["description"] = ""
        results = eu.get("results")
        if isinstance(results, dict):
            if results.get("effect_description") is None:
                results["effect_description"] = ""
        tags = eu.get("methodological_tags")
        if isinstance(tags, dict):
            if tags.get("approach_category") is None:
                tags["approach_category"] = ""
            if tags.get("assay_types") is None:
                tags["assay_types"] = []
        if eu.get("source_section") is None:
            eu["source_section"] = ""

    for ad in d.get("assertion_drafts", []) or []:
        if not isinstance(ad, dict):
            continue
        for key in (
            "draft_id",
            "natural_language",
            "canonical_form",
            "negatable_form",
            "predicate",
        ):
            if ad.get(key) is None:
                ad[key] = ""
        for key in ("evidence_unit_ids", "parent_assertion_ids"):
            if ad.get(key) is None:
                ad[key] = []
        conditions = ad.get("conditions")
        if conditions is None:
            ad["conditions"] = []
        elif isinstance(conditions, list):
            ad["conditions"] = [
                {"condition_type": "biological_context", "value": c} if isinstance(c, str) else c
                for c in conditions
            ]
        for entity_key in ("subject_entity", "object_entity"):
            entity = ad.get(entity_key)
            if isinstance(entity, dict):
                etype = entity.get("entity_type", "")
                if etype not in _VALID_ENTITY_TYPES:
                    entity["entity_type"] = _ENTITY_TYPE_MAP.get(etype, "other")
                if entity.get("surface_form") is None:
                    entity["surface_form"] = ""
                if entity.get("aliases") is None:
                    entity["aliases"] = []
        hedging = ad.get("hedging")
        if isinstance(hedging, dict):
            ch = hedging.get("causality_hedge", "")
            if ch not in _VALID_CAUSALITY_HEDGES:
                hedging["causality_hedge"] = _CAUSALITY_HEDGE_MAP.get(ch, "unclear")
        scope = ad.get("scope")
        if isinstance(scope, dict):
            for key in ("species", "tissue", "cell_type", "disease"):
                if scope.get(key) is None:
                    scope[key] = []
        prov = ad.get("provenance")
        if isinstance(prov, dict):
            if prov.get("source_sentence") is None:
                prov["source_sentence"] = ""
            if prov.get("section_name") is None:
                prov["section_name"] = ""
        elif prov is None:
            ad["provenance"] = {
                "source_sentence": ad.get("natural_language", ""),
                "section_name": "",
            }

    for cc in d.get("citation_contexts", []) or []:
        if not isinstance(cc, dict):
            continue
        if cc.get("citation_id") is None:
            cc["citation_id"] = ""
        if cc.get("citing_sentence") is None:
            cc["citing_sentence"] = ""
        if cc.get("cited_claim_paraphrase") is None:
            cc["cited_claim_paraphrase"] = ""
        if cc.get("linked_assertion_draft_ids") is None:
            cc["linked_assertion_draft_ids"] = []

    for key in ("evidence_units", "assertion_drafts", "citation_contexts"):
        if d.get(key) is None:
            d[key] = []

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
                ExtractionResult.model_validate(cached)
                skipped_cached += 1
                continue
            except Exception:
                pass  # Re-extract on cache corruption

        # Get text
        text = paper.get("full_text", "")
        if not text:
            skipped_no_text += 1
            continue

        # Truncate
        if len(text) > 5000:
            text = section_aware_truncate(text, 80_000, TRUNCATION_CONFIG)

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
    total_assertions = 0
    total_evidence = 0
    total_citations = 0
    total_input_tokens = 0
    total_output_tokens = 0

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
                coerced = _coerce_extraction_dict(raw_dict)
                validated = ExtractionResult.model_validate(coerced)

                # Save to cache
                cache_path = EXTRACTION_CACHE_DIR / f"{phash}.json"
                cache_path.write_text(
                    json.dumps(validated.model_dump(), indent=2, default=str, ensure_ascii=False)
                )

                n_assertions = len(validated.assertion_drafts)
                n_evidence = len(validated.evidence_units)
                n_citations = len(validated.citation_contexts)
                total_assertions += n_assertions
                total_evidence += n_evidence
                total_citations += n_citations
                successes += 1

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

    # Cost estimate
    input_cost = total_input_tokens * 0.40 / 1_000_000  # 50% batch discount on $0.80
    output_cost = total_output_tokens * 2.00 / 1_000_000  # 50% batch discount on $4.00
    total_cost = input_cost + output_cost

    print(f"\n{'=' * 60}")
    print("  Batch Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Batch ID:           {batch_id}")
    print(f"  Succeeded:          {successes}")
    print(f"  Parse errors:       {parse_errors}")
    print(f"  API errors:         {api_errors}")
    print(f"  Total assertions:   {total_assertions}")
    print(f"  Total evidence:     {total_evidence}")
    print(f"  Total citations:    {total_citations}")
    print(f"  Input tokens:       {total_input_tokens:,}")
    print(f"  Output tokens:      {total_output_tokens:,}")
    print(f"  Est. cost (batch):  ${total_cost:.2f}")
    print(f"    Input:            ${input_cost:.2f}")
    print(f"    Output:           ${output_cost:.2f}")
    print(f"{'=' * 60}")

    # Save run log
    run_log = {
        "batch_id": batch_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "method": "anthropic_message_batches",
        "successes": successes,
        "parse_errors": parse_errors,
        "api_errors": api_errors,
        "total_assertions": total_assertions,
        "total_evidence_units": total_evidence,
        "total_citation_contexts": total_citations,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 2),
    }
    log_path = OUTPUT_DIR / "batch_run_log.json"
    log_path.write_text(json.dumps(run_log, indent=2, default=str))
    print(f"\n  Run log: {log_path}")


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Batch KG Extraction via Anthropic Batches API")
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
