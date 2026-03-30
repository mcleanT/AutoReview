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
from kg_coerce import coerce_kg_dict
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
MAX_OUTPUT_TOKENS = 64000
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
        "references",
        "conclusion",
    ],
    drop_sections=[
        "acknowledgments",
        "acknowledgements",
        "supplementary",
        "abstract",
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
    """Extract valid JSON from LLM output with 4-tier fallback including truncation repair."""
    raw = raw.strip()
    # Tier 1: direct parse
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # Tier 2: markdown fence extraction
    fence_match = _MARKDOWN_FENCE_RE.search(raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Tier 3: brace-slice
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Tier 4: truncation repair — close unclosed brackets/braces
    try:
        repaired = _repair_truncated_json(raw)
        json.loads(repaired)
        logger.info("json.repaired_truncation", original_len=len(raw), repaired_len=len(repaired))
        return repaired
    except (json.JSONDecodeError, Exception):
        pass
    msg = f"No valid JSON found in response ({len(raw)} chars)"
    raise json.JSONDecodeError(msg, raw[:200], 0)


def _repair_truncated_json(raw: str) -> str:
    """Attempt to repair JSON truncated mid-output by closing unclosed structures."""
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw[:200], 0)

    text = raw[start:]
    in_string = False
    escape_next = False
    stack: list[str] = []

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{" or ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack:
        return text  # Already balanced

    result = text
    if in_string:
        result += '"'

    # Strip trailing incomplete tokens
    result = result.rstrip()
    while result and result[-1] in (",", ":", " ", "\n", "\r", "\t"):
        result = result[:-1]

    # Close all open brackets/braces in reverse order
    for bracket in reversed(stack):
        result += "}" if bracket == "{" else "]"

    return result


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
                    "messages": [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "{"},
                    ],
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
            # Prepend the assistant prefill character that the API consumed
            if content and not content.lstrip().startswith("{"):
                content = "{" + content

            total_input_tokens += message.usage.input_tokens
            total_output_tokens += message.usage.output_tokens

            try:
                json_str = _extract_json_str(content)
                raw_dict = json.loads(json_str)
                coerced = coerce_kg_dict(raw_dict)
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


def retry_failures(api_key: str) -> None:
    """Resubmit papers that failed parsing in the last batch run."""
    raw_files = sorted(EXTRACTION_CACHE_DIR.glob("*_raw.txt"))
    failed_hashes = {f.stem.replace("_raw", "") for f in raw_files}
    if not failed_hashes:
        logger.info("retry.no_failures")
        return

    logger.info("retry.found_failures", count=len(failed_hashes))

    papers_path = OUTPUT_DIR / "papers.json"
    with open(papers_path) as f:
        papers = json.load(f)

    system_prompt = _load_system_prompt()
    client = anthropic.Anthropic(api_key=api_key)
    requests = []

    for paper in papers:
        doi = paper.get("doi")
        title = paper.get("title")
        phash = _paper_hash(doi, title)

        if phash not in failed_hashes:
            continue

        # Remove stale cache and raw file
        (EXTRACTION_CACHE_DIR / f"{phash}.json").unlink(missing_ok=True)
        (EXTRACTION_CACHE_DIR / f"{phash}_raw.txt").unlink(missing_ok=True)

        text = paper.get("full_text", "")
        if not text:
            continue

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
                    "messages": [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "{"},
                    ],
                },
            }
        )

    if not requests:
        logger.info("retry.nothing_to_submit")
        return

    logger.info("retry.submitting", count=len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("retry.submitted", batch_id=batch.id)
    poll_batch(api_key, batch.id)


def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Batch KG Extraction via Anthropic Batches API (KGExtraction schema)"
    )
    parser.add_argument("--poll", type=str, help="Resume polling for an existing batch ID")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Resubmit papers that failed parsing in the last run",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    if args.retry_failures:
        retry_failures(api_key)
    elif args.poll:
        poll_batch(api_key, args.poll)
    else:
        batch_id = submit_batch(api_key)
        if batch_id:
            poll_batch(api_key, batch_id)


if __name__ == "__main__":
    main()
