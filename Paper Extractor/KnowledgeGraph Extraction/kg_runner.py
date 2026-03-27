#!/usr/bin/env python3
"""KG Extraction Runner — Gastruloid Organoid Development.

Extracts structured knowledge graph data from ~100 papers using Haiku
via ``claude -p`` (no API key required, uses Claude Code auth).

Token-saving strategies:
  - Haiku model (cheapest)
  - System prompt held constant across calls (enables prompt caching)
  - Section-aware truncation (drop references, acknowledgments, etc.)
  - Disk-cached extraction results (resume without re-extracting)
  - Full-text disk cache (don't re-download papers)
  - Abstract-only fallback for papers without full text

Usage:
    python kg_runner.py                # Full run (search → extract)
    python kg_runner.py --extract-only # Skip search, use cached papers.json
    python kg_runner.py --dry-run      # Search + full text only, no extraction
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — make autoreview and mycelium importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]  # AutoReview/
_AUTONOMOUS_SCI_SRC = _AUTOREVIEW_ROOT.parent / "Autonomous Science" / "src"

sys.path.insert(0, str(_AUTOREVIEW_ROOT))
sys.path.insert(0, str(_AUTONOMOUS_SCI_SRC))

import structlog
from mycelium.extraction_schema import ExtractionResult

from autoreview.config.models import SectionTruncationConfig
from autoreview.extraction.truncation import section_aware_truncate
from autoreview.llm.claude_code import ClaudeCodeProvider
from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.aggregator import SearchAggregator
from autoreview.search.full_text import FullTextResolver
from autoreview.search.full_text_cache import CachedFullTextResolver, FullTextCache
from autoreview.search.openalex import OpenAlexSearch
from autoreview.search.pubmed import PubMedSearch
from autoreview.search.semantic_scholar import SemanticScholarSearch

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

TOPIC = "gastruloid organoid development"
MAX_PAPERS = 100
MODEL = "haiku"
CONCURRENCY = 1  # Sequential — minimizes token flux per hour
MAX_OUTPUT_TOKENS = 16384  # KG extractions can be large

OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
EXTRACTION_CACHE_DIR = OUTPUT_DIR / "extractions"
FULL_TEXT_CACHE_DIR = OUTPUT_DIR / "full_text_cache"

# Search queries — broad enough to capture the field, specific enough
# to stay on-topic. Multiple queries per source for better recall.
SEARCH_QUERIES: dict[str, list[str]] = {
    "pubmed": [
        "gastruloid[Title/Abstract]",
        "gastruloid AND organoid AND development",
        "gastruloid AND symmetry breaking",
        "gastruloid AND germ layer AND differentiation",
        "in vitro gastrulation model AND stem cell",
        "gastruloid AND Wnt AND BMP",
        "gastruloid AND elongation AND axial",
    ],
    "semantic_scholar": [
        "gastruloid organoid development self-organization",
        "gastruloid symmetry breaking embryo model",
        "stem cell gastruloid morphogenesis germ layer",
        "gastruloid Wnt signaling axial patterning",
    ],
    "openalex": [
        "gastruloid organoid development differentiation",
        "gastruloid embryo model self-organization",
        "in vitro gastrulation stem cell aggregates",
    ],
}

# Section truncation — aggressive for token savings while keeping
# the content KG extraction needs most.
TRUNCATION_CONFIG = SectionTruncationConfig(
    enabled=True,
    keep_sections=[
        "abstract",
        "introduction",
        "results",
        "discussion",
        "conclusion",
        "methods",  # Keep methods — valuable for evidence unit extraction
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
# Helpers
# ---------------------------------------------------------------------------


def _paper_hash(paper: CandidatePaper) -> str:
    """Stable short hash for a paper (for filenames)."""
    raw = paper.doi.lower().strip() if paper.doi else paper.title.lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_extraction_prompt() -> str:
    """Load the mycelium extraction prompt, split into system + user template."""
    prompt_path = _SCRIPT_DIR.parent / "mycelium_extraction_prompt.md"
    text = prompt_path.read_text()
    # Split at the {PAPER_TEXT} placeholder — everything before it is the system prompt
    marker = "{PAPER_TEXT}"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip()


def _safe_filename(title: str) -> str:
    """Convert paper title to a safe filename snippet."""
    clean = re.sub(r"[^\w\s-]", "", title[:60]).strip()
    return re.sub(r"\s+", "_", clean)


_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json_str(raw: str) -> str:
    """Extract a JSON object string from possibly-fenced LLM output."""
    raw = raw.strip()
    # Try raw parse first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # Try markdown fences
    fence_match = _MARKDOWN_FENCE_RE.search(raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Try finding first { ... last }
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
# Enum coercion maps — LLM produces invalid values; map to closest valid one
# ---------------------------------------------------------------------------

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
    # --- PaperProvenance: coerce nulls on required string fields ---
    prov = d.get("paper_provenance")
    if isinstance(prov, dict):
        for key in ("doi", "title", "journal", "publication_date"):
            if prov.get(key) is None:
                prov[key] = ""
        # Coerce null lists to empty lists
        for key in ("authors", "funding_sources"):
            if prov.get(key) is None:
                prov[key] = []

    # --- ExtractionMetadata: fill defaults if missing ---
    meta = d.get("extraction_metadata")
    if not isinstance(meta, dict):
        d["extraction_metadata"] = {
            "extraction_model": "claude-haiku-4-5-20251001",
            "extraction_version": "1.0.0",
            "extraction_timestamp": datetime.now(UTC).isoformat(),
        }
    else:
        for key, default in [
            ("extraction_model", "claude-haiku-4-5-20251001"),
            ("extraction_version", "1.0.0"),
            ("extraction_timestamp", datetime.now(UTC).isoformat()),
        ]:
            if not meta.get(key):
                meta[key] = default

    # --- EvidenceUnits: fix enum values and nulls ---
    for eu in d.get("evidence_units", []) or []:
        if not isinstance(eu, dict):
            continue
        # evidence_strength enum
        strength = eu.get("evidence_strength", "")
        if strength not in _VALID_EVIDENCE_STRENGTHS:
            eu["evidence_strength"] = _EVIDENCE_STRENGTH_MAP.get(strength, "direct_experimental")
        # evidence_direction enum
        direction = eu.get("evidence_direction", "")
        if direction not in _VALID_EVIDENCE_DIRECTIONS:
            eu["evidence_direction"] = _EVIDENCE_DIRECTION_MAP.get(direction, "supports")
        # Coerce null lists
        for key in ("assertion_draft_ids", "limitations_stated_by_authors"):
            if eu.get(key) is None:
                eu[key] = []
        # Coerce nested objects
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

    # --- AssertionDrafts: coerce nulls, enums, structural ---
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
        # conditions: coerce plain strings to Condition objects
        conditions = ad.get("conditions")
        if conditions is None:
            ad["conditions"] = []
        elif isinstance(conditions, list):
            ad["conditions"] = [
                {"condition_type": "biological_context", "value": c} if isinstance(c, str) else c
                for c in conditions
            ]
        # entity_type enum coercion (subject and object)
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
        # hedging: causality_hedge enum
        hedging = ad.get("hedging")
        if isinstance(hedging, dict):
            ch = hedging.get("causality_hedge", "")
            if ch not in _VALID_CAUSALITY_HEDGES:
                hedging["causality_hedge"] = _CAUSALITY_HEDGE_MAP.get(ch, "unclear")
        # Fix nested scope
        scope = ad.get("scope")
        if isinstance(scope, dict):
            for key in ("species", "tissue", "cell_type", "disease"):
                if scope.get(key) is None:
                    scope[key] = []
        # Fix nested provenance — synthesize if missing
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

    # --- CitationContexts: coerce nulls ---
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

    # Coerce top-level null lists
    for key in ("evidence_units", "assertion_drafts", "citation_contexts"):
        if d.get(key) is None:
            d[key] = []

    return d


# ---------------------------------------------------------------------------
# Phase 1: Search
# ---------------------------------------------------------------------------


# Relevance keywords — paper must mention at least one in title or abstract
# to be considered on-topic. Broad enough to catch the field, narrow enough
# to exclude generic organoid / stem cell papers.
_RELEVANCE_KEYWORDS = re.compile(
    r"gastruloid|gastrulation.model|in.vitro.gastrulation|"
    r"embryoid.body.elongat|axial.elongation.organoid|"
    r"symmetry.breaking.*(embryo|aggregate|organoid)|"
    r"trunk.like.structure|somitoid|segmentation.clock.organoid",
    re.IGNORECASE,
)


def _is_gastruloid_relevant(paper: CandidatePaper) -> bool:
    """Check if a paper is specifically about gastruloid / in vitro gastrulation."""
    text = (paper.title or "") + " " + (paper.abstract or "")
    return bool(_RELEVANCE_KEYWORDS.search(text))


async def search_papers() -> list[CandidatePaper]:
    """Search for gastruloid papers across PubMed, Semantic Scholar, OpenAlex."""
    logger.info("search.start", topic=TOPIC, max_papers=MAX_PAPERS)

    aggregator = SearchAggregator(date_range="2014-2026")
    aggregator.add_source(PubMedSearch())
    aggregator.add_source(SemanticScholarSearch())
    aggregator.add_source(OpenAlexSearch())

    papers = await aggregator.search(SEARCH_QUERIES, max_results_per_source=200)
    logger.info("search.raw_results", count=len(papers))

    # Filter to gastruloid-relevant papers only
    relevant = [p for p in papers if _is_gastruloid_relevant(p)]
    off_topic = len(papers) - len(relevant)
    logger.info("search.relevance_filter", relevant=len(relevant), off_topic=off_topic)

    # Sort by citation count (descending) to prioritize impactful papers
    relevant.sort(key=lambda p: p.citation_count or 0, reverse=True)
    selected = relevant[:MAX_PAPERS]

    logger.info(
        "search.selected",
        count=len(selected),
        top_cited=selected[0].citation_count if selected else 0,
        min_cited=selected[-1].citation_count if selected else 0,
    )
    return selected


# ---------------------------------------------------------------------------
# Phase 2: Full text retrieval
# ---------------------------------------------------------------------------


async def get_full_texts(papers: list[CandidatePaper]) -> list[ScreenedPaper]:
    """Wrap papers as ScreenedPaper and resolve full text with caching."""
    screened = [
        ScreenedPaper(
            paper=p,
            relevance_score=4,
            rationale="gastruloid topic match",
            include=True,
        )
        for p in papers
    ]

    resolver = FullTextResolver()
    cache = FullTextCache(FULL_TEXT_CACHE_DIR)
    cached_resolver = CachedFullTextResolver(resolver, cache)

    try:
        source_counts = await cached_resolver.resolve(screened, max_concurrent=5)
        logger.info("full_text.resolved", source_counts=source_counts)
    finally:
        await cached_resolver.close()

    return screened


# ---------------------------------------------------------------------------
# Phase 3: Extraction
# ---------------------------------------------------------------------------


async def extract_paper(
    llm: ClaudeCodeProvider,
    system_prompt: str,
    paper: ScreenedPaper,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Extract KG data from a single paper via claude -p."""
    p = paper.paper
    phash = _paper_hash(p)
    cache_path = EXTRACTION_CACHE_DIR / f"{phash}.json"
    short_title = p.title[:80] if p.title else phash

    # --- Check extraction cache ---
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            ExtractionResult.model_validate(cached)
            return {
                "index": index,
                "hash": phash,
                "title": short_title,
                "status": "cached",
                "assertions": len(cached.get("assertion_drafts", [])),
                "evidence_units": len(cached.get("evidence_units", [])),
            }
        except Exception:
            pass  # Re-extract on cache corruption

    # --- Determine input text ---
    text = p.full_text or p.abstract or ""
    if not text:
        return {
            "index": index,
            "hash": phash,
            "title": short_title,
            "status": "skipped",
            "reason": "no_text_available",
        }

    text_type = "full_text" if p.full_text else "abstract"

    # Skip abstract-only papers — extraction prompt requires full text detail
    if text_type == "abstract":
        return {
            "index": index,
            "hash": phash,
            "title": short_title,
            "status": "skipped",
            "reason": "abstract_only",
        }

    # --- Truncate full-text papers ---
    original_len = len(text)
    if len(text) > 5000:
        text = section_aware_truncate(text, 80_000, TRUNCATION_CONFIG)
    truncated_len = len(text)

    # --- Build the user prompt (paper text only) ---
    user_prompt = text

    # --- Call claude -p ---
    async with semaphore:
        logger.info(
            "extract.start",
            index=f"{index}/{total}",
            hash=phash,
            text_type=text_type,
            chars=truncated_len,
            truncated_from=original_len if original_len != truncated_len else None,
        )
        start = time.monotonic()
        response = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await llm.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.0,
                    model_override="haiku",
                )
                break
            except Exception as e:
                last_error = e
                backoff = 30 * (2**attempt)  # 30s, 60s, 120s
                logger.warning(
                    "extract.retry",
                    hash=phash,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                    error=str(e)[:100],
                )
                await asyncio.sleep(backoff)
        if response is None:
            duration = time.monotonic() - start
            logger.error("extract.llm_error", hash=phash, error=str(last_error)[:200])
            return {
                "index": index,
                "hash": phash,
                "title": short_title,
                "status": "error",
                "error": f"llm_call: {last_error!s}"[:200],
                "duration_s": round(duration, 1),
            }
        duration = time.monotonic() - start

    # --- Parse and validate JSON ---
    try:
        json_str = _extract_json_str(response.content)
        raw_dict = json.loads(json_str)
        coerced = _coerce_extraction_dict(raw_dict)
        result = ExtractionResult.model_validate(coerced)
    except Exception as e:
        logger.error(
            "extract.parse_error",
            hash=phash,
            error=str(e)[:200],
            content_preview=response.content[:300],
        )
        # Save raw output for debugging
        debug_path = EXTRACTION_CACHE_DIR / f"{phash}_raw.txt"
        debug_path.write_text(response.content)
        return {
            "index": index,
            "hash": phash,
            "title": short_title,
            "status": "parse_error",
            "error": str(e)[:200],
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "duration_s": round(duration, 1),
        }

    # --- Cache successful extraction ---
    cache_path.write_text(
        json.dumps(result.model_dump(), indent=2, default=str, ensure_ascii=False)
    )

    stats = {
        "index": index,
        "hash": phash,
        "title": short_title,
        "doi": p.doi,
        "status": "success",
        "text_type": text_type,
        "chars_sent": truncated_len,
        "assertions": len(result.assertion_drafts),
        "evidence_units": len(result.evidence_units),
        "citations": len(result.citation_contexts),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cache_creation": response.cache_creation_input_tokens,
        "cache_read": response.cache_read_input_tokens,
        "duration_s": round(duration, 1),
    }
    logger.info("extract.success", **stats)
    return stats


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run(*, extract_only: bool = False, dry_run: bool = False) -> None:
    """Run the full KG extraction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    papers_cache = OUTPUT_DIR / "papers.json"

    # ---- Phase 1: Search ----
    if extract_only and papers_cache.exists():
        raw = json.loads(papers_cache.read_text())
        papers = [CandidatePaper.model_validate(p) for p in raw]
        print(f"Phase 1: Loaded {len(papers)} papers from cache")
    else:
        print("Phase 1: Searching for papers...")
        papers = await search_papers()
        papers_cache.write_text(
            json.dumps(
                [p.model_dump() for p in papers],
                default=str,
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"  Found {len(papers)} papers across PubMed/S2/OpenAlex")

    # ---- Phase 2: Full text retrieval ----
    print("Phase 2: Retrieving full text...")
    screened = await get_full_texts(papers)

    with_full = sum(1 for s in screened if s.paper.full_text)
    with_abstract = sum(1 for s in screened if not s.paper.full_text and s.paper.abstract)
    no_text = len(screened) - with_full - with_abstract
    print(f"  Full text: {with_full} | Abstract only: {with_abstract} | No text: {no_text}")

    if dry_run:
        print("\n--- Dry run complete (no extraction) ---")
        # Save paper list with text availability
        summary_lines = []
        for s in screened:
            status = "full" if s.paper.full_text else ("abstract" if s.paper.abstract else "none")
            summary_lines.append(
                f"  [{status:8s}] {s.paper.doi or 'no-doi':40s} {s.paper.title[:60]}"
            )
        (OUTPUT_DIR / "paper_list.txt").write_text("\n".join(summary_lines))
        print(f"  Paper list saved to {OUTPUT_DIR / 'paper_list.txt'}")
        return

    # ---- Phase 3: Extraction ----
    system_prompt = _load_extraction_prompt()
    print(f"Phase 3: Extracting from {len(screened)} papers with Haiku via claude -p")
    print(f"  System prompt: {len(system_prompt):,} chars")
    print(f"  Concurrency: {CONCURRENCY}")
    print(f"  Results cached to: {EXTRACTION_CACHE_DIR}")
    print()

    llm = ClaudeCodeProvider(model="haiku", max_tokens_generate=MAX_OUTPUT_TOKENS)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Clear stale raw debug files from previous failed runs
    for raw_file in EXTRACTION_CACHE_DIR.glob("*_raw.txt"):
        raw_file.unlink()

    # Process papers sequentially — minimizes token flux per hour
    all_results: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    successes = 0
    cached = 0
    failures = 0
    skipped = 0

    for i, sp in enumerate(screened):
        result = await extract_paper(llm, system_prompt, sp, semaphore, i + 1, len(screened))
        all_results.append(result)
        # Rate-limit: sleep between papers to avoid hitting usage caps
        if result["status"] not in ("cached", "skipped"):
            await asyncio.sleep(3)

        status = result["status"]
        if status == "cached":
            cached += 1
        elif status == "success":
            successes += 1
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
        elif status == "skipped":
            skipped += 1
        else:
            failures += 1

        done = successes + cached + failures + skipped
        print(
            f"\r  [{done:3d}/{len(screened)}] "
            f"{successes} extracted, {cached} cached, {failures} failed, {skipped} skipped | "
            f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out",
            end="",
            flush=True,
        )

    print()  # newline after progress

    # ---- Summary ----
    total_assertions = sum(
        r.get("assertions", 0) for r in all_results if r["status"] in ("success", "cached")
    )
    total_evidence = sum(
        r.get("evidence_units", 0) for r in all_results if r["status"] in ("success", "cached")
    )

    print(f"\n{'=' * 60}")
    print(f"  Run Complete: {TOPIC}")
    print(f"{'=' * 60}")
    print(f"  Papers searched:    {len(papers)}")
    print(f"  Full text:          {with_full}")
    print(f"  Abstract only:      {with_abstract}")
    print(f"  Extracted (new):    {successes}")
    print(f"  Extracted (cached): {cached}")
    print(f"  Failed:             {failures}")
    print(f"  Skipped (no text):  {skipped}")
    print(f"  Total assertions:   {total_assertions}")
    print(f"  Total evidence:     {total_evidence}")
    print(f"  Input tokens:       {total_input_tokens:,}")
    print(f"  Output tokens:      {total_output_tokens:,}")
    print(f"{'=' * 60}")

    # Save run log
    run_log = {
        "topic": TOPIC,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "system_prompt_chars": len(system_prompt),
        "papers_searched": len(papers),
        "papers_with_full_text": with_full,
        "papers_abstract_only": with_abstract,
        "extractions_new": successes,
        "extractions_cached": cached,
        "extractions_failed": failures,
        "extractions_skipped": skipped,
        "total_assertions": total_assertions,
        "total_evidence_units": total_evidence,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": sorted(all_results, key=lambda r: r.get("index", 0)),
    }
    log_path = OUTPUT_DIR / "run_log.json"
    log_path.write_text(json.dumps(run_log, indent=2, default=str, ensure_ascii=False))
    print(f"\n  Run log: {log_path}")

    # List failures for debugging
    failed = [r for r in all_results if r["status"] in ("error", "parse_error")]
    if failed:
        print(f"\n  Failed papers ({len(failed)}):")
        for r in failed:
            print(f"    {r['hash']} | {r['title'][:50]} | {r.get('error', '?')[:60]}")


def main() -> None:
    global CONCURRENCY, MAX_PAPERS  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="KG Extraction Runner — Gastruloid")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip search, use cached papers.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search + full text only, no extraction",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"Max concurrent claude -p calls (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=MAX_PAPERS,
        help=f"Max papers to process (default: {MAX_PAPERS})",
    )
    args = parser.parse_args()

    CONCURRENCY = args.concurrency
    MAX_PAPERS = args.max_papers

    asyncio.run(run(extract_only=args.extract_only, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
