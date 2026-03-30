#!/usr/bin/env python3
"""Retry full text resolution for inaccessible papers.

Combines:
1. Papers from expansion_candidates.json that didn't make it into papers.json
2. Papers from papers_archived.json not in corpus or candidates
Clears negative cache entries, then re-resolves with (hopefully) S2 API key set.

Usage:
    S2_API_KEY=... python retry_inaccessible.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_AUTOREVIEW_ROOT))

import structlog

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.full_text import FullTextResolver
from autoreview.search.full_text_cache import CachedFullTextResolver, FullTextCache

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
FULL_TEXT_CACHE_DIR = OUTPUT_DIR / "full_text_cache"
PAPERS_PATH = OUTPUT_DIR / "papers.json"


def _cache_key(paper: dict) -> str:
    doi = (paper.get("doi") or "").lower().strip()
    title = (paper.get("title") or "").lower().strip()
    raw = doi if doi else title
    return hashlib.sha256(raw.encode()).hexdigest()


def clear_negative_cache(papers: list[dict]) -> int:
    """Remove negative cache entries so papers get retried."""
    cache_root = FULL_TEXT_CACHE_DIR / "full_text_cache"
    cleared = 0
    for paper in papers:
        key = _cache_key(paper)
        cache_path = cache_root / key[:2] / f"{key}.json"
        if cache_path.exists():
            try:
                entry = json.loads(cache_path.read_text())
                if entry.get("text") is None:
                    cache_path.unlink()
                    cleared += 1
            except Exception:
                pass
    return cleared


async def main() -> None:
    import os

    s2_key = os.environ.get("S2_API_KEY", "")
    if s2_key:
        print(f"  S2 API key set: {s2_key[:8]}...")
    else:
        print("  WARNING: S2_API_KEY not set — S2 lookups will be rate-limited")

    # Load current corpus
    corpus = json.loads(PAPERS_PATH.read_text())
    corpus_dois = {(p.get("doi") or "").lower().strip() for p in corpus if p.get("doi")}
    corpus_titles = {(p.get("title") or "").lower().strip()[:80] for p in corpus}
    print(f"  Current corpus: {len(corpus)} papers")

    # Collect papers to retry
    retry_papers: list[dict] = []

    # Source 1: Expansion candidates not in corpus
    candidates_path = OUTPUT_DIR / "expansion_candidates.json"
    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
        for c in candidates:
            doi = (c.get("doi") or "").lower().strip()
            title = (c.get("title") or "").lower().strip()[:80]
            if (doi and doi in corpus_dois) or (title and title in corpus_titles):
                continue
            retry_papers.append(c)
        print(f"  From expansion candidates: {len(retry_papers)} to retry")

    # Source 2: Archived papers from original corpus
    archived_path = OUTPUT_DIR / "papers_archived.json"
    if archived_path.exists():
        archived = json.loads(archived_path.read_text())
        archived_papers = archived.get("abstract_only", []) + archived.get("no_text", [])
        added = 0
        retry_dois = {(p.get("doi") or "").lower().strip() for p in retry_papers if p.get("doi")}
        for p in archived_papers:
            doi = (p.get("doi") or "").lower().strip()
            title = (p.get("title") or "").lower().strip()[:80]
            if (doi and doi in corpus_dois) or (title and title in corpus_titles):
                continue
            if doi and doi in retry_dois:
                continue
            retry_papers.append(p)
            if doi:
                retry_dois.add(doi)
            added += 1
        print(f"  From archived papers: {added} to retry")

    print(f"  Total to retry: {len(retry_papers)}")

    if not retry_papers:
        print("  Nothing to retry.")
        return

    # Clear negative cache entries
    cleared = clear_negative_cache(retry_papers)
    print(f"  Cleared {cleared} negative cache entries")

    # Convert to CandidatePaper objects
    papers = []
    for p in retry_papers:
        try:
            papers.append(CandidatePaper.model_validate(p))
        except Exception:
            papers.append(
                CandidatePaper(
                    title=p.get("title", "Unknown"),
                    authors=p.get("authors", []),
                    year=p.get("year"),
                    journal=p.get("journal"),
                    doi=p.get("doi"),
                    abstract=p.get("abstract"),
                    source_database=p.get("source_database", "retry"),
                    external_ids=p.get("external_ids", {}),
                    citation_count=p.get("citation_count"),
                )
            )

    # Resolve full text
    screened = [
        ScreenedPaper(paper=p, relevance_score=4, rationale="retry", include=True) for p in papers
    ]

    resolver = FullTextResolver()
    cache = FullTextCache(FULL_TEXT_CACHE_DIR)
    cached_resolver = CachedFullTextResolver(resolver, cache)

    try:
        source_counts = await cached_resolver.resolve(screened, max_concurrent=10)
        logger.info("retry.resolved", source_counts=source_counts)
    finally:
        await cached_resolver.close()

    # Collect successes
    new_with_text = [s.paper for s in screened if s.paper.full_text]
    still_failed = [s.paper for s in screened if not s.paper.full_text]

    print("\n  Retry results:")
    print(f"    Recovered:     {len(new_with_text)}")
    print(f"    Still failed:  {len(still_failed)}")

    if new_with_text:
        # Merge into corpus
        merged = corpus + [p.model_dump() for p in new_with_text]
        PAPERS_PATH.write_text(json.dumps(merged, indent=2, default=str, ensure_ascii=False))

        source_dist: dict[str, int] = {}
        for p in new_with_text:
            src = p.full_text_source or "unknown"
            source_dist[src] = source_dist.get(src, 0) + 1

        print(f"\n{'=' * 70}")
        print("  Retry Complete")
        print(f"{'=' * 70}")
        print(f"  Previous corpus:  {len(corpus)}")
        print(f"  Recovered:        {len(new_with_text)}")
        print(f"  New total:        {len(merged)}")
        print(f"  Still missing:    {len(still_failed)}")
        print("\n  Sources:")
        for src, count in sorted(source_dist.items(), key=lambda x: -x[1]):
            print(f"    {src:<25} {count:>4}")
        print(f"{'=' * 70}")

    # Update inaccessible report
    if still_failed:
        inacc_records = []
        for p in still_failed:
            doi = p.doi or ""
            record = {
                "doi": doi,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "authors": (p.authors or [])[:3],
                "has_abstract": bool(p.abstract),
                "citation_count": p.citation_count,
            }
            if "10.1016" in doi:
                record["publisher"] = "Elsevier"
            elif "10.1002" in doi:
                record["publisher"] = "Wiley"
            elif "10.1007" in doi or "10.1038" in doi:
                record["publisher"] = "Springer/Nature"
            elif "10.1101" in doi:
                record["publisher"] = "bioRxiv"
            elif "10.1242" in doi:
                record["publisher"] = "Company of Biologists"
            elif "10.1093" in doi:
                record["publisher"] = "Oxford"
            else:
                record["publisher"] = "Other"
            inacc_records.append(record)

        inacc_path = OUTPUT_DIR / "papers_inaccessible.json"
        inacc_path.write_text(json.dumps(inacc_records, indent=2, ensure_ascii=False))

        by_pub: dict[str, int] = {}
        for r in inacc_records:
            pub = r["publisher"]
            by_pub[pub] = by_pub.get(pub, 0) + 1
        print(f"\n  Updated inaccessible report ({len(inacc_records)} papers):")
        for pub, count in sorted(by_pub.items(), key=lambda x: -x[1]):
            print(f"    {pub:<25} {count:>4}")


if __name__ == "__main__":
    asyncio.run(main())
