#!/usr/bin/env python3
"""Gastruloid Corpus Retrieval Script.

Retrieves the full gastruloid scientific literature corpus using the
AutoReview search framework across all 6 search sources.

Usage:
    python retrieve_corpus.py                    # Full run: search + full text
    python retrieve_corpus.py --search-only      # Search only, skip full text
    python retrieve_corpus.py --fulltext-only    # Skip search, resolve full text for existing papers.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — make autoreview importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]  # AutoReview/

sys.path.insert(0, str(_AUTOREVIEW_ROOT))

import structlog

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.aggregator import SearchAggregator
from autoreview.search.core_api import CORESearch
from autoreview.search.crossref import CrossRefSearch
from autoreview.search.europe_pmc import EuropePMCSearch
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

TOPIC = "gastruloid scientific literature corpus"

OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
FULL_TEXT_CACHE_DIR = OUTPUT_DIR / "full_text_cache"

# ---------------------------------------------------------------------------
# Search Queries — comprehensive coverage of the gastruloid field
# ---------------------------------------------------------------------------

SEARCH_QUERIES: dict[str, list[str]] = {
    "pubmed": [
        # Core term — most important
        "gastruloid[Title/Abstract]",
        # In vitro gastrulation models
        "in vitro gastrulation model[Title/Abstract] AND stem cell[Title/Abstract]",
        # Symmetry breaking in embryo models
        "symmetry breaking[Title/Abstract] AND (embryo model[Title/Abstract] OR organoid[Title/Abstract])",
        # Axial elongation and anteroposterior patterning
        "axial elongation[Title/Abstract] AND organoid[Title/Abstract]",
        "anteroposterior patterning[Title/Abstract] AND stem cell aggregate[Title/Abstract]",
        # Somitoid and segmentation clock
        "somitoid[Title/Abstract]",
        "segmentation clock[Title/Abstract] AND organoid[Title/Abstract]",
        # Trunk-like structures
        "trunk-like structure[Title/Abstract] AND stem cell[Title/Abstract]",
        # BMP/Wnt/Nodal signaling in gastruloids
        "gastruloid[Title/Abstract] AND (BMP[Title/Abstract] OR Wnt[Title/Abstract] OR Nodal[Title/Abstract])",
        # Germ layer specification in aggregates
        "germ layer specification[Title/Abstract] AND stem cell aggregate[Title/Abstract]",
        # Embryoid body elongation
        "embryoid body[Title/Abstract] AND elongation[Title/Abstract]",
        # Post-implantation embryo models
        "post-implantation embryo model[Title/Abstract]",
        "post-implantation-like[Title/Abstract]",
        # Self-organization in embryo-like structures
        "self-organization[Title/Abstract] AND embryo-like[Title/Abstract]",
        "pluripotent stem cell[Title/Abstract] AND gastrulation[Title/Abstract] AND aggregate[Title/Abstract]",
        # Stem cell-derived embryo models
        "stem cell-derived embryo model[Title/Abstract]",
        "embryo model self-organization[Title/Abstract] AND pluripotent[Title/Abstract]",
    ],
    "semantic_scholar": [
        # Core term
        "gastruloid",
        "gastruloid organoid development self-organization",
        "gastruloid symmetry breaking embryo model",
        "stem cell gastruloid morphogenesis germ layer",
        "gastruloid Wnt BMP signaling axial patterning",
        # In vitro gastrulation
        "in vitro gastrulation stem cell aggregate model",
        "post-implantation embryo-like model pluripotent",
        # Somitoid and segmentation
        "somitoid segmentation clock organoid stem cell",
        # Trunk-like structures
        "trunk-like structure stem cell self-organization",
        # Germ layer
        "germ layer specification embryoid body pluripotent aggregate",
        # Elongation
        "embryoid body elongation axial patterning",
        # Symmetry breaking
        "symmetry breaking pluripotent stem cell aggregate embryo",
        # Self-organization
        "embryo-like structure self-organization pluripotent stem cell",
    ],
    "openalex": [
        # Core term
        "gastruloid",
        "gastruloid organoid development differentiation",
        "gastruloid embryo model self-organization",
        # In vitro gastrulation
        "in vitro gastrulation stem cell aggregates",
        "post-implantation embryo model pluripotent stem cell",
        # Symmetry breaking
        "symmetry breaking embryo organoid stem cell aggregate",
        # Axial elongation
        "axial elongation organoid stem cell",
        "anteroposterior patterning stem cell aggregate",
        # Somitoid
        "somitoid segmentation clock organoid",
        # Trunk-like
        "trunk-like structure pluripotent stem cell",
        # Germ layer
        "germ layer specification stem cell aggregate embryoid body",
        # Self-organization
        "self-organization embryo-like structure pluripotent",
    ],
    "core": [
        # Core term
        "gastruloid",
        "gastruloid organoid development",
        "in vitro gastrulation model stem cell",
        "symmetry breaking embryo model organoid",
        "axial elongation organoid stem cell",
        "somitoid segmentation clock stem cell",
        "trunk-like structure pluripotent stem cell",
        "post-implantation embryo model",
        "embryo-like structure self-organization pluripotent",
        "germ layer specification stem cell aggregate",
    ],
    "crossref": [
        # Core term
        "gastruloid",
        "gastruloid organoid development morphogenesis",
        "in vitro gastrulation stem cell aggregate",
        "symmetry breaking embryo model pluripotent",
        "axial elongation organoid patterning",
        "somitoid segmentation clock organoid",
        "trunk-like structure stem cell self-organization",
        "post-implantation embryo-like model",
        "germ layer specification embryoid body elongation",
        "pluripotent stem cell gastrulation aggregate self-organization",
    ],
    "europe_pmc": [
        # Core term
        "gastruloid",
        "gastruloid organoid development",
        "in vitro gastrulation model stem cell",
        "symmetry breaking embryo model organoid",
        "axial elongation organoid stem cell patterning",
        "somitoid segmentation clock organoid stem cell",
        "trunk-like structure pluripotent stem cell self-organization",
        "post-implantation embryo model pluripotent",
        "embryo-like structure self-organization",
        "germ layer specification stem cell aggregate elongation",
        "gastruloid Wnt BMP Nodal signaling",
    ],
}

# ---------------------------------------------------------------------------
# Relevance filter — broadened to capture the full field
# ---------------------------------------------------------------------------

_RELEVANCE_KEYWORDS = re.compile(
    r"gastruloid|gastrulation.model|in.vitro.gastrulation|"
    r"embryoid.body.elongat|axial.elongation.organoid|"
    r"symmetry.breaking.*(embryo|aggregate|organoid|stem.cell)|"
    r"trunk.like.structure|somitoid|segmentation.clock.organoid|"
    r"post.implantation.*model|embryo.like.structure|"
    r"stem.cell.*self.organiz.*gastrulat|"
    r"pluripotent.*gastrulat",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Review filter — exclude non-primary research
# ---------------------------------------------------------------------------

_REVIEW_KEYWORDS = re.compile(
    r"\breview\b|\bmeta.analysis\b|\bsystematic review\b|\bperspective\b|"
    r"\beditorial\b|\bcommentary\b|\bopinion\b|\bmini.review\b|"
    r"\boverview\b|\bstate of the art\b",
    re.IGNORECASE,
)

_REVIEW_JOURNAL_RE = re.compile(r"\breview\b", re.IGNORECASE)


def _is_gastruloid_relevant(paper: CandidatePaper) -> bool:
    """Check if a paper is specifically about gastruloid / in vitro gastrulation."""
    text = (paper.title or "") + " " + (paper.abstract or "")
    return bool(_RELEVANCE_KEYWORDS.search(text))


def _is_likely_review(paper: CandidatePaper) -> bool:
    """Heuristic: flag papers whose title suggests they are reviews.

    Only checks the title — abstracts of original research may discuss
    "review of literature" in their introductions.
    Also checks if the journal name indicates a review journal.
    """
    title = paper.title or ""
    if _REVIEW_KEYWORDS.search(title):
        return True
    # Check journal field for review journals (e.g. "Nature Reviews")
    journal = paper.journal or ""
    if _REVIEW_JOURNAL_RE.search(journal):
        return True
    return False


# ---------------------------------------------------------------------------
# Phase 1: Search
# ---------------------------------------------------------------------------


async def search_papers() -> tuple[list[CandidatePaper], dict[str, Any]]:
    """Search for gastruloid papers across all 6 search sources.

    Returns:
        Tuple of (selected_papers, filter_stats_dict)
    """
    logger.info("search.start", topic=TOPIC, sources=6)

    aggregator = SearchAggregator(date_range="2013-2026")
    aggregator.add_source(PubMedSearch())
    aggregator.add_source(SemanticScholarSearch())
    aggregator.add_source(OpenAlexSearch())
    aggregator.add_source(CORESearch())
    aggregator.add_source(CrossRefSearch())
    aggregator.add_source(EuropePMCSearch())

    papers = await aggregator.search(SEARCH_QUERIES, max_results_per_source=500)
    total_raw = len(papers)
    logger.info("search.raw_results", count=total_raw)

    # Filter: reviews first
    reviews = [p for p in papers if _is_likely_review(p)]
    non_reviews = [p for p in papers if not _is_likely_review(p)]
    reviews_filtered = len(reviews)
    logger.info(
        "search.review_filter", reviews_removed=reviews_filtered, remaining=len(non_reviews)
    )

    # Filter: relevance
    relevant = [p for p in non_reviews if _is_gastruloid_relevant(p)]
    off_topic = len(non_reviews) - len(relevant)
    logger.info("search.relevance_filter", relevant=len(relevant), off_topic=off_topic)

    # Sort by citation count descending
    relevant.sort(key=lambda p: p.citation_count or 0, reverse=True)

    logger.info(
        "search.complete",
        total_raw=total_raw,
        reviews_filtered=reviews_filtered,
        off_topic_filtered=off_topic,
        final_corpus=len(relevant),
        top_cited=relevant[0].citation_count if relevant else 0,
    )

    filter_stats = {
        "total_raw": total_raw,
        "reviews_filtered": reviews_filtered,
        "off_topic_filtered": off_topic,
        "final_corpus": len(relevant),
    }
    return relevant, filter_stats


# ---------------------------------------------------------------------------
# Phase 2: Full text retrieval
# ---------------------------------------------------------------------------


async def get_full_texts(papers: list[CandidatePaper], cache_dir: Path) -> list[ScreenedPaper]:
    """Wrap papers as ScreenedPaper and resolve full text with caching."""
    screened = [
        ScreenedPaper(
            paper=p,
            relevance_score=4,
            rationale="gastruloid corpus",
            include=True,
        )
        for p in papers
    ]

    resolver = FullTextResolver()
    cache = FullTextCache(cache_dir)
    cached_resolver = CachedFullTextResolver(resolver, cache)

    try:
        source_counts = await cached_resolver.resolve(screened, max_concurrent=5)
        logger.info("full_text.resolved", source_counts=source_counts)
    finally:
        await cached_resolver.close()

    return screened


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------


def _compute_stats(
    papers: list[CandidatePaper],
    filter_stats: dict[str, Any],
    screened: list[ScreenedPaper] | None = None,
) -> dict[str, Any]:
    """Compute corpus statistics for corpus_stats.json."""
    with_full = (
        sum(1 for p in papers if p.full_text)
        if screened is None
        else sum(1 for s in screened if s.paper.full_text)
    )
    with_abstract = (
        sum(1 for p in papers if not p.full_text and p.abstract)
        if screened is None
        else sum(1 for s in screened if not s.paper.full_text and s.paper.abstract)
    )
    no_text = len(papers) - with_full - with_abstract

    # Year distribution
    years = [p.year for p in papers if p.year is not None]
    year_dist: dict[str, Any] = {}
    if years:
        year_dist = {
            "min": min(years),
            "max": max(years),
            "median": median(years),
            "distribution": {},
        }
        for y in sorted(set(years)):
            year_dist["distribution"][str(y)] = years.count(y)

    # Source database distribution
    source_dist: dict[str, int] = {}
    for p in papers:
        db = p.source_database or "unknown"
        source_dist[db] = source_dist.get(db, 0) + 1

    # Top 10 most cited
    top_10 = sorted(papers, key=lambda p: p.citation_count or 0, reverse=True)[:10]
    top_10_list = [
        {
            "title": p.title,
            "doi": p.doi,
            "year": p.year,
            "journal": p.journal,
            "citation_count": p.citation_count,
        }
        for p in top_10
    ]

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "topic": TOPIC,
        "filter_stats": filter_stats,
        "corpus_size": len(papers),
        "full_text_coverage": {
            "count": with_full,
            "percent": round(100 * with_full / len(papers), 1) if papers else 0,
        },
        "abstract_only": {
            "count": with_abstract,
            "percent": round(100 * with_abstract / len(papers), 1) if papers else 0,
        },
        "no_text": {
            "count": no_text,
            "percent": round(100 * no_text / len(papers), 1) if papers else 0,
        },
        "year_distribution": year_dist,
        "source_distribution": source_dist,
        "top_10_most_cited": top_10_list,
    }


def _print_stats_summary(stats: dict[str, Any]) -> None:
    """Print a human-readable summary of corpus statistics."""
    fs = stats["filter_stats"]
    ft = stats["full_text_coverage"]
    ab = stats["abstract_only"]
    nt = stats["no_text"]
    yd = stats.get("year_distribution", {})

    print(f"\n{'=' * 70}")
    print("  Gastruloid Corpus Retrieval Summary")
    print(f"{'=' * 70}")
    print(f"  Search results (pre-filter):   {fs['total_raw']:>6}")
    print(f"  Reviews filtered out:          {fs['reviews_filtered']:>6}")
    print(f"  Off-topic filtered out:        {fs['off_topic_filtered']:>6}")
    print("  ─────────────────────────────────────")
    print(f"  Final corpus size:             {stats['corpus_size']:>6}")
    print()
    print(f"  Full text coverage:            {ft['count']:>6}  ({ft['percent']:.1f}%)")
    print(f"  Abstract only:                 {ab['count']:>6}  ({ab['percent']:.1f}%)")
    print(f"  No text:                       {nt['count']:>6}  ({nt['percent']:.1f}%)")
    if yd:
        print()
        print(
            f"  Year range:   {yd.get('min', '?')} – {yd.get('max', '?')}  (median: {yd.get('median', '?')})"
        )
    print()
    print("  Source database distribution:")
    for src, count in sorted(stats["source_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {src:<25} {count:>5}")
    print()
    print("  Top 10 most-cited papers:")
    for i, p in enumerate(stats["top_10_most_cited"], 1):
        title_snip = (p["title"] or "")[:55]
        citations = p.get("citation_count") or 0
        year = p.get("year") or "?"
        print(f"    {i:>2}. [{year}] ({citations:>5} cit.) {title_snip}...")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_paper_list(screened: list[ScreenedPaper], path: Path) -> None:
    """Write human-readable paper list with text availability indicator."""
    lines = []
    for s in screened:
        p = s.paper
        if p.full_text:
            status = "full"
        elif p.abstract:
            status = "abstract"
        else:
            status = "none"
        doi_str = (p.doi or "no-doi")[:40]
        title_snip = (p.title or "")[:60]
        lines.append(f"[{status:8s}] {doi_str:<40} {title_snip}")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("output.paper_list_written", path=str(path), count=len(lines))


def _write_papers_json(papers: list[CandidatePaper], path: Path) -> None:
    """Write full paper list as JSON."""
    path.write_text(
        json.dumps(
            [p.model_dump() for p in papers],
            default=str,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("output.papers_json_written", path=str(path), count=len(papers))


def _write_corpus_stats(stats: dict[str, Any], path: Path) -> None:
    """Write corpus statistics as JSON."""
    path.write_text(
        json.dumps(stats, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("output.corpus_stats_written", path=str(path))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def main(*, search_only: bool = False, fulltext_only: bool = False) -> None:
    """Run the corpus retrieval pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    papers_cache = OUTPUT_DIR / "papers.json"

    filter_stats: dict[str, Any] = {
        "total_raw": 0,
        "reviews_filtered": 0,
        "off_topic_filtered": 0,
        "final_corpus": 0,
    }

    # ---- Phase 1: Search ----
    if fulltext_only and papers_cache.exists():
        logger.info("search.skipped", reason="fulltext_only", loading_from=str(papers_cache))
        raw = json.loads(papers_cache.read_text(encoding="utf-8"))
        papers = [CandidatePaper.model_validate(p) for p in raw]
        filter_stats["final_corpus"] = len(papers)
        filter_stats["total_raw"] = len(papers)
        logger.info("search.loaded_from_cache", count=len(papers))
    else:
        logger.info("phase.search_start")
        papers, filter_stats = await search_papers()
        _write_papers_json(papers, papers_cache)
        logger.info(
            "phase.search_complete",
            corpus_size=len(papers),
            papers_json=str(papers_cache),
        )

    if search_only:
        logger.info("fulltext.skipped", reason="search_only")
        # Write paper list without full text info
        screened_stub = [
            ScreenedPaper(paper=p, relevance_score=4, rationale="gastruloid corpus", include=True)
            for p in papers
        ]
        _write_paper_list(screened_stub, OUTPUT_DIR / "paper_list.txt")
        stats = _compute_stats(papers, filter_stats)
        _write_corpus_stats(stats, OUTPUT_DIR / "corpus_stats.json")
        _print_stats_summary(stats)
        return

    # ---- Phase 2: Full text retrieval ----
    logger.info("phase.fulltext_start", paper_count=len(papers))
    screened = await get_full_texts(papers, FULL_TEXT_CACHE_DIR)

    # Update paper objects with full text data
    resolved_papers = [s.paper for s in screened]

    # Re-save papers.json with full text populated
    _write_papers_json(resolved_papers, papers_cache)

    # ---- Outputs ----
    _write_paper_list(screened, OUTPUT_DIR / "paper_list.txt")

    stats = _compute_stats(resolved_papers, filter_stats, screened)
    _write_corpus_stats(stats, OUTPUT_DIR / "corpus_stats.json")

    _print_stats_summary(stats)

    logger.info(
        "phase.complete",
        corpus_size=len(papers),
        papers_json=str(papers_cache),
        paper_list=str(OUTPUT_DIR / "paper_list.txt"),
        corpus_stats=str(OUTPUT_DIR / "corpus_stats.json"),
        full_text_cache=str(FULL_TEXT_CACHE_DIR),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gastruloid Corpus Retrieval — search + full text across 6 sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--search-only",
        action="store_true",
        help="Search only; skip full text retrieval",
    )
    mode.add_argument(
        "--fulltext-only",
        action="store_true",
        help="Skip search; resolve full text for existing papers.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(search_only=args.search_only, fulltext_only=args.fulltext_only))
