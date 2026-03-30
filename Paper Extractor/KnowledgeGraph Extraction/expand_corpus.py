#!/usr/bin/env python3
"""Expand the gastruloid corpus via OpenAlex bulk retrieval.

Pulls all papers matching gastruloid + adjacent model system terms from OpenAlex,
filters reviews using the type field, deduplicates against existing corpus,
resolves full text, and merges into papers.json.

Usage:
    python expand_corpus.py                    # Full run: retrieve + full text
    python expand_corpus.py --retrieve-only    # Just pull from OpenAlex, skip full text
    python expand_corpus.py --resolve-only     # Skip retrieval, resolve full text for cached candidates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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
CANDIDATES_PATH = OUTPUT_DIR / "expansion_candidates.json"
PAPERS_PATH = OUTPUT_DIR / "papers.json"

_OPENALEX_EMAIL = "mst36@psu.edu"

# OpenAlex types to exclude
_EXCLUDE_TYPES = {
    "review",
    "editorial",
    "letter",
    "erratum",
    "commentary",
    "retraction",
    "book-chapter",
}

# Search terms — each becomes a separate OpenAlex query
SEARCH_TERMS = [
    # Core
    "gastruloid",
    # Adjacent model systems
    "blastoid trophoblast stem cell",
    "ETiX embryo stem cell",
    "ETX embryo stem cell",
    "somitoid presomitic mesoderm",
    "neuruloid neural tube organoid pluripotent",
    # Broader gastruloid-adjacent
    "micropatterned gastrulation stem cell",
    "synthetic embryo pluripotent self-organization",
    "embryoid body elongation axial patterning",
    "assembloid gastrulation germ layer",
    "trunk-like structure stem cell organoid",
    "post-implantation embryo model pluripotent in vitro",
]

# Relevance keywords — at least one must match title+abstract for non-core terms
_RELEVANCE_TERMS = {
    "gastruloid",
    "blastoid",
    "somitoid",
    "neuruloid",
    "assembloid",
    "gastrulation",
    "germ layer",
    "axial elongation",
    "symmetry breaking",
    "trunk-like",
    "embryo-like",
    "embryo model",
    "post-implantation",
    "self-organization",
    "anteroposterior",
    "mesoderm",
    "endoderm",
    "ectoderm",
    "pluripotent",
    "embryoid body",
    "organoid",
}


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _is_relevant(title: str, abstract: str) -> bool:
    """Check if paper is relevant to gastruloid/embryo model research."""
    text = (title + " " + abstract).lower()
    # "gastruloid" in text is auto-relevant
    if "gastruloid" in text:
        return True
    # For non-gastruloid papers, require at least 2 relevance terms
    matches = sum(1 for term in _RELEVANCE_TERMS if term in text)
    return matches >= 2


def _openalex_search(query: str, per_page: int = 200) -> list[dict]:
    """Paginated OpenAlex search returning all matching works."""
    import requests

    all_works = []
    cursor = "*"
    page = 0

    while cursor:
        url = (
            f"https://api.openalex.org/works"
            f"?filter=default.search:{query}"
            f",type:article|preprint"
            f",from_publication_date:2013-01-01"
            f"&select=id,doi,display_name,publication_year,type,"
            f"authorships,primary_location,cited_by_count,"
            f"abstract_inverted_index,ids"
            f"&per_page={per_page}"
            f"&cursor={cursor}"
            f"&mailto={_OPENALEX_EMAIL}"
        )

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("openalex.page_error", query=query, page=page, error=str(e)[:100])
            break

        works = data.get("results", [])
        if not works:
            break

        all_works.extend(works)
        page += 1

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")

        if page % 5 == 0:
            logger.info("openalex.paging", query=query[:30], page=page, total_so_far=len(all_works))

        time.sleep(0.1)  # Polite rate limiting

    return all_works


def _work_to_candidate(work: dict) -> CandidatePaper | None:
    """Convert OpenAlex work dict to CandidatePaper."""
    title = work.get("display_name") or ""
    if not title:
        return None

    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)

    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    journal = source.get("display_name")

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # Build external_ids
    ids = work.get("ids", {})
    external_ids: dict[str, str] = {}
    if work.get("id"):
        external_ids["openalex"] = work["id"]
    pmid = ids.get("pmid", "")
    if pmid:
        external_ids["pmid"] = pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "")
    pmcid = ids.get("pmcid", "")
    if pmcid:
        external_ids["pmcid"] = pmcid.replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "")

    # Check for OA PDF URL
    oa = location.get("pdf_url") or ""
    if oa:
        external_ids["s2_pdf_url"] = oa

    return CandidatePaper(
        title=title,
        authors=authors,
        year=work.get("publication_year"),
        journal=journal,
        doi=doi if doi else None,
        abstract=abstract if abstract else None,
        source_database="openalex",
        external_ids=external_ids,
        citation_count=work.get("cited_by_count"),
    )


def retrieve_from_openalex() -> list[CandidatePaper]:
    """Pull all papers from OpenAlex across all search terms."""
    all_works: dict[str, dict] = {}  # doi_or_id → work dict

    for term in SEARCH_TERMS:
        logger.info("openalex.search", query=term)
        works = _openalex_search(term)
        logger.info("openalex.search_done", query=term[:30], results=len(works))

        for w in works:
            # Deduplicate by DOI or OpenAlex ID
            doi = (w.get("doi") or "").lower()
            oa_id = w.get("id", "")
            key = doi if doi else oa_id
            if key and key not in all_works:
                all_works[key] = w

        time.sleep(0.5)  # Between queries

    logger.info("openalex.total_unique", count=len(all_works))

    # Filter reviews and convert
    candidates = []
    reviews_skipped = 0
    irrelevant_skipped = 0
    type_dist: dict[str, int] = {}

    for key, w in all_works.items():
        pub_type = w.get("type", "unknown")
        type_dist[pub_type] = type_dist.get(pub_type, 0) + 1

        if pub_type in _EXCLUDE_TYPES:
            reviews_skipped += 1
            continue

        title = w.get("display_name") or ""
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))

        if not _is_relevant(title, abstract):
            irrelevant_skipped += 1
            continue

        paper = _work_to_candidate(w)
        if paper:
            candidates.append(paper)

    logger.info(
        "openalex.filtered",
        candidates=len(candidates),
        reviews_skipped=reviews_skipped,
        irrelevant_skipped=irrelevant_skipped,
        type_dist=type_dist,
    )

    return candidates


def deduplicate(
    new_papers: list[CandidatePaper],
    existing_papers: list[CandidatePaper],
) -> list[CandidatePaper]:
    """Remove papers already in the existing corpus."""
    existing_dois = set()
    existing_titles = set()

    for p in existing_papers:
        if p.doi:
            existing_dois.add(p.doi.lower().strip())
        if p.title:
            existing_titles.add(p.title.lower().strip()[:80])

    novel = []
    dupes = 0
    for p in new_papers:
        doi = (p.doi or "").lower().strip()
        title = (p.title or "").lower().strip()[:80]

        if doi and doi in existing_dois:
            dupes += 1
            continue
        if title and title in existing_titles:
            dupes += 1
            continue

        novel.append(p)
        # Add to sets so we don't duplicate within new papers either
        if doi:
            existing_dois.add(doi)
        if title:
            existing_titles.add(title)

    logger.info("dedup.done", novel=len(novel), duplicates=dupes)
    return novel


async def resolve_full_text(
    papers: list[CandidatePaper],
) -> tuple[list[CandidatePaper], list[CandidatePaper]]:
    """Resolve full text for papers.

    Returns (papers_with_full_text, papers_without_full_text).
    """
    screened = [
        ScreenedPaper(paper=p, relevance_score=4, rationale="expanded corpus", include=True)
        for p in papers
    ]

    resolver = FullTextResolver()
    cache = FullTextCache(FULL_TEXT_CACHE_DIR)
    cached_resolver = CachedFullTextResolver(resolver, cache)

    try:
        source_counts = await cached_resolver.resolve(screened, max_concurrent=5)
        logger.info("fulltext.resolved", source_counts=source_counts)
    finally:
        await cached_resolver.close()

    full_text = [s.paper for s in screened if s.paper.full_text]
    no_full_text = [s.paper for s in screened if not s.paper.full_text]

    logger.info(
        "fulltext.filtered",
        full_text=len(full_text),
        no_full_text=len(no_full_text),
    )

    return full_text, no_full_text


async def main(*, retrieve_only: bool = False, resolve_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing corpus
    existing: list[CandidatePaper] = []
    if PAPERS_PATH.exists():
        raw = json.loads(PAPERS_PATH.read_text())
        existing = [CandidatePaper.model_validate(p) for p in raw]
    logger.info("existing.loaded", count=len(existing))

    # Phase 1: Retrieve from OpenAlex
    if resolve_only and CANDIDATES_PATH.exists():
        logger.info("retrieve.skipped", loading_from=str(CANDIDATES_PATH))
        raw = json.loads(CANDIDATES_PATH.read_text())
        novel = [CandidatePaper.model_validate(p) for p in raw]
    else:
        candidates = retrieve_from_openalex()
        novel = deduplicate(candidates, existing)

        # Save candidates for potential --resolve-only rerun
        CANDIDATES_PATH.write_text(
            json.dumps([p.model_dump() for p in novel], indent=2, default=str, ensure_ascii=False)
        )
        logger.info("candidates.saved", count=len(novel), path=str(CANDIDATES_PATH))

    if retrieve_only:
        print(
            f"\n  Retrieval complete: {len(novel)} new candidates saved to {CANDIDATES_PATH.name}"
        )
        print("  Run with --resolve-only to fetch full text")
        return

    # Phase 2: Resolve full text
    logger.info("fulltext.start", candidates=len(novel))
    new_with_text, inaccessible = await resolve_full_text(novel)

    # Phase 3: Save inaccessible papers report
    if inaccessible:
        inaccessible_records = []
        for p in inaccessible:
            record: dict[str, Any] = {
                "doi": p.doi,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "authors": p.authors[:3],
                "has_abstract": bool(p.abstract),
                "citation_count": p.citation_count,
                "external_ids": p.external_ids,
            }
            # Determine likely access route for manual retrieval
            doi = p.doi or ""
            if "10.1016" in doi:
                record["access_hint"] = "Elsevier — ScienceDirect institutional access or API key"
            elif "10.1002" in doi:
                record["access_hint"] = "Wiley — TDM token or institutional access"
            elif "10.1007" in doi or "10.1038" in doi:
                record["access_hint"] = "Springer/Nature — institutional access or SharedIt link"
            elif "10.1242" in doi:
                record["access_hint"] = "Company of Biologists — free after embargo, check PMC"
            elif "10.1093" in doi:
                record["access_hint"] = "Oxford Academic — institutional access"
            elif "10.1101" in doi:
                record["access_hint"] = "bioRxiv/medRxiv preprint — should be OA, retry"
            else:
                record["access_hint"] = "Check Unpaywall, Sci-Hub, or institutional access"
            inaccessible_records.append(record)

        inaccessible_path = OUTPUT_DIR / "papers_inaccessible.json"
        inaccessible_path.write_text(json.dumps(inaccessible_records, indent=2, ensure_ascii=False))

        # Also write a human-readable summary
        summary_path = OUTPUT_DIR / "inaccessible_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Inaccessible Papers — {len(inaccessible)} papers without full text\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Group by publisher/access hint
            by_hint: dict[str, list[dict]] = {}
            for r in inaccessible_records:
                hint = r["access_hint"]
                by_hint.setdefault(hint, []).append(r)

            for hint, papers_list in sorted(by_hint.items(), key=lambda x: -len(x[1])):
                f.write(f"\n--- {hint} ({len(papers_list)} papers) ---\n")
                for r in papers_list:
                    doi_str = r["doi"] or "no-doi"
                    f.write(f"  {doi_str:<45} {r['year'] or '?':>5}  {(r['title'] or '')[:60]}\n")

        logger.info(
            "inaccessible.saved",
            count=len(inaccessible),
            json_path=str(inaccessible_path),
            summary_path=str(summary_path),
        )

    # Phase 4: Merge into corpus
    merged = existing + new_with_text
    PAPERS_PATH.write_text(
        json.dumps([p.model_dump() for p in merged], indent=2, default=str, ensure_ascii=False)
    )

    # Source distribution for new papers
    source_dist: dict[str, int] = {}
    for p in new_with_text:
        src = p.full_text_source or "unknown"
        source_dist[src] = source_dist.get(src, 0) + 1

    print(f"\n{'=' * 70}")
    print("  Corpus Expansion Complete")
    print(f"{'=' * 70}")
    print(f"  Previous corpus:    {len(existing)} papers")
    print(f"  New papers added:   {len(new_with_text)}")
    print(f"  Inaccessible:       {len(inaccessible)}")
    print(f"  Total corpus:       {len(merged)} papers")
    print(f"  Saved to:           {PAPERS_PATH}")
    if inaccessible:
        print(f"  Inaccessible list:  {OUTPUT_DIR / 'papers_inaccessible.json'}")
        print(f"  Inaccessible guide: {OUTPUT_DIR / 'inaccessible_summary.txt'}")
    if source_dist:
        print("\n  Full text sources for new papers:")
        for src, count in sorted(source_dist.items(), key=lambda x: -x[1]):
            print(f"    {src:<25} {count:>4}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand gastruloid corpus via OpenAlex")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--retrieve-only", action="store_true", help="Just pull from OpenAlex, skip full text"
    )
    group.add_argument(
        "--resolve-only", action="store_true", help="Skip retrieval, resolve full text"
    )
    args = parser.parse_args()
    asyncio.run(main(retrieve_only=args.retrieve_only, resolve_only=args.resolve_only))
