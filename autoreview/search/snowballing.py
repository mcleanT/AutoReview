"""Backward and forward citation snowballing via Semantic Scholar REST API."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_FIELDS = "paperId,title,abstract,year,authors,journal,externalIds,citationCount"


@dataclass(frozen=True)
class SnowballConfig:
    """Configuration for citation snowballing."""

    max_seed_papers: int = 10
    max_refs_per_paper: int = 50
    max_citations_per_paper: int = 50
    enable_backward: bool = True
    enable_forward: bool = True


def _normalize_title(title: str) -> str:
    """Normalize a paper title for DOI-less deduplication.

    Lowercases, strips punctuation, collapses whitespace, and strips
    unicode accents so minor formatting differences don't create duplicates.
    """
    # NFKD decompose + strip combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    lower = ascii_only.lower()
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r"[^a-z0-9\s]", "", lower)
    # Collapse whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


def _s2_paper_to_candidate(data: dict[str, Any], source_label: str) -> CandidatePaper | None:
    """Convert a Semantic Scholar API paper dict to a CandidatePaper.

    Args:
        data: S2 API paper object (with fields from S2_PAPER_FIELDS).
        source_label: Value for source_database field (e.g. "snowball_backward").

    Returns:
        CandidatePaper or None if data is invalid (missing title).
    """
    try:
        title = data.get("title")
        if not title:
            return None

        authors = [a["name"] for a in data.get("authors", []) if a.get("name")]

        ext_raw = data.get("externalIds") or {}
        doi = ext_raw.get("DOI")
        external_ids: dict[str, str] = {}
        if ext_raw.get("PubMed"):
            external_ids["pmid"] = str(ext_raw["PubMed"])
        if ext_raw.get("ArXiv"):
            external_ids["arxiv"] = str(ext_raw["ArXiv"])
        if data.get("paperId"):
            external_ids["s2_id"] = data["paperId"]

        journal = (data.get("journal") or {}).get("name")

        return CandidatePaper(
            title=title,
            authors=authors,
            year=data.get("year"),
            journal=journal,
            doi=doi,
            abstract=data.get("abstract"),
            source_database=source_label,
            external_ids=external_ids,
            citation_count=data.get("citationCount"),
        )
    except Exception as e:
        logger.warning("snowball.parse_error", error=str(e))
        return None


async def _fetch_related(
    paper_id: str,
    endpoint: str,
    limit: int,
    client: httpx.AsyncClient,
    limiter: RateLimiter,
) -> list[dict[str, Any]]:
    """Fetch references or citations for a paper from the S2 API.

    Args:
        paper_id: Semantic Scholar paper ID (or DOI/ArXiv ID).
        endpoint: Either "references" or "citations".
        limit: Maximum number of related papers to retrieve.
        client: Shared httpx.AsyncClient.
        limiter: RateLimiter instance to respect API rate limits.

    Returns:
        List of raw paper dicts extracted from the S2 API response.
    """
    url = f"{S2_API_BASE}/paper/{paper_id}/{endpoint}"
    params: dict[str, str | int] = {"fields": S2_PAPER_FIELDS, "limit": min(limit, 500)}

    await limiter.acquire()
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "snowball.http_error",
            paper_id=paper_id,
            endpoint=endpoint,
            status=e.response.status_code,
        )
        return []
    except httpx.RequestError as e:
        logger.warning("snowball.request_error", paper_id=paper_id, endpoint=endpoint, error=str(e))
        return []

    data = resp.json()
    items = data.get("data", [])

    # References wrap the paper under "citedPaper"; citations under "citingPaper"
    nested_key = "citedPaper" if endpoint == "references" else "citingPaper"
    papers: list[dict[str, Any]] = []
    for item in items:
        paper = item.get(nested_key) or item
        if paper:
            papers.append(paper)
    return papers


async def snowball_papers(
    seed_papers: list[CandidatePaper],
    config: SnowballConfig | None = None,
    existing_dois: set[str] | None = None,
    existing_titles: set[str] | None = None,
) -> list[CandidatePaper]:
    """Discover new papers via backward and forward citation snowballing.

    For each seed paper (up to config.max_seed_papers), fetch its references
    (backward) and/or papers that cite it (forward) using the Semantic Scholar
    Graph API. Deduplicates against existing_dois / existing_titles and against
    papers found within this snowball run.

    Args:
        seed_papers: Papers to snowball from (uses their S2 ID, DOI, or title).
        config: Snowball configuration; defaults to SnowballConfig().
        existing_dois: Set of already-known normalized DOIs to skip.
        existing_titles: Set of already-known normalized titles to skip.

    Returns:
        List of new CandidatePaper objects not seen before.
    """
    if config is None:
        config = SnowballConfig()

    seen_dois: set[str] = set(existing_dois or [])
    seen_titles: set[str] = set(existing_titles or [])

    api_key = os.environ.get("S2_API_KEY")
    # With API key: 1 req/s; without: 1 req/s (free tier)
    rate = 1.0 if api_key else 1.0
    limiter = RateLimiter(rate)

    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    # Trim seeds to configured maximum
    seeds = seed_papers[: config.max_seed_papers]

    new_papers: list[CandidatePaper] = []

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for seed in seeds:
            # Determine best paper identifier for S2 API
            s2_id = seed.external_ids.get("s2_id")
            doi = seed.doi
            if s2_id:
                paper_id = s2_id
            elif doi:
                paper_id = f"DOI:{doi}"
            else:
                # Cannot fetch without a usable identifier
                logger.debug(
                    "snowball.skip_seed",
                    title=seed.title[:80],
                    reason="no s2_id or doi",
                )
                continue

            log = logger.bind(seed_title=seed.title[:80], paper_id=paper_id)

            if config.enable_backward:
                raw_refs = await _fetch_related(
                    paper_id,
                    "references",
                    config.max_refs_per_paper,
                    client,
                    limiter,
                )
                log.debug("snowball.backward_fetched", count=len(raw_refs))
                for raw in raw_refs:
                    candidate = _s2_paper_to_candidate(raw, "snowball_backward")
                    if candidate is None:
                        continue
                    if _is_duplicate(candidate, seen_dois, seen_titles):
                        continue
                    _register(candidate, seen_dois, seen_titles)
                    new_papers.append(candidate)

            if config.enable_forward:
                raw_cites = await _fetch_related(
                    paper_id,
                    "citations",
                    config.max_citations_per_paper,
                    client,
                    limiter,
                )
                log.debug("snowball.forward_fetched", count=len(raw_cites))
                for raw in raw_cites:
                    candidate = _s2_paper_to_candidate(raw, "snowball_forward")
                    if candidate is None:
                        continue
                    if _is_duplicate(candidate, seen_dois, seen_titles):
                        continue
                    _register(candidate, seen_dois, seen_titles)
                    new_papers.append(candidate)

    logger.info("snowball.complete", new_papers=len(new_papers), seeds=len(seeds))
    return new_papers


def _is_duplicate(
    paper: CandidatePaper,
    seen_dois: set[str],
    seen_titles: set[str],
) -> bool:
    """Return True if this paper is already known."""
    if paper.doi and paper.doi in seen_dois:
        return True
    norm = _normalize_title(paper.title)
    return norm in seen_titles


def _register(
    paper: CandidatePaper,
    seen_dois: set[str],
    seen_titles: set[str],
) -> None:
    """Add a paper's identifiers to the dedup sets."""
    if paper.doi:
        seen_dois.add(paper.doi)
    seen_titles.add(_normalize_title(paper.title))
