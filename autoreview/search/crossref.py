from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

CROSSREF_API_BASE = "https://api.crossref.org/works"


def _strip_jats_tags(text: str) -> str:
    """Strip JATS XML tags from a CrossRef abstract string.

    CrossRef abstracts are returned as JATS XML fragments, e.g.:
      <jats:p>This is the abstract.</jats:p>
    """
    # Remove all XML/HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_year(data: dict[str, Any]) -> int | None:
    """Extract publication year from CrossRef work record.

    Tries published-print first, then published-online, then issued.
    """
    for field in ("published-print", "published-online", "issued"):
        date_parts = data.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int) and year > 1000:
                return year
    return None


class CrossRefSearch:
    """CrossRef search via REST API (httpx, async).

    Uses the CrossRef polite pool via the ``mailto`` query parameter for
    50 req/sec (vs. 1 req/sec anonymous).  Pass an email address or set
    CROSSREF_EMAIL / ENTREZ_EMAIL in the environment.
    """

    def __init__(self, email: str | None = None) -> None:
        self._email = email or os.environ.get("CROSSREF_EMAIL") or os.environ.get("ENTREZ_EMAIL")
        # Polite pool: 50 req/sec; anonymous: 1 req/sec
        rate = 50.0 if self._email else 1.0
        self._limiter = RateLimiter(rate)

    @property
    def source_name(self) -> str:
        return "crossref"

    def _parse_work(self, item: dict[str, Any]) -> CandidatePaper | None:
        try:
            titles = item.get("title", [])
            if not titles:
                return None
            title = titles[0]
            if not title:
                return None

            # Authors: each has optional "given" and "family" fields
            authors: list[str] = []
            for author in item.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                name = f"{given} {family}".strip() if given else family
                if name:
                    authors.append(name)

            year = _parse_year(item)

            doi = item.get("DOI")
            if doi:
                doi = doi.lower().strip()

            # Abstract: CrossRef returns JATS XML fragments — strip tags
            raw_abstract = item.get("abstract")
            abstract: str | None = None
            if raw_abstract:
                abstract = _strip_jats_tags(raw_abstract)
                if not abstract:
                    abstract = None

            # Journal name from container-title
            container_titles = item.get("container-title", [])
            journal = container_titles[0] if container_titles else None

            # External IDs
            external_ids: dict[str, str] = {}

            # Store CrossRef link array as JSON for full-text resolution
            links = item.get("link", [])
            if links:
                external_ids["crossref_links"] = json.dumps(links)

            citation_count = item.get("is-referenced-by-count")

            return CandidatePaper(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                abstract=abstract,
                source_database="crossref",
                external_ids=external_ids,
                citation_count=citation_count,
            )
        except Exception as e:
            logger.warning("crossref.parse_error", error=str(e))
            return None

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        papers: list[CandidatePaper] = []
        per_query_max = max(max_results // len(queries), 20) if queries else max_results
        seen_dois: set[str] = set()

        params_base: dict[str, Any] = {"rows": min(per_query_max, 1000)}
        if self._email:
            params_base["mailto"] = self._email

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": f"AutoReview/1.0 (mailto:{self._email or 'anonymous'})"},
        ) as client:
            for query in queries:
                await self._limiter.acquire()
                params = {**params_base, "query": query}

                try:
                    resp = await client.get(CROSSREF_API_BASE, params=params)
                    if resp.status_code == 429:
                        logger.warning("crossref.rate_limited", query=query[:80])
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "crossref.search_error",
                        status=e.response.status_code,
                        query=query[:80],
                    )
                    continue
                except httpx.RequestError as e:
                    logger.warning("crossref.request_error", error=str(e), query=query[:80])
                    continue

                items = data.get("message", {}).get("items", [])
                query_papers: list[CandidatePaper] = []
                for item in items:
                    doi_key = (item.get("DOI") or "").lower().strip()
                    if doi_key and doi_key in seen_dois:
                        continue
                    if doi_key:
                        seen_dois.add(doi_key)
                    paper = self._parse_work(item)
                    if paper:
                        query_papers.append(paper)

                logger.info("crossref.search", query=query[:80], results=len(query_papers))
                papers.extend(query_papers)

        logger.info("crossref.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        """Fetch a single work by DOI.

        Args:
            paper_id: A CrossRef DOI (e.g. '10.1038/nature12345').
        """
        await self._limiter.acquire()
        params: dict[str, Any] = {}
        if self._email:
            params["mailto"] = self._email

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": f"AutoReview/1.0 (mailto:{self._email or 'anonymous'})"},
        ) as client:
            try:
                resp = await client.get(f"{CROSSREF_API_BASE}/{paper_id}", params=params)
                resp.raise_for_status()
                data = resp.json()
                item = data.get("message", {})
                return self._parse_work(item)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("crossref.details_error", paper_id=paper_id, error=str(e))
                return None
