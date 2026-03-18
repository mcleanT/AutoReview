from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

CORE_API_BASE = "https://api.core.ac.uk/v3"
_MAX_RETRIES = 3
_RETRY_BACKOFF = [2.0, 5.0, 10.0]


class CORESearch:
    """CORE (core.ac.uk) search via REST API v3 (httpx).

    CORE provides access to 300M+ open access research papers.
    Rate limit: 10 req/sec with API key.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("CORE_API_KEY")
        self._limiter = RateLimiter(10.0 if self._api_key else 1.0)
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    @property
    def source_name(self) -> str:
        return "core"

    def _parse_work(self, data: dict[str, Any]) -> CandidatePaper | None:
        try:
            title = data.get("title")
            if not title:
                return None

            # Authors: CORE returns list of author objects with "name" field
            raw_authors = data.get("authors") or []
            authors: list[str] = []
            for a in raw_authors:
                if isinstance(a, dict):
                    name = a.get("name") or a.get("displayName")
                elif isinstance(a, str):
                    name = a
                else:
                    name = None
                if name:
                    authors.append(name)

            doi = data.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[16:]

            external_ids: dict[str, str] = {}
            core_id = data.get("id") or data.get("coreId")
            if core_id:
                external_ids["core"] = str(core_id)

            download_url = data.get("downloadUrl")
            if download_url:
                external_ids["core_pdf_url"] = download_url

            year = data.get("yearPublished") or data.get("year")
            if isinstance(year, str):
                try:
                    year = int(year[:4])
                except (ValueError, TypeError):
                    year = None

            abstract = data.get("abstract")

            # Journal from data_provider or publisher
            journal = None
            journals = data.get("journals")
            if journals and isinstance(journals, list) and journals:
                journal = journals[0].get("title") if isinstance(journals[0], dict) else None
            if not journal:
                journal = data.get("publisher")

            return CandidatePaper(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                abstract=abstract,
                source_database="core",
                external_ids=external_ids,
            )
        except Exception as e:
            logger.warning("core.parse_error", error=str(e))
            return None

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        papers: list[CandidatePaper] = []
        per_query_max = max(max_results // len(queries), 20) if queries else max_results
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            for query in queries:
                offset = 0
                query_papers: list[CandidatePaper] = []

                while len(query_papers) < per_query_max:
                    await self._limiter.acquire()
                    limit = min(100, per_query_max - len(query_papers))

                    for attempt in range(_MAX_RETRIES):
                        try:
                            resp = await client.get(
                                f"{CORE_API_BASE}/search/works",
                                params={
                                    "q": query,
                                    "limit": limit,
                                    "offset": offset,
                                },
                            )
                            if resp.status_code == 429:
                                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                                logger.warning("core.rate_limited", query=query[:80], wait=wait)
                                await asyncio.sleep(wait)
                                continue
                            resp.raise_for_status()
                            break
                        except httpx.HTTPStatusError as e:
                            if attempt < _MAX_RETRIES - 1:
                                logger.warning(
                                    "core.search_retry",
                                    status=e.response.status_code,
                                    attempt=attempt + 1,
                                    query=query[:80],
                                )
                                await asyncio.sleep(_RETRY_BACKOFF[attempt])
                            else:
                                logger.warning(
                                    "core.search_error",
                                    status=e.response.status_code,
                                    query=query[:80],
                                )
                                resp = None
                            continue
                        except httpx.RequestError as e:
                            if attempt < _MAX_RETRIES - 1:
                                await asyncio.sleep(_RETRY_BACKOFF[attempt])
                            else:
                                logger.warning("core.request_error", error=str(e), query=query[:80])
                                resp = None
                            continue
                    else:
                        break

                    if resp is None:
                        break

                    try:
                        data = resp.json()
                    except Exception as e:
                        logger.warning("core.json_parse_error", error=str(e))
                        break

                    results = data.get("results", [])
                    if not results:
                        break

                    for r in results:
                        core_id = str(r.get("id") or r.get("coreId") or "")
                        dedup_key = core_id or (r.get("doi") or r.get("title") or "")
                        if dedup_key and dedup_key in seen_ids:
                            continue
                        if dedup_key:
                            seen_ids.add(dedup_key)
                        paper = self._parse_work(r)
                        if paper:
                            query_papers.append(paper)

                    offset += limit
                    total = data.get("totalHits", 0)
                    if offset >= total:
                        break

                logger.info("core.search", query=query[:80], results=len(query_papers))
                papers.extend(query_papers)

        logger.info("core.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        """Fetch details for a single CORE paper by its CORE ID."""
        await self._limiter.acquire()
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.get(f"{CORE_API_BASE}/outputs/{paper_id}")
                    if resp.status_code == 429:
                        wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return self._parse_work(resp.json())
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "core.details_error",
                        paper_id=paper_id,
                        status=e.response.status_code,
                    )
                    return None
                except httpx.RequestError as e:
                    logger.warning("core.details_request_error", paper_id=paper_id, error=str(e))
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(_RETRY_BACKOFF[attempt])
                    else:
                        return None
        return None
