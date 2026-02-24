from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,year,authors,journal,externalIds,citationCount"


class SemanticScholarSearch:
    """Semantic Scholar search via REST API (httpx)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("S2_API_KEY")
        rate = 1.0 if self._api_key else 0.5
        self._limiter = RateLimiter(rate)
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["x-api-key"] = self._api_key

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    def _parse_paper(self, data: dict[str, Any]) -> CandidatePaper | None:
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
                title=title, authors=authors, year=data.get("year"),
                journal=journal, doi=doi, abstract=data.get("abstract"),
                source_database="semantic_scholar", external_ids=external_ids,
                citation_count=data.get("citationCount"),
            )
        except Exception as e:
            logger.warning("s2.parse_error", error=str(e))
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

                    try:
                        resp = await client.get(
                            f"{S2_API_BASE}/paper/search",
                            params={"query": query, "offset": offset, "limit": limit, "fields": S2_FIELDS},
                        )
                        if resp.status_code == 429:
                            logger.warning("s2.rate_limited", query=query[:80])
                            await asyncio.sleep(5)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                    except httpx.HTTPStatusError as e:
                        logger.warning("s2.search_error", status=e.response.status_code, query=query[:80])
                        break
                    except httpx.RequestError as e:
                        logger.warning("s2.request_error", error=str(e), query=query[:80])
                        break

                    results = data.get("data", [])
                    if not results:
                        break

                    for r in results:
                        s2_id = r.get("paperId", "")
                        if s2_id in seen_ids:
                            continue
                        seen_ids.add(s2_id)
                        paper = self._parse_paper(r)
                        if paper:
                            query_papers.append(paper)

                    offset += limit
                    if offset >= data.get("total", 0):
                        break

                logger.info("s2.search", query=query[:80], results=len(query_papers))
                papers.extend(query_papers)

        logger.info("s2.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        await self._limiter.acquire()
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            try:
                resp = await client.get(f"{S2_API_BASE}/paper/{paper_id}", params={"fields": S2_FIELDS})
                resp.raise_for_status()
                return self._parse_paper(resp.json())
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("s2.details_error", paper_id=paper_id, error=str(e))
                return None
