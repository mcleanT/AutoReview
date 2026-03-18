from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

EUROPE_PMC_API_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Europe PMC allows up to 10 req/sec — be polite and stay well under
_RATE_LIMIT = 8.0


class EuropePMCSearch:
    """Europe PMC search via REST API (httpx).

    Europe PMC indexes 40M+ life sciences literature records including
    PubMed, PubMed Central, preprints, and patents. No API key required.
    PMCIDs from results enable direct full-text XML access.
    Rate limit: ~10 req/sec (we use 8 to stay polite).
    """

    def __init__(self) -> None:
        self._limiter = RateLimiter(_RATE_LIMIT)

    @property
    def source_name(self) -> str:
        return "europe_pmc"

    def _parse_result(self, data: dict[str, Any]) -> CandidatePaper | None:
        try:
            title = data.get("title")
            if not title:
                return None

            # authorString is semicolon-separated, e.g. "Smith A, Jones B; White C"
            # Europe PMC uses ", " between authors within a multi-author list
            # and the full field is a single comma-separated string like "Smith A, Jones B."
            author_string = data.get("authorString", "") or ""
            # Strip trailing period, then split on ", "
            author_string = author_string.rstrip(".")
            authors: list[str] = [a.strip() for a in author_string.split(",") if a.strip()]

            doi = data.get("doi")

            year_raw = data.get("pubYear")
            year: int | None = None
            if year_raw is not None:
                try:
                    year = int(year_raw)
                except (ValueError, TypeError):
                    year = None

            abstract = data.get("abstractText")

            external_ids: dict[str, str] = {}

            pmid = data.get("pmid")
            if pmid:
                external_ids["pmid"] = str(pmid)

            pmcid = data.get("pmcid")
            if pmcid:
                external_ids["pmcid"] = str(pmcid)

            epmc_id = data.get("id")
            if epmc_id:
                external_ids["europepmc"] = str(epmc_id)

            # Journal name
            journal = data.get("journalTitle") or data.get("journalAbbreviation")

            # Citation count
            citation_count: int | None = None
            cited_by = data.get("citedByCount")
            if cited_by is not None:
                try:
                    citation_count = int(cited_by)
                except (ValueError, TypeError):
                    citation_count = None

            return CandidatePaper(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                abstract=abstract,
                source_database="europe_pmc",
                external_ids=external_ids,
                citation_count=citation_count,
            )
        except Exception as e:
            logger.warning("europe_pmc.parse_error", error=str(e))
            return None

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        papers: list[CandidatePaper] = []
        per_query_max = max(max_results // len(queries), 20) if queries else max_results
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "AutoReview/1.0 (europe-pmc-search)"},
        ) as client:
            for query in queries:
                cursor_mark = "*"
                query_papers: list[CandidatePaper] = []

                while len(query_papers) < per_query_max:
                    await self._limiter.acquire()
                    page_size = min(1000, per_query_max - len(query_papers))

                    try:
                        resp = await client.get(
                            f"{EUROPE_PMC_API_BASE}/search",
                            params={
                                "query": query,
                                "resultType": "core",
                                "pageSize": page_size,
                                "cursorMark": cursor_mark,
                                "format": "json",
                            },
                        )
                        if resp.status_code == 429:
                            logger.warning("europe_pmc.rate_limited", query=query[:80])
                            await asyncio.sleep(5)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                    except httpx.HTTPStatusError as e:
                        logger.warning(
                            "europe_pmc.search_error",
                            status=e.response.status_code,
                            query=query[:80],
                        )
                        break
                    except httpx.RequestError as e:
                        logger.warning("europe_pmc.request_error", error=str(e), query=query[:80])
                        break

                    result_list = data.get("resultList", {})
                    results = result_list.get("result", []) if result_list else []
                    if not results:
                        break

                    for r in results:
                        # Deduplicate by pmid > pmcid > europepmc id > title
                        dedup_key = (
                            r.get("pmid")
                            or r.get("pmcid")
                            or r.get("id")
                            or r.get("title", "")[:60]
                        )
                        if dedup_key and dedup_key in seen_ids:
                            continue
                        if dedup_key:
                            seen_ids.add(dedup_key)
                        paper = self._parse_result(r)
                        if paper:
                            query_papers.append(paper)

                    # Europe PMC uses cursor-based pagination
                    next_cursor = data.get("nextCursorMark")
                    if not next_cursor or next_cursor == cursor_mark:
                        break
                    cursor_mark = next_cursor

                    # Check if we've consumed all available results
                    hit_count = data.get("hitCount", 0)
                    if len(query_papers) >= hit_count:
                        break

                logger.info("europe_pmc.search", query=query[:80], results=len(query_papers))
                papers.extend(query_papers)

        logger.info("europe_pmc.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        """Fetch details for a single paper by its Europe PMC ID or PMID.

        Args:
            paper_id: Europe PMC ID, PMID, or PMCID.

        Returns:
            CandidatePaper or None if not found.
        """
        await self._limiter.acquire()
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "AutoReview/1.0 (europe-pmc-search)"},
        ) as client:
            try:
                resp = await client.get(
                    f"{EUROPE_PMC_API_BASE}/search",
                    params={
                        "query": f"EXT_ID:{paper_id}",
                        "resultType": "core",
                        "pageSize": 1,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                result_list = data.get("resultList", {})
                results = result_list.get("result", []) if result_list else []
                if results:
                    return self._parse_result(results[0])
                return None
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("europe_pmc.details_error", paper_id=paper_id, error=str(e))
                return None
