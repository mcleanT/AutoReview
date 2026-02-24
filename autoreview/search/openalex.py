from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any

import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()


class OpenAlexSearch:
    """OpenAlex search via pyalex (wrapped in executor since it's sync)."""

    def __init__(self, email: str | None = None) -> None:
        self._email = email or os.environ.get("OPENALEX_EMAIL")
        self._limiter = RateLimiter(10.0)

    @property
    def source_name(self) -> str:
        return "openalex"

    def _setup_pyalex(self) -> None:
        import pyalex
        if self._email:
            pyalex.config.email = self._email

    def _sync_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self._setup_pyalex()
        from pyalex import Works
        results = []
        for page in Works().search(query).paginate(per_page=min(200, max_results)):
            results.extend(page)
            if len(results) >= max_results:
                break
        return results[:max_results]

    def _parse_work(self, work: dict[str, Any]) -> CandidatePaper | None:
        try:
            title = work.get("title")
            if not title:
                return None

            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)

            doi = work.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[16:]

            external_ids: dict[str, str] = {}
            oa_id = work.get("id", "")
            if oa_id:
                external_ids["openalex"] = oa_id
            ids = work.get("ids", {})
            if ids.get("pmid"):
                pmid = ids["pmid"]
                if pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
                    pmid = pmid.split("/")[-1]
                external_ids["pmid"] = pmid

            abstract = None
            inv_index = work.get("abstract_inverted_index")
            if inv_index:
                # Reconstruct abstract from inverted index
                word_positions: list[tuple[int, str]] = []
                for word, positions in inv_index.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)

            source = work.get("primary_location", {}).get("source", {}) or {}
            journal = source.get("display_name")

            return CandidatePaper(
                title=title, authors=authors,
                year=work.get("publication_year"),
                journal=journal, doi=doi, abstract=abstract,
                source_database="openalex",
                external_ids=external_ids,
                citation_count=work.get("cited_by_count"),
            )
        except Exception as e:
            logger.warning("openalex.parse_error", error=str(e))
            return None

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        loop = asyncio.get_event_loop()
        all_papers: list[CandidatePaper] = []
        per_query = max(max_results // len(queries), 20) if queries else max_results

        for query in queries:
            await self._limiter.acquire()
            try:
                works = await loop.run_in_executor(None, partial(self._sync_search, query, per_query))
                papers = [p for w in works if (p := self._parse_work(w)) is not None]
                all_papers.extend(papers)
                logger.info("openalex.search", query=query[:80], results=len(papers))
            except Exception as e:
                logger.warning("openalex.search_error", query=query[:80], error=str(e))

        logger.info("openalex.search.complete", total_papers=len(all_papers))
        return all_papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        loop = asyncio.get_event_loop()
        try:
            self._setup_pyalex()
            from pyalex import Works
            work = await loop.run_in_executor(None, lambda: Works()[paper_id])
            return self._parse_work(work) if work else None
        except Exception as e:
            logger.warning("openalex.details_error", paper_id=paper_id, error=str(e))
            return None
