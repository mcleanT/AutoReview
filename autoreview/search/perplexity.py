from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

PERPLEXITY_API_BASE = "https://api.perplexity.ai"


class PerplexitySearch:
    """Perplexity Sonar API for AI-powered paper discovery."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self._limiter = RateLimiter(5.0)

    @property
    def source_name(self) -> str:
        return "perplexity"

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        """Use Perplexity Sonar to discover papers.

        Perplexity returns natural language with citations, not structured
        metadata. Papers found here should be cross-referenced with S2/OpenAlex
        for complete metadata.
        """
        if not self._api_key:
            logger.warning("perplexity.no_api_key")
            return []

        papers: list[CandidatePaper] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries:
                await self._limiter.acquire()
                try:
                    response = await client.post(
                        f"{PERPLEXITY_API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "sonar",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a research assistant. List academic papers "
                                    "relevant to the query. For each paper, provide: title, "
                                    "authors, year, and DOI if available.",
                                },
                                {"role": "user", "content": query},
                            ],
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Extract citations from the response
                    citations = data.get("citations", [])
                    for citation in citations:
                        if isinstance(citation, str):
                            papers.append(
                                CandidatePaper(
                                    title=citation[:200],
                                    authors=[],
                                    source_database="perplexity",
                                    external_ids={"perplexity_url": citation},
                                )
                            )

                    logger.info("perplexity.search", query=query[:80], citations=len(citations))

                except httpx.HTTPStatusError as e:
                    logger.warning("perplexity.error", status=e.response.status_code, query=query[:80])
                except httpx.RequestError as e:
                    logger.warning("perplexity.request_error", error=str(e), query=query[:80])

        logger.info("perplexity.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        return None  # Perplexity doesn't support direct paper lookup
