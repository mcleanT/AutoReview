"""Full-text retrieval via the Unpaywall API.

Looks up open-access availability by DOI and fetches full text from
available PDF or HTML sources. Gracefully degrades when papers are
not open access or when text extraction fails.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog
from pydantic import Field

from autoreview.models.base import AutoReviewModel
from autoreview.models.paper import ScreenedPaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


class UnpaywallResult(AutoReviewModel):
    """Result from an Unpaywall DOI lookup."""

    doi: str
    is_oa: bool = False
    best_oa_url: str | None = None
    oa_status: str = ""  # e.g. "gold", "green", "hybrid", "bronze", "closed"
    pdf_url: str | None = None
    html_url: str | None = None


class UnpaywallClient:
    """Client for retrieving open-access full text via Unpaywall."""

    def __init__(
        self,
        email: str,
        requests_per_second: float = 10.0,
    ) -> None:
        self.email = email
        self._client = httpx.AsyncClient(timeout=30.0)
        self._limiter = RateLimiter(requests_per_second)

    async def lookup_doi(self, doi: str) -> UnpaywallResult | None:
        """Look up a DOI on Unpaywall to check open-access availability.

        Returns None on network errors or non-200 responses.
        """
        await self._limiter.acquire()
        url = f"{_UNPAYWALL_BASE}/{doi}?email={self.email}"

        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return None

            data = resp.json()
            best_oa = data.get("best_oa_location") or {}
            return UnpaywallResult(
                doi=doi,
                is_oa=data.get("is_oa", False),
                best_oa_url=best_oa.get("url"),
                oa_status=data.get("oa_status", "closed"),
                pdf_url=best_oa.get("url_for_pdf"),
                html_url=best_oa.get("url_for_landing_page"),
            )
        except Exception as e:
            logger.warning("unpaywall.lookup_failed", doi=doi, error=str(e))
            return None

    async def fetch_full_text(self, result: UnpaywallResult) -> str | None:
        """Fetch and extract full text from an Unpaywall result.

        Tries PDF first, then HTML. Returns None if extraction fails.
        """
        if not result.is_oa:
            return None

        # Try PDF first
        if result.pdf_url:
            try:
                await self._limiter.acquire()
                resp = await self._client.get(result.pdf_url, follow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 0:
                    text = _extract_text_from_pdf(resp.content)
                    if text and len(text) > 100:
                        return text
            except Exception as e:
                logger.debug("unpaywall.pdf_fetch_failed", doi=result.doi, error=str(e))

        # Fallback to HTML
        if result.html_url:
            try:
                await self._limiter.acquire()
                resp = await self._client.get(result.html_url, follow_redirects=True)
                if resp.status_code == 200:
                    text = _extract_text_from_html(resp.text)
                    if text and len(text) > 100:
                        return text
            except Exception as e:
                logger.debug("unpaywall.html_fetch_failed", doi=result.doi, error=str(e))

        return None

    async def enrich_papers(
        self,
        papers: list[ScreenedPaper],
        max_concurrent: int = 5,
    ) -> tuple[int, int]:
        """Enrich screened papers with full text where available.

        Args:
            papers: List of screened papers to enrich.
            max_concurrent: Maximum concurrent downloads.

        Returns:
            Tuple of (attempted, enriched) counts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        attempted = 0
        enriched = 0

        async def _process(sp: ScreenedPaper) -> bool:
            if not sp.paper.doi:
                return False
            if sp.paper.full_text:
                return False  # Already has full text

            result = await self.lookup_doi(sp.paper.doi)
            if not result or not result.is_oa:
                return False

            text = await self.fetch_full_text(result)
            if text:
                sp.paper.full_text = text
                return True
            return False

        async def _bounded_process(sp: ScreenedPaper) -> bool:
            async with semaphore:
                return await _process(sp)

        tasks = []
        for sp in papers:
            if sp.paper.doi:
                attempted += 1
                tasks.append(_bounded_process(sp))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if r is True:
                    enriched += 1

        logger.info(
            "unpaywall.enrich_complete",
            attempted=attempted,
            enriched=enriched,
        )
        return attempted, enriched

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts).strip() or None
    except ImportError:
        logger.warning("unpaywall.pymupdf_not_installed")
        return None
    except Exception as e:
        logger.debug("unpaywall.pdf_extraction_failed", error=str(e))
        return None


def _extract_text_from_html(html: str) -> str | None:
    """Extract text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text if text else None
    except ImportError:
        logger.warning("unpaywall.beautifulsoup_not_installed")
        return None
    except Exception as e:
        logger.debug("unpaywall.html_extraction_failed", error=str(e))
        return None
