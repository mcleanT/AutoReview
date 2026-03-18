"""Full-text retrieval via the Unpaywall API.

Looks up DOIs on Unpaywall and fetches full text from any available
PDF or HTML location. Tries all available URLs regardless of the
is_oa flag, since Unpaywall can return usable locations (bronze,
author manuscripts, institutional copies) even for non-OA papers.

Iteration strategy:
1. Try ``best_oa_location`` first (pdf then html).
2. If that fails, iterate through the full ``oa_locations`` array in
   order, trying ``url_for_pdf`` then ``url_for_landing_page`` for
   each entry, stopping on the first successful extraction.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

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
    # Full oa_locations array from the API response, used for fallback iteration.
    oa_locations: list[dict[str, Any]] = []


class UnpaywallClient:
    """Client for retrieving full text via Unpaywall."""

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
            oa_locations: list[dict[str, Any]] = data.get("oa_locations") or []
            return UnpaywallResult(
                doi=doi,
                is_oa=data.get("is_oa", False),
                best_oa_url=best_oa.get("url"),
                oa_status=data.get("oa_status", "closed"),
                pdf_url=best_oa.get("url_for_pdf"),
                html_url=best_oa.get("url_for_landing_page"),
                oa_locations=oa_locations,
            )
        except Exception as e:
            logger.warning("unpaywall.lookup_failed", doi=doi, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_pdf_url(self, doi: str, pdf_url: str) -> str | None:
        """Fetch and extract text from a single PDF URL.

        Returns extracted text on success, None otherwise.
        """
        try:
            await self._limiter.acquire()
            resp = await self._client.get(pdf_url, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 0:
                text = _extract_text_from_pdf(resp.content)
                if text and len(text) > 100:
                    return text
        except Exception as e:
            logger.debug("unpaywall.pdf_fetch_failed", doi=doi, url=pdf_url, error=str(e))
        return None

    async def _try_html_url(self, doi: str, html_url: str) -> str | None:
        """Fetch and extract text from a single HTML/landing-page URL.

        Returns extracted text on success, None otherwise.
        """
        try:
            await self._limiter.acquire()
            resp = await self._client.get(html_url, follow_redirects=True)
            if resp.status_code == 200:
                text = _extract_text_from_html(resp.text)
                if text and len(text) > 100:
                    return text
        except Exception as e:
            logger.debug("unpaywall.html_fetch_failed", doi=doi, url=html_url, error=str(e))
        return None

    async def fetch_full_text(self, result: UnpaywallResult) -> str | None:
        """Fetch and extract full text from an Unpaywall result.

        Strategy:
        1. Try ``best_oa_location`` (pdf then html).
        2. If that fails, iterate through ``oa_locations`` in order,
           trying ``url_for_pdf`` then ``url_for_landing_page`` for each
           entry, stopping on the first successful extraction.

        Attempts all available URLs regardless of is_oa status — Unpaywall
        can surface bronze, green, or institutional copies that are
        accessible even for non-OA papers.

        Returns None if no URLs are available or all extractions fail.
        """
        doi = result.doi

        # ---- Step 1: best_oa_location --------------------------------
        if result.pdf_url:
            text = await self._try_pdf_url(doi, result.pdf_url)
            if text:
                logger.debug(
                    "unpaywall.fetch_success", doi=doi, source="best_oa_location", url_type="pdf"
                )
                return text

        if result.html_url:
            text = await self._try_html_url(doi, result.html_url)
            if text:
                logger.debug(
                    "unpaywall.fetch_success", doi=doi, source="best_oa_location", url_type="html"
                )
                return text

        # ---- Step 2: iterate oa_locations ----------------------------
        locations = result.oa_locations
        if not locations:
            if result.pdf_url or result.html_url:
                # best_oa_location already tried above; nothing more to do
                logger.debug(
                    "unpaywall.all_locations_failed",
                    doi=doi,
                    locations_tried=1,
                    reason="best_oa_location only (no oa_locations array)",
                )
            return None

        tried = 0
        for idx, loc in enumerate(locations):
            pdf_url: str | None = loc.get("url_for_pdf")
            html_url: str | None = loc.get("url_for_landing_page")

            # Skip locations whose URLs duplicate best_oa_location (already tried)
            if pdf_url and pdf_url == result.pdf_url:
                pdf_url = None
            if html_url and html_url == result.html_url:
                html_url = None

            if not pdf_url and not html_url:
                continue

            tried += 1

            if pdf_url:
                text = await self._try_pdf_url(doi, pdf_url)
                if text:
                    logger.info(
                        "unpaywall.fetch_success",
                        doi=doi,
                        source="oa_locations",
                        location_index=idx,
                        url_type="pdf",
                    )
                    return text

            if html_url:
                text = await self._try_html_url(doi, html_url)
                if text:
                    logger.info(
                        "unpaywall.fetch_success",
                        doi=doi,
                        source="oa_locations",
                        location_index=idx,
                        url_type="html",
                    )
                    return text

        logger.warning(
            "unpaywall.all_locations_failed",
            doi=doi,
            locations_tried=tried + (1 if result.pdf_url or result.html_url else 0),
            reason="all URLs exhausted",
        )
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
            if not result or (not result.pdf_url and not result.html_url):
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
        import fitz  # type: ignore[import-untyped]  # PyMuPDF

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
