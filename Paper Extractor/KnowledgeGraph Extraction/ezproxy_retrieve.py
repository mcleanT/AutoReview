#!/usr/bin/env python3
"""EZproxy-based full text retrieval for inaccessible papers.

Uses Penn's EZproxy gateway (https://proxy.library.upenn.edu/login?url=)
to retrieve full text for papers in papers_inaccessible.json.
Requires VPN access to Penn's network.

Open-access preprint DOIs (10.1101/, 10.20944/, 10.2139/, 10.64898/) are
fetched directly without EZproxy.

Usage:
    python ezproxy_retrieve.py              # Full run
    python ezproxy_retrieve.py --dry-run    # Show what would be fetched, no requests
    python ezproxy_retrieve.py --limit N    # Process only first N papers
    python ezproxy_retrieve.py --direct-only  # Only fetch direct OA content, skip EZproxy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_AUTOREVIEW_ROOT))

import httpx
import structlog
from bs4 import BeautifulSoup

from autoreview.models.paper import CandidatePaper

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
INACCESSIBLE_PATH = OUTPUT_DIR / "papers_inaccessible.json"
PAPERS_PATH = OUTPUT_DIR / "papers.json"

EZPROXY_BASE = "https://proxy.library.upenn.edu/login?url="
EZPROXY_LOGIN = "https://proxy.library.upenn.edu/login"

# Minimum characters for a retrieved text to be considered successful
MIN_TEXT_LENGTH = 5000

# Rate limiting: seconds between requests
REQUEST_DELAY = 1.0

# Concurrency limit for parallel fetches
SEM = asyncio.Semaphore(5)

# HTTP client settings
TIMEOUT = 30.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# DOI prefixes that are open-access and can be fetched directly (no EZproxy needed)
DIRECT_OA_PREFIXES = (
    "10.1101/",  # bioRxiv
    "10.20944/",  # Preprints.org
    "10.2139/",  # SSRN
    "10.64898/",  # Reviewed preprints (eLife/bioRxiv)
    "10.7554/",  # eLife (fully OA)
    "10.1371/",  # PLOS (fully OA)
)

# Boilerplate phrases that indicate non-article content (navigation, headers, etc.)
BOILERPLATE_PHRASES = {
    "sign in",
    "log in",
    "create account",
    "subscribe now",
    "access denied",
    "cookies",
    "privacy policy",
    "terms of use",
    "terms and conditions",
    "skip to main content",
    "jump to navigation",
    "search this site",
    "follow us on twitter",
    "follow us on facebook",
    "advertisement",
    "supplementary material",
    "download pdf",
    "view article",
    "cite this article",
    "article metrics",
    "article history",
    "received:",
    "accepted:",
    "published:",
    "copyright",
    "all rights reserved",
    "open access",
    "creative commons",
}


# ---------------------------------------------------------------------------
# Publisher URL builders
# ---------------------------------------------------------------------------


def _doi_suffix(doi: str) -> str:
    """Return the suffix after the first slash in a DOI."""
    parts = doi.split("/", 1)
    return parts[1] if len(parts) > 1 else doi


def build_publisher_url(doi: str) -> str:
    """Construct the canonical publisher URL for a given DOI.

    Uses DOI prefix routing to known publisher URL patterns. Falls back
    to the doi.org redirect for unknown publishers.
    """
    doi = doi.strip().lower()
    suffix = _doi_suffix(doi)

    if doi.startswith("10.1038/"):
        return f"https://www.nature.com/articles/{suffix}"

    if doi.startswith("10.1007/"):
        return f"https://link.springer.com/article/{doi}"

    if doi.startswith("10.1016/"):
        # ScienceDirect requires PII; use doi.org redirect to reach the canonical page
        return f"https://doi.org/{doi}"

    if doi.startswith("10.1242/"):
        return f"https://journals.biologists.com/dev/article-lookup/doi/{doi}"

    if doi.startswith("10.1042/"):
        return f"https://portlandpress.com/biochemsoctrans/article-lookup/doi/{doi}"

    if doi.startswith("10.1093/"):
        return f"https://doi.org/{doi}"

    if doi.startswith("10.1002/"):
        return f"https://onlinelibrary.wiley.com/doi/{doi}"

    if doi.startswith("10.1101/"):
        return f"https://www.biorxiv.org/content/{doi}"

    if doi.startswith("10.3390/"):
        return f"https://doi.org/{doi}"

    if doi.startswith("10.3389/"):
        return f"https://www.frontiersin.org/articles/{doi}"

    if doi.startswith("10.1126/"):
        # AAAS (Science journal)
        return f"https://www.science.org/doi/{doi}"

    if doi.startswith("10.7554/"):
        # eLife — fully OA
        return f"https://elifesciences.org/articles/{suffix}"

    if doi.startswith("10.1371/"):
        # PLOS — fully OA
        return f"https://doi.org/{doi}"

    if doi.startswith("10.1073/"):
        # PNAS
        return f"https://www.pnas.org/doi/{doi}"

    if doi.startswith("10.1074/"):
        # Journal of Biological Chemistry
        return f"https://www.jbc.org/article/{suffix}/fulltext"

    if doi.startswith("10.1681/"):
        # JASN (Journal of the American Society of Nephrology)
        return f"https://jasn.asnjournals.org/content/{suffix}"

    if doi.startswith("10.1089/"):
        # Mary Ann Liebert publishers
        return f"https://www.liebertpub.com/doi/{doi}"

    if doi.startswith("10.1155/"):
        # Hindawi (now Wiley Open Access)
        return f"https://doi.org/{doi}"

    if doi.startswith("10.1096/"):
        # FASEB Journal
        return f"https://doi.org/{doi}"

    # Default: follow doi.org redirect
    return f"https://doi.org/{doi}"


def build_ezproxy_url(publisher_url: str) -> str:
    """Wrap a publisher URL in the Penn EZproxy gateway.

    EZproxy expects the target URL to be passed unencoded as the ``url``
    query parameter value.  URL-encoding (e.g. via ``quote_plus``) causes
    the proxy to fall back to its menu page.
    """
    return f"{EZPROXY_BASE}{publisher_url}"


# ---------------------------------------------------------------------------
# HTML text extractors (publisher-specific)
# ---------------------------------------------------------------------------


def _paragraphs_from_soup(soup: BeautifulSoup, min_chars: int = 50) -> list[str]:
    """Extract all <p> text longer than min_chars from a parsed page."""
    return [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) >= min_chars
    ]


def _is_boilerplate(text: str) -> bool:
    """Return True if the paragraph looks like navigation/UI boilerplate."""
    lower = text.lower()
    return any(phrase in lower for phrase in BOILERPLATE_PHRASES)


def _filter_boilerplate(paragraphs: list[str]) -> list[str]:
    """Remove obvious navigation/boilerplate paragraphs."""
    return [p for p in paragraphs if not _is_boilerplate(p)]


def _quality_check(text: str) -> bool:
    """Return True if the text passes basic quality checks.

    Rejects if:
    - Less than MIN_TEXT_LENGTH characters
    - More than 30% of lines are boilerplate
    """
    if len(text) < MIN_TEXT_LENGTH:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    boilerplate_count = sum(1 for line in lines if _is_boilerplate(line))
    boilerplate_ratio = boilerplate_count / len(lines)
    return boilerplate_ratio <= 0.30


def extract_nature_springer(soup: BeautifulSoup) -> str:
    """Extract article body from Nature/Springer pages."""
    paragraphs: list[str] = []

    # Nature: article-section content divs
    for section in soup.find_all("div", class_=re.compile(r"c-article-section__content")):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in section.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    # Springer: main article body
    if not paragraphs:
        for container in soup.find_all(
            "div", class_=re.compile(r"(article-body|ArticleBody|c-article-body)")
        ):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    # Fallback: generic <p> extraction
    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_elsevier(soup: BeautifulSoup) -> str:
    """Extract article body from ScienceDirect/Elsevier pages."""
    paragraphs: list[str] = []

    # ScienceDirect: div.Body sections
    body = soup.find("div", class_=re.compile(r"\bBody\b"))
    if body:
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in body.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    # ScienceDirect: sections with body-related classes
    if not paragraphs:
        for section in soup.find_all("section", class_=re.compile(r"body", re.I)):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in section.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    # Fallback
    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_company_of_biologists(soup: BeautifulSoup) -> str:
    """Extract article body from Company of Biologists (Development, etc.)."""
    paragraphs: list[str] = []

    for container in soup.find_all(
        "div", class_=re.compile(r"(article-body|article__body|section__body)", re.I)
    ):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    if not paragraphs:
        # HighWire-style: article sections
        for section in soup.find_all("div", class_=re.compile(r"section", re.I)):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in section.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_biorxiv(soup: BeautifulSoup) -> str:
    """Extract article body from bioRxiv HTML pages."""
    paragraphs: list[str] = []

    # bioRxiv uses div.article with section elements
    for container in soup.find_all("div", class_=re.compile(r"article|fulltext", re.I)):
        for section in container.find_all("div", class_=re.compile(r"section|abstract", re.I)):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in section.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_oxford_academic(soup: BeautifulSoup) -> str:
    """Extract article body from Oxford Academic (OUP) pages."""
    paragraphs: list[str] = []

    for container in soup.find_all(
        "div",
        class_=re.compile(r"(article-body|widget-ArticleFulltext|cf-article-body)", re.I),
    ):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_wiley(soup: BeautifulSoup) -> str:
    """Extract article body from Wiley Online Library pages."""
    paragraphs: list[str] = []

    for container in soup.find_all(
        "div",
        class_=re.compile(r"(article__body|article-section|hlFld-Abstract|hlFld-Fulltext)", re.I),
    ):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_science_aaas(soup: BeautifulSoup) -> str:
    """Extract article body from Science (AAAS) pages.

    Science.org uses JavaScript-heavy rendering; try multiple container
    classes and ARIA roles to maximise coverage.
    """
    paragraphs: list[str] = []

    # Primary: core-self-markup, article__body, bodySection
    for container in soup.find_all(
        "div",
        class_=re.compile(r"(core-self-markup|article__body|bodySection|article-content)", re.I),
    ):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    # ARIA role="doc-abstract" sections (abstract + body sections)
    if not paragraphs:
        for section in soup.find_all(attrs={"role": "doc-abstract"}):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in section.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_pnas(soup: BeautifulSoup) -> str:
    """Extract article body from PNAS pages.

    PNAS uses a HighWire/Silverchair layout similar to Science.
    """
    paragraphs: list[str] = []

    # Primary: core-self-markup, article__body, executive-summary
    for container in soup.find_all(
        "div",
        class_=re.compile(r"(core-self-markup|article__body|executive-summary|article-body)", re.I),
    ):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= 50
        )

    # ARIA roles
    if not paragraphs:
        for section in soup.find_all(attrs={"role": re.compile(r"(doc-abstract|main)", re.I)}):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in section.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_mdpi(soup: BeautifulSoup) -> str:
    """Extract article body from MDPI pages (open access)."""
    paragraphs: list[str] = []

    # MDPI html-body containers
    for container in soup.find_all("div", class_=re.compile(r"html-body", re.I)):
        paragraphs.extend(
            p.get_text(" ", strip=True)
            for p in container.find_all(["p", "div"], class_=re.compile(r"html-p", re.I))
            if len(p.get_text(strip=True)) >= 50
        )

    # Also try individual html-p paragraphs at the top level
    if not paragraphs:
        for p in soup.find_all("div", class_=re.compile(r"html-p", re.I)):
            text = p.get_text(" ", strip=True)
            if len(text) >= 50:
                paragraphs.append(text)

    if not paragraphs:
        paragraphs = _paragraphs_from_soup(soup)

    return "\n\n".join(_filter_boilerplate(paragraphs))


def extract_generic(soup: BeautifulSoup) -> str:
    """Generic fallback: extract <p> tags, skip first/last 20% (boilerplate zones)."""
    all_paragraphs = _paragraphs_from_soup(soup)
    if not all_paragraphs:
        return ""

    n = len(all_paragraphs)
    skip_n = max(1, int(n * 0.20))

    # Skip first and last 20% (nav, header, footer boilerplate)
    if n > 10:
        trimmed = all_paragraphs[skip_n : n - skip_n]
    else:
        trimmed = all_paragraphs

    return "\n\n".join(_filter_boilerplate(trimmed))


def extract_text(html: str, doi: str) -> str:
    """Route to the appropriate publisher-specific extractor based on DOI prefix."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, header, footer elements before extraction
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    doi_lower = doi.lower()

    if doi_lower.startswith("10.1038/") or doi_lower.startswith("10.1007/"):
        return extract_nature_springer(soup)
    if doi_lower.startswith("10.1016/"):
        return extract_elsevier(soup)
    if doi_lower.startswith("10.1242/") or doi_lower.startswith("10.1042/"):
        return extract_company_of_biologists(soup)
    if doi_lower.startswith("10.1101/"):
        return extract_biorxiv(soup)
    if doi_lower.startswith("10.1093/"):
        return extract_oxford_academic(soup)
    if doi_lower.startswith("10.1002/"):
        return extract_wiley(soup)
    if doi_lower.startswith("10.1126/"):
        return extract_science_aaas(soup)
    if doi_lower.startswith("10.1073/"):
        return extract_pnas(soup)
    if doi_lower.startswith("10.3390/"):
        return extract_mdpi(soup)

    return extract_generic(soup)


# ---------------------------------------------------------------------------
# Direct fetchers for open-access / preprint content (no EZproxy needed)
# ---------------------------------------------------------------------------


async def fetch_biorxiv_direct(client: httpx.AsyncClient, doi: str) -> str | None:
    """Fetch full text for a bioRxiv preprint via the TDM plain-text endpoint.

    Tries versioned (.full.txtv1) then unversioned (.full.txt) URLs.
    Returns plain text on success, None on failure.
    """
    doi_lower = doi.strip().lower()
    candidates = [
        f"https://www.biorxiv.org/content/{doi_lower}v1.full.txt",
        f"https://www.biorxiv.org/content/{doi_lower}.full.txt",
    ]
    # Also query the API to discover the latest version number
    try:
        api_url = f"https://api.biorxiv.org/details/biorxiv/{doi_lower}"
        api_resp = await client.get(api_url, timeout=15.0)
        if api_resp.status_code == 200:
            data = api_resp.json()
            collection = data.get("collection") or []
            if collection:
                version = collection[-1].get("version", "1")
                candidates.insert(
                    0, f"https://www.biorxiv.org/content/{doi_lower}v{version}.full.txt"
                )
    except Exception:
        pass  # API lookup is best-effort

    for url in candidates:
        try:
            resp = await client.get(url, timeout=TIMEOUT)
            if resp.status_code == 200 and len(resp.text) >= MIN_TEXT_LENGTH:
                logger.info("biorxiv.direct_success", doi=doi_lower, url=url[:80])
                return resp.text
        except httpx.RequestError as exc:
            logger.debug("biorxiv.direct_error", url=url[:80], error=str(exc))

    return None


async def fetch_reviewed_preprint_direct(client: httpx.AsyncClient, doi: str) -> str | None:
    """Fetch full text for a reviewed preprint (10.64898 DOI prefix).

    Reviewed preprints are co-hosted on bioRxiv/eLife. Resolve via doi.org
    to find the underlying bioRxiv DOI, then use the TDM API.
    """
    doi_lower = doi.strip().lower()
    try:
        # Follow doi.org redirect to find the actual landing page
        resp = await client.get(f"https://doi.org/{doi_lower}", timeout=TIMEOUT)
        final_url = str(resp.url)
        # Extract a bioRxiv DOI from the redirect target URL if present
        biorxiv_match = re.search(r"(10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+)", final_url)
        if biorxiv_match:
            biorxiv_doi = biorxiv_match.group(1)
            logger.info(
                "reviewed_preprint.resolved_biorxiv",
                original_doi=doi_lower,
                biorxiv_doi=biorxiv_doi,
            )
            return await fetch_biorxiv_direct(client, biorxiv_doi)

        # If no bioRxiv redirect, try parsing the landing page for content
        if resp.status_code == 200 and len(resp.text) >= MIN_TEXT_LENGTH:
            text = extract_text(resp.text, doi_lower)
            if text and _quality_check(text):
                return text
    except httpx.RequestError as exc:
        logger.debug("reviewed_preprint.error", doi=doi_lower, error=str(exc))

    return None


async def fetch_preprints_org_direct(client: httpx.AsyncClient, doi: str) -> str | None:
    """Fetch full text for a Preprints.org paper (10.20944 DOI prefix).

    Tries the direct download endpoint and the HTML version page.
    """
    doi_lower = doi.strip().lower()
    manuscript_id = doi_lower.split("/")[-1]
    candidates = [
        f"https://www.preprints.org/manuscript/{manuscript_id}/download",
        f"https://www.preprints.org/manuscript/{manuscript_id}/v1",
        f"https://doi.org/{doi_lower}",
    ]
    for url in candidates:
        try:
            resp = await client.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "")
            if "text/plain" in content_type:
                if len(resp.text) >= MIN_TEXT_LENGTH:
                    logger.info("preprints_org.direct_success", doi=doi_lower, url=url[:80])
                    return resp.text
            elif "text/html" in content_type:
                text = extract_text(resp.text, doi_lower)
                if text and _quality_check(text):
                    logger.info("preprints_org.html_success", doi=doi_lower, url=url[:80])
                    return text
        except httpx.RequestError as exc:
            logger.debug("preprints_org.error", url=url[:80], error=str(exc))

    return None


async def fetch_ssrn_direct(client: httpx.AsyncClient, doi: str) -> str | None:
    """Fetch abstract/body content for an SSRN paper (10.2139 DOI prefix).

    SSRN is primarily abstract-only without institutional access, but this
    captures whatever is publicly available.
    """
    doi_lower = doi.strip().lower()
    # The abstract_id is the last numeric segment of the DOI (e.g. 10.2139/ssrn.1234567)
    abstract_id = doi_lower.split(".")[-1]
    url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}"
    try:
        resp = await client.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        paragraphs: list[str] = []
        # SSRN abstract block
        for container in soup.find_all(
            "div", class_=re.compile(r"(abstract-text|abstract|ssrn-body)", re.I)
        ):
            paragraphs.extend(
                p.get_text(" ", strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) >= 50
            )
        if not paragraphs:
            paragraphs = _paragraphs_from_soup(soup)

        text = "\n\n".join(_filter_boilerplate(paragraphs))
        if text and _quality_check(text):
            logger.info("ssrn.direct_success", doi=doi_lower, url=url[:80])
            return text
    except httpx.RequestError as exc:
        logger.debug("ssrn.error", doi=doi_lower, error=str(exc))

    return None


async def fetch_direct_oa(client: httpx.AsyncClient, doi: str) -> str | None:
    """Dispatch to the correct direct-fetch function for open-access DOI prefixes.

    Returns extracted text on success, None if the DOI is not a recognised
    direct-OA prefix or if the fetch fails.
    """
    doi_lower = doi.strip().lower()
    if doi_lower.startswith("10.1101/"):
        return await fetch_biorxiv_direct(client, doi_lower)
    if doi_lower.startswith("10.64898/"):
        return await fetch_reviewed_preprint_direct(client, doi_lower)
    if doi_lower.startswith("10.20944/"):
        return await fetch_preprints_org_direct(client, doi_lower)
    if doi_lower.startswith("10.2139/"):
        return await fetch_ssrn_direct(client, doi_lower)
    return None


def _is_direct_oa(doi: str) -> bool:
    """Return True if this DOI should be fetched directly (no EZproxy needed)."""
    doi_lower = doi.strip().lower()
    return any(doi_lower.startswith(prefix) for prefix in DIRECT_OA_PREFIXES)


# ---------------------------------------------------------------------------
# EZproxy HTTP client
# ---------------------------------------------------------------------------


class EZproxyClient:
    """Async HTTP client that routes requests through Penn's EZproxy gateway."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> EZproxyClient:
        self._client = httpx.AsyncClient(
            headers=HEADERS,
            cookies=httpx.Cookies(),
            follow_redirects=True,
            timeout=TIMEOUT,
            verify=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def raw(self) -> httpx.AsyncClient:
        assert self._client is not None
        return self._client

    async def initialize_session(self) -> bool:
        """Hit the EZproxy login page to prime the session cookies.

        Returns True if the server is reachable, False on network error.
        """
        assert self._client is not None
        try:
            resp = await self._client.get(EZPROXY_LOGIN)
            logger.info(
                "ezproxy.session_init",
                status=resp.status_code,
                url=str(resp.url),
            )
            return resp.status_code < 500
        except httpx.RequestError as exc:
            logger.error("ezproxy.session_init_failed", error=str(exc))
            return False

    async def fetch_via_proxy(self, publisher_url: str) -> httpx.Response | None:
        """Fetch a publisher page through the EZproxy gateway.

        Returns the httpx.Response on success, None on network error.
        """
        assert self._client is not None
        proxy_url = build_ezproxy_url(publisher_url)
        try:
            resp = await self._client.get(proxy_url)
            logger.info(
                "ezproxy.fetch",
                proxy_url=proxy_url[:80],
                status=resp.status_code,
                content_length=len(resp.content),
            )
            return resp
        except httpx.RequestError as exc:
            logger.warning("ezproxy.fetch_error", url=proxy_url[:80], error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Per-paper retrieval logic
# ---------------------------------------------------------------------------


async def retrieve_paper(
    client: EZproxyClient,
    record: dict,
    dry_run: bool = False,
    direct_only: bool = False,
) -> tuple[CandidatePaper | None, str]:
    """Attempt to retrieve full text for a single inaccessible paper record.

    For preprint DOIs (bioRxiv, Preprints.org, SSRN, reviewed preprints),
    a direct OA fetch is tried first.  If it fails (or for non-preprint DOIs),
    falls back to EZproxy — unless ``direct_only`` is True.

    Args:
        client: Initialized EZproxyClient.
        record: Dict from papers_inaccessible.json (doi, title, year, journal, authors, ...).
        dry_run: If True, just log what would be fetched without making HTTP requests.
        direct_only: If True, skip EZproxy and only attempt direct OA fetches.

    Returns:
        Tuple of (CandidatePaper with full_text set, status_message).
        CandidatePaper is None if retrieval failed.
    """
    doi = (record.get("doi") or "").strip()
    title = record.get("title") or "Unknown"
    year = record.get("year")
    journal = record.get("journal")
    authors = record.get("authors") or []

    if not doi:
        return None, "no_doi"

    publisher_url = build_publisher_url(doi)

    if dry_run:
        logger.info(
            "ezproxy.dry_run",
            doi=doi,
            publisher_url=publisher_url,
            proxy_url=build_ezproxy_url(publisher_url)[:80],
            is_direct_oa=_is_direct_oa(doi),
        )
        return None, "dry_run"

    text: str | None = None
    fetch_source: str = "ezproxy_penn"

    # --- 1. Try direct OA fetch first for known preprint/OA prefixes ---
    if _is_direct_oa(doi):
        text = await fetch_direct_oa(client.raw, doi)
        if text and _quality_check(text):
            fetch_source = "direct_oa"
        else:
            text = None
            if direct_only:
                logger.info("retrieve.direct_oa_failed", doi=doi)
                return None, "direct_oa_failed"

    # --- 2. Fall back to EZproxy if direct fetch didn't yield text ---
    if text is None and not direct_only:
        resp = await client.fetch_via_proxy(publisher_url)

        if resp is None:
            return None, "network_error"

        if resp.status_code in (401, 403):
            logger.warning("ezproxy.access_denied", doi=doi, status=resp.status_code)
            return None, f"http_{resp.status_code}"

        if resp.status_code != 200:
            logger.warning("ezproxy.http_error", doi=doi, status=resp.status_code)
            return None, f"http_{resp.status_code}"

        html = resp.text
        extracted = extract_text(html, doi)
        if extracted and _quality_check(extracted):
            text = extracted

    if not text:
        logger.info(
            "retrieve.text_too_short",
            doi=doi,
            text_length=len(text) if text else 0,
        )
        return None, "text_too_short"

    logger.info(
        "retrieve.success", doi=doi, text_length=len(text), title=title[:60], source=fetch_source
    )

    paper = CandidatePaper(
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        abstract=None,  # not stored in inaccessible records
        source_database=fetch_source,
        full_text=text,
        full_text_source=fetch_source,
        citation_count=record.get("citation_count"),
        external_ids=record.get("external_ids") or {},
    )
    return paper, "success"


# ---------------------------------------------------------------------------
# Parallel fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_one(
    client: EZproxyClient,
    record: dict,
    existing_dois: set[str],
    dry_run: bool,
    direct_only: bool,
) -> tuple[CandidatePaper | None, str]:
    """Fetch a single paper under the global semaphore."""
    async with SEM:
        result = await retrieve_paper(client, record, dry_run=dry_run, direct_only=direct_only)
        if not dry_run:
            await asyncio.sleep(0.3)  # 300 ms between semaphore releases
        return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def main(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    direct_only: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load inaccessible papers
    if not INACCESSIBLE_PATH.exists():
        logger.error("inaccessible.not_found", path=str(INACCESSIBLE_PATH))
        sys.exit(1)

    inaccessible_records: list[dict] = json.loads(INACCESSIBLE_PATH.read_text())
    logger.info("inaccessible.loaded", count=len(inaccessible_records))

    if limit is not None:
        inaccessible_records = inaccessible_records[:limit]
        logger.info("inaccessible.limited", limit=limit)

    # Load existing corpus
    existing: list[CandidatePaper] = []
    if PAPERS_PATH.exists():
        raw = json.loads(PAPERS_PATH.read_text())
        existing = [CandidatePaper.model_validate(p) for p in raw]
    logger.info("existing.loaded", count=len(existing))

    # Build sets of existing DOIs and normalized titles to avoid duplicates
    # (catches both exact DOI matches and preprint/published pairs with same title)
    existing_dois: set[str] = set()
    existing_titles: set[str] = set()
    for p in existing:
        if p.doi:
            existing_dois.add(p.doi.lower().strip())
        if p.title:
            existing_titles.add(re.sub(r"[^a-z0-9]", "", p.title.lower()))

    # Partition into direct-OA and EZproxy-needed
    papers_direct: list[dict] = []
    papers_ezproxy: list[dict] = []
    papers_skipped_dup: list[dict] = []

    for record in inaccessible_records:
        doi = (record.get("doi") or "").strip().lower()
        title_norm = re.sub(r"[^a-z0-9]", "", (record.get("title") or "").lower())
        if (doi and doi in existing_dois) or (title_norm and title_norm in existing_titles):
            papers_skipped_dup.append(record)
            continue
        if _is_direct_oa(doi):
            papers_direct.append(record)
        else:
            papers_ezproxy.append(record)

    papers_to_fetch = papers_direct + ([] if direct_only else papers_ezproxy)

    logger.info(
        "fetch.plan",
        direct_oa=len(papers_direct),
        ezproxy=len(papers_ezproxy),
        skipped_duplicates=len(papers_skipped_dup),
        direct_only=direct_only,
        total_to_fetch=len(papers_to_fetch),
    )

    # Stats tracking
    stats: dict[str, int] = {
        "success": 0,
        "no_doi": 0,
        "already_in_corpus": len(papers_skipped_dup),
        "dry_run": 0,
        "network_error": 0,
        "text_too_short": 0,
        "http_error": 0,
        "direct_oa_failed": 0,
    }

    retrieved: list[CandidatePaper] = []

    async with EZproxyClient() as client:
        if not dry_run and not direct_only:
            ok = await client.initialize_session()
            if not ok:
                logger.error(
                    "ezproxy.unreachable",
                    hint="Are you connected to Penn VPN? (vpn.upenn.edu)",
                )
                sys.exit(1)

        # Dispatch all papers in parallel under the semaphore
        tasks = [
            _fetch_one(client, record, existing_dois, dry_run, direct_only)
            for record in papers_to_fetch
        ]
        results = await asyncio.gather(*tasks)

    for record, (paper, status) in zip(papers_to_fetch, results):
        doi = (record.get("doi") or "").strip().lower()
        if status == "success" and paper is not None:
            retrieved.append(paper)
            existing_dois.add(doi)
            stats["success"] += 1
        elif status == "no_doi":
            stats["no_doi"] += 1
        elif status == "dry_run":
            stats["dry_run"] += 1
        elif status == "network_error":
            stats["network_error"] += 1
        elif status in ("text_too_short", "direct_oa_failed"):
            stats["text_too_short"] += 1
        elif status.startswith("http_"):
            stats["http_error"] += 1
        else:
            stats[status] = stats.get(status, 0) + 1

    # Merge into corpus (skip if dry-run)
    if not dry_run and retrieved:
        merged = existing + retrieved
        PAPERS_PATH.write_text(
            json.dumps(
                [p.model_dump() for p in merged],
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        )
        logger.info(
            "corpus.merged",
            previous=len(existing),
            added=len(retrieved),
            total=len(merged),
            path=str(PAPERS_PATH),
        )

    # Save retrieved papers list separately for audit
    if not dry_run and retrieved:
        retrieved_path = OUTPUT_DIR / "ezproxy_retrieved.json"
        retrieved_path.write_text(
            json.dumps(
                [p.model_dump() for p in retrieved],
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        )
        logger.info("ezproxy.retrieved_saved", path=str(retrieved_path))

    # Summary
    total = len(existing) + len(retrieved)
    width = 70
    print(f"\n{'=' * width}")
    print("  EZproxy Full Text Retrieval — Penn Institutional Access")
    print(f"{'=' * width}")
    print(f"  Papers attempted:       {len(papers_to_fetch):>6}")
    print(f"  Already in corpus:      {stats['already_in_corpus']:>6}")
    print(f"  No DOI (skipped):       {stats['no_doi']:>6}")
    print(f"  Successfully retrieved: {stats['success']:>6}")
    print(f"  Text too short:         {stats['text_too_short']:>6}")
    print(f"  Network errors:         {stats['network_error']:>6}")
    print(f"  HTTP errors:            {stats['http_error']:>6}")
    if dry_run:
        print(f"\n  DRY RUN — {stats['dry_run']} proxy URLs logged, no changes made.")
    else:
        print(f"\n  Previous corpus size:   {len(existing):>6}")
        print(f"  Added this run:         {len(retrieved):>6}")
        print(f"  New corpus total:       {total:>6}")
        if retrieved:
            print("\n  Full text sources: direct_oa / ezproxy_penn")
            print(f"  Merged into:      {PAPERS_PATH}")
            print(f"  Retrieved list:   {OUTPUT_DIR / 'ezproxy_retrieved.json'}")
    if direct_only:
        print("\n  NOTE: --direct-only mode — EZproxy was not used.")
    print(f"{'=' * width}\n")

    if not dry_run and not direct_only and stats["network_error"] > 0:
        print(
            "  NOTE: Network errors detected. Ensure you are connected to\n"
            "  Penn VPN (vpn.upenn.edu) before running.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve full text for inaccessible papers via Penn EZproxy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log proxy URLs without making requests or modifying papers.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N papers (useful for testing)",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help=(
            "Only fetch direct OA content (bioRxiv, Preprints.org, SSRN, eLife). "
            "Skips EZproxy entirely — useful for testing without VPN."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            dry_run=args.dry_run,
            limit=args.limit,
            direct_only=args.direct_only,
        )
    )
