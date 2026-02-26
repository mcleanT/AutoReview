from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

PERPLEXITY_API_BASE = "https://api.perplexity.ai"

# Regex to extract a 4-digit year from a URL path segment (e.g. /2023/ or /2021.)
_URL_YEAR_RE = re.compile(r"/(\d{4})(?=[/.\-])")

# Regex to extract year from text (e.g. "(2023)" or ", 2021.")
_TEXT_YEAR_RE = re.compile(r"(?:^|[\s,(])(\d{4})(?=[)\s,.\-;]|$)")

# Regex to extract DOI from a URL (e.g. https://doi.org/10.1038/s41586-023-0001)
_DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s]+)", re.IGNORECASE)

# Regex to find a DOI anywhere in text
_DOI_TEXT_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"')\]]+)")

# Inline citation markers like [1], [2] in Perplexity response text
_CITATION_REF_RE = re.compile(r"\[(\d+)\]")


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a URL if present."""
    match = _DOI_URL_RE.search(url)
    if match:
        doi = match.group(1)
        # Strip trailing punctuation that may have been captured
        doi = doi.rstrip(".")
        return doi
    return None


def _extract_year_from_url(url: str) -> int | None:
    """Extract a plausible publication year from a URL path."""
    match = _URL_YEAR_RE.search(url)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2030:
            return year
    return None


def _extract_year_from_text(text: str) -> int | None:
    """Extract the most likely publication year from citation text."""
    matches = _TEXT_YEAR_RE.findall(text)
    # Filter to plausible publication years, prefer the most recent
    years = [int(y) for y in matches if 1900 <= int(y) <= 2030]
    return max(years) if years else None


def _extract_title_from_url(url: str) -> str | None:
    """Try to extract a meaningful title from a URL path.

    Many academic URLs contain slugified titles, e.g.:
    https://example.com/articles/gut-microbiome-and-neurodegeneration
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        return None

    # Get the last meaningful path segment
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None

    slug = segments[-1]

    # Skip segments that look like pure IDs (all digits, short hex, etc.)
    if re.match(r"^[\da-f\-]{5,40}$", slug, re.IGNORECASE):
        return None
    if slug.isdigit():
        return None
    # Skip very short segments or file extensions
    if len(slug) < 8:
        return None

    # Remove common file extensions
    slug = re.sub(r"\.(html?|pdf|xml|aspx|php)$", "", slug, flags=re.IGNORECASE)

    # Convert hyphens/underscores to spaces and title-case
    title = re.sub(r"[-_]+", " ", slug).strip()
    if len(title) < 8:
        return None

    # Title-case it for readability
    title = title.title()
    return title


def _build_citation_context(response_text: str) -> dict[int, str]:
    """Extract text snippets surrounding each citation reference [N].

    Returns a mapping from 1-based citation index to the sentence/context
    where that citation was referenced.
    """
    context: dict[int, str] = {}

    # Split into sentences (rough heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", response_text)
    for sentence in sentences:
        refs = _CITATION_REF_RE.findall(sentence)
        for ref in refs:
            idx = int(ref)
            # Clean up the sentence: remove citation markers for readability
            clean = _CITATION_REF_RE.sub("", sentence).strip()
            if clean and idx not in context:
                context[idx] = clean

    return context


def _parse_citation_string(
    citation: str,
    index: int,
    citation_context: dict[int, str],
) -> CandidatePaper | None:
    """Parse a citation string (URL) into a CandidatePaper.

    Returns None if no meaningful title can be extracted.
    """
    url = citation.strip()

    # Extract structured info from the URL
    doi = _extract_doi_from_url(url)
    year = _extract_year_from_url(url)

    # Try to get a title from the URL slug
    title = _extract_title_from_url(url)

    # Use the context snippet from the response as an abstract
    # Citation indices in Perplexity are 1-based
    abstract = citation_context.get(index + 1)

    # If we couldn't extract year from URL, try the context text
    if year is None and abstract:
        year = _extract_year_from_text(abstract)

    # If we still have no title, this citation is not useful
    if not title:
        return None

    external_ids: dict[str, str] = {"perplexity_url": url}

    return CandidatePaper(
        title=title,
        authors=[],
        year=year,
        doi=doi,
        abstract=abstract,
        source_database="perplexity",
        external_ids=external_ids,
    )


def _parse_citation_object(
    citation: dict[str, Any],
    index: int,
    citation_context: dict[int, str],
) -> CandidatePaper | None:
    """Parse a structured citation object into a CandidatePaper.

    Some Perplexity models return citation objects with metadata fields
    like title, author, url, published_date.
    """
    title = citation.get("title")
    url = citation.get("url", "")

    # If no title in the object, try to extract from URL
    if not title and url:
        title = _extract_title_from_url(url)

    if not title:
        return None

    # Extract authors
    authors: list[str] = []
    raw_authors = citation.get("author") or citation.get("authors")
    if isinstance(raw_authors, str):
        authors = [a.strip() for a in raw_authors.split(",") if a.strip()]
    elif isinstance(raw_authors, list):
        authors = [str(a).strip() for a in raw_authors if a]

    # Extract year
    year: int | None = None
    published = citation.get("published_date") or citation.get("year")
    if isinstance(published, int):
        year = published
    elif isinstance(published, str):
        year = _extract_year_from_text(published)
    if year is None and url:
        year = _extract_year_from_url(url)

    # Extract DOI
    doi = citation.get("doi")
    if not doi and url:
        doi = _extract_doi_from_url(url)

    # Abstract: prefer snippet from citation object, fall back to context
    abstract = citation.get("snippet") or citation.get("abstract")
    if not abstract:
        abstract = citation_context.get(index + 1)

    external_ids: dict[str, str] = {}
    if url:
        external_ids["perplexity_url"] = url

    return CandidatePaper(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        abstract=abstract,
        source_database="perplexity",
        external_ids=external_ids,
    )


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

                # Retry loop for transient errors (429, 5xx)
                max_retries = 3
                for attempt in range(max_retries):
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

                        if response.status_code == 429:
                            if attempt < max_retries - 1:
                                wait = 2 ** (attempt + 1)
                                logger.warning(
                                    "perplexity.rate_limited",
                                    query=query[:80], attempt=attempt + 1, wait=wait,
                                )
                                await asyncio.sleep(wait)
                                continue
                            else:
                                logger.error("perplexity.rate_limit_exhausted", query=query[:80])
                                break

                        if response.status_code >= 500:
                            if attempt < max_retries - 1:
                                wait = 2 ** (attempt + 1)
                                logger.warning(
                                    "perplexity.server_error_retry",
                                    status=response.status_code, attempt=attempt + 1, wait=wait,
                                )
                                await asyncio.sleep(wait)
                                continue

                        response.raise_for_status()
                        data = response.json()

                        # Extract the response text for citation context
                        response_text = ""
                        choices = data.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            response_text = message.get("content", "")

                        citation_context = _build_citation_context(response_text)

                        # Extract citations from the response
                        citations = data.get("citations", [])
                        parsed_count = 0
                        skipped_count = 0

                        for i, citation in enumerate(citations):
                            paper: CandidatePaper | None = None

                            if isinstance(citation, dict):
                                paper = _parse_citation_object(citation, i, citation_context)
                            elif isinstance(citation, str):
                                paper = _parse_citation_string(citation, i, citation_context)

                            if paper is not None:
                                papers.append(paper)
                                parsed_count += 1
                            else:
                                skipped_count += 1

                        logger.info(
                            "perplexity.search",
                            query=query[:80],
                            total_citations=len(citations),
                            parsed=parsed_count,
                            skipped=skipped_count,
                        )
                        break  # Success — exit retry loop

                    except httpx.HTTPStatusError as e:
                        logger.warning("perplexity.error", status=e.response.status_code, query=query[:80])
                        break  # Non-retryable HTTP error
                    except httpx.RequestError as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "perplexity.request_error_retry",
                                error=str(e), attempt=attempt + 1,
                            )
                            await asyncio.sleep(2 ** (attempt + 1))
                        else:
                            logger.warning("perplexity.request_error", error=str(e), query=query[:80])

        logger.info("perplexity.search.complete", total_papers=len(papers))
        return papers[:max_results]

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        return None  # Perplexity doesn't support direct paper lookup
