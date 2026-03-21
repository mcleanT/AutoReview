"""Multi-source full-text resolver.

Chains multiple strategies to maximise full-text coverage:

1. Elsevier ScienceDirect API  (DOI-based, requires ELSEVIER_API_KEY)
2. Wiley TDM API               (DOI-based, requires WILEY_TDM_TOKEN + institutional IP)
3. Semantic Scholar openAccessPdf  (stored during search)
3b. Semantic Scholar API lookup    (DOI-based, discovers OA PDFs & arXiv mirrors;
                                    optional S2_API_KEY for higher rate limits)
4. CORE API  (core_pdf_url or direct output download, 300M+ OA records)
5. PubMed Central  (PMID/PMCID -> JATS XML)
5b. Europe PMC  (PMCID -> JATS XML, superset of PMC with sometimes better availability)
6. arXiv / bioRxiv / medRxiv  (preprint PDF, also detects arXiv DOI prefix 10.48550/)
7. PLOS  (10.1371/ prefix, guaranteed OA)
8. MDPI  (10.3390/ prefix, guaranteed OA)
9. Frontiers  (10.3389/ prefix, guaranteed OA)
9b. ACL Anthology  (10.18653/ prefix, guaranteed OA)
9c. JMIR  (10.2196/ prefix, guaranteed OA)
10. Unpaywall  (DOI-based OA lookup)
11. Springer Nature Open Access API  (DOI-based, last resort)

Each strategy is tried in order; the first one that returns usable text wins.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.rate_limiter import RateLimiter

# Reuse the PDF / HTML text extraction helpers from the unpaywall module.
from autoreview.search.unpaywall import _extract_text_from_pdf

logger = structlog.get_logger()

# Transient HTTP status codes worth retrying
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_MAX_TEXT_CHARS = 200_000

# Elsevier DOI prefixes (covers ScienceDirect, Cell Press, Lancet, etc.)
_ELSEVIER_DOI_PREFIXES = (
    "10.1016/",  # ScienceDirect, Cell, Lancet, etc.
    "10.1006/",  # Academic Press (legacy Elsevier)
    "10.1053/",  # Elsevier subsidiary
)

# Wiley DOI prefixes (covers Wiley Online Library and Blackwell imprints)
_WILEY_DOI_PREFIXES = (
    "10.1002/",  # Wiley Online Library (main prefix)
    "10.1111/",  # Blackwell Publishing
    "10.1046/",  # Blackwell (legacy)
    "10.1113/",  # Physiological Society / Blackwell
    "10.1096/",  # Federation of American Societies for Experimental Biology (Wiley)
    "10.3322/",  # CA: A Cancer Journal for Clinicians (Wiley)
)

# Springer Nature DOI prefixes (covers Nature, Springer, BMC, Palgrave, etc.)
_SPRINGER_NATURE_DOI_PREFIXES = (
    "10.1038/",  # Nature Publishing Group
    "10.1007/",  # Springer journals & books
    "10.1186/",  # BioMed Central
    "10.1057/",  # Palgrave Macmillan
    "10.1140/",  # European Physical Journal
    "10.1365/",  # Springer Fachmedien
    "10.1251/",  # Springer Japan
)

# Open Access publisher DOI prefixes
_PLOS_DOI_PREFIX = "10.1371/"
_MDPI_DOI_PREFIX = "10.3390/"
_FRONTIERS_DOI_PREFIX = "10.3389/"
_ACL_DOI_PREFIX = "10.18653/"
_JMIR_DOI_PREFIX = "10.2196/"

# bioRxiv / medRxiv versions to try when API lookup fails
_BIORXIV_FALLBACK_VERSIONS = ("v1", "v2", "v3")

# ---------------------------------------------------------------------------
# Elsevier XML text extraction
# ---------------------------------------------------------------------------


def _extract_text_from_elsevier_xml(xml_text: str) -> str | None:
    """Extract article body text from Elsevier's full-text-retrieval XML."""
    try:
        from bs4 import BeautifulSoup

        # Strip DOCTYPE to prevent lxml from resolving external DTDs
        clean = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, count=1)

        for parser in ("lxml-xml", "xml", "html.parser"):
            try:
                soup = BeautifulSoup(clean, parser)
                break
            except Exception:
                continue
        else:
            return None

        # Elsevier XML uses <ce:sections> for the article body
        body = soup.find("ce:sections") or soup.find("body") or soup.find("rawtext")
        if body:
            text = body.get_text(separator="\n", strip=True)
            if text and len(text) > 100:
                return text[:_MAX_TEXT_CHARS]

        # Fallback: gather all <ce:para> tags
        paras = soup.find_all("ce:para")
        if paras:
            text = "\n".join(p.get_text(strip=True) for p in paras)
            if text and len(text) > 100:
                return text[:_MAX_TEXT_CHARS]

        return None
    except ImportError:
        logger.warning("full_text.bs4_not_installed")
        return None
    except Exception as e:
        logger.debug("full_text.elsevier_xml_extraction_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# JATS XML text extraction (for PubMed Central)
# ---------------------------------------------------------------------------


def _extract_text_from_jats_xml(xml_bytes: bytes) -> str | None:
    """Extract body text from JATS/NLM XML returned by PMC efetch."""
    try:
        from bs4 import BeautifulSoup

        # Strip DOCTYPE declarations that reference external DTDs.
        # lxml-xml tries to resolve external entities (e.g. JATS DTDs on
        # nlm.nih.gov), which either hangs, times out, or silently returns
        # an empty parse tree.  Removing the DOCTYPE is safe because we
        # only need element text, not entity definitions.
        clean = xml_bytes if isinstance(xml_bytes, bytes) else xml_bytes.encode("utf-8")
        clean = re.sub(rb"<!DOCTYPE[^>]*>", b"", clean, count=1)

        # Try lxml-xml first, fall back to html.parser for environments
        # where the lxml XML parser is not available.
        for parser in ("lxml-xml", "xml", "html.parser"):
            try:
                soup = BeautifulSoup(clean, parser)
                break
            except Exception:
                continue
        else:
            return None

        # Try <body> first (full article text)
        body = soup.find("body")
        if body:
            # Remove figures, tables, and their captions to keep text clean
            for tag in body.find_all(["fig", "table-wrap", "supplementary-material"]):
                tag.decompose()
            text = body.get_text(separator="\n", strip=True)
            if text and len(text) > 100:
                return text[:_MAX_TEXT_CHARS]

        # Fallback: extract all <p> tags from the article
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)
            if text and len(text) > 100:
                return text[:_MAX_TEXT_CHARS]

        return None
    except ImportError:
        logger.warning("full_text.bs4_or_lxml_not_installed")
        return None
    except Exception as e:
        logger.debug("full_text.jats_extraction_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# FullTextResolver
# ---------------------------------------------------------------------------


class FullTextResolver:
    """Resolve full text for screened papers using multiple sources.

    Strategies are tried in this order per paper:
      1. Elsevier ScienceDirect API          (DOI-based, structured XML)
      2. Wiley TDM API                       (DOI-based, requires token + institutional IP)
      3. Semantic Scholar openAccessPdf URL   (if stored in external_ids)
      3b. Semantic Scholar API lookup         (DOI-based live query; finds OA PDFs & arXiv mirrors
                                               for IEEE/ACM papers not covered by cached s2_pdf_url;
                                               optional S2_API_KEY for higher rate limits)
      4. CORE API                             (core_pdf_url or direct output download)
      4b. CrossRef links                      (PDF > XML > HTML from external_ids)
      5. PubMed Central via PMCID/PMID        (free JATS XML)
      5b. Europe PMC                          (PMCID -> JATS XML, PMC superset)
      6. arXiv PDF                             (if arXiv ID present)
      7. bioRxiv / medRxiv PDF                 (preprint DOI, with version API)
      8. PLOS direct PDF                       (10.1371/ prefix, guaranteed OA)
      9. MDPI direct PDF                       (10.3390/ prefix, guaranteed OA)
     10. Frontiers direct PDF                  (10.3389/ prefix, guaranteed OA)
     11. Unpaywall                             (DOI-based OA lookup)
     12. Springer Nature Open Access API      (DOI-based JATS XML, last resort)
    """

    def __init__(
        self,
        unpaywall_email: str | None = None,
        entrez_email: str | None = None,
        elsevier_api_key: str | None = None,
        wiley_tdm_token: str | None = None,
        springer_api_key: str | None = None,
        core_api_key: str | None = None,
        crossref_email: str | None = None,
        requests_per_second: float = 10.0,
    ) -> None:
        self._unpaywall_email = unpaywall_email or os.environ.get("UNPAYWALL_EMAIL")
        self._entrez_email = entrez_email or os.environ.get(
            "ENTREZ_EMAIL", "autoreview@example.com"
        )
        self._crossref_email = (
            crossref_email or os.environ.get("CROSSREF_EMAIL") or os.environ.get("ENTREZ_EMAIL")
        )
        self._elsevier_api_key = elsevier_api_key or os.environ.get("ELSEVIER_API_KEY")
        self._wiley_tdm_token = wiley_tdm_token or os.environ.get("WILEY_TDM_TOKEN")
        self._springer_api_key = springer_api_key or os.environ.get("SPRINGER_API_KEY")
        self._core_api_key = core_api_key or os.environ.get("CORE_API_KEY")
        _raw_s2 = os.environ.get("S2_API_KEY", "").strip()
        self._s2_api_key = _raw_s2 if _raw_s2 and _raw_s2.isascii() else None
        self._client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "AutoReview/1.0 (full-text-resolver)"},
        )
        self._limiter = RateLimiter(requests_per_second)

        # Per-API rate limiters reflecting documented limits.
        # Semantic Scholar: 1 req/s (free-tier API key).
        # Springer OA API: 500 calls/day ≈ 0.35 req/s.
        # NCBI/PubMed: 3 req/s without key, 10 req/s with key.
        _s2_rps = 1.0
        self._limiters: dict[str, RateLimiter] = {
            "semantic_scholar": RateLimiter(_s2_rps),
            "springer_oa": RateLimiter(0.35),
            "ncbi": RateLimiter(10.0 if os.environ.get("NCBI_API_KEY") else 3.0),
        }

        # Lazy-initialised Unpaywall client
        self._unpaywall: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self,
        papers: list[ScreenedPaper],
        max_concurrent: int = 5,
    ) -> dict[str, int]:
        """Enrich papers with full text from the best available source.

        Modifies papers in-place by setting ``paper.full_text`` and
        ``paper.full_text_source``.  Skips papers that already have
        ``full_text`` set, so this method is safe to call multiple times
        (e.g. a retry pass later in the pipeline for transient failures).

        Returns a dict of ``{source_name: count}`` for audit logging.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        source_counts: dict[str, int] = {}
        lock = asyncio.Lock()

        # Step 0: Enrich papers from OpenAlex/S2 with PMIDs/PMCIDs via DOI
        # lookup. This enables the PMC and Europe PMC strategies for papers
        # that don't already carry these IDs.
        await self._enrich_doi_to_pmid(papers)

        # Pre-compute batch PMID -> PMCID mapping for PMC strategy
        pmid_to_pmcid = await self._batch_pmid_to_pmcid(papers)

        async def _process(sp: ScreenedPaper) -> None:
            if sp.paper.full_text:
                return  # Already has full text

            async with semaphore:
                source, text = await self._try_all(sp.paper, pmid_to_pmcid)

            if text:
                sp.paper.full_text = text[:_MAX_TEXT_CHARS]
                sp.paper.full_text_source = source
                async with lock:
                    source_counts[source] = source_counts.get(source, 0) + 1

        tasks = [_process(sp) for sp in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any unexpected exceptions that bubbled up through gather
        for sp, result in zip(papers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "full_text.resolve_paper_exception",
                    doi=sp.paper.doi,
                    paper_id=sp.paper.id,
                    error=str(result),
                    exc_type=type(result).__name__,
                )

        logger.info(
            "full_text.resolve_complete",
            total=len(papers),
            enriched=sum(source_counts.values()),
            by_source=source_counts,
        )
        return source_counts

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        await self._client.aclose()
        if self._unpaywall is not None:
            await self._unpaywall.close()

    def _get_limiter(self, api_name: str) -> RateLimiter:
        """Return the per-API RateLimiter, falling back to the default limiter."""
        return self._limiters.get(api_name, self._limiter)

    # ------------------------------------------------------------------
    # Text validation helpers
    # ------------------------------------------------------------------

    def _is_paywall_or_error_text(self, text: str) -> bool:
        """Check if extracted text is actually a paywall page or error message.

        Only flags short texts (< 2000 chars) — long texts with these phrases
        embedded are likely real papers that merely discuss access issues.
        """
        if len(text) >= 2000:
            return False

        lower = text.lower()
        patterns = [
            "access denied",
            "please log in",
            "please sign in",
            "please login",
            "subscribe to",
            "subscription required",
            "purchase this article",
            "buy this article",
            "institutional login",
            "institutional access",
            "403 forbidden",
            "401 unauthorized",
            "you do not have access",
            "sign in to access",
        ]
        for pattern in patterns:
            if pattern in lower:
                return True

        # "This content is available" + "subscribers" in same short text
        return "this content is available" in lower and "subscribers" in lower

    def _validate_extracted_text(self, text: str, paper_title: str) -> bool:
        """Validate that extracted text is usable for analysis.

        Returns True when the text passes all quality checks, False otherwise.
        Checks performed:
        - Minimum length of 500 chars (100-char threshold is just metadata).
        - Not predominantly non-printable chars (binary/PDF extraction failure).
        - Not just a reference list (>60% of lines starting with a number or "[").
        """
        if len(text) < 500:
            return False

        # Non-ASCII / non-printable check: reject if >50% of chars are non-printable
        non_printable = sum(
            1 for ch in text if not ch.isprintable() and ch not in ("\n", "\r", "\t")
        )
        if non_printable / len(text) > 0.50:
            logger.debug(
                "full_text.text_quality_non_printable",
                paper_title=paper_title[:60],
                ratio=round(non_printable / len(text), 2),
            )
            return False

        # Reference-list heuristic: skip texts where >60% of lines look like citations
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            ref_lines = sum(1 for ln in lines if ln[:1].isdigit() or ln.startswith("["))
            if ref_lines / len(lines) > 0.60:
                logger.debug(
                    "full_text.text_quality_mostly_references",
                    paper_title=paper_title[:60],
                    ratio=round(ref_lines / len(lines), 2),
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Strategy chain
    # ------------------------------------------------------------------

    async def _try_all(
        self,
        paper: CandidatePaper,
        pmid_to_pmcid: dict[str, str],
    ) -> tuple[str, str | None]:
        """Try each strategy in priority order.  Return (source, text)."""
        strategies = [
            ("elsevier_api", self._try_elsevier_api),
            ("wiley_tdm", self._try_wiley_tdm),
            ("s2_pdf", self._try_s2_pdf),
            ("s2_api_lookup", self._try_s2_api_lookup),
            # CORE has 300M+ OA records — high hit rate, try before PMC
            ("core", self._try_core),
            # CrossRef links stored during search (PDF, XML, HTML)
            ("crossref", self._try_crossref),
            ("pmc", lambda p: self._try_pmc(p, pmid_to_pmcid)),
            ("europe_pmc", self._try_europe_pmc),
            ("arxiv", self._try_arxiv),
            ("biorxiv", self._try_biorxiv),
            ("plos", self._try_plos),
            ("mdpi", self._try_mdpi),
            ("frontiers", self._try_frontiers),
            ("acl_anthology", self._try_acl_anthology),
            ("jmir", self._try_jmir),
            ("unpaywall", self._try_unpaywall),
            # Rate-limited (500 calls/day) — last resort for OA articles
            ("springer_oa", self._try_springer_oa),
            # Institutional access — requires VPN/campus IP, last resort
            ("springer_inst", self._try_springer_institutional),
        ]

        attempted: list[str] = []
        skipped: list[str] = []

        for name, fn in strategies:
            try:
                text = await fn(paper)
                if text is None:
                    # None return means the strategy was skipped (wrong prefix,
                    # missing key, etc.) — distinguish from an attempted failure
                    skipped.append(name)
                    logger.debug(
                        "full_text.strategy_skipped",
                        strategy=name,
                        doi=paper.doi,
                        paper_id=paper.id,
                    )
                    continue
                attempted.append(name)
                if len(text) > 100:
                    # Fix 2: Paywall / error-page detection
                    if self._is_paywall_or_error_text(text):
                        logger.warning(
                            "full_text.strategy_paywall_detected",
                            strategy=name,
                            doi=paper.doi,
                            paper_id=paper.id,
                            chars=len(text),
                        )
                        continue
                    # Fix 4: Text quality validation
                    if not self._validate_extracted_text(text, paper.title or ""):
                        logger.warning(
                            "full_text.strategy_text_quality_failed",
                            strategy=name,
                            doi=paper.doi,
                            paper_id=paper.id,
                            chars=len(text),
                        )
                        continue
                    logger.info(
                        "full_text.strategy_succeeded",
                        strategy=name,
                        doi=paper.doi,
                        paper_id=paper.id,
                        chars=len(text),
                    )
                    logger.debug(
                        "full_text.try_all_summary",
                        doi=paper.doi,
                        attempted=attempted,
                        skipped=skipped,
                        succeeded=name,
                    )
                    return name, text
                # Returned something but too short — count as failed attempt
                logger.debug(
                    "full_text.strategy_text_too_short",
                    strategy=name,
                    doi=paper.doi,
                    paper_id=paper.id,
                    chars=len(text),
                )
            except Exception as e:
                attempted.append(name)
                logger.debug(
                    "full_text.strategy_failed",
                    strategy=name,
                    doi=paper.doi,
                    paper_id=paper.id,
                    error=str(e),
                )

        logger.warning(
            "full_text.all_strategies_failed",
            doi=paper.doi,
            paper_id=paper.id,
            attempted=attempted,
            skipped=skipped,
        )
        return "", None

    # ------------------------------------------------------------------
    # HTTP helper with retry for transient failures
    # ------------------------------------------------------------------

    async def _http_get_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        max_retries: int = 2,
        backoff_base: float = 2.0,
    ) -> httpx.Response:
        """GET with exponential backoff on transient HTTP errors.

        Retries on status codes 429, 500, 502, 503, 504, and on
        connection/timeout errors.  Returns the last response (even if
        it was an error) so callers can inspect status_code as usual.

        ``max_retries=2`` means up to 3 total attempts (1 initial + 2 retries).
        Backoff sleeps: 2s, 4s (with ``backoff_base=2.0``).
        """
        resp: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            await self._limiter.acquire()
            try:
                resp = await self._client.get(url, headers=headers)
                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    return resp
                # Retryable status — back off and try again
                if attempt < max_retries:
                    delay = backoff_base * (2**attempt)
                    logger.debug(
                        "full_text.http_retry",
                        url=url[:80],
                        status=resp.status_code,
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    return resp  # Last attempt — return whatever we got
            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt < max_retries:
                    delay = backoff_base * (2**attempt)
                    logger.debug(
                        "full_text.http_retry_error",
                        url=url[:80],
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        # Last attempt always returns or raises; this is unreachable
        assert resp is not None
        return resp

    # ------------------------------------------------------------------
    # Strategy 1: Elsevier ScienceDirect API
    # ------------------------------------------------------------------

    async def _try_elsevier_api(self, paper: CandidatePaper) -> str | None:
        """Fetch full text from Elsevier's ScienceDirect API.

        Requires an API key (ELSEVIER_API_KEY env var). Institutional access
        is granted via IP-based authentication (e.g. university VPN).
        """
        if not self._elsevier_api_key or not paper.doi:
            return None
        if not paper.doi.startswith(_ELSEVIER_DOI_PREFIXES):
            return None

        try:
            # Try plain text first (cleanest output, no parsing needed)
            resp = await self._http_get_with_retry(
                f"https://api.elsevier.com/content/article/doi/{paper.doi}",
                headers={
                    "X-ELS-APIKey": self._elsevier_api_key,
                    "Accept": "text/plain",
                },
            )
            if resp.status_code == 200 and len(resp.text) > 500:
                logger.debug(
                    "full_text.elsevier_api_success",
                    paper_id=paper.id,
                    chars=len(resp.text),
                    format="text/plain",
                )
                return resp.text[:_MAX_TEXT_CHARS]

            # Fallback: structured XML (richer but needs parsing)
            resp = await self._http_get_with_retry(
                f"https://api.elsevier.com/content/article/doi/{paper.doi}",
                headers={
                    "X-ELS-APIKey": self._elsevier_api_key,
                    "Accept": "text/xml",
                },
            )
            if resp.status_code == 200 and len(resp.text) > 500:
                text = _extract_text_from_elsevier_xml(resp.text)
                if text:
                    logger.debug(
                        "full_text.elsevier_api_success",
                        paper_id=paper.id,
                        chars=len(text),
                        format="text/xml",
                    )
                    return text

            if resp.status_code in (401, 403):
                logger.debug(
                    "full_text.elsevier_api_auth_failed",
                    paper_id=paper.id,
                    status=resp.status_code,
                )
            return None
        except Exception as e:
            logger.debug(
                "full_text.elsevier_api_failed",
                paper_id=paper.id,
                error=str(e),
            )
            return None

    # ------------------------------------------------------------------
    # Strategy 2: Wiley TDM API
    # ------------------------------------------------------------------

    async def _try_wiley_tdm(self, paper: CandidatePaper) -> str | None:
        """Fetch full text from Wiley's TDM API.

        Requires WILEY_TDM_TOKEN env var and institutional IP access.
        Returns PDF content converted to text via PyMuPDF.
        """
        if not self._wiley_tdm_token or not paper.doi:
            return None
        if not paper.doi.startswith(_WILEY_DOI_PREFIXES):
            return None

        from urllib.parse import quote as url_quote

        encoded_doi = url_quote(paper.doi, safe="")
        url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{encoded_doi}"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(
                url,
                headers={
                    "Wiley-TDM-Client-Token": self._wiley_tdm_token,
                    "User-Agent": "TDMClient/1.0.0 AutoReview/1.0",
                },
            )
            if resp.status_code == 200 and len(resp.content) >= 1000:
                text = _extract_text_from_pdf(resp.content)
                if text:
                    logger.debug(
                        "full_text.wiley_tdm_success",
                        paper_id=paper.id,
                        doi=paper.doi,
                        chars=len(text),
                    )
                return text

            if resp.status_code in (401, 403):
                logger.debug(
                    "full_text.wiley_tdm_auth_failed",
                    paper_id=paper.id,
                    doi=paper.doi,
                    status=resp.status_code,
                )
            return None
        except Exception as e:
            logger.debug(
                "full_text.wiley_tdm_failed",
                paper_id=paper.id,
                doi=paper.doi,
                error=str(e),
            )
            return None

    # ------------------------------------------------------------------
    # Strategy 3: Springer Nature Open Access API (JATS XML)
    # ------------------------------------------------------------------

    async def _try_springer_oa(self, paper: CandidatePaper) -> str | None:
        """Fetch full text from Springer Nature's Open Access API.

        Returns JATS XML for OA articles published by Nature, Springer,
        BioMed Central, and other Springer Nature imprints (~460K articles).
        Requires SPRINGER_API_KEY env var.
        """
        if not self._springer_api_key or not paper.doi:
            return None
        if not paper.doi.startswith(_SPRINGER_NATURE_DOI_PREFIXES):
            return None

        await self._get_limiter("springer_oa").acquire()
        try:
            resp = await self._client.get(
                "https://api.springernature.com/openaccess/jats",
                params={
                    "q": f"doi:{paper.doi}",
                    "api_key": self._springer_api_key,
                },
            )
            if resp.status_code != 200:
                if resp.status_code in (401, 403):
                    logger.debug(
                        "full_text.springer_oa_auth_failed",
                        paper_id=paper.id,
                        status=resp.status_code,
                    )
                return None

            # Only process if response contains article body
            if "<body>" not in resp.text:
                return None

            text = _extract_text_from_jats_xml(resp.content)
            if text:
                logger.debug(
                    "full_text.springer_oa_success",
                    paper_id=paper.id,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug(
                "full_text.springer_oa_failed",
                paper_id=paper.id,
                error=str(e),
            )
            return None

    # ------------------------------------------------------------------
    # Strategy: Springer/Nature institutional PDF (requires VPN/campus IP)
    # ------------------------------------------------------------------

    async def _try_springer_institutional(self, paper: CandidatePaper) -> str | None:
        """Fetch PDF from link.springer.com or nature.com via institutional access.

        Springer/Nature expose predictable PDF URLs that resolve when the
        request originates from a subscribing institution's IP range.
        No API key needed — authentication is IP-based.

        This is a last-resort strategy placed after Springer OA so it only
        fires for subscription-only content that earlier strategies missed.
        """
        doi = paper.doi
        if not doi:
            return None
        if not doi.startswith(_SPRINGER_NATURE_DOI_PREFIXES):
            return None

        # Nature papers (10.1038/) use nature.com; others use link.springer.com
        if doi.startswith("10.1038/"):
            pdf_url = f"https://www.nature.com/articles/{doi}.pdf"
        else:
            pdf_url = f"https://link.springer.com/content/pdf/{doi}.pdf"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                if resp.status_code in (401, 403):
                    logger.debug(
                        "full_text.springer_inst_no_access",
                        paper_id=paper.id,
                        doi=doi,
                        status=resp.status_code,
                    )
                return None

            # Verify we got a PDF, not an HTML login page
            if resp.content[:4] != b"%PDF":
                logger.debug(
                    "full_text.springer_inst_not_pdf",
                    paper_id=paper.id,
                    doi=doi,
                    content_type=resp.headers.get("content-type", "?"),
                )
                return None

            text = _extract_text_from_pdf(resp.content)
            if text:
                logger.debug(
                    "full_text.springer_inst_success",
                    paper_id=paper.id,
                    doi=doi,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug(
                "full_text.springer_inst_failed",
                paper_id=paper.id,
                doi=doi,
                error=str(e),
            )
            return None

    # ------------------------------------------------------------------
    # Strategy 3: Semantic Scholar openAccessPdf
    # ------------------------------------------------------------------

    async def _try_s2_pdf(self, paper: CandidatePaper) -> str | None:
        url = paper.external_ids.get("s2_pdf_url")
        if not url:
            return None

        await self._get_limiter("semantic_scholar").acquire()
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                return None
            return _extract_text_from_pdf(resp.content)
        except Exception as e:
            logger.debug("full_text.s2_pdf_failed", paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 3b: Semantic Scholar API lookup (DOI-based)
    # ------------------------------------------------------------------

    async def _try_s2_api_lookup(self, paper: CandidatePaper) -> str | None:
        """Query Semantic Scholar API by DOI to discover OA PDF or arXiv mirror.

        Works without an API key (at lower rate limits). Discovers OA copies
        of paywalled papers, especially IEEE/ACM conference papers with arXiv
        mirrors. Set S2_API_KEY env var for higher rate limits (~10 req/s vs
        ~0.5 req/s anonymous).

        Strategy A: use ``openAccessPdf.url`` returned by the S2 API.
        Strategy B: construct ``https://arxiv.org/pdf/{arxiv_id}`` from
                    ``externalIds.ArXiv`` if no OA PDF was found.
        """
        if not paper.doi:
            return None

        try:
            headers: dict[str, str] = {"User-Agent": "AutoReview/1.0"}
            if self._s2_api_key:
                headers["x-api-key"] = self._s2_api_key

            resp = await self._http_get_with_retry(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper.doi}"
                "?fields=openAccessPdf,externalIds",
                headers=headers,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()

            # Strategy A: Use openAccessPdf URL if available
            oa_pdf = data.get("openAccessPdf")
            if oa_pdf and oa_pdf.get("url"):
                pdf_url = oa_pdf["url"]
                await self._get_limiter("semantic_scholar").acquire()
                pdf_resp = await self._client.get(pdf_url)
                if pdf_resp.status_code == 200 and len(pdf_resp.content) >= 1000:
                    text = _extract_text_from_pdf(pdf_resp.content)
                    if text:
                        logger.debug(
                            "full_text.s2_api_oa_pdf_success",
                            doi=paper.doi,
                            paper_id=paper.id,
                            url=pdf_url,
                            chars=len(text),
                        )
                        return text

            # Strategy B: Use ArXiv ID if available
            ext_ids = data.get("externalIds", {})
            arxiv_id = ext_ids.get("ArXiv")
            if arxiv_id:
                clean_id = re.sub(r"v\d+$", "", arxiv_id)
                arxiv_url = f"https://arxiv.org/pdf/{clean_id}"
                await self._get_limiter("semantic_scholar").acquire()
                arxiv_resp = await self._client.get(arxiv_url)
                if arxiv_resp.status_code == 200 and len(arxiv_resp.content) >= 1000:
                    text = _extract_text_from_pdf(arxiv_resp.content)
                    if text:
                        logger.debug(
                            "full_text.s2_api_arxiv_success",
                            doi=paper.doi,
                            paper_id=paper.id,
                            arxiv_id=arxiv_id,
                            chars=len(text),
                        )
                        return text

            return None
        except Exception as e:
            logger.debug(
                "full_text.s2_api_lookup_failed",
                doi=paper.doi,
                paper_id=paper.id,
                error=str(e),
            )
            return None

    # ------------------------------------------------------------------
    # Strategy 3: CORE API (core_pdf_url or direct output download)
    # ------------------------------------------------------------------

    async def _try_core(self, paper: CandidatePaper) -> str | None:
        """Fetch full-text PDF from CORE (core.ac.uk).

        Tries two sources in order:
          1. ``core_pdf_url`` stored in external_ids during search (direct link).
          2. CORE API output download endpoint using the ``core`` ID.

        Requires CORE_API_KEY for the second strategy (used in Authorization header).
        """
        # Strategy A: use the download URL cached during search
        pdf_url = paper.external_ids.get("core_pdf_url")
        if pdf_url:
            await self._limiter.acquire()
            try:
                headers: dict[str, str] = {}
                if self._core_api_key:
                    headers["Authorization"] = f"Bearer {self._core_api_key}"
                resp = await self._client.get(pdf_url, headers=headers)
                if resp.status_code == 200 and len(resp.content) >= 1000:
                    text = _extract_text_from_pdf(resp.content)
                    if text:
                        logger.debug(
                            "full_text.core_pdf_url_success",
                            paper_id=paper.id,
                            chars=len(text),
                        )
                        return text
            except Exception as e:
                logger.debug("full_text.core_pdf_url_failed", paper_id=paper.id, error=str(e))

        # Strategy B: use the CORE output download endpoint
        core_id = paper.external_ids.get("core")
        if core_id and self._core_api_key:
            await self._limiter.acquire()
            try:
                resp = await self._client.get(
                    f"https://api.core.ac.uk/v3/outputs/{core_id}/download",
                    headers={"Authorization": f"Bearer {self._core_api_key}"},
                )
                if resp.status_code == 200 and len(resp.content) >= 1000:
                    text = _extract_text_from_pdf(resp.content)
                    if text:
                        logger.debug(
                            "full_text.core_download_success",
                            paper_id=paper.id,
                            core_id=core_id,
                            chars=len(text),
                        )
                        return text
            except Exception as e:
                logger.debug(
                    "full_text.core_download_failed",
                    paper_id=paper.id,
                    core_id=core_id,
                    error=str(e),
                )

        return None

    # ------------------------------------------------------------------
    # Strategy 3b: CrossRef links (PDF > XML > HTML)
    # ------------------------------------------------------------------

    async def _try_crossref(self, paper: CandidatePaper) -> str | None:
        """Fetch full text using link URLs stored from CrossRef search results.

        CrossRef ``link`` objects carry ``content-type`` and ``URL`` fields.
        We try link types in priority order: PDF → XML (JATS) → HTML.
        Returns None (skip) if no crossref_links are stored for this paper.
        """
        raw = paper.external_ids.get("crossref_links")
        if not raw:
            return None

        try:
            links: list[dict[str, str]] = json.loads(raw)
        except Exception:
            return None

        if not links:
            return None

        def _priority(link: dict[str, str]) -> int:
            ct = link.get("content-type", "")
            if ct == "application/pdf":
                return 0
            if ct in ("text/xml", "application/xml"):
                return 1
            if ct == "text/html":
                return 2
            return 3

        sorted_links = sorted(links, key=_priority)

        for link in sorted_links:
            url = link.get("URL")
            content_type = link.get("content-type", "")
            if not url:
                continue
            # Skip unknown content types entirely
            if _priority(link) == 3:
                continue

            try:
                # Fix 3: use retry helper so transient failures are retried
                resp = await self._http_get_with_retry(
                    url,
                    headers={
                        "User-Agent": (
                            f"AutoReview/1.0 (mailto:{self._crossref_email or 'anonymous'})"
                        )
                    },
                )
                min_size = 1000 if content_type == "application/pdf" else 100
                if resp.status_code != 200 or len(resp.content) < min_size:
                    continue

                if content_type == "application/pdf":
                    text = _extract_text_from_pdf(resp.content)
                    if text and len(text) > 100:
                        logger.debug(
                            "full_text.crossref_pdf_success",
                            paper_id=paper.id,
                            url=url,
                            chars=len(text),
                        )
                        return text

                elif content_type in ("text/xml", "application/xml"):
                    text = _extract_text_from_jats_xml(resp.content)
                    if text and len(text) > 100:
                        logger.debug(
                            "full_text.crossref_xml_success",
                            paper_id=paper.id,
                            url=url,
                            chars=len(text),
                        )
                        return text

                elif content_type == "text/html":
                    html_text = resp.text
                    if len(html_text) > 100:
                        logger.debug(
                            "full_text.crossref_html_success",
                            paper_id=paper.id,
                            url=url,
                            chars=len(html_text),
                        )
                        return html_text[:_MAX_TEXT_CHARS]

            except Exception as e:
                logger.debug(
                    "full_text.crossref_link_failed",
                    paper_id=paper.id,
                    url=url,
                    error=str(e),
                )
                continue

        return None

    # ------------------------------------------------------------------
    # Strategy 4: PubMed Central (JATS XML)
    # ------------------------------------------------------------------

    async def _enrich_doi_to_pmid(
        self,
        papers: list[ScreenedPaper],
    ) -> None:
        """Bulk-enrich papers with PMID/PMCID via the NCBI ID Converter API.

        Papers from OpenAlex and Semantic Scholar often lack PMID/PMCID even
        when the article is indexed in PubMed Central.  This step resolves
        DOIs to PMIDs and PMCIDs in batches of 200, then stores the results in
        ``paper.external_ids`` so that the downstream ``_try_pmc`` and
        ``_try_europe_pmc`` strategies can pick them up.

        Only processes papers that:
          - Have a DOI
          - Do NOT already have a ``pmid`` in ``external_ids``
          - Have not already been enriched with full text
        """
        doi_to_paper: dict[str, CandidatePaper] = {}
        for sp in papers:
            p = sp.paper
            if p.full_text:
                continue
            if p.external_ids.get("pmid"):
                continue  # Already has PMID — skip
            if not p.doi:
                continue
            doi_to_paper[p.doi] = p

        if not doi_to_paper:
            return

        dois = list(doi_to_paper.keys())
        enriched = 0
        base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

        for i in range(0, len(dois), 200):
            batch = dois[i : i + 200]
            try:
                resp = await self._http_get_with_retry(
                    f"{base_url}?tool=autoreview"
                    f"&email={self._entrez_email}"
                    f"&ids={','.join(batch)}"
                    f"&format=json",
                )
                if resp.status_code != 200:
                    logger.debug(
                        "full_text.doi_to_pmid_batch_failed",
                        status=resp.status_code,
                        batch_start=i,
                    )
                    continue
                data = resp.json()
                for record in data.get("records", []):
                    # The converter echoes the input DOI in the ``doi`` field.
                    # Values may be str or int, so coerce to str first.
                    doi = str(record.get("doi") or "").strip()
                    pmid = str(record.get("pmid") or "").strip()
                    pmcid = str(record.get("pmcid") or "").strip()
                    if not doi or doi not in doi_to_paper:
                        continue
                    paper = doi_to_paper[doi]
                    if pmid:
                        paper.external_ids["pmid"] = pmid
                        enriched += 1
                    if pmcid:
                        paper.external_ids["pmcid"] = pmcid
            except Exception as e:
                logger.debug(
                    "full_text.doi_to_pmid_batch_error",
                    batch_start=i,
                    error=str(e),
                )

        logger.info(
            "full_text.doi_to_pmid_enrichment",
            candidates=len(doi_to_paper),
            enriched=enriched,
        )

    async def _batch_pmid_to_pmcid(
        self,
        papers: list[ScreenedPaper],
    ) -> dict[str, str]:
        """Batch-convert PMIDs to PMCIDs using the NCBI ID converter.

        Papers that already carry a ``pmcid`` in external_ids are included
        directly without an API call.
        """
        result: dict[str, str] = {}
        pmids_to_lookup: list[str] = []

        for sp in papers:
            if sp.paper.full_text:
                continue  # skip already-enriched papers
            pmcid = sp.paper.external_ids.get("pmcid")
            pmid = sp.paper.external_ids.get("pmid")
            if pmcid and pmid:
                result[pmid] = pmcid
            elif pmid:
                pmids_to_lookup.append(pmid)

        if not pmids_to_lookup:
            return result

        # NCBI ID converter accepts up to 200 IDs per request
        base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        for i in range(0, len(pmids_to_lookup), 200):
            batch = pmids_to_lookup[i : i + 200]
            await self._get_limiter("ncbi").acquire()
            try:
                resp = await self._client.get(
                    base_url,
                    params={
                        "ids": ",".join(batch),
                        "format": "json",
                        "email": self._entrez_email,
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for record in data.get("records", []):
                    pmid = record.get("pmid", "")
                    pmcid = record.get("pmcid", "")
                    if pmid and pmcid:
                        result[pmid] = pmcid
            except Exception as e:
                logger.debug("full_text.pmid_convert_failed", error=str(e))

        logger.info(
            "full_text.pmid_to_pmcid",
            looked_up=len(pmids_to_lookup),
            found=len(result),
        )
        return result

    async def _try_pmc(
        self,
        paper: CandidatePaper,
        pmid_to_pmcid: dict[str, str],
    ) -> str | None:
        pmcid = paper.external_ids.get("pmcid")
        if not pmcid:
            pmid = paper.external_ids.get("pmid")
            if pmid:
                pmcid = pmid_to_pmcid.get(pmid)
        if not pmcid:
            return None

        # Normalise: ensure it starts with "PMC"
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        await self._get_limiter("ncbi").acquire()
        try:
            resp = await self._client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
            )
            if resp.status_code != 200:
                return None
            return _extract_text_from_jats_xml(resp.content)
        except Exception as e:
            logger.debug("full_text.pmc_failed", paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 4b: Europe PMC full-text XML (JATS)
    # ------------------------------------------------------------------

    async def _try_europe_pmc(self, paper: CandidatePaper) -> str | None:
        """Fetch full-text JATS XML from Europe PMC.

        Tries two paths in order:

        1. PMCID-based full-text XML endpoint (direct, fastest).
        2. DOI-based search API — used when no PMCID is available.  Europe PMC
           indexes a superset of PubMed Central and the search API returns a
           PMCID that we can then fetch.  Only attempted when a DOI is present.

        No API key required.
        """
        pmcid = paper.external_ids.get("pmcid")

        if pmcid:
            # Path 1: PMCID known — fetch full-text XML directly
            if not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            await self._limiter.acquire()
            try:
                resp = await self._client.get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                )
                if resp.status_code == 200:
                    text = _extract_text_from_jats_xml(resp.content)
                    if text:
                        return text
            except Exception as e:
                logger.debug(
                    "full_text.europe_pmc_pmcid_failed",
                    paper_id=paper.id,
                    pmcid=pmcid,
                    error=str(e),
                )

        # Path 2: DOI-based search — discover PMCID then fetch full text
        if not paper.doi:
            return None

        await self._limiter.acquire()
        try:
            resp = await self._client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": f"DOI:{paper.doi}",
                    "format": "json",
                    "resultType": "core",
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                return None
            hit = results[0]
            found_pmcid: str = hit.get("pmcid", "")
            if not found_pmcid:
                return None
            # Store it for future pipeline steps
            paper.external_ids["pmcid"] = found_pmcid
            logger.debug(
                "full_text.europe_pmc_doi_lookup_hit",
                paper_id=paper.id,
                doi=paper.doi,
                pmcid=found_pmcid,
            )
        except Exception as e:
            logger.debug(
                "full_text.europe_pmc_doi_search_failed",
                paper_id=paper.id,
                doi=paper.doi,
                error=str(e),
            )
            return None

        # Fetch the full-text XML using the discovered PMCID
        if not found_pmcid.upper().startswith("PMC"):
            found_pmcid = f"PMC{found_pmcid}"
        await self._limiter.acquire()
        try:
            resp = await self._client.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{found_pmcid}/fullTextXML",
            )
            if resp.status_code != 200:
                return None
            return _extract_text_from_jats_xml(resp.content)
        except Exception as e:
            logger.debug("full_text.europe_pmc_failed", paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 5: arXiv PDF
    # ------------------------------------------------------------------

    async def _try_arxiv(self, paper: CandidatePaper) -> str | None:
        arxiv_id = paper.external_ids.get("arxiv")
        # Fall back to extracting arXiv ID from DOI (10.48550/arXiv.2312.10997 -> 2312.10997)
        if not arxiv_id and paper.doi and paper.doi.lower().startswith("10.48550/arxiv."):
            arxiv_id = re.sub(r"^10\.48550/arXiv\.", "", paper.doi, flags=re.IGNORECASE)
        if not arxiv_id:
            return None

        # Normalise: strip version suffix for the PDF URL; arxiv serves latest
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{clean_id}"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                return None
            return _extract_text_from_pdf(resp.content)
        except Exception as e:
            logger.debug("full_text.arxiv_failed", paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 5: bioRxiv / medRxiv (DOI-based PDF, with version resolution)
    # ------------------------------------------------------------------

    async def _get_biorxiv_latest_version(self, doi: str, server: str) -> str | None:
        """Query the bioRxiv/medRxiv API to get the latest version number.

        Returns the version string (e.g. "v3") or None if the API call fails.
        """
        await self._limiter.acquire()
        try:
            resp = await self._client.get(
                f"https://api.biorxiv.org/details/{server}/{doi}",
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            collection = data.get("collection", [])
            if not collection:
                return None
            # The last entry in the collection is the most recent version
            latest = collection[-1]
            version = latest.get("version")
            if version:
                return f"v{version}"
            return None
        except Exception as e:
            logger.debug(
                "full_text.biorxiv_version_api_failed",
                doi=doi,
                server=server,
                error=str(e),
            )
            return None

    async def _fetch_biorxiv_pdf(self, doi: str, server: str, version: str) -> bytes | None:
        """Fetch a single bioRxiv/medRxiv PDF for the given DOI and version."""
        host = "biorxiv" if server == "biorxiv" else "medrxiv"
        pdf_url = f"https://www.{host}.org/content/{doi}{version}.full.pdf"
        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code == 200 and len(resp.content) >= 1000:
                return resp.content
            return None
        except Exception:
            return None

    async def _try_biorxiv(self, paper: CandidatePaper) -> str | None:
        """Fetch preprint PDF from bioRxiv or medRxiv.

        First queries the bioRxiv API to discover the latest version.
        If the API call fails, falls back to trying v1, v2, v3 sequentially.
        Then tries medRxiv with the same approach if bioRxiv yields nothing.
        """
        doi = paper.doi
        if not doi or not doi.startswith("10.1101/"):
            return None

        for server in ("biorxiv", "medrxiv"):
            # 1. Try to discover the latest version via the API
            version = await self._get_biorxiv_latest_version(doi, server)

            if version:
                content = await self._fetch_biorxiv_pdf(doi, server, version)
                if content:
                    logger.debug(
                        "full_text.biorxiv_version_api_hit",
                        doi=doi,
                        server=server,
                        version=version,
                        paper_id=paper.id,
                    )
                    text = _extract_text_from_pdf(content)
                    if text:
                        return text

            # 2. Fallback: try v1, v2, v3 sequentially
            logger.debug(
                "full_text.biorxiv_version_fallback",
                doi=doi,
                server=server,
                paper_id=paper.id,
            )
            for ver in _BIORXIV_FALLBACK_VERSIONS:
                content = await self._fetch_biorxiv_pdf(doi, server, ver)
                if content:
                    text = _extract_text_from_pdf(content)
                    if text:
                        return text

        logger.debug("full_text.biorxiv_all_versions_failed", doi=doi, paper_id=paper.id)
        return None

    # ------------------------------------------------------------------
    # Strategy 6: PLOS direct PDF (10.1371/ prefix — all OA)
    # ------------------------------------------------------------------

    _PLOS_JOURNAL_SLUG: dict[str, str] = {
        "pone": "plosone",
        "pbio": "plosbiology",
        "pmed": "plosmedicine",
        "pgen": "plosgenetics",
        "pcbi": "ploscompbiol",
        "ppat": "plospathogens",
        "pntd": "plosntds",
    }

    async def _try_plos(self, paper: CandidatePaper) -> str | None:
        """Fetch PDF directly from PLOS journals.

        All PLOS articles are Open Access.  The printable PDF is always
        available at a canonical URL that varies by journal slug.
        """
        doi = paper.doi
        if not doi or not doi.startswith(_PLOS_DOI_PREFIX):
            return None

        # Extract journal code from DOI: 10.1371/journal.pone.1234 → "pone"
        slug = "plosone"  # default fallback
        parts = doi.split("/", 1)
        if len(parts) > 1 and parts[1].startswith("journal."):
            code = parts[1].split(".")[1] if len(parts[1].split(".")) > 1 else ""
            slug = self._PLOS_JOURNAL_SLUG.get(code, "plosone")

        pdf_url = f"https://journals.plos.org/{slug}/article/file?id={doi}&type=printable"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                logger.debug(
                    "full_text.plos_http_failed",
                    doi=doi,
                    paper_id=paper.id,
                    status=resp.status_code,
                )
                return None
            text = _extract_text_from_pdf(resp.content)
            if text:
                logger.debug(
                    "full_text.plos_success",
                    doi=doi,
                    paper_id=paper.id,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug("full_text.plos_failed", doi=doi, paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 7: MDPI direct PDF (10.3390/ prefix — all OA)
    # ------------------------------------------------------------------

    async def _try_mdpi(self, paper: CandidatePaper) -> str | None:
        """Fetch PDF directly from MDPI journals.

        All MDPI articles are Open Access.  MDPI's Akamai CDN blocks
        bot-like User-Agents with 403, so we use a browser-like UA for
        these requests.  Strategy: resolve DOI → download PDF.
        """
        doi = paper.doi
        if not doi or not doi.startswith(_MDPI_DOI_PREFIX):
            return None

        # MDPI blocks bot UAs with 403 — use browser-like headers
        _mdpi_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        # Resolve DOI to get the canonical article URL.  MDPI DOI suffixes
        # (e.g. "jcm14041086") do NOT map directly to URL paths — the
        # article URL uses ISSN/vol/issue format.
        await self._limiter.acquire()
        try:
            head_resp = await self._client.head(
                f"https://doi.org/{doi}",
                follow_redirects=True,
                headers=_mdpi_headers,
            )
            article_url = str(head_resp.url)

            if "mdpi.com" in article_url:
                base = article_url.rstrip("/")
                pdf_url = base + "/pdf"
                await self._limiter.acquire()
                resp = await self._client.get(pdf_url, headers=_mdpi_headers)
                if resp.status_code == 200 and len(resp.content) >= 1000:
                    text = _extract_text_from_pdf(resp.content)
                    if text:
                        logger.debug(
                            "full_text.mdpi_success",
                            doi=doi,
                            paper_id=paper.id,
                            chars=len(text),
                            method="pdf_resolved",
                        )
                        return text
        except Exception as e:
            logger.debug("full_text.mdpi_doi_resolve_failed", doi=doi, error=str(e))

        logger.debug(
            "full_text.mdpi_all_failed",
            doi=doi,
            paper_id=paper.id,
        )
        return None

    # ------------------------------------------------------------------
    # Strategy 8: Frontiers direct PDF (10.3389/ prefix — all OA)
    # ------------------------------------------------------------------

    async def _try_frontiers(self, paper: CandidatePaper) -> str | None:
        """Fetch PDF directly from Frontiers journals.

        All Frontiers articles are Open Access.  The PDF URL follows the
        pattern https://www.frontiersin.org/articles/{doi}/pdf which
        redirects to the actual PDF.
        """
        doi = paper.doi
        if not doi or not doi.startswith(_FRONTIERS_DOI_PREFIX):
            return None

        pdf_url = f"https://www.frontiersin.org/articles/{doi}/pdf"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                logger.debug(
                    "full_text.frontiers_http_failed",
                    doi=doi,
                    paper_id=paper.id,
                    status=resp.status_code,
                )
                return None
            text = _extract_text_from_pdf(resp.content)
            if text:
                logger.debug(
                    "full_text.frontiers_success",
                    doi=doi,
                    paper_id=paper.id,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug("full_text.frontiers_failed", doi=doi, paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 9: ACL Anthology (10.18653/ prefix — all OA)
    # ------------------------------------------------------------------

    async def _try_acl_anthology(self, paper: CandidatePaper) -> str | None:
        """Fetch PDF from ACL Anthology (aclanthology.org).

        All ACL Anthology papers are Open Access.  DOI format:
        10.18653/v1/2024.emnlp-main.519 -> https://aclanthology.org/2024.emnlp-main.519.pdf
        """
        doi = paper.doi
        if not doi or not doi.startswith(_ACL_DOI_PREFIX):
            return None

        # Extract the paper ID from the DOI (after "10.18653/v1/")
        suffix = doi[len("10.18653/v1/") :] if doi.startswith("10.18653/v1/") else None
        if not suffix:
            return None

        pdf_url = f"https://aclanthology.org/{suffix}.pdf"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                return None
            text = _extract_text_from_pdf(resp.content)
            if text:
                logger.debug(
                    "full_text.acl_anthology_success",
                    doi=doi,
                    paper_id=paper.id,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug("full_text.acl_anthology_failed", doi=doi, paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 9b: JMIR (10.2196/ prefix — all OA)
    # ------------------------------------------------------------------

    async def _try_jmir(self, paper: CandidatePaper) -> str | None:
        """Fetch full text from JMIR journals (jmir.org).

        All JMIR articles are Open Access.  Uses DOI resolution to find
        the canonical article URL, then fetches XML full text.
        """
        doi = paper.doi
        if not doi or not doi.startswith(_JMIR_DOI_PREFIX):
            return None

        # JMIR XML endpoint: https://www.jmir.org/api/download-xml/{article_id}
        # Article ID is the DOI suffix (e.g. "38052" from "10.2196/38052")
        article_id = doi[len(_JMIR_DOI_PREFIX) :]
        if not article_id:
            return None

        # Try the JMIR XML API first
        xml_url = f"https://www.jmir.org/api/download-xml/{article_id}"
        await self._limiter.acquire()
        try:
            resp = await self._client.get(xml_url)
            if resp.status_code == 200 and len(resp.content) >= 500:
                text = _extract_text_from_jats_xml(resp.content)
                if text:
                    logger.debug(
                        "full_text.jmir_success",
                        doi=doi,
                        paper_id=paper.id,
                        chars=len(text),
                    )
                    return text
        except Exception as e:
            logger.debug("full_text.jmir_xml_failed", doi=doi, error=str(e))

        # Fallback: resolve DOI and try PDF
        await self._limiter.acquire()
        try:
            head_resp = await self._client.head(
                f"https://doi.org/{doi}",
                follow_redirects=True,
            )
            article_url = str(head_resp.url)
            if "jmir.org" in article_url:
                pdf_url = article_url.rstrip("/") + "/PDF"
                await self._limiter.acquire()
                resp = await self._client.get(pdf_url)
                if resp.status_code == 200 and len(resp.content) >= 1000:
                    text = _extract_text_from_pdf(resp.content)
                    if text:
                        logger.debug(
                            "full_text.jmir_pdf_success",
                            doi=doi,
                            paper_id=paper.id,
                            chars=len(text),
                        )
                        return text
        except Exception as e:
            logger.debug("full_text.jmir_pdf_failed", doi=doi, error=str(e))

        return None

    # ------------------------------------------------------------------
    # Strategy 10: Unpaywall (DOI lookup, tries all available URLs)
    # ------------------------------------------------------------------

    async def _try_unpaywall(self, paper: CandidatePaper) -> str | None:
        if not self._unpaywall_email or not paper.doi:
            return None

        # Lazy-init the Unpaywall client
        if self._unpaywall is None:
            from autoreview.search.unpaywall import UnpaywallClient

            self._unpaywall = UnpaywallClient(email=self._unpaywall_email)

        result = await self._unpaywall.lookup_doi(paper.doi)
        if not result or (not result.pdf_url and not result.html_url):
            return None
        return await self._unpaywall.fetch_full_text(result)
