"""Multi-source full-text resolver.

Chains multiple strategies to maximise full-text coverage:

1. Elsevier ScienceDirect API  (DOI-based, requires ELSEVIER_API_KEY)
2. Semantic Scholar openAccessPdf  (stored during search)
3. CORE API  (core_pdf_url or direct output download, 300M+ OA records)
4. PubMed Central  (PMID/PMCID -> JATS XML)
4b. Europe PMC  (PMCID -> JATS XML, superset of PMC with sometimes better availability)
5. arXiv / bioRxiv / medRxiv  (preprint PDF)
6. PLOS  (10.1371/ prefix, guaranteed OA)
7. MDPI  (10.3390/ prefix, guaranteed OA)
8. Frontiers  (10.3389/ prefix, guaranteed OA)
9. Unpaywall  (DOI-based OA lookup)
10. Springer Nature Open Access API  (DOI-based, last resort)

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

_MAX_TEXT_CHARS = 200_000

# Elsevier DOI prefixes (covers ScienceDirect, Cell Press, Lancet, etc.)
_ELSEVIER_DOI_PREFIXES = (
    "10.1016/",  # ScienceDirect, Cell, Lancet, etc.
    "10.1006/",  # Academic Press (legacy Elsevier)
    "10.1053/",  # Elsevier subsidiary
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

# bioRxiv / medRxiv versions to try when API lookup fails
_BIORXIV_FALLBACK_VERSIONS = ("v1", "v2", "v3")

# ---------------------------------------------------------------------------
# Elsevier XML text extraction
# ---------------------------------------------------------------------------


def _extract_text_from_elsevier_xml(xml_text: str) -> str | None:
    """Extract article body text from Elsevier's full-text-retrieval XML."""
    try:
        from bs4 import BeautifulSoup

        for parser in ("lxml-xml", "xml", "html.parser"):
            try:
                soup = BeautifulSoup(xml_text, parser)
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

        # Try lxml-xml first, fall back to html.parser for environments
        # where the lxml XML parser is not available.
        for parser in ("lxml-xml", "xml", "html.parser"):
            try:
                soup = BeautifulSoup(xml_bytes, parser)
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
      2. Semantic Scholar openAccessPdf URL   (if stored in external_ids)
      3. CORE API                             (core_pdf_url or direct output download)
      3b. CrossRef links                      (PDF > XML > HTML from external_ids)
      4. PubMed Central via PMCID/PMID        (free JATS XML)
      4b. Europe PMC                          (PMCID -> JATS XML, PMC superset)
      5. arXiv PDF                             (if arXiv ID present)
      6. bioRxiv / medRxiv PDF                 (preprint DOI, with version API)
      7. PLOS direct PDF                       (10.1371/ prefix, guaranteed OA)
      8. MDPI direct PDF                       (10.3390/ prefix, guaranteed OA)
      9. Frontiers direct PDF                  (10.3389/ prefix, guaranteed OA)
     10. Unpaywall                             (DOI-based OA lookup)
     11. Springer Nature Open Access API      (DOI-based JATS XML, last resort)
    """

    def __init__(
        self,
        unpaywall_email: str | None = None,
        entrez_email: str | None = None,
        elsevier_api_key: str | None = None,
        springer_api_key: str | None = None,
        core_api_key: str | None = None,
        crossref_email: str | None = None,
        requests_per_second: float = 10.0,
    ) -> None:
        self._unpaywall_email = unpaywall_email
        self._entrez_email = entrez_email or os.environ.get(
            "ENTREZ_EMAIL", "autoreview@example.com"
        )
        self._crossref_email = (
            crossref_email or os.environ.get("CROSSREF_EMAIL") or os.environ.get("ENTREZ_EMAIL")
        )
        self._elsevier_api_key = elsevier_api_key or os.environ.get("ELSEVIER_API_KEY")
        self._springer_api_key = springer_api_key or os.environ.get("SPRINGER_API_KEY")
        self._core_api_key = core_api_key or os.environ.get("CORE_API_KEY")
        self._client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "AutoReview/1.0 (full-text-resolver)"},
        )
        self._limiter = RateLimiter(requests_per_second)

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
        ``paper.full_text_source``.

        Returns a dict of ``{source_name: count}`` for audit logging.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        source_counts: dict[str, int] = {}
        lock = asyncio.Lock()

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
            ("s2_pdf", self._try_s2_pdf),
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
            ("unpaywall", self._try_unpaywall),
            # Rate-limited (500 calls/day) — last resort for OA articles
            ("springer_oa", self._try_springer_oa),
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

        await self._limiter.acquire()
        try:
            # Try plain text first (cleanest output, no parsing needed)
            resp = await self._client.get(
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
            await self._limiter.acquire()
            resp = await self._client.get(
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
    # Strategy 2: Springer Nature Open Access API (JATS XML)
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

        await self._limiter.acquire()
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
    # Strategy 3: Semantic Scholar openAccessPdf
    # ------------------------------------------------------------------

    async def _try_s2_pdf(self, paper: CandidatePaper) -> str | None:
        url = paper.external_ids.get("s2_pdf_url")
        if not url:
            return None

        await self._limiter.acquire()
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                return None
            return _extract_text_from_pdf(resp.content)
        except Exception as e:
            logger.debug("full_text.s2_pdf_failed", paper_id=paper.id, error=str(e))
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

            await self._limiter.acquire()
            try:
                resp = await self._client.get(
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
            await self._limiter.acquire()
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

        await self._limiter.acquire()
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
        """Fetch full-text JATS XML from Europe PMC using the paper's PMCID.

        Europe PMC provides a dedicated full-text XML endpoint for PMC articles.
        It is a superset of PubMed Central and sometimes has better availability.
        Only works for papers with a ``pmcid`` in ``external_ids``.
        No API key required.
        """
        pmcid = paper.external_ids.get("pmcid")
        if not pmcid:
            return None

        # Normalise: ensure it starts with "PMC"
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
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

        All MDPI articles are Open Access.  The PDF URL is constructed
        from the DOI suffix (everything after '10.3390/').
        """
        doi = paper.doi
        if not doi or not doi.startswith(_MDPI_DOI_PREFIX):
            return None

        doi_suffix = doi[len(_MDPI_DOI_PREFIX) :]
        pdf_url = f"https://www.mdpi.com/{doi_suffix}/pdf"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                logger.debug(
                    "full_text.mdpi_http_failed",
                    doi=doi,
                    paper_id=paper.id,
                    status=resp.status_code,
                )
                return None
            text = _extract_text_from_pdf(resp.content)
            if text:
                logger.debug(
                    "full_text.mdpi_success",
                    doi=doi,
                    paper_id=paper.id,
                    chars=len(text),
                )
            return text
        except Exception as e:
            logger.debug("full_text.mdpi_failed", doi=doi, paper_id=paper.id, error=str(e))
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
    # Strategy 9: Unpaywall (DOI lookup, tries all available URLs)
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
