"""Multi-source full-text resolver.

Chains multiple strategies to maximise full-text coverage:

1. Elsevier ScienceDirect API  (DOI-based, requires ELSEVIER_API_KEY)
2. Springer Nature Open Access API  (DOI-based, requires SPRINGER_API_KEY)
3. Semantic Scholar openAccessPdf  (stored during search)
4. PubMed Central  (PMID/PMCID -> JATS XML)
5. arXiv / bioRxiv / medRxiv  (preprint PDF)
6. Unpaywall  (DOI-based OA lookup)

Each strategy is tried in order; the first one that returns usable text wins.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
import structlog

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

# Reuse the PDF / HTML text extraction helpers from the unpaywall module.
from autoreview.search.unpaywall import _extract_text_from_html, _extract_text_from_pdf

_MAX_TEXT_CHARS = 200_000

# Elsevier DOI prefixes (covers ScienceDirect, Cell Press, Lancet, etc.)
_ELSEVIER_DOI_PREFIXES = (
    "10.1016/",   # ScienceDirect, Cell, Lancet, etc.
    "10.1006/",   # Academic Press (legacy Elsevier)
    "10.1053/",   # Elsevier subsidiary
)

# Springer Nature DOI prefixes (covers Nature, Springer, BMC, Palgrave, etc.)
_SPRINGER_NATURE_DOI_PREFIXES = (
    "10.1038/",   # Nature Publishing Group
    "10.1007/",   # Springer journals & books
    "10.1186/",   # BioMed Central
    "10.1057/",   # Palgrave Macmillan
    "10.1140/",   # European Physical Journal
    "10.1365/",   # Springer Fachmedien
    "10.1251/",   # Springer Japan
)

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
        body = (
            soup.find("ce:sections")
            or soup.find("body")
            or soup.find("rawtext")
        )
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
      2. Springer Nature Open Access API     (DOI-based, JATS XML)
      3. Semantic Scholar openAccessPdf URL   (if stored in external_ids)
      4. PubMed Central via PMCID/PMID        (free JATS XML)
      5. arXiv PDF                             (if arXiv ID present)
      6. bioRxiv / medRxiv PDF                 (preprint DOI)
      7. Unpaywall                             (DOI-based OA lookup)
    """

    def __init__(
        self,
        unpaywall_email: str | None = None,
        entrez_email: str | None = None,
        elsevier_api_key: str | None = None,
        springer_api_key: str | None = None,
        requests_per_second: float = 10.0,
    ) -> None:
        self._unpaywall_email = unpaywall_email
        self._entrez_email = entrez_email or os.environ.get(
            "ENTREZ_EMAIL", "autoreview@example.com"
        )
        self._elsevier_api_key = elsevier_api_key or os.environ.get(
            "ELSEVIER_API_KEY"
        )
        self._springer_api_key = springer_api_key or os.environ.get(
            "SPRINGER_API_KEY"
        )
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
        await asyncio.gather(*tasks, return_exceptions=True)

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
        for name, fn in [
            ("elsevier_api", self._try_elsevier_api),
            ("s2_pdf", self._try_s2_pdf),
            ("pmc", lambda p: self._try_pmc(p, pmid_to_pmcid)),
            ("arxiv", self._try_arxiv),
            ("biorxiv", self._try_biorxiv),
            ("unpaywall", self._try_unpaywall),
            # Rate-limited (500 calls/day) — last resort for OA articles
            ("springer_oa", self._try_springer_oa),
        ]:
            try:
                text = await fn(paper)
                if text and len(text) > 100:
                    return name, text
            except Exception as e:
                logger.debug(
                    "full_text.strategy_failed",
                    strategy=name,
                    paper_id=paper.id,
                    error=str(e),
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
    # Strategy 6: bioRxiv / medRxiv (DOI-based PDF)
    # ------------------------------------------------------------------

    async def _try_biorxiv(self, paper: CandidatePaper) -> str | None:
        """Fetch preprint PDF from bioRxiv or medRxiv."""
        doi = paper.doi
        if not doi or not doi.startswith("10.1101/"):
            return None

        # bioRxiv/medRxiv PDFs follow a predictable URL pattern
        pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"

        await self._limiter.acquire()
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200 or len(resp.content) < 1000:
                # Try medRxiv if bioRxiv fails
                pdf_url = f"https://www.medrxiv.org/content/{doi}v1.full.pdf"
                await self._limiter.acquire()
                resp = await self._client.get(pdf_url)
                if resp.status_code != 200 or len(resp.content) < 1000:
                    return None
            return _extract_text_from_pdf(resp.content)
        except Exception as e:
            logger.debug("full_text.biorxiv_failed", paper_id=paper.id, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Strategy 7: Unpaywall (existing, as fallback)
    # ------------------------------------------------------------------

    async def _try_unpaywall(self, paper: CandidatePaper) -> str | None:
        if not self._unpaywall_email or not paper.doi:
            return None

        # Lazy-init the Unpaywall client
        if self._unpaywall is None:
            from autoreview.search.unpaywall import UnpaywallClient

            self._unpaywall = UnpaywallClient(email=self._unpaywall_email)

        result = await self._unpaywall.lookup_doi(paper.doi)
        if not result or not result.is_oa:
            return None
        return await self._unpaywall.fetch_full_text(result)
