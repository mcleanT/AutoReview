from __future__ import annotations

import asyncio
import contextlib
import os
import time
from functools import partial
from typing import Any

import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger()

_MAX_RETRIES = 3
_RETRY_BACKOFF = [2.0, 5.0, 10.0]


class PubMedSearch:
    """PubMed search via NCBI Entrez API (biopython)."""

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._email = email or os.environ.get("ENTREZ_EMAIL", "autoreview@example.com")
        self._api_key = api_key or os.environ.get("NCBI_API_KEY")
        rate = 10.0 if self._api_key else 3.0
        self._limiter = RateLimiter(rate)

    @property
    def source_name(self) -> str:
        return "pubmed"

    def _setup_entrez(self) -> Any:
        from Bio import Entrez

        Entrez.email = self._email
        if self._api_key:
            Entrez.api_key = self._api_key
        return Entrez

    def _sync_search_with_retry(self, query: str, max_results: int) -> list[str]:
        for attempt in range(_MAX_RETRIES):
            try:
                entrez = self._setup_entrez()
                handle = entrez.esearch(
                    db="pubmed", term=query, retmax=max_results, sort="relevance"
                )
                results = entrez.read(handle)
                handle.close()
                return results.get("IdList", [])
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "pubmed.search_retry",
                        query=query[:80],
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    time.sleep(_RETRY_BACKOFF[attempt])
                else:
                    logger.error("pubmed.search_failed", query=query[:80], error=str(e))
                    return []

    def _sync_fetch_with_retry(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        entrez = self._setup_entrez()
        all_articles: list[dict[str, Any]] = []
        for i in range(0, len(pmids), 200):
            batch = pmids[i : i + 200]
            for attempt in range(_MAX_RETRIES):
                try:
                    handle = entrez.efetch(
                        db="pubmed",
                        id=",".join(batch),
                        rettype="xml",
                        retmode="xml",
                    )
                    records = entrez.read(handle)
                    handle.close()
                    all_articles.extend(records.get("PubmedArticle", []))
                    break
                except Exception as e:
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning(
                            "pubmed.fetch_retry",
                            batch_start=i,
                            attempt=attempt + 1,
                            error=str(e),
                        )
                        time.sleep(_RETRY_BACKOFF[attempt])
                    else:
                        logger.error(
                            "pubmed.fetch_failed",
                            batch_start=i,
                            error=str(e),
                        )
        return all_articles

    def _parse_article(self, article: dict[str, Any]) -> CandidatePaper | None:
        try:
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            title = str(article_data.get("ArticleTitle", ""))
            if not title:
                return None

            authors = []
            for author in article_data.get("AuthorList", []):
                last = author.get("LastName", "")
                first = author.get("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

            year = None
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            if "Year" in pub_date:
                with contextlib.suppress(ValueError, TypeError):
                    year = int(pub_date["Year"])

            journal = article_data.get("Journal", {}).get("Title")
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else None

            doi = None
            for eid in article_data.get("ELocationID", []):
                if hasattr(eid, "attributes") and eid.attributes.get("EIdType") == "doi":
                    doi = str(eid)
                    break

            pubmed_data = article.get("PubmedData", {})
            pmid = str(medline.get("PMID", ""))
            if not doi:
                for aid in pubmed_data.get("ArticleIdList", []):
                    if hasattr(aid, "attributes") and aid.attributes.get("IdType") == "doi":
                        doi = str(aid)
                        break

            external_ids: dict[str, str] = {}
            if pmid:
                external_ids["pmid"] = pmid

            # Extract PMCID if available
            for aid in pubmed_data.get("ArticleIdList", []):
                if hasattr(aid, "attributes") and aid.attributes.get("IdType") == "pmc":
                    pmcid = str(aid)
                    if pmcid:
                        external_ids["pmcid"] = pmcid
                    break

            return CandidatePaper(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                abstract=abstract,
                source_database="pubmed",
                external_ids=external_ids,
            )
        except Exception as e:
            logger.warning("pubmed.parse_error", error=str(e))
            return None

    async def search(self, queries: list[str], max_results: int = 100) -> list[CandidatePaper]:
        loop = asyncio.get_running_loop()
        all_pmids: list[str] = []
        per_query_max = max(max_results // len(queries), 20) if queries else max_results

        for query in queries:
            await self._limiter.acquire()
            pmids = await loop.run_in_executor(
                None,
                partial(self._sync_search_with_retry, query, per_query_max),
            )
            all_pmids.extend(pmids)
            logger.info("pubmed.search", query=query[:80], results=len(pmids))

        unique_pmids = list(dict.fromkeys(all_pmids))[:max_results]
        if not unique_pmids:
            return []

        await self._limiter.acquire()
        articles = await loop.run_in_executor(
            None,
            partial(self._sync_fetch_with_retry, unique_pmids),
        )

        papers = [p for a in articles if (p := self._parse_article(a)) is not None]
        logger.info("pubmed.search.complete", total_papers=len(papers))
        return papers

    async def get_paper_details(self, paper_id: str) -> CandidatePaper | None:
        loop = asyncio.get_running_loop()
        await self._limiter.acquire()
        articles = await loop.run_in_executor(
            None,
            partial(self._sync_fetch_with_retry, [paper_id]),
        )
        return self._parse_article(articles[0]) if articles else None
