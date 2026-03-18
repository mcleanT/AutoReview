"""Tests for CrossRefSearch backend."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest
import respx

from autoreview.models.paper import CandidatePaper
from autoreview.search.crossref import CROSSREF_API_BASE, CrossRefSearch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_WORK = {
    "DOI": "10.1038/nature12345",
    "title": ["Deep Learning for Genomics"],
    "author": [
        {"given": "Alice", "family": "Smith"},
        {"given": "Bob", "family": "Jones"},
    ],
    "published-print": {"date-parts": [[2022, 6, 1]]},
    "abstract": (
        "<jats:p>We present a novel approach to genomic analysis using deep learning.</jats:p>"
    ),
    "container-title": ["Nature"],
    "is-referenced-by-count": 142,
    "link": [
        {"content-type": "application/pdf", "URL": "https://example.com/paper.pdf"},
        {"content-type": "text/html", "URL": "https://example.com/paper.html"},
    ],
}

_SAMPLE_WORK_2 = {
    "DOI": "10.1126/science.abc1234",
    "title": ["Transformer Models in Biology"],
    "author": [
        {"given": "Carol", "family": "White"},
    ],
    "published-online": {"date-parts": [[2021, 3, 15]]},
    "abstract": "<jats:p>Transformer architectures for biological sequences.</jats:p>",
    "container-title": ["Science"],
    "is-referenced-by-count": 89,
    "link": [],
}

_SEARCH_RESPONSE = {
    "status": "ok",
    "message": {
        "total-results": 2,
        "items": [_SAMPLE_WORK, _SAMPLE_WORK_2],
    },
}


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestCrossRefSearchSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_candidate_papers(self):
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert len(papers) == 2
        assert all(isinstance(p, CandidatePaper) for p in papers)

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_fields_correctly(self):
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        paper = papers[0]
        assert paper.title == "Deep Learning for Genomics"
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.year == 2022
        assert paper.doi == "10.1038/nature12345"
        assert "novel approach to genomic analysis" in (paper.abstract or "")
        assert paper.source_database == "crossref"
        assert paper.journal == "Nature"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_authors_given_family(self):
        """Authors with given+family names are combined as 'given family'."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        assert "Alice Smith" in papers[0].authors
        assert "Bob Jones" in papers[0].authors

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_year_from_published_print(self):
        """Year is extracted from published-print.date-parts."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        assert papers[0].year == 2022

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_year_from_published_online_fallback(self):
        """Year falls back to published-online when published-print is absent."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        # _SAMPLE_WORK_2 has only published-online
        assert papers[1].year == 2021

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_strips_jats_xml_from_abstract(self):
        """JATS XML tags are stripped from CrossRef abstracts."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        abstract = papers[0].abstract or ""
        assert "<jats:" not in abstract
        assert "novel approach to genomic analysis" in abstract

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_citation_count(self):
        """is-referenced-by-count maps to citation_count."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        assert papers[0].citation_count == 142
        assert papers[1].citation_count == 89

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_crossref_links_in_external_ids(self):
        """Link array is stored as JSON in external_ids['crossref_links']."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        raw = papers[0].external_ids.get("crossref_links")
        assert raw is not None
        links = json.loads(raw)
        assert len(links) == 2
        content_types = {lnk["content-type"] for lnk in links}
        assert "application/pdf" in content_types

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_no_crossref_links_when_empty(self):
        """No crossref_links key when link array is empty."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["genomics"], max_results=10)

        # _SAMPLE_WORK_2 has empty link array
        assert "crossref_links" not in papers[1].external_ids

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_includes_mailto_in_params(self):
        """The polite pool email is sent as a query parameter."""
        captured_params: dict[str, str] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_params.update(dict(request.url.params))
            return httpx.Response(200, json=_SEARCH_RESPONSE)

        respx.get(CROSSREF_API_BASE).mock(side_effect=capture_request)

        searcher = CrossRefSearch(email="polite@example.com")
        await searcher.search(queries=["test"], max_results=5)

        assert captured_params.get("mailto") == "polite@example.com"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_http_error_gracefully(self):
        """HTTP errors are caught and an empty list is returned."""
        respx.get(CROSSREF_API_BASE).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_empty_results(self):
        """Empty items list returns empty paper list."""
        empty_response = {"status": "ok", "message": {"total-results": 0, "items": []}}
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=empty_response))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["xyznonexistent"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_skips_work_without_title(self):
        """Works missing a title are silently skipped."""
        no_title_work = dict(_SAMPLE_WORK)
        no_title_work["title"] = []
        response = {"status": "ok", "message": {"total-results": 1, "items": [no_title_work]}}
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=response))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["test"], max_results=5)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_deduplicates_by_doi(self):
        """Duplicate DOIs across queries are deduplicated."""
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["deep learning", "genomics"], max_results=20)

        dois = [p.doi for p in papers if p.doi]
        assert len(dois) == len(set(dois))


# ---------------------------------------------------------------------------
# get_paper_details tests
# ---------------------------------------------------------------------------


class TestCrossRefGetPaperDetails:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_returns_paper(self):
        doi = "10.1038/nature12345"
        respx.get(f"{CROSSREF_API_BASE}/{doi}").mock(
            return_value=httpx.Response(200, json={"status": "ok", "message": _SAMPLE_WORK})
        )

        searcher = CrossRefSearch(email="test@example.com")
        paper = await searcher.get_paper_details(doi)

        assert paper is not None
        assert paper.title == "Deep Learning for Genomics"
        assert paper.source_database == "crossref"

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_handles_404(self):
        doi = "10.9999/doesnotexist"
        respx.get(f"{CROSSREF_API_BASE}/{doi}").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        searcher = CrossRefSearch(email="test@example.com")
        paper = await searcher.get_paper_details(doi)

        assert paper is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_handles_request_error(self):
        doi = "10.1038/nature12345"
        respx.get(f"{CROSSREF_API_BASE}/{doi}").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        searcher = CrossRefSearch(email="test@example.com")
        paper = await searcher.get_paper_details(doi)

        assert paper is None


# ---------------------------------------------------------------------------
# Edge-case field mapping
# ---------------------------------------------------------------------------


class TestCrossRefFieldMapping:
    @pytest.mark.asyncio
    @respx.mock
    async def test_author_family_only(self):
        """Authors with only a family name are handled."""
        work = dict(_SAMPLE_WORK)
        work["author"] = [{"family": "Nakamura"}]
        response = {"status": "ok", "message": {"total-results": 1, "items": [work]}}
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=response))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["test"], max_results=5)

        assert papers[0].authors == ["Nakamura"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_abstract_is_none(self):
        """Missing abstract results in None."""
        work = dict(_SAMPLE_WORK)
        del work["abstract"]
        response = {"status": "ok", "message": {"total-results": 1, "items": [work]}}
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=response))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["test"], max_results=5)

        assert papers[0].abstract is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_doi_is_lowercased(self):
        """DOIs are normalized to lowercase."""
        work = dict(_SAMPLE_WORK)
        work["DOI"] = "10.1038/NATURE12345"
        response = {"status": "ok", "message": {"total-results": 1, "items": [work]}}
        respx.get(CROSSREF_API_BASE).mock(return_value=httpx.Response(200, json=response))

        searcher = CrossRefSearch(email="test@example.com")
        papers = await searcher.search(queries=["test"], max_results=5)

        assert papers[0].doi == "10.1038/nature12345"

    @pytest.mark.asyncio
    async def test_polite_pool_rate_with_email(self):
        """Rate limiter is set to 50 req/sec when email is provided."""
        searcher = CrossRefSearch(email="test@example.com")
        assert searcher._limiter.requests_per_second == 50.0

    @pytest.mark.asyncio
    async def test_anonymous_rate_without_email(self):
        """Rate limiter falls back to 1 req/sec when no email is available."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure no email env vars are set
            env_backup_cr = os.environ.pop("CROSSREF_EMAIL", None)
            env_backup_e = os.environ.pop("ENTREZ_EMAIL", None)
            try:
                searcher = CrossRefSearch(email=None)
                assert searcher._limiter.requests_per_second == 1.0
            finally:
                if env_backup_cr is not None:
                    os.environ["CROSSREF_EMAIL"] = env_backup_cr
                if env_backup_e is not None:
                    os.environ["ENTREZ_EMAIL"] = env_backup_e
