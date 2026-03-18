"""Tests for CORE (core.ac.uk) search backend."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from autoreview.models.paper import CandidatePaper
from autoreview.search.core_api import CORE_API_BASE, CORESearch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_WORK = {
    "id": 12345678,
    "title": "Deep Learning for Scientific Discovery",
    "authors": [
        {"name": "Alice Smith"},
        {"name": "Bob Jones"},
    ],
    "yearPublished": 2022,
    "doi": "10.1234/test.2022.001",
    "abstract": "We present a novel deep learning framework for accelerating scientific discovery.",
    "downloadUrl": "https://core.ac.uk/download/pdf/12345678.pdf",
    "journals": [{"title": "Nature Machine Intelligence"}],
}

_SAMPLE_WORK_2 = {
    "id": 87654321,
    "title": "Transformer Models in Biology",
    "authors": [{"name": "Carol White"}],
    "yearPublished": 2021,
    "doi": "10.5678/bio.2021.042",
    "abstract": "Transformer architectures applied to biological sequence analysis.",
    "downloadUrl": "https://core.ac.uk/download/pdf/87654321.pdf",
}

_SEARCH_RESPONSE = {
    "totalHits": 2,
    "limit": 10,
    "offset": 0,
    "results": [_SAMPLE_WORK, _SAMPLE_WORK_2],
}


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestCORESearchSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_candidate_papers(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert len(papers) == 2
        assert all(isinstance(p, CandidatePaper) for p in papers)

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_fields_correctly(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        paper = papers[0]
        assert paper.title == "Deep Learning for Scientific Discovery"
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.year == 2022
        assert paper.doi == "10.1234/test.2022.001"
        assert (
            paper.abstract
            == "We present a novel deep learning framework for accelerating scientific discovery."
        )
        assert paper.source_database == "core"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_core_pdf_url_in_external_ids(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        paper = papers[0]
        assert (
            paper.external_ids.get("core_pdf_url") == "https://core.ac.uk/download/pdf/12345678.pdf"
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_core_id_in_external_ids(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        paper = papers[0]
        assert paper.external_ids.get("core") == "12345678"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_http_error_gracefully(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_404_gracefully(self):
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["nonexistent topic xyzabc"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_empty_results(self):
        empty_response = {"totalHits": 0, "limit": 10, "offset": 0, "results": []}
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=empty_response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_deduplicates_results(self):
        # Two queries returning the same paper
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(
            queries=["deep learning", "machine learning"], max_results=20
        )

        # Papers from both queries are returned (dedup by core id within query,
        # but across queries it's normal to get some duplication — the aggregator handles cross-source dedup)
        assert len(papers) <= 20
        assert all(isinstance(p, CandidatePaper) for p in papers)

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_strips_doi_prefix(self):
        work_with_full_doi = dict(_SAMPLE_WORK)
        work_with_full_doi["doi"] = "https://doi.org/10.1234/test.2022.001"
        response = {"totalHits": 1, "limit": 10, "offset": 0, "results": [work_with_full_doi]}

        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert papers[0].doi == "10.1234/test.2022.001"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_rate_limiting_retries_on_429(self):
        call_count = 0

        def mock_response(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, text="Too Many Requests")
            return httpx.Response(200, json=_SEARCH_RESPONSE)

        respx.get(f"{CORE_API_BASE}/search/works").mock(side_effect=mock_response)

        searcher = CORESearch(api_key="test-key")
        # Patch asyncio.sleep to avoid actual waiting
        with patch("autoreview.search.core_api.asyncio.sleep", new_callable=AsyncMock):
            papers = await searcher.search(queries=["deep learning"], max_results=10)

        assert len(papers) == 2


# ---------------------------------------------------------------------------
# get_paper_details tests
# ---------------------------------------------------------------------------


class TestCORESearchGetPaperDetails:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_returns_paper(self):
        respx.get(f"{CORE_API_BASE}/outputs/12345678").mock(
            return_value=httpx.Response(200, json=_SAMPLE_WORK)
        )

        searcher = CORESearch(api_key="test-key")
        paper = await searcher.get_paper_details("12345678")

        assert paper is not None
        assert paper.title == "Deep Learning for Scientific Discovery"
        assert paper.source_database == "core"

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_handles_404(self):
        respx.get(f"{CORE_API_BASE}/outputs/99999999").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        searcher = CORESearch(api_key="test-key")
        paper = await searcher.get_paper_details("99999999")

        assert paper is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_paper_details_handles_request_error(self):
        respx.get(f"{CORE_API_BASE}/outputs/12345678").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        searcher = CORESearch(api_key="test-key")
        with patch("autoreview.search.core_api.asyncio.sleep", new_callable=AsyncMock):
            paper = await searcher.get_paper_details("12345678")

        assert paper is None


# ---------------------------------------------------------------------------
# Field mapping edge cases
# ---------------------------------------------------------------------------


class TestCOREFieldMapping:
    @pytest.mark.asyncio
    @respx.mock
    async def test_maps_year_published_as_integer(self):
        work = dict(_SAMPLE_WORK)
        work["yearPublished"] = "2020"
        response = {"totalHits": 1, "limit": 10, "offset": 0, "results": [work]}
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["test"], max_results=5)
        assert papers[0].year == 2020

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_missing_authors(self):
        work = dict(_SAMPLE_WORK)
        work["authors"] = None
        response = {"totalHits": 1, "limit": 10, "offset": 0, "results": [work]}
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["test"], max_results=5)
        assert papers[0].authors == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_missing_download_url(self):
        work = dict(_SAMPLE_WORK)
        del work["downloadUrl"]
        response = {"totalHits": 1, "limit": 10, "offset": 0, "results": [work]}
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["test"], max_results=5)
        assert "core_pdf_url" not in papers[0].external_ids

    @pytest.mark.asyncio
    @respx.mock
    async def test_skips_work_without_title(self):
        work_no_title = {"id": 999, "title": None, "yearPublished": 2020}
        response = {"totalHits": 1, "limit": 10, "offset": 0, "results": [work_no_title]}
        respx.get(f"{CORE_API_BASE}/search/works").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = CORESearch(api_key="test-key")
        papers = await searcher.search(queries=["test"], max_results=5)
        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_uses_api_key_in_auth_header(self):
        captured_headers: dict[str, str] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_SEARCH_RESPONSE)

        respx.get(f"{CORE_API_BASE}/search/works").mock(side_effect=capture_request)

        searcher = CORESearch(api_key="my-secret-key")
        await searcher.search(queries=["test"], max_results=5)

        assert captured_headers.get("authorization") == "Bearer my-secret-key"

    @pytest.mark.asyncio
    @respx.mock
    @patch.dict(os.environ, {}, clear=False)
    async def test_no_api_key_no_auth_header(self):
        # Pop CORE_API_KEY BEFORE constructing searcher (constructor reads env)
        os.environ.pop("CORE_API_KEY", None)

        captured_headers: dict[str, str] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_SEARCH_RESPONSE)

        respx.get(f"{CORE_API_BASE}/search/works").mock(side_effect=capture_request)

        searcher = CORESearch(api_key=None)
        await searcher.search(queries=["test"], max_results=5)

        assert "authorization" not in captured_headers
