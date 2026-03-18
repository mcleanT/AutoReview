"""Tests for Europe PMC search backend."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoreview.models.paper import CandidatePaper
from autoreview.search.europe_pmc import EUROPE_PMC_API_BASE, EuropePMCSearch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_RESULT = {
    "id": "33456789",
    "title": "Mechanisms of CRISPR-Cas9 genome editing",
    "authorString": "Zhang F, Li W, Chen X.",
    "pubYear": "2021",
    "doi": "10.1038/s41586-021-00001-1",
    "abstractText": "We describe the molecular mechanisms underlying CRISPR-Cas9 genome editing.",
    "pmid": "33456789",
    "pmcid": "PMC8765432",
    "isOpenAccess": "Y",
    "citedByCount": 142,
    "journalTitle": "Nature",
}

_SAMPLE_RESULT_2 = {
    "id": "34567890",
    "title": "Deep learning for protein structure prediction",
    "authorString": "Jumper J, Evans R.",
    "pubYear": "2021",
    "doi": "10.1038/s41586-021-03819-2",
    "abstractText": "We present AlphaFold2, a deep learning method for protein structure prediction.",
    "pmid": "34567890",
    "pmcid": "PMC8371605",
    "isOpenAccess": "Y",
    "citedByCount": 5432,
    "journalTitle": "Nature",
}

_SEARCH_RESPONSE = {
    "hitCount": 2,
    "nextCursorMark": "*",
    "resultList": {
        "result": [_SAMPLE_RESULT, _SAMPLE_RESULT_2],
    },
}

_SEARCH_RESPONSE_SINGLE = {
    "hitCount": 1,
    "nextCursorMark": "*",
    "resultList": {
        "result": [_SAMPLE_RESULT],
    },
}

_EMPTY_RESPONSE = {
    "hitCount": 0,
    "nextCursorMark": "*",
    "resultList": {
        "result": [],
    },
}


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestEuropePMCSearchSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_candidate_papers(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert len(papers) == 2
        assert all(isinstance(p, CandidatePaper) for p in papers)

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_fields_correctly(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        paper = papers[0]
        assert paper.title == "Mechanisms of CRISPR-Cas9 genome editing"
        assert paper.year == 2021
        assert paper.doi == "10.1038/s41586-021-00001-1"
        assert (
            paper.abstract
            == "We describe the molecular mechanisms underlying CRISPR-Cas9 genome editing."
        )
        assert paper.source_database == "europe_pmc"
        assert paper.journal == "Nature"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_parses_authors_from_author_string(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        paper = papers[0]
        assert "Zhang F" in paper.authors
        assert "Li W" in paper.authors
        assert "Chen X" in paper.authors

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_pmid_in_external_ids(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers[0].external_ids.get("pmid") == "33456789"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_pmcid_in_external_ids(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers[0].external_ids.get("pmcid") == "PMC8765432"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_stores_europepmc_id_in_external_ids(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers[0].external_ids.get("europepmc") == "33456789"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_maps_citation_count(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE_SINGLE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers[0].citation_count == 142

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_http_error_gracefully(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_empty_results(self):
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=_EMPTY_RESPONSE)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["nonexistent xyzabc"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_skips_result_without_title(self):
        no_title_result = dict(_SAMPLE_RESULT)
        no_title_result["title"] = None
        response = {
            "hitCount": 1,
            "nextCursorMark": "*",
            "resultList": {"result": [no_title_result]},
        }
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["CRISPR"], max_results=10)

        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_handles_missing_optional_fields(self):
        minimal_result = {
            "id": "99999",
            "title": "Minimal Paper Title",
            "authorString": "",
            "pubYear": None,
            "doi": None,
            "abstractText": None,
            "isOpenAccess": "N",
            "citedByCount": None,
        }
        response = {
            "hitCount": 1,
            "nextCursorMark": "*",
            "resultList": {"result": [minimal_result]},
        }
        respx.get(f"{EUROPE_PMC_API_BASE}/search").mock(
            return_value=httpx.Response(200, json=response)
        )

        searcher = EuropePMCSearch()
        papers = await searcher.search(queries=["test"], max_results=5)

        assert len(papers) == 1
        assert papers[0].title == "Minimal Paper Title"
        assert papers[0].year is None
        assert papers[0].doi is None
        assert papers[0].abstract is None
        assert papers[0].citation_count is None


# ---------------------------------------------------------------------------
# source_name test
# ---------------------------------------------------------------------------


class TestEuropePMCSourceName:
    def test_source_name(self):
        searcher = EuropePMCSearch()
        assert searcher.source_name == "europe_pmc"
