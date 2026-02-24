from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.search.aggregator import SearchAggregator, _normalize_title, _merge_papers


class TestNormalizeTitle:
    def test_basic_normalization(self):
        assert _normalize_title("Hello World!") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize_title("A test: with, punct.") == "a test with punct"

    def test_collapses_whitespace(self):
        assert _normalize_title("  too   many   spaces  ") == "too many spaces"

    def test_handles_accents(self):
        result = _normalize_title("café résumé")
        assert "cafe" in result


class TestMergePapers:
    def test_fills_missing_fields(self):
        primary = CandidatePaper(
            title="Test Paper", authors=["A"], doi="10.1234/test",
            source_database="pubmed", abstract=None, year=None,
        )
        secondary = CandidatePaper(
            title="Test Paper", authors=["A", "B"], doi="10.1234/test",
            source_database="s2", abstract="An abstract", year=2023,
            citation_count=10,
        )
        merged = _merge_papers(primary, secondary)
        assert merged.abstract == "An abstract"
        assert merged.year == 2023
        assert merged.citation_count == 10
        assert merged.authors == ["A"]

    def test_merges_external_ids(self):
        primary = CandidatePaper(
            title="T", authors=["A"], source_database="pubmed",
            external_ids={"pmid": "123"},
        )
        secondary = CandidatePaper(
            title="T", authors=["A"], source_database="s2",
            external_ids={"s2_id": "abc"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["pmid"] == "123"
        assert merged.external_ids["s2_id"] == "abc"


class TestDeduplicate:
    def test_dedup_by_doi(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"),
            CandidatePaper(title="Paper A", authors=["X"], doi="10.1234/a", source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1

    def test_dedup_by_title(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(title="Same Paper Title", authors=["X"], doi=None, source_database="pubmed"),
            CandidatePaper(title="same paper title", authors=["X"], doi=None, source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1

    def test_no_doi_not_merged_with_doi_paper(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"),
            CandidatePaper(title="Paper A", authors=["X"], doi=None, source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        assert result[0].doi == "10.1234/a"

    def test_different_papers_kept(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"),
            CandidatePaper(title="Paper B", authors=["Y"], doi="10.1234/b", source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 2


class TestAggregatorSearch:
    async def test_graceful_degradation(self):
        class GoodSource:
            source_name = "good"
            async def search(self, queries, max_results=100):
                return [CandidatePaper(title="Good Paper", authors=["A"], source_database="good")]
            async def get_paper_details(self, paper_id):
                return None

        class BadSource:
            source_name = "bad"
            async def search(self, queries, max_results=100):
                raise ConnectionError("API down")
            async def get_paper_details(self, paper_id):
                return None

        agg = SearchAggregator(sources=[GoodSource(), BadSource()])
        result = await agg.search({"good": ["test"], "bad": ["test"]})
        assert len(result) == 1
        assert result[0].title == "Good Paper"

    async def test_empty_when_no_queries(self):
        agg = SearchAggregator(sources=[])
        result = await agg.search({})
        assert result == []
