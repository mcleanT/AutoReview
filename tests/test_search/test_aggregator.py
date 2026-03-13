from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.search.aggregator import (
    SearchAggregator,
    _filter_by_year,
    _merge_papers,
    _normalize_title,
    _parse_date_range,
)


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
            title="Test Paper",
            authors=["A"],
            doi="10.1234/test",
            source_database="pubmed",
            abstract=None,
            year=None,
        )
        secondary = CandidatePaper(
            title="Test Paper",
            authors=["A", "B"],
            doi="10.1234/test",
            source_database="s2",
            abstract="An abstract",
            year=2023,
            citation_count=10,
        )
        merged = _merge_papers(primary, secondary)
        assert merged.abstract == "An abstract"
        assert merged.year == 2023
        assert merged.citation_count == 10
        assert merged.authors == ["A"]

    def test_merges_external_ids(self):
        primary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="pubmed",
            external_ids={"pmid": "123"},
        )
        secondary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="s2",
            external_ids={"s2_id": "abc"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["pmid"] == "123"
        assert merged.external_ids["s2_id"] == "abc"


class TestDeduplicate:
    def test_dedup_by_doi(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"
            ),
            CandidatePaper(title="Paper A", authors=["X"], doi="10.1234/a", source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1

    def test_dedup_by_title(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Same Paper Title", authors=["X"], doi=None, source_database="pubmed"
            ),
            CandidatePaper(title="same paper title", authors=["X"], doi=None, source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1

    def test_no_doi_not_merged_with_doi_paper(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"
            ),
            CandidatePaper(title="Paper A", authors=["X"], doi=None, source_database="s2"),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        assert result[0].doi == "10.1234/a"

    def test_different_papers_kept(self):
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Paper A", authors=["X"], doi="10.1234/a", source_database="pubmed"
            ),
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


class TestParseDateRange:
    def test_full_range(self):
        assert _parse_date_range("2015-2025") == (2015, 2025)

    def test_open_start(self):
        assert _parse_date_range("-2020") == (None, 2020)

    def test_open_end(self):
        assert _parse_date_range("2020-") == (2020, None)

    def test_none(self):
        assert _parse_date_range(None) == (None, None)

    def test_empty_string(self):
        assert _parse_date_range("") == (None, None)

    def test_same_year(self):
        assert _parse_date_range("2020-2020") == (2020, 2020)


class TestFilterByYear:
    def _make_paper(self, title: str, year: int | None) -> CandidatePaper:
        return CandidatePaper(
            title=title,
            authors=["A"],
            source_database="test",
            year=year,
        )

    def test_filters_below_range(self):
        papers = [self._make_paper("Old", 2010), self._make_paper("In", 2018)]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 1
        assert result[0].title == "In"

    def test_filters_above_range(self):
        papers = [self._make_paper("Future", 2030), self._make_paper("In", 2018)]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 1
        assert result[0].title == "In"

    def test_inclusive_bounds(self):
        papers = [
            self._make_paper("Start", 2015),
            self._make_paper("End", 2020),
            self._make_paper("Mid", 2018),
        ]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 3

    def test_drops_year_none(self):
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=2015, year_to=2025)
        assert len(result) == 1
        assert result[0].title == "Has Year"

    def test_drops_year_none_even_without_range(self):
        """year=None always dropped when any date_range is set."""
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=None, year_to=2025)
        assert len(result) == 1

    def test_no_filtering_when_both_none(self):
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=None, year_to=None)
        assert len(result) == 2

    def test_open_end_range(self):
        papers = [self._make_paper("Old", 2010), self._make_paper("New", 2025)]
        result = _filter_by_year(papers, year_from=2020, year_to=None)
        assert len(result) == 1
        assert result[0].title == "New"


class TestAggregatorSearchWithYearFilter:
    @pytest.mark.asyncio
    async def test_filters_papers_by_year(self):
        class MockSource:
            source_name = "mock"

            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(title="Old", authors=["A"], source_database="mock", year=2010),
                    CandidatePaper(
                        title="In Range", authors=["A"], source_database="mock", year=2018
                    ),
                    CandidatePaper(
                        title="Future", authors=["A"], source_database="mock", year=2030
                    ),
                ]

        agg = SearchAggregator(sources=[MockSource()], date_range="2015-2020")
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 1
        assert result[0].title == "In Range"

    @pytest.mark.asyncio
    async def test_no_filter_without_date_range(self):
        class MockSource:
            source_name = "mock"

            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(title="A", authors=["A"], source_database="mock", year=2010),
                    CandidatePaper(title="B", authors=["A"], source_database="mock", year=2020),
                ]

        agg = SearchAggregator(sources=[MockSource()])
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_drops_null_year_with_date_range(self):
        class MockSource:
            source_name = "mock"

            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(
                        title="No Year", authors=["A"], source_database="mock", year=None
                    ),
                    CandidatePaper(
                        title="Has Year", authors=["A"], source_database="mock", year=2018
                    ),
                ]

        agg = SearchAggregator(sources=[MockSource()], date_range="2015-2020")
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 1
        assert result[0].title == "Has Year"
