from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.search.aggregator import (
    SearchAggregator,
    _filter_by_year,
    _merge_papers,
    _normalize_title,
    _parse_date_range,
    _source_rank,
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


class TestSourceRank:
    def test_pubmed_has_highest_priority(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="pubmed")
        assert _source_rank(p) == 0

    def test_europe_pmc_is_second(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="europe_pmc")
        assert _source_rank(p) == 1

    def test_s2_is_third(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="s2")
        assert _source_rank(p) == 2

    def test_openalex_is_fourth(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="openalex")
        assert _source_rank(p) == 3

    def test_core_is_fifth(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="core")
        assert _source_rank(p) == 4

    def test_crossref_is_sixth(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="crossref")
        assert _source_rank(p) == 5

    def test_unknown_source_has_lowest_priority(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="unknown_db")
        assert _source_rank(p) > 5

    def test_case_insensitive(self):
        p = CandidatePaper(title="T", authors=["A"], source_database="PubMed")
        assert _source_rank(p) == 0


class TestMergePapersExternalIds:
    def test_combines_non_overlapping_ids(self):
        """Two papers with completely different IDs should all appear in merged result."""
        primary = CandidatePaper(
            title="Paper X",
            authors=["A"],
            source_database="pubmed",
            external_ids={"pmid": "111", "pmcid": "PMC999"},
        )
        secondary = CandidatePaper(
            title="Paper X",
            authors=["A"],
            source_database="s2",
            external_ids={"s2_id": "abc123", "arxiv": "2301.00001"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["pmid"] == "111"
        assert merged.external_ids["pmcid"] == "PMC999"
        assert merged.external_ids["s2_id"] == "abc123"
        assert merged.external_ids["arxiv"] == "2301.00001"

    def test_primary_ids_win_on_conflict(self):
        """When both records share an ID key, the primary record's value is kept."""
        primary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="pubmed",
            external_ids={"pmid": "PUBMED_ID"},
        )
        secondary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="s2",
            external_ids={"pmid": "S2_PMID_VERSION"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["pmid"] == "PUBMED_ID"

    def test_s2_pdf_url_preserved_when_openalex_wins(self):
        """s2_pdf_url from S2 record should be kept even when OpenAlex record is primary."""
        primary = CandidatePaper(
            title="Some Paper",
            authors=["A"],
            source_database="openalex",
            external_ids={"openalex": "W12345", "pmid": "9999"},
        )
        secondary = CandidatePaper(
            title="Some Paper",
            authors=["A"],
            source_database="s2",
            external_ids={"s2_id": "s2abc", "s2_pdf_url": "https://pdfs.s2.org/paper.pdf"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["openalex"] == "W12345"
        assert merged.external_ids["pmid"] == "9999"
        assert merged.external_ids["s2_id"] == "s2abc"
        assert merged.external_ids["s2_pdf_url"] == "https://pdfs.s2.org/paper.pdf"

    def test_pmcid_preserved_from_pubmed_when_s2_wins(self):
        """pmcid from PubMed should be retained even when S2 record is primary."""
        # s2 has higher priority so it becomes primary, but pmcid from pubmed should survive
        primary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="s2",
            external_ids={"s2_id": "s2xyz", "arxiv": "2301.99999"},
        )
        secondary = CandidatePaper(
            title="T",
            authors=["A"],
            source_database="pubmed",
            external_ids={"pmid": "12345", "pmcid": "PMC67890"},
        )
        merged = _merge_papers(primary, secondary)
        assert merged.external_ids["s2_id"] == "s2xyz"
        assert merged.external_ids["arxiv"] == "2301.99999"
        assert merged.external_ids["pmid"] == "12345"
        assert merged.external_ids["pmcid"] == "PMC67890"


class TestDeduplicateExternalIdMerging:
    def test_dedup_merges_external_ids_by_doi(self):
        """After dedup, the winner should have IDs from both source records."""
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Important Paper",
                authors=["A"],
                doi="10.1234/test",
                source_database="openalex",
                external_ids={"openalex": "W99", "pmid": "55555"},
            ),
            CandidatePaper(
                title="Important Paper",
                authors=["A"],
                doi="10.1234/test",
                source_database="s2",
                external_ids={"s2_id": "s2XYZ", "s2_pdf_url": "https://pdfs.s2.org/x.pdf"},
            ),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        merged = result[0]
        # s2 has higher priority than openalex, so s2 should be primary
        assert merged.source_database == "s2"
        assert merged.external_ids["s2_id"] == "s2XYZ"
        assert merged.external_ids["s2_pdf_url"] == "https://pdfs.s2.org/x.pdf"
        assert merged.external_ids["openalex"] == "W99"
        assert merged.external_ids["pmid"] == "55555"

    def test_dedup_source_priority_pubmed_over_openalex(self):
        """PubMed record should be primary when merged with OpenAlex record."""
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Biomedical Study",
                authors=["B"],
                doi="10.1111/bio",
                source_database="openalex",
                external_ids={"openalex": "W200"},
            ),
            CandidatePaper(
                title="Biomedical Study",
                authors=["B"],
                doi="10.1111/bio",
                source_database="pubmed",
                external_ids={"pmid": "777", "pmcid": "PMC888"},
            ),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        merged = result[0]
        assert merged.source_database == "pubmed"
        assert merged.external_ids["pmid"] == "777"
        assert merged.external_ids["pmcid"] == "PMC888"
        assert merged.external_ids["openalex"] == "W200"

    def test_dedup_merges_external_ids_by_title(self):
        """No-DOI papers matched by title should also get merged external IDs."""
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Preprint Title",
                authors=["C"],
                doi=None,
                source_database="crossref",
                external_ids={"crossref_id": "https://example.com/preprint"},
            ),
            CandidatePaper(
                title="Preprint Title",
                authors=["C"],
                doi=None,
                source_database="s2",
                external_ids={"s2_id": "s2PRE", "arxiv": "2305.01234"},
            ),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        merged = result[0]
        # s2 has higher priority than crossref
        assert merged.source_database == "s2"
        assert merged.external_ids["s2_id"] == "s2PRE"
        assert merged.external_ids["arxiv"] == "2305.01234"
        assert merged.external_ids["crossref_id"] == "https://example.com/preprint"

    def test_dedup_no_cross_population_when_ids_already_present(self):
        """When primary already has all IDs, secondary adds nothing new."""
        agg = SearchAggregator()
        papers = [
            CandidatePaper(
                title="Full Paper",
                authors=["D"],
                doi="10.9999/full",
                source_database="pubmed",
                external_ids={"pmid": "AAA", "pmcid": "PMCBBB", "arxiv": "2201.00000"},
            ),
            CandidatePaper(
                title="Full Paper",
                authors=["D"],
                doi="10.9999/full",
                source_database="openalex",
                external_ids={"openalex": "W333"},
            ),
        ]
        result = agg._deduplicate(papers)
        assert len(result) == 1
        merged = result[0]
        # Original pubmed IDs intact, openalex ID added
        assert merged.external_ids["pmid"] == "AAA"
        assert merged.external_ids["pmcid"] == "PMCBBB"
        assert merged.external_ids["arxiv"] == "2201.00000"
        assert merged.external_ids["openalex"] == "W333"
