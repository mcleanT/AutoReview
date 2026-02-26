"""Tests for OpenAlexSearch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoreview.search.openalex import OpenAlexSearch


def _make_work(
    *,
    title: str = "Test OpenAlex Paper",
    authorships: list[dict] | None = None,
    year: int = 2023,
    journal_name: str | None = "Journal of Testing",
    doi: str | None = "https://doi.org/10.1234/test",
    abstract_inverted_index: dict | None = None,
    openalex_id: str = "https://openalex.org/W1234",
    pmid: str | None = None,
    cited_by_count: int = 10,
) -> dict:
    """Build a mock OpenAlex work dict."""
    if authorships is None:
        authorships = [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ]

    work: dict = {
        "title": title,
        "authorships": authorships,
        "publication_year": year,
        "doi": doi,
        "id": openalex_id,
        "cited_by_count": cited_by_count,
        "abstract_inverted_index": abstract_inverted_index,
    }

    ids: dict = {}
    if pmid:
        ids["pmid"] = pmid
    work["ids"] = ids

    primary_location: dict = {}
    if journal_name:
        primary_location["source"] = {"display_name": journal_name}
    else:
        primary_location["source"] = None
    work["primary_location"] = primary_location

    return work


class TestParseWork:
    """Test _parse_work parsing logic."""

    def test_parses_complete_work(self):
        work = _make_work(
            title="Deep Learning for Drug Discovery",
            authorships=[
                {"author": {"display_name": "Alice Smith"}},
                {"author": {"display_name": "Carol Lee"}},
            ],
            year=2024,
            journal_name="Nature Machine Intelligence",
            doi="https://doi.org/10.1038/nmi-2024-001",
            cited_by_count=55,
            openalex_id="https://openalex.org/W9999",
        )
        search = OpenAlexSearch(email="test@test.com")
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.title == "Deep Learning for Drug Discovery"
        assert paper.authors == ["Alice Smith", "Carol Lee"]
        assert paper.year == 2024
        assert paper.journal == "Nature Machine Intelligence"
        assert paper.doi == "10.1038/nmi-2024-001"
        assert paper.citation_count == 55
        assert paper.source_database == "openalex"
        assert paper.external_ids["openalex"] == "https://openalex.org/W9999"

    def test_doi_prefix_stripped(self):
        work = _make_work(doi="https://doi.org/10.5555/test123")
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.doi == "10.5555/test123"

    def test_doi_without_prefix_kept(self):
        work = _make_work(doi="10.5555/test123")
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        # CandidatePaper normalizes DOI (lowercase, strip url prefix)
        assert paper.doi == "10.5555/test123"

    def test_missing_doi(self):
        work = _make_work(doi=None)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.doi is None

    def test_missing_title_returns_none(self):
        work = _make_work(title=None)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is None

    def test_empty_title_returns_none(self):
        work = _make_work()
        work["title"] = ""
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is None

    def test_missing_authors(self):
        work = _make_work(authorships=[])
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.authors == []

    def test_author_without_display_name_skipped(self):
        work = _make_work(
            authorships=[
                {"author": {"display_name": "Good Author"}},
                {"author": {"display_name": None}},
                {"author": {}},
            ]
        )
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.authors == ["Good Author"]

    def test_missing_journal(self):
        work = _make_work(journal_name=None)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.journal is None

    def test_pmid_extraction(self):
        work = _make_work(pmid="https://pubmed.ncbi.nlm.nih.gov/12345678")
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.external_ids.get("pmid") == "12345678"

    def test_pmid_without_url_prefix(self):
        work = _make_work(pmid="12345678")
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.external_ids.get("pmid") == "12345678"

    def test_missing_year(self):
        work = _make_work()
        work["publication_year"] = None
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.year is None


class TestAbstractReconstruction:
    """Test the inverted abstract index reconstruction — the key non-trivial logic."""

    def test_simple_inverted_index(self):
        inv_index = {
            "The": [0],
            "gut": [1],
            "microbiome": [2],
            "is": [3],
            "important.": [4],
        }
        work = _make_work(abstract_inverted_index=inv_index)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract == "The gut microbiome is important."

    def test_inverted_index_with_repeated_words(self):
        inv_index = {
            "The": [0, 4],
            "study": [1, 5],
            "examined": [2],
            "results.": [3],
            "confirmed": [6],
            "findings.": [7],
        }
        work = _make_work(abstract_inverted_index=inv_index)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract == "The study examined results. The study confirmed findings."

    def test_inverted_index_with_many_positions(self):
        """Words appearing at many positions should all be placed correctly."""
        inv_index = {
            "a": [0, 2, 4],
            "b": [1, 3, 5],
        }
        work = _make_work(abstract_inverted_index=inv_index)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract == "a b a b a b"

    def test_no_inverted_index(self):
        work = _make_work(abstract_inverted_index=None)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract is None

    def test_empty_inverted_index(self):
        work = _make_work(abstract_inverted_index={})
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        # Empty dict is falsy, so the reconstruction branch is skipped
        assert paper.abstract is None

    def test_single_word_abstract(self):
        inv_index = {"Abstract": [0]}
        work = _make_work(abstract_inverted_index=inv_index)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract == "Abstract"

    def test_realistic_abstract(self):
        """Test with a realistic multi-sentence inverted index."""
        inv_index = {
            "Background:": [0],
            "This": [1, 7],
            "review": [2],
            "examines": [3],
            "recent": [4],
            "advances.": [5],
            "Methods:": [6],
            "study": [8],
            "uses": [9],
            "meta-analysis.": [10],
        }
        work = _make_work(abstract_inverted_index=inv_index)
        search = OpenAlexSearch()
        paper = search._parse_work(work)

        assert paper is not None
        assert paper.abstract == (
            "Background: This review examines recent advances. "
            "Methods: This study uses meta-analysis."
        )


class TestOpenAlexSearch:
    """Test the async search method with mocked pyalex calls."""

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_returns_papers(self, mock_sync_search):
        mock_sync_search.return_value = [
            _make_work(title="Paper One", doi="https://doi.org/10.1/one"),
            _make_work(title="Paper Two", doi="https://doi.org/10.1/two"),
        ]

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["machine learning"], max_results=10)

        assert len(papers) == 2
        assert papers[0].title == "Paper One"
        assert papers[1].title == "Paper Two"

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_empty_results(self, mock_sync_search):
        mock_sync_search.return_value = []

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["nonexistent"], max_results=10)

        assert papers == []

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_skips_unparseable_works(self, mock_sync_search):
        mock_sync_search.return_value = [
            _make_work(title="Good Paper"),
            _make_work(title=None),  # no title -> None
        ]

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["test"], max_results=10)

        assert len(papers) == 1
        assert papers[0].title == "Good Paper"

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_respects_max_results(self, mock_sync_search):
        mock_sync_search.return_value = [
            _make_work(title=f"Paper {i}") for i in range(20)
        ]

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["test"], max_results=5)

        assert len(papers) <= 5

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_handles_api_error(self, mock_sync_search):
        mock_sync_search.side_effect = Exception("API Error")

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["test query"], max_results=10)

        # Should gracefully handle the error and return empty
        assert papers == []

    @patch("autoreview.search.openalex.OpenAlexSearch._sync_search_with_retry")
    async def test_search_multiple_queries(self, mock_sync_search):
        mock_sync_search.side_effect = [
            [_make_work(title="From Query 1")],
            [_make_work(title="From Query 2")],
        ]

        search = OpenAlexSearch(email="test@test.com")
        papers = await search.search(["q1", "q2"], max_results=50)

        assert len(papers) == 2
        assert mock_sync_search.call_count == 2


class TestOpenAlexGetPaperDetails:
    """Test get_paper_details.

    pyalex is imported locally inside get_paper_details, so we inject a
    fake pyalex module into sys.modules before calling the method.
    """

    async def test_returns_paper(self):
        import sys

        work_data = _make_work(title="Detailed Work")
        mock_works_instance = MagicMock()
        mock_works_instance.__getitem__ = MagicMock(return_value=work_data)
        mock_works_cls = MagicMock(return_value=mock_works_instance)

        fake_pyalex = MagicMock()
        fake_pyalex.Works = mock_works_cls

        search = OpenAlexSearch(email="test@test.com")
        original = sys.modules.get("pyalex")
        sys.modules["pyalex"] = fake_pyalex
        try:
            paper = await search.get_paper_details("W1234")
        finally:
            if original is not None:
                sys.modules["pyalex"] = original
            else:
                sys.modules.pop("pyalex", None)

        assert paper is not None
        assert paper.title == "Detailed Work"
        assert paper.source_database == "openalex"

    async def test_returns_none_on_error(self):
        import sys

        mock_works_instance = MagicMock()
        mock_works_instance.__getitem__ = MagicMock(side_effect=Exception("Not found"))
        mock_works_cls = MagicMock(return_value=mock_works_instance)

        fake_pyalex = MagicMock()
        fake_pyalex.Works = mock_works_cls

        search = OpenAlexSearch(email="test@test.com")
        original = sys.modules.get("pyalex")
        sys.modules["pyalex"] = fake_pyalex
        try:
            paper = await search.get_paper_details("W_nonexistent")
        finally:
            if original is not None:
                sys.modules["pyalex"] = original
            else:
                sys.modules.pop("pyalex", None)

        assert paper is None


class TestOpenAlexInit:
    """Test initialization."""

    def test_source_name(self):
        search = OpenAlexSearch()
        assert search.source_name == "openalex"

    @patch.dict("os.environ", {"OPENALEX_EMAIL": "env@test.com"})
    def test_email_from_env(self):
        search = OpenAlexSearch()
        assert search._email == "env@test.com"

    def test_explicit_email(self):
        search = OpenAlexSearch(email="explicit@test.com")
        assert search._email == "explicit@test.com"
