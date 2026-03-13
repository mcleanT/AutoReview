"""Tests for PubMedSearch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from autoreview.search.pubmed import PubMedSearch


def _make_string_element(value: str, attributes: dict | None = None):
    """Create a mock Bio.Entrez StringElement-like object with .attributes."""
    s = MagicMock(spec=str)
    s.__str__ = lambda self: value
    s.attributes = attributes or {}
    # Make it behave like a string in join/concatenation
    s.__repr__ = lambda self: value
    return s


def _make_article(
    *,
    title: str = "Test Paper Title",
    authors: list[dict] | None = None,
    year: str = "2023",
    journal: str = "Nature Reviews",
    abstract_parts: list[str] | None = None,
    doi: str | None = "10.1038/test-doi",
    pmid: str = "12345678",
    pmcid: str | None = None,
    elocation_doi: bool = True,
) -> dict:
    """Build a mock PubMed article dict matching Bio.Entrez.read() structure."""
    if authors is None:
        authors = [
            {"LastName": "Smith", "ForeName": "John"},
            {"LastName": "Doe", "ForeName": "Alice"},
        ]

    article_data: dict = {
        "ArticleTitle": title,
        "AuthorList": authors,
        "Journal": {
            "Title": journal,
            "JournalIssue": {"PubDate": {"Year": year}},
        },
    }

    if abstract_parts is not None:
        article_data["Abstract"] = {"AbstractText": abstract_parts}

    if doi and elocation_doi:
        doi_elem = _make_string_element(doi, {"EIdType": "doi"})
        article_data["ELocationID"] = [doi_elem]
    else:
        article_data["ELocationID"] = []

    article_id_list = []
    if doi and not elocation_doi:
        doi_aid = _make_string_element(doi, {"IdType": "doi"})
        article_id_list.append(doi_aid)
    if pmcid:
        pmc_aid = _make_string_element(pmcid, {"IdType": "pmc"})
        article_id_list.append(pmc_aid)

    return {
        "MedlineCitation": {
            "PMID": pmid,
            "Article": article_data,
        },
        "PubmedData": {
            "ArticleIdList": article_id_list,
        },
    }


class TestParseArticle:
    """Test _parse_article parsing logic directly."""

    def test_parses_complete_article(self):
        article = _make_article(
            title="Gut Microbiome and Neurodegeneration",
            authors=[
                {"LastName": "Smith", "ForeName": "John"},
                {"LastName": "Zhang", "ForeName": "Li"},
            ],
            year="2023",
            journal="Nature Reviews Neuroscience",
            abstract_parts=["Background: The gut-brain axis.", " Methods: We reviewed."],
            doi="10.1038/s41583-023-0001",
            pmid="99887766",
        )
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.title == "Gut Microbiome and Neurodegeneration"
        assert paper.authors == ["Smith John", "Zhang Li"]
        assert paper.year == 2023
        assert paper.journal == "Nature Reviews Neuroscience"
        assert paper.doi == "10.1038/s41583-023-0001"
        assert paper.abstract == "Background: The gut-brain axis.  Methods: We reviewed."
        assert paper.source_database == "pubmed"
        assert paper.external_ids["pmid"] == "99887766"

    def test_missing_doi(self):
        article = _make_article(doi=None)
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.doi is None

    def test_doi_from_article_id_list_fallback(self):
        """If DOI not in ELocationID, fall back to ArticleIdList."""
        article = _make_article(doi="10.1234/fallback-doi", elocation_doi=False)
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.doi == "10.1234/fallback-doi"

    def test_missing_abstract(self):
        article = _make_article(abstract_parts=None)
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.abstract is None

    def test_empty_abstract(self):
        article = _make_article(abstract_parts=[])
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.abstract is None

    def test_missing_title_returns_none(self):
        article = _make_article(title="")
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is None

    def test_missing_year(self):
        article = _make_article()
        # Remove Year from PubDate
        article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"] = {}
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.year is None

    def test_invalid_year(self):
        article = _make_article(year="not-a-year")
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.year is None

    def test_missing_authors(self):
        article = _make_article(authors=[])
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.authors == []

    def test_author_last_name_only(self):
        article = _make_article(authors=[{"LastName": "Consortium"}])
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.authors == ["Consortium"]

    def test_pmcid_extraction(self):
        article = _make_article(pmcid="PMC9876543")
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article(article)

        assert paper is not None
        assert paper.external_ids.get("pmcid") == "PMC9876543"

    def test_malformed_article_returns_none(self):
        """A completely broken article dict should not raise, just return None."""
        search = PubMedSearch(email="test@test.com")
        paper = search._parse_article({})

        # Empty dict has no title so returns None
        assert paper is None


class TestPubMedSearch:
    """Test the async search method with mocked Entrez calls."""

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    @patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
    async def test_search_returns_papers(self, mock_search, mock_fetch):
        mock_search.return_value = ["111", "222"]
        mock_fetch.return_value = [
            _make_article(title="Paper One", pmid="111", doi="10.1/one"),
            _make_article(title="Paper Two", pmid="222", doi="10.1/two"),
        ]

        search = PubMedSearch(email="test@test.com")
        papers = await search.search(["gut microbiome"], max_results=10)

        assert len(papers) == 2
        assert papers[0].title == "Paper One"
        assert papers[1].title == "Paper Two"
        mock_search.assert_called_once()
        mock_fetch.assert_called_once()

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    @patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
    async def test_search_empty_results(self, mock_search, mock_fetch):
        mock_search.return_value = []

        search = PubMedSearch(email="test@test.com")
        papers = await search.search(["nonexistent topic"], max_results=10)

        assert papers == []
        mock_fetch.assert_not_called()

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    @patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
    async def test_search_deduplicates_pmids(self, mock_search, mock_fetch):
        """Same PMID from multiple queries should only be fetched once."""
        mock_search.side_effect = [["111", "222"], ["222", "333"]]
        mock_fetch.return_value = [
            _make_article(title="Paper A", pmid="111"),
            _make_article(title="Paper B", pmid="222"),
            _make_article(title="Paper C", pmid="333"),
        ]

        search = PubMedSearch(email="test@test.com")
        await search.search(["query1", "query2"], max_results=100)

        # _sync_fetch should receive unique PMIDs only
        fetch_call_args = mock_fetch.call_args[0][0]
        assert fetch_call_args == ["111", "222", "333"]

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    @patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
    async def test_search_skips_unparseable_articles(self, mock_search, mock_fetch):
        mock_search.return_value = ["111", "222"]
        mock_fetch.return_value = [
            _make_article(title="Good Paper", pmid="111"),
            _make_article(title="", pmid="222"),  # empty title -> None
        ]

        search = PubMedSearch(email="test@test.com")
        papers = await search.search(["test query"], max_results=10)

        assert len(papers) == 1
        assert papers[0].title == "Good Paper"

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    @patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
    async def test_search_multiple_queries(self, mock_search, mock_fetch):
        mock_search.side_effect = [["111"], ["222"]]
        mock_fetch.return_value = [
            _make_article(title="P1", pmid="111"),
            _make_article(title="P2", pmid="222"),
        ]

        search = PubMedSearch(email="test@test.com")
        papers = await search.search(["q1", "q2"], max_results=50)

        assert mock_search.call_count == 2
        assert len(papers) == 2


class TestPubMedGetPaperDetails:
    """Test get_paper_details."""

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    async def test_returns_paper(self, mock_fetch):
        mock_fetch.return_value = [_make_article(title="Detailed Paper", pmid="999")]

        search = PubMedSearch(email="test@test.com")
        paper = await search.get_paper_details("999")

        assert paper is not None
        assert paper.title == "Detailed Paper"

    @patch("autoreview.search.pubmed.PubMedSearch._sync_fetch_with_retry")
    async def test_returns_none_for_empty(self, mock_fetch):
        mock_fetch.return_value = []

        search = PubMedSearch(email="test@test.com")
        paper = await search.get_paper_details("nonexistent")

        assert paper is None


class TestPubMedInit:
    """Test initialization and configuration."""

    def test_source_name(self):
        search = PubMedSearch(email="test@test.com")
        assert search.source_name == "pubmed"

    def test_rate_limit_with_api_key(self):
        search = PubMedSearch(email="test@test.com", api_key="my-key")
        assert search._limiter.requests_per_second == 10.0

    def test_rate_limit_without_api_key(self):
        search = PubMedSearch(email="test@test.com")
        assert search._limiter.requests_per_second == 3.0

    @patch.dict("os.environ", {"ENTREZ_EMAIL": "env@test.com"})
    def test_email_from_env(self):
        search = PubMedSearch()
        assert search._email == "env@test.com"

    def test_default_email(self):
        search = PubMedSearch()
        assert search._email == "autoreview@example.com"


class TestSyncSearch:
    """Test _sync_search_with_retry with mocked Entrez."""

    @patch("autoreview.search.pubmed.PubMedSearch._setup_entrez")
    def test_sync_search_returns_id_list(self, mock_setup):
        mock_entrez = MagicMock()
        mock_handle = MagicMock()
        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = {"IdList": ["111", "222", "333"]}
        mock_setup.return_value = mock_entrez

        search = PubMedSearch(email="test@test.com")
        result = search._sync_search_with_retry("test query", 100)

        assert result == ["111", "222", "333"]
        mock_entrez.esearch.assert_called_once_with(
            db="pubmed", term="test query", retmax=100, sort="relevance"
        )
        mock_handle.close.assert_called_once()

    @patch("autoreview.search.pubmed.PubMedSearch._setup_entrez")
    def test_sync_search_empty_results(self, mock_setup):
        mock_entrez = MagicMock()
        mock_handle = MagicMock()
        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = {"IdList": []}
        mock_setup.return_value = mock_entrez

        search = PubMedSearch(email="test@test.com")
        result = search._sync_search_with_retry("nonexistent", 100)

        assert result == []


class TestSyncFetch:
    """Test _sync_fetch_with_retry with mocked Entrez."""

    @patch("autoreview.search.pubmed.PubMedSearch._setup_entrez")
    def test_sync_fetch_returns_articles(self, mock_setup):
        mock_entrez = MagicMock()
        mock_handle = MagicMock()
        mock_entrez.efetch.return_value = mock_handle
        mock_entrez.read.return_value = {
            "PubmedArticle": [{"article": 1}, {"article": 2}]
        }
        mock_setup.return_value = mock_entrez

        search = PubMedSearch(email="test@test.com")
        result = search._sync_fetch_with_retry(["111", "222"])

        assert len(result) == 2
        mock_entrez.efetch.assert_called_once()
        mock_handle.close.assert_called_once()

    @patch("autoreview.search.pubmed.PubMedSearch._setup_entrez")
    def test_sync_fetch_empty_pmids(self, mock_setup):
        search = PubMedSearch(email="test@test.com")
        result = search._sync_fetch_with_retry([])

        assert result == []
        mock_setup.assert_not_called()

    @patch("autoreview.search.pubmed.PubMedSearch._setup_entrez")
    def test_sync_fetch_batches_large_lists(self, mock_setup):
        """PMIDs should be fetched in batches of 200."""
        mock_entrez = MagicMock()
        mock_handle = MagicMock()
        mock_entrez.efetch.return_value = mock_handle
        mock_entrez.read.return_value = {"PubmedArticle": []}
        mock_setup.return_value = mock_entrez

        search = PubMedSearch(email="test@test.com")
        pmids = [str(i) for i in range(350)]
        search._sync_fetch_with_retry(pmids)

        # 350 PMIDs => 2 batches (200 + 150)
        assert mock_entrez.efetch.call_count == 2
