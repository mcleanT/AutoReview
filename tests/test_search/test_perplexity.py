"""Tests for PerplexitySearch citation parsing and paper extraction."""

from __future__ import annotations

import httpx
import respx

from autoreview.search.perplexity import (
    PERPLEXITY_API_BASE,
    PerplexitySearch,
    _build_citation_context,
    _extract_doi_from_url,
    _extract_title_from_url,
    _extract_year_from_text,
    _extract_year_from_url,
    _parse_citation_object,
    _parse_citation_string,
)

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestExtractDOIFromURL:
    def test_standard_doi_url(self):
        url = "https://doi.org/10.1038/s41586-023-06112-8"
        assert _extract_doi_from_url(url) == "10.1038/s41586-023-06112-8"

    def test_dx_doi_url(self):
        url = "https://dx.doi.org/10.1016/j.cell.2022.03.001"
        assert _extract_doi_from_url(url) == "10.1016/j.cell.2022.03.001"

    def test_no_doi_in_url(self):
        url = "https://www.nature.com/articles/gut-microbiome-review"
        assert _extract_doi_from_url(url) is None

    def test_doi_with_trailing_period(self):
        url = "https://doi.org/10.1038/s41586-023-06112-8."
        assert _extract_doi_from_url(url) == "10.1038/s41586-023-06112-8"


class TestExtractYearFromURL:
    def test_year_in_path(self):
        url = "https://www.nature.com/articles/2023/some-paper"
        assert _extract_year_from_url(url) == 2023

    def test_year_with_hyphen_separator(self):
        url = "https://journals.plos.org/plosone/2021-article-about-microbiome"
        assert _extract_year_from_url(url) == 2021

    def test_no_year(self):
        url = "https://www.nature.com/articles/some-paper"
        assert _extract_year_from_url(url) is None

    def test_implausible_year_ignored(self):
        url = "https://example.com/data/1234/paper"
        # 1234 is outside 1900-2030 range
        assert _extract_year_from_url(url) is None


class TestExtractYearFromText:
    def test_year_in_parentheses(self):
        assert _extract_year_from_text("Smith et al. (2023) showed that...") == 2023

    def test_year_after_comma(self):
        assert _extract_year_from_text("Published in Nature, 2021.") == 2021

    def test_multiple_years_picks_most_recent(self):
        assert _extract_year_from_text("Since 2015, studies (2020, 2023) have...") == 2023

    def test_no_year(self):
        assert _extract_year_from_text("The results were inconclusive.") is None


class TestExtractTitleFromURL:
    def test_slugified_title(self):
        url = "https://www.nature.com/articles/gut-microbiome-and-neurodegeneration"
        title = _extract_title_from_url(url)
        assert title == "Gut Microbiome And Neurodegeneration"

    def test_underscore_slug(self):
        url = "https://journals.plos.org/plosone/article/role_of_inflammation_in_alzheimers"
        title = _extract_title_from_url(url)
        assert title == "Role Of Inflammation In Alzheimers"

    def test_pure_id_skipped(self):
        url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        assert _extract_title_from_url(url) is None

    def test_short_segment_skipped(self):
        url = "https://example.com/p/abc"
        assert _extract_title_from_url(url) is None

    def test_hex_id_skipped(self):
        url = "https://example.com/papers/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert _extract_title_from_url(url) is None

    def test_html_extension_stripped(self):
        url = "https://example.com/articles/dopamine-signaling-pathways.html"
        title = _extract_title_from_url(url)
        assert title == "Dopamine Signaling Pathways"

    def test_empty_path(self):
        url = "https://example.com/"
        assert _extract_title_from_url(url) is None


class TestBuildCitationContext:
    def test_extracts_context_for_citation(self):
        text = "Gut dysbiosis is linked to PD progression [1]. Another study confirmed this [2]."
        context = _build_citation_context(text)
        assert 1 in context
        assert 2 in context
        assert "Gut dysbiosis" in context[1]
        # Citation markers should be removed from context
        assert "[1]" not in context[1]

    def test_first_mention_wins(self):
        text = "First mention [1]. Second mention of same source [1]."
        context = _build_citation_context(text)
        assert "First mention" in context[1]

    def test_no_citations(self):
        text = "No citations here."
        context = _build_citation_context(text)
        assert context == {}


class TestParseCitationString:
    def test_url_with_title_slug_and_doi(self):
        url = "https://doi.org/10.1038/gut-microbiome-and-neurodegeneration"
        context = {1: "Gut microbiome alterations precede neurodegeneration (2023)."}
        paper = _parse_citation_string(url, 0, context)
        assert paper is not None
        assert paper.doi == "10.1038/gut-microbiome-and-neurodegeneration"
        assert paper.abstract is not None
        assert "Gut microbiome" in paper.abstract
        assert paper.year == 2023

    def test_url_with_title_slug_no_doi(self):
        url = "https://www.nature.com/articles/gut-microbiome-neurodegeneration-review"
        context = {}
        paper = _parse_citation_string(url, 0, context)
        assert paper is not None
        assert "Gut Microbiome" in paper.title
        assert paper.doi is None

    def test_url_only_numeric_id_filtered_out(self):
        url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        context = {}
        paper = _parse_citation_string(url, 0, context)
        assert paper is None

    def test_year_extracted_from_url(self):
        url = "https://www.nature.com/articles/2022/some-important-discovery-in-neuroscience"
        context = {}
        paper = _parse_citation_string(url, 0, context)
        assert paper is not None
        assert paper.year == 2022

    def test_year_falls_back_to_context_text(self):
        url = "https://www.nature.com/articles/some-important-discovery-in-neuroscience"
        context = {1: "Smith et al. (2021) demonstrated results."}
        paper = _parse_citation_string(url, 0, context)
        assert paper is not None
        assert paper.year == 2021

    def test_perplexity_url_in_external_ids(self):
        url = "https://www.nature.com/articles/gut-microbiome-neurodegeneration-review"
        paper = _parse_citation_string(url, 0, {})
        assert paper is not None
        assert paper.external_ids.get("perplexity_url") == url
        assert paper.source_database == "perplexity"


class TestParseCitationObject:
    def test_structured_citation_with_all_fields(self):
        citation = {
            "title": "Gut Microbiome in Alzheimer's Disease",
            "author": "Smith J, Doe A",
            "url": "https://doi.org/10.1038/s41586-023-06112-8",
            "published_date": "2023-06-15",
            "snippet": "This review examines the gut-brain axis...",
        }
        paper = _parse_citation_object(citation, 0, {})
        assert paper is not None
        assert paper.title == "Gut Microbiome in Alzheimer's Disease"
        assert paper.authors == ["Smith J", "Doe A"]
        assert paper.year == 2023
        assert paper.doi == "10.1038/s41586-023-06112-8"
        assert paper.abstract == "This review examines the gut-brain axis..."

    def test_structured_citation_authors_as_list(self):
        citation = {
            "title": "Some Paper",
            "authors": ["Alice A", "Bob B"],
            "url": "https://example.com/articles/some-paper-about-microbiome",
        }
        paper = _parse_citation_object(citation, 0, {})
        assert paper is not None
        assert paper.authors == ["Alice A", "Bob B"]

    def test_structured_citation_year_as_int(self):
        citation = {
            "title": "Some Paper",
            "year": 2022,
            "url": "https://example.com/articles/some-paper-here-about-topic",
        }
        paper = _parse_citation_object(citation, 0, {})
        assert paper is not None
        assert paper.year == 2022

    def test_falls_back_to_url_title(self):
        citation = {
            "url": "https://www.nature.com/articles/gut-microbiome-neurodegeneration-review",
        }
        paper = _parse_citation_object(citation, 0, {})
        assert paper is not None
        assert "Gut Microbiome" in paper.title

    def test_no_title_no_url_returns_none(self):
        citation = {"snippet": "Some text"}
        paper = _parse_citation_object(citation, 0, {})
        assert paper is None

    def test_abstract_falls_back_to_context(self):
        citation = {
            "title": "A Study on Gut Health",
            "url": "https://example.com/articles/a-study-on-gut-health",
        }
        context = {1: "Research shows gut health impacts brain function."}
        paper = _parse_citation_object(citation, 0, context)
        assert paper is not None
        assert paper.abstract == "Research shows gut health impacts brain function."

    def test_doi_from_citation_object(self):
        citation = {
            "title": "Paper With DOI",
            "doi": "10.1038/s41586-023-06112-8",
        }
        paper = _parse_citation_object(citation, 0, {})
        assert paper is not None
        assert paper.doi == "10.1038/s41586-023-06112-8"


# ---------------------------------------------------------------------------
# Integration tests for the full search method
# ---------------------------------------------------------------------------


class TestPerplexitySearchIntegration:
    @respx.mock
    async def test_search_parses_string_citations(self):
        """String citations with parseable URLs produce CandidatePapers."""
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Several studies have explored the gut-brain axis [1]. "
                                    "Another study found inflammation markers [2]. "
                                    "A third source had no useful URL [3]."
                                ),
                            }
                        }
                    ],
                    "citations": [
                        "https://www.nature.com/articles/gut-brain-axis-in-neurodegenerative-diseases",
                        "https://doi.org/10.1016/j.cell.2022.inflammation-markers-in-pd",
                        "https://pubmed.ncbi.nlm.nih.gov/99999999/",
                    ],
                },
            )
        )

        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["gut brain axis"])

        # Citation [3] should be filtered out (numeric-only slug)
        assert len(papers) == 2

        # First paper: title from URL slug, context as abstract
        assert "Gut Brain Axis" in papers[0].title
        assert papers[0].abstract is not None
        assert "gut-brain axis" in papers[0].abstract
        assert papers[0].source_database == "perplexity"

        # Second paper: DOI extracted from URL
        assert papers[1].doi is not None
        assert "10.1016" in papers[1].doi

    @respx.mock
    async def test_search_parses_dict_citations(self):
        """Structured citation objects with metadata produce richer CandidatePapers."""
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Relevant research includes [1].",
                            }
                        }
                    ],
                    "citations": [
                        {
                            "title": "The Gut-Brain Connection",
                            "author": "Smith J, Doe A",
                            "url": "https://doi.org/10.1038/s41586-2023-00001",
                            "published_date": "2023-03-15",
                            "snippet": "A comprehensive review of gut-brain interactions.",
                        }
                    ],
                },
            )
        )

        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["gut brain connection"])

        assert len(papers) == 1
        p = papers[0]
        assert p.title == "The Gut-Brain Connection"
        assert p.authors == ["Smith J", "Doe A"]
        assert p.year == 2023
        assert p.doi is not None
        assert p.abstract == "A comprehensive review of gut-brain interactions."

    @respx.mock
    async def test_search_filters_unparseable_citations(self):
        """Citations with no extractable title are filtered out."""
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Some text [1] [2]."}}],
                    "citations": [
                        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                        "https://example.com/p/x",
                    ],
                },
            )
        )

        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["test query"])
        assert len(papers) == 0

    @respx.mock
    async def test_search_handles_http_error(self):
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(return_value=httpx.Response(429))
        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["test query"])
        assert papers == []

    @respx.mock
    async def test_search_handles_missing_api_key(self):
        search = PerplexitySearch(api_key=None)
        # Also unset the env var
        import os

        original = os.environ.pop("PERPLEXITY_API_KEY", None)
        try:
            papers = await search.search(["test query"])
            assert papers == []
        finally:
            if original is not None:
                os.environ["PERPLEXITY_API_KEY"] = original

    @respx.mock
    async def test_search_respects_max_results(self):
        """The max_results parameter limits returned papers."""
        citations = [
            f"https://example.com/articles/paper-about-topic-number-{i:03d}-and-more"
            for i in range(20)
        ]
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Many papers found."}}],
                    "citations": citations,
                },
            )
        )

        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["test query"], max_results=5)
        assert len(papers) <= 5

    @respx.mock
    async def test_search_handles_empty_citations(self):
        respx.post(f"{PERPLEXITY_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "No papers found."}}],
                    "citations": [],
                },
            )
        )

        search = PerplexitySearch(api_key="test-key")
        papers = await search.search(["obscure topic"])
        assert papers == []
