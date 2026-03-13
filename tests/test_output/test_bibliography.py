"""Tests for bibliography formatting."""

from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.output.bibliography import BibliographyFormatter


@pytest.fixture
def paper_single_author():
    return CandidatePaper(
        title="Single Author Study",
        authors=["Smith J"],
        year=2023,
        journal="Nature",
        doi="10.1038/test001",
        source_database="pubmed",
    )


@pytest.fixture
def paper_two_authors():
    return CandidatePaper(
        title="Two Author Study",
        authors=["Smith J", "Doe A"],
        year=2022,
        journal="Science",
        doi="10.1126/test002",
        source_database="pubmed",
    )


@pytest.fixture
def paper_many_authors():
    return CandidatePaper(
        title="Many Author Study",
        authors=[f"Author{i}" for i in range(25)],
        year=2021,
        journal="Cell",
        doi="10.1016/test003",
        source_database="pubmed",
    )


@pytest.fixture
def paper_no_optional():
    return CandidatePaper(
        title="Minimal Paper",
        authors=["Unknown A"],
        source_database="pubmed",
    )


class TestAPAFormatting:
    def test_single_author(self, paper_single_author):
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper_single_author)
        assert "Smith J" in entry
        assert "(2023)" in entry
        assert "Single Author Study" in entry
        assert "*Nature*" in entry
        assert "doi.org/10.1038/test001" in entry

    def test_two_authors(self, paper_two_authors):
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper_two_authors)
        assert "Smith J & Doe A" in entry

    def test_many_authors_truncated(self, paper_many_authors):
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper_many_authors)
        # 20+ authors: first 19, ..., last
        assert "..." in entry
        assert "Author24" in entry  # last author

    def test_no_year(self, paper_no_optional):
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper_no_optional)
        assert "(n.d.)" in entry

    def test_no_journal(self, paper_no_optional):
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper_no_optional)
        # Should not crash, just omit journal
        assert "Minimal Paper" in entry

    def test_no_authors(self):
        paper = CandidatePaper(
            title="No Authors",
            authors=[],
            source_database="pubmed",
        )
        fmt = BibliographyFormatter(style="apa")
        entry = fmt.format_entry(paper)
        assert "Unknown" in entry


class TestVancouverFormatting:
    def test_basic_entry(self, paper_single_author):
        fmt = BibliographyFormatter(style="vancouver")
        entry = fmt.format_entry(paper_single_author)
        assert "Smith J" in entry
        assert "2023" in entry
        assert "doi:10.1038/test001" in entry

    def test_many_authors_et_al(self, paper_many_authors):
        fmt = BibliographyFormatter(style="vancouver")
        entry = fmt.format_entry(paper_many_authors)
        assert "et al" in entry

    def test_numbered_bibliography(self, paper_single_author, paper_two_authors):
        fmt = BibliographyFormatter(style="vancouver")
        bib = fmt.format_bibliography(
            [paper_single_author, paper_two_authors],
            cited_ids=[paper_single_author.id, paper_two_authors.id],
        )
        assert bib.startswith("1. ")
        assert "2. " in bib


class TestACSFormatting:
    def test_basic_entry(self, paper_single_author):
        fmt = BibliographyFormatter(style="acs")
        entry = fmt.format_entry(paper_single_author)
        assert "Smith J" in entry
        assert "**2021**" in entry or "**2023**" in entry
        assert "DOI:" in entry


class TestBibliographySection:
    def test_format_bibliography_with_cited_ids(self, paper_single_author, paper_two_authors):
        fmt = BibliographyFormatter(style="apa")
        # Order by cited_ids
        bib = fmt.format_bibliography(
            [paper_single_author, paper_two_authors],
            cited_ids=[paper_two_authors.id, paper_single_author.id],
        )
        lines = bib.split("\n\n")
        assert len(lines) == 2
        # First entry should be paper_two_authors
        assert "Two Author Study" in lines[0]
        assert "Single Author Study" in lines[1]

    def test_format_bibliography_no_cited_ids_sorted(self):
        p1 = CandidatePaper(
            title="Alpha Study",
            authors=["Adams B"],
            year=2023,
            source_database="pubmed",
        )
        p2 = CandidatePaper(
            title="Zeta Study",
            authors=["Zhang L"],
            year=2022,
            source_database="pubmed",
        )
        fmt = BibliographyFormatter(style="apa")
        bib = fmt.format_bibliography([p2, p1])
        # Should be sorted by first author: Adams before Zhang
        assert bib.index("Adams B") < bib.index("Zhang L")

    def test_empty_bibliography(self):
        fmt = BibliographyFormatter(style="apa")
        bib = fmt.format_bibliography([])
        assert bib == ""
