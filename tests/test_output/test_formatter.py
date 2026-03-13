"""Tests for output formatting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.models.paper import CandidatePaper
from autoreview.output.formatter import (
    OutputFormatter,
    _latex_escape,
    _markdown_to_latex,
    _resolve_citations,
)


@pytest.fixture
def papers():
    p1 = CandidatePaper(
        title="Paper Alpha",
        authors=["Smith J", "Doe A"],
        year=2023,
        journal="Nature",
        doi="10.1038/alpha",
        source_database="pubmed",
    )
    p2 = CandidatePaper(
        title="Paper Beta",
        authors=["Zhang L"],
        year=2022,
        journal="Science",
        doi="10.1126/beta",
        source_database="semantic_scholar",
    )
    return [p1, p2]


@pytest.fixture
def kb_with_draft(papers):
    kb = KnowledgeBase(
        topic="test topic",
        domain="biomedical",
        output_dir="/tmp/test_output",
    )
    kb.candidate_papers = papers
    kb.full_draft = (
        f"This is a review. Studies show positive results [@{papers[0].id}]. "
        f"However, others disagree [@{papers[1].id}]. "
        f"Both findings are important [@{papers[0].id}]."
    )
    return kb


class TestResolveCitations:
    def test_apa_style(self, papers):
        text = f"Results show [@{papers[0].id}] and [@{papers[1].id}]."
        resolved, cited_ids = _resolve_citations(text, papers, style="apa")
        # APA: (LastName, Year)
        assert "(Smith, 2023)" in resolved
        assert "(Zhang, 2022)" in resolved
        assert len(cited_ids) == 2

    def test_vancouver_style(self, papers):
        text = f"Results show [@{papers[0].id}] and [@{papers[1].id}]."
        resolved, cited_ids = _resolve_citations(text, papers, style="vancouver")
        assert "[1]" in resolved
        assert "[2]" in resolved

    def test_repeated_citation_same_number(self, papers):
        text = f"First [@{papers[0].id}]. Second [@{papers[1].id}]. Third [@{papers[0].id}]."
        resolved, cited_ids = _resolve_citations(text, papers, style="vancouver")
        # papers[0] should always be [1]
        assert resolved.count("[1]") == 2
        assert resolved.count("[2]") == 1
        assert len(cited_ids) == 2  # Only 2 unique

    def test_unknown_paper_id(self, papers):
        text = "Unknown [@nonexistent_id]."
        resolved, cited_ids = _resolve_citations(text, papers, style="apa")
        # Falls back to numbered
        assert "[1]" in resolved

    def test_no_citations(self, papers):
        text = "No citations here."
        resolved, cited_ids = _resolve_citations(text, papers)
        assert resolved == text
        assert len(cited_ids) == 0


class TestLatexEscape:
    def test_special_characters(self):
        assert _latex_escape("a & b") == r"a \& b"
        assert _latex_escape("100%") == r"100\%"
        assert _latex_escape("$10") == r"\$10"
        assert _latex_escape("item #1") == r"item \#1"
        assert _latex_escape("under_score") == r"under\_score"

    def test_no_special_chars(self):
        assert _latex_escape("plain text") == "plain text"


class TestMarkdownToLatex:
    def test_heading_conversion(self):
        md = "# Title\n## Section\n### Subsection\n#### Sub-subsection"
        latex = _markdown_to_latex(md)
        assert r"\title{Title}" in latex
        assert r"\maketitle" in latex
        assert r"\section{Section}" in latex
        assert r"\subsection{Subsection}" in latex
        assert r"\subsubsection{Sub-subsection}" in latex

    def test_document_structure(self):
        latex = _markdown_to_latex("Hello world")
        assert r"\documentclass{article}" in latex
        assert r"\begin{document}" in latex
        assert r"\end{document}" in latex
        assert "Hello world" in latex


class TestOutputFormatter:
    def test_format_markdown_empty_draft(self):
        kb = KnowledgeBase(topic="test", domain="test", output_dir="/tmp/test")
        fmt = OutputFormatter()
        result = fmt.format_markdown(kb)
        assert result == ""

    def test_format_markdown_with_draft(self, kb_with_draft, papers):
        fmt = OutputFormatter(style="apa")
        result = fmt.format_markdown(kb_with_draft)
        assert "References" in result
        assert "---" in result
        # Should have resolved citations
        assert "[@" not in result

    def test_format_latex_empty(self):
        kb = KnowledgeBase(topic="test", domain="test", output_dir="/tmp/test")
        fmt = OutputFormatter()
        latex, bibtex = fmt.format_latex(kb)
        assert latex == ""
        assert bibtex == ""

    def test_format_latex_with_draft(self, kb_with_draft):
        fmt = OutputFormatter()
        latex, bibtex = fmt.format_latex(kb_with_draft)
        assert r"\documentclass{article}" in latex
        assert r"\begin{document}" in latex
        assert "@article{" in bibtex

    def test_save_markdown(self, kb_with_draft):
        with tempfile.TemporaryDirectory() as tmpdir:
            fmt = OutputFormatter(style="apa")
            created = fmt.save(kb_with_draft, tmpdir, fmt="markdown")
            assert len(created) == 1
            assert created[0].endswith("review.md")
            content = Path(created[0]).read_text()
            assert "References" in content

    def test_save_latex(self, kb_with_draft):
        with tempfile.TemporaryDirectory() as tmpdir:
            fmt = OutputFormatter()
            created = fmt.save(kb_with_draft, tmpdir, fmt="latex")
            assert len(created) == 2
            assert any(p.endswith("review.tex") for p in created)
            assert any(p.endswith("references.bib") for p in created)

    def test_save_all_formats(self, kb_with_draft):
        with tempfile.TemporaryDirectory() as tmpdir:
            fmt = OutputFormatter()
            created = fmt.save(kb_with_draft, tmpdir, fmt="all")
            # Markdown (1 file) + LaTeX (2 files) = 3 minimum; docx adds 1 if pandoc installed
            assert len(created) >= 3
            assert any(p.endswith("review.md") for p in created)
            assert any(p.endswith("review.tex") for p in created)
            assert any(p.endswith("references.bib") for p in created)

    def test_save_creates_output_dir(self, kb_with_draft):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "output"
            fmt = OutputFormatter()
            created = fmt.save(kb_with_draft, str(nested), fmt="markdown")
            assert nested.exists()
            assert len(created) == 1

    def test_vancouver_format(self, kb_with_draft):
        fmt = OutputFormatter(style="vancouver")
        result = fmt.format_markdown(kb_with_draft)
        assert "[1]" in result or "[2]" in result
