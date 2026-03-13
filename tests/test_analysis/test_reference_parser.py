# tests/test_analysis/test_reference_parser.py
from __future__ import annotations

from paper.analysis.reference_parser import (
    extract_doi,
    extract_title_heuristic,
    parse_reference_line,
)


class TestExtractDoi:
    def test_extracts_doi_from_line(self):
        line = "[1] Smith J et al. Title here. Nature. 2020. doi:10.1038/s41586-020-1234-5"
        assert extract_doi(line) == "10.1038/s41586-020-1234-5"

    def test_extracts_doi_url(self):
        line = "[2] Doe A. Title. https://doi.org/10.1126/science.abc1234"
        assert extract_doi(line) == "10.1126/science.abc1234"

    def test_returns_none_when_no_doi(self):
        line = "[3] Zhang L. Some paper title. Journal of Things. 2019;45:123-130."
        assert extract_doi(line) is None

    def test_handles_parenthesized_doi(self):
        line = "[4] Title (doi: 10.1234/test.5678)"
        assert extract_doi(line) == "10.1234/test.5678"


class TestExtractTitleHeuristic:
    def test_bracket_numbered_format(self):
        line = "[1] Smith J, Doe A. The role of X in Y. Nature. 2020;580:123-130."
        title = extract_title_heuristic(line)
        assert "role of X in Y" in title

    def test_dot_numbered_format(self):
        line = "1. Smith J, Doe A. The role of X in Y. Nature. 2020;580:123-130."
        title = extract_title_heuristic(line)
        assert "role of X in Y" in title

    def test_returns_empty_for_unparseable(self):
        line = ""
        title = extract_title_heuristic(line)
        assert title == ""


class TestParseReferenceLine:
    def test_returns_doi_and_title(self):
        line = "[1] Smith J. Great discovery in science. Nature. 2020. doi:10.1038/test123"
        result = parse_reference_line(line)
        assert result["doi"] == "10.1038/test123"
        assert "Great discovery" in result["title"]

    def test_returns_title_only_when_no_doi(self):
        line = "[2] Doe A. Another finding. Science. 2019;365:100-105."
        result = parse_reference_line(line)
        assert result["doi"] is None
        assert len(result["title"]) > 0

    def test_returns_raw_line(self):
        line = "[1] Smith J. Great discovery. Nature. 2020."
        result = parse_reference_line(line)
        assert result["raw"] == line
