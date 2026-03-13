"""Tests for citation validation."""

from __future__ import annotations

from autoreview.critique.models import CritiqueSeverity
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
)
from autoreview.validation.citation_validator import (
    CitationValidationReport,
    CitationValidator,
)


def _make_extraction(paper_id: str) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[
            Finding(
                claim="Test finding",
                evidence_strength=EvidenceStrength.MODERATE,
                paper_id=paper_id,
            )
        ],
        methods_summary="Test methods",
        limitations="Test limitations",
    )


def _make_extractions(*paper_ids: str) -> dict[str, PaperExtraction]:
    return {pid: _make_extraction(pid) for pid in paper_ids}


class TestValidateSection:
    def test_all_citations_valid(self):
        text = "Results show X [@paper1] and Y [@paper2]."
        extractions = _make_extractions("paper1", "paper2")
        validator = CitationValidator()
        report = validator.validate_section(text, ["paper1", "paper2"], extractions)

        assert sorted(report.valid_citations) == ["paper1", "paper2"]
        assert report.invalid_citations == []
        assert report.uncited_papers == []
        assert report.total_citation_markers == 2
        assert report.unique_citations == 2

    def test_invalid_citation_detected(self):
        text = "Results show X [@paper1] and Y [@nonexistent]."
        extractions = _make_extractions("paper1")
        validator = CitationValidator()
        report = validator.validate_section(text, ["paper1"], extractions)

        assert "paper1" in report.valid_citations
        assert "nonexistent" in report.invalid_citations

    def test_uncited_paper_detected(self):
        text = "Results show X [@paper1]."
        extractions = _make_extractions("paper1", "paper2")
        validator = CitationValidator()
        report = validator.validate_section(text, ["paper1", "paper2"], extractions)

        assert "paper2" in report.uncited_papers
        assert "paper1" not in report.uncited_papers

    def test_empty_text_returns_clean_report(self):
        extractions = _make_extractions("paper1")
        validator = CitationValidator()
        report = validator.validate_section("", ["paper1"], extractions)

        assert report.valid_citations == []
        assert report.invalid_citations == []
        assert report.uncited_papers == ["paper1"]
        assert report.total_citation_markers == 0

    def test_duplicate_citations_counted(self):
        text = "Results [@paper1] confirm [@paper1] previous work."
        extractions = _make_extractions("paper1")
        validator = CitationValidator()
        report = validator.validate_section(text, ["paper1"], extractions)

        assert report.total_citation_markers == 2
        assert report.unique_citations == 1


class TestValidateFullDraft:
    def test_full_draft_validation(self):
        text = "Introduction [@paper1]. Body [@paper2]. Conclusion [@paper3]."
        extractions = _make_extractions("paper1", "paper2", "paper3", "paper4")
        validator = CitationValidator()
        report = validator.validate_full_draft(text, extractions)

        assert report.section_id == "full_draft"
        assert sorted(report.valid_citations) == ["paper1", "paper2", "paper3"]
        assert report.invalid_citations == []
        assert "paper4" in report.uncited_papers

    def test_full_draft_invalid_citations(self):
        text = "Claim [@real] and [@phantom]."
        extractions = _make_extractions("real")
        validator = CitationValidator()
        report = validator.validate_full_draft(text, extractions)

        assert "phantom" in report.invalid_citations
        assert "real" in report.valid_citations


class TestToCritiqueIssues:
    def test_severity_mapping(self):
        report = CitationValidationReport(
            section_id="s1",
            invalid_citations=["phantom1"],
            uncited_papers=["uncited1"],
        )
        issues = CitationValidator.to_critique_issues(report)

        # invalid -> CRITICAL, uncited -> MAJOR
        {i.description: i.severity for i in issues}
        critical_issues = [i for i in issues if i.severity == CritiqueSeverity.CRITICAL]
        major_issues = [i for i in issues if i.severity == CritiqueSeverity.MAJOR]

        assert len(critical_issues) == 1
        assert "phantom1" in critical_issues[0].description
        assert len(major_issues) == 1
        assert "uncited1" in major_issues[0].description

    def test_empty_report_no_issues(self):
        report = CitationValidationReport(section_id="s1")
        issues = CitationValidator.to_critique_issues(report)
        assert issues == []

    def test_multiple_invalid_citations(self):
        report = CitationValidationReport(
            section_id="s1",
            invalid_citations=["fake1", "fake2", "fake3"],
        )
        issues = CitationValidator.to_critique_issues(report)
        assert len(issues) == 3
        assert all(i.severity == CritiqueSeverity.CRITICAL for i in issues)
