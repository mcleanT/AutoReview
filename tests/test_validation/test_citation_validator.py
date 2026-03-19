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


class TestValidateAgainstPlan:
    def test_validate_against_plan_checks_primary_compliance(self):
        from autoreview.validation.citation_validator import CitationValidator
        from autoreview.writing.citation_selector import PaperCitation, SectionCitationPlan

        validator = CitationValidator()
        plan = SectionCitationPlan(
            section_id="sec_3",
            citation_budget=10,
            primary_papers=[
                PaperCitation(
                    paper_id="p1", tier="primary", priority_score=0.9, citation_guidance=""
                ),
                PaperCitation(
                    paper_id="p2", tier="primary", priority_score=0.8, citation_guidance=""
                ),
            ],
            supporting_papers=[],
            contextual_papers=[],
            coverage_notes=[],
        )
        # Text only cites p1, missing p2
        text = "Some findings [@p1] show results."
        report = validator.validate_against_plan(text, plan)
        assert "p2" in report.uncited_primary
        assert "p1" not in report.uncited_primary

    def test_validate_against_plan_budget_deviation(self):
        from autoreview.validation.citation_validator import CitationValidator
        from autoreview.writing.citation_selector import PaperCitation, SectionCitationPlan

        validator = CitationValidator()
        plan = SectionCitationPlan(
            section_id="sec_1",
            citation_budget=10,
            primary_papers=[
                PaperCitation(
                    paper_id="p1", tier="primary", priority_score=0.9, citation_guidance=""
                ),
            ],
            supporting_papers=[],
            contextual_papers=[],
            coverage_notes=[],
        )
        # Cite 15 papers (5 over budget)
        text = " ".join(f"[@p{i}]" for i in range(1, 16))
        report = validator.validate_against_plan(text, plan)
        # deviation = (15 - 10) / 10 = 0.5
        assert abs(report.budget_deviation - 0.5) < 1e-4
        assert report.total_citations == 15
        assert report.citation_budget == 10

    def test_validate_against_plan_additive_citations(self):
        """Citations outside the plan should be flagged as additive."""
        from autoreview.validation.citation_validator import CitationValidator
        from autoreview.writing.citation_selector import PaperCitation, SectionCitationPlan

        validator = CitationValidator()
        plan = SectionCitationPlan(
            section_id="sec_2",
            citation_budget=5,
            primary_papers=[
                PaperCitation(
                    paper_id="p1", tier="primary", priority_score=0.9, citation_guidance=""
                ),
            ],
            supporting_papers=[
                PaperCitation(
                    paper_id="p2", tier="supporting", priority_score=0.6, citation_guidance=""
                ),
            ],
            contextual_papers=[],
            coverage_notes=[],
        )
        # p3 is additive (not in plan)
        text = "Results [@p1] and [@p2] plus extra [@p3]."
        report = validator.validate_against_plan(text, plan)
        assert "p3" in report.additive_citations
        assert "p1" not in report.additive_citations
        assert "p2" not in report.additive_citations

    def test_validate_against_plan_all_primary_cited(self):
        """No uncited_primary when all PRIMARY papers are cited."""
        from autoreview.validation.citation_validator import CitationValidator
        from autoreview.writing.citation_selector import PaperCitation, SectionCitationPlan

        validator = CitationValidator()
        plan = SectionCitationPlan(
            section_id="sec_4",
            citation_budget=5,
            primary_papers=[
                PaperCitation(
                    paper_id="p1", tier="primary", priority_score=0.9, citation_guidance=""
                ),
                PaperCitation(
                    paper_id="p2", tier="primary", priority_score=0.8, citation_guidance=""
                ),
            ],
            supporting_papers=[],
            contextual_papers=[],
            coverage_notes=[],
        )
        text = "Results from [@p1] and [@p2] are clear."
        report = validator.validate_against_plan(text, plan)
        assert report.uncited_primary == []


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
