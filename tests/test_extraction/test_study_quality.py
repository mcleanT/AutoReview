"""Tests for study design classification and quality scoring fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)


class TestStudyDesignEnum:
    """Tests for the StudyDesign StrEnum."""

    def test_all_designs_defined(self) -> None:
        expected = {
            "rct",
            "cohort",
            "case_control",
            "cross_sectional",
            "case_series",
            "case_report",
            "in_vitro",
            "computational",
            "meta_analysis",
            "systematic_review",
            "narrative_review",
            "other",
        }
        actual = {member.value for member in StudyDesign}
        assert actual == expected


def _make_extraction(**kwargs) -> PaperExtraction:
    """Helper to build a minimal valid PaperExtraction."""
    defaults = dict(
        paper_id="paper-001",
        key_findings=[
            Finding(
                claim="Test claim",
                evidence_strength=EvidenceStrength.MODERATE,
                paper_id="paper-001",
            )
        ],
        methods_summary="Standard methods",
        limitations="None reported",
    )
    defaults.update(kwargs)
    return PaperExtraction(**defaults)


class TestPaperExtractionQuality:
    """Tests for study quality fields on PaperExtraction."""

    def test_study_design_defaults_to_none(self) -> None:
        extraction = _make_extraction()
        assert extraction.study_design is None

    def test_quality_score_defaults_to_none(self) -> None:
        extraction = _make_extraction()
        assert extraction.quality_score is None

    def test_sample_size_defaults_to_none(self) -> None:
        extraction = _make_extraction()
        assert extraction.sample_size is None

    def test_quality_score_range(self) -> None:
        # Valid boundary values
        low = _make_extraction(quality_score=0.0)
        assert low.quality_score == 0.0

        high = _make_extraction(quality_score=1.0)
        assert high.quality_score == 1.0

        mid = _make_extraction(quality_score=0.75)
        assert mid.quality_score == 0.75

    def test_quality_score_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            _make_extraction(quality_score=-0.1)

        with pytest.raises(ValidationError):
            _make_extraction(quality_score=1.01)

    def test_study_design_valid_value(self) -> None:
        extraction = _make_extraction(study_design=StudyDesign.RCT)
        assert extraction.study_design == StudyDesign.RCT
        assert extraction.study_design == "rct"

    def test_study_design_string_coercion(self) -> None:
        extraction = _make_extraction(study_design="meta_analysis")
        assert extraction.study_design == StudyDesign.META_ANALYSIS

    def test_sample_size_set(self) -> None:
        extraction = _make_extraction(sample_size=250)
        assert extraction.sample_size == 250

    def test_all_quality_fields_together(self) -> None:
        extraction = _make_extraction(
            study_design=StudyDesign.COHORT,
            quality_score=0.8,
            sample_size=1200,
        )
        assert extraction.study_design == StudyDesign.COHORT
        assert extraction.quality_score == 0.8
        assert extraction.sample_size == 1200
