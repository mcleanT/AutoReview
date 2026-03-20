"""Tests for structured contradiction resolution with evidence weighting."""

from __future__ import annotations

from autoreview.analysis.contradiction_resolver import (
    ContradictionResolution,
    EvidenceBalance,
    _compare_methodologies,
    _compute_side_strength,
    resolve_contradiction,
)
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extraction(
    paper_id: str,
    study_design: StudyDesign | None = None,
    quality_score: float | None = None,
    sample_size: int | None = None,
) -> PaperExtraction:
    """Build a minimal PaperExtraction for testing."""
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
        limitations="None noted",
        study_design=study_design,
        quality_score=quality_score,
        sample_size=sample_size,
    )


# ---------------------------------------------------------------------------
# _compute_side_strength
# ---------------------------------------------------------------------------


def test_compute_side_strength_empty_list():
    """Empty list returns 0.0 strength."""
    assert _compute_side_strength([]) == 0.0


def test_compute_side_strength_single_rct():
    """Single RCT with perfect quality and no size info gives expected score."""
    ext = _make_extraction("p1", StudyDesign.RCT, quality_score=1.0, sample_size=None)
    strength = _compute_side_strength([ext])
    # design_w=0.90*0.4=0.36, quality_w=1.0*0.4=0.40, size_w=0.0*0.2=0.0 → 0.76
    assert abs(strength - 0.76) < 1e-6


def test_compute_side_strength_no_metadata():
    """Extraction with no design, quality, or sample size uses defaults."""
    ext = _make_extraction("p1", study_design=None, quality_score=None, sample_size=None)
    strength = _compute_side_strength([ext])
    # design_w=0.15*0.4=0.06, quality_w=0.0*0.4=0.0, size_w=0.0*0.2=0.0 → 0.06
    assert abs(strength - 0.06) < 1e-6


# ---------------------------------------------------------------------------
# test_stronger_evidence_side_identified
# ---------------------------------------------------------------------------


def test_stronger_evidence_side_identified():
    """2 RCTs with good quality vs 1 case series → SIDE_A_STRONGER."""
    side_a = [
        _make_extraction("rct1", StudyDesign.RCT, quality_score=0.85, sample_size=500),
        _make_extraction("rct2", StudyDesign.RCT, quality_score=0.80, sample_size=300),
    ]
    side_b = [
        _make_extraction("cs1", StudyDesign.CASE_SERIES, quality_score=0.40, sample_size=12),
    ]

    resolution = resolve_contradiction(
        claim_a="Treatment X reduces mortality.",
        claim_b="Treatment X has no effect on mortality.",
        side_a_extractions=side_a,
        side_b_extractions=side_b,
    )

    assert isinstance(resolution, ContradictionResolution)
    assert resolution.evidence_balance == EvidenceBalance.SIDE_A_STRONGER
    assert resolution.side_a_strength > resolution.side_b_strength
    assert 0.0 <= resolution.confidence <= 1.0


# ---------------------------------------------------------------------------
# test_balanced_evidence_detected
# ---------------------------------------------------------------------------


def test_balanced_evidence_detected():
    """Two sides with matching cohort studies → BALANCED."""
    side_a = [
        _make_extraction("c1", StudyDesign.COHORT, quality_score=0.70, sample_size=1000),
    ]
    side_b = [
        _make_extraction("c2", StudyDesign.COHORT, quality_score=0.70, sample_size=1000),
    ]

    resolution = resolve_contradiction(
        claim_a="Diet A improves cardiovascular outcomes.",
        claim_b="Diet A has no effect on cardiovascular outcomes.",
        side_a_extractions=side_a,
        side_b_extractions=side_b,
    )

    assert resolution.evidence_balance == EvidenceBalance.BALANCED
    # Strengths should be equal (or very close) for symmetric inputs
    assert abs(resolution.side_a_strength - resolution.side_b_strength) < 1e-6


# ---------------------------------------------------------------------------
# test_resolution_includes_methodology_comparison
# ---------------------------------------------------------------------------


def test_resolution_includes_methodology_comparison():
    """Methodology comparison string is non-empty and mentions both sides."""
    side_a = [
        _make_extraction("m1", StudyDesign.META_ANALYSIS, quality_score=0.95, sample_size=5000),
    ]
    side_b = [
        _make_extraction("in1", StudyDesign.IN_VITRO, quality_score=0.50, sample_size=None),
    ]

    resolution = resolve_contradiction(
        claim_a="Drug Z is safe at standard doses.",
        claim_b="Drug Z causes hepatotoxicity.",
        side_a_extractions=side_a,
        side_b_extractions=side_b,
    )

    assert resolution.methodology_comparison  # non-empty
    assert "Side A" in resolution.methodology_comparison
    assert "Side B" in resolution.methodology_comparison


# ---------------------------------------------------------------------------
# test_insufficient_data_when_both_sides_empty
# ---------------------------------------------------------------------------


def test_insufficient_data_when_both_sides_empty():
    """No extractions on either side → INSUFFICIENT_DATA with zero confidence."""
    resolution = resolve_contradiction(
        claim_a="Claim A.",
        claim_b="Claim B.",
        side_a_extractions=[],
        side_b_extractions=[],
    )

    assert resolution.evidence_balance == EvidenceBalance.INSUFFICIENT_DATA
    assert resolution.confidence == 0.0
    assert resolution.side_a_strength == 0.0
    assert resolution.side_b_strength == 0.0


# ---------------------------------------------------------------------------
# test_side_b_stronger
# ---------------------------------------------------------------------------


def test_side_b_stronger():
    """RCT on side B vs case report on side A → SIDE_B_STRONGER."""
    side_a = [
        _make_extraction("cr1", StudyDesign.CASE_REPORT, quality_score=0.30, sample_size=1),
    ]
    side_b = [
        _make_extraction("rct1", StudyDesign.RCT, quality_score=0.88, sample_size=800),
        _make_extraction("rct2", StudyDesign.RCT, quality_score=0.92, sample_size=600),
    ]

    resolution = resolve_contradiction(
        claim_a="Vaccine has severe side effects.",
        claim_b="Vaccine side-effect profile is acceptable.",
        side_a_extractions=side_a,
        side_b_extractions=side_b,
    )

    assert resolution.evidence_balance == EvidenceBalance.SIDE_B_STRONGER
    assert resolution.side_b_strength > resolution.side_a_strength


# ---------------------------------------------------------------------------
# test_resolution_hypothesis_non_empty
# ---------------------------------------------------------------------------


def test_resolution_hypothesis_non_empty():
    """Resolution hypothesis is always a non-empty string."""
    for balance_scenario in [
        (
            [_make_extraction("a1", StudyDesign.RCT, 0.9, 500)],
            [_make_extraction("b1", StudyDesign.CASE_SERIES, 0.3, 5)],
        ),
        (
            [_make_extraction("a2", StudyDesign.COHORT, 0.7, 200)],
            [_make_extraction("b2", StudyDesign.COHORT, 0.7, 200)],
        ),
        ([], []),
    ]:
        side_a, side_b = balance_scenario
        resolution = resolve_contradiction("Claim A", "Claim B", side_a, side_b)
        assert resolution.resolution_hypothesis, "hypothesis should not be empty"


# ---------------------------------------------------------------------------
# test_compare_methodologies_standalone
# ---------------------------------------------------------------------------


def test_compare_methodologies_standalone():
    """_compare_methodologies returns a string with both side summaries."""
    side_a = [_make_extraction("p1", StudyDesign.RCT, 0.8, 300)]
    side_b = [_make_extraction("p2", StudyDesign.COHORT, 0.6, 800)]
    result = _compare_methodologies(side_a, side_b)
    assert "Side A" in result
    assert "Side B" in result
    assert len(result) > 10
