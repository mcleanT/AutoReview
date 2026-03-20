"""Structured contradiction resolution with evidence-quality weighting.

Resolves contradictions between opposing claims by comparing the quality of
supporting evidence on each side, accounting for study design hierarchy,
quality scores, and sample sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import structlog

from autoreview.extraction.models import PaperExtraction, StudyDesign

logger = structlog.get_logger()

# Evidence hierarchy weight for each study design.
# Higher values indicate stronger causal inference capability.
_DESIGN_WEIGHT: dict[StudyDesign, float] = {
    StudyDesign.META_ANALYSIS: 1.00,
    StudyDesign.SYSTEMATIC_REVIEW: 0.95,
    StudyDesign.RCT: 0.90,
    StudyDesign.COHORT: 0.70,
    StudyDesign.CASE_CONTROL: 0.60,
    StudyDesign.CROSS_SECTIONAL: 0.50,
    StudyDesign.NARRATIVE_REVIEW: 0.45,
    StudyDesign.CASE_SERIES: 0.35,
    StudyDesign.CASE_REPORT: 0.25,
    StudyDesign.IN_VITRO: 0.20,
    StudyDesign.COMPUTATIONAL: 0.20,
    StudyDesign.OTHER: 0.15,
}

# Default design weight when study_design is None.
_DEFAULT_DESIGN_WEIGHT: float = 0.15

# Minimum absolute ratio difference to call one side stronger.
# Below this threshold the sides are considered BALANCED.
_BALANCE_THRESHOLD: float = 0.15


class EvidenceBalance(StrEnum):
    """Direction and balance of evidence across a contradiction."""

    SIDE_A_STRONGER = "side_a_stronger"
    SIDE_B_STRONGER = "side_b_stronger"
    BALANCED = "balanced"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class ContradictionResolution:
    """Structured resolution of a contradiction between two claims.

    Attributes:
        evidence_balance: Which side has stronger support, or if balanced.
        confidence: Confidence in the resolution (0–1); lower when data is sparse.
        side_a_strength: Aggregated evidence strength score for side A (0–1).
        side_b_strength: Aggregated evidence strength score for side B (0–1).
        methodology_comparison: Human-readable comparison of the two sides.
        resolution_hypothesis: Suggested explanation for the contradiction.
    """

    evidence_balance: EvidenceBalance
    confidence: float
    side_a_strength: float
    side_b_strength: float
    methodology_comparison: str
    resolution_hypothesis: str


def _design_weight(extraction: PaperExtraction) -> float:
    """Return the study-design hierarchy weight for a single extraction."""
    if extraction.study_design is None:
        return _DEFAULT_DESIGN_WEIGHT
    return _DESIGN_WEIGHT.get(extraction.study_design, _DEFAULT_DESIGN_WEIGHT)


def _normalise_sample_size(extractions: list[PaperExtraction]) -> float:
    """Return a 0–1 score for average log-normalised sample size.

    Papers with unknown sample size contribute 0 to the average.
    """
    if not extractions:
        return 0.0
    scores = []
    for ext in extractions:
        if ext.sample_size and ext.sample_size > 0:
            # log10(10) ≈ 1, log10(10_000) = 4; cap at log10(1_000_000) = 6
            scores.append(min(math.log10(ext.sample_size) / 6.0, 1.0))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def _compute_side_strength(extractions: list[PaperExtraction]) -> float:
    """Compute an aggregated evidence-strength score for a set of extractions.

    Formula per paper:
        strength = design_w * 0.4 + quality_w * 0.4 + size_w * 0.2

    The per-paper scores are then averaged across all extractions.

    Returns a score in [0, 1].  Returns 0.0 for an empty list.
    """
    if not extractions:
        return 0.0

    paper_scores = []
    for ext in extractions:
        design_w = _design_weight(ext)
        quality_w = ext.quality_score if ext.quality_score is not None else 0.0
        size_w = _normalise_sample_size([ext])
        score = design_w * 0.4 + quality_w * 0.4 + size_w * 0.2
        paper_scores.append(score)

    return sum(paper_scores) / len(paper_scores)


def _summarise_side(extractions: list[PaperExtraction]) -> str:
    """Build a short human-readable summary of one side's evidence base."""
    if not extractions:
        return "no papers"

    design_counts: dict[str, int] = {}
    for ext in extractions:
        label = str(ext.study_design) if ext.study_design else "unknown design"
        design_counts[label] = design_counts.get(label, 0) + 1

    parts = [
        f"{count} {design.replace('_', ' ')}" for design, count in sorted(design_counts.items())
    ]
    n_str = ", ".join(parts)

    quality_vals = [ext.quality_score for ext in extractions if ext.quality_score is not None]
    if quality_vals:
        avg_q = sum(quality_vals) / len(quality_vals)
        n_str += f" (avg quality {avg_q:.2f})"

    sizes = [ext.sample_size for ext in extractions if ext.sample_size]
    if sizes:
        n_str += f"; median N={sorted(sizes)[len(sizes) // 2]:,}"

    return n_str


def _compare_methodologies(
    side_a: list[PaperExtraction],
    side_b: list[PaperExtraction],
) -> str:
    """Generate a plain-text methodology comparison between the two sides."""
    a_summary = _summarise_side(side_a)
    b_summary = _summarise_side(side_b)
    return f"Side A evidence base: {a_summary}. Side B evidence base: {b_summary}."


def resolve_contradiction(
    claim_a: str,
    claim_b: str,
    side_a_extractions: list[PaperExtraction],
    side_b_extractions: list[PaperExtraction],
) -> ContradictionResolution:
    """Resolve a contradiction by comparing evidence quality on each side.

    Args:
        claim_a: The claim made by side A.
        claim_b: The claim made by side B (contradicts claim_a).
        side_a_extractions: Papers supporting claim_a.
        side_b_extractions: Papers supporting claim_b.

    Returns:
        A ContradictionResolution with balance assessment, strength scores,
        methodology comparison, and a resolution hypothesis.
    """
    logger.debug(
        "resolving_contradiction",
        n_side_a=len(side_a_extractions),
        n_side_b=len(side_b_extractions),
    )

    strength_a = _compute_side_strength(side_a_extractions)
    strength_b = _compute_side_strength(side_b_extractions)
    total = strength_a + strength_b

    # Determine evidence balance.
    if total == 0.0:
        balance = EvidenceBalance.INSUFFICIENT_DATA
        confidence = 0.0
    else:
        ratio_a = strength_a / total
        ratio_b = strength_b / total
        diff = abs(ratio_a - ratio_b)

        if diff < _BALANCE_THRESHOLD:
            balance = EvidenceBalance.BALANCED
        elif ratio_a > ratio_b:
            balance = EvidenceBalance.SIDE_A_STRONGER
        else:
            balance = EvidenceBalance.SIDE_B_STRONGER

        # Confidence scales with total evidence and the size of the imbalance.
        n_papers = len(side_a_extractions) + len(side_b_extractions)
        paper_factor = min(n_papers / 10.0, 1.0)  # saturates at 10 papers
        confidence = round(diff * paper_factor, 4)

    methodology_comparison = _compare_methodologies(side_a_extractions, side_b_extractions)

    # Build a resolution hypothesis.
    if balance == EvidenceBalance.INSUFFICIENT_DATA:
        hypothesis = (
            f"Insufficient evidence to resolve the contradiction between "
            f'"{claim_a}" and "{claim_b}". Additional studies are needed.'
        )
    elif balance == EvidenceBalance.BALANCED:
        hypothesis = (
            f'The contradiction between "{claim_a}" and "{claim_b}" may reflect '
            f"genuine context-dependence (e.g., population, dosage, or follow-up "
            f"duration differences) rather than a true scientific disagreement."
        )
    elif balance == EvidenceBalance.SIDE_A_STRONGER:
        hypothesis = (
            f'The preponderance of higher-quality evidence supports "{claim_a}". '
            f'The opposing claim ("{claim_b}") may stem from studies with '
            f"limited sample sizes or lower-hierarchy designs."
        )
    else:
        hypothesis = (
            f'The preponderance of higher-quality evidence supports "{claim_b}". '
            f'The opposing claim ("{claim_a}") may stem from studies with '
            f"limited sample sizes or lower-hierarchy designs."
        )

    resolution = ContradictionResolution(
        evidence_balance=balance,
        confidence=confidence,
        side_a_strength=round(strength_a, 4),
        side_b_strength=round(strength_b, 4),
        methodology_comparison=methodology_comparison,
        resolution_hypothesis=hypothesis,
    )

    logger.info(
        "contradiction_resolved",
        balance=balance,
        confidence=confidence,
        side_a_strength=strength_a,
        side_b_strength=strength_b,
    )

    return resolution
