"""Tests for batch evaluation aggregation."""

from __future__ import annotations

from autoreview.evaluation.aggregator import _stat, aggregate_results
from autoreview.evaluation.models import (
    AggregatedScores,
    ARISECategoryScore,
    ARISERubricResult,
    ARISESubcriterionScore,
    CitationScore,
    EvaluationResult,
    StatSummary,
    SynthesisScore,
    TopicCoverageScore,
    WritingQualityScore,
)


def _make_citation_score(recall: float, precision: float, f1: float) -> CitationScore:
    return CitationScore(
        recall=recall,
        precision=precision,
        f1=f1,
        matched_count=1,
        reference_count=2,
        generated_count=2,
        matched_titles=["Paper A"],
        missed_titles=["Paper B"],
        hallucinated_titles=[],
    )


def _make_result(
    recall: float,
    precision: float,
    f1: float,
    synthesis: float,
    coverage: float,
    writing: float,
    overall: float,
    arise_total: float | None = None,
) -> EvaluationResult:
    arise_result = None
    if arise_total is not None:
        subcrit = ARISESubcriterionScore(name="sub1", score=arise_total / 20, comment="ok")
        cat = ARISECategoryScore(
            category="cat1", subcriteria=[subcrit], category_score=subcrit.score
        )
        arise_result = ARISERubricResult(
            category_scores=[cat],
            total_score=arise_total,
            normalized_score=arise_total / 100,
        )

    return EvaluationResult(
        timestamp="2026-01-01T00:00:00",
        generated_path="/tmp/gen.md",
        reference_path="/tmp/ref.pdf",
        citation_score=_make_citation_score(recall, precision, f1),
        synthesis_score=SynthesisScore(
            generated_score=synthesis,
            reference_score=4.0,
            delta=synthesis - 4.0,
            dimension_scores={},
            generated_observations="ok",
            reference_observations="ok",
        ),
        topic_coverage=TopicCoverageScore(
            generated_coverage=coverage,
            reference_coverage=0.8,
            topics_in_both=["A"],
            topics_only_in_reference=["B"],
            topics_only_in_generated=[],
        ),
        writing_quality=WritingQualityScore(
            generated_score=writing,
            reference_score=4.0,
            delta=writing - 4.0,
            dimension_scores={},
        ),
        overall_score=overall,
        arise_result=arise_result,
    )


def test_aggregate_results():
    """Three results with known scores → verify mean, std, min, max."""
    r1 = _make_result(
        recall=0.8, precision=0.7, f1=0.75, synthesis=3.5, coverage=0.6, writing=3.0, overall=0.65
    )
    r2 = _make_result(
        recall=0.9, precision=0.8, f1=0.85, synthesis=4.0, coverage=0.7, writing=3.5, overall=0.75
    )
    r3 = _make_result(
        recall=0.7, precision=0.6, f1=0.65, synthesis=3.0, coverage=0.5, writing=2.5, overall=0.55
    )

    agg = aggregate_results([r1, r2, r3])

    assert isinstance(agg, AggregatedScores)
    assert agg.n_topics == 3

    # Recall: [0.8, 0.9, 0.7] → mean=0.8, min=0.7, max=0.9
    assert agg.citation_recall.mean == round((0.8 + 0.9 + 0.7) / 3, 4)
    assert agg.citation_recall.min == 0.7
    assert agg.citation_recall.max == 0.9
    assert agg.citation_recall.n == 3

    # Overall: [0.65, 0.75, 0.55] → mean=0.65
    assert agg.overall_score.mean == round((0.65 + 0.75 + 0.55) / 3, 4)
    assert agg.overall_score.min == 0.55
    assert agg.overall_score.max == 0.75

    # No ARISE → None
    assert agg.arise_total is None


def test_aggregate_empty():
    """Empty list returns n_topics=0 and all zero stats."""
    agg = aggregate_results([])

    assert agg.n_topics == 0
    assert agg.citation_recall.mean == 0.0
    assert agg.citation_recall.n == 0
    assert agg.citation_precision.mean == 0.0
    assert agg.overall_score.mean == 0.0
    assert agg.overall_score.std == 0.0
    assert agg.arise_total is None


def test_aggregate_with_arise():
    """Results with arise_result → arise_total aggregation is computed."""
    r1 = _make_result(
        recall=0.8,
        precision=0.7,
        f1=0.75,
        synthesis=3.5,
        coverage=0.6,
        writing=3.0,
        overall=0.65,
        arise_total=70.0,
    )
    r2 = _make_result(
        recall=0.9,
        precision=0.8,
        f1=0.85,
        synthesis=4.0,
        coverage=0.7,
        writing=3.5,
        overall=0.75,
        arise_total=80.0,
    )

    agg = aggregate_results([r1, r2])

    assert agg.arise_total is not None
    assert agg.arise_total.mean == round((70.0 + 80.0) / 2, 4)
    assert agg.arise_total.min == 70.0
    assert agg.arise_total.max == 80.0
    assert agg.arise_total.n == 2


def test_stat_summary_single():
    """Single value → std=0.0."""
    stat = _stat([0.75])

    assert isinstance(stat, StatSummary)
    assert stat.mean == 0.75
    assert stat.std == 0.0
    assert stat.min == 0.75
    assert stat.max == 0.75
    assert stat.n == 1
