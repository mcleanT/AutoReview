"""Aggregate evaluation results across multiple topics."""

from __future__ import annotations

import statistics

import structlog

from autoreview.evaluation.models import AggregatedScores, EvaluationResult, StatSummary

logger = structlog.get_logger()


def _stat(values: list[float]) -> StatSummary:
    n = len(values)
    if n == 0:
        return StatSummary(mean=0.0, std=0.0, min=0.0, max=0.0, n=0)
    m = statistics.mean(values)
    s = statistics.stdev(values) if n > 1 else 0.0
    return StatSummary(
        mean=round(m, 4),
        std=round(s, 4),
        min=round(min(values), 4),
        max=round(max(values), 4),
        n=n,
    )


def aggregate_results(
    results: list[EvaluationResult],
    group_key: str | None = None,
    group_values: dict[str, list[EvaluationResult]] | None = None,
) -> AggregatedScores:
    """Aggregate evaluation results, optionally with grouping breakdowns."""
    recalls = [r.citation_score.recall for r in results]
    precisions = [r.citation_score.precision for r in results]
    f1s = [r.citation_score.f1 for r in results]
    synthesis = [r.synthesis_score.generated_score for r in results]
    coverage = [r.topic_coverage.generated_coverage for r in results]
    writing = [r.writing_quality.generated_score for r in results]
    overall = [r.overall_score for r in results]

    arise_totals = [r.arise_result.total_score for r in results if r.arise_result is not None]

    agg = AggregatedScores(
        n_topics=len(results),
        citation_recall=_stat(recalls),
        citation_precision=_stat(precisions),
        citation_f1=_stat(f1s),
        synthesis_score=_stat(synthesis),
        topic_coverage=_stat(coverage),
        writing_quality=_stat(writing),
        overall_score=_stat(overall),
        arise_total=_stat(arise_totals) if arise_totals else None,
    )

    # Add group breakdowns if provided
    if group_values:
        if group_key == "model_tier":
            agg.by_model_tier = {k: aggregate_results(v) for k, v in group_values.items()}
        elif group_key == "domain":
            agg.by_domain = {k: aggregate_results(v) for k, v in group_values.items()}

    return agg
