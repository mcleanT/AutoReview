from __future__ import annotations
from autoreview.models.base import AutoReviewModel


class CitationScore(AutoReviewModel):
    recall: float
    matched_count: int
    reference_count: int
    generated_count: int
    matched_titles: list[str]
    missed_titles: list[str]


class SynthesisScore(AutoReviewModel):
    generated_score: float
    reference_score: float
    delta: float
    dimension_scores: dict[str, float]
    generated_observations: str
    reference_observations: str


class TopicCoverageScore(AutoReviewModel):
    generated_coverage: float
    reference_coverage: float
    topics_in_both: list[str]
    topics_only_in_reference: list[str]
    topics_only_in_generated: list[str]


class WritingQualityScore(AutoReviewModel):
    generated_score: float
    reference_score: float
    delta: float
    dimension_scores: dict[str, float]


class EvaluationResult(AutoReviewModel):
    timestamp: str
    generated_path: str
    reference_path: str
    citation_score: CitationScore
    synthesis_score: SynthesisScore
    topic_coverage: TopicCoverageScore
    writing_quality: WritingQualityScore
    overall_score: float
