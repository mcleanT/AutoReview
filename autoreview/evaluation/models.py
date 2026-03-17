from __future__ import annotations

from typing import Literal  # noqa: F401

from autoreview.models.base import AutoReviewModel


class StatSummary(AutoReviewModel):
    mean: float
    std: float
    min: float
    max: float
    n: int


class TopicConfig(AutoReviewModel):
    topic: str
    domain: str
    generated_path: str
    reference_path: str
    model_tier: str | None = None


class BatchEvaluationConfig(AutoReviewModel):
    topics: list[TopicConfig]
    judge_model: str = "claude-sonnet-4-6"
    rubrics: list[str] = ["autoreview", "arise"]


class CitationScore(AutoReviewModel):
    recall: float
    precision: float
    f1: float
    matched_count: int
    reference_count: int
    generated_count: int
    matched_titles: list[str]
    missed_titles: list[str]
    hallucinated_titles: list[str]


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


class ARISESubcriterionScore(AutoReviewModel):
    name: str
    score: float  # 1-5
    comment: str


class ARISECategoryScore(AutoReviewModel):
    category: str
    subcriteria: list[ARISESubcriterionScore]
    category_score: float  # mean of subcriteria


class ARISERubricResult(AutoReviewModel):
    category_scores: list[ARISECategoryScore]
    total_score: float  # sum of all subcriteria scores (max 100)
    normalized_score: float  # total / 100


class StructuralMetrics(AutoReviewModel):
    word_count: int
    section_count: int
    citation_count: int
    citations_per_1000_words: float
    avg_section_length_words: float
    section_balance: float  # std/mean of section lengths (0 = perfectly balanced)
    flesch_kincaid_grade: float


class EvaluationResult(AutoReviewModel):
    timestamp: str
    generated_path: str
    reference_path: str
    citation_score: CitationScore
    synthesis_score: SynthesisScore
    topic_coverage: TopicCoverageScore
    writing_quality: WritingQualityScore
    overall_score: float
    arise_result: ARISERubricResult | None = None
    structural_metrics: StructuralMetrics | None = None


class AggregatedScores(AutoReviewModel):
    n_topics: int
    citation_recall: StatSummary
    citation_precision: StatSummary
    citation_f1: StatSummary
    synthesis_score: StatSummary
    topic_coverage: StatSummary
    writing_quality: StatSummary
    overall_score: StatSummary
    arise_total: StatSummary | None = None
    by_model_tier: dict[str, AggregatedScores] | None = None
    by_domain: dict[str, AggregatedScores] | None = None


class Claim(AutoReviewModel):
    """A distinct factual assertion extracted from a review."""

    text: str
    category: Literal["empirical", "methodological", "contextual", "synthesis", "limitation"]
    section_id: str | None = None


class InformationMetrics(AutoReviewModel):
    """Aggregated information metrics for a single topic across depth levels."""

    claims_per_depth: dict[str, int]
    new_claims_per_increment: dict[str, int]
    new_claims_by_category: dict[str, dict[str, int]]
    claim_novelty_rate: dict[str, float]
    concepts_per_depth: dict[str, int]
    concept_growth: dict[str, int]
    claims_per_1k_words: dict[str, float]
    concepts_per_1k_words: dict[str, float]
    citations_per_claim: dict[str, float]
    concept_overlap: dict[str, float] | None = None
    reference_claim_coverage: dict[str, float] | None = None


class DepthRunConfig(AutoReviewModel):
    """Configuration for a single depth comparison run."""

    topic: str
    domain: str
    depth: str
    generated_path: str
    reference_path: str | None = None
    evaluation_path: str | None = None
    tier: str | None = None


class DepthComparisonResult(AutoReviewModel):
    """Full comparison result for a single topic across depths."""

    topic: str
    domain: str
    depths: list[str]
    evaluation_scores: dict[str, float]
    information_metrics: InformationMetrics
    cost_per_depth: dict[str, float] | None = None
