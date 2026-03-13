from __future__ import annotations

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
