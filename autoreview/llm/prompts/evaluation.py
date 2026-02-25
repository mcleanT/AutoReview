from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel

EVALUATION_SYSTEM_PROMPT = (
    "You are an expert scientific editor evaluating academic review papers. "
    "Score both papers objectively on the same rubric. Respond with structured JSON only."
)

_SYNTHESIS_RUBRIC = """\
Score each review 0-5 for each dimension (mean = overall_score):
- cross_paper: identifies patterns/trends across papers vs. paper-by-paper summaries
- contradiction_handling: explicitly identifies and explains conflicting findings
- gap_identification: names research gaps and future directions
- evidence_weighting: differentiates strong vs. weak/preliminary evidence
- narrative_coherence: argument builds logically and cohesively"""

_TOPIC_INSTRUCTION = """\
Extract 10-30 distinct sub-topics from each review.
Identify which reference sub-topics are also covered by the generated review.
generated_coverage = len(topics_matched) / len(topics_in_reference)"""

_WRITING_RUBRIC = """\
Score each review 0-5 for:
- clarity: arguments stated clearly and precisely
- coherence: paragraphs and sections flow logically
- academic_register: appropriately formal and precise tone
- flow: smooth transitions between sections and ideas
Mean of 4 dimensions = overall_score."""


class SynthesisScoreResult(AutoReviewModel):
    generated_score: float = Field(..., ge=0, le=5)
    reference_score: float = Field(..., ge=0, le=5)
    dimension_scores_generated: dict[str, float]
    dimension_scores_reference: dict[str, float]
    generated_observations: str
    reference_observations: str


class TopicCoverageResult(AutoReviewModel):
    topics_in_generated: list[str]
    topics_in_reference: list[str]
    topics_matched: list[str]
    topics_missed: list[str]
    generated_coverage: float = Field(..., ge=0, le=1)


class WritingQualityResult(AutoReviewModel):
    generated_score: float = Field(..., ge=0, le=5)
    reference_score: float = Field(..., ge=0, le=5)
    dimension_scores_generated: dict[str, float]
    dimension_scores_reference: dict[str, float]
    generated_strengths: str
    generated_weaknesses: str


def build_synthesis_prompt(
    generated_text: str, reference_text: str, max_chars: int = 40000
) -> str:
    return (
        f"{_SYNTHESIS_RUBRIC}\n\n"
        f"=== GENERATED REVIEW ===\n{generated_text[:max_chars]}\n\n"
        f"=== REFERENCE REVIEW ===\n{reference_text[:max_chars]}"
    )


def build_topic_coverage_prompt(
    generated_text: str, reference_text: str, max_chars: int = 40000
) -> str:
    return (
        f"{_TOPIC_INSTRUCTION}\n\n"
        f"=== GENERATED REVIEW ===\n{generated_text[:max_chars]}\n\n"
        f"=== REFERENCE REVIEW ===\n{reference_text[:max_chars]}"
    )


def build_writing_quality_prompt(
    generated_text: str, reference_text: str, max_chars: int = 40000
) -> str:
    return (
        f"{_WRITING_RUBRIC}\n\n"
        f"=== GENERATED REVIEW ===\n{generated_text[:max_chars]}\n\n"
        f"=== REFERENCE REVIEW ===\n{reference_text[:max_chars]}"
    )
