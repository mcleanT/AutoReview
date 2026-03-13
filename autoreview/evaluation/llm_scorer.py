from __future__ import annotations

from typing import TYPE_CHECKING, Any  # noqa: F401

import structlog

from autoreview.evaluation.models import SynthesisScore, TopicCoverageScore, WritingQualityScore
from autoreview.llm.prompts.evaluation import (
    EVALUATION_SYSTEM_PROMPT,
    SynthesisScoreResult,
    TopicCoverageResult,
    WritingQualityResult,
    build_synthesis_prompt,
    build_topic_coverage_prompt,
    build_writing_quality_prompt,
)

logger = structlog.get_logger()


class LLMScorer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def score_synthesis(self, generated_text: str, reference_text: str) -> SynthesisScore:
        prompt = build_synthesis_prompt(generated_text, reference_text)
        resp = await self.llm.generate_structured(
            prompt, SynthesisScoreResult, system=EVALUATION_SYSTEM_PROMPT
        )
        r = resp.parsed
        logger.info(
            "llm_scorer.synthesis",
            generated=r.generated_score,
            reference=r.reference_score,
        )
        return SynthesisScore(
            generated_score=r.generated_score,
            reference_score=r.reference_score,
            delta=round(r.generated_score - r.reference_score, 3),
            dimension_scores=r.dimension_scores_generated,
            generated_observations=r.generated_observations,
            reference_observations=r.reference_observations,
        )

    async def score_topic_coverage(
        self, generated_text: str, reference_text: str
    ) -> TopicCoverageScore:
        prompt = build_topic_coverage_prompt(generated_text, reference_text)
        resp = await self.llm.generate_structured(
            prompt, TopicCoverageResult, system=EVALUATION_SYSTEM_PROMPT
        )
        r = resp.parsed
        logger.info("llm_scorer.topic_coverage", coverage=r.generated_coverage)
        matched_set = set(r.topics_matched)
        return TopicCoverageScore(
            generated_coverage=r.generated_coverage,
            reference_coverage=1.0,
            topics_in_both=r.topics_matched,
            topics_only_in_reference=r.topics_missed,
            topics_only_in_generated=[t for t in r.topics_in_generated if t not in matched_set],
        )

    async def score_writing_quality(
        self, generated_text: str, reference_text: str
    ) -> WritingQualityScore:
        prompt = build_writing_quality_prompt(generated_text, reference_text)
        resp = await self.llm.generate_structured(
            prompt, WritingQualityResult, system=EVALUATION_SYSTEM_PROMPT
        )
        r = resp.parsed
        logger.info(
            "llm_scorer.writing_quality", generated=r.generated_score, reference=r.reference_score
        )
        return WritingQualityScore(
            generated_score=r.generated_score,
            reference_score=r.reference_score,
            delta=round(r.generated_score - r.reference_score, 3),
            dimension_scores=r.dimension_scores_generated,
        )
