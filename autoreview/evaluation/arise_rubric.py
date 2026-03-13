from __future__ import annotations

import structlog

from autoreview.evaluation.models import (
    ARISECategoryScore,
    ARISERubricResult,
    ARISESubcriterionScore,
)
from autoreview.llm.prompts.arise_evaluation import (
    ARISE_SYSTEM_PROMPT,
    ARISERawResponse,
    build_arise_evaluation_prompt,
)

logger = structlog.get_logger()

_SCORE_MIN = 1.0
_SCORE_MAX = 5.0


def _clamp(value: float) -> float:
    """Clamp a score to the valid 1-5 range."""
    return max(_SCORE_MIN, min(_SCORE_MAX, value))


class ARISERubricScorer:
    """Score a review paper against the ARISE rubric using an LLM judge."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def score(self, review_text: str) -> ARISERubricResult:
        """Score the review text and return an ARISERubricResult."""
        prompt = build_arise_evaluation_prompt(review_text)
        resp = await self.llm.generate_structured(
            prompt,
            ARISERawResponse,
            system=ARISE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        raw: ARISERawResponse = resp.parsed  # type: ignore[assignment]

        category_scores: list[ARISECategoryScore] = []
        for cat in raw.categories:
            subcriteria = [
                ARISESubcriterionScore(
                    name=sub.name,
                    score=_clamp(sub.score),
                    comment=sub.comment,
                )
                for sub in cat.subcriteria
            ]
            cat_mean = sum(s.score for s in subcriteria) / len(subcriteria) if subcriteria else 0.0
            category_scores.append(
                ARISECategoryScore(
                    category=cat.category,
                    subcriteria=subcriteria,
                    category_score=round(cat_mean, 4),
                )
            )

        all_subcriteria_scores = [sub.score for cat in category_scores for sub in cat.subcriteria]
        total_score = round(sum(all_subcriteria_scores), 4)
        normalized_score = round(total_score / 100.0, 6)

        logger.info(
            "arise_rubric.score",
            total_score=total_score,
            normalized_score=normalized_score,
            n_categories=len(category_scores),
            n_subcriteria=len(all_subcriteria_scores),
        )

        return ARISERubricResult(
            category_scores=category_scores,
            total_score=total_score,
            normalized_score=normalized_score,
        )
