from __future__ import annotations

import pytest

from autoreview.evaluation.arise_rubric import ARISERubricScorer
from autoreview.evaluation.models import ARISERubricResult
from autoreview.llm.prompts.arise_evaluation import (
    ARISECategoryRaw,
    ARISERawResponse,
    ARISESubcriterionRaw,
)
from autoreview.llm.provider import LLMStructuredResponse

# ---------------------------------------------------------------------------
# Helpers — build a canned ARISERawResponse with known scores
# ---------------------------------------------------------------------------

_CATEGORIES_SPEC: list[tuple[str, list[tuple[str, float]]]] = [
    (
        "Significance & Contribution",
        [
            ("Importance of topic", 4.0),
            ("Originality of perspective", 3.0),
            ("Contribution to field", 5.0),
        ],
    ),
    (
        "Presentation Quality",
        [
            ("Clarity of writing", 3.0),
            ("Logical organization", 4.0),
            ("Figures/tables discussion", 2.0),
        ],
    ),
    (
        "Thoroughness",
        [
            ("Breadth of literature", 5.0),
            ("Depth of analysis", 4.0),
            ("Identification of gaps", 3.0),
        ],
    ),
    (
        "Critical Analysis",
        [
            ("Balanced perspective", 4.0),
            ("Evidence-based arguments", 3.0),
            ("Identification of limitations", 4.0),
        ],
    ),
    (
        "Methodology",
        [
            ("Search strategy transparency", 2.0),
            ("Inclusion/exclusion criteria", 3.0),
            ("Synthesis methodology", 4.0),
        ],
    ),
    (
        "Currency & Relevance",
        [
            ("Timeliness of references", 5.0),
            ("Relevance to current debates", 4.0),
        ],
    ),
    (
        "Ethical & Broader Impact",
        [
            ("Ethical considerations", 3.0),
            ("Societal implications", 2.0),
            ("Future directions", 4.0),
        ],
    ),
]

# Pre-compute expected values from spec
_EXPECTED_TOTAL = sum(score for _, subs in _CATEGORIES_SPEC for _, score in subs)
_EXPECTED_NORMALIZED = _EXPECTED_TOTAL / 100.0


def _make_canned_response() -> ARISERawResponse:
    categories = [
        ARISECategoryRaw(
            category=cat_name,
            subcriteria=[
                ARISESubcriterionRaw(name=name, score=score, comment="OK") for name, score in subs
            ],
        )
        for cat_name, subs in _CATEGORIES_SPEC
    ]
    return ARISERawResponse(categories=categories)


class MockARISELLM:
    """Mock LLM that returns a canned ARISERawResponse."""

    async def generate_structured(
        self,
        prompt: str,
        response_model: type,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.0,
        model_override: str | None = None,
    ) -> LLMStructuredResponse:
        return LLMStructuredResponse(
            parsed=_make_canned_response(),
            input_tokens=500,
            output_tokens=300,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arise_scorer_aggregation() -> None:
    """Mock LLM returns known scores; verify category means and total."""
    scorer = ARISERubricScorer(MockARISELLM())
    result = await scorer.score("Some review text")

    assert isinstance(result, ARISERubricResult)

    # Check category scores (means of their subcriteria)
    cat_map = {c.category: c for c in result.category_scores}

    # "Significance & Contribution": (4+3+5)/3 = 4.0
    assert cat_map["Significance & Contribution"].category_score == pytest.approx(4.0, abs=1e-4)

    # "Presentation Quality": (3+4+2)/3 = 3.0
    assert cat_map["Presentation Quality"].category_score == pytest.approx(3.0, abs=1e-4)

    # "Currency & Relevance": (5+4)/2 = 4.5
    assert cat_map["Currency & Relevance"].category_score == pytest.approx(4.5, abs=1e-4)

    # total_score = sum of all 20 individual sub-scores
    assert result.total_score == pytest.approx(_EXPECTED_TOTAL, abs=1e-3)


@pytest.mark.asyncio
async def test_arise_all_categories_present() -> None:
    """All 7 categories and 20 subcriteria must appear in the result."""
    scorer = ARISERubricScorer(MockARISELLM())
    result = await scorer.score("Some review text")

    assert len(result.category_scores) == 7

    total_subcriteria = sum(len(c.subcriteria) for c in result.category_scores)
    assert total_subcriteria == 20

    expected_categories = {cat for cat, _ in _CATEGORIES_SPEC}
    actual_categories = {c.category for c in result.category_scores}
    assert actual_categories == expected_categories


@pytest.mark.asyncio
async def test_arise_normalized_score() -> None:
    """normalized_score must equal total_score / 100."""
    scorer = ARISERubricScorer(MockARISELLM())
    result = await scorer.score("Some review text")

    assert result.normalized_score == pytest.approx(result.total_score / 100.0, rel=1e-6)
    assert result.normalized_score == pytest.approx(_EXPECTED_NORMALIZED, rel=1e-6)


@pytest.mark.asyncio
async def test_arise_score_clamping() -> None:
    """If the LLM returns a score outside [1, 5], it must be clamped."""

    class ClampTestLLM:
        async def generate_structured(
            self,
            prompt: str,
            response_model: type,
            system: str = "",
            max_tokens: int | None = None,
            temperature: float = 0.0,
            model_override: str | None = None,
        ) -> LLMStructuredResponse:
            # Build the full 7-category response but override two scores
            categories = []
            for i, (cat_name, subs) in enumerate(_CATEGORIES_SPEC):
                subcriteria_raw = []
                for j, (name, score) in enumerate(subs):
                    # First subcriterion of first category: score too high (6.0 → clamp to 5)
                    if i == 0 and j == 0:
                        score = 6.0
                    # First subcriterion of second category: score too low (0.0 → clamp to 1)
                    elif i == 1 and j == 0:
                        score = 0.0
                    subcriteria_raw.append(
                        ARISESubcriterionRaw(name=name, score=score, comment="test")
                    )
                categories.append(ARISECategoryRaw(category=cat_name, subcriteria=subcriteria_raw))
            return LLMStructuredResponse(
                parsed=ARISERawResponse(categories=categories),
                input_tokens=100,
                output_tokens=100,
            )

    scorer = ARISERubricScorer(ClampTestLLM())
    result = await scorer.score("Some review text")

    # Locate the first subcriterion of "Significance & Contribution"
    sig_cat = next(c for c in result.category_scores if c.category == "Significance & Contribution")
    assert sig_cat.subcriteria[0].score == pytest.approx(5.0)  # clamped from 6.0

    # Locate the first subcriterion of "Presentation Quality"
    pres_cat = next(c for c in result.category_scores if c.category == "Presentation Quality")
    assert pres_cat.subcriteria[0].score == pytest.approx(1.0)  # clamped from 0.0

    # All scores must be in [1, 5]
    for cat in result.category_scores:
        for sub in cat.subcriteria:
            assert 1.0 <= sub.score <= 5.0, f"Out-of-range score {sub.score} for {sub.name}"
