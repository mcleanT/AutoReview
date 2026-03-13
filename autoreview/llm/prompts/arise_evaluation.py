from __future__ import annotations

from autoreview.models.base import AutoReviewModel

ARISE_SYSTEM_PROMPT = (
    "You are an expert peer reviewer and scientific editor with deep experience evaluating "
    "academic review papers. Your task is to score a review paper against the ARISE rubric "
    "(Assessment of Review papers In Science and Engineering). "
    "Apply each criterion objectively and consistently. "
    "Respond with structured JSON only — no preamble or explanation outside the JSON."
)

ARISE_RUBRIC_DEFINITION = """\
ARISE RUBRIC — score each sub-criterion 1 (poor) to 5 (excellent).

Category 1: Significance & Contribution
  1.1 Importance of topic          — Does the review address a topic of substantial scientific importance?
  1.2 Originality of perspective   — Does it offer a novel synthesis, framing, or angle?
  1.3 Contribution to field        — Does it meaningfully advance understanding beyond existing reviews?

Category 2: Presentation Quality
  2.1 Clarity of writing           — Is the prose clear, precise, and accessible to the target audience?
  2.2 Logical organization         — Are sections sequenced coherently with clear signposting?
  2.3 Figures/tables discussion    — Are figures, tables, and visual aids used effectively and discussed?

Category 3: Thoroughness
  3.1 Breadth of literature        — Does it cover the major relevant literature comprehensively?
  3.2 Depth of analysis            — Does it go beyond summary to analyse and interpret findings?
  3.3 Identification of gaps       — Does it explicitly identify open questions and research gaps?

Category 4: Critical Analysis
  4.1 Balanced perspective         — Does it represent multiple viewpoints fairly?
  4.2 Evidence-based arguments     — Are claims grounded in cited evidence?
  4.3 Identification of limitations — Does it critically evaluate the limitations of reviewed studies?

Category 5: Methodology
  5.1 Search strategy transparency — Is the literature search strategy described?
  5.2 Inclusion/exclusion criteria — Are inclusion/exclusion criteria stated or inferable?
  5.3 Synthesis methodology        — Is the method for synthesising evidence explained?

Category 6: Currency & Relevance
  6.1 Timeliness of references     — Are the most recent and relevant papers cited?
  6.2 Relevance to current debates — Does it engage with active scientific debates?

Category 7: Ethical & Broader Impact
  7.1 Ethical considerations       — Are ethical dimensions of the topic acknowledged?
  7.2 Societal implications        — Are broader societal impacts discussed?
  7.3 Future directions            — Does it offer concrete, actionable future research directions?"""


def build_arise_evaluation_prompt(review_text: str, max_chars: int = 40000) -> str:
    """Build the user prompt for ARISE rubric evaluation."""
    truncated = review_text[:max_chars]
    return (
        f"{ARISE_RUBRIC_DEFINITION}\n\n"
        "Score every sub-criterion listed above. "
        "Return a JSON object with a 'categories' array. "
        "Each element must have 'category' (the category name, e.g. 'Significance & Contribution'), "
        "and 'subcriteria' (array of objects with 'name', 'score' (1-5 float), 'comment').\n\n"
        f"=== REVIEW TO EVALUATE ===\n{truncated}"
    )


# ---------------------------------------------------------------------------
# Pydantic response models for structured LLM output parsing
# ---------------------------------------------------------------------------


class ARISESubcriterionRaw(AutoReviewModel):
    name: str
    score: float  # 1-5
    comment: str


class ARISECategoryRaw(AutoReviewModel):
    category: str
    subcriteria: list[ARISESubcriterionRaw]


class ARISERawResponse(AutoReviewModel):
    categories: list[ARISECategoryRaw]
