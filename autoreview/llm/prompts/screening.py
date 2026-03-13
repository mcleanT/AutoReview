from __future__ import annotations

from autoreview.models.base import AutoReviewModel


class ScreeningDecision(AutoReviewModel):
    """Decision for a single paper in a screening batch."""

    paper_index: int
    relevance_score: int  # 1-5
    rationale: str


class ScreeningBatchResult(AutoReviewModel):
    """Result of screening a batch of papers."""

    decisions: list[ScreeningDecision]


SCREENING_SYSTEM_PROMPT = """\
You are an expert systematic reviewer screening papers for inclusion in a scientific review. \
For each paper, assess its relevance to the review topic on a scale of 1-5:

5 = Highly relevant, directly addresses the core topic
4 = Relevant, addresses the topic with useful findings or methodology
3 = Moderately relevant, addresses related aspects that may inform the review
2 = Marginally relevant, tangentially related
1 = Not relevant to the review topic

Consider: topic relevance, methodology appropriateness, unique contributions, and recency. \
Provide a brief rationale for each score.
"""


def build_screening_prompt(
    scope_document: str,
    papers: list[dict[str, str]],
) -> str:
    """Build a screening prompt for a batch of papers."""
    paper_blocks = []
    for p in papers:
        abstract = p.get("abstract") or "(No abstract available)"
        paper_blocks.append(f"[Paper {p['index']}]\nTitle: {p['title']}\nAbstract: {abstract}")

    return f"""\
## Review Scope
{scope_document}

## Papers to Screen
{chr(10).join(paper_blocks)}

Score each paper's relevance (1-5) to the review scope. Provide a brief rationale for each.
"""
