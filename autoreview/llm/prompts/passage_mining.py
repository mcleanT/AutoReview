"""Prompt models and builders for passage-based evidence mining."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class UndercitedClaim(AutoReviewModel):
    """A claim in a draft section that needs more supporting evidence."""
    text: str
    evidence_weakness: str
    current_citations: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"]


class SectionMiningResult(AutoReviewModel):
    """Output of passage mining for a single section."""
    section_id: str
    undercited_claims: list[UndercitedClaim] = Field(default_factory=list)
    topic_expansions: list[str] = Field(default_factory=list)


SectionMiningResult.model_rebuild()


PASSAGE_MINING_SYSTEM_PROMPT = """\
You are a scientific editor reviewing draft review-paper sections to identify where \
additional evidence would strengthen the text. Focus on factual claims, mechanistic \
assertions, and quantitative statements. Ignore stylistic issues.
"""


def build_passage_mining_prompt(
    section_id: str,
    section_text: str,
    cited_paper_summaries: str,
) -> str:
    return f"""\
## Section ID: {section_id}

## Draft Text
{section_text}

## Papers Already Cited in This Section
{cited_paper_summaries}

Identify:
1. **Unsupported claims** — factual assertions with no [@paper_id] citation.
2. **Undercited claims** — important claims cited by only 1–2 sources, especially \
if described as "preliminary", "limited", or "conflicting".
3. **Topic expansions** — sub-topics mentioned briefly that could be substantiated \
by additional literature.

For each undercited or unsupported claim, generate 2–3 specific PubMed/Semantic Scholar \
search queries. Prioritize by impact on review quality: "high" (key mechanistic or \
quantitative claims), "medium" (supporting context), "low" (minor details).
Return at most 8 claims total.
"""
