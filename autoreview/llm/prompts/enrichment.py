"""Prompt models and builders for contextual enrichment."""

from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel
from autoreview.models.enrichment import ContextualExtraction, EnrichmentQuery


class EnrichmentQueryResult(AutoReviewModel):
    """LLM response model for enrichment query generation."""

    section_id: str
    section_title: str
    queries: list[EnrichmentQuery] = Field(default_factory=list)


class ContextualExtractionResult(AutoReviewModel):
    """LLM response model for contextual extraction from a single paper."""

    paper_id: str
    paper_title: str
    background_summary: str
    key_concepts: list[str] = Field(default_factory=list)
    methodological_context: str = ""
    cross_field_connections: list[str] = Field(default_factory=list)
    relevance_to_section: str = ""

    def to_contextual_extraction(self) -> ContextualExtraction:
        return ContextualExtraction(
            paper_id=self.paper_id,
            paper_title=self.paper_title,
            background_summary=self.background_summary,
            key_concepts=self.key_concepts,
            methodological_context=self.methodological_context,
            cross_field_connections=self.cross_field_connections,
            relevance_to_section=self.relevance_to_section,
        )


ENRICHMENT_QUERY_SYSTEM_PROMPT = """\
You are an expert research librarian identifying ADJACENT knowledge that would make \
a review paper section richer and more contextualized. You are NOT looking for more \
papers on the same topic — you are looking for:

- **Background**: foundational concepts a non-specialist reader needs
- **Methodological context**: how key techniques work, their strengths and limitations
- **Clinical implications**: translational or practical significance of findings
- **Cross-field connections**: related work in adjacent disciplines that illuminates the topic
- **Mechanistic detail**: underlying biological, chemical, or physical mechanisms
- **Historical context**: how understanding of this topic evolved over time

Generate 3–5 focused queries per section. Each query should target a DIFFERENT \
type of enrichment. Prefer queries that would retrieve review articles, textbook-level \
explanations, or seminal papers from adjacent fields.
"""


CONTEXTUAL_EXTRACTION_SYSTEM_PROMPT = """\
You are extracting contextual background information from a paper to enrich a review \
section. Focus on:

1. **Background summary**: What foundational knowledge does this paper provide?
2. **Key concepts**: Important terms, mechanisms, or frameworks explained
3. **Methodological context**: How methods described here relate to the review topic
4. **Cross-field connections**: Links to other disciplines or research areas
5. **Relevance**: How this material helps frame or contextualize the review section

This is supplementary material — extract what helps a reader understand context, \
not primary evidence for the review's main claims.
"""


def build_enrichment_query_prompt(
    section_id: str,
    section_title: str,
    section_description: str,
    narrative_directive: str,
    scope_document: str,
) -> str:
    return f"""\
## Section to Enrich
**ID:** {section_id}
**Title:** {section_title}
**Description:** {section_description}

## Narrative Directive
{narrative_directive or "(No narrative directive available)"}

## Review Scope
{scope_document}

Generate 3–5 search queries that would retrieve ADJACENT contextual material \
for this section. Each query should target a different enrichment type \
(background, methodological_context, clinical_implication, cross_field_connection, \
mechanistic_detail, or historical_context).

Do NOT generate queries for the section's primary topic — those papers are already \
retrieved. Focus on what would make the section deeper and more informative for \
readers who are not specialists in this exact sub-field.
"""


class BatchContextualExtractionResult(AutoReviewModel):
    """LLM response model for batched contextual extraction from multiple papers."""

    extractions: list[ContextualExtractionResult] = Field(default_factory=list)


def build_contextual_extraction_prompt(
    paper_title: str,
    paper_abstract: str,
    section_title: str,
    section_description: str,
) -> str:
    return f"""\
## Paper
**Title:** {paper_title}
**Abstract:** {paper_abstract}

## Target Review Section
**Title:** {section_title}
**Description:** {section_description}

Extract contextual background information from this paper that would help enrich \
the review section above. Focus on foundational knowledge, key concepts, \
methodological insights, and cross-disciplinary connections.
"""


def build_batch_contextual_extraction_prompt(
    papers: list[dict[str, str]],
    section_title: str,
    section_description: str,
) -> str:
    """Build a prompt for extracting contextual info from multiple papers at once.

    Args:
        papers: List of dicts with 'id', 'title', 'abstract' keys.
        section_title: Target section title.
        section_description: Target section description.
    """
    paper_blocks = []
    for i, p in enumerate(papers):
        paper_blocks.append(
            f"### Paper {i + 1}\n"
            f"**ID:** {p['id']}\n"
            f"**Title:** {p['title']}\n"
            f"**Abstract:** {p['abstract']}"
        )
    papers_text = "\n\n".join(paper_blocks)

    return f"""\
## Papers to Extract From

{papers_text}

## Target Review Section
**Title:** {section_title}
**Description:** {section_description}

For EACH paper above, extract contextual background information that would help \
enrich the review section. Return one extraction per paper with matching paper_id \
and paper_title. Focus on foundational knowledge, key concepts, methodological \
insights, and cross-disciplinary connections.
"""
