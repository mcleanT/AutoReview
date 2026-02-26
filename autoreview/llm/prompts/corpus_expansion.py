"""Prompt models and builders for corpus expansion — turning enrichment insights into citable evidence."""

from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class CorpusExpansionQuery(AutoReviewModel):
    """A single targeted search query for primary research papers."""

    query: str
    source_section_id: str
    rationale: str
    target_concepts: list[str] = Field(default_factory=list)


class CorpusExpansionQueryResult(AutoReviewModel):
    """LLM response: search queries generated for one section's corpus expansion."""

    section_id: str
    queries: list[CorpusExpansionQuery] = Field(default_factory=list)


CORPUS_EXPANSION_SYSTEM_PROMPT = """\
You are an expert research librarian generating search queries to find PRIMARY RESEARCH \
PAPERS — original studies with citable evidence — that would strengthen a review paper section.

You are given concepts and cross-field connections discovered during contextual enrichment. \
Your job is to turn these insights into targeted queries that retrieve papers with \
empirical data, experimental results, or systematic analyses that can be cited as evidence.

IMPORTANT distinctions:
- You are NOT looking for background material, reviews, or textbook content
- You ARE looking for original research papers with results, data, and conclusions
- Each query should target a specific concept or connection that currently lacks \
  citable evidence in the section
- Avoid queries that would retrieve papers already in the corpus

Generate 3–5 focused queries per section. Each query should:
1. Target a specific concept or connection from the enrichment insights
2. Be phrased to retrieve empirical/experimental papers (not reviews)
3. Include relevant methodological terms to improve precision
"""


def build_corpus_expansion_query_prompt(
    section_id: str,
    section_title: str,
    section_description: str,
    key_concepts: list[str],
    cross_field_connections: list[str],
    existing_paper_ids: list[str],
    scope_document: str,
) -> str:
    concepts_text = "\n".join(f"- {c}" for c in key_concepts) if key_concepts else "(none)"
    connections_text = (
        "\n".join(f"- {c}" for c in cross_field_connections)
        if cross_field_connections
        else "(none)"
    )
    existing_text = (
        f"{len(existing_paper_ids)} papers already assigned"
        if existing_paper_ids
        else "No papers currently assigned"
    )

    return f"""\
## Section
**ID:** {section_id}
**Title:** {section_title}
**Description:** {section_description}
**Existing corpus:** {existing_text}

## Key Concepts Discovered During Enrichment
{concepts_text}

## Cross-Field Connections
{connections_text}

## Review Scope
{scope_document}

Generate 3–5 search queries targeting PRIMARY RESEARCH PAPERS (original studies with \
empirical data) that would provide citable evidence for the concepts and connections \
listed above. Each query should target a different concept or connection.

Do NOT generate queries for material already well-covered in the section's existing corpus. \
Focus on the enrichment-discovered concepts that currently lack direct evidence.
"""
