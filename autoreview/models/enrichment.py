"""Contextual enrichment models for broadening review scope."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoreview.models.base import AutoReviewModel

EnrichmentType = Literal[
    "background",
    "methodological_context",
    "clinical_implication",
    "cross_field_connection",
    "mechanistic_detail",
    "historical_context",
]


class EnrichmentQuery(AutoReviewModel):
    """A single query designed to retrieve adjacent contextual material."""

    query: str
    rationale: str
    enrichment_type: EnrichmentType


class SectionEnrichmentQueries(AutoReviewModel):
    """Enrichment queries generated for one outline section."""

    section_id: str
    section_title: str
    queries: list[EnrichmentQuery] = Field(default_factory=list)


class ContextualExtraction(AutoReviewModel):
    """Lightweight contextual extraction from a single paper."""

    paper_id: str
    paper_title: str
    background_summary: str
    key_concepts: list[str] = Field(default_factory=list)
    methodological_context: str = ""
    cross_field_connections: list[str] = Field(default_factory=list)
    relevance_to_section: str = ""


class SectionEnrichment(AutoReviewModel):
    """Complete enrichment results for one outline section."""

    section_id: str
    section_title: str
    queries_generated: list[EnrichmentQuery] = Field(default_factory=list)
    papers_found: int = 0
    papers_screened: int = 0
    contextual_extractions: list[ContextualExtraction] = Field(default_factory=list)


class CorpusExpansionResult(AutoReviewModel):
    """Results of corpus expansion for one section."""

    section_id: str
    section_title: str
    queries_generated: int = 0
    papers_found: int = 0
    papers_screened: int = 0
    papers_extracted: int = 0
    new_paper_ids: list[str] = Field(default_factory=list)
