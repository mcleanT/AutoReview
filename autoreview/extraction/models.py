"""Extraction models for structured paper analysis."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from autoreview.models.base import AutoReviewModel, TimestampedModel


class StudyDesign(StrEnum):
    """Study design classification for a paper."""

    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    IN_VITRO = "in_vitro"
    COMPUTATIONAL = "computational"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    NARRATIVE_REVIEW = "narrative_review"
    OTHER = "other"


class EvidenceStrength(StrEnum):
    """Strength of evidence supporting a finding."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    PRELIMINARY = "preliminary"


class RelationshipType(StrEnum):
    """Type of relationship between papers or claims."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    REPLICATES = "replicates"


class Finding(AutoReviewModel):
    """A single claim extracted from a paper.

    Represents an atomic finding with its evidence strength
    and optional quantitative backing.
    """

    claim: str
    evidence_strength: EvidenceStrength
    quantitative_result: str | None = None
    context: str | None = None
    paper_id: str


class RelationshipClaim(AutoReviewModel):
    """A relationship between two papers.

    Captures how one paper's findings relate to another's,
    enabling contradiction detection and evidence chain building.
    """

    source_paper_id: str
    target_paper_id: str
    relationship_type: RelationshipType
    description: str


class MethodologyRecord(AutoReviewModel):
    """Methodology details extracted from a paper.

    Captures approach, datasets, metrics, and reproducibility
    information for methodological comparison across papers.
    """

    approach: str
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reproducibility_notes: str | None = None


class PaperExtraction(TimestampedModel):
    """Full structured extraction from a single paper.

    Contains all findings, methods, limitations, and relationships
    extracted by the LLM during the parallel extraction phase.
    """

    paper_id: str
    key_findings: list[Finding]
    methods_summary: str
    limitations: str
    relationships: list[RelationshipClaim] = Field(default_factory=list)
    methodology_details: MethodologyRecord | None = None
    domain_specific_fields: dict[str, Any] = Field(default_factory=dict)
    study_design: StudyDesign | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_size: int | None = None
