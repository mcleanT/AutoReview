"""Evidence map models for thematic clustering and contradiction detection."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from autoreview.models.base import AutoReviewModel, TimestampedModel


class SubTheme(AutoReviewModel):
    """A sub-theme within a broader thematic cluster.

    Groups related findings under a narrower topic within a parent theme.
    """

    name: str
    description: str
    paper_ids: list[str] = Field(default_factory=list)
    finding_claims: list[str] = Field(default_factory=list)


class Theme(TimestampedModel):
    """A named cluster of related findings.

    Represents a major thematic area identified during evidence
    clustering, containing papers and optional sub-themes.
    """

    name: str
    description: str
    paper_ids: list[str] = Field(default_factory=list)
    sub_themes: list[SubTheme] = Field(default_factory=list)


class ConsensusClaim(AutoReviewModel):
    """A claim supported by multiple papers.

    Represents a point of agreement across the literature,
    with evidence strength proportional to the number of
    supporting sources.
    """

    claim: str
    supporting_paper_ids: list[str]
    strength: str
    evidence_count: int


class Contradiction(AutoReviewModel):
    """Opposing claims with supporting papers.

    Captures conflicting findings in the literature along with
    a hypothesis about why the disagreement exists.
    """

    claim_a: str
    claim_b: str
    paper_ids_a: list[str]
    paper_ids_b: list[str]
    possible_explanation: str | None = None
    theme: str | None = None


class GapSeverity(StrEnum):
    """Severity of an identified gap in the literature."""

    MAJOR = "major"
    MINOR = "minor"


class IdentifiedGap(AutoReviewModel):
    """An expected sub-topic with insufficient evidence.

    Identified by comparing the scope document's expected coverage
    against the actual findings in the evidence map.
    """

    expected_topic: str
    current_coverage: str
    severity: GapSeverity
    suggested_queries: list[str] = Field(default_factory=list)


class EvidenceMap(TimestampedModel):
    """Full evidence landscape built from paper extractions.

    The analytical core of the pipeline: themes, consensus claims,
    contradictions, gaps, and paper-to-theme mappings that drive
    outline generation and section writing.
    """

    themes: list[Theme] = Field(default_factory=list)
    consensus_claims: list[ConsensusClaim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    gaps: list[IdentifiedGap] = Field(default_factory=list)
    paper_theme_mapping: dict[str, list[str]] = Field(default_factory=dict)
    coverage_score: float = 0.0
    evidence_chains: list[Any] = Field(default_factory=list)
    temporal_progressions: list[Any] = Field(default_factory=list)
    enriched_contradictions: list[Any] = Field(default_factory=list)
