"""Critique models for the self-critique system."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from autoreview.models.base import AutoReviewModel, TimestampedModel


class CritiqueSeverity(str, Enum):
    """Severity level of a critique issue."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class CritiqueTarget(str, Enum):
    """What artifact the critique evaluates."""

    OUTLINE = "outline"
    SECTION = "section"
    FULL_DRAFT = "full_draft"


class CritiqueIssue(AutoReviewModel):
    """A single issue identified during critique.

    Localized to a specific section or element, with a severity
    rating and optional suggested fix.
    """

    severity: CritiqueSeverity
    location: str
    description: str
    suggested_fix: str | None = None


class CritiqueReport(TimestampedModel):
    """Complete critique report for an outline, section, or full draft.

    Contains an overall pass/fail decision, numeric scores across
    configurable dimensions, and a list of specific issues.
    """

    target: CritiqueTarget
    target_id: str | None = None
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[CritiqueIssue] = Field(default_factory=list)
    identified_gaps: list[str] = Field(default_factory=list)
