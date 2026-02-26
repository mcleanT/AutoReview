"""Paper models for search results and screening."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from autoreview.models.base import AutoReviewModel, TimestampedModel


class CandidatePaper(TimestampedModel):
    """A raw search result from a literature database.

    Represents a paper discovered during the multi-source search phase,
    before screening for relevance.
    """

    title: str
    authors: list[str]
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    abstract: str | None = None
    source_database: str
    external_ids: dict[str, str] = Field(default_factory=dict)
    citation_count: int | None = None
    full_text: str | None = None
    full_text_source: str | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi(cls, v: str | None) -> str | None:
        """Normalize DOIs by lowercasing and stripping common URL prefixes."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        v = v.lower()
        # Strip common DOI URL prefixes
        v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
        return v

    @property
    def doi_normalized(self) -> str | None:
        """Return the normalized DOI (lowercase, no URL prefix).

        Since the field_validator already normalizes on assignment,
        this property simply returns the stored value.
        """
        return self.doi


class ScreenedPaper(TimestampedModel):
    """A paper that has been screened for relevance.

    Contains the original candidate paper along with an LLM-assigned
    relevance score and rationale for the screening decision.
    """

    paper: CandidatePaper
    relevance_score: int = Field(ge=1, le=5)
    rationale: str
    include: bool
