"""Narrative planning models for the NarrativeArchitect pipeline stage."""

from __future__ import annotations

from autoreview.models.base import AutoReviewModel


class SectionNarrativeDirective(AutoReviewModel):
    """Narrative planning directive for a single review section.

    Provides soft guidance to the section writer about the rhetorical
    role of the section, its central claim, structural approach,
    and how it connects to adjacent sections.
    """

    section_id: str
    narrative_role: str
    """The role this section plays in the paper's argument.

    e.g. 'sets up the central tension'.
    """

    central_claim: str
    """The thesis statement or central claim for this section."""

    structural_suggestion: str
    """Suggested structural approach.

    e.g. 'comparative: mechanistic vs. epidemiological evidence'.
    """

    key_insights: list[str]
    """3–5 analytical insights to foreground (not paper IDs — synthesized observations)."""

    transition_from_prev: str
    """Hint for the opening transition from the preceding section."""

    transition_to_next: str
    """Hint for the closing sentence that sets up the following section."""


class NarrativePlan(AutoReviewModel):
    """Top-level narrative plan for a complete review paper.

    Produced by the NarrativeArchitect before section writing begins.
    Contains the paper's central argument and per-section directives.
    """

    central_argument: str
    """One-sentence thesis that the entire paper builds toward."""

    narrative_arc: str
    """How the argument develops from introduction through conclusion."""

    section_directives: list[SectionNarrativeDirective]
    """Ordered directives for each section in the outline."""
