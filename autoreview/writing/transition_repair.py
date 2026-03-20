"""Transition repair utilities for post-assembly section boundaries.

After sections are assembled, this module identifies and repairs abrupt
transitions between consecutive sections by generating bridge sentences
that create intellectual momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class SectionBoundary:
    """Captures the textual boundary between two consecutive sections."""

    section_a_id: str
    section_b_id: str
    ending_text: str
    opening_text: str


def extract_boundary(
    section_a_text: str,
    section_b_text: str,
    context_chars: int = 500,
    section_a_id: str = "",
    section_b_id: str = "",
) -> SectionBoundary:
    """Extract the boundary context between two adjacent sections.

    For long sections, takes the last ``context_chars`` characters of
    section A and the first ``context_chars`` characters of section B.
    Short sections (shorter than ``context_chars``) are used in full.

    Args:
        section_a_text: Full text of the preceding section.
        section_b_text: Full text of the following section.
        context_chars: Number of characters to take from each side.
        section_a_id: Optional identifier for section A.
        section_b_id: Optional identifier for section B.

    Returns:
        A SectionBoundary capturing the ending and opening context.
    """
    ending_text = (
        section_a_text[-context_chars:] if len(section_a_text) > context_chars else section_a_text
    )
    opening_text = (
        section_b_text[:context_chars] if len(section_b_text) > context_chars else section_b_text
    )

    logger.debug(
        "transition_repair.extract_boundary",
        section_a_id=section_a_id,
        section_b_id=section_b_id,
        ending_chars=len(ending_text),
        opening_chars=len(opening_text),
    )

    return SectionBoundary(
        section_a_id=section_a_id,
        section_b_id=section_b_id,
        ending_text=ending_text,
        opening_text=opening_text,
    )


def build_transition_repair_prompt(boundary: SectionBoundary) -> str:
    """Build an LLM prompt requesting a bridge sentence for a section boundary.

    Args:
        boundary: The SectionBoundary containing ending and opening context.

    Returns:
        A formatted prompt string to send to an LLM.
    """
    prompt = (
        "You are editing a scientific review paper. "
        "Two consecutive sections have an abrupt transition. "
        "Your task is to write 1–3 sentences that bridge the end of the first "
        "section to the beginning of the second, creating intellectual momentum "
        "and a sense of logical flow for the reader.\n\n"
        "--- END OF SECTION"
        + (f" {boundary.section_a_id}" if boundary.section_a_id else "")
        + " ---\n"
        f"{boundary.ending_text}\n\n"
        "--- BEGINNING OF SECTION"
        + (f" {boundary.section_b_id}" if boundary.section_b_id else "")
        + " ---\n"
        f"{boundary.opening_text}\n\n"
        "Write ONLY the bridge sentences (1–3 sentences). "
        "Do not repeat content from either section. "
        "Do not include headings or section labels."
    )

    logger.debug(
        "transition_repair.build_prompt",
        section_a_id=boundary.section_a_id,
        section_b_id=boundary.section_b_id,
        prompt_chars=len(prompt),
    )

    return prompt
