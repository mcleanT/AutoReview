from __future__ import annotations

from autoreview.critique.rubrics import (
    HOLISTIC_RUBRICS,
    SECTION_RUBRICS,
    format_compact_rubrics,
    format_rubrics_for_prompt,
)

_SECTION_RUBRIC_BLOCK = format_rubrics_for_prompt(SECTION_RUBRICS)
_HOLISTIC_RUBRIC_BLOCK = format_rubrics_for_prompt(HOLISTIC_RUBRICS)

SECTION_CRITIQUE_SYSTEM_PROMPT = f"""\
You are an expert scientific editor evaluating a section of a review paper. \
Score each dimension using the rubric anchors below. Use the anchor descriptions \
to calibrate your scores — a score should reflect the level whose description best \
matches the text, not a vague impression.

## Scoring Rubric

{_SECTION_RUBRIC_BLOCK}

Flag any critical issues (paper-by-paper summary, missing citations, logical gaps, \
formulaic structure, or paragraphs that are lists of findings dressed as prose).
"""

HOLISTIC_CRITIQUE_SYSTEM_PROMPT = f"""\
You are an expert scientific editor evaluating a complete review paper draft. \
Score each dimension using the rubric anchors below. Use the anchor descriptions \
to calibrate your scores — a score should reflect the level whose description best \
matches the text, not a vague impression.

## Scoring Rubric

{_HOLISTIC_RUBRIC_BLOCK}

Identify redundancy pairs (sections with overlapping content) and provide specific fixes.
"""


_SECTION_CRITIQUE_COMPACT_TEMPLATE = """\
You are an expert scientific editor evaluating a section of a review paper. \
Score each dimension. Focus on the weakest areas from the previous cycle.

## Scoring Rubric (compact — focus on weak dimensions)

{rubric_block}

Flag any critical issues (paper-by-paper summary, missing citations, logical gaps, \
formulaic structure, or paragraphs that are lists of findings dressed as prose).
"""

_HOLISTIC_CRITIQUE_COMPACT_TEMPLATE = """\
You are an expert scientific editor evaluating a complete review paper draft. \
Score each dimension. Focus on the weakest areas from the previous cycle.

## Scoring Rubric (compact — focus on weak dimensions)

{rubric_block}

Identify redundancy pairs (sections with overlapping content) and provide specific fixes.
"""


def get_section_critique_system_prompt(
    previous_scores: dict[str, float] | None = None,
) -> str:
    """Return full rubrics on first call, compact rubrics on subsequent calls."""
    if not previous_scores:
        return SECTION_CRITIQUE_SYSTEM_PROMPT
    compact = format_compact_rubrics(previous_scores, SECTION_RUBRICS)
    return _SECTION_CRITIQUE_COMPACT_TEMPLATE.format(rubric_block=compact)


def get_holistic_critique_system_prompt(
    previous_scores: dict[str, float] | None = None,
) -> str:
    """Return full rubrics on first call, compact rubrics on subsequent calls."""
    if not previous_scores:
        return HOLISTIC_CRITIQUE_SYSTEM_PROMPT
    compact = format_compact_rubrics(previous_scores, HOLISTIC_RUBRICS)
    return _HOLISTIC_CRITIQUE_COMPACT_TEMPLATE.format(rubric_block=compact)


def build_section_critique_prompt(
    section_id: str,
    section_title: str,
    section_text: str,
    outline_context: str,
    adjacent_context: str = "",
) -> str:
    return f"""\
## Section Being Critiqued
**ID:** {section_id}
**Title:** {section_title}

## Section Text
{section_text}

## Outline Context
{outline_context}

## Adjacent Sections
{adjacent_context}

Evaluate this section on all dimensions. Provide specific issues with severity and suggestions.
"""


def build_holistic_critique_prompt(full_draft: str, scope_document: str) -> str:
    return f"""\
## Review Scope
{scope_document}

## Full Draft
{full_draft}

Evaluate this complete draft on all dimensions. Identify redundancy pairs, transition issues, \
and any misalignment between introduction and conclusion.
"""


def build_revision_prompt(
    text: str,
    issues_text: str,
    context: str = "",
    dimension_feedback: str = "",
) -> str:
    dimension_block = ""
    if dimension_feedback:
        dimension_block = (
            f"\n## Dimension Scores & Focus Areas\n{dimension_feedback}\n\n"
            "Focus especially on the weakest dimensions identified above.\n"
        )
    return f"""\
## Current Text
{text}

## Issues to Address
{issues_text}
{dimension_block}
## Context
{context}

Revise the text to address all listed issues. Maintain the same structure and citation markers. \
Focus on synthesis quality and narrative coherence.
"""
