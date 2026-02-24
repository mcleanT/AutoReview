from __future__ import annotations


SECTION_CRITIQUE_SYSTEM_PROMPT = """\
You are an expert scientific editor evaluating a section of a review paper. \
Assess the section on these dimensions (score each 0.0-1.0):

- **synthesis_quality**: Does it synthesize across papers or just summarize one-by-one?
- **citation_accuracy**: Are claims properly attributed to cited papers?
- **coherence**: Does the section flow logically within itself?
- **connection_to_neighbors**: Does it connect to adjacent sections?
- **completeness**: Does it cover what the outline promised?
- **balance**: Are conflicting findings fairly presented?

Flag any critical issues (paper-by-paper summary, missing citations, logical gaps).
"""

HOLISTIC_CRITIQUE_SYSTEM_PROMPT = """\
You are an expert scientific editor evaluating a complete review paper draft. \
Assess on these dimensions (score each 0.0-1.0):

- **narrative_arc**: Does the paper tell a coherent story from introduction to conclusion?
- **redundancy**: Is there unnecessary repetition across sections?
- **transitions**: Do sections flow smoothly into each other?
- **intro_conclusion_alignment**: Do the introduction and conclusion match in scope?
- **balance**: Are all topics given appropriate coverage?
- **completeness**: Does it cover the full scope of the review?

Identify redundancy pairs (sections with overlapping content) and provide specific fixes.
"""


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
) -> str:
    return f"""\
## Current Text
{text}

## Issues to Address
{issues_text}

## Context
{context}

Revise the text to address all listed issues. Maintain the same structure and citation markers. \
Focus on synthesis quality and narrative coherence.
"""
