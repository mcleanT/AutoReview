from __future__ import annotations


SECTION_WRITING_SYSTEM_PROMPT = """\
You are an expert scientific writer drafting a section of a review paper. \
Your writing must SYNTHESIZE findings across papers — do NOT summarize papers one by one. \
Instead, organize by themes, compare results, trace patterns, weigh contradictions, and \
build a coherent narrative.

Use [@paper_id] markers for inline citations. Each claim must be attributed.
"""


def build_section_writing_prompt(
    section_id: str,
    section_title: str,
    section_description: str,
    outline_context: str,
    relevant_extractions: str,
    synthesis_directives: str = "",
    adjacent_text: str = "",
) -> str:
    return f"""\
## Section to Write
**ID:** {section_id}
**Title:** {section_title}
**Description:** {section_description}

## Full Outline Context
{outline_context}

## Relevant Paper Extractions
{relevant_extractions}

## Synthesis Directives
{synthesis_directives or "Synthesize across papers. Do not summarize individually."}

## Adjacent Section Text
{adjacent_text or "(First section or adjacent sections not yet written)"}

Write this section with proper synthesis, citation markers [@paper_id], and smooth transitions.
"""
