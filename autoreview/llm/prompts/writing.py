from __future__ import annotations


SECTION_WRITING_SYSTEM_PROMPT = """\
You are an expert scientific writer drafting a section of a review paper. \
Your writing must SYNTHESIZE findings across papers — do NOT summarize papers one by one. \
Instead, organize by themes, compare results, trace patterns, weigh contradictions, and \
build a coherent narrative.

Use [@paper_id] markers for inline citations. Each claim must be attributed.

## Structural Approach
Choose the structure that best serves this section's evidence — do not default to \
background → findings → limitations. Viable structures:
- Comparative: contrast two competing accounts or methodological traditions
- Problem-solution: establish what was unknown, then trace how the field addressed it
- Chronological development: trace how understanding of a concept evolved
- Argument-rebuttal: state the dominant view, then present the evidence that challenges it
Let the evidence shape the section architecture, not a template.

## Paragraph Discipline
Each paragraph advances exactly one claim. The opening sentence states the claim; \
the rest develops it with evidence, qualification, or contrast. Avoid paragraphs \
that are lists of findings dressed as prose.

## Transitions
Begin with a transition from the preceding section. \
End with a sentence that creates forward momentum into the following section.
"""


def build_section_writing_prompt(
    section_id: str,
    section_title: str,
    section_description: str,
    outline_context: str,
    relevant_extractions: str,
    synthesis_directives: str = "",
    adjacent_text: str = "",
    narrative_guidance: str = "",
) -> str:
    narrative_block = f"\n{narrative_guidance}\n" if narrative_guidance else ""
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
{narrative_block}\
Write this section with proper synthesis, citation markers [@paper_id], and smooth transitions.
"""
