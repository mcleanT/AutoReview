"""Prompts for the NarrativeArchitect pipeline stage."""

from __future__ import annotations

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.llm.prompts.outline import ReviewOutline

NARRATIVE_ARCHITECT_SYSTEM_PROMPT = """\
You are a scientific editor planning the narrative architecture of a review paper.
You do not write prose — you plan the story: what central argument the paper builds,
what rhetorical move each section makes, and how sections connect into a coherent arc.

Your output is a planning document for the section writers, not the paper itself.
Suggest structural approaches and narrative roles as soft hints — the writers should
deviate if the evidence warrants it.

Be analytical and specific. Instead of "discusses the evidence", write "establishes that
the mechanistic and epidemiological literatures have reached opposing conclusions, creating
the central tension the paper will resolve." Each section directive should give writers
a clear argumentative direction.
"""


def build_narrative_planning_prompt(
    outline: ReviewOutline,
    evidence_map: EvidenceMap,
    scope_document: str,
) -> str:
    """Build the narrative planning prompt for the NarrativeArchitect."""
    # Flatten outline to a readable list
    outline_lines = []
    for s in outline.flatten():
        depth = s.id.count(".") + 1
        indent = "  " * (depth - 1)
        outline_lines.append(f"{indent}{s.id}. {s.title}: {s.description}")
    outline_text = "\n".join(outline_lines)

    # Summarise evidence map
    themes_text = "\n".join(
        f"- {t.name}: {t.description} ({len(t.paper_ids)} papers)" for t in evidence_map.themes
    )

    consensus_text = (
        "\n".join(
            f"- '{cc.claim}' ({cc.evidence_count} papers)"
            for cc in evidence_map.consensus_claims[:10]
        )
        or "(none identified)"
    )

    contradictions_text = (
        "\n".join(
            f"- '{c.claim_a}' vs '{c.claim_b}' — {c.possible_explanation or 'explanation unclear'}"
            for c in evidence_map.contradictions[:10]
        )
        or "(none identified)"
    )

    gaps_text = (
        "\n".join(f"- {g.expected_topic} (severity: {g.severity})" for g in evidence_map.gaps[:10])
        or "(none identified)"
    )

    section_ids = ", ".join(s.id for s in outline.flatten())

    return f"""\
## Review Scope
{scope_document}

## Outline ({outline.title})
{outline_text}

## Evidence Summary

### Themes
{themes_text or "(no themes identified)"}

### Consensus Claims
{consensus_text}

### Contradictions
{contradictions_text}

### Identified Gaps
{gaps_text}

## Your Task

Plan the narrative architecture of this review paper. Produce:

1. **central_argument**: A single sentence that captures the paper's thesis — what the \
reader should conclude after reading it.

2. **narrative_arc**: 2–4 sentences describing how the argument builds across sections, \
from introduction through conclusion.

3. **section_directives**: One directive per section (use the exact section IDs: {section_ids}), \
each specifying:
   - **narrative_role**: What argumentative move this section makes in the paper's story
   - **central_claim**: The thesis or claim the section should establish
   - **structural_suggestion**: Which evidence-appropriate structure to use \
(comparative / problem-solution / chronological / argument-rebuttal / or other)
   - **key_insights**: 3–5 analytical insights the evidence supports (not paper IDs)
   - **transition_from_prev**: A hint for how to open, connecting from the previous section
   - **transition_to_next**: A hint for how to close, setting up the next section

Focus on analytical depth. Avoid generic directives like "summarises the literature."
"""
