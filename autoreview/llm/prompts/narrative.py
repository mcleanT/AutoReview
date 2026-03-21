"""Prompts for the NarrativeArchitect pipeline stage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.llm.prompts.outline import ReviewOutline

if TYPE_CHECKING:
    from autoreview.config.models import DepthLevel

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
    depth: DepthLevel | None = None,
    figures: dict[str, Any] | None = None,
    tables: dict[str, Any] | None = None,
) -> str:
    """Build the narrative planning prompt for the NarrativeArchitect."""
    from autoreview.config.depth import get_depth_profile

    if depth is not None:
        lo, hi = get_depth_profile(depth).key_insights_range
    else:
        lo, hi = 3, 5
    insights_range = f"{lo}\u2013{hi}"

    # Flatten outline to a readable list
    outline_lines = []
    for s in outline.flatten():
        section_depth = s.id.count(".") + 1
        indent = "  " * (section_depth - 1)
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

    # Quality directives for ARISE scoring
    quality_directives = """
## Quality Directives

In addition to the section-by-section narrative plan, include these directives:

1. **Productive Tensions Framework**: Frame the contested contradictions as the "Productive Tensions Framework" — a named analytical contribution. Each contradiction should dissolve into a context-dependent design choice when stratified appropriately.

2. **Methodology expansion**: For the Methods section, direct the writer to describe: study design classification (computational, systematic review, cohort, etc.), quality scoring methodology (0.0-1.0 scale), sample size weighting, and how contradictions were identified through structured resolution.

3. **Prior survey comparison**: For the Introduction, direct the writer to compare this review to prior surveys of the topic, noting differentiators in scale, temporal coverage, and analytical approach.

4. **Limitations**: For the Limitations section, direct the writer to cover: percentage of abstract-only papers, language/venue bias, preprint dominance, absence of formal meta-analysis, and search cutoff exclusions.

5. **Ethics and societal considerations**: For the Ethics section, direct the writer to cover: retrieval corpus bias, intellectual property concerns, privacy-preserving gaps, environmental cost of retrieval infrastructure, and epistemic authority shifts.
"""

    visual_refs = ""
    if figures or tables:
        lines = ["6. **Visual references**: Direct section writers to reference these visuals:"]
        if figures:
            for fig in figures.values():
                lines.append(
                    f"   - {fig.caption} (insert reference in section {fig.anchor.section_id})"
                )
        if tables:
            for tbl in tables.values():
                lines.append(
                    f"   - {tbl.caption} (insert reference near section {tbl.anchor.section_id})"
                )
        visual_refs = "\n".join(lines)
    quality_directives += visual_refs

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

{quality_directives}

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
   - **key_insights**: {insights_range} analytical insights the evidence supports (not paper IDs)
   - **transition_from_prev**: A hint for how to open, connecting from the previous section
   - **transition_to_next**: A hint for how to close, setting up the next section

Focus on analytical depth. Avoid generic directives like "summarises the literature."
"""
