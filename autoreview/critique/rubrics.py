"""Rubric-based scoring anchors for critique dimensions.

Each dimension has 5 anchor levels (0.0-0.2 through 0.9-1.0) that provide
concrete behavioral descriptions. These anchors are embedded into critique
system prompts so the LLM can score against specific criteria rather than
vague numeric ranges.
"""

from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class RubricAnchor(AutoReviewModel):
    """A single scoring anchor within a dimension rubric."""

    range_label: str  # e.g. "0.0-0.2"
    low: float
    high: float
    description: str


class DimensionRubric(AutoReviewModel):
    """Complete rubric for one scoring dimension."""

    dimension: str
    anchors: list[RubricAnchor] = Field(min_length=5, max_length=5)


def _rubric(dimension: str, anchors: list[tuple[str, float, float, str]]) -> DimensionRubric:
    """Helper to build a DimensionRubric from compact tuples."""
    return DimensionRubric(
        dimension=dimension,
        anchors=[
            RubricAnchor(range_label=label, low=lo, high=hi, description=desc)
            for label, lo, hi, desc in anchors
        ],
    )


# ---------------------------------------------------------------------------
# Section-level rubrics (8 dimensions)
# ---------------------------------------------------------------------------

SECTION_RUBRICS: list[DimensionRubric] = [
    _rubric("synthesis_quality", [
        ("0.0-0.2", 0.0, 0.2, "Lists papers individually ('Study A found X. Study B found Y.') with no connecting threads."),
        ("0.3-0.4", 0.3, 0.4, "Groups findings by topic but merely juxtaposes them without comparison or analysis."),
        ("0.5-0.6", 0.5, 0.6, "Identifies patterns and makes some comparisons, but remains descriptive rather than analytical."),
        ("0.7-0.8", 0.7, 0.8, "Builds narrative chains across papers, weights evidence by strength, and resolves contradictions with explanations."),
        ("0.9-1.0", 0.9, 1.0, "Novel insights emerge from integration; reader gains understanding impossible from individual papers alone."),
    ]),
    _rubric("citation_accuracy", [
        ("0.0-0.2", 0.0, 0.2, "Major claims are uncited or attributed to wrong papers; phantom citations present."),
        ("0.3-0.4", 0.3, 0.4, "Most claims cited but several misattributions or missing citations for key assertions."),
        ("0.5-0.6", 0.5, 0.6, "Citations generally correct but some claims lack attribution or cite tangentially relevant papers."),
        ("0.7-0.8", 0.7, 0.8, "All substantive claims properly attributed; citations match the actual findings of referenced papers."),
        ("0.9-1.0", 0.9, 1.0, "Precise attribution throughout; citations contextualize exactly which finding from each paper is being referenced."),
    ]),
    _rubric("coherence", [
        ("0.0-0.2", 0.0, 0.2, "Paragraphs are disjointed; no logical thread connects successive ideas."),
        ("0.3-0.4", 0.3, 0.4, "Some logical grouping but frequent non-sequiturs or abrupt topic shifts within the section."),
        ("0.5-0.6", 0.5, 0.6, "Generally follows a logical order but transitions are weak or mechanical ('Furthermore...', 'Additionally...')."),
        ("0.7-0.8", 0.7, 0.8, "Clear logical progression with substantive transitions that explain why one topic leads to the next."),
        ("0.9-1.0", 0.9, 1.0, "Each paragraph flows inevitably into the next; the section reads as a sustained argument, not a collection of points."),
    ]),
    _rubric("connection_to_neighbors", [
        ("0.0-0.2", 0.0, 0.2, "No reference to adjacent sections; section could be read entirely in isolation."),
        ("0.3-0.4", 0.3, 0.4, "Perfunctory opening/closing sentences mention adjacent sections without substantive connection."),
        ("0.5-0.6", 0.5, 0.6, "Opening transition from previous section present but closing connection to next section is weak or absent."),
        ("0.7-0.8", 0.7, 0.8, "Both opening and closing transitions are substantive, connecting specific ideas across section boundaries."),
        ("0.9-1.0", 0.9, 1.0, "Section explicitly builds on concepts from preceding section and sets up ideas that the following section develops."),
    ]),
    _rubric("completeness", [
        ("0.0-0.2", 0.0, 0.2, "Major topics from the outline are missing; section is a stub."),
        ("0.3-0.4", 0.3, 0.4, "Covers some outline topics but skips others; significant gaps in promised coverage."),
        ("0.5-0.6", 0.5, 0.6, "All major outline topics mentioned but some treated superficially."),
        ("0.7-0.8", 0.7, 0.8, "All outline topics covered with adequate depth; sub-topics addressed proportionally."),
        ("0.9-1.0", 0.9, 1.0, "Comprehensive coverage exceeding outline expectations; identifies and addresses sub-topics the outline did not anticipate."),
    ]),
    _rubric("balance", [
        ("0.0-0.2", 0.0, 0.2, "One-sided presentation; conflicting evidence ignored or dismissed without justification."),
        ("0.3-0.4", 0.3, 0.4, "Conflicting findings mentioned but given disproportionate weight favoring one side."),
        ("0.5-0.6", 0.5, 0.6, "Both sides of disagreements presented but without analysis of why results differ."),
        ("0.7-0.8", 0.7, 0.8, "Conflicting findings presented fairly with analysis of methodological or contextual reasons for disagreement."),
        ("0.9-1.0", 0.9, 1.0, "Nuanced treatment where contradictions become a source of insight; reader understands the state of debate and its implications."),
    ]),
    _rubric("structural_variety", [
        ("0.0-0.2", 0.0, 0.2, "Rigid template: background then findings then limitations, regardless of content."),
        ("0.3-0.4", 0.3, 0.4, "Mostly template-driven with occasional deviation; structure does not serve the evidence."),
        ("0.5-0.6", 0.5, 0.6, "Some structural adaptation to content but defaults to familiar patterns when evidence would benefit from a different approach."),
        ("0.7-0.8", 0.7, 0.8, "Structure clearly chosen to serve the evidence (comparative, chronological, argument-rebuttal, etc.) and is internally consistent."),
        ("0.9-1.0", 0.9, 1.0, "Structure is invisible — the reader follows the argument without noticing the organizational scaffold; evidence dictates form."),
    ]),
    _rubric("paragraph_cohesion", [
        ("0.0-0.2", 0.0, 0.2, "Paragraphs are lists of findings; no opening claim, no unifying idea."),
        ("0.3-0.4", 0.3, 0.4, "Some paragraphs have topic sentences but many drift or pack multiple unrelated claims."),
        ("0.5-0.6", 0.5, 0.6, "Most paragraphs open with a claim but supporting evidence is loosely organized within them."),
        ("0.7-0.8", 0.7, 0.8, "Each paragraph opens with a clear claim and the body develops it with evidence, qualification, or contrast."),
        ("0.9-1.0", 0.9, 1.0, "Every paragraph is a tight unit: claim, evidence, and implication, with no wasted sentences."),
    ]),
]

# ---------------------------------------------------------------------------
# Holistic-level rubrics (7 dimensions)
# ---------------------------------------------------------------------------

HOLISTIC_RUBRICS: list[DimensionRubric] = [
    _rubric("narrative_arc", [
        ("0.0-0.2", 0.0, 0.2, "No discernible narrative; sections could be reordered without loss of meaning."),
        ("0.3-0.4", 0.3, 0.4, "Sections follow a generic order (intro-body-conclusion) but no thread connects them."),
        ("0.5-0.6", 0.5, 0.6, "A narrative thread is identifiable but weakens in the middle sections."),
        ("0.7-0.8", 0.7, 0.8, "Clear narrative arc: introduction poses questions, body sections build toward answers, conclusion synthesizes."),
        ("0.9-1.0", 0.9, 1.0, "Compelling intellectual journey; the reader's understanding evolves progressively from introduction to conclusion."),
    ]),
    _rubric("redundancy", [
        ("0.0-0.2", 0.0, 0.2, "Multiple sections repeat the same claims with the same evidence; extensive duplication."),
        ("0.3-0.4", 0.3, 0.4, "Some sections overlap significantly; the same studies discussed in substantially the same way."),
        ("0.5-0.6", 0.5, 0.6, "Minor overlap between sections but each adds some new perspective; a few redundant sentences."),
        ("0.7-0.8", 0.7, 0.8, "No redundancy; when a study appears in multiple sections, it serves a different analytical purpose each time."),
        ("0.9-1.0", 0.9, 1.0, "Zero redundancy with masterful cross-referencing; each mention of shared evidence adds new insight."),
    ]),
    _rubric("transitions", [
        ("0.0-0.2", 0.0, 0.2, "No transitions between sections; each section begins as if starting a new document."),
        ("0.3-0.4", 0.3, 0.4, "Generic transitions ('In this section...', 'Next we discuss...') without substantive connection."),
        ("0.5-0.6", 0.5, 0.6, "Transitions reference what came before but do not explain why the reader should care about what follows."),
        ("0.7-0.8", 0.7, 0.8, "Transitions connect specific ideas from previous section to the focus of the next, creating logical bridges."),
        ("0.9-1.0", 0.9, 1.0, "Seamless flow where the end of one section creates intellectual momentum that propels the reader into the next."),
    ]),
    _rubric("intro_conclusion_alignment", [
        ("0.0-0.2", 0.0, 0.2, "Introduction and conclusion address different scopes or make contradictory claims."),
        ("0.3-0.4", 0.3, 0.4, "General alignment but conclusion introduces claims not set up in the introduction."),
        ("0.5-0.6", 0.5, 0.6, "Introduction and conclusion share scope but conclusion does not fully address questions posed in introduction."),
        ("0.7-0.8", 0.7, 0.8, "Conclusion directly answers questions posed in introduction; scope is consistent throughout."),
        ("0.9-1.0", 0.9, 1.0, "Conclusion transforms the introduction's questions into deeper understanding; the reader re-reads the introduction with new eyes."),
    ]),
    _rubric("balance", [
        ("0.0-0.2", 0.0, 0.2, "Some topics get disproportionate coverage; key areas of the scope are barely mentioned."),
        ("0.3-0.4", 0.3, 0.4, "Uneven coverage with one or two sections dominating the paper."),
        ("0.5-0.6", 0.5, 0.6, "Reasonable coverage distribution but minor topics get either too much or too little space."),
        ("0.7-0.8", 0.7, 0.8, "Coverage proportional to importance and evidence availability; no section feels rushed or bloated."),
        ("0.9-1.0", 0.9, 1.0, "Perfect balance where depth of coverage reflects both the maturity and importance of each sub-topic."),
    ]),
    _rubric("completeness", [
        ("0.0-0.2", 0.0, 0.2, "Major areas from the scope document are missing entirely."),
        ("0.3-0.4", 0.3, 0.4, "Most scope areas covered but significant gaps in one or two critical areas."),
        ("0.5-0.6", 0.5, 0.6, "All scope areas mentioned but some treated too superficially to be useful."),
        ("0.7-0.8", 0.7, 0.8, "Full scope coverage with adequate depth; future directions and gaps clearly identified."),
        ("0.9-1.0", 0.9, 1.0, "Exhaustive coverage that exceeds scope; identifies emerging areas and anticipates future developments."),
    ]),
    _rubric("prose_flow", [
        ("0.0-0.2", 0.0, 0.2, "Choppy sentences; abrupt shifts between ideas; reads like a collection of notes."),
        ("0.3-0.4", 0.3, 0.4, "Some sentences flow but frequent awkward transitions and weak connective tissue between paragraphs."),
        ("0.5-0.6", 0.5, 0.6, "Generally readable but occasional jarring transitions or mechanical phrasing."),
        ("0.7-0.8", 0.7, 0.8, "Smooth prose within and between paragraphs; varied sentence structure; natural reading rhythm."),
        ("0.9-1.0", 0.9, 1.0, "Elegant academic prose; the writing itself enhances comprehension and engagement."),
    ]),
]

# Dimension name sets for validation
SECTION_DIMENSION_NAMES = frozenset(r.dimension for r in SECTION_RUBRICS)
HOLISTIC_DIMENSION_NAMES = frozenset(r.dimension for r in HOLISTIC_RUBRICS)


def format_rubrics_for_prompt(rubrics: list[DimensionRubric]) -> str:
    """Format rubrics into a block suitable for embedding in critique system prompts.

    Returns a markdown-formatted string with each dimension and its anchor descriptions.
    """
    blocks = []
    for rubric in rubrics:
        lines = [f"### {rubric.dimension}"]
        for anchor in rubric.anchors:
            lines.append(f"- **{anchor.range_label}**: {anchor.description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_compact_rubrics(
    dimension_scores: dict[str, float],
    rubrics: list[DimensionRubric],
    bottom_n: int = 4,
) -> str:
    """Format a compact rubric block with only weak dimensions and their target anchors.

    Used on repeat critique calls to reduce token usage while still providing
    scoring guidance for the dimensions that need improvement.
    """
    if not dimension_scores:
        return format_rubrics_for_prompt(rubrics)

    rubric_map = {r.dimension: r for r in rubrics}
    # Sort weakest first
    sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1])
    weak_dims = sorted_dims[:bottom_n]

    blocks = []
    # Include all dimension names for scoring, but only detail weak ones
    all_dims = [r.dimension for r in rubrics]
    blocks.append("Score ALL dimensions: " + ", ".join(all_dims) + "\n")
    blocks.append("**Focus areas (weakest dimensions — detailed anchors below):**\n")

    for dim, score in weak_dims:
        rubric = rubric_map.get(dim)
        if not rubric:
            continue
        # Find current and target anchor
        target_anchor = None
        for i, anchor in enumerate(rubric.anchors):
            if anchor.low <= score <= anchor.high:
                if i + 1 < len(rubric.anchors):
                    target_anchor = rubric.anchors[i + 1]
                break
        if target_anchor:
            blocks.append(
                f"- **{dim}** (current: {score:.2f}): "
                f"Target {target_anchor.range_label} — {target_anchor.description}"
            )

    return "\n".join(blocks)


def format_dimension_feedback(
    dimension_scores: dict[str, float],
    rubrics: list[DimensionRubric],
) -> str:
    """Identify the weakest dimensions and provide targeted guidance for revision.

    Returns a formatted block listing each dimension's score and anchor-based
    feedback, sorted from weakest to strongest so the writer focuses on the
    most impactful improvements first.
    """
    if not dimension_scores:
        return ""

    rubric_map = {r.dimension: r for r in rubrics}
    scored = sorted(dimension_scores.items(), key=lambda x: x[1])

    lines = []
    for dim, score in scored:
        rubric = rubric_map.get(dim)
        if not rubric:
            lines.append(f"- **{dim}** ({score:.2f}): (no rubric available)")
            continue

        # Find the anchor level the score falls into
        current_anchor = None
        next_anchor = None
        for i, anchor in enumerate(rubric.anchors):
            if anchor.low <= score <= anchor.high:
                current_anchor = anchor
                if i + 1 < len(rubric.anchors):
                    next_anchor = rubric.anchors[i + 1]
                break

        if current_anchor and next_anchor:
            lines.append(
                f"- **{dim}** ({score:.2f}): Currently at {current_anchor.range_label} level — "
                f"{current_anchor.description} "
                f"To improve, aim for {next_anchor.range_label}: {next_anchor.description}"
            )
        elif current_anchor:
            lines.append(
                f"- **{dim}** ({score:.2f}): At {current_anchor.range_label} level — "
                f"{current_anchor.description}"
            )
        else:
            lines.append(f"- **{dim}** ({score:.2f})")

    weakest_dim = scored[0][0] if scored else None
    header = ""
    if weakest_dim:
        header = f"**Weakest dimension: {weakest_dim}** — prioritize this for the biggest quality gain.\n\n"

    return header + "\n".join(lines)
