from __future__ import annotations

import re
from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.extraction.models import PaperExtraction
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.prompts.writing import (
    SECTION_WRITING_SYSTEM_PROMPT,
    build_section_writing_prompt,
)
from autoreview.models.base import AutoReviewModel
from autoreview.models.enrichment import SectionEnrichment
from autoreview.models.narrative import NarrativePlan, SectionNarrativeDirective

logger = structlog.get_logger()


class SectionDraft(AutoReviewModel):
    """A drafted section with metadata."""

    section_id: str
    title: str
    text: str
    citations_used: list[str] = []


def _format_outline_context(
    outline: ReviewOutline,
    current_section_id: str | None = None,
) -> str:
    """Format outline with focused context.

    Full detail for current+neighbors, title-only for rest.
    """
    flat = outline.flatten()

    # Find current section index for neighbor detection
    current_idx: int | None = None
    if current_section_id:
        for i, s in enumerate(flat):
            if s.id == current_section_id:
                current_idx = i
                break

    # Neighbor indices (current, prev, next)
    neighbor_ids: set[str] = set()
    if current_idx is not None:
        neighbor_ids.add(flat[current_idx].id)
        if current_idx > 0:
            neighbor_ids.add(flat[current_idx - 1].id)
        if current_idx < len(flat) - 1:
            neighbor_ids.add(flat[current_idx + 1].id)

    lines = []
    for s in flat:
        depth = s.id.count(".") + 1
        indent = "  " * (depth - 1)
        if s.id in neighbor_ids:
            lines.append(f"{indent}{s.id}. {s.title}: {s.description}")
        else:
            lines.append(f"{indent}{s.id}. {s.title}")
    return "\n".join(lines)


def _format_extractions(
    paper_ids: list[str],
    extractions: dict[str, PaperExtraction],
) -> str:
    """Format relevant paper extractions for the section writer."""
    blocks = []
    for pid in paper_ids:
        ext = extractions.get(pid)
        if not ext:
            continue
        findings = "; ".join(f.claim for f in ext.key_findings)
        blocks.append(
            f"[@{pid}]\n"
            f"  Findings: {findings}\n"
            f"  Methods: {ext.methods_summary}\n"
            f"  Limitations: {ext.limitations}"
        )
    return "\n\n".join(blocks) if blocks else "(No extractions available)"


def _generate_synthesis_directives(
    section: OutlineSection,
    evidence_map: EvidenceMap,
    extractions: dict[str, PaperExtraction] | None = None,
) -> str:
    """Generate enriched synthesis directives from the evidence map for this section.

    When extractions are available, computes evidence strength distribution and
    includes evidence chains, enriched contradictions, and temporal progressions
    from the evidence map. Falls back to basic directives when data is unavailable.
    """
    from autoreview.analysis.evidence_chains import (
        EnrichedContradiction,
        EvidenceChain,
        EvidenceChainBuilder,
        TemporalProgression,
    )

    directives = []

    # Evidence strength profile
    if extractions and section.paper_ids:
        builder = EvidenceChainBuilder(llm=None)
        strength_dist = builder.compute_strength_distribution(section.paper_ids, extractions)
        if strength_dist.writer_guidance:
            directives.append(
                f"**Evidence Profile:** {strength_dist.dominant_type} dominant "
                f"({strength_dist.strong}S/{strength_dist.moderate}M/"
                f"{strength_dist.weak}W/{strength_dist.preliminary}P). "
                f"{strength_dist.writer_guidance}"
            )

    section_set = set(section.paper_ids)

    # Evidence chains from evidence map
    if evidence_map.evidence_chains:
        chain_descs = []
        for raw_chain in evidence_map.evidence_chains:
            try:
                chain = (
                    EvidenceChain.model_validate(raw_chain)
                    if isinstance(raw_chain, dict)
                    else raw_chain
                )
                if set(chain.paper_ids) & section_set:
                    pids = ", ".join(f"[@{pid}]" for pid in chain.paper_ids)
                    chain_descs.append(f"- [{chain.chain_type}] {chain.description} ({pids})")
            except Exception:
                continue
        if chain_descs:
            directives.append(
                "**Evidence Chains — trace these in your prose:**\n" + "\n".join(chain_descs)
            )

    # Enriched contradictions
    if evidence_map.enriched_contradictions:
        contra_descs = []
        for raw_ec in evidence_map.enriched_contradictions:
            try:
                ec = (
                    EnrichedContradiction.model_validate(raw_ec)
                    if isinstance(raw_ec, dict)
                    else raw_ec
                )
                if (set(ec.paper_ids_a) | set(ec.paper_ids_b)) & section_set:
                    framing = f" Framing: {ec.framing_strategy}" if ec.framing_strategy else ""
                    contra_descs.append(f"- '{ec.claim_a}' vs '{ec.claim_b}'.{framing}")
            except Exception:
                continue
        if contra_descs:
            directives.append("**Contradictions to address:**\n" + "\n".join(contra_descs))
    else:
        # Fall back to basic contradictions
        for c in evidence_map.contradictions:
            if c.theme and any(ref in (c.theme or "") for ref in section.theme_refs):
                directives.append(
                    f"- Address contradiction: '{c.claim_a}' vs '{c.claim_b}' "
                    f"(possible explanation: {c.possible_explanation or 'unclear'})"
                )

    # Temporal progressions
    if evidence_map.temporal_progressions:
        for raw_tp in evidence_map.temporal_progressions:
            try:
                tp = (
                    TemporalProgression.model_validate(raw_tp)
                    if isinstance(raw_tp, dict)
                    else raw_tp
                )
                if set(tp.paper_ids) & section_set:
                    directives.append(
                        f"**Temporal Progression:** {tp.early_period} "
                        f"focused on: {tp.early_focus}. "
                        f"{tp.late_period} shifted to: {tp.late_focus}. "
                        f"Consider using chronological structure to trace this evolution."
                    )
            except Exception:
                continue

    # Consensus claims
    for cc in evidence_map.consensus_claims:
        overlap = set(cc.supporting_paper_ids) & section_set
        if overlap:
            directives.append(
                f"- Highlight consensus: '{cc.claim}' "
                f"({cc.evidence_count} papers, {cc.strength} evidence)"
            )

    return "\n\n".join(directives) if directives else ""


def _extract_citations(text: str) -> list[str]:
    """Extract [@paper_id] citation markers from text."""
    return re.findall(r"\[@([^\]]+)\]", text)


def _format_contextual_enrichment(enrichment: SectionEnrichment) -> str:
    """Format contextual enrichment data into a prompt-ready block."""
    if not enrichment.contextual_extractions:
        return ""

    blocks = []
    for ext in enrichment.contextual_extractions:
        concepts = ", ".join(ext.key_concepts) if ext.key_concepts else "(none)"
        connections = (
            "; ".join(ext.cross_field_connections) if ext.cross_field_connections else "(none)"
        )
        blocks.append(
            f"### {ext.paper_title}\n"
            f"**Background:** {ext.background_summary}\n"
            f"**Key concepts:** {concepts}\n"
            f"**Methodological context:** {ext.methodological_context or '(none)'}\n"
            f"**Cross-field connections:** {connections}\n"
            f"**Relevance:** {ext.relevance_to_section or '(not specified)'}"
        )
    return "\n\n".join(blocks)


class SectionWriter:
    """Writes review paper sections with cross-section context."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def write_section(
        self,
        section: OutlineSection,
        outline: ReviewOutline,
        extractions: dict[str, PaperExtraction],
        evidence_map: EvidenceMap,
        preceding_text: str = "",
        following_text: str = "",
        directive: SectionNarrativeDirective | None = None,
        enrichment: SectionEnrichment | None = None,
    ) -> SectionDraft:
        """Write a single section with full context."""
        outline_context = _format_outline_context(outline, current_section_id=section.id)
        relevant = _format_extractions(section.paper_ids, extractions)
        directives = _generate_synthesis_directives(section, evidence_map, extractions)

        adjacent = ""
        if preceding_text:
            # Only include last 1000 chars (closing paragraph) for transition context
            truncated_preceding = (
                preceding_text[-1000:] if len(preceding_text) > 1000 else preceding_text
            )
            adjacent += f"### Preceding Section (ending)\n{truncated_preceding}\n\n"
        if following_text:
            adjacent += f"### Following Section\n{following_text[:500]}"

        narrative_guidance = ""
        if directive:
            key_insights_text = "\n".join(f"  - {insight}" for insight in directive.key_insights)
            narrative_guidance = (
                f"## Narrative Guidance (soft hints — deviate if evidence warrants)\n"
                f"**Role in paper**: {directive.narrative_role}\n"
                f"**Central claim for this section**: {directive.central_claim}\n"
                f"**Structural suggestion**: {directive.structural_suggestion}\n"
                f"**Key insights to foreground**:\n{key_insights_text}\n"
                f"**Opening transition hint**: {directive.transition_from_prev}\n"
                f"**Closing transition hint**: {directive.transition_to_next}\n"
            )

        enrichment_text = ""
        if enrichment:
            enrichment_text = _format_contextual_enrichment(enrichment)

        prompt = build_section_writing_prompt(
            section_id=section.id,
            section_title=section.title,
            section_description=section.description,
            outline_context=outline_context,
            relevant_extractions=relevant,
            synthesis_directives=directives,
            adjacent_text=adjacent,
            narrative_guidance=narrative_guidance,
            contextual_enrichment=enrichment_text,
        )

        response = await self.llm.generate(
            prompt=prompt,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.55,
        )

        citations = _extract_citations(response.content)

        draft = SectionDraft(
            section_id=section.id,
            title=section.title,
            text=response.content,
            citations_used=citations,
        )

        logger.info(
            "section_writer.complete",
            section_id=section.id,
            title=section.title,
            word_count=len(response.content.split()),
            citations=len(citations),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return draft

    async def revise_section_with_evidence(
        self,
        section_id: str,
        section_title: str,
        existing_text: str,
        new_paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
    ) -> SectionDraft:
        """Revise an existing section to incorporate newly found papers.

        Args:
            section_id: Section identifier.
            section_title: Human-readable section title.
            existing_text: The current draft text.
            new_paper_ids: IDs of newly retrieved papers to incorporate.
            extractions: All available extractions (including new papers).

        Returns:
            Revised SectionDraft with new evidence incorporated.
        """
        new_evidence = _format_extractions(new_paper_ids, extractions)

        prompt = (
            f"## Section: {section_title}\n\n"
            f"## Existing Draft\n{existing_text}\n\n"
            f"## Newly Available Evidence\n{new_evidence}\n\n"
            "Revise the section to incorporate the new evidence where it strengthens "
            "the text. Add [@paper_id] citations for new claims. Preserve the existing "
            "structure and arguments. Do not pad with unnecessary content."
        )

        response = await self.llm.generate(
            prompt=prompt,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.3,
        )

        citations = _extract_citations(response.content)

        draft = SectionDraft(
            section_id=section_id,
            title=section_title,
            text=response.content,
            citations_used=citations,
        )

        logger.info(
            "section_writer.revised",
            section_id=section_id,
            new_papers=len(new_paper_ids),
            citations=len(citations),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return draft

    async def write_all_sections(
        self,
        outline: ReviewOutline,
        extractions: dict[str, PaperExtraction],
        evidence_map: EvidenceMap,
        narrative_plan: NarrativePlan | None = None,
        contextual_enrichment: dict[str, SectionEnrichment] | None = None,
    ) -> dict[str, SectionDraft]:
        """Write all sections sequentially with cross-section context."""
        drafts: dict[str, SectionDraft] = {}
        top_level = outline.sections  # Write top-level sections only (subsections included)

        # Build directive lookup by section_id
        directive_map: dict[str, SectionNarrativeDirective] = {}
        if narrative_plan:
            for d in narrative_plan.section_directives:
                directive_map[d.section_id] = d

        for i, section in enumerate(top_level):
            preceding = drafts[top_level[i - 1].id].text if i > 0 else ""
            following = ""  # Not yet written
            directive = directive_map.get(section.id)
            enrichment = contextual_enrichment.get(section.id) if contextual_enrichment else None

            draft = await self.write_section(
                section=section,
                outline=outline,
                extractions=extractions,
                evidence_map=evidence_map,
                preceding_text=preceding,
                following_text=following,
                directive=directive,
                enrichment=enrichment,
            )
            drafts[section.id] = draft

        logger.info("section_writer.all_complete", sections=len(drafts))
        return drafts
