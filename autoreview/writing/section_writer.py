from __future__ import annotations

import re
from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.extraction.models import PaperExtraction
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.prompts.writing import SECTION_WRITING_SYSTEM_PROMPT, build_section_writing_prompt
from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()


class SectionDraft(AutoReviewModel):
    """A drafted section with metadata."""
    section_id: str
    title: str
    text: str
    citations_used: list[str] = []


def _format_outline_context(outline: ReviewOutline) -> str:
    """Format outline titles/descriptions for context."""
    lines = []
    for s in outline.flatten():
        depth = s.id.count(".") + 1
        indent = "  " * (depth - 1)
        lines.append(f"{indent}{s.id}. {s.title}: {s.description}")
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
) -> str:
    """Generate synthesis directives from the evidence map for this section."""
    directives = []

    # Check for contradictions relevant to this section
    for c in evidence_map.contradictions:
        if c.theme and any(ref in (c.theme or "") for ref in section.theme_refs):
            directives.append(
                f"- Address contradiction: '{c.claim_a}' vs '{c.claim_b}' "
                f"(possible explanation: {c.possible_explanation or 'unclear'})"
            )

    # Check for consensus claims
    for cc in evidence_map.consensus_claims:
        overlap = set(cc.supporting_paper_ids) & set(section.paper_ids)
        if overlap:
            directives.append(f"- Highlight consensus: '{cc.claim}' ({cc.evidence_count} papers)")

    return "\n".join(directives) if directives else ""


def _extract_citations(text: str) -> list[str]:
    """Extract [@paper_id] citation markers from text."""
    return re.findall(r"\[@([^\]]+)\]", text)


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
    ) -> SectionDraft:
        """Write a single section with full context."""
        outline_context = _format_outline_context(outline)
        relevant = _format_extractions(section.paper_ids, extractions)
        directives = _generate_synthesis_directives(section, evidence_map)

        adjacent = ""
        if preceding_text:
            adjacent += f"### Preceding Section\n{preceding_text[-2000:]}\n\n"
        if following_text:
            adjacent += f"### Following Section\n{following_text[:2000:]}"

        prompt = build_section_writing_prompt(
            section_id=section.id,
            section_title=section.title,
            section_description=section.description,
            outline_context=outline_context,
            relevant_extractions=relevant,
            synthesis_directives=directives,
            adjacent_text=adjacent,
        )

        response = await self.llm.generate(
            prompt=prompt,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.3,
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
    ) -> dict[str, SectionDraft]:
        """Write all sections sequentially with cross-section context."""
        drafts: dict[str, SectionDraft] = {}
        top_level = outline.sections  # Write top-level sections only (subsections included)

        for i, section in enumerate(top_level):
            preceding = drafts[top_level[i - 1].id].text if i > 0 else ""
            following = ""  # Not yet written

            draft = await self.write_section(
                section=section,
                outline=outline,
                extractions=extractions,
                evidence_map=evidence_map,
                preceding_text=preceding,
                following_text=following,
            )
            drafts[section.id] = draft

        logger.info("section_writer.all_complete", sections=len(drafts))
        return drafts
