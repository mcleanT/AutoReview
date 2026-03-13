from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.critique.models import CritiqueReport
from autoreview.llm.prompts.outline import (
    OUTLINE_REVISION_SYSTEM_PROMPT,
    OUTLINE_SYSTEM_PROMPT,
    ReviewOutline,
    build_outline_prompt,
    build_outline_revision_prompt,
)

logger = structlog.get_logger()


def _format_evidence_summary(evidence_map: EvidenceMap) -> str:
    """Format the evidence map into a text summary for the outliner."""
    blocks = []

    blocks.append("### Themes")
    for theme in evidence_map.themes:
        subs = ", ".join(st.name for st in theme.sub_themes) if theme.sub_themes else "none"
        blocks.append(
            f"- **{theme.name}**: {theme.description} "
            f"({len(theme.paper_ids)} papers, sub-themes: {subs})"
        )

    if evidence_map.consensus_claims:
        blocks.append("\n### Consensus Claims")
        for cc in evidence_map.consensus_claims:
            blocks.append(f"- {cc.claim} ({cc.evidence_count} papers, {cc.strength})")

    if evidence_map.contradictions:
        blocks.append("\n### Contradictions")
        for c in evidence_map.contradictions:
            blocks.append(
                f"- {c.claim_a} vs {c.claim_b} "
                f"(explanation: {c.possible_explanation or 'unknown'})"
            )

    if evidence_map.gaps:
        blocks.append("\n### Identified Gaps")
        for g in evidence_map.gaps:
            blocks.append(f"- [{g.severity}] {g.expected_topic}: {g.current_coverage}")

    return "\n".join(blocks)


class OutlineGenerator:
    """Generates review paper outlines from evidence maps."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def generate(
        self,
        evidence_map: EvidenceMap,
        scope_document: str,
        required_sections: list[str] | None = None,
        previous_outline: ReviewOutline | None = None,
        critique_report: CritiqueReport | None = None,
    ) -> ReviewOutline:
        """Generate or revise a review outline from the evidence map.

        When ``previous_outline`` and ``critique_report`` are provided, the
        prompt instructs the LLM to revise the existing outline based on the
        specific critique issues rather than generating from scratch.
        """
        if required_sections is None:
            required_sections = [
                "Introduction", "Methods of Review", "Results",
                "Discussion", "Future Directions",
            ]

        evidence_summary = _format_evidence_summary(evidence_map)

        if previous_outline is not None and critique_report is not None:
            # Revision mode: incorporate critique feedback
            from autoreview.critique.outline_critic import _outline_to_text
            from autoreview.critique.revision import _format_issues

            previous_outline_text = _outline_to_text(previous_outline)
            critique_issues_text = _format_issues(critique_report.issues)

            prompt = build_outline_revision_prompt(
                scope_document=scope_document,
                evidence_summary=evidence_summary,
                required_sections=required_sections,
                previous_outline_text=previous_outline_text,
                critique_issues_text=critique_issues_text,
                critique_score=critique_report.overall_score,
            )
            system_prompt = OUTLINE_REVISION_SYSTEM_PROMPT

            logger.info(
                "outline.revising",
                previous_score=critique_report.overall_score,
                issues_count=len(critique_report.issues),
            )
        else:
            # Fresh generation mode
            prompt = build_outline_prompt(scope_document, evidence_summary, required_sections)
            system_prompt = OUTLINE_SYSTEM_PROMPT

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=ReviewOutline,
            system=system_prompt,
        )
        outline: ReviewOutline = response.parsed

        logger.info(
            "outline.generated",
            sections=len(outline.sections),
            total_subsections=len(outline.flatten()) - len(outline.sections),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return outline
