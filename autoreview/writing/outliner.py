from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap, Theme
from autoreview.llm.prompts.outline import (
    OUTLINE_SYSTEM_PROMPT,
    ReviewOutline,
    build_outline_prompt,
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
    ) -> ReviewOutline:
        """Generate a review outline from the evidence map."""
        if required_sections is None:
            required_sections = [
                "Introduction", "Methods of Review", "Results",
                "Discussion", "Future Directions",
            ]

        evidence_summary = _format_evidence_summary(evidence_map)
        prompt = build_outline_prompt(scope_document, evidence_summary, required_sections)

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=ReviewOutline,
            system=OUTLINE_SYSTEM_PROMPT,
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
