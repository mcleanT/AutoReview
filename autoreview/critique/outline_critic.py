from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueReport, CritiqueTarget
from autoreview.llm.prompts.outline import (
    OUTLINE_CRITIQUE_SYSTEM_PROMPT,
    ReviewOutline,
    build_outline_critique_prompt,
)

logger = structlog.get_logger()


def _outline_to_text(outline: ReviewOutline) -> str:
    """Convert outline to readable text for critique."""
    lines = [f"# {outline.title}\n"]
    for section in outline.flatten():
        depth = section.id.count(".") + 1
        prefix = "#" * (depth + 1)
        lines.append(f"{prefix} [{section.id}] {section.title}")
        lines.append(f"  {section.description}")
        if section.paper_ids:
            lines.append(f"  Papers: {len(section.paper_ids)}")
        lines.append(f"  Est. words: {section.estimated_word_count}")
        lines.append("")
    return "\n".join(lines)


class OutlineCritic:
    """Evaluates review outlines for completeness and structure."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def critique(
        self,
        outline: ReviewOutline,
        scope_document: str,
    ) -> CritiqueReport:
        """Critique an outline and return a report."""
        outline_text = _outline_to_text(outline)
        prompt = build_outline_critique_prompt(outline_text, scope_document)

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=CritiqueReport,
            system=OUTLINE_CRITIQUE_SYSTEM_PROMPT,
        )
        report: CritiqueReport = response.parsed
        # Ensure target is set correctly
        report.target = CritiqueTarget.OUTLINE

        logger.info(
            "outline_critique.complete",
            passed=report.passed,
            score=report.overall_score,
            issues=len(report.issues),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return report
