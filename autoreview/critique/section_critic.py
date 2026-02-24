from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueReport, CritiqueTarget
from autoreview.critique.revision import revise_text, should_continue_revision
from autoreview.llm.prompts.critique import (
    SECTION_CRITIQUE_SYSTEM_PROMPT,
    build_section_critique_prompt,
)
from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.writing.section_writer import SectionDraft

logger = structlog.get_logger()


class SectionCritic:
    """Evaluates individual sections of a review paper."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def critique(
        self,
        draft: SectionDraft,
        outline: ReviewOutline,
        adjacent_text: str = "",
    ) -> CritiqueReport:
        """Critique a single section draft."""
        # Build outline context
        outline_lines = []
        for s in outline.flatten():
            outline_lines.append(f"[{s.id}] {s.title}: {s.description}")
        outline_context = "\n".join(outline_lines)

        prompt = build_section_critique_prompt(
            section_id=draft.section_id,
            section_title=draft.title,
            section_text=draft.text,
            outline_context=outline_context,
            adjacent_context=adjacent_text,
        )

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=CritiqueReport,
            system=SECTION_CRITIQUE_SYSTEM_PROMPT,
        )
        report: CritiqueReport = response.parsed
        report.target = CritiqueTarget.SECTION
        report.target_id = draft.section_id

        logger.info(
            "section_critique.complete",
            section_id=draft.section_id,
            passed=report.passed,
            score=report.overall_score,
            issues=len(report.issues),
        )

        return report


async def section_critique_loop(
    llm: Any,
    critic: SectionCritic,
    draft: SectionDraft,
    outline: ReviewOutline,
    adjacent_text: str = "",
    max_cycles: int = 2,
    threshold: float = 0.80,
) -> tuple[SectionDraft, list[CritiqueReport]]:
    """Run critique → revision loop for a single section."""
    critiques: list[CritiqueReport] = []
    scores: list[float] = []
    current_draft = draft

    for cycle in range(max_cycles):
        report = await critic.critique(current_draft, outline, adjacent_text)
        critiques.append(report)
        scores.append(report.overall_score)

        logger.info(
            "section_critique_loop.cycle",
            section=current_draft.section_id,
            cycle=cycle + 1,
            score=report.overall_score,
            passed=report.passed,
        )

        if report.passed or not should_continue_revision(
            scores, threshold=threshold, max_iterations=max_cycles
        ):
            break

        # Revise
        revised_text = await revise_text(llm, current_draft.text, report)
        current_draft = SectionDraft(
            section_id=current_draft.section_id,
            title=current_draft.title,
            text=revised_text,
            citations_used=current_draft.citations_used,
        )

    return current_draft, critiques
