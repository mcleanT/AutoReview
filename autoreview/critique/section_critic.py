from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueIssue, CritiqueReport, CritiqueTarget
from autoreview.critique.revision import revise_text, should_continue_revision
from autoreview.llm.prompts.critique import (
    build_section_critique_prompt,
    get_section_critique_system_prompt,
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
        previous_scores: dict[str, float] | None = None,
    ) -> CritiqueReport:
        """Critique a single section draft."""
        # Build focused outline context (full detail for current+neighbors, title-only for rest)
        flat = outline.flatten()
        current_idx: int | None = None
        for i, s in enumerate(flat):
            if s.id == draft.section_id:
                current_idx = i
                break
        neighbor_ids: set[str] = set()
        if current_idx is not None:
            neighbor_ids.add(flat[current_idx].id)
            if current_idx > 0:
                neighbor_ids.add(flat[current_idx - 1].id)
            if current_idx < len(flat) - 1:
                neighbor_ids.add(flat[current_idx + 1].id)
        outline_lines = []
        for s in flat:
            if s.id in neighbor_ids:
                outline_lines.append(f"[{s.id}] {s.title}: {s.description}")
            else:
                outline_lines.append(f"[{s.id}] {s.title}")
        outline_context = "\n".join(outline_lines)

        prompt = build_section_critique_prompt(
            section_id=draft.section_id,
            section_title=draft.title,
            section_text=draft.text,
            outline_context=outline_context,
            adjacent_context=adjacent_text,
        )

        system_prompt = get_section_critique_system_prompt(previous_scores)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=CritiqueReport,
            system=system_prompt,
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
    extra_issues: list[CritiqueIssue] | None = None,
) -> tuple[SectionDraft, list[CritiqueReport]]:
    """Run critique → revision loop for a single section."""
    critiques: list[CritiqueReport] = []
    scores: list[float] = []
    current_draft = draft

    previous_scores: dict[str, float] | None = None
    for cycle in range(max_cycles):
        report = await critic.critique(
            current_draft, outline, adjacent_text,
            previous_scores=previous_scores,
        )

        # Inject extra issues (e.g. from citation validation) into the first cycle
        if cycle == 0 and extra_issues:
            report.issues = list(extra_issues) + list(report.issues)

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

        # Track scores for compact rubrics on next cycle
        previous_scores = report.dimension_scores

        # Revise
        revised_text = await revise_text(llm, current_draft.text, report)
        current_draft = SectionDraft(
            section_id=current_draft.section_id,
            title=current_draft.title,
            text=revised_text,
            citations_used=current_draft.citations_used,
        )

    return current_draft, critiques
