from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueIssue, CritiqueReport, CritiqueTarget
from autoreview.critique.revision import revise_text, should_continue_revision
from autoreview.llm.prompts.critique import (
    build_holistic_critique_prompt,
    get_holistic_critique_system_prompt,
)

logger = structlog.get_logger()


class HolisticCritic:
    """Evaluates the complete assembled draft."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def critique(
        self,
        full_draft: str,
        scope_document: str,
        previous_scores: dict[str, float] | None = None,
    ) -> CritiqueReport:
        """Critique the full assembled draft."""
        prompt = build_holistic_critique_prompt(full_draft, scope_document)

        system_prompt = get_holistic_critique_system_prompt(previous_scores)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=CritiqueReport,
            system=system_prompt,
        )
        report: CritiqueReport = response.parsed
        report.target = CritiqueTarget.FULL_DRAFT

        logger.info(
            "holistic_critique.complete",
            passed=report.passed,
            score=report.overall_score,
            issues=len(report.issues),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return report


async def holistic_critique_loop(
    llm: Any,
    critic: HolisticCritic,
    full_draft: str,
    scope_document: str,
    max_cycles: int = 3,
    threshold: float = 0.80,
    convergence_delta: float = 0.02,
    extra_issues: list[CritiqueIssue] | None = None,
) -> tuple[str, list[CritiqueReport]]:
    """Run holistic critique → cross-section revision loop."""
    critiques: list[CritiqueReport] = []
    scores: list[float] = []
    current_draft = full_draft
    previous_scores: dict[str, float] | None = None

    for cycle in range(max_cycles):
        report = await critic.critique(
            current_draft, scope_document,
            previous_scores=previous_scores,
        )

        # Inject extra issues (e.g. from citation validation) into the first cycle
        if cycle == 0 and extra_issues:
            report.issues = list(extra_issues) + list(report.issues)

        critiques.append(report)
        scores.append(report.overall_score)

        logger.info(
            "holistic_critique_loop.cycle",
            cycle=cycle + 1,
            score=report.overall_score,
            passed=report.passed,
            issues=len(report.issues),
        )

        if report.passed or not should_continue_revision(
            scores,
            threshold=threshold,
            convergence_delta=convergence_delta,
            max_iterations=max_cycles,
        ):
            break

        # Track scores for compact rubrics on next cycle
        previous_scores = report.dimension_scores

        # Cross-section revision with full draft context
        current_draft = await revise_text(
            llm, current_draft, report,
            context=f"Full review scope: {scope_document[:1000]}",
        )

    return current_draft, critiques
