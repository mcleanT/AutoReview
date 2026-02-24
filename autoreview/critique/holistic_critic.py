from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueReport, CritiqueTarget
from autoreview.critique.revision import revise_text, should_continue_revision
from autoreview.llm.prompts.critique import (
    HOLISTIC_CRITIQUE_SYSTEM_PROMPT,
    build_holistic_critique_prompt,
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
    ) -> CritiqueReport:
        """Critique the full assembled draft."""
        prompt = build_holistic_critique_prompt(full_draft, scope_document)

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=CritiqueReport,
            system=HOLISTIC_CRITIQUE_SYSTEM_PROMPT,
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
) -> tuple[str, list[CritiqueReport]]:
    """Run holistic critique → cross-section revision loop."""
    critiques: list[CritiqueReport] = []
    scores: list[float] = []
    current_draft = full_draft

    for cycle in range(max_cycles):
        report = await critic.critique(current_draft, scope_document)
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

        # Cross-section revision with full draft context
        current_draft = await revise_text(
            llm, current_draft, report,
            context=f"Full review scope: {scope_document[:1000]}",
        )

    return current_draft, critiques
