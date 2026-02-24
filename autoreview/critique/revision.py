from __future__ import annotations

from typing import Any

import structlog

from autoreview.critique.models import CritiqueIssue, CritiqueReport
from autoreview.llm.prompts.critique import build_revision_prompt
from autoreview.llm.prompts.outline import ReviewOutline

logger = structlog.get_logger()


def should_continue_revision(
    scores: list[float],
    threshold: float = 0.80,
    convergence_delta: float = 0.02,
    max_iterations: int = 3,
) -> bool:
    """Determine whether to continue revision cycles.

    Stops if:
    1. Score exceeds threshold
    2. Max iterations reached
    3. Improvement < convergence_delta (score plateau)
    """
    if not scores:
        return True

    if len(scores) >= max_iterations:
        logger.info("revision.max_iterations", iterations=len(scores))
        return False

    if scores[-1] >= threshold:
        logger.info("revision.threshold_met", score=scores[-1], threshold=threshold)
        return False

    if len(scores) >= 2:
        improvement = scores[-1] - scores[-2]
        if improvement < convergence_delta:
            logger.info(
                "revision.converged",
                improvement=improvement,
                delta=convergence_delta,
            )
            return False

    return True


def _format_issues(issues: list[CritiqueIssue]) -> str:
    """Format critique issues into text for revision prompts."""
    lines = []
    for issue in issues:
        fix = f" Suggested fix: {issue.suggested_fix}" if issue.suggested_fix else ""
        lines.append(f"- [{issue.severity}] {issue.location}: {issue.description}{fix}")
    return "\n".join(lines) if lines else "No specific issues identified."


async def revise_text(
    llm: Any,
    text: str,
    critique: CritiqueReport,
    context: str = "",
) -> str:
    """Revise text based on critique feedback."""
    issues_text = _format_issues(critique.issues)
    prompt = build_revision_prompt(text, issues_text, context)

    response = await llm.generate(
        prompt=prompt,
        system="You are an expert scientific writer revising a review paper section.",
        temperature=0.3,
    )

    logger.info(
        "revision.complete",
        target=critique.target,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )

    return response.content


async def outline_critique_loop(
    llm: Any,
    outline_generator: Any,
    outline_critic: Any,
    evidence_map: Any,
    scope_document: str,
    required_sections: list[str] | None = None,
    max_cycles: int = 2,
    threshold: float = 0.80,
) -> tuple[ReviewOutline, list[CritiqueReport]]:
    """Run the outline generation → critique → revision loop.

    Returns:
        Tuple of (final outline, list of critique reports).
    """
    outline = await outline_generator.generate(
        evidence_map, scope_document, required_sections
    )
    critiques: list[CritiqueReport] = []
    scores: list[float] = []

    for cycle in range(max_cycles):
        report = await outline_critic.critique(outline, scope_document)
        critiques.append(report)
        scores.append(report.overall_score)

        logger.info(
            "outline_critique_loop.cycle",
            cycle=cycle + 1,
            score=report.overall_score,
            passed=report.passed,
            issues=len(report.issues),
        )

        if report.passed or not should_continue_revision(
            scores, threshold=threshold, max_iterations=max_cycles
        ):
            break

        # Revise: regenerate outline with critique feedback
        outline = await outline_generator.generate(
            evidence_map, scope_document, required_sections
        )

    return outline, critiques
