from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap, GapSeverity, IdentifiedGap, Theme
from autoreview.llm.prompts.clustering import (
    GAP_ANALYSIS_SYSTEM_PROMPT,
    GapAnalysisResult,
    build_gap_analysis_prompt,
)

logger = structlog.get_logger()


def _format_themes_for_gap_analysis(themes: list[Theme]) -> str:
    """Format themes into a text block for gap analysis."""
    blocks = []
    for theme in themes:
        sub_text = ""
        if theme.sub_themes:
            subs = ", ".join(st.name for st in theme.sub_themes)
            sub_text = f"\n  Sub-themes: {subs}"
        blocks.append(
            f"Theme: {theme.name}\n"
            f"  Description: {theme.description}\n"
            f"  Papers: {len(theme.paper_ids)}{sub_text}"
        )
    return "\n\n".join(blocks)


class GapDetector:
    """Detects gaps in evidence coverage compared to the review scope."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def detect_gaps(
        self,
        themes: list[Theme],
        scope_document: str,
    ) -> tuple[list[IdentifiedGap], float]:
        """Detect gaps in evidence coverage.

        Returns:
            Tuple of (list of identified gaps, coverage score 0.0-1.0).
        """
        themes_text = _format_themes_for_gap_analysis(themes)

        prompt = build_gap_analysis_prompt(scope_document, themes_text)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=GapAnalysisResult,
            system=GAP_ANALYSIS_SYSTEM_PROMPT,
        )
        result: GapAnalysisResult = response.parsed

        gaps: list[IdentifiedGap] = []
        for g in result.gaps:
            severity = GapSeverity.MAJOR if g.severity.lower() == "major" else GapSeverity.MINOR
            gaps.append(
                IdentifiedGap(
                    expected_topic=g.expected_topic,
                    current_coverage=g.current_coverage,
                    severity=severity,
                    suggested_queries=g.suggested_queries,
                )
            )

        coverage_score = max(0.0, min(1.0, result.coverage_score))

        logger.info(
            "gap_detection.complete",
            gaps_found=len(gaps),
            major_gaps=sum(1 for g in gaps if g.severity == "major"),
            coverage_score=coverage_score,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return gaps, coverage_score
