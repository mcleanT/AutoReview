"""NarrativeArchitect — plans the narrative structure of a review paper before writing."""

from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.llm.prompts.narrative import (
    NARRATIVE_ARCHITECT_SYSTEM_PROMPT,
    build_narrative_planning_prompt,
)
from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.models.narrative import NarrativePlan

logger = structlog.get_logger()


class NarrativeArchitect:
    """Plans the narrative architecture of a review paper.

    Runs between outline validation and section writing. Produces a
    NarrativePlan with a central argument, narrative arc, and per-section
    directives that guide section writers toward structural variety and
    argumentative coherence.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def plan(
        self,
        outline: ReviewOutline,
        evidence_map: EvidenceMap,
        scope_document: str,
    ) -> NarrativePlan:
        """Generate a narrative plan for the review paper.

        Args:
            outline: The validated review outline.
            evidence_map: The full evidence map from clustering.
            scope_document: The scope document from query expansion.

        Returns:
            A NarrativePlan with directives for all outline sections.
        """
        prompt = build_narrative_planning_prompt(outline, evidence_map, scope_document)

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=NarrativePlan,
            system=NARRATIVE_ARCHITECT_SYSTEM_PROMPT,
        )
        plan: NarrativePlan = response.parsed

        logger.info(
            "narrative_architect.complete",
            central_argument=plan.central_argument[:100],
            sections=len(plan.section_directives),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return plan
