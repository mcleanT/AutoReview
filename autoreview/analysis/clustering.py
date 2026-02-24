from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.evidence_map import (
    ConsensusClaim,
    Contradiction,
    EvidenceMap,
    SubTheme,
    Theme,
)
from autoreview.extraction.models import PaperExtraction
from autoreview.llm.prompts.clustering import (
    CLUSTERING_SYSTEM_PROMPT,
    CONTRADICTION_SYSTEM_PROMPT,
    ClusteringResult,
    ContradictionResult,
    build_clustering_prompt,
    build_contradiction_prompt,
)

logger = structlog.get_logger()


def _format_findings_for_clustering(extractions: dict[str, PaperExtraction]) -> str:
    """Format all findings into a text block for the clustering prompt."""
    blocks = []
    for paper_id, ext in extractions.items():
        findings_text = "; ".join(
            f"[{f.evidence_strength}] {f.claim}" for f in ext.key_findings
        )
        blocks.append(f"Paper {paper_id[:8]}:\n  Findings: {findings_text}")
    return "\n\n".join(blocks)


def _format_theme_findings(
    theme_paper_ids: list[str],
    extractions: dict[str, PaperExtraction],
) -> str:
    """Format findings for a specific theme's papers."""
    blocks = []
    for pid in theme_paper_ids:
        ext = extractions.get(pid)
        if not ext:
            continue
        for f in ext.key_findings:
            blocks.append(f"[Paper {pid[:8]}] [{f.evidence_strength}] {f.claim}")
    return "\n".join(blocks)


class ThematicClusterer:
    """Clusters findings into themes using LLM-based analysis."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def cluster(
        self,
        extractions: dict[str, PaperExtraction],
        scope_document: str,
    ) -> list[Theme]:
        """Cluster findings into themes and sub-themes."""
        findings_text = _format_findings_for_clustering(extractions)

        prompt = build_clustering_prompt(scope_document, findings_text)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=ClusteringResult,
            system=CLUSTERING_SYSTEM_PROMPT,
        )
        result: ClusteringResult = response.parsed

        logger.info(
            "clustering.complete",
            themes=len(result.themes),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        # Convert to domain models
        themes: list[Theme] = []
        for tc in result.themes:
            sub_themes = [
                SubTheme(
                    name=st.name,
                    description=st.description,
                    paper_ids=st.paper_ids,
                    finding_claims=st.finding_claims,
                )
                for st in tc.sub_themes
            ]
            themes.append(
                Theme(
                    name=tc.name,
                    description=tc.description,
                    paper_ids=tc.paper_ids,
                    sub_themes=sub_themes,
                )
            )

        return themes

    async def detect_contradictions(
        self,
        themes: list[Theme],
        extractions: dict[str, PaperExtraction],
    ) -> tuple[list[ConsensusClaim], list[Contradiction]]:
        """Detect consensus claims and contradictions within each theme."""
        all_consensus: list[ConsensusClaim] = []
        all_contradictions: list[Contradiction] = []

        for theme in themes:
            findings_text = _format_theme_findings(theme.paper_ids, extractions)
            if not findings_text:
                continue

            prompt = build_contradiction_prompt(theme.name, findings_text)
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=ContradictionResult,
                system=CONTRADICTION_SYSTEM_PROMPT,
            )
            result: ContradictionResult = response.parsed

            for cc in result.consensus_claims:
                all_consensus.append(
                    ConsensusClaim(
                        claim=cc.claim,
                        supporting_paper_ids=cc.supporting_paper_ids,
                        strength=cc.strength,
                        evidence_count=cc.evidence_count,
                    )
                )

            for ci in result.contradictions:
                all_contradictions.append(
                    Contradiction(
                        claim_a=ci.claim_a,
                        claim_b=ci.claim_b,
                        paper_ids_a=ci.paper_ids_a,
                        paper_ids_b=ci.paper_ids_b,
                        possible_explanation=ci.possible_explanation,
                        theme=theme.name,
                    )
                )

            logger.info(
                "contradiction_detection.theme",
                theme=theme.name,
                consensus=len(result.consensus_claims),
                contradictions=len(result.contradictions),
            )

        return all_consensus, all_contradictions

    async def build_evidence_map(
        self,
        extractions: dict[str, PaperExtraction],
        scope_document: str,
        gaps: list[Any] | None = None,
        coverage_score: float = 0.0,
    ) -> EvidenceMap:
        """Full pipeline: cluster → detect contradictions → build evidence map."""
        themes = await self.cluster(extractions, scope_document)
        consensus, contradictions = await self.detect_contradictions(themes, extractions)

        # Build paper-theme mapping
        paper_theme_mapping: dict[str, list[str]] = {}
        for theme in themes:
            for pid in theme.paper_ids:
                paper_theme_mapping.setdefault(pid, []).append(theme.name)

        return EvidenceMap(
            themes=themes,
            consensus_claims=consensus,
            contradictions=contradictions,
            gaps=gaps or [],
            paper_theme_mapping=paper_theme_mapping,
            coverage_score=coverage_score,
        )
