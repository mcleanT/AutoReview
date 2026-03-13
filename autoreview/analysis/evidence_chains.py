"""Evidence chain building and synthesis directive generation.

Builds structured evidence chains, temporal progressions, evidence strength
distributions, and enriched contradictions that feed into section-level
synthesis directives for the writer.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import Field

from autoreview.analysis.evidence_map import Contradiction, EvidenceMap
from autoreview.extraction.models import PaperExtraction
from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvidenceChainLink(AutoReviewModel):
    """A single link in an evidence chain."""

    paper_id: str
    claim: str
    evidence_strength: str
    year: int | None = None
    methodology_type: str = ""


class EvidenceChain(AutoReviewModel):
    """A chain of connected findings across papers."""

    chain_id: str
    theme: str
    description: str
    links: list[EvidenceChainLink] = Field(default_factory=list)
    chain_type: str  # progressive, replication, methodological_escalation, contradiction_resolution
    paper_ids: list[str] = Field(default_factory=list)


class TemporalProgression(AutoReviewModel):
    """How research focus shifted over time within a theme."""

    theme: str
    early_period: str  # e.g. "2015-2018"
    early_focus: str
    late_period: str  # e.g. "2019-2024"
    late_focus: str
    trajectory: str  # brief description of the shift
    paper_ids: list[str] = Field(default_factory=list)


class EvidenceStrengthDistribution(AutoReviewModel):
    """Distribution of evidence strength across papers in a section."""

    strong: int = 0
    moderate: int = 0
    weak: int = 0
    preliminary: int = 0
    dominant_type: str = ""
    writer_guidance: str = ""


class EnrichedContradiction(AutoReviewModel):
    """A contradiction enriched with methodology differences and framing strategy."""

    claim_a: str
    claim_b: str
    paper_ids_a: list[str] = Field(default_factory=list)
    paper_ids_b: list[str] = Field(default_factory=list)
    possible_explanation: str | None = None
    theme: str | None = None
    methodological_differences: str = ""
    framing_strategy: str = ""


class SynthesisDirectiveBundle(AutoReviewModel):
    """Complete synthesis directives for a section."""

    section_id: str
    evidence_chains: list[EvidenceChain] = Field(default_factory=list)
    strength_distribution: EvidenceStrengthDistribution | None = None
    enriched_contradictions: list[EnrichedContradiction] = Field(default_factory=list)
    temporal_progressions: list[TemporalProgression] = Field(default_factory=list)
    consensus_claims_text: str = ""
    directive_text: str = ""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class EvidenceChainBuilder:
    """Builds evidence chains and enriched synthesis directives."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def build_chains(
        self,
        extractions: dict[str, PaperExtraction],
        themes: list[Any],
        paper_years: dict[str, int],
    ) -> list[EvidenceChain]:
        """Build evidence chains from extractions and themes (LLM-assisted)."""
        from autoreview.llm.prompts.evidence_chains import (
            EVIDENCE_CHAIN_SYSTEM_PROMPT,
            build_evidence_chain_prompt,
        )

        all_chains: list[EvidenceChain] = []
        chain_counter = 0

        for theme in themes:
            # Collect findings with metadata for this theme
            findings_with_meta: list[dict] = []
            relationships: list[dict] = []

            for pid in theme.paper_ids:
                ext = extractions.get(pid)
                if not ext:
                    continue
                year = paper_years.get(pid)
                for f in ext.key_findings:
                    findings_with_meta.append(
                        {
                            "paper_id": pid,
                            "claim": f.claim,
                            "evidence_strength": f.evidence_strength,
                            "year": year,
                        }
                    )
                for r in ext.relationships:
                    relationships.append(
                        {
                            "source_paper_id": r.source_paper_id,
                            "target_paper_id": r.target_paper_id,
                            "relationship_type": r.relationship_type,
                            "description": r.description,
                        }
                    )

            if len(findings_with_meta) < 2:
                continue

            prompt = build_evidence_chain_prompt(theme.name, findings_with_meta, relationships)

            try:
                # Request structured chains from LLM
                from pydantic import Field as PydField

                from autoreview.models.base import AutoReviewModel

                class ChainResult(AutoReviewModel):
                    chains: list[dict] = PydField(default_factory=list)

                response = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=ChainResult,
                    system=EVIDENCE_CHAIN_SYSTEM_PROMPT,
                )
                result: ChainResult = response.parsed

                for c in result.chains:
                    chain_counter += 1
                    chain_paper_ids = c.get("paper_ids", [])
                    links = []
                    for pid in chain_paper_ids:
                        ext = extractions.get(pid)
                        if ext and ext.key_findings:
                            links.append(
                                EvidenceChainLink(
                                    paper_id=pid,
                                    claim=ext.key_findings[0].claim,
                                    evidence_strength=ext.key_findings[0].evidence_strength,
                                    year=paper_years.get(pid),
                                )
                            )

                    all_chains.append(
                        EvidenceChain(
                            chain_id=f"chain_{chain_counter}",
                            theme=theme.name,
                            description=c.get("description", ""),
                            links=links,
                            chain_type=c.get("chain_type", "progressive"),
                            paper_ids=chain_paper_ids,
                        )
                    )

            except Exception as e:
                logger.warning("evidence_chains.build_failed", theme=theme.name, error=str(e))

        logger.info("evidence_chains.built", total_chains=len(all_chains))
        return all_chains

    def compute_strength_distribution(
        self,
        paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
    ) -> EvidenceStrengthDistribution:
        """Compute evidence strength distribution for a set of papers (no LLM)."""
        counts = {"strong": 0, "moderate": 0, "weak": 0, "preliminary": 0}

        for pid in paper_ids:
            ext = extractions.get(pid)
            if not ext:
                continue
            for f in ext.key_findings:
                strength = f.evidence_strength
                if strength in counts:
                    counts[strength] += 1

        total = sum(counts.values())
        dominant = max(counts, key=counts.get) if total > 0 else ""

        guidance_map = {
            "strong": "Lead with confident assertions; most evidence is robust.",
            "moderate": (
                "Present findings with appropriate confidence; "
                "note that most evidence is moderate-strength."
            ),
            "weak": (
                "Use hedging language; emphasize that evidence is "
                "suggestive rather than conclusive."
            ),
            "preliminary": "Frame as an emerging area; highlight the need for further research.",
        }

        return EvidenceStrengthDistribution(
            strong=counts["strong"],
            moderate=counts["moderate"],
            weak=counts["weak"],
            preliminary=counts["preliminary"],
            dominant_type=dominant,
            writer_guidance=guidance_map.get(dominant, ""),
        )

    async def enrich_contradictions(
        self,
        contradictions: list[Contradiction],
        extractions: dict[str, PaperExtraction],
    ) -> list[EnrichedContradiction]:
        """Enrich contradictions with methodology differences and framing strategies.

        LLM-assisted enrichment.
        """
        from autoreview.llm.prompts.evidence_chains import (
            CONTRADICTION_ENRICHMENT_SYSTEM_PROMPT,
            build_contradiction_enrichment_prompt,
        )

        enriched: list[EnrichedContradiction] = []

        for c in contradictions:
            methods_a = [
                extractions[pid].methods_summary for pid in c.paper_ids_a if pid in extractions
            ]
            methods_b = [
                extractions[pid].methods_summary for pid in c.paper_ids_b if pid in extractions
            ]

            try:
                prompt = build_contradiction_enrichment_prompt(
                    c.claim_a,
                    c.claim_b,
                    methods_a,
                    methods_b,
                )

                from autoreview.models.base import AutoReviewModel

                class EnrichmentResult(AutoReviewModel):
                    methodological_differences: str = ""
                    framing_strategy: str = ""

                response = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=EnrichmentResult,
                    system=CONTRADICTION_ENRICHMENT_SYSTEM_PROMPT,
                )
                result: EnrichmentResult = response.parsed

                enriched.append(
                    EnrichedContradiction(
                        claim_a=c.claim_a,
                        claim_b=c.claim_b,
                        paper_ids_a=c.paper_ids_a,
                        paper_ids_b=c.paper_ids_b,
                        possible_explanation=c.possible_explanation,
                        theme=c.theme,
                        methodological_differences=result.methodological_differences,
                        framing_strategy=result.framing_strategy,
                    )
                )

            except Exception as e:
                logger.warning("contradiction_enrichment.failed", error=str(e))
                # Fall back to unenriched version
                enriched.append(
                    EnrichedContradiction(
                        claim_a=c.claim_a,
                        claim_b=c.claim_b,
                        paper_ids_a=c.paper_ids_a,
                        paper_ids_b=c.paper_ids_b,
                        possible_explanation=c.possible_explanation,
                        theme=c.theme,
                    )
                )

        return enriched

    def detect_temporal_progressions(
        self,
        paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
        paper_years: dict[str, int],
    ) -> list[TemporalProgression]:
        """Detect temporal progressions in research focus (computational, no LLM)."""
        # Filter papers with known years
        papers_with_years = [
            (pid, paper_years[pid])
            for pid in paper_ids
            if pid in paper_years and pid in extractions
        ]

        if len(papers_with_years) < 4:
            return []

        papers_with_years.sort(key=lambda x: x[1])
        years = [y for _, y in papers_with_years]
        min_year, max_year = years[0], years[-1]

        if max_year - min_year < 3:
            return []

        midpoint = min_year + (max_year - min_year) // 2
        early = [(pid, y) for pid, y in papers_with_years if y <= midpoint]
        late = [(pid, y) for pid, y in papers_with_years if y > midpoint]

        if not early or not late:
            return []

        # Summarize focus from findings
        def _summarize_focus(paper_list: list[tuple[str, int]]) -> str:
            claims = []
            for pid, _ in paper_list[:5]:
                ext = extractions.get(pid)
                if ext and ext.key_findings:
                    claims.append(ext.key_findings[0].claim)
            return "; ".join(claims[:3]) if claims else "(no findings)"

        early_focus = _summarize_focus(early)
        late_focus = _summarize_focus(late)
        early_years = [y for _, y in early]
        late_years = [y for _, y in late]

        return [
            TemporalProgression(
                theme="overall",
                early_period=f"{min(early_years)}-{max(early_years)}",
                early_focus=early_focus,
                late_period=f"{min(late_years)}-{max(late_years)}",
                late_focus=late_focus,
                trajectory=(
                    f"Research shifted from early findings "
                    f"({min(early_years)}-{max(early_years)}) to later "
                    f"developments ({min(late_years)}-{max(late_years)})"
                ),
                paper_ids=[pid for pid, _ in papers_with_years],
            )
        ]

    async def build_section_directives(
        self,
        section_id: str,
        section_paper_ids: list[str],
        evidence_map: EvidenceMap,
        extractions: dict[str, PaperExtraction],
        paper_years: dict[str, int],
        evidence_chains: list[EvidenceChain] | None = None,
        enriched_contradictions: list[EnrichedContradiction] | None = None,
        temporal_progressions: list[TemporalProgression] | None = None,
    ) -> SynthesisDirectiveBundle:
        """Build a complete synthesis directive bundle for a section."""
        section_set = set(section_paper_ids)

        # 1. Strength distribution
        strength_dist = self.compute_strength_distribution(section_paper_ids, extractions)

        # 2. Relevant evidence chains (matched by paper_ids overlap)
        # Use param if provided, else fall back to evidence_map data
        all_chains = evidence_chains or []
        if not all_chains and evidence_map.evidence_chains:
            for raw in evidence_map.evidence_chains:
                try:
                    all_chains.append(
                        EvidenceChain.model_validate(raw) if isinstance(raw, dict) else raw
                    )
                except Exception:
                    continue
        relevant_chains = []
        for chain in all_chains:
            if set(chain.paper_ids) & section_set:
                relevant_chains.append(chain)

        # 3. Relevant enriched contradictions
        all_enriched = enriched_contradictions or []
        if not all_enriched and evidence_map.enriched_contradictions:
            for raw in evidence_map.enriched_contradictions:
                try:
                    all_enriched.append(
                        EnrichedContradiction.model_validate(raw) if isinstance(raw, dict) else raw
                    )
                except Exception:
                    continue
        relevant_contradictions = []
        for ec in all_enriched:
            if (set(ec.paper_ids_a) | set(ec.paper_ids_b)) & section_set:
                relevant_contradictions.append(ec)

        # 4. Relevant temporal progressions
        all_progressions = temporal_progressions or []
        if not all_progressions and evidence_map.temporal_progressions:
            for raw in evidence_map.temporal_progressions:
                try:
                    all_progressions.append(
                        TemporalProgression.model_validate(raw) if isinstance(raw, dict) else raw
                    )
                except Exception:
                    continue
        relevant_progressions = []
        for tp in all_progressions:
            if set(tp.paper_ids) & section_set:
                relevant_progressions.append(tp)

        # 5. Consensus claims text
        consensus_lines = []
        for cc in evidence_map.consensus_claims:
            if set(cc.supporting_paper_ids) & section_set:
                consensus_lines.append(
                    f"- '{cc.claim}' (supported by {cc.evidence_count} papers, "
                    f"{cc.strength} evidence)"
                )
        consensus_text = "\n".join(consensus_lines)

        # 6. Build the directive text
        directive_parts = []

        if strength_dist.writer_guidance:
            directive_parts.append(
                f"**Evidence Profile:** {strength_dist.dominant_type} dominant "
                f"({strength_dist.strong}S/{strength_dist.moderate}M/"
                f"{strength_dist.weak}W/{strength_dist.preliminary}P). "
                f"{strength_dist.writer_guidance}"
            )

        if relevant_chains:
            chain_descs = []
            for chain in relevant_chains:
                pids = ", ".join(f"[@{pid}]" for pid in chain.paper_ids)
                chain_descs.append(f"- [{chain.chain_type}] {chain.description} ({pids})")
            directive_parts.append(
                "**Evidence Chains — trace these in your prose:**\n" + "\n".join(chain_descs)
            )

        if relevant_contradictions:
            contra_descs = []
            for ec in relevant_contradictions:
                framing = f" Framing: {ec.framing_strategy}" if ec.framing_strategy else ""
                contra_descs.append(f"- '{ec.claim_a}' vs '{ec.claim_b}'.{framing}")
            directive_parts.append("**Contradictions to address:**\n" + "\n".join(contra_descs))

        if relevant_progressions:
            for tp in relevant_progressions:
                directive_parts.append(
                    f"**Temporal Progression:** {tp.early_period} focused on: {tp.early_focus}. "
                    f"{tp.late_period} shifted to: {tp.late_focus}. "
                    f"Consider using chronological structure to trace this evolution."
                )

        if consensus_text:
            directive_parts.append(f"**Consensus Claims:**\n{consensus_text}")

        directive_text = "\n\n".join(directive_parts)

        return SynthesisDirectiveBundle(
            section_id=section_id,
            evidence_chains=relevant_chains,
            strength_distribution=strength_dist,
            enriched_contradictions=relevant_contradictions,
            temporal_progressions=relevant_progressions,
            consensus_claims_text=consensus_text,
            directive_text=directive_text,
        )
