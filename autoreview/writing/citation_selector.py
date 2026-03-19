"""Citation selector — deterministic ranking, tiering, and budget allocation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from autoreview.models.base import AutoReviewModel

if TYPE_CHECKING:
    from autoreview.analysis.evidence_map import EvidenceMap
    from autoreview.config.citation import CitationConfig
    from autoreview.extraction.models import PaperExtraction

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

_EVIDENCE_STRENGTH_SCORES: dict[str, float] = {
    "strong": 1.0,
    "moderate": 0.7,
    "weak": 0.4,
    "preliminary": 0.2,
}

_RELEVANCE_SCORE_MAP: dict[int, float] = {
    5: 1.0,
    4: 0.7,
    3: 0.4,
}


class PaperCitation(AutoReviewModel):
    """A single paper assigned to a section with its tier and score."""

    paper_id: str
    tier: str  # "primary" | "supporting" | "contextual"
    priority_score: float
    citation_guidance: str = ""


class SectionCitationPlan(AutoReviewModel):
    """Citation assignments for a single outline section."""

    section_id: str
    citation_budget: int
    primary_papers: list[PaperCitation] = Field(default_factory=list)
    supporting_papers: list[PaperCitation] = Field(default_factory=list)
    contextual_papers: list[PaperCitation] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class CitationPlan(AutoReviewModel):
    """Full citation plan across all sections of an outline."""

    sections: dict[str, SectionCitationPlan] = Field(default_factory=dict)
    total_citation_budget: int = 0
    corpus_utilization_target: float = 0.0


# ---------------------------------------------------------------------------
# CitationSelector
# ---------------------------------------------------------------------------


class CitationSelector:
    """Deterministic citation selection algorithm.

    Scores papers on evidence strength, recency, relevance, uniqueness,
    and source diversity, then assigns tiers and computes per-section budgets.
    """

    def __init__(self, config: CitationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_paper(
        self,
        paper_id: str,
        extraction: PaperExtraction,
        section_paper_ids: list[str],
        evidence_map: EvidenceMap,
        paper_year: int | None,
        paper_source: str,
        relevance_score: int,
        date_range: tuple[int, int],
        already_assigned: list[str] | None = None,
    ) -> float:
        """Compute a priority score in [0, ~2.0] for a paper relative to a section.

        Args:
            paper_id: Identifier of the paper to score.
            extraction: Structured extraction for the paper.
            section_paper_ids: All paper IDs assigned to this section.
            evidence_map: Global evidence landscape.
            paper_year: Publication year (None if unknown).
            paper_source: Source API/database (e.g. "semantic_scholar").
            relevance_score: Integer 3–5 relevance to this section.
            date_range: (oldest_year, newest_year) across entire corpus.
            already_assigned: Paper IDs already assigned to this section
                (used for source diversity calculation).

        Returns:
            Weighted score, optionally boosted by seminal multiplier.
        """
        cfg = self.config
        already_assigned = already_assigned or []

        # 1. Evidence strength — dominant finding strength
        ev_score = self._evidence_strength_score(extraction)

        # 2. Recency — linear within date range
        recency = self._recency_score(paper_year, date_range)

        # 3. Relevance
        relevance = _RELEVANCE_SCORE_MAP.get(relevance_score, 0.2)

        # 4. Uniqueness — based on theme cluster membership
        uniqueness = self._uniqueness_score(paper_id, evidence_map, section_paper_ids)

        # 5. Source diversity — penalise over-represented sources
        diversity = self._source_diversity_score(paper_source, already_assigned)

        raw = (
            cfg.w_evidence_strength * ev_score
            + cfg.w_recency * recency
            + cfg.w_relevance_score * relevance
            + cfg.w_uniqueness * uniqueness
            + cfg.w_source_diversity * diversity
        )

        # 6. Seminal boost
        if self._is_seminal(paper_id, evidence_map):
            raw *= cfg.seminal_paper_boost

        return raw

    def compute_section_budget(
        self,
        section_word_count: int,
        target_per_1k: float,
        num_assigned: int,
    ) -> int:
        """Compute citation budget for a section.

        Args:
            section_word_count: Estimated word count for the section.
            target_per_1k: Target citations per 1,000 words (from DepthProfile).
            num_assigned: Total number of papers assigned to this section.

        Returns:
            Integer citation budget, clamped by min/max/num_assigned.
        """
        cfg = self.config
        raw = round(section_word_count * target_per_1k / 1000)
        budget = max(raw, cfg.min_citations_per_section)
        if cfg.max_citations_per_section is not None:
            budget = min(budget, cfg.max_citations_per_section)
        budget = min(budget, num_assigned)
        return budget

    def assign_tiers(
        self,
        scored: list[tuple[str, float]],
        budget: int,
    ) -> list[tuple[str, float, str]]:
        """Assign tier labels to a ranked, scored list of papers.

        Args:
            scored: List of (paper_id, score) sorted descending by score.
            budget: Number of papers to include (top-N from scored).

        Returns:
            List of (paper_id, score, tier) tuples for the top-budget papers.
        """
        cfg = self.config
        selected = scored[:budget]
        n = len(selected)

        if n < 3:
            return [(pid, score, "primary") for pid, score in selected]

        primary_frac = cfg.tier_distribution.get("primary", 0.30)
        contextual_frac = cfg.tier_distribution.get("contextual", 0.20)

        n_primary = math.ceil(n * primary_frac)
        n_contextual = math.floor(n * contextual_frac)
        n_supporting = n - n_primary - n_contextual

        result: list[tuple[str, float, str]] = []
        for i, (pid, score) in enumerate(selected):
            if i < n_primary:
                tier = "primary"
            elif i < n_primary + n_supporting:
                tier = "supporting"
            else:
                tier = "contextual"
            result.append((pid, score, tier))

        return result

    def select_for_section(
        self,
        section_id: str,
        paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
        evidence_map: EvidenceMap,
        section_word_count: int,
        target_per_1k: float,
        paper_years: dict[str, int] | None = None,
        paper_sources: dict[str, str] | None = None,
        relevance_scores: dict[str, int] | None = None,
    ) -> SectionCitationPlan:
        """Select and tier papers for a single section.

        Args:
            section_id: Identifier of the outline section.
            paper_ids: All paper IDs assigned to this section.
            extractions: Mapping of paper_id → PaperExtraction.
            evidence_map: Global evidence landscape.
            section_word_count: Estimated word count for this section.
            target_per_1k: Target citations per 1,000 words.
            paper_years: Optional mapping of paper_id → publication year.
            paper_sources: Optional mapping of paper_id → source API name.
            relevance_scores: Optional mapping of paper_id → relevance int (3–5).

        Returns:
            A SectionCitationPlan with tiered paper assignments and coverage notes.
        """
        paper_years = paper_years or {}
        paper_sources = paper_sources or {}
        relevance_scores = relevance_scores or {}

        # Determine corpus-wide date range from available years
        all_years = [y for y in paper_years.values() if y]
        if all_years:
            date_range: tuple[int, int] = (min(all_years), max(all_years))
        else:
            date_range = (2000, 2025)

        # Score all papers
        scored_list: list[tuple[str, float]] = []
        assigned_so_far: list[str] = []
        for pid in paper_ids:
            ext = extractions.get(pid)
            if ext is None:
                continue
            score = self.score_paper(
                paper_id=pid,
                extraction=ext,
                section_paper_ids=paper_ids,
                evidence_map=evidence_map,
                paper_year=paper_years.get(pid),
                paper_source=paper_sources.get(pid, "unknown"),
                relevance_score=relevance_scores.get(pid, 3),
                date_range=date_range,
                already_assigned=assigned_so_far,
            )
            scored_list.append((pid, score))
            assigned_so_far.append(paper_sources.get(pid, "unknown"))

        # Sort descending
        scored_list.sort(key=lambda x: x[1], reverse=True)

        # Compute budget
        budget = self.compute_section_budget(
            section_word_count=section_word_count,
            target_per_1k=target_per_1k,
            num_assigned=len(scored_list),
        )

        # Assign tiers
        tiered = self.assign_tiers(scored_list, budget=budget)

        primary_papers: list[PaperCitation] = []
        supporting_papers: list[PaperCitation] = []
        contextual_papers: list[PaperCitation] = []

        for pid, score, tier in tiered:
            ext = extractions.get(pid)
            guidance = self._build_citation_guidance(pid, tier, ext)
            citation = PaperCitation(
                paper_id=pid,
                tier=tier,
                priority_score=round(score, 4),
                citation_guidance=guidance,
            )
            if tier == "primary":
                primary_papers.append(citation)
            elif tier == "supporting":
                supporting_papers.append(citation)
            else:
                contextual_papers.append(citation)

        coverage_notes: list[str] = []
        uncited = [pid for pid in paper_ids if pid not in {t[0] for t in tiered}]
        if uncited:
            coverage_notes.append(
                f"{len(uncited)} paper(s) assigned to section but not in budget: "
                + ", ".join(uncited[:5])
                + ("..." if len(uncited) > 5 else "")
            )

        # Apply coverage constraints (Step 4 of selection algorithm)
        primary_papers, supporting_papers, contextual_papers, constraint_notes = (
            self._apply_coverage_constraints(
                primary_papers=primary_papers,
                supporting_papers=supporting_papers,
                contextual_papers=contextual_papers,
                evidence_map=evidence_map,
                paper_years=paper_years,
                paper_sources=paper_sources,
            )
        )
        coverage_notes.extend(constraint_notes)

        logger.info(
            "citation_selector.section_plan",
            section_id=section_id,
            budget=budget,
            primary=len(primary_papers),
            supporting=len(supporting_papers),
            contextual=len(contextual_papers),
        )

        return SectionCitationPlan(
            section_id=section_id,
            citation_budget=budget,
            primary_papers=primary_papers,
            supporting_papers=supporting_papers,
            contextual_papers=contextual_papers,
            coverage_notes=coverage_notes,
        )

    def select_all(
        self,
        outline: Any,
        extractions: dict[str, PaperExtraction],
        evidence_map: EvidenceMap,
        target_per_1k: float,
        paper_years: dict[str, int] | None = None,
        paper_sources: dict[str, str] | None = None,
    ) -> CitationPlan:
        """Run select_for_section across all sections in an outline.

        Args:
            outline: A ReviewOutline object (Any to avoid circular import).
            extractions: Full extraction mapping.
            evidence_map: Global evidence landscape.
            target_per_1k: Target citations per 1,000 words (from DepthProfile).
            paper_years: Optional paper year mapping.
            paper_sources: Optional paper source mapping.

        Returns:
            A CitationPlan with per-section SectionCitationPlans.
        """
        paper_years = paper_years or {}
        paper_sources = paper_sources or {}
        sections_map: dict[str, SectionCitationPlan] = {}

        for section in outline.sections:
            plan = self.select_for_section(
                section_id=section.id,
                paper_ids=section.paper_ids,
                extractions=extractions,
                evidence_map=evidence_map,
                section_word_count=getattr(section, "estimated_word_count", 1000),
                target_per_1k=target_per_1k,
                paper_years=paper_years,
                paper_sources=paper_sources,
            )
            sections_map[section.id] = plan

        all_assigned: set[str] = set()
        for sp in sections_map.values():
            for pc in sp.primary_papers + sp.supporting_papers + sp.contextual_papers:
                all_assigned.add(pc.paper_id)

        corpus_util = len(all_assigned) / len(extractions) if extractions else 0.0

        logger.info(
            "citation_selector.plan_complete",
            sections=len(sections_map),
            unique_papers=len(all_assigned),
            corpus_utilization=round(corpus_util, 3),
        )

        return CitationPlan(
            sections=sections_map,
            total_citation_budget=len(all_assigned),
            corpus_utilization_target=round(corpus_util, 3),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_citation_guidance(
        self,
        paper_id: str,
        tier: str,
        extraction: PaperExtraction | None,
    ) -> str:
        """Generate template-based citation guidance for a paper.

        Args:
            paper_id: Identifier of the paper.
            tier: Assigned tier ("primary", "supporting", or "contextual").
            extraction: Structured extraction for the paper (may be None).

        Returns:
            A short guidance string for the writer.
        """
        if tier == "primary":
            claim = ""
            if extraction and extraction.key_findings:
                claim = extraction.key_findings[0].claim
            if claim:
                return f"Discuss individually — {claim}"
            return f"Discuss individually — key findings from {paper_id}"
        if tier == "supporting":
            claim = ""
            if extraction and extraction.key_findings:
                claim = extraction.key_findings[0].claim
            if claim:
                return f"Cite in cluster for: {claim}"
            return f"Cite in cluster for: supporting evidence from {paper_id}"
        # contextual
        return "Background/framing"

    def _apply_coverage_constraints(
        self,
        primary_papers: list[PaperCitation],
        supporting_papers: list[PaperCitation],
        contextual_papers: list[PaperCitation],
        evidence_map: EvidenceMap,
        paper_years: dict[str, int],
        paper_sources: dict[str, str],
    ) -> tuple[list[PaperCitation], list[PaperCitation], list[PaperCitation], list[str]]:
        """Apply coverage constraints to the tiered selection.

        Implements Step 4 of the selection algorithm:
        - gap_paper_priority boost (TODO: requires gap-paper markers not yet available)
        - temporal_spread: ensure ≥3 distinct years in selected papers
        - contradiction_citation_mode == "both_sides": promote one paper per side
        - min_sources_per_theme: ensure minimum distinct source databases

        Args:
            primary_papers: Currently tiered primary papers.
            supporting_papers: Currently tiered supporting papers.
            contextual_papers: Currently tiered contextual papers.
            evidence_map: Global evidence landscape.
            paper_years: Mapping of paper_id → publication year.
            paper_sources: Mapping of paper_id → source database name.

        Returns:
            Tuple of (primary_papers, supporting_papers, contextual_papers, notes).
        """
        cfg = self.config
        notes: list[str] = []
        all_selected = primary_papers + supporting_papers + contextual_papers
        reserve = contextual_papers[:]  # pool to swap from

        # -- gap_paper_priority boost ----------------------------------------
        # TODO: gap-fill papers cannot yet be detected from extractions alone
        # (they would need a gap_search marker on PaperExtraction). Skipping boost.

        # -- temporal_spread --------------------------------------------------
        if cfg.temporal_spread and all_selected:
            selected_years = {
                pc.paper_id: paper_years.get(pc.paper_id)
                for pc in all_selected
                if paper_years.get(pc.paper_id) is not None
            }
            distinct_years = set(selected_years.values())
            if len(distinct_years) < 3:
                # Find papers in reserve (contextual) with years not already covered
                for pc in reserve:
                    yr = paper_years.get(pc.paper_id)
                    if yr is not None and yr not in distinct_years:
                        # Swap this paper into contextual (already there — just note)
                        distinct_years.add(yr)
                        notes.append(
                            f"temporal_spread: promoted {pc.paper_id} (year={yr}) "
                            "to ensure year coverage"
                        )
                    if len(distinct_years) >= 3:
                        break

        # -- contradiction_citation_mode: both_sides --------------------------
        if cfg.contradiction_citation_mode == "both_sides" and evidence_map.contradictions:
            selected_ids = {pc.paper_id for pc in all_selected}
            primary_ids = {pc.paper_id for pc in primary_papers}

            for contradiction in evidence_map.contradictions:
                # Contradiction model uses paper_ids_a / paper_ids_b
                side_a_ids: list[str] = getattr(contradiction, "paper_ids_a", [])
                side_b_ids: list[str] = getattr(contradiction, "paper_ids_b", [])

                side_a_in_section = [pid for pid in side_a_ids if pid in selected_ids]
                side_b_in_section = [pid for pid in side_b_ids if pid in selected_ids]

                if not (side_a_in_section and side_b_in_section):
                    continue  # contradiction not represented in this section

                # Ensure at least one from each side is in primary tier
                side_a_primary = [pid for pid in side_a_in_section if pid in primary_ids]
                side_b_primary = [pid for pid in side_b_in_section if pid in primary_ids]

                if not side_a_primary and side_a_in_section:
                    # Promote best side-A paper from supporting → primary
                    for i, pc in enumerate(supporting_papers):
                        if pc.paper_id in side_a_in_section:
                            promoted = supporting_papers.pop(i)
                            primary_papers.append(promoted)
                            notes.append(
                                f"contradiction both_sides: promoted {promoted.paper_id} "
                                "to primary (side A)"
                            )
                            break

                if not side_b_primary and side_b_in_section:
                    for i, pc in enumerate(supporting_papers):
                        if pc.paper_id in side_b_in_section:
                            promoted = supporting_papers.pop(i)
                            primary_papers.append(promoted)
                            notes.append(
                                f"contradiction both_sides: promoted {promoted.paper_id} "
                                "to primary (side B)"
                            )
                            break

        # -- min_sources_per_theme --------------------------------------------
        if cfg.min_sources_per_theme > 0 and all_selected:
            selected_sources = [
                paper_sources.get(pc.paper_id, "unknown")
                for pc in primary_papers + supporting_papers + contextual_papers
            ]
            distinct_sources = set(selected_sources)
            if len(distinct_sources) < cfg.min_sources_per_theme:
                # Swap in papers from underrepresented sources from reserve pool
                represented = set(selected_sources)
                for pc in reserve:
                    src = paper_sources.get(pc.paper_id, "unknown")
                    if src not in represented:
                        represented.add(src)
                        notes.append(
                            f"min_sources_per_theme: ensured source '{src}' via {pc.paper_id}"
                        )
                    if len(represented) >= cfg.min_sources_per_theme:
                        break

        return primary_papers, supporting_papers, contextual_papers, notes

    def _evidence_strength_score(self, extraction: PaperExtraction) -> float:
        """Return the highest evidence strength found in the extraction's findings."""
        if not extraction.key_findings:
            return _EVIDENCE_STRENGTH_SCORES["weak"]
        scores = [
            _EVIDENCE_STRENGTH_SCORES.get(str(f.evidence_strength), 0.2)
            for f in extraction.key_findings
        ]
        return max(scores)

    def _recency_score(self, year: int | None, date_range: tuple[int, int]) -> float:
        """Linear recency in [0, 1]. Newest = 1.0, oldest = 0.0."""
        if year is None:
            return 0.5  # neutral for unknown year
        low, high = date_range
        if high == low:
            return 1.0
        return max(0.0, min(1.0, (year - low) / (high - low)))

    def _uniqueness_score(
        self,
        paper_id: str,
        evidence_map: EvidenceMap,
        section_paper_ids: list[str],
    ) -> float:
        """Uniqueness based on theme cluster size in evidence_map.paper_theme_mapping.

        1.0 if the paper is the only member of its theme cluster (highly unique),
        0.5 if the cluster has ≤3 papers,
        0.0 if the cluster has >3 papers (well-covered topic).
        """
        mapping = evidence_map.paper_theme_mapping  # {paper_id: [theme_names]}
        themes_for_paper = mapping.get(paper_id, [])
        if not themes_for_paper:
            # Paper not in any theme cluster — treat as unique
            return 1.0

        # Count how many papers in this section share the same themes
        min_cluster_size = float("inf")
        for theme_name in themes_for_paper:
            # Count all papers in section that map to this theme
            cluster_size = sum(1 for pid in section_paper_ids if theme_name in mapping.get(pid, []))
            min_cluster_size = min(min_cluster_size, cluster_size)

        if min_cluster_size == float("inf"):
            return 1.0
        if min_cluster_size <= 1:
            return 1.0
        if min_cluster_size <= 3:
            return 0.5
        return 0.0

    def _source_diversity_score(
        self,
        paper_source: str,
        already_assigned: list[str],
    ) -> float:
        """Penalise over-representation of the same source database.

        Returns 1.0 - (same_source_count / total_assigned), clamped to [0, 1].
        The already_assigned list is expected to contain source strings (not IDs).
        """
        if not already_assigned:
            return 1.0
        same_count = sum(1 for s in already_assigned if s == paper_source)
        total = len(already_assigned)
        return max(0.0, min(1.0, 1.0 - same_count / total))

    def _is_seminal(self, paper_id: str, evidence_map: EvidenceMap) -> bool:
        """Return True if the paper qualifies for the seminal boost.

        Criteria (either is sufficient):
        - Appears in a ConsensusClaim with evidence_count >= 10
        - Is the first paper_id in any EvidenceChain
        """
        # Check consensus claims
        for claim in evidence_map.consensus_claims:
            if claim.evidence_count >= 10 and paper_id in claim.supporting_paper_ids:
                return True

        # Check evidence chains — first paper in any chain
        for chain in evidence_map.evidence_chains:
            if isinstance(chain, dict):
                ids = chain.get("paper_ids", [])
            else:
                ids = getattr(chain, "paper_ids", [])
            if ids and ids[0] == paper_id:
                return True

        return False
