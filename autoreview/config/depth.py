"""Depth profile configuration and evidence-weighted word budget allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from autoreview.config.models import DepthLevel

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

    from autoreview.analysis.evidence_map import EvidenceMap  # noqa: F401
    from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline  # noqa: F401

logger = structlog.get_logger()


@dataclass(frozen=True)
class DepthProfile:
    """Deterministic parameter set for a given depth level."""

    base_word_multiplier: float
    key_insights_range: tuple[int, int]
    evidence_chain_detail: str  # "critical_only" | "standard" | "exhaustive"
    total_word_budget: int
    min_section_words: int
    section_type_dampening: dict[str, float]
    max_tokens_override: int | None
    citation_density: str = "standard"
    target_citations_per_1k_words: float = 9.0
    min_total_citations: int = 50


_DEPTH_PROFILES: dict[DepthLevel, DepthProfile] = {
    DepthLevel.LOW: DepthProfile(
        base_word_multiplier=0.6,
        key_insights_range=(2, 3),
        evidence_chain_detail="critical_only",
        total_word_budget=4000,
        min_section_words=200,
        section_type_dampening={
            "introduction": 0.7,
            "conclusion": 0.6,
            "methods": 0.7,
            "body": 1.0,
        },
        max_tokens_override=None,
        citation_density="sparse",
        target_citations_per_1k_words=6.0,
        min_total_citations=25,
    ),
    DepthLevel.MEDIUM: DepthProfile(
        base_word_multiplier=1.0,
        key_insights_range=(3, 5),
        evidence_chain_detail="standard",
        total_word_budget=8000,
        min_section_words=400,
        section_type_dampening={
            "introduction": 0.8,
            "conclusion": 0.7,
            "methods": 0.8,
            "body": 1.0,
        },
        max_tokens_override=None,
        citation_density="standard",
        target_citations_per_1k_words=9.0,
        min_total_citations=75,
    ),
    DepthLevel.DEEP: DepthProfile(
        base_word_multiplier=2.5,
        key_insights_range=(7, 10),
        evidence_chain_detail="exhaustive",
        total_word_budget=25000,
        min_section_words=600,
        section_type_dampening={
            "introduction": 0.85,
            "conclusion": 0.75,
            "methods": 0.85,
            "body": 1.0,
        },
        max_tokens_override=16384,
        citation_density="dense",
        target_citations_per_1k_words=12.0,
        min_total_citations=150,
    ),
    DepthLevel.EXHAUSTIVE: DepthProfile(
        base_word_multiplier=4.0,
        key_insights_range=(10, 15),
        evidence_chain_detail="exhaustive",
        total_word_budget=40000,
        min_section_words=800,
        section_type_dampening={
            "introduction": 0.85,
            "conclusion": 0.75,
            "methods": 0.85,
            "body": 1.0,
        },
        max_tokens_override=16384,
        citation_density="exhaustive",
        target_citations_per_1k_words=16.0,
        min_total_citations=300,
    ),
}


def get_depth_profile(depth: DepthLevel) -> DepthProfile:
    """Look up the depth profile for a given depth level."""
    return _DEPTH_PROFILES[depth]


_DEPTH_INSTRUCTIONS: dict[DepthLevel, str] = {
    DepthLevel.LOW: (
        "Distill to the most critical findings. One paragraph per key claim. "
        "Prioritize clarity and impact over completeness. "
        "Target approximately {target_word_count} words."
    ),
    DepthLevel.MEDIUM: (
        "Balance thoroughness with readability. Cover major findings with "
        "supporting context. Standard academic review depth. "
        "Target approximately {target_word_count} words."
    ),
    DepthLevel.DEEP: (
        "Exhaustively trace evidence chains. Include methodological comparisons, "
        "conflicting results with resolution analysis, temporal evolution of findings, "
        "and secondary implications. Prioritize completeness over brevity. "
        "Target approximately {target_word_count} words."
    ),
    DepthLevel.EXHAUSTIVE: (
        "Produce a comprehensive, encyclopedic treatment of all evidence. "
        "Trace every evidence chain in full detail. Include all methodological comparisons, "
        "conflicting results with thorough resolution analysis, complete temporal evolution, "
        "secondary and tertiary implications, and all identified gaps and future directions. "
        "Cite densely ({target_word_count} words at ~16 citations per 1000 words). "
        "Completeness is paramount — do not omit any relevant study or finding. "
        "Target approximately {target_word_count} words."
    ),
}


def get_depth_instructions(depth: DepthLevel, target_word_count: int) -> str:
    """Return depth-specific prose instructions for the section writer."""
    return _DEPTH_INSTRUCTIONS[depth].format(target_word_count=target_word_count)


def classify_section_type(title: str) -> str:
    """Classify a section title into a type for dampening lookup.

    Uses fuzzy substring matching against known section categories.
    Returns one of: "introduction", "conclusion", "methods", "body".
    """
    lower = title.lower()
    if "introduction" in lower or "background" in lower:
        return "introduction"
    if "conclusion" in lower or "concluding" in lower:
        return "conclusion"
    if "method" in lower or "search strategy" in lower or "review methodology" in lower:
        return "methods"
    return "body"


class EvidenceWeightedAllocator:
    """Distributes word budget across sections based on evidence density."""

    def __init__(self, profile: DepthProfile) -> None:
        self.profile = profile

    def allocate(
        self,
        outline: ReviewOutline,
        evidence_map: EvidenceMap,
        extractions: dict[str, Any],
    ) -> None:
        """Mutate outline sections' estimated_word_count in place."""
        sections = outline.sections
        if not sections:
            return

        evidence_sections: list[tuple[int, OutlineSection, float]] = []
        fixed_total = 0

        for i, section in enumerate(sections):
            density = self._compute_density(section, extractions, evidence_map)
            if density == 0.0:
                fixed_alloc = int(self.profile.base_word_multiplier * 500)
                section.estimated_word_count = fixed_alloc
                fixed_total += fixed_alloc
            else:
                evidence_sections.append((i, section, density))

        if not evidence_sections:
            return

        remaining_budget = self.profile.total_word_budget - fixed_total
        total_density = sum(d for _, _, d in evidence_sections)

        for _, section, density in evidence_sections:
            share = density / total_density
            raw = share * remaining_budget
            dampening = self.profile.section_type_dampening.get(
                classify_section_type(section.title), 1.0
            )
            adjusted = raw * dampening
            section.estimated_word_count = max(int(adjusted), self.profile.min_section_words)

        current_total = sum(s.estimated_word_count for s in sections)
        body_sections = [
            (i, s, d) for i, s, d in evidence_sections if classify_section_type(s.title) == "body"
        ]

        if body_sections and current_total != self.profile.total_word_budget:
            delta = self.profile.total_word_budget - current_total
            body_density_total = sum(d for _, _, d in body_sections)
            if body_density_total > 0:
                for _, section, density in body_sections:
                    adjustment = int(delta * (density / body_density_total))
                    new_count = section.estimated_word_count + adjustment
                    section.estimated_word_count = max(new_count, self.profile.min_section_words)

        logger.info(
            "depth_allocation_complete",
            depth=self.profile.evidence_chain_detail,
            budget=self.profile.total_word_budget,
            actual=sum(s.estimated_word_count for s in sections),
            sections={s.id: s.estimated_word_count for s in sections},
        )

    def _compute_density(
        self,
        section: OutlineSection,
        extractions: dict[str, Any],
        evidence_map: EvidenceMap,
    ) -> float:
        n_papers = len(section.paper_ids)
        n_findings = sum(
            len(extractions[pid].key_findings) for pid in section.paper_ids if pid in extractions
        )
        section_pids = set(section.paper_ids)
        n_chains = 0
        for chain in evidence_map.evidence_chains:
            chain_pids = (
                chain.get("paper_ids", [])
                if isinstance(chain, dict)
                else getattr(chain, "paper_ids", [])
            )
            if set(chain_pids) & section_pids:
                n_chains += 1
        return float(n_papers + n_findings + n_chains)
