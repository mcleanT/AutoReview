# autoreview/config/citation.py
"""Citation policy configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CitationConfig(BaseModel):
    """Controls citation policy — how aggressively and selectively to cite."""

    model_config = ConfigDict(extra="forbid")

    # Density
    citation_density: Literal["sparse", "standard", "dense", "exhaustive"] = "standard"
    min_citations_per_section: int = 8
    max_citations_per_section: int | None = None
    target_corpus_utilization: float = 0.25

    # Selection strategy
    selection_strategy: Literal[
        "evidence_weighted",
        "recency_weighted",
        "diversity_optimized",
        "impact_weighted",
        "balanced",
    ] = "balanced"

    # Paper tiering
    paper_tiers_enabled: bool = True
    tier_distribution: dict[str, float] = Field(
        default_factory=lambda: {"primary": 0.30, "supporting": 0.50, "contextual": 0.20}
    )
    seminal_paper_boost: float = 1.5

    # Priority scoring weights (must sum to 1.0)
    w_evidence_strength: float = 0.30
    w_recency: float = 0.20
    w_relevance_score: float = 0.25
    w_uniqueness: float = 0.15
    w_source_diversity: float = 0.10

    # Citation style
    cluster_citations: bool = True
    max_cluster_size: int = 5
    require_individual_discussion: Literal["primary_only", "all_tiers", "none"] = "primary_only"

    # Coverage constraints
    min_sources_per_theme: int = 2
    temporal_spread: bool = True
    contradiction_citation_mode: Literal["both_sides", "stronger_side", "all"] = "both_sides"
    gap_paper_priority: Literal["boost", "normal", "deprioritize"] = "boost"

    # Writer guidance
    writer_citation_hints: Literal[
        "none", "ranked_list", "annotated_tiers", "budget_with_tiers"
    ] = "budget_with_tiers"
    uncited_paper_warning: Literal["ignore", "warn", "force_include"] = "warn"

    @classmethod
    def for_depth(cls, depth: str) -> CitationConfig:
        """Return a CitationConfig with depth-appropriate defaults.

        Args:
            depth: One of "low", "medium", "deep", or "exhaustive".

        Returns:
            CitationConfig with selection_strategy and citation_density matched to depth.
        """
        strategy_map = {
            "low": "evidence_weighted",
            "medium": "balanced",
            "deep": "diversity_optimized",
            "exhaustive": "balanced",
        }
        density_map = {
            "low": "sparse",
            "medium": "standard",
            "deep": "dense",
            "exhaustive": "exhaustive",
        }
        strategy = strategy_map.get(depth, "balanced")
        density = density_map.get(depth, "standard")
        return cls(
            selection_strategy=strategy,  # type: ignore[arg-type]
            citation_density=density,  # type: ignore[arg-type]
        )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> CitationConfig:
        total = (
            self.w_evidence_strength
            + self.w_recency
            + self.w_relevance_score
            + self.w_uniqueness
            + self.w_source_diversity
        )
        if abs(total - 1.0) >= 1e-6:
            msg = f"Citation scoring weights must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self


class ReferenceMatchConfig(BaseModel):
    """Auto-calibrate citation settings to match a reference paper's structural profile.

    Matches structural characteristics only (word count, citation density, section count,
    citation style). NEVER copies or is influenced by the reference paper's actual citations.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    reference_path: Path | None = None

    # Auto-extracted targets (populated by pre-pipeline analysis)
    target_word_count: int | None = None
    target_citation_count: int | None = None
    target_citation_density: float | None = None
    target_section_count: int | None = None
    inferred_citation_style: str | None = None

    # Tolerances
    word_count_tolerance: float = 0.15
    citation_count_tolerance: float = 0.20

    # What to match
    match_word_count: bool = True
    match_citation_density: bool = True
    match_section_structure: bool = True
    match_citation_style: bool = True
