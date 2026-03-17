"""Depth profile configuration and evidence-weighted word budget allocation."""

from __future__ import annotations

from dataclasses import dataclass

from autoreview.config.models import DepthLevel


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
}


def get_depth_instructions(depth: DepthLevel, target_word_count: int) -> str:
    """Return depth-specific prose instructions for the section writer."""
    return _DEPTH_INSTRUCTIONS[depth].format(target_word_count=target_word_count)
