"""Per-dimension minimum score gates for the critique system.

Prevents a section from passing overall if any critical dimension falls
below its floor, even when the aggregate score is above threshold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionFailure:
    """A single dimension that fell below its required minimum."""

    dimension: str
    score: float
    minimum: float


@dataclass(frozen=True)
class DimensionGateResult:
    """Result of checking dimension gates against a set of scores."""

    passed: bool
    failures: list[DimensionFailure]


# Minimum floors applied after every section critique cycle.
DEFAULT_SECTION_GATES: dict[str, float] = {
    "synthesis_quality": 0.65,
    "citation_accuracy": 0.60,
}

# Minimum floors applied after every holistic critique cycle.
DEFAULT_HOLISTIC_GATES: dict[str, float] = {
    "synthesis_quality": 0.65,
    "narrative_arc": 0.60,
    "citation_accuracy": 0.60,
}


def check_dimension_gates(
    dimension_scores: dict[str, float],
    gates: dict[str, float],
) -> DimensionGateResult:
    """Check dimension scores against per-dimension minimum floors.

    Dimensions present in *gates* but absent from *dimension_scores* are
    silently ignored — the gate only fires when a score is actually present
    and below the floor.

    Args:
        dimension_scores: Mapping of dimension name → score (0–1).
        gates: Mapping of dimension name → minimum required score.

    Returns:
        DimensionGateResult with passed=True when no dimension is below its
        floor, or passed=False with the list of failures otherwise.
    """
    failures: list[DimensionFailure] = []
    for dimension, minimum in gates.items():
        score = dimension_scores.get(dimension)
        if score is not None and score < minimum:
            failures.append(DimensionFailure(dimension=dimension, score=score, minimum=minimum))

    return DimensionGateResult(passed=len(failures) == 0, failures=failures)
