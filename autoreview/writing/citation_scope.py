from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class CitationScopeResult:
    """Result of validating citation scope for a section."""

    out_of_scope: list[str]
    uncited_assigned: list[str]
    utilization_rate: float
    all_in_scope: bool


def validate_citation_scope(
    cited_ids: list[str],
    assigned_ids: list[str],
) -> CitationScopeResult:
    """Validate that section citations match the assigned paper set.

    Args:
        cited_ids: Paper IDs actually cited in the section text.
        assigned_ids: Paper IDs assigned to this section.

    Returns:
        CitationScopeResult with scope violations and utilization metrics.
    """
    assigned_set = set(assigned_ids)
    cited_set = set(cited_ids)

    out_of_scope = sorted(cited_set - assigned_set)
    uncited_assigned = sorted(assigned_set - cited_set)

    if assigned_set:
        utilization_rate = len(cited_set & assigned_set) / len(assigned_set)
    else:
        utilization_rate = 1.0

    all_in_scope = len(out_of_scope) == 0

    if out_of_scope:
        logger.warning(
            "citation_scope.out_of_scope",
            out_of_scope=out_of_scope,
            count=len(out_of_scope),
        )

    return CitationScopeResult(
        out_of_scope=out_of_scope,
        uncited_assigned=uncited_assigned,
        utilization_rate=utilization_rate,
        all_in_scope=all_in_scope,
    )
