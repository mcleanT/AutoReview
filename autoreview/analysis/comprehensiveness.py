"""Comprehensiveness checks for literature search validation."""
from __future__ import annotations

from enum import Enum
from typing import Any

import structlog
from pydantic import Field

from autoreview.models.base import AutoReviewModel
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()


class CheckStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ComprehensiveCheckResult(AutoReviewModel):
    """Result of a single comprehensiveness check."""

    check_name: str
    status: CheckStatus
    score: float = 0.0
    details: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class CoverageAnomalyChecker:
    """Detects anomalies in search and screening results.

    Checks: per-source result counts, screening rejection rate,
    papers missing abstracts, and total screened count.
    """

    def __init__(
        self,
        min_screened: int = 10,
        max_rejection_rate: float = 0.80,
        max_no_abstract_rate: float = 0.50,
    ) -> None:
        self.min_screened = min_screened
        self.max_rejection_rate = max_rejection_rate
        self.max_no_abstract_rate = max_no_abstract_rate

    def check(
        self,
        candidates: list[CandidatePaper],
        screened: list[ScreenedPaper],
        expected_sources: list[str] | None = None,
    ) -> ComprehensiveCheckResult:
        warnings: list[str] = []
        total_candidates = len(candidates)
        total_screened = len(screened)

        # Per-source counts
        source_counts: dict[str, int] = {}
        for p in candidates:
            source_counts[p.source_database] = source_counts.get(p.source_database, 0) + 1

        if expected_sources:
            for src in expected_sources:
                if source_counts.get(src, 0) == 0:
                    warnings.append(f"{src} returned 0 results")

        # Rejection rate
        rejection_rate = 1.0 - (total_screened / total_candidates) if total_candidates > 0 else 0.0
        if rejection_rate > self.max_rejection_rate:
            warnings.append(
                f"High rejection rate: {rejection_rate:.0%} of papers rejected during screening"
            )

        # Low screened count
        if total_screened < self.min_screened:
            warnings.append(f"Only {total_screened} papers passed screening (minimum: {self.min_screened})")

        # Missing abstracts
        no_abstract = sum(1 for p in candidates if not p.abstract)
        no_abstract_rate = no_abstract / total_candidates if total_candidates > 0 else 0.0
        if no_abstract_rate > self.max_no_abstract_rate:
            warnings.append(
                f"{no_abstract_rate:.0%} of candidates lack an abstract, reducing screening quality"
            )

        status = CheckStatus.WARNING if warnings else CheckStatus.PASSED
        details = "; ".join(warnings) if warnings else "All coverage metrics within normal range"
        score = 1.0 - (len(warnings) * 0.25)  # rough penalty per warning

        logger.info(
            "comprehensiveness.coverage_anomaly",
            status=status,
            total_candidates=total_candidates,
            total_screened=total_screened,
            rejection_rate=round(rejection_rate, 3),
            warnings=len(warnings),
        )

        return ComprehensiveCheckResult(
            check_name="coverage_anomaly",
            status=status,
            score=max(0.0, score),
            details=details,
            metrics={
                "total_candidates": total_candidates,
                "total_screened": total_screened,
                "rejection_rate": round(rejection_rate, 3),
                "no_abstract_rate": round(no_abstract_rate, 3),
                "source_counts": source_counts,
            },
        )
