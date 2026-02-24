"""Comprehensiveness checks for literature search validation."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from autoreview.models.base import AutoReviewModel


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
