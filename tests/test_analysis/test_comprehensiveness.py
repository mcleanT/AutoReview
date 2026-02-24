from __future__ import annotations

import pytest

from autoreview.analysis.comprehensiveness import CheckStatus, ComprehensiveCheckResult


class TestComprehensiveCheckResult:
    def test_create_passed_result(self):
        result = ComprehensiveCheckResult(
            check_name="test_check",
            status=CheckStatus.PASSED,
            score=0.95,
            details="All good",
            metrics={"items_checked": 10},
        )
        assert result.check_name == "test_check"
        assert result.status == "passed"
        assert result.score == 0.95

    def test_create_warning_result(self):
        result = ComprehensiveCheckResult(
            check_name="test_check",
            status=CheckStatus.WARNING,
            score=0.4,
            details="Below threshold",
            metrics={},
        )
        assert result.status == "warning"
