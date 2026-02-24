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


from autoreview.analysis.comprehensiveness import CoverageAnomalyChecker
from autoreview.models.paper import CandidatePaper, ScreenedPaper


def _make_candidates(n: int, source: str = "pubmed", with_abstract: bool = True) -> list[CandidatePaper]:
    return [
        CandidatePaper(
            title=f"Paper {i}",
            authors=["Author A"],
            source_database=source,
            abstract=f"Abstract {i}" if with_abstract else None,
        )
        for i in range(n)
    ]


def _make_screened(papers: list[CandidatePaper], score: int = 4) -> list[ScreenedPaper]:
    return [
        ScreenedPaper(paper=p, relevance_score=score, rationale="Relevant", include=score >= 3)
        for p in papers
    ]


class TestCoverageAnomalyChecker:
    def test_healthy_pipeline_passes(self):
        candidates = _make_candidates(100, "pubmed") + _make_candidates(100, "semantic_scholar")
        screened = _make_screened(candidates[:50])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.status == "passed"

    def test_warns_on_high_rejection_rate(self):
        candidates = _make_candidates(100)
        screened = _make_screened(candidates[:10])  # 90% rejected
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.status == "warning"
        assert "rejection" in result.details.lower()

    def test_warns_on_zero_results_from_source(self):
        # All papers from pubmed, zero from semantic_scholar
        candidates = _make_candidates(50, "pubmed")
        screened = _make_screened(candidates[:30])
        checker = CoverageAnomalyChecker()
        result = checker.check(
            candidates, screened,
            expected_sources=["pubmed", "semantic_scholar"],
        )
        assert result.status == "warning"
        assert "semantic_scholar" in result.details.lower()

    def test_warns_on_low_screened_count(self):
        candidates = _make_candidates(20)
        screened = _make_screened(candidates[:5])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.status == "warning"

    def test_warns_on_many_missing_abstracts(self):
        candidates = _make_candidates(100, with_abstract=False)
        screened = _make_screened(candidates[:50])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.status == "warning"
        assert "abstract" in result.details.lower()

    def test_metrics_populated(self):
        candidates = _make_candidates(80) + _make_candidates(20, "semantic_scholar")
        screened = _make_screened(candidates[:60])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert "total_candidates" in result.metrics
        assert "total_screened" in result.metrics
        assert "rejection_rate" in result.metrics
        assert "source_counts" in result.metrics
