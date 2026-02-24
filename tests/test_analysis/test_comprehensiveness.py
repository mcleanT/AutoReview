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


from autoreview.analysis.comprehensiveness import QueryCoverageChecker
from autoreview.llm.provider import LLMStructuredResponse


class MockQueryCoverageLLM:
    """Returns a structured coverage assessment."""

    def __init__(self, uncovered: list[str] | None = None):
        self.uncovered = uncovered or []

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        from autoreview.llm.prompts.comprehensiveness import QueryCoverageResult, SubTopicCoverage
        if response_model == QueryCoverageResult:
            subtopics = [
                SubTopicCoverage(
                    sub_topic="Gut-brain axis",
                    covered=True,
                    matching_queries=["gut brain interaction"],
                ),
                SubTopicCoverage(
                    sub_topic="Microbiome composition",
                    covered=True,
                    matching_queries=["microbiome diversity"],
                ),
            ]
            for topic in self.uncovered:
                subtopics.append(
                    SubTopicCoverage(sub_topic=topic, covered=False, matching_queries=[])
                )
            return LLMStructuredResponse(
                parsed=QueryCoverageResult(
                    sub_topic_assessments=subtopics,
                    overall_coverage_score=0.5 if self.uncovered else 1.0,
                ),
                input_tokens=300,
                output_tokens=200,
            )
        raise ValueError(f"Unexpected: {response_model}")


class TestQueryCoverageChecker:
    async def test_full_coverage_passes(self):
        llm = MockQueryCoverageLLM(uncovered=[])
        checker = QueryCoverageChecker(llm)
        queries = {"pubmed": ["q1"], "semantic_scholar": ["q2"]}
        result = await checker.check(queries, "scope doc text")
        assert result.status == "passed"
        assert result.score == 1.0

    async def test_missing_subtopic_warns(self):
        llm = MockQueryCoverageLLM(uncovered=["Viral metagenomics"])
        checker = QueryCoverageChecker(llm)
        queries = {"pubmed": ["q1"]}
        result = await checker.check(queries, "scope doc text")
        assert result.status == "warning"
        assert "Viral metagenomics" in result.details

    async def test_metrics_include_subtopic_details(self):
        llm = MockQueryCoverageLLM(uncovered=[])
        checker = QueryCoverageChecker(llm)
        result = await checker.check({"pubmed": ["q1"]}, "scope")
        assert "sub_topics_covered" in result.metrics
        assert "sub_topics_total" in result.metrics
