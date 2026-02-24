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


from autoreview.analysis.comprehensiveness import BorderlineRescreener
from autoreview.llm.prompts.screening import ScreeningBatchResult, ScreeningDecision


class MockRescreeningLLM:
    """First call returns borderline scores, second call promotes some."""

    def __init__(self):
        self.call_count = 0

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        self.call_count += 1
        if response_model == ScreeningBatchResult:
            # Second pass: promote the first paper, keep second as marginal
            return LLMStructuredResponse(
                parsed=ScreeningBatchResult(decisions=[
                    ScreeningDecision(paper_index=0, relevance_score=3, rationale="Relevant on second look"),
                    ScreeningDecision(paper_index=1, relevance_score=2, rationale="Still marginal"),
                ]),
                input_tokens=300,
                output_tokens=150,
            )
        raise ValueError(f"Unexpected: {response_model}")


class TestBorderlineRescreener:
    async def test_promotes_borderline_papers(self):
        llm = MockRescreeningLLM()
        rescreener = BorderlineRescreener(llm)
        borderline = [
            CandidatePaper(title="Borderline 1", authors=["A"], source_database="test", abstract="Abstract 1"),
            CandidatePaper(title="Borderline 2", authors=["A"], source_database="test", abstract="Abstract 2"),
        ]
        result, promoted = await rescreener.rescreen(borderline, "scope doc")
        assert result.check_name == "borderline_rescreening"
        assert len(promoted) == 1
        assert promoted[0].relevance_score == 3
        assert result.metrics["promoted_count"] == 1
        assert result.metrics["borderline_count"] == 2

    async def test_no_borderlines_passes(self):
        llm = MockRescreeningLLM()
        rescreener = BorderlineRescreener(llm)
        result, promoted = await rescreener.rescreen([], "scope doc")
        assert result.status == "passed"
        assert len(promoted) == 0
        assert result.details == "No borderline papers to re-screen"


from autoreview.analysis.comprehensiveness import PostGapRevalidator
from autoreview.analysis.evidence_map import (
    EvidenceMap, GapSeverity, IdentifiedGap, Theme,
)
from autoreview.llm.prompts.clustering import GapAnalysisResult, GapItem


class MockPostGapLLM:
    """Returns improved or unchanged gap analysis."""

    def __init__(self, post_score: float = 0.9, remaining_major: int = 0):
        self.post_score = post_score
        self.remaining_major = remaining_major

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        if response_model == GapAnalysisResult:
            gaps = []
            for i in range(self.remaining_major):
                gaps.append(GapItem(
                    expected_topic=f"Unfilled gap {i}",
                    current_coverage="Still missing",
                    severity="major",
                    suggested_queries=[],
                ))
            return LLMStructuredResponse(
                parsed=GapAnalysisResult(gaps=gaps, coverage_score=self.post_score),
                input_tokens=400,
                output_tokens=200,
            )
        raise ValueError(f"Unexpected: {response_model}")


class TestPostGapRevalidator:
    async def test_gaps_filled_passes(self):
        llm = MockPostGapLLM(post_score=0.9, remaining_major=0)
        revalidator = PostGapRevalidator(llm)
        themes = [Theme(name="T1", description="D", paper_ids=["p1"])]
        pre_gaps = [
            IdentifiedGap(
                expected_topic="Topic A",
                current_coverage="None",
                severity=GapSeverity.MAJOR,
                suggested_queries=["query"],
            )
        ]
        result = await revalidator.check(themes, "scope", pre_gaps, pre_coverage=0.6)
        assert result.status == "passed"
        assert result.metrics["pre_coverage"] == 0.6
        assert result.metrics["post_coverage"] == 0.9

    async def test_remaining_major_gaps_warns(self):
        llm = MockPostGapLLM(post_score=0.65, remaining_major=2)
        revalidator = PostGapRevalidator(llm)
        themes = [Theme(name="T1", description="D", paper_ids=["p1"])]
        pre_gaps = [
            IdentifiedGap(
                expected_topic="Topic A",
                current_coverage="None",
                severity=GapSeverity.MAJOR,
                suggested_queries=[],
            )
        ]
        result = await revalidator.check(themes, "scope", pre_gaps, pre_coverage=0.6)
        assert result.status == "warning"
        assert result.metrics["remaining_major_gaps"] == 2

    async def test_no_pre_gaps_skips(self):
        llm = MockPostGapLLM()
        revalidator = PostGapRevalidator(llm)
        result = await revalidator.check([], "scope", pre_gaps=[], pre_coverage=0.9)
        assert result.status == "passed"
        assert result.details == "No prior gaps to revalidate"


from autoreview.analysis.comprehensiveness import BenchmarkValidator


class MockS2Client:
    """Mock Semantic Scholar client for benchmark validation."""

    def __init__(
        self,
        review_title: str = "A Comprehensive Review",
        review_citations: int = 500,
        reference_dois: list[str] | None = None,
    ):
        self.review_title = review_title
        self.review_citations = review_citations
        self.reference_dois = reference_dois if reference_dois is not None else ["10.1000/ref1", "10.1000/ref2", "10.1000/ref3"]

    async def find_review(self, topic: str) -> dict | None:
        return {
            "paperId": "review123",
            "title": self.review_title,
            "citationCount": self.review_citations,
            "externalIds": {"DOI": "10.1000/review"},
        }

    async def get_references(self, paper_id: str) -> list[str]:
        return self.reference_dois


class TestBenchmarkValidator:
    async def test_high_recall_passes(self):
        # Pipeline found 2 of 3 reference DOIs
        s2 = MockS2Client(reference_dois=["10.1000/ref1", "10.1000/ref2", "10.1000/ref3"])
        validator = BenchmarkValidator(s2_client=s2)
        pipeline_dois = {"10.1000/ref1", "10.1000/ref2", "10.1000/other"}
        result = await validator.check("test topic", pipeline_dois)
        assert result.status == "passed"
        assert result.metrics["recall"] == pytest.approx(2 / 3)

    async def test_low_recall_warns(self):
        s2 = MockS2Client(reference_dois=["10.1000/a", "10.1000/b", "10.1000/c", "10.1000/d"])
        validator = BenchmarkValidator(s2_client=s2)
        pipeline_dois = {"10.1000/a"}  # Only 1 of 4
        result = await validator.check("test topic", pipeline_dois)
        assert result.status == "warning"
        assert result.metrics["recall"] == pytest.approx(0.25)

    async def test_no_review_found_passes(self):
        class NoReviewClient:
            async def find_review(self, topic: str) -> dict | None:
                return None
            async def get_references(self, paper_id: str) -> list[str]:
                return []
        validator = BenchmarkValidator(s2_client=NoReviewClient())
        result = await validator.check("obscure topic", set())
        assert result.status == "passed"
        assert "no benchmark review" in result.details.lower()

    async def test_no_reference_dois_passes(self):
        s2 = MockS2Client(reference_dois=[])
        validator = BenchmarkValidator(s2_client=s2)
        result = await validator.check("topic", {"10.1000/a"})
        assert result.status == "passed"


class TestComprehensiveCheckIntegration:
    def test_knowledge_base_has_field(self):
        from autoreview.models.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(topic="test", domain="general", output_dir="/tmp/test")
        assert hasattr(kb, "comprehensiveness_checks")
        assert kb.comprehensiveness_checks == []

    def test_results_serialize(self):
        from autoreview.models.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(topic="test", domain="general", output_dir="/tmp/test")
        kb.comprehensiveness_checks.append(
            ComprehensiveCheckResult(
                check_name="test",
                status=CheckStatus.WARNING,
                score=0.7,
                details="Test warning",
                metrics={"key": "value"},
            )
        )
        data = kb.model_dump_json()
        restored = KnowledgeBase.model_validate_json(data)
        assert len(restored.comprehensiveness_checks) == 1
        assert restored.comprehensiveness_checks[0].check_name == "test"
        assert restored.comprehensiveness_checks[0].status == "warning"
