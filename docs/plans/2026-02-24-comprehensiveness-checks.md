# Comprehensiveness Checks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 5 comprehensiveness checks to the AutoReview pipeline that validate literature search coverage at key checkpoints.

**Architecture:** A single `autoreview/analysis/comprehensiveness.py` module contains all 5 check classes. Each returns a `ComprehensiveCheckResult` that accumulates on `KnowledgeBase.comprehensiveness_checks`. Pipeline nodes call checks at natural integration points. All checks log via structlog and never halt the pipeline.

**Tech Stack:** Python 3.11+, Pydantic v2, structlog, httpx (for Semantic Scholar API in benchmark check), pytest + pytest-asyncio

---

### Task 1: Add ComprehensiveCheckResult model and KnowledgeBase field

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py` (create new)
- Modify: `autoreview/models/knowledge_base.py`
- Test: `tests/test_analysis/test_comprehensiveness.py` (create new)

**Step 1: Write the failing test**

In `tests/test_analysis/test_comprehensiveness.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestComprehensiveCheckResult -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Create `autoreview/analysis/comprehensiveness.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestComprehensiveCheckResult -v`
Expected: PASS

**Step 5: Add comprehensiveness_checks field to KnowledgeBase**

In `autoreview/models/knowledge_base.py`, add import and field:

```python
# Add to imports at top:
from autoreview.analysis.comprehensiveness import ComprehensiveCheckResult

# Add field to KnowledgeBase class, after critique_history:
comprehensiveness_checks: list[ComprehensiveCheckResult] = Field(default_factory=list)
```

**Step 6: Run all existing tests**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still PASS

**Step 7: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py autoreview/models/knowledge_base.py
git commit -m "feat: add ComprehensiveCheckResult model and KnowledgeBase field"
```

---

### Task 2: Implement Coverage Anomaly Detection check

This is a pure-computation check (no LLM). It runs after screening and examines result counts and rejection rates.

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
from autoreview.analysis.comprehensiveness import CoverageAnomalyChecker, CheckStatus
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestCoverageAnomalyChecker -v`
Expected: FAIL with ImportError for CoverageAnomalyChecker

**Step 3: Implement CoverageAnomalyChecker**

Add to `autoreview/analysis/comprehensiveness.py`:

```python
import structlog

from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()


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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestCoverageAnomalyChecker -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add coverage anomaly detection check"
```

---

### Task 3: Implement Query Coverage check

This check uses the LLM to verify generated queries cover all scope document sub-topics.

**Files:**
- Create: `autoreview/llm/prompts/comprehensiveness.py`
- Modify: `autoreview/analysis/comprehensiveness.py`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestQueryCoverageChecker -v`
Expected: FAIL with ImportError

**Step 3: Create the prompt module**

Create `autoreview/llm/prompts/comprehensiveness.py`:

```python
"""Prompts for comprehensiveness validation checks."""
from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class SubTopicCoverage(AutoReviewModel):
    """Assessment of whether a sub-topic is covered by queries."""
    sub_topic: str
    covered: bool
    matching_queries: list[str] = Field(default_factory=list)


class QueryCoverageResult(AutoReviewModel):
    """Result of query coverage assessment."""
    sub_topic_assessments: list[SubTopicCoverage]
    overall_coverage_score: float = 0.0


QUERY_COVERAGE_SYSTEM_PROMPT = """\
You are an expert research librarian evaluating whether a set of search queries \
adequately covers all expected sub-topics for a literature review. For each sub-topic \
identified in the scope document, determine whether at least one query would likely \
retrieve relevant papers on that sub-topic.
"""


def build_query_coverage_prompt(
    scope_document: str,
    queries_by_source: dict[str, list[str]],
) -> str:
    query_blocks = []
    for source, queries in queries_by_source.items():
        query_blocks.append(f"**{source}:**")
        for q in queries:
            query_blocks.append(f"  - {q}")
    queries_text = "\n".join(query_blocks)

    return f"""\
## Review Scope Document
{scope_document}

## Generated Search Queries
{queries_text}

Evaluate whether these queries cover all sub-topics expected by the scope document.
For each sub-topic in the scope:
1. Identify the sub-topic
2. Determine if at least one query would retrieve papers on it (covered=true/false)
3. List matching queries if covered

Provide an overall_coverage_score from 0.0 (no sub-topics covered) to 1.0 (all covered).
"""
```

**Step 4: Implement QueryCoverageChecker**

Add to `autoreview/analysis/comprehensiveness.py`:

```python
from autoreview.llm.prompts.comprehensiveness import (
    QUERY_COVERAGE_SYSTEM_PROMPT,
    QueryCoverageResult,
    build_query_coverage_prompt,
)


class QueryCoverageChecker:
    """Validates that search queries cover all scope document sub-topics."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def check(
        self,
        queries_by_source: dict[str, list[str]],
        scope_document: str,
    ) -> ComprehensiveCheckResult:
        prompt = build_query_coverage_prompt(scope_document, queries_by_source)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=QueryCoverageResult,
            system=QUERY_COVERAGE_SYSTEM_PROMPT,
        )
        result: QueryCoverageResult = response.parsed

        uncovered = [a.sub_topic for a in result.sub_topic_assessments if not a.covered]
        total = len(result.sub_topic_assessments)
        covered = total - len(uncovered)

        if uncovered:
            status = CheckStatus.WARNING
            details = f"Queries miss {len(uncovered)} sub-topic(s): {', '.join(uncovered)}"
        else:
            status = CheckStatus.PASSED
            details = f"All {total} sub-topics covered by queries"

        score = max(0.0, min(1.0, result.overall_coverage_score))

        logger.info(
            "comprehensiveness.query_coverage",
            status=status,
            covered=covered,
            total=total,
            uncovered=uncovered,
        )

        return ComprehensiveCheckResult(
            check_name="query_coverage",
            status=status,
            score=score,
            details=details,
            metrics={
                "sub_topics_covered": covered,
                "sub_topics_total": total,
                "uncovered_topics": uncovered,
            },
        )
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestQueryCoverageChecker -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add autoreview/llm/prompts/comprehensiveness.py autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add query coverage check with LLM validation"
```

---

### Task 4: Implement Borderline Re-screening check

This modifies `PaperScreener.screen()` to also return borderline papers (score=2), then re-screens them.

**Files:**
- Modify: `autoreview/extraction/extractor.py`
- Modify: `autoreview/analysis/comprehensiveness.py`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestBorderlineRescreener -v`
Expected: FAIL with ImportError

**Step 3: Modify PaperScreener to capture borderline rejects**

In `autoreview/extraction/extractor.py`, modify `PaperScreener.screen()` to add a `borderline_papers` attribute. Change the method to track papers with score == (threshold - 1):

```python
# In PaperScreener.screen(), after the existing loop over decisions,
# change to also track borderline papers:

async def screen(
    self,
    papers: list[CandidatePaper],
    scope_document: str,
    threshold: int = 3,
) -> list[ScreenedPaper]:
    all_screened: list[ScreenedPaper] = []
    self.borderline_papers: list[CandidatePaper] = []

    for batch_start in range(0, len(papers), self.batch_size):
        batch = papers[batch_start:batch_start + self.batch_size]
        batch_papers = [
            {"index": str(batch_start + i), "title": p.title, "abstract": p.abstract or ""}
            for i, p in enumerate(batch)
        ]

        prompt = build_screening_prompt(scope_document, batch_papers)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=ScreeningBatchResult,
            system=SCREENING_SYSTEM_PROMPT,
        )
        result: ScreeningBatchResult = response.parsed

        logger.info(
            "screening.batch",
            batch_start=batch_start,
            batch_size=len(batch),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        for decision in result.decisions:
            paper_idx = decision.paper_index - batch_start
            if 0 <= paper_idx < len(batch):
                paper = batch[paper_idx]
                score = max(1, min(5, decision.relevance_score))
                screened = ScreenedPaper(
                    paper=paper,
                    relevance_score=score,
                    rationale=decision.rationale,
                    include=score >= threshold,
                )
                if screened.include:
                    all_screened.append(screened)
                elif score == threshold - 1:
                    self.borderline_papers.append(paper)

    logger.info(
        "screening.complete",
        total_papers=len(papers),
        included=len(all_screened),
        borderline=len(self.borderline_papers),
    )
    return all_screened
```

**Step 4: Implement BorderlineRescreener**

Add to `autoreview/analysis/comprehensiveness.py`:

```python
from autoreview.extraction.extractor import PaperScreener
from autoreview.llm.prompts.screening import (
    SCREENING_SYSTEM_PROMPT,
    ScreeningBatchResult,
    build_screening_prompt,
)


class BorderlineRescreener:
    """Re-screens borderline papers (score = threshold - 1) with enriched context."""

    def __init__(self, llm: Any, batch_size: int = 20) -> None:
        self.llm = llm
        self.batch_size = batch_size

    async def rescreen(
        self,
        borderline_papers: list[CandidatePaper],
        scope_document: str,
        threshold: int = 3,
    ) -> tuple[ComprehensiveCheckResult, list[ScreenedPaper]]:
        if not borderline_papers:
            return ComprehensiveCheckResult(
                check_name="borderline_rescreening",
                status=CheckStatus.PASSED,
                score=1.0,
                details="No borderline papers to re-screen",
                metrics={"borderline_count": 0, "promoted_count": 0},
            ), []

        enriched_scope = (
            f"{scope_document}\n\n"
            "IMPORTANT: These papers were previously scored as borderline relevant. "
            "Re-evaluate carefully — consider whether the paper could contribute to "
            "any sub-topic of this review, even tangentially. Be inclusive rather than "
            "exclusive on this second pass."
        )

        screener = PaperScreener(self.llm, batch_size=self.batch_size)
        promoted = await screener.screen(
            borderline_papers,
            scope_document=enriched_scope,
            threshold=threshold,
        )

        promoted_count = len(promoted)
        total = len(borderline_papers)

        logger.info(
            "comprehensiveness.borderline_rescreening",
            borderline_count=total,
            promoted=promoted_count,
        )

        return ComprehensiveCheckResult(
            check_name="borderline_rescreening",
            status=CheckStatus.PASSED if promoted_count == 0 else CheckStatus.WARNING,
            score=1.0 - (promoted_count / total) if total > 0 else 1.0,
            details=f"Re-screened {total} borderline papers, promoted {promoted_count}",
            metrics={"borderline_count": total, "promoted_count": promoted_count},
        ), promoted
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestBorderlineRescreener -v`
Expected: All PASS

**Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All PASS (including existing PaperScreener tests — the new `borderline_papers` attribute is additive)

**Step 7: Commit**

```bash
git add autoreview/extraction/extractor.py autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add borderline re-screening check"
```

---

### Task 5: Implement Post-Gap Re-validation check

This re-runs gap detection after the gap search and compares coverage scores.

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestPostGapRevalidator -v`
Expected: FAIL with ImportError

**Step 3: Implement PostGapRevalidator**

Add to `autoreview/analysis/comprehensiveness.py`:

```python
from autoreview.analysis.evidence_map import IdentifiedGap, Theme
from autoreview.analysis.gap_detector import GapDetector


class PostGapRevalidator:
    """Re-runs gap detection after supplementary search to verify gaps were filled."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def check(
        self,
        themes: list[Theme],
        scope_document: str,
        pre_gaps: list[IdentifiedGap],
        pre_coverage: float,
    ) -> ComprehensiveCheckResult:
        if not pre_gaps:
            return ComprehensiveCheckResult(
                check_name="post_gap_revalidation",
                status=CheckStatus.PASSED,
                score=1.0,
                details="No prior gaps to revalidate",
                metrics={"pre_coverage": pre_coverage, "post_coverage": pre_coverage},
            )

        detector = GapDetector(self.llm)
        post_gaps, post_coverage = await detector.detect_gaps(themes, scope_document)

        remaining_major = [g for g in post_gaps if g.severity == "major"]
        pre_major = [g for g in pre_gaps if g.severity == "major"]

        if remaining_major:
            status = CheckStatus.WARNING
            unfilled = [g.expected_topic for g in remaining_major]
            details = (
                f"Coverage improved {pre_coverage:.2f} -> {post_coverage:.2f}, "
                f"but {len(remaining_major)} major gap(s) remain: {', '.join(unfilled)}"
            )
        else:
            status = CheckStatus.PASSED
            details = (
                f"All major gaps filled. Coverage improved {pre_coverage:.2f} -> {post_coverage:.2f}"
            )

        logger.info(
            "comprehensiveness.post_gap_revalidation",
            status=status,
            pre_coverage=pre_coverage,
            post_coverage=post_coverage,
            pre_major_gaps=len(pre_major),
            remaining_major_gaps=len(remaining_major),
        )

        return ComprehensiveCheckResult(
            check_name="post_gap_revalidation",
            status=status,
            score=post_coverage,
            details=details,
            metrics={
                "pre_coverage": pre_coverage,
                "post_coverage": post_coverage,
                "pre_major_gaps": len(pre_major),
                "remaining_major_gaps": len(remaining_major),
                "remaining_gap_topics": [g.expected_topic for g in remaining_major],
            },
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestPostGapRevalidator -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add post-gap re-validation check"
```

---

### Task 6: Implement Benchmark Validation check

This autonomously finds a highly-cited review via Semantic Scholar and measures recall.

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
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
        self.reference_dois = reference_dois or ["10.1000/ref1", "10.1000/ref2", "10.1000/ref3"]

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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestBenchmarkValidator -v`
Expected: FAIL with ImportError

**Step 3: Implement SemanticScholarBenchmarkClient and BenchmarkValidator**

Add to `autoreview/analysis/comprehensiveness.py`:

```python
import httpx


class SemanticScholarBenchmarkClient:
    """Queries Semantic Scholar to find a benchmark review and its references."""

    S2_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None) -> None:
        import os
        self._api_key = api_key or os.environ.get("S2_API_KEY")
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["x-api-key"] = self._api_key

    async def find_review(self, topic: str) -> dict | None:
        """Find the most highly-cited review paper on a topic."""
        query = f"{topic} review"
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            try:
                resp = await client.get(
                    f"{self.S2_API}/paper/search",
                    params={
                        "query": query,
                        "limit": 10,
                        "fields": "paperId,title,citationCount,externalIds,publicationTypes",
                    },
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("benchmark.s2_search_failed", error=str(e))
                return None

        # Filter for review papers, sort by citations
        reviews = [
            p for p in data
            if p.get("citationCount", 0) >= 50
            and any(
                t in (p.get("publicationTypes") or [])
                for t in ["Review", "Meta-Analysis"]
            )
        ]
        if not reviews:
            # Fall back to highest-cited paper regardless of type
            reviews = [p for p in data if p.get("citationCount", 0) >= 100]

        if not reviews:
            return None

        reviews.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
        return reviews[0]

    async def get_references(self, paper_id: str) -> list[str]:
        """Get DOIs of all references of a paper."""
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            try:
                resp = await client.get(
                    f"{self.S2_API}/paper/{paper_id}/references",
                    params={"fields": "externalIds", "limit": 1000},
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("benchmark.s2_references_failed", error=str(e))
                return []

        dois = []
        for ref in data:
            cited = ref.get("citedPaper", {})
            ext = cited.get("externalIds") or {}
            doi = ext.get("DOI")
            if doi:
                dois.append(doi.lower().strip())
        return dois


class BenchmarkValidator:
    """Validates pipeline recall against a benchmark review's references."""

    def __init__(
        self,
        s2_client: Any | None = None,
        recall_threshold: float = 0.5,
    ) -> None:
        self.s2_client = s2_client or SemanticScholarBenchmarkClient()
        self.recall_threshold = recall_threshold

    async def check(
        self,
        topic: str,
        pipeline_dois: set[str],
    ) -> ComprehensiveCheckResult:
        review = await self.s2_client.find_review(topic)
        if not review:
            logger.info("benchmark.no_review_found", topic=topic)
            return ComprehensiveCheckResult(
                check_name="benchmark_validation",
                status=CheckStatus.PASSED,
                score=1.0,
                details="No benchmark review found for this topic; skipping validation",
                metrics={"benchmark_found": False},
            )

        paper_id = review["paperId"]
        review_title = review.get("title", "Unknown")
        review_citations = review.get("citationCount", 0)

        reference_dois = await self.s2_client.get_references(paper_id)
        if not reference_dois:
            logger.info("benchmark.no_reference_dois", review_title=review_title)
            return ComprehensiveCheckResult(
                check_name="benchmark_validation",
                status=CheckStatus.PASSED,
                score=1.0,
                details=f"Benchmark review '{review_title}' has no extractable reference DOIs",
                metrics={"benchmark_found": True, "benchmark_title": review_title, "reference_count": 0},
            )

        normalized_pipeline = {d.lower().strip() for d in pipeline_dois}
        matched = [d for d in reference_dois if d in normalized_pipeline]
        recall = len(matched) / len(reference_dois)

        if recall >= self.recall_threshold:
            status = CheckStatus.PASSED
            details = (
                f"Recall {recall:.0%} against benchmark '{review_title}' "
                f"({review_citations} citations): {len(matched)}/{len(reference_dois)} references found"
            )
        else:
            status = CheckStatus.WARNING
            missing = [d for d in reference_dois if d not in normalized_pipeline]
            details = (
                f"Low recall {recall:.0%} against benchmark '{review_title}' "
                f"({review_citations} citations): {len(matched)}/{len(reference_dois)} found, "
                f"{len(missing)} missing"
            )

        logger.info(
            "comprehensiveness.benchmark_validation",
            status=status,
            recall=round(recall, 3),
            benchmark_title=review_title,
            matched=len(matched),
            total_refs=len(reference_dois),
        )

        return ComprehensiveCheckResult(
            check_name="benchmark_validation",
            status=status,
            score=recall,
            details=details,
            metrics={
                "benchmark_found": True,
                "benchmark_title": review_title,
                "benchmark_citations": review_citations,
                "reference_count": len(reference_dois),
                "matched_count": len(matched),
                "recall": round(recall, 3),
            },
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestBenchmarkValidator -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add benchmark validation check via Semantic Scholar"
```

---

### Task 7: Integrate all checks into pipeline nodes

Wire the 5 checks into `PipelineNodes` at their natural integration points.

**Files:**
- Modify: `autoreview/pipeline/nodes.py`
- Test: `tests/test_pipeline/test_runner.py` (verify pipeline still builds)

**Step 1: Modify query_expansion node**

In `autoreview/pipeline/nodes.py`, add the query coverage check at the end of `query_expansion()`:

```python
# At top of nodes.py, add import:
from autoreview.analysis.comprehensiveness import (
    BenchmarkValidator,
    BorderlineRescreener,
    CoverageAnomalyChecker,
    PostGapRevalidator,
    QueryCoverageChecker,
)

# At the end of query_expansion(), after the audit entry:
        # Comprehensiveness: query coverage check
        query_checker = QueryCoverageChecker(self.llm)
        qc_result = await query_checker.check(kb.search_queries, kb.scope_document or "")
        kb.comprehensiveness_checks.append(qc_result)
```

**Step 2: Modify screening node**

In the `screening()` method, add coverage anomaly check and borderline re-screening:

```python
    async def screening(self, kb: KnowledgeBase) -> None:
        """Node: Screen papers for relevance."""
        screener = PaperScreener(self.llm, batch_size=self.config.search.screening_batch_size)
        kb.screened_papers = await screener.screen(
            kb.candidate_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )
        kb.current_phase = PipelinePhase.SCREENING
        kb.add_audit_entry("screening", "complete", f"Screened to {len(kb.screened_papers)} papers")

        # Comprehensiveness: coverage anomaly check
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        anomaly_checker = CoverageAnomalyChecker()
        anomaly_result = anomaly_checker.check(
            kb.candidate_papers, kb.screened_papers, expected_sources=all_dbs,
        )
        kb.comprehensiveness_checks.append(anomaly_result)

        # Comprehensiveness: borderline re-screening
        rescreener = BorderlineRescreener(self.llm)
        rescreen_result, promoted = await rescreener.rescreen(
            screener.borderline_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )
        kb.comprehensiveness_checks.append(rescreen_result)
        if promoted:
            kb.screened_papers.extend(promoted)
            kb.add_audit_entry(
                "screening", "borderline_promoted",
                f"Promoted {len(promoted)} borderline papers",
            )
```

**Step 3: Modify gap_search node**

In the `gap_search()` method, add post-gap re-validation and benchmark validation at the end:

```python
    async def gap_search(self, kb: KnowledgeBase) -> None:
        """Node: Gap-aware supplementary search (conditional)."""
        # Store pre-gap state for revalidation
        pre_gaps = list(kb.evidence_map.gaps) if kb.evidence_map and kb.evidence_map.gaps else []
        pre_coverage = kb.evidence_map.coverage_score if kb.evidence_map else 0.0

        if not kb.evidence_map or not kb.evidence_map.gaps:
            # Still run benchmark validation even if no gaps
            await self._run_benchmark_validation(kb)
            return

        major_gaps = [g for g in kb.evidence_map.gaps if g.severity == "major"]
        if not major_gaps:
            await self._run_benchmark_validation(kb)
            return

        logger.info("gap_search.triggered", major_gaps=len(major_gaps))

        # ... (existing gap search logic unchanged) ...

        # Comprehensiveness: post-gap re-validation
        revalidator = PostGapRevalidator(self.llm)
        reval_result = await revalidator.check(
            kb.evidence_map.themes if kb.evidence_map else [],
            kb.scope_document or "",
            pre_gaps=pre_gaps,
            pre_coverage=pre_coverage,
        )
        kb.comprehensiveness_checks.append(reval_result)

        # Update evidence map with new coverage info
        if kb.evidence_map and reval_result.metrics.get("post_coverage"):
            kb.evidence_map.coverage_score = reval_result.metrics["post_coverage"]

        await self._run_benchmark_validation(kb)

# Add new private helper method to PipelineNodes:
    async def _run_benchmark_validation(self, kb: KnowledgeBase) -> None:
        """Run benchmark validation check."""
        pipeline_dois = set()
        for p in kb.candidate_papers:
            if p.doi:
                pipeline_dois.add(p.doi.lower().strip())
        for sp in kb.screened_papers:
            if sp.paper.doi:
                pipeline_dois.add(sp.paper.doi.lower().strip())

        validator = BenchmarkValidator()
        bench_result = await validator.check(kb.topic, pipeline_dois)
        kb.comprehensiveness_checks.append(bench_result)
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add autoreview/pipeline/nodes.py
git commit -m "feat: integrate comprehensiveness checks into pipeline nodes"
```

---

### Task 8: Final integration test and cleanup

Write a test that verifies the full pipeline wiring includes comprehensiveness checks.

**Files:**
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write integration test**

Append to `tests/test_analysis/test_comprehensiveness.py`:

```python
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
```

**Step 2: Run integration test**

Run: `python -m pytest tests/test_analysis/test_comprehensiveness.py::TestComprehensiveCheckIntegration -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_analysis/test_comprehensiveness.py
git commit -m "test: add integration tests for comprehensiveness checks"
```
