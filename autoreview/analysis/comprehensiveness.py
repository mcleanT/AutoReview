"""Comprehensiveness checks for literature search validation."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

import httpx
import structlog
from pydantic import Field

from autoreview.analysis.evidence_map import IdentifiedGap, Theme
from autoreview.analysis.gap_detector import GapDetector
from autoreview.extraction.extractor import PaperScreener
from autoreview.llm.prompts.comprehensiveness import (
    QUERY_COVERAGE_SYSTEM_PROMPT,
    QueryCoverageResult,
    build_query_coverage_prompt,
)
from autoreview.models.base import AutoReviewModel
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()


class CheckStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class RemediationAction(AutoReviewModel):
    """A prescribed remediation action for a failed check."""

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    priority: int = 1


class ComprehensiveCheckResult(AutoReviewModel):
    """Result of a single comprehensiveness check."""

    check_name: str
    status: CheckStatus
    score: float = 0.0
    details: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    remediation: RemediationAction | None = None


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
            warnings.append(
                f"Only {total_screened} papers passed screening (minimum: {self.min_screened})"
            )

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

        # Determine remediation action
        remediation = None
        if warnings:
            failed_sources = [
                src for src in (expected_sources or []) if source_counts.get(src, 0) == 0
            ]
            if rejection_rate > self.max_rejection_rate:
                remediation = RemediationAction(
                    action="lower_screening_threshold",
                    params={"current_rejection_rate": round(rejection_rate, 3)},
                )
            elif failed_sources:
                remediation = RemediationAction(
                    action="expand_queries",
                    params={"failed_sources": failed_sources},
                )

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
            remediation=remediation,
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

        remediation = None
        if uncovered:
            status = CheckStatus.WARNING
            details = f"Queries miss {len(uncovered)} sub-topic(s): {', '.join(uncovered)}"
            remediation = RemediationAction(
                action="expand_queries",
                params={"uncovered_topics": uncovered},
            )
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
            remediation=remediation,
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

        remediation = None
        if remaining_major:
            status = CheckStatus.WARNING
            unfilled = [g.expected_topic for g in remaining_major]
            details = (
                f"Coverage improved {pre_coverage:.2f} -> {post_coverage:.2f}, "
                f"but {len(remaining_major)} major gap(s) remain: {', '.join(unfilled)}"
            )
            remediation = RemediationAction(
                action="retry_gap_search",
                params={
                    "remaining_gaps": [
                        {
                            "expected_topic": g.expected_topic,
                            "current_coverage": g.current_coverage,
                            "severity": str(g.severity),
                        }
                        for g in remaining_major
                    ],
                },
            )
        else:
            status = CheckStatus.PASSED
            details = (
                f"All major gaps filled. Coverage improved "
                f"{pre_coverage:.2f} -> {post_coverage:.2f}"
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
            remediation=remediation,
        )


class SemanticScholarBenchmarkClient:
    """Queries Semantic Scholar to find a benchmark review and its references."""

    S2_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("S2_API_KEY")
        self._headers: dict[str, str] = {}
        if self._api_key:
            self._headers["x-api-key"] = self._api_key

    async def find_review(self, topic: str) -> dict[str, Any] | None:
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
            p
            for p in data
            if p.get("citationCount", 0) >= 50
            and any(t in (p.get("publicationTypes") or []) for t in ["Review", "Meta-Analysis"])
        ]
        if not reviews:
            # Fall back to highest-cited paper regardless of type
            reviews = [p for p in data if p.get("citationCount", 0) >= 100]

        if not reviews:
            return None

        reviews.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
        return dict(reviews[0])

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
                metrics={
                    "benchmark_found": True,
                    "benchmark_title": review_title,
                    "reference_count": 0,
                },
            )

        normalized_pipeline = {d.lower().strip() for d in pipeline_dois}
        matched = [d for d in reference_dois if d in normalized_pipeline]
        recall = len(matched) / len(reference_dois)

        if recall >= self.recall_threshold:
            status = CheckStatus.PASSED
            details = (
                f"Recall {recall:.0%} against benchmark '{review_title}' "
                f"({review_citations} citations): "
                f"{len(matched)}/{len(reference_dois)} references found"
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
                "recall": recall,
            },
        )
