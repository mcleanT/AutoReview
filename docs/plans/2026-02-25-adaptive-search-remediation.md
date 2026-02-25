# Adaptive Search Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make comprehensiveness checks actionable — when checks detect problems, the pipeline auto-remediates by expanding queries, retrying gap search, or adjusting screening thresholds.

**Architecture:** Each `ComprehensiveCheckResult` gains a `remediation` field describing what action to take. A `RemediationDispatcher` reads the field and executes the prescribed action (expand queries, retry gaps, lower threshold). Iteration caps prevent runaway cost.

**Tech Stack:** Pydantic v2 models, async Python, structlog, existing `SearchAggregator`/`PaperScreener`/`GapDetector` infrastructure.

---

### Task 1: Add RemediationAction Model to ComprehensiveCheckResult

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py:32-39`
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing test**

Add to `tests/test_analysis/test_comprehensiveness.py`:

```python
from autoreview.analysis.comprehensiveness import RemediationAction


class TestRemediationAction:
    def test_create_remediation_action(self):
        action = RemediationAction(
            action="expand_queries",
            params={"uncovered_topics": ["neuroinflammation"]},
            priority=2,
        )
        assert action.action == "expand_queries"
        assert action.params["uncovered_topics"] == ["neuroinflammation"]
        assert action.priority == 2

    def test_default_priority(self):
        action = RemediationAction(action="retry_gap_search", params={})
        assert action.priority == 1

    def test_check_result_with_remediation(self):
        result = ComprehensiveCheckResult(
            check_name="query_coverage",
            status=CheckStatus.WARNING,
            score=0.6,
            details="Missing topics",
            metrics={},
            remediation=RemediationAction(
                action="expand_queries",
                params={"uncovered_topics": ["topic_a"]},
            ),
        )
        assert result.remediation is not None
        assert result.remediation.action == "expand_queries"

    def test_check_result_without_remediation(self):
        result = ComprehensiveCheckResult(
            check_name="query_coverage",
            status=CheckStatus.PASSED,
            score=1.0,
            details="All good",
            metrics={},
        )
        assert result.remediation is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_comprehensiveness.py::TestRemediationAction -v`
Expected: FAIL — `RemediationAction` not importable

**Step 3: Write minimal implementation**

In `autoreview/analysis/comprehensiveness.py`, add `RemediationAction` model after `CheckStatus` (line 30), before `ComprehensiveCheckResult`:

```python
class RemediationAction(AutoReviewModel):
    """A prescribed remediation action for a failed check."""

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    priority: int = 1
```

Add `remediation` field to `ComprehensiveCheckResult` (line 39):

```python
class ComprehensiveCheckResult(AutoReviewModel):
    """Result of a single comprehensiveness check."""

    check_name: str
    status: CheckStatus
    score: float = 0.0
    details: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    remediation: RemediationAction | None = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_comprehensiveness.py::TestRemediationAction -v`
Expected: PASS

**Step 5: Verify no regressions**

Run: `pytest tests/test_analysis/test_comprehensiveness.py -v`
Expected: All existing tests still pass (remediation defaults to `None`)

**Step 6: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: add RemediationAction model to ComprehensiveCheckResult"
```

---

### Task 2: Add Remediation Config Fields

**Files:**
- Modify: `autoreview/config/models.py:7-14`
- Modify: `autoreview/config/defaults/biomedical.yaml`
- Modify: `autoreview/config/defaults/cs_ai.yaml`
- Modify: `autoreview/config/defaults/chemistry.yaml`
- Test: `tests/test_pipeline/test_runner.py` (existing tests still pass)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_runner.py`:

```python
class TestSearchConfig:
    def test_remediation_config_defaults(self):
        from autoreview.config.models import SearchConfig
        config = SearchConfig()
        assert config.max_query_expansion_rounds == 2
        assert config.max_gap_search_rounds == 2
        assert config.min_coverage_threshold == 0.75
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_runner.py::TestSearchConfig -v`
Expected: FAIL — `SearchConfig` has no attribute `max_query_expansion_rounds`

**Step 3: Write minimal implementation**

In `autoreview/config/models.py`, add three fields to `SearchConfig` (after line 14):

```python
class SearchConfig(BaseModel):
    """Configuration for literature search."""
    model_config = ConfigDict(extra="forbid")

    date_range: str = "2015-2025"
    max_results_per_source: int = 500
    relevance_threshold: int = 3
    screening_batch_size: int = 20
    max_query_expansion_rounds: int = 2
    max_gap_search_rounds: int = 2
    min_coverage_threshold: float = 0.75
```

Add the new fields to each domain YAML. Append to `biomedical.yaml` search section (after line 14):

```yaml
search:
  date_range: "2015-2025"
  max_results_per_source: 500
  relevance_threshold: 3
  screening_batch_size: 20
  max_query_expansion_rounds: 2
  max_gap_search_rounds: 2
  min_coverage_threshold: 0.75
```

Same additions to `cs_ai.yaml` search section (after line 12) and `chemistry.yaml` search section (after line 12).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_runner.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add autoreview/config/models.py autoreview/config/defaults/biomedical.yaml autoreview/config/defaults/cs_ai.yaml autoreview/config/defaults/chemistry.yaml tests/test_pipeline/test_runner.py
git commit -m "feat: add remediation config fields to SearchConfig"
```

---

### Task 3: Add Targeted Query Expansion Prompt

**Files:**
- Modify: `autoreview/llm/prompts/query_expansion.py`
- Test: `tests/test_llm/test_query_expansion_prompts.py`

**Step 1: Write the failing test**

Create `tests/test_llm/test_query_expansion_prompts.py`:

```python
"""Tests for query expansion prompt builders."""
from __future__ import annotations

from autoreview.llm.prompts.query_expansion import build_targeted_query_expansion_prompt


class TestTargetedQueryExpansionPrompt:
    def test_includes_uncovered_topics(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["neuroinflammation", "blood-brain barrier"],
            domain="biomedical",
            date_range="2015-2025",
        )
        assert "neuroinflammation" in prompt
        assert "blood-brain barrier" in prompt

    def test_includes_domain_and_date(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["topic_a"],
            domain="cs_ai",
            date_range="2018-2025",
        )
        assert "cs_ai" in prompt
        assert "2018-2025" in prompt

    def test_returns_nonempty_string(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["topic"],
            domain="biomedical",
        )
        assert len(prompt) > 100
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_query_expansion_prompts.py -v`
Expected: FAIL — `build_targeted_query_expansion_prompt` not importable

**Step 3: Write minimal implementation**

Add to `autoreview/llm/prompts/query_expansion.py` (after line 31):

```python
TARGETED_EXPANSION_SYSTEM_PROMPT = """\
You are an expert research librarian. You previously generated search queries for a \
literature review, but some sub-topics were not adequately covered. Generate additional \
targeted queries to fill these specific gaps. Use different terminology, synonyms, and \
broader formulations than your initial queries.
"""


def build_targeted_query_expansion_prompt(
    uncovered_topics: list[str],
    domain: str,
    date_range: str = "2015-2025",
) -> str:
    """Build a prompt to generate queries for specific uncovered sub-topics."""
    topics_list = "\n".join(f"- {t}" for t in uncovered_topics)
    return f"""\
The following sub-topics were NOT adequately covered by the initial search queries \
and need additional targeted queries:

**Uncovered Sub-Topics:**
{topics_list}

**Domain:** {domain}
**Date range:** {date_range}

For EACH uncovered sub-topic, generate:
- pubmed_queries: 1-2 Boolean/MeSH queries targeting this specific sub-topic
- semantic_scholar_queries: 1-2 semantic search queries
- openalex_queries: 1-2 general academic queries

Use different terminology than typical queries for these topics — try synonyms, \
related concepts, and broader formulations to maximize discovery.
"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm/test_query_expansion_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add autoreview/llm/prompts/query_expansion.py tests/test_llm/test_query_expansion_prompts.py
git commit -m "feat: add targeted query expansion prompt for uncovered sub-topics"
```

---

### Task 4: Add Retry Gap Queries Prompt

**Files:**
- Modify: `autoreview/llm/prompts/clustering.py`
- Test: `tests/test_llm/test_clustering_prompts.py`

**Step 1: Write the failing test**

Create `tests/test_llm/test_clustering_prompts.py`:

```python
"""Tests for clustering prompt builders."""
from __future__ import annotations

from autoreview.llm.prompts.clustering import build_retry_gap_queries_prompt


class TestRetryGapQueriesPrompt:
    def test_includes_remaining_gaps(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[
                {"expected_topic": "Drug resistance mechanisms", "current_coverage": "No papers found"},
            ],
            previous_queries=["drug resistance", "antimicrobial resistance"],
        )
        assert "Drug resistance mechanisms" in prompt
        assert "No papers found" in prompt

    def test_includes_previous_queries(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[{"expected_topic": "Topic A", "current_coverage": "Sparse"}],
            previous_queries=["query_1", "query_2"],
        )
        assert "query_1" in prompt
        assert "query_2" in prompt

    def test_instructs_alternative_terminology(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[{"expected_topic": "X", "current_coverage": "None"}],
            previous_queries=["old query"],
        )
        # Should instruct to use different terminology
        assert "alternative" in prompt.lower() or "different" in prompt.lower() or "synonym" in prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_clustering_prompts.py -v`
Expected: FAIL — `build_retry_gap_queries_prompt` not importable

**Step 3: Write minimal implementation**

Add to `autoreview/llm/prompts/clustering.py` (after line 141):

```python
RETRY_GAP_SEARCH_SYSTEM_PROMPT = """\
You are an expert research librarian. A literature review pipeline identified gaps in \
evidence coverage and searched for papers to fill them, but some gaps remain unfilled. \
The previous queries did not return sufficient results. Generate NEW queries using \
alternative terminology, synonyms, broader formulations, and related concepts.
"""


def build_retry_gap_queries_prompt(
    remaining_gaps: list[dict],
    previous_queries: list[str],
) -> str:
    """Build a prompt to generate retry queries for unfilled gaps."""
    gaps_text = []
    for gap in remaining_gaps:
        topic = gap.get("expected_topic", "Unknown")
        coverage = gap.get("current_coverage", "None")
        gaps_text.append(f"- **{topic}**: {coverage}")
    gaps_block = "\n".join(gaps_text)

    prev_block = "\n".join(f"- {q}" for q in previous_queries)

    return f"""\
The following gaps in evidence coverage remain unfilled after an initial search attempt:

**Remaining Gaps:**
{gaps_block}

**Previous Queries That Were Insufficient:**
{prev_block}

Generate NEW search queries for each remaining gap. You MUST use different and alternative \
terminology from the previous queries listed above — try synonyms, related concepts, \
broader or narrower formulations, and cross-disciplinary terms. For each gap, produce \
3-5 queries suitable for academic search engines.
"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm/test_clustering_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add autoreview/llm/prompts/clustering.py tests/test_llm/test_clustering_prompts.py
git commit -m "feat: add retry gap queries prompt for unfilled gaps"
```

---

### Task 5: Make Checks Populate Remediation Fields

**Files:**
- Modify: `autoreview/analysis/comprehensiveness.py:126-176` (QueryCoverageChecker)
- Modify: `autoreview/analysis/comprehensiveness.py:42-123` (CoverageAnomalyChecker)
- Modify: `autoreview/analysis/comprehensiveness.py:234-296` (PostGapRevalidator)
- Test: `tests/test_analysis/test_comprehensiveness.py`

**Step 1: Write the failing tests**

Add to `tests/test_analysis/test_comprehensiveness.py`:

```python
class TestChecksPopulateRemediation:
    async def test_query_coverage_warning_has_remediation(self):
        llm = MockQueryCoverageLLM(uncovered=["Viral metagenomics"])
        checker = QueryCoverageChecker(llm)
        result = await checker.check({"pubmed": ["q1"]}, "scope")
        assert result.remediation is not None
        assert result.remediation.action == "expand_queries"
        assert "Viral metagenomics" in result.remediation.params["uncovered_topics"]

    async def test_query_coverage_pass_no_remediation(self):
        llm = MockQueryCoverageLLM(uncovered=[])
        checker = QueryCoverageChecker(llm)
        result = await checker.check({"pubmed": ["q1"]}, "scope")
        assert result.remediation is None

    def test_anomaly_high_rejection_has_remediation(self):
        candidates = _make_candidates(100)
        screened = _make_screened(candidates[:10])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.remediation is not None
        assert result.remediation.action == "lower_screening_threshold"

    def test_anomaly_zero_source_has_remediation(self):
        candidates = _make_candidates(50, "pubmed")
        screened = _make_screened(candidates[:30])
        checker = CoverageAnomalyChecker()
        result = checker.check(
            candidates, screened,
            expected_sources=["pubmed", "semantic_scholar"],
        )
        assert result.remediation is not None
        assert result.remediation.action == "expand_queries"
        assert "semantic_scholar" in result.remediation.params.get("failed_sources", [])

    def test_anomaly_pass_no_remediation(self):
        candidates = _make_candidates(100, "pubmed") + _make_candidates(100, "semantic_scholar")
        screened = _make_screened(candidates[:50])
        checker = CoverageAnomalyChecker()
        result = checker.check(candidates, screened)
        assert result.remediation is None

    async def test_post_gap_remaining_gaps_has_remediation(self):
        llm = MockPostGapLLM(post_score=0.65, remaining_major=2)
        revalidator = PostGapRevalidator(llm)
        themes = [Theme(name="T1", description="D", paper_ids=["p1"])]
        pre_gaps = [
            IdentifiedGap(
                expected_topic="Topic A",
                current_coverage="None",
                severity=GapSeverity.MAJOR,
                suggested_queries=["old query"],
            )
        ]
        result = await revalidator.check(themes, "scope", pre_gaps, pre_coverage=0.6)
        assert result.remediation is not None
        assert result.remediation.action == "retry_gap_search"
        assert len(result.remediation.params["remaining_gaps"]) == 2

    async def test_post_gap_filled_no_remediation(self):
        llm = MockPostGapLLM(post_score=0.9, remaining_major=0)
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
        assert result.remediation is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_comprehensiveness.py::TestChecksPopulateRemediation -v`
Expected: FAIL — remediation is `None` on warning results

**Step 3: Write implementation**

**QueryCoverageChecker** — in `comprehensiveness.py`, modify the return at line 166-176. When `uncovered` is non-empty, add remediation:

```python
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

        # ... (logging unchanged)

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
```

**CoverageAnomalyChecker** — modify the return at line 111-123. Determine the most important remediation action based on which warnings fired:

```python
        # Determine remediation action
        remediation = None
        if warnings:
            # Check for specific actionable conditions
            failed_sources = [
                src for src in (expected_sources or [])
                if source_counts.get(src, 0) == 0
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

        status = CheckStatus.WARNING if warnings else CheckStatus.PASSED
        # ... rest unchanged, but add remediation=remediation to the return
```

**PostGapRevalidator** — modify the return at line 284-296. When remaining major gaps exist, add remediation:

```python
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
                f"All major gaps filled. Coverage improved {pre_coverage:.2f} -> {post_coverage:.2f}"
            )

        # ... logging unchanged, add remediation=remediation to return
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_comprehensiveness.py -v`
Expected: All tests PASS (new + existing)

**Step 5: Commit**

```bash
git add autoreview/analysis/comprehensiveness.py tests/test_analysis/test_comprehensiveness.py
git commit -m "feat: checks populate remediation fields on WARNING status"
```

---

### Task 6: Create RemediationDispatcher

**Files:**
- Create: `autoreview/pipeline/remediation.py`
- Test: `tests/test_pipeline/test_remediation.py`

**Step 1: Write the failing tests**

Create `tests/test_pipeline/test_remediation.py`:

```python
"""Tests for remediation dispatcher."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoreview.analysis.comprehensiveness import (
    CheckStatus,
    ComprehensiveCheckResult,
    RemediationAction,
)
from autoreview.config import load_config
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.pipeline.remediation import RemediationDispatcher


class TestRemediationDispatcher:
    def _make_kb(self) -> KnowledgeBase:
        kb = KnowledgeBase(topic="test topic", domain="biomedical", output_dir="/tmp/test")
        kb.search_queries = {"pubmed": ["q1"], "semantic_scholar": ["q2"]}
        kb.scope_document = "Review covers topic A and topic B."
        return kb

    def _make_check(
        self, action: str, params: dict | None = None, status: str = "warning",
    ) -> ComprehensiveCheckResult:
        return ComprehensiveCheckResult(
            check_name="test_check",
            status=CheckStatus(status),
            score=0.5,
            details="Test",
            metrics={},
            remediation=RemediationAction(action=action, params=params or {}),
        )

    @pytest.mark.asyncio
    async def test_passed_check_no_action(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        result = ComprehensiveCheckResult(
            check_name="test",
            status=CheckStatus.PASSED,
            score=1.0,
            details="All good",
            metrics={},
        )
        executed = await dispatcher.execute(kb, result)
        assert executed is False

    @pytest.mark.asyncio
    async def test_no_remediation_no_action(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        result = ComprehensiveCheckResult(
            check_name="test",
            status=CheckStatus.WARNING,
            score=0.5,
            details="Warning but no action",
            metrics={},
            remediation=None,
        )
        executed = await dispatcher.execute(kb, result)
        assert executed is False

    @pytest.mark.asyncio
    async def test_expand_queries_respects_max_rounds(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check(
            "expand_queries",
            params={"uncovered_topics": ["topic_x"]},
        )

        # Exhaust rounds
        dispatcher._round_counts["expand_queries"] = config.search.max_query_expansion_rounds
        executed = await dispatcher.execute(kb, check)
        assert executed is False

    @pytest.mark.asyncio
    async def test_unknown_action_returns_false(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check("unknown_action", params={})
        executed = await dispatcher.execute(kb, check)
        assert executed is False

    @pytest.mark.asyncio
    async def test_round_counter_increments(self):
        config = load_config(domain="biomedical")
        mock_llm = MagicMock()
        dispatcher = RemediationDispatcher(llm=mock_llm, config=config)

        # Patch the internal method to avoid real LLM calls
        dispatcher._expand_queries = AsyncMock(return_value=True)

        kb = self._make_kb()
        check = self._make_check(
            "expand_queries",
            params={"uncovered_topics": ["topic_x"]},
        )

        await dispatcher.execute(kb, check)
        assert dispatcher._round_counts["expand_queries"] == 1

        await dispatcher.execute(kb, check)
        assert dispatcher._round_counts["expand_queries"] == 2

    @pytest.mark.asyncio
    async def test_lower_threshold_max_one_round(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check(
            "lower_screening_threshold",
            params={"current_rejection_rate": 0.85},
        )

        # First round should be allowed, max is 1
        dispatcher._lower_screening_threshold = AsyncMock(return_value=True)
        executed = await dispatcher.execute(kb, check)
        assert executed is True

        # Second round should be blocked
        executed = await dispatcher.execute(kb, check)
        assert executed is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline/test_remediation.py -v`
Expected: FAIL — `RemediationDispatcher` not importable

**Step 3: Write implementation**

Create `autoreview/pipeline/remediation.py`:

```python
"""Remediation dispatcher for actionable comprehensiveness checks."""
from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.comprehensiveness import ComprehensiveCheckResult
from autoreview.config.models import DomainConfig
from autoreview.models.knowledge_base import KnowledgeBase

logger = structlog.get_logger()

# Max rounds per action type
_ACTION_MAX_ROUNDS: dict[str, str] = {
    "expand_queries": "max_query_expansion_rounds",
    "retry_gap_search": "max_gap_search_rounds",
    "lower_screening_threshold": "max_screening_threshold_drops",
}


class RemediationDispatcher:
    """Executes remediation actions prescribed by comprehensiveness checks.

    Each check result may include a ``RemediationAction`` describing what
    the pipeline should do to fix the detected issue. This dispatcher
    reads the action and calls the appropriate method, tracking round
    counts to enforce iteration caps.
    """

    def __init__(self, llm: Any, config: DomainConfig) -> None:
        self.llm = llm
        self.config = config
        self._round_counts: dict[str, int] = {}

    def _max_rounds(self, action: str) -> int:
        """Get the max rounds for an action from config."""
        if action == "expand_queries":
            return self.config.search.max_query_expansion_rounds
        elif action == "retry_gap_search":
            return self.config.search.max_gap_search_rounds
        elif action == "lower_screening_threshold":
            return 1  # Always max 1
        return 0

    async def execute(
        self, kb: KnowledgeBase, check_result: ComprehensiveCheckResult,
    ) -> bool:
        """Execute the remediation action from a check result.

        Returns True if a remediation was performed, False otherwise.
        """
        if check_result.remediation is None:
            return False

        action = check_result.remediation.action
        params = check_result.remediation.params

        # Check round cap
        current_rounds = self._round_counts.get(action, 0)
        max_rounds = self._max_rounds(action)
        if current_rounds >= max_rounds:
            logger.info(
                "remediation.max_rounds_reached",
                action=action,
                rounds=current_rounds,
                max_rounds=max_rounds,
            )
            return False

        # Dispatch to handler
        handler = {
            "expand_queries": self._expand_queries,
            "retry_gap_search": self._retry_gap_search,
            "lower_screening_threshold": self._lower_screening_threshold,
        }.get(action)

        if handler is None:
            logger.warning("remediation.unknown_action", action=action)
            return False

        logger.info(
            "remediation.executing",
            action=action,
            round=current_rounds + 1,
            max_rounds=max_rounds,
            params=list(params.keys()),
        )

        result = await handler(kb, params)
        self._round_counts[action] = current_rounds + 1
        return result

    async def _expand_queries(
        self, kb: KnowledgeBase, params: dict[str, Any],
    ) -> bool:
        """Generate additional queries for uncovered sub-topics."""
        from autoreview.llm.prompts.query_expansion import (
            TARGETED_EXPANSION_SYSTEM_PROMPT,
            build_targeted_query_expansion_prompt,
        )
        from autoreview.models.base import AutoReviewModel
        from pydantic import Field

        uncovered_topics = params.get("uncovered_topics", [])
        if not uncovered_topics:
            return False

        class TargetedQueryResult(AutoReviewModel):
            pubmed_queries: list[str] = Field(default_factory=list)
            semantic_scholar_queries: list[str] = Field(default_factory=list)
            openalex_queries: list[str] = Field(default_factory=list)

        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=uncovered_topics,
            domain=self.config.domain,
            date_range=self.config.search.date_range,
        )

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=TargetedQueryResult,
            system=TARGETED_EXPANSION_SYSTEM_PROMPT,
        )
        result: TargetedQueryResult = response.parsed

        # Merge new queries into existing
        added = 0
        for db, new_queries in [
            ("pubmed", result.pubmed_queries),
            ("semantic_scholar", result.semantic_scholar_queries),
            ("openalex", result.openalex_queries),
        ]:
            if db in kb.search_queries and new_queries:
                existing = set(kb.search_queries[db])
                for q in new_queries:
                    if q not in existing:
                        kb.search_queries[db].append(q)
                        added += 1

        kb.add_audit_entry(
            "remediation",
            "expand_queries",
            f"Added {added} queries for {len(uncovered_topics)} uncovered topics",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        logger.info(
            "remediation.queries_expanded",
            uncovered_topics=uncovered_topics,
            queries_added=added,
        )
        return added > 0

    async def _retry_gap_search(
        self, kb: KnowledgeBase, params: dict[str, Any],
    ) -> bool:
        """Retry gap search with alternative queries for remaining gaps."""
        from autoreview.extraction.extractor import PaperExtractor, PaperScreener
        from autoreview.llm.prompts.clustering import (
            RETRY_GAP_SEARCH_SYSTEM_PROMPT,
            build_retry_gap_queries_prompt,
        )
        from autoreview.models.base import AutoReviewModel
        from autoreview.search.aggregator import SearchAggregator
        from pydantic import Field

        remaining_gaps = params.get("remaining_gaps", [])
        previous_queries = params.get("previous_queries", [])
        if not remaining_gaps:
            return False

        # Generate new queries via LLM
        class RetryGapQueryResult(AutoReviewModel):
            queries: list[str] = Field(default_factory=list)

        prompt = build_retry_gap_queries_prompt(remaining_gaps, previous_queries)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=RetryGapQueryResult,
            system=RETRY_GAP_SEARCH_SYSTEM_PROMPT,
        )
        new_queries = response.parsed.queries

        if not new_queries:
            return False

        # Build search sources (primary + secondary only)
        gap_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
        )
        sources = []
        for db in gap_dbs:
            try:
                if db == "semantic_scholar":
                    from autoreview.search.semantic_scholar import SemanticScholarSearch
                    sources.append(SemanticScholarSearch())
                elif db == "pubmed":
                    from autoreview.search.pubmed import PubMedSearch
                    sources.append(PubMedSearch())
                elif db == "openalex":
                    from autoreview.search.openalex import OpenAlexSearch
                    sources.append(OpenAlexSearch())
            except Exception:
                pass

        if not sources:
            return False

        queries_by_source = {db: new_queries for db in gap_dbs}
        agg = SearchAggregator(sources=sources)
        new_papers = await agg.search(queries_by_source, max_results_per_source=200)

        # Screen and extract
        screener = PaperScreener(self.llm)
        new_screened = await screener.screen(
            new_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        extractor = PaperExtractor(
            self.llm,
            domain_fields=self.config.extraction.domain_fields,
        )
        new_extractions = await extractor.extract_batch(
            [sp.paper for sp in new_screened],
        )

        # Merge
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        kb.add_audit_entry(
            "remediation",
            "retry_gap_search",
            f"Retry found {len(new_screened)} papers, {len(new_extractions)} extractions "
            f"for {len(remaining_gaps)} remaining gaps",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        logger.info(
            "remediation.gap_search_retried",
            remaining_gaps=len(remaining_gaps),
            new_papers=len(new_screened),
            new_extractions=len(new_extractions),
        )
        return len(new_screened) > 0

    async def _lower_screening_threshold(
        self, kb: KnowledgeBase, params: dict[str, Any],
    ) -> bool:
        """Re-screen rejected papers at a lower threshold."""
        from autoreview.extraction.extractor import PaperScreener

        current_threshold = self.config.search.relevance_threshold
        new_threshold = current_threshold - 1
        if new_threshold < 1:
            return False

        # Find papers that were not screened in (not in screened_papers)
        screened_ids = {sp.paper.id for sp in kb.screened_papers}
        rejected = [p for p in kb.candidate_papers if p.id not in screened_ids]

        if not rejected:
            return False

        screener = PaperScreener(self.llm, batch_size=self.config.search.screening_batch_size)
        newly_screened = await screener.screen(
            rejected,
            scope_document=kb.scope_document or "",
            threshold=new_threshold,
        )

        if newly_screened:
            kb.screened_papers.extend(newly_screened)

        kb.add_audit_entry(
            "remediation",
            "lower_screening_threshold",
            f"Re-screened {len(rejected)} papers at threshold {new_threshold}, "
            f"promoted {len(newly_screened)}",
        )

        logger.info(
            "remediation.threshold_lowered",
            old_threshold=current_threshold,
            new_threshold=new_threshold,
            rejected_count=len(rejected),
            promoted=len(newly_screened),
        )
        return len(newly_screened) > 0
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_remediation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add autoreview/pipeline/remediation.py tests/test_pipeline/test_remediation.py
git commit -m "feat: create RemediationDispatcher with expand, retry, and threshold actions"
```

---

### Task 7: Wire Dispatcher into Pipeline Nodes

**Files:**
- Modify: `autoreview/pipeline/nodes.py:39-44` (constructor)
- Modify: `autoreview/pipeline/nodes.py:46-88` (query_expansion node)
- Modify: `autoreview/pipeline/nodes.py:126-162` (screening node)
- Modify: `autoreview/pipeline/nodes.py:248-349` (gap_search node)
- Test: `tests/test_pipeline/test_runner.py` (existing tests still pass)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_runner.py`:

```python
class TestPipelineNodesHaveDispatcher:
    def test_nodes_have_dispatcher(self):
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(llm=None, config=config)
        assert hasattr(nodes, "dispatcher")
        assert nodes.dispatcher is not None
```

Also add the import at the top:
```python
from autoreview.pipeline.nodes import PipelineNodes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_runner.py::TestPipelineNodesHaveDispatcher -v`
Expected: FAIL — no `dispatcher` attribute

**Step 3: Write implementation**

Modify `autoreview/pipeline/nodes.py`:

**Constructor** (line 42-44) — add dispatcher:

```python
class PipelineNodes:
    """Collection of pipeline node functions."""

    def __init__(self, llm: Any, config: DomainConfig) -> None:
        self.llm = llm
        self.config = config
        # Lazy import to avoid circular dependency at module level
        from autoreview.pipeline.remediation import RemediationDispatcher
        self.dispatcher = RemediationDispatcher(llm, config)
```

**query_expansion node** (after line 88) — call dispatcher if remediation exists:

```python
        # Comprehensiveness: query coverage check
        query_checker = QueryCoverageChecker(self.llm)
        qc_result = await query_checker.check(kb.search_queries, kb.scope_document or "")
        kb.comprehensiveness_checks.append(qc_result)

        # Auto-remediate: expand queries for uncovered topics
        if qc_result.remediation:
            await self.dispatcher.execute(kb, qc_result)
            # Re-check after expansion
            qc_recheck = await query_checker.check(kb.search_queries, kb.scope_document or "")
            kb.comprehensiveness_checks.append(qc_recheck)
            # Second round if still warning
            if qc_recheck.remediation:
                await self.dispatcher.execute(kb, qc_recheck)
```

**screening node** (after line 147) — call dispatcher if anomaly check has remediation:

```python
        anomaly_checker = CoverageAnomalyChecker()
        anomaly_result = anomaly_checker.check(
            kb.candidate_papers, kb.screened_papers, expected_sources=all_dbs,
        )
        kb.comprehensiveness_checks.append(anomaly_result)

        # Auto-remediate: lower threshold or expand queries
        if anomaly_result.remediation:
            await self.dispatcher.execute(kb, anomaly_result)
```

**gap_search node** (after line 343) — call dispatcher if revalidation has remediation:

```python
        revalidator = PostGapRevalidator(self.llm)
        reval_result = await revalidator.check(
            kb.evidence_map.themes if kb.evidence_map else [],
            kb.scope_document or "",
            pre_gaps=pre_gaps,
            pre_coverage=pre_coverage,
        )
        kb.comprehensiveness_checks.append(reval_result)

        # Auto-remediate: retry gap search for remaining gaps
        if reval_result.remediation:
            # Pass previous queries so retry uses different terms
            prev_queries = []
            for gap in pre_gaps:
                prev_queries.extend(gap.suggested_queries)
            reval_result.remediation.params["previous_queries"] = prev_queries

            await self.dispatcher.execute(kb, reval_result)
            # Re-validate after retry
            reval_recheck = await revalidator.check(
                kb.evidence_map.themes if kb.evidence_map else [],
                kb.scope_document or "",
                pre_gaps=pre_gaps,
                pre_coverage=pre_coverage,
            )
            kb.comprehensiveness_checks.append(reval_recheck)
            # Second round if still warning
            if reval_recheck.remediation:
                reval_recheck.remediation.params["previous_queries"] = prev_queries
                await self.dispatcher.execute(kb, reval_recheck)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_runner.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass, no regressions

**Step 6: Commit**

```bash
git add autoreview/pipeline/nodes.py tests/test_pipeline/test_runner.py
git commit -m "feat: wire RemediationDispatcher into pipeline nodes"
```

---

### Task 8: Integration Test — Full Remediation Flow

**Files:**
- Create: `tests/test_pipeline/test_remediation_integration.py`

**Step 1: Write the integration test**

```python
"""Integration test for remediation flow through pipeline nodes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.analysis.comprehensiveness import CheckStatus
from autoreview.config import load_config
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.pipeline.nodes import PipelineNodes


class MockRemediationLLM:
    """Mock LLM that supports query expansion and coverage checking."""

    def __init__(self):
        self.call_count = 0

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        self.call_count += 1
        name = response_model.__name__

        if name == "QueryExpansionResult":
            from autoreview.models.base import AutoReviewModel
            from pydantic import Field
            class QueryExpansionResult(AutoReviewModel):
                pubmed_queries: list[str] = Field(default_factory=list)
                semantic_scholar_queries: list[str] = Field(default_factory=list)
                openalex_queries: list[str] = Field(default_factory=list)
                perplexity_questions: list[str] = Field(default_factory=list)
                scope_document: str = ""
            return LLMStructuredResponse(
                parsed=QueryExpansionResult(
                    pubmed_queries=["query1"],
                    semantic_scholar_queries=["query2"],
                    openalex_queries=["query3"],
                    perplexity_questions=["question1"],
                    scope_document="This review covers topic A, topic B, and topic C.",
                ),
                input_tokens=100, output_tokens=200,
            )

        if name == "QueryCoverageResult":
            from autoreview.llm.prompts.comprehensiveness import QueryCoverageResult, SubTopicCoverage
            # First call: missing topic C. After expansion: all covered.
            if self.call_count <= 2:
                return LLMStructuredResponse(
                    parsed=QueryCoverageResult(
                        sub_topic_assessments=[
                            SubTopicCoverage(sub_topic="Topic A", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic B", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic C", covered=False, matching_queries=[]),
                        ],
                        overall_coverage_score=0.67,
                    ),
                    input_tokens=100, output_tokens=100,
                )
            else:
                return LLMStructuredResponse(
                    parsed=QueryCoverageResult(
                        sub_topic_assessments=[
                            SubTopicCoverage(sub_topic="Topic A", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic B", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic C", covered=True, matching_queries=["q"]),
                        ],
                        overall_coverage_score=1.0,
                    ),
                    input_tokens=100, output_tokens=100,
                )

        if name == "TargetedQueryResult":
            from autoreview.models.base import AutoReviewModel
            from pydantic import Field
            class TargetedQueryResult(AutoReviewModel):
                pubmed_queries: list[str] = Field(default_factory=list)
                semantic_scholar_queries: list[str] = Field(default_factory=list)
                openalex_queries: list[str] = Field(default_factory=list)
            return LLMStructuredResponse(
                parsed=TargetedQueryResult(
                    pubmed_queries=["topic C expanded query"],
                    semantic_scholar_queries=["topic C semantic"],
                    openalex_queries=["topic C openalex"],
                ),
                input_tokens=100, output_tokens=100,
            )

        raise ValueError(f"Unexpected model: {name}")


class TestQueryExpansionRemediation:
    @pytest.mark.asyncio
    async def test_query_expansion_auto_remediates(self):
        """After query expansion, missing sub-topics trigger targeted expansion."""
        config = load_config(domain="biomedical")
        llm = MockRemediationLLM()
        nodes = PipelineNodes(llm=llm, config=config)
        kb = KnowledgeBase(topic="test topic", domain="biomedical", output_dir="/tmp/test")

        await nodes.query_expansion(kb)

        # Should have original queries plus expanded ones
        assert len(kb.search_queries["pubmed"]) > 1
        assert "topic C expanded query" in kb.search_queries["pubmed"]

        # Should have multiple comprehensiveness checks (initial + recheck)
        qc_checks = [c for c in kb.comprehensiveness_checks if c.check_name == "query_coverage"]
        assert len(qc_checks) >= 2

        # Should have audit entry for remediation
        remediation_entries = [e for e in kb.audit_log if e.node_name == "remediation"]
        assert len(remediation_entries) >= 1
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_pipeline/test_remediation_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_pipeline/test_remediation_integration.py
git commit -m "test: add integration test for query expansion remediation flow"
```

---

### Task 9: Final Verification

**Step 1: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass, no regressions

**Step 2: Verify import chain is clean**

Run: `python -c "from autoreview.pipeline.remediation import RemediationDispatcher; print('OK')"`
Expected: `OK`

Run: `python -c "from autoreview.analysis.comprehensiveness import RemediationAction; print('OK')"`
Expected: `OK`

**Step 3: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: final cleanup for adaptive search remediation"
```

---

## Summary

| Task | What | New/Modify | Test File |
|------|------|------------|-----------|
| 1 | RemediationAction model | Modify `comprehensiveness.py` | `test_comprehensiveness.py` |
| 2 | Config fields | Modify `models.py` + 3 YAMLs | `test_runner.py` |
| 3 | Targeted expansion prompt | Modify `query_expansion.py` | `test_query_expansion_prompts.py` |
| 4 | Retry gap prompt | Modify `clustering.py` | `test_clustering_prompts.py` |
| 5 | Checks populate remediation | Modify `comprehensiveness.py` | `test_comprehensiveness.py` |
| 6 | RemediationDispatcher | Create `remediation.py` | `test_remediation.py` |
| 7 | Wire into nodes | Modify `nodes.py` | `test_runner.py` |
| 8 | Integration test | — | `test_remediation_integration.py` |
| 9 | Final verification | — | Full suite |

**Totals:** 1 new file, 7 modified files, 4 new test files
