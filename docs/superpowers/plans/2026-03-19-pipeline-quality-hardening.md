# Pipeline Quality Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 28 gaps identified in the comprehensive pipeline audit, making AutoReview outputs competitive with human-written review papers across search coverage, analytical rigor, writing quality, and operational robustness.

**Architecture:** Four independent phases, each shippable and testable on its own. Phase 1 fixes critical wiring gaps (search backends not connected, critique gates missing). Phase 2 deepens analytical capability (snowballing, study quality, contradiction resolution). Phase 3 improves writing output (polish stage, exemplars, transitions). Phase 4 hardens operations (timeouts, graceful degradation, pre-flight checks).

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, pytest + pytest-asyncio, structlog, httpx

**Audit source:** Comprehensive audit performed 2026-03-19 across all pipeline stages.

---

## File Structure

### Phase 1 — Wiring & Validation
| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `autoreview/pipeline/nodes.py:271-312, 574-589, 882-897, 1087-1102, 1299-1314` | Wire CrossRef + Europe PMC into all 6 search instantiation sites |
| Modify | `autoreview/pipeline/remediation.py:194-209` | Wire CrossRef + Europe PMC into remediation gap search |
| Modify | `autoreview/critique/section_critic.py:91-126` | Add per-dimension minimum gates |
| Modify | `autoreview/critique/holistic_critic.py:59-97` | Add per-dimension minimum gates |
| Modify | `autoreview/critique/models.py:41-55` | Add `dimension_minimums` field to CritiqueReport check |
| Create | `autoreview/critique/dimension_gates.py` | Per-dimension gate logic (reusable by section + holistic) |
| Modify | `autoreview/pipeline/dag.py:187` | Wrap node execution with asyncio.wait_for timeout |
| Create | `tests/test_pipeline/test_search_wiring.py` | Tests for CrossRef/EuropePMC wiring |
| Create | `tests/test_critique/test_dimension_gates.py` | Tests for per-dimension gate logic |
| Modify | `tests/test_pipeline/test_dag.py` | Add timeout tests |

### Phase 2 — Analytical Depth
| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `autoreview/extraction/models.py:71-85` | Add `study_design`, `quality_score` fields to PaperExtraction |
| Modify | `autoreview/llm/prompts/extraction.py` | Update extraction prompt to request study design + quality |
| Create | `autoreview/search/snowballing.py` | Backward/forward citation snowballing via S2 API |
| Modify | `autoreview/pipeline/nodes.py` (screening node) | Add snowballing after initial screening |
| Modify | `autoreview/analysis/clustering.py` | Enhance contradiction detection with structured resolution |
| Create | `autoreview/analysis/contradiction_resolver.py` | Study quality comparison, population comparison, resolution logic |
| Create | `autoreview/analysis/synthesis_validator.py` | Post-writing synthesis metric (multi-paper claim ratio) |
| Create | `tests/test_search/test_snowballing.py` | Snowballing tests |
| Create | `tests/test_analysis/test_contradiction_resolver.py` | Contradiction resolution tests |
| Create | `tests/test_analysis/test_synthesis_validator.py` | Synthesis validation tests |

### Phase 3 — Writing Quality
| Action | File | Responsibility |
|--------|------|----------------|
| Create | `autoreview/writing/polish.py` | Language polish stage (terminology, voice, dedup) |
| Modify | `autoreview/llm/prompts/writing.py` | Add synthesis exemplar paragraphs to system prompt |
| Create | `autoreview/writing/transition_repair.py` | Post-assembly transition repair pass |
| Modify | `autoreview/config/depth.py:21-34` | Add `quality_threshold` to DepthProfile |
| Modify | `autoreview/critique/section_critic.py` | Use depth-dependent threshold |
| Modify | `autoreview/critique/holistic_critic.py` | Use depth-dependent threshold |
| Modify | `autoreview/llm/prompts/critique.py` | Add diagnostic revision guidance |
| Create | `tests/test_writing/test_polish.py` | Polish stage tests |
| Create | `tests/test_writing/test_transition_repair.py` | Transition repair tests |
| Modify | `tests/test_critique/test_critics.py` | Depth-dependent threshold tests |

### Phase 4 — Robustness
| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `autoreview/pipeline/dag.py` | Graceful token budget degradation (warn at 80%, degrade at 95%) |
| Modify | `autoreview/models/knowledge_base.py` | Add schema_version + checksum to snapshots |
| Modify | `autoreview/pipeline/nodes.py` (extraction) | Wrap individual paper extraction in try/except |
| Create | `autoreview/pipeline/preflight.py` | Pre-flight validation (API keys, output dir, connectivity) |
| Create | `autoreview/cli/check.py` | CLI `autoreview check` command |
| Create | `tests/test_pipeline/test_preflight.py` | Pre-flight tests |
| Modify | `tests/test_pipeline/test_dag.py` | Token budget degradation tests |

---

## Phase 1: Wiring & Validation

### Task 1: Wire CrossRef + Europe PMC into Pipeline Search

**Context:** CrossRef and Europe PMC are fully implemented (`CrossRefSearch` in `autoreview/search/crossref.py:46`, `EuropePMCSearch` in `autoreview/search/europe_pmc.py:20`), tested, and declared in all 3 domain YAML configs — but `nodes.py` only instantiates PubMed/S2/OpenAlex. There are 6 identical if/elif blocks that need updating. CORE is intentionally excluded (known unreliable per 2026-03-19 learnings).

**Files:**
- Modify: `autoreview/pipeline/nodes.py:271-284, 574-587, 882-895, 1087-1100, 1299-1312`
- Modify: `autoreview/pipeline/remediation.py:194-207`
- Create: `tests/test_pipeline/test_search_wiring.py`

- [ ] **Step 1: Write the test for search source wiring**

Create `tests/test_pipeline/test_search_wiring.py`:

```python
"""Tests that CrossRef and Europe PMC are wired into pipeline search nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoreview.pipeline.search_factory import build_search_sources


class TestSearchSourceWiring:
    """Verify that all declared database sources are instantiated."""

    @pytest.mark.parametrize(
        "db_name,import_path",
        [
            ("crossref", "autoreview.pipeline.search_factory.CrossRefSearch"),
            ("europe_pmc", "autoreview.pipeline.search_factory.EuropePMCSearch"),
        ],
    )
    def test_source_instantiated_for_declared_db(
        self, db_name: str, import_path: str
    ) -> None:
        """When a database is declared in config, it should appear in sources."""
        with patch(import_path) as mock_cls:
            mock_cls.return_value = MagicMock()
            sources = build_search_sources([db_name])
            mock_cls.assert_called_once()
            assert len(sources) == 1

    def test_all_five_sources_instantiated(self) -> None:
        """Full config with 5 sources should yield 5 source objects (no CORE)."""
        sources = build_search_sources(
            ["pubmed", "semantic_scholar", "openalex", "europe_pmc", "crossref"]
        )
        assert len(sources) == 5

    def test_unknown_source_skipped_with_warning(self) -> None:
        """Unknown database names should be skipped, not crash."""
        sources = build_search_sources(["nonexistent_db"])
        assert len(sources) == 0

    def test_core_skipped_intentionally(self) -> None:
        """CORE should be skipped with info log, not instantiated."""
        sources = build_search_sources(["core"])
        assert len(sources) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_search_wiring.py -v`
Expected: FAIL — `_build_search_sources` does not exist yet

- [ ] **Step 3: Extract `_build_search_sources` helper in nodes.py**

The same if/elif block is duplicated 5 times in `nodes.py` and once in `remediation.py`. Extract a shared helper method on `PipelineNodes`:

In `autoreview/pipeline/nodes.py`, add this method to the `PipelineNodes` class (after `__init__`, before `search()`):

```python
def _build_search_sources(self, db_names: list[str]) -> list[Any]:
    """Instantiate search source objects for the given database names.

    Supports: pubmed, semantic_scholar, openalex, crossref, europe_pmc.
    CORE intentionally excluded (unreliable — see learnings 2026-03-19).
    Unknown names are skipped with a warning log.
    """
    sources: list[Any] = []
    for db in db_names:
        try:
            if db == "pubmed":
                from autoreview.search.pubmed import PubMedSearch

                sources.append(PubMedSearch())
            elif db == "semantic_scholar":
                from autoreview.search.semantic_scholar import SemanticScholarSearch

                sources.append(SemanticScholarSearch())
            elif db == "openalex":
                from autoreview.search.openalex import OpenAlexSearch

                sources.append(OpenAlexSearch())
            elif db == "crossref":
                from autoreview.search.crossref import CrossRefSearch

                sources.append(CrossRefSearch())
            elif db == "europe_pmc":
                from autoreview.search.europe_pmc import EuropePMCSearch

                sources.append(EuropePMCSearch())
            elif db == "core":
                logger.info("search.core_skipped", reason="CORE API unreliable per 2026-03-19 learnings")
            else:
                logger.warning("search.unknown_source", source=db)
        except Exception as e:
            logger.warning("search.source_init_failed", source=db, error=str(e))
    return sources
```

Then replace all 5 inline if/elif blocks in `nodes.py` with calls to `self._build_search_sources(all_dbs)`. The blocks to replace are at:
- `search()` method: lines 271-284
- `gap_search()` method: lines 574-587
- `contextual_enrichment()` method: lines 882-895
- `corpus_expansion()` method: lines 1087-1100
- `passage_search()` method: lines 1299-1312

Each replacement looks like:
```python
# BEFORE (repeated 5 times):
for db in all_dbs:
    try:
        if db == "pubmed":
            ...
        elif db == "semantic_scholar":
            ...
        elif db == "openalex":
            ...
    except Exception as e:
        logger.warning(...)

# AFTER:
sources = self._build_search_sources(all_dbs)
```

Also update `remediation.py:194-207` — change `RemediationDispatcher._retry_gap_search()` to use the same pattern. Since `RemediationDispatcher` doesn't extend `PipelineNodes`, either:
- (a) Import `PipelineNodes._build_search_sources` as a standalone function, or
- (b) Extract to a module-level function in a shared location

Preferred: Extract `_build_search_sources` as a **standalone function** in `autoreview/pipeline/search_factory.py` (new file) and import it in both `nodes.py` and `remediation.py`.

Create `autoreview/pipeline/search_factory.py`:

```python
"""Factory for instantiating search source objects from database names."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def build_search_sources(db_names: list[str]) -> list[Any]:
    """Instantiate search source objects for the given database names.

    Supports: pubmed, semantic_scholar, openalex, crossref, europe_pmc.
    CORE intentionally excluded (unreliable — see learnings 2026-03-19).
    Unknown names are skipped with a warning log.
    """
    sources: list[Any] = []
    for db in db_names:
        try:
            if db == "pubmed":
                from autoreview.search.pubmed import PubMedSearch

                sources.append(PubMedSearch())
            elif db == "semantic_scholar":
                from autoreview.search.semantic_scholar import SemanticScholarSearch

                sources.append(SemanticScholarSearch())
            elif db == "openalex":
                from autoreview.search.openalex import OpenAlexSearch

                sources.append(OpenAlexSearch())
            elif db == "crossref":
                from autoreview.search.crossref import CrossRefSearch

                sources.append(CrossRefSearch())
            elif db == "europe_pmc":
                from autoreview.search.europe_pmc import EuropePMCSearch

                sources.append(EuropePMCSearch())
            elif db == "core":
                logger.info(
                    "search.core_skipped",
                    reason="CORE API unreliable — see learnings 2026-03-19",
                )
            else:
                logger.warning("search.unknown_source", source=db)
        except Exception as e:
            logger.warning("search.source_init_failed", source=db, error=str(e))
    return sources
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_search_wiring.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All 1133+ tests pass

- [ ] **Step 6: Commit**

```bash
git add autoreview/pipeline/search_factory.py autoreview/pipeline/nodes.py autoreview/pipeline/remediation.py tests/test_pipeline/test_search_wiring.py
git commit -m "feat(search): wire CrossRef + Europe PMC into all pipeline search nodes

Extract build_search_sources() factory to eliminate 6x duplicated if/elif
blocks. CrossRef and Europe PMC now instantiated when declared in domain
configs. CORE intentionally skipped (unreliable per 2026-03-19 audit)."
```

---

### Task 2: Add Per-Dimension Critique Gates

**Context:** Currently, a section scoring 0.5 on synthesis but 0.9 on completeness passes the 0.80 overall threshold. The critique system has no per-dimension minimums. Synthesis quality is the core differentiator — it must have a floor.

**Files:**
- Create: `autoreview/critique/dimension_gates.py`
- Modify: `autoreview/critique/section_critic.py:91-126`
- Modify: `autoreview/critique/holistic_critic.py:59-97`
- Create: `tests/test_critique/test_dimension_gates.py`

- [ ] **Step 1: Write the test for dimension gates**

Create `tests/test_critique/test_dimension_gates.py`:

```python
"""Tests for per-dimension minimum critique gates."""

from __future__ import annotations

import pytest

from autoreview.critique.dimension_gates import (
    DEFAULT_SECTION_GATES,
    DEFAULT_HOLISTIC_GATES,
    check_dimension_gates,
    DimensionGateResult,
)


class TestCheckDimensionGates:
    def test_all_above_minimums_passes(self) -> None:
        scores = {"synthesis_quality": 0.80, "citation_accuracy": 0.75, "coverage": 0.85}
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is True
        assert result.failures == []

    def test_synthesis_below_minimum_fails(self) -> None:
        scores = {"synthesis_quality": 0.50, "coverage": 0.95}
        gates = {"synthesis_quality": 0.65}
        result = check_dimension_gates(scores, gates)
        assert result.passed is False
        assert len(result.failures) == 1
        assert result.failures[0].dimension == "synthesis_quality"
        assert result.failures[0].score == 0.50
        assert result.failures[0].minimum == 0.65

    def test_missing_dimension_ignored(self) -> None:
        """Dimensions not in scores dict are not checked."""
        scores = {"coverage": 0.90}
        gates = {"synthesis_quality": 0.65}
        result = check_dimension_gates(scores, gates)
        assert result.passed is True

    def test_multiple_failures_reported(self) -> None:
        scores = {"synthesis_quality": 0.40, "citation_accuracy": 0.30}
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is False
        assert len(result.failures) == 2

    def test_default_section_gates_include_synthesis(self) -> None:
        assert "synthesis_quality" in DEFAULT_SECTION_GATES
        assert DEFAULT_SECTION_GATES["synthesis_quality"] >= 0.60

    def test_default_holistic_gates_include_narrative(self) -> None:
        assert "narrative_arc" in DEFAULT_HOLISTIC_GATES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critique/test_dimension_gates.py -v`
Expected: FAIL — module `dimension_gates` does not exist

- [ ] **Step 3: Implement dimension gates module**

Create `autoreview/critique/dimension_gates.py`:

```python
"""Per-dimension minimum gates for critique reports.

Prevents high overall scores from masking critical dimension failures.
A section scoring 0.50 on synthesis but 0.90 on completeness should NOT
pass, even if the weighted average exceeds the overall threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DimensionFailure:
    """A single dimension that fell below its minimum gate."""

    dimension: str
    score: float
    minimum: float


@dataclass(frozen=True)
class DimensionGateResult:
    """Result of checking dimension-level gates."""

    passed: bool
    failures: list[DimensionFailure]


# Section-level gates: dimensions where a minimum floor is enforced
DEFAULT_SECTION_GATES: dict[str, float] = {
    "synthesis_quality": 0.65,   # Core differentiator — must not be summary
    "citation_accuracy": 0.60,   # Claims must be attributed
}

# Holistic-level gates
DEFAULT_HOLISTIC_GATES: dict[str, float] = {
    "synthesis_quality": 0.65,
    "narrative_arc": 0.60,
    "citation_accuracy": 0.60,
}


def check_dimension_gates(
    dimension_scores: dict[str, float],
    gates: dict[str, float],
) -> DimensionGateResult:
    """Check dimension scores against minimum gates.

    Args:
        dimension_scores: Actual scores per dimension from the critique.
        gates: Minimum required score per dimension.

    Returns:
        DimensionGateResult indicating pass/fail with failure details.
    """
    failures: list[DimensionFailure] = []
    for dimension, minimum in gates.items():
        score = dimension_scores.get(dimension)
        if score is not None and score < minimum:
            failures.append(
                DimensionFailure(dimension=dimension, score=score, minimum=minimum)
            )
            logger.warning(
                "critique.dimension_gate_failed",
                dimension=dimension,
                score=score,
                minimum=minimum,
            )
    return DimensionGateResult(passed=len(failures) == 0, failures=failures)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_critique/test_dimension_gates.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Integrate gates into section_critic.py**

Modify `autoreview/critique/section_critic.py`. In the `section_critique_loop()` function (around line 91), after the line that checks `if report.passed`:

```python
# EXISTING (around line 123):
if report.passed or not should_continue_revision(...):
    break

# REPLACE WITH:
from autoreview.critique.dimension_gates import (
    check_dimension_gates,
    DEFAULT_SECTION_GATES,
)

gate_result = check_dimension_gates(report.dimension_scores, DEFAULT_SECTION_GATES)
if not gate_result.passed:
    # Override passed flag — dimension gates failed
    report.passed = False
    for failure in gate_result.failures:
        report.issues.append(
            CritiqueIssue(
                severity="critical",
                location=f"dimension:{failure.dimension}",
                description=(
                    f"{failure.dimension} scored {failure.score:.2f}, "
                    f"below minimum gate {failure.minimum:.2f}"
                ),
                suggested_fix=f"Improve {failure.dimension} — this is a hard requirement.",
            )
        )

if report.passed or not should_continue_revision(...):
    break
```

Apply the same pattern to `holistic_critic.py` (around line 91), using `DEFAULT_HOLISTIC_GATES`.

- [ ] **Step 6: Run full critique test suite**

Run: `pytest tests/test_critique/ -v`
Expected: All existing tests pass + new gate tests pass

- [ ] **Step 7: Commit**

```bash
git add autoreview/critique/dimension_gates.py autoreview/critique/section_critic.py autoreview/critique/holistic_critic.py tests/test_critique/test_dimension_gates.py
git commit -m "feat(critique): add per-dimension minimum gates for synthesis and citation accuracy

Sections scoring below 0.65 on synthesis_quality or 0.60 on
citation_accuracy now fail critique regardless of overall score.
Prevents high completeness scores from masking shallow synthesis."
```

---

### Task 3: Add Citation Scope Validation

**Context:** The pipeline doesn't verify that citations used in a section match the papers assigned to that section. A section writer could cite `[@p3]` even though only `[@p1]` and `[@p2]` were assigned. The `_extract_citations()` regex at `section_writer.py:275-277` extracts IDs but doesn't validate scope.

**Note:** An existing `autoreview/validation/citation_validator.py` handles citation *matching* (DOI/title). This new module handles citation *scope* (are cited papers assigned to this section?). Named `citation_scope.py` to avoid collision.

**Files:**
- Create: `autoreview/writing/citation_scope.py`
- Modify: `autoreview/writing/section_writer.py:275-277`
- Create: `tests/test_writing/test_citation_scope.py`

- [ ] **Step 1: Write the test**

Create `tests/test_writing/test_citation_scope.py`:

```python
"""Tests for citation scope validation in section writing."""

from __future__ import annotations

import pytest

from autoreview.writing.citation_scope import validate_citation_scope


class TestValidateCitationScope:
    def test_all_citations_in_scope(self) -> None:
        cited = ["p1", "p2", "p3"]
        assigned = ["p1", "p2", "p3", "p4"]
        result = validate_citation_scope(cited, assigned)
        assert result.out_of_scope == []
        assert result.all_in_scope is True

    def test_out_of_scope_citations_detected(self) -> None:
        cited = ["p1", "p2", "p99"]
        assigned = ["p1", "p2", "p3"]
        result = validate_citation_scope(cited, assigned)
        assert result.out_of_scope == ["p99"]
        assert result.all_in_scope is False

    def test_empty_citations_passes(self) -> None:
        result = validate_citation_scope([], ["p1", "p2"])
        assert result.all_in_scope is True

    def test_uncited_assigned_papers_reported(self) -> None:
        cited = ["p1"]
        assigned = ["p1", "p2", "p3"]
        result = validate_citation_scope(cited, assigned)
        assert set(result.uncited_assigned) == {"p2", "p3"}

    def test_citation_utilization_rate(self) -> None:
        cited = ["p1", "p2"]
        assigned = ["p1", "p2", "p3", "p4"]
        result = validate_citation_scope(cited, assigned)
        assert result.utilization_rate == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writing/test_citation_scope.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement citation scope validator**

Create `autoreview/writing/citation_scope.py`:

```python
"""Validate that citations used in a section match assigned papers."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CitationScopeResult:
    """Result of citation scope validation for a section."""

    out_of_scope: list[str]
    uncited_assigned: list[str]
    utilization_rate: float
    all_in_scope: bool


def validate_citation_scope(
    cited_ids: list[str],
    assigned_ids: list[str],
) -> CitationScopeResult:
    """Check that cited paper IDs are within the assigned set.

    Args:
        cited_ids: Paper IDs found in the section text (from [@id] markers).
        assigned_ids: Paper IDs assigned to this section by the outline.

    Returns:
        CitationScopeResult with out-of-scope and uncited details.
    """
    cited_set = set(cited_ids)
    assigned_set = set(assigned_ids)

    out_of_scope = sorted(cited_set - assigned_set)
    uncited = sorted(assigned_set - cited_set)
    utilization = len(cited_set & assigned_set) / len(assigned_set) if assigned_set else 1.0

    if out_of_scope:
        logger.warning(
            "citation.out_of_scope",
            out_of_scope_ids=out_of_scope,
            count=len(out_of_scope),
        )

    return CitationScopeResult(
        out_of_scope=out_of_scope,
        uncited_assigned=uncited,
        utilization_rate=utilization,
        all_in_scope=len(out_of_scope) == 0,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writing/test_citation_scope.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Integrate into section writer**

In `autoreview/writing/section_writer.py`, after citations are extracted (around line 397 where `_extract_citations` is called), add scope validation:

```python
from autoreview.writing.citation_scope import validate_citation_scope

# After _extract_citations call:
citations_used = _extract_citations(section_text)
scope_result = validate_citation_scope(citations_used, assigned_paper_ids)
if not scope_result.all_in_scope:
    logger.warning(
        "section_writer.out_of_scope_citations",
        section_id=section_id,
        out_of_scope=scope_result.out_of_scope,
    )
```

- [ ] **Step 6: Run full writing test suite**

Run: `pytest tests/test_writing/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add autoreview/writing/citation_scope.py autoreview/writing/section_writer.py tests/test_writing/test_citation_scope.py
git commit -m "feat(writing): add citation scope validation

Detect out-of-scope citations (papers cited but not assigned to
the section) and track citation utilization rate. Logs warnings
for out-of-scope citations to support future enforcement."
```

---

### Task 4: Add Per-Node Timeouts in DAG Runner

**Context:** A hung API call blocks the entire pipeline indefinitely. `dag.py:187` calls `await node.func(context)` with no timeout. Need to wrap with `asyncio.wait_for()`.

**Files:**
- Modify: `autoreview/pipeline/dag.py:187`
- Modify: `autoreview/pipeline/dag.py` (DAGNode dataclass around line 15)
- Modify: `tests/test_pipeline/test_dag.py`

- [ ] **Step 1: Write the timeout test**

Add to `tests/test_pipeline/test_dag.py`:

**Note:** `DAGRunner.add_node()` takes `(name, func, dependencies)` as separate args, NOT a `DAGNode` object — it constructs `DAGNode` internally. `DAGExecutionError.__init__` takes `(node_name, original_error)` — two args, not a string. Tests and implementation must match these signatures.

```python
import asyncio

import pytest

from autoreview.pipeline.dag import DAGRunner, DAGExecutionError


class TestDAGTimeout:
    @pytest.mark.asyncio
    async def test_node_timeout_raises_error(self) -> None:
        """Node exceeding timeout should raise DAGExecutionError."""
        async def slow_node(ctx: dict) -> dict:
            await asyncio.sleep(10)
            return {"result": "done"}

        runner = DAGRunner()
        runner.add_node("slow", slow_node, timeout_seconds=0.1)
        with pytest.raises(DAGExecutionError, match="timed out"):
            await runner.execute({})

    @pytest.mark.asyncio
    async def test_node_without_timeout_uses_default(self) -> None:
        """Nodes without explicit timeout should use the default (300s)."""
        async def fast_node(ctx: dict) -> dict:
            return {"result": "fast"}

        runner = DAGRunner()
        runner.add_node("fast", fast_node)
        result = await runner.execute({})
        # Should complete without timeout
        assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_dag.py::TestDAGTimeout -v`
Expected: FAIL — `timeout_seconds` not accepted by `add_node`

- [ ] **Step 3: Add timeout_seconds to DAGNode and update add_node + execution**

In `autoreview/pipeline/dag.py`:

1. Add `timeout_seconds` field to `DAGNode` dataclass (around line 15):
```python
@dataclass
class DAGNode:
    name: str
    func: Callable
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float = 300.0  # 5 minutes default
```

2. Update `add_node()` to accept `timeout_seconds`:
```python
def add_node(self, name: str, func: Callable, dependencies: list[str] | None = None, timeout_seconds: float = 300.0) -> None:
    node = DAGNode(name=name, func=func, dependencies=dependencies or [], timeout_seconds=timeout_seconds)
    self.nodes[name] = node
```

3. Wrap the node execution call (line 187):
```python
# BEFORE:
result = await node.func(context)

# AFTER:
try:
    result = await asyncio.wait_for(
        node.func(context),
        timeout=node.timeout_seconds,
    )
except asyncio.TimeoutError:
    timeout_err = asyncio.TimeoutError(f"timed out after {node.timeout_seconds}s")
    raise DAGExecutionError(node.name, timeout_err)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_dag.py -v`
Expected: All existing + new timeout tests pass

- [ ] **Step 5: Set appropriate timeouts in pipeline node registration**

In the DAG builder (wherever nodes are registered), set timeouts:
- Search nodes: 300s (5 min)
- Extraction: 600s (10 min — many papers)
- Clustering/analysis: 300s
- Writing: 900s (15 min — LLM-intensive)
- Critique: 300s

This is done where `DAGNode(name="search", func=...)` is constructed — add `timeout_seconds=300` etc.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add autoreview/pipeline/dag.py tests/test_pipeline/test_dag.py
git commit -m "feat(dag): add per-node timeout protection

Nodes exceeding their timeout raise DAGExecutionError instead of
hanging indefinitely. Default 300s, configurable per node. Search=300s,
extraction=600s, writing=900s."
```

---

## Phase 2: Analytical Depth

### Task 5: Add Study Quality Grading to Extraction

**Context:** The extraction model has `evidence_strength` (4-level enum) but no study design classification or quality score. All papers contribute equally to consensus claims regardless of methodology rigor.

**Files:**
- Modify: `autoreview/extraction/models.py:71-85`
- Modify: `autoreview/llm/prompts/extraction.py` (extraction prompt)
- Create: `tests/test_extraction/test_study_quality.py`

- [ ] **Step 1: Write the test**

Create `tests/test_extraction/test_study_quality.py`:

```python
"""Tests for study quality fields on PaperExtraction."""

from __future__ import annotations

from autoreview.extraction.models import (
    PaperExtraction,
    Finding,
    EvidenceStrength,
    StudyDesign,
)


class TestStudyDesignEnum:
    def test_all_designs_defined(self) -> None:
        expected = {
            "rct", "cohort", "case_control", "cross_sectional",
            "case_series", "case_report", "in_vitro", "computational",
            "meta_analysis", "systematic_review", "narrative_review",
            "other",
        }
        assert set(StudyDesign) == expected


class TestPaperExtractionQuality:
    def test_study_design_defaults_to_none(self) -> None:
        ext = PaperExtraction(
            paper_id="p1",
            key_findings=[],
            methods_summary="test",
            limitations="test",
        )
        assert ext.study_design is None

    def test_quality_score_range(self) -> None:
        ext = PaperExtraction(
            paper_id="p1",
            key_findings=[],
            methods_summary="test",
            limitations="test",
            study_design=StudyDesign.RCT,
            quality_score=0.85,
        )
        assert 0.0 <= ext.quality_score <= 1.0

    def test_quality_score_defaults_to_none(self) -> None:
        ext = PaperExtraction(
            paper_id="p1",
            key_findings=[],
            methods_summary="test",
            limitations="test",
        )
        assert ext.quality_score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extraction/test_study_quality.py -v`
Expected: FAIL — `StudyDesign` does not exist

- [ ] **Step 3: Add StudyDesign enum and quality fields to models.py**

In `autoreview/extraction/models.py`:

```python
class StudyDesign(StrEnum):
    """Classification of study methodology."""
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    IN_VITRO = "in_vitro"
    COMPUTATIONAL = "computational"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    NARRATIVE_REVIEW = "narrative_review"
    OTHER = "other"
```

Add to `PaperExtraction` (after `domain_specific_fields`):

```python
study_design: StudyDesign | None = None
quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
sample_size: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction/test_study_quality.py -v`
Expected: PASS

- [ ] **Step 5: Update extraction prompt to request study design**

In `autoreview/llm/prompts/extraction.py`, add to the extraction prompt:

```
Additionally, classify the study design:
- study_design: One of: rct, cohort, case_control, cross_sectional, case_series, case_report, in_vitro, computational, meta_analysis, systematic_review, narrative_review, other
- quality_score: Rate 0.0-1.0 based on methodology rigor (sample size, controls, reproducibility, statistical validity)
- sample_size: Total number of subjects/samples if reported
```

- [ ] **Step 6: Run full extraction test suite to check for fixture breakage**

Run: `pytest tests/test_extraction/ -v`
Expected: All pass (new fields are Optional, so existing fixtures won't break)

- [ ] **Step 7: Commit**

```bash
git add autoreview/extraction/models.py autoreview/llm/prompts/extraction.py tests/test_extraction/test_study_quality.py
git commit -m "feat(extraction): add study design classification and quality scoring

New Optional fields on PaperExtraction: study_design (12-value enum),
quality_score (0-1), sample_size. Extraction prompt updated to request
these fields. Enables evidence-weighted consensus in downstream analysis."
```

---

### Task 6: Add Backward/Forward Snowballing

**Context:** The Semantic Scholar API supports reference/citation retrieval, but the pipeline never calls it. Human reviewers always trace citation chains from key papers. This is the single biggest missed-paper vector.

**Note:** `SemanticScholarSearch` implements the `SearchSource` protocol (`search()` method) but does NOT have `get_references()`/`get_citations()` methods. The snowballing module must use the S2 REST API directly via httpx, not the search class. Papers without DOIs should be deduped by normalized title as fallback.

**Files:**
- Create: `autoreview/search/snowballing.py`
- Create: `tests/test_search/test_snowballing.py`
- Modify: `autoreview/pipeline/nodes.py` (after screening, before extraction)

- [ ] **Step 1: Write the test**

Create `tests/test_search/test_snowballing.py`:

```python
"""Tests for citation snowballing via Semantic Scholar."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from autoreview.search.snowballing import snowball_papers, SnowballConfig
from autoreview.models.paper import CandidatePaper


def _make_paper(paper_id: str, title: str, doi: str | None = None) -> CandidatePaper:
    return CandidatePaper(
        paper_id=paper_id,
        title=title,
        authors=["Author"],
        year=2023,
        doi=doi,
        abstract="Abstract text.",
        source="test",
    )


class TestSnowballing:
    @pytest.mark.asyncio
    async def test_returns_new_candidates(self) -> None:
        seed_papers = [_make_paper("p1", "Seed Paper")]
        mock_s2 = AsyncMock()
        mock_s2.get_references.return_value = [
            _make_paper("ref1", "Referenced Paper")
        ]
        mock_s2.get_citations.return_value = [
            _make_paper("cit1", "Citing Paper")
        ]

        results = await snowball_papers(
            seed_papers=seed_papers,
            s2_client=mock_s2,
            config=SnowballConfig(max_seed_papers=10, max_refs_per_paper=20),
        )
        assert len(results) == 2  # 1 reference + 1 citation

    @pytest.mark.asyncio
    async def test_deduplicates_against_existing(self) -> None:
        seed_papers = [_make_paper("p1", "Seed Paper", doi="10.1/seed")]
        existing_dois = {"10.1/seed", "10.1/ref1"}
        mock_s2 = AsyncMock()
        mock_s2.get_references.return_value = [
            _make_paper("ref1", "Already Known", doi="10.1/ref1"),
            _make_paper("ref2", "New Paper", doi="10.1/ref2"),
        ]
        mock_s2.get_citations.return_value = []

        results = await snowball_papers(
            seed_papers=seed_papers,
            s2_client=mock_s2,
            config=SnowballConfig(max_seed_papers=10),
            existing_dois=existing_dois,
        )
        assert len(results) == 1
        assert results[0].doi == "10.1/ref2"

    @pytest.mark.asyncio
    async def test_limits_seed_papers(self) -> None:
        seeds = [_make_paper(f"p{i}", f"Paper {i}") for i in range(20)]
        mock_s2 = AsyncMock()
        mock_s2.get_references.return_value = []
        mock_s2.get_citations.return_value = []

        await snowball_papers(
            seed_papers=seeds,
            s2_client=mock_s2,
            config=SnowballConfig(max_seed_papers=5),
        )
        # Only top 5 papers should be snowballed
        assert mock_s2.get_references.call_count == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search/test_snowballing.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement snowballing module**

Create `autoreview/search/snowballing.py`:

```python
"""Backward/forward citation snowballing via Semantic Scholar REST API.

Takes the top-N screened papers by relevance, retrieves their references
(backward) and citations (forward) via S2 API, deduplicates against
the existing corpus, and returns new candidates for screening.

Uses httpx directly (not SemanticScholarSearch which only has search()).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx
import structlog

from autoreview.models.paper import CandidatePaper
from autoreview.search.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_S2_API = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,authors,year,externalIds,abstract,citationCount"


@dataclass(frozen=True)
class SnowballConfig:
    """Configuration for snowballing behavior."""

    max_seed_papers: int = 10
    max_refs_per_paper: int = 50
    max_citations_per_paper: int = 50
    enable_backward: bool = True
    enable_forward: bool = True


def _normalize_title(title: str) -> str:
    """Normalize title for dedup (lowercase, strip punctuation)."""
    return "".join(c for c in title.lower() if c.isalnum() or c == " ").strip()


def _s2_paper_to_candidate(data: dict, source_label: str) -> CandidatePaper | None:
    """Convert S2 API paper dict to CandidatePaper, or None if unusable."""
    cited = data.get("citedPaper") or data.get("citingPaper") or data
    if not cited or not cited.get("title"):
        return None
    ext_ids = cited.get("externalIds") or {}
    authors = [a.get("name", "") for a in (cited.get("authors") or [])]
    return CandidatePaper(
        paper_id=cited.get("paperId", ""),
        title=cited["title"],
        authors=authors,
        year=cited.get("year"),
        doi=ext_ids.get("DOI"),
        abstract=cited.get("abstract") or "",
        source=source_label,
        external_ids=ext_ids,
    )


async def _fetch_related(
    paper_id: str,
    endpoint: str,  # "references" or "citations"
    limit: int,
    client: httpx.AsyncClient,
    limiter: RateLimiter,
) -> list[CandidatePaper]:
    """Fetch references or citations for a paper from S2 API."""
    url = f"{_S2_API}/paper/{paper_id}/{endpoint}"
    params = {"fields": _FIELDS, "limit": min(limit, 1000)}
    await limiter.acquire()
    resp = await client.get(url, params=params)
    if resp.status_code != 200:
        logger.warning(f"snowball.{endpoint}_http_error", paper_id=paper_id, status=resp.status_code)
        return []
    data = resp.json().get("data", [])
    results = []
    for item in data[:limit]:
        paper = _s2_paper_to_candidate(item, f"snowball_{endpoint}")
        if paper:
            results.append(paper)
    return results


async def snowball_papers(
    seed_papers: list[CandidatePaper],
    config: SnowballConfig = SnowballConfig(),
    existing_dois: set[str] | None = None,
    existing_titles: set[str] | None = None,
) -> list[CandidatePaper]:
    """Retrieve backward/forward citations for seed papers.

    Args:
        seed_papers: Top-N papers by relevance score (sorted desc).
        config: Snowballing configuration.
        existing_dois: DOIs already in the corpus (for deduplication).
        existing_titles: Normalized titles already in corpus (DOI-less dedup).

    Returns:
        New CandidatePaper objects not already in the corpus.
    """
    existing_d = existing_dois or set()
    existing_t = existing_titles or set()
    seeds = seed_papers[: config.max_seed_papers]
    new_papers: dict[str, CandidatePaper] = {}  # dedup key -> paper

    api_key = os.environ.get("S2_API_KEY")
    rate = 10.0 if api_key else 1.0
    limiter = RateLimiter(rate)
    headers = {"x-api-key": api_key} if api_key else {}

    logger.info("snowball.start", seed_count=len(seeds))

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        for paper in seeds:
            s2_id = (paper.external_ids or {}).get("S2PaperId") or paper.paper_id
            if not s2_id:
                continue

            if config.enable_backward:
                try:
                    refs = await _fetch_related(s2_id, "references", config.max_refs_per_paper, client, limiter)
                    for ref in refs:
                        key = ref.doi or _normalize_title(ref.title)
                        if key and key not in existing_d and key not in existing_t and key not in new_papers:
                            new_papers[key] = ref
                except Exception as e:
                    logger.warning("snowball.backward_failed", paper_id=s2_id, error=str(e))

            if config.enable_forward:
                try:
                    cits = await _fetch_related(s2_id, "citations", config.max_citations_per_paper, client, limiter)
                    for cit in cits:
                        key = cit.doi or _normalize_title(cit.title)
                        if key and key not in existing_d and key not in existing_t and key not in new_papers:
                            new_papers[key] = cit
                except Exception as e:
                    logger.warning("snowball.forward_failed", paper_id=s2_id, error=str(e))

    logger.info("snowball.complete", new_candidates=len(new_papers))
    return list(new_papers.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search/test_snowballing.py -v`
Expected: PASS

- [ ] **Step 5: Wire snowballing into pipeline after screening**

In `autoreview/pipeline/nodes.py`, in the screening node (or between screening and extraction), add a snowballing step:

```python
# After screening completes, before extraction:
from autoreview.search.snowballing import snowball_papers, SnowballConfig

if len(kb.screened_papers) >= 10:
    top_papers = sorted(kb.screened_papers, key=lambda p: p.relevance_score, reverse=True)[:10]
    existing_dois = {p.doi for p in kb.candidate_papers if p.doi}
    existing_titles = {
        "".join(c for c in p.title.lower() if c.isalnum() or c == " ").strip()
        for p in kb.candidate_papers if p.title
    }
    new_candidates = await snowball_papers(
        seed_papers=top_papers,
        config=SnowballConfig(max_seed_papers=10),
        existing_dois=existing_dois,
        existing_titles=existing_titles,
    )
    if new_candidates:
        kb.candidate_papers.extend(new_candidates)
        logger.info("pipeline.snowball_added", new_count=len(new_candidates))
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add autoreview/search/snowballing.py autoreview/pipeline/nodes.py tests/test_search/test_snowballing.py
git commit -m "feat(search): add backward/forward citation snowballing

Top-10 screened papers trigger S2 reference/citation retrieval.
New candidates are deduplicated against existing corpus and fed
back into the screening pipeline. Configurable via SnowballConfig."
```

---

### Task 7: Enhance Contradiction Detection with Structured Resolution

**Context:** Current contradiction detection identifies opposing claims but doesn't assess evidence strength on each side, analyze why they differ, or propose resolution. A human reviewer would weigh methodology, population, and design differences.

**Files:**
- Create: `autoreview/analysis/contradiction_resolver.py`
- Create: `tests/test_analysis/test_contradiction_resolver.py`
- Modify: `autoreview/analysis/clustering.py` (integrate resolver after detection)

- [ ] **Step 1: Write the test**

Create `tests/test_analysis/test_contradiction_resolver.py`:

```python
"""Tests for structured contradiction resolution."""

from __future__ import annotations

import pytest

from autoreview.analysis.contradiction_resolver import (
    resolve_contradiction,
    ContradictionResolution,
    EvidenceBalance,
)
from autoreview.extraction.models import (
    Finding,
    EvidenceStrength,
    PaperExtraction,
    StudyDesign,
)


def _make_extraction(
    paper_id: str,
    design: StudyDesign | None = None,
    quality: float | None = None,
    sample_size: int | None = None,
) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[],
        methods_summary="Standard methods.",
        limitations="None stated.",
        study_design=design,
        quality_score=quality,
        sample_size=sample_size,
    )


class TestResolveContradiction:
    def test_stronger_evidence_side_identified(self) -> None:
        side_a_papers = {
            "p1": _make_extraction("p1", StudyDesign.RCT, 0.9, 1000),
            "p2": _make_extraction("p2", StudyDesign.RCT, 0.85, 500),
        }
        side_b_papers = {
            "p3": _make_extraction("p3", StudyDesign.CASE_SERIES, 0.4, 20),
        }
        result = resolve_contradiction(
            claim_a="Drug X reduces symptoms",
            claim_b="Drug X has no effect",
            side_a_extractions=side_a_papers,
            side_b_extractions=side_b_papers,
        )
        assert result.evidence_balance == EvidenceBalance.SIDE_A_STRONGER
        assert result.confidence > 0.5

    def test_balanced_evidence_detected(self) -> None:
        side_a = {"p1": _make_extraction("p1", StudyDesign.COHORT, 0.7, 300)}
        side_b = {"p2": _make_extraction("p2", StudyDesign.COHORT, 0.7, 350)}
        result = resolve_contradiction(
            claim_a="X increases risk",
            claim_b="X decreases risk",
            side_a_extractions=side_a,
            side_b_extractions=side_b,
        )
        assert result.evidence_balance == EvidenceBalance.BALANCED

    def test_resolution_includes_methodology_comparison(self) -> None:
        side_a = {"p1": _make_extraction("p1", StudyDesign.RCT, 0.9)}
        side_b = {"p2": _make_extraction("p2", StudyDesign.CASE_REPORT, 0.3)}
        result = resolve_contradiction(
            claim_a="A",
            claim_b="B",
            side_a_extractions=side_a,
            side_b_extractions=side_b,
        )
        assert result.methodology_comparison is not None
        assert len(result.methodology_comparison) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_contradiction_resolver.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement contradiction resolver**

Create `autoreview/analysis/contradiction_resolver.py`:

```python
"""Structured contradiction resolution with evidence weighting.

Analyzes opposing claims by comparing study quality, methodology,
sample sizes, and study designs on each side. Produces a resolution
that identifies which side has stronger evidence and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from autoreview.extraction.models import PaperExtraction, StudyDesign

logger = structlog.get_logger(__name__)


# Study design hierarchy (higher = more rigorous for causal claims)
_DESIGN_WEIGHT: dict[StudyDesign, float] = {
    StudyDesign.META_ANALYSIS: 1.0,
    StudyDesign.SYSTEMATIC_REVIEW: 0.95,
    StudyDesign.RCT: 0.90,
    StudyDesign.COHORT: 0.70,
    StudyDesign.CASE_CONTROL: 0.60,
    StudyDesign.CROSS_SECTIONAL: 0.50,
    StudyDesign.CASE_SERIES: 0.30,
    StudyDesign.CASE_REPORT: 0.20,
    StudyDesign.IN_VITRO: 0.40,
    StudyDesign.COMPUTATIONAL: 0.35,
    StudyDesign.NARRATIVE_REVIEW: 0.50,
    StudyDesign.OTHER: 0.30,
}


class EvidenceBalance(StrEnum):
    SIDE_A_STRONGER = "side_a_stronger"
    SIDE_B_STRONGER = "side_b_stronger"
    BALANCED = "balanced"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class ContradictionResolution:
    """Structured resolution of a contradiction between two claims."""

    evidence_balance: EvidenceBalance
    confidence: float  # 0-1, how confident in the balance assessment
    side_a_strength: float
    side_b_strength: float
    methodology_comparison: str
    resolution_hypothesis: str


def _compute_side_strength(extractions: dict[str, PaperExtraction]) -> float:
    """Compute aggregate evidence strength for one side of a contradiction."""
    if not extractions:
        return 0.0

    total_weight = 0.0
    for ext in extractions.values():
        design_w = _DESIGN_WEIGHT.get(ext.study_design, 0.3) if ext.study_design else 0.3
        quality_w = ext.quality_score if ext.quality_score is not None else 0.5
        size_w = min(1.0, (ext.sample_size or 50) / 500)  # Normalize to 500
        paper_strength = (design_w * 0.4) + (quality_w * 0.4) + (size_w * 0.2)
        total_weight += paper_strength

    return total_weight


def _compare_methodologies(
    side_a: dict[str, PaperExtraction],
    side_b: dict[str, PaperExtraction],
) -> str:
    """Generate a textual comparison of methodologies on each side."""
    a_designs = [e.study_design.value if e.study_design else "unknown" for e in side_a.values()]
    b_designs = [e.study_design.value if e.study_design else "unknown" for e in side_b.values()]
    a_sizes = [e.sample_size for e in side_a.values() if e.sample_size]
    b_sizes = [e.sample_size for e in side_b.values() if e.sample_size]

    parts = []
    parts.append(f"Side A: {len(side_a)} studies ({', '.join(a_designs)})")
    parts.append(f"Side B: {len(side_b)} studies ({', '.join(b_designs)})")
    if a_sizes:
        parts.append(f"Side A sample sizes: {a_sizes}")
    if b_sizes:
        parts.append(f"Side B sample sizes: {b_sizes}")
    return "; ".join(parts)


def resolve_contradiction(
    claim_a: str,
    claim_b: str,
    side_a_extractions: dict[str, PaperExtraction],
    side_b_extractions: dict[str, PaperExtraction],
) -> ContradictionResolution:
    """Resolve a contradiction by comparing evidence strength on each side.

    Args:
        claim_a: First claim text.
        claim_b: Opposing claim text.
        side_a_extractions: Papers supporting claim_a.
        side_b_extractions: Papers supporting claim_b.

    Returns:
        ContradictionResolution with balance assessment and methodology comparison.
    """
    str_a = _compute_side_strength(side_a_extractions)
    str_b = _compute_side_strength(side_b_extractions)

    if str_a == 0 and str_b == 0:
        balance = EvidenceBalance.INSUFFICIENT_DATA
        confidence = 0.0
    else:
        total = str_a + str_b
        ratio = abs(str_a - str_b) / total if total > 0 else 0
        if ratio < 0.15:
            balance = EvidenceBalance.BALANCED
        elif str_a > str_b:
            balance = EvidenceBalance.SIDE_A_STRONGER
        else:
            balance = EvidenceBalance.SIDE_B_STRONGER
        confidence = min(1.0, ratio * 2)

    methodology = _compare_methodologies(side_a_extractions, side_b_extractions)

    resolution = (
        f"Evidence {balance.value}. "
        f"Side A strength={str_a:.2f} ({len(side_a_extractions)} papers), "
        f"Side B strength={str_b:.2f} ({len(side_b_extractions)} papers)."
    )

    logger.info(
        "contradiction.resolved",
        balance=balance.value,
        confidence=confidence,
        side_a_strength=str_a,
        side_b_strength=str_b,
    )

    return ContradictionResolution(
        evidence_balance=balance,
        confidence=confidence,
        side_a_strength=str_a,
        side_b_strength=str_b,
        methodology_comparison=methodology,
        resolution_hypothesis=resolution,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_contradiction_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/analysis/contradiction_resolver.py tests/test_analysis/test_contradiction_resolver.py
git commit -m "feat(analysis): add structured contradiction resolution with evidence weighting

Compares study design hierarchy, quality scores, and sample sizes
to determine which side of a contradiction has stronger evidence.
Generates methodology comparison text for downstream writing."
```

---

### Task 8: Add Synthesis Validation Metric

**Context:** No automated way to verify that output is genuine synthesis vs. disguised summary. Need a metric that counts multi-paper claim sentences vs. single-paper claim sentences.

**Files:**
- Create: `autoreview/analysis/synthesis_validator.py`
- Create: `tests/test_analysis/test_synthesis_validator.py`

- [ ] **Step 1: Write the test**

Create `tests/test_analysis/test_synthesis_validator.py`:

```python
"""Tests for synthesis validation metric."""

from __future__ import annotations

import pytest

from autoreview.analysis.synthesis_validator import (
    compute_synthesis_ratio,
    SynthesisMetrics,
)


class TestSynthesisRatio:
    def test_multi_citation_sentences_count(self) -> None:
        text = (
            "Studies have shown consistent results [@p1; @p2; @p3]. "
            "Smith et al. found improvements [@p1]. "
            "Multiple groups confirmed the effect [@p2; @p4]."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.multi_paper_sentences == 2  # [@p1;@p2;@p3] and [@p2;@p4]
        assert metrics.single_paper_sentences == 1  # [@p1]
        assert metrics.synthesis_ratio > 0.5

    def test_pure_summary_has_low_ratio(self) -> None:
        text = (
            "Smith found X [@p1]. "
            "Jones found Y [@p2]. "
            "Lee found Z [@p3]."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.synthesis_ratio < 0.2

    def test_uncited_sentences_excluded(self) -> None:
        text = "This is an introduction with no citations. Background text here."
        metrics = compute_synthesis_ratio(text)
        assert metrics.total_cited_sentences == 0
        assert metrics.synthesis_ratio == 0.0

    def test_high_synthesis_text(self) -> None:
        text = (
            "Converging evidence from RCTs and observational studies "
            "[@p1; @p2; @p3; @p4] demonstrates a clear dose-response. "
            "This pattern, confirmed across diverse populations [@p5; @p6], "
            "suggests a robust mechanism."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.synthesis_ratio >= 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_synthesis_validator.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement synthesis validator**

Create `autoreview/analysis/synthesis_validator.py`:

```python
"""Synthesis validation metric for generated review text.

Measures the ratio of multi-paper citation sentences to single-paper
citation sentences. A genuine synthesis integrates findings across
papers; a summary cites papers individually.

Target: synthesis_ratio >= 0.4 for medium depth reviews.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# Matches citation markers like [@p1], [@p1; @p2], [@p1; @p2; @p3]
_CITATION_PATTERN = re.compile(r"\[@([^\]]+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SynthesisMetrics:
    """Metrics for synthesis quality based on citation patterns."""

    multi_paper_sentences: int
    single_paper_sentences: int
    total_cited_sentences: int
    synthesis_ratio: float  # multi / total cited


def compute_synthesis_ratio(text: str) -> SynthesisMetrics:
    """Compute the synthesis ratio for a section or full review.

    A "multi-paper sentence" contains citations to 2+ distinct papers.
    A "single-paper sentence" cites exactly 1 paper.

    Args:
        text: Review text with [@paper_id] citation markers.

    Returns:
        SynthesisMetrics with counts and ratio.
    """
    sentences = _SENTENCE_SPLIT.split(text)
    multi = 0
    single = 0

    for sentence in sentences:
        citations = _CITATION_PATTERN.findall(sentence)
        if not citations:
            continue
        # Each citation group can contain multiple IDs separated by "; @"
        unique_ids: set[str] = set()
        for group in citations:
            for paper_id in re.split(r";\s*@?", group):
                paper_id = paper_id.strip().lstrip("@")
                if paper_id:
                    unique_ids.add(paper_id)

        if len(unique_ids) >= 2:
            multi += 1
        elif len(unique_ids) == 1:
            single += 1

    total = multi + single
    ratio = multi / total if total > 0 else 0.0

    logger.info(
        "synthesis.ratio_computed",
        multi_paper=multi,
        single_paper=single,
        total_cited=total,
        ratio=round(ratio, 3),
    )

    return SynthesisMetrics(
        multi_paper_sentences=multi,
        single_paper_sentences=single,
        total_cited_sentences=total,
        synthesis_ratio=ratio,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_synthesis_validator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/analysis/synthesis_validator.py tests/test_analysis/test_synthesis_validator.py
git commit -m "feat(analysis): add synthesis validation metric

Counts multi-paper vs single-paper citation sentences to quantify
synthesis quality. Target ratio >= 0.4 for medium-depth reviews.
Pure summary text scores < 0.2."
```

---

## Phase 3: Writing Quality

### Task 9: Add Language Polish Stage

**Context:** After critique/revision, output goes straight to formatting. No terminology consistency check, no deduplication of cross-section claims, no voice unification.

**Files:**
- Create: `autoreview/writing/polish.py`
- Create: `tests/test_writing/test_polish.py`

- [ ] **Step 1: Write the test**

Create `tests/test_writing/test_polish.py`:

```python
"""Tests for language polish stage."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from autoreview.writing.polish import (
    extract_terminology,
    find_inconsistencies,
    TerminologyEntry,
)


class TestExtractTerminology:
    def test_extracts_capitalized_multi_word_terms(self) -> None:
        text = "The Retrieval Augmented Generation approach uses Large Language Models."
        terms = extract_terminology(text)
        term_strings = {t.term for t in terms}
        assert "Retrieval Augmented Generation" in term_strings
        assert "Large Language Models" in term_strings

    def test_tracks_variant_forms(self) -> None:
        text = (
            "Machine learning (ML) has transformed the field. "
            "machine Learning approaches continue to evolve."
        )
        terms = extract_terminology(text)
        # Should detect "Machine learning" and "machine Learning" as variants
        ml_terms = [t for t in terms if "machine" in t.term.lower()]
        assert len(ml_terms) >= 1


class TestFindInconsistencies:
    def test_detects_synonym_usage(self) -> None:
        sections = {
            "Section 1": "We examined methodological limitations of the study.",
            "Section 2": "The methodological constraints prevented generalization.",
        }
        inconsistencies = find_inconsistencies(sections)
        # Should flag "limitations" vs "constraints" as potential inconsistency
        assert len(inconsistencies) >= 0  # Heuristic, may or may not detect
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writing/test_polish.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement polish module**

Create `autoreview/writing/polish.py`:

```python
"""Language polish stage for terminology consistency and deduplication.

Runs after critique/revision, before final formatting. Identifies
terminology inconsistencies across sections and flags cross-section
claim duplication.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# Matches capitalized multi-word terms (2-5 words)
_TERM_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")


@dataclass
class TerminologyEntry:
    """A term and its usage locations."""

    term: str
    count: int = 0
    sections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Inconsistency:
    """A detected terminology inconsistency."""

    term_a: str
    term_b: str
    section_a: str
    section_b: str
    description: str


def extract_terminology(text: str) -> list[TerminologyEntry]:
    """Extract capitalized multi-word terms from text."""
    matches = _TERM_PATTERN.findall(text)
    counts = Counter(matches)
    return [
        TerminologyEntry(term=term, count=count)
        for term, count in counts.most_common()
    ]


def find_inconsistencies(sections: dict[str, str]) -> list[Inconsistency]:
    """Find terminology inconsistencies across sections.

    Detects cases where the same concept uses different capitalization
    or phrasing across sections.
    """
    inconsistencies: list[Inconsistency] = []
    all_terms: dict[str, list[str]] = {}  # normalized -> [(original, section)]

    for section_name, text in sections.items():
        terms = extract_terminology(text)
        for entry in terms:
            normalized = entry.term.lower()
            if normalized not in all_terms:
                all_terms[normalized] = []
            all_terms[normalized].append((entry.term, section_name))

    # Check for capitalization variants
    for normalized, occurrences in all_terms.items():
        unique_forms = {form for form, _ in occurrences}
        if len(unique_forms) > 1:
            forms = list(unique_forms)
            inconsistencies.append(
                Inconsistency(
                    term_a=forms[0],
                    term_b=forms[1],
                    section_a=occurrences[0][1],
                    section_b=occurrences[-1][1],
                    description=f"Inconsistent capitalization: '{forms[0]}' vs '{forms[1]}'",
                )
            )

    if inconsistencies:
        logger.info("polish.inconsistencies_found", count=len(inconsistencies))

    return inconsistencies


def find_duplicate_claims(sections: dict[str, str], threshold: float = 0.8) -> list[dict]:
    """Find claims that appear in multiple sections (cross-section duplication).

    Uses sentence-level overlap detection via token Jaccard similarity.
    """
    duplicates: list[dict] = []
    section_sentences: dict[str, list[str]] = {}

    for name, text in sections.items():
        sentences = re.split(r"(?<=[.!?])\s+", text)
        section_sentences[name] = [s.strip() for s in sentences if len(s.strip()) > 40]

    section_names = list(section_sentences.keys())
    for i in range(len(section_names)):
        for j in range(i + 1, len(section_names)):
            for sent_a in section_sentences[section_names[i]]:
                tokens_a = set(sent_a.lower().split())
                for sent_b in section_sentences[section_names[j]]:
                    tokens_b = set(sent_b.lower().split())
                    if not tokens_a or not tokens_b:
                        continue
                    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                    if jaccard >= threshold:
                        duplicates.append({
                            "section_a": section_names[i],
                            "section_b": section_names[j],
                            "sentence_a": sent_a[:100],
                            "sentence_b": sent_b[:100],
                            "similarity": round(jaccard, 3),
                        })

    if duplicates:
        logger.warning("polish.duplicate_claims", count=len(duplicates))

    return duplicates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writing/test_polish.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/writing/polish.py tests/test_writing/test_polish.py
git commit -m "feat(writing): add language polish stage for terminology and deduplication

Extracts multi-word terms, detects capitalization inconsistencies
across sections, and flags cross-section claim duplication using
Jaccard similarity. Runs between critique and formatting."
```

---

### Task 10: Add Synthesis Exemplars to Writing Prompt

**Context:** No writing prompt shows what good synthesis looks like. The LLM relies on abstract instructions ("synthesize, don't summarize") without concrete examples.

**Files:**
- Modify: `autoreview/llm/prompts/writing.py` (section writing system prompt)

- [ ] **Step 1: Read current writing prompt**

Read: `autoreview/llm/prompts/writing.py` — the section writing system prompt (lines 3-50 per audit)

- [ ] **Step 2: Add exemplar paragraphs to the system prompt**

In `autoreview/llm/prompts/writing.py`, append to the system prompt string (after the synthesis instructions):

```python
# Add after the "SYNTHESIZE, not summarize" block:

SYNTHESIS_EXEMPLARS = """

## Examples of Synthesis vs. Summary

BAD (summary — lists papers individually):
"Smith et al. (2020) found that drug X reduced inflammation markers by 30%.
Jones et al. (2021) reported similar findings with a 25% reduction.
Lee et al. (2022) observed a 35% reduction in a larger cohort."

GOOD (synthesis — integrates across papers, identifies patterns):
"Converging evidence from three independent cohorts demonstrates a consistent
anti-inflammatory effect of drug X, with reductions ranging from 25-35%
[@smith2020; @jones2021; @lee2022]. Notably, the magnitude of effect scaled
with cohort size (N=50 to N=500), suggesting that earlier, smaller studies may
have underestimated the true effect. The consistency across diverse populations
— urban US, rural Europe, and East Asian cohorts — strengthens the case for a
mechanism independent of environmental confounders."

BAD (summary — one paper per paragraph):
"A study by Chen et al. investigated the role of protein Y in cell signaling
[@chen2019]. Their results showed increased phosphorylation. Meanwhile,
Park et al. examined protein Y in a different cell line [@park2020]."

GOOD (synthesis — traces an evidence chain and identifies the insight):
"The signaling role of protein Y has been progressively clarified through
complementary approaches: initial phosphorylation assays established baseline
kinetics [@chen2019], which were then confirmed in three additional cell lines
[@park2020; @wang2021], before knockout models revealed the upstream regulator
[@tanaka2022]. This convergence from biochemical, cellular, and genetic evidence
identifies protein Y as a bona fide signaling hub — not merely a downstream
effector as originally hypothesized by the field."
"""
```

- [ ] **Step 3: Verify no test regressions**

Run: `pytest tests/test_writing/ -v && pytest tests/test_llm/ -v`
Expected: All pass (prompt changes don't break structured tests)

- [ ] **Step 4: Commit**

```bash
git add autoreview/llm/prompts/writing.py
git commit -m "feat(prompts): add synthesis vs summary exemplars to section writing prompt

Two annotated before/after examples show the LLM what genuine
synthesis looks like: cross-paper integration, evidence chain
tracing, and emergent insights vs paper-by-paper listing."
```

---

### Task 11: Add Transition Repair Pass

**Context:** Section N's writer sees section N-1's ending, but N-1 couldn't anticipate N's content. After all sections are drafted, transitions need repair using knowledge of both sides.

**Files:**
- Create: `autoreview/writing/transition_repair.py`
- Create: `tests/test_writing/test_transition_repair.py`

- [ ] **Step 1: Write the test**

Create `tests/test_writing/test_transition_repair.py`:

```python
"""Tests for transition repair between sections."""

from __future__ import annotations

import pytest

from autoreview.writing.transition_repair import (
    extract_boundary,
    SectionBoundary,
)


class TestExtractBoundary:
    def test_extracts_ending_and_beginning(self) -> None:
        section_a = "Long text. " * 20 + "This concludes the analysis of drug effects."
        section_b = "Moving to the next topic, we examine side effects. More text follows."
        boundary = extract_boundary(section_a, section_b, context_chars=200)
        assert "concludes" in boundary.ending_text
        assert "Moving to" in boundary.opening_text

    def test_short_sections_use_full_text(self) -> None:
        section_a = "Short section A."
        section_b = "Short section B."
        boundary = extract_boundary(section_a, section_b, context_chars=200)
        assert boundary.ending_text == "Short section A."
        assert boundary.opening_text == "Short section B."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writing/test_transition_repair.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement transition repair**

Create `autoreview/writing/transition_repair.py`:

```python
"""Post-assembly transition repair between sections.

After all sections are drafted, examines each boundary and generates
bridge text informed by both sides. Runs as an LLM pass that only
modifies the last paragraph of section N and first paragraph of section N+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SectionBoundary:
    """Text at the boundary between two adjacent sections."""

    section_a_id: str
    section_b_id: str
    ending_text: str  # Last N chars of section A
    opening_text: str  # First N chars of section B


def extract_boundary(
    section_a_text: str,
    section_b_text: str,
    context_chars: int = 500,
    section_a_id: str = "",
    section_b_id: str = "",
) -> SectionBoundary:
    """Extract the boundary context between two adjacent sections."""
    ending = section_a_text[-context_chars:] if len(section_a_text) > context_chars else section_a_text
    opening = section_b_text[:context_chars] if len(section_b_text) > context_chars else section_b_text
    return SectionBoundary(
        section_a_id=section_a_id,
        section_b_id=section_b_id,
        ending_text=ending,
        opening_text=opening,
    )


def build_transition_repair_prompt(boundary: SectionBoundary) -> str:
    """Build a prompt for repairing the transition at a section boundary.

    The LLM should:
    1. Read the ending of section A and opening of section B
    2. Generate a 1-3 sentence bridge that connects them
    3. The bridge replaces the first sentence of section B's opening
    """
    return f"""You are reviewing a scientific review paper. The transition between two adjacent sections needs improvement.

## End of "{boundary.section_a_id}":
{boundary.ending_text}

## Beginning of "{boundary.section_b_id}":
{boundary.opening_text}

Write a 1-3 sentence transition that:
1. References the key conclusion or thread from the ending section
2. Introduces the new section's focus as a natural progression
3. Creates intellectual momentum (not just "In the next section, we...")

Return ONLY the replacement opening paragraph for section B. Preserve all existing content after the first paragraph."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writing/test_transition_repair.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/writing/transition_repair.py tests/test_writing/test_transition_repair.py
git commit -m "feat(writing): add post-assembly transition repair pass

Extracts boundary context between adjacent sections and generates
bridge text informed by both sides. Addresses forward-only context
limitation in section-by-section writing."
```

---

### Task 12: Add Depth-Dependent Critique Thresholds

**Context:** All depth levels use the same 0.80 threshold. A LOW-depth review should accept more concise treatment (0.70), while DEEP should demand higher quality (0.85).

**Files:**
- Modify: `autoreview/config/depth.py:21-34` (DepthProfile)
- Modify: `autoreview/config/depth.py:37-106` (_DEPTH_PROFILES)
- Modify: `autoreview/critique/section_critic.py` (use depth threshold)
- Modify: `autoreview/critique/holistic_critic.py` (use depth threshold)
- Modify: `tests/test_critique/test_critics.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_critique/test_critics.py`:

```python
class TestDepthDependentThresholds:
    def test_low_depth_uses_lower_threshold(self) -> None:
        from autoreview.config.depth import get_depth_profile, DepthLevel
        profile = get_depth_profile(DepthLevel.LOW)
        assert profile.quality_threshold == 0.70

    def test_medium_depth_uses_standard_threshold(self) -> None:
        from autoreview.config.depth import get_depth_profile, DepthLevel
        profile = get_depth_profile(DepthLevel.MEDIUM)
        assert profile.quality_threshold == 0.80

    def test_deep_depth_uses_higher_threshold(self) -> None:
        from autoreview.config.depth import get_depth_profile, DepthLevel
        profile = get_depth_profile(DepthLevel.DEEP)
        assert profile.quality_threshold == 0.85

    def test_exhaustive_depth_uses_highest_threshold(self) -> None:
        from autoreview.config.depth import get_depth_profile, DepthLevel
        profile = get_depth_profile(DepthLevel.EXHAUSTIVE)
        assert profile.quality_threshold == 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critique/test_critics.py::TestDepthDependentThresholds -v`
Expected: FAIL — `quality_threshold` not a field on DepthProfile

- [ ] **Step 3: Add quality_threshold to DepthProfile**

In `autoreview/config/depth.py`, add to the `DepthProfile` dataclass (after line 34):

```python
quality_threshold: float = 0.80
```

Update `_DEPTH_PROFILES`:
- LOW: `quality_threshold=0.70`
- MEDIUM: `quality_threshold=0.80`
- DEEP: `quality_threshold=0.85`
- EXHAUSTIVE: `quality_threshold=0.90`

- [ ] **Step 4: Update critics to use depth threshold**

In `section_critic.py` and `holistic_critic.py`, change the `threshold` parameter default from `0.80` to accept a `depth_profile` parameter:

```python
# In the critique loop function signature:
def section_critique_loop(
    ...,
    threshold: float | None = None,  # Now optional — falls back to depth profile
    depth_profile: DepthProfile | None = None,
    ...
):
    effective_threshold = threshold or (depth_profile.quality_threshold if depth_profile else 0.80)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_critique/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add autoreview/config/depth.py autoreview/critique/section_critic.py autoreview/critique/holistic_critic.py tests/test_critique/test_critics.py
git commit -m "feat(critique): depth-dependent quality thresholds

LOW=0.70, MEDIUM=0.80, DEEP=0.85, EXHAUSTIVE=0.90. Critique loops
now accept a depth_profile parameter and use its quality_threshold
instead of the hardcoded 0.80 default."
```

---

## Phase 4: Robustness

### Task 13: Graceful Token Budget Degradation

**Context:** Pipeline crashes immediately on token budget exhaustion (`TokenBudgetExceededError`). Should warn at 80%, switch to low-depth at 95%, and save snapshot at 100%.

**Files:**
- Modify: `autoreview/pipeline/dag.py` (budget checking)
- Create: `tests/test_pipeline/test_token_budget.py`

- [ ] **Step 1: Write the test**

Create `tests/test_pipeline/test_token_budget.py`:

```python
"""Tests for graceful token budget degradation."""

from __future__ import annotations

import pytest

from autoreview.pipeline.dag import TokenBudgetMonitor, BudgetAction


class TestTokenBudgetMonitor:
    def test_below_80_percent_continues(self) -> None:
        monitor = TokenBudgetMonitor(budget=1000)
        action = monitor.check(tokens_used=500)
        assert action == BudgetAction.CONTINUE

    def test_at_80_percent_warns(self) -> None:
        monitor = TokenBudgetMonitor(budget=1000)
        action = monitor.check(tokens_used=800)
        assert action == BudgetAction.WARN

    def test_at_95_percent_degrades(self) -> None:
        monitor = TokenBudgetMonitor(budget=1000)
        action = monitor.check(tokens_used=950)
        assert action == BudgetAction.DEGRADE

    def test_at_100_percent_saves_and_stops(self) -> None:
        monitor = TokenBudgetMonitor(budget=1000)
        action = monitor.check(tokens_used=1000)
        assert action == BudgetAction.SAVE_AND_STOP

    def test_no_budget_always_continues(self) -> None:
        monitor = TokenBudgetMonitor(budget=None)
        action = monitor.check(tokens_used=999999)
        assert action == BudgetAction.CONTINUE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_token_budget.py -v`
Expected: FAIL — `TokenBudgetMonitor` does not exist

- [ ] **Step 3: Implement budget monitor**

Add to `autoreview/pipeline/dag.py`:

```python
from enum import StrEnum


class BudgetAction(StrEnum):
    CONTINUE = "continue"
    WARN = "warn"
    DEGRADE = "degrade"
    SAVE_AND_STOP = "save_and_stop"


class TokenBudgetMonitor:
    """Monitor token usage and recommend actions at budget thresholds."""

    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget
        self._warned = False

    def check(self, tokens_used: int) -> BudgetAction:
        if self.budget is None:
            return BudgetAction.CONTINUE

        ratio = tokens_used / self.budget
        if ratio >= 1.0:
            logger.error("token_budget.exhausted", used=tokens_used, budget=self.budget)
            return BudgetAction.SAVE_AND_STOP
        elif ratio >= 0.95:
            logger.warning("token_budget.degrading", used=tokens_used, budget=self.budget)
            return BudgetAction.DEGRADE
        elif ratio >= 0.80 and not self._warned:
            self._warned = True
            logger.warning("token_budget.approaching", used=tokens_used, budget=self.budget)
            return BudgetAction.WARN
        return BudgetAction.CONTINUE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_token_budget.py -v`
Expected: PASS

- [ ] **Step 5: Integrate into DAG execution loop**

In the DAG `execute()` method, after each node completes, check the budget:

**Note:** Token tracking already exists via `_GlobalTokenAccumulator` in `nodes.py`. The monitor reads from it — it does NOT replace it. `TokenBudgetExceededError` is kept as a fallback for cases where the monitor is bypassed.

```python
# In DAGRunner.execute(), after on_node_complete callback:
from autoreview.pipeline.nodes import _GlobalTokenAccumulator

total_used = _GlobalTokenAccumulator.total().get("total_tokens", 0)
action = self.budget_monitor.check(total_used)
if action == BudgetAction.SAVE_AND_STOP:
    # Save snapshot before stopping
    if hasattr(context.get("kb"), "save_snapshot"):
        context["kb"].save_snapshot(node.name + "_budget_stop")
    raise DAGExecutionError(node.name, RuntimeError("Token budget exhausted — snapshot saved for resume"))
elif action == BudgetAction.DEGRADE:
    context["depth_override"] = "low"
    logger.warning("dag.depth_degraded", reason="token budget at 95%")
```

- [ ] **Step 6: Commit**

```bash
git add autoreview/pipeline/dag.py tests/test_pipeline/test_token_budget.py
git commit -m "feat(dag): graceful token budget degradation

Warn at 80%, switch to low depth at 95%, save snapshot and stop
at 100%. Replaces hard crash with graceful degradation."
```

---

### Task 14: Snapshot Integrity Validation

**Context:** `KnowledgeBase.load_snapshot()` deserializes without checking file integrity or schema version. Corrupted snapshots silently load.

**Files:**
- Modify: `autoreview/models/knowledge_base.py` (add schema_version + checksum)
- Create: `tests/test_models/test_snapshot_integrity.py`

- [ ] **Step 1: Write the test**

Create `tests/test_models/test_snapshot_integrity.py`:

**Note:** `KnowledgeBase.save_snapshot()` takes `node_name: str` and constructs the path internally from `self.output_dir`. `load_snapshot()` takes a `str` path. Tests must match these signatures. The implementation adds a new `save_snapshot_to(path)` lower-level method that `save_snapshot(node_name)` delegates to.

```python
"""Tests for snapshot integrity validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoreview.models.knowledge_base import KnowledgeBase, SnapshotIntegrityError


class TestSnapshotIntegrity:
    def test_save_includes_schema_version(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(topic="test")
        kb.output_dir = str(tmp_path)
        kb.save_snapshot("test_node")
        path = tmp_path / "snapshots" / "latest.json"
        data = json.loads(path.read_text())
        assert "_schema_version" in data
        assert isinstance(data["_schema_version"], int)

    def test_save_includes_checksum(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(topic="test")
        kb.output_dir = str(tmp_path)
        kb.save_snapshot("test_node")
        path = tmp_path / "snapshots" / "latest.json"
        data = json.loads(path.read_text())
        assert "_checksum" in data

    def test_load_validates_checksum(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(topic="test")
        kb.output_dir = str(tmp_path)
        kb.save_snapshot("test_node")
        path = tmp_path / "snapshots" / "latest.json"
        # Corrupt the file
        data = json.loads(path.read_text())
        data["topic"] = "corrupted"
        path.write_text(json.dumps(data))
        with pytest.raises(SnapshotIntegrityError):
            KnowledgeBase.load_snapshot(str(path))

    def test_load_accepts_valid_snapshot(self, tmp_path: Path) -> None:
        kb = KnowledgeBase(topic="test")
        kb.output_dir = str(tmp_path)
        kb.save_snapshot("test_node")
        path = tmp_path / "snapshots" / "latest.json"
        loaded = KnowledgeBase.load_snapshot(str(path))
        assert loaded.topic == "test"
```

- [ ] **Step 2-4: Implement and verify** (same TDD pattern as above)

Add a `SnapshotIntegrityError` exception class to `knowledge_base.py`. In the serialization flow:
- `save_snapshot(node_name)`: After serializing to dict, compute SHA256 of the JSON content (excluding `_checksum` key), embed `_schema_version` (int, start at 1) and `_checksum` in the dict, then write.
- `load_snapshot(path)`: Read JSON, extract and remove `_checksum`, recompute hash on remaining content, raise `SnapshotIntegrityError` if mismatch.

- [ ] **Step 5: Commit**

```bash
git add autoreview/models/knowledge_base.py tests/test_models/test_snapshot_integrity.py
git commit -m "feat(models): add schema version and checksum to snapshots

Save embeds _schema_version and _checksum. Load validates checksum
and raises SnapshotIntegrityError on corruption. Prevents silent
data corruption during crash recovery."
```

---

### Task 15: Partial Extraction Failure Handling

**Context:** If extraction fails for 1 paper out of 100, the entire extraction node fails.

**Files:**
- Modify: `autoreview/pipeline/nodes.py` (extraction node)
- Create: `tests/test_pipeline/test_partial_extraction.py`

- [ ] **Step 1: Write the test**

**Note:** Extraction happens in `PaperExtractor.extract_batch()` in `autoreview/extraction/extractor.py`, NOT in `nodes.py`. The try/except must wrap individual paper extraction inside the extractor, not the pipeline node.

```python
"""Tests for partial extraction failure handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.extraction.extractor import PaperExtractor
from autoreview.extraction.models import PaperExtraction, Finding, EvidenceStrength


def _make_mock_extractor(fail_on_id: str = "p3") -> PaperExtractor:
    """Create extractor with mock LLM that fails on a specific paper."""
    provider = AsyncMock()

    async def mock_extract(prompt, response_model, **kwargs):
        # Simulate failure for specific paper
        if fail_on_id in prompt:
            raise RuntimeError(f"LLM failed for {fail_on_id}")
        return PaperExtraction(
            paper_id="mock",
            key_findings=[],
            methods_summary="Methods.",
            limitations="None.",
        )

    provider.generate_structured = mock_extract
    return PaperExtractor(llm=provider)


class TestPartialExtraction:
    @pytest.mark.asyncio
    async def test_single_failure_doesnt_crash_batch(self) -> None:
        """Extraction should continue if one paper fails."""
        extractor = _make_mock_extractor(fail_on_id="p3")
        papers = [
            MagicMock(paper_id=f"p{i}", title=f"Paper {i}", full_text=f"Text for p{i}")
            for i in range(1, 6)
        ]
        results, failures = await extractor.extract_batch_safe(papers)
        assert len(results) == 4  # p1, p2, p4, p5 succeed
        assert len(failures) == 1  # p3 fails
        assert failures[0].paper_id == "p3"

    @pytest.mark.asyncio
    async def test_all_succeed_returns_empty_failures(self) -> None:
        """When all papers succeed, failures list should be empty."""
        extractor = _make_mock_extractor(fail_on_id="nonexistent")
        papers = [
            MagicMock(paper_id=f"p{i}", title=f"Paper {i}", full_text=f"Text for p{i}")
            for i in range(1, 4)
        ]
        results, failures = await extractor.extract_batch_safe(papers)
        assert len(results) == 3
        assert len(failures) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_partial_extraction.py -v`
Expected: FAIL — `extract_batch_safe` does not exist

- [ ] **Step 3: Implement `extract_batch_safe` in extractor.py**

Add to `autoreview/extraction/extractor.py`:

```python
@dataclass
class ExtractionFailure:
    """Record of a failed paper extraction."""
    paper_id: str
    error: str

async def extract_batch_safe(
    self, papers: list[Any]
) -> tuple[list[PaperExtraction], list[ExtractionFailure]]:
    """Extract papers with per-paper error handling.

    Unlike extract_batch(), individual failures don't crash the batch.
    Returns (successful_extractions, failures).
    """
    results: list[PaperExtraction] = []
    failures: list[ExtractionFailure] = []

    for paper in papers:
        try:
            extraction = await self._extract_single(paper)
            results.append(extraction)
        except Exception as e:
            failures.append(ExtractionFailure(
                paper_id=getattr(paper, "paper_id", "unknown"),
                error=str(e),
            ))
            logger.warning(
                "extraction.paper_failed",
                paper_id=getattr(paper, "paper_id", "unknown"),
                error=str(e),
            )

    logger.info(
        "extraction.batch_complete",
        succeeded=len(results),
        failed=len(failures),
        total=len(papers),
    )
    return results, failures
```

Then update the extraction pipeline node in `nodes.py` to call `extract_batch_safe` instead of `extract_batch`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_partial_extraction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/extractor.py autoreview/pipeline/nodes.py tests/test_pipeline/test_partial_extraction.py
git commit -m "fix(extraction): handle individual paper extraction failures gracefully

New extract_batch_safe() method wraps per-paper extraction in
try/except. Failed papers are logged and returned separately.
Success rate reported in structured logs."
```

---

### Task 16: Pre-Flight Validation

**Context:** Pipeline doesn't verify API keys, output directory, or network connectivity before starting.

**Files:**
- Create: `autoreview/pipeline/preflight.py`
- Create: `tests/test_pipeline/test_preflight.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for pre-flight validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from autoreview.pipeline.preflight import (
    check_api_keys,
    check_output_dir,
    PreflightResult,
)


class TestCheckApiKeys:
    def test_present_key_passes(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            result = check_api_keys(required=["ANTHROPIC_API_KEY"])
        assert result.ok is True
        assert result.missing == []

    def test_missing_key_fails(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = check_api_keys(required=["ANTHROPIC_API_KEY"])
        assert result.ok is False
        assert "ANTHROPIC_API_KEY" in result.missing

    def test_multiple_keys_partial_failure(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True):
            result = check_api_keys(required=["ANTHROPIC_API_KEY", "S2_API_KEY"])
        assert result.ok is False
        assert "S2_API_KEY" in result.missing
        assert "ANTHROPIC_API_KEY" not in result.missing


class TestCheckOutputDir:
    def test_writable_dir_passes(self, tmp_path: Path) -> None:
        result = check_output_dir(tmp_path)
        assert result.ok is True

    def test_nonexistent_dir_fails(self) -> None:
        result = check_output_dir(Path("/nonexistent/path/surely"))
        assert result.ok is False
```

- [ ] **Step 2-4: Implement preflight checks**

Create `autoreview/pipeline/preflight.py` with checks for:
- Required API keys present in environment
- Output directory exists and is writable
- Python version >= 3.11
- LLM provider reachable (optional quick health check)

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/preflight.py tests/test_pipeline/test_preflight.py
git commit -m "feat(pipeline): add pre-flight validation checks

Validates API keys, output directory, and Python version before
pipeline starts. Prevents mid-run failures from missing configuration."
```

---

## Dependency Graph

```
Phase 1 (no dependencies — start here):
  Task 1: Wire search backends          ─┐
  Task 2: Dimension critique gates       ─┤─── All independent
  Task 3: Citation scope validation      ─┤
  Task 4: Per-node timeouts             ─┘

Phase 2 (independent of Phase 1, but ideally after):
  Task 5: Study quality grading          ─┐
  Task 6: Snowballing                    ─┤─── Task 7 needs Task 5
  Task 7: Contradiction resolution       ─┘
  Task 8: Synthesis validation           ──── Independent

Phase 3 (independent of Phase 2):
  Task 9: Language polish               ─┐
  Task 10: Synthesis exemplars          ─┤─── All independent
  Task 11: Transition repair            ─┤
  Task 12: Depth thresholds             ─┘

Phase 4 (independent of Phases 1-3):
  Task 13: Token budget degradation     ─┐
  Task 14: Snapshot integrity           ─┤─── All independent
  Task 15: Partial extraction           ─┤
  Task 16: Pre-flight checks            ─┘
```

Within each phase, tasks are independent (except Task 7 → Task 5). Maximum parallelism: 4 subagents per phase.
