---
name: add-pipeline-node
description: Step-by-step checklist for adding a new DAG node to the AutoReview pipeline
---

# Add Pipeline Node

Step-by-step checklist for adding a new DAG node to the AutoReview pipeline. Each step includes exact file paths, code patterns, and copy-pasteable templates.

---

## Current DAG Order (15 nodes)

| # | Node Name | Dependencies | PipelinePhase |
|---|-----------|-------------|---------------|
| 1 | `query_expansion` | *(none)* | `QUERY_EXPANSION` |
| 2 | `search` | `query_expansion` | `SEARCH` |
| 3 | `screening` | `search` | `SCREENING` |
| 4 | `full_text_retrieval` | `screening` | `FULL_TEXT_RETRIEVAL` |
| 5 | `extraction` | `full_text_retrieval` | `EXTRACTION` |
| 6 | `clustering` | `extraction` | `CLUSTERING` |
| 7 | `gap_search` | `clustering` | `GAP_SEARCH` |
| 8 | `outline` | `gap_search` | `OUTLINE` |
| 9 | `narrative_planning` | `outline` | `NARRATIVE_PLANNING` |
| 10 | `contextual_enrichment` | `narrative_planning` | `CONTEXTUAL_ENRICHMENT` |
| 11 | `corpus_expansion` | `contextual_enrichment` | `CORPUS_EXPANSION` |
| 12 | `section_writing` | `corpus_expansion` | `SECTION_WRITING` |
| 13 | `passage_search` | `section_writing` | `PASSAGE_SEARCH` |
| 14 | `assembly` | `passage_search` | *(uses HOLISTIC_CRITIQUE internally)* |
| 15 | `final_polish` | `assembly` | `FINAL_POLISH` |

DAG wiring is in `autoreview/pipeline/runner.py` inside `build_pipeline()`.

---

## Checklist

### Step 1: Add PipelinePhase enum value

**File:** `autoreview/models/knowledge_base.py`

Add a new value to the `PipelinePhase` enum. Place it in execution order relative to existing phases.

```python
class PipelinePhase(str, Enum):
    """Current phase of the review generation pipeline."""

    INITIALIZED = "initialized"
    QUERY_EXPANSION = "query_expansion"
    SEARCH = "search"
    SCREENING = "screening"
    FULL_TEXT_RETRIEVAL = "full_text_retrieval"
    EXTRACTION = "extraction"
    CLUSTERING = "clustering"
    OUTLINE = "outline"
    NARRATIVE_PLANNING = "narrative_planning"
    CONTEXTUAL_ENRICHMENT = "contextual_enrichment"
    CORPUS_EXPANSION = "corpus_expansion"
    GAP_SEARCH = "gap_search"
    SECTION_WRITING = "section_writing"
    SECTION_CRITIQUE = "section_critique"
    PASSAGE_SEARCH = "passage_search"
    # ADD NEW PHASE HERE in execution order:
    YOUR_NODE_NAME = "your_node_name"
    HOLISTIC_CRITIQUE = "holistic_critique"
    FINAL_POLISH = "final_polish"
    COMPLETE = "complete"
    FAILED = "failed"
```

---

### Step 2: Add KnowledgeBase fields (if needed)

**File:** `autoreview/models/knowledge_base.py`

If the node produces new data that downstream nodes consume, add a field to `KnowledgeBase`. Always use a Pydantic `Field` with a default value so existing snapshots remain loadable.

```python
class KnowledgeBase(TimestampedModel):
    # ... existing fields ...

    # ADD: new field with default so old snapshots deserialize safely
    your_node_results: dict[str, YourResultModel] = Field(default_factory=dict)
```

**Existing field patterns to follow:**
- `dict[str, T]` keyed by paper ID or section ID: `extractions`, `section_drafts`, `contextual_enrichment`, `corpus_expansion_results`
- `list[T]`: `candidate_papers`, `screened_papers`, `critique_history`, `comprehensiveness_checks`
- `T | None`: `evidence_map`, `outline`, `narrative_plan`, `full_draft`, `scope_document`

If the result type is a new Pydantic model, define it in the appropriate `models/` or analysis module and import it in `knowledge_base.py`.

---

### Step 3: Create prompt module

**File:** `autoreview/llm/prompts/<your_node_name>.py`

Every node that calls the LLM needs a prompt module. The pattern has two parts:
1. A system prompt constant (string)
2. A `build_*_prompt()` function that returns the user prompt

Optionally, define Pydantic response models for structured output in the same file.

**Template (based on `screening.py` pattern):**

```python
from __future__ import annotations

from autoreview.models.base import AutoReviewModel


class YourNodeResult(AutoReviewModel):
    """Structured response model for your node."""
    field_one: str
    field_two: list[str]


YOUR_NODE_SYSTEM_PROMPT = """\
You are an expert [role description]. \
[Instructions for the LLM about what to do and how to structure output.]
"""


def build_your_node_prompt(
    scope_document: str,
    evidence_map_summary: str,
    # ... other inputs the prompt needs
) -> str:
    """Build the user prompt for your node."""
    return f"""\
## Review Scope
{scope_document}

## Evidence Summary
{evidence_map_summary}

## Task
[Specific instructions for what to produce.]
"""
```

**Existing prompt modules for reference:**
- Simple batch processing: `autoreview/llm/prompts/screening.py` (55 lines)
- Complex with response models: `autoreview/llm/prompts/corpus_expansion.py` (91 lines)
- Multi-section: `autoreview/llm/prompts/enrichment.py` (167 lines)
- Clustering/analysis: `autoreview/llm/prompts/clustering.py` (178 lines)

---

### Step 4: Create node method

**File:** `autoreview/pipeline/nodes.py`

Add an `async def` method to the `PipelineNodes` class. Every node follows this exact pattern:

```python
async def your_node_name(self, kb: KnowledgeBase) -> None:
    """Node: Brief description of what this node does."""
    # 1. Create a token tracker (wraps self.llm for usage counting)
    tracker = _TokenAccumulator(self.llm, self._global_tokens)

    # 2. Import and instantiate helper classes with the tracker
    from autoreview.your_module import YourHelper
    helper = YourHelper(tracker)

    # 3. Read inputs from KnowledgeBase
    # (Access kb.evidence_map, kb.outline, kb.extractions, etc.)

    # 4. Do the work (call LLM via tracker, process results)
    result = await helper.do_work(...)

    # 5. Write results back to KnowledgeBase
    kb.your_node_results = result

    # 6. Update phase
    kb.current_phase = PipelinePhase.YOUR_NODE_NAME

    # 7. Add audit entry with token usage
    kb.add_audit_entry(
        "your_node_name",
        "complete",
        f"Summary: {len(result)} items processed",
        tracker.usage,
    )
```

**Key details:**
- **Signature**: `async def your_node_name(self, kb: KnowledgeBase) -> None` -- always returns `None`, mutates `kb` in place
- **`_TokenAccumulator`**: Always create one per node. Pass it to helper classes instead of `self.llm`. Access `tracker.usage` (returns `dict[str, int]` with keys: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`)
- **`self._global_tokens`**: The second arg to `_TokenAccumulator`. Enforces pipeline-wide token budget. Raises `TokenBudgetExceeded` if exceeded.
- **`self.config`**: Access `DomainConfig` for thresholds, batch sizes, etc.
- **`self.dispatcher`**: The `RemediationDispatcher` for comprehensiveness remediation.
- **Lazy imports**: Use local imports for heavy modules to avoid circular dependencies (see `query_expansion` node for example).

**Real example -- the `outline` node (lines 527-551 of nodes.py):**

```python
async def outline(self, kb: KnowledgeBase) -> None:
    """Node: Generate and critique the outline."""
    tracker = _TokenAccumulator(self.llm, self._global_tokens)
    generator = OutlineGenerator(tracker)
    critic = OutlineCritic(tracker)

    review_outline, critiques = await outline_critique_loop(
        llm=tracker,
        outline_generator=generator,
        outline_critic=critic,
        evidence_map=kb.evidence_map,
        scope_document=kb.scope_document or "",
        required_sections=self.config.outline.required_sections,
        max_cycles=self.config.outline.max_critique_cycles,
        threshold=self.config.critique.score_threshold,
    )

    kb.outline = review_outline.model_dump()
    kb.critique_history.extend(critiques)
    kb.current_phase = PipelinePhase.OUTLINE
    kb.add_audit_entry(
        "outline", "complete",
        f"Sections: {len(review_outline.sections)}",
        tracker.usage,
    )
```

---

### Step 5: Wire into DAG

**File:** `autoreview/pipeline/runner.py`

Add a `dag.add_node()` call in `build_pipeline()`. Place it in the correct position based on dependencies.

```python
def build_pipeline(llm: Any, config: DomainConfig) -> tuple[DAGRunner, PipelineNodes]:
    nodes = PipelineNodes(llm, config)
    dag = DAGRunner()

    dag.add_node("query_expansion", nodes.query_expansion)
    dag.add_node("search", nodes.search, dependencies=["query_expansion"])
    dag.add_node("screening", nodes.screening, dependencies=["search"])
    dag.add_node("full_text_retrieval", nodes.full_text_retrieval, dependencies=["screening"])
    dag.add_node("extraction", nodes.extraction, dependencies=["full_text_retrieval"])
    dag.add_node("clustering", nodes.clustering, dependencies=["extraction"])
    dag.add_node("gap_search", nodes.gap_search, dependencies=["clustering"])
    dag.add_node("outline", nodes.outline, dependencies=["gap_search"])
    dag.add_node("narrative_planning", nodes.narrative_planning, dependencies=["outline"])
    dag.add_node("contextual_enrichment", nodes.contextual_enrichment, dependencies=["narrative_planning"])
    dag.add_node("corpus_expansion", nodes.corpus_expansion, dependencies=["contextual_enrichment"])
    dag.add_node("section_writing", nodes.section_writing, dependencies=["corpus_expansion"])
    dag.add_node("passage_search", nodes.passage_search, dependencies=["section_writing"])
    dag.add_node("assembly", nodes.assembly, dependencies=["passage_search"])
    dag.add_node("final_polish", nodes.final_polish, dependencies=["assembly"])

    # ADD YOUR NODE: insert in the correct position
    # dag.add_node("your_node_name", nodes.your_node_name, dependencies=["predecessor_node"])
    # Then update any node that previously depended on "predecessor_node" to depend on "your_node_name"

    return dag, nodes
```

**DAGRunner.add_node() signature:**
```python
def add_node(
    self,
    name: str,                                    # Unique string identifier
    func: Callable[..., Coroutine[Any, Any, Any]], # The async method
    dependencies: list[str] | None = None,         # Names of prerequisite nodes
) -> None
```

**Important**: When inserting a node between two existing nodes, you must:
1. Add the new node with `dependencies=[previous_node]`
2. Update the next node's dependencies to point to your new node instead of the previous node

Example -- inserting `my_analysis` between `clustering` and `gap_search`:
```python
dag.add_node("clustering", nodes.clustering, dependencies=["extraction"])
dag.add_node("my_analysis", nodes.my_analysis, dependencies=["clustering"])  # NEW
dag.add_node("gap_search", nodes.gap_search, dependencies=["my_analysis"])   # CHANGED from "clustering"
```

---

### Step 6: Add comprehensiveness check (if applicable)

**File:** `autoreview/analysis/comprehensiveness.py`

If your node should validate its own output quality, add a checker class. Checkers return a `ComprehensiveCheckResult` which can optionally include a `RemediationAction`.

**Key models:**
```python
class CheckStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"

class RemediationAction(AutoReviewModel):
    action: str                                 # Handler name in RemediationDispatcher
    params: dict[str, Any] = Field(default_factory=dict)
    priority: int = 1

class ComprehensiveCheckResult(AutoReviewModel):
    check_name: str
    status: CheckStatus
    score: float = 0.0
    details: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    remediation: RemediationAction | None = None
```

**Checker class template:**
```python
class YourNodeChecker:
    """Validates output quality of your_node_name."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def check(
        self,
        # inputs your checker needs
        results: dict[str, Any],
    ) -> ComprehensiveCheckResult:
        # Compute quality metrics
        score = ...
        if score >= self.threshold:
            return ComprehensiveCheckResult(
                check_name="your_node_quality",
                status=CheckStatus.PASSED,
                score=score,
                details=f"Quality score {score:.2f} meets threshold {self.threshold}",
            )
        return ComprehensiveCheckResult(
            check_name="your_node_quality",
            status=CheckStatus.WARNING,
            score=score,
            details=f"Quality score {score:.2f} below threshold {self.threshold}",
            remediation=RemediationAction(
                action="your_remediation_action",
                params={"key": "value"},
            ),
        )
```

**Existing checkers for reference:**
- `QueryCoverageChecker` -- LLM-based, checks query coverage against scope
- `CoverageAnomalyChecker` -- rule-based, checks search/screening statistics
- `BorderlineRescreener` -- LLM-based, re-evaluates borderline papers
- `BenchmarkValidator` -- API-based, validates against known benchmark papers
- `PostGapRevalidator` -- rule-based, validates gap search results

**Wire the check into your node method:**
```python
# In your node method, after producing results:
checker = YourNodeChecker(threshold=0.75)
check_result = checker.check(results)
kb.comprehensiveness_checks.append(check_result)

if check_result.remediation:
    await self.dispatcher.execute(kb, check_result)
```

---

### Step 7: Add remediation action (if applicable)

**File:** `autoreview/pipeline/remediation.py`

If your comprehensiveness check prescribes a remediation action, register a handler in `RemediationDispatcher`.

**Current dispatch table (line ~65 of remediation.py):**
```python
handler = {
    "expand_queries": self._expand_queries,
    "retry_gap_search": self._retry_gap_search,
    "lower_screening_threshold": self._lower_screening_threshold,
}.get(action)
```

**To add a new action:**

1. Add handler method:
```python
async def _your_remediation_action(
    self, kb: KnowledgeBase, params: dict[str, Any],
) -> None:
    """Handle your_remediation_action remediation."""
    # Read params
    some_param = params.get("key", "default")
    # Perform remediation (re-run with adjusted parameters, expand data, etc.)
    ...
    # Log what was done
    kb.add_audit_entry(
        "remediation",
        "your_remediation_action",
        f"Details of what was fixed",
    )
```

2. Register in dispatch table:
```python
handler = {
    "expand_queries": self._expand_queries,
    "retry_gap_search": self._retry_gap_search,
    "lower_screening_threshold": self._lower_screening_threshold,
    "your_remediation_action": self._your_remediation_action,  # ADD
}.get(action)
```

3. Add max rounds in `_max_rounds()`:
```python
def _max_rounds(self, action: str) -> int:
    if action == "expand_queries":
        return self.config.search.max_query_expansion_rounds
    elif action == "retry_gap_search":
        return self.config.search.max_gap_search_rounds
    elif action == "lower_screening_threshold":
        return 1
    elif action == "your_remediation_action":
        return 2  # or read from config
    return 0
```

---

### Step 8: Write tests

**File:** `tests/test_pipeline/test_<your_node_name>.py`

Tests use a mock LLM that returns `LLMStructuredResponse` objects. The pattern:

```python
"""Tests for the <your_node_name> pipeline node."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from autoreview.config import load_config
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.pipeline.nodes import PipelineNodes


class MockLLM:
    """Mock LLM that returns predetermined structured responses."""

    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0,
    ):
        # Return appropriate mock data based on response_model
        if response_model == YourResultModel:
            return LLMStructuredResponse(
                parsed=YourResultModel(field_one="test", field_two=["a", "b"]),
                input_tokens=100,
                output_tokens=50,
            )
        raise ValueError(f"Unexpected model: {response_model}")

    async def generate(self, prompt, system="", max_tokens=None, temperature=0.3):
        from autoreview.llm.provider import LLMResponse
        return LLMResponse(content="mock response", input_tokens=50, output_tokens=25)


def _make_kb(**overrides) -> KnowledgeBase:
    """Create a KnowledgeBase with test defaults."""
    defaults = {
        "topic": "test topic",
        "domain": "biomedical",
        # Add whatever upstream data your node needs:
        # "evidence_map": ...,
        # "outline": ...,
    }
    defaults.update(overrides)
    return KnowledgeBase(**defaults)


@pytest.fixture
def config():
    return load_config("biomedical")


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.mark.asyncio
async def test_your_node_name_basic(mock_llm, config):
    """Test that the node runs and updates KnowledgeBase correctly."""
    kb = _make_kb()
    # Set up any required upstream state on kb
    # kb.evidence_map = ...

    nodes = PipelineNodes(mock_llm, config)
    await nodes.your_node_name(kb)

    # Assert phase was updated
    assert kb.current_phase == PipelinePhase.YOUR_NODE_NAME

    # Assert results were stored
    assert kb.your_node_results is not None  # or more specific checks

    # Assert audit entry was added
    assert any(e.node_name == "your_node_name" for e in kb.audit_log)


@pytest.mark.asyncio
async def test_your_node_name_empty_input(mock_llm, config):
    """Test graceful handling of empty/minimal input."""
    kb = _make_kb()
    nodes = PipelineNodes(mock_llm, config)
    await nodes.your_node_name(kb)
    # Assert it doesn't crash and produces sensible defaults
```

**Existing test files for reference:**
- `tests/test_pipeline/test_corpus_expansion.py` -- full mock LLM with multiple response types
- `tests/test_pipeline/test_passage_search_node.py` -- simpler single-response mock
- `tests/test_pipeline/test_remediation.py` -- testing comprehensiveness + remediation flow
- `tests/test_pipeline/test_remediation_integration.py` -- integration test for remediation

**Run tests:** `pytest tests/test_pipeline/test_<your_node_name>.py --tb=line -q`

---

### Step 9: Update config (if needed)

If your node needs configurable parameters:

**File:** `autoreview/config/models.py`

Add a new config section or extend an existing one:

```python
class YourNodeConfig(BaseModel):
    """Configuration for your node."""
    model_config = ConfigDict(extra="forbid")

    max_items: int = 100
    quality_threshold: float = 0.75
    # ... other parameters
```

Then add it to `DomainConfig`:
```python
class DomainConfig(BaseModel):
    # ... existing fields ...
    your_node: YourNodeConfig = Field(default_factory=YourNodeConfig)
```

**File:** `autoreview/config/defaults/*.yaml`

Update each domain YAML to include defaults for your config section:
```yaml
your_node:
  max_items: 100
  quality_threshold: 0.75
```

**Existing config sections:** `SearchConfig`, `ExtractionConfig`, `CritiqueConfig`, `WritingConfig`, `OutlineConfig`, `LLMConfig`

---

## Files Modified Summary

For a complete new node, you will typically touch:

| File | Change |
|------|--------|
| `autoreview/models/knowledge_base.py` | Add `PipelinePhase` enum value + optional KB field |
| `autoreview/llm/prompts/<node>.py` | **New file**: prompt module |
| `autoreview/pipeline/nodes.py` | Add `async def <node>(self, kb)` method |
| `autoreview/pipeline/runner.py` | Add `dag.add_node()` call in `build_pipeline()` |
| `tests/test_pipeline/test_<node>.py` | **New file**: tests with mock LLM |
| `autoreview/analysis/comprehensiveness.py` | *(optional)* Add checker class |
| `autoreview/pipeline/remediation.py` | *(optional)* Add remediation handler |
| `autoreview/config/models.py` | *(optional)* Add config section |
| `autoreview/config/defaults/*.yaml` | *(optional)* Add YAML defaults |

---

## Quick Reference: Key Imports

```python
# KnowledgeBase and phase
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase

# Token tracking (inside nodes.py only -- these are module-private)
# _TokenAccumulator, _GlobalTokenAccumulator

# LLM provider types
from autoreview.llm.provider import LLMProvider, LLMResponse, LLMStructuredResponse

# Base model for Pydantic schemas
from autoreview.models.base import AutoReviewModel, TimestampedModel

# Comprehensiveness
from autoreview.analysis.comprehensiveness import (
    CheckStatus, ComprehensiveCheckResult, RemediationAction,
)

# Config
from autoreview.config.models import DomainConfig

# Pipeline
from autoreview.pipeline.nodes import PipelineNodes
from autoreview.pipeline.dag import DAGRunner
from autoreview.pipeline.runner import build_pipeline, run_pipeline
```
