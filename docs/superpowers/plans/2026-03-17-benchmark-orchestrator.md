# Benchmark Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single orchestrator (`paper/run_benchmark.py`) to run 120-140 deduplicated pipeline runs, evaluate outputs, and execute all 10 analysis scripts for the benchmark paper.

**Architecture:** Typer CLI wrapping existing `run_pipeline()`, `run_evaluation()`, and `ARISERubricScorer`. Topics defined in YAML with Pydantic validation. Run registry (JSON) tracks completion for dedup/resume. Analysis scripts share common utilities extracted from Analysis 10. Each analysis loads a unified DataFrame via `load_all_evaluations()`.

**Tech Stack:** Python 3.11+, typer, pydantic, pandas, numpy, scipy, matplotlib, structlog, PyYAML

**Spec:** `docs/superpowers/specs/2026-03-17-benchmark-orchestrator-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `paper/models.py` | Pydantic models: `ReferenceInfo`, `TopicEntry`, `TopicsConfig`, `RunRegistryEntry`, `RunRegistry` |
| `paper/topics.yaml` | Topic definitions (placeholder with 2 examples for testing) |
| `paper/analysis/common.py` | Shared utilities: palette, styling, `load_all_evaluations()`, `fdr_correct()`, report/JSON helpers |
| `paper/run_benchmark.py` | Typer CLI orchestrator with 5 subcommands |
| `paper/analysis/main_comparison.py` | Analysis 1 |
| `paper/analysis/domain_analysis.py` | Analysis 2 |
| `paper/analysis/rubric_agreement.py` | Analysis 3 |
| `paper/analysis/ablation_analysis.py` | Analysis 4 |
| `paper/analysis/retrieval_decomposition.py` | Analysis 5 |
| `paper/analysis/citation_analysis.py` | Analysis 6 |
| `paper/analysis/model_comparison.py` | Analysis 7 |
| `paper/analysis/cost_analysis.py` | Analysis 8 |
| `paper/analysis/contamination_analysis.py` | Analysis 9 |
| `tests/test_paper/test_models.py` | Tests for topic/registry models |
| `tests/test_paper/test_common.py` | Tests for shared analysis utilities |
| `tests/test_paper/test_run_benchmark.py` | Tests for orchestrator matrix/registry logic |
| `tests/test_paper/test_analysis_*.py` | Tests for each analysis script |

### Modified Files

| File | Change |
|------|--------|
| `paper/__init__.py` | Already exists, no change needed |
| `paper/analysis/__init__.py` | Already exists, no change needed |

### Key Reference Files (read before implementing)

| File | Why |
|------|-----|
| `autoreview/models/base.py` | `AutoReviewModel` base class |
| `autoreview/evaluation/models.py` | `EvaluationResult`, `CitationScore`, `ARISERubricResult`, etc. |
| `autoreview/evaluation/cost_analyzer.py` | `PRICING` dict, `CostSummary` model |
| `autoreview/evaluation/evaluator.py` | `run_evaluation()` signature |
| `autoreview/evaluation/batch_runner.py` | Semaphore concurrency pattern |
| `autoreview/pipeline/runner.py` | `run_pipeline()` signature, `build_pipeline()` |
| `autoreview/pipeline/dag.py` | `DAGRunner.execute()` for skip_nodes |
| `autoreview/cli.py` | Existing CLI patterns (typer, config setup) |
| `autoreview/config/models.py` | `DomainConfig`, `WritingConfig`, `DepthLevel` |
| `paper/analysis/depth_comparison.py` | Analysis 10 template — figure styling, stats, reporting |

---

## Chunk 1: Foundation — Models, Topics, Common Utilities

### Task 1: Topic and Registry Pydantic Models

**Files:**
- Create: `paper/models.py`
- Create: `tests/test_paper/__init__.py`
- Create: `tests/test_paper/test_models.py`

- [ ] **Step 1: Write tests for topic models**

```python
# tests/test_paper/test_models.py
"""Tests for paper benchmark models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper.models import (
    ReferenceInfo,
    RunRegistry,
    RunRegistryEntry,
    TopicEntry,
    TopicsConfig,
    expand_run_matrix,
    load_topics,
)


def _make_topic(
    id: str = "test_topic",
    tier: str = "B",
    domain: str = "cs_ai",
    conditions: list[str] | None = None,
    date_range: str | None = None,
    ablation: bool = False,
) -> dict[str, Any]:
    return {
        "id": id,
        "title": f"Test topic {id}",
        "domain": domain,
        "tier": tier,
        "reference": {
            "doi": "10.1234/test",
            "title": "Test Reference",
            "year": 2023,
            "citation_count": 100,
            "pdf_path": "paper/references/test.pdf",
        },
        "conditions": conditions or ["end_to_end"],
        "date_range": date_range,
        "ablation": ablation,
    }


class TestTopicEntry:
    def test_basic_creation(self) -> None:
        topic = TopicEntry(**_make_topic())
        assert topic.id == "test_topic"
        assert topic.tier == "B"
        assert topic.conditions == ["end_to_end"]

    def test_tier_a_with_date_range(self) -> None:
        topic = TopicEntry(
            **_make_topic(tier="A", conditions=["end_to_end", "retrieval_controlled"], date_range="-2017")
        )
        assert topic.tier == "A"
        assert topic.date_range == "-2017"
        assert "retrieval_controlled" in topic.conditions

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(Exception):
            TopicEntry(**_make_topic(tier="C"))


class TestTopicsConfig:
    def test_load_from_dict(self) -> None:
        config = TopicsConfig(
            metadata={"created": "2026-03-17"},
            topics=[TopicEntry(**_make_topic())],
        )
        assert len(config.topics) == 1

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
metadata:
  created: "2026-03-17"
topics:
  - id: "test_topic"
    title: "Test topic"
    domain: cs_ai
    tier: B
    reference:
      doi: "10.1234/test"
      title: "Test Ref"
      year: 2023
      citation_count: 100
      pdf_path: "paper/references/test.pdf"
    conditions: [end_to_end]
"""
        yaml_path = tmp_path / "topics.yaml"
        yaml_path.write_text(yaml_content)
        config = load_topics(yaml_path)
        assert len(config.topics) == 1
        assert config.topics[0].id == "test_topic"


class TestRunMatrix:
    def test_expand_basic(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 topic x 1 model x medium x end_to_end = 1 run
        assert len(matrix) == 1
        key = matrix[0]
        assert key == ("test_topic", "claude-sonnet-4-6", "medium", "end_to_end")

    def test_tier_a_expands_conditions(self) -> None:
        topics = [TopicEntry(**_make_topic(tier="A", conditions=["end_to_end", "retrieval_controlled"]))]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 topic x 1 model x medium x 2 conditions = 2
        assert len(matrix) == 2

    def test_ablation_topics_expand(self) -> None:
        topics = [TopicEntry(**_make_topic(ablation=True))]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 end_to_end + 4 ablation conditions (sonnet only) = 5
        conditions = [key[3] for key in matrix]
        assert "end_to_end" in conditions
        assert "no_critique_loops" in conditions

    def test_depth_runs_added_for_sonnet(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6", "claude-opus-4-6"]
        matrix = expand_run_matrix(topics, models, include_depth=True)
        depths = [(k[1], k[2]) for k in matrix]
        # Sonnet gets low + deep in addition to medium; Opus gets medium only
        assert ("claude-sonnet-4-6", "low") in depths
        assert ("claude-sonnet-4-6", "deep") in depths
        assert ("claude-opus-4-6", "low") not in depths

    def test_dedup_removes_duplicates(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models, include_depth=True)
        # Should not have duplicate (topic, sonnet, medium, end_to_end)
        assert len(matrix) == len(set(matrix))


class TestRunRegistry:
    def test_empty_registry(self) -> None:
        reg = RunRegistry()
        assert len(reg.runs) == 0
        assert not reg.is_completed("topic|model|medium|end_to_end")

    def test_register_and_check(self) -> None:
        reg = RunRegistry()
        reg.register_start("topic|model|medium|end_to_end")
        assert not reg.is_completed("topic|model|medium|end_to_end")

        reg.register_complete(
            "topic|model|medium|end_to_end",
            output_dir="out/dir",
            review_path="out/dir/review.md",
        )
        assert reg.is_completed("topic|model|medium|end_to_end")

    def test_save_and_load(self, tmp_path: Path) -> None:
        reg = RunRegistry()
        reg.register_complete("k", output_dir="d", review_path="r")
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = RunRegistry.load(path)
        assert loaded.is_completed("k")

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        reg = RunRegistry.load(tmp_path / "nonexistent.json")
        assert len(reg.runs) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper.models'`

- [ ] **Step 3: Implement models**

```python
# paper/models.py
"""Pydantic models for benchmark orchestration."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml  # type: ignore[import-untyped]

from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()

ABLATION_CONDITIONS = [
    "no_evidence_chains",
    "no_critique_loops",
    "no_passage_mining",
    "no_comprehensiveness",
]

MODELS_DEFAULT = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

# Only Sonnet gets depth and ablation runs
SONNET_MODEL = "claude-sonnet-4-6"

RunKey = tuple[str, str, str, str]  # (topic_id, model, depth, condition)


class ReferenceInfo(AutoReviewModel):
    doi: str
    title: str
    year: int
    citation_count: int
    pdf_path: str


class TopicEntry(AutoReviewModel):
    id: str
    title: str
    domain: str
    tier: Literal["A", "B"]
    reference: ReferenceInfo
    conditions: list[str]
    date_range: str | None = None
    ablation: bool = False


class TopicsConfig(AutoReviewModel):
    metadata: dict[str, Any] = {}
    topics: list[TopicEntry] = []


class RunRegistryEntry(AutoReviewModel):
    status: Literal["running", "completed", "failed", "permanently_failed"] = "running"
    output_dir: str = ""
    review_path: str = ""
    snapshot_path: str = ""
    evaluation_path: str = ""
    started_at: str = ""
    completed_at: str = ""
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    error: str = ""
    failure_count: int = 0


class RunRegistry(AutoReviewModel):
    runs: dict[str, RunRegistryEntry] = {}
    last_updated: str = ""

    def is_completed(self, key: str) -> bool:
        entry = self.runs.get(key)
        return entry is not None and entry.status == "completed"

    def is_evaluated(self, key: str) -> bool:
        entry = self.runs.get(key)
        return entry is not None and bool(entry.evaluation_path)

    def register_start(self, key: str) -> None:
        self.runs[key] = RunRegistryEntry(
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        self.last_updated = datetime.now(UTC).isoformat()

    def register_complete(
        self,
        key: str,
        output_dir: str,
        review_path: str,
        snapshot_path: str = "",
        cost_usd: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        entry = self.runs.get(key, RunRegistryEntry())
        entry.status = "completed"
        entry.output_dir = output_dir
        entry.review_path = review_path
        entry.snapshot_path = snapshot_path
        entry.completed_at = datetime.now(UTC).isoformat()
        entry.cost_usd = cost_usd
        entry.tokens_input = tokens_input
        entry.tokens_output = tokens_output
        self.runs[key] = entry
        self.last_updated = datetime.now(UTC).isoformat()

    def register_failure(self, key: str, error: str) -> None:
        entry = self.runs.get(key, RunRegistryEntry())
        entry.status = "failed"
        entry.error = error
        entry.failure_count += 1
        if entry.failure_count >= 3:
            entry.status = "permanently_failed"
        self.runs[key] = entry
        self.last_updated = datetime.now(UTC).isoformat()

    def register_evaluation(self, key: str, evaluation_path: str) -> None:
        entry = self.runs.get(key)
        if entry:
            entry.evaluation_path = evaluation_path
            self.last_updated = datetime.now(UTC).isoformat()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> RunRegistry:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())

    @property
    def completed_count(self) -> int:
        return sum(1 for e in self.runs.values() if e.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self.runs.values() if e.status in ("failed", "permanently_failed"))


def load_topics(path: Path) -> TopicsConfig:
    """Load topics from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return TopicsConfig.model_validate(data)


def make_run_key(topic_id: str, model: str, depth: str, condition: str) -> str:
    """Create a registry key string from components."""
    return f"{topic_id}|{model}|{depth}|{condition}"


def parse_run_key(key: str) -> tuple[str, str, str, str]:
    """Parse a registry key back into (topic_id, model, depth, condition)."""
    parts = key.split("|")
    return (parts[0], parts[1], parts[2], parts[3])


def expand_run_matrix(
    topics: list[TopicEntry],
    models: list[str] | None = None,
    include_depth: bool = True,
    include_ablation: bool = True,
) -> list[RunKey]:
    """Expand all unique (topic_id, model, depth, condition) tuples with dedup.

    Rules:
    - All topics x all models x medium x each condition
    - Depth runs (low, deep): Sonnet only, all topics, end_to_end only
    - Ablation runs: Sonnet only, topics with ablation=True, medium only
    """
    if models is None:
        models = MODELS_DEFAULT

    seen: set[RunKey] = set()
    result: list[RunKey] = []

    def _add(key: RunKey) -> None:
        if key not in seen:
            seen.add(key)
            result.append(key)

    for topic in topics:
        for model in models:
            for condition in topic.conditions:
                _add((topic.id, model, "medium", condition))

        # Depth runs: Sonnet only, end_to_end only
        if include_depth and SONNET_MODEL in models:
            for depth in ("low", "deep"):
                _add((topic.id, SONNET_MODEL, depth, "end_to_end"))

        # Ablation runs: Sonnet only, ablation topics, medium
        if include_ablation and topic.ablation and SONNET_MODEL in models:
            for ablation_cond in ABLATION_CONDITIONS:
                _add((topic.id, SONNET_MODEL, "medium", ablation_cond))

    return result
```

- [ ] **Step 4: Create `tests/test_paper/__init__.py`**

```python
# empty
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_paper/test_models.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add paper/models.py tests/test_paper/__init__.py tests/test_paper/test_models.py
git commit -m "feat(paper): add topic and run registry models with matrix expansion"
```

---

### Task 2: Topics YAML Placeholder

**Files:**
- Create: `paper/topics.yaml`

- [ ] **Step 1: Create placeholder topics YAML with 2 example topics**

```yaml
# paper/topics.yaml
# Benchmark evaluation topics for the AutoReview paper.
# Populate with full set from benchmark_candidates.md before running.

metadata:
  created: "2026-03-17"
  description: "Benchmark evaluation topics for AutoReview paper"

topics:
  # --- Tier A: Landmark Reviews (500+ citations, 2015-2019) ---
  - id: "microglia_homeostasis"
    title: "Microglia homeostasis, neuroinflammation, and neurodegeneration"
    domain: biomedical
    tier: A
    reference:
      doi: "10.1016/j.immuni.2017.08.008"
      title: "Microglia in Neurodegeneration"
      year: 2017
      citation_count: 2800
      pdf_path: "paper/references/colonna_2017.pdf"
    conditions: [end_to_end, retrieval_controlled]
    date_range: "-2017"
    ablation: true

  # --- Tier B: Contemporary Reviews (50-200 citations, 2023-2024) ---
  - id: "rag_architectures"
    title: "Retrieval-augmented generation for large language models"
    domain: cs_ai
    tier: B
    reference:
      doi: "10.48550/arXiv.2312.10997"
      title: "Retrieval-Augmented Generation for Large Language Models: A Survey"
      year: 2023
      citation_count: 180
      pdf_path: "paper/references/gao_2023.pdf"
    conditions: [end_to_end]
    ablation: false
```

- [ ] **Step 2: Verify it loads**

Run: `python -c "from paper.models import load_topics; c = load_topics(__import__('pathlib').Path('paper/topics.yaml')); print(f'{len(c.topics)} topics loaded')"`
Expected: `2 topics loaded`

- [ ] **Step 3: Commit**

```bash
git add paper/topics.yaml
git commit -m "feat(paper): add placeholder topics YAML with 2 example topics"
```

---

### Task 3: Shared Analysis Utilities (`common.py`)

**Files:**
- Create: `paper/analysis/common.py`
- Create: `tests/test_paper/test_common.py`

- [ ] **Step 1: Write tests for common utilities**

```python
# tests/test_paper/test_common.py
"""Tests for shared analysis utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.analysis.common import (
    CB_PALETTE,
    apply_style,
    fdr_correct,
    load_all_evaluations,
    save_analysis_json,
)


class TestFDRCorrect:
    def test_single_pvalue(self) -> None:
        result = fdr_correct([0.05])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.05)

    def test_monotonicity(self) -> None:
        """Adjusted p-values must be monotonically increasing when sorted by raw p."""
        raw = [0.001, 0.01, 0.03, 0.05, 0.5]
        adj = fdr_correct(raw)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-10

    def test_capped_at_one(self) -> None:
        adj = fdr_correct([0.9, 0.95, 0.99])
        assert all(p <= 1.0 for p in adj)

    def test_empty(self) -> None:
        assert fdr_correct([]) == []


class TestSaveAnalysisJSON:
    def test_saves_and_loads(self, tmp_path: Path) -> None:
        data = {"metric": 0.95, "nested": {"a": [1, 2, 3]}}
        out = tmp_path / "sub" / "analysis.json"
        save_analysis_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["metric"] == 0.95
        assert loaded["nested"]["a"] == [1, 2, 3]


class TestApplyStyle:
    def test_sets_rcparams(self) -> None:
        import matplotlib.pyplot as plt
        apply_style()
        assert plt.rcParams["axes.labelsize"] == 12


class TestLoadAllEvaluations:
    def _make_eval_json(self, path: Path) -> None:
        """Write a minimal evaluation.json file."""
        data = {
            "timestamp": "2026-03-17T00:00:00",
            "generated_path": "gen.md",
            "reference_path": "ref.pdf",
            "overall_score": 0.75,
            "citation_score": {
                "recall": 0.6, "precision": 0.8, "f1": 0.69,
                "matched_count": 10, "reference_count": 15, "generated_count": 12,
                "matched_titles": [], "missed_titles": [], "hallucinated_titles": [],
            },
            "synthesis_score": {
                "generated_score": 3.5, "reference_score": 4.0, "delta": -0.5,
                "dimension_scores": {}, "generated_observations": "", "reference_observations": "",
            },
            "topic_coverage": {
                "generated_coverage": 0.8, "reference_coverage": 1.0,
                "topics_in_both": [], "topics_only_in_reference": [], "topics_only_in_generated": [],
            },
            "writing_quality": {
                "generated_score": 3.0, "reference_score": 4.0, "delta": -1.0,
                "dimension_scores": {},
            },
            "structural_metrics": {
                "word_count": 5000, "section_count": 8, "citation_count": 45,
                "citations_per_1000_words": 9.0, "avg_section_length_words": 625,
                "section_balance": 0.3, "flesch_kincaid_grade": 14.5,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def _make_registry(self, results_dir: Path) -> None:
        """Write a minimal run_registry.json."""
        from paper.models import RunRegistry
        reg = RunRegistry()
        key = "test_topic|claude-sonnet-4-6|medium|end_to_end"
        run_dir = results_dir / "test_topic" / "claude-sonnet-4-6_medium_end_to_end"
        eval_path = run_dir / "evaluation.json"
        self._make_eval_json(eval_path)
        reg.register_complete(
            key,
            output_dir=str(run_dir),
            review_path=str(run_dir / "review.md"),
            cost_usd=2.50,
            tokens_input=400000,
            tokens_output=80000,
        )
        reg.register_evaluation(key, str(eval_path))
        reg.save(results_dir / "run_registry.json")

    def test_loads_from_registry(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        self._make_registry(results_dir)

        # Also need a topics.yaml for domain/tier lookup
        from paper.models import TopicsConfig, TopicEntry, ReferenceInfo
        topics = TopicsConfig(topics=[TopicEntry(
            id="test_topic", title="Test", domain="cs_ai", tier="B",
            reference=ReferenceInfo(doi="x", title="y", year=2023, citation_count=100, pdf_path="z"),
            conditions=["end_to_end"],
        )])

        df = load_all_evaluations(results_dir, topics)
        assert len(df) == 1
        assert df.iloc[0]["topic_id"] == "test_topic"
        assert df.iloc[0]["system"] == "autoreview"
        assert df.iloc[0]["overall_score"] == 0.75
        assert df.iloc[0]["cost_usd"] == 2.50

    def test_arise_outputs_loaded(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)

        # Empty registry
        from paper.models import RunRegistry, TopicsConfig, TopicEntry, ReferenceInfo
        RunRegistry().save(results_dir / "run_registry.json")

        # ARISE output
        arise_dir = results_dir / "arise" / "test_topic"
        self._make_eval_json(arise_dir / "evaluation.json")

        topics = TopicsConfig(topics=[TopicEntry(
            id="test_topic", title="Test", domain="cs_ai", tier="B",
            reference=ReferenceInfo(doi="x", title="y", year=2023, citation_count=100, pdf_path="z"),
            conditions=["end_to_end"],
        )])

        df = load_all_evaluations(results_dir, topics)
        assert len(df) == 1
        assert df.iloc[0]["system"] == "arise"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_common.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement common utilities**

```python
# paper/analysis/common.py
"""Shared utilities for benchmark analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from paper.models import TopicsConfig

logger = structlog.get_logger()

# Colorblind-safe palette (Wong 2011, matches depth_comparison.py)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

FONT_CONFIG = {
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}


def apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(FONT_CONFIG)


def fdr_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values in the same order as input.
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute raw adjusted values
    raw_adj = [0.0] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        raw_adj[rank] = min(p * m / (rank + 1), 1.0)

    # Cumulative minimum from right (enforce monotonicity)
    for i in range(len(raw_adj) - 2, -1, -1):
        raw_adj[i] = min(raw_adj[i], raw_adj[i + 1])

    # Map back to original order
    result = [0.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = raw_adj[rank]

    return result


def save_analysis_json(data: dict[str, Any], path: Path) -> None:
    """Save analysis results as JSON with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("analysis.json_saved", path=str(path))


def generate_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Generate a markdown table string."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_all_evaluations(results_dir: Path, topics: TopicsConfig) -> pd.DataFrame:
    """Load all evaluation results into a unified DataFrame.

    Reads run_registry.json for AutoReview runs, scans arise/ for ARISE outputs.
    """
    from paper.models import RunRegistry, parse_run_key

    topic_lookup = {t.id: t for t in topics.topics}
    registry = RunRegistry.load(results_dir / "run_registry.json")
    rows: list[dict[str, Any]] = []

    # AutoReview runs from registry
    for key, entry in registry.runs.items():
        if entry.status != "completed" or not entry.evaluation_path:
            continue

        eval_path = Path(entry.evaluation_path)
        if not eval_path.exists():
            logger.warning("common.missing_eval", key=key, path=str(eval_path))
            continue

        eval_data = json.loads(eval_path.read_text())
        topic_id, model, depth, condition = parse_run_key(key)
        topic_info = topic_lookup.get(topic_id)

        row = _eval_to_row(eval_data)
        row.update({
            "topic_id": topic_id,
            "domain": topic_info.domain if topic_info else "unknown",
            "tier": topic_info.tier if topic_info else "unknown",
            "system": "autoreview",
            "model": model,
            "depth": depth,
            "condition": condition,
            "cost_usd": entry.cost_usd,
            "tokens_input": entry.tokens_input,
            "tokens_output": entry.tokens_output,
        })
        rows.append(row)

    # ARISE outputs
    arise_dir = results_dir / "arise"
    if arise_dir.exists():
        for topic_dir in arise_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            eval_path = topic_dir / "evaluation.json"
            if not eval_path.exists():
                continue

            eval_data = json.loads(eval_path.read_text())
            topic_id = topic_dir.name
            topic_info = topic_lookup.get(topic_id)

            row = _eval_to_row(eval_data)
            row.update({
                "topic_id": topic_id,
                "domain": topic_info.domain if topic_info else "unknown",
                "tier": topic_info.tier if topic_info else "unknown",
                "system": "arise",
                "model": "arise",
                "depth": "medium",
                "condition": "end_to_end",
                "cost_usd": None,
                "tokens_input": None,
                "tokens_output": None,
            })
            rows.append(row)

    return pd.DataFrame(rows)


def _eval_to_row(eval_data: dict[str, Any]) -> dict[str, Any]:
    """Extract flat row from an evaluation JSON dict."""
    row: dict[str, Any] = {
        "overall_score": eval_data.get("overall_score", 0.0),
        "citation_recall": eval_data.get("citation_score", {}).get("recall", 0.0),
        "citation_precision": eval_data.get("citation_score", {}).get("precision", 0.0),
        "citation_f1": eval_data.get("citation_score", {}).get("f1", 0.0),
        "synthesis_score": eval_data.get("synthesis_score", {}).get("generated_score", 0.0),
        "topic_coverage": eval_data.get("topic_coverage", {}).get("generated_coverage", 0.0),
        "writing_quality": eval_data.get("writing_quality", {}).get("generated_score", 0.0),
    }

    # ARISE rubric
    arise = eval_data.get("arise_result")
    row["arise_total"] = arise.get("total_score") if arise else None

    # Hallucination rate (for Analysis 6)
    cs = eval_data.get("citation_score", {})
    hallucinated = len(cs.get("hallucinated_titles", []))
    generated = cs.get("generated_count", 0)
    row["hallucination_rate"] = hallucinated / generated if generated > 0 else 0.0

    # Structural metrics
    sm = eval_data.get("structural_metrics")
    if sm:
        row["word_count"] = sm.get("word_count", 0)
        row["section_count"] = sm.get("section_count", 0)
        row["citation_count"] = sm.get("citation_count", 0)
        row["citations_per_1000_words"] = sm.get("citations_per_1000_words", 0.0)
        row["flesch_kincaid_grade"] = sm.get("flesch_kincaid_grade", 0.0)
    else:
        row.update({"word_count": 0, "section_count": 0, "citation_count": 0,
                     "citations_per_1000_words": 0.0, "flesch_kincaid_grade": 0.0})

    return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paper/test_common.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/common.py tests/test_paper/test_common.py
git commit -m "feat(paper): add shared analysis utilities with load_all_evaluations"
```

---

## Chunk 2: Orchestrator CLI

### Task 4: Orchestrator — `generate-matrix` and `run` Subcommands

**Files:**
- Create: `paper/run_benchmark.py`
- Create: `tests/test_paper/test_run_benchmark.py`

- [ ] **Step 1: Write tests for matrix generation and registry integration**

```python
# tests/test_paper/test_run_benchmark.py
"""Tests for benchmark orchestrator."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from paper.models import (
    ReferenceInfo,
    RunRegistry,
    TopicEntry,
    TopicsConfig,
    expand_run_matrix,
    load_topics,
    make_run_key,
)


@pytest.fixture()
def sample_topics() -> TopicsConfig:
    return TopicsConfig(
        metadata={"created": "2026-03-17"},
        topics=[
            TopicEntry(
                id="topic_a", title="Topic A", domain="biomedical", tier="A",
                reference=ReferenceInfo(doi="x", title="y", year=2017, citation_count=500, pdf_path="z"),
                conditions=["end_to_end", "retrieval_controlled"],
                date_range="-2017", ablation=True,
            ),
            TopicEntry(
                id="topic_b", title="Topic B", domain="cs_ai", tier="B",
                reference=ReferenceInfo(doi="x", title="y", year=2023, citation_count=100, pdf_path="z"),
                conditions=["end_to_end"],
            ),
        ],
    )


class TestMatrixGeneration:
    def test_full_matrix_count(self, sample_topics: TopicsConfig) -> None:
        models = ["claude-sonnet-4-6", "claude-opus-4-6"]
        matrix = expand_run_matrix(sample_topics.topics, models, include_depth=True, include_ablation=True)
        # topic_a: 2 models x 2 conditions x medium = 4
        #        + sonnet x {low,deep} x end_to_end = 2 (depth)
        #        + sonnet x 4 ablation x medium = 4
        # topic_b: 2 models x 1 condition x medium = 2
        #        + sonnet x {low,deep} x end_to_end = 2 (depth)
        # Total = 4 + 2 + 4 + 2 + 2 = 14
        assert len(matrix) == 14

    def test_completed_runs_skipped(self, sample_topics: TopicsConfig) -> None:
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(sample_topics.topics, models)
        registry = RunRegistry()
        key = make_run_key("topic_b", "claude-sonnet-4-6", "medium", "end_to_end")
        registry.register_complete(key, output_dir="d", review_path="r")
        remaining = [k for k in matrix if not registry.is_completed(make_run_key(*k))]
        assert len(remaining) == len(matrix) - 1


class TestCostEstimate:
    def test_estimate_produces_output(self, sample_topics: TopicsConfig) -> None:
        from paper.run_benchmark import estimate_cost
        matrix = expand_run_matrix(sample_topics.topics, ["claude-sonnet-4-6"])
        cost = estimate_cost(matrix)
        assert cost > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the orchestrator**

```python
# paper/run_benchmark.py
"""Benchmark orchestrator for the AutoReview paper.

Subcommands:
    generate-matrix  Show all runs with dedup and cost estimate
    run              Execute pipeline runs
    evaluate         Evaluate completed runs
    analyze          Run analysis scripts
    full             run -> evaluate -> analyze
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
import typer

from paper.models import (
    SONNET_MODEL,
    RunKey,
    RunRegistry,
    TopicsConfig,
    expand_run_matrix,
    load_topics,
    make_run_key,
    parse_run_key,
)

app = typer.Typer(name="run-benchmark", help="Benchmark orchestrator for AutoReview paper.", no_args_is_help=True)
logger = structlog.get_logger()

# Default model list
DEFAULT_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]

# Rough per-run cost estimates by (model_prefix, depth) in USD
_COST_ESTIMATES: dict[tuple[str, str], float] = {
    ("opus", "medium"): 15.0,
    ("opus", "low"): 6.0,
    ("opus", "deep"): 35.0,
    ("sonnet", "medium"): 2.0,
    ("sonnet", "low"): 0.80,
    ("sonnet", "deep"): 5.0,
    ("haiku", "medium"): 0.20,
    ("haiku", "low"): 0.08,
    ("haiku", "deep"): 0.50,
}


def _model_tier(model: str) -> str:
    if "opus" in model:
        return "opus"
    if "haiku" in model:
        return "haiku"
    return "sonnet"


def estimate_cost(matrix: list[RunKey]) -> float:
    """Estimate total API cost for the run matrix."""
    total = 0.0
    for _, model, depth, _ in matrix:
        tier = _model_tier(model)
        total += _COST_ESTIMATES.get((tier, depth), 2.0)
    return total


def _format_matrix_summary(
    matrix: list[RunKey], registry: RunRegistry
) -> str:
    """Format a human-readable summary of the run matrix."""
    lines: list[str] = []
    completed = sum(1 for k in matrix if registry.is_completed(make_run_key(*k)))
    remaining = len(matrix) - completed

    lines.append(f"Total unique runs: {len(matrix)}")
    lines.append(f"  Completed: {completed}")
    lines.append(f"  Remaining: {remaining}")
    lines.append(f"  Estimated cost (remaining): ${estimate_cost([k for k in matrix if not registry.is_completed(make_run_key(*k))]):.0f}")

    # Breakdown by batch
    batch_counts: dict[str, int] = {}
    for topic_id, model, depth, condition in matrix:
        if condition in ("no_evidence_chains", "no_critique_loops", "no_passage_mining", "no_comprehensiveness"):
            batch = "3e: Ablation"
        elif condition == "retrieval_controlled":
            batch = "3c: Retrieval-controlled"
        elif depth != "medium":
            batch = "3f: Depth"
        else:
            batch = "3a/3b: End-to-end"
        batch_counts[batch] = batch_counts.get(batch, 0) + 1

    lines.append("\nBy batch:")
    for batch, count in sorted(batch_counts.items()):
        lines.append(f"  {batch}: {count} runs")

    return "\n".join(lines)


@app.command(name="generate-matrix")
def generate_matrix(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics", help="Path to topics YAML"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir", help="Results directory"),
    models: str = typer.Option(",".join(DEFAULT_MODELS), "--models", help="Comma-separated model list"),
) -> None:
    """Show the full run matrix with dedup and cost estimate."""
    config = load_topics(topics_path)
    model_list = [m.strip() for m in models.split(",")]
    matrix = expand_run_matrix(config.topics, model_list)
    registry = RunRegistry.load(results_dir / "run_registry.json")

    typer.echo(_format_matrix_summary(matrix, registry))


@app.command()
def run(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    models: str = typer.Option(",".join(DEFAULT_MODELS), "--models"),
    max_concurrent: int = typer.Option(2, "--max-concurrent"),
    batch_filter: str | None = typer.Option(None, "--batch", help="Filter to batches: 3a,3b,3c,3e,3f"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Execute pipeline runs."""
    _setup_logging(verbose)
    config = load_topics(topics_path)
    model_list = [m.strip() for m in models.split(",")]
    matrix = expand_run_matrix(config.topics, model_list)
    registry = RunRegistry.load(results_dir / "run_registry.json")

    topic_lookup = {t.id: t for t in config.topics}

    # Filter to remaining runs
    remaining = [
        k for k in matrix
        if not registry.is_completed(make_run_key(*k))
        and registry.runs.get(make_run_key(*k), type("E", (), {"status": "pending"})()).status != "permanently_failed"
    ]

    # Apply batch filter if specified
    if batch_filter:
        allowed = {b.strip() for b in batch_filter.split(",")}
        remaining = [k for k in remaining if _classify_batch(k) in allowed]


def _classify_batch(key: RunKey) -> str:
    """Classify a run key into its batch label."""
    _, model, depth, condition = key
    if condition in ("no_evidence_chains", "no_critique_loops", "no_passage_mining", "no_comprehensiveness"):
        return "3e"
    if condition == "retrieval_controlled":
        return "3c"
    if depth != "medium":
        return "3f"
    # Distinguish 3a (Tier B) from 3b (Tier A) would require topic lookup;
    # for filtering purposes, both are "3ab"
    return "3ab"

    typer.echo(f"Remaining runs: {len(remaining)} of {len(matrix)} total")
    if dry_run:
        typer.echo("Dry run — no execution.")
        return

    asyncio.run(_execute_runs(remaining, topic_lookup, registry, results_dir, max_concurrent))


async def _execute_runs(
    runs: list[RunKey],
    topic_lookup: dict[str, Any],
    registry: RunRegistry,
    results_dir: Path,
    max_concurrent: int,
) -> None:
    """Execute pipeline runs with semaphore concurrency."""
    from autoreview.config import load_config
    from autoreview.config.models import DepthLevel
    from autoreview.llm.factory import create_llm_provider
    from autoreview.models.knowledge_base import KnowledgeBase
    from autoreview.pipeline.runner import run_pipeline

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_single(key: RunKey) -> None:
        topic_id, model, depth, condition = key
        run_key_str = make_run_key(*key)
        topic = topic_lookup.get(topic_id)
        if not topic:
            logger.error("benchmark.unknown_topic", topic_id=topic_id)
            return

        async with semaphore:
            logger.info("benchmark.run.start", key=run_key_str)
            registry.register_start(run_key_str)

            output_dir = str(results_dir / topic_id / f"{model}_{depth}_{condition}")

            try:
                overrides: dict[str, Any] = {"llm": {"model": model}}
                config = load_config(domain=topic.domain, overrides=overrides)
                config.writing.depth = DepthLevel(depth)

                if topic.date_range:
                    config.search.date_range = topic.date_range

                # Ablation config
                if condition == "no_critique_loops":
                    config.critique.max_revision_cycles = 0
                if condition == "no_evidence_chains":
                    config.writing.evidence_chains = False

                kb = KnowledgeBase(topic=topic.title, domain=topic.domain, output_dir=output_dir)
                llm = create_llm_provider(config.llm)

                if condition == "retrieval_controlled":
                    # Bibliography injection + resume from full_text_retrieval
                    from paper.analysis.inject_bibliography import inject_bibliography
                    kb = await inject_bibliography(
                        pdf_path=Path(topic.reference.pdf_path),
                        topic=topic.title,
                        domain=topic.domain,
                        output_dir=output_dir,
                        llm=llm,
                    )
                    kb = await run_pipeline(llm=llm, config=config, kb=kb, start_from="full_text_retrieval")
                else:
                    skip_nodes: list[str] = []
                    if condition == "no_passage_mining":
                        skip_nodes.append("passage_search")
                    elif condition == "no_comprehensiveness":
                        skip_nodes.append("gap_search")
                    # Note: no_evidence_chains handled via config flag (prerequisite task)
                    kb = await run_pipeline(
                        llm=llm, config=config, kb=kb,
                        skip_nodes=set(skip_nodes) if skip_nodes else None,
                    )

                # Find generated review
                review_path = next(Path(output_dir).glob("*.md"), Path(output_dir) / "review.md")
                tokens = kb.total_tokens()

                registry.register_complete(
                    run_key_str,
                    output_dir=output_dir,
                    review_path=str(review_path),
                    cost_usd=0.0,  # Computed from tokens post-hoc
                    tokens_input=tokens.get("input_tokens", 0),
                    tokens_output=tokens.get("output_tokens", 0),
                )
                logger.info("benchmark.run.complete", key=run_key_str)

            except Exception as e:
                logger.error("benchmark.run.failed", key=run_key_str, error=str(e))
                registry.register_failure(run_key_str, str(e))

            finally:
                registry.save(results_dir / "run_registry.json")

    await asyncio.gather(*[_run_single(k) for k in runs], return_exceptions=True)


@app.command()
def evaluate(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    max_concurrent: int = typer.Option(5, "--max-concurrent"),
    judge_model: str = typer.Option("claude-sonnet-4-6", "--judge-model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate completed runs with both rubrics."""
    _setup_logging(verbose)
    config = load_topics(topics_path)
    registry = RunRegistry.load(results_dir / "run_registry.json")
    topic_lookup = {t.id: t for t in config.topics}

    asyncio.run(_evaluate_runs(registry, topic_lookup, results_dir, max_concurrent, judge_model))


async def _evaluate_runs(
    registry: RunRegistry,
    topic_lookup: dict[str, Any],
    results_dir: Path,
    max_concurrent: int,
    judge_model: str,
) -> None:
    """Evaluate all completed, unevaluated runs."""
    from autoreview.config.models import LLMConfig
    from autoreview.evaluation.arise_rubric import ARISERubricScorer
    from autoreview.evaluation.evaluator import run_evaluation
    from autoreview.llm.factory import create_llm_provider

    judge_config = LLMConfig(model=judge_model)
    judge_llm = create_llm_provider(judge_config)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval_single(key: str) -> None:
        entry = registry.runs[key]
        topic_id = parse_run_key(key)[0]
        topic = topic_lookup.get(topic_id)
        if not topic:
            return

        async with semaphore:
            logger.info("benchmark.eval.start", key=key)
            run_dir = Path(entry.output_dir)
            eval_path = run_dir / "evaluation.json"

            try:
                result = await run_evaluation(
                    generated_path=Path(entry.review_path),
                    reference_path=Path(topic.reference.pdf_path),
                    output_dir=run_dir,
                    judge_llm=judge_llm,
                )

                # ARISE rubric
                scorer = ARISERubricScorer(judge_llm)
                gen_text = Path(entry.review_path).read_text(encoding="utf-8")
                result.arise_result = await scorer.score(gen_text)

                eval_path.write_text(result.model_dump_json(indent=2))
                registry.register_evaluation(key, str(eval_path))
                registry.save(results_dir / "run_registry.json")
                logger.info("benchmark.eval.complete", key=key)

            except Exception as e:
                logger.error("benchmark.eval.failed", key=key, error=str(e))

    # AutoReview runs
    tasks = []
    for key, entry in registry.runs.items():
        if entry.status == "completed" and not entry.evaluation_path:
            tasks.append(_eval_single(key))

    # ARISE outputs
    arise_dir = results_dir / "arise"
    if arise_dir.exists():
        for topic_dir in arise_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            eval_path = topic_dir / "evaluation.json"
            if eval_path.exists():
                continue
            review_path = topic_dir / "review.md"
            if not review_path.exists():
                continue

            topic = topic_lookup.get(topic_dir.name)
            if not topic:
                continue

            async def _eval_arise(tp: Any, rp: Path, ep: Path) -> None:
                async with semaphore:
                    try:
                        result = await run_evaluation(
                            generated_path=rp,
                            reference_path=Path(tp.reference.pdf_path),
                            output_dir=rp.parent,
                            judge_llm=judge_llm,
                        )
                        scorer = ARISERubricScorer(judge_llm)
                        result.arise_result = await scorer.score(rp.read_text(encoding="utf-8"))
                        ep.write_text(result.model_dump_json(indent=2))
                    except Exception as e:
                        logger.error("benchmark.eval_arise.failed", topic=tp.id, error=str(e))

            tasks.append(_eval_arise(topic, review_path, eval_path))

    await asyncio.gather(*tasks, return_exceptions=True)


@app.command()
def analyze(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    output_dir: Path = typer.Option(Path("paper/output"), "--output-dir"),
    analyses: str | None = typer.Option(None, "--analyses", help="Comma-separated analysis numbers, e.g. 1,2,3"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run analysis scripts on evaluation results."""
    _setup_logging(verbose)
    config = load_topics(topics_path)

    which: list[int] | None = None
    if analyses:
        which = [int(x.strip()) for x in analyses.split(",")]

    asyncio.run(_run_analyses(config, results_dir, output_dir, which))


async def _run_analyses(
    topics: TopicsConfig,
    results_dir: Path,
    output_dir: Path,
    which: list[int] | None,
) -> None:
    """Run selected analysis scripts."""
    from paper.analysis.common import load_all_evaluations

    df = load_all_evaluations(results_dir, topics)
    logger.info("benchmark.analyze.loaded", n_rows=len(df))

    analysis_map: dict[int, tuple[str, Any]] = {
        1: ("main_comparison", None),
        2: ("domain_analysis", None),
        3: ("rubric_agreement", None),
        4: ("ablation_analysis", None),
        5: ("retrieval_decomposition", None),
        6: ("citation_analysis", None),
        7: ("model_comparison", None),
        8: ("cost_analysis", None),
        9: ("contamination_analysis", None),
        10: ("depth_comparison", None),
    }

    to_run = which or list(analysis_map.keys())

    for n in to_run:
        name, _ = analysis_map.get(n, (None, None))
        if not name:
            logger.warning("benchmark.analyze.unknown", analysis=n)
            continue

        analysis_out = output_dir / f"analysis_{n:02d}_{name}"
        logger.info("benchmark.analyze.start", analysis=n, name=name)

        try:
            if n == 10:
                # Generate depth_runs.json shim then call depth_comparison
                _generate_depth_shim(results_dir, analysis_out, topics)
                from paper.analysis.depth_comparison import main as depth_main
                await depth_main(analysis_out, analysis_out)
            else:
                mod = __import__(f"paper.analysis.{name}", fromlist=["main"])
                await mod.main(results_dir, analysis_out, df)
        except Exception as e:
            logger.error("benchmark.analyze.failed", analysis=n, error=str(e))


def _generate_depth_shim(results_dir: Path, output_dir: Path, topics: TopicsConfig) -> None:
    """Generate depth_runs.json from registry for Analysis 10."""
    import json
    from paper.models import RunRegistry, parse_run_key

    topic_lookup = {t.id: t for t in topics.topics}
    registry = RunRegistry.load(results_dir / "run_registry.json")
    runs = []
    for key, entry in registry.runs.items():
        if entry.status != "completed":
            continue
        topic_id, model, depth, condition = parse_run_key(key)
        if model != SONNET_MODEL or condition != "end_to_end":
            continue
        topic_info = topic_lookup.get(topic_id)
        runs.append({
            "topic": topic_id,
            "domain": topic_info.domain if topic_info else "unknown",
            "depth": depth,
            "generated_path": entry.review_path,
            "reference_path": topic_info.reference.pdf_path if topic_info else "",
            "evaluation_path": entry.evaluation_path,
            "tier": topic_info.tier if topic_info else "unknown",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_runs.json").write_text(json.dumps({"runs": runs}, indent=2))


@app.command()
def full(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    output_dir: Path = typer.Option(Path("paper/output"), "--output-dir"),
    models: str = typer.Option(",".join(DEFAULT_MODELS), "--models"),
    max_concurrent: int = typer.Option(2, "--max-concurrent"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full benchmark: generate -> evaluate -> analyze."""
    _setup_logging(verbose)
    typer.echo("=== Phase 1: Pipeline Runs ===")
    run(topics_path=topics_path, results_dir=results_dir, models=models, max_concurrent=max_concurrent, verbose=verbose)
    typer.echo("\n=== Phase 2: Evaluations ===")
    evaluate(topics_path=topics_path, results_dir=results_dir, max_concurrent=5, verbose=verbose)
    typer.echo("\n=== Phase 3: Analyses ===")
    analyze(topics_path=topics_path, results_dir=results_dir, output_dir=output_dir, verbose=verbose)


def _setup_logging(verbose: bool = False) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if verbose else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.processors.NAME_TO_LEVEL["debug" if verbose else "info"]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/run_benchmark.py tests/test_paper/test_run_benchmark.py
git commit -m "feat(paper): add benchmark orchestrator with generate-matrix, run, evaluate, analyze subcommands"
```

---

## Chunk 3: Analysis Scripts 1-5

Each analysis follows the same template. Tests use synthetic DataFrames — no LLM calls needed. All analyses accept `(results_dir: Path, output_dir: Path, df: pd.DataFrame)` so the orchestrator can pass the pre-loaded DataFrame.

**Note — deviation from spec:** The spec template shows `async def main(results_dir, output_dir)` (2 args). This plan uses `async def main(results_dir, output_dir, df)` (3 args) so the orchestrator loads the DataFrame once and passes it to all analyses. This avoids repeated disk I/O and ensures all analyses operate on the same snapshot. Implementers should use the 3-arg signature from this plan, not the spec template.

### Task 5: Analysis 1 — Main System Comparison

**Files:**
- Create: `paper/analysis/main_comparison.py`
- Create: `tests/test_paper/test_analysis_main_comparison.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_paper/test_analysis_main_comparison.py
"""Tests for Analysis 1: Main System Comparison."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic data: 3 systems x 3 topics."""
    rows = []
    for system, model in [("autoreview", "claude-sonnet-4-6"), ("arise", "arise")]:
        for topic in ["t1", "t2", "t3"]:
            rows.append({
                "topic_id": topic, "domain": "cs_ai", "tier": "B",
                "system": system, "model": model, "depth": "medium",
                "condition": "end_to_end",
                "overall_score": 0.7 if system == "autoreview" else 0.6,
                "synthesis_score": 3.5 if system == "autoreview" else 3.0,
                "topic_coverage": 0.8, "writing_quality": 3.5,
                "citation_recall": 0.6, "citation_precision": 0.7,
                "citation_f1": 0.65, "arise_total": 70.0,
                "word_count": 5000, "section_count": 8, "citation_count": 40,
                "citations_per_1000_words": 8.0, "flesch_kincaid_grade": 14.0,
                "cost_usd": 2.0, "tokens_input": 400000, "tokens_output": 80000,
            })
    return pd.DataFrame(rows)


class TestMainComparison:
    def test_compute_system_summary(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.main_comparison import compute_system_summary
        summary = compute_system_summary(sample_df)
        assert "autoreview" in summary
        assert "arise" in summary
        assert summary["autoreview"]["overall_score"]["mean"] == pytest.approx(0.7)

    def test_generate_report(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main
        import asyncio
        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        assert (tmp_path / "output" / "report.md").exists()
        assert (tmp_path / "output" / "analysis.json").exists()

    def test_figures_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main
        import asyncio
        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        assert (tmp_path / "output" / "system_comparison_bar.pdf").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_analysis_main_comparison.py -v`

- [ ] **Step 3: Implement Analysis 1**

Implementation pattern: load/filter → group by system → compute per-metric mean/std → Friedman test + post-hoc Wilcoxon → radar chart + grouped bar → markdown report + JSON export. See `paper/analysis/depth_comparison.py` for figure styling reference.

Key functions:
- `compute_system_summary(df) -> dict[str, dict[str, dict[str, float]]]`
- `compute_statistical_tests(df) -> dict`
- `plot_system_comparison_bar(df, out)`
- `plot_radar_chart(summary, out)`
- `generate_report(summary, tests, out)`
- `main(results_dir, output_dir, df)`

- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

```bash
git add paper/analysis/main_comparison.py tests/test_paper/test_analysis_main_comparison.py
git commit -m "feat(paper): add Analysis 1 — main system comparison"
```

---

### Task 6: Analysis 2 — Cross-Domain Variation

**Files:**
- Create: `paper/analysis/domain_analysis.py`
- Create: `tests/test_paper/test_analysis_domain.py`

- [ ] **Step 1: Write tests** — Follow Task 5 pattern. Key assertion:

```python
def test_compute_domain_summary(self, sample_df):
    from paper.analysis.domain_analysis import compute_domain_summary
    summary = compute_domain_summary(sample_df)
    assert "cs_ai" in summary
    assert "biomedical" in summary

def test_kruskal_wallis_runs(self, sample_df):
    from paper.analysis.domain_analysis import compute_domain_tests
    tests = compute_domain_tests(sample_df)
    assert "overall_score" in tests
    assert "p_value" in tests["overall_score"]
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Group by domain × system, Kruskal-Wallis per metric, heatmap + grouped bar. Functions: `compute_domain_summary(df)`, `compute_domain_tests(df)`, `plot_domain_heatmap(df, out)`, `plot_domain_bar(df, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 2 — cross-domain variation"
```

---

### Task 7: Analysis 3 — Cross-Rubric Agreement

**Files:**
- Create: `paper/analysis/rubric_agreement.py`
- Create: `tests/test_paper/test_analysis_rubric.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_compute_rubric_correlation(self, sample_df):
    from paper.analysis.rubric_agreement import compute_rubric_correlation
    corr = compute_rubric_correlation(sample_df)
    assert "spearman_rho" in corr
    assert "pearson_r" in corr
    assert -1 <= corr["spearman_rho"] <= 1

def test_generates_scatter(self, sample_df, tmp_path):
    from paper.analysis.rubric_agreement import main
    import asyncio
    asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
    assert (tmp_path / "out" / "rubric_scatter.pdf").exists()
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Spearman/Pearson between `overall_score` and `arise_total`, scatter + Bland-Altman. Functions: `compute_rubric_correlation(df)`, `plot_rubric_scatter(df, out)`, `plot_bland_altman(df, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 3 — cross-rubric agreement"
```

---

### Task 8: Analysis 4 — Component Ablation

**Files:**
- Create: `paper/analysis/ablation_analysis.py`
- Create: `tests/test_paper/test_analysis_ablation.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_compute_ablation_deltas(self, sample_df):
    from paper.analysis.ablation_analysis import compute_ablation_deltas
    deltas = compute_ablation_deltas(sample_df)
    assert "no_critique_loops" in deltas
    assert "overall_score" in deltas["no_critique_loops"]

def test_heatmap_generated(self, sample_df, tmp_path):
    from paper.analysis.ablation_analysis import main
    import asyncio
    asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
    assert (tmp_path / "out" / "ablation_heatmap.pdf").exists()
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Filter ablation topics (condition != "end_to_end"), compute delta from baseline, paired Wilcoxon per condition, heatmap. Functions: `compute_ablation_deltas(df)`, `plot_ablation_heatmap(deltas, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 4 — component ablation"
```

---

### Task 9: Analysis 5 — Retrieval vs Synthesis Decomposition

**Files:**
- Create: `paper/analysis/retrieval_decomposition.py`
- Create: `tests/test_paper/test_analysis_retrieval.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_compute_condition_comparison(self, sample_df):
    from paper.analysis.retrieval_decomposition import compute_condition_comparison
    comp = compute_condition_comparison(sample_df)
    assert "end_to_end" in comp
    assert "retrieval_controlled" in comp

def test_grouped_bar_generated(self, sample_df, tmp_path):
    from paper.analysis.retrieval_decomposition import main
    import asyncio
    asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
    assert (tmp_path / "out" / "retrieval_decomposition_bar.pdf").exists()
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Filter Tier A, compare end_to_end vs retrieval_controlled, grouped bar (e2e vs controlled). Functions: `compute_condition_comparison(df)`, `plot_condition_bar(comp, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 5 — retrieval vs synthesis decomposition"
```

---

## Chunk 4: Analysis Scripts 6-9

### Task 10: Analysis 6 — Citation Quality

**Files:**
- Create: `paper/analysis/citation_analysis.py`
- Create: `tests/test_paper/test_analysis_citation.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_citation_breakdown_by_domain(self, sample_df):
    from paper.analysis.citation_analysis import compute_citation_breakdown
    breakdown = compute_citation_breakdown(sample_df, group_by="domain")
    assert "cs_ai" in breakdown
    assert "mean_recall" in breakdown["cs_ai"]

def test_hallucination_rate(self, sample_df):
    from paper.analysis.citation_analysis import compute_hallucination_rate
    rate = compute_hallucination_rate(sample_df)
    assert 0 <= rate <= 1
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Citation metrics by domain/tier/model, hallucination rate. Functions: `compute_citation_breakdown(df, group_by)`, `compute_hallucination_rate(df)`, `plot_citation_bar(breakdown, out)`, `plot_hallucination_histogram(df, out)`, `main(results_dir, output_dir, df)`. Note: hallucination data comes from `CitationScore.hallucinated_titles` in the evaluation JSON — use `len(hallucinated_titles) / generated_count` for the rate.
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 6 — citation quality"
```

---

### Task 11: Analysis 7 — Model Tier Comparison

**Files:**
- Create: `paper/analysis/model_comparison.py`
- Create: `tests/test_paper/test_analysis_model.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_model_tier_summary(self, sample_df):
    from paper.analysis.model_comparison import compute_model_summary
    summary = compute_model_summary(sample_df)
    assert "claude-sonnet-4-6" in summary
    assert "overall_score" in summary["claude-sonnet-4-6"]

def test_line_chart_generated(self, sample_df, tmp_path):
    from paper.analysis.model_comparison import main
    import asyncio
    asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
    assert (tmp_path / "out" / "model_comparison_line.pdf").exists()
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Filter medium depth, group by model, Friedman + post-hoc Wilcoxon with FDR. Functions: `compute_model_summary(df)`, `compute_model_tests(df)`, `plot_model_line(summary, out)`, `plot_cost_frontier(df, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 7 — model tier comparison"
```

---

### Task 12: Analysis 8 — Cost-Quality Tradeoff

**Files:**
- Create: `paper/analysis/cost_analysis.py`
- Create: `tests/test_paper/test_analysis_cost.py`

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_cost_per_quality_point(self, sample_df):
    from paper.analysis.cost_analysis import compute_cost_efficiency
    eff = compute_cost_efficiency(sample_df)
    assert "cost_per_quality_point" in eff.columns
    assert all(eff["cost_per_quality_point"] >= 0)

def test_scatter_generated(self, sample_df, tmp_path):
    from paper.analysis.cost_analysis import main
    import asyncio
    asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
    assert (tmp_path / "out" / "cost_quality_scatter.pdf").exists()
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — Compute `cost_per_quality_point = cost_usd / overall_score` and `cost_per_citation_f1 = cost_usd / citation_f1`. Functions: `compute_cost_efficiency(df)`, `plot_cost_quality_scatter(df, out)`, `main(results_dir, output_dir, df)`
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 8 — cost-quality tradeoff"
```

---

### Task 13: Analysis 9 — Contamination Analysis

**Files:**
- Create: `paper/analysis/contamination_analysis.py`
- Create: `tests/test_paper/test_analysis_contamination.py`

**Note:** This analysis needs the actual text of both generated reviews and reference PDFs. Generated reviews are loaded from `review_path` (markdown). Reference text is extracted from PDFs using the existing `autoreview.evaluation.pdf_extractor.extract_text_from_pdf()` function. The analysis reads `reference.pdf_path` from topics.yaml for each topic.

- [ ] **Step 1: Write tests** — Key assertion:

```python
def test_ngram_overlap(self):
    from paper.analysis.contamination_analysis import compute_ngram_overlap
    text_a = "the quick brown fox jumps over the lazy dog"
    text_b = "the quick brown fox runs past the lazy cat"
    overlap = compute_ngram_overlap(text_a, text_b, n=3)
    assert 0 < overlap < 1  # Partial overlap

def test_no_overlap(self):
    from paper.analysis.contamination_analysis import compute_ngram_overlap
    overlap = compute_ngram_overlap("alpha beta gamma", "delta epsilon zeta", n=2)
    assert overlap == 0.0

def test_perfect_overlap(self):
    from paper.analysis.contamination_analysis import compute_ngram_overlap
    text = "the quick brown fox"
    overlap = compute_ngram_overlap(text, text, n=2)
    assert overlap == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify fail**
- [ ] **Step 3: Implement** — `compute_ngram_overlap(text_a, text_b, n)` returns Jaccard index of n-gram sets. `compute_all_overlaps(results_dir, topics, registry)` iterates topics, extracts texts, computes 1-5 gram overlaps. Uses `extract_text_from_pdf()` for reference PDFs. Functions: `compute_ngram_overlap(text_a, text_b, n)`, `compute_all_overlaps(results_dir, topics, registry)`, `plot_overlap_histogram(overlaps, out)`, `plot_tier_comparison(overlaps, out)`, `main(results_dir, output_dir, df)`. **Note:** `main()` also accepts `df` for consistency but primarily reads text files directly.
- [ ] **Step 4: Run tests to verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(paper): add Analysis 9 — contamination analysis"
```

---

## Chunk 5: Ablation Prerequisites

These are small changes to the pipeline runner. They only block batch 3e (ablation runs).

### Task 14: Add `skip_nodes` Parameter to Pipeline Runner

**Files:**
- Modify: `autoreview/pipeline/runner.py:126-132`
- Modify: `autoreview/pipeline/dag.py:134-141`
- Create: `tests/test_pipeline/test_skip_nodes.py`

- [ ] **Step 1: Write test for skip_nodes in DAGRunner**

```python
# tests/test_pipeline/test_skip_nodes.py
"""Tests for skip_nodes functionality."""
from __future__ import annotations

import pytest

from autoreview.pipeline.dag import DAGRunner


@pytest.mark.asyncio
async def test_skip_nodes_bypasses_node() -> None:
    dag = DAGRunner()
    call_log: list[str] = []

    async def node_a(ctx: dict) -> None:
        call_log.append("a")

    async def node_b(ctx: dict) -> None:
        call_log.append("b")

    async def node_c(ctx: dict) -> None:
        call_log.append("c")

    dag.add_node("a", node_a)
    dag.add_node("b", node_b, dependencies=["a"])
    dag.add_node("c", node_c, dependencies=["b"])

    await dag.execute({}, skip_nodes={"b"})
    assert "a" in call_log
    assert "b" not in call_log
    assert "c" in call_log
```

- [ ] **Step 2: Run test to verify fail**

Run: `python -m pytest tests/test_pipeline/test_skip_nodes.py -v`
Expected: FAIL — `execute() got unexpected keyword argument 'skip_nodes'`

- [ ] **Step 3: Add `skip_nodes` parameter to `DAGRunner.execute()`**

In `autoreview/pipeline/dag.py`, modify `execute()` signature to accept `skip_nodes: set[str] | None = None`. In `_run_node`, if node name is in skip_nodes, skip execution and log a warning.

- [ ] **Step 4: Thread `skip_nodes` through `run_pipeline()`**

In `autoreview/pipeline/runner.py`, add `skip_nodes: set[str] | None = None` parameter to `run_pipeline()` and pass it to `dag.execute()`.

- [ ] **Step 5: Run tests to verify pass**
- [ ] **Step 6: Commit**

```bash
git add autoreview/pipeline/dag.py autoreview/pipeline/runner.py tests/test_pipeline/test_skip_nodes.py
git commit -m "feat(pipeline): add skip_nodes parameter for ablation support"
```

---

### Task 15: Add `evidence_chains` Flag to WritingConfig

**Files:**
- Modify: `autoreview/config/models.py:126-135`
- Modify: `autoreview/pipeline/nodes.py` (section_writing node)
- Create: `tests/test_pipeline/test_evidence_chains_flag.py`

- [ ] **Step 1: Write test**

```python
# tests/test_pipeline/test_evidence_chains_flag.py
"""Test evidence_chains config flag."""
from __future__ import annotations

from autoreview.config.models import WritingConfig


def test_evidence_chains_default_true() -> None:
    config = WritingConfig()
    assert config.evidence_chains is True


def test_evidence_chains_can_disable() -> None:
    config = WritingConfig(evidence_chains=False)
    assert config.evidence_chains is False
```

- [ ] **Step 2: Run test to verify fail**
- [ ] **Step 3: Add `evidence_chains: bool = True` to `WritingConfig`**
- [ ] **Step 4: Thread through section_writing node** — When `config.writing.evidence_chains is False`, skip evidence chain construction in the section writing prompt
- [ ] **Step 5: Run tests to verify pass**
- [ ] **Step 6: Commit**

```bash
git add autoreview/config/models.py autoreview/pipeline/nodes.py tests/test_pipeline/test_evidence_chains_flag.py
git commit -m "feat(config): add evidence_chains flag for ablation support"
```

---

## Dependency Graph

```
Task 1 (models) ──────────┬──→ Task 4 (orchestrator)
Task 2 (topics.yaml) ─────┤
Task 3 (common.py) ───────┼──→ Tasks 5-13 (analyses 1-9)
                           │
Task 14 (skip_nodes) ──────┤  ← only blocks batch 3e ablation runs
Task 15 (evidence_chains) ─┘
```

**Parallel batches:**
- Batch 1: Tasks 1, 2, 3 (all independent)
- Batch 2: Task 4 (depends on Task 1)
- Batch 3: Tasks 5-13 (depend on Task 3, independent of each other)
- Batch 4: Tasks 14-15 (independent, can parallel with Batch 3)

---

## Verification Checklist

After all tasks complete:

- [ ] `python -m pytest tests/test_paper/ -v` — all paper tests pass
- [ ] `python -m pytest tests/test_pipeline/test_skip_nodes.py tests/test_pipeline/test_evidence_chains_flag.py -v` — ablation tests pass
- [ ] `python -m paper.run_benchmark generate-matrix` — shows run matrix with cost estimate
- [ ] `python -m paper.run_benchmark run --dry-run` — shows remaining runs without executing
- [ ] All 10 analysis scripts importable: `python -c "from paper.analysis.main_comparison import main; print('ok')"`
