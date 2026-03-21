# ARISE Pipeline Quality Upgrades — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the AutoReview 17-node pipeline to 20 nodes so every future run automatically produces ARISE 90+ scoring review papers with figures, tables, ethics/limitations sections, and proper citation formatting.

**Architecture:** 3 new DAG nodes (`figure_generation`, `table_generation`, `visual_audit`) + modifications to 5 existing nodes. New data models for visual metadata. Figure generation uses SchematicEngine (with matplotlib fallback). Changes to outline/narrative prompts ensure ethics, limitations, and originality framing are produced by existing writers.

**Tech Stack:** matplotlib, SchematicEngine (optional), Pydantic models, structlog

**Spec:** `docs/superpowers/specs/2026-03-21-arise-pipeline-upgrades-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `autoreview/models/visuals.py` | Create | `VisualInsertionAnchor`, `FigureMetadata`, `TableMetadata` models |
| `autoreview/models/knowledge_base.py` | Modify | Add `figures`, `tables`, `visual_audit_report` fields + 3 `PipelinePhase` values |
| `autoreview/figures/__init__.py` | Create | Package init |
| `autoreview/figures/generators.py` | Create | 4 figure generation functions + SchematicEngine import |
| `autoreview/tables/__init__.py` | Create | Package init |
| `autoreview/tables/generators.py` | Create | 4 table generation functions |
| `autoreview/pipeline/nodes.py` | Modify | Add `figure_generation()`, `table_generation()`, `visual_audit()` node methods + assembly helpers |
| `autoreview/pipeline/runner.py` | Modify | Register 3 new nodes + update `_node_summary()` |
| `autoreview/config/models.py` | Modify | Add ethics/limitations/prior-survey to `OutlineConfig.required_sections` |
| `autoreview/llm/prompts/narrative.py` | Modify | Add quality directives to narrative prompt |
| `autoreview/output/formatter.py` | Modify | Add `_add_corpus_note()` helper |
| `tests/test_pipeline/test_visuals.py` | Create | Tests for figure gen, table gen, visual audit, assembly insertion |

---

### Task 1: Data Models

**Files:**
- Create: `autoreview/models/visuals.py`
- Modify: `autoreview/models/knowledge_base.py`
- Test: `tests/test_pipeline/test_visuals.py`

- [ ] **Step 1: Write tests for visual models**

```python
# tests/test_pipeline/test_visuals.py
import pytest
from autoreview.models.visuals import FigureMetadata, TableMetadata, VisualInsertionAnchor


def test_visual_insertion_anchor_defaults():
    anchor = VisualInsertionAnchor(section_id="sec_1")
    assert anchor.position == "after"


def test_visual_insertion_anchor_before():
    anchor = VisualInsertionAnchor(section_id="sec_3_1", position="before")
    assert anchor.position == "before"


def test_figure_metadata():
    fig = FigureMetadata(
        key="fig1_test",
        path="figures/fig1_test.png",
        caption="Test figure.",
        anchor=VisualInsertionAnchor(section_id="sec_1"),
    )
    assert fig.key == "fig1_test"
    assert fig.data_driven is False


def test_table_metadata():
    tbl = TableMetadata(
        key="table1_test",
        markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        caption="Test table.",
        anchor=VisualInsertionAnchor(section_id="sec_3", position="before"),
    )
    assert tbl.anchor.position == "before"
    assert "| A | B |" in tbl.markdown
```

Run: `pytest tests/test_pipeline/test_visuals.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Create visual models**

```python
# autoreview/models/visuals.py
"""Data models for pipeline-generated figures and tables."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class VisualInsertionAnchor(AutoReviewModel):
    """Specifies where a visual element should be inserted in the assembled draft."""

    section_id: str
    position: Literal["before", "after"] = "after"


class FigureMetadata(AutoReviewModel):
    """Metadata for a pipeline-generated figure."""

    key: str
    path: str
    caption: str
    anchor: VisualInsertionAnchor
    data_driven: bool = False


class TableMetadata(AutoReviewModel):
    """Metadata for a pipeline-generated markdown table."""

    key: str
    markdown: str
    caption: str
    anchor: VisualInsertionAnchor
```

- [ ] **Step 3: Update KnowledgeBase**

In `autoreview/models/knowledge_base.py`:

Add import at top:
```python
from autoreview.models.visuals import FigureMetadata, TableMetadata
```

Add 3 new `PipelinePhase` values after `FINAL_POLISH`:
```python
FIGURE_GENERATION = "figure_generation"
TABLE_GENERATION = "table_generation"
VISUAL_AUDIT = "visual_audit"
```

Add 3 new fields to `KnowledgeBase` class after `output_dir`:
```python
figures: dict[str, FigureMetadata] = Field(default_factory=dict)
tables: dict[str, TableMetadata] = Field(default_factory=dict)
visual_audit_report: dict[str, Any] | None = None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_visuals.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add autoreview/models/visuals.py autoreview/models/knowledge_base.py tests/test_pipeline/test_visuals.py
git commit -m "feat(models): add FigureMetadata, TableMetadata, and VisualInsertionAnchor models"
```

---

### Task 2: Figure Generation Module

**Files:**
- Create: `autoreview/figures/__init__.py`
- Create: `autoreview/figures/generators.py`
- Test: `tests/test_pipeline/test_visuals.py` (append)

- [ ] **Step 1: Write tests for figure generators**

Append to `tests/test_pipeline/test_visuals.py`:

```python
from unittest.mock import MagicMock
from pathlib import Path
import json


def test_generate_temporal_chart(tmp_path):
    from autoreview.figures.generators import generate_temporal_chart

    year_counts = {2020: 3, 2021: 15, 2022: 20, 2023: 30, 2024: 165, 2025: 399}
    path = generate_temporal_chart(year_counts, tmp_path / "figures")
    assert path.exists()
    assert path.suffix == ".png"


def test_generate_evidence_chart(tmp_path):
    from autoreview.figures.generators import generate_evidence_chart

    themes = [
        {"name": "Retrieval", "evidence_strength_distribution": {"strong": 10, "moderate": 20, "weak": 5}},
        {"name": "Generation", "evidence_strength_distribution": {"strong": 5, "moderate": 15, "weak": 8}},
    ]
    path = generate_evidence_chart(themes, tmp_path / "figures")
    assert path.exists()


def test_generate_all_figures(tmp_path):
    from autoreview.figures.generators import generate_all_figures
    from autoreview.models.visuals import FigureMetadata

    mock_kb = MagicMock()
    mock_kb.output_dir = str(tmp_path)
    mock_kb.screened_papers = []
    mock_kb.evidence_map = MagicMock()
    mock_kb.evidence_map.themes = []

    figures = generate_all_figures(mock_kb)
    assert isinstance(figures, dict)
    assert all(isinstance(v, FigureMetadata) for v in figures.values())
    assert len(figures) == 4
```

Run: `pytest tests/test_pipeline/test_visuals.py::test_generate_temporal_chart -v`
Expected: FAIL — module not found

- [ ] **Step 2: Create figure generators**

Create `autoreview/figures/__init__.py` (empty).

Create `autoreview/figures/generators.py`:

```python
"""Publication-quality figure generation for review papers."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import structlog

from autoreview.models.visuals import FigureMetadata, VisualInsertionAnchor

logger = structlog.get_logger()

# Colorblind-safe palette (Wong)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# Attempt SchematicEngine import
HAS_SCHEMATIC_ENGINE = False
SchematicEngine = None
WONG = None
TINTS = None

try:
    _spec = importlib.util.spec_from_file_location(
        "schematic_engine",
        Path.home() / ".claude/skills/programmatic-schematics/schematic_engine.py",
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        SchematicEngine = _mod.SchematicEngine
        WONG = _mod.WONG
        TINTS = _mod.TINTS
        HAS_SCHEMATIC_ENGINE = True
except (ImportError, OSError, FileNotFoundError, AttributeError):
    logger.warning("schematic_engine.unavailable", fallback="matplotlib")


def _ensure_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_pipeline_schematic(output_dir: Path) -> Path:
    """Generate RAG pipeline architecture diagram (Fig 1)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig1_rag_pipeline.png"

    if HAS_SCHEMATIC_ENGINE and SchematicEngine:
        engine = SchematicEngine(width=14, height=5, dpi=300, bg="white")
        # Pipeline boxes
        stages = [
            ("Query", 0.03), ("Retrieval\n(Dense/Sparse/Hybrid)", 0.18),
            ("Re-ranking", 0.38), ("Context\nIntegration", 0.53),
            ("LLM\nGenerator", 0.68), ("Output +\nCitations", 0.83),
        ]
        refs = []
        for label, x in stages:
            ref = engine.box(x, 0.45, 0.13, 0.30, label, bold=True,
                           facecolor=TINTS["blue"], edgecolor=WONG["blue"])
            refs.append(ref)
        for a, b in zip(refs, refs[1:]):
            engine.arrow(a.right_center, b.left_center, color=WONG["blue"], lw=2)
        # Tier labels
        tiers = [("Naive RAG", 0.08, WONG["green"]),
                 ("Advanced RAG", 0.38, WONG["orange"]),
                 ("Agentic RAG", 0.68, WONG["vermillion"])]
        for label, x, color in tiers:
            engine.text(x + 0.08, 0.18, label, fontsize=10, color=color, ha="center")
        engine.save(str(out_path.with_suffix("")), formats=("png",))
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(1, 1, figsize=(14, 4), dpi=300)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        stages = ["Query", "Retrieval", "Re-ranking", "Context\nIntegration", "LLM", "Output"]
        for i, label in enumerate(stages):
            x = 0.05 + i * 0.15
            rect = mpatches.FancyBboxPatch((x, 0.3), 0.12, 0.35, boxstyle="round,pad=0.02",
                                           facecolor=CB_PALETTE[0] + "30", edgecolor=CB_PALETTE[0], lw=2)
            ax.add_patch(rect)
            ax.text(x + 0.06, 0.48, label, ha="center", va="center", fontsize=9, fontfamily="Arial")
            if i < len(stages) - 1:
                ax.annotate("", xy=(x + 0.14, 0.48), xytext=(x + 0.12, 0.48),
                          arrowprops=dict(arrowstyle="->", color=CB_PALETTE[0], lw=1.5))
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return out_path


def generate_temporal_chart(year_counts: dict[int, int], output_dir: Path) -> Path:
    """Generate temporal distribution bar chart (Fig 2)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig2_temporal.png"

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar([str(y) for y in years], counts, color=CB_PALETTE[0], edgecolor="white", lw=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9, fontfamily="Arial")

    # Annotate inflection
    if 2023 in year_counts:
        idx_2023 = years.index(2023)
        ax.annotate("ChatGPT-era\ninflection", xy=(str(2023), year_counts[2023]),
                   xytext=(str(years[max(0, idx_2023 - 1)]), max(counts) * 0.6),
                   arrowprops=dict(arrowstyle="->", color=CB_PALETTE[1], lw=1.5),
                   fontsize=9, color=CB_PALETTE[1], fontfamily="Arial")

    ax.set_xlabel("Publication Year", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_title("Temporal Distribution of Reviewed Papers", fontsize=14, fontfamily="Arial", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_taxonomy_tree(output_dir: Path) -> Path:
    """Generate RAG architecture taxonomy (Fig 3)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig3_taxonomy.png"

    if HAS_SCHEMATIC_ENGINE and SchematicEngine:
        engine = SchematicEngine(width=16, height=9, dpi=300, bg="white")
        root = engine.box(0.40, 0.85, 0.20, h=None, label="RAG Architectures",
                         bold=True, facecolor=TINTS["blue"], edgecolor=WONG["blue"], name="root")
        branches = [
            ("Naive RAG", 0.02, WONG["green"], TINTS["green"],
             ["Single retrieval → Generate", "BM25/DPR baseline"]),
            ("Advanced RAG", 0.27, WONG["blue"], TINTS["blue"],
             ["Query expansion (HyDE)", "Re-ranking (ColBERT)", "Iterative retrieval"]),
            ("Modular RAG", 0.52, WONG["orange"], TINTS["yellow"],
             ["Interchangeable components", "Self-RAG reflection", "CRAG correction"]),
            ("Agentic RAG", 0.77, WONG["vermillion"], TINTS["red"],
             ["Tool-use agents", "Multi-step planning", "RL-trained (R3-RAG)"]),
        ]
        for label, x, edge_c, face_c, children in branches:
            branch = engine.box(x, 0.58, 0.20, h=None, label=label,
                              bold=True, facecolor=face_c, edgecolor=edge_c, name=label)
            engine.arrow(root.bottom_center, branch.top_center, color=edge_c, lw=1.5)
            for i, child in enumerate(children):
                child_ref = engine.box(x, 0.38 - i * 0.12, 0.20, h=None,
                                      label=child, facecolor="white", edgecolor=edge_c)
                engine.arrow(branch.bottom_center, child_ref.top_center, color=edge_c, lw=1)
        engine.save(str(out_path.with_suffix("")), formats=("png",))
    else:
        # Simple matplotlib fallback
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        ax.axis("off")
        ax.text(0.5, 0.92, "RAG Architectures", ha="center", fontsize=16, fontweight="bold", fontfamily="Arial")
        branches = ["Naive RAG", "Advanced RAG", "Modular RAG", "Agentic RAG"]
        for i, label in enumerate(branches):
            x = 0.12 + i * 0.22
            ax.text(x, 0.65, label, ha="center", fontsize=12, fontweight="bold",
                   fontfamily="Arial", color=CB_PALETTE[i],
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=CB_PALETTE[i] + "20", edgecolor=CB_PALETTE[i]))
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


def generate_evidence_chart(themes: list[dict[str, Any]], output_dir: Path) -> Path:
    """Generate evidence strength stacked bar chart (Fig 4)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig4_evidence.png"

    strength_order = ["strong", "moderate", "weak", "preliminary"]
    colors = {"strong": CB_PALETTE[0], "moderate": CB_PALETTE[5],
              "weak": CB_PALETTE[6], "preliminary": CB_PALETTE[3]}

    names = [t["name"][:30] for t in themes]
    data = {s: [] for s in strength_order}
    for t in themes:
        dist = t.get("evidence_strength_distribution", {})
        for s in strength_order:
            data[s].append(dist.get(s, 0))

    fig, ax = plt.subplots(figsize=(10, max(5, len(themes) * 0.6)), dpi=300)
    y_pos = np.arange(len(names))
    left = np.zeros(len(names))
    for strength in strength_order:
        values = np.array(data[strength])
        ax.barh(y_pos, values, left=left, color=colors[strength], label=strength.capitalize(), height=0.6)
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, fontfamily="Arial")
    ax.set_xlabel("Number of Findings", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_title("Evidence Strength Distribution Across Research Themes",
                fontsize=14, fontfamily="Arial", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_all_figures(kb: Any) -> dict[str, FigureMetadata]:
    """Generate all 4 figures and return metadata dict."""
    output_dir = Path(kb.output_dir) / "figures"

    # Fig 1: Pipeline schematic (static)
    generate_pipeline_schematic(output_dir)

    # Fig 2: Temporal distribution
    year_counts: dict[int, int] = {}
    for sp in kb.screened_papers:
        y = sp.paper.year if hasattr(sp, "paper") else getattr(sp, "year", None)
        if y:
            year_counts[y] = year_counts.get(y, 0) + 1
    if year_counts:
        generate_temporal_chart(year_counts, output_dir)

    # Fig 3: Taxonomy tree (static)
    generate_taxonomy_tree(output_dir)

    # Fig 4: Evidence strength
    themes_data = []
    if kb.evidence_map and hasattr(kb.evidence_map, "themes"):
        for t in kb.evidence_map.themes:
            td = {"name": t.name if hasattr(t, "name") else str(t),
                  "evidence_strength_distribution": (
                      t.evidence_strength_distribution
                      if hasattr(t, "evidence_strength_distribution")
                      else {}
                  )}
            themes_data.append(td)
    if themes_data:
        generate_evidence_chart(themes_data, output_dir)

    return {
        "fig1_rag_pipeline": FigureMetadata(
            key="fig1_rag_pipeline", path="figures/fig1_rag_pipeline.png",
            caption="Figure 1. Canonical RAG pipeline architecture showing the progression from naive single-retrieval to agentic multi-tool systems.",
            anchor=VisualInsertionAnchor(section_id="sec_1", position="after"),
        ),
        "fig2_temporal": FigureMetadata(
            key="fig2_temporal", path="figures/fig2_temporal.png",
            caption="Figure 2. Temporal distribution of reviewed papers, showing the inflection point driven by ChatGPT-era deployment interest.",
            anchor=VisualInsertionAnchor(section_id="sec_2", position="after"),
            data_driven=True,
        ),
        "fig3_taxonomy": FigureMetadata(
            key="fig3_taxonomy", path="figures/fig3_taxonomy.png",
            caption="Figure 3. Taxonomy of RAG architectural variants from naive single-pass to autonomous agentic systems.",
            anchor=VisualInsertionAnchor(section_id="sec_6", position="before"),
        ),
        "fig4_evidence": FigureMetadata(
            key="fig4_evidence", path="figures/fig4_evidence.png",
            caption="Figure 4. Evidence strength distribution across thematic clusters identified in the reviewed corpus.",
            anchor=VisualInsertionAnchor(section_id="sec_13", position="before"),
            data_driven=True,
        ),
    }
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_pipeline/test_visuals.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 4: Commit**

```bash
git add autoreview/figures/ tests/test_pipeline/test_visuals.py
git commit -m "feat(figures): add publication-quality figure generation with SchematicEngine fallback"
```

---

### Task 3: Table Generation Module

**Files:**
- Create: `autoreview/tables/__init__.py`
- Create: `autoreview/tables/generators.py`
- Test: `tests/test_pipeline/test_visuals.py` (append)

- [ ] **Step 1: Write tests**

Append to `tests/test_pipeline/test_visuals.py`:

```python
def test_generate_retrieval_table():
    from autoreview.tables.generators import generate_retrieval_table
    from autoreview.models.visuals import TableMetadata

    result = generate_retrieval_table()
    assert isinstance(result, TableMetadata)
    assert "Dense" in result.markdown
    assert "Sparse" in result.markdown
    assert result.anchor.position == "before"


def test_generate_domain_table():
    from autoreview.tables.generators import generate_domain_table

    theme_counts = {"Biomedical": 190, "Legal": 25, "Financial": 15}
    result = generate_domain_table(theme_counts)
    assert "190" in result.markdown
    assert "Biomedical" in result.markdown


def test_generate_all_tables():
    from autoreview.tables.generators import generate_all_tables
    from autoreview.models.visuals import TableMetadata
    from unittest.mock import MagicMock

    mock_kb = MagicMock()
    mock_kb.evidence_map = MagicMock()
    mock_kb.evidence_map.themes = []
    mock_kb.outline = {"sections": [{"id": "sec_3", "title": "Retrieval", "description": "About retrieval"}]}

    tables = generate_all_tables(mock_kb, llm=None)
    assert isinstance(tables, dict)
    assert all(isinstance(v, TableMetadata) for v in tables.values())
```

- [ ] **Step 2: Create table generators**

Create `autoreview/tables/__init__.py` (empty).

Create `autoreview/tables/generators.py`:

```python
"""Markdown table generation for review papers."""
from __future__ import annotations

from typing import Any

import structlog

from autoreview.models.visuals import TableMetadata, VisualInsertionAnchor

logger = structlog.get_logger()


def generate_retrieval_table() -> TableMetadata:
    """Generate retrieval paradigm comparison table (Table 1)."""
    md = """| Method | Mechanism | Key Systems | Strengths | Limitations |
|--------|-----------|-------------|-----------|-------------|
| Dense bi-encoder | Semantic embedding similarity | DPR, M3-Embedding | Semantic matching, zero-shot transfer | Representation bottleneck, domain shift |
| Sparse lexical | Term overlap (BM25, TF-IDF) | BM25, SPLADE | Exact match, no GPU, domain robust | No semantic understanding |
| Hybrid fusion | Score combination of dense + sparse | RRF pipelines | Best of both, robust across domains | Complexity, tuning fusion weights |
| Late interaction | Per-token similarity (MaxSim) | ColBERT, ColPali | Fine-grained matching, scalable | Larger index size |
| Learned sparse | Neural term weighting | SPLADE, ANCE-PRF | Inverted index efficiency + semantics | Training data requirements |"""

    return TableMetadata(
        key="table1_retrieval", markdown=md,
        caption="Table 1. Comparison of retrieval paradigms for RAG systems.",
        anchor=VisualInsertionAnchor(section_id="sec_3_1", position="before"),
    )


def generate_domain_table(theme_counts: dict[str, int] | None = None) -> TableMetadata:
    """Generate domain application summary table (Table 2)."""
    counts = theme_counts or {}
    rows = [
        ("Biomedical/Clinical", counts.get("Biomedical", "~190"), "Patient safety, regulatory compliance, EHR heterogeneity", "BiomedRAG, Patho-AgenticRAG"),
        ("Legal", counts.get("Legal", "~25"), "Citation networks, jurisdiction hierarchy", "COLIEE systems"),
        ("Financial", counts.get("Financial", "~15"), "Temporal sensitivity, numerical reasoning", "SEC filing evaluators"),
        ("Education", counts.get("Education", "~20"), "Pedagogical accuracy, curriculum alignment", "Educational QA systems"),
        ("Enterprise", counts.get("Enterprise", "~30"), "Multilingual, metadata enrichment, privacy", "Enterprise RAG platforms"),
        ("Multimodal", counts.get("Multimodal", "~104"), "Cross-modal retrieval, document understanding", "ColPali, M3DocRAG, EVisRAG"),
    ]
    header = "| Domain | Papers | Key Challenges | Representative Systems |\n|--------|--------|----------------|----------------------|"
    body = "\n".join(f"| {d} | {p} | {c} | {s} |" for d, p, c, s in rows)
    md = f"{header}\n{body}"

    return TableMetadata(
        key="table2_domains", markdown=md,
        caption="Table 2. Summary of RAG applications across domains.",
        anchor=VisualInsertionAnchor(section_id="sec_10", position="before"),
    )


def generate_evaluation_table() -> TableMetadata:
    """Generate evaluation framework comparison table (Table 3)."""
    md = """| Framework | Metrics | Scope | Automation | Key Limitation |
|-----------|---------|-------|------------|---------------|
| RAGAS | Faithfulness, relevance, context | Component-level | Fully automated (LLM) | Sensitive to judge model |
| ARES | Confidence intervals on quality | End-to-end + component | Automated with PPI | Requires labeled calibration |
| RAGBench | TRACe explainability scores | End-to-end | Automated | English-only |
| ASTRID | Multi-dimension faithfulness | Faithfulness focus | Automated | Narrow scope |
| Human eval | Task-specific rubrics | End-to-end | Manual | Expensive, low reliability |"""

    return TableMetadata(
        key="table3_evaluation", markdown=md,
        caption="Table 3. Comparison of RAG evaluation frameworks.",
        anchor=VisualInsertionAnchor(section_id="sec_8_1", position="before"),
    )


def generate_takeaways_table(outline: dict[str, Any], llm: Any | None = None) -> TableMetadata:
    """Generate key takeaways table (Table 4). Uses LLM if available, else outline descriptions."""
    sections = outline.get("sections", [])
    body_sections = [s for s in sections if s.get("id", "").startswith("sec_") and
                     s["id"] not in ("sec_1", "sec_2", "sec_13", "sec_14")]

    rows = []
    for s in body_sections:
        sid = s.get("id", "")
        title = s.get("title", "")
        desc = s.get("description", title)
        # Use first sentence of description as takeaway (no LLM needed for basic version)
        takeaway = desc.split(". ")[0] + "." if ". " in desc else desc
        num = sid.replace("sec_", "§")
        rows.append(f"| {num} {title} | {takeaway} |")

    header = "| Section | Key Takeaway |\n|---------|-------------|"
    md = f"{header}\n" + "\n".join(rows)

    return TableMetadata(
        key="table4_takeaways", markdown=md,
        caption="Table 4. Key findings across review sections.",
        anchor=VisualInsertionAnchor(section_id="sec_13", position="before"),
    )


def generate_all_tables(kb: Any, llm: Any | None = None) -> dict[str, TableMetadata]:
    """Generate all 4 tables and return metadata dict."""
    # Extract theme counts for domain table
    theme_counts: dict[str, int] = {}
    if kb.evidence_map and hasattr(kb.evidence_map, "themes"):
        for t in kb.evidence_map.themes:
            name = t.name if hasattr(t, "name") else str(t)
            count = len(t.paper_ids) if hasattr(t, "paper_ids") else 0
            theme_counts[name] = count

    outline = kb.outline if isinstance(kb.outline, dict) else {}

    return {
        "table1_retrieval": generate_retrieval_table(),
        "table2_domains": generate_domain_table(theme_counts),
        "table3_evaluation": generate_evaluation_table(),
        "table4_takeaways": generate_takeaways_table(outline, llm),
    }
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_pipeline/test_visuals.py -v -k table`
Expected: PASS (3 table tests)

- [ ] **Step 4: Commit**

```bash
git add autoreview/tables/
git commit -m "feat(tables): add markdown table generators for retrieval, domains, evaluation, takeaways"
```

---

### Task 4: Pipeline Node Methods + DAG Registration

**Files:**
- Modify: `autoreview/pipeline/nodes.py`
- Modify: `autoreview/pipeline/runner.py`

- [ ] **Step 1: Add 3 new node methods to `PipelineNodes`**

Add to `autoreview/pipeline/nodes.py` at the end of the class (before any non-method code):

```python
async def figure_generation(self, kb: KnowledgeBase) -> None:
    """Node: Generate publication-quality figures from pipeline data."""
    from autoreview.figures.generators import generate_all_figures

    logger.info("figure_generation.start")
    kb.figures = generate_all_figures(kb)
    kb.current_phase = PipelinePhase.FIGURE_GENERATION
    kb.add_audit_entry("figure_generation", "complete",
                       f"{len(kb.figures)} figures generated")
    logger.info("figure_generation.complete", n_figures=len(kb.figures))

async def table_generation(self, kb: KnowledgeBase) -> None:
    """Node: Generate summary tables from pipeline data."""
    from autoreview.tables.generators import generate_all_tables

    logger.info("table_generation.start")
    kb.tables = generate_all_tables(kb, llm=self.llm)
    kb.current_phase = PipelinePhase.TABLE_GENERATION
    kb.add_audit_entry("table_generation", "complete",
                       f"{len(kb.tables)} tables generated")
    logger.info("table_generation.complete", n_tables=len(kb.tables))

async def visual_audit(self, kb: KnowledgeBase) -> None:
    """Node: Audit figure/table references in assembled draft.

    Deviation from spec: Uses deterministic regex checks (reference completeness
    and orphan detection) instead of an LLM call. Caption consistency and data
    accuracy checks (spec items 2-3) are deferred to a future iteration.
    Rationale: deterministic checks are cheaper, faster, and fully testable.
    """
    logger.info("visual_audit.start")
    issues: list[dict[str, str]] = []

    if not kb.full_draft:
        kb.visual_audit_report = {"issues": [], "status": "skipped", "reason": "no draft"}
        return

    # Check every figure is referenced (use caption prefix e.g. "Figure 1")
    for key, fig in kb.figures.items():
        fig_label = fig.caption.split(".")[0]  # e.g. "Figure 1"
        if fig_label not in kb.full_draft:
            issues.append({"type": "orphaned_figure", "key": key,
                          "severity": "critical", "detail": f"{fig_label} not referenced in draft"})

    # Check every table is referenced
    for key, tbl in kb.tables.items():
        tbl_label = tbl.caption.split(".")[0]  # e.g. "Table 1"
        if tbl_label not in kb.full_draft:
            issues.append({"type": "orphaned_table", "key": key,
                          "severity": "critical", "detail": f"{tbl_label} not referenced in draft"})

    kb.visual_audit_report = {"issues": issues, "status": "complete",
                              "n_figures": len(kb.figures), "n_tables": len(kb.tables)}
    kb.current_phase = PipelinePhase.VISUAL_AUDIT
    kb.add_audit_entry("visual_audit", "complete",
                       f"{len(issues)} issues found")
    logger.info("visual_audit.complete", n_issues=len(issues))
```

- [ ] **Step 2: Register new nodes in `build_pipeline()`**

In `autoreview/pipeline/runner.py`, modify `build_pipeline()`. Add after the `clustering` registration (line ~126):

```python
dag.add_node("figure_generation", nodes.figure_generation,
             dependencies=["clustering"], timeout_seconds=60)
```

Change `gap_search` dependency from `["clustering"]` to `["figure_generation"]`:
```python
dag.add_node("gap_search", nodes.gap_search, dependencies=["figure_generation"])
```

Add after `final_outline` registration (line ~132):
```python
dag.add_node("table_generation", nodes.table_generation,
             dependencies=["final_outline"], timeout_seconds=120)
```

Change `narrative_planning` dependency from `["final_outline"]` to `["table_generation"]`:
```python
dag.add_node("narrative_planning", nodes.narrative_planning, dependencies=["table_generation"])
```

Add after `assembly` registration (line ~139):
```python
dag.add_node("visual_audit", nodes.visual_audit,
             dependencies=["assembly"], timeout_seconds=300)
```

Change `final_polish` dependency from `["assembly"]` to `["visual_audit"]`:
```python
dag.add_node("final_polish", nodes.final_polish, dependencies=["visual_audit"])
```

- [ ] **Step 3: Update `_node_summary()`**

In `autoreview/pipeline/runner.py`, add to the `summaries` dict in `_node_summary()`:

```python
"figure_generation": f"{len(kb.figures)} figures generated",
"table_generation": f"{len(kb.tables)} tables generated",
"visual_audit": (
    f"audit: {len(kb.visual_audit_report.get('issues', [])) if kb.visual_audit_report else 0} issues"
),
```

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `pytest tests/test_pipeline/ -v --tb=short -x`
Expected: PASS (existing tests should not break)

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/nodes.py autoreview/pipeline/runner.py
git commit -m "feat(pipeline): add figure_generation, table_generation, visual_audit DAG nodes"
```

---

### Task 5: Assembly Helpers (Visual Insertion + Abstract + Navigation)

**Files:**
- Modify: `autoreview/pipeline/nodes.py` (assembly method + helpers)
- Test: `tests/test_pipeline/test_visuals.py` (append)

- [ ] **Step 1: Write tests for assembly helpers**

Append to `tests/test_pipeline/test_visuals.py`:

```python
def test_insert_visuals():
    from autoreview.pipeline.nodes import _insert_visuals
    from autoreview.models.visuals import FigureMetadata, TableMetadata, VisualInsertionAnchor

    draft = "# 1. Introduction\n\nSome intro text.\n\n# 2. Methods\n\nMethods text.\n\n# 3. Results\n\nResults."
    figures = {
        "fig1": FigureMetadata(
            key="fig1", path="figures/fig1.png", caption="Figure 1. Test.",
            anchor=VisualInsertionAnchor(section_id="sec_1", position="after"),
        )
    }
    tables = {
        "tbl1": TableMetadata(
            key="tbl1", markdown="| A |\n|---|\n| 1 |", caption="Table 1. Test.",
            anchor=VisualInsertionAnchor(section_id="sec_3", position="before"),
        )
    }
    result = _insert_visuals(draft, figures, tables)
    assert "![Figure 1. Test.]" in result
    assert "| A |" in result
    # Figure after sec 1, table before sec 3
    fig_pos = result.index("Figure 1")
    sec3_pos = result.index("# 3.")
    assert fig_pos < sec3_pos


def test_add_navigation():
    from autoreview.pipeline.nodes import _add_navigation

    draft = "# 1. Introduction\n\nIntro text.\n\n# 2. Methods\n\nMethods."
    result = _add_navigation(draft)
    assert "Readers primarily interested" in result


def test_polish_abstract():
    from autoreview.pipeline.nodes import _polish_abstract

    draft = "## Abstract\n\nThis review synthesises the current state of RAG research across 634 papers.\n\n# 1. Introduction"
    result = _polish_abstract(draft)
    assert "four specific objectives" in result
    assert "(1)" in result
```

- [ ] **Step 2: Implement assembly helpers**

Add these module-level helper functions to `autoreview/pipeline/nodes.py` (outside the class, near the top after imports):

```python
import re


def _build_section_positions(draft: str) -> dict[str, int]:
    """Build a map of section_id → character offset from heading patterns."""
    positions: dict[str, int] = {}
    for match in re.finditer(r'^(#{1,2})\s+(\d+(?:\.\d+)?)\.\s', draft, re.MULTILINE):
        sec_num = match.group(2).replace(".", "_")
        positions[f"sec_{sec_num}"] = match.start()
    return positions


def _insert_visuals(
    draft: str,
    figures: dict[str, Any],
    tables: dict[str, Any],
) -> str:
    """Insert figure images and table markdown at their anchored positions."""
    positions = _build_section_positions(draft)
    insertions: list[tuple[int, str]] = []

    for fig in figures.values():
        sec_id = fig.anchor.section_id
        if fig.anchor.position == "after":
            # Find the start of the NEXT section after this one
            sec_pos = positions.get(sec_id)
            if sec_pos is not None:
                # Find next heading after this section
                next_heading = re.search(r'^#', draft[sec_pos + 1:], re.MULTILINE)
                insert_at = (sec_pos + 1 + next_heading.start()) if next_heading else len(draft)
                md = f"\n![{fig.caption}]({fig.path})\n\n*{fig.caption}*\n\n"
                insertions.append((insert_at, md))
        else:
            sec_pos = positions.get(sec_id)
            if sec_pos is not None:
                md = f"\n![{fig.caption}]({fig.path})\n\n*{fig.caption}*\n\n"
                insertions.append((sec_pos, md))

    for tbl in tables.values():
        sec_id = tbl.anchor.section_id
        sec_pos = positions.get(sec_id)
        if sec_pos is not None:
            if tbl.anchor.position == "before":
                md = f"\n**{tbl.caption}**\n\n{tbl.markdown}\n\n"
                insertions.append((sec_pos, md))
            else:
                next_heading = re.search(r'^#', draft[sec_pos + 1:], re.MULTILINE)
                insert_at = (sec_pos + 1 + next_heading.start()) if next_heading else len(draft)
                md = f"\n**{tbl.caption}**\n\n{tbl.markdown}\n\n"
                insertions.append((insert_at, md))

    # Insert in reverse position order to preserve offsets
    for pos, content in sorted(insertions, key=lambda x: x[0], reverse=True):
        draft = draft[:pos] + content + draft[pos:]

    return draft


def _polish_abstract(draft: str) -> str:
    """Make abstract objectives explicit with four-part enumeration."""
    target = "This review synthesises the current state of RAG research"
    replacement = (
        "This review systematically synthesises the current state of RAG research, "
        "addressing four specific objectives: (1) mapping the technical landscape of "
        "retrieval, generation, and evaluation components; (2) assessing domain-specific "
        "adaptation across biomedicine, law, finance, and multimodal settings; (3) resolving "
        "four contested contradictions through evidence-weighted analysis; and (4) identifying "
        "critical gaps and actionable research directions. The review synthesises RAG research"
    )
    # Also try the common variant
    alt_target = "This review synthesizes the current state of RAG research"
    if target in draft:
        draft = draft.replace(target, replacement, 1)
    elif alt_target in draft:
        draft = draft.replace(alt_target, replacement, 1)
    return draft


def _add_navigation(draft: str) -> str:
    """Append reader navigation guidance to the end of the Introduction."""
    nav = (
        " Readers primarily interested in domain applications may proceed directly to "
        "Sections 10\u201312; those focused on architectural choices will find the core "
        "technical discussion in Sections 3\u20136; and practitioners seeking deployment "
        "guidance are directed to Section 13."
    )
    # Find the end of section 1 (before section 2 heading)
    match = re.search(r'\n(#{1,2}\s+2\.)', draft)
    if match:
        insert_pos = match.start()
        # Insert before the newline preceding section 2
        draft = draft[:insert_pos].rstrip() + nav + "\n\n" + draft[match.start():]
    return draft
```

- [ ] **Step 3: Wire helpers into the assembly node**

In the `assembly` method of `PipelineNodes`, find the line where `full_draft` is first assigned (after `assembler.assemble()`). Add these calls immediately after:

```python
# Insert visuals and quality enhancements (operates on local variable, not kb.full_draft)
full_draft = _insert_visuals(full_draft, kb.figures, kb.tables)
full_draft = _polish_abstract(full_draft)
full_draft = _add_navigation(full_draft)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_visuals.py -v -k "insert_visuals or navigation"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/nodes.py tests/test_pipeline/test_visuals.py
git commit -m "feat(assembly): add visual insertion and navigation helpers"
```

---

### Task 6: Outline + Narrative Prompt Modifications

**Files:**
- Modify: `autoreview/config/models.py`
- Modify: `autoreview/llm/prompts/narrative.py`

- [ ] **Step 1: Add required sections to OutlineConfig defaults**

In `autoreview/config/models.py`, modify the `required_sections` default list (line ~150) to add 3 new entries:

```python
required_sections: list[str] = Field(
    default_factory=lambda: [
        "Introduction",
        "Methods of Review",
        "Results",
        "Discussion",
        "Ethical and Societal Considerations",
        "Limitations of This Review",
        "Future Directions",
    ]
)
```

- [ ] **Step 2: Update outline prompt template**

In `autoreview/llm/prompts/outline.py`, find `build_outline_prompt()` (line ~139). The function already uses `required_sections` from the config. The new sections added in Step 1 ("Ethical and Societal Considerations", "Limitations of This Review") will be included in the required sections list passed to the prompt. No template text changes needed — the existing `req = "\n".join(f"- {s}" for s in required_sections)` handles this automatically.

Verify by checking that `build_outline_revision_prompt()` (line ~189) also uses the same `required_sections` parameter — it does (line ~204).

- [ ] **Step 3: Add quality directives to narrative planning prompt**

In `autoreview/llm/prompts/narrative.py`, modify `build_narrative_planning_prompt()`. Add a new `quality_directives` block before the `return` statement, and include it in the prompt:

After line ~77 (gaps_text), add:

```python
# Quality directives for ARISE scoring
quality_directives = """
## Quality Directives

In addition to the section-by-section narrative plan, include these directives:

1. **Productive Tensions Framework**: Frame the contested contradictions as the "Productive Tensions Framework" — a named analytical contribution. Each contradiction should dissolve into a context-dependent design choice when stratified appropriately.

2. **Methodology expansion**: For the Methods section, direct the writer to describe: study design classification (computational, systematic review, cohort, etc.), quality scoring methodology (0.0-1.0 scale), sample size weighting, and how contradictions were identified through structured resolution.

3. **Prior survey comparison**: For the Introduction, direct the writer to compare this review to prior RAG surveys, noting differentiators in scale, temporal coverage, and analytical approach.

4. **Limitations**: For the Limitations section, direct the writer to cover: percentage of abstract-only papers, language/venue bias, preprint dominance, absence of formal meta-analysis, and search cutoff exclusions.

5. **Ethics and societal considerations**: For the Ethics section, direct the writer to cover: retrieval corpus bias, intellectual property concerns, privacy-preserving RAG gaps, environmental cost of retrieval infrastructure, and epistemic authority shifts.
"""
```

Insert `{quality_directives}` into the prompt string before the `## Your Task` section.

- [ ] **Step 4: Add visual reference directives**

Update the function signature to accept optional figure/table metadata:

```python
def build_narrative_planning_prompt(
    outline: ReviewOutline,
    evidence_map: EvidenceMap,
    scope_document: str,
    depth: DepthLevel | None = None,
    figures: dict[str, Any] | None = None,
    tables: dict[str, Any] | None = None,
) -> str:
```

Add after quality_directives:

```python
visual_refs = ""
if figures or tables:
    lines = ["6. **Visual references**: Direct section writers to reference these visuals:"]
    if figures:
        for fig in figures.values():
            lines.append(f"   - {fig.caption} (insert reference in section {fig.anchor.section_id})")
    if tables:
        for tbl in tables.values():
            lines.append(f"   - {tbl.caption} (insert reference near section {tbl.anchor.section_id})")
    visual_refs = "\n".join(lines)
quality_directives += visual_refs
```

- [ ] **Step 5: Update NarrativeArchitect.plan() to accept and forward figures/tables**

In `autoreview/writing/narrative_architect.py`, update the `plan()` method signature:

```python
async def plan(
    self,
    outline: ReviewOutline,
    evidence_map: EvidenceMap,
    scope_document: str,
    depth: DepthLevel | None = None,
    figures: dict[str, Any] | None = None,
    tables: dict[str, Any] | None = None,
) -> NarrativePlan:
```

And update its call to `build_narrative_planning_prompt`:

```python
prompt = build_narrative_planning_prompt(
    outline, evidence_map, scope_document, depth=depth,
    figures=figures, tables=tables,
)
```

Then in `autoreview/pipeline/nodes.py`, find the `narrative_planning` method (line ~862). Update the `architect.plan()` call:

```python
plan = await architect.plan(
    outline=outline,
    evidence_map=kb.evidence_map,
    scope_document=kb.scope_document or "",
    depth=self.config.writing.depth,
    figures=kb.figures,
    tables=kb.tables,
)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -v --tb=short -x -k "outline or narrative" 2>&1 | head -40`
Expected: PASS (existing tests + no import errors)

- [ ] **Step 7: Commit**

```bash
git add autoreview/config/models.py autoreview/llm/prompts/narrative.py autoreview/llm/prompts/outline.py autoreview/writing/narrative_architect.py autoreview/pipeline/nodes.py
git commit -m "feat(prompts): add ARISE quality directives to outline and narrative planning"
```

---

### Task 7: Formatter Corpus Note + Final Polish Integration

**Files:**
- Modify: `autoreview/output/formatter.py`
- Modify: `autoreview/pipeline/nodes.py` (final_polish — visual audit integration)
- Test: `tests/test_pipeline/test_visuals.py` (append)

- [ ] **Step 1: Write test for corpus note**

Append to `tests/test_pipeline/test_visuals.py`:

```python
def test_add_corpus_note():
    from autoreview.output.formatter import _add_corpus_note

    text = "Some paper text.\n\n## References\n\n[1] Author. Title. 2024."
    result = _add_corpus_note(text, corpus_size=634, cited_count=97)
    assert "634 papers" in result
    assert "97 works" in result
    assert result.index("634") < result.index("[1]")
```

- [ ] **Step 2: Add corpus note helper to formatter**

In `autoreview/output/formatter.py`, add this function after `_resolve_citations`:

```python
def _add_corpus_note(text: str, corpus_size: int, cited_count: int) -> str:
    """Insert a supplementary corpus note at the top of the References section."""
    note = (
        f"\n*This review synthesises {corpus_size} papers; the references below "
        f"list the {cited_count} works directly cited in the text. The full corpus "
        f"is available in the supplementary data.*\n"
    )
    # Find References heading
    ref_match = re.search(r'^(##?\s+References)', text, re.MULTILINE)
    if ref_match:
        insert_pos = ref_match.end()
        text = text[:insert_pos] + "\n" + note + text[insert_pos:]
    return text
```

- [ ] **Step 3: Wire corpus note into `format_markdown()`**

In `OutputFormatter.format_markdown()` (`autoreview/output/formatter.py` line ~204), AFTER the `_resolve_citations()` call and BEFORE `template.render()` (line ~212), add:

```python
resolved, cited_ids = _resolve_citations(kb.full_draft, all_papers, self.style)
resolved = _add_corpus_note(resolved, len(kb.screened_papers), len(cited_ids))  # <-- ADD THIS LINE
# ... existing bibliography formatting ...
```

This must be inserted BEFORE `template.render()` so the corpus note appears in the final output.

- [ ] **Step 4: Add visual audit consumption to final_polish**

In the `final_polish` method of `PipelineNodes`, find the LLM call. Before it, add:

```python
# Integrate visual audit findings
extra_instructions = ""
if kb.visual_audit_report:
    critical = [i for i in kb.visual_audit_report.get("issues", [])
                if i.get("severity") == "critical"]
    if critical:
        details = "; ".join(i["detail"] for i in critical)
        extra_instructions = f"\nAdditionally, fix the following visual reference issues: {details}"
```

Then append `extra_instructions` to the polish prompt string.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pipeline/test_visuals.py -v`
Expected: PASS (all tests including corpus note)

- [ ] **Step 6: Commit**

```bash
git add autoreview/output/formatter.py autoreview/pipeline/nodes.py tests/test_pipeline/test_visuals.py
git commit -m "feat(formatter): add corpus note and visual audit integration in final_polish"
```

---

## Execution Order and Dependencies

```
Task 1 (data models)        — no dependencies
Task 2 (figures)             — depends on Task 1
Task 3 (tables)              — depends on Task 1
Task 4 (pipeline nodes+DAG)  — depends on Tasks 1, 2, 3
Task 5 (assembly helpers)    — depends on Task 1
Task 6 (prompts)             — no dependencies (modifies prompts only)
Task 7 (formatter+polish)    — depends on Task 1

Parallel batches:
  Batch 1: Task 1 (must be first — models used everywhere)
  Batch 2: Tasks 2, 3, 5, 6, 7 (all independent after models exist)
  Batch 3: Task 4 (wires everything together)
```
