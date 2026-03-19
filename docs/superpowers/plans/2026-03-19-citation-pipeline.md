# Citation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate paper drop-offs in the AutoReview pipeline by restructuring the DAG (two-pass outline) and adding a citation selection stage with configurable policy.

**Architecture:** Split the single outline node into draft (haiku, pre-enrichment) and final (sonnet, post-expansion). Add a deterministic `CitationSelector` between final outline and section writing that ranks papers, assigns tiers, and computes per-section budgets. New `CitationConfig` and `ReferenceMatchConfig` Pydantic models control policy. Writer prompts receive tiered paper lists instead of flat dumps.

**Tech Stack:** Python 3.11+, Pydantic v2, structlog, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-19-citation-pipeline-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `autoreview/config/citation.py` | `CitationConfig`, `ReferenceMatchConfig` Pydantic models |
| `autoreview/writing/citation_selector.py` | `CitationSelector` — scoring, tiering, budgeting algorithm; `CitationPlan` / `SectionCitationPlan` / `PaperCitation` models |
| `autoreview/analysis/reference_analyzer.py` | `ReferenceAnalyzer` — extract structural profile from reference PDF |
| `tests/test_config/test_citation_config.py` | Unit tests for CitationConfig validation |
| `tests/test_writing/test_citation_selector.py` | Unit tests for CitationSelector algorithm |
| `tests/test_analysis/test_reference_analyzer.py` | Unit tests for reference PDF analysis |

### Modified Files
| File | Change |
|------|--------|
| `autoreview/config/models.py` | Add `EXHAUSTIVE` to `DepthLevel`; add `citation` field to `WritingConfig`; extend `OutlineConfig` with `draft_model`/`final_model` |
| `autoreview/config/depth.py` | Add 3 citation fields to `DepthProfile`; add `EXHAUSTIVE` entry to `_DEPTH_PROFILES`; add `EXHAUSTIVE` to `_DEPTH_INSTRUCTIONS` |
| `autoreview/models/knowledge_base.py` | Add `DRAFT_OUTLINE`, `FINAL_OUTLINE`, `CITATION_SELECTION` to `PipelinePhase`; add `draft_outline`, `citation_plan` fields to `KnowledgeBase` |
| `autoreview/pipeline/runner.py` | Update `build_pipeline()` DAG: split `outline` into `draft_outline`→`final_outline`, insert `citation_selection`, reorder enrichment/expansion between them |
| `autoreview/pipeline/nodes.py` | Add `draft_outline()`, `final_outline()`, `citation_selection()` node methods; remove old `outline()` |
| `autoreview/writing/outliner.py` | Add `generate_draft()` method for lightweight outline |
| `autoreview/llm/prompts/outline.py` | Add `DRAFT_OUTLINE_SYSTEM_PROMPT` and `build_draft_outline_prompt()`; add `_OUTLINE_DEPTH_GUIDANCE[DepthLevel.EXHAUSTIVE]` |
| `autoreview/writing/section_writer.py` | Modify `_format_extractions()` to accept `CitationPlan` and format tiered lists; update `write_section()` signature |
| `autoreview/llm/prompts/writing.py` | Add citation tier instructions to `SECTION_WRITING_SYSTEM_PROMPT`; add `citation_budget` param to `build_section_writing_prompt()` |
| `autoreview/validation/citation_validator.py` | Add `validate_against_plan()` method that checks PRIMARY citation compliance and budget tolerance |
| `tests/test_config/test_depth.py` | Add tests for EXHAUSTIVE depth level and new citation fields |
| `tests/test_config/test_models.py` | Add tests for new WritingConfig.citation field |
| `tests/test_pipeline/test_runner.py` | Update DAG node count assertions; verify new node ordering |
| `tests/test_writing/test_outline_and_writing.py` | Update for draft/final outline split |
| `tests/test_validation/test_citation_validator.py` | Add tests for plan-based validation |

---

## Task Dependency Graph

```
Task 1 (CitationConfig models) ──┐
Task 2 (DepthProfile extensions) ─┤
Task 3 (KnowledgeBase + Phase)   ─┼──→ Task 5 (CitationSelector) ──→ Task 7 (DAG restructure) ──→ Task 9 (Integration test)
Task 4 (Reference Analyzer)      ─┘                                      ↑
                                                                   Task 6 (Outline split) ──┘
                                                                   Task 8 (Writer prompt changes)
```

Tasks 1-4 are independent. Task 5 depends on 1-3. Task 6 depends on 2-3. Task 7 depends on 5-6. Task 8 depends on 5. Task 9 depends on 7-8.

---

### Task 1: CitationConfig + ReferenceMatchConfig Models

**Files:**
- Create: `autoreview/config/citation.py`
- Modify: `autoreview/config/models.py` (add `citation` to `WritingConfig`, extend `OutlineConfig`)
- Test: `tests/test_config/test_citation_config.py`

- [ ] **Step 1: Write failing tests for CitationConfig**

```python
# tests/test_config/test_citation_config.py
from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_citation_config_defaults():
    from autoreview.config.citation import CitationConfig
    cfg = CitationConfig()
    assert cfg.citation_density == "standard"
    assert cfg.selection_strategy == "balanced"
    assert cfg.paper_tiers_enabled is True
    assert cfg.min_citations_per_section == 8
    assert cfg.w_evidence_strength == 0.30


def test_citation_config_weights_must_sum_to_one():
    from autoreview.config.citation import CitationConfig
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        CitationConfig(w_evidence_strength=0.5, w_recency=0.5, w_relevance_score=0.5)


def test_citation_config_weights_valid():
    from autoreview.config.citation import CitationConfig
    cfg = CitationConfig(
        w_evidence_strength=0.40, w_recency=0.20,
        w_relevance_score=0.20, w_uniqueness=0.10, w_source_diversity=0.10
    )
    assert cfg.w_evidence_strength == 0.40


def test_citation_config_extra_forbid():
    from autoreview.config.citation import CitationConfig
    with pytest.raises(ValidationError):
        CitationConfig(nonexistent_field="bad")


def test_reference_match_config_defaults():
    from autoreview.config.citation import ReferenceMatchConfig
    cfg = ReferenceMatchConfig()
    assert cfg.enabled is False
    assert cfg.reference_path is None
    assert cfg.word_count_tolerance == 0.15


def test_reference_match_config_extra_forbid():
    from autoreview.config.citation import ReferenceMatchConfig
    with pytest.raises(ValidationError):
        ReferenceMatchConfig(nonexistent_field="bad")


def test_writing_config_has_citation():
    from autoreview.config.models import WritingConfig
    wc = WritingConfig()
    assert hasattr(wc, 'citation')
    assert wc.citation.citation_density == "standard"


def test_outline_config_has_model_fields():
    from autoreview.config.models import OutlineConfig
    oc = OutlineConfig()
    assert oc.draft_model == "haiku"
    assert oc.final_model == "sonnet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && conda run -n autoreview python -m pytest tests/test_config/test_citation_config.py -v`
Expected: FAIL — `autoreview.config.citation` does not exist

- [ ] **Step 3: Implement CitationConfig and ReferenceMatchConfig**

```python
# autoreview/config/citation.py
"""Citation policy configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CitationConfig(BaseModel):
    """Controls citation policy — how aggressively and selectively to cite."""

    model_config = ConfigDict(extra="forbid")

    # Density
    citation_density: Literal["sparse", "standard", "dense", "exhaustive"] = "standard"
    min_citations_per_section: int = 8
    max_citations_per_section: int | None = None
    target_corpus_utilization: float = 0.25

    # Selection strategy
    selection_strategy: Literal[
        "evidence_weighted", "recency_weighted",
        "diversity_optimized", "impact_weighted", "balanced"
    ] = "balanced"

    # Paper tiering
    paper_tiers_enabled: bool = True
    tier_distribution: dict[str, float] = Field(
        default_factory=lambda: {"primary": 0.30, "supporting": 0.50, "contextual": 0.20}
    )
    seminal_paper_boost: float = 1.5

    # Priority scoring weights (must sum to 1.0)
    w_evidence_strength: float = 0.30
    w_recency: float = 0.20
    w_relevance_score: float = 0.25
    w_uniqueness: float = 0.15
    w_source_diversity: float = 0.10

    # Citation style
    cluster_citations: bool = True
    max_cluster_size: int = 5
    require_individual_discussion: Literal["primary_only", "all_tiers", "none"] = "primary_only"

    # Coverage constraints
    min_sources_per_theme: int = 2
    temporal_spread: bool = True
    contradiction_citation_mode: Literal["both_sides", "stronger_side", "all"] = "both_sides"
    gap_paper_priority: Literal["boost", "normal", "deprioritize"] = "boost"

    # Writer guidance
    writer_citation_hints: Literal[
        "none", "ranked_list", "annotated_tiers", "budget_with_tiers"
    ] = "budget_with_tiers"
    uncited_paper_warning: Literal["ignore", "warn", "force_include"] = "warn"

    @model_validator(mode="after")
    def validate_weights_sum(self) -> CitationConfig:
        total = (
            self.w_evidence_strength + self.w_recency + self.w_relevance_score
            + self.w_uniqueness + self.w_source_diversity
        )
        if abs(total - 1.0) >= 1e-6:
            msg = f"Citation scoring weights must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self


class ReferenceMatchConfig(BaseModel):
    """Auto-calibrate citation settings to match a reference paper's structural profile.

    Matches structural characteristics only (word count, citation density, section count,
    citation style). NEVER copies or is influenced by the reference paper's actual citations.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    reference_path: Path | None = None

    # Auto-extracted targets (populated by pre-pipeline analysis)
    target_word_count: int | None = None
    target_citation_count: int | None = None
    target_citation_density: float | None = None
    target_section_count: int | None = None
    inferred_citation_style: str | None = None

    # Tolerances
    word_count_tolerance: float = 0.15
    citation_count_tolerance: float = 0.20

    # What to match
    match_word_count: bool = True
    match_citation_density: bool = True
    match_section_structure: bool = True
    match_citation_style: bool = True
```

- [ ] **Step 4: Modify WritingConfig and OutlineConfig**

In `autoreview/config/models.py`:
- Add import: `from autoreview.config.citation import CitationConfig`
- Add to `WritingConfig`: `citation: CitationConfig = Field(default_factory=CitationConfig)`
- Add to `OutlineConfig`: `draft_model: str = "haiku"` and `final_model: str = "sonnet"`
- Add `EXHAUSTIVE = "exhaustive"` to `DepthLevel` enum

- [ ] **Step 4b: Add target_corpus_utilization to CritiqueConfig**

In `autoreview/config/models.py`, add to `CritiqueConfig`:
- `target_corpus_utilization: float = 0.25`

Add test to `tests/test_config/test_citation_config.py`:

```python
def test_critique_config_has_corpus_utilization():
    from autoreview.config.models import CritiqueConfig
    cc = CritiqueConfig()
    assert cc.target_corpus_utilization == 0.25
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && conda run -n autoreview python -m pytest tests/test_config/test_citation_config.py -v`
Expected: All PASS

- [ ] **Step 6: Run full config test suite for regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && conda run -n autoreview python -m pytest tests/test_config/ -v`
Expected: All PASS (no regressions in existing config tests)

- [ ] **Step 7: Commit**

```bash
git add autoreview/config/citation.py autoreview/config/models.py tests/test_config/test_citation_config.py
git commit -m "feat(config): add CitationConfig and ReferenceMatchConfig models

Add citation policy configuration with density, selection strategy,
paper tiering, priority weights, and coverage constraints.
Add reference match mode for benchmark calibration.
Extend WritingConfig and OutlineConfig with new fields."
```

---

### Task 2: DepthProfile Extensions

**Files:**
- Modify: `autoreview/config/depth.py`
- Modify: `tests/test_config/test_depth.py`

- [ ] **Step 1: Write failing tests for EXHAUSTIVE depth and citation fields**

```python
# Append to tests/test_config/test_depth.py
def test_exhaustive_depth_level():
    from autoreview.config.models import DepthLevel
    assert DepthLevel.EXHAUSTIVE == "exhaustive"


def test_exhaustive_depth_profile():
    from autoreview.config.depth import get_depth_profile
    from autoreview.config.models import DepthLevel
    profile = get_depth_profile(DepthLevel.EXHAUSTIVE)
    assert profile.total_word_budget == 40000
    assert profile.citation_density == "exhaustive"
    assert profile.target_citations_per_1k_words == 16.0
    assert profile.min_total_citations == 300


def test_medium_depth_citation_fields():
    from autoreview.config.depth import get_depth_profile
    from autoreview.config.models import DepthLevel
    profile = get_depth_profile(DepthLevel.MEDIUM)
    assert profile.citation_density == "standard"
    assert profile.target_citations_per_1k_words == 9.0
    assert profile.min_total_citations == 75


def test_exhaustive_depth_instructions():
    from autoreview.config.depth import get_depth_instructions
    from autoreview.config.models import DepthLevel
    result = get_depth_instructions(DepthLevel.EXHAUSTIVE, 40000)
    assert "40000" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_config/test_depth.py -v -k "exhaustive or citation_fields"`
Expected: FAIL

- [ ] **Step 3: Extend DepthProfile with citation fields and EXHAUSTIVE level**

**IMPORTANT: Add EXHAUSTIVE to DepthLevel enum AND add the EXHAUSTIVE entry to _DEPTH_PROFILES in the same edit. Existing test `test_all_profiles_have_body_dampening_of_one` iterates all DepthLevel values and will break if the enum has EXHAUSTIVE but `_DEPTH_PROFILES` doesn't.**

In `autoreview/config/depth.py`:
- Add 3 new fields to `DepthProfile` (after existing `max_tokens_override`):
  - `citation_density: str = "standard"`
  - `target_citations_per_1k_words: float = 9.0`
  - `min_total_citations: int = 50`
- Add citation values to all existing `_DEPTH_PROFILES` entries:
  - LOW: `citation_density="sparse"`, `target_citations_per_1k_words=6.0`, `min_total_citations=25`
  - MEDIUM: `citation_density="standard"`, `target_citations_per_1k_words=9.0`, `min_total_citations=75`
  - DEEP: `citation_density="dense"`, `target_citations_per_1k_words=12.0`, `min_total_citations=150`
- Add `DepthLevel.EXHAUSTIVE` entry to `_DEPTH_PROFILES`:
  - `base_word_multiplier=4.0`, `key_insights_range=(10, 15)`, `evidence_chain_detail="exhaustive"`,
  - `total_word_budget=40000`, `min_section_words=800`, same dampening as DEEP,
  - `max_tokens_override=16384`, `citation_density="exhaustive"`,
  - `target_citations_per_1k_words=16.0`, `min_total_citations=300`
- Add `DepthLevel.EXHAUSTIVE` entry to `_DEPTH_INSTRUCTIONS`

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_config/test_depth.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/depth.py autoreview/config/models.py tests/test_config/test_depth.py
git commit -m "feat(depth): add EXHAUSTIVE depth level and citation density fields

Extend DepthProfile with citation_density, target_citations_per_1k_words,
and min_total_citations. Add EXHAUSTIVE depth level (40K words, 300+ refs)."
```

---

### Task 3: KnowledgeBase + PipelinePhase Updates

**Files:**
- Modify: `autoreview/models/knowledge_base.py`
- Test: existing `tests/test_pipeline/test_runner.py` (regression check)

- [ ] **Step 1: Add new phases and fields to KnowledgeBase**

**Note: Keep `OUTLINE = 'outline'` in `PipelinePhase` for backward-compatibility with existing snapshots. Only add the three new phases; do not remove OUTLINE.**

In `autoreview/models/knowledge_base.py`:
- Add to `PipelinePhase` enum (after `GAP_SEARCH`, before `SECTION_WRITING`):
  - `DRAFT_OUTLINE = "draft_outline"`
  - `FINAL_OUTLINE = "final_outline"`
  - `CITATION_SELECTION = "citation_selection"`
- Add to `KnowledgeBase` class (after `outline` field):
  - `draft_outline: Any = None`
  - `citation_plan: Any = None`

- [ ] **Step 2: Verify existing tests still pass**

Run: `conda run -n autoreview python -m pytest tests/test_pipeline/ tests/test_models.py -v`
Expected: All PASS (new fields have defaults, backward compatible)

- [ ] **Step 3: Commit**

```bash
git add autoreview/models/knowledge_base.py
git commit -m "feat(kb): add draft_outline, citation_plan fields and new pipeline phases

Add DRAFT_OUTLINE, FINAL_OUTLINE, CITATION_SELECTION to PipelinePhase.
Add draft_outline and citation_plan fields to KnowledgeBase for crash recovery."
```

---

### Task 4: Reference Analyzer

**Files:**
- Create: `autoreview/analysis/reference_analyzer.py`
- Test: `tests/test_analysis/test_reference_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analysis/test_reference_analyzer.py
from __future__ import annotations

import pytest


def test_analyze_structural_profile_returns_expected_keys():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer
    # Create a minimal mock PDF text
    analyzer = ReferenceAnalyzer()
    profile = analyzer.analyze_text(
        "Abstract\nSome words here.\n\n1. Introduction\nMore text.\n\n"
        "References\n[1] Author A. Title. 2020.\n[2] Author B. Title. 2021.\n"
    )
    assert "word_count" in profile
    assert "citation_count" in profile
    assert "citation_density" in profile
    assert "section_count" in profile
    assert "citation_style" in profile


def test_analyze_text_counts_references():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer
    analyzer = ReferenceAnalyzer()
    text = "Some body text [1] and [2].\n\nReferences\n[1] Paper A.\n[2] Paper B.\n[3] Paper C.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_count"] == 3


def test_analyze_text_detects_numbered_style():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer
    analyzer = ReferenceAnalyzer()
    text = "Body [1] text [2].\n\nReferences\n[1] A.\n[2] B.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_style"] == "numbered"


def test_analyze_text_detects_author_year_style():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer
    analyzer = ReferenceAnalyzer()
    text = "Body (Smith et al., 2020) text (Jones, 2021).\n\nReferences\nSmith 2020.\nJones 2021.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_style"] == "author-year"


def test_analyze_text_empty_references_returns_none_count():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer
    analyzer = ReferenceAnalyzer()
    text = "Body text without references section."
    profile = analyzer.analyze_text(text)
    assert profile["citation_count"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_analysis/test_reference_analyzer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ReferenceAnalyzer**

```python
# autoreview/analysis/reference_analyzer.py
"""Extract structural profile from a reference paper for reference_match mode."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_NUMBERED_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_AUTHOR_CITE_RE = re.compile(r"\([A-Z][a-z]+ et al\.?,? \d{4}\)")
_HEADING_RE = re.compile(r"^#{1,3}\s+|\n\d+\.?\s+[A-Z]", re.MULTILINE)


class ReferenceAnalyzer:
    """Extract structural profile from reference paper text."""

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze raw text and return structural profile."""
        word_count = len(text.split())

        # Find references section
        ref_start = max(
            text.rfind("References\n"),
            text.rfind("REFERENCES\n"),
            text.rfind("Bibliography\n"),
        )

        citation_count: int | None = None
        if ref_start >= 0:
            ref_section = text[ref_start:]
            refs = _NUMBERED_CITE_RE.findall(ref_section)
            if refs:
                citation_count = max(int(r) for r in refs)
            else:
                # Count lines that look like reference entries
                lines = [l for l in ref_section.split("\n") if l.strip() and not l.startswith("#")]
                citation_count = max(len(lines) - 1, 0) or None

        # Detect citation style
        numbered_count = len(_NUMBERED_CITE_RE.findall(text[:ref_start] if ref_start > 0 else text))
        author_year_count = len(_AUTHOR_CITE_RE.findall(text))
        if numbered_count > author_year_count:
            citation_style = "numbered"
        elif author_year_count > 0:
            citation_style = "author-year"
        else:
            citation_style = "unknown"

        # Count sections (headings)
        section_count = len(_HEADING_RE.findall(text))

        # Citation density
        citation_density: float | None = None
        if citation_count and word_count > 0:
            citation_density = round(citation_count / (word_count / 1000), 1)

        return {
            "word_count": word_count,
            "citation_count": citation_count,
            "citation_density": citation_density,
            "section_count": section_count,
            "citation_style": citation_style,
        }

    def analyze_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Analyze a PDF file and return structural profile.

        Falls back to defaults if PDF cannot be parsed.
        """
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            text = "".join(page.get_text() for page in doc)
            return self.analyze_text(text)
        except Exception as e:
            logger.warning("reference_analyzer.pdf_failed", path=str(pdf_path), error=str(e))
            return {
                "word_count": None,
                "citation_count": None,
                "citation_density": None,
                "section_count": None,
                "citation_style": None,
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_analysis/test_reference_analyzer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/analysis/reference_analyzer.py tests/test_analysis/test_reference_analyzer.py
git commit -m "feat(analysis): add ReferenceAnalyzer for reference_match mode

Extracts word count, citation count/density, section count, and citation
style from reference paper text or PDF. Falls back gracefully on parse failure."
```

---

### Task 5: Citation Selector (Core Algorithm)

**Files:**
- Create: `autoreview/writing/citation_selector.py`
- Test: `tests/test_writing/test_citation_selector.py`

- [ ] **Step 1: Write failing tests for CitationSelector**

```python
# tests/test_writing/test_citation_selector.py
from __future__ import annotations

import pytest


def _make_extraction(paper_id, evidence_strength="moderate", year=2023, source="semantic_scholar"):
    """Helper to build a minimal PaperExtraction-like object."""
    from autoreview.extraction.models import Finding, PaperExtraction
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[Finding(claim=f"Claim from {paper_id}", evidence_strength=evidence_strength, paper_id=paper_id)],
        methods_summary="Method",
        limitations="None",
        relationships=[],
    )


def _make_evidence_map(themes=None, consensus_claims=None, evidence_chains=None):
    """Helper to build a minimal EvidenceMap."""
    from autoreview.analysis.evidence_map import (
        ConsensusClaim, EvidenceMap, IdentifiedGap, Theme,
    )
    from autoreview.analysis.evidence_chains import EvidenceChain
    return EvidenceMap(
        themes=themes or [],
        consensus_claims=consensus_claims or [],
        contradictions=[],
        gaps=[],
        evidence_chains=evidence_chains or [],
    )


def test_score_paper_basic():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector
    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    ext = _make_extraction("p1", evidence_strength="strong", year=2024)
    evidence_map = _make_evidence_map()
    score = selector.score_paper(
        paper_id="p1",
        extraction=ext,
        section_paper_ids=["p1", "p2"],
        evidence_map=evidence_map,
        paper_year=2024,
        paper_source="semantic_scholar",
        relevance_score=5,
        date_range=(2015, 2025),
    )
    assert 0.0 <= score <= 2.0  # max with seminal boost


def test_compute_section_budget():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector
    cfg = CitationConfig(min_citations_per_section=5)
    selector = CitationSelector(cfg)
    budget = selector.compute_section_budget(
        section_word_count=1500,
        target_per_1k=9.0,
        num_assigned=100,
    )
    assert budget == 14  # round(1500 * 9.0 / 1000) = 14, > min 5, < 100


def test_compute_section_budget_capped_by_assigned():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector
    cfg = CitationConfig(min_citations_per_section=8)
    selector = CitationSelector(cfg)
    budget = selector.compute_section_budget(
        section_word_count=1500, target_per_1k=9.0, num_assigned=5
    )
    assert budget == 5  # capped by num_assigned


def test_assign_tiers_basic():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector
    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    scored = [("p1", 0.9), ("p2", 0.8), ("p3", 0.7), ("p4", 0.6),
              ("p5", 0.5), ("p6", 0.4), ("p7", 0.3), ("p8", 0.2),
              ("p9", 0.1), ("p10", 0.05)]
    tiers = selector.assign_tiers(scored, budget=10)
    primary = [pid for pid, _, tier in tiers if tier == "primary"]
    supporting = [pid for pid, _, tier in tiers if tier == "supporting"]
    contextual = [pid for pid, _, tier in tiers if tier == "contextual"]
    assert len(primary) >= 1
    assert len(primary) + len(supporting) + len(contextual) == 10


def test_assign_tiers_small_section():
    """Sections with <3 papers should all be PRIMARY."""
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector
    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    scored = [("p1", 0.9), ("p2", 0.8)]
    tiers = selector.assign_tiers(scored, budget=2)
    assert all(tier == "primary" for _, _, tier in tiers)


def test_select_for_section_returns_citation_plan():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector, SectionCitationPlan
    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    extractions = {f"p{i}": _make_extraction(f"p{i}") for i in range(1, 21)}
    evidence_map = _make_evidence_map()
    plan = selector.select_for_section(
        section_id="sec_3",
        paper_ids=[f"p{i}" for i in range(1, 21)],
        extractions=extractions,
        evidence_map=evidence_map,
        section_word_count=1500,
        target_per_1k=9.0,
    )
    assert isinstance(plan, SectionCitationPlan)
    assert plan.citation_budget > 0
    assert len(plan.primary_papers) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_writing/test_citation_selector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CitationSelector**

Create `autoreview/writing/citation_selector.py` with:
- `PaperCitation`, `SectionCitationPlan`, `CitationPlan` Pydantic models
  - `SectionCitationPlan` must include `coverage_notes: list[str] = Field(default_factory=list)`
- `CitationSelector` class with methods:
  - `score_paper()` — apply weighted formula from spec
  - `compute_section_budget()` — word count * target/1K, clamped by min/max/assigned
  - `assign_tiers()` — ceil for primary, floor for contextual, remainder supporting; <3 all primary
  - `select_for_section()` — orchestrates scoring + budgeting + tiering for one section
  - `select_all()` — runs `select_for_section()` for all outline sections, returns `CitationPlan`

Evidence strength mapping: strong=1.0, moderate=0.7, weak=0.4, preliminary=0.2
Recency: linear (newest=1.0, oldest=0.0) within date_range
Relevance: score 5→1.0, 4→0.7, 3→0.4
Uniqueness: based on theme cluster membership from evidence_map.paper_theme_mapping
Source diversity: 1.0 - (same_source_count / total_assigned), clamped [0,1]
Seminal boost: consensus_claim with evidence_count >= 10 OR first in evidence_chain.paper_ids

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_writing/test_citation_selector.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/writing/citation_selector.py tests/test_writing/test_citation_selector.py
git commit -m "feat(writing): add CitationSelector for ranked paper tiering

Deterministic scoring algorithm with configurable weights for evidence
strength, recency, relevance, uniqueness, and source diversity.
Assigns papers to primary/supporting/contextual tiers with budgets."
```

---

### Task 6: Outline Split (Draft + Final)

**Files:**
- Modify: `autoreview/writing/outliner.py`
- Modify: `autoreview/llm/prompts/outline.py`
- Modify: `tests/test_writing/test_outline_and_writing.py`

- [ ] **Step 1: Write failing test for draft outline**

```python
# Append to tests/test_writing/test_outline_and_writing.py
import pytest


@pytest.mark.asyncio
async def test_generate_draft_returns_lightweight_outline():
    """Draft outline should produce section structure using the lightweight prompt."""
    from autoreview.llm.prompts.outline import ReviewOutline
    from autoreview.writing.outliner import OutlineGenerator

    mock_llm = MockWritingLLM()  # already defined in this file; returns _make_outline()
    gen = OutlineGenerator(mock_llm)
    result = await gen.generate_draft(
        evidence_map=_make_evidence_map(),
        scope_document="RAG review scope",
    )
    assert isinstance(result, ReviewOutline)
    assert len(result.sections) >= 1
    # generate_structured must have been called (draft outline uses structured generation)
    assert "generate_structured" in mock_llm.calls
```

- [ ] **Step 2: Add EXHAUSTIVE depth guidance to outline prompts**

In `autoreview/llm/prompts/outline.py`, add `DepthLevel.EXHAUSTIVE` entry to `_OUTLINE_DEPTH_GUIDANCE`.

- [ ] **Step 3: Add draft outline prompt and system prompt**

In `autoreview/llm/prompts/outline.py`:
- Add `DRAFT_OUTLINE_SYSTEM_PROMPT` — instructs LLM to produce lightweight topic buckets
- Add `build_draft_outline_prompt()` — scope + evidence summary only, no paper assignment instructions

- [ ] **Step 4: Add `generate_draft()` method to OutlineGenerator**

In `autoreview/writing/outliner.py`:
- Add `generate_draft()` that uses `DRAFT_OUTLINE_SYSTEM_PROMPT` and `build_draft_outline_prompt()`
- Existing `generate()` becomes the "final outline" path (no rename needed)

- [ ] **Step 5: Run tests**

Run: `conda run -n autoreview python -m pytest tests/test_writing/test_outline_and_writing.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add autoreview/writing/outliner.py autoreview/llm/prompts/outline.py tests/test_writing/test_outline_and_writing.py
git commit -m "feat(outline): add draft outline for two-pass pipeline

Draft outline produces lightweight topic buckets for enrichment targeting.
Existing generate() serves as final outline with full paper assignments."
```

---

### Task 7: DAG Restructure

**Files:**
- Modify: `autoreview/pipeline/runner.py`
- Modify: `autoreview/pipeline/nodes.py`
- Modify: `tests/test_pipeline/test_runner.py`

- [ ] **Step 1: Write failing test for new DAG structure**

```python
# Append to tests/test_pipeline/test_runner.py
def test_build_pipeline_has_17_nodes():
    """New pipeline should have 17 nodes including draft/final outline and citation_selection.

    Math: 15 existing - 1 removed (outline) + 3 added
    (draft_outline, final_outline, citation_selection) = 17.
    """
    # ... build pipeline with mock llm and default config ...
    dag, nodes = build_pipeline(mock_llm, config)
    assert len(dag.nodes) == 17
    assert "draft_outline" in dag.nodes
    assert "final_outline" in dag.nodes
    assert "citation_selection" in dag.nodes
    assert "outline" not in dag.nodes  # old node removed


def test_enrichment_depends_on_draft_outline():
    dag, _ = build_pipeline(mock_llm, config)
    assert "draft_outline" in dag.nodes["contextual_enrichment"].dependencies


def test_final_outline_depends_on_corpus_expansion():
    dag, _ = build_pipeline(mock_llm, config)
    assert "corpus_expansion" in dag.nodes["final_outline"].dependencies


def test_citation_selection_depends_on_final_outline():
    dag, _ = build_pipeline(mock_llm, config)
    assert "final_outline" in dag.nodes["citation_selection"].dependencies
```

- [ ] **Step 2: Implement new node methods in PipelineNodes**

In `autoreview/pipeline/nodes.py`:
- Add `draft_outline()` — calls `OutlineGenerator.generate_draft()`, stores in `kb.draft_outline`
- Add `final_outline()` — calls `OutlineGenerator.generate()` with full corpus, runs `EvidenceWeightedAllocator`, stores in `kb.outline`
- Add `citation_selection()` — instantiates `CitationSelector`, calls `select_all()`, stores in `kb.citation_plan`
- Remove or deprecate old `outline()` method

- [ ] **Step 2b: Update contextual_enrichment and corpus_expansion nodes for draft outline compatibility**

In `autoreview/pipeline/nodes.py`:
- Modify `contextual_enrichment` node: read from `kb.draft_outline` instead of `kb.outline`. Replace the `paper_ids >= 15` guard with an EvidenceMap theme count check (e.g., skip if `len(kb.evidence_map.themes) == 0`).
- Modify `corpus_expansion` node: ensure it adds newly found papers to `kb.extractions` (the full extraction set), **not** to `kb.draft_outline.sections`. The `final_outline` node reads `kb.extractions` to get the full corpus, so newly expanded papers are visible to the final outline without touching the draft outline.

- [ ] **Step 3: Restructure DAG in runner.py**

In `autoreview/pipeline/runner.py` `build_pipeline()`:
```python
# Replace old outline node with:
dag.add_node("draft_outline", nodes.draft_outline, dependencies=["gap_search"])
dag.add_node("contextual_enrichment", nodes.contextual_enrichment, dependencies=["draft_outline"])
dag.add_node("corpus_expansion", nodes.corpus_expansion, dependencies=["contextual_enrichment"])
dag.add_node("final_outline", nodes.final_outline, dependencies=["corpus_expansion"])
dag.add_node("narrative_planning", nodes.narrative_planning, dependencies=["final_outline"])
dag.add_node("citation_selection", nodes.citation_selection, dependencies=["narrative_planning"])
dag.add_node("section_writing", nodes.section_writing, dependencies=["citation_selection"])
# ... rest unchanged
```

Update `_node_summary()` with entries for `draft_outline`, `final_outline`, `citation_selection`.

- [ ] **Step 4: Run tests**

Run: `conda run -n autoreview python -m pytest tests/test_pipeline/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/runner.py autoreview/pipeline/nodes.py tests/test_pipeline/test_runner.py
git commit -m "feat(pipeline): restructure DAG with two-pass outline and citation selection

Split outline into draft_outline (pre-enrichment) and final_outline (post-expansion).
Insert citation_selection between narrative_planning and section_writing.
DAG now has 17 nodes (15 - 1 removed + 3 added). Final outline sees full corpus."
```

---

### Task 8: Writer Prompt Changes

**Files:**
- Modify: `autoreview/writing/section_writer.py`
- Modify: `autoreview/llm/prompts/writing.py`
- Modify: `tests/test_writing/test_section_writer_depth.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_writing/test_section_writer_depth.py
def test_format_extractions_with_citation_plan():
    """When a CitationPlan is provided, extractions should be tiered."""
    from autoreview.writing.section_writer import _format_extractions_tiered
    from autoreview.writing.citation_selector import SectionCitationPlan, PaperCitation
    # Build a plan with primary and supporting papers
    plan = SectionCitationPlan(
        section_id="sec_3",
        citation_budget=10,
        primary_papers=[PaperCitation(paper_id="p1", tier="primary", priority_score=0.9, citation_guidance="Discuss method")],
        supporting_papers=[PaperCitation(paper_id="p2", tier="supporting", priority_score=0.6, citation_guidance="")],
        contextual_papers=[],
        coverage_notes=[],
    )
    # ... provide extractions for p1, p2 ...
    result = _format_extractions_tiered(plan, extractions)
    assert "PRIMARY" in result
    assert "SUPPORTING" in result
    assert "[@p1]" in result
```

- [ ] **Step 2: Add `_format_extractions_tiered()` to section_writer.py**

New function that formats papers by tier with budget header. Called instead of `_format_extractions()` when `citation_plan` is available.

- [ ] **Step 3: Update `write_section()` to accept optional CitationPlan**

Add `citation_plan: SectionCitationPlan | None = None` parameter. When provided, use `_format_extractions_tiered()` instead of `_format_extractions()`.

- [ ] **Step 4: Add citation tier instructions to writing prompt**

In `autoreview/llm/prompts/writing.py`:
- Add `citation_tier_instructions` parameter to `build_section_writing_prompt()`
- Add citation budget and tier instructions block to the prompt template

- [ ] **Step 5: Update `write_all_sections()` to pass citation plans**

Accept optional `citation_plan: CitationPlan | None` parameter. Look up per-section plan and pass to `write_section()`.

- [ ] **Step 6: Run tests**

Run: `conda run -n autoreview python -m pytest tests/test_writing/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add autoreview/writing/section_writer.py autoreview/llm/prompts/writing.py tests/test_writing/
git commit -m "feat(writing): add tiered citation support to section writer

Section writer now accepts CitationPlan and formats papers by tier
(PRIMARY/SUPPORTING/CONTEXTUAL) with citation budgets. Writers receive
explicit guidance on which papers to discuss individually vs cluster."
```

---

### Task 9: Validation Updates + Integration Test

**Files:**
- Modify: `autoreview/validation/citation_validator.py`
- Modify: `tests/test_validation/test_citation_validator.py`
- Create: `tests/test_pipeline/test_citation_pipeline_integration.py`

- [ ] **Step 1: Write failing test for plan-based validation**

```python
# Append to tests/test_validation/test_citation_validator.py
def test_validate_against_plan_checks_primary_compliance():
    from autoreview.validation.citation_validator import CitationValidator
    from autoreview.writing.citation_selector import SectionCitationPlan, PaperCitation
    validator = CitationValidator()
    plan = SectionCitationPlan(
        section_id="sec_3", citation_budget=10,
        primary_papers=[
            PaperCitation(paper_id="p1", tier="primary", priority_score=0.9, citation_guidance=""),
            PaperCitation(paper_id="p2", tier="primary", priority_score=0.8, citation_guidance=""),
        ],
        supporting_papers=[], contextual_papers=[],
        coverage_notes=[],
    )
    # Text only cites p1, missing p2
    text = "Some findings [@p1] show results."
    report = validator.validate_against_plan(text, plan)
    assert "p2" in report.uncited_primary
```

- [ ] **Step 2: Implement `validate_against_plan()` method**

Add to `CitationValidator`:
- `validate_against_plan(text, plan) -> PlanValidationReport`
- Checks: all PRIMARY papers cited, total citations within budget ±30%, additive citations flagged

- [ ] **Step 3: Write integration test**

```python
# tests/test_pipeline/test_citation_pipeline_integration.py
"""Integration test: verify the citation pipeline eliminates drop-offs."""
from __future__ import annotations

import pytest


def test_new_dag_nodes_exist():
    """New nodes must be present and old outline node must be absent."""
    from unittest.mock import MagicMock
    from autoreview.pipeline.runner import build_pipeline
    from autoreview.config.models import ReviewConfig

    mock_llm = MagicMock()
    config = ReviewConfig()
    dag, _nodes = build_pipeline(mock_llm, config)

    assert "draft_outline" in dag.nodes
    assert "final_outline" in dag.nodes
    assert "citation_selection" in dag.nodes
    assert "outline" not in dag.nodes


def test_citation_plan_field_exists_on_knowledge_base():
    """KnowledgeBase must expose citation_plan field (not None after selection)."""
    from autoreview.models.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    # Field exists and defaults to None
    assert hasattr(kb, "citation_plan")
    assert kb.citation_plan is None

    # Field can be set
    kb.citation_plan = object()
    assert kb.citation_plan is not None
```

- [ ] **Step 4: Run all tests**

Run: `conda run -n autoreview python -m pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/validation/citation_validator.py tests/test_validation/ tests/test_pipeline/test_citation_pipeline_integration.py
git commit -m "feat(validation): add plan-based citation validation

CitationValidator now checks PRIMARY paper compliance and budget tolerance
against the CitationPlan. Passage search additions treated as additive."
```

---

### Task 10: Run-Local Skill Update

**Files:**
- Modify: `.claude/skills/run-local/skill.md`

- [ ] **Step 1: Update stage list**

Replace the 15-stage pipeline with the 17-stage pipeline from the spec. Update:
- Phase groupings (Phase 3 split into Part 1 and Part 2)
- Stage 8a (draft_outline) and 8b (final_outline)
- Stage 12 (citation_selection)
- Model assignments for new stages
- Validation gates for new stages

- [ ] **Step 2: Update search instructions**

Ensure all search stages reference Python SearchAggregator (already done in previous session, verify still correct).

- [ ] **Step 3: Update Common Mistakes table**

Add entry: "Skipping citation_selection | Citation selection is mandatory — it produces tiered paper lists that writers need"

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/run-local/skill.md
git commit -m "docs(skill): update run-local skill to 17-stage pipeline

Add draft_outline, final_outline, and citation_selection stages.
Update phase groupings and validation gates."
```
