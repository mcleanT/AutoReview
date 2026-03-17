# Depth Feature Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-level depth control (low/medium/deep) that adjusts review granularity and length via evidence-weighted word budget allocation.

**Architecture:** A `DepthProfile` lookup + `EvidenceWeightedAllocator` distribute word budgets across sections based on evidence density. Depth flows into three injection points: outline descriptions, narrative directive insight counts, and section writing prompts. The allocator runs as a post-processing step inside the existing outline node — no DAG changes.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest + pytest-asyncio, typer CLI

**Spec:** `docs/superpowers/specs/2026-03-16-depth-feature-design.md`

---

## Chunk 1: Core Data Model + Allocator

### Task 1: DepthLevel enum + WritingConfig field

**Files:**
- Modify: `autoreview/config/models.py:119-128`
- Test: `tests/test_config/test_depth.py` (new)

- [ ] **Step 1: Write failing test for DepthLevel enum and WritingConfig.depth field**

```python
# tests/test_config/test_depth.py
from autoreview.config.models import DepthLevel, WritingConfig


def test_depth_level_values():
    assert DepthLevel.LOW == "low"
    assert DepthLevel.MEDIUM == "medium"
    assert DepthLevel.DEEP == "deep"


def test_writing_config_default_depth():
    config = WritingConfig()
    assert config.depth == DepthLevel.MEDIUM


def test_writing_config_accepts_depth():
    config = WritingConfig(depth=DepthLevel.DEEP)
    assert config.depth == DepthLevel.DEEP


def test_writing_config_depth_serialization():
    """Verify DepthLevel survives Pydantic model_dump/model_validate roundtrip (extra='forbid')."""
    config = WritingConfig(depth=DepthLevel.LOW)
    dumped = config.model_dump()
    assert dumped["depth"] == "low"
    restored = WritingConfig.model_validate(dumped)
    assert restored.depth == DepthLevel.LOW
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: FAIL — `ImportError: cannot import name 'DepthLevel'`

- [ ] **Step 3: Implement DepthLevel enum and add depth field to WritingConfig**

In `autoreview/config/models.py`:
- Add `from enum import Enum` to imports (line 1 area — `Enum` is not currently imported).
- Add `DepthLevel` enum (with `str, Enum` base) after the existing imports (near line 7).
- Add `depth: DepthLevel = DepthLevel.MEDIUM` field to `WritingConfig` (after line 127).

```python
# Add to imports at top of file
from enum import Enum
```

```python
# Add near line 7, after existing imports
class DepthLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    DEEP = "deep"
```

```python
# Add to WritingConfig class body, after analysis_temperature field (line 127)
    depth: DepthLevel = DepthLevel.MEDIUM
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/models.py tests/test_config/test_depth.py
git commit -m "feat(config): add DepthLevel enum and depth field to WritingConfig"
```

---

### Task 2: DepthProfile dataclass + DEPTH_PROFILES lookup

**Files:**
- Create: `autoreview/config/depth.py`
- Test: `tests/test_config/test_depth.py` (append)

- [ ] **Step 1: Write failing tests for DepthProfile**

Append to `tests/test_config/test_depth.py`:

```python
from autoreview.config.depth import DepthProfile, get_depth_profile


def test_get_depth_profile_returns_profile():
    profile = get_depth_profile(DepthLevel.LOW)
    assert isinstance(profile, DepthProfile)


def test_low_profile_values():
    p = get_depth_profile(DepthLevel.LOW)
    assert p.base_word_multiplier == 0.6
    assert p.key_insights_range == (2, 3)
    assert p.evidence_chain_detail == "critical_only"
    assert p.total_word_budget == 4000
    assert p.min_section_words == 200
    assert p.max_tokens_override is None


def test_medium_profile_values():
    p = get_depth_profile(DepthLevel.MEDIUM)
    assert p.base_word_multiplier == 1.0
    assert p.key_insights_range == (3, 5)
    assert p.evidence_chain_detail == "standard"
    assert p.total_word_budget == 8000
    assert p.min_section_words == 400
    assert p.max_tokens_override is None


def test_deep_profile_values():
    p = get_depth_profile(DepthLevel.DEEP)
    assert p.base_word_multiplier == 2.5
    assert p.key_insights_range == (7, 10)
    assert p.evidence_chain_detail == "exhaustive"
    assert p.total_word_budget == 25000
    assert p.min_section_words == 600
    assert p.max_tokens_override == 16384


def test_deep_profile_has_higher_dampening_than_low():
    low = get_depth_profile(DepthLevel.LOW)
    deep = get_depth_profile(DepthLevel.DEEP)
    # Deep dampens less (higher multiplier = less dampening)
    assert deep.section_type_dampening["introduction"] > low.section_type_dampening["introduction"]
    assert deep.section_type_dampening["conclusion"] > low.section_type_dampening["conclusion"]


def test_all_profiles_have_body_dampening_of_one():
    for level in DepthLevel:
        p = get_depth_profile(level)
        assert p.section_type_dampening["body"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config/test_depth.py::test_get_depth_profile_returns_profile -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoreview.config.depth'`

- [ ] **Step 3: Implement DepthProfile and DEPTH_PROFILES lookup**

Create `autoreview/config/depth.py`:

```python
"""Depth profile configuration and evidence-weighted word budget allocation."""

from __future__ import annotations

from dataclasses import dataclass

from autoreview.config.models import DepthLevel


@dataclass(frozen=True)
class DepthProfile:
    """Deterministic parameter set for a given depth level."""

    base_word_multiplier: float
    key_insights_range: tuple[int, int]
    evidence_chain_detail: str  # "critical_only" | "standard" | "exhaustive"
    total_word_budget: int
    min_section_words: int
    section_type_dampening: dict[str, float]
    max_tokens_override: int | None


_DEPTH_PROFILES: dict[DepthLevel, DepthProfile] = {
    DepthLevel.LOW: DepthProfile(
        base_word_multiplier=0.6,
        key_insights_range=(2, 3),
        evidence_chain_detail="critical_only",
        total_word_budget=4000,
        min_section_words=200,
        section_type_dampening={
            "introduction": 0.7,
            "conclusion": 0.6,
            "methods": 0.7,
            "body": 1.0,
        },
        max_tokens_override=None,
    ),
    DepthLevel.MEDIUM: DepthProfile(
        base_word_multiplier=1.0,
        key_insights_range=(3, 5),
        evidence_chain_detail="standard",
        total_word_budget=8000,
        min_section_words=400,
        section_type_dampening={
            "introduction": 0.8,
            "conclusion": 0.7,
            "methods": 0.8,
            "body": 1.0,
        },
        max_tokens_override=None,
    ),
    DepthLevel.DEEP: DepthProfile(
        base_word_multiplier=2.5,
        key_insights_range=(7, 10),
        evidence_chain_detail="exhaustive",
        total_word_budget=25000,
        min_section_words=600,
        section_type_dampening={
            "introduction": 0.85,
            "conclusion": 0.75,
            "methods": 0.85,
            "body": 1.0,
        },
        max_tokens_override=16384,
    ),
}


def get_depth_profile(depth: DepthLevel) -> DepthProfile:
    """Look up the depth profile for a given depth level."""
    return _DEPTH_PROFILES[depth]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/depth.py tests/test_config/test_depth.py
git commit -m "feat(config): add DepthProfile dataclass and depth profile lookup"
```

---

### Task 3: Section-type classifier

**Files:**
- Modify: `autoreview/config/depth.py`
- Test: `tests/test_config/test_depth.py` (append)

- [ ] **Step 1: Write failing tests for classify_section_type**

Append to `tests/test_config/test_depth.py`:

```python
from autoreview.config.depth import classify_section_type


def test_classify_introduction():
    assert classify_section_type("Introduction") == "introduction"
    assert classify_section_type("Background and Introduction") == "introduction"
    assert classify_section_type("1. Background") == "introduction"


def test_classify_conclusion():
    assert classify_section_type("Conclusion") == "conclusion"
    assert classify_section_type("Concluding Remarks") == "conclusion"
    assert classify_section_type("Summary and Conclusions") == "conclusion"


def test_classify_methods():
    assert classify_section_type("Methods of Review") == "methods"
    assert classify_section_type("Search Strategy") == "methods"
    assert classify_section_type("Review Methodology") == "methods"


def test_classify_body_default():
    assert classify_section_type("Deep Learning Architectures") == "body"
    assert classify_section_type("Results and Discussion") == "body"
    assert classify_section_type("Future Directions") == "body"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config/test_depth.py::test_classify_introduction -v`
Expected: FAIL — `ImportError: cannot import name 'classify_section_type'`

- [ ] **Step 3: Implement classify_section_type**

Add to `autoreview/config/depth.py`:

```python
def classify_section_type(title: str) -> str:
    """Classify a section title into a type for dampening lookup.

    Uses fuzzy substring matching against known section categories.
    Returns one of: "introduction", "conclusion", "methods", "body".
    """
    lower = title.lower()
    if "introduction" in lower or "background" in lower:
        return "introduction"
    if "conclusion" in lower or "concluding" in lower:
        return "conclusion"
    if "method" in lower or "search strategy" in lower or "review methodology" in lower:
        return "methods"
    return "body"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/depth.py tests/test_config/test_depth.py
git commit -m "feat(config): add section-type classifier for depth dampening"
```

---

### Task 4: EvidenceWeightedAllocator

**Files:**
- Modify: `autoreview/config/depth.py`
- Test: `tests/test_config/test_depth.py` (append)

**Implementation note:** `EvidenceMap.evidence_chains` is `list[Any]` storing serialized dicts (not typed objects). The allocator must use `chain["paper_ids"]` dict access or deserialize via `EvidenceChain.model_validate()`. Use dict access for simplicity since we only need `paper_ids`.

- [ ] **Step 1: Write failing tests for EvidenceWeightedAllocator**

Append to `tests/test_config/test_depth.py`:

```python
from autoreview.config.depth import EvidenceWeightedAllocator
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.analysis.evidence_map import EvidenceMap


def _make_outline(sections: list[dict]) -> ReviewOutline:
    """Create a ReviewOutline from a list of section dicts."""
    outline_sections = []
    for s in sections:
        outline_sections.append(
            OutlineSection(
                id=s["id"],
                title=s["title"],
                description=s.get("description", "Test section"),
                paper_ids=s.get("paper_ids", []),
                theme_refs=s.get("theme_refs", []),
            )
        )
    return ReviewOutline(
        title="Test Review",
        sections=outline_sections,
    )


def _make_evidence_map(chains: list[dict] | None = None) -> EvidenceMap:
    """Create a minimal EvidenceMap with optional evidence chains."""
    em = EvidenceMap(
        themes=[],
        consensus_claims=[],
        contradictions=[],
        gaps=[],
    )
    em.evidence_chains = chains or []
    return em


def _make_extractions(paper_findings: dict[str, int]) -> dict:
    """Create a mock extractions dict. paper_findings maps paper_id -> number of findings."""
    from unittest.mock import MagicMock

    extractions = {}
    for pid, n_findings in paper_findings.items():
        mock = MagicMock()
        mock.key_findings = [f"finding_{i}" for i in range(n_findings)]
        extractions[pid] = mock
    return extractions


def test_allocator_basic_proportional():
    """Section with more evidence gets more words."""
    outline = _make_outline([
        {"id": "s1", "title": "Topic A", "paper_ids": ["p1"]},
        {"id": "s2", "title": "Topic B", "paper_ids": ["p1", "p2", "p3"]},
    ])
    extractions = _make_extractions({"p1": 2, "p2": 3, "p3": 1})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    # s2 has more evidence, should get more words
    assert outline.sections[1].estimated_word_count > outline.sections[0].estimated_word_count


def test_allocator_respects_floor():
    """No section gets fewer words than min_section_words."""
    outline = _make_outline([
        {"id": "s1", "title": "Tiny Topic", "paper_ids": ["p1"]},
        {"id": "s2", "title": "Big Topic", "paper_ids": ["p1", "p2", "p3", "p4", "p5"]},
    ])
    extractions = _make_extractions({"p1": 1, "p2": 5, "p3": 5, "p4": 5, "p5": 5})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    assert outline.sections[0].estimated_word_count >= profile.min_section_words


def test_allocator_dampens_introduction():
    """Introduction gets fewer words than an equivalent body section."""
    outline = _make_outline([
        {"id": "s1", "title": "Introduction", "paper_ids": ["p1", "p2"]},
        {"id": "s2", "title": "Core Topic", "paper_ids": ["p1", "p2"]},
    ])
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    # Same evidence but intro is dampened
    assert outline.sections[0].estimated_word_count < outline.sections[1].estimated_word_count


def test_allocator_zero_evidence_section_gets_fixed_allocation():
    """Sections with no papers get depth-scaled fixed allocation."""
    outline = _make_outline([
        {"id": "s1", "title": "Core Topic", "paper_ids": ["p1", "p2"]},
        {"id": "s2", "title": "Future Directions", "paper_ids": []},
    ])
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()

    for level in DepthLevel:
        profile = get_depth_profile(level)
        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline, evidence_map, extractions)

        expected_fixed = int(profile.base_word_multiplier * 500)
        assert outline.sections[1].estimated_word_count == expected_fixed


def test_allocator_evidence_chains_increase_density():
    """Sections touched by evidence chains get more words."""
    outline = _make_outline([
        {"id": "s1", "title": "Topic A", "paper_ids": ["p1"]},
        {"id": "s2", "title": "Topic B", "paper_ids": ["p2"]},
    ])
    extractions = _make_extractions({"p1": 2, "p2": 2})
    # Chain touches p2 but not p1
    chains = [{"paper_ids": ["p2", "p3"], "chain_id": "c1"}]
    evidence_map = _make_evidence_map(chains=chains)
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    # s2 has a chain touching it, should get more
    assert outline.sections[1].estimated_word_count > outline.sections[0].estimated_word_count


def test_allocator_depth_scales_output():
    """Deep mode produces higher word counts than low mode."""
    outline_low = _make_outline([
        {"id": "s1", "title": "Topic", "paper_ids": ["p1", "p2"]},
    ])
    outline_deep = _make_outline([
        {"id": "s1", "title": "Topic", "paper_ids": ["p1", "p2"]},
    ])
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()

    EvidenceWeightedAllocator(get_depth_profile(DepthLevel.LOW)).allocate(
        outline_low, evidence_map, extractions
    )
    EvidenceWeightedAllocator(get_depth_profile(DepthLevel.DEEP)).allocate(
        outline_deep, evidence_map, extractions
    )

    assert outline_deep.sections[0].estimated_word_count > outline_low.sections[0].estimated_word_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config/test_depth.py::test_allocator_basic_proportional -v`
Expected: FAIL — `ImportError: cannot import name 'EvidenceWeightedAllocator'`

- [ ] **Step 3: Implement EvidenceWeightedAllocator**

Add to `autoreview/config/depth.py`:

```python
from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoreview.analysis.evidence_map import EvidenceMap
    from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline

logger = structlog.get_logger()


class EvidenceWeightedAllocator:
    """Distributes word budget across sections based on evidence density."""

    def __init__(self, profile: DepthProfile) -> None:
        self.profile = profile

    def allocate(
        self,
        outline: ReviewOutline,
        evidence_map: EvidenceMap,
        extractions: dict,
    ) -> None:
        """Mutate outline sections' estimated_word_count in place."""
        sections = outline.sections
        if not sections:
            return

        # Separate zero-evidence sections
        evidence_sections: list[tuple[int, OutlineSection, float]] = []
        fixed_total = 0

        for i, section in enumerate(sections):
            density = self._compute_density(section, extractions, evidence_map)
            if density == 0.0:
                fixed_alloc = int(self.profile.base_word_multiplier * 500)
                section.estimated_word_count = fixed_alloc
                fixed_total += fixed_alloc
            else:
                evidence_sections.append((i, section, density))

        if not evidence_sections:
            return

        remaining_budget = self.profile.total_word_budget - fixed_total
        total_density = sum(d for _, _, d in evidence_sections)

        # Proportional allocation
        for idx, section, density in evidence_sections:
            share = density / total_density
            raw = share * remaining_budget
            dampening = self.profile.section_type_dampening.get(
                classify_section_type(section.title), 1.0
            )
            adjusted = raw * dampening
            section.estimated_word_count = max(
                int(adjusted), self.profile.min_section_words
            )

        # Surplus/deficit redistribution to body sections
        current_total = sum(s.estimated_word_count for s in sections)
        body_sections = [
            (i, s, d)
            for i, s, d in evidence_sections
            if classify_section_type(s.title) == "body"
        ]

        if body_sections and current_total != self.profile.total_word_budget:
            delta = self.profile.total_word_budget - current_total
            body_density_total = sum(d for _, _, d in body_sections)
            if body_density_total > 0:
                for _, section, density in body_sections:
                    adjustment = int(delta * (density / body_density_total))
                    new_count = section.estimated_word_count + adjustment
                    section.estimated_word_count = max(
                        new_count, self.profile.min_section_words
                    )

        logger.info(
            "depth_allocation_complete",
            depth=self.profile.evidence_chain_detail,
            budget=self.profile.total_word_budget,
            actual=sum(s.estimated_word_count for s in sections),
            sections={s.id: s.estimated_word_count for s in sections},
        )

    def _compute_density(
        self,
        section: OutlineSection,
        extractions: dict,
        evidence_map: EvidenceMap,
    ) -> float:
        n_papers = len(section.paper_ids)

        n_findings = sum(
            len(extractions[pid].key_findings)
            for pid in section.paper_ids
            if pid in extractions
        )

        section_pids = set(section.paper_ids)
        n_chains = 0
        for chain in evidence_map.evidence_chains:
            # evidence_chains stores serialized dicts, use dict access
            chain_pids = chain.get("paper_ids", []) if isinstance(chain, dict) else getattr(chain, "paper_ids", [])
            if set(chain_pids) & section_pids:
                n_chains += 1

        return float(n_papers + n_findings + n_chains)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/depth.py tests/test_config/test_depth.py
git commit -m "feat(config): add EvidenceWeightedAllocator for depth-aware word budgets"
```

---

## Chunk 2: Prompt Injection Points

### Task 5: Depth instructions helper

**Files:**
- Modify: `autoreview/config/depth.py`
- Test: `tests/test_config/test_depth.py` (append)

- [ ] **Step 1: Write failing tests for get_depth_instructions**

Append to `tests/test_config/test_depth.py`:

```python
from autoreview.config.depth import get_depth_instructions


def test_depth_instructions_low():
    text = get_depth_instructions(DepthLevel.LOW, 300)
    assert "critical findings" in text.lower()
    assert "300" in text


def test_depth_instructions_medium():
    text = get_depth_instructions(DepthLevel.MEDIUM, 800)
    assert "thoroughness" in text.lower() or "readability" in text.lower()
    assert "800" in text


def test_depth_instructions_deep():
    text = get_depth_instructions(DepthLevel.DEEP, 2000)
    assert "exhaustive" in text.lower()
    assert "2000" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config/test_depth.py::test_depth_instructions_low -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement get_depth_instructions**

Add to `autoreview/config/depth.py`:

```python
_DEPTH_INSTRUCTIONS: dict[DepthLevel, str] = {
    DepthLevel.LOW: (
        "Distill to the most critical findings. One paragraph per key claim. "
        "Prioritize clarity and impact over completeness. "
        "Target approximately {target_word_count} words."
    ),
    DepthLevel.MEDIUM: (
        "Balance thoroughness with readability. Cover major findings with "
        "supporting context. Standard academic review depth. "
        "Target approximately {target_word_count} words."
    ),
    DepthLevel.DEEP: (
        "Exhaustively trace evidence chains. Include methodological comparisons, "
        "conflicting results with resolution analysis, temporal evolution of findings, "
        "and secondary implications. Prioritize completeness over brevity. "
        "Target approximately {target_word_count} words."
    ),
}


def get_depth_instructions(depth: DepthLevel, target_word_count: int) -> str:
    """Return depth-specific prose instructions for the section writer."""
    return _DEPTH_INSTRUCTIONS[depth].format(target_word_count=target_word_count)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config/test_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/depth.py tests/test_config/test_depth.py
git commit -m "feat(config): add depth instruction text generator"
```

---

### Task 6: Inject depth into section writing prompt

**Files:**
- Modify: `autoreview/llm/prompts/writing.py:53-90`
- Test: `tests/test_prompts/test_writing_depth.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_prompts/test_writing_depth.py
from autoreview.llm.prompts.writing import build_section_writing_prompt


def test_section_writing_prompt_includes_depth_instructions():
    prompt = build_section_writing_prompt(
        section_id="s1",
        section_title="Deep Learning",
        section_description="Overview of DL methods",
        outline_context="Full outline here",
        relevant_extractions="Extractions here",
        target_word_count=1200,
        depth_instructions="Exhaustively trace evidence chains. Target approximately 1200 words.",
    )
    assert "1200" in prompt
    assert "Exhaustively trace" in prompt


def test_section_writing_prompt_without_depth_is_unchanged():
    """Backwards compatibility: no depth args produces same prompt as before."""
    prompt = build_section_writing_prompt(
        section_id="s1",
        section_title="Deep Learning",
        section_description="Overview",
        outline_context="Outline",
        relevant_extractions="Extractions",
    )
    # Should not contain depth-specific blocks
    assert "DEPTH AND LENGTH GUIDANCE" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompts/test_writing_depth.py -v`
Expected: FAIL — `TypeError: build_section_writing_prompt() got an unexpected keyword argument 'target_word_count'`

- [ ] **Step 3: Add target_word_count and depth_instructions to build_section_writing_prompt**

In `autoreview/llm/prompts/writing.py`, modify `build_section_writing_prompt` (lines 53-62) to add two new optional parameters:

```python
def build_section_writing_prompt(
    section_id: str,
    section_title: str,
    section_description: str,
    outline_context: str,
    relevant_extractions: str,
    synthesis_directives: str = "",
    adjacent_text: str = "",
    narrative_guidance: str = "",
    contextual_enrichment: str = "",
    target_word_count: int | None = None,      # NEW
    depth_instructions: str = "",               # NEW
) -> str:
```

The function body uses a single f-string return (NOT a parts list). Add a `depth_block` variable alongside the existing `narrative_block` and `enrichment_block` (around line 64-70), then insert it into the f-string:

```python
    narrative_block = f"\n{narrative_guidance}\n" if narrative_guidance else ""
    enrichment_block = ""
    if contextual_enrichment:
        enrichment_block = (
            f"\n## Contextual Background Material (supplementary — use for framing, not primary evidence)\n"
            f"{contextual_enrichment}\n"
        )
    # NEW: depth block
    depth_block = ""
    if depth_instructions:
        depth_block = f"\n## DEPTH AND LENGTH GUIDANCE\n\n{depth_instructions}\n"
```

Then insert `{depth_block}` into the f-string return, after `{enrichment_block}` and before the final "Write this section..." line (line 88-89):

```python
    return f"""\
## Section to Write
...
{narrative_block}{enrichment_block}{depth_block}\
Write this section with proper synthesis, citation markers [@paper_id], and smooth transitions.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompts/test_writing_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/llm/prompts/writing.py tests/test_prompts/test_writing_depth.py
git commit -m "feat(prompts): add depth instructions to section writing prompt"
```

---

### Task 7: Inject depth into outline prompt

**Files:**
- Modify: `autoreview/llm/prompts/outline.py:71-96`
- Test: `tests/test_prompts/test_outline_depth.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_prompts/test_outline_depth.py
from autoreview.config.models import DepthLevel
from autoreview.llm.prompts.outline import build_outline_prompt


def test_outline_prompt_low_depth():
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
        depth=DepthLevel.LOW,
    )
    assert "critical" in prompt.lower() or "key findings" in prompt.lower()


def test_outline_prompt_deep_depth():
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
        depth=DepthLevel.DEEP,
    )
    assert "exhaustive" in prompt.lower() or "methodological" in prompt.lower()


def test_outline_prompt_no_depth_backwards_compatible():
    """Without depth arg, prompt is unchanged from current behavior."""
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
    )
    assert "DEPTH GUIDANCE" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompts/test_outline_depth.py -v`
Expected: FAIL — `TypeError: build_outline_prompt() got an unexpected keyword argument 'depth'`

- [ ] **Step 3: Add depth parameter to build_outline_prompt**

In `autoreview/llm/prompts/outline.py`, add `from autoreview.config.models import DepthLevel` to imports.

Modify `build_outline_prompt` (line 71) to accept an optional `depth` parameter:

```python
def build_outline_prompt(
    scope_document: str,
    evidence_summary: str,
    required_sections: list[str],
    depth: DepthLevel | None = None,   # NEW
) -> str:
```

Add a module-level constant for depth guidance (before the function):

```python
_OUTLINE_DEPTH_GUIDANCE: dict[DepthLevel, str] = {
    DepthLevel.LOW: (
        "\n## DEPTH GUIDANCE\n\n"
        "This is a concise review. Section descriptions should emphasize key findings "
        "and critical takeaways only. Focus on the most impactful points.\n"
    ),
    DepthLevel.MEDIUM: "",  # No extra guidance for medium (current behavior)
    DepthLevel.DEEP: (
        "\n## DEPTH GUIDANCE\n\n"
        "This is an exhaustive, in-depth review. Section descriptions should request "
        "comprehensive coverage: trace methodological evolution, compare conflicting "
        "findings, discuss edge cases, and explore secondary implications.\n"
    ),
}
```

The function body uses a single f-string return (NOT a parts list). Add a `depth_block` variable before the return, then insert it into the f-string:

```python
    req = "\n".join(f"- {s}" for s in required_sections)
    # NEW: depth guidance block
    depth_block = ""
    if depth and depth in _OUTLINE_DEPTH_GUIDANCE:
        depth_block = _OUTLINE_DEPTH_GUIDANCE[depth]

    return f"""\
## Review Scope
{scope_document}

## Evidence Summary
{evidence_summary}

## Required Sections
{req}
{depth_block}
Generate a hierarchical outline for this review paper. Each section needs:
...
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompts/test_outline_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/llm/prompts/outline.py tests/test_prompts/test_outline_depth.py
git commit -m "feat(prompts): add depth guidance to outline generation prompt"
```

---

### Task 8: Inject depth into narrative planning prompt

**Files:**
- Modify: `autoreview/llm/prompts/narrative.py:24-28,103`
- Test: `tests/test_prompts/test_narrative_depth.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_prompts/test_narrative_depth.py
from autoreview.config.models import DepthLevel
from autoreview.llm.prompts.narrative import build_narrative_planning_prompt
from autoreview.llm.prompts.outline import ReviewOutline, OutlineSection
from autoreview.analysis.evidence_map import EvidenceMap


def _make_test_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test",
        sections=[
            OutlineSection(
                id="s1", title="Topic", description="Desc",
                paper_ids=["p1"], theme_refs=[],
            ),
        ],
    )


def _make_test_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        themes=[], consensus_claims=[], contradictions=[], gaps=[],
    )


def test_narrative_prompt_low_depth_key_insights_range():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
        depth=DepthLevel.LOW,
    )
    assert "2–3" in prompt or "2-3" in prompt


def test_narrative_prompt_deep_depth_key_insights_range():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
        depth=DepthLevel.DEEP,
    )
    assert "7–10" in prompt or "7-10" in prompt


def test_narrative_prompt_no_depth_keeps_default():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
    )
    # Default behavior: "3–5" (current hardcoded value)
    assert "3–5" in prompt or "3-5" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompts/test_narrative_depth.py -v`
Expected: FAIL — `TypeError: build_narrative_planning_prompt() got an unexpected keyword argument 'depth'`

- [ ] **Step 3: Add depth parameter to build_narrative_planning_prompt**

In `autoreview/llm/prompts/narrative.py`, modify `build_narrative_planning_prompt` (lines 24-28) to accept `depth`:

```python
def build_narrative_planning_prompt(
    outline: ReviewOutline,
    evidence_map: EvidenceMap,
    scope_document: str,
    depth: DepthLevel | None = None,   # NEW
) -> str:
```

Replace the hardcoded "3–5" on line 103 with a dynamic range based on depth:

```python
    # Replace the hardcoded key_insights line
    from autoreview.config.depth import get_depth_profile
    if depth:
        lo, hi = get_depth_profile(depth).key_insights_range
    else:
        lo, hi = 3, 5
    # Use the range in the prompt text
    # Replace: "3–5 analytical insights"
    # With: f"{lo}–{hi} analytical insights"
```

Add the import at the top: `from autoreview.config.models import DepthLevel`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompts/test_narrative_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/llm/prompts/narrative.py tests/test_prompts/test_narrative_depth.py
git commit -m "feat(prompts): add depth-aware key insights range to narrative prompt"
```

---

## Chunk 3: Pipeline Wiring

### Task 8b: Wire depth through OutlineGenerator

**Files:**
- Modify: `autoreview/writing/outliner.py:58-108`

The outline prompt's new `depth` parameter must be forwarded through `OutlineGenerator.generate()`, which is the actual caller of `build_outline_prompt()`. Without this, depth never reaches the outline prompt at runtime.

- [ ] **Step 1: Add depth parameter to OutlineGenerator.generate**

In `autoreview/writing/outliner.py`, add `from autoreview.config.models import DepthLevel` to imports (line 1 area).

Modify `generate` method signature (lines 58-65) to accept `depth`:

```python
    async def generate(
        self,
        evidence_map: EvidenceMap,
        scope_document: str,
        required_sections: list[str] | None = None,
        previous_outline: ReviewOutline | None = None,
        critique_report: CritiqueReport | None = None,
        depth: DepthLevel | None = None,   # NEW
    ) -> ReviewOutline:
```

Pass `depth` to `build_outline_prompt()` call (line 108):

```python
        # Fresh generation mode (line 107-108)
        else:
            prompt = build_outline_prompt(
                scope_document, evidence_summary, required_sections, depth=depth
            )
```

Also pass `depth` to `build_outline_revision_prompt()` call (lines 91-98) if it accepts depth, OR leave the revision prompt as-is (revision already has the outline context from the first pass, so depth guidance is less critical for revisions).

- [ ] **Step 2: Verify existing tests still pass**

Run: `python -m pytest tests/test_writing/ -v --tb=short`
Expected: PASS (depth defaults to None, preserving current behavior)

- [ ] **Step 3: Commit**

```bash
git add autoreview/writing/outliner.py
git commit -m "feat(writing): wire depth parameter through OutlineGenerator"
```

---

### Task 8c: Wire depth through write_all_sections

**Files:**
- Modify: `autoreview/writing/section_writer.py:371-408`

The pipeline calls `write_all_sections()`, not `write_section()` directly. The `depth` parameter added to `write_section()` in Task 9 is unreachable unless `write_all_sections()` also accepts and forwards it.

- [ ] **Step 1: Add depth parameter to write_all_sections**

In `autoreview/writing/section_writer.py`, modify `write_all_sections` signature (lines 371-378):

```python
    async def write_all_sections(
        self,
        outline: ReviewOutline,
        extractions: dict[str, PaperExtraction],
        evidence_map: EvidenceMap,
        narrative_plan: NarrativePlan | None = None,
        contextual_enrichment: dict[str, SectionEnrichment] | None = None,
        depth: DepthLevel | None = None,   # NEW
    ) -> dict[str, SectionDraft]:
```

Forward `depth` to the `self.write_section()` call (lines 395-404):

```python
            draft = await self.write_section(
                section=section,
                outline=outline,
                extractions=extractions,
                evidence_map=evidence_map,
                preceding_text=preceding,
                following_text=following,
                directive=directive,
                enrichment=enrichment,
                depth=depth,        # NEW
            )
```

Add import: `from autoreview.config.models import DepthLevel` at top of file.

- [ ] **Step 2: Verify existing tests still pass**

Run: `python -m pytest tests/test_writing/ -v --tb=short`
Expected: PASS (depth defaults to None)

- [ ] **Step 3: Commit**

```bash
git add autoreview/writing/section_writer.py
git commit -m "feat(writing): wire depth through write_all_sections"
```

---

### Task 9: Wire depth into SectionWriter

**Files:**
- Modify: `autoreview/writing/section_writer.py:233-292`
- Test: `tests/test_writing/test_section_writer_depth.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_writing/test_section_writer_depth.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from autoreview.config.models import DepthLevel
from autoreview.writing.section_writer import SectionWriter
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.llm.provider import LLMResponse


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Generated section text with [citations].",
            input_tokens=100,
            output_tokens=200,
        )
    )
    return llm


@pytest.fixture
def section():
    return OutlineSection(
        id="s1",
        title="Deep Learning",
        description="Overview",
        paper_ids=["p1"],
        theme_refs=[],
        estimated_word_count=1200,
    )


@pytest.fixture
def outline(section):
    return ReviewOutline(title="Test", sections=[section])


@pytest.fixture
def evidence_map():
    return EvidenceMap(
        themes=[], consensus_claims=[], contradictions=[], gaps=[],
    )


@pytest.mark.asyncio
async def test_section_writer_passes_depth_to_prompt(mock_llm, section, outline, evidence_map):
    writer = SectionWriter(mock_llm)
    await writer.write_section(
        section=section,
        outline=outline,
        extractions={},
        evidence_map=evidence_map,
        depth=DepthLevel.DEEP,
    )
    # Verify generate was called with max_tokens override
    call_kwargs = mock_llm.generate.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 16384


@pytest.mark.asyncio
async def test_section_writer_medium_no_max_tokens_override(mock_llm, section, outline, evidence_map):
    writer = SectionWriter(mock_llm)
    await writer.write_section(
        section=section,
        outline=outline,
        extractions={},
        evidence_map=evidence_map,
        depth=DepthLevel.MEDIUM,
    )
    call_kwargs = mock_llm.generate.call_args
    # Medium has no override, so max_tokens should not be passed (or be None)
    assert call_kwargs.kwargs.get("max_tokens") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_writing/test_section_writer_depth.py -v`
Expected: FAIL — `TypeError: write_section() got an unexpected keyword argument 'depth'`

- [ ] **Step 3: Add depth parameter to SectionWriter.write_section**

In `autoreview/writing/section_writer.py`:

1. Add `depth: DepthLevel | None = None` parameter to `write_section` signature (line 233-243).
2. Import `get_depth_profile`, `get_depth_instructions` from `autoreview.config.depth`.
3. Before the `build_section_writing_prompt` call (line 276), compute depth args:

```python
        # Compute depth parameters
        target_word_count = section.estimated_word_count if depth else None
        depth_instructions_text = ""
        max_tokens_override = None
        if depth:
            from autoreview.config.depth import get_depth_instructions, get_depth_profile
            depth_instructions_text = get_depth_instructions(depth, section.estimated_word_count)
            max_tokens_override = get_depth_profile(depth).max_tokens_override
```

4. Pass `target_word_count` and `depth_instructions` to `build_section_writing_prompt` (line 276-286):

```python
        prompt = build_section_writing_prompt(
            ...,  # existing args unchanged
            target_word_count=target_word_count,
            depth_instructions=depth_instructions_text,
        )
```

5. Pass `max_tokens` to `self.llm.generate()` **conditionally** (line 288-292) — only include when non-None to avoid passing `None` explicitly:

```python
        generate_kwargs: dict = dict(
            prompt=prompt,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.55,
        )
        if max_tokens_override is not None:
            generate_kwargs["max_tokens"] = max_tokens_override
        response = await self.llm.generate(**generate_kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writing/test_section_writer_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/writing/section_writer.py tests/test_writing/test_section_writer_depth.py
git commit -m "feat(writing): wire depth config into SectionWriter"
```

---

### Task 10: Wire depth into NarrativeArchitect

**Files:**
- Modify: `autoreview/writing/narrative_architect.py:32-48`
- Test: `tests/test_writing/test_narrative_architect_depth.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_writing/test_narrative_architect_depth.py
import pytest
from unittest.mock import AsyncMock, patch

from autoreview.config.models import DepthLevel
from autoreview.writing.narrative_architect import NarrativeArchitect
from autoreview.llm.prompts.outline import ReviewOutline, OutlineSection
from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.llm.provider import LLMResponse


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate_structured = AsyncMock(
        return_value=type("Resp", (), {
            "parsed": type("Plan", (), {
                "central_argument": "Test argument",
                "narrative_arc": "Test arc",
                "section_directives": [],
            })(),
            "input_tokens": 100,
            "output_tokens": 200,
        })()
    )
    return llm


@pytest.fixture
def outline():
    return ReviewOutline(
        title="Test",
        sections=[
            OutlineSection(
                id="s1", title="Topic", description="Desc",
                paper_ids=["p1"], theme_refs=[],
            ),
        ],
    )


@pytest.fixture
def evidence_map():
    return EvidenceMap(
        themes=[], consensus_claims=[], contradictions=[], gaps=[],
    )


@pytest.mark.asyncio
async def test_narrative_architect_accepts_depth(mock_llm, outline, evidence_map):
    architect = NarrativeArchitect(mock_llm)
    # Should not raise
    await architect.plan(
        outline=outline,
        evidence_map=evidence_map,
        scope_document="Test scope",
        depth=DepthLevel.DEEP,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_writing/test_narrative_architect_depth.py -v`
Expected: FAIL — `TypeError: plan() got an unexpected keyword argument 'depth'`

- [ ] **Step 3: Add depth parameter to NarrativeArchitect.plan**

In `autoreview/writing/narrative_architect.py`:

1. Add `depth: DepthLevel | None = None` to `plan()` signature (lines 32-37).
2. Pass `depth=depth` to `build_narrative_planning_prompt()` call (line 48).
3. Add import: `from autoreview.config.models import DepthLevel`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writing/test_narrative_architect_depth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/writing/narrative_architect.py tests/test_writing/test_narrative_architect_depth.py
git commit -m "feat(writing): wire depth config into NarrativeArchitect"
```

---

### Task 11: Wire depth into pipeline nodes

**Files:**
- Modify: `autoreview/pipeline/nodes.py:680-705,707-727,1079-1127`

This task wires together all the pieces in the pipeline orchestration layer. The three methods to modify:

- [ ] **Step 1: Modify `outline` method (~line 680)**

Two changes: (a) pass `depth` to `OutlineGenerator.generate()`, and (b) call the allocator after outline generation.

The outline node calls `OutlineGenerator.generate()` or the outline critique loop. Find where `OutlineGenerator` is instantiated and `generate()` is called, and pass `depth=self.config.writing.depth`.

Then add the allocator call after outline generation:

```python
    # After outline is generated and stored in kb.outline (around line 697)
    from autoreview.config.depth import EvidenceWeightedAllocator, get_depth_profile
    depth = self.config.writing.depth
    profile = get_depth_profile(depth)
    allocator = EvidenceWeightedAllocator(profile)
    # Reconstruct outline for mutation
    outline = ReviewOutline.model_validate(kb.outline)
    allocator.allocate(outline, kb.evidence_map, kb.extractions)
    kb.outline = outline.model_dump()
```

Note: The outline node may call `outline_critique_loop()` from `autoreview/critique/revision.py` which internally calls `OutlineGenerator.generate()`. If so, `depth` must also be forwarded through `outline_critique_loop()` → `OutlineGenerator.generate()`. Check the actual call site and add `depth` to the full propagation chain.

- [ ] **Step 2: Modify `narrative_planning` method (~line 707)**

Pass `depth` to `architect.plan()`:

```python
    # Where architect.plan() is called (around line 714-718)
    # Add depth=self.config.writing.depth to the call
```

- [ ] **Step 3: Modify `section_writing` method (~line 1079)**

Pass `depth` to `writer.write_all_sections()` (NOT `write_section()` — the pipeline calls `write_all_sections` at line 1087):

```python
    drafts = await writer.write_all_sections(
        outline,
        kb.extractions,
        kb.evidence_map,
        narrative_plan=kb.narrative_plan,
        contextual_enrichment=kb.contextual_enrichment or None,
        depth=self.config.writing.depth,   # NEW
    )
```

- [ ] **Step 4: Run existing pipeline tests to verify no regressions**

Run: `python -m pytest tests/ -v --tb=short -x`
Expected: All existing tests PASS (depth defaults to MEDIUM, which preserves current behavior)

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/nodes.py
git commit -m "feat(pipeline): wire depth config through outline, narrative, and section nodes"
```

---

## Chunk 4: CLI + Integration

### Task 12: Add --depth CLI flag

**Files:**
- Modify: `autoreview/cli.py:57-76,146-163`

- [ ] **Step 1: Add --depth flag to run command**

In `autoreview/cli.py`, add to the `run` command options (after `--date-range`, around line 73-74):

```python
    depth: str = typer.Option("medium", "--depth", help="Review depth: low, medium, or deep"),
```

In the command body, apply the depth to config (near where `--date-range` is applied, around line 108):

```python
    from autoreview.config.models import DepthLevel
    config.writing.depth = DepthLevel(depth)
    if config.writing.depth == DepthLevel.DEEP:
        typer.echo("Note: deep mode generates significantly longer output — expect higher token costs.")
```

- [ ] **Step 2: Add --depth flag to resume command**

Add the same `--depth` option to the `resume` command (around line 160-161):

```python
    depth: str | None = typer.Option(None, "--depth", help="Override review depth: low, medium, or deep"),
```

In the resume body, apply depth with the resume warning:

```python
    if depth:
        from autoreview.config.models import DepthLevel
        new_depth = DepthLevel(depth)
        if new_depth != config.writing.depth:
            # Check if resuming past outline
            if resume_node not in ("query_expansion", "search", "screening",
                                    "full_text_retrieval", "extraction", "clustering",
                                    "gap_search", "outline"):
                typer.echo(
                    "Warning: --depth changed but outline word counts are from original run. "
                    "Re-run from 'outline' for full depth recalculation."
                )
        config.writing.depth = new_depth
        if new_depth == DepthLevel.DEEP:
            typer.echo("Note: deep mode generates significantly longer output — expect higher token costs.")
```

- [ ] **Step 3: Test CLI manually**

Run: `python -m autoreview run --help`
Expected: `--depth` flag appears in help output with description

Run: `python -m autoreview resume --help`
Expected: `--depth` flag appears in help output

- [ ] **Step 4: Commit**

```bash
git add autoreview/cli.py
git commit -m "feat(cli): add --depth flag to run and resume commands"
```

---

### Task 13: Integration test — depth ordering

**Files:**
- Create: `tests/test_integration/test_depth_integration.py`

- [ ] **Step 1: Write integration test verifying depth scales word counts**

```python
# tests/test_integration/test_depth_integration.py
"""Integration test: verify that depth levels produce different word allocations."""
import pytest
from unittest.mock import MagicMock

from autoreview.config.models import DepthLevel
from autoreview.config.depth import EvidenceWeightedAllocator, get_depth_profile
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.analysis.evidence_map import EvidenceMap


def _make_realistic_outline() -> ReviewOutline:
    """Simulate a typical 6-section review outline."""
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(id="s1", title="Introduction", description="Intro",
                          paper_ids=["p1", "p2"], theme_refs=[]),
            OutlineSection(id="s2", title="Methods of Review", description="Search strategy",
                          paper_ids=["p1"], theme_refs=[]),
            OutlineSection(id="s3", title="Deep Learning Architectures", description="DL overview",
                          paper_ids=["p1", "p2", "p3", "p4"], theme_refs=[]),
            OutlineSection(id="s4", title="Training Strategies", description="Training",
                          paper_ids=["p2", "p3", "p5"], theme_refs=[]),
            OutlineSection(id="s5", title="Applications and Results", description="Applications",
                          paper_ids=["p1", "p3", "p4", "p5", "p6"], theme_refs=[]),
            OutlineSection(id="s6", title="Conclusion", description="Summary",
                          paper_ids=["p1", "p2"], theme_refs=[]),
        ],
    )


def _make_realistic_extractions() -> dict:
    extractions = {}
    findings_per_paper = {"p1": 3, "p2": 4, "p3": 5, "p4": 2, "p5": 3, "p6": 4}
    for pid, n in findings_per_paper.items():
        mock = MagicMock()
        mock.key_findings = [f"finding_{i}" for i in range(n)]
        extractions[pid] = mock
    return extractions


def _make_realistic_evidence_map() -> EvidenceMap:
    em = EvidenceMap(themes=[], consensus_claims=[], contradictions=[], gaps=[])
    em.evidence_chains = [
        {"paper_ids": ["p1", "p2", "p3"], "chain_id": "c1"},
        {"paper_ids": ["p3", "p4", "p5"], "chain_id": "c2"},
        {"paper_ids": ["p5", "p6"], "chain_id": "c3"},
    ]
    return em


def test_depth_levels_produce_increasing_total_words():
    """low < medium < deep total word count."""
    extractions = _make_realistic_extractions()
    evidence_map = _make_realistic_evidence_map()
    totals = {}

    for level in DepthLevel:
        outline = _make_realistic_outline()
        profile = get_depth_profile(level)
        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline, evidence_map, extractions)
        totals[level] = sum(s.estimated_word_count for s in outline.sections)

    assert totals[DepthLevel.LOW] < totals[DepthLevel.MEDIUM]
    assert totals[DepthLevel.MEDIUM] < totals[DepthLevel.DEEP]


def test_introduction_dampened_relative_to_body():
    """Introduction gets fewer words than a body section with same evidence."""
    outline = _make_realistic_outline()
    extractions = _make_realistic_extractions()
    evidence_map = _make_realistic_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    intro = outline.sections[0]  # Introduction
    # Find a body section with similar paper count
    body_sections = [s for s in outline.sections if s.title not in
                     ("Introduction", "Methods of Review", "Conclusion")]
    # At least one body section should have more words than intro
    assert any(s.estimated_word_count > intro.estimated_word_count for s in body_sections)


def test_all_sections_above_floor():
    """Every section meets the minimum word count for its depth level."""
    for level in DepthLevel:
        outline = _make_realistic_outline()
        extractions = _make_realistic_extractions()
        evidence_map = _make_realistic_evidence_map()
        profile = get_depth_profile(level)

        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline, evidence_map, extractions)

        for section in outline.sections:
            # Zero-evidence sections get fixed allocation which may be below min_section_words
            if section.paper_ids:
                assert section.estimated_word_count >= profile.min_section_words, (
                    f"{level}: {section.title} has {section.estimated_word_count} words, "
                    f"below floor of {profile.min_section_words}"
                )


def test_budget_overflow_floors_preserved():
    """When many sections hit the floor, total may exceed budget (soft target). Floors are never violated."""
    # 10 sections with minimal evidence at LOW depth
    # Budget = 4000, floor = 200, so 10 * 200 = 2000 from floors alone
    sections = [
        {"id": f"s{i}", "title": f"Topic {i}", "paper_ids": [f"p{i}"]}
        for i in range(10)
    ]
    outline = _make_realistic_outline.__wrapped__(sections) if hasattr(_make_realistic_outline, '__wrapped__') else ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(id=s["id"], title=s["title"], description="Desc",
                          paper_ids=s["paper_ids"], theme_refs=[])
            for s in sections
        ],
    )
    extractions = _make_realistic_extractions()
    # Only p1 exists in extractions, others will have 0 findings but non-zero paper count
    for i in range(10):
        pid = f"p{i}"
        if pid not in extractions:
            mock = MagicMock()
            mock.key_findings = [f"finding_0"]
            extractions[pid] = mock

    evidence_map = _make_realistic_evidence_map()
    profile = get_depth_profile(DepthLevel.LOW)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    # All sections must be >= floor
    for section in outline.sections:
        if section.paper_ids:
            assert section.estimated_word_count >= profile.min_section_words
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration/test_depth_integration.py -v`
Expected: PASS (all tests)

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS — no regressions from depth feature

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration/test_depth_integration.py
git commit -m "test: add integration tests for depth feature word allocation"
```

---

## Dependency Graph

```
Task 1 (DepthLevel enum)
  └→ Task 2 (DepthProfile)
       └→ Task 3 (Section classifier)
            └→ Task 4 (Allocator)
                 └→ Task 11 (Pipeline nodes - outline)

Task 5 (Depth instructions helper)
  └→ Task 6 (Writing prompt)
       └→ Task 9 (SectionWriter)
            └→ Task 8c (write_all_sections)
                 └→ Task 11 (Pipeline nodes - section_writing)

Task 7 (Outline prompt)
  └→ Task 8b (OutlineGenerator)
       └→ Task 11 (Pipeline nodes - outline)

Task 2 (DepthProfile) ─→ Task 8 (Narrative prompt, calls get_depth_profile)
Task 8 (Narrative prompt)
  └→ Task 10 (NarrativeArchitect)
       └→ Task 11 (Pipeline nodes - narrative_planning)

Task 11 (Pipeline nodes) ─→ Task 12 (CLI)
Task 12 (CLI) ─→ Task 13 (Integration test)
```

**Parallelizable batches:**
- Batch 1: Task 1
- Batch 2: Tasks 2, 5, 7 (depend only on Task 1)
- Batch 3: Tasks 3, 6, 8, 8b (Task 8 depends on Task 2; Task 8b depends on Task 7)
- Batch 4: Tasks 4, 9, 10 (depend on batch 3)
- Batch 5: Task 8c (depends on Task 9)
- Batch 6: Task 11 (depends on tasks 4, 8b, 8c, 10)
- Batch 7: Task 12 (depends on task 11)
- Batch 8: Task 13 (depends on task 12)
