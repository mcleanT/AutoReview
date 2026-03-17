# Design Spec: Review Depth Control

**Date**: 2026-03-16
**Status**: Approved
**Author**: Claude + MST

---

## Problem

AutoReview currently produces reviews at a single, implicit depth — every section is written with roughly the same level of detail regardless of the user's intent. There is no way to request a concise highlights-only review or an exhaustive deep-dive. The `estimated_word_count` field on `OutlineSection` exists but is never used in the writing prompts.

## Goal

Add a `depth` feature with three levels — **low**, **medium**, **deep** — that controls the granularity and length of the generated review. The outline structure (sections, ordering) remains the same across all depth levels. Only the amount of detail extracted per section and the physical length of the output changes.

| Depth | Target Behavior | Approximate Length |
|-------|----------------|-------------------|
| Low | Critical findings only. Distilled, high-impact. | Shorter than average review (~4,000 words) |
| Medium | Standard academic review depth. Major findings with supporting context. | Average review length (~8,000 words) |
| Deep | Exhaustive. Evidence chains, contradictions, methodological comparisons, temporal evolution. | Book-chapter depth (~25,000+ words) |

## Design Decisions

1. **Granularity, not scope** — Depth controls detail per section, not the number of sections. The outline structure is depth-invariant.
2. **Global with automatic section-type adjustment** — One `--depth` flag applies to the whole run. Introduction and Conclusion sections automatically receive dampened word counts relative to body sections.
3. **Critique system is depth-unaware** — The critique/revision system continues to evaluate synthesis quality regardless of depth. Depth affects writing prompts only.

## Architecture

### Core Data Model

#### `DepthLevel` (Enum)

```python
class DepthLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    DEEP = "deep"
```

#### `DepthProfile` (Frozen Dataclass)

A deterministic lookup keyed by `DepthLevel`. No LLM involved.

| Parameter | Low | Medium | Deep |
|-----------|-----|--------|------|
| `base_word_multiplier` | 0.6 | 1.0 | 2.5 |
| `key_insights_range` | (2, 3) | (3, 5) | (7, 10) |
| `evidence_chain_detail` | `"critical_only"` | `"standard"` | `"exhaustive"` |
| `total_word_budget` | ~4,000 | ~8,000 | ~25,000+ |
| `min_section_words` | 200 | 400 | 600 |
| `section_type_dampening` | see below | see below | see below |

**Section-type dampening** (multiplier applied after proportional allocation):

| Section Type | Low | Medium | Deep |
|-------------|-----|--------|------|
| Introduction | 0.7 | 0.8 | 0.85 |
| Conclusion | 0.6 | 0.7 | 0.75 |
| Methods of Review | 0.7 | 0.8 | 0.85 |
| Body sections | 1.0 | 1.0 | 1.0 |

#### `EvidenceWeightedAllocator`

Takes a `DepthProfile`, the `ReviewOutline`, the `EvidenceMap`, and the extractions dict (`dict[str, PaperExtraction]`). Distributes the word budget across sections proportionally based on evidence density.

**Evidence density per section** is computed from the pipeline's existing data structures:

```python
def _compute_density(
    section: OutlineSection,
    extractions: dict[str, PaperExtraction],
    evidence_map: EvidenceMap,
) -> float:
    # Count of papers assigned to this section
    n_papers = len(section.paper_ids)

    # Count unique findings by summing len(extractions[pid].key_findings)
    # for each pid in section.paper_ids (skip missing extractions gracefully)
    n_findings = sum(
        len(extractions[pid].key_findings)
        for pid in section.paper_ids
        if pid in extractions
    )

    # Count evidence chains that overlap with this section's papers
    section_pids = set(section.paper_ids)
    n_chains = sum(
        1 for chain in evidence_map.evidence_chains
        if set(chain.paper_ids) & section_pids
    )

    return float(n_papers + n_findings + n_chains)
```

**Section-type classification** is determined by fuzzy title matching against known categories. A helper function normalizes the section title to lowercase and checks for substring membership:

- **Introduction**: title contains "introduction" or "background"
- **Conclusion**: title contains "conclusion" or "concluding"
- **Methods of Review**: title contains "method" or "search strategy" or "review methodology"
- **Body** (default): anything not matching the above

This avoids brittle exact-match comparisons against LLM-generated titles.

**Allocation algorithm:**

1. Compute `density(s)` for each section `s`. Sections with zero evidence (e.g., "Future Directions") receive a fixed allocation of `base_word_multiplier * 500` words (depth-scaled but evidence-independent), and are excluded from proportional allocation.
2. Compute proportional share: `share(s) = density(s) / sum(all densities)` (over evidence-bearing sections only)
3. Allocate: `raw_words(s) = share(s) * remaining_budget` (where `remaining_budget = total_word_budget - sum(fixed allocations)`)
4. Apply section-type dampening: `adjusted_words(s) = raw_words(s) * dampening(section_type(s))`
5. Enforce floor: `final_words(s) = max(adjusted_words(s), min_section_words)`
6. If `sum(all final_words) > total_word_budget`: scale body sections down proportionally to fit. If `sum(all final_words) < total_word_budget`: distribute surplus to body sections proportionally by density.
7. Write `final_words(s)` into `OutlineSection.estimated_word_count`

**Budget overflow edge case**: When many sections hit the `min_section_words` floor (especially in low-depth mode with many sections), the sum may exceed `total_word_budget`. In this case, the budget is treated as a **soft target** — floors are never violated. The `total_word_budget` values are approximate guidance, not hard caps.

This is a **deterministic computation** — no LLM calls, runs in microseconds.

### Injection Points

Depth flows into the pipeline at three reinforcing points:

#### 1. Outline Generation

The outline prompt receives `DepthLevel` as context. It does not change the section structure, but affects the `description` field on each `OutlineSection`:

- **Low**: descriptions emphasize "key findings" and "critical takeaways"
- **Medium**: standard descriptions (current behavior)
- **Deep**: descriptions request exhaustive coverage — "trace methodological evolution, compare conflicting findings, discuss edge cases"

After the outline LLM call returns, the `EvidenceWeightedAllocator` **overwrites** `estimated_word_count` on each section. The LLM's own word count estimates are discarded.

#### 2. Narrative Directives

`SectionNarrativeDirective.key_insights` is bounded by the depth profile:

- **Low**: narrative planner selects 2-3 highest-impact insights only
- **Medium**: 3-5 insights (current behavior)
- **Deep**: 7-10 insights, including secondary evidence chains, contradictions, and methodological nuances

The narrative prompt receives both the `DepthLevel` and the allocator's `target_word_count` for the section, allowing it to calibrate how many narrative threads to plan.

**Bounding mechanism**: The depth-specific insight range is injected into the narrative prompt text itself (replacing the current hardcoded "3-5" in the system prompt with the depth profile's `key_insights_range`). The LLM is instructed to generate within the specified range. No post-processing truncation is applied — the prompt guides the output directly.

#### 3. Section Writing Prompt

`build_section_writing_prompt()` receives two new parameters:

- **`target_word_count: int`** — from the allocator
- **`depth_instructions: str`** — depth-specific prose guidance

Depth instructions by level:

| Depth | Instruction |
|-------|-------------|
| Low | "Distill to the most critical findings. One paragraph per key claim. Prioritize clarity and impact over completeness. Target approximately {target_word_count} words." |
| Medium | "Balance thoroughness with readability. Cover major findings with supporting context. Standard academic review depth. Target approximately {target_word_count} words." |
| Deep | "Exhaustively trace evidence chains. Include methodological comparisons, conflicting results with resolution analysis, temporal evolution of findings, and secondary implications. Prioritize completeness over brevity. Target approximately {target_word_count} words." |

These are injected as new formatted blocks within the existing prompt — no new prompt files.

### Pipeline Integration

#### Config Change

One new field on `WritingConfig`:

```python
class WritingConfig(BaseModel):
    depth: DepthLevel = DepthLevel.MEDIUM
    # ... existing fields unchanged
```

Default is `medium`, preserving current behavior for all existing runs.

#### DAG — No New Node

The `EvidenceWeightedAllocator` runs as a **post-processing step inside the outline node**, not as a separate DAG node. After the outline LLM call completes and the outline is parsed, the allocator adjusts word counts before the node returns its result. This avoids DAG changes, new snapshot stages, and resume complexity.

The allocator needs access to the `EvidenceMap` (from `clustering`) and the extractions dict (`kb.extractions`, from `extraction`). Both are available in the pipeline state when the outline node runs, since `outline` depends on `gap_search` which depends on `clustering` which depends on `extraction`.

**`max_tokens_generate` adjustment**: The current `LLMConfig.max_tokens_generate` default is 4096, which is sufficient for low and medium depth but will truncate deep-mode sections (which may target 3,000+ words ≈ 4,000+ tokens). The depth feature adds a `max_tokens_override` lookup to `DepthProfile`:

| Depth | `max_tokens_override` |
|-------|----------------------|
| Low | `None` (use default 4096) |
| Medium | `None` (use default 4096) |
| Deep | 16384 |

The section writer passes this override to `self.llm.generate(max_tokens=...)` when present, bypassing the global default for that call only.

#### CLI

One new flag on both `run` and `resume`:

```bash
autoreview run "topic" --depth low|medium|deep
autoreview resume <snapshot> --depth low|medium|deep
```

Default: `medium`. Stored in the pipeline snapshot so `resume` preserves the depth setting from the original run.

**Resume semantics**: If `--depth` is passed to `resume`, it is only effective when resuming from the `outline` node or earlier (since the allocator runs inside the outline node). If resuming from a later node (e.g., `section_writing`), a changed `--depth` flag will update prompt-level instructions (depth_instructions, key_insights_range) but will **not** re-run the allocator — the word counts baked into the outline remain from the original run. The CLI emits a warning in this case: `"Warning: --depth changed but outline word counts are from original run. Re-run from 'outline' for full depth recalculation."`

### What Does NOT Change

- **Search/retrieval pipeline** — depth affects writing, not what papers are found
- **Extraction** — same findings are extracted regardless of depth
- **Critique system** — same rubric, same thresholds, same revision cycles
- **Assembly/polish** — works on whatever the sections produce
- **Evaluation framework** — rubrics evaluate quality, not length
- **Outline structure** — same sections and ordering at all depth levels

## File Changes Summary

All paths are relative to `autoreview/`. The pipeline nodes live in a single file (`pipeline/nodes.py`) as methods on the `PipelineNodes` class.

| File | Change |
|------|--------|
| `config/models.py` | Add `DepthLevel` enum, `depth` field to `WritingConfig`. Note: `WritingConfig` uses `extra="forbid"`, so `DepthLevel(str, Enum)` serialization must be verified. |
| `config/depth.py` (new) | `DepthProfile` dataclass, `DEPTH_PROFILES` lookup, `EvidenceWeightedAllocator` |
| `llm/prompts/outline.py` | Pass `DepthLevel` into outline system prompt for description guidance |
| `llm/prompts/narrative.py` | Pass `DepthLevel` + `target_word_count` to bound `key_insights` range (replace hardcoded "3-5") |
| `llm/prompts/writing.py` | Add `target_word_count` and `depth_instructions` parameters to `build_section_writing_prompt()` |
| `writing/section_writer.py` | Pass depth config, allocated word count, and `max_tokens_override` through to prompt builder and `self.llm.generate()` |
| `writing/narrative_architect.py` | Update `NarrativeArchitect.plan()` signature to accept `DepthLevel` and per-section word counts, forward to prompt builder |
| `pipeline/nodes.py` | In `outline` method (~line 680): call `EvidenceWeightedAllocator.allocate()` after outline generation, passing `kb.extractions` and `evidence_map`. In `narrative_planning` method (~line 707): forward depth config. In `section_writing` method (~line 1079): forward depth config + word counts. |
| `cli.py` | Add `--depth` flag to `run` and `resume` commands, with resume warning for post-outline depth changes |
| Domain YAML configs | No changes — depth is orthogonal to domain |

## Testing Strategy

- **Unit test `EvidenceWeightedAllocator`**: verify proportional allocation, dampening, floor enforcement, surplus redistribution with mock evidence maps
- **Unit test `DepthProfile` lookup**: verify all three profiles return expected parameters
- **Unit test prompt builders**: verify depth instructions and word counts appear in generated prompts at each level
- **Integration test**: run a small fixture topic at all three depth levels, verify output length ordering (low < medium < deep)
- **No changes to critique tests** — critique is depth-unaware

## Resolved Design Questions

1. **Tuning the word budgets** — the 4K / 8K / 25K targets are initial estimates and soft targets. The allocator will never violate `min_section_words` floors even if that exceeds the budget. Empirical tuning is expected after initial runs.
2. **Deep mode token costs** — deep mode automatically increases `max_tokens_generate` to 16384 for section writing calls (see `max_tokens_override` in Pipeline Integration). The CLI prints an informational message: `"Note: deep mode generates significantly longer output — expect higher token costs."` This is informational, not a confirmation prompt.
3. **Zero-evidence sections** — sections with zero assigned papers (e.g., "Future Directions") receive a fixed allocation of `base_word_multiplier * 500` words, scaled by depth but independent of evidence density. They are excluded from proportional allocation. This ensures these sections scale with depth without being starved by the evidence-weighted algorithm.
