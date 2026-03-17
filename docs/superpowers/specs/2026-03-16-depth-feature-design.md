# Design Spec: Review Depth Control

**Date**: 2026-03-16
**Status**: Draft
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

Takes a `DepthProfile`, the `ReviewOutline`, and the `EvidenceMap`. Distributes the word budget across sections proportionally based on evidence density.

**Evidence density per section** is computed as:

```
density(section) = (
    count(assigned_papers)
    + count(unique_findings)
    + count(evidence_chains_touching_section)
)
```

**Allocation algorithm:**

1. Compute `density(s)` for each section `s`
2. Compute proportional share: `share(s) = density(s) / sum(all densities)`
3. Allocate: `raw_words(s) = share(s) * total_word_budget`
4. Apply section-type dampening: `adjusted_words(s) = raw_words(s) * dampening(section_type(s))`
5. Enforce floor: `final_words(s) = max(adjusted_words(s), min_section_words)`
6. Redistribute surplus from dampened/floored sections proportionally to body sections
7. Write `final_words(s)` into `OutlineSection.estimated_word_count`

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

The allocator needs access to the `EvidenceMap` which is produced by the `clustering` node. Since `outline` already depends on `gap_search` (which depends on `clustering`), the evidence map is available in the pipeline state when the outline node runs.

#### CLI

One new flag on both `run` and `resume`:

```bash
autoreview run "topic" --depth low|medium|deep
autoreview resume <snapshot> --depth low|medium|deep
```

Default: `medium`. Stored in the pipeline snapshot so `resume` preserves the depth setting from the original run.

### What Does NOT Change

- **Search/retrieval pipeline** — depth affects writing, not what papers are found
- **Extraction** — same findings are extracted regardless of depth
- **Critique system** — same rubric, same thresholds, same revision cycles
- **Assembly/polish** — works on whatever the sections produce
- **Evaluation framework** — rubrics evaluate quality, not length
- **Outline structure** — same sections and ordering at all depth levels

## File Changes Summary

| File | Change |
|------|--------|
| `config/models.py` | Add `DepthLevel` enum, `depth` field to `WritingConfig` |
| `config/depth.py` (new) | `DepthProfile` dataclass, `DEPTH_PROFILES` lookup, `EvidenceWeightedAllocator` |
| `llm/prompts/outline.py` | Pass `DepthLevel` into outline system prompt for description guidance |
| `llm/prompts/narrative.py` | Pass `DepthLevel` + `target_word_count` to bound `key_insights` range |
| `llm/prompts/writing.py` | Add `target_word_count` and `depth_instructions` parameters to `build_section_writing_prompt()` |
| `writing/section_writer.py` | Pass depth config and allocated word count through to prompt builder |
| `pipeline/nodes/outline.py` | Call `EvidenceWeightedAllocator.allocate()` after outline generation |
| `pipeline/nodes/narrative_planning.py` | Forward depth config to narrative prompt builder |
| `pipeline/nodes/section_writing.py` | Forward depth config + word counts to section writer |
| `cli.py` | Add `--depth` flag to `run` and `resume` commands |
| Domain YAML configs | No changes — depth is orthogonal to domain |

## Testing Strategy

- **Unit test `EvidenceWeightedAllocator`**: verify proportional allocation, dampening, floor enforcement, surplus redistribution with mock evidence maps
- **Unit test `DepthProfile` lookup**: verify all three profiles return expected parameters
- **Unit test prompt builders**: verify depth instructions and word counts appear in generated prompts at each level
- **Integration test**: run a small fixture topic at all three depth levels, verify output length ordering (low < medium < deep)
- **No changes to critique tests** — critique is depth-unaware

## Open Questions

1. **Tuning the word budgets** — the 4K / 8K / 25K targets are initial estimates. May need adjustment after empirical runs.
2. **Deep mode token costs** — a 25K+ word review requires significantly more LLM output tokens. Should there be a cost warning in the CLI for deep mode?
3. **Evidence density edge case** — if a section has zero assigned papers (e.g., a "Future Directions" section), the allocator gives it `min_section_words`. Is this the right behavior, or should such sections get a fixed allocation independent of evidence?
