# Claims Graph v2: Condition-Aware Merging

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Improve node-to-node relationships — especially contradiction detection — by making assertion merging condition-aware so that claims from different experimental contexts remain separate edges.

---

## Problem

The v1 graph construction pipeline merges all assertions sharing the same `(subject_id, predicate, object_id)` triple into a single edge (`dedup.py:merge_assertions`, line 528). This discards experimental context:

- "BMP4 induces mesoderm" from mouse ESCs and human iPSCs collapse into one edge
- "BMP4 induces mesoderm" at 10 ng/mL (positive) and 1 ng/mL (negative) collapse into one edge with `direction_conflict=True`
- The NLI pipeline then flags false-positive contradictions on context-dependent biology
- Real contradictions (same system, opposing results) are hidden inside noisy merged edges

## Solution

Change the merge key from `(subject_id, predicate, object_id)` to `(subject_id, predicate, object_id, condition_signature)`, where `condition_signature` is a hash of comparability-relevant experimental conditions.

**Preservation requirement:** All v1 code paths are preserved. New functions are added alongside existing ones. The `build_graph` function gains a `version` parameter (default `1`) to select the merge strategy. v4 extraction data without conditions continues to work identically.

---

## Design

### 1. Condition Signature: Three-Tier Hierarchy

| Tier | Fields | Effect on merge key | Rationale |
|------|--------|-------------------|-----------|
| **Hard partition** | `organism`, `in_vitro` | Different values = separate edges, always | Mouse vs human, in vitro vs in vivo are never comparable |
| **Soft partition** | `model_system` (normalized) | Different values = separate edges, with fuzzy normalization | "mouse ESC gastruloids" and "mESC-derived gastruloids" should merge; "mouse ESC gastruloids" and "mouse embryo" should not |
| **Scoring modifier** | `cell_type`, `treatment`, `developmental_stage`, `concentration` | Same edge, modifies contradiction weight | Dose differences are interesting, not disqualifying |

**Condition signature computation:**

```python
def compute_condition_signature(
    organism: str | None,
    in_vitro: bool | None,
    model_system_class: str | None,
) -> str:
    """Deterministic hash of hard + soft partition fields."""
    key = f"{(organism or '').lower().strip()}|{in_vitro}|{(model_system_class or '').lower().strip()}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]
```

**Model system normalization:** Lightweight fuzzy bucketing using rapidfuzz (same union-find approach as entity dedup pass 3, threshold ~80). Running registry of seen model system strings. Synonyms collapse:
- "mouse ESC gastruloids", "mESC-derived gastruloids" → same bucket
- "mouse embryo", "E8.5 mouse embryo" → same bucket (different from gastruloids)

### 2. Condition-Aware Assertion Merging

New function `merge_assertions_v2(assertions) -> MergeResult` in `dedup.py`:

- **Merge key:** `(subject_id, predicate, object_id, condition_signature)`
- Same interface as `merge_assertions` — returns `MergeResult` with merged assertions and audit log
- Accumulates scoring-modifier conditions (cell types, treatments, stages) across merged claims into lists for downstream analysis
- Direction conflict detection unchanged — but now only flags conflicts within the same experimental context

The existing `merge_assertions` function is preserved unchanged for v1 compatibility.

### 3. New Data Models

**`ConditionContext` (new model in `models.py`):**

```python
class ConditionContext(AutoReviewModel):
    """Structured summary of the experimental context for a condition-partitioned edge."""
    organism: str | None = None
    model_system_class: str | None = None
    in_vitro: bool | None = None
    cell_types: list[str] = []       # accumulated from all merged claims
    treatments: list[str] = []        # accumulated (scoring modifier tier)
    stages: list[str] = []            # accumulated
```

**New fields on `KGEdge`:**

```python
condition_signature: str | None = None      # partition hash
condition_context: ConditionContext | None = None  # structured summary
```

### 4. Contradiction Taxonomy

The NLI pipeline gains a structured contradiction taxonomy based on condition context:

| Type | Source | Weight | Description |
|------|--------|--------|-------------|
| `within_context` | Same (S,P,O) + same condition_signature, opposing evidence | Full (1.0) | Real scientific disagreement |
| `cross_context` | Same (S,P,O), different condition_signature, opposing directions | Discounted (0.3) | Context-dependent biology — interesting but expected |
| `structural` | Opposing predicates on same (S,O) + same condition_signature | Full (1.0) | Direct predicate opposition |
| `nli_semantic` | DeBERTa contradiction in natural language text | Varies | Depends on context match |

**New pair selection logic in `classify_cross_claims`:**

- Within-context pairs: Same (S,P,O) + same `condition_signature` but `direction_conflict` → `method="within_context_conflict"`, full weight
- Cross-context pairs: Same (S,P,O) + different `condition_signature` → `method="cross_context"`, discounted weight
- Structural pairs: Opposing predicates on same (S,O) + same `condition_signature` → `method="structural"`, full weight

**New field on `NLIPairResult`:** `contradiction_type: str | None = None`

### 5. Pipeline Orchestration

`build_graph` in `__init__.py` gains a `version` parameter:

```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,  # NEW: 1 = v1 merge, 2 = condition-aware merge
) -> nx.MultiDiGraph:
```

When `version=2`:
- Step 5 calls `merge_assertions_v2` instead of `merge_assertions`
- Step 5b propagates `condition_context` onto merged assertions
- Step 6 includes `condition_signature` and `condition_context` on KGEdge
- `edge_id` hash includes the condition signature: `f"{subject_id}|{predicate}|{object_id}|{condition_signature}"`

When `version=1`: Behavior is identical to current code.

### 6. Visualization Updates

`interactive.py` changes for v2 graphs:

- Tooltip shows condition context (organism, model system, in vitro/in vivo)
- Parallel edges between the same entities get visual distinction (different dash patterns or subtle color shifts per condition_signature)
- New filter toggle: "Show cross-context contradictions"
- Contradiction edges rendered with taxonomy color coding:
  - `within_context`: red (high value)
  - `cross_context`: orange (interesting)
  - `structural`: magenta (predicate opposition)
  - `nli_semantic`: yellow (NLI-detected)

---

## Module Change Summary

| Module | Change | Scope |
|--------|--------|-------|
| `dedup.py` | Add `merge_assertions_v2`, `normalize_model_system`, `compute_condition_signature`, `ModelSystemRegistry` | New functions alongside existing |
| `models.py` | Add `ConditionContext` model, add `condition_signature` and `condition_context` fields to `KGEdge` | Additive — new optional fields |
| `__init__.py` | Add `version` param to `build_graph`, branch on merge strategy | Backwards-compatible |
| `graph.py` | Add `condition_signature` flat attr on edges | Additive |
| `confidence.py` | No changes | Evidence is already per-edge; condition partitioning scopes it automatically |
| `nli.py` | Add `contradiction_type` to `NLIPairResult`, update pair selection for cross-context vs within-context, attach `condition_signature` to claim dicts | Additive |
| `interactive.py` | Condition context in tooltips, parallel edge styling, contradiction taxonomy coloring, cross-context toggle | Additive |
| `ingest.py` | No changes | Already parses v5 conditions |

## Backwards Compatibility

- v4 extractions (no conditions) get `condition_signature=None`. All-null signatures hash identically, so merging behavior is unchanged.
- `version=1` (default) uses the original `merge_assertions` function — zero behavior change.
- All existing tests continue to pass against v1 code paths.
- Existing graph pickle files remain loadable — new KGEdge fields default to `None`.

## Testing Strategy

- Unit tests for `compute_condition_signature`: deterministic, stable across runs
- Unit tests for `normalize_model_system`: synonym bucketing, edge cases (None, empty string)
- Unit tests for `merge_assertions_v2`: verify same (S,P,O) with different conditions produce separate edges; same conditions merge
- Integration test: build v2 graph from micro_v5 corpus (11 papers), compare edge count and contradiction count against v1
- Regression test: v1 path produces identical output to current code
