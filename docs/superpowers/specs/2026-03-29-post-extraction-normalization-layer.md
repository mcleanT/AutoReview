# Post-Extraction Claim Normalization Layer

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Improve cross-paper claim matching for contradiction detection by normalizing extraction output before entity dedup and assertion merging.

---

## Problem

The v8.7 extraction prompt produces claims with:
- **Verbose/compound objects** that never match across papers ("self-organization into endoderm and mesoderm subpopulations" vs "mesoderm differentiation")
- **Predicate non-compliance** (LLM outputs `promotes` despite closed vocabulary requiring `induces`)
- **Null quantitative context** despite numerical values present in the natural language text
- **Inconsistent entity naming** (parenthetical synonyms, leading articles, trailing descriptors)

The graph construction pipeline (v1 and v2) can only detect contradictions between claims that share entity pairs. When entity names are too verbose or inconsistent to match, real contradictions are invisible.

## Solution

A new `normalize.py` module in `autoreview/knowledge_graph/` with a `ClaimNormalizer` class that runs two normalization passes:

1. **Pre-dedup pass** — text cleaning + compound decomposition before entity dedup, so normalized names can match across papers
2. **Post-dedup pass** — quantitative context backfill after entity resolution, enriching assertions for contradiction scoring

**Preservation requirement:** `normalize=False` by default. All existing code paths unchanged. Normalization is opt-in and independent of graph version (works with v1 and v2).

---

## Design

### 1. Module Structure

**New file:** `autoreview/knowledge_graph/normalize.py`

```python
class ClaimNormalizer:
    def __init__(self, llm_decompose: bool = True):
        """
        Args:
            llm_decompose: Enable LLM fallback for compound objects
                that rule-based patterns can't handle.
        """

    def pre_dedup(self, claims: list[dict]) -> tuple[list[dict], NormalizationReport]:
        """Run before entity dedup. Text cleaning + decomposition. May produce new claims (1->N)."""

    def post_dedup(self, assertions: list[dict]) -> tuple[list[dict], NormalizationReport]:
        """Run after entity dedup, before merge. Quantitative backfill. Always 1->1."""
```

**Pipeline insertion in `__init__.py`:**

```
Step 1:  Ingest
Step 1b: if normalize: ClaimNormalizer.pre_dedup(claims)
Step 2:  Entity dedup
Step 3:  Predicate normalization
Step 3b: if normalize: ClaimNormalizer.post_dedup(assertions)
Step 4:  Merge (v1 or v2)
Step 5:  KGEdge construction
...
```

### 2. Pre-dedup Transform 1: Text Cleaning

Normalize subject and object `name` fields to improve entity dedup matching.

**Transforms applied in order:**

1. **Strip parenthetical synonyms** — `"Invariant chain (CD74)"` → `"Invariant chain"`. Regex: `r'\s*\([^)]*\)'`. The synonym is preserved in an `_extracted_aliases` list on the claim dict.

2. **Strip leading articles/qualifiers** — `"the Wnt signaling pathway"` → `"Wnt signaling pathway"`. Prefix list: `the, a, an, this, these, that, those`.

3. **Collapse whitespace** — Multiple spaces, tabs, newlines → single space, then strip.

4. **Strip trailing descriptors** — `"mesoderm differentiation process"` → `"mesoderm differentiation"`. Suffix list: `process, pathway, mechanism, activity, event, response, cascade, system`. Only strip if the remaining string is >2 words (avoids mangling short names like "Wnt pathway").

5. **Lowercase normalization** — Not applied to the display name (entity dedup handles case), but applied to an internal `_normalized` key for matching decisions.

**Preservation:** Original `name` values stored in `_original_subject_name` / `_original_object_name` on the claim dict for audit trail.

### 3. Pre-dedup Transform 2: Predicate String Cleaning

Fix malformed predicates before the existing `PredicateNormalizer` synonym table.

**Transforms:**

1. **Whitespace/punctuation cleanup** — `"activ ates"` → `"activates"`, `"induces."` → `"induces"`. Strip internal whitespace, trailing punctuation.

2. **Underscore normalization** — `"is required for"` → `"is_required_for"`. If the string with spaces replaced by underscores matches a canonical predicate, convert.

3. **Tense normalization** — `"inhibited"` → `"inhibits"`, `"induced"` → `"induces"`. Small mapping (~15 predicates × common inflections: past tense, gerund). Not a full lemmatizer.

**Ordering:** Runs before `PredicateNormalizer` in the pipeline. The existing normalizer is unchanged.

### 4. Pre-dedup Transform 3: Compound Object Decomposition

Split claims with multi-concept objects into atomic claims.

**Rule-based patterns (first pass):**

| Pattern | Example | Result |
|---------|---------|--------|
| Conjunction split | `"endoderm and mesoderm differentiation"` | → `"endoderm differentiation"` + `"mesoderm differentiation"` |
| Comma list | `"T, Sox2, and Nanog expression"` | → `"T expression"` + `"Sox2 expression"` + `"Nanog expression"` |
| Prepositional compound | `"self-organization into endoderm and mesoderm"` | → `"endoderm"` + `"mesoderm"` (prepositional wrapper stripped, bare nouns kept) |
| Slash split | `"BMP4/WNT signaling"` | → `"BMP4 signaling"` + `"WNT signaling"` |

**Head noun distribution:** The shared qualifier (head noun) distributes across conjuncts. `"endoderm and mesoderm subpopulations"` → each conjunct gets `"subpopulations"`. Detected by checking if the last token after the conjunction matches a known biological noun class: `process, differentiation, expression, signaling, pathway, formation, specification, development, migration, proliferation, apoptosis, activation, inhibition, regulation`.

**LLM fallback (second pass):**

Claims where the object is >6 words AND no rule-based pattern matched get sent to a cheap model (haiku) with a structured prompt:

```
Decompose this biological concept into atomic concepts.
Input: "self-organization of human gastruloids into homogenous subpopulations of endoderm and mesoderm"
Output: ["endoderm differentiation", "mesoderm differentiation"]
If the concept is already atomic, return it unchanged as a single-element list.
```

Batched — all flagged objects in a single API call to minimize latency/cost.

**Output:** Each decomposed claim is a new dict with:
- Same subject, predicate, direction, conditions, evidence as the parent
- New object name (the atomic concept)
- `_decomposed_from: original_claim_id` for audit trail
- New `claim_id` suffixed: `c_001_d1`, `c_001_d2`

**Scope limits:**
- No subject decomposition (subjects are typically single entities)
- No decomposition if object is ≤3 words (already atomic)
- No semantic judgment on whether decomposition changes meaning — purely structural

### 5. Post-dedup Transform: Quantitative Context Backfill

When `quantitative_context` is null, parse the `natural_language` claim text to extract structured values.

**Extraction patterns (regex-based):**

| Field | Pattern | Examples |
|-------|---------|----------|
| `concentration` | `\d+\.?\d*\s*(ng/mL\|µM\|nM\|mM\|µg/mL\|%)\s+\w+` | "10 ng/mL BMP4", "3 µM CHIR99021" |
| `timepoint` | `(at\s+)?\d+\.?\d*\s*(h\|hr\|hrs\|hours\|d\|days\|min\|minutes)` | "at 48h", "day 5", "72 hours" |
| `dose` | `\d+\.?\d*\s*(mg/kg\|µg/kg\|Gy\|U/mL)` | "5 mg/kg", "2 Gy" |
| `fold_change` | `\d+\.?\d*-fold\s+(increase\|decrease\|change\|reduction\|upregulation)` | "3.5-fold increase" |

**Rules:**
- Only runs when `quantitative_context` is null or all fields are null — never overwrites existing values
- Extracts from `natural_language` field first, falls back to `conditions.treatment` string
- Multiple matches for the same field: take the first (most prominent in the sentence)
- Unit normalization: `hr/hrs/hours` → `h`, `days` → `d`, `minutes` → `min`

**Scope limits:**
- No unit conversion (10 ng/mL stays 10 ng/mL, not 0.01 µg/mL)
- No inference of missing values from other claims
- No validation against evidence units

### 6. Pipeline Integration

`build_graph` in `__init__.py` gains two parameters:

```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
    normalize: bool = False,      # NEW
    llm_decompose: bool = True,   # NEW
) -> nx.MultiDiGraph:
```

- `normalize=False` by default — zero behavior change for existing callers
- `normalize=True` inserts the two normalization passes
- `llm_decompose=True` enables haiku fallback for compound objects; set `False` for deterministic-only runs or tests
- Normalization is independent of `version` — works with v1 and v2

### 7. Audit Trail

`ClaimNormalizer` returns a `NormalizationReport` alongside transformed claims:

```python
@dataclass
class NormalizationReport:
    text_cleaned: int        # claims with name changes
    predicates_cleaned: int  # predicates fixed
    claims_decomposed: int   # parent claims that were split
    claims_produced: int     # total atomic claims from decomposition
    quant_backfilled: int    # assertions with quantitative context added
    llm_calls: int           # LLM fallback invocations
```

Logged via structlog at info level. Stored on the graph as `graph.graph["normalization_report"]` for downstream inspection.

---

## Module Change Summary

| Module | Change | Scope |
|--------|--------|-------|
| `normalize.py` | New file: `ClaimNormalizer`, `NormalizationReport`, text cleaning, predicate cleaning, decomposition, quant backfill | New module |
| `__init__.py` | Add `normalize` and `llm_decompose` params to `build_graph`, insert normalization passes | Backwards-compatible |
| `models.py` | No changes | — |
| `dedup.py` | No changes | — |
| `ingest.py` | No changes | — |

## Backwards Compatibility

- `normalize=False` (default) skips all normalization — zero behavior change
- All existing tests pass without modification
- Normalization is additive — it creates new/modified claim dicts, never mutates shared state
- Decomposed claims carry `_decomposed_from` audit trail back to original claim_id

## Testing Strategy

- Unit tests for text cleaning: parenthetical stripping, article removal, whitespace collapse, trailing descriptor removal, alias extraction
- Unit tests for predicate cleaning: whitespace fix, underscore normalization, tense normalization
- Unit tests for compound decomposition: conjunction split, comma list, prepositional compound, slash split, head noun distribution, ≤3 word skip, >6 word LLM flag
- Unit tests for quantitative backfill: concentration, timepoint, dose, fold_change extraction; unit normalization; no-overwrite rule
- Integration test: build graph with `normalize=True` from micro_v5, verify more cross-paper entity matches than `normalize=False`
- Integration test: `normalize=True, llm_decompose=False` for deterministic-only mode
- Regression test: `normalize=False` produces identical output to current code

## Future Work (Deferred)

- **Ontology grounding** — Map normalized entities to GO/CL/UBERON terms for semantic matching
- **Conditions deep cleaning** — Normalize cell_type, treatment strings within conditions
- **Evidence deduplication** — Merge duplicate evidence units across papers
