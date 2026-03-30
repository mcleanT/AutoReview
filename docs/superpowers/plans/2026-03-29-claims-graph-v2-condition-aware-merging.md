# Claims Graph v2: Condition-Aware Merging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make assertion merging condition-aware so that claims from different experimental contexts remain separate edges, improving contradiction detection quality.

**Architecture:** Add a condition signature (hash of organism + in_vitro + normalized model_system) to the merge key. Claims with the same (S,P,O) but different conditions become parallel edges in the MultiDiGraph. The NLI pipeline gains a contradiction taxonomy (within_context, cross_context, structural, nli_semantic) to classify contradiction pairs. All v1 code paths are preserved — `build_graph(version=2)` activates the new logic.

**Tech Stack:** Python 3.11+, Pydantic, NetworkX, rapidfuzz, structlog

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `autoreview/knowledge_graph/models.py` | Modify | Add `ConditionContext` model, new fields on `KGEdge` |
| `autoreview/knowledge_graph/dedup.py` | Modify | Add `ModelSystemRegistry`, `normalize_model_system`, `compute_condition_signature`, `merge_assertions_v2` |
| `autoreview/knowledge_graph/__init__.py` | Modify | Add `version` param to `build_graph`, branch merge strategy |
| `autoreview/knowledge_graph/graph.py` | Modify | Add `condition_signature` flat attr on edges |
| `autoreview/knowledge_graph/nli.py` | Modify | Add `contradiction_type` field, update `_build_claims` and pair selection |
| `tests/test_knowledge_graph/test_models.py` | Modify | Tests for `ConditionContext` |
| `tests/test_knowledge_graph/test_dedup.py` | Modify | Tests for condition signature, model system normalization, `merge_assertions_v2` |
| `tests/test_knowledge_graph/test_nli.py` | Modify | Tests for contradiction taxonomy |
| `tests/test_knowledge_graph/test_pipeline.py` | Modify | Integration tests for `build_graph(version=2)` |
| `tests/test_knowledge_graph/conftest.py` | Modify | Add v5-format fixtures with conditions |

---

### Task 1: Add ConditionContext model and new KGEdge fields

**Files:**
- Modify: `autoreview/knowledge_graph/models.py:121-152`
- Test: `tests/test_knowledge_graph/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_knowledge_graph/test_models.py`:

```python
class TestConditionContext:
    def test_default_construction(self):
        from autoreview.knowledge_graph.models import ConditionContext

        ctx = ConditionContext()
        assert ctx.organism is None
        assert ctx.model_system_class is None
        assert ctx.in_vitro is None
        assert ctx.cell_types == []
        assert ctx.treatments == []
        assert ctx.stages == []

    def test_full_construction(self):
        from autoreview.knowledge_graph.models import ConditionContext

        ctx = ConditionContext(
            organism="Mus musculus",
            model_system_class="mouse esc gastruloids",
            in_vitro=True,
            cell_types=["mESC", "E14Tg2a"],
            treatments=["10 ng/mL BMP4"],
            stages=["day 5"],
        )
        assert ctx.organism == "Mus musculus"
        assert ctx.cell_types == ["mESC", "E14Tg2a"]


class TestKGEdgeV2Fields:
    def test_condition_fields_default_none(self):
        from autoreview.knowledge_graph.models import (
            AssertionType,
            BetaPosterior,
            KGEdge,
        )

        edge = KGEdge(
            edge_id="test",
            subject_id="s1",
            object_id="o1",
            predicate="induces",
            direction="positive",
            assertion_type=AssertionType.mechanistic_causal,
            confidence=BetaPosterior(),
            evidence_links=[],
            source_assertions=[],
            publication_date=None,
        )
        assert edge.condition_signature is None
        assert edge.condition_context is None

    def test_condition_fields_populated(self):
        from autoreview.knowledge_graph.models import (
            AssertionType,
            BetaPosterior,
            ConditionContext,
            KGEdge,
        )

        ctx = ConditionContext(organism="Mus musculus", in_vitro=True)
        edge = KGEdge(
            edge_id="test",
            subject_id="s1",
            object_id="o1",
            predicate="induces",
            direction="positive",
            assertion_type=AssertionType.mechanistic_causal,
            confidence=BetaPosterior(),
            evidence_links=[],
            source_assertions=[],
            publication_date=None,
            condition_signature="abc123",
            condition_context=ctx,
        )
        assert edge.condition_signature == "abc123"
        assert edge.condition_context.organism == "Mus musculus"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_models.py::TestConditionContext -v`
Expected: FAIL — `ImportError: cannot import name 'ConditionContext'`

- [ ] **Step 3: Write minimal implementation**

In `autoreview/knowledge_graph/models.py`, add after the `QuantitativeContext` class (after line 127):

```python
class ConditionContext(AutoReviewModel):
    """Structured summary of the experimental context for a condition-partitioned edge."""

    organism: str | None = None
    model_system_class: str | None = None
    in_vitro: bool | None = None
    cell_types: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
```

Add the `Field` import at the top if not already present (it's already imported via `computed_field` — add `Field` to the pydantic import):

```python
from pydantic import Field, computed_field
```

Add two new fields to `KGEdge` after `quantitative_context` (after line 152):

```python
    # v2 condition-aware merging fields
    condition_signature: str | None = None
    condition_context: ConditionContext | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_models.py::TestConditionContext tests/test_knowledge_graph/test_models.py::TestKGEdgeV2Fields -v`
Expected: PASS

- [ ] **Step 5: Run all existing model tests to verify no regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_models.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add autoreview/knowledge_graph/models.py tests/test_knowledge_graph/test_models.py
git commit -m "feat(kg): add ConditionContext model and v2 fields on KGEdge"
```

---

### Task 2: Add condition signature computation and model system normalization

**Files:**
- Modify: `autoreview/knowledge_graph/dedup.py`
- Test: `tests/test_knowledge_graph/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_knowledge_graph/test_dedup.py`:

```python
class TestConditionSignature:
    """Test condition signature computation for v2 merging."""

    def test_deterministic(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature("Mus musculus", True, "mouse esc gastruloids")
        sig2 = compute_condition_signature("Mus musculus", True, "mouse esc gastruloids")
        assert sig1 == sig2

    def test_different_organism_different_signature(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_mouse = compute_condition_signature("Mus musculus", True, "esc gastruloids")
        sig_human = compute_condition_signature("Homo sapiens", True, "ipsc organoids")
        assert sig_mouse != sig_human

    def test_different_in_vitro_different_signature(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_vitro = compute_condition_signature("Mus musculus", True, "esc gastruloids")
        sig_vivo = compute_condition_signature("Mus musculus", False, "mouse embryo")
        assert sig_vitro != sig_vivo

    def test_case_insensitive(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature("Mus musculus", True, "Mouse ESC Gastruloids")
        sig2 = compute_condition_signature("mus musculus", True, "mouse esc gastruloids")
        assert sig1 == sig2

    def test_none_values_hash_consistently(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature(None, None, None)
        sig2 = compute_condition_signature(None, None, None)
        assert sig1 == sig2

    def test_none_vs_empty_different(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_none = compute_condition_signature(None, None, None)
        sig_mouse = compute_condition_signature("Mus musculus", True, "gastruloids")
        assert sig_none != sig_mouse


class TestModelSystemNormalization:
    """Test fuzzy bucketing of model system strings."""

    def test_identical_strings(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("mouse ESC gastruloids")
        assert cls1 == cls2

    def test_synonym_merge(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("mESC-derived gastruloids")
        assert cls1 == cls2

    def test_different_systems_separate(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("zebrafish embryo")
        assert cls1 != cls2

    def test_none_returns_empty(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        assert reg.normalize(None) == ""
        assert reg.normalize("") == ""

    def test_case_insensitive(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("Mouse Embryo")
        cls2 = reg.normalize("mouse embryo")
        assert cls1 == cls2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py::TestConditionSignature -v`
Expected: FAIL — `ImportError: cannot import name 'compute_condition_signature'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/dedup.py`, after the existing `merge_assertions` function (after line 591). Place the new functions in a new section:

```python
# ---------------------------------------------------------------------------
# v2: Condition-Aware Merging
# ---------------------------------------------------------------------------

_MODEL_SYSTEM_FUZZY_THRESHOLD = 80


def compute_condition_signature(
    organism: str | None,
    in_vitro: bool | None,
    model_system_class: str | None,
) -> str:
    """Deterministic hash of hard + soft partition fields for condition-aware merging.

    Args:
        organism: Species name (e.g., "Mus musculus").
        in_vitro: True for cell culture, False for in vivo, None for unknown.
        model_system_class: Normalized model system bucket string.

    Returns:
        12-character hex hash string.
    """
    key = (
        f"{(organism or '').lower().strip()}"
        f"|{in_vitro}"
        f"|{(model_system_class or '').lower().strip()}"
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]  # noqa: S324


class ModelSystemRegistry:
    """Fuzzy bucketing of model system strings.

    Uses union-find with rapidfuzz to cluster synonymous model system
    descriptions into canonical buckets. First string seen becomes the
    canonical representative; subsequent strings matching above threshold
    join that bucket.
    """

    def __init__(self, threshold: int = _MODEL_SYSTEM_FUZZY_THRESHOLD) -> None:
        self._threshold = threshold
        self._canonical: dict[str, str] = {}  # normalized → canonical bucket
        self._buckets: list[str] = []  # list of canonical representatives

    def normalize(self, raw: str | None) -> str:
        """Normalize a model system string to its canonical bucket.

        Args:
            raw: Raw model system string from extraction.

        Returns:
            Canonical bucket string (lowercased), or "" for None/empty.
        """
        if not raw or not raw.strip():
            return ""

        normalized = raw.lower().strip()
        if normalized in self._canonical:
            return self._canonical[normalized]

        # Check against existing buckets
        for bucket in self._buckets:
            ratio = fuzz.ratio(normalized, bucket)
            if ratio >= self._threshold:
                self._canonical[normalized] = bucket
                return bucket

        # New bucket
        self._buckets.append(normalized)
        self._canonical[normalized] = normalized
        return normalized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py::TestConditionSignature tests/test_knowledge_graph/test_dedup.py::TestModelSystemNormalization -v`
Expected: PASS

- [ ] **Step 5: Run all existing dedup tests for regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add autoreview/knowledge_graph/dedup.py tests/test_knowledge_graph/test_dedup.py
git commit -m "feat(kg): add condition signature computation and model system normalization"
```

---

### Task 3: Add merge_assertions_v2

**Files:**
- Modify: `autoreview/knowledge_graph/dedup.py`
- Test: `tests/test_knowledge_graph/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_knowledge_graph/test_dedup.py`:

```python
class TestMergeAssertionsV2:
    """Test condition-aware assertion merging."""

    def _make_assertion(
        self,
        subject_id: str = "ent1",
        object_id: str = "ent2",
        predicate: str = "induces",
        direction: str = "positive",
        draft_id: str = "a_001",
        paper_id: str = "p1",
        organism: str | None = "Mus musculus",
        model_system: str | None = "mouse ESC gastruloids",
        in_vitro: bool | None = True,
        conditions: dict | None = None,
    ) -> dict:
        return {
            "subject_id": subject_id,
            "object_id": object_id,
            "predicate": predicate,
            "direction": direction,
            "assertion_type": "mechanistic_causal",
            "draft_id": draft_id,
            "evidence_unit_ids": [f"e_{draft_id}"],
            "paper_id": paper_id,
            "publication_date": "2023-01-15",
            "organism": organism,
            "model_system": model_system,
            "in_vitro": in_vitro,
            "conditions": conditions or {},
            "certainty": "high",
            "section_source": "primary_empirical",
        }

    def test_same_spo_same_conditions_merge(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1"),
            self._make_assertion(draft_id="a_002", paper_id="p2"),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        assert len(result.assertions[0]["source_assertions"]) == 2

    def test_same_spo_different_organism_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001", paper_id="p1", organism="Mus musculus"
            ),
            self._make_assertion(
                draft_id="a_002", paper_id="p2", organism="Homo sapiens",
                model_system="human iPSC organoids",
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 2

    def test_same_spo_different_in_vitro_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001", paper_id="p1", in_vitro=True,
            ),
            self._make_assertion(
                draft_id="a_002", paper_id="p2", in_vitro=False,
                model_system="mouse embryo",
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 2

    def test_condition_signature_on_merged(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1"),
        ]
        result = merge_assertions_v2(assertions)
        assert result.assertions[0].get("condition_signature") is not None
        assert len(result.assertions[0]["condition_signature"]) == 12

    def test_condition_context_accumulated(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001", paper_id="p1",
                conditions={"cell_type": ["mESC"], "treatment": ["10 ng/mL BMP4"]},
            ),
            self._make_assertion(
                draft_id="a_002", paper_id="p2",
                conditions={"cell_type": ["E14Tg2a"], "treatment": ["3 µM CHIR99021"]},
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        ctx = result.assertions[0]["condition_context"]
        assert "mESC" in ctx["cell_types"]
        assert "E14Tg2a" in ctx["cell_types"]
        assert "10 ng/mL BMP4" in ctx["treatments"]
        assert "3 µM CHIR99021" in ctx["treatments"]

    def test_direction_conflict_within_context(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1", direction="positive"),
            self._make_assertion(draft_id="a_002", paper_id="p2", direction="negative"),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        assert result.assertions[0]["direction_conflict"] is True

    def test_v4_data_without_conditions(self):
        """v4 data (no conditions) merges like v1 — all-null signatures hash the same."""
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001", paper_id="p1",
                organism=None, model_system=None, in_vitro=None,
            ),
            self._make_assertion(
                draft_id="a_002", paper_id="p2",
                organism=None, model_system=None, in_vitro=None,
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1

    def test_v1_merge_still_works(self):
        """v1 merge_assertions is still importable and functional."""
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "induces",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py::TestMergeAssertionsV2::test_same_spo_same_conditions_merge -v`
Expected: FAIL — `ImportError: cannot import name 'merge_assertions_v2'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/dedup.py`, after the `ModelSystemRegistry` class:

```python
def merge_assertions_v2(assertions: list[dict[str, Any]]) -> MergeResult:
    """Merge assertions sharing the same (subject_id, predicate, object_id, condition_signature).

    Condition-aware variant of merge_assertions. Claims with the same (S,P,O)
    but different experimental contexts (organism, in_vitro, model_system) become
    separate edges. Scoring-modifier conditions (cell_type, treatment, stage)
    are accumulated across merged claims.

    Args:
        assertions: List of raw assertion dicts. Each must have subject_id,
            object_id, predicate, and may have organism, model_system, in_vitro,
            conditions.

    Returns:
        MergeResult with merged assertions (each with condition_signature and
        condition_context) and audit merge_log.
    """
    ms_registry = ModelSystemRegistry()

    # Compute condition signature for each assertion
    for assertion in assertions:
        ms_class = ms_registry.normalize(assertion.get("model_system"))
        assertion["_model_system_class"] = ms_class
        assertion["_condition_signature"] = compute_condition_signature(
            assertion.get("organism"),
            assertion.get("in_vitro"),
            ms_class,
        )

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for assertion in assertions:
        key = (
            assertion["subject_id"],
            assertion["predicate"],
            assertion["object_id"],
            assertion["_condition_signature"],
        )
        groups[key].append(assertion)

    result = MergeResult()

    for (subject_id, predicate, object_id, cond_sig), group in groups.items():
        draft_ids = [a["draft_id"] for a in group]
        paper_ids = [a["paper_id"] for a in group]

        # Accumulate evidence_unit_ids (flatten + deduplicate)
        all_evidence: list[str] = []
        for a in group:
            all_evidence.extend(a.get("evidence_unit_ids") or [])
        unique_evidence = list(dict.fromkeys(all_evidence))

        # Direction: unanimous → keep; conflict → None
        directions = {a.get("direction") for a in group}
        if len(directions) == 1:
            direction = directions.pop()
            direction_conflict = False
        else:
            direction = None
            direction_conflict = True

        # Assertion type: majority vote
        type_counts: dict[str, int] = defaultdict(int)
        for a in group:
            type_counts[a.get("assertion_type", "mechanistic_causal")] += 1
        assertion_type = max(type_counts, key=lambda t: type_counts[t])

        # Publication date: earliest
        dates = [a["publication_date"] for a in group if a.get("publication_date")]
        earliest_date = min(dates) if dates else None

        # Accumulate scoring-modifier conditions
        all_cell_types: list[str] = []
        all_treatments: list[str] = []
        all_stages: list[str] = []
        for a in group:
            conds = a.get("conditions") or {}
            for ct in conds.get("cell_type") or []:
                if ct not in all_cell_types:
                    all_cell_types.append(ct)
            for tr in conds.get("treatment") or []:
                if tr not in all_treatments:
                    all_treatments.append(tr)
            stage = conds.get("developmental_stage")
            if stage and stage not in all_stages:
                all_stages.append(stage)

        # Build condition context dict
        first = group[0]
        condition_context = {
            "organism": first.get("organism"),
            "model_system_class": first.get("_model_system_class", ""),
            "in_vitro": first.get("in_vitro"),
            "cell_types": all_cell_types,
            "treatments": all_treatments,
            "stages": all_stages,
        }

        merged = {
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "direction": direction,
            "direction_conflict": direction_conflict,
            "assertion_type": assertion_type,
            "evidence_unit_ids": unique_evidence,
            "source_assertions": draft_ids,
            "publication_date": earliest_date,
            "condition_signature": cond_sig,
            "condition_context": condition_context,
        }
        result.assertions.append(merged)

        if len(group) > 1:
            result.merge_log.append(
                {
                    "merged_draft_ids": draft_ids,
                    "papers": paper_ids,
                    "direction_conflict": direction_conflict,
                    "triple": (subject_id, predicate, object_id),
                    "condition_signature": cond_sig,
                }
            )

    logger.info(
        "assertion_merge_v2_complete",
        input_count=len(assertions),
        output_count=len(result.assertions),
        merges=len(result.merge_log),
    )
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py::TestMergeAssertionsV2 -v`
Expected: All PASS

- [ ] **Step 5: Run all existing dedup tests for regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_dedup.py -v`
Expected: All PASS (including original `TestAssertionMerging`)

- [ ] **Step 6: Commit**

```bash
git add autoreview/knowledge_graph/dedup.py tests/test_knowledge_graph/test_dedup.py
git commit -m "feat(kg): add merge_assertions_v2 with condition-aware merging"
```

---

### Task 4: Add v5-format fixtures to conftest

**Files:**
- Modify: `tests/test_knowledge_graph/conftest.py`

- [ ] **Step 1: Add v5-format fixtures**

Add to `tests/test_knowledge_graph/conftest.py`:

```python
@pytest.fixture
def sample_v5_claim() -> dict:
    """A v5-format claim with conditions, certainty, section_source."""
    return {
        "claim_id": "c_001",
        "natural_language": "BMP4 induces mesoderm differentiation in mouse ESC gastruloids",
        "subject": {
            "name": "BMP4",
            "type": "protein",
            "ontology_id": "UniProt:P21275",
        },
        "object": {
            "name": "mesoderm differentiation",
            "type": "biological_process",
            "ontology_id": "GO:0007498",
        },
        "predicate": "induces",
        "direction": "positive",
        "claim_type": "mechanistic_causal",
        "causal_type": "sufficient",
        "conditions": {
            "species": ["Mus musculus"],
            "cell_type": ["mESC"],
            "tissue": [],
            "treatment": ["10 ng/mL BMP4"],
            "developmental_stage": "day 5",
            "in_vitro": True,
        },
        "evidence_strength": "direct_experimental",
        "certainty": "high",
        "section_source": "primary_empirical",
        "source_doi": None,
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "quantitative_context": {"concentration": "10 ng/mL BMP4", "timepoint": "48h", "dose": None},
        "evidence_links": [{"evidence_id": "e_001", "direction": "supports"}],
    }


@pytest.fixture
def sample_v5_evidence() -> dict:
    """A v5-format evidence unit."""
    return {
        "evidence_id": "e_001",
        "description": "BMP4 treatment of mouse ESC gastruloids at day 3",
        "result_summary": "BMP4 at 10 ng/mL induced robust T/Brachyury expression by day 5",
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "perturbation": None,
        "readout": "T/Brachyury immunofluorescence",
        "result_direction": "positive",
        "effect_size": "3.5-fold increase",
        "p_value": "p < 0.001",
        "sample_size": "n=50",
        "key_figure": "Figure 2A",
        "approach": "cell_biology",
        "assay_types": ["immunofluorescence"],
        "evidence_strength": "direct_experimental",
    }


@pytest.fixture
def sample_v5_extraction_dir(tmp_path: Path, sample_v5_claim: dict, sample_v5_evidence: dict) -> Path:
    """Write v5-format extraction JSONs with varying conditions for v2 graph testing.

    Creates 3 papers:
    - Paper 1: BMP4 induces mesoderm (mouse, in vitro, gastruloids)
    - Paper 2: BMP4 induces mesoderm (mouse, in vitro, gastruloids) — same context, should merge
    - Paper 3: BMP4 induces mesoderm (human, in vitro, iPSC organoids) — different context, separate
    """
    for i, (paper_hash, organism, model_sys, cell_types) in enumerate([
        ("mouse_p1", "Mus musculus", "mouse ESC gastruloids", ["mESC"]),
        ("mouse_p2", "Mus musculus", "mouse ESC gastruloids", ["E14Tg2a"]),
        ("human_p3", "Homo sapiens", "human iPSC organoids", ["iPSC"]),
    ]):
        claim = json.loads(json.dumps(sample_v5_claim))
        evidence = json.loads(json.dumps(sample_v5_evidence))

        claim["claim_id"] = f"c_{i + 1:03d}"
        claim["organism"] = organism
        claim["model_system"] = model_sys
        claim["conditions"]["species"] = [organism]
        claim["conditions"]["cell_type"] = cell_types
        claim["evidence_links"] = [{"evidence_id": f"e_{i + 1:03d}", "direction": "supports"}]

        evidence["evidence_id"] = f"e_{i + 1:03d}"
        evidence["organism"] = organism
        evidence["model_system"] = model_sys

        data = {
            "doi": f"10.1234/paper-{i}",
            "title": f"Paper {i} on BMP4",
            "journal": "Nature",
            "publication_date": f"2023-0{i + 1}-15",
            "claims": [claim],
            "evidence": [evidence],
            "citation_contexts": [],
        }
        (tmp_path / f"{paper_hash}.json").write_text(json.dumps(data, indent=2))

    return tmp_path
```

- [ ] **Step 2: Verify fixture is usable**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_pipeline.py -v --collect-only`
Expected: Existing tests collected, no errors

- [ ] **Step 3: Commit**

```bash
git add tests/test_knowledge_graph/conftest.py
git commit -m "test(kg): add v5-format fixtures with conditions for v2 testing"
```

---

### Task 5: Update build_graph with version parameter

**Files:**
- Modify: `autoreview/knowledge_graph/__init__.py`
- Modify: `autoreview/knowledge_graph/graph.py`
- Test: `tests/test_knowledge_graph/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_knowledge_graph/test_pipeline.py`:

```python
class TestBuildGraphV2:
    def test_v2_pipeline_condition_aware(self, sample_v5_extraction_dir: Path):
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_v5_extraction_dir, version=2)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Papers 1 and 2 share same (S,P,O) + same conditions → merge to 1 edge
        # Paper 3 has same (S,P,O) but different conditions → separate edge
        # So expect 2 edges (not 1 as in v1)
        assert graph.number_of_edges() == 2

        # Every edge should have condition_signature
        for _u, _v, _k, data in graph.edges(keys=True, data=True):
            assert "condition_signature" in data
            assert data["condition_signature"] is not None

    def test_v1_pipeline_unchanged(self, sample_v5_extraction_dir: Path):
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_v5_extraction_dir, version=1)

        # v1 merges all 3 papers into 1 edge (same S,P,O regardless of conditions)
        assert graph.number_of_edges() == 1

    def test_v1_default(self, sample_extraction_dir: Path):
        """Default version=1 gives same result as before."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_extraction_dir)
        assert graph.number_of_edges() == 2  # same as existing test
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_pipeline.py::TestBuildGraphV2::test_v2_pipeline_condition_aware -v`
Expected: FAIL — `TypeError: build_graph() got an unexpected keyword argument 'version'`

- [ ] **Step 3: Update graph.py to emit condition_signature**

In `autoreview/knowledge_graph/graph.py`, modify `build_nx_graph` to add `condition_signature` to edge attrs. Change the edge-adding loop (lines 51-66):

```python
    # Add edges
    for edge in edges:
        graph.add_edge(
            edge.subject_id,
            edge.object_id,
            key=edge.edge_id,
            # Flat, GraphML-safe attributes
            predicate=edge.predicate,
            direction=edge.direction or "",
            assertion_type=str(edge.assertion_type),
            confidence_mean=edge.confidence.mean,
            evidence_count=len(edge.evidence_links),
            publication_date=edge.publication_date or "",
            source_assertions=",".join(edge.source_assertions),
            condition_signature=edge.condition_signature or "",
            # Rich object for programmatic access (not GraphML-safe)
            _kg_edge=edge,
        )
```

- [ ] **Step 4: Update __init__.py build_graph with version parameter**

In `autoreview/knowledge_graph/__init__.py`:

Update the existing dedup import block (lines 12-17) to add the new functions:

```python
from autoreview.knowledge_graph.dedup import (
    EntityRegistry,
    deduplicate_entities,
    merge_assertions,
    merge_assertions_v2,
    normalize_predicate,
)
```

Add `ConditionContext` to the models import block (lines 28-36):

```python
from autoreview.knowledge_graph.models import (
    AssertionType,
    BetaPosterior,
    Certainty,
    ConditionContext,
    KGEdge,
    KGEvidenceLink,
    QuantitativeContext,
    SectionSource,
)
```

Change the `build_graph` function signature (line 117-121):

```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
) -> nx.MultiDiGraph:
    """Full pipeline: ingest → dedup → graph → confidence.

    Args:
        extraction_dir: Path to a directory of ``*.json`` extraction files.
        use_mrf: If ``True``, run HL-MRF inference after Beta-Binomial scoring.
        mrf_config: Optional MRF config. Uses defaults when ``None``.
        version: Graph construction version. 1 = original merge on (S,P,O).
            2 = condition-aware merge on (S,P,O,condition_signature).

    Returns:
        A scored NetworkX MultiDiGraph ready for analysis and serialization.
    """
```

Change Step 5 (assertion merging, around lines 210-217) to branch on version:

```python
    # ------------------------------------------------------------------
    # Step 5: Assertion merging
    # ------------------------------------------------------------------
    if version >= 2:
        merge_result = merge_assertions_v2(normalized_assertions)
    else:
        merge_result = merge_assertions(normalized_assertions)
    log.info(
        "kg.pipeline.merge_done",
        merged_assertions=len(merge_result.assertions),
        merges=len(merge_result.merge_log),
        version=version,
    )
```

Change Step 5b (propagate v5 fields, around lines 219-249) to also handle v2 fields. After the existing v5 field propagation loop, add:

```python
    # v2: propagate condition_signature and condition_context from merge result
    if version >= 2:
        for merged in merge_result.assertions:
            # condition_signature and condition_context are already set by merge_assertions_v2
            pass  # fields are already on the merged dict
```

Change Step 6 (build KGEdge, around lines 256-338) to include condition fields. After the `quantitative_context` resolution block, add:

```python
        # v2 condition fields
        condition_signature = merged.get("condition_signature")
        raw_cc = merged.get("condition_context")
        condition_context = None
        if isinstance(raw_cc, dict):
            condition_context = ConditionContext(**raw_cc)
```

And add these to the KGEdge constructor (after `quantitative_context=quantitative_context,`):

```python
                condition_signature=condition_signature,
                condition_context=condition_context,
```

Update the `edge_id` computation for v2 to include condition_signature (around line 264):

```python
        # Deterministic edge_id from the canonical triple (+ condition for v2)
        if version >= 2 and merged.get("condition_signature"):
            edge_key = f"{subject_id}|{predicate}|{object_id}|{merged['condition_signature']}"
        else:
            edge_key = f"{subject_id}|{predicate}|{object_id}"
        edge_id = hashlib.sha256(edge_key.encode()).hexdigest()[:16]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_pipeline.py::TestBuildGraphV2 -v`
Expected: All PASS

- [ ] **Step 6: Run all pipeline tests for regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_pipeline.py -v`
Expected: All PASS (including original `TestBuildGraph`)

- [ ] **Step 7: Commit**

```bash
git add autoreview/knowledge_graph/__init__.py autoreview/knowledge_graph/graph.py tests/test_knowledge_graph/test_pipeline.py
git commit -m "feat(kg): add version param to build_graph, v2 uses condition-aware merging"
```

---

### Task 6: Update NLI with contradiction taxonomy

**Files:**
- Modify: `autoreview/knowledge_graph/nli.py`
- Test: `tests/test_knowledge_graph/test_nli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_knowledge_graph/test_nli.py`:

```python
class TestContradictionTaxonomy:
    """Test the v2 contradiction type classification."""

    def test_nli_pair_result_has_contradiction_type(self):
        from autoreview.knowledge_graph.nli import NLIPairResult

        result = NLIPairResult(
            claim_a_id="a",
            claim_b_id="b",
            p_contradiction=0.9,
            p_entailment=0.05,
            p_neutral=0.05,
            method="nli",
            shared_entities=["BMP4"],
            contradiction_type="within_context",
        )
        assert result.contradiction_type == "within_context"

    def test_nli_pair_result_contradiction_type_default_none(self):
        from autoreview.knowledge_graph.nli import NLIPairResult

        result = NLIPairResult(
            claim_a_id="a",
            claim_b_id="b",
            p_contradiction=0.9,
            p_entailment=0.05,
            p_neutral=0.05,
            method="nli",
            shared_entities=["BMP4"],
        )
        assert result.contradiction_type is None

    def test_classify_contradiction_type_within_context(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        # Same (S,P,O) + same condition signature
        claim_a = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "within_context"

    def test_classify_contradiction_type_cross_context(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        # Same (S,P,O) + different condition signature
        claim_a = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "def456"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "cross_context"

    def test_classify_contradiction_type_structural(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        # Same (S,O) + opposing predicates + same condition
        claim_a = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "inhibits",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "structural"

    def test_classify_contradiction_type_nli_semantic(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        # Different (S,P,O) — NLI-detected contradiction
        claim_a = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e3", "obj_id": "e4", "predicate": "regulates",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "nli_semantic"

    def test_classify_missing_condition_signature(self):
        """v1 graphs without condition_signature default to nli_semantic."""
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {},
        }
        claim_b = {
            "subj_id": "e1", "obj_id": "e2", "predicate": "induces",
            "edge_data": {},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "nli_semantic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_nli.py::TestContradictionTaxonomy::test_nli_pair_result_has_contradiction_type -v`
Expected: FAIL — validation error or missing field

- [ ] **Step 3: Add contradiction_type to NLIPairResult**

In `autoreview/knowledge_graph/nli.py`, add to `NLIPairResult` (after line 187):

```python
    contradiction_type: str | None = None
```

- [ ] **Step 4: Add _classify_contradiction_type function**

Add after the `_contexts_mismatch` function (after line 487):

```python
def _classify_contradiction_type(
    claim_a: dict[str, Any],
    claim_b: dict[str, Any],
) -> str:
    """Classify the type of contradiction between two claims.

    Uses condition signatures and predicate relationships to determine
    whether a contradiction is within-context, cross-context, structural,
    or NLI-semantic.

    Args:
        claim_a: Claim dict with subj_id, obj_id, predicate, edge_data.
        claim_b: Claim dict with subj_id, obj_id, predicate, edge_data.

    Returns:
        One of: "within_context", "cross_context", "structural", "nli_semantic".
    """
    same_subject = claim_a["subj_id"] == claim_b["subj_id"]
    same_object = claim_a["obj_id"] == claim_b["obj_id"]
    same_predicate = claim_a["predicate"] == claim_b["predicate"]

    sig_a = claim_a.get("edge_data", {}).get("condition_signature")
    sig_b = claim_b.get("edge_data", {}).get("condition_signature")
    same_condition = sig_a is not None and sig_b is not None and sig_a == sig_b

    if same_subject and same_object:
        # Check for structural opposition (opposing predicates)
        if not same_predicate and same_condition:
            opposition = _predicates_oppose(claim_a["predicate"], claim_b["predicate"])
            if opposition is not None:
                return "structural"

        if same_predicate:
            if same_condition:
                return "within_context"
            if sig_a is not None and sig_b is not None:
                return "cross_context"

    return "nli_semantic"
```

- [ ] **Step 5: Update _build_claims to attach condition_signature**

In `_build_claims` (around line 354), add `condition_signature` to the claim dict. After `"edge_data": data,` add:

```python
            "condition_signature": data.get("condition_signature"),
```

- [ ] **Step 6: Update classify_cross_claims to populate contradiction_type**

In `classify_cross_claims`, where `NLIPairResult` is constructed (around lines 825-837), add the contradiction_type:

```python
            contradiction_type = _classify_contradiction_type(claims[a_id], claims[b_id])
            pair_results.append(
                NLIPairResult(
                    claim_a_id=a_id,
                    claim_b_id=b_id,
                    p_contradiction=p_contra,
                    p_entailment=nli_out["p_entail"],
                    p_neutral=nli_out["p_neutral"],
                    method=method,
                    shared_entities=shared,
                    context_mismatch=nli_out.get("context_mismatch"),
                    original_p_contradiction=nli_out.get("original_p_contra"),
                    contradiction_type=contradiction_type,
                )
            )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_nli.py::TestContradictionTaxonomy -v`
Expected: All PASS

- [ ] **Step 8: Run all existing NLI tests for regression**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_nli.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add autoreview/knowledge_graph/nli.py tests/test_knowledge_graph/test_nli.py
git commit -m "feat(kg): add contradiction taxonomy to NLI pipeline"
```

---

### Task 7: Full regression test

**Files:**
- Test: `tests/test_knowledge_graph/`

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/ -v --tb=short`
Expected: All PASS

- [ ] **Step 2: If any failures, fix them**

Common issues:
- Import errors from changed signatures → fix imports
- Existing fixtures missing new required fields → new fields default to None, should be fine

- [ ] **Step 3: Commit any fixes**

```bash
git add -A tests/test_knowledge_graph/
git commit -m "fix(kg): resolve test regressions from v2 changes"
```

---

### Task 8: Smoke test with real micro_v5 data

> **Note:** Visualization updates (interactive.py — tooltips, parallel edge styling, contradiction taxonomy coloring) are deferred to a follow-up task after validating the data layer works end-to-end.

**Files:**
- No new files — script run in terminal

- [ ] **Step 1: Run v1 and v2 on micro_v5 corpus and compare**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -c "
from autoreview.knowledge_graph import build_graph
from pathlib import Path

extraction_dir = Path('Paper Extractor/KnowledgeGraph Extraction/gastruloid_run/micro_v5')

g1 = build_graph(extraction_dir, version=1)
print(f'v1: {g1.number_of_nodes()} nodes, {g1.number_of_edges()} edges')

g2 = build_graph(extraction_dir, version=2)
print(f'v2: {g2.number_of_nodes()} nodes, {g2.number_of_edges()} edges')

# v2 should have >= v1 edges (same or more due to condition partitioning)
assert g2.number_of_edges() >= g1.number_of_edges(), f'v2 has fewer edges: {g2.number_of_edges()} < {g1.number_of_edges()}'

# Check condition_signature populated on v2 edges
sigs = set()
for u, v, k, data in g2.edges(keys=True, data=True):
    sig = data.get('condition_signature', '')
    if sig:
        sigs.add(sig)
print(f'v2: {len(sigs)} unique condition signatures')
print('SMOKE TEST PASSED')
"
```

Expected: v2 has more edges than v1, multiple unique condition signatures.

- [ ] **Step 2: Commit**

No code changes — this is a validation step.
