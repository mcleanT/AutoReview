# Phase 2: HL-MRF Confidence Propagation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NLI-based contradiction detection with graph-structural detection and add non-local confidence propagation via Hinge-Loss Markov Random Fields (HL-MRFs), so that claim confidence accounts for transitive evidence chains and condition-aware contradiction resolution.

**Architecture:** Builds on existing KG modules (`dedup.py`, `confidence.py`, `nli.py`, `graph.py`). Adds four new modules to `autoreview/knowledge_graph/`: `predicate_algebra.py` (opposition + composition), `condition_compat.py` (experimental context similarity), `structural_contradictions.py` (graph-structural contradiction detector), and `hlmrf.py` (HL-MRF inference engine). The existing edge-local Beta-Binomial scoring becomes the unary potential initializer for the MRF. PSL-style rules are defined in-module and grounded against the NetworkX graph. Inference is a convex optimization solved with scipy L-BFGS-B — no Java/pslpython dependency.

**Tech Stack:** Python 3.11+, NetworkX, SciPy (L-BFGS-B optimizer), NumPy, Pydantic, existing autoreview infrastructure.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `autoreview/knowledge_graph/predicate_algebra.py` | Create | Opposition table, composition table, direction logic |
| `autoreview/knowledge_graph/condition_compat.py` | Create | Experimental context similarity scoring |
| `autoreview/knowledge_graph/structural_contradictions.py` | Create | Graph-structural contradiction detection |
| `autoreview/knowledge_graph/hlmrf.py` | Create | HL-MRF inference engine (rule grounding + optimization) |
| `autoreview/knowledge_graph/__init__.py` | Modify | Export new public API, wire MRF into pipeline |
| `autoreview/knowledge_graph/confidence.py` | Modify | Add `posteriors_to_unary_potentials()` bridge function |
| `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md` | Modify | Entity consistency + pathway completeness rules |
| `tests/test_knowledge_graph/test_predicate_algebra.py` | Create | Tests for predicate algebra |
| `tests/test_knowledge_graph/test_condition_compat.py` | Create | Tests for condition compatibility |
| `tests/test_knowledge_graph/test_structural_contradictions.py` | Create | Tests for structural contradiction detector |
| `tests/test_knowledge_graph/test_hlmrf.py` | Create | Tests for HL-MRF engine |

## Dependency Graph

```
Batch 1 (independent):
  Task 1: predicate_algebra.py + tests
  Task 2: condition_compat.py + tests
  Task 3: hlmrf.py + tests (pure math, no domain deps)
  Task 7: extraction prompt updates

Batch 2 (needs Tasks 1 + 2):
  Task 4: structural_contradictions.py + tests

Batch 3 (needs Tasks 1 + 2 + 4):
  Task 5: PSL rule definitions (in hlmrf.py rule_templates)

Batch 4 (needs all above):
  Task 6: pipeline integration + end-to-end test
```

---

### Task 1: Predicate Algebra Module

**Files:**
- Create: `autoreview/knowledge_graph/predicate_algebra.py`
- Test: `tests/test_knowledge_graph/test_predicate_algebra.py`

This module extracts and extends the predicate opposition families from `nli.py` (lines 35-118) and adds a new composition table for transitive inference. It becomes the single source of truth for predicate logic.

- [ ] **Step 1: Write failing tests for predicate opposition**

```python
# tests/test_knowledge_graph/test_predicate_algebra.py
from __future__ import annotations

import pytest

from autoreview.knowledge_graph.predicate_algebra import (
    are_opposing,
    compose_predicates,
    get_canonical_predicate,
    CompositionResult,
)


class TestPredicateOpposition:
    def test_activates_inhibits_opposing(self):
        assert are_opposing("induces", "inhibits") is True

    def test_same_predicate_not_opposing(self):
        assert are_opposing("induces", "induces") is False

    def test_required_vs_not_required(self):
        assert are_opposing("is_required_for", "is_not_required_for") is True

    def test_unrelated_predicates_not_opposing(self):
        assert are_opposing("induces", "localizes_to") is False

    def test_opposition_is_symmetric(self):
        assert are_opposing("inhibits", "induces") is True
        assert are_opposing("induces", "inhibits") is True

    def test_canonical_predicate_normalization(self):
        assert get_canonical_predicate("activates") == "induces"
        assert get_canonical_predicate("suppresses") == "inhibits"
        assert get_canonical_predicate("localizes_to") == "localizes_to"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_predicate_algebra.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement predicate opposition**

```python
# autoreview/knowledge_graph/predicate_algebra.py
"""Predicate algebra: opposition, composition, and canonicalization.

Single source of truth for predicate logic in the knowledge graph.
Extracted from nli.py opposition families and extended with composition rules.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonical predicate families — the normalized form after PredicateNormalizer
# ---------------------------------------------------------------------------

# After dedup.py normalization, all predicates map to these canonical forms.
# Opposition is defined between canonical predicates only.
OPPOSITION_PAIRS: list[tuple[str, str]] = [
    ("induces", "inhibits"),
    ("is_required_for", "is_not_required_for"),
    ("contains", "does_not_contain"),
    ("regulates", "does_not_regulate"),
    ("differentiates_into", "does_not_generate"),
    ("affects", "does_not_affect"),
    ("is_located_in", "is_not_located_in"),
    ("correlates_with", "does_not_correlate_with"),
    ("interacts_with", "does_not_interact_with"),
]

# Build bidirectional lookup for O(1) opposition checks
_OPPOSITION_MAP: dict[str, str] = {}
for a, b in OPPOSITION_PAIRS:
    _OPPOSITION_MAP[a] = b
    _OPPOSITION_MAP[b] = a

# Predicate synonym → canonical form (mirrors dedup.py PredicateNormalizer
# but adds opposition-relevant forms not covered there)
_CANONICAL_MAP: dict[str, str] = {
    "activates": "induces",
    "triggers": "induces",
    "initiates": "induces",
    "promotes": "induces",
    "stimulates": "induces",
    "upregulates": "induces",
    "suppresses": "inhibits",
    "blocks": "inhibits",
    "represses": "inhibits",
    "downregulates": "inhibits",
    "prevents": "inhibits",
    "attenuates": "inhibits",
    "is_necessary_for": "is_required_for",
    "is_essential_for": "is_required_for",
    "is_critical_for": "is_required_for",
    "localizes_to": "is_located_in",
    "is_expressed_in": "is_located_in",
    "binds_to": "interacts_with",
    "associates_with": "interacts_with",
}


def get_canonical_predicate(predicate: str) -> str:
    """Return the canonical form of a predicate."""
    return _CANONICAL_MAP.get(predicate, predicate)


def are_opposing(pred_a: str, pred_b: str) -> bool:
    """Check if two predicates are logically opposing.

    Canonicalizes both predicates before checking.
    """
    canon_a = get_canonical_predicate(pred_a)
    canon_b = get_canonical_predicate(pred_b)
    return _OPPOSITION_MAP.get(canon_a) == canon_b


# ---------------------------------------------------------------------------
# Predicate composition — transitive inference rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionResult:
    """Result of composing two predicates transitively."""

    composed_predicate: str
    strength: float  # 0.0-1.0, how confident the composition is
    rule: str  # human-readable rule description


# Composition table: (pred_AB, pred_BC) → CompositionResult for A→B→C
# Only meaningful compositions are listed. Absence = no composition.
COMPOSITION_TABLE: dict[tuple[str, str], CompositionResult] = {
    # Transitive activation
    ("induces", "induces"): CompositionResult(
        "induces", 0.7, "A activates B, B activates C → A promotes C"
    ),
    # Activation of inhibitor
    ("induces", "inhibits"): CompositionResult(
        "inhibits", 0.6, "A activates B, B inhibits C → A suppresses C"
    ),
    # Double negative
    ("inhibits", "inhibits"): CompositionResult(
        "induces", 0.5, "A inhibits B, B inhibits C → A promotes C (double negative)"
    ),
    # Inhibition of activator
    ("inhibits", "induces"): CompositionResult(
        "inhibits", 0.5, "A inhibits B, B activates C → A suppresses C"
    ),
    # Required-for chains
    ("is_required_for", "is_required_for"): CompositionResult(
        "is_required_for", 0.6, "A required for B, B required for C → A required for C"
    ),
    ("is_required_for", "induces"): CompositionResult(
        "induces", 0.4, "A required for B, B activates C → A promotes C (weak)"
    ),
    ("is_required_for", "inhibits"): CompositionResult(
        "inhibits", 0.4, "A required for B, B inhibits C → A suppresses C (weak)"
    ),
    # Regulatory chains
    ("regulates", "induces"): CompositionResult(
        "regulates", 0.3, "A regulates B, B activates C → A regulates C (direction unknown)"
    ),
    ("regulates", "inhibits"): CompositionResult(
        "regulates", 0.3, "A regulates B, B inhibits C → A regulates C (direction unknown)"
    ),
    # Localization chains
    ("induces", "is_located_in"): CompositionResult(
        "regulates", 0.3, "A activates B, B localizes to C → A influences C localization"
    ),
    # Differentiation chains
    ("induces", "differentiates_into"): CompositionResult(
        "induces", 0.5, "A activates B, B differentiates into C → A promotes C formation"
    ),
    ("is_required_for", "differentiates_into"): CompositionResult(
        "is_required_for", 0.5, "A required for B, B differentiates into C → A required for C"
    ),
    ("inhibits", "differentiates_into"): CompositionResult(
        "inhibits", 0.5, "A inhibits B, B differentiates into C → A blocks C formation"
    ),
}


def compose_predicates(pred_ab: str, pred_bc: str) -> CompositionResult | None:
    """Compose two predicates transitively: A→B (pred_ab) + B→C (pred_bc).

    Canonicalizes both predicates before lookup.

    Returns:
        CompositionResult if the composition is meaningful, None otherwise.
    """
    canon_ab = get_canonical_predicate(pred_ab)
    canon_bc = get_canonical_predicate(pred_bc)
    return COMPOSITION_TABLE.get((canon_ab, canon_bc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_graph/test_predicate_algebra.py -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for predicate composition**

```python
# Append to tests/test_knowledge_graph/test_predicate_algebra.py

class TestPredicateComposition:
    def test_activates_activates_composes(self):
        result = compose_predicates("induces", "induces")
        assert result is not None
        assert result.composed_predicate == "induces"
        assert result.strength == 0.7

    def test_activates_inhibits_composes(self):
        result = compose_predicates("induces", "inhibits")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_double_negative(self):
        result = compose_predicates("inhibits", "inhibits")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_unrelated_predicates_no_composition(self):
        result = compose_predicates("localizes_to", "correlates_with")
        assert result is None

    def test_synonym_predicates_compose(self):
        """Synonyms should canonicalize before lookup."""
        result = compose_predicates("activates", "suppresses")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_required_for_chain(self):
        result = compose_predicates("is_required_for", "is_required_for")
        assert result is not None
        assert result.composed_predicate == "is_required_for"

    def test_composition_strength_bounded(self):
        """All composition strengths should be in (0, 1]."""
        for (_, _), result in COMPOSITION_TABLE.items():
            assert 0.0 < result.strength <= 1.0
```

- [ ] **Step 6: Run tests to verify composition tests pass**

Run: `pytest tests/test_knowledge_graph/test_predicate_algebra.py -v`
Expected: PASS (implementation already covers these)

- [ ] **Step 7: Commit**

```bash
git add autoreview/knowledge_graph/predicate_algebra.py tests/test_knowledge_graph/test_predicate_algebra.py
git commit -m "feat(kg): add predicate algebra module with opposition and composition tables"
```

---

### Task 2: Condition Compatibility Module

**Files:**
- Create: `autoreview/knowledge_graph/condition_compat.py`
- Test: `tests/test_knowledge_graph/test_condition_compat.py`

Scores how comparable two claims' experimental contexts are. Returns a coupling strength [0, 1] that gates propagation in the MRF.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge_graph/test_condition_compat.py
from __future__ import annotations

import pytest

from autoreview.knowledge_graph.condition_compat import (
    condition_coupling,
    ConditionVector,
)


class TestConditionCoupling:
    def test_identical_conditions_full_coupling(self):
        a = ConditionVector(organism="Mus musculus", model_system="mESC gastruloids", in_vitro=True)
        b = ConditionVector(organism="Mus musculus", model_system="mESC gastruloids", in_vitro=True)
        assert condition_coupling(a, b) == 1.0

    def test_same_species_different_system(self):
        a = ConditionVector(organism="Mus musculus", model_system="mESC gastruloids", in_vitro=True)
        b = ConditionVector(organism="Mus musculus", model_system="mouse embryo", in_vitro=False)
        score = condition_coupling(a, b)
        assert 0.3 < score < 0.7  # partial coupling

    def test_different_species_low_coupling(self):
        a = ConditionVector(organism="Mus musculus", model_system="mESC gastruloids", in_vitro=True)
        b = ConditionVector(organism="Homo sapiens", model_system="hiPSC organoids", in_vitro=True)
        score = condition_coupling(a, b)
        assert score < 0.5

    def test_missing_conditions_moderate_coupling(self):
        """When conditions are unknown, assume moderate coupling."""
        a = ConditionVector(organism="Mus musculus", model_system=None, in_vitro=None)
        b = ConditionVector(organism="Mus musculus", model_system=None, in_vitro=None)
        score = condition_coupling(a, b)
        assert 0.5 <= score <= 1.0

    def test_completely_unknown_default(self):
        a = ConditionVector()
        b = ConditionVector()
        score = condition_coupling(a, b)
        assert score == 0.5  # agnostic default

    def test_coupling_is_symmetric(self):
        a = ConditionVector(organism="Mus musculus", model_system="gastruloids", in_vitro=True)
        b = ConditionVector(organism="Homo sapiens", model_system="organoids", in_vitro=True)
        assert condition_coupling(a, b) == condition_coupling(b, a)

    def test_from_kg_edge(self):
        """ConditionVector can be extracted from a KGEdge's fields."""
        edge_data = {
            "organism": "Mus musculus",
            "model_system": "mouse ESC gastruloids",
            "in_vitro": True,
            "conditions": {"species": ["Mus musculus"], "cell_type": ["mESC"]},
        }
        cv = ConditionVector.from_edge_data(edge_data)
        assert cv.organism == "Mus musculus"
        assert cv.in_vitro is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_condition_compat.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement condition compatibility**

```python
# autoreview/knowledge_graph/condition_compat.py
"""Condition compatibility scoring for knowledge graph claim comparison.

Computes a coupling strength [0, 1] between two claims based on their
experimental contexts. Used to gate propagation in the MRF: claims in
similar conditions should influence each other more strongly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConditionVector:
    """Structured experimental context extracted from a KG edge."""

    organism: str | None = None
    model_system: str | None = None
    in_vitro: bool | None = None
    cell_types: tuple[str, ...] = ()
    tissue: str | None = None
    developmental_stage: str | None = None

    @classmethod
    def from_edge_data(cls, data: dict[str, Any]) -> ConditionVector:
        """Extract a ConditionVector from a KGEdge's attribute dict."""
        conditions = data.get("conditions") or {}
        cell_types = conditions.get("cell_type") or []
        tissue_list = conditions.get("tissue") or []
        return cls(
            organism=data.get("organism"),
            model_system=data.get("model_system"),
            in_vitro=data.get("in_vitro"),
            cell_types=tuple(cell_types),
            tissue=tissue_list[0] if tissue_list else None,
            developmental_stage=conditions.get("developmental_stage"),
        )


# Species groupings for cross-species coupling
_SPECIES_GROUPS: dict[str, str] = {
    "Mus musculus": "rodent",
    "Rattus norvegicus": "rodent",
    "Homo sapiens": "primate",
    "Macaca fascicularis": "primate",
    "Danio rerio": "fish",
    "Xenopus laevis": "amphibian",
    "Xenopus tropicalis": "amphibian",
    "Gallus gallus": "bird",
    "Drosophila melanogaster": "insect",
    "Caenorhabditis elegans": "nematode",
}


def _species_score(org_a: str | None, org_b: str | None) -> float:
    """Score species compatibility: 1.0 same, 0.6 same group, 0.3 different, 0.5 unknown."""
    if org_a is None or org_b is None:
        return 0.5
    if org_a == org_b:
        return 1.0
    group_a = _SPECIES_GROUPS.get(org_a, org_a)
    group_b = _SPECIES_GROUPS.get(org_b, org_b)
    if group_a == group_b:
        return 0.6
    return 0.3


def _system_score(sys_a: str | None, sys_b: str | None) -> float:
    """Score model system similarity: 1.0 same, 0.5 different, 0.5 unknown."""
    if sys_a is None or sys_b is None:
        return 0.5
    if sys_a.lower().strip() == sys_b.lower().strip():
        return 1.0
    return 0.5


def _vitro_score(a: bool | None, b: bool | None) -> float:
    """Score in_vitro match: 1.0 same, 0.6 different, 0.5 unknown."""
    if a is None or b is None:
        return 0.5
    return 1.0 if a == b else 0.6


def condition_coupling(a: ConditionVector, b: ConditionVector) -> float:
    """Compute coupling strength between two experimental contexts.

    Returns a value in [0, 1] where:
    - 1.0 = identical conditions, full propagation
    - 0.5 = unknown/agnostic
    - 0.0 = completely incompatible (not currently produced)

    Weights: species (0.5), system (0.3), in_vitro (0.2).
    """
    sp = _species_score(a.organism, b.organism)
    sy = _system_score(a.model_system, b.model_system)
    iv = _vitro_score(a.in_vitro, b.in_vitro)

    return 0.5 * sp + 0.3 * sy + 0.2 * iv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_graph/test_condition_compat.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/condition_compat.py tests/test_knowledge_graph/test_condition_compat.py
git commit -m "feat(kg): add condition compatibility scoring for MRF propagation gating"
```

---

### Task 3: HL-MRF Inference Engine

**Files:**
- Create: `autoreview/knowledge_graph/hlmrf.py`
- Test: `tests/test_knowledge_graph/test_hlmrf.py`

Pure math module — no domain dependencies. Implements Hinge-Loss MRF inference as convex optimization. Variables are continuous [0, 1] truth values. Rules are weighted soft constraints. Inference minimizes the weighted sum of squared hinge losses using scipy L-BFGS-B.

- [ ] **Step 1: Write failing tests for rule grounding and inference**

```python
# tests/test_knowledge_graph/test_hlmrf.py
from __future__ import annotations

import pytest
import numpy as np

from autoreview.knowledge_graph.hlmrf import (
    HLMRFEngine,
    GroundRule,
    RuleTemplate,
)


class TestHLMRFEngine:
    def test_single_unary_rule(self):
        """A single unary rule should push the variable toward its target."""
        engine = HLMRFEngine()
        engine.add_variable("x", init=0.5)
        # Rule: x should be close to 1.0 (weight=10)
        engine.add_ground_rule(GroundRule(
            head_var="x",
            body_vars=[],
            body_coeffs=[],
            target=1.0,
            weight=10.0,
            rule_type="unary",
        ))
        result = engine.solve()
        assert result["x"] > 0.9

    def test_contradiction_rule_pushes_apart(self):
        """Two contradicting variables: if one is high, the other should be low."""
        engine = HLMRFEngine()
        engine.add_variable("a", init=0.8)
        engine.add_variable("b", init=0.8)
        # Unary: a should be high (strong evidence)
        engine.add_ground_rule(GroundRule(
            head_var="a", body_vars=[], body_coeffs=[],
            target=0.9, weight=10.0, rule_type="unary",
        ))
        # Unary: b should be high (moderate evidence)
        engine.add_ground_rule(GroundRule(
            head_var="b", body_vars=[], body_coeffs=[],
            target=0.7, weight=5.0, rule_type="unary",
        ))
        # Contradiction: a + b <= 1.0 (they can't both be true)
        engine.add_ground_rule(GroundRule(
            head_var="a", body_vars=["b"], body_coeffs=[1.0],
            target=1.0, weight=8.0, rule_type="contradiction",
        ))
        result = engine.solve()
        # a should stay high (stronger evidence), b should decrease
        assert result["a"] > result["b"]
        assert result["a"] + result["b"] < 1.8  # pushed apart from 0.9+0.7

    def test_composition_rule_boosts_inferred(self):
        """Transitive evidence: A→B high, B→C high → A→C should increase."""
        engine = HLMRFEngine()
        engine.add_variable("ab", init=0.9)
        engine.add_variable("bc", init=0.9)
        engine.add_variable("ac", init=0.3)  # sparse direct evidence
        # Strong unary for ab and bc
        engine.add_ground_rule(GroundRule(
            head_var="ab", body_vars=[], body_coeffs=[],
            target=0.9, weight=10.0, rule_type="unary",
        ))
        engine.add_ground_rule(GroundRule(
            head_var="bc", body_vars=[], body_coeffs=[],
            target=0.9, weight=10.0, rule_type="unary",
        ))
        # Weak unary for ac (sparse evidence)
        engine.add_ground_rule(GroundRule(
            head_var="ac", body_vars=[], body_coeffs=[],
            target=0.3, weight=2.0, rule_type="unary",
        ))
        # Composition: ab & bc → ac (if ab and bc are true, ac should be too)
        engine.add_ground_rule(GroundRule(
            head_var="ac", body_vars=["ab", "bc"], body_coeffs=[0.5, 0.5],
            target=0.7, weight=5.0, rule_type="composition",
        ))
        result = engine.solve()
        # ac should be boosted above its unary evidence
        assert result["ac"] > 0.4

    def test_all_values_bounded_0_1(self):
        """All solution values must be in [0, 1]."""
        engine = HLMRFEngine()
        for name in ["x", "y", "z"]:
            engine.add_variable(name, init=0.5)
            engine.add_ground_rule(GroundRule(
                head_var=name, body_vars=[], body_coeffs=[],
                target=1.5, weight=10.0, rule_type="unary",  # target > 1
            ))
        result = engine.solve()
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_empty_engine_returns_inits(self):
        """No rules → solution equals initialization."""
        engine = HLMRFEngine()
        engine.add_variable("x", init=0.42)
        result = engine.solve()
        assert abs(result["x"] - 0.42) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_hlmrf.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement HL-MRF engine**

```python
# autoreview/knowledge_graph/hlmrf.py
"""Hinge-Loss Markov Random Field inference engine.

Implements PSL-style inference as convex optimization over continuous
truth values in [0, 1]. No Java/pslpython dependency — uses scipy
L-BFGS-B for bounded optimization.

The objective is a weighted sum of squared hinge losses:
    minimize Σ_r  w_r * max(0, body_r - head_r)^2

where each ground rule defines a soft constraint on variable truth values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from scipy.optimize import minimize

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GroundRule:
    """A grounded (instantiated) rule in the HL-MRF.

    For unary rules: target is the evidence-based truth value.
    For contradiction rules: head + body should not exceed target.
    For composition rules: if body vars are high, head should be high.
    """

    head_var: str
    body_vars: list[str]
    body_coeffs: list[float]
    target: float
    weight: float
    rule_type: str  # "unary", "contradiction", "composition"


@dataclass
class RuleTemplate:
    """A template for generating ground rules from graph structure."""

    name: str
    weight: float
    rule_type: str
    description: str = ""


class HLMRFEngine:
    """HL-MRF inference engine using scipy L-BFGS-B optimization."""

    def __init__(self, max_iter: int = 200, tol: float = 1e-6) -> None:
        self._variables: dict[str, float] = {}  # name → init value
        self._var_index: dict[str, int] = {}  # name → index in array
        self._ground_rules: list[GroundRule] = []
        self._max_iter = max_iter
        self._tol = tol

    def add_variable(self, name: str, init: float = 0.5) -> None:
        """Register a variable with an initial truth value."""
        self._variables[name] = np.clip(init, 0.0, 1.0)
        self._var_index[name] = len(self._var_index)

    def add_ground_rule(self, rule: GroundRule) -> None:
        """Add a grounded rule to the model."""
        self._ground_rules.append(rule)

    @property
    def n_variables(self) -> int:
        return len(self._variables)

    @property
    def n_rules(self) -> int:
        return len(self._ground_rules)

    def _objective(self, x: np.ndarray) -> float:
        """Compute the weighted sum of squared hinge losses."""
        total = 0.0
        for rule in self._ground_rules:
            head_idx = self._var_index.get(rule.head_var)
            if head_idx is None:
                continue

            if rule.rule_type == "unary":
                # Squared distance to target
                diff = x[head_idx] - rule.target
                total += rule.weight * diff * diff

            elif rule.rule_type == "contradiction":
                # Hinge: penalize when head + Σ(coeff_i * body_i) > target
                body_sum = sum(
                    c * x[self._var_index[v]]
                    for v, c in zip(rule.body_vars, rule.body_coeffs)
                    if v in self._var_index
                )
                violation = max(0.0, x[head_idx] + body_sum - rule.target)
                total += rule.weight * violation * violation

            elif rule.rule_type == "composition":
                # If body vars are high, head should approach target
                body_vals = [
                    x[self._var_index[v]]
                    for v in rule.body_vars
                    if v in self._var_index
                ]
                if not body_vals:
                    continue
                body_strength = min(body_vals)  # AND semantics: weakest link
                # Hinge: penalize when head < target * body_strength
                expected = rule.target * body_strength
                shortfall = max(0.0, expected - x[head_idx])
                total += rule.weight * shortfall * shortfall

        return total

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of the objective."""
        grad = np.zeros_like(x)

        for rule in self._ground_rules:
            head_idx = self._var_index.get(rule.head_var)
            if head_idx is None:
                continue

            if rule.rule_type == "unary":
                diff = x[head_idx] - rule.target
                grad[head_idx] += 2.0 * rule.weight * diff

            elif rule.rule_type == "contradiction":
                body_sum = sum(
                    c * x[self._var_index[v]]
                    for v, c in zip(rule.body_vars, rule.body_coeffs)
                    if v in self._var_index
                )
                violation = x[head_idx] + body_sum - rule.target
                if violation > 0:
                    grad[head_idx] += 2.0 * rule.weight * violation
                    for v, c in zip(rule.body_vars, rule.body_coeffs):
                        if v in self._var_index:
                            grad[self._var_index[v]] += 2.0 * rule.weight * violation * c

            elif rule.rule_type == "composition":
                body_vals = [
                    x[self._var_index[v]]
                    for v in rule.body_vars
                    if v in self._var_index
                ]
                if not body_vals:
                    continue
                body_strength = min(body_vals)
                expected = rule.target * body_strength
                shortfall = expected - x[head_idx]
                if shortfall > 0:
                    grad[head_idx] -= 2.0 * rule.weight * shortfall

        return grad

    def solve(self) -> dict[str, float]:
        """Run HL-MRF inference and return posterior truth values.

        Returns:
            Mapping of variable name → truth value in [0, 1].
        """
        if not self._variables:
            return {}

        if not self._ground_rules:
            return dict(self._variables)

        n = len(self._variables)
        x0 = np.array([self._variables[name] for name in sorted(
            self._variables, key=lambda k: self._var_index[k]
        )])
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            self._objective,
            x0,
            jac=self._gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self._max_iter, "ftol": self._tol},
        )

        if not result.success:
            log.warning("hlmrf.convergence_warning", message=result.message)

        solution = {}
        for name, idx in self._var_index.items():
            solution[name] = float(np.clip(result.x[idx], 0.0, 1.0))

        log.info(
            "hlmrf.solved",
            n_variables=n,
            n_rules=len(self._ground_rules),
            iterations=result.nit,
            converged=result.success,
        )
        return solution
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_graph/test_hlmrf.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/hlmrf.py tests/test_knowledge_graph/test_hlmrf.py
git commit -m "feat(kg): add HL-MRF inference engine with scipy L-BFGS-B optimization"
```

---

### Task 4: Structural Contradiction Detector

**Depends on:** Task 1 (predicate_algebra), Task 2 (condition_compat)

**Files:**
- Create: `autoreview/knowledge_graph/structural_contradictions.py`
- Test: `tests/test_knowledge_graph/test_structural_contradictions.py`

Detects contradictions purely from graph structure — no NLI model needed. Uses predicate opposition + direction comparison + condition disambiguation.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge_graph/test_structural_contradictions.py
from __future__ import annotations

import pytest
import networkx as nx

from autoreview.knowledge_graph.structural_contradictions import (
    detect_contradictions,
    ContradictionPair,
    ContradictionType,
)


def _make_graph_with_edges(edges: list[dict]) -> nx.MultiDiGraph:
    """Helper: build a minimal graph from edge specs."""
    G = nx.MultiDiGraph()
    for e in edges:
        G.add_edge(
            e["subject"], e["object"],
            predicate=e["predicate"],
            direction=e.get("direction", "positive"),
            organism=e.get("organism"),
            model_system=e.get("model_system"),
            in_vitro=e.get("in_vitro"),
            conditions=e.get("conditions", {}),
            edge_id=e.get("edge_id", f"{e['subject']}_{e['predicate']}_{e['object']}"),
        )
    return G


class TestStructuralContradictions:
    def test_opposing_predicates_detected(self):
        """Same S/O, opposing predicates → contradiction."""
        G = _make_graph_with_edges([
            {"subject": "A", "object": "B", "predicate": "induces",
             "organism": "Mus musculus", "model_system": "mESC", "in_vitro": True},
            {"subject": "A", "object": "B", "predicate": "inhibits",
             "organism": "Mus musculus", "model_system": "mESC", "in_vitro": True},
        ])
        pairs = detect_contradictions(G)
        assert len(pairs) == 1
        assert pairs[0].contradiction_type == ContradictionType.PREDICATE_OPPOSITION

    def test_same_predicate_opposite_direction(self):
        """Same S/P/O, opposite direction → contradiction."""
        G = _make_graph_with_edges([
            {"subject": "A", "object": "B", "predicate": "induces", "direction": "positive",
             "organism": "Mus musculus", "model_system": "mESC", "in_vitro": True},
            {"subject": "A", "object": "B", "predicate": "induces", "direction": "negative",
             "organism": "Mus musculus", "model_system": "mESC", "in_vitro": True},
        ])
        pairs = detect_contradictions(G)
        assert len(pairs) == 1
        assert pairs[0].contradiction_type == ContradictionType.DIRECTION_CONFLICT

    def test_different_conditions_flagged_as_boundary(self):
        """Same S/O, opposing predicates, different conditions → boundary."""
        G = _make_graph_with_edges([
            {"subject": "A", "object": "B", "predicate": "induces",
             "organism": "Mus musculus", "model_system": "mESC", "in_vitro": True},
            {"subject": "A", "object": "B", "predicate": "inhibits",
             "organism": "Homo sapiens", "model_system": "hiPSC", "in_vitro": True},
        ])
        pairs = detect_contradictions(G)
        assert len(pairs) == 1
        assert pairs[0].contradiction_type == ContradictionType.BOUNDARY_CONDITION

    def test_no_contradiction_different_objects(self):
        """Different objects → no contradiction."""
        G = _make_graph_with_edges([
            {"subject": "A", "object": "B", "predicate": "induces"},
            {"subject": "A", "object": "C", "predicate": "inhibits"},
        ])
        pairs = detect_contradictions(G)
        assert len(pairs) == 0

    def test_no_contradiction_compatible_predicates(self):
        """Non-opposing predicates on same S/O → no contradiction."""
        G = _make_graph_with_edges([
            {"subject": "A", "object": "B", "predicate": "induces"},
            {"subject": "A", "object": "B", "predicate": "is_located_in"},
        ])
        pairs = detect_contradictions(G)
        assert len(pairs) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_structural_contradictions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement structural contradiction detector**

```python
# autoreview/knowledge_graph/structural_contradictions.py
"""Graph-structural contradiction detection.

Replaces NLI-based detection with deterministic graph operations:
1. Find edge pairs sharing the same (subject, object)
2. Check predicate opposition or direction conflict
3. Compare conditions to classify as real contradiction vs. boundary

No ML model required.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.condition_compat import (
    ConditionVector,
    condition_coupling,
)
from autoreview.knowledge_graph.predicate_algebra import are_opposing

log = structlog.get_logger(__name__)

# Condition coupling below this threshold → boundary, not contradiction
_BOUNDARY_THRESHOLD = 0.6


class ContradictionType(StrEnum):
    PREDICATE_OPPOSITION = "predicate_opposition"
    DIRECTION_CONFLICT = "direction_conflict"
    BOUNDARY_CONDITION = "boundary_condition"


@dataclass
class ContradictionPair:
    """A detected contradiction between two graph edges."""

    edge_a_id: str
    edge_b_id: str
    subject: str
    object: str
    predicate_a: str
    predicate_b: str
    direction_a: str | None
    direction_b: str | None
    contradiction_type: ContradictionType
    condition_coupling: float
    condition_a: ConditionVector
    condition_b: ConditionVector


def detect_contradictions(
    graph: nx.MultiDiGraph,
    boundary_threshold: float = _BOUNDARY_THRESHOLD,
) -> list[ContradictionPair]:
    """Detect contradictions from graph structure.

    Scans all edge pairs sharing the same (subject, object) node pair
    and checks for predicate opposition or direction conflict.

    Args:
        graph: Knowledge graph MultiDiGraph.
        boundary_threshold: Condition coupling below this = boundary, not contradiction.

    Returns:
        List of detected ContradictionPair instances.
    """
    # Group edges by (subject, object) pair
    edge_groups: dict[tuple[str, str], list[tuple[str, str, int, dict[str, Any]]]] = defaultdict(list)
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_groups[(u, v)].append((u, v, key, data))

    pairs: list[ContradictionPair] = []

    for (subj, obj), edges in edge_groups.items():
        # Compare all pairs within the group
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                _, _, _, data_a = edges[i]
                _, _, _, data_b = edges[j]

                pred_a = data_a.get("predicate", "")
                pred_b = data_b.get("predicate", "")
                dir_a = data_a.get("direction")
                dir_b = data_b.get("direction")

                # Check 1: predicate opposition
                is_opposing = are_opposing(pred_a, pred_b)

                # Check 2: same predicate, opposite direction
                is_direction_conflict = (
                    pred_a == pred_b
                    and dir_a is not None
                    and dir_b is not None
                    and dir_a != dir_b
                )

                if not is_opposing and not is_direction_conflict:
                    continue

                # Compute condition coupling to classify
                cond_a = ConditionVector.from_edge_data(data_a)
                cond_b = ConditionVector.from_edge_data(data_b)
                coupling = condition_coupling(cond_a, cond_b)

                if is_opposing or is_direction_conflict:
                    if coupling < boundary_threshold:
                        ctype = ContradictionType.BOUNDARY_CONDITION
                    elif is_opposing:
                        ctype = ContradictionType.PREDICATE_OPPOSITION
                    else:
                        ctype = ContradictionType.DIRECTION_CONFLICT

                    pairs.append(ContradictionPair(
                        edge_a_id=data_a.get("edge_id", ""),
                        edge_b_id=data_b.get("edge_id", ""),
                        subject=subj,
                        object=obj,
                        predicate_a=pred_a,
                        predicate_b=pred_b,
                        direction_a=dir_a,
                        direction_b=dir_b,
                        contradiction_type=ctype,
                        condition_coupling=coupling,
                        condition_a=cond_a,
                        condition_b=cond_b,
                    ))

    log.info(
        "structural_contradictions.detected",
        total=len(pairs),
        real_contradictions=sum(1 for p in pairs if p.contradiction_type != ContradictionType.BOUNDARY_CONDITION),
        boundaries=sum(1 for p in pairs if p.contradiction_type == ContradictionType.BOUNDARY_CONDITION),
    )
    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_graph/test_structural_contradictions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/structural_contradictions.py tests/test_knowledge_graph/test_structural_contradictions.py
git commit -m "feat(kg): add structural contradiction detector replacing NLI"
```

---

### Task 5: PSL Rule Grounding (wire domain into HL-MRF)

**Depends on:** Tasks 1, 2, 3, 4

**Files:**
- Create: `autoreview/knowledge_graph/mrf_scoring.py`
- Test: `tests/test_knowledge_graph/test_mrf_scoring.py`

This module bridges the domain (graph, contradictions, compositions) with the generic HL-MRF engine. It grounds PSL-style rules against the actual graph and runs inference.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge_graph/test_mrf_scoring.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
import networkx as nx

from autoreview.knowledge_graph.mrf_scoring import (
    MRFConfig,
    score_graph_mrf,
    MRFResult,
)


def _make_scored_graph() -> nx.MultiDiGraph:
    """Build a minimal graph with Beta-Binomial confidence already scored."""
    G = nx.MultiDiGraph()
    # Entity nodes
    G.add_node("e1", canonical_name="BMP4", entity_type="protein")
    G.add_node("e2", canonical_name="SMAD1", entity_type="protein")
    G.add_node("e3", canonical_name="mesoderm", entity_type="biological_process")

    # Edge: BMP4 → SMAD1 (strong evidence)
    G.add_edge("e1", "e2",
        predicate="induces", direction="positive",
        confidence_mean=0.85, edge_id="edge_ab",
        organism="Mus musculus", model_system="mESC", in_vitro=True,
        conditions={},
    )
    # Edge: SMAD1 → mesoderm (strong evidence)
    G.add_edge("e2", "e3",
        predicate="induces", direction="positive",
        confidence_mean=0.80, edge_id="edge_bc",
        organism="Mus musculus", model_system="mESC", in_vitro=True,
        conditions={},
    )
    # Edge: BMP4 → mesoderm (sparse direct evidence)
    G.add_edge("e1", "e3",
        predicate="induces", direction="positive",
        confidence_mean=0.35, edge_id="edge_ac",
        organism="Mus musculus", model_system="mESC", in_vitro=True,
        conditions={},
    )
    return G


class TestMRFScoring:
    def test_transitive_boost(self):
        """BMP4→SMAD1 + SMAD1→mesoderm should boost BMP4→mesoderm."""
        G = _make_scored_graph()
        result = score_graph_mrf(G)
        # The transitive edge should be boosted above its edge-local 0.35
        ac_score = result.posteriors.get("edge_ac", 0.0)
        assert ac_score > 0.4, f"Expected boost above 0.4, got {ac_score}"

    def test_strong_edges_stay_strong(self):
        """Well-evidenced edges should not decrease much."""
        G = _make_scored_graph()
        result = score_graph_mrf(G)
        ab_score = result.posteriors.get("edge_ab", 0.0)
        assert ab_score > 0.75

    def test_result_has_all_edges(self):
        """Result should contain a posterior for every edge."""
        G = _make_scored_graph()
        result = score_graph_mrf(G)
        assert set(result.posteriors.keys()) == {"edge_ab", "edge_bc", "edge_ac"}

    def test_empty_graph(self):
        """Empty graph should return empty result."""
        G = nx.MultiDiGraph()
        result = score_graph_mrf(G)
        assert result.posteriors == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_mrf_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MRF scoring**

```python
# autoreview/knowledge_graph/mrf_scoring.py
"""MRF-based confidence scoring for the knowledge graph.

Bridges domain knowledge (graph structure, contradictions, predicate
composition, condition compatibility) with the HL-MRF inference engine.

Grounds three types of rules against the graph:
1. Unary (evidence) rules: push each edge toward its Beta-Binomial posterior
2. Contradiction rules: opposing claims should not both be highly confident
3. Composition rules: transitive evidence chains boost inferred claims
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.condition_compat import (
    ConditionVector,
    condition_coupling,
)
from autoreview.knowledge_graph.hlmrf import GroundRule, HLMRFEngine
from autoreview.knowledge_graph.predicate_algebra import (
    are_opposing,
    compose_predicates,
)
from autoreview.knowledge_graph.structural_contradictions import (
    ContradictionType,
    detect_contradictions,
)
from autoreview.models.base import AutoReviewModel

log = structlog.get_logger(__name__)


class MRFConfig(AutoReviewModel):
    """Configuration for MRF-based scoring."""

    # Rule weights
    evidence_weight: float = 10.0
    contradiction_weight: float = 8.0
    composition_weight: float = 5.0

    # HL-MRF solver settings
    max_iter: int = 200
    tol: float = 1e-6

    # Contradiction detection
    boundary_threshold: float = 0.6


@dataclass
class MRFResult:
    """Result of MRF scoring."""

    posteriors: dict[str, float] = field(default_factory=dict)
    n_variables: int = 0
    n_rules: int = 0
    n_contradictions: int = 0
    n_compositions: int = 0
    converged: bool = True


def score_graph_mrf(
    graph: nx.MultiDiGraph,
    config: MRFConfig | None = None,
) -> MRFResult:
    """Score all edges in the graph using HL-MRF inference.

    Steps:
    1. Create one variable per edge (initialized to Beta-Binomial posterior mean)
    2. Add unary rules from edge-local confidence
    3. Detect structural contradictions and add contradiction rules
    4. Find transitive chains and add composition rules
    5. Solve and return posteriors

    Args:
        graph: NetworkX MultiDiGraph with confidence_mean already computed.
        config: MRF configuration (weights, thresholds).

    Returns:
        MRFResult with posterior truth values per edge.
    """
    if config is None:
        config = MRFConfig()

    engine = HLMRFEngine(max_iter=config.max_iter, tol=config.tol)

    # Collect all edges with their IDs
    edge_data: dict[str, dict[str, Any]] = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_id = data.get("edge_id")
        if edge_id is None:
            continue
        edge_data[edge_id] = {**data, "_u": u, "_v": v, "_key": key}

    if not edge_data:
        return MRFResult()

    # Step 1: Variables from Beta-Binomial posteriors
    for edge_id, data in edge_data.items():
        init = data.get("confidence_mean", 0.5)
        engine.add_variable(edge_id, init=init)

    # Step 2: Unary evidence rules
    for edge_id, data in edge_data.items():
        target = data.get("confidence_mean", 0.5)
        engine.add_ground_rule(GroundRule(
            head_var=edge_id,
            body_vars=[],
            body_coeffs=[],
            target=target,
            weight=config.evidence_weight,
            rule_type="unary",
        ))

    # Step 3: Contradiction rules
    contradictions = detect_contradictions(graph, boundary_threshold=config.boundary_threshold)
    n_contradiction_rules = 0
    for pair in contradictions:
        if pair.contradiction_type == ContradictionType.BOUNDARY_CONDITION:
            continue  # boundaries are informative, not penalized
        if pair.edge_a_id not in edge_data or pair.edge_b_id not in edge_data:
            continue
        # Contradiction: a + b should not exceed 1.0
        weight = config.contradiction_weight * pair.condition_coupling
        engine.add_ground_rule(GroundRule(
            head_var=pair.edge_a_id,
            body_vars=[pair.edge_b_id],
            body_coeffs=[1.0],
            target=1.0,
            weight=weight,
            rule_type="contradiction",
        ))
        n_contradiction_rules += 1

    # Step 4: Composition rules (find A→B→C chains)
    # Build adjacency: node → list of (edge_id, target_node, predicate, data)
    outgoing: dict[str, list[tuple[str, str, str, dict]]] = {}
    for edge_id, data in edge_data.items():
        u = data["_u"]
        outgoing.setdefault(u, []).append((edge_id, data["_v"], data.get("predicate", ""), data))

    n_composition_rules = 0
    for mid_node, ab_edges in outgoing.items():
        bc_edges = outgoing.get(mid_node)  # This is wrong — need edges FROM mid_node's targets
        # Actually: ab_edges are edges INTO mid_node? No — outgoing[u] = edges FROM u.
        # For A→B→C: need edges FROM A (gives A→B), then edges FROM B (gives B→C)
        pass

    # Corrected: iterate over A→B edges, then look for B→C edges
    for a_node in set(data["_u"] for data in edge_data.values()):
        a_edges = outgoing.get(a_node, [])
        for ab_id, b_node, pred_ab, data_ab in a_edges:
            b_edges = outgoing.get(b_node, [])
            for bc_id, c_node, pred_bc, data_bc in b_edges:
                if a_node == c_node:
                    continue  # skip self-loops

                comp = compose_predicates(pred_ab, pred_bc)
                if comp is None:
                    continue

                # Look for an existing A→C edge to boost
                ac_edges = [
                    (eid, d) for eid, d in edge_data.items()
                    if d["_u"] == a_node and d["_v"] == c_node
                ]

                if not ac_edges:
                    continue  # only boost existing edges (don't hallucinate)

                # Condition coupling gates the composition strength
                cond_ab = ConditionVector.from_edge_data(data_ab)
                cond_bc = ConditionVector.from_edge_data(data_bc)
                coupling = condition_coupling(cond_ab, cond_bc)

                for ac_id, _ in ac_edges:
                    weight = config.composition_weight * comp.strength * coupling
                    engine.add_ground_rule(GroundRule(
                        head_var=ac_id,
                        body_vars=[ab_id, bc_id],
                        body_coeffs=[0.5, 0.5],
                        target=comp.strength,
                        weight=weight,
                        rule_type="composition",
                    ))
                    n_composition_rules += 1

    # Step 5: Solve
    posteriors = engine.solve()

    log.info(
        "mrf_scoring.complete",
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
        n_contradictions=n_contradiction_rules,
        n_compositions=n_composition_rules,
    )

    return MRFResult(
        posteriors=posteriors,
        n_variables=engine.n_variables,
        n_rules=engine.n_rules,
        n_contradictions=n_contradiction_rules,
        n_compositions=n_composition_rules,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_graph/test_mrf_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/mrf_scoring.py tests/test_knowledge_graph/test_mrf_scoring.py
git commit -m "feat(kg): add MRF scoring module bridging domain rules to HL-MRF engine"
```

---

### Task 6: Pipeline Integration

**Depends on:** Tasks 1-5

**Files:**
- Modify: `autoreview/knowledge_graph/__init__.py`
- Modify: `autoreview/knowledge_graph/confidence.py` (add bridge function)
- Test: `tests/test_knowledge_graph/test_pipeline.py` (add MRF integration test)

Wire MRF scoring into the `build_graph` pipeline as an optional step after Beta-Binomial scoring.

- [ ] **Step 1: Write failing integration test**

```python
# Append to tests/test_knowledge_graph/test_pipeline.py (or create new test file)
# tests/test_knowledge_graph/test_mrf_integration.py

from __future__ import annotations

from pathlib import Path

import pytest

from autoreview.knowledge_graph import build_graph


class TestMRFIntegration:
    def test_build_graph_with_mrf(self, sample_extraction_dir: Path):
        """Full pipeline with MRF scoring should complete without error."""
        graph = build_graph(sample_extraction_dir, use_mrf=True)
        assert graph.number_of_nodes() > 0
        # Check that MRF posteriors are written to edges
        for u, v, key, data in graph.edges(data=True, keys=True):
            if data.get("edge_id"):
                assert "mrf_confidence" in data or "confidence_mean" in data

    def test_build_graph_without_mrf(self, sample_extraction_dir: Path):
        """Default pipeline (no MRF) should still work."""
        graph = build_graph(sample_extraction_dir)
        assert graph.number_of_nodes() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_graph/test_mrf_integration.py -v`
Expected: FAIL with `TypeError: build_graph() got an unexpected keyword argument 'use_mrf'`

- [ ] **Step 3: Add `use_mrf` parameter to `build_graph`**

Modify `autoreview/knowledge_graph/__init__.py`:

1. Add imports for `mrf_scoring` and `MRFConfig`
2. Add `use_mrf: bool = False` and `mrf_config: MRFConfig | None = None` parameters to `build_graph()`
3. After Step 8 (Beta-Binomial scoring), add Step 8b:

```python
    # ------------------------------------------------------------------
    # Step 8b: MRF confidence propagation (optional)
    # ------------------------------------------------------------------
    if use_mrf:
        from autoreview.knowledge_graph.mrf_scoring import MRFConfig, MRFResult, score_graph_mrf

        mrf_cfg = mrf_config or MRFConfig()
        mrf_result = score_graph_mrf(graph, config=mrf_cfg)

        # Write MRF posteriors back to edges
        for u, v, key, data in graph.edges(data=True, keys=True):
            edge_id = data.get("edge_id")
            if edge_id and edge_id in mrf_result.posteriors:
                graph[u][v][key]["mrf_confidence"] = mrf_result.posteriors[edge_id]

        log.info(
            "kg.pipeline.mrf_done",
            n_rules=mrf_result.n_rules,
            n_contradictions=mrf_result.n_contradictions,
            n_compositions=mrf_result.n_compositions,
        )
```

4. Update `__all__` to export `MRFConfig`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_knowledge_graph/ -v --tb=short`
Expected: PASS (all existing tests + new integration test)

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/__init__.py tests/test_knowledge_graph/test_mrf_integration.py
git commit -m "feat(kg): wire MRF scoring into build_graph pipeline"
```

---

### Task 7: Extraction Prompt Updates

**Files:**
- Modify: `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md`

Two small additions for graph quality: entity consistency within a single paper, and pathway completeness.

- [ ] **Step 1: Add entity consistency rule to CLAIM RULES section**

After the line `- **Scoped**: capture ALL qualifying conditions in the \`conditions\` object`, add:

```markdown
- **Consistent entity naming**: Use the most specific canonical name for each entity consistently across ALL claims in this extraction. If the paper uses multiple names for the same entity (e.g., "T", "Brachyury", "TBXT"), pick the most common form in the paper and use it everywhere. Do not alternate between synonyms across claims.
```

- [ ] **Step 2: Add pathway completeness rule to CLAIM RULES section**

After the entity consistency rule, add:

```markdown
- **Pathway completeness**: For multi-step mechanisms (e.g., "X activates Y, which in turn inhibits Z"), extract each step as a separate claim. Do not collapse pathway steps into a single claim — the graph needs each edge to enable transitive inference.
```

- [ ] **Step 3: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md"
git commit -m "feat(kg): add entity consistency and pathway completeness rules to extraction prompt"
```

---

## Execution Batching

```
Batch 1 (parallel, no deps):
  Agent → Task 1: predicate_algebra.py + tests
  Agent → Task 2: condition_compat.py + tests
  Agent → Task 3: hlmrf.py + tests
  Agent → Task 7: extraction prompt updates

Batch 2 (needs Tasks 1 + 2):
  Agent → Task 4: structural_contradictions.py + tests

Batch 3 (needs Tasks 1 + 2 + 3 + 4):
  Agent → Task 5: mrf_scoring.py + tests

Batch 4 (needs all above):
  Agent → Task 6: pipeline integration + tests
```
