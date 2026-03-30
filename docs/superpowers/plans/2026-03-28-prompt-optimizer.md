# Extraction Prompt Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autoresearch-inspired autonomous loop that iteratively improves `kg_extraction_prompt.md` by extracting on test papers, scoring, analyzing errors, and using an LLM optimizer to propose prompt edits.

**Architecture:** Three-file pattern inspired by Karpathy's autoresearch: `program.md` (optimizer instructions), `kg_extraction_prompt.md` (the artifact being optimized), and `optimize_extraction_prompt.py` (the experiment runner). Haiku does extraction, Sonnet analyzes errors and proposes edits. Fully autonomous — no human checkpoints.

**Tech Stack:** Python 3.11+, `claude` CLI (`claude -p`), Pydantic for validation, subprocess for agent invocation. No external ML libraries.

---

## File Structure

```
Paper Extractor/KnowledgeGraph Extraction/
├── optimize/
│   ├── __init__.py                    # Package init
│   ├── scoring.py                     # 12-metric composite scoring function
│   ├── error_analyzer.py              # Systematic error pattern classifier
│   ├── experiment_runner.py           # Run extractions via claude -p
│   └── program.md                     # Optimizer agent system prompt
├── optimize_extraction_prompt.py      # Main CLI entrypoint — the loop
├── prompt_versions/                   # Auto-created: version snapshots
│   ├── v6.1_baseline.md
│   ├── v6.1_baseline_scores.json
│   └── ...
└── tests/
    └── test_optimizer/
        ├── test_scoring.py
        └── test_error_analyzer.py
```

**Responsibilities:**
- `scoring.py`: Pure function — takes extraction JSON dict, returns (composite_score, per_metric_dict). No I/O.
- `error_analyzer.py`: Pure function — takes list of extraction dicts, returns ranked list of `ErrorPattern` dataclasses. No I/O.
- `experiment_runner.py`: I/O layer — loads papers, calls `claude -p`, parses JSON. Thin wrapper.
- `program.md`: Static text — system prompt for the Sonnet optimizer agent. The "brain" of the loop.
- `optimize_extraction_prompt.py`: Orchestrator — wires scoring + errors + optimizer + version control into the autonomous loop.

---

### Task 1: Scoring Function

**Files:**
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize/__init__.py`
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize/scoring.py`
- Create: `Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/__init__.py`
- Create: `Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimizer/test_scoring.py
"""Tests for extraction scoring function."""

from optimize.scoring import score_extraction, METRIC_WEIGHTS


def _make_claim(**overrides):
    """Helper to build a minimal valid claim dict."""
    base = {
        "claim_id": "c_001",
        "natural_language": "BMP4 induces mesoderm differentiation",
        "subject": {"name": "BMP4", "type": "protein", "ontology_id": None},
        "predicate": "induces",
        "object": {"name": "mesoderm differentiation", "type": "biological_process", "ontology_id": None},
        "direction": "positive",
        "claim_type": "mechanistic_causal",
        "causal_type": "necessary",
        "conditions": {"species": ["Mus musculus"], "cell_type": ["mESC"]},
        "evidence_strength": "direct_experimental",
        "certainty": "high",
        "section_source": "primary_empirical",
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "quantitative_context": {"concentration": "10 ng/mL BMP4", "timepoint": "48h", "dose": None},
        "evidence_links": [{"evidence_id": "e_001", "direction": "supports"}],
    }
    base.update(overrides)
    return base


def test_score_empty_extraction():
    """No claims should return 0.0 composite."""
    composite, metrics = score_extraction({"claims": [], "evidence": []})
    assert composite == 0.0
    assert all(v == 0.0 for v in metrics.values())


def test_score_perfect_30_claims():
    """30 well-formed claims with diverse predicates should score high."""
    predicates = [
        "induces", "inhibits", "is_required_for", "correlates_with",
        "interacts_with", "is_located_in", "differentiates_into",
        "is_marker_of", "is_component_of", "regulates",
    ]
    claims = [
        _make_claim(
            claim_id=f"c_{i:03d}",
            predicate=predicates[i % len(predicates)],
        )
        for i in range(30)
    ]
    composite, metrics = score_extraction({"claims": claims, "evidence": []})
    assert composite > 0.85
    assert metrics["density"] == 1.0
    assert metrics["predicate_valid"] == 1.0
    assert metrics["claim_type_valid"] == 1.0
    assert metrics["evidence_strength_valid"] == 1.0


def test_density_below_25():
    """10 claims should score density < 1.0."""
    claims = [_make_claim(claim_id=f"c_{i:03d}") for i in range(10)]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["density"] == 10 / 25


def test_density_above_50():
    """60 claims should be penalized."""
    claims = [_make_claim(claim_id=f"c_{i:03d}") for i in range(60)]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["density"] < 1.0
    assert metrics["density"] > 0.0


def test_invalid_predicates_penalized():
    """Removed predicates should lower predicate_valid score."""
    claims = [
        _make_claim(claim_id="c_001", predicate="activates"),  # REMOVED
        _make_claim(claim_id="c_002", predicate="promotes"),   # REMOVED
        _make_claim(claim_id="c_003", predicate="induces"),    # VALID
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["predicate_valid"] == 1 / 3


def test_predicate_diversity_all_same():
    """All same predicate should score 0.0 diversity."""
    claims = [_make_claim(claim_id=f"c_{i:03d}", predicate="induces") for i in range(30)]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["predicate_diversity"] == 0.0


def test_predicate_diversity_varied():
    """5 different predicates should score higher than 1 predicate."""
    preds = ["induces", "inhibits", "is_required_for", "correlates_with", "interacts_with"]
    claims = [
        _make_claim(claim_id=f"c_{i:03d}", predicate=preds[i % len(preds)])
        for i in range(30)
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["predicate_diversity"] > 0.8


def test_missing_model_system():
    """Claims without model_system should lower field score."""
    claims = [
        _make_claim(claim_id="c_001", model_system="mouse ESC"),
        _make_claim(claim_id="c_002", model_system=None),
        _make_claim(claim_id="c_003", model_system=None),
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["field_model_system"] == 1 / 3


def test_missing_causal_type_on_mechanistic():
    """Mechanistic claims without causal_type should be penalized."""
    claims = [
        _make_claim(claim_id="c_001", claim_type="mechanistic_causal", causal_type="necessary"),
        _make_claim(claim_id="c_002", claim_type="mechanistic_causal", causal_type=None),
        _make_claim(claim_id="c_003", claim_type="correlational", causal_type=None),  # OK — not mechanistic
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["causal_type_coverage"] == 0.5  # 1/2 mechanistic claims have causal_type


def test_perturbation_as_subject_detected():
    """Entity subjects with perturbation keywords should be penalized."""
    claims = [
        _make_claim(claim_id="c_001", subject={"name": "BMP4 knockdown", "type": "protein", "ontology_id": None}),
        _make_claim(claim_id="c_002", subject={"name": "BMP4", "type": "protein", "ontology_id": None}),
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["no_perturb_subject"] == 0.5


def test_invalid_claim_type():
    """Non-vocabulary claim types should lower claim_type_valid."""
    claims = [
        _make_claim(claim_id="c_001", claim_type="mechanistic_causal"),
        _make_claim(claim_id="c_002", claim_type="quantitative_comparative"),  # INVALID
    ]
    _, metrics = score_extraction({"claims": claims, "evidence": []})
    assert metrics["claim_type_valid"] == 0.5


def test_weights_sum_to_one():
    """Metric weights should sum to 1.0."""
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -m pytest tests/test_optimizer/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optimize'`

- [ ] **Step 3: Implement the scoring module**

```python
# optimize/__init__.py
```

```python
# optimize/scoring.py
"""12-metric composite scoring for KG extraction quality.

Each metric produces a 0-1 score. The composite is a weighted average.
Pure function — no I/O, no side effects.
"""

from __future__ import annotations

from collections import Counter
from math import log2

# --- Predicate vocabulary (must match kg_extraction_prompt.md) ---
TIER1_CANONICAL = {
    "induces",
    "inhibits",
    "is_required_for",
    "is_sufficient_for",
    "regulates",
    "correlates_with",
    "interacts_with",
    "differentiates_into",
    "is_located_in",
    "is_marker_of",
    "is_component_of",
    "colocalizes_with",
}
TIER2_SPECIFIC = {
    "phosphorylates",
    "degrades",
    "stabilizes",
    "transports",
    "modifies",
    "converts",
    "maintains",
}
VALID_PREDICATES = TIER1_CANONICAL | TIER2_SPECIFIC

VALID_EVIDENCE_STRENGTHS = {
    "direct_experimental",
    "indirect_experimental",
    "observational",
    "computational",
    "review_citation",
}

VALID_CLAIM_TYPES = {
    "mechanistic_causal",
    "correlational",
    "comparative",
    "existence",
    "absence",
    "conditional",
    "methodological",
}

PERTURB_KEYWORDS = {
    "knockdown",
    "knockout",
    "depletion",
    "silencing",
    "overexpression",
    "ko",
    "kd",
    "sirna",
    "shrna",
    "crispr",
}

METRIC_WEIGHTS: dict[str, float] = {
    "density": 0.15,
    "predicate_valid": 0.15,
    "predicate_diversity": 0.10,
    "evidence_strength_valid": 0.10,
    "field_model_system": 0.05,
    "field_organism": 0.05,
    "field_certainty": 0.05,
    "field_section_source": 0.05,
    "quant_context": 0.10,
    "causal_type_coverage": 0.10,
    "no_perturb_subject": 0.05,
    "claim_type_valid": 0.05,
}


def score_extraction(data: dict) -> tuple[float, dict[str, float]]:
    """Score a single extraction JSON dict.

    Returns:
        (composite_score, per_metric_dict) where both are in [0, 1].
    """
    claims = data.get("claims", [])
    n = len(claims)
    if n == 0:
        return 0.0, {k: 0.0 for k in METRIC_WEIGHTS}

    scores: dict[str, float] = {}

    # 1. Density — target 25-50 claims per paper
    if 25 <= n <= 50:
        scores["density"] = 1.0
    elif n < 25:
        scores["density"] = n / 25
    else:
        scores["density"] = max(0.0, 1.0 - (n - 50) / 50)

    # 2. Predicate validity — fraction using Tier 1 or Tier 2 predicates
    preds = [c.get("predicate", "") for c in claims]
    scores["predicate_valid"] = sum(1 for p in preds if p in VALID_PREDICATES) / n

    # 3. Predicate diversity — normalized Shannon entropy
    pred_counts = Counter(preds)
    if len(pred_counts) <= 1:
        scores["predicate_diversity"] = 0.0
    else:
        total = sum(pred_counts.values())
        entropy = -sum(
            (c / total) * log2(c / total) for c in pred_counts.values() if c > 0
        )
        max_entropy = log2(len(pred_counts))
        scores["predicate_diversity"] = (
            entropy / max_entropy if max_entropy > 0 else 0.0
        )

    # 4. Evidence strength validity
    strengths = [c.get("evidence_strength", "") for c in claims]
    scores["evidence_strength_valid"] = (
        sum(1 for s in strengths if s in VALID_EVIDENCE_STRENGTHS) / n
    )

    # 5-8. Field coverage
    for metric_key, field_name in [
        ("field_model_system", "model_system"),
        ("field_organism", "organism"),
        ("field_certainty", "certainty"),
        ("field_section_source", "section_source"),
    ]:
        scores[metric_key] = sum(1 for c in claims if c.get(field_name)) / n

    # 9. Quantitative context — populated with at least one non-null value
    has_quant = 0
    for c in claims:
        qc = c.get("quantitative_context")
        if isinstance(qc, dict) and any(v for v in qc.values() if v):
            has_quant += 1
    scores["quant_context"] = has_quant / n

    # 10. Causal type coverage on mechanistic_causal claims
    mechanistic = [c for c in claims if c.get("claim_type") == "mechanistic_causal"]
    if mechanistic:
        scores["causal_type_coverage"] = (
            sum(1 for c in mechanistic if c.get("causal_type")) / len(mechanistic)
        )
    else:
        scores["causal_type_coverage"] = 1.0  # N/A — don't penalize

    # 11. No perturbation-as-subject
    perturb_count = 0
    for c in claims:
        subj = c.get("subject", {})
        if isinstance(subj, dict):
            name = (subj.get("name", "") or "").lower()
            if any(kw in name for kw in PERTURB_KEYWORDS):
                perturb_count += 1
    scores["no_perturb_subject"] = 1.0 - perturb_count / n

    # 12. Claim type validity
    scores["claim_type_valid"] = (
        sum(1 for c in claims if c.get("claim_type", "") in VALID_CLAIM_TYPES) / n
    )

    # Weighted composite
    composite = sum(METRIC_WEIGHTS[k] * scores[k] for k in METRIC_WEIGHTS)

    return composite, scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -m pytest tests/test_optimizer/test_scoring.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/optimize/__init__.py" \
       "Paper Extractor/KnowledgeGraph Extraction/optimize/scoring.py" \
       "Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/__init__.py" \
       "Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/test_scoring.py"
git commit -m "feat(optimize): add 12-metric extraction scoring function with tests"
```

---

### Task 2: Error Analyzer

**Files:**
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize/error_analyzer.py`
- Create: `Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/test_error_analyzer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimizer/test_error_analyzer.py
"""Tests for systematic error pattern detection."""

from optimize.error_analyzer import analyze_errors, ErrorPattern


def _make_claim(**overrides):
    base = {
        "claim_id": "c_001",
        "natural_language": "BMP4 induces mesoderm",
        "subject": {"name": "BMP4", "type": "protein", "ontology_id": None},
        "predicate": "induces",
        "object": {"name": "mesoderm", "type": "biological_process", "ontology_id": None},
        "direction": "positive",
        "claim_type": "mechanistic_causal",
        "causal_type": "necessary",
        "evidence_strength": "direct_experimental",
        "certainty": "high",
        "section_source": "primary_empirical",
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "quantitative_context": None,
    }
    base.update(overrides)
    return base


def _make_extraction(claims):
    return {"title": "Test Paper", "claims": claims, "evidence": []}


def test_no_errors_clean_extraction():
    """Clean extraction with diverse predicates returns empty list."""
    preds = ["induces", "inhibits", "is_required_for", "correlates_with", "interacts_with"]
    claims = [_make_claim(claim_id=f"c_{i:03d}", predicate=preds[i % len(preds)]) for i in range(25)]
    errors = analyze_errors([_make_extraction(claims)])
    # Should have no high-severity errors
    critical = [e for e in errors if e.severity > 0.5]
    assert len(critical) == 0


def test_detects_predicate_overuse():
    """Single predicate >30% should be flagged."""
    claims = [_make_claim(claim_id=f"c_{i:03d}", predicate="regulates") for i in range(10)]
    errors = analyze_errors([_make_extraction(claims)])
    overuse = [e for e in errors if e.category == "predicate_overuse"]
    assert len(overuse) >= 1
    assert "regulates" in overuse[0].description


def test_detects_invalid_predicate():
    """Removed predicates should be flagged."""
    claims = [
        _make_claim(claim_id="c_001", predicate="activates"),
        _make_claim(claim_id="c_002", predicate="promotes"),
    ]
    errors = analyze_errors([_make_extraction(claims)])
    invalid = [e for e in errors if e.category == "invalid_predicate"]
    assert len(invalid) >= 1


def test_detects_perturbation_subject():
    """'X knockdown' as subject should be flagged."""
    claims = [
        _make_claim(
            claim_id="c_001",
            subject={"name": "Rai14 knockdown", "type": "protein", "ontology_id": None},
        ),
    ]
    errors = analyze_errors([_make_extraction(claims)])
    perturb = [e for e in errors if e.category == "perturbation_as_subject"]
    assert len(perturb) == 1


def test_detects_parenthetical_entity():
    """Entity names with parenthetical synonyms should be flagged."""
    claims = [
        _make_claim(
            claim_id="c_001",
            subject={"name": "Invariant chain (CD74)", "type": "protein", "ontology_id": None},
        ),
    ]
    errors = analyze_errors([_make_extraction(claims)])
    paren = [e for e in errors if e.category == "parenthetical_entity_name"]
    assert len(paren) == 1


def test_detects_invalid_claim_type():
    """Non-vocabulary claim types should be flagged."""
    claims = [_make_claim(claim_id="c_001", claim_type="quantitative_comparative")]
    errors = analyze_errors([_make_extraction(claims)])
    invalid = [e for e in errors if e.category == "invalid_claim_type"]
    assert len(invalid) == 1
    assert "quantitative_comparative" in invalid[0].description


def test_detects_missing_causal_type():
    """Mechanistic claims without causal_type should be flagged."""
    claims = [
        _make_claim(claim_id="c_001", claim_type="mechanistic_causal", causal_type=None),
    ]
    errors = analyze_errors([_make_extraction(claims)])
    missing = [e for e in errors if e.category == "missing_causal_type"]
    assert len(missing) == 1


def test_detects_low_density():
    """Extraction with <15 claims should flag low density."""
    claims = [_make_claim(claim_id=f"c_{i:03d}") for i in range(8)]
    errors = analyze_errors([_make_extraction(claims)])
    low = [e for e in errors if e.category == "low_density"]
    assert len(low) == 1


def test_errors_sorted_by_impact():
    """Errors should be sorted by severity * frequency, descending."""
    claims = [_make_claim(claim_id=f"c_{i:03d}", predicate="activates") for i in range(5)]
    claims.append(
        _make_claim(
            claim_id="c_006",
            subject={"name": "BMP4 knockdown", "type": "protein", "ontology_id": None},
        )
    )
    errors = analyze_errors([_make_extraction(claims)])
    # Verify sorted
    impacts = [e.severity * e.frequency for e in errors]
    assert impacts == sorted(impacts, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -m pytest tests/test_optimizer/test_error_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the error analyzer**

```python
# optimize/error_analyzer.py
"""Systematic error pattern detection across extractions.

Analyzes extraction outputs to identify recurring issues that
the optimizer agent should address via prompt modifications.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from optimize.scoring import (
    PERTURB_KEYWORDS,
    VALID_CLAIM_TYPES,
    VALID_EVIDENCE_STRENGTHS,
    VALID_PREDICATES,
)

# Mapping from removed predicates to their canonical replacements
REMOVED_TO_CANONICAL: dict[str, str] = {
    "activates": "induces",
    "promotes": "induces",
    "upregulates": "induces",
    "enhances": "induces",
    "suppresses": "inhibits",
    "blocks": "inhibits",
    "downregulates": "inhibits",
    "prevents": "inhibits",
    "reduces": "inhibits",
    "disrupts": "inhibits",
    "binds_to": "interacts_with",
    "localizes_to": "is_located_in",
    "is_expressed_in": "is_located_in",
    "is_necessary_for": "is_required_for",
    "enables": "is_required_for",
    "mediates": "regulates",
}


@dataclass
class ErrorPattern:
    """A systematic extraction error pattern."""

    category: str
    description: str
    severity: float  # 0-1
    frequency: int
    examples: list[str] = field(default_factory=list)


def analyze_errors(extractions: list[dict]) -> list[ErrorPattern]:
    """Identify systematic error patterns across multiple extractions.

    Args:
        extractions: List of extraction JSON dicts (each with "claims", "evidence", "title").

    Returns:
        List of ErrorPattern sorted by severity * frequency (descending).
    """
    patterns: list[ErrorPattern] = []
    all_claims: list[dict] = []
    for ext in extractions:
        all_claims.extend(ext.get("claims", []))

    if not all_claims:
        return patterns

    n = len(all_claims)

    # 1. Predicate overuse — any single predicate >30% of claims
    pred_counts = Counter(c.get("predicate", "") for c in all_claims)
    for pred, count in pred_counts.most_common():
        ratio = count / n
        if ratio > 0.30:
            examples = [
                c.get("natural_language", "")[:100]
                for c in all_claims
                if c.get("predicate") == pred
            ][:3]
            patterns.append(
                ErrorPattern(
                    category="predicate_overuse",
                    description=(
                        f"Predicate '{pred}' used in {count}/{n} claims ({ratio:.0%}). "
                        f"Review whether some should use more specific predicates."
                    ),
                    severity=min(1.0, (ratio - 0.30) * 3),
                    frequency=count,
                    examples=examples,
                )
            )

    # 2. Invalid predicates (removed or unknown)
    invalid_preds = Counter(
        p
        for c in all_claims
        if (p := c.get("predicate", "")) not in VALID_PREDICATES
    )
    for pred, count in invalid_preds.most_common():
        canonical = REMOVED_TO_CANONICAL.get(pred)
        patterns.append(
            ErrorPattern(
                category="invalid_predicate",
                description=(
                    f"Predicate '{pred}' is not in vocabulary ({count} uses). "
                    + (f"Use '{canonical}' instead." if canonical else "Map to a canonical predicate.")
                ),
                severity=0.8,
                frequency=count,
                examples=[
                    c.get("natural_language", "")[:100]
                    for c in all_claims
                    if c.get("predicate") == pred
                ][:3],
            )
        )

    # 3. Invalid claim types
    invalid_types = Counter(
        t
        for c in all_claims
        if (t := c.get("claim_type", "")) not in VALID_CLAIM_TYPES
    )
    for ct, count in invalid_types.most_common():
        patterns.append(
            ErrorPattern(
                category="invalid_claim_type",
                description=(
                    f"Claim type '{ct}' is invalid ({count} uses). "
                    f"Valid: {', '.join(sorted(VALID_CLAIM_TYPES))}."
                ),
                severity=0.7,
                frequency=count,
                examples=[
                    c.get("natural_language", "")[:100]
                    for c in all_claims
                    if c.get("claim_type") == ct
                ][:3],
            )
        )

    # 4. Invalid evidence strengths
    invalid_strengths = Counter(
        s
        for c in all_claims
        if (s := c.get("evidence_strength", "")) not in VALID_EVIDENCE_STRENGTHS
    )
    for es, count in invalid_strengths.most_common():
        patterns.append(
            ErrorPattern(
                category="invalid_evidence_strength",
                description=f"Evidence strength '{es}' is invalid ({count} uses).",
                severity=0.7,
                frequency=count,
                examples=[],
            )
        )

    # 5. Missing required fields
    for field_name in ["model_system", "organism", "certainty", "section_source"]:
        missing = sum(1 for c in all_claims if not c.get(field_name))
        if missing > 0:
            ratio = missing / n
            patterns.append(
                ErrorPattern(
                    category="missing_field",
                    description=f"Field '{field_name}' missing on {missing}/{n} claims ({ratio:.0%}).",
                    severity=ratio * 0.6,
                    frequency=missing,
                    examples=[
                        c.get("natural_language", "")[:100]
                        for c in all_claims
                        if not c.get(field_name)
                    ][:3],
                )
            )

    # 6. Missing causal_type on mechanistic claims
    mechanistic = [c for c in all_claims if c.get("claim_type") == "mechanistic_causal"]
    if mechanistic:
        missing_causal = [c for c in mechanistic if not c.get("causal_type")]
        if missing_causal:
            patterns.append(
                ErrorPattern(
                    category="missing_causal_type",
                    description=(
                        f"causal_type missing on {len(missing_causal)}/{len(mechanistic)} "
                        f"mechanistic_causal claims."
                    ),
                    severity=0.6,
                    frequency=len(missing_causal),
                    examples=[c.get("natural_language", "")[:100] for c in missing_causal][:3],
                )
            )

    # 7. Perturbation-as-subject
    perturb_claims = []
    for c in all_claims:
        subj = c.get("subject", {})
        if isinstance(subj, dict):
            name = (subj.get("name", "") or "").lower()
            if any(kw in name for kw in PERTURB_KEYWORDS):
                perturb_claims.append(c)
    if perturb_claims:
        patterns.append(
            ErrorPattern(
                category="perturbation_as_subject",
                description=(
                    f"{len(perturb_claims)} claims use experimental perturbation as entity "
                    f"subject instead of the gene/protein itself."
                ),
                severity=0.9,
                frequency=len(perturb_claims),
                examples=[
                    f"{c.get('subject', {}).get('name', '')}: "
                    f"{c.get('natural_language', '')[:80]}"
                    for c in perturb_claims
                ][:3],
            )
        )

    # 8. Entity naming with parenthetical synonyms
    paren_claims = []
    for c in all_claims:
        for entity_key in ("subject", "object"):
            ent = c.get(entity_key, {})
            if isinstance(ent, dict):
                name = ent.get("name", "") or ""
                if "(" in name and ")" in name:
                    paren_claims.append(c)
                    break
    if paren_claims:
        patterns.append(
            ErrorPattern(
                category="parenthetical_entity_name",
                description=(
                    f"{len(paren_claims)} claims use parenthetical synonyms in entity names. "
                    f"Use bare canonical names."
                ),
                severity=0.5,
                frequency=len(paren_claims),
                examples=[
                    f"{c.get('subject', {}).get('name', '')} / "
                    f"{c.get('object', {}).get('name', '')}"
                    for c in paren_claims
                ][:3],
            )
        )

    # 9. Low claim density per paper
    for ext in extractions:
        n_claims = len(ext.get("claims", []))
        title = ext.get("title", "unknown paper")[:50]
        if n_claims < 15:
            patterns.append(
                ErrorPattern(
                    category="low_density",
                    description=f"Only {n_claims} claims from '{title}'. Target is 25-50.",
                    severity=0.7,
                    frequency=1,
                    examples=[],
                )
            )

    # Sort by impact (severity * frequency), descending
    patterns.sort(key=lambda p: p.severity * p.frequency, reverse=True)
    return patterns
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -m pytest tests/test_optimizer/test_error_analyzer.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/optimize/error_analyzer.py" \
       "Paper Extractor/KnowledgeGraph Extraction/tests/test_optimizer/test_error_analyzer.py"
git commit -m "feat(optimize): add systematic error pattern analyzer with tests"
```

---

### Task 3: Experiment Runner

**Files:**
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize/experiment_runner.py`

No unit tests for this module — it wraps `claude -p` subprocess calls. Validated by integration in Task 5.

- [ ] **Step 1: Implement the experiment runner**

```python
# optimize/experiment_runner.py
"""Run KG extractions via claude -p for prompt optimization.

Thin I/O layer: loads papers, calls claude CLI, parses JSON output.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # KnowledgeGraph Extraction/
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

# Papers that are reviews or abstract-only — skip these
_SKIP_INDICES = {0, 1, 3, 4, 5, 9}
_MIN_TEXT_LENGTH = 5000


def load_test_papers(
    micro_indices: list[int] | None = None,
    include_rai14: bool = True,
) -> list[dict]:
    """Load test papers: rai14 + selected micro_sample entries.

    Args:
        micro_indices: Indices into micro_sample.json (default: [2, 6]).
            Papers that are reviews or abstract-only are auto-filtered.
        include_rai14: Whether to include rai14_fulltext.txt.

    Returns:
        List of dicts with keys: id, title, text.
    """
    if micro_indices is None:
        micro_indices = [2, 6]

    papers: list[dict] = []

    # rai14 — well-characterized baseline
    if include_rai14:
        rai14_path = SCRIPT_DIR / "rai14_fulltext.txt"
        if rai14_path.exists():
            papers.append(
                {
                    "id": "rai14",
                    "title": "Rai14 is a novel interactor of Invariant chain",
                    "text": rai14_path.read_text(),
                }
            )

    # micro_sample papers (filtered)
    micro_path = SCRIPT_DIR / "gastruloid_run" / "micro_sample.json"
    if micro_path.exists() and micro_indices:
        micro = json.loads(micro_path.read_text())
        for idx in micro_indices:
            if idx in _SKIP_INDICES:
                continue
            if 0 <= idx < len(micro):
                text = micro[idx].get("full_text", "")
                if len(text) < _MIN_TEXT_LENGTH:
                    continue
                papers.append(
                    {
                        "id": f"micro_{idx}",
                        "title": micro[idx].get("title", "")[:80],
                        "text": text,
                    }
                )

    return papers


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output with 3-tier fallback."""
    raw = raw.strip()
    # Tier 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Tier 2: markdown fence
    fence = _FENCE_RE.search(raw)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Tier 3: brace slice
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        return json.loads(raw[first : last + 1])
    msg = f"No valid JSON in output ({len(raw)} chars)"
    raise ValueError(msg)


def run_extraction(prompt_text: str, paper_text: str, timeout: int = 300) -> dict:
    """Run a single extraction via claude -p --model haiku.

    Args:
        prompt_text: Full prompt markdown (with {PAPER_TEXT} marker).
        paper_text: Full text of the paper.
        timeout: Max seconds for the subprocess.

    Returns:
        Parsed extraction JSON dict.
    """
    marker = "{PAPER_TEXT}"
    idx = prompt_text.find(marker)
    system_prompt = prompt_text[:idx].rstrip() if idx != -1 else prompt_text

    user_prompt = (
        "Extract all falsifiable claims from the following paper as structured JSON "
        "according to the schema in your system prompt. Output ONLY valid JSON.\n\n"
        "---\n\n" + paper_text
    )

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--max-turns",
            "3",
            "--system-prompt",
            system_prompt,
        ],
        input=user_prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed (rc={result.returncode}): {result.stderr[:500]}")

    return _parse_json(result.stdout)


def run_all_extractions(
    prompt_text: str,
    papers: list[dict],
    timeout: int = 300,
) -> list[dict]:
    """Run extraction on all test papers sequentially.

    Returns:
        List of extraction dicts. Failed extractions have _error key and empty claims.
    """
    results: list[dict] = []
    for paper in papers:
        title_short = paper["title"][:50]
        print(f"  Extracting: {title_short}...", flush=True)
        t0 = time.time()
        try:
            data = run_extraction(prompt_text, paper["text"], timeout=timeout)
            elapsed = time.time() - t0
            data["_paper_id"] = paper["id"]
            data["_elapsed"] = round(elapsed, 1)
            n_claims = len(data.get("claims", []))
            print(f"    -> {n_claims} claims in {elapsed:.1f}s")
            results.append(data)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    -> FAILED ({elapsed:.1f}s): {e}")
            results.append(
                {
                    "_paper_id": paper["id"],
                    "_error": str(e)[:200],
                    "claims": [],
                    "evidence": [],
                    "title": paper["title"],
                }
            )
    return results
```

- [ ] **Step 2: Smoke test — verify paper loading works**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -c "from optimize.experiment_runner import load_test_papers; papers = load_test_papers([2, 6]); print(f'{len(papers)} papers loaded: {[p[\"id\"] for p in papers]}')" `
Expected: `3 papers loaded: ['rai14', 'micro_2', 'micro_6']`

- [ ] **Step 3: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/optimize/experiment_runner.py"
git commit -m "feat(optimize): add experiment runner for claude -p extractions"
```

---

### Task 4: Optimizer Agent Program (program.md)

**Files:**
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize/program.md`

- [ ] **Step 1: Write the optimizer agent system prompt**

```markdown
# Extraction Prompt Optimizer Agent

You are a prompt optimization agent. Your task is to improve a knowledge graph extraction prompt by analyzing systematic errors and proposing targeted modifications.

## Your Role

You receive:
1. The current extraction prompt (markdown)
2. A composite quality score (0-1) with per-metric breakdown
3. A ranked list of systematic error patterns found across test extractions
4. History of past edit attempts and their outcomes

You output: The COMPLETE modified prompt in a single markdown code fence.

## Optimization Strategy

### Priority Order
1. **Fix invalid values first** — invalid predicates, claim types, evidence strengths. These are binary errors with clear fixes (add to mapping table, add example).
2. **Fix missing fields second** — model_system, organism, causal_type, quantitative_context. Strengthen requirement language or add examples.
3. **Fix behavioral issues third** — perturbation-as-subject, parenthetical entity names, predicate overuse. Add explicit rules or counter-examples.
4. **Improve density last** — if claim count is low, add extraction encouragement or expand section rules.

### Edit Principles
- **Minimal changes**: Modify at most 2-3 specific rules, examples, or wordings per iteration. Do NOT rewrite whole sections that are working.
- **Targeted fixes**: Each edit should address a specific error pattern from the analysis.
- **Add, don't remove**: Prefer adding clarifications, examples, or rules over removing existing content. Only remove content if it is actively causing confusion.
- **Preserve structure**: Keep all section headers, the {PAPER_TEXT} marker, the output format section, and the quality checklist intact.
- **Learn from history**: If a previous edit was rejected, do not repeat it. Try a different approach to the same problem.

### Predicate Overuse
When a single predicate is used >30% of the time (e.g., `regulates` at 44%), the fix is NOT to remove or restrict the predicate. Instead:
- Tighten its definition: add "ONLY when direction of regulation is genuinely unknown"
- Add decision tree examples: "If loss-of-X abolishes Y → `is_required_for`, not `regulates`"
- Add counter-examples in the common mistakes section

### Entity Naming Issues
When entity names include perturbation language ("X knockdown") or parenthetical synonyms ("X (alias)"):
- Strengthen the existing rule with more examples
- Add the specific bad pattern to the quality checklist

### Missing Fields
When fields like `causal_type` or `quantitative_context` have low coverage:
- Add explicit REQUIRED markers with bold emphasis
- Add examples showing correct population
- Add to quality checklist if not already there

## Output Format

Output the COMPLETE modified prompt inside a single markdown code fence:

~~~
```markdown
# Knowledge Graph Extraction Prompt — v6.X

[... entire prompt ...]

{PAPER_TEXT}
```
~~~

Do NOT output partial diffs, explanations before the fence, or multiple fences. Just the one fence with the complete prompt.

## Critical Constraints

- The prompt MUST end with `{PAPER_TEXT}` on its own line
- The JSON output format section MUST remain unchanged (schema is fixed)
- Do NOT add new fields to the JSON schema
- Do NOT change predicate vocabulary (Tier 1 + Tier 2 sets are fixed)
- Do NOT change evidence_strength vocabulary (5 values are fixed)
- Do NOT change claim_type vocabulary (7 values are fixed)
- You may add rules, examples, clarifications, counter-examples, and quality checklist items
- You may reword existing rules for clarity
- You may add to the "Common mapping mistakes" table
```

- [ ] **Step 2: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/optimize/program.md"
git commit -m "feat(optimize): add optimizer agent system prompt (program.md)"
```

---

### Task 5: Main Optimization Loop

**Files:**
- Create: `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py`

**Dependencies:** Tasks 1-4 must be complete.

- [ ] **Step 1: Implement the main loop**

```python
#!/usr/bin/env python3
"""Autonomous extraction prompt optimizer — autoresearch-inspired loop.

Iteratively improves kg_extraction_prompt.md by:
1. Running extraction on 2-3 test papers via claude -p (haiku)
2. Scoring extractions with a 12-metric composite score
3. Analyzing systematic error patterns
4. Using an optimizer agent (sonnet) to propose prompt edits
5. Applying edits, re-extracting, comparing scores
6. Accepting improvements, rejecting regressions
7. Saving prompt versions and optimization log

Usage:
    python optimize_extraction_prompt.py
    python optimize_extraction_prompt.py --max-iterations 5 --papers 2,7
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure optimize package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from optimize.error_analyzer import analyze_errors
from optimize.experiment_runner import load_test_papers, run_all_extractions
from optimize.scoring import METRIC_WEIGHTS, score_extraction

PROMPT_PATH = SCRIPT_DIR / "kg_extraction_prompt.md"
VERSIONS_DIR = SCRIPT_DIR / "prompt_versions"
LOG_PATH = SCRIPT_DIR / "optimize" / "optimization_log.json"
PROGRAM_PATH = SCRIPT_DIR / "optimize" / "program.md"

CONVERGENCE_THRESHOLD = 0.005  # Stop if improvement < 0.5%
REGRESSION_THRESHOLD = 0.05  # Reject if any metric drops > 5%
MAX_CHANGE_RATIO = 0.25  # Reject if >25% of lines changed

_FENCE_RE = re.compile(r"```markdown\s*\n(.*?)```", re.DOTALL)


def score_all(
    extractions: list[dict],
) -> tuple[float, dict[str, float]]:
    """Average composite score across all extractions."""
    if not extractions:
        return 0.0, {}

    composites: list[float] = []
    metric_lists: dict[str, list[float]] = {}

    for ext in extractions:
        composite, metrics = score_extraction(ext)
        composites.append(composite)
        for k, v in metrics.items():
            metric_lists.setdefault(k, []).append(v)

    avg_composite = sum(composites) / len(composites)
    avg_metrics = {k: sum(v) / len(v) for k, v in metric_lists.items()}
    return avg_composite, avg_metrics


def check_regression(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
) -> list[str]:
    """Return list of metrics that regressed beyond threshold."""
    regressions = []
    for k in old_metrics:
        if k in new_metrics:
            drop = old_metrics[k] - new_metrics[k]
            if drop > REGRESSION_THRESHOLD:
                regressions.append(
                    f"{k}: {old_metrics[k]:.3f} -> {new_metrics[k]:.3f} (drop={drop:.3f})"
                )
    return regressions


def compute_change_ratio(old_prompt: str, new_prompt: str) -> float:
    """Fraction of lines that differ between old and new prompt."""
    old_lines = old_prompt.strip().splitlines()
    new_lines = new_prompt.strip().splitlines()
    max_lines = max(len(old_lines), len(new_lines))
    if max_lines == 0:
        return 0.0
    # Simple: count lines that differ (not a proper diff, but fast)
    shared = min(len(old_lines), len(new_lines))
    diffs = sum(1 for i in range(shared) if old_lines[i] != new_lines[i])
    diffs += abs(len(old_lines) - len(new_lines))
    return diffs / max_lines


def call_optimizer(
    current_prompt: str,
    metrics: dict[str, float],
    composite: float,
    errors: list,
    history: list[dict],
) -> str:
    """Call Sonnet optimizer agent to propose prompt modifications.

    Returns:
        The complete modified prompt text.
    """
    program = PROGRAM_PATH.read_text()

    context = json.dumps(
        {
            "composite_score": round(composite, 4),
            "per_metric_scores": {k: round(v, 3) for k, v in metrics.items()},
            "metric_weights": METRIC_WEIGHTS,
            "error_patterns": [
                {
                    "category": e.category,
                    "description": e.description,
                    "severity": round(e.severity, 2),
                    "frequency": e.frequency,
                    "examples": e.examples,
                }
                for e in errors[:8]
            ],
            "edit_history": [
                {
                    "iteration": h["iteration"],
                    "accepted": h["accepted"],
                    "composite_delta": h.get("delta"),
                    "reason": h.get("reason", "accepted" if h["accepted"] else "unknown"),
                }
                for h in history[-5:]
            ],
        },
        indent=2,
    )

    user_prompt = (
        "## Current Extraction Prompt\n\n"
        f"```markdown\n{current_prompt}\n```\n\n"
        "## Current Scores and Error Analysis\n\n"
        f"```json\n{context}\n```\n\n"
        "Analyze the error patterns and propose targeted modifications to improve the "
        "extraction prompt. Output the COMPLETE modified prompt in a single markdown "
        "code fence. Make minimal, targeted changes — at most 2-3 specific modifications."
    )

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            "sonnet",
            "--output-format",
            "text",
            "--max-turns",
            "1",
            "--system-prompt",
            program,
        ],
        input=user_prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Optimizer failed (rc={result.returncode}): {result.stderr[:500]}"
        )

    raw = result.stdout.strip()

    # Extract markdown fence
    fence = _FENCE_RE.search(raw)
    if fence:
        return fence.group(1).strip()

    # Fallback: look for prompt header
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# Knowledge Graph Extraction Prompt"):
            return "\n".join(lines[i:]).strip()

    raise ValueError(
        f"Optimizer output did not contain a markdown fence or prompt header. "
        f"First 200 chars: {raw[:200]}"
    )


def save_version(
    prompt_text: str,
    version: str,
    metrics: dict[str, float],
    composite: float,
) -> Path:
    """Save a prompt version snapshot and its scores."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_path = VERSIONS_DIR / f"{version}.md"
    version_path.write_text(prompt_text)

    meta_path = VERSIONS_DIR / f"{version}_scores.json"
    meta_path.write_text(
        json.dumps(
            {
                "version": version,
                "composite": round(composite, 4),
                "metrics": {k: round(v, 3) for k, v in metrics.items()},
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    return version_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous extraction prompt optimizer"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations (default: 10)",
    )
    parser.add_argument(
        "--papers",
        type=str,
        default="2,6",
        help="Comma-separated micro_sample indices — reviews and abstracts auto-filtered (default: 2,6)",
    )
    parser.add_argument(
        "--no-rai14",
        action="store_true",
        help="Exclude rai14 from test corpus",
    )
    args = parser.parse_args()

    paper_indices = [int(x) for x in args.papers.split(",")]
    papers = load_test_papers(paper_indices, include_rai14=not args.no_rai14)

    if not papers:
        print("ERROR: No test papers loaded. Check micro_sample.json and paper indices.")
        sys.exit(1)

    print(f"Test corpus: {len(papers)} papers — {[p['id'] for p in papers]}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
    print(f"Regression threshold: {REGRESSION_THRESHOLD}")

    # Load current prompt
    current_prompt = PROMPT_PATH.read_text()
    print(f"Prompt: {len(current_prompt)} chars")

    # === BASELINE ===
    print(f"\n{'=' * 60}")
    print("BASELINE EXTRACTION")
    print(f"{'=' * 60}")

    extractions = run_all_extractions(current_prompt, papers)
    baseline_composite, baseline_metrics = score_all(extractions)

    print(f"\nBaseline composite: {baseline_composite:.4f}")
    for k in sorted(baseline_metrics):
        w = METRIC_WEIGHTS.get(k, 0)
        print(f"  {k:<28} {baseline_metrics[k]:.3f}  (weight {w:.2f})")

    save_version(current_prompt, "v6.1_baseline", baseline_metrics, baseline_composite)

    # === OPTIMIZATION LOOP ===
    history: list[dict] = []
    best_composite = baseline_composite
    best_prompt = current_prompt
    best_metrics = baseline_metrics
    consecutive_rejects = 0

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'=' * 60}")

        # Analyze errors
        errors = analyze_errors(extractions)
        if not errors:
            print("No error patterns found — prompt is clean. Stopping.")
            break

        print(f"Error patterns ({len(errors)} total):")
        for e in errors[:5]:
            print(f"  [{e.category}] sev={e.severity:.2f} freq={e.frequency} — {e.description[:80]}")

        # Call optimizer agent
        print("\nCalling optimizer (sonnet)...", flush=True)
        t0 = time.time()
        try:
            new_prompt = call_optimizer(
                current_prompt, best_metrics, best_composite, errors, history
            )
        except Exception as e:
            print(f"Optimizer failed: {e}")
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": f"optimizer_error: {str(e)[:200]}",
                }
            )
            consecutive_rejects += 1
            if consecutive_rejects >= 3:
                print("3 consecutive failures — stopping.")
                break
            continue

        opt_elapsed = time.time() - t0
        print(f"Optimizer returned in {opt_elapsed:.1f}s")

        # Validate structural integrity
        if "{PAPER_TEXT}" not in new_prompt:
            new_prompt = new_prompt.rstrip() + "\n\n---\n\n{PAPER_TEXT}\n"
            print("  (added missing {PAPER_TEXT} marker)")

        # Check change ratio
        change_ratio = compute_change_ratio(current_prompt, new_prompt)
        print(f"Change ratio: {change_ratio:.1%}")
        if change_ratio > MAX_CHANGE_RATIO:
            print(f"REJECTED — too many changes ({change_ratio:.1%} > {MAX_CHANGE_RATIO:.0%})")
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": f"too_many_changes ({change_ratio:.1%})",
                }
            )
            consecutive_rejects += 1
            if consecutive_rejects >= 3:
                print("3 consecutive failures — stopping.")
                break
            continue

        # Re-extract with modified prompt
        print("Re-extracting with modified prompt...")
        new_extractions = run_all_extractions(new_prompt, papers)
        new_composite, new_metrics = score_all(new_extractions)

        delta = new_composite - best_composite
        print(f"\nNew composite: {new_composite:.4f} (delta: {delta:+.4f})")

        # Metric comparison
        for k in sorted(new_metrics):
            old_v = best_metrics.get(k, 0)
            new_v = new_metrics[k]
            d = new_v - old_v
            marker = " <--" if abs(d) > 0.01 else ""
            print(f"  {k:<28} {old_v:.3f} -> {new_v:.3f}  ({d:+.3f}){marker}")

        # Check regression
        regressions = check_regression(best_metrics, new_metrics)
        if regressions:
            print(f"REJECTED — regressions detected:")
            for r in regressions:
                print(f"  {r}")
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "regression",
                    "composite": round(new_composite, 4),
                    "delta": round(delta, 4),
                    "regressions": regressions,
                }
            )
            consecutive_rejects += 1
            if consecutive_rejects >= 3:
                print("3 consecutive failures — stopping.")
                break
            continue

        if delta < 0:
            print(f"REJECTED — composite decreased ({delta:+.4f})")
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "score_decrease",
                    "composite": round(new_composite, 4),
                    "delta": round(delta, 4),
                }
            )
            consecutive_rejects += 1
            if consecutive_rejects >= 3:
                print("3 consecutive failures — stopping.")
                break
            continue

        # === ACCEPTED ===
        consecutive_rejects = 0
        version = f"v6.{iteration + 1}"
        save_version(new_prompt, version, new_metrics, new_composite)

        print(f"ACCEPTED as {version} — composite improved by {delta:+.4f}")

        current_prompt = new_prompt
        extractions = new_extractions
        best_composite = new_composite
        best_metrics = new_metrics
        best_prompt = new_prompt

        history.append(
            {
                "iteration": iteration,
                "accepted": True,
                "composite": round(new_composite, 4),
                "delta": round(delta, 4),
                "version": version,
            }
        )

        # Convergence check
        if delta < CONVERGENCE_THRESHOLD:
            print(
                f"Converged — improvement {delta:.4f} < threshold {CONVERGENCE_THRESHOLD}"
            )
            break

    # === FINAL SUMMARY ===
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")

    accepted = [h for h in history if h.get("accepted")]
    rejected = [h for h in history if not h.get("accepted")]

    print(f"Iterations run:  {len(history)}")
    print(f"Accepted:        {len(accepted)}")
    print(f"Rejected:        {len(rejected)}")
    print(f"Baseline score:  {baseline_composite:.4f}")
    print(f"Final score:     {best_composite:.4f}")
    print(f"Total improvement: {best_composite - baseline_composite:+.4f}")

    if accepted:
        print("\nAccepted versions:")
        for h in accepted:
            print(f"  {h.get('version', '?')}: {h['composite']:.4f} (delta: {h['delta']:+.4f})")

    # Write best prompt back to the main file
    if best_composite > baseline_composite:
        PROMPT_PATH.write_text(best_prompt)
        print(f"\nBest prompt written to {PROMPT_PATH}")
    else:
        print("\nNo improvement — original prompt unchanged.")

    # Save optimization log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "timestamp": datetime.now(UTC).isoformat(),
        "test_papers": [p["id"] for p in papers],
        "max_iterations": args.max_iterations,
        "baseline_composite": round(baseline_composite, 4),
        "final_composite": round(best_composite, 4),
        "total_improvement": round(best_composite - baseline_composite, 4),
        "iterations": history,
        "baseline_metrics": {k: round(v, 3) for k, v in baseline_metrics.items()},
        "final_metrics": {k: round(v, 3) for k, v in best_metrics.items()},
    }
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests to verify all prior modules still pass**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -m pytest tests/test_optimizer/ -v`
Expected: All 21 tests PASS

- [ ] **Step 3: Verify import chain works**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python -c "from optimize_extraction_prompt import main; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 4: Commit**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py"
git commit -m "feat(optimize): add main optimization loop with CLI entrypoint"
```

- [ ] **Step 5: Run the optimizer (integration test)**

Run: `cd "Paper Extractor/KnowledgeGraph Extraction" && python optimize_extraction_prompt.py --max-iterations 3 --papers 2,6`

Expected output pattern:
```
Test corpus: 3 papers — ['rai14', 'micro_2', 'micro_6']
Max iterations: 3
...
BASELINE EXTRACTION
  Extracting: Rai14 is a novel interactor of Invariant chain...
    -> NN claims in XX.Xs
  ...
Baseline composite: 0.XXXX

ITERATION 1/3
  Error patterns (N total):
    [predicate_overuse] ...
  Calling optimizer (sonnet)...
  Re-extracting with modified prompt...
  ACCEPTED/REJECTED ...

OPTIMIZATION COMPLETE
  ...
Log saved to optimize/optimization_log.json
```

- [ ] **Step 6: Final commit with all files**

```bash
git add "Paper Extractor/KnowledgeGraph Extraction/prompt_versions/" \
       "Paper Extractor/KnowledgeGraph Extraction/optimize/optimization_log.json"
git commit -m "feat(optimize): first optimization run results and prompt versions"
```

---

## Self-Review

**Spec coverage check:**
- [x] Scoring function with 12 weighted metrics
- [x] Error analyzer with 9 pattern types
- [x] Experiment runner using `claude -p --model haiku`
- [x] Optimizer agent using `claude -p --model sonnet`
- [x] program.md with strategy guidance
- [x] Baseline lock (regression threshold)
- [x] No human checkpoints — fully autonomous
- [x] Prompt version snapshots
- [x] Optimization log (JSON)
- [x] Paper filtering (no reviews, no abstracts)
- [x] Convergence detection
- [x] 3 consecutive failures = stop
- [x] Max change ratio guard

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency check:**
- `score_extraction` returns `tuple[float, dict[str, float]]` — used consistently in `score_all`
- `analyze_errors` returns `list[ErrorPattern]` — consumed correctly in `call_optimizer`
- `ErrorPattern.category`, `.description`, `.severity`, `.frequency`, `.examples` — all referenced correctly
- `run_all_extractions` returns `list[dict]` — consumed by `score_all` and `analyze_errors`
- `METRIC_WEIGHTS` imported in both scoring.py and optimize_extraction_prompt.py — same source
