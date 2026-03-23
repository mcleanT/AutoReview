# Hybrid Extractor + Dual-Layer Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add factual accuracy scoring to the extraction benchmark and build a hybrid extractor that refines programmatic output with a cheap LLM pass.

**Architecture:** Two workstreams sharing the benchmark runner. Workstream 1 adds `scoring_factual.py` with number/entity/topic extraction for factual accuracy measurement. Workstream 2 adds `hybrid.py` that wraps `ProgrammaticExtractor` + LLM refinement via the existing `LLMProvider` protocol. The benchmark runner gets `--strategy` and `--alpha` flags to evaluate all 5 strategies with dual-layer scoring.

**Tech Stack:** Python 3.11+, Pydantic, structlog, sentence-transformers (scoring only), asyncio (hybrid/direct strategies), existing `LLMProvider` protocol.

**Spec:** `docs/superpowers/specs/2026-03-22-hybrid-extractor-benchmark-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `autoreview/extraction/scoring_factual.py` | Create | Factual accuracy scoring: `_extract_facts`, `_extract_numbers`, `_factual_key_findings_score`, `_factual_quant_score`, `_factual_methods_score`, `_factual_limitations_score`, `score_extraction_pair_factual` |
| `autoreview/extraction/scoring.py` | Modify | Add `compute_dual_composite()` |
| `autoreview/extraction/hybrid.py` | Create | `HybridExtractor` class: context builder, LLM refinement, JSON parsing |
| `autoreview/llm/prompts/hybrid_extraction.py` | Create | `HYBRID_SYSTEM_PROMPT`, `build_refinement_prompt()` |
| `scripts/benchmark_extractor.py` | Modify | Add `--strategy`, `--alpha`, async `run_benchmark`, dual-layer output, token tracking |
| `tests/test_extraction/test_scoring_factual.py` | Create | Unit tests for factual scoring functions |
| `tests/test_extraction/test_hybrid.py` | Create | Unit tests for `HybridExtractor` with mocked LLM |

---

## Task 1: Factual Scoring — Number and Entity Extraction

**Files:**
- Create: `autoreview/extraction/scoring_factual.py`
- Test: `tests/test_extraction/test_scoring_factual.py`

- [ ] **Step 1: Write tests for `_extract_numbers`**

```python
# tests/test_extraction/test_scoring_factual.py
"""Tests for factual accuracy scoring functions."""

from autoreview.extraction.scoring_factual import (
    _extract_numbers,
    _extract_facts,
    _fact_overlap,
)


class TestExtractNumbers:
    def test_percentages(self):
        assert _extract_numbers("achieves 67.6% accuracy") == {67.6}

    def test_multiple_numbers(self):
        nums = _extract_numbers("from 61.9% to 92.6%, comparable to 92.9%")
        assert nums == {61.9, 92.6, 92.9}

    def test_integers(self):
        assert _extract_numbers("540B parameters and 3173 questions") == {540, 3173}

    def test_no_numbers(self):
        assert _extract_numbers("no quantitative data here") == set()

    def test_empty_string(self):
        assert _extract_numbers("") == set()

    def test_none_input(self):
        assert _extract_numbers(None) == set()


class TestExtractFacts:
    def test_proper_nouns(self):
        facts = _extract_facts("Flan-PaLM achieves 67.6% on MedQA")
        assert "Flan-PaLM" in facts or "Flan" in facts
        assert "MedQA" in facts
        assert "67.6" in facts

    def test_filters_common_words(self):
        facts = _extract_facts("The model However performs well")
        assert "The" not in facts
        assert "However" not in facts

    def test_empty(self):
        assert _extract_facts("") == set()


class TestFactOverlap:
    def test_full_overlap(self):
        assert _fact_overlap({"A", "B"}, {"A", "B", "C"}) == 1.0

    def test_partial_overlap(self):
        assert _fact_overlap({"A", "B"}, {"A", "C"}) == 0.5

    def test_no_overlap(self):
        assert _fact_overlap({"A", "B"}, {"C", "D"}) == 0.0

    def test_empty_gold(self):
        assert _fact_overlap(set(), {"A"}) == 1.0

    def test_empty_pred(self):
        assert _fact_overlap({"A"}, set()) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_extraction/test_scoring_factual.py -v
```
Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement `_extract_numbers`, `_extract_facts`, `_fact_overlap`**

```python
# autoreview/extraction/scoring_factual.py
"""Factual accuracy scoring for extraction benchmark.

Complements the existing embedding-similarity scoring in scoring.py with
metrics that measure whether the correct facts (numbers, entities, topics)
were captured, regardless of wording.
"""

from __future__ import annotations

import re

import structlog

from autoreview.extraction.models import PaperExtraction

logger = structlog.get_logger()

# --- Fact extraction primitives ---

_NUMBER_RE = re.compile(r"\d+\.?\d*")

_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:[-/][A-Z]?[A-Za-z]*)*\b"
)

_STOPWORDS: frozenset[str] = frozenset({
    "The", "This", "These", "That", "Those", "However", "Moreover",
    "Furthermore", "Additionally", "Our", "We", "They", "It", "Its",
    "Here", "There", "When", "Where", "While", "Although", "Despite",
    "Between", "Among", "Within", "Without", "After", "Before",
    "During", "Since", "Both", "Each", "Every", "Many", "Most",
    "Some", "Several", "Such", "Other", "More", "Less", "First",
    "Second", "Third", "Finally", "Overall", "Specifically",
    "Notably", "Importantly", "Interestingly", "Table", "Figure",
    "Section", "Appendix", "Results", "Methods", "Discussion",
    "Introduction", "Conclusion", "Abstract",
})


def _extract_numbers(text: str | None) -> set[float]:
    """Extract all numeric values from text.

    Returns a set of floats. Percentages are returned as the bare number
    (67.6% → 67.6). Handles integers and decimals.
    """
    if not text:
        return set()
    nums: set[float] = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            continue
    return nums


def _extract_facts(text: str) -> set[str]:
    """Extract factual elements: numbers and proper nouns.

    Numbers are stored as their string representation (preserving
    original precision). Proper nouns are filtered against a stopword
    list to remove common English words that happen to be capitalized.
    """
    if not text:
        return set()
    facts: set[str] = set()
    # Numbers
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group())
    # Proper nouns (capitalized words, filtered)
    for m in _PROPER_NOUN_RE.finditer(text):
        word = m.group()
        if word not in _STOPWORDS and len(word) > 1:
            facts.add(word)
    return facts


def _fact_overlap(gold_facts: set[str], pred_facts: set[str]) -> float:
    """Fraction of gold facts found in predicted facts (recall-oriented)."""
    if not gold_facts:
        return 1.0
    if not pred_facts:
        return 0.0
    matched = sum(1 for gf in gold_facts if gf in pred_facts)
    return matched / len(gold_facts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_extraction/test_scoring_factual.py -v
```
Expected: All 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/scoring_factual.py tests/test_extraction/test_scoring_factual.py
git commit -m "feat(scoring): add factual accuracy primitives — number/entity extraction"
```

---

## Task 2: Factual Scoring — Per-Field Scoring Functions

**Files:**
- Modify: `autoreview/extraction/scoring_factual.py`
- Modify: `tests/test_extraction/test_scoring_factual.py`

- [ ] **Step 1: Write tests for per-field factual scoring**

```python
# Append to tests/test_extraction/test_scoring_factual.py

from autoreview.extraction.scoring_factual import (
    _factual_key_findings_score,
    _factual_quant_score,
    _factual_methods_score,
    _factual_limitations_score,
    score_extraction_pair_factual,
)
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)


class TestFactualKeyFindingsScore:
    def test_exact_match(self):
        gold = ["Flan-PaLM achieves 67.6% on MedQA"]
        pred = ["Flan-PaLM achieves 67.6% on MedQA dataset"]
        score = _factual_key_findings_score(pred, gold)
        assert score >= 0.8  # High overlap on facts

    def test_different_wording_same_facts(self):
        gold = ["GPT-4 scores 86.4% on USMLE"]
        pred = ["On USMLE, GPT-4 achieved an accuracy of 86.4%"]
        score = _factual_key_findings_score(pred, gold)
        assert score >= 0.8

    def test_missing_facts(self):
        gold = ["GPT-4 scores 86.4% on USMLE"]
        pred = ["The model performs well on medical exams"]
        score = _factual_key_findings_score(pred, gold)
        assert score < 0.3

    def test_empty_gold(self):
        assert _factual_key_findings_score(["claim"], []) == 0.0

    def test_empty_pred(self):
        assert _factual_key_findings_score([], ["claim"]) == 0.0

    def test_both_empty(self):
        assert _factual_key_findings_score([], []) == 1.0


class TestFactualQuantScore:
    def test_matching_numbers(self):
        score = _factual_quant_score(
            "accuracy of 67.6% surpassing 50.3%",
            "67.6% accuracy on MedQA; +17.3% over PubMedGPT (50.3%)",
        )
        assert score >= 0.5  # Found 67.6 and 50.3

    def test_no_match(self):
        score = _factual_quant_score("accuracy of 90.0%", "precision of 45.2%")
        assert score == 0.0

    def test_both_empty(self):
        assert _factual_quant_score("", "") == 1.0
        assert _factual_quant_score(None, None) == 1.0

    def test_one_empty(self):
        assert _factual_quant_score("67.6%", "") == 0.0
        assert _factual_quant_score("", "67.6%") == 0.0


class TestFactualMethodsScore:
    def test_keyword_coverage(self):
        gold = "The study uses Flan-PaLM 540B on MultiMedQA with chain-of-thought prompting"
        pred = "We evaluate Flan-PaLM on the MultiMedQA benchmark using CoT"
        score = _factual_methods_score(pred, gold)
        assert score > 0.4  # Flan-PaLM, MultiMedQA found

    def test_no_overlap(self):
        gold = "ResNet-50 trained on ImageNet using SGD"
        pred = "We built a web scraping tool in Python"
        score = _factual_methods_score(pred, gold)
        assert score < 0.2


class TestFactualLimitationsScore:
    def test_topic_match(self):
        gold = "Limited sample size of 140 questions. English-only evaluation."
        pred = "The study only evaluated on a small sample. Only English data was used."
        score = _factual_limitations_score(pred, gold)
        assert score >= 0.5  # sample_size + language_bias topics

    def test_no_topic_match(self):
        gold = "Retrospective design limits causal inference"
        pred = "The model is computationally expensive"
        score = _factual_limitations_score(pred, gold)
        assert score < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_extraction/test_scoring_factual.py -v -k "Factual"
```
Expected: ImportError on the new functions.

- [ ] **Step 3: Implement per-field factual scoring functions**

Append to `autoreview/extraction/scoring_factual.py`:

```python
# --- Limitation topic taxonomy ---

_LIMITATION_TOPICS: dict[str, list[str]] = {
    "sample_size": ["small sample", "limited sample", "few participants", "n=", "sample size"],
    "generalizability": ["not generaliz", "limited to", "single center", "specific to", "narrow scope"],
    "language_bias": ["english only", "monolingual", "english-language", "single language"],
    "study_design": ["cross-sectional", "retrospective", "correlational", "no control"],
    "data_quality": ["self-report", "missing data", "incomplete", "noisy", "annotation quality"],
    "temporal": ["outdated", "fixed date", "point in time", "snapshot", "continual learning"],
    "bias": ["selection bias", "potential bias", "confound", "unbalanced"],
    "reproducibility": ["not open-source", "not reproducible", "proprietary", "not released"],
    "scope": ["narrow scope", "beyond the scope", "did not consider", "excluded", "only consider"],
    "evaluation": ["single rater", "limited evaluation", "no human eval", "automated only"],
    "domain_specificity": ["domain-specific", "not tested on", "single domain"],
    "statistical_power": ["underpowered", "statistical power", "insufficient"],
    "cost_resources": ["expensive", "computational cost", "resource intensive"],
    "safety_ethics": ["safety", "ethical", "bias evaluation", "fairness", "equity"],
    "hallucination": ["hallucination", "factual error", "fabricat"],
    "benchmark_limitations": ["benchmark", "dataset limitation", "test set"],
    "human_subjects": ["irb", "consent", "privacy", "participant"],
    "comparison": ["limited comparison", "no baseline", "few baselines"],
    "future_work": ["future work", "further research", "remains to be"],
    "validity": ["ecological validity", "external validity", "construct validity"],
}


def _detect_limitation_topics(text: str) -> set[str]:
    """Detect which limitation topics are mentioned in text."""
    lower = text.lower()
    topics: set[str] = set()
    for topic, keywords in _LIMITATION_TOPICS.items():
        if any(kw in lower for kw in keywords):
            topics.add(topic)
    return topics


def _factual_key_findings_score(
    pred_claims: list[str],
    gold_claims: list[str],
) -> float:
    """Score key findings by factual element overlap."""
    if not gold_claims:
        return 1.0 if not pred_claims else 0.0
    if not pred_claims:
        return 0.0

    gold_facts = [_extract_facts(claim) for claim in gold_claims]
    pred_facts = [_extract_facts(claim) for claim in pred_claims]

    scores: list[float] = []
    for gf in gold_facts:
        if not gf:
            scores.append(1.0)
            continue
        # Find predicted claim with highest factual overlap
        best = max(
            (_fact_overlap(gf, pf) for pf in pred_facts),
            default=0.0,
        )
        scores.append(best)
    return sum(scores) / len(scores)


def _factual_quant_score(
    pred_quant: str | None,
    gold_quant: str | None,
) -> float:
    """Score quantitative result by numeric value match."""
    pred_nums = _extract_numbers(pred_quant)
    gold_nums = _extract_numbers(gold_quant)
    if not gold_nums and not pred_nums:
        return 1.0
    if not gold_nums or not pred_nums:
        return 0.0
    # Fraction of gold numbers found in predicted (within 1% tolerance)
    matched = sum(
        1
        for g in gold_nums
        if any(abs(g - p) / max(abs(g), 1e-9) < 0.01 for p in pred_nums)
    )
    return matched / len(gold_nums)


def _factual_methods_score(pred_text: str, gold_text: str) -> float:
    """Score methods by keyword/entity coverage.

    Extracts proper nouns and numbers from gold, checks coverage in pred.
    """
    gold_facts = _extract_facts(gold_text)
    if not gold_facts:
        return 1.0 if not pred_text.strip() else 0.5
    pred_facts = _extract_facts(pred_text)
    return _fact_overlap(gold_facts, pred_facts)


def _factual_limitations_score(pred_text: str, gold_text: str) -> float:
    """Score limitations by topic coverage."""
    gold_topics = _detect_limitation_topics(gold_text)
    if not gold_topics:
        return 1.0 if not pred_text.strip() else 0.5
    pred_topics = _detect_limitation_topics(pred_text)
    if not pred_topics:
        return 0.0
    matched = len(gold_topics & pred_topics)
    return matched / len(gold_topics)


def score_extraction_pair_factual(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
    similarity_scores: dict[str, float],
) -> dict[str, float]:
    """Compute factual accuracy scores for all fields.

    Fields with new factual metrics: key_findings, quantitative_result,
    methods_summary, limitations.
    Fields passed through from similarity: evidence_strength, study_design,
    quality_score, sample_size.

    Args:
        predicted: The extraction to evaluate.
        ground_truth: The reference extraction.
        similarity_scores: Pre-computed similarity scores for pass-through fields.
    """
    scores: dict[str, float] = {}

    # Factual metrics for key_findings
    pred_claims = [f.claim for f in predicted.key_findings]
    gold_claims = [f.claim for f in ground_truth.key_findings]
    scores["key_findings"] = _factual_key_findings_score(pred_claims, gold_claims)

    # Factual metrics for quantitative_result (average across matched pairs)
    pred_findings = predicted.key_findings
    gold_findings = ground_truth.key_findings
    quant_scores: list[float] = []
    # Simple positional pairing (gold[i] ↔ pred[i]) for quant scoring
    for i, gf in enumerate(gold_findings):
        g_q = gf.quantitative_result or ""
        # Find best matching pred by claim fact overlap
        best_q_score = 0.0
        for pf in pred_findings:
            p_q = pf.quantitative_result or ""
            qs = _factual_quant_score(p_q, g_q)
            if qs > best_q_score:
                best_q_score = qs
        quant_scores.append(best_q_score)
    scores["quantitative_result"] = (
        sum(quant_scores) / len(quant_scores) if quant_scores else 0.0
    )

    # Factual metrics for methods_summary
    scores["methods_summary"] = _factual_methods_score(
        predicted.methods_summary, ground_truth.methods_summary
    )

    # Factual metrics for limitations
    scores["limitations"] = _factual_limitations_score(
        predicted.limitations, ground_truth.limitations
    )

    # Pass-through: reuse similarity scores for unchanged fields
    for field in ("evidence_strength", "study_design", "quality_score", "sample_size"):
        scores[field] = similarity_scores.get(field, 0.0)

    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_extraction/test_scoring_factual.py -v
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/scoring_factual.py tests/test_extraction/test_scoring_factual.py
git commit -m "feat(scoring): add per-field factual accuracy scoring functions"
```

---

## Task 3: Dual-Layer Composite Scoring

**Files:**
- Modify: `autoreview/extraction/scoring.py` (add `compute_dual_composite`)
- Modify: `tests/test_extraction/test_scoring_factual.py` (add composite tests)

- [ ] **Step 1: Write test for `compute_dual_composite`**

```python
# Append to tests/test_extraction/test_scoring_factual.py

import pytest

from autoreview.extraction.scoring import compute_dual_composite


class TestDualComposite:
    def test_equal_weight(self):
        sim = {"key_findings": 0.8, "evidence_strength": 0.9, "quantitative_result": 0.3,
               "methods_summary": 0.7, "limitations": 0.6, "study_design": 0.9,
               "quality_score": 0.9, "sample_size": 0.5}
        fact = {"key_findings": 0.9, "evidence_strength": 0.9, "quantitative_result": 0.7,
                "methods_summary": 0.8, "limitations": 0.7, "study_design": 0.9,
                "quality_score": 0.9, "sample_size": 0.5}
        result = compute_dual_composite(sim, fact, alpha=0.5)
        assert "similarity" in result
        assert "factual" in result
        assert "combined" in result
        assert result["combined"] == pytest.approx(
            0.5 * result["similarity"] + 0.5 * result["factual"], abs=1e-6
        )

    def test_alpha_zero(self):
        sim = {"key_findings": 0.5, "evidence_strength": 0.5, "quantitative_result": 0.5,
               "methods_summary": 0.5, "limitations": 0.5, "study_design": 0.5,
               "quality_score": 0.5, "sample_size": 0.5}
        fact = {"key_findings": 1.0, "evidence_strength": 1.0, "quantitative_result": 1.0,
                "methods_summary": 1.0, "limitations": 1.0, "study_design": 1.0,
                "quality_score": 1.0, "sample_size": 1.0}
        result = compute_dual_composite(sim, fact, alpha=0.0)
        assert result["combined"] == pytest.approx(result["factual"], abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction/test_scoring_factual.py::TestDualComposite -v
```

- [ ] **Step 3: Add `compute_dual_composite` to `scoring.py`**

Append to `autoreview/extraction/scoring.py`:

```python
def compute_dual_composite(
    similarity_scores: dict[str, float],
    factual_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Compute dual-layer composite score.

    Args:
        similarity_scores: Per-field embedding similarity scores.
        factual_scores: Per-field factual accuracy scores (with pass-through
            fields copied from similarity_scores).
        alpha: Blend weight. 1.0 = similarity only, 0.0 = factual only.

    Returns:
        Dict with 'similarity', 'factual', and 'combined' composite scores.
    """
    sim_composite = compute_composite_score(similarity_scores)
    fact_composite = compute_composite_score(factual_scores)
    combined = alpha * sim_composite + (1 - alpha) * fact_composite
    return {
        "similarity": sim_composite,
        "factual": fact_composite,
        "combined": combined,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_extraction/test_scoring_factual.py -v
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/scoring.py tests/test_extraction/test_scoring_factual.py
git commit -m "feat(scoring): add compute_dual_composite for alpha-blended scoring"
```

---

## Task 4: Benchmark Runner — Dual-Layer Reporting

**Files:**
- Modify: `scripts/benchmark_extractor.py`

This task adds factual scoring and dual-layer reporting to the existing benchmark runner. Strategy flags come in Task 6.

- [ ] **Step 1: Add factual scoring imports and dual-layer scoring to `run_benchmark`**

In `scripts/benchmark_extractor.py`, add imports:

```python
from autoreview.extraction.scoring_factual import score_extraction_pair_factual
from autoreview.extraction.scoring import compute_dual_composite
```

- [ ] **Step 2: Modify the scoring loop in `run_benchmark`**

In the per-paper loop (after computing `scores` and `composite`), also compute factual scores:

```python
            # After: scores = score_extraction_pair_with_embeddings(...)
            # After: composite = compute_composite_score(scores)

            # Factual scoring layer
            factual_scores = score_extraction_pair_factual(
                programmatic_result, llm_extraction, similarity_scores=scores
            )
            factual_composite = compute_composite_score(factual_scores)
```

Accumulate factual scores in a parallel accumulator dict, same pattern as `field_scores_accum`.

- [ ] **Step 3: Add `--alpha` CLI argument and dual-layer output to `print_report`**

Add `--alpha` argument (default 0.5). In the results dict, include:

```python
    results["factual_composite"] = avg_factual_composite
    results["combined_composite"] = alpha * avg_composite + (1 - alpha) * avg_factual_composite
    results["avg_factual_field_scores"] = { ... }
    results["alpha"] = alpha
```

Update `print_report` to show both layers side by side.

- [ ] **Step 4: Run benchmark to verify dual-layer output works**

```bash
python scripts/benchmark_extractor.py 2>/dev/null | head -30
```
Expected: Report shows similarity composite, factual composite, and combined composite.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_extractor.py
git commit -m "feat(benchmark): add dual-layer scoring with factual accuracy metrics"
```

---

## Task 5: Hybrid Extraction Prompt

**Files:**
- Create: `autoreview/llm/prompts/hybrid_extraction.py`

- [ ] **Step 1: Write the refinement prompt module**

```python
# autoreview/llm/prompts/hybrid_extraction.py
"""Prompt templates for hybrid extraction refinement."""

from __future__ import annotations

from autoreview.extraction.models import PaperExtraction

HYBRID_SYSTEM_PROMPT = """You are an expert research analyst refining a draft paper extraction.
You receive a programmatic draft extraction alongside source excerpts from the paper.
Your job is to synthesize the draft into a polished, accurate structured extraction.

Rules:
- key_findings: Produce 8-12 synthesized claims. Combine related sentences. Include specific numbers.
- evidence_strength: strong (large-scale, RCT, meta-analysis with clear results), moderate (well-designed study with quantitative results), weak (small sample, no quantitative backing), preliminary (abstract-only, pilot, preprint).
- quantitative_result: Extract specific numbers, percentages, effect sizes. Include comparisons (X vs Y).
- methods_summary: 3-5 sentence structured summary covering approach, datasets, evaluation method.
- limitations: Numbered list of study-specific methodological limitations.
- sample_size: The primary dataset/sample size as a single integer, or null if not stated.

Output valid JSON matching the schema. Do not include fields not listed above."""


def build_refinement_prompt(
    draft: PaperExtraction,
    context: str,
) -> str:
    """Build the refinement prompt from draft extraction and source context.

    Args:
        draft: The programmatic extraction to refine.
        context: Condensed source material (abstract, top sentences, section excerpts).
    """
    # Format top 10 draft findings
    findings_lines: list[str] = []
    for i, f in enumerate(draft.key_findings[:10]):
        quant = f.quantitative_result or "none"
        findings_lines.append(
            f"  {i + 1}. [{f.evidence_strength}] {f.claim[:200]} (quant: {quant})"
        )
    draft_findings = "\n".join(findings_lines) if findings_lines else "  (no findings extracted)"

    return f"""Refine this draft paper extraction using the source material below.

## Source Material

{context}

## Draft Extraction (top 10 findings from programmatic analysis)

{draft_findings}

## Draft Methods Summary

{draft.methods_summary[:500]}

## Draft Limitations

{draft.limitations[:500]}

## Instructions

Produce a refined JSON extraction with these exact fields:
- paper_id: "{draft.paper_id}"
- key_findings: list of objects with {{claim, evidence_strength, quantitative_result, paper_id}}
- methods_summary: string (3-5 sentences)
- limitations: string (numbered list)
- study_design: one of [rct, cohort, case_control, cross_sectional, case_series, case_report, in_vitro, computational, meta_analysis, systematic_review, narrative_review, other]
- quality_score: float 0.0-1.0
- sample_size: integer or null

Output ONLY the JSON object, no markdown fences or explanation."""
```

- [ ] **Step 2: Commit**

```bash
git add autoreview/llm/prompts/hybrid_extraction.py
git commit -m "feat(prompts): add hybrid extraction refinement prompt"
```

---

## Task 6: Hybrid Extractor Class

**Files:**
- Create: `autoreview/extraction/hybrid.py`
- Test: `tests/test_extraction/test_hybrid.py`

- [ ] **Step 1: Write tests for `HybridExtractor`**

```python
# tests/test_extraction/test_hybrid.py
"""Tests for the hybrid extractor (programmatic + LLM refinement)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.hybrid import HybridExtractor
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.models.paper import CandidatePaper, ScreenedPaper


def _make_screened_paper(
    title: str = "Test Paper",
    abstract: str = "This is a test abstract about machine learning.",
    full_text: str | None = "Full text content here.",
) -> ScreenedPaper:
    paper = CandidatePaper(
        title=title,
        authors=["Author A"],
        abstract=abstract,
        full_text=full_text,
        source_database="test",
    )
    return ScreenedPaper(paper=paper, relevance_score=4, rationale="test", include=True)


def _make_valid_extraction_json(paper_id: str) -> str:
    return json.dumps({
        "paper_id": paper_id,
        "key_findings": [
            {
                "claim": "The model achieves 95% accuracy",
                "evidence_strength": "strong",
                "quantitative_result": "95% accuracy",
                "paper_id": paper_id,
            }
        ],
        "methods_summary": "We trained a neural network on ImageNet.",
        "limitations": "1. Small sample size. 2. Single dataset.",
        "study_design": "computational",
        "quality_score": 0.75,
        "sample_size": 1000,
    })


@pytest.mark.asyncio
class TestHybridExtractor:
    async def test_abstract_only_short_circuits(self):
        """Abstract-only papers skip LLM and return programmatic draft."""
        sp = _make_screened_paper(full_text=None)
        mock_llm = AsyncMock()
        config = ExtractionConfig()
        hybrid = HybridExtractor(
            programmatic=ProgrammaticExtractor(config),
            llm=mock_llm,
        )
        result = await hybrid.extract(sp)
        # LLM should NOT have been called
        mock_llm.generate_structured.assert_not_called()
        mock_llm.generate.assert_not_called()
        assert result.paper_id == sp.paper.id

    async def test_refine_with_structured_llm(self):
        """API mode: uses generate_structured for Pydantic parsing."""
        sp = _make_screened_paper()
        paper_id = sp.paper.id

        mock_response = MagicMock()
        mock_response.parsed = PaperExtraction(
            paper_id=paper_id,
            key_findings=[
                Finding(
                    claim="Refined claim",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id=paper_id,
                )
            ],
            methods_summary="Refined methods.",
            limitations="Refined limitations.",
            study_design=StudyDesign.META_ANALYSIS,  # Intentionally wrong
            quality_score=0.99,  # Intentionally different from programmatic
            sample_size=500,
        )

        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(return_value=mock_response)

        config = ExtractionConfig()
        hybrid = HybridExtractor(
            programmatic=ProgrammaticExtractor(config),
            llm=mock_llm,
        )
        result = await hybrid.extract(sp)
        assert result.key_findings[0].claim == "Refined claim"
        # study_design and quality_score should come from programmatic, NOT LLM
        assert result.study_design != StudyDesign.META_ANALYSIS
        assert result.quality_score != 0.99
        mock_llm.generate_structured.assert_called_once()

    async def test_fallback_on_parse_failure(self):
        """When LLM returns garbage, falls back to programmatic draft."""
        sp = _make_screened_paper()

        mock_llm = AsyncMock()
        # No generate_structured → falls to generate path
        del mock_llm.generate_structured
        mock_llm.generate = AsyncMock(return_value=MagicMock(content="not valid json"))

        config = ExtractionConfig()
        hybrid = HybridExtractor(
            programmatic=ProgrammaticExtractor(config),
            llm=mock_llm,
        )
        result = await hybrid.extract(sp)
        # Should fall back to draft — still has findings from programmatic
        assert len(result.key_findings) > 0
        assert result.paper_id == sp.paper.id
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_extraction/test_hybrid.py -v
```

- [ ] **Step 3: Implement `HybridExtractor`**

```python
# autoreview/extraction/hybrid.py
"""Hybrid extractor: programmatic pre-filter + LLM refinement.

Uses ProgrammaticExtractor for fast, free structured extraction, then
refines key fields (key_findings, methods_summary, limitations) with
a cheap LLM pass. study_design and quality_score are kept from the
programmatic output (already ~90% accurate).
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.extraction.truncation import ParsedSection, parse_sections
from autoreview.llm.prompts.hybrid_extraction import (
    HYBRID_SYSTEM_PROMPT,
    build_refinement_prompt,
)
from autoreview.models.paper import ScreenedPaper

logger = structlog.get_logger()


def _extract_json_block(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try to find JSON in code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


def _find_section_text(
    sections: list[ParsedSection],
    patterns: list[str],
    max_chars: int = 500,
) -> str | None:
    """Find first section matching any pattern, return truncated text."""
    for sec in sections:
        name_lower = sec.name.lower()
        if any(p in name_lower for p in patterns):
            return sec.text.strip()[:max_chars]
    return None


class HybridExtractor:
    """Refines programmatic extraction output using a cheap LLM pass.

    The programmatic extractor runs first (free, instant), producing a
    PaperExtraction with raw sentences as findings and abstract-derived
    methods/limitations. The LLM then refines this into synthesized claims
    and structured summaries.

    Fields kept from programmatic (no LLM needed):
    - study_design (~90% accuracy)
    - quality_score (~90% correlation)
    """

    def __init__(
        self,
        programmatic: ProgrammaticExtractor,
        llm: Any,
        refine_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.programmatic = programmatic
        self.llm = llm
        self.refine_model = refine_model

    async def extract(self, sp: ScreenedPaper) -> PaperExtraction:
        """Full hybrid extraction: programmatic -> LLM refinement."""
        # Step 1: Programmatic extraction (free, instant, synchronous)
        draft = self.programmatic.extract(sp)

        # Step 1.5: Short-circuit for abstract-only papers
        if not sp.paper.full_text:
            return draft

        # Step 2: Build condensed context for LLM
        context = self._build_refinement_context(draft, sp)

        # Step 3: LLM refinement
        refined = await self._refine_with_llm(draft, context, sp.paper.id)

        # Step 4: Keep programmatic fields that don't need LLM
        refined.study_design = draft.study_design
        refined.quality_score = draft.quality_score
        if refined.sample_size is None:
            refined.sample_size = draft.sample_size

        return refined

    def _build_refinement_context(
        self,
        draft: PaperExtraction,
        sp: ScreenedPaper,
    ) -> str:
        """Build condensed context for LLM refinement (~2K tokens)."""
        paper = sp.paper
        parts: list[str] = []

        parts.append(f"Title: {paper.title}")

        if paper.abstract:
            parts.append(f"Abstract: {paper.abstract}")

        # Top 10 key findings from programmatic extraction
        parts.append("Top extracted sentences:")
        for i, f in enumerate(draft.key_findings[:10]):
            quant = f" [{f.quantitative_result}]" if f.quantitative_result else ""
            parts.append(f"  {i + 1}. {f.claim[:200]}{quant}")

        # Section excerpts
        sections = parse_sections(paper.full_text) if paper.full_text else []
        methods_text = _find_section_text(
            sections, ["method", "approach", "experimental", "implementation"]
        )
        if methods_text:
            parts.append(f"Methods section excerpt: {methods_text}")

        lim_text = _find_section_text(
            sections, ["limitation", "shortcoming", "weakness"]
        )
        if lim_text:
            parts.append(f"Limitations section excerpt: {lim_text}")

        return "\n\n".join(parts)

    async def _refine_with_llm(
        self,
        draft: PaperExtraction,
        context: str,
        paper_id: str,
    ) -> PaperExtraction:
        """Send condensed context to LLM and parse structured output."""
        prompt = build_refinement_prompt(draft, context)

        if hasattr(self.llm, "generate_structured"):
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=PaperExtraction,
                system=HYBRID_SYSTEM_PROMPT,
                model_override=self.refine_model,
            )
            extraction = response.parsed
        else:
            response = await self.llm.generate(
                prompt=prompt,
                system=HYBRID_SYSTEM_PROMPT,
                model_override=self.refine_model,
            )
            json_str = _extract_json_block(response.content)
            try:
                extraction = PaperExtraction.model_validate_json(json_str)
            except Exception:
                logger.warning(
                    "hybrid_refinement.parse_failed",
                    paper_id=paper_id,
                )
                return draft

        extraction.paper_id = paper_id
        return extraction

    async def extract_batch(
        self,
        papers: list[ScreenedPaper],
    ) -> tuple[list[PaperExtraction], list]:
        """Extract from a batch of papers."""
        from autoreview.extraction.extractor import ExtractionFailure

        extractions: list[PaperExtraction] = []
        failures: list[ExtractionFailure] = []
        for sp in papers:
            try:
                result = await self.extract(sp)
                extractions.append(result)
            except Exception as exc:
                logger.error(
                    "hybrid_extraction.failed",
                    paper_id=sp.paper.id,
                    error=str(exc),
                )
                failures.append(
                    ExtractionFailure(paper_id=sp.paper.id, error=str(exc))
                )
        return extractions, failures
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_extraction/test_hybrid.py -v
```
Expected: All 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/hybrid.py tests/test_extraction/test_hybrid.py
git commit -m "feat(extraction): add HybridExtractor — programmatic + LLM refinement"
```

---

## Task 7: Benchmark Runner — Strategy Support

**Files:**
- Modify: `scripts/benchmark_extractor.py`

This task adds `--strategy` flag to run different extraction strategies (A/B/C/D) through the benchmark.

- [ ] **Step 1: Add strategy argument and async support**

Add CLI argument:
```python
parser.add_argument(
    "--strategy",
    choices=["programmatic", "hybrid-haiku", "hybrid-sonnet", "direct-haiku"],
    default="programmatic",
    help="Extraction strategy to benchmark",
)
```

- [ ] **Step 2: Add extractor factory based on strategy**

```python
async def _create_extractor(strategy: str, config: ExtractionConfig):
    """Create the appropriate extractor for the given strategy."""
    if strategy == "programmatic":
        return ProgrammaticExtractor(config), False  # (extractor, is_async)

    # LLM-based strategies need a provider
    from autoreview.llm.claude import ClaudeLLMProvider

    if strategy == "hybrid-haiku":
        llm = ClaudeLLMProvider()
        programmatic = ProgrammaticExtractor(config)
        return HybridExtractor(programmatic, llm, refine_model="claude-haiku-4-5-20251001"), True
    elif strategy == "hybrid-sonnet":
        llm = ClaudeLLMProvider()
        programmatic = ProgrammaticExtractor(config)
        return HybridExtractor(programmatic, llm, refine_model="claude-sonnet-4-6-20250514"), True
    elif strategy == "direct-haiku":
        from autoreview.extraction.extractor import PaperExtractor
        llm = ClaudeLLMProvider()
        return PaperExtractor(llm, model_override="claude-haiku-4-5-20251001"), True
```

- [ ] **Step 3: Make `run_benchmark` async and handle both sync/async extractors**

Convert `run_benchmark` to `async def run_benchmark(...)`. In the per-paper loop:

```python
    if is_async:
        programmatic_result = await extractor.extract(sp)
    else:
        programmatic_result = extractor.extract(sp)
```

Wrap `main()` with `asyncio.run()`.

- [ ] **Step 4: Add token tracking for LLM strategies**

Track input/output tokens from LLM responses and include in results:

```python
    results["total_input_tokens"] = total_input_tokens
    results["total_output_tokens"] = total_output_tokens
    results["actual_cost_usd"] = _estimate_cost(strategy, total_input_tokens, total_output_tokens)
```

- [ ] **Step 5: Test with programmatic strategy (should match existing behavior)**

```bash
python scripts/benchmark_extractor.py --strategy programmatic 2>/dev/null | tail -20
```
Expected: Same results as before (composite ~0.6955).

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_extractor.py
git commit -m "feat(benchmark): add --strategy flag with async support and token tracking"
```

---

## Task 8: Integration Test — Run Hybrid on Sample Papers

**Files:**
- Create: `tests/test_extraction/test_hybrid_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_extraction/test_hybrid_integration.py
"""Integration test: run hybrid extractor on corpus papers with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.hybrid import HybridExtractor
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from scripts.benchmark_extractor import (
    build_candidate_paper,
    build_screened_paper,
    load_corpus,
)

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "extraction_corpus"


@pytest.mark.asyncio
@pytest.mark.skipif(not CORPUS_DIR.exists(), reason="corpus not available")
async def test_hybrid_on_corpus_papers():
    """Run hybrid extractor on 3 real corpus papers with mocked LLM."""
    corpus = load_corpus(CORPUS_DIR)[:3]
    config = ExtractionConfig()
    programmatic = ProgrammaticExtractor(config)

    # Mock LLM that returns a valid extraction JSON
    mock_llm = AsyncMock()

    def make_response(prompt, **kwargs):
        # Extract paper_id from prompt
        resp = MagicMock()
        resp.parsed = PaperExtraction(
            paper_id="test",
            key_findings=[],
            methods_summary="Mocked methods.",
            limitations="Mocked limitations.",
        )
        return resp

    mock_llm.generate_structured = AsyncMock(side_effect=make_response)

    hybrid = HybridExtractor(programmatic, mock_llm)

    for entry in corpus:
        candidate = build_candidate_paper(entry)
        screened = build_screened_paper(candidate)
        result = await hybrid.extract(screened)

        assert isinstance(result, PaperExtraction)
        assert result.paper_id == candidate.id
        # study_design and quality_score should come from programmatic
        assert result.study_design is not None
        assert result.quality_score is not None
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_extraction/test_hybrid_integration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_extraction/test_hybrid_integration.py
git commit -m "test(hybrid): add integration test with corpus papers"
```

---

## Task 9: Run Comparison Matrix

This task runs the actual benchmarks. Requires API keys for LLM strategies.

- [ ] **Step 1: Run Strategy A (programmatic) with dual scoring**

```bash
python scripts/benchmark_extractor.py --strategy programmatic 2>/dev/null | tee results/extractor_benchmark/strategy_A.txt
```

- [ ] **Step 2: Run Strategy B (hybrid-haiku)**

```bash
python scripts/benchmark_extractor.py --strategy hybrid-haiku 2>/dev/null | tee results/extractor_benchmark/strategy_B.txt
```

- [ ] **Step 3: Run Strategy C (hybrid-sonnet)**

```bash
python scripts/benchmark_extractor.py --strategy hybrid-sonnet 2>/dev/null | tee results/extractor_benchmark/strategy_C.txt
```

- [ ] **Step 4: Run Strategy D (direct-haiku)**

```bash
python scripts/benchmark_extractor.py --strategy direct-haiku 2>/dev/null | tee results/extractor_benchmark/strategy_D.txt
```

- [ ] **Step 5: Compile comparison matrix**

Combine all results into `results/extractor_benchmark/comparison_matrix.json` and generate a summary table comparing all 5 strategies across both scoring layers.

- [ ] **Step 6: Commit results**

```bash
git add results/extractor_benchmark/
git commit -m "data: add 5-strategy comparison matrix with dual-layer scoring"
```

---

## Dependency Graph

```
Task 1 (fact extraction primitives)
  ↓
Task 2 (per-field factual scoring)
  ↓
Task 3 (dual composite) ──────────────┐
  ↓                                    │
Task 4 (benchmark dual-layer)          │
                                       │
Task 5 (refinement prompt)             │
  ↓                                    │
Task 6 (HybridExtractor) ─────────────┤
  ↓                                    │
Task 7 (benchmark --strategy) ←────────┘
  ↓
Task 8 (integration test)
  ↓
Task 9 (run comparison matrix)
```

Tasks 1-3 and Task 5 can proceed in parallel. Tasks 4 and 6 depend on their predecessors. Task 7 merges both workstreams. Tasks 8-9 are sequential at the end.
