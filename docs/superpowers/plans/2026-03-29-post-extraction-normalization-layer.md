# Post-Extraction Claim Normalization Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-extraction normalization layer that cleans entity names, fixes predicates, decomposes compound objects, and backfills quantitative context — improving cross-paper claim matching for contradiction detection.

**Architecture:** New `normalize.py` module with a `ClaimNormalizer` class exposing two entry points: `pre_dedup()` (text cleaning + decomposition, runs before entity dedup) and `post_dedup()` (quantitative backfill, runs after dedup but before merge). Pipeline integration via two new params on `build_graph()`: `normalize` and `llm_decompose`.

**Tech Stack:** Python 3.11+, regex, structlog, Pydantic (via AutoReviewModel), rapidfuzz (already a dependency), pytest

**Spec:** `docs/superpowers/specs/2026-03-29-post-extraction-normalization-layer.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `autoreview/knowledge_graph/normalize.py` | Create | All normalization transforms, `ClaimNormalizer`, `NormalizationReport` |
| `tests/test_knowledge_graph/test_normalize.py` | Create | Unit tests for all transforms + `ClaimNormalizer` |
| `autoreview/knowledge_graph/__init__.py` | Modify (lines 128-133, 169-240) | Add `normalize`/`llm_decompose` params, insert normalization passes |

---

### Task 1: Text Cleaning

**Files:**
- Create: `autoreview/knowledge_graph/normalize.py`
- Create: `tests/test_knowledge_graph/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_knowledge_graph/test_normalize.py
"""Tests for post-extraction claim normalization."""

from __future__ import annotations


class TestTextCleaning:
    """Tests for entity name text cleaning transforms."""

    def test_strip_parenthetical_synonym(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("Invariant chain (CD74)")
        assert name == "Invariant chain"
        assert "CD74" in aliases

    def test_strip_multiple_parentheticals(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("T (Brachyury) (TBXT)")
        assert name == "T"
        assert "Brachyury" in aliases
        assert "TBXT" in aliases

    def test_strip_leading_article(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("the Wnt signaling pathway")
        assert name == "Wnt signaling"

    def test_strip_multiple_leading_articles(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("these cell populations")
        assert name == "cell populations"

    def test_collapse_whitespace(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("mesoderm   differentiation\t process")
        assert name == "mesoderm differentiation"

    def test_strip_trailing_descriptor(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("Wnt signaling pathway")
        assert name == "Wnt signaling"

    def test_no_strip_trailing_descriptor_if_too_short(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("Wnt pathway")
        assert name == "Wnt pathway"

    def test_preserves_clean_name(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("BMP4")
        assert name == "BMP4"
        assert aliases == []

    def test_combined_transforms(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("the BMP4 (bone morphogenetic protein 4) signaling cascade")
        assert name == "BMP4 signaling"
        assert "bone morphogenetic protein 4" in aliases

    def test_empty_string(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("")
        assert name == ""
        assert aliases == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestTextCleaning -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoreview.knowledge_graph.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# autoreview/knowledge_graph/normalize.py
"""Post-extraction claim normalization for improved cross-paper matching."""

from __future__ import annotations

import re

_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")

_LEADING_ARTICLES = frozenset({"the", "a", "an", "this", "these", "that", "those"})

_TRAILING_DESCRIPTORS = frozenset({
    "process",
    "pathway",
    "mechanism",
    "activity",
    "event",
    "response",
    "cascade",
    "system",
})


def _strip_parentheticals(name: str) -> tuple[str, list[str]]:
    """Remove parenthetical content, returning cleaned name + extracted aliases."""
    aliases = [m.strip("() ") for m in _PARENTHETICAL_RE.findall(name)]
    cleaned = _PARENTHETICAL_RE.sub("", name).strip()
    return cleaned, [a for a in aliases if a]


def _strip_leading_articles(name: str) -> str:
    """Remove leading articles/qualifiers."""
    words = name.split()
    while words and words[0].lower() in _LEADING_ARTICLES:
        words.pop(0)
    return " ".join(words) if words else name


def _collapse_whitespace(name: str) -> str:
    """Normalize whitespace to single spaces."""
    return re.sub(r"\s+", " ", name).strip()


def _strip_trailing_descriptors(name: str) -> str:
    """Remove trailing generic descriptors if >2 words remain after removal."""
    words = name.split()
    if len(words) > 2 and words[-1].lower() in _TRAILING_DESCRIPTORS:
        return " ".join(words[:-1])
    return name


def clean_entity_name(name: str) -> tuple[str, list[str]]:
    """Apply all text cleaning transforms.

    Returns:
        Tuple of (cleaned_name, extracted_aliases).
    """
    if not name:
        return name, []
    cleaned, aliases = _strip_parentheticals(name)
    cleaned = _strip_leading_articles(cleaned)
    cleaned = _collapse_whitespace(cleaned)
    cleaned = _strip_trailing_descriptors(cleaned)
    return cleaned, aliases
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestTextCleaning -v`
Expected: 10 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add text cleaning transforms for entity name normalization"
```

---

### Task 2: Predicate String Cleaning

**Files:**
- Modify: `autoreview/knowledge_graph/normalize.py`
- Modify: `tests/test_knowledge_graph/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
class TestPredicateCleaning:
    """Tests for predicate string cleaning transforms."""

    def test_strip_trailing_punctuation(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("induces.") == "induces"
        assert clean_predicate("inhibits;") == "inhibits"

    def test_collapse_internal_whitespace(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("activ ates") == "activates"

    def test_underscore_normalization(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("is required for") == "is_required_for"
        assert clean_predicate("is located in") == "is_located_in"
        assert clean_predicate("interacts with") == "interacts_with"

    def test_tense_past(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibited") == "inhibits"
        assert clean_predicate("induced") == "induces"
        assert clean_predicate("promoted") == "promotes"

    def test_tense_gerund(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibiting") == "inhibits"
        assert clean_predicate("inducing") == "induces"

    def test_already_canonical(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("induces") == "induces"
        assert clean_predicate("is_required_for") == "is_required_for"

    def test_combined_cleanup_then_tense(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibited.") == "inhibits"

    def test_unknown_predicate_passthrough(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("some_unknown_pred") == "some_unknown_pred"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestPredicateCleaning -v`
Expected: FAIL with `ImportError: cannot import name 'clean_predicate'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/normalize.py`:

```python
_PREDICATE_TENSE_MAP: dict[str, str] = {
    "induced": "induces",
    "inducing": "induces",
    "inhibited": "inhibits",
    "inhibiting": "inhibits",
    "activated": "activates",
    "activating": "activates",
    "promoted": "promotes",
    "promoting": "promotes",
    "regulated": "regulates",
    "regulating": "regulates",
    "expressed": "expresses",
    "expressing": "expresses",
    "suppressed": "suppresses",
    "suppressing": "suppresses",
    "blocked": "blocks",
    "blocking": "blocks",
    "degraded": "degrades",
    "degrading": "degrades",
    "stabilized": "stabilizes",
    "stabilizing": "stabilizes",
    "phosphorylated": "phosphorylates",
    "phosphorylating": "phosphorylates",
    "transported": "transports",
    "transporting": "transports",
    "converted": "converts",
    "converting": "converts",
    "maintained": "maintains",
    "maintaining": "maintains",
    "modified": "modifies",
    "modifying": "modifies",
}

_KNOWN_PREDICATES: frozenset[str] | None = None


def _get_known_predicates() -> frozenset[str]:
    """Lazily build the set of all known predicate strings (canonical + synonyms)."""
    global _KNOWN_PREDICATES  # noqa: PLW0603
    if _KNOWN_PREDICATES is None:
        from autoreview.knowledge_graph.dedup import PredicateNormalizer

        table = PredicateNormalizer.SYNONYM_TABLE
        _KNOWN_PREDICATES = frozenset(table.keys()) | frozenset(table.values())
    return _KNOWN_PREDICATES


def clean_predicate(predicate: str) -> str:
    """Fix malformed predicates: whitespace, punctuation, tense, underscores.

    Runs BEFORE PredicateNormalizer — fixes surface form so synonym lookup hits.
    """
    # 1. Strip trailing punctuation and collapse whitespace
    cleaned = re.sub(r"[.,;:!?]+$", "", predicate)
    cleaned = _collapse_whitespace(cleaned)

    # 2. Tense normalization
    if cleaned in _PREDICATE_TENSE_MAP:
        cleaned = _PREDICATE_TENSE_MAP[cleaned]

    # 3. Underscore normalization — if spaces->underscores matches a known predicate
    if " " in cleaned:
        underscored = cleaned.replace(" ", "_")
        if underscored in _get_known_predicates():
            cleaned = underscored

    return cleaned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestPredicateCleaning -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add predicate string cleaning with tense and underscore normalization"
```

---

### Task 3: Rule-Based Compound Object Decomposition

**Files:**
- Modify: `autoreview/knowledge_graph/normalize.py`
- Modify: `tests/test_knowledge_graph/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
class TestCompoundDecomposition:
    """Tests for rule-based compound object decomposition."""

    def test_conjunction_with_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("endoderm and mesoderm differentiation")
        assert len(result) == 2
        assert "endoderm differentiation" in result
        assert "mesoderm differentiation" in result

    def test_comma_and_list_with_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("T, Sox2, and Nanog expression levels")
        assert len(result) == 3
        assert any("T" in r and "expression" in r for r in result)
        assert any("Sox2" in r and "expression" in r for r in result)
        assert any("Nanog" in r and "expression" in r for r in result)

    def test_slash_split_with_tail(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("BMP4/WNT signaling in gastruloids")
        assert len(result) == 2
        assert any("BMP4" in r for r in result)
        assert any("WNT" in r for r in result)

    def test_prepositional_compound(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("self-organization into endoderm and mesoderm")
        assert len(result) == 2
        assert "endoderm" in result
        assert "mesoderm" in result

    def test_short_object_not_decomposed(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("mesoderm differentiation")
        assert result == ["mesoderm differentiation"]

    def test_three_word_object_not_decomposed(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("lateral plate mesoderm")
        assert result == ["lateral plate mesoderm"]

    def test_no_pattern_match_returns_original(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("mesoderm differentiation in mouse gastruloids")
        assert result == ["mesoderm differentiation in mouse gastruloids"]

    def test_conjunction_without_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("lateral mesoderm and neural crest")
        assert len(result) == 2
        assert "lateral mesoderm" in result
        assert "neural crest" in result

    def test_flag_for_llm(self):
        from autoreview.knowledge_graph.normalize import flag_for_llm_decomposition

        assert flag_for_llm_decomposition("self-organization of human gastruloids into homogenous subpopulations of endoderm and mesoderm") is True
        assert flag_for_llm_decomposition("mesoderm differentiation") is False

    def test_slash_not_in_units(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        # "ng/mL" should not be split — only single-word slash patterns
        result = decompose_object("10 ng/mL BMP4 treatment effect")
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestCompoundDecomposition -v`
Expected: FAIL with `ImportError: cannot import name 'decompose_object'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/normalize.py`:

```python
_BIOLOGICAL_HEAD_NOUNS = frozenset({
    "differentiation",
    "expression",
    "signaling",
    "pathway",
    "formation",
    "specification",
    "development",
    "migration",
    "proliferation",
    "apoptosis",
    "activation",
    "inhibition",
    "regulation",
    "transcription",
    "phosphorylation",
    "degradation",
    "maturation",
    "polarization",
    "morphogenesis",
    "patterning",
    "induction",
    "repression",
    "production",
    "secretion",
    "localization",
    "maintenance",
    "organization",
    "subpopulations",
    "populations",
    "cells",
    "levels",
})

_CONJUNCTION_SPLIT_RE = re.compile(r",\s*(?:and|or)\s+|\s+(?:and|or)\s+")
_PREP_RE = re.compile(r"^.*?\b(?:into|of|in|from|to|through|via)\s+(.+)$", re.IGNORECASE)


def _find_head_noun(text: str) -> str | None:
    """Find a trailing biological head noun that distributes across conjuncts."""
    words = text.split()
    if words and words[-1].lower() in _BIOLOGICAL_HEAD_NOUNS:
        return words[-1]
    # Check last two words for compound head nouns like "expression levels"
    if len(words) >= 2 and words[-1].lower() in _BIOLOGICAL_HEAD_NOUNS:
        return words[-1]
    return None


def _decompose_slash(text: str) -> list[str] | None:
    """Split 'BMP4/WNT signaling' -> ['BMP4 signaling', 'WNT signaling'].

    Only splits single-token slash patterns (not units like ng/mL).
    """
    if "/" not in text:
        return None
    words = text.split()
    # Find the word containing /
    slash_idx = None
    for i, w in enumerate(words):
        if "/" in w:
            parts = w.split("/")
            # Only split if both sides are single tokens (not units like ng/mL)
            if len(parts) == 2 and all(len(p) > 1 and p[0].isupper() for p in parts):
                slash_idx = i
                break
    if slash_idx is None:
        return None
    slash_word = words[slash_idx]
    parts = slash_word.split("/")
    prefix = " ".join(words[:slash_idx])
    suffix = " ".join(words[slash_idx + 1 :])
    result = []
    for part in parts:
        tokens = [t for t in [prefix, part, suffix] if t]
        result.append(" ".join(tokens))
    return result


def _decompose_conjunction(text: str) -> list[str] | None:
    """Split on 'and'/'or' with head noun distribution."""
    if " and " not in text and " or " not in text:
        return None
    head = _find_head_noun(text)
    parts = _CONJUNCTION_SPLIT_RE.split(text)
    parts = [p.strip().rstrip(",").strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return None
    result = []
    for part in parts:
        part_words = part.split()
        # If head noun exists and part doesn't already end with it, append it
        if head and not (part_words and part_words[-1].lower() == head.lower()):
            result.append(f"{part} {head}")
        else:
            result.append(part)
    return result


def _decompose_prepositional(text: str) -> list[str] | None:
    """Strip prepositional wrappers, decompose inner content.

    'self-organization into endoderm and mesoderm' -> ['endoderm', 'mesoderm']
    """
    m = _PREP_RE.match(text)
    if not m:
        return None
    inner = m.group(1)
    # Try conjunction decomposition on inner content
    result = _decompose_conjunction(inner)
    if result:
        return result
    return None


def decompose_object(object_name: str) -> list[str]:
    """Decompose a compound object into atomic concepts.

    Returns a list of 1+ strings. Single-element list means no decomposition.
    """
    if len(object_name.split()) <= 3:
        return [object_name]
    for fn in (_decompose_slash, _decompose_conjunction, _decompose_prepositional):
        result = fn(object_name)
        if result and len(result) > 1:
            return result
    return [object_name]


def flag_for_llm_decomposition(object_name: str) -> bool:
    """Check if an undecomposed object should be sent to LLM fallback."""
    return len(object_name.split()) > 6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestCompoundDecomposition -v`
Expected: 10 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add rule-based compound object decomposition"
```

---

### Task 4: LLM Fallback for Compound Decomposition

**Files:**
- Modify: `autoreview/knowledge_graph/normalize.py`
- Modify: `tests/test_knowledge_graph/test_normalize.py`

**Context:** The LLM fallback is an async function injected via the `ClaimNormalizer` constructor. This allows tests to mock it and production code to wire in the actual LLM client. The function signature is `Callable[[list[str]], Awaitable[list[list[str]]]]` — takes a batch of object strings, returns a list of decompositions (each a list of atomic concepts).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
import asyncio


class TestLLMDecomposition:
    """Tests for LLM fallback decomposition with mocked LLM."""

    def test_llm_fallback_decomposes_verbose_object(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [["endoderm differentiation", "mesoderm differentiation"]]

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(
                ["self-organization of human gastruloids into homogenous subpopulations of endoderm and mesoderm"],
                mock_llm,
            )
        )
        assert result == [["endoderm differentiation", "mesoderm differentiation"]]

    def test_llm_fallback_atomic_passthrough(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [["mesoderm differentiation"]]

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["mesoderm differentiation"], mock_llm)
        )
        assert result == [["mesoderm differentiation"]]

    def test_llm_fallback_batch(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [
                ["endoderm", "mesoderm"],
                ["neural crest migration", "neural tube closure"],
            ]

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["obj1 long enough words", "obj2 long enough words"], mock_llm)
        )
        assert len(result) == 2
        assert result[0] == ["endoderm", "mesoderm"]
        assert result[1] == ["neural crest migration", "neural tube closure"]

    def test_llm_fallback_none_fn_returns_originals(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["some verbose object name here"], None)
        )
        assert result == [["some verbose object name here"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestLLMDecomposition -v`
Expected: FAIL with `ImportError: cannot import name 'llm_decompose_objects'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/normalize.py` (add `from collections.abc import Awaitable, Callable` to imports):

```python
from collections.abc import Awaitable, Callable

LLMDecomposeFn = Callable[[list[str]], Awaitable[list[list[str]]]]


async def llm_decompose_objects(
    objects: list[str],
    llm_fn: LLMDecomposeFn | None,
) -> list[list[str]]:
    """Decompose objects using an LLM function.

    Args:
        objects: List of verbose object strings to decompose.
        llm_fn: Async function that takes a batch of strings and returns
            decompositions. If None, returns each object as a single-element list.

    Returns:
        List of decompositions, one per input object.
    """
    if llm_fn is None:
        return [[obj] for obj in objects]
    return await llm_fn(objects)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestLLMDecomposition -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add LLM fallback for compound object decomposition"
```

---

### Task 5: Quantitative Context Backfill

**Files:**
- Modify: `autoreview/knowledge_graph/normalize.py`
- Modify: `tests/test_knowledge_graph/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
class TestQuantitativeBackfill:
    """Tests for extracting quantitative context from natural language text."""

    def test_extract_concentration(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces mesoderm differentiation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"

    def test_extract_timepoint(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Mesoderm markers appear at 48h of culture",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "48h"

    def test_normalize_time_units(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Expression peaks at 72 hours post-treatment",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "72h"

    def test_extract_dose(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Animals received 5 mg/kg of the compound",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["dose"] == "5 mg/kg"

    def test_extract_multiple_fields(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces T expression at 48h",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"
        assert assertion["quantitative_context"]["timepoint"] == "48h"

    def test_no_overwrite_existing(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces mesoderm at 48h",
            "quantitative_context": {
                "concentration": "5 ng/mL",
                "timepoint": None,
                "dose": None,
            },
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        # Existing concentration preserved
        assert assertion["quantitative_context"]["concentration"] == "5 ng/mL"
        # Null timepoint backfilled
        assert assertion["quantitative_context"]["timepoint"] == "48h"

    def test_no_match_returns_false(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 induces mesoderm differentiation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is False
        assert assertion["quantitative_context"] is None

    def test_fallback_to_treatment(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 induces mesoderm",
            "quantitative_context": None,
            "conditions": {"treatment": ["10 ng/mL BMP4"]},
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"

    def test_day_timepoint(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "At day 5 gastruloids show elongation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "5d"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestQuantitativeBackfill -v`
Expected: FAIL with `ImportError: cannot import name 'backfill_quantitative_context'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/normalize.py`:

```python
_CONCENTRATION_RE = re.compile(
    r"(\d+\.?\d*)\s*(ng/mL|µM|nM|mM|µg/mL|μM|μg/mL)", re.IGNORECASE
)
_TIMEPOINT_RE = re.compile(
    r"(?:at\s+)?(\d+\.?\d*)\s*(h|hr|hrs|hours|d|days|min|minutes)\b", re.IGNORECASE
)
_DAY_TIMEPOINT_RE = re.compile(r"(?:at\s+)?day\s+(\d+\.?\d*)", re.IGNORECASE)
_DOSE_RE = re.compile(r"(\d+\.?\d*)\s*(mg/kg|µg/kg|μg/kg|Gy|U/mL)", re.IGNORECASE)

_TIME_UNIT_MAP: dict[str, str] = {
    "hr": "h",
    "hrs": "h",
    "hours": "h",
    "h": "h",
    "days": "d",
    "d": "d",
    "minutes": "min",
    "min": "min",
}


def _extract_concentration(text: str) -> str | None:
    m = _CONCENTRATION_RE.search(text)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _extract_timepoint(text: str) -> str | None:
    # Check "day N" pattern first
    m = _DAY_TIMEPOINT_RE.search(text)
    if m:
        val = m.group(1)
        return f"{val}d"
    m = _TIMEPOINT_RE.search(text)
    if m:
        val = m.group(1)
        unit = _TIME_UNIT_MAP.get(m.group(2).lower(), m.group(2))
        return f"{val}{unit}"
    return None


def _extract_dose(text: str) -> str | None:
    m = _DOSE_RE.search(text)
    return f"{m.group(1)} {m.group(2)}" if m else None


def backfill_quantitative_context(assertion: dict) -> bool:
    """Parse quantitative values from natural_language when quantitative_context is null.

    Modifies the assertion dict in place. Returns True if any field was backfilled.
    """
    qc = assertion.get("quantitative_context")
    all_null = qc is None or (
        isinstance(qc, dict)
        and qc.get("concentration") is None
        and qc.get("timepoint") is None
        and qc.get("dose") is None
    )
    if not all_null and qc is not None:
        # Existing qc with some values — only backfill null fields
        texts = [assertion.get("natural_language", "")]
        conditions = assertion.get("conditions") or {}
        treatments = conditions.get("treatment", [])
        if treatments:
            texts.extend(treatments)
        combined = " ".join(texts)

        changed = False
        if qc.get("concentration") is None:
            val = _extract_concentration(combined)
            if val:
                qc["concentration"] = val
                changed = True
        if qc.get("timepoint") is None:
            val = _extract_timepoint(combined)
            if val:
                qc["timepoint"] = val
                changed = True
        if qc.get("dose") is None:
            val = _extract_dose(combined)
            if val:
                qc["dose"] = val
                changed = True
        return changed

    # qc is None or all fields null — try to build from scratch
    texts = [assertion.get("natural_language", "")]
    conditions = assertion.get("conditions") or {}
    treatments = conditions.get("treatment", [])
    if treatments:
        texts.extend(treatments)
    combined = " ".join(texts)

    concentration = _extract_concentration(combined)
    timepoint = _extract_timepoint(combined)
    dose = _extract_dose(combined)

    if concentration is None and timepoint is None and dose is None:
        return False

    assertion["quantitative_context"] = {
        "concentration": concentration,
        "timepoint": timepoint,
        "dose": dose,
    }
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestQuantitativeBackfill -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add quantitative context backfill from natural language text"
```

---

### Task 6: ClaimNormalizer Class and NormalizationReport

**Files:**
- Modify: `autoreview/knowledge_graph/normalize.py`
- Modify: `tests/test_knowledge_graph/test_normalize.py`

**Context:** This task ties all transforms into the `ClaimNormalizer` class. The `pre_dedup` method takes both entities and assertions (deviation from spec which says `pre_dedup(claims)` — needed because text cleaning must keep entity `canonical_name` and assertion `subject_canonical_name`/`object_canonical_name` in sync, and decomposition must create new entity entries). The `post_dedup` method takes assertions only, matching the spec.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
class TestClaimNormalizer:
    """Tests for the ClaimNormalizer orchestrator."""

    def _make_entity(self, name: str, entity_type: str = "biological_process", paper_id: str = "p1") -> dict:
        return {
            "canonical_name": name,
            "entity_type": entity_type,
            "ontology_id": None,
            "ontology_source": None,
            "aliases": [],
            "paper_ids": [paper_id],
        }

    def _make_assertion(
        self,
        subject: str,
        obj: str,
        predicate: str = "induces",
        draft_id: str = "a_001",
        natural_language: str = "",
    ) -> dict:
        return {
            "draft_id": draft_id,
            "subject_canonical_name": subject,
            "object_canonical_name": obj,
            "predicate": predicate,
            "direction": "positive",
            "assertion_type": "mechanistic_causal",
            "evidence_unit_ids": ["e_001"],
            "paper_id": "p1",
            "publication_date": "2023-01-15",
            "natural_language": natural_language,
            "quantitative_context": None,
            "conditions": None,
            "model_system": None,
            "organism": None,
            "in_vitro": None,
        }

    def test_pre_dedup_text_cleaning(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [self._make_entity("the Wnt signaling pathway")]
        assertions = [self._make_assertion("BMP4", "the Wnt signaling pathway")]

        new_ents, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        assert new_ents[0]["canonical_name"] == "Wnt signaling"
        assert new_asserts[0]["object_canonical_name"] == "Wnt signaling"
        assert report.text_cleaned >= 1

    def test_pre_dedup_predicate_cleaning(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("mesoderm differentiation"),
        ]
        assertions = [self._make_assertion("BMP4", "mesoderm differentiation", predicate="promoted.")]

        _, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        assert new_asserts[0]["predicate"] == "promotes"
        assert report.predicates_cleaned == 1

    def test_pre_dedup_decomposition(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("endoderm and mesoderm differentiation"),
        ]
        assertions = [
            self._make_assertion("BMP4", "endoderm and mesoderm differentiation", draft_id="a_001"),
        ]

        new_ents, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        # Original assertion decomposed into 2
        assert len(new_asserts) == 2
        obj_names = {a["object_canonical_name"] for a in new_asserts}
        assert "endoderm differentiation" in obj_names
        assert "mesoderm differentiation" in obj_names
        # New entities created
        ent_names = {e["canonical_name"] for e in new_ents}
        assert "endoderm differentiation" in ent_names
        assert "mesoderm differentiation" in ent_names
        assert report.claims_decomposed == 1
        assert report.claims_produced == 2

    def test_pre_dedup_decomposed_claim_audit_trail(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("endoderm and mesoderm differentiation"),
        ]
        assertions = [
            self._make_assertion("BMP4", "endoderm and mesoderm differentiation", draft_id="a_001"),
        ]

        _, new_asserts, _ = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        for a in new_asserts:
            assert a["_decomposed_from"] == "a_001"
            assert a["draft_id"].startswith("a_001_d")

    def test_post_dedup_quantitative_backfill(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        assertions = [
            self._make_assertion(
                "BMP4",
                "mesoderm differentiation",
                natural_language="BMP4 at 10 ng/mL induces mesoderm at 48h",
            ),
        ]

        new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.post_dedup(assertions)
        )
        assert new_asserts[0]["quantitative_context"]["concentration"] == "10 ng/mL"
        assert new_asserts[0]["quantitative_context"]["timepoint"] == "48h"
        assert report.quant_backfilled == 1

    def test_normalization_report_fields(self):
        from autoreview.knowledge_graph.normalize import NormalizationReport

        report = NormalizationReport()
        assert report.text_cleaned == 0
        assert report.predicates_cleaned == 0
        assert report.claims_decomposed == 0
        assert report.claims_produced == 0
        assert report.quant_backfilled == 0
        assert report.llm_calls == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestClaimNormalizer -v`
Expected: FAIL with `ImportError: cannot import name 'ClaimNormalizer'`

- [ ] **Step 3: Write minimal implementation**

Add to `autoreview/knowledge_graph/normalize.py` (add `import copy`, `from dataclasses import dataclass, field`, and `import structlog` to imports):

```python
import copy
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class NormalizationReport:
    """Audit summary of normalization transforms applied."""

    text_cleaned: int = 0
    predicates_cleaned: int = 0
    claims_decomposed: int = 0
    claims_produced: int = 0
    quant_backfilled: int = 0
    llm_calls: int = 0


class ClaimNormalizer:
    """Post-extraction normalization for improved cross-paper claim matching.

    Two entry points:
    - pre_dedup: text cleaning + compound decomposition (before entity dedup)
    - post_dedup: quantitative backfill (after entity dedup, before merge)
    """

    def __init__(
        self,
        llm_decompose: bool = True,
        llm_fn: LLMDecomposeFn | None = None,
    ):
        self._llm_decompose = llm_decompose
        self._llm_fn = llm_fn

    async def pre_dedup(
        self,
        entities: list[dict],
        assertions: list[dict],
    ) -> tuple[list[dict], list[dict], NormalizationReport]:
        """Run before entity dedup: text cleaning, predicate cleaning, decomposition.

        Args:
            entities: List of entity dicts from ingest (modified in place).
            assertions: List of assertion dicts from ingest.

        Returns:
            Tuple of (entities, assertions, report). Entities and assertions
            may be modified or expanded (decomposition creates new entries).
        """
        report = NormalizationReport()

        # Phase 1: Text cleaning on entity names
        name_map: dict[str, str] = {}  # old_name -> new_name
        for entity in entities:
            old_name = entity["canonical_name"]
            new_name, aliases = clean_entity_name(old_name)
            if new_name != old_name:
                name_map[old_name] = new_name
                entity["canonical_name"] = new_name
                entity["_original_name"] = old_name
                existing_aliases = entity.get("aliases", [])
                entity["aliases"] = list(dict.fromkeys(existing_aliases + aliases))
                report.text_cleaned += 1

        # Phase 1b: Text cleaning on assertion entity names (must stay in sync)
        for assertion in assertions:
            for field_name in ("subject_canonical_name", "object_canonical_name"):
                old_name = assertion[field_name]
                new_name, _ = clean_entity_name(old_name)
                if new_name != old_name:
                    assertion[f"_original_{field_name}"] = old_name
                    assertion[field_name] = new_name

        # Phase 2: Predicate cleaning
        for assertion in assertions:
            old_pred = assertion["predicate"]
            new_pred = clean_predicate(old_pred)
            if new_pred != old_pred:
                assertion["predicate"] = new_pred
                report.predicates_cleaned += 1

        # Phase 3: Compound object decomposition
        new_assertions: list[dict] = []
        to_remove: set[int] = set()
        llm_candidates: list[tuple[int, str]] = []  # (assertion_index, object_name)

        for i, assertion in enumerate(assertions):
            obj_name = assertion["object_canonical_name"]
            parts = decompose_object(obj_name)
            if len(parts) > 1:
                to_remove.add(i)
                report.claims_decomposed += 1
                for j, part in enumerate(parts):
                    new_a = copy.deepcopy(assertion)
                    new_a["object_canonical_name"] = part
                    new_a["draft_id"] = f"{assertion['draft_id']}_d{j + 1}"
                    new_a["_decomposed_from"] = assertion["draft_id"]
                    new_assertions.append(new_a)
                    # Add entity for new object
                    entities.append({
                        "canonical_name": part,
                        "entity_type": self._find_entity_type(entities, obj_name),
                        "ontology_id": None,
                        "ontology_source": None,
                        "aliases": [],
                        "paper_ids": [assertion.get("paper_id", "")],
                    })
                report.claims_produced += len(parts)
            elif flag_for_llm_decomposition(obj_name) and self._llm_decompose:
                llm_candidates.append((i, obj_name))

        # Phase 3b: LLM fallback for undecomposed verbose objects
        if llm_candidates and self._llm_decompose:
            objects_to_decompose = [obj for _, obj in llm_candidates]
            decompositions = await llm_decompose_objects(objects_to_decompose, self._llm_fn)
            report.llm_calls += 1 if self._llm_fn else 0
            for (idx, _orig_obj), parts in zip(llm_candidates, decompositions):
                if len(parts) > 1:
                    to_remove.add(idx)
                    assertion = assertions[idx]
                    report.claims_decomposed += 1
                    for j, part in enumerate(parts):
                        new_a = copy.deepcopy(assertion)
                        new_a["object_canonical_name"] = part
                        new_a["draft_id"] = f"{assertion['draft_id']}_d{j + 1}"
                        new_a["_decomposed_from"] = assertion["draft_id"]
                        new_assertions.append(new_a)
                        entities.append({
                            "canonical_name": part,
                            "entity_type": self._find_entity_type(entities, assertion["object_canonical_name"]),
                            "ontology_id": None,
                            "ontology_source": None,
                            "aliases": [],
                            "paper_ids": [assertion.get("paper_id", "")],
                        })
                    report.claims_produced += len(parts)

        # Rebuild assertions list: keep non-removed + add decomposed
        assertions = [a for i, a in enumerate(assertions) if i not in to_remove] + new_assertions

        log.info(
            "pre_dedup_normalization_complete",
            text_cleaned=report.text_cleaned,
            predicates_cleaned=report.predicates_cleaned,
            claims_decomposed=report.claims_decomposed,
            claims_produced=report.claims_produced,
            llm_calls=report.llm_calls,
        )
        return entities, assertions, report

    async def post_dedup(
        self,
        assertions: list[dict],
    ) -> tuple[list[dict], NormalizationReport]:
        """Run after entity dedup, before merge: quantitative context backfill.

        Args:
            assertions: List of assertion dicts with resolved entity IDs.

        Returns:
            Tuple of (assertions, report).
        """
        report = NormalizationReport()
        for assertion in assertions:
            if backfill_quantitative_context(assertion):
                report.quant_backfilled += 1
        log.info(
            "post_dedup_normalization_complete",
            quant_backfilled=report.quant_backfilled,
        )
        return assertions, report

    @staticmethod
    def _find_entity_type(entities: list[dict], name: str) -> str:
        """Look up entity_type for a given canonical name. Defaults to 'biological_process'."""
        for e in entities:
            if e["canonical_name"] == name:
                return e["entity_type"]
        return "biological_process"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestClaimNormalizer -v`
Expected: 6 PASSED

- [ ] **Step 5: Run all normalize tests**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py -v`
Expected: All 43 tests PASSED

- [ ] **Step 6: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): add ClaimNormalizer class with NormalizationReport"
```

---

### Task 7: Pipeline Integration

**Files:**
- Modify: `autoreview/knowledge_graph/__init__.py` (lines 128-133 for signature, lines 169-240 for pipeline flow)
- Modify: `tests/test_knowledge_graph/test_normalize.py` (add integration tests)

**Context:** `build_graph` is at line 128. Step 1 (ingest) is at line 169. Entity dedup is at line 181. Predicate norm + remap is at line 191. Merge branching is at line 229.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_graph/test_normalize.py`:

```python
class TestPipelineIntegration:
    """Integration tests for normalization in the build_graph pipeline."""

    def test_normalize_false_unchanged(self, sample_v5_extraction_dir):
        """Regression: normalize=False produces identical output to current code."""
        from autoreview.knowledge_graph import build_graph

        graph_without = build_graph(sample_v5_extraction_dir, version=2)
        graph_with = build_graph(sample_v5_extraction_dir, version=2, normalize=False)
        assert graph_without.number_of_edges() == graph_with.number_of_edges()
        assert graph_without.number_of_nodes() == graph_with.number_of_nodes()

    def test_normalize_true_accepted(self, sample_v5_extraction_dir):
        """normalize=True runs without error and produces a valid graph."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_v5_extraction_dir, version=2, normalize=True, llm_decompose=False)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_normalization_report_on_graph(self, sample_v5_extraction_dir):
        """NormalizationReport is stored on the graph object."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_v5_extraction_dir, version=2, normalize=True, llm_decompose=False)
        report = graph.graph.get("normalization_report")
        assert report is not None
        assert hasattr(report, "text_cleaned")
        assert hasattr(report, "quant_backfilled")

    def test_normalize_with_v1_version(self, sample_v5_extraction_dir):
        """Normalization works with v1 merge strategy too."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(sample_v5_extraction_dir, version=1, normalize=True, llm_decompose=False)
        assert graph.number_of_nodes() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestPipelineIntegration -v`
Expected: FAIL with `TypeError: build_graph() got an unexpected keyword argument 'normalize'`

- [ ] **Step 3: Modify `build_graph` signature**

In `autoreview/knowledge_graph/__init__.py`, update the function signature at line 128:

Change:
```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
) -> nx.MultiDiGraph:
```

To:
```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
    normalize: bool = False,
    llm_decompose: bool = True,
) -> nx.MultiDiGraph:
```

- [ ] **Step 4: Add import for ClaimNormalizer**

Add to the imports section of `__init__.py` (near the existing dedup imports):

```python
from autoreview.knowledge_graph.normalize import ClaimNormalizer
```

- [ ] **Step 5: Insert pre_dedup normalization pass**

After Step 1 (ingest, ~line 178) and before Step 2 (entity dedup, ~line 181), insert:

```python
    # Step 1b: Pre-dedup normalization (text cleaning + decomposition)
    pre_dedup_report = None
    if normalize:
        import asyncio

        normalizer = ClaimNormalizer(llm_decompose=llm_decompose)
        corpus.all_entities, corpus.all_assertions, pre_dedup_report = (
            asyncio.get_event_loop().run_until_complete(
                normalizer.pre_dedup(corpus.all_entities, corpus.all_assertions)
            )
        )
        log.info("pre_dedup_normalization", report=pre_dedup_report)
```

- [ ] **Step 6: Insert post_dedup normalization pass**

After Step 3 (predicate norm + remap, ~line 226) and before Step 4 (merge, ~line 229), insert:

```python
    # Step 3b: Post-dedup normalization (quantitative backfill)
    post_dedup_report = None
    if normalize:
        import asyncio

        normalizer = ClaimNormalizer(llm_decompose=False)  # no decomposition in post-dedup
        normalized_assertions, post_dedup_report = (
            asyncio.get_event_loop().run_until_complete(
                normalizer.post_dedup(normalized_assertions)
            )
        )
        log.info("post_dedup_normalization", report=post_dedup_report)
```

- [ ] **Step 7: Store NormalizationReport on graph**

After the graph is built (after `build_nx_graph`, ~line 380), add:

```python
    # Store normalization audit trail
    if normalize:
        from autoreview.knowledge_graph.normalize import NormalizationReport

        combined_report = NormalizationReport(
            text_cleaned=pre_dedup_report.text_cleaned if pre_dedup_report else 0,
            predicates_cleaned=pre_dedup_report.predicates_cleaned if pre_dedup_report else 0,
            claims_decomposed=pre_dedup_report.claims_decomposed if pre_dedup_report else 0,
            claims_produced=pre_dedup_report.claims_produced if pre_dedup_report else 0,
            quant_backfilled=post_dedup_report.quant_backfilled if post_dedup_report else 0,
            llm_calls=pre_dedup_report.llm_calls if pre_dedup_report else 0,
        )
        graph.graph["normalization_report"] = combined_report
```

- [ ] **Step 8: Run integration tests**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/test_normalize.py::TestPipelineIntegration -v`
Expected: 4 PASSED

- [ ] **Step 9: Run full test suite**

Run: `cd /Users/mst36/Desktop/Projects/Science/AutoReview && python -m pytest tests/test_knowledge_graph/ -v`
Expected: All tests PASSED (existing + new)

- [ ] **Step 10: Commit**

```bash
cd /Users/mst36/Desktop/Projects/Science/AutoReview && git add autoreview/knowledge_graph/__init__.py autoreview/knowledge_graph/normalize.py tests/test_knowledge_graph/test_normalize.py && git commit -m "feat(kg): integrate normalization layer into build_graph pipeline"
```
