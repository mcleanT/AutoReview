"""Factual accuracy scoring for extraction benchmark.

Complements the existing embedding-similarity scoring in scoring.py with
metrics that measure whether the correct facts (numbers, entities, topics)
were captured, regardless of wording.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

# --- Fact extraction primitives ---

_NUMBER_RE = re.compile(r"\d+\.?\d*")

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:[-/][A-Z]?[A-Za-z]*)*\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "These",
        "That",
        "Those",
        "However",
        "Moreover",
        "Furthermore",
        "Additionally",
        "Our",
        "We",
        "They",
        "It",
        "Its",
        "Here",
        "There",
        "When",
        "Where",
        "While",
        "Although",
        "Despite",
        "Between",
        "Among",
        "Within",
        "Without",
        "After",
        "Before",
        "During",
        "Since",
        "Both",
        "Each",
        "Every",
        "Many",
        "Most",
        "Some",
        "Several",
        "Such",
        "Other",
        "More",
        "Less",
        "First",
        "Second",
        "Third",
        "Finally",
        "Overall",
        "Specifically",
        "Notably",
        "Importantly",
        "Interestingly",
        "Table",
        "Figure",
        "Section",
        "Appendix",
        "Results",
        "Methods",
        "Discussion",
        "Introduction",
        "Conclusion",
        "Abstract",
    }
)


def _extract_numbers(text: str | None) -> set[float]:
    """Extract all numeric values from text.

    Returns a set of floats. Percentages are returned as the bare number
    (67.6% -> 67.6). Handles integers and decimals.
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
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group())
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
