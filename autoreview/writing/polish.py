"""Language polish stage: terminology consistency and duplicate claim detection."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_TERM_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")


@dataclass
class TerminologyEntry:
    term: str
    count: int = 0
    sections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Inconsistency:
    term_a: str
    term_b: str
    section_a: str
    section_b: str
    description: str


def extract_terminology(text: str) -> list[TerminologyEntry]:
    """Extract capitalized multi-word terms from text and count occurrences.

    Args:
        text: Input text to extract terms from.

    Returns:
        List of TerminologyEntry objects sorted by descending count.
    """
    matches = _TERM_PATTERN.findall(text)
    counts: Counter[str] = Counter(matches)
    entries = [TerminologyEntry(term=term, count=count) for term, count in counts.items()]
    entries.sort(key=lambda e: e.count, reverse=True)
    logger.debug("extract_terminology", term_count=len(entries))
    return entries


def find_inconsistencies(sections: dict[str, str]) -> list[Inconsistency]:
    """Detect capitalization variant inconsistencies for the same base term across sections.

    For each term found in a section, checks whether a lowercase variant (or vice versa)
    appears in any other section, signalling inconsistent usage.

    Args:
        sections: Mapping of section name to section text.

    Returns:
        List of Inconsistency objects describing conflicting term usages.
    """
    # Build mapping: section_name -> set of capitalized terms found
    section_terms: dict[str, set[str]] = {}
    for sec_name, text in sections.items():
        found = set(_TERM_PATTERN.findall(text))
        section_terms[sec_name] = found

    inconsistencies: list[Inconsistency] = []
    sec_names = list(section_terms.keys())

    for i, sec_a in enumerate(sec_names):
        for sec_b in sec_names[i + 1 :]:
            terms_a = section_terms[sec_a]
            terms_b = section_terms[sec_b]
            for term_a in terms_a:
                for term_b in terms_b:
                    if term_a == term_b:
                        continue
                    # Detect capitalization variants: same text modulo case of first letter
                    if term_a.lower() == term_b.lower() and term_a != term_b:
                        incon = Inconsistency(
                            term_a=term_a,
                            term_b=term_b,
                            section_a=sec_a,
                            section_b=sec_b,
                            description=(
                                f"Term '{term_a}' in '{sec_a}' and '{term_b}' in '{sec_b}' "
                                "differ only in capitalisation."
                            ),
                        )
                        inconsistencies.append(incon)

    logger.debug("find_inconsistencies", inconsistency_count=len(inconsistencies))
    return inconsistencies


def _sentence_tokens(sentence: str) -> frozenset[str]:
    """Return the set of lowercased word tokens from a sentence."""
    return frozenset(re.findall(r"\b\w+\b", sentence.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def find_duplicate_claims(
    sections: dict[str, str],
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Detect near-duplicate sentences across sections using Jaccard similarity.

    Only sentences longer than 40 characters are compared.

    Args:
        sections: Mapping of section name to section text.
        threshold: Jaccard similarity threshold above which sentences are considered
            duplicates. Defaults to 0.8.

    Returns:
        List of dicts with keys: sentence_a, sentence_b, section_a, section_b,
        similarity.
    """
    # Collect (section_name, sentence) pairs for sentences >40 chars
    sentence_pairs: list[tuple[str, str]] = []
    for sec_name, text in sections.items():
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence) > 40:
                sentence_pairs.append((sec_name, sentence))

    duplicates: list[dict[str, Any]] = []
    for i, (sec_a, sent_a) in enumerate(sentence_pairs):
        tokens_a = _sentence_tokens(sent_a)
        for sec_b, sent_b in sentence_pairs[i + 1 :]:
            if sec_a == sec_b:
                continue  # skip within-section comparisons
            tokens_b = _sentence_tokens(sent_b)
            sim = _jaccard(tokens_a, tokens_b)
            if sim >= threshold:
                duplicates.append(
                    {
                        "sentence_a": sent_a,
                        "sentence_b": sent_b,
                        "section_a": sec_a,
                        "section_b": sec_b,
                        "similarity": sim,
                    }
                )

    logger.debug("find_duplicate_claims", duplicate_count=len(duplicates))
    return duplicates
