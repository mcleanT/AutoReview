"""Post-extraction claim normalization for improved cross-paper matching."""

from __future__ import annotations

import re

_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")

_LEADING_ARTICLES = frozenset({"the", "a", "an", "this", "these", "that", "those"})

_TRAILING_DESCRIPTORS = frozenset(
    {
        "process",
        "pathway",
        "mechanism",
        "activity",
        "event",
        "response",
        "cascade",
        "system",
    }
)


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

    # 3. Space removal — if removing all spaces yields a known predicate or tense entry
    if " " in cleaned:
        nospace = cleaned.replace(" ", "")
        if nospace in _PREDICATE_TENSE_MAP or nospace in _get_known_predicates():
            cleaned = _PREDICATE_TENSE_MAP.get(nospace, nospace)
        else:
            # 4. Underscore normalization — if spaces->underscores matches a known predicate
            underscored = cleaned.replace(" ", "_")
            if underscored in _get_known_predicates():
                cleaned = underscored

    return cleaned


_BIOLOGICAL_HEAD_NOUNS = frozenset(
    {
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
    }
)

_CONJUNCTION_SPLIT_RE = re.compile(r",\s*(?:and|or)\s+|\s+(?:and|or)\s+")
_PREP_RE = re.compile(r"^.*?\b(?:into|of|in|from|to|through|via)\s+(.+)$", re.IGNORECASE)


def _find_tail_suffix(part: str) -> tuple[str, str] | None:
    """Find the leading single-token prefix and the biological tail suffix in a part.

    For 'Nanog expression levels', returns ('Nanog', 'expression levels').
    For 'mesoderm differentiation', returns ('mesoderm', 'differentiation').
    For 'lateral mesoderm', returns None (no bio head noun).

    The tail suffix starts at the first biological head noun word encountered from
    the left (after the first token).
    """
    words = part.split()
    if len(words) < 2:
        return None
    for i in range(1, len(words)):
        if words[i].lower() in _BIOLOGICAL_HEAD_NOUNS:
            prefix = " ".join(words[:i])
            tail = " ".join(words[i:])
            return prefix, tail
    return None


def _find_head_noun(text: str) -> str | None:
    """Find a trailing biological head noun that distributes across conjuncts.
    Only used for simple single-word head noun detection in the full text."""
    words = text.split()
    if words and words[-1].lower() in _BIOLOGICAL_HEAD_NOUNS:
        return words[-1]
    return None


def _decompose_slash(text: str) -> list[str] | None:
    """Split 'BMP4/WNT signaling' -> ['BMP4 signaling', 'WNT signaling'].
    Only splits single-token slash patterns (not units like ng/mL)."""
    if "/" not in text:
        return None
    words = text.split()
    slash_idx = None
    for i, w in enumerate(words):
        if "/" in w:
            parts = w.split("/")
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
    """Split on 'and'/'or' with head noun distribution.

    Handles Oxford-comma lists: 'T, Sox2, and Nanog expression levels'
    -> ['T expression levels', 'Sox2 expression levels', 'Nanog expression levels']

    Also handles simple conjunctions: 'lateral mesoderm and neural crest'
    -> ['lateral mesoderm', 'neural crest']
    """
    if " and " not in text and " or " not in text:
        return None
    # First split on the conjunction markers (', and', ', or', ' and ', ' or ')
    parts = _CONJUNCTION_SPLIT_RE.split(text)
    # Then further split any remaining comma-separated tokens within each part
    expanded: list[str] = []
    for part in parts:
        sub = [s.strip().rstrip(",").strip() for s in part.split(",") if s.strip()]
        expanded.extend(sub)
    parts = [p for p in expanded if p]
    if len(parts) < 2:
        return None
    # Detect tail suffix from the last part (handles multi-word tails like
    # 'expression levels')
    last_part = parts[-1]
    tail_info = _find_tail_suffix(last_part)
    if tail_info:
        # last part has a prefix + biological tail; distribute the tail to all parts
        _, tail = tail_info
        result = []
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last part already contains the tail — keep as-is
                result.append(last_part)
            else:
                result.append(f"{part} {tail}")
        return result
    # No tail suffix — check for a single-word head noun on the full text
    head = _find_head_noun(text)
    result = []
    for part in parts:
        part_words = part.split()
        if head and not (part_words and part_words[-1].lower() == head.lower()):
            result.append(f"{part} {head}")
        else:
            result.append(part)
    return result


def _decompose_prepositional(text: str) -> list[str] | None:
    """Strip prepositional wrappers, decompose inner content.
    'self-organization into endoderm and mesoderm' -> ['endoderm', 'mesoderm']"""
    m = _PREP_RE.match(text)
    if not m:
        return None
    inner = m.group(1)
    result = _decompose_conjunction(inner)
    if result:
        return result
    return None


def decompose_object(object_name: str) -> list[str]:
    """Decompose a compound object into atomic concepts.
    Returns a list of 1+ strings. Single-element list = no decomposition."""
    if len(object_name.split()) <= 3:
        return [object_name]
    for fn in (_decompose_slash, _decompose_prepositional, _decompose_conjunction):
        result = fn(object_name)
        if result and len(result) > 1:
            return result
    return [object_name]


def flag_for_llm_decomposition(object_name: str) -> bool:
    """Check if an undecomposed object should be sent to LLM fallback."""
    return len(object_name.split()) > 6


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
