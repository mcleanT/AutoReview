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
