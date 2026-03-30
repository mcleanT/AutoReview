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
