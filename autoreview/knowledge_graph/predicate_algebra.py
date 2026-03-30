"""Predicate algebra for the knowledge graph.

Single source of truth for predicate logic: opposition, composition, and
canonicalization. This module has NO dependencies on other autoreview modules —
it is pure data + pure functions.

Public API
----------
get_canonical_predicate(predicate) -> str
    Map a synonym predicate to its canonical form. Unknown predicates are
    returned unchanged.

are_opposing(pred_a, pred_b) -> bool
    Return True if the two predicates (after canonicalization) are logical
    opposites (e.g., "induces" vs "inhibits").

compose_predicates(pred_ab, pred_bc) -> CompositionResult | None
    Return the composed predicate for a two-hop chain A→B→C, or None if no
    composition rule exists.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

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
    "enhances": "induces",
    "reduces": "inhibits",
    "enables": "is_required_for",
    "localizes_to": "is_located_in",
    "is_expressed_in": "is_located_in",
    "binds_to": "interacts_with",
    "associates_with": "interacts_with",
}


def get_canonical_predicate(predicate: str) -> str:
    """Return the canonical form of a predicate.

    Synonyms (e.g., "activates", "triggers") map to their canonical predicate
    (e.g., "induces"). Unknown predicates are returned unchanged.

    Args:
        predicate: Any predicate string.

    Returns:
        Canonical predicate string.
    """
    return _CANONICAL_MAP.get(predicate, predicate)


# ---------------------------------------------------------------------------
# Opposition
# ---------------------------------------------------------------------------

# Each pair is (canonical_a, canonical_b); opposition is bidirectional.
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
    ("degrades", "stabilizes"),
]

# Build bidirectional lookup: (a, b) and (b, a) → True
_OPPOSITION_SET: frozenset[tuple[str, str]] = frozenset(
    pair for a, b in OPPOSITION_PAIRS for pair in [(a, b), (b, a)]
)


def are_opposing(pred_a: str, pred_b: str) -> bool:
    """Return True if two predicates are logical opposites.

    Both predicates are canonicalized before the lookup, so synonym forms
    (e.g., "activates" vs "suppresses") are handled correctly.

    Args:
        pred_a: First predicate (canonical or synonym).
        pred_b: Second predicate (canonical or synonym).

    Returns:
        True if the predicates are logically opposing.
    """
    canon_a = get_canonical_predicate(pred_a)
    canon_b = get_canonical_predicate(pred_b)
    return (canon_a, canon_b) in _OPPOSITION_SET


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionResult:
    """Result of composing two predicates transitively.

    Attributes:
        composed_predicate: The resulting predicate for the A→C link.
        strength: Confidence weight for the inferred edge (0.0–1.0).
        rule: Human-readable description of the composition rule applied.
    """

    composed_predicate: str
    strength: float
    rule: str


# (pred_AB_canonical, pred_BC_canonical) → CompositionResult
COMPOSITION_TABLE: dict[tuple[str, str], CompositionResult] = {
    ("induces", "induces"): CompositionResult(
        "induces", 0.7, "A activates B, B activates C → A promotes C"
    ),
    ("induces", "inhibits"): CompositionResult(
        "inhibits", 0.6, "A activates B, B inhibits C → A suppresses C"
    ),
    ("inhibits", "inhibits"): CompositionResult(
        "induces", 0.5, "A inhibits B, B inhibits C → A promotes C (double negative)"
    ),
    ("inhibits", "induces"): CompositionResult(
        "inhibits", 0.5, "A inhibits B, B activates C → A suppresses C"
    ),
    ("is_required_for", "is_required_for"): CompositionResult(
        "is_required_for", 0.6, "A required for B, B required for C → A required for C"
    ),
    ("is_required_for", "induces"): CompositionResult(
        "induces", 0.4, "A required for B, B activates C → A promotes C (weak)"
    ),
    ("is_required_for", "inhibits"): CompositionResult(
        "inhibits", 0.4, "A required for B, B inhibits C → A suppresses C (weak)"
    ),
    ("regulates", "induces"): CompositionResult(
        "regulates", 0.3, "A regulates B, B activates C → A regulates C (direction unknown)"
    ),
    ("regulates", "inhibits"): CompositionResult(
        "regulates", 0.3, "A regulates B, B inhibits C → A regulates C (direction unknown)"
    ),
    ("induces", "is_located_in"): CompositionResult(
        "regulates", 0.3, "A activates B, B localizes to C → A influences C localization"
    ),
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
    """Compose two predicates transitively for a two-hop chain A→B→C.

    Both predicates are canonicalized before the table lookup, so synonym
    forms are handled transparently.

    Args:
        pred_ab: Predicate on the A→B edge (canonical or synonym).
        pred_bc: Predicate on the B→C edge (canonical or synonym).

    Returns:
        A :class:`CompositionResult` if a composition rule exists, else None.
    """
    canon_ab = get_canonical_predicate(pred_ab)
    canon_bc = get_canonical_predicate(pred_bc)
    return COMPOSITION_TABLE.get((canon_ab, canon_bc))
