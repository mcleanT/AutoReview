"""Error pattern analyzer for KG extraction quality.

Detects systematic error patterns across a batch of extractions and returns
them sorted by impact (severity × frequency) descending.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ErrorPattern:
    category: str
    description: str
    severity: float
    frequency: int
    examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_nl(claim: dict) -> str:
    """Return the natural_language string for a claim, falling back to a terse repr."""
    return claim.get("natural_language") or (
        f"{(claim.get('subject') or {}).get('name', '?')} "
        f"{claim.get('predicate', '?')} "
        f"{(claim.get('object') or {}).get('name', '?')}"
    )


def _entity_name(entity: dict | None) -> str:
    if not entity:
        return ""
    return entity.get("name") or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_errors(extractions: list[dict]) -> list[ErrorPattern]:
    """Detect systematic error patterns across a batch of extractions.

    Args:
        extractions: List of extraction dicts, each containing a "claims" list
                     and optionally a "title" key.

    Returns:
        List of ErrorPattern objects sorted by severity × frequency descending.
    """
    # Flatten all claims, keyed by extraction index for low-density check.
    all_claims: list[dict] = []
    for ex in extractions:
        all_claims.extend(ex.get("claims") or [])

    patterns: list[ErrorPattern] = []

    total = len(all_claims)
    if total == 0:
        return patterns

    # ------------------------------------------------------------------
    # 1. predicate_overuse
    # ------------------------------------------------------------------
    pred_counts: Counter[str] = Counter(c.get("predicate") or "__missing__" for c in all_claims)
    for pred, count in pred_counts.items():
        ratio = count / total
        if ratio > 0.30:
            severity = min(1.0, (ratio - 0.30) * 3)
            examples = [_get_nl(c) for c in all_claims if c.get("predicate") == pred][:3]
            patterns.append(
                ErrorPattern(
                    category="predicate_overuse",
                    description=(
                        f"Predicate '{pred}' used in {ratio:.0%} of claims (threshold: 30%)."
                    ),
                    severity=severity,
                    frequency=count,
                    examples=examples,
                )
            )

    # ------------------------------------------------------------------
    # 2. invalid_predicate
    # ------------------------------------------------------------------
    invalid_pred_claims = [c for c in all_claims if c.get("predicate") not in VALID_PREDICATES]
    if invalid_pred_claims:
        # Group by predicate for description
        bad_preds: Counter[str] = Counter(
            c.get("predicate") or "__missing__" for c in invalid_pred_claims
        )
        desc_parts: list[str] = []
        for pred, cnt in bad_preds.most_common():
            canonical = REMOVED_TO_CANONICAL.get(pred or "")
            if canonical:
                desc_parts.append(f"'{pred}' (use '{canonical}') ×{cnt}")
            else:
                desc_parts.append(f"'{pred}' ×{cnt}")
        description = "Invalid predicates detected: " + ", ".join(desc_parts) + "."
        examples = [_get_nl(c) for c in invalid_pred_claims[:3]]
        patterns.append(
            ErrorPattern(
                category="invalid_predicate",
                description=description,
                severity=0.8,
                frequency=len(invalid_pred_claims),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 3. invalid_claim_type
    # ------------------------------------------------------------------
    invalid_ct_claims = [c for c in all_claims if c.get("claim_type") not in VALID_CLAIM_TYPES]
    if invalid_ct_claims:
        bad_types: Counter[str] = Counter(
            c.get("claim_type") or "__missing__" for c in invalid_ct_claims
        )
        desc_parts = [f"'{t}' ×{cnt}" for t, cnt in bad_types.most_common()]
        description = "Invalid claim_type values: " + ", ".join(desc_parts) + "."
        examples = [_get_nl(c) for c in invalid_ct_claims[:3]]
        patterns.append(
            ErrorPattern(
                category="invalid_claim_type",
                description=description,
                severity=0.7,
                frequency=len(invalid_ct_claims),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 4. invalid_evidence_strength
    # ------------------------------------------------------------------
    invalid_ev_claims = [
        c for c in all_claims if c.get("evidence_strength") not in VALID_EVIDENCE_STRENGTHS
    ]
    if invalid_ev_claims:
        bad_ev: Counter[str] = Counter(
            c.get("evidence_strength") or "__missing__" for c in invalid_ev_claims
        )
        desc_parts = [f"'{e}' ×{cnt}" for e, cnt in bad_ev.most_common()]
        description = "Invalid evidence_strength values: " + ", ".join(desc_parts) + "."
        examples = [_get_nl(c) for c in invalid_ev_claims[:3]]
        patterns.append(
            ErrorPattern(
                category="invalid_evidence_strength",
                description=description,
                severity=0.7,
                frequency=len(invalid_ev_claims),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 5. missing_field
    # ------------------------------------------------------------------
    for field_name in ["model_system", "organism", "certainty", "section_source"]:
        missing = [c for c in all_claims if not c.get(field_name)]
        if missing:
            ratio = len(missing) / total
            severity = ratio * 0.6
            examples = [_get_nl(c) for c in missing[:3]]
            patterns.append(
                ErrorPattern(
                    category="missing_field",
                    description=(
                        f"Field '{field_name}' missing in {ratio:.0%} of claims "
                        f"({len(missing)}/{total})."
                    ),
                    severity=severity,
                    frequency=len(missing),
                    examples=examples,
                )
            )

    # ------------------------------------------------------------------
    # 6. missing_causal_type
    # ------------------------------------------------------------------
    mechanistic = [c for c in all_claims if c.get("claim_type") == "mechanistic_causal"]
    missing_ct = [c for c in mechanistic if not c.get("causal_type")]
    if missing_ct:
        examples = [_get_nl(c) for c in missing_ct[:3]]
        patterns.append(
            ErrorPattern(
                category="missing_causal_type",
                description=(f"{len(missing_ct)} mechanistic_causal claim(s) missing causal_type."),
                severity=0.6,
                frequency=len(missing_ct),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 7. perturbation_as_subject
    # ------------------------------------------------------------------
    perturb_claims = [
        c
        for c in all_claims
        if any(kw in (_entity_name(c.get("subject") or {}).lower()) for kw in PERTURB_KEYWORDS)
    ]
    if perturb_claims:
        examples = [_get_nl(c) for c in perturb_claims[:3]]
        patterns.append(
            ErrorPattern(
                category="perturbation_as_subject",
                description=(
                    f"{len(perturb_claims)} claim(s) use a perturbation condition as the subject "
                    "instead of the affected entity."
                ),
                severity=0.9,
                frequency=len(perturb_claims),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 8. parenthetical_entity_name
    # ------------------------------------------------------------------
    paren_claims = [
        c
        for c in all_claims
        if (
            (
                "(" in _entity_name(c.get("subject") or {})
                and ")" in _entity_name(c.get("subject") or {})
            )
            or (
                "(" in _entity_name(c.get("object") or {})
                and ")" in _entity_name(c.get("object") or {})
            )
        )
    ]
    if paren_claims:
        examples = [_get_nl(c) for c in paren_claims[:3]]
        patterns.append(
            ErrorPattern(
                category="parenthetical_entity_name",
                description=(
                    f"{len(paren_claims)} claim(s) have entity names with parenthetical synonyms; "
                    "prefer canonical names only."
                ),
                severity=0.5,
                frequency=len(paren_claims),
                examples=examples,
            )
        )

    # ------------------------------------------------------------------
    # 9. low_density (per-extraction)
    # ------------------------------------------------------------------
    for ex in extractions:
        claims = ex.get("claims") or []
        if len(claims) < 15:
            title = ex.get("title") or "Unknown paper"
            patterns.append(
                ErrorPattern(
                    category="low_density",
                    description=(
                        f"Only {len(claims)} claims extracted from '{title}' "
                        f"(minimum recommended: 15)."
                    ),
                    severity=0.7,
                    frequency=1,
                    examples=[],
                )
            )

    # ------------------------------------------------------------------
    # Sort by severity × frequency descending
    # ------------------------------------------------------------------
    patterns.sort(key=lambda p: p.severity * p.frequency, reverse=True)
    return patterns
