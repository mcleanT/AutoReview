"""Composite scoring function for KG extraction quality.

Pure functions — no I/O. Returns a (composite_score, per_metric_dict) tuple.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER1_CANONICAL: frozenset[str] = frozenset(
    {
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
)

TIER2_SPECIFIC: frozenset[str] = frozenset(
    {
        "phosphorylates",
        "degrades",
        "stabilizes",
        "transports",
        "modifies",
        "converts",
        "maintains",
    }
)

VALID_PREDICATES: frozenset[str] = TIER1_CANONICAL | TIER2_SPECIFIC

VALID_EVIDENCE_STRENGTHS: frozenset[str] = frozenset(
    {
        "direct_experimental",
        "indirect_experimental",
        "observational",
        "computational",
        "review_citation",
    }
)

VALID_CLAIM_TYPES: frozenset[str] = frozenset(
    {
        "mechanistic_causal",
        "correlational",
        "comparative",
        "existence",
        "absence",
        "conditional",
        "methodological",
    }
)

PERTURB_KEYWORDS: frozenset[str] = frozenset(
    {
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
)

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

_METRIC_NAMES: tuple[str, ...] = tuple(METRIC_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shannon_entropy_normalized(counts: dict[str, int]) -> float:
    """Normalized Shannon entropy in [0, 1].

    Returns 0.0 when there is only one distinct key (or total == 0).
    Approaches 1.0 as the distribution approaches uniform over many keys.
    Normalised by log2(n_categories) so it's bounded at 1.0.
    """
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _subject_has_perturb_keyword(claim: dict) -> bool:
    subject = claim.get("subject") or {}
    name: str = (subject.get("name") or "").lower()
    return any(kw in name for kw in PERTURB_KEYWORDS)


def _quant_context_present(claim: dict) -> bool:
    qc = claim.get("quantitative_context")
    if not qc:
        return False
    return any(bool(v) for v in qc.values())


# ---------------------------------------------------------------------------
# Per-metric computations
# ---------------------------------------------------------------------------


def _metric_density(n: int) -> float:
    if n == 0:
        return 0.0
    if n < 25:
        return n / 25
    if n <= 50:
        return 1.0
    # linear penalty above 50; reaches 0 at n == 100
    return max(0.0, 1.0 - (n - 50) / 50)


def _metric_predicate_valid(claims: list[dict]) -> float:
    return sum(1 for c in claims if c.get("predicate") in VALID_PREDICATES) / len(claims)


def _metric_predicate_diversity(claims: list[dict]) -> float:
    counts: dict[str, int] = {}
    for c in claims:
        pred = c.get("predicate") or "__missing__"
        counts[pred] = counts.get(pred, 0) + 1
    return _shannon_entropy_normalized(counts)


def _metric_evidence_strength_valid(claims: list[dict]) -> float:
    return sum(1 for c in claims if c.get("evidence_strength") in VALID_EVIDENCE_STRENGTHS) / len(
        claims
    )


def _metric_field_fraction(claims: list[dict], field: str) -> float:
    return sum(1 for c in claims if c.get(field)) / len(claims)


def _metric_quant_context(claims: list[dict]) -> float:
    return sum(1 for c in claims if _quant_context_present(c)) / len(claims)


def _metric_causal_type_coverage(claims: list[dict]) -> float:
    mechanistic = [c for c in claims if c.get("claim_type") == "mechanistic_causal"]
    if not mechanistic:
        return 1.0
    return sum(1 for c in mechanistic if c.get("causal_type")) / len(mechanistic)


def _metric_no_perturb_subject(claims: list[dict]) -> float:
    return 1.0 - sum(1 for c in claims if _subject_has_perturb_keyword(c)) / len(claims)


def _metric_claim_type_valid(claims: list[dict]) -> float:
    return sum(1 for c in claims if c.get("claim_type") in VALID_CLAIM_TYPES) / len(claims)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_extraction(data: dict) -> tuple[float, dict[str, float]]:
    """Score a KG extraction result against 12 quality metrics.

    Args:
        data: Extraction JSON dict containing a "claims" list.

    Returns:
        Tuple of (composite_score, per_metric_dict). Both values are in [0, 1].
        composite_score is the weighted sum over METRIC_WEIGHTS.
        per_metric_dict maps metric name → score in [0, 1].
    """
    claims: list[dict] = data.get("claims") or []
    n = len(claims)

    if n == 0:
        metrics = {name: 0.0 for name in _METRIC_NAMES}
        return 0.0, metrics

    metrics: dict[str, float] = {
        "density": _metric_density(n),
        "predicate_valid": _metric_predicate_valid(claims),
        "predicate_diversity": _metric_predicate_diversity(claims),
        "evidence_strength_valid": _metric_evidence_strength_valid(claims),
        "field_model_system": _metric_field_fraction(claims, "model_system"),
        "field_organism": _metric_field_fraction(claims, "organism"),
        "field_certainty": _metric_field_fraction(claims, "certainty"),
        "field_section_source": _metric_field_fraction(claims, "section_source"),
        "quant_context": _metric_quant_context(claims),
        "causal_type_coverage": _metric_causal_type_coverage(claims),
        "no_perturb_subject": _metric_no_perturb_subject(claims),
        "claim_type_valid": _metric_claim_type_valid(claims),
    }

    composite = sum(METRIC_WEIGHTS[name] * score for name, score in metrics.items())
    return composite, metrics
