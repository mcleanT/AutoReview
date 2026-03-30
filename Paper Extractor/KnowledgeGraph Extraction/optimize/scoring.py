"""Composite scoring function for KG extraction quality.

Pure functions — no I/O. Returns a (composite_score, per_metric_dict) tuple.
"""

from __future__ import annotations

import math
from collections import defaultdict

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
    # --- Claim quality (55%) — the core extraction job ---
    "quant_context": 0.12,
    "conditions_coverage": 0.10,
    "density": 0.10,
    "predicate_diversity": 0.10,
    "causal_type_coverage": 0.05,
    "entity_consistency": 0.05,
    "no_perturb_subject": 0.03,
    # --- Evidence quality (25%) — grounds claims in experiments ---
    "evidence_linkage": 0.04,
    "evidence_depth": 0.06,
    "evidence_completeness": 0.07,
    "evidence_density": 0.05,
    "evidence_strength_valid": 0.03,
    "field_certainty": 0.02,
    # --- Cross-paper connectivity (12%) — enables synthesis ---
    "doi_coverage": 0.05,
    "citation_contexts": 0.04,
    "field_section_source": 0.03,
    # --- Format validity (8%) — coercion handles most of this ---
    "predicate_valid": 0.04,
    "claim_type_valid": 0.02,
    "field_model_system": 0.01,
    "field_organism": 0.01,
}

_METRIC_NAMES: tuple[str, ...] = tuple(METRIC_WEIGHTS.keys())

# Metrics to exclude in rapid mode (truncation removes Discussion/References,
# making doi_coverage and citation_contexts uncomputable)
RAPID_EXCLUDE: frozenset[str] = frozenset({"doi_coverage", "citation_contexts"})


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


import re

_HAS_NUMBER_RE = re.compile(r"\d")


def _quant_context_present(claim: dict) -> bool:
    """Check if quantitative_context has at least one value containing a number.

    Values like {"other": "see figure"} don't count — only values with actual
    quantities (concentrations, timepoints, doses, p-values, fold-changes).
    """
    qc = claim.get("quantitative_context")
    if not qc:
        return False
    return any(_HAS_NUMBER_RE.search(str(v)) for v in qc.values() if v)


# ---------------------------------------------------------------------------
# Per-metric computations
# ---------------------------------------------------------------------------


def _metric_density(n: int, paper_chars: int = 0) -> float:
    """Score claim density, optionally scaling with paper size.

    When paper_chars > 0, the sweet-spot floor and ceiling scale linearly
    with paper length (roughly 5 claims per 10K chars as floor, 10 per 10K
    as ceiling).  When paper_chars == 0 (unknown), falls back to a wide
    fixed range of 15–120 that avoids penalising any reasonable extraction.
    """
    if n == 0:
        return 0.0
    if paper_chars > 0:
        # Scale sweet-spot with paper size
        units = paper_chars / 10_000  # 10K-char units
        floor = max(10, int(units * 5))  # ~5 claims per 10K chars
        ceiling = max(30, int(units * 10))  # ~10 claims per 10K chars
    else:
        # Unknown paper size — wide permissive range
        floor = 15
        ceiling = 120
    if n < floor:
        return n / floor
    if n <= ceiling:
        return 1.0
    # Gentle penalty above ceiling; reaches 0 at 2x ceiling
    return max(0.0, 1.0 - (n - ceiling) / ceiling)


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
# Graph-utility metrics (new)
# ---------------------------------------------------------------------------


def _metric_evidence_linkage(claims: list[dict]) -> float:
    return sum(1 for c in claims if c.get("evidence_links")) / len(claims)


def _metric_evidence_depth(claims: list[dict]) -> float:
    """Average evidence links per primary claim, scored with 1 link as baseline.

    Skips attributed_prior claims (external evidence, not local).
    1 link → 0.0 (baseline, no credit), 2 links → 0.5, 3+ links → 1.0.
    Claims with 0 links score -0.5 (penalty) but metric is floored at 0.0.
    """
    primary = [c for c in claims if c.get("section_source") != "attributed_prior"]
    if not primary:
        return 1.0
    total = 0.0
    for c in primary:
        n_links = len(c.get("evidence_links") or [])
        # 0 = -0.5, 1 = 0.0 (baseline), 2 = 0.2, 3 = 0.4, 4 = 0.6, 5+ = 0.8-1.0
        total += min(1.0, (n_links - 1) / 5.0)
    return max(0.0, total / len(primary))


def _metric_evidence_density(n_evidence: int) -> float:
    if n_evidence == 0:
        return 0.0
    if n_evidence < 5:
        return n_evidence / 5
    if n_evidence <= 30:
        return 1.0
    return max(0.0, 1.0 - (n_evidence - 30) / 30)


def _metric_evidence_completeness(evidence: list[dict]) -> float:
    if not evidence:
        return 0.0
    scores = []
    for ev in evidence:
        populated = sum(1 for f in ("result_summary", "readout", "key_figure") if ev.get(f))
        scores.append(populated / 3)
    return sum(scores) / len(scores)


def _metric_doi_coverage(claims: list[dict]) -> float:
    prior = [c for c in claims if c.get("section_source") == "attributed_prior"]
    if not prior:
        return 1.0
    return sum(1 for c in prior if c.get("source_doi")) / len(prior)


def _conditions_populated_count(claim: dict) -> int:
    conds = claim.get("conditions") or {}
    count = 0
    for list_field in ("species", "cell_type", "tissue", "treatment"):
        if conds.get(list_field):  # non-empty list
            count += 1
    if conds.get("developmental_stage") is not None:
        count += 1
    if conds.get("in_vitro") is not None:
        count += 1
    return count


def _metric_conditions_coverage(claims: list[dict]) -> float:
    return sum(1 for c in claims if _conditions_populated_count(c) >= 2) / len(claims)


def _metric_entity_consistency(claims: list[dict]) -> float:
    name_groups: dict[str, set[str]] = defaultdict(set)
    for c in claims:
        for key in ("subject", "object"):
            ent = c.get(key) or {}
            name = ent.get("name") or ""
            if name:
                name_groups[name.lower()].add(name)
    if not name_groups:
        return 1.0
    fragmented = sum(1 for variants in name_groups.values() if len(variants) > 1)
    return 1.0 - fragmented / len(name_groups)


def _metric_citation_contexts(data: dict) -> float:
    prior = [c for c in (data.get("claims") or []) if c.get("section_source") == "attributed_prior"]
    if not prior:
        return 1.0
    contexts = data.get("citation_contexts") or []
    return min(1.0, len(contexts) / len(prior))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_extraction(
    data: dict,
    rapid: bool = False,
    paper_chars: int = 0,
) -> tuple[float, dict[str, float]]:
    """Score a KG extraction result against quality metrics.

    Args:
        data: Extraction JSON dict containing "claims", "evidence", and
            "citation_contexts" lists.
        rapid: When True, exclude metrics that depend on sections removed
            by rapid truncation (Discussion/References). Excluded metrics
            are still computed and returned but do not affect the composite.
        paper_chars: Length of the source paper text in characters. When > 0,
            the density metric scales its sweet-spot with paper size.

    Returns:
        Tuple of (composite_score, per_metric_dict). Both values are in [0, 1].
        composite_score is the weighted sum over METRIC_WEIGHTS (renormalized
        when rapid=True excludes metrics).
        per_metric_dict maps metric name → score in [0, 1].
    """
    claims: list[dict] = data.get("claims") or []
    n = len(claims)

    if n == 0:
        metrics = {name: 0.0 for name in _METRIC_NAMES}
        return 0.0, metrics

    evidence: list[dict] = data.get("evidence") or []
    n_evidence = len(evidence)

    metrics: dict[str, float] = {
        # Graph structure
        "density": _metric_density(n, paper_chars),
        "evidence_linkage": _metric_evidence_linkage(claims),
        "evidence_depth": _metric_evidence_depth(claims),
        "evidence_density": _metric_evidence_density(n_evidence),
        "doi_coverage": _metric_doi_coverage(claims),
        "conditions_coverage": _metric_conditions_coverage(claims),
        "entity_consistency": _metric_entity_consistency(claims),
        "citation_contexts": _metric_citation_contexts(data),
        # Extraction quality
        "predicate_diversity": _metric_predicate_diversity(claims),
        "quant_context": _metric_quant_context(claims),
        "evidence_completeness": _metric_evidence_completeness(evidence),
        "causal_type_coverage": _metric_causal_type_coverage(claims),
        "no_perturb_subject": _metric_no_perturb_subject(claims),
        # Format validity
        "predicate_valid": _metric_predicate_valid(claims),
        "evidence_strength_valid": _metric_evidence_strength_valid(claims),
        "claim_type_valid": _metric_claim_type_valid(claims),
        "field_model_system": _metric_field_fraction(claims, "model_system"),
        "field_organism": _metric_field_fraction(claims, "organism"),
        "field_certainty": _metric_field_fraction(claims, "certainty"),
        "field_section_source": _metric_field_fraction(claims, "section_source"),
    }

    if rapid:
        active = {k: v for k, v in METRIC_WEIGHTS.items() if k not in RAPID_EXCLUDE}
        total_weight = sum(active.values())
        composite = sum(active[name] * metrics[name] for name in active) / total_weight
    else:
        composite = sum(METRIC_WEIGHTS[name] * score for name, score in metrics.items())
    return composite, metrics
