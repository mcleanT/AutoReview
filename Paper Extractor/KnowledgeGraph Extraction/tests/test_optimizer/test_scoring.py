"""Tests for the KG extraction scoring function."""

import pytest
from optimize.scoring import METRIC_WEIGHTS, score_extraction


def _make_claim(**overrides) -> dict:
    """Return a valid claim dict with sensible defaults."""
    base = {
        "subject": {"name": "BMP4", "type": "protein"},
        "predicate": "induces",
        "object": {"name": "Sox17", "type": "gene"},
        "evidence_strength": "direct_experimental",
        "claim_type": "mechanistic_causal",
        "causal_type": "activation",
        "model_system": "hESC",
        "organism": "human",
        "certainty": "high",
        "section_source": "Results",
        "quantitative_context": {"value": "2-fold", "p_value": "0.01"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_score_empty_extraction():
    composite, metrics = score_extraction({"claims": []})
    assert composite == 0.0
    assert all(v == 0.0 for v in metrics.values())


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def test_density_below_25():
    claims = [_make_claim() for _ in range(10)]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["density"] == pytest.approx(10 / 25)


def test_density_above_50():
    claims = [_make_claim() for _ in range(60)]
    _, metrics = score_extraction({"claims": claims})
    assert 0.0 < metrics["density"] < 1.0


# ---------------------------------------------------------------------------
# Perfect-score smoke test
# ---------------------------------------------------------------------------


def test_score_perfect_30_claims():
    predicates = [
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
    ]
    claims = [_make_claim(predicate=predicates[i % len(predicates)]) for i in range(30)]
    composite, metrics = score_extraction({"claims": claims})
    assert composite > 0.85
    assert metrics["density"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Predicate validity
# ---------------------------------------------------------------------------


def test_invalid_predicates_penalized():
    claims = [
        _make_claim(predicate="promotes"),  # invalid
        _make_claim(predicate="activates"),  # invalid
        _make_claim(predicate="induces"),  # valid
    ]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["predicate_valid"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Predicate diversity
# ---------------------------------------------------------------------------


def test_predicate_diversity_all_same():
    claims = [_make_claim(predicate="induces") for _ in range(30)]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["predicate_diversity"] == pytest.approx(0.0)


def test_predicate_diversity_varied():
    predicates = ["induces", "inhibits", "is_required_for", "regulates", "correlates_with"]
    claims = [_make_claim(predicate=predicates[i % len(predicates)]) for i in range(25)]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["predicate_diversity"] > 0.8


# ---------------------------------------------------------------------------
# Field presence
# ---------------------------------------------------------------------------


def test_missing_model_system():
    claims = [
        _make_claim(model_system="hESC"),
        _make_claim(model_system=None),
        _make_claim(model_system=""),
    ]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["field_model_system"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Causal type coverage
# ---------------------------------------------------------------------------


def test_missing_causal_type_on_mechanistic():
    claims = [
        _make_claim(claim_type="mechanistic_causal", causal_type="activation"),  # ok
        _make_claim(claim_type="mechanistic_causal", causal_type=None),  # missing
        _make_claim(claim_type="correlational", causal_type=None),  # non-mech — ignored
    ]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["causal_type_coverage"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Perturbation subject detection
# ---------------------------------------------------------------------------


def test_perturbation_as_subject_detected():
    claims = [
        _make_claim(subject={"name": "BMP4 knockdown", "type": "perturbation"}),
        _make_claim(subject={"name": "BMP4", "type": "protein"}),
    ]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["no_perturb_subject"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Claim type validity
# ---------------------------------------------------------------------------


def test_invalid_claim_type():
    claims = [
        _make_claim(claim_type="mechanistic_causal"),
        _make_claim(claim_type="quantitative_comparative"),  # invalid
    ]
    _, metrics = score_extraction({"claims": claims})
    assert metrics["claim_type_valid"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Weight sanity
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9
