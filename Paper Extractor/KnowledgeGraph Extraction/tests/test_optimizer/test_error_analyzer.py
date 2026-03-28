"""Tests for the KG extraction error analyzer."""

from optimize.error_analyzer import analyze_errors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claim(**overrides) -> dict:
    """Return a valid claim dict with sensible defaults."""
    base = {
        "subject": {"name": "BMP4", "type": "protein"},
        "predicate": "induces",
        "object": {"name": "Sox17", "type": "gene"},
        "natural_language": "BMP4 induces Sox17 expression.",
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


def _make_extraction(claims: list[dict], title: str = "Test Paper") -> dict:
    """Wrap claims into an extraction dict."""
    return {"title": title, "claims": claims}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_errors_clean_extraction():
    """25 claims with 5 diverse valid predicates → no critical errors (severity > 0.5)."""
    predicates = ["induces", "inhibits", "is_required_for", "regulates", "correlates_with"]
    claims = [_make_claim(predicate=predicates[i % len(predicates)]) for i in range(25)]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    critical = [p for p in patterns if p.severity > 0.5]
    assert critical == [], f"Unexpected critical errors: {[p.category for p in critical]}"


def test_detects_predicate_overuse():
    """10 claims all 'regulates' → overuse[0].category == 'predicate_overuse', 'regulates' in description."""
    claims = [_make_claim(predicate="regulates") for _ in range(10)]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    overuse = [p for p in patterns if p.category == "predicate_overuse"]
    assert len(overuse) >= 1, "Expected predicate_overuse pattern"
    assert overuse[0].category == "predicate_overuse"
    assert "regulates" in overuse[0].description


def test_detects_invalid_predicate():
    """2 claims with 'activates' and 'promotes' → invalid_predicate found."""
    claims = [
        _make_claim(predicate="activates"),
        _make_claim(predicate="promotes"),
        _make_claim(predicate="induces"),  # valid, to have at least one valid
    ]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    invalid = [p for p in patterns if p.category == "invalid_predicate"]
    assert len(invalid) >= 1, "Expected invalid_predicate pattern"


def test_detects_perturbation_subject():
    """1 claim with subject name 'Rai14 knockdown' → perturbation_as_subject found."""
    claims = [
        _make_claim(subject={"name": "Rai14 knockdown", "type": "perturbation"}),
    ]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    perturb = [p for p in patterns if p.category == "perturbation_as_subject"]
    assert len(perturb) >= 1, "Expected perturbation_as_subject pattern"


def test_detects_parenthetical_entity():
    """1 claim with subject name 'Invariant chain (CD74)' → parenthetical_entity_name found."""
    claims = [
        _make_claim(subject={"name": "Invariant chain (CD74)", "type": "protein"}),
    ]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    parens = [p for p in patterns if p.category == "parenthetical_entity_name"]
    assert len(parens) >= 1, "Expected parenthetical_entity_name pattern"


def test_detects_invalid_claim_type():
    """1 claim with claim_type 'quantitative_comparative' → invalid_claim_type found, type name in description."""
    claims = [
        _make_claim(claim_type="quantitative_comparative"),
    ]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    invalid_ct = [p for p in patterns if p.category == "invalid_claim_type"]
    assert len(invalid_ct) >= 1, "Expected invalid_claim_type pattern"
    assert "quantitative_comparative" in invalid_ct[0].description


def test_detects_missing_causal_type():
    """1 mechanistic_causal claim with causal_type=None → missing_causal_type found."""
    claims = [
        _make_claim(claim_type="mechanistic_causal", causal_type=None),
    ]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    missing_ct = [p for p in patterns if p.category == "missing_causal_type"]
    assert len(missing_ct) >= 1, "Expected missing_causal_type pattern"


def test_detects_low_density():
    """8 claims → low_density found."""
    claims = [_make_claim() for _ in range(8)]
    extraction = _make_extraction(claims, title="Short Paper")
    patterns = analyze_errors([extraction])
    low_d = [p for p in patterns if p.category == "low_density"]
    assert len(low_d) >= 1, "Expected low_density pattern"


def test_errors_sorted_by_impact():
    """Multiple error types → verify patterns are sorted by severity * frequency descending."""
    # Mix of error types to generate multiple patterns
    predicates = ["induces"] * 10  # overuse
    claims = [
        _make_claim(predicate="activates"),  # invalid predicate
        _make_claim(predicate="promotes"),  # invalid predicate
        _make_claim(subject={"name": "Rai14 knockdown", "type": "p"}),  # perturbation subject
    ] + [_make_claim(predicate=p) for p in predicates]
    extraction = _make_extraction(claims)
    patterns = analyze_errors([extraction])
    assert len(patterns) >= 2, "Need at least 2 patterns to check sort order"
    impacts = [p.severity * p.frequency for p in patterns]
    assert impacts == sorted(impacts, reverse=True), f"Patterns not sorted by impact: {impacts}"
