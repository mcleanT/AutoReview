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
        "conditions": {"species": ["Mus musculus"], "cell_type": ["mESC"]},
        "evidence_links": [{"evidence_id": "e_001", "direction": "supports"}],
    }
    base.update(overrides)
    return base


def _make_extraction(
    claims: list[dict], title: str = "Test Paper", evidence: list[dict] | None = None
) -> dict:
    """Wrap claims into an extraction dict with sufficient evidence."""
    if evidence is None:
        evidence = [
            {
                "evidence_id": f"e_{i:03d}",
                "result_summary": f"Result {i}",
                "readout": "qPCR",
                "key_figure": f"Fig {i}",
            }
            for i in range(10)
        ]
    return {"title": title, "claims": claims, "evidence": evidence}


# ---------------------------------------------------------------------------
# Tests — existing patterns
# ---------------------------------------------------------------------------


def test_no_errors_clean_extraction():
    """25 claims with diverse valid predicates, evidence + links → no critical errors (severity > 0.5)."""
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


# ---------------------------------------------------------------------------
# Tests — new patterns (patterns 10-15)
# ---------------------------------------------------------------------------


def test_missing_evidence_links():
    """Claims without evidence_links trigger missing_evidence_links pattern."""
    extractions = [
        {
            "claims": [
                {"predicate": "induces", "subject": {"name": "X"}, "object": {"name": "Y"}},
                {
                    "predicate": "inhibits",
                    "subject": {"name": "A"},
                    "object": {"name": "B"},
                    "evidence_links": [{"evidence_id": "e_001"}],
                },
            ],
            "evidence": [
                {
                    "evidence_id": "e_001",
                    "result_summary": "x",
                    "readout": "y",
                    "key_figure": "Fig 1",
                }
            ],
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "missing_evidence_links" in cats


def test_missing_doi():
    """attributed_prior claims without source_doi trigger missing_doi pattern."""
    extractions = [
        {
            "claims": [
                {
                    "predicate": "induces",
                    "section_source": "attributed_prior",
                    "subject": {"name": "X"},
                    "object": {"name": "Y"},
                },
                {
                    "predicate": "inhibits",
                    "section_source": "attributed_prior",
                    "source_doi": "10.1234/abc",
                    "subject": {"name": "A"},
                    "object": {"name": "B"},
                },
            ],
            "evidence": [],
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "missing_doi" in cats


def test_low_evidence_density():
    """Fewer than 5 evidence units per extraction triggers low_evidence_density."""
    extractions = [
        {
            "claims": [{"predicate": "induces", "subject": {"name": "X"}, "object": {"name": "Y"}}],
            "evidence": [{"evidence_id": "e_001"}],
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "low_evidence_density" in cats


def test_incomplete_evidence():
    """Evidence missing required fields (result_summary, readout, key_figure) triggers incomplete_evidence."""
    extractions = [
        {
            "claims": [{"predicate": "induces", "subject": {"name": "X"}, "object": {"name": "Y"}}],
            "evidence": [
                {
                    "evidence_id": "e_001",
                    "description": "test",
                    "result_summary": "",
                    "readout": "qPCR",
                    "key_figure": "",
                }
            ],
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "incomplete_evidence" in cats


def test_sparse_conditions():
    """Claims with fewer than 2 conditions fields trigger sparse_conditions."""
    extractions = [
        {
            "claims": [
                {
                    "predicate": "induces",
                    "conditions": {},
                    "subject": {"name": "X"},
                    "object": {"name": "Y"},
                }
            ]
            * 5
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "sparse_conditions" in cats


def test_entity_name_fragmentation():
    """Same entity referenced with different casing triggers entity_name_fragmentation."""
    extractions = [
        {
            "claims": [
                {
                    "predicate": "induces",
                    "subject": {"name": "BMP4"},
                    "object": {"name": "mesoderm"},
                },
                {
                    "predicate": "inhibits",
                    "subject": {"name": "Bmp4"},
                    "object": {"name": "mesoderm"},
                },
            ]
        }
    ]
    patterns = analyze_errors(extractions)
    cats = [p.category for p in patterns]
    assert "entity_name_fragmentation" in cats
