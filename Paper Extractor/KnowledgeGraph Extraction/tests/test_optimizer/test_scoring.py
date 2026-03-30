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
    """30 well-formed claims with evidence + links → composite in valid range and density == 1.0."""
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
    # Build evidence units (10, within the 5-30 optimal range)
    evidence = [
        {
            "evidence_id": f"e_{i:03d}",
            "result_summary": f"Result {i}",
            "readout": "qPCR",
            "key_figure": f"Fig {i}",
        }
        for i in range(10)
    ]
    claims = [
        _make_claim(
            predicate=predicates[i % len(predicates)],
            evidence_links=[{"evidence_id": f"e_{i % 10:03d}", "direction": "supports"}],
        )
        for i in range(30)
    ]
    composite, metrics = score_extraction({"claims": claims, "evidence": evidence})
    assert composite > 0.7
    assert metrics["density"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Metric count
# ---------------------------------------------------------------------------


def test_metric_count():
    """score_extraction should return exactly 19 metrics."""
    _, metrics = score_extraction({"claims": [_make_claim()]})
    assert len(metrics) == 19


# ---------------------------------------------------------------------------
# Weight sanity
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_weight_count():
    """METRIC_WEIGHTS should have exactly 19 entries."""
    assert len(METRIC_WEIGHTS) == 19


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
# evidence_linkage (new)
# ---------------------------------------------------------------------------


def test_evidence_linkage_full():
    """All claims have evidence links."""
    data = {
        "claims": [
            {
                "predicate": "induces",
                "evidence_links": [{"evidence_id": "e_001", "direction": "supports"}],
            },
            {
                "predicate": "inhibits",
                "evidence_links": [{"evidence_id": "e_002", "direction": "supports"}],
            },
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["evidence_linkage"] == 1.0


def test_evidence_linkage_none():
    """No claims have evidence links."""
    data = {
        "claims": [
            {"predicate": "induces", "evidence_links": []},
            {"predicate": "inhibits"},
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["evidence_linkage"] == 0.0


# ---------------------------------------------------------------------------
# evidence_density (new)
# ---------------------------------------------------------------------------


def test_evidence_density_in_range():
    """Evidence count in target range (5-30) → score 1.0."""
    evidence = [
        {"evidence_id": f"e_{i:03d}", "result_summary": "x", "readout": "y", "key_figure": "Fig 1"}
        for i in range(10)
    ]
    data = {"claims": [{"predicate": "induces"}], "evidence": evidence}
    _, metrics = score_extraction(data)
    assert metrics["evidence_density"] == 1.0


def test_evidence_density_zero():
    data = {"claims": [{"predicate": "induces"}], "evidence": []}
    _, metrics = score_extraction(data)
    assert metrics["evidence_density"] == 0.0


# ---------------------------------------------------------------------------
# evidence_completeness (new)
# ---------------------------------------------------------------------------


def test_evidence_completeness():
    """Evidence with mixed field completeness."""
    evidence = [
        {"result_summary": "found X", "readout": "qPCR", "key_figure": "Fig 1A"},  # 3/3
        {"result_summary": "found Y", "readout": "Western", "key_figure": ""},  # 2/3
    ]
    data = {"claims": [{"predicate": "induces"}], "evidence": evidence}
    _, metrics = score_extraction(data)
    # First has 3/3, second has 2/3. Average = 5/6
    assert abs(metrics["evidence_completeness"] - 5 / 6) < 0.01


# ---------------------------------------------------------------------------
# doi_coverage (new)
# ---------------------------------------------------------------------------


def test_doi_coverage_all_present():
    data = {
        "claims": [
            {
                "predicate": "induces",
                "section_source": "attributed_prior",
                "source_doi": "10.1234/abc",
            },
            {
                "predicate": "inhibits",
                "section_source": "attributed_prior",
                "source_doi": "10.5678/def",
            },
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["doi_coverage"] == 1.0


def test_doi_coverage_none_present():
    data = {
        "claims": [
            {"predicate": "induces", "section_source": "attributed_prior", "source_doi": None},
            {"predicate": "inhibits", "section_source": "attributed_prior"},
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["doi_coverage"] == 0.0


def test_doi_coverage_no_prior_claims():
    """No attributed_prior claims should return 1.0 (don't penalize)."""
    data = {
        "claims": [
            {"predicate": "induces", "section_source": "primary_empirical"},
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["doi_coverage"] == 1.0


# ---------------------------------------------------------------------------
# conditions_coverage (new)
# ---------------------------------------------------------------------------


def test_conditions_coverage():
    data = {
        "claims": [
            {
                "predicate": "induces",
                "conditions": {"species": ["Mus musculus"], "cell_type": ["mESC"]},
            },  # 2 fields
            {"predicate": "inhibits", "conditions": {"species": ["Homo sapiens"]}},  # 1 field
            {"predicate": "regulates", "conditions": {}},  # 0 fields
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    # 1 out of 3 claims has ≥2 conditions
    assert abs(metrics["conditions_coverage"] - 1 / 3) < 0.01


# ---------------------------------------------------------------------------
# entity_consistency (new)
# ---------------------------------------------------------------------------


def test_entity_consistency_perfect():
    data = {
        "claims": [
            {"predicate": "induces", "subject": {"name": "BMP4"}, "object": {"name": "mesoderm"}},
            {"predicate": "inhibits", "subject": {"name": "BMP4"}, "object": {"name": "ectoderm"}},
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["entity_consistency"] == 1.0


def test_entity_consistency_fragmented():
    data = {
        "claims": [
            {"predicate": "induces", "subject": {"name": "BMP4"}, "object": {"name": "mesoderm"}},
            {"predicate": "inhibits", "subject": {"name": "Bmp4"}, "object": {"name": "mesoderm"}},
        ],
        "evidence": [],
    }
    _, metrics = score_extraction(data)
    # "bmp4" group has 2 variants (BMP4, Bmp4) → 1 fragmented out of 2 groups
    assert metrics["entity_consistency"] == 0.5


# ---------------------------------------------------------------------------
# citation_contexts (new)
# ---------------------------------------------------------------------------


def test_citation_contexts_present():
    data = {
        "claims": [
            {"predicate": "induces", "section_source": "attributed_prior"},
            {"predicate": "inhibits", "section_source": "attributed_prior"},
        ],
        "evidence": [],
        "citation_contexts": [{"cited_doi": "10.1234/abc"}, {"cited_doi": "10.5678/def"}],
    }
    _, metrics = score_extraction(data)
    assert metrics["citation_contexts"] == 1.0


def test_citation_contexts_missing():
    data = {
        "claims": [
            {"predicate": "induces", "section_source": "attributed_prior"},
        ],
        "evidence": [],
        "citation_contexts": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["citation_contexts"] == 0.0


def test_citation_contexts_no_prior():
    """No attributed_prior claims — should return 1.0."""
    data = {
        "claims": [{"predicate": "induces", "section_source": "primary_empirical"}],
        "evidence": [],
        "citation_contexts": [],
    }
    _, metrics = score_extraction(data)
    assert metrics["citation_contexts"] == 1.0


# ---------------------------------------------------------------------------
# Composite score is always in [0, 1]
# ---------------------------------------------------------------------------


def test_composite_in_range():
    """Composite score is always in [0, 1] regardless of input shape."""
    data = {
        "claims": [_make_claim() for _ in range(5)],
        "evidence": [{"result_summary": "x", "readout": "y", "key_figure": "Fig 1"}],
    }
    composite, _ = score_extraction(data)
    assert 0.0 <= composite <= 1.0
