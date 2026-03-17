import pytest
from pydantic import ValidationError

from autoreview.evaluation.models import (
    CitationScore,
    Claim,
    DepthComparisonResult,
    DepthRunConfig,
    EvaluationResult,
    InformationMetrics,
    SynthesisScore,
    TopicCoverageScore,
    WritingQualityScore,
)


def test_citation_score_fields():
    score = CitationScore(
        recall=0.62,
        precision=0.61,
        f1=0.615,
        matched_count=22,
        reference_count=35,
        generated_count=36,
        matched_titles=["Title A"],
        missed_titles=["Title B"],
        hallucinated_titles=["Title C"],
    )
    assert score.recall == 0.62
    assert score.matched_count == 22


def test_synthesis_score_delta():
    score = SynthesisScore(
        generated_score=3.8,
        reference_score=4.2,
        delta=-0.4,
        dimension_scores={"cross_paper": 3.5},
        generated_observations="ok",
        reference_observations="great",
    )
    assert score.delta == -0.4


def test_evaluation_result_roundtrip():
    result = EvaluationResult(
        timestamp="2026-02-24T00:00:00",
        generated_path="output/review.md",
        reference_path="reference.pdf",
        citation_score=CitationScore(
            recall=0.62,
            precision=0.61,
            f1=0.615,
            matched_count=22,
            reference_count=35,
            generated_count=36,
            matched_titles=[],
            missed_titles=[],
            hallucinated_titles=[],
        ),
        synthesis_score=SynthesisScore(
            generated_score=3.8,
            reference_score=4.2,
            delta=-0.4,
            dimension_scores={},
            generated_observations="",
            reference_observations="",
        ),
        topic_coverage=TopicCoverageScore(
            generated_coverage=0.70,
            reference_coverage=1.0,
            topics_in_both=[],
            topics_only_in_reference=[],
            topics_only_in_generated=[],
        ),
        writing_quality=WritingQualityScore(
            generated_score=3.5, reference_score=4.0, delta=-0.5, dimension_scores={}
        ),
        overall_score=0.65,
    )
    data = result.model_dump()
    assert data["overall_score"] == 0.65
    assert data["citation_score"]["recall"] == 0.62


def test_claim_model():
    claim = Claim(
        text="GLP-1 agonists reduce HbA1c by 1.5%",
        category="empirical",
        section_id="results_1",
    )
    assert claim.text == "GLP-1 agonists reduce HbA1c by 1.5%"
    assert claim.category == "empirical"
    assert claim.section_id == "results_1"


def test_claim_category_validation():
    """Only valid categories are accepted."""
    with pytest.raises(ValidationError):
        Claim.model_validate({"text": "test", "category": "invalid_category"})


def test_claim_optional_section_id():
    claim = Claim(text="test claim", category="synthesis")
    assert claim.section_id is None


def test_information_metrics():
    metrics = InformationMetrics(
        claims_per_depth={"low": 45, "medium": 102, "deep": 230},
        new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
        new_claims_by_category={
            "low_to_medium": {
                "empirical": 20,
                "methodological": 15,
                "contextual": 10,
                "synthesis": 8,
                "limitation": 4,
            },
            "medium_to_deep": {
                "empirical": 50,
                "methodological": 30,
                "contextual": 25,
                "synthesis": 15,
                "limitation": 8,
            },
        },
        claim_novelty_rate={"low_to_medium": 0.559, "medium_to_deep": 0.557},
        concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
        concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
        claims_per_1k_words={"low": 11.25, "medium": 12.75, "deep": 9.2},
        concepts_per_1k_words={"low": 7.5, "medium": 8.125, "deep": 4.8},
        citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
    )
    assert metrics.claims_per_depth["medium"] == 102
    assert metrics.reference_claim_coverage is None


def test_information_metrics_with_reference_coverage():
    metrics = InformationMetrics(
        claims_per_depth={"low": 45, "medium": 102, "deep": 230},
        new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
        new_claims_by_category={"low_to_medium": {}, "medium_to_deep": {}},
        claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
        concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
        concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
        claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
        concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
        citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
        reference_claim_coverage={"low": 0.35, "medium": 0.62, "deep": 0.81},
    )
    assert metrics.reference_claim_coverage is not None
    assert metrics.reference_claim_coverage["deep"] == 0.81


def test_depth_run_config():
    cfg = DepthRunConfig(
        topic="gut microbiome",
        domain="biomedical",
        depth="medium",
        generated_path="/output/medium/review.md",
        reference_path="/refs/gut_microbiome.pdf",
        evaluation_path="/output/medium/evaluation.json",
        tier="A",
    )
    assert cfg.depth == "medium"
    assert cfg.tier == "A"


def test_depth_run_config_no_reference():
    """Tier B topics may have no reference review."""
    cfg = DepthRunConfig(
        topic="llm hallucination",
        domain="cs",
        depth="low",
        generated_path="/output/low/review.md",
        tier="B",
    )
    assert cfg.reference_path is None


def test_depth_comparison_result():
    result = DepthComparisonResult(
        topic="gut microbiome",
        domain="biomedical",
        depths=["low", "medium", "deep"],
        evaluation_scores={"low": 0.65, "medium": 0.78, "deep": 0.82},
        information_metrics=InformationMetrics(
            claims_per_depth={"low": 45, "medium": 102, "deep": 230},
            new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
            new_claims_by_category={"low_to_medium": {}, "medium_to_deep": {}},
            claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
            concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
            concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
            claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
            concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
            citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
        ),
        cost_per_depth={"low": 1.50, "medium": 3.20, "deep": 8.75},
    )
    assert result.evaluation_scores["deep"] == 0.82
