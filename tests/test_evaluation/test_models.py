from autoreview.evaluation.models import (
    CitationScore,
    EvaluationResult,
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
