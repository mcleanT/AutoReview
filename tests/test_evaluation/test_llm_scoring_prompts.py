from autoreview.llm.prompts.evaluation import (
    SynthesisScoreResult, TopicCoverageResult, WritingQualityResult,
    build_synthesis_prompt, build_topic_coverage_prompt, build_writing_quality_prompt,
)


def test_synthesis_score_result_model():
    result = SynthesisScoreResult(
        generated_score=3.8,
        reference_score=4.2,
        dimension_scores_generated={"cross_paper": 3.5, "contradiction_handling": 4.0, "gap_identification": 3.5, "evidence_weighting": 4.0, "narrative_coherence": 3.5},
        dimension_scores_reference={"cross_paper": 4.0, "contradiction_handling": 4.5, "gap_identification": 4.0, "evidence_weighting": 4.0, "narrative_coherence": 4.5},
        generated_observations="Good synthesis.",
        reference_observations="Excellent.",
    )
    assert result.generated_score == 3.8


def test_topic_coverage_result_model():
    result = TopicCoverageResult(
        topics_in_generated=["SASP", "telomere"],
        topics_in_reference=["SASP", "telomere", "immune senescence"],
        topics_matched=["SASP", "telomere"],
        topics_missed=["immune senescence"],
        generated_coverage=0.667,
    )
    assert abs(result.generated_coverage - 0.667) < 0.001


def test_writing_quality_result_model():
    result = WritingQualityResult(
        generated_score=3.5,
        reference_score=4.0,
        dimension_scores_generated={"clarity": 3.5, "coherence": 3.5, "academic_register": 3.5, "flow": 3.5},
        dimension_scores_reference={"clarity": 4.0, "coherence": 4.0, "academic_register": 4.0, "flow": 4.0},
        generated_strengths="Clear.",
        generated_weaknesses="Abrupt transitions.",
    )
    assert result.generated_score == 3.5


def test_build_synthesis_prompt_includes_both_texts():
    prompt = build_synthesis_prompt("Generated text.", "Reference text.")
    assert "Generated text." in prompt
    assert "Reference text." in prompt


def test_build_topic_coverage_prompt_includes_both_texts():
    prompt = build_topic_coverage_prompt("Gen.", "Ref.")
    assert "Gen." in prompt
    assert "Ref." in prompt


def test_build_writing_quality_prompt_includes_both_texts():
    prompt = build_writing_quality_prompt("Gen.", "Ref.")
    assert "Gen." in prompt
    assert "Ref." in prompt
