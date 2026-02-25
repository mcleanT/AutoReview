import pytest
from autoreview.evaluation.llm_scorer import LLMScorer
from autoreview.evaluation.models import SynthesisScore, TopicCoverageScore, WritingQualityScore
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.llm.prompts.evaluation import (
    SynthesisScoreResult, TopicCoverageResult, WritingQualityResult,
)


class MockScorerLLM:
    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        if response_model == SynthesisScoreResult:
            return LLMStructuredResponse(
                parsed=SynthesisScoreResult(
                    generated_score=3.8, reference_score=4.2,
                    dimension_scores_generated={"cross_paper": 3.5, "contradiction_handling": 4.0, "gap_identification": 3.5, "evidence_weighting": 4.0, "narrative_coherence": 3.5},
                    dimension_scores_reference={"cross_paper": 4.0, "contradiction_handling": 4.5, "gap_identification": 4.0, "evidence_weighting": 4.0, "narrative_coherence": 4.5},
                    generated_observations="Good.", reference_observations="Excellent.",
                ),
                input_tokens=500, output_tokens=200,
            )
        elif response_model == TopicCoverageResult:
            return LLMStructuredResponse(
                parsed=TopicCoverageResult(
                    topics_in_generated=["SASP", "telomere"],
                    topics_in_reference=["SASP", "telomere", "immune senescence"],
                    topics_matched=["SASP", "telomere"],
                    topics_missed=["immune senescence"],
                    generated_coverage=0.667,
                ),
                input_tokens=500, output_tokens=200,
            )
        elif response_model == WritingQualityResult:
            return LLMStructuredResponse(
                parsed=WritingQualityResult(
                    generated_score=3.5, reference_score=4.0,
                    dimension_scores_generated={"clarity": 3.5, "coherence": 3.5, "academic_register": 3.5, "flow": 3.5},
                    dimension_scores_reference={"clarity": 4.0, "coherence": 4.0, "academic_register": 4.0, "flow": 4.0},
                    generated_strengths="Clear.", generated_weaknesses="Abrupt transitions.",
                ),
                input_tokens=500, output_tokens=200,
            )
        raise ValueError(f"Unexpected: {response_model}")


async def test_score_synthesis():
    scorer = LLMScorer(MockScorerLLM())
    result = await scorer.score_synthesis("gen text", "ref text")
    assert isinstance(result, SynthesisScore)
    assert result.generated_score == 3.8
    assert result.delta == pytest.approx(-0.4, abs=0.01)


async def test_score_topic_coverage():
    scorer = LLMScorer(MockScorerLLM())
    result = await scorer.score_topic_coverage("gen text", "ref text")
    assert isinstance(result, TopicCoverageScore)
    assert abs(result.generated_coverage - 0.667) < 0.001
    assert "SASP" in result.topics_in_both
    assert "immune senescence" in result.topics_only_in_reference


async def test_score_writing_quality():
    scorer = LLMScorer(MockScorerLLM())
    result = await scorer.score_writing_quality("gen text", "ref text")
    assert isinstance(result, WritingQualityScore)
    assert result.generated_score == 3.5
    assert result.delta == pytest.approx(-0.5, abs=0.01)
