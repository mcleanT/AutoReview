import pytest
from pathlib import Path
from unittest.mock import patch
from autoreview.evaluation.evaluator import run_evaluation
from autoreview.evaluation.models import EvaluationResult
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.llm.prompts.evaluation import SynthesisScoreResult, TopicCoverageResult, WritingQualityResult


class MockEvaluatorLLM:
    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        if response_model == SynthesisScoreResult:
            return LLMStructuredResponse(
                parsed=SynthesisScoreResult(
                    generated_score=3.8, reference_score=4.2,
                    dimension_scores_generated={}, dimension_scores_reference={},
                    generated_observations="ok", reference_observations="ok",
                ),
                input_tokens=100, output_tokens=50,
            )
        elif response_model == TopicCoverageResult:
            return LLMStructuredResponse(
                parsed=TopicCoverageResult(
                    topics_in_generated=["A"], topics_in_reference=["A", "B"],
                    topics_matched=["A"], topics_missed=["B"],
                    generated_coverage=0.5,
                ),
                input_tokens=100, output_tokens=50,
            )
        elif response_model == WritingQualityResult:
            return LLMStructuredResponse(
                parsed=WritingQualityResult(
                    generated_score=3.5, reference_score=4.0,
                    dimension_scores_generated={}, dimension_scores_reference={},
                    generated_strengths="ok", generated_weaknesses="ok",
                ),
                input_tokens=100, output_tokens=50,
            )
        raise ValueError(f"Unexpected: {response_model}")


async def test_run_evaluation_returns_result(tmp_path):
    generated_md = tmp_path / "review.md"
    generated_md.write_text(
        "# Review\nSome text.\n\n## References\n[1] Smith J. Senescence. Nature, 2020.\n"
    )
    ref_pdf = tmp_path / "reference.pdf"
    ref_pdf.write_text("")

    with patch(
        "autoreview.evaluation.evaluator.extract_text_from_pdf",
        return_value="Reference text.\n\nReferences\n[1] Smith J. Senescence. Nature, 2020.\n[2] Jones K. SASP. Cell, 2021.\n",
    ):
        result = await run_evaluation(
            generated_path=generated_md,
            reference_path=ref_pdf,
            output_dir=tmp_path / "evals",
            llm=MockEvaluatorLLM(),
        )

    assert isinstance(result, EvaluationResult)
    assert 0 <= result.overall_score <= 1
    assert result.citation_score.reference_count == 2


async def test_run_evaluation_overall_score_range(tmp_path):
    generated_md = tmp_path / "review.md"
    generated_md.write_text("# Review\n\n## References\n[1] Smith J. A. Nature, 2020.\n")
    ref_pdf = tmp_path / "reference.pdf"
    ref_pdf.write_text("")

    with patch(
        "autoreview.evaluation.evaluator.extract_text_from_pdf",
        return_value="References\n[1] Smith J. A. Nature, 2020.\n",
    ):
        result = await run_evaluation(
            generated_path=generated_md,
            reference_path=ref_pdf,
            output_dir=tmp_path,
            llm=MockEvaluatorLLM(),
        )

    assert 0.0 <= result.overall_score <= 1.0
