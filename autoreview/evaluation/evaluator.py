from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog

from autoreview.evaluation.citation_matcher import match_citations, parse_bibliography_from_markdown
from autoreview.evaluation.llm_scorer import LLMScorer
from autoreview.evaluation.models import (
    EvaluationResult,
    SynthesisScore,
    TopicCoverageScore,
    WritingQualityScore,
)
from autoreview.evaluation.pdf_extractor import extract_bibliography_lines, extract_text_from_pdf
from autoreview.evaluation.report_generator import save_report

logger = structlog.get_logger()

_WEIGHTS = {"citation": 0.25, "synthesis": 0.30, "topic": 0.25, "writing": 0.20}


async def run_evaluation(
    generated_path: Path,
    reference_path: Path,
    output_dir: Path,
    llm,
) -> EvaluationResult:
    generated_text = generated_path.read_text(encoding="utf-8")
    reference_text = extract_text_from_pdf(reference_path)

    gen_refs = parse_bibliography_from_markdown(generated_text)
    ref_refs = extract_bibliography_lines(reference_text)
    citation_score = match_citations(gen_refs, ref_refs)

    scorer = LLMScorer(llm)
    tasks = [
        scorer.score_synthesis(generated_text, reference_text),
        scorer.score_topic_coverage(generated_text, reference_text),
        scorer.score_writing_quality(generated_text, reference_text),
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    synthesis_score: SynthesisScore | None = None
    topic_coverage: TopicCoverageScore | None = None
    writing_quality: WritingQualityScore | None = None

    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            logger.warning("evaluator.scoring_failed", index=i, error=str(r))
        elif i == 0:
            synthesis_score = r
        elif i == 1:
            topic_coverage = r
        elif i == 2:
            writing_quality = r

    if synthesis_score is None:
        synthesis_score = SynthesisScore(
            generated_score=0, reference_score=0, delta=0,
            dimension_scores={}, generated_observations="error", reference_observations="error",
        )
    if topic_coverage is None:
        topic_coverage = TopicCoverageScore(
            generated_coverage=0, reference_coverage=1.0,
            topics_in_both=[], topics_only_in_reference=[], topics_only_in_generated=[],
        )
    if writing_quality is None:
        writing_quality = WritingQualityScore(
            generated_score=0, reference_score=0, delta=0, dimension_scores={}
        )

    overall_score = round(
        _WEIGHTS["citation"] * citation_score.recall
        + _WEIGHTS["synthesis"] * (synthesis_score.generated_score / 5)
        + _WEIGHTS["topic"] * topic_coverage.generated_coverage
        + _WEIGHTS["writing"] * (writing_quality.generated_score / 5),
        4,
    )

    result = EvaluationResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        generated_path=str(generated_path),
        reference_path=str(reference_path),
        citation_score=citation_score,
        synthesis_score=synthesis_score,
        topic_coverage=topic_coverage,
        writing_quality=writing_quality,
        overall_score=overall_score,
    )

    save_report(result, output_dir)
    logger.info("evaluator.complete", overall_score=overall_score)
    return result
