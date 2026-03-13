"""Batch evaluation runner for multiple topics."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from autoreview.evaluation.arise_rubric import ARISERubricScorer
from autoreview.evaluation.evaluator import run_evaluation

if TYPE_CHECKING:
    from autoreview.evaluation.models import BatchEvaluationConfig, EvaluationResult, TopicConfig

logger = structlog.get_logger()


async def _evaluate_single(
    topic: TopicConfig,
    judge_llm: object,
    rubrics: list[str],
    output_dir: Path,
) -> EvaluationResult:
    """Run evaluation for a single topic."""
    topic_dir = output_dir / topic.topic.replace(" ", "_")
    topic_dir.mkdir(parents=True, exist_ok=True)

    result = await run_evaluation(
        generated_path=Path(topic.generated_path),
        reference_path=Path(topic.reference_path),
        output_dir=topic_dir,
        judge_llm=judge_llm,
    )

    # Run ARISE rubric if requested
    if "arise" in rubrics:
        scorer = ARISERubricScorer(judge_llm)
        generated_text = Path(topic.generated_path).read_text(encoding="utf-8")
        result.arise_result = await scorer.score(generated_text)

    return result


async def run_batch_evaluation(
    config: BatchEvaluationConfig,
    judge_llm: object,
    output_dir: Path,
    max_concurrent: int = 3,
) -> list[EvaluationResult]:
    """Run evaluation across multiple topics with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _with_semaphore(topic: TopicConfig) -> EvaluationResult:
        async with semaphore:
            logger.info("batch_runner.evaluating", topic=topic.topic, domain=topic.domain)
            return await _evaluate_single(topic, judge_llm, config.rubrics, output_dir)

    tasks = [_with_semaphore(t) for t in config.topics]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, log them
    valid: list[EvaluationResult] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            logger.error("batch_runner.topic_failed", topic=config.topics[i].topic, error=str(r))
        else:
            valid.append(r)

    logger.info("batch_runner.complete", total=len(config.topics), succeeded=len(valid))
    return valid
