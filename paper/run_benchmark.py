"""Benchmark orchestrator for the AutoReview paper.

Subcommands:
    generate-matrix  Show all runs with dedup and cost estimate
    run              Execute pipeline runs
    evaluate         Evaluate completed runs
    analyze          Run analysis scripts
    full             run -> evaluate -> analyze
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
import typer

from paper.models import (
    SONNET_MODEL,
    RunKey,
    RunRegistry,
    TopicsConfig,
    expand_run_matrix,
    load_topics,
    make_run_key,
    parse_run_key,
)

app = typer.Typer(
    name="run-benchmark", help="Benchmark orchestrator for AutoReview paper.", no_args_is_help=True
)
logger = structlog.get_logger()

# Default model list
DEFAULT_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]

# Rough per-run cost estimates by (model_prefix, depth) in USD
_COST_ESTIMATES: dict[tuple[str, str], float] = {
    ("opus", "medium"): 15.0,
    ("opus", "low"): 6.0,
    ("opus", "deep"): 35.0,
    ("sonnet", "medium"): 2.0,
    ("sonnet", "low"): 0.80,
    ("sonnet", "deep"): 5.0,
    ("haiku", "medium"): 0.20,
    ("haiku", "low"): 0.08,
    ("haiku", "deep"): 0.50,
}

# Writing-only cost estimates (stages 8-15 only, ~50-55% of full cost)
_WRITING_COST_ESTIMATES: dict[tuple[str, str], float] = {
    ("opus", "medium"): 8.50,
    ("opus", "low"): 3.50,
    ("opus", "deep"): 20.0,
    ("sonnet", "medium"): 1.00,
    ("sonnet", "low"): 0.40,
    ("sonnet", "deep"): 2.50,
    ("haiku", "medium"): 0.10,
    ("haiku", "low"): 0.04,
    ("haiku", "deep"): 0.25,
}


def _model_tier(model: str) -> str:
    if "opus" in model:
        return "opus"
    if "haiku" in model:
        return "haiku"
    return "sonnet"


def estimate_cost(matrix: list[RunKey], *, shared_corpus: bool = True) -> float:
    """Estimate total API cost for the run matrix.

    Args:
        matrix: List of run keys.
        shared_corpus: If True, non-corpus runs use writing-only cost estimates.
    """
    total = 0.0
    for key in matrix:
        _, model, depth, _ = key
        tier = _model_tier(model)
        if shared_corpus and not _is_corpus_run(key):
            total += _WRITING_COST_ESTIMATES.get((tier, depth), 1.0)
        else:
            total += _COST_ESTIMATES.get((tier, depth), 2.0)
    return total


def _format_matrix_summary(matrix: list[RunKey], registry: RunRegistry) -> str:
    """Format a human-readable summary of the run matrix."""
    lines: list[str] = []
    completed = sum(1 for k in matrix if registry.is_completed(make_run_key(*k)))
    remaining = len(matrix) - completed

    lines.append(f"Total unique runs: {len(matrix)}")
    lines.append(f"  Completed: {completed}")
    lines.append(f"  Remaining: {remaining}")
    lines.append(
        f"  Estimated cost (remaining): ${estimate_cost([k for k in matrix if not registry.is_completed(make_run_key(*k))]):.0f}"
    )

    # Corpus vs writing breakdown
    corpus_count = sum(1 for k in matrix if _is_corpus_run(k))
    writing_count = len(matrix) - corpus_count
    lines.append(f"  Corpus (full Sonnet): {corpus_count}")
    lines.append(f"  Writing-only: {writing_count}")

    # Breakdown by batch
    batch_counts: dict[str, int] = {}
    for topic_id, model, depth, condition in matrix:
        if condition in (
            "no_evidence_chains",
            "no_critique_loops",
            "no_passage_mining",
            "no_comprehensiveness",
        ):
            batch = "3e: Ablation"
        elif condition == "retrieval_controlled":
            batch = "3c: Retrieval-controlled"
        elif depth != "medium":
            batch = "3f: Depth"
        else:
            batch = "3a/3b: End-to-end"
        batch_counts[batch] = batch_counts.get(batch, 0) + 1

    lines.append("\nBy batch:")
    for batch, count in sorted(batch_counts.items()):
        lines.append(f"  {batch}: {count} runs")

    return "\n".join(lines)


def _classify_batch(key: RunKey) -> str:
    """Classify a run key into its batch label."""
    _, model, depth, condition = key
    if condition in (
        "no_evidence_chains",
        "no_critique_loops",
        "no_passage_mining",
        "no_comprehensiveness",
    ):
        return "3e"
    if condition == "retrieval_controlled":
        return "3c"
    if depth != "medium":
        return "3f"
    # Distinguish 3a (Tier B) from 3b (Tier A) would require topic lookup;
    # for filtering purposes, both are "3ab"
    return "3ab"


def _is_corpus_run(key: RunKey) -> bool:
    """Check if a run is a corpus-producing run (Sonnet e2e or RC, medium depth)."""
    _, model, depth, condition = key
    return (
        model == SONNET_MODEL
        and depth == "medium"
        and condition in ("end_to_end", "retrieval_controlled")
    )


def _find_corpus_snapshot(
    registry: RunRegistry,
    topic_id: str,
    condition: str,
) -> Path | None:
    """Find the corpus snapshot path for a writing-only run."""
    corpus_condition = (
        "retrieval_controlled" if condition == "retrieval_controlled" else "end_to_end"
    )
    corpus_key = make_run_key(topic_id, SONNET_MODEL, "medium", corpus_condition)

    entry = registry.runs.get(corpus_key)
    if not entry or entry.status != "completed":
        return None

    snapshot_node = "clustering" if condition == "no_comprehensiveness" else "gap_search"

    snapshots_dir = Path(entry.output_dir) / "snapshots"
    if not snapshots_dir.exists():
        return None

    matches = sorted(snapshots_dir.glob(f"*_{snapshot_node}.json"))
    return matches[-1] if matches else None


@app.command(name="generate-matrix")
def generate_matrix(
    topics_path: Path = typer.Option(
        Path("paper/topics.yaml"), "--topics", help="Path to topics YAML"
    ),
    results_dir: Path = typer.Option(
        Path("paper/results"), "--results-dir", help="Results directory"
    ),
    models: str = typer.Option(
        ",".join(DEFAULT_MODELS), "--models", help="Comma-separated model list"
    ),
) -> None:
    """Show the full run matrix with dedup and cost estimate."""
    config = load_topics(topics_path)
    model_list = [m.strip() for m in models.split(",")]
    matrix = expand_run_matrix(config.topics, model_list, include_depth=True)
    registry = RunRegistry.load(results_dir / "run_registry.json")

    typer.echo(_format_matrix_summary(matrix, registry))


@app.command()
def run(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    models: str = typer.Option(",".join(DEFAULT_MODELS), "--models"),
    max_concurrent: int = typer.Option(2, "--max-concurrent"),
    batch_filter: str | None = typer.Option(
        None, "--batch", help="Filter to batches: 3a,3b,3c,3e,3f"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Execute pipeline runs."""
    _setup_logging(verbose)
    config = load_topics(topics_path)
    model_list = [m.strip() for m in models.split(",")]
    matrix = expand_run_matrix(config.topics, model_list, include_depth=True)
    registry = RunRegistry.load(results_dir / "run_registry.json")

    topic_lookup = {t.id: t for t in config.topics}

    # Filter to remaining runs
    remaining = [
        k
        for k in matrix
        if not registry.is_completed(make_run_key(*k))
        and registry.runs.get(make_run_key(*k), type("E", (), {"status": "pending"})()).status
        != "permanently_failed"
    ]

    # Apply batch filter if specified
    if batch_filter:
        allowed = {b.strip() for b in batch_filter.split(",")}
        remaining = [k for k in remaining if _classify_batch(k) in allowed]

    typer.echo(f"Remaining runs: {len(remaining)} of {len(matrix)} total")
    if dry_run:
        typer.echo("Dry run — no execution.")
        return

    asyncio.run(_execute_runs(remaining, topic_lookup, registry, results_dir, max_concurrent))


async def _execute_runs(
    runs: list[RunKey],
    topic_lookup: dict[str, Any],
    registry: RunRegistry,
    results_dir: Path,
    max_concurrent: int,
) -> None:
    """Execute pipeline runs with shared-corpus optimization.

    Phase 1: Sonnet corpus runs (e2e + RC) — full pipeline producing snapshots.
    Phase 2: All other runs — fork from corpus snapshots when available,
             falling back to full pipeline if no snapshot exists.
    """
    from autoreview.config import load_config
    from autoreview.config.models import DepthLevel
    from autoreview.llm.factory import create_llm_provider
    from autoreview.models.knowledge_base import KnowledgeBase
    from autoreview.pipeline.runner import run_pipeline

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_single(key: RunKey) -> None:
        topic_id, model, depth, condition = key
        run_key_str = make_run_key(*key)
        topic = topic_lookup.get(topic_id)
        if not topic:
            logger.error("benchmark.unknown_topic", topic_id=topic_id)
            return

        async with semaphore:
            logger.info("benchmark.run.start", key=run_key_str)
            registry.register_start(run_key_str)

            output_dir = str(results_dir / topic_id / f"{model}_{depth}_{condition}")

            try:
                overrides: dict[str, Any] = {"llm": {"model": model}}
                config = load_config(domain=topic.domain, overrides=overrides)
                config.writing.depth = DepthLevel(depth)

                if topic.date_range:
                    config.search.date_range = topic.date_range

                # Ablation config flags
                if condition == "no_critique_loops":
                    config.critique.max_revision_cycles = 0
                if condition == "no_evidence_chains":
                    config.writing.evidence_chains = False

                llm = create_llm_provider(config.llm)

                # --- Shared-corpus fork logic ---
                snapshot_path = None
                if not _is_corpus_run(key):
                    snapshot_path = _find_corpus_snapshot(registry, topic_id, condition)

                if snapshot_path:
                    # Phase 2: Fork from corpus snapshot — writing stages only
                    logger.info(
                        "benchmark.run.fork",
                        key=run_key_str,
                        snapshot=str(snapshot_path),
                    )
                    kb = KnowledgeBase.load_snapshot(str(snapshot_path))
                    kb.output_dir = output_dir
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    skip_nodes: set[str] | None = None
                    if condition == "no_passage_mining":
                        skip_nodes = {"passage_search"}

                    kb = await run_pipeline(
                        llm=llm,
                        config=config,
                        kb=kb,
                        start_from="outline",
                        skip_nodes=skip_nodes,
                    )

                elif condition == "retrieval_controlled":
                    # Full retrieval-controlled pipeline
                    from paper.analysis.inject_bibliography import inject_bibliography

                    kb = await inject_bibliography(
                        pdf_path=Path(topic.reference.pdf_path),
                        topic=topic.title,
                        domain=topic.domain,
                        output_dir=output_dir,
                        llm=llm,
                    )
                    kb = await run_pipeline(
                        llm=llm, config=config, kb=kb, start_from="full_text_retrieval"
                    )

                else:
                    # Full pipeline (corpus run or fallback when no snapshot)
                    if not _is_corpus_run(key):
                        logger.warning(
                            "benchmark.run.no_corpus_snapshot",
                            key=run_key_str,
                            msg="Falling back to full pipeline",
                        )
                    kb = KnowledgeBase(
                        topic=topic.title, domain=topic.domain, output_dir=output_dir
                    )
                    skip_list: list[str] = []
                    if condition == "no_passage_mining":
                        skip_list.append("passage_search")
                    elif condition == "no_comprehensiveness":
                        skip_list.append("gap_search")
                    kb = await run_pipeline(
                        llm=llm,
                        config=config,
                        kb=kb,
                        skip_nodes=set(skip_list) if skip_list else None,
                    )

                # Register completion
                review_path = next(Path(output_dir).glob("*.md"), Path(output_dir) / "review.md")
                tokens = kb.total_tokens()

                registry.register_complete(
                    run_key_str,
                    output_dir=output_dir,
                    review_path=str(review_path),
                    cost_usd=0.0,
                    tokens_input=tokens.get("input_tokens", 0),
                    tokens_output=tokens.get("output_tokens", 0),
                )
                logger.info("benchmark.run.complete", key=run_key_str)

            except Exception as e:
                logger.error("benchmark.run.failed", key=run_key_str, error=str(e))
                registry.register_failure(run_key_str, str(e))

            finally:
                registry.save(results_dir / "run_registry.json")

    # Split into corpus (Phase 1) and writing (Phase 2) runs
    corpus_runs = [k for k in runs if _is_corpus_run(k)]
    writing_runs = [k for k in runs if not _is_corpus_run(k)]

    if corpus_runs:
        logger.info("benchmark.phase1.start", n_corpus=len(corpus_runs))
        await asyncio.gather(*[_run_single(k) for k in corpus_runs], return_exceptions=True)
        logger.info("benchmark.phase1.complete")

    if writing_runs:
        logger.info("benchmark.phase2.start", n_writing=len(writing_runs))
        await asyncio.gather(*[_run_single(k) for k in writing_runs], return_exceptions=True)
        logger.info("benchmark.phase2.complete")


@app.command()
def evaluate(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    max_concurrent: int = typer.Option(5, "--max-concurrent"),
    judge_model: str = typer.Option("claude-sonnet-4-6", "--judge-model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate completed runs with both rubrics."""
    _setup_logging(verbose)
    config = load_topics(topics_path)
    registry = RunRegistry.load(results_dir / "run_registry.json")
    topic_lookup = {t.id: t for t in config.topics}

    asyncio.run(_evaluate_runs(registry, topic_lookup, results_dir, max_concurrent, judge_model))


async def _evaluate_runs(
    registry: RunRegistry,
    topic_lookup: dict[str, Any],
    results_dir: Path,
    max_concurrent: int,
    judge_model: str,
) -> None:
    """Evaluate all completed, unevaluated runs."""
    from autoreview.config.models import LLMConfig
    from autoreview.evaluation.arise_rubric import ARISERubricScorer
    from autoreview.evaluation.evaluator import run_evaluation
    from autoreview.llm.factory import create_llm_provider

    judge_config = LLMConfig(model=judge_model)
    judge_llm = create_llm_provider(judge_config)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval_single(key: str) -> None:
        entry = registry.runs[key]
        topic_id = parse_run_key(key)[0]
        topic = topic_lookup.get(topic_id)
        if not topic:
            return

        async with semaphore:
            logger.info("benchmark.eval.start", key=key)
            run_dir = Path(entry.output_dir)
            eval_path = run_dir / "evaluation.json"

            try:
                result = await run_evaluation(
                    generated_path=Path(entry.review_path),
                    reference_path=Path(topic.reference.pdf_path),
                    output_dir=run_dir,
                    judge_llm=judge_llm,
                )

                # ARISE rubric
                scorer = ARISERubricScorer(judge_llm)
                gen_text = Path(entry.review_path).read_text(encoding="utf-8")
                result.arise_result = await scorer.score(gen_text)

                eval_path.write_text(result.model_dump_json(indent=2))
                registry.register_evaluation(key, str(eval_path))
                registry.save(results_dir / "run_registry.json")
                logger.info("benchmark.eval.complete", key=key)

            except Exception as e:
                logger.error("benchmark.eval.failed", key=key, error=str(e))

    # AutoReview runs
    tasks = []
    for key, entry in registry.runs.items():
        if entry.status == "completed" and not entry.evaluation_path:
            tasks.append(_eval_single(key))

    # ARISE outputs
    arise_dir = results_dir / "arise"
    if arise_dir.exists():
        for topic_dir in arise_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            eval_path = topic_dir / "evaluation.json"
            if eval_path.exists():
                continue
            review_path = topic_dir / "review.md"
            if not review_path.exists():
                continue

            topic = topic_lookup.get(topic_dir.name)
            if not topic:
                continue

            async def _eval_arise(tp: Any, rp: Path, ep: Path) -> None:
                async with semaphore:
                    try:
                        result = await run_evaluation(
                            generated_path=rp,
                            reference_path=Path(tp.reference.pdf_path),
                            output_dir=rp.parent,
                            judge_llm=judge_llm,
                        )
                        scorer = ARISERubricScorer(judge_llm)
                        result.arise_result = await scorer.score(rp.read_text(encoding="utf-8"))
                        ep.write_text(result.model_dump_json(indent=2))
                    except Exception as e:
                        logger.error("benchmark.eval_arise.failed", topic=tp.id, error=str(e))

            tasks.append(_eval_arise(topic, review_path, eval_path))

    await asyncio.gather(*tasks, return_exceptions=True)


@app.command()
def analyze(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    output_dir: Path = typer.Option(Path("paper/output"), "--output-dir"),
    analyses: str | None = typer.Option(
        None, "--analyses", help="Comma-separated analysis numbers, e.g. 1,2,3"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run analysis scripts on evaluation results."""
    _setup_logging(verbose)
    config = load_topics(topics_path)

    which: list[int] | None = None
    if analyses:
        which = [int(x.strip()) for x in analyses.split(",")]

    asyncio.run(_run_analyses(config, results_dir, output_dir, which))


async def _run_analyses(
    topics: TopicsConfig,
    results_dir: Path,
    output_dir: Path,
    which: list[int] | None,
) -> None:
    """Run selected analysis scripts."""
    from paper.analysis.common import load_all_evaluations

    df = load_all_evaluations(results_dir, topics)
    logger.info("benchmark.analyze.loaded", n_rows=len(df))

    analysis_map: dict[int, tuple[str, Any]] = {
        1: ("main_comparison", None),
        2: ("domain_analysis", None),
        3: ("rubric_agreement", None),
        4: ("ablation_analysis", None),
        5: ("retrieval_decomposition", None),
        6: ("citation_analysis", None),
        7: ("model_comparison", None),
        8: ("cost_analysis", None),
        9: ("contamination_analysis", None),
        10: ("depth_comparison", None),
        11: ("arise_comparison", None),
    }

    to_run = which or list(analysis_map.keys())

    for n in to_run:
        name, _ = analysis_map.get(n, (None, None))
        if not name:
            logger.warning("benchmark.analyze.unknown", analysis=n)
            continue

        analysis_out = output_dir / f"analysis_{n:02d}_{name}"
        logger.info("benchmark.analyze.start", analysis=n, name=name)

        try:
            if n == 10:
                # Generate depth_runs.json shim then call depth_comparison
                _generate_depth_shim(results_dir, analysis_out, topics)
                from paper.analysis.depth_comparison import main as depth_main

                await depth_main(analysis_out, analysis_out, skip_extraction=True)
            else:
                mod = __import__(f"paper.analysis.{name}", fromlist=["main"])
                await mod.main(results_dir, analysis_out, df)
        except Exception as e:
            logger.error("benchmark.analyze.failed", analysis=n, error=str(e))


def _generate_depth_shim(results_dir: Path, output_dir: Path, topics: TopicsConfig) -> None:
    """Generate depth_runs.json from registry for Analysis 10."""
    import json

    from paper.models import RunRegistry, parse_run_key

    topic_lookup = {t.id: t for t in topics.topics}
    registry = RunRegistry.load(results_dir / "run_registry.json")
    runs = []
    for key, entry in registry.runs.items():
        if entry.status != "completed":
            continue
        topic_id, model, depth, condition = parse_run_key(key)
        if model != SONNET_MODEL or condition != "end_to_end":
            continue
        topic_info = topic_lookup.get(topic_id)
        runs.append(
            {
                "topic": topic_id,
                "domain": topic_info.domain if topic_info else "unknown",
                "depth": depth,
                "generated_path": entry.review_path,
                "reference_path": topic_info.reference.pdf_path if topic_info else "",
                "evaluation_path": entry.evaluation_path,
                "tier": topic_info.tier if topic_info else "unknown",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_runs.json").write_text(json.dumps({"runs": runs}, indent=2))


@app.command()
def full(
    topics_path: Path = typer.Option(Path("paper/topics.yaml"), "--topics"),
    results_dir: Path = typer.Option(Path("paper/results"), "--results-dir"),
    output_dir: Path = typer.Option(Path("paper/output"), "--output-dir"),
    models: str = typer.Option(",".join(DEFAULT_MODELS), "--models"),
    max_concurrent: int = typer.Option(2, "--max-concurrent"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full benchmark: generate -> evaluate -> analyze."""
    _setup_logging(verbose)
    typer.echo("=== Phase 1: Pipeline Runs ===")
    run(
        topics_path=topics_path,
        results_dir=results_dir,
        models=models,
        max_concurrent=max_concurrent,
        batch_filter=None,
        dry_run=False,
        verbose=verbose,
    )
    typer.echo("\n=== Phase 2: Evaluations ===")
    evaluate(
        topics_path=topics_path,
        results_dir=results_dir,
        max_concurrent=5,
        judge_model="claude-sonnet-4-6",
        verbose=verbose,
    )
    typer.echo("\n=== Phase 3: Analyses ===")
    analyze(
        topics_path=topics_path,
        results_dir=results_dir,
        output_dir=output_dir,
        analyses=None,
        verbose=verbose,
    )


def _setup_logging(verbose: bool = False) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if verbose else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.processors.NAME_TO_LEVEL["debug" if verbose else "info"]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


if __name__ == "__main__":
    app()
