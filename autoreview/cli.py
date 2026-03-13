from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="autoreview",
    help="Fully autonomous pipeline for generating publication-ready scientific review papers.",
    no_args_is_help=True,
)


def _clear_output_dir(output_dir: str) -> None:
    """Remove snapshots and generated outputs from a previous run."""
    import shutil

    out = Path(output_dir)
    if not out.exists():
        return

    snapshots = out / "snapshots"
    if snapshots.exists():
        shutil.rmtree(snapshots)

    # Remove generated review files but keep the directory itself
    for ext in ("*.md", "*.tex", "*.docx", "*.json"):
        for f in out.glob(ext):
            f.unlink()


def _setup_logging(verbose: bool = False) -> None:
    """Configure structlog."""
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


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic or question for the review"),
    domain: str = typer.Option(
        "general", "--domain", "-d", help="Domain preset (biomedical, cs_ai, chemistry, general)"
    ),
    output_dir: str = typer.Option("output", "--output-dir", "-o", help="Output directory"),
    output_format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, latex, docx)"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model"),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider (claude, ollama). Auto-detected if omitted."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Clear all previous snapshots and outputs before running"
    ),
    date_range: str | None = typer.Option(
        None, "--date-range", help="Year range filter, e.g. '2015-2020', '-2019', '2020-'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run the full AutoReview pipeline to generate a review paper."""
    _setup_logging(verbose)
    logger = structlog.get_logger()

    if fresh:
        _clear_output_dir(output_dir)
        typer.echo("Cleared previous outputs (--fresh)")

    logger.info(
        "pipeline.start",
        topic=topic,
        domain=domain,
        output_dir=output_dir,
        output_format=output_format,
        fresh=fresh,
    )

    from autoreview.config import load_config
    from autoreview.llm.factory import create_llm_provider
    from autoreview.models.knowledge_base import KnowledgeBase

    overrides: dict = {}
    if model:
        overrides["llm"] = {"model": model}
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider

    config = load_config(domain=domain, overrides=overrides or None)

    if date_range is not None:
        config.search.date_range = date_range

    kb = KnowledgeBase(
        topic=topic,
        domain=domain,
        output_dir=output_dir,
    )

    typer.echo(f"AutoReview: Generating review paper on '{topic}'")
    typer.echo(f"Domain: {domain} | Output: {output_dir} | Format: {output_format}")

    kb.save_snapshot("initialized")

    from autoreview.output.formatter import OutputFormatter
    from autoreview.pipeline.runner import run_pipeline

    try:
        llm = create_llm_provider(config.llm, provider=provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    try:
        kb = asyncio.run(run_pipeline(llm=llm, config=config, kb=kb))
    except Exception as e:
        logger.error("pipeline.failed", error=str(e))
        typer.echo(f"Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    formatter = OutputFormatter(style=config.writing.citation_format)
    created = formatter.save(kb, output_dir, fmt=output_format)

    typer.echo("Review paper generated successfully!")
    for path in created:
        typer.echo(f"  -> {path}")


@app.command()
def resume(
    snapshot: str = typer.Argument(..., help="Path to a snapshot JSON file"),
    start_from: str | None = typer.Option(
        None, "--start-from", "-s", help="DAG node to start from"
    ),
    output_format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, latex, docx)"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="LLM model to use (e.g. qwen3.5:35b)"
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider (claude, ollama). Auto-detected if omitted."
    ),
    date_range: str | None = typer.Option(
        None, "--date-range", help="Year range filter, e.g. '2015-2020', '-2019', '2020-'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Resume pipeline from a saved snapshot."""
    _setup_logging(verbose)

    from autoreview.models.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.load_snapshot(snapshot)
    typer.echo(f"Loaded snapshot: {snapshot}")
    typer.echo(f"Topic: {kb.topic} | Phase: {kb.current_phase}")
    typer.echo(f"Papers: {len(kb.candidate_papers)} candidates, {len(kb.screened_papers)} screened")

    from autoreview.config import load_config
    from autoreview.llm.factory import create_llm_provider
    from autoreview.output.formatter import OutputFormatter
    from autoreview.pipeline.runner import run_pipeline

    config = load_config(domain=kb.domain)

    if date_range is not None:
        config.search.date_range = date_range

    if model:
        config.llm.model = model

    try:
        llm = create_llm_provider(config.llm, provider=provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    try:
        kb = asyncio.run(run_pipeline(llm=llm, config=config, kb=kb, start_from=start_from))
    except Exception as e:
        typer.echo(f"Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    formatter = OutputFormatter(style=config.writing.citation_format)
    created = formatter.save(kb, kb.output_dir, fmt=output_format)

    typer.echo("Review paper generated successfully!")
    for path in created:
        typer.echo(f"  -> {path}")


@app.command()
def inspect(
    snapshot: str = typer.Argument(..., help="Path to a snapshot JSON file"),
) -> None:
    """Inspect a pipeline snapshot."""
    from autoreview.models.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.load_snapshot(snapshot)
    typer.echo(f"Topic: {kb.topic}")
    typer.echo(f"Domain: {kb.domain}")
    typer.echo(f"Phase: {kb.current_phase}")
    typer.echo(f"Candidate papers: {len(kb.candidate_papers)}")
    typer.echo(f"Screened papers: {len(kb.screened_papers)}")
    typer.echo(f"Extractions: {len(kb.extractions)}")
    typer.echo(f"Evidence map: {'yes' if kb.evidence_map else 'no'}")
    typer.echo(f"Section drafts: {len(kb.section_drafts)}")
    typer.echo(f"Full draft: {'yes' if kb.full_draft else 'no'}")
    typer.echo(f"Critique history: {len(kb.critique_history)} reports")
    typer.echo(f"Audit log: {len(kb.audit_log)} entries")

    tokens = kb.total_tokens()
    if tokens["input_tokens"] > 0 or tokens["output_tokens"] > 0:
        typer.echo(
            f"Total tokens: {tokens['input_tokens']:,} input, {tokens['output_tokens']:,} output"
        )


@app.command()
def evaluate(
    generated: str = typer.Argument(..., help="Path to generated review Markdown file"),
    reference: str = typer.Argument(..., help="Path to reference PDF file"),
    output_dir: str = typer.Option(
        "output/evaluations", "--output-dir", "-o", help="Directory for evaluation outputs"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model"),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider (claude, ollama). Auto-detected if omitted."
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Override LLM model used for judging (defaults to --model if omitted).",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="LLM provider for judging (defaults to --provider if omitted).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Evaluate a generated review against a published reference PDF."""
    _setup_logging(verbose)

    from autoreview.config import load_config
    from autoreview.config.models import LLMConfig
    from autoreview.evaluation.evaluator import run_evaluation
    from autoreview.llm.factory import create_llm_provider

    overrides: dict = {}
    if model:
        overrides["llm"] = {"model": model}
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider

    config = load_config(overrides=overrides or None)

    try:
        llm = create_llm_provider(config.llm, provider=provider)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if judge_model:
        try:
            judge_config = LLMConfig(model=judge_model, api_key=config.llm.api_key)
            judge_llm = create_llm_provider(judge_config, provider=judge_provider or provider)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from e
    else:
        judge_llm = llm

    typer.echo(f"Evaluating: {generated}")
    typer.echo(f"Against: {reference}")

    result = asyncio.run(
        run_evaluation(
            generated_path=Path(generated),
            reference_path=Path(reference),
            output_dir=Path(output_dir),
            judge_llm=judge_llm,
        )
    )

    typer.echo(f"\nOverall score: {result.overall_score:.2f}")
    typer.echo(f"  Citation recall:   {result.citation_score.recall:.1%}")
    typer.echo(
        f"  Synthesis depth:   {result.synthesis_score.generated_score:.1f}/5 "
        f"(ref: {result.synthesis_score.reference_score:.1f})"
    )
    typer.echo(f"  Topical coverage:  {result.topic_coverage.generated_coverage:.1%}")
    typer.echo(
        f"  Writing quality:   {result.writing_quality.generated_score:.1f}/5 "
        f"(ref: {result.writing_quality.reference_score:.1f})"
    )
    typer.echo(f"\nReport saved to: {output_dir}/")


@app.command()
def benchmark(
    topic: str = typer.Argument(..., help="Research topic for benchmark prompts"),
    models: str = typer.Option(..., "--models", help="Comma-separated list of models to benchmark"),
    output_dir: str = typer.Option(
        "output/benchmarks", "--output-dir", "-o", help="Directory for benchmark results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Benchmark LLM providers by running targeted pipeline stages."""
    _setup_logging(verbose)

    from autoreview.benchmark.runner import run_benchmark

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        typer.echo("Error: provide at least one model via --models", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Benchmarking {len(model_list)} model(s): {', '.join(model_list)}")
    typer.echo(f"Topic: {topic}")

    try:
        asyncio.run(run_benchmark(topic=topic, model_names=model_list, output_dir=output_dir))
    except Exception as e:
        typer.echo(f"Benchmark failed: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="batch-evaluate")
def batch_evaluate(
    config_path: str = typer.Argument(..., help="Path to batch evaluation YAML config"),
    output_dir: str = typer.Option(
        "output/batch_eval", "--output-dir", "-o", help="Directory for batch evaluation outputs"
    ),
    judge_model: str = typer.Option(
        "claude-sonnet-4-6", "--judge-model", "-j", help="Model to use for evaluation judging"
    ),
    judge_provider: str | None = typer.Option(
        None, "--judge-provider", help="LLM provider for judge (claude, ollama)"
    ),
    max_concurrent: int = typer.Option(
        3, "--max-concurrent", "-c", help="Max concurrent evaluations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run batch evaluation across multiple topics from a YAML config."""
    from collections import defaultdict

    import yaml

    _setup_logging(verbose)

    from autoreview.evaluation.models import BatchEvaluationConfig

    config_data = yaml.safe_load(Path(config_path).read_text())
    batch_config = BatchEvaluationConfig(**config_data)

    # Override judge model if specified via CLI
    batch_config.judge_model = judge_model

    # Create judge LLM
    from autoreview.config.models import LLMConfig
    from autoreview.llm.factory import create_llm_provider

    judge_config = LLMConfig(model=judge_model)
    judge_llm = create_llm_provider(judge_config, provider=judge_provider)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from autoreview.evaluation.aggregator import aggregate_results
    from autoreview.evaluation.batch_runner import run_batch_evaluation

    results = asyncio.run(run_batch_evaluation(batch_config, judge_llm, out, max_concurrent))

    # Save per-topic results
    for i, r in enumerate(results):
        topic_json = out / f"topic_{i:03d}.json"
        topic_json.write_text(r.model_dump_json(indent=2))

    # Aggregate
    agg = aggregate_results(results)

    # Group by model tier if available
    by_tier: dict[str, list] = defaultdict(list)
    by_domain: dict[str, list] = defaultdict(list)
    for topic_cfg, result in zip(batch_config.topics, results, strict=False):
        if topic_cfg.model_tier:
            by_tier[topic_cfg.model_tier].append(result)
        by_domain[topic_cfg.domain].append(result)

    if by_tier:
        agg.by_model_tier = {k: aggregate_results(v) for k, v in by_tier.items()}
    if by_domain:
        agg.by_domain = {k: aggregate_results(v) for k, v in by_domain.items()}

    # Save aggregated results
    agg_path = out / "aggregated.json"
    agg_path.write_text(agg.model_dump_json(indent=2))

    # Print summary
    print(f"\nBatch Evaluation Complete: {agg.n_topics} topics evaluated")
    print(f"  Overall score:      {agg.overall_score.mean:.3f} ± {agg.overall_score.std:.3f}")
    print(f"  Citation recall:    {agg.citation_recall.mean:.3f} ± {agg.citation_recall.std:.3f}")
    print(
        f"  Citation precision: "
        f"{agg.citation_precision.mean:.3f} ± {agg.citation_precision.std:.3f}"
    )
    print(f"  Citation F1:        {agg.citation_f1.mean:.3f} ± {agg.citation_f1.std:.3f}")
    print(f"  Synthesis:          {agg.synthesis_score.mean:.2f}/5 ± {agg.synthesis_score.std:.2f}")
    print(f"  Topic coverage:     {agg.topic_coverage.mean:.3f} ± {agg.topic_coverage.std:.3f}")
    print(f"  Writing quality:    {agg.writing_quality.mean:.2f}/5 ± {agg.writing_quality.std:.2f}")
    if agg.arise_total:
        print(f"  ARISE total:        {agg.arise_total.mean:.1f}/100 ± {agg.arise_total.std:.1f}")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    app()
