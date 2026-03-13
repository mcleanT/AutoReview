"""Benchmark runner — compares LLM providers on targeted pipeline stages.

Runs 3 stages per model using self-contained prompts with synthetic data:
  1. Query Expansion (structured output, simple schema)
  2. Paper Extraction (structured output, complex nested schema + enums)
  3. Section Writing (free-form generation)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import Field

from autoreview.config.models import LLMConfig
from autoreview.extraction.models import (
    PaperExtraction,
)
from autoreview.llm.factory import create_llm_provider
from autoreview.llm.provider import LLMProvider
from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Benchmark-local model (mirrors the pipeline's inline QueryExpansionResult)
# ---------------------------------------------------------------------------
class QueryExpansionResult(AutoReviewModel):
    pubmed_queries: list[str] = Field(default_factory=list)
    semantic_scholar_queries: list[str] = Field(default_factory=list)
    openalex_queries: list[str] = Field(default_factory=list)
    perplexity_questions: list[str] = Field(default_factory=list)
    scope_document: str = ""


# ---------------------------------------------------------------------------
# Synthetic prompts (no external search needed)
# ---------------------------------------------------------------------------


def _query_expansion_prompt(topic: str) -> tuple[str, str]:
    system = (
        "You are an expert research librarian generating search queries for a literature review."
    )
    prompt = (
        f"Generate comprehensive search queries for a scientific literature review on: "
        f"'{topic}'\n\n"
        f"Produce:\n"
        f"- 3-5 PubMed Boolean/MeSH queries\n"
        f"- 3-5 Semantic Scholar semantic queries\n"
        f"- 3-5 OpenAlex general academic queries\n"
        f"- 2-3 Perplexity natural-language discovery questions\n"
        f"- A scope document (1 paragraph) defining review boundaries, "
        f"expected sub-topics, and exclusions\n"
    )
    return system, prompt


SYNTHETIC_ABSTRACT = """
Background: The gut-brain axis has emerged as a critical pathway in neurodegenerative diseases.
Recent studies demonstrate that alterations in gut microbiome composition, particularly reductions
in Lactobacillus and Bifidobacterium species, correlate with increased neuroinflammation markers
in patients with Alzheimer's disease (AD) and Parkinson's disease (PD).

Methods: We conducted a longitudinal cohort study (n=342) comparing gut microbiome profiles via
16S rRNA sequencing between AD patients (n=114), PD patients (n=108), and age-matched controls
(n=120) over 24 months. Neuroinflammation was assessed via CSF IL-6, TNF-alpha, and serum
neurofilament light chain (NfL).

Results: AD patients showed 40% reduction in Firmicutes/Bacteroidetes ratio compared to controls
(p<0.001). PD patients exhibited distinct dysbiosis patterns with elevated Enterobacteriaceae
(2.3-fold, p<0.01). Both groups showed elevated CSF IL-6 (AD: 8.2 vs 3.1 pg/mL; PD: 6.7 vs
3.1 pg/mL; p<0.001). Fecal microbiota transplantation in a subset (n=28) reduced NfL levels by
22% at 6 months.

Conclusions: Gut dysbiosis is disease-specific in AD vs PD, suggesting distinct microbiome-mediated
neuroinflammatory pathways. FMT shows preliminary therapeutic promise but requires larger RCTs.
"""


def _extraction_prompt() -> tuple[str, str]:
    system = (
        "You are an expert scientific data extractor. Extract structured information "
        "from research paper abstracts with precision."
    )
    prompt = (
        f"Extract structured findings from this paper abstract.\n\n"
        f"Paper ID: synth-001\n"
        f"Title: Gut Microbiome Dysbiosis in Alzheimer's and Parkinson's Disease: "
        f"A Longitudinal Cohort Study\n\n"
        f"Abstract:\n{SYNTHETIC_ABSTRACT}\n\n"
        f"Extract: key findings with evidence strength, methods summary, limitations, "
        f"methodology details (approach, datasets, metrics). "
        f"Use paper_id 'synth-001' for all findings."
    )
    return system, prompt


def _writing_prompt(topic: str) -> tuple[str, str]:
    system = (
        "You are an expert scientific writer producing publication-quality review paper sections."
    )
    prompt = (
        f"Write an 'Introduction' section (400-600 words) for a review paper on: '{topic}'\n\n"
        f"Context from extracted evidence:\n"
        f"- Gut dysbiosis patterns differ between Alzheimer's and Parkinson's disease\n"
        f"- Firmicutes/Bacteroidetes ratio reduced 40% in AD patients\n"
        f"- Enterobacteriaceae elevated 2.3-fold in PD patients\n"
        f"- Neuroinflammation markers (IL-6, TNF-alpha) elevated in both conditions\n"
        f"- Fecal microbiota transplantation shows preliminary therapeutic promise\n"
        f"- Gut-brain axis is a bidirectional communication pathway\n\n"
        f"Requirements:\n"
        f"- Synthesize across findings, don't summarize one-by-one\n"
        f"- Establish the significance of the topic\n"
        f"- Identify key themes and knowledge gaps\n"
        f"- Use academic prose with logical flow\n"
    )
    return system, prompt


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


class StageResult(AutoReviewModel):
    """Result of a single benchmark stage."""

    stage: str
    model: str
    success: bool
    error: str | None = None
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    output_preview: str = ""


async def _run_query_expansion(llm: LLMProvider, model_name: str, topic: str) -> StageResult:
    system, prompt = _query_expansion_prompt(topic)
    t0 = time.monotonic()
    try:
        response = await llm.generate_structured(
            prompt=prompt,
            response_model=QueryExpansionResult,
            system=system,
        )
        elapsed = time.monotonic() - t0
        parsed: QueryExpansionResult = response.parsed  # type: ignore[assignment]
        total_queries = (
            len(parsed.pubmed_queries)
            + len(parsed.semantic_scholar_queries)
            + len(parsed.openalex_queries)
            + len(parsed.perplexity_questions)
        )
        return StageResult(
            stage="query_expansion",
            model=model_name,
            success=True,
            elapsed_seconds=round(elapsed, 2),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            output_preview=f"{total_queries} queries, scope: {len(parsed.scope_document)} chars",
        )
    except Exception as e:
        return StageResult(
            stage="query_expansion",
            model=model_name,
            success=False,
            error=str(e),
            elapsed_seconds=round(time.monotonic() - t0, 2),
        )


async def _run_extraction(llm: LLMProvider, model_name: str) -> StageResult:
    system, prompt = _extraction_prompt()
    t0 = time.monotonic()
    try:
        response = await llm.generate_structured(
            prompt=prompt,
            response_model=PaperExtraction,
            system=system,
        )
        elapsed = time.monotonic() - t0
        parsed: PaperExtraction = response.parsed  # type: ignore[assignment]
        return StageResult(
            stage="paper_extraction",
            model=model_name,
            success=True,
            elapsed_seconds=round(elapsed, 2),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            output_preview=(
                f"{len(parsed.key_findings)} findings, methods: {len(parsed.methods_summary)} chars"
            ),
        )
    except Exception as e:
        return StageResult(
            stage="paper_extraction",
            model=model_name,
            success=False,
            error=str(e),
            elapsed_seconds=round(time.monotonic() - t0, 2),
        )


async def _run_writing(llm: LLMProvider, model_name: str, topic: str) -> StageResult:
    system, prompt = _writing_prompt(topic)
    t0 = time.monotonic()
    try:
        response = await llm.generate(
            prompt=prompt,
            system=system,
            temperature=0.3,
        )
        elapsed = time.monotonic() - t0
        word_count = len(response.content.split())
        return StageResult(
            stage="section_writing",
            model=model_name,
            success=True,
            elapsed_seconds=round(elapsed, 2),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            output_preview=f"{word_count} words",
        )
    except Exception as e:
        return StageResult(
            stage="section_writing",
            model=model_name,
            success=False,
            error=str(e),
            elapsed_seconds=round(time.monotonic() - t0, 2),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _print_table(results: list[StageResult]) -> None:
    """Print a comparison table to stdout."""
    import typer

    header = (
        f"{'Model':<25} {'Stage':<20} {'Status':<8} "
        f"{'Time (s)':<10} {'In Tok':<10} {'Out Tok':<10} {'Preview'}"
    )
    typer.echo("\n" + "=" * len(header))
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in results:
        status = "OK" if r.success else "FAIL"
        typer.echo(
            f"{r.model:<25} {r.stage:<20} {status:<8} {r.elapsed_seconds:<10.1f} "
            f"{r.input_tokens:<10} {r.output_tokens:<10} {r.output_preview}"
        )
    typer.echo("=" * len(header) + "\n")


async def run_benchmark(
    topic: str,
    model_names: list[str],
    output_dir: str = "output/benchmarks",
) -> list[StageResult]:
    """Run benchmarks for all models and produce a comparison report."""
    import typer

    all_results: list[StageResult] = []

    for model_name in model_names:
        typer.echo(f"\n--- Benchmarking: {model_name} ---")

        config = LLMConfig(model=model_name)
        try:
            llm = create_llm_provider(config)
        except ValueError as e:
            typer.echo(f"  Skipping {model_name}: {e}", err=True)
            for stage in ("query_expansion", "paper_extraction", "section_writing"):
                all_results.append(
                    StageResult(
                        stage=stage,
                        model=model_name,
                        success=False,
                        error=str(e),
                    )
                )
            continue

        typer.echo("  Stage 1/3: Query expansion...")
        all_results.append(await _run_query_expansion(llm, model_name, topic))

        typer.echo("  Stage 2/3: Paper extraction...")
        all_results.append(await _run_extraction(llm, model_name))

        typer.echo("  Stage 3/3: Section writing...")
        all_results.append(await _run_writing(llm, model_name, topic))

    _print_table(all_results)

    # Save JSON report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "topic": topic,
        "models": model_names,
        "results": [r.model_dump() for r in all_results],
    }
    report_file = out_path / f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(report, indent=2))
    typer.echo(f"Report saved to: {report_file}")

    return all_results
