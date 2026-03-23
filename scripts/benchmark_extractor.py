#!/usr/bin/env python3
"""Benchmark extraction strategies against the LLM ground-truth corpus.

Loads papers from the extraction corpus, runs the selected extraction strategy
on each, scores the output against the LLM extraction using dual-layer scoring
(similarity + factual), and prints a formatted report.  Results are persisted
to JSON and a JSONL history file for tracking iterations.

Supports two scoring modes for the similarity layer:
- **embedding** (default): Uses sentence-transformers all-MiniLM-L6-v2 for
  key_findings matching via cosine similarity + Hungarian assignment.
  More accurate when comparing programmatic extraction vs. LLM paraphrases.
- **word-overlap**: Falls back automatically if sentence-transformers is
  unavailable, or can be forced with ``--no-embeddings``.

Extraction strategies:
- **programmatic** (default): Zero-token deterministic extractor.
- **hybrid-haiku**: Programmatic draft + Haiku LLM refinement.
- **hybrid-sonnet**: Programmatic draft + Sonnet LLM refinement.
- **direct-haiku**: Full LLM extraction with Haiku (no programmatic draft).

Usage:
    python scripts/benchmark_extractor.py
    python scripts/benchmark_extractor.py --strategy hybrid-haiku
    python scripts/benchmark_extractor.py --strategy programmatic --alpha 0.5
    python scripts/benchmark_extractor.py --no-embeddings
    python scripts/benchmark_extractor.py --corpus-dir data/extraction_corpus --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import structlog

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.extraction.scoring import (
    compute_composite_score,
    compute_dual_composite,
    load_embedding_model,
    score_extraction_pair,
    score_extraction_pair_with_embeddings,
)
from autoreview.extraction.scoring_factual import score_extraction_pair_factual
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()

# Fields in the order they appear in the report
SCORED_FIELDS = [
    "key_findings",
    "evidence_strength",
    "quantitative_result",
    "methods_summary",
    "limitations",
    "study_design",
    "quality_score",
    "sample_size",
]

# Extra fields reported when embedding scoring is active
EMBEDDING_EXTRA_FIELDS = [
    "key_findings_precision",
    "key_findings_recall",
]

# Factual fields reported in the factual scoring section
FACTUAL_FIELDS = [
    "key_findings_factual",
    "quantitative_result_factual",
    "methods_summary_factual",
    "limitations_factual",
    "evidence_strength",
    "study_design",
    "quality_score",
    "sample_size",
]


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load all paper entries from the corpus directory via manifest.json."""
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found", path=str(manifest_path))
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    papers = []
    for entry in manifest["papers"]:
        paper_id = entry["paper_id"]
        paper_path = corpus_dir / f"{paper_id}.json"
        if not paper_path.exists():
            logger.warning("corpus_file_missing", paper_id=paper_id)
            continue
        with open(paper_path) as f:
            papers.append(json.load(f))

    return papers


def build_candidate_paper(corpus_entry: dict) -> CandidatePaper:
    """Build a CandidatePaper from a corpus entry."""
    meta = corpus_entry["metadata"]
    paper = CandidatePaper(
        title=meta["title"],
        authors=meta.get("authors", []),
        year=meta.get("year"),
        journal=meta.get("journal"),
        doi=meta.get("doi"),
        abstract=corpus_entry.get("abstract"),
        source_database=meta.get("source_database", "unknown"),
        citation_count=meta.get("citation_count"),
        full_text=corpus_entry.get("full_text"),
        full_text_source=corpus_entry.get("full_text_source"),
    )
    # Override auto-generated id to match the corpus paper_id
    paper.id = corpus_entry["paper_id"]
    return paper


def build_screened_paper(paper: CandidatePaper) -> ScreenedPaper:
    """Wrap a CandidatePaper into a ScreenedPaper for extraction."""
    return ScreenedPaper(
        paper=paper,
        relevance_score=4,
        rationale="benchmark corpus entry",
        include=True,
    )


def build_llm_extraction(corpus_entry: dict) -> PaperExtraction:
    """Build a PaperExtraction from the LLM ground-truth dict."""
    return PaperExtraction.model_validate(corpus_entry["llm_extraction"])


async def run_benchmark(
    corpus_dir: Path,
    verbose: bool = False,
    use_embeddings: bool = True,
    strategy: str = "programmatic",
    alpha: float = 0.5,
) -> dict:
    """Run the full benchmark and return results dict.

    Args:
        corpus_dir: Path to the extraction corpus.
        verbose: Include per-paper details in output.
        use_embeddings: Try to load embedding model for key_findings scoring.
        strategy: Extraction strategy to use (programmatic, hybrid-haiku,
            hybrid-sonnet, direct-haiku).
        alpha: Blend weight for dual composite: 1.0=similarity only,
            0.0=factual only.
    """
    corpus = load_corpus(corpus_dir)
    if not corpus:
        logger.error("no_papers_loaded")
        sys.exit(1)

    # --- Scoring mode selection ---
    embedding_model = None
    scoring_mode = "word-overlap"
    if use_embeddings:
        embedding_model = load_embedding_model()
        if embedding_model is not None:
            scoring_mode = "embedding (all-MiniLM-L6-v2)"
    print(f"Scoring mode: {scoring_mode}")
    print(f"Strategy: {strategy}")

    # --- Extractor factory ---
    config = ExtractionConfig()
    is_async = False
    if strategy == "programmatic":
        extractor = ProgrammaticExtractor(config)
    elif strategy in ("hybrid-haiku", "hybrid-sonnet"):
        from autoreview.extraction.hybrid import HybridExtractor
        from autoreview.llm.claude_code import ClaudeCodeProvider

        model = "haiku" if "haiku" in strategy else "sonnet"
        llm = ClaudeCodeProvider(model=model)
        extractor = HybridExtractor(ProgrammaticExtractor(config), llm)
        is_async = True
    elif strategy == "direct-haiku":
        from autoreview.extraction.extractor import PaperExtractor
        from autoreview.llm.claude_code import ClaudeCodeProvider

        llm = ClaudeCodeProvider(model="haiku")
        extractor = PaperExtractor(llm)
        is_async = True
    else:
        logger.error("unknown_strategy", strategy=strategy)
        sys.exit(1)

    per_paper_results: list[dict] = []
    all_report_fields = SCORED_FIELDS + (EMBEDDING_EXTRA_FIELDS if embedding_model else [])
    field_scores_accum: dict[str, list[float]] = {f: [] for f in all_report_fields}
    composites: list[float] = []
    factual_field_scores_accum: dict[str, list[float]] = {f: [] for f in FACTUAL_FIELDS}
    factual_composites: list[float] = []

    n_full_text = 0
    n_abstract_only = 0
    n_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    t0 = time.perf_counter()

    # --- Process one paper (used by both sequential and concurrent paths) ---
    async def _process_paper(entry: dict) -> dict:
        """Extract, score, and return result dict for a single paper."""
        paper_id = entry["paper_id"]
        try:
            candidate = build_candidate_paper(entry)
            screened = build_screened_paper(candidate)
            llm_extraction = build_llm_extraction(entry)

            # --- Run extraction ---
            if is_async:
                if strategy == "direct-haiku":
                    extraction_result = await extractor.extract_one(candidate, relevance_score=4)
                else:
                    extraction_result = await extractor.extract(screened)
            else:
                extraction_result = extractor.extract(screened)

            # --- Similarity scoring ---
            if embedding_model is not None:
                scores = score_extraction_pair_with_embeddings(
                    extraction_result, llm_extraction, model=embedding_model
                )
            else:
                scores = score_extraction_pair(extraction_result, llm_extraction)
            composite = compute_composite_score(scores)

            # --- Factual scoring layer ---
            factual_scores = score_extraction_pair_factual(
                extraction_result, llm_extraction, similarity_scores=scores
            )
            factual_composite = compute_composite_score(factual_scores)

            return {
                "paper_id": paper_id,
                "title": candidate.title[:80],
                "has_full_text": bool(candidate.full_text),
                "composite": round(composite, 4),
                "factual_composite": round(factual_composite, 4),
                "field_scores": {k: round(v, 4) for k, v in scores.items()},
                "factual_field_scores": {k: round(v, 4) for k, v in factual_scores.items()},
            }

        except Exception as exc:
            logger.error(
                "benchmark.paper_failed",
                paper_id=paper_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {
                "paper_id": paper_id,
                "title": entry.get("metadata", {}).get("title", "unknown")[:80],
                "has_full_text": bool(entry.get("full_text")),
                "composite": 0.0,
                "factual_composite": 0.0,
                "field_scores": {},
                "factual_field_scores": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

    # --- Run papers concurrently for async strategies, sequentially otherwise ---
    # claude -p is I/O-bound (waiting for API), so 20 concurrent is fine
    max_concurrent = 20 if is_async else 1
    sem = asyncio.Semaphore(max_concurrent)

    async def _process_with_sem(entry: dict) -> dict:
        async with sem:
            return await _process_paper(entry)

    if is_async:
        print(f"Processing {len(corpus)} papers concurrently (max {max_concurrent})...")
        per_paper_results = await asyncio.gather(*[_process_with_sem(entry) for entry in corpus])
        per_paper_results = list(per_paper_results)
    else:
        per_paper_results = []
        for entry in corpus:
            result = await _process_paper(entry)
            per_paper_results.append(result)

    # --- Aggregate results ---
    for paper_result in per_paper_results:
        if "error" in paper_result:
            n_errors += 1
            continue

        if paper_result["has_full_text"]:
            n_full_text += 1
        else:
            n_abstract_only += 1

        scores = paper_result["field_scores"]
        factual_scores = paper_result.get("factual_field_scores", {})

        for field in all_report_fields:
            field_scores_accum[field].append(scores.get(field, 0.0))
        composites.append(paper_result["composite"])

        for field in FACTUAL_FIELDS:
            factual_field_scores_accum[field].append(factual_scores.get(field, 0.0))
        factual_composites.append(paper_result.get("factual_composite", 0.0))

    elapsed = time.perf_counter() - t0

    # Compute averages
    avg_field_scores = {
        field: round(mean(vals), 4) if vals else 0.0 for field, vals in field_scores_accum.items()
    }
    avg_composite = round(mean(composites), 4) if composites else 0.0

    avg_factual_scores = {
        field: round(mean(vals), 4) if vals else 0.0
        for field, vals in factual_field_scores_accum.items()
    }
    factual_composite_avg = round(mean(factual_composites), 4) if factual_composites else 0.0

    dual = compute_dual_composite(avg_field_scores, avg_factual_scores, alpha=alpha)

    # Sort papers by composite for worst-10
    scored_papers = [p for p in per_paper_results if "error" not in p]
    scored_papers.sort(key=lambda p: p["composite"])

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scoring_mode": scoring_mode,
        "strategy": strategy,
        "alpha": alpha,
        "n_papers": len(corpus),
        "n_full_text": n_full_text,
        "n_abstract_only": n_abstract_only,
        "n_errors": n_errors,
        "elapsed_seconds": round(elapsed, 2),
        "overall_composite": avg_composite,
        "factual_composite": factual_composite_avg,
        "combined_composite": round(dual["combined"], 4),
        "avg_field_scores": avg_field_scores,
        "avg_factual_field_scores": avg_factual_scores,
        "worst_10": scored_papers[:10],
    }
    if total_input_tokens or total_output_tokens:
        results["total_input_tokens"] = total_input_tokens
        results["total_output_tokens"] = total_output_tokens
    if verbose:
        results["per_paper"] = per_paper_results

    return results


def print_report(results: dict) -> None:
    """Print a formatted benchmark report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("EXTRACTOR BENCHMARK")
    print(sep)
    scoring_mode = results.get("scoring_mode", "word-overlap")
    strategy = results.get("strategy", "programmatic")
    alpha = results.get("alpha", 0.5)
    print(f"Strategy:     {strategy}")
    print(f"Scoring mode: {scoring_mode}")
    print(f"Alpha (blend): {alpha}  (1.0=similarity only, 0.0=factual only)")
    n_errors = results["n_errors"]
    error_part = f", {n_errors} errors" if n_errors else ""
    print(
        f"Papers: {results['n_papers']} "
        f"({results['n_full_text']} full-text, "
        f"{results['n_abstract_only']} abstract-only"
        f"{error_part})"
    )
    print(f"Extraction time: {results['elapsed_seconds']:.1f}s")
    if "total_input_tokens" in results:
        print(
            f"LLM tokens: {results['total_input_tokens']} in / {results['total_output_tokens']} out"
        )

    print("\n--- Similarity Layer ---")
    print(f"Overall Composite Score:  {results['overall_composite']:.4f}")
    report_fields = SCORED_FIELDS[:]
    if "embedding" in scoring_mode:
        report_fields += EMBEDDING_EXTRA_FIELDS
    print("\nPer-field similarity scores:")
    for field in report_fields:
        score = results["avg_field_scores"].get(field, 0.0)
        print(f"  {field:30s}: {score:.4f}")

    print("\n--- Factual Layer ---")
    print(f"Factual Composite Score:  {results['factual_composite']:.4f}")
    print("\nPer-field factual scores:")
    for field in FACTUAL_FIELDS:
        score = results["avg_factual_field_scores"].get(field, 0.0)
        print(f"  {field:30s}: {score:.4f}")

    print(f"\n--- Combined (alpha={alpha}) ---")
    print(f"Combined Composite Score: {results['combined_composite']:.4f}")

    print("\nWorst 10 papers (by similarity composite):")
    for p in results["worst_10"]:
        paper_id = p["paper_id"]
        composite = p["composite"]
        title = p.get("title", "")[:40]
        print(f"  {paper_id:40s} composite={composite:.3f}  {title}")

    print(sep)
    print()


def save_results(results: dict, output_dir: Path) -> None:
    """Save benchmark results to JSON and append to history JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("results_saved", path=str(results_path))

    # History line
    history_path = output_dir / "benchmark_history.jsonl"
    summary = {
        "timestamp": results["timestamp"],
        "scoring_mode": results.get("scoring_mode", "word-overlap"),
        "strategy": results.get("strategy", "programmatic"),
        "alpha": results.get("alpha", 0.5),
        "n_papers": results["n_papers"],
        "overall_composite": results["overall_composite"],
        "factual_composite": results.get("factual_composite", 0.0),
        "combined_composite": results.get("combined_composite", 0.0),
        "avg_field_scores": results["avg_field_scores"],
        "elapsed_seconds": results["elapsed_seconds"],
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    logger.info("history_appended", path=str(history_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark extraction strategies against the LLM ground-truth corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/extraction_corpus"),
        help="Path to the extraction corpus directory (default: data/extraction_corpus)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extractor_benchmark"),
        help="Path to save results (default: results/extractor_benchmark)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-paper scores in JSON output",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Force word-overlap scoring even if sentence-transformers is available",
    )
    parser.add_argument(
        "--strategy",
        choices=["programmatic", "hybrid-haiku", "hybrid-sonnet", "direct-haiku"],
        default="programmatic",
        help="Extraction strategy to benchmark (default: programmatic)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend weight for dual composite: 1.0=similarity only, 0.0=factual only (default: 0.5)",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_benchmark(
            args.corpus_dir,
            verbose=args.verbose,
            use_embeddings=not args.no_embeddings,
            strategy=args.strategy,
            alpha=args.alpha,
        )
    )
    print_report(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
