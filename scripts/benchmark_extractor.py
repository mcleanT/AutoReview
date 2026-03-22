#!/usr/bin/env python3
"""Benchmark the programmatic extractor against the LLM ground-truth corpus.

Loads papers from the extraction corpus, runs ProgrammaticExtractor on each,
scores the output against the LLM extraction, and prints a formatted report.
Results are persisted to JSON and a JSONL history file for tracking iterations.

Supports two scoring modes:
- **embedding** (default): Uses sentence-transformers all-MiniLM-L6-v2 for
  key_findings matching via cosine similarity + Hungarian assignment.
  More accurate when comparing programmatic extraction vs. LLM paraphrases.
- **word-overlap**: Falls back automatically if sentence-transformers is
  unavailable, or can be forced with ``--no-embeddings``.

Usage:
    python scripts/benchmark_extractor.py
    python scripts/benchmark_extractor.py --no-embeddings
    python scripts/benchmark_extractor.py --corpus-dir data/extraction_corpus --verbose
"""

from __future__ import annotations

import argparse
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
    load_embedding_model,
    score_extraction_pair,
    score_extraction_pair_with_embeddings,
)
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


def run_benchmark(
    corpus_dir: Path,
    verbose: bool = False,
    use_embeddings: bool = True,
) -> dict:
    """Run the full benchmark and return results dict.

    Args:
        corpus_dir: Path to the extraction corpus.
        verbose: Include per-paper details in output.
        use_embeddings: Try to load embedding model for key_findings scoring.
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

    config = ExtractionConfig()
    extractor = ProgrammaticExtractor(config)

    per_paper_results: list[dict] = []
    all_report_fields = SCORED_FIELDS + (EMBEDDING_EXTRA_FIELDS if embedding_model else [])
    field_scores_accum: dict[str, list[float]] = {f: [] for f in all_report_fields}
    composites: list[float] = []

    n_full_text = 0
    n_abstract_only = 0
    n_errors = 0

    t0 = time.perf_counter()

    for entry in corpus:
        paper_id = entry["paper_id"]
        try:
            candidate = build_candidate_paper(entry)
            screened = build_screened_paper(candidate)
            programmatic_result = extractor.extract(screened)
            llm_extraction = build_llm_extraction(entry)

            if embedding_model is not None:
                scores = score_extraction_pair_with_embeddings(
                    programmatic_result, llm_extraction, model=embedding_model
                )
            else:
                scores = score_extraction_pair(programmatic_result, llm_extraction)
            composite = compute_composite_score(scores)

            if candidate.full_text:
                n_full_text += 1
            else:
                n_abstract_only += 1

            for field in all_report_fields:
                field_scores_accum[field].append(scores.get(field, 0.0))
            composites.append(composite)

            paper_result = {
                "paper_id": paper_id,
                "title": candidate.title[:80],
                "has_full_text": bool(candidate.full_text),
                "composite": round(composite, 4),
                "field_scores": {k: round(v, 4) for k, v in scores.items()},
            }
            per_paper_results.append(paper_result)

        except Exception as exc:
            n_errors += 1
            logger.error(
                "benchmark.paper_failed",
                paper_id=paper_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            per_paper_results.append(
                {
                    "paper_id": paper_id,
                    "title": entry.get("metadata", {}).get("title", "unknown")[:80],
                    "has_full_text": bool(entry.get("full_text")),
                    "composite": 0.0,
                    "field_scores": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    elapsed = time.perf_counter() - t0

    # Compute averages
    avg_field_scores = {
        field: round(mean(vals), 4) if vals else 0.0 for field, vals in field_scores_accum.items()
    }
    avg_composite = round(mean(composites), 4) if composites else 0.0

    # Sort papers by composite for worst-10
    scored_papers = [p for p in per_paper_results if "error" not in p]
    scored_papers.sort(key=lambda p: p["composite"])

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scoring_mode": scoring_mode,
        "n_papers": len(corpus),
        "n_full_text": n_full_text,
        "n_abstract_only": n_abstract_only,
        "n_errors": n_errors,
        "elapsed_seconds": round(elapsed, 2),
        "overall_composite": avg_composite,
        "avg_field_scores": avg_field_scores,
        "worst_10": scored_papers[:10],
    }
    if verbose:
        results["per_paper"] = per_paper_results

    return results


def print_report(results: dict) -> None:
    """Print a formatted benchmark report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("PROGRAMMATIC EXTRACTOR BENCHMARK")
    print(sep)
    scoring_mode = results.get("scoring_mode", "word-overlap")
    print(f"Scoring mode: {scoring_mode}")
    n_errors = results["n_errors"]
    error_part = f", {n_errors} errors" if n_errors else ""
    print(
        f"Papers: {results['n_papers']} "
        f"({results['n_full_text']} full-text, "
        f"{results['n_abstract_only']} abstract-only"
        f"{error_part})"
    )
    print(f"Extraction time: {results['elapsed_seconds']:.1f}s")
    print(f"\nOverall Composite Score: {results['overall_composite']:.4f}")

    print("\nPer-field scores:")
    report_fields = SCORED_FIELDS[:]
    if "embedding" in scoring_mode:
        report_fields += EMBEDDING_EXTRA_FIELDS
    for field in report_fields:
        score = results["avg_field_scores"].get(field, 0.0)
        print(f"  {field:30s}: {score:.4f}")

    print("\nWorst 10 papers:")
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
        "n_papers": results["n_papers"],
        "overall_composite": results["overall_composite"],
        "avg_field_scores": results["avg_field_scores"],
        "elapsed_seconds": results["elapsed_seconds"],
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    logger.info("history_appended", path=str(history_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the programmatic extractor against the LLM ground-truth corpus."
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
    args = parser.parse_args()

    results = run_benchmark(
        args.corpus_dir,
        verbose=args.verbose,
        use_embeddings=not args.no_embeddings,
    )
    print_report(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
