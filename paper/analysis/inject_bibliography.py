"""Bibliography injection: extract references from a PDF, resolve, build KnowledgeBase.

Usage:
    python paper/analysis/inject_bibliography.py \
        --pdf paper/references/review.pdf \
        --topic "Topic string" \
        --domain biomedical \
        --output paper/snapshots/injected.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import structlog

from autoreview.evaluation.pdf_extractor import extract_bibliography_lines, extract_text_from_pdf
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper, ScreenedPaper
from paper.analysis.reference_parser import parse_all_references
from paper.analysis.reference_resolver import (
    ResolutionCache,
    ResolvedReference,
    resolve_reference,
)

logger = structlog.get_logger()


def build_injected_kb(
    papers: list[CandidatePaper],
    topic: str,
    domain: str,
    output_dir: str,
) -> KnowledgeBase:
    """Build a pre-populated KnowledgeBase from resolved papers.

    All papers are added as candidates and auto-screened with include=True,
    relevance_score=5. Full text is NOT populated — the pipeline will fetch
    it when resumed from full_text_retrieval.
    """
    screened = [
        ScreenedPaper(
            paper=paper,
            relevance_score=5,
            rationale="Injected from reference bibliography",
            include=True,
        )
        for paper in papers
    ]

    kb = KnowledgeBase(
        topic=topic,
        domain=domain,
        output_dir=output_dir,
        candidate_papers=papers,
        screened_papers=screened,
        current_phase=PipelinePhase.SCREENING,
    )
    return kb


def build_resolution_report(
    total_extracted: int,
    resolved: list[ResolvedReference],
    failed_lines: list[str],
) -> dict:
    """Build a summary report of the resolution process."""
    confidence_counts = Counter(r.confidence.value for r in resolved)
    return {
        "total_extracted": total_extracted,
        "resolved_count": len(resolved),
        "failed_count": len(failed_lines),
        "by_confidence": {
            "high": confidence_counts.get("high", 0),
            "medium": confidence_counts.get("medium", 0),
            "low": confidence_counts.get("low", 0),
        },
        "failed_references": failed_lines[:20],
    }


async def run_injection(
    pdf_path: Path,
    topic: str,
    domain: str,
    output_path: Path,
    cache_path: Path | None = None,
) -> dict:
    """Full injection pipeline: extract -> parse -> resolve -> build KB -> save.

    Returns the resolution report dict.
    """
    # 1. Extract bibliography from PDF
    text = extract_text_from_pdf(pdf_path)
    ref_lines = extract_bibliography_lines(text)
    logger.info("inject.extracted", ref_count=len(ref_lines))

    if not ref_lines:
        logger.warning("inject.no_references_found", pdf=str(pdf_path))
        return build_resolution_report(0, [], [])

    # 2. Parse reference lines
    parsed = parse_all_references(ref_lines)

    # 3. Set up search sources for resolution
    from autoreview.search.openalex import OpenAlexSearch
    from autoreview.search.semantic_scholar import SemanticScholarSearch

    sources = [SemanticScholarSearch(), OpenAlexSearch()]

    # 4. Resolve with optional caching
    cache = ResolutionCache(cache_path) if cache_path else None
    resolved: list[ResolvedReference] = []
    failed_lines: list[str] = []

    for p in parsed:
        raw = p["raw"] or ""

        # Check cache first
        if cache:
            cached = cache.get(raw)
            if cached is not None:
                resolved.append(cached)
                continue

        result = await resolve_reference(p, sources)
        if result is not None:
            resolved.append(result)
            if cache:
                cache.put(result)
        else:
            failed_lines.append(raw)

    if cache:
        cache.save()

    # 5. Build KB
    papers = [r.paper for r in resolved]
    kb = build_injected_kb(
        papers=papers,
        topic=topic,
        domain=domain,
        output_dir=str(output_path.parent),
    )

    # 6. Save snapshot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = kb.model_dump_json(indent=2)
    output_path.write_text(json_data)
    logger.info("inject.saved", path=str(output_path), papers=len(papers))

    # 7. Build and save report
    report = build_resolution_report(len(ref_lines), resolved, failed_lines)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("inject.report", **{k: v for k, v in report.items() if k != "failed_references"})

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject bibliography from reference PDF into KnowledgeBase"
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to reference review PDF")
    parser.add_argument("--topic", required=True, help="Research topic string")
    parser.add_argument("--domain", default="general", help="Domain (biomedical, cs_ai, etc.)")
    parser.add_argument("--output", required=True, type=Path, help="Output snapshot JSON path")
    parser.add_argument("--cache", type=Path, default=None, help="Resolution cache JSON path")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.processors.NAME_TO_LEVEL["info"]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    report = asyncio.run(
        run_injection(
            pdf_path=args.pdf,
            topic=args.topic,
            domain=args.domain,
            output_path=args.output,
            cache_path=args.cache,
        )
    )

    print("\nResolution Report:")
    print(f"  Total references: {report['total_extracted']}")
    print(
        f"  Resolved: {report['resolved_count']} "
        f"(high: {report['by_confidence']['high']}, "
        f"medium: {report['by_confidence']['medium']}, "
        f"low: {report['by_confidence']['low']})"
    )
    print(f"  Failed: {report['failed_count']}")
    if report["failed_references"]:
        print("\n  Unresolved references:")
        for line in report["failed_references"]:
            print(f"    - {line[:100]}")


if __name__ == "__main__":
    main()
