"""Reconstruct a KnowledgeBase snapshot from legacy ARISE per-stage snapshot files.

Usage:
    python scripts/reconstruct_kb.py output/arise/arise_rag_v2
    python scripts/reconstruct_kb.py output/arise/arise_rag_v2 \
        --output-dir output/arise/arise_rag_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_stage(snapshots_dir: Path, filename: str) -> dict:
    path = snapshots_dir / filename
    with path.open() as f:
        return json.load(f)


def build_candidate_paper(
    p: dict, full_text: str | None = None, full_text_source: str | None = None
) -> dict:
    """Map a legacy paper dict to a CandidatePaper-compatible dict."""
    authors = p.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    return {
        "title": p["title"],
        "authors": authors,
        "year": p.get("year"),
        "doi": p.get("doi"),
        "abstract": p.get("abstract"),
        "source_database": p["source"],
        "external_ids": p.get("external_ids", {}),
        "full_text": full_text,
        "full_text_source": full_text_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a KnowledgeBase snapshot from legacy ARISE stage files."
    )
    parser.add_argument(
        "run_dir", help="Path to the ARISE run directory (e.g. output/arise/arise_rag_v2)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for the reconstructed snapshot (default: run_dir + '_v3')",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    snapshots_dir = run_dir / "snapshots"

    if not snapshots_dir.exists():
        print(f"ERROR: snapshots directory not found: {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir.parent / (run_dir.name + "_v3")

    # --- Load stage files ---
    query_expansion = load_stage(snapshots_dir, "01_query_expansion.json")
    screening = load_stage(snapshots_dir, "03_screening.json")
    full_text = load_stage(snapshots_dir, "04_full_text.json")
    gap_search = load_stage(snapshots_dir, "07_gap_search.json")

    # --- Build full-text lookup by DOI for attaching full text to screened papers ---
    # Normalise DOIs to lowercase for case-insensitive matching
    full_text_by_doi: dict[str, dict] = {}
    for p in full_text.get("papers", []):
        doi = p.get("doi")
        if doi:
            full_text_by_doi[doi.lower()] = p

    # --- candidate_papers: from full_text stage ---
    candidate_papers = []
    for p in full_text.get("papers", []):
        has_full_text = p.get("has_full_text", False)
        full_text_source = "legacy" if has_full_text else None
        candidate_papers.append(
            build_candidate_paper(
                p, full_text=p.get("full_text"), full_text_source=full_text_source
            )
        )

    # --- screened_papers: from screening stage ---
    screened_papers = []
    for sp in screening.get("papers", []):
        doi = sp.get("doi", "")
        doi_key = doi.lower() if doi else ""

        # Look up full text for this paper from the full_text stage
        ft_record = full_text_by_doi.get(doi_key)
        if ft_record is not None:
            has_full_text = ft_record.get("has_full_text", False)
            ft_text = ft_record.get("full_text")
            ft_source = "legacy" if has_full_text else None
            # Use full_text record for the inner paper so we get full_text attached
            inner_paper = build_candidate_paper(
                ft_record, full_text=ft_text, full_text_source=ft_source
            )
        else:
            # Fall back to screening record data (no full text available)
            inner_paper = build_candidate_paper(sp)

        screened_papers.append(
            {
                "paper": inner_paper,
                "relevance_score": sp["relevance_score"],
                "rationale": sp["score_rationale"],
                "include": True,
            }
        )

    # --- search_queries: combine pubmed + semantic from query_expansion ---
    search_queries: dict[str, list[str]] = {}
    pubmed_queries = query_expansion.get("pubmed_queries", [])
    semantic_queries = query_expansion.get("semantic_queries", [])
    if pubmed_queries:
        search_queries["pubmed"] = pubmed_queries
    if semantic_queries:
        search_queries["semantic"] = semantic_queries

    # --- Assemble KnowledgeBase ---
    # Import here so the script can be run from the repo root with the package on PYTHONPATH
    from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
    from autoreview.models.paper import CandidatePaper as CandidatePaperModel
    from autoreview.models.paper import ScreenedPaper as ScreenedPaperModel

    candidate_paper_objects = [CandidatePaperModel.model_validate(p) for p in candidate_papers]
    screened_paper_objects = [ScreenedPaperModel.model_validate(sp) for sp in screened_papers]

    kb = KnowledgeBase(
        topic=gap_search["topic"],
        domain=gap_search["domain"],
        search_queries=search_queries,
        candidate_papers=candidate_paper_objects,
        screened_papers=screened_paper_objects,
        current_phase=PipelinePhase.FULL_TEXT_RETRIEVAL,
        output_dir=str(output_dir),
    )

    kb.save_snapshot("reconstructed")

    saved_path = Path(kb.output_dir) / "snapshots" / "latest.json"
    print(f"Reconstructed KnowledgeBase snapshot saved to: {saved_path.resolve()}")
    print(f"  topic:             {kb.topic}")
    print(f"  domain:            {kb.domain}")
    print(f"  candidate_papers:  {len(kb.candidate_papers)}")
    print(f"  screened_papers:   {len(kb.screened_papers)}")
    n_queries = sum(len(v) for v in kb.search_queries.values())
    n_sources = len(kb.search_queries)
    print(f"  search_queries:    {n_queries} queries across {n_sources} sources")
    print(f"  current_phase:     {kb.current_phase}")


if __name__ == "__main__":
    main()
