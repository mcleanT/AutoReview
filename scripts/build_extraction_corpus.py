#!/usr/bin/env python3
"""Build a training corpus for the programmatic extractor.

Assembles 220 papers pairing source text + metadata with LLM ground-truth
extractions into individual JSON files under ``data/extraction_corpus/``.

Usage::

    python scripts/build_extraction_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "output" / "arise" / "arise_llm_eval_v2" / "snapshots" / "latest.json"
CACHE_ROOT = ROOT / "output" / "arise" / "arise_llm_eval_v2" / ".cache" / "full_text_cache"
OUTPUT_DIR = ROOT / "data" / "extraction_corpus"


# ---------------------------------------------------------------------------
# Cache key helpers (mirrored from autoreview.search.full_text_cache)
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    title = unicodedata.normalize("NFD", title)
    title = "".join(c for c in title if unicodedata.category(c) != "Mn")
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _cache_key(doi: str | None, title: str) -> str:
    """SHA-256 hex digest matching the pipeline's cache key derivation."""
    raw = doi.lower().strip() if doi else _normalize_title(title)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache_entry(doi: str | None, title: str) -> dict | None:
    """Load the full-text cache entry for a paper, or return None."""
    h = _cache_key(doi, title)
    cache_path = CACHE_ROOT / h[:2] / f"{h}.json"
    if not cache_path.exists():
        return None
    with cache_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load snapshot
    logger.info("loading_snapshot", path=str(SNAPSHOT))
    with SNAPSHOT.open() as f:
        snap = json.load(f)

    # Index screened papers by id
    screened_by_id: dict[str, dict] = {}
    for sp in snap["screened_papers"]:
        paper = sp["paper"]
        screened_by_id[paper["id"]] = paper

    extractions: dict[str, dict] = snap["extractions"]
    logger.info("snapshot_loaded", n_screened=len(screened_by_id), n_extractions=len(extractions))

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_papers: list[dict] = []
    n_with_full_text = 0
    n_abstract_only = 0
    n_missing_metadata = 0

    for paper_id, extraction in extractions.items():
        paper = screened_by_id.get(paper_id)
        if paper is None:
            logger.warning("paper_not_found_in_screened", paper_id=paper_id)
            n_missing_metadata += 1
            continue

        # Load full text from cache
        cache_entry = _load_cache_entry(paper.get("doi"), paper["title"])

        full_text = None
        full_text_source = None
        has_full_text = False
        if cache_entry is not None and cache_entry.get("text"):
            full_text = cache_entry["text"]
            full_text_source = cache_entry.get("source")
            has_full_text = True

        if has_full_text:
            n_with_full_text += 1
        else:
            n_abstract_only += 1

        # Build corpus entry
        entry = {
            "paper_id": paper_id,
            "metadata": {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "journal": paper.get("journal", ""),
                "doi": paper.get("doi", ""),
                "citation_count": paper.get("citation_count"),
                "source_database": paper.get("source_database", ""),
            },
            "abstract": paper.get("abstract", ""),
            "full_text": full_text,
            "full_text_source": full_text_source,
            "llm_extraction": extraction,
        }

        out_path = OUTPUT_DIR / f"{paper_id}.json"
        with out_path.open("w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        manifest_papers.append(
            {
                "paper_id": paper_id,
                "has_full_text": has_full_text,
                "title": paper.get("title", ""),
            }
        )

    # Write manifest
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_papers": len(manifest_papers),
        "n_with_full_text": n_with_full_text,
        "n_abstract_only": n_abstract_only,
        "papers": manifest_papers,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info(
        "corpus_built",
        n_papers=len(manifest_papers),
        n_with_full_text=n_with_full_text,
        n_abstract_only=n_abstract_only,
        n_missing_metadata=n_missing_metadata,
        output_dir=str(OUTPUT_DIR),
    )

    print("\n" + "=" * 60)
    print("EXTRACTION CORPUS BUILD SUMMARY")
    print("=" * 60)
    print(f"  Total papers:       {len(manifest_papers)}")
    print(f"  With full text:     {n_with_full_text}")
    print(f"  Abstract only:      {n_abstract_only}")
    print(f"  Missing metadata:   {n_missing_metadata}")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print(f"  Manifest:           {manifest_path}")
    print("=" * 60)

    if n_missing_metadata > 0:
        logger.warning("some_papers_missing_metadata", count=n_missing_metadata)
        sys.exit(1)


if __name__ == "__main__":
    main()
