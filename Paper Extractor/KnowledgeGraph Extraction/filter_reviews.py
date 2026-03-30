#!/usr/bin/env python3
"""Filter review papers from the gastruloid corpus using OpenAlex publication type API.

Uses DOI-based lookup against OpenAlex to get authoritative publication types.
Falls back to heuristic title+abstract+journal regex for papers without DOI or
not found in OpenAlex.

Usage:
    python filter_reviews.py                  # Dry run — report only
    python filter_reviews.py --apply          # Write filtered papers.json + archive reviews
    python filter_reviews.py --apply --backup # Backup original papers.json first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run"
PAPERS_PATH = OUTPUT_DIR / "papers.json"

# OpenAlex types that indicate review/non-primary content
_REVIEW_TYPES = {
    "review",
    "editorial",
    "letter",
    "erratum",
    "commentary",
    "retraction",
    "book-chapter",
}

# Heuristic fallback patterns
_REVIEW_TITLE_RE = re.compile(
    r"\breview\b|\bmeta.analysis\b|\bsystematic review\b|\bperspective\b"
    r"|\beditorial\b|\bcommentary\b|\bopinion\b|\bmini.review\b"
    r"|\boverview\b|\bstate of the art\b|\brecent advances\b"
    r"|\bdecision letter\b|\bauthor response\b|\beditor.s evaluation\b",
    re.IGNORECASE,
)
_REVIEW_ABSTRACT_RE = re.compile(
    r"\bin this review\b|\bwe review\b|\bhere.{0,5}we review\b|\bthis review\b"
    r"|\bwe summarize\b|\bwe discuss recent\b"
    r"|\bcomprehensive overview\b|\bwe survey\b"
    r"|\bwe provide an overview\b|\bthis perspective\b",
    re.IGNORECASE,
)
_REVIEW_JOURNAL_RE = re.compile(
    r"\bcurrent opinion\b|\btrends in\b|\bannual review\b|\breviews and reports\b",
    re.IGNORECASE,
)

# OpenAlex polite pool — include email for faster rate limits
_OPENALEX_EMAIL = "mst36@psu.edu"
_BATCH_SIZE = 50  # OpenAlex filter supports up to 50 DOIs per request


def _openalex_batch_lookup(dois: list[str]) -> dict[str, dict]:
    """Look up publication types for a batch of DOIs via OpenAlex."""
    results = {}
    doi_filter = "|".join(f"https://doi.org/{d}" for d in dois)
    url = (
        f"https://api.openalex.org/works"
        f"?filter=doi:{quote(doi_filter, safe='|:/')}"
        f"&select=doi,type,title"
        f"&per_page={len(dois)}"
        f"&mailto={_OPENALEX_EMAIL}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for work in data.get("results", []):
            doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
            if doi:
                results[doi] = {
                    "type": work.get("type", "unknown"),
                    "title": work.get("title", ""),
                }
    except Exception as e:
        print(f"  OpenAlex batch error: {e}", file=sys.stderr)

    return results


def _heuristic_is_review(paper: dict) -> tuple[bool, str]:
    """Fallback heuristic when API lookup unavailable."""
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    journal = paper.get("journal") or ""

    if _REVIEW_TITLE_RE.search(title):
        return True, "title_match"
    if _REVIEW_ABSTRACT_RE.search(abstract):
        return True, "abstract_match"
    if _REVIEW_JOURNAL_RE.search(journal):
        return True, "journal_match"
    return False, ""


def filter_corpus(apply: bool = False, backup: bool = False) -> None:
    papers = json.loads(PAPERS_PATH.read_text())
    print(f"Loaded {len(papers)} papers from {PAPERS_PATH.name}")

    # Phase 1: OpenAlex batch lookup for papers with DOIs
    doi_papers = {}
    no_doi = []
    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        if doi:
            doi_papers[doi] = p
        else:
            no_doi.append(p)

    print(f"\nPhase 1: OpenAlex lookup for {len(doi_papers)} papers with DOIs...")
    openalex_types: dict[str, dict] = {}
    doi_list = list(doi_papers.keys())

    for i in range(0, len(doi_list), _BATCH_SIZE):
        batch = doi_list[i : i + _BATCH_SIZE]
        results = _openalex_batch_lookup(batch)
        openalex_types.update(results)
        found = len(results)
        print(f"  Batch {i // _BATCH_SIZE + 1}: {found}/{len(batch)} found")
        if i + _BATCH_SIZE < len(doi_list):
            time.sleep(0.2)  # Polite rate limiting

    print(
        f"  OpenAlex coverage: {len(openalex_types)}/{len(doi_papers)} ({100 * len(openalex_types) / len(doi_papers):.0f}%)"
    )

    # Phase 2: Classify each paper
    primary = []
    reviews = []
    type_dist: dict[str, int] = {}

    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        title = p.get("title") or ""

        # Try OpenAlex first
        oa = openalex_types.get(doi)
        if oa:
            pub_type = oa["type"]
            type_dist[pub_type] = type_dist.get(pub_type, 0) + 1

            if pub_type in _REVIEW_TYPES:
                reviews.append({"paper": p, "reason": f"openalex:{pub_type}", "source": "openalex"})
                continue
            elif pub_type == "article":
                # Double-check with heuristic for articles in review journals
                is_rev, reason = _heuristic_is_review(p)
                if is_rev:
                    reviews.append(
                        {
                            "paper": p,
                            "reason": f"heuristic:{reason}+openalex:article",
                            "source": "hybrid",
                        }
                    )
                    continue
                primary.append(p)
                continue
            else:
                primary.append(p)
                continue

        # Fallback: heuristic
        is_rev, reason = _heuristic_is_review(p)
        if is_rev:
            reviews.append({"paper": p, "reason": f"heuristic:{reason}", "source": "heuristic"})
        else:
            primary.append(p)

    # Report
    print(f"\n{'=' * 70}")
    print("  Review Filter Results")
    print(f"{'=' * 70}")
    print(f"  Primary research:  {len(primary)} ({100 * len(primary) / len(papers):.1f}%)")
    print(f"  Reviews/filtered:  {len(reviews)} ({100 * len(reviews) / len(papers):.1f}%)")
    print(f"  No DOI (heuristic only): {len(no_doi)}")

    print("\n  OpenAlex type distribution:")
    for t, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        marker = " ** FILTERED" if t in _REVIEW_TYPES else ""
        print(f"    {t:<25} {count:>4}{marker}")

    api_reviews = sum(1 for r in reviews if r["source"] == "openalex")
    hybrid_reviews = sum(1 for r in reviews if r["source"] == "hybrid")
    heuristic_reviews = sum(1 for r in reviews if r["source"] == "heuristic")
    print("\n  Filter source breakdown:")
    print(f"    OpenAlex type alone:      {api_reviews}")
    print(f"    Hybrid (OA+heuristic):    {hybrid_reviews}")
    print(f"    Heuristic only:           {heuristic_reviews}")

    print("\n  Filtered papers:")
    for r in sorted(reviews, key=lambda x: x["reason"]):
        title = (r["paper"].get("title") or "")[:75]
        print(f"    [{r['reason']:<40s}] {title}")

    if apply:
        if backup:
            backup_path = OUTPUT_DIR / "papers_pre_filter.json"
            backup_path.write_text(PAPERS_PATH.read_text())
            print(f"\n  Backup saved: {backup_path}")

        PAPERS_PATH.write_text(json.dumps(primary, indent=2, ensure_ascii=False))
        print(f"\n  papers.json updated: {len(primary)} primary research papers")

        reviews_path = OUTPUT_DIR / "papers_reviews.json"
        review_records = [
            {"reason": r["reason"], "source": r["source"], **r["paper"]} for r in reviews
        ]
        reviews_path.write_text(json.dumps(review_records, indent=2, ensure_ascii=False))
        print(f"  Reviews archived: {reviews_path}")
    else:
        print("\n  DRY RUN — no files changed. Use --apply to filter.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter review papers using OpenAlex + heuristics")
    parser.add_argument("--apply", action="store_true", help="Write filtered corpus")
    parser.add_argument("--backup", action="store_true", help="Backup original papers.json")
    filter_corpus(**vars(parser.parse_args()))
