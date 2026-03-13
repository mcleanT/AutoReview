"""Resolve parsed reference lines to CandidatePaper records via academic APIs."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel
from rapidfuzz import fuzz

from autoreview.models.paper import CandidatePaper

logger = structlog.get_logger()


class ResolutionConfidence(StrEnum):
    HIGH = "high"  # DOI exact match
    MEDIUM = "medium"  # Title fuzzy match >= 85%
    LOW = "low"  # Title fuzzy match 70-85%


class ResolvedReference(BaseModel):
    """A reference line that was successfully resolved to a paper."""

    raw_line: str
    paper: CandidatePaper
    confidence: ResolutionConfidence
    match_score: float = 1.0  # 1.0 for DOI, fuzzy ratio for title


async def resolve_reference(
    parsed: dict[str, str | None],
    search_sources: list[Any],
) -> ResolvedReference | None:
    """Resolve a single parsed reference to a CandidatePaper.

    Strategy:
    1. If DOI available, look up via get_paper_details("DOI:xxx")
    2. If no DOI or DOI lookup fails, search by title and fuzzy-match
    3. Return None if unresolvable

    Args:
        parsed: Dict with 'raw', 'doi', 'title' keys from reference_parser.
        search_sources: List of search source instances (S2, OpenAlex, etc.)

    Returns:
        ResolvedReference or None if unresolvable.
    """
    raw = parsed["raw"] or ""
    doi = parsed.get("doi")
    title = parsed.get("title") or ""

    # Strategy 1: DOI lookup
    if doi:
        for source in search_sources:
            try:
                paper = await source.get_paper_details(f"DOI:{doi}")
                if paper is not None:
                    logger.info(
                        "resolver.doi_match",
                        doi=doi,
                        source=getattr(source, "source_name", "unknown"),
                    )
                    return ResolvedReference(
                        raw_line=raw,
                        paper=paper,
                        confidence=ResolutionConfidence.HIGH,
                        match_score=1.0,
                    )
            except Exception as e:
                logger.warning("resolver.doi_lookup_failed", doi=doi, error=str(e))

    # Strategy 2: Title search + fuzzy match
    if title and len(title) > 10:
        for source in search_sources:
            try:
                source_name = getattr(source, "source_name", "unknown")
                results = await source.search([title], 5)
                if not results:
                    continue

                best_score = 0.0
                best_paper = None
                for candidate in results:
                    score = fuzz.ratio(title.lower(), candidate.title.lower())
                    if score > best_score:
                        best_score = score
                        best_paper = candidate

                if best_paper and best_score >= 70:
                    if best_score >= 85:
                        confidence = ResolutionConfidence.MEDIUM
                    else:
                        confidence = ResolutionConfidence.LOW
                    logger.info(
                        "resolver.title_match",
                        title=title[:60],
                        score=best_score,
                        confidence=confidence.value,
                        source=source_name,
                    )
                    return ResolvedReference(
                        raw_line=raw,
                        paper=best_paper,
                        confidence=confidence,
                        match_score=best_score / 100,
                    )
            except Exception as e:
                logger.warning("resolver.title_search_failed", title=title[:60], error=str(e))

    logger.warning("resolver.unresolvable", raw=raw[:80])
    return None


class ResolutionCache:
    """Cache resolution results keyed by reference line hash."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: dict[str, dict] = {}
        if cache_path.exists():
            self._data = json.loads(cache_path.read_text())

    def _hash(self, line: str) -> str:
        return hashlib.sha256(line.encode()).hexdigest()[:16]

    def get(self, raw_line: str) -> ResolvedReference | None:
        key = self._hash(raw_line)
        entry = self._data.get(key)
        if entry is None:
            return None
        return ResolvedReference.model_validate(entry)

    def put(self, resolved: ResolvedReference) -> None:
        key = self._hash(resolved.raw_line)
        self._data[key] = resolved.model_dump(mode="json")

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._data, indent=2))
