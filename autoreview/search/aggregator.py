from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

import structlog

from autoreview.models.paper import CandidatePaper

logger = structlog.get_logger()


def _normalize_title(title: str) -> str:
    """Normalize a paper title for fuzzy matching."""
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _merge_papers(primary: CandidatePaper, secondary: CandidatePaper) -> CandidatePaper:
    """Merge metadata from two records of the same paper. Primary wins."""
    data = primary.model_dump()
    sec = secondary.model_dump()

    for field in ["abstract", "year", "journal", "doi", "citation_count"]:
        if data.get(field) is None and sec.get(field) is not None:
            data[field] = sec[field]

    data["external_ids"] = {**sec.get("external_ids", {}), **data.get("external_ids", {})}

    if not data.get("authors") and sec.get("authors"):
        data["authors"] = sec["authors"]

    return CandidatePaper.model_validate(data)


def _parse_date_range(date_range: str | None) -> tuple[int | None, int | None]:
    """Parse 'YYYY-YYYY', '-YYYY', 'YYYY-', or None into (year_from, year_to).

    Range is inclusive on both bounds. Returns (None, None) for empty/None input.
    """
    if not date_range:
        return (None, None)
    parts = date_range.strip().split("-", 1)
    year_from: int | None = int(parts[0]) if parts[0] else None
    year_to: int | None = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return (year_from, year_to)


def _filter_by_year(
    papers: list[CandidatePaper],
    year_from: int | None,
    year_to: int | None,
) -> list[CandidatePaper]:
    """Drop papers outside the year range. Always drop year=None with logged warning.

    When both year_from and year_to are None (no date_range set), returns all
    papers unfiltered.
    """
    if year_from is None and year_to is None:
        return papers

    filtered: list[CandidatePaper] = []
    for paper in papers:
        if paper.year is None:
            logger.warning(
                "year_filter.dropped_null_year",
                title=paper.title[:80],
                source_database=paper.source_database,
                doi=paper.doi,
            )
            continue
        if year_from is not None and paper.year < year_from:
            continue
        if year_to is not None and paper.year > year_to:
            continue
        filtered.append(paper)

    dropped = len(papers) - len(filtered)
    if dropped:
        logger.info(
            "year_filter.applied",
            kept=len(filtered),
            dropped=dropped,
            year_from=year_from,
            year_to=year_to,
        )
    return filtered


class SearchAggregator:
    """Aggregates results from multiple search sources with deduplication."""

    def __init__(self, sources: list[Any] | None = None, date_range: str | None = None) -> None:
        self.sources: list[Any] = sources or []
        self._year_from, self._year_to = _parse_date_range(date_range)

    def add_source(self, source: Any) -> None:
        self.sources.append(source)

    async def search(
        self,
        queries_by_source: dict[str, list[str]],
        max_results_per_source: int = 500,
    ) -> list[CandidatePaper]:
        tasks = []
        source_names = []
        for source in self.sources:
            name = source.source_name
            qs = queries_by_source.get(name, [])
            if not qs:
                continue
            tasks.append(source.search(qs, max_results_per_source))
            source_names.append(name)

        if not tasks:
            logger.warning("aggregator.no_sources_with_queries")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[CandidatePaper] = []
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error("aggregator.source_failed", source=name, error=str(result))
                continue
            filtered = _filter_by_year(result, self._year_from, self._year_to)
            logger.info(
                "aggregator.source_results",
                source=name,
                raw=len(result),
                after_year_filter=len(filtered),
            )
            all_papers.extend(filtered)

        deduplicated = self._deduplicate(all_papers)
        logger.info(
            "aggregator.complete", total_raw=len(all_papers), deduplicated=len(deduplicated)
        )
        return deduplicated

    def _deduplicate(self, papers: list[CandidatePaper]) -> list[CandidatePaper]:
        doi_groups: dict[str, list[CandidatePaper]] = {}
        no_doi: list[CandidatePaper] = []

        for paper in papers:
            if paper.doi:
                key = paper.doi.lower().strip()
                doi_groups.setdefault(key, []).append(paper)
            else:
                no_doi.append(paper)

        merged_by_doi: list[CandidatePaper] = []
        doi_titles: set[str] = set()

        for group in doi_groups.values():
            primary = group[0]
            for secondary in group[1:]:
                primary = _merge_papers(primary, secondary)
            merged_by_doi.append(primary)
            doi_titles.add(_normalize_title(primary.title))

        title_groups: dict[str, list[CandidatePaper]] = {}
        for paper in no_doi:
            norm = _normalize_title(paper.title)
            if norm in doi_titles:
                continue
            title_groups.setdefault(norm, []).append(paper)

        merged_by_title: list[CandidatePaper] = []
        for group in title_groups.values():
            primary = group[0]
            for secondary in group[1:]:
                primary = _merge_papers(primary, secondary)
            merged_by_title.append(primary)

        return merged_by_doi + merged_by_title
