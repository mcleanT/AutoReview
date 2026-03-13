"""Parse raw reference lines into structured data (DOI + title)."""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+")

_JOURNAL_SIGNALS = re.compile(
    r"\b(?:Nature|Science|Cell|JAMA|Lancet|BMJ|PLoS|PNAS|Neuron|"
    r"J\.\s|Journal|Proceedings|Annals|Archives|Reviews|Frontiers)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"(?:^|\D)((?:19|20)\d{2})(?:\D|$)")


def extract_doi(line: str) -> str | None:
    """Extract a DOI from a reference line.

    Handles formats: doi:10.xxx, https://doi.org/10.xxx, bare 10.xxx/yyy.
    Returns the DOI without URL prefix, or None if not found.
    """
    cleaned = re.sub(r"https?://(?:dx\.)?doi\.org/", "", line)
    m = _DOI_RE.search(cleaned)
    if not m:
        return None
    doi = m.group(0)
    doi = doi.rstrip(".,;)")
    return doi


def extract_title_heuristic(line: str) -> str:
    """Extract an approximate title from a reference line using heuristics.

    Strategy: strip leading number/bracket prefix, skip author block (up to first period
    after names), take text until the next period that looks like a journal signal or year.
    """
    if not line.strip():
        return ""

    text = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", line).strip()

    segments = text.split(". ")
    if len(segments) < 2:
        return text[:200]

    for i, seg in enumerate(segments):
        if i == 0:
            continue
        words = seg.split()
        if len(words) >= 3:
            return seg.strip().rstrip(".")
        if i >= 3:
            return seg.strip().rstrip(".")

    return segments[1].strip().rstrip(".") if len(segments) > 1 else ""


def parse_reference_line(line: str) -> dict[str, str | None]:
    """Parse a single reference line into structured data.

    Returns:
        dict with keys: 'raw' (original line), 'doi' (str|None), 'title' (str).
    """
    doi = extract_doi(line)
    title = extract_title_heuristic(line)
    return {"raw": line, "doi": doi, "title": title}


def parse_all_references(lines: list[str]) -> list[dict[str, str | None]]:
    """Parse a list of reference lines into structured data.

    Args:
        lines: Raw reference lines from pdf_extractor.extract_bibliography_lines().

    Returns:
        List of dicts, each with 'raw', 'doi', and 'title' keys.
    """
    parsed = []
    for line in lines:
        result = parse_reference_line(line)
        parsed.append(result)
        if result["doi"]:
            logger.debug("reference_parser.doi_found", doi=result["doi"], title=result["title"])
        else:
            logger.debug(
                "reference_parser.no_doi",
                title=result["title"][:60] if result["title"] else "",
            )
    logger.info(
        "reference_parser.complete",
        total=len(parsed),
        with_doi=sum(1 for p in parsed if p["doi"]),
    )
    return parsed
