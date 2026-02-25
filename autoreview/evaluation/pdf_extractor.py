from __future__ import annotations

import re
from pathlib import Path

import structlog

logger = structlog.get_logger()

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "with", "by", "from", "that", "this", "is",
    "was", "are",
}
_REF_HEADER = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_BRACKET = re.compile(r"^\s*\[\d+\]\s+\S")
_NUMBERED_DOT = re.compile(r"^\s*\d+\.\s+[A-Z]")


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")
    text_parts: list[str] = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    full_text = "\n".join(text_parts)
    logger.info("pdf_extractor.extracted", path=str(pdf_path), chars=len(full_text))
    return full_text


def _parse_ref_lines(ref_text: str) -> list[str]:
    """Parse numbered reference lines from text that begins right after a reference header."""
    lines: list[str] = []
    for raw in ref_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _NUMBERED_BRACKET.match(line) or _NUMBERED_DOT.match(line):
            lines.append(line)
        elif lines and not re.match(
            r"^\s*(figure|table|appendix|footnote)", line, re.IGNORECASE
        ):
            lines[-1] += " " + line  # continuation of previous ref
    return lines


def extract_bibliography_lines(text: str) -> list[str]:
    match = _REF_HEADER.search(text)
    if not match:
        return []
    return _parse_ref_lines(text[match.end():])


def normalize_title_for_matching(title: str) -> str:
    title = re.sub(r"[^\w\s]", " ", title.lower())
    tokens = [w for w in title.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(sorted(tokens))
