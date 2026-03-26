"""Extract structural profile from a reference paper for reference_match mode."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_NUMBERED_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_AUTHOR_CITE_RE = re.compile(r"\([A-Z][a-z]+ et al\.?,? \d{4}\)")
_HEADING_RE = re.compile(r"^#{1,3}\s+|\n\d+\.?\s+[A-Z]", re.MULTILINE)


class ReferenceAnalyzer:
    """Extract structural profile from reference paper text."""

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze raw text and return structural profile."""
        word_count = len(text.split())

        # Find references section
        ref_start = max(
            text.rfind("References\n"),
            text.rfind("REFERENCES\n"),
            text.rfind("Bibliography\n"),
        )

        citation_count: int | None = None
        if ref_start >= 0:
            ref_section = text[ref_start:]
            refs = _NUMBERED_CITE_RE.findall(ref_section)
            if refs:
                citation_count = max(int(r) for r in refs)
            else:
                # Count lines that look like reference entries
                lines = [
                    line
                    for line in ref_section.split("\n")
                    if line.strip() and not line.startswith("#")
                ]
                citation_count = max(len(lines) - 1, 0) or None

        # Detect citation style
        numbered_count = len(_NUMBERED_CITE_RE.findall(text[:ref_start] if ref_start > 0 else text))
        author_year_count = len(_AUTHOR_CITE_RE.findall(text))
        if numbered_count > author_year_count:
            citation_style = "numbered"
        elif author_year_count > 0:
            citation_style = "author-year"
        else:
            citation_style = "unknown"

        # Count sections (headings)
        section_count = len(_HEADING_RE.findall(text))

        # Citation density
        citation_density: float | None = None
        if citation_count and word_count > 0:
            citation_density = round(citation_count / (word_count / 1000), 1)

        return {
            "word_count": word_count,
            "citation_count": citation_count,
            "citation_density": citation_density,
            "section_count": section_count,
            "citation_style": citation_style,
        }

    def analyze_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Analyze a PDF file and return structural profile.

        Falls back to defaults if PDF cannot be parsed.
        """
        try:
            import fitz  # type: ignore[import-untyped]

            doc = fitz.open(str(pdf_path))
            text = "".join(page.get_text() for page in doc)
            return self.analyze_text(text)
        except Exception as e:
            logger.warning("reference_analyzer.pdf_failed", path=str(pdf_path), error=str(e))
            return {
                "word_count": None,
                "citation_count": None,
                "citation_density": None,
                "section_count": None,
                "citation_style": None,
            }
