from __future__ import annotations

import re
import statistics

import structlog
import textstat

from autoreview.evaluation.models import StructuralMetrics

logger = structlog.get_logger()

# Matches markdown headings at levels 1–3: #, ##, ###
_HEADING_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)

# Bracket-style citations: [1], [23]
_BRACKET_CITE_RE = re.compile(r"\[\d+\]")

# Author-et-al-style citations: (Smith et al., 2023) or (Smith et al. 2023)
_AUTHOR_CITE_RE = re.compile(r"\([A-Z][a-z]+ et al\.?,? \d{4}\)")


def _strip_markdown(text: str) -> str:
    """Convert markdown to plain text suitable for readability scoring."""
    # Replace [link text](url) with just link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (**, __, *, _)
    text = re.sub(r"\*{1,2}|_{1,2}", "", text)
    # Remove inline code backticks
    text = re.sub(r"`[^`]*`", "", text)
    return text


def _count_words(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split())


def _split_into_sections(text: str) -> list[str]:
    """Split markdown text into sections delimited by level 1–3 headings.

    Returns a list of section body strings (the heading line itself is excluded).
    If there are no headings, the entire text is treated as one section.
    """
    parts = _HEADING_RE.split(text)
    # The first part is text before the first heading (preamble); subsequent
    # parts begin after each heading marker.  We keep all non-empty parts.
    sections = [p.strip() for p in parts if p.strip()]
    return sections if sections else [text]


def _count_citations(text: str) -> int:
    """Count citation occurrences using both bracket and author-et-al patterns."""
    bracket_matches = set(_BRACKET_CITE_RE.findall(text))
    author_matches = set(_AUTHOR_CITE_RE.findall(text))
    return len(bracket_matches) + len(author_matches)


def compute_structural_metrics(markdown_text: str) -> StructuralMetrics:
    """Compute deterministic structural metrics from markdown text.

    No LLM calls are made.  All metrics are derived from regex analysis and
    the ``textstat`` library.

    Args:
        markdown_text: The full markdown content of the generated review.

    Returns:
        A :class:`StructuralMetrics` instance with all fields populated.
        For empty input all numeric fields are zero.
    """
    if not markdown_text.strip():
        logger.debug("structural_metrics.empty_input")
        return StructuralMetrics(
            word_count=0,
            section_count=0,
            citation_count=0,
            citations_per_1000_words=0.0,
            avg_section_length_words=0.0,
            section_balance=0.0,
            flesch_kincaid_grade=0.0,
        )

    plain_text = _strip_markdown(markdown_text)
    word_count = _count_words(plain_text)

    sections = _split_into_sections(markdown_text)
    section_count = len(sections)

    section_lengths = [_count_words(_strip_markdown(s)) for s in sections]
    avg_section_length = sum(section_lengths) / section_count

    if section_count > 1 and avg_section_length > 0:
        section_balance = round(statistics.stdev(section_lengths) / avg_section_length, 4)
    else:
        section_balance = 0.0

    citation_count = _count_citations(markdown_text)
    citations_per_1000_words = (
        round(citation_count / (word_count / 1000), 4) if word_count > 0 else 0.0
    )

    fk_grade = round(textstat.flesch_kincaid_grade(plain_text), 2)

    logger.debug(
        "structural_metrics.computed",
        word_count=word_count,
        section_count=section_count,
        citation_count=citation_count,
        fk_grade=fk_grade,
    )

    return StructuralMetrics(
        word_count=word_count,
        section_count=section_count,
        citation_count=citation_count,
        citations_per_1000_words=citations_per_1000_words,
        avg_section_length_words=round(avg_section_length, 2),
        section_balance=section_balance,
        flesch_kincaid_grade=fk_grade,
    )
