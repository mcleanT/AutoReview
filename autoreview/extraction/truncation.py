"""Section-aware text truncation for extraction pipeline.

Parses academic paper structure and drops low-value sections (References,
Acknowledgments, etc.) while preserving high-value content (Abstract,
Introduction, Results, Discussion, Conclusion). Falls back to head/tail
truncation when section parsing fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from autoreview.config.models import SectionTruncationConfig

logger = structlog.get_logger()

# Max chars for a line to be considered a heading (avoids matching body text)
_MAX_HEADING_LEN = 80

# Regex for common academic section headings:
# - Numbered: "1. Introduction", "2.1 Methods"
# - Roman numeral: "I. Methods", "II. Results"
# - Bare uppercase: "RESULTS", "DISCUSSION"
# - Title case on its own line: "Introduction", "Results and Discussion"
_HEADING_PATTERN = re.compile(
    r"^"
    r"(?:"
    r"(?:\d+(?:\.\d+)*\.?[ \t]+)"  # Numbered: "1. ", "2.1 "
    r"|(?:[IVXivx]+\.?[ \t]+)"  # Roman numeral: "I. ", "IV "
    r")?"
    r"([A-Z][A-Za-z \t,&\-/]+)"  # Heading text (no newline in capture)
    r"[ \t]*$",
    re.MULTILINE,
)


@dataclass
class ParsedSection:
    """A detected section in a paper."""

    name: str
    start: int
    end: int  # exclusive; set during post-processing
    text: str = ""


def parse_sections(text: str) -> list[ParsedSection]:
    """Parse academic paper text into sections based on heading detection.

    Returns a list of ParsedSection objects. Returns an empty list if no
    headings are detected (triggers fallback to head/tail truncation).
    """
    # Filter matches to only include lines short enough to be headings
    matches = [
        m for m in _HEADING_PATTERN.finditer(text) if len(m.group(0).strip()) <= _MAX_HEADING_LEN
    ]

    if len(matches) < 2:
        # Not enough structure to parse meaningfully
        return []

    sections: list[ParsedSection] = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            ParsedSection(
                name=name,
                start=start,
                end=end,
                text=text[start:end],
            )
        )

    return sections


def _normalize_name(name: str) -> str:
    """Normalize a section name for matching against config lists."""
    return name.lower().strip()


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Check if a section name matches any pattern in the list."""
    normalized = _normalize_name(name)
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in normalized or normalized in pattern_lower:
            return True
    return False


def _head_tail_truncate(text: str, max_chars: int) -> str:
    """Truncate text preserving head (abstract/intro) and tail (results/discussion).

    This is the legacy truncation strategy, used as a fallback when section
    parsing fails or the text is still over max_chars after section filtering.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 3  # abstract + intro
    tail = max_chars - head  # results + discussion
    return text[:head] + "\n\n[...truncated middle section...]\n\n" + text[-tail:]


def section_aware_truncate(
    text: str,
    max_chars: int,
    config: SectionTruncationConfig,
) -> str:
    """Truncate paper text using section-aware logic.

    1. Parse sections from the text
    2. Drop sections matching config.drop_sections
    3. Cap Introduction and Methods per config
    4. Fall back to head/tail if no sections detected or still over max_chars

    Always performs section filtering (to save tokens) even if text is under
    max_chars. Only applies head/tail truncation if the result is still over
    max_chars after section filtering.
    """
    sections = parse_sections(text)

    if not sections:
        # No structure detected — use head/tail if over limit, else return as-is
        if len(text) > max_chars:
            logger.debug("truncation.no_sections_detected", text_len=len(text))
            return _head_tail_truncate(text, max_chars)
        return text

    # Collect text before first section heading (often contains title/abstract)
    preamble = text[: sections[0].start]

    kept_parts: list[str] = []
    if preamble.strip():
        kept_parts.append(preamble)

    for section in sections:
        name_lower = _normalize_name(section.name)

        # Drop sections in the drop list
        if _matches_any(section.name, config.drop_sections):
            continue

        section_text = section.text

        # Cap Introduction
        if (
            "introduction" in name_lower
            and config.intro_max_chars > 0
            and len(section_text) > config.intro_max_chars
        ):
            section_text = (
                section_text[: config.intro_max_chars] + "\n[...introduction truncated...]\n"
            )

        # Cap or drop Methods
        if "method" in name_lower:
            if config.methods_max_chars == 0:
                continue  # Drop entirely
            if len(section_text) > config.methods_max_chars:
                section_text = (
                    section_text[: config.methods_max_chars] + "\n[...methods truncated...]\n"
                )

        kept_parts.append(section_text)

    result = "\n".join(kept_parts)
    original_len = len(text)
    reduction_pct = round((1 - len(result) / original_len) * 100, 1) if original_len else 0

    logger.debug(
        "truncation.section_aware",
        original_len=original_len,
        result_len=len(result),
        reduction_pct=reduction_pct,
        sections_found=len(sections),
    )

    # If still over max_chars after section filtering, apply head/tail as final safety
    if len(result) > max_chars:
        result = _head_tail_truncate(result, max_chars)

    return result
