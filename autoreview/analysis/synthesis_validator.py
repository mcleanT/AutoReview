"""Synthesis validation metric for review paper quality assessment."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# Matches one or more citation groups: [@id1; @id2] or [@id1][@id2]
_CITATION_BLOCK_RE = re.compile(r"\[@([^\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SynthesisMetrics:
    """Metrics quantifying the degree of synthesis in a review text.

    Attributes:
        multi_paper_sentences: Sentences that cite 2+ distinct papers.
        single_paper_sentences: Sentences that cite exactly 1 paper.
        total_cited_sentences: Total sentences containing at least one citation.
        synthesis_ratio: multi_paper_sentences / total_cited_sentences,
            or 0.0 when total_cited_sentences == 0.
    """

    multi_paper_sentences: int
    single_paper_sentences: int
    total_cited_sentences: int
    synthesis_ratio: float


def compute_synthesis_ratio(text: str) -> SynthesisMetrics:
    """Compute synthesis metrics for *text* using ``[@paper_id]`` citation markers.

    A sentence is considered "multi-paper" if it contains citation markers
    referencing 2 or more distinct paper IDs.  Within a single bracket block,
    IDs are separated by ``; @`` (e.g. ``[@p1; @p2; @p3]``).

    Args:
        text: Review text containing ``[@paper_id]`` citation markers.

    Returns:
        SynthesisMetrics with counts and ratio.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())

    multi_paper = 0
    single_paper = 0

    for sentence in sentences:
        matches = _CITATION_BLOCK_RE.findall(sentence)
        if not matches:
            continue

        # Collect all unique paper IDs across every citation block in this sentence
        paper_ids: set[str] = set()
        for block in matches:
            # Each block may contain multiple IDs separated by "; @"
            # e.g. "p1; @p2; @p3" → ["p1", "p2", "p3"]
            ids = [part.strip().lstrip("@") for part in re.split(r";\s*@", block)]
            paper_ids.update(ids)

        if len(paper_ids) >= 2:
            multi_paper += 1
        else:
            single_paper += 1

    total_cited = multi_paper + single_paper
    synthesis_ratio = multi_paper / total_cited if total_cited > 0 else 0.0

    logger.info(
        "synthesis_validator.compute",
        multi_paper_sentences=multi_paper,
        single_paper_sentences=single_paper,
        total_cited_sentences=total_cited,
        synthesis_ratio=round(synthesis_ratio, 4),
    )

    return SynthesisMetrics(
        multi_paper_sentences=multi_paper,
        single_paper_sentences=single_paper,
        total_cited_sentences=total_cited,
        synthesis_ratio=synthesis_ratio,
    )
