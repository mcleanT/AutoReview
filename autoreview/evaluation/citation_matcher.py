from __future__ import annotations

import re

import structlog

from autoreview.evaluation.models import CitationScore
from autoreview.evaluation.pdf_extractor import (
    _parse_ref_lines,
    extract_bibliography_lines,
    normalize_title_for_matching,
)

logger = structlog.get_logger()

_REF_HEADER_MD = re.compile(
    r"^#{1,3}\s*(references|bibliography)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_bibliography_from_markdown(md_text: str) -> list[str]:
    match = _REF_HEADER_MD.search(md_text)
    if not match:
        return []
    return _parse_ref_lines(md_text[match.end():])


def _title_from_ref(ref: str) -> str:
    ref = re.sub(r"^\[?\d+[\]\.]\s*", "", ref)
    parts = re.split(r"\.\s+", ref, maxsplit=2)
    return parts[1] if len(parts) >= 2 else ref


def match_citations(
    generated_refs: list[str],
    reference_refs: list[str],
    threshold: float = 0.45,
) -> CitationScore:
    ref_norms = [normalize_title_for_matching(_title_from_ref(r)) for r in reference_refs]
    gen_norms = [normalize_title_for_matching(_title_from_ref(g)) for g in generated_refs]

    matched: list[str] = []
    missed: list[str] = []

    for i, ref_norm in enumerate(ref_norms):
        ref_tokens = set(ref_norm.split())
        if not ref_tokens:
            continue
        best = max(
            (
                len(ref_tokens & set(g.split())) / len(ref_tokens | set(g.split()))
                for g in gen_norms
                if g
            ),
            default=0.0,
        )
        (matched if best >= threshold else missed).append(reference_refs[i])

    recall = len(matched) / len(reference_refs) if reference_refs else 0.0
    logger.info(
        "citation_matcher",
        matched=len(matched),
        total=len(reference_refs),
        recall=f"{recall:.2f}",
    )

    return CitationScore(
        recall=recall,
        matched_count=len(matched),
        reference_count=len(reference_refs),
        generated_count=len(generated_refs),
        matched_titles=matched,
        missed_titles=missed,
    )
