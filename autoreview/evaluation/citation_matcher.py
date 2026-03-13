from __future__ import annotations

import re

import structlog
from rapidfuzz import fuzz

from autoreview.evaluation.models import CitationScore
from autoreview.evaluation.pdf_extractor import _parse_ref_lines, normalize_title_for_matching

logger = structlog.get_logger()

_REF_HEADER_MD = re.compile(
    r"^#{1,3}\s*(references|bibliography)",
    re.IGNORECASE | re.MULTILINE,
)

_DOI_RE = re.compile(r"10\.\d{4,}/\S+", re.IGNORECASE)

JACCARD_THRESHOLD: float = 0.45
FUZZY_THRESHOLD: float = 80.0


def parse_bibliography_from_markdown(md_text: str) -> list[str]:
    match = _REF_HEADER_MD.search(md_text)
    if not match:
        return []
    return _parse_ref_lines(md_text[match.end() :])


def _title_from_ref(ref: str) -> str:
    ref = re.sub(r"^\[?\d+[\]\.]\s*", "", ref)
    parts = re.split(r"\.\s+", ref, maxsplit=2)
    return parts[1] if len(parts) >= 2 else ref


def _extract_doi(ref: str) -> str | None:
    """Extract a DOI from a reference string, or return None if absent."""
    m = _DOI_RE.search(ref)
    if m:
        return m.group(0).lower().rstrip(".,;)")
    return None


def _best_match_score(
    query_norm: str,
    query_doi: str | None,
    candidates_norm: list[str],
    candidates_doi: list[str | None],
) -> float:
    """Return the best match score in [0, 1] for query against all candidates.

    Matching cascade:
      1. DOI exact match  → 1.0
      2. Jaccard on normalised title tokens (threshold 0.45)
      3. rapidfuzz token_sort_ratio (threshold 80)
    Returns 0.0 if no candidate meets any threshold.
    """
    best: float = 0.0

    query_tokens = set(query_norm.split())

    for cand_norm, cand_doi in zip(candidates_norm, candidates_doi, strict=False):
        # --- 1. DOI match ---
        if query_doi is not None and cand_doi is not None and query_doi == cand_doi:
            return 1.0

        # --- 2. Jaccard ---
        cand_tokens = set(cand_norm.split())
        union = query_tokens | cand_tokens
        if union:
            jaccard = len(query_tokens & cand_tokens) / len(union)
            if jaccard >= JACCARD_THRESHOLD:
                best = max(best, jaccard)
                continue  # already good enough; no need for fuzzy

        # --- 3. rapidfuzz token_sort_ratio ---
        ratio = fuzz.token_sort_ratio(query_norm, cand_norm)
        if ratio >= FUZZY_THRESHOLD:
            # Normalise to [0, 1] so all scores live on the same scale
            best = max(best, ratio / 100.0)

    return best


def match_citations(
    generated_refs: list[str],
    reference_refs: list[str],
    threshold: float = 0.45,
) -> CitationScore:
    """Match generated references against reference references bidirectionally.

    Forward pass  → recall + missed_titles
    Reverse pass  → precision + hallucinated_titles
    F1            → harmonic mean of precision and recall
    """
    ref_norms = [normalize_title_for_matching(_title_from_ref(r)) for r in reference_refs]
    gen_norms = [normalize_title_for_matching(_title_from_ref(g)) for g in generated_refs]

    ref_dois = [_extract_doi(r) for r in reference_refs]
    gen_dois = [_extract_doi(g) for g in generated_refs]

    # --- Forward pass: for each reference ref, find best match in generated refs ---
    matched: list[str] = []
    missed: list[str] = []

    for i, (ref_norm, ref_doi) in enumerate(zip(ref_norms, ref_dois, strict=False)):
        if not ref_norm and ref_doi is None:
            continue
        score = _best_match_score(ref_norm, ref_doi, gen_norms, gen_dois)
        (matched if score >= threshold else missed).append(reference_refs[i])

    recall = len(matched) / len(reference_refs) if reference_refs else 0.0

    # --- Reverse pass: for each generated ref, find best match in reference refs ---
    precision_matched: list[str] = []
    hallucinated: list[str] = []

    for j, (gen_norm, gen_doi) in enumerate(zip(gen_norms, gen_dois, strict=False)):
        if not gen_norm and gen_doi is None:
            continue
        score = _best_match_score(gen_norm, gen_doi, ref_norms, ref_dois)
        (precision_matched if score >= threshold else hallucinated).append(generated_refs[j])

    precision = len(precision_matched) / len(generated_refs) if generated_refs else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    logger.info(
        "citation_matcher",
        matched=len(matched),
        total_reference=len(reference_refs),
        total_generated=len(generated_refs),
        recall=f"{recall:.2f}",
        precision=f"{precision:.2f}",
        f1=f"{f1:.2f}",
        hallucinated=len(hallucinated),
    )

    return CitationScore(
        recall=recall,
        precision=precision,
        f1=f1,
        matched_count=len(matched),
        reference_count=len(reference_refs),
        generated_count=len(generated_refs),
        matched_titles=matched,
        missed_titles=missed,
        hallucinated_titles=hallucinated,
    )
