from __future__ import annotations

from autoreview.models.base import AutoReviewModel

CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert scientific analyst. Extract distinct factual assertions "
    "from the provided review text. Each claim should be a single, self-contained "
    "statement of fact. Categorize each claim precisely. Do not invent claims — "
    "only extract what is explicitly stated or directly implied by the text."
)

_CLAIM_CATEGORIES = """\
Categorize each claim into exactly one category:
- empirical: Specific findings, results, statistics (e.g., "X showed a 30% improvement over Y")
- methodological: Descriptions of approaches, techniques, tools (e.g., "Single-cell RNA-seq enables...")
- contextual: Historical context, background, definitions (e.g., "First described in 1998...")
- synthesis: Cross-study comparisons, contradictions, consensus (e.g., "Despite conflicting results...")
- limitation: Caveats, gaps, open questions (e.g., "No studies have examined...")"""

_CLAIM_INSTRUCTION = """\
Extract every distinct factual claim from the following review section.
Return a JSON list of objects with "text" and "category" fields.

{categories}

Be exhaustive — capture all claims, not just the most prominent ones.
Each claim should be a single sentence that stands on its own."""

_CONCEPT_INSTRUCTION = """\
Extract all distinct scientific concepts, methods, named entities, and technical terms
from the following review text. Return a flat list of normalized strings (lowercase).
Include: disease names, gene/protein names, techniques, metrics, theoretical frameworks,
drug names, organism names, and domain-specific terminology.
Exclude: generic academic terms (e.g., "study", "results", "analysis")."""


class ClaimExtractionItem(AutoReviewModel):
    """Single claim as returned by LLM."""

    text: str
    category: str


class ClaimExtractionResult(AutoReviewModel):
    """LLM response model for claim extraction."""

    claims: list[ClaimExtractionItem]


class ConceptExtractionResult(AutoReviewModel):
    """LLM response model for concept extraction."""

    concepts: list[str]


def build_claim_extraction_prompt(section_text: str, max_chars: int = 40000) -> str:
    instruction = _CLAIM_INSTRUCTION.format(categories=_CLAIM_CATEGORIES)
    return f"{instruction}\n\n=== REVIEW SECTION ===\n{section_text[:max_chars]}"


def build_concept_extraction_prompt(review_text: str, max_chars: int = 40000) -> str:
    return f"{_CONCEPT_INSTRUCTION}\n\n=== REVIEW TEXT ===\n{review_text[:max_chars]}"
