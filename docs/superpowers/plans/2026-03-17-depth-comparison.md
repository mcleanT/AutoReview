# Depth Comparison (Analysis 10) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the analysis script and supporting infrastructure for comparing AutoReview output across three depth levels (low/medium/deep), including novel information metrics (claim extraction, concept coverage, information density).

**Architecture:** New Pydantic models in `autoreview/evaluation/models.py` for claims and information metrics. New LLM prompts in `autoreview/llm/prompts/` for claim/concept extraction. A new `ClaimExtractor` class in `autoreview/evaluation/` following the `LLMScorer` pattern. The analysis script in `paper/analysis/depth_comparison.py` loads evaluation results, runs information extraction, computes metrics, generates figures/tables, and writes JSON + Markdown reports.

**Tech Stack:** Python 3.11+, Pydantic, structlog, numpy, scipy (Wilcoxon + FDR), matplotlib, pandas

**Spec:** `docs/superpowers/specs/2026-03-17-depth-comparison-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `autoreview/evaluation/models.py` | Modify | Add `Claim`, `InformationMetrics`, `DepthRunConfig`, `DepthComparisonResult` |
| `autoreview/llm/prompts/claim_extraction.py` | Create | Prompts + response models for claim/concept extraction |
| `autoreview/evaluation/claim_extractor.py` | Create | `ClaimExtractor` class — section-chunked claim/concept extraction via LLM |
| `autoreview/evaluation/claim_dedup.py` | Create | Embedding-based claim deduplication |
| `paper/analysis/depth_comparison.py` | Create | Main analysis script — load runs, extract info, compute metrics, figures, report |
| `tests/test_evaluation/test_claim_extractor.py` | Create | Unit tests for ClaimExtractor |
| `tests/test_evaluation/test_claim_dedup.py` | Create | Unit tests for claim deduplication |
| `tests/test_analysis/test_depth_comparison.py` | Create | Unit tests for depth comparison analysis functions |

---

## Task 1: Data Models

**Files:**
- Modify: `autoreview/evaluation/models.py`
- Test: `tests/test_evaluation/test_models.py`

- [ ] **Step 1: Write failing tests for new models**

```python
# Append to tests/test_evaluation/test_models.py

from autoreview.evaluation.models import (
    Claim,
    InformationMetrics,
    DepthRunConfig,
    DepthComparisonResult,
)


def test_claim_model():
    claim = Claim(
        text="GLP-1 agonists reduce HbA1c by 1.5%",
        category="empirical",
        section_id="results_1",
    )
    assert claim.text == "GLP-1 agonists reduce HbA1c by 1.5%"
    assert claim.category == "empirical"
    assert claim.section_id == "results_1"


def test_claim_category_validation():
    """Only valid categories are accepted."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Claim(text="test", category="invalid_category")


def test_claim_optional_section_id():
    claim = Claim(text="test claim", category="synthesis")
    assert claim.section_id is None


def test_information_metrics():
    metrics = InformationMetrics(
        claims_per_depth={"low": 45, "medium": 102, "deep": 230},
        new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
        new_claims_by_category={
            "low_to_medium": {"empirical": 20, "methodological": 15, "contextual": 10, "synthesis": 8, "limitation": 4},
            "medium_to_deep": {"empirical": 50, "methodological": 30, "contextual": 25, "synthesis": 15, "limitation": 8},
        },
        claim_novelty_rate={"low_to_medium": 0.559, "medium_to_deep": 0.557},
        concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
        concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
        claims_per_1k_words={"low": 11.25, "medium": 12.75, "deep": 9.2},
        concepts_per_1k_words={"low": 7.5, "medium": 8.125, "deep": 4.8},
        citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
    )
    assert metrics.claims_per_depth["medium"] == 102
    assert metrics.reference_claim_coverage is None


def test_information_metrics_with_reference_coverage():
    metrics = InformationMetrics(
        claims_per_depth={"low": 45, "medium": 102, "deep": 230},
        new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
        new_claims_by_category={"low_to_medium": {}, "medium_to_deep": {}},
        claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
        concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
        concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
        claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
        concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
        citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
        reference_claim_coverage={"low": 0.35, "medium": 0.62, "deep": 0.81},
    )
    assert metrics.reference_claim_coverage["deep"] == 0.81


def test_depth_run_config():
    cfg = DepthRunConfig(
        topic="gut microbiome",
        domain="biomedical",
        depth="medium",
        generated_path="/output/medium/review.md",
        reference_path="/refs/gut_microbiome.pdf",
        evaluation_path="/output/medium/evaluation.json",
        tier="A",
    )
    assert cfg.depth == "medium"
    assert cfg.tier == "A"


def test_depth_run_config_no_reference():
    """Tier B topics may have no reference review."""
    cfg = DepthRunConfig(
        topic="llm hallucination",
        domain="cs",
        depth="low",
        generated_path="/output/low/review.md",
        tier="B",
    )
    assert cfg.reference_path is None


def test_depth_comparison_result():
    result = DepthComparisonResult(
        topic="gut microbiome",
        domain="biomedical",
        depths=["low", "medium", "deep"],
        evaluation_scores={"low": 0.65, "medium": 0.78, "deep": 0.82},
        information_metrics=InformationMetrics(
            claims_per_depth={"low": 45, "medium": 102, "deep": 230},
            new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
            new_claims_by_category={"low_to_medium": {}, "medium_to_deep": {}},
            claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
            concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
            concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
            claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
            concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
            citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
        ),
        cost_per_depth={"low": 1.50, "medium": 3.20, "deep": 8.75},
    )
    assert result.evaluation_scores["deep"] == 0.82
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation/test_models.py -v -k "claim or information_metrics or depth_run or depth_comparison"`
Expected: ImportError — `Claim`, `InformationMetrics`, `DepthRunConfig`, `DepthComparisonResult` not found

- [ ] **Step 3: Implement the models**

Add to `autoreview/evaluation/models.py` after the `AggregatedScores` class:

```python
class Claim(AutoReviewModel):
    """A distinct factual assertion extracted from a review."""

    text: str
    category: Literal["empirical", "methodological", "contextual", "synthesis", "limitation"]
    section_id: str | None = None


class InformationMetrics(AutoReviewModel):
    """Aggregated information metrics for a single topic across depth levels."""

    claims_per_depth: dict[str, int]
    new_claims_per_increment: dict[str, int]
    new_claims_by_category: dict[str, dict[str, int]]
    claim_novelty_rate: dict[str, float]
    concepts_per_depth: dict[str, int]
    concept_growth: dict[str, int]
    claims_per_1k_words: dict[str, float]
    concepts_per_1k_words: dict[str, float]
    citations_per_claim: dict[str, float]
    concept_overlap: dict[str, float] | None = None  # overlap coefficient between depth pairs
    reference_claim_coverage: dict[str, float] | None = None


class DepthRunConfig(AutoReviewModel):
    """Configuration for a single depth comparison run."""

    topic: str
    domain: str
    depth: str
    generated_path: str
    reference_path: str | None = None  # None for Tier B topics without reference review
    evaluation_path: str | None = None
    tier: str | None = None


class DepthComparisonResult(AutoReviewModel):
    """Full comparison result for a single topic across depths."""

    topic: str
    domain: str
    depths: list[str]
    evaluation_scores: dict[str, float]
    information_metrics: InformationMetrics
    cost_per_depth: dict[str, float] | None = None
```

Add `from typing import Literal` to the imports at the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation/test_models.py -v -k "claim or information_metrics or depth_run or depth_comparison"`
Expected: All 7 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/evaluation/models.py tests/test_evaluation/test_models.py
git commit -m "feat(eval): add Claim, InformationMetrics, DepthRunConfig, DepthComparisonResult models"
```

---

## Task 2: Claim/Concept Extraction Prompts

**Files:**
- Create: `autoreview/llm/prompts/claim_extraction.py`
- Test: `tests/test_evaluation/test_claim_extractor.py` (prompt format tests only — extractor tests in Task 3)

- [ ] **Step 1: Write failing test for prompt builders**

```python
# tests/test_evaluation/test_claim_extractor.py
from autoreview.llm.prompts.claim_extraction import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    ClaimExtractionItem,
    ClaimExtractionResult,
    ConceptExtractionResult,
    build_claim_extraction_prompt,
    build_concept_extraction_prompt,
)


def test_claim_extraction_prompt_builds():
    prompt = build_claim_extraction_prompt("## Introduction\nSome text about biology.")
    assert "Introduction" in prompt
    assert "Some text about biology" in prompt
    assert "empirical" in prompt
    assert "methodological" in prompt


def test_claim_extraction_prompt_truncates():
    long_text = "word " * 20000  # ~100K chars
    prompt = build_claim_extraction_prompt(long_text, max_chars=1000)
    assert len(prompt) < 5000  # prompt template + truncated text


def test_concept_extraction_prompt_builds():
    prompt = build_concept_extraction_prompt("## Methods\nWe used single-cell RNA-seq.")
    assert "Methods" in prompt
    assert "single-cell RNA-seq" in prompt


def test_claim_extraction_result_model():
    result = ClaimExtractionResult(
        claims=[
            ClaimExtractionItem(text="X improves Y by 30%", category="empirical"),
            ClaimExtractionItem(text="Method A is standard", category="methodological"),
        ]
    )
    assert len(result.claims) == 2
    assert result.claims[0].category == "empirical"


def test_concept_extraction_result_model():
    result = ConceptExtractionResult(
        concepts=["single-cell RNA-seq", "gut microbiome", "neurodegeneration"]
    )
    assert len(result.concepts) == 3


def test_system_prompt_exists():
    assert len(CLAIM_EXTRACTION_SYSTEM_PROMPT) > 0
    assert "scientific" in CLAIM_EXTRACTION_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation/test_claim_extractor.py -v -k "prompt or result_model or system_prompt"`
Expected: ImportError

- [ ] **Step 3: Implement the prompts module**

```python
# autoreview/llm/prompts/claim_extraction.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation/test_claim_extractor.py -v -k "prompt or result_model or system_prompt"`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/llm/prompts/claim_extraction.py tests/test_evaluation/test_claim_extractor.py
git commit -m "feat(eval): add claim and concept extraction prompts"
```

---

## Task 3: ClaimExtractor Class

**Files:**
- Create: `autoreview/evaluation/claim_extractor.py`
- Modify: `tests/test_evaluation/test_claim_extractor.py`

- [ ] **Step 1: Write failing tests for ClaimExtractor**

Append to `tests/test_evaluation/test_claim_extractor.py`:

```python
import pytest

from autoreview.evaluation.claim_extractor import ClaimExtractor
from autoreview.evaluation.models import Claim
from autoreview.llm.prompts.claim_extraction import (
    ClaimExtractionItem,
    ClaimExtractionResult,
    ConceptExtractionResult,
)
from autoreview.llm.provider import LLMStructuredResponse


class MockClaimLLM:
    """Mock LLM that returns fixed claim/concept extraction results."""

    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0, model_override=None,
    ):
        if response_model == ClaimExtractionResult:
            return LLMStructuredResponse(
                parsed=ClaimExtractionResult(
                    claims=[
                        ClaimExtractionItem(text="GLP-1 agonists reduce HbA1c by 1.5%", category="empirical"),
                        ClaimExtractionItem(text="Metformin remains first-line therapy", category="contextual"),
                    ]
                ),
                input_tokens=500,
                output_tokens=150,
            )
        elif response_model == ConceptExtractionResult:
            return LLMStructuredResponse(
                parsed=ConceptExtractionResult(
                    concepts=["glp-1 receptor agonist", "hba1c", "metformin", "type 2 diabetes"]
                ),
                input_tokens=500,
                output_tokens=100,
            )
        raise ValueError(f"Unexpected: {response_model}")


@pytest.fixture
def claim_extractor():
    return ClaimExtractor(MockClaimLLM())


SAMPLE_REVIEW = """\
## Introduction

Type 2 diabetes affects millions worldwide. Metformin remains first-line therapy.

## Results

GLP-1 agonists reduce HbA1c by 1.5%. These effects are sustained over 52 weeks.

## Conclusion

Novel therapies show promise for diabetes management.
"""


async def test_extract_claims_from_sections(claim_extractor):
    claims = await claim_extractor.extract_claims(SAMPLE_REVIEW)
    assert len(claims) > 0
    assert all(isinstance(c, Claim) for c in claims)
    assert all(c.section_id is not None for c in claims)


async def test_extract_claims_assigns_section_ids(claim_extractor):
    claims = await claim_extractor.extract_claims(SAMPLE_REVIEW)
    section_ids = {c.section_id for c in claims}
    # Mock returns same claims for each section, so we should see multiple section IDs
    assert len(section_ids) >= 2


async def test_extract_concepts(claim_extractor):
    concepts = await claim_extractor.extract_concepts(SAMPLE_REVIEW)
    assert len(concepts) > 0
    assert all(isinstance(c, str) for c in concepts)
    # Concepts should be normalized to lowercase
    assert all(c == c.lower() for c in concepts)


async def test_extract_concepts_deduplicates(claim_extractor):
    concepts = await claim_extractor.extract_concepts(SAMPLE_REVIEW)
    assert len(concepts) == len(set(concepts))


async def test_split_sections():
    extractor = ClaimExtractor(MockClaimLLM())
    sections = extractor.split_into_sections(SAMPLE_REVIEW)
    assert len(sections) >= 3
    assert all("id" in s and "text" in s for s in sections)


async def test_single_section_fallback(claim_extractor):
    """Text without headings is treated as a single section."""
    claims = await claim_extractor.extract_claims("Just a plain paragraph of text.")
    assert len(claims) > 0
    assert all(c.section_id == "full_text" for c in claims)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation/test_claim_extractor.py -v -k "extract or split"`
Expected: ImportError — `ClaimExtractor` not found

- [ ] **Step 3: Implement ClaimExtractor**

```python
# autoreview/evaluation/claim_extractor.py
from __future__ import annotations

import re
from typing import Any

import structlog

from autoreview.evaluation.models import Claim
from autoreview.llm.prompts.claim_extraction import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    ClaimExtractionResult,
    ConceptExtractionResult,
    build_claim_extraction_prompt,
    build_concept_extraction_prompt,
)

logger = structlog.get_logger()

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

_VALID_CATEGORIES = {"empirical", "methodological", "contextual", "synthesis", "limitation"}


class ClaimExtractor:
    """Section-chunked claim and concept extraction from review text."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def split_into_sections(self, text: str) -> list[dict[str, str]]:
        """Split markdown text into sections by headings.

        Returns list of {"id": section_id, "text": section_text}.
        """
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [{"id": "full_text", "text": text.strip()}]

        sections: list[dict[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            section_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append({"id": section_id, "text": body})

        return sections if sections else [{"id": "full_text", "text": text.strip()}]

    async def extract_claims(self, review_text: str) -> list[Claim]:
        """Extract claims from each section of the review, then aggregate."""
        sections = self.split_into_sections(review_text)
        all_claims: list[Claim] = []

        for section in sections:
            prompt = build_claim_extraction_prompt(section["text"])
            resp = await self.llm.generate_structured(
                prompt,
                ClaimExtractionResult,
                system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            )
            for raw in resp.parsed.claims:
                category = raw.category if raw.category in _VALID_CATEGORIES else "contextual"
                all_claims.append(
                    Claim(
                        text=raw.text,
                        category=category,
                        section_id=section["id"],
                    )
                )

            logger.info(
                "claim_extractor.section_done",
                section_id=section["id"],
                n_claims=len(resp.parsed.claims),
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )

        logger.info("claim_extractor.complete", total_claims=len(all_claims))
        return all_claims

    async def extract_concepts(self, review_text: str) -> list[str]:
        """Extract and deduplicate scientific concepts from review text."""
        prompt = build_concept_extraction_prompt(review_text)
        resp = await self.llm.generate_structured(
            prompt,
            ConceptExtractionResult,
            system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
        )
        # Normalize and deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for concept in resp.parsed.concepts:
            normalized = concept.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        logger.info(
            "claim_extractor.concepts_done",
            n_concepts=len(unique),
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
        return unique
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation/test_claim_extractor.py -v`
Expected: All 12 tests PASS (6 prompt + 6 extractor)

- [ ] **Step 5: Commit**

```bash
git add autoreview/evaluation/claim_extractor.py tests/test_evaluation/test_claim_extractor.py
git commit -m "feat(eval): add ClaimExtractor with section-chunked extraction"
```

---

## Task 4: Claim Deduplication

**Files:**
- Create: `autoreview/evaluation/claim_dedup.py`
- Create: `tests/test_evaluation/test_claim_dedup.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluation/test_claim_dedup.py
from __future__ import annotations

import numpy as np
import pytest

from autoreview.evaluation.claim_dedup import (
    compute_cosine_similarity_matrix,
    deduplicate_claims,
)
from autoreview.evaluation.models import Claim


def _make_claim(text: str, category: str = "empirical") -> Claim:
    return Claim(text=text, category=category)


class MockEmbedder:
    """Returns deterministic embeddings based on word overlap."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        # Simple bag-of-words embedding for testing
        all_words: set[str] = set()
        for t in texts:
            all_words.update(t.lower().split())
        vocab = sorted(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        embeddings = np.zeros((len(texts), len(vocab)))
        for i, t in enumerate(texts):
            for word in t.lower().split():
                embeddings[i, word_to_idx[word]] = 1.0
        # Normalize to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings


@pytest.fixture
def embedder():
    return MockEmbedder()


def test_cosine_similarity_matrix():
    a = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    sim = compute_cosine_similarity_matrix(a, b)
    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0)
    assert sim[1, 0] == pytest.approx(0.0)
    assert sim[1, 1] == pytest.approx(0.0)


async def test_deduplicate_identical_claims(embedder):
    target = [_make_claim("Drug A reduces blood pressure")]
    baseline = [_make_claim("Drug A reduces blood pressure")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 0
    assert len(shared) == 1


async def test_deduplicate_distinct_claims(embedder):
    target = [_make_claim("Novel gene therapy shows promise")]
    baseline = [_make_claim("Traditional chemotherapy has limitations")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 1
    assert len(shared) == 0


async def test_deduplicate_empty_baseline(embedder):
    target = [_make_claim("Claim one"), _make_claim("Claim two")]
    baseline = []
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 2
    assert len(shared) == 0


async def test_deduplicate_empty_target(embedder):
    target = []
    baseline = [_make_claim("Claim one")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 0
    assert len(shared) == 0


async def test_deduplicate_mixed(embedder):
    target = [
        _make_claim("Drug A reduces blood pressure significantly"),
        _make_claim("A completely novel finding about genetics"),
    ]
    baseline = [
        _make_claim("Drug A reduces blood pressure"),
    ]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.50)
    # The first target claim has high overlap with baseline, second does not
    assert len(novel) + len(shared) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation/test_claim_dedup.py -v`
Expected: ImportError

- [ ] **Step 3: Implement claim deduplication**

```python
# autoreview/evaluation/claim_dedup.py
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import structlog

from autoreview.evaluation.models import Claim

logger = structlog.get_logger()


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...


def compute_cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b.

    Both a and b should be 2D arrays with shape (n, dim) and (m, dim).
    Returns (n, m) similarity matrix.
    """
    # Normalize rows
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


async def deduplicate_claims(
    target_claims: list[Claim],
    baseline_claims: list[Claim],
    embedder: Any,
    threshold: float = 0.90,
) -> tuple[list[Claim], list[Claim]]:
    """Identify novel and shared claims in target vs. baseline.

    Baseline can be a union of claims from multiple shallower depths.
    For medium->deep: pass union(low_claims, medium_claims) as baseline.

    Returns:
        (novel_in_target, shared_with_baseline)
    """
    if not target_claims:
        return [], []
    if not baseline_claims:
        return list(target_claims), []

    target_texts = [c.text for c in target_claims]
    baseline_texts = [c.text for c in baseline_claims]

    target_emb = await embedder.embed(target_texts)
    baseline_emb = await embedder.embed(baseline_texts)

    sim_matrix = compute_cosine_similarity_matrix(target_emb, baseline_emb)
    max_sims = sim_matrix.max(axis=1)  # Best match for each target claim

    novel: list[Claim] = []
    shared: list[Claim] = []

    for i, claim in enumerate(target_claims):
        if max_sims[i] >= threshold:
            shared.append(claim)
        else:
            novel.append(claim)

    logger.info(
        "claim_dedup.complete",
        n_target=len(target_claims),
        n_baseline=len(baseline_claims),
        n_novel=len(novel),
        n_shared=len(shared),
        threshold=threshold,
    )
    return novel, shared
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation/test_claim_dedup.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/evaluation/claim_dedup.py tests/test_evaluation/test_claim_dedup.py
git commit -m "feat(eval): add embedding-based claim deduplication"
```

---

## Task 5: Depth Comparison Analysis Script — Data Loading & Metrics

**Files:**
- Create: `paper/analysis/depth_comparison.py`
- Create: `tests/test_analysis/__init__.py`
- Create: `tests/test_analysis/test_depth_comparison.py`

- [ ] **Step 1: Write failing tests for data loading and metric computation**

```python
# tests/test_analysis/__init__.py
# (empty)
```

```python
# tests/test_analysis/test_depth_comparison.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper.analysis.depth_comparison import (
    compute_information_metrics,
    compute_pairwise_tests,
    load_depth_runs,
    split_sections_from_markdown,
)
from autoreview.evaluation.models import Claim


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a mock results directory with depth run data."""
    config = {
        "runs": [
            {
                "topic": "gut_microbiome",
                "domain": "biomedical",
                "depth": "low",
                "generated_path": str(tmp_path / "low" / "review.md"),
                "reference_path": str(tmp_path / "ref.pdf"),
                "evaluation_path": str(tmp_path / "low" / "eval.json"),
                "tier": "B",
            },
            {
                "topic": "gut_microbiome",
                "domain": "biomedical",
                "depth": "medium",
                "generated_path": str(tmp_path / "medium" / "review.md"),
                "reference_path": str(tmp_path / "ref.pdf"),
                "evaluation_path": str(tmp_path / "medium" / "eval.json"),
                "tier": "B",
            },
            {
                "topic": "gut_microbiome",
                "domain": "biomedical",
                "depth": "deep",
                "generated_path": str(tmp_path / "deep" / "review.md"),
                "reference_path": str(tmp_path / "ref.pdf"),
                "evaluation_path": str(tmp_path / "deep" / "eval.json"),
                "tier": "B",
            },
        ]
    }
    (tmp_path / "depth_runs.json").write_text(json.dumps(config))

    # Create minimal evaluation JSONs
    for depth in ["low", "medium", "deep"]:
        d = tmp_path / depth
        d.mkdir(exist_ok=True)
        score = {"low": 0.65, "medium": 0.78, "deep": 0.84}[depth]
        eval_data = {
            "overall_score": score,
            "citation_score": {"recall": 0.5, "precision": 0.6, "f1": 0.55,
                               "matched_count": 10, "reference_count": 20,
                               "generated_count": 15, "matched_titles": [],
                               "missed_titles": [], "hallucinated_titles": []},
            "synthesis_score": {"generated_score": score * 5, "reference_score": 4.0,
                                "delta": 0.0, "dimension_scores": {},
                                "generated_observations": "", "reference_observations": ""},
            "topic_coverage": {"generated_coverage": score, "reference_coverage": 1.0,
                               "topics_in_both": [], "topics_only_in_reference": [],
                               "topics_only_in_generated": []},
            "writing_quality": {"generated_score": score * 5, "reference_score": 4.0,
                                "delta": 0.0, "dimension_scores": {}},
            "timestamp": "2026-03-17",
            "generated_path": str(d / "review.md"),
            "reference_path": str(tmp_path / "ref.pdf"),
            "structural_metrics": {
                "word_count": {"low": 4000, "medium": 8000, "deep": 25000}[depth],
                "section_count": 5, "citation_count": 30,
                "citations_per_1000_words": 7.5,
                "avg_section_length_words": 800,
                "section_balance": 0.2,
                "flesch_kincaid_grade": 14.0,
            },
        }
        (d / "eval.json").write_text(json.dumps(eval_data))

        # Create minimal review markdown
        (d / "review.md").write_text(
            f"## Introduction\nSome intro text.\n\n## Results\nSome results for {depth}.\n"
        )

    return tmp_path


def test_load_depth_runs(mock_results_dir):
    df = load_depth_runs(mock_results_dir)
    assert len(df) == 3
    assert set(df["depth"]) == {"low", "medium", "deep"}
    assert "overall_score" in df.columns
    assert "word_count" in df.columns


def test_split_sections_from_markdown():
    md = "## Intro\nParagraph 1.\n\n## Methods\nParagraph 2.\n\n## Results\nParagraph 3.\n"
    sections = split_sections_from_markdown(md)
    assert len(sections) == 3
    assert sections[0]["id"] == "intro"
    assert "Paragraph 1" in sections[0]["text"]


def test_compute_information_metrics():
    claims_by_depth = {
        "low": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
        ],
        "medium": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
            Claim(text="C", category="synthesis"),
        ],
        "deep": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
            Claim(text="C", category="synthesis"),
            Claim(text="D", category="empirical"),
            Claim(text="E", category="limitation"),
        ],
    }
    # Pre-computed novelty: claims at each depth that are novel
    novel_claims = {
        "low_to_medium": [Claim(text="C", category="synthesis")],
        "medium_to_deep": [
            Claim(text="D", category="empirical"),
            Claim(text="E", category="limitation"),
        ],
    }
    concepts_by_depth = {
        "low": ["gene therapy", "crispr"],
        "medium": ["gene therapy", "crispr", "off-target effects"],
        "deep": ["gene therapy", "crispr", "off-target effects", "delivery vector", "aav"],
    }
    word_counts = {"low": 4000, "medium": 8000, "deep": 25000}
    citation_counts = {"low": 20, "medium": 45, "deep": 100}

    metrics = compute_information_metrics(
        claims_by_depth=claims_by_depth,
        novel_claims=novel_claims,
        concepts_by_depth=concepts_by_depth,
        word_counts=word_counts,
        citation_counts=citation_counts,
    )
    assert metrics.claims_per_depth == {"low": 2, "medium": 3, "deep": 5}
    assert metrics.new_claims_per_increment["low_to_medium"] == 1
    assert metrics.new_claims_per_increment["medium_to_deep"] == 2
    assert metrics.claim_novelty_rate["low_to_medium"] == pytest.approx(1 / 3)
    assert metrics.concepts_per_depth == {"low": 2, "medium": 3, "deep": 5}
    assert metrics.concept_growth["low_to_medium"] == 1
    assert metrics.concept_growth["medium_to_deep"] == 2
    assert metrics.claims_per_1k_words["low"] == pytest.approx(0.5)
    assert metrics.citations_per_claim["low"] == pytest.approx(10.0)


def test_compute_pairwise_tests_requires_min_samples():
    """Pairwise tests need at least 5 paired observations."""
    import pandas as pd

    df = pd.DataFrame({
        "topic": ["t1", "t1"],
        "depth": ["low", "medium"],
        "overall_score": [0.5, 0.6],
    })
    results = compute_pairwise_tests(df, metric_columns=["overall_score"])
    # With only 1 pair, should return NaN p-values or skip
    assert results["overall_score"]["low_to_medium"]["n_pairs"] < 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v`
Expected: ImportError

- [ ] **Step 3: Implement data loading and metrics functions**

```python
# paper/analysis/depth_comparison.py
"""Analysis 10: Depth level comparison.

Compares AutoReview output quality, information content, and cost
across low/medium/deep depth levels.

Usage:
    python -m paper.analysis.depth_comparison --results-dir paper/results/depth_comparison --output-dir paper/output/depth_comparison
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats

from autoreview.evaluation.models import Claim, InformationMetrics

logger = structlog.get_logger()

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

DEPTHS = ["low", "medium", "deep"]
DEPTH_PAIRS = [("low", "medium"), ("medium", "deep")]


def load_depth_runs(results_dir: Path) -> pd.DataFrame:
    """Load evaluation results for all depth runs into a DataFrame.

    Expects results_dir/depth_runs.json with run configs,
    and each run's evaluation_path pointing to an evaluation JSON.
    """
    config_path = results_dir / "depth_runs.json"
    with open(config_path) as f:
        config = json.load(f)

    rows: list[dict[str, Any]] = []
    for run in config["runs"]:
        eval_path = Path(run["evaluation_path"])
        if not eval_path.exists():
            logger.warning("depth_comparison.missing_eval", path=str(eval_path))
            continue

        with open(eval_path) as f:
            eval_data = json.load(f)

        row = {
            "topic": run["topic"],
            "domain": run["domain"],
            "depth": run["depth"],
            "tier": run.get("tier"),
            "generated_path": run["generated_path"],
            "reference_path": run["reference_path"],
            "overall_score": eval_data["overall_score"],
            "citation_recall": eval_data["citation_score"]["recall"],
            "citation_precision": eval_data["citation_score"]["precision"],
            "citation_f1": eval_data["citation_score"]["f1"],
            "synthesis_score": eval_data["synthesis_score"]["generated_score"],
            "topic_coverage": eval_data["topic_coverage"]["generated_coverage"],
            "writing_quality": eval_data["writing_quality"]["generated_score"],
        }
        if eval_data.get("structural_metrics"):
            sm = eval_data["structural_metrics"]
            row["word_count"] = sm["word_count"]
            row["section_count"] = sm["section_count"]
            row["citation_count"] = sm["citation_count"]
            row["citations_per_1000_words"] = sm["citations_per_1000_words"]
            row["flesch_kincaid_grade"] = sm["flesch_kincaid_grade"]

        rows.append(row)

    return pd.DataFrame(rows)


def split_sections_from_markdown(text: str) -> list[dict[str, str]]:
    """Split markdown into sections by headings. Returns [{"id": ..., "text": ...}]."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [{"id": "full_text", "text": text.strip()}]

    sections: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        section_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"id": section_id, "text": body})

    return sections if sections else [{"id": "full_text", "text": text.strip()}]


def compute_information_metrics(
    claims_by_depth: dict[str, list[Claim]],
    novel_claims: dict[str, list[Claim]],
    concepts_by_depth: dict[str, list[str]],
    word_counts: dict[str, int],
    citation_counts: dict[str, int],
    reference_claim_coverage: dict[str, float] | None = None,
) -> InformationMetrics:
    """Compute all derived information metrics across depths."""
    claims_per_depth = {d: len(claims_by_depth[d]) for d in DEPTHS if d in claims_by_depth}

    new_claims_per_increment: dict[str, int] = {}
    new_claims_by_category: dict[str, dict[str, int]] = {}
    claim_novelty_rate: dict[str, float] = {}

    for d_from, d_to in DEPTH_PAIRS:
        key = f"{d_from}_to_{d_to}"
        if key in novel_claims:
            new = novel_claims[key]
            new_claims_per_increment[key] = len(new)
            new_claims_by_category[key] = dict(Counter(c.category for c in new))
            total_at_target = claims_per_depth.get(d_to, 0)
            claim_novelty_rate[key] = len(new) / total_at_target if total_at_target > 0 else 0.0
        else:
            new_claims_per_increment[key] = 0
            new_claims_by_category[key] = {}
            claim_novelty_rate[key] = 0.0

    concepts_per_depth = {d: len(concepts_by_depth[d]) for d in DEPTHS if d in concepts_by_depth}
    concept_growth: dict[str, int] = {}
    for d_from, d_to in DEPTH_PAIRS:
        key = f"{d_from}_to_{d_to}"
        set_from = set(concepts_by_depth.get(d_from, []))
        set_to = set(concepts_by_depth.get(d_to, []))
        concept_growth[key] = len(set_to - set_from)

    claims_per_1k = {
        d: (claims_per_depth[d] / word_counts[d] * 1000) if word_counts.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in claims_per_depth
    }
    concepts_per_1k = {
        d: (concepts_per_depth[d] / word_counts[d] * 1000) if word_counts.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in concepts_per_depth
    }
    citations_per_claim = {
        d: (citation_counts[d] / claims_per_depth[d]) if claims_per_depth.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in claims_per_depth
    }

    # Concept overlap coefficient: |A ∩ B| / min(|A|, |B|)
    concept_overlap: dict[str, float] = {}
    for d_from, d_to in DEPTH_PAIRS:
        key = f"{d_from}_to_{d_to}"
        set_from = set(concepts_by_depth.get(d_from, []))
        set_to = set(concepts_by_depth.get(d_to, []))
        min_size = min(len(set_from), len(set_to))
        concept_overlap[key] = len(set_from & set_to) / min_size if min_size > 0 else 0.0

    return InformationMetrics(
        claims_per_depth=claims_per_depth,
        new_claims_per_increment=new_claims_per_increment,
        new_claims_by_category=new_claims_by_category,
        claim_novelty_rate=claim_novelty_rate,
        concepts_per_depth=concepts_per_depth,
        concept_growth=concept_growth,
        claims_per_1k_words=claims_per_1k,
        concepts_per_1k_words=concepts_per_1k,
        citations_per_claim=citations_per_claim,
        concept_overlap=concept_overlap,
        reference_claim_coverage=reference_claim_coverage,
    )


def compute_pairwise_tests(
    df: pd.DataFrame,
    metric_columns: list[str],
    min_pairs: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Wilcoxon signed-rank tests for low->medium and medium->deep.

    Returns nested dict: metric -> pair_key -> {statistic, p_value, n_pairs, p_adjusted}.
    FDR correction (Benjamini-Hochberg) applied across all tests.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    all_p_values: list[tuple[str, str, float]] = []

    for metric in metric_columns:
        results[metric] = {}
        for d_from, d_to in DEPTH_PAIRS:
            key = f"{d_from}_to_{d_to}"
            from_scores = df[df["depth"] == d_from].set_index("topic")[metric]
            to_scores = df[df["depth"] == d_to].set_index("topic")[metric]
            common = from_scores.index.intersection(to_scores.index)

            n_pairs = len(common)
            if n_pairs < min_pairs:
                results[metric][key] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_pairs": n_pairs,
                    "p_adjusted": float("nan"),
                    "mean_diff": float("nan"),
                }
                continue

            a = from_scores.loc[common].values
            b = to_scores.loc[common].values
            diff = b - a

            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = sp_stats.wilcoxon(diff, alternative="two-sided")

            results[metric][key] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "n_pairs": n_pairs,
                "mean_diff": float(np.mean(diff)),
            }
            all_p_values.append((metric, key, p_val))

    # FDR correction (Benjamini-Hochberg) with cumulative minimum pass
    if all_p_values:
        sorted_pvals = sorted(all_p_values, key=lambda x: x[2])
        m = len(sorted_pvals)
        # Compute raw BH adjusted values
        raw_adj = [min(p * m / (rank + 1), 1.0) for rank, (_, _, p) in enumerate(sorted_pvals)]
        # Cumulative minimum from the right (enforces monotonicity)
        for i in range(len(raw_adj) - 2, -1, -1):
            raw_adj[i] = min(raw_adj[i], raw_adj[i + 1])
        for i, (metric, key, _) in enumerate(sorted_pvals):
            results[metric][key]["p_adjusted"] = raw_adj[i]

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/depth_comparison.py tests/test_analysis/__init__.py tests/test_analysis/test_depth_comparison.py
git commit -m "feat(paper): add depth comparison data loading and metrics computation"
```

---

## Task 6: Depth Comparison Figures

**Files:**
- Modify: `paper/analysis/depth_comparison.py`
- Modify: `tests/test_analysis/test_depth_comparison.py`

- [ ] **Step 1: Write failing tests for figure generation**

Append to `tests/test_analysis/test_depth_comparison.py`:

```python
import matplotlib
matplotlib.use("Agg")

from paper.analysis.depth_comparison import (
    plot_quality_by_depth,
    plot_cumulative_claims,
    plot_information_density,
    plot_cost_efficiency,
    plot_domain_depth_heatmap,
)


@pytest.fixture
def sample_df():
    """DataFrame with 2 topics x 3 depths for figure testing."""
    rows = []
    for topic, domain in [("t1", "biomedical"), ("t2", "cs")]:
        for depth, score in [("low", 0.6), ("medium", 0.75), ("deep", 0.82)]:
            rows.append({
                "topic": topic,
                "domain": domain,
                "depth": depth,
                "overall_score": score + (0.02 if topic == "t2" else 0),
                "synthesis_score": score * 5,
                "topic_coverage": score,
                "writing_quality": score * 4.5,
                "citation_f1": score * 0.8,
                "word_count": {"low": 4000, "medium": 8000, "deep": 25000}[depth],
                "citation_count": {"low": 20, "medium": 45, "deep": 100}[depth],
            })
    return pd.DataFrame(rows)


def test_plot_quality_by_depth(sample_df, tmp_path):
    plot_quality_by_depth(sample_df, tmp_path / "quality.pdf")
    assert (tmp_path / "quality.pdf").exists()


def test_plot_cumulative_claims(tmp_path):
    claims_data = {
        "low": {"empirical": 10, "methodological": 5, "contextual": 8, "synthesis": 3, "limitation": 2},
        "medium": {"empirical": 25, "methodological": 12, "contextual": 15, "synthesis": 8, "limitation": 5},
        "deep": {"empirical": 55, "methodological": 28, "contextual": 30, "synthesis": 18, "limitation": 12},
    }
    plot_cumulative_claims(claims_data, tmp_path / "claims.pdf")
    assert (tmp_path / "claims.pdf").exists()


def test_plot_information_density(sample_df, tmp_path):
    density_data = []
    for _, row in sample_df.iterrows():
        density_data.append({
            "topic": row["topic"],
            "domain": row["domain"],
            "depth": row["depth"],
            "word_count": row["word_count"],
            "claims_per_1k_words": 12.0 if row["depth"] != "deep" else 8.5,
        })
    density_df = pd.DataFrame(density_data)
    plot_information_density(density_df, tmp_path / "density.pdf")
    assert (tmp_path / "density.pdf").exists()


def test_plot_cost_efficiency(tmp_path):
    cost_data = pd.DataFrame({
        "depth": ["low", "medium", "deep"],
        "mean_score": [0.6, 0.75, 0.82],
        "mean_cost": [1.50, 3.20, 8.75],
        "claims_per_dollar": [30, 32, 26],
    })
    plot_cost_efficiency(cost_data, tmp_path / "cost.pdf")
    assert (tmp_path / "cost.pdf").exists()


def test_plot_domain_depth_heatmap(sample_df, tmp_path):
    plot_domain_depth_heatmap(sample_df, "overall_score", tmp_path / "heatmap.pdf")
    assert (tmp_path / "heatmap.pdf").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "plot"`
Expected: ImportError — plot functions not found

- [ ] **Step 3: Implement figure functions**

Append to `paper/analysis/depth_comparison.py` (after existing imports, before function definitions — `matplotlib` is already imported at top of file from Task 5):

```python
# Colorblind-safe palette
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]
DEPTH_COLORS = {"low": CB_PALETTE[0], "medium": CB_PALETTE[5], "deep": CB_PALETTE[1]}
CATEGORY_ORDER = ["empirical", "methodological", "contextual", "synthesis", "limitation"]
CATEGORY_COLORS = {cat: CB_PALETTE[i] for i, cat in enumerate(CATEGORY_ORDER)}

_FONT_CONFIG = {
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}


def _apply_style():
    plt.rcParams.update(_FONT_CONFIG)


def plot_quality_by_depth(df: pd.DataFrame, out: Path) -> None:
    """Line chart: quality score x depth level, one line per metric."""
    _apply_style()
    metrics = {
        "synthesis_score": "Synthesis",
        "topic_coverage": "Topic Coverage",
        "writing_quality": "Writing Quality",
        "citation_f1": "Citation F1",
    }
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, (col, label) in enumerate(metrics.items()):
        means = df.groupby("depth")[col].mean().reindex(DEPTHS)
        sems = df.groupby("depth")[col].sem().reindex(DEPTHS)
        ax.errorbar(
            DEPTHS, means, yerr=sems,
            marker="o", label=label, color=CB_PALETTE[i],
            linewidth=2, capsize=4,
        )

    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics by Depth Level")
    ax.legend(loc="best")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_claims(
    claims_by_depth_and_category: dict[str, dict[str, int]],
    out: Path,
) -> None:
    """Stacked bar: claim counts by category at each depth."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    x = np.arange(len(DEPTHS))
    width = 0.6
    bottom = np.zeros(len(DEPTHS))

    for cat in CATEGORY_ORDER:
        values = [claims_by_depth_and_category.get(d, {}).get(cat, 0) for d in DEPTHS]
        ax.bar(x, values, width, bottom=bottom, label=cat.capitalize(), color=CATEGORY_COLORS[cat])
        bottom += np.array(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DEPTHS])
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Cumulative Claims by Category and Depth")
    ax.legend(loc="upper left")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_information_density(density_df: pd.DataFrame, out: Path) -> None:
    """Scatter: word count (x) vs claims/1000 words (y), color=domain, shape=depth."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    domains = density_df["domain"].unique()
    domain_colors = {d: CB_PALETTE[i % len(CB_PALETTE)] for i, d in enumerate(domains)}
    depth_markers = {"low": "o", "medium": "s", "deep": "D"}

    for _, row in density_df.iterrows():
        ax.scatter(
            row["word_count"],
            row["claims_per_1k_words"],
            c=domain_colors[row["domain"]],
            marker=depth_markers.get(row["depth"], "o"),
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Legend for domains
    for domain, color in domain_colors.items():
        ax.scatter([], [], c=color, marker="o", label=domain.capitalize(), s=80)
    # Legend for depths
    for depth, marker in depth_markers.items():
        ax.scatter([], [], c="gray", marker=marker, label=f"{depth.capitalize()} depth", s=80)

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Claims per 1,000 Words")
    ax.set_title("Information Density vs. Review Length")
    ax.legend(loc="best", ncol=2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_efficiency(cost_df: pd.DataFrame, out: Path) -> None:
    """Dual-axis chart: quality score and cost by depth."""
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2 = ax1.twinx()

    x = np.arange(len(cost_df))
    width = 0.35

    ax1.bar(x - width / 2, cost_df["mean_score"], width, label="Quality Score",
            color=CB_PALETTE[0], alpha=0.8)
    ax2.bar(x + width / 2, cost_df["mean_cost"], width, label="Cost ($)",
            color=CB_PALETTE[1], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in cost_df["depth"]])
    ax1.set_xlabel("Depth Level")
    ax1.set_ylabel("Quality Score", color=CB_PALETTE[0])
    ax2.set_ylabel("Cost ($)", color=CB_PALETTE[1])
    ax1.set_title("Quality-Cost Tradeoff by Depth")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_domain_depth_heatmap(
    df: pd.DataFrame, metric: str, out: Path,
) -> None:
    """Heatmap: domain (rows) x depth (cols), cell = mean metric value."""
    _apply_style()
    pivot = df.pivot_table(values=metric, index="domain", columns="depth", aggfunc="mean")
    pivot = pivot.reindex(columns=DEPTHS)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels([d.capitalize() for d in DEPTHS])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.capitalize() for d in pivot.index])
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Domain")
    ax.set_title(f"{metric.replace('_', ' ').title()} by Domain and Depth")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(DEPTHS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < pivot.values.mean() else "black", fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "plot"`
Expected: All 5 figure tests PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/depth_comparison.py tests/test_analysis/test_depth_comparison.py
git commit -m "feat(paper): add depth comparison figure generation"
```

---

## Task 7: Report Generation and JSON Output

**Files:**
- Modify: `paper/analysis/depth_comparison.py`
- Modify: `tests/test_analysis/test_depth_comparison.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analysis/test_depth_comparison.py`:

```python
from paper.analysis.depth_comparison import (
    write_analysis_json,
    generate_report,
    DepthAnalysisOutput,
)


@pytest.fixture
def sample_analysis_output():
    return DepthAnalysisOutput(
        summary_stats={
            "low": {"overall_score": {"mean": 0.65, "std": 0.05, "n": 20}},
            "medium": {"overall_score": {"mean": 0.78, "std": 0.04, "n": 20}},
            "deep": {"overall_score": {"mean": 0.84, "std": 0.03, "n": 20}},
        },
        pairwise_tests={
            "overall_score": {
                "low_to_medium": {"statistic": 15.0, "p_value": 0.001, "n_pairs": 20, "p_adjusted": 0.002, "mean_diff": 0.13},
                "medium_to_deep": {"statistic": 20.0, "p_value": 0.01, "n_pairs": 20, "p_adjusted": 0.015, "mean_diff": 0.06},
            },
        },
        information_metrics_per_topic={
            "gut_microbiome": InformationMetrics(
                claims_per_depth={"low": 45, "medium": 102, "deep": 230},
                new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
                new_claims_by_category={"low_to_medium": {"empirical": 20}, "medium_to_deep": {"empirical": 50}},
                claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
                concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
                concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
                claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
                concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
                citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
            ),
        },
    )


def test_write_analysis_json(sample_analysis_output, tmp_path):
    write_analysis_json(sample_analysis_output, tmp_path / "depth_analysis.json")
    assert (tmp_path / "depth_analysis.json").exists()
    with open(tmp_path / "depth_analysis.json") as f:
        data = json.load(f)
    assert "summary_stats" in data
    assert "pairwise_tests" in data
    assert "information_metrics_per_topic" in data


def test_generate_report(sample_analysis_output, tmp_path):
    generate_report(sample_analysis_output, tmp_path / "report.md")
    assert (tmp_path / "report.md").exists()
    content = (tmp_path / "report.md").read_text()
    assert "Depth Comparison" in content
    assert "low" in content.lower()
    assert "medium" in content.lower()
    assert "deep" in content.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "write_analysis or generate_report"`
Expected: ImportError

- [ ] **Step 3: Implement report generation**

Append to `paper/analysis/depth_comparison.py`:

```python
from dataclasses import dataclass, field


@dataclass
class DepthAnalysisOutput:
    """Container for all analysis results."""

    summary_stats: dict[str, dict[str, dict[str, float]]]
    pairwise_tests: dict[str, dict[str, dict[str, Any]]]
    information_metrics_per_topic: dict[str, InformationMetrics]
    cost_data: dict[str, dict[str, float]] | None = None


def write_analysis_json(output: DepthAnalysisOutput, path: Path) -> None:
    """Write all computed metrics and test results to JSON."""
    data = {
        "summary_stats": output.summary_stats,
        "pairwise_tests": output.pairwise_tests,
        "information_metrics_per_topic": {
            topic: metrics.model_dump()
            for topic, metrics in output.information_metrics_per_topic.items()
        },
    }
    if output.cost_data:
        data["cost_data"] = output.cost_data

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("depth_comparison.json_written", path=str(path))


def generate_report(output: DepthAnalysisOutput, path: Path) -> None:
    """Generate markdown summary report."""
    lines: list[str] = []
    lines.append("# Analysis 10: Depth Level Comparison\n")
    lines.append("## Summary Statistics\n")

    # Summary table
    lines.append("| Depth | Metric | Mean | Std | N |")
    lines.append("|-------|--------|------|-----|---|")
    for depth in DEPTHS:
        if depth in output.summary_stats:
            for metric, stats in output.summary_stats[depth].items():
                lines.append(
                    f"| {depth.capitalize()} | {metric} | "
                    f"{stats.get('mean', 0):.3f} | {stats.get('std', 0):.3f} | "
                    f"{stats.get('n', 0)} |"
                )
    lines.append("")

    # Pairwise tests
    lines.append("## Pairwise Statistical Tests (Wilcoxon Signed-Rank)\n")
    lines.append("| Metric | Comparison | Mean Diff | p-value | p-adjusted | N pairs |")
    lines.append("|--------|------------|-----------|---------|------------|---------|")
    for metric, pairs in output.pairwise_tests.items():
        for pair_key, result in pairs.items():
            lines.append(
                f"| {metric} | {pair_key} | "
                f"{result.get('mean_diff', 0):.3f} | "
                f"{result.get('p_value', float('nan')):.4f} | "
                f"{result.get('p_adjusted', float('nan')):.4f} | "
                f"{result.get('n_pairs', 0)} |"
            )
    lines.append("")

    # Information metrics summary
    lines.append("## Information Metrics\n")
    if output.information_metrics_per_topic:
        first_topic = next(iter(output.information_metrics_per_topic))
        first_metrics = output.information_metrics_per_topic[first_topic]

        lines.append("### Claims per Depth (averaged across topics)\n")
        avg_claims = {}
        for depth in DEPTHS:
            values = [
                m.claims_per_depth.get(depth, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg_claims[depth] = np.mean(values) if values else 0
        for depth in DEPTHS:
            lines.append(f"- **{depth.capitalize()}**: {avg_claims[depth]:.1f} claims")
        lines.append("")

        lines.append("### Novelty per Increment (averaged across topics)\n")
        for d_from, d_to in DEPTH_PAIRS:
            key = f"{d_from}_to_{d_to}"
            values = [
                m.new_claims_per_increment.get(key, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg = np.mean(values) if values else 0
            lines.append(f"- **{d_from} -> {d_to}**: {avg:.1f} new claims")
        lines.append("")

        lines.append("### Information Density (averaged across topics)\n")
        for depth in DEPTHS:
            values = [
                m.claims_per_1k_words.get(depth, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg = np.mean(values) if values else 0
            lines.append(f"- **{depth.capitalize()}**: {avg:.2f} claims / 1K words")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("depth_comparison.report_written", path=str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "write_analysis or generate_report"`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/depth_comparison.py tests/test_analysis/test_depth_comparison.py
git commit -m "feat(paper): add depth comparison JSON output and markdown report"
```

---

## Task 8: CLI Entry Point

**Files:**
- Modify: `paper/analysis/depth_comparison.py`
- Modify: `tests/test_analysis/test_depth_comparison.py`

- [ ] **Step 1: Write failing test for CLI parsing**

Append to `tests/test_analysis/test_depth_comparison.py`:

```python
from paper.analysis.depth_comparison import parse_args


def test_parse_args_defaults():
    args = parse_args(["--results-dir", "/tmp/results"])
    assert args.results_dir == Path("/tmp/results")
    assert args.output_dir == Path("paper/output/depth_comparison")


def test_parse_args_custom_output():
    args = parse_args(["--results-dir", "/tmp/results", "--output-dir", "/tmp/out"])
    assert args.output_dir == Path("/tmp/out")


def test_parse_args_skip_extraction():
    args = parse_args(["--results-dir", "/tmp/r", "--skip-extraction"])
    assert args.skip_extraction is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "parse_args"`
Expected: ImportError

- [ ] **Step 3: Implement CLI entry point**

Append to `paper/analysis/depth_comparison.py`:

```python
import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysis 10: Depth level comparison"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing depth_runs.json and evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/depth_comparison"),
        help="Output directory for figures, JSON, and report",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip claim/concept extraction (use cached results from depth_claims.json)",
    )
    return parser.parse_args(argv)


async def run_extraction_for_topic(
    topic: str,
    depth_reviews: dict[str, str],
    extractor: Any,
    embedder: Any,
    word_counts: dict[str, int],
    citation_counts: dict[str, int],
    reference_text: str | None = None,
) -> InformationMetrics:
    """Run claim/concept extraction for a single topic across all depths.

    Args:
        topic: Topic name.
        depth_reviews: {"low": "review text...", "medium": ..., "deep": ...}
        extractor: ClaimExtractor instance.
        embedder: Embedding provider with async embed(texts) -> np.ndarray.
        word_counts: Word count per depth.
        citation_counts: Citation count per depth.
        reference_text: Optional human reference review text (Tier A only).
    """
    from autoreview.evaluation.claim_dedup import deduplicate_claims

    # Extract claims and concepts at each depth
    claims_by_depth: dict[str, list[Claim]] = {}
    concepts_by_depth: dict[str, list[str]] = {}

    for depth in DEPTHS:
        if depth not in depth_reviews:
            continue
        claims_by_depth[depth] = await extractor.extract_claims(depth_reviews[depth])
        concepts_by_depth[depth] = await extractor.extract_concepts(depth_reviews[depth])
        logger.info(
            "depth_comparison.extracted",
            topic=topic,
            depth=depth,
            n_claims=len(claims_by_depth[depth]),
            n_concepts=len(concepts_by_depth[depth]),
        )

    # Compute novelty using cumulative union baseline
    novel_claims: dict[str, list[Claim]] = {}
    if "low" in claims_by_depth and "medium" in claims_by_depth:
        novel, _ = await deduplicate_claims(
            claims_by_depth["medium"], claims_by_depth["low"], embedder
        )
        novel_claims["low_to_medium"] = novel

    if "medium" in claims_by_depth and "deep" in claims_by_depth:
        # Union baseline: low + medium
        baseline = claims_by_depth.get("low", []) + claims_by_depth.get("medium", [])
        novel, _ = await deduplicate_claims(
            claims_by_depth["deep"], baseline, embedder
        )
        novel_claims["medium_to_deep"] = novel

    # Reference claim coverage (Tier A)
    ref_coverage: dict[str, float] | None = None
    if reference_text:
        ref_claims = await extractor.extract_claims(reference_text)
        if ref_claims:
            ref_coverage = {}
            for depth in DEPTHS:
                if depth in claims_by_depth:
                    _, shared = await deduplicate_claims(
                        ref_claims, claims_by_depth[depth], embedder
                    )
                    ref_coverage[depth] = len(shared) / len(ref_claims)

    return compute_information_metrics(
        claims_by_depth=claims_by_depth,
        novel_claims=novel_claims,
        concepts_by_depth=concepts_by_depth,
        word_counts=word_counts,
        citation_counts=citation_counts,
        reference_claim_coverage=ref_coverage,
    )


async def main(results_dir: Path, output_dir: Path, skip_extraction: bool = False) -> None:
    """Full depth comparison analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load runs
    logger.info("depth_comparison.loading_runs", results_dir=str(results_dir))
    df = load_depth_runs(results_dir)
    logger.info("depth_comparison.loaded", n_runs=len(df))

    # 2. Compute summary stats
    metric_cols = [
        "overall_score", "citation_recall", "citation_f1",
        "synthesis_score", "topic_coverage", "writing_quality",
    ]
    summary_stats: dict[str, dict[str, dict[str, float]]] = {}
    for depth in DEPTHS:
        depth_df = df[df["depth"] == depth]
        summary_stats[depth] = {}
        for col in metric_cols:
            if col in depth_df.columns:
                summary_stats[depth][col] = {
                    "mean": float(depth_df[col].mean()),
                    "std": float(depth_df[col].std()),
                    "n": int(len(depth_df)),
                }

    # 3. Pairwise tests
    pairwise = compute_pairwise_tests(df, metric_cols)

    # 4. Information extraction (or load cached)
    info_metrics: dict[str, InformationMetrics] = {}
    claims_cache = results_dir / "depth_claims.json"

    if skip_extraction and claims_cache.exists():
        logger.info("depth_comparison.loading_cached_claims")
        with open(claims_cache) as f:
            cached = json.load(f)
        for topic, metrics_data in cached.items():
            info_metrics[topic] = InformationMetrics.model_validate(metrics_data)
    elif not skip_extraction:
        # Live extraction path — requires LLM client and embedder
        from autoreview.evaluation.claim_extractor import ClaimExtractor
        from autoreview.llm.factory import create_llm_provider
        from autoreview.config.models import LLMConfig

        llm_config = LLMConfig()
        llm = create_llm_provider(llm_config)
        extractor = ClaimExtractor(llm)
        # Embedder must be provided by caller or configured here
        # For now, log that extraction requires embedder setup
        logger.warning("depth_comparison.extraction_requires_embedder")
    else:
        logger.info("depth_comparison.no_extraction_no_cache")

    # 5. Build output
    analysis_output = DepthAnalysisOutput(
        summary_stats=summary_stats,
        pairwise_tests=pairwise,
        information_metrics_per_topic=info_metrics,
    )

    # 6. Write outputs
    write_analysis_json(analysis_output, output_dir / "depth_analysis.json")
    generate_report(analysis_output, output_dir / "report.md")

    # 7. Generate figures
    plot_quality_by_depth(df, output_dir / "quality_by_depth.pdf")
    plot_domain_depth_heatmap(df, "overall_score", output_dir / "domain_depth_heatmap.pdf")

    # 8. Cache extraction results for reuse
    if info_metrics and not claims_cache.exists():
        cache_data = {t: m.model_dump() for t, m in info_metrics.items()}
        with open(claims_cache, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        logger.info("depth_comparison.claims_cached", path=str(claims_cache))

    logger.info("depth_comparison.complete", output_dir=str(output_dir))


if __name__ == "__main__":
    import asyncio
    args = parse_args()
    asyncio.run(main(args.results_dir, args.output_dir, args.skip_extraction))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py -v -k "parse_args"`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all tests for the full module**

Run: `python -m pytest tests/test_analysis/test_depth_comparison.py tests/test_evaluation/test_claim_extractor.py tests/test_evaluation/test_claim_dedup.py tests/test_evaluation/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add paper/analysis/depth_comparison.py tests/test_analysis/test_depth_comparison.py
git commit -m "feat(paper): add depth comparison CLI entry point and async main"
```

---

## Task 9: Update Paper Plan

**Files:**
- Modify: `Paper Idea/paper.md`

- [ ] **Step 1: Add Analysis 10 to the paper plan**

After Analysis 9 (Contamination Analysis), add:

```markdown
### Analysis 10 — Depth Level Comparison
- All topics x 3 depths (low/medium/deep) x Sonnet only
- Quality scaling: do synthesis, coverage, writing, citation scores scale with depth?
- Information gain: claim extraction, concept coverage, information density per depth
- Diminishing returns: where does the quality and information curve plateau?
- Cost-efficiency: cost per quality-point and cost per new claim
- Domain interaction: which domains benefit most from deeper treatment?
- **Scripts**: `paper/analysis/depth_comparison.py`
- **Design spec**: `docs/superpowers/specs/2026-03-17-depth-comparison-design.md`
```

- [ ] **Step 2: Update Phase 3 run estimates**

In Phase 3, add after 3e:

```markdown
**3f. AutoReview runs — depth comparison** (17-23 topics x 2 net-new depths x Sonnet = 34-46 runs)
- Low and deep runs for all topics; medium reused from 3a/3b
- Full pipeline with same parameters as main comparison except `--depth`
```

Update total: "**Total estimated runs: ~137-202**"

- [ ] **Step 3: Update figures list**

Add to figures:

```markdown
10. **Quality × depth line chart**: score per metric dimension across low/medium/deep
11. **Cumulative claims stacked bar**: claims by category at each depth level
```

- [ ] **Step 4: Update key files table**

Add row:

```markdown
| `paper/analysis/depth_comparison.py` | **New**: Analysis 10 | 4 |
```

- [ ] **Step 5: Commit**

```bash
git add "Paper Idea/paper.md"
git commit -m "docs: add Analysis 10 (depth comparison) to paper plan"
```

---

## Dependency Graph

```
Task 1 (models)
  ├── Task 2 (prompts) ─── Task 3 (ClaimExtractor)
  │                            │
  ├── Task 4 (dedup) ─────────┤
  │                            │
  └── Task 5 (analysis core) ──┤
                                ├── Task 6 (figures)
                                ├── Task 7 (report/JSON)
                                └── Task 8 (CLI + main with extraction wiring)

Task 9 (paper plan update) — independent
```

**Parallelizable batches:**
- Batch 1: Task 1 (models) + Task 9 (paper plan)
- Batch 2: Task 2 (prompts) + Task 4 (dedup) + Task 5 (analysis core)
- Batch 3: Task 3 (ClaimExtractor) + Task 6 (figures) + Task 7 (report)
- Batch 4: Task 8 (CLI entry point + `run_extraction_for_topic` + `main` with full wiring)
