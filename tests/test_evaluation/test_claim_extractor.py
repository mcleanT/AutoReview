import pytest

from autoreview.evaluation.claim_extractor import ClaimExtractor
from autoreview.evaluation.models import Claim
from autoreview.llm.prompts.claim_extraction import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    ClaimExtractionItem,
    ClaimExtractionResult,
    ConceptExtractionResult,
    build_claim_extraction_prompt,
    build_concept_extraction_prompt,
)
from autoreview.llm.provider import LLMStructuredResponse


def test_claim_extraction_prompt_builds():
    prompt = build_claim_extraction_prompt("## Introduction\nSome text about biology.")
    assert "Introduction" in prompt
    assert "Some text about biology" in prompt
    assert "empirical" in prompt
    assert "methodological" in prompt


def test_claim_extraction_prompt_truncates():
    long_text = "word " * 20000
    prompt = build_claim_extraction_prompt(long_text, max_chars=1000)
    assert len(prompt) < 5000


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


# ---------------------------------------------------------------------------
# ClaimExtractor tests
# ---------------------------------------------------------------------------


class MockClaimLLM:
    """Mock LLM that returns fixed claim/concept extraction results."""

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        if response_model == ClaimExtractionResult:
            return LLMStructuredResponse(
                parsed=ClaimExtractionResult(
                    claims=[
                        ClaimExtractionItem(
                            text="GLP-1 agonists reduce HbA1c by 1.5%", category="empirical"
                        ),
                        ClaimExtractionItem(
                            text="Metformin remains first-line therapy", category="contextual"
                        ),
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


@pytest.mark.asyncio
async def test_extract_claims_from_sections(claim_extractor):
    claims = await claim_extractor.extract_claims(SAMPLE_REVIEW)
    assert len(claims) > 0
    assert all(isinstance(c, Claim) for c in claims)
    assert all(c.section_id is not None for c in claims)


@pytest.mark.asyncio
async def test_extract_claims_assigns_section_ids(claim_extractor):
    claims = await claim_extractor.extract_claims(SAMPLE_REVIEW)
    section_ids = {c.section_id for c in claims}
    assert len(section_ids) >= 2


@pytest.mark.asyncio
async def test_extract_concepts(claim_extractor):
    concepts = await claim_extractor.extract_concepts(SAMPLE_REVIEW)
    assert len(concepts) > 0
    assert all(isinstance(c, str) for c in concepts)
    assert all(c == c.lower() for c in concepts)


@pytest.mark.asyncio
async def test_extract_concepts_deduplicates(claim_extractor):
    concepts = await claim_extractor.extract_concepts(SAMPLE_REVIEW)
    assert len(concepts) == len(set(concepts))


@pytest.mark.asyncio
async def test_split_sections():
    extractor = ClaimExtractor(MockClaimLLM())
    sections = extractor.split_into_sections(SAMPLE_REVIEW)
    assert len(sections) >= 3
    assert all("id" in s and "text" in s for s in sections)


@pytest.mark.asyncio
async def test_single_section_fallback(claim_extractor):
    """Text without headings is treated as a single section."""
    claims = await claim_extractor.extract_claims("Just a plain paragraph of text.")
    assert len(claims) > 0
    assert all(c.section_id == "full_text" for c in claims)
