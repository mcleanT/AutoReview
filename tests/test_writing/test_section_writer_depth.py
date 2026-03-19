from unittest.mock import AsyncMock

import pytest

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.config.models import DepthLevel
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMResponse
from autoreview.writing.citation_selector import PaperCitation, SectionCitationPlan
from autoreview.writing.section_writer import SectionWriter, _format_extractions_tiered


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Generated section text with [@p1] citations.",
            input_tokens=100,
            output_tokens=200,
        )
    )
    return llm


@pytest.fixture
def section():
    return OutlineSection(
        id="s1",
        title="Deep Learning",
        description="Overview",
        paper_ids=["p1"],
        theme_refs=[],
        estimated_word_count=1200,
    )


@pytest.fixture
def outline(section):
    return ReviewOutline(title="Test", sections=[section])


@pytest.fixture
def evidence_map():
    return EvidenceMap(themes=[], consensus_claims=[], contradictions=[], gaps=[])


@pytest.mark.asyncio
async def test_section_writer_passes_depth_to_prompt(mock_llm, section, outline, evidence_map):
    writer = SectionWriter(mock_llm)
    await writer.write_section(
        section=section,
        outline=outline,
        extractions={},
        evidence_map=evidence_map,
        depth=DepthLevel.DEEP,
    )
    call_kwargs = mock_llm.generate.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 16384


@pytest.mark.asyncio
async def test_section_writer_medium_no_max_tokens_override(
    mock_llm, section, outline, evidence_map
):
    writer = SectionWriter(mock_llm)
    await writer.write_section(
        section=section,
        outline=outline,
        extractions={},
        evidence_map=evidence_map,
        depth=DepthLevel.MEDIUM,
    )
    call_kwargs = mock_llm.generate.call_args
    # Medium has no override, max_tokens should NOT be in kwargs
    assert "max_tokens" not in call_kwargs.kwargs


def test_format_extractions_with_citation_plan():
    """When a CitationPlan is provided, extractions should be tiered."""
    plan = SectionCitationPlan(
        section_id="sec_3",
        citation_budget=10,
        primary_papers=[
            PaperCitation(
                paper_id="p1",
                tier="primary",
                priority_score=0.9,
                citation_guidance="Discuss method",
            )
        ],
        supporting_papers=[
            PaperCitation(
                paper_id="p2", tier="supporting", priority_score=0.6, citation_guidance=""
            )
        ],
        contextual_papers=[],
        coverage_notes=[],
    )
    extractions = {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(
                    claim="Neural networks improve accuracy",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id="p1",
                )
            ],
            methods_summary="Deep learning",
            limitations="Small dataset",
            relationships=[],
        ),
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[
                Finding(
                    claim="Transformer models generalise well",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id="p2",
                )
            ],
            methods_summary="Transformers",
            limitations="Compute cost",
            relationships=[],
        ),
    }
    result = _format_extractions_tiered(plan, extractions)
    assert "PRIMARY" in result
    assert "SUPPORTING" in result
    assert "[@p1]" in result
    assert "[@p2]" in result
    assert "Citation Budget: ~10" in result
    assert "Discuss method" in result
