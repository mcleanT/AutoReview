"""Tests for evidence-based section revision."""

from __future__ import annotations

from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMResponse
from autoreview.writing.section_writer import SectionDraft, SectionWriter


class MockRevisionLLM:
    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3):
        return LLMResponse(
            content=(
                "Senescent cells secrete SASP factors including IL-6 and IL-8 [@p1]. "
                "Recent evidence confirms this in hepatocytes [@p2] and renal tubular cells [@p3]."
            ),
            input_tokens=500,
            output_tokens=120,
        )


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[
                Finding(
                    claim="Hepatocyte senescence drives liver fibrosis via IL-6",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id="p2",
                )
            ],
            methods_summary="Mouse model + human biopsy",
            limitations="Animal model",
        ),
        "p3": PaperExtraction(
            paper_id="p3",
            key_findings=[
                Finding(
                    claim="Renal senescence promotes CKD progression",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id="p3",
                )
            ],
            methods_summary="Cohort study",
            limitations="Observational",
        ),
    }


class TestReviseSection:
    async def test_revise_returns_section_draft(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        assert isinstance(draft, SectionDraft)
        assert draft.section_id == "1"
        assert len(draft.text) > 0

    async def test_revise_includes_new_citations(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        assert "[@p2]" in draft.text or "[@p3]" in draft.text

    async def test_revise_tracks_citations_used(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        # The mock LLM returns text with [@p1], [@p2], [@p3] — all should be tracked
        assert "p1" in draft.citations_used or "p2" in draft.citations_used
