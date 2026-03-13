from __future__ import annotations

from autoreview.analysis.evidence_map import (
    ConsensusClaim,
    Contradiction,
    EvidenceMap,
    Theme,
)
from autoreview.critique.holistic_critic import HolisticCritic, holistic_critique_loop
from autoreview.critique.models import (
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    CritiqueTarget,
)
from autoreview.critique.outline_critic import OutlineCritic, _outline_to_text
from autoreview.critique.revision import should_continue_revision
from autoreview.critique.section_critic import SectionCritic
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.writing.assembler import DraftAssembler
from autoreview.writing.outliner import OutlineGenerator, _format_evidence_summary
from autoreview.writing.section_writer import SectionDraft, SectionWriter, _extract_citations


def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="1",
                title="Introduction",
                description="Overview",
                paper_ids=["p1"],
                estimated_word_count=500,
            ),
            OutlineSection(
                id="2",
                title="Methods",
                description="Review methodology",
                paper_ids=[],
                estimated_word_count=300,
            ),
            OutlineSection(
                id="3",
                title="Results",
                description="Key findings",
                paper_ids=["p1", "p2"],
                estimated_word_count=1000,
                subsections=[
                    OutlineSection(
                        id="3.1",
                        title="Theme A",
                        description="First theme",
                        paper_ids=["p1"],
                        estimated_word_count=500,
                    ),
                ],
            ),
        ],
    )


def _make_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        themes=[
            Theme(name="Theme A", description="First theme", paper_ids=["p1", "p2"]),
        ],
        consensus_claims=[
            ConsensusClaim(
                claim="Finding X",
                supporting_paper_ids=["p1", "p2"],
                strength="moderate",
                evidence_count=2,
            ),
        ],
        contradictions=[
            Contradiction(
                claim_a="A",
                claim_b="B",
                paper_ids_a=["p1"],
                paper_ids_b=["p2"],
                theme="Theme A",
            ),
        ],
    )


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(claim="Claim A", evidence_strength=EvidenceStrength.STRONG, paper_id="p1")
            ],
            methods_summary="Methods A",
            limitations="Limits A",
        ),
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[
                Finding(claim="Claim B", evidence_strength=EvidenceStrength.MODERATE, paper_id="p2")
            ],
            methods_summary="Methods B",
            limitations="Limits B",
        ),
    }


class MockWritingLLM:
    """Mock LLM for writing/critique tests."""

    def __init__(self, critique_score: float = 0.85) -> None:
        self.critique_score = critique_score
        self.calls: list[str] = []

    async def generate(
        self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None
    ):
        self.calls.append("generate")
        return LLMResponse(
            content="This section synthesizes findings from multiple studies [@p1] [@p2].",
            input_tokens=500,
            output_tokens=300,
        )

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        self.calls.append("generate_structured")
        if response_model == ReviewOutline:
            return LLMStructuredResponse(
                parsed=_make_outline(),
                input_tokens=800,
                output_tokens=400,
            )
        elif response_model == CritiqueReport:
            return LLMStructuredResponse(
                parsed=CritiqueReport(
                    target=CritiqueTarget.OUTLINE,
                    passed=self.critique_score >= 0.80,
                    overall_score=self.critique_score,
                    dimension_scores={"coverage": 0.85, "synthesis": 0.80},
                    issues=[
                        CritiqueIssue(
                            severity=CritiqueSeverity.MINOR,
                            location="Section 3",
                            description="Could use better transitions",
                        ),
                    ]
                    if self.critique_score < 0.80
                    else [],
                ),
                input_tokens=600,
                output_tokens=300,
            )
        raise ValueError(f"Unexpected: {response_model}")


class TestOutline:
    def test_outline_flatten(self):
        outline = _make_outline()
        flat = outline.flatten()
        ids = [s.id for s in flat]
        assert ids == ["1", "2", "3", "3.1"]

    def test_outline_get_section(self):
        outline = _make_outline()
        s = outline.get_section("3.1")
        assert s is not None
        assert s.title == "Theme A"
        assert outline.get_section("99") is None

    def test_outline_to_text(self):
        outline = _make_outline()
        text = _outline_to_text(outline)
        assert "Introduction" in text
        assert "3.1" in text

    def test_format_evidence_summary(self):
        em = _make_evidence_map()
        text = _format_evidence_summary(em)
        assert "Theme A" in text
        assert "Finding X" in text
        assert "Contradictions" in text


class TestOutlineGenerator:
    async def test_generate(self):
        llm = MockWritingLLM()
        gen = OutlineGenerator(llm)
        outline = await gen.generate(_make_evidence_map(), "Test scope")
        assert outline.title == "Test Review"
        assert len(outline.sections) == 3


class TestOutlineCritic:
    async def test_critique(self):
        llm = MockWritingLLM()
        critic = OutlineCritic(llm)
        outline = _make_outline()
        report = await critic.critique(outline, "Test scope")
        assert report.target == "outline"
        assert report.overall_score == 0.85


class TestRevision:
    def test_should_continue_empty(self):
        assert should_continue_revision([]) is True

    def test_should_stop_at_threshold(self):
        assert should_continue_revision([0.85], threshold=0.80) is False

    def test_should_stop_at_max_iterations(self):
        assert should_continue_revision([0.5, 0.6, 0.7], max_iterations=3) is False

    def test_should_stop_at_convergence(self):
        assert should_continue_revision([0.70, 0.71], convergence_delta=0.02) is False

    def test_should_continue_improving(self):
        assert (
            should_continue_revision([0.60, 0.70], threshold=0.80, convergence_delta=0.02) is True
        )


class TestSectionWriter:
    async def test_write_section(self):
        llm = MockWritingLLM()
        writer = SectionWriter(llm)
        outline = _make_outline()
        section = outline.sections[0]
        draft = await writer.write_section(
            section,
            outline,
            _make_extractions(),
            _make_evidence_map(),
        )
        assert draft.section_id == "1"
        assert "synthesizes" in draft.text

    def test_extract_citations(self):
        text = "Study found X [@p1] and Y [@p2] with confirmation [@p1]."
        citations = _extract_citations(text)
        assert citations == ["p1", "p2", "p1"]


class TestSectionCritic:
    async def test_critique(self):
        llm = MockWritingLLM()
        critic = SectionCritic(llm)
        draft = SectionDraft(section_id="1", title="Intro", text="Some text")
        report = await critic.critique(draft, _make_outline())
        assert report.target == "section"
        assert report.target_id == "1"


class TestDraftAssembler:
    def test_assemble(self):
        assembler = DraftAssembler()
        outline = _make_outline()
        drafts = {
            "1": SectionDraft(section_id="1", title="Introduction", text="Intro text"),
            "2": SectionDraft(section_id="2", title="Methods", text="Methods text"),
            "3": SectionDraft(section_id="3", title="Results", text="Results text"),
        }
        full = assembler.assemble(outline, drafts)
        assert "# Test Review" in full
        assert "## Introduction" in full
        assert "Intro text" in full
        assert "Methods text" in full


class TestHolisticCritic:
    async def test_critique(self):
        llm = MockWritingLLM()
        critic = HolisticCritic(llm)
        report = await critic.critique("Full draft text", "Test scope")
        assert report.target == "full_draft"

    async def test_holistic_loop_passes(self):
        llm = MockWritingLLM(critique_score=0.85)
        critic = HolisticCritic(llm)
        draft, critiques = await holistic_critique_loop(
            llm,
            critic,
            "Draft text",
            "Scope",
        )
        assert len(critiques) == 1
        assert critiques[0].passed is True
