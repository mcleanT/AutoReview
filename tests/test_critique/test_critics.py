"""Tests for OutlineCritic, SectionCritic, and HolisticCritic."""

from __future__ import annotations

from autoreview.critique.holistic_critic import HolisticCritic
from autoreview.critique.models import (
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    CritiqueTarget,
)
from autoreview.critique.outline_critic import OutlineCritic, _outline_to_text
from autoreview.critique.section_critic import SectionCritic
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.writing.section_writer import SectionDraft

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Gut Microbiome and Neurodegeneration",
        sections=[
            OutlineSection(
                id="1",
                title="Introduction",
                description="Overview of the gut-brain axis",
                paper_ids=["p1", "p2"],
                estimated_word_count=500,
            ),
            OutlineSection(
                id="2",
                title="Mechanisms",
                description="Molecular mechanisms linking gut to brain",
                paper_ids=["p1", "p3"],
                estimated_word_count=800,
                subsections=[
                    OutlineSection(
                        id="2.1",
                        title="Inflammatory Pathways",
                        description="Cytokine-mediated neuroinflammation",
                        paper_ids=["p3"],
                        estimated_word_count=400,
                    ),
                ],
            ),
            OutlineSection(
                id="3",
                title="Future Directions",
                description="Gaps and opportunities",
                paper_ids=[],
                estimated_word_count=300,
            ),
        ],
    )


def _make_section_draft(section_id: str = "1", title: str = "Introduction") -> SectionDraft:
    return SectionDraft(
        section_id=section_id,
        title=title,
        text=(
            "The gut-brain axis has emerged as a critical factor in neurodegeneration. "
            "Multiple studies demonstrate that gut dysbiosis precedes motor symptoms [@p1]. "
            "Furthermore, inflammatory mediators produced by gut bacteria can cross the "
            "blood-brain barrier [@p2], contributing to neuroinflammation [@p3]."
        ),
        citations_used=["p1", "p2", "p3"],
    )


def _make_passing_report(target: CritiqueTarget, target_id: str | None = None) -> CritiqueReport:
    return CritiqueReport(
        target=target,
        target_id=target_id,
        passed=True,
        overall_score=0.88,
        dimension_scores={"coverage": 0.90, "synthesis": 0.85},
        issues=[],
    )


def _make_failing_report(target: CritiqueTarget, target_id: str | None = None) -> CritiqueReport:
    return CritiqueReport(
        target=target,
        target_id=target_id,
        passed=False,
        overall_score=0.55,
        dimension_scores={"coverage": 0.50, "synthesis": 0.60},
        issues=[
            CritiqueIssue(
                severity=CritiqueSeverity.MAJOR,
                location="Section 2",
                description="Lacks synthesis across studies",
                suggested_fix="Integrate findings from p1 and p3 instead of listing them",
            ),
            CritiqueIssue(
                severity=CritiqueSeverity.MINOR,
                location="Section 1",
                description="Transition to Section 2 is abrupt",
            ),
        ],
        identified_gaps=["Vagal nerve signaling pathway not discussed"],
    )


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------


class MockStructuredLLM:
    """Mock LLM that returns a configurable CritiqueReport from generate_structured."""

    def __init__(self, report: CritiqueReport) -> None:
        self.report = report
        self.calls: list[dict] = []

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        self.calls.append(
            {
                "method": "generate_structured",
                "prompt": prompt,
                "response_model": response_model,
                "system": system,
            }
        )
        return LLMStructuredResponse(
            parsed=self.report,
            input_tokens=500,
            output_tokens=300,
        )

    async def generate(
        self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None
    ):
        self.calls.append(
            {
                "method": "generate",
                "prompt": prompt,
                "system": system,
            }
        )
        return LLMResponse(
            content="Revised text here.",
            input_tokens=400,
            output_tokens=200,
        )


class MockSequenceLLM:
    """Mock LLM that returns different CritiqueReports on successive calls.

    Useful for testing loops where the first critique fails and
    subsequent ones pass.
    """

    def __init__(self, reports: list[CritiqueReport]) -> None:
        self.reports = list(reports)
        self._call_index = 0
        self.structured_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        self.structured_calls.append(
            {
                "prompt": prompt,
                "response_model": response_model,
            }
        )
        report = self.reports[min(self._call_index, len(self.reports) - 1)]
        self._call_index += 1
        return LLMStructuredResponse(
            parsed=report,
            input_tokens=500,
            output_tokens=300,
        )

    async def generate(
        self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None
    ):
        self.generate_calls.append({"prompt": prompt})
        return LLMResponse(
            content="Revised text after addressing critique feedback.",
            input_tokens=400,
            output_tokens=200,
        )


# ===========================================================================
# OutlineCritic tests
# ===========================================================================


class TestOutlineCritic:
    """Tests for the OutlineCritic class."""

    async def test_critique_returns_report(self):
        """OutlineCritic.critique should return the CritiqueReport from the LLM."""
        expected = _make_passing_report(CritiqueTarget.OUTLINE)
        llm = MockStructuredLLM(expected)
        critic = OutlineCritic(llm)

        report = await critic.critique(_make_outline(), "Test scope document")

        assert isinstance(report, CritiqueReport)
        assert report.passed is True
        assert report.overall_score == 0.88

    async def test_critique_sets_target_to_outline(self):
        """The returned report must have target set to OUTLINE
        regardless of what the LLM returns.
        """
        # Give the LLM a report with wrong target -- critic should override it
        wrong_target_report = CritiqueReport(
            target=CritiqueTarget.SECTION,
            passed=True,
            overall_score=0.90,
        )
        llm = MockStructuredLLM(wrong_target_report)
        critic = OutlineCritic(llm)

        report = await critic.critique(_make_outline(), "Test scope")

        assert report.target == CritiqueTarget.OUTLINE

    async def test_critique_passes_correct_model(self):
        """generate_structured should be called with CritiqueReport as response_model."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.OUTLINE))
        critic = OutlineCritic(llm)

        await critic.critique(_make_outline(), "scope")

        assert len(llm.calls) == 1
        assert llm.calls[0]["response_model"] is CritiqueReport

    async def test_critique_prompt_contains_outline_text(self):
        """The prompt should contain the rendered outline text."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.OUTLINE))
        critic = OutlineCritic(llm)
        outline = _make_outline()

        await critic.critique(outline, "scope about gut microbiome")

        prompt = llm.calls[0]["prompt"]
        # The prompt should contain section titles from the outline
        assert "Introduction" in prompt
        assert "Mechanisms" in prompt
        assert "Inflammatory Pathways" in prompt

    async def test_critique_prompt_contains_scope_document(self):
        """The prompt should embed the scope document."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.OUTLINE))
        critic = OutlineCritic(llm)

        await critic.critique(_make_outline(), "Gut microbiome and PD review scope")

        prompt = llm.calls[0]["prompt"]
        assert "Gut microbiome and PD review scope" in prompt

    async def test_critique_with_issues(self):
        """A failing report with issues should be returned with issues intact."""
        expected = _make_failing_report(CritiqueTarget.OUTLINE)
        llm = MockStructuredLLM(expected)
        critic = OutlineCritic(llm)

        report = await critic.critique(_make_outline(), "scope")

        assert report.passed is False
        assert len(report.issues) == 2
        assert report.issues[0].severity == CritiqueSeverity.MAJOR
        assert report.overall_score == 0.55

    async def test_critique_preserves_dimension_scores(self):
        """Dimension scores from the LLM should pass through."""
        expected = _make_passing_report(CritiqueTarget.OUTLINE)
        llm = MockStructuredLLM(expected)
        critic = OutlineCritic(llm)

        report = await critic.critique(_make_outline(), "scope")

        assert report.dimension_scores == {"coverage": 0.90, "synthesis": 0.85}


class TestOutlineToText:
    """Tests for the _outline_to_text helper."""

    def test_includes_all_sections(self):
        outline = _make_outline()
        text = _outline_to_text(outline)

        assert "Introduction" in text
        assert "Mechanisms" in text
        assert "Inflammatory Pathways" in text
        assert "Future Directions" in text

    def test_includes_section_ids(self):
        outline = _make_outline()
        text = _outline_to_text(outline)

        assert "[1]" in text
        assert "[2]" in text
        assert "[2.1]" in text
        assert "[3]" in text

    def test_includes_paper_counts(self):
        outline = _make_outline()
        text = _outline_to_text(outline)

        # Section 1 has 2 papers
        assert "Papers: 2" in text

    def test_includes_word_counts(self):
        outline = _make_outline()
        text = _outline_to_text(outline)

        assert "Est. words: 500" in text
        assert "Est. words: 800" in text

    def test_includes_title(self):
        outline = _make_outline()
        text = _outline_to_text(outline)

        assert "# Gut Microbiome and Neurodegeneration" in text


# ===========================================================================
# SectionCritic tests
# ===========================================================================


class TestSectionCritic:
    """Tests for the SectionCritic class."""

    async def test_critique_returns_report(self):
        """SectionCritic.critique should return the CritiqueReport from the LLM."""
        expected = _make_passing_report(CritiqueTarget.SECTION, target_id="1")
        llm = MockStructuredLLM(expected)
        critic = SectionCritic(llm)

        report = await critic.critique(_make_section_draft(), _make_outline())

        assert isinstance(report, CritiqueReport)
        assert report.passed is True

    async def test_critique_sets_target_to_section(self):
        """The returned report must have target set to SECTION."""
        wrong_target = CritiqueReport(
            target=CritiqueTarget.FULL_DRAFT,
            passed=True,
            overall_score=0.85,
        )
        llm = MockStructuredLLM(wrong_target)
        critic = SectionCritic(llm)

        report = await critic.critique(_make_section_draft(), _make_outline())

        assert report.target == CritiqueTarget.SECTION

    async def test_critique_sets_target_id(self):
        """The returned report should carry the section_id from the draft."""
        expected = _make_passing_report(CritiqueTarget.SECTION)
        llm = MockStructuredLLM(expected)
        critic = SectionCritic(llm)

        draft = _make_section_draft(section_id="2.1", title="Inflammatory Pathways")
        report = await critic.critique(draft, _make_outline())

        assert report.target_id == "2.1"

    async def test_critique_prompt_contains_section_text(self):
        """The prompt should contain the section text being critiqued."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        draft = _make_section_draft()
        await critic.critique(draft, _make_outline())

        prompt = llm.calls[0]["prompt"]
        assert "gut-brain axis" in prompt
        assert "[@p1]" in prompt

    async def test_critique_prompt_contains_outline_context(self):
        """The prompt should contain outline context (all section titles)."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        await critic.critique(_make_section_draft(), _make_outline())

        prompt = llm.calls[0]["prompt"]
        assert "Introduction" in prompt
        assert "Mechanisms" in prompt
        assert "Future Directions" in prompt

    async def test_critique_without_adjacent_text(self):
        """Critique works when no adjacent_text is provided (defaults to empty)."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        report = await critic.critique(_make_section_draft(), _make_outline())

        assert report.passed is True
        # Adjacent context should be empty in the prompt
        prompt = llm.calls[0]["prompt"]
        assert "Adjacent Sections" in prompt

    async def test_critique_with_adjacent_text(self):
        """Adjacent section text should appear in the prompt when provided."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        adjacent = "The previous section discussed general mechanisms of the gut-brain axis."
        report = await critic.critique(
            _make_section_draft(), _make_outline(), adjacent_text=adjacent
        )

        assert report.passed is True
        prompt = llm.calls[0]["prompt"]
        assert "previous section discussed general mechanisms" in prompt

    async def test_critique_passes_correct_response_model(self):
        """generate_structured should be called with CritiqueReport."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        await critic.critique(_make_section_draft(), _make_outline())

        assert llm.calls[0]["response_model"] is CritiqueReport

    async def test_critique_uses_section_critique_system_prompt(self):
        """The system prompt should be the section critique system prompt."""
        from autoreview.llm.prompts.critique import SECTION_CRITIQUE_SYSTEM_PROMPT

        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.SECTION))
        critic = SectionCritic(llm)

        await critic.critique(_make_section_draft(), _make_outline())

        assert llm.calls[0]["system"] == SECTION_CRITIQUE_SYSTEM_PROMPT

    async def test_critique_with_failing_report(self):
        """A failing report should carry its issues through correctly."""
        expected = _make_failing_report(CritiqueTarget.SECTION, target_id="2")
        llm = MockStructuredLLM(expected)
        critic = SectionCritic(llm)

        draft = _make_section_draft(section_id="2", title="Mechanisms")
        report = await critic.critique(draft, _make_outline())

        assert report.passed is False
        assert len(report.issues) == 2
        assert report.identified_gaps == ["Vagal nerve signaling pathway not discussed"]


# ===========================================================================
# HolisticCritic tests
# ===========================================================================


class TestHolisticCritic:
    """Tests for the HolisticCritic class."""

    async def test_critique_returns_report(self):
        """HolisticCritic.critique should return the CritiqueReport from the LLM."""
        expected = _make_passing_report(CritiqueTarget.FULL_DRAFT)
        llm = MockStructuredLLM(expected)
        critic = HolisticCritic(llm)

        report = await critic.critique(
            "Full draft of the review paper...",
            "Scope document for gut microbiome review",
        )

        assert isinstance(report, CritiqueReport)
        assert report.passed is True
        assert report.overall_score == 0.88

    async def test_critique_sets_target_to_full_draft(self):
        """The returned report must have target set to FULL_DRAFT."""
        wrong_target = CritiqueReport(
            target=CritiqueTarget.OUTLINE,
            passed=True,
            overall_score=0.90,
        )
        llm = MockStructuredLLM(wrong_target)
        critic = HolisticCritic(llm)

        report = await critic.critique("Full draft text", "scope")

        assert report.target == CritiqueTarget.FULL_DRAFT

    async def test_critique_prompt_contains_draft(self):
        """The prompt should contain the full draft text."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.FULL_DRAFT))
        critic = HolisticCritic(llm)

        draft_text = "# Review of Gut Microbiome\n\nThis review examines..."
        await critic.critique(draft_text, "scope")

        prompt = llm.calls[0]["prompt"]
        assert "Review of Gut Microbiome" in prompt
        assert "This review examines" in prompt

    async def test_critique_prompt_contains_scope(self):
        """The prompt should contain the scope document."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.FULL_DRAFT))
        critic = HolisticCritic(llm)

        scope = "This review covers the gut-brain axis in Parkinson's disease"
        await critic.critique("Draft text", scope)

        prompt = llm.calls[0]["prompt"]
        assert "gut-brain axis in Parkinson" in prompt

    async def test_critique_uses_holistic_system_prompt(self):
        """The system prompt should be the holistic critique system prompt."""
        from autoreview.llm.prompts.critique import HOLISTIC_CRITIQUE_SYSTEM_PROMPT

        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.FULL_DRAFT))
        critic = HolisticCritic(llm)

        await critic.critique("Draft", "scope")

        assert llm.calls[0]["system"] == HOLISTIC_CRITIQUE_SYSTEM_PROMPT

    async def test_critique_passes_correct_response_model(self):
        """generate_structured should be called with CritiqueReport."""
        llm = MockStructuredLLM(_make_passing_report(CritiqueTarget.FULL_DRAFT))
        critic = HolisticCritic(llm)

        await critic.critique("Draft", "scope")

        assert llm.calls[0]["response_model"] is CritiqueReport

    async def test_critique_with_issues(self):
        """A failing holistic critique should carry issues."""
        expected = _make_failing_report(CritiqueTarget.FULL_DRAFT)
        llm = MockStructuredLLM(expected)
        critic = HolisticCritic(llm)

        report = await critic.critique("Draft", "scope")

        assert report.passed is False
        assert len(report.issues) == 2
        assert report.overall_score == 0.55
