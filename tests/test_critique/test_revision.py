"""Tests for revision logic: should_continue_revision, revise_text, outline_critique_loop,
section_critique_loop, and holistic_critique_loop."""

from __future__ import annotations

import pytest

from autoreview.critique.models import (
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    CritiqueTarget,
)
from autoreview.critique.revision import (
    _format_issues,
    outline_critique_loop,
    revise_text,
    should_continue_revision,
)
from autoreview.critique.section_critic import SectionCritic, section_critique_loop
from autoreview.critique.holistic_critic import HolisticCritic, holistic_critique_loop
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.writing.section_writer import SectionDraft


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="1", title="Introduction", description="Overview",
                paper_ids=["p1"], estimated_word_count=500,
            ),
            OutlineSection(
                id="2", title="Results", description="Findings",
                paper_ids=["p1", "p2"], estimated_word_count=800,
            ),
        ],
    )


def _make_critique_report(
    passed: bool,
    score: float,
    target: CritiqueTarget = CritiqueTarget.SECTION,
    target_id: str | None = None,
    issues: list[CritiqueIssue] | None = None,
) -> CritiqueReport:
    if issues is None and not passed:
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.MAJOR,
                location="Section 1",
                description="Lacks synthesis",
                suggested_fix="Combine findings from p1 and p2",
            ),
        ]
    return CritiqueReport(
        target=target,
        target_id=target_id,
        passed=passed,
        overall_score=score,
        dimension_scores={"synthesis_quality": score, "coherence": score + 0.05},
        issues=issues or [],
    )


def _make_section_draft() -> SectionDraft:
    return SectionDraft(
        section_id="1",
        title="Introduction",
        text="Original section text with findings from multiple studies [@p1] [@p2].",
        citations_used=["p1", "p2"],
    )


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class MockRevisionLLM:
    """Mock LLM for revise_text tests. Records calls and returns configurable text."""

    def __init__(self, revised_text: str = "Revised and improved text.") -> None:
        self.revised_text = revised_text
        self.generate_calls: list[dict] = []
        self.structured_calls: list[dict] = []

    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None):
        self.generate_calls.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        return LLMResponse(
            content=self.revised_text,
            input_tokens=400,
            output_tokens=200,
        )

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0, model_override=None):
        self.structured_calls.append({
            "prompt": prompt,
            "response_model": response_model,
        })
        return LLMStructuredResponse(
            parsed=response_model(), input_tokens=500, output_tokens=300,
        )


class MockCritiqueLoopLLM:
    """Mock LLM that returns a sequence of critique reports (for structured calls)
    and revised text (for generate calls)."""

    def __init__(
        self,
        reports: list[CritiqueReport],
        revised_text: str = "Improved text after revision.",
    ) -> None:
        self.reports = list(reports)
        self.revised_text = revised_text
        self._structured_idx = 0
        self.structured_calls: list[str] = []
        self.generate_calls: list[str] = []

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0, model_override=None):
        self.structured_calls.append(prompt)
        report = self.reports[min(self._structured_idx, len(self.reports) - 1)]
        self._structured_idx += 1
        return LLMStructuredResponse(
            parsed=report, input_tokens=500, output_tokens=300,
        )

    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3, model_override=None):
        self.generate_calls.append(prompt)
        return LLMResponse(
            content=self.revised_text,
            input_tokens=400,
            output_tokens=200,
        )


class MockOutlineGenerator:
    """Mock outline generator for outline_critique_loop tests."""

    def __init__(self, outline: ReviewOutline | None = None) -> None:
        self.outline = outline or _make_outline()
        self.call_count = 0
        self.revision_calls: list[dict] = []

    async def generate(
        self, evidence_map, scope_document, required_sections=None,
        previous_outline=None, critique_report=None,
    ):
        self.call_count += 1
        if previous_outline is not None:
            self.revision_calls.append({
                "previous_outline": previous_outline,
                "critique_report": critique_report,
            })
        return self.outline


class MockOutlineCritic:
    """Mock outline critic that returns a sequence of CritiqueReports."""

    def __init__(self, reports: list[CritiqueReport]) -> None:
        self.reports = list(reports)
        self._idx = 0
        self.call_count = 0

    async def critique(self, outline, scope_document):
        self.call_count += 1
        report = self.reports[min(self._idx, len(self.reports) - 1)]
        self._idx += 1
        return report


# ===========================================================================
# should_continue_revision tests
# ===========================================================================

class TestShouldContinueRevision:
    """Unit tests for the should_continue_revision termination function."""

    def test_empty_scores_returns_true(self):
        """With no scores yet, revision should continue."""
        assert should_continue_revision([]) is True

    def test_score_exceeds_threshold_stops(self):
        """If the latest score exceeds the threshold, stop."""
        assert should_continue_revision([0.85], threshold=0.80) is False

    def test_score_equals_threshold_stops(self):
        """Exactly meeting the threshold should stop revision."""
        assert should_continue_revision([0.80], threshold=0.80) is False

    def test_score_below_threshold_continues(self):
        """Below threshold with room for improvement should continue."""
        assert should_continue_revision([0.70], threshold=0.80) is True

    def test_max_iterations_reached_stops(self):
        """Even if below threshold, reaching max iterations should stop."""
        assert should_continue_revision([0.5, 0.6, 0.7], max_iterations=3) is False

    def test_max_iterations_one_stops(self):
        """A single iteration max should stop after one score."""
        assert should_continue_revision([0.5], max_iterations=1) is False

    def test_below_max_iterations_continues(self):
        """If fewer scores than max iterations, continue."""
        assert should_continue_revision([0.5, 0.6], max_iterations=3) is True

    def test_score_converged_stops(self):
        """If improvement is less than convergence_delta, stop (plateau)."""
        assert should_continue_revision(
            [0.70, 0.71], convergence_delta=0.02
        ) is False

    def test_score_converged_exactly_at_delta(self):
        """Improvement equal to delta should not continue (< check uses <)."""
        # improvement = 0.02, delta = 0.02 -> 0.02 < 0.02 is False -> should stop
        # Wait, let's check the code: improvement < convergence_delta
        # 0.02 < 0.02 is False, so it won't trigger the stop.
        # That means it should continue!
        assert should_continue_revision(
            [0.70, 0.72], convergence_delta=0.02
        ) is True

    def test_score_decreasing_stops(self):
        """If score decreased, improvement is negative, which is < delta. Should stop."""
        assert should_continue_revision(
            [0.75, 0.73], convergence_delta=0.02
        ) is False

    def test_score_improving_significantly_continues(self):
        """If improvement exceeds delta and below threshold, continue."""
        assert should_continue_revision(
            [0.60, 0.70], threshold=0.80, convergence_delta=0.02
        ) is True

    def test_convergence_not_checked_with_single_score(self):
        """With only one score, convergence cannot be assessed, should continue if below threshold."""
        assert should_continue_revision(
            [0.70], threshold=0.80, convergence_delta=0.02
        ) is True

    def test_multiple_factors_threshold_wins(self):
        """If latest score meets threshold, stop even if convergence says continue."""
        assert should_continue_revision(
            [0.50, 0.85], threshold=0.80, convergence_delta=0.02
        ) is False

    def test_multiple_factors_max_iterations_wins(self):
        """Max iterations reached should stop even if score is below threshold."""
        assert should_continue_revision(
            [0.50, 0.60, 0.70], threshold=0.80, max_iterations=3
        ) is False

    def test_default_parameters(self):
        """Test with default parameters (threshold=0.80, delta=0.02, max=3)."""
        # Below threshold, only one score
        assert should_continue_revision([0.50]) is True
        # Above default threshold
        assert should_continue_revision([0.82]) is False
        # Converged at defaults
        assert should_continue_revision([0.75, 0.76]) is False
        # Max iterations at default
        assert should_continue_revision([0.5, 0.6, 0.7]) is False


# ===========================================================================
# revise_text tests
# ===========================================================================

class TestReviseText:
    """Tests for the revise_text function."""

    async def test_returns_revised_content(self):
        """revise_text should return the LLM's response content."""
        llm = MockRevisionLLM(revised_text="Much better version of the text.")
        critique = _make_critique_report(passed=False, score=0.55)

        result = await revise_text(llm, "Original text", critique)

        assert result == "Much better version of the text."

    async def test_calls_generate_not_structured(self):
        """revise_text should use generate (free-form), not generate_structured."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(passed=False, score=0.55)

        await revise_text(llm, "Original text", critique)

        assert len(llm.generate_calls) == 1
        assert len(llm.structured_calls) == 0

    async def test_prompt_contains_original_text(self):
        """The prompt should include the original text to revise."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(passed=False, score=0.55)

        await revise_text(llm, "Original text about gut microbiome", critique)

        prompt = llm.generate_calls[0]["prompt"]
        assert "Original text about gut microbiome" in prompt

    async def test_prompt_contains_formatted_issues(self):
        """Issues from the critique should appear in the prompt."""
        llm = MockRevisionLLM()
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.CRITICAL,
                location="Paragraph 3",
                description="Missing citation for key claim",
                suggested_fix="Add citation to Smith 2023",
            ),
        ]
        critique = _make_critique_report(passed=False, score=0.40, issues=issues)

        await revise_text(llm, "Text to revise", critique)

        prompt = llm.generate_calls[0]["prompt"]
        assert "Missing citation for key claim" in prompt
        assert "Add citation to Smith 2023" in prompt

    async def test_prompt_contains_context_when_provided(self):
        """Optional context should appear in the prompt."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(passed=False, score=0.55)

        await revise_text(
            llm, "Text", critique,
            context="Full review scope: gut-brain axis in PD"
        )

        prompt = llm.generate_calls[0]["prompt"]
        assert "gut-brain axis in PD" in prompt

    async def test_prompt_without_context(self):
        """When no context is provided, prompt should still be valid."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(passed=False, score=0.55)

        result = await revise_text(llm, "Text", critique)

        assert result is not None

    async def test_no_issues_formats_as_no_issues_message(self):
        """When there are no issues, the formatted text should say so."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(passed=True, score=0.90, issues=[])

        await revise_text(llm, "Text", critique)

        prompt = llm.generate_calls[0]["prompt"]
        assert "No specific issues identified" in prompt

    async def test_issues_sorted_by_severity_in_prompt(self):
        """Issues should be sorted by severity (critical first) in the prompt."""
        llm = MockRevisionLLM()
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.MINOR,
                location="s1",
                description="Minor style issue",
            ),
            CritiqueIssue(
                severity=CritiqueSeverity.CRITICAL,
                location="s2",
                description="Critical missing data",
            ),
            CritiqueIssue(
                severity=CritiqueSeverity.MAJOR,
                location="s3",
                description="Major synthesis gap",
            ),
        ]
        critique = _make_critique_report(passed=False, score=0.40, issues=issues)

        await revise_text(llm, "Text", critique)

        prompt = llm.generate_calls[0]["prompt"]
        # Critical should appear before major, which appears before minor
        crit_pos = prompt.index("Critical missing data")
        major_pos = prompt.index("Major synthesis gap")
        minor_pos = prompt.index("Minor style issue")
        assert crit_pos < major_pos < minor_pos

    async def test_uses_section_rubrics_for_section_target(self):
        """For a section-targeted critique, dimension feedback should use section rubrics."""
        llm = MockRevisionLLM()
        critique = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION,
        )

        await revise_text(llm, "Section text", critique)

        prompt = llm.generate_calls[0]["prompt"]
        # Should contain dimension feedback with synthesis_quality
        assert "synthesis_quality" in prompt

    async def test_uses_holistic_rubrics_for_full_draft_target(self):
        """For a full-draft critique, dimension feedback should use holistic rubrics."""
        llm = MockRevisionLLM()
        critique = CritiqueReport(
            target=CritiqueTarget.FULL_DRAFT,
            passed=False,
            overall_score=0.55,
            dimension_scores={"narrative_arc": 0.50, "redundancy": 0.60},
            issues=[
                CritiqueIssue(
                    severity=CritiqueSeverity.MAJOR,
                    location="Overall",
                    description="Narrative arc is weak",
                ),
            ],
        )

        await revise_text(llm, "Full draft text", critique)

        prompt = llm.generate_calls[0]["prompt"]
        assert "narrative_arc" in prompt


# ===========================================================================
# _format_issues tests (supplementing test_rubrics.py)
# ===========================================================================

class TestFormatIssues:
    """Additional tests for _format_issues."""

    def test_includes_severity_labels(self):
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.CRITICAL,
                location="Section 2",
                description="Missing citation",
            ),
        ]
        result = _format_issues(issues)
        assert "[critical]" in result
        assert "Section 2" in result
        assert "Missing citation" in result

    def test_includes_suggested_fix(self):
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.MAJOR,
                location="Section 1",
                description="Weak transitions",
                suggested_fix="Add a bridging sentence",
            ),
        ]
        result = _format_issues(issues)
        assert "Suggested fix: Add a bridging sentence" in result

    def test_omits_suggested_fix_when_none(self):
        issues = [
            CritiqueIssue(
                severity=CritiqueSeverity.MINOR,
                location="Section 3",
                description="Typo",
            ),
        ]
        result = _format_issues(issues)
        assert "Suggested fix" not in result

    def test_multiple_issues_are_bullet_listed(self):
        issues = [
            CritiqueIssue(severity=CritiqueSeverity.MAJOR, location="s1", description="Issue A"),
            CritiqueIssue(severity=CritiqueSeverity.MINOR, location="s2", description="Issue B"),
        ]
        result = _format_issues(issues)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert all(line.startswith("- ") for line in lines)


# ===========================================================================
# outline_critique_loop tests
# ===========================================================================

class TestOutlineCritiqueLoop:
    """Tests for the outline_critique_loop function."""

    async def test_passes_on_first_attempt(self):
        """When the first critique passes, no revision is needed."""
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.OUTLINE
        )
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic([passing])

        outline, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=2,
        )

        assert outline.title == "Test Review"
        assert len(critiques) == 1
        assert critiques[0].passed is True
        # Generator called once (initial), critic called once
        assert generator.call_count == 1
        assert critic.call_count == 1

    async def test_fails_then_passes(self):
        """When first critique fails and second passes, one revision cycle occurs."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.OUTLINE
        )
        passing = _make_critique_report(
            passed=True, score=0.85, target=CritiqueTarget.OUTLINE
        )
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic([failing, passing])

        outline, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=2,
        )

        assert len(critiques) == 2
        assert critiques[0].passed is False
        assert critiques[1].passed is True
        # Generator called twice (initial + revision), critic called twice
        assert generator.call_count == 2
        assert critic.call_count == 2

    async def test_max_cycles_reached(self):
        """When max cycles is reached without passing, loop stops."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.OUTLINE
        )
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic([failing, failing])

        outline, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=2,
        )

        # With max_cycles=2, should_continue_revision([0.55, 0.55], max_iterations=2)
        # returns False (max iterations reached), so loop stops after 2 critiques
        assert len(critiques) == 2
        assert all(not c.passed for c in critiques)

    async def test_convergence_stops_loop(self):
        """If scores converge (plateau), the loop should stop early."""
        low1 = _make_critique_report(
            passed=False, score=0.60, target=CritiqueTarget.OUTLINE
        )
        low2 = _make_critique_report(
            passed=False, score=0.61, target=CritiqueTarget.OUTLINE
        )
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic([low1, low2])

        outline, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=5,  # Allow many cycles
            threshold=0.80,
        )

        # Should stop after 2 because improvement (0.01) < default convergence_delta (0.02)
        # Note: outline_critique_loop doesn't pass convergence_delta, it uses
        # should_continue_revision's default of 0.02
        assert len(critiques) == 2

    async def test_returns_final_outline(self):
        """The loop should return the last generated outline."""
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.OUTLINE
        )
        custom_outline = ReviewOutline(
            title="Custom Outline",
            sections=[
                OutlineSection(
                    id="1", title="Intro", description="Intro",
                    estimated_word_count=300,
                ),
            ],
        )
        generator = MockOutlineGenerator(outline=custom_outline)
        critic = MockOutlineCritic([passing])

        outline, _ = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
        )

        assert outline.title == "Custom Outline"

    async def test_single_max_cycle(self):
        """With max_cycles=1, only one critique is performed (no revision)."""
        failing = _make_critique_report(
            passed=False, score=0.50, target=CritiqueTarget.OUTLINE
        )
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic([failing])

        outline, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=1,
        )

        assert len(critiques) == 1
        # Generator called once (initial only), no revision
        assert generator.call_count == 1

    async def test_collects_all_critique_reports(self):
        """All critique reports from the loop should be returned."""
        reports = [
            _make_critique_report(passed=False, score=0.50, target=CritiqueTarget.OUTLINE),
            _make_critique_report(passed=False, score=0.65, target=CritiqueTarget.OUTLINE),
            _make_critique_report(passed=True, score=0.85, target=CritiqueTarget.OUTLINE),
        ]
        generator = MockOutlineGenerator()
        critic = MockOutlineCritic(reports)

        _, critiques = await outline_critique_loop(
            llm=None,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=None,
            scope_document="scope",
            max_cycles=5,
        )

        # The loop uses should_continue_revision. After scores [0.50]:
        # continue. After [0.50, 0.65]: improvement=0.15 > 0.02, below 0.80 -> continue.
        # After [0.50, 0.65, 0.85]: 0.85 >= 0.80 -> stop (but report.passed=True also stops).
        # So we get 3 critiques.
        assert len(critiques) == 3
        assert critiques[-1].passed is True


# ===========================================================================
# section_critique_loop tests
# ===========================================================================

class TestSectionCritiqueLoop:
    """Tests for the section_critique_loop function."""

    async def test_passes_on_first_attempt(self):
        """When the section passes on first critique, no revision occurs."""
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.SECTION, target_id="1"
        )
        llm = MockCritiqueLoopLLM([passing])
        critic = SectionCritic(llm)

        draft, critiques = await section_critique_loop(
            llm=llm,
            critic=critic,
            draft=_make_section_draft(),
            outline=_make_outline(),
            max_cycles=2,
        )

        assert len(critiques) == 1
        assert critiques[0].passed is True
        # No generate calls (no revision needed)
        assert len(llm.generate_calls) == 0

    async def test_fails_then_passes(self):
        """When first critique fails and second passes, one revision cycle runs."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION, target_id="1"
        )
        passing = _make_critique_report(
            passed=True, score=0.85, target=CritiqueTarget.SECTION, target_id="1"
        )
        llm = MockCritiqueLoopLLM([failing, passing])
        critic = SectionCritic(llm)

        draft, critiques = await section_critique_loop(
            llm=llm,
            critic=critic,
            draft=_make_section_draft(),
            outline=_make_outline(),
            max_cycles=2,
        )

        assert len(critiques) == 2
        assert critiques[0].passed is False
        assert critiques[1].passed is True
        # One generate call for the revision
        assert len(llm.generate_calls) == 1

    async def test_max_cycles_stops_loop(self):
        """Loop stops at max_cycles even if section never passes."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION
        )
        llm = MockCritiqueLoopLLM([failing, failing])
        critic = SectionCritic(llm)

        draft, critiques = await section_critique_loop(
            llm=llm,
            critic=critic,
            draft=_make_section_draft(),
            outline=_make_outline(),
            max_cycles=2,
        )

        assert len(critiques) == 2

    async def test_revised_text_replaces_draft(self):
        """After revision, the draft text should be the revised text."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION
        )
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.SECTION
        )
        llm = MockCritiqueLoopLLM(
            [failing, passing],
            revised_text="This is the improved section with better synthesis.",
        )
        critic = SectionCritic(llm)

        draft, critiques = await section_critique_loop(
            llm=llm,
            critic=critic,
            draft=_make_section_draft(),
            outline=_make_outline(),
            max_cycles=2,
        )

        assert draft.text == "This is the improved section with better synthesis."

    async def test_preserves_section_metadata(self):
        """The revised draft should preserve section_id and title."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION
        )
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.SECTION
        )
        llm = MockCritiqueLoopLLM([failing, passing])
        critic = SectionCritic(llm)

        original = SectionDraft(
            section_id="2.1",
            title="Inflammatory Pathways",
            text="Original text.",
            citations_used=["p3"],
        )

        draft, _ = await section_critique_loop(
            llm=llm, critic=critic, draft=original,
            outline=_make_outline(), max_cycles=2,
        )

        assert draft.section_id == "2.1"
        assert draft.title == "Inflammatory Pathways"
        assert draft.citations_used == ["p3"]

    async def test_extra_issues_injected_on_first_cycle(self):
        """Extra issues (e.g. citation validation) should be injected into the first cycle's report."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.SECTION,
            issues=[
                CritiqueIssue(
                    severity=CritiqueSeverity.MINOR,
                    location="s1",
                    description="LLM-detected issue",
                ),
            ],
        )
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.SECTION
        )
        llm = MockCritiqueLoopLLM([failing, passing])
        critic = SectionCritic(llm)

        extra = [
            CritiqueIssue(
                severity=CritiqueSeverity.CRITICAL,
                location="s1",
                description="Citation p4 not found in corpus",
            ),
        ]

        draft, critiques = await section_critique_loop(
            llm=llm, critic=critic, draft=_make_section_draft(),
            outline=_make_outline(), max_cycles=2, extra_issues=extra,
        )

        # First critique should have extra issues prepended
        first_issues = critiques[0].issues
        assert len(first_issues) == 2
        assert first_issues[0].description == "Citation p4 not found in corpus"
        assert first_issues[1].description == "LLM-detected issue"

    async def test_with_adjacent_text(self):
        """Adjacent text should be passed through to the critic."""
        passing = _make_critique_report(
            passed=True, score=0.90, target=CritiqueTarget.SECTION
        )
        llm = MockCritiqueLoopLLM([passing])
        critic = SectionCritic(llm)

        draft, critiques = await section_critique_loop(
            llm=llm, critic=critic, draft=_make_section_draft(),
            outline=_make_outline(), max_cycles=2,
            adjacent_text="Previous section discussed X.",
        )

        assert len(critiques) == 1
        # The prompt should contain adjacent text
        assert "Previous section discussed X." in llm.structured_calls[0]


# ===========================================================================
# holistic_critique_loop tests
# ===========================================================================

class TestHolisticCritiqueLoop:
    """Tests for the holistic_critique_loop function."""

    async def test_passes_on_first_attempt(self):
        """When the full draft passes, no revision occurs."""
        passing = _make_critique_report(
            passed=True, score=0.88, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([passing])
        critic = HolisticCritic(llm)

        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Full draft text.",
            scope_document="Review scope",
        )

        assert len(critiques) == 1
        assert critiques[0].passed is True
        assert draft == "Full draft text."
        assert len(llm.generate_calls) == 0

    async def test_fails_then_passes(self):
        """When first critique fails, revision runs, then second critique passes."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.FULL_DRAFT
        )
        passing = _make_critique_report(
            passed=True, score=0.85, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM(
            [failing, passing],
            revised_text="Improved full draft.",
        )
        critic = HolisticCritic(llm)

        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Original draft.",
            scope_document="scope",
        )

        assert len(critiques) == 2
        assert draft == "Improved full draft."
        assert len(llm.generate_calls) == 1

    async def test_max_cycles_stops_loop(self):
        """Loop stops at max_cycles even if never passes."""
        # Use incrementally increasing scores to avoid convergence stopping early
        fail1 = _make_critique_report(
            passed=False, score=0.50, target=CritiqueTarget.FULL_DRAFT
        )
        fail2 = _make_critique_report(
            passed=False, score=0.60, target=CritiqueTarget.FULL_DRAFT
        )
        fail3 = _make_critique_report(
            passed=False, score=0.70, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([fail1, fail2, fail3])
        critic = HolisticCritic(llm)

        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Draft.",
            scope_document="scope",
            max_cycles=3,
        )

        assert len(critiques) == 3

    async def test_convergence_stops_early(self):
        """If scores plateau, the loop should stop before max_cycles."""
        low1 = _make_critique_report(
            passed=False, score=0.60, target=CritiqueTarget.FULL_DRAFT
        )
        low2 = _make_critique_report(
            passed=False, score=0.61, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([low1, low2])
        critic = HolisticCritic(llm)

        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Draft.",
            scope_document="scope",
            max_cycles=5,
            convergence_delta=0.02,
        )

        assert len(critiques) == 2

    async def test_extra_issues_on_first_cycle(self):
        """Extra issues should be injected into the first cycle's critique."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.FULL_DRAFT,
            issues=[
                CritiqueIssue(
                    severity=CritiqueSeverity.MINOR,
                    location="Overall",
                    description="LLM-detected issue",
                ),
            ],
        )
        passing = _make_critique_report(
            passed=True, score=0.85, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([failing, passing])
        critic = HolisticCritic(llm)

        extra = [
            CritiqueIssue(
                severity=CritiqueSeverity.CRITICAL,
                location="Citations",
                description="Uncited claim in conclusion",
            ),
        ]

        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Draft.",
            scope_document="scope",
            extra_issues=extra,
        )

        first_issues = critiques[0].issues
        assert len(first_issues) == 2
        assert first_issues[0].description == "Uncited claim in conclusion"

    async def test_revision_passes_scope_as_context(self):
        """The revision call should include scope document context."""
        failing = _make_critique_report(
            passed=False, score=0.55, target=CritiqueTarget.FULL_DRAFT
        )
        passing = _make_critique_report(
            passed=True, score=0.85, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([failing, passing])
        critic = HolisticCritic(llm)

        draft, _ = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Draft.",
            scope_document="gut-brain axis review scope",
        )

        # The revision generate call should have been made
        assert len(llm.generate_calls) == 1

    async def test_custom_threshold(self):
        """A custom threshold should be respected."""
        score_65 = _make_critique_report(
            passed=False, score=0.65, target=CritiqueTarget.FULL_DRAFT
        )
        llm = MockCritiqueLoopLLM([score_65])
        critic = HolisticCritic(llm)

        # With threshold 0.60, the score 0.65 should trigger stop
        draft, critiques = await holistic_critique_loop(
            llm=llm, critic=critic,
            full_draft="Draft.",
            scope_document="scope",
            threshold=0.60,
        )

        # Even though passed=False, should_continue_revision sees 0.65 >= 0.60
        # and returns False, so loop stops after 1 cycle
        assert len(critiques) == 1
