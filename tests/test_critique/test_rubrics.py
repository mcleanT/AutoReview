"""Tests for rubric-based critique scoring."""

from __future__ import annotations

from autoreview.critique.models import CritiqueIssue, CritiqueSeverity
from autoreview.critique.rubrics import (
    HOLISTIC_DIMENSION_NAMES,
    HOLISTIC_RUBRICS,
    SECTION_DIMENSION_NAMES,
    SECTION_RUBRICS,
    DimensionRubric,
    format_dimension_feedback,
    format_rubrics_for_prompt,
)
from autoreview.critique.revision import _format_issues


class TestRubricFormatForPrompt:
    def test_contains_all_anchors(self):
        formatted = format_rubrics_for_prompt(SECTION_RUBRICS)
        for rubric in SECTION_RUBRICS:
            assert f"### {rubric.dimension}" in formatted
            for anchor in rubric.anchors:
                assert anchor.range_label in formatted
                assert anchor.description in formatted

    def test_holistic_rubrics_formatted(self):
        formatted = format_rubrics_for_prompt(HOLISTIC_RUBRICS)
        for rubric in HOLISTIC_RUBRICS:
            assert f"### {rubric.dimension}" in formatted


class TestFormatDimensionFeedback:
    def test_identifies_weakest_dimension(self):
        scores = {
            "synthesis_quality": 0.3,
            "citation_accuracy": 0.8,
            "coherence": 0.7,
        }
        feedback = format_dimension_feedback(scores, SECTION_RUBRICS)
        assert "Weakest dimension: synthesis_quality" in feedback

    def test_empty_scores_returns_empty(self):
        feedback = format_dimension_feedback({}, SECTION_RUBRICS)
        assert feedback == ""

    def test_provides_next_level_guidance(self):
        scores = {"synthesis_quality": 0.35}
        feedback = format_dimension_feedback(scores, SECTION_RUBRICS)
        # Should mention the current level and the next level to aim for
        assert "0.3-0.4" in feedback
        assert "0.5-0.6" in feedback

    def test_top_score_has_no_next_level(self):
        scores = {"synthesis_quality": 0.95}
        feedback = format_dimension_feedback(scores, SECTION_RUBRICS)
        assert "0.9-1.0" in feedback


class TestDimensionCoverage:
    def test_all_section_dimensions_have_rubrics(self):
        expected = {
            "synthesis_quality", "citation_accuracy", "coherence",
            "connection_to_neighbors", "completeness", "balance",
            "structural_variety", "paragraph_cohesion",
        }
        assert SECTION_DIMENSION_NAMES == expected

    def test_all_holistic_dimensions_have_rubrics(self):
        expected = {
            "narrative_arc", "redundancy", "transitions",
            "intro_conclusion_alignment", "balance", "completeness",
            "prose_flow",
        }
        assert HOLISTIC_DIMENSION_NAMES == expected

    def test_each_rubric_has_five_anchors(self):
        for rubric in SECTION_RUBRICS + HOLISTIC_RUBRICS:
            assert len(rubric.anchors) == 5, f"{rubric.dimension} has {len(rubric.anchors)} anchors"

    def test_anchor_ranges_are_ordered(self):
        for rubric in SECTION_RUBRICS + HOLISTIC_RUBRICS:
            for i in range(len(rubric.anchors) - 1):
                assert rubric.anchors[i].high < rubric.anchors[i + 1].low, (
                    f"{rubric.dimension}: anchor {i} overlaps with {i+1}"
                )


class TestIssuesSortedBySeverity:
    def test_issues_sorted_by_severity(self):
        issues = [
            CritiqueIssue(severity=CritiqueSeverity.MINOR, location="s1", description="minor issue"),
            CritiqueIssue(severity=CritiqueSeverity.CRITICAL, location="s2", description="critical issue"),
            CritiqueIssue(severity=CritiqueSeverity.MAJOR, location="s3", description="major issue"),
        ]
        formatted = _format_issues(issues)
        lines = formatted.strip().split("\n")
        assert "[critical]" in lines[0]
        assert "[major]" in lines[1]
        assert "[minor]" in lines[2]

    def test_empty_issues(self):
        formatted = _format_issues([])
        assert formatted == "No specific issues identified."
