"""Tests for per-dimension critique gates."""

from __future__ import annotations

from autoreview.critique.dimension_gates import (
    DEFAULT_HOLISTIC_GATES,
    DEFAULT_SECTION_GATES,
    check_dimension_gates,
)


class TestCheckDimensionGates:
    def test_all_above_minimums_passes(self):
        scores = {"synthesis_quality": 0.80, "citation_accuracy": 0.75}
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is True
        assert result.failures == []

    def test_synthesis_below_minimum_fails(self):
        scores = {"synthesis_quality": 0.50, "citation_accuracy": 0.75}
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is False
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.dimension == "synthesis_quality"
        assert failure.score == 0.50
        assert failure.minimum == 0.65

    def test_missing_dimension_ignored(self):
        # citation_accuracy not in scores — gate should be skipped, not fail
        scores = {"synthesis_quality": 0.80}
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is True
        assert result.failures == []

    def test_multiple_failures_reported(self):
        scores = {
            "synthesis_quality": 0.40,
            "citation_accuracy": 0.30,
            "coherence": 0.90,
        }
        gates = {"synthesis_quality": 0.65, "citation_accuracy": 0.60}
        result = check_dimension_gates(scores, gates)
        assert result.passed is False
        assert len(result.failures) == 2
        failed_dims = {f.dimension for f in result.failures}
        assert failed_dims == {"synthesis_quality", "citation_accuracy"}

    def test_default_section_gates_include_synthesis(self):
        assert "synthesis_quality" in DEFAULT_SECTION_GATES
        assert DEFAULT_SECTION_GATES["synthesis_quality"] == 0.65

    def test_default_holistic_gates_include_narrative(self):
        assert "narrative_arc" in DEFAULT_HOLISTIC_GATES
        assert DEFAULT_HOLISTIC_GATES["narrative_arc"] == 0.60
