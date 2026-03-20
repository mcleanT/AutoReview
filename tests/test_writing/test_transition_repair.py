"""Tests for the transition_repair module."""

from __future__ import annotations

import pytest

from autoreview.writing.transition_repair import (
    SectionBoundary,
    build_transition_repair_prompt,
    extract_boundary,
)


class TestExtractBoundary:
    def test_extracts_ending_and_beginning(self) -> None:
        """Long sections: last 200 chars contain 'concludes', first chars contain 'Moving to'."""
        section_a = "A" * 800 + " This section concludes with a summary of findings."
        section_b = "Moving to the next topic, we examine further details. " + "B" * 800

        boundary = extract_boundary(
            section_a,
            section_b,
            context_chars=200,
            section_a_id="1",
            section_b_id="2",
        )

        assert isinstance(boundary, SectionBoundary)
        assert boundary.section_a_id == "1"
        assert boundary.section_b_id == "2"

        # ending_text should be the last 200 chars of section_a
        assert len(boundary.ending_text) == 200
        assert "concludes" in boundary.ending_text

        # opening_text should be the first 200 chars of section_b
        assert len(boundary.opening_text) == 200
        assert boundary.opening_text.startswith("Moving to")

    def test_short_sections_use_full_text(self) -> None:
        """Short sections (shorter than context_chars) are used in full."""
        short_a = "Short ending text."
        short_b = "Short opening text."

        boundary = extract_boundary(
            short_a,
            short_b,
            context_chars=500,
        )

        assert boundary.ending_text == short_a
        assert boundary.opening_text == short_b

    def test_default_ids_are_empty_strings(self) -> None:
        boundary = extract_boundary("end text", "start text")
        assert boundary.section_a_id == ""
        assert boundary.section_b_id == ""

    def test_exact_length_boundary(self) -> None:
        """Section exactly equal to context_chars uses full text (not truncated)."""
        text = "x" * 500
        boundary = extract_boundary(text, text, context_chars=500)
        assert boundary.ending_text == text
        assert boundary.opening_text == text

    def test_frozen_dataclass(self) -> None:
        """SectionBoundary is immutable."""
        boundary = extract_boundary("a", "b")
        with pytest.raises(AttributeError):
            boundary.ending_text = "new"  # type: ignore[misc]


class TestBuildTransitionRepairPrompt:
    def test_prompt_contains_boundary_texts(self) -> None:
        boundary = SectionBoundary(
            section_a_id="intro",
            section_b_id="methods",
            ending_text="This concludes the introduction.",
            opening_text="We now describe the methods.",
        )
        prompt = build_transition_repair_prompt(boundary)

        assert "This concludes the introduction." in prompt
        assert "We now describe the methods." in prompt

    def test_prompt_contains_section_ids(self) -> None:
        boundary = SectionBoundary(
            section_a_id="1",
            section_b_id="2",
            ending_text="end",
            opening_text="start",
        )
        prompt = build_transition_repair_prompt(boundary)
        assert "1" in prompt
        assert "2" in prompt

    def test_prompt_is_non_empty_string(self) -> None:
        boundary = SectionBoundary(
            section_a_id="",
            section_b_id="",
            ending_text="some end",
            opening_text="some start",
        )
        prompt = build_transition_repair_prompt(boundary)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_requests_bridge_sentences(self) -> None:
        boundary = SectionBoundary(
            section_a_id="",
            section_b_id="",
            ending_text="end",
            opening_text="start",
        )
        prompt = build_transition_repair_prompt(boundary)
        assert "bridge" in prompt.lower() or "1" in prompt
