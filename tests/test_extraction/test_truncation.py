"""Tests for section-aware text truncation."""

from __future__ import annotations

from autoreview.config.models import SectionTruncationConfig
from autoreview.extraction.truncation import (
    _head_tail_truncate,
    parse_sections,
    section_aware_truncate,
)


class TestParseSections:
    def test_numbered_headings(self):
        text = (
            "Some preamble text.\n"
            "1. Introduction\n"
            "This is the intro.\n"
            "2. Methods\n"
            "These are methods.\n"
            "3. Results\n"
            "These are results.\n"
        )
        sections = parse_sections(text)
        assert len(sections) == 3
        assert sections[0].name == "Introduction"
        assert sections[1].name == "Methods"
        assert sections[2].name == "Results"

    def test_roman_numeral_headings(self):
        text = (
            "I. Introduction\n"
            "Intro text.\n"
            "II. Methods\n"
            "Methods text.\n"
            "III. Results\n"
            "Results text.\n"
        )
        sections = parse_sections(text)
        assert len(sections) == 3
        assert sections[0].name == "Introduction"
        assert sections[1].name == "Methods"
        assert sections[2].name == "Results"

    def test_bare_uppercase_headings(self):
        text = (
            "INTRODUCTION\n"
            "Some intro text here.\n"
            "METHODS\n"
            "Some methods here.\n"
            "RESULTS\n"
            "Some results.\n"
            "DISCUSSION\n"
            "Discussion text.\n"
        )
        sections = parse_sections(text)
        assert len(sections) >= 4

    def test_no_structure_returns_empty(self):
        text = "This is just a blob of text with no section headings at all."
        sections = parse_sections(text)
        assert sections == []

    def test_single_heading_returns_empty(self):
        text = "1. Introduction\nJust one section, not enough structure."
        sections = parse_sections(text)
        assert sections == []

    def test_section_text_content(self):
        text = (
            "1. Introduction\n"
            "Intro content here.\n"
            "2. Results\n"
            "Results content here.\n"
        )
        sections = parse_sections(text)
        assert "Intro content" in sections[0].text
        assert "Results content" in sections[1].text


class TestHeadTailTruncate:
    def test_short_text_unchanged(self):
        text = "Short text"
        assert _head_tail_truncate(text, 1000) == text

    def test_long_text_truncated(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        result = _head_tail_truncate(text, 200)
        assert "[...truncated middle section...]" in result
        assert result.startswith("A")
        assert result.endswith("C" * 100)

    def test_preserves_head_tail_ratio(self):
        text = "x" * 900
        result = _head_tail_truncate(text, 300)
        # Head = 300//3 = 100 chars, followed by "\n\n[...truncated middle section...]\n\n"
        head_part = result.split("\n\n[...truncated middle section...]")[0]
        assert len(head_part) == 300 // 3


class TestSectionAwareTruncate:
    def test_short_text_unchanged(self):
        config = SectionTruncationConfig()
        text = "Short text"
        assert section_aware_truncate(text, 1000, config) == text

    def test_drops_references(self):
        config = SectionTruncationConfig()
        # Use explicit + to avoid implicit string literal concatenation with * operator
        ref_lines = "Ref 1. Author A. Title. Journal 2020.\n" * 200
        text = (
            "1. Introduction\n"
            "Intro text is here.\n"
            "2. Results\n"
            "Results text is here.\n"
            "3. References\n"
            + ref_lines
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "Intro text" in result
        assert "Results text" in result
        assert "Ref 1. Author A" not in result

    def test_drops_acknowledgments(self):
        config = SectionTruncationConfig()
        text = (
            "1. Introduction\n"
            "Intro content.\n"
            "2. Results\n"
            "Results content.\n"
            "3. Acknowledgments\n"
            "Thanks to everyone.\n"
            "4. Discussion\n"
            "Discuss content.\n"
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "Thanks to everyone" not in result
        assert "Discuss content" in result

    def test_caps_introduction(self):
        config = SectionTruncationConfig(intro_max_chars=50)
        long_intro = "a" * 200  # lowercase to avoid heading false-positives
        text = (
            "1. Introduction\n"
            + long_intro + "\n"
            "2. Results\n"
            "Results text.\n"
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "[...introduction truncated...]" in result
        assert "Results text" in result

    def test_caps_methods(self):
        config = SectionTruncationConfig(methods_max_chars=50)
        long_methods = "m" * 200  # lowercase to avoid heading false-positives
        text = (
            "1. Introduction\n"
            "Intro content.\n"
            "2. Methods\n"
            + long_methods + "\n"
            "3. Results\n"
            "Results content.\n"
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "[...methods truncated...]" in result

    def test_drops_methods_when_zero(self):
        config = SectionTruncationConfig(methods_max_chars=0)
        text = (
            "1. Introduction\n"
            "Intro content.\n"
            "2. Methods\n"
            "Detailed methods here.\n"
            "3. Results\n"
            "Results content.\n"
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "Detailed methods" not in result
        assert "Intro" in result
        assert "Results" in result

    def test_fallback_to_head_tail_when_no_structure(self):
        config = SectionTruncationConfig()
        text = "x" * 10_000
        result = section_aware_truncate(text, 5000, config)
        assert "[...truncated middle section...]" in result
        assert len(result) <= 5100  # 5000 + marker text

    def test_fallback_when_still_over_max_after_filtering(self):
        config = SectionTruncationConfig()
        # Huge text with sections that are all kept
        long_intro = "i" * 50_000
        long_results = "r" * 50_000
        text = (
            "1. Introduction\n"
            + long_intro + "\n"
            "2. Results\n"
            + long_results + "\n"
        )
        result = section_aware_truncate(text, 10_000, config)
        # Should fall back to head/tail after section filtering is still too large
        assert len(result) <= 10_100

    def test_preserves_preamble(self):
        config = SectionTruncationConfig()
        text = (
            "Title: My Paper\nAbstract: This is the abstract.\n\n"
            "1. Introduction\n"
            "Intro content.\n"
            "2. Results\n"
            "Results content.\n"
        )
        result = section_aware_truncate(text, 100_000, config)
        assert "This is the abstract" in result

    def test_disabled_config_uses_head_tail(self):
        config = SectionTruncationConfig(enabled=False)
        text = "x" * 10_000
        # When disabled, extractor._truncate_full_text falls back to head/tail
        # but section_aware_truncate itself doesn't check enabled—that's the caller's job
        # This test just verifies the function still works
        result = section_aware_truncate(text, 5000, config)
        assert len(result) <= 5100
