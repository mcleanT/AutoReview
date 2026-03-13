"""Tests for query expansion prompt builders."""

from __future__ import annotations

from autoreview.llm.prompts.query_expansion import build_targeted_query_expansion_prompt


class TestTargetedQueryExpansionPrompt:
    def test_includes_uncovered_topics(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["neuroinflammation", "blood-brain barrier"],
            domain="biomedical",
            date_range="2015-2025",
        )
        assert "neuroinflammation" in prompt
        assert "blood-brain barrier" in prompt

    def test_includes_domain_and_date(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["topic_a"],
            domain="cs_ai",
            date_range="2018-2025",
        )
        assert "cs_ai" in prompt
        assert "2018-2025" in prompt

    def test_returns_nonempty_string(self):
        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=["topic"],
            domain="biomedical",
        )
        assert len(prompt) > 100
