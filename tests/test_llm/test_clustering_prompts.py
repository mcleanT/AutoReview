"""Tests for clustering prompt builders."""
from __future__ import annotations

from autoreview.llm.prompts.clustering import build_retry_gap_queries_prompt


class TestRetryGapQueriesPrompt:
    def test_includes_remaining_gaps(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[
                {"expected_topic": "Drug resistance mechanisms", "current_coverage": "No papers found"},
            ],
            previous_queries=["drug resistance", "antimicrobial resistance"],
        )
        assert "Drug resistance mechanisms" in prompt
        assert "No papers found" in prompt

    def test_includes_previous_queries(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[{"expected_topic": "Topic A", "current_coverage": "Sparse"}],
            previous_queries=["query_1", "query_2"],
        )
        assert "query_1" in prompt
        assert "query_2" in prompt

    def test_instructs_alternative_terminology(self):
        prompt = build_retry_gap_queries_prompt(
            remaining_gaps=[{"expected_topic": "X", "current_coverage": "None"}],
            previous_queries=["old query"],
        )
        # Should instruct to use different terminology
        assert "alternative" in prompt.lower() or "different" in prompt.lower() or "synonym" in prompt.lower()
