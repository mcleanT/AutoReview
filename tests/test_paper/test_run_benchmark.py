# tests/test_paper/test_run_benchmark.py
"""Tests for benchmark orchestrator."""

from __future__ import annotations

import pytest

from paper.models import (
    ReferenceInfo,
    RunRegistry,
    TopicEntry,
    TopicsConfig,
    expand_run_matrix,
    make_run_key,
)


@pytest.fixture()
def sample_topics() -> TopicsConfig:
    return TopicsConfig(
        metadata={"created": "2026-03-17"},
        topics=[
            TopicEntry(
                id="topic_a",
                title="Topic A",
                domain="biomedical",
                tier="A",
                reference=ReferenceInfo(
                    doi="x", title="y", year=2017, citation_count=500, pdf_path="z"
                ),
                conditions=["end_to_end", "retrieval_controlled"],
                date_range="-2017",
                ablation=True,
            ),
            TopicEntry(
                id="topic_b",
                title="Topic B",
                domain="cs_ai",
                tier="B",
                reference=ReferenceInfo(
                    doi="x", title="y", year=2023, citation_count=100, pdf_path="z"
                ),
                conditions=["end_to_end"],
            ),
        ],
    )


class TestMatrixGeneration:
    def test_full_matrix_count(self, sample_topics: TopicsConfig) -> None:
        models = ["claude-sonnet-4-6", "claude-opus-4-6"]
        matrix = expand_run_matrix(
            sample_topics.topics, models, include_depth=True, include_ablation=True
        )
        # topic_a: 2 models x 2 conditions x medium = 4
        #        + sonnet x {low,deep} x end_to_end = 2 (depth)
        #        + sonnet x 4 ablation x medium = 4
        # topic_b: 2 models x 1 condition x medium = 2
        #        + sonnet x {low,deep} x end_to_end = 2 (depth)
        # Total = 4 + 2 + 4 + 2 + 2 = 14
        assert len(matrix) == 14

    def test_completed_runs_skipped(self, sample_topics: TopicsConfig) -> None:
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(sample_topics.topics, models)
        registry = RunRegistry()
        key = make_run_key("topic_b", "claude-sonnet-4-6", "medium", "end_to_end")
        registry.register_complete(key, output_dir="d", review_path="r")
        remaining = [k for k in matrix if not registry.is_completed(make_run_key(*k))]
        assert len(remaining) == len(matrix) - 1


class TestCostEstimate:
    def test_estimate_produces_output(self, sample_topics: TopicsConfig) -> None:
        from paper.run_benchmark import estimate_cost

        matrix = expand_run_matrix(sample_topics.topics, ["claude-sonnet-4-6"])
        cost = estimate_cost(matrix)
        assert cost > 0
