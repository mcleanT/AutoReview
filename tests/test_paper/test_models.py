# tests/test_paper/test_models.py
"""Tests for paper benchmark models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from paper.models import (
    RunRegistry,
    TopicEntry,
    TopicsConfig,
    expand_run_matrix,
    load_topics,
)


def _make_topic(
    id: str = "test_topic",
    tier: str = "B",
    domain: str = "cs_ai",
    conditions: list[str] | None = None,
    date_range: str | None = None,
    ablation: bool = False,
) -> dict[str, Any]:
    return {
        "id": id,
        "title": f"Test topic {id}",
        "domain": domain,
        "tier": tier,
        "reference": {
            "doi": "10.1234/test",
            "title": "Test Reference",
            "year": 2023,
            "citation_count": 100,
            "pdf_path": "paper/references/test.pdf",
        },
        "conditions": conditions or ["end_to_end"],
        "date_range": date_range,
        "ablation": ablation,
    }


class TestTopicEntry:
    def test_basic_creation(self) -> None:
        topic = TopicEntry(**_make_topic())
        assert topic.id == "test_topic"
        assert topic.tier == "B"
        assert topic.conditions == ["end_to_end"]

    def test_tier_a_with_date_range(self) -> None:
        topic = TopicEntry(
            **_make_topic(
                tier="A", conditions=["end_to_end", "retrieval_controlled"], date_range="-2017"
            )
        )
        assert topic.tier == "A"
        assert topic.date_range == "-2017"
        assert "retrieval_controlled" in topic.conditions

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(Exception):
            TopicEntry(**_make_topic(tier="C"))


class TestTopicsConfig:
    def test_load_from_dict(self) -> None:
        config = TopicsConfig(
            metadata={"created": "2026-03-17"},
            topics=[TopicEntry(**_make_topic())],
        )
        assert len(config.topics) == 1

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
metadata:
  created: "2026-03-17"
topics:
  - id: "test_topic"
    title: "Test topic"
    domain: cs_ai
    tier: B
    reference:
      doi: "10.1234/test"
      title: "Test Ref"
      year: 2023
      citation_count: 100
      pdf_path: "paper/references/test.pdf"
    conditions: [end_to_end]
"""
        yaml_path = tmp_path / "topics.yaml"
        yaml_path.write_text(yaml_content)
        config = load_topics(yaml_path)
        assert len(config.topics) == 1
        assert config.topics[0].id == "test_topic"


class TestRunMatrix:
    def test_expand_basic(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 topic x 1 model x medium x end_to_end = 1 run
        assert len(matrix) == 1
        key = matrix[0]
        assert key == ("test_topic", "claude-sonnet-4-6", "medium", "end_to_end")

    def test_tier_a_expands_conditions(self) -> None:
        topics = [
            TopicEntry(**_make_topic(tier="A", conditions=["end_to_end", "retrieval_controlled"]))
        ]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 topic x 1 model x medium x 2 conditions = 2
        assert len(matrix) == 2

    def test_ablation_topics_expand(self) -> None:
        topics = [TopicEntry(**_make_topic(ablation=True))]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models)
        # 1 end_to_end + 4 ablation conditions (sonnet only) = 5
        conditions = [key[3] for key in matrix]
        assert "end_to_end" in conditions
        assert "no_critique_loops" in conditions

    def test_depth_runs_added_for_sonnet(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6", "claude-opus-4-6"]
        matrix = expand_run_matrix(topics, models, include_depth=True)
        depths = [(k[1], k[2]) for k in matrix]
        # Sonnet gets low + deep in addition to medium; Opus gets medium only
        assert ("claude-sonnet-4-6", "low") in depths
        assert ("claude-sonnet-4-6", "deep") in depths
        assert ("claude-opus-4-6", "low") not in depths

    def test_dedup_removes_duplicates(self) -> None:
        topics = [TopicEntry(**_make_topic())]
        models = ["claude-sonnet-4-6"]
        matrix = expand_run_matrix(topics, models, include_depth=True)
        # Should not have duplicate (topic, sonnet, medium, end_to_end)
        assert len(matrix) == len(set(matrix))


class TestRunRegistry:
    def test_empty_registry(self) -> None:
        reg = RunRegistry()
        assert len(reg.runs) == 0
        assert not reg.is_completed("topic|model|medium|end_to_end")

    def test_register_and_check(self) -> None:
        reg = RunRegistry()
        reg.register_start("topic|model|medium|end_to_end")
        assert not reg.is_completed("topic|model|medium|end_to_end")

        reg.register_complete(
            "topic|model|medium|end_to_end",
            output_dir="out/dir",
            review_path="out/dir/review.md",
        )
        assert reg.is_completed("topic|model|medium|end_to_end")

    def test_save_and_load(self, tmp_path: Path) -> None:
        reg = RunRegistry()
        reg.register_complete("k", output_dir="d", review_path="r")
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = RunRegistry.load(path)
        assert loaded.is_completed("k")

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        reg = RunRegistry.load(tmp_path / "nonexistent.json")
        assert len(reg.runs) == 0
