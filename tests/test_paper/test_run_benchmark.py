# tests/test_paper/test_run_benchmark.py
"""Tests for benchmark orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paper.models import (
    ReferenceInfo,
    RunRegistry,
    TopicEntry,
    TopicsConfig,
    expand_run_matrix,
    make_run_key,
)
from paper.run_benchmark import _execute_runs, _find_corpus_snapshot, _is_corpus_run


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


class TestCorpusHelpers:
    def test_sonnet_e2e_is_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "end_to_end")
        assert _is_corpus_run(key) is True

    def test_sonnet_rc_is_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "retrieval_controlled")
        assert _is_corpus_run(key) is True

    def test_opus_e2e_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-opus-4-6", "medium", "end_to_end")
        assert _is_corpus_run(key) is False

    def test_sonnet_ablation_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "no_evidence_chains")
        assert _is_corpus_run(key) is False

    def test_sonnet_depth_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "deep", "end_to_end")
        assert _is_corpus_run(key) is False


class TestFindCorpusSnapshot:
    def test_returns_snapshot_path_when_corpus_completed(self, tmp_path: Path) -> None:
        registry = RunRegistry()
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_end_to_end")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            output_dir=output_dir,
            review_path="r.md",
        )
        snap_dir = Path(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        snap_file = snap_dir / "20260317T120000_gap_search.json"
        snap_file.write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is not None
        assert result.name == "20260317T120000_gap_search.json"

    def test_returns_clustering_snapshot_for_no_comprehensiveness(self, tmp_path: Path) -> None:
        registry = RunRegistry()
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_end_to_end")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            output_dir=output_dir,
            review_path="r.md",
        )
        snap_dir = Path(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "20260317T120000_clustering.json").write_text("{}")
        (snap_dir / "20260317T120100_gap_search.json").write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "no_comprehensiveness")
        assert result is not None
        assert "clustering" in result.name

    def test_returns_none_when_no_corpus_run(self) -> None:
        registry = RunRegistry()
        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is None

    def test_returns_none_when_corpus_failed(self) -> None:
        registry = RunRegistry()
        registry.register_failure(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            error="boom",
        )
        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is None

    def test_rc_condition_looks_up_rc_corpus(self, tmp_path: Path) -> None:
        registry = RunRegistry()
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_retrieval_controlled")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "retrieval_controlled"),
            output_dir=output_dir,
            review_path="r.md",
        )
        snap_dir = Path(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "20260317T120000_gap_search.json").write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "retrieval_controlled")
        assert result is not None


class TestPhaseSeparation:
    def test_corpus_runs_sorted_first(self) -> None:
        runs: list[tuple[str, str, str, str]] = [
            ("t1", "claude-opus-4-6", "medium", "end_to_end"),
            ("t1", "claude-sonnet-4-6", "medium", "end_to_end"),
            ("t1", "claude-haiku-4-5", "medium", "end_to_end"),
            ("t1", "claude-sonnet-4-6", "medium", "no_evidence_chains"),
            ("t1", "claude-sonnet-4-6", "deep", "end_to_end"),
        ]
        corpus = [k for k in runs if _is_corpus_run(k)]
        writing = [k for k in runs if not _is_corpus_run(k)]

        assert len(corpus) == 1
        assert corpus[0][1] == "claude-sonnet-4-6"
        assert corpus[0][3] == "end_to_end"
        assert len(writing) == 4


class TestSharedCorpusCostEstimate:
    def test_writing_only_runs_cost_less(self, sample_topics: TopicsConfig) -> None:
        from paper.run_benchmark import estimate_cost

        models = ["claude-opus-4-6", "claude-sonnet-4-6"]
        matrix = expand_run_matrix(sample_topics.topics, models, include_depth=True)

        full_cost = estimate_cost(matrix, shared_corpus=False)
        optimized_cost = estimate_cost(matrix, shared_corpus=True)

        assert optimized_cost < full_cost

    def test_corpus_only_same_cost_either_way(self) -> None:
        from paper.run_benchmark import estimate_cost

        corpus_only: list[tuple[str, str, str, str]] = [
            ("t1", "claude-sonnet-4-6", "medium", "end_to_end")
        ]
        assert estimate_cost(corpus_only, shared_corpus=True) == estimate_cost(
            corpus_only, shared_corpus=False
        )


class TestForkLogic:
    """Integration tests for the shared-corpus fork path in _execute_runs."""

    def _make_registry_with_corpus(self, tmp_path: Path, topic_id: str = "topic_a") -> RunRegistry:
        """Create a registry with a completed Sonnet corpus run and snapshot files."""
        registry = RunRegistry()
        output_dir = str(tmp_path / topic_id / "claude-sonnet-4-6_medium_end_to_end")
        registry.register_complete(
            make_run_key(topic_id, "claude-sonnet-4-6", "medium", "end_to_end"),
            output_dir=output_dir,
            review_path=str(tmp_path / topic_id / "review.md"),
        )
        # Create snapshot files
        snap_dir = Path(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        # Write minimal valid KB JSON for load_snapshot
        minimal_kb = (
            '{"topic": "Test Topic", "domain": "general", "output_dir": "' + output_dir + '"}'
        )
        (snap_dir / "20260317T120000_clustering.json").write_text(minimal_kb)
        (snap_dir / "20260317T120100_gap_search.json").write_text(minimal_kb)
        return registry

    @patch("autoreview.pipeline.runner.run_pipeline", new_callable=AsyncMock)
    @patch("autoreview.llm.factory.create_llm_provider")
    @patch("autoreview.config.load_config")
    @patch("autoreview.models.knowledge_base.KnowledgeBase.load_snapshot")
    def test_opus_e2e_forks_from_corpus_snapshot(
        self,
        mock_load_snapshot: MagicMock,
        mock_load_config: MagicMock,
        mock_create_llm: MagicMock,
        mock_run_pipeline: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """An Opus e2e run should fork from the Sonnet corpus gap_search snapshot."""

        registry = self._make_registry_with_corpus(tmp_path)

        # Mock config
        mock_config = MagicMock()
        mock_config.writing.depth = "medium"
        mock_load_config.return_value = mock_config

        # Mock load_snapshot to return a KB mock
        output_dir = tmp_path / "topic_a" / "claude-opus-4-6_medium_end_to_end"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "review.md").write_text("# Review")

        mock_kb = MagicMock()
        mock_kb.total_tokens.return_value = {"input_tokens": 100, "output_tokens": 50}
        mock_kb.output_dir = str(output_dir)
        mock_load_snapshot.return_value = mock_kb
        mock_run_pipeline.return_value = mock_kb

        topic_lookup = {
            "topic_a": MagicMock(
                id="topic_a",
                title="Test Topic",
                domain="general",
                date_range=None,
                reference=MagicMock(pdf_path="ref.pdf"),
            ),
        }

        runs = [("topic_a", "claude-opus-4-6", "medium", "end_to_end")]

        asyncio.run(_execute_runs(runs, topic_lookup, registry, tmp_path, max_concurrent=2))

        # Verify run_pipeline was called with start_from="outline"
        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args
        assert call_kwargs.kwargs.get("start_from") == "outline" or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "outline"
        )

    @patch("autoreview.pipeline.runner.run_pipeline", new_callable=AsyncMock)
    @patch("autoreview.llm.factory.create_llm_provider")
    @patch("autoreview.config.load_config")
    @patch("autoreview.models.knowledge_base.KnowledgeBase.load_snapshot")
    def test_no_passage_mining_passes_skip_nodes(
        self,
        mock_load_snapshot: MagicMock,
        mock_load_config: MagicMock,
        mock_create_llm: MagicMock,
        mock_run_pipeline: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """no_passage_mining fork should pass skip_nodes={'passage_search'}."""

        registry = self._make_registry_with_corpus(tmp_path)

        mock_config = MagicMock()
        mock_config.writing.depth = "medium"
        mock_load_config.return_value = mock_config

        output_dir = tmp_path / "topic_a" / "claude-sonnet-4-6_medium_no_passage_mining"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "review.md").write_text("# Review")

        mock_kb = MagicMock()
        mock_kb.total_tokens.return_value = {"input_tokens": 100, "output_tokens": 50}
        mock_kb.output_dir = str(output_dir)
        mock_load_snapshot.return_value = mock_kb
        mock_run_pipeline.return_value = mock_kb

        topic_lookup = {
            "topic_a": MagicMock(
                id="topic_a",
                title="Test Topic",
                domain="general",
                date_range=None,
                reference=MagicMock(pdf_path="ref.pdf"),
            ),
        }

        runs = [("topic_a", "claude-sonnet-4-6", "medium", "no_passage_mining")]

        asyncio.run(_execute_runs(runs, topic_lookup, registry, tmp_path, max_concurrent=2))

        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args
        # Check skip_nodes contains passage_search
        skip_nodes = call_kwargs.kwargs.get("skip_nodes") or (
            call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        )
        assert skip_nodes == {"passage_search"}

    @patch("autoreview.pipeline.runner.run_pipeline", new_callable=AsyncMock)
    @patch("autoreview.llm.factory.create_llm_provider")
    @patch("autoreview.config.load_config")
    def test_no_corpus_falls_back_to_full_pipeline(
        self,
        mock_load_config: MagicMock,
        mock_create_llm: MagicMock,
        mock_run_pipeline: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """When no corpus snapshot exists, should fall back to full pipeline."""

        registry = RunRegistry()  # Empty — no corpus runs

        mock_config = MagicMock()
        mock_config.writing.depth = "medium"
        mock_load_config.return_value = mock_config

        output_dir = tmp_path / "topic_a" / "claude-opus-4-6_medium_end_to_end"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "review.md").write_text("# Review")

        mock_kb = MagicMock()
        mock_kb.total_tokens.return_value = {"input_tokens": 100, "output_tokens": 50}
        mock_kb.output_dir = str(output_dir)
        mock_run_pipeline.return_value = mock_kb

        topic_lookup = {
            "topic_a": MagicMock(
                id="topic_a",
                title="Test Topic",
                domain="general",
                date_range=None,
                reference=MagicMock(pdf_path="ref.pdf"),
            ),
        }

        runs = [("topic_a", "claude-opus-4-6", "medium", "end_to_end")]

        asyncio.run(_execute_runs(runs, topic_lookup, registry, tmp_path, max_concurrent=2))

        # Should NOT have start_from="outline" — full pipeline
        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args
        start_from = call_kwargs.kwargs.get("start_from")
        assert start_from is None  # Full pipeline, no start_from
