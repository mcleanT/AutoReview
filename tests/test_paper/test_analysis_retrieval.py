# tests/test_paper/test_analysis_retrieval.py
"""Tests for Analysis 5: Retrieval vs Synthesis Decomposition."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_retrieval_df() -> pd.DataFrame:
    """Synthetic DataFrame with Tier A topics, end_to_end and retrieval_controlled."""
    rows = []
    topics = ["topic_a", "topic_b", "topic_c", "topic_d", "topic_e"]
    model = "gpt-4o"
    depth = "medium"

    rng = np.random.default_rng(7)

    for topic in topics:
        base_score = rng.uniform(0.6, 0.85)

        # end_to_end condition
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "end_to_end",
                "overall_score": base_score,
                "citation_f1": base_score - 0.05,
                "synthesis_score": base_score + 0.03,
                "topic_coverage": base_score,
                "writing_quality": base_score + 0.01,
                "tier": "A",
                "domain": "cs_ai",
            }
        )
        # retrieval_controlled condition — controlled retrieval, different synthesis
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "retrieval_controlled",
                "overall_score": base_score + 0.05,
                "citation_f1": base_score + 0.08,
                "synthesis_score": base_score + 0.02,
                "topic_coverage": base_score + 0.04,
                "writing_quality": base_score + 0.01,
                "tier": "A",
                "domain": "cs_ai",
            }
        )

    # Add some Tier B rows (should be excluded)
    for topic in ["tier_b_topic_1", "tier_b_topic_2"]:
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "end_to_end",
                "overall_score": 0.7,
                "citation_f1": 0.65,
                "synthesis_score": 0.72,
                "topic_coverage": 0.68,
                "writing_quality": 0.71,
                "tier": "B",
                "domain": "bio",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return _make_retrieval_df()


class TestComputeConditionComparison:
    def test_both_conditions_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        comp = compute_condition_comparison(sample_df)
        assert "end_to_end" in comp
        assert "retrieval_controlled" in comp

    def test_metrics_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        comp = compute_condition_comparison(sample_df)
        for condition in ["end_to_end", "retrieval_controlled"]:
            assert "overall_score" in comp[condition]
            assert "synthesis_score" in comp[condition]

    def test_stats_keys(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        comp = compute_condition_comparison(sample_df)
        entry = comp["end_to_end"]["overall_score"]
        assert "mean" in entry
        assert "std" in entry
        assert "n" in entry

    def test_tier_b_excluded(self, sample_df: pd.DataFrame) -> None:
        """Tier B topics should not inflate counts."""
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        comp = compute_condition_comparison(sample_df)
        # 5 Tier A topics for end_to_end
        assert comp["end_to_end"]["overall_score"]["n"] == 5

    def test_means_are_finite(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        comp = compute_condition_comparison(sample_df)
        for condition in ["end_to_end", "retrieval_controlled"]:
            for metric_stats in comp[condition].values():
                assert np.isfinite(metric_stats["mean"])

    def test_empty_df_no_crash(self) -> None:
        """Empty DataFrame should return conditions with empty stats."""
        from paper.analysis.retrieval_decomposition import compute_condition_comparison

        df = pd.DataFrame(
            columns=[
                "tier",
                "condition",
                "topic_id",
                "model",
                "depth",
                "overall_score",
                "citation_f1",
                "synthesis_score",
                "topic_coverage",
                "writing_quality",
            ]
        )
        comp = compute_condition_comparison(df)
        # Both conditions should be present in output (with NaN means)
        assert "end_to_end" in comp
        assert "retrieval_controlled" in comp


class TestPlotConditionBar:
    def test_bar_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.retrieval_decomposition import (
            compute_condition_comparison,
            plot_condition_bar,
        )

        comp = compute_condition_comparison(sample_df)
        out = tmp_path / "retrieval_decomposition_bar.pdf"
        plot_condition_bar(comp, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_bar_chart_empty_no_crash(self, tmp_path: Path) -> None:
        from paper.analysis.retrieval_decomposition import plot_condition_bar

        plot_condition_bar({}, tmp_path / "empty_bar.pdf")


class TestRetrievalMain:
    def test_bar_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.retrieval_decomposition import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "retrieval_decomposition_bar.pdf").exists()

    def test_json_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.retrieval_decomposition import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "retrieval_decomposition.json").exists()

    def test_json_structure(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.retrieval_decomposition import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        data = json.loads((tmp_path / "out" / "retrieval_decomposition.json").read_text())
        assert "condition_comparison" in data
        assert "end_to_end" in data["condition_comparison"]
        assert "retrieval_controlled" in data["condition_comparison"]
