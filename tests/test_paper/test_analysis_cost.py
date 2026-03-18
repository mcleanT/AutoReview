"""Tests for Analysis 8: Cost-Quality Tradeoff."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture with cost_usd and overall_score > 0."""
    return pd.DataFrame(
        {
            "model": ["gpt-4o", "gpt-4o", "claude-3-5-sonnet", "claude-3-5-sonnet"],
            "topic_id": ["t1", "t2", "t1", "t2"],
            "cost_usd": [0.50, 0.75, 0.30, 0.40],
            "overall_score": [0.80, 0.85, 0.78, 0.82],
            "citation_f1": [0.70, 0.75, 0.65, 0.72],
        }
    )


class TestComputeCostEfficiency:
    def test_columns_added(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.cost_analysis import compute_cost_efficiency

        eff = compute_cost_efficiency(sample_df)
        assert "cost_per_quality_point" in eff.columns
        assert "cost_per_citation_f1" in eff.columns

    def test_cost_per_quality_point(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.cost_analysis import compute_cost_efficiency

        eff = compute_cost_efficiency(sample_df)
        assert "cost_per_quality_point" in eff.columns
        assert all(eff["cost_per_quality_point"] >= 0)

    def test_cost_per_citation_f1_values(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.cost_analysis import compute_cost_efficiency

        eff = compute_cost_efficiency(sample_df)
        assert all(eff["cost_per_citation_f1"] >= 0)
        # Spot-check: first row 0.50 / 0.70 ≈ 0.714
        assert abs(eff["cost_per_citation_f1"].iloc[0] - 0.50 / 0.70) < 1e-6

    def test_zero_overall_score_returns_zero(self) -> None:
        from paper.analysis.cost_analysis import compute_cost_efficiency

        df = pd.DataFrame(
            {
                "model": ["m1"],
                "cost_usd": [1.0],
                "overall_score": [0.0],
                "citation_f1": [0.0],
            }
        )
        eff = compute_cost_efficiency(df)
        assert eff["cost_per_quality_point"].iloc[0] == 0.0
        assert eff["cost_per_citation_f1"].iloc[0] == 0.0

    def test_original_df_not_mutated(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.cost_analysis import compute_cost_efficiency

        original_cols = set(sample_df.columns)
        compute_cost_efficiency(sample_df)
        assert set(sample_df.columns) == original_cols


class TestPlotCostQualityScatter:
    def test_scatter_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.cost_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "cost_quality_scatter.pdf").exists()

    def test_output_dir_created(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.cost_analysis import plot_cost_quality_scatter

        out = tmp_path / "nested" / "dir" / "scatter.pdf"
        plot_cost_quality_scatter(sample_df, out)
        assert out.exists()
