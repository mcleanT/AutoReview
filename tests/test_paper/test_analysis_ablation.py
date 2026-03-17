# tests/test_paper/test_analysis_ablation.py
"""Tests for Analysis 4: Component Ablation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_ablation_df() -> pd.DataFrame:
    """Synthetic DataFrame with end_to_end baseline + two ablation conditions."""
    rows = []
    topics = ["topic_a", "topic_b", "topic_c", "topic_d", "topic_e"]
    model = "gpt-4o"
    depth = "medium"

    rng = np.random.default_rng(42)

    for topic in topics:
        # Baseline (end_to_end)
        base_score = rng.uniform(0.6, 0.9)
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "end_to_end",
                "overall_score": base_score,
                "citation_f1": base_score - 0.05,
                "synthesis_score": base_score + 0.02,
                "topic_coverage": base_score - 0.01,
                "writing_quality": base_score + 0.03,
                "tier": "A",
                "domain": "cs_ai",
            }
        )
        # Ablation: no_critique_loops — slightly worse
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "no_critique_loops",
                "overall_score": base_score - 0.1,
                "citation_f1": base_score - 0.12,
                "synthesis_score": base_score - 0.08,
                "topic_coverage": base_score - 0.09,
                "writing_quality": base_score - 0.07,
                "tier": "A",
                "domain": "cs_ai",
            }
        )
        # Ablation: no_outline — also worse
        rows.append(
            {
                "topic_id": topic,
                "model": model,
                "depth": depth,
                "condition": "no_outline",
                "overall_score": base_score - 0.15,
                "citation_f1": base_score - 0.13,
                "synthesis_score": base_score - 0.11,
                "topic_coverage": base_score - 0.14,
                "writing_quality": base_score - 0.10,
                "tier": "A",
                "domain": "cs_ai",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return _make_ablation_df()


class TestComputeAblationDeltas:
    def test_ablation_conditions_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        deltas = compute_ablation_deltas(sample_df)
        assert "no_critique_loops" in deltas
        assert "no_outline" in deltas

    def test_end_to_end_not_in_deltas(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        deltas = compute_ablation_deltas(sample_df)
        assert "end_to_end" not in deltas

    def test_retrieval_controlled_not_in_deltas(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        # Add a retrieval_controlled row — should also be excluded
        extra = sample_df.copy()
        extra_row = extra[extra["condition"] == "end_to_end"].iloc[0].copy()
        extra_row["condition"] = "retrieval_controlled"
        df_with_ctrl = pd.concat([sample_df, pd.DataFrame([extra_row])], ignore_index=True)

        deltas = compute_ablation_deltas(df_with_ctrl)
        assert "retrieval_controlled" not in deltas

    def test_overall_score_key_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        deltas = compute_ablation_deltas(sample_df)
        assert "overall_score" in deltas["no_critique_loops"]

    def test_deltas_are_negative_for_worse_conditions(self, sample_df: pd.DataFrame) -> None:
        """Ablated conditions are intentionally worse, so deltas should be negative."""
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        deltas = compute_ablation_deltas(sample_df)
        assert deltas["no_critique_loops"]["overall_score"] < 0
        assert deltas["no_outline"]["overall_score"] < 0

    def test_no_baseline_returns_nan(self) -> None:
        """When no baseline exists for a topic, delta should be NaN (condition skipped)."""
        from paper.analysis.ablation_analysis import compute_ablation_deltas

        # Only ablation rows, no end_to_end baseline
        df = pd.DataFrame(
            [
                {
                    "topic_id": "t1",
                    "model": "gpt-4o",
                    "depth": "medium",
                    "condition": "no_outline",
                    "overall_score": 0.5,
                    "citation_f1": 0.4,
                    "synthesis_score": 0.6,
                    "topic_coverage": 0.5,
                    "writing_quality": 0.5,
                }
            ]
        )
        deltas = compute_ablation_deltas(df)
        # no_outline exists but no baseline match → mean of empty list → nan
        assert "no_outline" in deltas
        assert np.isnan(deltas["no_outline"]["overall_score"])


class TestPlotAblationHeatmap:
    def test_heatmap_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.ablation_analysis import compute_ablation_deltas, plot_ablation_heatmap

        deltas = compute_ablation_deltas(sample_df)
        out = tmp_path / "ablation_heatmap.pdf"
        plot_ablation_heatmap(deltas, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_heatmap_empty_deltas_no_crash(self, tmp_path: Path) -> None:
        from paper.analysis.ablation_analysis import plot_ablation_heatmap

        # Empty deltas dict should not raise
        plot_ablation_heatmap({}, tmp_path / "empty_heatmap.pdf")


class TestAblationMain:
    def test_heatmap_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.ablation_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "ablation_heatmap.pdf").exists()

    def test_json_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.ablation_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "ablation_analysis.json").exists()

    def test_json_has_deltas_key(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        import json

        from paper.analysis.ablation_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        data = json.loads((tmp_path / "out" / "ablation_analysis.json").read_text())
        assert "deltas" in data
        assert "no_critique_loops" in data["deltas"]
