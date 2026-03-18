"""Tests for Analysis 7: Model tier comparison."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame with 3 model tiers x 3 topics, medium depth only.

    Includes cost_usd column as required by the spec.
    """
    models = ["claude-haiku-3", "claude-sonnet-4-6", "claude-opus-4"]
    topics = ["topic_a", "topic_b", "topic_c"]

    # Scores increase with model tier so Friedman test has signal
    base_scores = {
        "claude-haiku-3": 0.55,
        "claude-sonnet-4-6": 0.70,
        "claude-opus-4": 0.82,
    }
    costs = {
        "claude-haiku-3": 0.5,
        "claude-sonnet-4-6": 2.0,
        "claude-opus-4": 8.0,
    }

    rows = []
    for model in models:
        for i, topic in enumerate(topics):
            score = base_scores[model] + (i * 0.01)  # small topic variation
            rows.append(
                {
                    "topic_id": topic,
                    "domain": "cs_ai",
                    "tier": "A",
                    "model": model,
                    "depth": "medium",
                    "system": "autoreview",
                    "condition": "end_to_end",
                    "overall_score": score,
                    "citation_recall": score - 0.05,
                    "citation_precision": score + 0.05,
                    "citation_f1": score,
                    "hallucination_rate": 0.1 - base_scores[model] * 0.05,
                    "synthesis_score": score * 4,
                    "topic_coverage": score,
                    "writing_quality": score * 4,
                    "cost_usd": costs[model],
                    "tokens_input": 100000,
                    "tokens_output": 20000,
                }
            )

    # Add a non-medium row to ensure filtering works
    rows.append(
        {
            "topic_id": "topic_a",
            "domain": "cs_ai",
            "tier": "A",
            "model": "claude-haiku-3",
            "depth": "deep",
            "system": "autoreview",
            "condition": "end_to_end",
            "overall_score": 0.75,
            "citation_recall": 0.70,
            "citation_precision": 0.80,
            "citation_f1": 0.75,
            "hallucination_rate": 0.05,
            "synthesis_score": 3.0,
            "topic_coverage": 0.75,
            "writing_quality": 3.0,
            "cost_usd": 3.0,
            "tokens_input": 200000,
            "tokens_output": 40000,
        }
    )

    # Add an ARISE row to ensure system filtering works
    rows.append(
        {
            "topic_id": "topic_a",
            "domain": "cs_ai",
            "tier": "A",
            "model": "arise",
            "depth": "medium",
            "system": "arise",
            "condition": "end_to_end",
            "overall_score": 0.60,
            "citation_recall": 0.55,
            "citation_precision": 0.65,
            "citation_f1": 0.60,
            "hallucination_rate": 0.12,
            "synthesis_score": 2.4,
            "topic_coverage": 0.60,
            "writing_quality": 2.4,
            "cost_usd": None,
            "tokens_input": None,
            "tokens_output": None,
        }
    )

    return pd.DataFrame(rows)


class TestComputeModelSummary:
    def test_model_tier_summary(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_summary

        summary = compute_model_summary(sample_df)
        assert "claude-sonnet-4-6" in summary
        assert "overall_score" in summary["claude-sonnet-4-6"]

    def test_filters_to_medium_depth(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_summary

        summary = compute_model_summary(sample_df)
        # "arise" system should be excluded; "deep" depth rows excluded
        assert "arise" not in summary

    def test_all_three_models_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_summary

        summary = compute_model_summary(sample_df)
        assert set(summary.keys()) == {"claude-haiku-3", "claude-sonnet-4-6", "claude-opus-4"}

    def test_summary_has_mean_std_n(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_summary

        summary = compute_model_summary(sample_df)
        stats = summary["claude-sonnet-4-6"]["overall_score"]
        assert "mean" in stats
        assert "std" in stats
        assert "n" in stats

    def test_higher_tier_has_higher_mean(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_summary

        summary = compute_model_summary(sample_df)
        haiku_mean = summary["claude-haiku-3"]["overall_score"]["mean"]
        sonnet_mean = summary["claude-sonnet-4-6"]["overall_score"]["mean"]
        opus_mean = summary["claude-opus-4"]["overall_score"]["mean"]
        assert haiku_mean < sonnet_mean < opus_mean


class TestComputeModelTests:
    def test_returns_friedman_and_wilcoxon(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_tests

        result = compute_model_tests(sample_df)
        assert "friedman" in result
        assert "wilcoxon" in result

    def test_friedman_has_metrics(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_tests

        result = compute_model_tests(sample_df)
        assert "overall_score" in result["friedman"]
        friedman = result["friedman"]["overall_score"]
        assert "statistic" in friedman
        assert "p_value" in friedman

    def test_wilcoxon_pair_keys_format(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_tests

        result = compute_model_tests(sample_df)
        wilcoxon = result["wilcoxon"].get("overall_score", {})
        for key in wilcoxon:
            assert "|" in key  # pair keys are "model_a|model_b"

    def test_wilcoxon_has_p_adjusted(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.model_comparison import compute_model_tests

        result = compute_model_tests(sample_df)
        for metric, pairs in result["wilcoxon"].items():
            for pair_key, stats in pairs.items():
                assert "p_adjusted" in stats, f"Missing p_adjusted for {metric}/{pair_key}"

    def test_p_adjusted_in_valid_range(self, sample_df: pd.DataFrame) -> None:
        import math

        from paper.analysis.model_comparison import compute_model_tests

        result = compute_model_tests(sample_df)
        for _metric, pairs in result["wilcoxon"].items():
            for _pair_key, stats in pairs.items():
                p_adj = stats["p_adjusted"]
                if not math.isnan(p_adj):
                    assert 0.0 <= p_adj <= 1.0


class TestPlotModelLine:
    def test_line_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import compute_model_summary, plot_model_line

        summary = compute_model_summary(sample_df)
        out = tmp_path / "model_line.pdf"
        plot_model_line(summary, out)
        assert out.exists()

    def test_line_chart_nonempty(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import compute_model_summary, plot_model_line

        summary = compute_model_summary(sample_df)
        out = tmp_path / "model_line.pdf"
        plot_model_line(summary, out)
        assert out.stat().st_size > 0


class TestPlotCostFrontier:
    def test_cost_frontier_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import plot_cost_frontier

        out = tmp_path / "cost_frontier.pdf"
        plot_cost_frontier(sample_df, out)
        assert out.exists()

    def test_cost_frontier_nonempty(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import plot_cost_frontier

        out = tmp_path / "cost_frontier.pdf"
        plot_cost_frontier(sample_df, out)
        assert out.stat().st_size > 0


class TestMain:
    def test_line_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "model_comparison_line.pdf").exists()

    def test_cost_frontier_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "model_cost_frontier.pdf").exists()

    def test_json_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.model_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "model_comparison.json").exists()

    def test_json_has_expected_keys(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        import json

        from paper.analysis.model_comparison import main

        out_dir = tmp_path / "out"
        asyncio.run(main(tmp_path, out_dir, sample_df))

        data = json.loads((out_dir / "model_comparison.json").read_text())
        assert "model_summary" in data
        assert "statistical_tests" in data
