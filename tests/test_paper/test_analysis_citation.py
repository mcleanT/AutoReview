"""Tests for Analysis 6: Citation quality."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame mimicking load_all_evaluations() output."""
    rows = [
        {
            "topic_id": "t1",
            "domain": "cs_ai",
            "tier": "A",
            "model": "claude-sonnet-4-6",
            "depth": "medium",
            "system": "autoreview",
            "condition": "end_to_end",
            "overall_score": 0.80,
            "citation_recall": 0.70,
            "citation_precision": 0.85,
            "citation_f1": 0.77,
            "hallucination_rate": 0.05,
            "synthesis_score": 3.5,
            "topic_coverage": 0.75,
            "writing_quality": 3.2,
            "cost_usd": 2.0,
        },
        {
            "topic_id": "t2",
            "domain": "cs_ai",
            "tier": "A",
            "model": "claude-sonnet-4-6",
            "depth": "medium",
            "system": "autoreview",
            "condition": "end_to_end",
            "overall_score": 0.65,
            "citation_recall": 0.60,
            "citation_precision": 0.70,
            "citation_f1": 0.65,
            "hallucination_rate": 0.10,
            "synthesis_score": 3.0,
            "topic_coverage": 0.65,
            "writing_quality": 2.8,
            "cost_usd": 1.8,
        },
        {
            "topic_id": "t3",
            "domain": "biology",
            "tier": "B",
            "model": "claude-haiku-3",
            "depth": "medium",
            "system": "autoreview",
            "condition": "end_to_end",
            "overall_score": 0.55,
            "citation_recall": 0.50,
            "citation_precision": 0.60,
            "citation_f1": 0.55,
            "hallucination_rate": 0.20,
            "synthesis_score": 2.5,
            "topic_coverage": 0.55,
            "writing_quality": 2.5,
            "cost_usd": 0.8,
        },
    ]
    return pd.DataFrame(rows)


class TestComputeCitationBreakdown:
    def test_citation_breakdown_by_domain(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        assert "cs_ai" in breakdown
        assert "mean_recall" in breakdown["cs_ai"]

    def test_breakdown_contains_all_groups(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        assert set(breakdown.keys()) == {"cs_ai", "biology"}

    def test_breakdown_by_tier(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="tier")
        assert "A" in breakdown
        assert "B" in breakdown
        assert "mean_f1" in breakdown["A"]

    def test_breakdown_by_model(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="model")
        assert "claude-sonnet-4-6" in breakdown
        assert "mean_precision" in breakdown["claude-sonnet-4-6"]

    def test_mean_recall_values_are_correct(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        # cs_ai has two rows with recall 0.70 and 0.60 → mean 0.65
        assert breakdown["cs_ai"]["mean_recall"] == pytest.approx(0.65)

    def test_all_metric_keys_present(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_citation_breakdown

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        expected_keys = {"mean_recall", "mean_precision", "mean_f1", "mean_hallucination_rate"}
        for group in breakdown.values():
            assert expected_keys == set(group.keys())


class TestComputeHallucinationRate:
    def test_hallucination_rate(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_hallucination_rate

        rate = compute_hallucination_rate(sample_df)
        assert 0 <= rate <= 1

    def test_hallucination_rate_value(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.citation_analysis import compute_hallucination_rate

        rate = compute_hallucination_rate(sample_df)
        # mean of 0.05, 0.10, 0.20 = 0.1167
        assert rate == pytest.approx(0.35 / 3, abs=1e-6)

    def test_empty_df_returns_zero(self) -> None:
        from paper.analysis.citation_analysis import compute_hallucination_rate

        empty = pd.DataFrame(columns=["hallucination_rate"])
        assert compute_hallucination_rate(empty) == 0.0

    def test_missing_column_returns_zero(self) -> None:
        from paper.analysis.citation_analysis import compute_hallucination_rate

        df = pd.DataFrame({"other_col": [1, 2]})
        assert compute_hallucination_rate(df) == 0.0


class TestPlotCitationBar:
    def test_citation_bar_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.citation_analysis import (
            compute_citation_breakdown,
            plot_citation_bar,
        )

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        out = tmp_path / "citation_bar.pdf"
        plot_citation_bar(breakdown, out)
        assert out.exists()

    def test_output_is_nonempty(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.citation_analysis import (
            compute_citation_breakdown,
            plot_citation_bar,
        )

        breakdown = compute_citation_breakdown(sample_df, group_by="domain")
        out = tmp_path / "citation_bar.pdf"
        plot_citation_bar(breakdown, out)
        assert out.stat().st_size > 0


class TestPlotHallucinationHistogram:
    def test_histogram_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.citation_analysis import plot_hallucination_histogram

        out = tmp_path / "hallucination_hist.pdf"
        plot_hallucination_histogram(sample_df, out)
        assert out.exists()

    def test_histogram_nonempty(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.citation_analysis import plot_hallucination_histogram

        out = tmp_path / "hallucination_hist.pdf"
        plot_hallucination_histogram(sample_df, out)
        assert out.stat().st_size > 0


class TestMain:
    def test_main_creates_outputs(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.citation_analysis import main

        results_dir = tmp_path / "results"
        output_dir = tmp_path / "out"
        main(results_dir, output_dir, sample_df)

        assert (output_dir / "citation_analysis.json").exists()
        assert (output_dir / "citation_bar_domain.pdf").exists()
        assert (output_dir / "citation_bar_tier.pdf").exists()
        assert (output_dir / "hallucination_histogram.pdf").exists()

    def test_main_json_has_expected_keys(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        import json

        from paper.analysis.citation_analysis import main

        output_dir = tmp_path / "out"
        main(tmp_path, output_dir, sample_df)

        data = json.loads((output_dir / "citation_analysis.json").read_text())
        assert "overall_hallucination_rate" in data
        assert "breakdown_by_domain" in data
        assert "breakdown_by_tier" in data
        assert "breakdown_by_model" in data
