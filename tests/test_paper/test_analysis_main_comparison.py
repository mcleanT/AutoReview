# tests/test_paper/test_analysis_main_comparison.py
"""Tests for Analysis 1: Main System Comparison."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic data: 3 systems x 3 topics."""
    rows = []
    for system, model in [("autoreview", "claude-sonnet-4-6"), ("arise", "arise")]:
        for topic in ["t1", "t2", "t3"]:
            rows.append(
                {
                    "topic_id": topic,
                    "domain": "cs_ai",
                    "tier": "B",
                    "system": system,
                    "model": model,
                    "depth": "medium",
                    "condition": "end_to_end",
                    "overall_score": 0.7 if system == "autoreview" else 0.6,
                    "synthesis_score": 3.5 if system == "autoreview" else 3.0,
                    "topic_coverage": 0.8,
                    "writing_quality": 3.5,
                    "citation_recall": 0.6,
                    "citation_precision": 0.7,
                    "citation_f1": 0.65,
                    "arise_total": 70.0,
                    "word_count": 5000,
                    "section_count": 8,
                    "citation_count": 40,
                    "citations_per_1000_words": 8.0,
                    "flesch_kincaid_grade": 14.0,
                    "cost_usd": 2.0,
                    "tokens_input": 400000,
                    "tokens_output": 80000,
                }
            )
    return pd.DataFrame(rows)


class TestMainComparison:
    def test_compute_system_summary(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.main_comparison import compute_system_summary

        summary = compute_system_summary(sample_df)
        assert "autoreview" in summary
        assert "arise" in summary
        assert summary["autoreview"]["overall_score"]["mean"] == pytest.approx(0.7)

    def test_system_summary_has_all_metrics(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.main_comparison import METRICS, compute_system_summary

        summary = compute_system_summary(sample_df)
        for system in ["autoreview", "arise"]:
            for metric in METRICS:
                assert metric in summary[system], f"Missing {metric} for {system}"
                assert "mean" in summary[system][metric]
                assert "std" in summary[system][metric]
                assert "n" in summary[system][metric]

    def test_compute_statistical_tests(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.main_comparison import compute_statistical_tests

        tests = compute_statistical_tests(sample_df)
        assert "overall_score" in tests
        assert "friedman" in tests["overall_score"]
        assert "pairwise" in tests["overall_score"]
        # With identical scores per system, Wilcoxon should handle gracefully
        friedman = tests["overall_score"]["friedman"]
        assert "statistic" in friedman
        assert "p_value" in friedman

    def test_pairwise_has_fdr(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.main_comparison import compute_statistical_tests

        tests = compute_statistical_tests(sample_df)
        for metric, result in tests.items():
            for pair_key, pw in result.get("pairwise", {}).items():
                assert "p_adjusted" in pw, f"Missing p_adjusted for {metric}/{pair_key}"

    def test_generate_report(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        assert (tmp_path / "output" / "report.md").exists()
        assert (tmp_path / "output" / "analysis.json").exists()

    def test_figures_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        assert (tmp_path / "output" / "system_comparison_bar.pdf").exists()

    def test_radar_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        assert (tmp_path / "output" / "radar_chart.pdf").exists()

    def test_report_contains_systems(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.main_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "output", sample_df))
        report = (tmp_path / "output" / "report.md").read_text()
        assert "autoreview" in report.lower()
        assert "arise" in report.lower()
