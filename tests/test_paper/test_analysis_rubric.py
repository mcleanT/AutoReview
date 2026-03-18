# tests/test_paper/test_analysis_rubric.py
"""Tests for Analysis 3: Cross-Rubric Agreement."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic data with varied scores to produce meaningful correlations."""
    # Use scores that are positively correlated but not identical
    overall_scores = [0.55, 0.62, 0.68, 0.71, 0.75, 0.79, 0.83, 0.88]
    arise_totals = [52.0, 58.0, 65.0, 69.0, 74.0, 77.0, 81.0, 87.0]
    rows = []
    for i, (os, at) in enumerate(zip(overall_scores, arise_totals, strict=False)):
        rows.append(
            {
                "topic_id": f"t{i}",
                "domain": "cs_ai" if i % 2 == 0 else "biomedical",
                "tier": "B",
                "system": "autoreview" if i % 2 == 0 else "arise",
                "model": "claude-sonnet-4-6",
                "depth": "medium",
                "condition": "end_to_end",
                "overall_score": os,
                "synthesis_score": 3.0 + os * 2,
                "topic_coverage": os * 0.9,
                "writing_quality": 3.2,
                "citation_recall": 0.5 + os * 0.3,
                "citation_precision": 0.6 + os * 0.2,
                "citation_f1": 0.55 + os * 0.25,
                "arise_total": at,
                "word_count": 4500 + i * 200,
                "section_count": 8,
                "citation_count": 35 + i * 2,
                "citations_per_1000_words": 7.5,
                "flesch_kincaid_grade": 13.5,
                "cost_usd": 1.9 + i * 0.1,
                "tokens_input": 380000,
                "tokens_output": 75000,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_df_no_arise() -> pd.DataFrame:
    """DataFrame with no arise_total values (all NaN)."""
    rows = []
    for i in range(5):
        rows.append(
            {
                "topic_id": f"t{i}",
                "domain": "cs_ai",
                "tier": "B",
                "system": "autoreview",
                "model": "claude-sonnet-4-6",
                "depth": "medium",
                "condition": "end_to_end",
                "overall_score": 0.6 + i * 0.05,
                "synthesis_score": 3.5,
                "topic_coverage": 0.7,
                "writing_quality": 3.4,
                "citation_recall": 0.55,
                "citation_precision": 0.62,
                "citation_f1": 0.58,
                "arise_total": None,
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


class TestRubricAgreement:
    def test_compute_rubric_correlation(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.rubric_agreement import compute_rubric_correlation

        corr = compute_rubric_correlation(sample_df)
        assert "spearman_rho" in corr
        assert "pearson_r" in corr
        assert -1 <= corr["spearman_rho"] <= 1

    def test_correlation_structure(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.rubric_agreement import compute_rubric_correlation

        corr = compute_rubric_correlation(sample_df)
        assert "spearman_rho" in corr
        assert "spearman_p" in corr
        assert "pearson_r" in corr
        assert "pearson_p" in corr
        assert "n" in corr
        assert corr["n"] == len(sample_df)

    def test_correlation_is_positive(self, sample_df: pd.DataFrame) -> None:
        """Scores are designed to be positively correlated."""
        from paper.analysis.rubric_agreement import compute_rubric_correlation

        corr = compute_rubric_correlation(sample_df)
        assert corr["spearman_rho"] > 0
        assert corr["pearson_r"] > 0

    def test_pearson_range(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.rubric_agreement import compute_rubric_correlation

        corr = compute_rubric_correlation(sample_df)
        assert -1 <= corr["pearson_r"] <= 1

    def test_correlation_handles_no_arise(self, sample_df_no_arise: pd.DataFrame) -> None:
        from paper.analysis.rubric_agreement import compute_rubric_correlation

        corr = compute_rubric_correlation(sample_df_no_arise)
        import math

        assert math.isnan(corr["spearman_rho"])
        assert corr["n"] == 0

    def test_generates_scatter(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.rubric_agreement import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "rubric_scatter.pdf").exists()

    def test_generates_bland_altman(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.rubric_agreement import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "bland_altman.pdf").exists()

    def test_report_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.rubric_agreement import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "report.md").exists()
        assert (tmp_path / "out" / "analysis.json").exists()

    def test_report_contains_correlation(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.rubric_agreement import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        report = (tmp_path / "out" / "report.md").read_text()
        assert "spearman" in report.lower()
        assert "pearson" in report.lower()
