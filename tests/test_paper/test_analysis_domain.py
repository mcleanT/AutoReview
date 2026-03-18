# tests/test_paper/test_analysis_domain.py
"""Tests for Analysis 2: Cross-Domain Variation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic data: 2 domains x 2 systems x 3 topics each."""
    rows = []
    for domain, base_score in [("cs_ai", 0.72), ("biomedical", 0.65)]:
        for system, delta in [("autoreview", 0.05), ("arise", 0.0)]:
            for topic in ["t1", "t2", "t3"]:
                rows.append(
                    {
                        "topic_id": f"{domain}_{topic}",
                        "domain": domain,
                        "tier": "B",
                        "system": system,
                        "model": "claude-sonnet-4-6" if system == "autoreview" else "arise",
                        "depth": "medium",
                        "condition": "end_to_end",
                        "overall_score": base_score + delta,
                        "synthesis_score": 3.5 + (0.3 if system == "autoreview" else 0.0),
                        "topic_coverage": 0.78,
                        "writing_quality": 3.4,
                        "citation_recall": 0.58,
                        "citation_precision": 0.65,
                        "citation_f1": 0.61,
                        "arise_total": (base_score + delta) * 100,
                        "word_count": 5200,
                        "section_count": 9,
                        "citation_count": 42,
                        "citations_per_1000_words": 8.1,
                        "flesch_kincaid_grade": 13.8,
                        "cost_usd": 2.1,
                        "tokens_input": 420000,
                        "tokens_output": 82000,
                    }
                )
    return pd.DataFrame(rows)


class TestDomainAnalysis:
    def test_compute_domain_summary(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import compute_domain_summary

        summary = compute_domain_summary(sample_df)
        assert "cs_ai" in summary
        assert "biomedical" in summary

    def test_domain_summary_structure(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import METRICS, compute_domain_summary

        summary = compute_domain_summary(sample_df)
        for domain in ["cs_ai", "biomedical"]:
            for metric in METRICS:
                assert metric in summary[domain], f"Missing {metric} for {domain}"
                assert "mean" in summary[domain][metric]
                assert "std" in summary[domain][metric]
                assert "n" in summary[domain][metric]
                assert summary[domain][metric]["n"] > 0

    def test_domain_summary_values(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import compute_domain_summary

        summary = compute_domain_summary(sample_df)
        # cs_ai mean overall = (0.72+0.05 + 0.72) / 2 systems x 3 topics = 0.745
        assert summary["cs_ai"]["overall_score"]["mean"] == pytest.approx(0.745)

    def test_kruskal_wallis_runs(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import compute_domain_tests

        tests = compute_domain_tests(sample_df)
        assert "overall_score" in tests
        assert "p_value" in tests["overall_score"]

    def test_kruskal_wallis_has_fdr(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import compute_domain_tests

        tests = compute_domain_tests(sample_df)
        for metric, result in tests.items():
            assert "p_adjusted" in result, f"Missing p_adjusted for {metric}"

    def test_kruskal_wallis_has_domain_info(self, sample_df: pd.DataFrame) -> None:
        from paper.analysis.domain_analysis import compute_domain_tests

        tests = compute_domain_tests(sample_df)
        for _metric, result in tests.items():
            assert "domains" in result
            assert "statistic" in result

    def test_heatmap_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.domain_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "domain_heatmap.pdf").exists()

    def test_bar_chart_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.domain_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "domain_bar.pdf").exists()

    def test_report_and_json_generated(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.domain_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        assert (tmp_path / "out" / "report.md").exists()
        assert (tmp_path / "out" / "analysis.json").exists()

    def test_report_mentions_domains(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        from paper.analysis.domain_analysis import main

        asyncio.run(main(tmp_path, tmp_path / "out", sample_df))
        report = (tmp_path / "out" / "report.md").read_text()
        assert "cs_ai" in report.lower() or "cs ai" in report.lower()
        assert "biomedical" in report.lower()
