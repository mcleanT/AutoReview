import asyncio

import pandas as pd
import pytest


@pytest.fixture()
def arise_df() -> pd.DataFrame:
    """Synthetic ARISE benchmark data."""
    rows = []
    for topic in ["arise_llm_reasoning", "arise_rag", "arise_time_series"]:
        rows.append(
            {
                "topic_id": topic,
                "domain": "cs_ai",
                "tier": "ARISE",
                "system": "autoreview",
                "model": "claude-sonnet-4-6",
                "depth": "medium",
                "condition": "end_to_end",
                "overall_score": 0.85,
                "synthesis_score": 4.0,
                "topic_coverage": 0.8,
                "writing_quality": 4.0,
                "citation_recall": 0.7,
                "citation_precision": 0.8,
                "citation_f1": 0.75,
                "arise_total": 88.0,
                "hallucination_rate": 0.02,
                "word_count": 8000,
                "section_count": 10,
                "citation_count": 60,
                "citations_per_1000_words": 7.5,
                "flesch_kincaid_grade": 14.0,
                "cost_usd": 2.0,
                "tokens_input": 400000,
                "tokens_output": 80000,
            }
        )
    return pd.DataFrame(rows)


class TestARISEComparison:
    def test_load_arise_runs(self, arise_df):
        from paper.analysis.arise_comparison import load_arise_runs

        result = load_arise_runs(arise_df)
        assert len(result) == 3
        assert all(result["tier"] == "ARISE")

    def test_compute_comparison(self, arise_df):
        from paper.analysis.arise_comparison import compute_arise_comparison

        comp = compute_arise_comparison(arise_df)
        assert "autoreview_mean" in comp
        assert "arise_published" in comp
        assert comp["autoreview_mean"] == pytest.approx(88.0)

    def test_figures_generated(self, arise_df, tmp_path):
        from paper.analysis.arise_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "out", arise_df))
        assert (tmp_path / "out" / "arise_comparison.json").exists()
        assert (tmp_path / "out" / "system_comparison_bar.pdf").exists()

    def test_report_generated(self, arise_df, tmp_path):
        from paper.analysis.arise_comparison import main

        asyncio.run(main(tmp_path, tmp_path / "out", arise_df))
        assert (tmp_path / "out" / "report.md").exists()
