"""Tests for Analysis 9: Contamination Analysis."""
from __future__ import annotations

import pytest


class TestComputeNgramOverlap:
    def test_ngram_overlap(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        text_a = "the quick brown fox jumps over the lazy dog"
        text_b = "the quick brown fox runs past the lazy cat"
        overlap = compute_ngram_overlap(text_a, text_b, n=3)
        assert 0 < overlap < 1  # Partial overlap

    def test_no_overlap(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        overlap = compute_ngram_overlap("alpha beta gamma", "delta epsilon zeta", n=2)
        assert overlap == 0.0

    def test_perfect_overlap(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        text = "the quick brown fox"
        overlap = compute_ngram_overlap(text, text, n=2)
        assert overlap == pytest.approx(1.0)

    def test_empty_texts_return_zero(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        assert compute_ngram_overlap("", "", n=2) == 0.0

    def test_text_shorter_than_n_returns_zero(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        # single token, n=3 → no trigrams possible
        assert compute_ngram_overlap("hello", "hello world", n=3) == 0.0

    def test_case_insensitive(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        overlap_lower = compute_ngram_overlap("the quick fox", "the quick fox", n=2)
        overlap_mixed = compute_ngram_overlap("The Quick Fox", "the quick fox", n=2)
        assert overlap_lower == pytest.approx(overlap_mixed)

    def test_unigram_overlap(self) -> None:
        from paper.analysis.contamination_analysis import compute_ngram_overlap

        # "a b" vs "a c" → shared={a}, union={a,b,c} → 1/3
        overlap = compute_ngram_overlap("a b", "a c", n=1)
        assert overlap == pytest.approx(1 / 3)


class TestPlotFunctions:
    def test_overlap_histogram_generated(self, tmp_path) -> None:
        import pandas as pd

        from paper.analysis.contamination_analysis import plot_overlap_histogram

        df = pd.DataFrame({"overlap": [0.1, 0.2, 0.3, 0.4, 0.05]})
        out = tmp_path / "histogram.pdf"
        plot_overlap_histogram(df, out)
        assert out.exists()

    def test_tier_comparison_generated(self, tmp_path) -> None:
        import pandas as pd

        from paper.analysis.contamination_analysis import plot_tier_comparison

        df = pd.DataFrame(
            {
                "overlap": [0.1, 0.2, 0.3, 0.4, 0.05, 0.15],
                "tier": ["A", "A", "A", "B", "B", "B"],
            }
        )
        out = tmp_path / "tier_comparison.pdf"
        plot_tier_comparison(df, out)
        assert out.exists()
