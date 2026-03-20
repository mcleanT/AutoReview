"""Tests for autoreview.analysis.synthesis_validator."""

from __future__ import annotations

import pytest

from autoreview.analysis.synthesis_validator import SynthesisMetrics, compute_synthesis_ratio


class TestComputeSynthesisRatio:
    def test_multi_citation_sentences_count(self):
        """Text with [@p1; @p2; @p3], [@p1], [@p2; @p4] → multi=2, single=1, ratio>0.5."""
        text = (
            "Multiple studies converge on this finding [@p1; @p2; @p3]. "
            "Smith alone reported the effect [@p1]. "
            "Two groups confirm the mechanism [@p2; @p4]."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.multi_paper_sentences == 2
        assert metrics.single_paper_sentences == 1
        assert metrics.total_cited_sentences == 3
        assert metrics.synthesis_ratio > 0.5

    def test_pure_summary_has_low_ratio(self):
        """Each sentence cites exactly one paper → ratio < 0.2."""
        text = "Smith found X [@p1]. Jones found Y [@p2]. Lee found Z [@p3]."
        metrics = compute_synthesis_ratio(text)
        assert metrics.single_paper_sentences == 3
        assert metrics.multi_paper_sentences == 0
        assert metrics.synthesis_ratio < 0.2

    def test_uncited_sentences_excluded(self):
        """Text with no citation markers → total_cited_sentences=0, ratio=0.0."""
        text = "This is a sentence without any citations. Another sentence here."
        metrics = compute_synthesis_ratio(text)
        assert metrics.total_cited_sentences == 0
        assert metrics.synthesis_ratio == 0.0
        assert metrics.multi_paper_sentences == 0
        assert metrics.single_paper_sentences == 0

    def test_high_synthesis_text(self):
        """Text where every cited sentence references multiple papers → ratio >= 0.8."""
        text = (
            "The consensus across studies supports this view [@a1; @a2; @a3]. "
            "Both groups independently replicated the finding [@b1; @b2]. "
            "Three independent labs confirmed the result [@c1; @c2; @c3; @c4]. "
            "Converging evidence links these pathways [@d1; @d2]."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.multi_paper_sentences == 4
        assert metrics.single_paper_sentences == 0
        assert metrics.synthesis_ratio >= 0.8

    def test_empty_text(self):
        """Empty string → all zero metrics."""
        metrics = compute_synthesis_ratio("")
        assert metrics.total_cited_sentences == 0
        assert metrics.synthesis_ratio == 0.0

    def test_synthesis_metrics_is_frozen(self):
        """SynthesisMetrics dataclass should be immutable."""
        metrics = SynthesisMetrics(
            multi_paper_sentences=2,
            single_paper_sentences=1,
            total_cited_sentences=3,
            synthesis_ratio=2 / 3,
        )
        with pytest.raises(Exception):
            metrics.multi_paper_sentences = 5  # type: ignore[misc]

    def test_mixed_cited_and_uncited(self):
        """Uncited sentences are excluded from totals."""
        text = (
            "Background context with no citation here. "
            "Evidence integrates two sources [@x1; @x2]. "
            "A single-source finding [@y1]. "
            "Another uncited statement."
        )
        metrics = compute_synthesis_ratio(text)
        assert metrics.total_cited_sentences == 2
        assert metrics.multi_paper_sentences == 1
        assert metrics.single_paper_sentences == 1
        assert metrics.synthesis_ratio == pytest.approx(0.5)
