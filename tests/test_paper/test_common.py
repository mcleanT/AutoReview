# tests/test_paper/test_common.py
"""Tests for shared analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper.analysis.common import (
    apply_style,
    fdr_correct,
    load_all_evaluations,
    save_analysis_json,
)


class TestFDRCorrect:
    def test_single_pvalue(self) -> None:
        result = fdr_correct([0.05])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.05)

    def test_monotonicity(self) -> None:
        """Adjusted p-values must be monotonically increasing when sorted by raw p."""
        raw = [0.001, 0.01, 0.03, 0.05, 0.5]
        adj = fdr_correct(raw)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-10

    def test_capped_at_one(self) -> None:
        adj = fdr_correct([0.9, 0.95, 0.99])
        assert all(p <= 1.0 for p in adj)

    def test_empty(self) -> None:
        assert fdr_correct([]) == []


class TestSaveAnalysisJSON:
    def test_saves_and_loads(self, tmp_path: Path) -> None:
        data = {"metric": 0.95, "nested": {"a": [1, 2, 3]}}
        out = tmp_path / "sub" / "analysis.json"
        save_analysis_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["metric"] == 0.95
        assert loaded["nested"]["a"] == [1, 2, 3]


class TestApplyStyle:
    def test_sets_rcparams(self) -> None:
        import matplotlib.pyplot as plt

        apply_style()
        assert plt.rcParams["axes.labelsize"] == 12


class TestLoadAllEvaluations:
    def _make_eval_json(self, path: Path) -> None:
        """Write a minimal evaluation.json file."""
        data = {
            "timestamp": "2026-03-17T00:00:00",
            "generated_path": "gen.md",
            "reference_path": "ref.pdf",
            "overall_score": 0.75,
            "citation_score": {
                "recall": 0.6,
                "precision": 0.8,
                "f1": 0.69,
                "matched_count": 10,
                "reference_count": 15,
                "generated_count": 12,
                "matched_titles": [],
                "missed_titles": [],
                "hallucinated_titles": [],
            },
            "synthesis_score": {
                "generated_score": 3.5,
                "reference_score": 4.0,
                "delta": -0.5,
                "dimension_scores": {},
                "generated_observations": "",
                "reference_observations": "",
            },
            "topic_coverage": {
                "generated_coverage": 0.8,
                "reference_coverage": 1.0,
                "topics_in_both": [],
                "topics_only_in_reference": [],
                "topics_only_in_generated": [],
            },
            "writing_quality": {
                "generated_score": 3.0,
                "reference_score": 4.0,
                "delta": -1.0,
                "dimension_scores": {},
            },
            "structural_metrics": {
                "word_count": 5000,
                "section_count": 8,
                "citation_count": 45,
                "citations_per_1000_words": 9.0,
                "avg_section_length_words": 625,
                "section_balance": 0.3,
                "flesch_kincaid_grade": 14.5,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def _make_registry(self, results_dir: Path) -> None:
        """Write a minimal run_registry.json."""
        from paper.models import RunRegistry

        reg = RunRegistry()
        key = "test_topic|claude-sonnet-4-6|medium|end_to_end"
        run_dir = results_dir / "test_topic" / "claude-sonnet-4-6_medium_end_to_end"
        eval_path = run_dir / "evaluation.json"
        self._make_eval_json(eval_path)
        reg.register_complete(
            key,
            output_dir=str(run_dir),
            review_path=str(run_dir / "review.md"),
            cost_usd=2.50,
            tokens_input=400000,
            tokens_output=80000,
        )
        reg.register_evaluation(key, str(eval_path))
        reg.save(results_dir / "run_registry.json")

    def test_loads_from_registry(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        self._make_registry(results_dir)

        # Also need a topics.yaml for domain/tier lookup
        from paper.models import ReferenceInfo, TopicEntry, TopicsConfig

        topics = TopicsConfig(
            topics=[
                TopicEntry(
                    id="test_topic",
                    title="Test",
                    domain="cs_ai",
                    tier="B",
                    reference=ReferenceInfo(
                        doi="x", title="y", year=2023, citation_count=100, pdf_path="z"
                    ),
                    conditions=["end_to_end"],
                )
            ]
        )

        df = load_all_evaluations(results_dir, topics)
        assert len(df) == 1
        assert df.iloc[0]["topic_id"] == "test_topic"
        assert df.iloc[0]["system"] == "autoreview"
        assert df.iloc[0]["overall_score"] == 0.75
        assert df.iloc[0]["cost_usd"] == 2.50

    def test_arise_outputs_loaded(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)

        # Empty registry
        from paper.models import ReferenceInfo, RunRegistry, TopicEntry, TopicsConfig

        RunRegistry().save(results_dir / "run_registry.json")

        # ARISE output
        arise_dir = results_dir / "arise" / "test_topic"
        self._make_eval_json(arise_dir / "evaluation.json")

        topics = TopicsConfig(
            topics=[
                TopicEntry(
                    id="test_topic",
                    title="Test",
                    domain="cs_ai",
                    tier="B",
                    reference=ReferenceInfo(
                        doi="x", title="y", year=2023, citation_count=100, pdf_path="z"
                    ),
                    conditions=["end_to_end"],
                )
            ]
        )

        df = load_all_evaluations(results_dir, topics)
        assert len(df) == 1
        assert df.iloc[0]["system"] == "arise"
