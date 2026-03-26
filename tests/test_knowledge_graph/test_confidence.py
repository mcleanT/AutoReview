"""Tests for Beta-Binomial confidence scoring with independence weighting."""

from __future__ import annotations

import pytest


class TestEvidenceWeights:
    def test_weight_table_values(self):
        from autoreview.knowledge_graph.confidence import EVIDENCE_WEIGHTS

        assert EVIDENCE_WEIGHTS["direct_experimental"] == 1.0
        assert EVIDENCE_WEIGHTS["observational_controlled"] == 0.7
        assert EVIDENCE_WEIGHTS["observational_uncontrolled"] == 0.4
        assert EVIDENCE_WEIGHTS["computational_prediction"] == 0.3
        assert EVIDENCE_WEIGHTS["expert_opinion"] == 0.2


class TestScoreEdge:
    def test_single_supporting_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p1",
                "first_author": "Smith",
                "last_author": "Jones",
            },
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(2.0)  # 1.0 prior + 1.0 weight
        assert posterior.beta_param == pytest.approx(1.0)  # unchanged

    def test_contradicting_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {
                "evidence_direction": "contradicts",
                "evidence_strength": "direct_experimental",
                "paper_id": "p1",
                "first_author": "Smith",
                "last_author": "Jones",
            },
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(1.0)
        assert posterior.beta_param == pytest.approx(2.0)

    def test_mixed_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p1",
                "first_author": "Smith",
                "last_author": "Jones",
            },
            {
                "evidence_direction": "contradicts",
                "evidence_strength": "observational_controlled",
                "paper_id": "p2",
                "first_author": "Lee",
                "last_author": "Park",
            },
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(2.0)  # 1.0 + 1.0
        assert posterior.beta_param == pytest.approx(1.7)  # 1.0 + 0.7


class TestIndependenceWeighting:
    def test_same_author_group_discounted(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p1",
                "first_author": "Smith",
                "last_author": "Jones",
            },
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p2",
                "first_author": "Smith",
                "last_author": "Jones",
            },
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p3",
                "first_author": "Smith",
                "last_author": "Jones",
            },
        ]
        posterior = score_edge(evidence)
        # First: 1.0, second: 0.5, third: 0.25 → total alpha = 1.0 + 1.75
        assert posterior.alpha == pytest.approx(2.75)

    def test_independent_labs_full_weight(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p1",
                "first_author": "Smith",
                "last_author": "Jones",
            },
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p2",
                "first_author": "Lee",
                "last_author": "Park",
            },
            {
                "evidence_direction": "supports",
                "evidence_strength": "direct_experimental",
                "paper_id": "p3",
                "first_author": "Chen",
                "last_author": "Wang",
            },
        ]
        posterior = score_edge(evidence)
        # 3 independent groups: 1.0 + 1.0 + 1.0 → alpha = 1.0 + 3.0
        assert posterior.alpha == pytest.approx(4.0)


class TestDerivedMetrics:
    def test_controversy_score(self):
        from autoreview.knowledge_graph.confidence import compute_derived_metrics

        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=3.0, beta_param=2.5)
        metrics = compute_derived_metrics(
            bp, evidence_count=5, paper_ids=["p1", "p2"], author_groups=2
        )
        assert metrics["controversy_score"] == pytest.approx(2.5 / 3.0)
        assert metrics["evidence_diversity"] == 2
        assert metrics["independent_source_count"] == 2

    def test_controversy_zero_for_unanimous(self):
        from autoreview.knowledge_graph.confidence import compute_derived_metrics

        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=5.0, beta_param=1.0)
        metrics = compute_derived_metrics(
            bp, evidence_count=4, paper_ids=["p1", "p2"], author_groups=2
        )
        assert metrics["controversy_score"] == pytest.approx(1.0 / 5.0)
