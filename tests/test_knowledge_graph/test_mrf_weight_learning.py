"""Tests for MRF weight learning."""

from __future__ import annotations

import networkx as nx

from autoreview.knowledge_graph.mrf_scoring import MRFConfig, score_graph_mrf
from autoreview.knowledge_graph.mrf_weight_learning import (
    WeightLearningConfig,
    WeightLearningResult,
    learn_weights,
)


def _make_graph_with_ground_truth() -> tuple[nx.MultiDiGraph, dict[str, float]]:
    """Build a small graph with known ground-truth confidence values."""
    graph = nx.MultiDiGraph()
    graph.add_node("A", canonical_name="A", entity_type="protein")
    graph.add_node("B", canonical_name="B", entity_type="protein")
    graph.add_node("C", canonical_name="C", entity_type="biological_process")

    graph.add_edge(
        "A",
        "B",
        predicate="induces",
        direction="positive",
        confidence_mean=0.7,
        edge_id="ab",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    graph.add_edge(
        "B",
        "C",
        predicate="induces",
        direction="positive",
        confidence_mean=0.6,
        edge_id="bc",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )
    graph.add_edge(
        "A",
        "C",
        predicate="induces",
        direction="positive",
        confidence_mean=0.3,
        edge_id="ac",
        organism="Mus musculus",
        model_system="mESC",
        in_vitro=True,
        conditions={},
    )

    ground_truth = {"ab": 0.75, "bc": 0.65, "ac": 0.55}
    return graph, ground_truth


def test_weight_learning_config_defaults() -> None:
    """WeightLearningConfig should have sensible defaults."""
    config = WeightLearningConfig()
    assert config.method == "grid_search"
    assert config.n_epochs > 0
    assert 0.0 < config.validation_split < 1.0


def test_learn_weights_returns_result() -> None:
    """learn_weights should return a WeightLearningResult with valid MRFConfig."""
    graph, gt = _make_graph_with_ground_truth()
    result = learn_weights(graph, gt)
    assert isinstance(result, WeightLearningResult)
    assert isinstance(result.config, MRFConfig)
    assert result.config.evidence_weight > 0
    assert result.config.contradiction_weight > 0
    assert result.config.composition_weight > 0
    assert result.best_loss < 1.0
    assert result.n_evaluations > 1


def test_learn_weights_grid_search_with_validation() -> None:
    """With validation_split > 0, should report train and validation loss."""
    graph, gt = _make_graph_with_ground_truth()
    config = WeightLearningConfig(validation_split=0.5)
    result = learn_weights(graph, gt, config=config)
    assert result.train_loss is not None
    assert result.val_loss is not None


def test_learned_weights_improve_on_defaults() -> None:
    """Learned weights should produce posteriors closer to ground truth than defaults."""
    graph, gt = _make_graph_with_ground_truth()
    result = learn_weights(graph, gt)

    default_posteriors = score_graph_mrf(graph).posteriors
    default_mse = sum((default_posteriors[eid] - gt[eid]) ** 2 for eid in gt) / len(gt)

    learned_posteriors = score_graph_mrf(graph, config=result.config).posteriors
    learned_mse = sum((learned_posteriors[eid] - gt[eid]) ** 2 for eid in gt) / len(gt)

    assert learned_mse <= default_mse + 1e-6


def test_learn_weights_gradient_method() -> None:
    """Gradient method should also return a valid result."""
    graph, gt = _make_graph_with_ground_truth()
    config = WeightLearningConfig(method="gradient", n_epochs=10)
    result = learn_weights(graph, gt, config=config)
    assert isinstance(result, WeightLearningResult)
    assert isinstance(result.config, MRFConfig)
    assert result.config.evidence_weight > 0
    assert result.best_loss < 1.0


def test_learn_weights_invalid_method_raises() -> None:
    """Unknown method should raise ValueError."""
    import pytest

    graph, gt = _make_graph_with_ground_truth()
    config = WeightLearningConfig(method="bogus")
    with pytest.raises(ValueError, match="Unknown method"):
        learn_weights(graph, gt, config=config)
