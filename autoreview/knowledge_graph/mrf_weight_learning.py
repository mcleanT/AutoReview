"""MRF weight learning: learn optimal rule weights from ground-truth annotations.

Supports two methods:
- "grid_search": Exhaustive search over a discretized weight grid.
- "gradient": Gradient-based optimization via finite differences.

Public API
----------
learn_weights(graph, ground_truth, config=None) -> WeightLearningResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import networkx as nx
import numpy as np
import structlog

from autoreview.knowledge_graph.mrf_scoring import MRFConfig, score_graph_mrf
from autoreview.models.base import AutoReviewModel

log = structlog.get_logger(__name__)


class WeightLearningConfig(AutoReviewModel):
    """Configuration for weight learning.

    Attributes:
        method: Learning method ("grid_search" or "gradient").
        learning_rate: Step size for gradient descent.
        n_epochs: Number of gradient descent epochs.
        validation_split: Fraction of ground truth held out for validation.
        evidence_weight_range: (min, max, step) for evidence weight grid.
        contradiction_weight_range: (min, max, step) for contradiction weight grid.
        composition_weight_range: (min, max, step) for composition weight grid.
    """

    method: str = "grid_search"
    learning_rate: float = 0.1
    n_epochs: int = 50
    validation_split: float = 0.2
    evidence_weight_range: tuple[float, float, float] = (5.0, 20.0, 5.0)
    contradiction_weight_range: tuple[float, float, float] = (3.0, 15.0, 4.0)
    composition_weight_range: tuple[float, float, float] = (2.0, 10.0, 3.0)


@dataclass
class WeightLearningResult:
    """Result of weight learning."""

    config: MRFConfig
    best_loss: float
    n_evaluations: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    loss_history: list[float] = field(default_factory=list)


def _compute_mse(
    posteriors: dict[str, float],
    ground_truth: dict[str, float],
) -> float:
    """Compute MSE between posteriors and ground truth for overlapping edges."""
    common = set(posteriors.keys()) & set(ground_truth.keys())
    if not common:
        return float("inf")
    return sum((posteriors[eid] - ground_truth[eid]) ** 2 for eid in common) / len(common)


def _split_ground_truth(
    ground_truth: dict[str, float],
    validation_split: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split ground truth into train and validation sets (deterministic)."""
    keys = sorted(ground_truth.keys())
    n_val = max(1, int(len(keys) * validation_split))
    val_keys = set(keys[-n_val:])
    train = {k: v for k, v in ground_truth.items() if k not in val_keys}
    val = {k: v for k, v in ground_truth.items() if k in val_keys}
    return train, val


def _grid_search(
    graph: nx.MultiDiGraph,
    ground_truth: dict[str, float],
    config: WeightLearningConfig,
) -> WeightLearningResult:
    """Exhaustive grid search over weight space."""
    e_range = np.arange(*config.evidence_weight_range)
    c_range = np.arange(*config.contradiction_weight_range)
    p_range = np.arange(*config.composition_weight_range)

    train_gt, val_gt = _split_ground_truth(ground_truth, config.validation_split)

    best_loss = float("inf")
    best_config = MRFConfig()
    n_evals = 0
    loss_history: list[float] = []

    for ew, cw, pw in product(e_range, c_range, p_range):
        mrf_config = MRFConfig(
            evidence_weight=float(ew),
            contradiction_weight=float(cw),
            composition_weight=float(pw),
        )
        result = score_graph_mrf(graph, config=mrf_config)
        train_loss = _compute_mse(result.posteriors, train_gt)
        loss_history.append(train_loss)
        n_evals += 1

        if train_loss < best_loss:
            best_loss = train_loss
            best_config = mrf_config

    # Evaluate best config on validation set
    val_result = score_graph_mrf(graph, config=best_config)
    val_loss = _compute_mse(val_result.posteriors, val_gt) if val_gt else None
    train_loss_final = _compute_mse(val_result.posteriors, train_gt)

    return WeightLearningResult(
        config=best_config,
        best_loss=best_loss,
        n_evaluations=n_evals,
        train_loss=train_loss_final,
        val_loss=val_loss,
        loss_history=loss_history,
    )


def _gradient_learning(
    graph: nx.MultiDiGraph,
    ground_truth: dict[str, float],
    config: WeightLearningConfig,
) -> WeightLearningResult:
    """Gradient-based weight learning via finite differences."""
    train_gt, val_gt = _split_ground_truth(ground_truth, config.validation_split)

    weights = np.array([10.0, 8.0, 5.0], dtype=np.float64)
    eps = 0.5
    lr = config.learning_rate
    loss_history: list[float] = []

    for _epoch in range(config.n_epochs):
        mrf_config = MRFConfig(
            evidence_weight=float(weights[0]),
            contradiction_weight=float(weights[1]),
            composition_weight=float(weights[2]),
        )
        result = score_graph_mrf(graph, config=mrf_config)
        base_loss = _compute_mse(result.posteriors, train_gt)
        loss_history.append(base_loss)

        grad = np.zeros(3)
        for i in range(3):
            w_perturbed = weights.copy()
            w_perturbed[i] += eps
            perturbed_config = MRFConfig(
                evidence_weight=float(w_perturbed[0]),
                contradiction_weight=float(w_perturbed[1]),
                composition_weight=float(w_perturbed[2]),
            )
            perturbed_result = score_graph_mrf(graph, config=perturbed_config)
            perturbed_loss = _compute_mse(perturbed_result.posteriors, train_gt)
            grad[i] = (perturbed_loss - base_loss) / eps

        weights -= lr * grad
        weights = np.maximum(weights, 0.1)

    final_config = MRFConfig(
        evidence_weight=float(weights[0]),
        contradiction_weight=float(weights[1]),
        composition_weight=float(weights[2]),
    )
    final_result = score_graph_mrf(graph, config=final_config)
    train_loss = _compute_mse(final_result.posteriors, train_gt)
    val_loss = _compute_mse(final_result.posteriors, val_gt) if val_gt else None

    return WeightLearningResult(
        config=final_config,
        best_loss=train_loss,
        n_evaluations=config.n_epochs * 4,
        train_loss=train_loss,
        val_loss=val_loss,
        loss_history=loss_history,
    )


def learn_weights(
    graph: nx.MultiDiGraph,
    ground_truth: dict[str, float],
    config: WeightLearningConfig | None = None,
) -> WeightLearningResult:
    """Learn optimal MRF rule weights from ground-truth edge confidence annotations.

    Args:
        graph: A scored knowledge graph (with confidence_mean on edges).
        ground_truth: Mapping from edge_id to expert-annotated confidence [0, 1].
        config: Learning configuration. Defaults to grid search.

    Returns:
        WeightLearningResult containing the best MRFConfig and loss metrics.
    """
    if config is None:
        config = WeightLearningConfig()

    log.info(
        "mrf_weight_learning.start",
        method=config.method,
        n_ground_truth=len(ground_truth),
    )

    if config.method == "grid_search":
        result = _grid_search(graph, ground_truth, config)
    elif config.method == "gradient":
        result = _gradient_learning(graph, ground_truth, config)
    else:
        msg = f"Unknown method: {config.method}"
        raise ValueError(msg)

    log.info(
        "mrf_weight_learning.done",
        best_loss=result.best_loss,
        evidence_weight=result.config.evidence_weight,
        contradiction_weight=result.config.contradiction_weight,
        composition_weight=result.config.composition_weight,
    )

    return result
