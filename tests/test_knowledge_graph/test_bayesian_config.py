"""Tests for bayesian/config.py."""

from __future__ import annotations


def test_bayesian_config_defaults() -> None:
    """BayesianConfig should have sensible defaults matching the spec."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    config = BayesianConfig()
    # Sampling defaults
    assert config.n_warmup == 500
    assert config.n_samples == 1000
    assert config.n_chains == 4
    assert config.target_accept_prob == 0.8
    assert config.max_tree_depth == 10
    # Model defaults
    assert config.hierarchical is False
    assert config.w_contradiction == 8.0
    assert config.w_composition == 5.0
    # Targeting defaults
    assert config.hotspot_top_k == 10
    assert config.hotspot_hop_radius == 2
    assert config.bimodality_significance == 0.05
    # Laplace
    assert config.use_laplace_fallback is True
    # Corpus
    assert config.corpus_id == "default"
    # Seed
    assert config.seed == 0
    # Composition chain discovery (matches MRFConfig)
    assert config.max_composition_hops == 3
    assert config.composition_decay == 0.7
    assert config.boundary_threshold == 0.6


def test_bayesian_config_custom_values() -> None:
    """BayesianConfig should accept custom values."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig

    config = BayesianConfig(
        n_warmup=200,
        n_samples=500,
        n_chains=2,
        w_contradiction=12.0,
        hotspot_top_k=5,
        seed=42,
    )
    assert config.n_warmup == 200
    assert config.n_samples == 500
    assert config.n_chains == 2
    assert config.w_contradiction == 12.0
    assert config.hotspot_top_k == 5
    assert config.seed == 42


def test_bayesian_config_is_autoreview_model() -> None:
    """BayesianConfig should be a Pydantic AutoReviewModel for serialization."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.models.base import AutoReviewModel

    assert issubclass(BayesianConfig, AutoReviewModel)
    config = BayesianConfig()
    d = config.model_dump()
    assert isinstance(d, dict)
    assert "n_warmup" in d
