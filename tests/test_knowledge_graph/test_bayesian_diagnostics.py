"""Tests for bayesian/diagnostics.py."""

from __future__ import annotations

import numpy as np


def test_bayesian_diagnostics_dataclass() -> None:
    """BayesianDiagnostics should store convergence info."""
    from autoreview.knowledge_graph.bayesian.diagnostics import BayesianDiagnostics

    diag = BayesianDiagnostics(
        converged=True,
        n_divergences=0,
        min_ess=800.0,
        max_rhat=1.001,
        problematic_variables=[],
        bimodal_edges=[],
        sampling_time_seconds=5.2,
    )
    assert diag.converged is True
    assert diag.n_divergences == 0
    assert diag.min_ess == 800.0
    assert diag.max_rhat == 1.001
    assert diag.problematic_variables == []
    assert diag.bimodal_edges == []
    assert diag.sampling_time_seconds == 5.2


def test_detect_bimodality_unimodal() -> None:
    """Unimodal Beta(5, 2) samples should NOT be flagged as bimodal."""
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality

    rng = np.random.default_rng(42)
    samples = rng.beta(5.0, 2.0, size=2000)
    assert detect_bimodality(samples, significance=0.05) is False


def test_detect_bimodality_bimodal() -> None:
    """Mixture of Beta(2, 10) and Beta(10, 2) should be flagged as bimodal."""
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality

    rng = np.random.default_rng(42)
    low_mode = rng.beta(2.0, 10.0, size=1000)
    high_mode = rng.beta(10.0, 2.0, size=1000)
    samples = np.concatenate([low_mode, high_mode])
    rng.shuffle(samples)
    assert detect_bimodality(samples, significance=0.05) is True


def test_detect_bimodality_small_sample() -> None:
    """Fewer than 10 samples should return False (insufficient data)."""
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality

    samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert detect_bimodality(samples, significance=0.05) is False


def test_detect_bimodality_uniform() -> None:
    """Uniform samples should NOT be flagged as bimodal."""
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality

    rng = np.random.default_rng(42)
    samples = rng.uniform(0.0, 1.0, size=2000)
    assert detect_bimodality(samples, significance=0.05) is False
