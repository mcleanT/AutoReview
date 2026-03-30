"""Bayesian inference for knowledge graph confidence scoring.

Parallel to the HL-MRF system — produces full posterior distributions
instead of MAP point estimates. Requires JAX and NumPyro:
    pip install autoreview[bayesian]

Public API
----------
score_graph_bayesian(graph, config=None) -> BayesianResult
    Full Bayesian inference: Laplace + targeted NUTS on hotspots.
update_graph_bayesian(graph, changed_edge_ids, prior_result, config=None) -> BayesianResult
    Incremental update after adding new edges.
"""

from __future__ import annotations

from autoreview.knowledge_graph.bayesian.config import BayesianConfig
from autoreview.knowledge_graph.bayesian.diagnostics import BayesianDiagnostics
from autoreview.knowledge_graph.bayesian.scoring import (
    BayesianResult,
    score_graph_bayesian,
    update_graph_bayesian,
)

__all__ = [
    "BayesianConfig",
    "BayesianDiagnostics",
    "BayesianResult",
    "score_graph_bayesian",
    "update_graph_bayesian",
]
