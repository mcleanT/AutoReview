"""Configuration for Bayesian inference."""

from __future__ import annotations

from autoreview.models.base import AutoReviewModel


class BayesianConfig(AutoReviewModel):
    """Configuration for Bayesian inference over the knowledge graph.

    Attributes:
        n_warmup: NUTS warmup (burn-in) samples per chain.
        n_samples: NUTS posterior samples per chain.
        n_chains: Number of parallel MCMC chains.
        target_accept_prob: NUTS target acceptance probability.
        max_tree_depth: Maximum NUTS tree depth.
        hierarchical: Whether to use the hierarchical model (Phase 2+).
        w_contradiction: Fixed contradiction factor weight (flat model).
        w_composition: Fixed composition factor weight (flat model).
        hotspot_top_k: Number of contradiction-centrality hotspots for targeted MCMC.
        hotspot_hop_radius: BFS radius around hotspot nodes for subgraph extraction.
        bimodality_significance: Hartigan dip test p-value threshold.
        use_laplace_fallback: Use Laplace CIs for non-hotspot edges.
        corpus_id: Identifier for ground truth association.
        seed: JAX PRNG seed.
        max_composition_hops: Max transitive chain length for composition discovery.
        composition_decay: Per-hop decay factor for composition strength.
        boundary_threshold: Min condition coupling to classify a real contradiction.
    """

    # Sampling
    n_warmup: int = 500
    n_samples: int = 1000
    n_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10

    # Model
    hierarchical: bool = False
    w_contradiction: float = 8.0
    w_composition: float = 5.0

    # Targeting
    hotspot_top_k: int = 10
    hotspot_hop_radius: int = 2
    bimodality_significance: float = 0.05

    # Laplace
    use_laplace_fallback: bool = True

    # Corpus
    corpus_id: str = "default"

    # Reproducibility
    seed: int = 0

    # Chain discovery (matches MRFConfig defaults)
    max_composition_hops: int = 3
    composition_decay: float = 0.7
    boundary_threshold: float = 0.6
