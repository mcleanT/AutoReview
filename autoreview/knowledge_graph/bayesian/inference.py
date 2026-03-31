"""Bayesian inference for the knowledge graph: Laplace approximation and NUTS sampling.

Provides three public callables:

- ``extract_inference_subgraph(graph, target_nodes, hop_radius) -> nx.MultiDiGraph``
  BFS-based subgraph extraction around a set of target nodes.

- ``laplace_approximate(inputs, config) -> LaplaceResult``
  MAP estimation via L-BFGS-B + Hessian-based Gaussian approximation.

- ``sample_posterior(inputs, config, rng_key) -> NUTSResult``
  Full NUTS sampling via NumPyro MCMC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import structlog

from autoreview.knowledge_graph.bayesian.diagnostics import BayesianDiagnostics

if TYPE_CHECKING:
    import jax

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import ModelInputs

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LaplaceResult:
    """Results from Laplace approximation (MAP + Hessian).

    Attributes:
        means: MAP estimates per edge_id, clamped to [0, 1].
        variances: Posterior variances from inverse Hessian diagonal.
        credible_intervals: 95% CI per edge_id as (lower, upper), clamped to [0, 1].
        converged: Whether L-BFGS-B optimisation converged.
        n_iterations: Number of optimiser iterations taken.
    """

    means: dict[str, float]
    variances: dict[str, float]
    credible_intervals: dict[str, tuple[float, float]]
    converged: bool = True
    n_iterations: int = 0


@dataclass
class NUTSResult:
    """Results from NUTS posterior sampling.

    Attributes:
        samples: Per-edge posterior samples, shape (n_samples,) per edge_id.
        diagnostics: MCMC convergence diagnostics, or None if not computed.
    """

    samples: dict[str, np.ndarray]
    diagnostics: BayesianDiagnostics | None = field(default=None)  # noqa: F821


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------


def extract_inference_subgraph(
    graph: nx.MultiDiGraph,
    target_nodes: set[str],
    hop_radius: int = 2,
) -> nx.MultiDiGraph:
    """Extract a subgraph around ``target_nodes`` within ``hop_radius`` hops.

    BFS expanding over both predecessors and successors from each target node.
    Only nodes reachable via the BFS from target_nodes are included — disconnected
    components are excluded even if they are within hop_radius of each other.

    Args:
        graph: Full knowledge graph.
        target_nodes: Seed nodes to expand from.
        hop_radius: Number of BFS hops in each direction.

    Returns:
        A copy of the induced subgraph over the included nodes.
    """
    included: set[str] = set()
    frontier: set[str] = set(target_nodes) & set(graph.nodes())

    included.update(frontier)

    for _ in range(hop_radius):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        new_nodes = next_frontier - included
        included.update(new_nodes)
        frontier = new_nodes
        if not frontier:
            break

    return graph.subgraph(included).copy()


# ---------------------------------------------------------------------------
# Laplace approximation
# ---------------------------------------------------------------------------


def laplace_approximate(
    inputs: ModelInputs,
    config: BayesianConfig,
) -> LaplaceResult:
    """Compute the Laplace approximation to the posterior.

    Finds the MAP estimate via L-BFGS-B (scipy), then approximates posterior
    variance from the diagonal of the inverse Hessian of the negative
    log-posterior at the MAP.

    Args:
        inputs: Pre-computed :class:`~autoreview.knowledge_graph.bayesian.model.ModelInputs`.
        config: :class:`~autoreview.knowledge_graph.bayesian.config.BayesianConfig`
            controlling model weights.

    Returns:
        :class:`LaplaceResult` with per-edge means, variances, and 95% CIs.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize as scipy_minimize

    from autoreview.knowledge_graph.bayesian.model import kg_flat_model

    n = inputs.n_edges

    # ------------------------------------------------------------------
    # Define log-posterior as a pure JAX function over a flat truth vector
    # ------------------------------------------------------------------
    def log_posterior(truth_vals: jnp.ndarray) -> jnp.ndarray:
        log_p, _ = log_density(
            kg_flat_model,
            (inputs,),
            {
                "w_contradiction": config.w_contradiction,
                "w_composition": config.w_composition,
            },
            {"truth": truth_vals},
        )
        return jnp.asarray(log_p)  # log_density returns Any; coerce to Array

    neg_log_post = lambda t: -log_posterior(t)  # noqa: E731
    val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_post))

    # ------------------------------------------------------------------
    # Initial values: Beta prior means, clamped away from boundaries
    # ------------------------------------------------------------------
    x0 = inputs.alpha_priors / (inputs.alpha_priors + inputs.beta_priors)
    x0 = np.clip(x0, 0.01, 0.99)

    # ------------------------------------------------------------------
    # MAP via scipy L-BFGS-B (supports box constraints)
    # ------------------------------------------------------------------
    def scipy_objective(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = val_and_grad_fn(jnp.array(x_np, dtype=jnp.float32))
        return float(val), np.asarray(grad, dtype=np.float64)

    opt_result = scipy_minimize(
        scipy_objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.01, 0.99)] * n,
        options={"maxiter": 200},
    )
    map_est = jnp.array(opt_result.x, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Hessian of neg-log-posterior at MAP (float64 for numerical stability)
    # ------------------------------------------------------------------
    hessian_fn = jax.hessian(neg_log_post)
    hess = hessian_fn(map_est)

    # Regularise and invert — ridge of 1e-4 guards against ill-conditioning
    hess_reg = hess + 1e-4 * jnp.eye(n)
    cov = jnp.linalg.inv(hess_reg)
    variances = jnp.maximum(jnp.diag(cov), 1e-10)
    stds = jnp.sqrt(variances)

    # 95% CI from Gaussian approximation, clamped to [0, 1]
    map_np = np.asarray(map_est)
    stds_np = np.asarray(stds)
    ci_low = np.clip(map_np - 1.96 * stds_np, 0.0, 1.0)
    ci_high = np.clip(map_np + 1.96 * stds_np, 0.0, 1.0)

    means = {eid: float(map_np[i]) for i, eid in enumerate(inputs.edge_ids)}
    variances_dict = {eid: float(variances[i]) for i, eid in enumerate(inputs.edge_ids)}
    ci_dict = {eid: (float(ci_low[i]), float(ci_high[i])) for i, eid in enumerate(inputs.edge_ids)}

    log.debug(
        "laplace_approximate done",
        n_edges=n,
        converged=bool(opt_result.success),
        n_iterations=int(opt_result.nit),
    )

    return LaplaceResult(
        means=means,
        variances=variances_dict,
        credible_intervals=ci_dict,
        converged=bool(opt_result.success),
        n_iterations=int(opt_result.nit),
    )


# ---------------------------------------------------------------------------
# NUTS posterior sampling
# ---------------------------------------------------------------------------


def sample_posterior(
    inputs: ModelInputs,
    config: BayesianConfig,
    rng_key: jax.Array,
) -> NUTSResult:
    """Sample from the posterior using NUTS via NumPyro.

    Args:
        inputs: Pre-computed :class:`~autoreview.knowledge_graph.bayesian.model.ModelInputs`.
        config: :class:`~autoreview.knowledge_graph.bayesian.config.BayesianConfig`
            controlling NUTS hyper-parameters and chain count.
        rng_key: JAX PRNG key for reproducibility.

    Returns:
        :class:`NUTSResult` with per-edge sample arrays and MCMC diagnostics.
    """
    import numpyro
    import numpyro.infer

    from autoreview.knowledge_graph.bayesian.diagnostics import compute_mcmc_diagnostics
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model

    kernel = numpyro.infer.NUTS(
        kg_flat_model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=config.n_warmup,
        num_samples=config.n_samples,
        num_chains=config.n_chains,
        progress_bar=False,
    )

    start = time.monotonic()
    mcmc.run(
        rng_key,
        inputs,
        w_contradiction=config.w_contradiction,
        w_composition=config.w_composition,
    )
    elapsed = time.monotonic() - start

    # Extract per-edge samples: raw shape is (total_samples, n_edges)
    raw_samples = mcmc.get_samples()["truth"]  # (total_samples, n_edges)
    samples: dict[str, np.ndarray] = {
        eid: np.asarray(raw_samples[:, i]) for i, eid in enumerate(inputs.edge_ids)
    }

    diagnostics = compute_mcmc_diagnostics(mcmc, inputs.edge_ids, config.bimodality_significance)
    diagnostics.sampling_time_seconds = elapsed

    log.debug(
        "sample_posterior done",
        n_edges=inputs.n_edges,
        n_samples=config.n_samples,
        n_chains=config.n_chains,
        elapsed_s=f"{elapsed:.2f}",
        converged=diagnostics.converged,
    )

    return NUTSResult(samples=samples, diagnostics=diagnostics)
