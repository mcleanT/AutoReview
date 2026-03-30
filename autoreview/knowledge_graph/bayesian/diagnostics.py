"""MCMC diagnostics and bimodality detection for Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class BayesianDiagnostics:
    """Convergence diagnostics from Bayesian inference.

    Attributes:
        converged: All R-hat < 1.01, no divergences.
        n_divergences: Number of NUTS divergent transitions.
        min_ess: Minimum effective sample size across variables.
        max_rhat: Maximum R-hat across variables.
        problematic_variables: Variables with poor diagnostics.
        bimodal_edges: Edges flagged by Hartigan's dip test.
        sampling_time_seconds: Wall-clock time for sampling.
    """

    converged: bool
    n_divergences: int
    min_ess: float
    max_rhat: float
    problematic_variables: list[str] = field(default_factory=list)
    bimodal_edges: list[str] = field(default_factory=list)
    sampling_time_seconds: float = 0.0


def detect_bimodality(
    samples: np.ndarray,
    significance: float = 0.05,
) -> bool:
    """Detect bimodality in posterior samples using Hartigan's dip test.

    Args:
        samples: 1D array of posterior samples.
        significance: P-value threshold. Below this = bimodal.

    Returns:
        True if samples are significantly bimodal.
    """
    if len(samples) < 10:
        return False

    try:
        from diptest import diptest as dip_test
    except ImportError:
        log.warning("bayesian.diagnostics.diptest_missing")
        return False

    _, p_value = dip_test(np.asarray(samples, dtype=np.float64))
    return bool(p_value < significance)


def compute_mcmc_diagnostics(
    mcmc,
    edge_ids: list[str],
    significance: float = 0.05,
) -> BayesianDiagnostics:
    """Compute convergence diagnostics from a NumPyro MCMC run.

    Args:
        mcmc: A fitted numpyro.infer.MCMC object.
        edge_ids: Edge IDs corresponding to truth variable indices.
        significance: Bimodality p-value threshold.

    Returns:
        BayesianDiagnostics with convergence info and bimodality flags.
    """
    import arviz as az

    idata = az.from_numpyro(mcmc)

    # Per-edge summary: R-hat, ESS
    summary = az.summary(idata, var_names=["truth"])
    r_hats = summary["r_hat"].values
    ess_bulk = summary["ess_bulk"].values

    # Divergences
    n_divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].values.sum())

    # Per-edge bimodality detection
    truth_samples = idata.posterior["truth"].values  # (chains, draws, n_edges)
    bimodal_edges: list[str] = []
    for i, eid in enumerate(edge_ids):
        flat = truth_samples[:, :, i].flatten()
        if detect_bimodality(flat, significance):
            bimodal_edges.append(eid)

    # Problematic variables
    problematic: list[str] = []
    for i, eid in enumerate(edge_ids):
        if i < len(r_hats) and (r_hats[i] > 1.01 or ess_bulk[i] < 400):
            problematic.append(eid)

    max_rhat = float(np.max(r_hats)) if len(r_hats) > 0 else 1.0
    min_ess = float(np.min(ess_bulk)) if len(ess_bulk) > 0 else 0.0

    return BayesianDiagnostics(
        converged=bool(max_rhat < 1.01 and n_divergences == 0),
        n_divergences=n_divergences,
        min_ess=min_ess,
        max_rhat=max_rhat,
        problematic_variables=problematic,
        bimodal_edges=bimodal_edges,
    )
