# Bayesian Inference Upgrade — Phase 1: Laplace + Flat MCMC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel Bayesian inference path (NumPyro/JAX) that produces posterior distributions, credible intervals, and bimodality flags for every knowledge graph edge, alongside the existing HL-MRF system.

**Architecture:** A new `bayesian/` package within `knowledge_graph/` implements a NumPyro factor graph model with Beta priors (from Tier 1 scoring), softplus contradiction factors, and softplus composition factors. Inference uses Laplace approximation for full-graph fast estimates and NUTS sampling for targeted hotspot subgraphs. The existing HL-MRF system remains unchanged and serves as the baseline for future model comparison.

**Tech Stack:** Python 3.11+, JAX (autodiff + JIT), NumPyro (NUTS sampling), ArviZ (MCMC diagnostics), diptest (bimodality detection), scipy, networkx, Pydantic, structlog

---

## Scope

This plan covers **Phase 1 only** from the spec at `docs/superpowers/specs/2026-03-30-bayesian-inference-upgrade.md`. Subsequent phases will be separate plans:
- **Phase 2:** Ground truth curation (`ground_truth/` package + `model_comparison.py`)
- **Phase 3:** Hierarchical priors + weight learning
- **Phase 4:** HL-MRF retirement (default change)

## Context

The KG has a two-tier confidence system:
- **Tier 1:** Edge-local Beta-Binomial posteriors (`confidence.py`) — `score_all_edges()` writes `confidence_mean` to each edge. Note: the full alpha/beta params are NOT stored on the graph; only the mean is written.
- **Tier 2:** HL-MRF global inference (`hlmrf.py` + `mrf_scoring.py`) — convex optimization producing MAP point estimates via L-BFGS-B.

The Bayesian system adds a parallel Tier 2 path producing full posterior distributions. It reuses existing infrastructure: `structural_contradictions.detect_contradictions()`, `predicate_algebra.compose_predicates()`, `condition_compat.condition_coupling()`, and `analysis.score_contradiction_centrality()`.

**Known limitation:** `score_all_edges()` only writes `confidence_mean` (float), not alpha/beta. The Bayesian model derives Beta prior params from `confidence_mean + evidence_count` as an approximation: `alpha = mean * (2 + evidence_count)`, `beta = (1 - mean) * (2 + evidence_count)`. A future improvement is to store exact alpha/beta in `score_all_edges()`.

## Dependency Ordering

```
Batch 1 (independent):
  Task 1: Dependencies + bayesian/config.py
  Task 2: bayesian/diagnostics.py — bimodality detection + MCMC diagnostics

Batch 2 (needs Task 1):
  Task 3: bayesian/model.py — factor graph + model input preparation + chain BFS

Batch 3 (needs Tasks 1-3):
  Task 4: bayesian/inference.py — Laplace, NUTS, subgraph extraction

Batch 4 (needs Tasks 1-4):
  Task 5: bayesian/scoring.py — BayesianResult + orchestration

Batch 5 (needs Task 5):
  Task 6: Package exports + build_graph integration

Batch 6 (needs Task 6):
  Task 7: Integration tests + regression
```

## File Map

| File | Action | Task | Responsibility |
|------|--------|------|---------------|
| `pyproject.toml` | Modify | 1 | Add `bayesian` optional dependency group |
| `autoreview/knowledge_graph/bayesian/__init__.py` | Create | 1 (empty), 6 (exports) | Package init + public API |
| `autoreview/knowledge_graph/bayesian/config.py` | Create | 1 | `BayesianConfig` |
| `autoreview/knowledge_graph/bayesian/diagnostics.py` | Create | 2 | `BayesianDiagnostics`, `detect_bimodality`, `compute_mcmc_diagnostics` |
| `autoreview/knowledge_graph/bayesian/model.py` | Create | 3 | `ModelInputs`, `CompositionChain`, `prepare_model_inputs`, `kg_flat_model` |
| `autoreview/knowledge_graph/bayesian/inference.py` | Create | 4 | `LaplaceResult`, `NUTSResult`, `extract_inference_subgraph`, `laplace_approximate`, `sample_posterior` |
| `autoreview/knowledge_graph/bayesian/scoring.py` | Create | 5 | `BayesianResult`, `score_graph_bayesian`, `update_graph_bayesian` |
| `autoreview/knowledge_graph/__init__.py` | Modify | 6 | Add bayesian imports/exports, add `bayesian` param to `build_graph()` |
| `tests/test_knowledge_graph/test_bayesian_config.py` | Create | 1 | Config tests |
| `tests/test_knowledge_graph/test_bayesian_diagnostics.py` | Create | 2 | Diagnostics + bimodality tests |
| `tests/test_knowledge_graph/test_bayesian_model.py` | Create | 3 | Factor graph + model input tests |
| `tests/test_knowledge_graph/test_bayesian_inference.py` | Create | 4 | Laplace + NUTS + subgraph tests |
| `tests/test_knowledge_graph/test_bayesian_scoring.py` | Create | 5 | Scoring orchestration tests |
| `tests/test_knowledge_graph/test_bayesian_integration.py` | Create | 7 | End-to-end + regression tests |

---

## Task 1: Dependencies + BayesianConfig

**Files:**
- Modify: `pyproject.toml:37-57`
- Create: `autoreview/knowledge_graph/bayesian/__init__.py`
- Create: `autoreview/knowledge_graph/bayesian/config.py`
- Test: `tests/test_knowledge_graph/test_bayesian_config.py`

- [ ] **Step 1: Write failing test for BayesianConfig**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoreview.knowledge_graph.bayesian'`

- [ ] **Step 3: Add bayesian optional dependencies to pyproject.toml**

In `pyproject.toml`, add after the `analysis` section:

```toml
bayesian = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "numpyro>=0.15.0",
    "arviz>=0.18.0",
    "diptest>=0.8.0",
]
```

And update the `all` section:
```toml
all = [
    "autoreview[output,analysis,bayesian,dev]",
]
```

- [ ] **Step 4: Create bayesian package with config module**

Create `autoreview/knowledge_graph/bayesian/__init__.py`:

```python
"""Bayesian inference for knowledge graph confidence scoring.

Parallel to the HL-MRF system — produces full posterior distributions
instead of MAP point estimates. Requires JAX and NumPyro.
"""
```

Create `autoreview/knowledge_graph/bayesian/config.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Run existing tests for regression**

Run: `python -m pytest tests/test_knowledge_graph/test_mrf_scoring.py tests/test_knowledge_graph/test_hlmrf.py tests/test_knowledge_graph/test_confidence.py -v`
Expected: All PASS (no existing behavior changed)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml autoreview/knowledge_graph/bayesian/__init__.py autoreview/knowledge_graph/bayesian/config.py tests/test_knowledge_graph/test_bayesian_config.py
git commit -m "feat(kg-bayesian): Task 1 — BayesianConfig and bayesian optional dependencies"
```

---

## Task 2: Diagnostics — BayesianDiagnostics + Bimodality Detection

**Files:**
- Create: `autoreview/knowledge_graph/bayesian/diagnostics.py`
- Test: `tests/test_knowledge_graph/test_bayesian_diagnostics.py`

- [ ] **Step 1: Write failing tests for diagnostics**

```python
"""Tests for bayesian/diagnostics.py."""
from __future__ import annotations

import numpy as np
import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement diagnostics.py**

Create `autoreview/knowledge_graph/bayesian/diagnostics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_diagnostics.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/bayesian/diagnostics.py tests/test_knowledge_graph/test_bayesian_diagnostics.py
git commit -m "feat(kg-bayesian): Task 2 — BayesianDiagnostics and bimodality detection"
```

---

## Task 3: NumPyro Factor Graph Model + Input Preparation

**Files:**
- Create: `autoreview/knowledge_graph/bayesian/model.py`
- Test: `tests/test_knowledge_graph/test_bayesian_model.py`

- [ ] **Step 1: Write failing tests for model**

```python
"""Tests for bayesian/model.py."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import jax.numpy as jnp
import networkx as nx
from numpyro.infer.util import log_density


def _make_3node_graph() -> nx.MultiDiGraph:
    """A->B->C with A->C direct edge. Induces 1 composition chain."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "A", "B", predicate="induces", direction="positive",
        confidence_mean=0.85, evidence_count=3, edge_id="ab",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "B", "C", predicate="induces", direction="positive",
        confidence_mean=0.80, evidence_count=2, edge_id="bc",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "A", "C", predicate="induces", direction="positive",
        confidence_mean=0.30, evidence_count=1, edge_id="ac",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    return G


def _make_contradiction_graph() -> nx.MultiDiGraph:
    """Two opposing edges on same subject-object pair."""
    G = nx.MultiDiGraph()
    for n in ["X", "Y"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "X", "Y", predicate="induces", direction="positive",
        confidence_mean=0.75, evidence_count=2, edge_id="xy_ind",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "X", "Y", predicate="inhibits", direction="negative",
        confidence_mean=0.70, evidence_count=2, edge_id="xy_inh",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    return G


def test_prepare_model_inputs_edge_ids() -> None:
    """prepare_model_inputs should collect all edges with sorted IDs."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert set(inputs.edge_ids) == {"ab", "bc", "ac"}
    assert inputs.n_edges == 3


def test_prepare_model_inputs_alpha_beta_derived() -> None:
    """Alpha/beta should be derived from confidence_mean + evidence_count."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    idx = inputs.edge_index["ab"]
    # ab: mean=0.85, evidence_count=3 -> kappa=5.0
    # alpha=0.85*5=4.25, beta=0.15*5=0.75
    assert abs(inputs.alpha_priors[idx] - 4.25) < 0.01
    assert abs(inputs.beta_priors[idx] - 0.75) < 0.01


def test_prepare_model_inputs_contradictions() -> None:
    """Should detect opposing predicates as contradictions."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_contradiction_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert len(inputs.contra_a_idx) > 0, "Should detect at least one contradiction"


def test_prepare_model_inputs_compositions() -> None:
    """A->B->C chain should produce a composition chain targeting A->C."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    assert len(inputs.composition_chains) > 0, "Should discover A->B->C composition"
    chain = inputs.composition_chains[0]
    target_eid = inputs.edge_ids[chain.target_idx]
    assert target_eid == "ac", f"Target should be ac, got {target_eid}"


def test_kg_flat_model_finite_log_density() -> None:
    """Model should produce finite log-density for reasonable truth values."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model, prepare_model_inputs

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    truth_vals = jnp.array([0.5] * inputs.n_edges)
    log_p, _ = log_density(
        kg_flat_model, (inputs,),
        {"w_contradiction": 8.0, "w_composition": 5.0},
        {"truth": truth_vals},
    )
    assert jnp.isfinite(log_p), f"Log-density should be finite, got {log_p}"


def test_kg_flat_model_gradient_defined() -> None:
    """Gradient of log-density w.r.t. truth should be finite (no NaN)."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model, prepare_model_inputs

    G = _make_3node_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())

    def log_post(truth_vals):
        log_p, _ = log_density(
            kg_flat_model, (inputs,),
            {"w_contradiction": 8.0, "w_composition": 5.0},
            {"truth": truth_vals},
        )
        return log_p

    truth_vals = jnp.array([0.5] * inputs.n_edges)
    grad = jax.grad(log_post)(truth_vals)
    assert jnp.all(jnp.isfinite(grad)), f"Gradient should be finite, got {grad}"


def test_kg_flat_model_no_constraints_matches_prior() -> None:
    """With no contradictions or compositions, MAP should equal Beta prior mean."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.model import ModelInputs, kg_flat_model

    inputs = ModelInputs(
        edge_ids=["e1"],
        edge_index={"e1": 0},
        n_edges=1,
        alpha_priors=np.array([5.0]),
        beta_priors=np.array([2.0]),
        contra_a_idx=np.array([], dtype=np.intp),
        contra_b_idx=np.array([], dtype=np.intp),
        contra_coupling=np.array([], dtype=np.float64),
        composition_chains=[],
    )

    # Log-density should be maximized near Beta mode = (a-1)/(a+b-2) = 4/5 = 0.8
    def neg_log_post(truth_vals):
        log_p, _ = log_density(
            kg_flat_model, (inputs,), {"w_contradiction": 8.0, "w_composition": 5.0},
            {"truth": truth_vals},
        )
        return -log_p

    # Evaluate at mode vs away from mode
    val_at_mode = neg_log_post(jnp.array([0.8]))
    val_away = neg_log_post(jnp.array([0.3]))
    assert val_at_mode < val_away, "Log-density should be higher at Beta mode"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement model.py**

Create `autoreview/knowledge_graph/bayesian/model.py`:

```python
"""NumPyro factor graph model for knowledge graph Bayesian inference.

Defines the flat model (Phase 1): Beta priors from Tier 1 scoring,
softplus contradiction factors, softplus composition factors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class CompositionChain:
    """A discovered transitive composition chain.

    Attributes:
        chain_indices: Indices into the edge_ids array for the chain edges.
        target_idx: Index of the target direct edge being supported.
        strength: Composition strength from predicate algebra.
        coupling: Min condition_coupling along the chain.
        decay: composition_decay^(n_hops - 1).
    """

    chain_indices: list[int]
    target_idx: int
    strength: float
    coupling: float
    decay: float


@dataclass
class ModelInputs:
    """Structured inputs for the NumPyro factor graph model.

    Attributes:
        edge_ids: Sorted list of edge IDs.
        edge_index: Mapping from edge_id to index in arrays.
        n_edges: Total number of edges.
        alpha_priors: Beta alpha parameters, shape (n_edges,).
        beta_priors: Beta beta parameters, shape (n_edges,).
        contra_a_idx: Contradiction pair first edge indices.
        contra_b_idx: Contradiction pair second edge indices.
        contra_coupling: Condition coupling per contradiction pair.
        composition_chains: Discovered transitive composition chains.
    """

    edge_ids: list[str]
    edge_index: dict[str, int]
    n_edges: int
    alpha_priors: np.ndarray
    beta_priors: np.ndarray
    contra_a_idx: np.ndarray
    contra_b_idx: np.ndarray
    contra_coupling: np.ndarray
    composition_chains: list[CompositionChain] = field(default_factory=list)


def _derive_beta_params(data: dict) -> tuple[float, float]:
    """Derive Beta prior parameters from edge attributes.

    Uses confidence_mean as the mode center and evidence_count to control
    concentration (kappa = 2 + evidence_count). More evidence = tighter prior.

    Args:
        data: Edge attribute dict from the NetworkX graph.

    Returns:
        (alpha, beta_param) for a Beta distribution.
    """
    mean = float(data.get("confidence_mean", 0.5))
    mean = max(0.01, min(0.99, mean))
    evidence_count = max(1, int(data.get("evidence_count", 1)))
    kappa = 2.0 + evidence_count
    alpha = mean * kappa
    beta_param = (1.0 - mean) * kappa
    return max(0.1, alpha), max(0.1, beta_param)


def _discover_composition_chains(
    edge_ids: list[str],
    edge_data: dict[str, dict],
    edge_index: dict[str, int],
    max_hops: int,
    composition_decay: float,
) -> list[CompositionChain]:
    """BFS chain discovery for transitive composition rules.

    Same algorithm as mrf_scoring._ground_rules Step 5, but returns
    CompositionChain dataclasses instead of GroundRules.

    Args:
        edge_ids: Sorted list of edge IDs.
        edge_data: edge_id -> enriched attribute dict (with _u, _v keys).
        edge_index: edge_id -> index mapping.
        max_hops: Maximum chain length (number of edges in chain).
        composition_decay: Per-hop decay factor.

    Returns:
        List of discovered CompositionChain objects.
    """
    from autoreview.knowledge_graph.condition_compat import (
        ConditionVector,
        condition_coupling,
    )
    from autoreview.knowledge_graph.predicate_algebra import compose_predicates

    # Build outgoing index: node -> [(edge_id, next_node, predicate, data)]
    outgoing: dict[str, list[tuple[str, str, str, dict]]] = {}
    for eid, data in edge_data.items():
        u = data["_u"]
        outgoing.setdefault(u, []).append(
            (eid, data["_v"], data.get("predicate", ""), data)
        )

    # Build direct_edges index: (start, end) -> [edge_id]
    direct_edges: dict[tuple[str, str], list[str]] = {}
    for eid, data in edge_data.items():
        pair = (data["_u"], data["_v"])
        direct_edges.setdefault(pair, []).append(eid)

    chains: list[CompositionChain] = []

    for node_a, first_edges in outgoing.items():
        for first_eid, node_b, pred_first, data_first in first_edges:
            stack: list[tuple] = [
                (node_b, {node_a, node_b}, [first_eid], [data_first], None, pred_first, 1),
            ]

            while stack:
                (
                    cur_node,
                    visited,
                    chain_eids,
                    chain_datas,
                    composed_result,
                    last_pred,
                    depth,
                ) = stack.pop()

                if depth >= max_hops:
                    continue

                for next_eid, next_node, next_pred, next_data in outgoing.get(
                    cur_node, []
                ):
                    if next_node in visited:
                        continue

                    if composed_result is None:
                        new_comp = compose_predicates(last_pred, next_pred)
                    else:
                        new_comp = compose_predicates(
                            composed_result.composed_predicate, next_pred
                        )

                    if new_comp is None:
                        continue

                    new_chain_eids = chain_eids + [next_eid]
                    new_chain_datas = chain_datas + [next_data]
                    new_visited = visited | {next_node}

                    target_candidates = direct_edges.get((node_a, next_node), [])
                    if target_candidates:
                        n_hops = len(new_chain_eids)
                        decay_factor = composition_decay ** (n_hops - 1)

                        min_coup = 1.0
                        for i in range(len(new_chain_datas) - 1):
                            cv_i = ConditionVector.from_edge_data(new_chain_datas[i])
                            cv_j = ConditionVector.from_edge_data(
                                new_chain_datas[i + 1]
                            )
                            min_coup = min(min_coup, condition_coupling(cv_i, cv_j))

                        for target_eid in target_candidates:
                            if target_eid in edge_index:
                                chain_idxs = [
                                    edge_index[eid] for eid in new_chain_eids
                                ]
                                chains.append(
                                    CompositionChain(
                                        chain_indices=chain_idxs,
                                        target_idx=edge_index[target_eid],
                                        strength=new_comp.strength,
                                        coupling=min_coup,
                                        decay=decay_factor,
                                    )
                                )

                    stack.append(
                        (
                            next_node,
                            new_visited,
                            new_chain_eids,
                            new_chain_datas,
                            new_comp,
                            next_pred,
                            depth + 1,
                        )
                    )

    return chains


def prepare_model_inputs(
    graph,
    config,
) -> ModelInputs:
    """Extract structured model inputs from a scored knowledge graph.

    Collects edges, derives Beta prior params from confidence_mean +
    evidence_count, detects contradictions, discovers composition chains.

    Args:
        graph: NetworkX MultiDiGraph with scored edges.
        config: BayesianConfig with chain discovery params.

    Returns:
        ModelInputs ready for kg_flat_model.
    """
    from autoreview.knowledge_graph.structural_contradictions import (
        ContradictionType,
        detect_contradictions,
    )

    # Collect and index edges
    edge_data: dict[str, dict] = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_id: str = data.get("edge_id", f"{u}__{key}__{v}")
        enriched = dict(data)
        enriched["_u"] = u
        enriched["_v"] = v
        enriched["_key"] = key
        edge_data[edge_id] = enriched

    edge_ids = sorted(edge_data.keys())
    edge_index = {eid: i for i, eid in enumerate(edge_ids)}
    n_edges = len(edge_ids)

    # Derive Beta priors
    alphas = np.zeros(n_edges, dtype=np.float64)
    betas = np.zeros(n_edges, dtype=np.float64)
    for eid in edge_ids:
        a, b = _derive_beta_params(edge_data[eid])
        alphas[edge_index[eid]] = a
        betas[edge_index[eid]] = b

    # Contradiction pairs
    contradictions = detect_contradictions(graph, config.boundary_threshold)
    contra_a: list[int] = []
    contra_b: list[int] = []
    contra_c: list[float] = []
    for cp in contradictions:
        if cp.contradiction_type == ContradictionType.BOUNDARY_CONDITION:
            continue
        if cp.edge_a_id in edge_index and cp.edge_b_id in edge_index:
            contra_a.append(edge_index[cp.edge_a_id])
            contra_b.append(edge_index[cp.edge_b_id])
            contra_c.append(float(cp.condition_coupling))

    # Composition chains
    composition_chains = _discover_composition_chains(
        edge_ids,
        edge_data,
        edge_index,
        max_hops=config.max_composition_hops,
        composition_decay=config.composition_decay,
    )

    log.info(
        "bayesian.model.inputs_prepared",
        n_edges=n_edges,
        n_contradictions=len(contra_a),
        n_compositions=len(composition_chains),
    )

    return ModelInputs(
        edge_ids=edge_ids,
        edge_index=edge_index,
        n_edges=n_edges,
        alpha_priors=alphas,
        beta_priors=betas,
        contra_a_idx=np.array(contra_a, dtype=np.intp),
        contra_b_idx=np.array(contra_b, dtype=np.intp),
        contra_coupling=np.array(contra_c, dtype=np.float64),
        composition_chains=composition_chains,
    )


def kg_flat_model(
    inputs: ModelInputs,
    w_contradiction: float = 8.0,
    w_composition: float = 5.0,
) -> None:
    """NumPyro factor graph model for KG Bayesian inference (flat, Phase 1).

    Layer 1: Beta priors per edge from Tier 1 scoring.
    Layer 2: Softplus contradiction factors (vectorized).
    Layer 3: Softplus composition factors (per-chain loop).

    Args:
        inputs: Structured model inputs from prepare_model_inputs.
        w_contradiction: Contradiction factor weight.
        w_composition: Composition factor weight.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # Layer 1: Beta priors
    truth = numpyro.sample(
        "truth",
        dist.Beta(
            jnp.array(inputs.alpha_priors),
            jnp.array(inputs.beta_priors),
        ),
    )

    # Layer 2: Contradiction factors (vectorized)
    if len(inputs.contra_a_idx) > 0:
        a_vals = truth[inputs.contra_a_idx]
        b_vals = truth[inputs.contra_b_idx]
        penalties = jax.nn.softplus(a_vals + b_vals - 1.0)
        coupling = jnp.array(inputs.contra_coupling)
        total_contra = jnp.sum(w_contradiction * coupling * penalties)
        numpyro.factor("contradictions", -total_contra)

    # Layer 3: Composition factors (per-chain)
    for i, chain in enumerate(inputs.composition_chains):
        chain_vals = truth[jnp.array(chain.chain_indices)]
        chain_min = jnp.min(chain_vals)
        expected = chain.strength * chain_min
        shortfall = jax.nn.softplus(expected - truth[chain.target_idx])
        numpyro.factor(
            f"comp_{i}",
            -w_composition * chain.coupling * chain.decay * shortfall,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_model.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/bayesian/model.py tests/test_knowledge_graph/test_bayesian_model.py
git commit -m "feat(kg-bayesian): Task 3 — NumPyro flat factor graph model with chain BFS"
```

---

## Task 4: Inference — Laplace, NUTS, Subgraph Extraction

**Files:**
- Create: `autoreview/knowledge_graph/bayesian/inference.py`
- Test: `tests/test_knowledge_graph/test_bayesian_inference.py`

- [ ] **Step 1: Write failing tests for inference**

```python
"""Tests for bayesian/inference.py."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx


def _make_simple_graph() -> nx.MultiDiGraph:
    """Two independent edges — no contradictions, no compositions."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "A", "B", predicate="induces", direction="positive",
        confidence_mean=0.80, evidence_count=4, edge_id="ab",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "B", "C", predicate="induces", direction="positive",
        confidence_mean=0.60, evidence_count=2, edge_id="bc",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    return G


def _make_chain_graph() -> nx.MultiDiGraph:
    """A->B->C with A->C direct — composition chain present."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "A", "B", predicate="induces", direction="positive",
        confidence_mean=0.85, evidence_count=3, edge_id="ab",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "B", "C", predicate="induces", direction="positive",
        confidence_mean=0.80, evidence_count=2, edge_id="bc",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "A", "C", predicate="induces", direction="positive",
        confidence_mean=0.30, evidence_count=1, edge_id="ac",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    return G


def _make_4node_graph() -> nx.MultiDiGraph:
    """A->B->C->D linear chain with disconnected E->F."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D", "E", "F"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    for src, dst, eid in [("A", "B", "ab"), ("B", "C", "bc"), ("C", "D", "cd"), ("E", "F", "ef")]:
        G.add_edge(
            src, dst, predicate="induces", direction="positive",
            confidence_mean=0.70, evidence_count=2, edge_id=eid,
            organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
        )
    return G


def test_extract_inference_subgraph_basic() -> None:
    """Subgraph around node B with hop_radius=1 should include A, B, C."""
    from autoreview.knowledge_graph.bayesian.inference import extract_inference_subgraph

    G = _make_4node_graph()
    sub = extract_inference_subgraph(G, target_nodes={"B"}, hop_radius=1)
    assert set(sub.nodes()) == {"A", "B", "C"}
    assert sub.number_of_edges() == 2  # ab, bc


def test_extract_inference_subgraph_disconnected() -> None:
    """Subgraph around B should NOT include disconnected E, F."""
    from autoreview.knowledge_graph.bayesian.inference import extract_inference_subgraph

    G = _make_4node_graph()
    sub = extract_inference_subgraph(G, target_nodes={"B"}, hop_radius=1)
    assert "E" not in sub.nodes()
    assert "F" not in sub.nodes()


def test_laplace_approximate_returns_result() -> None:
    """Laplace on a simple graph should return means, variances, CIs."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import laplace_approximate
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_simple_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    result = laplace_approximate(inputs, BayesianConfig())
    assert "ab" in result.means
    assert "bc" in result.means
    assert all(0.0 <= v <= 1.0 for v in result.means.values())
    assert all(v >= 0.0 for v in result.variances.values())
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        assert 0.0 <= lo <= hi <= 1.0


def test_laplace_map_near_prior_mean() -> None:
    """With no constraints, Laplace MAP should be near Beta prior mode."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import laplace_approximate
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_simple_graph()
    inputs = prepare_model_inputs(G, BayesianConfig())
    result = laplace_approximate(inputs, BayesianConfig())
    # ab: mean=0.80, evidence_count=4 -> alpha=4.8, beta=1.2
    # Beta mode = (a-1)/(a+b-2) = 3.8/4.0 = 0.95... hmm, not quite mean
    # Beta mean = a/(a+b) = 4.8/6.0 = 0.80
    # MAP is near mode for peaked Betas, but mode can differ from mean
    assert abs(result.means["ab"] - 0.80) < 0.15, (
        f"MAP should be near prior mean, got {result.means['ab']:.3f}"
    )


def test_sample_posterior_returns_samples() -> None:
    """NUTS should produce samples with correct structure."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import sample_posterior
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_simple_graph()
    config = BayesianConfig(n_warmup=50, n_samples=100, n_chains=1, seed=42)
    inputs = prepare_model_inputs(G, config)
    rng_key = jax.random.PRNGKey(42)
    result = sample_posterior(inputs, config, rng_key)
    assert "ab" in result.samples
    assert "bc" in result.samples
    assert len(result.samples["ab"]) == 100  # n_samples
    assert all(0.0 <= s <= 1.0 for s in result.samples["ab"])


def test_sample_posterior_diagnostics() -> None:
    """NUTS diagnostics should report convergence info."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.inference import sample_posterior
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    G = _make_simple_graph()
    config = BayesianConfig(n_warmup=100, n_samples=200, n_chains=2, seed=42)
    inputs = prepare_model_inputs(G, config)
    rng_key = jax.random.PRNGKey(42)
    result = sample_posterior(inputs, config, rng_key)
    assert result.diagnostics is not None
    assert isinstance(result.diagnostics.max_rhat, float)
    assert isinstance(result.diagnostics.min_ess, float)
    assert result.diagnostics.sampling_time_seconds > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_inference.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement inference.py**

Create `autoreview/knowledge_graph/bayesian/inference.py`:

```python
"""Inference engines for Bayesian knowledge graph scoring.

Three inference methods:
- extract_inference_subgraph: BFS subgraph extraction for targeted inference
- laplace_approximate: fast Gaussian approximation via MAP + Hessian
- sample_posterior: full NUTS MCMC via NumPyro
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class LaplaceResult:
    """Result from Laplace approximation.

    Attributes:
        means: edge_id -> MAP estimate (posterior mean).
        variances: edge_id -> posterior variance.
        credible_intervals: edge_id -> (low, high) 95% CI from Gaussian approx.
        converged: Whether the MAP optimizer converged.
        n_iterations: Number of optimizer iterations.
    """

    means: dict[str, float] = field(default_factory=dict)
    variances: dict[str, float] = field(default_factory=dict)
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    converged: bool = True
    n_iterations: int = 0


@dataclass
class NUTSResult:
    """Result from NUTS MCMC sampling.

    Attributes:
        samples: edge_id -> 1D array of posterior samples.
        diagnostics: MCMC convergence diagnostics.
    """

    samples: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: "BayesianDiagnostics | None" = None


def extract_inference_subgraph(
    graph,
    target_nodes: set[str],
    hop_radius: int = 2,
):
    """Extract a subgraph for targeted Bayesian inference.

    BFS from target_nodes up to hop_radius hops. Returns a self-contained
    copy of the subgraph with all edge attributes preserved.

    Args:
        graph: NetworkX MultiDiGraph.
        target_nodes: Seed nodes for BFS expansion.
        hop_radius: Maximum BFS depth.

    Returns:
        Independent MultiDiGraph copy of the subgraph.
    """
    included_nodes: set[str] = set()
    frontier = target_nodes & set(graph.nodes())

    for _ in range(hop_radius + 1):
        included_nodes |= frontier
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in set(graph.predecessors(node)) | set(
                graph.successors(node)
            ):
                if neighbor not in included_nodes:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    subgraph = graph.subgraph(included_nodes).copy()
    log.info(
        "bayesian.inference.subgraph_extracted",
        target_nodes=len(target_nodes),
        included_nodes=len(included_nodes),
        n_edges=subgraph.number_of_edges(),
    )
    return subgraph


def laplace_approximate(inputs, config) -> LaplaceResult:
    """Compute Laplace approximation: MAP + Hessian-based CIs.

    Finds the MAP estimate of the log-posterior via L-BFGS-B, then
    computes the Hessian at the MAP to get a Gaussian posterior
    approximation. Credible intervals are +-1.96 standard deviations.

    Args:
        inputs: ModelInputs from prepare_model_inputs.
        config: BayesianConfig.

    Returns:
        LaplaceResult with means, variances, and credible intervals.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize as scipy_minimize

    from autoreview.knowledge_graph.bayesian.model import kg_flat_model

    n = inputs.n_edges
    if n == 0:
        return LaplaceResult()

    model_args = (inputs,)
    model_kwargs = {
        "w_contradiction": config.w_contradiction,
        "w_composition": config.w_composition,
    }

    def log_posterior(truth_vals: jnp.ndarray) -> jnp.ndarray:
        log_p, _ = log_density(
            kg_flat_model, model_args, model_kwargs, {"truth": truth_vals}
        )
        return log_p

    neg_log_post = lambda t: -log_posterior(t)
    val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_post))

    # Initial values: Beta prior means
    x0 = inputs.alpha_priors / (inputs.alpha_priors + inputs.beta_priors)
    x0 = np.clip(x0, 0.01, 0.99)

    def scipy_objective(x_np):
        val, grad = val_and_grad_fn(jnp.array(x_np))
        return float(val), np.asarray(grad, dtype=np.float64)

    log.info("bayesian.inference.laplace_start", n_edges=n)

    result = scipy_minimize(
        scipy_objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.01, 0.99)] * n,
        options={"maxiter": 200},
    )
    map_est = jnp.array(result.x)

    # Hessian of neg-log-posterior at MAP
    hessian_fn = jax.hessian(neg_log_post)
    H = hessian_fn(map_est)

    # Covariance = inverse of (positive-definite) Hessian + ridge for stability
    H_reg = H + 1e-6 * jnp.eye(n)
    cov = jnp.linalg.inv(H_reg)
    variances = jnp.maximum(jnp.diag(cov), 1e-10)
    stds = jnp.sqrt(variances)

    # 95% CI from Gaussian approximation, clipped to [0, 1]
    ci_low = jnp.clip(map_est - 1.96 * stds, 0.0, 1.0)
    ci_high = jnp.clip(map_est + 1.96 * stds, 0.0, 1.0)

    log.info(
        "bayesian.inference.laplace_done",
        converged=bool(result.success),
        n_iterations=int(result.nit),
    )

    return LaplaceResult(
        means={eid: float(map_est[i]) for i, eid in enumerate(inputs.edge_ids)},
        variances={eid: float(variances[i]) for i, eid in enumerate(inputs.edge_ids)},
        credible_intervals={
            eid: (float(ci_low[i]), float(ci_high[i]))
            for i, eid in enumerate(inputs.edge_ids)
        },
        converged=bool(result.success),
        n_iterations=int(result.nit),
    )


def sample_posterior(inputs, config, rng_key) -> NUTSResult:
    """Run NUTS MCMC sampling on the factor graph model.

    Args:
        inputs: ModelInputs from prepare_model_inputs.
        config: BayesianConfig with sampling parameters.
        rng_key: JAX PRNG key.

    Returns:
        NUTSResult with per-edge samples and diagnostics.
    """
    import numpyro
    import numpyro.infer

    from autoreview.knowledge_graph.bayesian.diagnostics import (
        compute_mcmc_diagnostics,
    )
    from autoreview.knowledge_graph.bayesian.model import kg_flat_model

    if inputs.n_edges == 0:
        from autoreview.knowledge_graph.bayesian.diagnostics import (
            BayesianDiagnostics,
        )

        return NUTSResult(
            diagnostics=BayesianDiagnostics(
                converged=True, n_divergences=0, min_ess=0.0, max_rhat=1.0
            )
        )

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

    log.info(
        "bayesian.inference.nuts_start",
        n_edges=inputs.n_edges,
        n_warmup=config.n_warmup,
        n_samples=config.n_samples,
        n_chains=config.n_chains,
    )

    start = time.monotonic()
    mcmc.run(
        rng_key,
        inputs,
        w_contradiction=config.w_contradiction,
        w_composition=config.w_composition,
    )
    elapsed = time.monotonic() - start

    # Extract per-edge samples
    raw_samples = mcmc.get_samples()["truth"]  # (total_samples, n_edges)
    samples: dict[str, np.ndarray] = {}
    for i, eid in enumerate(inputs.edge_ids):
        samples[eid] = np.asarray(raw_samples[:, i])

    diagnostics = compute_mcmc_diagnostics(
        mcmc, inputs.edge_ids, config.bimodality_significance
    )
    diagnostics.sampling_time_seconds = elapsed

    log.info(
        "bayesian.inference.nuts_done",
        converged=diagnostics.converged,
        max_rhat=diagnostics.max_rhat,
        min_ess=diagnostics.min_ess,
        n_divergences=diagnostics.n_divergences,
        elapsed_seconds=round(elapsed, 2),
    )

    return NUTSResult(samples=samples, diagnostics=diagnostics)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_inference.py -v`
Expected: All 7 tests PASS (NUTS tests may take 10-30 seconds each)

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/bayesian/inference.py tests/test_knowledge_graph/test_bayesian_inference.py
git commit -m "feat(kg-bayesian): Task 4 — Laplace approximation, NUTS sampling, subgraph extraction"
```

---

## Task 5: Scoring Orchestration — BayesianResult + score_graph_bayesian

**Files:**
- Create: `autoreview/knowledge_graph/bayesian/scoring.py`
- Test: `tests/test_knowledge_graph/test_bayesian_scoring.py`

- [ ] **Step 1: Write failing tests for scoring**

```python
"""Tests for bayesian/scoring.py."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx


def _make_scored_graph() -> nx.MultiDiGraph:
    """3-node graph with composition chain and moderate confidence."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n, canonical_name=n, entity_type="protein")
    G.add_edge(
        "A", "B", predicate="induces", direction="positive",
        confidence_mean=0.85, evidence_count=3, edge_id="ab",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "B", "C", predicate="induces", direction="positive",
        confidence_mean=0.80, evidence_count=2, edge_id="bc",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    G.add_edge(
        "A", "C", predicate="induces", direction="positive",
        confidence_mean=0.30, evidence_count=1, edge_id="ac",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )
    return G


def test_bayesian_result_empty() -> None:
    """Empty BayesianResult should have sensible defaults."""
    from autoreview.knowledge_graph.bayesian.scoring import BayesianResult

    result = BayesianResult()
    assert result.posteriors == {}
    assert result.n_variables == 0


def test_score_graph_bayesian_returns_all_edges() -> None:
    """score_graph_bayesian should return posteriors for every edge."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    G = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=0,  # Laplace only, no NUTS hotspots
    )
    result = score_graph_bayesian(G, config=config)
    assert "ab" in result.posteriors
    assert "bc" in result.posteriors
    assert "ac" in result.posteriors
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())
    assert result.n_variables == 3


def test_score_graph_bayesian_credible_intervals() -> None:
    """Credible intervals should bracket the posterior mean."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    G = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=0,
    )
    result = score_graph_bayesian(G, config=config)
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        mean = result.posteriors[eid]
        assert lo <= mean <= hi, f"{eid}: CI [{lo:.3f}, {hi:.3f}] does not contain mean {mean:.3f}"


def test_score_graph_bayesian_with_hotspots() -> None:
    """With hotspot_top_k > 0, hotspot edges should have NUTS samples."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    G = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=3, hotspot_hop_radius=1,
    )
    result = score_graph_bayesian(G, config=config)
    # Should have some edges with posterior samples from NUTS
    assert len(result.posterior_samples) > 0


def test_score_graph_bayesian_empty_graph() -> None:
    """Empty graph should return empty result."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import score_graph_bayesian

    G = nx.MultiDiGraph()
    result = score_graph_bayesian(G, config=BayesianConfig())
    assert result.posteriors == {}
    assert result.n_variables == 0


def test_update_graph_bayesian_returns_all_edges() -> None:
    """update_graph_bayesian should return posteriors for all edges."""
    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.scoring import (
        score_graph_bayesian,
        update_graph_bayesian,
    )

    G = _make_scored_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=0,
    )
    prior = score_graph_bayesian(G, config=config)

    # Add a new edge
    G.add_edge(
        "A", "C", predicate="inhibits", direction="negative",
        confidence_mean=0.40, evidence_count=1, edge_id="ac2",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )

    updated = update_graph_bayesian(G, ["ac2"], prior, config=config)
    assert "ac2" in updated.posteriors
    assert "ab" in updated.posteriors
    assert "bc" in updated.posteriors
    assert "ac" in updated.posteriors
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scoring.py**

Create `autoreview/knowledge_graph/bayesian/scoring.py`:

```python
"""Orchestration: graph -> BayesianResult.

Main public API for Bayesian inference over the knowledge graph.
Combines Laplace approximation (fast, full graph) with targeted
NUTS sampling on contradiction-centrality hotspots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class BayesianResult:
    """Result of Bayesian inference over the knowledge graph.

    Attributes:
        posteriors: edge_id -> posterior mean in [0, 1].
        credible_intervals: edge_id -> (low, high) 95% CI.
        posterior_samples: edge_id -> 1D array of MCMC samples (hotspot edges only).
        bimodality_flags: edge_id -> True if posterior is bimodal.
        diagnostics: MCMC diagnostics (None if Laplace-only).
        n_variables: Number of edge variables.
        n_contradictions: Number of contradiction factor pairs.
        n_compositions: Number of composition chains.
    """

    posteriors: dict[str, float] = field(default_factory=dict)
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    posterior_samples: dict[str, np.ndarray] = field(default_factory=dict)
    bimodality_flags: dict[str, bool] = field(default_factory=dict)
    diagnostics: "BayesianDiagnostics | None" = None
    n_variables: int = 0
    n_contradictions: int = 0
    n_compositions: int = 0


def score_graph_bayesian(
    graph,
    config=None,
) -> BayesianResult:
    """Run Bayesian inference on a knowledge graph.

    Pipeline:
    1. Prepare model inputs (Beta priors, contradictions, compositions).
    2. Laplace approximation on full graph (fast CIs for all edges).
    3. Identify contradiction-centrality hotspots.
    4. Run NUTS on hotspot subgraphs (rich posteriors for controversial edges).
    5. Merge Laplace + NUTS results into BayesianResult.

    Args:
        graph: NetworkX MultiDiGraph with scored edges (confidence_mean, evidence_count).
        config: BayesianConfig. Uses defaults if None.

    Returns:
        BayesianResult with posteriors, CIs, samples, and bimodality flags.
    """
    import jax

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality
    from autoreview.knowledge_graph.bayesian.inference import (
        extract_inference_subgraph,
        laplace_approximate,
        sample_posterior,
    )
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    if config is None:
        config = BayesianConfig()

    # Step 1: Prepare model inputs
    inputs = prepare_model_inputs(graph, config)

    if inputs.n_edges == 0:
        log.info("bayesian.scoring.empty_graph")
        return BayesianResult()

    rng_key = jax.random.PRNGKey(config.seed)

    # Step 2: Laplace approximation (fast, full graph)
    laplace_result = laplace_approximate(inputs, config)

    # Initialize from Laplace
    posteriors = dict(laplace_result.means)
    credible_intervals = dict(laplace_result.credible_intervals)
    posterior_samples: dict[str, np.ndarray] = {}
    bimodality_flags: dict[str, bool] = {eid: False for eid in inputs.edge_ids}

    # Step 3: Identify hotspots for targeted NUTS
    nuts_diagnostics = None
    if config.hotspot_top_k > 0:
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        centrality = score_contradiction_centrality(graph)
        hotspot_nodes = {
            entry["node_id"] for entry in centrality[: config.hotspot_top_k]
        }

        if hotspot_nodes:
            # Step 4: NUTS on hotspot subgraph
            subgraph = extract_inference_subgraph(
                graph, hotspot_nodes, config.hotspot_hop_radius
            )
            sub_inputs = prepare_model_inputs(subgraph, config)

            if sub_inputs.n_edges > 0:
                rng_key, sub_key = jax.random.split(rng_key)
                nuts_result = sample_posterior(sub_inputs, config, sub_key)
                nuts_diagnostics = nuts_result.diagnostics

                # Step 5: Merge — NUTS edges override Laplace
                for eid, samples in nuts_result.samples.items():
                    posteriors[eid] = float(np.mean(samples))
                    credible_intervals[eid] = (
                        float(np.percentile(samples, 2.5)),
                        float(np.percentile(samples, 97.5)),
                    )
                    posterior_samples[eid] = samples
                    bimodality_flags[eid] = detect_bimodality(
                        samples, config.bimodality_significance
                    )

    log.info(
        "bayesian.scoring.done",
        n_edges=inputs.n_edges,
        n_nuts_edges=len(posterior_samples),
        n_laplace_only=inputs.n_edges - len(posterior_samples),
    )

    return BayesianResult(
        posteriors=posteriors,
        credible_intervals=credible_intervals,
        posterior_samples=posterior_samples,
        bimodality_flags=bimodality_flags,
        diagnostics=nuts_diagnostics,
        n_variables=inputs.n_edges,
        n_contradictions=len(inputs.contra_a_idx),
        n_compositions=len(inputs.composition_chains),
    )


def update_graph_bayesian(
    graph,
    changed_edge_ids: list[str],
    prior_result: BayesianResult,
    config=None,
) -> BayesianResult:
    """Incrementally update Bayesian posteriors after adding new edges.

    Extracts the subgraph around changed edges, runs NUTS on that subgraph,
    and merges new posteriors into the prior result.

    Args:
        graph: Updated graph (with new edges already added).
        changed_edge_ids: Edge IDs that are new/changed since prior solve.
        prior_result: Result from previous score_graph_bayesian call.
        config: BayesianConfig.

    Returns:
        Updated BayesianResult.
    """
    import jax

    from autoreview.knowledge_graph.bayesian.config import BayesianConfig
    from autoreview.knowledge_graph.bayesian.diagnostics import detect_bimodality
    from autoreview.knowledge_graph.bayesian.inference import (
        extract_inference_subgraph,
        sample_posterior,
    )
    from autoreview.knowledge_graph.bayesian.model import prepare_model_inputs

    if config is None:
        config = BayesianConfig()

    # Find nodes involved in changed edges
    target_nodes: set[str] = set()
    for u, v, _key, data in graph.edges(data=True, keys=True):
        edge_id = data.get("edge_id")
        if edge_id in changed_edge_ids:
            target_nodes.add(u)
            target_nodes.add(v)

    if not target_nodes:
        return prior_result

    # Extract and score subgraph
    subgraph = extract_inference_subgraph(
        graph, target_nodes, config.hotspot_hop_radius
    )
    sub_inputs = prepare_model_inputs(subgraph, config)

    if sub_inputs.n_edges == 0:
        return prior_result

    rng_key = jax.random.PRNGKey(config.seed + 1)
    nuts_result = sample_posterior(sub_inputs, config, rng_key)

    # Merge: updated edges get NUTS posteriors, rest keep prior values
    posteriors = dict(prior_result.posteriors)
    credible_intervals = dict(prior_result.credible_intervals)
    posterior_samples = dict(prior_result.posterior_samples)
    bimodality_flags = dict(prior_result.bimodality_flags)

    for eid, samples in nuts_result.samples.items():
        posteriors[eid] = float(np.mean(samples))
        credible_intervals[eid] = (
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        )
        posterior_samples[eid] = samples
        bimodality_flags[eid] = detect_bimodality(
            samples, config.bimodality_significance
        )

    log.info(
        "bayesian.scoring.incremental_done",
        n_changed=len(changed_edge_ids),
        n_updated=len(nuts_result.samples),
    )

    return BayesianResult(
        posteriors=posteriors,
        credible_intervals=credible_intervals,
        posterior_samples=posterior_samples,
        bimodality_flags=bimodality_flags,
        diagnostics=nuts_result.diagnostics,
        n_variables=len(posteriors),
        n_contradictions=len(sub_inputs.contra_a_idx),
        n_compositions=len(sub_inputs.composition_chains),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_scoring.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/bayesian/scoring.py tests/test_knowledge_graph/test_bayesian_scoring.py
git commit -m "feat(kg-bayesian): Task 5 — score_graph_bayesian orchestration with Laplace + NUTS"
```

---

## Task 6: Package Exports + build_graph Integration

**Files:**
- Modify: `autoreview/knowledge_graph/bayesian/__init__.py`
- Modify: `autoreview/knowledge_graph/__init__.py:1-69` (imports/exports), `129-136` (build_graph signature), `446-465` (add bayesian step)

- [ ] **Step 1: Update bayesian/__init__.py with public API exports**

Replace contents of `autoreview/knowledge_graph/bayesian/__init__.py`:

```python
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
```

- [ ] **Step 2: Add bayesian imports and build_graph parameter to knowledge_graph/__init__.py**

In `autoreview/knowledge_graph/__init__.py`, add after the existing `mrf_weight_learning` import (line 44):

```python
# Bayesian inference (lazy import — only loaded when bayesian=True)
# Avoid top-level import since JAX/NumPyro are optional dependencies
```

Add to `__all__` list (after `"score_contradiction_centrality"`):

```python
    "score_graph_bayesian",
    "update_graph_bayesian",
    "BayesianConfig",
    "BayesianResult",
```

Update the `build_graph` signature (line 129) to add two new parameters:

```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
    normalize: bool = False,
    llm_decompose: bool = True,
    bayesian: bool = False,
    bayesian_config: "BayesianConfig | None" = None,
) -> nx.MultiDiGraph:
```

Update the docstring to include the new parameters (add after `llm_decompose` docs):

```python
        bayesian: If ``True``, run Bayesian inference (Laplace + targeted NUTS)
            after Beta-Binomial scoring and write ``bayesian_confidence``,
            ``bayesian_ci_low``, ``bayesian_ci_high``, and ``bayesian_bimodal``
            onto each edge. Requires ``pip install autoreview[bayesian]``.
        bayesian_config: Optional :class:`BayesianConfig` to control Bayesian
            inference hyperparameters. Uses defaults when ``None``.
```

Add a new Step 8c block after the MRF block (after line 464, before Step 9):

```python
    # ------------------------------------------------------------------
    # Step 8c: Bayesian inference (optional)
    # ------------------------------------------------------------------
    if bayesian:
        from autoreview.knowledge_graph.bayesian import (
            BayesianConfig,
            score_graph_bayesian,
        )

        b_cfg = bayesian_config or BayesianConfig()
        bayesian_result = score_graph_bayesian(graph, config=b_cfg)

        for u, v, key, data in graph.edges(data=True, keys=True):
            edge_id = data.get("edge_id")
            if edge_id and edge_id in bayesian_result.posteriors:
                graph[u][v][key]["bayesian_confidence"] = bayesian_result.posteriors[
                    edge_id
                ]
                if edge_id in bayesian_result.credible_intervals:
                    ci = bayesian_result.credible_intervals[edge_id]
                    graph[u][v][key]["bayesian_ci_low"] = ci[0]
                    graph[u][v][key]["bayesian_ci_high"] = ci[1]
                if edge_id in bayesian_result.bimodality_flags:
                    graph[u][v][key]["bayesian_bimodal"] = (
                        bayesian_result.bimodality_flags[edge_id]
                    )

        log.info(
            "kg.pipeline.bayesian_done",
            n_variables=bayesian_result.n_variables,
            n_contradictions=bayesian_result.n_contradictions,
            n_compositions=bayesian_result.n_compositions,
        )
```

- [ ] **Step 3: Run existing tests for regression**

Run: `python -m pytest tests/test_knowledge_graph/test_mrf_scoring.py tests/test_knowledge_graph/test_hlmrf.py tests/test_knowledge_graph/test_confidence.py -v`
Expected: All PASS (no existing behavior changed)

- [ ] **Step 4: Commit**

```bash
git add autoreview/knowledge_graph/bayesian/__init__.py autoreview/knowledge_graph/__init__.py
git commit -m "feat(kg-bayesian): Task 6 — package exports and build_graph(bayesian=True) integration"
```

---

## Task 7: Integration Tests + Regression

**Files:**
- Create: `tests/test_knowledge_graph/test_bayesian_integration.py`

- [ ] **Step 1: Write integration and regression tests**

```python
"""Integration tests for the full Bayesian inference pipeline."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import networkx as nx


def _make_full_graph() -> nx.MultiDiGraph:
    """A realistic 5-node graph with contradictions, compositions, mixed confidence."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D", "E"]:
        G.add_node(n, canonical_name=n, entity_type="protein")

    edges = [
        ("A", "B", "induces", "positive", 0.85, 4, "ab"),
        ("B", "C", "induces", "positive", 0.80, 3, "bc"),
        ("A", "C", "induces", "positive", 0.30, 1, "ac"),   # weak direct, chain supports
        ("C", "D", "inhibits", "negative", 0.75, 2, "cd"),
        ("C", "D", "induces", "positive", 0.60, 2, "cd2"),  # contradicts cd
        ("D", "E", "induces", "positive", 0.70, 2, "de"),
    ]
    for src, dst, pred, direction, mean, count, eid in edges:
        G.add_edge(
            src, dst, predicate=pred, direction=direction,
            confidence_mean=mean, evidence_count=count, edge_id=eid,
            organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
        )
    return G


def test_full_pipeline_laplace_only() -> None:
    """Full pipeline with Laplace only (hotspot_top_k=0) should complete."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(hotspot_top_k=0, seed=42)
    result = score_graph_bayesian(G, config=config)

    assert result.n_variables == 6
    assert len(result.posteriors) == 6
    assert len(result.credible_intervals) == 6
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())
    for eid in result.credible_intervals:
        lo, hi = result.credible_intervals[eid]
        assert lo <= result.posteriors[eid] <= hi


def test_full_pipeline_with_nuts() -> None:
    """Full pipeline with NUTS hotspots should complete and flag bimodality."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=3, hotspot_hop_radius=1,
    )
    result = score_graph_bayesian(G, config=config)

    assert result.n_variables == 6
    assert len(result.posteriors) == 6
    # Hotspot edges should have samples
    assert len(result.posterior_samples) > 0
    # Diagnostics should be populated
    assert result.diagnostics is not None


def test_incremental_update_adds_new_edge() -> None:
    """Incremental update should incorporate a new edge."""
    from autoreview.knowledge_graph.bayesian import (
        BayesianConfig,
        score_graph_bayesian,
        update_graph_bayesian,
    )

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=0,
    )
    prior = score_graph_bayesian(G, config=config)

    # Add contradicting edge
    G.add_edge(
        "A", "B", predicate="inhibits", direction="negative",
        confidence_mean=0.50, evidence_count=1, edge_id="ab2",
        organism="Mus musculus", model_system="mESC", in_vitro=True, conditions={},
    )

    updated = update_graph_bayesian(G, ["ab2"], prior, config=config)
    assert "ab2" in updated.posteriors
    assert len(updated.posteriors) == 7  # 6 original + 1 new


def test_mrf_unchanged_regression() -> None:
    """HL-MRF system should produce identical results (no code changed)."""
    from autoreview.knowledge_graph.mrf_scoring import MRFConfig, score_graph_mrf

    G = _make_full_graph()
    result = score_graph_mrf(G, config=MRFConfig())

    assert len(result.posteriors) == 6
    assert result.converged is True
    # All posteriors in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in result.posteriors.values())


def test_bayesian_posteriors_in_valid_range() -> None:
    """All posterior means should be in [0, 1] and CIs should be ordered."""
    from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian

    G = _make_full_graph()
    config = BayesianConfig(
        n_warmup=50, n_samples=100, n_chains=1, seed=42,
        hotspot_top_k=2,
    )
    result = score_graph_bayesian(G, config=config)

    for eid, mean in result.posteriors.items():
        assert 0.0 <= mean <= 1.0, f"{eid}: posterior mean {mean} out of range"

    for eid, (lo, hi) in result.credible_intervals.items():
        assert 0.0 <= lo <= hi <= 1.0, f"{eid}: CI [{lo}, {hi}] invalid"
        assert lo <= result.posteriors[eid] <= hi, (
            f"{eid}: mean {result.posteriors[eid]} outside CI [{lo}, {hi}]"
        )
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_knowledge_graph/test_bayesian_integration.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Run full knowledge graph test suite**

Run: `python -m pytest tests/test_knowledge_graph/ -v --timeout=120`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_knowledge_graph/test_bayesian_integration.py
git commit -m "feat(kg-bayesian): Task 7 — integration tests and HL-MRF regression verification"
```

---

## Verification

After all tasks complete:

```bash
# Full test suite
python -m pytest tests/test_knowledge_graph/ -v --timeout=120

# Type checking (bayesian modules)
python -m mypy autoreview/knowledge_graph/bayesian/ --ignore-missing-imports

# Lint
ruff check autoreview/knowledge_graph/bayesian/

# Quick smoke test
python -c "
from autoreview.knowledge_graph.bayesian import BayesianConfig, score_graph_bayesian
import networkx as nx
G = nx.MultiDiGraph()
G.add_node('A', canonical_name='A', entity_type='protein')
G.add_node('B', canonical_name='B', entity_type='protein')
G.add_edge('A', 'B', predicate='induces', direction='positive',
           confidence_mean=0.8, evidence_count=3, edge_id='ab',
           organism='Mus musculus', model_system='mESC', in_vitro=True, conditions={})
r = score_graph_bayesian(G, config=BayesianConfig(hotspot_top_k=0))
print(f'Posterior: {r.posteriors}')
print(f'CI: {r.credible_intervals}')
print('OK')
"
```

## Risks

1. **JAX installation conflicts:** JAX can conflict with TensorFlow or other CUDA consumers. Pin versions in pyproject.toml. Test in CI.
2. **NUTS convergence on real graphs:** Pathological structures (many tight contradictions) may cause divergences. Laplace fallback handles this — diagnostics module flags issues.
3. **Hessian cost for large graphs:** Full N×N Hessian via `jax.hessian` is O(N²) memory and O(N) backward passes. Acceptable for N<2000. For larger graphs, add diagonal Hessian approximation (follow-up work).
4. **macOS multiprocessing:** NumPyro with `n_chains > 1` on macOS may need `JAX_PLATFORM_NAME=cpu`. Tests use `n_chains=1` or `n_chains=2` to avoid issues.
5. **Alpha/beta approximation:** Using `confidence_mean + evidence_count` to derive Beta priors loses information from independence discounting and section source weights. Future improvement: store exact alpha/beta in `score_all_edges()`.

## Follow-Up Plans

- **Phase 2:** Ground truth curation (`ground_truth/` package) + model comparison (`model_comparison.py`)
- **Phase 3:** Hierarchical priors in `bayesian/model.py` — per-assertion-type weights, learned evidence tier multipliers
- **Phase 4:** HL-MRF retirement — change `bayesian=True` to default in `build_graph()`
