# Bayesian Inference Upgrade: HL-MRF to NumPyro Factor Graph

**Date:** 2026-03-30
**Status:** Approved
**Goal:** Replace the HL-MRF convex optimization system with a full Bayesian inference model (NumPyro/JAX) that produces posterior distributions — not just point estimates — for every edge in the knowledge graph. Includes agent-directed ground truth curation and hierarchical weight learning.

---

## Problem

The current HL-MRF system (`hlmrf.py` + `mrf_scoring.py`) produces MAP point estimates via L-BFGS-B convex optimization. This has three limitations for downstream automated agents (mycelium, autoscience):

1. **No uncertainty quantification at the graph level.** The Beta-Binomial CIs from `confidence.py` are edge-local — they don't account for how contradictions and composition chains affect uncertainty. An edge with `confidence_mean=0.7` looks identical whether it's supported by a strong transitive chain or actively contested by 3 opposing claims.

2. **No multimodality detection.** A contradicted edge with balanced evidence from both sides gets MAP ~0.5, indistinguishable from an edge with sparse, ambiguous evidence. Agents can't tell "genuine controversy" from "insufficient data."

3. **Hand-tuned weights.** The three rule weights (evidence=10.0, contradiction=8.0, composition=5.0) and evidence tier multipliers are fixed scalars. `mrf_weight_learning.py` learns them via grid search, but as global constants — no per-assertion-type or per-evidence-tier adaptation.

## Solution

Build a parallel Bayesian inference path using NumPyro/JAX that:
- Produces full posterior distributions (credible intervals, bimodality flags) per edge
- Reuses existing infrastructure (contradiction detection, composition chains, condition coupling)
- Includes an agent-directed ground truth curation pipeline
- Supports hierarchical priors that learn per-group weights from data
- Eventually replaces the HL-MRF entirely once validated

**Preservation requirement:** The HL-MRF system remains unchanged and operational throughout. It serves as the fast fallback and the baseline for model comparison. Removal happens only after validation against ground truth confirms the Bayesian model is strictly better.

---

## Architecture

### Module Map

```
autoreview/knowledge_graph/
├── confidence.py                # Tier 1: Beta-Binomial (UNCHANGED)
├── hlmrf.py                     # Tier 2: HL-MRF MAP engine (UNCHANGED)
├── mrf_scoring.py               # Tier 2: rule grounding (UNCHANGED)
├── bayesian/                    # NEW — Bayesian inference package
│   ├── __init__.py              # Public API
│   ├── model.py                 # NumPyro factor graph definition
│   ├── inference.py             # Sampling (NUTS), Laplace, subgraph extraction
│   ├── scoring.py               # Orchestration: graph → BayesianResult
│   ├── diagnostics.py           # R-hat, ESS, divergence checks, bimodality
│   └── config.py                # BayesianConfig, priors, phase flags
├── ground_truth/                # NEW — ground truth curation package
│   ├── __init__.py              # Public API
│   ├── schema.py                # GroundTruthSet, GroundTruthEdge models
│   ├── database_anchors.py      # Reactome/KEGG pathway matching
│   ├── rank_oracle.py           # Rank-order pair generation for validation
│   └── curator.py               # Agent-facing curation orchestration
├── model_comparison.py          # NEW — WAIC/LOO, Bayesian vs HL-MRF comparison
├── structural_contradictions.py # UNCHANGED — consumed by both paths
├── condition_compat.py          # UNCHANGED — consumed by both paths
├── predicate_algebra.py         # UNCHANGED — consumed by both paths
└── analysis.py                  # UNCHANGED — contradiction_centrality used for targeting
```

### Data Flow

```
                                  ┌─────────────────────────┐
                                  │ build_graph() pipeline   │
                                  │ Steps 1-7 (unchanged)   │
                                  └──────────┬──────────────┘
                                             │
                                    scored graph (Beta-Binomial)
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  │                  ▼
                   mrf_scoring.py            │         bayesian/scoring.py
                   (fast MAP)                │         (NUTS sampling)
                          │                  │                  │
                   MRFResult                 │          BayesianResult
                   {posteriors}              │          {posteriors,
                          │                  │           credible_intervals,
                          │                  │           posterior_samples,
                          │                  │           bimodality_flags}
                          │                  │                  │
                          └──────────────────┼──────────────────┘
                                             │
                                     model_comparison.py
                                     (validate Bayesian ≥ MRF)
                                             │
                                     ComparisonResult
```

### Consumer Interface

Downstream agents (mycelium, autoscience) interact via:

```python
from autoreview.knowledge_graph.bayesian import score_graph_bayesian, BayesianResult

result: BayesianResult = score_graph_bayesian(graph, config=BayesianConfig())

# Point estimates (same interface as MRFResult)
result.posteriors["edge_abc"]           # float, posterior mean

# Rich uncertainty (new)
result.credible_intervals["edge_abc"]   # (float, float), 95% CI
result.posterior_samples["edge_abc"]    # ndarray, raw MCMC samples
result.bimodality_flags["edge_abc"]     # bool, Hartigan's dip test
result.diagnostics                      # BayesianDiagnostics
```

---

## Bayesian Model (`bayesian/model.py`)

### Flat Model (Phase 1)

```python
def kg_flat_model(
    edge_ids: list[str],
    alpha_priors: jnp.ndarray,          # from Beta-Binomial α per edge
    beta_priors: jnp.ndarray,           # from Beta-Binomial β per edge
    contradiction_pairs: list[tuple],    # (idx_a, idx_b, coupling)
    composition_chains: list[tuple],     # (chain_indices, target_idx, strength, decay)
    w_contradiction: float = 8.0,
    w_composition: float = 5.0,
) -> None:
```

**Layer 1 — Edge truth priors:**

Each edge gets a Beta prior parameterized by its Beta-Binomial posterior from Tier 1:

```python
truth = numpyro.sample("truth", dist.Beta(alpha_priors, beta_priors))
```

This directly consumes the existing `confidence.py` output. An edge with strong supporting evidence (α=5.0, β=1.2) gets a tight prior near 0.8. An edge with weak evidence (α=1.1, β=1.0) gets a wide, nearly uniform prior.

**Layer 2 — Contradiction factors:**

For each contradiction pair (a, b) with condition coupling c:

```python
penalty = jax.nn.softplus(truth[a] + truth[b] - 1.0)
numpyro.factor(f"contra_{a}_{b}", -w_contradiction * c * penalty)
```

`softplus` replaces the HL-MRF's `max(0, ...)` hinge — smooth everywhere, better for HMC gradient flow. The coupling score from `condition_compat.condition_coupling()` gates the penalty strength.

**Layer 3 — Composition factors:**

For each composition chain `[e1, ..., en]` targeting edge `t`:

```python
chain_min = jnp.min(jnp.array([truth[e] for e in chain]))
expected = composition_strength * chain_min
shortfall = jax.nn.softplus(expected - truth[target])
numpyro.factor(f"comp_{target}", -w_composition * decay * shortfall)
```

Same AND semantics (weakest link) as the HL-MRF, same multi-hop BFS chain discovery, same decay factor.

### Hierarchical Model (Phase 2)

Extends the flat model by replacing fixed weights with group-level random variables:

```python
def kg_hierarchical_model(..., assertion_types, evidence_tiers):
    # Hyperpriors
    mu_evidence = numpyro.sample("mu_evidence", dist.Normal(10.0, 5.0))
    sigma_evidence = numpyro.sample("sigma_evidence", dist.HalfNormal(3.0))

    # Per-assertion-type evidence weights
    with numpyro.plate("assertion_types", n_types):
        w_evidence = numpyro.sample("w_evidence",
                                     dist.Normal(mu_evidence, sigma_evidence))

    # Per-evidence-tier strength multipliers (replaces EVIDENCE_WEIGHTS dict)
    with numpyro.plate("evidence_tiers", n_tiers):
        tier_multiplier = numpyro.sample("tier_multiplier",
                                          dist.LogNormal(prior_means, 0.5))

    # Independence discount (replaces _INDEPENDENCE_DISCOUNT = 0.5)
    independence_discount = numpyro.sample("independence_discount",
                                            dist.Beta(2.0, 2.0))

    # Contradiction and composition weights
    w_contradiction = numpyro.sample("w_contradiction", dist.LogNormal(2.0, 0.5))
    w_composition = numpyro.sample("w_composition", dist.LogNormal(1.5, 0.5))
```

Partial pooling: assertion types with many edges (e.g., mechanistic_causal) get precise group estimates. Sparse types (e.g., methodological) shrink toward the global mean `mu_evidence`.

---

## Inference (`bayesian/inference.py`)

### Laplace Approximation (Fast Path)

Computes the Hessian of the log-posterior at the MAP and inverts it for a Gaussian approximation:

```python
def laplace_approximate(
    graph: nx.MultiDiGraph,
    config: BayesianConfig,
) -> LaplaceResult:
```

Uses `jax.hessian` on the model's log-density. Returns per-edge mean + variance (credible intervals) without sampling. Cost: ~1 second for 2000 edges. Used for:
- Initial uncertainty estimates before full MCMC
- Fast incremental updates (recompute Hessian at perturbed MAP)
- Validating that MCMC posteriors are consistent (if the model is well-behaved, Laplace and MCMC should agree for unimodal posteriors)

### NUTS Sampling (Full Path)

```python
def sample_posterior(
    graph: nx.MultiDiGraph,
    config: BayesianConfig,
    target_nodes: set[str] | None = None,
) -> BayesianResult:
```

When `target_nodes` is None: full-graph MCMC (batch mode, minutes).
When `target_nodes` is provided: extracts the 2-hop subgraph around those nodes and samples only that subgraph (seconds). Used for:
- Agent-triggered updates ("I just resolved this contradiction, recompute")
- Targeted deep analysis of high contradiction_centrality nodes

**Sampling parameters (in BayesianConfig):**
- `n_warmup: int = 500`
- `n_samples: int = 1000`
- `n_chains: int = 4`
- `target_accept_prob: float = 0.8`
- `max_tree_depth: int = 10`

### Subgraph Extraction

```python
def extract_inference_subgraph(
    graph: nx.MultiDiGraph,
    target_nodes: set[str],
    hop_radius: int = 2,
) -> nx.MultiDiGraph:
```

Mirrors the `solve_incremental` logic from `hlmrf.py` — BFS from target nodes, include all edges within hop_radius. Returns a self-contained subgraph that can be passed to the model. Edges outside the subgraph become fixed observations (their Beta-Binomial means) rather than free variables.

---

## Diagnostics (`bayesian/diagnostics.py`)

### Per-Chain Diagnostics

After sampling, compute standard MCMC diagnostics via ArviZ:

- **R-hat** per variable: convergence across chains (flag if > 1.01)
- **ESS** (effective sample size) per variable: sampling efficiency (flag if < 400)
- **Divergences**: HMC trajectory failures (flag if > 0)
- **Tree depth saturation**: fraction of samples hitting max_tree_depth

### Bimodality Detection

For each edge posterior, run **Hartigan's dip test** on the samples:

```python
def detect_bimodality(
    samples: np.ndarray,
    significance: float = 0.05,
) -> bool:
```

Bimodal posteriors signal genuine scientific controversy — the evidence supports two incompatible truth values. This is the key signal agents can't get from MAP estimates.

### BayesianDiagnostics Dataclass

```python
@dataclass
class BayesianDiagnostics:
    converged: bool                           # all R-hat < 1.01, no divergences
    n_divergences: int
    min_ess: float
    max_rhat: float
    problematic_variables: list[str]          # variables with poor diagnostics
    bimodal_edges: list[str]                  # edges flagged by dip test
    sampling_time_seconds: float
```

---

## Ground Truth Curation (`ground_truth/`)

### Schema (`ground_truth/schema.py`)

```python
class GroundTruthEdge(AutoReviewModel):
    edge_id: str
    truth_value: float | None = None        # absolute confidence [0, 1]
    source: str                              # "database_anchor", "expert", "rank_oracle"
    justification: str                       # why this value
    anchor_database: str | None = None       # "reactome", "kegg", etc.
    curator: str | None = None               # "agent", "human", "{user_id}"
    confidence_in_judgment: float = 1.0      # how sure the curator is (meta-uncertainty)

class GroundTruthPair(AutoReviewModel):
    """Rank-order ground truth: edge_a should have higher confidence than edge_b."""
    edge_a_id: str
    edge_b_id: str
    justification: str
    source: str                              # "evidence_tier", "expert"

class GroundTruthSet(AutoReviewModel):
    corpus_id: str
    created_at: str
    edges: list[GroundTruthEdge]
    pairs: list[GroundTruthPair]             # rank-order constraints
    metadata: dict[str, Any] = {}
```

### Database Anchors (`ground_truth/database_anchors.py`)

```python
def match_reactome_pathways(
    graph: nx.MultiDiGraph,
    reactome_data: Path,
) -> list[GroundTruthEdge]:
```

Matches KG edges to known Reactome pathway relationships by entity name/ontology ID. Edges matching established pathway steps get truth ~0.90-0.95. Edges contradicting established pathways get truth ~0.05-0.15.

**KEGG equivalent** for metabolic/signaling pathways.

Entity matching uses the existing `EntityRegistry` surface-to-ID mapping + ontology IDs where available. Fuzzy matching (rapidfuzz) for entity names without ontology IDs.

### Rank Oracle (`ground_truth/rank_oracle.py`)

```python
def generate_rank_pairs(
    graph: nx.MultiDiGraph,
    n_pairs: int = 100,
    strategy: str = "evidence_tier",
) -> list[GroundTruthPair]:
```

Generates edge pairs where the rank ordering is clear from structural features:
- **evidence_tier**: edge with `direct_experimental` from `primary_empirical` > edge with `review_citation` from `attributed_prior`
- **replication**: edge with 3 independent author groups > edge with 1
- **consistency**: edge with no direction_conflict > edge with direction_conflict

These pairs don't require reading papers — the ordering follows from the evidence structure.

### Curator (`ground_truth/curator.py`)

```python
def curate_ground_truth(
    graph: nx.MultiDiGraph,
    corpus_config: CorpusConfig,
    human_review_callback: Callable | None = None,
) -> GroundTruthSet:
```

Orchestrates the three-tier curation:
1. Run `match_reactome_pathways()` + `match_kegg_pathways()` → database anchors
2. Run `generate_rank_pairs()` → rank-order constraints
3. Select top-N edges by `contradiction_centrality` → candidates for human review
4. If `human_review_callback` provided, present candidates and collect judgments
5. Merge all three sources into a `GroundTruthSet`

The `human_review_callback` is the agent interface — mycelium/autoscience can implement this to present candidates to the user or make autonomous judgments for clear-cut cases.

---

## Model Comparison (`model_comparison.py`)

### Comparison Against Ground Truth

```python
def compare_models(
    graph: nx.MultiDiGraph,
    ground_truth: GroundTruthSet,
    bayesian_result: BayesianResult,
    mrf_result: MRFResult,
) -> ComparisonResult:
```

**Absolute metrics (using GroundTruthEdge with truth_value):**
- MSE of posteriors vs ground truth
- Calibration: do 95% CIs contain the ground truth 95% of the time?
- Brier score

**Rank metrics (using GroundTruthPair):**
- Pairwise accuracy: fraction of pairs where the model ranks correctly
- NDCG (normalized discounted cumulative gain)
- Kendall's tau

**Model comparison (Bayesian):**
- WAIC (widely applicable information criterion) via ArviZ
- LOO-CV (leave-one-out cross-validation) via ArviZ
- Both computed on the Bayesian model's posterior samples

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    bayesian_mse: float
    mrf_mse: float
    bayesian_calibration: float             # empirical coverage of 95% CI
    bayesian_pairwise_accuracy: float
    mrf_pairwise_accuracy: float
    waic: float
    loo: float
    recommendation: str                      # "bayesian_better", "mrf_better", "inconclusive"
    details: dict[str, Any]
```

---

## Integration with build_graph()

### Phase 1 (Laplace + Flat MCMC)

`build_graph()` gains an optional `bayesian` parameter:

```python
def build_graph(
    extraction_dir: Path,
    use_mrf: bool = False,
    mrf_config: MRFConfig | None = None,
    version: int = 1,
    bayesian: bool = False,                  # NEW
    bayesian_config: BayesianConfig | None = None,  # NEW
) -> nx.MultiDiGraph:
```

When `bayesian=True`:
- Runs Laplace approximation on the full graph (fast, ~1 second)
- Identifies top-K controversy hotspots via `score_contradiction_centrality()`
- Runs NUTS sampling on each hotspot subgraph
- Writes `bayesian_confidence`, `bayesian_ci_low`, `bayesian_ci_high`, `bayesian_bimodal` as edge attributes

When `bayesian=False`: behavior is unchanged.

### Phase 2 (After Validation)

Once `compare_models()` confirms `recommendation == "bayesian_better"`:
- `bayesian=True` becomes the default
- Full-graph NUTS replaces Laplace for batch builds
- `use_mrf` becomes a legacy fast-path option
- The HL-MRF code remains but is no longer the primary inference system

### Incremental Updates

```python
def update_graph_bayesian(
    graph: nx.MultiDiGraph,
    changed_edge_ids: list[str],
    prior_result: BayesianResult,
    config: BayesianConfig | None = None,
) -> BayesianResult:
```

Extracts the 2-hop subgraph around changed edges, runs NUTS on that subgraph with the prior result's posteriors as initial values, merges updated posteriors back. Parallel to `update_graph_mrf()`.

---

## Configuration (`bayesian/config.py`)

```python
class BayesianConfig(AutoReviewModel):
    # Sampling
    n_warmup: int = 500
    n_samples: int = 1000
    n_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10

    # Model
    hierarchical: bool = False               # Phase 1: False, Phase 2: True
    w_contradiction: float = 8.0             # fixed weight (flat model)
    w_composition: float = 5.0               # fixed weight (flat model)

    # Targeting
    hotspot_top_k: int = 10                  # number of hotspot subgraphs for targeted MCMC
    hotspot_hop_radius: int = 2              # BFS radius around hotspot nodes
    bimodality_significance: float = 0.05    # Hartigan's dip test threshold

    # Laplace
    use_laplace_fallback: bool = True        # Laplace for non-hotspot edges

    # Corpus
    corpus_id: str = "default"               # for ground truth association
```

---

## Phasing

### Phase 1: Laplace + Flat MCMC (no ground truth needed)

**Deliverables:**
- `bayesian/model.py` — flat NumPyro factor graph
- `bayesian/inference.py` — Laplace approximation + NUTS sampling + subgraph extraction
- `bayesian/scoring.py` — `score_graph_bayesian()` orchestration
- `bayesian/diagnostics.py` — R-hat, ESS, divergences, bimodality detection
- `bayesian/config.py` — BayesianConfig
- Integration into `build_graph(bayesian=True)`

**Validation:** Compare Laplace intervals to Beta-Binomial CIs. Compare NUTS posterior means to HL-MRF MAP. They should be close for well-behaved subgraphs — divergence indicates model issues.

**Dependencies:** `numpyro`, `jax`, `arviz`

### Phase 2: Ground Truth Curation

**Deliverables:**
- `ground_truth/schema.py` — GroundTruthSet, GroundTruthEdge, GroundTruthPair models
- `ground_truth/database_anchors.py` — Reactome/KEGG matching
- `ground_truth/rank_oracle.py` — structural rank-pair generation
- `ground_truth/curator.py` — orchestration with human review callback
- `model_comparison.py` — WAIC/LOO, MSE, rank metrics

**Validation:** Curate ground truth for gastruloid corpus. Run comparison. Determine if Bayesian model improves on HL-MRF.

**Dependencies:** Reactome/KEGG data files (downloadable), high-quality extraction corpus

### Phase 3: Hierarchical Priors + Weight Learning

**Deliverables:**
- Hierarchical model variant in `bayesian/model.py`
- WAIC/LOO comparison: hierarchical vs flat
- Learned weights exportable as `BayesianConfig` for new corpora

**Validation:** WAIC improvement over flat model. Learned evidence tier multipliers should correlate with the hand-tuned `EVIDENCE_WEIGHTS` (sanity check).

**Dependencies:** Phase 1 + Phase 2 complete. Ground truth curated.

### Phase 4: HL-MRF Retirement

**Precondition:** `compare_models()` returns `recommendation == "bayesian_better"` consistently across multiple corpus builds.

**Action:** Make `bayesian=True` the default in `build_graph()`. HL-MRF remains available via `use_mrf=True` as a fast-path option. No code deletion — just a default change.

---

## New Dependencies

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `jax` | >=0.4.20 | Autodiff, JIT compilation | ~150MB |
| `jaxlib` | >=0.4.20 | JAX backend (CPU) | ~300MB |
| `numpyro` | >=0.15.0 | Probabilistic programming, NUTS | ~5MB |
| `arviz` | >=0.18.0 | MCMC diagnostics, WAIC/LOO | ~20MB |

GPU acceleration (optional): `jaxlib[cuda12]` for NVIDIA GPUs. Not required — CPU is sufficient for subgraph-scale inference (20-80 edges).

---

## Testing Strategy

- **Unit tests for model.py:** verify factor graph produces valid log-densities, gradient checks
- **Unit tests for inference.py:** Laplace on toy graph matches analytical solution, NUTS converges on 3-node graph
- **Unit tests for diagnostics.py:** bimodality detection on synthetic bimodal/unimodal samples
- **Unit tests for ground_truth/:** database anchor matching on mock data, rank pair generation
- **Integration test:** full pipeline on micro corpus — build_graph → score_graph_bayesian → diagnostics pass
- **Regression test:** HL-MRF path unchanged — score_graph_mrf produces identical results
- **Comparison test:** on toy graph with known ground truth, Bayesian model achieves lower MSE than HL-MRF

## Backwards Compatibility

- `build_graph()` default is `bayesian=False` — zero behavior change
- `MRFResult`, `score_graph_mrf()`, `update_graph_mrf()` — unchanged
- Existing graph pickle files remain loadable — new Bayesian attributes are additive
- All existing tests pass without modification

## Risks

1. **JAX installation conflicts:** JAX can conflict with TensorFlow or other CUDA consumers. Mitigate: pin versions, test in CI.
2. **MCMC convergence on real graphs:** Pathological graph structures (many tight contradictions) may cause divergences. Mitigate: diagnostics module flags issues, Laplace fallback for problematic subgraphs.
3. **Reactome/KEGG entity matching quality:** Fuzzy matching may produce false anchor edges. Mitigate: require ontology ID match for high-confidence anchors, flag fuzzy matches for review.
4. **Bimodality false positives:** Hartigan's dip test on small sample sizes. Mitigate: require n_samples >= 1000, significance threshold 0.05.
