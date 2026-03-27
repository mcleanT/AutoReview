---
name: nli-score
description: >
  Run cross-claim NLI contradiction detection on a knowledge graph. Scores all
  claim pairs using DeBERTa, applies false positive filtering, updates Beta-Binomial
  posteriors, and regenerates the interactive HTML visualization.
---

# NLI Score — Cross-Claim Contradiction Detection

Score a knowledge graph with the cross-claim NLI pipeline: detect contradictions between claims, apply false positive filtering, and update Beta-Binomial confidence posteriors.

## Prerequisites

- A knowledge graph pickle (`.pkl`) from the KG build pipeline
- PyTorch with MPS/CUDA support (CPU fallback available but ~6x slower)
- `cross-encoder/nli-deberta-v3-base` model (auto-downloaded on first run)

## Quick Path — CLI

If the user provides an input path, run the CLI directly:

```bash
python -m autoreview.cli nli-score <input_path> [--device auto] [--html] [--json-report]
```

**Common invocations:**
```bash
# Default: score, save, generate HTML + JSON report
autoreview nli-score output/knowledge_graph/gastruloid_kg.pkl

# Custom output path, MPS device
autoreview nli-score output/knowledge_graph/gastruloid_kg.pkl -o output/knowledge_graph/scored -d mps

# Skip HTML generation
autoreview nli-score output/knowledge_graph/gastruloid_kg.pkl --no-html

# Evidence-level diagnostic only (read-only, no graph modification)
autoreview nli-diagnose output/knowledge_graph/gastruloid_kg.pkl
```

## Python API Path

If more control is needed or this is part of a larger workflow:

```python
from autoreview.knowledge_graph import load_graph, save_graph, classify_cross_claims, NLIConfig
from autoreview.knowledge_graph.interactive import generate_interactive_html

# Load
graph = load_graph("output/knowledge_graph/your_kg.pkl")

# Configure
config = NLIConfig(
    device="mps",           # "auto", "mps", "cuda", "cpu"
    batch_size=64,          # pairs per inference batch
    contradiction_threshold=0.3,  # min p_contra for Beta updates
    filter_parallel_assertions=True,  # skip "X does A" vs "X does B"
    use_predicate_opposition=True,    # deterministic induces/inhibits detection
)

# Score
result = classify_cross_claims(graph, config)

# Save
save_graph(graph, "output/knowledge_graph/scored_kg")
generate_interactive_html(graph, "output/knowledge_graph/interactive_kg.html")
```

## What the Pipeline Does

1. **Build claim pairs** — Each graph edge is a claim (`subject predicate object`). Claims sharing entities become pairs (~30K pairs for a 2,900-edge graph).

2. **Pre-filter false positives:**
   - **Parallel assertions** — Same subject+predicate, different object (e.g., "X generates cardiac" vs "X generates skeletal") are complementary, not contradictory. Skipped.
   - **Structural predicate opposition** — Same entity pair with opposing predicates (induces/inhibits, required/not_required) are deterministic contradictions. No model needed.

3. **NLI classification** — Remaining pairs run through DeBERTa cross-encoder. ~76 seconds on MPS for 28K pairs.

4. **Beta-Binomial update** — Contradiction-only: opposing evidence increases beta_param (counter-evidence), never boosts alpha from entailment.

5. **Output** — Updated graph edges carry: `confidence_mean`, `controversy_score`, `_nli_alpha`, `_nli_beta`, `_nli_cross_beta`.

## Result Interpretation

| Metric | Meaning |
|--------|---------|
| `confidence_mean` | α/(α+β) — 1.0 = fully supported, 0.0 = fully contradicted |
| `controversy_score` | min(α,β)/max(α,β) — 0.0 = unanimous, 1.0 = perfectly balanced evidence |
| `_nli_cross_beta` | Total contradiction weight accumulated from opposing claims |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `claims_updated: 0` | Graph edges missing `_kg_edge` attribute — ensure graph was built with `build_graph()`, not manually |
| All `controversy_score = 0` | No contradictions found — check if graph has diverse claims |
| Very slow | Ensure MPS/CUDA is available: `python -c "import torch; print(torch.backends.mps.is_available())"` |
| Import error for torch | Install: `pip install torch transformers` |

## Evidence-Level Diagnostic

To check if evidence summaries contain enough semantic content for NLI:

```python
from autoreview.knowledge_graph import load_graph, diagnose_evidence_directions, NLIConfig

graph = load_graph("output/knowledge_graph/your_kg.pkl")
result = diagnose_evidence_directions(graph, NLIConfig(device="mps"))
print(result.label_distribution)
# If 95%+ neutral: evidence summaries are methodological, not conclusive
# Cross-claim NLI (classify_cross_claims) works better in this case
```
