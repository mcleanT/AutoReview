# Benchmark Orchestrator Design Spec

**Date**: 2026-03-17
**Status**: Draft
**Goal**: Single orchestrator script to run the full benchmark analysis suite (120-140 deduplicated pipeline runs, evaluations, and all 10 analyses) with minimum engineering effort.

---

## Context

The AutoReview benchmark paper requires ~137-202 pipeline runs across multiple topics, models, depths, and conditions, followed by evaluation and 10 analysis scripts. Currently there is no batch orchestration — each run must be invoked manually via CLI. The evaluation infrastructure (batch_runner, citation matching, ARISE rubric, structural metrics) is complete, but only Analysis 10 (depth comparison) has an implementation script.

**Approach chosen**: Single orchestrator script (Approach A) that reuses all existing infrastructure, deduplicates overlapping runs, and templates analyses 1-9 from Analysis 10's pattern.

---

## 1. Topics Schema (`paper/topics.yaml`)

Formalizes the curated candidate set into a machine-readable format.

```yaml
# paper/topics.yaml
metadata:
  created: "2026-03-17"
  description: "Benchmark evaluation topics for AutoReview paper"
  n_tier_a: 7-8    # 500+ citations, 2015-2019
  n_tier_b: 12-15  # 50-200 citations, 2023-2024

topics:
  # --- Tier A: Landmark Reviews ---
  - id: "microglia_homeostasis"
    title: "Microglia homeostasis and neurodegeneration"
    domain: biomedical
    tier: A
    reference:
      doi: "10.1038/s41586-017-0105-9"
      title: "Microglia: New Roles for the Synaptic Stripper"
      year: 2017
      citation_count: 2800
      pdf_path: "paper/references/colonna_2017.pdf"
    conditions: [end_to_end, retrieval_controlled]
    date_range: "-2017"

  # --- Tier B: Contemporary Reviews ---
  - id: "rag_architectures"
    title: "Retrieval-augmented generation architectures"
    domain: cs_ai
    tier: B
    reference:
      doi: "10.48550/arXiv.2312.10997"
      title: "Retrieval-Augmented Generation for Large Language Models: A Survey"
      year: 2023
      citation_count: 180
      pdf_path: "paper/references/gao_2023.pdf"
    conditions: [end_to_end]
    # No date_range for Tier B
```

### Pydantic Schema

```python
class ReferenceInfo(AutoReviewModel):
    doi: str
    title: str
    year: int
    citation_count: int
    pdf_path: str

class TopicEntry(AutoReviewModel):
    id: str                          # slug, used as directory name
    title: str                       # human-readable topic description
    domain: str                      # biomedical, cs_ai, materials, social_science
    tier: Literal["A", "B"]
    reference: ReferenceInfo
    conditions: list[str]            # end_to_end, retrieval_controlled
    date_range: str | None = None    # e.g. "-2017" for Tier A
    ablation: bool = False           # include in ablation runs

class TopicsConfig(AutoReviewModel):
    metadata: dict[str, Any]
    topics: list[TopicEntry]
```

---

## 2. Run Matrix & Deduplication

### Run Key

Each unique pipeline run is identified by a 4-tuple:

```python
RunKey = tuple[str, str, str, str]  # (topic_id, model, depth, condition)
```

### Run Batches

| Batch | Formula | Estimated Count | Analyses Fed |
|-------|---------|-----------------|-------------|
| 3a: Tier B end-to-end | 15 topics x 3 models x medium | 45 | A1,A2,A3,A6,A7,A8,A9 |
| 3b: Tier A end-to-end | 8 topics x 3 models x medium | 24 | A1,A2,A3,A5,A6,A7,A8,A9 |
| 3c: Tier A retrieval-controlled | 8 topics x 3 models x medium | 24 | A5 |
| 3d: ARISE baseline | 23 topics (external, not orchestrated) | 23 | A1,A3 |
| 3e: Ablation | 8 topics x 4 conditions x Sonnet x medium | 32 | A4 |
| 3f: Depth | 23 topics x {low, deep} x Sonnet | 46 | A10 |

**Deduplication savings:**
- 3f medium runs (23) reused from 3a/3b Sonnet runs → saves 23 runs
- Ablation baseline already exists as 3a/3b Sonnet/medium → saves 8 runs

**Deduplicated total: ~120-140 AutoReview runs + 23 ARISE runs (external)**

### Run Registry (`paper/results/run_registry.json`)

```json
{
  "runs": {
    "microglia_homeostasis|claude-sonnet-4-6|medium|end_to_end": {
      "status": "completed",
      "output_dir": "paper/results/microglia_homeostasis/claude-sonnet-4-6_medium_end_to_end",
      "review_path": "paper/results/.../review.md",
      "snapshot_path": "paper/results/.../snapshots/final.json",
      "evaluation_path": "paper/results/.../evaluation.json",
      "started_at": "2026-03-18T10:00:00",
      "completed_at": "2026-03-18T10:45:00",
      "cost_usd": 2.34,
      "tokens": {"input": 450000, "output": 85000}
    }
  },
  "metadata": {
    "last_updated": "2026-03-18T10:45:00",
    "total_completed": 45,
    "total_failed": 2,
    "total_pending": 93
  }
}
```

---

## 3. Orchestrator (`paper/run_benchmark.py`)

### CLI Interface

```
python -m paper.run_benchmark generate-matrix [--topics paper/topics.yaml]
python -m paper.run_benchmark run [--max-concurrent 2] [--batch 3a,3b] [--dry-run]
python -m paper.run_benchmark evaluate [--max-concurrent 5] [--judge-model claude-sonnet-4-6]
python -m paper.run_benchmark analyze [--analyses 1,2,3] [--output-dir paper/output]
python -m paper.run_benchmark full [--max-concurrent 2]
```

### `generate-matrix` Subcommand

Reads `paper/topics.yaml`, expands the full run matrix, applies dedup, and prints:
1. Summary table of all runs by batch
2. Dedup savings
3. Cost estimate (using `cost_analyzer.py` pricing)
4. Runs already completed (from registry)
5. Remaining runs

No execution — pure planning output.

### `run` Subcommand

```python
async def run_benchmark(
    topics_path: Path,
    results_dir: Path,
    max_concurrent: int = 2,
    batches: list[str] | None = None,  # Filter to specific batches
    dry_run: bool = False,
    models: list[str] = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
) -> None:
```

**Execution flow:**
1. Load `topics.yaml` and `run_registry.json`
2. Expand full matrix → apply dedup → subtract completed runs
3. If `--batch` specified, filter to those batches only
4. For each remaining run (semaphore-limited):
   a. Set up `DomainConfig` with correct model, depth, date_range
   b. Create `KnowledgeBase` with correct output_dir
   c. For retrieval-controlled: run `inject_bibliography.py` first, then `run_pipeline(start_from="full_text_retrieval")`
   d. For end-to-end: `run_pipeline()`
   e. On success: update registry (status=completed, paths, cost)
   f. On failure: update registry (status=failed, error message), continue
5. Save registry after each run (crash-safe)

**Imports vs subprocess**: Uses `from autoreview.pipeline.runner import run_pipeline` directly. This avoids subprocess overhead, gives structured error handling, and allows direct access to the KnowledgeBase for cost tracking.

### `evaluate` Subcommand

```python
async def evaluate_benchmark(
    results_dir: Path,
    max_concurrent: int = 5,
    judge_model: str = "claude-sonnet-4-6",
) -> None:
```

For each completed run in registry without an evaluation:
1. Resolve `generated_path` from the registry's `review_path` field (these are equivalent — the registry stores the path to the final review markdown)
2. Resolve `reference_path` by looking up the topic's `reference.pdf_path` in `topics.yaml`
3. Call `run_evaluation(generated_path=Path(review_path), reference_path=Path(ref_pdf_path), output_dir=run_dir, judge_llm=judge_llm)` (existing evaluator signature)
4. Run ARISE rubric via `ARISERubricScorer` on the generated review text
5. Save `evaluation.json` alongside the run output
6. Update registry with `evaluation_path`

**ARISE output evaluation:** ARISE runs have no registry entry. The orchestrator scans `paper/results/arise/{topic_id}/review.md` for any topic that exists in `topics.yaml`. For each found ARISE output, it runs the same evaluation pipeline (step 3-5), saving to `paper/results/arise/{topic_id}/evaluation.json`.

### `analyze` Subcommand

Runs analysis scripts in order. Each analysis reads from the results directory:

```python
async def analyze_benchmark(
    results_dir: Path,
    output_dir: Path,
    analyses: list[int] | None = None,  # Default: all 10
) -> None:
```

1. Aggregate all evaluation JSONs into a single DataFrame
2. Run each analysis script's `main()` function
3. Each writes to `output_dir/analysis_{N:02d}_{name}/`

### `full` Subcommand

Sequential: `run → evaluate → analyze`. Convenience wrapper.

---

## 4. Analysis Script Template

### Shared Utilities (`paper/analysis/common.py`)

Extracted from `depth_comparison.py`:

```python
# paper/analysis/common.py

# Colorblind-safe palette (matches depth_comparison.py)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

FONT_CONFIG = {
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}

def apply_style() -> None: ...
def fdr_correct(p_values: list[float]) -> list[float]: ...
def save_analysis_json(data: dict, path: Path) -> None: ...
def generate_markdown_report(title: str, sections: list[ReportSection], path: Path) -> None: ...

def load_all_evaluations(results_dir: Path) -> pd.DataFrame:
    """Scan results directory and build a unified DataFrame of all evaluations.

    Reads `run_registry.json` to discover all completed runs, then loads each
    run's `evaluation.json`. Also scans `results_dir/arise/` for ARISE outputs.

    Returns DataFrame with columns:
        - topic_id: str (from registry key)
        - domain: str (from topics.yaml lookup)
        - tier: str ("A" or "B")
        - system: str ("autoreview" or "arise")
        - model: str (e.g. "claude-sonnet-4-6", "arise" for ARISE runs)
        - depth: str ("low", "medium", "deep")
        - condition: str ("end_to_end", "retrieval_controlled", ablation conditions)
        - overall_score: float
        - citation_recall, citation_precision, citation_f1: float
        - synthesis_score, topic_coverage, writing_quality: float
        - arise_total: float | None (ARISE rubric score, if evaluated)
        - word_count, section_count, citation_count: int (structural metrics)
        - citations_per_1000_words, flesch_kincaid_grade: float
        - cost_usd: float | None (from registry)
        - tokens_input, tokens_output: int | None (from registry)

    AutoReview runs: parsed from registry key format "topic|model|depth|condition"
    ARISE runs: system="arise", model="arise", depth="medium", condition="end_to_end"
    """
    ...
```

**Note on Analysis 10 integration:** `depth_comparison.py` currently reads from a bespoke `depth_runs.json` format. The orchestrator's `analyze` subcommand must generate a `depth_runs.json` shim from the registry before calling `depth_comparison.main()`. This shim maps registry entries (topic, model=sonnet, depth=*, condition=end_to_end) to the `DepthRunConfig` format that `depth_comparison.py` expects. ~30 LOC adapter in the analyze subcommand.

### Per-Analysis Script Pattern

Each analysis follows this structure:

```python
"""Analysis N: {Name}.

{Description}

Usage:
    python -m paper.analysis.{module} --results-dir paper/results --output-dir paper/output/{name}
"""

from paper.analysis.common import apply_style, load_all_evaluations, ...

@dataclass
class AnalysisNOutput:
    """Container for Analysis N results."""
    ...

def load_data(results_dir: Path) -> pd.DataFrame:
    """Load and filter evaluation data for this analysis."""
    ...

def compute_metrics(df: pd.DataFrame) -> AnalysisNOutput:
    """Compute analysis-specific metrics."""
    ...

def compute_statistical_tests(df: pd.DataFrame) -> dict:
    """Run statistical tests specific to this analysis."""
    ...

def generate_report(output: AnalysisNOutput, path: Path) -> None:
    """Write markdown report."""
    ...

def write_analysis_json(output: AnalysisNOutput, path: Path) -> None:
    """Export all results as JSON for cross-analysis aggregation."""
    ...

def plot_*(df: pd.DataFrame, out: Path) -> None:
    """Generate publication figures."""
    ...

async def main(results_dir: Path, output_dir: Path) -> None:
    """Full analysis pipeline."""
    ...
```

### Analysis-Specific Details

**Analysis 1 — Main System Comparison**
- Filter: all topics, all systems (AutoReview/ARISE/human)
- Group by: system
- Metrics: All evaluation dimensions
- Stats: Friedman test + post-hoc Wilcoxon with FDR
- Figures: Radar chart (dimension scores), grouped bar (overall by system)
- Special: Must merge AutoReview eval results with ARISE eval results

**Analysis 2 — Cross-Domain Variation**
- Filter: all topics, AutoReview only (or all systems)
- Group by: domain x system
- Stats: Kruskal-Wallis per metric, domain x system interaction
- Figures: Heatmap (domain x metric), grouped bar (domain x system)

**Analysis 3 — Cross-Rubric Agreement**
- Filter: all topics evaluated with both rubrics
- Metrics: Spearman/Pearson correlation between AutoReview 4-dim and ARISE 7-cat
- Figures: Scatter (AutoReview score vs ARISE score), Bland-Altman plot

**Analysis 4 — Component Ablation**
- Filter: ablation topics only, Sonnet model
- Compare: full pipeline vs each ablation condition
- Metrics: Delta from baseline per metric
- Stats: Paired Wilcoxon per condition
- Figures: Heatmap (condition x metric delta)

**Analysis 5 — Retrieval vs Synthesis Decomposition** (Tier A only)
- Filter: Tier A topics, end-to-end vs retrieval-controlled
- Metrics: Quality delta, retrieval overlap (fraction of reference bib found by AI)
- Figures: Grouped bar (e2e vs controlled vs human)

**Analysis 6 — Citation Quality**
- Filter: all topics, all systems
- Metrics: Recall, precision, F1, hallucination rate from CitationScore
- Breakdown: by domain, by tier, by model
- Figures: Grouped bar, hallucination histogram

**Analysis 7 — Model Tier Comparison**
- Filter: all topics, AutoReview only, medium depth
- Group by: model tier
- Metrics: All eval dimensions + cost
- Stats: Friedman + post-hoc
- Figures: Line chart (model x metric), quality-cost frontier scatter

**Analysis 8 — Cost-Quality Tradeoff** (supplementary)
- Filter: all topics, all models
- Metrics: Cost from registry, quality from evaluations
- Derived: cost per quality-point, cost per citation-F1 point
- Figures: Scatter (cost vs quality, colored by model)

**Analysis 9 — Contamination Analysis** (supplementary)
- Filter: all topics, generated vs reference text
- Compute: N-gram overlap (1-5 grams) between generated and reference
- Breakdown: by tier (A should show higher overlap if memorized)
- Figures: Histogram of overlap scores, Tier A vs B comparison

**Analysis 10 — Depth Comparison** (ALREADY IMPLEMENTED)
- `paper/analysis/depth_comparison.py` — no changes needed
- Just needs to be wired into the orchestrator's `analyze` subcommand

---

## 5. Results Directory Structure

```
paper/
├── topics.yaml                                # Topic definitions
├── references/                                # Reference PDFs (gitignored)
│   ├── colonna_2017.pdf
│   ├── gao_2023.pdf
│   └── ...
├── results/                                   # All run outputs (gitignored)
│   ├── run_registry.json                      # Master run registry
│   ├── {topic_id}/
│   │   ├── {model}_{depth}_{condition}/
│   │   │   ├── review.md                      # Generated review
│   │   │   ├── snapshots/                     # Pipeline snapshots
│   │   │   ├── evaluation.json                # Evaluation result
│   │   │   └── cost.json                      # Token/cost data
│   │   └── ...
│   └── arise/                                 # ARISE outputs
│       ├── {topic_id}/
│       │   ├── review.md
│       │   └── evaluation.json
│       └── ...
├── output/                                    # Analysis outputs (committed)
│   ├── analysis_01_main_comparison/
│   │   ├── report.md
│   │   ├── analysis.json
│   │   ├── radar_chart.pdf
│   │   └── grouped_bar.pdf
│   ├── analysis_02_domain/
│   │   └── ...
│   ├── ...
│   └── analysis_10_depth_comparison/
│       └── ...
└── run_benchmark.py                           # The orchestrator
```

---

## 6. ARISE Integration

ARISE is a separate codebase. The orchestrator does NOT execute ARISE — it expects outputs to exist.

**Workflow:**
1. User clones ARISE repo, configures with Claude Sonnet 4.6
2. Runs ARISE on all 23 topics (manually or via ARISE's own runner)
3. Places outputs in `paper/results/arise/{topic_id}/review.md`
4. Orchestrator's `evaluate` subcommand detects ARISE outputs and evaluates them with the same rubrics

**Alternative (future):** Add a thin ARISE wrapper to the orchestrator. Out of scope for this spec.

---

## 7. Concurrency & Cost

### Concurrency Settings

| Phase | Default Concurrency | Rationale |
|-------|-------------------|-----------|
| Pipeline runs | 2 | Memory-intensive, API rate limits |
| Evaluations | 5 | Lighter LLM calls, mostly judge scoring |
| Analyses | 1 (sequential) | Fast, CPU-only, no parallelism needed |

### Cost Estimate

| Component | Per-Unit Cost | Count | Subtotal |
|-----------|--------------|-------|----------|
| Sonnet pipeline run | ~$2 | ~65 | ~$130 |
| Opus pipeline run | ~$15 | ~23 | ~$345 |
| Haiku pipeline run | ~$0.20 | ~23 | ~$5 |
| Deep pipeline run (Sonnet) | ~$5 | ~23 | ~$115 |
| Low pipeline run (Sonnet) | ~$0.80 | ~23 | ~$18 |
| Evaluation (2 rubrics) | ~$0.50 | ~140 | ~$70 |
| Claim extraction (A10) | ~$0.30 | ~69 | ~$21 |
| **Total estimate** | | | **~$700** |

The `generate-matrix` command computes this dynamically from current pricing.

---

## 8. Error Handling & Resume

### Run-Level Recovery

- Each run saves pipeline snapshots (existing feature)
- On crash, re-running the orchestrator detects incomplete runs via registry status
- Incomplete runs resume from last snapshot via `run_pipeline(start_from=...)`
- After 3 consecutive failures on the same run, mark as `permanently_failed` and skip

### Registry Crash Safety

- Registry written to disk after each run completion
- Uses atomic write (write to temp file, then rename)
- On orchestrator crash, registry reflects all completed runs

### Evaluation Recovery

- Same pattern: skip if `evaluation.json` already exists
- Evaluation failures don't affect other topics

---

## 9. Ablation Condition Mapping

The ablation conditions require pipeline configuration changes. These map to existing config knobs:

| Condition | Config Change |
|-----------|--------------|
| `no_evidence_chains` | `config.writing.evidence_chains = False` | **Prerequisite**: add `evidence_chains: bool = True` to `WritingConfig`, thread through section_writing node to skip chain construction when False. ~20 LOC in `config/models.py` + `nodes.py`. |
| `no_critique_loops` | `config.critique.max_revision_cycles = 0` | **Already supported** — existing config knob. |
| `no_passage_mining` | Skip passage_search node | **Prerequisite**: add `skip_nodes: list[str] = []` parameter to `run_pipeline()` and `DAGRunner.execute()`. When a node is in skip_nodes, the DAG skips it and passes through the KB unchanged. ~30 LOC in `runner.py` + `dag.py`. |
| `no_comprehensiveness` | Skip gap_search node | Same `skip_nodes` mechanism as above. |

**Prerequisite tasks (block ablation runs only):**
1. Add `evidence_chains` flag to `WritingConfig` + thread through section_writing node (~20 LOC)
2. Add `skip_nodes` parameter to `run_pipeline()` and `DAGRunner.execute()` (~30 LOC)

These are prerequisites for batch 3e (ablation) only. All other batches (3a, 3b, 3c, 3f) can proceed without them.

---

## 10. Implementation Priority

Ordered by dependency and effort:

| Priority | Component | Est. LOC | Dependencies |
|----------|-----------|----------|-------------|
| 1 | `paper/topics.yaml` + schema | ~80 | None |
| 2 | `paper/analysis/common.py` | ~150 | None |
| 3 | `paper/run_benchmark.py` — generate-matrix | ~200 | topics.yaml |
| 4 | `paper/run_benchmark.py` — run | ~300 | generate-matrix |
| 5 | `paper/run_benchmark.py` — evaluate | ~150 | run |
| 6 | Analyses 1-9 scripts | ~2200 total | common.py |
| 7 | `paper/run_benchmark.py` — analyze | ~100 | analysis scripts |
| 8 | Ablation config flags | ~50 | pipeline runner |
| **Total** | | **~3230** | |

---

## 11. What This Spec Does NOT Cover

- **ARISE execution**: User's responsibility to run ARISE and place outputs
- **Reference PDF collection**: Manual process (download 21 PDFs)
- **Primary pipeline changes**: No modifications to the 15-node DAG
- **LaTeX paper writing**: Phase 5, separate spec
- **Human evaluation**: Out of scope for this automated benchmark
