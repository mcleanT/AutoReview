# Analysis 10: Depth Level Comparison

**Date**: 2026-03-17
**Status**: Review
**Related**: Paper plan (`Paper Idea/paper.md`), Depth system (`autoreview/config/depth.py`)

---

## Purpose

Add a new paper analysis (Analysis 10) comparing AutoReview output quality, information content, and cost across the three depth levels (low, medium, deep). This analysis answers: does deeper generation produce proportionally better reviews, or do returns diminish? And critically — how much *new knowledge* does each depth increment actually add?

## Research Questions

1. **Quality vs. depth**: Do synthesis, coverage, writing quality, and citation scores scale with depth?
2. **Diminishing returns**: Is the low→medium jump larger than medium→deep? Where does the curve plateau?
3. **Information gain**: How many new factual claims, scientific concepts, and entities appear at each depth increment?
4. **Information density**: Do deeper reviews maintain claims-per-word density or become diluted?
5. **Cost-efficiency**: What is the cost per quality-point and cost per new claim at each depth?
6. **Domain interaction**: Which domains benefit most from deeper treatment?
7. **Structural shifts**: How do word count, citation density, section balance, and readability change across depths?

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Topics | All (17-23), balanced across 4 domains |
| Depths | low, medium, deep |
| Model | Sonnet 4.6 only (isolates depth as the variable) |
| Total runs | 51-69 (topics × 3 depths) |
| Evaluation rubrics | AutoReview 4-dimension + ARISE 7-category |
| Judge model | Sonnet 4.6 (consistent with all other analyses) |

### Conditions

Each topic is run three times with identical configuration except for `--depth`:

| Condition | Depth | Expected words | Evidence detail |
|-----------|-------|---------------|-----------------|
| Shallow | low | ~4,000 | critical_only |
| Standard | medium | ~8,000 | standard |
| Comprehensive | deep | ~25,000 | exhaustive |

All other pipeline parameters (search, critique, passage mining) remain constant. This isolates depth as the sole experimental variable.

### Relationship to Other Analyses

- **Analysis 7 (Model Tier)** varies the model while holding depth constant (medium). Analysis 10 varies depth while holding the model constant (Sonnet). Together they map the 2D quality-cost surface.
- **Analysis 4 (Ablation)** removes pipeline components. Analysis 10 keeps all components but changes how much each contributes.
- **Run reuse strategy**: The depth parameter affects `EvidenceWeightedAllocator` (outline node) and all downstream writing, but does NOT affect search, screening, full-text retrieval, extraction, or clustering. The `evidence_chain_detail` field in `DepthProfile` controls prose instructions in the writing stage, not evidence chain construction in the analysis stage. Therefore, each topic needs only ONE search→clustering pipeline run, and depth-specific runs fork from the outline node onward. Medium-depth Sonnet runs from the main comparison (Phase 3a/3b) are reused directly. Low and deep runs resume from the same pre-outline snapshot with `--depth` override.

## Information Metrics

Three new metrics that do not exist in the current evaluation framework. These measure *knowledge content* rather than *quality*.

### 1. Claim Extraction (primary)

Use an LLM to extract distinct factual assertions from each generated review.

**Input**: Full review text at a given depth.
**Output**: List of claims, each tagged with a category.

**Claim categories**:
- **Empirical**: Specific findings, results, statistics ("X showed a 30% improvement over Y")
- **Methodological**: Descriptions of approaches, techniques, tools ("Single-cell RNA-seq enables...")
- **Contextual**: Historical context, background, definitions ("First described in 1998...")
- **Synthesis**: Cross-study comparisons, contradictions, consensus ("Despite conflicting results...")
- **Limitation**: Caveats, gaps, open questions ("No studies have examined...")

**Derived metrics**:
- Total claims per depth level
- New claims per depth increment using cumulative union baseline:
  - New at medium = claims in medium not semantically present in low
  - New at deep = claims in deep not semantically present in union(low, medium)
- New claims by category per increment (what *kind* of knowledge gets added?)
- Claim novelty rate: fraction of claims at depth N+1 not present in union of all shallower depths
- **Reference claim coverage** (Tier A topics only): extract claims from the human reference review, then report what fraction appear at each depth level. Transforms relative metrics into a grounded absolute metric.

**Implementation**: Section-chunked LLM extraction — each review is split by section headings and claims are extracted per section, then aggregated. This avoids context degradation on deep reviews (~25K words). Uses Haiku as the extraction model (not Sonnet) to avoid self-evaluation bias, since Sonnet generates the reviews. Claim deduplication via embedding similarity (cosine threshold 0.90). Borderline cases (0.85-0.95 band) are flagged; a random sample of 50 borderline pairs is manually validated to calibrate the threshold.

### 2. Concept/Entity Coverage

Extract distinct scientific concepts, methods, named entities, and technical terms.

**Input**: Full review text.
**Output**: Set of normalized concept strings.

**Derived metrics**:
- Concept count per depth
- Concept set growth: |concepts_medium - concepts_low|, |concepts_deep - concepts_medium|
- Concept overlap coefficient between depth pairs

**Implementation**: LLM extraction of concepts as a flat list, normalized to lowercase canonical forms. Deduplication via exact match after normalization (simpler than claim dedup since concepts are shorter).

### 3. Information Density

Ratios that test whether deeper reviews maintain or dilute knowledge concentration.

**Derived metrics**:
- Claims per 1,000 words
- Concepts per 1,000 words
- Citations per claim (how well-supported are assertions?)

**Interpretation guide**:
- Density maintained across depths → deeper reviews add proportional knowledge
- Density drops at deep → deeper reviews are more verbose without proportional content gain
- Density increases at deep → deeper reviews are more efficiently packed (unlikely but worth checking)

## Output Artifacts

### Figures

1. **Quality × depth line chart**: One line per metric dimension (synthesis, coverage, writing, citation F1). X-axis: depth level. Y-axis: normalized score. Error bars across topics. Shows whether quality scales linearly, sublinearly, or plateaus.

2. **Cumulative claims stacked bar**: Three bars (low/medium/deep), stacked by claim category. Visually shows what kinds of knowledge each depth increment adds.

3. **Information density scatter**: X-axis: word count (continuous, per review). Y-axis: claims/1000 words. Color: domain. Marker shape: depth level. Shows whether verbosity and density are correlated across the full range, more diagnostic than a categorical x-axis with only 3 values.

4. **Cost-efficiency dual-axis chart**: Left Y-axis: quality score. Right Y-axis: cost ($). X-axis: depth. Shows the quality-cost tradeoff. Optional: overlay claims-per-dollar as a third series.

5. **Domain × depth heatmap**: Rows: domains. Columns: depth levels. Cell value: quality score or claim count. Highlights which domains benefit most from deeper treatment.

### Tables

1. **Summary statistics**: Per-depth mean/SD for all metrics (quality scores, claims, concepts, density, cost, time).
2. **Pairwise depth comparisons**: Wilcoxon signed-rank test (primary, robust to non-normality with N=17-23) for low→medium and medium→deep jumps per metric. Paired t-test reported as secondary check. FDR correction (Benjamini-Hochberg) applied across all tests (~28 tests: 14 metrics × 2 depth jumps) to control false discovery rate.
3. **Per-domain breakdown**: Mean scores at each depth, per domain.

### JSON Output

```
paper/results/depth_comparison/
  depth_runs.json          # metadata for all runs (topic, depth, output path, cost)
  depth_evaluation.json    # evaluation scores per run
  depth_claims.json        # extracted claims per run
  depth_concepts.json      # extracted concepts per run
  depth_analysis.json      # computed metrics, statistical tests, derived values
```

## Script Design

**File**: `paper/analysis/depth_comparison.py`

### Functions

```python
# --- Data loading ---
def load_depth_runs(results_dir: Path) -> pd.DataFrame
    """Load evaluation results for all depth runs into a single DataFrame."""

# --- Information extraction ---
async def extract_claims(text: str, llm_client) -> list[Claim]
    """Extract factual claims from a review, categorized by type."""

async def extract_concepts(text: str, llm_client) -> list[str]
    """Extract scientific concepts and entities from a review."""

async def deduplicate_claims(
    target_claims: list[Claim],
    baseline_claims: list[Claim],
    embedder,
    threshold: float = 0.90,
) -> tuple[list[Claim], list[Claim]]
    """Return (novel_in_target, shared_with_baseline) using embedding similarity.

    Baseline can be a union of claims from multiple shallower depths.
    For medium→deep: pass union(low_claims, medium_claims) as baseline.
    """

# --- Metrics ---
def compute_information_metrics(
    claims_by_depth: dict[str, list[Claim]],
    concepts_by_depth: dict[str, list[str]],
    word_counts: dict[str, int],
) -> InformationMetrics
    """Compute all derived information metrics across depths."""

def compute_depth_statistics(df: pd.DataFrame) -> DepthStatistics
    """Aggregate quality scores, costs, and structural metrics by depth."""

def run_pairwise_tests(df: pd.DataFrame) -> PairwiseResults
    """Wilcoxon signed-rank (primary) + paired t-test (secondary) for low→medium
    and medium→deep. FDR correction (Benjamini-Hochberg) across all tests."""

# --- Figures ---
def plot_quality_by_depth(stats: DepthStatistics, out: Path) -> None
def plot_cumulative_claims(claims_data: dict, out: Path) -> None
def plot_information_density(metrics: InformationMetrics, out: Path) -> None
def plot_cost_efficiency(stats: DepthStatistics, out: Path) -> None
def plot_domain_depth_heatmap(df: pd.DataFrame, out: Path) -> None

# --- Report ---
def generate_report(
    stats: DepthStatistics,
    info_metrics: InformationMetrics,
    pairwise: PairwiseResults,
    out: Path,
) -> None
    """Generate markdown summary report with tables and figure references."""

def write_analysis_json(
    stats: DepthStatistics,
    info_metrics: InformationMetrics,
    pairwise: PairwiseResults,
    out: Path,
) -> None
    """Write depth_analysis.json with all computed metrics and test results."""

# --- Entry point ---
async def main(results_dir: Path, output_dir: Path) -> None
    """Load runs, extract claims/concepts, compute metrics, write JSON + figures + report."""
```

### Data Models

```python
class Claim(AutoReviewModel):
    """A distinct factual assertion extracted from a review."""
    text: str
    category: Literal["empirical", "methodological", "contextual", "synthesis", "limitation"]
    section_id: str | None = None  # source section, populated during chunked extraction

class InformationMetrics(AutoReviewModel):
    """Aggregated information metrics for a single topic across depth levels."""
    claims_per_depth: dict[str, int]
    new_claims_per_increment: dict[str, int]       # "low→medium": N, "medium→deep": N
    new_claims_by_category: dict[str, dict[str, int]]
    claim_novelty_rate: dict[str, float]
    concepts_per_depth: dict[str, int]
    concept_growth: dict[str, int]
    claims_per_1k_words: dict[str, float]
    concepts_per_1k_words: dict[str, float]
    citations_per_claim: dict[str, float]
    # Tier A only: fraction of reference review claims found at each depth
    reference_claim_coverage: dict[str, float] | None = None
```

## Integration with Paper Plan

### Paper structure additions

- **Section 5** gains subsection **5k. Depth comparison**: quality scaling, information gain, cost-efficiency across low/medium/deep.
- **Introduction contributions list**: add contribution: "(7) depth-level analysis revealing the quality-cost-information surface and identifying diminishing returns thresholds."
- **Figures**: quality×depth line chart and cumulative claims bar go in main paper. Density scatter, cost-efficiency, and domain heatmap are supplementary candidates.
- **Discussion**: adds a paragraph on depth as a user-facing knob — when is "medium" good enough? When does "deep" pay off?

### Phase 3 additions

- **3f. AutoReview runs — depth comparison** (17-23 topics × 3 depths × Sonnet = 51-69 runs)
- Note: the medium-depth Sonnet runs overlap with Phase 3a/3b. Those runs should be reused, not duplicated. Net new runs: ~34-46 (low + deep only).

### Run count impact

Current estimated runs: 103-156. Adding depth comparison (net new): +34-46. New total: **137-202 runs**.

### Cost estimate

**Pipeline runs**: At ~$2-5 per run (Sonnet), net new depth runs (low + deep only) cost ~$68-230.

**Claim/concept extraction**: Deep reviews (~25K words, ~35K tokens input) are the expensive case. Per-review extraction cost at Haiku pricing (~$0.25/M input, $1.25/M output): ~$0.01 input + ~$0.05-0.15 output per review (100-500 claims). Across 51-69 reviews at mixed depths: ~$2-8. Reference claim extraction for Tier A topics adds ~$1-3.

**Embedding costs**: ~500 claims/review × 3 depths × 20 topics = ~30K embeddings. At typical embedding API rates: ~$1-3.

**Total incremental cost**: Pipeline runs ($68-230) + extraction ($3-11) + embeddings ($1-3) = **~$72-244**. The extraction/embedding costs are negligible compared to pipeline runs.

## Decisions

- **Haiku for claim extraction** (not Sonnet) — avoids self-evaluation bias since Sonnet generates the reviews. Haiku is cheap, fast, and different enough that self-favor bias is broken. All depth conditions are extracted with the same model, so any systematic Haiku bias affects all conditions equally and cancels out in comparative analysis.
- **Section-chunked extraction** for deep reviews — a single LLM call on 25K words risks context degradation. Splitting by section headings and aggregating preserves extraction quality. Shallow/medium reviews can use single-call extraction since they fit comfortably in context.
- **Claim deduplication via embeddings** rather than exact match — claims at different depths will be paraphrased, not identical. Threshold 0.90 cosine similarity. Borderline cases (0.85-0.95 band) are flagged; a random sample of 50 borderline pairs is manually validated to calibrate the threshold before running full analysis.
- **Cumulative union baseline for novelty** — "new at deep" means claims in deep not present in union(low, medium), not just claims in deep not present in medium. This prevents double-counting claims that appear first at medium and again (paraphrased) at deep.
- **Sonnet-only** to isolate depth as the variable. The depth×model interaction surface is interesting but doubles the run count. If Analysis 7 (model tier) and Analysis 10 (depth) both show significant effects, a targeted depth×model interaction study on a subset could follow.
- **Reuse medium-depth Sonnet runs** from the main comparison to avoid redundant API calls.
- **LLM-based claim extraction** rather than NLP-based (e.g., OpenIE) — scientific claims are nuanced and require domain understanding that rule-based systems miss.
- **Wilcoxon signed-rank as primary test** — robust to non-normality with N=17-23. FDR correction (Benjamini-Hochberg) across all ~28 tests.
- **Pydantic models** (AutoReviewModel subclasses) for Claim and InformationMetrics, matching the project convention in `evaluation/models.py`.
