# Design: Year Filtering & Bibliography Injection for Evaluation

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Two features — one core pipeline enhancement (year filtering), one evaluation-only tool (bibliography injection)

---

## 1. Year Filtering in SearchAggregator (Core Pipeline)

### Problem

`SearchConfig.date_range` exists as a string (e.g., `"2015-2025"`) but is only passed to LLM prompt builders for query expansion. No actual filtering occurs — search sources may return papers outside the specified range.

### Design

**Parsing `date_range` into bounds:**

| Input | `year_from` | `year_to` |
|-------|-------------|-----------|
| `"2015-2025"` | `2015` | `2025` |
| `"-2020"` | `None` | `2020` |
| `"2020-"` | `2020` | `None` |
| `""` or `None` | `None` | `None` (no filtering) |

**Filter location:** Post-filter in `SearchAggregator`, applied after each source returns results and before deduplication.

**`year=None` handling:** Always drop papers with `year=None`. Log a structured warning for each dropped paper including title, source database, and DOI (if available). This ensures temporal control is strict and every exclusion is traceable.

**Rationale for post-filter over native API params:**
- Perplexity has no reliable year filter
- One filter location vs. four source-specific implementations
- Acts as a safety net regardless of source behavior
- Native API filters can be added later per-source as an optimization (reduces API traffic)

**Range semantics:** Inclusive on both bounds. `"2015-2020"` includes papers from 2015 and 2020.

**Validation edge cases** (handled by Pydantic validator on `SearchConfig.date_range`):
- Whitespace stripped: `" 2015 - 2025 "` → `"2015-2025"`
- Reversed ranges rejected: `"2025-2015"` → `ValidationError`
- Same year valid: `"2020-2020"` → filters to exactly 2020
- Malformed input rejected: `"2015-01-2025"` → `ValidationError`

### Files to modify

| File | Change |
|------|--------|
| `autoreview/search/aggregator.py` | Add `_parse_date_range()` and `_filter_by_year()` methods; accept `date_range` in constructor; apply filter after each source returns |
| `autoreview/pipeline/nodes.py` | Pass `config.search.date_range` to `SearchAggregator` at **all 5 instantiation sites**: primary search (L290), gap search (L582), contextual enrichment (L800), corpus expansion (L981), passage search (L1176) |
| `autoreview/config/models.py` | Add Pydantic validator on `SearchConfig.date_range` to enforce format and reject reversed/malformed ranges |
| `autoreview/cli.py` | Add `--date-range` option to both `run` and `resume` commands; overrides `config.search.date_range` when provided |

**CRITICAL**: The year filter must be applied at all 5 `SearchAggregator` instantiation sites in `nodes.py`. Missing even one allows post-publication papers to leak in during gap search, contextual enrichment, corpus expansion, or passage search — undermining the time-controlled comparison for Tier A topics.

### Behavior

```python
# Parsing
def _parse_date_range(date_range: str | None) -> tuple[int | None, int | None]:
    """Parse 'YYYY-YYYY', '-YYYY', 'YYYY-', or None into (year_from, year_to).
    Range is inclusive on both bounds."""

# Filtering (called after each source search)
def _filter_by_year(papers: list[CandidatePaper], year_from: int | None, year_to: int | None) -> list[CandidatePaper]:
    """Drop papers outside range. Always drop year=None with logged warning
    including paper title, source_database, and DOI."""
```

---

## 2. Bibliography Injection (Evaluation Tooling)

### Problem

To compare AI synthesis quality against human reviewers on equal footing, we need a "retrieval-controlled" condition: give the AI the same bibliography as the reference review and measure synthesis quality only.

### Design

**Location:** `paper/analysis/inject_bibliography.py` — standalone research script, not a pipeline CLI command.

**Workflow:**

1. **Input:** Reference review PDF path + topic string + domain string
2. **Extract bibliography** from PDF using existing `autoreview/evaluation/pdf_extractor.py` (returns raw reference lines as `list[str]`)
3. **Parse reference lines** into structured data (new: reference line parser):
   - Extract DOIs via regex (`10.\d{4,9}/[-._;()/:A-Z0-9]+`)
   - Extract approximate titles via heuristic parsing (text before journal name / year / volume)
   - For unparseable lines, use LLM fallback (single batch call to extract DOI + title from raw reference strings)
4. **Resolve each parsed reference** to a full `CandidatePaper` record:
   - DOI lookup first via Semantic Scholar or OpenAlex `get_paper_details`
   - Title search fallback if no DOI extracted
   - Log match confidence: `high` (DOI exact match), `medium` (title fuzzy match >=85%), `low` (title fuzzy match 70-85%)
5. **Build pre-populated `KnowledgeBase`:**
   - `candidate_papers` = all resolved papers
   - `screened_papers` = all papers with `include=True`, `relevance_score=5`
   - `topic`, `domain` set from inputs
   - Do NOT populate `full_text` — it is excluded from snapshot serialization anyway, and `full_text_retrieval` will fetch it
6. **Save** as snapshot JSON + cache resolution results (JSON keyed by reference line hash for re-runs)
7. **Output** a resolution report: total references extracted, resolved count (by confidence tier), failed count, list of unresolvable references

**Usage pattern:**

```bash
# Step 1: Inject bibliography
python paper/analysis/inject_bibliography.py \
    --pdf paper/references/car_t_resistance_2019.pdf \
    --topic "CAR-T therapy resistance mechanisms" \
    --domain biomedical \
    --output paper/snapshots/car_t_injected.json

# Step 2: Resume pipeline from full_text_retrieval (not extraction!)
autoreview resume paper/snapshots/car_t_injected.json \
    --start-from full_text_retrieval \
    --model claude-sonnet-4-6
```

**Why `start_from=full_text_retrieval`:** Starting from `extraction` would skip full-text retrieval, leaving the AI with only abstracts. Starting from `full_text_retrieval` ensures the AI gets full paper content — a fair test of synthesis, not a handicap from missing data.

**Resolution failure handling:** Unresolvable references are logged and excluded. The exclusion rate is itself a reportable metric (programmatic recoverability of reference bibliographies).

### Files to create

| File | Purpose |
|------|---------|
| `paper/analysis/inject_bibliography.py` | Bibliography extraction, reference line parsing, resolution, KB construction |
| `paper/analysis/reference_parser.py` | Reference line → structured data (DOI regex + title heuristic + LLM fallback) |
| `paper/analysis/README.md` | Usage documentation for all analysis scripts |

### Dependencies

- `autoreview.evaluation.pdf_extractor` — bibliography extraction from PDFs
- `autoreview.search.semantic_scholar` / `autoreview.search.openalex` — DOI/title resolution
- `autoreview.models.paper` — `CandidatePaper`, `ScreenedPaper`
- `autoreview.models.knowledge_base` — `KnowledgeBase`

---

## 3. Updated Paper Experimental Design

### Two-Tier Reference Set

| Tier | Citation threshold | Expected age | Purpose | Target N |
|------|-------------------|-------------|---------|----------|
| **A (landmark)** | 500+ citations | 2015-2019 | Time-controlled fair comparison, retrieval vs. synthesis decomposition | 5-8 |
| **B (contemporary)** | 50-200 citations | 2023-2024 | Active frontiers, lower contamination risk, real-world use case | 12-15 |

**Total:** 17-23 topics across 4 domains (biomedical, CS, materials science, social science), balanced across domains.

### Experimental Conditions

**Tier A topics (3 conditions each):**

| Condition | Search | Bibliography | Time-gated | Tests |
|-----------|--------|-------------|------------|-------|
| End-to-end | AI searches | AI-retrieved | `--date-range "-{ref_year}"` | Full pipeline vs. human |
| Retrieval-controlled | Skipped | Injected from reference PDF | N/A | Synthesis quality in isolation |
| Human reference | N/A | N/A | N/A | Gold standard |

**Tier B topics (2 conditions each):**

| Condition | Search | Bibliography | Time-gated | Tests |
|-----------|--------|-------------|------------|-------|
| End-to-end | AI searches | AI-retrieved | No | Pipeline on current science |
| Human reference | N/A | N/A | N/A | Gold standard |

**Model tiers:** All conditions run with Opus 4.6, Sonnet 4.6, Haiku 4.5.

**Run count estimate** (lower bound assumes 5 Tier A + 12 Tier B = 17 topics; upper bound assumes 8 + 15 = 23):
- Tier A: 5-8 topics x 2 AI conditions x 3 models = 30-48 runs
- Tier B: 12-15 topics x 1 AI condition x 3 models = 36-45 runs
- ARISE: 17-23 topics x 1 run = 17-23 runs
- Ablations: 5-10 topics x 4 conditions x 1 model (Sonnet) = 20-40 runs
- **Total: ~103-156 runs**

### Planned Analyses

**Analysis 1 — Main System Comparison**
- AutoReview vs. ARISE vs. human reference
- Evaluated on both AutoReview's 4-dimension rubric and ARISE's 7-category rubric
- Metrics: synthesis depth, topic coverage, writing quality, citation quality (recall, precision, F1)
- Statistical testing: paired comparisons with appropriate corrections

**Analysis 2 — Cross-Domain Variation**
- Per-domain breakdown (biomedical, CS, materials, social science)
- Which domains are more/less amenable to automation?
- Domain x system interaction effects

**Analysis 3 — Cross-Rubric Agreement**
- Do the two rubrics rank systems the same way?
- Correlation between AutoReview rubric scores and ARISE rubric scores
- Identifies rubric-specific biases

**Analysis 4 — Component Ablation**
- 4 conditions: no evidence chains, no critique loops, no passage mining, no comprehensiveness checks
- Run on 5-10 topics with Sonnet only
- Measures marginal contribution of each pipeline component
- Presented as heatmap: component x metric contribution matrix

**Analysis 5 — Retrieval vs. Synthesis Decomposition** (Tier A only, supplemental)
- End-to-end vs. retrieval-controlled comparison
- Isolates whether quality gaps come from retrieval failures or synthesis limitations
- Also reports retrieval overlap: what fraction of the reference bibliography did the AI independently find in the end-to-end condition?
- Three possible findings: retrieval bottleneck, synthesis bottleneck, or both adequate

**Analysis 6 — Citation Quality**
- Recall, precision, F1 against reference bibliography
- Hallucination rate (citations that don't resolve to real papers)
- Coverage depth (are high-impact papers in the field captured?)
- Retrieval overlap between end-to-end and retrieval-controlled conditions (Tier A)

**Analysis 7 — Model Tier Comparison**
- Opus vs. Sonnet vs. Haiku quality scores across all metrics
- Where does the quality-cost curve plateau?
- Is Haiku "good enough" for certain domains or metrics?
- Does Opus justify 10x cost over Sonnet?

**Analysis 8 — Cost-Quality Tradeoff**
- Tokens, wall-clock time, API cost per review
- Across systems (AutoReview vs. ARISE) and model tiers
- Cost per quality-point for practical deployment decisions

**Analysis 9 — Contamination Analysis**
- N-gram overlap (unigram through 5-gram) between generated and reference reviews
- Flags potential memorization from training data
- Particularly relevant for Tier A (high-citation, likely in training data)
- Reported as a limitation with quantitative bounds

### Figures and Tables

1. **System architecture diagram** — AutoReview 15-node DAG (simplified)
2. **Radar/spider chart** — dimension scores for AutoReview vs. ARISE vs. reference
3. **Grouped bar chart** — scores by domain (4 domains x 3 systems)
4. **Ablation heatmap** — component x metric contribution matrix
5. **Qualitative comparison** — side-by-side excerpt table
6. **Model tier line chart** — quality score vs. model tier per metric (quality-cost frontier)
7. **Cost scatter plot** — quality score vs. API cost per review (colored by model tier and system)
8. **Retrieval decomposition bar chart** — end-to-end vs. retrieval-controlled vs. human (Tier A, supplemental)
9. **Contamination histogram** — n-gram overlap distribution across topics and tiers
