# Paper Plan: Evaluating Autonomous LLM Review Generation

> **Design spec**: `docs/superpowers/specs/2026-03-13-year-filter-bib-injection-design.md`

## Context

The field of automated scientific review/survey generation has exploded (AutoSurvey, ARISE, SurveyX, SurveyG, PaperQA2), but **no rigorous multi-domain evaluation** has compared these systems against each other and against human-written reviews using standardized metrics. AutoReview has a complete evaluation framework in code that has never been run. We publish this as an **empirical evaluation paper** with a provocative framing: SOTA LLMs may be making the traditional review article obsolete.

**Central thesis**: State-of-the-art LLM pipelines can now generate scientific review papers approaching human quality across multiple domains — challenging the necessity of manually written reviews.

---

## Paper Design

### Title (working)
*"The End of the Review Article? A Multi-Domain Evaluation of Autonomous LLM-Generated Scientific Reviews"*

### Target venues
- **Primary**: EMNLP 2026 or ACL 2026 (main conference or system demonstration track)
- **Secondary**: NeurIPS 2026, or domain journal (National Science Review, JAMIA)

### Core experimental design

| Dimension | Value |
|---|---|
| Systems compared | **AutoReview** vs. **ARISE** (open-source, rubric-guided) vs. **human reference reviews** |
| Model tiers | **Opus 4.6** vs. **Sonnet 4.6** vs. **Haiku 4.5** (AutoReview run with each to measure quality-cost tradeoff) |
| Domains | 4: biomedical, computer science, materials science, social science |
| Reference tiers | **Tier A** (landmark, 500+ citations, 2015-2019) + **Tier B** (contemporary, 50-200 citations, 2023-2024) |
| Topics | 5-8 Tier A + 12-15 Tier B = 17-23 total, balanced across domains |
| Evaluation rubrics | AutoReview's 4-dimension rubric + ARISE's 7-category rubric (cross-evaluation) |
| Ablation conditions | Full pipeline, no evidence chains, no critique loops, no passage mining, no comprehensiveness checks (5-10 topics) |
| Judge model | Claude Sonnet 4.6 (same model for all evaluations, separated from generation) |

### Two-tier reference set rationale

**Tier A (landmark reviews):** Reviews with 500+ citations, published ~2015-2019. These enable time-controlled comparison — we gate AI search to only use literature available before the reference review's publication date (`--date-range "-{ref_year}"`). This ensures human and AI had access to the same evidence pool. Also enables the retrieval vs. synthesis decomposition (inject the reference's bibliography and test synthesis only). Trade-off: these reviews are likely in the LLM's training data, creating contamination risk.

**Tier B (contemporary reviews):** Reviews with 50-200 citations, published 2023-2024. Lower contamination risk, tests the pipeline on active scientific frontiers. No time-gating needed — this is the real-world use case. Only the end-to-end condition is run.

### Experimental conditions

**Tier A topics (3 conditions):**

| Condition | Search | Bibliography | Time-gated | Tests |
|-----------|--------|-------------|------------|-------|
| End-to-end | AI searches | AI-retrieved | `--date-range "-{ref_year}"` | Full pipeline vs. human |
| Retrieval-controlled | Skipped | Injected from reference PDF | N/A | Synthesis quality in isolation |
| Human reference | N/A | N/A | N/A | Gold standard |

**Tier B topics (2 conditions):**

| Condition | Search | Bibliography | Time-gated | Tests |
|-----------|--------|-------------|------------|-------|
| End-to-end | AI searches | AI-retrieved | No | Pipeline on current science |
| Human reference | N/A | N/A | N/A | Gold standard |

### Paper structure

#### 1. Introduction (~1.5 pages)
- The literature crisis: exponential publication growth, human reviewers overwhelmed
- The provocative claim: SOTA models may obsolete manual review writing
- What we test: can a fully autonomous pipeline produce reviews comparable to human experts?
- Contributions: (1) first multi-domain head-to-head evaluation of review generation systems, (2) cross-rubric evaluation methodology, (3) component ablation revealing what makes automated reviews good, (4) model tier analysis (Opus vs. Sonnet vs. Haiku) showing quality-cost frontier, (5) retrieval vs. synthesis decomposition, (6) cost-quality analysis

#### 2. Related Work (~1.5 pages)
- Automated survey generation: AutoSurvey/2, SurveyX, SurveyG, ARISE, Wang et al. (NSR 2025)
- RAG-based literature tools: PaperQA2, LitLLM, Elicit
- Systematic review automation: clinical screening/extraction tools
- Evaluation benchmarks: SurveyBench, SurveyEval, SurGE
- Position: first cross-system, cross-domain, cross-rubric empirical evaluation

#### 3. Systems Under Evaluation (~2.5 pages)
- **AutoReview**: 15-node DAG pipeline, multi-source search (PubMed + Semantic Scholar + OpenAlex + Perplexity + full-text retrieval), 3-level critique (outline + section + holistic), evidence chain construction, comprehensiveness checking with remediation, passage mining for undercited claims, narrative planning
- **ARISE**: Agentic rubric-guided iterative survey engine, multiple reviewer agents, 7-category rubric, iterative refinement targeting >=90%
- **Human reference reviews**: Published review papers, two tiers by citation count and recency

#### 4. Evaluation Methodology (~2.5 pages)
- **Two-tier topic selection**: criteria, rationale for Tier A (temporal control) vs. Tier B (frontier testing)
- **Time-controlled comparison**: AI search gated to pre-publication literature for Tier A
- **Retrieval-controlled condition**: bibliography injection from reference PDFs, pipeline resumes from full-text retrieval
- **Reference review identification**: Tier A (500+ citations), Tier B (50-200 citations, 2023-2024)
- **Automated metrics**:
  - Citation recall, precision, F1 (Jaccard + DOI + fuzzy matching)
  - LLM-scored synthesis depth (5 sub-dimensions, 0-5 scale)
  - LLM-scored topic coverage (sub-topic extraction and matching)
  - LLM-scored writing quality (4 sub-dimensions, 0-5 scale)
  - Structural metrics: word count, citations/1000 words, section balance, Flesch-Kincaid
- **Cross-rubric evaluation**: all outputs evaluated on both ARISE 7-category and AutoReview 4-dimension rubrics
- **Ablation design**: 4 conditions x 5-10 topics, Sonnet only
- **Model tier comparison**: all topics x 3 model tiers

#### 5. Results (~3.5 pages)
- **5a. Main comparison**: AutoReview vs. ARISE vs. reference, per metric and overall
- **5b. Cross-domain analysis**: quality variation across 4 domains, domain x system interactions
- **5c. Cross-rubric agreement**: correlation between rubrics, rubric-specific biases
- **5d. Ablation**: which components contribute most to quality? (heatmap)
- **5e. Retrieval vs. synthesis decomposition** (Tier A): end-to-end vs. retrieval-controlled, retrieval overlap metric
- **5f. Citation analysis**: recall, precision, F1, hallucination rate, coverage depth
- **5g. Model tier comparison**: Opus vs. Sonnet vs. Haiku quality-cost frontier
- **5h. Cost-quality tradeoff**: tokens, time, API cost per review
- **5i. Qualitative examples**: side-by-side excerpts showing synthesis quality differences
- **5j. Contamination analysis**: n-gram overlap between generated and reference reviews

#### 6. Discussion (~1.5 pages)
- Answering the provocative question: where do we stand?
- Retrieval vs. synthesis: where is the bottleneck?
- Which domains are more/less amenable to automation?
- What still falls short compared to human reviewers?
- Contamination and memorization concerns
- Ethical implications: disclosure, integrity, the future of peer review
- Limitations: LLM-as-judge bias, no human evaluation, reproducibility across model versions

#### 7. Conclusion (~0.5 pages)

**Page budget note:** EMNLP/ACL main conference allows 8 pages + unlimited references. The above structure totals ~13.5 pages of content. Analyses 5 (retrieval decomposition), 8 (cost-quality), and 9 (contamination) are candidates for supplementary material. Main paper (~8 pages): Intro, Related Work, Systems, Methodology, Results (1-4, 6-7), Discussion, Conclusion. Supplementary: retrieval decomposition, cost analysis, contamination analysis, full per-topic results tables, qualitative examples.

### Figures and tables
1. **System architecture diagram**: AutoReview 15-node DAG (simplified)
2. **Radar/spider chart**: dimension scores for AutoReview vs. ARISE vs. reference
3. **Grouped bar chart**: scores by domain (4 domains x 3 systems)
4. **Ablation heatmap**: component x metric contribution matrix
5. **Qualitative comparison**: side-by-side excerpt table
6. **Model tier line chart**: quality score vs. model tier per metric (quality-cost frontier)
7. **Cost scatter plot**: quality score vs. API cost per review (colored by model tier and system)
8. **Retrieval decomposition bar chart**: end-to-end vs. retrieval-controlled vs. human (Tier A)
9. **Contamination histogram**: n-gram overlap distribution across topics and tiers

---

## Analyses

### Analysis 1 — Main System Comparison
- AutoReview vs. ARISE vs. human reference
- Evaluated on both AutoReview 4-dimension rubric and ARISE 7-category rubric
- Metrics: synthesis depth, topic coverage, writing quality, citation quality (recall, precision, F1)
- Statistical testing: paired comparisons with appropriate corrections
- **Scripts**: `paper/analysis/main_comparison.py`

### Analysis 2 — Cross-Domain Variation
- Per-domain breakdown (biomedical, CS, materials, social science)
- Which domains are more/less amenable to automation?
- Domain x system interaction effects
- **Scripts**: `paper/analysis/domain_analysis.py`

### Analysis 3 — Cross-Rubric Agreement
- Do the two rubrics rank systems the same way?
- Correlation between AutoReview rubric scores and ARISE rubric scores
- Identifies rubric-specific biases
- **Scripts**: `paper/analysis/rubric_agreement.py`

### Analysis 4 — Component Ablation
- 4 conditions: no evidence chains, no critique loops, no passage mining, no comprehensiveness checks
- Run on 5-10 topics with Sonnet only
- Measures marginal contribution of each pipeline component
- Presented as heatmap: component x metric contribution matrix
- **Scripts**: `paper/analysis/ablation_analysis.py`

### Analysis 5 — Retrieval vs. Synthesis Decomposition (Tier A, supplemental)
- End-to-end vs. retrieval-controlled comparison
- Isolates whether quality gaps come from retrieval failures or synthesis limitations
- Reports retrieval overlap: what fraction of reference bibliography did AI independently find?
- Three possible findings: retrieval bottleneck, synthesis bottleneck, or both adequate
- **Scripts**: `paper/analysis/retrieval_decomposition.py`

### Analysis 6 — Citation Quality
- Recall, precision, F1 against reference bibliography
- Hallucination rate (citations that don't resolve to real papers)
- Coverage depth (are high-impact papers captured?)
- Retrieval overlap between end-to-end and retrieval-controlled conditions (Tier A)
- **Scripts**: `paper/analysis/citation_analysis.py`

### Analysis 7 — Model Tier Comparison
- Opus vs. Sonnet vs. Haiku quality scores across all metrics
- Where does the quality-cost curve plateau?
- Is Haiku "good enough" for certain domains or metrics?
- Does Opus justify 10x cost over Sonnet?
- **Scripts**: `paper/analysis/model_comparison.py`

### Analysis 8 — Cost-Quality Tradeoff
- Tokens, wall-clock time, API cost per review
- Across systems (AutoReview vs. ARISE) and model tiers
- Cost per quality-point for practical deployment decisions
- **Scripts**: `paper/analysis/cost_analysis.py`

### Analysis 9 — Contamination Analysis
- N-gram overlap (unigram through 5-gram) between generated and reference reviews
- Flags potential memorization from training data
- Particularly relevant for Tier A (high-citation, likely in training data)
- Reported as limitation with quantitative bounds
- **Scripts**: `paper/analysis/contamination_analysis.py`

---

## Implementation Plan

### Phase 1: Evaluation infrastructure improvements (COMPLETE)

**1a. Strengthen citation matching** (`autoreview/evaluation/citation_matcher.py`)
- DOI-based matching (exact match before Jaccard)
- Fuzzy string matching via `rapidfuzz` for title comparison
- Citation precision, recall, and F1

**1b. ARISE rubric evaluation** (`autoreview/evaluation/arise_rubric.py`)
- 7-category rubric as single LLM call with structured output for all 20 subcriteria
- 0-100 scale for ARISE comparability

**1c. Structural/quantitative metrics** (`autoreview/evaluation/structural_metrics.py`)
- Word count, citations per 1000 words, section balance, Flesch-Kincaid readability

**1d. Judge model separation**
- `--judge-model` CLI flag on `evaluate` command

**1e. Batch runner and cost analyzer**
- `autoreview/evaluation/batch_runner.py` — asyncio.Semaphore concurrency control
- `autoreview/evaluation/cost_analyzer.py` — static pricing table

### Phase 1.5: Year filtering and bibliography injection (NEW)

**1.5a. Year filtering in SearchAggregator** (core pipeline)
- Parse `SearchConfig.date_range` into `(year_from, year_to)` with open-ended support
- Post-filter in `SearchAggregator` after each source returns
- Always drop `year=None` papers with structured logging warning
- **CRITICAL**: Must apply to all 5 `SearchAggregator` instantiation sites in `nodes.py` (primary search, gap search, contextual enrichment, corpus expansion, passage search)
- Add `--date-range` CLI flag to both `run` and `resume` commands
- Files: `autoreview/search/aggregator.py`, `autoreview/pipeline/nodes.py`, `autoreview/config/models.py`, `autoreview/cli.py`

**1.5b. Bibliography injection script** (evaluation tooling)
- Extract bibliography from reference PDF via `pdf_extractor.py` (returns raw lines)
- Parse reference lines: DOI regex extraction + title heuristic + LLM fallback for unparseable lines
- Resolve parsed references to full `CandidatePaper` records via DOI/title lookup
- Build pre-populated `KnowledgeBase` with `screened_papers` (all `include=True`); do NOT populate `full_text` (excluded from snapshots, fetched by `full_text_retrieval`)
- Cache resolution results (JSON keyed by reference line hash) for re-runs
- Log resolution report: resolved count (by confidence tier: high/medium/low), failed count, unresolvable references
- Files: `paper/analysis/inject_bibliography.py`, `paper/analysis/reference_parser.py`

### Phase 2: Topic selection and reference collection

**2a. Select 17-23 evaluation topics** (balanced across 4 domains)

Tier A (landmark, 500+ citations, ~2015-2019):
- 5-8 topics, 1-2 per domain
- Must have clearly identifiable publication date for time-gating

Tier B (contemporary, 50-200 citations, 2023-2024):
- 12-15 topics, 3-4 per domain
- Active frontiers with recent review coverage

| Domain | Example Tier A topics | Example Tier B topics |
|---|---|---|
| Biomedical | Cellular senescence genetics, CAR-T therapy resistance | Long COVID mechanisms, GLP-1 receptor agonists beyond diabetes |
| Computer Science | Federated learning privacy, vision transformers | LLM hallucination mitigation, RAG architectures |
| Materials Science | Perovskite solar cell stability, metal-organic frameworks | High-entropy alloys, biodegradable polymers |
| Social Science | Social media and adolescent mental health | AI bias in hiring, climate migration patterns |

**2b. Identify reference reviews**
- Tier A: published reviews with 500+ citations, collect PDFs
- Tier B: published reviews with 50-200 citations from 2023-2024, collect PDFs
- Document: title, journal, year, citation count, DOI

### Phase 3: Run experiments

**3a. AutoReview runs — Tier B end-to-end** (15 topics x 3 models = 45 runs)
- Full pipeline, no time-gating
- Save all snapshots and audit logs

**3b. AutoReview runs — Tier A end-to-end** (8 topics x 3 models = 24 runs)
- Full pipeline with `--date-range "-{ref_year}"` for time-gating
- Save all snapshots and audit logs

**3c. AutoReview runs — Tier A retrieval-controlled** (8 topics x 3 models = 24 runs)
- Use `inject_bibliography.py` to build KB from reference PDF
- Resume from `full_text_retrieval`
- Save all snapshots and audit logs

**3d. ARISE runs** (23 topics x 1 run = 23 runs)
- Clone ARISE repo, configure with Claude Sonnet 4.6 if possible
- Run on all topics (no time-gating — ARISE has no equivalent mechanism)
- Save outputs in comparable format

**3e. Ablation runs** (5-10 topics x 4 conditions x 1 model = 20-40 runs)
- Conditions: no evidence chains, no critique loops, no passage mining, no comprehensiveness checks
- Sonnet only, at least 1-2 per domain

**Total estimated runs: ~103-156** (lower bound: 5 Tier A + 12 Tier B; upper bound: 8 Tier A + 15 Tier B)

### Phase 4: Evaluation and analysis

**4a. Run evaluation pipeline**
- Evaluate all outputs against reference reviews using both rubrics
- All evaluations use Claude Sonnet 4.6 as judge model
- Save results as structured JSON

**4b. Analysis scripts** (`paper/analysis/`)
- `main_comparison.py` — Analysis 1
- `domain_analysis.py` — Analysis 2
- `rubric_agreement.py` — Analysis 3
- `ablation_analysis.py` — Analysis 4
- `retrieval_decomposition.py` — Analysis 5
- `citation_analysis.py` — Analysis 6
- `model_comparison.py` — Analysis 7
- `cost_analysis.py` — Analysis 8
- `contamination_analysis.py` — Analysis 9
- `figures.py` — generate all publication figures

### Phase 5: Paper writing
- LaTeX format for venue submission
- All figures generated programmatically from analysis results

---

## Key files to modify/create

| File | Action | Phase |
|---|---|---|
| `autoreview/search/aggregator.py` | Add year post-filter, `year=None` drop with logging | 1.5 |
| `autoreview/pipeline/nodes.py` | Pass `date_range` to aggregator | 1.5 |
| `autoreview/config/models.py` | Validate `date_range` format | 1.5 |
| `paper/analysis/inject_bibliography.py` | **New**: bibliography extraction + resolution + KB builder | 1.5 |
| `paper/topics.yaml` | **New**: all evaluation topics with tier, domain, reference metadata | 2 |
| `paper/references/` | **New**: reference review PDFs | 2 |
| `paper/analysis/main_comparison.py` | **New**: Analysis 1 | 4 |
| `paper/analysis/domain_analysis.py` | **New**: Analysis 2 | 4 |
| `paper/analysis/rubric_agreement.py` | **New**: Analysis 3 | 4 |
| `paper/analysis/ablation_analysis.py` | **New**: Analysis 4 | 4 |
| `paper/analysis/retrieval_decomposition.py` | **New**: Analysis 5 | 4 |
| `paper/analysis/citation_analysis.py` | **New**: Analysis 6 | 4 |
| `paper/analysis/model_comparison.py` | **New**: Analysis 7 | 4 |
| `paper/analysis/cost_analysis.py` | **New**: Analysis 8 | 4 |
| `paper/analysis/contamination_analysis.py` | **New**: Analysis 9 | 4 |
| `paper/analysis/figures.py` | **New**: all publication figures | 4 |
| `paper/results/` | **New**: evaluation results JSON | 4 |

---

## Verification

1. **Unit tests**: year filter parsing and edge cases, bibliography injection resolution logic
2. **Smoke test**: run full evaluate pipeline on 1 topic (existing benchmark output) with year filtering
3. **Bibliography injection test**: extract + resolve from a known reference PDF, verify paper count and metadata quality
4. **Cross-rubric sanity check**: verify ARISE rubric produces scores in expected range on AutoReview output
5. **Cost estimation**: compute estimated API cost for full run (~136-156 runs) before committing

---

## Estimated scope
- **Phase 1.5 (year filter + bib injection)**: ~1 session
- **Phase 2 (topic selection + references)**: ~1 session
- **Phase 3 (experimental runs)**: ~$100-200 API cost, can be batched
- **Phase 4 (analysis + figures)**: ~1-2 sessions
- **Phase 5 (paper writing)**: ~3-5 sessions
- **Total remaining**: ~7-10 sessions
