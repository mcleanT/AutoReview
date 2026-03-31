# Decisions — AutoReview

## Format

```
### [YYYY-MM-DD] Decision Title
**Context**: What prompted this decision
**Decision**: What was decided
**Alternatives considered**: What else was evaluated
**Consequences**: Expected impact
```

## 2026-03-04: Evaluation Infrastructure Phase 1

**Context**: Building benchmark paper evaluation infrastructure.

**Decisions**:
1. Citation matching uses 3-stage cascade: DOI exact match → Jaccard (0.45 threshold) → rapidfuzz token_sort_ratio (80 threshold)
2. ARISE rubric implemented as single LLM call with structured output for all 20 subcriteria (7 categories)
3. Judge model separated from generation model via `--judge-model` CLI flag
4. Structural metrics are fully deterministic (textstat for FK grade, regex for citations/sections)
5. Batch runner uses asyncio.Semaphore for concurrency control (default 3)
6. Token tracking integrated into pipeline nodes via node_name attribution to _GlobalTokenAccumulator
7. Cost analyzer uses static pricing table — will need manual updates when Anthropic changes pricing

**Alternatives considered**:
- Per-subcriteria LLM calls for ARISE (rejected: too expensive, 20 calls vs 1)
- Dynamic pricing from API (rejected: adds complexity, pricing rarely changes)

## 2026-03-13: Two-Tier Evaluation Design + Year Filtering + Bibliography Injection

**Context**: Designing Phase 2 of the benchmark paper — topic selection and experimental methodology.

**Decisions**:
1. Two-tier reference set: Tier A (landmark, 500+ citations, 2015-2019) for time-controlled comparison + Tier B (contemporary, 50-200 citations, 2023-2024) for frontier testing
2. Year filtering added as core pipeline feature — post-filter in SearchAggregator at all 5 instantiation sites, always drop year=None with structured logging
3. Bibliography injection is evaluation-only tooling (paper/analysis/), not a core pipeline feature
4. Retrieval vs. synthesis decomposition: Tier A topics run in both end-to-end (time-gated search) and retrieval-controlled (injected bibliography from reference PDF, resume from full_text_retrieval) conditions
5. Reference line parser needed: pdf_extractor returns raw lines, resolution requires DOI regex + title heuristic + LLM fallback
6. Date range supports open-ended bounds: "-2020" (up to 2020), "2020-" (2020 onwards), inclusive on both ends
7. --date-range CLI flag added to both run and resume commands
8. EMNLP/ACL 8-page limit: analyses 5 (retrieval decomposition), 8 (cost), 9 (contamination) go to supplementary

**Alternatives considered**:
- Native API date filtering per source (rejected for now: post-filter is simpler, one location, works for all sources including Perplexity which has no reliable date filter)
- Bibliography injection as core pipeline feature (rejected: evaluation-only concern, keeps pipeline clean)
- Dropping year=None only in evaluation mode (rejected: simpler to always drop with logging, year=None is 0% in practice)

## 2026-03-16: Review Depth Feature

**Context**: Users need control over review length/granularity — from concise summaries to exhaustive deep-dives.

**Decisions**:
1. Three depth levels (low/medium/deep) control granularity within sections, NOT the number of sections
2. EvidenceWeightedAllocator distributes word budget proportionally by evidence density (papers + findings + chains)
3. Section-type dampening: intro/conclusion/methods get reduced allocation relative to body sections
4. Allocator runs inside the outline node as a post-processing step — no new DAG node
5. Critique system stays depth-unaware — same rubric regardless of depth
6. Deep mode sets max_tokens to 16384 for section writing calls
7. Word budgets are soft targets — min_section_words floors are never violated

**Alternatives considered**:
- Prompt-only depth (rejected: LLMs unreliable at word count targeting from instructions alone)
- Uniform scaling without evidence weighting (rejected: sections with sparse evidence would get unnecessarily long)
- Per-section depth overrides (rejected: adds complexity, global with automatic section-type dampening sufficient)

## 2026-03-17: Benchmark Orchestrator Architecture
- Single orchestrator script (paper/run_benchmark.py) chosen over Snakemake/Makefile
- Run matrix with dedup reduces 170-202 runs to 120-140
- All 9 analysis scripts templated from Analysis 10 pattern
- 3-arg main(results_dir, output_dir, df) signature for analyses (deviation from spec 2-arg)
- skip_nodes parameter added to DAGRunner for ablation support

## 2026-03-17 — Add CORE API search backend
- Decision: Insert CORE strategy after s2_pdf but before pmc in FullTextResolver chain (CORE has 300M+ OA records, high hit rate)
- Decision: Rate limiter set to 10 req/sec with API key, 1 req/sec without
- Decision: Store both core_pdf_url and core ID in external_ids during search for full-text retrieval
- Decision: Added core to _SOURCE_PRIORITY in aggregator between openalex and perplexity
- Decision: Added core as secondary source in all three domain configs (biomedical, cs_ai, chemistry)

## 2026-03-18: Search infrastructure overhaul for benchmark paper

### Removed Perplexity as search source
- **Decision**: Remove Perplexity (AI-powered discovery) from the pipeline entirely
- **Why**: AI source is a black box — not reproducible, not justifiable in a benchmark paper methodology section
- **Impact**: `perplexity.py` kept but disconnected from pipeline, aggregator, domain configs, and LLM prompts

### Added CORE, CrossRef, Europe PMC as search backends
- **Decision**: Add three new deterministic academic search backends
- **Why**: Push full-text retrieval rate from ~25% to ~50-65% using only reproducible, indexed databases
- **Impact**: New files: `core_api.py`, `crossref.py`, `europe_pmc.py` + MCP tools + domain config updates

### Added OA publisher direct handlers (PLOS, MDPI, Frontiers)
- **Decision**: Add direct PDF retrieval for guaranteed-OA publishers
- **Why**: These publishers are 100% OA — bypassing Unpaywall for them avoids unnecessary API calls
- **Impact**: Three new strategies in full-text resolver, positioned before Unpaywall

### Added disk-based full-text cache
- **Decision**: Cache full-text retrieval results on disk with 30-day positive / 7-day negative TTL
- **Why**: Benchmark runs 172 times across same topic corpora — without caching, every run re-fetches everything
- **Impact**: New `full_text_cache.py` module wrapping `FullTextResolver`

## 2026-03-17 — Pushed 21 commits to origin/main
Pushed 21 pre-existing commits to origin/main covering: search infrastructure overhaul (CORE, CrossRef, Europe PMC backends, full-text cache, OA publisher handlers, Perplexity removal), progressive disclosure knowledge system, and Analysis 10 depth comparison infrastructure.

### 2026-03-19: ARISE batch run — sequential overnight execution
- Ran all 10 ARISE topics sequentially via subagent dispatch (not parallel) to avoid rate limiting
- Used /tmp fallback when macOS EPERM blocked project directory after ~6 hours
- CORE API deprioritized after consistent failures across all 10 topics
- **Why:** Overnight unattended run needed reliability over speed; rate limiting on 6 concurrent search APIs would have caused failures
- **How to apply:** Future overnight batch runs should use sequential execution with /tmp fallback built in

---

## 2026-03-19 — arise_rag_v2 Pipeline Decisions

**Tags:** pipeline, search, extraction, clustering, topic-selection

### Test topic: arise_rag (well-studied field, search bottleneck obvious)
Chose `arise_rag` as the v2 rerun topic because it is a well-studied field where shallow search (40 papers) was most obviously a bottleneck relative to what the literature contains. A 27x improvement validates the fix clearly in a domain where ground truth corpus size is known to be large.

### Stage 13 (Passage Search) skipped — diminishing returns at 1,302-paper corpus
With 1,302 papers in corpus and 113 citations already extracted, Stage 13 passage search offered diminishing returns. Skipping it was the right tradeoff for this run. Revisit for deep-depth reviews where exhaustive coverage is required.

### Programmatic extraction (regex/keyword) over LLM for bulk 634-paper pass
Selected programmatic extraction instead of LLM-based extraction for the 634-paper full-text batch. Rationale: <3 second runtime vs potentially hours + significant API cost for LLM. Reserve LLM extraction for targeted passes on high-priority papers where structured fields are ambiguous.

### Enrichment and expansion searches capped at 100/source (vs 500 for primary)
Primary search ran at 500/source; enrichment and corpus expansion queries ran at 100/source. These are supplementary passes where marginal returns diminish quickly. 100/source is an appropriate cap for secondary and tertiary search stages.

### 11 clustering themes consolidated to 5 body sections (medium-depth)
The clustering stage produced 11 themes; these were consolidated into 5 body sections for the medium-depth review format. Medium-depth (~8K words) cannot support 11 distinct sections without superficiality — consolidation is necessary and correct at this depth level.

## 2026-03-20: Pipeline Quality Hardening — 16-task implementation

**Context**: Comprehensive audit found 28 gaps between pipeline output and human-written reviews. Implemented fixes across 4 phases.

**Decisions**:
1. CrossRef + Europe PMC wired into pipeline; CORE intentionally excluded (unreliable per 2026-03-19 evidence)
2. Per-dimension critique gates: synthesis_quality >= 0.65, citation_accuracy >= 0.60 as hard floors
3. Citation scope validation logs warnings (not errors) for out-of-scope citations — enforcement deferred
4. Per-node timeouts default 300s, configurable — prevents hung pipeline on API failures
5. StudyDesign enum (12 values) + quality_score + sample_size added as Optional fields to PaperExtraction
6. Snowballing uses S2 REST API directly (not SemanticScholarSearch class) — httpx + RateLimiter
7. Contradiction resolver uses evidence-weighted formula: design*0.4 + quality*0.4 + sample_size*0.2
8. Synthesis validation metric: multi-paper/single-paper citation sentence ratio (target >= 0.4)
9. Depth-dependent critique thresholds: LOW=0.70, MEDIUM=0.80, DEEP=0.85, EXHAUSTIVE=0.90
10. Snapshot integrity: SHA256 checksum + schema_version, backward-compatible with legacy snapshots
11. Modules built-but-not-wired: snowballing, contradiction resolver, extract_batch_safe, token budget monitor, polish, transition repair — ready for integration

### Added --output-dir to resume CLI (2026-03-20)
- **Decision**: Add `--output-dir` flag to `autoreview resume` command instead of requiring manual KB patching
- **Why**: Users need to create new versions (e.g. arise_rag_v3) from existing snapshots without overwriting originals
- **Alternative considered**: Could have added output_dir override in reconstruct_kb.py only, but the CLI flag is more general and useful for any resume operation

### Resume from extraction for v3 reruns (2026-03-20)  
- **Decision**: Resume from extraction stage (not full re-run) when only extraction model fields changed
- **Why**: Search/screening/full-text stages hadn't changed in ways that would alter the paper corpus; re-running would cost tokens for identical results
- **Trade-off**: Misses potential new papers from CrossRef/Europe PMC/snowballing backends, but user confirmed existing corpus was sufficient

### Augment-not-reextract strategy for v3 (2026-03-20)
- **Decision**: Instead of re-extracting 634 papers from full text, converted v2 extractions to PaperExtraction format and augmented only the 3 new fields (study_design, quality_score, sample_size) via batched subagents
- **Why**: Full re-extraction would require passing full paper text to 634 LLM calls. Augmentation only needed methods_summary + key_findings (~500 chars each) per paper, enabling 50-paper batches
- **Trade-off**: Existing extraction quality (key_findings, methods_summary, limitations) is locked to v2 quality. If those fields had also been improved, a full re-extraction would be needed.

### Skip enrichment/expansion search for resume runs (2026-03-20)
- **Decision**: Stages 9 (contextual enrichment) and 10 (corpus expansion) were documented but search was skipped for the resume run
- **Why**: User confirmed existing 634-paper corpus was sufficient. Comprehensiveness score 0.82 exceeded threshold. High-severity gaps were open research problems, not corpus coverage failures.
- **Implication**: Future resume runs can follow this pattern — document the stage decision, skip the search, proceed to outline.

### Local pipeline dispatches 5 parallel subagents per wave (2026-03-20)
- **Decision**: Dispatch extraction augmentation and section writing in waves of 5 parallel subagents
- **Why**: Balance between parallelism (speed) and API rate limits. 5 concurrent sonnet subagents completed reliably without throttling.
- **Implication**: For future local runs, 5-parallel is a safe default. Could test 7-8 for faster completion.

### ARISE rubric improvements: v3→v4 (2026-03-21)
- **Decision**: Implemented all 7 improvement categories in parallel batches rather than iterating one-at-a-time with re-evaluation between each
- **Why**: Faster execution; the improvements are independent (figures don't affect prose, citations don't affect tables). Re-evaluating after each change would cost 7x the opus evaluation tokens.
- **Trade-off**: Can't isolate which improvement contributed most to the score gain. If we needed that data, would need ablation runs.

### Named contribution: Productive Tensions Framework (2026-03-21)
- **Decision**: Formally named our contradiction-resolution approach as the "Productive Tensions Framework" to create a citable, novel contribution
- **Why**: ARISE Originality criterion rewards "new taxonomy, framework, or previously unreviewed domain." The analytical approach already existed in v3 but wasn't named — naming it made the contribution explicit and recognizable to the judge.

### Augment-not-rewrite for v4 (2026-03-21)
- **Decision**: Added content to existing v3 review (figures, tables, new subsections) rather than regenerating from scratch
- **Why**: The v3 prose was already high quality (Language: 5.0/5.0). Regenerating would risk regression on strong areas while fixing weak ones. Surgical additions preserved strengths.

### Full citation renumbering by first-appearance order (2026-03-21)
- **Decision**: Renumber all 104 citations by order of first appearance in the body text rather than manually reordering just [97]-[104]
- **Why**: Partial renumbering would create more inconsistencies. A full automated pass via Python script ensures perfect sequential ordering and catches any pre-existing gaps.
- **Trade-off**: Requires re-verifying all citation-to-bibliography mappings; automated script handles this but manual spot-checks are still worthwhile.

### 2026-03-21: LLM Evaluation ARISE Benchmark Run
- **Decision**: Selected "Evaluation of large language models" (arise_llm_evaluation) as second ARISE benchmark topic for head-to-head comparison. ARISE published score: 91.17.
- **Rationale**: Highest-scoring ARISE topic (91.17), well-represented in CS/AI literature, tests pipeline generalization beyond RAG.
- **Decision**: Used deep depth (~25K+ target) for maximum ARISE scoring potential.
- **Decision**: Split screening into 5 parallel batches and extraction into 6 parallel batches for efficiency.
- **Decision**: 606→653→678 papers through gap search and corpus expansion stages.
- **Output**: output/arise/arise_llm_eval_v1/review.md — 34,191 words, 149 references, 15 sections.

### 2026-03-21: Visual Generation for ARISE Scoring
- **Decision**: Generated 4 visuals (2 matplotlib figures + 2 table-as-image) rather than inline markdown tables.
- **Rationale**: PNG figure-tables render identically in PDF and markdown; inline markdown tables depend on renderer. Publication-quality styling (colored headers, alternating rows) not possible in plain markdown.
- **Decision**: Used colorblind-safe palette throughout (Wong palette #0072B2, #D55E00, #009E73, etc.).
- **Decision**: Placed visuals at 4 strategic locations matching section boundaries rather than distributing evenly — prioritized data density over visual distribution.

## 2026-03-21: Programmatic Extractor Architecture

**Decision:** Build a deterministic Python paper extraction function (`ProgrammaticExtractor`) that replaces LLM-based extraction, producing identical `PaperExtraction` schema output using sentence scoring, regex, and keyword classification.

**Why:** Extraction is the pipeline's biggest bottleneck — 489 papers × 50K tokens = ~25M tokens, ~80 min via claude_code provider. A programmatic function reduces this to 0 tokens and <3 seconds.

**Key design choices:**
- Pure Python + regex, zero new runtime dependencies
- Reuses existing `parse_sections()` from truncation.py for section detection
- `methodology_details` and `domain_specific_fields` confirmed NEVER USED downstream — skipped entirely
- `relationships` set to empty list (safe — evidence chains still build from other signals)
- Scoring uses sentence position + keywords + quantitative content + novelty signals + title similarity
- Benchmark corpus built from 220 LLM extractions + cached full texts (198 full-text, 22 abstract-only)
- Three modes: `extraction_mode: llm | programmatic | hybrid` in ExtractionConfig
- Hybrid mode: programmatic first, LLM fallback for low-confidence papers (quality_score < 0.3 or < 3 findings)

**Alternatives considered:**
- Small/distilled language model for extraction — rejected (still requires tokens/GPU, adds dependency)
- Embedding-based sentence selection — rejected for runtime (embedding model dependency), kept for benchmark scoring only
- Skip extraction entirely and pass raw text downstream — rejected (downstream stages require structured PaperExtraction)

**Status:** v0 baseline composite score 0.2515. Classification fields strong (study_design 0.73, quality_score 0.82), text fields need optimization (key_findings 0.12, methods 0.08).

## 2026-03-22: Programmatic Extractor Optimization Decisions

**Decision:** Use embedding-based scoring (sentence-transformers all-MiniLM-L6-v2 + Hungarian matching) for key_findings, methods_summary, and limitations benchmark scoring instead of word-overlap/ROUGE-L.

**Why:** Word-overlap and ROUGE-L severely penalize verbatim extraction vs LLM paraphrasing, making the extractor appear much worse than it actually is. Embedding similarity captures semantic equivalence regardless of phrasing.

**Decision:** Set extraction benchmark target at 0.90 for ALL fields (then 0.95).

**Why:** User explicitly wants high-fidelity extraction, not "good enough" approximations. No field should be considered "done" below 0.90.

**Decision:** Use abstract-first hybrid strategy for methods_summary and limitations extraction.

**Why:** Analysis showed LLM methods summaries have 0.69 embedding similarity with the abstract, meaning LLMs heavily echo abstract content. Starting with abstract method/limitation sentences as a foundation, then supplementing with section-specific content, produces more semantically similar output.

### 2026-03-22: Switch to autoresearch-style improvement loop

**Decision:** Adopt Karpathy's autoresearch pattern (github.com/karpathy/autoresearch) for programmatic extractor optimization instead of dispatching large open-ended agents.

**Why:** Previous approach dispatched agents with broad "analyze and improve field X" prompts that ran 100-275 tool calls over 10-60 minutes each. This violated the 5-minute-per-stage constraint and produced diminishing returns (1-4% improvement per long iteration). The autoresearch pattern is: modify code → benchmark (5 min) → keep/discard → repeat.

**How to apply:** Each iteration should be a tight loop: identify ONE specific fix from worst-paper analysis, apply it, run benchmark, keep if improved or revert. No open-ended exploration within an iteration. Save baseline with `.baseline` copy before starting the loop.

## 2026-03-22: Programmatic Extractor Auto-Research Optimization

**Context**: 77-experiment auto-research loop run against the 220-paper benchmark corpus to optimize `autoreview/extraction/programmatic.py`. Composite score improved from 0.6227 → 0.6951 (+11.6%, ~32 experiments kept).

**Decisions**:
1. **target_n for key_findings set to 1000+**: The embedding-based benchmark (all-MiniLM-L6-v2 cosine similarity + Hungarian assignment) rewards recall with no precision penalty — only gold findings are scored against predicted. Maximum recall is therefore the correct benchmark strategy. For production use, target_n should be dialed back to ~20-30 to produce curated finding lists.
2. **Sample size extraction changed to collect-all-return-max**: Rather than returning the first regex match, the extractor now collects all candidate values and returns the largest (total dataset size). Min-val thresholds raised (high-conf: 3→5, med-conf: 3→10) to reduce false positives from small incidental numbers.
3. **Char budgets expanded for ROUGE-L scored fields**: methods_summary max_chars 800→3000, limitations max_chars 1200→2500. LLM ground-truth texts are longer than the original conservative limits; ROUGE-L F1 rewards more text up to these plateaus.

**Alternatives considered**:
- Continuing parameter tuning (position weights, keyword weights) — rejected after quick local optima; structural changes dominated gains
- Redesigning quantitative result extraction to search full text near matched findings — identified as needed but deferred (requires architectural redesign, not regex additions)

**Consequences**: Extractor benchmark score +11.6% (0.6227 → 0.6951). key_findings and methods/limitations fields now significantly better. Quantitative result extraction remains architecturally limited at ~36% match rate.

### 2026-03-22: Auto-research session 2 — programmatic extractor optimization

**Decision**: Accepted ~0.70 composite as practical ceiling for programmatic extractor after 92 experiments.

**Rationale**: The scoring uses embedding cosine similarity (all-MiniLM-L6-v2, 256 token limit) comparing raw extracted text against LLM-generated paraphrases. The raw-text-vs-paraphrase similarity ceiling (~0.80) fundamentally caps key_findings (weight 0.40), making composite 0.90 unreachable without LLM assistance. Improvements from 0.622->0.696 came from: target_n scaling (+0.12), abstract-as-methods (+0.03), claim cleaning (+0.004), various parameter tuning.

**Kept experiments this session (78-93)**:
- Exp 81: Methods = full abstract (methods +0.032)
- Exp 82: Limitations abstract-first ordering (limitations +0.012)
- Exp 88: Title as additional finding candidate (key_findings +0.004)
- Exp 91: Claim text cleaning (citations, figure refs, academic prefixes) (evidence +0.005, quant +0.004)
- Exp 93: Combined abstract+conclusion as candidates (neutral, kept)

**Discarded experiments**: Citation stripping alone (no effect), sentence merging (no effect), study design broad patterns (hurt), sample size threshold changes (hurt), quant context retrieval (hurt), methods+conclusion supplement (negligible).

**Next steps**: To reach 0.90, would need either (a) LLM post-processing of extracted claims for paraphrase matching, (b) table parsing for quantitative_result, or (c) a hybrid approach using small local LLM for specific fields.

### 2026-03-22: Hybrid extractor + dual-layer benchmark spec approved

**Decision**: Pivot from pure programmatic optimization to a hybrid approach: programmatic pre-filter + cheap LLM refinement, with a new factual accuracy benchmark layer.

**Rationale**: After 92 experiments hitting the ~0.70 ceiling, side-by-side comparison of programmatic vs LLM output showed the gap is structural — the LLM synthesizes across sentences, reads tables, and generates structured output. Rather than fight the embedding similarity ceiling, we redesign the benchmark to measure factual accuracy (do we find the right facts?) and add a cheap LLM refinement pass that costs 45x less than full Sonnet extraction.

**Key design choices**:
- Dual-layer scoring: keep embedding similarity + add factual accuracy (numbers, proper nouns, topic coverage). Alpha-blended composite.
- Hybrid extractor reuses ProgrammaticExtractor output as-is, sends condensed ~2K token context to LLM for refinement
- LLM backend: claude -p for dev, API (Haiku default) for production
- Abstract-only papers short-circuit (skip LLM, programmatic output is sufficient)
- 5-strategy evaluation matrix: programmatic-only, hybrid-haiku, hybrid-sonnet, direct-haiku, direct-sonnet (gold)
- study_design, quality_score kept programmatic (already 0.90 accuracy)

**Spec**: docs/superpowers/specs/2026-03-22-hybrid-extractor-benchmark-design.md

### 2026-03-22: Hybrid extractor implementation complete (8/9 tasks)

**Decision**: Implemented dual-layer benchmark scoring and HybridExtractor, ready for LLM strategy comparison runs.

**What was built**:
- `autoreview/extraction/scoring_factual.py` — 20-category limitation taxonomy, entity/number extraction, per-field factual scoring
- `autoreview/extraction/hybrid.py` — HybridExtractor with async LLM refinement, abstract-only short-circuit, JSON parse fallback
- `autoreview/llm/prompts/hybrid_extraction.py` — Refinement prompt for cheap LLM pass
- `autoreview/extraction/scoring.py` — Added `compute_dual_composite` with alpha blending
- `scripts/benchmark_extractor.py` — Added `--strategy`, `--alpha` flags, async support, factual+combined reporting
- 63 new tests across 3 test files, all passing

**Baseline results (Strategy A — programmatic only)**:
- Similarity composite: 0.6955
- Factual composite: 0.2237
- Combined (alpha=0.5): 0.4596

**Remaining**: Task 9 — run comparison matrix with LLM strategies (hybrid-haiku, hybrid-sonnet, direct-haiku)

### 2026-03-22: Quality scoring heuristics for computational benchmark papers — PHYBench (`5221c2b4a14c431b951d77b2f6ca7b12`)

**Decision**: Quality score 0.77 assigned to PHYBench, reflecting strong design (original problems, systematic curation, clear baselines) offset by single-domain constraint (physics only).

**Template for computational papers**: Assign high quality (>=0.75) when a benchmark study (a) uses original, non-contaminated items, (b) employs systematic curation to remove flawed items, and (c) compares against established baselines including human performance. Deduct for narrow domain coverage or lack of inter-rater reliability reporting.

**Evidence strength for benchmark findings**: Findings from well-designed benchmarks with clear quantitative results should be tagged `evidence_strength: moderate`. This reflects that rigorous benchmarking provides reliable signal but lacks RCT/cohort experimental structure — it is still stronger than expert opinion or case studies.


### 2026-03-22: Provider-aware concurrency defaults for extraction

**Decision**: Auto-detect LLM provider type and set concurrency accordingly: ClaudeCodeProvider (claude -p) gets 20 concurrent, API providers get 5 concurrent.

**Rationale**: Benchmarking showed 20 parallel claude -p calls use only ~16% memory and are I/O-bound (waiting for API responses). Local CPU impact is minimal. API providers need lower concurrency to respect rate limits. Provider detection uses class name check — simple and no coupling to provider internals.

**Implementation**: Both HybridExtractor and PaperExtractor auto-detect from provider type. Config has explicit max_concurrent_claude_code=20 and max_concurrent_api=5 fields. Benchmark runner uses 20 for async strategies. Branch: feat/parallel-extraction.

### 2026-03-22: LLM-as-Judge Scoring Granularity (informed by PsyCrisis paper)

**Context**: PsyCrisis (psycrisis-2026) demonstrated that binary pass/fail scoring per criterion outperforms continuous Likert scales for LLM judge consistency and traceability in domain-specific evaluation.

**Decision**: Retain ARISE 0-3 continuous scale for now but document binary threshold alternative (pass/fail + rationale) as a candidate improvement for AutoReview's custom evaluation rubric.

**Alternatives considered**: Switching ARISE subcriteria to binary immediately — deferred because it would break comparability with published ARISE scores. The binary approach is preferred for any new custom rubrics (e.g., synthesis quality, citation accuracy gates) where comparability is not required.

**Consequences**: If AutoReview's critique gates (synthesis_quality >= 0.65, citation_accuracy >= 0.60) are later exposed as judge rubrics, they should use binary + rationale format rather than continuous scores for improved reproducibility. Source: psycrisis-2026.

### 2026-03-22: Classifying exploratory computational studies in extraction schema

**Context**: Rai (2023) LLM ethics audit uses 14 prompts across ethical scenarios -- a small exploratory design that does not fit neatly into RCT, survey, or empirical_evaluation categories.

**Decision**: Classify ethics audits and similar exploratory prompt-based studies as 'computational' study_design when they: (a) use LLM API calls as the primary data collection mechanism, (b) apply no statistical hypothesis testing, and (c) have no human subjects. The 'computational' label captures that data is generated via model inference even if the analysis is qualitative.

**Alternatives considered**: 'qualitative' (rejected -- implies human participant interviews); 'empirical_evaluation' (rejected -- implies controlled conditions with statistical analysis). 'computational' is the best available fit and should be noted in the extraction schema enum docs.

**Consequences**: Downstream consumers filtering for computational papers will correctly include ethics audits. Meta-analyses should additionally filter by sample_size and quality_score to distinguish large-scale benchmarks from small exploratory audits.

### 2026-03-22: Quality scoring for audits vs traditional empirical studies

**Context**: Rai (2023) scored 0.55 quality -- penalized for small sample (14 prompts) and lack of statistical rigor, yet the audit yielded novel findings about authoritarian biases.

**Decision**: Apply the standard quality_score formula to audits without a domain-specific adjustment. Small sample and absence of formal validity testing should lower quality_score as appropriate -- this signals to downstream consumers that findings need replication. However, when reporting key_findings from audit papers, do NOT further penalize for small N: qualitative audits with novel scenario designs contribute categorical insights (e.g., 'authoritarian bias under law enforcement framing') that are valid at any sample size. The quality_score governs evidence weight in meta-analysis; it does not gate which findings are reported.

**Consequences**: Audits with N<20 will have quality_score 0.4-0.6, placing them in the low-medium tier alongside other exploratory work. Section writers should present their findings as 'preliminary evidence' or 'initial indications' unless corroborated by larger studies.

### 2026-03-22: Extraction Schema for Benchmark-Framework Papers

**Context**: Papers whose primary contribution is a benchmark (e.g., a self-evolving benchmark framework) present schema ambiguity — the benchmark IS the contribution, but the datasets field normally captures datasets generated or used by the paper.

**Decision**: When a paper's primary contribution is a benchmark (not a training corpus), note it explicitly in the dataset field with a 'benchmark' annotation (e.g., 'benchmark: GSM8K, CLUTRR, StrategyQA, BoolQ [evaluation inputs, not generated datasets]'). The contribution field should be tagged 'benchmark-framework' to distinguish from model or dataset contributions.

**Why**: Standard schema conflates benchmark-as-contribution with datasets-used-for-training. Explicit annotation prevents downstream filtering errors (e.g., incorrectly treating a benchmark paper as a dataset paper, or excluding it from model-paper analyses when it should be included as an evaluation tool).


### 2026-03-23: sample_size for multi-dataset benchmark papers (MultiMedQA)

**Context**: MultiMedQA bundles six existing datasets plus a new HealthSearchQA dataset (3,173 questions). The total combined corpus is much larger than 3,173.

**Decision**: Set sample_size to 3,173 (HealthSearchQA primary dataset) rather than the total MultiMedQA size. HealthSearchQA is the primary new contribution of the paper; the existing six datasets are evaluation baselines, not novel contributions.

**Consequences**: Downstream filtering by sample_size will correctly weight this paper as a mid-scale dataset contribution rather than a large corpus paper. Extraction notes should clarify total MultiMedQA corpus size separately.

### 2026-03-23: Consolidating duplicate findings from programmatic draft

**Context**: Programmatic extraction of MultiMedQA paper produced 10 findings with MedQA performance results appearing 3 times across different finding entries.

**Decision**: Synthesize duplicate findings into single strong claims with exact quantitative results. Consolidated to 9 non-redundant claims. Prefer one precise finding over multiple overlapping findings that dilute signal in downstream meta-analysis.

**Consequences**: Extraction pipeline should include a deduplication/synthesis pass before finalizing key_findings. This is especially important for papers with multiple evaluation tables that report the same model on multiple datasets.

### 2026-03-23: Quality score 0.85 for strong benchmarking studies with incomplete inter-rater reporting

**Context**: MultiMedQA has strong benchmarking methodology with human evaluation across multiple clinician-assessed dimensions (factuality, comprehension, reasoning, harm), but inter-rater reliability metrics are not documented in available excerpts.

**Decision**: Quality score 0.85 reflects strong benchmarking methodology; deduct 0.15 for sparse inter-rater reliability and framework detail in provided excerpts. This is the standard deduction for missing IRR in human evaluation studies -- apply consistently.

**Consequences**: Papers with documented IRR (Cohen kappa >= 0.7, Krippendorff alpha >= 0.7) can score 0.90+. Papers with human evaluation but no IRR documentation are capped at ~0.85 regardless of other methodological strengths.


### 2026-03-23: Three-layer anti-hallucination system for hybrid extraction

**Decision**: Implemented grounding verification in HybridExtractor with three complementary layers: verified numbers in prompt, grounding constraints in system prompt, and post-filter with 20% numeric tolerance.

**Rationale**: Haiku hallucination rate is ~7% for quantitative values. For a review paper pipeline, semantic synthesis of claims is acceptable but fabricated numbers are not. The post-filter keeps claim text (useful for review) but strips quantitative_result when numbers do not match source text within 20% tolerance. Branch: feat/parallel-extraction.

### 2026-03-23: Benchmark comparison matrix complete — Direct-Haiku wins

**Decision**: Direct-Haiku (full paper → Haiku via claude -p) is the best extraction strategy for cost/quality tradeoff.

**Results** (similarity composite vs Sonnet gold):
- Programmatic: 0.6955 ($0) — good key_findings but raw text for methods/limitations
- Hybrid-Haiku: 0.6933 (~$0.30) — LLM calls failing due to tool-blocking issues, mostly fell back to programmatic
- Direct-Haiku: 0.7476 (~$1.12) — best overall, retains ~75% of Sonnet quality at 12x lower cost
- Direct-Sonnet (gold): 1.0000 (~$13.42)

**Key insight**: The hybrid approach did not outperform because (a) --disallowedTools broke most LLM calls and (b) the 2K token condensed context loses too much information vs sending the full paper. Direct-Haiku with full paper context produces better methods/limitations/sample_size.

**Next steps**: Either fix claude-p tool blocking for hybrid, or adopt Direct-Haiku as default extraction strategy with programmatic grounding filter as post-check.

### 2026-03-23: Exhaustive extraction prompt (V4) with strict evidence criteria

**Decision**: Updated extraction prompts to: (1) exhaustive mode — extract ALL findings, one per result, no consolidation, (2) strict evidence strength — default to weaker, require quantitative results for moderate, require RCT/meta-analysis/N>1000 for strong, (3) detailed comparisons — always include both sides with exact numbers.

**Results**: Haiku V4 with exhaustive prompt produces 2-3x more findings than Sonnet (21.5 avg vs 9.3) while capturing 630f Sonnet's specific numbers and 960n best papers. Evidence strength now matches Sonnet's calibration on qualitative vs quantitative papers. Branch: feat/parallel-extraction.

### 2026-03-23: Local model benchmarking — Qwen 3.5 35B via Ollama
- **Decision**: Added `direct-qwen` strategy to benchmark runner to compare local Qwen 3.5 35B extraction against Haiku and Sonnet
- **Context**: User wants to evaluate whether a local open-weight model can match cloud API quality for paper extraction
- **Implementation**: OllamaLLMProvider already existed; added strategy to benchmark_extractor.py with num_ctx=32768, max_tokens_structured=8192, sequential (GPU-bound) concurrency
- **Benchmark design**: 20-paper full-text subset (manifest_subset20.json) for faster iteration; Sonnet extractions serve as ground truth (no rerun needed)

### 2026-03-25: Sequential processing for overnight KG extraction runs
**Context:** Building kg_runner.py for extracting structured KG data from 100 gastruloid papers overnight via claude -p.
**Decision:** Set concurrency=1 (sequential processing) for overnight runs.
**Rationale:** Minimizes token flux per hour, avoids rate-limit pressure, and makes progress predictable — at ~2 min/paper the 100-paper run completes in ~3.5 hours unattended.

### 2026-03-25: Relevance keyword filter after citation-count sort in SearchAggregator
**Context:** SearchAggregator sorts by citation count; for niche topics this surfaces high-citation tangentially related papers (cancer stem cells, MOFA, etc.) above the target field.
**Decision:** Apply regex relevance filter requiring "gastruloid" or related terms in title/abstract after aggregation, before extraction.
**Rationale:** Citation-count ranking is domain-agnostic; a relevance filter restores precision for niche queries without sacrificing recall.

### 2026-03-25: ClaudeCodeProvider (claude -p) over direct API for KG extraction
**Context:** kg_runner.py needed an LLM backend for calling the mycelium extraction prompt at scale.
**Decision:** Use `claude -p` (ClaudeCodeProvider) instead of direct Anthropic API calls.
**Rationale:** No API key management needed, uses existing Claude Code auth, consistent with the pattern established by the programmatic extractor benchmark runner.

### 2026-03-25: System-prompt / user-prompt split for prompt caching in KG extraction
**Context:** The mycelium extraction prompt is ~34K chars — constant across all 100 paper calls.
**Decision:** Pass the extraction prompt as the system prompt and paper text as the user prompt.
**Rationale:** Anthropic prompt caching operates on the system prompt; keeping it constant across calls enables cache hits and reduces effective token cost for large batches.

### 2026-03-25: Disk-cached extraction results per paper hash in kg_runner.py
**Context:** Overnight runs risk interruption; re-extracting already-processed papers wastes tokens and time.
**Decision:** Cache extraction results to disk keyed by a hash of the paper content.
**Rationale:** Enables transparent resume on crash/interrupt without re-processing completed papers.

---

## 2026-03-25 — Switch KG extraction from `claude -p` to direct Anthropic API

**Context:** Batch KG extraction via `claude -p` subprocess was producing 26K–95K tokens per paper with no ceiling, returning generic rc=1 errors on rate limits, and unable to use prompt caching for the 34K system prompt.

**Decision:** Replace `claude -p` calls in kg_runner.py with direct ClaudeLLMProvider (Anthropic Python SDK) calls.

**Rationale:**
- `claude -p` has no `--max-tokens` flag — output is unbounded regardless of MAX_OUTPUT_TOKENS config
- `claude -p` does not expose `cache_control: {"type": "ephemeral"}` — every call re-processes the full system prompt
- `claude -p` returns generic rc=1 "unknown error" for rate limits — retry logic cannot distinguish error types
- ClaudeLLMProvider already exists in the codebase and supports all three missing capabilities

**Trade-off:** Slightly more code than a subprocess call, but eliminates all three root-cause issues in one change.

### Comprehensive Corpus Retrieval Strategy (2026-03-25)
- **Decision**: Use all 6 search sources with 56 queries and no paper cap for corpus retrieval
- **Context**: Original kg_runner.py used 3 sources, 14 queries, capped at 100 papers
- **Result**: 371 papers (3.7x expansion), 311 with full text (83.8% coverage)
- **Script**: `Paper Extractor/KnowledgeGraph Extraction/retrieve_corpus.py`
- **Rationale**: KG extraction benefits from comprehensive field coverage; more papers = richer knowledge graph

---
## 2026-03-25 — Corpus Archival: Abstract-Only Papers

**Decision**: Archived 60 abstract-only / no-full-text papers from the gastruloid KG corpus, leaving 311 full-text papers as the primary corpus in papers.json.

**Rationale**: KG extraction from abstracts alone produces shallow, low-confidence triples. Full-text papers provide the methods, results, and discussion sections needed for meaningful entity/relation extraction. Archival (not deletion) preserves optionality.

---
## 2026-03-25 — Batch Extraction via Anthropic Message Batches API

**Decision**: Used the Anthropic Message Batches API for KG extraction rather than sequential per-paper calls.

**Rationale**: 50% cost discount (~$4.30 vs ~$8.60 estimated) with no change to extraction quality. The same prompt, truncation config, and enum coercion logic from kg_runner.py were reused in batch_extract.py. Resumability added via --poll flag so batch results can be retrieved after the server-side job completes without resubmission.

**Batch ID**: msgbatch_01TB1rvAVqEwtxwdo9dH4Dt4 (311 papers, submitted 2026-03-25)

## Accept 93% extraction rate for gastruloid prototype KG (2026-03-25)

**Context:** Batch run produced 289/311 papers extracted successfully; 22 papers were unrecoverable after JSON repair (mid-string truncation with no valid partial structure).

**Decision:** Accept 289/311 (93%) as sufficient for the prototype knowledge graph rather than re-submitting the 22 failed papers.

**Rationale:**
- Re-submission would cost additional API credits and time for marginal gain on a prototype
- 289 papers with 2,707 assertions and 3,124 evidence units is sufficient to validate the KG construction pipeline
- The 22 failures are likely the longest/most complex papers -- re-submission with higher max_tokens can be deferred to the production run

**Alternatives rejected:**
- Re-submit all 22: deferred to production run
- Re-submit with split extraction strategy: deferred; would require schema refactor

---

## 2026-03-25 — Knowledge Graph Module: Code Location

**Decision**: Knowledge graph code lives at `autoreview/knowledge_graph/` inside the AutoReview module tree.

**Rationale**: Keeps the graph tightly coupled to pipeline data models and avoids a separate top-level package. Consistent with the existing `autoreview/` namespace used by all other pipeline components.

**Alternatives considered**: Top-level `knowledge_graph/` package (rejected — would fragment the namespace), separate repo (rejected — premature).

---

## 2026-03-25 — Knowledge Graph Persistence: NetworkX Prototype with Graduation Path

**Decision**: Use NetworkX in-memory graphs for the prototype. Formalize a graduation path to SQLite (medium scale) or Neo4j (production scale) when the prototype validates the schema.

**Rationale**: NetworkX requires zero infrastructure, ships with the Python ecosystem, and is sufficient for exploring graph algorithms on hundreds-to-low-thousands of nodes. Graduation triggers: node count >10K, need for persistent cross-session queries, or multi-user access.

**Alternatives considered**: Start directly with SQLite (rejected — premature optimization before schema is stable), Neo4j from day one (rejected — ops overhead unjustified at prototype stage).

---

## 2026-03-25 — Knowledge Graph: Primary Use Case Scope

**Decision**: Build a general-purpose knowledge graph supporting ALL three primary use cases: exploration/discovery, contradiction detection, and gap analysis.

**Rationale**: User confirmed all three use cases are equally important. A general graph with layered analyses (rather than a single-purpose structure) avoids prematurely optimizing the schema for one use case at the expense of others.

**Pending**: Entity deduplication strategy — options under consideration are strict ontology match, hybrid fuzzy matching, or LLM-assisted deduplication. Decision deferred.


## 2026-03-25 — Entity Deduplication Strategy

**Decision**: Entity deduplication in the knowledge graph will use a **hybrid approach**: ontology ID matching (exact match on canonical IDs such as MeSH, UniProt, GO terms) as the primary key, with fuzzy string matching as a fallback for entities lacking ontology coverage.

**Rationale**: Pure string matching produces false positives (e.g., "IL-6" vs "Interleukin-6") and false negatives (abbreviation variants). Ontology IDs provide ground-truth identity when available. Fuzzy matching catches cases where the same concept appears under different surface forms without an assigned ID.

**Status**: Finalized. Not yet implemented.

---

## 2026-03-25 — KG Hybrid Architecture (B+C)

**Decision**: Adopt B+C hybrid approach for knowledge graph: layered pipeline (B) combined with Pydantic data contract models (C).

**Module structure**: 8 modules under `autoreview/knowledge_graph/` — entity extraction, edge extraction, deduplication, belief propagation, graph construction, querying, serialization, and integration.

**Pydantic models approved**:
- `KGEntity` — deterministic `entity_id` via SHA256 hash of (type, canonical_name), alias accumulation during dedup
- `KGEdge` — typed relationships with evidence linking
- `BetaPosterior` — Beta(1,1) uninformative prior for belief propagation, evidence strength weighting
- `KGEvidenceLink` — provenance linking edges to source documents

**Rationale**: Deterministic entity hashing enables stable deduplication across pipeline runs. Beta(1,1) uninformative prior avoids biasing claims before evidence is accumulated. Pydantic models enforce the data contract at module boundaries.

**Status**: Section 2 (Pydantic models) presented; awaiting user approval to proceed to Section 3.

## Use subagent extraction for remaining papers after batch API truncation (2026-03-26)

**Decision**: For the 22 papers that failed batch API extraction (truncated + JSON repair could not recover), used Claude Code sonnet subagents for re-extraction rather than re-submitting to batch API.

**Rationale**:
- Re-submitting to batch API risks the same truncation failure (same model, same token limit)
- Subagents use sonnet (better schema compliance than haiku) and have no output token ceiling in the same way
- Subagents self-validate with Pydantic and self-correct errors inline — batch API cannot do this
- 5–6 parallel subagents complete a batch of ~25 papers in ~5–6 minutes, acceptable latency
- Avoids API key reuse patterns that could trigger rate-limit issues on batch re-submission

**Outcome**: 22/22 remaining papers extracted successfully; 311/311 total, 100
## Use subagent extraction for remaining papers after batch API truncation (2026-03-26)

**Decision**: For the 22 papers that failed batch API extraction (truncated + JSON repair could not recover), used Claude Code sonnet subagents for re-extraction rather than re-submitting to batch API.

**Rationale**:
- Re-submitting to batch API risks the same truncation failure (same model, same token limit)
- Subagents use sonnet (better schema compliance than haiku) and have no output token ceiling in the same way
- Subagents self-validate with Pydantic and self-correct errors inline — batch API cannot do this
- 5-6 parallel subagents complete a batch of ~25 papers in ~5-6 minutes, acceptable latency
- Avoids API key reuse patterns that could trigger rate-limit issues on batch re-submission

**Outcome**: 22/22 remaining papers extracted successfully; 311/311 total, 100% validated.

## 2026-03-25 — Knowledge Graph Spec: Assertion Dedup, Predicate Normalization, Evidence Independence

### Decision: Add assertion-level deduplication to KG spec
- **Context**: Three-tier claim architecture proposal (from prior conversation) highlighted that raw edges without dedup make cross-paper discovery unreliable
- **Decision**: Same `(subject_id, canonical_predicate, object_id)` triple collapses into a single `KGEdge` with accumulated evidence links
- **Rationale**: ~2,900 raw assertions → estimated ~1,500-2,000 merged edges; dedup is what makes the graph useful for finding convergent evidence across papers
- **Location**: `docs/superpowers/specs/2026-03-25-knowledge-graph-prototype-design.md`

### Decision: Add predicate normalization (synonym families)
- **Context**: Different papers express the same relationship differently (induces/activates/triggers), causing spurious duplicate edges
- **Decision**: Maintain synonym family lookup table with exact match + fuzzy fallback (rapidfuzz, cutoff 85); all synonyms map to canonical form before dedup
- **Rationale**: Normalization is prerequisite for meaningful triple merging; fuzzy fallback handles novel phrasings not yet in lookup table

### Decision: Add evidence independence weighting to confidence scoring
- **Context**: Multiple papers from the same lab are correlated evidence, not independent replication
- **Decision**: Diminishing returns decay: 0.5x weight per additional evidence unit from the same author group
- **Rationale**: Independence is the correct statistical criterion for evidence accumulation; same-lab replications should not linearly inflate confidence

### Decision: Make three-tier architecture explicit in KG spec
- **Tiers**: T1 = assertions (KGEdge — mechanism-level claims, what scientists argue about), T2 = evidence (KGEvidenceLink — experimental demonstrations, what determines support), T3 = provenance (paper_id + author metadata, what determines independence)
- **Context**: Structure was implicit in existing spec; made explicit after user proposal clarified the conceptual separation
- **Impact**: Guides all downstream KG query and confidence scoring logic

## 2026-03-25 — Knowledge Graph Prototype: Schema and Architecture Decisions

### KG Extractions Use Mycelium ExtractionResult Schema (Not AutoReview PaperExtraction)
- **Decision**: The KG ingest module must parse mycelium ExtractionResult format directly
- **Rationale**: Extraction JSONs in gastruloid_run/extractions use mycelium schema with top-level keys: `paper_provenance`, `evidence_units`, `assertion_drafts`, `citation_contexts`, `extraction_metadata` — NOT AutoReview PaperExtraction model
- **Impact**: Task 1 (ingest module) must map ExtractionResult fields, not PaperExtraction fields

### networkx Added as New Dependency
- **Decision**: Add `networkx` to pyproject.toml optional deps or core deps for KG prototype
- **Rationale**: networkx is NOT currently installed in conda env; rapidfuzz is in pyproject.toml but also not installed — both need `pip install` after adding
- **Impact**: Add to pyproject.toml `[project.optional-dependencies]` or `[project.dependencies]` before running any KG tasks

### PredicateNormalizer Is a Class (Not a Function)
- **Decision**: Implement `PredicateNormalizer` as a stateful class with a `.log` attribute
- **Rationale**: Code reviewer flagged BLOCKER-2: normalization log is needed for auditability; a plain function cannot accumulate the log across multiple calls
- **Impact**: Task 3 (predicate normalization) implements `PredicateNormalizer` with `.normalize(pred)` method and `.log: list[NormalizationEntry]`

### MergeResult Dataclass Wraps Merged Assertions + Merge Log
- **Decision**: Dedup/merge step returns a `MergeResult` dataclass, not a bare list
- **Rationale**: Code reviewer flagged BLOCKER-3: assertion_merge_log was missing; wrapping in a dataclass makes the log a first-class output, enabling downstream auditability
- **Impact**: Task 4 (dedup) returns `MergeResult(assertions: list[KGAssertion], merge_log: list[MergeEntry])`

### Implementation Plan: 10 Tasks, 9 Sequential Batches (Tasks 7+8 Parallel)
- **Decision**: Tasks run sequentially with one parallel batch (Tasks 7 analysis + 8 visualization)
- **Rationale**: Strong schema dependencies chain tasks 1→2→3→4→5→6→(7‖8)→9→10; code reviewer corrected Task 4 to depend on Task 3, not run in parallel
- **Plan file**: `docs/superpowers/plans/2026-03-26-knowledge-graph-prototype.md`

### Self-Loop Tests Added to Graph and Dedup Modules
- **Decision**: Both test_graph.py and test_dedup.py include explicit self-loop rejection tests
- **Rationale**: Code reviewer flagged BLOCKER-4: self-loops (subject == object) are a degenerate case that must be caught and rejected at graph ingestion
- **Impact**: Ingest and dedup modules must validate subject \!= object on all assertions

---

## 2026-03-26 — Dual-Prompt Extraction Architecture (KG vs AutoReview)

**Context:** Built a new KG-specific extraction pipeline alongside the existing AutoReview extraction pipeline.

**Decision:** Maintain two separate extraction prompts with distinct design goals:
- **AutoReview prompt**: Capped (12 assertions, 15 evidence, 12K tokens), rich schema with provenance/epistemic_function/citation_contexts/nested ontology objects. Optimised for reliability and structured output compatible with the review pipeline.
- **KG prompt**: Uncapped, lean schema (~100 tokens/claim vs ~300), section-aware tagging (primary_empirical / interpretive / attributed_prior / methodological), Abstract skipped as redundant. Optimised for maximum recall.

**Schema differences (KG vs AutoReview):**
- Dropped: `provenance`, `epistemic_function`, `citation_contexts`, nested ontology condition objects
- Kept: `conditions` (flat strings), `evidence_strength`, `certainty`, `predicate` — all edge-relevant fields
- Contradiction handling: Discussion contradictions with prior work flagged HIGH VALUE and extracted as paired claims (one `attributed_prior` + one `primary_empirical`/`interpretive`)

**Rationale:** The original caps caused 83
---

## 2026-03-26 — Dual-Prompt Extraction Architecture (KG vs AutoReview)

**Context:** Built a new KG-specific extraction pipeline alongside the existing AutoReview extraction pipeline.

**Decision:** Maintain two separate extraction prompts with distinct design goals:
- **AutoReview prompt**: Capped (12 assertions, 15 evidence, 12K tokens), rich schema with provenance/epistemic_function/citation_contexts/nested ontology objects. Optimised for reliability and structured output compatible with the review pipeline.
- **KG prompt**: Uncapped, lean schema (~100 tokens/claim vs ~300), section-aware tagging (primary_empirical / interpretive / attributed_prior / methodological), Abstract skipped as redundant. Optimised for maximum recall.

**Schema differences (KG vs AutoReview):**
- Dropped: `provenance`, `epistemic_function`, `citation_contexts`, nested ontology condition objects
- Kept: `conditions` (flat strings), `evidence_strength`, `certainty`, `predicate` — all edge-relevant fields
- Contradiction handling: Discussion contradictions with prior work flagged HIGH VALUE and extracted as paired claims (one `attributed_prior` + one `primary_empirical`/`interpretive`)

**Rationale:** The original caps caused 83% truncation of actual claim content. Two prompts let each use case optimise independently without compromising the other.

**Files:**
- `Paper Extractor/KnowledgeGraph Extraction/kg_schema.py`
- `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md`
- `Paper Extractor/KnowledgeGraph Extraction/batch_extract_kg.py`

### Claim-centric graph visualization over entity-centric (2026-03-26)
- Chose claims-as-nodes with shared-entity edges over traditional entity graph
- Rationale: entity graph with 2,462 nodes was intractable for text visualization; claim graph surfaces assertions directly
- Used vis.js via CDN (no pyvis dependency) for full control over filter panel
- Trade-off: 13.78 MB HTML file, 29K edges can be sluggish — mitigated by filters

### Semantic contradiction detection: graph-constrained NLI + LLM (2026-03-26)
- Chose Approach C: use existing 29K shared-entity pairs as candidate set
- Two-tier: DeBERTa NLI cross-encoder → LLM verification on top candidates
- Rejected: brute-force embedding (4.2M pairs), literal predicate matching (misses nuance)
- Rationale: graph structure already constrains to biologically related claims, integrates back into Beta-Binomial

### Semantic contradiction detection: graph-constrained NLI + LLM (2026-03-26)
- Chose Approach C over brute-force embedding (4.2M pairs) or literal predicate matching
- Pipeline: 29K shared-entity pairs → DeBERTa NLI cross-encoder → top ~500 → LLM verify
- Cost: $0 for NLI (local), ~$0.05 for LLM verification (Haiku)
- NLI p_contra integrates directly into Beta-Binomial as calibrated probability weight
- Rejected Approach A (NLI-only, misses scientific nuance) and B (expensive LLM on all candidates)

### Cross-claim beta propagation design (2026-03-26)
- Extend confidence.py with score_cross_claim_contradictions() function
- NLI p_contra gates how much claim B's evidence counts against claim A (and vice versa)
- Preserves existing evidence strength weighting and author independence discounting
- Three gaps identified: cross-claim propagation (blocking), extraction always "supports" (blocking), independence model extension (quality)

### Extraction prompt improvements needed (2026-03-26)
- KG extraction prompt: add result_summary field, add assertion_links with per-claim direction
- Mycelium prompt: add explicit "refutes" classification guidance with examples
- ingest.py: concatenate effect_description into experiment_summary (data already exists, just dropped)
- Priority: ingest.py fix is zero-cost (uses existing data), prompt fixes need re-extraction

### 2026-03-26: KG extraction v4 as standard extraction approach

**Decision**: The v4 KG extraction pipeline (kg_schema.py, kg_extraction_prompt.md, batch_extract_kg.py) is the standard for all knowledge graph extraction going forward. The AutoReview pipeline (mycelium_extraction_prompt.md, batch_extract.py) remains untouched for review paper generation.

**Key design choices**:
- evidence_links with per-claim direction (supports/refutes/mixed) instead of flat evidence_ids
- result_summary on evidence (conclusions, not methods)
- Predicate coercion layer handles Haiku vocabulary drift deterministically
- Post-processing flips absence claims to refutes
- Prompt hardening: predicate table format, claim_type/section_source disambiguation, figure coverage requirement

**Validated**: 4 prompt iterations on Veenvliet 2020 (Science). Haiku v4 + coercion = 0 schema violations, 65 claims, 39 evidence units, 3 native refutes links.

### Namespace evidence IDs with paper_hash (2026-03-26)
- **Decision**: Prefix all per-paper IDs (evidence_id, draft_id, evidence_unit_ids, assertion_draft_ids) with `paper_hash::` in ingest.py.
- **Why**: Without namespacing, dict comprehension collision loses 99% of evidence provenance.
- **Trade-off**: Slightly longer IDs, but correctness is non-negotiable.
- **Scope**: Applied to all four ID fields in ingest.py; __init__.py lookup unchanged (reads namespaced IDs naturally).

### Contradiction viz: separate computation from rendering (2026-03-26)
- **Decision**: Created contradiction_viz.py (computation) separate from interactive.py (rendering).
- **Why**: Contradiction scoring (community labels, disagreement scores, cross-paper classification) has value beyond HTML — feeds CLI, reports, future analysis.
- **Trade-off**: Extra module, but clean separation. interactive.py accepts optional ContradictionVizData.

## 2026-03-27 — Contradiction Visualization Architecture

- **Separate computation module for contradiction viz**: `contradiction_viz.py` handles computation of contradiction edges, community auto-labeling, and cross-community disagreement scores. Kept separate from `interactive.py` (rendering) — decouples analysis logic from HTML/JS generation and makes each testable independently.
- **Community disagreement normalization**: Cross-community disagreement score normalized by `sqrt(size_a * size_b)` to prevent large communities from dominating the ranking. Raw edge counts would systematically favor large-community pairs.
- **Default contradiction threshold = 0.9**: Set higher than the analytical threshold (0.8) for performance — fewer edges rendered by default. Users can lower it interactively. Prioritizes load-time responsiveness over completeness.
- **Claim ID format standardized**: NLI-format `"{u}__{predicate}__{v}__{k}"` used consistently across `interactive.py`, `nli.py`, and `contradiction_viz.py`. Single canonical format prevents ID mismatch bugs when joining claim sets across modules.

### Use igraph as primary layout engine for KG visualization
- **Date**: 2026-03-27
- **Decision**: Added igraph Fruchterman-Reingold as the primary layout engine in `interactive.py`, with NetworkX spring_layout as fallback
- **Rationale**: NetworkX spring_layout is too slow for large claim graphs (2900+ nodes, 29K+ edges). igraph's C-backed implementation computes the same layout ~10-100x faster. Layout took ~10s for 2899 nodes with igraph.
- **Trade-offs**: Adds igraph as a dependency. Mitigated by try/except ImportError fallback to NetworkX.
- **Alternatives considered**: graph-tool (faster but harder to install via pip), vis.js headless stabilization (still browser-bound)

## 2026-03-27 — VOI Uncertainty Metric: Beta Variance over Differential Entropy
- **Decision**: Used Beta variance (not differential entropy) for uncertainty metric in VOI calculation. `scipy.stats.beta.entropy(1,1) = 0.0` (differential entropy of uniform on [0,1]), making it useless as a normalizer. Beta variance = αβ/((α+β)²(α+β+1)) normalized by max variance (1/12) gives a proper 0-1 uncertainty scale.
- **Decision**: VOI formula is `betweenness × uncertainty × blast_radius` with fallback to 1 when `blast_radius=0`. This ensures high-betweenness uncertain contradictions still rank even if they do not fragment the graph.
- **Decision**: Resolution simulation uses 3 scenarios: `a_wins` removes node B, `b_wins` removes node A, `dissolved` removes just the edge. `blast_radius` = max delta_components across all 3 scenarios.

## Topology Highlight Design — 2026-03-27
- **Decision**: Topology highlight uses red border (#FF3333, 3.5px) rather than changing node color, so it composes with both assertion-type and community coloring modes without conflict. Hidden by default (toggle off), sidebar section hidden entirely when no topology data is loaded.
- **Rationale**: Composability — users can combine topology highlighting with any other color mode without visual conflict.

### Community subfield labeling: combined LLM + heuristic approach (2026-03-27)
**Decision**: Use heuristic subfield labels (entity-type priority ranking) for all communities, with optional LLM enrichment for significant communities (>=5 claims or in disagreements).
**Rationale**: Heuristic is fast/free and covers all 319 communities. LLM gives much better domain-specific labels ("BMP4 AP Axis Patterning" vs "BMP4 / Wnt3a") but costs tokens. Combined approach: heuristic as baseline, LLM as optional upgrade for ~30 significant communities in a single batched call.
**Alternatives rejected**: LLM-only (too expensive for 319 communities), heuristic-only (labels too generic for understanding subfields).

### V5 KG Extraction Pipeline
**Date**: 2026-03-27
**Decision**: Upgrade extraction pipeline from v4 to v5 with context-aware contradiction detection.
**Key changes**:
- Added claim-level model_system, organism, in_vitro fields (previously only at evidence level)
- Added certainty→Beta prior width mapping (high/medium/low)
- Added section_source→epistemic weight (primary_empirical 1.0, interpretive 0.7, attributed_prior 0.5)
- NLI context mismatch discount: cross-organism/model contradictions get 0.3x p_contradiction
- Citation contexts with relationship="contradicts" now annotate graph edges
- Ingestion handles both v4 (assertion_drafts) and v5 (claims) JSON formats
**Rationale**: ~30-40% of NLI-detected contradictions were false positives from cross-context comparisons (different species, model systems, or in_vitro vs in_vivo). The v5 changes eliminate these by propagating experimental context to the claim level.
**Files changed**: models.py, ingest.py, confidence.py, nli.py, __init__.py, kg_extraction_prompt.md, test_models.py

### V5 KG extraction schema expansion
**Date**: 2026-03-27
**Decision**: Expanded KG schema with SectionSource, Certainty, QuantitativeContext models; added model_system and organism fields to evidence; extended EntityType (+5) and EvidenceStrength (+2) enums.
**Rationale**: Richer epistemic metadata enables context-aware NLI contradiction detection (organism/model_system mismatch discounting) and section-source weighting in confidence scoring.
**Tradeoffs**: More fields increase prompt complexity and LLM output tokens; 43% quantitative_context fill rate suggests this field needs prompt reinforcement.

### 5-tab analytical visualization + rendering performance (2026-03-27)
**Decision**: Generate 5 pre-plotted tabs (Assertion Types, Community Clusters, Cross-Paper Disputes, Mechanistic Core, Controversial Frontier) with straight solid edges and hub degree capping.
**Rationale**: Different analytical perspectives on the same KG data. Performance required removing vis.js curved/dashed edges and capping hub degree at 20 to keep rendering interactive.

### NLI improvement strategy: extraction-first (2026-03-27)
- **Decision**: Prioritize populating KGEdge v5 context fields (natural_language, negatable_form, conditions, model_system) in the extraction prompt before swapping NLI model.
- **Why**: The NLI model can't calibrate what it can't see. Bare triples provide insufficient signal for nuanced contradiction scoring. Context fields are already defined in the data model but never populated.
- **Plan**: (1) Update extraction prompt to populate v5 fields, (2) Use natural_language as NLI input instead of bare triples, (3) Then evaluate model swap.

### Citation evidence stubs for attributed_prior claims (2026-03-27)
- **Context**: v5 extraction produced 42 claims but only 20 evidence units (0.48 ratio), with all 18 attributed_prior claims having zero evidence links — blocking hypothesis generation for contradiction resolution.
- **Decision**: Generate citation evidence stubs (review_citation strength) for every attributed_prior claim, carrying citing_sentence, source_doi, model_system, organism.
- **Alternative rejected**: Simply exclude attributed_prior from evidence counts — would leave dangling edges unable to participate in VOI ranking or hypothesis framing.
- **Rationale**: Stubs enable the topology/VOI system to frame resolution hypotheses like "Paper A found X in human cells, contradicting Smith et al. who found Y in mouse DCs" — the stub provides the model system context for the cited work.

## 2026-03-27 — KG extraction version history as standalone technical document
- Decided to write extraction_improvements_update.md covering v1→v5 KG extraction evolution for PI communication
- Rationale: version-by-version narrative makes schema rationale and design decisions legible to non-implementation readers

## 2026-03-28 — Batch KG Extraction Model and Token Decisions

### Haiku-only for KG extraction (no Sonnet fallback)
**Decision**: Use Haiku 4.5 exclusively for KG extraction. No Sonnet fallback path.
**Rationale**: User requirement. Cost at scale ($18 estimated for 311 papers) is acceptable with Haiku. Sonnet would 5-10x the cost without proportional quality gain for structured JSON extraction with a detailed schema.
**Applies to**: `batch_extract_kg.py`, `kg-extract` skill

### MAX_OUTPUT_TOKENS = 64000 for Haiku 4.5
**Decision**: Set `MAX_OUTPUT_TOKENS = 64000` (not 128K, not 16K).
**Rationale**: 64K is the hard API ceiling for Haiku 4.5. Setting 128K causes immediate API rejection. The previous value of 16K was causing severe truncation — average output is ~26K tokens/paper, with large papers hitting the 64K ceiling.
**Applies to**: `batch_extract_kg.py`

### Streaming required when max_tokens is high
**Decision**: Use streaming (`client.messages.stream()`) for single-paper extractions when max_tokens is set to 64K.
**Rationale**: High max_tokens values can cause non-streaming calls to exceed the 10-minute API timeout. Streaming keeps the connection alive during long generation.
**Applies to**: Direct API extraction paths in `batch_extract_kg.py`

### Corpus filtering must precede full extraction run
**Decision**: Do not proceed with full 311-paper batch extraction until `retrieve_corpus.py` review filter is improved.
**Rationale**: Review/perspective papers (e.g., from "Current Opinion in…" journals) dominate extraction output with `attributed_prior` claims, inflating cost without adding novel primary empirical knowledge. The current title-only `\breview\b` filter is insufficient.
**Next step**: Improve filter to include journal-name heuristics and/or abstract-level review detection before re-running full corpus extraction.

### D016 — Remove contradiction bias from KG extraction prompt (blinded graph philosophy)
- **Date**: 2026-03-27
- **Status**: accepted
- **Context**: KG extraction prompt contained explicit "HIGH VALUE", "CRITICAL for graph quality", and "HIGH PRIORITY" markers that directed LLMs to surface contradictions during extraction. This pre-labels contradictions at extraction time.
- **Decision**: Remove all contradiction-specific emphasis from `kg_extraction_prompt.md`. Extraction focuses purely on information collection. Contradictions are detected downstream via blinded NLI comparison across papers.
- **Rationale**: Pre-labeling contradictions during extraction introduces author-framing bias. Blinded NLI comparison across independent extractions is more principled: the same claim from two papers will be compared without the extractor having been primed to flag disagreement. Contradiction detection becomes a graph-level signal rather than a per-paper annotation.
- **Applies to**: `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md`, NLI scoring stage

## Use OpenAlex as Primary Source for Corpus Expansion (2026-03-27)

- **Decision**: OpenAlex is the primary source for bulk corpus expansion, not PubMed or Semantic Scholar
- **Rationale**: 100% DOI coverage for this corpus, authoritative `type` field for review detection, free batch API, supports 12+ search terms in parallel
- **Alternatives considered**: PubMed (no type field, API rate limits), Semantic Scholar (lower coverage for embryo organoid literature)
- **Consequence**: expand_corpus.py is built around OpenAlex; review filter in retrieve_corpus.py uses OpenAlex batch lookup as primary layer

## Relevance Threshold: 2+ Terms for Non-Gastruloid Papers (2026-03-27)

- **Decision**: Papers retrieved under broad search terms (e.g., assembloid, trunk-like structure) must match 2+ relevance terms from a curated list to enter the corpus; "gastruloid" in title/abstract auto-passes
- **Rationale**: Broad terms return thousands of papers; single-term relevance is insufficient to prevent corpus bloat
- **Threshold**: 2 relevance terms chosen empirically — 1 term had ~30% false positive rate in spot-check; 3 terms excluded too many genuine papers
- **Implementation**: relevance_terms list in expand_corpus.py; easily tunable

## Exclude Abstract-Only Papers from Extraction Corpus (2026-03-27)

- **Decision**: Papers where only abstract (not full text) was retrievable are archived to papers_archived.json and excluded from the extraction corpus
- **Rationale**: User requirement: complete papers only. Abstract-only extraction would produce sparse, low-confidence claims and inflate the inaccessible paper count
- **Consequence**: retrieve_corpus.py now performs abstract-only exclusion after full text retrieval attempt; papers_archived.json stores all non-primary records

## Token Ceiling: 64K Limit + Truncation Repair, No Section-Level Extraction (2026-03-27)

- **Decision**: Proceed with 64K output token ceiling (Haiku 4.5 hard limit) and existing truncation repair logic; do not implement section-level extraction
- **Rationale**: Micro extraction test showed 0/5 primary research papers hit the ceiling; the 64K limit is sufficient for the current corpus
- **Alternatives considered**: Section-level extraction (Option 2) would increase complexity and API calls without demonstrated benefit
- **Trigger for revisit**: If batch extraction shows >5% of papers hitting the ceiling, revisit section-level extraction
- **Note**: MAX_OUTPUT_TOKENS corrected from 128000 to 64000 in batch_extract_kg.py (was silently ignored by API; now explicit)

### D017 — Adopt PSL/HL-MRF over discrete MRFs for Phase 2 confidence propagation
- **Date**: 2026-03-27
- **Status**: accepted
- **Context**: Phase 2 requires confidence propagation across a KG with potentially 100K+ edges. Evaluated discrete MRFs with loopy belief propagation vs. PSL/Hinge-Loss MRFs.
- **Decision**: Use PSL/HL-MRF with a custom scipy L-BFGS-B implementation (no Java/pslpython dependency).
- **Rationale**: HL-MRFs operate on continuous [0,1] truth values and reduce to a convex optimization problem with guaranteed convergence. Discrete MRF+LBP oscillates and does not scale past ~100K edges. Custom scipy engine avoids the Java runtime requirement of the reference pslpython library.

### D018 — Replace NLI with structural contradiction detection
- **Date**: 2026-03-27
- **Status**: accepted
- **Context**: Phase 2 contradiction detection originally planned to use NLI models (e.g. DeBERTa) on claim text. KG extraction now produces structured triples with controlled predicates and explicit direction fields.
- **Decision**: Drop NLI; implement structural contradiction detection via predicate opposition table + direction comparison + condition disambiguation.
- **Rationale**: Structured triples make contradictions detectable from graph structure alone. NLI was designed for unstructured text and adds ~800ms/pair latency with no accuracy gain on controlled-vocabulary predicates. Structural detection covers ~950f cases; condition metadata handles the boundary-vs-contradiction disambiguation that NLI cannot reliably perform anyway.

### D019 — Extraction v6: two-tier predicate vocabulary
- **Date**: 2026-03-28
- **Decision**: Replaced flat 31-predicate vocabulary with two-tier system: 12 canonical (always preferred) + 7 specific (biochemically precise). Removed all synonym predicates from allowed set — LLM must use canonical forms.
- **Rationale**: Reduces predicate fragmentation, improves entity resolution, enables structural contradiction detection and composition rules. Downstream normalization handles any remaining synonyms.
- **Alternatives**: Keep full vocabulary (rejected: too many synonyms cause inconsistent triples)

### D020 — Evidence strength: 5-value standardized set
- **Date**: 2026-03-28
- **Decision**: Standardized evidence_strength to 5 values across all modules: direct_experimental (1.0), indirect_experimental (0.8), observational (0.5), computational (0.3), review_citation (0.15). Merged observational_controlled/uncontrolled, computational_prediction. Dropped expert_opinion.
- **Rationale**: Previous 7-value set had mismatches across schema/ingest/confidence modules. indirect_experimental and review_citation had zero weight (bug). Simpler set is easier for LLMs to apply consistently.

### D019 — Extraction v6: two-tier predicate vocabulary
- **Date**: 2026-03-28
- **Decision**: Replaced flat 31-predicate vocabulary with two-tier system: 12 canonical (always preferred) + 7 specific (biochemically precise). Removed all synonym predicates from allowed set.
- **Rationale**: Reduces predicate fragmentation, improves structural contradiction detection and composition rules. v6.1 test: 97% canonical predicates vs 74% in v5.

### D020 — Evidence strength: 5-value standardized set
- **Date**: 2026-03-28
- **Decision**: Standardized evidence_strength to 5 values: direct_experimental (1.0), indirect_experimental (0.8), observational (0.5), computational (0.3), review_citation (0.15). Fixed zero-weight bug for indirect_experimental and review_citation.

## D021 — Autoresearch-Inspired Prompt Optimization Loop (2026-03-28)

Built a fully autonomous prompt optimization loop at `Paper Extractor/KnowledgeGraph Extraction/optimize/`. Architecture: Haiku runs extractions, Sonnet acts as optimizer agent guided by `program.md`. 12-metric composite scorer (density, predicate validity/diversity, evidence strength, field coverage, causal_type, perturbation detection, claim type validity). Regression guard rejects changes that drop any metric >5
## D021 — Autoresearch-Inspired Prompt Optimization Loop (2026-03-28)

Built a fully autonomous prompt optimization loop at `Paper Extractor/KnowledgeGraph Extraction/optimize/`. Architecture: Haiku runs extractions, Sonnet acts as optimizer agent guided by `program.md`. 12-metric composite scorer (density, predicate validity/diversity, evidence strength, field coverage, causal_type, perturbation detection, claim type validity). Regression guard rejects changes that drop any metric >5%. Convergence detection stops at <0.5% improvement. Prompt versions saved for audit trail. Inspired by Karpathy autoresearch 3-file pattern: program.md + artifact + runner.

### D022 — Scoring rebalanced toward graph utility (2026-03-28)
- **Decision**: Shifted optimizer scoring weights from format validation (~65
### D022 — Scoring rebalanced toward graph utility (2026-03-28)
- **Decision**: Shifted optimizer scoring weights from format validation (~65%) toward graph-utility metrics. New split: Graph structure 46%, Extraction quality 30%, Format validity 24%. Added 7 new metrics: evidence_linkage (10%), evidence_density (7%), evidence_completeness (6%), doi_coverage (8%), conditions_coverage (6%), entity_consistency (5%), citation_contexts (2%). Total metrics: 19 (was 12).
- **Rationale**: Format validation is largely handled by the production coercion pipeline; the optimizer should focus on what actually makes the knowledge graph useful — evidence linkage, DOI traceability, and conditions coverage for cross-paper edge construction.

### D023 — Keep References section in batch extraction truncation (2026-03-28)
- **Decision**: Updated `batch_extract_kg.py` truncation config to preserve References and Conclusion sections. Only non-informative end-matter (Acknowledgements, Author Contributions, Funding) is dropped.
- **Rationale**: `source_doi` on `attributed_prior` claims is the key to cross-paper graph edges. Dropping References made DOI resolution impossible. The old truncation config was designed for ExtractionResult schema (which had no DOI field), not the KG schema.

### D024 — Apply production coercion pipeline before optimizer scoring (2026-03-28)
- **Decision**: `experiment_runner.py` now applies the full `kg_coerce.py` pipeline before computing scores, matching production behavior.
- **Rationale**: Without coercion, the optimizer wasted iterations fixing predicate remapping and field normalization that the production pipeline already handles automatically. Coercion is extracted into shared `kg_coerce.py` for reuse by both batch extraction and the optimizer.

## D025 — Expanded optimizer test corpus (2026-03-28)
Removed `_SKIP_INDICES` skip list from `experiment_runner.py` (was {0,1,3,4,5,9}) and added `extra_corpus_path` parameter to `load_test_papers()`. Selected 3 diverse review papers (cardiac, neurogenesis, vertebrate organogenesis) from `papers_reviews.json` for `optimize/extra_test_papers.json`. Expanded corpus from 3 to 13 papers for better domain diversity during optimization runs.
- **Files**: `experiment_runner.py`, `optimize_extraction_prompt.py`, `optimize/extra_test_papers.json` (new)
- **Rationale**: Broader corpus catches prompt regressions across domains; removed skip list after confirming indices 1,3,4,5 have 20K–85K chars of usable text

## D026 — Configurable version prefix for optimizer (2026-03-28)
Added `--version-prefix` CLI arg to `optimize_extraction_prompt.py` (default `v7`) to avoid overwriting existing `v6.x` versioned outputs. Also added `--extra-papers` arg for extra corpus path and configurable baseline/version labels.
- **Rationale**: Without this, re-running the optimizer would silently overwrite prior version outputs; explicit prefix makes run provenance clear

## D027 — Expanded optimizer corpus to 25 papers (2026-03-28)
**Decision**: Replaced 3 review papers in `optimize/extra_test_papers.json` with 15 real research papers, bringing the total optimizer corpus to 25 papers (rai14 + 9 micro_sample + 15 corpus papers).
**Why**: Review papers have different extraction characteristics than research papers (broader claims, less experimental detail, fewer methods/results triples). The optimizer should calibrate prompt quality on research paper extraction, not reviews.
**Filter criteria**: Papers from `papers.json` with DOI required, 15K–150K chars, excluding reviews/editorials/protocols.

## D028 — experiment_runner: switched to --output-format json + single-turn extraction
**Date:** 2026-03-28
**Context:** experiment_runner.py was using `--output-format text` + `--max-turns 5`, causing large JSON outputs to be silently truncated and multi-turn splits to lose the JSON prefix.
**Decision:** Switched to `--output-format json` (full response preserved in wrapper) + `--tools "" --max-turns 1` (single turn, no tool calls, no concatenation artifacts). Added section-aware truncation matching production batch_extract_kg.py (100K char limit, same keep/drop sections). Added "Maximum 50 claims" instruction to keep output within CLI token budget. Brace-slice JSON parsing handles markdown fences the model may add.
**Alternatives considered:** Direct API via ANTHROPIC_API_KEY — would allow explicit max_tokens=64K control matching production, but requires additional auth wiring. Deferred as future improvement; CLI fix is sufficient for optimizer runs.
**Outcome:** Previously failing papers (micro_3: 45c/45e, micro_4: 20c/20e) now succeed with complete extraction.

### Use CLI (claude -p) for Optimizer Extractions (2026-03-28)
- **Decision**: Use `claude -p` CLI for KG extraction in the prompt optimizer, not direct Anthropic API
- **Context**: User preference; CLI is already integrated in the experiment runner and works consistently with `--output-format json --max-turns 1 --tools ""`
- **Alternatives**: Direct Anthropic API via SDK (more reliable, no subprocess overhead, better error handling) — rejected by user preference

### Expanded Optimizer Corpus to 25 Papers (2026-03-28)
- **Decision**: Expand optimizer corpus from 10 to 25 papers (rai14 + 9 micro_sample + 15 extra research papers)
- **Context**: Larger corpus improves signal quality for prompt optimization; `--extra-papers` arg allows flexible corpus extension without code changes
- **Alternatives**: Smaller corpus (faster, cheaper) — rejected to get better coverage of paper types and edge cases

### experiment_runner.py: CLI subprocess to Anthropic streaming API (2026-03-28)
- **Decision**: Replace `subprocess.run(["claude", "-p", ...])` with `anthropic.Anthropic().messages.stream()` in `experiment_runner.py`
- **Context**: Non-streaming Anthropic API raises "Streaming is required for operations that may take longer than 10 minutes" when max_tokens=64000 with large paper inputs. CLI invocation was also fragile (output truncation, JSON parsing issues from prior session).
- **Alternatives**: Keep CLI with reduced max_tokens (would truncate large extractions); use non-streaming API with lower token budget (loses completeness for large papers); use async streaming (more complex, no benefit for single sequential runs)
- **Consequences**: Requires `ANTHROPIC_API_KEY` env var. Model pinned to `claude-haiku-4-5-20251001`. Real token usage reported from `response.usage` instead of character-based estimates. Output tokens confirmed at 33K–54K per paper (4–7x prior estimates), raising per-paper cost from ~$0.02 estimated to ~$0.15 actual.

## 2026-03-28 — KG Extraction Optimizer: Remove Claim Cap

**Decision**: Removed the "Maximum 50 claims" hard cap from the optimizer user prompt in `Paper Extractor/KnowledgeGraph Extraction/optimize/experiment_runner.py`.

**Rationale**: 7/10 papers in pilot runs were hitting the 50-claim ceiling, artificially suppressing extraction density for dense papers. The v6.2 prompt retains soft "25-50 claims" density guidance. Without the hard cap, dense papers are estimated to yield 80-150 claims, bounded by the 64K output token budget (~200 claims theoretical max). This change applies only to the optimizer; production `batch_extract_kg.py` is unchanged.

## 2026-03-28 — KG Extraction Optimizer: Disable Input Truncation

**Decision**: Disabled input truncation in the optimizer (`_TRUNCATION_LIMIT = 0`) in `experiment_runner.py`.

**Rationale**: Haiku 4.5 has a 200K token context window. Only 3/25 baseline papers exceeded 100K characters, and truncation was minimal (1-10
## 2026-03-28 — KG Extraction Optimizer: Remove Hard Claim Cap

**Decision**: Removed the "Maximum 50 claims" hard cap from the optimizer user prompt in `Paper Extractor/KnowledgeGraph Extraction/optimize/experiment_runner.py`.

**Rationale**: 7/10 papers in pilot runs were hitting the 50-claim ceiling, artificially suppressing extraction density for dense papers. The v6.2 prompt retains soft "25-50 claims" density guidance. Without the hard cap, dense papers are estimated to yield 80-150 claims, bounded by the 64K output token budget (~200 claims theoretical max). This change applies only to the optimizer; production `batch_extract_kg.py` is unchanged.

## 2026-03-28 — KG Extraction Optimizer: Disable Input Truncation

**Decision**: Disabled input truncation in the optimizer (`_TRUNCATION_LIMIT = 0`) in `experiment_runner.py`.

**Rationale**: Haiku 4.5 has a 200K token context window. Only 3/25 baseline papers exceeded 100K characters, and truncation was minimal (1-10%). Feeding full paper text improves extraction completeness for optimizer runs. Production `batch_extract_kg.py` retains its own truncation logic unchanged.

## 2026-03-28 — Evidence Depth Metric & Optimizer Improvements

### evidence_depth metric design (optimize/scoring.py)
- Added `evidence_depth` metric with weight 0.06, taken from `evidence_linkage` (dropped 0.08 → 0.04)
- Scoring: 1 link = 0.0 (floor, not partial credit — 1 is the minimum acceptable), 2 links = 0.5, 3+ links = 1.0
- Unlinked claims get penalty of -0.5 (they are noise, not partial progress)
- `attributed_prior` claims are excluded — their evidence lives in the cited paper, not locally
- Rationale: binary evidence_linkage gave no signal for multi-evidence quality; capping at 3+ links avoids incentivizing spurious links
- Total evidence weight category increased from 25
## 2026-03-28 — Evidence Depth Metric & Optimizer Improvements

### evidence_depth metric design (optimize/scoring.py)
- Added `evidence_depth` metric with weight 0.06, taken from `evidence_linkage` (dropped 0.08 → 0.04)
- Scoring: 1 link = 0.0 (floor, not partial credit — 1 is the minimum acceptable), 2 links = 0.5, 3+ links = 1.0
- Unlinked claims get penalty of -0.5 (they are noise, not partial progress)
- `attributed_prior` claims are excluded — their evidence lives in the cited paper, not locally
- Rationale: binary evidence_linkage gave no signal for multi-evidence quality; capping at 3+ links avoids incentivizing spurious links
- Total evidence weight category increased from 25% to 27%

### Truncation removal (optimize/experiment_runner.py)
- Removed "Maximum 50 claims" hard cap from user prompt
- Disabled input truncation (_TRUNCATION_LIMIT=0): Haiku 4.5 200K context fits all papers
- Rationale: 7/10 papers hit the 50-claim cap, meaning dense papers were silently truncated

### Optimizer prompt update (optimize/program.md)
- Updated agent prompt to include evidence_depth metric so optimizer knows to target multi-evidence linking

## 2026-03-29 — Accept 32K CLI output token limit for KG optimization
- **Decision**: Use `claude -p` CLI output (32K max tokens for Haiku) as the extraction backend for the prompt optimizer rather than the API
- **Rationale**: 32K is sufficient for typical paper extractions (tested: 53 claims/44 evidence on a 36K char paper); switching to the API would require significant refactoring for marginal gain
- **Trade-off**: Occasional truncation on very long papers, but random sampling across 3 papers/iteration provides enough signal; error handling catches failures gracefully

## 2026-03-29 — Do not filter problematic papers from optimizer sample pool
- **Decision**: Keep all papers in the optimizer sample pool (including micro_1 with no full text)
- **Rationale**: Random sampling already dilutes impact of any single bad paper; filtering adds maintenance overhead; error handling manages failures

## 2026-03-29 — Optimizer CLI Approach + Scoring Fix

- **CLI-only optimizer**: Continue using `claude -p` CLI for extractions rather than Anthropic API. API key has no credits; CLI requires no credentials and the 32K output limit is acceptable as an optimization signal.
- **Filter failed extractions from scoring**: `score_all()` must skip entries whose key contains `_error` before computing composite scores. Including failures as 0.0 makes the baseline artificially low and prevents any candidate prompt from being accepted.
- **Rate-limit delay**: Add `time.sleep(2)` between sequential `claude -p` calls in `run_all_extractions()`. Without it, ~20+ rapid calls trigger CLI rate limiting (exit code 1, 1-3s response time).
- **Parallelize per-iteration extractions**: The 3 extractions per optimizer iteration (2 test + 1 full baseline) are independent and should be run with `concurrent.futures.ThreadPoolExecutor` for ~3x speedup. Not yet implemented — planned for next session.

## KG optimizer v8: CLI-only extraction (no API)
- **Date**: 2026-03-29
- **Decision**: Use `claude -p` CLI for both Haiku extractions and Sonnet optimizer agent calls
- **Rationale**: API key has zero credits; Claude subscription provides unlimited CLI usage. Hybrid API+CLI approach was attempted and reverted.
- **Tradeoff**: CLI adds subprocess overhead (~1-2s per call) but is functionally equivalent for batch optimization work

## KG optimizer v8: 3-worker parallel extraction per iteration
- **Date**: 2026-03-29
- **Decision**: Use `ThreadPoolExecutor(max_workers=3)` to parallelize per-paper extractions within each optimizer iteration
- **Rationale**: Each `claude -p` call is an independent subprocess (no shared state); 3 workers gives ~3x speedup (6-8 min vs 18-24 min per iteration) without overwhelming the CLI rate limits
- **Workers=3 chosen**: Conservative — avoids triggering rate limiting while still providing meaningful speedup

## 2026-03-29 — KG Optimizer: Composite-Only Acceptance

**Decision**: Removed per-metric regression guard from `optimize_extraction_prompt.py`. The optimizer now accepts a candidate prompt purely on composite score improvement, with no per-metric floor checks.

**Why**: v8 got stuck at 10 consecutive rejects. Iterations reaching composite ~0.88 were blocked by single-metric regressions (e.g., doi_coverage, citation_contexts) even when the overall weighted score improved. The composite score already weights metrics appropriately — enforcing per-metric floors is double-counting and over-constrains the search.

**Decision**: Added high-water mark (HWM) tracking to the optimizer. The optimizer now records the best composite score and corresponding prompt across ALL iterations (accepted and rejected), and uses the HWM prompt for final evaluation.

**Why**: The best-ever composite may appear in a "rejected" iteration (e.g., composite improved but was followed by a worse iteration that got accepted). Without HWM tracking, that best prompt would be silently discarded. HWM ensures the final evaluation always uses the globally best observed prompt.

**Related**: `QUALITY_REGRESSION_THRESHOLD` kept in code but marked legacy/unused. `MAX_CONSECUTIVE_REJECTS` bumped 10 → 25 for overnight patience.

## 2026-03-29 — Optimizer Diversity Rotation

**Decision**: Added strategy diversity rotation and failed-approach memory to the KG extraction prompt optimizer loop (`optimize_extraction_prompt.py`).

**Why**: The v8 optimizer got stuck repeating the same 3 strategies (quant_context rules, evidence_depth checklists, predicate tightening) across 10+ consecutive rejects. The optimizer had no memory of what it had tried before and no pressure to try different approaches. Without diversity pressure, LLM optimizers converge to a local optimum and cycle through the same surface-level edits.

**What changed**:
- Added `DIVERSITY_STRATEGIES` list of 10 techniques that rotate each iteration: structural_rewrite, example_driven, counter_example, constraint_tightening, checklist_approach, negative_space, cross_metric_synergy, simplification, workflow_reframing, weakest_link_focus
- `call_optimizer()` now injects the current strategy as an "Optimization Lens" hint into the user message
- After 5+ consecutive rejects, the optimizer receives an escalating warning that lists all recently failed approach summaries and explicitly instructs it to try something fundamentally different
- All history entries now include `optimizer_summary` so the optimizer has memory of what strategies were attempted
- `optimize/program.md` updated to tell the optimizer to read `optimizer_summary` in history and honor the lens

## 2026-03-29 — Save all optimizer iterations to disk

**Decision**: Save every scored iteration prompt to `prompt_versions/`, not just accepted ones.

**Why**: During v8 optimization, prompts reaching ~0.88 composite were rejected by the HWM guard and lost permanently. These sub-accepted prompts may contain valuable intermediate improvements or serve as recovery points if later iterations regress.

**Implementation**: `optimize_extraction_prompt.py` now saves a `prompt_versions/vN.M.md` file for every iteration immediately after scoring, before the accept/reject decision. Removed the duplicate `save_version` call in the accept path to avoid double-writes.

## 2026-03-29 — KG Optimizer v10: Rapid Mode Infrastructure

### D029 — Rapid mode for optimizer: aggressive section truncation + short-paper pool
- **Date**: 2026-03-29
- **Decision**: Added `--rapid` mode to the KG extraction prompt optimizer with two components: (1) `_truncate_paper_rapid()` in `experiment_runner.py` that keeps only Results + Methods + References sections with a 20K char hard cap, and (2) `max_text_length=80_000` filter for sample pool selection.
- **Rationale**: Full-paper input (avg 81K chars) costs ~$0.09/iteration and is slow. Rapid mode targets ~$0.03/iteration with ~3x speedup. Quality signal remains high because extraction-relevant content concentrates in Results + Methods.
- **Files**: `experiment_runner.py`, `optimize_extraction_prompt.py`, `batch_extract_kg.py`
- **Alternative considered**: Short-paper curation (`micro_sample_short.json`) — superseded by on-the-fly truncation, which is more flexible.

### D030 — Permanent corpus exclusion constants: `_SKIP_INDICES` and `_SKIP_CORPUS_IDS`
- **Date**: 2026-03-29
- **Decision**: Reviews and abstract-only papers permanently excluded from the optimizer corpus via module-level constants `_SKIP_INDICES = {0, 1, 9}` and `_SKIP_CORPUS_IDS = {"corpus_0"}` in `experiment_runner.py`.
- **Rationale**: These papers cause systematic failures or produce non-representative extractions (review papers: all attributed_prior, no primary empirical signal; abstract-only: too sparse for quality scoring). Permanent exclusion via constants is safer than per-run flags — exclusions always apply regardless of CLI args.
- **Pool size after exclusions**: 10 papers from 173 micro_sample.json candidates.

### D031 — v8.7 as baseline for v10 optimizer (composite 0.8484)
- **Date**: 2026-03-29
- **Decision**: Used `--skip-baseline v8.7_scores.json` (composite 0.8484, 25,146 chars) as the starting baseline for v10, skipping expensive re-evaluation of the known-best prompt.
- **Rationale**: Re-evaluating the baseline on 10 papers would cost ~$0.90 and ~40 min with no new information. The v8.7 scores are stable (3-paper sample, same corpus). `--skip-baseline` flag reads pre-computed scores from disk and uses them directly.
- **Note**: v8.7 is the result of 50+ iterations since the v6.1 baseline (23K chars). Prompt grew by ~9
## 2026-03-29 — KG Optimizer v10: Rapid Mode Infrastructure

### D029 — Rapid mode for optimizer: aggressive section truncation + short-paper pool
- **Date**: 2026-03-29
- **Decision**: Added `--rapid` mode to the KG extraction prompt optimizer with two components: (1) `_truncate_paper_rapid()` in `experiment_runner.py` that keeps only Results + Methods + References sections with a 20K char hard cap, and (2) `max_text_length=80_000` filter for sample pool selection.
- **Rationale**: Full-paper input (avg 81K chars) costs ~$0.09/iteration and is slow. Rapid mode targets ~$0.03/iteration with ~3x speedup. Quality signal remains high because extraction-relevant content concentrates in Results + Methods.
- **Files**: `experiment_runner.py`, `optimize_extraction_prompt.py`, `batch_extract_kg.py`
- **Alternative considered**: Short-paper curation (micro_sample_short.json) — superseded by on-the-fly truncation, which is more flexible.

### D030 — Permanent corpus exclusion constants: _SKIP_INDICES and _SKIP_CORPUS_IDS
- **Date**: 2026-03-29
- **Decision**: Reviews and abstract-only papers permanently excluded from the optimizer corpus via module-level constants _SKIP_INDICES = {0, 1, 9} and _SKIP_CORPUS_IDS = {"corpus_0"} in `experiment_runner.py`.
- **Rationale**: These papers cause systematic failures or produce non-representative extractions (review papers: all attributed_prior, no primary empirical signal; abstract-only: too sparse for quality scoring). Permanent exclusion via constants is safer than per-run flags — exclusions always apply regardless of CLI args.
- **Pool size after exclusions**: 10 papers from 173 micro_sample.json candidates.

### D031 — v8.7 as baseline for v10 optimizer (composite 0.8484)
- **Date**: 2026-03-29
- **Decision**: Used --skip-baseline v8.7_scores.json (composite 0.8484, 25,146 chars) as the starting baseline for v10, skipping expensive re-evaluation of the known-best prompt.
- **Rationale**: Re-evaluating the baseline on 10 papers would cost ~$0.90 and ~40 min with no new information. The v8.7 scores are stable (3-paper sample, same corpus). --skip-baseline flag reads pre-computed scores from disk and uses them directly.
- **Note**: v8.7 is the result of 50+ iterations since the v6.1 baseline (23K chars). Prompt grew by ~9% (to 25K chars) — bloat reduction is deferred to Phase 2.

## D032 — Exclude doi_coverage and citation_contexts from rapid mode composite scoring
**Date:** 2026-03-29
**Context:** v10 optimizer running with --rapid mode; first 5 iterations (v10.1–v10.5) all rejected because doi_coverage cratered from 1.0 to 0.0
**Decision:** Exclude `doi_coverage` and `citation_contexts` from rapid mode composite scoring. Rapid section-aware truncation removes Discussion (source of attributed_prior claims) and may truncate References (DOI source). These metrics hit 0.0 not because extraction quality is bad but because the input lacks the relevant sections. Weights are renormalized so composite stays in [0,1]. Metrics are still computed and returned — just excluded from the weighted composite.
**Alternatives considered:** Adding the sections back (defeats the purpose of rapid mode); penalizing the score (unfair to the optimizer).
**Files:** `Paper Extractor/KnowledgeGraph Extraction/optimize/scoring.py`, `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py`

## D033 — Tournament selection for prompt optimizer (2026-03-29)

**Context**: The v10 optimizer had 5+ consecutive rejections because each iteration tested only 1 optimizer candidate on 3 papers. The single optimizer would fix one metric but break 2-3 others, leading to ~80
## D033 — Tournament selection for prompt optimizer (2026-03-29)

**Context**: The v10 optimizer had 5+ consecutive rejections because each iteration tested only 1 optimizer candidate on 3 papers. The single optimizer would fix one metric but break 2-3 others, leading to ~80% rejection rate.

**Decision**: Replace single-candidate hill climbing with parallel tournament selection. Run N optimizer calls (default 5) with different lenses in parallel, screen all N on 1 (screen) paper cheaply, then validate the best on 3 different (validation) papers. Screen paper is excluded from validation set to prevent overfitting.

**Implementation**: New `run_optimizer_tournament()` function using `ThreadPoolExecutor`; `call_optimizer()` gains `strategy_index: int | None` param; main loop restructured to shuffle papers -> pick 1 screen + 3 validation (non-overlapping) -> tournament -> screen -> validate winner. Added `--candidates N` CLI arg (default 5).

**Trade-offs**: Wall time per iteration increases ~2 min (5->7 min) but exploration increases 5x. Net expected improvement: dramatically higher probability of finding a net-positive edit per iteration.

**File**: `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py` (1147 -> 1332 lines)

## D034 — Tournament selection replaces single-candidate hill climbing (confirmed implementation)
**Date:** 2026-03-29
**Context:** Prior D033 entry described the design; this entry confirms the implementation was completed and the optimizer relaunched.
**Decision:** `run_optimizer_tournament()` is the active code path. The main loop runs N=5 parallel Sonnet optimizer calls, each with a different strategy lens (strategy_index 0-4). All 5 candidates are screened on 1 paper; the best is validated on 3 different papers (non-overlapping with screen paper). Positional argument bug fixed (`run_all_extractions(prompt, [paper], 600, None, rapid)` — `rapid` is now passed as the 5th keyword-style positional arg, not as `max_workers`). Optimizer relaunched as PID 43233.
**Files:** `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py`

## D035 — PENDING: Consolidate 5 Sonnet optimizer calls into 1 multi-candidate call
**Date:** 2026-03-29
**Status:** PENDING — discussed but not yet implemented
**Context:** Each tournament iteration makes 5 separate Sonnet CLI calls (one per lens/strategy). Each call is independent and uses the same context window.
**Proposed decision:** Replace 5 calls with 1 Sonnet call that returns 5 distinct candidate edit sets in a single structured response. Estimated API cost reduction: ~80
## D034 — Tournament selection confirmed implementation + positional arg bug fix
**Date:** 2026-03-29
**Context:** D033 described the tournament selection design. This entry records the confirmed implementation details and a critical bug found during execution.
**Bug fixed:** `run_all_extractions(prompt, [paper], 600, rapid, ...)` was passing `rapid=True` as the 4th positional arg (`max_workers`), causing 5 parallel extraction workers instead of 1. Fixed to `run_all_extractions(prompt, [paper], 600, None, rapid)`. Optimizer relaunched as PID 43233 with correct argument order.
**Decision confirmed:** `run_optimizer_tournament()` is the active code path. N=5 parallel Sonnet optimizer calls, each with a distinct strategy_index (0-4). Screen on 1 paper, validate winner on 3 non-overlapping papers.
**Files:** `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py`

## D035 — PENDING: Consolidate 5 Sonnet optimizer calls into 1 multi-candidate call
**Date:** 2026-03-29
**Status:** PENDING — discussed but not yet implemented
**Context:** Each tournament iteration makes 5 separate Sonnet CLI calls (one per lens/strategy). Each call is independent and uses the same base context.
**Proposed decision:** Replace 5 calls with 1 Sonnet call that returns 5 distinct candidate edit sets in a single structured response. Estimated API cost reduction: ~80% per iteration (5 calls -> 1 call with comparable total tokens).
**Concerns:** Single-call diversity may be lower — the model may anchor on its first candidate when generating subsequent ones. Needs empirical validation before committing.
**Trigger for implementation:** When tournament convergence rate is measured and optimizer API cost is confirmed as the primary bottleneck.
**Files:** `Paper Extractor/KnowledgeGraph Extraction/optimize_extraction_prompt.py`

### v8.7 prompt validated for quality but blocked on completeness (2026-03-29)
- DECISION: v8.7 prompt produces accurate, detailed claims but catastrophically under-extracts (10 vs 34 claims). Cannot use for production until resolved.
- Need to determine if this is a token limit issue, prompt verbosity issue, or Haiku capability issue before choosing fix path
- Audit report at /tmp/rai14_extraction_audit.md

## 2026-03-29 — Claims Graph v2: Condition-Aware Merging

- **Decision:** Merge key changed from (S,P,O) to (S,P,O,condition_signature) for v2 graph construction. v1 preserved as default (`build_graph(version=1)`).
- **Three-tier condition hierarchy:** Hard partition (organism, in_vitro), soft partition (model_system via fuzzy bucketing), scoring modifiers (cell_type, treatment, stage).
- **Rationale:** v1 merged claims across incompatible experimental contexts, causing false-positive contradictions and hiding real ones. v2 keeps context-different claims as parallel edges.
- **Trade-off:** Fuzzy threshold for ModelSystemRegistry set to 65 (not 80 as planned) because rapidfuzz ratio for domain synonyms ("mouse ESC gastruloids" vs "mESC-derived gastruloids") only scores ~71.

### Sonnet recommended for KG extraction over Haiku (2026-03-29)
- DECISION: Sonnet should replace Haiku for KG extraction — dramatically better consistency, completeness, and structural integrity
- Haiku v8.7 prompt produces 3-57 claims depending on paper (unacceptable variance); Sonnet consistently produces 33-39
- Next steps: (1) refactor extraction to use direct API calls instead of CLI, (2) test on wet-lab paper for evidence granularity, (3) evaluate Batch API for cost optimization
- Minor prompt edit made: reinforced panel-level evidence splitting in EVIDENCE RULES section

---

## Decision — 2026-03-29: MRF incremental solving with hop-radius subgraph

- **Context**: Claims Graph v2 adds new edges incrementally as papers are processed
- **Decision**: Added `solve_incremental()` to `HLMRFEngine` — freezes variables outside `hop_radius` hops of affected vars, re-solves only the local subgraph
- **Rationale**: Full re-solve on every new paper is O(N²) in rules; incremental is O(k²) where k = affected neighborhood. hop_radius=2 captures most constraint propagation.
- **Trade-off**: Warm-start from prior solution may not find global optimum if affected subgraph is large. Accept this for online updates; full re-solve available for offline batch.
- **Files**: `autoreview/knowledge_graph/hlmrf.py`, `autoreview/knowledge_graph/mrf_scoring.py`

## 2026-03-29 — MRF Incremental Solve + Diagnostics Architecture

**Decision:** Extract `_ground_rules()` as a private helper shared by `score_graph_mrf` and `update_graph_mrf` rather than duplicating grounding logic.
- **Rationale:** Both full-solve and incremental-update paths need identical variable registration and rule grounding; duplication would drift. A shared helper enforces consistency.
- **Rejected alternative:** Subclass `HLMRFEngine` per use-case — too heavy for what is essentially a procedural pipeline step.

**Decision:** Expose convergence metadata as read-only properties on `HLMRFEngine` (not returned from `solve()`).
- **Rationale:** `solve()` already returns the variable dict; callers needing diagnostics inspect `.last_converged` etc. separately. Avoids breaking the return-type contract for all existing callers.

**Decision:** `MRFDiagnostics` stored on `MRFResult.diagnostics` (nullable) rather than a separate return value.
- **Rationale:** Keeps the public API surface stable — callers that do not need diagnostics are unaffected. `None` for empty graphs.

**Decision:** `composition_decay` default = 0.7 and `max_composition_hops` default = 3 chosen empirically for typical 2–4 hop KG chains; expose in `MRFConfig` for tuning.

### Direct API calls required for extraction pipeline (2026-03-29)
- DECISION: Must refactor extraction to use Anthropic Python SDK (`client.messages.create()`) instead of CLI or subagent approaches
- CLI: no max_tokens control, Sonnet timeouts at 900s, max-turns=1 conflicts
- Subagent: 23 min per paper, hits 32K output token limit, massive context overhead
- Direct API: single request-response, explicit max_tokens=64000, 2-3 min expected, full diagnostics (stop_reason, usage)

## 2026-03-29 — Post-Extraction Claim Normalization Layer

- **Two-phase normalization (pre_dedup / post_dedup)**: Chose to split normalization into a pre-dedup pass (text cleaning + decomposition) and a post-dedup pass (quantitative backfill). Text cleaning must happen before entity dedup so that canonical name matching works on clean strings; quantitative backfill does not affect entity resolution and is safely deferred.

- **Hybrid decomposition (rule-based + LLM fallback)**: Rule-based patterns run first; LLM fallback triggers only for object strings longer than 6 words. Balances determinism and auditability with coverage for complex, free-form claims.

- **ClaimNormalizer accepts both entities and assertions**: Deviation from the original spec (which described operating on claims only). Necessary because `entity.canonical_name` and `assertion.subject_canonical_name` / `assertion.object_canonical_name` must stay in sync; normalizing only one side would leave dangling references.

- **Cascade order for decomposition — slash → prepositional → conjunction**: Prepositional must strip its wrapper phrase before conjunction splitting can correctly parse the inner content. Slash splitting is applied first as it is the most unambiguous delimiter.

- **normalize=False default, independent of graph version**: The flag is opt-in and does not gate on KG schema version. Keeps the change fully backwards-compatible; callers must explicitly pass normalize=True to activate the layer.

### Set CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000 for all extractions (2026-03-30)
- DECISION: Use Haiku with CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000 as the production extraction configuration
- This resolves all truncation issues discovered on 2026-03-29
- Haiku outperforms Sonnet on completeness (96% vs 79%), speed (3.5x faster), cost (~10x cheaper), and structural integrity (0 vs 5 orphaned evidence)
- Sonnet advantages (certainty calibration, less redundancy) are minor and addressable via prompt tweaks
- Previous decisions about Sonnet being required and CLI being unviable are SUPERSEDED by this finding
- The env var must be set in experiment_runner.py or batch_extract_kg.py for all extraction runs

## 2026-03-29 — Bayesian Inference Architecture & Ground Truth Strategy

### Bayesian model will REPLACE HL-MRF (not layer on top)
- **Decision**: Once validated, the Bayesian model becomes the primary inference engine; HL-MRF is deprecated.
- **Rationale**: Maintaining two parallel inference paths long-term adds complexity with no benefit. The HL-MRF serves as the baseline to beat during validation, then is retired.
- **Approach B chosen**: Parallel model with shared interface (NumPyro/JAX), not a layer on top of HL-MRF.

### NumPyro/JAX as sampling framework
- **Decision**: NumPyro/JAX selected over PyMC/Stan for the Bayesian inference model.
- **Rationale**: Better GPU utilization, JAX JIT compilation for large KGs, active biomedical NLP ecosystem.

### Ground truth curation strategy: hybrid tiered approach
- **Decision**: Database anchors (Reactome/KEGG matching) + rank-order validation + expert spot-check of high `contradiction_centrality` edges.
- **Rationale**: No single ground truth standard exists for biomedical KGs. Tiered approach mirrors literature practice (STRING, PharmGKB, CTD, GO).
- **`contradiction_centrality` as triage tool**: High-centrality edges are first candidates for human review — flags both extraction noise and genuine scientific disputes.

### Ground truth curation will be agent-directed
- **Decision**: Automated database matching + targeted human judgment. Not purely manual.
- **Rationale**: Scale of ~1000-paper corpus makes manual review infeasible; agent narrows to highest-value spots.

### Build corpus-agnostic interfaces, validate on gastruloid corpus first
- **Decision**: Interfaces designed generically; first validation target is the ~1000-paper gastruloid corpus (pending extraction).
- **Rationale**: Gastruloid corpus is the immediate use case and has an existing extraction pipeline.

### Migration path: 5-step Laplace → MCMC
- Identified migration path: (1) Laplace approximation, (2) flat NumPyro factor graph, (3) ground truth curation, (4) hierarchical priors, (5) integration.
- **Status**: Brainstorming complete; spec authorship pending user approval.

### 2026-03-30 — KG prompt v11.0: interpretive claims + negative results + certainty calibration
- Added 4 targeted prompt additions to v8.7 baseline, saved as v11.0
- Negative results: extract mechanisms tested and ruled out (e.g., YAP/TAZ)
- Interpretive claims: minimum 3-5 from Discussion for meta-level contradiction resolution
- Computational certainty: model predictions default to "medium", Discussion extrapolations to "low"
- Validated composite 0.8253 (no regression from 0.8241), quant_context improved 0.35→0.49
- ISSUE: primary_empirical claims dropped 53→27 on Etoc paper (collateral damage pattern). Needs density floor adjustment.

### 2026-03-30 — Graph contradiction: B+C approach (interpretive claims + graph clustering)
- Do NOT add extraction-time "findings" layer; author framing should not influence ground truth
- (B) Interpretive claims serve as natural cluster anchors for meta-level reasoning
- (C) Graph-time clustering by (subject, predicate_family, object) tuples in new cluster.py module
- Implementation not yet done — needs next session

### 2026-03-30 — CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000 set in experiment_runner.py
- Root cause of ALL extraction truncation across multiple sessions
- Added env var to subprocess.run() call in run_extraction()

### 2026-03-30 — Meta-contradiction structure: graph-time hierarchical clustering (not extraction-time schema)
- **Decision**: Do NOT add extraction-time "findings" or "meta-claim" schema layer for contradiction detection
- **Rationale**: Author framing at extraction time poisons ground truth; meta-level structure should emerge from claim relationships, not be injected
- **Chosen approach**: Graph-time hierarchical clustering by (subject, predicate_family, object) tuples in a new `cluster.py` module
- Interpretive claims (added in v11.0) serve as natural cluster anchors — they are still ground-level claims, not meta-claims
- Implementation target: `autoreview/knowledge_graph/cluster.py` with `build_claim_clusters()` function
- Status: design finalized, implementation pending

### 2026-03-30 — v11.1 set as standard KG extraction prompt
- v11.1 adds module independence controls and density floor (40-70) over v11.0
- Recovered primary_empirical claims from 27→44 while keeping interpretive/negative/certainty improvements
- Full composite 0.8465 (up from 0.8241 v8.7 baseline)
- Density scoring metric updated: sweet spot 30-70 (was 25-50)
- Opus audit confirms 83% of major findings captured, 0 critical gaps

### 2026-03-30 — Meta-level contradiction: graph-time hierarchical clustering
- **Decision**: Built TopicCluster → Finding → FindingContradiction hierarchy algorithmically from graph structure (no extraction schema changes)
- **Why**: Field-level scientific contradictions operate at a higher level than individual claim edges. Two papers may have individually-consistent granular claims but reach opposite conclusions.
- **Approach**: Predicate class collapse table groups related predicates (e.g., induces + is_sufficient_for → activating). Findings partition by (direction, organism_class, in_vitro). Three contradiction types: directional, boundary, interpretive.
- **HL-MRF integration**: Finding-level truth variables with aggregation (edge→finding), finding contradiction (finding↔finding, weight=12.0), downward propagation (finding→edge, weight=3.0). Toggled via enable_finding_layer flag.
- **Literature basis**: SemMedDB predication grouping, PSL/HL-MRF cluster variables, GraphRAG community detection.

## 2026-03-30: Bayesian Inference Package — Key Architectural Decisions

**Context**: Implementing Phase 1 Bayesian upgrade for the knowledge graph system (bayesian/ package, 6 modules).

**Decisions**:
1. **Alpha/beta derivation from confidence_mean + evidence_count** — score_all_edges() only stores confidence_mean (not full alpha/beta). Derived via kappa=2+evidence_count. Known approximation; future improvement: store exact alpha/beta in score_all_edges().
2. **Softplus instead of hinge loss** — jax.nn.softplus(x) replaces max(0,x) in factor graph for smooth HMC gradient flow. Hinge loss produces non-differentiable kinks that break NUTS sampling.
3. **Laplace + targeted NUTS hybrid** — full-graph Laplace approximation is fast (~1s via scipy L-BFGS-B MAP + JAX Hessian); NUTS MCMC runs only on contradiction-centrality hotspot subgraphs extracted via BFS. Avoids O(n) NUTS cost on large graphs.
4. **JAX float32 mode** — stayed with JAX default float32 for Hessian computation. float64 overhead unjustified for [0,1]-bounded confidence variables where float32 precision is adequate.
5. **BFS composition chain discovery mirrors mrf_scoring._ground_rules** — bayesian/model.py replicates the BFS logic from HL-MRF to ensure composition factors are consistent across both inference backends.
6. **HL-MRF system left unchanged** — bayesian/ is a parallel backend; build_graph(bayesian=True) activates it. Existing MRF pipeline untouched.

**Alternatives considered**:
- Full NUTS on entire graph (rejected: too slow, O(n) cost)
- Hinge loss for contradiction factors (rejected: non-differentiable, breaks HMC)
- Storing exact alpha/beta in score_all_edges() (deferred: requires schema change, noted as future improvement)

**Consequences**: bayesian_confidence, bayesian_ci_low, bayesian_ci_high, bayesian_bimodal written as edge attributes. New optional dependency group: jax, jaxlib, numpyro, arviz, diptest.

### 2026-03-30 — v0.4.0 docs: README, CHANGELOG, ARCHITECTURE
- **Decision**: Updated all three docs to reflect the finding layer and other v0.3→v0.4 additions (Bayesian inference, normalization, weight learning)
- **Why**: The finding layer is a major architectural addition that changes how contradictions are detected and resolved. Users and contributors need to understand the hierarchy.

### 2026-03-30 — v11.2/v11.2.1: Testing proportional density guidance
- **Context**: v11.1 had fixed "40-70 claims, 30+ primary_empirical" floor. User concerned this causes overextraction on small papers and underextraction on large ones.
- **v11.2 attempt**: Replaced with soft proportional language ("scale with paper content, 25-40 for 4-5 figures"). Result: Haiku under-extracted on medium (49 vs 62 claims) and large (50 claims for 117K paper). Too permissive.
- **v11.2.1 attempt**: Replaced with structural anchors ("2 claims per main figure panel, 1 per supp panel"). Testing in progress — early results show quality recovered (adj 0.9284 vs 0.9271 baseline) but claim counts still lower (42 vs 62 for medium).
- **Status**: Awaiting large paper results before deciding. The tension is between paper-agnostic guidance and preventing Haiku attention reallocation.

### 2026-03-30 — v11.2.1 set as new KG extraction standard
- **Decision**: Replace v11.1 (fixed 40-70 density floor) with v11.2.1 (per-figure-panel anchors + size-aware density scorer)
- **Rationale**: Fixed floors do not generalize across paper sizes. v11.2.1 tested on small/medium/large papers with adj composites 0.883/0.928/0.983. Opus audit confirms 96% finding coverage on medium paper. Scorer density metric now scales with paper_chars.
- **Supersedes**: v11.1 standard (2026-03-30 earlier)

---
## 2026-03-30 — KG Extraction Prompt v11.3: Two-Pass Citation Architecture

**Context**: corpus_10 (117K chars, large paper) extracted 0 attributed_prior claims despite ~45 citations in Introduction/Discussion. Model created 11 citation_contexts entries but skipped the corresponding attributed_prior claims entirely.

**Root cause**: "Prioritize Results completeness first — before moving to Discussion or Introduction" language caused Haiku to deprioritize citation extraction. When it did read Discussion (9 interpretive claims), it produced citation_context metadata without generating the paired attributed_prior claim.

**Decision**: Shipped v11.3 with three targeted prompt changes:
1. ROLE section — replaced vague attributed_prior instruction with explicit triple-output requirement: every cited claim must produce (claim, evidence stub, citation_context) together, with a concrete numbered-citation example [N]
2. Replaced "prioritize Results first" with an explicit two-pass instruction: Pass 1 = Results (completeness), Pass 2 = Citations & Interpretation (mandatory, not optional)
3. Quality checklist — added self-check #10 (count citations vs attributed_prior claims; go back if under-extracting) and bidirectional constraint #12 (every attributed_prior ↔ citation_context)

**Alternatives rejected**: Increasing token budget (output was not truncated — issue was priority, not capacity); adding more examples of Results claims (misdiagnoses the problem).

**Status**: v11.3 round 1 improved medium papers (12→16 attributed_prior) but large still returned 0. Round 2 with stronger language in progress.

---
## Decision: v11.3 prompt changes reverted — two-pass framing catastrophically broke extraction
Date: 2026-03-30

- v11.3 prompt changes (two-pass instruction, stronger attributed_prior language) REVERTED — they catastrophically broke extraction on large papers (96 claims → 2 claims). The "two-pass" framing confused Haiku on long papers.
- Decision: implement citation extraction as a SEPARATE SECOND PASS rather than modifying the main prompt. Architecture: main v11.2.1 extraction runs first, then `needs_citation_supplement()` checks if attributed_prior == 0 AND paper has >5 citations. If triggered, a focused citation-only prompt runs on just Intro/Discussion/References sections. Results merged into the main extraction.
- Key insight: single-prompt solutions for citation extraction do not work on large papers. The model allocates all attention to Results and never processes citations regardless of prompt language. Separating the concern into a dedicated pass is the correct architecture.

## Citation supplement pass architecture (2026-03-30)
- Citation supplement pass integrated into batch_extract_kg.py via `--supplement` flag. Architecture: main batch runs first, then `--supplement` scans cached extractions, identifies papers with 0 attributed_prior AND >5 citations in Intro/Discussion, submits a second batch with focused citation-only prompt on just Intro/Discussion/References sections, merges results back into cache. Uses same Anthropic batch API (50% discount). Supplement custom_ids use `{phash}_sup` suffix.

---
## 2026-03-31 — CI Infrastructure Decisions

- **Excluded `Paper Extractor/` from ruff linting** via `exclude` in pyproject.toml. Rationale: these are research/exploration scripts, not library code. Applying library-grade lint standards (580 errors) would obscure signal and impede rapid iteration. If any Paper Extractor code graduates to the main library, it should be moved into `autoreview/` and become subject to linting at that point.
- **Added mypy `[[tool.mypy.overrides]]` for numpyro, arviz, diptest, igraph** with `ignore_missing_imports = true`. These are scientific computing packages without PEP 561 type stubs. Overrides are scoped per-package to avoid masking errors in packages that do have stubs.

---
## 2026-03-30 — EZproxy Retrieval as Standalone Script (Not Pipeline-Integrated)

**Decision:** Built `ezproxy_retrieve.py` as a standalone corpus-curation script rather than integrating it into AutoReview's `FullTextResolver`.

**Rationale:**
- EZproxy requires VPN connectivity and Penn institutional credentials — environment-specific dependencies that are not portable
- The general AutoReview pipeline is designed to run in any environment (local, PARCC cluster, CI); baking in EZproxy would break pipeline runs outside Penn network
- EZproxy retrieval is a one-time corpus-preparation step, not a per-paper runtime need
- Standalone script keeps the concern isolated: corpus curators run it manually with VPN active, then ingest results into the corpus directory

**Trade-off:** Future corpus expansions on non-Penn systems will not have access to this retrieval path. Acceptable because ~97 remaining inaccessible papers are mostly tangential, and open-access fetchers (bioRxiv TDM, Europe PMC, Unpaywall) cover the majority of priority papers without institutional access.

**Alternative considered:** Adding an optional `--ezproxy` flag to `FullTextResolver` with graceful skip when VPN is unavailable. Rejected because it adds complexity for a rarely-exercised code path.
