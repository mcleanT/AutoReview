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
