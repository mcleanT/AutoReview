# Changelog

## v0.4.0 — Project Split

### Knowledge Graph Split
- Knowledge graph pipeline, Bayesian inference, MRF scoring, and claim extraction moved to standalone **Scientific Claims Knowledge Graph** project
- Removed `autoreview/knowledge_graph/`, `tests/test_knowledge_graph/`, `Paper Extractor/KnowledgeGraph Extraction/`
- Removed `nli-score` and `nli-diagnose` CLI commands
- Removed KG dependencies: `networkx`, `sentence-transformers`, `jax`, `numpyro`, `arviz`, `diptest`

## v0.3.0 — Extraction & Pipeline Quality

### Hybrid Extraction
- Programmatic zero-token paper extractor: deterministic extraction with no LLM calls
- `HybridExtractor` combining programmatic baseline with LLM refinement
- Dual-layer scoring with alpha-blended composite scores and per-field factual accuracy (number/entity extraction)
- Grounding verification to prevent hallucinated numbers in LLM refinement pass
- Direct-Haiku as default extraction strategy

### Benchmarking
- Dual-layer scoring exposed via `--strategy` and `--alpha` flags on `autoreview benchmark`
- Qwen 3.5 35B local model benchmarking support
- Manifest-based benchmark paper selection

### Pipeline Quality
- Publication-quality visual generation: `SchematicEngine` figures and markdown tables (retrieval, domains, evaluation, takeaways)
- Visual integration: automatic figure/table insertion in assembly with visual audit pass
- ARISE quality directives injected into outline and narrative planning prompts
- Citation snowballing wired into the screening stage
- Contradiction resolver wired into clustering
- Token budget monitoring with graceful degradation when limits are approached
- Pre-flight validation checks before pipeline launch
- Depth-dependent quality thresholds in critique rubrics
- Pipeline consolidation: visual node count reduced from 20 to 17

### Infrastructure
- CI lint and mypy fixes across the codebase
- Ruff format cleanup on all new modules

## v0.2.0 — Review Depth Control

### New Feature: Depth Levels
- Three-level depth control (`--depth low|medium|deep`) for review generation
- `low`: concise, critical-findings-only reviews (~4,000 words)
- `medium`: standard academic review depth (~8,000 words, default)
- `deep`: exhaustive, book-chapter depth with full evidence chains (~25,000+ words)

### Evidence-Weighted Word Allocation
- `EvidenceWeightedAllocator` distributes word budgets across sections proportionally based on evidence density (paper count + findings + evidence chains)
- Section-type dampening: Introduction, Conclusion, and Methods sections automatically receive reduced allocation relative to body sections
- Zero-evidence sections (e.g., Future Directions) get fixed depth-scaled allocations
- Word budget is a soft target — minimum section word counts are never violated

### Pipeline Integration
- Depth flows through three injection points: outline descriptions, narrative directive insight counts (2-3 / 3-5 / 7-10), and section writing prompts
- `max_tokens` automatically increased to 16384 for deep-mode section writing
- `--depth` flag available on both `run` and `resume` commands
- Resume with changed depth emits a warning when outline word counts can't be recalculated

### Infrastructure
- Year filtering via `--date-range` on all search sources (Phase 1.5)
- Bibliography injection tooling for evaluation (Phase 1.5)
- ARISE rubric, batch evaluation, cost analysis, and structural metrics
- Full codebase lint/format cleanup (ruff + mypy)

## v0.1.0 — Initial Public Release

### Pipeline
- 15-node DAG pipeline: query expansion, multi-source search, screening, full-text retrieval, extraction, thematic clustering, gap-aware search, outline generation, narrative planning, contextual enrichment, corpus expansion, section writing, passage search, assembly, and final polish
- Crash recovery via JSON snapshots after every DAG node
- Pipeline resume from any saved snapshot
- Remediation dispatcher with expand, retry, and threshold actions for adaptive search

### Literature Search
- PubMed (NCBI Entrez), Semantic Scholar, OpenAlex, and Perplexity Sonar integration
- Full-text retrieval via Unpaywall, Elsevier, and Springer APIs
- LLM-driven query expansion with domain-appropriate Boolean and semantic queries

### Extraction & Analysis
- Structured extraction of findings, methods, relationships, and limitations per paper
- Thematic clustering with contradiction detection and consensus identification
- Evidence chain tracing across papers
- Gap detection against scope document
- Comprehensiveness checks with 5 checker classes

### Writing & Critique
- Three-level self-critique: outline, per-section, and holistic
- Configurable critique rubrics with dimension weights
- Narrative architecture planning for coherent review structure
- Contextual enrichment with cross-section awareness
- Citation validation

### Evaluation
- Evaluate generated reviews against published reference PDFs
- Citation recall, synthesis depth, topical coverage, and writing quality metrics

### Infrastructure
- Domain configuration via YAML (ships with biomedical, cs_ai, chemistry presets)
- LLM provider abstraction with Claude and Ollama support
- MCP server exposing search tools for Claude Code integration
- CLI commands: `run`, `resume`, `inspect`, `evaluate`, `benchmark`
- Batched extraction with progress tracking
