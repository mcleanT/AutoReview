# Changelog

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
