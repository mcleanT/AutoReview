# Changelog

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
