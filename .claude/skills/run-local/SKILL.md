---
name: run-local
description: Use when the user asks to run the AutoReview pipeline locally, in-terminal, or without API calls
---

# Run Local

## Overview

"Run locally" means Claude Code acts as the LLM directly in the terminal session, rather than making calls through the Anthropic API. Real search sources (PubMed, Semantic Scholar, OpenAlex) are used for paper discovery, but all LLM reasoning/synthesis is performed by Claude Code in-terminal.

## Key Rules

- **Claude Code IS the LLM** — do NOT run `python run_local.py` (that requires `ANTHROPIC_API_KEY`). Instead, execute each pipeline stage manually in the chat session.
- **Always use real search sources** (PubMed, Semantic Scholar, OpenAlex) — never fake/synthetic papers
- **Run ALL 15 pipeline stages** — do not skip any. See the complete checklist below.
- **Perplexity** is often disabled (no API key); degrade gracefully if unavailable

## The 15 Pipeline Stages (MANDATORY — run every one)

Execute these in order. Each stage depends on the previous stage's output. Check off each stage as you complete it.

### Phase 1: Literature Discovery

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 1 | **query_expansion** | Generate Boolean queries (PubMed), semantic queries (S2), and NL questions. Also produce a scope document defining boundaries and expected sub-topics. | Output: queries per source + scope document |
| 2 | **search** | Run real searches against PubMed, Semantic Scholar, and OpenAlex concurrently. Deduplicate by DOI. Target: 200-500 raw candidates. | Output: `candidate_papers` list |
| 3 | **screening** | Score each abstract for relevance (1-5). Apply threshold (default 3). Target working corpus: 50-200 papers. | Output: `screened_papers` list |
| 4 | **full_text_retrieval** | Attempt to retrieve open-access full text for screened papers. Strategies: S2 openAccessPdf, PubMed Central JATS XML, arXiv/bioRxiv/medRxiv PDFs, Unpaywall. Even if no full texts are found, this stage must run. | Output: papers enriched with `full_text` field where available |
| 5 | **extraction** | Extract structured data from each paper (full text if available, else abstract). Populate `PaperExtraction` model: key findings, methods, limitations, relationships. | Output: `extractions` list |

### Phase 2: Analysis

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 6 | **clustering** | Cluster findings into themes/sub-themes. Identify consensus claims, contradictions, and gaps. | Output: `evidence_map` with themes, contradictions, gaps |
| 7 | **gap_search** | If gaps were identified in clustering, generate targeted queries and search for papers to fill them. Re-screen and extract new papers. Skip search only if no gaps exist AND coverage score >= threshold. | Output: additional papers + updated evidence map |

### Phase 3: Planning

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 8 | **outline** | Generate a hierarchical section outline from the evidence map. Run outline critique loop (up to 2 revisions). | Output: `outline` with sections, sub-sections, paper assignments |
| 9 | **narrative_planning** | Plan the narrative architecture: arc, transitions, emphasis per section. Uses `NarrativeArchitect`. | Output: `narrative_plan` with section directives |

### Phase 4: Enrichment

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 10 | **contextual_enrichment** | For each outline section, generate enrichment queries to find adjacent/contextual material. Search, screen (threshold=2), and extract top 5 papers per section. | Output: `contextual_enrichment` dict (section_id → SectionEnrichment) |
| 11 | **corpus_expansion** | Using insights from contextual enrichment (key concepts, cross-field connections), generate targeted queries to expand the primary research corpus. Search, screen, extract, and integrate. | Output: `corpus_expansion_result` with new papers integrated |

### Phase 5: Writing

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 12 | **section_writing** | Write each section with: assigned findings, full outline context, narrative directives, contextual enrichment. Synthesize across papers — do NOT summarize paper-by-paper. Run per-section critique loop (up to 2 revisions). | Output: `section_drafts` dict |
| 13 | **passage_search** | Mine written sections for undercited claims. Generate targeted queries for high/medium priority claims. Search for supporting papers. Also performs citation snowballing from top-cited papers in corpus. | Output: additional supporting papers, updated sections |

### Phase 6: Assembly & Polish

| # | Node | What to do | Key inputs/outputs |
|---|------|------------|--------------------|
| 14 | **assembly** | Assemble all sections into a full draft. Run holistic critique loop evaluating narrative arc, redundancy, transitions, balance (up to 3 revisions or convergence). | Output: `full_draft` |
| 15 | **final_polish** | Language polishing, terminology consistency, citation formatting (Vancouver numbered refs), bibliography assembly. Save to `output/review.md`. | Output: final formatted document |

## Running Searches

Use Python scripts via Bash to run real searches. Example:

```python
python3 -c "
import asyncio
from autoreview.search.pubmed import PubMedSearch
async def run():
    s = PubMedSearch()
    papers = await s.search(queries=['your query here'], max_results=50)
    for p in papers:
        print(f'{p.title} | {p.year} | {p.doi}')
asyncio.run(run())
"
```

Alternatively, if the MCP server is running, use the `search_pubmed`, `search_semantic_scholar`, and `search_openalex` tools directly.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running `python run_local.py` | Don't — that requires ANTHROPIC_API_KEY. Execute stages manually in chat |
| Skipping stages | Run ALL 15 stages. Consult the checklist above |
| Following CLAUDE.md's simplified 8-step description | Use the 15-node DAG from `pipeline/runner.py`, not the high-level overview |
| Using StubLLMProvider | Not applicable — Claude Code IS the LLM |
| Adding synthetic/fake papers | Remove all mock data; use real search sources only |
| Extracting only from abstracts | Run `full_text_retrieval` (stage 4) before extraction to get full texts where available |
| Skipping narrative_planning | Must run before section_writing — it provides section directives |
| Skipping contextual_enrichment / corpus_expansion | These enrich sections with adjacent material; must run before writing |
| Skipping passage_search | Must run after section_writing to ground undercited claims |
