---
name: run-local
description: >
  Run the AutoReview pipeline locally using Claude Code subagents as the LLM backend.
  Use when the user says "run locally", "run the pipeline", "run in this terminal",
  "no API calls", "run without API", or wants to execute a pipeline run. MUST present
  the configuration interview BEFORE dispatching any pipeline work. This is the SINGLE
  SOURCE OF TRUTH for local pipeline execution — every stage listed here is mandatory
  unless the user explicitly skips it. Always use this skill whenever a pipeline
  execution is requested, even for partial or resumed runs.
---

# Run AutoReview Locally — Execution Contract

This document is a **hard execution contract**. Every stage listed here MUST execute
unless the user explicitly says otherwise. The #1 recurring failure is silently skipping
stages because they weren't mentioned prominently enough — this document fixes that by
making every stage explicit with validation gates.

## Core Concept

"Run locally" means Claude Code itself acts as the LLM for each pipeline stage:
1. Read the stage description and upstream outputs
2. Dispatch a subagent to produce the stage's output
3. **Verify the output against the stage's validation gate**
4. Only then proceed to the next stage

Claude Code IS the LLM — do NOT run `autoreview run` or `python -m autoreview`
(those require `ANTHROPIC_API_KEY`). Each stage executes via subagent dispatch.

---

## MANDATORY: Configuration Interview

Never start execution without completing this interview. Present all questions together
and let the user respond. Use defaults for anything not explicitly overridden.

### Step 1 — Topic & Output

| Setting | Question | Default |
|---------|----------|---------|
| **Topic** | What is the research topic / question? | *(required)* |
| **Output dir** | Where should outputs be saved? | `output/` |
| **Output format** | markdown, latex, or docx? | `markdown` |

### Step 2 — Domain & Search

| Setting | Question | Default |
|---------|----------|---------|
| **Domain** | biomedical, cs_ai, chemistry, or general? | `general` |
| **Date range** | Year filter for search (e.g., `2015-2024`, `-2020`, `2022-`)? | Domain default |
| **Fresh run** | Clear previous snapshots before starting? | No |

### Step 3 — Depth & Quality

| Setting | Question | Default |
|---------|----------|---------|
| **Depth** | `low` (~4K words), `medium` (~8K), or `deep` (~25K+)? | `medium` |
| **Max critique cycles** | How many outline/section revision rounds? | 2 (outline), 3 (section) |

### Step 4 — Model & Execution

| Setting | Question | Default |
|---------|----------|---------|
| **Subagent model** | Which model for subagent work? | `sonnet` |
| **Start from** | Resume from a specific stage? (or run full pipeline) | Full pipeline |
| **Verbose** | Enable detailed logging? | No |

### Step 5 — Confirm & Launch

Present a summary table of all settings. Wait for user confirmation before proceeding.

```
┌─────────────────────────────────────────────┐
│         AutoReview Local Run Config         │
├──────────────┬──────────────────────────────┤
│ Topic        │ {topic}                      │
│ Domain       │ {domain}                     │
│ Depth        │ {depth}                      │
│ Date range   │ {date_range}                 │
│ Output       │ {output_dir} ({format})      │
│ Model        │ {model}                      │
│ Start from   │ {start_from or "beginning"}  │
│ Fresh        │ {yes/no}                     │
│ Verbose      │ {yes/no}                     │
└──────────────┴──────────────────────────────┘
```

---

## COMPLETE Pipeline Architecture (15 Stages)

Every stage below is MANDATORY. Do not skip any unless the user explicitly says so.
Stages are grouped into 6 phases. Within each phase, stages run sequentially.

```
Phase 1: Literature Discovery
  ● 1. query_expansion
  ● 2. search           (uses MCP tools for real searches)
  ● 3. screening
  ● 4. full_text_retrieval
  ● 5. extraction

Phase 2: Analysis
  ● 6. clustering
  ● 7. gap_search        (conditional — skips search if no gaps AND coverage ≥ threshold)

Phase 3: Planning
  ● 8. outline           (+ outline critique loop, + EvidenceWeightedAllocator for depth)
  ● 9. narrative_planning

Phase 4: Enrichment
  ● 10. contextual_enrichment
  ● 11. corpus_expansion

Phase 5: Writing
  ● 12. section_writing   (+ per-section critique loop)
  ● 13. passage_search

Phase 6: Assembly & Polish
  ● 14. assembly          (+ holistic critique loop)
  ● 15. final_polish
```

---

## Stage-by-Stage Execution Protocol

Each stage includes: what to do, what model to use, available tools, and a
**validation gate** that must pass before proceeding.

### Stage 1: Query Expansion

**Model**: sonnet
**Tools**: None (pure generation)

**What to do**: Given the topic, generate:
- 3-5 Boolean queries for PubMed
- 3-5 semantic queries for Semantic Scholar / OpenAlex
- A scope document defining: review boundaries, expected sub-topics, exclusion criteria

**Validation gate**:
- ≥3 PubMed queries generated
- ≥3 semantic queries generated
- Scope document present with ≥3 expected sub-topics

**Output**: Save queries + scope document to `{output_dir}/snapshots/query_expansion.json`

---

### Stage 2: Search

**Model**: sonnet
**Tools**: `search_pubmed`, `search_semantic_scholar`, `search_openalex` (MCP tools)

**What to do**: Execute queries from Stage 1 against real search sources. Deduplicate
results by DOI. Target: 200-500 raw candidates.

**Protocol**:
1. Call each MCP search tool with the appropriate queries
2. Merge results, dedup by DOI (keep first seen)
3. Log source counts

**Validation gate**:
- ≥50 unique candidates found (warn if <100)
- At least 2 search sources returned results
- No synthetic/fake papers — all must have real DOIs or PMIDs

**Output**: `candidate_papers` list saved to snapshot

---

### Stage 3: Screening

**Model**: sonnet
**Tools**: None (LLM scoring)

**What to do**: Score each candidate's abstract for relevance (1-5) against the scope
document. Apply threshold (default: 3). Process in batches of 20.

**Validation gate**:
- Every candidate has a relevance score
- ≥30 papers pass threshold (warn if <50)
- Scores are integers 1-5 with rationale

**Output**: `screened_papers` list with relevance scores

---

### Stage 4: Full-Text Retrieval

**Model**: haiku (lightweight)
**Tools**: Bash (for HTTP requests to open-access sources)

**What to do**: Attempt to retrieve full text for screened papers. Strategies:
- Semantic Scholar `openAccessPdf`
- PubMed Central JATS XML
- arXiv/bioRxiv/medRxiv PDFs
- Unpaywall API

Even if no full texts are found, this stage must run and report results.

**Validation gate**:
- Stage executed (even if 0 full texts retrieved)
- Full-text retrieval percentage logged

**Output**: Papers enriched with `full_text` field where available

---

### Stage 5: Extraction

**Model**: sonnet
**Tools**: None (structured generation)

**What to do**: For each screened paper (full text if available, else abstract), extract:
- Key findings (claim, evidence strength, quantitative results)
- Methods summary
- Limitations
- Relationships to other papers
- Domain-specific fields (per domain config)

Process papers in batches of 5-10 to manage context.

**Validation gate**:
- Every screened paper has an extraction
- Each extraction has ≥1 key finding
- Evidence strength is one of: strong, moderate, weak, preliminary

**Output**: `extractions` dict (paper_id → PaperExtraction)

---

### Stage 6: Clustering

**Model**: sonnet (or opus for complex topics)
**Tools**: None

**What to do**: Analyze all extractions to produce an EvidenceMap:
- Thematic clusters with sub-themes
- Consensus claims (supported by ≥3 papers)
- Contradictions (with possible explanations)
- Evidence chains (linked findings across papers)
- Gaps (expected sub-topics from scope document with insufficient evidence)

**Validation gate**:
- ≥3 themes identified
- Gaps list present (may be empty)
- Each theme has ≥2 papers assigned

**Output**: `evidence_map` with themes, contradictions, gaps, evidence chains

---

### Stage 7: Gap Search

**Model**: sonnet
**Tools**: `search_pubmed`, `search_semantic_scholar`, `search_openalex`

**What to do**: If gaps were identified in clustering, generate targeted queries and
search for papers to fill them. Screen and extract new papers. Integrate into evidence map.

Skip search ONLY if: no gaps exist AND coverage ≥ threshold. The stage itself must still
run (to evaluate whether search is needed).

**Validation gate**:
- Stage executed
- If gaps existed: ≥1 targeted search performed
- Updated evidence map produced

**Output**: Updated evidence map with any new papers integrated

---

### Stage 8: Outline

**Model**: sonnet (generation), haiku (critique)
**Tools**: None

**What to do**:
1. Generate hierarchical outline from evidence map + scope document
2. Apply depth-appropriate section descriptions:
   - `low`: emphasize critical takeaways
   - `medium`: standard descriptions
   - `deep`: request exhaustive coverage
3. Run outline critique loop (up to max_critique_cycles revisions)
4. After outline finalized, run `EvidenceWeightedAllocator`:
   - Compute evidence density per section
   - Distribute word budget based on depth profile
   - Apply section-type dampening (intro/conclusion/methods get less)

**Validation gate**:
- Outline has ≥4 top-level sections
- Every section has paper IDs assigned
- `estimated_word_count` set on each section (from allocator)
- Required sections present (Introduction, Methods of Review, Discussion, etc.)

**Output**: `outline` with sections, paper assignments, word count allocations

---

### Stage 9: Narrative Planning

**Model**: sonnet
**Tools**: None

**What to do**: Plan narrative architecture:
- Central argument / thesis
- Narrative arc
- Per-section directives: role, central claim, structural suggestion,
  key insights (bounded by depth: 2-3 for low, 3-5 for medium, 7-10 for deep)
- Transitions between sections

**Validation gate**:
- Narrative plan has a central argument
- Every outline section has a directive
- Key insights count matches depth profile range

**Output**: `narrative_plan` with section directives

---

### Stage 10: Contextual Enrichment

**Model**: sonnet
**Tools**: `search_pubmed`, `search_semantic_scholar`, `search_openalex`

**What to do**: For each outline section, generate enrichment queries to find
adjacent/contextual material. Search, screen (threshold=2), and extract top 5 papers
per section. This provides cross-field context for the writer.

**Validation gate**:
- Each section has an enrichment entry (even if empty)
- Enrichment sources are distinct from primary corpus

**Output**: `contextual_enrichment` dict (section_id → enrichment data)

---

### Stage 11: Corpus Expansion

**Model**: sonnet
**Tools**: `search_pubmed`, `search_semantic_scholar`, `search_openalex`

**What to do**: Using insights from contextual enrichment (key concepts, cross-field
connections), generate targeted queries to expand the primary research corpus. Search,
screen, extract, and integrate new papers.

**Validation gate**:
- Stage executed
- Any new papers integrated into extractions and evidence map

**Output**: Updated corpus with additional papers

---

### Stage 12: Section Writing

**Model**: sonnet (writing), haiku (critique)
**Tools**: None

**What to do**: Write each section with:
- Assigned findings from extractions
- Full outline context (for cross-section awareness)
- Narrative directives from Stage 9
- Contextual enrichment from Stage 10
- Depth-specific instructions:
  - `low`: "Distill to critical findings. Target ~{word_count} words."
  - `medium`: "Balance thoroughness with readability. Target ~{word_count} words."
  - `deep`: "Exhaustively trace evidence chains. Target ~{word_count} words."
- Word count target from the allocator (Stage 8)

CRITICAL: Synthesize across papers — do NOT summarize paper-by-paper.

Run per-section critique loop (up to max cycles). Critique evaluates: citation accuracy,
synthesis quality, coherence, connection to adjacent sections.

**Validation gate**:
- Every outline section has a draft
- Each draft uses [@paper_id] citation markers
- Word counts approximately match allocator targets (±30%)
- No section is pure paper-by-paper summary

**Output**: `section_drafts` dict (section_id → text)

---

### Stage 13: Passage Search

**Model**: sonnet
**Tools**: `search_pubmed`, `search_semantic_scholar`, `search_openalex`

**What to do**: Mine written sections for undercited claims. For high/medium priority
claims, generate targeted queries and search for supporting papers. Also perform
citation snowballing from top-cited papers in the corpus.

**Validation gate**:
- Stage executed
- Undercited claims identified and logged
- Any new supporting papers integrated

**Output**: Updated sections with additional citations where found

---

### Stage 14: Assembly

**Model**: sonnet (assembly + revision), haiku (critique)
**Tools**: None

**What to do**:
1. Assemble all sections into a full draft
2. Run holistic critique loop (up to 3 revisions or convergence):
   - Evaluate: narrative arc, redundancy, transitions, balance, intro/conclusion alignment
   - Revise based on critique feedback

**Validation gate**:
- Full draft assembled with all sections
- At least 1 holistic critique cycle completed
- Overall quality score logged

**Output**: `full_draft` text

---

### Stage 15: Final Polish

**Model**: sonnet
**Tools**: Bash (for file writing)

**What to do**:
1. Language polishing: fix awkward phrasing, ensure terminology consistency
2. Citation formatting: convert [@paper_id] markers to numbered references
3. Bibliography assembly from structured citation data
4. Apply output format template (markdown/latex/docx)
5. Save final document to `{output_dir}/review.{format}`

**Validation gate**:
- Final document saved to disk
- All [@paper_id] markers resolved to numbered citations
- Bibliography present at end of document
- Document is valid in the target format

**Output**: Final formatted review paper at `{output_dir}/review.{format}`

---

## Subagent Dispatch Pattern

For each stage, dispatch a subagent with this pattern:

```
Agent(
  model="{model}",
  prompt="""
  You are executing Stage {N}: {stage_name} of the AutoReview pipeline.

  ## Context
  Topic: {topic}
  Domain: {domain}
  Depth: {depth}

  ## Upstream Data
  {serialized outputs from previous stages}

  ## Task
  {stage-specific instructions from above}

  ## Output
  Save results to {output_dir}/snapshots/{stage_name}.json
  Report: status (DONE/BLOCKED), key metrics, any concerns.
  """
)
```

**Model selection per stage**:
| Model | Stages |
|-------|--------|
| opus | Complex topics: clustering (Stage 6), assembly critique (Stage 14) |
| sonnet | Most stages: query expansion, search orchestration, screening, extraction, outline, narrative planning, enrichment, section writing, passage search, assembly, final polish |
| haiku | Lightweight: full-text retrieval (Stage 4), critique evaluation, validation |

---

## Running Real Searches

Use MCP tools for real searches when available:
- `search_pubmed` — PubMed/MEDLINE via NCBI Entrez
- `search_semantic_scholar` — Semantic Scholar API
- `search_openalex` — OpenAlex API

If MCP tools are not available, fall back to Python scripts:

```python
python3 -c "
import asyncio
from autoreview.search.pubmed import PubMedSearch
async def run():
    s = PubMedSearch()
    papers = await s.search(queries=['your query'], max_results=50)
    for p in papers:
        print(f'{p.title} | {p.year} | {p.doi}')
asyncio.run(run())
"
```

---

## Snapshot & Resume

After each stage, save the pipeline state to `{output_dir}/snapshots/{stage_name}.json`.
This enables resuming from any stage if the session is interrupted.

To resume: re-invoke this skill with `start_from` set to the desired stage. The skill
will load the most recent snapshot and continue from there.

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running `autoreview run` or `python -m autoreview` | Don't — that requires ANTHROPIC_API_KEY. Dispatch subagents instead |
| Skipping stages | Run ALL 15 stages. Check the architecture diagram above |
| Using synthetic/fake papers | Use MCP search tools for real papers only |
| Extracting only from abstracts | Run full_text_retrieval (Stage 4) first |
| Skipping enrichment stages | contextual_enrichment + corpus_expansion must run before writing |
| Ignoring depth settings | Pass depth config to outline (allocator), narrative (insights range), and writing (instructions) |
| Summarizing paper-by-paper | Section writing must SYNTHESIZE across papers |
| Not saving snapshots | Save after every stage for crash recovery |
| Running all subagents as opus | Use model selection table — haiku for validation, sonnet for most work |
