---
name: debug-pipeline
description: Debug AutoReview pipeline failures — snapshot inspection, node diagnosis, resume procedures
---

# Debug AutoReview Pipeline

Use this skill when the user reports a pipeline failure, wants to inspect pipeline state, or needs to resume from a snapshot. This guide covers all 15 DAG nodes, common failure patterns, and step-by-step debugging procedures.

---

## Pipeline Node Reference

The pipeline is a linear DAG of 15 nodes. Each node receives a `KnowledgeBase` instance, mutates it, and sets `kb.current_phase`. Snapshots are saved after every node completion (and on failure with a `_FAILED` suffix).

| # | Node Name | Phase Set | Key Input (from KB) | Key Output (on KB) | Common Failures |
|---|---|---|---|---|---|
| 1 | `query_expansion` | `QUERY_EXPANSION` | `topic`, `domain` | `search_queries`, `scope_document` | LLM structured output parse error; empty queries generated |
| 2 | `search` | `SEARCH` | `search_queries` | `candidate_papers` | API rate limits; missing API keys (`NCBI_API_KEY`, `PERPLEXITY_API_KEY`); all search sources fail to initialize |
| 3 | `screening` | `SCREENING` | `candidate_papers`, `scope_document` | `screened_papers` | `TokenBudgetExceeded` on large corpus; LLM batch scoring errors |
| 4 | `full_text_retrieval` | `FULL_TEXT_RETRIEVAL` | `screened_papers` | `screened_papers[].paper.full_text` (mutated in-place) | Network timeouts; missing `UNPAYWALL_EMAIL` / `ELSEVIER_API_KEY`; non-fatal (graceful degradation) |
| 5 | `extraction` | `EXTRACTION` | `screened_papers` | `extractions` (dict of `PaperExtraction`) | `TokenBudgetExceeded`; LLM structured output failures on individual papers; concurrent task errors |
| 6 | `clustering` | `CLUSTERING` | `extractions`, `scope_document` | `evidence_map` (themes, contradictions, gaps, chains) | LLM output too large; evidence map validation errors; empty extractions dict |
| 7 | `gap_search` | `GAP_SEARCH` | `evidence_map.gaps` | Additional papers added to `candidate_papers`, `screened_papers`, `extractions` | Search API failures; remediation dispatcher round limits exceeded |
| 8 | `outline` | `OUTLINE` | `evidence_map`, `scope_document` | `outline` (serialized `ReviewOutline`) | Critique loop not converging; missing required sections from config |
| 9 | `narrative_planning` | `NARRATIVE_PLANNING` | `outline`, `evidence_map`, `scope_document` | `narrative_plan` | LLM structured output parse error; `ReviewOutline.model_validate` fails on corrupted outline |
| 10 | `contextual_enrichment` | `CONTEXTUAL_ENRICHMENT` | `outline`, `narrative_plan`, `scope_document` | `contextual_enrichment` (dict of `SectionEnrichment`) | No search sources available (skipped gracefully); API rate limits |
| 11 | `corpus_expansion` | `CORPUS_EXPANSION` | `contextual_enrichment` | `corpus_expansion_results`, updated `extractions` | Skipped if no enrichment data; `TokenBudgetExceeded`; search API failures |
| 12 | `section_writing` | `SECTION_CRITIQUE` | `outline`, `extractions`, `evidence_map`, `narrative_plan`, `contextual_enrichment` | `section_drafts` | `TokenBudgetExceeded` (most token-intensive node); critique loop stuck; citation validation errors |
| 13 | `passage_search` | `PASSAGE_SEARCH` | `section_drafts`, `extractions` | Updated `section_drafts` with new citations | Skipped if no section drafts; search API failures; miner finds no undercited claims |
| 14 | `assembly` | `HOLISTIC_CRITIQUE` | `section_drafts`, `outline` | `full_draft` | Holistic critique loop not converging; `TokenBudgetExceeded`; citation validation errors |
| 15 | `final_polish` | `FINAL_POLISH` | `full_draft` | Polished `full_draft` | LLM output truncation on very long drafts; skipped if `full_draft` is None |

### Node Dependencies (Linear Chain)

```
query_expansion -> search -> screening -> full_text_retrieval -> extraction
    -> clustering -> gap_search -> outline -> narrative_planning
    -> contextual_enrichment -> corpus_expansion -> section_writing
    -> passage_search -> assembly -> final_polish
```

---

## Debugging Flowchart

When a pipeline run fails, follow this sequence:

```
1. FIND THE SNAPSHOT
   |
   v
   ls output/snapshots/         # List all snapshots
   # Look for files ending in _FAILED.json or find latest.json
   |
   v
2. LOAD AND INSPECT
   |
   v
   python -c "
   from autoreview.models.knowledge_base import KnowledgeBase
   kb = KnowledgeBase.load_snapshot('output/snapshots/latest.json')
   print(f'Phase: {kb.current_phase}')
   print(f'Topic: {kb.topic}')
   print(f'Candidates: {len(kb.candidate_papers)}')
   print(f'Screened: {len(kb.screened_papers)}')
   print(f'Extractions: {len(kb.extractions)}')
   print(f'Section drafts: {len(kb.section_drafts)}')
   print(f'Full draft: {bool(kb.full_draft)}')
   print(f'Tokens: {kb.total_tokens()}')
   "
   |
   v
3. CHECK THE PHASE
   |
   +-- Phase is FAILED --> Check audit log for the failing node name
   |                        (the _FAILED snapshot filename also has the node name)
   |
   +-- Phase is a valid phase (e.g., EXTRACTION) --> Pipeline died mid-node
   |                                                  or completed that phase
   |
   +-- Phase is INITIALIZED --> Pipeline never started
   |
   v
4. INSPECT AUDIT LOG
   |
   v
   python -c "
   from autoreview.models.knowledge_base import KnowledgeBase
   kb = KnowledgeBase.load_snapshot('output/snapshots/latest.json')
   for entry in kb.audit_log[-10:]:
       print(f'[{entry.timestamp}] {entry.node_name}: {entry.action} - {entry.details}')
       if entry.token_usage:
           print(f'  Tokens: {entry.token_usage}')
   "
   |
   v
5. CHECK TOKEN USAGE
   |
   v
   python -c "
   from autoreview.models.knowledge_base import KnowledgeBase
   kb = KnowledgeBase.load_snapshot('output/snapshots/latest.json')
   totals = kb.total_tokens()
   print(f'Total input tokens:  {totals[\"input_tokens\"]:,}')
   print(f'Total output tokens: {totals[\"output_tokens\"]:,}')
   print(f'Combined: {sum(totals.values()):,}')
   "
   |
   v
6. APPLY FIX (see Common Fixes table below)
   |
   v
7. RESUME FROM LAST GOOD SNAPSHOT
```

---

## Snapshot Inspection Commands

### Load and summarize a snapshot

```python
from autoreview.models.knowledge_base import KnowledgeBase

kb = KnowledgeBase.load_snapshot("output/snapshots/latest.json")

# Overview
print(f"Topic: {kb.topic}")
print(f"Domain: {kb.domain}")
print(f"Phase: {kb.current_phase}")
print(f"Candidates: {len(kb.candidate_papers)}")
print(f"Screened: {len(kb.screened_papers)}")
print(f"Extractions: {len(kb.extractions)}")
print(f"Has evidence map: {kb.evidence_map is not None}")
print(f"Has outline: {kb.outline is not None}")
print(f"Section drafts: {list(kb.section_drafts.keys())}")
print(f"Has full draft: {kb.full_draft is not None}")
print(f"Critique reports: {len(kb.critique_history)}")
print(f"Comprehensiveness checks: {len(kb.comprehensiveness_checks)}")
print(f"Iteration counts: {kb.iteration_counts}")
print(f"Tokens: {kb.total_tokens()}")
```

### View audit log

```python
for entry in kb.audit_log:
    tokens = ""
    if entry.token_usage:
        inp = entry.token_usage.get("input_tokens", 0)
        out = entry.token_usage.get("output_tokens", 0)
        tokens = f" [tokens: {inp:,}in / {out:,}out]"
    print(f"[{entry.timestamp:%H:%M:%S}] {entry.node_name}: {entry.action} — {entry.details}{tokens}")
```

### View search queries

```python
for source, queries in kb.search_queries.items():
    print(f"\n{source} ({len(queries)} queries):")
    for q in queries:
        print(f"  - {q}")
```

### Inspect evidence map

```python
if kb.evidence_map:
    em = kb.evidence_map
    print(f"Themes: {len(em.themes)}")
    print(f"Consensus claims: {len(em.consensus_claims)}")
    print(f"Contradictions: {len(em.contradictions)}")
    print(f"Gaps: {len(em.gaps)}")
    print(f"Evidence chains: {len(em.evidence_chains)}")
    if hasattr(em, 'coverage_score'):
        print(f"Coverage score: {em.coverage_score}")
    for gap in em.gaps:
        print(f"  GAP [{gap.severity}]: {gap.topic} — {gap.description}")
```

### Inspect critique history

```python
for cr in kb.critique_history:
    print(f"Target: {cr.target} | Score: {cr.overall_score:.2f} | Passed: {cr.passed}")
    print(f"  Dimensions: {cr.dimension_scores}")
    for issue in cr.issues[:3]:  # First 3 issues
        print(f"  [{issue.severity}] {issue.location}: {issue.description}")
```

### Inspect comprehensiveness checks

```python
for check in kb.comprehensiveness_checks:
    print(f"Checker: {check.checker_name} | Status: {check.status}")
    print(f"  Message: {check.message}")
    if check.remediation:
        print(f"  Remediation: {check.remediation.action} — {check.remediation.params}")
```

### View candidate paper sources

```python
from collections import Counter
sources = Counter(p.source for p in kb.candidate_papers)
print(f"Papers by source: {dict(sources)}")
```

### Check screened paper scores

```python
from collections import Counter
scores = Counter(sp.relevance_score for sp in kb.screened_papers)
print(f"Score distribution: {dict(sorted(scores.items()))}")
print(f"Papers above threshold: {len([sp for sp in kb.screened_papers if sp.relevance_score >= 3])}")
```

---

## Common Fixes

| Problem | Cause | Fix |
|---|---|---|
| "No papers found" / 0 candidates | API keys missing or rate-limited | Check `NCBI_API_KEY`, `PERPLEXITY_API_KEY` env vars. Verify network connectivity. Semantic Scholar and OpenAlex do not need keys. |
| "TokenBudgetExceeded" | Corpus too large for configured budget | Increase `llm.token_budget` in domain config, or reduce `search.max_results_per_source`, or raise `search.relevance_threshold` to screen more aggressively |
| Extraction failed on specific paper | LLM structured output parsing failed on paper content | Check `kb.extractions` to see which papers succeeded. The node processes papers concurrently via `max_concurrent` (default 10). Failing papers are skipped gracefully in most cases. Resume from `extraction` node. |
| Full text retrieval got 0 papers | No open-access sources found | Non-fatal: extraction falls back to abstracts. Set `UNPAYWALL_EMAIL` for broader OA coverage. Optional: `ELSEVIER_API_KEY`, `SPRINGER_API_KEY`. |
| Critique score not converging | Revision loop stuck below threshold | Check `critique.score_threshold` (default 0.80) and `critique.max_revision_cycles` (default 3). Lower the threshold or increase max cycles. Check `critique.convergence_delta` (default 0.05) — if score stops improving by this amount, loop terminates. |
| "Cycle detected in DAG" | Bug in DAG wiring | Should not happen with standard config. Check `runner.py` for dependency definitions. |
| `DAGExecutionError` wrapping an inner exception | A node threw an unhandled exception | The `node_name` attribute on `DAGExecutionError` identifies which node. The `original_error` has the root cause. A `_FAILED` snapshot is saved automatically. |
| "Node 'X' depends on unknown node 'Y'" | DAG wiring error after code change | Verify all node names in `runner.py` `build_pipeline()` match method names on `PipelineNodes`. |
| Outline missing required sections | LLM did not generate expected sections | Check `outline.required_sections` in config. The outline critique loop checks for these. If critique loop maxed out, the outline may still be incomplete. |
| Search source init failed (logged as warning) | Import or initialization error for a search source | Pipeline continues with remaining sources. Check the warning log for which source and why. Common: Perplexity fails without `PERPLEXITY_API_KEY`. |
| `ReviewOutline.model_validate` fails | Corrupted or incompatible outline in snapshot | The outline is stored as a dict (`kb.outline = review_outline.model_dump()`). If schema changed between runs, validation may fail. Fix: re-run from `outline` node. |
| Gap search finds no new papers | Gap queries too specific or APIs rate-limited | Check `evidence_map.gaps` for the queries. Gap search only uses primary + secondary databases (not discovery). Remediation dispatcher has round caps: `search.max_gap_search_rounds` (default 2). |
| Section writing extremely slow | This is the most token-intensive node | It writes all sections, runs citation validation, and runs per-section critique loops. Expected to be the longest node. Check `TokenBudgetExceeded` if it dies mid-way. |
| Pipeline crashes with no `_FAILED` snapshot | Error occurred before the `on_node_error` callback | Check stderr/logs for the raw exception. The `DAGRunner` wraps errors in `DAGExecutionError` but the `on_node_error` callback in `runner.py` saves the snapshot before re-raising. |

---

## Key File Paths

| File | Purpose |
|---|---|
| `autoreview/pipeline/nodes.py` | All 15 DAG node implementations (`PipelineNodes` class) |
| `autoreview/pipeline/runner.py` | `build_pipeline()` wiring + `run_pipeline()` execution |
| `autoreview/pipeline/dag.py` | `DAGRunner` — topological async execution engine |
| `autoreview/pipeline/remediation.py` | `RemediationDispatcher` — expand/retry/threshold actions |
| `autoreview/models/knowledge_base.py` | `KnowledgeBase` — central state; `save_snapshot()` / `load_snapshot()` |
| `autoreview/models/paper.py` | `CandidatePaper`, `ScreenedPaper` |
| `autoreview/config/models.py` | `DomainConfig` and all sub-configs (`SearchConfig`, `CritiqueConfig`, etc.) |
| `autoreview/cli.py` | CLI commands including `resume` |
| `autoreview/extraction/models.py` | `PaperExtraction`, `Finding`, `MethodologyRecord` |
| `autoreview/analysis/evidence_map.py` | `EvidenceMap`, `Theme`, `Contradiction`, `IdentifiedGap` |
| `autoreview/critique/models.py` | `CritiqueReport`, `CritiqueIssue` |
| `autoreview/analysis/comprehensiveness.py` | 5 checker classes: `QueryCoverageChecker`, `CoverageAnomalyChecker`, `BorderlineRescreener`, `PostGapRevalidator`, `BenchmarkValidator` |
| `output/snapshots/` | Snapshot directory (created at runtime) |
| `output/snapshots/latest.json` | Always points to the most recent snapshot |

---

## Resume Procedures

### Resume from CLI

The `resume` command loads a snapshot and continues the pipeline from a specified node:

```bash
# Resume from the last good snapshot, auto-detecting where to continue
python -m autoreview resume output/snapshots/latest.json

# Resume from a specific node (skips all prior nodes)
python -m autoreview resume output/snapshots/latest.json --start-from extraction

# Resume with a specific output format
python -m autoreview resume output/snapshots/latest.json --start-from assembly --format latex

# Resume with verbose logging
python -m autoreview resume output/snapshots/latest.json --start-from clustering --verbose
```

### Resume from Python

```python
import asyncio
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.config import load_config
from autoreview.llm.claude import ClaudeLLMProvider
from autoreview.pipeline.runner import run_pipeline

# Load snapshot
kb = KnowledgeBase.load_snapshot("output/snapshots/latest.json")

# Set up LLM and config
config = load_config(domain=kb.domain)
llm = ClaudeLLMProvider(
    model=config.llm.model,
    api_key="your-api-key",
    max_tokens_generate=config.llm.max_tokens_generate,
    max_tokens_structured=config.llm.max_tokens_structured,
)

# Resume from a specific node
kb = asyncio.run(run_pipeline(llm=llm, config=config, kb=kb, start_from="extraction"))
```

### How `start_from` works

When `start_from` is specified, the `DAGRunner` uses `_downstream_nodes()` to find the start node and all its downstream dependents via BFS on the reverse dependency graph. It then runs only those nodes in topological order. Nodes before the start point are skipped entirely — their data must already exist in the loaded `KnowledgeBase`.

### Choosing the right `start_from` node

| Scenario | Resume from |
|---|---|
| Search APIs were rate-limited | `search` |
| Added new API keys for full text | `full_text_retrieval` |
| Extraction failed on some papers | `extraction` |
| Evidence map looks wrong | `clustering` |
| Outline is incomplete | `outline` |
| Section quality is poor | `section_writing` |
| Final draft needs re-polishing | `final_polish` |
| Token budget was hit during writing | Increase budget in config, then `section_writing` |

### Finding available snapshots

```bash
# List all snapshots with timestamps
ls -la output/snapshots/

# Snapshot naming convention:
#   {YYYYMMDD}T{HHMMSS}_{node_name}.json       — successful node completion
#   {YYYYMMDD}T{HHMMSS}_{node_name}_FAILED.json — node failure
#   latest.json                                  — always the most recent
```

Each snapshot is a full serialization of the `KnowledgeBase` (excluding `full_text` fields on papers to save space). You can load any historical snapshot, not just `latest.json`.
