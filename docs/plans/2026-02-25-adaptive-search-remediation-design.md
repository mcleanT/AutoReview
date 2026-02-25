# Design: Adaptive Search Remediation & Full-Text Brainstorm

**Date**: 2026-02-25
**Status**: Approved

## Problem

AutoReview's comprehensiveness checks detect search quality problems but never act on them. Three specific limitations:

1. **Checks are advisory-only**: All 5 checks (QueryCoverage, CoverageAnomaly, BorderlineRescreener, PostGapRevalidator, BenchmarkValidator) store results in `kb.comprehensiveness_checks` but only BorderlineRescreener modifies pipeline state. Warnings are logged and ignored.

2. **Static queries**: Query expansion is a single one-shot LLM call. If the generated queries miss sub-topics identified in the scope document, nothing corrects course.

3. **Gap search runs once**: If supplementary search doesn't fill identified gaps, PostGapRevalidator warns but the pipeline continues with unfilled gaps.

Additionally, full-text access is limited to Unpaywall OA papers (~15-30% of corpus), leaving most papers extracted from abstracts only.

## Approach: Remediation Actions on Checks

Each `ComprehensiveCheckResult` gains a `remediation` field — a structured action the pipeline should execute. A `RemediationDispatcher` reads the remediation and calls the appropriate action. Checks diagnose *and* prescribe; the dispatcher executes.

This was chosen over two alternatives:
- **Central remediation controller** — over-engineered for 5 checks, harder to test individual remediations
- **Retry loops inside nodes** — scatters retry logic, bloats nodes, harder to enforce global caps

## Design

### Remediation Model

New model on `ComprehensiveCheckResult`:

```python
class RemediationAction(AutoReviewModel):
    action: str              # "expand_queries", "retry_gap_search", "lower_screening_threshold"
    params: dict[str, Any]   # action-specific parameters
    priority: int = 1        # higher = more urgent
```

- Checks with `WARNING` or `FAILED` populate `remediation`
- `PASSED` checks leave it `None`

### RemediationDispatcher

New class in `autoreview/pipeline/remediation.py`:

- Constructor: `(llm, config, max_rounds=2)`
- Core method: `async def execute(self, kb, check_result) -> bool`
- Tracks rounds per action type to enforce caps
- Three action methods:
  - `_expand_queries(kb, uncovered_topics)` — LLM generates queries for missing topics, runs search + screening, merges results
  - `_retry_gap_search(kb, remaining_gaps, previous_queries)` — generates new queries avoiding previous failures, searches, screens, extracts, re-validates
  - `_lower_screening_threshold(kb, params)` — re-screens rejected papers at threshold-1

### Adaptive Query Expansion

Flow change in `query_expansion` node:

1. Generate queries + scope document (existing)
2. Run `QueryCoverageChecker` (existing)
3. If WARNING with remediation, dispatcher calls `_expand_queries()`
4. `_expand_queries()`:
   - Takes `uncovered_topics` from remediation params
   - Calls LLM with `build_targeted_query_expansion_prompt(uncovered_topics, domain, date_range)` — scoped prompt for filling specific holes
   - Merges new queries into `kb.search_queries`
   - Re-runs `QueryCoverageChecker` on expanded set
   - Loops if still WARNING and rounds < max
5. Cap: 2 rounds

The `search` node is unchanged — it reads whatever queries exist in `kb.search_queries`.

`QueryCoverageChecker` changes: populates remediation when WARNING:
```python
remediation=RemediationAction(
    action="expand_queries",
    params={"uncovered_topics": uncovered_topics}
)
```

### Iterative Gap Search

Flow change in `gap_search` node:

1. Detect gaps + run gap search (existing)
2. Run `PostGapRevalidator` (existing)
3. If WARNING with remediation, dispatcher calls `_retry_gap_search()`
4. `_retry_gap_search()`:
   - Takes `remaining_gaps` and `previous_queries` from params
   - Calls LLM with `build_retry_gap_queries_prompt(remaining_gaps, previous_queries)` — tells LLM previous queries failed, try alternative terminology
   - Searches primary + secondary databases
   - Screens + extracts new papers, merges into kb
   - Re-runs `PostGapRevalidator`
   - Loops if still WARNING and rounds < max
5. Cap: 2 rounds

`PostGapRevalidator` changes: populates remediation when major gaps remain:
```python
remediation=RemediationAction(
    action="retry_gap_search",
    params={
        "remaining_gaps": [gap.model_dump() for gap in remaining_major_gaps],
        "previous_queries": queries_used
    }
)
```

`CoverageAnomalyChecker` changes: populates remediation for anomalies:
- High rejection rate (>80%) → `action: "lower_screening_threshold"`
- Source returning 0 results → `action: "expand_queries"` with `failed_sources`

### Iteration Caps

| Action | Max rounds | Rationale |
|--------|-----------|-----------|
| `expand_queries` | 2 | Diminishing returns on sub-topic discovery |
| `retry_gap_search` | 2 | If 2 rounds can't fill a gap, literature likely doesn't exist |
| `lower_screening_threshold` | 1 | Only drop by 1 point; lower introduces noise |

Worst case: ~4 extra LLM calls + search API calls. Realistic: 1-2 extra.

### Configuration

New fields in `SearchConfig`:
```python
max_query_expansion_rounds: int = 2
max_gap_search_rounds: int = 2
min_coverage_threshold: float = 0.75   # previously hardcoded
```

### Pipeline Flow

```
query_expansion
  ├─ Generate queries + scope (existing)
  ├─ QueryCoverageChecker → WARNING?
  │    └─ Dispatcher._expand_queries() (up to 2 rounds)
  └─ Final queries in kb.search_queries

search (unchanged)

screening
  ├─ Screen papers (existing)
  ├─ CoverageAnomalyChecker → WARNING?
  │    └─ Dispatcher._lower_screening_threshold() or ._expand_queries()
  ├─ BorderlineRescreener (existing)
  └─ Re-run search+screening if queries were expanded

full_text_retrieval (unchanged)
extraction (unchanged)
clustering (unchanged)

gap_search
  ├─ Detect gaps + run gap search (existing)
  ├─ PostGapRevalidator → WARNING?
  │    └─ Dispatcher._retry_gap_search() (up to 2 rounds)
  ├─ BenchmarkValidator (existing)
  └─ Final evidence map

...rest of pipeline unchanged...
```

## Files

### New files
- `autoreview/pipeline/remediation.py` — `RemediationAction`, `RemediationDispatcher`
- `tests/test_pipeline/test_remediation.py` — dispatcher tests

### Modified files
- `autoreview/analysis/comprehensiveness.py` — checks populate `remediation` field
- `autoreview/pipeline/nodes.py` — nodes call dispatcher when remediation exists
- `autoreview/config/models.py` — new config fields
- `autoreview/config/defaults/biomedical.yaml` — new defaults
- `autoreview/config/defaults/cs_ai.yaml` — new defaults
- `autoreview/config/defaults/chemistry.yaml` — new defaults
- `autoreview/llm/prompts/query_expansion.py` — add `build_targeted_query_expansion_prompt()`
- `autoreview/llm/prompts/clustering.py` — add `build_retry_gap_queries_prompt()`

### Test coverage
- `tests/test_pipeline/test_remediation.py` — dispatcher round caps, each action type, integration with checks
- Updated comprehensiveness tests — checks return remediation fields

## Full-Text Brainstorm (Biomedical-First)

Separate from the remediation work. Not in current implementation scope — roadmap for next iteration.

### Problem
Unpaywall captures ~15-30% of papers as full text. Remaining 70-85% extracted from abstracts, losing methodology details and downgrading evidence strength.

### Prioritized sources for biomedical

**1. PubMed Central (PMC)** — Highest impact
- ~8M free full-text articles
- Free E-Utilities API (`efetch` with `db=pmc`), structured XML
- We already have PMIDs — lookup is `pmid → pmcid → full XML`
- XML is cleaner than PDF extraction (sections, tables, figures)
- Rate limit: 3 req/sec (10 with NCBI API key)
- Estimated boost: +15-25% of biomedical corpus

**2. bioRxiv/medRxiv** — Medium impact
- ~600K preprints, growing
- Free API at `api.biorxiv.org`, PDF links
- Covers recent work not yet in journals
- Estimated boost: +5-10%

**3. arXiv** — Lower biomedical impact
- ~2.5M papers, IDs already captured from Semantic Scholar
- Free API, zero cost
- Estimated boost: +2-5% biomedical, +20-30% CS/AI

### Architecture sketch
The `full_text_retrieval` node would try sources in a configurable cascade per domain:
```
Unpaywall → PMC (if PMID) → bioRxiv (if DOI matches) → arXiv (if arxiv ID)
```
First successful source wins. Domain config specifies order and which sources to enable.
