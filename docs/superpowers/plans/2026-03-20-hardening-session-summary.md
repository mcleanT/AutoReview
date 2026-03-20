# Pipeline Quality Hardening — Session Summary

**Date:** 2026-03-20 (overnight session)
**Branch:** main
**Commits:** 15 new commits (e773411..83ff819)
**Tests:** 1133 → 1250 (+117 new tests), 0 failures

---

## What Was Done

Comprehensive pipeline audit identified 28 gaps between AutoReview output and human-written review papers. An implementation plan was created with 16 tasks across 4 phases. All 16 tasks were executed via subagent-driven development with parallel dispatch per phase.

### Phase 1 — Wiring & Validation (4 tasks, all complete)

| Task | What | Status |
|------|------|--------|
| **1. Search backend wiring** | Extracted `build_search_sources()` factory, replaced 6 duplicated if/elif blocks. CrossRef + Europe PMC now active in pipeline. CORE skipped (unreliable). | Wired |
| **2. Critique dimension gates** | `synthesis_quality >= 0.65` and `citation_accuracy >= 0.60` enforced as hard floors. Sections can't pass critique by compensating with high completeness. | Wired |
| **3. Citation scope validation** | Detects when sections cite papers not assigned to them. Logs warnings + tracks utilization rate. | Wired |
| **4. Per-node timeouts** | `asyncio.wait_for()` wraps every DAG node. Default 300s, configurable per node. | Wired |

### Phase 2 — Analytical Depth (4 tasks, all complete)

| Task | What | Status |
|------|------|--------|
| **5. Study quality grading** | `StudyDesign` enum (12 values) + `quality_score` (0-1) + `sample_size` on PaperExtraction. Extraction prompt updated. | Wired (model + prompt) |
| **6. Citation snowballing** | `snowball_papers()` via S2 REST API. Top-10 seeds, backward+forward, DOI/title dedup. 17 tests. | Built, not yet wired into screening node |
| **7. Contradiction resolver** | Evidence-weighted resolution: study design hierarchy, quality scores, sample sizes. Determines which side is stronger. 10 tests. | Built, not yet wired into clustering |
| **8. Synthesis validation metric** | Counts multi-paper vs single-paper citation sentences. `synthesis_ratio >= 0.4` = good synthesis. 7 tests. | Built, standalone utility |

### Phase 3 — Writing Quality (4 tasks, all complete)

| Task | What | Status |
|------|------|--------|
| **9. Language polish** | Terminology extraction, cross-section inconsistency detection, Jaccard-based claim duplication. 13 tests. | Built, not yet wired as pipeline stage |
| **10. Synthesis exemplars** | 2 annotated before/after examples added to section writing system prompt. | Wired |
| **11. Transition repair** | `extract_boundary()` + `build_transition_repair_prompt()` for post-assembly repair. 9 tests. | Built, not yet wired as pipeline stage |
| **12. Depth-dependent thresholds** | LOW=0.70, MEDIUM=0.80, DEEP=0.85, EXHAUSTIVE=0.90. Critics accept `depth_profile` parameter. 6 tests. | Wired |

### Phase 4 — Robustness (4 tasks, all complete)

| Task | What | Status |
|------|------|--------|
| **13. Token budget monitor** | `TokenBudgetMonitor` with WARN/DEGRADE/SAVE_AND_STOP actions. 10 tests. | Built, not yet wired into DAG execute() |
| **14. Snapshot integrity** | SHA256 checksum + schema version on KB snapshots. Legacy backward-compatible. 5 tests. | Wired |
| **15. Partial extraction** | `extract_batch_safe()` returns (results, failures) instead of crashing. | Built, not yet used by pipeline node |
| **16. Pre-flight checks** | `check_api_keys()` + `check_output_dir()`. 5 tests. | Built, no CLI command yet |

---

## What's Wired vs. Ready-to-Wire

**Immediately active in pipeline runs:**
- CrossRef + Europe PMC search backends
- Per-dimension critique gates (synthesis + citation accuracy floors)
- Citation scope validation in section writer
- Per-node timeout protection
- StudyDesign + quality fields in extraction
- Synthesis exemplars in writing prompt
- Depth-dependent critique thresholds
- Snapshot integrity checksums

**Built, tested, ready to integrate (safe next steps):**

| Module | Where to wire | Complexity |
|--------|--------------|------------|
| `snowball_papers()` | After screening in `nodes.py` | Medium — needs screening node modification |
| `resolve_contradiction()` | Inside `clustering.py` after contradiction detection | Medium — needs clustering flow understanding |
| `extract_batch_safe()` | Replace `extract_batch` call in extraction node | Low — drop-in replacement |
| `TokenBudgetMonitor` | In `DAGRunner.execute()` after each node | Medium — needs `_GlobalTokenAccumulator` integration |
| `polish.py` functions | New pipeline stage between critique and formatting | Medium — new DAG node |
| `transition_repair.py` | After section assembly, before holistic critique | Medium — new DAG node |
| `preflight.py` | CLI command + call at pipeline start | Low — new CLI subcommand |

---

## Known Issues

1. **Pyright false positives**: `search_factory` import resolution warnings — file exists, tests pass, pyright config issue
2. **`autoreview/cli/check.py`** not created — plan specified it but wasn't in task scope
3. **Contradiction resolver uses `list[PaperExtraction]`** instead of `dict[str, PaperExtraction]` — intentional improvement (dict keys unused)
4. **`citation_scope.py` logger** uses `structlog.get_logger()` instead of `structlog.get_logger(__name__)` — minor inconsistency

---

## Recommended Next Steps

1. **Wire remaining modules** — The 6 "ready to integrate" items above. Priority: `extract_batch_safe` (easiest, biggest reliability win), then snowballing (biggest coverage win).

2. **Run a pipeline end-to-end** — Test on one ARISE topic to verify the wired improvements (search backends, critique gates, depth thresholds) work in practice.

3. **Complete ARISE benchmark** — 3/10 topics still pending from the overnight batch run.

4. **Commit all changes** — Everything is committed on main (15 commits). Consider whether to squash or keep granular history.

---

## Stats

- **Session duration:** ~2 hours
- **Subagents dispatched:** 16 implementers + 1 code reviewer = 17
- **Files created:** 12 new modules + 12 new test files
- **Files modified:** 10 existing files
- **Test growth:** 1133 → 1250 (+10.3%)
- **All 1250 tests passing, 0 failures**
