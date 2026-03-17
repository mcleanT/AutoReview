# Shared-Corpus Benchmark Optimization

## Problem

The benchmark orchestrator (`paper/run_benchmark.py`) currently runs the full 15-stage pipeline for every run in the matrix (172 runs). Since the first 7 stages (corpus building: search, screening, extraction, clustering, gap analysis) use a single LLM and produce the same evidence base regardless of which model writes the final review, this wastes ~$281 and ~7 hours re-running corpus stages with Opus/Haiku when Sonnet corpus results already exist.

## Design

### Execution Model

Sonnet e2e/RC runs already in the matrix serve double duty — they produce both the corpus snapshots AND the Sonnet comparison data point. No new run types or matrix changes needed.

**Two-phase execution:**

- **Phase 1 (Corpus)**: All Sonnet runs (e2e + retrieval_controlled) — full pipeline, stages 1-15. Corpus snapshots are saved per-node as a side effect (already implemented in `runner.py`).
- **Phase 2 (Writing)**: Everything else — restore KB from the appropriate Sonnet snapshot, run stages 8-15 only with the target model/depth/condition.

### Fork Mapping

| Writing Run | Restores From | Snapshot Node | start_from |
|------------|---------------|---------------|------------|
| Opus/Haiku e2e | Sonnet e2e same topic | `gap_search` | `outline` |
| Opus/Haiku RC | Sonnet RC same topic | `gap_search` | `outline` |
| Ablation (except no_comprehensiveness) | Sonnet e2e same topic | `gap_search` | `outline` |
| Ablation (no_comprehensiveness) | Sonnet e2e same topic | `clustering` | `outline` |
| Depth (low/deep) | Sonnet e2e medium same topic | `gap_search` | `outline` |

### Implementation Scope

Changes are isolated to `paper/run_benchmark.py`. No changes to:
- `expand_run_matrix()` or `paper/models.py`
- Pipeline DAG, runner, or node implementations
- Analysis scripts

#### `_execute_runs()` Modification

Sort runs into two lists:
1. **Corpus runs**: Sonnet e2e + RC runs (`model == SONNET_MODEL and condition in ("end_to_end", "retrieval_controlled")`)
2. **Writing runs**: Everything else

Run corpus batch first, wait for all to complete, then run writing batch.

#### `_run_single()` Fork Logic

For writing runs:
1. Determine the corpus condition: `"retrieval_controlled"` if the writing run is RC, else `"end_to_end"`
2. Build corpus key: `make_run_key(topic_id, SONNET_MODEL, "medium", corpus_condition)`
3. Look up corpus entry in registry — must be `status == "completed"`
4. Resolve snapshot path: `{corpus_entry.output_dir}/snapshots/{snapshot_node}.json`
5. Load KB from snapshot, update `output_dir` to the writing run's directory
6. Create LLM provider with target model
7. Call `run_pipeline(llm=target_llm, config=config, kb=kb, start_from="outline", skip_nodes=...)`

Snapshot node selection:
```python
if condition == "no_comprehensiveness":
    snapshot_node = "clustering"
else:
    snapshot_node = "gap_search"
```

#### KnowledgeBase Snapshot Loading

Verify that `KnowledgeBase` supports loading from a snapshot file path. If not, add a `load_snapshot(path)` classmethod. This is a small utility addition to the KB model.

### Edge Cases

- **Fallback**: If no corpus snapshot exists for a writing run (corpus failed, or user runs Opus-only), fall back to full pipeline with a warning log.
- **Corpus failure**: If a Sonnet corpus run fails, dependent writing runs skip. On retry, corpus is re-attempted first.
- **ARISE topics**: Only have Sonnet e2e runs (Phase 1 only). No Phase 2 forks needed. No special handling.
- **Snapshot path**: Resolved from the corpus registry entry's `output_dir` field.

### Cost & Time Impact

| Phase | Runs | Cost | Time (concurrent=5) |
|-------|------|------|---------------------|
| Phase 1: Corpus (Sonnet full) | 38 | $76 | ~3h |
| Phase 2: Writing-only | 134 | ~$338 | ~5h |
| **Total** | **172** | **~$414** | **~8h** |

**Savings vs. current**: $281 (40%) cost reduction, ~7h (47%) time reduction.
