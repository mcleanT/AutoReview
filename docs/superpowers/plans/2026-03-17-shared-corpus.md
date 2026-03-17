# Shared-Corpus Benchmark Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the benchmark orchestrator so corpus-building stages (1-7) run once per topic with Sonnet, and writing stages (8-15) fork from those snapshots with the target model — saving ~$281 and ~7 hours.

**Architecture:** Changes are isolated to `paper/run_benchmark.py`. The orchestrator splits runs into Phase 1 (Sonnet corpus runs — full pipeline) and Phase 2 (writing-only runs — fork from corpus snapshots via `KnowledgeBase.load_snapshot()` + `run_pipeline(start_from="outline")`). No changes to the pipeline, DAG runner, models, or analysis scripts.

**Tech Stack:** Python 3.11+, asyncio, typer, structlog, existing pipeline infrastructure

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `paper/run_benchmark.py` | Modify | Add helpers, two-phase execution, fork logic |
| `tests/test_paper/test_run_benchmark.py` | Modify | Tests for new helpers and phase separation |

---

## Chunk 1: Implementation

### Task 1: Add helper functions with tests

**Files:**
- Modify: `paper/run_benchmark.py:108-124` (after `_classify_batch`, before `generate_matrix`)
- Modify: `tests/test_paper/test_run_benchmark.py`

- [ ] **Step 1: Write failing tests for `_is_corpus_run`**

Add to `tests/test_paper/test_run_benchmark.py`:

```python
from paper.run_benchmark import _is_corpus_run, _find_corpus_snapshot


class TestCorpusHelpers:
    def test_sonnet_e2e_is_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "end_to_end")
        assert _is_corpus_run(key) is True

    def test_sonnet_rc_is_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "retrieval_controlled")
        assert _is_corpus_run(key) is True

    def test_opus_e2e_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-opus-4-6", "medium", "end_to_end")
        assert _is_corpus_run(key) is False

    def test_sonnet_ablation_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "medium", "no_evidence_chains")
        assert _is_corpus_run(key) is False

    def test_sonnet_depth_is_not_corpus_run(self) -> None:
        key = ("topic_a", "claude-sonnet-4-6", "deep", "end_to_end")
        assert _is_corpus_run(key) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestCorpusHelpers -v`
Expected: FAIL with ImportError (functions don't exist yet)

- [ ] **Step 3: Implement `_is_corpus_run`**

Add to `paper/run_benchmark.py` after `_classify_batch` (after line 124):

```python
def _is_corpus_run(key: RunKey) -> bool:
    """Check if a run is a corpus-producing run (Sonnet e2e or RC, medium depth).

    Corpus runs execute the full pipeline and produce snapshots that
    writing-only runs can fork from.
    """
    _, model, depth, condition = key
    return (
        model == SONNET_MODEL
        and depth == "medium"
        and condition in ("end_to_end", "retrieval_controlled")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestCorpusHelpers -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Write failing tests for `_find_corpus_snapshot`**

Add to `tests/test_paper/test_run_benchmark.py`:

```python
class TestFindCorpusSnapshot:
    def test_returns_snapshot_path_when_corpus_completed(self, tmp_path: Path) -> None:
        """When a Sonnet e2e corpus run is completed with a gap_search snapshot, return its path."""
        from pathlib import Path as P

        registry = RunRegistry()
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_end_to_end")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            output_dir=output_dir,
            review_path="r.md",
        )
        # Create a snapshot file
        snap_dir = P(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        snap_file = snap_dir / "20260317T120000_gap_search.json"
        snap_file.write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is not None
        assert result.name == "20260317T120000_gap_search.json"

    def test_returns_clustering_snapshot_for_no_comprehensiveness(self, tmp_path: Path) -> None:
        """no_comprehensiveness ablation should fork from clustering, not gap_search."""
        from pathlib import Path as P

        registry = RunRegistry()
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_end_to_end")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            output_dir=output_dir,
            review_path="r.md",
        )
        snap_dir = P(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "20260317T120000_clustering.json").write_text("{}")
        (snap_dir / "20260317T120100_gap_search.json").write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "no_comprehensiveness")
        assert result is not None
        assert "clustering" in result.name

    def test_returns_none_when_no_corpus_run(self) -> None:
        """When no corpus run exists, return None (caller falls back to full pipeline)."""
        registry = RunRegistry()
        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is None

    def test_returns_none_when_corpus_failed(self) -> None:
        """When the corpus run failed, return None."""
        registry = RunRegistry()
        registry.register_failure(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "end_to_end"),
            error="boom",
        )
        result = _find_corpus_snapshot(registry, "topic_a", "end_to_end")
        assert result is None

    def test_rc_condition_looks_up_rc_corpus(self, tmp_path: Path) -> None:
        """retrieval_controlled writing runs should look up the RC corpus, not e2e."""
        from pathlib import Path as P

        registry = RunRegistry()
        # Only RC corpus exists, not e2e
        output_dir = str(tmp_path / "topic_a" / "claude-sonnet-4-6_medium_retrieval_controlled")
        registry.register_complete(
            make_run_key("topic_a", "claude-sonnet-4-6", "medium", "retrieval_controlled"),
            output_dir=output_dir,
            review_path="r.md",
        )
        snap_dir = P(output_dir) / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "20260317T120000_gap_search.json").write_text("{}")

        result = _find_corpus_snapshot(registry, "topic_a", "retrieval_controlled")
        assert result is not None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestFindCorpusSnapshot -v`
Expected: FAIL with ImportError

- [ ] **Step 7: Implement `_find_corpus_snapshot`**

Add to `paper/run_benchmark.py` after `_is_corpus_run`:

```python
def _find_corpus_snapshot(
    registry: RunRegistry,
    topic_id: str,
    condition: str,
) -> Path | None:
    """Find the corpus snapshot path for a writing-only run.

    Looks up the completed Sonnet corpus run for the given topic and returns
    the path to the appropriate snapshot file. Returns None if no corpus run
    exists (caller should fall back to full pipeline).

    Args:
        registry: Run registry with completed runs.
        topic_id: Topic identifier.
        condition: The writing run's condition (used to determine which
            corpus run and snapshot node to look up).

    Returns:
        Path to the snapshot JSON file, or None if unavailable.
    """
    corpus_condition = "retrieval_controlled" if condition == "retrieval_controlled" else "end_to_end"
    corpus_key = make_run_key(topic_id, SONNET_MODEL, "medium", corpus_condition)

    entry = registry.runs.get(corpus_key)
    if not entry or entry.status != "completed":
        return None

    snapshot_node = "clustering" if condition == "no_comprehensiveness" else "gap_search"

    snapshots_dir = Path(entry.output_dir) / "snapshots"
    if not snapshots_dir.exists():
        return None

    matches = sorted(snapshots_dir.glob(f"*_{snapshot_node}.json"))
    return matches[-1] if matches else None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestCorpusHelpers tests/test_paper/test_run_benchmark.py::TestFindCorpusSnapshot -v`
Expected: PASS (all 10 tests)

- [ ] **Step 9: Commit**

```bash
git add paper/run_benchmark.py tests/test_paper/test_run_benchmark.py
git commit -m "feat(paper): add corpus run helpers for shared-corpus optimization"
```

---

### Task 2: Modify `_execute_runs` for two-phase execution

**Files:**
- Modify: `paper/run_benchmark.py:191-287` (the `_execute_runs` function)

- [ ] **Step 1: Write failing test for phase separation**

Add to `tests/test_paper/test_run_benchmark.py`:

```python
class TestPhaseSeparation:
    def test_corpus_runs_sorted_first(self) -> None:
        """Verify _is_corpus_run correctly partitions the run list."""
        runs: list[tuple[str, str, str, str]] = [
            ("t1", "claude-opus-4-6", "medium", "end_to_end"),
            ("t1", "claude-sonnet-4-6", "medium", "end_to_end"),
            ("t1", "claude-haiku-4-5", "medium", "end_to_end"),
            ("t1", "claude-sonnet-4-6", "medium", "no_evidence_chains"),
            ("t1", "claude-sonnet-4-6", "deep", "end_to_end"),
        ]
        corpus = [k for k in runs if _is_corpus_run(k)]
        writing = [k for k in runs if not _is_corpus_run(k)]

        assert len(corpus) == 1
        assert corpus[0][1] == "claude-sonnet-4-6"
        assert corpus[0][3] == "end_to_end"
        assert len(writing) == 4
```

- [ ] **Step 2: Run test to verify it passes (uses already-implemented helper)**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestPhaseSeparation -v`
Expected: PASS (the partitioning logic uses `_is_corpus_run` which is already implemented)

- [ ] **Step 3: Rewrite `_execute_runs` with two-phase logic**

Replace the `_execute_runs` function in `paper/run_benchmark.py` (lines 191-287) with:

```python
async def _execute_runs(
    runs: list[RunKey],
    topic_lookup: dict[str, Any],
    registry: RunRegistry,
    results_dir: Path,
    max_concurrent: int,
) -> None:
    """Execute pipeline runs with shared-corpus optimization.

    Phase 1: Sonnet corpus runs (e2e + RC) — full pipeline producing snapshots.
    Phase 2: All other runs — fork from corpus snapshots when available,
             falling back to full pipeline if no snapshot exists.
    """
    from autoreview.config import load_config
    from autoreview.config.models import DepthLevel
    from autoreview.llm.factory import create_llm_provider
    from autoreview.models.knowledge_base import KnowledgeBase
    from autoreview.pipeline.runner import run_pipeline

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_single(key: RunKey) -> None:
        topic_id, model, depth, condition = key
        run_key_str = make_run_key(*key)
        topic = topic_lookup.get(topic_id)
        if not topic:
            logger.error("benchmark.unknown_topic", topic_id=topic_id)
            return

        async with semaphore:
            logger.info("benchmark.run.start", key=run_key_str)
            registry.register_start(run_key_str)

            output_dir = str(results_dir / topic_id / f"{model}_{depth}_{condition}")

            try:
                overrides: dict[str, Any] = {"llm": {"model": model}}
                config = load_config(domain=topic.domain, overrides=overrides)
                config.writing.depth = DepthLevel(depth)

                if topic.date_range:
                    config.search.date_range = topic.date_range

                # Ablation config flags
                if condition == "no_critique_loops":
                    config.critique.max_revision_cycles = 0
                if condition == "no_evidence_chains":
                    config.writing.evidence_chains = False

                llm = create_llm_provider(config.llm)

                # --- Shared-corpus fork logic ---
                snapshot_path = None
                if not _is_corpus_run(key):
                    snapshot_path = _find_corpus_snapshot(registry, topic_id, condition)

                if snapshot_path:
                    # Phase 2: Fork from corpus snapshot — writing stages only
                    logger.info(
                        "benchmark.run.fork",
                        key=run_key_str,
                        snapshot=str(snapshot_path),
                    )
                    kb = KnowledgeBase.load_snapshot(str(snapshot_path))
                    kb.output_dir = output_dir
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    skip_nodes: set[str] | None = None
                    if condition == "no_passage_mining":
                        skip_nodes = {"passage_search"}

                    kb = await run_pipeline(
                        llm=llm,
                        config=config,
                        kb=kb,
                        start_from="outline",
                        skip_nodes=skip_nodes,
                    )

                elif condition == "retrieval_controlled":
                    # Full retrieval-controlled pipeline
                    from paper.analysis.inject_bibliography import inject_bibliography

                    kb = await inject_bibliography(
                        pdf_path=Path(topic.reference.pdf_path),
                        topic=topic.title,
                        domain=topic.domain,
                        output_dir=output_dir,
                        llm=llm,
                    )
                    kb = await run_pipeline(
                        llm=llm, config=config, kb=kb, start_from="full_text_retrieval"
                    )

                else:
                    # Full pipeline (corpus run or fallback when no snapshot)
                    if not _is_corpus_run(key):
                        logger.warning(
                            "benchmark.run.no_corpus_snapshot",
                            key=run_key_str,
                            msg="Falling back to full pipeline",
                        )
                    kb = KnowledgeBase(
                        topic=topic.title, domain=topic.domain, output_dir=output_dir
                    )
                    skip_list: list[str] = []
                    if condition == "no_passage_mining":
                        skip_list.append("passage_search")
                    elif condition == "no_comprehensiveness":
                        skip_list.append("gap_search")
                    kb = await run_pipeline(
                        llm=llm,
                        config=config,
                        kb=kb,
                        skip_nodes=set(skip_list) if skip_list else None,
                    )

                # Register completion
                review_path = next(
                    Path(output_dir).glob("*.md"), Path(output_dir) / "review.md"
                )
                tokens = kb.total_tokens()

                registry.register_complete(
                    run_key_str,
                    output_dir=output_dir,
                    review_path=str(review_path),
                    cost_usd=0.0,
                    tokens_input=tokens.get("input_tokens", 0),
                    tokens_output=tokens.get("output_tokens", 0),
                )
                logger.info("benchmark.run.complete", key=run_key_str)

            except Exception as e:
                logger.error("benchmark.run.failed", key=run_key_str, error=str(e))
                registry.register_failure(run_key_str, str(e))

            finally:
                registry.save(results_dir / "run_registry.json")

    # Split into corpus (Phase 1) and writing (Phase 2) runs
    corpus_runs = [k for k in runs if _is_corpus_run(k)]
    writing_runs = [k for k in runs if not _is_corpus_run(k)]

    if corpus_runs:
        logger.info("benchmark.phase1.start", n_corpus=len(corpus_runs))
        await asyncio.gather(*[_run_single(k) for k in corpus_runs], return_exceptions=True)
        logger.info("benchmark.phase1.complete")

    if writing_runs:
        logger.info("benchmark.phase2.start", n_writing=len(writing_runs))
        await asyncio.gather(*[_run_single(k) for k in writing_runs], return_exceptions=True)
        logger.info("benchmark.phase2.complete")
```

- [ ] **Step 4: Run all existing tests to verify no regressions**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py -v`
Expected: PASS (all tests including the new ones)

- [ ] **Step 5: Commit**

```bash
git add paper/run_benchmark.py tests/test_paper/test_run_benchmark.py
git commit -m "feat(paper): implement shared-corpus two-phase execution"
```

---

### Task 3: Update cost estimates and matrix summary

**Files:**
- Modify: `paper/run_benchmark.py:39-50` (cost estimates)
- Modify: `paper/run_benchmark.py:61-67` (`estimate_cost`)
- Modify: `paper/run_benchmark.py:70-105` (`_format_matrix_summary`)
- Modify: `tests/test_paper/test_run_benchmark.py`

- [ ] **Step 1: Write failing test for updated cost estimates**

Add to `tests/test_paper/test_run_benchmark.py`:

```python
class TestSharedCorpusCostEstimate:
    def test_writing_only_runs_cost_less(self, sample_topics: TopicsConfig) -> None:
        """Writing-only runs should have lower cost estimates than full runs."""
        from paper.run_benchmark import estimate_cost

        models = ["claude-opus-4-6", "claude-sonnet-4-6"]
        matrix = expand_run_matrix(sample_topics.topics, models, include_depth=True)

        # Full cost (no corpus optimization)
        full_cost = estimate_cost(matrix, shared_corpus=False)
        # Optimized cost (writing-only for non-corpus runs)
        optimized_cost = estimate_cost(matrix, shared_corpus=True)

        assert optimized_cost < full_cost

    def test_corpus_only_same_cost_either_way(self) -> None:
        """A single Sonnet e2e run costs the same regardless of shared_corpus flag."""
        from paper.run_benchmark import estimate_cost

        corpus_only = [("t1", "claude-sonnet-4-6", "medium", "end_to_end")]
        assert estimate_cost(corpus_only, shared_corpus=True) == estimate_cost(
            corpus_only, shared_corpus=False
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py::TestSharedCorpusCostEstimate -v`
Expected: FAIL (estimate_cost doesn't accept `shared_corpus` parameter)

- [ ] **Step 3: Add writing-only cost table and update `estimate_cost`**

Add the writing-only cost table after the existing `_COST_ESTIMATES` dict in `paper/run_benchmark.py`:

```python
# Writing-only cost estimates (stages 8-15 only, ~50-55% of full cost)
_WRITING_COST_ESTIMATES: dict[tuple[str, str], float] = {
    ("opus", "medium"): 8.50,
    ("opus", "low"): 3.50,
    ("opus", "deep"): 20.0,
    ("sonnet", "medium"): 1.00,
    ("sonnet", "low"): 0.40,
    ("sonnet", "deep"): 2.50,
    ("haiku", "medium"): 0.10,
    ("haiku", "low"): 0.04,
    ("haiku", "deep"): 0.25,
}
```

Update `estimate_cost`:

```python
def estimate_cost(matrix: list[RunKey], *, shared_corpus: bool = True) -> float:
    """Estimate total API cost for the run matrix.

    Args:
        matrix: List of run keys.
        shared_corpus: If True, non-corpus runs use writing-only cost estimates.
    """
    total = 0.0
    for key in matrix:
        _, model, depth, _ = key
        tier = _model_tier(model)
        if shared_corpus and not _is_corpus_run(key):
            total += _WRITING_COST_ESTIMATES.get((tier, depth), 1.0)
        else:
            total += _COST_ESTIMATES.get((tier, depth), 2.0)
    return total
```

- [ ] **Step 4: Update `_format_matrix_summary` to show corpus/writing breakdown**

In `_format_matrix_summary`, add after the "Remaining" line (around line 78):

```python
    # Corpus vs writing breakdown
    corpus_count = sum(1 for k in matrix if _is_corpus_run(k))
    writing_count = len(matrix) - corpus_count
    lines.append(f"  Corpus (full Sonnet): {corpus_count}")
    lines.append(f"  Writing-only: {writing_count}")
```

And update the cost line to use `shared_corpus=True`:

```python
    lines.append(
        f"  Estimated cost (remaining): ${estimate_cost([k for k in matrix if not registry.is_completed(make_run_key(*k))], shared_corpus=True):.0f}"
    )
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_paper/test_run_benchmark.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add paper/run_benchmark.py tests/test_paper/test_run_benchmark.py
git commit -m "feat(paper): update cost estimates for shared-corpus optimization"
```

---

### Task 4: Full test suite verification

- [ ] **Step 1: Run the full paper test suite**

Run: `python -m pytest tests/test_paper/ -v`
Expected: All 131+ tests PASS

- [ ] **Step 2: Run type checking**

Run: `python -m pyright paper/run_benchmark.py`
Expected: No errors

- [ ] **Step 3: Verify generate-matrix output reflects new breakdown**

Run: `python -m paper.run_benchmark generate-matrix --models claude-opus-4-6,claude-sonnet-4-6,claude-haiku-4-5`
Expected: Output shows corpus/writing breakdown and reduced cost estimate (~$414 vs previous ~$695)

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix(paper): address review feedback for shared-corpus optimization"
```
