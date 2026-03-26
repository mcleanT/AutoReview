# Last Session — AutoReview

**Date**: 2026-03-25
**Branch**: main
**Commits this session**: 12

## What Was Done

Full Knowledge Graph prototype implemented end-to-end. The `autoreview/knowledge_graph/` module was built from scratch across 10 subagent tasks with all 83 tests passing.

### Module Files Created
- `autoreview/knowledge_graph/models.py` — Pydantic models: Entity, Assertion, Evidence, KnowledgeGraph
- `autoreview/knowledge_graph/ingest.py` — Load extracted JSON into KG models; handles `"key": null` edge case
- `autoreview/knowledge_graph/dedup.py` — Entity deduplication and normalization
- `autoreview/knowledge_graph/graph.py` — NetworkX graph construction from entities + assertions
- `autoreview/knowledge_graph/confidence.py` — Evidence-weighted confidence scoring
- `autoreview/knowledge_graph/analysis.py` — Louvain community detection, contradiction detection, hub identification
- `autoreview/knowledge_graph/viz.py` — Graph visualization utilities
- `autoreview/knowledge_graph/__init__.py` — Public API

## Real Corpus Results (303 papers, gastruloid domain)

| Metric | Value |
|--------|-------|
| Raw entities | 5,894 |
| Deduplicated nodes | 2,462 (58% reduction) |
| Raw assertions | 2,947 |
| Merged edges | 2,899 (1.6% collapsed) |
| Evidence units | 3,331 |
| Louvain communities | 492 |
| Contradictions (controversy > 0.5) | 522 |
| Top hubs | human RA-gastruloids, WNT/beta-catenin signaling, BMP-treated hESCs |

## Key Bug Fixed
- `ingest.py`: `"object_entity": null` in JSON bypasses `d.get("key", default)` -- fixed with `or {}` pattern

## Key Observations
- Entity dedup (58%) is the dominant compression step; assertion merging (1.6%) is minimal
- Extraction model produces highly specific assertions -- good for grounding, but graph is sparser at assertion level than expected
- Subagent-driven development with 10 tasks completed cleanly with zero BLOCKED statuses

## What Is In Progress / Next Steps
- KG module is complete but not yet wired into the main pipeline DAG
- Next: integrate KG construction as a pipeline node after the extraction stage
- Consider improving entity resolution (fuzzy matching, embedding similarity) to boost dedup further

## State of Key Files
- `autoreview/knowledge_graph/` -- complete, all tests passing
- `autoreview/extraction/programmatic.py` -- unchanged this session (baseline + checkpoint files present from prior work)
- `kg_run.log` -- output from real corpus KG run (303 papers)
