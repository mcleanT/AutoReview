## Last Session: 2026-03-27

### What changed
- `autoreview/knowledge_graph/interactive.py`: Bigger nodes (20-70px), straight shared edges, curved contradiction edges, clickable contradiction edge detail panel, Playground tab
- `autoreview/knowledge_graph/nli.py`: Default model swapped to MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli, auto-detect label indices from model config, rich claim text using natural_language/context fields when available
- `autoreview/knowledge_graph/contradiction_viz.py`: Pydantic models and functions for contradiction data
- `autoreview/knowledge_graph/community_labeling.py`: Async LLM enrichment for community subfield labels
- `autoreview/llm/prompts/community_labeling.py`: LLM prompt for batched community labeling
- `scripts/generate_5tab_viz.py`: 6-tab HTML generation with Playground tab

### Committed
- `2099571` feat(kg): interactive contradiction viz, NLI pipeline improvements, community labeling (9 files, +4849 lines)

### Remaining uncommitted (from prior sessions)
- KG models, ingest, CLI, confidence, tests, README, CHANGELOG, extraction prompt

### Current state
- Branch: main
- 6-tab visualization working in Safari
- NLI pipeline ready for enriched extraction (uses natural_language when available)
