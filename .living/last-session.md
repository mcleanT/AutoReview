## Last Session: 2026-03-27

### What changed
- `autoreview/knowledge_graph/interactive.py`: Bigger nodes (20-70px), straight shared edges, clickable contradiction edges with detail panel
- `scripts/generate_5tab_viz.py`: Added Playground tab (6 tabs total)

### Key finding
- KGEdge v5 context fields (natural_language, negatable_form, conditions, model_system, organism, certainty) are 0% populated
- NLI receives bare triples not sentences — root cause of poor calibration
- Better extraction > model swap for NLI quality

### Current state
- Branch: main, uncommitted changes
- 6-tab visualization with clickable contradiction edges
- Next: update extraction prompt to populate v5 context fields

### Open questions
- Should extraction re-run be prioritized before model swap?
- How to format natural_language + conditions for NLI input?
