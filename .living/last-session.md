## Last Session: 2026-03-27

### Commits
- 2099571 feat(kg): interactive contradiction viz, NLI pipeline improvements, community labeling
- d0055db feat(kg): v4 KG models, ingest pipeline, confidence scoring, NLI CLI command
- 259855b docs: v0.3.0 changelog, extraction prompt v5 context fields, README KG section
- 058fdab chore: update .living/ index, session state, and log entries

### What changed
- interactive.py: 6-tab viz with Playground, bigger nodes (20-70px), straight shared edges, curved contradiction edges, clickable contradiction edge detail panel
- nli.py: MoritzLaurer model default, auto-detect label indices, rich claim text from natural_language/context fields
- contradiction_viz.py, community_labeling.py: community subfield labeling (LLM + heuristic)
- models.py, ingest.py, confidence.py, cli.py: v5 context fields, coercion layer, nli-score CLI
- kg_extraction_prompt.md: v5 context fields, citation evidence stubs, contradiction extraction rules

### Key findings
- KGEdge v5 context fields 0% populated — root cause of poor NLI calibration
- Better extraction more impactful than model swap (both complementary)
- MoritzLaurer DeBERTa-v3-base best drop-in replacement (adversarial training)

### Current state
- Branch: main, all code committed
- Remaining untracked: extraction data artifacts (JSONs, PDFs, batch scripts)
