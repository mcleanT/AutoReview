---
name: run-pipeline
description: >
  Run the AutoReview pipeline using the claude_code provider — the real Python pipeline
  executes natively with LLM calls routed through `claude -p`. Use when the user says
  "run the pipeline", "run autoreview", "generate a review", "run locally", or wants to
  produce a review paper. This is the PREFERRED method — it runs the actual pipeline code
  with full fidelity (KnowledgeBase in memory, critique loops, Pydantic validation,
  deterministic steps as Python). Presents the configuration interview before launching.
---

# Run AutoReview Pipeline (Client-Side)

Run the real AutoReview Python pipeline with LLM calls routed through Claude Code.
No API key needed — uses your existing Claude Code authentication.

## How It Works

```
autoreview run --provider claude_code --topic "..." --domain cs_ai --depth deep
```

This runs the **actual Python pipeline** (`DAGRunner`, `PipelineNodes`, `KnowledgeBase`).
When a node needs an LLM response, the `ClaudeCodeProvider` routes the call through
`claude -p` (Claude Code CLI in print mode). Everything else runs as native Python:

- KnowledgeBase stays in memory — no serialization, no information loss
- Full texts are available to extraction (not thrown away)
- CitationSelector runs as Python math — deterministic, instant
- OutputFormatter resolves citations in <1 second — no LLM needed
- EvidenceWeightedAllocator runs as math — no LLM needed
- Critique loops execute automatically — quality gates enforced
- Pydantic validates every LLM response — bad output caught immediately
- ALL papers passed to section writers — no truncation or capping

## Why Not the Old `run-local` Skill?

The old skill replaced the entire pipeline with subagent prompts. That caused:
- Full texts lost at serialization boundaries
- Section writers capped at 35 papers (dropped 70%+ of references)
- Citation resolution done by LLM (19 min instead of <1 sec)
- Critique loops skipped entirely
- No Pydantic validation
- 3+ hours runtime

This skill runs the same code as `--provider claude`. Expected runtime: **30-60 minutes**.

---

## MANDATORY: Configuration Interview

Present all settings together. Use defaults for anything not explicitly overridden.

### Settings

| Setting | Question | Default | CLI Flag |
|---------|----------|---------|----------|
| **Topic** | Research topic or question? | *(required)* | positional arg |
| **Domain** | biomedical, cs_ai, chemistry, or general? | `general` | `--domain` |
| **Depth** | `low` (~4K words), `medium` (~8K), or `deep` (~25K+)? | `medium` | `--depth` |
| **Output dir** | Where to save outputs? | `output/` | `--output-dir` |
| **Output format** | markdown, latex, or docx? | `markdown` | `--format` |
| **Date range** | Year filter (e.g., `2015-2024`, `-2020`, `2022-`)? | Domain default | `--date-range` |
| **Model** | Which Claude model? | `sonnet` | `--model` |
| **Fresh run** | Clear previous snapshots? | No | `--fresh` |
| **Start from** | Resume from a specific node? | Full pipeline | via `resume` command |

### Confirm & Launch

Present a summary table and wait for user confirmation:

```
┌─────────────────────────────────────────────────────┐
│           AutoReview Pipeline Config                │
├──────────────┬──────────────────────────────────────┤
│ Topic        │ {topic}                              │
│ Domain       │ {domain}                             │
│ Depth        │ {depth}                              │
│ Date range   │ {date_range or "domain default"}     │
│ Output       │ {output_dir} ({format})              │
│ Model        │ {model}                              │
│ Provider     │ claude_code (CLI bridge)              │
│ Fresh        │ {yes/no}                             │
│ Start from   │ {node or "beginning"}                │
└──────────────┴──────────────────────────────────────┘
```

---

## Execution

### Fresh Run

After the user confirms, run:

```bash
autoreview run \
  "{topic}" \
  --domain {domain} \
  --depth {depth} \
  --output-dir {output_dir} \
  --format {format} \
  --provider claude_code \
  {--model model if specified} \
  {--date-range range if specified} \
  {--fresh if requested} \
  --verbose
```

Run this via the Bash tool. The pipeline will execute all 17 stages automatically.
Monitor the output for progress messages (each node prints a summary on completion).

**IMPORTANT**: This is a long-running command. Use `run_in_background: true` if the
user wants to do other work while it runs, or use a generous timeout (600000ms = 10 min
chunks, restart if needed).

### Resume from Snapshot

If the pipeline was interrupted or the user wants to restart from a specific stage:

```bash
autoreview resume \
  {output_dir}/snapshots/{last_snapshot}.json \
  --start-from {node_name} \
  --provider claude_code \
  {--model model if specified} \
  --verbose
```

Find the latest snapshot:
```bash
ls -t {output_dir}/snapshots/*.json | head -5
```

### Node Names (for --start-from)

```
query_expansion → search → screening → full_text_retrieval → extraction →
clustering → gap_search → draft_outline → contextual_enrichment →
corpus_expansion → final_outline → narrative_planning → citation_selection →
section_writing → passage_search → assembly → final_polish
```

---

## Post-Run

After the pipeline completes:

1. **Check the output**: Read `{output_dir}/review.md` (or `.tex`/`.docx`)
2. **Check token usage**: Read `{output_dir}/token_usage.json`
3. **Check progress**: Read `{output_dir}/progress.json` for per-node timing
4. **Convert to PDF** if requested (invoke `convert-to-pdf` skill)
5. **Evaluate** if requested:
   ```bash
   autoreview evaluate {output_dir}/review.md {reference_pdf} --provider claude_code
   ```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY not set` | Make sure you pass `--provider claude_code` |
| `claude CLI failed` | Check `claude --version` works. Try `claude -p "hello"` |
| Pipeline hangs | A `claude -p` call may be slow on large prompts. Check stderr. |
| Rate limit | The CLI uses your subscription limits. Wait and retry. |
| Snapshot not found (resume) | List snapshots: `ls output/snapshots/` |
| Node fails | Check `{output_dir}/progress.json` for error details. Resume from the failed node. |

## Comparison: This Skill vs Old `run-local`

| Aspect | `run-pipeline` (this) | `run-local` (old) |
|--------|----------------------|-------------------|
| Execution | Real Python pipeline | Subagent simulation |
| KnowledgeBase | In-memory | Serialized JSON files |
| Full texts | Preserved | Lost |
| Citation resolution | Python (<1s) | LLM (19 min) |
| Critique loops | Run automatically | Skipped |
| Papers per section | ALL (no cap) | 35 max |
| Pydantic validation | Every response | None |
| Runtime | 30-60 min | 3+ hours |
| Fidelity | Same as API runs | Degraded |
