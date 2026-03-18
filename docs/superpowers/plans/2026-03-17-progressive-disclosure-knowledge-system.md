# Progressive Disclosure Knowledge System — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a three-tier progressive disclosure knowledge system that reduces session-start context load from 1,000–4,300 lines to <100 lines while enabling cross-project knowledge transfer.

**Architecture:** Global knowledge stored in `~/.claude/knowledge/` (15 domain files), discoverable via MEMORY.md routing tables (auto-loaded). Project `.living/` files accessed on-demand via compact INDEX.md summaries. Weekly silent audit via mycelium hook + sonnet subagent.

**Tech Stack:** Shell (hooks), Markdown (knowledge files), Claude Code hooks API (JSON additionalContext)

**Spec:** `docs/superpowers/specs/2026-03-17-progressive-disclosure-knowledge-system-design.md`

---

## File Structure

### New Files (create)

| File | Purpose |
|------|---------|
| `~/.claude/knowledge/skills.md` | Skills routing table — mycelium skills first, then all others |
| `~/.claude/knowledge/external-apis.md` | Publisher access, API keys, backoff strategies |
| `~/.claude/knowledge/data-pipelines.md` | DAG design, snapshots, retry, orchestration |
| `~/.claude/knowledge/figure-standards.md` | Palettes, DPI, journal formatting |
| `~/.claude/knowledge/python-patterns.md` | Async I/O, structlog, Pydantic, type hints |
| `~/.claude/knowledge/debugging-patterns.md` | Conda, matplotlib, pytest, environment |
| `~/.claude/knowledge/scientific-analysis.md` | Statistical tests, normalization, QC |
| `~/.claude/knowledge/spatial-biology.md` | Spatial transcriptomics shared knowledge |
| `~/.claude/knowledge/llm-patterns.md` | Prompt construction, token mgmt, model selection |
| `~/.claude/knowledge/git-workflows.md` | Worktrees, hooks, commits, PRs |
| `~/.claude/knowledge/environment-setup.md` | Conda, env vars, VPN, institutional access |
| `~/.claude/knowledge/testing-patterns.md` | pytest conventions, mocking, async tests |
| `~/.claude/knowledge/data-formats.md` | AnnData, JSON/YAML, PDF parsing |
| `~/.claude/knowledge/publishing-workflows.md` | PDF gen, citations, journal requirements |
| `~/.claude/knowledge/writing-conventions.md` | Scientific prose, IMRAD, citation style |
| `~/.claude/knowledge/.last-audit` | Timestamp + summary |
| `~/.claude/knowledge/.audit-log.md` | Append-only audit history |
| `Science/.living/INDEX.md` | Auto-generated index for Science portfolio |
| `Science/AutoReview/.living/INDEX.md` | Auto-generated index for AutoReview |
| `Science/Autonomous Science/.living/INDEX.md` | Auto-generated index for Autonomous Science |
| `Science/Gastruloids/L Metric/.living/INDEX.md` | Auto-generated index for Gastruloids |
| `Science/SpaceBar/Code/.living/INDEX.md` | Auto-generated index for SpaceBar |

### Modified Files

| File | Change |
|------|--------|
| `~/.claude/projects/-Users-mst36-Desktop-Projects-Science-AutoReview/memory/MEMORY.md` | Append global knowledge domain table |
| `~/.claude/projects/-Users-mst36-Desktop-Projects-Science-Autonomous-Science/memory/MEMORY.md` | Append global knowledge domain table |
| `~/.claude/projects/-Users-mst36-Desktop-Projects-Science/memory/MEMORY.md` | Append global knowledge domain table |
| `/Users/mst36/tools/mycelium/skills/core/hooks/mycelium-health.sh` | Add audit trigger check |
| `Science/CLAUDE.md` | Update session-start instructions to use INDEX.md |
| `Science/AutoReview/CLAUDE.md` | Update Living Repository Protocol section |

---

## Task Dependency Graph

```
Task 1 (knowledge dir + domain templates)  ──┐
Task 2 (seed 8 core domains from .living/)   ──┤──> Task 5 (MEMORY.md routing tables)
Task 3 (generate INDEX.md for all projects)  ──┤──> Task 6 (CLAUDE.md instruction updates)
Task 4 (skills.md routing table)             ──┘──> Task 7 (mycelium-health.sh audit trigger)
                                                  └──> Task 8 (validation + commit)
```

**Batch 1 (independent):** Tasks 1, 3, 4
**Batch 2 (needs Task 1):** Task 2
**Batch 3 (needs Tasks 1-4):** Tasks 5, 6, 7
**Batch 4 (needs all):** Task 8

---

### Task 1: Create Knowledge Directory and Domain File Templates

**Files:**
- Create: `~/.claude/knowledge/` (directory)
- Create: 14 domain files (all except skills.md — that's Task 4)
- Create: `~/.claude/knowledge/.last-audit`
- Create: `~/.claude/knowledge/.audit-log.md`

- [ ] **Step 1: Create the knowledge directory**

```bash
mkdir -p ~/.claude/knowledge
```

- [ ] **Step 2: Create domain file templates for growth domains (7 files)**

Each growth domain file starts with a header template and zero entries. Create these files:
- `llm-patterns.md`
- `git-workflows.md`
- `environment-setup.md`
- `testing-patterns.md`
- `data-formats.md`
- `publishing-workflows.md`
- `writing-conventions.md`

Template for each:

```markdown
# {Domain Name}

> **When to read:** {1-line trigger description}

---

<!-- Entries below. Format:
### [domain-tag] One-line summary
**What:** ...
**Evidence:** ...
**When useful:** ...
**Scope:** transferable
**Status:** active | unreviewed
**Last validated:** YYYY-MM-DD
-->
```

Example for `llm-patterns.md`:
```markdown
# LLM Patterns

> **When to read:** When constructing prompts, managing tokens, selecting models, or tuning LLM behavior

---
```

- [ ] **Step 3: Create empty core domain file templates (7 files)**

Same header template for: `external-apis.md`, `data-pipelines.md`, `figure-standards.md`, `python-patterns.md`, `debugging-patterns.md`, `scientific-analysis.md`, `spatial-biology.md`. These will be populated in Task 2.

- [ ] **Step 4: Create audit artifacts**

```bash
# .last-audit — set to now so audit doesn't trigger immediately
echo "$(date +%s) initial-setup" > ~/.claude/knowledge/.last-audit
```

```markdown
# Knowledge Audit Log

> Append-only. Rotated at 50 entries (older moved to .audit-log-archive.md).

---

### [2026-03-17] Initial setup
**Action:** Created knowledge directory with 15 domain files (7 core, 7 growth, 1 skills)
**Result:** System initialized, first audit scheduled in 7 days
```

- [ ] **Step 5: Verify all files created**

```bash
ls -la ~/.claude/knowledge/
# Expected: 15 .md domain files + .last-audit + .audit-log.md = 17 items
wc -l ~/.claude/knowledge/*.md
# Expected: each file ~8-10 lines (header template only)
```

- [ ] **Step 6: Commit**

```bash
# Nothing to commit in git — these are in ~/.claude/ which is outside any repo
# Just verify the files are in place
```

---

### Task 2: Seed Core Domains from Existing .living/ Files

**Files:**
- Read: `Science/.living/learnings.md` (199 lines, 27 entries)
- Read: `Science/.living/conventions.md` (52 lines)
- Read: `Science/AutoReview/.living/learnings.md` (47 lines)
- Read: `Science/AutoReview/.living/conventions.md` (889 lines)
- Read: `Science/Autonomous Science/.living/learnings.md` (1,105 lines)
- Read: `Science/Gastruloids/L Metric/.living/learnings.md` (1,368 lines)
- Read: `Science/Gastruloids/L Metric/.living/conventions.md` (1,204 lines)
- Read: `Science/SpaceBar/Code/.living/learnings.md` (185 lines)
- Modify: 7 core domain files in `~/.claude/knowledge/`

**Important:** This is a sonnet-level task. The subagent must read existing `.living/` files, identify entries that are transferable (not project-specific), classify each into the correct domain, and reformat to the new entry template. This requires semantic judgment.

- [ ] **Step 1: Seed `external-apis.md`**

Read cross-project learnings and project-local learnings. Extract entries about:
- Elsevier/ScienceDirect download blocking
- Semantic Scholar API rate limits
- PubMed/OpenAlex API patterns
- API key management (S2_API_KEY, ELSEVIER_API_KEY)
- Exponential backoff with jitter

Reformat each to new template with `when_useful` trigger. Set `status: active`, `last-validated: 2026-03-17`.

- [ ] **Step 2: Seed `data-pipelines.md`**

Extract entries about:
- DAG snapshot/restart patterns (AutoReview)
- Multi-agent orchestration (Autonomous Science)
- Stage isolation and error propagation
- Pipeline state management

- [ ] **Step 3: Seed `figure-standards.md`**

Extract entries about:
- Colorblind-safe palette (`["#0072B2", "#D55E00", ...]`)
- DPI requirements (300 min raster, prefer vector)
- Typography standards (Arial/Helvetica, 12pt labels)
- Journal-specific formatting (Nature, Cell)
- Matplotlib patterns (`constrained_layout=True`, `matplotlib.use('Agg')`)

- [ ] **Step 4: Seed `python-patterns.md`**

Extract entries about:
- Async I/O patterns
- Pydantic model conventions
- Structlog over print()
- Type hint conventions
- Python 3.11+ features used

- [ ] **Step 5: Seed `debugging-patterns.md`**

Extract entries about:
- Conda environment issues
- Matplotlib backend errors
- pytest test discovery
- Import path issues
- Data loading failures

- [ ] **Step 6: Seed `scientific-analysis.md`**

Extract entries about:
- Statistical test selection
- Normalization methods
- QC steps and thresholds
- Multiple testing correction

- [ ] **Step 7: Seed `spatial-biology.md`**

Extract entries about:
- AnnData/scanpy patterns
- Spatial transcriptomics workflows
- Gene co-expression analysis
- Tissue morphology metrics

- [ ] **Step 8: Verify seeded domains**

```bash
wc -l ~/.claude/knowledge/*.md
# Core domains should now have 15-80 lines each
# Growth domains should still be ~8-10 lines (templates only)
```

---

### Task 3: Generate INDEX.md for All Projects

**Files:**
- Create: `Science/.living/INDEX.md`
- Create: `Science/AutoReview/.living/INDEX.md`
- Create: `Science/Autonomous Science/.living/INDEX.md`
- Create: `Science/Gastruloids/L Metric/.living/INDEX.md`
- Create: `Science/SpaceBar/Code/.living/INDEX.md`

**Important:** Each INDEX.md is generated by reading the `.living/` directory for that project, counting entries in each file, extracting key topics from headers, and recording last-modified dates. This is a sonnet task due to the topic extraction.

- [ ] **Step 1: Generate Science portfolio INDEX.md**

Read `Science/.living/` files. Count entries (### headers) in learnings.md and decisions.md. Count sections in conventions.md. Extract top 3-5 topic keywords from each file. Record `stat -f %Sm` modified dates.

Output format:
```markdown
# .living/ Index
Last audit: 2026-03-17

| File | Entries | Last updated | Key topics |
|------|---------|-------------|------------|
| conventions.md | N sections | YYYY-MM-DD | topic1, topic2, topic3 |
| decisions.md | N entries | YYYY-MM-DD | topic1, topic2, topic3 |
| learnings.md | N entries | YYYY-MM-DD | topic1, topic2, topic3 |
| cross-project-index.md | N projects | YYYY-MM-DD | project registry |
| knowledge-flows.md | — | YYYY-MM-DD | propagation rules |

## Local skills
See `.living/skills/` for project-specific skill packs.
```

- [ ] **Step 2: Generate AutoReview INDEX.md**

Same process for `Science/AutoReview/.living/`.

- [ ] **Step 3: Generate Autonomous Science INDEX.md**

Same process. Note: this project also has `ANALYSIS_MANIFEST.md` (10,705 lines) — include as a row in the index with a note about its size.

- [ ] **Step 4: Generate Gastruloids/L Metric INDEX.md**

Same process. Note: this project has image-analysis skills under `.living/skills/` — include a "Local skills" section.

- [ ] **Step 5: Generate SpaceBar/Code INDEX.md**

Same process.

- [ ] **Step 6: Verify all INDEX.md files**

```bash
for dir in \
  "/Users/mst36/Desktop/Projects/Science/.living" \
  "/Users/mst36/Desktop/Projects/Science/AutoReview/.living" \
  "/Users/mst36/Desktop/Projects/Science/Autonomous Science/.living" \
  "/Users/mst36/Desktop/Projects/Science/Gastruloids/L Metric/.living" \
  "/Users/mst36/Desktop/Projects/Science/SpaceBar/Code/.living"; do
  if [ -f "$dir/INDEX.md" ]; then
    echo "OK: $dir/INDEX.md ($(wc -l < "$dir/INDEX.md") lines)"
  else
    echo "MISSING: $dir/INDEX.md"
  fi
done
# Expected: all OK, each 10-20 lines
```

- [ ] **Step 7: Commit INDEX.md files**

```bash
cd /Users/mst36/Desktop/Projects/Science
git add .living/INDEX.md
git add "AutoReview/.living/INDEX.md"
git add "Autonomous Science/.living/INDEX.md"
git add "Gastruloids/L Metric/.living/INDEX.md"
git add "SpaceBar/Code/.living/INDEX.md"
git commit -m "feat: add .living/INDEX.md for progressive disclosure (auto-generated)"
```

---

### Task 4: Create Skills Routing Table

**Files:**
- Create: `~/.claude/knowledge/skills.md`

**Important:** This is the most critical domain file — it routes agents to the right skills. Mycelium skills listed first (closed system), then all other available skills grouped by domain.

- [ ] **Step 1: Inventory installed skills**

Read the skill list from the session's available skills (already visible in system reminders). Categorize by domain.

- [ ] **Step 2: Write skills.md**

```markdown
# Skills

> **When to read:** When starting a task, considering which tool/skill to invoke, or when the mycelium system needs self-maintenance

---

## Mycelium System Skills (self-maintaining)

| Skill | Trigger | What it does |
|-------|---------|-------------|
| mycelium | "set up mycelium", "initialize living repo", "crystallize learnings" | Scaffolds/maintains living repository framework |
| deep-memory-init | First session in new project, creating CLAUDE.md | Loads full user profile and project conventions |
| project-navigator | "switch to X", "cd to project" | Cross-project navigation with context loading |
| ship | "ship it", "commit and push" | Lint, test, commit, push workflow |

## Pipeline & Execution

| Skill | Trigger | What it does |
|-------|---------|-------------|
| run-local | "run pipeline", "run locally" | AutoReview pipeline via subagents |
| run-pipeline-local | "run pipeline" (Autonomous Science) | Autonomous Science pipeline |
| pipeline-runner | "run full pipeline", "execute all stages" | Generic sequential stage runner |
| debug-pipeline | Pipeline failures, snapshot inspection | Diagnose and resume pipeline runs |
| add-pipeline-node | Adding new DAG nodes | Step-by-step checklist |
| operating-dag-pipelines | 5+ stage DAG operations | Prevents context bloat |
| run-debug-fix-loop | Code fails, needs iterative fixing | Structured debug loop |

## Research & Analysis

| Skill | Trigger | What it does |
|-------|---------|-------------|
| research-lookup | Research questions, literature search | Perplexity Sonar Pro via OpenRouter |
| perplexity-search | Web search for current info | AI-powered search with citations |
| fetch-papers | DOIs, "download paper" | PDF retrieval with institutional VPN |
| scientific-brainstorming | Open-ended research ideation | Interdisciplinary exploration |
| scientific-critical-thinking | Evaluating claims, evidence quality | GRADE/Cochrane frameworks |
| exploratory-data-analysis | Analyzing scientific data files | 200+ format EDA reports |
| statistical-analysis | Test selection, power analysis | Guided analysis with APA reporting |
| analytics-audit | "audit", "check best practices" | Statistical rigor validation |

## Writing & Publishing

| Skill | Trigger | What it does |
|-------|---------|-------------|
| scientific-writing | Manuscript drafting, IMRAD | Two-stage outline → prose |
| peer-review | Formal manuscript review | Checklist-based evaluation |
| scholar-evaluation | Quantitative scholarly assessment | ScholarEval framework scoring |
| research-grants | NSF/NIH/DOE proposals | Agency-specific formatting |
| convert-to-pdf | "make a PDF" | Markdown/pipeline output → PDF |
| pdf | Any .pdf file operations | Read, merge, split, OCR PDFs |

## Visualization

| Skill | Trigger | What it does |
|-------|---------|-------------|
| figure-generator | Creating plots, charts | Publication-quality with enforced standards |
| scientific-visualization | Journal submission figures | Multi-panel, significance annotations |
| programmatic-schematics | Architecture diagrams | Matplotlib-based engine |
| scientific-schematics | Neural nets, flowcharts, pathways | Nano Banana Pro AI generation |
| generate-image | Photos, illustrations, artwork | FLUX/Gemini AI generation |
| infographics | Professional infographics | 10 types, 8 industry styles |

## Presentations & Dissemination

| Skill | Trigger | What it does |
|-------|---------|-------------|
| scientific-slides | Conference talks, seminars | PowerPoint/Beamer slide structure |
| pptx | Any .pptx operations | Create, read, edit presentations |
| paper-2-web | Paper → website/video/poster | Academic dissemination formats |

## Development

| Skill | Trigger | What it does |
|-------|---------|-------------|
| write-tests | Writing AutoReview tests | Project conventions, mock patterns |
| skill-creator | Creating/modifying skills | Skill creation + eval benchmarking |
| claude-api | Anthropic SDK code | API/SDK usage patterns |
| update-config | settings.json changes | Hooks, permissions, env vars |
| format-output | Citation pipeline, templates | AutoReview output formatting |

## Quality & Review

| Skill | Trigger | What it does |
|-------|---------|-------------|
| quality-evaluator | "evaluate this", "score output" | Structured quality assessment |
| model-comparison | "compare sonnet vs opus" | LLM output comparison |
| cost-tracker | Token usage, cost breakdown | Usage analysis across runs |
| find-redundancies | Token waste identification | Session transcript optimization |
| simplify | Review changed code | Reuse, quality, efficiency check |
```

- [ ] **Step 3: Verify skills.md**

```bash
wc -l ~/.claude/knowledge/skills.md
# Expected: ~120-140 lines
grep -c "^|" ~/.claude/knowledge/skills.md
# Expected: ~40+ table rows (one per skill)
```

---

### Task 5: Add Global Knowledge Domain Table to MEMORY.md Files

**Files:**
- Modify: `~/.claude/projects/-Users-mst36-Desktop-Projects-Science-AutoReview/memory/MEMORY.md` (5 lines)
- Modify: `~/.claude/projects/-Users-mst36-Desktop-Projects-Science-Autonomous-Science/memory/MEMORY.md` (252 lines)
- Modify: `~/.claude/projects/-Users-mst36-Desktop-Projects-Science/memory/MEMORY.md` (56 lines)

- [ ] **Step 1: Define the domain table block**

This exact block is appended to every MEMORY.md:

```markdown

## Global Knowledge Domains

Check domains relevant to your current task before starting work. Read the domain file only when needed.

| Domain | Summary | File |
|--------|---------|------|
| skills | Mycelium + all skills with trigger conditions | ~/.claude/knowledge/skills.md |
| external-apis | Rate limits, auth, backoff, publisher access | ~/.claude/knowledge/external-apis.md |
| data-pipelines | DAG design, snapshots, retry, orchestration | ~/.claude/knowledge/data-pipelines.md |
| figure-standards | Palettes, DPI, typography, journal formats | ~/.claude/knowledge/figure-standards.md |
| python-patterns | Async I/O, structlog, Pydantic, type hints | ~/.claude/knowledge/python-patterns.md |
| debugging-patterns | Conda, matplotlib, pytest, environment | ~/.claude/knowledge/debugging-patterns.md |
| scientific-analysis | Statistical tests, normalization, QC | ~/.claude/knowledge/scientific-analysis.md |
| spatial-biology | Spatial transcriptomics, AnnData, morphology | ~/.claude/knowledge/spatial-biology.md |
| llm-patterns | Prompt construction, tokens, model selection | ~/.claude/knowledge/llm-patterns.md |
| git-workflows | Worktrees, hooks, commits, PRs | ~/.claude/knowledge/git-workflows.md |
| environment-setup | Conda, env vars, VPN, institutional access | ~/.claude/knowledge/environment-setup.md |
| testing-patterns | pytest, mocking, async test setup | ~/.claude/knowledge/testing-patterns.md |
| data-formats | AnnData, JSON/YAML, PDF parsing | ~/.claude/knowledge/data-formats.md |
| publishing-workflows | PDF generation, citations, journal reqs | ~/.claude/knowledge/publishing-workflows.md |
| writing-conventions | Scientific prose, IMRAD, citation style | ~/.claude/knowledge/writing-conventions.md |
```

- [ ] **Step 2: Append to AutoReview MEMORY.md**

Read current file (5 lines), append the domain table block.

- [ ] **Step 3: Append to Autonomous Science MEMORY.md**

Read current file (252 lines), append the domain table block. Note: this file is large — use `printf >>` to append, do NOT read/rewrite.

- [ ] **Step 4: Append to Science portfolio MEMORY.md**

Read current file (56 lines), append the domain table block.

- [ ] **Step 5: Create MEMORY.md for Gastruloids project**

Claude Code creates `~/.claude/projects/` entries on first session open. Gastruloids and SpaceBar may not have entries yet. Check and create if missing:

```bash
GAST_DIR="$HOME/.claude/projects/-Users-mst36-Desktop-Projects-Science-Gastruloids-L-Metric/memory"
mkdir -p "$GAST_DIR"
```

Write a minimal MEMORY.md with the domain table:
```markdown
# Memory Index — Gastruloids / L Metric

## Global Knowledge Domains
[... same domain table as above ...]
```

- [ ] **Step 6: Create MEMORY.md for SpaceBar project**

```bash
SBAR_DIR="$HOME/.claude/projects/-Users-mst36-Desktop-Projects-Science-SpaceBar-Code/memory"
mkdir -p "$SBAR_DIR"
```

Write a minimal MEMORY.md with the domain table.

- [ ] **Step 7: Verify all MEMORY.md files have the domain table**

```bash
for f in \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-AutoReview/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-Autonomous-Science/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-Gastruloids-L-Metric/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-SpaceBar-Code/memory/MEMORY.md; do
  echo "=== $(basename $(dirname $(dirname $f))) ==="
  grep -c "Global Knowledge Domains" "$f" 2>/dev/null || echo "MISSING"
done
# Expected: 1 match per file (5 total)
```

---

### Task 6: Update CLAUDE.md Session-Start Instructions

**Files:**
- Modify: `/Users/mst36/Desktop/Projects/Science/CLAUDE.md` (lines 7-13)
- Modify: `/Users/mst36/Desktop/Projects/Science/AutoReview/CLAUDE.md` (lines ~38-46)

- [ ] **Step 1: Update Science/CLAUDE.md**

Replace the "Session Start — Mycelium Context Load" section:

**Before:**
```markdown
## Session Start — Mycelium Context Load

At the start of every session, silently read the following files to load project context (do NOT print a summary unless asked):

1. The current project's `.living/conventions.md`, `.living/decisions.md`, `.living/learnings.md`
2. The meta-level cross-project learnings at `Science/.living/learnings.md`
3. Any active domain skills listed in the project's `.living/skills/ACTIVE_SKILLS.yaml`

This ensures you always have the project's accumulated knowledge, conventions, and cross-project insights loaded before the user's first prompt.
```

**After:**
```markdown
## Session Start — Progressive Disclosure Context Load

At the start of every session, load context progressively (do NOT print a summary unless asked):

1. Read `.living/INDEX.md` — compact summary of project knowledge (entry counts, key topics, last-updated dates)
2. MEMORY.md is auto-loaded and contains the global knowledge domain table — check domains relevant to your current task
3. Read full `.living/` files (conventions.md, decisions.md, learnings.md) only when the current task touches those areas
4. Skills routing: check `~/.claude/knowledge/skills.md` when considering which skill to invoke (mycelium skills listed first)

**Fallback:** If `.living/INDEX.md` does not exist for the current project, fall back to reading full `.living/` files (legacy behavior). The weekly audit will generate INDEX.md progressively.

This ensures minimal context load at session start while keeping all project knowledge discoverable on demand.
```

- [ ] **Step 2: Update AutoReview/CLAUDE.md**

Replace the "Living Repository Protocol" section:

**Before:**
```markdown
## Living Repository Protocol

Read `.living/` before starting work:
- `.living/decisions.md` — project decisions log
- `.living/learnings.md` — lessons learned
- `.living/conventions.md` — project-specific conventions
```

**After:**
```markdown
## Living Repository Protocol

Read `.living/INDEX.md` for a compact summary of project knowledge before starting work. Read full files only when the current task touches those areas:
- `.living/conventions.md` — project-specific conventions (read when writing code)
- `.living/decisions.md` — project decisions log (read when making architectural choices)
- `.living/learnings.md` — lessons learned (read when debugging or encountering known issues)
```

- [ ] **Step 3: Verify sonnet subagent convention is in place**

The `.living/` update convention was already changed from haiku to sonnet in both `~/.claude/CLAUDE.md` and `Science/CLAUDE.md` (done during spec phase). Verify:

```bash
grep -c "sonnet subagent" /Users/mst36/.claude/CLAUDE.md
# Expected: 1
grep -c "sonnet subagent" /Users/mst36/Desktop/Projects/Science/CLAUDE.md
# Expected: 1
```

- [ ] **Step 4: Commit CLAUDE.md changes**

```bash
cd /Users/mst36/Desktop/Projects/Science
git add CLAUDE.md AutoReview/CLAUDE.md
git commit -m "docs: update session-start to progressive disclosure (INDEX.md + on-demand loading)"
```

---

### Task 7: Add Audit Trigger to mycelium-health.sh

**Files:**
- Modify: `/Users/mst36/tools/mycelium/skills/core/hooks/mycelium-health.sh` (69 lines)

- [ ] **Step 1: Read the current hook**

Read full file at `/Users/mst36/tools/mycelium/skills/core/hooks/mycelium-health.sh`.

- [ ] **Step 2: Restructure the hook to accumulate messages and emit one combined JSON**

The hook must output at most ONE JSON object. Restructure it to collect all messages into a variable and emit a combined `additionalContext` at the end. Replace the entire hook with this logic:

1. Initialize `MESSAGES=""` at the top (after `set -euo pipefail` and input parsing)
2. Knowledge audit check (runs before `SOURCE == "startup"` gate):
   - If `~/.claude/knowledge/` missing: append init instruction to `MESSAGES`
   - If `.last-audit` is >7 days old: append audit-due instruction to `MESSAGES`
   - If no `.last-audit`: create it with current timestamp
3. The existing `.living/` health checks: instead of emitting JSON directly, append warnings to `MESSAGES`
4. At the very end of the script, if `MESSAGES` is non-empty, emit one combined JSON:

```bash
# --- At top of script, after INPUT=$(cat) ---
MESSAGES=""

# --- Knowledge audit check (before SOURCE gate) ---
KNOWLEDGE_DIR="$HOME/.claude/knowledge"
if [ ! -d "$KNOWLEDGE_DIR" ]; then
  MESSAGES="${MESSAGES}KNOWLEDGE SYSTEM INIT: ~/.claude/knowledge/ does not exist. Create the directory and initialize domain files per the progressive disclosure spec. Run this silently as a background task.\n\n"
elif [ -f "$KNOWLEDGE_DIR/.last-audit" ]; then
  AUDIT_TS=$(awk '{print $1}' "$KNOWLEDGE_DIR/.last-audit")
  NOW_TS=$(date +%s)
  AUDIT_AGE_DAYS=$(( (NOW_TS - AUDIT_TS) / 86400 ))
  if [ "$AUDIT_AGE_DAYS" -ge 7 ]; then
    MESSAGES="${MESSAGES}KNOWLEDGE AUDIT DUE: Last audit ${AUDIT_AGE_DAYS} days ago. Dispatch a sonnet subagent in the background to run the knowledge audit (read ~/.claude/knowledge/ domain files, check staleness, regenerate INDEX.md files). Do this silently alongside your current task.\n\n"
  fi
else
  echo "$(date +%s) auto-initialized" > "$KNOWLEDGE_DIR/.last-audit"
fi

# --- Existing .living/ checks (modify to append to MESSAGES instead of emitting JSON) ---
# Replace each `cat <<JSON ... JSON` block with:
#   MESSAGES="${MESSAGES}WARNING TEXT HERE\n\n"

# --- At very end of script ---
if [ -n "$MESSAGES" ]; then
  # Escape for JSON
  ESCAPED=$(printf '%s' "$MESSAGES" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null)
  printf '{"additionalContext": %s}\n' "$ESCAPED"
fi
exit 0
```

The key change: all warnings/instructions accumulate in `$MESSAGES`, and one single JSON is emitted at the end.

- [ ] **Step 3: Verify the hook still works**

```bash
# Test with a mock JSON input
echo '{"cwd": "/Users/mst36/Desktop/Projects/Science/AutoReview", "source": "startup"}' | bash /Users/mst36/tools/mycelium/skills/core/hooks/mycelium-health.sh
# Expected: either silent (all healthy) or JSON additionalContext if audit due
echo $?
# Expected: 0
```

- [ ] **Step 4: Commit hook change**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/hooks/mycelium-health.sh
git commit -m "feat: add weekly knowledge audit trigger to SessionStart hook"
```

---

### Task 8: Validation and Final Commit

**Files:**
- Verify: All `~/.claude/knowledge/*.md` files (15 domains + audit artifacts)
- Verify: All `.living/INDEX.md` files (5 projects)
- Verify: All MEMORY.md files (3 projects with domain tables)
- Verify: Updated CLAUDE.md files (2 files)
- Verify: Updated mycelium-health.sh hook

- [ ] **Step 1: Verify knowledge directory completeness**

```bash
echo "=== Knowledge files ==="
ls ~/.claude/knowledge/*.md | wc -l
# Expected: 15 domain files + 1 audit log = 16

echo "=== Core domains populated ==="
for f in external-apis data-pipelines figure-standards python-patterns debugging-patterns scientific-analysis spatial-biology; do
  lines=$(wc -l < ~/.claude/knowledge/$f.md)
  echo "$f.md: $lines lines"
done
# Expected: each >15 lines (header + seeded entries)

echo "=== Skills routing table ==="
wc -l ~/.claude/knowledge/skills.md
# Expected: ~120-140 lines
```

- [ ] **Step 2: Verify INDEX.md files**

```bash
for dir in \
  "/Users/mst36/Desktop/Projects/Science/.living" \
  "/Users/mst36/Desktop/Projects/Science/AutoReview/.living" \
  "/Users/mst36/Desktop/Projects/Science/Autonomous Science/.living" \
  "/Users/mst36/Desktop/Projects/Science/Gastruloids/L Metric/.living" \
  "/Users/mst36/Desktop/Projects/Science/SpaceBar/Code/.living"; do
  if [ -f "$dir/INDEX.md" ]; then
    echo "OK: $dir/INDEX.md ($(wc -l < "$dir/INDEX.md") lines)"
  else
    echo "MISSING: $dir/INDEX.md"
  fi
done
# Expected: all OK, each 10-20 lines
```

- [ ] **Step 3: Verify MEMORY.md domain tables**

```bash
for f in \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-AutoReview/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science-Autonomous-Science/memory/MEMORY.md \
  ~/.claude/projects/-Users-mst36-Desktop-Projects-Science/memory/MEMORY.md; do
  if grep -q "Global Knowledge Domains" "$f" 2>/dev/null; then
    echo "OK: $f"
  else
    echo "MISSING TABLE: $f"
  fi
done
# Expected: all OK
```

- [ ] **Step 4: Verify CLAUDE.md updates**

```bash
grep -c "Progressive Disclosure" /Users/mst36/Desktop/Projects/Science/CLAUDE.md
# Expected: 1
grep -c "INDEX.md" /Users/mst36/Desktop/Projects/Science/AutoReview/CLAUDE.md
# Expected: 1
```

- [ ] **Step 5: Test hook**

```bash
echo '{"cwd": "/Users/mst36/Desktop/Projects/Science/AutoReview", "source": "startup"}' | bash /Users/mst36/tools/mycelium/skills/core/hooks/mycelium-health.sh
echo "Exit code: $?"
# Expected: exit 0, possibly with additionalContext JSON
```

- [ ] **Step 6: Final commit for any remaining changes**

```bash
cd /Users/mst36/Desktop/Projects/Science
git add -A .living/INDEX.md */.living/INDEX.md
git status
git commit -m "feat: progressive disclosure knowledge system — initial setup complete"
```
