# Progressive Disclosure Bootstrap — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make progressive disclosure fully self-bootstrapping from the mycelium skill — any user running `init` or `knowledge-init` gets the complete system without manual setup.

**Architecture:** Add domain file templates to the mycelium repo, Python scripts for knowledge init and INDEX.md generation, and a new `knowledge-init` mode in the skill. The existing `init` mode calls knowledge-init as a final step.

**Tech Stack:** Python 3.11+, Shell (hooks), Markdown templates

**Target repo:** `/Users/mst36/tools/mycelium` (branch: `feat/progressive-disclosure`)

---

## File Structure

### New Files (create)

| File | Purpose |
|------|---------|
| `skills/core/templates/knowledge/domain-header.md` | Template for each domain file header |
| `skills/core/templates/knowledge/entry-template.md` | Reference template for knowledge entries |
| `skills/core/templates/knowledge/domain-table.md` | MEMORY.md routing table block to append |
| `skills/core/templates/knowledge/domains.yaml` | Domain registry — names, descriptions, triggers |
| `skills/core/scripts/init_knowledge.py` | Creates `~/.claude/knowledge/`, generates domain files from templates, initializes audit artifacts |
| `skills/core/scripts/generate_index.py` | Generates `.living/INDEX.md` for a given project directory |

### Modified Files

| File | Change |
|------|--------|
| `commands/core.md` | Add `knowledge-init` mode, update `init` to call it |

---

## Task Dependency Graph

```
Task 1 (templates)  ──┐
                      ├──> Task 3 (init_knowledge.py)
Task 2 (domains.yaml) ┘                              ──> Task 5 (skill modes)
                                                      ──> Task 6 (commit)
Task 4 (generate_index.py) ───────────────────────────┘
```

**Batch 1 (independent):** Tasks 1, 2, 4
**Batch 2 (needs 1+2):** Task 3
**Batch 3 (needs all):** Tasks 5, 6

---

### Task 1: Create Knowledge Templates

**Files:**
- Create: `skills/core/templates/knowledge/domain-header.md`
- Create: `skills/core/templates/knowledge/entry-template.md`
- Create: `skills/core/templates/knowledge/domain-table.md`

- [ ] **Step 1: Create domain header template**

`skills/core/templates/knowledge/domain-header.md`:
```markdown
# {{DOMAIN_TITLE}}

> **When to read:** {{TRIGGER_DESCRIPTION}}

---
```

- [ ] **Step 2: Create entry template**

`skills/core/templates/knowledge/entry-template.md`:
```markdown
### [{{DOMAIN_TAG}}] {{SUMMARY}}
**What:** {{DESCRIPTION}}
**Evidence:** {{PROJECT}}, {{DATE}}, {{CONTEXT}}
**When useful:** {{TRIGGER_CONDITION}}
**Scope:** transferable
**Status:** unreviewed
**Last validated:** {{DATE}}
```

- [ ] **Step 3: Create MEMORY.md domain table template**

`skills/core/templates/knowledge/domain-table.md`:
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

- [ ] **Step 4: Commit templates**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/templates/knowledge/
git commit -m "feat: add knowledge system templates (domain header, entry, routing table)"
```

---

### Task 2: Create Domain Registry

**Files:**
- Create: `skills/core/templates/knowledge/domains.yaml`

- [ ] **Step 1: Create domains.yaml**

This file is the single source of truth for all domain definitions. The init script reads this to generate domain files.

```yaml
# Domain Registry — Progressive Disclosure Knowledge System
# Each domain becomes a file at ~/.claude/knowledge/{name}.md

domains:
  # Core domains (expected to accumulate entries quickly)
  - name: skills
    title: Skills
    trigger: "When starting a task, considering which tool/skill to invoke, or when the mycelium system needs self-maintenance"
    core: true
    note: "Skills domain uses table format, not entry template. Populated by skills sync during audit."

  - name: external-apis
    title: External APIs
    trigger: "When calling external APIs, handling rate limits, managing API keys, or accessing publisher content"
    core: true

  - name: data-pipelines
    title: Data Pipelines
    trigger: "When designing DAG pipelines, implementing snapshots/restart, or handling stage errors"
    core: true

  - name: figure-standards
    title: Figure Standards
    trigger: "When creating publication figures, choosing palettes, setting DPI, or formatting for journals"
    core: true

  - name: python-patterns
    title: Python Patterns
    trigger: "When writing async code, using Pydantic, configuring structlog, or applying type hints"
    core: true

  - name: debugging-patterns
    title: Debugging Patterns
    trigger: "When troubleshooting conda, matplotlib backends, pytest discovery, or import issues"
    core: true

  - name: scientific-analysis
    title: Scientific Analysis
    trigger: "When choosing statistical tests, normalizing data, running QC, or correcting for multiple comparisons"
    core: true

  - name: spatial-biology
    title: Spatial Biology
    trigger: "When working with spatial transcriptomics, AnnData, tissue morphology, or gene co-expression"
    core: true

  # Growth domains (start empty, accumulate over time)
  - name: llm-patterns
    title: LLM Patterns
    trigger: "When constructing prompts, managing tokens, selecting models, or tuning LLM behavior"
    core: false

  - name: git-workflows
    title: Git Workflows
    trigger: "When using worktrees, configuring hooks, managing commits, or creating PRs"
    core: false

  - name: environment-setup
    title: Environment Setup
    trigger: "When configuring conda, setting env vars, fixing PATH issues, or accessing VPN/institutional resources"
    core: false

  - name: testing-patterns
    title: Testing Patterns
    trigger: "When writing pytest tests, creating mocks, setting up async tests, or configuring test fixtures"
    core: false

  - name: data-formats
    title: Data Formats
    trigger: "When working with AnnData, JSON/YAML schemas, PDF parsing, or cross-project I/O patterns"
    core: false

  - name: publishing-workflows
    title: Publishing Workflows
    trigger: "When generating PDFs, formatting citations, meeting journal requirements, or preparing submissions"
    core: false

  - name: writing-conventions
    title: Writing Conventions
    trigger: "When writing scientific prose, structuring IMRAD sections, or applying citation styles"
    core: false
```

- [ ] **Step 2: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/templates/knowledge/domains.yaml
git commit -m "feat: add domain registry (15 domains with triggers and core flags)"
```

---

### Task 3: Create init_knowledge.py Script

**Files:**
- Create: `skills/core/scripts/init_knowledge.py`

- [ ] **Step 1: Write the script**

The script should:
1. Accept `--knowledge-dir` (default `~/.claude/knowledge/`) and `--mycelium-root` (for finding templates)
2. Read `domains.yaml` for the domain list
3. Create the knowledge directory if it doesn't exist
4. For each domain: generate `{name}.md` using the header template, skip if file already exists (preserve existing entries)
5. Create `.last-audit` with current timestamp if it doesn't exist
6. Create `.audit-log.md` with initial entry if it doesn't exist
7. Print summary of what was created vs skipped

Key behaviors:
- **Idempotent**: safe to run multiple times — never overwrites existing domain files
- **Template-driven**: reads `domain-header.md` template and `domains.yaml` for domain definitions
- The `skills.md` domain is special — it gets a different template (table-based, not entry-based). The script should create a minimal skills.md with the mycelium section header if it doesn't exist.

- [ ] **Step 2: Make executable**

```bash
chmod +x skills/core/scripts/init_knowledge.py
```

- [ ] **Step 3: Test manually**

```bash
# Test with a temp directory
python3 skills/core/scripts/init_knowledge.py --knowledge-dir /tmp/test-knowledge --mycelium-root /Users/mst36/tools/mycelium
ls /tmp/test-knowledge/*.md | wc -l  # Expected: 16 (15 domains + audit-log)
cat /tmp/test-knowledge/.last-audit  # Expected: timestamp
rm -rf /tmp/test-knowledge
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/scripts/init_knowledge.py
git commit -m "feat: add init_knowledge.py — bootstraps ~/.claude/knowledge/ from templates"
```

---

### Task 4: Create generate_index.py Script

**Files:**
- Create: `skills/core/scripts/generate_index.py`

- [ ] **Step 1: Write the script**

The script should:
1. Accept `--living-dir` (path to a `.living/` directory)
2. Scan all `.md` files in the directory
3. For each file: count `###` headers (entries) or `##` headers (sections for conventions.md), get last-modified date, extract top 3-5 topic keywords from headers
4. Handle both old entry format (`### [YYYY-MM-DD] Title`) and new format (`### [domain-tag] Title`)
5. Check for `.living/skills/` subdirectory and list skill packs
6. Write `INDEX.md` with the standard table format
7. Support `--dry-run` flag to print without writing

Key behaviors:
- **Idempotent**: overwrites INDEX.md each time (it's auto-generated)
- **Handles large files efficiently**: reads headers only, not full content (use line-by-line scanning)
- **Cross-platform dates**: use `os.path.getmtime()` not `stat` shell commands

- [ ] **Step 2: Make executable**

```bash
chmod +x skills/core/scripts/generate_index.py
```

- [ ] **Step 3: Test manually**

```bash
# Test against AutoReview
python3 skills/core/scripts/generate_index.py --living-dir /Users/mst36/Desktop/Projects/Science/AutoReview/.living --dry-run
# Expected: table with conventions.md, decisions.md, learnings.md rows
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/scripts/generate_index.py
git commit -m "feat: add generate_index.py — creates .living/INDEX.md from directory scan"
```

---

### Task 5: Update Skill Definition with knowledge-init Mode

**Files:**
- Modify: `commands/core.md`

- [ ] **Step 1: Add knowledge-init mode**

Add a new mode section after the existing `init` mode section:

```markdown
## Mode: `knowledge-init`

**Trigger**: "knowledge init", "set up knowledge", "initialize knowledge system", "progressive disclosure"

**Purpose**: Bootstrap the global progressive disclosure knowledge system (`~/.claude/knowledge/`).

**Steps**:
1. Run `skills/core/scripts/init_knowledge.py` to create domain files from templates. Existing files are preserved (idempotent).
2. For each project with a `.living/` directory: run `skills/core/scripts/generate_index.py` to create/update `.living/INDEX.md`.
3. Check each project's MEMORY.md (in `~/.claude/projects/*/memory/`). If missing the "Global Knowledge Domains" table, append it from `skills/core/templates/knowledge/domain-table.md`.
4. Verify the knowledge system is functional: check `~/.claude/knowledge/.last-audit` exists, confirm domain file count, report summary.

**Notes**:
- This mode is **global** — it sets up `~/.claude/knowledge/` which is shared across all projects.
- Safe to run multiple times. Existing domain files and their entries are never overwritten.
- The weekly audit (triggered by `mycelium-health.sh`) handles ongoing maintenance: staleness checks, INDEX.md regeneration, skills sync, dedup.
- Domain files start empty (header + trigger only). Entries accumulate through the post-action protocol's knowledge promotion step and the crystallize mode.
```

- [ ] **Step 2: Update init mode to call knowledge-init**

In the existing `init` mode steps, add a final step before validation:

```markdown
10. **Bootstrap knowledge system**: If `~/.claude/knowledge/` does not exist, run `skills/core/scripts/init_knowledge.py` to set up the global progressive disclosure knowledge system. Generate `.living/INDEX.md` for the newly scaffolded project. Append the domain routing table to the project's MEMORY.md if not already present.
```

(Renumber the existing step 10 to 11 and step 11 to 12.)

- [ ] **Step 3: Update hook table description**

Ensure the hooks table in the "Automated Enforcement" section has the updated `mycelium-health.sh` description (should already be there from earlier commit — verify).

- [ ] **Step 4: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add commands/core.md
git commit -m "feat: add knowledge-init mode, integrate knowledge bootstrap into init"
```

---

### Task 6: Final Validation and Commit

- [ ] **Step 1: Verify all new files exist**

```bash
cd /Users/mst36/tools/mycelium
echo "=== Templates ==="
ls skills/core/templates/knowledge/
echo "=== Scripts ==="
ls skills/core/scripts/init_knowledge.py skills/core/scripts/generate_index.py
echo "=== Modes in core.md ==="
grep "## Mode:" commands/core.md
```

- [ ] **Step 2: Test init_knowledge.py end-to-end**

```bash
python3 skills/core/scripts/init_knowledge.py --knowledge-dir /tmp/test-knowledge --mycelium-root /Users/mst36/tools/mycelium
ls /tmp/test-knowledge/*.md | wc -l
cat /tmp/test-knowledge/.last-audit
# Run again to verify idempotency
python3 skills/core/scripts/init_knowledge.py --knowledge-dir /tmp/test-knowledge --mycelium-root /Users/mst36/tools/mycelium
rm -rf /tmp/test-knowledge
```

- [ ] **Step 3: Test generate_index.py end-to-end**

```bash
python3 skills/core/scripts/generate_index.py --living-dir /Users/mst36/Desktop/Projects/Science/AutoReview/.living --dry-run
```

- [ ] **Step 4: Verify git log**

```bash
cd /Users/mst36/tools/mycelium
git log --oneline feat/progressive-disclosure ^origin/feature/post-action-hook-enforcement
```

- [ ] **Step 5: Push updated branch**

```bash
cd /Users/mst36/tools/mycelium
git push origin feat/progressive-disclosure
```
