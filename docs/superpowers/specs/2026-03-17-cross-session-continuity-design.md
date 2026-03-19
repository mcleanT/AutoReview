# Cross-Session Continuity — Design Spec

**Goal:** Make session-to-session context preservation a native mycelium feature, so every mycelium-enabled repo automatically saves and restores session context without manual intervention.

**Status:** Design approved, pending implementation.

---

## Problem

Claude Code sessions are ephemeral — when a session ends, the agent loses all context about what was done, what state the project is in, and what should happen next. Currently, session continuity in the user's environment relies on manually-maintained hooks (`session-resume.sh`, `session-save-reminder.sh`) that are not part of mycelium and must be manually installed per-project.

The `.living/` files (decisions.md, learnings.md) capture permanent project knowledge, but they're too large to load at session start (~1,000-4,300 lines). The progressive disclosure system (INDEX.md + MEMORY.md domain table) solves the discovery problem, but doesn't address **session-specific orientation** — "what was I doing, what's the current state, what's next."

## Solution

Integrate session summary save/resume into mycelium's existing hook infrastructure:

1. **Crystallization subagent** writes `.claude/last-session.md` as part of the existing post-batch crystallization step
2. **`mycelium-health.sh`** (SessionStart) loads the summary into `additionalContext` and displays it to the user
3. **`mycelium-stop-check.sh`** (Stop) warns if significant work was done but no summary was written

No new scripts, files, or modes — this extends three existing components.

---

## Session Summary Format

The session summary uses a 5-section structured template. The crystallization subagent writes this file at `.claude/last-session.md`:

```markdown
SESSION RESUME — Last session (YYYY-MM-DD HH:MM):

## What was worked on
- [Semantic summary of accomplishments — what was built/fixed/analyzed, not file lists]

## Key decisions made
- [Decision]: [rationale] (see .living/decisions.md for full context)

## Blockers & surprises
- [Resolved/Unresolved]: [what happened, resolution or current status]

## Current state
- Branch: X | Tests: N passing | [environment notes]
- [Key metrics, uncommitted changes, data state]

## Next steps
- [Actionable items with specific commands where relevant]
```

### Design Rationale for Each Section

Based on cross-cutting analysis of persistent memory systems (claude-mem, memory-mcp, Cog, memsearch, Claude Code official session memory), the most valuable information to persist ranked by impact:

1. **Decisions and rationale** — prevents re-litigating past choices
2. **Debugging solutions / error patterns** — prevents rediscovery of fixes
3. **Gotchas and non-obvious constraints** — captures surprises
4. **Architecture understanding gained** — orients the agent in the codebase
5. **Progress / what was done** — needed for orientation but least durable

The 5-section template maps to this hierarchy:
- "What was worked on" → #5 (orientation)
- "Key decisions made" → #1 (compressed rationale, points to `.living/decisions.md`)
- "Blockers & surprises" → #2 + #3 (debugging solutions and gotchas)
- "Current state" → #4 (grounded in git/test facts)
- "Next steps" → actionable intent (requires agent judgment, not scriptable)

### Progressive Disclosure Integration

The session summary is a **Tier 0** entry point — even lighter than INDEX.md:
- **Tier 0**: `.claude/last-session.md` (~15 lines, loaded automatically at SessionStart)
- **Tier 1**: MEMORY.md domain table (~20 lines, always in context)
- **Tier 2**: `~/.claude/knowledge/{domain}.md` (on demand)
- **Tier 3**: `.living/` full files (on demand)

The session summary's "Key decisions" and "Blockers & surprises" sections point to `.living/` for full context, maintaining the pull-on-demand pattern.

---

## Component Changes

### 1. `commands/core.md` — Skill Definition

**Location:** Post-Action Hook Protocol section and Subagent-Driven Sessions section.

**Changes:**

Add session summary writing to the crystallization protocol. The crystallization subagent's mandate expands from:
- Update `.living/decisions.md` and `.living/learnings.md`

To:
- Update `.living/decisions.md` and `.living/learnings.md`
- Write `.claude/last-session.md` using the 5-section template

**Specific edit locations in `commands/core.md`:**

1. **Subagent-Driven Sessions section** (lines 257-259): Expand bullet 2 from:
   > "Appends entries to `.living/learnings.md` and `.living/decisions.md`"

   To:
   > "Appends entries to `.living/learnings.md` and `.living/decisions.md`"
   > "Writes `.claude/last-session.md` using the 5-section session summary template (see below)"

2. **Post-Action Hook Protocol section**: Add the session summary template and full-session coverage rule after the existing knowledge promotion step.

**Full-session coverage rule:** The subagent must summarize ALL work since session start (using the mtime of `.claude/session-start-ts.tmp`), not just the most recent batch. It should run `git log --since=<timestamp>` and review `.living/` diffs from that timestamp forward. If crystallization fires multiple times in a session, each write rebuilds the summary covering the entire session — earlier work plus new work. The summary gets more expansive as the session progresses.

**Subagent brief template** (documented in skill definition):

> "Summarize ALL work since session start (timestamp: {session_start_ts}). Update `.living/decisions.md` and `.living/learnings.md` with new entries. Then write `.claude/last-session.md` with 5 sections: What was worked on (semantic), Key decisions made (with rationale), Blockers & surprises (resolved/unresolved), Current state (branch, tests, environment), Next steps (actionable). Run `git log --since={session_start_ts}` and `git diff --stat` to ground your summary in facts. Check `.living/` file diffs for new entries to reference."

### 2. `skills/core/hooks/mycelium-health.sh` — SessionStart Hook

**Current behavior** (actual code execution order):
1. Record session-start timestamp to `.claude/session-start-ts.tmp`
2. Check knowledge audit staleness (7-day cycle)
3. Guard: if `SOURCE != startup`, exit early (only full checks on fresh session start)
4. Check `.living/` completeness (decisions.md, learnings.md, conventions.md)
5. Emit single `additionalContext` JSON with accumulated MESSAGES

**Added behavior:**

Insert after step 3 (the `SOURCE != startup` guard, line 50 in current code) and before step 4 (`.living/` directory check, line 59):
- Check if `.claude/last-session.md` exists
- Check if file modification time is within the last 7 days (use the existing macOS/Linux `stat` dual-syntax pattern from `mycelium-stop-check.sh` lines 48-56; compare file mtime against `$(date +%s) - 604800`)
- If both true: read file contents and prepend to MESSAGES accumulator as a `SESSION RESUME` block
- If file is empty or missing: skip silently

**Important:** The session resume content goes into the `additionalContext` JSON via the MESSAGES accumulator — NOT as separate plain-text stdout. Claude Code SessionStart hooks must emit valid JSON on stdout. The user sees the resume because the agent displays `additionalContext` content as part of its first response. To also show the user the resume before the agent responds, write to **stderr** (which Claude Code displays in the terminal without parsing as JSON).

**Output format (what the agent receives in additionalContext):**

```
SESSION RESUME — Last session (2026-03-17 22:27):
## What was worked on
- Refactored search infrastructure to support 6 backends...

## Key decisions made
- Removed Perplexity backend: unreliable results, high cost vs CORE/CrossRef

## Blockers & surprises
- Resolved: Europe PMC rate limit (429s) — added exponential backoff

## Current state
- Branch: main | Tests: 1028 passing | All changes uncommitted

## Next steps
- Add ANTHROPIC_API_KEY and S2_API_KEY to .env
- Commit all search infrastructure changes
- Run benchmark
```

**What the user sees in their terminal (stderr, displayed immediately):**

The same content written to stderr so it appears in the terminal before the agent responds.

**Edge cases:**
- No `.claude/last-session.md` → nothing added (first session or file expired)
- File >7 days old → ignored (stale context is worse than no context)
- File exists but empty → ignored
- `.claude/` directory doesn't exist → nothing added

### 3. `skills/core/hooks/mycelium-stop-check.sh` — Stop Hook

**Current behavior:**
1. Check if `mycelium-reminded.tmp` exists (was significant work done?)
2. If yes, check if `.living/` was updated after the reminder
3. If not updated → exit 0 (non-blocking; the current implementation cleans up and exits silently — blocking output format is described in comments but not implemented)
4. If updated or no work done → exit 0

**Note:** The current stop hook does NOT actually block (exit 2) when `.living/` is not updated — it exits 0 in all paths. The blocking behavior described in the hook's header comments is aspirational but not implemented. This spec does not change that behavior; it only adds the session summary warning to the "both checks passed" path.

**Added behavior:**

In the code path where `.living/` was updated (the "both checks passed" branch, after line 65 in current code), before the final exit 0:
- Compare the mtime of `.claude/last-session.md` against the mtime of `.claude/session-start-ts.tmp` (which was created at session start, so its mtime represents the session start time). Use the existing macOS/Linux `stat` dual-syntax pattern already in the codebase (lines 48-56).
- If `last-session.md` doesn't exist or its mtime is older than `session-start-ts.tmp` → emit a **non-blocking warning** (exit 0 + stdout message):

```
Session summary not written. Next session will lack context.
Dispatch crystallization subagent or write .claude/last-session.md before stopping.
```

- If `last-session.md` is newer → clean exit, no message

**Design choice:** This is a warning (exit 0), not a block (exit 2). The `.living/` update is the more important gate; the session summary is a convenience feature. Blocking on it would be too aggressive for short or exploratory sessions.

---

## What Does NOT Change

- **No new files or scripts** — no `generate_session_summary.py`, no new templates
- **No changes to `init` mode** — `.claude/last-session.md` is created organically by the first crystallization, not pre-scaffolded
- **No changes to `init_knowledge.py` or `generate_index.py`**
- **No changes to templates or `domains.yaml`**
- **No hook registration changes** — same hooks, same events, same settings.local.json entries
- **No new modes** — this is an extension of the existing crystallization protocol, not a new skill mode

## Prerequisites / Assumptions

- **`.claude/` is gitignored.** The session summary file (`.claude/last-session.md`) and temp files (`.claude/session-start-ts.tmp`, `.claude/mycelium-reminded.tmp`) contain machine-specific, session-specific content and must not be committed. Mycelium's `init` mode already adds `.claude/` to `.gitignore` via `init_repo.py`. If a repo was set up before this convention, the implementer should verify `.claude/` is in `.gitignore`.

---

## Summary File Lifecycle

```
Session start:
  mycelium-health.sh reads .claude/last-session.md
  → additionalContext JSON (agent sees SESSION RESUME)
  → stderr (user sees resume in terminal)
  → records session-start-ts.tmp

During session:
  Work happens → post-action fires → .living/ updated

Crystallization (after significant work):
  Subagent writes .living/decisions.md, .living/learnings.md
  Subagent writes .claude/last-session.md (5-section template)
  Summary covers ALL work since session-start-ts.tmp
  If crystallization fires again later, summary is rebuilt (full session)

Session end:
  mycelium-stop-check.sh verifies:
    1. .living/ updated (current behavior: non-blocking check)
    2. last-session.md written (warning if not, does not block)
  .claude/mycelium-reminded.tmp cleaned up

Next session:
  mycelium-health.sh loads last-session.md → cycle repeats
```

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| First session in a new repo | No `.claude/last-session.md` exists → no resume, no warning |
| Short exploratory session (no significant work) | `mycelium-reminded.tmp` never created → no checks at stop |
| Session crashes without clean stop | Summary from last successful crystallization persists; next session loads it |
| Multiple crystallizations in one session | Each overwrites with full-session summary (cumulative, not incremental) |
| Summary file >7 days old | Ignored at SessionStart (stale context is worse than no context) |
| Read-only session (only file reads, no code execution) | No post-action trigger → no crystallization → no summary expected |
| `.claude/` directory doesn't exist | Hook checks silently skip (no errors) |

---

## Scope

**Total changes:** ~55 lines across 3 existing files.

| File | Lines Added | Nature of Change |
|------|-------------|------------------|
| `commands/core.md` | ~30 | Document session summary template, update crystallization protocol |
| `skills/core/hooks/mycelium-health.sh` | ~15 | Load and display `.claude/last-session.md` at session start |
| `skills/core/hooks/mycelium-stop-check.sh` | ~10 | Non-blocking warning if summary missing after significant work |
