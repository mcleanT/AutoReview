# Cross-Session Continuity — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make session-to-session context preservation a native mycelium feature by extending three existing files: the skill definition, the SessionStart hook, and the Stop hook.

**Architecture:** The crystallization subagent (already dispatched after significant work) gains the additional responsibility of writing `.claude/last-session.md` with a 5-section semantic summary. The SessionStart hook loads this file into `additionalContext` and displays it to the user via stderr. The Stop hook warns if the summary wasn't written.

**Tech Stack:** Bash (hooks), Markdown (skill definition)

**Target repo:** `/Users/mst36/tools/mycelium` (branch: `feat/cross-session-continuity`, off `feat/progressive-disclosure`)

**Spec:** `docs/superpowers/specs/2026-03-17-cross-session-continuity-design.md`

---

## File Structure

### Modified Files

| File | Change | Purpose |
|------|--------|---------|
| `skills/core/hooks/mycelium-health.sh` | Add session resume loading (~15 lines) | Load `.claude/last-session.md` at SessionStart into additionalContext + stderr |
| `skills/core/hooks/mycelium-stop-check.sh` | Add session summary warning (~10 lines) | Non-blocking warning if summary not written after significant work |
| `commands/core.md` | Add session summary template + update crystallization protocol (~30 lines) | Document the 5-section format and update subagent mandate |

### No New Files

This feature extends existing components only.

---

## Task Dependency Graph

```
Task 1 (health hook)  ──┐
                        ├──> Task 3 (skill definition) ──> Task 4 (manual test + commit)
Task 2 (stop hook)   ──┘
```

**Batch 1 (independent):** Tasks 1, 2
**Batch 2 (needs 1+2):** Task 3
**Batch 3 (needs all):** Task 4

---

### Task 1: Extend `mycelium-health.sh` with Session Resume Loading

**Files:**
- Modify: `skills/core/hooks/mycelium-health.sh:48-57` (insert between SOURCE guard and .living/ checks)

- [ ] **Step 1: Read the current file**

Read `skills/core/hooks/mycelium-health.sh` to understand the full structure. The insertion point is between line 57 (end of the early-exit `SOURCE != startup` block) and line 59 (`LIVING_DIR` assignment).

- [ ] **Step 2: Add session resume loading**

Insert the following block between line 57 (`fi` closing the SOURCE guard) and line 59 (`LIVING_DIR="$REPO_ROOT/.living"`):

```bash
# --- Session resume: load last-session.md if recent ---
SESSION_FILE="$REPO_ROOT/.claude/last-session.md"
if [ -f "$SESSION_FILE" ]; then
  SESSION_MTIME=$(stat -f "%m" "$SESSION_FILE" 2>/dev/null || stat -c "%Y" "$SESSION_FILE" 2>/dev/null || echo "0")
  NOW_TS=$(date +%s)
  SESSION_AGE_DAYS=$(( (NOW_TS - SESSION_MTIME) / 86400 ))
  if [ "$SESSION_AGE_DAYS" -lt 7 ]; then
    SESSION_CONTENT=$(cat "$SESSION_FILE")
    if [ -n "$SESSION_CONTENT" ]; then
      # Show resume to user immediately via stderr
      echo "$SESSION_CONTENT" >&2
      echo "---" >&2
      # Add to agent context via MESSAGES accumulator
      MESSAGES="${MESSAGES}${SESSION_CONTENT}\n\n"
    fi
  fi
fi
```

- [ ] **Step 3: Test the hook manually**

Create a test session file and run the hook:

```bash
cd /Users/mst36/tools/mycelium

# Create a mock .claude/last-session.md
mkdir -p .claude
cat > .claude/last-session.md << 'EOF'
SESSION RESUME — Last session (2026-03-17 22:27):

## What was worked on
- Added cross-session continuity to mycelium hooks

## Key decisions made
- Used stderr for user display, additionalContext for agent context

## Blockers & surprises
- None

## Current state
- Branch: feat/cross-session-continuity | Tests: N/A (shell scripts)

## Next steps
- Test the full flow end-to-end
EOF

# Test: pipe mock stdin JSON, verify output includes session content
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh 2>/dev/null
# Expected: JSON output with additionalContext containing the session resume text

# Test stderr output
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh 2>&1 1>/dev/null
# Expected: The session resume text printed to stderr

# Clean up test file
rm .claude/last-session.md
```

- [ ] **Step 4: Test edge cases**

```bash
cd /Users/mst36/tools/mycelium

# Test: no last-session.md (should produce no session resume)
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh
# Expected: no session resume in output (may have .living/ warnings, that's fine)

# Test: empty last-session.md (should be ignored)
touch .claude/last-session.md
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh
# Expected: no session resume in output
rm .claude/last-session.md

# Test: old last-session.md (>7 days, should be ignored)
cat > .claude/last-session.md << 'EOF'
SESSION RESUME — stale content
EOF
touch -t 202603010000 .claude/last-session.md  # Set to March 1
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh
# Expected: no session resume in output (file too old)
rm .claude/last-session.md
```

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add skills/core/hooks/mycelium-health.sh
git commit -m "feat: load .claude/last-session.md at SessionStart for cross-session continuity"
```

---

### Task 2: Extend `mycelium-stop-check.sh` with Session Summary Warning

**Files:**
- Modify: `skills/core/hooks/mycelium-stop-check.sh:62-66` (add check before exit 0 in the "both checks passed" branch)

- [ ] **Step 1: Read the current file**

Read `skills/core/hooks/mycelium-stop-check.sh` to understand the full structure. The insertion point is in the block at lines 62-66 where `.living/` was updated (the "success" path). Replace the entire block (including the closing `fi` on line 66) with the expanded version below.

- [ ] **Step 2: Add session summary warning**

Replace lines 62-66 (including the closing `fi`):

```bash
# If either was updated after the post-action hook fired, protocol was followed
if [ "$LEARNINGS_UPDATED" = true ] || [ "$DECISIONS_UPDATED" = true ]; then
  # Clean up reminder file — cycle complete
  rm -f "$REMINDER_FILE"
  exit 0
fi
```

With:

```bash
# If either was updated after the post-action hook fired, protocol was followed
if [ "$LEARNINGS_UPDATED" = true ] || [ "$DECISIONS_UPDATED" = true ]; then
  # Clean up reminder file — cycle complete
  rm -f "$REMINDER_FILE"

  # Check if session summary was written (non-blocking warning)
  SESSION_FILE="$REPO_ROOT/.claude/last-session.md"
  SESSION_START_FILE="$REPO_ROOT/.claude/session-start-ts.tmp"
  if [ -f "$SESSION_START_FILE" ]; then
    START_MTIME=$(stat -f "%m" "$SESSION_START_FILE" 2>/dev/null || stat -c "%Y" "$SESSION_START_FILE" 2>/dev/null || echo "0")
    SESSION_MTIME=$(stat -f "%m" "$SESSION_FILE" 2>/dev/null || stat -c "%Y" "$SESSION_FILE" 2>/dev/null || echo "0")
    if [ "$SESSION_MTIME" -lt "$START_MTIME" ] || [ ! -f "$SESSION_FILE" ]; then
      echo "Session summary not written. Next session will lack context."
      echo "Dispatch crystallization subagent or write .claude/last-session.md before stopping."
    fi
  fi

  exit 0
fi
```

- [ ] **Step 3: Test the stop hook — summary exists and is recent**

```bash
cd /Users/mst36/tools/mycelium
mkdir -p .claude

# Create session-start timestamp (simulates SessionStart hook)
date +%s > .claude/session-start-ts.tmp
sleep 1

# Create a recent session summary (simulates crystallization)
cat > .claude/last-session.md << 'EOF'
SESSION RESUME — test
## What was worked on
- Test
EOF

# Create reminder file (simulates post-action hook firing)
date +%s > .claude/mycelium-reminded.tmp

# Ensure .living/ was "updated" after reminder
sleep 1
mkdir -p .living
touch .living/learnings.md

# Run stop hook
echo '{}' | bash skills/core/hooks/mycelium-stop-check.sh
# Expected: no output (clean exit, summary exists and is recent)
echo "Exit code: $?"
# Expected: 0
```

- [ ] **Step 4: Test the stop hook — summary missing**

```bash
cd /Users/mst36/tools/mycelium
mkdir -p .claude

# Create session-start timestamp
date +%s > .claude/session-start-ts.tmp
sleep 1

# NO session summary file — simulate forgotten crystallization
rm -f .claude/last-session.md

# Create reminder file
date +%s > .claude/mycelium-reminded.tmp

# Ensure .living/ was "updated" after reminder
sleep 1
touch .living/learnings.md

# Run stop hook
echo '{}' | bash skills/core/hooks/mycelium-stop-check.sh
# Expected output:
#   Session summary not written. Next session will lack context.
#   Dispatch crystallization subagent or write .claude/last-session.md before stopping.
echo "Exit code: $?"
# Expected: 0 (warning, not block)
```

- [ ] **Step 5: Test the stop hook — stale summary (from previous session)**

```bash
cd /Users/mst36/tools/mycelium
mkdir -p .claude

# Create an OLD session summary (simulates leftover from previous session)
cat > .claude/last-session.md << 'EOF'
SESSION RESUME — stale from previous session
EOF

sleep 1

# Create session-start timestamp AFTER the summary (current session started after it was written)
date +%s > .claude/session-start-ts.tmp
sleep 1

# Create reminder file
date +%s > .claude/mycelium-reminded.tmp

# Ensure .living/ was "updated" after reminder
sleep 1
touch .living/learnings.md

# Run stop hook
echo '{}' | bash skills/core/hooks/mycelium-stop-check.sh
# Expected output:
#   Session summary not written. Next session will lack context.
#   Dispatch crystallization subagent or write .claude/last-session.md before stopping.
echo "Exit code: $?"
# Expected: 0 (warning, not block)
```

- [ ] **Step 6: Clean up test artifacts and commit**

```bash
cd /Users/mst36/tools/mycelium
rm -f .claude/session-start-ts.tmp .claude/mycelium-reminded.tmp .claude/last-session.md

git add skills/core/hooks/mycelium-stop-check.sh
git commit -m "feat: warn at Stop if session summary not written after significant work"
```

---

### Task 3: Update `commands/core.md` Skill Definition

**Files:**
- Modify: `commands/core.md:189-192` (add session summary step to post-action protocol)
- Modify: `commands/core.md:200` (update health hook description in table)
- Modify: `commands/core.md:202` (update stop hook description in table)
- Modify: `commands/core.md:206` (update stop hook logic description)
- Modify: `commands/core.md:257-260` (update crystallization subagent mandate)

- [ ] **Step 1: Add session summary step to post-action protocol**

After step 7 (line 192, "Convention feedback"), add step 8:

```markdown
8. **Write session summary**: Write or update `.claude/last-session.md` with a 5-section summary covering ALL work since session start. Use the session summary template:

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

   **Full-session coverage**: Run `git log --since=<session-start-timestamp>` and `git diff --stat` to capture all work since session start. If crystallization fires multiple times in a session, each write rebuilds the summary covering the entire session (cumulative, not incremental). The summary should get more expansive as the session progresses.
```

- [ ] **Step 2: Update hook table descriptions**

Update the hook table (lines 198-202) to reflect the new capabilities:

Change the `mycelium-health.sh` row from:
```
| `mycelium-health.sh` | SessionStart | Warns if `.living/` is missing or incomplete; records session timestamp; triggers weekly knowledge audit if `~/.claude/knowledge/.last-audit` is >7 days old |
```

To:
```
| `mycelium-health.sh` | SessionStart | Loads `.claude/last-session.md` for session resume (agent + user); warns if `.living/` is missing or incomplete; records session timestamp; triggers weekly knowledge audit if `~/.claude/knowledge/.last-audit` is >7 days old |
```

Change the `mycelium-stop-check.sh` row from:
```
| `mycelium-stop-check.sh` | Stop | Blocks session end if significant work was done without updating `.living/` |
```

To:
```
| `mycelium-stop-check.sh` | Stop | Checks `.living/` was updated after significant work; warns if session summary (`.claude/last-session.md`) was not written |
```

- [ ] **Step 3: Update stop hook logic description**

Update line 206. Change:

```markdown
**Stop hook logic**: The stop hook only blocks if `mycelium-post-action.sh` fired during the session (indicated by the presence of `.claude/mycelium-reminded.tmp`) AND `.living/` was not updated afterward. Read-only sessions, config-only sessions, and sessions without code execution are never blocked. When `.living/` is updated after the post-action hook fires, the reminder file is cleaned up automatically at session end.
```

To:

```markdown
**Stop hook logic**: The stop hook checks if `mycelium-post-action.sh` fired during the session (indicated by the presence of `.claude/mycelium-reminded.tmp`). If `.living/` was not updated afterward, it warns. If `.living/` was updated but `.claude/last-session.md` was not written (or is older than the session start), it emits a non-blocking warning reminding you to write the session summary. Read-only sessions, config-only sessions, and sessions without code execution are never checked.
```

- [ ] **Step 4: Update crystallization subagent mandate**

Update lines 257-260 in the Subagent-Driven Sessions section. Change:

```markdown
2. **After all subagent batches complete**, the main context dispatches a crystallization subagent (lightweight model) that:
   - Reviews the summary of what was accomplished
   - Appends entries to `.living/learnings.md` and `.living/decisions.md`
   - Checks cross-project relevance (if applicable)
```

To:

```markdown
2. **After all subagent batches complete**, the main context dispatches a crystallization subagent that:
   - Reviews the summary of what was accomplished
   - Appends entries to `.living/learnings.md` and `.living/decisions.md`
   - Writes `.claude/last-session.md` using the 5-section session summary template (covering ALL work since session start, not just the latest batch — run `git log --since=<session-start-ts>` and `git diff --stat` to ground the summary in facts)
   - Checks cross-project relevance (if applicable)
```

- [ ] **Step 5: Commit**

```bash
cd /Users/mst36/tools/mycelium
git add commands/core.md
git commit -m "feat: add session summary template and cross-session continuity to skill definition"
```

---

### Task 4: Manual Integration Test and Final Commit

- [ ] **Step 1: Verify `.claude/` is in `.gitignore`**

```bash
cd /Users/mst36/tools/mycelium
grep -q "\.claude/" .gitignore 2>/dev/null && echo "OK: .claude/ is gitignored" || echo "MISSING: add .claude/ to .gitignore"
# If missing, add it:
# echo ".claude/" >> .gitignore
# git add .gitignore
# git commit -m "chore: gitignore .claude/ session files"
```

- [ ] **Step 2: Verify all changes are in place**

```bash
cd /Users/mst36/tools/mycelium
echo "=== Health hook: session resume loading ==="
grep -n "Session resume" skills/core/hooks/mycelium-health.sh
grep -n "last-session.md" skills/core/hooks/mycelium-health.sh

echo "=== Stop hook: session summary warning ==="
grep -n "Session summary" skills/core/hooks/mycelium-stop-check.sh
grep -n "last-session.md" skills/core/hooks/mycelium-stop-check.sh

echo "=== Skill definition: session summary template ==="
grep -n "session summary" commands/core.md
grep -n "last-session.md" commands/core.md
```

Expected: Each section shows matches confirming the new code/text is present.

- [ ] **Step 3: End-to-end test — full session lifecycle**

```bash
cd /Users/mst36/tools/mycelium
mkdir -p .claude .living
touch .living/learnings.md .living/decisions.md .living/conventions.md

# 1. Simulate SessionStart with no prior session
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh
# Expected: JSON output with .living/ health only (no session resume)

# 2. Simulate a session that did work — create session summary
sleep 1
cat > .claude/last-session.md << 'EOF'
SESSION RESUME — Last session (2026-03-17 23:00):

## What was worked on
- Implemented cross-session continuity hooks

## Key decisions made
- Used stderr for user display to avoid corrupting JSON output

## Blockers & surprises
- None this session

## Current state
- Branch: feat/cross-session-continuity | All hooks updated

## Next steps
- Push branch and create PR
EOF

# 3. Simulate NEXT session's SessionStart — should load the resume
echo '{"cwd":"/Users/mst36/tools/mycelium","source":"startup"}' | bash skills/core/hooks/mycelium-health.sh 2>/dev/null
# Expected: JSON with additionalContext containing the session resume

# 4. Simulate Stop with summary written (should be clean)
date +%s > .claude/session-start-ts.tmp
sleep 1
date +%s > .claude/mycelium-reminded.tmp
sleep 1
touch .living/learnings.md  # "update" .living/
echo '{}' | bash skills/core/hooks/mycelium-stop-check.sh
# Expected: no output (clean exit — .living/ updated AND summary is recent)
echo "Exit: $?"

# Clean up
rm -f .claude/last-session.md .claude/session-start-ts.tmp .claude/mycelium-reminded.tmp
```

- [ ] **Step 4: Verify git log**

```bash
cd /Users/mst36/tools/mycelium
git log --oneline -5
# Expected: 3 new commits (health hook, stop hook, skill definition)
```

- [ ] **Step 5: Push branch**

```bash
cd /Users/mst36/tools/mycelium
git push origin feat/cross-session-continuity
```
