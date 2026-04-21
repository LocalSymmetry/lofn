# SUPERVISED PIPELINE — Architecture
*Added: 2026-04-21*

## The Problem
Subagents (orchestrator, audio, vision) burn context on thinking/debate and fail to save output before timing out. Without active supervision, the main session discovers failures only after the run completes.

## The Solution: Active Controller Pattern

The main session (or cron session) is the **controller**, not just the launcher. It checks every 5 minutes, reads disk artifacts to verify progress, and intervenes when an agent is stalled.

---

## Architecture

```
CONTROLLER (main session / cron)
  │
  ├── Phase 1: Fetch facts → save to disk (no subagents)
  ├── Phase 2: Write golden seed → save to disk (no subagents)
  ├── Phase 3: Spawn orchestrator (10-min timeout)
  │     ↓
  │   CONTROLLER checks at 5 min:
  │     - Is orchestrator_metaprompt.md on disk?
  │     - If yes → orchestrator on track
  │     - If no → read orchestrator's debate output from announce,
  │               extract key insights, write metaprompt MANUALLY,
  │               kill orchestrator if still running, spawn audio directly
  │
  ├── Phase 4: Spawn lofn-audio (60-min timeout)
  │     ↓
  │   CONTROLLER checks every 5 min:
  │     - Are step files appearing on disk? (00, 01, 02... incrementing)
  │     - If yes → agent is making progress, let it run
  │     - If no progress for 10 min → steer with specific instruction
  │     - If agent announces completion → verify file count + line counts
  │
  ├── Phase 5: Spawn lofn-qa
  │     ↓
  │   CONTROLLER checks at 10 min:
  │     - Is QA_REPORT.md on disk?
  │     - Read QA verdict
  │     - If FAIL → spawn targeted rerun (step 08 only, or specific song)
  │     - If PASS → deliver to Telegram
  │
  └── Phase 6: Deliver confirmed set to Telegram with QA cert
```

---

## Controller Check Protocol (every 5 minutes)

For each active subagent, run this check:

```bash
# Check what's on disk
ls -la OUTPUT_DIR/ | wc -l  # file count
wc -l OUTPUT_DIR/song_*.md  # line counts
ls -la OUTPUT_DIR/*.md | awk '{print $5, $9}'  # sizes
```

**Intervention triggers:**
- No new files after 10 min → steer agent
- Files present but line counts low (< 60 lines/song) → note for QA
- Agent announces "completing" but files not on disk → extract from announce, write manually
- Agent timed out → check what was saved, continue from last saved step

---

## Orchestrator Intervention Protocol

The orchestrator consistently produces excellent panel insights but fails to save the metaprompt before timing out. The fix:

1. Let orchestrator run its 10-min debate
2. When it announces/times out, read the announce text
3. Extract key insights: aha moments, panel decisions, creative direction
4. Write the metaprompt manually from those insights (main session does this)
5. Save to orchestrator_metaprompt.md
6. Spawn lofn-audio

The panel debate is real thinking — it produces better insights than the main session alone. We just need to catch the output.

---

## Cron Implementation

The cron prompt includes explicit 5-minute check instructions:

```
After spawning each subagent, check disk progress every 5 minutes:
- Run: exec ls -la OUTPUT_DIR/ && wc -l OUTPUT_DIR/song_*.md 2>/dev/null
- If no progress after 10 min: steer the agent with specific instruction
- If orchestrator times out without saving metaprompt: extract insights from
  its announce message and write the metaprompt manually, then spawn audio
- If audio produces songs < 60 lines: note for QA, don't wait for QA to catch it
```

---

## Why This Works

1. **The orchestrator's panel debate is genuinely valuable** — it produces insights the main session wouldn't generate alone (e.g., "the 6-song suite IS the boxing match", "90 seconds answers 300 years")
2. **The main session can read the announce text** and extract those insights even when the orchestrator fails to save them
3. **Disk artifacts are ground truth** — checking files every 5 min catches failures before they compound
4. **Targeted reruns** — QA + controller means we rerun only what failed (step 08 specifically, or specific songs), not the whole pipeline

---

## Controller Checklist Per Phase

### After spawning orchestrator (check at 5 min):
- [ ] Is `orchestrator_metaprompt.md` saved?
- [ ] If not: is agent still running? Steer it: "Write the metaprompt NOW to [path]"
- [ ] If timed out: extract insights from announce, write metaprompt manually

### After spawning lofn-audio (check at 5, 10, 20, 30 min):
- [ ] Are step files appearing incrementally? (00→01→02...)
- [ ] At 20 min: are song files appearing? (song_01, song_02...)
- [ ] At 30 min: do song files have ≥ 60 lines?
- [ ] If stalled: steer with "Stop current step, save what you have, continue from step [N]"

### After lofn-audio completes:
- [ ] Count song files (expect 6)
- [ ] Check line counts (min 60 each)
- [ ] Check for [SONG FORM:] declarations
- [ ] Spawn QA if all looks viable

### After QA:
- [ ] Read QA_REPORT.md verdict
- [ ] If FAIL: identify which songs need rerun, spawn targeted step-08 rerun
- [ ] If PASS: deliver to Telegram with QA certification
