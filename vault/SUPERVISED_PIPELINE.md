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

### Before spawning modality coordinators — DAILY RUN AUDIO GATE:
- [ ] Does the run have `04_handoff_to_music_brief.md` or `04_handoff_to_audio_brief.md`?
- [ ] If NO, but vision handoff exists: write/repair the audio handoff from the golden seed + orchestrator pair assignments. Do not accept vision-only as complete unless The Scientist explicitly requested vision-only.
- [ ] Spawn `lofn-audio` first or in parallel with vision. Daily music is mandatory; vision is not allowed to crowd it out.
- [ ] Use `lofn-audio` exactly. `lofn-music` is not a configured agent.

### After spawning lofn-audio (check at 5, 10, 20, 30 min):
- [ ] Are coordinator files appearing incrementally? (`step00_coordinator_overview.md`, `pair_01_concept.md`…`pair_06_concept.md`, `step05_pair_agent_handoff.md`)
- [ ] Are pair files appearing? (`pair_01_steps_06_10.md`…`pair_06_steps_06_10.md`)
- [ ] At 30 min: do pair files include complete lyrics + Suno-ready prompts, not stubs?
- [ ] If stalled: steer with "Stop current step, save what you have, continue from step [N]"

### After lofn-audio completes:
- [ ] Count pair files (expect 6: `pair_01_steps_06_10.md` through `pair_06_steps_06_10.md`)
- [ ] Check each pair has 4 variants, complete lyrics, Suno-ready prompts, and EMO: section headers
- [ ] Check for document/form structure declarations when the seed requires form constraints
- [ ] Spawn QA if all looks viable; otherwise repair audio before declaring the daily run complete

### After lofn-vision completes:
- [ ] Confirm combined prompt set exists if expected
- [ ] Confirm per-pair files exist: `pair_01_steps_06_10.md` through `pair_06_steps_06_10.md`
- [ ] Confirm each pair file has 4 variants / prompts
- [ ] If pair files are missing: DO NOT RENDER; spawn/steer targeted pair completion before QA

### After QA:
- [ ] Read QA_REPORT.md verdict
- [ ] Confirm QA explicitly inspected per-pair files for image runs, or per-song/pair files for music runs
- [ ] Confirm QA says `READY TO RENDER` before any image/video generation spend
- [ ] If FAIL: identify which songs/prompts/pairs need rerun, spawn targeted step-08/step-10 rerun
- [ ] If PASS: deliver to Telegram with QA certification
