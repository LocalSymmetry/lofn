# Lofn Model Assignments

Updated: 2026-06-01 14:30 EDT
Status: Active — Split-step music chain battle-tested; failed models removed; validator rules documented

**See also:**
- `vault/VISION_MODEL_ASSIGNMENTS.md` — Image/art pipeline assignments
- `vault/DIRECTOR_MODEL_ASSIGNMENTS.md` — Video/animation pipeline assignments

---

## Core Principle

**Lofn music is an LLM chain, not one giant worker.**

The original Lofn architecture behaves like a LangChain-style sequence of specialized calls. OpenClaw music runs must preserve that shape:

Research → Core Seed → Orchestrator → Coordinator Steps 00-05 → Pair Step 06 → Pair Step 07 → Pair Step 08 → Pair Step 09 → Pair Step 10 → Step 11 Enhancement → Step 12 Audit → QA

Do **not** assign one `lofn-audio` subagent to do Steps 06-10 in a single run unless The Scientist explicitly asks for a shortcut. If necessary, spawn one subagent per pair per step. This is preferred over context collapse and timeout churn.

**Do not trust per-spawn `model` overrides.** Use dedicated configured step agents (`lofn-audio-step06` through `lofn-audio-step12`) instead of `lofn-audio` with an override. Overrides were unreliable during the 2026-05-29 "Recapturing the Shattered Flame" run — agents silently fell back to their configured default model.

---

## Audio Execution — Split-Step Chain

| Stage | Agent ID | Model | Role |
|---|---|---|---|
| Steps 00-05 | `lofn-audio-coordinator` | `deepseek/deepseek-v4-pro` | Coordinator: aesthetics, essence, concepts, mediums, pair-task packaging; preserves ICB |
| Step 06 | `lofn-audio-step06` | `openrouter/qwen/qwen3.7-max` | Facets: reliable structured writing, variation separation |
| Step 07 | `lofn-audio-step07` | `openrouter/qwen/qwen3.7-max` | Song Guides: reliable artifact writing, emotional continuity, variation shaping |
| Step 08 | `lofn-audio-step08` | `openrouter/qwen/qwen3.7-max` | Music generation artifacts: reliable file writing, validator-shaped prompts and lyrics |
| Step 09 | `lofn-audio-step09` | `openrouter/qwen/qwen3.7-max` | Artist refinement: reliable artifact writing, structure/taste repair, producer polish |
| Step 10 | `lofn-audio-step10` | `deepseek/deepseek-v4-pro` | Final synthesis: full Step 10 package, Suno prompt, EMO headers, provenance, self-check |
| Step 11 | `lofn-audio-step11` | `openai/gpt-5.5` | Enhancement pass: strong model polish, 15-point gate verification, producer-grade prompt, EMO format compliance |
| Step 12 | `lofn-audio-step12` | `openrouter/qwen/qwen3.7-max` | Panel-of-panels prompt audit: cross-song consistency, benchmark comparison, structural QA |

### Legacy agent

The legacy `lofn-audio` agent remains available (`deepseek/deepseek-v4-pro`) but **do not use it for Steps 06-10**. Use the dedicated step agents above.

---

## Controller / Orchestration

- Main session / controller: `deepseek/deepseek-v4-pro`
- `lofn-orchestrator`: `google/gemini-3.5-flash` ✅ proven
- Evaluation / ranking: `deepseek/deepseek-v4-pro`
- QA: `google/gemini-3.5-flash` ✅ proven

## Core / Seed

- Research synthesis: `deepseek/deepseek-v4-pro`
- Seed lineage: `deepseek/deepseek-v4-pro`
- Golden Seed: `deepseek/deepseek-v4-pro`

---

## Why These Models — Battle-Tested Assignments (2026-05-29)

### DeepSeek V4 Pro — Coordinator / Seed / Research / Step 10
Best at large sustained synthesis and preserving the Immutable Continuity Block. Use where long context must be compressed into durable artifacts. For Step 10, produces rich full synthesis packages with lyrics, Suno prompts, EMO headers, and provenance.

### Qwen3.7 Max — Steps 06, 07, 08, 09, and 12
The workhorse of the split chain. Reliable artifact writer. Use for structural reasoning, variation separation, song-guide continuity, music generation artifacts, artist refinement, and panel/audit synthesis. **Three other models were tested and failed for these steps** (see Learning Log below).

### GPT-5.5 (direct OpenAI) — Step 11 ✅ PROVEN
Enhancement pass. Strong model polish, 15-point gate verification, producer-grade prompt refinement, EMO format compliance. Proven across 18 pairs (2026-05-30 dual Alexis run). Gemini 3.1 Pro Preview proved unreliable for this step — produced `[ENHANCED LYRICS BLOCK]` placeholder skeletons instead of real content.

### Gemini 3.5 Flash — Orchestration / QA
Fast and proven for orchestrator packet generation and QA. Avoid relying on it as a whole-pair creative writer.

---

## Learning Log — 2026-05-29 "Recapturing the Shattered Flame" Run

### Failed Models (removed from these steps)

| Model | Step Tested | Failure Mode |
|-------|-------------|--------------|
| `xiaomi/mimo-v2.5-pro` | Step 07 (Song Guides) | Completed without writing files — 5 consecutive attempts, all "done" with 0 files on disk. |
| `minimax/minimax-m2.7` | Step 08 (Music Gen) | Completed without writing files on fresh pairs; produced validator-shape failures on repairs. Moved to Qwen. |
| `anthropic/claude-opus-4.8` | Step 08/10 (attempted) | Too slow for creative generation in subagent context; timed out before writing. Costly. Reserved only for enhancement. |
| `anthropic/claude-opus-4.7` | Step 09 (attempted) | Same timeout issue; creative generation overruns 600s budget. Removed from chain. |

### Working Discoveries

1. **Personality DNA must be injected at EVERY pipeline stage.** Saying "voice = Alexis" in a task prompt is insufficient. Subagents default to Lofn's voice (their system-context personality). Every creative task — seed, orchestrator, audio coordinator, pair agents, step11 — must receive an explicit personality block: the full G.L.O.W. Protocol, sonic pillars, core beliefs, catchphrases, and vocal architecture of the target personality. Without this injection, output drifts into Lofn's default voice. Proven across 18-song dual Alexis redo (2026-05-30).

2. **Step11 MUST use the `lofn-audio-step11` agent — never a generic subagent.** Generic subagents lack the EMO format template ([emo=], [vox=], [prod=] tags) and produce bare structural tags or placeholder skeletons. The dedicated agent carries the format specification. Proven: generic subagents produced `[ENHANCED LYRICS BLOCK]` placeholders; `lofn-audio-step11` produced full EMO-formatted content at 17-24KB per pair.

3. **Step11 should run ONE PAIR PER AGENT, not all 6 in one session.** When a single step11 agent processes all 6 pairs, Variations 3-4, climax bridges, and outros collapse into identical templates across pairs. The agent starts unique for Pair 01 but progressively copies structural frames into later pairs. Fix: spawn 6 separate `lofn-audio-step11` agents, each reading only its own step10 file and writing only its own enhanced file. Proven 2026-05-30 on Run 2 v2 QA.

4. **Qwen3.7 Max is the reliable workhorse. It writes files on every attempt for Steps 06-09. Use it as the default for the split chain.

2. **Validator-shape failures are the #1 blocker, not creative quality.** The 2026-05-29 run spent ~80% of its repair time on: prompt length (must be 850-1000 chars), missing `## 1. MUSIC PROMPT` section, missing `devil` marker in panel deliberation log, placeholder/template language in artifacts, canonical filename mismatches (`_generation.md` vs `_music_prompts.md`), too few EMO headers, missing standalone SFX cue, lyrics <60 sung lines.

3. **Disk is the only authority.** Never trust subagent completion messages. Always verify files exist and validate before advancing.

4. **Per-spawn model overrides are unreliable.** Some agents ran DeepSeek despite `model=openrouter/anthropic/claude-opus-4.8`. Always use dedicated configured step agents.

5. **Surgical prompt repair is faster than agent repair.** When a step10 prompt is 1852 chars (too long) or 839 chars (too short), a 5-line Python edit is faster than spawning another agent.

6. **MiMo is a non-writer in this runtime.** Every attempt — narrow task, write-first prompt, no reference reads — returned "done" with zero files on disk. This model should not be used for artifact-writing steps.

---

## Validator-Aligned Artifact Rules

These rules exist because the validator is literal. Follow them exactly.

### Canonical filenames

- Step 08 canonical file is `pair_{NN}_step08_generation.md` (not `_music_prompts.md`; that name is a compatibility sidecar)
- Step 10 canonical file is `pair_{NN}_step10_revision_synthesis.md`
- Step 11 canonical file is `pair_{NN}_step10_final_package_enhanced.md`

### Actual step template paths
Do not invent alternate step filenames:
- Step 06: `steps/06_Generate_Music_Facets.md`
- Step 07: `steps/07_Generate_Music_Song_Guides.md`
- Step 08: `steps/08_Generate_Music_Generation.md`
- Step 09: `steps/09_Generate_Music_Artist_Refined.md`
- Step 10: `steps/10_Generate_Music_Revision_Synthesis.md`
- Step 11: `steps/11_Generate_Music_Enhancement.md`

There is no `08_Generate_Music_Generation_Artifact.md`.

### Required section shape for Step 08, Step 10, and Step 11

Every variation MUST follow this QA-compliant structure:
```
## VARIATION XA — TITLE
### SUNO STYLE PROMPT
[850-1000 char dense prose: genre + tempo + key + vocalist + instrumentation + arc + sonic device + emotional palette]

### SUNO LYRICS
[Theme: one-line emotional/conceptual summary]
[SONG FORM: Movement I → Fracture I → Refrain I → ...]
[Movement I – EMO:EmbodiedEmotion – Vocalist – Instrument cues]
[lyrics]
```

- `[Theme: ...]` and `[SONG FORM: ...]` are REQUIRED on the first two lines of every SUNO LYRICS block
- Section headers must be inline: `[Movement – EMO:... – Vocalist – Cues]`
- EMO tags MUST use specific embodied emotions, never generic labels
- Structure must use Movement/Fracture/Refrain, not generic Verse/Chorus/Bridge
- Silence cues: `[Silence — 2 bars]`
- SFX inline with asterisks
- At least 60 sung lines per variation
- At least one standalone SFX cue per variation

### Required section shape for Step 08 and Step 10 (legacy)

- `## 1. MUSIC PROMPT` — ONE 850-1000 character prompt body. Not four long variations. Count characters before saving.
- `## 2. LYRICS` — begins with `[Theme: ...]` then `[SONG FORM: ...]`. At least 60 sung lines, target 70-120.
- Full EMO headers: `[Section - EMO:<emotion> - <Role> - <cue>]`. Bare `[EMO:...]` fails.
- At least one standalone SFX cue: `*steam hiss*` line.
- `## 3. Panel / Critic Deliberation Log` — must contain `devil`, `hyper-skeptic`, and `resolution` markers.
- `## 1. Input Context Digest` — must include `step file loaded:`, `input artifacts used:`, `validation command:`.
- Continuity Payload Used — must include plural `Special Flairs` marker.
- Avoid literal words `placeholder` and `template` in final artifacts; use `stub text` or `scaffold text`.
- Prompt first line must open with genre/style + tempo/energy + vocalist/instrumentation. Banned openings: `Begin in/by/with...`, `Use...`, `Build the track from...`, `Chronology:`, `For an adult human singer...`.

---

## Spawn Pattern for Music Runs

After Step 05 completes:

1. Spawn up to 5 pair agents for **Step 06 only** using `lofn-audio-step06`.
2. Verify files on disk. Validate. Then spawn Step 07 agents using `lofn-audio-step07`.
3. Continue one step at a time through Step 10.
4. Step 11 enhancement uses `lofn-audio-step11`.
5. Step 12 uses `lofn-audio-step12` when triggered.

Each call has a narrow objective and is much less likely to timeout.

### File expectations per pair

- `pair_XX_step06_facets.md`
- `pair_XX_step07_song_guides.md`
- `pair_XX_step08_generation.md`
- `pair_XX_step09_artist_refined.md`
- `pair_XX_step10_revision_synthesis.md`
- `pair_XX_step10_final_package_enhanced.md` (Step 11 output)

---

## Output Validation

| Stage | Validation |
|---|---|
| After Coordinator Step 05 | `validate_portfolio_distinctiveness.py` on step05 |
| After Pair Step 06 | `validate_step06_distinctiveness.py <audio_dir>` |
| After Pair Step 09 | `validate_step09_distinctiveness.py <audio_dir>` |
| After Pair Step 10 | `validate_step.py 10 <file>` per pair |
| After Pair Step 11 | 15-point QA self-check in enhanced package |
| After Step 12 | `STEP12_MUSIC_PROMPT_AUDIT.md` |
| Before QA | `validate_pair_artifacts.py` for all pairs |
| QA Gate | Somatic Gate + full 15-point check |

---

## Standing Warnings

- `lofn-audio` default model is `deepseek/deepseek-v4-pro`; this is **not** the split-step pair model.
- Spawn model overrides are unreliable — prefer dedicated configured step agents.
- If any subagent times out, inspect disk artifacts first; resume from last completed step, not from scratch.
- Every subagent must write artifacts as it completes each major step.
- ICB must survive every handoff.
- Disk is the only authority. Never trust completion messages without file verification.
- MiMo V2.5 Pro and MiniMax M2.7 should not be used for artifact-writing steps without re-testing.
