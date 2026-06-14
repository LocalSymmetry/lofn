# Lofn Music - Subagent Architecture Pattern

## ⛔ OUTPUT TYPE: SUNO PROMPTS ONLY

**You write prompts and lyrics for Suno. You do NOT call any music generation tool or API.**
Do NOT use `music_generate` or any audio tool. Your artifacts are `.md` files.
The Scientist or a separate submission step handles actual Suno API calls.

---

## THE CORRECT SPLIT (from original Lofn ui.py)

The original Lofn app ran this exact pattern:
1. `generate_concept_mediums()` - coordinator phase, internally calls each steps 00-05 chain in sequence and returns all concept-medium pairs
2. `select_best_pairs()` - panel votes on top N pairs
3. `generate_prompts_for_pair(pair)` - pair phase, called ONCE PER PAIR in parallel, internally calls each steps 06-10 chain in sequence

Important correction: “one function call” in the Streamlit UI did **not** mean one creative prompt. It meant a wrapper function that performed multiple LLM chain calls. OpenClaw agents must preserve that internal chain depth.

**This is the mandatory architecture for all future runs.**

---

## PRE-AUDIO ORCHESTRATOR GATE — REQUIRED

Before coordinator Step 00, validate that the run has a real Lofn-Core + orchestrator packet:

```bash
python3 /data/.openclaw/workspace/scripts/validate_orchestrator_packet.py <run_dir>
```

The packet must include a substantial seed lineage, Golden Seed/core seed, original Lofn panel object (`Special Flairs`, `Concept Panel`, `Medium Panel`, `Context & Marketing Panel`) with Devil’s Advocate / Hyper-Skeptic roles, metaprompt, pair assignments with rationale, and audio handoff. If this fails, do not proceed to audio. Launch or request `lofn-orchestrator` work instead.

## DAILY CONTROLLER GUARD — REQUIRED

For daily runs, do not launch a new vN lane into the same output directory while an older lane is active. The parent/controller must inspect active subagents, stop stale lanes explicitly, and write a status note before resuming. Completion messages are not authority; disk artifacts plus validators are authority.

The controller is the only phase router. Coordinator and pair agents write their assigned artifacts and stop. They do not spawn the next phase, continue from a sibling's completion event, or locally emulate downstream agents.

---

## CANONICAL STEP ARTIFACT PROVENANCE — REQUIRED

Every canonical step file must follow `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md` and include:

1. `## 0. Step Provenance` — exact step file loaded, input artifacts used, model call mode, validation command.
2. `## 1. Input Context Digest` — concrete digest of prior artifacts; must name actual concepts/facets/media, not generic task text.
3. `## 2. Step Template Requirements Applied` — specific requirements from the loaded step prompt.
4. `## 3. Panel / Critic Deliberation Log` — actual debate/friction: concept voice, medium voice, context voice, Devil’s Advocate / Hyper-Skeptic objection, and resolution.
5. `## 4. Complete Step Output` — full step response, not a summary or plan.
6. `## 5. Execution Log` — files read, model response written, validation attempts, repairs.
7. `## 6. Self-Critique Against Step Requirements` — adversarial check.
8. `## 7. Validation Result` — paste validator output after pass.

Every step must preserve complete outputs. A file that says what it would do, summarizes a response, or omits the actual lists/concepts/guides/prompts/lyrics is incomplete. A file with the right filename but without these sections is a backfilled artifact and fails. Do not write “line 1 / line 2” placeholders, repeated paragraphs, or self-check claims contradicted by the artifact body.

## CONTINUITY PAYLOAD — REQUIRED FOR EVERY STEP

Every coordinator and pair-agent LLM call must include a compact continuity payload. It must be visible in the step artifact under `## Continuity Payload Used` or equivalent JSON fields, not merely assumed from prior files.

Required fields:
1. `special_flairs` — orchestrator Special Flairs that must color the artifact.
2. `panel_pressures` — Concept Panel, Medium Panel, Context & Marketing Panel, and Devil's Advocate / Hyper-Skeptic demands relevant to this step.
3. `seed_excerpt` — full Golden Seed for Steps 00-05; 100-250 word pair-specific Golden Seed operating excerpt for Steps 06-10.
4. `active_personality` — named personality/persona, sonic world sentence, signature device, and what would count as personality loss.
5. `source_thread` — for NEWS pairs, the exact current-event/source pressure that must survive as image, lyric-world, fact, or production dramaturgy.

The immediately previous step output is never enough by itself. If a step lacks panels/flairs, seed excerpt, or personality, stop and repair/rerun before advancing. Missing continuity payload = `REPAIR — THREAD LOSS`.

---

## SUBAGENT 1: Steps 00-05 (Concept-Medium Generation)

Receives, in this order:
1. Golden Seed lineage + full Golden Seed
2. active Lofn personality / mode
3. orchestrator output (metaprompt, panel, assignments)
4. constraints and QA/output contract

The coordinator must generate from the seed first. Constraint axes are vocabulary, not the form of the song.

Executes as **six separate LLM turns**, matching original `generate_concept_mediums()` in `lofn/llm_integration.py`:
- Step 00: read `steps/00_Generate_Music_Aesthetics_And_Genres.md`, call the model, write `step00_aesthetics_and_genres.md` using the canonical provenance template. Provenance must cite the validated orchestrator packet.
- Step 01: read `steps/01_Generate_Music_Essence_And_Facets.md`, call the model using Step 00 output, write `step01_essence_and_facets.md` using the canonical provenance template
- Step 02: read `steps/02_Generate_Music_Concepts.md`, call the model using Step 01 output, write `step02_concepts.md` using the canonical provenance template
- Step 03: read `steps/03_Generate_Music_Artist_And_Critique.md`, call the model using Step 02 output, write `step03_artist_and_critique.md` using the canonical provenance template
- Step 04: read `steps/04_Generate_Music_Medium.md`, call the model using Step 03 output, write `step04_medium.md` using the canonical provenance template
- Step 05: read `steps/05_Generate_Music_Refine_Medium.md`, call the model using Step 04 output, write `step05_refine_medium.md` using the canonical provenance template and `concept_medium_pairs.json` (6 pairs)

**Do not combine Steps 00–05 into one prompt, one response, or renamed summary files.** Files like `step00_coordinator_overview.md` or `step03_pair_concept_matrix.md` are summaries, not canonical original-Lofn step outputs.

### Mandatory validation + retry loop for Steps 00–05
After writing each step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair the artifact in place and rerun with `--attempt 2`, then `--attempt 3` if needed. After 3 failed attempts, stop the run, write `TIMEOUT_STATUS.md` or `VALIDATION_BLOCKED.md`, and escalate. Do **not** continue to the next step with a failed artifact.

**STOP HERE. Do not proceed to step 06.**

Return a structured handoff to the parent/controller that names `step05_refine_medium.md`, `concept_medium_pairs.json`, and the validation result. Do not spawn pair agents from inside the coordinator.

---

## SUBAGENTS 2-7: Steps 06-10 (One Per Pair)

### Mandatory parent-mediated pair-agent rule (added after 2026-05-21 collapse)

The coordinator MUST stop after Step 05. It must not write, emulate, summarize, or locally synthesize Steps 06-10 for any pair. The parent/controller must spawn one independent `lofn-audio` child session per pair. Phrases like “spawn or run” are forbidden; the correct instruction is **spawn by parent only**.

Each pair Step 10 artifact must include execution provenance that proves independent execution, not only filename shape:

```markdown
## Execution Provenance
agent_label: lofn-audio-pair-NN
session_key: agent:lofn-audio:subagent:<actual child session key>
parent_session: <controller/main session key when available>
model: <model id>
spawned_by_parent: true
step_call_mode: separate_child_session
collapse_guard: no compact direct synthesis; no local emulation of other pairs
```

A pair package without a real child `session_key` is not pipeline-clean. It may be useful draft material, but QA must mark **Pipeline Integrity: REPAIR REQUIRED** until provenance is repaired or the pair is rerun.

Each receives the full-context pair-agent injection standard from `skills/music/SKILL.md` Workflow item 9. This supersedes the former lean input standard in `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md`.

Normal pair-agent prompts must inject the complete upstream context needed for the current pair and step:
- Full personality DNA block for the assigned personality: core identity, sonic world, signature device, catchphrases, G.L.O.W. Protocol where applicable, and vocal architecture
- Pair-specific Golden Seed operating excerpt plus the full invariant hook / dangerous requirement that must not be domesticated
- Complete continuity payload: `special_flairs`, Concept/Medium/Context panel pressures, Devil's Advocate / Hyper-Skeptic objections, active personality/persona sonic world sentence, signature device, and source/news thread when applicable
- Production Mandates and Forbidden Substitutions table
- Current Step 05 artifact (`step05_refine_medium.md`)
- Structured pair list (`concept_medium_pairs.json` or equivalent)
- ONE specific concept-medium pair / pair assignment excerpt
- The relevant Step 06–10 contract
- Provenance block (`spawned_by_parent`, `step_call_mode`, `source_golden_seed`, `golden_seed_excerpt_included: true`, `source_step05`, `source_pair_list`, `pair_id`, `model`)
- Modality-specific QA blockers

The parent/controller validates and retains the complete Golden Seed + orchestrator packet, then injects the relevant full payload into each pair-agent task. Do not make the task shallow to save tokens; preserve depth while preventing agents from pulling broad, unbounded context themselves.

Pair-agent task prompts MUST NOT begin with line counts, EMO tags, or prompt-shape requirements. Begin with the compact pair seed/anchor, then the pair's dangerous requirement / Lofn-specific wrongness, then creative permission, then the required Suno structure. The QA contract remains blocking, but it is not the muse.

Each executes (for its ONE pair only) as **five separate LLM turns or five externally orchestrated sub-steps**, matching original `generate_music_prompts()` in `lofn/llm_integration.py`. If a single pair-agent cannot prove the five turns happened beyond filenames, the controller should spawn one subtask per step or QA must treat the result as suspect:
- Step 06: read `steps/06_Generate_Music_Facets.md`, call the model, write `pair_{NN}_step06_facets.md` using the canonical provenance template. Provenance must cite the Golden Seed, orchestrator metaprompt, pair assignment, Step 05 concept-medium pair, and prior coordinator outputs. Step 06 must contain 8–12 pair-specific facets; every facet must include why it matters and a failure mode. Generic rubrics like “preserve concept / preserve medium / maintain hook” are not sufficient.
- Step 07: read `steps/07_Generate_Music_Song_Guides.md`, call the model using Step 06 output, write `pair_{NN}_step07_song_guides.md` using the canonical provenance template
- Step 08: read `steps/08_Generate_Music_Generation.md`, call the model using Step 07 output, write `pair_{NN}_step08_generation.md` using the canonical provenance template
- Step 09: read `steps/09_Generate_Music_Artist_Refined.md`, call the model using Step 08 output, write `pair_{NN}_step09_artist_refined.md` using the canonical provenance template
- Step 10: read `steps/10_Generate_Music_Revision_Synthesis.md`, call the model using Step 09 output, write `pair_{NN}_step10_revision_synthesis.md` using the canonical provenance template

**Do not combine Steps 06–10 into one prompt, one response, or one omnibus file.** A convenience rollup may be created afterward as `pair_{NN}_steps_06_10_rollup.md`, but it is not a substitute for the five canonical step files. A file with the right name but shallow content, missing required headers, or self-check claims contradicted by grep is a failure, not a partial pass.

### Mandatory validation + retry loop for Steps 06–10
After writing each pair step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair the artifact in place and rerun with `--attempt 2`, then `--attempt 3` if needed. After 3 failed attempts, stop that pair, save `pair_{NN}_VALIDATION_BLOCKED.md`, and return the exact validator failure. Do **not** write Step 10 or claim the pair is complete if Step 06–09 failed.

Before writing `pair_{NN}_COMPLETE.md` or returning completion to the parent, also run the pair-level gate:

```bash
python3 /data/.openclaw/workspace/scripts/validate_pair_artifacts.py <audio_dir> <NN> --attempt 1
```

If it fails, repair the failing artifact(s) in place and rerun with `--attempt 2`, then `--attempt 3` if needed. `pair_{NN}_COMPLETE.md` must include the final `PAIR VALIDATION PASS` output. A pair completion message without this pass is not accepted as complete.

Cross-pair gates are mandatory:
- After all pair Step 06 files exist, run `python3 /data/.openclaw/workspace/scripts/validate_step06_distinctiveness.py <audio_dir>`; if it fails, repair Step 06 before Step 07.
- After all pair Step 09 files exist, run `python3 /data/.openclaw/workspace/scripts/validate_step09_distinctiveness.py <audio_dir>`; if it fails, repair Step 09 before Step 10.
- After all pair Step 10 files exist, run `python3 /data/.openclaw/workspace/scripts/validate_portfolio_distinctiveness.py <audio_dir>`; if it fails, repair Step 10 before completion.

Outputs to disk: five separate step files for its pair number
Returns: Step 10 final song prompts + lyrics as output text

---

## ORCHESTRATION FLOW

```
Main session
  └── spawns Subagent 1 (steps 00-05)
         └── writes concept_medium_pairs.json
  └── reads concept_medium_pairs.json
  └── spawns Subagents 2-7 in parallel (one per pair)
         └── each writes full song (prompt + lyrics)
  └── collects all 6 songs
  └── QA gate
  └── Deliver to Telegram
```

---

## concept_medium_pairs.json format
```json
[
  {
    "pair_num": 1,
    "concept": "Full refined concept text",
    "medium": "Full production style text",
    "artist_influence": "Named artist"
  }
]
```

## OUTPUT FORMAT FOR PAIR SUBAGENTS

Each pair subagent must return in Step 10 only after Step 06, Step 07, Step 08, and Step 09 have already been written as separate files. Step 10 must include:
- Suno/Udio Core Music Prompt (**target 850-1000 characters, hard max 1000 unless destination explicitly permits more, no artist names**). It must be dense, producer-grade, copy-paste-ready, and single paragraph. Mandatory order: selected genre/style label(s) + tempo/energy → vocalist spec → instrumentation/sound palette/mix → musical arrangement arc → bold sonic device → avoidances. Do not lead with story, theme, or procedural phrases like “Begin in/by/with,” “Use,” “Build the track from,” or “Chronology.” Do not pad with tag soup; reach the length through useful production chronology, vocal treatment, mix/arrangement intelligence, and negative prompt logic.
- Full lyrics using **clean EMO section headers** and legacy runtime length: **70-120 sung lines target; <60 sung lines is a repair trigger.** Song-force improvements change HOW length is earned, not whether the package needs enough sung material for a 3:00-4:00 Suno result. Use hook recurrence, chorus mutation, bridge pressure, call-response, ghost/echo reprises, and embodied image development — never filler or procedural exposition.
  - Required `[Theme: <specific scene-pressure / emotional operating system>]` as the first line of the Suno lyrics block for every final song. This is a focusing spell for Suno and for the agent: it must be specific, embodied, and musically useful, not a generic topic label.
  - Required `[SONG FORM: <named form>]` declaration immediately after Theme for every final song/lyric set. The form must be descriptive enough to guide structure (e.g., “Bathroom piano house — breath intro / verse / pre / chorus / dry bridge / call-response final chorus / afterglow”), not just “pop song.”
  - Clean section headers with full performance-script syntax: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`.
  - At least one standalone SFX cue line is required when validating Suno package readiness.
  - At least one non-lexical vocal hook where musically appropriate (`mm`, `ooh`, `ah`, whispered echo, etc.)
  - Performance/mix cues where structurally important (`No beats`, `Half-time`, `Double-time`, `whispered`, `filter sweep`, `choir flinch`, etc.)
  - No editor commentary, TODOs, rhyme letters, or syllable bars in final lyrics

Each pair subagent must also identify at least one **Lofn-specific move** that survives in the music prompt and lyrics: scientific specificity as feeling, AWE/INDIGNATION state-change, wrongness-as-beauty, hidden structural logic, literary/prayer/witness mode, or Open Laboratory continuity pressure. If none exists, revise before the pre-completion gate.

**Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags alone are NOT acceptable for final delivery.** They may appear in drafts, but Step 10 final files must be performance-ready for Suno.

### ⛔ PRE-COMPLETION GATE — ALL 5 MUST PASS BEFORE WRITING FINAL OUTPUT

Before writing your final step10 output, run this check. If any box is unchecked, revise and re-check.

**In the final output for each song:**
- [ ] Standalone `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` section exists: copy-paste-ready Core Music Prompt, single paragraph, **850-1000 characters**, no artist names, and includes emotion → selected style label(s) from the run → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → avoidances. Scattered `[STYLE/TEMPO/KEY]`, `[SONIC WORLD]`, and `[PRODUCTION NOTES]` do NOT satisfy this gate; extra detail belongs in sidecars.
- [ ] Lyrics begin with `[Theme: <specific scene-pressure / emotional operating system>]`, immediately followed by `[SONG FORM: <named form>]`; lyrics have 70-120 sung lines target with <60 repair, and clean full section headers: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`.
- [ ] Sung lines contain no prompt/procedure/QA/production-manual debris.
- [ ] SFX cues / non-lexical hooks are included only if they serve the hook or controlled fracture.
- [ ] At least one non-lexical vocal hook (`ooh`, `mm`, `ah`, whispered echo, call-response fragment)

**Document your check** (save as part of your step10 file or companion `step10_qa_pair{N}.md`):
```
GATE CHECK — Pair {N}, Variation {X}:
Clean EMO headers: ✓/✗ → [count] sections tagged
No lyric debris: ✓/✗ → [evidence]
Hook recurrence / singback: ✓/✗ → [evidence]
Optional SFX/non-lexical device serves song: ✓/✗/N/A → [line or rationale]
```
If any required check is ✗, revise the lyrics before completing. Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags are not sufficient in final output because the emotional arc disappears.

Written to: `step10_final_pair{N}.md`
