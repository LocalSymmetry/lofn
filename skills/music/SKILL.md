---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio - Router

This router prevents context collapse. The full tuned music pipeline text is preserved byte-for-byte in `references/music_full_legacy.md`; newer Simple Surface / Complex Engine references and Step 05-10 files ADD song-force / anti-blandness requirements but do not waive the legacy Suno/package QA gate unless The Scientist explicitly approves an exception for a specific run.

## Workflow

1. Confirm this is a music/audio task: Suno prompt, lyrics, song guide, audio pipeline, or music production brief.
1a. Read `/data/.openclaw/workspace/vault/LOFN_MODEL_ASSIGNMENTS.md` before spawning or assigning music pipeline agents; follow its active step-specific model map unless The Scientist overrides it in the current request.
1b. **QUALITY CURRICULUM (MANDATORY):** Read `references/quality_curriculum.md` before generating any output. This contains our four benchmark songs (Triple Arch Over Me, Five Wrong Colors, I Will Stop the Almost, The Blue Screen Breathes) with full prompt and lyrics. Your output must try to match or beat this standard. These are our best work, tested on our platforms.
2. Read `references/music_full_legacy.md` before doing any substantive music work.
3. Read `TASK_TEMPLATE.md` before writing any final pair/song output. Its Step 10 output contract is mandatory.
4. For final song delivery, read `references/producer_grade_suno_prompt_guide.md` and `steps/10_Generate_Music_Revision_Synthesis.md` before writing Step 10 or equivalent final files.
4a. **Suno Prompt Construction Guide (MANDATORY for Steps 10-12):** Read `/data/.openclaw/workspace/vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md` before writing any Suno music prompt in Step 10 (synthesis), Step 11 (enhancement), or Step 12 (panel audit). This is the permanent vault reference for producer-grade prompt construction; every prompt must pass the guide's opening-moment, spatial-language, kinetic-defect, bold-device, and acoustic-ban checks.
4b. **Step 07 model note:** Step 07 uses Qwen3.7 Max. MiMo V2.5 Pro was tested and removed from Step 07 because it repeatedly completed without writing the required artifact.
4c. **Split-Step Pair Execution (MANDATORY, 2026-05-29):** After coordinator Step 05, do NOT spawn one monolithic pair agent for Steps 06-10 by default. Lofn music is a LangChain-style LLM chain: use the dedicated configured agents in `vault/LOFN_MODEL_ASSIGNMENTS.md` for each step (`lofn-audio-step06`, `lofn-audio-step07`, `lofn-audio-step08`, `lofn-audio-step09`, `lofn-audio-step10`). Spawn one pair per step, up to 5 concurrent children, then advance to the next step after disk artifacts exist. This prevents context collapse, timeout churn, and unreliable per-spawn model overrides.
4d. **Step 11 - Enhancement:** After all pair agents complete Step 10, spawn enhancement agents (1 per pair, 5 concurrent max) using `lofn-audio-step11` per `vault/LOFN_MODEL_ASSIGNMENTS.md`. Each reads its Step 10 output + coordinator context + the 15-point QA checklist + `vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md`. Produces `pair_0X_step10_final_package_enhanced.md`. Current model: `openrouter/anthropic/claude-opus-4.8`. Timeout: 300-900s each. This is now a permanent pipeline step - do not skip it.
4e. **Step 12 - Panel-of-Panels Prompt Audit (SCIENTIST-GATED):** After all Step 11 enhanced packages are complete and QA passes, convene a Poe-style panel of producers using `lofn-audio-step12` to audit all 6 Suno prompts against gold-standard reference songs. Output: `STEP12_MUSIC_PROMPT_AUDIT.md`. Current model: `openrouter/qwen/qwen3.7-max` per `vault/LOFN_MODEL_ASSIGNMENTS.md`.
5. If the task is an accessible broad-release run, read `../lofn-core/assets/seed_packet.template.json`, `../lofn-core/references/archetypes.md`, and `../qa/references/eligibility_7_properties.md` as needed.
6. If using a specific archetype, read only its card from `../lofn-core/references/archetype_*.md`.
7. For pair-agent or multi-step execution, read `../orchestration/references/warm_handoff_checkpoint.md` and write checkpoints after every major step.
8. **Require a real orchestrator packet before audio execution.** Validate Lofn-Core + orchestrator artifacts upstream: `01_seed_lineage.md`, `02_golden_seed.md` or `core_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, and `06_audio_handoff.md`. Validate with `scripts/validate_orchestrator_packet.py <run_dir>` before coordinator Step 00. If it fails, stop and request/launch real `lofn-orchestrator` work. **The `06_audio_handoff.md` must contain the Immutable Continuity Block (ICB)** per `vault/PIPELINE_CONTINUITY_STANDARD.md`.
9. **Lean pair-agent input standard:** after Step 05, parent-spawned pair agents must receive a compact pair brief, not the full upstream packet. Read `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md` before spawning or writing pair-agent prompts.
10. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`, with the Simple Surface / Complex Engine overrides in `references/simple_surface_complex_engine.md`, `references/golden_seed_alloy.md`, Step 05-10 prompts, and `../qa/references/suno_15_point_qa.md`.
11. Preserve artifact names, step order, Suno prompt requirements, lyric requirements, EMO header requirements, and subagent split rules from the legacy text.
12. **Original-Lofn step fidelity:** Steps 00-05 must be six separate prompt/response turns with canonical saved artifacts. Steps 06-10 must be five separate prompt/response turns per pair, each with its own saved artifact. Use dedicated configured step agents per `vault/LOFN_MODEL_ASSIGNMENTS.md`; do not rely on `lofn-audio` default model or per-spawn overrides for the split chain. Steps 09-10 are Chinese/DeepSeek-family models (`lofn-audio-step09` / Qwen3.7 Max, `lofn-audio-step10` / DeepSeek V4 Pro). Step 11 is the only expensive Claude pass: `lofn-audio-step11` / `openrouter/anthropic/claude-opus-4.8`. Step 12 uses `lofn-audio-step12` / `openrouter/qwen/qwen3.7-max`.
13. Every canonical step artifact must include call/response provenance using `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md`.
14. After every step artifact, run `scripts/validate_with_retries.py <step> <file>` with up to 3 attempts.
15. Do not call music generation tools; this skill writes Suno-ready text artifacts only.


## Validator-Aligned Artifact Rules — 2026-05-29

These rules exist because the validator is literal. Follow them exactly; do not infer alternate names or headings.


### Actual music step template paths

Use these exact files. Do not invent alternate step filenames.

- Step 06: `steps/06_Generate_Music_Facets.md`
- Step 07: `steps/07_Generate_Music_Song_Guides.md`
- Step 08: `steps/08_Generate_Music_Generation.md`
- Step 09: `steps/09_Generate_Music_Artist_Refined.md`
- Step 10: `steps/10_Generate_Music_Revision_Synthesis.md`
- Step 11: `steps/11_Generate_Music_Enhancement.md`

There is no `08_Generate_Music_Generation_Artifact.md`. If you look for it, stop and use `steps/08_Generate_Music_Generation.md`.

### Canonical filenames

- Step 08 canonical file is `pair_{NN}_step08_generation.md`. Do not write only `pair_{NN}_step08_music_prompts.md`; that name is a compatibility sidecar only.
- Step 10 canonical file is `pair_{NN}_step10_revision_synthesis.md`.
- Step 11 canonical file is `pair_{NN}_step10_final_package_enhanced.md`.

### Step 08 and Step 10 required section shape

For every Step 08 or Step 10 artifact, include these exact top-level sections in this order where possible:

1. `## 0. Step Provenance`
2. `## 1. Input Context Digest`
3. `## Continuity Payload Used`
4. `## 1. MUSIC PROMPT`
5. `## 2. LYRICS` or `## 2. SUNO LYRICS`
6. Production / notes / self-critique / validation sections

The validator extracts the music prompt from the text between `## 1. MUSIC PROMPT` and the next lyrics heading or next `##` heading. Therefore:

- Put exactly ONE standalone generator prompt body under `## 1. MUSIC PROMPT`.
- The extracted prompt body must be **850-1000 characters**. Count characters before saving.
- Do not put four long variation prompts under the same `## 1. MUSIC PROMPT` heading; that creates a 3000-6000 character prompt and fails validation.
- If preserving 4 variations, put portfolio notes outside the validated prompt section, e.g. `## Variation Portfolio Notes`, after the lyrics or production notes.
- The first prompt line must begin with genre/style + tempo/energy + vocalist/instrumentation, not narrative/procedural phrasing.

### Lyrics validator requirements

- Lyrics section must begin immediately with `[Theme: ...]` then `[SONG FORM: ...]`.
- Use full EMO headers: `[Section - EMO:<emotion> - <Role> - <cue>]`. Bare `[EMO: ...]` fails.
- Include at least one standalone SFX cue line like `*steam hiss*`.
- Lyrics need at least 60 sung lines; target 70-120. Step 08 drafts must still satisfy this if they include lyrics.

### Provenance validator requirements

Every step artifact must include:

- `## 1. Input Context Digest`
- validation command provenance, e.g. `Validation command: python3 scripts/validate_with_retries.py 08 <file>`
- `Continuity Payload Used` with the plural marker `Special Flairs`
- self-critique and validation result

Avoid the literal words `placeholder` or `template` in final artifacts, even in self-critique/provenance, because validator gates may flag them. Use `stub text` or `scaffold text` instead.

### Subagent behavior

- Do not read validator source files unless explicitly asked to debug the validator. Use the validation command, not the source.
- Write the artifact early, then repair in place. Do not spend the whole run reading references.
- For repair tasks, change only the failing section unless the prompt explicitly asks for broader repair.

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context. The coordinator stops after Step 05; parent/controller spawns independent step agents.
- Do not collapse or bypass Lofn-Core + lofn-orchestrator. Audio agents are not allowed to self-author a shallow orchestrator replacement.
- **MANDATORY 3-PANEL ORCHESTRATOR STRUCTURE:** The orchestrator panel debate MUST convene THREE separate panels of 6 expert voices each = 18 total voices, each with its own Hyper-Skeptic/Devil's Advocate. A single 6-voice debate room is a COLLAPSE FAILURE.
- **MANDATORY CARDINALITY:** Every music run produces 6 pairs × 4 variations = 24 songs minimum unless The Scientist explicitly overrides.
- Do not collapse Steps 00-05 into one coordinator prompt/response.
- Do not collapse Steps 06-10 into one pair-agent prompt/response. Original Lofn calls each step sequentially; OpenClaw runs must mirror that granularity with separate artifacts.
- **Final music deliverables MUST include a standalone Suno/Udio style prompt for every song.**
- **Required Suno prompt shape:** target 850-1000 characters, no artist names, dense producer-grade language. Order: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation/sound palette with physical adjectives → musical arrangement arc with narrative → bold sonic device → avoidances. Must include vivid opening moment, spatial language, kinetic defect, and explicit acoustic/no-acoustic declarations.
- **Required lyric length:** 70-120 sung lines for a 3:00-4:00 minute runtime. <60 lines triggers QA repair.
- **Theme + Song Form Blocks (MANDATORY):** Every final lyric block starts with `[Theme: ...]` followed immediately by `[SONG FORM: ...]`.
- **EMO Header Format (MANDATORY):** Every lyrics section header uses `[Section - EMO:<emotion(s)> - <Role> - <cues>]`. Use `/data/.openclaw/workspace/skills/lofn-core/refs/EMOTION_TAXONOMY.md`.
- **Verse Structure Diversity Rule:** No two songs in the same run may default to the same rigid quatrain-based verse structure. Rhyme schemes and poetic techniques must differ across songs.
- A song guide with lyrics and production notes but no standalone Suno style prompt is incomplete.
- Validation is inline, not just final QA.
- Pair completion validation is mandatory: `scripts/validate_pair_artifacts.py <audio_dir> <NN> --attempt 1`.
- Provenance is mandatory.
- Cross-pair distinctiveness is mandatory at multiple stages.
- **Design toward the Suno 15-Point QA Gate** (`../qa/references/suno_15_point_qa.md`).
- **Original Lofn 3-panel object is mandatory input to music.** The 3 Hyper-Skeptics form a collective SOMATIC GATE.
- **Continuity payload is mandatory at EVERY step:** Special Flairs, panel pressures, Golden Seed/pair seed excerpt, active personality/persona, and source/news thread where applicable.
- **Immutable Continuity Block (ICB) format:** `06_audio_handoff.md` must carry an `⚠️ IMMUTABLE CONTINUITY BLOCK — DO NOT SUMMARIZE` section. Every downstream step must carry it or a faithful required compact excerpt.

## Creative Ordering Correction - 2026-05-10

The Suno 15-point gate remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active personality/persona, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, file names, and safety requirements.

Never lead a creative music agent with the checklist.

### Personality-Specific Sonic Identity Gate

Every final song package must prove which active personality/persona made it:

- **Active personality named**
- **Personality sonic world sentence**
- **Personality signature device**
- **Seed-derived weirdness preserved**

If a song could have been written without the named personality, mark it for creative repair before QA delivery.

### Step-to-Step Continuity Payload

To prevent Step 07-10 blandification, every music prompt must carry a small but living packet forward.

Required fields:
- `special_flairs`
- `panel_pressures`
- `seed_excerpt`
- `active_personality`
- `news_or_source_thread`

Each step artifact must include a `Continuity Payload Used` section or equivalent JSON fields. QA must treat missing continuity payload as `REPAIR — THREAD LOSS`, even if Suno formatting passes.
