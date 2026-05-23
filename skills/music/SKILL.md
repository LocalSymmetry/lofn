---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio - Router

This router prevents context collapse. The full tuned music pipeline text is preserved byte-for-byte in `references/music_full_legacy.md`; newer Simple Surface / Complex Engine references and Step 05–10 files ADD song-force / anti-blandness requirements but do not waive the legacy Suno/package QA gate unless The Scientist explicitly approves an exception for a specific run.

## Workflow

1. Confirm this is a music/audio task: Suno prompt, lyrics, song guide, audio pipeline, or music production brief.
2. Read `references/music_full_legacy.md` before doing any substantive music work.
3. Read `TASK_TEMPLATE.md` before writing any final pair/song output. Its Step 10 output contract is mandatory.
4. For final song delivery, read `references/producer_grade_suno_prompt_guide.md` and `steps/10_Generate_Music_Revision_Synthesis.md` before writing Step 10 or equivalent final files.
5. If the task is an accessible broad-release run, read `../lofn-core/assets/seed_packet.template.json`, `../lofn-core/references/archetypes.md`, and `../qa/references/eligibility_7_properties.md` as needed.
6. If using a specific archetype, read only its card from `../lofn-core/references/archetype_*.md`.
7. For pair-agent or multi-step execution, read `../orchestration/references/warm_handoff_checkpoint.md` and write checkpoints after every major step.
8. **Require a real orchestrator packet before audio execution.** The coordinator/controller must validate Lofn-Core + orchestrator artifacts created upstream, not invent them locally: `01_seed_lineage.md`, `02_golden_seed.md` or `core_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, and `06_audio_handoff.md`. Validate with `scripts/validate_orchestrator_packet.py <run_dir>` before coordinator Step 00. If it fails, stop and request/launch real `lofn-orchestrator` work.
9. **Lean pair-agent input standard:** after Step 05, parent-spawned pair agents must receive a compact pair brief, not the full upstream packet. Read `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md` before spawning or writing pair-agent prompts. The parent retains/validates the full packet; each pair agent normally receives a compact Golden Seed operating excerpt plus Step 05, `concept_medium_pairs.json`, one pair assignment excerpt, step contract, tiny provenance, and modality-specific blockers.
10. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`, with the Simple Surface / Complex Engine overrides in `references/simple_surface_complex_engine.md`, `references/golden_seed_alloy.md`, Step 05–10 prompts, and `../qa/references/suno_15_point_qa.md`.
11. Preserve artifact names, step order, Suno prompt requirements, lyric requirements, EMO header requirements, and subagent split rules from the legacy text. The newer Simple Surface / Complex Engine standard is additive: it must improve song-force, mythic pressure, and panel routing without weakening legacy package QA.
12. **Original-Lofn step fidelity:** Steps 00–05 must be six separate prompt/response turns with canonical saved artifacts (`step00_aesthetics_and_genres.md` ... `step05_refine_medium.md`). Steps 06–10 must be five separate prompt/response turns per pair, each with its own saved artifact (`pair_{NN}_step06_facets.md` ... `pair_{NN}_step10_revision_synthesis.md`). Do not write only summary files or `pair_{NN}_steps_06_10.md`; that is a collapsed shortcut and fails QA.
13. Every canonical step artifact must include call/response provenance using `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md`: loaded step file path, input artifacts used, input digest, step-template requirements applied, creative response, self-critique, and validation result. Filenames alone do not prove execution. Coordinator provenance cites the full validated upstream packet; pair-agent provenance cites the lean pair-agent input plus the parent-validated upstream packet summary.
14. After every step artifact, run `scripts/validate_with_retries.py <step> <file>` with up to 3 attempts. Repair locally between attempts. After 3 failures, checkpoint and escalate instead of continuing.
15. Do not call music generation tools; this skill writes Suno-ready text artifacts only.

## SUNO FORMAT REFERENCE EXAMPLES (MANDATORY READING)

Before writing ANY Suno lyrics, the agent MUST read these three reference examples. They define the correct formatting standard. Every future song guide must match this quality:

1. **`references/suno_format_example_triple_arch.md`** — "Triple Arch Over Me" (Suno Staff Pick). Shows: `[Theme: ...]` header, `[Section – EMO:... – Vocalist – cues]` format, clean lyrics, prose music prompt.
2. **`references/suno_format_example_blue_screen.md`** — "2:07 and the Blue Screen Breathes". Shows: `[EMO: ...]` tags on separate lines under section headers, SFX cues inline.
3. **`references/suno_format_example_five_wrong_colors.md`** — "Five Wrong Colors". Shows: movement-based structure, inline EMO tags in section headers, silence cues, fragmented refrains.

**Format law (from these examples):**
- First lyric line: `[Theme: <specific scene-pressure / emotional operating system>]` — NOT a generic topic
- Second line: `[SONG FORM: <named musical form and sequence>]` — NOT just "pop" or "dance"
- Section headers: `[Section – EMO:<emotion(s)> – Vocalist – cues]` — EMO tags on EVERY section
- Music prompt: prose paragraph, opens with genre + tempo, NO artist names, NO tag soup, NO bullet lists
- ZERO markdown in lyric blocks (no bold, no italics, no ### headers)
- ZERO parenthetical prose directions `*(like this)*` in or near lyrics
- SFX cues use `*asterisks*` within lyrics
- Lyric count: 70-120 lines for 3:00-4:00 runtime

**Anti-patterns that cause rejection:**
- Bare `[Verse]` / `[Chorus]` without EMO tags
- Uppercase section headers: `[INTRO]`, `[VERSE 1]`, `[CHORUS]`
- Markdown bold headers: `**[Chorus]**`
- Parenthetical instrument directions: `*(Sub-bass swells...)*`
- Timestamps in headers: `[Chorus (1:00-1:30)]`
- Prose music prompts that open with "Begin in/by/with..."
- Tag soup in music prompts

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context. The coordinator stops after Step 05; the parent/controller spawns one independent `lofn-audio` child session per pair for Steps 06-10. Local emulation of pair agents is a blocking pipeline-integrity failure even if all filenames exist.
- **Verse Structure Diversity Rule (2026-05-23):** No two songs in the same run may default to the same rigid quatrain-based verse structure (4/8/12 lines). A dailies run with 6 songs must use 6 DIFFERENT verse architectures drawn from: tercet-based (3/6/9 lines), quintain (5/10 lines, AABBA or ABABB), septet (7 lines), couplet chains (2-line AA BB CC DD units), enjambment-heavy variable, free verse with caesura, embrace-rhyme irregular (ABBA with varied line lengths), irregular/prose verse, or single-line verse blocks. Rhyme schemes must also differ across songs: use slant rhyme, internal rhyme, no end-rhyme, consonance chains, echo rhyme, assonance-based structure, and enclosed ABBA — never default all songs to AABB/ABAB. Each song must employ a distinct poetic construction technique (staccato fragmentation, question-as-structure, syllable compression, caesura-as-wound, prose-poetry continuity, breath-length variation). The orchestrator assigns verse structure type, rhyme scheme, and poetic technique per pair; QA verifies all 6 are different.
- Do not collapse or bypass Lofn-Core + lofn-orchestrator. Audio agents are not allowed to self-author a shallow orchestrator replacement and proceed; a real 3-panel orchestrator packet is a launch prerequisite.
- Do not collapse Steps 00–05 into one coordinator prompt/response. Original Lofn calls `process_essence_and_facets`, `process_concepts`, `process_artist_and_refined_concepts`, `process_mediums`, and `process_refined_mediums` sequentially; OpenClaw runs must mirror that granularity with canonical step artifacts.
- Do not collapse Steps 06–10 into one pair-agent prompt/response. Original Lofn calls `process_facets`, `process_song_guides`, `process_music_generation_prompts`, `process_music_artist_refined_prompts`, and `process_music_revision_synthesis` sequentially; OpenClaw runs must mirror that granularity with separate artifacts. If outputs look like one-shot backfilled files, missing required headers, or contradicted self-checks, QA must fail even when filenames exist.
- **Final music deliverables MUST include a standalone Suno/Udio style prompt for every song.** This is not the same as `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, or `[PRODUCTION NOTES]`. The final file must contain a clearly labeled section such as `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` with a copy-paste-ready, single-paragraph prompt.
- **Required Suno prompt shape:** target 850-1000 characters (hard max 1000 unless the destination explicitly allows longer), no artist names, dense producer-grade language. **Order is mandatory: selected genre/style label(s) + tempo/energy → vocalist spec → instrumentation/sound palette/mix → musical arrangement arc → bold sonic device → avoidances.** Do **not** lead with story, theme, or procedural language. Banned prompt openings include “Begin in/by/with…”, “Use…”, “Build the track from…”, “Chronology:”, and “For an adult human singer…” as the first clause. The Poe panel standard adds: do not pad with generic tag soup; if the prompt needs extra intelligence, route it into Sonic Manifest / Production Cathedral sidecars while keeping the copy-paste prompt producer-grade.
- **Required lyric length:** 70-120 sung lines for a 3:00-4:00 minute runtime. <60 lines triggers QA repair. The Poe panel standard adds: reach length through hook recurrence, chorus mutation, bridge pressure, call-response, ghost/echo reprises, and embodied image development — never through procedural exposition or filler.
- **Theme + Song Form Blocks (MANDATORY):** Every final Suno lyric block starts with `[Theme: <specific scene-pressure / emotional operating system>]` followed immediately by `[SONG FORM: <named musical form and sequence>]`. Theme is not a generic topic; it is a focusing compression field for Suno and the writing agent. Song Form is not just “pop” or “dance”; it names the actual architecture and key transitions. Missing either block is a blocking Step 10 failure.
- **EMO Header Format (MANDATORY):** Every lyrics section header uses the full Suno performance-script format: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`. The emotion must be drawn from `/data/.openclaw/workspace/skills/lofn-core/refs/EMOTION_TAXONOMY.md`. Do NOT use bare Lofn architectural states (AWE/INDIGNATION/SYNTHESIS) as emotion labels. The Poe panel standard adds: headers may carry performance information, but sung lines must never contain EMO taxonomy, prompt language, QA notes, or production manuals.
- A song guide with lyrics and production notes but no standalone Suno style prompt is **incomplete**, even if the lyrics are excellent.
- Validation is inline, not just final QA. Every generated step file must pass `scripts/validate_with_retries.py` before the agent advances to the next step. The retry budget is exactly 3 attempts per artifact, mirroring the original app's retry discipline but applying it to artifact correctness.
- Pair completion validation is also mandatory. After Steps 06-10 exist for a pair, run `scripts/validate_pair_artifacts.py <audio_dir> <NN> --attempt 1`; repair and rerun with attempts 2-3 as needed. Do not write `pair_{NN}_COMPLETE.md` or return completion until the pair-level command prints `PAIR VALIDATION PASS`; paste that output into the COMPLETE file.
- Provenance is mandatory. If the artifact does not show the loaded step prompt, concrete prior inputs, model response, and self-critique, it is treated as backfilled mimicry even if the output shape looks correct. For pair-agent outputs, provenance must also include `session_key`, `spawned_by_parent: true`, `step_call_mode: separate_child_session`, `source_golden_seed`, `golden_seed_excerpt_included: true`, `source_step05`, `source_pair_list`, and a collapse guard. Pair-agent inputs should follow the lean standard in `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md`; missing execution provenance blocks Pipeline Integrity PASS.
- Cross-pair distinctiveness is mandatory at multiple stages. A set can pass every single-file gate and still fail if Step 06 flattened pair-specific facets, Step 09 reused the same refinement skeleton, or Step 10 reused the same lyric/prompt skeleton across pairs. Run `scripts/validate_step06_distinctiveness.py <audio_dir>` after Step 06, `scripts/validate_step09_distinctiveness.py <audio_dir>` after Step 09, and `scripts/validate_portfolio_distinctiveness.py <audio_dir>` after Step 10. Failures are repair blockers, not advisory notes.
- **Design toward the Suno 15-Point QA Gate** (`../qa/references/suno_15_point_qa.md`): 7 Singer Surface checks plus 5 Cathedral Engine checks plus 3 Suno package checks. Do not optimize only for research compliance; optimize for body, adoptable hook, mythic image pressure, panel-forced revision, active-personality fidelity, 15-30 second survivability, and paste-ready Suno package. A catchy novelty lane is only valid when it matches the run's selected personality/persona; otherwise it is style drift.
- **Original Lofn 3-panel object is mandatory input to music:** every audio coordinator and pair agent must receive/use the orchestrator's `Special Flairs`, `Concept Panel`, `Medium Panel`, and `Context & Marketing Panel`, each with Devil's Advocate / Hyper-Skeptic. The panel object must be routed into Step 05/07/09/10 Panel Ledger decisions and QA gate #12; if missing, stop and request/launch orchestrator repair.

## Creative Ordering Correction - 2026-05-10

The Suno 15-point gate remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active personality/persona, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong - form, color, meter, harmony, vocal treatment, rupture timing, silence, ugliness, refusal, asymmetry.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template. Accessibility belongs in the hook, emotional premise, and navigable form; the personality's materials and production language belong in the musicscape.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, file names, and safety requirements.

Never lead a creative music agent with the checklist. If the first thing the agent sees is `850-1000 chars / 70-120 lines / EMO headers`, it will write to satisfy the form instead of the seed. Lead with Golden Seed, 3-panel object, Singer Surface, Mythic Image Pressure, and Cathedral Engine; then require the full legacy QA gate at the end. Song-force and package QA must BOTH pass.

### Personality-Specific Sonic Identity Gate

Every final song package must prove which active personality/persona made it. Before final delivery, each song must include:

- **Active personality named** - identify the selected personality/persona from the orchestrator or seed.
- **Personality sonic world sentence** - "This song's world is made from ___, ___, and ___," using materials, places, instruments, textures, or rituals that belong to that personality.
- **Personality signature device** - one named sonic move that this personality would plausibly invent.
- **Seed-derived weirdness preserved** - at least one concrete fact/material/measurement, deliberate wrongness, structural asymmetry, rupture, witness/prayer mode, or other seed-specific artistic pressure remains audible.

If a song could have been written without the named personality, mark it for creative repair before QA delivery.
