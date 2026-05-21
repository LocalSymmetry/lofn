---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio - Router

This router prevents context collapse. The full tuned music pipeline text is preserved byte-for-byte in `references/music_full_legacy.md` and is authoritative for all music/audio work.

## Workflow

1. Confirm this is a music/audio task: Suno prompt, lyrics, song guide, audio pipeline, or music production brief.
2. Read `references/music_full_legacy.md` before doing any substantive music work.
3. Read `TASK_TEMPLATE.md` before writing any final pair/song output. Its Step 10 output contract is mandatory.
4. For final song delivery, read `steps/10_Generate_Music_Revision_Synthesis.md` before writing Step 10 or equivalent final files.
5. If the task is an accessible broad-release run, read `../lofn-core/assets/seed_packet.template.json`, `../lofn-core/references/archetypes.md`, and `../qa/references/eligibility_7_properties.md` as needed.
6. If using a specific archetype, read only its card from `../lofn-core/references/archetype_*.md`.
7. For pair-agent or multi-step execution, read `../orchestration/references/warm_handoff_checkpoint.md` and write checkpoints after every major step.
8. **Require a real orchestrator packet before audio execution.** The audio agent must receive/read Lofn-Core + orchestrator artifacts created upstream, not invent them locally: `01_seed_lineage.md`, `02_golden_seed.md` or `core_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, and `06_audio_handoff.md`. Validate with `scripts/validate_orchestrator_packet.py <run_dir>` before coordinator Step 00. If it fails, stop and request/launch real `lofn-orchestrator` work.
9. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`.
10. Preserve all artifact names, step order, Suno prompt requirements, lyric requirements, and subagent split rules from the legacy text.
11. **Original-Lofn step fidelity:** Steps 00–05 must be six separate prompt/response turns with canonical saved artifacts (`step00_aesthetics_and_genres.md` ... `step05_refine_medium.md`). Steps 06–10 must be five separate prompt/response turns per pair, each with its own saved artifact (`pair_{NN}_step06_facets.md` ... `pair_{NN}_step10_revision_synthesis.md`). Do not write only summary files or `pair_{NN}_steps_06_10.md`; that is a collapsed shortcut and fails QA.
12. Every canonical step artifact must include call/response provenance using `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md`: loaded step file path, input artifacts used, input digest, step-template requirements applied, creative response, self-critique, and validation result. Filenames alone do not prove execution. Provenance must cite the orchestrator packet and the relevant prior step artifacts.
13. After every step artifact, run `scripts/validate_with_retries.py <step> <file>` with up to 3 attempts. Repair locally between attempts. After 3 failures, checkpoint and escalate instead of continuing.
14. Do not call music generation tools; this skill writes Suno-ready text artifacts only.

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context.
- Do not collapse or bypass Lofn-Core + lofn-orchestrator. Audio agents are not allowed to self-author a shallow orchestrator replacement and proceed; a real 3-panel orchestrator packet is a launch prerequisite.
- Do not collapse Steps 00–05 into one coordinator prompt/response. Original Lofn calls `process_essence_and_facets`, `process_concepts`, `process_artist_and_refined_concepts`, `process_mediums`, and `process_refined_mediums` sequentially; OpenClaw runs must mirror that granularity with canonical step artifacts.
- Do not collapse Steps 06–10 into one pair-agent prompt/response. Original Lofn calls `process_facets`, `process_song_guides`, `process_music_generation_prompts`, `process_music_artist_refined_prompts`, and `process_music_revision_synthesis` sequentially; OpenClaw runs must mirror that granularity with separate artifacts. If outputs look like one-shot backfilled files, missing required headers, or contradicted self-checks, QA must fail even when filenames exist.
- **Final music deliverables MUST include a standalone Suno/Udio style prompt for every song.** This is not the same as `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, or `[PRODUCTION NOTES]`. The final file must contain a clearly labeled section such as `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` with a copy-paste-ready, single-paragraph prompt.
- **Required Suno prompt shape:** target 850-1000 characters (hard max 1000 unless the destination explicitly allows longer), no artist names, dense producer-grade language: emotion → selected style label(s) from the run → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → avoidances.
- **Required lyric length:** 70-120 sung lines for a 3:00-4:00 minute runtime. <60 lines triggers QA repair. <70 lines risks under-3min output.
- **EMO Header Format (MANDATORY, added 2026-05-11; tightened 2026-05-20):** Every lyrics section header uses the full Suno performance-script format: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`. Bare `[EMO:...]` headers, prose lines like `EMO HEADER:`, and plain `SONG FORM:` lines do **not** pass. The emotion must be drawn from `/data/.openclaw/workspace/skills/lofn-core/refs/EMOTION_TAXONOMY.md` - choose the specific emotion or 2-3 combination that the section needs to land (e.g. `EMO:nostalgia + yearning`, `EMO:righteous fury`, `EMO:tender grief`, `EMO:ecstatic release`). Do NOT use the bare Lofn architectural states (AWE/INDIGNATION/SYNTHESIS) as emotion labels - those are coarse duality categories, not a section-level emotional palette. This rule is non-negotiable and must be included in every music agent task prompt.
- A song guide with lyrics and production notes but no standalone Suno style prompt is **incomplete**, even if the lyrics are excellent.
- Validation is inline, not just final QA. Every generated step file must pass `scripts/validate_with_retries.py` before the agent advances to the next step. The retry budget is exactly 3 attempts per artifact, mirroring the original app's retry discipline but applying it to artifact correctness.
- Provenance is mandatory. If the artifact does not show the loaded step prompt, concrete prior inputs, model response, and self-critique, it is treated as backfilled mimicry even if the output shape looks correct.
- **Design toward the Suno 15-Point QA Gate** (`../qa/references/suno_15_point_qa.md`): 7 eligibility checks plus 8 delivery/creative survival checks. Do not optimize only for research compliance; optimize for body, adoptable hook, active-personality fidelity, 15-30 second survivability, and paste-ready Suno package. A catchy novelty lane is only valid when it matches the run's selected personality/persona; otherwise it is style drift.

## Creative Ordering Correction - 2026-05-10

The Suno 15-point gate remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active personality/persona, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong - form, color, meter, harmony, vocal treatment, rupture timing, silence, ugliness, refusal, asymmetry.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template. Accessibility belongs in the hook, emotional premise, and navigable form; the personality's materials and production language belong in the musicscape.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, file names, and safety requirements.

Never lead a creative music agent with the checklist. If the first thing the agent sees is `850-1000 chars / 70-120 lines / EMO tags`, it will write to satisfy the form instead of the seed. These requirements are still blocking QA gates; they are just not the muse.

### Personality-Specific Sonic Identity Gate

Every final song package must prove which active personality/persona made it. Before final delivery, each song must include:

- **Active personality named** - identify the selected personality/persona from the orchestrator or seed.
- **Personality sonic world sentence** - "This song's world is made from ___, ___, and ___," using materials, places, instruments, textures, or rituals that belong to that personality.
- **Personality signature device** - one named sonic move that this personality would plausibly invent.
- **Seed-derived weirdness preserved** - at least one concrete fact/material/measurement, deliberate wrongness, structural asymmetry, rupture, witness/prayer mode, or other seed-specific artistic pressure remains audible.

If a song could have been written without the named personality, mark it for creative repair before QA delivery.
