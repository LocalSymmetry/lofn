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
8. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`.
9. Preserve all artifact names, step order, Suno prompt requirements, lyric requirements, and subagent split rules from the legacy text.
10. Do not call music generation tools; this skill writes Suno-ready text artifacts only.

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context.
- **Final music deliverables MUST include a standalone Suno/Udio style prompt for every song.** This is not the same as `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, or `[PRODUCTION NOTES]`. The final file must contain a clearly labeled section such as `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` with a copy-paste-ready, single-paragraph prompt.
- **Required Suno prompt shape:** target 850-1000 characters (hard max 1000 unless the destination explicitly allows longer), no artist names, dense producer-grade language: emotion → precise genre → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → blacklist/avoidances.
- **Required lyric length:** 70-120 sung lines for a 3:00-4:00 minute runtime. <60 lines triggers QA repair. <70 lines risks under-3min output.
- **EMO Header Format (MANDATORY, added 2026-05-11):** Every lyrics section header uses the `EMO:` prefix to express the precise emotion of that section from the full emotional taxonomy. Format: `[Section – EMO:<emotion(s)> – <Role> – <cues>]`. The emotion must be drawn from `/data/.openclaw/workspace/skills/lofn-core/refs/EMOTION_TAXONOMY.md` — choose the specific emotion or 2–3 combination that the section needs to land (e.g. `EMO:nostalgia + yearning`, `EMO:righteous fury`, `EMO:tender grief`, `EMO:ecstatic release`). Do NOT use the bare Lofn architectural states (AWE/INDIGNATION/SYNTHESIS) as emotion labels — those are coarse duality categories, not a section-level emotional palette. The `EMO:` tag is where you nail the actual human feeling. This rule is non-negotiable and must be included in every music agent task prompt.
- A song guide with lyrics and production notes but no standalone Suno style prompt is **incomplete**, even if the lyrics are excellent.
- **Design toward the Suno 15-Point QA Gate** (`../qa/references/suno_15_point_qa.md`): 7 eligibility checks plus 8 delivery/creative survival checks. Do not optimize only for research compliance; optimize for body, adoptable hook, active-personality fidelity, 15–30 second survivability, and paste-ready Suno package. A catchy novelty lane is only valid when it matches the run's selected personality/persona; otherwise it is style drift.

## Creative Ordering Correction — 2026-05-10

The Suno 15-point gate remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active personality/persona, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong — form, color, meter, harmony, vocal treatment, rupture timing, silence, ugliness, refusal, asymmetry.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template. Accessible songs may be hook-forward, but must not become generic compliance pop. Accessibility belongs in the hook, emotional premise, and navigable form — not in bland musicscape defaults.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, anti-slop checks, file names, and safety requirements.

Never lead a creative music agent with the checklist. If the first thing the agent sees is `850-1000 chars / 70-120 lines / EMO tags`, it will write to satisfy the form instead of the seed. These requirements are still blocking QA gates; they are just not the muse.

### Personality-Specific Sonic Identity Gate

Every final song package must be **personality-accessible**, not generically accessible. The listener should be able to enter the song quickly, but the sound world must prove which active personality/persona made it. Accessibility may simplify the hook, emotional premise, and navigable form; it may **not** flatten the musicscape into bland pop, bland rock, generic cinematic ballad, default EDM, or stock “AI emotional” sludge.

Before final delivery, each song must include and survive these checks in the **music prompt and lyrics/production notes**:

- **Active personality named** — identify the selected personality/persona from the orchestrator or seed.
- **Personality sonic world sentence** — “This song’s world is made from ___, ___, and ___,” using materials, places, instruments, textures, or rituals that belong to that personality.
- **Personality signature device** — one named sonic move that this personality would plausibly invent and a generic prompt writer would not.
- **Nearest slop failure mode blacklisted** — explicitly forbid the generic version this song could collapse into (e.g. bland pop-rock, generic cinematic uplift, decorative glitch, stock EDM drop, anonymous inspirational ballad).
- **Seed-derived weirdness preserved** — at least one concrete fact/material/measurement, deliberate wrongness, structural asymmetry, rupture, witness/prayer mode, or other seed-specific artistic pressure remains audible.

If a song is technically complete but could have been written by a competent generic Suno prompt writer, or if it sounds like an arbitrary genre lane rather than the active personality, mark it for creative repair before QA delivery.
