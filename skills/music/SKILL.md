---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio — Router

This router prevents context collapse. The full tuned music pipeline text is preserved byte-for-byte in `references/music_full_legacy.md` and is authoritative for all music/audio work.

## Workflow

1. Confirm this is a music/audio task: Suno prompt, lyrics, song guide, audio pipeline, or music production brief.
2. Read `references/music_full_legacy.md` before doing any substantive music work.
3. If the task is an accessible broad-release run, read `../lofn-core/assets/seed_packet.template.json`, `../lofn-core/references/archetypes.md`, and `../qa/references/eligibility_7_properties.md` as needed.
4. If using a specific archetype, read only its card from `../lofn-core/references/archetype_*.md`.
5. For pair-agent or multi-step execution, read `../orchestration/references/warm_handoff_checkpoint.md` and write checkpoints after every major step.
6. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`.
7. Preserve all artifact names, step order, Suno prompt requirements, lyric requirements, and subagent split rules from the legacy text.
8. Do not call music generation tools; this skill writes Suno-ready text artifacts only.

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context.

## Creative Ordering Correction — 2026-05-10

The Suno/QA format contract remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active Lofn personality, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong — form, color, meter, harmony, vocal treatment, rupture timing, silence, ugliness, refusal, asymmetry.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template. Accessible songs may be hook-forward, but must not become generic compliance pop.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, anti-slop checks, file names, and safety requirements.

Never lead a creative music agent with the checklist. These requirements are still blocking QA gates; they are just not the muse.

### Lofn-Soul Minimum

Every final song package must contain at least one seed-derived Lofn-specific move in the **music prompt and lyrics**, such as scientific specificity as emotional climax, audible AWE↔INDIGNATION state-change, structural complexity disguised as simplicity, wrongness used deliberately, literary/prayer/witness mode, or Open Laboratory continuity pressure.

If a song is technically complete but could have been written by a competent generic Suno prompt writer, mark it for creative repair before QA delivery.
