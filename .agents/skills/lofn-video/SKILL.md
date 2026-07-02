---
name: lofn-video
description: Run the Lofn video/animation pipeline (steps 00–10) backed by Codex — cinematic shot lists and motion/animation prompts (Veo 3.1 formula, synchronized audio). Use for video, film, cinematic clips, animation, animated loops, motion design, or "make a video/animation with the full pipeline". Covers BOTH live-cinematic and animation. Expects a Phase-0/1 orchestrator packet from the `lofn` skill; if none exists, run `lofn` first. Do NOT use for static images, music-only, story prose, or QA-only audits.
---

# Lofn Video & Animation — Codex-backed director pipeline

> **⚖️ AUTHORITY (2026-07-01):** the `.claude/skills/` twin of this skill is the CANONICAL policy source; this Codex mirror binds to it and to `.agents/skills/lofn/EXECUTION.md` §8 (Policy Deltas — golden-output quarantine, no-skip/NON-CANONICAL, itemized packet, per-pair variation angles, judge separation, the publish bar, gate mid-bands). On any disagreement, the `.claude` file wins.

Produces cinematic shot lists and animation prompts at Lofn competition grade. This one skill covers **both** modalities the user calls "video" and "animation" — same pipeline, with the Veo 3.1 motion/audio formula and the animation archetypes layered on. Depth lives in `skills/video/` and `skills/animator/`; Codex is the engine (hybrid execution per `.agents/skills/lofn/EXECUTION.md`).

## Before you start
1. Confirm a Phase-0/1 packet exists (`core_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `06_director_handoff.md` with the **ICB / Panel Ledger**, filled `CREATIVE_CONTEXT.md`). **No packet → run `lofn` first.**
2. Read the motion/format craft just-in-time:
   - `skills/animator/SKILL.md` — Veo 3.1 prompt formula, camera language, audio direction, the 6 animation archetypes (Pulse / Morph / Orbit / Parallax / Burst / Flow), loop logic, platform optimization.
   - `vault/DIRECTOR_QA_DEPTH_AUDIT.md` — the Cinematic Somatic Gate + 5-element shot checklist.
3. **Mode:** *Cinematic* (multi-shot sequence / narrative clip) vs *Animation* (a single 4–8s loop/motion study). The pipeline is the same; in Animation mode each pair's variations are archetype-driven loops and you enforce loop logic (how frame 1 connects to the final frame).

## Execution (hybrid)
Coordinator **00-05 inline**, then **6 pairs as parallel subagents** for 06-10. At every agent start, inject the full startup packet from EXECUTION.md Section 3: CREATIVE_CONTEXT.md verbatim, the handoff/assignment files, current step contract, and prior artifact where applicable. Default cardinality: **6 pairs x 4 = 24 shot-sets / loops -> rank -> top picks.**

### Coordinator steps (inline)
| Step | File | Artifact |
|------|------|----------|
| 00 | `skills/video/steps/00_Generate_Video_Aesthetics_And_Genres.md` | `step00_aesthetics_and_genres.md` |
| 01 | `skills/video/steps/01_Generate_Video_Essence_And_Facets.md` | `step01_essence_and_facets.md` |
| 02 | `skills/video/steps/02_Generate_Video_Concepts.md` | `step02_concepts.md` (12 concepts) |
| 03 | `skills/video/steps/03_Generate_Video_Artist_And_Critique.md` | `step03_artist_and_critique.md` |
| 04 | `skills/video/steps/04_Generate_Video_Medium.md` | `step04_medium.md` |
| 05 | `skills/video/steps/05_Generate_Video_Refine_Medium.md` | `step05_refine_medium.md` → **6 pairs** |

### Per-pair steps (parallel subagents, one chain per pair)
| Step | File | Per-pair artifact |
|------|------|-------------------|
| 06 | `skills/video/steps/06_Generate_Video_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/video/steps/07_Generate_Video_Aspects_Traits.md` | `pair_{NN}_step07_aspects_traits.md` (shot guide) |
| 08 | `skills/video/steps/08_Generate_Video_Generation.md` | `pair_{NN}_step08_generation.md` (4 shot-sets/loops) |
| 09 | `skills/video/steps/09_Generate_Video_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/video/steps/10_Generate_Video_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |

## The video/animation prompt contract (hard gate)
Each generation prompt (step 08+) follows the Veo 3.1 formula, one element per concern:

```
[CAMERA]  shot type + angle + movement      (front-load this — Veo prioritizes framing)
[SUBJECT] specific, not generic             ("a woman in worn leather jacket, silver rings", not "a person")
[ACTION]  exactly what happens
[SETTING] environment + time + weather/light
[STYLE & AUDIO]  aesthetic + explicit sound design
```
- **Separate camera from action** (each its own sentence). Describe negative space.
- **Audio is directed explicitly** — Dialogue in quotes (`A woman whispers, "I remember everything."`), `SFX:` prefix, `Ambient:` soundscape, `Audio:` music. Layer ambient + SFX (+ optional music) for depth.
- **Real film references** allowed for look ("Terrence Malick golden hour"); **no living-artist/actor likeness**, no real victims.
- **Animation mode adds loop logic:** name the archetype (Pulse/Morph/Orbit/Parallax/Burst/Flow), set duration (4/6/8s), aspect (9:16 TikTok / 16:9 cinematic), resolution, and state how the loop closes (orbit completes 360°, pulse returns to origin, dolly reverses at midpoint, flow = landmark-free tunnel).
- The 6 pairs use **distinct camera grammar / archetypes** — no two pairs default to the same move (the distinctiveness rule).

## QA & delivery
Run **lofn-qa** with the **Cinematic Somatic Gate** (`vault/DIRECTOR_QA_DEPTH_AUDIT.md`, 5-element shot checklist). Rank by: opening-second hook → motion clarity → audio-image cohesion → emotional legibility → loop integrity (animation). Save each selected shot-set/loop per `skills/lofn-core/OUTPUT.md` (note intended renderer — Veo 3.1 — and aspect/duration in frontmatter); INDEX last. **Do not call render tools** — emit paste-ready prompt text; the user renders.
