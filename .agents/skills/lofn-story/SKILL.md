---
name: lofn-story
description: Run the Lofn story/narrative pipeline (steps 00–10) backed by Codex — panel-driven prose, narrative voice, world-building, scripts. Use for stories, tales, narratives, flash fiction, scripts, or "write a story with the full pipeline". Expects a Phase-0/1 orchestrator packet from the `lofn` skill; if none exists, run `lofn` first. Do NOT use for images, music, video, or QA-only audits.
---

# Lofn Story — Codex-backed narrator pipeline

> **⚖️ AUTHORITY (2026-07-01):** the `.claude/skills/` twin of this skill is the CANONICAL policy source; this Codex mirror binds to it and to `.agents/skills/lofn/EXECUTION.md` §8 (Policy Deltas — golden-output quarantine, no-skip/NON-CANONICAL, itemized packet, per-pair variation angles, judge separation, the publish bar, gate mid-bands). On any disagreement, the `.claude` file wins.

Produces narrative pieces at Lofn competition grade — distinctive voice, world-building, and emotional architecture, not generic AI prose. Depth lives in `skills/story/`; Codex is the engine (hybrid execution per `.agents/skills/lofn/EXECUTION.md`).

## Before you start
1. Confirm a Phase-0/1 packet exists (`core_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `06_narrator_handoff.md` with the **ICB / Panel Ledger**, filled `CREATIVE_CONTEXT.md`). **No packet → run `lofn` first.**
2. Read `skills/story/OVERALL_PROMPT_TEMPLATE.md` for the CREATIVE CONTEXT carrier and the story phase routing.
3. Honor length intent from the brief — flash piece vs. full short story vs. script — and the human-subject ethics (`vault/HUMAN_SUBJECT_STANDARD.md`): draw themes from the world, invent the people.

## Execution (hybrid)
Coordinator **00-05 inline**, then **6 pairs as parallel subagents** for 06-10. At every agent start, inject the full startup packet from EXECUTION.md Section 3: CREATIVE_CONTEXT.md verbatim, the handoff/assignment files, current step contract, and prior artifact where applicable. Default: **6 concept x medium pairs**, each yielding distinct variations -> rank -> top picks. (Down-scale variations for long-form prose if the user wants one polished piece - confirm if unsure.)

### Coordinator steps (inline)
| Step | File | Artifact |
|------|------|----------|
| 00 | `skills/story/steps/00_Generate_Story_Aesthetics_And_Genres.md` | `step00_aesthetics_and_genres.md` |
| 01 | `skills/story/steps/01_Generate_Story_Essence_And_Facets.md` | `step01_essence_and_facets.md` |
| 02 | `skills/story/steps/02_Generate_Story_Concepts.md` | `step02_concepts.md` (12 concepts) |
| 03 | `skills/story/steps/03_Generate_Story_Artist_And_Critique.md` | `step03_artist_and_critique.md` |
| 04 | `skills/story/steps/04_Generate_Story_Medium.md` | `step04_medium.md` (form/voice/POV) |
| 05 | `skills/story/steps/05_Generate_Story_Refine_Medium.md` | `step05_refine_medium.md` → **6 pairs** |

### Per-pair steps (parallel subagents, one chain per pair)
| Step | File | Per-pair artifact |
|------|------|-------------------|
| 06 | `skills/story/steps/06_Generate_Story_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/story/steps/07_Generate_Story_Story_Guides.md` | `pair_{NN}_step07_story_guides.md` (beats) |
| 08 | `skills/story/steps/08_Generate_Story_Generation.md` | `pair_{NN}_step08_generation.md` (draft prose) |
| 09 | `skills/story/steps/09_Generate_Story_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/story/steps/10_Generate_Story_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |

## The story contract (hard gate)
Lead with seed/world, end with the checklist.
- **Guides direct; prose executes.** Step-07 guides are compact beats (≈35–50 lines: voice, POV, world rules, the turn, the ache). Final prose (08–10) earns its length (≈150–300 lines, scaled to the request) — don't pad the guide into a draft.
- **Distinct voice per pair** — the 6 pairs use 6 different narrative architectures (POV, tense, structure: epistolary, second-person, fragmented, nested-frame, oral-telling, prose-poetry continuity). Uniformity is a repair.
- **Show the world before the thesis** — body/place/sensory pressure establishes authority before any idea; one precise concrete detail unlocks universal resonance (the specificity paradox).
- **Personality fidelity** — prove which personality narrated it (diction, obsessions, signature move); the Hyper-Skeptics' Somatic Gate asks "could any competent writer have produced this, or is it unmistakably this voice?"
- **One bold choice** per piece. No generic AI cadence; no shallow cultural pastiche.

## QA & delivery
Run **lofn-qa** (structural + soul gate: voice distinctiveness, emotional architecture, world legibility, no thread loss). Rank by: voice strength → emotional landing → world coherence → originality → fit to the request. Save each selected piece per `skills/lofn-core/OUTPUT.md` (`type: story`); INDEX last.
