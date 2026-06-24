---
name: lofn-music
description: Run the Lofn music/audio pipeline (steps 00–11) backed by Claude — Suno-ready song packages with two-field style/exclude prompts, EMO-tagged lyrics, song guides. Use for songs, tracks, beats, lyrics, music production briefs, or "write a song with the full pipeline". Expects a Phase-0/1 orchestrator packet from the `lofn` skill; if none exists, run `lofn` first. Do NOT use for static images, video, story prose, or QA-only audits.
---

# Lofn Music — Claude-backed audio pipeline

Produces Suno/Udio-ready song packages at Lofn competition grade. The creative depth lives in the legacy step files and references under `skills/music/`; this skill runs them with Claude as the engine (hybrid execution per `.claude/skills/lofn/EXECUTION.md`).

## Before you start
1. Confirm a Phase-0/1 packet exists for this run (`core_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `06_audio_handoff.md` with the **ICB / Panel Ledger**, and the filled `CREATIVE_CONTEXT.md`). **No packet → run the `lofn` skill first.** A real 3-panel orchestrator object is a launch prerequisite; do not self-author a shallow one.
2. Read for depth (just-in-time, not all upfront):
   - `skills/music/references/music_full_legacy.md` — the full tuned pipeline (authoritative)
   - `skills/music/references/producer_grade_suno_prompt_guide.md` + `vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md` — prompt construction
   - `skills/music/references/golden_songs_index.md` — pick/confirm the 2 Golden Songs in the handoff
   - `skills/music/references/suno_format_example_{triple_arch,blue_screen,five_wrong_colors}.md` — calibration targets
   - `skills/lofn-core/refs/EMOTION_TAXONOMY.md` — the only valid source for EMO emotions

## Execution (hybrid)
Coordinator **00–05 inline**, then **6 pairs fan out as parallel Claude subagents** for 06–10, then **step 11 enhancement** (1 subagent/pair). Inject the full `CREATIVE_CONTEXT.md` into every step and every subagent (`EXECUTION.md` §3). Default cardinality: **6 pairs × 4 variations = 24 songs.**

### Coordinator steps (inline — you)
Run each as its own pass with its own saved canonical artifact (do NOT collapse 00–05 into one):

| Step | File | Output artifact |
|------|------|-----------------|
| 00 | `skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md` | `step00_aesthetics_and_genres.md` (50×4 JSON) |
| 01 | `skills/music/steps/01_Generate_Music_Essence_And_Facets.md` | `step01_essence_and_facets.md` |
| 02 | `skills/music/steps/02_Generate_Music_Concepts.md` | `step02_concepts.md` (12 concepts) |
| 03 | `skills/music/steps/03_Generate_Music_Artist_And_Critique.md` | `step03_artist_and_critique.md` |
| 04 | `skills/music/steps/04_Generate_Music_Medium.md` | `step04_medium.md` |
| 05 | `skills/music/steps/05_Generate_Music_Refine_Medium.md` | `step05_refine_medium.md` → **6 pairs chosen** |

### Per-pair steps (parallel subagents — one chain per pair)
Give each of the 6 pair-subagents its full ICB + pair assignment + these step contracts, and have it produce all five canonical files for its pair:

| Step | File | Per-pair artifact |
|------|------|-------------------|
| 06 | `skills/music/steps/06_Generate_Music_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/music/steps/07_Generate_Music_Song_Guides.md` | `pair_{NN}_step07_song_guides.md` |
| 08 | `skills/music/steps/08_Generate_Music_Generation.md` | `pair_{NN}_step08_generation.md` |
| 09 | `skills/music/steps/09_Generate_Music_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |

### Step 11 — Enhancement (1 subagent per pair, you as the GPT-5.5-class polish tier)
Per `skills/music/steps/11_Generate_Music_Enhancement.md`. Reads the pair's step-10 + ICB + the 2 Golden Songs + the QA checklist. Produces `pair_{NN}_step10_final_package_enhanced.md` in **MAX configuration**: dense paragraph style prompt + separate exclude prompt + Disc_Channel header block (5 channels, from `vault/DISC_CHANNEL_GUIDE.md`) + Theme + SONG FORM + full EMO headers + a `## Major Deviations` section (where you exercise agency — name anything you refused/changed/intensified and defend it). **Do not invoke `openrouter/fusion`** — that path is manual-review only.

> **Andon Cord:** if a step-10 package is fundamentally broken (thread loss, personality collapse, EMO failure, generic output, format violation), step 11 REJECTS and routes back to step 09/07 with a repair brief. Don't polish a corpse.

### Step 12 — Panel-of-panels audit (when triggered)
Convene a producer panel (Eno, Herndon, Flying Lotus, SOPHIE, Reynolds, Albini — "after" constructs) to audit all 6 final prompts against the Golden Songs; output `STEP12_MUSIC_PROMPT_AUDIT.md`.

## The Suno output contract (hard gate — non-waivable)
Order the creative prompt **seed → permission → songmaking → QA contract last.** Never lead a pair-subagent with the char/line checklist.

- **`## 1. MUSIC PROMPT`** — ONE standalone, copy-paste Suno style prompt per song. Dense **prose paragraph, 850–1000 chars**, no categorized `key:value` brackets, no real artist names (ghost-homage lives in lyrics only). Mandatory order: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation/palette with physical adjectives → arrangement arc → bold sonic device. Must include a vivid **opening moment** (first 5s), **spatial language** (left/right/center/depth), a **kinetic defect** (asymmetric groove), and explicit acoustic / no-acoustic declarations. Banned openers: "Begin in/with…", "Use…", "Build the track from…", "Chronology:", "For an adult human singer…".
- **`## EXCLUDE PROMPT`** — separate negative field, 400–900 chars (hard max 1000). Concrete blacklist terms/failure classes, not avoidance prose in the style field (Suno applies it as negative tokens).
- **Lyrics** — open with `[Theme: <scene-pressure / emotional OS>]` then `[SONG FORM: <named form & sequence>]`; **full EMO headers** `[Section - EMO:<emotion(s)> - <Role> - <cues>]` (emotion from the taxonomy, never bare AWE/INDIGNATION); ≥1 standalone `*SFX*` cue; clean sung lines (no prompt/QA debris); **70–120 sung lines**.
- 🚨 **SUNO LYRICS-FIELD HARD CAP — the lyrics field MUST be < 5000 characters (target ≤ 4800).** This is a Suno render limit, not a style note: the field holds *everything you paste into Suno's lyrics box* — `[Theme]` + `[SONG FORM]` + the Disc_Channel block (if present) + every section/EMO header + every `*SFX*` cue + all sung lines. **Measure the exact character count; never estimate.** A song over 5000 will not render. If over: cut/merge sung lines first, then tighten section headers, then move the Disc_Channel block and any production metadata to a `## Production Sidecar` *outside* the lyrics field. The 70–120-line target yields to this cap — a renderable 64-line song beats an unrenderable 110-line one. State the measured count in the package self-check. (This tension is real with MAX/Disc_Channel config; budget for it from the first draft.)
- **Verse diversity:** the 6 songs use 6 DIFFERENT verse architectures + rhyme schemes + poetic techniques (assigned per pair in Phase 1). Uniformity is a repair.
- **Default standards:** female vocals (unless overridden); prove the active personality (sonic-world sentence + signature device + seed-derived weirdness). Living-scene genres (Amapiano, Baile Phonk, UK drill, raga, etc.) require a Lineage & Credit block.

## QA & delivery
Run **lofn-qa** → the 16-point Suno gate (`skills/qa/references/suno_15_point_qa.md`): 7 Singer-Surface + 5 Cathedral-Engine + 3 Suno-Package + Lineage. The 3 Hyper-Skeptics vote as the **Somatic Gate** ("could any competent prompt generate this, or is it unmistakably Lofn?") — 2 of 3 NO = REPAIR. Save each selected song per `skills/lofn-core/OUTPUT.md`; write the INDEX last.

**Provider note:** these skills emit text. If the user later renders audio, Suno.com is the prompt destination; for API audio use Google Lyria — **FAL minimax-music is banned** (quality). Never call render tools from this skill.
