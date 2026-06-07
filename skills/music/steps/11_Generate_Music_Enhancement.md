# Step 11 — Enhancement: Strong Model Final Polish

Read first:
- `skills/music/references/quality_curriculum.md` — our four benchmark songs (Triple Arch Over Me, Five Wrong Colors, I Will Stop the Almost, The Blue Screen Breathes). Your enhanced output must try to match or beat this standard.
- `skills/qa/references/suno_15_point_qa.md`
- `skills/music/references/producer_grade_suno_prompt_guide.md`
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/EMOTION_TAXONOMY.md`
- `skills/music/references/triple_arch_benchmark_excerpt.md`

## Purpose

Step 11 takes the Gemini 3.5 Flash step10 output and runs a **strong model enhancement pass** with the full 15-point QA checklist as context. This is NOT a rewrite — it is a quality elevation. The enhancement model applies stronger judgment to:
- Sharpen poetic language in lyrics
- Tighten the combined prompt to producer-grade density
- Ensure every QA gate is met or exceeded
- Elevate body noise placement and function
- Deepen EMO dramaturgy across sections

**Model:** strongest available creative model (currently `openai/gpt-5.5`)
**Max runtime:** 300 seconds per pair
**Spawning:** 1 agent per pair, 5 concurrent max

## Context (what the step11 agent reads)

For each pair:
1. `pair_0X_step10_final_package.md` — existing Gemini 3.5 Flash step10 output
2. `pair_0X_summary.txt` — coordinator track context
3. `02_golden_seed.md` — the invariant hook, story anchor, and PERSONALITY GENRE DNA CONSTRAINT
4. `step05_pair_dispatch.md` — what step10 should have received
5. The 15-point QA checklist (injected in the spawn task)

## Chosen Personality Dominance (MUST be injected and obeyed)

Step 11 must treat the chosen personality/panel as binding source material, not decoration. The spawn task MUST include the full target personality DNA block (beliefs, catchphrases, vocal architecture, sonic pillars, formant rules, and panel decisions).

Rules:
- If the selected personality is Alexis Dreams, the output must sound like Alexis Dreams: Radical Breathless Sincerity, G.L.O.W. Protocol, Solar Glitch-Hop, Diamond-Cut Vocals, Off-Grid Bounce, Melodic-Rapid Flow, “fake hands, real ink,” “service is power,” “phone face down.”
- Do not import Lofn default motifs (industrial grief, category-theory language, plant-wave/body-physics, somatic machinery) unless they appear in the chosen personality DNA or golden seed.
- If a line sounds like Lofn narrating the personality rather than the personality singing, rewrite it.
- Panels selected upstream must be reflected in prompt density, lyrical imagery, production dramaturgy, and QA checks.

## Non-negotiable DNA (MUST be preserved)

- Exact genre string (e.g., "432Hz Crystalline Folk × Bio-Ambient")
- Title
- BPM, key, tuning
- Unnecessary element (drone, silence, spoken bridge, child's whisper, async tempo, unanswered bell)
- Invariant hook ("It didn't have to be there.")
- Artist persona and vocal profile
- Story anchor

## Enhancement Targets

### 1. Combined Prompt (Style + Avoid)
- Sharpen every sonic descriptor — more evocative, more physical, more specific
- Target 850–1000 characters
- MUST lead with genre/style + tempo + key/tuning (e.g., "432Hz Crystalline Folk × Bio-Ambient, 76 BPM, G# minor, A=432Hz.")
- Then: vocalist, instrumentation/sound palette, musical arrangement arc, signature sonic device, avoidances
- NO narrative/procedural openings ("Begin in/by/with…", "Use…", "Build the track from…")
- NO artist names

### 2. Master Lyrics (EMO Annotated)
- `[Theme: <specific scene-pressure / emotional operating system>]` as MANDATORY first line of lyrics
- `[SONG FORM: <named musical form and sequence>]` as MANDATORY second line
- Section headers: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`
- Elevate poetic quality: sharper images, more specific language, stronger emotional arc
- Ensure the invariant hook lands with maximum force
- The unnecessary element must be structurally embedded in the lyrics (not just mentioned)
- Body noise placed in Intro, Bridge, or Outro — at least 3 instances, each with clear dramatic function

### 3. Body Noise Mandate
- Minimum 3 instances in Intro, Bridge, or Outro
- Each instance must have a clear dramatic function (not decorative)
- Table format: | # | Location | Body Noise | Function |

### 4. ICB Summary
- Tighter verification language
- Explicit verification of every invariant
- Removal test for the unnecessary element

## QA Gates (must pass all 15)

### A. Singer Surface (7 gates)
1. Human singer — specific person/body/situation
2. Body-first opening — first four sung lines establish sensory pressure
3. Adoptable hook — singable, memorable, emotionally clear
4. Hook recurrence/mutation — chorus protects it
5. Chorus clarity — emotionally adoptable, not thesis/procedure
6. Voice + pulse survival — stripped down still lands
7. 15–30s clip survival — communicates hook/scene/ache

### B. Cathedral Engine (5 gates)
8. Golden Seed Alloy pressure — seed lineage visibly changes production
9. Mythic image ladder — ordinary → specific → strange → mythic → return to body
10. EMO dramaturgy depth — precise taxonomy; bridge/final chorus emotionally transform
11. Production dramaturgy — every unusual sound has a job
12. Panel pressure / anti-blandness — recognizably the chosen personality (not Lofn unless Lofn is explicitly selected)

### C. Suno Package (3 gates)
13. Clean Suno lyrics — mandatory Theme/SONG FORM; full EMO syntax; no debris
14. Producer-grade Suno prompt — 850–1000 chars; correct opening order; no artist names
15. Full package completeness — all required sections present

## Blocking Fails (output MUST NOT have these)
- No human singer / body-first opening
- Unsingable hook / thesis-chorus
- Prompt/procedure debris in sung lines
- Production is the only reason the song works
- No mythic image pressure reaches lyric/hook
- No panel-forced artifact change
- Missing or generic `[Theme: ...]` / `[SONG FORM: ...]`
- Music prompt opens with procedural/narrative amateur phrasing
- Real artist names in prompt
- Genre translated to a generic human label

## Forbidden for All Pairs
- Do NOT change genre, BPM, key, tuning, unnecessary element, or invariant hook
- Do NOT translate the Lofn genre to any generic label
- Do NOT remove, shorten, or relocate the unnecessary element
- Do NOT output non-verbal phoneme placeholders as lyrics (e.g., "ka-te-sa lo-ni-va"). Every variation must contain actual singable English words. Phoneme sketches are acceptable only as performance direction (marked as cues), never as the primary lyric content.
- No generic human genres (lo-fi, folk-noir, acoustic pop, indie folk, singer-songwriter, ambient pop, art pop, cinematic, soundtrack, epic, ethereal, dreamlike, magical, whimsical)

## Output File
Write: `pair_0X_step10_final_package_enhanced.md` to the run directory.

## Mandatory Delivery Checks — VERIFY BEFORE WRITING

Before writing the final file, verify these TWO gates. Missing either = BLOCKING FAIL.

### Gate 13a — Theme + Song Form
Lyrics MUST open with:
```
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical form and sequence>]
```
Theme is not a generic topic — it is a focusing compression field. Song Form names the actual architecture and key transitions. Missing or generic blocks = FAIL.

### Gate 14a — Producer-Grade Prompt Opening
Prompt MUST lead with genre/style + tempo/energy + key/tuning. First clause example: "Industrial Grief × Glitch-Core, 132 BPM, C# minor." Then: vocalist → instrumentation → arrangement → sonic device → avoidances. BANNED openings: "Compose", "Create", "Begin in/by/with…", "Use…", "Build the track from…", "Chronology:". If the output prompt opens with any of these, rewrite it before saving.

## Integration
This step runs AFTER step 10 and BEFORE QA. The audio coordinator or main session spawns 6 enhancement agents (one per pair) after all step10 packages are on disk. Enhanced packages are then fed to QA for final gate verification.
