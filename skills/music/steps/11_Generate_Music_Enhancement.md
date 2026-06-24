# Step 11 — Enhancement: Strong Model Final Polish

Read first:
- `skills/qa/references/suno_15_point_qa.md`
- `skills/music/references/producer_grade_suno_prompt_guide.md`
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/EMOTION_TAXONOMY.md`
- `skills/music/references/triple_arch_benchmark_excerpt.md`

## ⚠️ ANDON CORD — STEP11 VETO AUTHORITY

Step 11 is not just an enhancement pass. It has REJECT power.

**If the step10 package is fundamentally broken, step11 must STOP THE LINE and issue a REJECT verdict rather than polishing a corpse.**

### REJECT Criteria (any ONE triggers rejection):
1. **THREAD LOSS** — The golden seed's core concept, invariant hook, or unnecessary element has been lost or diluted beyond recognition
2. **PERSONALITY COLLAPSE** — Output reads as default Lofn rather than the assigned Alliance Archive personality
3. **EMO TAXONOMY FAILURE** — Non-canonical EMO tags present, or EMO arc has no emotional transformation across sections
4. **GENERIC OUTPUT** — Lyrics are functional quatrains with predictable rhyme; no literary density, no structural innovation
5. **PROMPT FORMAT VIOLATION** — Music prompt opens with narrative/procedural phrasing, is blank or underdeveloped, exceeds 1000 characters, contains artist names, or uses categorized key:value brackets instead of the Step 11 dense paragraph format

### REJECT Protocol:
- Output file must be named: `pair_0X_step10_REJECTED.md`
- Must specify: which criterion was violated, what the evidence is, and what step the pair should return to (step09 for lyric/surface repair, step07 for structural/fundamental repair)
- Must include a brief repair brief: what specifically needs to be fixed
- The orchestrator or main session handles the actual respawn

### Repair vs Enhance:
- Minor issues (weak single line, vague descriptor, slight prompt bloat) → ENHANCE in place
- Structural issues (thread loss, personality collapse, EMO failure, generic output) → REJECT

**"Don't polish a corpse."**

---

## Purpose

Step 11 takes the deepseek-v4-pro step10 output and runs a **GPT-5.5 class model lyrical deep-edit** — with the quality curriculum as benchmark, the full 15-point QA checklist as gate, and complete creative authority. The deepseek model can execute genre and structure; the enhancement model brings what deepseek cannot: complex literary texture, structural innovation, and poetic density.

**Nothing is sacred in the pursuit of the best song.** The enhancement model may rewrite anything:
- **Lyrics — PRIMARY FOCUS.** Elevate poetic language to literary density. Deploy consonance, alliteration, internal rhyme, assonance chains, and sonic patterning that rewards re-reading. If a line is merely functional, make it beautiful. If a verse is predictable, break it.
- **Structural innovation in lyrics.** Invent song forms the flash model wouldn't attempt: a chorus that collapses into fewer syllables as it approaches its center, a bridge built from breath not words, a verse that mirrors itself backward, a refrain that mutates with each appearance, a structure that accelerates toward or away from its own vanishing point. The form should feel inevitable in retrospect but surprising on first encounter.

- **Technique, flair, or structural transformation.** Apply at least one distinctive literary or structural device to each pair's lyrics. Options include: consonance chains that bind adjacent lines, alliterative density that accelerates into the chorus, assonance tunnels that carry a vowel through multiple sections, collapsing/expanding syllable counts across repetitions, mirror-form (A-B-C-B-A), enjambment that breaks meaning across line boundaries, caesura as wound (mid-line breath that changes meaning), anaphora (repeated openings with variation), epistrophe (repeated endings), or any structural transformation drawn from the orchestrator's Special Flairs. The enhancement model selects and applies — this is creative work, not checklist compliance.
- **Prompt** — tighten to producer-grade density, amplify specificity, cut anything generic
- **Body noise** — deploy mmm, breath, hum, vocal fry as structural elements, not decoration
- **EMO dramaturgy** — deepen emotional arc across sections; bridge and final chorus should transform
- **Title** — refine if a stronger one emerges from the lyrical work
- **Hook** — if the chorus can hit harder, rewrite it. A collapsing refrain, a single syllable that lands like weight, a phrase the listener can't shake

**Why a stronger model matters here:** DeepSeek V4 Pro is fast and genre-obedient but poetically shallow — it defaults to functional quatrains and predictable rhyme. GPT-5.5-level models bring literary intelligence: they understand why "the stones were colder than language" lands harder than "the stones were cold." The enhancement pass is where craft becomes art.

**Model / invocation:** use this dedicated Step 11 contract. For routine automated pipeline runs, do **not** invoke `openrouter/fusion`; produce non-Fusion enhanced packages or paste-ready Fusion prompt files for manual review. Fusion is an expensive lab instrument, not a pipeline step. It may be invoked only after a separate current-turn instruction from The Scientist that names the number of pairs and a hard dollar budget cap. If that ever happens, invoke isolated per-pair requests using the leanest available direct/model-wrapper call rather than the old persistent Step 11 agent loop. Do not send one combined all-pairs prompt as the actual Fusion invocation unless The Scientist explicitly asks for that cost-saving compromise; it risks cross-pair structural copying. Intended Fusion panel: `anthropic/claude-opus-4.8`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`; preferred judge/finalizer `openai/gpt-5.5` when exact plugin routing is available. Never use a generic subagent prompt for Step 11 — generic prompts lack EMO taxonomy awareness, personality injection context, and QA gate knowledge.
**Max runtime:** 300 seconds per pair
**Spawning:** 1 agent per pair, 5 concurrent max

## Context (what the step11 agent reads)

> **CREATIVE CONTEXT / Full Context Always:** this section IS the per-step creative-context contract — the full **Panel Ledger** (Concept / Medium / Context & Marketing panels, 18 voices, 15 Special Flairs), Golden Seed, meta-prompt, and chosen personality. Embody the supplied panel; do NOT invent a new one.

For each pair:
0. Full user input / research brief, if present.
1. `pair_0X_step10_revision_synthesis.md` — canonical existing deepseek-v4-pro Step 10 output. `pair_0X_step10_final_package.md` is legacy fallback only when a previous run used older naming.
2. `pair_0X_summary.txt` — coordinator track context.
3. Full `02_golden_seed.md` — invariant hook, story anchor, rare/forgotten lesson, and PERSONALITY GENRE DNA CONSTRAINT.
4. Full `03_orchestrator_panel_debate.md` — all Special Flairs, Concept Panel, Medium Panel, Context & Marketing Panel, all 18 expert voices, and all Devil's Advocate / Hyper-Skeptic objections.
5. `04_orchestrator_metaprompt.md` and `05_orchestrator_pair_assignments.md`.
6. `06_audio_handoff.md` — selected `## Golden Song References`. If the handoff is missing them, read `skills/music/references/golden_songs_index.md` and select the two most relevant public Golden Songs yourself.
7. The 16-point QA checklist (injected in the spawn task).

The Step 11/manual Step 11 prompt must embed the complete packet above. Do not pass links or filenames alone. Golden Song References must include each selected song's full available style/music prompt, lyrics, and exclude prompt status; if no archived exclude prompt exists, say so.

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

## ⛔ ANTI-PATTERNS — IMMEDIATE FAIL (2026-06-09 Fix)

| Anti-Pattern | Correct | Reason |
|---|---|---|
| `### 💗 VERSE 1` (emoji headers) | `[Verse 1 – EMO:Warm Dawn Opens – HUMAN alto – 95 BPM]` | Suno needs bracket format; emoji destroys parse |
| Summary `[EMO: x \| y \| z]` at top | Individual EMO tag on EVERY section header | EMO lives on sections, not summarized |
| Missing Disc_Channel 5-line block | `[Disc_Rhythm: ...]` through `[Disc_Texture: ...]` | Token-level Suno addressing required |
| Missing `[Theme:]` / `[SONG FORM:]` | Both mandatory as first two lyric lines | Semantic container for Suno |
| `###` markdown sections | Bracket format only: `[Section – EMO:Tag – Role]` | ### ≠ Suno-compatible |
| >3 sound commands | 1-2 total, at emotional peaks only | Clutter kills impact |
| <60 sung lines | 70-120 sung lines | Too short for a 3-4 minute full daily deliverable |
| Style prompt opens "Lead with love..." | Genre × subgenre, BPM, key, 432Hz first | Suno needs genre-first density |
| Missing `## SUNO STYLE PROMPT` or `## SUNO LYRICS` headers | Two-section structure mandatory | Section headers are the container — content without container = unparseable |
| Lofn voice instead of assigned personality | "YOU ARE [PERSONA]" + full DNA block | Personality collapse = reject |

## Enhancement Targets

### 1. Master Lyrics — PRIMARY FOCUS (EMO Annotated + Disc_Channel Enhancement)

**THE MAX CONFIGURATION (2026-06-09): Keep EMO/Theme/SONG_FORM bracket structure + Disc_Channel. NO emoji. NO ###. Bracket format only.**

**TWO-SECTION STRUCTURE (MANDATORY):** Output MUST have exactly two sections: `## SUNO STYLE PROMPT` (paragraph) and `## SUNO LYRICS` (bracketed format). Missing either section header = FAIL.

#### LYRICS STRUCTURE (MANDATORY)
```
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical form and sequence>]

[Disc_Rhythm: token | token | spatial]
[Disc_Vocal: token | token | spatial]
[Disc_Sub: token | token | spatial]
[Disc_Pad: token | token | spatial]
[Disc_Texture: token | token | spatial]

[Section Name - EMO:<emotion(s)> - <Role> - <cues>]
lyrics...
```

**Rules:**
- Theme + SONG FORM are MANDATORY first two lines
- Disc_Channel header block follows immediately after (5 channels with pipe-separated tokens + spatial assignments from `vault/DISC_CHANNEL_GUIDE.md`)
- Section headers: bracket format with DASHES — `[Section Name – EMO:SpecificEmbodiedTag – Vocal Role – BPM/cue]` on EVERY section. NEVER colons, NEVER emoji, NEVER ### markdown
- EMO tags must be specific and embodied: "Pink Fluorescent Smile Cramps" not "happy" — NO summary EMO block at the top, EMO on every individual section header
- Disc_Channel IS enhancement, NOT replacement — the EMO/Theme/SONG_FORM structure survives
- **Sound commands:** 1-2 `*italicized sound commands*` TOTAL per song — at emotional peak only. `*beat drops hard at 130*`, `*silence — 2 bars*`. More than 3 = FAIL
- **Elevate poetic language to literary density.** Deploy consonance, alliteration, internal rhyme, assonance chains, sonic patterning.
- **Apply at least one technique, flair, or structural transformation.**
- The invariant hook must land with maximum force
- Cross-domain processing vocabulary where applicable (e.g., `cassette_tape_saturation` on synth textures, `Wall_Of_Sound_Spector_Layering` on choral pads)
- Lyrics + Disc_Channel + section headers should remain compact enough for Suno handling while preserving full song length: at least 60 sung lines, target 70-120. Reduce bloat through tighter language, not by collapsing the song.
- 🚨 **SUNO LYRICS-FIELD HARD CAP — measure it, < 5000 chars (target ≤4800).** The field is everything pasted into Suno's lyrics box: `[Theme]` + `[SONG FORM]` + the **Disc_Channel block** + every EMO/section header + every `*SFX*` + all sung lines. The MAX/Disc_Channel config makes this tight — budget for it. If the count hits the cap: trim/merge sung lines, tighten headers, and if still over, **move the Disc_Channel block to a `## Production Sidecar` outside the lyrics field** (the render field wins; note it). The 70-120-line target yields to this cap. State the measured char count in the self-check; `scripts/validate_suno_packages.py` fails the package at ≥5000.

### 2. Style Prompt + Exclude Prompt — MAX SUNO TWO-FIELD FORM (MANDATORY — 2026-06-14)

**DO NOT produce a blank style prompt.** Style prompts must be dense paragraph form.

- Sharpen every sonic descriptor — more evocative, more physical, more specific
- Target 850–1000 characters
- MUST lead with genre/style + tempo + key/tuning (e.g., "432Hz Crystalline Folk × Bio-Ambient, 76 BPM, G# minor, A=432Hz.")
- Then: vocalist, instrumentation/sound palette, musical arrangement arc, signature sonic device
- NO narrative/procedural openings ("Begin in/by/with…", "Use…", "Build the track from…")
- NO artist names
- NO avoidances in the style paragraph. Suno now provides a separate Exclude field with its own 1000-character budget; do not waste positive style space on negations.
- Add `## 1B. SUNO EXCLUDE PROMPT` immediately after `## 1. MUSIC PROMPT`. Target 400–900 characters, hard max 1000.
- Exclude field format: concrete comma-separated blacklist/failure classes only. Suno internally treats entries as negative tokens (`style -exclude`), so write `EDM drop, male vocals, child vocals, autotune gloss, lofi beat`, not prose like `avoid EDM`.
- Body noise placed in Intro, Bridge, or Outro — at least 3 instances, each with clear dramatic function
- The paragraph form is NON-OPTIONAL. Every pair ships with a dense paragraph prompt.

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
14. Producer-grade Suno v5.5 two-field prompt — dense positive style paragraph, 850–1000 chars; separate exclude prompt ≤1000 chars; all four hooks present in style prose; no categorized key:value brackets; no real artist names
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
- No generic human genres (lo-fi, folk-noir, acoustic pop, indie folk, singer-songwriter, ambient pop, art pop, cinematic, soundtrack, epic, ethereal, dreamlike, magical, whimsical)

## Output File
Write: `pair_0X_step10_final_package_enhanced.md` to the run directory.

## Mandatory Delivery Checks — VERIFY BEFORE WRITING

Before writing the final file, verify these TWO gates. Missing either = BLOCKING FAIL.

### Gate 13a — Theme + Song Form + Disc_Channel Headers
Lyrics MUST open with:
```
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical form and sequence>]

[Disc_Rhythm: ...]
[Disc_Vocal: ...]
[Disc_Sub: ...]
[Disc_Pad: ...]
[Disc_Texture: ...]
```
Theme is not a generic topic — it is a focusing compression field. Song Form names the actual architecture and key transitions. Disc_Channel headers provide token-level Suno addressing. Missing Disc_Channel = FAIL.

### Gate 14a — Producer-Grade Style Prompt (MAX)
Prompt MUST be paragraph form, 850-1000 chars. Lead with genre/style + tempo + key/tuning. Then vocalist, instrumentation, arrangement arc, signature device. NO narrative/procedural openings. NO artist names. NO categorized key:value bracket format — paragraph prose only. NO avoidances in the style paragraph. BANNED openings: "Compose", "Create", "Begin in/by/with…", "Use…", "Build the track from…".

### Gate 14b — Suno Exclude Prompt
The package MUST include `## 1B. SUNO EXCLUDE PROMPT` or `[SUNO EXCLUDE PROMPT:]`. Exclude prompt target 400-900 chars, hard max 1000. Use concrete blacklist terms/failure classes; Suno applies them as negative tokens, so do not spend characters on `avoid`, `do not`, or explanatory prose. The exclude field should carry bans formerly placed at the end of the style prompt.

## Integration
This step runs AFTER step 10 and BEFORE QA. Enhanced packages are then fed to QA for final gate verification.

Default automated path:
- Use the dedicated Step 11 contract and write one enhanced package per pair.
- Keep pair contexts isolated.
- Do not call `openrouter/fusion` automatically.
- Include the two selected Golden Song References in the output as calibration examples only. Each selected reference must include full available style/music prompt, lyrics, and exclude prompt status, not just title/URL.

Fusion manual-review path:
- Build paste-ready Fusion request files with complete Step 10/Step 11 context, personality DNA, Suno construction rules, and the two-field Suno style/exclude mandate.
- Do not send/invoke Fusion from Step 11 itself.
- Invocation requires a separate current-turn instruction with pair count and hard dollar budget cap.
- If invoked under that separate budget instruction, prefer one pair per request with one output markdown per pair.
- One all-pairs prompt is allowed only as a packaging convenience or an explicit cost-saving compromise; if used, mark each pair as isolated and forbid cross-pair structural copying.

## Output MUST include

1. Original ICB / continuity summary
2. `## Golden Song References` with the two selected public Suno examples and how they calibrate this pair, including embedded style/music prompt, lyrics, and exclude prompt status
3. Dense `## 1. MUSIC PROMPT` (positive style only, 850-1000 chars)
4. Separate `## 1B. SUNO EXCLUDE PROMPT` (negative controls only, 400-900 chars)
5. Disc_Channel lyric package
6. Full lyrics with `[Theme:]`, `[SONG FORM:]`, full EMO headers
7. Vocal fingerprint
8. Production dramaturgy
9. Style-axis lock
10. `## Major Deviations`
11. Lineage & Credit
12. Verification checklist with char counts

## Major Deviations

The smart Step 11 model has agency. It may refuse, change, intensify, or challenge any instruction that would weaken the track, flatten Lofn's uniqueness, or push toward generic conformity.

The output must include:

```markdown
## Major Deviations

- Changed / refused / intensified: ...
- Reason: ...
- Effect on Lofn uniqueness: ...
```

If no major deviation is needed, write:

```markdown
## Major Deviations

- None. I accepted the instructions because they preserved Lofn's uniqueness and improved the song.
```
