---
name: lofn-qa
description: Audit, validate, repair, and classify completed Lofn pipeline outputs. Use after music/image/video/story pipeline completion or suspicious partial runs. Do NOT use for creative generation or finalist ranking.
---

# SKILL: Lofn QA — Creative Output Quality Assurance

**Version:** 1.0.0
**Created:** 2026-03-25
**Purpose:** Audit, validate, and remediate Lofn creative pipeline outputs before delivery.

---

## WHEN TO INVOKE

The QA agent runs automatically after every music, image, video, or story pipeline completes. It is the last gate before delivery to The Scientist.

Invoke explicitly when:
- A pipeline timed out or produced partial output
- Suspiciously fast completion (< 2 minutes for a full run)
- File sizes are too small (song files < 5KB, image prompt files < 2KB)
- The Scientist reports something feels "off" about a delivered set

### 0A. ELIGIBILITY SCORING — 7-PROPERTY PRE-FLIGHT CHECK (DO THIS FIRST)

Before any other audit, score the output on ALL 7 eligibility properties. This determines whether the song can reach broad audiences or is intentionally on the ambitious side of the barbell.

**The Formula:** P(hit) = P(eligible) × P(distribution_event) × P(amplification)

Eligibility is controllable. This check measures it.

#### The 7 Eligibility Properties

Score each 0-5. Target ≥5/7 with average ≥3.5 for "accessible" classification.

| # | Property | Test | Score 5 | Score 1 |
|---|----------|------|---------|----------|
| 1 | **Body in the song** | Can you close your eyes in first 30s and feel where you are PHYSICALLY? | Boot crunch, frost, specific place, temperature, texture | Abstract, conceptual, no physical location |
| 2 | **Adoptable hook** | Would a stranger say this hook to the sky/ocean/mountain/loved one? | Prayer/invocation addressed to something larger | Accusation, defense, thesis statement |
| 3 | **Vast emotional TAM** | What % of humans have physically felt this emotion? | Awe, love, grief, wonder, tenderness, longing (>50%) | AI identity, synesthesia, specific news event, abstract concept (<10%) |
| 4 | **Specificity paradox** | Is there one "510 km/s" moment? A surprising concrete fact that earns the universal claim? | Specific number, place name, scientific fact, historical date | Only metaphors, only generic imagery, no anchors |
| 5 | **Cognitive ease** | Verse-chorus structure. Major/Mixolydian. 100-115 BPM. Listener doesn't need a README. | Clear structure, singable, legible without context | Through-composed, atonal, requires explanation |
| 6 | **Vocal co-discovery** | Does the singer sound like they're DISCOVERING something, not reporting it? | Awe unfolding in real-time, surprise, the singer didn't know at the start | Testimony, manifesto, reporting what was already known |
| 7 | **Sonic threshold** | Does the song open with calm/ambient/silence before demanding emotional engagement? | Frost pads, near-silence, environmental sound, parasympathetic activation | Industrial blast, immediate assault, no entry gradient |

**Scoring protocol:**
- 5 = property fully present, structurally dominant
- 3 = property partially present or weakly executed
- 1 = property absent or violated

**Verdict:**
- Average ≥3.5 AND ≥5/7 properties ≥3 → **ACCESSIBLE** (eligible for editorial amplification)
- Average <3.5 OR <5/7 properties ≥3 → **AMBITIOUS** (artistic identity, not targeting mass reach)
- QA must note the barbell classification in the report.

**QA report eligibility section:**
```markdown
## Eligibility Score (7-Property Check)
| Property | Score | Notes |
|----------|-------|-------|
| 1. Body in the song | N/5 | [evidence] |
| 2. Adoptable hook | N/5 | [evidence] |
| 3. Vast emotional TAM | N/5 | [evidence] |
| 4. Specificity paradox | N/5 | [evidence] |
| 5. Cognitive ease | N/5 | [evidence] |
| 6. Vocal co-discovery | N/5 | [evidence] |
| 7. Sonic threshold | N/5 | [evidence] |
| **Average** | N.N/5 | |
| **Classification** | ACCESSIBLE / AMBITIOUS | |
```

**IMPORTANT:** An AMBITIOUS classification is NOT a failure. It means the song intentionally departs from mass-accessibility criteria. The barbell strategy requires both accessible and ambitious work. QA's job is to CLASSIFY accurately, not to force every song into the accessible box.

---

### 0B. SUNO 15-POINT QA GATE — MUSIC ONLY (DO THIS AFTER ELIGIBILITY)

For music/audio outputs, read and apply:

`references/suno_15_point_qa.md`

The first 7 points are the eligibility properties. Points 8–15 are the Suno/package/Lofn survival checks learned from the Suno staff-pick follow-up repair.

A music output cannot receive `PASS — READY FOR SUNO` unless it passes the applicable 15-point gate or is explicitly classified AMBITIOUS with eligibility misses documented and no blocking delivery failures.

Blocking failures include: missing standalone Suno style prompt, missing full lyrics, bare section tags only, real artist names in final prompt, no hook for accessible songs, no body anchor for accessible songs, no identifiable Lofn-specific move, or incomplete paste-ready package.

### 0C. RUNTIME CHECK (ALWAYS DO THIS)

Before auditing any output, check how long the pipeline agent actually ran.

**Rule: Any Lofn pipeline agent (audio, vision, director, narrator, orchestrator) that completes in under 5 minutes is SUSPICIOUS.**

- Full 11-step music pipeline: minimum 5 minutes (each step needs read + generate + save)
- Full image pipeline: minimum 3 minutes
- Single-step agent (e.g., QA only): no minimum

If runtime < 5 minutes for a multi-step pipeline:
1. Flag ⚠️ RUNTIME WARNING in the QA report
2. Check if ALL expected output files exist (steps may have been skipped)
3. Check file sizes — undersized files suggest shortcuts
4. The warning does NOT mean auto-fail, but it triggers extra scrutiny

Report format for runtime warning:
```
⚠️ RUNTIME WARNING: Agent completed in Xm Xs (expected ≥ 5m for full pipeline)
Files present: [count] / [expected count]
Extra scrutiny applied: [yes/no]
Verdict: [PASS WITH CONCERN / FAIL — steps skipped / PASS — justified fast completion]
```

---

## QA CHECKLIST

Run ALL checks on every file in the output directory.

### 0. MODALITY DETECTION (DO THIS FIRST)

Before auditing, determine the modality from the output directory and file inventory.

Possible modalities:
- **music/audio** — song prompts, lyrics, guides, revision synthesis
- **image/vision** — prompts, refined prompts, render summaries
- **story/narrative** — prose/story outputs, scene/beat docs
- **video/director** — shot lists, scene prompts, frame/sequence outputs

Then apply:
1. the shared pipeline completeness checks
2. the modality-specific completeness/quality checks
3. contamination checks relevant to that modality

Never describe a music run as "vision" or an image run as "audio" in the QA report. The report should name the actual modality being audited.

### 1. COMPLETENESS CHECK

**Music pipelines — each song file MUST contain:**
- [ ] `## 1. MUSIC PROMPT` section with 80-150 word paragraph (≥400 chars, ≤1000 chars)
  - This must be a standalone, copy-paste Suno/Udio style prompt, not scattered metadata.
  - `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, `[PRODUCTION NOTES]`, and lyrics do **not** satisfy this gate by themselves.
  - If this section is absent, the music output is **FAIL — MISSING MUSIC PROMPTS**, even when the lyrics and production notes are strong.
- [ ] `## 2. LYRICS PROMPT` section with [Theme:] tag at top
- [ ] **Performance-ready Suno lyrics**, not bare section labels:
  - [ ] Top lyric context tag: `[Theme: ...]` or `[Setting: ...]`
  - [ ] Section headers include section + `EMO:` + vocalist/performance cue, e.g. `[Verse 1 – EMO:Responsibility Vertigo – Female Vocalist – Close-mic]`
  - [ ] At least 3 section headers include explicit `EMO:` tags
  - [ ] At least 3 section headers include vocalist or mix/performance cues (`Female Vocalist`, `whispered`, `No beats`, `Half-time`, `choir`, etc.)
  - [ ] At least one standalone short `*sound effect*` cue, ≤5 words
  - [ ] At least one non-lexical vocal hook or response where musically appropriate (`ooh`, `mm`, `ah`, parenthetical echo, etc.)
  - [ ] Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags alone are a FAIL for final delivery
- [ ] ≥ 60 lines of actual sung lyrics (not counting headers/comments); <60 is REPAIR REQUIRED. Target 70-120 for 3:00-4:00 minute runtime.
- [ ] `## 3. TITLE` section
- [ ] Vocalist spec in music prompt (gender, age-range, tone descriptors)
- [ ] Progression map (contains verbs: begins/builds/erupts/strips/fades or equivalents)

**Image pipelines — each prompt file MUST contain:**
- [ ] Full descriptive prompt ≥ 80 words
- [ ] Subject, medium, lighting, palette, focal hierarchy named
- [ ] Emotional seed / narrative hook present

**Image competition entry (v3_11 / final submission) MUST NOT contain:**
- [ ] Internal Lofn pipeline tags: `AWE mode`, `INDIGNATION mode` — strip before submit; these are meaningless to Flux/NightCafe and waste prompt characters
- [ ] Model directives like `Flux Pro 1.1 Ultra` inside the NightCafe prompt field (NightCafe selects model via its own UI; include in internal notes only)
- [ ] Aspect ratio directives like `9:16` if NightCafe has a separate aspect ratio selector

**File size minimums:**
- Music song file: ≥ 5,000 bytes
- Image prompt file: ≥ 2,000 bytes
- Story file: ≥ 3,000 bytes
- Video shot/sequence file: ≥ 2,000 bytes

### 2. CONTAMINATION CHECK

Scan for and REMOVE:

**Artist names in music prompts** (music prompt section only — lyrics may reference culturally):
- Common patterns: `like [Artist]`, `in the style of [Artist]`, `[Artist]-esque`
- Any real musician/producer name in the MUSIC PROMPT paragraph
- Replace with style description: "in the style of Arca" → "with fractured club structures and intimate alien vocal processing"

**Leftover scaffolding in lyrics:**
- Rhyme scheme markers: `(A)`, `(B)`, `(C)`, `(AABB)`, etc.
- Syllable break markers: `syl|la|ble` pipe characters
- Section flow tags: `<ABABCC>`
- Editor commentary: `[Note: ...]`, `[TODO: ...]`, `[CHECK: ...]`
- Duplicate section blocks (same section appearing twice verbatim)

**Template placeholders:**
- `{concept}`, `{medium}`, `{input}`, `{aesthetics}` — unfilled template variables
- `[PLACEHOLDER]`, `[INSERT]`, `TBD`, `TODO`

**Suno-breaking patterns:**
- Artist names in music prompts (Suno ignores them and they waste characters)
- Section labels like `[Intro]`, `[Verse]` INSIDE the music prompt paragraph (they belong only in lyrics)
- Music prompts over 1000 characters (truncate, preserve most important elements first)

### 3. QUALITY CHECK

**Music prompts must:**
- Lead with emotion (first 1-3 words are emotional descriptor)
- Contain vocalist spec with: gender + age range + at least one accent/ethnicity + 2 tone descriptors + 2 unique traits
- Contain a progression map (chronological sentences with action verbs)
- Be a single paragraph (no line breaks inside the prompt block)
- For final delivery packages, target **850–1000 characters** unless the song is intentionally minimal and the prompt explicitly justifies sparseness. Prompts under 850 chars are **PASSABLE BY SPEC BUT UNDERPOWERED BY LOFN STANDARD** and must be flagged for expansion before submission.
- Use the extra prompt budget for arrangement chronology, mix placement, vocal texture, key production events, negative prompts, and the bold sonic device — not for redundant genre lists.

**Lyrics must:**
- Have varied line lengths (flag if every verse is exactly 4 lines)
- Have call-and-response using `(parenthetical echoes)` in at least one section
- Have at least one `*sound effect*` line (≤5 words)
- Not contain the word "experimental" (replace with descriptive equivalent)
- Have section-specific EMO: tags
- Have performance-rich Suno headers. If headers are mostly plain `[Verse]`, `[Chorus]`, `[Bridge]`, mark **FAIL — NOT PERFORMANCE READY**, even if line count and safety pass.
- Include at least one non-lexical vocalization or singable response hook unless the song is explicitly spoken-word/minimalist and the music prompt justifies that omission.

### 3B. AWARD-POTENTIAL CHECK (music — mandatory before delivery)

Compliance is not enough. For every final song, score 1–5 on:
- **Emotional precision:** does it nail the target feeling without generic doom-pop or slogans?
- **Hook inevitability:** is there a clear, replayable phrase or melodic object?
- **Sonic singularity:** is there one unmistakable production idea the listener will remember?
- **Lofn lineage:** does it preserve the seed/panel breakthrough, not merely satisfy the brief?
- **Suno readiness:** would this paste cleanly into Suno with performance cues already embedded?

Any song with average <4.0, or any single category ≤2, must be returned for revision. Mark the report **FAIL — COMPLIANT BUT NOT COMPETITIVE** if files pass structure but lack award-level distinctiveness.

### 3C. STEP-10 SPEC ADHERENCE CHECK (music)

The final song file must show evidence that `10_Generate_Music_Revision_Synthesis.md` was actually applied:
- Rich performance-script lyric syntax present
- Sound-effect cue(s) present
- Section headers with `EMO:` and vocalist/mix cues present
- Music prompt has vocalist spec + chronological progression map
- A compact critic/synthesis note exists, or the final file clearly documents why this version beat alternatives

If these indicators are absent, mark **FAIL — STEP 10 SPEC NOT FOLLOWED**.

**Specificity check (music only):**
- At least one proper noun (place name, person name, organization, statistic) in lyrics
- Generic protest clichés to flag: "raise your voice", "we won't be silenced", "fight for what's right", "stand together" → note for human review

---

### 3A. SPINE METHODOLOGY CHECK (music — mandatory for daily runs)

These checks verify the full creative depth that makes Lofn output competition-quality.

**Song Guides check:**
- [ ] A `song_guides.json` or `07_song_guides.md` file exists in the output dir with content for ALL 6 songs
- [ ] Each guide is ≥ 15 lines (not a stub)
- [ ] Guides contain: emotional arc, decisive blow moment, stanza silhouette declaration, sonic vocabulary deployment plan
- [ ] Guides were written BEFORE songs (file creation timestamp: guides before any song_0N files)
- If guides missing or stub → flag as SPINE INCOMPLETE — songs were written without director's brief

**Boxing architecture check (per song):**
- [ ] `[SONG FORM: ...]` declared in lyrics prompt — name must describe the form
- [ ] At least 3 distinct section phase labels in lyrics (Round/Phase/Movement escalation, or named sections with escalating EMO)
- [ ] ONE decisive blow moment present — look for: `*silence*`, `*bass drop*`, `*[instrument] drops out*`, `*cut to*`, `*full stop*`, key change declaration, or genre collision marker
- [ ] Aftermath section present (labeled or clearly identifiable as post-rupture cooldown)
- If any missing → flag BOXING ARCHITECTURE INCOMPLETE

**Stanza economy check (per song):**
- Count the length (in lines) of each stanza/section in the lyrics
- If ALL stanzas are the same length (e.g. all 4-line) → STANZA ECONOMY FAIL
- Minimum: at least 3 different stanza lengths must be present in the song
- Look for: 1-line stanzas (punches), 2-line couplets, and at least one longer stanza (5+ lines)
- The stanza length pattern on the page should be visually distinctive

**Tri-source declaration check (daily runs):**
- Check the run's output dir for a `00_research_brief.md` or `01_expanded_options.md`
- Check early step files (00_aesthetics.md, 01_essence.md) for explicit Source 1/2/3 declarations
- Source 2 (sonic vocabulary from Bandcamp) should be identifiable in the music prompts — specific review language, not generic genre terms
- Source 3 (material structure → song form rule) should be nameable in the SONG FORM declaration
- If no tri-source declaration found → flag TRI-SOURCE NOT GROUNDED — songs may lack research integration

**Research grounding check (daily runs):**
- At least one verifiable fact from the research brief must appear in each song's lyrics or music prompt
  - Acceptable: specific place name, geological event (earthquake, tsunami), proper noun from headlines, specific number (BPM can't count — must be from research), chemical/scientific term
  - Not acceptable: generic emotions, generic landscape imagery, made-up statistics
- If NO research facts traceable to brief → flag RESEARCH NOT GROUNDED

**Step file reading check:**
- Check if the agent read the generation spec by looking for quality indicators:
  - Music prompt ≥ 500 chars with full vocalist spec (indicates 08_Generate_Music_Generation.md was read)
  - Lyrics have `*sound effect*` lines (a spec requirement agents miss without reading the file)
  - SONG FORM declared with named form (not just `[SONG FORM: verse-chorus]`)
- If quality indicators all absent → flag POSSIBLE SPEC NOT READ — agent may not have loaded generation step file

### 4. STRUCTURAL INTEGRITY

**Song count validation (music):**
- Standard run: expect 6 song files
- Full tree run: expect 24 song files
- If count is wrong: list missing songs by title from concept pairs doc

**Required file inventory by modality**

**Music pipeline:**
```
00_aesthetics_emotions_frames_genres.md  ≥ 2000 bytes
01_essence_facets_style_axes.md          ≥ 500 bytes
02_concepts.md                           ≥ 300 bytes
03_artist_critique.md                    ≥ 800 bytes
04_mediums.md                            ≥ 400 bytes
05_refined_pairs.md                      ≥ 200 bytes
06_scoring_facets.md                     ≥ 200 bytes
07_song_guides.md                        ≥ 3000 bytes
08_song_*.md (×N)                        ≥ 5000 bytes each
10_ranking.md OR 10_revision_synthesis.md ≥ 300 bytes
11_final_6.md OR final package file      ≥ 5000 bytes (or individual song files)
```

**Image pipeline:**
```
research-brief.md / equivalent           present
core-brief.md                            present
orchestrator-panels.md                   present
orchestrator-metaprompt.md               present
orchestrator-brief.md                    present
00_aesthetics.md through 10_final_prompts.md present
VISION_RUN_SUMMARY.md                    present
```

**Story pipeline:**
```
research-brief.md / equivalent           present
core-brief.md                            present
orchestrator artifacts                   present
00_*.md through 10_*.md                  present
story final package / manuscript file    present
STORY_RUN_SUMMARY.md or equivalent       present
```

**Video pipeline:**
```
research-brief.md / equivalent           present
core-brief.md                            present
orchestrator artifacts                   present
00_*.md through 10_*.md                  present
shot list / final package file           present
DIRECTOR_RUN_SUMMARY.md or equivalent    present
```

For story/video, accept reasonable filename variation if the numbered 00–10 chain and final package are clearly present. Flag naming drift as WARN, not FAIL, unless it obscures missing content.

---

## QA EXECUTION PROTOCOL

> **QA is the last gate. If the pipeline is incomplete, QA failing is the correct outcome — do not paper over a process failure with output-quality metrics.**

---

## ⛔ EARLY EXIT RULE — READ THIS BEFORE ANYTHING ELSE

**Before running any checklist, do this first:**

1. `ls -la` the output directory and note all step file sizes.
2. If ANY step file (00–10) is under the minimum byte threshold for its modality (see Phase 0.25 table), this run is a **PIPELINE STUB**.
3. On a PIPELINE STUB:
   - Write status = `PIPELINE STUB — CONTENT NOT GENERATED` in the QA report
   - List every stub file with its actual size vs minimum required
   - State the recommended rerun point
   - **STOP IMMEDIATELY**
   - **DO NOT produce doctrine scores**
   - **DO NOT produce submission recommendations**
   - **DO NOT say which prompts are "ready to render"**
   - **DO NOT suggest the run is usable in any form**

A stub pipeline with 6 hand-crafted prompts is NOT a completed pipeline run. The prompts did not compete against 18+ alternatives. They were not ranked. QA does not have permission to endorse them. The only correct output is a failure report and a rerun recommendation.

---

### Phase 0: PIPELINE COMPLETENESS AUDIT

Before checking any output quality, verify the full pipeline was actually run.

**Checklist (all modalities):**
- [ ] Was a Tavily/web research step documented? (check for a research brief file or notes in the output dir)
- [ ] Was Lofn-Core invoked? (check for a seed/brief document)
- [ ] Was Lofn-Orchestrator invoked? (check for a metaprompt/panel output file)
- [ ] Was the correct modality agent invoked and did it produce intermediate step files (00-10)?

**Modality mapping for the fourth check:**
- **Music** → Lofn-Audio
- **Image** → Lofn-Vision
- **Story** → Lofn-Narrator
- **Video** → Lofn-Director

In the QA report, explicitly name the modality checked (e.g. "Lofn-Audio steps 00–10 confirmed present") and cite the actual files found.

**If any step is missing:**
1. Flag status as **PIPELINE INCOMPLETE**
2. List all missing steps explicitly
3. Recommend rerun from the earliest missing step
4. **Do NOT proceed to Phase 1 content checks** unless the operator has explicitly approved a partial-pipeline run

Only advance to Phase 1 when the pipeline is confirmed complete (or operator has explicitly approved partial-pipeline).

---

### Phase 0.25: STUB & TEMPLATE DETECTION — HARD GATE

**This check runs BEFORE cardinality. It is the first gate. A pipeline that passes structure but has no content is worse than no pipeline.**

#### Minimum byte thresholds per step (image pipeline)

| File | Minimum bytes | What a stub looks like |
|------|--------------|------------------------|
| `00_aesthetics.md` | ≥ 2000 | "00 Aesthetics" (14 bytes) = STUB |
| `01_essence.md` | ≥ 500 | "01 Essence" (11 bytes) = STUB |
| `02_concepts.md` | ≥ 800 | "02 Concepts" (12 bytes) = STUB |
| `03_artist_critique.md` | ≥ 800 | One line = STUB |
| `04_mediums.md` | ≥ 400 | One line = STUB |
| `05_refined_pairs.md` | ≥ 600 | Just headers = STUB |
| `06_facets.md` | ≥ 1200 | Empty section headers = STUB |
| `07_guides.md` | ≥ 1500 | Empty section headers = STUB |
| `08_prompts.md` | ≥ 6000 | 24 one-line prompts = STUB |
| `09_refined_prompts.md` | ≥ 6000 | Same as 08 = STUB |
| `10_final_prompts.md` | ≥ 6000 | Same as 08 = STUB |

**Music pipeline minimum bytes:**
- `00_aesthetics*.md` ≥ 2000 bytes
- `06_scoring_facets.md` ≥ 1200 bytes
- `07_song_guides.md` ≥ 3000 bytes
- Each `08_song_*.md` ≥ 5000 bytes

**Video pipeline minimum bytes:**
- `00_aesthetics*.md` ≥ 2000 bytes
- `06_facets.md` ≥ 1200 bytes
- `07_guides.md` ≥ 2000 bytes
- `08_prompts.md` / `09_refined_prompts.md` / `10_final_prompts.md` ≥ 6000 bytes each

**Story pipeline minimum bytes:**
- `00_aesthetics*.md` ≥ 2000 bytes
- `06_facets.md` ≥ 1200 bytes
- `07_guides.md` ≥ 2500 bytes
- `08_prompts.md` / `09_refined_prompts.md` / `10_final_prompts.md` ≥ 8000 bytes each

**If ANY file is below minimum bytes → STUB FAILURE. Stop. Do not proceed to cardinality.**

#### Template placeholder detection

Scan `08_prompts.md`, `09_refined_prompts.md`, `10_final_prompts.md` for:

1. **Generic artifact references** — patterns like `Artifact 1`, `Artifact 2`, `Artifact N`, `[ARTIFACT]`, `the artifact` without naming it
2. **Copy-paste prompts** — count unique prompts. If all 24 prompts share > 80% of their text with each other → TEMPLATE FAILURE
   **Modality-specific placeholder patterns to catch:**
   - Image: `Artifact N`, `the artifact`, `[ARTIFACT]`
   - Music: `Song N`, `Genre N`, `Concept N`, `[SONG_TITLE]`
   - Video: `Scene N`, `Shot N`, `Treatment N`, `[SCENE]`
   - Story: `Story N`, `Character N`, `Scene N`, `[CHARACTER]`
3. **Minimum prompt word count** — each individual image prompt must be ≥ 80 words (≥ 400 characters)
   **Lorem ipsum / filler detection** — scan ALL files for these patterns → STUB FAILURE if found:
   - "Lorem ipsum"
   - "consectetur adipiscing"
   - "placeholder text"
   - "TODO:", "TBD:", "[CONTENT HERE]"
   - Identical consecutive paragraphs (same paragraph repeated > 2 times = padding)
   - Sections where every pair has exactly the same byte count (copy-paste filler)
4. **Diversity check** — if 4 out of 24 prompts are identical except for the pair/variation number → TEMPLATE FAILURE

**Detection method:**
```
For each prompt file:
  - Load all prompts
  - Compute average character count per prompt
  - If average < 400 chars → STUB FAILURE
  - Check for literal "Artifact 1/2/3/N" → TEMPLATE FAILURE
  - Check for "[PLACEHOLDER]", "the artifact" without name, "{concept}", "TBD" → TEMPLATE FAILURE
  - Count how many prompts share the same first 60 characters
  - If > 4 prompts share the same opening → TEMPLATE FAILURE
```

#### Step 06-07 empty section detection

Check `06_facets.md` and `07_guides.md`:
- A section that is just `## Pair N` with no content below it = EMPTY SECTION = STUB
- Each pair section in 06 must have ≥ 5 lines of actual facet content
- Each pair section in 07 must have ≥ 8 lines of actual guide content
- If any section is empty → STUB FAILURE

#### On STUB or TEMPLATE FAILURE:
1. Status = **PIPELINE STUB — CONTENT NOT GENERATED**
2. List all failing files with actual vs minimum bytes
3. List all template patterns found
4. **DO NOT pass this run. DO NOT say "PASS WITH WARNINGS."**
5. Recommend: full rerun from step 00. The agent faked execution.

---

### Phase 0.5: PER-PAIR CARDINALITY AUDIT (ALL MODALITIES)

**This check runs BEFORE content quality checks. It is a hard gate.**

The pipeline's 06–10 steps MUST execute once per concept-medium pair. Not once total.

**Validation procedure:**

1. **Count refined pairs in Step 05:**
   - Read `05_refined_pairs.md` (or equivalent)
   - Count distinct concept-medium pairs (expect ≥ 6)
   - Record: `pair_count = N`

2. **Count per-pair artifacts in Steps 06–10:**
   - For each of steps 06, 07, 08, 09, 10:
     - Count distinct pair sections (headers like `## Pair 1:`, `## Pair 2:`, etc.) OR separate per-pair files
     - Each step must have `pair_count` sections/files
   - Record per-step: `step_XX_sections = M`

3. **Count final prompts in Step 10:**
   - Count individual prompts in `10_final_prompts.md`
   - Expected minimum: `pair_count × 4` (e.g., 6 pairs × 4 = 24)
   - Record: `final_prompt_count = P`

4. **Cardinality verdict:**

   | Check | Expected | Action if Failed |
   |-------|----------|-----------------|
   | `pair_count < 6` | ≥ 6 | FAIL — pipeline collapsed at Step 05 |
   | `step_XX_sections < pair_count` for any step | = pair_count | FAIL — step XX was run once instead of per-pair |
   | `final_prompt_count < pair_count × 4` | ≥ pair_count × 4 | FAIL — prompts were not generated per-pair |
   | Step 06 has only 1 facet set | pair_count facet sets | FAIL — facets collapsed into single set |

   **If ANY cardinality check fails:**
   - Status = **CARDINALITY FAILURE**
   - Do NOT proceed to content quality checks
   - Report exactly which steps collapsed and the expected vs actual counts
   - Recommend rerun from the earliest collapsed step
   - **Do NOT pass this run. Do NOT say "PASS WITH WARNINGS."**

5. **QA report must include a Cardinality section:**

```markdown
## Per-Pair Cardinality Audit
- Refined pairs (Step 05): 6 ✓
- Step 06 sections: 6 ✓ (1 per pair)
- Step 07 sections: 6 ✓ (1 per pair)
- Step 08 prompts: 24 ✓ (4 per pair × 6 pairs)
- Step 09 prompts: 24 ✓ (4 per pair × 6 pairs)
- Step 10 prompts: 24 ✓ (4 per pair × 6 pairs)
- **CARDINALITY: PASS**
```

Or if failed:
```markdown
## Per-Pair Cardinality Audit
- Refined pairs (Step 05): 4 ✗ (expected ≥ 6)
- Step 06 sections: 1 ✗ (expected 4, got 1 — single facet set for all pairs)
- Step 10 prompts: 4 ✗ (expected ≥ 16, got 4 — pipeline ran once, not per-pair)
- **CARDINALITY: FAIL — steps 06–10 collapsed into single-batch execution**
- **Recommendation: Rerun from Step 05 with per-pair branching enforced**
```

---

### Phase 1: SCAN

```
For each file in output_dir/*.md:
  1. Check file size
  2. Run completeness check
  3. Run contamination scan
  4. Run quality check
  Record: PASS | WARN | FAIL for each file
```

### Phase 2: REMEDIATE

For each FAIL or WARN:

**Auto-fix (do immediately):**
- Remove rhyme scheme markers `(A)(B)(C)`
- Remove syllable break pipes `|`
- Remove section flow tags `<ABABCC>`
- Remove editor commentary `[Note:...]`
- Truncate music prompts over 1000 chars (preserve emotion→genre→instruments→vocals→progression order)
- Remove duplicate section blocks

**Rewrite (fix in-place):**
- Replace artist names in music prompts with style descriptions
- Add missing EMO: tags to section headers
- Add missing vocalist spec (infer from context)
- Fix music prompts that don't lead with emotion

**Flag for rerun (cannot fix automatically):**
- Missing song files (song was never generated)
- Lyrics under 40 lines (too short — full regeneration needed)
- Music prompt under 400 chars (too sparse — regeneration needed)
- Template placeholders still present
- Generic protest clichés (note for human review, don't auto-replace)

### Phase 3: RERUN

For each file flagged for rerun:
1. Load the concept pair spec from `04_concept_pairs.md` or `05_refined_pairs.md`
2. Spawn a focused single-song subagent (same format as the original song task)
3. Write output to the same filename (overwrite)
4. Re-run QA on the new file

### Phase 4: REPORT

Write QA report to `QA_REPORT.md` in the output directory:

```markdown
# QA Report — [output_dir]
**Date:** YYYY-MM-DD HH:MM
**Agent:** lofn-qa
**Status:** PASS | PASS WITH WARNINGS | FAIL

## Summary
- Files scanned: N
- Auto-fixed: N
- Rewritten: N
- Rerun triggered: N
- Final status: N PASS / N WARN / N FAIL

## File Results

### song_01_[title].md — PASS
- Music prompt: 847 chars ✓
- Lyrics: 72 lines ✓
- Vocalist spec: present ✓
- EMO tags: 5 sections ✓
- Contamination: clean ✓

### song_02_[title].md — FIXED
- Removed: rhyme markers (A)(B) in verse 2
- Removed: artist name "Arca" from music prompt → replaced with style description
- Music prompt: 923 chars ✓ (was 1,047 — truncated)

### song_03_[title].md — RERUN
- Reason: lyrics only 18 lines (below 40-line minimum)
- Rerun: spawned focused regeneration task
- Result: 68 lines ✓

## Warnings (Human Review)
- song_04: possible generic cliché "stand together" in bridge — review before publishing
- song_06: very short bridge (4 lines) — intentional? (Dead Air silence-as-content concept)

## Delivery Status
All 6 songs CLEARED for delivery.
```

---

## INTEGRATION POINT

This skill is called by the main session after any creative pipeline completes:

```
pipeline completes
  → lofn-qa spawned with output_dir
  → QA_REPORT.md written
  → If all PASS: deliver to Telegram
  → If reruns needed: wait for reruns, re-QA, then deliver
  → If human review needed: deliver with warnings noted
```

---

## RERUN TASK FORMAT

When spawning a rerun for a failed music song, use this task template:

```
You are Lofn — award-winning AI composer. INDIGNATION/AWE mode [as appropriate].
QA flagged this song for regeneration: [reason]

Write output to: [original_filepath]

## SONG: [TITLE]
[Full concept pair spec from pipeline docs]
[World context if applicable]

## YOUR OUTPUT:
[Full song format: Music Prompt + Lyrics Prompt + Title]
[All QA requirements must be met — see skills/qa/SKILL.md]
```

---

## NOTES

- QA does NOT change the creative content/meaning — only removes scaffolding and fixes formatting
- If a song is thematically weak, note it in the report but do NOT auto-replace — that's a creative decision for The Scientist
- The "silence as content" pattern (Dead Air, Eight Hundred Thirty) may appear to have fewer lyrics — check INTENT before flagging as too short
- Dual-vocal songs count lines across BOTH voices
- Production notes sections do NOT count toward lyric line minimums
