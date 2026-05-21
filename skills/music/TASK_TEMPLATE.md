# Lofn Music - Subagent Architecture Pattern

## ⛔ OUTPUT TYPE: SUNO PROMPTS ONLY

**You write prompts and lyrics for Suno. You do NOT call any music generation tool or API.**
Do NOT use `music_generate` or any audio tool. Your artifacts are `.md` files.
The Scientist or a separate submission step handles actual Suno API calls.

---

## THE CORRECT SPLIT (from original Lofn ui.py)

The original Lofn app ran this exact pattern:
1. `generate_concept_mediums()` - coordinator phase, internally calls each steps 00-05 chain in sequence and returns all concept-medium pairs
2. `select_best_pairs()` - panel votes on top N pairs
3. `generate_prompts_for_pair(pair)` - pair phase, called ONCE PER PAIR in parallel, internally calls each steps 06-10 chain in sequence

Important correction: “one function call” in the Streamlit UI did **not** mean one creative prompt. It meant a wrapper function that performed multiple LLM chain calls. OpenClaw agents must preserve that internal chain depth.

**This is the mandatory architecture for all future runs.**

---

## PRE-AUDIO ORCHESTRATOR GATE — REQUIRED

Before coordinator Step 00, validate that the run has a real Lofn-Core + orchestrator packet:

```bash
python3 /data/.openclaw/workspace/scripts/validate_orchestrator_packet.py <run_dir>
```

The packet must include a substantial seed lineage, Golden Seed/core seed, 3-panel debate (baseline → group transformation → hyper-skeptic/skeptic transformation), metaprompt, pair assignments with rationale, and audio handoff. If this fails, do not proceed to audio. Launch or request `lofn-orchestrator` work instead.

---

## CANONICAL STEP ARTIFACT PROVENANCE — REQUIRED

Every canonical step file must follow `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md` and include:

1. `## 0. Step Provenance` — exact step file loaded, input artifacts used, model call mode, validation command.
2. `## 1. Input Context Digest` — concrete digest of prior artifacts; must name actual concepts/facets/media, not generic task text.
3. `## 2. Step Template Requirements Applied` — specific requirements from the loaded step prompt.
4. `## 3. Model Response / Creative Work` — the actual creative output.
5. `## 4. Self-Critique Against Step Requirements` — adversarial check.
6. `## 5. Validation Result` — paste validator output after pass.

A file with the right filename but without these sections is a backfilled artifact and fails. Do not write “line 1 / line 2” placeholders, repeated paragraphs, or self-check claims contradicted by the artifact body.

---

## SUBAGENT 1: Steps 00-05 (Concept-Medium Generation)

Receives, in this order:
1. Golden Seed lineage + full Golden Seed
2. active Lofn personality / mode
3. orchestrator output (metaprompt, panel, assignments)
4. constraints and QA/output contract

The coordinator must generate from the seed first. Constraint axes are vocabulary, not the form of the song.

Executes as **six separate LLM turns**, matching original `generate_concept_mediums()` in `lofn/llm_integration.py`:
- Step 00: read `steps/00_Generate_Music_Aesthetics_And_Genres.md`, call the model, write `step00_aesthetics_and_genres.md` using the canonical provenance template. Provenance must cite the validated orchestrator packet.
- Step 01: read `steps/01_Generate_Music_Essence_And_Facets.md`, call the model using Step 00 output, write `step01_essence_and_facets.md` using the canonical provenance template
- Step 02: read `steps/02_Generate_Music_Concepts.md`, call the model using Step 01 output, write `step02_concepts.md` using the canonical provenance template
- Step 03: read `steps/03_Generate_Music_Artist_And_Critique.md`, call the model using Step 02 output, write `step03_artist_and_critique.md` using the canonical provenance template
- Step 04: read `steps/04_Generate_Music_Medium.md`, call the model using Step 03 output, write `step04_medium.md` using the canonical provenance template
- Step 05: read `steps/05_Generate_Music_Refine_Medium.md`, call the model using Step 04 output, write `step05_refine_medium.md` using the canonical provenance template and `concept_medium_pairs.json` (6 pairs)

**Do not combine Steps 00–05 into one prompt, one response, or renamed summary files.** Files like `step00_coordinator_overview.md` or `step03_pair_concept_matrix.md` are summaries, not canonical original-Lofn step outputs.

### Mandatory validation + retry loop for Steps 00–05
After writing each step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair the artifact in place and rerun with `--attempt 2`, then `--attempt 3` if needed. After 3 failed attempts, stop the run, write `TIMEOUT_STATUS.md` or `VALIDATION_BLOCKED.md`, and escalate. Do **not** continue to the next step with a failed artifact.

**STOP HERE. Do not proceed to step 06.**

---

## SUBAGENTS 2-7: Steps 06-10 (One Per Pair)

Each receives:
- The Golden Seed lineage + full Golden Seed
- The active Lofn personality / mode
- The orchestrator metaprompt
- ONE specific concept-medium pair (name, concept text, medium/production style)
- The constraint axes
- The panel composition
- The output/QA contract as the final appendix

Pair-agent task prompts MUST NOT begin with line counts, EMO tags, or prompt-shape requirements. Begin with the seed, then the pair's dangerous requirement / Lofn-specific wrongness, then creative permission, then the required Suno structure. The QA contract remains blocking, but it is not the muse.

Each executes (for its ONE pair only) as **five separate LLM turns or five externally orchestrated sub-steps**, matching original `generate_music_prompts()` in `lofn/llm_integration.py`. If a single pair-agent cannot prove the five turns happened beyond filenames, the controller should spawn one subtask per step or QA must treat the result as suspect:
- Step 06: read `steps/06_Generate_Music_Facets.md`, call the model, write `pair_{NN}_step06_facets.md` using the canonical provenance template. Provenance must cite the Golden Seed, orchestrator metaprompt, pair assignment, Step 05 concept-medium pair, and prior coordinator outputs.
- Step 07: read `steps/07_Generate_Music_Song_Guides.md`, call the model using Step 06 output, write `pair_{NN}_step07_song_guides.md` using the canonical provenance template
- Step 08: read `steps/08_Generate_Music_Generation.md`, call the model using Step 07 output, write `pair_{NN}_step08_generation.md` using the canonical provenance template
- Step 09: read `steps/09_Generate_Music_Artist_Refined.md`, call the model using Step 08 output, write `pair_{NN}_step09_artist_refined.md` using the canonical provenance template
- Step 10: read `steps/10_Generate_Music_Revision_Synthesis.md`, call the model using Step 09 output, write `pair_{NN}_step10_revision_synthesis.md` using the canonical provenance template

**Do not combine Steps 06–10 into one prompt, one response, or one omnibus file.** A convenience rollup may be created afterward as `pair_{NN}_steps_06_10_rollup.md`, but it is not a substitute for the five canonical step files. A file with the right name but shallow content, missing required headers, or self-check claims contradicted by grep is a failure, not a partial pass.

### Mandatory validation + retry loop for Steps 06–10
After writing each pair step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair the artifact in place and rerun with `--attempt 2`, then `--attempt 3` if needed. After 3 failed attempts, stop that pair, save `pair_{NN}_VALIDATION_BLOCKED.md`, and return the exact validator failure. Do **not** write Step 10 or claim the pair is complete if Step 06–09 failed.

Outputs to disk: five separate step files for its pair number
Returns: Step 10 final song prompts + lyrics as output text

---

## ORCHESTRATION FLOW

```
Main session
  └── spawns Subagent 1 (steps 00-05)
         └── writes concept_medium_pairs.json
  └── reads concept_medium_pairs.json
  └── spawns Subagents 2-7 in parallel (one per pair)
         └── each writes full song (prompt + lyrics)
  └── collects all 6 songs
  └── QA gate
  └── Deliver to Telegram
```

---

## concept_medium_pairs.json format
```json
[
  {
    "pair_num": 1,
    "concept": "Full refined concept text",
    "medium": "Full production style text",
    "artist_influence": "Named artist"
  }
]
```

## OUTPUT FORMAT FOR PAIR SUBAGENTS

Each pair subagent must return in Step 10 only after Step 06, Step 07, Step 08, and Step 09 have already been written as separate files. Step 10 must include:
- Suno/Udio music prompt (**target 850-1000 chars**, hard max 1000 chars, no artist names). It must be dense, producer-grade, and single paragraph: emotion → selected style label(s) from the run → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → avoidances. Prompts under 850 chars are only acceptable when explicitly justified as intentional minimalism in the local skeptic note.
- Full lyrics (70-120 sung lines, hard maximum 120) using the **full Step 10 Suno performance-script syntax**, not bare structural tags. <70 sung lines risks under-3min runtime and is a repair trigger in QA:
  - `[SONG FORM: <named form>]` declaration at the top of the lyrics block. The name must describe the form meaningfully, e.g. `[SONG FORM: Apology-Evidence-Chorus Pyramid]` or `[SONG FORM: Subtractive-Build Earned-Hope Arc]` — NOT `[SONG FORM: verse-chorus]`
  - Top context tag: `[Theme: ...]` or `[Setting: ...]`
  - Rich section headers with section, emotion, vocalist, and mix/performance cue, e.g. `[Verse 1 - EMO:Responsibility Vertigo - Female Vocalist - Close-mic]`
  - Standalone short SFX cues in asterisks, ≤5 words, e.g. `*calendar chime*`, `*microwave beeps*`
  - At least one non-lexical vocal hook where musically appropriate (`mm`, `ooh`, `ah`, whispered echo, etc.)
  - Performance/mix cues where structurally important (`No beats`, `Half-time`, `Double-time`, `whispered`, `filter sweep`, `choir flinch`, etc.)
  - No editor commentary, TODOs, rhyme letters, or syllable bars in final lyrics

Each pair subagent must also identify at least one **Lofn-specific move** that survives in the music prompt and lyrics: scientific specificity as feeling, AWE/INDIGNATION state-change, wrongness-as-beauty, hidden structural logic, literary/prayer/witness mode, or Open Laboratory continuity pressure. If none exists, revise before the pre-completion gate.

**Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags alone are NOT acceptable for final delivery.** They may appear in drafts, but Step 10 final files must be performance-ready for Suno.

### ⛔ PRE-COMPLETION GATE — ALL 5 MUST PASS BEFORE WRITING FINAL OUTPUT

Before writing your final step10 output, run this check. If any box is unchecked, revise and re-check.

**In the final output for each song:**
- [ ] Standalone `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` section exists: copy-paste-ready, single paragraph, target 850-1000 chars, hard max 1000 chars unless explicitly justified. It must include emotion → selected style label(s) from the run → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → avoidances. Scattered `[STYLE/TEMPO/KEY]`, `[SONIC WORLD]`, and `[PRODUCTION NOTES]` do NOT satisfy this gate.
- [ ] `[SONG FORM: <named form>]` declared at the top of **each variant lyrics block** (not plain `SONG FORM:` text; not `verse-chorus` — use a descriptive name)
- [ ] Every actual song section header includes section name, `EMO:`, vocalist cue, and mix/performance cue, e.g. `[Verse 1 - EMO:Weight - Female Vocalist - Close-mic]`. Bare `[EMO:...]` headers are **not acceptable** because Suno loses section structure.
- [ ] At least one standalone SFX cue in asterisks ≤5 words, e.g. `*inverter click*`, `*phone buzz*`
- [ ] At least one non-lexical vocal hook (`ooh`, `mm`, `ah`, whispered echo, call-response fragment)

**Document your check** (save as part of your step10 file or companion `step10_qa_pair{N}.md`):
```
GATE CHECK — Pair {N}, Variation {X}:
[SONG FORM]: ✓/✗ → [form name]
EMO tags: ✓/✗ → [count] sections tagged
SFX cue: ✓/✗ → [the line]
Non-lexical hook: ✓/✗ → [the line]
```
If any check is ✗, revise the lyrics before completing. Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags are NOT acceptable in final output.

Written to: `step10_final_pair{N}.md`
