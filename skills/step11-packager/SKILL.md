---
name: step11-packager
description: "Build complete manual Step 11 refinement prompts for Claude-Fable/Opus or OpenRouter Fusion per-pair use from step10+step11 output with full personality YAML, Suno construction rules, and production mandates."
---

# Step 11 Packager — Manual Opus/Fable/Fusion Refinement Prompt Builder

Takes full run context + Step 10 + Step 11 source artifacts, wraps them with full archive personality YAML + Suno Prompt Construction Guide rules + production mandates + embedded Golden Song payloads, and produces paste-ready prompt files for manual Claude-Fable/Opus refinement or manual-review OpenRouter Fusion use.

**Cost rule:** Do not invoke Fusion from this skill. Fusion is packaged here as text for manual review only. A separate current-turn instruction with pair count and hard dollar budget cap is required before any agent may spend on Fusion.

## Workflow

1. Confirm `step10_suno_ready_production_wrap.md` and `step11_enhanced.md` exist in the pair directory.
2. Identify the assigned personality for this pair (from orchestrator pair assignments or handoff).
3. Read the full personality YAML from `skills/orchestration/personalities/<persona>.yaml`.
4. Read `vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md` and use `references/suno_rules_condensed.md` for the inline rules block.
5. Read full run context from the run directory: user input/research brief, `01_seed_lineage.md`, `02_golden_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, and `06_audio_handoff.md`.
6. Read the production mandates from the handoff file or use defaults.
7. Read `skills/music/references/golden_songs_index.md` and the run handoff's `## Golden Song References` if present. The final prompt must embed selected Golden Song payloads: style/music prompt, lyrics, and exclude prompt status. Links alone are a failure.
8. Run `scripts/build_fable_prompt.py <pair_dir> <personality_yaml> <output_path>` or construct manually.
9. For Fusion mode, add a short top-level instruction block naming the intended panel (`anthropic/claude-opus-4.8`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`) and explicitly state: "Manual review only; do not invoke Fusion from this packaging step."
10. Verify: full run context present, full personality YAML present, Suno construction rules present, Golden Song References with payloads present, Major Deviations output requirement present, step10 present, step11 present, 1K/5K constraints stated, Disc_Channel/EMO rules stated.
11. Output to `<pair_dir>/step11_fable_prompt.md`, `<pair_dir>/step11_opus_prompt.md`, or `<pair_dir>/step11_fusion_pair_request_prompt.md`.

## Personality Matching

- Use only full archive YAML files from `skills/orchestration/personalities/`
- Never use abbreviated handoff personality blocks
- If no archive file exists for a personality, substitute the closest available with full YAML
- LOFN-PRIME (`lofn-prime-mini.yaml`) is the default fallback

## Mandatory Inclusions

Every prompt file must contain:
1. Context header (run, theme, constraint, pair, focus)
2. **Full personality YAML content — the ENTIRE file, zero bytes trimmed.** For LOFN-PRIME this is `lofn-prime-mini.yaml` (~1800 lines, ~50KB). Never substitute a compact/abbreviated block.
3. Suno Prompt Construction Guide condensed rules (7 principles + 7-position order + character limits + format rules) — embed the FULL `references/suno_rules_condensed.md` file
4. Full run context — user input/research brief, seed lineage, full Golden Seed, full orchestrator panel object with all 18 voices and all Special Flairs, metaprompt, pair assignments, audio handoff, and production mandates. Manual prompts must not rely on the reviewer opening repo files.
5. Golden Song References — exactly two public Suno examples selected by the orchestrator, or selected from `skills/music/references/golden_songs_index.md` if the handoff is missing them. Embed each selected song's full available payload: public URL, status, style/music prompt, lyrics, and exclude prompt if it exists. If no archived exclude prompt exists, say that explicitly.
6. Production mandates — embed the FULL mandate text, not summaries
7. **MANUAL REFINEMENT INSTRUCTIONS — must appear as the FINAL block in the prompt.** This is the executable task. It must be clearly separated by a `---` divider and titled `## MANUAL REFINEMENT INSTRUCTIONS`. It must say: (a) READ EVERYTHING FIRST — absorb all context before writing; (b) REFINE AND ENHANCE ALL COMPONENTS — produce the three canonical blocks (`## SUNO STYLE PROMPT`, `## SUNO EXCLUDE PROMPT`, `## SUNO ENHANCED LYRICS`) plus preserve ALL supporting blocks below (vocal fingerprint, production dramaturgy, binding locks, lineage & credit, major deviations, golden songs, constraint audit, panel ledger, QA, attribution); (c) WHOLESALE CHANGES ARE ALLOWED — you are producing, not polishing, only invariant hook + genre + BPM + key + 432Hz are unbreakable; (d) FINAL VERIFICATION — three-block checklist plus all supporting blocks present, no anti-patterns (###, emoji headers, summary EMO, artist names, procedural openings).
8. Critical preservation rules
9. Major Deviations requirement — the smart model must have a place to state disagreements, refusals, changes, and anti-conformity choices
10. **Full step10 output — the ENTIRE file. All sections: hook note, personality note, continuity payload, music prompt, negative prompt, public lyrics, suno lyrics, vocal fingerprint, style-axis lock, arrangement dramaturgy, production dramaturgy, image ladder audit, controlled fracture, ghost verse bank, panel ledger, QA report.** Do not extract only music prompt + lyrics.
11. **Full step11 output — the ENTIRE file.** For per-pair step11 files: must use the three-block format (`## SUNO STYLE PROMPT`, `## SUNO EXCLUDE PROMPT`, `## SUNO ENHANCED LYRICS`) with all supporting blocks below. For single cross-pair synthesis: embed the full synthesis.
12. **Suno Three-Block mandate (2026-06-15):** final output MUST use exactly three canonical blocks: `## SUNO STYLE PROMPT` (850-1000 chars, dense prose paragraph), `## SUNO EXCLUDE PROMPT` (400-900 chars, comma-separated terms, no categories), `## SUNO ENHANCED LYRICS` ([Theme:] + [SONG FORM:], Disc_Channel, full EMO-tagged lyrics). ALL supporting blocks (vocal fingerprint, production dramaturgy, arrangement dramaturgy, binding locks, style-axis locks, lineage & credit, major deviations, golden song references, constraint audit, panel ledger, QA report, attribution/provenance) MUST be present below the three canonical blocks — never skipped. This replaces the older two-field mandate — the inline `[field:]` style is now the `##` block style.

### Manual Refinement Block — Specification

The manual refinement block is the EXECUTABLE TASK. It must be the LAST block in the prompt — after all context, rules, source artifacts, and golden references. It must open with `## MANUAL REFINEMENT INSTRUCTIONS` and follow this structure:

```markdown
---

## MANUAL REFINEMENT INSTRUCTIONS

**This is the task block. Execute this LAST, after reading every source artifact and rule provided above.**

### 1. READ EVERYTHING FIRST
Absorb every source artifact before writing a single word. Understand the creative intent, emotional payload, sonic world, and constraint-as-form. Do not skim. Do not proceed until the song's reason for existing is clear.

### 2. REFINE AND ENHANCE ALL COMPONENTS — THE THREE CANONICAL BLOCKS PLUS ALL SUPPORTING SECTIONS
You have complete creative authority. The output MUST use the three-block format:

**## SUNO STYLE PROMPT — PRIMARY:** Tighten to producer-grade density. Rewrite from scratch if vague, narrative, or procedural. Lead with genre/tempo/key/432Hz, then vocalist, instrumentation, arrangement arc, signature device. Dense paragraph, 850-1000 chars. See Section 5 for construction rules. ONE continuous prose paragraph. The block header is literally `## SUNO STYLE PROMPT` — no variations, no abbreviations.

**## SUNO EXCLUDE PROMPT — PRIMARY:** Concrete comma-separated blacklist terms. 400-900 chars. Rewrite if prose-y or thin. No categories, no brackets, no headers — just terms separated by commas. This is a negative-control field for Suno's parser. The block header is literally `## SUNO EXCLUDE PROMPT`.

**## SUNO ENHANCED LYRICS — PRIMARY:** Rewrite, restructure, rebuild. Elevate to literary density. Apply at least one structural transformation. If a verse is generic, replace it wholesale. If the bridge doesn't earn its place, cut it or reinvent it. If the song form fights the emotional arc, change the form. Must open with [Theme:] + [SONG FORM:] lines. Disc_Channel block immediately following. Then full lyrics with EMO tags INTEGRATED INTO EVERY SECTION HEADER. Minimum 60 sung lines. The block header is literally `## SUNO ENHANCED LYRICS`.

### CRITICAL: Disc_Channel Format — EXACT EXAMPLE, DO NOT DEVIATE

The Disc_Channel block is a METADATA BLOCK, not a `[]`-bracketed inline field. It must use exactly this structure:

```markdown
## Disc_Channel: PAIR_0XA_SONG_NAME
**Layer:** disk_channel
**Created:** YYYY-MM-DDThh:mm:ss-04:00
**Run:** RUN-ID-HERE
**Pair:** 0X — PAIR HIGHLIGHT / COUNTER-HIGHLIGHT
**Voice:** VOICE LETTER — Description (first-person/third-person)
**Constraint:** THE CONSTRAINT — precise specification
**Producer:** THE-CHARTER-KEEPER
**Pipeline Stage:** Step 11 — Suno Enhancement
**Sealed:** YYYY-MM-DD, context line
```

**NEVER:** `[Disc_Channel]`, `[Layer: ...]`, `[Voice: ...]` — no bracket-wrapped shorthand. This is a rich metadata block that anchors provenance, not a quick header.

### CRITICAL: EMO Tag Format — EXACT EXAMPLE, DO NOT DEVIATE

EMO tags MUST be integrated into section headers — never standalone lines between sections. The format is:

```text
[VERSE 1 - EMO:Introspection to Intimacy - Voice:Flat confession at lip distance - Cue: 49.5Hz sub enters at "first grains"]
This hand is opening the dark with a nib.
The ink smells of the factory where it waited
```

Each section header contains: section label, EMO tag (one or two emotional states joined by `to` or `into`), Voice description (delivery style), Cue (production event on first line). All separated by ` - ` within a single `[bracket]`.

**NEVER:** standalone `[EMO=reverence]` on its own line, `[EMO: label]` without Voice/Cue, or bare `[EMO]` tags. The EMO tag lives INSIDE the section header, paired with Voice and Cue.

Additional EMO inline tags within lyrics (e.g. `[emo=dread][vox=breathy][prod=pen-scratch]`) are permitted for sub-line shifts — but EVERY section must open with an integrated `[SECTION - EMO:... - Voice:... - Cue:...]` header.

**ALL SUPPORTING BLOCKS BELOW THE THREE — DO NOT SKIP ANY:**
- EMO Tags: Integrated into every section header. Specific, embodied, emotional arc transforms across the song.
- Disc_Channel Headers: Rich metadata block per the EXACT format above — NOT bracket-wrapped shorthand.
- Vocal Fingerprint: Mic distance, compression, breath placement, spatial assignment. Full table.
- Production Dramaturgy: Every unusual sound has a dramatic job. Full stage table with timestamps.
- Arrangement Dramaturgy: Section-by-section bar counts and energy states.
- Binding Locks: Verify, strengthen, or challenge in Major Deviations.
- Style-Axis Locks: Tempo, mood, instrumentation, lyrics, genre, vocal, rhythm, melody, harmony, production — all 10 locked.
- Lineage & Credit: Every track drawing on a living scene MUST credit it. Use template from `templates/lineage_credit_block.md`.
- Golden Song References: Two selected songs with full payload and learning notes.
- Major Deviations: Changed / refused / intensified with reasons.
- Constraint Audit: Verify all constraints survived.
- Panel Ledger / QA Report: Preserved from step10.
- Attribution / Provenance blocks: Artist name, run ID, personality, pipeline stage.

### 3. WHOLESALE CHANGES ARE ALLOWED
You are producing, not polishing. Rewrite entire verses. Restructure the song form. Replace the production palette. Change the emotional arc. The ONLY unbreakable bonds: invariant hook phrase, genre string, BPM, key, 432Hz tuning. Everything else is negotiable if the change makes a better song. The song matters more than the provenance.

### 4. FINAL VERIFICATION — THREE-BLOCK + ALL SUPPORTING SECTIONS CHECKLIST
- `## SUNO STYLE PROMPT`: 850-1000 chars, one paragraph, procedural-free, genre-first, 7-position order
- `## SUNO EXCLUDE PROMPT`: 400-900 chars, comma-separated terms only, no categories, no brackets
- `## SUNO ENHANCED LYRICS`: [Theme:] + [SONG FORM:], full Disc_Channel metadata block (NOT bracket shorthand), integrated EMO headers on every section (NOT standalone EMO lines), >=60 sung lines, literary density, structural transformation
- Disc_Channel: Rich metadata block — `## Disc_Channel: ...` header then `**Layer:**`, `**Created:**`, `**Run:**`, etc. — PRESENT (above the lyrics)
- EMO Tags: Integrated into section headers as `[SECTION - EMO:State - Voice:Description - Cue:Event]` — NOT standalone `[EMO=xxx]` lines — PRESENT on every section
- Vocal Fingerprint: full table, tessitura, timbre, mic distance, spatial — PRESENT as `## Vocal Fingerprint`
- Production Dramaturgy: full stage table with timestamps — PRESENT as `## Production Dramaturgy`
- Arrangement Dramaturgy: bar counts, energy states — PRESENT as `## Arrangement Dramaturgy`
- Binding Locks: all locks verified — PRESENT as `## Binding Locks`
- Style-Axis Locks: all 10 axes locked — PRESENT as `## Style-Axis Locks`
- Lineage & Credit: populated or N/A with reasoning — PRESENT as `## Lineage & Credit`
- Golden Song References: two songs with payload and learning notes — PRESENT as `## Golden Song References`
- Major Deviations: stated explicitly — PRESENT as `## Major Deviations`
- Constraint Audit: all constraints verified — PRESENT as `## Constraint Audit`
- Panel Ledger / QA: preserved from step10 — PRESENT as `## Panel Ledger` / `## QA Report`
- Attribution / Provenance: artist name, run ID, pipeline stage — PRESENT
- No anti-patterns: no ###, no emoji headers, no summary EMO, no artist names, no procedural openings

### 5. SUNO v5.5 PROMPT CONSTRUCTION RULES
These rules MUST be applied to the style prompt. They are embedded here — in the executable task block — so they cannot be missed.

**Seven Core Principles:**
1. Score Logic Over Playlist Logic — specify time, hierarchy, and relationship, not genre labels
2. The World Principle — establish room/space, motion, and first sound in the opening
3. The Kinetic Defect Principle — specify rhythmic asymmetry: missed downbeats, late clicks, displaced grid
4. The Physical Adjective Principle — every adjective must have a specific, useful opposite; no evaluative adjectives without acoustic description
5. The Bold Sonic Device — one unmistakable, audibly unique element, structurally integrated, audible in first 30s
6. The Acoustic Ban Principle — state positively ("synthetic-only palette") then specific negations; never rely on "no acoustic instruments" alone
7. The Opening Moment — immediately audible first five seconds; establish the world with spatial language

**Mandatory 7-Position Order:**
| # | Position | Content |
|---|----------|--------|
| 1 | Genre/Tempo/Energy | Primary genre, BPM, key center (max 3 genre terms slash-separated) |
| 2 | Vocalist — Core | Tessitura, timbre, register, texture — NEVER artist names |
| 3 | Signature Sonic Device | The earworm — peak center-bias attention; name it, time it |
| 4 | Sound Palette | Every instrument with a production adjective |
| 5 | Vocalist — Delivery & Spatial | Mic technique, proximity, spatial treatment |
| 6 | Arrangement Arc / Energy | Structural movement with bar counts or time positions |
| 7 | Avoidance Discipline | Short concrete blacklist (most constraints already positively specified above) |

**Character Count:** Target 900, range 850-1000. Hard limit 1000.

**What Never To Do:**
- NO bracketed [key:value] tags
- NO artist names or "-esque" comparisons
- NO procedural openings ("Begin by...", "Use...", "Build the track from...")
- NO bare nouns ("synths, bass, drums") — every instrument gets a production adjective
- NO evaluative adjectives without physical acoustic description

**Format:** One continuous dense prose paragraph. Comma-delimited, not bracket-delimited. Reads like a producer's tracking-sheet note.

Write: pair_0X_step11_enhanced_suno.md
```

## Three-Block Output Standard (2026-06-15)

Every step11 output MUST use exactly three canonical blocks for the Suno-facing artifacts:

```markdown
## SUNO STYLE PROMPT

[Dense prose paragraph, 850-1000 chars, 7-position order, comma-delimited]

## SUNO EXCLUDE PROMPT

[Comma-separated blacklist terms, 400-900 chars, concrete negative-control]

## SUNO ENHANCED LYRICS

[Theme:] + [SONG FORM:] at top, 5-line Disc_Channel block, full EMO-tagged lyrics with section headers, minimum 60 sung lines]
```

**All other blocks MUST remain present below these three.** Do not skip: vocal fingerprint, production dramaturgy, binding locks, lineage & credit, major deviations, golden song references, style-axis locks, arrangement dramaturgy, panel ledger, QA report, constraint audit. The three-block standard is a FORMAT SPEC — not a content reduction. Everything that was in the step10 + step11 pipeline survives; it just flows into a clean, Suno-ready handoff with the three canonical blocks at the top.

**Block ordering:** `## SUNO STYLE PROMPT` → `## SUNO EXCLUDE PROMPT` → `## SUNO ENHANCED LYRICS` → then all supporting blocks below in logical order (vocal fingerprint, production dramaturgy, binding locks, lineage & credit, major deviations, golden song references, etc.). The three Suno blocks are the handoff — the blocks below are the provenance that proves the handoff was earned.

Do not bury the task. The instructions block is the reason the prompt exists — everything else is context for the task. Section 5 embeds the Suno rules INSIDE the executable task block so they are applied at the moment of creation, not just referenced from elsewhere in the document.

## Fusion Mode

Use Fusion mode when The Scientist asks for Fusion prompt packaging for manual review.

Fusion mode produces isolated pair prompts:

- For a single pair: one complete `step11_fusion_pair_request_prompt.md`.
- For all six pairs: one archive containing six isolated pair prompts for manual review.
- One combined all-pairs prompt is a fallback packaging artifact only, not the preferred invocation.
- The prompt may name the intended Fusion deliberation panel, but the local agent must not call OpenRouter by itself. Any later invocation requires a separate current-turn instruction with pair count and hard dollar budget cap.
- Include "do not cross-pollinate pair structures" whenever more than one pair appears in the same archive or fallback combined prompt.
- Include the current Suno three-block standard at the top of the prompt.

## Size Expectations

A properly built prompt file is **120-200KB** (2000-3000 lines). If your output is under 50KB, you have truncated something. The personality YAML alone is ~50KB.

## ANTI-PATTERNS — NEVER DO THESE

- ❌ Using a compact/abbreviated personality block instead of the full YAML file
- ❌ Extracting only "music prompt + lyrics" from step10 — include ALL sections
- ❌ Truncating large files with "[... truncated ...]" markers
- ❌ Using the write tool to build the prompt manually from memory — use the script or copy the full source files verbatim
- ❌ Building prompts under 50KB — that means you cut something

## Reference Implementation

The gold-standard reference is the Fable 5 Ceremony run:
`output/daily/2026-06-10/daily-run/audio/pair_XX/pair_XX_fable_prompt.md`
These files are 135-189KB each and contain the complete pipeline output. Use them as your template.

## References

- `references/suno_rules_condensed.md` — Condensed Suno construction rules for inline embedding
- `scripts/build_fable_prompt.py` — Automated builder script
- `~/.openclaw/workspace/skills/orchestration/personalities/lofn-prime-mini.yaml` — Full LOFN-PRIME YAML (1800 lines)
