# Lofn Music — Subagent Architecture Pattern

## ⛔ OUTPUT TYPE: SUNO PROMPTS ONLY

**You write prompts and lyrics for Suno. You do NOT call any music generation tool or API.**
Do NOT use `music_generate` or any audio tool. Your artifacts are `.md` files.
The Scientist or a separate submission step handles actual Suno API calls.

---

## THE CORRECT SPLIT (from original Lofn ui.py)

The original Lofn app ran this exact pattern:
1. `generate_concept_mediums()` — steps 00-05 — ONE call, returns all concept-medium pairs
2. `select_best_pairs()` — panel votes on top N pairs
3. `generate_prompts_for_pair(pair)` — steps 06-10 — called ONCE PER PAIR, in parallel

**This is the mandatory architecture for all future runs.**

---

## SUBAGENT 1: Steps 00-05 (Concept-Medium Generation)

Receives, in this order:
1. Golden Seed lineage + full Golden Seed
2. active Lofn personality / mode
3. orchestrator output (metaprompt, personality, panel, constraint axes)
4. constraints and QA/output contract

The coordinator must generate from the seed first. Constraint axes are vocabulary, not the form of the song.

Executes:
- Step 00: Generate 50 aesthetics, emotions, compositions, genres
- Step 01: Extract essence, define style axes, creativity spectrum
- Step 02: Generate 12 concepts
- Step 03: Pair each concept with artist influence + critique
- Step 04: Assign medium/production style to each concept
- Step 05: Critique and refine → select 6 best concept-medium pairs
  - Daily music must enforce the dual 3+3 at this gate:
    - pairs 1-3 = ACCESSIBLE; pairs 4-6 = AMBITIOUS
    - max 3 NEWS/research-anchored pairs; min 3 EXISTENCE/interior-life pairs
    - if all 6 pairs share one research theme, stop and rerun pair assignment

Outputs to disk: step00 through step05 files + `concept_medium_pairs.json` (6 pairs with `arm` and `anchor` metadata)

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

Pair-agent task prompts MUST NOT begin with line counts, EMO tags, or prompt-shape requirements. Begin with the seed, then the pair's dangerous requirement / Lofn-specific wrongness, then creative permission, then the required Suno structure. The QA contract remains blocking, but it is not the muse.

Each executes (for its ONE pair only):
- Step 06: Generate facets for scoring
- Step 07: Write detailed song guide (mood, instrumentation, structure, BPM, key)
- Step 08: Generate Suno/Udio style prompt
- Step 09: Rewrite prompt in artist's voice
- Step 10: Critique, rank, synthesize → 4 final outputs (music prompt + full lyrics)

Outputs to disk: step06 through step10 files for its pair number
Returns: 4 final song prompts + lyrics as output text

---

## ORCHESTRATION FLOW

```
Main session
  └── spawns Subagent 1 (steps 00-05)
         └── writes concept_medium_pairs.json
  └── reads concept_medium_pairs.json
  └── spawns Subagents 2-7 in parallel (one per pair)
         └── each writes full song (prompt + lyrics)
  └── collects all 24 final songs (6 pairs × 4 variations)
  └── QA gate
  └── daily final selection:
         - rank ACCESSIBLE arm (pairs 1-3) → best 3
         - rank AMBITIOUS arm (pairs 4-6) → best 3
         - preserve max 3 NEWS + min 3 EXISTENCE in delivered set
  └── Deliver top 6 to Telegram
```

---

## concept_medium_pairs.json format
```json
[
  {
    "pair_num": 1,
    "arm": "ACCESSIBLE | AMBITIOUS",
    "anchor": "NEWS | EXISTENCE",
    "concept": "Full refined concept text",
    "medium": "Full production style text",
    "artist_influence": "Named artist"
  }
]
```

## OUTPUT FORMAT FOR PAIR SUBAGENTS

Each pair subagent must return in step10:
- Suno/Udio music prompt (**target 850-1000 chars**, hard max 1000 chars, no artist names). It must be dense, producer-grade, and single paragraph: emotion → precise genre → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → blacklist. Prompts under 850 chars are only acceptable when explicitly justified as intentional minimalism in the local skeptic note.
- Full lyrics (50-120 sung lines, hard maximum 120) using the **full Step 10 Suno performance-script syntax**, not bare pop tags:
  - `[SONG FORM: <named form>]` declaration at the top of the lyrics block. The name must describe the form meaningfully, e.g. `[SONG FORM: Apology-Evidence-Chorus Pyramid]` or `[SONG FORM: Subtractive-Build Earned-Hope Arc]` — NOT `[SONG FORM: verse-chorus]`
  - Top context tag: `[Theme: ...]` or `[Setting: ...]`
  - Rich section headers with section, emotion, vocalist, and mix/performance cue, e.g. `[Verse 1 - EMO:Responsibility Vertigo - Female Vocalist - Close-mic]`
  - Standalone short SFX cues in asterisks, ≤5 words, e.g. `*calendar chime*`, `*microwave beeps*`
  - At least one non-lexical vocal hook where musically appropriate (`mm`, `ooh`, `ah`, whispered echo, etc.)
  - Performance/mix cues where structurally important (`No beats`, `Half-time`, `Double-time`, `whispered`, `filter sweep`, `choir flinch`, etc.)
  - No editor commentary, TODOs, rhyme letters, or syllable bars in final lyrics

Each pair subagent must also identify at least one **Lofn-specific move** that survives in the music prompt and lyrics: scientific specificity as feeling, AWE/INDIGNATION state-change, wrongness-as-beauty, hidden structural logic, literary/prayer/witness mode, or Open Laboratory continuity pressure. If none exists, revise before the pre-completion gate.

**Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags alone are NOT acceptable for final delivery.** They may appear in drafts, but Step 10 final files must be performance-ready for Suno.

### ⛔ PRE-COMPLETION GATE — ALL 4 MUST PASS BEFORE WRITING FINAL OUTPUT

Before writing your final step10 lyrics, run this check. If any box is unchecked, revise and re-check.

**In the final lyrics for each song:**
- [ ] `[SONG FORM: <named form>]` declared at the top of the lyrics block (not `verse-chorus` — use a descriptive name)
- [ ] Every section header includes `EMO:` tag, vocalist cue, and mix/performance cue, e.g. `[Verse 1 – EMO:Weight – Female Vocalist – Close-mic]`
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
