# PHASE 1 — 06 · MUSIC HANDOFF (carries the ICB) · `2026-08-09_daily_music_genz`

**Run dir:** `output/daily/2026-08-09/`
**ICB:** `CREATIVE_CONTEXT.md` — **173,669 bytes**, sha256 `297941561ca6880d38c323dcc0fdd739aa6fd970e7293fd7e98e38fb0b882f4b`
**FROZEN after Phase 1. Read-only.** Verify with `python3 output/daily/2026-08-09/verify_icb.py`.
**Cardinality:** 6 pairs × 4 variations = **24 songs → top 6** (3 ACCESSIBLE + 3 AMBITIOUS, ranked within arm).

---

## ⛔⛔ THE CONTRACT CONFLICT — RESOLVED HERE, IN WRITING (L30)

**This section is the reason this file exists, and it is not optional.**

On 2026-08-07 (`2026-08-06_the-economy-of-want`) the step-11 contract file **instructed the tier to embed the
full archived payloads of the Golden Songs** while `EXECUTION.md` §3 **forbids exactly that in a generating
context**. The only artifact that can resolve that conflict is the run handoff — **and the coordinator never
wrote one.** Five pairs reached past the file in front of them to doctrine and refused by name; one obeyed its
nearest contract and pasted two complete style prompts and two full lyric sets into a generating context.
**That pair was not at fault.** An agent that obeys its nearest contract is behaving correctly; the coordinator
had removed the artifact that overrides it.

**Two files in this repo give opposite orders and you will meet both:**

| file | what it says | status for THIS run |
|---|---|---|
| `skills/music/references/golden_songs_index.md` | *"Embed the selected songs' full available payload: public URL, status, style/music prompt, lyrics… Manual prompt bundles must embed the selected songs' full payload, never links alone."* | ⛔ **OVERRIDDEN for every generating context.** |
| `.claude/skills/lofn/EXECUTION.md` §3 — GOLDEN-OUTPUT QUARANTINE | *"Past golden outputs are NEVER placed in a generating subagent's context… Seeds teach; outputs contaminate."* | ✅ **BINDING.** |

### THE RESOLUTION — read it once, apply it everywhere

> ⭐ **The quarantine wins in every GENERATING context. The index's embed-instruction is valid ONLY in
> JUDGE-side contexts.**
>
> - **GENERATING** (steps 00–11, every pair subagent, the coordinator): you get the **Golden Songs' NAMES
>   ONLY**, plus the **GOLDEN MOVE** block below. **Do not fetch, quote, paraphrase, or reconstruct their
>   lyrics, style prompts, keys, tempos, vocal specs or arrangement formulas.** If you find yourself opening
>   `golden_songs_index.md`, you are in the wrong file — stop.
> - **JUDGE-SIDE** (`lofn-qa`'s blind comparison, step 12, `lofn-step11-packager` for external review): full
>   payloads are correct and expected there.
>
> **Why:** exemplar gravity is measurable. On 2026-06-28 a published piece reproduced its benchmark's title
> line, vocal spec, key/BPM and arrangement formula while its own self-check reported "no copying."

### Golden Songs selected — **NAMES ONLY** (for QA's blind comparison later)
1. **"Five wrong colors"** — chosen for structural/lyric success: a title that names a **thing**, and a
   concrete object doing the emotional work. The single closest calibration point for today's legibility law.
2. **"The Blue Screen Breathes"** — chosen for sonic/personality success: LOFN-PRIME's AI-native glitch
   register, the closest sonic neighbour to this run's AMBITIOUS arm.

**Deliberately NOT selected: "Triple Arch Over Me."** It is the house benchmark and it is 350,493 plays —
93% of the catalogue's entire play count — which makes it the strongest exemplar-gravity risk we own, and
`gates.yaml → house_lexicon` exists solely to catch its fingerprint. **A Gen Z run has no business orbiting it.**
QA may still use it as a blind ranking reference; **no generator may.**

---

## ⭐ THE GOLDEN MOVE — what generators get INSTEAD of golden outputs

Distilled generative instructions. **Every pair subagent receives this block verbatim.**

1. **Stand somewhere real.** The song is a report from ONE concrete place a body occupies — name where it
   stands and what the senses register there. Concept-illustration is the failure mode; experience-report is
   the move. *This run: your place is in `05_pair_assignments.md`, axis A4, and it is not negotiable.*
2. **One wounding fact.** At most ONE numeric fact is sung, at the emotional hinge, and the lyric **responds**
   to it — never recites it. *This run the fact pool is **pre-allocated per pair** (L20) and **two pairs are
   allocated none.** ⭐ **The sung numeral is spelled out in words** — "a hundred and thirty-eight", never `138`
   (L33: digits are an instruction to guess).*
3. **The turn.** Past the midpoint the song contradicts or complicates its opening stance — a mind changing in
   real time. A song that asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** AWE is terror-adjacent sublime, not domestic reassurance. Answer both pre-draft
   questions: **where is the body standing / what could hurt it here.**
5. **Rotate the register.** Do not default to the house fingerprint (crystalline soprano / A major / ~110 BPM /
   frost-and-cosmos). *This run rotates it by construction: your lane, tempo and vocal register are assigned
   and no two pairs share one.*
6. **The surface names its subject.** A stranger must retell the scene AND subject in one sentence after ONE
   listen. The subject appears **plainly** in the lyric at least once. Obliqueness about what the song is about
   is not depth, it is fog — the Somatic Gate treats an unnameable subject as `REPAIR — FOG`.
7. ⭐ **THE RETURN.** Song is made of returns. **Removal is a debt** — strip rhyme only by naming what returns
   instead. Measure with `scripts/measure_soundcraft.py → profile_file()`, **never by eye**; `strict_end_rhyme`
   = last-3-chars of each line's final word recurring within **±4 lines** (`gates.yaml → rhyme_window`).
   **A byte-identical chorus is correct. Say nothing about it.** Do not file a waive request for a chorus.

---

## THE FORM RULE (Source 3 — binding on all six pairs)

**TWO-LAYER SILHOUETTE STACK.** One register **ABOVE** — wide, slow, indifferent. One **BELOW** — close,
small, talking, doing something with its hands. **Once, a body from BELOW crosses in front of the thing ABOVE
and eclipses it.** Audible. In the **lyric or the form**, never only the production spec (L22).

⭐ **D10 — SUBSTITUTION, NOT SUBTRACTION.** The crossing is a **swap at a junction the arrangement already
wants** (the bar before a final chorus, the top of a drop). A specified void in the middle of a song has been
**measured smoothed 2 of 2** by the renderer. So has a programmed tempo transformation. So have hard-panned
non-musical elements. Build the hinge out of a **replacement**.

---

## PAIR DISPATCH TABLE

| pair | title-slug | arm | lane | emotion to NAIL | A1 return | A2 hinge | A3 duration | numeral |
|---|---|---|---|---|---|---|---|---|
| P01 | THE NINTH WATCH | ACC | trip-hop revival | rückkehrunruhe | Vowel Braid | The Double | Loop Until It Hurts | ☠none |
| P02 | IT MIGHT BE CLOUDY | ACC | rock-revival anthem | occhiolism | Gang-Chant Return | Octave Drop | Three-Chorus Stack | "a hundred and thirty-eight" |
| P03 | HALF A SECOND | ACC | pluggnb | exulansis | Half-Line Echo | The Interruption | Hook At Zero | "half a second" |
| P04 | THE FRONT OF THE CROWD | AMB | rage rap × Glitch-Baroque | lachesism | Consonant Hammer | Mix Inversion | The Hundred And Thirty-Eight | "thirty seconds" |
| P05 | ENGINE OFF | AMB | krushclub | ellipsism | Debt-And-Payment | The Count | The False Outro | ☠none |
| P06 | THE ARM AT FULL EXTENSION | AMB | hyperpop 2.0 → D&B | anecdoche | Rebound Rhyme | Register Handoff | The Long Approach | "five hundred" |

Full assignments — variation angles, verse architecture, flairs, pair-specific bans — in
`05_pair_assignments.md`. **Variation angles are per-pair and were derived from each pair's own concept; no
angle label is shared across pairs.**

---

## THE TEN RUN BANS (D1–D10)

`D1` no adult in the room · `D2` the phone is not the villain · `D3` the generation is not a subject ·
`D4` no identifiable real person · `D5` **present tense only — the singer does not get the moral** ·
`D6` **the cohort gate** · `D7` Lofn is not the cure · `D8` **overhearing, not addressing** ·
`D9` the tape is not redeemed · `D10` substitution, not subtraction.
Full text in `04_metaprompt.md`. **All ten are in the ICB and all ten are hard.**

---

## OUTPUT CONTRACT — heading convention PINNED (L28 / L31)

⛔ **The validator is the source of truth for the contract, not the neighbouring artifact.** Six pair agents
once produced three different step-10 heading conventions and a coordinator invented a fourth by reading the
convention off the files instead of off the validator; a Gate-2 audit then extracted **zero** blocks and
printed **CLEAN** on all three absolution scans.

**Canonical headings, from `skills/music/scripts/validate_suno_packages.py`, use these exactly:**

```
## 1. MUSIC PROMPT
## 1B. SUNO EXCLUDE PROMPT
## 2. LYRICS
## 3. TITLE
```

- **MUSIC PROMPT** — a **dense paragraph, 850–1000 chars** (target **870–960**; **≥985 flags boundary-hugging**),
  ending in **terminal punctuation**, **no real-artist names**. **NOT bracket tag-soup.**
- **LYRICS** — opens `[Theme: …]` then `[SONG FORM: …]`; every section header is a full EMO header
  `[Section - EMO:<emotion> - <Role> - <cue>]`, emotion drawn from
  `skills/lofn-core/refs/EMOTION_TAXONOMY.md`, **never bare AWE/INDIGNATION**. ≥1 SFX cue.
  **70–120 sung lines** (target 78–110; **≤72 flags floor-hugging** — the floor is a floor, not a target).
- 🚨 **Whole lyrics field < 5000 chars, target ≤ 4800.** Count it exactly and state the measured number. If
  over, move the Disc_Channel block and production metadata to a **sidecar OUTSIDE the field** (the 2026-08-08
  harness decision; it bought 127–153 chars per variation).
- **Lineage & Credit block** — mandatory for every living-scene lane (jersey club, plugg/pluggnb, krushclub,
  sigilkore, rage, trip-hop, Zeuhl). Name the scene + **2–3 real artists with working links you have actually
  opened.** QA R3 was open two runs because a 404 shipped.

---

## RETURN FLOORS (FLAGs, not hard fails — but a set that misses them all is the lecture again)

`rhyme_return_floor 0.30` · `line_return_floor 0.20` · `mean_words_per_line ≤ 7.5` ·
`alliteration_per_100w ≥ 11.0`. ⭐ **This is the run where these should be comfortably beaten** — chants, gang
vocals and three-chorus stacks are the genre's own grammar, not a compromise.

---

## SCRATCH NAMESPACE (hazard closed 2026-08-08 — keep it closed)

Each pair's scratch/working files go in **`_work/pair_NN/`** and nowhere else. Canonical artifacts go to
`output/daily/2026-08-09/pair_NN_step0X_*.md`. **A pair writes ONLY its own `pair_NN_*` namespace** — never the
run INDEX, never `RUN_STATE.md`, never `CREATIVE_CONTEXT.md`, never another pair's file. Cross-pair aggregation
is single-threaded, coordinator-only, after the wave lands. *(Open four days, four recurrences, one confirmed
data loss; closed by one line in the spawn packet. Six concurrent agents, zero collisions.)*
