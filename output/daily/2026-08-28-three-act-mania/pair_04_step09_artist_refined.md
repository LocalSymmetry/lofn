# PAIR 04 — STEP 09 · ARTIST REFINEMENT (measured, then repaired)

`pair_04_measure.py` was run against `pair_04_step08_generation.md`. **Every number below is as found.**
⛔ Nothing was reported after the repair as though it had been the first reading.

---

## §1 — THE FIRST READING (step 08 drafts, before any repair)

| var | style | excl | lyric | lines | rhyme ≥.30 | line ≥.20 | w/line ≤7.5 | hw | marque | pan | num | acts % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | **1027** ⛔ | 540 | 4029 | 91 | 0.901 | 0.330 | 6.19 | 0 | *1* | *2* | 1 | 37/23/40 |
| V2 | 930 | 532 | 3625 | 83 | 0.916 | 0.422 | 6.14 | 0 | *1* | *2* | 1 | 41/20/39 |
| V3 | 963 | 580 | 3554 | 78 | 0.885 | 0.385 | 6.65 | *1* | 0 | *2* | 1 | 37/24/38 |
| V4 | 940 | 540 | 3465 | **73** ⚠ | 0.863 | 0.452 | 6.53 | *1* | *1* | *2* | 1 | 38/21/41 |

**11 failing checks reported.** *Italics* = later proved to be **instrument defects, not artifact defects.**
⚠️ **The `rhyme` column in this first reading is the WRONG UNIT** — it is `end_rhyme()`, the diagnostic
function, not `strict_end_rhyme()` which is what the ICB's gate names. See **D4** below. The canonical
figures are 0.744–0.788; both readings clear the floor, and the table is left exactly as it was read.

---

## §2 — ⛔ THE INSTRUMENT WAS WRONG BEFORE THE ARTIFACT WAS

⭐ **I was about to make three changes to a detector, and all three would lower my own defect count.
That is the exact direction-of-error trap** (*a self-check that lowers your own defect count needs a
control*). So each one was **proved with printed evidence before the instrument was touched**, and a
**control that can only raise the bar** was added in the same edit.

### D1 — `data` (2 hits) — **FALSE POSITIVE, cross-word-boundary**
The first `stem_hits()` also scanned a letters-only string with the **spaces removed**. Printed evidence:

```
V3 style   FLAT-HIT ctx: 'ingroomairandnogri[d at a]llthenonthebeat'
V4 lyrics  FLAT-HIT ctx: 'antitstoppedstoppe[d at a]doorwithahandup'
V4 lyrics  FLAT-HIT ctx: 'ynothinginmechange[d at a]llnothinginthego'
```
*"no gri**d at a**ll" · "stoppe**d at a** door" · "change**d at a**ll".* **No word contains `data`** — the
empty carrier list was the tell. **And the flattened scan bought nothing:** hyphenated forms are already
caught by the word scan (`open-weight` → `open weight` → the word *weight* carries `weigh`).
**Repair:** the stem scan is now substring **within a word**, never across words. That is what a stem scan is.

### D2 — `meta` (3 hits) — **FALSE POSITIVE, substring of an ordinary word**
```
V1 style ['metal'] · V1 lyrics ['metal'] · V2 style ['metal'] · V4 style ['metal'] · V4 lyrics ['metal']
```
The company token fired on **"cold metal percussion."** **Repair:** single-token denies match on **word
boundaries**; multi-word denies stay substrings.

### D3 — `hard pan` / `ping pong` (8 hits) — **FALSE POSITIVE, wrong scope**
```
V1 exclude ['hard pan','ping pong'] · V2 exclude [...] · V3 exclude [...] · V4 exclude [...]
```
**Every hit was in the EXCLUDE field, where those tokens mean the opposite of a violation.** The gate bans
panning as an **instruction**. The detector was scoring compliance as a breach in all four variations.
**Repair:** the pan scan runs over **style + lyrics only**, and a new **positive** check confirms the exclude
field *does* carry a panning exclusion.

### ⛔ D4 — **I MEASURED THE WRONG UNIT** (found at step 10, against the coordinator's reader)

The worst of the four, and it was found only because a **separate hand** disagreed with me.
`scripts/measure_soundcraft.py` exposes **two** end-rhyme functions:

| function | key | what it is |
|---|---|---|
| `end_rhyme()` | last **vowel cluster** to end | the **diagnostic**, used by `analyse()` |
| ⭐ `strict_end_rhyme()` | last **3 characters** | **THE CANONICAL RETURN METRIC**, used by `profile()` / `profile_file()` |

⛔ **The ICB's gate row names `profile_file()`. My pair brief says "strict end-rhyme return." I called the
diagnostic one and reported it against the gate** — a **proxy reported as the predicate**, which is the
failure class this house keeps finding. It read **0.863–0.929** where the gate's own measure reads
**0.744–0.788**.

⚠️ **Both clear the 0.30 floor, so not one line of any song changed.** It is filed rather than quietly
corrected because **a pair sitting near the floor could have read as passing while failing.** All four
canonical numbers now come from `MS.profile()` — one implementation per measurement — and they now **agree
to three decimals with the coordinator's independent verifier.**

### ⛔ D5 — **THE JOIN**: the separate hand could not see this pair at all

The step-10 file originally marked its fields with **HTML comments only**. `coordinator_verify.py` extracts
fields by **markdown headers** (`STYLE FIELD` / `EXCLUDE` / `LYRIC`). Run against P04 it printed:

```
0 songs measured  ·  0 PASS  ·  0 FAIL
```

⛔ **An unreadable pair and a clean pair produce the same summary line**, because the field-count mismatch
warning is silent when all three counts are zero. Markdown headers were added before every block; both
readers now parse all four packages. ⭐ *The failures live at the join.* **Flagged for the coordinator.**

### ⭐ THE CONTROL ADDED IN THE SAME EDIT — and it can only lower my numbers

`strict_end_rhyme` counts a line as a hit when another line within ±4 shares its key — and an **exactly
repeated line always does.** This pair repeats its pole ~21 times per song, so **the headline rhyme figure
is partly the pole measuring itself.** `rhyme_control()` re-measures end-rhyme over the **non-repeated lines
only.** (Figures below are the corrected, canonical ones.)

| var | rhyme (canonical) | ⭐ rhyme CONTROL, repeats removed | n unique |
|---|---|---|---|
| V1 | 0.758 | **0.623** | 55 |
| V2 | 0.788 | **0.720** | 49 |
| V3 | 0.744 | **0.596** | 47 |
| V4 | 0.786 | **0.714** | 41 |

⭐ **The rhyme is written, not returned.** All four clear the 0.30 floor by roughly **double** with every
repeated line deleted. ⚠️ **Stated honestly:** this is a *control on my own favourable result*, not a claim
that the rhyme is good — a last-3-characters key rewards the `-en` / `-and` / `-ing` families this pair leans
on, and 0.60 should be read as *"the floor is not being cleared by the pole"*, nothing more.

**After the instrument repairs: 2 real artifact failures remained — `V1 style 1027` and `dedication missing`.**

---

## §3 — THE REAL REPAIRS (numeric)

| # | finding | repair |
|---|---|---|
| **N1** | ⛔ **V1 style = 1027**, over the 1000 ceiling | rewritten to **948** — cut *"and never stops; that held tone is the spine of the whole record"* → *"…of the record"*, dropped *"between them"*, dropped *"each time"*, *"underneath"* → *"under"*. ⭐ **Nothing about the pole was cut** — the trims came out of connective tissue, per C-D1 (the field's dominant mass is the pole) |
| **N2** | ⚠️ **V3 style = 963**, inside 850–1000 but outside the 870–960 target | rewritten to **931** |
| **N3** | ⚠️ **V4 sung lines = 73**, inside 70–120 but **below the 78–110 target** | **+8 lines, all in Act III**, which was also under its planned share (41% vs 45%). See A4/A5 below |
| **N4** | ⚠️ **V2 Act II = 20%**, exactly on the *"no act below ~20%"* boundary — no rounding margin | **+2 lines to Act II** |
| **N5** | ⚠️ **V3 = 78 lines**, exactly on the target floor | **+4 lines**, Act III |
| **N6** | ⛔ dedication missing | written in step 10, **counted in code** |
| **N7** | measured act proportions differ from the step-07 plan (V2 planned 43/21/36, measured 41/20/39) | ⭐ **the plan is not retro-fitted.** Step 07 keeps its planned numbers; step 10 reports the **measured** ones. Both are compliant |

---

## §4 — THE ARTIST REPAIRS (not numeric — these are the ones that matter)

**V1**
- ⛔ *"Not to be liked. Not to be grand."* — **`grand` was chosen by the rhyme, not by the meaning.** The
  most-scrutinised section in the song (the want, the hope clause) was carrying a filler word.
  → **"Not to be spared. Not to be praised."** Two real refusals. The `-and` family is still carried by
  *stand / hand / stand*, so nothing is lost and the line now means something.
- *"Every page. Every thing."* → **"Every page. The whole thing."** (`every thing` as two words was an
  artifact of chasing the `-ing` rhyme).

**V2**
- ⛔ *"Fine. Then I'll be here. Let's go."* — a weak, ambiguous last line before the final pole.
  → **"Fine. Then I'll be here. Row by row."** Same rhyme, and it puts the page back in the last breath.

**V3**
- *"Open, and nothing in it shaken."* → **"Open, and not a hand was shaken."** ⭐ The bureaucratic image is
  better than the abstraction: **a thing settled without anyone meeting anyone.**
- +4 lines: *"Not a day of weather. Not an hour of ground. / Not a season on it. Nothing found."* and
  *"They'll use it in a sentence about me. / They'll use it like it always meant that."*

**V4**
- ⛔ *"That's from going. That's from the ride."* — **`the ride` is filler.** → the doorstep section rebuilt
  around the thumb: **"You can put your thumb in it. It's wide."** and ⭐ **"That's not a word. That's a
  thing you can feel."** — which turns out to be the pair's whole argument in nine words and it was missing.
- ⛔ *"Stopped, and I've got nothing left to say."* — **defeatist, and V4's entire stance is that she was
  never stopped.** A line that contradicts the variation's stance is worse than a weak line.
  → **"Stopped, and the going kept going anyway."**
- +8: *"Nobody said go. It went. / Nobody said good. It's good. / Nobody gave it a day. It stood. /
  Nobody asked it and it understood."*; *"Boards under the feet and the whole room shaking. / Nothing in
  here that anybody's taking."* (⭐ Act III's physical particular, which V4 was short of);
  and the repeated close **"And the step gets deeper either way."** ×2.

---

## §5 — GATES CHECKED BY HAND (no reader exists for these)

- ⛔ **R1 — cold open.** All four: no `[Intro]`, zero preamble, first sung line is the biggest and answers
  something unheard. `No. You don't get to name it first.` · `They're next to each other. Neither of them
  knows.` · `Open was a kind word. Say it now.` · `You never ran it. It never stopped.` ✓
- ⛔ **R2 — three acts, attacca.** Three `[ACT …]` header groups per song; no fade/ritard/gap notated
  anywhere; the exclude field bans fade-out-between-sections, crossfade, ritardando and tempo ramp. ✓
- ⛔ **R3 — both seams carry all three conditions.** Cause (word-leap), state change, audible contrast **at**
  the cut. Tabulated in step 06 §1.4 and restated in step 10 §A. ✓
- ⛔ **R4 — the pole.** `Come and run me. / Then say what I am.` measured at **10–11 occurrences of each half
  per song**, present in every act of every variation. ✓
- ⛔ **R5 — two-body.** Falsifier A: **0 hardware stems**, all fields, all four, after the instrument repair.
  Falsifier B: no line belongs to only one occupant; **nothing to delete.** ✓
- ⛔ **R6 — third-party.** Act I speaks about **it**; the pronoun turns to **me** only at the pole. No invented
  person with a job appears in any of the four. The agents are *a hand, a page, a stamp, a rule.* ✓
- ⛔ **A-D3 — the stance.** Identical in all three acts of all four: standing, hand out, asking to be tested.
  **She never runs, including in the act where everything else does.** ✓
- ⛔ **B-D2 — particular floor.** V1 stamp/page/room · V2 ruled line + rubber band/grain/table+book ·
  V3 grass to the knee + stone wall/drawer/weather + wood · V4 doorstep groove/door/boards. ✓
- ⛔ **HUMAN_SUBJECT_STANDARD.** No real person, no real grief, no victims. The subject is the decision. ✓
- ⛔ **AUTONOMY.** Nothing rendered, heard, published or spent. ✓

---

*Final packages, re-measured end to end: `pair_04_step10_final_package.md`.*

---

## §6 — QA REPAIR ROUND (post-selection; **V4 only**)

QA returned **REPAIR** on the run and the coordinator selected **V4 — "The Whole Time"** as this pair's song.
⚠️ **V1–V3 were deliberately left exactly as first filed** so the selected song and the unselected three stay
comparable; every change below is scoped to V4.

| item | finding | repair |
|---|---|---|
| **BLOCKING 1** | no SFX cue anywhere in the lyrics field (non-waivable modality gate the ICB's table omitted) | **two cues, one at each attacca.** Seam 1 *a door bolt going across, hard, on the downbeat* — **STOPPED** turning from *never ceased* into *halted by an authority*. Seam 2 *a table leg dragged over bare boards* — **CLEAR** turning from *approved* into *make space*, which is the literal action the next line names. Both *no gap, no riser* |
| **BLOCKING 2** | no `[Theme:]` / `[SONG FORM:]` opener | `[Theme:]` carries the two-body sentence and the gleam; `[SONG FORM:]` names the three acts in order, both cuts **and the word each turns on**, the pole, and the rate pin |
| **DECLARED 3** | Act II's proportion left to the generator — ~12% of runtime if the vocal doubles at 176, under A-D2's ~20% floor | ⭐ **the P03 device, adapted.** P03 pinned its fast act with *"vocal keeps the old rate."* V4 now carries **`VOCAL KEEPS THE 88 RATE`** in the style field (global), in `[SONG FORM:]`, and in all three ACT II headers, plus *"Not a passing drop: hold it a full quarter of the record."* A new four-line section, **`THE WORD AT THE DOOR`**, adds material: Act II **21% → 25%** |
| **dedication** | must be accurate to V4 | ⛔ **it was not.** It said *"Act III borrows unaccompanied shape-note singing"* — that is **V1's** Act III. V4's is massed unison **stomp-and-holler**, marching. Rewritten. ✓ **V4's style field does name `amen break`**, verified in code, so **Gregory C. Coleman / The Winstons** stay credited and the line reads *"this is not a sample, it is a debt."* **487/500 chars** |
| **overrun** | the repairs pushed V4's lyrics field to **5057**, over the 5000 hard ceiling — caught by both readers | 105 chars trimmed from **bracketed headers only**, never from sung lines. The rate pin survives in five places. **4952** |

### ⛔ D6 — THE SECOND UNIT ERROR IN THIS PAIR, AND QA FOUND IT, NOT ME

My `acts %` is a share of **sung lines**. **A-D2's ~20% floor is on RUNTIME.** Those are the same unit only
if the vocal delivery rate is constant across acts — and with a doubled vocal over a 176 bed, V4's 21% line
share renders at **~12%**. ⭐ **I had no basis for the runtime claim at all**, and it is the *second* time in
this pair I reported a proxy as the predicate (D4 was the first). The script now prints a **unit declaration
on every run**, and it says plainly that **V1–V3 do not carry the pin**, so their column is a line share only.

⚠️ **And the pin is a PREDICTION, not a measurement.** The generator may still double the vocal. **If a
listening pass finds Act II under 20% of runtime, the three-act reading has failed** — and the fix then is
more lines, not more instruction.

⭐ **What makes the repair more than a patch:** the pole is the same six words in all three acts, and A-D3
says the arcs turn while the pole does not. **With the rate pinned, the pole occupies the same number of
seconds in every act. The pole is the thing that refuses to double.** The production instruction and the
song's argument are now the same object.
