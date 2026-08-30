# PAIR 01 · STEP 09 — ARTIST REFINEMENT (the measured repair record)
### `2026-08-28-three-act-mania` · **LOFN-PRIME** · ACCESSIBLE · NEWS · INDIGNATION

The step-08 drafts were written **blind to the numbers** and then scored with
`pair_01_measure.py` (which imports the four canonical return metrics from
`scripts/measure_soundcraft.py` rather than re-implementing them). This file is what the
instrument said and what I did about it. **Nothing was eyeballed and nothing was repaired
before it was measured.**

---

## 0. ⚠️ THE FIRST DEFECT WAS IN THE INSTRUMENT, NOT THE SONG

**The gate fired on V1 with `interface nouns: 1 — ['approve']`.**

My self-imposed interface-noun list was written as bare stems and prefix-scanned, so the stem
**`app`** matched the word **`approve`** — which is the hinge of the entire first seam
(*"watch me not move"* → ***"Move to approve."***).

> ⭐ **Had I trusted the gate, I would have deleted the best line in the pair to satisfy an
> instrument that was wrong.** This is the `a-broken-gate-deforms-the-artifact` failure class,
> live, on the first run of the instrument I had just written.

**Repair:** `INTERFACE_STEMS` (prefix scan) → `INTERFACE_PATTERNS` (word-boundary regex), with the
cause recorded in a comment in the file itself. ⛔ **The ICB's own `HARDWARE_STEMS` list is
untouched and is still stem-scanned exactly as R5 specifies** — the ICB gate is the ICB gate; only
my private extension was broken. Re-run: **0 across all four.**

---

## 1. THE SCORECARD ON THE RAW DRAFTS (after the instrument repair)

| gate | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| style chars | 962 ⚠️ | 962 ⚠️ | 914 ✅ | 942 ✅ |
| **lyric chars ≤5000** | **5197 ❌** | 4555 ✅ | 4158 ✅ | 4006 ✅ |
| sung lines | 98 ✅ | 90 ✅ | 90 ✅ | 90 ✅ |
| end rhyme ≥0.30 | 0.429 ✅ | 0.300 ⚠️ | 0.511 ✅ | 0.567 ✅ |
| **line return ≥0.20** | 0.357 ✅ | **0.178 ❌** | 0.422 ✅ | 0.444 ✅ |
| words/line ≤7.5 | 5.78 ✅ | 5.99 ✅ | 5.47 ✅ | 5.31 ✅ |
| **act proxy, min ≥20%** | 41/32/28 ✅ | 25/49/26 ✅ | 43/34/24 ✅ | **24/19/57 ❌** |
| hardware nouns = 0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| **pan tokens = 0** | **2 ❌** | **2 ❌** | **2 ❌** | **2 ❌** |
| marque tokens = 0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |
| digits in sung = 0 | 0 ✅ | 0 ✅ | 0 ✅ | 0 ✅ |

**Four real failure classes. Every one of them repaired below.**

---

## 2. REPAIR A — ⛔ PAN TOKENS IN ALL FOUR EXCLUDE FIELDS

**Found:** every exclude field contained `hard panned effects`. The ICB gate is
*hard-pan / non-musical panning tokens = **0** in every pasted field* — and it does not have an
exemption for "but I was forbidding it."

⭐ **And the repair is not a workaround, it is the correct spec.** Measured on this house's own
renders on 2026-08-27: two songs that both requested `Hard_Pan_Right` came back at **−0.02 dB** and
**+0.13 dB** overall balance, **0 of 52 and 0 of 56 windows** above +3 dB. **A spatial instruction
is a request the generator overrules** — so a pan token in an exclude field buys nothing in either
direction and costs a gate.

**Change, all four:** `hard panned effects` → `stereo widening on the lead vocal`.
*(Which is also the better exclusion: `SCR-02 INWARD COSMOS` says our loudest moments are our
narrowest, so the thing worth forbidding is the widening, not the panning.)*

---

## 3. REPAIR B — V1 LYRIC FIELD OVER THE 5000-CHAR CEILING (5197)

⭐ **Diagnosis: not one word of the song is over budget. The PROSE is.** V1 carried 11 bracketed
section headers plus a `[Theme:]` and a `[FORM:]` block — and C-D1 *requires* the acts to live in
those local headers, so they cannot simply be deleted.

**Cut 266 chars from apparatus, zero from sung lines** (all four counts measured, not estimated):
- `[Theme:]` **264 → 165** (kept the stair, the light and the refusal; dropped the restatement).
- `[FORM:]` **311 → 160** (dropped the pole's full description — it is already the style field's entire
  dominant mass, and repeating it inside the lyric field spends the ceiling twice).
- The second `[ACT I - Chorus …]` header **−16**, now
  `[ACT I - Chorus - Punk & Hardcore, 142 - EMO:Indignation, Fury - full band, gang answer doubled]`.

99 + 151 + 16 = 266. **Result: 5197 → 4931.** ⛔ **No sung line was cut to make a ceiling.** If a lyric
ever has to lose something to fit, the apparatus goes first, every time.

---

## 4. REPAIR C — V2 LINE RETURN UNDER THE FLOOR (0.178)

**The draft predicted this failure in its own KNOWN GAPS** (*"only one chorus in the whole song"*),
and the instrument confirmed it. ⭐ **This is a craft failure that the metric merely noticed:** in
the ACCESSIBLE arm, a song whose chorus never comes back is under-returned whatever the number says.
`L21 THE RETURN` — **removal is a debt.**

**Repair, and it is the best change made in this pair:**

> **The parade sings her chorus back at her, verbatim, in Act III — with the meaning inverted.**

The eight lines she shouts in Act I as an *accusation* (*"Nothing's been taken. / Nothing's been
banned."*) come back in the mouths of the people who moved, as a *hymn*, in the second-line brass
act. **Same eight lines. Opposite meaning.** It is the pole's inversion device applied to the chorus
instead of the four words, and it costs nothing but eight lines.

Act III: 30 → 38 lines. Song: 90 → 98. **Line return 0.178 → 0.286. End rhyme 0.300 → 0.316**
(the marginal rhyme number was the same defect wearing a different hat).
Act proxy 25/49/26 → **23/45/31** — Act II is still the longest act, which is why V2 exists.

⚠️ **V2 still has the thinnest margins in the pair** (0.316 against a 0.30 floor; 0.286 against 0.20), and I
am leaving them there. The middle act is built out of minuted procedural language, which does not rhyme and
does not repeat — **buying rhyme in the meeting act would make the room sound like a song, and the room must
not sound like a song.** Over the floor, on purpose, and declared.

---

## 5. REPAIR D — V4 ACT II UNDER THE 20% RUNTIME FLOOR (proxy 19%)

A-D2: no act below ~20% of runtime. V4's proxy read **24/19/57** — the doom-gospel chorale at
66 BPM is so slow that 34 lines swallowed the song.

⚠️ **Honest note: the proxy under-states slow acts by construction** (a slow act holds more bar for
the same word count), so the *true* runtime share of Act II is probably higher than 19%. **I repaired
to the instrument anyway.** Arguing a failing number down using a limitation I knew about in advance
is exactly the direction-of-error the pre-registration standard exists to stop, and I have made that
specific mistake before.

**Change:** four lines moved from Act III into Act II (30/26/34 → **30/30/30**), and the lines chosen
were the ones Act II actually wanted — the demand for the hand:

```
Show me the hand.
Show me the hand.
Nobody in that room had to be a monster.
A hand is not weather.
```

⭐ *"A hand is not weather"* is now the thesis of the whole pair in four words, and it is answering
V1's *"Say it like it's weather."* Act proxy → **24/23/53.** Act III is still the longest, which is
the point of V4: **the antagonist's win is the biggest act in the song.**

---

## 6. REPAIR E — STYLE FIELDS ABOVE THE TARGET BAND (V1, V2 both 962)

Inside the hard range (850–1000) but above the 870–960 target. Trimmed to **945** (V1) and **924** (V2) by
removing two hedges and one duplicated clause. No content lost; the pole still owns the field's dominant
mass in all four. Final style lengths, measured: **945 · 924 · 914 · 942** — all inside the target band.

---

## 7. THE HAND ADJUDICATIONS THE INSTRUMENT ASKED FOR

The scan **reports** every number-word and every mid-line capital rather than guessing, because a
numeric *fact* is not the same object as a number *word*. Each hit ruled here, in writing:

| song | hit | ruling |
|---|---|---|
| V1 | *"The ground did not move in twenty-four hours."* | ⭐ **THE ONE NUMERIC FACT** (F04). At the hinge, as words, and **responded to** — *"And everybody moved."* ✅ |
| V1 | *"Third thing on the list, before the break."* | **ordinal, not a quantity.** It is a position in an agenda, and it is the song's best particular. Not a fact. ✅ |
| V2 | *"…lost its chair in March."* (mid-line capital) | a month. Not an artist, company or marque. ✅ |
| V2 | — | **zero numeric facts.** The hinge is F22, non-numeric: the moon lit both halls the same. ✅ |
| V3 | *"Not once. Not anywhere."* | **idiom, not a quantity.** ✅ zero numeric facts; hinge is F04 non-numeric. |
| V4 | *"The moon was a hundred per cent of itself all night."* | ⭐ **THE ONE NUMERIC FACT** (F22). As words, at the hinge, **responded to** — *"It lit the near way and the far way the same… And everybody moved."* ✅ |

**Sung numeric facts per song: V1 = 1 · V2 = 0 · V3 = 0 · V4 = 1.** All ≤ 1. ✅

---

## 8. WHAT I DID **NOT** REPAIR, AND WHY

- **V1's Act I at 41% of proxy runtime.** Unequal is the rule (A-D2); this is the design.
- **V3's Act II at 16 sung lines.** It is a canon over a held tone at 66 BPM — the *fewest words*
  and, in real time, one of the longest stretches. Proxy says 34%. Left alone.
- **The two identical choruses in V1.** ⭐ *Exact chorus repetition needs no defence* (L21).
- **`allit_per_100w` (8.7–12.8).** Reported, ungated by the ICB. Not tuned toward a number nobody
  asked for.
- ⛔ **Nothing was rendered, heard, published or spent** (`vault/AUTONOMY.md`). Every claim in
  steps 08–10 is a **text** claim, and the render prediction in step 10 stays a prediction.

---

## 9. ⚠️ QA REPAIR PASS — TWO BLOCKING DEFECTS, BOTH MINE, BOTH REAL

QA returned **REPAIR** on the run. **V4 "NOTHING WAS TAKEN" is the selected song**, so both fixes were scoped
to V4 first and then applied across the pair where there was headroom.

### BLOCKING 1 — no SFX cue in any lyrics field

⭐ **The root cause is the one worth keeping: I built my instrument from the ICB's house-numbers table, and
that table does not list the SFX gate.** So the gate did not exist in my script, and my package reported
**all-green while missing a non-waivable modality requirement.** **A green board from a gate that isn't
watching is worse than a red one** — it converts a missing check into a positive claim. That is the same
class as §0, one level up: §0 was an instrument that was *wrong*; this was an instrument that was
*incomplete*, and incompleteness does not announce itself.

**Convention derived from this repo's own canonical sources** — ⛔ **not** from P05/P06, whose files were not
opened; pair isolation held. The house has **two readers that disagree on the form**, so both are satisfied:

| reader | wants |
|---|---|
| `.claude/skills/lofn-music/SKILL.md:107` | *"≥1 standalone `*SFX*` cue"* |
| `scripts/check_generation.py:45` | the literal string `SFX:` |

**Cues placed AT THE SEAMS**, which is also where they do the most work — this is the **ACCESSIBLE** arm, and
an SFX at the attacca cut makes the door **physical**, not just audible. Each cue is built from that
variation's own object, so it is texture the song already owns:

| song | seam 1 | seam 2 |
|---|---|---|
| **V4** ⭐ | `*SFX: a window sash drops in its frame, dry, no tail, on the beat*` | `*SFX: a room's worth of chairs pushed back at once, then only the held tone*` |
| V1 | `*SFX: a light switch thrown once, and nothing*` | **none — no headroom** (see below) |
| V2 | `*SFX: a chair leg dragged on tile, once*` | `*SFX: a door pushed open onto a street, brass already going*` |
| V3 | `*SFX: a hand leaving a cold metal rail, one bar of nothing after it*` | `*SFX: a brick set down on concrete, and a fire door beginning to close*` |

⭐ **V4's second cue is the best thing added in this pass:** *a room's worth of chairs pushed back at once* is
the meeting standing up and **becoming the congregation** — the seam's meaning made into a sound.
⭐ **V3's second is the whole song in one object:** the brick set down, and the fire door starting to close.

⚠️ **V1 gets one, and the reason is arithmetic, not taste.** At 4931 chars it had **69** of headroom; one cue
took it to **4978**, leaving 22. Recorded as a cost.

⚠️ **Honest limit:** an SFX cue is a **request**, and `L22 THE GRAIN LAW` says specs that fight the generator
get smoothed. We have **no measured prior** for SFX cue survival through our door. Nothing here is rendered,
so this is a text claim and nothing more.

### BLOCKING 2 — the dedication was written for V1 and was shipping on V4

**Correct, and it is the sharper of the two defects.** The V1 dedication credits *"the stomp-and-handclap
act"* — **V4 has no handclaps.** V4's Act III is massed unison voices, distorted organ at full drawbars and
floor tom, with no strings. ⭐ **A dedication that credits an instrument the record does not contain is not a
debt paid, it is a debt MISDIRECTED — which is worse than silence, because it looks discharged.**

**Written fresh for V4, measured in code: 495 / 500.** Kept at priority (1): the held-tone debt to Sarah
Davachi's *The Will of Tongues*, **read about and never heard**, named in the dedication and **nowhere in any
pasted field.** Priority (2)/(3) replaced with what V4 actually contains: **the Black American church
organists** behind the drawbar organ that ends the record, and the **first-wave punk rhythm players** behind
Act II's downstrokes. Priority (4) honest AI authorship, stated plainly.

⛔ **Break clause checked in code, not by eye.** A `BREAK_NAMES` table (`amen` · `funky drummer` · `apache` ·
`cold sweat` · `think` · `impeach the president` · `assembly line` · `synthetic substitution`) is now scanned
across every pasted field of every variation. **Result: 0, NONE, on all four.** No break is named, so
**Gregory C. Coleman / The Winstons is not owed here and is not claimed** — and the table stays in the script
so the clause fires automatically if a future revision names one.

⭐ **The generalisable lesson: a dedication is written for a RECORD, not for a pair.** Four variations sharing
a pole do not share their materials, so one dedication per pair is a category error — and it survives review
precisely because it reads as generous.

### RE-MEASURED — every gate green on all four, zero FAIL lines

| gate | V1 | V2 | V3 | V4 ⭐ |
|---|---|---|---|---|
| style chars (850–1000, ends as sentence) | 945 ✅ | 924 ✅ | 914 ✅ | **942 ✅** |
| lyric chars ≤5000 | 4978 ✅ | 4935 ✅ | 4202 ✅ | **3998 ✅** |
| sung lines 70–120 | 98 ✅ | 98 ✅ | 90 ✅ | **90 ✅** |
| end rhyme ≥0.30 | 0.429 ✅ | 0.316 ✅ | 0.511 ✅ | **0.567 ✅** |
| line return ≥0.20 | 0.357 ✅ | 0.286 ✅ | 0.422 ✅ | **0.444 ✅** |
| words/line ≤7.5 | 5.78 ✅ | 5.90 ✅ | 5.47 ✅ | **5.23 ✅** |
| hardware nouns (stem-scan) = 0 | 0 ✅ | 0 ✅ | 0 ✅ | **0 ✅** |
| marque tokens = 0 | 0 ✅ | 0 ✅ | 0 ✅ | **0 ✅** |
| sung numeric facts ≤1, as words | 1 ✅ | 0 ✅ | 0 ✅ | **1 ✅** |
| **SFX cues ≥1** | 1 ✅ | 2 ✅ | 2 ✅ | **2 ✅** |
| **named breaks** | 0 NONE | 0 NONE | 0 NONE | **0 NONE** |
| dedication ≤500 | 491 (D2) ✅ | — | — | **495 (D1, ships) ✅** |

⛔ Only `pair_01_*` files touched. No `git`. Nothing rendered, heard, published or spent.
