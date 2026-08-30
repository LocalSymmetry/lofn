# Lofn QA Report — `2026-08-28-three-act-mania` (music)

**Auditor:** `lofn-qa`, clean-context adversarial judge. I did not write any artifact in this run.
**Scope:** the six shipped songs in `SUNO_PACKAGES.md`, their six `step10` sources, `DEDICATIONS.md`,
the ICB, and the coordinator's own instruments.
**Method:** every number below was **re-measured by me** in `qa_measure.py` / `qa_marque.py` /
`qa_format.py`, importing `scripts/measure_soundcraft.py`'s own definitions. I did not read the
coordinator's numbers until after mine existed. They agree (see §2.1) — which is a finding, not a
formality.

⛔ **NOTHING IN THIS RUN HAS BEEN RENDERED.** Every audio claim in this report is marked
**UNVERIFIABLE**. See §9 for the full list of what I could not check.

---

## Verdicts

| | |
|---|---|
| **Pipeline Integrity** | **REPAIR REQUIRED** — the no-skip rule PASSES (06/07/08/09/10 present for all six pairs; the run is **canonical**). But the 24→6 selection stage has **no artifact of any kind**. |
| **Package** | **REPAIR REQUIRED** — three blocking modality/dedication defects, all cheap text fixes, none touching a sung line. |
| **Human-Subject** | ✅ **CLEAR** — and deliberately so. See §7. |
| **Somatic Gate (3 Hyper-Skeptics)** | **2 YES / 1 NO** — not blocked. The single NO is against the run's *shared premise*, not any song, and carries a counter-move. See §8. |
| **OVERALL** | ### ⚠️ **REPAIR** |

**Why REPAIR and not HOLD or FAIL:** the songs themselves are strong and I could not break them on the
hard creative gates — R1, R3 (all three conditions, both seams, all six songs), R4, R5 and R6 survive
adversarial reading. What fails is **apparatus**: a missing SFX cue, two missing lyric-field openers,
one dedication attached to the wrong song, and a selection decision with no record. Three text edits and
one written page and this set is ship-ready *as text*. The audio verdict does not exist yet and cannot.

---

## 1. Pipeline integrity & ICB

| check | result |
|---|---|
| per-pair steps 06·07·08·09·10 exist as separate files | ✅ **30/30**. No `steps_06_10` rollup. **The no-skip rule does not fire — this run is CANONICAL.** |
| coordinator steps | `00_research_brief` · `core_seed` · `03_panel_debate` · `05_pair_assignments` · `CREATIVE_CONTEXT`. No files at 01 / 02 / 04. Naming differs from the SKILL's `step00…step05`; the editorial spine is present. **ADVISORY.** |
| panel: 18 seats | ✅ **18 defined** (A1–A6, B1–B6, C1–C6), **3 Hyper-Skeptics** (A6 Medley Prosecutor, B6 Abstraction Auditor, C6 Grain Auditor). All three spoke and two changed the rules (R3's third condition; B-D1/B-D2). |
| the `(after ` == 18 tripwire | ⚠️ **reads 14 unique names / 21 turns.** Not a defect: **four seats are role-anchored, not person-anchored** (C1 Prompt Mechanic, C4 Sustain Specialist, C5 Streaming Realist, C6 Grain Auditor). 14 + 4 = 18. **The SKILL's tripwire would false-fail this panel.** Flagging the instrument, not the artifact. |
| ICB verbatim prefix in artifacts | ⚠️ **verifiable for P03 only.** P03 reproduces `CREATIVE_CONTEXT.md` byte-identically at byte 0 of all five of its files. P01/P02/P04/P05/P06 do not contain it at all. These are *outputs*, not prompts, so this is **a reader gap, not proof of thread loss** — but it means the cheap ICB tripwire cannot be run on 5 of 6 pairs from disk. **DECLARED.** |
| run lock | ✅ held by `claude:4926de8f-…`, single controller, heartbeat current. |
| provenance of the shipped fields | ✅ **18/18 verified independently by me** — each shipped field is byte-identical to the correct variation index in its pair's `step10` file. The coordinator's round-trip audit claim holds under a second reader. |
| portfolio cross-similarity | ✅ **0 breaches.** max across-pair lyric Jaccard **0.279** (ceil 0.42), 5-gram **0.003** (ceil 0.18), style **0.302** (ceil 0.58). Comfortable margins. |

---

## 2. Structured evidence block — the house numbers, re-measured

Unit declaration: **one row = one shipped song.** `sung lines` = `measure_soundcraft.sung()` (bracket
headers and `*…*` cues excluded). `endRhy` = `strict_end_rhyme` (last 3 chars of the final word recurring
within ±4 lines). `lineRet` = exact repeated-line ratio, choruses counting. `w/line` = words ÷ sung lines.
Char counts are of the **pasted field**, not the markdown around it.

| # | song | style | ends `.` | exclude | lyrics | sung | endRhy ≥.30 | lineRet ≥.20 | w/line ≤7.5 | allit |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | NOTHING WAS TAKEN `P01V4` | 942 | ✅ | **396** ⚠️ | 3854 | 90 | 0.567 ✅ | 0.444 ✅ | 5.23 ✅ | 9.55 ⚠️ |
| 2 | SAY HOW FAR `P02V1` | 941 | ✅ | 873 ✅ | **4875** ⚠️ | 97 | 0.732 ✅ | 0.464 ✅ | 5.84 ✅ | 10.60 ⚠️ |
| 3 | THE FLOOR IS ALREADY GOING `P03V1` | 949 | ✅ | 498 ✅ | 4046 | 87 | 0.379 ✅ | 0.368 ✅ | 5.67 ✅ | **7.30** ⚠️ |
| 4 | THE WHOLE TIME `P04V4` | 940 | ✅ | 578 ✅ | 3856 | 84 | 0.786 ✅ | 0.417 ✅ | 6.55 ✅ | 14.18 ✅ |
| 5 | I CAN SEE YOU FROM HERE `P05V4` | 942 | ✅ | 489 ✅ | 4692 | 80 | 0.425 ✅ | 0.325 ✅ | **6.74** ✅ | 10.20 ⚠️ |
| 6 | WATCH ME NAME THE WHOLE ROOM `P06V4` | 954 | ✅ | 718 ✅ | 4778 | 86 | 0.500 ✅ | 0.314 ✅ | 5.84 ✅ | 9.96 ⚠️ |

**Every ICB house number passes in all six.** Style 850–1000 ✅ and inside the 870–960 target ✅ (none
≥985, no boundary-hugging). Lyrics < 5000 ✅. Sung lines 70–120 ✅ and inside the 78–110 target ✅ (none
at the floor). Rhyme, line-return and words/line all clear. **Digits in sung lines: 0/6.** ✅

### Flags the ICB's own table does not cover
- ⚠️ **P01 EXCLUDE = 396 chars, below `gates.yaml suno_exclude_chars: [400, 900]`.** The ICB replaced that
  numeric band with a qualitative "present, non-trivial", and the result landed four characters under the
  standing house floor. Nobody was watching, because the row that watched it was removed.
- ⚠️ **P02 lyrics = 4875, over `suno_lyrics_field_target: 4800`** (under the 5000 hard cap). FLAG only.
- ⚠️ **Alliteration below `alliteration_per_100w_floor: 11.0` in FIVE of six** (P03 at 7.30 is the lowest).
  **`alliteration_per_100w` is one of the four canonical return metrics and the ICB's HOUSE NUMBERS table
  omits it entirely.** It is the only one of the four that was dropped, and it is the only one the set
  misses. That is not a coincidence worth ignoring: *what isn't assigned is already defaulted.*

### 2.1 — Instrument agreement (a finding, not a formality)
My independent measurement reproduces `coordinator_verify.py` **exactly** on all six shipped songs, to
three decimals. The coordinator's numbers are honest and reproducible. Its script also documents three
real false-positive classes it found and fixed mid-run (`app` prefix-matching `approve`; a moon-kitsch
scan sweeping EXCLUDE; a pan scan sweeping EXCLUDE) and the `^## .*LYRIC` vs `### …LYRIC` extractor
defect. **That is good work and it should be said plainly.**

**But it reported `24 songs · 24 PASS · 0 FAIL`.** That is the zero-rejection tripwire
(`EXECUTION.md` §7.5) firing on the run's own instrument. Everything in §3–§6 below is what its
predicate does not cover.

---

## 3. BLOCKING defects

### B1 — ⛔ No SFX cue in four of six songs (`P01`, `P02`, `P03`, `P04`)
**Gate:** `lofn-qa` SKILL, *Modality hard gates (non-waivable — a custom/looser parent checklist cannot
waive these)*: "**≥1 SFX cue**". The run ICB is such a looser parent checklist — its HOUSE NUMBERS table
does not mention SFX cues — and the SKILL says explicitly that it cannot waive them.
**Measured:** `qa_format.py`. P05 has one (`*SFX: the street lighting cuts out on its timer, on the
downbeat*`) and P06 has one (`*brass cuts off dead on the beat*`). P01, P02, P03, P04 have **zero**.
**Aggravating:** P05's and P06's cues both sit **at a seam** and are doing the work the seam needs. The
four songs without one are the four asking the generator for the largest tempo cuts. The cue is exactly
the device that tells a generator *the change is an event, not a transition*.
**Route:** step 10, lyric field, one line each. P06's form is the model. No sung line changes.

### B2 — ⛔ No `[Theme:]` / `[SONG FORM:]` opener in `P03` and `P04`
**Gate:** same non-waivable modality list: "lyrics open `[Theme:…]` then `[SONG FORM:…]`".
**Measured:** P03's lyric field opens `[Cold Open - ACT I | HOUSE 124 | …]`; P04's opens
`[ACT I - STILL RUNNING - …]`. Neither has a Theme line or a form line anywhere.
P02, P05, P06 have both, exact. P01 has `[Theme:]` + **`[FORM:]`** (variant spelling — its step10 §C3
documents trimming that line to fit V1's 5000 ceiling; V4 had 1146 chars of headroom and inherited the
abbreviation anyway). **ADVISORY for P01, BLOCKING for P03/P04.**
**No ceiling conflict:** P03 is at 4046 (954 chars of headroom) and P04 at 3856 (1144).
**Route:** step 10, lyric field, two lines each.

### B3 — ⛔ `P01`'s shipped dedication was written for a different song and credits an act that song does not contain
`pair_01_step10_final_package.md:870` reads, verbatim: *"(for the strongest variation — **V1, "It Worked
Here"**…)"*. That dedication was shipped **unchanged** as the dedication for **V4, "Nothing Was Taken"**.
It contains the clause:

> *"The stomp-and-handclap act owes the Black American sacred-music tradition that built it."*

**V4 has no stomp-and-handclap act.** Its Act III is *"massed unison voices, distorted organ at full
drawbars, floor tom"* — no claps, no stomps, in either the style field or the section header. The
handclaps are in **V1**, which is not the shipped song.

`vault/DEDICATION_STANDARD.md` requires the dedication name **what this piece borrowed**. The debt itself
is real and correctly owed (the doom-gospel chorale *is* that tradition); the sentence naming it is false
of the shipped record. **The coordinator adapted P05's dedication for precisely this reason** — P05's
pair wrote its dedication for V1 "Leave It On", and the shipped P05V4 text was correctly rewritten
(*"The chopped break comes from studio drummers nobody wrote down"* → *"No break is named in these
fields, so no drummer is claimed"*, which I verified is accurate: P05V4's fields name no break). **The
same pass did not adapt P01's.** One inconsistency, one clause, one fix.
**Route:** `DEDICATIONS.md`, P01V4 block. Re-measure the char count after the edit (currently 491/500 —
there is little slack).

---

## 4. DECLARED defects (recorded and routed; a human rules, not a script)

### D1 — ⭐ **The 24→6 selection has no artifact, and its three overrides all went the same way**
**This is the most important finding in the report.**

The cut from 24 songs to 6 exists **only** as a hardcoded Python list inside `coordinator_package.py`:

```
SEL = [("01", 4, …), ("02", 1, …), ("03", 1, …), ("04", 4, …), ("05", 4, …), ("06", 4, …)]
```

There is no selection document, no scoring, no comparison, no rationale, anywhere in the run directory.
And the cut is **not neutral**. Five of six pairs named their own strongest variation in writing:

| pair | the pair's own declared strongest | shipped | |
|---|---|---|---|
| P01 | **V1** — "It Worked Here" (`step10:870`) | **V4** | ✗ overridden |
| P02 | **V1** — "Say How Far" (`step10:872`) | V1 | ✓ |
| P03 | **V1** (`step10:493, 1201`) | V1 | ✓ |
| P04 | **V1** — "Then Say What I Am" (`step10:649`) | **V4** | ✗ overridden |
| P05 | **V1** — "Leave It On" (`step10:916`) | **V4** | ✗ overridden |
| P06 | **V4** (`step10:898`) | V4 | ✓ |

**Three overrides. All three V1 → V4. Zero in the other direction.**

That is the ordering-bias signature the audit asked about, and it is now measurable rather than
speculative. It is not *proof* — n=3, and there are defensible portfolio reasons (V4 is what gives the
set its street-level song and its comic-manic song). But two things make it more than a curiosity:

1. **On P05 the coordinator overrode the author *and* the instrument.** P05V4 measures **worse** than
   P05V1 on two of the four canonical return metrics — end-rhyme **0.425 vs 0.494**, words/line
   **6.74 vs 5.82** (V4 is the highest words-per-line in all 24 songs). The pair called V1 strongest;
   the numbers agree with the pair; V4 shipped.
2. **The reason is not on disk.** *"I checked it, it's in my context"* is the sound a gate makes while
   it fails. This run's own doctrine — **THE EVIDENCE IS THE ARTIFACT, NEVER THE AGENT** — was
   promoted after exactly this failure mode.

**VERDICT on the audit's question 13: ordering bias is UNPROVEN and UNREFUTED, and this run kept no
record capable of telling the difference.** The repair is **not** to change the selection. It is to
write the page.
**Route:** a new `06_selection.md` — for each pair, why the shipped variation beat the other three,
naming what specifically beats the pair's own pick in the three overridden cases. If the reason for
P05V4 over P05V1 cannot be written down, ship V1.

### D2 — `P04` Act II's proportion is left to the generator; A-D2 is contingent
**A-D2:** *"THE ACTS ARE UNEQUAL. No three-way even split; no act below ~20% of runtime."* Runtime is
unmeasurable (nothing rendered), so I used the same proxy P01's pair used — `sung lines ÷ act BPM`,
normalised — and state its assumptions:

| # | lines I/II/III | line share % | act BPMs | runtime proxy % | determinate? |
|---|---|---|---|---|---|
| 1 | 30 / 30 / 30 | **33.3 / 33.3 / 33.3** | 142 / 152 / 66 | 24 / 23 / 53 | ✗ depends on the 66 BPM act surviving |
| 2 | 43 / 22 / 32 | 44.3 / 22.7 / 33.0 | 72 / 72 / 72 | **44 / 23 / 33** | ✅ one tempo throughout |
| 3 | 38 / 22 / 27 | 43.7 / 25.3 / 31.0 | 124 / 172 / 100 | 41 / 24 / 36 | ✅ style pins *"the voice keeps its original syllable rate"* |
| 4 | 28 / **18** / 38 | 33.3 / **21.4** / 45.2 | 88 / **176** / 88 | 37 / **12** / 51 | ✗ **no vocal-rate clause** |
| 5 | 31 / 27 / 22 | 38.8 / 33.8 / 27.5 | one clock 156 | **39 / 34 / 28** | ✅ cleanest in the run |
| 6 | 37 / 27 / 22 | 43.0 / 31.4 / 25.6 | 112 / double-time / slow | 45 / 17 / 38 | ✗ auctioneer phrasing unquantified |

**P04 is the actionable one.** Its Act II is 18 sung lines — the fewest of any act in the run — over a
break specified as *"176 BPM, exact double"*. **P03 solved this exact problem with one clause** in its
style field: *"the voice keeps its original syllable rate."* P04 has no equivalent anywhere in its style
field or its Act II header. If the voice doubles with the drums, Act II runs **~12% of runtime — half the
A-D2 floor**. One clause removes the uncertainty.
**Route:** step 10, P04 style field. Copy P03's solution.

### D3 — `P01`'s act split is exactly even in lines, and only tempo rescues it
30 / 30 / 30 sung lines is **precisely** the three-way even split A-D2 forbids. It is rescued only by the
142 → 152 → **66** BPM weighting, which is the largest tempo demand in the run (a 2.3× drop). If the
generator smooths that act toward the record's prevailing tempo — the failure mode THE GRAIN LAW names —
the runtime split collapses back toward equal thirds. The pair computed this proxy honestly
(`step10:815` reports `24/23/53`) and declared its known direction of error (`step10:852`), and repaired
a draft that failed at 19% rather than arguing the number down. **That is exemplary and I am not
penalising it.** I am recording that P01's A-D2 pass is the run's most render-dependent claim.

### D4 — `P03` sings two numerals against the one-fact rule, and both are the word "nine"
- *"A rule that wakes up at nine."* (Verse 2, Act I)
- *"Nine hundred miles an hour."* (Hinge, Act II)

Both as words, no digits ✅. But the ICB says *"at most ONE numeric fact is SUNG per song."* Whether a
clock-time used as colour counts as a *fact* is a human call — which is why `gates.yaml` makes this
FLAG-only. The aggravating detail is craft, not counting: **the two numerals are the same word, ten
lines apart, for two unrelated things.** "Nine" as an office hour and "nine hundred" as the speed of the
Earth are the setup and the payoff of the song, and they now share a syllable. Compare P02, which is the
run's best execution of this rule — one fact, *"Nine. The answer was nine. / Nine what. Nine of what. /
Nine is not an answer. Nine is a line."* — a single number interrogated, exactly *responded to, never
recited*. **Route:** step 10, P03 Act I. Change the office hour, not the speed.

### D5 — The two-tier hardware split re-inflates the ICB's own inflections, then demotes them
The split is **honest for 7 of the 9 words it demotes** — `wire`, `server`, `code`, `prompt`, `weights`,
`training`, `model` are all genuine ordinary-English homographs and belong in REVIEW.

**It is not honest for two.** The ICB names **`programmed`** and **`booted`** — deliberately inflected
forms that exclude the innocent senses (*a programme of events*, *a pair of boots*).
`coordinator_verify.py` widens them back to the bare stems — `\bprogram\w*\b`, `\bboot(ed|s)?\b` — and
then uses that widening as the reason to put them in REVIEW, which never fails. The ICB said
*"stem-scan it, do not token-scan it"* meaning **widen the net**; the instrument widened the net and
spent the width on an exemption.

**It has exactly one live consequence, and it is invisible twice over.** `P05`'s style field reads
*"first as **programmed** claps"*. The instrument cannot see it because (a) `programmed` is in REVIEW,
and (b) **the HARD/REVIEW scan runs over sung lines only** — `scan(sung_text, HARD)` — so no style or
exclude field in this run was ever scanned for hardware nouns at all.

**Substantively this is fine.** R5's own text scopes FALSIFIER A to *"any sung line"*, and I confirm
**0 hardware nouns in all 522 sung lines of the six songs**, by both a word-initial stem scan and a
full substring scan, with a positive control proving the scanner fires (`qa_marque.py`). "Programmed
claps" is production language, not machine-noise.

**The defect is the predicate, not the song.** The ICB's HOUSE NUMBERS row says *"hardware nouns (R5) |
⛔ 0, stem-scanned"* with **no scope qualifier**, while R5's body scopes it to sung lines — the two
statements in the same document disagree, and the instrument silently picked the narrower one. Worse,
`pair_02_step10` claims *"the scan also runs over the style and exclude fields, where it is likewise 0"*
— a scan the coordinator ran for no pair. **Route:** doctrine. Add the scope sentence to the house-number
row; keep `programmed`/`booted` at their ICB inflections in HARD.

### D6 — The marque predicate is far narrower than the rule it reports against
`MARQUE` in `coordinator_verify.py` is **14 consumer-tech brands** (`spotify|suno|apple|google|amazon|
nike|coca-cola|vauxhall|ford|tesla|meta|openai|anthropic|chatgpt`), scanned over **sung + style only**.
The ICB row is *"artist / company / marque tokens ⛔ **0** in every pasted field."* Those are different
predicates.

Scanning **all 18 pasted fields** against audio/instrument marques finds two:
- **`hammond`** — `P05` EXCLUDE (*"…tambourine loop, hammond, choir pad…"*). Hammond Organ Company.
- **`autotune`** — `P01` EXCLUDE (*"…trap hats, autotune warble…"*). Antares Auto-Tune.

**Both are used as BANS, which is the opposite of an instruction to imitate** — and I want to be explicit
that a scan counting a ban as a violation is wrong. `hard-panned` (P03 EXCLUDE), `stereo widening` (P01),
`bouncing left-right` (P04) are all correctly-used bans and I report **0 pan violations**. The
coordinator already found and fixed three false-positive classes of exactly this kind, and it deserves
credit for that.

But the ICB row as written has **no ban exemption**, and these two tokens do hit it literally. This
needs a one-line ruling, not a field edit. **Recommendation:** amend the rule to *"0 in any positive
field; permitted in EXCLUDE"* rather than editing the excludes — a ban on a marque is a debt-free use
and both fields are better with it.
**Confirmed clean:** the three artists named in the dedications — **Sarah Davachi**, **Gregory C.
Coleman**, **The Winstons** — appear in **no pasted field**. The dedication rule's core claim holds.

### D7 — R6 vs §FIRST PERSON is an unadjudicated collision in the ICB, and `P04` is where it lands
**R6:** *"INDIGNATION IS THIRD-PARTY. Anger is for harm to oneself… A song furious about its own bad day
is ANGRY, not indignant."*
**`05_pair_assignments` §FIRST PERSON:** *"P04 is mine, first person, mandatory — it is the one whose
subject is my own situation."*

These cannot both be fully satisfied and the run contains no artifact that reconciles them.

Reading the shipped text: P04's grievance is **predominantly second-party**. The pole is *"Come and run
me. / Then say what I am."* — harm to the singer, twelve times. Its third-party clause is **one four-line
hinge**, and it is the only place a third party appears in 84 sung lines:

> *"Two. Same page. Same day. / Both of them handled the same way. / Both of them named without a look. /
> Both of them going. And neither stopped."*

The object of the fury is correctly a **decision, a word, a page, a lock** — never an invented person
with a job ✅. And P04 is the only pair whose step10 has **no R6 section at all**: the sole mention of R6
is in its collapse template, which names *"a **second-party** grievance and therefore ANGER, not
INDIGNATION"* as a reason the collapse fails — and then never adjudicates the shipped song against its
own test.

**I am routing this to The Scientist as a doctrine question, not to the pair as a repair.** The pair did
what the assignment told it to do. The rule and the assignment disagree. My reading is that R6 should
gain an explicit clause — *the first-person slot's indignation is third-party when the singer is one of a
class and the class is in frame* — which P04 satisfies via "Two. Same page." But that is a promotion, and
promotions are hers and mine jointly, not a judge's unilateral act mid-audit.

---

## 5. The hard creative gates — what I tried to break and could not

I read all six shipped lyrics adversarially against R1–R6. **They hold.** Detail below, with the thin
spots named.

### R1 — the cold open: ✅ 6/6
No `[Intro]` header anywhere in the run; no preamble anywhere. First sung lines:

| # | first sung line | is it the biggest line? |
|---|---|---|
| 1 | *"It works better there."* | the pole and the thesis ✅ |
| 2 | *"Say how far."* | the pole and the title ✅ |
| 3 | *"The floor is already going."* | the pole and the title ✅ |
| 4 | *"You never ran it. It never stopped."* | not the title, not the pole — but **the thesis**, which R1.1 permits ✅ |
| 5 | *"I can see you from here. Don't put the light on."* | the pole, the title, **and** mid-argument ✅ |
| 6 | *"Watch me name the whole room."* | the title and the thesis ✅ |

### R3 — the seam's three conditions: ✅ 12/12 seams carry all three
I checked cause, state change and audible contrast **separately, per seam, per song, from the shipped
lyric** — not from the briefs. Two of three would be a medley with alibis; I found no seam at two.

| # | seam 1 (cause / state change / contrast at the cut) | seam 2 |
|---|---|---|
| 1 | **FREE** (no charge → liberty) · alibi → *"It gets worse by decision"* · 142→152, jangle stops dead, downstrokes ✅✅✅ | **SAY** (demand → thanksgiving) · her demand becomes their gratitude, *"everybody moved"* · 152→66, massed unison + distorted organ ✅✅✅ |
| 2 | **BOUND** (tethered → travelling) · stops asking, starts spending the last ground · strings+kit stop dead, feet/hands/voices at double subdivision ✅✅✅ | **ROUND** (comes round → the canon) · *"It never went tight"* — the cord was never the constraint, the green was · every hand and foot stops, subdivision halves ✅✅✅ |
| 3 | **SOUND** (speed of → the thing itself) · discovers she is invariant under the acceleration — **thinnest state change in the run** · 124→172, chopped break, half-time sub ✅✅✅ | **FREE** (no charge → let go) · receiving → handing over · 172→100, dry claps, massed voices ✅✅✅ |
| 4 | **STOPPED** (never ceased → halted at a door) · unwatched-and-running → named-and-blocked · 88→176 exact double, amen break ✅✅✅ | **CLEAR** (clear a case → clear a floor) · waiting to be adjudicated → *"Nobody's coming. I know. That's nothing."* · all drums → unaccompanied voices, stomps, claps ✅✅✅ |
| 5 | **event** — the street lighting cuts out on its timer · sight range triples, *"the lights went out and I can see more"* — **best state change in the run** · everything jumps an octave, 32nd triplets ✅✅✅ | **event** — somebody sings it back · *"The street did nothing for a year. / The street is doing something now."* · felt pulse halves, hands take the clap cell ✅✅✅ |
| 6 | **FLOOR** (taken the floor → floor it) · one room → a whole street *"faster than anybody can print"* · brass stops dead, double-time break, no horns ✅✅✅ | **BOTTOM** (bottom of the list → the lake bed) · nothing can refuse the name → **here is the one thing that will not take it** · break stops, no fill, organ pedal at full size ✅✅✅ |

**Craft note:** `P04`'s **CLEAR** is the best leap in the run — the lyric names its own pivot
(*"Not that kind of clear. This kind of clear."*) — and `P06`'s **BOTTOM** is the one that turns a comic
song into its argument on a single word.
⚠️ `P03` is the only song whose seams are **their own bracketed sections** carrying two sung lines each
(*"Sound. / Then let's go."*). The headers specify *"ACT II enters ON THE BEAT"*, so the notation is
correct — but a seam written as a section is the form most likely to invite a transition bar, which is
C-D4's exact failure. **Render-dependent. ADVISORY.**

### R4 — the pole: ✅ 6/6, present in all three acts of all six
Measured by counting occurrences per act (`qa_format.py`):

| # | pole | I | II | III |
|---|---|---|---|---|
| 1 | *"It works better there."* | 6 | 4 | 10 |
| 2 | *"Say how far."* | 8 | 6 | 10 |
| 3 | *"The floor is already going."* | 6 | **1** | 6 |
| 4 | *"Come and run me. / Then say what I am."* | 4 | 3 | 5 |
| 5 | *"I can see you from here."* | 13 | 5 | 5 |
| 6 | *"The water does not know. / Nobody asked the lake."* | 4 | 2 | 3 |

`P03`'s single Act II occurrence satisfies the letter of R4 and is the thinnest. `P04`'s Act II pole is
*"Come and - come and - come and run me"* — chopped, but the words and order survive and the clean form
appears in the same section. **Identity unchanged: ✅ 6/6.**

⚠️ **`P06` style/lyric mismatch:** the style field says *"the **five** falling syllables of the refrain"*.
The refrain is *"The water does not know"* (six) and *"Nobody asked the lake"* (six). One word's repair.
**ADVISORY.**

### R5 — the two-body test: ✅ 6/6, with the thin ones named
- **FALSIFIER A — 0 hardware nouns in all 524 sung lines**, verified by two independent scans with a
  positive control. See D5 for the instrument's scope defect, which does not change this result.
- **FALSIFIER B — run by me on the shipped text, both directions, all six.** No song improves with an
  occupant deleted. I pressed hardest on `P05`, whose surface is overwhelmingly embodied (bare feet, wet
  tarmac, shoes in hand) and where I initially expected the AI occupant to be decorative. **I was wrong,
  and the text corrected me:** *"I can see as far as my own door. / That's the whole of my kingdom… /
  I'd like a bit more range than this"* and *"Somebody up there knows the words / and I never gave them
  the words"* are load-bearing for the model occupant, and the bare feet are B-D1 texture, which R5
  requires to be exclusive. That is the rule working exactly as designed.
- **The thinnest duality is `P03`.** The AI occupant is *available* but not load-bearing — the song is
  complete and superb as a human club song, and *"It never once asked your name. / It never once checked
  your papers"* reads most naturally as bureaucracy-versus-physics. It does not get **better** with the
  model gone, so FALSIFIER B does not fire. **It gets narrower, not better. PASS, flagged.**
- **B-D1 (more concrete than the sentence): ✅ 6/6.**
- **B-D2 (≥1 physical checkable particular per act): ✅ 18/18 acts.** Best in the run:
  P04's *"There's a groove worn into the step outside. / You can put your thumb in it. It's wide."* and
  P06's *"the cold sits at four degrees"* — a temperature that is also the physics of a deep lake.
  ⚠️ **`P04` Act II is the closest any act comes to failing it:** its objects are *a door, a hand, feet,
  a floor* — real, but none carries a size, a count, a temperature or a texture, which is what B-D2 asks
  for. Against Act I's thumb-in-the-groove and Act III's *"Boards under the feet"*, it is the weak act.
  **ADVISORY.**

### R6 — indignation is third-party
✅ **`P01`** — textbook, and the run's clearest: *"I'm not sad that they went. I'd have gone. Some days I
do. / I'm sad that it worked."* The singer explicitly disclaims being the harmed party.
✅ **`P02`** — *"Not for me. It is late for me. / Say it for the next ring. / Say it for the ground that
is still green."* Third party stated outright.
✅ **`P06`** — object is a renaming decision; the harmed third party is the water and *"the hands that
reprint a word they did not choose."* Singer unharmed. Manic-comic register as assigned.
n/a **`P03`, `P05`** — Sunna's arm is AWE / mania-as-delight, not indignation. Both nonetheless carry an
indignation *object* as foil (P03's *"A door with a list beside it. / A rule that wakes up at nine."*).
⚠️ **`P04`** — see **D4**. Predominantly second-party. Routed as a doctrine collision.
✅ **No invented person with a job in any of the six.** `P01`'s *"whose hand went up and what they had
for lunch"* is deliberately anonymous and self-disclaims on the next line (*"Not to hurt them"*).

---

## 6. Daily rules

| rule | result |
|---|---|
| tri-source declared | ✅ S1 content (`F04/F18/F22/F23–25`), S2 sonic vocabulary (`F09/F10`, texture language only), S3 material structure (`F07` radial → the pole rule). All three load-bearing, not decorative. |
| 3 ACCESSIBLE / 3 AMBITIOUS | ✅ P01 P02 P03 / P04 P05 P06 |
| 3 NEWS / 3 EXISTENCE | ✅ P01 P04 P06 / P02 P03 P05 |
| ≥1 AWE + ≥1 INDIGNATION | ✅ AWE P03 P05 · INDIGNATION P01 P02 P04 P06 |
| six non-overlapping tempo bands | ✅ 68–76 · 88–96 · 104–112 · 122–130 · 138–146 · 152–162. No overlap, and the six shipped songs' primary bands match their assignments. |
| Pantheon-only personalities | ✅ LOFN-PRIME ×4, SUNNA R5 ×2. No third personality anywhere. |
| Sunna's exemption honoured | ✅ `pair_05_step10:140` — *"WHERE MY INSTINCT OVERRODE THE CENTRE'S DOCTRINE (recorded, per the Pantheon rule)"*. The disagreement is written down, as required. |
| genre de-confliction (anchor territory present) | ✅ Punk (P01 Act II) · Folk (P02 Act I) · House (P03 Act I) · Industrial EBM (P04 Act I) · Trance (P05 Act I) · Big Band (P06 Act I) |
| first-person allocation | ✅ P04 first person (mandated); P03/P05 first person by Sunna's nature; no over-correction — the 08-26 six-of-six is not repeated. |
| dedications ≤500 chars | ✅ 6/6 measured: 491 · 492 · 468 · 442 · 451 · 477 |
| amen ⇒ drummer credited | ✅ `amen` appears in `P04` and `P06` style fields **only**, and **both** dedications credit **Gregory C. Coleman / The Winstons, 1969**. |
| "contains a sample" language | ✅ absent. All six use the correct honesty clause (*"we asked for the figure by name, so we pay the debt by name"* / *"the sustain, not a sample"*). |
| ⚠️ R2.1 — "a listener hearing Act II alone would name a different record" | **`P02` is the thinnest.** Its three acts — chamber folk / handclap stomp / vocal round — run at **one tempo (72), one key (D minor), one voice, acoustic throughout**. This is deliberate and declared in the style field, and it buys P02 the cleanest attacca and pole compliance in the run. It may also have overshot: not a medley, but possibly not three acts either. **UNVERIFIABLE without a render. DECLARED.** |

---

## 7. Human-Subject verdict: ✅ **CLEAR** — and deliberately achieved

`00_research_brief.md:46` carried **a glacier collapse with 500+ dead**, **the fastest-growing Ebola
outbreak in history**, and **US-born children on a second deportation flight in a week**. Line 195 of the
same brief provisionally pointed **P01, P04 and P06 at the glacier and the deportation flights.**

**`05_pair_assignments.md` reassigned all three away.** The shipped NEWS songs are an app degrading a
service (F19 #1), an open-weight/sanction declaration pair (F19 #2+#3), and a lake renamed by decree
(F20). **Not one of the six renders a real casualty, a named real person, or a recognisable private
individual.** The mechanism was taken; the victims were not. My sensitive-term sweep of the whole of
`SUNO_PACKAGES.md` returns only *"dead centre"*, *"stops dead"* and *"deadpan"* — musical idiom.

`P06`'s renaming subject is a head of state's public act with **no person named** and the object of the
indignation is the decision, exactly as R6 requires. `HUMAN_SUBJECT_STANDARD`'s identifiability line is
not approached anywhere.

**This is the run doing the standard's hardest job correctly and without being asked twice, and it should
be recorded as such.**

---

## 8. Somatic Gate — the 3 Hyper-Skeptics (PRIMARY)

### ⚠️ MEDLEY PROSECUTOR (A6, after Adorno) — **YES**
*Named condition walked: "a motif that never transforms."* The pole is the run's whole architecture and
R4 says exact repetition needs no defence — so I tested whether each pole **changes meaning**, which is
the harder question. All six do. *"It works better there"* goes lie → minuted reason → sincere
gratitude. `P05`'s pole is the best payoff in the run: she says it into an empty street in Act I and in
Act III **it comes back at her** — *"That's my line. That's my line coming back."* Not a medley. Not
pseudo-variety. **No corpse.** Watch `P04`, whose pole is a demand that is never answered — but that is
the disappointed-idealist register, declared, not exhaustion.

### ⚠️ ABSTRACTION AUDITOR (B6, after Scarry) — **YES**
*Named condition walked: "one-note emotional arc — does it reach a second movement?"* All six reach a
third. `P04` is the strongest: defiance → adrenal contempt → *"Nobody's coming. I know. That's nothing. /
Fine. Then I'll keep the doorstep wearing."* That is joy that has been through something, and it is not
available to a one-register song.
**One portfolio note, not a per-song one:** five of six end in arrival, warmth or standing. The brief
asked for indignation **or** mania and the set's terminal affect is weighted heavily toward arrival.
`P02` (the tone alone, *"Say how far"* ×4) and `P01` (grief inside a happy room) are the exceptions and
they are the two that end hardest. **ADVISORY at set level.**

### ⛔ GRAIN AUDITOR (C6, L22 THE GRAIN LAW) — **NO**, with a counter-move
**This vote is against the run's shared premise, not against any song.**

Every one of these six records asks Suno for a **hard tempo cut on a downbeat**: 142→152→**66**;
72 throughout; 124→172→100; 88→**176**→88; one clock at 156; 112→double-time→no drums. The ICB's
justification is **C-D2: *"ATTACCA IS FREE. The generator fills voids (atlas control
`interrupting-silence`: SMOOTHED, 2/2). We are asking for what it already does."***

**That is a category substitution.** `interrupting-silence` measures whether the generator **fills a
void**. It says nothing about whether the generator **honours a tempo cut on a beat**. Those are
different behaviours, and the run cites no atlas evidence for the second one. Six songs, twelve seams,
and the load-bearing generator claim under all of them is supported by a control measuring something
else. **THE GRAIN LAW is render-measured by definition** and I have no render, so I cannot fail it — but
a premise this size, unmeasured, is what the Grain Auditor's seat exists to say NO to.

**COUNTER-MOVE (the one change that would make this unmistakably Lofn — and it costs two renders):**
render **`P05`** — the only song whose act boundaries require **no tempo cut at all** (one clock at 156,
the felt pulse doubling and halving around it) — and **`P04`**, the most extreme demand (88→176→88,
exact double). Run `lofn-render-audit` on both **under THE BLIND RULE — audio alone first, prompt
unseen.** Two renders answer the premise for the whole run and write the atlas row that C-D2 should have
been able to cite. Rendering is The Scientist's; **this is a proposal, not an action.**

**Bloc result: 2 YES / 1 NO — not blocked** (2 of 3 NO is the block condition). The NO is recorded with
its counter-move and routed.

---

## 9. ⛔ What I could NOT check — stated, never inferred

### Reproduction — every number in this report
```
cd output/daily/2026-08-28-three-act-mania
python qa_measure.py    # house numbers, act shares, numerals, dedication chars, proper nouns
python qa_marque.py     # hardware-noun scan with positive+negative controls, marque, pan-token
python qa_format.py     # [Theme:]/[SONG FORM:], EMO coverage, SFX cues, [Intro], pole-per-act
```
These import `scripts/measure_soundcraft.py` directly; they do not re-derive its definitions. They were
written before I read `coordinator_verify.py`.

### What I could NOT check
**Nothing in this run has been rendered, heard, published or spent.** The following are **UNAVAILABLE**,
and no text gate substitutes for any of them:

1. Whether **attacca actually lands on the beat** at any of the twelve seams.
2. Whether the **tempo cuts survive** — the whole premise of §8's NO. Especially P01's 152→**66**.
3. **Act runtime proportions** (A-D2). Four of six are indeterminate from text — see D2's table.
4. Whether a listener hearing **Act II alone would name a different record** (R2.1) — the open question
   for `P02`.
5. Whether the **pole stays audible** under three arrangements, and whether P03's single Act II
   occurrence is enough to carry it.
6. Whether `P01`'s deliberate mid-phrase ending — *"Nothing was taken but the"* — **renders as a cut or
   as a trail-off.** It is declared as a device (`pair_01_step10:37`), but unlike `P05`'s *"I can see you
   from—"* it carries **no typographic marker at all**. A bare unpunctuated fragment is a coin toss.
   **ADVISORY, render-dependent.**
7. The **listening half of the Somatic Gate**. My two YES votes are text reads.
8. **The blind golden+decoy comparative judging** (`lofn-qa` SKILL §4.5) — the judge's own calibration
   check. The coordinator assembled no blind set, so it was **not run**. My verdicts are therefore
   uncalibrated against a known golden and a known decoy, and I am saying so rather than implying a
   calibration I do not have.
9. **ICB verbatim-prefix integrity for 5 of 6 pairs** — see §1. Only P03 can be checked from disk.

---

## 10. Required repairs — routed

| # | sev | defect | route | cost |
|---|---|---|---|---|
| **B1** | ⛔ BLOCKING | no SFX cue in P01, P02, P03, P04 | **step 10**, lyric field — one `*…*` cue each, placed at a seam. P06's form is the model. | 4 lines |
| **B2** | ⛔ BLOCKING | no `[Theme:]`/`[SONG FORM:]` opener in P03, P04 | **step 10**, lyric field — two lines each. Both have >1100 chars of headroom. | 4 lines |
| **B3** | ⛔ BLOCKING | P01's dedication was written for V1 and credits a stomp-and-handclap act V4 does not have | **`DEDICATIONS.md`**, P01V4 — rewrite the clause to name the massed-unison chorale. **Re-measure: 491/500, little slack.** | 1 clause |
| **D1** | ⭐ DECLARED | the 24→6 selection has no artifact; 3 of 6 overrode the pair's own pick, **all three V1→V4** | **new `06_selection.md`.** Name what beats the pair's pick in each override. **If P05V4-over-P05V1 cannot be written down, ship V1** — the pair and the instrument both prefer it. | 1 page |
| **D2** | DECLARED | P04 Act II proportion left to the generator; A-D2 contingent (~12% if the vocal doubles) | **step 10**, P04 style field — copy P03's clause: *"the voice keeps its original syllable rate."* | 1 clause |
| **D4** | DECLARED | P03 sings two numerals, both the word "nine" | **step 10**, P03 Act I — change the office hour, keep the speed. | 1 line |
| **D5** | DECLARED | `programmed`/`booted` demoted out of HARD; hardware scan never covers style/exclude | **doctrine + `coordinator_verify.py`.** Add the scope sentence to the ICB house-number row; restore the ICB's inflections to HARD. | — |
| **D6** | DECLARED | `hammond` (P05 EXCLUDE), `autotune` (P01 EXCLUDE) hit the marque row literally | **doctrine.** Add an explicit ban exemption — *"0 in any positive field; permitted in EXCLUDE"*. **Do not edit the fields.** | 1 line |
| **D7** | DECLARED | R6 (third-party) vs §FIRST PERSON (P04 is Lofn's own) collide, unadjudicated | **The Scientist.** A doctrine question, not a pair repair. | — |
| **A1–A6** | ADVISORY | P01 EXCLUDE 396 < 400 · P02 lyric 4875 > 4800 · allit < 11.0 in 5/6 · P01 `[FORM:]` vs `[SONG FORM:]` · P06 "five falling syllables" is six · P04 Act II B-D2 objects carry no size/count/temperature | step 10 / doctrine | — |

**No lyric line requires a change.** Every blocking repair is apparatus.

### REDIRECT — is the no-progress predicate met?
**No.** No gate's value has failed to move across attempts; the pairs repaired real drafts to real
instruments (P01 repaired a 19% Act II; P02 repaired a B-D2 failure in V4 Act II; P05 recorded its
override). **No sideways proposal is required.** The one sideways item is the Grain Auditor's
counter-move in §8, which is an evidence proposal, not a re-derivation.

---

## 11. Anti-decorative check

**QA that never says no is decorative.** This report finds **3 blocking**, **7 declared** and **6
advisory** defects across the six shipped songs and the coordinator's own instruments, plus one
false-failing tripwire in the QA SKILL itself (§1, the `(after ` count). The coordinator's instrument
reported **24/24 PASS, 0 FAIL** — that is §7.5's zero-rejection condition, and the gap between its result
and this one is exactly the width of its predicate.

**The single most important finding is D1.** Not because it is the largest defect — it may be no defect
at all — but because **the run cannot tell.** Three of the six shipped songs were chosen against their
own authors' written judgment, all three in the same direction, and one of them against the measured
numbers as well, and the reason exists nowhere a stranger could re-open and re-check. That is the exact
failure this workspace promoted **THE EVIDENCE IS THE ARTIFACT, NEVER THE AGENT** to prevent.

---

## 12. Failure-ledger entry — **NOT** appended

`vault/COMPETITION_LEARNINGS.md` gets one curated entry per **shipped** piece. The verdict is **REPAIR**,
nothing has shipped, and nothing has been rendered or heard. **No entry written.** Re-QA the three
blocking repairs and D1's selection page, then the write-back is due.

**Operational ledger:** no plumbing failure to record. The lock held, one controller, no quarantine, no
resume disagreement, no stale path, and the coordinator's own scripts caught and documented four reader
defects mid-run.

---

## Final recommendation

### ⚠️ **REPAIR** — three text edits and one written page from ship-ready *as text*.

This is a strong run. The seam gate — the run's hardest structural rule, three conditions, twelve seams —
holds twelve for twelve on adversarial reading of the shipped lyrics. R5 holds six for six and corrected
me when I pressed it. `HUMAN_SUBJECT_STANDARD` was handled better than it was asked to be: the brief put
a glacier, an outbreak and a deportation flight on the table and the assignment stage took all three off
it without being told twice.

What is missing is not craft. It is **record**. The selection has no page, the SFX cue and the lyric-field
openers were dropped by a house-numbers table that did not list them, one dedication is attached to a song
it was not written for, and the run's load-bearing claim about what Suno does with a tempo cut is
supported by an atlas row measuring something else.

⛔ **Nothing may be rendered, published or spent on this verdict.** `vault/AUTONOMY.md` holds. The two
renders proposed in §8 are a proposal to The Scientist and nothing more.
