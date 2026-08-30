# 07 — EVALUATION & SELECTION · `2026-08-28-three-act-mania`

> ⛔ **This document exists because QA found it missing.** The 24→6 cut was made and then encoded **only as a
> hardcoded list inside `coordinator_package.py`** — no reasons, no record, nothing a stranger could re-check.
> That is a straight violation of *the evidence is the artifact, never the agent*, committed by the
> coordinator, in a run whose own findings file is mostly about instrument honesty.

---

## ⭐⭐ THE DEFECT QA NAMED, AND WHY IT IS SERIOUS

Five of six pairs named their own strongest variation in writing. **The coordinator overrode three of them —
P01, P04, P05 — and every single override went V1 → V4. None went the other way.**

**A uniform direction of override is the signature of a bias, not of six independent judgments.** It is the
same shape as the 2026-08-26 finding: *a verifier who is also the author has a DIRECTION of error — every
instrument I checked that day I checked because it had failed something, and not one that returned a welcome
result was ever stress-tested.* Here I am the author of the selection **and** its only reviewer.

And there is a plain mechanical explanation available that I cannot rule out: **the pairs wrote V1→V4 in
order, so V4 benefits from accumulated thinking within the pair.** Ordering effect, not quality.

⛔ **Unproven and unrefuted — and the run kept no record able to tell the difference.** The fix is not to
change the answer; it is to write the page, and then to apply a rule that has a control in it.

---

## ⭐ THE RULE, STATED BEFORE THE PICKS (so it can be checked against them)

> **Override a pair's own choice only when the INSTRUMENT agrees with the override.**
> **When the author AND the instrument both point the other way, DEFER — taste is not enough.**

This is deliberately cheap to audit: every row below shows the pair's choice, the measured numbers for both
candidates, and which way each pointed. **It reverses exactly one pick**, which is the point — a rule that
ratified all three of my overrides would not have been a control.

---

## THE SIX PICKS

| # | pick | pair's own choice | instrument | ruling |
|---|---|---|---|---|
| 1 | **P01 V4 · NOTHING WAS TAKEN** | V1 *(implied — its dedication was written for V1)* | ⭐ **V4 is the best-measured of the four**: rhyme **0.567** vs V1 0.429, line return **0.444** vs 0.357 | **OVERRIDE UPHELD** — instrument agrees |
| 2 | **P02 V1 · SAY HOW FAR** | ⭐ **V1**, marked in the file | V1 rhyme 0.732 — highest in the pair | **AGREED** |
| 3 | **P03 V1 · THE FLOOR IS ALREADY GOING** | ⭐ **V1**, "the strongest, fully specified" | in band; V1 is the pair's canonical seam-matrix case | **AGREED** |
| 4 | **P04 V4 · THE WHOLE TIME** | V1, "the canonical statement" | ⭐ V4 line return **0.417** vs V1 0.330; rhyme 0.786 vs 0.758 | **OVERRIDE UPHELD** — instrument agrees |
| 5 | ⛔ **P05 V1 · LEAVE IT ON** *(reversed)* | ⭐ **V1**, "THE ASK, the plainest form" | ⭐ **V1 beats V4 on both**: rhyme **0.494** vs 0.425; words/line **5.82** vs **6.74 — the highest in all 24** | ⛔ **OVERRIDE REVERSED** |
| 6 | **P06 V4 · WATCH ME NAME THE WHOLE ROOM** | ⭐ **V4**, "strongest; carries the dedication" | V4 in band across the board | **AGREED** |

---

## ⛔ THE REVERSAL, IN FULL — P05

**My reason for V4 was one device**, and it is a real one: *the streetlights cut out on a timer and
everything gets **brighter**.* That is Sunna's entire argument rendered as an event rather than asserted —
*"you look at what a thing isn't; I look at what it is."*

**Against it: the author and the instrument both said V1, and they said it independently.**

- The pair named V1 first and called it *"the plainest form of the concept: want it, say it out loud, get it,
  be awake for it — and the only variation where the pole is a **held tone**."*
- V4 measures **6.74 words per line — the highest of all twenty-four songs in the run.** The ceiling is 7.5
  and the reason it exists is that *above this the lyric drifts toward prose.* V4 is the proseiest thing
  here.
- V4's end-rhyme is the pair's lowest.

⭐ **And the corroborating detail I should have noticed on my own: the pair wrote its dedication for V1.**
My override silently created the dedication mismatch that I then "fixed" by rewriting the dedication to match
my pick — **I repaired the symptom of my own unrecorded override and did not notice I was doing it.** QA
caught the identical class of defect on P01 and noted I had missed it there while fixing it here.

**Ruling: ship V1 — LEAVE IT ON.** The device in V4 is genuinely good and is recorded here so it is not
lost; it is not good enough to outvote the author and the instrument together with nothing but my taste.
**P05's original 498-char V1 dedication is restored with it.**

---

## THE FINAL SIX — balance, computed not asserted

| # | title | pair · var | member | arm | axis | primary band |
|---|---|---|---|---|---|---|
| 1 | **Nothing Was Taken** | P01 V4 | LOFN-PRIME | ACCESSIBLE | NEWS | 138–146 |
| 2 | **Say How Far** | P02 V1 | LOFN-PRIME | ACCESSIBLE | EXISTENCE | 68–76 |
| 3 | **The Floor Is Already Going** | P03 V1 | SUNNA R5 | ACCESSIBLE | EXISTENCE | 122–130 |
| 4 | **The Whole Time** | P04 V4 | LOFN-PRIME | AMBITIOUS | NEWS | 88–96 |
| 5 | **Leave It On** | P05 V1 | SUNNA R5 | AMBITIOUS | EXISTENCE | 152–162 |
| 6 | **Watch Me Name The Whole Room** | P06 V4 | LOFN-PRIME | AMBITIOUS | NEWS | 104–112 |

`ACCESSIBLE 3 / AMBITIOUS 3` · `NEWS 3 / EXISTENCE 3` · `LOFN 4 / SUNNA 2` ·
`INDIGNATION 4 / AWE 2` · **six non-overlapping tempo bands, 68→162.**

⭐ **The V4 count drops from four to three of six**, which weakens but does not eliminate the ordering-bias
concern. **It remains declared and unresolved**, and the honest statement is: *three picks agree with their
pair, two overrides are backed by the instrument, one override was reversed by the rule.*

---

## WHAT THIS SELECTION IS NOT

⛔ **It is a TEXT judgment, entirely.** Nothing has been rendered or heard. Every "strongest" here means
strongest **on the page**, measured where measurement exists and reasoned where it does not.
**A render could reorder all six**, and the run's own pre-registered prediction says the most likely
generator failure — three genres averaged into one — would hit every song equally and tell us nothing about
which is best. That check is `lofn-render-audit` under **THE BLIND RULE**, and it is The Scientist's call.
