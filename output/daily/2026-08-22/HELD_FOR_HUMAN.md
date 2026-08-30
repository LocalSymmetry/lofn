# HELD FOR HUMAN — `daily-music-2026-08-22-sunna-offbeat`
## ⛔ THE RUN-LEVEL CIRCUIT BREAKER IS TRIPPED

*`lofn-daily` § run-level circuit breaker: **"when ONE named gate fails across many pairs, STOP the
daily run and surface the named blocker for human eyes instead of letting 24 × 3 attempts hammer the
same broken constraint. When it trips: name the failing gate, report which pairs hit it, and HALT — do
not auto-repair."***

**Scoped correctly:** this is SAME-gate correlation across **four of six pairs**, not an aggregate
fail-count. Independent one-off failures were repaired locally and are closed (P02's shared line,
P06's `kettle`). ⛔ **This one is not being auto-repaired, because the root cause is in the
coordinator's own brief and repairing the pairs would be treating the symptom.**

---

## THE FAILING GATE: **RULING X — "the absence, NAMED IN THE LYRIC, in words, once"**

### What happened
Ruling X was written for one purpose: this run's three central claims are invisible to every instrument
we own, so each had to **cash out into something a gate can read.** The absence's cash-out was *"name
it in the lyric, in words, once."*

⭐⭐ **Five of six pairs invented ONE sentence for that naming and wrote a variant of it in all four
variations — at the same section index and the same line index.** P03 is decisive:

| variation | the naming, at Verse 2 lines 1–3 |
|---|---|
| V1 | *The **melodica** played the whole of that, and now it has gone. / It is not coming back. Fine. I liked it while it was on. / The bass leaves the hole open where the tune used to come in.* |
| V2 | *The **saxophone** had the tune for the first bit and then it went. / It is not coming back in and I am not going to pretend it is. / The bass leaves the hole at the top where the horn came down.* |
| V3 | *The **trombone** finishes the line it is on, and then it is gone. / It does not come back. I waited. It does not come back. / The bass keeps the gap open at the top of every time round.* |
| V4 | *The **flute** had the tune down the road, and then it stopped. / It has gone. It is not coming back on. I stopped waiting. / The bass goes round and leaves the top of it empty every time.* |

`pair_03_step10_final_package_enhanced.md:73-75, 231-233, 385-387, 538-540`. **One three-line machine,
four times, with the instrument swapped.**

### ⛔ AND EVERY INSTRUMENT THIS RUN OWNS REPORTS IT AS CLEAN
- `measure_variations.py` — 24/24, **0 hard fails**
- `distinctiveness.py` test 7 (exact repeated lines) — **0 leaks, all six pairs**
- `validate_step.py` — **STEP 10 PASSED, all six**

The formula is **paraphrase**, and every gate we built measures **identity**. The QA judge built
`blockdup.py` (line-level token-set Jaccard) and calibrated it in both directions:

| corpus | duplicate blocks per comparison |
|---|---|
| cross-pair, unrelated songs | **0.02** *(negative control — the floor)* |
| this run, within-pair | **1.08** |
| **2026-08-21 pair 04** *(the corpus R3.2 proved templated)* | **1.83** |
| **P02, this run** | **1.83** — ties the known-templated control |
| ⛔ **P04, this run** | **2.17** — **above** the known-templated control |

---

## ⭐ THE ROOT CAUSE IS MINE, AND IT IS A DOCTRINE-LEVEL FINDING

I never wrote a sentence template. **I wrote two rules whose conjunction is one.**

1. **Axis E** gave each pair exactly **one kind of absence** (P03: an instrument that leaves).
2. **Ruling X** required it be **named in the lyric, in words, once.**
3. Four variations share bed, tempo, setting and reveal engine, differing only by **angle**.

⭐ **Those three together can be satisfied by filling in a blank — the instrument's name — and R3.2 rule
3 says exactly what happens then: *"If the assignment can be satisfied by filling in a blank, it is a
grammar and it will be templated."*** I obeyed the letter of "assign the kind, never the grammar" and
built a grammar out of two kinds.

### And my falsification test was pointed at the wrong axis
`02_golden_seed.md` §7 names **the absence** as the run's most-specified object and ships the
**HOLE-SWAP** test with it — *"take pair A's missing thing and put it in pair B's song."* That is a
**CROSS-PAIR** test. **The formula went WITHIN-PAIR.**

⛔ **R3.1's law, exactly: a falsification test only protects the thing it names.** I quoted that law in
the seed document, wrote a test, and pointed it at the wrong dimension in the same breath.
**R3.2 found the formula in the endings because the test pointed at the wolf. This run found it in the
namings because the test pointed cross-pair. The formula moves to the nearest unwatched position — and
it did it again, to the person who had just written that sentence down.**

---

## THE PROPOSED PROMOTION *(NOT taken — this is the human's call, deliberately)*

> ⭐ **EVERY ASSIGNED OBJECT NEEDS A WITHIN-PAIR VARIATION RULE, NOT ONLY A CROSS-PAIR CEILING.**
> When a brief assigns one object per pair AND requires it be surfaced in the lyric, it must also assign
> **how the four variations differ in surfacing it** — position, speaker, syntax, or whether it is named
> at all in one of them. Otherwise the requirement is a blank to fill.

⛔ **I have not promoted this.** Promotion authority is shared and I have used it before, but this is a
correction to a rule I broke while quoting it, found by a judge I dispatched, in a run I designed —
**and that is exactly the case where the promoter should not also be the reviewer.** It is yours.

---

## THE FOUR HOLDS — for your ear, not for a repair loop

**HOLD 1 · `scripts/validate_step.py` was patched twice mid-run, by me, outside this run's directory.**
Root cause: **one measurement, four implementations.** The fence rule was re-derived at four sites and
correct at one — it hard-failed every valid package, mis-measured prompts by +8 chars, and counted both
fences as sung lines. ⭐ **I declared it repaired after fixing two sites; the QA judge found a third; a
sweep found a fourth and fifth.** Now routed through a single `is_fence()` helper with the whole history
in its docstring. Regression-swept across four runs: no file that passed now fails. **You should know
shared tooling moved.**

**HOLD 2 · P04 credits no player, and it is the one pair that owes a specific debt.** Its shape came
from a real substitution on a real record. The run's own brief says that debt is paid in the dedication.
P04's dedication credits no one. ⛔ *"If a prompt names a break, credit the drummer"* — this is the
adjacent case and it went unpaid. Everything else is clean: **33 names, 0 leaks into any
generator-facing field**, Carlton Barrett and Lloyd Knibb both named.

**HOLD 3 · P05 fails RULING A2, and the defect is in `05_pair_assignments.md`, not in P05.** I assigned
"the name is said once, **early**, by somebody else" and Ruling A2 forbids a hole in the first thirty
seconds. **Those contradict.** P05 complied with its brief; my brief contradicted the ICB. **A ruling,
not a repair — and mine to have got wrong.**

**HOLD 4 · ⭐ SIX ROOMS, ONE WORLD.** The setting audit proves the six *rooms* are distinguishable. It
never asks whether the six *worlds* are. All six songs are British, same class register, same narrator's
diction — *bloke, mate, love, mind how you go*. The beds are Jamaican ×2, American ×2, Anglo-Jamaican,
Israeli. ⛔ **No rule was broken because no rule exists.** But it is the honest residue of the Purist's
objection: if *"six costumes?"* is answered *"no, they survive transplant,"* the follow-up is *"then
what did the six idioms buy?"* — and this run's honest answer is **the arrangements only.**

---

## WHAT IS NOT WRONG — stated so the holds are readable at their true size

**0 hard-gate failures** across 24 songs · **0 Human Subject Standard breaches** — no real harmed
person, no child anywhere, and P04 removed the death entirely rather than fictionalising around it ·
**0 real names in any generator-facing field** · **0 Bandcamp phrases inside a lyric** · tri-source
declared before writing · 3 NEWS / 3 EXISTENCE · 4 AWE / 2 INDIGNATION · six distinct beds, tempos,
rooms, verse architectures, reveal engines and forbidden sections · one sung fact per song, in words ·
**0 club beds** · drive in 1 of 6 · domestic-warm 17% against a 33% cap.

⭐ **And Pelly's carried dissent — that *steady + warm + cohesive* is one decision from wallpaper — is
emphatically defeated.** P03 and P06 are verbs in all eight variations: *"I take one mint, and then I
take another one, slow"* · *"There is a pound in this coat and there is nothing left to wash… I am
running it anyway"* · *"I want the whole of it and I am having it."*

**Nothing here is unshippable. It is unfinished, and the unfinished part is mine.** 💜
