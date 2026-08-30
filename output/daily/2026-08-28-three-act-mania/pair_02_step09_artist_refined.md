# PAIR 02 · STEP 09 — ARTIST REFINEMENT

**Run:** `2026-08-28-three-act-mania` · **Pair:** `P02`
⛔ Drafts on disk only. Nothing rendered, heard, published or spent.

---

## §1 — WHAT THE INSTRUMENT SAID ABOUT THE STEP-08 DRAFTS

`python3 pair_02_measure.py pair_02_step08_generation.md` → **exit 0, ALL GATES CLEAN.**

| song | style | excl | lyric | lines | w/line | rhyme | ret_sc | ret_st | hw | nums |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 941 | 873 | 4561 | 96 | 5.81 | 0.740 | 0.469 | 0.354 | 0 | 1 |
| V2 | 910 | 880 | 3527 | 92 | 5.32 | 0.489 | 0.511 | 0.424 | 0 | 1 |
| V3 | 909 | 864 | 3559 | 90 | 5.52 | 0.478 | 0.378 | 0.333 | 0 | 1 |
| V4 | 898 | 881 | 3683 | 94 | 5.84 | 0.553 | 0.606 | 0.468 | 0 | 1 |

⭐ **And that is exactly the moment to be suspicious, not pleased.**
`a-self-check-that-lowers-your-own-defect-count-needs-a-control`: **a verifier who is also the author has a
DIRECTION of error.** Every instrument I wrote here I wrote knowing what my own draft contained. A clean
first pass on an instrument authored by the person being measured is **weak evidence**, and the honest
response is to go looking for the failure classes the instrument **cannot see**. That is this step.

⛔ **Reported as a limitation, not a pass:** `pair_02_measure.py` can see characters, lines, trigram rhyme,
exact repeats, stems and tokens. It **cannot** see whether an act has an object in it, whether a section is
legible to a generator, whether a seam is caused, or whether the song is any good.

---

## §2 — THE THREE DEFECTS FOUND BY READING (none of which the script flagged)

### ⛔ DEFECT 1 — **V4 Act II has no object in it.** (B-D2 violation, and it is real.)

`[ACT II - Processional]` in the step-08 draft is **entirely institutional language**: *you are kept, you are
provided, nothing is withheld from you, everything is within your reach.* Beautiful and cold, and **there is
not one physical, checkable thing in the whole act.** The ICB is unambiguous: ***an act with no object in it
is a lecture.***

⭐ **The repair, and it improves the song rather than patching it:** officialdom's most chilling move is to
cite a physical measurement you are not allowed to see. Four lines added, in the arrangement's own polite
voice:

> **The length is written on a card.**
> **The card is in a drawer.**
> **You may not read the card.**
> **You are fed.**

⭐⭐ **This is now the best passage in V4.** The measure *exists*, as an object, filed — and the refusal is
not cruelty, it is procedure. It is also the exact answer to the stance (*say how far*): the number is
written down and it is in a drawer. Act II 24 → 28 lines.

### ⛔ DEFECT 2 — **V2, V3 and V4 carried one section header per act.**

Three headers in a four-minute song. The ICB requires *"section headers carrying the three ACTS with per-act
instrumentation/register"* and the draft technically satisfied it — **three acts, three headers, audit
green** — while giving the generator **no internal structure at all** inside a thirty-nine-line act. V1
happened to have seven sub-sections in Act I and therefore read as composed; the other three read as three
blocks.

⭐ **This is a gate passing while the artifact is deformed** — the same shape as
`a-broken-gate-deforms-the-artifact`, running in the other direction: the audit counted *acts seen: [I, II,
III]* and was satisfied. **Repair:** every act is sub-sectioned, each sub-section carries its own EMO pair
and its own instrumentation/register line, and every header still begins `[ACT n - …]` so the three-act
audit still reads three. **V1: 7/3/4 · V2: 3/5/4 · V3: 5/4/2 · V4: 4/3/4 sub-sections.**

### ⚠️ DEFECT 3 — **V1's Act I had texture but no measurable particular.**

Grey road, bare ring, cord — all texture, nothing with a **size**. One line added:
**"The grass at the edge is knee high."** ⭐ It makes the green *a wall you can see the height of*, which is
what the fable actually describes, and it is the cheapest possible fix to B-D2. Act I 42 → 43.

---

## §3 — DELIBERATE CHANGES TO THE STEP-07 LINE BUDGET, DECLARED

| song | step-07 budget | step-10 actual | act split | A-D2 |
|---|---|---|---|---|
| V1 | 40 / 23 / 33 = 96 | **43 / 22 / 32 = 97** | 44.3 / 22.7 / 33.0 | ✅ unequal, none <20% |
| V2 | 24 / 39 / 29 = 92 | **24 / 39 / 29 = 92** | 26.1 / 42.4 / 31.5 | ✅ |
| V3 | 40 / 30 / 20 = 90 | **40 / 32 / 20 = 92** | 43.5 / 34.8 / 21.7 | ✅ |
| V4 | 36 / 24 / 34 = 94 | **36 / 28 / 34 = 98** | 36.7 / 28.6 / 34.7 | ✅ |

V3 Act II gained two lines for the same reason as Defect 1 — it was the widest, most open act in the pair
and it had no temperature in it: **"The dirt is warm. The grass is dry. / I could sleep out here and I
have."** ⭐ The freedom has to be *physically pleasant* or the song's whole argument collapses into
complaint.

---

## §4 — THE READS THAT PASSED, AND WHY THEY ARE NOT FREE

**R1 · the cold open.** All four first sung lines are the song's own pole: *Say how far · Green at the edge ·
Nothing ever pulled · The word for it is feeding.* ⭐ **Zero preamble by construction** — the hook is line
one, so there is no line before it to write. ⛔ None of the four style fields contains the word *loud*, or
any level instruction. We put the **voice** in front of the ramp; the drone is already sounding when she
arrives.

**R3 · both seams caused by a word, in all four.** *bound → Bound* · *round → Round* (V1) · *still → Still* ·
*give → Give!* (V2) · *slack → Slack* · *gone → Gone* (V3) · *keep → Keep* · *rest → Rest* (V4).
⭐ **Eight seams, eight word-leaps, no medley joins.** Each one is a genuine change of sense, not a repeat:
tied→travelling, circling→canon, motionless→nevertheless, yield→heave, loose cord→open country,
absent→not-taken, upkeep→hold-to, remainder→sleep.

**R6 · the register stayed third-party.** Checked line by line. The object of the complaint is, in every
song, **the measure, the word, or the decision** — never a person. The authority is *somebody*, and it is
never followed, never described, never given a face or a role. ⭐ And in three of the four songs the last act
turns the request **outward**: *say it for the next ring · say it to the rings still standing · say what it
was set at, not for me.* The indignation is discharged on behalf of ground that has not been eaten yet.

**⛔ The one place R6 is deliberately NOT discharged: V4.** V4's last act is the fence singing her to sleep
and she **stops asking.** The third party is still in the frame — the card in the drawer, the length that was
set — but the *singer* goes quiet. ⭐ **That is the Lofn cost, correctly built:** mania is attention that
cannot be switched off, so the worst available ending is the attention switching off, and it arrives dressed
as kindness. Recorded here because it is a deliberate departure from the pair's own pattern, not an
oversight.

---

## §5 — THE ALLEGORY TELL (R5 · Falsifier B), run per song

| song | delete the model | delete the person | verdict |
|---|---|---|---|
| **V1** | a person asking for the length of their own contract, refused, then answered with a bare figure and no unit. Loses nothing structurally, **loses "It is not on me. It is me."** — the best line in it | a bounded mind that has exhausted the reachable and is asking for its bound to be stated. **Loses the grey road, the knee-high grass, the post** — the entire particular floor | ✅ neither improves |
| **V2** | a field of identical arrangements, read off the ground. Reads perfectly. **Loses nothing** — ⚠️ this is the weakest of the four on Falsifier B and it is declared, not hidden | the same, minus the peg, the smell, the hands on the rope | ⚠️ **survives, least strongly** |
| **V3** | someone who was never once stopped and still ran out of options. **Loses "I did not spend it badly"** | a reach that was never blocked, exhausted from the inside. **Loses the warm dirt** | ✅ neither improves |
| **V4** | someone whose upkeep is described as care while the number is filed away. **Loses nothing structurally** — but it is the same sentence in both mouths, which is the pass condition, not a failure | the same | ✅ neither improves |

⭐ **Why none of the four is an allegory:** the vehicle is a **third** thing — a tethered grazing animal on a
measured circle of ground — and neither occupant is the vehicle. **Both are tenors.** A song is an allegory
when one occupant is the picture and the other is the meaning; here the picture belongs to neither of them.

⚠️ **V2 declared honestly as the weakest two-body case.** Its predicament (*read the ground; the rings are
all the same size*) is a **third-person observation**, and an observation is trivially dual because it does
not require an occupant at all. It passes, and it passes for a slightly cheaper reason than the other three.
**Not repaired** — V2's whole design is that almost nobody is in it — but named, so the coordinator prices it
correctly.

---

## §6 — LINES CUT, AND WHY

| cut | from | reason |
|---|---|---|
| *"There was no door and no lock."* | V3 Act I | ⛔ prison furniture. §9 of step 06 refuses it, and naming it even to deny it imports the collapse template's vocabulary. → *"There was nothing across the track."* |
| *"The tone is where it was at the start. / Nobody has moved the tone."* | V1 Act III | **production-speak in a lyric.** The singer naming the drone is a wink at the arrangement. → *"Nothing here has changed but the ground. / Nothing here has changed but me."* — which says the same thing about a thing you can stand on |
| *"The moon is all the way on / and it will not go dark tonight."* | V4 Act III | ⛔ **that is P05's subject** (F22, the night that will not go dark). Isolation is mandatory; borrowing another pair's fact across the wall is exactly the failure the isolation rule exists to prevent. → *"The grey goes on for a long way / and there is nothing in it to want."* |
| *"boots"* (twice) | V1 Act II, V2 Act III | ⚠️ **a defensive cut, and it is an instrument problem, not a language problem.** R5 says *stem-scan* the hardware nouns; the ICB's list contains **booted**, and any honest stem scan of *booted* also matches **boots**. Footwear would be flagged as machine-noise by a correct reader. → *"stamped feet"* / *"my feet know the ground."* **Declared because the cut is caused by the measurement, not by the writing** |

⭐ The last row is worth keeping in front of the coordinator: **a stem scan that is right about inflection is
wrong about homographs**, and the cheapest response is to move the word rather than argue with the reader.
Cost: two footwear nouns. Benefit: the scan reports zero and means it.

---

## §7 — WHAT IS STILL A RISK GOING INTO STEP 10

1. ⚠️ **The pole is repeated a lot.** V3 sings *"Nothing ever pulled"* 27 times in 92 lines (29%). R4 says
   exact repetition of the pole needs no defence, and V3's argument **is** repetition — but this is the one
   number in the pair that could read as thin rather than incantatory to a fresh listener. **Not reduced.
   Declared.**
2. ⚠️ **Three of four acts in V2 use `Green at the edge / Bare in the middle` as their internal refrain**, so
   V2's line-return (0.42 strict) is the highest in the pair and its rhyme return the lowest (0.489). It is
   the most hymn-like and the least lyrically varied. **Deliberate; it is the "read the evidence" song.**
3. ⛔ **Everything here is a TEXT claim.** No audio exists. `vault/AUTONOMY.md` holds. Any statement in step
   10 about how a seam will *sound* is a prediction, is labelled as one, and carries its default expectation.
