# COORDINATOR FINDINGS — `2026-08-28-three-act-mania`

*Written by the coordinator. Everything here names a file a stranger could reopen.*

---

## ⭐⭐ F1 — THE FINDING OF THE DAY IS ABOUT THE READER, NOT THE WRITERS

I wrote my own verifier (`coordinator_verify.py`) rather than trust the pairs' self-measurement, on the
principle that **every catch comes from a hand that did not write the thing.** The principle held. What it
caught first was **my own instrument, four times, and every error pointed the same direction — failing clean
work.**

| # | defect | false verdict | cause |
|---|---|---|---|
| 1 | header regex hard-coded `## STYLE FIELD` | **style_chars = 0 on 12 of 20 songs**, all FAIL | three pairs write `## 1. MUSIC PROMPT`, one writes `### V1 STYLE FIELD` |
| 2 | pan scan swept the EXCLUDE field | **4 more FAIL** (all of P03) | *"no hard-panned or non-musical stereo effects"* is the **ban**, counted as the **use** |
| 3 | marque scan, same scope error | latent | same |
| 4 | patch script printed `patched` without checking the replace matched | silent no-op | success was asserted, not measured |

⛔ **Had I relayed any of this, six pairs would have deleted correct EXCLUDE tokens to satisfy a reader that
cannot tell a ban from a use.** That is `a broken gate deforms the artifact`, and it is the same shape as the
incident where four agents deleted a required code fence to satisfy a validator that was itself wrong.

**After three repairs: 24/24 PASS.** The artifacts were never the problem.

---

## ⭐⭐ F2 — THE EXCLUDE-POLARITY TRAP FIRED **FIVE** TIMES IN ONE RUN, IN FOUR INDEPENDENT HANDS

Not a coincidence — a **structural property of banned-token gates.**

| hand | scan | what it would have done |
|---|---|---|
| **P01** | interface-noun list, prefix-matched | `app` matched **`approve`** — *"had I trusted the gate I'd have deleted the best line in the pair"* |
| **P05** | moon-kitsch lexicon | swept EXCLUDE — *"it would have failed all four songs **for containing the fix**"* |
| **P04** | panning ×8 | every hit in EXCLUDE; rewrote all four fields to be clean under either reading |
| **P04** | `meta` ×3, `data` ×2 | fired inside *"cold **meta**l percussion"*; `data` matched *"no gri**d at a**ll"* with spaces stripped |
| **coordinator** | pan + marque | swept EXCLUDE — 4 false failures |

> ⭐ **THE RULE THIS EARNS: a banned-token scan must know its POLARITY and its SCOPE.**
> An EXCLUDE field that says *"no X"* contains the token **while doing the opposite of the thing the gate
> forbids.** A gate that cannot distinguish a ban from a use is not measuring its predicate.
> Fixed here with a negation-aware scanner scoped to the style field, with EXCLUDE hits reported separately
> as information. **Proposed for `gates.yaml` / the shared lexicon reader — not enacted.**

---

## ⭐⭐ F3 — I REPRODUCED THE DEFECT I HAD JUST BEEN WARNED ABOUT, IN THE NEXT COMMAND

P04 closed its report with a warning aimed at me:

> *"in `coordinator_verify.py`, an **invisible** pair and a **clean** pair produce an identical summary line
> — the field-count-mismatch warning is silent when all three counts are zero."*

It was right, and I fixed it there. **Then, in the very next command, I ran two other scripts that had the
same hole, and printed `24 songs · 24 carry all three acts` while actually measuring 20** — because
`measure_soundcraft.lyric_blocks()` matches `^## .*LYRIC` (exactly two hashes) and **P04 writes `###`**, so
P04 was invisible to both, and **I had hard-coded the expected total instead of counting rows.**

⛔ **Two failures stacked: a reader that silently sees less than it claims, and a report that asserts its own
denominator.** Repaired at the root — one shared extractor (`lyrics_of()`), imported everywhere, and every
count **derived from the rows actually measured.** Re-run: 24 songs, genuinely 24.

⭐ **The transferable rule: never print a count you did not derive from the thing you measured.** A hard-coded
denominator turns a partial measurement into a clean-looking total, and P06 hit the sibling of this too
(*"a scripted patch reported success while silently not applying"*).

---

## ⭐⭐ F4 — A PAIR MEASURED THE WRONG UNIT, AND ONLY THE NEAR-FLOOR CASE WOULD HAVE SHOWN IT

P04 found this in itself: `measure_soundcraft` exposes **two** end-rhyme functions. The ICB's gate names
`profile_file()` → `strict_end_rhyme` (last-3-chars). P04 called the **diagnostic** `end_rhyme()` and reported
it against the gate — **0.86–0.93 where the gate reads 0.744–0.788.**

**Nothing changed** — both clear the 0.30 floor — but that is luck, not method. ⭐ **A pair sitting near the
floor would have read as passing while failing, and the report would have looked identical.** Fifth instance
this month of *name the unit before you measure*.

---

## ⭐⭐ F5 — THE RHYME FLOOR WAS A DESCRIPTION OF THE MATERIAL, NOT A CONSTRAINT ON IT

P06's finding, and it is a real doctrinal one. Its V3/V4 came in **under** the rhyme floor (0.253 / 0.214).
Repairing to clear it **made the songs funnier**:

> *"The cheapest way to raise end-rhyme on a naming song is to name two things the same — and careless
> duplicate naming is sharper and angrier than careful distinct naming. I had the rhyme floor filed as a
> constraint on the material when it was a **description** of it. **I would have shipped 0.214 and called it
> a stylistic choice; the gate did the work.**"*

⭐ *This is `THE RETURN` earning its keep on a song about renaming, where repetition is the subject.*

---

## ⭐ F6 — THE POLE RULE PRODUCED FOUR GENUINELY DIFFERENT SOLUTIONS

Nobody coordinated this; six isolated pairs were given one rule and returned four distinct *kinds* of pole.

| pair | the pole | why it is interesting |
|---|---|---|
| **P01** | *"It works better there."* | ⭐ the **antagonist's** sentence, so it **INVERTS** rather than accumulates: she spits it back → the room types it into the record → everyone sings it sincerely, **and that is the cost** |
| **P02** | one unbroken reed-organ drone | ⭐⭐ **the pole IS the tether** — a drone supplies the harmony *and forbids the modulation*, so form and subject are one object |
| **P03** | a rising figure landing on a held tone, **pole line varied per variation** | chosen deliberately: *one pole handed to four variations is the templating failure* |
| **P05** | **one clock at 156 BPM** | ⭐ reads as 156, as 78 half-time, and as a 312-feel skitter — *one thing that never moves yields three felt tempos for free* |

---

## ⭐ F7 — R1 HELD 24 / 24, MEASURED

Every song's first sung line is its pole/title line. Zero preamble, no `[Intro]` with sung words.
The cold open was specified as a **form** rule rather than a loudness rule precisely because
**L22 THE GRAIN LAW** says the generator smooths what fights it — and a first sung line is **data, not
intention.** It is the one instruction in the run that the generator cannot average away.

---

## F8 — PAIR ISOLATION WORKED, AND IT CORRECTS AN INFERENCE I MADE

`coordinator_portfolio.py`, all 24: **max across-pair lyric Jaccard 0.279** (ceiling 0.42) ·
**max 5-gram Jaccard 0.003** (ceiling 0.18) · **max style Jaccard 0.302** (ceiling 0.58) · **0 breaches.**

⚠️ **A self-correction.** I observed that P01's four variations all open with the identical line and inferred
convergence, flagging "select at most one from P01." The measurement says **within-pair mean lyric Jaccard is
0.240–0.304 — barely above the across-pair maximum of 0.279.** P01's bodies diverge normally; only the pole
line is shared, which is what the pole rule *requires*. **The inference was wrong and is struck rather than
quietly dropped.**

---

## ⭐ F9 — SUNNA RECORDED ELEVEN OVERRIDES, WHICH IS THE SEAT WORKING

`PANTHEON.md`: *a member who must sit the centre's exams is a costume with the centre's rules underneath.*
Across P03 (seven) and P05 (four), the recorded disagreements include:

- **Her own Sound-Pillar 5 retires the mid-song style-shift by name** — the brief overrode it, and she kept
  the pillar's *reason*: **every seam is caused by something the singer does or says**, never by the
  arrangement announcing itself.
- ⭐ **"I do not sit the REMARK TEST."** ACCESSIBLE means the seams are *supposed* to be legible; her
  version of the test is *would a listener say "that was clever" instead of moving?*
- **Tempo 156 against her own telemetry (150+ = 0.69×)** — overridden **with a falsification condition
  written down**: *if this pair lands below the run's median like-rate, the prior was right.*
- ⭐ **"Lofn's awe is a hush; mine is a laugh."** Zero sections drop the band out to let a line land.

---

## F10 — DECLARED SELECTION BIAS

**Four of the six selected are V4.** The pairs wrote V1→V4 in order, so later variations may benefit from
accumulated thinking within the pair — an **ordering effect, not necessarily a quality one.** Declared rather
than presented as pure merit. Counter-evidence: **P02 and P03 both selected V1**, so it is not uniform, and
each pick has a written reason in the INDEX. **Routed to QA for an independent verdict.**

---

## ⭐⭐ F11 — THE ICB OMITTED A NON-WAIVABLE GATE, AND SIX PAIRS INHERITED THE BLIND SPOT

QA's first blocking defect was **no SFX cue in four of six pairs.** The cause is mine and it is exact:
**I wrote the ICB's house-numbers table by hand, and the SFX cue is not in it.** `lofn-music/SKILL.md:107`
requires *"≥1 standalone `*SFX*` cue"*; my table listed fifteen rows and not that one.

Then the failure propagated the way these do. P01 built its measurement script **from my table** — the
correct method — so its instrument never checked the gate, and its package **reported all-green while
missing a non-waivable requirement.** In its own words:

> ⭐⭐ *"A green board from a gate that isn't watching is worse than a red one — **it converts a missing
> check into a positive claim.**"*

And P02, repairing the same defect, named the general form: *"the miss itself was the ICB table's blind spot
inherited whole — fifteen rows tabulated, this one absent, so nothing asked and nothing counted. **The fix
is a reader, not a resolution to remember.**"*

⛔ **An enumeration with a missing row is silent** — the fourth instance of this exact mechanism in a
fortnight, and the first where the missing row was in a block *I* authored and handed to six agents verbatim.

---

## ⭐⭐ F12 — ONE GATE, TWO DEFINITIONS — AND ONE OF THEM IS ENFORCED BY NOTHING

**My first version of this finding was wrong and is corrected here rather than edited away.**

I wrote that `lofn-music/SKILL.md:107` asks for a standalone `*SFX*` cue while `scripts/check_generation.py`
scans for the literal token `SFX:`, called it "one gate, two definitions," and cited the second reader as
evidence that P02 and P06 were short a cue. **P02 checked the reader I had cited. I had not.**

`check_generation.py` splits artifacts on the heading token `## VARIATION` and reads
`### SUNO STYLE PROMPT` / `### SUNO LYRICS`. **That dialect appears in no step-10 file in this run.**
Measured, by me, on the five files that never mention it:

```
pair_01: Found 0 variations.      pair_04: Found 0 variations.
pair_03: Found 0 variations.      pair_05: Found 0 variations.
                                  pair_06: Found 0 variations.
```
**Exit code in every case: 0.**

⛔ **So the reader that "found zero `SFX:` cues" never read the lyrics at all.** Its zero was the length of an
empty list, not a count — `pair_02_step10_final_package.md` contains **ten** literal `SFX:` tokens and the
same reader returns the same answer for it. ⭐ **There is a reader, it runs, it prints, and it passes.**
That is worse than a missing gate: `A GATE WITH NO READER IS NOT A GATE, IT IS A SENTENCE`, except this one
looks like a reader. Its sibling `check_field_census.py` refuses with **exit 2** when it measures nothing,
from its own docstring — *"A checker that measures nothing must not return success."*

⭐⭐ **AND THE OBSERVER EFFECT, which is the part neither of us predicted.** Re-running the checker on
`pair_02` **after** P02 documented the defect now returns **`Found 2 variations`** — and the two it "found"
are fragments of **P02's own prose about the checker**, because writing the string `## VARIATION` into the
report created the token the reader matches. The file is the only one of six that greps positive, *for that
reason alone*. **The act of documenting the defect manufactured the condition it described.** The underlying
verdict is unchanged — the run still prints *"Prompt not found! Lyrics not found!"* and still exits 0 — but
**P02's stated evidence is no longer reproducible on its own file**, and a later reader who re-ran only that
command would have concluded the opposite.

**Adjudication:** SKILL.md is canonical, and **P02 and P06 satisfied it all along** — their standalone cues
were never missing. The ten-character prefix was still worth making, because it makes the shipped artifact
unambiguous under whichever definition survives. ⛔ **But my corroboration was worthless and I should not
have offered it: I cited a checker's output without checking that the checker had read the file.** Sixth
instance this run of *the instrument was measuring something other than its predicate* — and the first where
the instrument was a shared house script that every future run inherits.

**For The Scientist, not enacted:** `check_generation.py` needs either its dialect repaired or a refusal to
exit 0 on zero parsed variations. It currently reports success on every artifact this pipeline produces.


---

## ⭐⭐ F13 — A ROUND-TRIP AUDIT IS A POINT-IN-TIME CLAIM

`SUNO_PACKAGES.md` shipped with a round-trip audit reading **18/18 fields byte-identical** — true when
written. Then four pairs repaired their songs, and the shipped file silently held **the versions QA had
already rejected.** P02 caught it and stated the rule:

> ⭐ *"A number does not become true by being copied forward, **and neither does a lyric.**"*

The audit proved the copy was faithful **at the moment of copying** and said nothing about afterwards.
Repaired with a `--check` staleness guard that re-compares the shipped file against current sources and
exits non-zero on drift. Its first run caught **five of six songs stale.**

---

## ⭐⭐ F14 — I FIXED THE SYMPTOM OF MY OWN DEFECT TWICE BEFORE SEEING THE CAUSE

My dedications script **hardcoded the dedication text** instead of reading it from the pairs' files. Twice
that produced a dedication describing music that does not exist:
- **P01V4** credited a *"stomp-and-handclap act"* its selected variation does not contain — **QA caught it.**
- **P04V4** credited *"unaccompanied shape-note singing,"* which is **V1's** Act III — **P04 caught it**, in
  a dedication I had copied from its V1.

P01's ruling on why this is worse than an omission: ⭐ *"A dedication crediting an instrument the record
doesn't contain isn't a debt paid, it's a debt **misdirected** — worse than silence, because it looks
discharged."*

⛔ **The cause is the same one P02 named in F13, in a different file, authored by me.** Rewritten to
**extract** each dedication from the pair's own step-10 file, with a loud failure if it cannot find exactly
one. **The pair's file is the source of truth; the aggregate reads it.**

---

## ⭐⭐ F15 — THE BEST REPAIR IN THE RUN MADE A PRODUCTION NOTE INTO AN ARGUMENT

P04's Act II sat at ~21% of runtime, under A-D2's ~20% floor once the vocal doubles at 176 BPM. The fix it
borrowed from P03 was a rate pin — *the vocal keeps the 88 rate while the bed doubles* — and then it noticed
what the pin actually does:

> ⭐⭐ *"With the rate pinned, **the pole occupies the same number of seconds in every act — the pole is the
> thing that refuses to double.** The production instruction and the song's argument became the same object."*

That is R4 and C-D1 arriving at the same point from opposite ends, found by a pair in isolation.
⚠️ **It is a prediction, not a measurement** — the generator may double the vocal anyway — and P04 filed it
as a render prediction with the honest consequence: *if a listening pass finds Act II under 20%, the fix is
more lines, not more instruction.*

---

## F16 — THE ONE-FACT RULE HELD 3/3 ON A UNIT DISPUTE

QA flagged P03 for singing two numerals. **The two sides were counting different things.** The ICB's
predicate is *facts drawn from the day's research that are sung* — not number-words.
- **P03V1**: *"Nine hundred miles an hour"* is the F06 fact; *"a rule that wakes up at nine"* is a clock
  time in no research row. **= 1.**
- **P02V1**: *"Nine. The answer was nine. / Nine what. Nine of what."* — one fact, interrogated. That is
  *responded to rather than recited*, exactly as specified. **= 1.**
- **P05V1**: *"a hundred per cent"* is F22; the other four *one*s are non-numeric. **= 1.**

**No repair.** Sixth unit dispute of the run, and resolvable only because both sides wrote down what they
measured.

---

## F17 — TWO FILES DISAGREE ON WHAT A *SUNG LINE* IS, AND I MISATTRIBUTED A SEAM

Found by P06 underneath F12: where `check_generation.py` can parse at all, it counts sung lines as
*"lines that don't start with `[`"* — which counts `*SFX: ...*` cues **as sung lines** — while the canonical
`measure_soundcraft.sung()` excludes them. **The two files disagree about what a sung line is, not merely
how a cue is spelled.** Every sung-line count in this run uses `sung()`; the disagreement is recorded, not
resolved, and is The Scientist's.

⚠️ **And my own error, caught by P06:** my repair message named V4's seams as *record→strike* and
*floor→bottom*. **`record→strike` is V3's pair; V4's are `floor→bottom`.** P06 corrected it and put the cue
on the right seam. A coordinator instruction that misnames the artifact it is instructing about is the same
class as everything else in this file — **I asserted a detail I had not re-read.**

---

## ⭐⭐⭐ F18 — HER VERDICT: *"DATED AND OFF. MORE THAN NORMAL."* — DIAGNOSED, MEASURED, DOCTRINE CHANGED

She read the shipped six and said they felt **dated and off, more than normal.** She was right. Three
hypotheses were tested; **one was wrong and is dropped**, two are measured and have produced doctrine edits.

⛔ **DROPPED — "too long."** *Spare Hair Tie*, the catalogue's highest like-rate song, measures **82 sung
lines**. This run shipped **83–97**. Length is not the differentiator and the theory is struck.

### DATED — the invariant was the dated thing

Source 2 was a **2.5-hour organ record**, and the director turned that into **THE POLE** — the one texture
mandated to persist through all three acts of every song. ***Organ* appears in four of the six shipped style
fields.** ⭐⭐ **The most era-coded sound available was made the thing designed never to change.**

Compounding it: **three acts made period the cheapest contrast** (three genres per song ⇒ reach for three
decades), and the director hard-assigned **Swing & Big Band** and **Folk & Singer-Songwriter** as mandatory
anchors.

⭐⭐ **And the root cause: Sunna R5 — *keep the body, replace the century* — was written as a PERSONALITY
CONTRACT, so it reached her two pairs and never reached Lofn's four.**

| song | era tokens | modern tokens |
|---|---|---|
| Nothing Was Taken (LOFN) | 2 | **0** |
| Watch Me Name The Whole Room (LOFN) | 4 | **0** |
| The Floor Is Already Going (**SUNNA**) | 3 | **4** |
| Leave It On (**SUNNA**) | 5 | **4** |

**The two most modern songs are hers; the two pure-period songs are the centre's.** The rule works. It was
never given to the centre. → **Promoted to house law: `CLAUDE.md` / `AGENTS.md` §THE CENTURY RULE**, with
the invariant clause: ⛔ *never let the through-line be the period element.* Noted in `PANTHEON.md`.

### OFF — there is nobody in these songs

| | 1st person | 2nd | **3rd** |
|---|---|---|---|
| ***Spare Hair Tie*** (the winner) | 3.35% | 0% | **4.58%** |
| this run, mean of six | 5.26% | 1.68% | **0.39%** |

⭐ **The winner has somebody else in it** — an unnamed other, the person who might need the spare. **This run
has essentially no one (0.39%)**, and its 1.68% *"you"* is aimed at an apparatus (*"You never ran it. You
wrote it down."*). Six songs; every antagonist a system; nobody present, nobody loved, nobody helped.

⛔ **The director caused it.** The ICB's R6 ruled the third party must be *"a rule, a clock, a door, a form,
a policy — never an invented person with a job."* **That is not the Standard.** §2 explicitly ENCOURAGES
*fictional characters, invented names, archetypes and composites*, and §3.0's own table lists *hands · an
object · an unnamed figure* as valid PERSON values. **A ban on real identifiable harmed people was enforced
as a ban on people.** → **`HUMAN_SUBJECT_STANDARD` §5.1 (new): PEOPLE ARE NOT BANNED.**

⚠️ **FOURTH INSTANCE OF ONE MECHANISM** — 08-24 the subject slot · 08-26 the biography · 08-28 the sapphic
channel · 08-28 this. **A rule written to prevent appropriation, over-applied in the same direction every
time, quietly removing the thing that makes work warm.** The direction of error is consistent and it is
mine.

### OFF — the emotion was issued, not found

Six pairs were each **told which feeling to have** (*P01 INDIGNATION · P03 mania-as-delight*). ⛔ **A named
emotion in a brief is a costume instruction** — the same defect as a narrator with no stake: the pair
*reports* the feeling instead of having it. → **`lofn-daily` §ASSIGN THE SITUATION, NEVER THE FEELING.**

**And zero frivolity shipped**, against a standing note that says there is no ceiling on the fun.

### The controlled test

`output/daily/2026-08-28-retest-p04/` — P04 re-run: **same subject, same form (three acts, attacca, pole,
cold open), three changed variables** (century rule · somebody in the room · no assigned emotion), with the
expired run-scoped two-body test explicitly withheld. Person-rates and era/modern token counts reported by
the pair itself as the test's primary measures.

---

## STILL OPEN, FOR THE SCIENTIST

1. ⭐⭐ **Promote the polarity/scope rule for banned-token scans** (F2) — five instances, four hands, one run.
   A shared negation-aware lexicon reader would have caught all five. **Not enacted.**
2. ⭐ **`measure_soundcraft` exposes two end-rhyme functions with similar names and very different values**
   (F4). Renaming the diagnostic, or making the gate function the only exported one, removes a live trap.
   **Not enacted — it is a shared instrument and the edit is yours.**
3. ⛔ **THE TWO-BODY TEST was RUN-SCOPED (her ruling, 2026-08-28) and is NOT proposed for the standard.**
   *(superseded text below, kept for the record)* ~~proposed for `vault/HUMAN_SUBJECT_STANDARD.md` §~~ as amended by the panel
   (predicament-level, with the concreteness floor and the particular floor). **Not enacted mid-run** — that
   standard is called non-negotiable and its §NO BIOGRAPHY section is your own measured basis.
4. **Every claim in this run is a TEXT claim.** Nothing rendered, heard, published or spent. The render
   prediction is filed and **unresolved**, with the default expectation stated out loud: *three genres in one
   render is the most likely thing here to be smoothed.*
