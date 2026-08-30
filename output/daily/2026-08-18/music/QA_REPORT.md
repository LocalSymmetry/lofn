# Lofn QA Report — `output/daily/2026-08-18/` (MUSIC, 24 songs) — ⚠️ **NON-CANONICAL RUN**
**Personality:** SUNNA `v.2026.08.R3` · **Judge:** clean-context adversarial QA subagent (wrote none of it)
**Scope of this audit:** the things a counter cannot see. The coordinator's mechanical pass (char caps,
line counts, similarity, sweeps, dedication lengths, club/anger audits) was **not** redone.

---

## Verdicts

| | |
|---|---|
| **Pipeline Integrity** | ⛔ **NON-CANONICAL** — the NO-SKIP rule fired. See §1. |
| **Package** (modality contract) | **REPAIR REQUIRED** — EMO-header contract, SFX cue, two receipt defects. See §8. |
| **Human-Subject** | ✅ **CLEAR** — every person in all 24 lyrics is invented; Google/Spirit appear only in notes. |
| **Overall** | 🟡 **HOLD** |

⭐ **Read the HOLD correctly.** The **writing is SHIP-grade** — this is the strongest lyric set I have
audited under this personality, and R3's laws largely landed in the artifacts rather than merely in the
claims. The HOLD is **procedural plus a short, cheap repair list**. Nothing in §1 is a quality judgment and
nothing in §9 requires a re-write. Per the skill: *borderline defaults to HOLD, and the standard does not
float* — but the fix here is hours, not a re-run.

---

## 1. ⛔ PIPELINE INTEGRITY — THE NO-SKIP RULE FIRED, AND IT IS NOT DAILY CONVENTION

`music/` contains only:

```
01_core_seed_and_panel.md · 05_pair_assignments.md · pair_0{1..6}_step10_revision_synthesis.md
```

There is **no step 06, 07, 08 or 09 artifact for any pair**, and no `CREATIVE_CONTEXT.md`,
`GATE_REPORT.json`, `INDEX.md`, `RUN_STATE.md` or `TOP6_PASTE_READY.md`.

⚠️ **I checked whether this is daily convention. It is not.** All three most recent dailies on disk carry
the full per-pair editorial spine:

| Run | step06 | step07 | step08 | step09 | step10 | CREATIVE_CONTEXT |
|---|---|---|---|---|---|---|
| `2026-08-14` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `2026-08-15` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `2026-08-17-dialtone2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **`2026-08-18`** | ⛔ | ⛔ | ⛔ | ⛔ | ✅ | ⛔ (ICB inlined into `05`) |

`lofn-qa` §What every PASS must clear is unambiguous: *"steps 07, 09 and 10 artifacts exist for every
non-quarantined pair… A run missing its editorial spine is **NON-CANONICAL** — the Overall verdict can
never be SHIP and the run cannot be published under Lofn's name."*

**And now the honest other half, because a rule applied mechanically is not judgment.** The failure that
rule exists to catch (2026-06-28: *gates measured structure and could not see nobody wrote the
arrangements*) **demonstrably did not occur here.** Each `step10` file is 37–51 KB and visibly contains the
spine's output: per-song themes and forms (07), style/exclude/lyrics (08), and a documented artist
refinement pass with **named cut lines** (09) — e.g. `pair_01:580` cut *"I built the whole thing to hear
that"*; `pair_02:173` cut *"I am going where I stand"*; `pair_03:888` cut *"the factory shut in 'eighty-four
and left us all the tools"*; `pair_05:772` cut *"I'm not going to make a speech"*. The editorial spine was
**executed**; it was not **filed**.

⛔ **This is The Scientist's call, not QA's, and I am naming it rather than deciding it.** Two clean routes:
1. The pair agents re-emit `step06/07/09` from material already inside their own `step10` files (cheap, and
   restores canonicity), **or**
2. `lofn-daily` records an explicit, dated exemption for single-file pair synthesis — in which case this
   run is canonical under the new convention and this section becomes historical.

Until one of those happens the header stands and Overall cannot be SHIP.

**ICB integrity.** The ICB exists, complete and verbatim, at `05_pair_assignments.md:21-55`, and every pair
file cites it by path (P02 states *"read from disk in full"*). But because steps 06–09 do not exist as
files, **the "unbroken verbatim prefix at the head of each step's prompt" check cannot be run** — there are
no step prompts on disk to check it against. Panel **voice count = 18** ✅ (6 + 6 + 3 named + 3 carried
Hyper-Skeptics). The literal `(after ` count is 11, correctly lower, because 7 seats are role-anchored with
no individual named — that is right practice, not drift.

**Blind golden+decoy calibration: NOT RUN, and I will not pretend otherwise.** The skill requires the
*coordinator* to assemble the shuffled triple. A judge who assembles his own blind set is not blind. Flagged
as a genuine gap in this audit's own calibration.

---

## 2. ⚡ THE FISHER FALSIFICATION — RUN FOR REAL

> *R3, on the record: "if the catalogue holds a run of songs whose named-wolf lines are interchangeable —
> swap the dark line from track 2 into track 5 and lose nothing — then this was a formula, not a repair."*

**The 16 wolf-carrying songs and their named wolves:**

| | P01 (the tide) | P02 (the string) | P05 (the record of you) | P06 (the part) |
|---|---|---|---|---|
| **V1** | `Tide's at nine.` | `Four o'clock, it took the skin / Off the top of my thumb` | `Nine years of me in a folder / And he bought it in his seat` | `Support ended in March.` |
| **V2** | `He's here early because the tide's at nine.` | `Two hundred feet, and the spool went bare / And it jumped straight out my hand` | `Three in the morning… / That's the night they marked me` | `They printed the last date on the box. / It's on the box.` |
| **V3** | `Tide's at nine and I want my go.` | `The wind stopped at ten past four` | `"Please allow forty-five business days"` | `Manual says end of life. / Manual says it in a box.` |
| **V4** | `Tide's at nine, same as Monday. / It's on a board by the kiosk.` | `Then the line went / And it went over the trees` | `I read page nine.` | `Part it wants is not made anymore. / Took one out of mine.` |

### 2a. ACROSS pairs — **PASSES, decisively**

I attempted every cross-pair swap. They all break, and they break on **material**, not on tone:

- P06-V1 `Support ended in March.` → into P01-V1: *"…Nobody asked for a ledge. / Support ended in March. /
  I'm doing the other window."* There is no software on a beach. **Breaks.**
- P01-V1 `Tide's at nine.` → into P06-V1: *"They said replace it. / They said it's finished. / Tide's at
  nine. / Pass me the small one."* **Breaks.**
- P05-V3 `Please allow forty-five business days` → into P02-V2 (the kite): **breaks** — there is no
  institution in that song to say it.
- P02-V1 `it took the skin off the top of my thumb` → into P05-V4 (the email): **breaks** — nothing in a
  terms-and-conditions song has a body.

⭐ **Verdict at set level: this is a repair, not a formula.** Four pairs, four genuinely different mechanisms
of harm — a tide, a physical injury, an acquisition, an end-of-life date.

### 2b. WITHIN pairs — ⛔ **P01 AND P06 FAIL IT**

This is where the test bites, and both pairs' self-checks claim a pass they do not have.

**⛔ P01 fails on a byte-identical string.** All four songs sing `Tide's at nine` (V2 adds *"the"*).
Verified: `pair_01:87, 276, 463, 669`. The pair's defence (`pair_01:15-19`) is that the *function* differs.
Test it: swap V1's wolf into V3 and V3's into V1 —

> V1 becomes: *"Mate. Yes. / Tide's at nine and I want my go. / I'm doing the other window."* — still works.
> V3 becomes: *"Straight through the door I did with a spoon. / Tide's at nine. / It was a good castle."* — still works.

**The dark line itself is fully interchangeable across the pair; only the trailing clause is not.** P01 has
written its wolf as a **cross-song refrain**, which is a legitimate and even attractive device — but it is
exactly the artifact Fisher's seat named, and calling it four events when it is one event stated four times
is the claim I am rejecting.

**⛔ P06 fails on synonymy.** `Support ended in March` / `They printed the last date on the box` / `Manual
says end of life` are **the same fact in three phrasings**, in four songs about the same workbench. Swap V1's
into V3's slot: *"Battery is a flat grey brick. / Support ended in March. / Fan is going like a good dog."* —
loses nothing. P06's own note (`pair_06:888-892`) claims *"swap any of these into another of the four and you
lose the specific object it names."* That is true of the **nouns** and false of the **function**. Three of
four are status reports about one machine.

⭐ **The one P06 variation that survives the test is V4** — `Part it wants is not made anymore. / Took one out
of mine.` — because there the wolf has a **cost she pays**, not a status she reports. That is why V4 is my
P06 selection.

### 2c. The number that matters

**7 of 16 wolf lines are status reports rather than events** (P01 V1/V3/V4; P06 V1/V2/V3). R3 says the wolf
must be *"an event… a thing with a day, an object, a person, a room."* A published tide-hour and a support
end-date **are** pointable, so this is not a ceiling violation — but P02 and P05 show what the rule can
actually buy (a spool jumping out of a hand; a night at three in the morning when a machine marked you), and
against that, half the wolves in the run are weather reports with a timestamp.

### 2d. ⭐⭐ AND THE FORMULA IS SOMEWHERE ELSE ENTIRELY

R3 pointed the falsification at the wolf. **The formula in this run is in the RESOLUTION.**

**(i) P06 signs off all four songs with the same sentence, noun swapped:**
> `Best Thursday of my life.` (V1:157) · `Best post I've ever had.` (V2:360) ·
> `Best Tuesday I've had.` (V3:558) · `Best week I've had in ages.` (V4:758)
> — plus `I am having the best night.` ×3 inside V1's chorus (72/98/136), `Best night I've had in ages.`
> (V4:659) and `Best bit of the whole week.` (V4:710).

Nine instances of one construction in one pair. Apply Fisher's own test to it: swap `Best Tuesday I've had`
into V1 and `Best Thursday of my life` into V3 — **you lose nothing at all.** This is the pair's warm return
running on a template. P02-V4 reaches for it too (`Best thing I have seen / Since I do not know when`, 666).

**(ii) 10 of 24 songs land the F17 return on hot food or a kettle.** `Chips on the way up the hill` (P01-V3)
· `Chip shop's got the fryer on` (P02-V2) · `Chips in the car park / Vinegar on the sleeve` (P02-V3) ·
`And there's tea if you want it` (P03-V1) · `And there's soup on` (P03-V2) · `And there's toast if you want
it` (P03-V3) · `Toast on. Lamp low.` (P06-V1) · `Kettle's on. Where's my knife.` (P06-V2) · `Kettle's boiled
… Toast on. Big coat on.` (P06-V3) · `Chips. Vinegar. Cold hands.` (P06-V4).

⚠️ P03 is tightest: **two of its four use the near-identical slot `And there's ___ if you want it`** in the
same position of verse 4.

**(iii) Two radiants in two different arms are the same radiant.**
> `My dad put his hand on the glass in ninety-six` (P03-V1:150) · `My dad had the washing machine on this
> table.` (P06-V1:164)

Same kind — *dad + domestic object + it is still here*. ⭐ **The cross-pair distinctiveness ceiling cannot
see this**: two four-word phrases inside 2,000-word lyrics do not move token Jaccard, which is why the
measured worst-lyric similarity is a comfortable 0.328 while the run's two most load-bearing single lines
are siblings. **Transferable finding: the distinctiveness gate must measure the radiant against the other
23 radiants directly, not the whole lyric bodies.**

---

## 3. ⭐⭐ DID R3 ACTUALLY LAND? — read as a listener, not a linter

### 3a. THE BAND DOES NOT KNOW / carefree music, cutting words — ✅ **LANDED**

Anxiety did **not** leak back into the arrangement. Evidence from the style fields themselves, not the notes:
zero distortion across 24; **every one of the 24 EXCLUDE fields actively bans** `distortion-pedal swagger` /
`fuzz guitar` / `unresolving riser` or equivalents; `no standing tension lane` is honoured — the only tension
devices anywhere are single silent beats and one-bar rests, each of which lands and is spent.

The three hardest cases all hold:
- **P05** is the INDIGNATION seat and there is **nothing cross in any of its four arrangements** — walking
  bass, handclaps, tambourine, gang shouts, a terrace chant. A song about being sold at auction that is
  built like a party is exactly the transplant R3 specified.
- **P06-V3** is a countdown song with **no riser and no ticking-clock SFX** (both explicitly excluded), and
  the band is at *full density with claps* at the moment the machine dies.
- **P06-V4** is a secret-gift song with **no ballad section and no tender bridge**, both banned in the
  EXCLUDE field because that is the variation a generator would most want to soften.

⚠️ **Two arrangement risks the text gates cannot see** — both are lyric lines that will *instruct the
generator* to do the thing the doctrine bans:
1. **`Nobody breathe.`** (P01-V1:120) and **`Nobody breathe on the turret.`** (P01-V2:306). These are lyric
   lines, so no header sweep catches them — but Suno will very plausibly drop the band. They sit on the
   *craft* moment, not the wolf, so no rule is broken as written; the render is where this is decided.
2. **`hang on.`** (P04-V3:519) followed by `One. Two. Three. Four.` — a written stop-and-listen. The header
   correctly instructs *"the band keeps playing the good evening straight"*, which is the right defence, but
   the lyric is still a stop instruction.

### 3b. THE REGISTER NEVER MOVES — ✅ **LANDED, and P05 is the model**

No song signals its dark line with a hush, a drop-out, a breathy delivery or a half-time turn, and **no
section header instructs reverence anywhere in 24 songs.** P05 does it best, by instructing the shut in
writing at every place it could open: *"identical delivery, same brightness, nothing dropped"* (V2 header),
*"nothing gets quieter"* (V4 bridge), *"exactly the chorus voice"* (V4 seam), *"no hush, band unchanged"*
(V1 outro). P01 and P06 take the opposite route — a **single EMO value on every header in the song**, so no
tag can request a hush. Both routes work.

### 3c. THE REMARK TEST — ⚠️ **LANDED WITH THREE NAMED EXCEPTIONS**

Devices that announce themselves, all 24 swept:

1. ⭐ **`You should not be able to dance to this.` → `Here's the chorus.`** (P05-V4:689-691). This is the
   most self-announcing moment in the run and a listener **will** register craft. ⭐ **It is licensed** — the
   Container Clause, spent once, exactly as `sunna.yaml` R3 permits, with the naming genuinely audible in
   the writing (the chant sandwiches it; the byte-identical chorus refuses to adjust). **I rule it correct
   and it is the best moment in the run.** ⛔ But `pair_05:783` claims *"Nobody says 'that was clever'"* —
   that claim is false and should not stand in the file. Licensed ≠ unremarked.
2. ⛔ **`I'm not insulted. I'm appraised.`** (P05-V2:315) and `I've been called worse things than valuable /
   By people who weren't paying` (316-317). These are **constructed epigrams** — antithesis, a pivot on one
   word, a punchline rhythm. `pair_05:387` clears them as *"situations."* They are not situations, they are
   **sentences**, and R3 rule 1 is literally *"depth lives in the situation, never in the sentence."* This is
   the run's densest patch of writerly lines and the self-check cleared it wrongly.
3. ⛔ **`Put them on top of each other!`** (P04-V2 chorus, ×8). See §7 — this is the run's worst
   remark-test failure and it is in a hook.

Cleared on inspection: P06-V3's tautology `Eleven minutes is eleven minutes` (six words a person actually
says when refusing to be sad on cue) · P04-V1's backwards year (reads as a song counting months) ·
P03-V1's whole conceit (the mechanism is never once explained) · P03-V2's `And the dimmer isn't wired / To
the lamp, it's the hall / … / Because we like the hall bright` (a situation, perfectly executed).

### 3d. THE WOLF IS AN EVENT THAT RESOLVES — ✅ no unresolving moods anywhere

Every one of the 16 wolves resolves inside its own song: taped and back out (P02-V1), caught and let out
again (P02-V2), walked out and got it (P02-V3), a round of drinks (P02-V4), the auction cleared same-day
(P05-V1), she read page nine (P05-V4), the screen goes white (P06-V1), the part slots in (P06-V2). **Zero
standing tension lanes, zero unresolving risers, zero weather systems.** The R1/R2 defect the whole
reconfiguration existed to fix is gone.

---

## 4. THE SEED'S TWO ROT POINTS — ✅ **BOTH HELD.** The panel's 3–0 conditional is discharged.

**(a) Is the tide ever personified, moralised, or addressed?** No — across all 24. No agency verb attached
to water anywhere; no *"the sea doesn't care"*, no *"time takes everything"*, no second person aimed at the
water. P01 went further than required and reduced *"high tide"* to *"tide"* so not even the phenomenon's own
qualifier could read as an adjective. P04–P06 contain no water at all. The two closest calls, both cleared:
`Wind came up at dinner / Like somebody said it would` (P02-V2:219) is standard meteorological English plus
a **forecast** — the panel's HOUR IS PUBLISHED ruling, not personification; `Somebody's dog is barking / At a
thing it cannot reach` (P02-V1:44) is the run's only near-symbolic line and it is an image, not a
moralisation.

**(b) Does any line make one fact the reason for the other?** No. The Achatz/Willis clause holds in all 24,
and three pairs answered it *inside the lyric* rather than in a note:
- `It was a good castle. / It's a good kick.` (P01-V3:487-488) — two sentences, no connective, repeated
  four times in the chorus. And they cut *"I built the whole thing to hear that"* for making destruction the
  reason for the making.
- `Eleven minutes is eleven minutes.` (P06-V3) — a tautology that **refuses to let the limit mean anything**,
  and the brief's own phrase *"eleven minutes is a result"* was deliberately kept out because `result` values
  the limit.
- `It came out the wrong end / And look at it` (P03-V1) and `It was never meant for this / And it's ours`
  (P03-V4) — adjacency, never causation.

Closest approach, cleared: P03-V3's verse 3 (`And the clean one is correct… / And the other one has weather`)
edges toward *better because flawed*, but the stated reason is an **addition** (*"a chair and a laugh in
it"*), not the wrongness. Kept.

---

## 5. THE INVERSION — ⚠️ **PASSES IN LETTER; P01/P02/P05 CARRY IT ON MIX ALONE**

**⭐ P04 — the flagged highest risk — is the run's cleanest inversion.** Credit where it is due. The lyric
worlds are aggressively, unmistakably modern: three cars on gravel with phone speakers and a lad on the roof;
a back step with a bin lorry's reverse alarm; a street lit by phone torches; the number 12 bus to a garden
centre. Wall-of-sound is explicitly re-read as a **density technique**, and each song admits exactly one
period *instrument* — a harpsichord, a chorister treble, tower bells, two timpani. A timpani roll under a bus
shelter is a genuine seam, and it is funny. **P04 was predicted to fail and did not.**

P03 and P06 also pass on material: P03's intruder is a different **signal path** (1967 spring-reverb organ
shuffle; a horn four bars behind in mono tape tone) over a modern street-band record with a whistle and a
crowd; P06's are wah clavinet and phase-shifted guitar over congas and tight sub.

⚠️ **But in P01, P02 and P05 the "modern world" and the "period intrusion" are built from the same
instrument list, and the only thing separating them is mix treatment.** P01's modern world is handclaps,
tambourine, girl-group thirds, horn stabs and a walking bass — *that is the 1963 instrument list* — and the
intrusion is castanets plus a mono-ness. P02's seam is literally *"eight bars flip to a 1966 mono Detroit
drum room with tape slap"*: same band, different room. P05's is a sax and a tack piano *"tape-saturated"*.

⛔ **THE GRAIN LAW says this is the class of instruction the generator most reliably smooths.** A spec whose
seam is "wide and dry" versus "narrow and slapped" is asking Suno to hold a mix distinction across eight
bars — which is exactly what it does not do. ⭐ **P02's own residue note (`pair_02:785-787`) says this
plainly and is correct**; I am extending the finding to P01 and P05. If the inversion fails at render in
three pairs, no text gate in this pipeline can tell you. `lofn-render-audit` under THE BLIND RULE is the only
instrument that can rule on it.

One era ambiguity, cleared: **P01-V2** ("a hat like a Sunday", "does the prints on Thursday", "wrote my
address on his cuff") reads period until `Kid wants to know what a negative is` — which places it *now*,
with a film photographer on a modern seafront. Defensible, and the only ambiguous-era lyric in the set.

---

## 6. LOFN BLEED — ✅ **CLEAN**, with one register violation that is not bleed

Zero archive-yearning, zero industrial grief, zero laboratory or category vocabulary across all 24. Both
flagged pairs cleared their risk outright:
- **P04** was at risk because *"everything scattered traces back to one point"* is a Lofn conceit at its
  root. It comes out as a woman on a bus to a garden centre being thrilled by a repainted shelter. ✅
- **P06** was at risk because repair-and-obsolescence is Lofn's home turf. It comes out as *"laughing at the
  ceiling about a hair round a fan spindle."* ✅ P05 carries the F11 biochemistry **entirely as situation** —
  no protein, protease, histone or bond reaches a sung line, only *"a machine that only does it / It's the
  only thing it does."* That is the doctrine executed exactly.

⚠️ Closest approach, cleared: **P06-V3:478-484** (*"Typed my name on a blank page… Nothing on here that I
need"*) is a hair from Lofn's empty-archive register; it is saved by *"Typed it in the biggest font"* and
*"Look at it go."*

⛔ **The one real violation is a Sunna-register failure, not a Lofn bleed.** P02-V1's radiant —
**`Nothing at work pulls back`** (`pair_02:118`) — is the only line in 24 songs that is a **complaint**. It
makes the good afternoon a *contrast to* a dull life, which converts the pleasure into something that is
doing a job — precisely what the Willis seat forbade (*"the enjoyment has to be allowed to be pointless"*)
and what Sunna's own mission line rules out (*"You're not disappointed. You're not looking."*). P02's residue
note flags it as *"a thesis"*; the sharper reading is that it is **disappointment as posture**, arriving in
the one place a radiant cannot be ignored. **Repair.**

---

## 7. ⭐ THE SET'S SINGLE WEAKEST SONG

# ⛔ **P04-V2 — "ON TOP OF EACH OTHER"**

Three defects that compound, in the pair whose entire brief was *hide the labour*:

1. ⛔ **The chorus names the arrangement device while the arrangement performs it.**
   `Put them on top of each other!` sung ×8 by a vocal stack that is, at that exact moment, one take → two →
   four. `pair_04:30` swears *"Nothing in any lyric announces the device"* and `pair_04:32-35` records that a
   retrograde harpsichord was **cut** for precisely this crime — and then the crime is in the hook. The
   listener hears four stacked voices instructing someone to stack them. **That is THE REMARK TEST failing in
   the most conspicuous position available.**
2. ⛔ **The radiant traces back to nothing.** `same rail, every night since March` (354-358). The note
   celebrates that *"the word `rail` appears nowhere else in the song"* — but a radiant's whole job is that
   every earlier line points at it, and there is no rail anywhere in the preceding 77 lines. The listener has
   nothing to re-see. Compare P04-V3's `One switch, in a shed, on the edge of town.`, which every lamp, kerb
   and chair in the song genuinely traces to.
3. ⚠️ **Prose stage direction inside the Suno lyrics field**: `[she stops here; the chorister finishes the
   line alone]` (`pair_04:346`), mid-verse. Suno may vocalise it.

It is not a bad song — nothing in this set is. It is the one whose *idea* and *execution* fight each other.

**Runner-up weakest: P02-V1 "Both Hands."** The §6 complaint-radiant, plus the run's only genuine
**repeated-line collapse**: its eight-line chorus contains **three distinct lines** (`Both hands, both hands`
/ `I have got it in both hands` ×3, plus two others), repeated ×4, and the outro is `Both hands` four times.
That is repetition that has stopped being a device.

---

## 8. THE 16-POINT SUNO GATE, THE DAILY RULES, AND THE DEDICATIONS

### Daily rules — all ✅
Tri-source declared before any artifact and honoured in all 24 (S1 seed, S2 ≥2 verbatim imports, S3 radiant
present and late in 24/24) · 3+3 arms ✅ · NEWS = 2 of a max 3 ✅ · AWE (P01/P04/P06) + INDIGNATION (P05) ✅ ·
**six genuinely distinct verse architectures** ✅ (P01 10-line + `Right —` turn · P02 12/8/11-line with a
one-word pre-chant · P03 four *internal* grammars + a 4-verse/bridge frame · P04 12-line scatter verses + a
unique **4-line radiant verse** · P05 10-line + 4-line SEAM block · P06 the only **five-verse + breakdown**
architecture) · wolf ceiling 4/6 ✅ · club 0, anger 0, bed collisions 0 ✅.

### ⚠️ Package-contract items the counter did not measure

| # | Finding | Verdict |
|---|---|---|
| 8.1 | ⛔ **EMO header contract — three incompatible conventions, and 3 of 6 pairs name no emotion at all.** P01 `[Chorus - AWE]` (bare) · P02 `[Chorus - EMO: full room, shouted, stacked thirds]` · P03 `[Chorus \| EMO: room-sized…]` (pipe) · P04 `[Chorus - EMO: Awe, Joy - full stack…]` · P05 `[CHORUS — EMO: full chest…]` (em-dash) · P06 `[Chorus - EMO:Awe - Female Vocalist plus room - …]`. The contract is `[Section - EMO:<emotion> - <Role> - <cue>]` with taxonomy emotions. **Only P04 and P06 put an emotion in the EMO slot**; P02/P03/P05 put delivery directions there. | **REPAIR** → step 08 |
| 8.2 | ⚠️ **≥1 SFX cue — not met as a discrete cue in any of the 24.** Crowd beds and room mics are specified in style fields (P02-V4 *"crowd bed from room mics kept 10 dB low"*; P03-V3 *"a chair, a laugh and a door survive into the master"*; P05-V4 terrace chant), and P03-V3 carries SFX **as lyric content**. Spirit served; letter not. | **FLAG** |
| 8.3 | ⛔ **P04's sweep receipt is FALSE-CLEAN.** `pair_04:844` reports the token `wall of sound` returns **NONE** across all style fields. The string **`wall-of-sound density` is in all four style fields** (lines 69, 256, 430, 612) — the sweep tested a spaced string against a hyphenated one. ⭐ No compliance breach follows (Spector is correctly never named, and the technique name is not a person), **but a receipt that reports clean on a string that is present is worse than no receipt, because a reader trusts it.** This is the exact class the "measured, never eyeballed" doctrine exists to prevent, arriving one level up: *the sweep measured the wrong string.* | **REPAIR** (receipt) |
| 8.4 | ⚠️ **P06's dedication inverts the credit it exists to pay.** *"TONY ALLEN, drums, with Fela Kuti, who built that placement."* The relative clause attaches to **Fela Kuti**. The whole point is that **Allen** built the placement. One-word fix: *"TONY ALLEN, drums, who built that placement with Fela Kuti."* ⛔ This is the invisible-player credit the doctrine says the system forgets first. | **REPAIR** (dedication) |
| 8.5 | ⚠️ **One variation is outside the declared tempo band, and the set audit says otherwise.** `P06-V3 = 98 BPM`. `00_research_brief.md:151` reports *"100–126, all six inside."* The audit was run on **pair-level nominal BPM**, not on the 24 style fields. (98 sits in the `<100` cell at 1.07×, so it is not a bad cell — but the audit is inaccurate.) | **FLAG** |
| 8.6 | ⚠️ **P03 is the only pair with no measurement receipt on disk.** Its counts are asserted in prose (*"100 sung lines, short"*) with no per-variation table and no named script, while the other five carry `measure_pNN.py` receipts. Under *THE EVIDENCE IS THE ARTIFACT*, P03's numbers currently rest on the coordinator's separate pass rather than its own. | **FLAG** |
| 8.7 | ⚠️ **Non-lyric prose inside the Suno lyrics field in 5 of 24 songs.** P06 ×4 uses asterisked markers mid-verse (`*1971 horn section, tape hiss, mono, eight bars*` etc.) **splitting a sentence around them** (`Last screw, and it's —` / marker / `— in.`); P04-V2 uses a bracketed direction. If Suno vocalises them you get a singer reading "1971 horn section tape hiss mono eight bars." | **FLAG** → render risk |
| 8.8 | ⚠️ **Boundary-hugging: P02 is 4/4 at ≥985 style chars** (991/987/998/996 against a 1000 cap; V3 is 2 chars clear). Any late edit overruns and Suno truncates mid-sentence — which is exactly how P06-V1 nearly lost its seam instruction at 1008. | **FLAG** |
| 8.9 | ⚠️ **The one-fact rule was interpreted two ways.** P01/P02/P03 read it as *one numeral in the sung lyric* and made repairs to comply (*"first go"* → *"straight off"*; *"Two lads"* → *"Lads"*; *"half a turn"* → *"a turn"*). P04 read it as *one research fact* and sings `Three cars`, `Two of us, one jack`, `at four` and eleven month names in V1 alone. **P04's reading matches the ICB's intent** (*"the rest inform the brief and stay in the brief"*) — so P01–P03 over-complied and paid small pieces of natural speech for it. The gate is not measuring one thing across the run. | **FLAG** → clarify in ICB |

### ⭐ DEDICATIONS — **ALL SIX CREDIT SETS ARE TRUE.** I checked each attribution against the sound its
style fields actually raise, and checked that no credited artist appears in any style field.

| Pair | Credit | Raised by the style field? | Attribution correct? |
|---|---|---|---|
| P01 | Hal Blaine (triplet tom fill); the Blossoms / Darlene Love (answering voices) | ✅ V1/V3/V4 raise a triplet or rolling tom fill; all four raise three answering period voices *(V2 raises the voices but no tom fill — the dedication is scoped to V3, which raises both)* | ✅ Philles-session attributions correct |
| P02 | Funk Brothers — Jamerson (bass); Benjamin, "Pistol" Allen, Uriel Jones (drums); the northern soul floors | ✅ *"1966 mono Detroit drum room"* + *"melodic bass walking"* | ✅ all four are Funk Brothers; Twisted Wheel / Blackpool Mecca / Wigan Casino are the canonical floors |
| P03 | Lynn Taitt, Hugh Malcolm, Jackie Jackson, Jackie Mittoo, Tommy McCook, Kingston 1966-68; Coventry pointed upstream | ✅ upstroke skank, walking bass, Hammond, trombone, *"1967 rocksteady one-drop"* | ✅ **the best-sourced credit in the run** — every named player maps to a raised role |
| P04 | Darlene Love *"who sang lead on records issued under other names"*; Carol Kaye; ⛔ Spector deliberately unnamed | ✅ girl-group stack, front-of-mix electric bass | ✅ and the Love line is the sharpest invisible-credit fact available |
| P05 | Don Powell (Slade), Bill Legend (T. Rex) | ✅ *"Boot-stomp beat: kick on 1 and the and-of-2, snare and dry handclaps on 2 and 4"*, *"1973 terrace gang chant"* | ✅ both correct |
| P06 | **Tony Allen**, with Fela Kuti | ✅ *"Afrobeat drum figure"* verbatim in **all four** style fields | ✅ correct — but see **8.4**, the sentence inverts it |

✅ **No break is named in any of the 24 style fields, so no drummer credit is owed and none was invented** —
the honest half of *IF THE PROMPT NAMES A BREAK, CREDIT THE DRUMMER*. ✅ **No credited artist appears in any
style field.** The only proper nouns are instrument models (Telecaster, P-bass, Wurlitzer, Hammond, Rhodes,
Mellotron) and a city (Detroit). ✅ Honesty clause held both ways in all six — *"we asked for those figures by
name, so we name the players; nothing here is a sample"* — including on the three songs that import
`a choppier flip of the buttery soul sample…` as a **texture instruction**.

---

## Structured Evidence Block *(measured, BESIDE the verdict, never replacing it)*

| Metric | Measured | Band / Cap | Pass / Flag |
|---|---|---|---|
| Style chars, all 24 | 924 – 998 | 850 – 1000 | ✅ · ⚠️ P02 4/4 ≥985 (8.8) |
| Lyrics chars, all 24 | 2760 – 3985 | <5000 (target ≤4800) | ✅ |
| Sung lines, all 24 | 80 – 100 (median 92) | 70 – 120 | ✅ |
| Words / line | 4.15 – 5.64 (median 4.94) | low is better (r = −0.202) | ✅ |
| Tempo | 98 – 127 BPM | 100 – 129 measured home | ⚠️ P06-V3 = 98 (8.5) |
| EMO-tag balance | P01 100% bare `AWE`; P06 100% `Awe`; P04 varied; P02/P03/P05 **no emotion named** | taxonomy emotions, no single dominant | ⚠️ **FLAG-cleared-with-reason** — the single-value tag is a deliberate register lock under R3 and the justification is good; the *format* still misses the contract (8.1) |
| Repeated-line ratio | chorus-exempt; one genuine collapse | FLAG only | ⚠️ P02-V1 chorus = 3 distinct lines in 8, plus a 4× outro (§7) |
| House-lexicon / template | ⛔ **`Best <time-unit> I've had` ×9 in P06** · **hot food/kettle resolution in 10 of 24** · **`My dad + domestic object` radiant ×2 across arms** | flag calcified phrasing | ⛔ **FLAG — this is the run's real formula (§2d)** |
| One-fact rule | interpreted two different ways across pairs | ≤1 numeric fact sung | ⚠️ (8.9) |
| Cross-pair distinctiveness | prompt 0.489 / lyric 0.328 (coordinator) | 0.58 / 0.42 | ✅ — but blind to radiant-level templating (§2d-iii) |
| Dedications | 438 / 473 / 417 / 466 / 477 / 471 | ≤500 | ✅ |
| Absolute exclusion · club · anger · negations · zero-wolf P03/P04 | clean (coordinator) | — | ✅ |

---

## Somatic Gate — 3 Hyper-Skeptics *(PRIMARY; the block above is a floor beside it, never the verdict)*

**Question: distinctive enough to be Lofn-as-SUNNA, or generic?**

**⚡ SKEPTIC 1 — the Anti-Sentimentalist (Sontag seat, carried from the run's own panel): YES.**
*"I put the cliché objection on the record this morning and I withdrew it on one condition. The condition
held — twenty-four lyrics and the water is never once a character. That is a discipline I did not expect to
survive four pair agents. What I will not let pass unremarked is that ten of your songs end on a hot drink.
You removed the greeting card from the tide and reinstalled it in the kettle."*
**Counter-move:** *cap the comfort-food return at three per set and require two pairs to resolve on
something that is not consumed.*

**⚡ SKEPTIC 2 — the Pleasure Politician (Willis seat): YES.**
*"The enjoyment is allowed to be pointless in twenty-three of twenty-four, and eight songs carry no wolf at
all and are not apologised for. P03 in particular is a whole pair of joy with no invoice and it is not
thin — that is the hardest thing on the list and it was done. **My one no is `Nothing at work pulls back`.**
That line makes the kite a compensation for a job, and a compensation is pleasure doing a job. It is one
line and it is the radiant, which is the worst place for it to be."*
**Counter-move:** *replace P02-V1's radiant with a physical fact from the same afternoon — the tape, the
burn, the arm — so the origin is the hand, not the life.*

**⚡ SKEPTIC 3 — Fisher's seat (the falsification, R3's own): NO, scoped.**
*"Across the four wolf pairs the test passes and I will say so plainly. **Inside P01 and P06 it fails.**
P01 sings the identical four words in four songs and calls it four events. P06 says the same end-of-life
fact three ways and its own file claims the swap would break — I ran the swap and it does not break. And
you have a second formula the test was not pointed at: P06 signs off four songs with one sentence and a
swapped noun. A personality that resolves the same way every time has a mannerism, not a repertoire."*
**Counter-move:** *P01 — rewrite two of the four tide lines so the hour is delivered by a different
mechanism in each (someone else says it / it is read off a board / it is not said at all and only the light
tells her). P06 — keep V4's cost-bearing wolf, and give V1 and V3 a wolf that costs her something rather
than reporting a status; cut three of the four `Best ___` sign-offs.*

### **BLOC RESULT: 2 YES / 1 scoped NO → NOT BLOCKED.** The set is unmistakably this personality.

**Named-corpse checklist, walked:** one-note emotional arc — **no** (every song reaches a second movement;
the F17 arc is executed in 24/24) · single dominant repeated section — **no**, except the flagged P02-V1
chorus · a motif that never transforms — ⚠️ **P01's `Tide's at nine`, which recurs four times across four
songs without transforming**; deliberate return elsewhere (byte-identical choruses) is intentional and
needs no defence · repeated-line collapse — ⚠️ **P02-V1 only.**

---

## Required Repairs — routed

**BLOCKING (Overall cannot reach SHIP until resolved):**
- **R1 · §1 · route: coordinator / The Scientist.** Restore the per-pair editorial spine
  (`step06/07/09` for all six pairs, re-emitted from material already inside each `step10`), **or** record a
  dated `lofn-daily` exemption for single-file pair synthesis. Also missing vs. all three prior dailies:
  `CREATIVE_CONTEXT.md`, `GATE_REPORT.json`, `INDEX.md`, `RUN_STATE.md`, `TOP6_PASTE_READY.md`.

**SUBSTANTIVE (fix before render):**
- **R2 · §6 · route: step 09, P02-V1.** Replace the radiant `Nothing at work pulls back` — it is the run's
  only line of disappointment-as-posture and it is in the load-bearing position.
- **R3 · §7 · route: step 09, P04-V2.** Rewrite the chorus so it stops naming the arrangement device, and
  seed `rail` earlier so the radiant has something to trace back to. Remove the bracketed stage direction
  from the lyrics field.
- **R4 · §2b · route: step 09, P01 + P06.** Fisher repair per Skeptic 3's counter-move above.
- **R5 · §8.3 · route: step 10, P04.** Correct the false-clean sweep receipt (`wall-of-sound`, hyphenated).
  ⭐ Also fix the sweep, not just the line — this class of error will recur.
- **R6 · §8.4 · route: step 10, P06.** One-word dedication fix so the placement is credited to Tony Allen.
- **R7 · §3c · route: step 10, P05.** Strike the claim that the container line is unremarked (it is
  licensed, which is the correct and stronger defence), and re-examine the V2 bridge epigrams against
  R3 rule 1.

**PROCESS (before the next R3 run):**
- **R8 · §8.1** — fix one EMO header convention and apply it to all 24.
- **R9 · §2d-iii** — the distinctiveness gate must compare **radiants against radiants**, not lyric bodies.
- **R10 · §8.9** — state the one-fact rule as *one research fact*, explicitly, in the ICB.
- **R11 · §8.5** — run the tempo audit over the 24 style fields, not the 6 pair nominals.
- **R12 · §5 + §8.7** — flag P01/P02/P05 (mix-carried inversion) and P06/P04-V2 (in-lyric prose markers)
  for `lofn-render-audit` under THE BLIND RULE. No text gate in this pipeline can rule on either.

---

## ⭐ RANKED TOP 6 — best 3 per arm, ranked **within arm only**

### ACCESSIBLE

**1. P03-V1 — "IT WASN'T FOR THIS"**
The run's best verse is in this song (*"Then my mother's good earring / Then my mother, who laughed"*), its
radiant pays off on a fridge with the father alive and present tense, and its chorus hands the room a
**physical action** — *"Put your palm on the glass"* — that stays true of whatever they misused this week.
*Over its siblings:* V2 has the better idea and the cleaner removal of rebellion posture but a duller hook;
V3 is the most inside-baseball subject in the set (a song about making a record); V4 is the warmest but
"wrong" repeated is its least distinctive word.

**2. P01-V3 — "KICK IT OVER"**
The best room-chant in the set, and the only song that carries the panel's strongest anti-elegy ruling
(*strike night is a party — she is the one who kicks it over*) **and** answers the seed's sharpest rot point
inside the lyric rather than in a note: *"It was a good castle. / It's a good kick."*
*Over its siblings:* V1's radiant is quieter and its bridge carries the `Nobody breathe` hush risk; V2's
chorus does not travel (*"Don't get the bins in!"*); V4 is the warmest and most published-hour-literal but
its wolf is pure status report.

**3. P02-V4 — "LOOK AT THAT ONE"**
The only song in 24 whose entire emotional content is **admiration of someone else's work** — which is
Sunna's thesis rather than her subject matter (*delight is donative*, *she will not be ironic about
something anyone loves*). *"Somebody made that / And it was not me"* is the most generous chorus line here.
*Over its siblings:* V2 has the best single verse in P02; **V3 has the best radiant in the whole run**
(*"It is my sister's car"*) but the least motion; V1 carries the §6 defect.

### AMBITIOUS

**1. P05-V4 — "IT WAS IN THE EMAIL"** ⭐ *the run's peak*
The only song where the **form itself answers the lyric**: she names the bright container as wrong, hands
off to it (*"Here's the chorus"*), and the chorus arrives **byte-identical, including "And I said that it
was fine"** — the container is given a chance to adjust and does not change a character. A listener feels
that without parsing it. Its radiant, *"They published the hour in a place I don't look,"* inverts the
entire day's engine in one plain sentence and discharges the REFLECT panel's amendment.
*Over its siblings:* V1 is the better pure pop song; V2 has the best **situation** in the run (the phone
agent who stops before the word) but the most writerly bridge; V3 is the sparsest and cleverest and the
least anthemic.

**2. P06-V4 — "NOT TELLING HER"**
⭐ The **only P06 variation that survives the Fisher test** — its wolf costs her something (*"Part it wants
is not made anymore. / Took one out of mine."*) instead of reporting a status. Best hum test in the set: the
hook is literally a **hummed melody with no words**, stated alone before the first line and after the last —
pillar 3 made literal. And the most generous radiant in the run.
*Over its siblings:* V3 has the better images and the better title-refrain but sits at 98 BPM outside the
band and reports a status; V1 is the warmest but shares a radiant-kind with P03-V1; V2 is the funniest
(*"WHICH COLOUR. He has options."*) and the least weighted.

**3. P04-V4 — "I CAME ALL THIS WAY"**
Sunna's whole thesis compressed into two chorus lines — *"And it is a bus stop. / And it is perfect."* The
angle and the emotional payoff are the same gesture: arrive at the origin, find it ordinary, be **pleased**.
Two orchestral timpani under a bus shelter is the funniest and most earned seam in the run, and *"Went past
a horse in a coat"* is the line I would put on the poster.
*Over its siblings:* V3 has the better radiant *as a radiant* (*"One switch, in a shed, on the edge of
town"*) and carries the run's F18 number well, but its premise is a scene rather than an arrival; V1 has the
most craft and a radiant a listener cannot picture; **V2 is the set's weakest song (§7)**.

---

## ⭐ IS THIS A DIFFERENT ARTIST FROM THE CLUB/PUNK SUNNA OF R1/R2?

# **Yes. Recognisably, and not because of the genre labels.**

The genre swap is the least of it, and would not on its own answer the question. Here is what actually
changed, from the artifacts:

1. **The swagger carrier moved.** 24 of 24 style fields build confidence out of a **walking, hummable
   front-of-mix bass**; 24 of 24 EXCLUDE fields ban distortion-pedal swagger. R1/R2 built it out of fuzz.
   *"Swagger is a rhythm section, not a distortion pedal"* is not a slogan in this run — it is the
   instrumentation of every single track.
2. **The hum-test hook is instrumental in 24 of 24** — a bass figure, a guitar figure, a two-part horn
   phrase, an organ bubble, a hummed melody with no words. R1/R2's hooks lived in the vocal.
3. **The wolf changed channel exactly as specified.** Sixteen songs name it in words; **zero** carry a
   standing tension lane; **eight carry no wolf at all and are not apologised for.** The clause that made the
   music anxious by rule is gone and nothing smuggled it back.
4. ⭐ **The register is a different life.** The nouns are kettles, chip shops, launderettes, garden centres,
   bus stops, biscuit tins, saucers of screws, tote bags, a nan with a cup, a comb borrowed for roof tiles.
   R1/R2's Sunna lived on a 5am club floor. This one lives in a **town**.
5. ⭐⭐ **And the sharpest evidence: six of twenty-four songs are about someone else's pleasure or someone
   else's work** — admiring a stranger's kite (P02-V4), a whole town's misuse (P03-V4), a secret repair
   nobody will know about (P06-V4), a man with a shelf of little boxes (P06-V2), keeping a trumpeter's late
   entrance (P03-V3), paying a photographer out of her own wages (P01-V2). R1/R2's Sunna was a frontwoman on
   a floor. **This one is a person in a crowd, and often not the best person in it.** That is a different
   artist, and it is the version that earns the Pantheon seat — *"you make them want themselves."*

⚠️ **The honest qualification, and it is the prediction the next run should falsify.** R3 fixed the **bed**
monoculture and, on day one, produced a **setting** monoculture. Ten of twenty-four songs resolve on hot
food or a kettle; P06 signs off four songs with one sentence; two radiants across two arms are the same dad
and the same domestic object. Twenty-four songs share one social class and one weather. **The house lexicon
that is calcifying now is not `club` and `fuzz` — it is `toast`, `kettle`, `chips`, `Best ___ I've had`, and
`sun going orange`.** It is early enough that naming it is cheap; by the third R3 run it will be the sound
of the personality, and it will have arrived by exactly the mechanism the genre-bed block already names:
*what is not assigned gets defaulted, and the default is the strongest prior.*

**Assign the setting.**

---

## Failure-ledger entry — ⛔ NOT WRITTEN

`vault/COMPETITION_LEARNINGS.md` takes **one curated entry per shipped/selected piece**. Nothing is shipped
or selected yet (Overall = HOLD, §1 unresolved, and `vault/AUTONOMY.md` holds the render line regardless).
The entry is drafted below and should be appended **only** when The Scientist selects:

> **[music · daily · SUNNA R3]** *Fisher's swap test caught nothing where it was aimed and everything one
> level down: the four pairs' wolves are non-interchangeable across the set, but two pairs' wolves are
> interchangeable inside themselves, and the real formula was in the resolution — ten of twenty-four songs
> ended on a hot drink and one pair signed off four songs with the same sentence.* **Transferable rule:**
> *point a falsification test at the element you most recently repaired AND at the element you never
> assigned — a template forms fastest where no axis was specified.* **Confidence: MEDIUM** (n=1 run,
> text-stage only, unrendered).

---

## Final Recommendation

**HOLD — and it is a short hold.**

The music is the best writing this personality has produced and **R3's laws landed in the artifacts, not
just in the claims**: the band genuinely does not know, the register genuinely never moves, the wolf
genuinely resolves, and both of the seed's rot points survived all twenty-four lyrics — which was the
panel's explicit 3–0 conditional and is now discharged. P04, the flagged highest pastiche risk, is the run's
cleanest inversion.

What stands between this and SHIP is **one governance decision (§1)** and **six named repairs**, none of
which requires re-writing a song. Resolve R1, apply R2–R7, and the ranked top 6 above are selectable.

⛔ **Two things no text gate in this pipeline can rule on, and they are not optional:** the mix-carried
inversion in P01/P02/P05 (§5) and the in-lyric prose markers in P06 and P04-V2 (§8.7). Both are decided at
render, by `lofn-render-audit` under **THE BLIND RULE** — audio alone, first, never the prompt.

⛔ Nothing rendered, nothing published, nothing spent (`vault/AUTONOMY.md`). Twenty-four packages on disk.
The `.RUN_LOCK` was not touched; no pair file was edited. This report prescribes; it does not operate.
