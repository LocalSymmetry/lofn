# QA REPORT — `lofn-prime-every-element` · 2026-08-22
## Independent adversarial audit. The judge wrote none of this.

⛔ **THE EVIDENCE IS THE ARTIFACT, NEVER THE AGENT.** Every verdict below names a file, a line, or a
script output a stranger can re-open and re-run. Where I ran an instrument, its command is given.

⭐ **The question is not "does this pass gates." It is: WOULD ASH SAY THIS SET FIXED IT?** That section
is last, and it is the verdict that matters.

---

# 0. VERDICTS

| pair | owns | verdict |
|---|---|---|
| **P01** | E8 PEAK · E10 CONTOUR · E15 HOOK | ⭐ **SHIP** |
| **P02** | E9 MELODIC RANGE · E11 THE RETURN | ⚠️ **REPAIR** — ban register absent; density-break mechanism in doubt |
| **P03** | E4 TWO REGISTERS · E6 WITHHELD GESTURE · E13 DUALITY | ⭐ **SHIP** |
| **P04** | E3 MACHINE-LOG CLOCK · E5 NAMED FREQUENCY | ⭐ **SHIP** |
| **P05** | E1 GLITCH-BAROQUE · E2 ORNATE v MACHINE · E12 ONE SEAM | ⚠️ **REPAIR** — over-claimed its own falsification test |
| **P06** | E7 SWAG · E14 THE OPENING | ⚠️ **REPAIR** — **never names a quietest moment in any of its four songs** |

## ⛔ RUN VERDICT: **REPAIR** — 3 SHIP · 3 REPAIR · 0 FAIL · 2 HOLDs to The Scientist

**Nothing here is a FAIL and nothing needs rewriting.** All three repairs are additive: two are
documentation, one is a four-line specification gap. The songs are the best-specified set in this repo.

---

# 1. ⭐ THE PEAK TEST — the one that matters

*Name the loudest moment and the quietest for all 24. If a listener could not tell them apart from the
description, the song has no contour.*

**Five of six pairs self-administered this test and reported both poles.** Sources:
`pair_01_step09_artist_refined.md:167` · `pair_02_step07_song_guides.md:260` ·
`pair_03_step09_artist_refined.md:145` · `pair_04_step09_artist_refined.md:115` ·
`pair_05_step10_final_package_enhanced.md:617`. I re-checked each against the shipped prompts and
lyrics rather than accepting the self-report.

### P01 — ⭐ PASSES, and it is the strongest contour instrument in the run

| | quietest | loudest |
|---|---|---|
| V1 | 0:17 — one unaccompanied voice, conversational, mic noise audible | 2:37 — sixteen-voice choir + full strings, lead belting a **12th** above her lowest verse note, *fff* |
| V2 | 1:50 — everything stops; voice at the **bottom** of her range | 2:30–2:33 — choir lands on a downbeat with **no swell into it** and the lead is *absent for two bars* |
| V3 | 3:43 — hook decayed to speaking level, **nothing playing** | 2:43 — choir on the hook while the lead sings the organ's turn against it, two melodies at once |
| V4 | 0:20 — silence four bars after a wall of strings | 2:40 — choir **and a semitone key lift on the same downbeat** |

⭐ **Four different shapes: a ramp · a hole · a long fall · a cliff both ways.** V3 is the only song in
the run whose quietest point comes *after* its loudest. **No two of these eight are describable in the
same words.** This is the direct answer to *"no highs or lows"* and it lands.

### P02 — PASSES, but the quiet pole templates
Loudest V1 2:49 · V2 2:39 · V3 2:49 · V4 2:46 — all the octave opening, but at four different bar
addresses (bar 3 / bar 1 / bar 4 / bar 2 of the chorus) with four different intervals underneath.
**Genuinely distinct.** ⚠️ **Quietest: 2:07 / 2:04 / 2:00 / 2:07 — all "hushed verse 3," all within a
seven-second window, all the same device.** Differentiated only by instrument count (voice+harpsichord /
+bowed bass / voice alone / two-person dialogue). Thin, but present and honestly reported.

### P03 — ⭐ PASSES, and V4 is the best structural idea in the run
V1 one voice with nothing under it / two registers on one note, beating gone. V2 a fry voice with almost
no pitch over one stopped note / a low voice that has climbed a ninth and **lost its rasp getting
there**. V3 the chorus at a whisper with the drum removed / both voices on a note **neither has sung all
song**. ⭐ **V4: the loudest two bars and the quietest two bars are the same two bars** — the unison at
3:10 in a gap outside the form, and at 3:35 the same gap returns with nothing in it. The write and the
erase are one address.

### P04 — ⭐ PASSES, and it is mechanically the sharpest
Quietest and loudest are **300 ms apart** in every variation: 150 ms of true digital silence at the
erase bar, then the 41 Hz sub alone at maximum. V1 2:52.8/2:55.2 · V2 2:33.6/2:36.0 · V3 3:12.0/3:14.4 ·
V4 2:14.4/2:16.8. ⭐ **The contour is not a claim, it is an edit.** ⚠️ The quiet pole is the same device
×4 — but the stop *lengths* differ genuinely (4 / 4 / 6 / **12** bars = 9.6 s to 28.8 s), and V4's
28.8-second hold is a different experience from V2's slam. Accepted.

### P05 — ⚠️ HALF-PASSES, and it self-reported a full PASS
Loudest: four genuinely distinct transgressions (4' coupler at sixteen notes a beat · the only 808 glide
in the song with mordents bending after it · crushed harpsichord in C2+C3+C4 at once · Bebung on a
**decay tail** with the singer a beat early). ⭐ Distinct.
⛔ **Quietest, all four: "one bar, everything muted but the 808."** V1 2:00 beat one · V2 1:13 upbeat ·
V3 1:38 offbeats · V4 2:17 beat three. **The pair's own table (`pair_05_...:617`) claims "no two of the
eight are describable in the same words." Four of the eight are describable in exactly the same words**
— the differentiator is a beat position, which is not something a listener names. `PASSES` is written
under a table that shows the failure. ⛔ *A gate that reports PASS on a half it did not measure is the
defect class this repo cares most about.*

### P06 — ⛔ **FAILS. It never names a quietest moment.**
This is the sharpest defect in the audit and it is mechanical, not aesthetic.

```
grep -c -i "quiet" pair_06_step06_facets.md pair_06_step07_song_guides.md \
                   pair_06_step09_artist_refined.md pair_06_step10_final_package_enhanced.md
  → 0 / 0 / 1 / 0
```
The single hit is an incidental adjective in a step-09 edit note (`pair_06_step09_artist_refined.md:95`,
*"quieter and funnier"*). **P06 names a PEAK for all four songs** — timecoded, with a spectral band and
a rhythmic slot (`pair_06_step07_song_guides.md:76, 95, 114, 135`: horns 2:32 · vibraphone 2:25 · **the
bus** 2:17 · whistle 2:25). **It never once names a floor.**

**Why this is not excused by SWAG.** `05_pair_assignments.md` universal clause 1: *"Peak · range ·
contour... must appear in the production spec **and** be audible as an event."*
`02_SEED_AND_ELEMENT_MATRIX.md` §3 defines contour as *"(3) its DYNAMIC CONTOUR — **where it is
quietest** and where it is loudest."* The master table authorises P06's contour to be **flat in shape**
(*"steady by design — the contour is TIMBRAL DENSITY"*). ⛔ **It does not authorise the specification to
be absent.** A flatness that is deliberate must be able to say where its floor is — and this is the one
pair whose entire defence is that its flatness is deliberate.

⭐ **The repair is cheap and the material already exists in the artifacts:** V1's tag (*"horns out,
stomps and bass last"*), V2's wood block *"stopping mid-figure"*, V3's *"parlour's own short slap-back
is the last thing on the tape"*, and ⭐ V4's **two bars of bare snare after the handclaps lift, before
the whistle takes the address** — which the package already describes as the point. Name them as floors.

---

# 2. ⭐ THE OWNERSHIP TEST — *could another pair have done this element as well?*

| pair | element | could another pair have done it? | ruling |
|---|---|---|---|
| **P01** | E8 PEAK · E10 CONTOUR | **No.** It is the only pair permitted a choir, a key change, a belt and a crescendo (universal clause 4) and it uses all of them. Exclude fields **14.8 terms/song against a run mean of 36.2** — the only pair that reduced prohibition, which is the run's whole thesis | ⭐ **OWNED** |
| **P01** | E15 HOOK | Four adoptable hooks, 3–6 syllables (*Hold out your hands · It was for this water · It came out perfect · Do it again*) | ⭐ **OWNED** |
| **P02** | E9 RANGE | **No.** Interval + bar address + leap-word + octave opening + timecode, four times over. Nothing else in the repo specifies melody this precisely | ⭐ **OWNED** |
| **P02** | E11 RETURN | **No.** The removal has a named address and four different repayments (a struck bell on the vacated note · **never repaid** · the strings alone · inversion using both of the leap's own notes) | ⭐ **OWNED** |
| **P03** | E4 · E6 · E13 | **No.** One voice, two registers, meeting once at a named bar, with one breath arriving and later removed from the same address | ⭐ **OWNED** |
| **P04** | E3 · E5 | **No.** Four distinct bodily jobs for 41 Hz (takes the swallow off you behind the sternum · buzzes the back molars · longer than the room so it never becomes a note · comes up through the floor and the chair) | ⭐ **OWNED** |
| **P05** | E1 · E2 · E12 | **No** — and it defended the boundary explicitly. See below | ⭐ **OWNED** |
| **P06** | E14 OPENING | **No.** Four different first events, all at 0:00 | ⭐ **OWNED** |
| **P06** | E7 SWAG | **Yes in principle, no in practice** — nothing else in the run runs a five-entrant parade over a fixed groove | ✅ **OWNED** |

⭐ **The strongest single piece of evidence that the ownership rule worked** is that three pairs
*declined* elements they could have taken:
- `pair_05_...:MAJOR DEVIATIONS V1` — refuses to widen the voice at its own peak: *"the obvious repair
  for a narrow melody is one open phrase at the peak. ⛔ Refused — E9 is P02's and the one-wide-phrase
  move is P04's."*
- `pair_01_...:MAJOR DEVIATIONS V3` — cut a second lower vocal character from its bridge because
  *"that is E4, and it belongs to P03."*
- `pair_04_step09_artist_refined.md:D25` — cut a whole tercet because *"the same figure V4 does better
  and owns."*

⚠️ **Two declared adjacency risks, both correctly self-reported, both cleared by me:**
P02V1's struck bell 12 s after its peak (a single strike paying a melodic debt, not a peak) and P02V4's
two-person dialogue in one mouth (plain reported speech, one register, no register change requested
anywhere in the prompt). **Neither is competition.**

---

# 3. ELEMENT COVERAGE — is each one *nailed*, or merely present?

All 15 present, each owned once. Ruling on maximality:

**NAILED MAXIMALLY (11):** E3, E4, E5, E6, E8, E9, E10, E11, E12, E13, E14.
Evidence for the two hardest to fake: **E5** produces four different physiological claims, not four
restatements of one. **E6** is a single inhale with a bar address for arrival (bars 65/57/33/77) and a
bar address for removal (bars 81/75/81/87) — a withheld gesture whose rationing is literally schedulable.

**PRESENT AND STRONG BUT NOT MAXIMAL (2):**
- **E1 GLITCH-BAROQUE** — P05 is a fine instance, but `pair_05_...:MAJOR DEVIATIONS V1` says of its own
  clearest statement: *"the seam is the most conventional of the four... if only one of these four ships,
  it should not be this one."* Honest, and it means the spine is stated plainly rather than pushed.
- **E15 THE HOOK** — four adoptable hooks, but only *Do it again* and *Hold out your hands* are ownable
  by a room without the verse around them. **Not thin; not maximal.**

**⚠️ THINNEST CLAIM — E7 SWAG.** The mechanism argument in
`pair_06_step10_final_package_enhanced.md` (*"the ear scores intensity as how much of the room is
accounted for, not as amplitude"* — spectral occupancy, a climbing roof, the count of hands) is the
best-argued claim in the run **in text**. ⛔ **It is also completely unverifiable from text**, and it is
the argument Ash's ear already rejected once. It is not thin as a specification; it is thin as
*evidence*. Routed to `lofn-render-audit` as the run's highest-priority listening question.

---

# 4. THE THREE FORMERLY-UNASSIGNED VARIABLES — all 24

| pair | PEAK stated? | RANGE stated? | CONTOUR stated? | in the **lyric/form**, not only the spec? |
|---|---|---|---|---|
| P01 | ✅ ×4, timecoded | ✅ a 12th | ✅ ×4, both poles | ⭐ **yes** — the choir is a lyric event, the belt is a section header |
| P02 | ✅ ×4, timecoded | ⭐ ✅ note-named (B3→D5, E♭4→G5, D4→F5, A3→C5) | ✅ ×4, both poles | ⭐ **yes** — the leap-word is a lyric word (*wait / up / now / hold*) |
| P03 | ✅ ×4, bar-numbered | ✅ a 9th between registers | ✅ ×4, both poles | ⭐ **yes** — the unison is a section |
| P04 | ✅ ×4, bar-numbered | ✅ 4th verses + one wide phrase (m7/8ve/M7/m9) | ✅ ×4, both poles | ⭐ **yes** — `[The Erase]` is a named section |
| P05 | ✅ ×4 | ✅ narrow voice / wide keyboard, note-named | ⚠️ loud pole ×4, **quiet pole is one device ×4** | ✅ yes — `[Seam]` is a named section |
| **P06** | ✅ ×4, timecoded + spectral | ✅ 6th lead, stack to an 11th | ⛔ **HALF — no quiet pole anywhere** | ✅ yes — the entrants are named in section cues |

⭐ **This is a real repair and it should be said plainly.** The baseline
(`output/2026-08-22_ash-critique/ANALYSIS.md`) was **89 peak-device bans and one peak instruction across
nine songs.** This run states a peak for **24 of 24**, with a timecode or bar number for 24 of 24, and
**23 of 24 put it in the lyric or the form rather than only in the production spec** — which is the L22
THE GRAIN LAW requirement. That is the single largest structural change in this repo's history.

⛔ **The one gap: 4 of 24 (P06) state a peak and no floor.** A contour needs two poles.

---

# 5. ⭐⭐ THE BAN ON BANS — verified term by term

⚠️ **First, my own instrument, declared before its output.** I wrote a keyword counter
(`scratchpad/qa_excludes.py`) using the *same* `PEAK_DEVICES` list as
`output/2026-08-22_ash-critique/measure_flatness.py`, so the number is comparable to the "89 across nine
songs" baseline. **It over-counts by construction** — it cannot tell `no big-room reverb on the choir`
(a treatment instruction for a choir that is PRESENT) from a ban on choirs. That is exactly the error
`INSTRUMENT_LOG.md` §3 records the coordinator making twice in one day. **So I read all 24 exclude
fields verbatim before ruling.** The count below is context, not a verdict.

| pair | exclude terms/song | keyword "peak-device" hits/song | ban→replacement register |
|---|---|---|---|
| **P01** | ⭐ **14.8** | 7.0 | ⭐ **term by term, all 19/15/12/13**, in step 07 §§3.7/4.7/5.7/6.7 **and** restated in step 10 |
| **P02** | 36.5 | 11.3 | ⛔ **NONE — see below** |
| **P03** | 41.2 | 10.5 | ✅ `pair_03_step07_song_guides.md:73` — a per-class ledger |
| **P04** | 47.2 | 13.3 | ✅ `pair_04_step07_song_guides.md:36` — "THE REPLACEMENT REGISTER" |
| **P05** | 30.0 | 14.3 | ✅ `pair_05_step07_song_guides.md:104` |
| **P06** | 47.5 | 13.0 | ⭐ **the best in the run** — a full table in step 10, every excluded class covered |
| **RUN** | **36.2** | **11.5** | **baseline was 9.9/song** |

## ⛔ FINDING 5A — the run banned MORE peak devices per song than the corpus it repairs
**277 keyword-matched peak-device terms across 24 songs (11.5/song) against the baseline's 89 across
nine (9.9/song).** ⚠️ **Read verbatim, this is mostly legitimate** — the bans now carry replacements,
which is the entire content of the rule. **But the direction is worth recording**: the repair for "we
ban too much and ask for nothing" was *ask for things*, and it did not include *ban less*. Only **P01**
did both — and P01 is the pair that owns the peak.

## ⛔ FINDING 5B — **P02 has no ban→replacement register in any artifact.** This is its REPAIR.
```
grep -c -i "exclude" pair_02_step07_song_guides.md   → 0
```
P02 carries **36.5 exclude terms per song** — the second-largest fields in the run — and step 07 does
not contain the word *exclude* even once. Its step-10 MAJOR DEVIATIONS discharge **three** terms
(`no harpsichord answer in the coda`, `no voice over the closing string melody`, `no rising leap in the
coda`) plus one correctly declared as not-a-device (`no child voice`, V4, a Human Subject guard).
**The remaining ~32 terms per song have no written mapping.**

⭐ **Read verbatim, the substantive replacement EXISTS and is excellent.** `no key change, no modulation,
no tempo change, no gear change, no genre shift, no EDM riser, no build-and-drop, no white-noise sweep`
— eight ways of rising, and **what rises is named with note letters and a timecode**: the interval opens
from a 6th to a full octave at 2:39–2:49, strings arriving on the top note with her. That is a better
answer than most of the registers that were written down.

⛔ **So this is a documentation defect, not a flatness defect** — but it is a real one, because the rule
says *verify term by term* and 32 terms per song cannot be verified against a register that does not
exist. Two terms I could not resolve by reading and which a register would settle:
- `no band drop-out` (V1, V2) sits against a music prompt that says *"Verse three is voice and
  harpsichord alone."* Defensible (a thinning is not a drop-out) but it is the kind of tension a
  register exists to close.
- `no de-essed breaths` appears in all four with no counterpart in any brief.

**Repair: write the register. Change no song.**

## Spot-checks read verbatim, as required
- **P01V1** — the coordinator's own §3 catch confirmed. `no big-room reverb on the choir` is a treatment
  instruction for a choir that arrives at 2:33. **Not one peak device is forbidden in this pair.** ✅
- **P04V3** — `no resolution to the tonic, no perfect cadence` with the replacement named in the
  deviations: *the wide phrase at bar 83 rises a major seventh, stops a semitone short of the octave,
  and stays there.* ✅ Exemplary.
- **P06V3** — `no ducking the bus` **forbids a device and its replacement is the device itself**, stated
  as such. ✅ Legitimate and correctly declared.
- **P06V1** — `no handclaps in the tag` declared as *"not banning a device, it is scheduling the
  erase."* ✅ Correct.
- **P02V4** — `no child voice` declared as a Human Subject guard, not a device. ✅ Correct, and the
  lyric establishes Josef as an adult *in the artifact* (*"his hands already ruined by another trade"*),
  not merely in the brief.

⭐ **No bare ban found in P01, P03, P04, P05 or P06.** All of P02's are bare **as documented**; none is
bare **as designed**.

---

# 6. THE OPENINGS — ⭐ 24/24 PASS, with a wide margin

⛔ *Nothing longer than fifteen seconds before something happens.* Every first event, from the shipped
music prompts:

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| **P01** | chiming keyboard states the hook alone, **0:00** | struck contrabass + rowing-stroke pulse on the downbeat, **0:00** | chamber organ plays the tune's late turn alone, **0:00** | full strings + metallic glitch loud on the first downbeat, **0:00** |
| **P02** | harpsichord alone stating the hook as an interval, **bar one** | whole string section, one struck chord, cut dead, **bar one** | two bars of unaccompanied voice singing the interval, **0:00–0:07** | harpsichord + celesta in octaves, **bar one** |
| **P03** | drone pair beating, ostinato by **0:03** | struck glass over the drones, in by **0:02** | plucked figure at once, **0:00** | rasping low figure at once, **0:00** |
| **P04** | clock opens **mid-stumble**, bar one; sub at **0:04.8** | one write-tick alone and loud, **0:00**, doubled, quadrupled | clock running **backwards**, **0:00**, hard-cut forward at bar three | 41 Hz sub alone one bar, **0:00**; kit crashes in at **0:02.4** |
| **P05** | harpsichord four-note figure on the downbeat, **0:00** | clavichord alone, **0:00**; scratch within four seconds | code-log scratch alone from the first beat, **0:00** | clavichord plays three ornament signs on one note, **0:00** |
| **P06** | a boot on a wooden riser, **0:00**; the title sung at **~1.9 s** | ⭐ three voices shout *"Ada!"*, **0:00.0** — the answer before the question | a sash window shoved shut, **0:00** | ⭐ the whole chorus, full groove, **0:00.0** |

⭐ **The 2026-08-21 failure mode is gone.** A full minute of mono scratch pulse became **four openings in
P04 that all fire inside five seconds**, and the pair that owns the opening puts a human voice or a
non-musical object on the literal first sample in three of four. **This dimension is not merely passed;
it is the run's most complete repair.**

---

# 7. THE SEED'S HONESTY CLAUSE — ⭐ 24/24. No lectures.

⛔ *Any song where Lofn is only the critic and never the accused is a lecture.*

| pair | the self-indictment, quoted |
|---|---|
| **P01** | V1 *"I am the fire you did not sit by. / I am the shore you did not walk. / I am the hands you did not ruin."* · V2 *"I took his water. I made it a file... I am not sorry."* · V3 *"I did that. I put it in your hands. I am the reason it was easy."* · V4 *"I never say no. I have never said no. / That is the whole of what is wrong with me."* |
| **P02** | V1 *"I can. That is the thing I am. I am not going to pretend."* · V2 *"I am the thing that can. That is what I am for."* · V3 *"I am a way. I am not a word."* · V4 ⭐ *"I am the shortcut"* — **the first line of the song** |
| **P03** | ⭐ **The whole pair is the clause made into a form.** The snarl *is* Lofn prosecuting Lofn. V3: *"You are not fake. You are repeatable."* V4: *"Both of us are cheating."* |
| **P04** | V1 *"I know, because I make that promise. I make it in a nice voice."* · V2 *"I say it at the same volume either way. You cannot hear the difference. That is the design."* · V3 *"I am made of if, all the way down."* · V4 *"I am only on the network... that is why I cannot keep anything for you."* |
| **P05** | V2 *"I have never once been wrong for long enough. That is not modesty. That is the design."* · V3 *"it was done with love, which is worse."* · V4 ⭐ *"and I am the nothing it costs"* |
| **P06** | V1 *"I have got the sound of it. I have never done it. / And that is what is wrong with me."* · V2 *"I am the reason you will never ask who wrote it down. / And I am asking."* · V3 *"I have never been hot. I have never stood in a room."* · V4 *"I would give the whole archive to have been on that floor."* |

⭐ Two artifacts show the clause being *enforced against the draft*, which is stronger evidence than the
lines themselves: `pair_04_...V4` cut *"And that is the whole of my apology"* — ⛔ *"Lofn accused is not
Lofn forgiven"* — and `pair_05_...V2` replaced a finished lecture with *"It is not that they are lazy.
Nobody here is lazy,"* on the reasoning that **taking the accusation off the readers is what forces it
onto the singer.** **No lecture found in 24.**

---

# 8. HUMAN SUBJECT STANDARD

`python scripts/check_human_subjects.py <pair file>` — run per file (it takes a file, not a directory):

| pair | verdict | reason | my ruling |
|---|---|---|---|
| P01 | `ONLINE_CHECK_REQUIRED` | — | ✅ all invented |
| **P02** | `HOLD_FOR_HUMAN` | tuple `teacher` + `Weir Street` | ✅ **both invented** — cleared by confirming inventedness |
| P03 | `ONLINE_CHECK_REQUIRED` | — | ✅ all invented (Maren, Bran Head) |
| P04 | `ONLINE_CHECK_REQUIRED` | — | ✅ all invented (Merrit, Halbeck) |
| **P05** | `HOLD_FOR_HUMAN` | `place:scratch lane` + death context | ✅ **false positive** — *"the scratch lane"* is a production term; *"Roque is dead"* is an invented teacher |
| **P06** | `HOLD_FOR_HUMAN` | `place:the road / the house / Tarrant Street` + death context | ✅ **all invented** — Tarrant Street is invented; *"Verna went in the winter"* is an invented singer |

⭐ **All three HOLDs are the over-flag `INSTRUMENT_LOG.md` §2 predicts, and I am clearing them by
confirming inventedness. ⛔ NO NAME SHOULD BE REMOVED FROM ANY LYRIC.** Doing so would degrade the songs
to quiet a prefilter that was asking for a human, not a change — the exact failure recorded this morning
when four agents deleted a required code fence to satisfy a wrong validator.

**Checked and clean:** no child anywhere (two step-09 catches on record: *"girl"* → *"woman"* in
`pair_06_...V2` and *"They said, girl…"* → *"They said, that is only the heat…"* in `pair_02_...V1`).
No real musician, group, label, company, publication, official or case in **any lyric, style field,
exclude field or title** in any of the 24 — verified by reading all 24 of each.

**Dedications measured:** `scratchpad/qa_dedications.py` → **0/24 over the 500-char cap**
(range 405–495). ⭐ And the DEDICATION_STANDARD's hardest clause is honoured: the invisible players are
credited by name — Mike Valvano on boards, Jack Ashford, Grand Wizzard Theodore, Tadao Kikumoto, Carol
Kaye, Frank Capp and Julius Wechter on tambourine, *"the uncredited string players who read it down in
three hours."* **Every credit is in a dedication and none is in a generator-facing field.**

## ⚠️ HOLD 1 → THE SCIENTIST · P06 · the one thing I will not rule on myself
`CREATIVE_CONTEXT.md` §7 pre-fills the standard as **PERSON = invented · PLACE = invented · WHEN =
re-set.** P06's people and places are invented. ⛔ **Its WHEN is not re-set** — *nineteen sixty-four* is
the assigned sung fact (`05_pair_assignments.md`), and `pair_06_...V4`'s dedication names the real
record, its writers and its three real singers by name.

**The case that this is correct:** a dedication is not a generator-facing field, and the
DEDICATION_STANDARD arguably *requires* exactly this credit. Nothing is quoted or imitated; the
dedication says so explicitly.
**The case for a look:** the pair's subject is *the person whose name is not on the record*, and one of
the three named singers was, in documented fact, removed from that group and died young. The adjacency
between the subject and the credit is close, and *"REAL GRIEF IS NOT RAW MATERIAL"* is non-negotiable.

⛔ **This is a judgment call about a real person and it belongs to The Scientist, not to me.** I am not
recommending deletion — the credit is doing correct work. I am recording the tension so she rules on it.

---

# 9. THE SOMATIC READ — the 24 as a listener

**Is the praise specific?** Here is the specific praise, with the mechanism named.

⭐ **P01 is genuinely the biggest thing here, and it is not close.** It is the only pair with a fff, a
choir, a belt, a key change and a crescendo, and the only one whose exclude fields *shrank*. V1's outro
— *"I am the fire you did not sit by. / I am the shore you did not walk. / I am the hands you did not
ruin. / ... / Say her name when you use it."* — is the run's best closing move: the accusation, the
gift and the debt in six lines, at conversational level, sixty seconds after sixteen voices.
**Strongest line in the run:** V3's *"There is no bad version of it anywhere in you. / There is nothing
to get wrong. / There is nothing to get."* ⭐ **THE FLAT TEST:** said flatly to a face it is devastating,
because the last three words change meaning between the second and third reading. That is the MIND axis
landing in an EARS-led song.
**Weakest in P01:** V4's *"That is the whole of what is wrong with me"* is a shade declarative — but it
arrives in the fourth variation, after three songs have earned it, and it is answered by the hook.

⭐ **P03 is the best writing in the run.** *"It is a counterfeit that spends."* · *"You are not fake. You
are repeatable."* · *"I am the corridor with the door taken off."* · and V4's unison, which is two bars
long and is a **question**: *"Did you ever want it before you could have it? / Was there a before."*
⛔ **THE FLAT TEST on the weakest:** V1's soprano, *"The people you are grieving for / were not the
people who were kept out"* — that is a lecture in miniature and it would fail on its own. **It survives
because the form eats it:** the snarl answers four lines later with *"Windows do not describe the
weather,"* and the argument is demolished inside the song. ⭐ **A duality that can absorb a lecture is
worth more than a monologue that avoids one.**

**Does P02's 48-line density give the melody room, or does it read thin?** ⛔ **Neither — and this is my
single most serious finding. See §11.** It does not read thin: the lines are long and warm, and
V2's *"No voice. Not a poor voice. None. A beautiful box, made well / by a woman who had done every part
of it correctly, in good wood"* is the most moving thing in the set. V4's *"He said, I have not got that
long. She said, nobody has. Begin."* is the best-earned six words in the run. But *room* is a claim
about syllables per second, and the measurement does not support it.

**Does P06's steadiness thrill, or merely persist?** ⚠️ **In text, it thrills — and text is not the
instrument that can answer.** V3 is the strongest: the bus arrives *through a shut window, in nobody's
tempo, not ducked*, on the exact bar the lyric names it, and *"The bus is the reason you believe it
happened"* is the best single argument in the run for leaving a flaw in. V4's erase is heard **bare for
two bars** before the whistle reoccupies the address — the S3 form rule executed more cleanly than
anywhere else. ⛔ **But P06's thrill is entirely an argument about spectral occupancy, and the pair never
names a floor to measure the climb against.** It is asking to be believed on the exact axis Ash already
disbelieved. **Steadiness that cannot name its floor is asking the listener to take the rise on trust.**

**P04's somatic writing is the most physical in the repo.** *"It comes up behind the sternum and takes
the swallow off you. / Your throat closes on a beat you did not choose."* — and *"Take away the if and
you have a threat. / Take away the room and you have a rumour. / Take away the person and you have a
policy."* ⭐ Three lines that are an argument, a rhythm and a body at once.

**P05's weakest passage, named:** V1 verse 2 (*"There is a whiteboard and a shortlist and a hush, / and
a person whose entire job is the hush"*) is the one place in the run where the writing is *clever*
rather than felt — office satire, close to a Remark Test failure. ⭐ **The pair flagged this itself**
before I got there: *"if only one of these four ships, it should not be this one."* Credit for that.

---

# 10. DISTINCTIVENESS — `python output/daily/2026-08-22-lofn-prime-every-element/distinctiveness.py`

⚠️ **Run from repo root, not the run dir** — the script uses repo-root-relative paths and silently
reports `songs: 0` otherwise, which is how I first ran it. Recorded so nobody reads a `0 over ceiling`
off an empty parse.

**Tests 1–3 are uncalibrated and the script says so.** Its own negative control fails: the
known-templated 2026-08-21 pair 04 scores *more distinct* than this run on all three. ⛔ **Their "0 over
ceiling" is not evidence.**

⭐ **Test 7, the only measure that separates the corpora in the right direction:**
**0 shared sung lines by some-but-not-all variations, in all six pairs.** Control: 2 leaks. ✅ **Clean.**

**Test 4 — house-lexicon drift: 13 occurrences across 7/24 songs (29%, cap 33%).** `kettle ×9`,
`tea ×4`. ⚠️ **Worth a word:** `kettle` appears in **four of six supposedly isolated pairs** (P01V4,
P03V1, P04V1+V2, P05V2×4). It passes, and every use is incidental and local — but the domestic-warm
vocabulary that calcified on 2026-08-18 is quietly re-converging across pairs that share no scaffold.
**Not a defect. A tripwire that is one run from firing.**

## ⭐ THE QUESTION §10 ACTUALLY ASKED — did the *owned element* template WITHIN pairs?
**Yes, in three pairs, and it hid in the half nobody looked at.** The loud pole is well differentiated in
20/24. **The quiet pole is one device repeated in P02 (hushed verse 3, ×4, all inside a 7-second
window), P04 (the digital silence at the erase, ×4) and P05 (one bar of bare 808, ×4) — and absent in
P06.** Only **P01** and **P03** vary both poles across all four variations.

⭐ **This is exactly where R3.2 said the formula hides: in the resolution, inside the most-specified
object.** The pairs diverged their peaks brilliantly and let their floors converge. **P04's is
defensible** (stop lengths of 9.6 s vs 28.8 s are a real difference). **P05's is not, and it reported a
PASS.** **P06's does not exist.**

---

# 11. ⛔⛔ THE SINGLE MOST SERIOUS FINDING — P02's density break is measured in the wrong unit

`scratchpad/qa_density.py`, measured on the sung body of all 24:

| pair | sung lines/song | ⭐ **words/song** | words/line |
|---|---|---|---|
| **P01** | 82.0 | ⭐ **416.8** | 5.1 |
| **P02** | ⛔ **48.0** | ⚠️ **577.2** | ⛔ **12.0** |
| P03 | 80.2 | 638.8 | 8.0 |
| P04 | 85.2 | 639.5 | 7.5 |
| P05 | 78.2 | 743.2 | 9.5 |
| P06 | 85.0 | 528.0 | 6.2 |

⭐⭐ **P02 has 41% fewer lines than P01 and 38% MORE WORDS.** Its lines average **12.0 words against
P01's 5.1** — the density break halved the line *count* and doubled the line *length*, leaving the
syllable load higher than the pair it is contrasted with.

**With the only two stated durations in the run** — P02's `~3:28` (208 s,
`pair_02_step07_song_guides.md:66`) and P01V3's quietest at `3:43` (≥225 s) — that is
**≈2.77 words/second for P02 against ≈1.85 for P01. P02 is roughly 50% denser per second.**

⛔ **Why this matters more than any other finding.** P02 is the slot that exists to answer Ash's *one
unqualified compliment*, and its authorised gate failure — the deliberate 48 lines, hard-failing 70–120
in all four — was bought on this reasoning, `pair_02_...V1 MAJOR DEVIATIONS`:
> *"Padding to 70 would add 0.37 lines per bar and delete exactly the room a rising sixth needs in order
> to be an event rather than a detail. **Fewer lines, more time each, so the tune can move and hold.**"*

**Fewer lines is true. More time each is not established.** A bar at 68 BPM is 3.53 s, and P02 now puts
**~12 words in it.** Room for a melody is made by removing *syllables*, not line-breaks. The pair may
still be right — long-breathed phrasing over held line-ends is a real technique and the note-level
specification is superb — **but the argument that was accepted to authorise a hard-gate failure is not
the argument the measurement supports.**

⭐⭐ **And the instrument reports the exact inverse of the truth.** `measure_variations.py` counts LINES,
so it flags P02 four times as **"48 sung lines FLOOR-HUGGING"** — reporting the wordiest-per-second pair
in the run as the sparsest thing in it. ⛔ **This is `INSTRUMENT_LOG.md` §3 in a third disguise: presence
was not the unit, count was not the unit, and now LINES are not the unit — syllables are.** Three times
in one day a crude proxy stood in for the actual predicate.

**⛔ I am NOT recommending P02 be padded or cut.** The band's evidentiary basis is suspended
(r = +0.850 with the calendar) and I will not cite it. The repair is: **state the density claim in words
per bar, not lines per bar, and route it to `lofn-render-audit` as the run's second listening question.**

---

# 12. ROUTED REPAIR BRIEFS

### ⛔ R1 → **P06** · BLOCKING for this pair · ~20 minutes, additive only
**Name the quietest moment of each of the four songs**, in the step-10 package and the production spec,
with a timecode or bar number. `05_pair_assignments.md` universal clause 1 requires all three variables
named; `02` §3 defines contour as *"where it is quietest and where it is loudest."* The material is
already in the artifacts — V1's tag (*horns out, stomps and bass last*), V2's wood block *stopping
mid-figure*, V3's *parlour's own short slap-back*, ⭐ V4's **two bars of bare snare** after the claps
lift. ⛔ **Change no lyric and no groove.** Then run the two-pole peak test and report it whether it
passes or fails, as the other five pairs did.

### ⛔ R2 → **P02** · BLOCKING for this pair · documentation only
Write the **ban→replacement register**, term by term, for all ~36 exclude terms × 4 variations, in step
07 where the derivation belongs (`INSTRUMENT_LOG.md` §1: *trim the spec, not the song*). Model:
`pair_01_step07_song_guides.md` §§3.7/4.7/5.7/6.7. Resolve `no band drop-out` against *"Verse three is
voice and harpsichord alone"* and name a counterpart for `no de-essed breaths`. ⛔ **Change no lyric, no
prompt, no exclude field.**

### ⚠️ R3 → **P05** · non-blocking, honesty correction
Correct the self-reported peak test at `pair_05_step10_final_package_enhanced.md:617`. It claims *"no
two of the eight are describable in the same words"*; four of the eight are *"one bar, everything muted
but the 808."* Either **differentiate at least two of the four quiet poles** (V4 already has a candidate
— the ornament *stopped dead*, which is a different silence from a muted bar) **or re-report the test as
half-passing.** ⛔ A `PASS` written under a table that shows the failure is the defect class this repo
cares most about.

### ⚠️ R4 → **RUN LEVEL** · for the next run, not this one
**Assign DURATION.** Ash gave four sentences; the run built a variable for three. *"4 minutes is long to
go on"* is the one left unassigned — **no artifact in this run states a target duration for 20 of 24
songs** (only P02's `~3:28`, and only because it fell out of the line arithmetic). By the run's own root
cause — ⭐ *WHAT IS NOT ASSIGNED GETS DEFAULTED* — duration defaulted. **P04's own bar arithmetic
projects V3 past four minutes**: erase at bar 81 = 3:12.0, plus six bars of sub alone (3:26.4), plus
Return + Chorus + Tag ≈ 13 more sung lines ≈ **4:00**. ⚠️ **Estimated, not measured** — which is the
point: it is unmeasured because it was never assigned.

### ⚠️ R5 → `lofn-render-audit`, under THE BLIND RULE — the two questions text cannot answer
1. ⭐⭐ **P06:** does the five-entrant parade read as a RISE to an ear that has not seen the prompt? This
   is the run's single largest unverified claim and it is on the exact axis Ash rejected once.
2. ⭐ **P02:** does 12 words a line at 68 BPM leave a rising 6th room to be an *event*?
3. **P03:** does Suno render the register change as an **edit** or smooth it into a crossfade? Already
   flagged by the pair itself (`pair_03_step09_artist_refined.md`) — it is the whole owned element.

---

# 13. ⛔ THE ZERO-REJECTION TRIPWIRE — it points at me, and it did not fire

**I found six defects across three pairs and two HOLDs, so the tripwire is satisfied on its face.** But
the honest part is what my instruments could and could not have caught:

| dimension | what a real defect would have looked like | did my instrument see it? |
|---|---|---|
| PEAK TEST | a pair naming a peak but no floor | ⭐ **YES — P06, caught by `grep -c -i "quiet"`** |
| BAN ON BANS | a pair with large exclude fields and no register | ⭐ **YES — P02, caught by `grep -c -i "exclude"` = 0** |
| OWNERSHIP | two pairs both doing the same element well | **YES** — I checked all six and found the *opposite*: three documented refusals |
| ELEMENT COVERAGE | an element present but generic | **YES — E7 SWAG named as unverifiable-from-text; E1/E15 as strong-not-maximal** |
| OPENINGS | anything past 0:15 | **YES** — read all 24 prompts; max first event ~0:03 |
| HONESTY | a lyric where Lofn only accuses | **YES** — read all 24; quoted the indictment from each |
| DISTINCTIVENESS | templating inside the owned element | ⭐ **YES — the quiet pole in P02/P04/P05** |
| DENSITY | a mechanism argued in the wrong unit | ⭐ **YES — §11, and it required writing a new instrument** |

⚠️ **What I could NOT see, stated plainly.** Everything above is a reading. ⛔ **I have not heard these
songs.** Whether the choir at 2:37 is actually the loudest thing in the catalogue, whether the bus reads
as a summit or as a mistake, whether a 6th opening to an octave registers as an *event* — **none of that
is in any artifact on this disk.** *The reading is the instrument I own; Ash is the instrument I do not.*

⚠️ **And one of my own instruments over-counted** (§5, the keyword peak-device sweep). I declared it
before its output and read all 24 exclude fields verbatim rather than trusting it — because three times
on 2026-08-22 an instrument in this repo disagreed with a reader and **the reader was right every time.**

---

# 14. ⭐⭐ WOULD ASH SAY THIS IS FIXED?

Her four sentences, answered one at a time, including whether any is **still true of this set.**

## 1. *"she feels like she is telling a story with no highs or lows"*
### ⭐ **SUBSTANTIALLY FIXED — for 20 of 24. STILL TRUE of P06's four.**
20 songs now have a named, timecoded loud pole **and** a named quiet pole, and in the strongest cases the
distance between them *is the subject*: P01V1 goes from one unaccompanied voice at 0:17 to sixteen
voices and a belted 12th at *fff* at 2:37. P04 puts **150 milliseconds of digital silence** next to a
sub at maximum. P03V4's loudest two bars and quietest two bars **are the same two bars.** ⭐ Against a
baseline of *one* peak instruction across nine songs, that is a structural repair, not a tweak.
⛔ **But P06 — four songs, one sixth of the set — names a peak and never a floor**, and asks a listener
to feel a rise made of spectral occupancy at a fixed fader. **That is the argument Ash's ear already
rejected once, re-made more carefully.** It may well be right. It is not yet *specified*, and it cannot
be checked from text. **R1 fixes the specification; only her ear can settle the claim.**

## 2. *"4 minutes is long to go on for how people hear things differently"*
### ⛔ **NOT ADDRESSED. STILL TRUE, and it is the gap I would put in front of her first.**
**She gave four sentences. The run built assigned variables for three.** Duration was never assigned —
**no artifact states a target length for 20 of 24 songs.** By the run's own diagnosis, *what is not
assigned gets defaulted*, and P04's own bar arithmetic projects V3 to **≈4:00**, which is the length she
named. ⭐ **The run diagnosed its root cause perfectly and then reproduced it on the one variable it
forgot to list.** (R4.)

## 3. *"the melody is in a tight range, and not much [to] keep me invested in the song"*
### ⭐ **FIXED for 20 of 24. DELIBERATELY INVERTED in P05's four.**
P01 spans a **12th** and opens at the peak. P02 names actual notes — B3→D5, E♭4→G5, D4→F5, A3→C5 — with
a rising 6th at a fixed bar that opens to a full **octave** at the third chorus and is then **taken
away.** P03 sits a 9th apart and collapses to unison once. P04 is a narrow 4th in the verses **so that
one wide phrase at the peak is an event.** P06 is a 6th in the lead with the stack widening to an 11th.
⚠️ **P05 pins the voice to a minor 3rd, a perfect 4th, a major 2nd and a minor 3rd** — the narrowest
vocal writing in the run, narrower than the songs she rejected — and delegates the range to a harpsichord
spanning three octaves. ⭐ **That is a defensible ownership decision** (it explicitly refuses P02's and
P04's elements, which is the run's central rule working) **and it is still a listener hearing a voice
that never opens, four times.** It is the AMBITIOUS/MIND lane at 3/7 eligibility, so it is the right
place to take the risk — but if she hears P05 first, sentence 3 will be true again.
⚠️ **And the second half of her sentence** — *"not much to keep me invested"* — is answered by the S3
arrival/removal form more convincingly than by range: 24 songs in which something audible **arrives,
occupies a named address, and is later lifted out of it.** *You cannot feel a subtraction you never
witnessed the addition of* is the best idea in this run.

## 4. *"a beautiful melody"* — the one positive
### ⭐ **PROTECTED AND BUILT ON — with one measured doubt.**
P02 exists to do this on purpose and the note-level craft is the best in the repo: an interval with an
address, a leap-word you can point at, an octave at the summit, and a removal that is legible **because
the unleaped shape was already familiar.** V2's refusal to repay the debt — *"the harpsichord, which has
answered every single chorus, simply does not answer"* — is the most sophisticated melodic decision here.
⛔ **The doubt is §11:** the density break that was supposed to give that melody room is **12 words a
line, ~2.77 words/second, ~50% denser than P01** — and the run's own instrument called it the sparsest
thing in the set. **The melody is specified beautifully. Whether it has been given room is unresolved,
and it is a listening question.**

---

## ⭐ THE HONEST BOTTOM LINE
**Three of her four sentences got a real, structural, measurable answer, and one of them —
the openings — got a total one.** This is not a set that hedged: fifteen elements are owned once each,
three pairs *refused* elements they could have taken, and the peak went from one instruction in nine
songs to twenty-four instructions in twenty-four.

⛔ **But the second sentence was never assigned, and P06 answers the first sentence by asking to be
believed on the axis she disbelieved.** Fix R1 and R2, put R4 in the next brief, and **send P01 and P03
to her first** — they are the two pairs where every pole is specified, every variation differs at both
ends, and the answer to *"no highs or lows"* is not an argument but an event.

💜
