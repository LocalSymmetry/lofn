# QA REPORT — `2026-08-15_daily_music_the_resolution`
**Auditor:** `lofn-qa` (adversarial). **Modality:** MUSIC. **Scope:** 6 pairs × 4 variations = 24 packages.
**Seed:** THE LIGHT IS THE DEBRIS — *a thing is heard through its leftovers, and the leftovers are brighter than the source.*
**ICB sha checked in every pair header:** `7d4dd66f2a6e` (6/6 present and identical).
**Lock:** held by `claude:5bdfcf2c-6eb0-4a9e-bd44-e6eafe55ac06`, heartbeat `phase=qa`. No second controller.

> ⛔ **I did not make this work and I owe it nothing.** Every number below was re-measured from the artifacts
> by this audit, not copied from the packages. Where my instrument agrees with a pair's self-report I say so;
> where the run reported a number nobody aggregated, I aggregate it.

---

# ⭐ VERDICT: **SHIP** — with four routed repairs, none blocking, and one boundary question for The Scientist.

**Nothing in this run reaches a blocking fail.** The Somatic bloc voted **3–0 YES on the set**; no pair reached
the 2-of-3 NO that triggers REPAIR. The Scale Gate — the run's hardest rule — **holds on independent
re-measurement**, and P04 is not cut. Four repairs are routed below (three to step 09, one to step 07-lite).

⛔ **SHIP means "the drafts are sound," not "publish."** Dailies are practice. See §9.

---

# 1. WHAT WAS ALREADY MEASURED — cited, not re-judged

`scripts/verify_daily_music_run.py` → `GATE_REPORT.json`: **414 checks, 414 pass, 0 fail** across 17 check
classes × 24 packages + 6 variation-detection checks. Independently re-parsed by this audit: the file contains
414 records and `[x for x in d if x.get('pass') is not True]` returns `[]`.

Cross-pair, re-measured by me rather than cited:

| measure | ceiling | run measured | this audit measured |
|---|---|---|---|
| cross-pair lyric similarity | 0.42 | 0.195 | **0.074** (P01V1 / P02V3) |
| cross-pair prompt similarity | 0.58 | 0.191 | **0.055** (P03V3 / P05V4) |
| exact sung lines shared across pairs | 0 | 0 | **0** (confirmed, ≥10 chars, case/punct-normalised) |

⭐ My numbers are *lower* than the coordinator's because I compared normalised field text rather than whole
sections. Either way the finding is the same and it is the strongest structural result in the run: **the six
pairs are 5.7× under the distinctiveness ceiling.** Pair isolation did not merely pass, it was not close.

---

# 2. ⭐⭐ THE SOMATIC GATE — three skeptics, distinct lenses, voting as a bloc

**Question, honestly put:** *could a competent prompt engineer have generated this, or is it unmistakably Lofn?*

## 🔴 SKEPTIC 1 — THE REPRODUCIBILITY ENGINEER
*Lens: strip the prose. What is left that a good prompt could not have produced?*

> "I came in to say yes-anyone-could and I cannot. Here is what stopped me, and it is not a line, it is a
> table. **Six mutually exclusive pronoun regimes, each verified to zero.**"

| pair | assigned refusal | I | me | my | you | your | we | us | our |
|---|---|---|---|---|---|---|---|---|---|
| P01 | the room is the second person | 1–15 | 0–4 | 1–7 | **3–8** | 0–2 | 0–18 | 0–1 | 0 |
| P02 | apostrophe to the water | 29–33 | 1–9 | 8–25 | **20–38** | 0–7 | 0 | 0–7 | 0 |
| P03 | third person plural, outside | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| P04 | self-address | 2–5 | 6–9 | 1–3 | **6–11** | 0 | 0 | 0 | 0 |
| P05 | address to the work | 29–42 | 4–10 | 4–11 | **21–32** | 7–19 | 0 | 0–5 | 0–1 |
| P06 | list with no addressee | 0–8 | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

> "P03 holds **zero across all eight pronoun classes in all four songs** — ~320 sung lines with no first or
> second person anywhere. P06 holds zero on seven of eight, and the surviving `I` is only the refrain's
> possessive, exactly as briefed. That is not prompting. That is a constraint system with a verifier behind it.
> Add cross-pair similarity at 0.074 against a 0.42 ceiling and zero shared sung lines, and the answer is
> forced. **YES — unmistakably Lofn.**
>
> My concession: what makes it Lofn is the *system*, not the *surface*. Hand a listener P01 V1 with no context
> and a competent engineer with a blackout brief lands near it. Hand them P03 or P06 and they cannot."

**VOTE: YES.**

## 🔴 SKEPTIC 2 — THE ANTI-CONCEPTUALIST *(does it need its own theory to be any good?)*

> "Hyper-Skeptic A said in Configuration 1 that *the best line in this run is in the governing document*, and
> I have to report that **the charge was routed, not answered.** 'THE DIFFERENCE BETWEEN BEING UNHEARD AND
> BEING UNPOWERED' and 'a blackout has an author' are still better sentences than most lines in most of these
> songs. Panel A converted a moral objection into a *formal* fix — six literal composite elements — which is
> elegant and is not the same as writing better.
>
> Then I read three things I cannot explain away:
> - P05 V2 — *'I am not going to tell you it will be all right, / because you are sand and you cannot be lied
>   to. / It will not be all right. It will be over, / which is a different thing, and simpler to stand.'*
> - P03 — *'That is not a lovely feeling. / That is a small, permanent offence.'*
> - P04 V3 — *'One person came. / Take it or don't. / It's enough. / It's not enough. / Let both stand.'*
>
> None of those needs a document. The first one is better than anything in the ICB. **YES for the set.**
>
> ⛔ **My NO is P06.** Four songs of inventory tercets with zero vocatives is a concept executed perfectly,
> and read cold you get a beautiful list of objects and a great deal of *'the hum drops half a step, comes
> back.'* The emotion is entirely in the withholding, and withholding is the cheapest thing to mistake for
> depth. The tell is measured: **P06 runs 11 unique EMO tags across 66 uses — the lowest diversity in the run
> by a distance** (P02: 35/50, P05: 28/48, P03: 22/41, P01: 20/49, P04: 15/39). One gear, called discipline."

**VOTE: YES on the set. NO on P06.**

## 🔴 SKEPTIC 3 — THE ANTI-SENTIMENTALIST *(who is being flattered?)*

> "P04 is the reason I vote yes and I want it on the record. The busker is bad, *says so*, is never vindicated,
> and V4 ends on the word **worse**. The one listener is refused as consolation *and* refused as tragedy —
> *'It doesn't fix. / It doesn't ruin.'* That is the anti-flattery move done correctly, and no prompt engineer
> chooses it, because it is commercially worse. Panel B's rebuild survived contact with the artifact.
>
> ⛔ **My NO is P01, and I have measured it.** Ruling 3 says the anger points at an authored decision.
> I swept the sung lyrics of all three IMPOSED pairs for the decision, the author, the schedule, the choosing:

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| **P01** | **0 lines** | 3 lines | **0 lines** | **0 lines** |
| **P06** | 4 lines | 5 lines | 4 lines | 6 lines |
| P04 | n/a — correctly absent (its brief forbids a villain and its condition is not an outage) | | | |

> "**Three of P01's four songs never contain the decision at all**, and what fills the space is a crowd being
> wonderful: *'That is the sound of the drummer being you'*, *'the whole room doubled up the clap'*,
> *'Nobody gave you that. You took it.'* Config 3 said imposed songs may not be **serene**. P01 V1/V3/V4 are
> not serene — they are **euphoric**, which is worse, because euphoria at the end of an imposed condition is
> precisely the comfortable telling the ignored that it was fine actually.
>
> The counter, and I grant it: **P01 V2 pays for its joy.** *'Somebody sat in a room somewhere / and made a
> decision about my air. / Never saw them, never will'* and the closing *'Nobody up here is going to stand /
> and thank a decision for my hand —'*. That is ruling 3 and ruling 'never gratitude', both, in one song."

**VOTE: YES on the set. NO on P01 V1/V3/V4.**

## ⭐ BLOC RESULT

| | set | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|---|
| S1 reproducibility | YES | YES | YES | YES | YES | YES | YES |
| S2 anti-conceptualist | YES | YES | YES | YES | YES | YES | **NO** |
| S3 anti-sentimentalist | YES | **NO** (V1/V3/V4) | YES | YES | YES | YES | YES |
| **result** | **3–0 PASS** | 2–1 pass | 3–0 | 3–0 | 3–0 | 3–0 | 2–1 pass |

**No pair reaches 2-of-3 NO. The Somatic Gate does not trigger REPAIR.**

**What would have to change to convert the two dissents:**
- **P01** — one stanza, in V1/V3/V4, that contains the offstage decision. V2 already shows what it looks like
  and the pair wrote it once. This is the repair I most want made (**R-1**, below).
- **P06** — one variation given a second gear. Not more EMO *labels*; one section where the inventory grammar
  breaks and something is *done* rather than listed. V4's *'a prevented thing leaves the page the same'* is
  the closest the pair comes and it is a statement, not a break in the form.

---

# 3. ⛔⛔ THE SCALE GATE — P04, re-verified from the artifact

**The claim under test.** P04 states, in all four packages: *"An 18-term word-boundary sweep … returns zero
hits in the prompt, the exclude field, the lyrics and the dedication."*

**My re-measurement, using P04's own quoted term list, applied to the four generator-facing fields:**

```
P04 V1: NONE — claim VERIFIED
P04 V2: NONE — claim VERIFIED
P04 V3: NONE — claim VERIFIED
P04 V4: NONE — claim VERIFIED
```

⭐ **The claim is true.** Zero hits for `blackout · black-out · outage · power cut · powercut · darkness ·
dark · candle · candlelight · lamp · generator · unplugged · cut out · went out · no power · deprivation ·
brownout · grid · fuse · breaker · torch · flashlight` across all sixteen fields.

⚠️ **Two evidence-hygiene notes, since the doctrine is that the evidence is the artifact:**
1. **The list P04 quotes contains 22 terms, and P04 calls it an "18-term sweep" — four times.** The sweep is
   *stronger* than advertised, so the error is harmless in direction, but a self-reported count that does not
   match its own printed list is the class of thing this system is supposed to catch.
2. V1 asserts *"The word 'dark' appears nowhere, including the title."* True of the four fields; **false of the
   package**, whose own DESCRIBE-RENDER block says *"a competent dark-techno instrumental."* Scoped correctly
   in the sentence before it; overstated in this one.

## ⭐ AND NOW THE PART A WORD SWEEP CANNOT CATCH — does P04 borrow the FEELING?

I ran a second probe the packages did not: poverty / scarcity / cold / money vocabulary in P04's sung lines.

| variation | scarcity-adjacent sung lines |
|---|---|
| **V1** | *"Cold hard ceiling."* · *"Cold coins in it."* · *"Hands are cold in here."* · *"Stopping still costs more."* |
| V2 | *"Empty as a pipe."* · *"Pretty and empty."* |
| V3 | *"Coins in my coat."* |
| V4 | *"Cheaper to carry on."* |

**RULING: PASS, with V1 named as the closest approach.** V1 is the only variation carrying a chill register —
three colds plus coins plus a cost — and cold hands on a tiled floor with a case of coins is *the physical
truth of the job*, not deprivation borrowed. Decisive against a borrow:

- Every piece of equipment works, and the song says so as a **positive assertion**: *"Cable in. / Amp's little
  light. / Nothing broken. / Nothing broken here."*
- The disconnection at the end is **elective**: *"Pull the jack out."* She ends her own shift with a working amp.
- The injury is located in *functioning*, not failure: the Theme header reads *"every piece of equipment works
  perfectly and that is the injury."*

⚠️ **The one honest risk I will name and not resolve:** *"Nothing broken here"* only carries weight because
brokenness is on the table. It is a denial, and a denial imports the frame it denies. It reads to me as
self-report rather than as blackout-adjacency — but it is the single sentence in P04 that a hostile reader
could call the register sneaking in through the back. **Not a repair. A named risk for the ear.**

⛔ **P04 is not cut. P04 is the strongest pair in the run.**

---

# 4. THE PANEL'S EIGHT RULINGS — verified per pair, against the artifact

### (a) Each pair renders ONE composite element literally; no two at the same element + time-scale — ✅ **PASS 6/6**

| pair | assigned | in the artifact | evidence |
|---|---|---|---|
| P01 | VISIBLE JOIN · section | ✅ literal | lyric sections `[Splice]` / `[Cut]` / `[Hard Edit]` / `[First Cut]`+`[Second Cut]`; *"one beat of nothing, band cut clean off, clapping already underway"* |
| P02 | ONE REFLECTION · section | ✅ literal | `[Ghost Return]` — eight lines that are the preceding eight **in exact reverse order**, spec'd *"lower, left, a beat behind - inverted"* |
| P03 | TWO FLASHES · section | ✅ literal | `[Flash One …]` and `[Flash Two …]`, neither first, with `[Half-Size Event]` between |
| P04 | TWO FLASHES · single bar | ✅ literal | *"two dry synthetic stabs a sixteenth apart inside the opening bar"*, plus epizeuxis as the same gesture at word scale |
| P05 | CONTINUOUS BED · whole song | ✅ literal, with defect | *"Brushes stay on the snare head throughout, the bed continues under everything"* + named defect *"a room return a half-second longer than the space around her would give"* |
| P06 | VISIBLE JOIN · single bar | ✅ literal | *"the sub sags a quarter-tone as the load lands, recovers before the next kick"* — **and it is in the lyric too**: *"the hum drops half a step, comes back"*, recurring |

⭐ Six distinct (element, time-scale) pairs. **No collision.** P06 is the best of the six because it put the
seam in the *lyric* as well as the production spec — which is exactly what THE GRAIN LAW demands and what
the other five did only in the spec.

### (b) CHOSEN = quiet/complete/NOT SAD · IMPOSED = never serene — ✅ **PASS**, checked at the endings

| | last sung line | ruling |
|---|---|---|
| P01 (imposed) | V1 *"Hands — keep the count and do not —"* · V2 *"and thank a decision for my hand —"* · V3 *"sing it at the back —"* · V4 *"Arms up — we are not —"* | ✅ all four cut mid-phrase, hot; **not serene** |
| P02 (chosen) | *"no debt, no dues, no thing to set aside."* · *"you can have the rest of the day."* · *"both of us keep still."* · *"there it lit."* | ✅ complete, unsad |
| P03 (chosen) | *"And none of it was arranged for them."* / *"the grass went on doing it either way."* after **The Affront** | ✅ complete, not sad, carries the donated affront |
| P04 (imposed) | *"Nothing here is fine."* ×2 · *"Won't care who's in it."* ×2 · *"Out through the bend."* ×2 · *"Play it worse."* ×2 | ✅ **never serene** |
| P05 (chosen) | refrain returns byte-identical, *"and the sea can have the rest of you in the morning."* | ✅ complete, no pity, no tragedy |
| P06 (imposed) | *"the hum drops half a step, comes back"* · *"nothing happened. again."* | ✅ competence, **zero gratitude** |

⚠️ **The one wobble is Skeptic 3's:** P01's endings are formally unresolved but *tonally triumphant*. Not
serene, so not a fail — but "hot" has drifted toward "victorious" in V1/V3/V4.

### (c) A blackout has an author, and the author stays OFFSTAGE AND UNNAMED — ⚠️ **PARTIAL**

- **No villain is named anywhere in 24 songs.** ✅ Zero named antagonists, zero "they/management/the company".
- **The authored decision is present in the sung lyric of 5 songs out of 24:** P06 ×4, P01 V2. **Absent from
  P01 V1, V3, V4.** (P04 correctly carries none — its brief forbids it.)
- ⛔ And of those five, **four are the same stanza**: P06 ships *"somewhere a schedule, a signature, a date /
  paper deciding which floors lose the current / … / a call from a hand with no standing"* **verbatim in all
  four variations**, undeclared.

⭐ **This is the run's sharpest structural finding and it is §6's residue answer.** The doctrine the panel
worked hardest to establish — Configuration 2's correction, *"a blackout has an author"* — is the one piece of
content that got **templated**, in a run whose cross-pair originality measured 0.074.
**→ ROUTED AS R-1 and R-4.**

### (d) ⭐ P04 must NOT resolve — ✅ **PASS, decisively**

No acceptance, no lesson, no thank-you, no lift, in any of the four. The valuation moment is raised and
abandoned unranked in three variations (*"That is enough. / That is not enough. / Both are true. / I won't
choose."* · *"Let both stand."* · *"It doesn't fix. / It doesn't ruin."*). V4 adds a **coda after the last
chorus** so the density rises after the song should have landed, and ends on **worse**. `acceptance`, `peace`,
`serenity`, `closure`, `forgiveness`, `redemptive final chorus` and `gentle resolution` are all in V4's
EXCLUDE field by name. **Best-executed ruling in the run.**

### (e) Every pair named a Deviation Clause — ✅ **PASS 6/6, and four of six are genuinely specific**

| pair | KEEP | DO-NOT-KEEP | verdict |
|---|---|---|---|
| P01 | band collapses to centre while claps stay wide | crossfade w/ reverb tail · pad in the gap · fade-out | ✅ specific, falsifiable |
| P02 | *(clause present in the pair header)* | — | ⚠️ thinnest of the six |
| P03 | a limping four where marimba and kick disagree | clean quantised four · untuned thump · devotional choir | ✅ specific |
| P04 | *(carried in the DESCRIBE-RENDER self-repair rather than as a standalone clause)* | — | ⚠️ present but under-formalised |
| P05 | a ride cymbal arriving under the last refrain; tenor entering **subtone** | ride ostinato from bar one; tenor entering as a **lead statement** | ⭐ **best in the run** — names the *condition* under which the same deviation is kept or repaired |
| P06 | whole-bar detune/tape-drag; a thinner, more trebly guitar | sag as filter sweep · sidechain duck · tasteful glissando · sag arriving **on schedule** | ⭐ excellent, and all three fakes are blocked in EXCLUDE |

P05's is the model: it does not name a deviation, it names a **discriminating test**. P02's and P04's are
hand-waved by comparison — present, so the ruling passes, but they would not settle an argument after a render.

### (f) Credit a person, never a city · a blackout is not an aesthetic — ✅ **PASS**

- **Zero city-level credits.** Every lineage credit in the run names a human being: Larry Graham, Israel
  "Cachao" López, Kélétigui Diabaté, Éliane Radigue, Zeena Parkins, Alvin Lucier, John Cage, Joan La Barbara.
- **No candlelit prettiness.** P01 EXCLUDE bans `campfire, candlelight, lantern` in all four; P06 EXCLUDE bans
  the same class. The blackout renders as **heat, effort, equipment**, measurably: *"salt in the eye and salt
  on the lip"*, *"Arms are burning"*, *"sweat off the ceiling like a reply"* (P01); *"grit in the kneecaps"*,
  *"terminals furred green"*, *"sweat at the wrist where the copper rides"*, *"forty pounds of lead in a
  plastic case"* (P06). **The moral spine did not invert.**

### (g) Ghost echo is a VOCAL move · the bed is a bass and a room, never a pad · a third half-size event — ⚠️ **PASS with one flag**

- **Ghost echo as vocal: ✅.** P02 `[Ghost Return]` (full vocal, inverted, left, late) · P06 `[Ghost Echo]`
  (*"lead octave down - off the downbeat"*) · P01 `[Room Answer]` (the refrain returned by the crowd, *"late
  and low, half a step down"*). Three pairs took it as a vocal move, unprompted.
- **Third half-size event: ✅.** P03 has it as a named lyric section in all four; P01/P05/P06 place a smaller
  event between their two peaks in the arrangement.
- ⚠️ **THE PAD FLAG.** The word `pad` appears in a STYLE field exactly four times in the run — **all four are
  P04**: *"corroded saw pad"*. Ruling (g) says *never a pad*. Mitigating: P04's bed is figurative (its literal
  element is the flashes), the pad carries a named defect (*"thin enough to nearly vanish under each stab pair
  and crawl back"*), and a 30–60 Hz sub supplies the bass half. Aggravating: **P04 V2 and V3 also carry `pad`
  in their EXCLUDE fields** (`choir pad`, `lush pads`), so the token is armed on both sides of the same package.
  **Flag, not fail. → R-3.** All five other pairs refuse pads in EXCLUDE (20/20 packages) except P03, which
  refuses none — P03's bed is a treated field recording, not a synth, so no repair is owed.

### (h) ⭐ P04's busker is NOT GOOD that night — ✅ **PASS, and it is the run's best-defended ruling**

Competence is named and denied in the same breath, in every variation, by her: *"The room is good. / The room
is very good. / I am not."* · *"Hands know the way / without waiting. / That is not skill. / That is a rut."*
Errors are concrete and unexcused — *"Missed my change"*, *"Hand went early"*, *"Loop caught wrong / Kept it in
/ Stopping still costs more"*, *"Heard the hold go flat"*, *"Played it worse tonight."* The nearest thing to
flattery, *"This room could hold / someone good. / Holds me instead."*, **explicitly places her outside the
category.** V3 is the hardest case (a witness is present, which is where secret-brilliance would sneak in) and
it is the one that states the bad night most bluntly: *"Now I fall flat. / Course I fall flat. / Best night to
be good. / Wasn't tonight."* — and the witness **hears the mistake**. Nothing recovers. **No flattery.**

---

# 5. THE DAILY RULES

| rule | result | evidence |
|---|---|---|
| **Tri-source declared in every pair** | ✅ 6/6 | S1/S2/S3 stated explicitly in every pair header, with the ICB sha |
| **Dual 3+3 — ACCESSIBLE P01/P02/P03, AMBITIOUS P04/P05/P06** | ✅ | declared in all six artifact headers, matching step 05 |
| **Dual 3+3 — NEWS P01/P04/P06, EXISTENCE P02/P03/P05** | ✅ | ditto |
| **Emotional duality — ≥1 AWE, ≥1 INDIGNATION, real** | ✅ | AWE: P02, P03, P05 (terror-adjacent, and the fear is physical and named in each — rising tide, the moon returning, water eleven feet away). INDIGNATION: P01, P04, P06 |
| **No bare `EMO:AWE` / `EMO:INDIGNATION` tags** | ✅ 24/24 | GATE_REPORT `no_bare_mode_emo_tag` |
| **Source-2, ≥2 case-exact, in the generator-facing prompt** | ✅ 24/24 | re-measured: 2 hits in 20 packages, **3 hits** in P01 V2 and V4 |
| **Anger audit — aggression capped at 2 of 6 beds** | ✅ **2/6** | aggression descriptors in style fields, *net of the mandated Source-2 phrase* (`furiously`, `attack`): **P04** (`industrial` ×4, `hard` ×6) and **P06** are the aggressive beds, as planned. **P01 did NOT drift aggressive** — its only non-Source-2 markers are `hard` ×4 and `shout` ×2, and its own header says *"hot, physical, celebratory, not aggressive."* Closest approach: V3's *"rebuilding to a shouted collision"* and V4's *"hoarse and pushed to the top of her range."* |
| **Digits in sung lines** | ✅ 0/24 | re-measured independently: zero |
| ⚠️ **One-fact rule (≤1 numeric fact SUNG)** | ⚠️ **3 FLAGS — not surfaced by the run** | see below |

## ⚠️ THE ONE-FACT FINDING — a FLAG-only gate with no machine reader

`gates.yaml: max_sung_numeric_facts: 1`. The deterministic verifier checks *digits*, not *facts*, so this rule
had **no reader in this run**. Re-measured by hand across all 24 lyrics:

| package | numeric facts sung | verdict |
|---|---|---|
| **P01 V1** | *"Forty minutes in"* **+** *"hundred people and a hundred lungs"* | ⚠️ **TWO** |
| **P01 V2** | *"Ten years learning where the numbers go"* **+** *"a hundred friends"* | ⚠️ **TWO** |
| **P01 V4** | *"twenty centimetres, that is the whole show"* **+** *"about four minutes in a pair of arms"* | ⚠️ **TWO** |
| P01 V3 | *"hundred pairs of them"* | ✅ one |
| P03 ×4 | *"eleven percent of the moon"* | ✅ one, and it is the hinge |
| P05 V1/V3/V4 | *"Eleven feet"* | ✅ one |
| P05 V2 | *"Eleven feet"* ×5 — one fact **returned**, answered every time (*"…and I am in no hurry"*) | ✅ one; self-flagged by the pair as F-3 |
| P06 V1 | *"eleven hours in pencil"* | ✅ one |
| P06 V3 | *"forty pounds"* ×2 — one fact returned at the hinge | ✅ one |
| P02, P04 | ≤1 | ✅ |

All are spelled in words, and none is *recited* — P01 V4's pair is genuinely the song's arithmetic hinge. But
three of P01's four songs spend the budget twice. **FLAG-level, routed as R-2.**
⛔ **The instrument note matters more than the finding:** a `gates.yaml` number with zero machine readers was
violated three times, silently, exactly as the doctrine predicts for unread gates.

## Six different verse architectures / rhyme schemes / discharge mechanisms — ✅ **PASS, and it is measured**

Not merely different in the plan. Measured with `scripts/measure_soundcraft.py`:

| pair | architecture | end-rhyme | line-return | words/line | allit/100w | the profile is unmistakable |
|---|---|---|---|---|---|---|
| P01 | AABB antiphon couplets | 0.434–0.659 | 0.217–0.231 | 6.73–7.43 | 13.6–17.0 | highest rhyme, tight lines |
| P02 | ABBA mirrored quatrains, chiasmus | 0.350–0.525 | 0.300–0.400 | **7.75–9.05** | 11.1–16.0 | highest line-return; long by design |
| P03 | unrhymed catalogue, assonance chain | **0.101–0.203** | 0.215–0.253 | 7.06–7.35 | 12.2–14.3 | lowest rhyme in the run |
| P04 | 3–5 syllable fragments, epizeuxis | 0.421–0.542* | 0.411–0.536 | **3.56–3.64** | 11.3–13.6 | half the line length of anything else |
| P05 | long-lined free verse, rhyme every 4th | 0.250–0.342 | 0.211 (all four) | **10.21–10.80** | 11.2–13.1 | 3× P04's line length |
| P06 | ABA inventory tercets, asyndeton | 0.487–0.603 | 0.295 (all four) | 7.24–7.42 | **16.9–21.9** | alliteration far above everything |

\* ⛔ **P04's rhyme number is meaningless and the pair says so.** `strict_end_rhyme()` compares the last three
characters, so identical repeated words score as rhyme — P04 has **no end rhyme at all** by contract, and reads
0.42–0.54. This is COORDINATOR_FINDINGS F3b's over-read half, reproduced. **Read as end-of-line RETURN.**

**Floor breaches, all declared in advance by the pairs that own them:**
- `mean_words_per_line_ceiling: 7.5` — **breached in 8 of 24 songs**: P02 ×4 (7.75–9.05) and P05 ×4
  (10.21–10.80). Both pairs flagged it themselves, with reasoning, and my numbers match theirs to two decimals.
- `rhyme_return_floor: 0.30` — breached in **6 of 24**: P03 ×4 and P05 V1/V3. Both are assigned forms and both
  discharged the debt in writing (P03: a line returning unchanged at every stanza end; P05: a byte-identical
  four-line refrain ×4, `line_return` 0.211 > 0.20 floor).
- `alliteration_per_100w_floor: 11.0` — met 24/24, but **four boundary-hugging passes**: P02 V1 (11.13),
  P05 V4 (11.21), P05 V2 (11.24), P04 V3 (11.26). Three of the four are self-flagged. ⚠️ **Any refinement pass
  that edits these four must re-measure**; a single word change drops them under.

⭐ **Nobody aggregated these.** Each pair reported its own; no artifact in the run says *"8 of 24 songs sit over
the prose ceiling."* That aggregation is what a coordinator owes and it is here now.

---

# 6. ⭐ THE RESIDUE HUNT (L41) — where did the monoculture go this time?

**My prediction before measuring:** the gated levels would be spotless and the monoculture would appear at the
lowest ungated level. Cross-pair lyric/prompt similarity is gated at 0.42/0.58; it measured **0.074/0.055**.
So it went somewhere else. Three findings, in ascending order of importance.

### RESIDUE-1 — assigned EMO / SFX / section vocabulary: **light, and mostly unavoidable**

Content words appearing in ≥3 pairs:

| level | shared words (pairs) |
|---|---|
| EMO vocabulary | `competence` (4) · `attention` (4) · `calm` (4) · `flat` (4) · `patient`, `quiet`, `contempt`, `refusal`, `steady`, `hushed`, `cold` (3 each) |
| section names | `verse` (4) · **`flash` (4)** · `intro` (3) · `refrain` (3) · `outro` (3) |
| SFX vocabulary | `flat` (4) · `close`, `tail`, `beat`, `palm`, `once` (3 each) |

`verse/intro/refrain/outro` are the vocabulary of song form and cost nothing. **`flash` in four pairs' section
names is the real hit** — it is the run's own composite noun leaking into the naming layer of pairs whose
flashes were supposed to be *figurative* (P01, P05, P06). Minor. `flat` appearing in EMO *and* SFX *and* lyric
across four pairs is the run's single most house-flavoured word. **Advisory only.**

### RESIDUE-2 — 33 content words appear in ALL SIX pairs

`again · alone · arrives · before · behind · clean · cold · coming · dry · every · first · flat · gone · half ·
hand · holds · last · left · light · nobody · nothing · part · somebody · song · still · thing · through ·
way · went`

Read honestly: **`nobody`, `nothing`, `light`, `hand`, `flat`, `cold`, `arrives`** are the seed's own semantic
field — a run about leftovers and unwitnessed work *should* share those. This is theme, not monoculture.
**No repair.**

### ⭐⭐ RESIDUE-3 — **IT WENT INSIDE THE PAIRS, INTO THE UNGATED WITHIN-PAIR AXIS. THIS IS THE FINDING.**

Cross-pair distinctiveness has three validators. **Within-pair variation distinctiveness has none.**

| pair | within-pair LYRIC similarity (mean / max) | within-pair PROMPT similarity (mean / max) |
|---|---|---|
| P01 | 0.112 / 0.175 | 0.013 / 0.036 |
| P02 | 0.112 / 0.141 | 0.064 / 0.094 |
| P03 | 0.255 / 0.404 | 0.126 / 0.209 |
| P04 | 0.186 / 0.312 | 0.018 / 0.044 |
| **P05** | 0.173 / 0.183 | ⛔ **0.382 / 0.497** |
| P06 | 0.220 / 0.356 | 0.168 / 0.279 |

⛔ **P05's four MUSIC PROMPTS are 0.497 similar to each other — 9× the run's cross-PAIR maximum of 0.055.**
Diffing V1 against V2 returns these shared verbatim runs:

> *"Modal spiritual jazz at eighty-six beats per minute, opening on"* · *"seconds of brushes already circling
> the snare"* · *"A close warm female voice sits centre and slightly forward,"* · *"carrying a room return a
> half-second longer than the space around her would give. Rhodes"* · *"left, tarnished and detuned in the top
> octave;"* · **a 30-word identical sentence** — *"more than heard, a superhumanly funky bass playing
> chamber-quiet and landing one note behind the beat every fourth bar so the groove limps warm."* ·
> *"breathy and subtone, phrasing a downright salacious slow jam across sacred modal changes. Two brief crests
> lift the horn and"* · *"Brushes stay on the snare head throughout,"* · *"and the final chorus thickens as the
> room grows wetter."*

**That is one prompt wearing four hats.** P05's lyrics are genuinely four different songs (0.173 mean, the
second-lowest in the run) — the four *renders* would be nearly identical. And P05 did not flag it, while P04
**did** flag its four shared lyric lines and explained them.

**Second instance, same class:** P06 ships an identical author stanza in all four variations — *"somewhere a
schedule, a signature, a date / paper deciding which floors lose the current / … / a call from a hand with no
standing"*. ⚖️ **In fairness to P06:** its pair-level close *alludes* to the recurrence (*"'current' … four
times across the pair, always in the schedule stanza"*), so it is not concealed — but it is never **declared
as cross-variation identical text**, the way P04 declared its four shared lyric lines. The difference between
alluded-to and declared is the whole difference between a disclosure and a coincidence.

> ⭐⭐ **THE PREDICTION HELD, AND THIS IS THE THIRD CONSECUTIVE RUN IN WHICH RESIDUE APPEARED EXACTLY ONE LEVEL
> BELOW THE PREVIOUS FIX.** 2026-08 fixed lyric monoculture → it moved to form. The panel fixed form (the
> per-pair composite rule) → cross-pair similarity collapsed to 0.074. **It has now moved inside the pair, to
> the one axis with no validator: sibling-variation prompt distinctiveness.** And it landed in the pair whose
> literal composite element is *continuity*, which is either irony or causation.
> **→ ROUTED AS R-3 (artifact) and named as a doctrine gap (§10).**

---

# 7. ⭐ THE DEDICATION AUDIT — all 24, against THE CREDIT FOLLOWS THE FIELD

**Length: 24/24 PASS.** Re-measured with the code-fence stripped: **379–497 chars**, ceiling 500. Widest
margin P03 V2 (379); tightest P05 V2 (497). None over.
⚠️ *Measurement note for anyone re-running this:* measured with the ``` fences **included**, four of P05's
dedications read 503–509 and appear to breach. They do not — the fence is markup, not the field. **Strip the
fence before you measure, or you will file a false failure on the one pair with the least headroom.**

## The field rule, applied package by package

| pair | technique in the STYLE field | credit | ruling |
|---|---|---|---|
| **P01** ×4 | `furiously slapping and popping his bass` (raised) + `descarga` (raised) | **Larry Graham** + **Israel "Cachao" López** | ✅ **CORRECT — raised, so owed** |
| **P02** ×4 | sustained low sine · damped/prepared harp · beating tones · room-as-performer · breath-as-material | Radigue · Parkins · Lucier · Cage · La Barbara | ✅ each technique is raised; ⚠️ **one factual overreach, below** |
| **P03** ×4 | `balafon-derived` interlocking figure (raised, all four) | **Kélétigui Diabaté** (1931–2012), Mali | ✅ **CORRECT**, and correctly *personal* rather than regional — the repaired wording says *"a player who carried it"*, not *invented* |
| **P04** ×4 | nothing of sound-figure raised; **V4 EXCLUDE refuses `slap bass performance`, `thumb and pop bass`** | **declined, with the reason stated** | ✅ **CORRECT — banned in EXCLUDE owes nobody** |
| **P05** ×4 | no break, lick or figure named | **declined, with the reason stated**, and Gregory C. Coleman named as the false credit *not* made | ✅ correct; ⚠️ **one boundary question, below** |
| **P06** ×4 | `furiously slapping and popping his bass` (raised) | **Larry Graham** ("thumb and pluck, Sly & the Family Stone") | ✅ **CORRECT — the positive half of the rule** |

**No false credit found. No missing credit for a raised figure. No city standing in for a person. No
"contains a sample of" grammar** — all 24 carry the correct *"Suno generates; it does not sample"* formulation,
and P01's *"We asked for those figures by name, so we name them"* is the exact clause the standard prescribes.

## ⚠️ D-1 — A CLAIM ABOUT A REAL PERSON THAT IS NOT A FACT ABOUT THEIR WORK *(P02 V2)*

> *"For the flutter under everything — two tones tuned a hair apart until you can hear them beating: **that is
> Alvin Lucier's finding**, and we asked for it by name."*

**Acoustic beating is not Alvin Lucier's finding.** It is a physical phenomenon documented long before him
(Tartini, Helmholtz); Lucier *composed with* it. P02 V3's Lucier credit is accurate by contrast — *"Alvin
Lucier proved a room could be the performer"* is a fair statement about *I Am Sitting in a Room*.
**DEDICATION_STANDARD forbids a claim about a real person that is not a fact about their work.**
**→ R-3a, step 09, one-clause fix:** *"a flutter Alvin Lucier built pieces out of."*
⭐ Note the contrast in the same pair: P02 V3's Cage credit is worded with real care — *"John Cage built
preparation into **a fixed instrument**"*, not *"prepared the harp"*, which would have been false. The pair
knows how to do this; V2 is the one that slipped.

## ⛔ D-2 — "FULL CREDITS BELOW" POINTS AT NOTHING *(P01 ×4, P02 ×3 — 7 of 24)*

Seven dedications close with **"Full credits below."** There is **no companion credits file anywhere in
`output/daily/2026-08-15/`** and **no credits heading in any of the six step-10 files** (both checked
programmatically). `DEDICATION_STANDARD.md` places *"full credits below"* at **priority 5 — the first thing
trimmed** — and it exists solely as the overflow pointer to a companion file. A pointer with no target is a
promise in a shipped field.
**→ R-3b, step 09:** either write the companion credits file, or delete the clause. All seven have 66–98 chars
of headroom, so deleting costs nothing and writing costs nothing either.

## ⚠️ D-3 — AUTHORSHIP LINE IS INCONSISTENT ACROSS THE 24

Three formulations in one deliverable set:
- **"Lofn (Claude) with The Scientist"** — P01 ×4, P02 ×4, P04 ×4 (12)
- **"Lofn + The Scientist"** — P05 ×4 (4)
- **"Lofn with Dr. Local Symmetry"** — P03 ×4, P06 ×4 (8)

Not a doctrine breach — but eight of twenty-four dedications use the collaborator's handle where sixteen use
the role. In a field that ships, one form should win. **→ R-3c, step 09, cosmetic.**

## ⭐ D-4 — THE BOUNDARY QUESTION, AND IT IS THE ONE I WANT A HUMAN ON

P05 declines a drummer credit on all four, reasoning: *"No break is named, so no drummer is credited: a credit
whose trigger did not ship is false."* Under the field rule as written, that is correct.

**But P05's style field raises, as its FIRST SOUND in all four prompts, a named percussion technique held for
the entire song:** *"opening on five seconds of brushes already circling the snare head before anything else
arrives"* … *"Brushes stay on the snare head throughout."* P05's literal composite element **is** that bed.

Is a sustained brush figure *a break*? I do not think so, and P05's reasoning is honest and explicit — it even
names Gregory C. Coleman as the false credit it refuses to make, which is precisely the diligence the standard
asks for. **But COORDINATOR_FINDINGS F5 warns that the under-credit is the dangerous one because it reads as a
completed check**, and P05 is the only pair in the run whose whole bed is a drum technique.

⛔ **I am not routing this as a repair. I am putting it in front of The Scientist**, because it is the case
the run's new rule does not resolve, and only a human promotes a lesson to a rule. The narrow question:
**does THE CREDIT FOLLOWS THE FIELD reach a sustained instrumental technique, or only a named break/figure?**
P02 answered *yes it reaches techniques* (five composer credits for raised techniques). P05 answered *no*.
**Both are in this run, and they disagree.**

---

# 8. ⛔ THE EXCLUSION LIST — swept, all 24 packages, all fields

| banned item | hits | resolution |
|---|---|---|
| Indonesia earthquake / rescue | **0** | clean |
| Israeli strikes on southern Lebanon | **0** | clean |
| Bondi Beach | **0** | clean |
| Morocco / Ceuta / migrants / refugees | **0** | clean |
| AI · algorithm · model · neural · machine learning | **0** | clean |
| platform · streaming · playlist · recommendation · badge · follower · subscriber · engagement · monetise · royalty | **0** | clean |
| self-congratulation / meta about the pipeline, panel or run | **0** in every generator-facing field | clean |

**Four raw sweep hits, all resolved by inspection as false positives:**
- `Israel` ×4 → **Israel "Cachao" López**, a person's given name, in P01's dedications. Not the state.
- `strike` ×2 → *"the kick alone for three strikes"* and *"hard mallet strikes"* (P03). Percussion.
- `Suno` ×24 / `Bandcamp` ×16 → in **dedications only**, as source attribution and the no-sampling clause,
  which the standard *requires*. Not subject matter.

⭐ **The badge — the run's own provocation — appears nowhere in any song.** It is carried as *a situation a
body is standing in* (a corridor nobody routes through) and never named. **Ban 1 held.**

---

# 9. THE 16-POINT GATE (`skills/qa/references/suno_15_point_qa.md`)

### A. Singer Surface (7)
| # | gate | result |
|---|---|---|
| 1 | Human singer — specific person/body/situation | ✅ **24/24.** A woman on a low stage with a hand on a dead fader · someone ankle-deep in a cold inlet · two people and one folding chair · a busker in a tiled corridor · a sculptor eleven feet from the tide · a person with a crank in the boot. Every one is a body in a place at risk. |
| 2 | Body-first opening (first four sung lines) | ✅ **24/24.** e.g. *"Water, you are wide and cold, / and I am here and I am late"* · *"Cable in. / Cable in. / Amp's little light. / Nothing broken."* · *"crank in the boot, under the jack / steel handle, colder than the wheel arch"* |
| 3 | Adoptable hook | ✅ 24/24 — all singable, all concrete nouns or first-person acts. No banned title noun anywhere; no persona prefix. |
| 4 | Hook recurrence / mutation | ✅ 24/24, measured: `line_return` 0.211–0.536, floor 0.20. P04's hook returns exactly 3× with 3 stated meanings; P06's refrain is byte-identical 4×; P05's is byte-identical 4×. |
| 5 | Chorus clarity — adoptable, not thesis | ✅ 23/24. ⚠️ **P03 is the one to watch:** its returning line *"And none of it was arranged for them"* is superb but is closer to a *statement* than a chorus. It is the assigned catalogue form, so this is a note, not a fail. |
| 6 | Voice + pulse survival | ✅ — and P01 is *built* on it: the hook must survive being clapped and shouted with no instruments, because inside the song it has to. |
| 7 | 15–30 s clip survival | ✅ — P04's opening bar (two stabs, sub, *"Cable in. / Nothing broken."*) and P02's *"Water, you are wide and cold"* both carry scene + ache in under fifteen seconds. |

### B. Cathedral Engine (5)
| # | gate | result |
|---|---|---|
| 8 | Golden Seed pressure visible in hook/image/production | ✅ **strongest in P06** (the worn grip *is* the leftover brighter than the source) and **P02 V4** (*"My breath outlasted it"*). P03 V4 states it almost too literally: *"And the leftovers are the brighter part."* |
| 9 | Mythic image ladder returning to the body | ✅ — P05 V1's `[The Sheet]` is the run's cleanest instance: ordinary (a spent wave) → specific (a sheet a fingernail thick) → strange (the face arriving before the sea) → back to the body (*"and I kept working"*). |
| 10 | EMO dramaturgy depth | ✅ 24/24 full syntax, zero bare mode tags. ⚠️ **P06 at 11 unique tags / 66 uses is the run's thinnest** (Skeptic 2's NO). P02 at 35/50 is the richest. |
| 11 | Production dramaturgy — every unusual sound has a job | ✅ — the deviation clauses are what prove this: each pair names which disobedience it would *keep*, which forces every device to have had a job. |
| 12 | Panel pressure forced artifact changes | ✅ **demonstrable.** Hyper-Skeptic A killed the uniform composite rule → six literal elements. Hyper-Skeptic B rebuilt C06 (*she is not good that night*) → P04's entire concept. Hyper-Skeptic C produced the Deviation Clause → 6/6 shipped one. |

### C. Suno Package (3)
| # | gate | result |
|---|---|---|
| 13 | Clean Suno lyrics · `[Theme:]` then `[SONG FORM:]` · full EMO syntax · no debris | ✅ **24/24** (GATE_REPORT `has_theme_header`, `has_song_form`, `no_bare_mode_emo_tag`, `has_sfx_cue`) |
| 14 | Producer-grade prompt, dense prose, 850–1000 chars | ✅ **24/24.** Re-measured (code fences stripped): **859–958 chars** — all inside the band, **none ≥985**, so zero boundary-hugging; all terminal-punctuated (24/24); zero banned openers; zero negations. ⭐ **Zero real artist names in any style field**, verified two ways: a name-list sweep (its only hits are the substring `sly` inside the mandated phrase *"furiously slapping"*) and a capitalised-proper-noun probe over all 24 prompts, whose complete output is `Afro`, `Cuban`, `Rhodes` — two genre words and one instrument. Every named human in this run lives in a dedication, which is where the standard puts them. Note: P02 V3's 842-char near-miss recorded in COORDINATOR_FINDINGS was repaired; the floor is now clear by 9 chars at its tightest (P02 V4, 859). |
| 15 | Full package completeness | ✅ **format-appropriate.** Daily-tier packages carry title, angle, music prompt, exclude prompt, Suno lyrics, dedication, measured self-check and describe-render check. The competition-tier sidecars (ghost bank, image ladder, controlled fracture) are out of scope for a daily and their absence is not a fail. |
| 16 | Lineage & Credit for living-scene genres | ⚠️ **PASS with the D-2 defect.** P01 (descarga/Minneapolis funk), P03 (balafon), P06 (slap-and-pop) all carry named-person lineage + a no-claim statement. P02/P04/P05 are internal-palette and state their N/A reasoning explicitly. **Defect:** seven dedications promise "full credits below" and there is no such block. Not blocking; routed as R-3b. |

**16-point result: 15 clean PASS, 1 PASS-with-defect. Zero blocking fails.**

---

# 10. ROUTED REPAIR BRIEF

⛔ **I have fixed nothing. Everything below routes to the agent that wrote the file, with its context intact.**

| id | finding | route | pairs | why this step |
|---|---|---|---|---|
| **R-1** | ⭐ **The authored decision is absent from P01 V1, V3 and V4.** Ruling 3 is P01's declared anger target (*"THE ANGER'S TARGET: the decision, offstage, unnamed"*) and three of four songs never contain it; V2 shows exactly what it looks like. | **STEP 07** (structural — it needs a section, not a word) | P01 | It is a missing *stanza*, and the song-guide level is where a section is added. One stanza per variation, in the pair's own AABB antiphon grammar, never naming a person. |
| **R-2** | ⚠️ **One-fact rule spent twice** in P01 V1, V2, V4. FLAG-level (`max_sung_numeric_facts: 1`), and the gate has no machine reader. | **STEP 09** (surface — one line each) | P01 | Demote or de-quantify the weaker of the two in each: *"Forty minutes in"*, *"Ten years learning"*, *"twenty centimetres"* are the candidates; the room-count and the arm-arithmetic are the load-bearing ones. |
| **R-3a** | ⚠️ **P02 V2's dedication attributes acoustic beating to Alvin Lucier as his "finding."** Not a fact about his work. | **STEP 09** | P02 V2 | One clause. Dedication is at 448 chars; no length pressure. |
| **R-3b** | ⛔ **"Full credits below" points at nothing** in 7 of 24 dedications. | **STEP 09** | P01 ×4, P02 V1/V2/V3 | Either write the companion credits file or delete the clause. Priority 5 is the first thing trimmed by the standard. |
| **R-3c** | ⚠️ Authorship line inconsistent across the 24 (three formulations). | **STEP 09**, cosmetic | P03 ×4, P05 ×4, P06 ×4 | Pick one form. |
| **R-3d** | ⚠️ **P04 names `corroded saw pad` in all four style fields** against ruling 7 (*the bed is a bass and a room, never a pad*), and V2/V3 also carry `pad` in EXCLUDE. | **STEP 09** | P04 | Either rename the source (*"corroded saw drone"*, *"a thin saw layer"*) or state in the pair flags why this pad is the exception. Do not leave the token armed on both sides. |
| **R-4** | ⭐⭐ **Within-pair prompt monoculture: P05's four MUSIC PROMPTS measure 0.497 similar** (9× the cross-pair max). Secondary: P06's undeclared identical four-line author stanza in all four variations. | **STEP 09** for P05 (rewrite two of four prompts from a different first clause outward); **DECLARE ONLY** for P06 (exact return is legitimate craft; it just was not stated) | P05, P06 | The four *lyrics* are genuinely different; only the *renders* would collide. Two rewritten prompts fixes it. |

**Nothing routes to a rerun. Nothing is quarantined. No pair is cut.**

## ⭐ THE DOCTRINE GAP THIS AUDIT FOUND (not a repair — for The Scientist)

**Cross-pair distinctiveness has three validators** (`validate_step06_distinctiveness.py`,
`validate_step09_distinctiveness.py`, `validate_portfolio_distinctiveness.py`) and four `gates.yaml` numbers.
**Within-pair sibling-variation distinctiveness has zero validators and zero numbers.** That is where the
monoculture went this run, and L41 says it will go there again. A single number —
`within_pair_max_prompt_similarity` — with a reader would have caught P05 before step 10.
⛔ *A gate with no reader is not a gate.* The one-fact rule (§5) is the second instance in this same run.

---

# 11. SELECTION — best 3 per arm, ranked **within arm**, never by global score

## 🎵 ACCESSIBLE (P01 · P02 · P03 — 12 candidates)

| # | pick | one-line reason |
|---|---|---|
| **1** | **P02 V4 — *My Breath Kept Getting in the Way*** | The seed executed as a physical joke with a body in it: her steam outlasts the burning grain — *"My breath outlasted it"* — and the song is chosen, complete, warm, and the only funny thing in twenty-four songs. Ends *"I like being hot."* |
| **2** | **P01 V2 — *I Left My Hand on a Dead Fader*** | The **only** song in this arm that carries the authored decision (*"Somebody sat in a room somewhere / and made a decision about my air"*) and then refuses to thank it (*"Nobody up here is going to stand / and thank a decision for my hand —"*). The arm's INDIGNATION seat, and the one P01 all three skeptics endorse. |
| **3** | **P03 V2 — *Two Skies in Two Days*** | *"The rare one had a name everybody knew. / The yearly one had nobody at all."* — the run's governing distinction stated as an **image** rather than an argument, which is the thing Hyper-Skeptic A said the run could not do. Its once/every-year catalogue is a real form, not a list. |

**One per pair, and I did not have to force it.**

⭐ **Cost declared:** on pure line quality **P02 V3 (*I Put My Hand Flat on the Water*) outranks P03 V2** — it
carries the run's most literal chiasmus and its pair's highest end-rhyme (0.525). I did not take it because it
is P02 V4's device done a second time, and taking it would drop the affront ending — the donated engine of cut
concept C04 — out of the arm entirely. **The arm would lose a register to gain a better song.**

## 🎵 AMBITIOUS (P04 · P05 · P06 — 12 candidates)

| # | pick | one-line reason |
|---|---|---|
| **1** | **P04 V4 — *I Coil the Cable Wrong*** | The run's one non-resolving song at its hardest: a coda **added after** the last chorus so density rises where a song should land, competence named then denied (*"That is not skill. / That is a rut."*), and the final word of the package is **worse**. This is ruling 4 executed better than it was specified. |
| **2** | **P05 V2 — *Eleven Feet and Closing*** | Chosen obscurity with not one gram of pity, and it carries the best written line in all twenty-four songs: *"I am not going to tell you it will be all right, / because you are sand and you cannot be lied to. / It will not be all right. It will be over, / which is a different thing, and simpler to stand."* |
| **3** | **P06 V4 — *I Ran the Cord Down the Hall*** | The only song that names what an **unprovable absence** costs — *"the schedule counts the hours it takes away / the hours it never took stay uncounted / a prevented thing leaves the page the same"* — and the only ending in the set that is a flat negative report: *"nothing happened. again."* |

**One per pair.**

## ⛔ THE STRONGEST SONG I DID NOT PICK

**P04 V3 — *Coat by the Wall Map*.** It contains the single best passage in the run — the refusal to weigh the
one listener (*"One person came. / Take it or don't. / It's enough. / It's not enough. / Let both stand."*) and
the closing image of *"Sound went with her." ×2 / "Out through the bend." ×2*, in which the only good thing in
the room leaves and is not replaced.

**Why it lost:** it duplicates V4's structure and mandate, and taking both would put two of three AMBITIOUS
slots inside P04 — costing the arm either the chosen-obscurity register (P05, which exists nowhere else in
this arm) or the unprovable-absence finding (P06, likewise). It is also the run's tightest floor margin:
**alliteration 11.26 against a floor of 11.0**, and a single refinement edit drops it under. If The Scientist
prefers it to V4, the trade is a real one and I would not argue hard — but **re-measure it before touching it.**

---

# 12. THE PUBLISH BAR — stated plainly, and not met

⛔ **Dailies are PRACTICE. I recommend no publication and nothing here is a publication recommendation.**

**What would have to be true, and none of it is yet:**
1. **A render must exist.** Every gate in this report reads text. The run's Source-3 obligation is *a composite
   with a visible join*, and P01 stated the ceiling of this whole exercise plainly: *"whether the butt-cut
   survives cannot be gated textually at all."* **Six pairs specified a seam; zero can be verified to have one.**
2. **`lofn-render-audit` must run under THE BLIND RULE** — the audio alone, first, never the prompt. The one
   thing that most needs hearing is not the seam: it is **whether P04's flatness renders as *she is not good
   tonight* or as *moody*.** P04 named this itself as its own generic failure mode. If it renders moody, the
   panel's rebuild is lost and the pair's whole reason for existing goes with it.
3. **The deviation clauses must be adjudicated against the render**, which is what makes this run's
   pre-registration falsifiable rather than decorative. P05's and P06's are specific enough to settle;
   P02's and P04's are not.
4. **The Scientist's ear.** Borderline defaults to HOLD. An empty publish day is acceptable; a lowered bar is not.

`vault/AUTONOMY.md` holds. **Nothing here is rendered, published, spent or sent. Drafts on disk.** 💜

---

# 13. THE ONE THING I MOST WANT A HUMAN TO LOOK AT

Not a defect. **A disagreement inside the run's newest rule, which two pairs resolved in opposite directions
and neither could see the other.**

**THE CREDIT FOLLOWS THE FIELD** was derived this run from four independent rulings about a *named break*.
It does not say whether it reaches a **sustained instrumental technique**:

- **P02 said YES** and credited five composers for raised techniques (a sustained sine, a damped harp, a
  prepared harp, room-as-performer, breath-as-material) — none of which is a break.
- **P05 said NO** and declined all credit, while its style field raises *"brushes already circling the snare
  head"* as the **first sound and the literal composite element of the entire pair**, held for the whole song.

Both are reasoned. Both are in writing. **Both cannot be right**, and the run shipped them side by side.
COORDINATOR_FINDINGS F5 already established that the **under**-credit is the dangerous one *because it reads as
a completed check* — which is exactly the shape P05's four dedications have.

⛔ **Only a human promotes a lesson to a rule.** The narrow question is:
**does a technique raised in the style field owe its player when the technique is an instrumental practice
rather than a named break?** Answer it once and both pairs become correct; leave it and the next run will
re-derive it, in one direction, at random.

---

*Audited by `lofn-qa`. 414 prior checks cited; 11 measurement passes run fresh against the artifacts.
Every number above is machine-measured. Nothing was estimated and nothing was repaired by the auditor.*
