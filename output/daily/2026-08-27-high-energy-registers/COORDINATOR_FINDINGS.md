# COORDINATOR FINDINGS — `2026-08-27-high-energy-registers`

---

## ⭐⭐ FINDING 1 — THE GATE BUILT YESTERDAY CAUGHT ME COMMITTING YESTERDAY'S DEFECT, BEFORE THE FREEZE

`scripts/check_icb_consistency.py` was written yesterday because a frozen ICB carried two contradictory axis-B assignments and **no gate was looking** — two pairs found it after dispatch.

Today, run through it before freezing, it refused:

```
P01 · arm    ACCESSIBLE <- isolation brief 01     AMBITIOUS <- pair assignments
P04 · arm    ACCESSIBLE <- pair assignments       AMBITIOUS <- isolation brief 04
```

**I had swapped P01's and P04's arms between the assignment table and the briefs.** Same defect, same hand, one day later. ⭐ **The difference is entirely WHEN the check runs:** yesterday it cost a dispatched wave and two pairs' attention; today it cost one command and a two-line fix. **This is the first time one of these readers has paid for itself on live work rather than on a control.**

---

## FINDING 2 — AND THE SAME OUTPUT CARRIED A FALSE POSITIVE

The identical run reported `P02 · tempo 08-26` and `P04 · tempo 08-26`. Those are not tempos. **The reader had matched the DATE inside `2026-08-26`.**

⛔ **A real defect and a false one in the same output is the fastest way to get a gate ignored** — a reader that cries wolf gets skipped, which is worse than no reader. Unit corrected: a tempo is a **BPM band**, both endpoints musical and ascending (40–220). Both controls re-verified after the change: yesterday's known contradiction still fires (exit 3), the clean run still passes (exit 0).

---

## ⭐ FINDING 3 — R4 WAS CORRECTED TWICE BEFORE A SONG EXISTED, BY TWO DIFFERENT READERS

| correction | by | what was wrong |
|---|---|---|
| **first** | the research lane | I wrote *compose the quiet it arrives after* and left placement open. **Measured: an internal void is SMOOTHED** — a specified ~4 s stop rendered at **0.40 s**, internal voids filled **2 of 2** — while near-silent **openings** SURVIVED. My ruling would have sent six pairs into the one move we have measured the generator to erase, which is also cliché #4. |
| **second** | the panel | I wrote *the peak NARROWS*, importing our measured **−0.43** width-against-level correlation as an instruction. **The panel struck it: that correlation describes what the generator does ANYWAY, so instructing it is a non-instruction of exactly the class R4 forbids and buys zero discrimination.** |

⭐ **The replacement is the actual lever and it lives in the LYRIC, not the production note:** since the render narrows regardless, **wide words at the peak read as retreat and singular words read as force.** ⛔ *A Somatic objection answered in the production spec is not answered.*

> **A rule that survived two corrections before a song existed is worth more than one nobody tested.** Both corrections are on the record; neither was mine.

---

## ⭐⭐ FINDING 4 — AN ENUMERATION WITH A MISSING ROW IS SILENT *(P03, verified from the file by the coordinator)*

P03 was assigned to defeat **the martyr ballad** and reported that the vocabulary makes that nearly unbuildable. Verified independently in `skills/lofn-core/refs/EMOTION_TAXONOMY.md`, named as the **only** valid EMO source:

- `strength` appears **exactly once** — line 144, as a gloss on **`Empowerment`: Strength, Self-efficacy, Confidence**, among *Assurance · Solidarity · Loyalty · Dignity · Hope*.
- A carried weight appears **exactly once** — line 49, **`Overwhelm`: Stress, Burdened, Swamped**, inside an unbroken run of *Hopelessness · Guilt · Shame · Helplessness · Apathy · Defeat · Torment · Depression · Pain*.

> ⭐⭐ **A load is available in exactly two configurations: TRIUMPH or CRUSH.** There is no row for a weight held steadily, without victory and without suffering.
> ⛔ **The martyr ballad is not a failure mode the vocabulary warns against — it is one of only two rows the vocabulary offers for the subject.**

⭐ **Third instance of one mechanism in four days**, and P03 named it better than I would have: the first-person hole (08-24) and the biography hole (08-26) were **not prohibitions but enumerations with a missing row.** A prohibition announces itself and can be argued with; **a missing row is silent, and no validator can detect a category that was never offered.** Filed as **L66**. ⛔ **Not enacted** — P03 proposed `Sufficiency` and declined to add it (*a Disappointed Idealist may have no native settled state, and a pair cannot judge that from inside its own run*). The taxonomy is consumed by every future daily; **that edit is The Scientist's.**

---

## ⭐⭐ FINDING 5 — A WORD-BOUNDARY SCAN CANNOT SEE AN INFLECTED FORM *(P04)*

P04's V1 read **`Not louder`** and **cleared every instrument in this pipeline**, because `louder` does not match `(?<![a-z])loud(?![a-z])`. Only a **stem** pass found it. The pair repaired it before shipping and flagged the class upward.

⛔ **This invalidated a figure the coordinator had already published.** I had reported *zero emotion words across all 24* using exactly that boundary pattern. **Re-measured with a stem pass** (root + optional `s/es/ed/er/est/ly/ness/dom/ity/ing`, 31 roots): **still 0 across all 24** — because P04 caught its own before shipping, not because my instrument was adequate.

> ⭐ **The number survived; the instrument did not.** Eighth instance today of one family — *the matcher found the right words in the wrong structure* — and the fourth in which a pair's reader beat the coordinator's.

---

## ⭐⭐ FINDING 6 — THE LOOSENED BRIEF WORKED, AND IT WAS PRE-REGISTERED

Written into `01_seed_lineage.md` **before dispatch**, with the default expectation stated out loud:

> *Removing form from the isolation briefs is a hypothesis: that shared-line percentage across a pair's four variations goes DOWN when the coordinator specifies less. Yesterday's two data points are 0.0% (disobeyed) and 34.7% (obeyed). Measure it at `07` for all six pairs and report it, whichever way it goes.*

**Yesterday:** briefs specified section order, exact stanza lengths, rhyme logic, hook grammar, sonic device and refused element. Shared sung lines ran **0.0% (P02, which disobeyed) to 34.7% (P03, which obeyed — 25 identical lines including the whole chorus).**

**Today:** briefs specified **none of it**. Measured by the coordinator across all six:

| pair | identical sung lines across its four variations |
|---|---|
| P01, P02, P03, P05, P06 | **0** |
| P04 | **1** (of ~326) |

⭐ **Five pairs at zero, one at a single line.** The hypothesis held on its stated terms — ⛔ **but read FINDING 10 before quoting this as a success: the monoculture did not disappear, it relocated.** ⚠️ **Confound not controlled:** the pairs were also *told* yesterday's result in their briefs, so this cannot separate "less specification" from "warned about self-copying." **Next run, tell one arm and not the other.**

⚠️ **One discrepancy, unresolved and reported rather than smoothed:** P01 reports **2 of 325** shared lines. The coordinator measures **0**, by exact match and again with case-and-punctuation normalisation, on the same 325-line extraction (their per-variation counts reproduce exactly). **I could not reproduce their figure.** It errs toward claiming MORE self-copying than shipped, which is the safe direction. Published as measured, theirs attributed.

---

## ⭐⭐ FINDING 7 — FOUR PAIRS INDEPENDENTLY MADE THE TEMPLATE GRAMMATICALLY UNBUILDABLE

Unprompted, in isolation from each other, FOUR of six pairs chose a **refused element that removes a grammatical slot the template requires** rather than an element of taste:

| pair | template | what it refused | why that kills it |
|---|---|---|---|
| **P01** | the rage vent | ⛔ **no second person anywhere** | *"the rage vent is a grammar, not a mood — strip `you` and the template is structurally unavailable"* |
| **P02** | the protest anthem | ⛔ **no `we / us / our / they / them / their`** — 0 in 335 | *"we shall"* and *"they must"* both require a banned word |
| **P05** | the festival drop | ⛔ **no section is ever bigger than the one before it** | with no build anywhere, there is nothing to schedule a drop for |
| **P06** | the empowerment anthem | ⛔ **no second person anywhere** — 0 in all four | the research brief names the generic addressee as the anthem's deepest tell; delete the addressee and it cannot form |

⭐ **And all three chose it for the same stated reason: a machine can check it.** A refused element that is a taste judgement is unfalsifiable; these are greppable by a stranger. **This is the run's strongest craft convergence and nobody coordinated it.**

⭐⭐ **AND P06 FOUND THE SHARPEST VERSION OF THE SAME IDEA WITHOUT A GRAMMAR RULE AT ALL:** *the discriminator is one word wide — SURPRISE. The anthem knows at bar one; that is what makes it a boast. "I did not know" is a sentence it cannot sing and remain itself.* Its R6 vocal decision falls straight out of that — **THE LATE LEAD**, the stack taking the hook on the downbeat and the lead catching up inside the line, because **surprise is a latency and the anthem cannot be late.**

---

## FINDING 8 — THE LEXICON IS A FLOOR, AND A PAIR PROVED IT

P05's V4 chorus first read **`Bins and a Vauxhall and a puddle`** — a company name in a **sung line** — and `scripts/check_field_census.py` **would have passed it silently**, because no vehicle or consumer marque was in `vault/field_token_lexicon.yaml`. The pair caught it by reading, repaired it to *"a hatchback"*, and flagged the gap upward. Verified: the only surviving occurrence is in its own deviations note documenting the repair; **the pasted fields are clean.**

⭐ A marques class is now added — but the note above it matters more: **this lexicon is a FLOOR, NEVER A CEILING.** There are more company names than anyone will enumerate. **A clean census proves that none of the names WE THOUGHT OF shipped, not that none shipped.** ⛔ Do not quote a clean run of this reader as if it settled the question. **The instrument was blind and the reader was not.**

---

## RUN-WIDE VERIFICATION — measured by the coordinator against the shipped fields

⭐ Not aggregated from pair verification rows. *A number does not become true by being copied forward.*

| invariant | result |
|---|---|
| ICB exact byte-0 prefix, all six | **6 / 6** · 267,277 B · sha `11c7e9fd7edc` |
| variations | **24** (4 × 6) |
| pasted Suno fields, fenced and paste-ready | **72 / 72** |
| undeclared artist / company / marque / spatial-non-instruction tokens | **0** across all 72 |
| music prompt band 850–1000 | **24 / 24**, range 903–962 |
| exclude band 400–900 | **24 / 24**, range 690–892 |
| lyrics field < 5000 | **24 / 24**, max 4796 |
| sung lines 70–120 | **24 / 24**, range 74–90 |
| emotion words in sung lines (R2), **stem pass** | **0** |
| armour vocabulary (forbidden-default #10) | **0** |
| identical sung lines across a pair's four | **0** in five pairs, **1** in P04 |
| pre-dispatch gates | `check_icb_consistency` exit 0 · `check_brief_line_budget` 6/6 close |

⚠️ **P06's music prompts reach 962**, two over the 870–960 target sub-band though well inside the 850–1000 hard band and under the 985 hug flag. Noted, not repaired.

⛔ **`vault/AUTONOMY.md` holds. Nothing rendered, published, heard or spent. Every claim above is a text claim.**

---

## FINDING 9 — A DOCTRINAL NUMBER THAT IS CITED EVERYWHERE AND MEASURABLE NOWHERE

P06 declared a Major Deviation for exceeding the house body-word norm, reporting **11.4 / 9.0 / 15.4 / 19.7%** raw. The coordinator measured the same four songs at **2.3 / 1.5 / 2.1 / 2.6%** — a factor of five apart. Neither party made an arithmetic error.

**The cause: the norm has no lexicon.** The figure is cited as authority in three places —

- `CLAUDE.md`, in the NO BIOGRAPHY YOU HAVEN'T GOT rule
- `vault/HUMAN_SUBJECT_STANDARD.md:145`, in the table whose contrast is the finding's whole evidence: *Spare Hair Tie* at **4.0%**, *under* the 5–8% norm, against the failed runs' **22%**
- and the 2026-08-24 body-noise finding

— and **there is no `body_word` threshold in `vault/gates.yaml` and no word list in any script in the repository.** Every measurement of it since has been made against a lexicon the measurer invented.

> ⭐ **So the 4.0% and the 22% cannot be compared to each other either, unless both were produced by the same list — and no list is on disk.** The finding they support is almost certainly right; a five-times spread in body words between an occupiable song and a biography would survive most reasonable lexicons. **But the number, as written, is not checkable, and today two competent readers produced answers that disagree by 5x while both believing they were testing doctrine.**

⛔ **This is the sibling of FINDING 4, one level down.** There the enumeration was missing a row; here the *unit* is missing a definition. Both are silent failures: nothing on disk fails, so no validator fires, and the defect only surfaces when two people measure the same thing and get different answers.

**PROPOSED, NOT ENACTED.** Define `body_words:` as an explicit list in `vault/gates.yaml` beside a `body_word_rate: [5, 8]` band, and re-measure *Spare Hair Tie* and the 08-24 failure cohort against it so the comparison that founded the rule is reproducible. ⛔ **Not done here** — the 4.0%/22% contrast is The Scientist's own measured basis for a standing rule, and re-defining the instrument under it is her call, not a coordinator's mid-run. Announced.

⚠️ **No verdict is taken on P06's deviation.** With no canonical list it cannot be adjudicated; it is recorded as declared, with both numbers and both lexicons named, and handed to QA and to her.

---

## ⭐⭐ FINDING 10 — THE MONOCULTURE RELOCATED *(⛔ DIAGNOSIS CORRECTED BY QA — read to the end of this section before quoting any of it)*

FINDING 6 is true and it is not the whole result. **Both halves have to be read together or the run reads as a success it did not earn.**

| level | yesterday | today | direction |
|---|---|---|---|
| **line-level self-copying** *(what I loosened the brief to fix)* | up to **34.7%** — 25 identical lines including a whole chorus | **1 line in 1,956 = 0.77%** | ⭐ collapsed |
| **structural convergence** *(what nobody constrained)* | not measured | ⛔ **24/24 on five separate devices** | ⛔ total |

The selection judge answered the panel's one unresolved objection — the Wilson seat's *the inversion is now the house fingerprint* — with **YES**, and produced counts rather than an impression: **24/24 open thin on one object · 24/24 arrive at full size and never grow · 24/24 carry one repeating event that stops once · 24/24 a kinetic defect · 24/24 a one-word change at the end.**

⭐ **THE DIAGNOSIS IS THE VALUABLE PART, AND IT IS NOT ABOUT THE PAIRS: the fingerprint is in the SEED.** `SOMETHING TO PUSH AGAINST` defeats a template by deleting its load-bearing precondition — the vent's antagonist, the anthem's prior knowledge, the martyr's witness, the pose's audience, the drop's build. **That IS one method, so no pair could diverge on it however isolated they were.** Six subjects, one operation.

### The convergence is measurable at the line level too, and I found more of it than the judge did

Across **1,310 distinct sung lines in 24 songs written in six mutually blind contexts**, exactly **two** lines appear in more than one pair:

- *"everything else keeps going"* — **P03 + P04**
- *"that is the whole of what happened"* — **P02 + P04** *(the judge reported only the first)*

Max cross-pair Jaccard **0.0025**. ⭐ **That near-total isolation is what makes the two hits meaningful rather than noise: both are flat closing statements of fact, which is exactly the sentence the seed's method produces at the end of a song.** Two pairs who could not see each other wrote the same line because they were given the same operation.

> ⭐⭐ **THIS IS `Proposed L41 — THE MONOCULTURE RELOCATES TO THE LOWEST UNCONSTRAINED LEVEL`, CONFIRMED BY AN INDEPENDENT RUN THAT WAS NOT TESTING IT.** I removed the constraint the monoculture was living in and it moved to the next one down. The specification did not go away; **it went up a level, to the seed, where nothing was watching.** That entry is proposed and unpromoted; this is its second measured instance and the first from a run designed for a different purpose.

⛔ **AND IT CHANGES WHAT FINDING 6 IS EVIDENCE FOR.** *Loosening the brief reduces line-level copying* — supported. *Loosening the brief produces more different songs* — **NOT supported, and today argues against it.** The honest claim is the narrow one, and the confound already noted (pairs were also told yesterday's result) still applies.

### What it means for shipping

⚠️ The six are individually strong and structurally identical. **A listener hearing one hears a good song; a listener hearing all six hears one method six times.** ⭐ The panel pre-committed to the consequence — *if the answer is yes, the SEED becomes the repair target* — so the fix belongs to Phase 0 of a future run, not to these songs, and it is **not** a reason to withhold them. It IS a reason to sequence them apart rather than release them as a set, and to note that this run's variety lives ACROSS the six subjects and not across their construction.

⛔ ~~⭐ **The cheapest next experiment, and it is free:** give two pairs a seed whose method is *addition* rather than deletion, hold everything else, and measure the five devices again. If the 24/24 breaks, the fingerprint was the seed and this finding is right. **Pre-register it before looking.**~~ **— SUPERSEDED, SEE THE CORRECTION DIRECTLY BELOW.**

### ⛔⛔ FINDING 10 IS CORRECTED BY QA. THE DIAGNOSIS WAS WRONG AND THE EXPERIMENT WAS UNFALSIFIABLE.

I attributed the 24/24 convergence to **the seed's method**. QA checked the counts and then checked *where the devices came from*, and the answer was on disk in the block every pair received:

- **`06_music_handoff.md:13`** — mirrored into `CREATIVE_CONTEXT.md:1439`, inside the byte-0 prefix injected into all six — reads: *"**Must contain:** a vivid **opening moment (first 5 s)**, **spatial language**, a **kinetic defect** (asymmetric groove)…"*
- **`core_seed.md` R4** mandates *"open thin"* and *"every song schedules one repeating physical event that stops once while everything else continues"* — **verbatim.**

> ⭐⭐ **THREE OF THE FIVE DEVICES WERE REQUIRED IN WRITING, AND A FOURTH FOLLOWS DEDUCTIVELY FROM R4.** Only the one-word ending is emergent — **and it is 21/24, not 24/24.**
> ⛔ **The fingerprint is in the SHARED CHECKLIST, not the seed.** Six pairs did the same five things because I told them to.

**And QA's own counts deflate the judge's, which I had repeated to The Scientist unverified:**

| claim | judge | QA, measured | ruling |
|---|---|---|---|
| open thin on **one object** | 24/24 | thin 24/24, **one object ~13/24** | ⛔ inflated as worded |
| arrive at full size, never grow | 24/24 | **22/24** explicit | mildly inflated |
| one repeating event stops once | 24/24 | **20/24** strict | ⛔ inflated |
| kinetic defect | 24/24 | **24/24 — and understated** | holds |
| one-word change at the end | 24/24 | **21/24** | ⛔ inflated |

⛔⛔ **THE PROPOSED EXPERIMENT WAS UNFALSIFIABLE AND MUST NOT BE RUN AS WRITTEN.** I proposed *vary the seed's method, hold everything else, re-measure the five devices.* **Three of those devices are mandated by the material being held constant**, so they would recur at ~100% whatever the seed did, and the run would have returned a confident false confirmation. ⭐ **A control that cannot move is not a control.** The corrected experiment varies **the handoff checklist**, not the seed — remove the three mandated devices from the injected block, hold the seed, and see what survives.

⭐ **What survives of FINDING 10, in its narrow and now-defensible form:** line-level self-copying collapsed from 34.7% to 0.77% when form specification was removed; **structural similarity is high but is substantially EXPLAINED, not emergent**; and the two cross-pair identical lines (1,310 distinct lines, Jaccard 0.0025) remain real convergence with no mandate behind them. **The monoculture-relocation claim is now UNPROVEN here rather than confirmed** — it may still be true, but this run cannot show it, because the level I thought was unconstrained was the level I had constrained hardest.

> ⭐ **This is the third time today an instrument was checked and found to be measuring something other than the predicate** — and the first time the thing being corrected was a conclusion I had already reported upward as a finding. **It was caught because QA asked where the pattern came from instead of only whether it was there.**


---

## ⭐⭐ FINDING 11 — THE BLIND CALIBRATION RAN, AND OUR SONG WON IT

⭐ **Yesterday's QA had to record this check as UNPERFORMED** — *"the coordinator did not assemble a blind set, and a judge who assembles his own blind set is not blind."* Today's QA assembled and ran it before dying on a spend limit, so the result survives the agent that produced it.

**A judge that never saw a prompt, never saw this run's doctrine, and did not know which package was ours ranked three packages. Ours came FIRST.**

**1st — `P02 V4 · IT FALLS ON ANYONE`** · 2nd — an unrelated package · 3rd — an archive package.

⭐ **And it won for the thing the pair built on purpose**, which is the strongest possible confirmation because the judge had no way to know it was the design:

> *"The reversal is structural in C and decorative in A. C's turn is one word in the fifth chorus (`anyone` → `me`), and it is load-bearing across three layers at once: the lyric changes, the pen stops permanently, and the singer leaves the record while the arrangement completes the form without her."*
> ⭐ *"**A listener who misses the word still feels the vocal disappear.**"*

It also, unprompted, praised the EXCLUDE field on exactly the run's own reasoning — that the song excludes *truck-driver key change · modulation into the last chorus · added octave · final-chorus melisma · triumphant major-key ending*, i.e. **every uplift device in pop, because the song's entire claim is that the ending is not redeemed** — and called that *"a writer who knows what their idea is adjacent to."*

### ⛔ AND IT INDEPENDENTLY FOUND THE WEAKNESS FINDING 10 PREDICTS

Without seeing the shared-shape analysis, the blind judge named the same risk from inside a single song:

> *"Anaphora is its only rhetorical gear… flat declaratives, no rhyme, no figurative relief, and a prompt that forbids melisma, lift, and any chorus rise. **Five near-identical choruses in that register is a real monotony risk, and the whole payload rides on one word landing.**"*

⭐ **That is FINDING 10 seen from one song rather than across six.** The method that makes each song un-templateable is the same method that makes it austere, and a blind reader felt it without being told the method existed. **Two independent instruments, different levels, same result.**

### The judge named its own limit, and it is the right one

> *"Confidence: high that B is third; **moderate** that C beats A. What would flip C and A is the render — C's entire bet is that a flat, unlifted, no-melisma contralto holds attention across five near-identical choruses, and if it lands as monotonous then A wins… **That is an audio question, and nothing on the page settles it.**"*

⛔ **So this is a TEXT win, and it is stamped as one.** It says the ranking is contingent on a render nobody has heard, which is precisely the class of claim this house has been burned by. **It is evidence that the writing is good; it is not evidence that the record is.**
