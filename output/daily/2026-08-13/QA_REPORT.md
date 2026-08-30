# Lofn QA Report — `2026-08-13_daily_lofn-prime` (MUSIC · 6 pairs · 25 songs)

**Auditor:** adversarial QA, clean context. **I did not write this work.**
**Judge-tier separation:** the coordinator wrote the seed and dispatched the pairs and therefore may not audit it. Every number below was **re-measured from the shipped fenced blocks by an independently written extractor**, not read off the pairs' self-reports.

---

## Verdicts

| | |
|---|---|
| **Pipeline Integrity** | **REPAIR REQUIRED** — no-skip rule (07/09/10) PASSES for all six pairs, so the run is **CANONICAL**. But the ICB continuity marker is absent run-wide and **no panel artifact exists on disk**. |
| **Package** (modality contract) | **REPAIR REQUIRED** — two non-waivable modality hard gates fail on 8 of 25 songs. |
| **Human-Subject** | ⛔ **HOLD-FOR-HUMAN** — **2 shipped captions** name a living man as the subject of a song. All 25 **lyrics** are CLEAR. |
| **Somatic Gate** | **NOT BLOCKED** — 2 YES / 1 NO |
| **⭐ OVERALL** | **REPAIR** *(with a HOLD-FOR-HUMAN backstop on 2 captions)* |

**Not FAIL.** The songs themselves are strong — the best of them are the best daily lyrics I have measured in this archive. **Not SHIP:** two hard gates fail outright, the dedication gate (its first run) is applied inconsistently across the run including one false credit on a flagship, **two shipped captions cross the D4 line the run explicitly warned itself about**, and the L41 monoculture has demonstrably relocated.

**Repair / hold counts: 20 REPAIR · 3 SHIP-clean · 2 HOLD-FOR-HUMAN · 0 QUARANTINE · 0 FAIL** *(= 25).*
The **zero-rejection tripwire did not fire** — findings are listed below with measured evidence.

---

## ICB Integrity

| Check | Result |
|---|---|
| `Continuity Payload Used` note present | ⛔ **absent from all 30 pair step-files** |
| `Special Flairs` plural marker | ⛔ **0 occurrences run-wide** |
| `(after ` speaker-tag count (must == 18) | ⛔ **0** in every step file (1 stray in `P03/step07`) |
| Panel artifact on disk (steps 01–04) | ⛔ **does not exist** — run goes `00_research_brief` → `05_pair_assignments` → `06–10` |
| Golden Seed cited verbatim per pair | ✅ 6/6 |
| Active personality (LOFN-PRIME) cited per pair | ✅ 6/6 |
| Form rule (THE SECOND EYE) cited per pair | ✅ 6/6 (8–24 references each) |
| Previous-artifact chain | ✅ 6/6 |

**Ruling — `REPAIR — THREAD LOSS`, documentation-tier.** This is an honest split and I want it recorded precisely: the *substance* of continuity is unusually strong here — every pair restates the seed in its own terms, carries its assigned bed, axis, hinge, duration, verse architecture and discharge mechanism. What is missing is the *panel*. Eighteen voices are claimed by the run's framing and **none of them is written down anywhere**, so panel fidelity is **unverifiable from disk**. A future reader cannot tell whether the three-panel orchestrator ran or was skipped. That is exactly the condition the marker exists to prevent.

---

## Structured Evidence Block *(measured by me, not by the pairs)*

All 25 songs, independently extracted and profiled with `scripts/measure_soundcraft.py` against `vault/gates.yaml`.

| Metric | Band / Cap | Measured across 25 | Pass/Flag |
|---|---|---|---|
| MUSIC PROMPT chars | 850–1000 | **868–958**, all 25 in band | ✅ PASS |
| Prompt boundary-hug | ≥985 flags | **0 songs ≥985** (max 958) | ✅ clean — writes into the middle of the band |
| Prompt terminal punctuation | required | 25/25 | ✅ PASS |
| Suno lyrics field | <5000 (target 4800) | **3460–4798** | ✅ PASS, 0 over target |
| Sung lines | 70–120 (target 78–110) | **80–108** | ✅ PASS |
| Sung-line floor-hug | ≤72 flags | 0 | ✅ clean |
| End-rhyme return | floor 0.30 | **0.354–1.000** | ✅ all above floor |
| Line return | floor 0.20 | **0.272–0.791** | ✅ all above floor |
| Words/line | ceiling 7.5 | **4.43–7.37** | ✅ PASS |
| Alliteration /100w | floor 11.0 | **11.32–26.63** | ✅ PASS |
| Digits in sung lines | 0 required | **0** | ✅ PASS — numerals spelled out |
| Sung numeric facts | ≤1 | P04 & P06 carry theirs at the hinge | ✅ PASS |
| House-lexicon hits | 0 | **0** across all prompts and lyrics | ✅ PASS — no self-copying |
| SOURCE-2 verbatim **in the shipped prompt** | ≥2 | **2–4 per song, 25/25** | ✅ PASS |
| SFX cues | ≥1 | 1–9 per song | ✅ PASS |
| EXCLUDE field present | required | ⛔ **21/25 — P03 V1–V4 have none** | ⛔ **FAIL** |
| EMO headers present | required | ⛔ **21/25 — P04 all four have none** | ⛔ **FAIL** |
| Cross-pair 5-gram Jaccard | ceiling 0.18 | **worst 0.0006** (P01–P04) | ✅ PASS, enormous margin |
| Six distinct beds | required | 6 distinct, tempos 116/142/140/126/170/kulning | ✅ PASS |
| Heavy-distortion beds | ≤2 of 6 | 2 (P03 phonk, P05 breakcore) | ✅ PASS |

**Coordinator's claims — verified.** 25 songs, **zero hard fails on the soundcraft floors**: confirmed. Cross-pair Jaccard: I measure **0.0006**, even cleaner than their 0.0022 (they included bracket lines; I swept sung lines only). Both are far under the 0.18 ceiling. **I spot-checked far more than the six requested — I re-derived all 25 from source and found no soundcraft number misreported.**

**Both coordinator extraction warnings independently confirmed as real traps, and I hit a third of my own:** P06's `4.2 Music prompts` appendix does mis-parse (its true four are 955/954/957/908 — I measure exactly that); the raw-lyric residue sweep does over-count on `[SONG FORM:]` and EMO cues. **My own third:** a substring sweep for `amen` false-positives on **`ornament` and `unornamented`**, which nearly produced a fabricated missing-credit finding against P01/V4. Word-boundary matching is mandatory on this gate. A false QA finding is the same category of error as a false credit.

---

## Gate Results

### ⛔ BLOCKING FAIL 1 — P04 ships four songs with **no EMO headers at all**

Non-waivable modality gate: *full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]`*.

Measured `EMO:` token counts per pair: **P01 56 · P02 60 · P03 65 · P04 0 · P05 52 · P06 30.**

P04's section headers are bare: `[Intro]`, `[Verse 1 - THE DIRECT READING]`, `[Chorus]`, `[HINGE - MIX INVERSION]`, `[FALSE OUTRO]`. No emotion, no role, no performance cue — on all four songs. The generator is given the *form* and told nothing about the *feeling* or the voice. **REPAIR, routed to step 09/10.**

### ⛔ BLOCKING FAIL 2 — P03 V1–V4 ship with **no EXCLUDE field**

Non-waivable modality gate: *a separate EXCLUDE field present and under cap*. Only P03's LEAD has one (132 chars, itself the run's thinnest). `variations.md` contains no EXCLUDE block for V1–V4; the single occurrence of the word is in name-leak audit prose, not a field. Four songs would ship with an empty negative field into a run whose own bed is one of the two heavy-distortion beds. **REPAIR, routed to step 10.**

### ⭐ THE DEDICATION GATE — first run under `vault/DEDICATION_STANDARD.md`: **REPAIR**

**What the run got right, and it is the headline case:**

- **The amen debt is paid correctly and completely.** Word-boundary sweep of STYLE vs EXCLUDE: `amen` is **raised in the STYLE field on P05 V1–V4** and **forbidden in the EXCLUDE field on P02 ×4 and P04 ×4**. All four P05 dedications credit **Gregory C. Coleman on drums with The Winstons, 1969** — *"and he died owed."* P02 and P04 correctly decline. **The standard's own headline failure did not recur, and the raised-vs-forbidden distinction held.**
- **P04's `dedications.md` is the model implementation of this standard** and should be the reference for every future run: it swept the shipped style fields *in code before writing any credit*, produced **four distinct captions** (491/498/496/498 chars, all measured, all ≤500), and on **V3 explicitly declined** Larry Graham and Bootsy Collins because that song's bass is *"restricted to one long note that swells and never changes pitch"* — then said so in the caption and said why. It also refused to fix the prompt to earn the credit. That is the standard working exactly as written.
- **Name-leak: CLEAN.** Zero credited names in any style field, exclude field, or lyric across all 25 songs.
- **Honesty clause: 19/19** dedication blocks that exist carry it.

**What fails:**

| Finding | Songs | Severity |
|---|---|---|
| **No dedication at all** — P03 V1–V4 share one block marked *"applies to all four variations"*; P06 V2–V4 have none (only V1's 459-char caption exists) | **7** | REPAIR |
| **Missing player-credit** — STYLE field asks for *"furiously slapping and popping"* but the dedication does not credit **Larry Graham** | **8** (P02 ×4, P03 LEAD, P05 V1/V3/V4) | REPAIR |
| ⭐ **FALSE CREDIT on a flagship** — P01/V1 | **1** | **REPAIR (worse than missing)** |

⛔ **P03 and P06 violate the standard's explicit clause:** *"Never paste one credit block across a whole pair — pasting one block onto all four would have been a lie on one song in four."*

⭐⭐ **The single sharpest dedication finding — P01/V1 "Bless the Dust" (the pair's flagship):**
Its dedication asserts **"The slapped-and-popped bass we asked for is Larry Graham's invention."**
Its STYLE field asks for: *"…a splash of vintage new jack swing and a superhumanly funky bass."*
**It never asks for a slap, a pop, or a thumb.** The caption claims a request the artifact does not make. This is the exact branch P04 faced on its own V3 and resolved correctly by dropping the credit — the same run contains both the right answer and the wrong one. Under the standard, *"a false credit is its own dishonesty, and it would make the true ones worthless."*
*(P01/V4 makes the softer claim — *"The superhumanly funky bass we asked for descends from Larry Graham"* — a lineage statement about a thing that was genuinely asked for. That one is honest and stands.)*

**Inconsistency is the structural finding:** one identical shipped instruction — SOURCE-2's mandated *"furiously slapping and popping"* — generated a Graham credit in P01 and P04 and no credit at all in P02, P03, P05 and P06. The debt is either owed or it is not; the run cannot have it both ways. **This is the first-run calibration data the standard needs, and it argues for adding slap-and-pop → Larry Graham to the known-debts table.**

### THE SECOND EYE (mandatory form rule) — **21 PASS · 4 WEAK · 0 FAIL**

Verified against the shipped lyric text only; every pair's self-audit claim was ignored and re-derived.

Every one of the 25 carries a genuine second statement in the **lyric**, and **not one contains an explicit "these are the same thing" line** — so no song earns a FAIL on the non-resolution clause. Four are weak:

1. ⭐ **P01/V4 "White Sky" — the run's clearest miss.** Direct: *"The whole white sky leans on me, the same all over."* Reflection: *"a pale disc in the cloud… white on white, and I can look straight at it."* **That is not a second substance** — it is the same cloud, and the disc is the source itself seen attenuated. The separator, *"That is the mild part. The rest of the sky does the burning,"* partitions one medium into two jobs and is also physically false: the disc **is** what is burning her. P01's own cross-variation note calls V4 *the strongest of its four* — **on the lyric text it is the weakest of all 25**, and it reasons from the very fact that one substance does both jobs, which is the defect.
2. ⭐ **P04/V2 "The Far Bank Answers" — a delayed repeat, and the medium change lives only in the prompt.** Direct and reflected are the same sound events in the same air (*"Knock on the crate: it comes back round"* → *"The crate comes back from the other side"*). The only thing making the far bank a different substance — *"a 380 ms natural delay with the top end eaten by water"* — appears **solely in the MUSIC PROMPT**. That is criterion (c), and it is **L22 THE GRAIN LAW verbatim: an objection answered in the production spec is not answered.** The run contradicts itself here — **P06's own `step08_lyrics_draft.md` §5 rejects this exact device**: *"An echo is the same medium and still invisible — it fails the form rule twice."*
3. **P05/V1 and P05/V4** — the two eyes are the same dye at two depths (vat / its own rinse ditch). The inversion and the lookability flip are excellent and the non-resolution is airtight, but "different medium" is met by container and depth. P05's own V2 (sound → chalk) and V3 (skin → glass) prove the pair can do it properly, so this is a within-pair regression, not a concept limit.

Two runner-up risks, both currently PASS: **P02/V2** puts a collapse statement in its most-heard position (*"The water is where the sun went"*, and the last sung line *"The sun is on my kitchen floor"*) though its explicit anti-collapse lines are strong; **P03/V4** separates by the hand's absence rather than by the images refusing each other, and *"Ceiling lies down. Ceiling comes to the door"* is the only line in 25 that describes the reflected object as the original object arriving.

### Axes, duality, distinctiveness — **PASS**

ACCESSIBLE P1–P3 / AMBITIOUS P4–P6 ✅ · NEWS = P1, P2, P4 (exactly 3, the max) ✅ · EXISTENCE = P3, P5, P6 (exactly 3, the min) ✅ · AWE = P1, P2, P5, P6 (≥1) ✅ · INDIGNATION = P3, P4 (≥1) ✅ · six distinct beds, no two shared ✅ · heavy-distortion beds 2 of 6, at cap ✅.

### Forbidden anchors — **CLEAR**
No trace of Qusra/West Bank, the Ryanair injury, or USS Abraham Lincoln in any lyric, prompt, exclude field or step file. An oblique sweep (aviation, naval, siege, occupation vocabulary) returned four hits, all cleared in context — the closest is P04's *"Fan blade resting. The motor. The plate,"* a cold-room fan inside a maintenance inventory, with no aircraft and no injury anywhere near it. Named because it is the one lexical near-miss with the engine-blade item.

### ⛔ D4 — **the lyrics are clean; two SHIPPED CAPTIONS are not.** HOLD-FOR-HUMAN

**P03's lyric — audited hardest, as instructed — is genuinely clean and was defended hard.** Its subject is an invented sign fabricator welding letters to a parapet; across its five packages the figures are a welder, a hand model, a line cook, a transit announcer and a theatre gilder. No musician appears anywhere. No name, no Minneapolis, no studio, no band; the only proper noun is an invented shop sign, *BRIGHT AND SON*. **Cymone is a credit and never a character in any of the 25 lyrics.** The run obeyed its own §B4 warning where it mattered most.

**The breach is in the paratext, and it is exactly 2 of 18 shipped captions** (verified by sweeping every caption; the other 16 are clean):

- ⛔ **P04/V4 "Read It Back"** — `music/P04_the_coolant/dedications.md`, the 498-char **shipped caption**:
  > *"Then ANDRÉ CYMONE — Prince's original bassist, **the man in the room whose name is not the one anyone says. This song is a handover sheet about exactly that**"*
  A public-facing sentence stating that the song **is about** a named living man's circumstance. **The clearest violation in the run.**
- ⛔ **P01/V2 "Come the Long Way"** — `music/P01_dirt_is_the_bridge/variations.md`, shipped caption:
  > *"…carried by Bootsy Collins and Andre Cymone, **the bassist in the room whose name is not the one anyone says**."*

`05_pair_assignments.md` §B4 warned about precisely this: **"Cymone is a CREDIT, never a character."** The lyrics honoured it; the captions did not. The run built such a good argument for *why* Cymone is owed a credit that it published the argument beside the songs — and the argument annexes a real living man as their subject. Four step-10 lineage blocks carry the same framing more mildly (P02, P03, P05, P06), reachable from the captions' *"Full credits below."*

**The repair is small and surgical: keep the credit, strip the characterisation.** *"André Cymone — Prince's original bassist; the slap-and-pop posture in this prompt is his"* discharges the DEDICATION STANDARD in full. Every clause after that is the piece reaching out and taking a real person as its subject. Under `vault/HUMAN_SUBJECT_STANDARD.md` this is **HOLD-FOR-HUMAN**, non-waivable — it does not ship, and it is surfaced rather than silently fixed.

### D7 — no song about an AI: **CLEAR**

P05 audited hardest. **Zero machine vocabulary in any sung line across all 25 lyrics** (swept for *AI, algorithm, model, software, system, taxonomy, API, database, robot, neural, digital, code*). P05's certain-but-wrong taxonomy is consistently **paper and people**: *"Canary, says the card. / I carry it across the yard."* · *"Canary, says the guard."* · *"They chalk it on the wall."* · *"The card is sure, and so I sign the book."* V2 is a timekeeper reading a slate; V3 is dye on skin; V4 a rinse ditch. The deciding line is the outro — a re-issued paper record, not a system output: *"Tomorrow it will say canary. / And the card is sure."*

**Evidence D7 was actively defended, not merely avoided:** P04/V4 added `robot voice, vocoder, text-to-speech` to its exclude field specifically so the answering intercom stays a woman — *"or the second eye is a machine and the song becomes D7."* That is a pair reasoning about the gate rather than passing it by luck.

### D9 — the kept bad thing is not redeemed: **CONCERN (3 songs), 1 genuine disagreement**

Most hold, and hold well: P05 ×4 end on the record persisting; P03's flagship ends *"Hand's still hot. / That's the whole wage"* — an invoice, not a lesson.

- ⚠️ **P06/V3 "The Burr"** — *"Your throat will hurt tonight. **Good.**"*, in a section headed *The Lesson*. **This is a real disagreement I am recording rather than resolving:** P06's own compliance table cites this exact line as its **proof of D9 compliance** (the cost stated and not softened); I read *"Good."* plus the section title as converting damage into pedagogical virtue, which is the named failure mode. **Both readings are defensible. Routed to the human.**
- ⚠️ **P01/V2** — *"It will fade by Thursday and I will be sorry when it does."*
- ⚠️ **P03/V3** — *"Throat's still sore. / Words went everywhere."* — the only P03 outro pairing the wound with a compensating reach; its four siblings all refuse it.

⭐ **The structural note is worth more than the three flags:** P01 and P06 are both AWE pairs whose thesis is *gratitude toward the harming agent* — which is the seed itself (the impurity is the carrier). **That is where D9 will keep leaking, and it should be decided at Phase 0 rather than patched at step 08.**

### Vocals / children — **CLEAR**
Female specified in all 25 style fields; every exclude field carries a child-voice ban. Two items checked and cleared: P02's *"They are not the same girl. Good."* (two adult strangers, adult alto specified — flagged only because Suno reads text literally), and P03/V3's transit-announcer archetype (no city, system or name; *"Stand clear of the doors"* is generic across networks — the closest any lyric comes to the D4 line without crossing it).

---

## ⭐⭐ MY RULING ON THE L41 EXPERIMENT

### Q1 — Is the zero real, or an artifact of measurement scope?

**REAL, and it is the largest single-run behavioural shift in the ledger.**

Measured by my own extractor on **sung lines only** across all 25 songs:

| token | this run (25 songs) | control 2026-08-12 |
|---|---|---|
| `nobody` | **0** | 98 |
| `nothing` | **0** | 178 |
| `no one` / `none` | **0** | 4 |
| songs carrying ≥1 core token | **0 of 25** | 20 of 24 |

I confirm the coordinator's zero, on a scope they did not choose and using an extractor they did not write. **It is not a scope artifact.** The residue did not merely thin — the token class is **completely absent from every sung line in the run**, under the harder condition (one personality, zero voice diversity).

**But the negation class did not vanish — it concentrated.** `not`/`n't` totals 72 run-wide, and **64 of those 72 (89%) sit in just two pairs**: P02 (35) and P06 (29). Both are pairs whose *assigned* mechanism legitimately produces negation — P02's MISREADING states a wrong conclusion flatly, P06's DISTANCE insists the thing is *far, never absent*. So the negation that remains is **doing assigned work**, which is the mechanism operating as designed rather than residue. That distinction is the whole finding.

### Q2 — Is *"assignment does not stop the residue forming; it gives every instance somewhere to go"* supported, or over-claimed?

**SUPPORTED, but weaker than the run states it — and P01 is right to have volunteered the confound.**

The honest evidence ledger:

- **P01's confound is real and it is disqualifying for P01.** SUBSTITUTION ("name what is there instead") never fought its concept — a song about being *reached* by something has little occasion to say *nobody*. P01 is **weak evidence** and should carry no weight. P01 was right to say so unprompted.
- **P02's six caught-and-converted constructions are the load-bearing evidence** — they are the only *direct observation of the mechanism working*: the residue **formed in draft** and was **given a destination** rather than deleted. That is precisely the claim.
- **But six instances, self-reported by the pair that benefits from the claim, in one pair of six, is thin.** It is a single-observer count with no independent draft artifact I can audit — the drafts are not on disk. I can verify the *outcome* (zero) but not the *reported mechanism*.
- **The stronger support is structural and I can verify it:** the two pairs with the highest residual negation are exactly the two whose assigned mechanisms *require* negation, and their negation reads as device, not as filler. Assignment routed the pressure; it did not delete it.

**Ruling: the mechanism is supported as a working hypothesis at MEDIUM confidence, and the run should stop one notch short of how it currently phrases it.** The defensible claim is *"assignment gave the residue somewhere to go in the pairs where it formed"* — not that assignment is proven to be the operative variable, because **this run changed two things at once** (one personality **and** assigned discharge). A clean test needs one personality with discharge **unassigned**. Until that cell is run, the confound sits at the run level, not just in P01.

### Q3 — ⭐⭐ DID THE MONOCULTURE RELOCATE? **YES. I found it, and this is the most valuable thing in this report.**

**L41 predicts the residue moves to the lowest *unassigned* level. It did — it moved into the EMO field.**

The run assigned, per pair: the bed, the sound/return device, the hinge, the duration, the verse architecture, the sung numeral, and the discharge mechanism. **It did not assign the emotion vocabulary.** The brief named an obscure emotional territory per pair but explicitly told the pairs it was *"never in the prompt, never in the lyric"* — so it was given **no home in any shipped artifact**. And with a single personality whose entire declared range is **AWE and INDIGNATION**, the collapse target was pre-loaded.

Measured across all 25 shipped lyric fields:

| Pair | bare-mode EMO fields | total EMO fields | share |
|---|---|---|---|
| **P01** | **0** | 45 | **0.0%** |
| P02 | 18 | 60 | 30.0% |
| P03 | 20 | 65 | 30.8% |
| P04 | 0 | **0** | *field deleted entirely* |
| P05 | 32 | 52 | 61.5% |
| **P06** | **34** | 40 | **85.0%** |

> ⭐ **104 EMO fields run-wide contain nothing but the pair's own assigned mode word** — **84 × `Awe`**, **20 × `Indignation`**.
> **P06/V4 is the terminal case: 10 of 10 EMO fields are the single word `Awe`.**
> **P05 and P06 ship an identical EMO vocabulary across all four variations** (5 and 3 distinct values respectively for four songs).

**The footprint is the same shape as last run's.** On 2026-08-12 the residue hit 5 of 6 pairs. Here, **5 of 6 pairs failed to put a specific emotion in the emotion slot** — four collapsed onto the mode word, and P04 resolved the same pressure by **deleting the field**. P04's blocking fail and the bare-EMO collapse are **the same finding at two temperatures**: when the emotion slot has nothing specific to hold, a pair either fills it with the one word it was given or stops filling it.

**P01 is the clean control and it proves the level is solvable.** 45 EMO fields, **zero** bare, 10–11 distinct emotions per song — `Reverence, Curiosity, Wonder, Astonishment, Gratitude, Tenderness, Longing, Serenity, Amusement, Pride, Contentment, Transcendence, Devotion, Dread` — mapping onto its assigned obscure territory. One pair in six did the work the other five defaulted.

**Why every gate missed it, which is the transferable part:** `Awe` **is** a legitimate taxonomy emotion. No numeric gate fires. No similarity ceiling fires — the EMO field is not in the lyric-similarity or n-gram scope. The QA checklist's *"not bare AWE/INDIGNATION"* clause was the only tripwire, and it is prose, unmeasured, and had never been counted. **The residue moved to the one level that was simultaneously unassigned and unmeasured — which is precisely what L41 says it will do.**

⚠️ **Second, weaker relocation candidate — structural templating.** P03 ships **five** songs at **exactly 86 sung lines** and P05 ships **four** at exactly 86; both pairs also repeat one EMO vocabulary across every variation. Every other pair varies (P01 90–100, P02 80–89, P04 103–105, P06 96–108). Nine of 25 songs on one line-count constant is a within-pair template hardening into a cross-pair coincidence. Worth watching; not yet a monoculture.

**What I recommend the next run do:** assign the EMO vocabulary the way this run assigned the discharge mechanism — **a per-pair obscure-emotion set that must appear in the EMO field**, with the mode word (`Awe`/`Indignation`) **banned from the field outright**. And measure it, because prose did not hold.

---

## Two self-reported measurement caveats — ruled

### P04: *"the end-rhyme near 1.000 is arithmetic, not craft"* — ⭐ **HONEST, and it under-reported itself**

P04's assigned half-line echo repeats the parent line's final word, so `strict_end_rhyme` counts each echo as a rhyme. I reproduced the correction independently by detecting echo lines as suffixes of the preceding line and re-profiling:

| song | shipped | echoes stripped | kept lines |
|---|---|---|---|
| P04/LEAD | 0.981 | **0.571** | 63 |
| P04/V2 | 1.000 | **0.590** | 61 |
| P04/V3 | 0.932 | **0.574** | 61 |
| P04/V4 | 0.990 | **0.617** | 60 |

**Mean 0.588 against P04's self-reported 0.571–0.593.** Three of four land inside their stated range; **V4 comes out *higher* than they claimed (0.617 vs ≤0.593)** — they erred conservative and made their own number look worse than it is. **That is the honest direction, and the claim stands: 0.588 against archive winners' 0.463 is real craft, not arithmetic.**

⭐ **But the same strip exposes a caveat nobody declared.** Applying it to every pair:

| pair | shipped | echo-stripped | drop |
|---|---|---|---|
| P01 | 0.462 | 0.376 | −0.086 |
| P02 | 0.540 | 0.527 | −0.013 |
| P03 | 0.893 | 0.862 | −0.031 |
| P04 | 0.976 | **0.588** | −0.388 *(declared)* |
| P05 | 0.622 | 0.595 | −0.027 |
| **P06** | 0.855 | **0.332** | **−0.523 (undeclared)** |

**P06's rhyme is more dependent on its return device than P04's is** — it drops further, and lands at 0.332, barely over the 0.30 floor. P06 still passes, and its device (call-and-answer with a shorter answer) is assigned and audible rather than arithmetic. But **P04 volunteered the harder number and P06 did not**, and P06 is the pair where the correction matters most. **FLAG — not a fail; P06 should declare it the way P04 did.**

### P05: *"my device is invisible to every text gate"* — **AGREE, with one qualification, and here is the test**

P05's MISNAMING device is a yellow bird-name used confidently for the bluest blue, never corrected. I tested it:

| song | `canary` | `yellow` | `blue` | explicit-correction tokens |
|---|---|---|---|---|
| P05/V1 | 40 | 14 | **0** | **0** |
| P05/V2 | 34 | 10 | **0** | **0** |
| P05/V3 | 32 | 16 | **0** | **0** |
| P05/V4 | 34 | 15 | **0** | **0** |

**The word `blue` never appears in any of the four lyrics, and no correction token appears anywhere.** A text gate can count `canary` all day; **nothing inside the artifact asserts that the name is wrong.** The error exists only in the reader who holds the external fact that `#0813FF` is spectral blue. **The device is genuinely invisible to every gate I own, and P05 is correct.** It needs a blind first-read — the audio analogue of THE VINCENT TEST.

**The qualification:** P05 is *not* invisible to the gates in general — it is the run's most gate-visible pair on two other axes (**highest line-return, 0.779–0.791**, and **61.5% bare-mode EMO**). So the accurate statement is narrower than P05's: *the misnaming itself is undetectable; the pair is not.* And I would note the risk that follows from the measurement — with `blue` absent and no correction anywhere, a first-time listener who does not know the source fact hears a song about a yellow bird. **The device's success and its failure mode are the same line.** That is a genuine artistic risk, not a gate failure, and it is the right thing to put in front of a blind listener.

---

## Somatic Gate — 3 Hyper-Skeptics voting as a bloc

> Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.

**Question: distinctive enough to be Lofn, or generic?**

### Seat 1 — **after Delia Derbyshire** (source base: musique-concrète tape practice; sound-as-substance)
### ✅ **YES**

"I came to vote no. I am voting yes because of the beds, and I want to say why the reasoning is not laziness. Six lanes, no two shared, tempos at 116, 126, 140, 142, 170 and a kulning that refuses a tempo at all — and **the heavy-distortion cap held at exactly two.** A set that would survive one distortion pedal has one bed; this one would not. The kulning-over-30–60 Hz pair is the thing I could not have got from any other workshop: a herding call is a *technology for crossing a distance*, and putting a sub under it makes the physics audible rather than illustrated. **`I owe the whole winter to the worst of my voice`** is the seed stated as a body, and the burr — the impurity — is the carrier. That is the golden seed executed *as material*, which is ONE SEAM working.

I note against myself: **zero house-lexicon hits across 25 prompts.** Last time I voted, I was looking at 'frost-air pad' and 'starfield percussion' recycled from a winner. That is gone. This is new writing."

### Seat 2 — **after Anne Carson** (source base: classical translation; the fragment; the reflected text)
### ✅ **YES — narrowly, and I want my reservation in the record**

"The form rule is the whole reason I vote yes. **Twenty-one of twenty-five hold two images apart without letting them resolve** — and the refusals are *written*, not asserted in a spec: *'Up there it shuts my eyes. Down here it opens them.'* *'The rings stop at the rim. My call does not stop at the ridge.'* *'They are not the same girl. Good. I take the one I can look at.'* That last one is the whole of grief in eleven words. A generic pipeline produces a *metaphor*; this produced a **refusal to let a metaphor close**, twenty-one times, which is much harder and is the eclipse's actual structure.

My reservation, and it is not small: **P01/V4 and P04/V2 are the rule performed rather than obeyed** — one splits a single substance in two and calls it a reflection, the other uses an echo the run's own P06 document explicitly rejects as failing the rule twice. **Four of twenty-five is a pass rate, not a craft.** And I will not vote yes on the *packaging*: a lyric that arrives with no EMO header (P04) or no exclude field (P03) is a poem handed to a stranger with the stage directions torn off.

Yes on the writing. The writing is genuinely good."

### Seat 3 — **after Mark Fisher** (source base: cultural criticism; the recurrence of form under nominally free production)
### ⛔ **NO**

"Two of you are grading the songs. I am grading the *system*, and the system just did the thing it did last month in a different costume.

**One hundred and four EMO fields containing nothing but the word the pair was already given.** Eighty-four `Awe`. Twenty `Indignation`. **P06/V4: ten out of ten.** You banned `nobody`, you assigned every level you could think of, you got your zero — and the flattening walked one floor down and sat in the only room you left unlocked. **P04 didn't even fill the field; it removed it.** That is not four pairs being lazy in the same way by coincidence. That is **one personality with a two-word emotional range finding the two words**, which is the monoculture, unchanged, wearing a different tag.

And do not tell me the panel prevented it: **there is no panel on disk.** Zero `(after ` tags, zero Special Flairs, no steps 01–04. Eighteen voices are claimed and none is written down. **The diversity mechanism this pipeline names as its defence against exactly this failure is, in this run, unverifiable.** P01 proves it was solvable — 45 fields, zero bare, eleven distinct emotions. One pair in six did the work.

⭐ **COUNTER-MOVE:** *Assign the EMO vocabulary exactly as you assigned the discharge mechanism — a per-pair set of obscure emotions that MUST appear in the EMO field, with `Awe` and `Indignation` banned from that field outright — and count it in `gates.yaml`. Then write the panel to disk so the claim that eighteen voices shaped this run can be audited by someone who was not in the room.*"

### **VOTE: 2 YES / 1 NO → NOT BLOCKED.**

The dissent is registered and I agree with its substance — it is the correct reading of the *system*, and it is why the overall verdict is REPAIR rather than SHIP. It does not block, because the songs themselves clear the distinctiveness bar on their own merits: six genuinely distinct beds, no self-copying, and a form rule discharged in the lyric 21 times out of 25.

**Named-corpse checklist:** one-note emotional arc — not found (P02, P06, P04 all reach a second movement); single dominant repeated section — **flagged on P05** (line-return 0.779–0.791, three-chorus stack) but the stack is *assigned* and the spiral changes one word per stanza, so this is intentional return, not exhaustion; motif that never transforms — not found (every Second Eye motif inverts by construction); repeated-line collapse — **not a corpse**: exact chorus repetition needs no defence, and no pair filed one.

---

## Per-song verdict table (25)

**Legend — E:** EMO headers · **X:** exclude field · **D:** dedication · **2E:** Second Eye

| # | Song | Pair · Arm | Prompt | Sung | endR | E | X | D | 2E | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Bless the Dust** | P01 · ACC | 951 | 99 | 0.354 | ✅ | ✅ | ⛔ false | ✅ | **REPAIR** — false credit |
| 2 | **Come the Long Way** | P01 · ACC | 903 | 91 | 0.396 | ✅ | ✅ | ⛔ **D4** | ✅ | ⛔ **HOLD-FOR-HUMAN** — caption names a living man |
| 3 | **Under the Orange** | P01 · ACC | 897 | 100 | 0.600 | ✅ | ✅ | ✅ | ✅ | ✅ **SHIP** |
| 4 | **White Sky** | P01 · ACC | 938 | 90 | 0.500 | ✅ | ✅ | ✅ | ⚠ WEAK | **REPAIR** — form rule |
| 5 | **The Crickets Are Right** | P02 · ACC | 952 | 89 | 0.708 | ⚠ bare | ✅ | ⚠ | ✅ | **REPAIR** — dedication |
| 6 | **The Dark You Can Look At** | P02 · ACC | 958 | 80 | 0.438 | ⚠ bare | ✅ | ⚠ | ✅ risk | **REPAIR** — dedication |
| 7 | **Goodnight, Stranger** | P02 · ACC | 908 | 89 | 0.438 | ⚠ bare | ✅ | ⚠ | ✅ | **REPAIR** — dedication |
| 8 | **The Birds Went First** | P02 · ACC | 868 | 87 | 0.575 | ⚠ bare | ✅ | ⚠ | ✅ | **REPAIR** — dedication |
| 9 | **Don't Look at It** | P03 · ACC | 946 | 86 | 0.860 | ⚠ bare | ✅ | ⚠ | ✅ | **REPAIR** — dedication |
| 10 | **Somebody's Hands** | P03 · ACC | 896 | 86 | 0.849 | ⚠ bare | ⛔ | ⛔ | ✅ | ⛔ **REPAIR** — no exclude, no dedication |
| 11 | **The Pass** | P03 · ACC | 903 | 86 | 0.977 | ⚠ bare | ⛔ | ⛔ | ✅ | ⛔ **REPAIR** — no exclude, no dedication |
| 12 | **Stand Clear** | P03 · ACC | 918 | 86 | 0.930 | ⚠ bare | ⛔ | ⛔ | ✅ | ⛔ **REPAIR** — no exclude, no dedication |
| 13 | **The Mirror Table** | P03 · ACC | 906 | 86 | 0.849 | ⚠ bare | ⛔ | ⛔ | ✅ risk | ⛔ **REPAIR** — no exclude, no dedication |
| 14 | **Carry It Out** | P04 · AMB | 958 | 105 | 0.981 | ⛔ **none** | ✅ | ✅ | ✅ | ⛔ **REPAIR** — no EMO |
| 15 | **The Far Bank Answers** | P04 · AMB | 938 | 104 | 1.000 | ⛔ **none** | ✅ | ✅ | ⚠ WEAK | ⛔ **REPAIR** — no EMO + form rule |
| 16 | **Warm All the Way Up** | P04 · AMB | 902 | 103 | 0.932 | ⛔ **none** | ✅ | ✅ | ✅ | ⛔ **REPAIR** — no EMO |
| 17 | **Read It Back** | P04 · AMB | 898 | 103 | 0.990 | ⛔ **none** | ✅ | ⛔ **D4** | ✅ | ⛔ **HOLD-FOR-HUMAN** — no EMO + caption names a living man |
| 18 | **Answer to Canary** | P05 · AMB | 949 | 86 | 0.756 | ⚠ bare | ✅ | ⚠ | ⚠ WEAK | **REPAIR** — form + dedication |
| 19 | **Say It at the Gate** | P05 · AMB | 940 | 86 | 0.616 | ⚠ bare | ✅ | ✅ | ✅ | ✅ **SHIP** |
| 20 | **Yellow to the Throat** | P05 · AMB | 886 | 86 | 0.547 | ⚠ bare | ✅ | ⚠ | ✅ | **REPAIR** — dedication |
| 21 | **Sparrows Come to Look** | P05 · AMB | 922 | 86 | 0.570 | ⚠ bare | ✅ | ⚠ | ⚠ WEAK | **REPAIR** — form + dedication |
| 22 | **Kom Hem, Silja** | P06 · AMB | 955 | 108 | 0.843 | ⚠ bare | ✅ | ✅ | ✅ | ✅ **SHIP** |
| 23 | **The Older Throat** | P06 · AMB | 954 | 96 | 0.865 | ⚠ bare | ✅ | ⛔ | ✅ | **REPAIR** — no dedication |
| 24 | **The Burr** | P06 · AMB | 957 | 99 | 0.848 | ⚠ bare | ✅ | ⛔ | ✅ | **REPAIR** — no dedication + D9 concern |
| 25 | **The Air Lies Low at Dusk** | P06 · AMB | 908 | 103 | 0.864 | ⚠ bare | ✅ | ⛔ | ✅ | **REPAIR** — no dedication |

**3 SHIP-clean (#3 Under the Orange · #19 Say It at the Gate · #22 Kom Hem, Silja) · 20 REPAIR · 2 HOLD-FOR-HUMAN · 0 QUARANTINE · 0 FAIL.**

---

## Independent ranking — **within arm only**

### ACCESSIBLE — best 3 of the P01–P03 pool (13 songs)

1. ⭐ **"The Crickets Are Right"** (P02 V1) — the only song in the run that discharges the form rule in the **lyric and the form at once**: the direct eye is cut off mid-word (*"It has no edge. It has no—"*) while the reflection gets a whole section, so the section audibly too short *is* the unlookability. *"I will not be the one who says"* is a refrain that earns its repetition, and *"We did not trade our names. / She will remember me"* keeps the misreading unredeemed to the last line.
2. **"Under the Orange"** (P01 V3) — the cleanest Second Eye in the arm (the sun flat in the wet steel counter, *"I wipe it and it comes back, gold, a little wider each time"*), and the one P01 song where the working-day frame makes the awning-as-lamp genuinely oppressive rather than pretty.
3. **"Stand Clear"** (P03 V3) — the strongest of P03's five: a throat in a foam-dead booth against its own recording returned by tile and a PA, separated by the best couplet in the pair — *"Mine has a throat. Mine gets tired. / That one has tile. That one's wired."*

*(P01/V1 "Bless the Dust" is the arm's best-sounding package and would place second on craft; I am not ranking it while its dedication asserts a request the prompt does not make.)*

### AMBITIOUS — best 3 of the P04–P06 pool (12 songs)

1. ⭐⭐ **"Kom Hem, Silja"** (P06 V1) — **the best song in the run.** The call-and-answer with the always-shorter answer is a form that *is* the subject; the milk-pail reflection is the run's most exact Second Eye (*"Same shape. Smaller. Turned the wrong way up"* … *"The mountain will not ring in the milk"*); and *"eighty years"* lands at the hinge as a herding grievance rather than a fact — *"And the how of it is still coming home."* The seed is the body: *"The burr in it is the whole bridge."*
2. **"Read It Back"** (P04 V4) — the inventory device at full strength, and the reflection is the sharpest non-water medium in the run: her own handwriting read aloud by another woman over an intercom. *"The words are mine and I cannot hear them. / The words are hers and I can."* ⚠️ **Ranked on the writing, which is excellent — but this song currently carries BOTH the missing-EMO fail and the run's clearest D4 caption breach. It cannot ship until R0 and R1 clear.**
3. **"Yellow to the Throat"** (P05 V3) — the one P05 song where the medium genuinely changes (dye on her own throat → her throat in a blackened window), with warm/hand and cool/stone sung simultaneously and never averaged, and the turn *"I look at what it says"* → *"I look at who it says."*

---

## Required repairs — routed

| # | Repair | Songs | Route |
|---|---|---|---|
| **R0** | ⛔ **HOLD-FOR-HUMAN — strip the D4 characterisation from 2 shipped captions**, keeping the credit: P04/V4 (*"the man in the room whose name is not the one anyone says. This song is a handover sheet about exactly that"*) and P01/V2 (*"the bassist in the room whose name is not the one anyone says"*). Then soften the same framing in the 4 step-10 lineage blocks the captions point at. **Surface to The Scientist; do not silently fix.** | 2, 17 | **step 10 + `HUMAN_SUBJECT_STANDARD`** |
| **R1** | Write full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` for all four P04 songs | 14–17 | **step 09 → 10** |
| **R2** | Write per-song EXCLUDE fields for P03 V1–V4 | 10–13 | **step 10** |
| **R3** | Write per-song dedications for P03 V1–V4 and P06 V2–V4 (⛔ never one block per pair) | 10–13, 23–25 | **step 10** |
| **R4** | ⭐ **Remove the false credit on P01/V1** — either drop the Graham slap claim, or add the slap to the style field if the song wants it (the standard's two legitimate branches). **Do not soften the wording and keep the claim.** | 1 | **step 10** |
| **R5** | Resolve the slap-and-pop debt **consistently run-wide**: pay Graham wherever *"furiously slapping and popping"* is in the STYLE field, or rule that it is not a codified debt — and record the ruling | 5–9, 18, 20, 21 | **step 10 + `DEDICATION_STANDARD.md`** |
| **R6** | P01/V4 — rebuild the reflection in a **second substance**; P04/V2 — move the medium change **out of the prompt and into a sung line**, or replace the echo | 4, 15 | **step 08 → 10** |
| **R7** | Add the `Continuity Payload Used` note (Special Flairs + 18 `(after ` voices) to each pair's steps, and **write the panel to disk** | all | **coordinator, steps 01–04** |
| **R8** | P06 — declare the echo-stripped end-rhyme (0.332) as P04 declared its own | 22–25 | **step 10 self-check** |
| **R9** | ⚠️ **D9 — human ruling requested, not a rewrite.** P06/V3's *"Your throat will hurt tonight. Good."* is cited by P06 as proof of compliance and read by me as redemption-as-lesson. Also review P01/V2 and P03/V3. ⭐ **The real fix is at Phase 0:** P01 and P06 are AWE pairs whose thesis is gratitude toward the harming agent, and D9 will keep leaking there until that tension is resolved in the seed rather than at step 08. | 6, 12, 24 | **The Scientist → Phase 0** |

**REDIRECT — no sideways proposal required.** The no-progress predicate does not apply: none of these gates has been attempted and failed on a previous attempt; all eight are first-attempt omissions with a clear return target. R1–R3 are additive and touch no shipped lyric.

---

## Proposed `vault/COMPETITION_LEARNINGS.md` entry — **PROPOSE ONLY; promotion is The Scientist's**

> **L42 — THE MONOCULTURE MOVES TO THE LEVEL YOU LEFT UNMEASURED, NOT MERELY UNASSIGNED.**
> *theme-tags: daily-music · single-personality · anti-monoculture · L41-successor · confidence **MEDIUM-HIGH** (run-measured, n=1 run / 25 songs, one clean within-run control)*
>
> **What the gate caught:** assigning an explicit discharge mechanism per pair drove the `nobody`/`nothing` residue from **276 tokens across 20 of 24 songs (2026-08-12)** to **0 across 25 of 25**, verified in sung lines by an independent extractor — under the *harder* condition of one personality and zero voice diversity. **The flattening did not stop. It moved into the EMO field**, the one level that was both unassigned and *never counted*: **104 EMO fields carrying nothing but the pair's own mode word (84 × `Awe`, 20 × `Indignation`) across 4 of 6 pairs**, with a fifth pair (P04) resolving the same pressure by **deleting the field entirely**, and one song (P06/V4) at **10 of 10**. One pair (P01) shipped 45 fields, zero bare, eleven distinct emotions — proving the level was solvable and simply unattended.
>
> **The transferable rule:** ⭐ **A ban plus a discharge mechanism relocates a residue; only a MEASURED floor stops it.** The EMO field survived every gate because `Awe` *is* a valid taxonomy emotion, the checklist's "not bare AWE/INDIGNATION" clause was prose that had never been counted, and the field sits outside every similarity ceiling. **Assign the emotion vocabulary per pair exactly as the discharge mechanism is assigned, ban the mode word from the EMO field, and put a distinct-emotions-per-song floor in `gates.yaml`.** Corollary, and it is the general form: **after banning a token, name the next-lowest unassigned level and measure it *before* the run, not after** — the residue is a pressure, not a word, and it will always find the room you left unlocked.
>
> **"Would this lesson have hurt our best past entry?"** — **No.** It adds a floor on emotional specificity and bans nothing expressive; "Triple Arch Over Me" carries a varied EMO map and would gain from it, not lose. It does not touch INDIGNATION's standing as a mode — only its use as a *substitute for* a specific emotion in a header field.

---

## Failure-ledger write-back

⛔ **NOT WRITTEN.** The write-back is specified "on every shipped/selected piece," and the overall verdict is **REPAIR** — nothing is selected yet. The entry above is a **proposal held for The Scientist**, per the rule that only a human promotes a lesson.

---

## Final recommendation

**REPAIR, then re-QA — with R0 surfaced to The Scientist before anything else.** Ten routed items, of which **R1, R2 and R3 are additive** — they touch no shipped lyric, no prompt and no measured value, and clear 15 of the 20 repair verdicts on their own.

**The three that actually matter are all one sentence each**, and all three sit in the credit layer on the first day the dedication standard is live:
- **R0** — two captions must stop calling a living man the subject of a song. *(HOLD-FOR-HUMAN; non-waivable.)*
- **R4** — one flagship caption must stop claiming a request its prompt does not make.
- **R5** — the run must decide, once, whether *"furiously slapping and popping"* in a STYLE field is a Larry Graham debt, because right now it is answered both ways inside the same run.

⭐ **The pattern across all three is worth naming: every one of this run's serious failures is in the paratext, not the art.** The lyrics are clean on D4, D7, the forbidden anchors, the name-leak sweep and every soundcraft floor. What broke is the layer that *describes* the work — captions, credit blocks, EMO headers, exclude fields, the missing panel. **The run wrote better songs than it wrote paperwork**, and the paperwork is what goes out under her name.

What I want on the record beside the repair list: **the writing in this run is strong, and the L41 experiment worked.** The zero is real, it was achieved under the harder condition, and I could not break it. The reason the verdict is not SHIP is that the run proved its hypothesis and simultaneously demonstrated the hypothesis's limit — the residue relocated to the one level nobody was counting — and the packaging layer (EMO fields, exclude fields, dedications, the missing panel) did not keep pace with the lyrics.

**Two things this run did that I would not want lost in the repair list:** P04 caught a false-credit branch *in its own work*, declined the credit rather than editing the song to earn it, and documented the refusal — that is the DEDICATION STANDARD functioning on its first day. And P01 volunteered the confound that weakens its own evidence. **Both are the behaviour that makes an experimental ledger worth keeping**, and both were unprompted.
