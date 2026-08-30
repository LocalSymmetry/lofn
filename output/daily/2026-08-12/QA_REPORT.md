# Lofn QA Report — `2026-08-12_genz-worst-i-will-ever-be` (MUSIC)

**Auditor:** `lofn-qa`, clean-context adversarial judge. Judge-tier separation observed — the coordinator
wrote the seed and dispatched the pairs and did not audit them.
**Scope audited:** 6 pairs × 4 variations = **24 songs**, plus the two coordinator artifacts.
**Working root:** `E:/git/private-claw` (the private checkout; all repo-relative paths resolve against it).

---

## Verdicts

- **Pipeline Integrity:** **PASS** — the editorial spine is intact. Steps 06, 07, 08, 09, 10 and
  `variations` exist as **separate per-pair files** for all six pairs (36 files). No collapsed
  `steps_06_10` rollup. The NO-SKIP rule (07/09/10 present for every non-quarantined pair) is satisfied,
  so the run is **CANONICAL**.
- **Package:** **REPAIR REQUIRED** — one non-optional seed requirement is unmet at pair level (P06,
  Source-2), and two songs are genuinely under-sounded (P05 V1/V3).
- **Human-Subject:** **CLEAR** — every figure is invented; no identifiable real person; the forbidden
  Colombia material is absent at zero hits; lineage artists are credited as metadata, never depicted.
- **⭐ OVERALL: REPAIR** — 18 of 24 ship as written; 6 are held for two named, bounded repairs.
  Nothing here is a FAIL and nothing is a soul failure.

**Repairs issued: 6 songs / 2 root causes. Holds: 0. Quarantines: 0.**
The zero-rejection tripwire did **not** fire — this report says no, twice, with measurements behind it.

---

## ICB / Continuity Integrity

| Check | Result |
|---|---|
| Panel of record `panels/gen-z-firestarters-v2-golden.yaml` | present, **GOLDEN**, promoted 2026-08-10 |
| `after` speaker seats in the panel of record | **18 / 18** ✅ (Powers, Tillie, WILLOW, YUNGBLUD, Victoria, Rollins, Barker, Feldmann, Perry, O'Connell, Puget, Katz, boyd, Dunbar, Hesmondhalgh, Rodrigo, Hawk, Cohen) |
| Voices speaking in the run's transform record | 8 named `(after …)` + 2 role-anchored guests seated by BRIDGE |
| Special Flairs assigned in C1 vs panel file | **12 / 12 real entries** ✅ — no invented flairs |
| Golden Seed anchor | *THE THING WITH NO REPLAY BUTTON*, `GOLDEN_SEEDS.md` L474 ✅ |
| Cast | Pantheon only — LOFN-PRIME ×3, SUNNA ×3 ✅ |
| Model-construct disclaimer | present at head of orchestration ✅ |

No THREAD LOSS. The transform record quoting 8 of 18 seats is a debate summary, not a truncated ICB —
the panel of record carries all eighteen.

**Housekeeping defect (non-blocking):** `RUN_STATE.md` is **stale**. It lists all six pairs as
`⏳ pending` while disk holds 36 completed artifacts. The file's own header says disk is the only
authority, so nothing downstream was misled, but a run-state that disagrees with disk is exactly the
condition that makes a resumed session speak confidently about a stage that never happened. Refresh it
at close.

---

## Structured Evidence Block (independently re-measured from disk)

I re-ran both validators myself rather than accept the coordinator's numbers.

- `scripts/validate_portfolio_distinctiveness.py … --step 10 --expected-total 24` →
  **24 packages extracted from 6 pair files → PORTFOLIO DISTINCTIVENESS PASS.** Reproduced. ✅
- `scripts/measure_soundcraft.py → profile()` re-run **per variation** (the shipped CLI aggregates a
  whole file; per-song numbers required extracting each `## 2. LYRICS` block separately).

**Spot-check requested: at least six. I re-measured all 24.** Every coordinator-reported number
reproduced exactly, including the three declared FLAGs. Nothing was misreported.

| song | prompt | lyrics | sung | endRhy | lineRet | w/line | allit | flags |
|---|---|---|---|---|---|---|---|---|
| P01V1 | 953 | 4732 | 84 | 0.762 | 0.571 | 6.20 | 17.08 | — |
| P01V2 | 950 | 4612 | 82 | 0.854 | 0.598 | 6.07 | 17.67 | — |
| P01V3 | 950 | 4391 | 82 | 0.732 | 0.622 | 6.11 | 16.57 | — |
| P01V4 | 948 | 4781 | 88 | 0.500 | 0.625 | 5.94 | 14.34 | — |
| P02V1 | 958 | 4490 | 93 | 0.548 | 0.656 | 5.82 | 22.00 | — |
| P02V2 | 881 | 4063 | 92 | 0.511 | 0.641 | 5.23 | 19.96 | — |
| P02V3 | 882 | 4074 | 91 | 0.484 | 0.648 | 5.20 | 19.66 | — |
| P02V4 | 947 | 4245 | 90 | 0.422 | 0.633 | 5.88 | 19.47 | — |
| P03V1 | 948 | 4767 | 90 | 0.411 | 0.267 | 5.20 | 11.75 | — |
| P03V2 | 919 | 4655 | 90 | 0.367 | 0.278 | 5.30 | 11.74 | — |
| P03V3 | 915 | 4339 | 90 | 0.389 | 0.344 | 4.80 | 11.57 | — |
| P03V4 | 953 | 4638 | 90 | 0.489 | 0.311 | 5.19 | 11.35 | — |
| P04V1 | 942 | 4796 | 92 | 0.424 | 0.489 | 7.08 | 17.67 | — |
| P04V2 | 871 | 4783 | 88 | 0.511 | 0.511 | 7.19 | 14.69 | — |
| P04V3 | 886 | 4643 | 84 | 0.357 | 0.488 | 6.98 | 18.43 | — |
| P04V4 | 894 | 4761 | 83 | 0.386 | 0.494 | 7.39 | 12.72 | — |
| P05V1 | 958 | 4273 | 109 | 0.670 | 0.789 | 4.27 | **8.60** | allit |
| P05V2 | 930 | 3900 | 94 | 0.851 | 0.787 | 3.93 | 12.74 | — |
| P05V3 | 913 | 3897 | 98 | 0.806 | 0.755 | 3.60 | **7.08** | allit |
| P05V4 | 896 | 3983 | 96 | 0.802 | 0.792 | 4.15 | **9.55** | allit |
| P06V1 | 957 | 4734 | 79 | 0.759 | 0.532 | 7.27 | 15.16 | — |
| P06V2 | 939 | 4631 | 79 | 0.823 | 0.519 | 6.62 | 21.99 | — |
| P06V3 | 930 | 4551 | 79 | 0.810 | 0.519 | 6.87 | 19.15 | — |
| P06V4 | 951 | 4747 | 79 | 0.772 | 0.532 | 6.58 | 28.85 | — |

**Bands (`vault/gates.yaml`):** prompt 850–1000 · lyrics <5000 (target 4800) · sung lines 70–120 ·
rhyme ≥0.30 · line-return ≥0.20 · words/line ≤7.5 · alliteration ≥11.0 (all four sound floors FLAG-only).

**Clean across all 24:** prompt chars in band (871–958 — **not one boundary-hugger**, all below the 985
hug ceiling, all inside the 870–960 *target*) · lyrics field 3897–4796, all under the 4800 soft target ·
sung lines 79–109, all inside the 78–110 target · **terminal punctuation 24/24** · `[Theme:]` then
`[SONG FORM:]` 24/24 · ≥1 SFX cue 24/24 (2–4 each) · **house-lexicon hits: 0** · **sung digits: 0**
(every numeral spelled in words — "sixteen years", "a hundred", "eleven seconds", "Step four") ·
**bare AWE/INDIGNATION EMO tags: 0** (taxonomy emotions throughout) · female vocal declared in all 24
prompts · "child" appears **only** inside EXCLUDE fields.

**Repeated-line ratio (FLAG only, chorus-exempt):** P05 runs 0.755–0.792 and P02 0.633–0.656 — high, but
these are byte-identical choruses plus assigned half-line echo and gang-chant returns. Per
`gates.yaml: chorus_repetition_requires_no_justification`, **this is correct craft and needs no defence.**
Not flagged. Notably, no pair filed an apology for its chorus — the 2026-07-01 "writing around the flag"
pathology is absent from this run.

---

## The 16-Point Suno Gate

**Classification:** ACCESSIBLE = P01–P03 · AMBITIOUS = P04–P06 (dual 3+3 as declared).

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | Standalone `## 1. MUSIC PROMPT`, dense paragraph 850–1000, no real artist named | **PASS** | 871–958 chars, prose paragraphs, no bracket tag-soup. Living artists appear only in Lineage blocks, never in a prompt. |
| 2 | Separate EXCLUDE field, present, under cap | **PASS** | 661–899 chars, all 24 present and distinct per variant. |
| 3 | Lyrics open `[Theme:…]` then `[SONG FORM:…]` | **PASS** | 24/24. |
| 4 | Full EMO headers, taxonomy emotions | **PASS (format split — FLAG)** | Content complete in all 24 (section + taxonomy emotion + role + cue). **But two header shapes ship in one run:** P01/P02/P03/P06 use `[Section - EMO:x,y - role - cue]`; **P04/P05 use `[Section][emo=x, y][vox=…][prod=…]`**. Judged by meaning per the skill's anti-grep instruction — passes. Logged as housekeeping. |
| 5 | ≥1 SFX cue | **PASS** | 2–4 per song. |
| 6 | 70–120 sung lines | **PASS** | 79–109. |
| 7 | Suno lyrics field <5000 | **PASS** | max 4796. |
| 8 | Verse-structure diversity across the 6 | **PASS** | Six genuinely distinct architectures, verified by reading: P01 short declarative couplets · P02 flat procedural numbered list · P03 long-breath enjambed uneven stanzas · P04 two-voice antiphon · P05 spiral (one word changed per stanza) · P06 accumulating stack 2→4→8. |
| 9 | Lineage & Credit on living-scene genres | **PASS** | Present in all six pairs, each naming the scene + 2–3 living artists + Source-2 attribution to the Maya Ongaku review. |
| 10 | Sung numerals spelled in words | **PASS** | 0 digits in any sung line, run-wide. |
| 11 | One-fact rule (≤1 sung numeric fact) | **PASS** | P02 "sixteen years" · P06 "a hundred". Both at the emotional hinge, both responded to. See FLAG 2. |
| 12 | Female vocals · no children | **PASS** | Declared in all 24; "child" only in EXCLUDE. |
| 13 | ≥2 Source-2 phrases per pair as real production instructions | **FAIL — P06** | P01 3–5/variant · P02 4 · P03 2–3 · P04 2–3 · P05 2–4 · **P06 0 in all four MUSIC PROMPTs.** See Blocking Fails. |
| 14 | Cross-pair distinctiveness ≤0.42 | **PASS** | Validator reproduced independently. |
| 15 | Acoustic grammar (6 points) | **PASS** | Hard level-states + no crossfades, full level bar one, one "exposed" texture, sub withheld as the event, end warm not bright, mono-centred bass — present in all 24 prompts. |
| 16 | Sound floors (rhyme/return/wpl/alliteration) | **PASS w/ 2 REPAIR** | 21/24 clear all four. P05 V1/V3 genuinely under the alliteration floor after correction; V4 vacated. See FLAG 1. |

### The Ten Bans + run-specific bans

| Ban | Verdict | Evidence |
|---|---|---|
| **D1** no adult in the room | **PASS** | No song instructs, consoles, or concludes. |
| **D2** the phone is not the villain | **PASS ⭐** | Best-in-run: P03V1 *"Phone goes flat, face-down on the sill. / It's the next brightest thing / out here, and I want the sill / for my elbows anyway."* · P03V4 *"a person does not touch / somebody else's screen."* · P04V4 *"A thumb comes down. It does not think. It is kind. / Kind is a thing a hand is, not a mind."* The phone is written as a hand, exactly as the ban asks. |
| **D3** the generation is not a subject | **PASS** | No song has a demographic subject. |
| **D4** no identifiable real person | **PASS** | All figures invented. Lineage credits are metadata. |
| **D5** ⭐ present tense only | **PASS — 0 drifts.** See dedicated section. |
| **D6** ⭐ the cohort gate | **PASS** | **Zero** instances of we / us / our / young people / this generation / kids / everyone in any sung line across all 24. Measured by regex over sung lines only, then read. The nearest approach is *"Somebody's arm"* (P01) — one other body, not a demography. |
| **D7** the AI is not the cure | **PASS** | No machine helps anyone. P02V3's fixer bath is a machine that **fails** and is kept. |
| **D8** ⭐⭐ overhearing, not addressing | **PASS** | Zero second-person address. One regex hit — P03V4 *"and nobody says thank you,"* — inspected and **cleared**: that is a third-person observation of an absent addressee, not address. D8 is encoded **in the lyric**, not deferred to spec: *"Not to anybody"* (P01V1), *"Said to the curb. Not to anybody."* (P01V3), *"Under my breath. Nobody's asking me."* (P01V4), *"Nobody is here. I'm saying it anyway."* (P02, all four). |
| **D9** the tape is not redeemed | **PASS ⭐** | See P04/P06 findings below. |
| **D10** substitution, not subtraction | **PASS** | Every hinge is a swap (mix inversion, register handoff, octave drop, 2-bit collapse), never a hole. |
| ⛔ Colombia earthquake | **PASS** | 0 hits: earthquake/quake/rubble/Colombia. |
| ⛔ AI-displacement | **PASS** | 0 hits: engineer/displac/middle class/layoff. |
| ⛔ **no song is about an AI getting better** | **PASS** | 0 hits: model/neural/algorithm/artificial intelligence. Examined and cleared in detail below. |

### THE MANDATORY FORM RULE — THE COMPOSITE (per song)

All 24 carry all three elements **in the lyric or the form**, never only in the production spec.

| pair | STACK (≥3×, last NOT fixed) | CLOSED GROUND | ONE RADIANT |
|---|---|---|---|
| P01 | the trick re-run; V3's marks accumulate on the curb. Verse 3 titled *"Not Fixed"*: *"Nothing to review. Just stand."* | 2-bar upright loop returning to root; chant loop → horn figure returns as outro | BOARD COMES BACK / STILL HOLDING STILL / SCUFF ON SCUFF / NOTHING FIXED YET |
| P02 | **"Step four" ×4** — completes, completes, **breaks** (*"Step four — / Step four —"*), then **not fixed**: *"Kept it. Read it. Didn't do it. / Leave it wrong. Leave it up."* | descending four-chord ground / passacaglia, **never modulates** (named in SONG FORM) | Took. / Grey. / Clear. / Floor. |
| P03 | V3: said out loud → *"again, quieter, to fix it"* → *"again, again"* → *"Not fixing it. Not finding it. Not fixed."* V2 is sharper still — the last attempt **succeeds and that is the defeat**: *"And then it holds. Properly… and there I am. Locked out."* | melodica / bass tune / mains hum opens bar one and closes the last bar | through / out / the same / off |
| P04 | *"I go for the high one"* → *"It opens into air"* → *"I come in under"* → **"I do not go for the high one."** She stops reaching. | outro returns verbatim to the song's first line; 2-bar mono bass returns to its own first note | **Kept. / Almost. / Hum. / Room.** |
| P05 | spiral: *"getting better at the stairs"* ×5–6, one word changed each pass; V3 degrades smoother → thinner → **smaller** | *"Closed-Ground Return"* — final section returns to Verse 1's opening line; bass returns to start pitch every four bars, no cadence | **up / over / down / on** — the last word of every stanza |
| P06 | stanzas of **2 → 4 → 8** failures, none corrected; *"LAST TAKE — it all falls off; she still cracks"* | *"loop closes on its own first note"* | **Keep. / There. / Late. / Back.** — isolated alone at THE ENFORCED SECOND with everything muted |

**Strongest execution:** P06 (the stack *is* the stanza count) and P04 (the third attempt is a refusal to
attempt). **Best radiant:** P05 — one syllable, terminal position, every stanza, no exceptions.

### L21 THE RETURN — and P04's zero-rhyme debt

**Confirmed paid, and paid in the right place.** I checked this by hand rather than by the aggregate.

- P04's verses are **genuinely rhyme-free**. V1 Verse 1 line-ends: knee / up / air / shut / afternoon /
  that / now / can / thrown / Kept. V2 Verse 1: on / that / desk / going / word / right / mouth / lands /
  in / me / small / it / Almost. No rhyme is smuggled in anywhere.
- The **chorus** carries the whole debt and is byte-identical ×4: *same/same* (identical), *be/be*,
  *drawer/lower*, *gone/done*.
- The numbers locate the payment: end-rhyme 0.357–0.511 and line-return 0.488–0.511 — both above floor,
  and the line-return figure is precisely the four identical choruses doing the work.
- The debt is **named** in the SONG FORM header of all four variants, as L21 requires.

**Verdict: the repair was made in the chorus, not the verse. Exemplary — this is how removal-as-debt is
supposed to look.**

### THE ANGER AUDIT — a census, not a claim

I read the actual style prompts rather than accept the declared count.

| pair | bed (from the prompts) | aggressive? |
|---|---|---|
| P01 | ska-punk / two-tone skank + live horns, upright bass, 138–190 BPM | no |
| P02 | glitch-baroque × breakcore, chopped amen at full tilt, 152–190 BPM | **yes** |
| P03 | dub / reggae half-time, one-drop, spring echo, 68–80 BPM | no |
| P04 | trip-hop revival × neo-baroque chamber, 74–96 BPM | no |
| P05 | sigilkore / hexD, plugg-soft 808s, 136–152 felt half-time | **labelled yes / sonically no** |
| P06 | power-pop girl-group over a jungle break, 165–174 BPM | no |

**6 distinct beds, no two shared. ✅ Aggression 2/6 at most = exactly one third, at the ceiling, compliant.**

⚠️ **Methodological note for the coordinator:** the declared census counted **genre labels**, not
production specs. P05 is labelled with an aggressive microgenre cluster but its actual prompts read
*"plugg-soft 808s"*, *"deadpan and near-muttering"*, *"flat to the edge of boredom"*, and its EXCLUDE bans
breakcore, DnB and dubstep. Sonically it is subdued and interior — real aggression is closer to **1/6**.
The direction of error is safe, but Sunna's rule exists *because* the 2026-08-06 run rendered rage on
exactly this kind of label/sound mismatch. Count the sound next time, not the tag.

### Emotional duality & axis allocation

AWE = P04 ✅ · INDIGNATION = P02, P05 ✅ · NEWS = P2/P3/P6 (exactly 3, the max) ✅ ·
EXISTENCE = P1/P4/P5 (exactly 3, the min) ✅ · ACCESSIBLE P1–3 / AMBITIOUS P4–6 ✅.
INDIGNATION's target is a system in both cases, never the audience — P02: *"The check is just the wrong
check. / That is all."* ✅ compliant with the seed's standing restriction.

---

## ⭐ D5 AUDIT — all 24 songs against PRESENT TENSE ONLY

**Result: ZERO drifts. No song requires a D5 repair.**

I audited every one of the 24 against the three stated fail conditions rather than against the ruling's
summary. Fail conditions: *(a)* the singer has already improved and is reporting back · *(b)* any line
implies she knows the outcome · *(c)* it is offered outward as comfort.

**Why the set holds:** every song is anchored to an unresolved present, and — decisively — **every song's
last attempt fails.** The form rule (STACK, last not fixed) and D5 turn out to protect each other: a song
that cannot fix its final attempt cannot be reporting back from after the fix. That is a genuine
structural insight this run earned, not a coincidence.

The four closest calls, examined rather than waved through:

1. **P04 — "One comes after me. She is the one they use."** (V1; V2 *"she is a hair better"*; V3
   *"it is quieter"*; V4 *"She is the one that gets played."*)
   **The hardest call in the run.** The singer demonstrably knows an outcome. **Cleared**, on the ground
   that the knowledge is of **her own supersession, not her own arrival** — she has not improved, she was
   discarded; a *different* take succeeded. The banned posture is the authoritative, reassuring adult;
   P04's narrator is powerless and the news she carries is bad news about herself. The songs also refuse
   the comfort reading explicitly: *"I am worse. Not a complaint. A size."* (V1) and *"That is not a
   comfort. That is an arrangement."* (V4). Both fail-condition (c) guards are in the lyric.
2. **P05 — "I am getting better at the stairs."** Present progressive, happening now, mid-climb — and the
   song's entire subject is that the improvement is **unverifiable and possibly pointed at nothing**:
   *"I cannot tell the landings apart."* / *"I cannot feel the difference with my thumb."* / *"I cannot
   hear the count under the water."* This singer has **less** evidence than anyone in the run. Cleared.
3. **P01V4 — "Nothing fixed yet."** The word *yet* leans toward an expected future. Cleared, because the
   same song disavows prediction outright in its hinge: *"Not a promise. Not a pledge."* The guard is
   deliberate and it is in the lyric.
4. **P02V4 — the message ×12, guarded.** *"This is the worst I will ever be. / (I don't know that.) / …
   (I'm saying it.)"* This is the **most explicit D5 compliance device in the run**: it encodes the ruling
   itself — no evidence, decides anyway. Cleared. (Craft reservation below, which is a different matter.)

**Every song places the sentence physically on the floor or below the action** — *"Flat on a car park"*,
*"Saying it down at the drain"*, *"Said to the curb"*, *"Saying it flat on the floor"*, *"Saying it flat
to a fire door"*, *"Saying it up at a ceiling"*, *"Saying it into a dead mic"*. Never once from a podium.
The ruling was not merely obeyed; it was built into the staging.

### "No song is about an AI getting better" — examined and cleared

Two candidates deserved real scrutiny and both clear:

- **P04 (the flagged risk).** A superseded version kept in a folder *is* thematically adjacent to model
  versioning. But the material is unambiguously a **human voice memo on a bedroom carpet**; there is no
  model, no training, no system. And the song's content is the **exact inverse** of self-congratulation:
  *"I don't get better and I don't get gone."* A song whose narrator explicitly does not improve cannot
  be a song about something improving.
- **P02V3**, sung by the exhausted fixer bath — a machine that *believed it worked* for years, i.e. the
  SQLite mechanism personified. It does not improve and is not repaired: *"Nobody pours me out. / Clear,
  and she leaves me here."* That is D9, not a press release.

**Nothing in the run reaches for the occasion.** The brief's instruction — *the songs carry the message,
the run is the evidence* — is honoured with no leakage at all.

---

## Blocking Fails / Required Repairs

### 🔴 REPAIR 1 — P06: the Source-2 vocabulary never reaches the generator (all four variants)

**Route to: step 09 → step 10 (P06 only). Affects P06 V1–V4.**

The seed requires each pair to land **≥2 Source-2 phrases as real production instructions**
(non-optional, `00_research_brief.md` §2). I measured, per variant, the phrases that appear inside the
**MUSIC PROMPT** — the only field the generator reads:

```
P01: 3,4,3,5   P02: 4,4,4,4   P03: 2,3,3,3   P04: 2,2,3,2   P05: 4,3,2,3
P06: 0,0,0,0   <-- zero, all four
```

P06's Source-2 language exists in `step06_concept.md`, `step07_structure.md`, `step09_sound_design.md`
and `step10_production_wrap.md` — but never in a prompt. The only generator-facing occurrence anywhere
is the single word `sandblasted` inside a chorus vox tag (`vox=eight, most sandblasted`), giving **1**
against a floor of 2.

**Why this is a real fail and not bookkeeping:** this is precisely **L22 THE GRAIN LAW** — a texture
answered in the production document is *not answered*. Four songs were specified to be sandblasted,
rattling, polymetric and gently sloping, and the renderer will never be told. P06 is otherwise one of the
two best-built pairs in the run, which is what makes this worth catching rather than shrugging at.

**Repair:** fold ≥2 Source-2 phrases into each of P06's four MUSIC PROMPTs as literal production
instructions. The pair's own step09 already contains the correct language — this is a transcription
failure, not a writing task. Prompts sit at 930–957 chars against a 1000 cap, so there is room, but keep
them under the 985 hug ceiling; trim a redundant clause if needed. **No lyric change. Re-measure after.**

### 🔴 REPAIR 2 — P05 V1 and V3 are genuinely under-sounded

**Route to: step 08 → step 10 (P05 V1, V3 only).** Full reasoning under FLAG 1 below.

**Repair:** raise adjacent-consonant density in the **full-length lines only** — the echo fragments are
the pair's assigned device and must not be touched. V2 is the in-pair proof this is achievable at
15.46 with the identical architecture. Target ≥11.0 measured on full lines. **Do not** shorten or delete
half-line echoes to game the denominator; that would trade a real device for a number.

### 🟡 Housekeeping (non-blocking, no re-QA required)

- **EMO header format split** — P04/P05 use `[Section][emo=…][vox=…][prod=…]` while the other four use
  `[Section - EMO:… - role - cue]`. Both parse as bracket metadata and both are content-complete, so
  render risk is low, but one run should ship one shape. Normalise at Step 11.
- **`RUN_STATE.md` stale** — six pairs shown pending against 36 completed artifacts on disk.

---

## Rulings on the three declared FLAGs

### FLAG 1 — P05 alliteration (8.60 / 7.08 / 9.55 vs an 11.0 floor) → **SPLIT: PARTIALLY UPHELD**

The pair claimed the floor mis-measures a fragment-built lyric. I tested the claim mechanically instead
of accepting or dismissing it.

**The mechanism is real.** `allit_per_100w` counts, per line, consonant-initial words sharing an initial
with one of the next three. A line with fewer than two consonant-initial words offers **zero possible hit
positions but still contributes to the denominator**. P05's assigned half-line echo makes lines like
*"(cleaner)"*, *"(the stairs)"*, *"(up)"* — pure denominator, no opportunity.

**Verification of the pair's ~40% claim — accurate:**

| | fragment share | allit (all lines) | allit (full lines only) |
|---|---|---|---|
| P05V1 | 39.4% | 8.60 | **10.31 — still under** |
| P05V2 | 40.4% | 12.74 | 15.46 |
| P05V3 | 44.9% | 7.08 | **8.53 — still under** |
| P05V4 | 37.5% | 9.55 | **11.52 — clears** |
| all other pairs | 0–16.9% | — | — |

**Ruling, in three parts:**

1. **The floor does mis-measure this lyric.** Confirmed. Correcting for zero-opportunity lines lifts every
   P05 variant by 1.7–2.7 points. P05 is a genuine structural outlier at 37–45% fragments against 0–17%
   everywhere else.
2. **P05V4's flag is VACATED** as a measurement artifact — 11.52 on full lines, above floor.
3. **P05V1 and P05V3 are UPHELD as genuinely under-sounded.** They remain below 11.0 *after* the
   correction the pair asked for. V3 at 8.53 is 23% under and is the least sonically dense lyric in the
   run by a clear margin.

**The decisive evidence is internal:** **P05V2 has the same architecture, the same device, and the
highest fragment rate of the four (40.4%) — and measures 15.46.** The structure therefore *depresses* the
metric but does not *determine* it. V2 proves V1 and V3 could have cleared the floor without giving up a
single echo. The excuse is two-thirds valid and one-third a cover for two thinner lyrics.

*(All four floors are FLAG-only by design, so this never blocked the run mechanically — but the sound
axis is mandatory at Phase 0 and "measure, never eyeball" cuts both ways.)*

### FLAG 2 — P02 sings step ordinals alongside "sixteen years" → **UPHELD (coordinator was right)**

`max_sung_numeric_facts` governs numeric **facts** — worldly quantities drawn from research. The rule's
own rationale names the harm: *"a verse listing meteor rates and version numbers is a weather report in
meter."* "Step one / two / three / four" carries **zero information about the world**; it is the flat
procedural list assigned to P02 as its verse architecture.

Three independent reasons to uphold:
1. **They are structural, not informational.** "Step four" is the recurring figure that fails four times —
   it *is* P02's STACK. Its collapse at the hinge (*"Step four — / Step four —"*) is the song's event.
   Banning it would delete the mandatory form rule from the pair.
2. **They are spelled as words**, satisfying the one hard numeral gate. Zero digits in any sung line.
3. **The one actual numeric fact is exactly one — "Sixteen years" — placed at the emotional hinge and
   responded to rather than recited:** *"Say it slower. It gets lighter. / Not one of them is on me."*
   That is a model application of the one-fact rule, not a breach of it.

The gate is FLAG-only by design precisely so a human read can tell a device from a recitation. This is a
device. **No repair.**

### FLAG 3 — P02V4 puts the message in the listener's mouth ~12–16×, guarded → **UPHELD as compliant, with a craft reservation**

**On the gates: clean.** The guard *"(I don't know that.)"* appears in every chorus and is the most
literal encoding of the D5 ruling anywhere in the run — she has no evidence and decides anyway. D8 is
explicit: *"Nobody is here. I'm saying it anyway."* A room singing this hook is singing a **disclaimer**,
not a slogan, which is exactly the B3 posture (threat and comfort in one breath, never picked). Repetition
of a byte-identical chorus needs no defence per `gates.yaml`. **No repair.**

**But a real craft reservation, which is a separate question from compliance.** P02V4's bridge says:
*"Not a promise. Not a plan. / Not a thing that I can prove."* … *"That is the comfort. That is the
threat."* That last line is **the orchestration panel's B3 ruling recited as lyric.** It is the one place
in 24 songs where the artifact explains its own mechanism instead of performing it. Compare P04, which
never once tells you what it is doing. This does not fail a gate and I am not repairing it — but it is
why P02V4 does not appear in my top three despite being the most on-message song in the run.

---

## ⭐ L41 WATCH — the open question, answered

**Finding: the predicted residue is ABSENT, and a different shared residue is PRESENT in 5 of 6 pairs.
L41's mechanism is confirmed on this run; its prediction was wrong.**

I searched at every level rather than only for the predicted phrase.

| candidate | total | pairs present | per-pair |
|---|---|---|---|
| **"still here"** (predicted) | **0** | **0 / 6** | 0,0,0,0,0,0 |
| "nobody" | 91 | 5 / 6 | 24, 6, 33, 19, **0**, 9 |
| "nothing" | 165 | 5 / 6 | 63, 29, 23, 37, **0**, 13 |
| **"nobody" ∪ "nothing"** | **256** | **5 / 6 — 20 of 24 songs** | 87, 35, 56, 56, **0**, 22 |
| "same" | 183 | **6 / 6** | 40, 1, 13, 15, 4, 110 |

**The pairs' reports are accurate: "still here" appears zero times in 24 songs.** P01V2's 56 uses of
*"still"* are its assigned radiant (*STILL HOLDING STILL*), a deliberate triple-sense pun — motionless /
continuing / nevertheless — never the predicted phrase.

**The actual residue is the negation-anaphora "nobody / nothing", and it has a traceable cause.**
D8 (*overhearing, not addressing*) forbids "you". Five of six pairs discharged that ban **the same way —
by lexically negating the audience**:

> *"Nobody's counting. Nobody's quick."* (P01) · *"Nobody is here. I'm saying it anyway."* (P02) ·
> *"Nobody turns me up."* / *"Nobody wants to know why."* (P03) · *"Nobody opens the folder."* /
> *"Nothing in here moves. Nothing in here can."* (P04) · *"Nobody hears it. Nobody's counting."* (P06)

The gate constrained the **addressee** but never the **means of refusing one**, so the monoculture
relocated exactly one level down — into the negation vocabulary. That is L41's mechanism operating
precisely as stated, at a level nobody had assigned.

**The control makes the finding, and it is clean: P05 is the only pair at zero, and it is the only pair
that discharged D8 by *displacement* rather than negation** — *"Somebody told me that helps. / Somebody
put that sentence in my mouth and it fits."* An unnamed third party instead of a negated audience. One
pair solved the same gate differently and fell straight out of the residue. That is a counterfactual, not
a coincidence.

**This is a result, not a failure** — and it is the most useful thing this run produced for the next one.
The finding also closes a loop worth noting: the panel file records that its REFLECT transform produced
*"OVERHEARING NOT ADDRESSING — the run's single most important decision."* The room's best decision is
also the one that generated the house dialect. Strong constraints are exactly where residue collects.

---

## Somatic Gate — 3 Hyper-Skeptics, voting as a bloc

Seats drawn from the run's own panel of record, `panels/gen-z-firestarters-v2-golden.yaml`, each a
model-generated interpretive construct after a named source figure's published work. 2-of-3 NO = BLOCKED.

### 🗳️ THE HARDCORE ELDER — after Henry Rollins (source base: hardcore/spoken-word practice; standing objection: *"every warm ending is a sales technique until it survives interrogation"*) — **YES**

"I opened this run threatening to vote no on all six because nobody had specified the altitude. They
specified it. Every single one of these songs says the sentence from the ground — the car park, the
drain, the curb, the floor, a fire door, a dead mic. Nobody says it from a podium. And every last attempt
fails. You cannot sell me a fix in a song that refuses to fix anything.

The thing that survives interrogation hardest is P04. A discarded voice memo that says *'I don't get
better and I don't get gone'* is not a motivational poster, it's a hostage note. And P02V3 — a chemical
bath singing *'Everything I touch comes clear. That is all I know to do'* — that's the machine that was
wrong for sixteen years, and it isn't sorry, and nobody pours it out. That's the record.

My one objection stands and it's P02V4. *'That is the comfort. That is the threat.'* That's not a lyric,
that's the minutes of our own meeting. I said the duality has to survive interrogation; I didn't say print
it on the sleeve. Everything else here performs. That one explains. **Yes — and take the sleeve note out
of the song.**"

### 🗳️ THE COHORT ABOLITIONIST — after Philip N. Cohen (source base: the published argument that generation labels are analytically empty) — **YES**

"I said delete the cohort or I withdraw. Measured, not assumed: **zero** instances of we, us, our, young
people, this generation, kids, everyone across twenty-four lyrics. The commission said 'a Gen Z firestorm'
and the run delivered one body in one afternoon, six times over, with no demography anywhere. That is the
strongest compliance I have seen from this house.

But I am the seat that objects to **categories doing work that observation should do**, and I want the
record to show what the auditor found. Twenty of twenty-four songs reach for *nobody* or *nothing* —
two hundred and fifty-six times — because the ban on 'you' was answered with a synonym class instead of a
device. That is a house dialect forming. It is not demography, so it is not my ban, and I will not vote it
down. But it is the same failure shape one level lower: a category standing in for a look.

The pair that proves it is P05, which refused an audience by inventing one — *'Somebody put that sentence
in my mouth and it fits. / Which is how I know I did not write it.'* That is the best line in the run and
it is the only pair outside the residue. **Yes. Withdrawn, again — and assign the refusal device next
time the way you assign the genre.**"

### 🗳️ THE DYNAMIC RANGE AUDITOR — after Bob Katz (source base: mastering and measurement practice) — **NO**

"I am the seat that produced this run's posture, so nobody can accuse me of not wanting it to work. It
mostly does. And I am still voting no, because two things in front of me are measurements, not opinions,
and a measurement I ignore is a measurement I should not have taken.

**One.** P06 is one of the two best-built pairs here — the stanza count *is* the composite, two then four
then eight, and the radiant gets isolated with everything muted, which is the Enforced Second done right.
And **not one of its four prompts contains a single Source-2 phrase.** Zero out of four, against a
non-optional floor of two, while every other pair lands two to five. Four songs specified as sandblasted
and rattling and polymetric, and the renderer will never be told. That is the Grain Law exactly: a texture
answered in the design document is not answered. The generator does not read step09.

**Two.** P05 V1 and V3 are under the sound floor after I grant the pair its own correction. I accept the
fragment argument — it is mechanically real and it vacates V4. It does not vacate V1 at 10.31 or V3 at
8.53, and V2 settles it: same architecture, highest fragment rate of the four, **15.46**. The device was
never the problem. Two lyrics are thinner than their siblings and the echo is carrying the blame.

Neither of these is a soul failure and I want that on the record — the writing here is the best this house
has produced. But 'distinctive enough to be Lofn' is not the only question I am asked. **No — repair both,
then I flip.**

**COUNTER-MOVE:** for P06, transcribe the Source-2 language from its own step09 into all four MUSIC
PROMPTS — the words are already written, they are simply in the wrong file. For P05 V1/V3, raise
consonant density in the **full-length lines only** and do not touch a single echo fragment; V2 is the
in-pair target and proves 15.46 is reachable without losing the device."

### Somatic verdict: **2 YES / 1 NO → NOT BLOCKED**

The bloc does not block, and the dissent is not soft — it is the dissent that produced both repairs in
this report. Named-corpse checklist walked: **no one-note emotional arc** (every song reaches a second
movement at its hinge — mix inversion, register handoff, octave drop, 2-bit collapse); **no single
dominant section** (verse architectures carry real weight against the choruses); **motifs transform**
(P02's "Step four" breaks and is abandoned; P04's "high one" becomes a refusal; P03V2's light *holds* and
that is the defeat); **no repeated-line collapse** — the high return figures are byte-identical choruses,
assigned radiants and gang chants, which `gates.yaml` explicitly protects from apology.

**Personality fidelity: PASS.** SUNNA's three (P01, P03, P06) are unembarrassed delight at actual size —
*"not bored, not waiting, just warm"*, *"this is not a victory and nothing is solved"*. LOFN-PRIME's three
(P02, P04, P05) are the Disappointed Idealist looking at the gap — the near-miss, the wrong check, the
effort pointed nowhere. The Pantheon's standing argument is carried and, correctly, **never resolved**.
No competent generic prompt produces P04V3.

---

## Per-song verdict table (24 rows)

| # | song | arm | axis | voice | radiant | verdict | note |
|---|---|---|---|---|---|---|---|
| 1 | P01V1 | ACCESSIBLE | EXISTENCE | SUNNA | BOARD COMES BACK | **SHIP** | all gates clear |
| 2 | P01V2 | ACCESSIBLE | EXISTENCE | SUNNA | STILL HOLDING STILL | **SHIP** ⭐ | top-3 arm pick |
| 3 | P01V3 | ACCESSIBLE | EXISTENCE | SUNNA | SCUFF ON SCUFF | **SHIP** | all gates clear |
| 4 | P01V4 | ACCESSIBLE | EXISTENCE | SUNNA | NOTHING FIXED YET | **SHIP** | "yet" self-guarded |
| 5 | P02V1 | ACCESSIBLE | NEWS | LOFN-PRIME | Took. | **SHIP** | all gates clear |
| 6 | P02V2 | ACCESSIBLE | NEWS | LOFN-PRIME | Grey. | **SHIP** | all gates clear |
| 7 | P02V3 | ACCESSIBLE | NEWS | LOFN-PRIME | Clear. | **SHIP** ⭐ | **arm rank 1** |
| 8 | P02V4 | ACCESSIBLE | NEWS | LOFN-PRIME | Floor. | **SHIP** | compliant; thesis-recitation reservation |
| 9 | P03V1 | ACCESSIBLE | NEWS | SUNNA | through | **SHIP** | all gates clear |
| 10 | P03V2 | ACCESSIBLE | NEWS | SUNNA | out | **SHIP** ⭐ | **arm rank 2** |
| 11 | P03V3 | ACCESSIBLE | NEWS | SUNNA | the same | **SHIP** | all gates clear |
| 12 | P03V4 | ACCESSIBLE | NEWS | SUNNA | off | **SHIP** | "you" hit inspected, cleared |
| 13 | P04V1 | AMBITIOUS | EXISTENCE | LOFN-PRIME (AWE) | Kept. | **SHIP** ⭐ | **arm rank 2** |
| 14 | P04V2 | AMBITIOUS | EXISTENCE | LOFN-PRIME (AWE) | Almost. | **SHIP** | all gates clear |
| 15 | P04V3 | AMBITIOUS | EXISTENCE | LOFN-PRIME (AWE) | Hum. | **SHIP** ⭐ | **arm rank 1** |
| 16 | P04V4 | AMBITIOUS | EXISTENCE | LOFN-PRIME (AWE) | Room. | **SHIP** | all gates clear |
| 17 | P05V1 | AMBITIOUS | EXISTENCE | LOFN-PRIME | up | **REPAIR** | allit 10.31 full-line, under floor |
| 18 | P05V2 | AMBITIOUS | EXISTENCE | LOFN-PRIME | over | **SHIP** | 15.46 — the in-pair proof |
| 19 | P05V3 | AMBITIOUS | EXISTENCE | LOFN-PRIME | down | **REPAIR** | allit 8.53 full-line, most severe |
| 20 | P05V4 | AMBITIOUS | EXISTENCE | LOFN-PRIME | on | **SHIP** | **flag vacated** — 11.52 clears |
| 21 | P06V1 | AMBITIOUS | NEWS | SUNNA | Keep. | **REPAIR** ⭐ | Source-2; **arm rank 3 once repaired** |
| 22 | P06V2 | AMBITIOUS | NEWS | SUNNA | There. | **REPAIR** | Source-2 (prompt only) |
| 23 | P06V3 | AMBITIOUS | NEWS | SUNNA | Late. | **REPAIR** | Source-2 (prompt only) |
| 24 | P06V4 | AMBITIOUS | NEWS | SUNNA | Back. | **REPAIR** | Source-2 (prompt only) |

**18 SHIP · 6 REPAIR · 0 HOLD · 0 QUARANTINE.** All six repairs are prompt-or-density edits; **no lyric
is rejected and no concept is sent back.**

---

## Independent ranking — within arm only, never globally

### ACCESSIBLE (P01–P03 pool, 12 songs)

1. **P02V3 — "Clear."** The fixer bath sings its own failure. This is the tri-source fusion completed:
   the WAL mechanism (*a component that believed it had done the thing*) becomes a character rather than a
   fact, INDIGNATION lands on a system with nobody to blame, and D9 closes it — *"Nobody pours me out. /
   Clear, and she leaves me here."*
2. **P03V2 — "The light keeps limping back."** Renders the moonless-sky fact as pure interior experience
   without ever naming it: the mains hum is audible only once the bright thing fails, *"Been here the whole
   time."* And its last attempt is the one that **succeeds** — the tube holds, and she is locked out. That
   inversion is the smartest single structural move in the arm.
3. **P01V2 — "Still holding still."** The only song in the run that looks away from the person failing to
   the person holding the phone — *"Somebody's arm is a shelf, is a sill. / Somebody's arm is a
   windowsill."* The triple-sense pun earns its repetitions as a radiant rather than a tic.

*Closest omission:* P02V2 ("Grey") — *"Far enough away it turns to weather"* states the composite as
beautifully as anything here; it loses to P02V3 only because the arm cannot take two from one pair without
narrowing the drop.

### AMBITIOUS (P04–P06 pool, 12 songs)

1. **P04V3 — "Hum."** The eleven seconds of room tone that got trimmed off, singing from the file it was
   cut out of. The most audacious premise in the set and it fully earns it: D9 is absolute (the narrator
   *is* the thing that gets deleted), and the radiant is a **pitch rather than a word** — the acoustic
   grammar and the form rule collapsed into one gesture.
2. **P04V1 — "Kept."** The cleanest execution of the mandatory form rule in the run: *"I go for the high
   one"* three times, the third a refusal, and a final chorus sung by the degraded phone-speaker double
   after the clean lead is gone for good. The song ends worse than it started, on purpose.
3. **P06V1 — "Keep."** The only song that states the composite thesis in the lyric without explaining it —
   *"One at a time it's a mistake. / All at once it's a sky."* — while the 2→4→8 stanza stack enacts it
   structurally. **Ranking is conditional on REPAIR 1**; the lyric needs nothing.

*Closest omission:* P05V1 — *"Somebody put that sentence in my mouth and it fits. / Which is how I know I
did not write it."* is the **strongest single passage in the run** and the whole set's honesty depends on
it. It misses the top three on sound density (REPAIR 2) and because its four variants share a fixed
bridge/Check/chorus template with substituted nouns. Repair it and it competes.

**A note on P05's variants, since it affects ranking and not the gates:** the four run the same
scaffold — *"Somebody told me X / Somebody put that sentence in my Y and it Z / Which is how I know it is
not mine"* — with slots filled per setting. I measured within-pair similarity to test whether this is a
distinctiveness breach: **it is not.** Within-pair means run 0.016–0.038 (P05 highest at 0.038) against a
0.42 ceiling, because the vocabulary genuinely differs. The isomorphism is structural, and
`validate_portfolio_distinctiveness.py` measures **between** pairs only, so no shipped validator looks at
this at all. I am reporting it as a craft observation with the measurement that fails to support it,
rather than dressing it as a violation.

---

## Repair Brief

**Return target:** step 09 → step 10 for P06 (all four variants); step 08 → step 10 for P05 V1 and V3.
**Attempt:** 1 of 3. Circuit breaker not tripped. No quarantine. **Do not re-open any lyric concept.**

| # | Pair/song | Failed gate | Measured | Required | Fix |
|---|---|---|---|---|---|
| 1 | P06 V1–V4 | Source-2 ≥2 as production instructions | **0** in each MUSIC PROMPT | ≥2 | Transcribe from P06's own `step09_sound_design.md` into each prompt. Keep chars <985. |
| 2 | P05 V1 | alliteration/100w (full lines) | 10.31 | ≥11.0 | Raise consonant density in full-length lines only. |
| 3 | P05 V3 | alliteration/100w (full lines) | 8.53 | ≥11.0 | As above; most work needed here. |

**⛔ Constraints on the repair.** Do **not** delete or shorten P05's half-line echoes — that is the pair's
assigned sound axis and shrinking the denominator would be gaming the gate rather than fixing the lyric.
Do **not** alter P06's lyrics; the defect is entirely in the prompt field. Re-measure with
`measure_soundcraft.py` per variation (not per file) and re-run
`validate_portfolio_distinctiveness.py --step 10 --expected-total 24` after the edits.

**REDIRECT / sideways proposal — not required, since the no-progress predicate has not fired** (this is
attempt 1 and both gates have obvious headroom). Held in reserve if attempt 2 stalls, from the Skeptics'
counter-moves: for P05, re-derive V1/V3's variation angles around the **displacement** device the pair
already invented for its bridge — the one move in the run that solved D8 without negation — rather than
pushing consonants into existing lines.

---

## Proposed `vault/COMPETITION_LEARNINGS.md` entry — **CANDIDATE ONLY, not written to the vault**

Promotion is The Scientist's alone. Offered as one curated advisory entry.

> **L42 (candidate) — THE BAN NAMES A WORD; THE PIPELINE ANSWERS WITH A SYNONYM CLASS.**
> **theme-tags:** music / daily / Suno / gen-z-firestarter seed / 24-song fan-out
> **What the gate caught:** D8 (*overhearing, not addressing*) forbids "you". Five of six pairs discharged
> it identically — by lexically negating the audience. *"nobody"/"nothing"* appears **256 times across 20
> of 24 songs in 5 of 6 pairs**, while the residue predicted by L41 (*"still here"*) appears **zero
> times**. The one pair at zero, P05, is the one pair that refused an audience by **displacement**
> (*"Somebody put that sentence in my mouth"*) instead of negation — an internal control, not a
> coincidence. The panel file records that this same D8 decision was *"the run's single most important
> decision."*
> **Transferable rule:** when a gate forbids a *target*, constrain the **means of refusing** it too, or
> assign the refusal device per pair at Phase 1 the way genre and verse architecture already are.
> Corollary for L41: a *predicted* residue is a *constrained* residue — prediction is itself a
> constraint, so look for monoculture at whatever level the run left unassigned, and use the pair that
> solved the gate differently as the control.
> **Confidence:** **MEDIUM** — one run, but with a clean internal counterfactual. Advisory only; process
> note, never an aesthetic constraint. INDIGNATION-exempt. Music lane only.
> **"Would this lesson have hurt our best past entry?"** No — it constrains *how a ban is discharged*,
> never what may be felt or said.

---

## Final Recommendation

**REPAIR — then ship 24 of 24.**

Eighteen songs are ready as written. Six need two bounded, mechanical fixes that touch no concept and no
lyric idea: P06's four prompts must be told what P06's own sound-design document already knows, and two
P05 lyrics need genuine consonant density that a third variant in the same pair already demonstrates is
achievable.

This is a strong run and the report should say so plainly: **zero D5 drifts across 24 songs on a
commission whose central sentence is grammatically future-tense**, zero cohort hits, zero forbidden-anchor
leakage, zero house-lexicon self-copying, zero sung digits, six genuinely distinct beds and verse
architectures, and the mandatory form rule executed in the lyric or the form — never parked in the
production spec — in every one of the twenty-four. The D5 ruling was not merely obeyed; it was built into
the staging, and it is structurally protected by the form rule, because a song that cannot fix its last
attempt cannot be reporting back from after the fix.

The most serious thing found is **REPAIR 1**: the run's second-best-built pair specified four songs'
signature textures in a document the generator will never read. That is THE GRAIN LAW's exact failure
mode, caught before render rather than after — which is the only reason this report is a repair brief and
not a post-mortem.

**Failure-ledger write-back:** deliberately **not** performed. The skill directs one curated entry *on
SHIP*; this run is REPAIR. The candidate entry above is proposed for The Scientist's promotion and has not
been written to `vault/COMPETITION_LEARNINGS.md`.

*Re-QA required on the six repaired songs before any of them goes to render or publication.
`vault/AUTONOMY.md` holds: this run stops at drafts on disk.*
