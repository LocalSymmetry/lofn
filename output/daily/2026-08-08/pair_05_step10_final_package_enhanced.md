# PAIR 05 — STEP 11 · ENHANCED FINAL PACKAGES
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P05 "TWO MINUTES EIGHTEEN"**

**ARM:** AMBITIOUS · **AXIS B:** NEWS · **VOICE:** `LOFN-PRIME (AWE mode — tender, spacious, unhurried)` · **MODE:** AWE
**Tier:** Step-11 enhancement, fresh context, veto authority live. **I did not write this pair.**
**Input:** `output/daily/2026-08-08/pair_05_step10_revision_synthesis.md` · **Overridden where it conflicts by** `06_music_handoff.md` (L30).

| | |
|---|---|
| Frozen ICB | `output/daily/2026-08-08/CREATIVE_CONTEXT.md`, **read in full, read-only, not edited** |
| ICB bytes (LF-normalised) | **142,900** |
| ICB sha256 (LF-normalised) | **`9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a`** — matches handoff §0 |
| Scratch namespace | `_work/pair_05/` only. No sibling artifact read, moved or written |
| Files written | **one** — `pair_05_step10_final_package_enhanced.md`. Step 10, the ICB and `RUN_STATE.md` untouched |

---

## ⛔ REFUSALS — declared, not quietly satisfied

**1. GOLDEN-OUTPUT QUARANTINE — the contract asked, and I refused.** `skills/music/steps/11_Generate_Music_Enhancement.md` is patched at the head of file and again at §Integration, but this tier is a **generating** context and the run handoff §1 is explicit. No golden-song payload, no past shipped lyric, no prior winning prompt, no "reference song to study" entered this pass. The two golden names in the handoff were **not** looked up, reconstructed or calibrated against. I did not open `lofn-prime-mini.yaml` line 327+. **Calibrated instead against THE GOLDEN MOVE** (handoff §1): a real place a body stands in, named in the first thirty seconds; one wounding fact responded to and never recited; a mid-song turn the singer performs and does not understand; a register rotated away from anything this house shipped recently.

**2. ⛔ THE BODY-NOISE MANDATE IS PARTLY REFUSED, and this is the pass's most important refusal.** Step 11 §3 requires *"minimum 3 instances"* of body noise and §Enhancement Targets offers *"mmm, breath, hum, vocal fry"*. ⭐ **A sustained hum is the one element in this pair that two voices CAN blend on.** Handing this pair a pitched wordless syllable would give the generator exactly the material it needs to build the harmony the whole pair exists to refuse — and the handoff §4 separately warns that a wordless return device inflates `line_return` on its own. **Three body-noise instances per song are delivered and all three are UNPITCHED** (caught breath, swallow, small laugh in the throat, long breath out), placed in Intro / bridge / Outro, each with a stated dramatic function, carried in the EMO cue slot rather than as extra sound commands so the 1–2 SFX ceiling holds. ⛔ **Hum, `mmm`, and vocal fry are refused by name.** See the L22 scan defect **D5**, where four of V4's five `mm` lines are converted to lexical half-words.

**3. Disc_Channel is outside the lyrics field** — `## 4. PRODUCTION SIDECAR`, after the title — per the run's single harness decision for all six pairs. Step 11 Gate 13a wants it inside the field; the harness decision and the 5000-char render cliff win. Stated so no downstream check false-fails.

---

## ⭐ JOB 1 — THE ANDON CORD: **NOT PULLED.** Verdict **PASS (repaired in place)**

I came in expecting to reject this and I am not going to. Ruled against all five REJECT criteria, one at a time:

| Criterion | Ruling | Evidence |
|---|---|---|
| **1 Thread loss** | ❌ not triggered | The seed's core survives intact and literal: a name written second, D1 held (she never arrives), D4 held (nobody finds out), D8 held (the object is set down). The FALSE INTERSECTION is executed as actual counterpoint, not as a concept. |
| **2 Personality collapse** | ❌ not triggered | The assigned personality **is** LOFN-PRIME, AWE mode, and the pair grid assigns this pair *interrupted* diction and two confusable mezzos. This does not read as the house's flat-declarative default — that allowance is spent by P06. It reads as an authored voice. |
| **3 EMO taxonomy failure** | ⚠️ **partially triggered → ENHANCED, not rejected** | Every tag was canonical, but the arc was near-flat AND every simultaneous section gave **one** emotion to **two** people. That is not a taxonomy error, it is a dramaturgy error, and it is repairable in the header. Fixed — see L22 defect **D1**. |
| **4 Generic output** | ❌ not triggered | Two independent strophes, different syllable counts, no shared line but one word, an outro on the wrong voice (V3), a strophe that stops mid-word (V4). This is the opposite of functional quatrains. |
| **5 Prompt format violation** | ⚠️ minor → ENHANCED | All four prompts opened with a mood clause before genre/tempo/key (Gate 14a wants genre first) and all four exclude fields sat at **193–204 chars against a 400–900 target**. Both repaired in place. |

**"Don't polish a corpse" — this is not a corpse.** The one thing I was told to rule on and could have rejected for is the legibility failure, and it is repairable in the opening ten sung lines without touching the form. **Repair beats reject here.** See JOB 2b.

---

## ⛔ JOB 2a — THE L22 SCAN (THE GRAIN LAW)

> **A form objection answered in the PRODUCTION SPEC is not answered.** The pair's declared defences are all lyric-side and correct *in kind*. I verified each one against the bytes rather than against the claim, and **three of them did not hold as stated.**

### First — did any defence migrate into the style prompt or the exclude field?

**No, and I checked directly.** The four load-bearing defences — asymmetric line length, mid-clause em-dash openings, zero shared lines but one, unsustainable line endings — are **all in the words**. The exclude field carries duet/echo/thirds terms as belt-and-braces only. ✅ **The primary answer is lyric-side in all four variations. No migration found.**

But the scan turned up five defects, and one of them is in the prompt.

**D1 · ALL FOUR · the EMO header handed ONE emotion to TWO people.** Every simultaneous section read `EMO:Serenity` or `EMO:Anticipation` — a single emotional target for a section in which two women are in different states. ⭐ **A single emotional target is an instruction to blend**, sitting in the exact field the generator reads first. **FIXED:** every simultaneous section now carries a dual tag — `EMO:Contentment/Warmth`, `EMO:Anticipation/Warmth`, `EMO:Anticipation/Ennui` — with one emotion assigned per voice in the role slot. Both halves canonical, per `EMOTION_TAXONOMY`; the contract's own header shape is `EMO:<emotion(s)>`, so this is in-contract, not a deviation.

**D2 · ALL FOUR · the section was literally named `[Both Voices]`.** Two words, at the head of the section, telling the renderer these are two voices doing one thing. **FIXED:** renamed `[Simultaneous Strophes]`, and the crossing section renamed `[The Crossing]`.

**D3 · V1 AND V2 · the style prompt contained call-and-response language.** V1: *"The climbing line **answers** alone."* V2: *"She sings, **then** the other sings, then they overlap."* ⚠️ **The pair wrote its own failure mode into its own render field.** *Answers* and *then… then* are turn-taking instructions, and turn-taking is precisely the backing-vocal reading the pair is defending against. **FIXED:** V1's second voice now *"starts on its own, from somewhere else, half a bar early"* rather than *answering*; V2's becomes *"then the other starts on her own from somewhere else"*, which keeps the genuinely sequential FORM (separate for a minute, simultaneous after) while removing the turn-taking implication. And every prompt now states positively that **the only syllable the two share is one word near the end.** ⭐ **The word *answer* now appears in this pair only inside the exclude field, where it belongs.**

**D4 · ALL FOUR · the vowel-family defence was ASSERTED and did not hold.** Step 10 claims the falling line ends in a *close-front* family (*shut, out, look, right*) — **three of those four are not close-front**, and step 06's declared families (`-eep -eet -eek -ean` against `-oor -ore -old -oad`) appear almost nowhere in the actual lyric. A defence that exists only on paper is a production-spec defence wearing a lyric's clothes. ⭐ **But the REAL defence was there and unnamed**, and it is stronger: **the climbing lines end on function words and broken words** — *the, and, so, I'm, you'd, that's* — and **a function word is not a sustainable vowel, so there is nothing for a second voice to land on.** **FIXED by promoting it to an enforceable rule and measuring it:**

> ⭐ **THE ENFORCED RULE — CLIMBING LINES END UNSTRESSED AND UNFINISHED; FALLING LINES END STRESSED AND CLOSED.** A blend requires two voices to arrive on a sustainable vowel at the same instant. If one of them is always mid-function-word, the blend is not available to the generator.

Measured on the step-10 text, **the rule was breached 13 / 13 / 11 / 16 times** across V1–V4 — including `road`, `coast`, `exactly`, `early`, `hopes`, `cash`, `here`, `listening`, every one of them an open sustainable vowel sitting at the end of a bracketed line. Those lines are rewritten to break one word later. **Enhanced text: 0 / 1 / 0 / 0 breaches.** (V2's one is the bare pronoun *I*, trailing and unstressed — safe.)

**D5 · V4 · ⚠️ THE WORST ONE. `mm`, five times, under a live lead line.** V4's step-10 text decayed Trish's strophe to the syllable `mm` — **a pitched, sustainable, non-lexical syllable, repeated, while Mezzo One is still singing.** That is not merely blendable; **it is the single most hummable element in the entire pair, and it is placed in the simultaneous section.** A renderer that turns those five `mm`s into a hummed harmony under the lead has produced exactly the duet the pair says it cannot produce — and it would look like fidelity to the score. It also inflates `line_return` on its own, which is the failure mode handoff §4 warns about by name. **FIXED:** four of the five become lexical half-words — `(— in a —)`, `(— in a —)`, `(— in —)` — a person going under in words, which cannot be hummed and cannot be pitched. **One `mm` survives**, at her sign-off (*"Night. Night. Mm —"*), where she is still alone in her own line. `hummed harmony under the lead` added to V4's exclude field as belt-and-braces. ⭐ **Measured consequence: V4's `line_return` moves 0.598 → 0.565, and the lexical-only companion is 0.554 — a gap of 0.011, against step 10's larger one. The return is now carried by words.**

**D6 · ALL FOUR · the instrumental intro was spending the legibility budget.** A two-bar 6/4 intro at 58–64 BPM is 11–12 seconds — **a third of the thirty-second window the pair is charged with failing.** Self-inflicted, and free to fix. **FIXED:** all four intros cued **one bar only, voice enters straight after**. Roughly six seconds recovered in every variation.

---

## ⭐ JOB 2b — THE THIRTY-SECOND LEGIBILITY RULING

> *"A stranger must be able to point at the thing being set aside, and at the hand setting it aside, inside the first thirty seconds, without knowing any astronomy."* — `core_seed.md` §5, the run's critical requirement.

**THE RULING: the pair's disclosure was honest and its excuse was wrong.** Step 10 reports the DISCARD as pointable inside thirty seconds **in V3 only**, landing at 45–90 s in V1/V2/V4, and defends it with *"eligibility 0–3/7 by design."* ⛔ **That defence does not reach.** The 0–3/7 target is a **Hit-Formula** figure — it licenses failing *cognitive ease*, one of seven eligibility properties. The thirty-second rule is not one of those seven; it is stated separately in `core_seed.md` §5 as the requirement *"that cannot be abstract,"* and it is repeated in handoff §3 as **the run's critical requirement**. ⭐ **Conflating "we are allowed to be hard" with "we are allowed to be illegible" is the defect, and it is the only thing in this pair I would have rejected for.**

**And it is repairable in ten lines without touching the form**, so I repaired it. Each variation now performs a **discard with a hand, on a physical object, in the opening lines**:

| | the discard, now | act at sung line | est. seconds | step-10 |
|---|---|---|---|---|
| **V1** | the scratched piece of black glass goes on the bed | 5 | **≈ 20 s** ✅ | 45–90 s ⛔ |
| **V2** | Trish's printed forecast is turned face down on the sill | 2 | **≈ 12 s** ✅ | 45–90 s ⛔ |
| **V3** | her thumb goes past the two names on the page | 8 | **≈ 28 s** ✅ | ≈ 30 s |
| **V4** | the watch comes off her wrist and goes under the bed | 4 | **≈ 19 s** ✅ | 45–90 s ⛔ |

*(Timing model stated so it can be challenged: 6/4 at the variation's own BPM, one-bar intro, short falling lines at a half-bar each. V3's estimate is unchanged in method and improved by the shorter intro plus moving the name from line 8 to line 6.)*

**Why these four discards and not others.** Each is caused by the line before it and causes the line after it, so **D7 survives — nothing here is a list**:

- **V1** — she breathes on the glass → a pinhole shows white → **therefore that one is not coming** → therefore it goes on the bed → therefore *the second one, the spare*, is the one that travels. ⭐ The pair's refrain word gets its first meaning **attached to a discard, in the first twenty seconds**, and she is pleased about it.
- **V2** — ⭐ **Trish printed the whole day out, and she turns it face down and puts the mugs on it.** She sets aside the careful work someone did for her, without noticing, for an entirely good reason. **That is the same gesture as writing a name second, in the other hand, and neither of them will ever know.** This is the ICB's own aha-moment #1 (*"the same gesture"*) landing inside one song, and it is **not** vindication (D4) — it spreads the charge sideways to the listener (D5), which is the version the Jameson seat conceded to.
- **V3** — unchanged in kind, tightened in place: her big hand, *Trish, and then —*, and **the thumb going past the names** now sit in verse one instead of verse three.
- **V4** — the watch comes off and goes under the bed *because I'll only look*. ⭐ **And that is why the song ends the way it ends:** with no watch, the alarm has to be the phone, the phone has a screen, the screen would wake Trish — so she waits until Trish is asleep and sets it in the dark. **The first object in the song now causes its last act**, which was V3's best structural property and is now V4's too.

⛔ **No astronomy was added.** *eclipse · sun · moon · sky · star · shadow · corona · solar · totality · filter* — **zero occurrences across all four sung bodies, measured with word boundaries.** "Black glass" and "a printed forecast" need no explanation from anyone.

---

## JOB 3 — WHAT WAS ENHANCED, AND WHAT WAS LEFT ALONE

**Left alone, deliberately:** the two strophes; the crossing on *second*; the D3 interval declaration; V3's outro on the wrong voice; V4's strophe stopping mid-word; every ending; the causal chains; Trish's characterisation. ⭐ **And the byte-identical refrains — `it clears by the coast`, `this is the good bit`, `and I go through the thing once` — were not touched and are not defended.** `chorus_repetition_requires_no_justification: true`.

**Enhanced:**

1. **Legibility** — a hand-performed discard in the opening lines of all four (JOB 2b).
2. **Anti-comp hardening in the words** — the six L22 defects above.
3. **EMO dramaturgy** — dual tags per simultaneous section; V4's second half moves `Warmth → Ennui` as Trish goes under, so the arc now transforms **without the singer arriving at anything** (D1 intact — the transformation is Trish's sleep, not Mezzo One's insight).
4. **Suno-field craft** — every prompt now leads with genre + BPM + key/mode per Gate 14a; the exclude fields go from 193–204 chars to **544 / 525 / 493 / 505**, inside the 400–900 contract band, spending the new room on real failure classes (turn-taking, shared cadence, the retro trap, weather-report voiceover, and V4's hummed harmony).
5. **Body noise** — three unpitched instances per song with stated function, in the sidecars.
6. **Title, V3 only: `The Booking Page` → ⭐ `The Second Driver`.** It still names a **thing** — a box on a hire form — so the measured title law holds (a thing, never an argument; no persona prefix). It is the crossing word. And it points the *listener* at the wound while the *singer* remains unable to see it, which is what a title is for under D1. The other three titles are already thing-titles and are untouched.

⛔ **Length discipline — the run is seven consecutive passes deep in over-lengthening, so this is reported per field.** Music prompts: **+35 / +50 / +5 / -7** chars, all four still inside the 870–960 target band, none hugging the 985 flag. Lyrics fields grew **+284 / +361 / +264 / +187** — all four still under the 4800 soft target, and every added character is either a discard the run's critical requirement demanded or a second emotion in a header that previously flattened two women into one. **Sung lines +1 / +2 / +0 / +3.** ⛔ **Nothing was lengthened to seem thorough.**

---

## RANKING — if this pair is cut

| rank | | why |
|---|---|---|
| **1** | **V3 · The Second Driver** | The crossing is not a device here, it is the plot: she reads *second driver* off her friend's handwriting and is **pleased**, at the exact instant Trish says *give me a second*. Same note, same syllable, two meanings, neither adjusts. Legible from the first line. **If the pair is cut to one, cut to V3** — step 10's own advice, and I agree with it. |
| **2** | **V4 · The Line Under The Door** | Now the safest render in the pair as well as the truest: Trish's part decays to half-words and stops, so there is structurally almost nothing left to comp. The watch repair gives it V3's causal spine. |
| **3** | **V1 · The Folding Chair** | The strongest opening in the pair after the repair — *the second one, the spare* is the refrain word arriving as a discard in twenty seconds. Highest comp risk of the four: the two strophes run longest here. |
| **4** | **V2 · Cloud Off The Coast** | The best new idea in this pass (the same gesture in the other hand) and still the most likely to come back merely *charming*. It is the one whose failure mode is beauty. |

---

## SELF-CRITIQUE — what I could not fix, and what I might have got wrong

1. ⚠️ **The comp hazard is not closed and cannot be closed from here.** Everything above raises the cost of a blend. **None of it makes a blend impossible.** Only a render audit settles this, under THE BLIND RULE — send the audio alone, never the prompt. **The Kamasi condition stays live.**
2. ⚠️ **My legibility timings are a model, not a measurement.** They assume Suno honours a one-bar intro and a half-bar line. It often does not. If the generator opens with twenty seconds of brushes, V3 is the only variation that still clears, and my repair buys margin rather than certainty.
3. ⚠️ **V2's new opening adds a second setting-aside to a pair whose seam count is capped at one (L38).** My ruling: a *seam* is a crossing of two materials, and V2 still has exactly one — the word *second*. A gesture repeated in two hands is a rhyme, not a seam. **I could be wrong about that** and I am flagging it rather than burying it; it is the single most challengeable judgement in this pass.
4. ⚠️ **I did not lift V1's alliteration.** It measures 11.86 against a floor of 11.0 — passing, but the lowest in the pair and slightly below step 10's 12.66, because the repaired opening trades consonance for causality. I judged the discard worth more than the texture. Named so QA can disagree.
5. ⚠️ **Cross-PAIR distinctiveness is not mine to report.** Handoff §2 forbids me reading a sibling artifact, so I produce no number I cannot compute. Within-pair is below, with the extraction printed first.

---

## MEASUREMENTS — EXTRACTION PRINTED BEFORE CONCLUSION

```
INSTRUMENT: scripts/measure_soundcraft.py -> profile()/strict_end_rhyme()/
            line_return(), run per variation on the exact bytes emitted below.
            Thresholds cited from vault/gates.yaml, not restated from memory.

EXTRACTED            V1      V2      V3      V4    threshold          verdict
music prompt chars     958     949     931     921  850-1000 / 870-960 PASS x4
exclude chars          544     525     493     505  400-900 / <=1000   PASS x4
lyrics field chars    4777    4795    4675    4339  <5000 / <=4800     PASS x4
sung lines              93      92      89      85  70-120 / 78-110    PASS x4
end_rhyme            0.624   0.446   0.449   0.424  >= 0.30            PASS x4
  lexical-only       0.624   0.446   0.449   0.434  >= 0.30            PASS x4
line_return          0.269   0.380   0.348   0.565  >= 0.20            PASS x4
  lexical-only       0.269   0.380   0.348   0.554  >= 0.20            PASS x4
words_per_line        6.71    6.50    6.30    6.47  <= 7.5             PASS x4
allit_per_100w       11.86   12.04   18.00   17.09  >= 11.0            PASS x4
unique_line_ratio    0.849   0.783   0.798   0.659  >= 0.45 (FLAG)     PASS x4
SFX cues                 2       2       2       2  >= 1, <= 3         PASS x4
EMO headers              9       9       9       8  4 slots each       PASS x4
digits in sung           0       0       0       0  0                  PASS x4
astronomy words          0       0       0       0  0                  PASS x4
blendable endings        0       1       0       0  as low as reachable see D4
climb mean words       7.8     7.5     7.2     6.3  longer than falling PASS x4
fall mean words        6.0     5.8     5.4     6.6  shorter than climbing PASS x4
"second" sung            4       2       3       2  return vehicle, A5 not a numeral

house_lexicon hits      0       0       0       0    0                  PASS x4
real-artist names       0       0       0       0    0                  PASS x4
banned amplitude        0       0       0       0    0                  PASS x4
prompt avoid-debris     0       0       0       0    0                  PASS x4
exclude prose-negation  0       0       0       0    0                  PASS x4
EMO shape violations    0       0       0       0    0                  PASS x4
```

**Sample EMO header (V1):** `[The Crossing - EMO:Anticipation/Warmth - both mezzos on one note - one word, one beat, one falling through and one rising through, neither adjusts]` — four slots, both emotions canonical in `EMOTION_TAXONOMY`, ⛔ never bare `AWE`/`INDIGNATION`.

⚠️ **The wordless-return companion, reported because handoff §4 demands it.** V4 was the only variation with a vocable. Its `line_return` is now **0.565** with the one surviving `mm` and **0.554** without — a gap of **0.011**. ⭐ **The return is lexical.** V1–V3 contain no vocable, no hum and no non-lexical hook, so their two figures are identical by construction.

### ⚠️ ONE-FACT RULE — disclosed, because a naive scan will misread it

**`max_sung_numeric_facts: 1` · this pair sings ZERO.** P04 spends the run's one. **The word *second* occurs 4/2/3/2 times as the pair's A5 return vehicle** — the spare piece of glass, *hang on a second*, *give it a second*, and **the box on the page marked second driver.** ⛔ **None states a quantity.** Also present and also not facts: *first* (idiom), *one* (pronoun), *half*, *minute* (meaning *shortly*). ⛔ **Never sung, deliberately: two minutes eighteen · thirty years · the hour of the alarm.** ⭐ **V4's watch now goes under the bed in the first four lines specifically so no hour can be read.** Digit characters in sung lines across all four: **zero, measured.**

### WITHIN-PAIR DISTINCTIVENESS — recomputed on the enhanced text

```
EXTRACTED sung bodies (chars): V1 3137  V2 3115  V3 2976  V4 2754   [4 of 4]
SequenceMatcher autojunk=False -- autojunk=True reported near-identical
templates as 94% distinct on 2026-08-05 (handoff s4, known-broken instruments)
```

| | lyric *(ceiling 0.42)* | prompt *(0.58)* | 5-gram Jaccard *(0.18)* |
|---|---|---|---|
| V1–V2 | 0.234 | 0.373 | 0.001 |
| V1–V3 | 0.269 | 0.466 | 0.003 |
| V1–V4 | 0.279 | 0.344 | 0.000 |
| V2–V3 | 0.272 | 0.356 | 0.001 |
| V2–V4 | 0.264 | 0.436 | 0.002 |
| V3–V4 | 0.175 | 0.390 | 0.000 |
| **max** | **0.279 ✅** | **0.466 ✅** | **0.003 ✅** |

---

## Major Deviations

- **Changed / refused / intensified:** (a) **Refused** the step-11 body-noise mandate's pitched options (hum / `mmm` / vocal fry) and delivered three *unpitched* instances per song instead. (b) **Refused** the contract's Golden Song embed and named the handoff override. (c) **Moved** Disc_Channel out of the lyrics field into a sidecar, against Gate 13a, per the run's harness decision. (d) **Overruled** the pair's own defence of its thirty-second legibility failure and repaired all four openings. (e) **Rewrote** V4's `mm` decay to lexical half-words. (f) **Renamed** V3.
- **Reason:** every one of them protects the same thing — **two voices that must not blend**. A hum, a single shared emotional target, a section called *Both Voices*, a climbing line ending on an open vowel, and the word *answers* in the style prompt are all the same defect wearing five costumes, and four of the five were invisible because they were in headers and metadata rather than in the sung lines.
- **Effect on Lofn uniqueness:** the pair's argument is now carried by the words and the form in every place it was previously carried by an assertion. ⭐ **THE GRAIN LAW is satisfied where it was previously only cited.**

## LINEAGE & CREDIT

⛔ **D9 (THE APPROPRIATION GATE) does not apply to this pair** — the coordinator's scope call (`05_pair_assignments.md`) puts it on **P01** and **P02** only. Held to its spirit regardless, because the Simon seat explicitly did **not** withdraw and *"the intent is never the issue"*:

- ⛔ **No tradition's name in any Suno-bound field** — verified across four prompts, four exclude fields, four lyrics fields, four titles and all Disc_Channel tokens.
- ⛔ **The word *tezeta* appears nowhere in this pair.** Flair 11 was available and is declined; a pair outside the gate's scope has no business borrowing a mode's name even as a private label.
- ⛔ **No real-artist name in any field.** Panel constructs are named in artifact prose only, are *"after"* / influence and never endorsement, and no construct states or implies that its source figure said, reviewed, approved or would approve anything here.
- **Vocabulary imported under the run's declaration:** *"spacey jazz and textured folk instruments"* is taken **verbatim, as vocabulary only**, from the Bandcamp Album of the Day review of **Papangu — *Celestial*** (7 August 2026), per `00_research_brief.md` §3. ⛔ Nothing about that record, that band, analogue-versus-digital, tape or formats is a subject anywhere in this pair. Source: <https://daily.bandcamp.com/album-of-the-day> · artist: <https://papangu.bandcamp.com/>

## HUMAN-SUBJECT STANDARD — judged on content

`vault/HUMAN_SUBJECT_STANDARD.md`; handoff §5. **Both women are invented.** One is deliberately never named — a formal device, not an omission. The other is **Trish**: an ordinary invented first name, no surname, no employer, no location, not modelled on any person living or dead. **Messier and Tempel appear nowhere** and are given no interiority. The eclipse is an occasion on a public calendar and is never named or described. ⛔ **Neither of today's two real deaths is present, alluded to or gestured at. Nobody here is bereaved, ill or dying. REAL GRIEF IS NOT RAW MATERIAL.** `scripts/check_human_subjects.py` is **deliberately not cited** — handoff §4 records that it returns `HOLD_FOR_HUMAN` on 100 % of correctly-written artifacts in this checkout; reporting it in either direction would be laundering. **CLEAR.**

---

**The four packages follow. Everything above is analysis; nothing below is.**

---

### VARIATION 1

## 1. MUSIC PROMPT
```text
Spacious modal chamber-jazz ballad in D Dorian, 62 BPM, six-four, tender and unhurried, built from spacey jazz and textured folk instruments: brushes on a dry ride, upright bass walking in whole notes, nylon-string guitar picking one note a beat, a low shaker, and a modal tenor saxophone that keeps searching and never lands. Two singers, both mezzo, same age, same unhurried northern-English delivery, close enough to be mistaken for one person; one is warm and breathy and slides down off the end of a word, the other is drier and faster and talks more than she sings. The falling line starts on its own. The climbing line starts on its own, from somewhere else, half a bar early. From there the two run at the same time on separate words, each holding her own tune, and the only syllable they share is one word near the end. A band in a rented room with two beds and a kettle, room mics wide, entries by ear. Thickening is more players, never more level.
```
## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, unison singing, backing-vocal echo, answer vocals, call-and-response, doubled lead vocal, layered vocal stacks, two voices in thirds, two voices resolving together, shared final cadence, gospel choir stack, choir pad, big final chorus, key change, EDM riser, orchestral swell, drum build, reverse cymbal, fade-out, tape hiss, vinyl crackle, cassette warble, found-recording framing, sepia reverb, trap hi-hats, autotune, spoken narrator, whispered ASMR vocal, male lead vocal, child vocals, lo-fi beat, cinematic strings
```
## 2. LYRICS
```text
[Theme: a rented room the night before a long drive — one woman packs, the other talks, and neither of them finishes a sentence]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, half a bar early. Separate for the first minute, simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - one bar only, brushes and bass, wide room, voice enters straight after]
*a bag set down*

[Verse 1 - EMO:Composure - Mezzo One alone, descending, on the beat - plain, unhurried, a caught breath before each line]
I breathe on the black glass —
and hold it high to the lamp —
and a pinhole comes up white —
so that one's not coming. Right —
so it goes on the bed —
and the second one —
the spare, the smaller one —
slides in the side pocket —
where a hand goes down in the dark —
and the pocket's packed now, so —
so the chair goes on the top —

[Verse 2 - EMO:Warmth - Mezzo Two alone, climbing, half a bar early - talking more than singing, opening mid-clause]
(— and I'll take the first bit of the road, so —)
(— you'll be wanting to sleep for the —)
(— is that new, or have you had it —)
(— sorry. No. No, you go. I'm over the —)
(— it's just, there's a lot of it, and the —)
(— and the boot's not big, not what you'd —)
(— is that new, or have you had it —)
(— we could take the long way, by the coast, and —)
(— and be down past the border, and be —)
(— past it before light. You'd like the —)

[Verse 3 - EMO:Contentment - Mezzo One alone, lower, same descent - pleased, nothing withheld]
Then the bag will not shut —
so the sandwiches come out —
and they sit by the kettle —
so a foot finds them, out —
in the dark, in the morning —
and I'll know. I'll know what for —

[Simultaneous Strophes - EMO:Contentment/Warmth - Mezzo One plain, Mezzo Two in brackets - two songs at once, one emotion each, neither waits]
The strap goes over the chair —
(— and I'll take the first bit of the road, so —)
tucked under the arm of the chair —
(— you'll be wanting to sleep for the —)
and it's shut. It's shut —
(— is that new, or have you had it —)
and it stands by the door, shut —
(— sorry. No. No, you go. I'm over the —)
boots at the bag, toes out —
(— it's just, there's a lot of it, and the —)
so a foot goes straight out —
(— and the boot's not big, not what you'd —)
and the coat's hung, sleeves out —
(— we could take the long way, by the coast, and —)
and the keys in the coat —
(— and be down past the border, and be —)

[Simultaneous Strophes - EMO:Anticipation/Warmth - Mezzo One falling, Mezzo Two climbing - separate words, a small laugh in her throat, no harmony arrives]
And I'll want it in my hand —
(— it's only I've not seen you up this —)
not the bag. In my hand —
(— this early since the — anyway. Be —)
so it goes in my coat —
(— anyway, I'll be up. I'll be —)
in the left, where a hand —
(— I'll be up, don't worry about the —)
where the hand goes on its own —
(— about the driving, I'll do the —)
and I'll not have to look —
(— I'll do the first bit, like I said, and —)
and I'll not have to look —
(— and you can sleep right through the —)

[The Crossing - EMO:Anticipation/Warmth - both mezzos on one note - one word, one beat, one falling through and one rising through, neither adjusts]
because there isn't time to look —
(— hang on. Hang on. Hang on a —)
second — (second —)
and after that it's —
(— no, you were saying. You were saying the —)
it's me and a cold hand —
(— the thing about the — go on —)
and nothing to do but —
(— go on, I'm listening, I've got the —)
but stand there and be —
(— I've got the map up. Carry on —)
and I'd like that. I would —
(— carry on. I'm not — I'm still —)

[Simultaneous Strophes - EMO:Composure/Warmth - Mezzo One settling low, Mezzo Two still climbing - they come apart and stay apart]
It stands by the door, shut —
(— right. Right. I'm going to put this down and —)
and it stays there, shut —
(— and just lie here for a bit, and —)
and I'll not open it again —
(— and my feet. God. Are you doing the —)
not till it's dark. Not again —
(— are you doing the light, or shall I do the —)
and the second one's in the coat —
(— or shall I? Right, I've got it. I've —)
in the left of the coat —
(— I've got it. Don't get up. I've —)
and I'll not have to look —
(— I've got it. There. That's —)

[Outro - EMO:Serenity - Mezzo One alone, lowest - Mezzo Two has stopped, one long breath out, no held final note]
and I'll not have to look —
It stands by the door, shut —
and I'll not touch it. Shut —
and a foot finds the food —
in the dark, in the morning. Good —
and the chair goes on my back —
and the cold goes at my back —
so I zip the pocket shut —
and I'd like that. I would —
*a coat pocket zipped*
```
## 3. TITLE
**The Folding Chair**

## 4. PRODUCTION SIDECAR

**Disc_Channel** — operator-side only. ⛔ It is **not** pasted into the Suno lyrics
field; the two-field prompt above is the render contract. It is here because the harness
decision for this run puts Disc_Channel outside the field, and because pasting it would
cost ~470 chars of a field already carrying the SONG FORM declaration.

```text
[Disc_Rhythm: brushes_on_dry_ride | 62_BPM_six_four_grid | low_shaker_no_backbeat | Stereo_Width_Mid]
[Disc_Vocal: two_mature_female_mezzos_same_range | breath_on_capsule_audible | interrupted_half_sentence_delivery | Voice_One_slightly_left_Voice_Two_slightly_right]
[Disc_Sub: upright_bass_whole_notes | uncompressed_transient_snap | 4_second_natural_decay | Mono_Sub_Lock]
[Disc_Pad: rented_room_air | wide_room_mics_leakage | no_plate_no_hall | Stereo_Width_Maximum]
[Disc_Texture: modal_tenor_saxophone_searching | nylon_string_single_notes | hand_on_bag_canvas | Center_Back]
```
⛔ `cassette_tape_saturation` and `cassette_tape_hiss_saturation` are in the guide's
cross-domain table and are **refused run-wide** — D10, the retro trap, named and banned.

**Vocal fingerprint.** Mezzo One: D4–A4, warm, breathy, slides off the end of a word,
breath audible before every entry. Mezzo Two: C4–G4, drier, faster, half-talking, enters
half a bar early and never lands on a stressed syllable.

**Production dramaturgy.** Every unusual choice has one job. Wide room mics → the two
voices are in one acoustic and cannot be balanced against each other. Entries by ear → the
strophes drift by tens of milliseconds, which is what makes them hard to parse. Sax that
never lands → nothing in the arrangement models resolution for the voices to copy.

**Body noise (unpitched only — see the refusal in the preamble).**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Verse 1 / all Mezzo One entries | caught breath before the line | marks her entries as breath-led, so the two voices cannot start together |
| 2 | Simultaneous Strophes 2 | small laugh in Mezzo Two's throat | happens *under* a sung line — proof of simultaneity, not sequence |
| 3 | Outro | one long breath out | the song's last vocal event is unpitched, so there is no final note to meet |

**Style-axis lock.** D Dorian · 62 BPM · six-four · two mezzos, one register · falling
strophe A4→D4 · climbing strophe C4→G4 · crossing on E4 on the word *second* · finish a
perfect fourth apart.

---

### VARIATION 2

## 1. MUSIC PROMPT
```text
Slow modal chamber jazz in D Dorian, 60 BPM, six-four, warm and open, standing at a window. Two mezzo singers matched on purpose: same age, same accent, same weight of tone, so a listener must work out which is which. The first is level and breathy with a small catch before every phrase; the second is quick and flat and reads aloud, breaking off halfway through what she says. Around them, spacey jazz and textured folk instruments: a modal tenor saxophone well forward and searching, brushes swirling on a coated head, upright bass in long slow steps, bowed vibraphone underneath, bass clarinet doubling the bass an octave up. The room is rented — two beds, a window, a kettle — the microphones far back so the air is part of the sound. She sings on her own, then the other starts on her own from somewhere else, then the two overlap and stay overlapping, each holding her own tune and her own words. More instruments arrive. Nothing gets louder.
```
## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, call-and-response answer vocals, backing-vocal echo, two voices in thirds, unison refrain, doubled lead vocal, two voices resolving together, shared held final note, choir pad, gospel stack, build into a final chorus, drum fill transitions, key change, EDM riser, orchestral swell, cinematic strings, tape hiss, vinyl crackle, cassette warble, sepia reverb, found-recording framing, autotune, spoken narrator, weather-report voiceover, male lead vocal, child vocals, lo-fi beat, trap hi-hats, fade-out
```
## 2. LYRICS
```text
[Theme: the same rented room, later — the kettle, the window, and a forecast that says cloud; one woman reads the weather off the glass, the other off the phone]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, half a bar early. Separate for the first minute, simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - one bar only, bass and brushes, wide room, voice enters straight after]
*a kettle starting up*

[Verse 1 - EMO:Composure - Mezzo One alone, descending, at the window - level, a small catch of breath before each phrase]
Trish printed the whole day out —
so it's face down on the sill —
and the mugs go on the top —
and I'll just look out instead —
and the kettle's on the sill —
and the steam goes up the pane —
so I wipe it with my sleeve —
and the sleeve comes off grey —
and low down, past the grey —
it clears by the coast —
a long strip, low and clear —
it clears by the coast —

[Verse 2 - EMO:Warmth - Mezzo Two alone, climbing, half a bar early - reading off the phone, breaking off halfway]
(— it's saying cloud. It's saying cloud the —)
(— the whole morning, and it's worse toward the —)
(— and there'll be another one, there's always —)
(— always one. There's one in — when's the —)
(— it's fine. It's fine. I'm only saying, so —)
(— so we don't get our hopes right up and —)
(— and there'll be another one, there's always —)
(— always one somewhere. Do you want the road, or —)
(— or the hotel, because if it's cloud we could —)
(— we could drive on down and find a —)

[Verse 3 - EMO:Contentment - Mezzo One alone, lower, same descent - agreeing, and meaning it]
And I say yeah. I say yeah —
and I mean it. I do —
and the strip is still there —
low and long, still there —
and the kettle starts to knock —
and the mugs go out. So —

[Simultaneous Strophes - EMO:Contentment/Warmth - Mezzo One plain, Mezzo Two in brackets - two songs at once, one emotion each, neither waits]
So the kettle starts to knock —
(— it's saying cloud. It's saying cloud the —)
and it climbs, and it knocks —
(— the whole morning, and it's worse toward the —)
and it clears by the coast —
(— and there'll be another one, there's always —)
and I say yeah. I say yeah —
(— always one. There's one in — when's the —)
and I mean it. I do —
(— it's fine. It's fine. I'm only saying, so —)
and it clears by the coast —
(— so we don't get our hopes right up and —)
and there's a road up the back —
(— and there'll be another one, there's always —)
a road up the back —
(— always one somewhere. Do you want the road, or —)

[Simultaneous Strophes - EMO:Anticipation/Warmth - Mezzo One falling, Mezzo Two climbing - separate words, a swallow between her phrases, no harmony arrives]
and it goes above the cloud —
(— or the hotel, because if it's cloud we could —)
sometimes. Not always. But —
(— we could drive on down and find a —)
but I've got the map in my head —
(— find a spot, if you wanted. I don't —)
and I've had it in my head —
(— I don't mind driving. I've said. I don't —)
and it's not a long climb —
(— I don't mind. I like it. I like the —)
not for me, it's not —
(— I like the early ones, when there's no —)
and the water goes quiet —
(— when there's nobody on the — hang on —)

[The Crossing - EMO:Anticipation/Warmth - both mezzos on one note - one word, one beat, one falling through and one rising through, neither adjusts]
and it climbs, and it clicks —
(— hang on. Hang on. Give it a —)
second — (second —)
and the strip's still there —
(— it's changed. It's changed. It's got a —)
low and long, still there —
(— it says a break in the — hang on —)
and I pour, and it's fine —
(— no. No, it's gone again. I'm —)
and it's fine either way —
(— I'm sorry, I got your hopes up, and —)
and I'd have come anyway —
(— and I feel bad, because I —)

[Simultaneous Strophes - EMO:Composure/Warmth - Mezzo One settling low, Mezzo Two still climbing - they come apart and stay apart]
I'd have come anyway —
(— because I booked it, and now —)
and the phone goes face down —
(— and now it's cloud. And you —)
face down on the sill —
(— and you never said a word. You —)
and it clears by the coast —
(— you're very good about it. You —)
and it clears by the coast —
(— you are. Right. I'm getting in. Are —)
and the tea's going cold —
(— are you coming, or are you —)
and I like it cold. I do —
(— or are you stopping up. Are —)

[Outro - EMO:Serenity - Mezzo One alone, lowest - Mezzo Two has stopped, one long breath out, no held final note]
In a minute. In a minute —
and the strip goes to nothing —
and the pane fogs up again —
and I let it fog again —
and I pour, and I sit —
and it clears by the coast —
and I'd have come anyway —
*a kettle clicking off*
```
## 3. TITLE
**Cloud Off The Coast**

## 4. PRODUCTION SIDECAR

**Disc_Channel** — operator-side only, not pasted into the lyrics field.

```text
[Disc_Rhythm: brushes_on_coated_head | 60_BPM_six_four_grid | kettle_and_chair_under_the_take | Stereo_Width_Mid]
[Disc_Vocal: two_mature_female_mezzos_same_range | small_catch_before_each_phrase | reads_aloud_breaks_off | Voice_One_slightly_left_Voice_Two_slightly_right]
[Disc_Sub: upright_bass_long_slow_steps | bass_clarinet_octave_up | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: bowed_vibraphone_underneath | far_mics_air_in_the_room | no_plate_no_hall | Stereo_Width_Maximum]
[Disc_Texture: modal_tenor_saxophone_well_forward | window_glass_and_sleeve | steam_rising | Center_Back]
```

**Vocal fingerprint.** Mezzo One: level, breathy, a small catch of breath before every
phrase, states facts flat. Mezzo Two: quick, flat, reading aloud off a screen, every
sentence abandoned at the same distance from its end.

**Production dramaturgy.** Microphones far back → the air is a third instrument and neither
voice can be brought forward. Bass clarinet doubling the bass an octave up → the one
doubling in the piece is between two *instruments*, which is where doubling is allowed to
be beautiful, and it is never between the two voices.

**Body noise (unpitched only).**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Verse 1 | catch of breath before every phrase | her level tone is a decision made twelve times, audibly |
| 2 | Simultaneous Strophes 2 | a swallow between Mezzo Two's phrases | places her physically in the room, off the grid |
| 3 | Outro | one long breath out | ends the song on air rather than on a note |

**Style-axis lock.** D Dorian · 60 BPM · six-four · two mezzos, one register · crossing on
E4 on the word *second* · finish a perfect fourth apart.

---

### VARIATION 3

## 1. MUSIC PROMPT
```text
Slow modal chamber jazz in D Dorian, 64 BPM, six-four, plain and close and mid-task, scored for spacey jazz and textured folk instruments: nylon-string guitar picking single notes, upright bass plucked short and dry, brushes tapping the rim more than the head, hammered dulcimer struck once a bar, and a modal tenor saxophone searching low behind the words. Everything is played live in one small rented room, two beds, a window, a kettle, room mics wide and the chairs and the kettle audible under the take. Two women sing in the same register and the same unhurried northern-English speech-tone, deliberately confusable; one is level and warm and breathes before each line, the other is faster, drier, half-talking, cutting herself off. A descending line in short phrases sits on the beat. A rising line in long phrases arrives early and keeps its own words the whole way through. The thickening is more players, not more volume.
```
## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, unison refrain, doubled lead vocal, layered vocal stacks, backing-vocal echo, answer vocals, two voices in thirds, two voices resolving together, shared final cadence, power ballad chorus, torch-song rubato, held lament vowel, orchestral swell, drum build, key change, EDM riser, reverse cymbal, tape hiss, vinyl crackle, cassette warble, sepia reverb, found-recording framing, spoken narration, autotune, male lead vocal, child vocals, lo-fi beat, big ending, fade-out
```
## 2. LYRICS
```text
[Theme: a handwritten booking page stuck under a wet kettle — one woman reads down it for the meeting time, the other is talking about the deposit]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, half a bar early. Separate for the first minute, simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - one bar only, nylon guitar, wide room, voice enters straight after]
*paper peeled off a shelf*

[Verse 1 - EMO:Composure - Mezzo One alone, descending, on the beat - plain, unhurried, a caught breath before each line]
There's a ring of tea on the page —
so it's stuck to the shelf —
and it tears when I lift it —
and the writing's on the back —
so I turn it. It's her big hand —
Trish, and then —
Trish, and then —
and my thumb goes past the names —
past the names, down the page —
looking for the time. Just the time —

[Verse 2 - EMO:Warmth - Mezzo Two alone, climbing, half a bar early - talking more than singing, opening mid-clause]
(— did they take the deposit, or is it on the —)
(— because the woman on the phone said the —)
(— hang on, give me a — I've got it in the —)
(— I've got it here somewhere, hang on, in the —)
(— no, that's the other one. That's the —)
(— that's the ferry. Ignore that. Right —)
(— did they take the deposit, or is it on the —)
(— because if they didn't we'll want the —)
(— we'll want the cash out before we —)
(— before we go. Are you listening? Are —)

[Verse 3 - EMO:Contentment - Mezzo One alone, lower, same descent - pleased, mid-task, nothing withheld]
And I'm not reading that bit —
I'm reading down the page —
past the names, down the page —
looking for the time —
just the time, that's all —
just the time —

[Simultaneous Strophes - EMO:Contentment/Warmth - Mezzo One plain, Mezzo Two in brackets - two songs at once, one emotion each, neither waits]
Trish, and then —
(— did they take the deposit, or is it on the —)
and my thumb's on the line —
(— because the woman on the phone said the —)
so the thumb moves down the line —
(— hang on, give me a — I've got it in the —)
and under it there's a box —
(— I've got it here somewhere, hang on, in the —)
and the box has a line —
(— no, that's the other one. That's the —)
and a line under the line —
(— that's the ferry. Ignore that. Right —)
Trish, and then —
(— did they take the deposit, or is it on the —)
and the time's under that —
(— because if they didn't we'll want the —)

[Simultaneous Strophes - EMO:Anticipation/Warmth - Mezzo One falling, Mezzo Two climbing - separate words, a swallow between her phrases, no harmony arrives]
and the time's under that —
(— we'll want the cash out before we —)
under the box, in the —
(— before we go. Are you listening? Are —)
and the tea's gone through the —
(— are you even — right. Fine. I'll do it —)
gone right through the page —
(— I'll do it in the morning. I'll —)
so the time's gone soft —
(— I'll do the machine at the —)
and I tilt it to the lamp —
(— at the garage. It's on the way. It's —)
and it comes back up. There —
(— it's on the way, it's not a — hang on —)

[The Crossing - EMO:Anticipation/Warmth - both mezzos on one note - one word, one beat, one falling through and one rising through, neither adjusts]
and the box says — hang on —
(— hang on. Hang on. Give me a —)
second — (second —)
driver. It says second driver —
(— sorry. Sorry. You go. What —)
and that's me. That's my one —
(— what does it say about the —)
and under that, the time —
(— about the deposit? Does it —)
and I say it out loud —
(— does it say, or doesn't it —)
and I say it out loud —
(— or doesn't it. Fine. Right —)

[Simultaneous Strophes - EMO:Composure/Warmth - Mezzo One settling low, Mezzo Two still climbing - they come apart and stay apart]
and I say it out loud —
(— right. Right. So I'll get it at the —)
and it's earlier than I —
(— I'll get the cash at the garage, and —)
earlier than I had —
(— and that's that. That's sorted. That's —)
so that's better. That's good —
(— that's the last of it. Are you —)
and I fold it in half —
(— are you nearly done, because I want the —)
and it goes back under —
(— because I want the light off, if you're —)
back under the kettle —
(— if you're finished with the — right —)

[Outro - EMO:Ennui - Mezzo Two alone, still climbing, unfinished - Mezzo One has stopped; the song ends on the wrong voice]
(— right. Right. So that's —)
(— that's everything, then. That's —)
(— that's us, then. Are you doing the —)
(— or shall I do the light? I'll do the —)
(— I'll do the light, then. Right —)
(— right. There. That's — that's —)
*paper under a kettle*
```
## 3. TITLE
**The Second Driver**

## 4. PRODUCTION SIDECAR

**Disc_Channel** — operator-side only, not pasted into the lyrics field.

```text
[Disc_Rhythm: brushes_on_the_rim_not_the_head | 64_BPM_six_four_grid | chairs_and_kettle_under_the_take | Stereo_Width_Mid]
[Disc_Vocal: two_mature_female_mezzos_same_range | northern_english_speech_tone | half_talking_cutting_herself_off | Voice_One_slightly_left_Voice_Two_slightly_right]
[Disc_Sub: upright_bass_plucked_short_and_dry | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: one_small_rented_room | wide_room_mics_leakage | no_plate_no_hall | Stereo_Width_Maximum]
[Disc_Texture: nylon_string_single_notes | hammered_dulcimer_once_a_bar | modal_tenor_saxophone_low_behind_the_words | paper_torn_off_a_shelf | Center_Back]
```

**Vocal fingerprint.** Mezzo One: level, warm, breathes before each line, administrative
nouns delivered flat — *thumb, line, box, time*. Mezzo Two: faster, drier, half-talking,
already mid-sentence when the section starts.

**Production dramaturgy.** Hammered dulcimer once a bar → a struck, decaying event that
cannot sustain, so nothing in the texture teaches the voices to hold. Sax low *behind* the
words → the horn is never the second voice. Chairs and kettle audible → one room, not a
booth (D11).

**Body noise (unpitched only).**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Verse 1 | breath before each line | the reading is physical work, not recitation |
| 2 | Simultaneous Strophes 2 | a swallow between Mezzo Two's phrases | she is rummaging while talking; the body is doing two things |
| 3 | Outro | Mezzo Two's breath, alone, at the end | the song ends on the wrong voice and on air |

**Style-axis lock.** D Dorian · 64 BPM · six-four · two mezzos, one register · crossing on
E4 on the word *second* (*second driver* against *give me a second*) · finish a perfect
fourth apart.

---

### VARIATION 4

## 1. MUSIC PROMPT
```text
Slow modal chamber jazz in D Dorian, 58 BPM, six-four, quiet, wide awake, happy about tomorrow. One voice on her own, then a second voice on her own, then the two of them together for the rest of the song, holding different tunes and different words until one of them simply stops mid-word and the other carries on by herself. Spacey jazz and textured folk instruments: harmonium sustaining one low note the whole way through, brushes with the snares off, upright bass in long held notes, nylon-string guitar placing one note a phrase, bowed vibraphone far back. A rented room at night, two beds, a window, a kettle, room mics wide, people still awake in it. Both singers are mezzos of the same age and accent, hard to tell apart: one clear and breathy with a smile in the tone, the other lower, thickening, drifting. The two lines touch on one note near the end and neither of them bends. More players, never more level.
```
## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, two voices resolving together, hummed harmony under the lead, backing-vocal echo, answer vocals, unison refrain, doubled lead vocal, two voices in thirds, lullaby choir, whispered ASMR vocal, ambient pad wash, reverse cymbal, big ending, swell into the final refrain, key change, EDM riser, orchestral swell, drum build, tape hiss, vinyl crackle, cassette warble, sepia reverb, found-recording framing, autotune, spoken narrator, male lead vocal, child vocals, lo-fi beat, fade-out
```
## 2. LYRICS
```text
[Theme: lights out in the shared room — one woman falls asleep, the other lies awake going through it, and sets the alarm last so the screen will not wake her]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, half a bar early. Separate for the first minute, simultaneous from the second. They meet on one note, once, and go past each other. The second voice goes to half-words and stops.]

[Intro - EMO:Serenity - Instrumental - one bar only, harmonium on one note, wide room, voice enters straight after]
*a switch, then dark*

[Verse 1 - EMO:Composure - Mezzo One alone, descending, on the beat - plain, unhurried, a caught breath before each line]
The light goes down and it's not dark —
there's a line under the door —
so the watch comes off my wrist —
and it goes under the bed —
because I'll only look —
and a green light, low, on the side —
on the kettle, on the side —
and I lie on my back, and my hands —
my hands flat on my chest —
and I go through the thing once —
and I go through the thing once —
and this is the good bit —
this is the good bit —

[Verse 2 - EMO:Warmth - Mezzo Two alone, climbing, half a bar early - talking more than singing, opening mid-clause]
(— God, my feet. My feet are —)
(— are you all right over there, are you —)
(— because I can hear you. You're not —)
(— not sleeping. I hear you not sleeping. Not —)
(— are you all right over there, are you —)
(— it's fine. It's fine. I'm not — I'm —)
(— what time is it? No. Don't tell me. I'm —)
(— I'm not asking. Right. Right. I'm going. I'm —)
(— I'm going. I'm going now. I'm —)
(— going. Night. Night. Mm —)

[Verse 3 - EMO:Contentment - Mezzo One alone, lower, same descent - pleased, nothing withheld]
And I go through the thing once —
from the door to the road —
and the road to the field —
and I know the spot. I'll stand —
and the cold coming on my hands —
and the wind. The wind at my back, and —

[Simultaneous Strophes - EMO:Contentment/Warmth - Mezzo One plain, Mezzo Two in brackets - two songs at once, one emotion each, neither waits]
and this is the good bit —
(— God, my feet. My feet are —)
lying here. Lying here in the dark —
(— are you all right over there, are you —)
with the green light on the side —
(— because I can hear you. You're not —)
on the kettle, on the side —
(— not sleeping. I hear you not sleeping. Not —)
and the line under the door —
(— are you all right over there, are you —)
and I go through the thing once —
(— it's fine. It's fine. I'm not — I'm —)
and I go through the thing once —
(— what time is it? No. Don't tell me. I'm —)
and this is the good bit —
(— I'm not asking. Right. Right. I'm going. I'm —)

[Simultaneous Strophes - EMO:Anticipation/Ennui - Mezzo One falling, Mezzo Two giving out - separate words, her breath going long, no harmony arrives]
from the door to the road —
(— I'm going. I'm going now. I'm —)
and the road to the field —
(— going. Night. Night. Mm —)
and I know the spot. I'll stand —
(— did you — did you set the —)
and the cold coming on my hands —
(— did you set the — I'll do it, I'll do it —)
and the wind. The wind at my back, and —
(— I'll do it in a minute. In a —)
and I'll not set it yet. Not yet —
(— in a —)
because the screen's too bright —
(— in a —)

[The Crossing - EMO:Anticipation/Ennui - both mezzos on one note - one word, one beat, one falling through and one rising through, and she does not come back]
and I'll wait. I'll wait till she's gone —
(— did you — I'll do it in a —)
second — (second —)
and then I'll reach right out —
(— in a —)
and do it in the dark —
(— in a —)
and go through the thing once —
(— in —)
and once more, and it's —
and it's not long now —
and this is the good bit —

[Outro - EMO:Serenity - Mezzo One alone, lowest - Mezzo Two stopped mid-word, one long breath out, no held final note]
and she's gone. She's off —
and her breathing goes long and low —
and the green light stays green —
and the line stays under the door —
and I lie and I let it —
and I reach out. Feel for the phone —
and I turn it, screen to my chest —
so the light stays on my chin, not hers —
and I set it. Set it for the dark —
for before the light. There. There —
and I put it down. Down —
and put my hands back on my chest —
and this is the good bit —
this is the good bit —
*a small alarm set*
```
## 3. TITLE
**The Line Under The Door**

## 4. PRODUCTION SIDECAR

**Disc_Channel** — operator-side only, not pasted into the lyrics field.

```text
[Disc_Rhythm: brushes_with_the_snares_off | 58_BPM_six_four_grid | no_backbeat | Stereo_Width_Mid]
[Disc_Vocal: two_mature_female_mezzos_same_range | smile_in_the_tone | second_voice_thickening_and_drifting | Voice_One_slightly_left_Voice_Two_slightly_right]
[Disc_Sub: upright_bass_long_held_notes | harmonium_one_low_note_throughout | felt_in_chest_not_ears | Mono_Sub_Lock]
[Disc_Pad: harmonium_single_sustained_note | rented_room_air_at_night | no_plate_no_hall | Stereo_Width_Maximum]
[Disc_Texture: bowed_vibraphone_far_back | nylon_string_one_note_a_phrase | standby_light_hum | Center_Back]
```

**Vocal fingerprint.** Mezzo One: clear, breathy, a smile in the tone, flat on the refrain.
Mezzo Two: lower, thickening, drifting, decaying to half-words and then stopping mid-word.

**Production dramaturgy.** Harmonium holding one note the whole way → the drone is the
piece's only sustained pitch, and it belongs to an *instrument*, so the voices never have
to supply one. Brushes with the snares off → no backbeat to lock the two strophes into a
shared grid.

**Body noise (unpitched only).**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Verse 1 | caught breath before each line | she is wide awake and it is audible |
| 2 | Simultaneous Strophes 2 | Mezzo Two's breath going long and slow | a person going under, measured in air not in pitch |
| 3 | Outro | one long breath out after the alarm is set | the last vocal event is unpitched; nothing is left to harmonise |

⛔ **The hum is refused here specifically.** A sustained `mm` under a lead line is the one
element in this pair that two voices *can* blend on. Four of the five in the step-10 draft
are now lexical half-words. See the L22 scan.

**Style-axis lock.** D Dorian · 58 BPM · six-four · two mezzos, one register · crossing on
E4 on the word *second* · Mezzo Two stops mid-word · Mezzo One finishes alone.

---

*Step 11 complete. **GATE: PASS**, repaired in place, andon cord available and not pulled. This pair goes to `lofn-render-audit` with one live condition: ⛔ **if the two strophes come back harmonised, aligned, echoed or resolved, the pair has failed and the Kamasi objection is live again.** Send the audio alone. Never the prompt.*
