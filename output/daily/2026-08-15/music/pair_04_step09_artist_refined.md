# P04 — STEP 09 · ARTIST REFINEMENT PASS
**Run:** `2026-08-15_daily_music_the_resolution` · **ICB sha:** `7d4dd66f2a6e`
**Pair:** P04 · Bed **B6** · INDIGNATION full · **DOES NOT RESOLVE**

This is the adversarial pass over the step-08 drafts, before the packages are assembled. **Nine defects found, all repaired.** Two of them would have failed a hard gate.

---

## D1 — ⛔ HARD GATE BREACH: A NUMERAL SURVIVED IN A SUNG LINE (V4)
Draft V4 carried *"the same eleven minutes late"*. The ICB permits **one** numeric fact sung, at the hinge, spelled out — and this pair spends that budget on **"one person came"** (V3's hinge, and its equivalent in every variation). A second numeral is a breach whether or not it is spelled out.
**Repair:** all incidental numerals removed from every sung line. V4 reads *"same wrong hour"*. The step-10 SELF-CHECK runs a **word-boundary numeral sweep** across every lyric — `(?<![a-z])(one|two|three|…|hundred)(?![a-z])` — and reports the hit list, not a tick.
**Kept deliberately:** the single hinge fact, spelled out in words, responded to and never recited.

## D2 — ⛔ SCALE-GATE ADJACENCY AUDIT ON *"AMP LIGHT ON"*
*"Amp light on"* sits one word away from the banned field. It was audited rather than reflexively cut, and it is **kept in V1** — because it is the **opposite** of the banned image. A lit power LED is a positive assertion that **the electricity is fine, the gear is fine, nothing is broken.** The scale gate bans the *deprivation* image; this line is the anti-image, and the song's engine (F4) is precisely that nothing has failed.
**Repair applied anyway:** the surrounding lines were tightened so the assertion is unmistakable — *nothing broken / nothing broken here* immediately follows. And the word **"dark"** appears nowhere in any title or lyric across all four variations (swept in step 10).

## D3 — LINE-LENGTH DRIFT OUT OF THE 3–5 SYLLABLE FORM
Drafts contained *"Good."* (one syllable), *"It doesn't care / who's in it"* (a spilled clause), and *"bigger than I sent it"* (six). The fragmented form is not decoration — it is the sound of a tired person saying short things while her hands work. A spilled line reads as composure and composure is a lie tonight.
**Repair:** every line re-cut to three-to-five syllables. *"Good."* → *"That's better."* → finally *"Don't clap for that."* The step-10 SELF-CHECK **measures the syllable distribution per variation with a script** and reports the out-of-band count as a number, with the offending lines quoted.

## D4 — END RHYME CREPT BACK IN
Draft V2 had *"…than it got"* four lines above *"…that's the part"* — a soft consonantal landing, and V1 had *"in it" / "minute"*. Stripping end rhyme is the discharged debt of this pair; letting it back in un-pays it.
**Repair:** final words re-chosen so that no two line-endings within ±4 lines share a rime, **except exact repeated lines (epizeuxis), which are returns and not rhymes.** ⭐ Noted for the measurement: `strict_end_rhyme()` compares the last three characters, so **this pair will SCORE rhyme it does not have** — its number is to be read as **end-of-line RETURN**, and the rhyme floor and the line-return floor are **not independent evidence** for P04.

## D5 — THE ACCUSATION WAS AMBIGUOUS ABOUT ITS TARGET
In the draft, chorus 2's *"don't play deaf"* could be heard as thrown at a passerby or at the absent audience. That reading turns the song into a complaint about other people and **hands it a villain**, which is banned.
**Repair:** the self-address bridge now lands **before** chorus 2 in all four variations and answers the question explicitly — *who's the you / nobody's here / the you is me*. Every second person in the song is retroactively re-heard as hers the moment that bridge lands. The bridge is a section, not a line, so it cannot be trimmed away in refinement.

## D6 — SHE WAS DRIFTING TOWARD SECRETLY EXCELLENT
Two draft lines let a listener imagine she was actually great and merely unlucky: *"this room could hold a real one"* (implies she nearly is one) and a discarded line about the room "deserving better material". Both flatter her sideways. **F1 says a brilliant unheard artist flatters everyone who hears the song, and that is the vanity the panel cut.**
**Repair:** *"This room could hold / a real one. / Holds me instead."* is kept **only** because its second half is a flat self-demotion, and the surrounding verse now carries a concrete, unglamorous error — a missed change, an early hand, a bad loop kept in because stopping costs more than continuing. **Errors are named as errors, in the first person, without excuse and without tragedy.** No line anywhere says or implies she is good.

## D7 — THE VISITOR WAS BEING TURNED INTO A LESSON
Draft V3's ending drifted toward *she was enough* — a grace note by stealth, and the single most likely way this song resolves without anyone noticing.
**Repair:** the two valuations are stated **adjacent, flat, unranked, and abandoned**: *that is enough / that is not enough / both are true / I won't choose.* The refusal is spoken once and never revisited. She is never thanked. She leaves before the case is shut. ⭐ **The song's ending does not know what she was worth, and neither do I, and that is the deliverable.**

## D8 — THE TWO FLASHES LIVED ONLY IN THE PRODUCTION SPEC
The drafts described two hard stabs inside one bar in the prompt, and the lyric did not carry the gesture. **THE GRAIN LAW: a somatic objection answered only in the PRODUCTION SPEC is not answered.**
**Repair:** **epizeuxis is now explicitly the two-flash at word scale** — every doubled line is the pair of stabs, said. The doubles are placed on the *stressed* half of the bar and the `[SONG FORM:]` header states the correspondence outright, so the renderer sees it in the lyrics field as well as the style field. The doubling density rises with the stab density across the arc.

## D9 — THE OUTRO COULD BE MISTAKEN FOR EXHAUSTION
*"Coil it tight / back here tomorrow"* read as resignation on a cold pass. Resignation is a resolution wearing work clothes.
**Repair:** the outros were re-cut so the **delivery accelerates while the pitch falls** — more lines, shorter, harder — and each ends on a statement that closes nothing:
- **V1:** *nothing about this is fine.*
- **V2:** *it doesn't care who's in it.*
- **V3:** *sound went with her.*
- **V4:** *play it worse.*
⛔ None of the four is a lesson, an acceptance, a thank-you, or a lift. **All four end louder and angrier than the first bar.**

---

## SOURCE-2 PLACEMENT, FIXED HERE SO IT CANNOT DRIFT AT STEP 10
Both required phrases go in the **generator-facing MUSIC PROMPT**, **case-exact**, describing **the corridor's own reverb**, against type:
- **`downright salacious slow jam`** — applied to the room tail, never to the groove (the groove is 140 BPM industrial grief).
- **`steamy`** — applied to the bloom on consonants, never to the vocal.
**The prettiness of the room is the insult**, so both phrases must sit next to a word that wastes them (*wasted*, *empty*, *nobody*). The step-10 SELF-CHECK quotes both with their location and confirms they are in the prompt field, not the design doc.

## CROSS-VARIATION SIMILARITY (checked before assembly)
Section maps differ (V2 has a room-alone break and no pre-chorus; V4 has a coda; V3's hinge sits dead centre). Lexical fields are disjoint by design — signal path / architecture / another body / habit. **The only verbatim material shared across the four is the refrain line**, which the rhyme debt requires. Chorus surrounds, verses, bridges, outros and SFX cues are written separately per variation.

## MEASUREMENT PLAN CARRIED INTO STEP 10
Everything below is measured **in python**, never by eye, and reported as numbers with FLAGS raised rather than ticked:
prompt chars (850–1000, target 870–960, ≥985 FLAG) · terminal punctuation · exclude chars (400–900) · lyrics field chars (<5000, target ≤4800) · sung line count (70–120, ≤72 FLAG) · exact count of *"you can hear me fine"* (**must equal 3**) · word-boundary sweep for the banned scale-gate lexicon · word-boundary numeral sweep · house-lexicon sweep · banned-opener check · negation-in-style check · syllables-per-line distribution · repeated-line ratio · dedication chars (≤500).
