# P02 · STEP 09 — ARTIST REFINEMENT
**Run:** `2026-08-15_daily_music_the_resolution` · **ICB sha:** `7d4dd66f2a6e`
**Pair:** P02 — *The water took the second one* · Bed **B3** · **AWE, terror-adjacent**

**TRI-SOURCE.** **S1** the absent comet and its copy at ankle height · **S2** *"steamy"* / *"so steamy you'll wish your speakers had a defogger"*, physical · **S3** ONE REFLECTION, section scale, literal.

---

## THE FOUR TITLES, LOCKED

| | title | why it passes the title gate |
|---|---|---|
| **V1** | **Ankle Deep At Two In The Morning** | a physical position and a time; no banned noun; no *dark*, no *sky*, no *current*; no persona prefix |
| **V2** | **The Tide Came Up My Shins** | a concrete event on a body |
| **V3** | **I Put My Hand Flat On The Water** | a first-person act; P02 owns *water* |
| **V4** | **My Breath Kept Getting In The Way** | a first-person act, and the pair's second leftover object |

⛔ Rejected titles and why: *"Four Lights, One Of Them A Copy"* — names the argument, and spends the pair's single reflection description in the title. *"The Cold Line"* — abstract. *"What The Water Kept"* — good, but it repeats the pair name's grammar and would blur V1 and V2.

---

## REFINEMENT DECISIONS — TAKEN AGAINST THE STEP-08 FLAGS

### F-A · LINE LENGTH — **partially discharged, and the residue is declared**
Mirrored quatrains with chiasmus lengthen lines by construction: line 3 must restate line 2's clauses in reverse, which costs words that a non-mirrored line does not spend. The lyrics were tightened toward monosyllables (*stone, silt, shin, calf, wrist, cold, flat, breath, gone*) and away from Latinate abstraction, and every stanza now contains at least one line under six words.
⚠️ **DECLARED FLAG, carried to step 10 and measured there:** `mean_words_per_line` will sit near or above the 7.5 ceiling in some variations. It is a FLAG, never a fail, and it is bought with a very high line-return and a very high end-rhyme — **the return floors are the thing the ceiling exists to protect, and this pair overshoots them, it does not dodge them.** Reported as a number, not as an excuse.

### F-B · "NOBODY SENT ME DOWN TO YOU" — **kept, and quarantined**
The line is the CHOSEN clause and it is stated as logistics. It is protected by a hard local rule now enforced across all four lyrics: **no line within a stanza of it may mention another person, an audience, a message, or a lack.** V4 gets the same clause as a joke instead (*"Nobody asked me down. I asked myself."*), which is the safest version of it in the pair.

### F-C · THE SINGLE REFLECTION DESCRIPTION — **assigned, one per lyric, and none of them uses the word**
- **V1:** *"Four came down and one was not / a thing at all — I could not tell."*
- **V2:** *"you had both of them inside your face … and the place / they were was flat."*
- **V3:** *"There were four of them and one was air, / was air and looked like fire all the same."* ⛔ The step-08 draft reused V1's four-light stanza verbatim here; that was caught in this pass and **rewritten**, because a byte-identical stanza across two variations is pair-internal monoculture wearing a different title.
- **V4:** **none.** V4 spends its mirror budget on breath instead — the warm copy of the body standing in the air where the light was. That is a leftover, not a reflection, and it keeps V4 clean.
⛔ V1's *"the hot half of the thing you hid"* was audited and kept: it is about **heat and concealment**, not about copies. Everything else mirroring in all four lyrics is **structural** — the ABBA closes on its own sound and the ghost section runs backwards.

### F-D · THE STEAM RHYME BETWEEN THE TWO SURFACES — **kept, deliberately**
*"steam"* is in V4's lyric and *"steamy"* is case-exact in all four style fields. This is the one place the borrowed vocabulary and the sung vocabulary touch, and it is on purpose: the phrase enters the record as **weather on a cold night**, not as a genre marker. Confirmed at step 10 that every style-field instance is byte-exact and that the long phrase is unaltered.

### F-E · WORD-BOUNDARY SWEEPS — **scheduled, not assumed**
`sky`, `current`, `dark`, and the house-lexicon list are swept at step 10 **with `(?<![a-z])token(?![a-z])`**, because a substring sweep for `sky` false-positives on nothing here but a substring sweep for `dark` would miss nothing and a substring sweep for `current` would flag `currents`/`concurrent` — and a gate that cannot tell a word from a substring manufactures findings.

---

## THE GHOST RETURN — TIGHTENED FROM "AN EFFECT" TO "A PERFORMANCE"

The step-08 draft left the ghost return as a production note. It is now **typed into the lyrics field backwards**, as its own section, in all four variations:

- **V1** returns *Shallows* + *Chorus*, line order reversed.
- **V2** returns *Chorus* + *The Height*, line order reversed.
- **V3** returns *Chorus* + *Flat*, line order reversed.
- **V4** returns *Chorus* + *Breath*, line order reversed.

Its header carries the whole instruction where a renderer will actually read it: **lower level · moved left · one beat behind · inverted.** ⭐ This is the pair's answer to *"a Somatic objection answered only in the PRODUCTION SPEC is not answered."* The reflection is in the lyric, so even a generator that ignores every mix instruction still performs a section that runs backwards.

---

## THE ANTI-PAD PASS — WHAT CHANGED IN THE STYLE FIELDS

The step-08 prompts described the bed correctly and would still have rendered as a pad, because *"chamber-drone, bowed strings, harp, low sine"* is four words from *"cinematic ambient"*. Four repairs, applied to all four:

1. **The harp got a verb and an object.** *Damped with a fingertip* · *played with a coin* · *prepared with paper woven through the strings* · *stopped at the soundboard and plucked near the pins*. A harp with a technique attached cannot default to a glissando.
2. **The reverb got a source.** The atmosphere is named as **the convolution tail of the harp and the voice**, so it cannot exist without a performance. This is also where **"so steamy you'll wish your speakers had a defogger"** finally sits honestly — it is a claim about a *wet return*, not about a mood.
3. **The floor got a beat frequency.** The low sine is tuned a hair off the cello so the two flutter a couple of times a second. A drone with an audible beat is not a pad, and it means there is never a long full stop for the generator to smooth.
4. **The groove got a limp.** Every third, fourth or fifth stroke lands a sixteenth off, and the pulse is named as a heel shifting in shallow water rather than as a percussion instrument.

---

## VARIATION-BY-VARIATION REFINEMENT

### V1 — ANKLE DEEP AT TWO IN THE MORNING
Kept the reckoning as the emotional hinge and moved the one sung number into the chorus so it returns three times and **means something different each time** (a count · a fact she is standing inside · a thing she has finished with). Verse 4 was rewritten off the *steam* image, which belongs to V4, and onto **the absent grain** — *"The grain that made it isn't here"* — which is the seed said plainly and is the strongest thing V1 owns. The exit is an action: the tide reaches the calf and she walks out, **no debt, no thing to set aside.**

### V2 — THE TIDE CAME UP MY SHINS
The refinement here was to **stop the song narrating dread and let the measurement narrate it.** The singer's register never changes; only the height does. *"I picked the calf before I picked the road"* moved from the last verse to a returning refrain, so the decision is heard three times and the risk stays chosen. Added *"the cold is just a rate that does not lie"* — the cleanest statement of terror-adjacent AWE in the pair. The bed carries the same irreversibility: **the low sine glides up one semitone and stays there.**

### V3 — I PUT MY HAND FLAT ON THE WATER
The chorus is the most literal chiasmus in the run and it was left untouched. The refinement was **structural**: a four-line hole in the middle (*The Hole*) where the arrangement drops to the room return alone while the sine keeps walking — the gesture, made audible. ⛔ It is **not a silence**; a specified full stop renders as a smear, so the bed carries it. Added *"and free is why"* logic to Verse 4 so the second hand-down is explicitly chosen, and *"I am not afraid / of anything that keeps a time its own"* to hold the fear clean.

### V4 — MY BREATH KEPT GETTING IN THE WAY
Made the breath the **percussion and the reverb send**, not just an image — the room only lights when she breathes, so the arrangement is played by her lungs. Removed the time-of-night so that **V1 remains the only variation with a sung number.** The close, *"Take the second one. You've got it. Keep,"* is the pair's whole posture in five words: the water can have it, she is warm, and she is going home with steam on her.

---

## CROSS-VARIATION DISTINCTIVENESS — CHECKED, NOT ASSUMED
Shared by design (they are one pair): the bed, the apostrophe, ABBA mirrored quatrains, the ghost return device, the seam at the waterline.
**Different in every variation:** the chorus grammar · the section map and section names · the rising element (tide named in body parts · a semitone glide · the cold up the arm · her own steam) · the vocal staging (centre · right-migrating-to-centre · centre-plus-a-late-left-copy · centre-and-very-close) · the harp technique · the kinetic defect's period (fourth · third · fifth · fourth-early) · which variation carries the number (V1 only) · which carries a hole (V3 only) · which carries no reflection description at all (V4 only).
⛔ **No two share a stanza.** There is no byte-identical text across the four lyrics at all — the one place there nearly was (V1's and V3's four-light stanza) was found in this pass and rewritten. Checked at step 10 by measurement rather than by eye.
