# STEP 09 — SOUND DESIGN · P03 · **DON'T LOOK AT IT**
**Bed:** BAILE PHONK — Brazilian baile funk (tamborzão) × dark phonk · **140 BPM · F minor**

> ⚠️ **The token is weak; the lane is not.** `phonk` measured as one of our two weakest style tokens (ban
> not applied — bad denominator). **So this spec names the bed by its MECHANISM, not by the word.** The
> shipped prompt says *tamborzão*, *annoyance-detuned cowbell*, *geothermal trunk-rattling mono sub-808*,
> *Memphis half-time* — every load-bearing element of the lane, and **the word "phonk" appears nowhere in
> the shipped style field.** Use the lane; don't lean on the word.

---

## 1. THE ARGUMENT FOR THE PALETTE

The song is set on a fabrication job. **Every percussive element in it is a tool the character actually
holds.** That is not colour — it is the ONE SEAM discipline: the piece is made **OF** the material, not
**ABOUT** it. There is no "industrial texture layer." There is a contactor, a grinder, and a chipping
hammer, and they are the drum kit's top end because that is what her afternoon sounds like.

## 2. ⭐ SIGNATURE STYLE 1 — **ARC-STRIKE PERCUSSION**

*The two deeply described genre-specific signatures required by the personality's override rules; this is
the first.*

| Element | Spec |
|---|---|
| **Source** | A contactor **clack** (the relay thump when the welder strikes), a **grinder squeal** (2–4 kHz, short), a **chipping hammer** tick. |
| **Treatment** | Chopped to tight sixteenths, gated hard, **hard-panned micro-edits** alternating L/R on the off-sixteenths. |
| **Job** | It replaces the hi-hat entirely. ⛔ **No generic trap hats** — the whole top end is tools. |
| **Doing** | **Impatience**, not decoration. It should read as somebody who wants to get down the ladder. |
| **Grain-law note** | Micro-edits survive the generator; a "field recording of a workshop" gets smoothed into mud. Ask for **rhythm**, not ambience. |

## 3. ⭐ SIGNATURE STYLE 2 — **BALLAST HUM BASSLINE**

| Element | Spec |
|---|---|
| **Source** | A **40 Hz mains-drone sine** — the hum of the sign's own ballast, the sound the finished object makes forever. |
| **Layer** | Under it, ⭐ *"a **superhumanly funky bass**"* that spends the break ⭐ *"**furiously slapping and popping**"* (SOURCE-2, verbatim, shipped). |
| **Placement** | **Mono, chest-first.** Somatic Bass territory (30–60 Hz) — the assignment says *demand to be felt physically*, and the drone is the lowest, dullest, most patient thing in the mix. |
| **Why it matters** | ⭐ **The drone is the sign. The slap is the hand.** They share a register and never share a rhythm. That is THE SECOND EYE stated in the bass: two versions of one object, occupying one frequency band, refusing to lock. |

⛔ **A production-spec-only answer is not an answer** (L22). The bass makes the argument audible; the lyric
and the form make it *true* (see `step07` §2). The spec is the reinforcement, never the delivery.

## 4. THE FUNK POSTURE, AND THE DEBT IT CREATES

The slap-and-pop lead is **deliberate**, not decorative: the electric bass is the instrument whose players
are most reliably the one in the room and not the name anyone says. A song about uncredited hands **carried
by that posture** is the thesis performed in the arrangement rather than argued in the lyric.

⛔ **Therefore the credit is owed and is paid in `step10`:** **ANDRÉ CYMONE** and **BOOTSY COLLINS** by name,
the **tamborzão / baile funk** producers of Rio and São Paulo whose beat this is, and **TEDDY RILEY** for the
new jack swing shuffle. ⛔ Never in a Suno field, never in a lyric.
**Honesty clause:** Suno generates, it does not sample — *we asked for the figure by name.*

## 5. ARRANGEMENT BY SECTION

| Section | Kick/Perc | Bass | Vocal | Space |
|---|---|---|---|---|
| **Hook (bar 1, cold)** | Full tamborzão + detuned cowbell | Sub-808 + drone | Compressed bratty snarl | Wide crowd ghosts |
| **Verse 1–3** | **Beat halves** — kick + cowbell only, arc-strike top | Drone only | Dry close-mic, flat, unbothered | Narrow, near |
| **Pre-Hook** | Building, chipping hammer doubles | ⭐ Slap bass enters | Contempt, edge on | Opening |
| **THE INTERRUPTION** | **Drops out mid-bar** | Cut | Overwritten by gang | ⭐ 2-bit crush → **one beat of true silence** |
| **SECOND EYE** | ⭐ **HALF-TIME**, *a splash of vintage new jack swing* | Drone + soft fingered line | Held, open, the only warm vocal | Widest — reverb is *rain on a road*, not a hall |
| **Bass Break** | Gang stomps on the two | ⭐ **Furiously slapping and popping**, exposed | Short bratty calls + crowd | Dry, tight |
| **Final Hook** | Maximal | Maximal | ⭐ Lead steps **out** after the inversion | Full |
| **Tag** | Cowbell + crowd only | Drone only | ⭐ **Lead vocal absent** | Crowd, alone |
| **Outro** | — | — | Spoken, off-mic | Room tone |

⭐ **STEREO NOTE (L22, render-measured):** let the field **narrow** into the final hook rather than widen.
Deviation observed on past renders and now a technique — it makes the handover to the crowd feel like being
pushed out of the frame rather than lifted into it, which is the subject.

## 6. VOCAL

**Female, late twenties. Dry, close-mic'd, no reverb in the verses.**
- **Verses:** flat, procedural, *unbothered*. She is reading a list. ⛔ No emoting. The flatness is the
  indignation — the bratty move is refusing to perform hurt.
- **Hook:** compressed bratty snarl, hard consonant onsets hammered (`D`, `H`, `D`, `S`).
- **SECOND EYE:** the only section where the voice opens. Not sweet — *held*.
- **Tag:** ⛔ **gone.** Crowd only.
- ⛔ No child voices. Female lead throughout.

## 7. STYLE-REDUCTION (the exclude field)
⛔ generic trap hi-hats · EDM risers · airhorns · child voices · orchestral swells · ambient intro ·
lo-fi vinyl-crackle bed · male lead vocal.

## 8. WHAT WOULD MAKE THIS RENDER FAIL, AND HOW THE SPEC AVOIDS IT
| Failure mode | Guard |
|---|---|
| The cold open gets a generated intro anyway | `[SONG FORM]` says it, the Hook header says **COLD OPEN ON BAR ONE**, and the prompt's first clause says it. Three places. |
| The one-beat silence gets filled | It is the **only** silence in the piece. Long stops get smoothed; a single beat survives. |
| The crowd tag gets a lead vocal put back on it | Stated in the prompt (*"crowd answers alone on the tag"*), the lyric header, and §5 above. |
| "Baile phonk" returns a gym-core cliché | The word is **not in the shipped field.** The mechanism is. |
| The half-time turn becomes a tempo change | Specified as *half-time*, not a modulation. |
