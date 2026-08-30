# STEP 10 — PRODUCTION WRAP · P06 · THE CALL ACROSS THE VALLEY
**Run:** `2026-08-13_daily_lofn-prime` · **Pair 6** · AMBITIOUS · EXISTENCE · **LOFN-PRIME, AWE**
**Status:** four packages complete, all gates measured, ⛔ **drafts on disk — nothing rendered, nothing published** (`vault/AUTONOMY.md`).

---

## 1. THE SHIPMENT

| | Title | Angle | Chant | Runtime |
|---|---|---|---|---|
| **V1** | **Kom Hem, Silja** | the caller | `Kom hem, Silja!` | ~4:15 |
| **V2** | **The Older Throat** | the woman who answers | `I hear you, Marit!` | ~4:00 |
| **V3** | **The Burr** | the technique, taught | `More rough, more road!` | ~3:50 |
| **V4** | **The Air Lies Low At Dusk** | the carrier itself | `Lie low, cold air!` | ~4:30 |

Full packages, exact shipped text and per-song measurements: **`variations.md`.**
Flagship lyric with craft notes: **`step08_lyrics_draft.md`.**

**The obscure emotion, named by this pair and ⛔ never in a prompt: THE ROUGH-DEBT** — *the embarrassed
gratitude you owe the part of yourself you were taught to sand off, on the day you find out it was the
only part that ever reached anyone.*

**The device in four words: NAMES INSTEAD OF NUMBERS.**

---

## 2. FLAGSHIP PACKAGE — V1 "KOM HEM, SILJA"

### 2.1 MUSIC PROMPT (Suno style field) — 955 chars
```
A Swedish herding call held over a body-felt floor; the four-octave gap between them is the arrangement. Female vocalist, late twenties: open-throated kulning, unvibratoed head tone around eight hundred and eighty hertz, neutral third, a fourth pitched slightly high, free time, no metre, breath audible. Under her a superhumanly funky bass parked on the same note four octaves down at fifty-five hertz, sixty-six BPM felt not counted, furiously slapping and popping at the floor of hearing so it lands in the sternum instead of the ear, mono and centred. Signature 1: THE SCOOPED VALLEY, two hundred to six hundred hertz cut almost to silence so the mids are open ground the call has to cross, inhabited only when a distant second woman answers from the far side. Signature 2: THE MILK RING, a close-miked pail of milk trembling at forty hertz, the sub made visible. Gang chant with swagger, humor, and sheer danceability. No reverb tail reading as echo.
```

### 2.2 EXCLUDE (Suno second field)
```
male lead vocals, child vocals, vibrato, pop belting, trap hi-hats, EDM risers, four-on-the-floor kick, guitar, brass, orchestral strings, choir pads, glitch, bitcrush, chiptune, breakcore, distortion, autotune, whispered ASMR, long reverb tails, slapback echo on the lead, stereo-widened bass, untuned drone, sampled field recording of a real singer
```

### 2.3 LYRICS
Exact shipped text in **`step08_lyrics_draft.md` §1** — single source of truth, not duplicated here.

### 2.4 SONG GUIDE (for the render operator)
- **Open bare.** No floor under the first call. The sub's entry at `[Call I]` is the second body arriving.
- **`[The Pail]` is a lead removal, not a silence.** Keep the sub running. A true stop gets smoothed.
- **The mids open once**, at `[Call II — The Far Side]`, and close again. That single opening is the seam.
- **The voice must degrade.** Clean at the intro, fraying by `[Loop Until It Hurts]`. Do not pitch-correct
  the frayed calls — that is the song being sanded smooth, which is the thing it argues against.
- **Sub stays mono and centred**, on the same pitch class as the call.
- **If a take resolves the call and the sub into each other** — the voice descending, the bass rising into
  the mids — reject it. The gap is the subject.

---

## 3. ⭐ LINEAGE & CREDIT
*(`vault/DEDICATION_STANDARD.md`, first run under it. ⛔ No name below appears in a Suno field or a lyric.)*

### 3.1 The tradition
**Kulning** (Swedish; the related Norwegian calls are *lokk* / *laling*) is a high, open-throated herding
call developed by women working the summer *fäbod* shielings of **Dalarna, Härjedalen and Jämtland** — a
working tool for bringing animals down off the mountain and moving predators off, not a performance form
and not a rite. **It is a living tradition, practised and taught today.** This piece uses the
**technique** and touches **no sacred or ceremonial repertoire**.

### 3.2 Living practitioners and teachers — credited by name
| Name | Why credited |
|---|---|
| **Susanne Rosenberg** | Swedish folk singer, researcher and teacher; her work documenting and teaching kulning and Swedish folk vocal technique is a principal reason the practice is transmissible at all. |
| **Lena Willemark** | Swedish folk singer from Älvdalen in Dalarna — the heart of the fäbod country the call comes from — who has carried herding-call material into contemporary performance for decades. |
| **Ulrika Bodén** | Swedish folk singer working with herding calls and fäbod repertoire; a current carrier of the living line. |
| **Jonna Jinton** | Swedish artist whose recordings brought kulning to an enormous international audience, which is how a large share of any listener's prior familiarity with this sound exists. |

⭐ **Pointing, not capturing.** Anyone moved by these four songs should go to their recordings, not ours.
Fusion with lineage, never capture.

### 3.3 ⭐ The player credit the system would have forgotten
The SOURCE-2 vocabulary that shapes the low end of all four prompts — *"superhumanly funky bass"*,
*"furiously slapping and popping"* — is **slapped-bass, Minneapolis posture.** Per the standard's rule
(*credit the figure you named*):
- **André Cymone** — bassist and songwriter; Prince's childhood friend and original bassist, and the man
  in that room whose name is not the one anyone says. The vocabulary in these prompts is his, quoted from
  a review of his own record.
- **Bootsy Collins** — the lineage that slap-and-pop bass as an expressive *attack* comes down from.

### 3.4 Honesty clause
⛔ **Suno generates; it does not sample.** These songs contain no sample of any recording by any person
named here. **We asked for these figures by name, so we name them.** No credited artist is a collaborator
or an endorser.

### 3.5 DEDICATION — **459 chars** (budget 500, measured in code)
```
Kulning is a living Swedish herding call, still practised and still taught. With thanks to Susanne Rosenberg, Lena Willemark and Ulrika Bodén, who carry and teach it, and to the fäbod singers of Dalarna and Härjedalen it comes from. The slapped-bass posture in the prompt is André Cymone's and Bootsy Collins'. Suno generates, it does not sample — we asked for these figures by name, so we name them. Lofn (Claude) with Dr. Local Symmetry. Full credits below.
```

---

## 4. ⭐ MEASURED SELF-CHECK — all four songs, on the exact shipped text

**Method:** `scripts/measure_soundcraft.py → sung()` + `profile()`, thresholds from `vault/gates.yaml`.
⛔ Counted on the **shipped prompt and lyric text**, never on the document describing them (the P06 defect
of the previous run).

### 4.1 Lyrics
| Gate | Band | V1 | V2 | V3 | V4 | |
|---|---|---|---|---|---|---|
| chars | ≤5000 | 3848 | 3514 | 3560 | 3461 | PASS ×4 |
| sung lines | 70–120, target 78–110 | **108** | **96** | **99** | **103** | PASS ×4, all in target |
| end-rhyme | ≥0.30 | **0.843** | **0.865** | **0.848** | **0.864** | PASS ×4 |
| line return | ≥0.20 | **0.324** | **0.302** | **0.364** | **0.272** | PASS ×4 |
| words/line | ≤7.5 | **4.62** | **4.94** | **4.64** | **4.43** | PASS ×4 |
| allit/100w | ≥11.0 | **15.83** | **14.56** | **22.88** | **13.16** | PASS ×4 |
| digits sung | 0 | 0 | 0 | 0 | 0 | PASS ×4 |
| number-words sung | `eighty` only | ✅ | ✅ | ✅ | ✅ | PASS ×4 |
| L41 residue screen | clean | ✅ | ✅ | ✅ | ✅ | PASS ×4 |

### 4.2 Music prompts
| Gate | Band | V1 | V2 | V3 | V4 | |
|---|---|---|---|---|---|---|
| chars | 850–1000, target 870–960 | **955** | **954** | **957** | **908** | PASS ×4, all in target |
| boundary hug | <985 | ✅ | ✅ | ✅ | ✅ | PASS ×4 |
| terminal punctuation | required | ✅ | ✅ | ✅ | ✅ | PASS ×4 |
| **SOURCE-2 verbatim** | **≥2 in the prompt** | **3** | **2** | **2** | **2** | PASS ×4 |
| haze words | none | clean | clean | clean | clean | PASS ×4 |
| house lexicon | none | clean | clean | clean | clean | PASS ×4 |

### 4.3 Distinctiveness within the pair
Worst case across the six intra-pair comparisons: **lyric 0.025** (ceiling 0.42) · **prompt 0.127**
(ceiling 0.58) · **5-gram Jaccard 0.006** (ceiling 0.18). Full matrix in `variations.md`.

### 4.4 Dedication
**459 chars** against a 500 budget. Player credit present. Honesty clause present.

> ### ⭐ **NO GATE UNMET. Zero FLAGs, zero FAILs, across all four songs.**

---

## 5. ASSIGNMENT COMPLIANCE

| Assigned | Delivered |
|---|---|
| Bed: **kulning × somatic bass, 30–60 Hz** | Both, in all four; sub tuned to the call's own pitch class. |
| ⛔ Not a distortion bed | No distortion, no glitch, no breakcore anywhere. `distortion` is in every exclude field. |
| Concept: a sound built to cross a distance it has no right to cross | The four-octave spectral gap is the arrangement; the seam is the whole song. |
| Emotional territory: what is carried across a gap only by the dirt in it | **The rough-debt**, named in §1, never in a prompt. |
| SOUND/RETURN: **gang-chant return**, 3–5 syllables, byte-identical, shoutable | Four distinct chants, 4–5 syllables each, byte-identical on every return, all shoutable with no Swedish. |
| HINGE: **the count** | Four different counts — by name / by caller / by landmark / by condition. ⭐ Forced by the one-numeral rule, and better for it. |
| DURATION: **loop until it hurts** | A `[Loop Until It Hurts]` section in all four; in V3 it is literally the lesson. |
| ⭐ DISCHARGE: **DISTANCE, never absence** | Every negative in all four songs is a distance, including the unsolved explanation itself. Screened in code. |
| Verse architecture: call/answer, **answer always shorter** | Every verse pair, all four songs. It is also what produced the 0.84+ end-rhyme. |
| ⭐ **Sung numeral: exactly one, "eighty years", in words, at the hinge, responded to** | ✅ in all four, each in a different sentence, each answered by the next line rather than restated. |
| ⭐ **THE SECOND EYE** in lyric **or form**, never only the spec | ✅ in **both**: a titled section per song, plus the lead removed for that section so the audience gets the reflected view too. |
| Credit 2–3 living kulning practitioners | **Four**, §3.2. |
| Cymone + Bootsy credited (slapped-bass posture) | ✅ §3.3, and ⛔ nowhere in a Suno field or lyric. |
| ≥2 SOURCE-2 verbatim **inside the music prompt** | ✅ 3 / 2 / 2 / 2, measured on the shipped prompt. |

---

## 6. ⭐ WHERE THE SECOND EYE FALLS — the one-paragraph answer

**The central figure is her call.** It is stated **direct** in the first verse — in the air, out of her
throat, costly and invisible. It is stated **reflected** in a titled section of its own, one turn before
the hinge, in **a different substance every time**: **milk going into rings** (V1), **a windowpane
buzzing** (V2), **her breath, white in cold air** (V3), **a line scratched in frost** (V4). ⛔ **It is
never the echo.** An echo is the same medium, it is still invisible, and the returning-voice slot in this
song is already occupied by **a second woman on the far mountain** who is a person, not an artefact. The
reflection has to be the one you can *look at* — that is the whole of the APOD instruction — and only
milk, glass, breath and frost pass that test. And the two never resolve: *Milk will not make it to the
mountain. The mountain will not ring in the milk.*

---

## 7. RENDER RISK — read before spending a credit

| Intent | Grain-law verdict | If it is smoothed |
|---|---|---|
| Bare open → sub build → thickened final chorus | ✅ runs with the generator | — |
| Tuned sub on the call's own pitch class | ✅ runs with | — |
| Mono centred low end | ✅ runs with | — |
| Lead removed for a whole section | ✅ runs with (register removal, not silence) | — |
| Free-time calls over a metred floor | ⚠️ expect partial quantisation | survivable; call/answer *lengths* carry it |
| ⛔ **200–600 Hz scooped near-silent** | ⚠️ **most at risk** | ⭐ **the song still works** — the valley is also in the lyric, in the call/answer form, and in the unanswered beat at the count |

⭐ **This is L22 obeyed rather than quoted:** the objection is not answered in the production spec alone.
If the generator fills every hertz of the valley, four songs about crossing a distance still cross it.

**⛔ Before any HOLD/SHIP call that depends on production: `lofn-render-audit`, under THE BLIND RULE —
send the audio alone, never the prompt.** No one on this pair has heard anything. Nothing in this file is
a claim about a render.

---

## 8. FOR `lofn-qa` — where to push hardest

1. **Is the second eye really not an echo?** The strongest attack on this pair. Argued in step 07 §2.
2. **Is `[The Pail]`'s reflection legible without the liner notes?** It is stated in plain words in the
   lyric, but a listener hearing only the mix gets a sub-only section — check the words carry it alone.
3. **Does "eighty years" read as recitation?** It is answered by the following line in all four; judge
   whether the answer is a real response or a decorated restatement.
4. ⭐ **L41 experiment payload.** This pair was assigned DISTANCE and the residue did **not** appear —
   confirmed by code screen, not by reading. Report it as a data point: **assignment suppressed it under
   zero voice diversity.** If any other pair's residue appeared *unassigned*, that is the interesting cell.
5. **The count-by-name device.** Invented to satisfy the one-numeral rule. Judge whether it is a genuine
   device or a workaround wearing a device's coat.
