# STEP 10 — PRODUCTION WRAP · P05 · **CANARY**
**Run:** `2026-08-13_daily_lofn-prime` · **Pair:** P5 · AMBITIOUS · EXISTENCE · **AWE × INDIGNATION**
**Ships:** four packages, all in `variations.md`. **Flagship = V1.** Fields below are the shipped text.

> ⭐ **THE OBSCURE EMOTION — LEDGERSKEW.** *The unpanicked vertigo of standing inside a record that is
> confidently wrong about you, watching it be re-issued tomorrow morning, and answering to it anyway,
> because answering is faster than being right — and because the sureness in the other voice is, strangely,
> a kind of care.* ⛔ Named here. **Never in a prompt.**

---

## 1. FLAGSHIP — **ANSWER TO CANARY**

### TITLE
```
Answer to Canary
```

### MUSIC PROMPT — *shipped text, 949 chars, terminal punctuation*
```
A glitch-baroque processional detonated by breakcore, staged as one long afternoon in a dye works, Harpsichord ground bass and a four-viol consort play strict counterpoint in their own rubato and never lock to the grid while pixel-sorted amen edits and stuttering breaks cut across them mid-bar, Signature 1: The Ticket Engine, a clipboard click, a ledger page and a rubber date-stamp chopped into thirty-second bursts, panned narrow and dry, used as the hi-hat, Signature 2: The Vat Sub, a superhumanly funky bass furiously slapping and popping against the ground bass, mono below sixty hertz, ducking on her inhale instead of the kick, Female late-twenties mezzo, crystalline and breathy with audible inhales, sung behind the beat, one bratty compressed snarl and no more, The final chorus is sung through a rubber respirator with the exhale valve left in as percussion, 170 BPM, F sharp minor, octave drop at the hinge, three-chorus stack to end.
```

### EXCLUDE
```
trap hi-hats, EDM risers, autotune, male lead vocal, children's voices, orchestral swell ending, fade-out, hard-panned percussion, untuned drone, sub-bass sidechained to the kick
```

### LYRICS FIELD
⭐ Verbatim in `variations.md` § **V1 — LYRICS FIELD** (4163 chars, 86 sung lines). Not restated here so
one file remains the single measured source.

### SONG GUIDE
One woman, one afternoon, a dye works. The ticket on the drum says **CANARY**; the colour in the drum is
the bluest blue there is. **Nobody in the building is lying and no one is going to fix it.** She carries
buckets from the vat room to the rinse ditch in the yard, and she says the wrong name all afternoon with
total conviction — *"I answer to it. I insist."* The verses spiral: **exactly one word changes per
stanza**, five changes, and the last one turns *carry* into *follow*. The chorus is the same figure
stated twice — **the vat you must not look into**, then, an octave down, **the ditch you may kneel over**
— and in the final stack both are sung at once with the low voice a beat late so they can never lock.
**The last chorus is performed through a rubber respirator, valve left in the mix.** The bird that goes
underground first is present in exactly six words and is never explained.

---

## 2. ⭐ LINEAGE & CREDIT
⛔ **Never inside a Suno field. Never inside a lyric.** Verified by grep: all four shipped prompts and
lyrics are clean of every name below.

### The players, by name
- ⭐ **André Cymone** — Prince's childhood friend and **first bassist**, Minneapolis. Every one of the four
  prompts asks for a **slapped and popped bass**, and that posture came from a room he was standing in.
  His name is the one that does not get said, which is — with no irony available to me — **exactly this
  pair's concept**, arriving in the credits instead of the lyric. *(He is a CREDIT. `D4` holds: the woman
  in these songs is invented and he is not a character in them.)*
- ⭐ **Bootsy Collins** — for the pocket that made a slapped bass funny as well as heavy. The
  *"swagger, humor"* in the shipped prompts is asking for his posture.
- ⭐ **Gregory C. Coleman**, drums, with **The Winstons** — *"Amen, Brother"*, 1969. All four prompts name
  **amen edits**, so all four owe him. He played the four bars breakcore is built on and **died homeless
  in Atlanta in 2006 having never received royalties for the most reused drums in recorded music.**
  *(V4's prompt originally said only "breakcore edits" while its dedication credited the amen — an
  overclaim. The prompt was corrected to name the amen rather than the credit softened.)*
- **Teddy Riley** — new jack swing, for V2's snare placement, which is the only thing V2 borrows from it.

### The traditions
Baroque **ground bass** and the English **viol consort** (Byrd, Lawes, Purcell's grounds) · **breakcore**
and the jungle/DnB break-chopping lineage · the **Minneapolis sound**.

### Honesty clause
⛔ **Suno generates; it does not sample.** The claim is *"we asked for these figures by name, so we name
them"* — never *"contains a sample of."* A false credit would make the true ones worthless.

### Dedications — ⭐ measured, all ≤500 chars
| | chars | |
|---|---|---|
| V1 *Answer to Canary* | **458** | ✅ |
| V2 *Say It at the Gate* | **482** | ✅ |
| V3 *Yellow to the Throat* | **490** | ✅ |
| V4 *Sparrows Come to Look* | **498** | ✅ |

---

## 3. ⭐ MEASURED SELF-CHECK — *measured in code, never by eye*
`scripts/measure_soundcraft.py → profile_file()` on the shipped lyric blocks; char counts on the shipped
prompt strings; similarity by `difflib.SequenceMatcher` + 5-gram Jaccard.

### Lyric gates
| gate | floor / ceiling | **V1** | **V2** | **V3** | **V4** | |
|---|---|---|---|---|---|---|
| sung lines | 70–120 *(target 78–110)* | **86** | **86** | **86** | **86** | ✅ |
| end-rhyme | ≥ 0.30 | **0.756** | **0.616** | **0.547** | **0.570** | ✅ |
| line-return | ≥ 0.20 | **0.779** | **0.791** | **0.779** | **0.779** | ✅ |
| words / line | ≤ 7.5 | **6.63** | **6.79** | **6.83** | **6.74** | ✅ |
| alliteration /100w | ≥ 11.0 | **13.16** | **14.04** | **14.99** | **16.72** | ✅ |
| unique-line ratio | ≥ 0.45 *(flag)* | **0.500** | **0.477** | **0.488** | **0.488** | ✅ |
| lyrics field chars | ≤ 5000 | **4163** | **4016** | **4005** | **4143** | ✅ |

*(Archive winners for reference: end-rhyme 0.463 · line-return 0.326 · w/line 6.69 · allit 14.2. All four
sit **above** the winners on return and at parity on texture — the spiral and the three-chorus stack are
doing that, by construction.)*

### Prompt gates
| gate | band | **V1** | **V2** | **V3** | **V4** | |
|---|---|---|---|---|---|---|
| music prompt chars | 850–1000, **target 870–960** | **949** | **940** | **886** | **922** | ✅ |
| boundary-hugging | flag at ≥ 985 | 949 | 940 | 886 | 922 | ✅ none |
| terminal punctuation | required | ✅ | ✅ | ✅ | ✅ | ✅ |
| ⭐ SOURCE-2 verbatim **in the shipped prompt** | ≥ 2 | **2** | **2** | **2** | **2** | ✅ |

**SOURCE-2, counted on the shipped prompt string and not on this document** *(P06's failure last run)*:
V1 `superhumanly funky bass` + `furiously slapping and popping` · V2 `swagger, humor, and sheer
danceability` + `a splash of vintage new jack swing` · V3 `downright salacious slow jam` + `furiously
slapping and popping` · V4 `swagger, humor, and sheer danceability` + `superhumanly funky bass`.
⭐ **All five phrases are spent across the set.**

### Distinctiveness — within the pair (portfolio ceilings apply across all twenty-four)
| pair | lyric *(≤0.42)* | prompt *(≤0.58)* | 5-gram Jaccard *(≤0.18)* |
|---|---|---|---|
| V1–V2 | 0.031 | 0.490 | 0.002 |
| V1–V3 | 0.026 | 0.353 | 0.023 |
| V1–V4 | 0.016 | 0.492 | 0.028 |
| V2–V3 | 0.025 | 0.429 | 0.005 |
| V2–V4 | 0.022 | 0.494 | 0.002 |
| V3–V4 | 0.023 | 0.347 | 0.025 |
✅ **Maximum lyric similarity 0.031 against a 0.42 ceiling.** Prompt similarity is the highest number here
(0.49) and it is load-bearing: the four share an *assigned* bed, two signature-style slots and one
vocalist. Still clear of 0.58.

### Deterministic string gates
| check | result |
|---|---|
| sung cardinal numerals *(assigned: **NONE**)* | ⭐ **zero in all four** |
| ordinals present | `first` — V1 (*"I go in first"*, ×3), V3 (*"Warm water first"*). **Order of entry, not a numeric fact.** Reported, not hidden. |
| ⭐ **L41 residue** `nobody` / `nothing` | **absent from all four** — the negative is discharged by MISNAMING, as assigned |
| correction vocabulary `wrong / correct / mistake / misfiled / actually / blue` | **absent from all four** |
| house lexicon (`gates.yaml`) | **zero hits** in any prompt or lyric |
| credit names leaking into a Suno field or lyric | **zero** — Cymone, Bootsy, Coleman, Winstons, Riley appear only in §2 |
| `D7` — AI / model / software as subject | **absent.** One body, one afternoon, a human. The certain taxonomy is a clipboard. |

---

## 4. ⭐ ASSIGNED AXES — DELIVERED
| axis | assigned | delivered |
|---|---|---|
| bed | glitch-baroque × breakcore, **collided not alternating** | AWE owns pitched material + voice, INDIGNATION owns transients + floor, **same bar, never acknowledged by each other** |
| sound/return | **rebound rhyme on beat 3** | every pre-chorus: line-end rhyme answered at beat 3 of the next line (`card→yard`, `coat→throat`, `wall→hall`) |
| hinge | **octave drop** | one bar, drums out, the line repeated an octave down — and it is the pivot between the two eyes |
| duration | **three-chorus stack** | sections 11-12-13 back to back: high → octave-down → both-at-once |
| discharge | ⭐ **MISNAMING, never corrected** | delivered, and **three would-be correction lines were cut to keep it** (Step 08) |
| verse | **the spiral, one word per stanza** | six stanzas, five changes, mapped in Step 07 §4 and `variations.md` |
| sung numeral | **NONE** | zero |
| ⭐ second eye | **direct + reflected, in lyric or form** | **the chorus itself, stated twice across the hinge, in the lyric AND the form**, staggered a beat so it cannot resolve |
| bold choice, **not a glitch** | required | ⭐ **the final chorus sung through a rubber respirator, valve left in as the hi-hat.** A physical object between mouth and mic, zero processing. |

---

## 5. WHAT I WOULD FLAG TO QA
1. ⭐ **`first` is sung.** Assigned numeral is NONE. I judge `first` an ordinal of entry order, not a
   numeric fact — and it is the one unglossed appearance of the canary's second meaning, so removing it
   would cost the song its point. **Flagging rather than hiding.** Overrule me and I will cut it.
2. **Line-return 0.78–0.79 is roughly 2.4× the archive winners.** That is the assigned spiral plus the
   assigned three-chorus stack multiplying, not drift. Unique-line ratio still clears its floor (0.477+),
   so it is dense return rather than collapse — but a listener who finds it hypnotic and a listener who
   finds it a loop are both hearing correctly, and **only a render tells us which.**
3. **The device is invisible to a text gate.** Every gate I own reads the lyric as a woman calling a
   colour yellow. **The song only works if the listener supplies the blue** — and I cannot measure that
   from here. This is the exact class of failure THE BLIND RULE exists for: it wants a first-read from
   somebody who has never seen this brief. If the blind read comes back *"a song about a dye works"*,
   the device landed; if it comes back *"a song about being misnamed"*, I over-signalled and V1 needs
   repair.
4. **Prompt similarity 0.49 within the pair** is the highest number in this package and it is structural
   (one assigned bed, shared signature slots). Clear of 0.58, but it is the number that would move first
   if the portfolio check is tight across all twenty-four.
5. ⛔ **No render has happened. No audio has been heard. Nothing here is a claim about how it sounds.**
