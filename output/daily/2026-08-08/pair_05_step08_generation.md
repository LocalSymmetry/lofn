# PAIR 05 — STEP 08 · RAW PROMPT GENERATION
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P05 "TWO MINUTES EIGHTEEN"**

**Continuity payload used:** frozen ICB, LF-sha `9b538e91…`, **142,900 B** · personality DNA **27,796 B** · **18** baseline voices · **15 Special Flairs** marker present.
**Step file:** `skills/music/steps/08_Generate_Music_Generation.md` · **Inputs:** `pair_05_step06_facets.md`, `pair_05_step07_guides.md`.

> **This is the RAW stage and it is reported as raw.** The prompts below are the first pass. **Three of them failed a gate on first measurement and one lyric failed a floor outright.** The failures are printed here with their measured values, and the repairs are step 09's work. Reporting a clean raw stage would be a lie about where the craft happened.

---

## THE FOUR VARIATIONS — angle labels taken verbatim from `05_pair_assignments.md`

| | Angle label *(fixed by the run)* | Title *(this pair's, a THING)* | The hour |
|---|---|---|---|
| **V1** | **THE PACKING** | **The Folding Chair** | early evening |
| **V2** | **THE FORECAST** | **Cloud Off The Coast** | after the forecast |
| **V3** | **THE HANDWRITING** | **The Booking Page** | late, at the shelf |
| **V4** | **THE ALARM SET FOR FOUR** | **The Line Under The Door** | lights out |

**Title law, applied.** Every title names a **THING** — a chair, a cloud, a page, a line of light — never an argument. ⛔ No persona-brand prefix. ⛔ **No number in any title**, deliberately: the pair-level name **TWO MINUTES EIGHTEEN** stays at pair level (it is the ICB's name for the slot), and **no variation title carries a figure**, so that nothing in this pair can be mistaken for the run's one sung number, which belongs to P04. `door` is on the measured strong-token list.

---

## RAW V1 — **The Folding Chair**

### RAW MUSIC PROMPT *(first pass)*
```text
Tender and unhurried. A spacious modal chamber-jazz ballad in D Dorian at 62 BPM in six-four, built from spacey jazz and textured folk instruments: brushed kit on a dry ride cymbal, upright bass walking in whole notes, nylon-string guitar, a low shaker, and a modal tenor saxophone searching between the phrases and never landing. Room mics wide and honest, a rented room with two beds, a window and a kettle, every player audible in every microphone. Two women sing, both mezzo, both mid-forties, both the same unhurried northern-English speech-tone, deliberately hard to tell apart: one warm and breathy on the ends of words with a slight downward slide, the other flatter and quicker, talking more than singing, clipping her own sentences. The first voice falls in short lines on the beat. The second climbs in long lines and enters half a bar early. They sing alone in turn, then at once, each keeping her own tune and her own words. It thickens by adding instruments. No risers.
```
**RAW MEASURED: 983 chars.** ⚠️ Inside the hard band `850–1000` but **outside the target band `870–960`** → FLAG. Also within 2 chars of `music_prompt_hug_ceiling 985`. **Repair required at step 09.**

### RAW LYRICS PROMPT — the specification the field is built from
- **Opener:** `[Theme: …]` then `[SONG FORM: …]`, in that order.
- **Sections:** intro (instrumental) · falling voice alone · climbing voice alone · falling voice alone, short · **four simultaneous sections** (the third is THE CROSSING) · outro.
- **Simultaneity mechanism in the field:** the climbing voice's lines are written as **standalone parenthesised lines** interleaved between the falling voice's lines, with the section header stating *"two songs at the same time, neither waits for the other."* The crossing is the one line carrying **both** voices at once.
- ⭐ **THE CAUSAL CHAIN (F6 / F-A), fixed before any line was written.** She breathes on the black glass and holds it to the lamp → **finding no pinholes is what lets her pack it** → it goes into the side pocket → **the pocket being full is what puts the chair on top** → **the chair on top is why the bag will not shut** → **the bag not shutting is why the sandwiches come out** → they go by the kettle **so a foot finds them in the morning**. Each act is produced by the one before. ⛔ Reorder any pair and the chain snaps.
- **Falling-voice refrain (returns byte-identically):** `the second one —`
- **Climbing-voice refrain (returns byte-identically):** `(— is that new, or have you had it —)` — asked twice, never finished, never answered.
- **THE CROSSING BAR, raw:**
```text
because there isn't time to look —
(— hang on. Hang on. Hang on a —)
second — (second —)
```
- **SFX:** `*a bag set down*` (open) · `*a coat pocket zipped*` (the completed act, D8).
- **Raw opening quatrain:**
```text
I breathe on the sheet of glass —
hold it high to the lamp —
looking for a little light —
and there's none. And that's right —
```

**RAW MEASURED (lyric):** `end_rhyme 0.308` · `line_return 0.286` · `words_per_line 6.36` · `allit_per_100w 11.23` · `sung_lines 91`.
⚠️ **Passes every floor by a hair.** `end_rhyme` clears `0.30` by 0.008 and `allit_per_100w` clears `11.0` by 0.23. **That is not a pass, that is a coin toss.** Repair at step 09.

---

## RAW V2 — **Cloud Off The Coast**

### RAW MUSIC PROMPT *(first pass)*
```text
Warm and open-windowed. A spacious modal chamber-jazz ballad in D Dorian at 60 BPM in six-four, made of spacey jazz and textured folk instruments: a modal tenor saxophone well forward and searching, brushes swirling on a coated head, upright bass in long slow steps, bowed vibraphone under everything, and a bass clarinet doubling the bass an octave up. Room mics wide, a rented room with two beds, a window and a kettle, air and distance in the microphones. Two women sing, both mezzo, both mid-forties, both the same unhurried northern-English speech-tone, deliberately hard to tell apart: one steady and breathy with a small catch before each phrase, the other quicker and flatter, reading aloud, breaking off mid-sentence. One voice falls in short lines on the beat. The other climbs in long lines, entering early. They sing in turn, then together, keeping separate tunes. New instruments arrive, never more volume. No swells.
```
**RAW MEASURED: 930 chars** ✅ in target band.

### RAW LYRICS PROMPT — specification
- ⭐ **CAUSAL CHAIN:** the kettle goes on the sill → **its steam is what fogs the pane** → **the fog is why she wipes it with her sleeve** → **wiping is what uncovers the low strip of clear sky** → **the strip is why there is a road up the back of the town** → the kettle knocks, clicks off, and she pours.
- **Falling-voice refrain:** `it clears by the coast —`
- **Climbing-voice refrain:** `(— and there'll be another one, there's always —)` ⭐ Trish's kindness, repeated, never finished.
- **The wince:** she answers *"And I say yeah. I say yeah — and I mean it. I do —"*. **She means it.**
- ⭐ **D6, in one line:** Trish says *"— and I feel bad, because I — because I booked it, and now — and now it's cloud"* — she takes the blame **for the weather.** She is one inch from the insight and forecloses it herself: *"— you're very good about it. You —"*.
- **THE CROSSING BAR, raw:** `and it climbs, and it clicks — / (— hang on. Hang on. Give it a —) / second — (second —)`
- **SFX:** `*a kettle starting up*` · `*a kettle clicking off*`.

**RAW MEASURED (lyric):** `end_rhyme 0.411` · `line_return 0.389` · `words_per_line 6.41` · `allit 11.96` · `sung_lines 90`. ✅ all floors clear.

---

## RAW V3 — **The Booking Page** ⭐ *the pair's centre of gravity*

### RAW MUSIC PROMPT *(first pass)*
```text
Plain and close. A spacious modal chamber-jazz ballad in D Dorian at 64 BPM in six-four, made of spacey jazz and textured folk instruments: nylon-string guitar picking single notes, upright bass plucked short and dry, brushes tapping the rim more than the head, a hammered dulcimer struck once a bar, and a modal tenor saxophone searching low behind the voices. Room mics wide, a rented room with two beds, a window and a kettle, the kettle and the chairs audible. Two women sing, both mezzo, both mid-forties, both the same unhurried northern-English speech-tone, deliberately hard to tell apart: one level and warm with a small breath before every line, the other faster and drier, half-talking, cutting herself off. One voice falls in short lines on the beat. The other climbs in long lines, half a bar early. They start apart, then run at once with separate words. Instruments are added, never level. No swells.
```
**RAW MEASURED: 915 chars** ✅ in target band.

### RAW LYRICS PROMPT — specification
- ⭐ **CAUSAL CHAIN:** there is a ring of tea on the page → **the ring is what has stuck it to the shelf** → **being stuck is why it tears when she lifts it** → **the writing turns out to be on the back, so she has to turn it over** → turning it puts **Trish's handwriting** in front of her → her thumb is on the line, **so she moves the thumb** → **moving the thumb uncovers a box** → the box has a line and a line under it → **the time is under the box** → the tea has gone through the paper, **so the time is soft** → **that is why she tilts it to the lamp** → she reads it, folds it, **and puts it back under the kettle.**
- **Falling-voice refrain:** ⭐ `Trish, and then —` — she reads down the page, says the other woman's name out loud, **and skips her own to get to the time.** Repeated. The listener waits for her name every time and never gets it. ⭐ **The song withholds her name from the listener exactly as the page withholds it from her — she is never named anywhere in this pair.**
- **Climbing-voice refrain:** `(— did they take the deposit, or is it on the —)`
- ⭐ **THE CROSSING BAR, raw — the pair's whole argument in one beat:**
```text
and the box says — hang on —
(— hang on. Hang on. Give me a —)
second — (second —)
driver. It says second driver —
(— sorry. Sorry. You go. What —)
and that's me. That's my one —
```
  **She reads the words *second driver* off a friend's handwriting and identifies the box as hers, cheerfully, and hears nothing.** Trish, at that instant, means *wait a moment*. Same note. Neither adjusts.
- ⛔ **Not a form being filled** (P04's device): nobody fills anything in. The page is a **handwritten copy** made days ago, off-screen. *A hand wrote it.* Declared at step 06 §4.12 for the cross-pair sweep.
- **The ending:** ⭐ the page goes **back under the wet kettle** — the most consequential object in the song, returned to the thing that was ruining it *(D8, the Tinguely price)*.
- **The outro is the WRONG VOICE:** Mezzo One stops; Trish carries on alone, still climbing, still unfinished. ⭐ *They finish in different places*, literally.
- **SFX:** `*paper peeled off a shelf*` · `*paper under a kettle*`.

**RAW MEASURED (lyric):** `end_rhyme 0.449` · `line_return 0.326` · `words_per_line 6.21` · `allit 17.90` · `sung_lines 89`. ✅ all floors clear.

---

## RAW V4 — **The Line Under The Door**

### RAW MUSIC PROMPT *(first pass)*
```text
Quiet and wide awake. A spacious modal chamber-jazz ballad in D Dorian at 58 BPM in six-four, made of spacey jazz and textured folk instruments: a harmonium holding one low note the whole way through, brushes with the snares off, upright bass in long held notes, nylon-string guitar answering once a phrase, and a bowed vibraphone very far back. Room mics wide, a rented room with two beds, a window and a kettle, a room at night with people still in it. Two women sing, both mezzo, both mid-forties, both the same unhurried northern-English speech-tone, deliberately hard to tell apart: one clear and breathy, smiling in the tone, the other lower and slower, thickening, drifting off mid-word. One voice falls in short lines on the beat. The other climbs in long lines, early, then thins to nothing. The second voice stops. The first carries on alone. Instruments are added, never level. No fade-out.
```
**RAW MEASURED: 901 chars** ✅ in target band.

### RAW LYRICS PROMPT — specification
- ⭐ **CAUSAL CHAIN:** the light goes off → **the room is not dark, because of a line under the door and a green standby light** → she lies on her back with her hands on her chest → she goes through it, silently → Trish surfaces once and mumbles → ⭐ **she has not set the alarm, and she will not set it yet, because the screen would wake Trish** → **so she waits until Trish is asleep** → **only then does she reach out and set it.** ⭐ **That is why the alarm is the last act in the song: the causal chain puts it there, not the doctrine.**
- **Falling-voice refrain:** ⭐ `this is the good bit —` — said in the dark, alone in a shared room, awake, **and she means it.**
- **Climbing-voice refrain:** `(— are you all right over there, are you —)` → decaying to `(— mm —)` → gone.
- **THE FORM MADE LITERAL:** Trish's strophe **stops mid-word and does not come back.** Mezzo One continues alone for the last fifteen lines. ⛔ No fade-out. She simply carries on and then stops.
- **THE CROSSING BAR, raw:** `and I'll wait till she's gone — / (— mm. Did you — in a —) / second — (second —)`
- **D8:** the alarm is set — **for the dark, for before it's light**, ⛔ never for a stated hour, because no number is sung in this pair.
- **SFX:** `*a switch, then dark*` · `*a small alarm set*`.

**RAW MEASURED (lyric):** `end_rhyme 0.439` · `line_return 0.598` · `words_per_line 5.89` · **`allit_per_100w 9.73`** · `sung_lines 82`.
⛔ **FAIL — `alliteration_per_100w_floor 11.0` breached at 9.73.** This is a genuine floor failure on the raw pass, printed rather than smoothed. **Repair is step 09's first job.**

---

## RAW-STAGE CROSS-VARIATION MEASUREMENT — extraction printed before conclusion

```
EXTRACTED: 4 music prompts (983 / 930 / 915 / 901 chars) and 4 lyric bodies
           (3088 / 3021 / 2948 / 2654 chars of sung text)
SequenceMatcher(autojunk=False) on the RAW music prompts, ceiling 0.58:
  V1-V2 0.681   V1-V3 0.686   V1-V4 0.666
  V2-V3 0.680   V2-V4 0.718   V3-V4 0.660
```
⛔ **FAIL — every pair of raw music prompts sits between 0.66 and 0.72, against a `portfolio_max_prompt_similarity` ceiling of 0.58.**

**Diagnosis, honestly.** The four prompts were written to one template — same sentence order, same clause shapes, same phrasing for the invariants. `step04_medium.md` fixes the genre, room, vocal configuration and dynamic shape for the whole pair, so *some* overlap is mandated; **0.68 is not that overlap, it is a habit.** ⭐ This is the run's own named disease — *"six pairs, one voice underneath"* — reproducing itself one level down, inside a single pair. **Repair at step 09.**

Lyric similarity and n-gram Jaccard on the raw pass were already clean (max 0.269 against 0.42; max 0.003 against 0.18) — **the template lived in the prompts, not the words.**

---

## RAW-STAGE GATE SUMMARY

| Gate | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| `music_prompt_chars` 850–1000 | 983 ✅ | 930 ✅ | 915 ✅ | 901 ✅ |
| `music_prompt_chars_target` 870–960 | **983 ⚠️ FLAG** | 930 ✅ | 915 ✅ | 901 ✅ |
| `alliteration_per_100w_floor` 11.0 | 11.23 ⚠️ hug | 11.96 ✅ | 17.90 ✅ | **9.73 ⛔ FAIL** |
| `rhyme_return_floor` 0.30 | 0.308 ⚠️ hug | 0.411 ✅ | 0.449 ✅ | 0.439 ✅ |
| `portfolio_max_prompt_similarity` 0.58 | **0.66–0.72 ⛔ FAIL across all six pairings** ||||

**Three defects carried into step 09: one hard floor failure (V4 alliteration), one similarity failure (all four prompts), two boundary hugs (V1 chars, V1 rhyme/alliteration).**
**Repair budget consumed at this step: 0 of 3.**

*Step 08 complete. Step 09 next: refinement in the panel's voices, and the three repairs.*
