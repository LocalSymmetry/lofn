# PAIR 05 — STEP 09 · ARTIST-REFINED PROMPTS
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P05 "TWO MINUTES EIGHTEEN"**

**Continuity payload used:** frozen ICB, LF-sha `9b538e91…`, **142,900 B** · personality DNA **27,796 B** · **18** baseline voices · **15 Special Flairs** marker present.
**Step file:** `skills/music/steps/09_Generate_Music_Artist_Refined.md` — *rewrite each raw prompt pair in a chosen artist's signature style; inject flair ≤150 words; **do not add new scene content**.*

> ⛔ **LIVE CONFLICT WITH THE STEP FILE, RESOLVED PER HANDOFF §1 (L30).**
> Step 09 asks for a rewrite "in a chosen **artist's** signature style." The run's hard output contract forbids a real-artist name in **any** Suno-bound field, and D9 forbids naming a living tradition in a render field.
> **Resolution:** the refinement is performed in the voices of the **supplied panel constructs** — which are model-generated interpretive seats, each *"after"* a named source figure, and are **never quotations of or endorsements by those people**. Their names appear **in this artifact's prose only** and **in no Suno-bound field**. Verified in §5. **No new scene content was added by any seat**; every intervention is subtractive, structural, or sonic.

---

## 1. THE REFINEMENT SESSION — five seats, three repairs

### ⛔ REPAIR 1 — V4's alliteration floor breach (`9.73` against a floor of `11.0`)

**THE RAW HORN (after Getatchew Mekurya)** — *the seat whose whole style comes from an unaccompanied vocal form:*
> *"The fourth one has gone soft in the mouth. Look at the lines: 'and I lie on my back', 'and I go through it once', 'and I reach out for the phone'. Every one of those is a sentence a person could **think**. None of them is a sentence a person has to **say**. There is nothing for the tongue to push against. That is why the number is low — it is not a scoring problem, it is a **speaking** problem, and in a song that is mostly one woman lying still in the dark, the tongue is the only thing moving."*

**What was actually changed** — twenty-one lines, all in the direction of speech, none in the direction of decoration:

| raw | refined | why |
|---|---|---|
| `and I lie on my back —` / `with my hands on my chest —` | `and I lie on my back, and my hands —` / `my hands flat on my chest —` | anadiplosis on *hands*; the hand becomes the subject of the next line, which is the pair's whole method |
| `and I go through it once —` *(×4)* | `and I go through the thing once —` *(×4)* | four instances, one edit; *through / the / thing* is a real consonant chain and the phrase is plainer |
| `and I know where I'll stand —` | `and I know the spot. I'll stand —` | *spot / stand*; and she now has a **place**, not a plan |
| `and the cold on my hands —` | `and the cold coming on my hands —` | *cold / coming*; the cold is arriving, not present — she is rehearsing tomorrow |
| `and the wind at my back, and —` | `and the wind. The wind at my back, and —` | *diacope*; ⭐ she says it twice **because she is rehearsing it**, which is character, not craft |
| `lying here in the dark —` | `lying here. Lying here in the dark —` | *epizeuxis*; the second one is slower |
| `and I set it for the dark —` | `and I set it. Set it for the dark —` | the act, then the act named |
| `and I put my hands back —` | `and put my hands back on my chest —` | ⭐ closes the loop with verse one — the same hands, the same place, the song over |

**MEASURED, before → after:** `allit_per_100w` **9.73 → 17.65** *(floor 11.0)*. `end_rhyme` 0.439 → 0.439. `line_return` 0.598 → 0.598. `words_per_line` 5.89 → 6.43 *(ceiling 7.5)*. `sung_lines` 82 → 82.
**Repair budget: 1 of 3 attempts used on this gate. Cleared on the first attempt.**

---

### ⚠️ REPAIR 2 — V1's two boundary hugs (`end_rhyme 0.308`, `allit 11.23`, `chars 983`)

**THE ORCHESTRATOR OF CHARACTER (after Duke Ellington)** — *the seat that refuses a part any competent player could have played:*
> *"Clearing a floor by eight thousandths is not clearing it. And I will tell you where the slack is: it is at the **end** of the song. Look — the first sixty lines have a rhyme system and the last twenty have nothing. You wrote the beginning like a composer and the end like a man who wanted to go home. She does not go home in this song. **Neither do you.**"*

He was right, and the diagnosis was specific: the two closing sections had **no** returning end-sound at all. Repairs, all inside the last two sections plus six single-word swaps earlier:

- `zipped and by the door —` → `and it stays there, shut —` *(pairs with `It stands by the door, shut —` two lines up)*
- `not till it's dark and cold —` → `not till it's dark. Not again —` *(pairs with `and I'll not open it again —`)*
- `in the left, where the hand —` → `in the left of the coat —` *(pairs with `and the second one's in the coat —`)*
- outro rebuilt so that `look / look`, `shut / shut`, `food / good`, `back / back` all return inside the four-line window
- `I breathe on the sheet of glass —` → **`I breathe on the black glass —`** ⭐ *black / breathe* is the alliteration, and **"black glass" is the more accurate object** — the raw line described a pane, the refined line describes the thing she is actually holding
- `and be down past the border before —` → `and be down past the border. Past —` *(the climbing line's own chain)*
- `tucked under the arm of the chair —` replaces `under the arm and over —` *(*tucked / the* and a real rhyme partner)*

**MEASURED, before → after:** `end_rhyme` **0.308 → 0.652**. `allit_per_100w` **11.23 → 12.66**. `line_return` 0.286 → 0.272 *(floor 0.20)*. `words_per_line` 6.36 → 6.61 *(ceiling 7.5)*. `sung_lines` 91 → 92.

**Prompt length:** 983 → **923** chars. Cuts were *"and honest"*, *"cymbal"*, *"the phrases"*, *"with a slight downward slide"* → *"sliding down at the end of a word"*, *"her own sentences"* → *"her sentences"*, *"her own tune and her own words"* → *"her own tune and words"*. **Nothing was cut that carried an instruction.** Now inside `870–960` and **62 chars clear of the `985` hug ceiling.**
**Repair budget: 1 of 3 used. Cleared on the first attempt.**

> ⚠️ **DISCLOSED, because a naive read of `end_rhyme 0.652` will over-credit it.** The interrupted diction breaks a great many climbing-voice lines on function words — *"…and the —"*, *"…I'm —"*, *"…you —"* — and the measure's crude last-three-characters key rewards that. **So the companion was computed:** counting **only** lines whose final word is a content word, the return rate is **V1 0.582 · V2 0.404 · V3 0.386 · V4 0.377** *(n = 55 / 52 / 44 / 53 lines)*. **All four clear the 0.30 floor on content words alone.** The rhyme is real; the function-word endings inflate it, and the inflation is now on the record rather than in the number.
> ⭐ And the honest converse: **the diction constraint and the return floor are the same mechanism here.** A half-sentence must break somewhere, and a person interrupted twice breaks in the same place twice. That is not gaming the instrument; it is what interrupted speech sounds like.

---

### ⛔ REPAIR 3 — all four music prompts, similarity `0.66–0.72` against a `0.58` ceiling

**THE FUSION ARCHITECT (after Mulatu Astatke)** — *patient, structural, allergic to ornament that does not carry weight:*
> *"You have written one prompt four times and changed the furniture. The room is fixed, the mode is fixed, the two voices are fixed — fine, those are the pair. But you have also fixed the **order of the sentences**, and there is no reason for that at all. Start the fourth one with what happens. Start the third one with the room. The instruction does not care which end you enter from, and the machine reads the front hardest — so put a **different** thing at the front of each."*

**THE BIG-BAND TRANSLATOR (after Russ Gershon)** — *the seat that does the boring work:*
> *"And say the invariants four different ways, because you are describing the same two singers to four different sessions. 'Both mezzo, both mid-forties, both the same accent' is a chart marking. 'Matched on purpose, so a listener has to work out which is which' is the same marking with the reason attached. Use the second one where the reason matters."*

**What was actually changed — sentence ORDER, per variation:**

| | order of the refined prompt |
|---|---|
| **V1** | emotion → genre → instruments → **room** → voices → progression → blacklist |
| **V2** | emotion → genre → **voices first** → instruments → room → progression → blacklist |
| **V3** | emotion → **room first** → genre + instruments → voices → progression → blacklist |
| **V4** | emotion → genre → **progression first** → instruments → room → voices → crossing → blacklist |

And the invariants re-voiced four ways:
- V1 *"two women in the same range, the same age, the same unhurried northern-English delivery, close enough to be mistaken for one person"*
- V2 *"Two mezzo singers are matched on purpose: same age, same accent, same weight of tone, so a listener has to work out which is which"*
- V3 *"Two women sing in the same register and the same unhurried northern-English speech-tone, deliberately confusable"*
- V4 *"Both singers are mezzos of the same age and accent, hard to tell apart"*

⭐ **The one string held byte-identical in all four is Source 2's assigned phrase, *"spacey jazz and textured folk instruments"*, which is imported verbatim by mandate** (`step04_medium.md`) and is 41 characters. That is the only deliberate shared substring.

**MEASURED, before → after (SequenceMatcher, `autojunk=False`, ceiling 0.58):**

| | raw | refined |
|---|---|---|
| V1–V2 | 0.681 | **0.319** |
| V1–V3 | 0.686 | **0.435** |
| V1–V4 | 0.666 | **0.421** |
| V2–V3 | 0.680 | **0.325** |
| V2–V4 | 0.718 | **0.420** |
| V3–V4 | 0.660 | **0.277** |

**Maximum 0.435 against a ceiling of 0.58.** ⭐ **Prompt char counts after the rewrite: 923 / 899 / 926 / 928 — all four inside the `870–960` target band, all four ≥57 chars clear of the `985` hug ceiling, all four ending in a full stop.**
**Repair budget: 1 of 3 used. Cleared on the first attempt.**

---

### THE TWO REFINEMENTS THAT WERE *REFUSED*

**THE SEARCHING DEVOTIONAL (after Alice Coltrane)** proposed a sustained bowed tone under the crossing bar — *"a phrase that does not resolve, that simply keeps being played."*
⛔ **Refused.** `step04_medium.md`: *"No long full stops, no untuned drones"* — L22, render-measured: they get smoothed and the intent is lost with them. **A tuned harmonium note holding one pitch is permitted and is already in V4.** The sustained tone would also be a **production** answer to the crossing, which is exactly the thing THE GRAIN LAW says does not count.

**THE MAXIMALIST (after Kamasi Washington)** proposed the two voices arriving together, once, in the final section — *"give me one moment of size."*
⛔ **Refused, and this is the pair's whole existence.** The two lines meet on **one note, for one beat, and go past each other.** They finish a perfect fourth apart. ⛔ **No harmony arrives. No unison is held. They do not end together.** His greyness objection was withdrawn *conditionally* — on the gap being audible — and it is answered by **the gap**, not by a merge. **If the render harmonises them, his objection is live again, and so is the whole pair's failure.** Flagged to the render audit in step 10.

---

## 2. THE FOUR REFINED MUSIC PROMPTS

*(Suno-bound text only. No seat names, no tradition names, no artist names, no section-label brackets.)*

### V1 — **The Folding Chair** · 923 chars
```text
Tender and unhurried. A spacious modal chamber-jazz ballad in D Dorian, 62 BPM, six-four, built from spacey jazz and textured folk instruments: brushes on a dry ride, upright bass walking in whole notes, nylon-string guitar picking one note a beat, a low shaker, and a modal tenor saxophone that keeps searching and never lands. Track it as a band in a rented room with two beds, a window and a kettle: room mics open wide, everyone leaking into everyone else's microphone, entries made by ear rather than by count. The singers are two women in the same range, the same age, the same unhurried northern-English delivery, close enough to be mistaken for one person. One is warm and breathy and slides down off the end of a word. The other is drier and faster and talks more than she sings. The falling line begins alone. The climbing line answers alone. Then both run at once with different words. Thicken by adding players.
```

### V2 — **Cloud Off The Coast** · 899 chars
```text
Warm, open, standing at a window. Slow modal chamber jazz in D Dorian, 60 BPM, six-four. Two mezzo singers are matched on purpose: same age, same accent, same weight of tone, so a listener has to work out which is which. The first is level and breathy with a small catch before every phrase; the second is quick and flat and reads aloud, breaking off halfway through what she is saying. Around them, spacey jazz and textured folk instruments: a modal tenor saxophone well forward and searching, brushes swirling on a coated head, upright bass in long slow steps, bowed vibraphone underneath, bass clarinet doubling the bass an octave up. The room is rented, two beds, a window, a kettle, and the microphones are far back so the air is part of the sound. She sings, then the other sings, then they overlap and stay overlapping, each holding her own tune. More instruments arrive. Nothing gets louder.
```

### V3 — **The Booking Page** · 926 chars
```text
Plain, close, mid-task. Everything is played live in one small rented room, two beds, a window, a kettle, room mics wide and the chairs and the kettle audible under the take. Slow modal chamber jazz in D Dorian, 64 BPM, six-four, scored for spacey jazz and textured folk instruments: nylon-string guitar picking single notes, upright bass plucked short and dry, brushes tapping the rim more than the head, hammered dulcimer struck once a bar, and a modal tenor saxophone searching low behind the words. Two women sing in the same register and the same unhurried northern-English speech-tone, deliberately confusable; one is level and warm and breathes before each line, the other is faster, drier, half-talking, cutting herself off. A descending line in short phrases sits on the beat. A rising line in long phrases arrives early. They begin separately and end up simultaneous, keeping separate words. Add players, not volume.
```

### V4 — **The Line Under The Door** · 928 chars
```text
Quiet, wide awake, happy about tomorrow. Slow modal chamber jazz in D Dorian, 58 BPM, six-four. It begins with one voice on her own, then a second voice on her own, then the two of them together for the rest of the song, holding different tunes and different words until one of them simply stops mid-word and the other carries on by herself. Spacey jazz and textured folk instruments: harmonium sustaining one low note the whole way through, brushes with the snares off, upright bass in long held notes, nylon-string guitar answering once a phrase, bowed vibraphone far back. A rented room at night, two beds, a window, a kettle, room mics wide, people still awake in it. Both singers are mezzos of the same age and accent, hard to tell apart: one clear and breathy with a smile in the tone, the other lower, thickening, drifting. The two lines touch on one note near the end and neither of them bends. Add players, never level.
```

---

## 3. WHAT THE REFINEMENT DID **NOT** TOUCH

- ⛔ **No new scene content.** No object, room, person, act or line of story entered at this step. Every change was a re-ordering, a re-voicing, a consonant, or a cut.
- ⛔ **The crossing was not moved, softened, doubled, or answered in the mix.**
- ⛔ **No irritation was added to the singer** to bring her into line with D2. *(F-B is declared. Complying would be the defect.)*
- ⛔ **No number entered any lyric.**
- ⛔ **Trish was not made to notice anything.** Her near-misses in V2 (*"you never said a word"*) and V1 (*"I've not seen you up this early"*) both **foreclose** rather than approach: she resolves the first into a compliment about temperament and the second into a remark about the hour. **D4 holds by construction, not by luck.**

---

## 4. AFTER-REFINEMENT MEASUREMENT — extraction printed before conclusion

```
EXTRACTED: 4 refined music prompts, char counts 923 / 899 / 926 / 928
           4 lyric bodies, sung-text char counts 3088 / 3021 / 2948 / 2654
           4 lyric line counts 92 / 90 / 89 / 82

music_prompt_chars       850-1000 :  923  899  926  928     PASS x4
music_prompt_chars_target 870-960 :  923  899  926  928     PASS x4  (none flagged)
music_prompt_hug_ceiling     >=985:  none                   PASS x4
terminal punctuation             :  "."  "."  "."  "."      PASS x4

rhyme_return_floor           0.30 : .652 .411 .449 .439     PASS x4
   content-word-only companion    : .582 .404 .386 .377     PASS x4
line_return_floor            0.20 : .272 .389 .326 .598     PASS x4
mean_words_per_line_ceiling   7.5 : 6.61 6.41 6.21 6.43     PASS x4
alliteration_per_100w_floor  11.0 :12.66 11.96 17.90 17.65  PASS x4
unique_line_ratio_floor      0.45 : .848 .778 .809 .634     PASS x4

step09 pairwise refined-prompt similarity, ceiling 0.62 (cross-pair) / 0.58 (portfolio):
   max within this pair = 0.435                              PASS
```

⚠️ **`step09_max_pair_similarity 0.62` is a CROSS-PAIR gate** and is the coordinator's to compute — handoff §2 forbids me reading a sibling's artifacts, so I do not claim a number I cannot produce. What I can assert is the input: **this pair's refined prompts contain one mandated shared string with the rest of the run (Source 2's *"spacey jazz and textured folk instruments"*, assigned to P05 alone in `step04_medium.md`) and no other phrase taken from anywhere.** The vocal spec, the room, the mode, the metre and the two-line form are this pair's alone in the grid.

**Repair budget after step 09: 1 of 3 used on each of three separate gates. All three cleared on the first attempt. Zero quarantines.**

**Gate: PASS.**

*Step 09 complete. Step 10 next: final packages, the reorder test, the describe-render self-check, and the comp hazard statement for the render audit.*
