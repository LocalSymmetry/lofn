# PAIR 05 — STEP 10 · REVISION SYNTHESIS (FINAL PACKAGES)
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P05 "TWO MINUTES EIGHTEEN"**

**ARM:** AMBITIOUS · **AXIS B:** NEWS · **VOICE:** `LOFN-PRIME (AWE mode — tender, spacious, unhurried)` · **MODE:** AWE
**Continuity payload used:** frozen ICB, LF-sha `9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a`, **142,900 B** · personality DNA **27,796 B**, inlined, unbroken, `THE ARCHIVE` absent · **18** baseline voices, skeptics at 6/12/18 · **15 Special Flairs** marker present.
**Step file:** `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` · **Inputs:** `pair_05_step06_facets.md`, `pair_05_step07_guides.md`, `pair_05_step08_generation.md`, `pair_05_step09_artist_refine.md`.

### ⛔ GOLDEN-OUTPUT QUARANTINE — `06_music_handoff.md` §1, CITED BY NAME
No golden-song payload, past shipped lyric, prior winning prompt or "reference song" entered any step of this pair. The two golden names were **not** looked up, reconstructed or calibrated against. `gates.yaml → house_lexicon` scanned across all four music prompts, all four exclude fields and all four lyrics fields: **0 hits, measured.**

---

## THE PAIR'S ARGUMENT, IN ONE PARAGRAPH

Two women share a rented room the night before they drive somewhere. One has been quietly getting ready for thirty years; the other came because she was asked. The other one wrote the booking out by hand and — **because she was holding the pen and one of the names was her own** — wrote her own name first. The one it is about is written second. She reads the page for a time, skips past her own name to get there, and thinks nothing of it. **She is happy.** ⭐ A whole life, unseen, summarised in the order of two words on a piece of paper that nobody chose — **and only the listener reads it that way.** ⛔ Nobody finds out, in the song or after it.

---

## ⭐ D3 — THE TWO LINES, NAMED BY INTERVAL *(declared at step 06, restated as the render contract)*

> Two mezzos in **D Dorian**, both inside **D4–G4**. **LINE ONE** descends **A4→D4** (5–7 syllables, downbeat entry, close-front end-vowels). **LINE TWO** ascends **C4→G4** (9–11 syllables, upbeat entry half a bar early, open-back end-vowels). They enter a **MAJOR SECOND** apart (D4 / C4) and beat. They cross **exactly once**, on **E4 — the SECOND degree of the mode — in UNISON, on the word "second"** — one falling through it, one rising through it, for one beat, **and neither adjusts.** They finish a **PERFECT FOURTH** apart (D4 / G4). ⛔ **No harmony arrives. No unison is held. They do not end together.**

**One crossing per song. Not two.** The two lines *run simultaneously* for most of each song — that is the **FORM**. They *cross* once — that is the **SEAM** (L38, N=1). In each lyrics field there is exactly one line carrying both voices on the same syllable.

---

## ⚠️⚠️ THE COMP HAZARD — the statement this pair is going to a render audit on

**THE FAILURE.** If the renderer comps the two strophes into a **duet** — harmonised, aligned, resolved, or turned into lead-and-backing-echo — **this pair's entire argument is deleted.** Not weakened. Deleted. The song becomes two friends singing a pretty jazz ballad together, which is the opposite of what it says.

**WHAT A RENDERER MUST DO FOR THE PAIR TO SURVIVE — three things, all audible:**
1. **The parenthesised lines must sound AT THE SAME TIME as the plain lines they sit between, not after them.** Overlap, not answer.
2. **The two voices must never sing the same word at the same time except once** — the `second — (second —)` line. Everywhere else they are on different words.
3. **They must not end together.** One finishes lower, one finishes higher, and in V4 one of them stops mid-word and does not come back.

**THE AUDIBLE SIGNATURE OF FAILURE — what a blind listener would hear if it broke:**
- **The tell is TIDINESS.** A clean lead vocal with a *quieter, later, shorter* second voice repeating or answering it. If the second voice ever sounds like a **backing vocal**, it has failed.
- **Thirds.** Any moment where the two voices move in parallel intervals. There are none written.
- **A shared cadence.** Both voices arriving on a held note at the end of a section, together, in tune with each other.
- ⭐ **The listener being able to follow both effortlessly.** If the simultaneous sections are comfortable, they were comped. **They are supposed to be slightly hard to parse. The effort of sorting the two voices IS the experience of the song.**

**WHERE THE DEFENCE LIVES — in the lyric and the form, never in the production spec** *(L22 THE GRAIN LAW; a Somatic objection answered in the mix is not answered)*:
- **Different syllable counts.** Falling lines run 4–7 words; climbing lines run 7–10. A backing echo is *shorter* than its lead. These are *longer*. There is nowhere to put them except over the top.
- **Different phrase onsets.** Every climbing line opens with an em-dash mid-clause — *"— and I'll take the first bit of the road —"*. It has no beginning to align to.
- **Words that cannot be sung together.** The two strophes share **no line, no phrase and almost no vocabulary.** Falling-line end-words sit in the close-front family (*shut, out, look, right*); climbing-line end-words sit in the open-back family and on broken function words (*road, before, the, I'm*). Two singers cannot blend into a unison on different vowels.
- **Neither speaker finishes a sentence**, so there is no cadence to resolve to.
- The exclude field carries `harmonised duet chorus`, `backing-vocal echo`, `two voices in thirds`, `unison refrain`, `two voices resolving together` — **belt and braces on a separate field, not a substitute for the above.**

⭐ **THE KAMASI CONDITION IS LIVE.** THE MAXIMALIST withdrew the greyness objection **conditionally, on the gap being audible.** ⛔ **Only a render audit can settle it.** If the render harmonises them, his objection is live again and this pair fails as a whole. **Flagged for `lofn-render-audit` under THE BLIND RULE — send the audio alone first, never the prompt.**

---

## PORTFOLIO MEASUREMENTS — all four, measured individually, never pair-wide

```
EXTRACTED BEFORE CONCLUDED (handoff §4: print what was extracted, assert the count)
  4 music prompts        chars 923 / 899 / 926 / 928
  4 exclude fields       chars 204 / 204 / 196 / 193
  4 Suno lyrics fields   chars 4493 / 4434 / 4411 / 4152
  4 sung-line counts           92 / 90 / 89 / 82
  4 sung-text bodies     chars 3088 / 3021 / 2948 / 2654
  instrument: scripts/measure_soundcraft.py -> profile_file(), one file per variation
```

**Within-pair distinctiveness, all six pairings, measured:**

| | lyric similarity *(ceiling 0.42)* | prompt similarity *(ceiling 0.58)* | 5-gram Jaccard *(ceiling 0.18)* |
|---|---|---|---|
| V1–V2 | 0.208 | 0.319 | 0.001 |
| V1–V3 | 0.251 | 0.435 | 0.003 |
| V1–V4 | 0.265 | 0.421 | 0.000 |
| V2–V3 | 0.269 | 0.325 | 0.001 |
| V2–V4 | 0.242 | 0.420 | 0.001 |
| V3–V4 | 0.177 | 0.277 | 0.000 |
| **max** | **0.269 ✅** | **0.435 ✅** | **0.003 ✅** |

⚠️ **The cross-PAIR gates (`step06_max_pair_similarity 0.50`, `step09_max_pair_similarity 0.62`) are the coordinator's** — handoff §2 forbids me reading a sibling artifact, so I do not report a number I cannot produce. **What is asserted is the input side:** the only string this pair shares with the rest of the run by mandate is Source 2's *"spacey jazz and textured folk instruments"* (41 chars, assigned to P05 alone in `step04_medium.md`).

---

# VARIATION 1

**Angle:** `THE PACKING` · **Hour:** early evening

## 1. MUSIC PROMPT
```text
Tender and unhurried. A spacious modal chamber-jazz ballad in D Dorian, 62 BPM, six-four, built from spacey jazz and textured folk instruments: brushes on a dry ride, upright bass walking in whole notes, nylon-string guitar picking one note a beat, a low shaker, and a modal tenor saxophone that keeps searching and never lands. Track it as a band in a rented room with two beds, a window and a kettle: room mics open wide, everyone leaking into everyone else's microphone, entries made by ear rather than by count. The singers are two women in the same range, the same age, the same unhurried northern-English delivery, close enough to be mistaken for one person. One is warm and breathy and slides down off the end of a word. The other is drier and faster and talks more than she sings. The falling line begins alone. The climbing line answers alone. Then both run at once with different words. Thicken by adding players.
```

## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, unison singing, backing-vocal echo, gospel choir stack, big final chorus, key change, EDM riser, orchestral swell, tape hiss, vinyl crackle, trap hi-hats, autotune, spoken narrator
```

## 2. LYRICS — VARIATION 1
```text
[Theme: a rented room the night before a long drive — one woman packs, the other talks, and neither of them finishes a sentence]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, entering half a bar early. Separate for the first minute. Simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - brushed kit, upright bass, wide room mics, no vocals]
*a bag set down*

[Verse 1 - EMO:Composure - Mezzo One - alone, descending, plain and unhurried]
I breathe on the black glass —
hold it high to the lamp —
looking for a little light —
and there's none. And that's right —
the second one —
the spare, the smaller one —
slides in the side pocket —
where a hand goes down in the dark —
and the pocket's packed now, so —
so the chair goes on the top —

[Verse 2 - EMO:Warmth - Mezzo Two - alone, climbing, entering early, talking more than singing]
(— and I'll take the first bit of the road —)
(— you'll be wanting to sleep for the —)
(— is that new, or have you had it —)
(— sorry. No. No, you go. I'm over the —)
(— it's just, there's a lot of it, and the —)
(— and the boot's not big, not exactly —)
(— is that new, or have you had it —)
(— we could take the long way, by the coast —)
(— and be down past the border. Past —)
(— past it before light. You'd like the —)

[Verse 3 - EMO:Contentment - Mezzo One - alone, lower, the same descent]
Then the bag will not shut —
so the sandwiches come out —
and they sit by the kettle —
so a foot finds them, out —
in the dark, in the morning —
and I'll know. I'll know what for —

[Both Voices - EMO:Serenity - Mezzo One plain and Mezzo Two in parentheses - two songs at the same time, neither waits for the other]
The strap goes over the chair —
(— and I'll take the first bit of the road —)
tucked under the arm of the chair —
(— you'll be wanting to sleep for the —)
and it's shut. It's shut —
(— is that new, or have you had it —)
and it stands by the door, shut —
(— sorry. No. No, you go. I'm over the —)
boots at the bag, toes out —
(— it's just, there's a lot of it, and the —)
so a foot goes straight out —
(— and the boot's not big, not exactly —)
and the coat's hung, sleeves out —
(— we could take the long way, by the coast —)
and the keys in the coat —
(— and be down past the border. Past —)

[Both Voices - EMO:Anticipation - the second voice keeps climbing, the first keeps falling - no harmony arrives]
And I'll want it in my hand —
(— it's only I've not seen you up this early —)
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

[Both Voices - EMO:Anticipation - THE CROSSING - both land on the same note for one beat, then each goes on her own way]
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
(— carry on. I'm not — I'm listening —)

[Both Voices - EMO:Composure - they come apart - Mezzo One settles low, Mezzo Two is still climbing]
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

[Outro - EMO:Serenity - Mezzo One alone, lowest, unhurried - Mezzo Two has stopped]
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

## 4. MEASURED — VARIATION 1
| Gate | Threshold | Measured | |
|---|---|---|---|
| `music_prompt_chars` | 850–1000 | **923** | ✅ |
| `music_prompt_chars_target` | 870–960 | **923** | ✅ |
| `music_prompt_hug_ceiling` | flag ≥985 | 923 | ✅ no flag |
| `music_prompt_terminal_punctuation` | true | `.` | ✅ |
| `suno_lyrics_field_max` | <5000 | **4,493** | ✅ |
| `suno_lyrics_field_target` | ≤4800 | 4,493 | ✅ |
| `sung_lines` | 70–120 | **92** | ✅ |
| `sung_lines_target` | 78–110 | 92 | ✅ |
| `sung_lines_floor_hug` | flag ≤72 | 92 | ✅ no flag |
| `rhyme_return_floor` | ≥0.30 | **0.652** *(content-word companion 0.582, n=55)* | ✅ |
| `line_return_floor` | ≥0.20 | **0.272** | ✅ |
| `mean_words_per_line_ceiling` | ≤7.5 | **6.61** | ✅ |
| `alliteration_per_100w_floor` | ≥11.0 | **12.66** | ✅ |
| `unique_line_ratio_floor` | ≥0.45 | **0.848** | ✅ |
| EMO header shape | 4 slots, taxonomy emotion, never bare AWE/INDIGNATION | 9/9 conform | ✅ |
| Lyrics opener | `[Theme:]` then `[SONG FORM:]` | ✅ | ✅ |
| SFX | ≥1 | **2** (`*a bag set down*`, `*a coat pocket zipped*`) | ✅ |
| `sung_numerals_spelled_out` | true | **0 digits, 0 sung numeric facts** | ✅ |
| `max_sung_numeric_facts` | 1 | **0** | ✅ |
| No real-artist names | any Suno field | **0** | ✅ |
| `house_lexicon` | 0 hits | **0** | ✅ |

**Sample EMO header:** `[Both Voices - EMO:Anticipation - THE CROSSING - both land on the same note for one beat, then each goes on her own way]`
**THE WINCE (D11):** `(— is that new, or have you had it —)`, asked twice, never finished, never answered. **The listener knows the answer. Neither woman ever says it.**
**D8:** ends on a completed physical act — the coat pocket zipped shut with the spare inside, the bag against the door, untouched.

---

# VARIATION 2

**Angle:** `THE FORECAST` · **Hour:** after the forecast

## 1. MUSIC PROMPT
```text
Warm, open, standing at a window. Slow modal chamber jazz in D Dorian, 60 BPM, six-four. Two mezzo singers are matched on purpose: same age, same accent, same weight of tone, so a listener has to work out which is which. The first is level and breathy with a small catch before every phrase; the second is quick and flat and reads aloud, breaking off halfway through what she is saying. Around them, spacey jazz and textured folk instruments: a modal tenor saxophone well forward and searching, brushes swirling on a coated head, upright bass in long slow steps, bowed vibraphone underneath, bass clarinet doubling the bass an octave up. The room is rented, two beds, a window, a kettle, and the microphones are far back so the air is part of the sound. She sings, then the other sings, then they overlap and stay overlapping, each holding her own tune. More instruments arrive. Nothing gets louder.
```

## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, call-and-response answer vocals, two voices in thirds, choir pad, build into a final chorus, drum fill transitions, EDM riser, tape hiss, vinyl crackle, cinematic strings, autotune
```

## 2. LYRICS — VARIATION 2
```text
[Theme: the same rented room, later — the kettle, the window, and a forecast that says cloud; one woman reads the sky, the other reads the phone]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, entering half a bar early. Separate for the first minute. Simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - upright bass, brushes, wide room, no vocals]
*a kettle starting up*

[Verse 1 - EMO:Composure - Mezzo One - alone, descending, at the window, plain]
The kettle's on the sill —
and the steam goes up the pane —
so I wipe it with my sleeve —
and the sleeve comes off grey —
and low down, past the grey —
it clears by the coast —
a long strip, low and clear —
it clears by the coast —
and the kettle's climbing still —
and the strip stays still —

[Verse 2 - EMO:Warmth - Mezzo Two - alone, climbing, entering early, reading the phone]
(— it's saying cloud. It's saying cloud all —)
(— all morning, and it's worse toward the —)
(— and there'll be another one, there's always —)
(— always one. There's one in — when's the —)
(— it's fine. It's fine. I'm only saying, so —)
(— so we don't get our hopes right up and —)
(— and there'll be another one, there's always —)
(— always one somewhere. Do you want the road, or —)
(— or the hotel, because if it's cloud we could —)
(— we could drive on down and find a —)

[Verse 3 - EMO:Contentment - Mezzo One - alone, lower, the same descent]
And I say yeah. I say yeah —
and I mean it. I do —
and the strip is still there —
low and long, still there —
and the kettle starts to knock —
and the mugs go out. So —

[Both Voices - EMO:Serenity - Mezzo One plain and Mezzo Two in parentheses - two songs at the same time, neither waits for the other]
So the kettle starts to knock —
(— it's saying cloud. It's saying cloud all —)
and it climbs, and it knocks —
(— all morning, and it's worse toward the —)
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

[Both Voices - EMO:Anticipation - the second voice keeps climbing, the first keeps falling - no harmony arrives]
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

[Both Voices - EMO:Anticipation - THE CROSSING - both land on the same note for one beat, then each goes on her own way]
and it climbs, and it clicks —
(— hang on. Hang on. Give it a —)
second — (second —)
and the strip's still there —
(— it's changed. It's changed. It says —)
low and long, still there —
(— it says a break in the — hang on —)
and I pour, and it's fine —
(— no. No, it's gone again. I'm —)
and it's fine either way —
(— I'm sorry, I got your hopes —)
and I'd have come anyway —
(— I got your hopes up, and I —)

[Both Voices - EMO:Composure - they come apart - Mezzo One settles low, Mezzo Two is still climbing]
I'd have come anyway —
(— and I feel bad, because I —)
and the phone goes face down —
(— because I booked it, and now —)
face down on the sill —
(— and now it's cloud. And you —)
and it clears by the coast —
(— and you never said a word. You —)
and it clears by the coast —
(— you're very good about it. You —)
and the tea's going cold —
(— you are. Right. I'm getting in. Are —)
and I like it cold. I do —
(— are you coming, or are you —)

[Outro - EMO:Serenity - Mezzo One alone, lowest, unhurried - Mezzo Two has stopped]
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

## 4. MEASURED — VARIATION 2
| Gate | Threshold | Measured | |
|---|---|---|---|
| `music_prompt_chars` | 850–1000 | **899** | ✅ |
| `music_prompt_chars_target` | 870–960 | **899** | ✅ |
| `music_prompt_hug_ceiling` | flag ≥985 | 899 | ✅ no flag |
| `music_prompt_terminal_punctuation` | true | `.` | ✅ |
| `suno_lyrics_field_max` | <5000 | **4,434** | ✅ |
| `suno_lyrics_field_target` | ≤4800 | 4,434 | ✅ |
| `sung_lines` | 70–120 | **90** | ✅ |
| `sung_lines_target` | 78–110 | 90 | ✅ |
| `sung_lines_floor_hug` | flag ≤72 | 90 | ✅ no flag |
| `rhyme_return_floor` | ≥0.30 | **0.411** *(content-word companion 0.404, n=52)* | ✅ |
| `line_return_floor` | ≥0.20 | **0.389** | ✅ |
| `mean_words_per_line_ceiling` | ≤7.5 | **6.41** | ✅ |
| `alliteration_per_100w_floor` | ≥11.0 | **11.96** | ✅ |
| `unique_line_ratio_floor` | ≥0.45 | **0.778** | ✅ |
| EMO header shape | 4 slots, taxonomy emotion | 9/9 conform | ✅ |
| Lyrics opener | `[Theme:]` then `[SONG FORM:]` | ✅ | ✅ |
| SFX | ≥1 | **2** (`*a kettle starting up*`, `*a kettle clicking off*`) | ✅ |
| `sung_numerals_spelled_out` | true | **0 digits, 0 sung numeric facts** | ✅ |
| `max_sung_numeric_facts` | 1 | **0** | ✅ |
| No real-artist names | any Suno field | **0** | ✅ |
| `house_lexicon` | 0 hits | **0** | ✅ |

**Sample EMO header:** `[Verse 3 - EMO:Contentment - Mezzo One - alone, lower, the same descent]`
**THE WINCE (D11):** *"and there'll be another one, there's always —"* → **"And I say yeah. I say yeah — and I mean it. I do —"**. ⭐ She agrees, instantly and sincerely, and the listener does the arithmetic she never will.
**D6, made explicit:** Trish blames herself — *"because I booked it, and now — and now it's cloud"* — **for the weather.** She comes within one inch of the insight and forecloses it herself: *"you're very good about it."* ⛔ She notices nothing. She never will.
**D8:** the phone face down on the sill, tea poured, and she sits.

---

# VARIATION 3 ⭐ *the pair's centre of gravity*

**Angle:** `THE HANDWRITING` · **Hour:** late, at the shelf

## 1. MUSIC PROMPT
```text
Plain, close, mid-task. Everything is played live in one small rented room, two beds, a window, a kettle, room mics wide and the chairs and the kettle audible under the take. Slow modal chamber jazz in D Dorian, 64 BPM, six-four, scored for spacey jazz and textured folk instruments: nylon-string guitar picking single notes, upright bass plucked short and dry, brushes tapping the rim more than the head, hammered dulcimer struck once a bar, and a modal tenor saxophone searching low behind the words. Two women sing in the same register and the same unhurried northern-English speech-tone, deliberately confusable; one is level and warm and breathes before each line, the other is faster, drier, half-talking, cutting herself off. A descending line in short phrases sits on the beat. A rising line in long phrases arrives early. They begin separately and end up simultaneous, keeping separate words. Add players, not volume.
```

## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, unison refrain, doubled lead vocal, layered vocal stacks, power ballad chorus, orchestral swell, drum build, EDM riser, tape hiss, vinyl crackle, spoken narration, autotune
```

## 2. LYRICS — VARIATION 3
```text
[Theme: a handwritten booking page stuck under a wet kettle — one woman reads down it for the meeting time, the other is talking about the deposit]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, entering half a bar early. Separate for the first minute. Simultaneous from the second. They meet on one note, once, and go past each other.]

[Intro - EMO:Serenity - Instrumental - nylon guitar, brushes, upright bass, wide room, no vocals]
*paper peeled off a shelf*

[Verse 1 - EMO:Composure - Mezzo One - alone, descending, plain and unhurried]
There's a ring of tea on the page —
so it's stuck to the shelf —
and it tears at the edge —
tears when I lift the page —
and the writing's on the back —
so I turn it. It's her hand —
her big hand, down the page —
Trish, and then —
Trish, and then —
and the small print under the —

[Verse 2 - EMO:Warmth - Mezzo Two - alone, climbing, entering early, talking more than singing]
(— did they take the deposit, or is it on the —)
(— because the woman on the phone said the —)
(— hang on, give me a — I've got it here —)
(— I've got it here somewhere, hang on, in the —)
(— no, that's the other one. That's the —)
(— that's the ferry. Ignore that. Right —)
(— did they take the deposit, or is it on the —)
(— because if they didn't we'll want the —)
(— we'll want the cash out before we —)
(— before we go. Are you listening? Are —)

[Verse 3 - EMO:Contentment - Mezzo One - alone, lower, the same descent]
And I'm not reading that bit —
I'm reading down the page —
past the names, down the page —
looking for the time —
just the time, that's all —
just the time —

[Both Voices - EMO:Serenity - Mezzo One plain and Mezzo Two in parentheses - two songs at the same time, neither waits for the other]
Trish, and then —
(— did they take the deposit, or is it on the —)
and my thumb's on the line —
(— because the woman on the phone said the —)
so the thumb moves down the line —
(— hang on, give me a — I've got it here —)
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

[Both Voices - EMO:Anticipation - the second voice keeps climbing, the first keeps falling - no harmony arrives]
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

[Both Voices - EMO:Anticipation - THE CROSSING - both land on the same note for one beat, then each goes on her own way]
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

[Both Voices - EMO:Composure - they come apart - Mezzo One settles low, Mezzo Two is still climbing]
and I say it out loud —
(— right. Right. So I'll get the cash —)
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

[Outro - EMO:Ennui - Mezzo Two alone, still climbing, still unfinished - Mezzo One has stopped]
(— right. Right. So that's —)
(— that's everything, then. That's —)
(— that's us, then. Are you getting —)
(— or shall I do the light? I'll do the —)
(— I'll do the light, then. Right —)
(— right. There. That's — that's —)
*paper under a kettle*
```

## 3. TITLE
**The Booking Page**

## 4. MEASURED — VARIATION 3
| Gate | Threshold | Measured | |
|---|---|---|---|
| `music_prompt_chars` | 850–1000 | **926** | ✅ |
| `music_prompt_chars_target` | 870–960 | **926** | ✅ |
| `music_prompt_hug_ceiling` | flag ≥985 | 926 | ✅ no flag |
| `music_prompt_terminal_punctuation` | true | `.` | ✅ |
| `suno_lyrics_field_max` | <5000 | **4,411** | ✅ |
| `suno_lyrics_field_target` | ≤4800 | 4,411 | ✅ |
| `sung_lines` | 70–120 | **89** | ✅ |
| `sung_lines_target` | 78–110 | 89 | ✅ |
| `sung_lines_floor_hug` | flag ≤72 | 89 | ✅ no flag |
| `rhyme_return_floor` | ≥0.30 | **0.449** *(content-word companion 0.386, n=44)* | ✅ |
| `line_return_floor` | ≥0.20 | **0.326** | ✅ |
| `mean_words_per_line_ceiling` | ≤7.5 | **6.21** | ✅ |
| `alliteration_per_100w_floor` | ≥11.0 | **17.90** | ✅ |
| `unique_line_ratio_floor` | ≥0.45 | **0.809** | ✅ |
| EMO header shape | 4 slots, taxonomy emotion | 9/9 conform | ✅ |
| Lyrics opener | `[Theme:]` then `[SONG FORM:]` | ✅ | ✅ |
| SFX | ≥1 | **2** (`*paper peeled off a shelf*`, `*paper under a kettle*`) | ✅ |
| `sung_numerals_spelled_out` | true | **0 digits, 0 sung numeric facts** | ✅ |
| `max_sung_numeric_facts` | 1 | **0** | ✅ |
| No real-artist names | any Suno field | **0** | ✅ |
| `house_lexicon` | 0 hits | **0** | ✅ |

**Sample EMO header:** `[Outro - EMO:Ennui - Mezzo Two alone, still climbing, still unfinished - Mezzo One has stopped]`
**THE WINCE (D11):** ⭐ `past the names, down the page —` and, five times, `Trish, and then —`. **She says the other woman's name out loud and skips her own to get to the time.** She is never named anywhere in this pair — ⭐ **the song withholds her name from the listener exactly as the page withholds it from her.**
**THE CROSSING, in full:** she reads the words **"second driver"** off her friend's handwriting and says *"and that's me. That's my one —"* — pleased, because the box is the one that matters for the driving rota. At that instant Trish, mid-sentence about a deposit, says *"give me a second."* **Same note. Same syllable. Two meanings. Neither adjusts.**
**D8:** ⭐ the page goes **back under the wet kettle**, into the tea ring that was already destroying it. Set down, not kept.
**Structural note:** the song ends on **the wrong voice.** Mezzo One stops; Trish carries on alone, still climbing, still unfinished, and the last word in the song is *"that's —"*.

---

# VARIATION 4

**Angle:** `THE ALARM SET FOR FOUR` · **Hour:** lights out

## 1. MUSIC PROMPT
```text
Quiet, wide awake, happy about tomorrow. Slow modal chamber jazz in D Dorian, 58 BPM, six-four. It begins with one voice on her own, then a second voice on her own, then the two of them together for the rest of the song, holding different tunes and different words until one of them simply stops mid-word and the other carries on by herself. Spacey jazz and textured folk instruments: harmonium sustaining one low note the whole way through, brushes with the snares off, upright bass in long held notes, nylon-string guitar answering once a phrase, bowed vibraphone far back. A rented room at night, two beds, a window, a kettle, room mics wide, people still awake in it. Both singers are mezzos of the same age and accent, hard to tell apart: one clear and breathy with a smile in the tone, the other lower, thickening, drifting. The two lines touch on one note near the end and neither of them bends. Add players, never level.
```

## 1B. SUNO EXCLUDE PROMPT
```text
harmonised duet chorus, whispered ASMR vocal, two voices resolving together, lullaby choir, ambient pad wash, reverse cymbal, EDM riser, tape hiss, vinyl crackle, big ending, autotune, fade-out
```

## 2. LYRICS — VARIATION 4
```text
[Theme: lights out in the shared room — one woman falls asleep, the other lies awake going through it, and sets the alarm last so the screen will not wake her]
[SONG FORM: two independent strophes. Voice One descends, short lines, on the beat. Voice Two climbs, long lines, entering half a bar early. Separate for the first minute. Simultaneous from the second. They meet on one note, once, and go past each other. The second voice stops mid-word and does not come back.]

[Intro - EMO:Serenity - Instrumental - upright bass, brushes with the snares off, harmonium holding one note, no vocals]
*a switch, then dark*

[Verse 1 - EMO:Composure - Mezzo One - alone, descending, plain and unhurried]
The light goes down and it's not dark —
there's a line under the door —
and a green light, low, on the side —
on the kettle, on the side —
and I lie on my back, and my hands —
my hands flat on my chest —
and I go through the thing once —
and I go through the thing once —
and this is the good bit —
this is the good bit —

[Verse 2 - EMO:Warmth - Mezzo Two - alone, climbing, entering early, talking more than singing]
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

[Verse 3 - EMO:Contentment - Mezzo One - alone, lower, the same descent]
And I go through the thing once —
from the door to the road —
and the road to the field —
and I know the spot. I'll stand —
and the cold coming on my hands —
and the wind. The wind at my back, and —

[Both Voices - EMO:Serenity - Mezzo One plain and Mezzo Two in parentheses - two songs at the same time, neither waits for the other]
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

[Both Voices - EMO:Anticipation - the second voice keeps climbing and starts to give out, the first keeps falling - no harmony arrives]
from the door to the road —
(— I'm going. I'm going now. I'm —)
and the road to the field —
(— going. Night. Night. Mm —)
and I know the spot. I'll stand —
(— mm. Did you — did you set the —)
and the cold coming on my hands —
(— did you set the — I'll do it, I'll do it —)
and the wind. The wind at my back, and —
(— I'll do it in a minute. Mm —)
and I'll not set it yet. Not yet —
(— mm —)
because the screen's too bright —
(— mm —)

[Both Voices - EMO:Anticipation - THE CROSSING - both land on the same note for one beat, then each goes on her own way, and the second voice does not come back]
and I'll wait. I'll wait till she's gone —
(— mm. Did you — in a —)
second — (second —)
and then I'll reach right out —
(— mm —)
and do it in the dark —
(— mm —)
and go through the thing once —
(— mm —)
and once more, and it's —
and it's not long now —
and this is the good bit —

[Outro - EMO:Serenity - Mezzo One alone, lowest, unhurried - Mezzo Two has stopped mid-word and does not come back]
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

## 4. MEASURED — VARIATION 4
| Gate | Threshold | Measured | |
|---|---|---|---|
| `music_prompt_chars` | 850–1000 | **928** | ✅ |
| `music_prompt_chars_target` | 870–960 | **928** | ✅ |
| `music_prompt_hug_ceiling` | flag ≥985 | 928 | ✅ no flag |
| `music_prompt_terminal_punctuation` | true | `.` | ✅ |
| `suno_lyrics_field_max` | <5000 | **4,152** | ✅ |
| `suno_lyrics_field_target` | ≤4800 | 4,152 | ✅ |
| `sung_lines` | 70–120 | **82** | ✅ |
| `sung_lines_target` | 78–110 | 82 | ✅ |
| `sung_lines_floor_hug` | flag ≤72 | 82 | ✅ no flag |
| `rhyme_return_floor` | ≥0.30 | **0.439** *(content-word companion 0.377, n=53)* | ✅ |
| `line_return_floor` | ≥0.20 | **0.598** | ✅ |
| `mean_words_per_line_ceiling` | ≤7.5 | **6.43** | ✅ |
| `alliteration_per_100w_floor` | ≥11.0 | **17.65** | ✅ |
| `unique_line_ratio_floor` | ≥0.45 | **0.634** | ✅ |
| EMO header shape | 4 slots, taxonomy emotion | 8/8 conform | ✅ |
| Lyrics opener | `[Theme:]` then `[SONG FORM:]` | ✅ | ✅ |
| SFX | ≥1 | **2** (`*a switch, then dark*`, `*a small alarm set*`) | ✅ |
| `sung_numerals_spelled_out` | true | **0 digits, 0 sung numeric facts** | ✅ |
| `max_sung_numeric_facts` | 1 | **0** | ✅ |
| No real-artist names | any Suno field | **0** | ✅ |
| `house_lexicon` | 0 hits | **0** | ✅ |

**Sample EMO header:** `[Verse 2 - EMO:Warmth - Mezzo Two - alone, climbing, entering early, talking more than singing]`
**THE WINCE (D11):** ⭐ `this is the good bit —`, said flat, twice, in the dark, awake, alone in a shared room. **She means it, and she is not wrong, and that is the whole injury.**
**D8:** ⭐ **the alarm is set** — for the dark, for before the light, ⛔ never for a stated hour. Then the phone goes down and her hands go back on her chest, where verse one left them.
**Form made literal:** Trish's strophe **stops mid-word and does not come back.** Mezzo One carries the last fourteen lines alone. ⭐ *They finish in different places*, audibly, with nothing to interpret.
⚠️ **`line_return 0.598` and `unique_line_ratio 0.634` are the highest in the pair, and that is deliberate and needs no defence** (`chorus_repetition_requires_no_justification: true`). Trish's strophe decays to the single syllable `mm` five times as she falls asleep. **That is a person going under, written down.** ⛔ Nothing was pre-emptively mutated to flatter a ratio.

---

## ⭐ THE REORDER TEST — D7 / finding F-A, applied per variation with the causal chain printed

> **The enforceable test: if a verse's objects can be reordered without loss, it is a list and it is a repair.**
> F-A names this pair by name: *packing for a trip is a natural inventory, and enumerations are the run's named failure mode.*

### V1 — **PASS.** The chain, and what breaks:
`breathes on the black glass → holds it to the lamp → finds no pinholes → THEREFORE it can be packed → into the side pocket → THEREFORE the pocket is full → THEREFORE the chair goes on top → THEREFORE the bag will not shut → THEREFORE the sandwiches come out → they go by the kettle → THEREFORE a foot finds them in the morning → the strap goes over the chair → THEREFORE it shuts → boots pointed toes-out → THEREFORE a foot goes straight out → the coat is hung → THEREFORE the keys go in the coat.`
**Break test:** you cannot put the chair on top before the pocket is full; you cannot take the sandwiches out before the bag refuses to shut; you cannot put the keys in a coat that has not been hung. **Every adjacent pair is a cause and an effect. Reordering does not weaken it — it makes it ungrammatical.** ✅ **And the hand does something different to each object: breathes on, slides in, straps over, takes out, points, hangs, drops in.**

### V2 — **PASS.** The chain:
`the kettle goes on the sill → its steam fogs the pane → THEREFORE she wipes it with her sleeve → THEREFORE the low strip of clear sky is visible → THEREFORE there is a road up the back of the town worth having in her head → the kettle knocks → clicks off → she pours and sits → the pane fogs again → and she lets it.`
**Break test:** the strip cannot be seen before the pane is wiped; the pane is not wiped before it fogs; it does not fog before the kettle is on. ✅ **Only three objects in the whole song — a kettle, a window, a phone — and each is handled once, differently.**

### V3 — **PASS, and this is the strictest of the four.** The chain:
`a tea ring has stuck the page to the shelf → THEREFORE it tears when she lifts it → the writing turns out to be on the back → THEREFORE she has to turn it over → THEREFORE she is looking at handwriting → her thumb is on the line → THEREFORE she moves the thumb → THEREFORE the box is uncovered → the box has a line and a line under it → the time is under the box → the tea has soaked through → THEREFORE the time is soft → THEREFORE she tilts it to the lamp → THEREFORE it comes back up → she reads it, says it twice, folds it in half → and puts it back under the kettle.`
**Break test:** she cannot read what her thumb is covering; the box is not visible until the thumb moves; the tilt is caused by the soaking, and the soaking is caused by the same ring that stuck the page down in the first line. ⭐ **The song's first object causes its last act.** ✅

### V4 — **PASS.** The chain:
`the light goes off → the room is not dark, because of a door-line and a standby light → she lies down with her hands on her chest → Trish talks, then mumbles, then goes → she has NOT set the alarm → she will not set it yet BECAUSE the screen would wake Trish → THEREFORE she waits until Trish is asleep → ONLY THEN does she reach out → turns the phone screen-to-chest so the light is on her chin and not on Trish → sets it → puts it down → puts her hands back where they started.`
⭐ **The causal chain is what puts the alarm last, not the doctrine.** D8's completed act is arrived at by the plot. **Break test:** she cannot set it before Trish is asleep without contradicting her own stated reason, which is in the lyric. ✅

---

## ⭐ THE DESCRIBE-RENDER SELF-CHECK — one inline pass, one self-repair

*Per variation: what would the prompt actually PRODUCE, then — adversarially — name **the one way this would render generic.***

### V1 — **The Folding Chair**
**Prediction.** Suno will very likely produce a warm mid-tempo jazz ballad: brushed kit, walking upright bass, nylon guitar, a competent female mezzo on the short lines, and a tenor sax that arrives as a tasteful solo in a middle section rather than searching between phrases. **The parenthesised lines are the coin toss: Suno's default reading of a standalone parenthetical is a backing-vocal echo placed *after* the lead line, not over it.**
**The one way this renders generic:** ⛔ **the parentheses become an echo, the two women become a singer and her harmony, and it is a nice jazz ballad about packing a bag.**
**Self-repair applied (once).** The parenthetical lines were rewritten to be *longer* than the lines they follow — 7–10 words against 4–6 — to **open mid-clause on an em-dash**, and to share **no word** with the line above. **An echo has nothing to echo and nowhere to attach; the only place the words fit is on top.** Plus `backing-vocal echo` and `unison singing` in the exclude field.

### V2 — **Cloud Off The Coast**
**Prediction.** The most conventional-sounding of the four and the most likely to come back beautiful: sax-forward slow jazz, bowed vibraphone, a lot of air. The risk is not ugliness, it is **charm** — the sax takes a full chorus, the two voices settle into verse-and-response, and it becomes a mood piece about weather.
**The one way this renders generic:** ⛔ **it becomes a pretty song about a rainy holiday**, and the fact that one of these women has been waiting thirty years disappears entirely.
**Self-repair applied (once).** Trish's strophe was rewritten so that **every one of her lines is a question, an apology, or a logistics offer** — there is no melodic shape in them to promote to a hook — and the falling refrain was fixed as a flat statement of fact (*"it clears by the coast"*) rather than a wish, so a renderer cannot make it yearn.

### V3 — **The Booking Page**
**Prediction.** The sparsest arrangement and the closest to speech; likely to come back talky, guitar-and-bass, with the voices well forward. **The specific risk is not genericism, it is a punchline:** if the renderer places Trish's *"give me a second"* as an **answer** to *"the box says —"*, the crossing becomes a joke and Trish becomes complicit in the wound.
**The one way this renders generic:** ⛔ **`Trish, and then —` gets sung as a lament** — held, with feeling — which tells the audience that the singer knows, and destroys D1 in three words.
**Self-repair applied (once).** `Trish, and then —` is three words on a falling third with no vowel worth sustaining, and it is surrounded by administrative nouns (thumb, line, box, time) so there is no emotional bed to sing it from. Trish's crossing line is **already in progress** (*"hang on. Hang on. Give me a —"*) before the box line lands, and her very next line is *"sorry. Sorry. You go. What —"*, which proves she was not listening. `power ballad chorus` and `doubled lead vocal` added to the exclude field.

### V4 — **The Line Under The Door**
**Prediction.** Probably the truest of the four, because Trish's part decays to the single syllable `mm` and then stops — **there is literally nothing left to comp into a duet.** Expect harmonium, brushes, bass, and one voice alone for the last minute.
**The one way this renders generic:** ⛔ **a swell and a fade-out.** The generator's strongest instinct is to make the final refrain bigger and then dissolve it, which would turn *"this is the good bit"* into a consolation — the exact opposite of the line.
**Self-repair applied (once).** The final line is the refrain said **flat, for the second time in a row**, immediately after a physical act, and the song's last event is an SFX cue rather than a vocal. `big ending`, `fade-out` and `ambient pad wash` are in the exclude field, and the music prompt ends *"Add players, never level."*

**Self-repair budget: one pass per variation, used, not repeated.**

---

## THE BINDING DECISIONS — checked one at a time, per variation

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| **D1** singer never arrives | ✅ she never wonders why Trish keeps asking | ✅ she agrees and means it | ✅ she reads past her own name | ✅ she calls it the good bit |
| **D2 / F-B** register | ⚠️ **tender, not irritated — DECLARED** (step 05 F-B). *Unmoved by the thing the listener is moved by:* ✅ in all four |||| 
| **D3** two lines by interval | ✅ | ✅ | ✅ | ✅ — declared at step 06 before a word was written |
| **D4** vindication ban | ✅ Trish's question dies twice | ✅ Trish forecloses it into a compliment | ✅ Trish never looks at the page | ✅ Trish is asleep |
| **D5** present tense, listener as defendant | ✅ first person, present, both voices |||| 
| **D6** skill not sin | ✅ | ✅ Trish blames herself for the weather | ✅ **a hand wrote its own name first — no crime exists** | ✅ |
| **D7** no enumerations | ✅ causal chain | ✅ three objects | ✅ strictest chain in the pair | ✅ |
| **D8** does not end with the object kept | ✅ pocket zipped, bag untouched by the door | ✅ phone face down, tea poured | ✅ **page back under the wet kettle** | ✅ **alarm set, phone down** |
| **D9** appropriation gate | **N/A — P01 and P02 only** (`05_pair_assignments.md`). No tradition named in any render field regardless |||| 
| **D10** unspent, not sepia | ✅ nothing is *found*; every object is in use, wet, torn or full |||| 
| **D11** gap audible | render-dependent — **stated above, flagged to the audit** |||| 

---

## ⛔ THE RUN'S CRITICAL REQUIREMENT — reported honestly, including where it is weakest

> *"A stranger must be able to point at the thing being set aside, and at the hand setting it aside, inside the first thirty seconds, without knowing any astronomy."*

**THE HAND is pointable inside thirty seconds in all four variations.** A woman's hands are doing something specific in the first four lines of every one: breathing on black glass and holding it to a lamp; wiping a fogged pane with a sleeve; peeling a stuck page off a shelf; lying down with her hands flat on her chest.

⚠️ **THE DISCARD is pointable inside thirty seconds in V3 only** — the handwriting, the two names, the box. In V1, V2 and V4 the thing being set aside is **her own claim on the trip**, and it becomes pointable at roughly forty-five to ninety seconds, when the second voice starts asking questions that go unanswered.

**This is reported, not laundered.** It is the price the AMBITIOUS arm was authorised to pay — `05_pair_assignments.md` sets this pair's eligibility at **0–3/7 by design** and states that *"two simultaneous strophes defeat cognitive ease on purpose."* **V3 is the pair's legibility card and is already named as its centre of gravity.** ⭐ **If the coordinator has to cut this pair to one variation, cut to V3.**

**THE SMALL ROOM TEST — passes strongly.** Two women at a desk, unamplified, one singing short falling lines and one talking across her, is the entire piece with nothing removed. ⛔ No astronomy is explained anywhere: the words *eclipse, sun, moon, sky, star, shadow, corona, solar, totality, filter* appear **zero times**, measured with word boundaries across all four lyrics.

---

## ⚠️ ONE-FACT RULE — disclosed, because a naive count will get this wrong

**`max_sung_numeric_facts: 1` · this pair sings ZERO.** P04 spends the run's one sung numeric fact. Nothing in these four songs states a quantity.

**The word `second` will trip a numeric scanner and it is not a number.** It occurs 2–4 times per variation as the pair's **return vehicle** (Axis 5, *a one-word refrain that changes meaning by position*): the spare piece of glass; *hang on a second*; *give it a second*; and the box on the page marked **second driver**. ⭐ **None of these states a quantity, a measurement, or a fact.** All are spelled in words, satisfying `sung_numerals_spelled_out` trivially.

Also present and also not facts: `first` (*"the first bit of the road"* — an idiom), `one` (*"another one"*, *"my one"* — a pronoun), `half` (*"I fold it in half"*), `minute` (*"in a minute"* — meaning *shortly*).
⛔ **Never sung, deliberately:** *two minutes eighteen* · *thirty years* · the hour of the alarm · any cloud percentage. **Digit characters in sung lines across all four variations: zero, measured.**

---

## HUMAN-SUBJECT STANDARD — judged directly on content

`vault/HUMAN_SUBJECT_STANDARD.md`; handoff §5.
- **Both women are invented.** One is deliberately **never named** — that is a formal device, not an omission. The other is **Trish**: an ordinary invented first name, no surname, no employer, no location, no identifying characteristic, not modelled on any person living or dead.
- **The eclipse of 12 August 2026 is an occasion on a public calendar**, used as the *reason* two invented people are in a room and never as a subject. ⛔ **It is not named or described in any lyric.**
- **Messier and Tempel appear nowhere** and are given no interiority.
- ⛔ **Neither of today's two real deaths is present, alluded to, or gestured at.** Nobody in this pair is bereaved, ill, or dying. **REAL GRIEF IS NOT RAW MATERIAL.**
- **`scripts/check_human_subjects.py` is deliberately not cited** — handoff §4 records that it returns `HOLD_FOR_HUMAN` on 100% of correctly-written artifacts in this checkout because spaCy is absent and its regex fallback reads capitalised bracket tokens as person names. **Reporting its output in either direction would be laundering.** Judged on content, above. **CLEAR.**

---

## LINEAGE & CREDIT

⛔ **D9 (THE APPROPRIATION GATE) does not apply to this pair.** The coordinator's scope call (`05_pair_assignments.md`, stated in the open so QA can challenge it) puts the gate on **P01** (Ethio-jazz function) and **P02** (gospel close-harmony function) — the run's two of six. This pair is written from **broadly shared chamber-jazz and chamber-folk vocabulary**: modal jazz ballad, brushes, upright bass, nylon guitar, harmonium, hammered dulcimer, tenor saxophone.

**Held to the gate's spirit anyway, because the Simon seat's objection was explicitly not withdrawn and *"the intent is never the issue"*:**
- ⛔ **No tradition's name appears in any Suno-bound field.** Verified across all four music prompts, all four exclude fields, and all four lyrics fields.
- ⛔ **The word *tezeta* appears nowhere in this pair.** Flair 11 (*Tezeta Function*) was **available in the shared palette and deliberately declined at step 07** — a pair outside the gate's scope has no business borrowing a mode's name even as a private label.
- ⛔ **No real-artist name in any field.** The panel constructs are named in this pair's *artifact prose only*, are **"after" / influence and never endorsement**, and no construct states or implies that its source figure said, reviewed, approved or would approve anything here.
- **Vocabulary imported from Source 2 under the run's declaration:** the phrase *"spacey jazz and textured folk instruments"* is taken **verbatim as vocabulary only** from the Bandcamp Album of the Day review of **Papangu — *Celestial*** (7 August 2026), per `00_research_brief.md` §3 and `step04_medium.md`. ⛔ **Nothing about that record, that band, analogue-versus-digital, tape, or formats is a subject anywhere in this pair.** Source: <https://daily.bandcamp.com/album-of-the-day> · artist: <https://papangu.bandcamp.com/>

---

## STEP-10 SELF-CHECK — EXTRACTION PRINTED BEFORE CONCLUSION

```
INSTRUMENT: scripts/measure_soundcraft.py -> profile_file(), run per variation on an
            isolated file in _work/pair_05/, so no cross-variation aggregation can
            flatter or damn a single number.

EXTRACTED   V1     V2     V3     V4     threshold        verdict
lyrics field chars 4493   4434   4411   4152   <5000 / <=4800   PASS x4
sung lines         92     90     89     82     70-120 / 78-110  PASS x4
prompt chars       923    899    926    928    850-1000/870-960 PASS x4
end_rhyme        0.652  0.411  0.449  0.439   >= 0.30          PASS x4
 content-word only 0.582 0.404  0.386  0.377   >= 0.30          PASS x4
line_return      0.272  0.389  0.326  0.598   >= 0.20          PASS x4
words_per_line    6.61   6.41   6.21   6.43   <= 7.5           PASS x4
allit_per_100w   12.66  11.96  17.90  17.65   >= 11.0          PASS x4
unique_line_ratio 0.848  0.778  0.809  0.634  >= 0.45          PASS x4
digits in sung lines  0      0      0      0   0                PASS x4
astronomy words       0      0      0      0   0                PASS x4
house_lexicon hits    0      0      0      0   0                PASS x4
real-artist names     0      0      0      0   0                PASS x4
banned amplitude      0      0      0      0   0                PASS x4
SFX cues              2      2      2      2   >= 1             PASS x4
EMO headers conform  9/9    9/9    9/9    8/8   4 slots each     PASS x4
```

⚠️ **A wordless return device would inflate `line_return`** — the handoff's warning from 2026-08-06. **This pair has no vocable, no hum and no non-lexical hook.** Every returning line is words. **The only single-syllable return is `mm` in V4**, five occurrences, and it is a person falling asleep, not a hook. **The content-word-only rhyme companion is reported above for exactly this reason: so the instrument cannot be mistaken for the craft.**

**Repair budget across the whole pair: 3 gates repaired, 1 attempt each, all cleared on the first attempt. Zero quarantines. Zero gates outside band at final.**

**GATE: PASS.**

---

*Step 10 complete. This pair goes to the render audit under THE BLIND RULE with one live condition: **if the two strophes come back harmonised, aligned, or resolved, the pair has failed and the Kamasi objection is live again.** Send the audio alone. Never the prompt.*
