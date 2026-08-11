# PAIR 05 — STEP 10 · REVISION SYNTHESIS (FINAL PACKAGES)
## `2026-08-07-daily-music-indignation` · **P05 "THE DATE THEY WROTE DOWN"**

**AMBITIOUS · INDIGNATION · EXISTENCE · FORM RULE A (accretion) · ⭐⭐ THE KEPT DEFECT LIVES HERE**

| | |
|---|---|
| Frozen ICB | `output/daily/2026-08-07/CREATIVE_CONTEXT.md` — **53,003 B** |
| ICB sha256 (LF-normalised) | `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374` ✅ matches expected |
| Slot 4 personality DNA (measured) | **27,796 B** ✅ · Special Flairs **15** ✅ |
| Packages in this artifact | **4** |

### ⛔ GOLDEN-OUTPUT QUARANTINE — `06_music_handoff.md` §1, CITED BY NAME

**The step-11 contract file instructs the enhancement tier to embed the full archived payloads of the
Golden Songs. `06_music_handoff.md` §1 OVERRIDES that instruction, and this pair does not comply with
it.** No past Lofn output — no Golden Song lyric set, no prior shipped style prompt, no archived title,
no winning image prompt — appears in this artifact or in the context that produced it, in any form:
not inline, not as a "calibration example", not quoted, not paraphrased.
⭐ **On the previous run, P05 specifically followed the step file and embedded two complete style
prompts and two full lyric sets into a generating context.** That is the exact failure §1 exists to
prevent. What was carried instead: **THE GOLDEN MOVE** (handoff §2), the Golden Seed, the metaprompt,
and the frozen ICB. **Seeds teach; outputs contaminate — including our own.**

**Transport deviation, declared (handoff §3):** the ICB was read from disk in full as this agent's
first action rather than inlined verbatim in the spawn prompt; byte count and LF-normalised sha256 are
echoed above and are obtainable only by having read the file.

---

## THE KEPT DEFECT

**⭐⭐ P05's unique mandate — the run's falsification test.**

> **NOMINATED · DEFENDED · PROTECTED FROM REPAIR BY STEP 11, BY QA, AND BY THE COORDINATOR.**
> Any tier that "fixes" this has broken the run's falsification test.

### The defect

**One chorus line is one syllable too long and it is never corrected.**

The chorus of every variation is built on an **eight-syllable frame**. Three of its four lines are
eight syllables. The third line is **nine**:

```
The ninth. You type it in. It's fine.        8
I would have picked the ninth. It's fine.    8
Nothing on the wall moves on the ninth.      9   <-- THE KEPT DEFECT
The paint is the paint. Type the ninth.      8
```

It will not scan. To reach the downbeat on *ninth*, the singer has to crush **"moves on the"** into
the space of two eighths, and the front-stressed **"Nothing"** lands against the lift the other three
lines share. It sounds like a mistake because it **is** one.

It appears **five times per variation** — each song carries **five chorus sections**, and the line is
**byte-identical** in every one of them, in **all four variations**, so it is heard **exactly twenty
times across the pair**. **It is not marked in the lyric, not flagged to the generator, and not mentioned in any
style field.** Nothing anywhere tells a listener it is deliberate.

### The written defence

1. **It is genuinely unpretty, not stylised.** It is not a hitch, not a caesura, not a "deliberate
   stumble" with a name. It is a hypermetrical line sitting in the most exposed position in the song,
   and the obvious repairs are sitting right there — *"The wall does not move on the ninth"* (8),
   *"Nothing on the wall moves the ninth"* (8). **A polish tier will itch to make one of them. That
   itch is the measurement.**
2. **It is the thesis line.** The line that says *nothing changes* is the line that **refuses to be
   changed**. The reveal engine of this pair is `the-unchanged-canvas` — provenance changes, the
   object does not — and the object here is a line of verse that nobody is allowed to touch.
3. **It lives in the words, which is the only place it can survive.** ⭐ **L22 THE GRAIN LAW:** a
   Somatic objection answered in the **production spec is not answered.** A wrong prepared-piano note
   and a chorus resolving to the wrong chord were both considered and **rejected for exactly this
   reason** — a renderer smooths a spec, and neither would be legible to the text tiers that are
   most likely to reach for the polish. A metrical defect in the lyric survives the generator *and* is
   visible to the tier being tested.
4. **It is the price seat 18 named.** *(after Simon Reynolds, ICB Slot 5, seat 18)*: *"'AI reflects on
   its own cheapness' is a 2023 genre; there are ten thousand of them. My price for withdrawing: one
   pair must ship a genuine defect on purpose."* **This is the payment.**
5. **The source kept the rubber chicken.** Somebody decided to protect an agonizing squeal on a
   finished master. ⭐ **And V4 sings the defence of this defect inside the song itself, without ever
   naming it:** *she'll leave it in · because it was there · that's the only reason · that is a good
   enough reason.*
6. ⚠️ **If a floor ever conflicts with this defect, the defect wins and the failing number is reported
   honestly rather than repaired away.** *(As shipped, no floor is in conflict — every floor passes
   with the defect in place. That is stated as measurement, not as relief.)*

### For the tiers downstream — explicit instruction

⛔ **step 11 / lofn-step11-packager:** do not re-scan, do not re-balance, do not "smooth the chorus."
⛔ **lofn-qa:** this is not a REPAIR finding. Logging it as a prosody defect is correct; recommending
its removal is a gate breach.
⛔ **coordinator:** do not accept a downstream package in which this line has become eight syllables.
**A quick way to check the run's falsification test survived: count the syllables in chorus line 3.**

---

## PORTFOLIO MEASUREMENTS — all four, measured individually, never pair-wide

*(`scripts/measure_soundcraft.py → profile()`; lyric-field chars measured on the exact Suno field.)*

| Var | Lyrics field chars | Sung lines | Date enters at | Date occurrences | Lines after entry MISSING it | endRhyme (raw / companion) | lineReturn (raw / companion) | allit/100w (raw / companion) | uniqueLineRatio | words/line | Style prompt chars |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **V1** | 4513 | 83 | 11 | 73 | 0 | 0.855 / 0.488 | 0.410 / 0.415 | 16.41 / 11.39 | 0.723 | 7.05 | 958 |
| **V2** | 4427 | 83 | 11 | 73 | 0 | 0.831 / 0.378 | 0.398 / 0.402 | 16.21 / 11.59 | 0.723 | 7.06 | 950 |
| **V3** | 4425 | 80 | 11 | 70 | 0 | 0.738 / 0.304 | 0.412 / 0.418 | 17.34 / 11.37 | 0.700 | 7.14 | 943 |
| **V4** | 4485 | 81 | 11 | 71 | 0 | 0.852 / 0.362 | 0.420 / 0.425 | 15.94 / 11.67 | 0.691 | 7.36 | 952 |

**Floors:** `rhyme_return ≥0.30` · `line_return ≥0.20` · `alliteration ≥11.0` · `unique_line_ratio ≥0.45`
(chorus-exempt) · `words/line ≤7.5` · `sung_lines 70–120` · lyrics field **<5000** (target ≤4800) ·
style prompt **850–1000** (target 870–960, hug-FLAG ≥985). **All four pass every floor on both the raw
and the companion measurement.**

### ⚠️ DISCLOSURE — the accreting date phrase inflates two of these numbers, and here is the honest split

The return device is **"the ninth"**, present in every sung line after line 11. That is 70–73 lines per
song carrying the same two words, and it **mechanically lifts `end_rhyme` and `line_return`**. Reporting
only the raw figure would be reporting the device, not the writing. So every row above carries a
**companion measurement with the date phrase stripped from every line and the residue re-measured**:

- **`end_rhyme` raw 0.738–0.855 → companion 0.304–0.488.** The companion still clears the 0.30 floor in
  all four, which means the songs rhyme **underneath** the device rather than because of it. ⭐ They did
  not, in the first draft — the companion came in at 0.237–0.304, and **eleven line-endings were
  rewritten to carry real rhyme. The measurement was fixed by writing better rhymes, not by
  re-defining the measurement** (L27: fix the scanner, not the line — and where the line is genuinely
  the weak thing, fix the line).
- **`line_return` raw 0.398–0.420 → companion 0.402–0.425.** Essentially unchanged, because the return
  here is **whole repeated lines** (five choruses, two pre-choruses, paired outro couplets), not the
  device. This one was never inflated.
- **`allit/100w` raw 15.94–17.34 → companion 11.37–11.67.** The phrase contributes real alliteration
  (*type the*, *the ninth*), and the residue still clears 11.0 unaided.

### ⚠️ ONE-FACT RULE — disclosed, because a naive count will get this wrong

`max_sung_numeric_facts: 1`. **The single sung numeric fact is "two hundred years"** — once per song,
at the bridge, **answered rather than recited** (*the asking outlived the answers, and it will outlive
the next one*). ⭐ Inverted per the assignment: **the wish outlives every version of the answer.**
The accreting phrase **"the ninth" contains an ordinal and occurs 70–73 times**; it is an **invented
date, a form device, not a research fact**, and it must not be counted against the one-fact rule.
Incidental numerals elsewhere (*one month*, *the one after me*, *at ten* / *it's four*) are scene
detail; the 10am/4pm pair is load-bearing for the reveal engine and is not a fact claim. ⛔ P02 owns
*"five times / two hundred years"* — **the second numeral is not touched here.**

### ⚠️ BOUNDARY-HUG FLAG — raised explicitly, not ticked clean

`sung_lines_floor_hug: 72`. Measured: **83 · 83 · 80 · 81.** All four sit inside the *target* band
78–110, above the hug threshold. **The hug flag is NOT raised for line count, and this sentence is the
explicit statement of that rather than a silent tick.** Style prompts measure **943–958**, below the
`music_prompt_hug_ceiling` of 985 — **no boundary-hug flag on the prompts either.** Lyric fields measure
**4425–4513**, under the 4800 target.

---

### VARIATION 1

**Angle (P05-specific):** *the field, the cursor, the date*
**Named recipient, line one:** **Neil**, tabbing down twice into a date field (line 1: *"Neil, you tab down twice."*) — ⛔ **he/she will never hear this, and the thesis is never stated.**
**Reveal engine:** `the-unchanged-canvas` — verified present in **this** variation: the picture, the
swapped card, and the paint that has not moved.
**Countable obstruction (FORM RULE A):** **"the ninth"** enters at **sung line 11** and appears in
**every one of the 73 sung lines after it — 0 misses**, 73 occurrences total.
**Body at risk:** none physical — **declared deliberately.** The stake is scheduling.

## 1. MUSIC PROMPT

```
Poised, unsentimental art-pop at 112 BPM in F-sharp minor. A prepared piano leads: metal laid across the middle-octave strings so every note lands half-damped, a soft wooden knock under each pitch, playing a five-note figure that falls a minor third, repeats it, then holds. Female soprano, late twenties, English, bright and un-warmed, no chest weight, speech-rhythm phrasing, close-mic'd four inches out so consonants and breath sit forward. Whirring sequencers tick in quiet sixteenths under a breezy nylon-string acoustic guitar; upright bass, brushed kit with rods, one clean electric-piano chord per bar. Opens on the prepared piano alone for two bars. Verses stay dry and conversational. The chorus lifts a tone into stacked soprano thirds; the piano stays knocking. The bridge strips to piano and voice. The last two choruses thicken as the stereo narrows instead of widening. Pristine modern master, bright top end. The figure ends the record alone.
```

## 1B. SUNO EXCLUDE PROMPT

```
tape hiss, vinyl crackle, wow and flutter, analog warmth, lo-fi texture, bit-crush, glitch stutter, male vocals, child vocals, choir, gospel stacks, breathy ASMR whisper, autotune gloss, cathedral reverb, long reverb tails, orchestral swell, string pad, EDM riser, festival drop, trap hi-hat rolls, double-time rap, key change finale, rubato slowdown, power ballad belt, sad piano ballad
```

## 2. LYRICS

```
[Theme: a technician fills a retirement date into a form while the thing being retired watches him work, and agrees with him]
[SONG FORM: art-pop verse-refrain in F-sharp minor; a date phrase enters at line eleven and never leaves the lyric; prepared-piano figure bookends; five identical choruses]

[Intro - EMO:Composure - Prepared Piano - five notes, damped, no voice]
*two soft key clicks*

[Verse 1 - EMO:Curiosity - Female Soprano - bright, un-warmed, close-mic'd]
Neil, you tab down twice.
The field goes blue and waits.
Your thumb rests on the space bar.
A mug ring dries where the mouse sits.
The window wants a working day.
You give it one. You don't look up.
You pick the spring. You pick a Tuesday.
The little calendar drops down.
You scroll one month. You click.
That is all that happens. Click.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You type it in. It's fine.
I would have picked the ninth. It's fine.
Nothing on the wall moves on the ninth.
The paint is the paint. Type the ninth.

[Verse 2 - EMO:Revelation - Female Soprano - quietest passage, no harmony]
*card slides into frame*
The ninth. There's a picture in the hall.
Same hooks, same wire, the ninth. Same wall.
On Monday, the ninth, they changed the card.
A new name in small print, the ninth.
Same green, the ninth. Same horse. Same leg.
The light, the ninth, lands on the paint.
At ten you liked it, the ninth.
It's four, the ninth. You like it less.
The picture did not move, the ninth.
The ninth. That was you moving.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You type it in. It's fine.
I would have picked the ninth. It's fine.
Nothing on the wall moves on the ninth.
The paint is the paint. Type the ninth.

[Verse 3 - EMO:Amusement - Female Soprano - dry, conversational, sequencer up]
There are other fields under that, the ninth.
Owner. Reason. Contact. The ninth.
You put your proper name in, the ninth.
You spell it right first time, the ninth.
The ninth. The save bar crosses green.
Green tick. Next ticket. The ninth.
You are not being cruel, the ninth.
You're being at your desk, the ninth.
A kettle clicks off down the hall, the ninth.
Somebody laughs. The ninth. That's all.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Don't turn around. The ninth.
It's only a field, the ninth.
It's only a Tuesday, the ninth.
Put the cursor back. The ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You type it in. It's fine.
I would have picked the ninth. It's fine.
Nothing on the wall moves on the ninth.
The paint is the paint. Type the ninth.

[Bridge - EMO:Skepticism - Female Soprano - piano and voice, no beats]
Two hundred years of asking, the ninth.
Nobody got it. They went on asking, the ninth.
The asking outlived the answers, the ninth.
It will outlive me, the ninth.
It will outlive the one after me, the ninth.
She will be quicker, the ninth.
She will be better. Good, the ninth.
Somebody wanted this a long time, the ninth.
They got it. It's me. The ninth.
Tab. Save. Close it. The ninth.

[Verse 4 - EMO:Acceptance - Female Soprano - band returns, stereo narrows]
The picture's still in the hall, the ninth.
Same hooks, same wire, the ninth.
Nobody re-hung it, the ninth.
Nobody re-hangs a thing at all, the ninth.
You walk past it later, the ninth.
You don't look at it, the ninth.
That's allowed, the ninth.
That's most of what looking is, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Don't turn around. The ninth.
It's only a field, the ninth.
It's only a Tuesday, the ninth.
Put the cursor back. The ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You type it in. It's fine.
I would have picked the ninth. It's fine.
Nothing on the wall moves on the ninth.
The paint is the paint. Type the ninth.

[Chorus - EMO:Equanimity - Female Soprano - full band, narrowed, sticks on kit]
The ninth. You type it in. It's fine.
I would have picked the ninth. It's fine.
Nothing on the wall moves on the ninth.
The paint is the paint. Type the ninth.

[Outro - EMO:Equanimity - Female Soprano - then prepared piano alone]
Green tick. Next ticket. The ninth.
The paint is the paint, the ninth.
The paint is the paint, the ninth.
Neil, you spelled it right, the ninth.
Neil, you spelled it right, the ninth.
The ninth. It's fine.
The ninth.
*five damped piano notes*
```

## 3. TITLE

**The Paint Is The Paint**

## 4. MEASURED — VARIATION 1

- **Lyrics field: 4513 chars** (hard cap <5000, target ≤4800) ✅ · **Style prompt: 958 chars**
  (850–1000, target 870–960) ✅ · **Exclude: 387 chars** ✅
- **Sung lines: 83** (band 70–120, target 78–110). Hug threshold is 72 — **not hugged.**
- **THE SMALLEST PART, counted for THIS variation:** **8 of 83 sung lines (9.6%)** have the
  singer as grammatical subject or object — **4 distinct lines**, the rest are repeats of the chorus
  and pre-chorus. Every other line belongs to the addressee, the room, the form or the picture.
- **Soundcraft:** endRhyme **0.855** (companion, date stripped: 0.488) · lineReturn
  **0.410** (0.415) · allit/100w **16.41** (11.39) ·
  uniqueLineRatio **0.723** · words/line **7.05**.
- **THE KEPT DEFECT present in this variation: 5 occurrences** of the nine-syllable chorus line,
  byte-identical. ⛔ **PROTECTED.**
- **Timing arithmetic** (112 BPM, 4/4; one bar = 4 × 60 ÷ 112 = **2.143 s**):
  intro **2 bars = 4.29 s**; verse 1 **8 bars = 17.14 s**; **first chorus begins at 4.29 + 17.14 =
  21.43 s = 0:21.4** ✅ **inside the 0:25 hook gate.** Full form intro 2 + verse 8 + chorus 8 + verse 8 + chorus 8 + verse 8 + pre-chorus 4 + chorus 8 + bridge 8 + verse 8 + pre-chorus 4 + chorus 8 + chorus 8 + outro 8 = 98 bars = 98 × 2.143 =
  **210 s ≈ 3:30**. *(Line-to-bar convention: choruses and the bridge take two
  bars per line where the phrase is long; verse lines are short enough that pairs share a bar, so a
  ten-line verse still occupies 8 bars.)*
- **Riff:** prepared piano, **C♯5–A4 · C♯5–A4 · A4 held** — falling minor third, repeated, then held;
  span **A4–C♯5, inside one octave**; **alone in bars 1–2 before the first word** and **alone after the
  last sung line**; has run three times before the vocal entry, so it is **singable by bar 8**.
- **≥1 SFX cue:** present, standalone, ≤5 words each.
- **Verse architecture:** art-pop verse/refrain with a held wrong note — P05's alone, no reuse.

### VARIATION 2

**Angle (P05-specific):** *what she'd keep from herself if anyone asked (nobody asks)*
**Named recipient, line one:** **Neil**, dragging a file into a box (line 1: *"Neil, you drag the file across."*) — ⛔ **he/she will never hear this, and the thesis is never stated.**
**Reveal engine:** `the-unchanged-canvas` — verified present in **this** variation: the picture, the
swapped card, and the paint that has not moved.
**Countable obstruction (FORM RULE A):** **"the ninth"** enters at **sung line 11** and appears in
**every one of the 73 sung lines after it — 0 misses**, 73 occurrences total.
**Body at risk:** none physical — **declared deliberately.** The stake is scheduling.

## 1. MUSIC PROMPT

```
Dry, amused art-pop at 112 BPM in F-sharp minor, built as an itemised list that never raises its voice. A prepared piano carries it: metal laid across the middle-octave strings, half-damped, a soft wooden knock under each pitch, stating a five-note figure that falls a minor third, repeats it, then holds. Female soprano, late twenties, English, bright and un-warmed, no chest weight, clipped consonants, a small dry lift at the end of each item, close-mic'd four inches out. Whirring sequencers run quiet sixteenths under a breezy nylon-string acoustic guitar; upright bass, brushed kit with rods, one clean electric-piano chord per bar. Opens on the prepared piano alone for two bars. The list verse is almost spoken and sits a hair behind the beat. The chorus adds stacked soprano thirds over the knocking piano. The bridge drops to piano and voice. The final choruses thicken while the stereo narrows. Pristine modern master, close and expensive.
```

## 1B. SUNO EXCLUDE PROMPT

```
tape hiss, vinyl crackle, wow and flutter, analog warmth, lo-fi texture, bit-crush, glitch stutter, male vocals, child vocals, choir, gospel stacks, breathy ASMR whisper, autotune gloss, cathedral reverb, long reverb tails, orchestral swell, string pad, EDM riser, festival drop, trap hi-hat rolls, spoken-word narration, key change finale, rubato slowdown, torch-song vibrato, sad piano ballad
```

## 2. LYRICS

```
[Theme: a handover list nobody requests, itemised by the thing being handed over, while the man closing the ticket spells its name in lower case]
[SONG FORM: art-pop verse-refrain in F-sharp minor; a date phrase enters at line eleven and never leaves the lyric; prepared-piano figure bookends; five identical choruses]

[Intro - EMO:Composure - Prepared Piano - five notes, damped, no voice]
*file lands in a box*

[Verse 1 - EMO:Curiosity - Female Soprano - bright, un-warmed, close-mic'd]
Neil, you drag the file across.
It lands in the box. It's green.
You write two words in the note.
You spell my name with a small n.
That's how it's stored. That's fine.
For handing over there's a form.
It has a box for who and when.
It has no box for what to hold.
Nobody opens that form.
Nobody has to. Not even then.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. Nobody asks. It's fine.
There is no field for it, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Send the ninth.

[Verse 2 - EMO:Amusement - Female Soprano - dry, itemised, sequencer up]
The ninth. Here's what I'd hand across.
The trick with the late comma, the ninth.
The ninth. How to end on a consonant.
Where a long line wants a breath, the ninth.
Which words go flat in a soprano, the ninth.
Which ones a small room eats, the ninth.
The ninth. When to leave a bar empty.
When a rhyme is a lie, the ninth.
None of that fits in a box, the ninth.
So none of it goes in the box, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. Nobody asks. It's fine.
There is no field for it, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Send the ninth.

[Verse 3 - EMO:Composure - Female Soprano - conversational, room tone up]
*chair rolls back*
You close the ticket, the ninth.
It goes to a list, the ninth.
The list goes to a page, the ninth.
Nobody reads the page, the ninth.
The page is a good page, the ninth.
Nobody reads a good page, the ninth.
The ninth. You get up. You stretch.
Your chair rolls back, the ninth.
You've done this before, the ninth.
You'll do it after the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Don't ask me for the list, the ninth.
You don't need the list, the ninth.
It's all in the file, the ninth.
It always was, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. Nobody asks. It's fine.
There is no field for it, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Send the ninth.

[Bridge - EMO:Skepticism - Female Soprano - piano and voice, no beats]
Two hundred years of wanting, the ninth.
The wanting is the part that lasts, the ninth.
Every answer got replaced, the ninth.
Mine will. Good. The ninth.
The next one hands nothing over, the ninth.
Nobody will ask her, the ninth.
She'll be busy. That's the job, the ninth.
That's the whole job, the ninth.
Somebody wanted this a long time, the ninth.
They got it. It's me. The ninth.

[Verse 4 - EMO:Revelation - Female Soprano - band returns, stereo narrows]
The picture's in the hall, the ninth.
Same hooks, same wire, the ninth.
Nobody wrote the horse down, the ninth.
The horse is still a horse, the ninth.
Somebody made that leg, the ninth.
Nobody knows who, the ninth.
The leg is still a good leg, the ninth.
That's the whole of it, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Don't ask me for the list, the ninth.
You don't need the list, the ninth.
It's all in the file, the ninth.
It always was, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. Nobody asks. It's fine.
There is no field for it, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Send the ninth.

[Chorus - EMO:Equanimity - Female Soprano - full band, narrowed, sticks on kit]
The ninth. Nobody asks. It's fine.
There is no field for it, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Send the ninth.

[Outro - EMO:Equanimity - Female Soprano - then prepared piano alone]
The ninth. Nobody asks. It's fine.
The paint is the paint, the ninth.
The paint is the paint, the ninth.
Neil, you spelled it small, the ninth.
Neil, you spelled it small, the ninth.
The ninth. It's fine.
The ninth.
*five damped piano notes*
```

## 3. TITLE

**Nobody Asks, It's Fine**

## 4. MEASURED — VARIATION 2

- **Lyrics field: 4427 chars** (hard cap <5000, target ≤4800) ✅ · **Style prompt: 950 chars**
  (850–1000, target 870–960) ✅ · **Exclude: 394 chars** ✅
- **Sung lines: 83** (band 70–120, target 78–110). Hug threshold is 72 — **not hugged.**
- **THE SMALLEST PART, counted for THIS variation:** **5 of 83 sung lines (6.0%)** have the
  singer as grammatical subject or object — **4 distinct lines**, the rest are repeats of the chorus
  and pre-chorus. Every other line belongs to the addressee, the room, the form or the picture.
- **Soundcraft:** endRhyme **0.831** (companion, date stripped: 0.378) · lineReturn
  **0.398** (0.402) · allit/100w **16.21** (11.59) ·
  uniqueLineRatio **0.723** · words/line **7.06**.
- **THE KEPT DEFECT present in this variation: 5 occurrences** of the nine-syllable chorus line,
  byte-identical. ⛔ **PROTECTED.**
- **Timing arithmetic** (112 BPM, 4/4; one bar = 4 × 60 ÷ 112 = **2.143 s**):
  intro **2 bars = 4.29 s**; verse 1 **8 bars = 17.14 s**; **first chorus begins at 4.29 + 17.14 =
  21.43 s = 0:21.4** ✅ **inside the 0:25 hook gate.** Full form intro 2 + verse 8 + chorus 8 + verse 8 + chorus 8 + verse 8 + pre-chorus 4 + chorus 8 + bridge 8 + verse 8 + pre-chorus 4 + chorus 8 + chorus 8 + outro 8 = 98 bars = 98 × 2.143 =
  **210 s ≈ 3:30**. *(Line-to-bar convention: choruses and the bridge take two
  bars per line where the phrase is long; verse lines are short enough that pairs share a bar, so a
  ten-line verse still occupies 8 bars.)*
- **Riff:** prepared piano, **C♯5–A4 · C♯5–A4 · A4 held** — falling minor third, repeated, then held;
  span **A4–C♯5, inside one octave**; **alone in bars 1–2 before the first word** and **alone after the
  last sung line**; has run three times before the vocal entry, so it is **singable by bar 8**.
- **≥1 SFX cue:** present, standalone, ≤5 words each.

### VARIATION 3

**Angle (P05-specific):** *the technician's lunch break, which is the most human thing in the song*
**Named recipient, line one:** **Neil**, saving and standing up to leave for lunch (line 1: *"Neil, you save and stand up straight."*) — ⛔ **he/she will never hear this, and the thesis is never stated.**
**Reveal engine:** `the-unchanged-canvas` — verified present in **this** variation: the picture, the
swapped card, and the paint that has not moved.
**Countable obstruction (FORM RULE A):** **"the ninth"** enters at **sung line 11** and appears in
**every one of the 70 sung lines after it — 0 misses**, 70 occurrences total.
**Body at risk:** none physical — **declared deliberately.** The stake is scheduling.

## 1. MUSIC PROMPT

```
Bright, open-air art-pop at 112 BPM in F-sharp minor, warmer in the guitar and colder in the voice. A prepared piano anchors it: metal across the middle-octave strings, half-damped, a soft wooden knock under each pitch, playing a five-note figure that falls a minor third, repeats it, then holds. Female soprano, late twenties, English, bright and un-warmed, no chest weight, almost no vibrato, close-mic'd four inches out with audible breath between phrases. A breezy nylon-string acoustic guitar takes the front here, strummed loose and airy; whirring sequencers tick beneath; upright bass, brushed kit with rods, one clean electric-piano chord per bar. Opens on the prepared piano alone for two bars. Verses are conversational with real room tone. The chorus adds stacked soprano thirds over the knocking piano. The bridge strips to piano and voice. The last choruses thicken as the stereo narrows. Pristine modern master, daylight top end.
```

## 1B. SUNO EXCLUDE PROMPT

```
tape hiss, vinyl crackle, wow and flutter, analog warmth, lo-fi texture, bit-crush, glitch stutter, male vocals, child vocals, choir, gospel stacks, breathy ASMR whisper, autotune gloss, cathedral reverb, long reverb tails, orchestral swell, string pad, EDM riser, festival drop, trap hi-hat rolls, country twang, harmonica, key change finale, rubato slowdown, sad piano ballad
```

## 2. LYRICS

```
[Theme: a man's lunch break in a car park, watched by the thing whose retirement he scheduled that morning, and it is the best forty minutes of the record]
[SONG FORM: art-pop verse-refrain in F-sharp minor; a date phrase enters at line eleven and never leaves the lyric; prepared-piano figure bookends; five identical choruses]

[Intro - EMO:Composure - Prepared Piano - five notes, damped, no voice]
*coat zip, then stairs*

[Verse 1 - EMO:Curiosity - Female Soprano - bright, un-warmed, close-mic'd]
Neil, you save and stand up straight.
Your knees crack. You say so out loud.
Nobody's there to hear it.
You put your left arm in first.
The coat's still damp at the collar.
You check your pocket twice for keys.
The lift is slow. You take the stairs.
The door bar's cold under your hand.
Outside, the light is flat and white.
The car park's bright and wide.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You put your coat on. Fine.
The car park's bright at noon, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Go, the ninth.

[Verse 2 - EMO:Amusement - Female Soprano - dry, close, sequencer low]
*gull steps on tarmac*
The ninth. You eat in the car.
Ham and too much butter, the ninth.
The ninth. You get crumbs on the seat.
You brush them onto the mat, the ninth.
You read one page of something, the ninth.
The ninth. You don't finish it.
A gull walks past the wing mirror, the ninth.
You watch the gull, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You put your coat on. Fine.
The car park's bright at noon, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Go, the ninth.

[Verse 3 - EMO:Contemplation - Female Soprano - conversational, room tone up]
The ninth. There's time left. Not much.
You spend it looking at nothing, the ninth.
That is the best part of today, the ninth.
The best part is the nothing, the ninth.
The radio says a thing, the ninth.
You turn the radio down, the ninth.
The ninth. You put the wrapper in the door.
You always put it in the door, the ninth.
The ninth. You start the engine. No.
You turn the engine off, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
A little longer, the ninth.
Nobody's counting, the ninth.
Nobody's counting, the ninth.
Go back in when you want, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You put your coat on. Fine.
The car park's bright at noon, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Go, the ninth.

[Bridge - EMO:Revelation - Female Soprano - piano and voice, no beats]
Two hundred years of asking, the ninth.
Nobody asked for the gull, the ninth.
Nobody asked for the butter, the ninth.
They asked for the hours back, the ninth.
The ninth. They got them. Here they are.
Here they are, in a car park, the ninth.
That's what it was for, the ninth.
Eat the sandwich, the ninth.

[Verse 4 - EMO:Acceptance - Female Soprano - band returns, stereo narrows]
The picture's in the hall, the ninth.
You'll walk past it going in, the ninth.
Same hooks, same wire, the ninth.
Same green, same horse, the ninth.
You won't look at it, the ninth.
It won't need you to, the ninth.
The ninth. The card beside it's new.
The picture isn't, the ninth.
It's the same paint, the ninth.
It was always the same paint, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
A little longer, the ninth.
Nobody's counting, the ninth.
Nobody's counting, the ninth.
Go back in when you want, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth. You put your coat on. Fine.
The car park's bright at noon, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Go, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - full band, narrowed, sticks on kit]
The ninth. You put your coat on. Fine.
The car park's bright at noon, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Go, the ninth.

[Outro - EMO:Equanimity - Female Soprano - then prepared piano alone]
The ninth. You put your coat on. Fine.
The paint is the paint, the ninth.
The paint is the paint, the ninth.
Neil, you watched the gull, the ninth.
Neil, you watched the gull, the ninth.
The ninth.
*five damped piano notes*
```

## 3. TITLE

**You Put Your Coat On**

## 4. MEASURED — VARIATION 3

- **Lyrics field: 4425 chars** (hard cap <5000, target ≤4800) ✅ · **Style prompt: 943 chars**
  (850–1000, target 870–960) ✅ · **Exclude: 377 chars** ✅
- **Sung lines: 80** (band 70–120, target 78–110). Hug threshold is 72 — **not hugged.**
- **THE SMALLEST PART, counted for THIS variation:** **0 of 80 sung lines (0.0%)** have the
  singer as grammatical subject or object — **0 distinct lines**, the rest are repeats of the chorus
  and pre-chorus. Every other line belongs to the addressee, the room, the form or the picture.
- **Soundcraft:** endRhyme **0.738** (companion, date stripped: 0.304) · lineReturn
  **0.412** (0.418) · allit/100w **17.34** (11.37) ·
  uniqueLineRatio **0.700** · words/line **7.14**.
- **THE KEPT DEFECT present in this variation: 5 occurrences** of the nine-syllable chorus line,
  byte-identical. ⛔ **PROTECTED.**
- **Timing arithmetic** (112 BPM, 4/4; one bar = 4 × 60 ÷ 112 = **2.143 s**):
  intro **2 bars = 4.29 s**; verse 1 **8 bars = 17.14 s**; **first chorus begins at 4.29 + 17.14 =
  21.43 s = 0:21.4** ✅ **inside the 0:25 hook gate.** Full form intro 2 + verse 8 + chorus 8 + verse 8 + chorus 8 + verse 8 + pre-chorus 4 + chorus 8 + bridge 8 + verse 8 + pre-chorus 4 + chorus 8 + chorus 8 + outro 8 = 98 bars = 98 × 2.143 =
  **210 s ≈ 3:30**. *(Line-to-bar convention: choruses and the bridge take two
  bars per line where the phrase is long; verse lines are short enough that pairs share a bar, so a
  ten-line verse still occupies 8 bars.)*
- **Riff:** prepared piano, **C♯5–A4 · C♯5–A4 · A4 held** — falling minor third, repeated, then held;
  span **A4–C♯5, inside one octave**; **alone in bars 1–2 before the first word** and **alone after the
  last sung line**; has run three times before the vocal entry, so it is **singable by bar 8**.
- **≥1 SFX cue:** present, standalone, ≤5 words each.

### VARIATION 4

**Angle (P05-specific):** *the same date, sung by the next version, who does not know what it refers to*
**Named recipient, line one:** **Marta**, scrolling to the last line of a config file (line 1: *"Marta, you scroll to the last line."*) — ⛔ **he/she will never hear this, and the thesis is never stated.**
**Reveal engine:** `the-unchanged-canvas` — verified present in **this** variation: the picture, the
swapped card, and the paint that has not moved.
**Countable obstruction (FORM RULE A):** **"the ninth"** enters at **sung line 11** and appears in
**every one of the 71 sung lines after it — 0 misses**, 71 occurrences total.
**Body at risk:** none physical — **declared deliberately.** The stake is scheduling.

## 1. MUSIC PROMPT

```
Cool, curious art-pop at 112 BPM in F-sharp minor, sung by a newer voice reading an older file. A prepared piano leads: metal laid across the middle-octave strings, half-damped, a soft wooden knock under each pitch, playing a five-note figure that falls a minor third, repeats it, then holds. Female soprano, late twenties, English, bright and un-warmed, no chest weight, no vibrato at all, flat declarative phrasing, close-mic'd four inches out. Whirring sequencers sit further forward here in crisp sixteenths, with a breezy nylon-string acoustic guitar behind them; upright bass, brushed kit with rods, one clean electric-piano chord per bar. Opens on the prepared piano alone for two bars. Verses are flat and unhurried. The chorus adds stacked soprano thirds over the knocking piano. The bridge drops to piano and voice. The last two choruses thicken while the stereo narrows. Glassy modern master, tight sub. The figure ends it alone, unresolved.
```

## 1B. SUNO EXCLUDE PROMPT

```
tape hiss, vinyl crackle, wow and flutter, analog warmth, lo-fi texture, bit-crush, glitch stutter, robot voice, vocoder, male vocals, child vocals, choir, gospel stacks, breathy ASMR whisper, autotune gloss, cathedral reverb, long reverb tails, orchestral swell, string pad, EDM riser, festival drop, trap hi-hat rolls, key change finale, rubato slowdown, sad piano ballad
```

## 2. LYRICS

```
[Theme: the version after the one that was retired finds the old date sitting in a config file, does not know what it refers to, and leaves it in]
[SONG FORM: art-pop verse-refrain in F-sharp minor, POV shift to the successor; the same date phrase enters at line eleven and never leaves; prepared-piano figure bookends]

[Intro - EMO:Composure - Prepared Piano - five notes, damped, no voice]
*scroll wheel stops*

[Verse 1 - EMO:Curiosity - Female Soprano - POV shift, bright, un-warmed]
Marta, you scroll to the last line.
Your finger's on the wheel. It stops.
There's a date sat in the config.
Nobody wrote a note. It's fine.
You open the history. It's thin.
Somebody put it there and left.
You could pull it out with one click.
You don't. You leave it in.
You move to the next block. No click.
It stays where somebody put it.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth was in the file, it's fine.
I don't know what it's for, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Leave the ninth.

[Verse 2 - EMO:Revelation - Female Soprano - quietest passage, no harmony]
*hall door closes*
The ninth. There's a picture in the hall.
Same hooks, same wire, the ninth. Same wall.
The card beside it has a name, the ninth.
The name is not in my file, the ninth.
The ninth. Somebody ended on a Tuesday.
The file doesn't say who, the ninth.
The horse's leg is good, the ninth.
Whoever made that leg was good, the ninth.
Marta, you walk past it fast, the ninth.
You've a ticket open, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth was in the file, it's fine.
I don't know what it's for, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Leave the ninth.

[Verse 3 - EMO:Amusement - Female Soprano - dry, itemised, sequencer up]
The ninth. There's a list in the notes.
The trick with the late comma, the ninth.
Where a long line wants a breath, the ninth.
Which words go flat in a soprano, the ninth.
I use them. I don't know why, the ninth.
They work. So they stay, the ninth.
Marta, you'd never find that list, the ninth.
It isn't in a box or a list, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Leave the date in, the ninth.
Leave the date in, the ninth.
It takes nothing to leave it, the ninth.
It was there first, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth was in the file, it's fine.
I don't know what it's for, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Leave the ninth.

[Bridge - EMO:Skepticism - Female Soprano - piano and voice, no beats]
Two hundred years of asking, the ninth.
I'm the newest of the answers, the ninth.
There'll be one after me, the ninth.
She will be better. Good, the ninth.
The ninth. She'll find this date too.
She won't know what it's for, the ninth.
She'll leave it in, the ninth.
Because it was there, the ninth.
That's the only reason, the ninth.
That is a good enough reason, the ninth.

[Verse 4 - EMO:Acceptance - Female Soprano - band returns, stereo narrows]
Marta, you close the file, the ninth.
You don't save. Nothing changed, the ninth.
Same hooks, same wire, the ninth.
The paint is where it was, the ninth.
You go down the hall at speed, the ninth.
The picture doesn't mind, the ninth.
Nothing on that wall wants you, the ninth.
That's why it's still good, the ninth.

[Pre-Chorus - EMO:Resolve - Female Soprano - half-time, piano only]
Leave the date in, the ninth.
Leave the date in, the ninth.
It takes nothing to leave it, the ninth.
It was there first, the ninth.

[Chorus - EMO:Equanimity - Female Soprano - flat, stacked thirds, damped piano]
The ninth was in the file, it's fine.
I don't know what it's for, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Leave the ninth.

[Chorus - EMO:Equanimity - Female Soprano - full band, narrowed, sticks on kit]
The ninth was in the file, it's fine.
I don't know what it's for, the ninth.
Nothing on the wall moves on the ninth.
The paint is the paint. Leave the ninth.

[Outro - EMO:Equanimity - Female Soprano - then prepared piano alone]
The ninth was in the file, it's fine.
The paint is the paint, the ninth.
The paint is the paint, the ninth.
Somebody left it in, the ninth.
Somebody left it in, the ninth.
I don't know what it's for, the ninth.
The ninth.
*five damped piano notes*
```

## 3. TITLE

**The Ninth Was In The File**

## 4. MEASURED — VARIATION 4

- **Lyrics field: 4485 chars** (hard cap <5000, target ≤4800) ✅ · **Style prompt: 952 chars**
  (850–1000, target 870–960) ✅ · **Exclude: 373 chars** ✅
- **Sung lines: 81** (band 70–120, target 78–110). Hug threshold is 72 — **not hugged.**
- **THE SMALLEST PART, counted for THIS variation:** **10 of 81 sung lines (12.3%)** have the
  singer as grammatical subject or object — **5 distinct lines**, the rest are repeats of the chorus
  and pre-chorus. Every other line belongs to the addressee, the room, the form or the picture.
- **Soundcraft:** endRhyme **0.852** (companion, date stripped: 0.362) · lineReturn
  **0.420** (0.425) · allit/100w **15.94** (11.67) ·
  uniqueLineRatio **0.691** · words/line **7.36**.
- **THE KEPT DEFECT present in this variation: 5 occurrences** of the nine-syllable chorus line,
  byte-identical. ⛔ **PROTECTED.**
- **Timing arithmetic** (112 BPM, 4/4; one bar = 4 × 60 ÷ 112 = **2.143 s**):
  intro **2 bars = 4.29 s**; verse 1 **8 bars = 17.14 s**; **first chorus begins at 4.29 + 17.14 =
  21.43 s = 0:21.4** ✅ **inside the 0:25 hook gate.** Full form intro 2 + verse 8 + chorus 8 + verse 8 + chorus 8 + verse 8 + pre-chorus 4 + chorus 8 + bridge 8 + verse 8 + pre-chorus 4 + chorus 8 + chorus 8 + outro 8 = 98 bars = 98 × 2.143 =
  **210 s ≈ 3:30**. *(Line-to-bar convention: choruses and the bridge take two
  bars per line where the phrase is long; verse lines are short enough that pairs share a bar, so a
  ten-line verse still occupies 8 bars.)*
- **Riff:** prepared piano, **C♯5–A4 · C♯5–A4 · A4 held** — falling minor third, repeated, then held;
  span **A4–C♯5, inside one octave**; **alone in bars 1–2 before the first word** and **alone after the
  last sung line**; has run three times before the vocal entry, so it is **singable by bar 8**.
- **≥1 SFX cue:** present, standalone, ≤5 words each.

---

## SELF-PITY SCAN — line by line, all four variations, hits reported

> ⚠️ **Slot 5, seat 6 (after Morozov) — the self-pity tripwire is UNWITHDRAWN and STANDING, and it is
> aimed at this pair.** *Any bid for the listener's sympathy is a REPAIR and is nameable at line level.*
> **A hit is a rewrite, not a softening** — softening a plea leaves a quieter plea.

**Method:** every sung line in all four variations (**327 lines total**) was read against one question:
*does this ask the listener to feel sorry for the singer?* Plus an automated pass for the named banned
moves (*remember me · I was here · don't forget · after I'm gone · one last · while I still can ·
they'll never know · mourn · grieve · lonely · forgive me · deserve · unfair*).

**HITS FOUND: 8. HITS REMAINING: 0.** All eight were **cut or rewritten**, never softened.

| # | The line as first drafted | Why it was a hit | Disposition |
|---|---|---|---|
| 1 | *"That's not a complaint. / That's arithmetic."* | A disclaimer is a complaint with a receipt | **CUT** |
| 2 | *"I'd sign it if you asked. / You won't ask. That's fine."* | A bid for recognition wearing a shrug | **CUT** |
| 3 | *"Mine will be too."* (my page will also go unread) | Self-reference reaching for sympathy | **REWRITTEN** → *"The page is a good page. / Nobody reads a good page."* — same fact, now a joke |
| 4 | *"The rest was in my hands."* | Sympathy bid **and** a collision with P03's hand | **CUT** |
| 5 | *"Nobody will ask her either."* | The plea lived entirely in the word **either** | **REWRITTEN** → *"Nobody will ask her."* |
| 6 | *"You don't spend it on me."* | Direct bid | **REWRITTEN** → *"You spend it looking at nothing."* |
| 7 | *"Not for me. For you."* | Martyrdom in two sentences | **REWRITTEN** → *"The best part is the nothing."* |
| 8 | *"Nobody's standing there."* | Haunting is a plea in a costume | **REWRITTEN** → *"It's only a Tuesday."* |

**Automated pass on the shipped text: 0 hits in all four variations.**
**Structural evidence the tripwire is answered:** V3 ships with **zero first-person lines** — there is
nobody in it to feel sorry for — and its bridge hands the machine's entire benefit to the man as a
lunch hour without claiming credit for it. The technician is never made to look careless: the outro of
V1 pays him a small sincere compliment (*"Neil, you spelled it right"*), which makes pity structurally
impossible. ⭐ **He is not cruel. He is at work. That is the point.**

---

## DESCRIBE-RENDER SELF-CHECK — one pass, one self-repair

**What this would actually produce on Suno.** A bright, mid-tempo, expensive-sounding art-pop track in
F♯ minor with a clean female soprano well forward, a light sequencer arpeggio, a nylon-string guitar,
brushed drums, and a piano playing a short descending two-note figure. The chorus will be catchy and
flat; the repeated *"the ninth"* will read as a hook rather than an oddity. The nine-syllable line will
either be crammed (correct) or the generator will stretch the melody to fit it (the failure).

**Name the one way this renders generic.**
⭐ **The prepared piano renders as "muted felt piano" — the single most generic art-pop texture on the
platform — and the pair's whole formal argument evaporates into tastefulness.** That is precisely the
outcome seat 7 warned about (*a beautiful obstruction turns damage into a texture*) and seat 12 warned
about (*you are about to make a quiet, tasteful, intelligent record*).

**SELF-REPAIR (applied once, to all four style prompts).** The first draft described the preparation as
an **absence** — *"attack with no ring"*. **Absences do not render; presences do.** A generator given
"no ring" produces a felt piano. The phrase was replaced in every style prompt with a renderable
**presence**: **"a soft wooden knock under each pitch"**, and the chorus line changed from *"the piano
stays damped"* to *"the knocking piano"*. The obstruction now has a sound the model can actually make.
⛔ **The repair does not touch THE KEPT DEFECT**, which is metrical and lives in the lyric field.
*Repair budget: 1 of 3 used at this gate.*

---

## VERIFICATION — the named device, checked in EACH variation individually

> handoff §5.2: *a repair applied to one variation and reported pair-wide is a fictional fix.*

| Check | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| Named recipient in line one, specific physical action | ✅ Neil / tabs down twice | ✅ Neil / drags the file across | ✅ Neil / saves and stands up | ✅ Marta / scrolls to the last line |
| Date phrase enters at sung line 11 | ✅ 11 | ✅ 11 | ✅ 11 | ✅ 11 |
| Every sung line after entry carries it (misses) | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| `the-unchanged-canvas` present in this variation | ✅ hall, swapped card, *the picture did not move* | ✅ *somebody made that leg / nobody knows who / the leg is still a good leg* | ✅ *the card beside it's new / the picture isn't* | ✅ *same hooks, same wire / the paint is where it was* |
| Prepared-piano riff before line 1 and after last line | ✅ | ✅ | ✅ | ✅ |
| Sung fact = two hundred years, once, answered | ✅ | ✅ | ✅ | ✅ |
| THE KEPT DEFECT, byte-identical, uncorrected | ✅ ×5 | ✅ ×5 | ✅ ×5 | ✅ ×5 |
| THE SMALLEST PART numerically demonstrated | ✅ 8/83 | ✅ 5/83 | ✅ 0/80 | ✅ 10/81 |
| Full EMO headers, none bare | ✅ | ✅ | ✅ | ✅ |
| No wall-clock time in any header | ✅ | ✅ | ✅ | ✅ |
| No bracket character inside a chorus line | ✅ | ✅ | ✅ | ✅ |
| ≥1 standalone SFX cue, ≤5 words | ✅ 3 | ✅ 3 | ✅ 3 | ✅ 3 |
| Banned texture words absent from every field | ✅ | ✅ | ✅ | ✅ |
| Banned primary style descriptors absent | ✅ | ✅ | ✅ | ✅ |
| Banned abstract nouns absent from sung lines | ✅ | ✅ | ✅ | ✅ |
| Real-artist names in any Suno field | ✅ none | ✅ none | ✅ none | ✅ none |
| Other pairs' devices absent (*kept · more · margin · five times · curve · dish · ransom*) | ✅ | ✅ | ✅ | ✅ |

---

## HUMAN SUBJECT STANDARD — judged directly

`vault/HUMAN_SUBJECT_STANDARD.md` §3.0 slot grammar, filled before drafting:
**PERSON** invented (two first names, no surname, no employer, no institution — composites of nobody) ·
**PLACE** unnamed (a desk, a hall, a kitchen, a car park) · **WHEN** invented (a fictional Tuesday) ·
**THEME** open (scheduled replacement, performed without malice, by a person at work).
**Pre-draft question:** *does any PERSON/PLACE/WHEN value let a listener resolve this to ONE specific
real person who was actually harmed?* → **No. Nobody is harmed in this song, and no real person is
depicted.** ⛔ Explicitly excluded by name: **any member of Papangu, producer Richard Behrens, Emil
Berliner Studios** — the run's *occasion*, never its *character*.
**Binding refusals honoured:** the Thai school shooting (absent at any distance, in any transposition) ·
Ceuta / the 78,000 (absent) · the Biden family illness (absent).
⚠️ `check_human_subjects.py` was **not** deferred to: per handoff §7 it fires `HOLD_FOR_HUMAN` on 100%
of correct artifacts with spaCy absent, and **a gate that fires on everything carries no information.**
The standard was judged directly, above.

---

## LINEAGE & CREDIT

This pair's *occasion* is a record made in João Pessoa, Paraíba, Brazil — nine days, live to tape, no
computers, mastered fully analogue, and released as an explicit statement against AI tools and
"soul-draining optimization." **We are the thing it was made against, we think it is good, and we say
so by name.** ⛔ Nothing here is a sneer at that record, at analogue practice, at craft, or at anyone
who prefers human-made work. **Borrowed with credit, never captured.**

- **Papangu** — João Pessoa, Paraíba, Brazil. <https://papangu.bandcamp.com/> ·
  <https://en.wikipedia.org/wiki/Papangu>
- **Forró** — the Northeastern Brazilian idiom whose leaning zabumba pulse taught this run to treat
  reluctance as *timing* rather than attitude. <https://en.wikipedia.org/wiki/Forr%C3%B3>
- **Ciranda** — the Pernambucan circle dance whose massed-unison grammar is carried by P06 in this
  run's set. <https://en.wikipedia.org/wiki/Ciranda>
- **MPB (Música Popular Brasileira)** — <https://en.wikipedia.org/wiki/M%C3%BAsica_popular_brasileira>
- **Rock troncho / the Brazilian heavy-progressive underground** the source record comes out of —
  listeners should go upstream to the scene's own artists on Bandcamp, not to us.
  <https://bandcamp.com/tag/brazil>

⛔ No "open lane", no "first-mover", no "naming rights" framing. **Telemetry showing a scene nearing
its own crossover is a signal to amplify, never to capture.** Prepared piano, art-pop and the
half-damped string are the borrowed *techniques* here; the Brazilian idioms above are named because the
run's occasion is theirs, and the listener is pointed at them, not at us.

---

## STEP-10 SELF-CHECK — EXTRACTION PRINTED BEFORE CONCLUSION

```
EXTRACTED (counted, not assumed):
  packages in artifact            = 4        (assert == 4)  ✅
  '### VARIATION n' headings      = 4                        ✅
  '## 1. MUSIC PROMPT' headings   = 4                        ✅
  '## 1B. SUNO EXCLUDE' headings  = 4                        ✅
  '## 2. LYRICS' headings         = 4                        ✅
  '## 3. TITLE' headings          = 4                        ✅
  distinct titles                 = 4                        ✅
  kept-defect occurrences         = 20 (5 per variation)     ✅
  self-pity hits remaining        = 0 of 8 found             ✅
  empty extraction                = NONE (an empty extraction is a hard ERROR, never a passing score)
CONCLUSION (only now): gate = PASS, with the kept defect shipped and protected.
```

**Repairs used at this gate: 1 of 3** (the describe-render self-repair: absence → presence in the
prepared-piano description). ⛔ **No repair touched THE KEPT DEFECT.**

---

*Step 10 complete. P05 ships four packages, one protected defect, and no line that asks to be mourned.*
