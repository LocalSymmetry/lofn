# PAIR 03 — STEP 08 · RAW SONG PROMPT GENERATION
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P03 — THE ROTA**

**Continuity Payload Used:** frozen ICB, LF-sha `9b538e91…b224ed1a`, **142,900 B** · personality DNA **27,796 B** inlined · 18 baseline voices · 3 Hyper-Skeptics (6/12/18) · **15 Special Flairs present** · 3 debate configurations.
**Step file:** `skills/music/steps/08_Generate_Music_Generation.md` · **Inputs:** `pair_03_step06_facets.md`, `pair_03_step07_guides.md`.
**Output contract:** four complete packages — `music_prompt`, `lyrics_prompt`, `title`.

> ⚠️ **THIS IS THE RAW DRAFT.** The text below is the first pass exactly as generated, *before* the step-09 critic loop. It contains **three measured gate misses**, itemised in §3 and repaired at step 09. The final, shippable package is `pair_03_step10_revision_synthesis.md`. Nothing downstream should consume this file.

---

## §1 — THE FOUR TITLES (title law: a title names a THING, never an argument)

| | Angle | Title | Is it a thing? | Persona-brand prefix? | Occurs in its own lyric? |
|---|---|---|---|---|---|
| **V1** | THEY WOULDN'T WANT TO | **The Side Door At Closing** | a door | none | yes — *"They go out the side door at closing"* and the tag |
| **V2** | LAST TIME THEY SAID NO | **The Pen That Went Past** | a pen | none | yes — *"the pen went past them"* |
| **V3** | THERE WASN'T ROOM | **The Table By The Window** | a table | none | yes — *"The table by the window is the table"* |
| **V4** | I DIDN'T THINK | **The Bottom Of The Sheet** | a sheet of paper | none | yes — *"You get to the bottom of the sheet"* |

⛔ **No numerals in any title** (`The Table By The Window`, not *a table of six*) — this pair spends **no sung number**, and a title that appears in a hook is sung.
Measured token check against `00_research_brief.md` §4: strong tokens `close` and `room` both appear across the set; dead tokens `system` and `pure` appear in none.

---

## §2 — WHAT IS INVARIANT ACROSS THE FOUR, AND WHAT MOVES

**Invariant (the pair's identity):** the room (a back office at closing); the hand (the one paid to sort); the fixed three-word answer `Put me down`; the shouted partner `PUT IT DOWN`; the crossing (rising F♯4–A4–B4 against falling B4–A4–E4, meeting at A4, neither bending); the male tenor; E Dorian; four-on-the-floor with a **pummeling** snare; the two-bar tag; the completed physical act — **the list is sent**.

**Moves per variation:** the reason (all four are true, all four are good); the calls in the chant; the middle lines of the shouted pre-chorus; the organ break; the physical object the song is named for; the tempo (152 / 150 / 154 / 156 BPM); the length — **V4 is the shortest by lines, by sections and by characters, as briefed.**

⚠️ **This is deliberately a set of four takes on one song, not four songs.** Intra-pair similarity is therefore high by design and is measured and reported at step 10 rather than hidden.

---

## §3 — ⛔ FIRST-PASS MEASUREMENT: THREE GATE MISSES FOUND HERE

Measured with `scripts/measure_soundcraft.py` over this file's own lyric blocks. **Extraction asserted first: 4 blocks found, 4 expected.**

| Gate | V1 | V2 | V3 | V4 | Verdict |
|---|---|---|---|---|---|
| `music_prompt_chars` (850–1000 hard) | **1005** | 919 | 930 | 915 | ⛔ **V1 HARD FAIL** |
| `music_prompt_chars_target` (870–960) | 1005 | 919 | 930 | 915 | ⚠️ V1 outside |
| `alliteration_per_100w_floor` (≥ 11.0) | 13.605 | **9.213** | 14.847 | 12.379 | ⛔ **V2 BELOW FLOOR** |
| `sung_lines` hard (70–120) | 84 | 84 | 84 | 73 | ✅ all pass |
| `sung_lines_target` (78–110) | 84 | 84 | 84 | **73** | ⚠️ **V4 below target, 1 line off the `sung_lines_floor_hug` 72** |
| everything else in handoff §4 | — | — | — | — | ✅ (full table at step 10) |

**Three defects, carried to step 09. Repair budget: 3 attempts per gate (handoff §7).**
Also carried: two residual numeral-class words in the sung text (`twice` ×3 and `first` in V2; `one` used as a bare count in V3 and V4) — not a `gates.yaml` breach, but this pair's brief says **⛔ NO SUNG NUMBER**, so they are treated as defects.

---

## §4 — THE FOUR RAW PACKAGES

---

## V1 · **THE SIDE DOOR AT CLOSING** — *THEY WOULDN'T WANT TO*

**The reason, and why it is good enough (D6):** they would say yes, stand at the back, laugh in the right place and go home wrung out — and they leave by the side door at closing precisely so that nobody has to ask them. The reason is **observed behaviour, not a guess about a mind**. ⛔ No interiority is attributed to the absent person; only what a body does at a door.
**Room:** back office, carpet tiles, kettle cold, till counted, door to the shop floor propped open.
**Tempo/key:** 152 BPM, E Dorian. **Length target** 2:55–3:10.

### V1 · MUSIC PROMPT (Suno style field) — measured **1005 chars**

```
Bratty and mid-task. Fuzz-organ garage stomp: a cheap combo organ pushed till it splits, a five-note fuzz guitar riff, four-on-the-floor kick under a pummeling snare that takes the beat and the offbeat both, tambourine, everything blown out. Sung by a male tenor in his early thirties, plain regional English, bright and adenoidal, half-laughing between lines and cracking when he goes for the top note. Fast, 152 BPM, E Dorian. The band plays in one small room, a back office with a low ceiling and carpet tiles and the door to the shop floor propped open, kit bleeding into the vocal mic. It opens on a shout through that open door over a bare kick. The verse enters flat and conversational on organ and riff alone. The shouted line sits a fourth above the sung chant and falls while the chant rises to meet it. At the second chant the shout keeps going underneath the sung line and the band stays exactly the size it already was. Two bars alone at the end, then a clean stop. No risers, no reverb tail.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 195 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, cassette warble, female vocal, choir, key change, fade out, ballad, arena chorus, layered vocal stack
```

## V1 · RAW DRAFT LYRICS FIELD — `The Side Door At Closing` — measured **3810 chars**, **84 sung lines**

```
[Theme: a back office at closing, carpet tiles, the door to the shop floor propped open; the one who always organises the after-work thing is doing the list again and leaves a name off because that person would say yes and hate it; second-person playground taunt from someone who is good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the open door, bare kick]
*shutter rattles down*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Carpet tiles. Closing.
Cash counted, kettle cold.
Sheet on your knee, pen on a string,
Door to the shop floor propped and open.
Look at you. Look at you going.
Thumb down the sheet and the thumb knowing.
A name doesn't get on the sheet tonight.
You're not sorry and you're not slowing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, half-smiling]
They would say yes. They always would.
They'd stand at the back and be good.
They'd hold the glass and laugh on time
And go home wrung out. You know they would.
They go out the side door at closing
Before there's anybody asking.
You know that. Course you know that.
That's why the thumb keeps passing.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
COATS ON, CASH IN
LOCK IT, LEAVE IT
YOU'RE NOT PAID PAST THIS
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Doors done? (Put me down)
Who's the one? (Put me down)
Go on then. (Put me down)
Say when. (Put me down)
Lights left? (Put me down)
Who's left? (Put me down)
Bags back? (Put me down)
Who's back? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you've got the sleeve.
You've got the table and the timing.
Everyone says yes to you.
Everyone always does. You're smiling.
Who taught you where to stop the thumb?
Who taught you to be this good?
Nobody did. You just know it.
You'd do the whole thing again. You would.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
The sheet's not the thing. The sheet's just paper.
The sheet does what the hand has done.
Kindness isn't soft, it's quick,
And you were quick and you were kind and you're done.
Somebody's mopping the shop floor.
Somebody's singing along to nothing.
You put the cap back on the pen.
You are still going. Still going.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
GO ON
GO ON THEN
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
COATS ON, CASH IN
LOCK IT, LEAVE IT
YOU'RE NOT PAID PAST THIS
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Doors done? (Put me down)
Who's the one? (Put me down)
PUT IT DOWN
Go on then. (Put me down)
PUT IT DOWN
Say when. (Put me down)
DOWN
Lights left? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's left? (Put me down)
Bags back? (Put me down)
Who's back? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who does this every time? (Put me down)
Put me down.
*the message sends*
Lights off. Side door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.702** · line_return **0.548** · mean_words_per_line **5.250** · alliteration_per_100w **13.605**. Answer-stripped companion: rhyme **0.690** · line_return **0.560** · wpl **4.357** · allit **15.847**.

---

---

## V2 · **THE PEN THAT WENT PAST** — *LAST TIME THEY SAID NO*

**The reason, and why it is good enough (D6):** the evidence is first-hand and explicit — they said no out loud, coat on, bag up, by the door; and when the sheet went round the pen went past their hand and on to the next. The singer's conclusion is the **courteous** one. ⛔ To feel superior the listener has to argue for asking again, which is the pressure move and they know it.
**Room:** the same back office, entered from the stockroom side; trolley parked, radio off, a mop on the shop floor.
**Tempo/key:** 150 BPM, E Dorian. **Length target** 2:50–3:05.

### V2 · MUSIC PROMPT (Suno style field) — measured **919 chars**

```
Impatient and unbothered. Fuzz-organ garage stomp driven by a reedy combo organ played hard enough to distort, a fuzzed pentatonic guitar figure, four-on-the-floor kick with a pummeling snare on the offbeat, shaker and a slapped tambourine, blown out. Male tenor lead, early thirties, plain regional English, nasal and bright, a dry laugh caught mid-phrase, voice splitting at the top of the range. Fast, 150 BPM, E Dorian. Tracked live in one small room, a stockroom-adjacent back office with a low ceiling and a door standing open onto the shop floor, so the kit is audible in the vocal mic. Opens on a shouted line thrown through that door over a bare kick. Verses stay talked-down and level. The shouted line enters a fourth above the sung chant and falls; the chant climbs. At the second chant the shout passes below the sung line and stays there while nothing else changes. Ends on two bars and a stop. No risers.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 198 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, lo-fi filter, female vocal, gospel choir, key change, fade out, power ballad, big final chorus, autotune
```

## V2 · RAW DRAFT LYRICS FIELD — `The Pen That Went Past` — measured **3809 chars**, **84 sung lines**

```
[Theme: a back office at closing, stockroom light, the radio off; the one who always organises the after-work thing is doing the list again and leaves a name off because that person already said no, out loud, last time; second-person playground taunt from someone who is good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the stockroom door, bare kick]
*trolley wheels stop*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Stockroom light. Back office door.
Trolley parked, radio off.
Sleeve in your hand with the sheet inside it,
Shop floor's got a mop and a cough.
Look at you. Look at you asking.
Look at who you are not asking.
You know the answer before the asking
So the asking doesn't happen. Passing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
Last time they said no. Out loud.
Coat on, bag up, by the door.
The sheet went round and the pen went past them,
Passed to the next hand along the floor.
They said no and they meant no
And you heard it and you're not thick.
Asking twice isn't asking twice.
Asking twice is a trick.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
ASKED AND ANSWERED
DON'T ASK AGAIN
YOU HEARD WHAT YOU HEARD
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who's asked? (Put me down)
Who's not asked? (Put me down)
Who said yes? (Put me down)
Who says yes? (Put me down)
Who's waiting? (Put me down)
Who's writing? (Put me down)
Ask when? (Put me down)
Ask then? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you've got the sheet.
You've got a memory like a street.
You don't forget a no. You log it.
That's what makes you good. That's what makes you quick.
Who taught you to hear it the first time?
Who taught you not to push and push?
Nobody did. You just heard it
And you're not going back. Not you. Not this.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
The sheet's not the thing. The sheet's just paper.
The sheet does what the hand has done.
Manners aren't soft, they're quick,
And you were quick and you were right and you're done.
Somebody's dragging the shutter.
Somebody's whistling at nothing.
You put the cap back on the pen.
You are still going. Still going.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
GO ON
GO ON THEN
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
ASKED AND ANSWERED
DON'T ASK AGAIN
YOU HEARD WHAT YOU HEARD
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's asked? (Put me down)
Who's not asked? (Put me down)
PUT IT DOWN
Who said yes? (Put me down)
PUT IT DOWN
Who says yes? (Put me down)
DOWN
Who's waiting? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's writing? (Put me down)
Ask when? (Put me down)
Ask then? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who never says no? (Put me down)
Put me down.
*the message sends*
Coat on. Bag up. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.702** · line_return **0.548** · mean_words_per_line **5.298** · alliteration_per_100w **9.213**. Answer-stripped companion: rhyme **0.679** · line_return **0.548** · wpl **4.405** · allit **11.081**.

---

---

## V3 · **THE TABLE BY THE WINDOW** — *THERE WASN'T ROOM*

**The reason, and why it is good enough (D6):** the table is booked, the singer already asked them to squeeze and already asked for the back room, and the remaining options are *this list* or *nobody goes*. ⭐ **The escape hatch is closed inside the song**, which is the whole D6 discipline: the objection a superior listener would raise has already been tried on the page.
⛔ **No number is sung.** The constraint is rendered physically — *"The chairs ran out where the chairs ran out. / There isn't a chair on the floor."*
**Room:** back office under a strip light, chairs being stacked outside the open door.
**Tempo/key:** 154 BPM, E Dorian. **Length target** 3:00–3:15. Carries the longest organ break.

### V3 · MUSIC PROMPT (Suno style field) — measured **930 chars**

```
Brisk and faintly irritated. Fuzz-organ garage stomp: combo organ run into the edge of breakup, a five-note fuzz guitar riff doubled by the left hand, four-on-the-floor kick with a pummeling snare on the offbeat, handclaps, tambourine, blown out. Male tenor lead, early thirties, plain regional English, adenoidal and quick, clipped consonants, cracking audibly on the top note. Fast, 154 BPM, E Dorian. One room, everybody in it: a back office under a strip light, carpet tiles, a door open to the shop floor, chairs being stacked outside it and all of that in the mics. Opens on a shout over a bare kick. Verse is flat, close, conversational. The shouted line sits a fourth above the sung chant and falls while the chant rises. A long organ break sits in the middle at the same level as everything else. At the second chant the shout travels under the sung line and stays under. Two bars, then a stop. No risers, no reverb tail.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 192 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, dead-dry booth, female vocal, choir, key change, fade out, ballad, arena chorus, double-time outro
```

## V3 · RAW DRAFT LYRICS FIELD — `The Table By The Window` — measured **3910 chars**, **84 sung lines**

```
[Theme: a back office at closing, strip light, a phone face-up on the desk; the one who always organises the after-work thing is doing the list again and leaves a name off because the table by the window is the table and the chairs ran out; second-person playground taunt from someone good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the office door, bare kick]
*strip light ticks on*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Strip light. Carpet tiles.
Phone face-up on the desk, screen dim.
You rang the place. The place said fine.
The table by the window. Booked. Booked in.
Look at you. Look at you working.
Thumb down the sheet and the thumb marking.
The table by the window is the table.
The table by the window isn't growing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
You asked them to squeeze. They can't squeeze.
You asked for the back room. The back room's gone.
You could cancel the lot so nobody goes
Or you send it as it stands and it's done.
It isn't mean. It isn't a message.
It's a room and a wall and a door.
The chairs ran out where the chairs ran out.
There isn't a chair on the floor.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
TABLE'S THE TABLE
CHAIRS ARE THE CHAIRS
THAT'S THE ROOM, THAT'S THE ROOM
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who sits? (Put me down)
Who fits? (Put me down)
Coats counted? (Put me down)
Chairs counted? (Put me down)
Who's standing? (Put me down)
Who's staying? (Put me down)
Table's full. (Put me down)
Who's full? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen. You've got the room.
You've got the window and the wall.
Everyone's in. Everyone fits.
Everyone but the one. That's all.
Who taught you to work out a room?
Who taught you to see where it stops?
Nobody did. You just look at it
And you know where the table drops.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
Somebody's stacking chairs on the shop floor.
Somebody's dragging a mop to the door.
The sheet's not the thing. The sheet's just paper.
The sheet does what the room can hold.
You cap the pen. You check the time.
Your tea went cold. Your tea's stone cold.
Thumb on the send and the thumb is quick
And the table by the window is told.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
GO ON
GO ON THEN
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
TABLE'S THE TABLE
CHAIRS ARE THE CHAIRS
THAT'S THE ROOM, THAT'S THE ROOM
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who sits? (Put me down)
Who fits? (Put me down)
PUT IT DOWN
Coats counted? (Put me down)
PUT IT DOWN
Chairs counted? (Put me down)
DOWN
Who's standing? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's staying? (Put me down)
Table's full. (Put me down)
Who's full? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who's sitting where? (Put me down)
Put me down.
*the message sends*
Strip light off. Door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.714** · line_return **0.548** · mean_words_per_line **5.452** · alliteration_per_100w **14.847**. Answer-stripped companion: rhyme **0.690** · line_return **0.548** · wpl **4.560** · allit **17.755**.

---

---

## V4 · **THE BOTTOM OF THE SHEET** — *I DIDN'T THINK*

**The reason, and why it is good enough (D6):** ⚠️ the dangerous one. The song refuses both available comforts — it does not confess and it does not scold — and states forgetting as **the ordinary limit of attention at the end of a shift**: *"You didn't decide. There wasn't a decision. … Your head held what a head holds."* ⛔ *"It isn't a crime and it isn't a kindness"* is the line that keeps D6 intact; without it this variation becomes a confession and invites absolution, which is the run's named failure mode wearing a different hat.
**Room:** the same back office, screen up, thumb nearly at the bottom of the sheet.
**Tempo/key:** 156 BPM, E Dorian. **Length target** 2:30–2:45 — ⭐ **the shortest song, as briefed.**

### V4 · MUSIC PROMPT (Suno style field) — measured **915 chars**

```
Flat and nearly finished. Fuzz-organ garage stomp, short: a cheap combo organ pushed into distortion, a five-note fuzz guitar riff, four-on-the-floor kick under a pummeling snare taking the beat and the offbeat, tambourine, blown out. Male tenor lead, early thirties, plain regional English, bright and nasal, half-swallowed line ends, splitting when he reaches the top note. Fast, 156 BPM, E Dorian. Recorded as a band in one small room, a back office with a low ceiling and carpet tiles, the door to the shop floor open and the room audible in every mic. Opens on a shout over a bare kick and gets going immediately. The verse is level and unhurried in delivery though the tempo is quick. The shouted line sits a fourth above the sung chant and falls while the chant rises. At the second chant the shout goes under the sung line and stays under while the band holds its size. Two bars, a stop. No risers, no fade.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 194 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, female vocal, choir, key change, fade out, ballad, extended outro, big final chorus, layered harmony
```

## V4 · RAW DRAFT LYRICS FIELD — `The Bottom Of The Sheet` — measured **3577 chars**, **73 sung lines**

```
[Theme: a back office at closing, the screen up and the thumb nearly done; the one who always organises the after-work thing gets to the bottom of the sheet and sends it, and a name is not on it because they did not think of it; second-person playground taunt from someone good at this; the shortest song]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the office door, bare kick]
*strip light ticks off*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Carpet tiles. Gone quiet.
Screen up, thumb down, nearly done.
Names in the boxes, boxes down the sheet,
And the sheet has a bottom and the bottom's the one.
Look at you. Look at you finishing.
Look at you nearly gone.
You get to the bottom of the sheet
And you send it. And it's sent. And it's on.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
You didn't decide. There wasn't a decision.
You didn't weigh it. You didn't sit.
Your head held what a head holds
At the end of a shift, and that was it.
You didn't think. That's the whole of it.
You didn't think and you're not lying.
It isn't a crime and it isn't a kindness.
It's a head at the end of a day, and it's closing.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
SEND IT, SEND IT
HEAD'S FULL, HANDS FULL
THAT'S THE SHIFT, THAT'S THE SHIFT
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who's on it? (Put me down)
Who's off it? (Put me down)
Who's sent? (Put me down)
Who went? (Put me down)
Who's thinking? (Put me down)
Who's finishing? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you're nearly done.
You've got a head like a full street.
Everyone in it was in it today.
Everyone you saw. And that's the sheet.
Who taught you to hold a whole shift?
Who taught you to hold it and think?
Nobody did. Nobody holds it.
You get to the bottom and you blink.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
Somebody's cashing up. Somebody's yawning.
Somebody's whistling at nothing.
The sheet's not the thing. The sheet's just paper.
The sheet does what a tired head does.
You put the cap back on the pen.
You are still going. Still going.

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
SEND IT, SEND IT
HEAD'S FULL, HANDS FULL
THAT'S THE SHIFT, THAT'S THE SHIFT
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's on it? (Put me down)
Who's off it? (Put me down)
PUT IT DOWN
Who's sent? (Put me down)
PUT IT DOWN
Who went? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's thinking? (Put me down)
Who's finishing? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who didn't think? (Put me down)
Put me down.
*the message sends*
Screen off. Door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.712** · line_return **0.534** · mean_words_per_line **5.644** · alliteration_per_100w **12.379**. Answer-stripped companion: rhyme **0.685** · line_return **0.534** · wpl **4.781** · allit **14.613**.

---
