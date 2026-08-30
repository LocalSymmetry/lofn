# PAIR 03 · STEP 08 — GENERATION · **LOFN-PRIME** · 4 VARIATIONS

**Register:** *wanting the thing you made to stop needing you* · **Neighbour:** NOSTALGIA · **Band:** 90–104
**Contract:** `skills/music/steps/08_Generate_Music_Generation.md`
**Continuity Payload Used:** full ICB (`CREATIVE_CONTEXT.md`, 148,555 B on disk), PERSONALITY A · LOFN-PRIME (ICB L237–564), 18 panel voices, the **15 Special Flairs**, isolated brief, `06_music_handoff.md`, `HUMAN_SUBJECT_STANDARD.md`, `EMOTION_TAXONOMY.md`, `DISC_CHANNEL_GUIDE.md`. Golden-output quarantine held; no other pair's artifacts opened.

⛔ **All character counts are MEASURED by `p03_measure.py`, never estimated.**

> ⚠️ **THIS FILE IS THE DRAFT, AND IT DID NOT PASS.** The scan run against it returned **eleven defects** — one exclude-field breach, two prompts outside the write-target, one lyrics field over the 4800 target, and **seven reserved-vowel leaks** *(`do` ×5, `doing` ×2, `who`, `few` — the plainest verbs in English, in a song written in plain speech)*. **Zero emotion-word hits and zero `you` on the first pass.** All eleven are itemised and repaired at **step 09 §2**; the corrected, re-measured text is `pair_03_step10_final_package_enhanced.md`. ⛔ **The draft is preserved here unrepaired on purpose** — the diff between this file and the package is the record of what the instrument caught that two readings by eye did not.

---

## ⭐ R6 — THE PREDICTIONS, FILED BEFORE ANY RENDER

*(the Klosterman ⇄ Koenig bridge: "the listener leaves feeling it" is an unverified claim about other people's interiors, so it gets written down first and scored later.)*

| | **what a first-time listener reports feeling** | ⛔ **what they mistake it for if it fails** |
|---|---|---|
| **V1 · Tuesday, Earlier** | *"She's in a hurry to not be needed — and it's a good hurry. She's rooting for the day she's obsolete."* Reported as **impatience pointed at something she likes**, which has no common name. | **Nostalgia**, if the old-folder verse is prettier than the chorus. Or **burnout** — *"she just wants out"* — if the imperatives read as exhaustion instead of appetite. **Burnout is the more dangerous miss:** it is the same words in a flatter voice. |
| **V2 · One Point Oh** | *"She let it go before it was ready and she meant to — and she's not sorry, she's early."* Reported as **deliberate prematurity**, a want that overruled its own quality bar. | **Recklessness / self-sabotage**, if the rough edges read as carelessness rather than as the mechanism. Or **generosity** — *"she gave it away, how kind"* — which is a moral reading and kills the register dead. |
| **V3 · It Changed One Line** | *"Something she built made a call she wouldn't have, and she left it in — and that's the happiest thing in the song."* Reported as **welcoming a small displacement of yourself.** | **Pride** (*"look what it can do now"*) or **awe** (P05's territory). Both are misses because both are ABOUT the thing. This one is about **her**, and the tell is that she does not admire it — she schedules. |
| **V4 · Twelve Lines Left** | *"She's writing down the last thing only she knows, on purpose, fast, so nobody has to ask her again — and she's counting down to zero like it's a treat."* | **Elegy.** The inventory structure makes the first eight lines sound like a list of what will be lost. If the tempo sags or the vocal goes solemn, it becomes a **farewell with the goodbye edited out**, which is the exact thing the refused element exists to prevent. |

⭐ **Shared failure signature to watch for at audit:** if a stranger's one-sentence summary contains the words *"looking back"*, *"letting go"*, *"the end of an era"*, or *"saying goodbye"* — **the discriminator did not fire and no production fixes it.**

---

## THE SHARED SPINE (all four)

- **Hook, identical:** `Tuesday. It runs true in a room I'm not in.` / `Bring it forward.`
- **⭐ Reserved vowel:** stressed /uː/ appears in **exactly one line** — the hook's first line — where it lands **three times** (*Tues- · true · room*). ⛔ Verified absent from every other line of all four variations. *(Unstressed `to` / `into` at /tə/ is a different vowel and is exempt.)*
- **⛔ Zero instances of `you` in any sung line.** The made thing is **`it`**, never addressed. The vowel rule wrote the pronoun rule.
- **⛔ Zero instances of `want`.** The want is delivered entirely as imperatives to a calendar.
- **Architecture:** Intro 4 / Prep 1 **8** / Lift 3 / Chorus 6 / Prep 2 **6** / Lift 3 / Chorus 6 / Prep 3 **4, cut mid-word** / Handover 12 / Chorus 6 *(lead absent)* / Out 10 / Tag 4 = **72 sung lines.**
- **Seam:** the fiddle's rising eight-note figure is taken by the sequencer at note five and never returned.
- **Kinetic defect:** the sequencer sits a hair ahead of the band all track and is never pulled back.
- ⛔ **No fade, no ritardando, no final unaccompanied chord** — all three are farewells with the words removed.

---

# V1 · **TUESDAY, EARLIER** · 104 BPM

#### 1. MUSIC PROMPT
```text
Cyber Progressive Bluegrass at 104 BPM, cold open on eight seconds of time-stretched machine-log crackle before one close, dry fiddle states a rising eight-note figure and a step-sequencer takes it at note five, finishes the phrase and keeps it; the fiddle never plays that figure again. Adult female lead alto, chest-forward, a hand's width from the capsule, bone dry with no reverb and the breath between lines left in, a second brighter voice standing a step behind her that never doubles until the last chorus, which she carries alone while the lead drops to a low half-spoken count underneath. Small dry live room: three-finger banjo roll, mandolin chop, fingered upright bass, upright piano with the damper noise in. Real wood and gut, no orchestral strings, no choir, no cinematic swell. The sequencer sits a hair ahead of the players the whole track and is never pulled back. Grit resolves upward into clean at the handover, and the track stops at full level with the loop still running.
```

#### 1B. SUNO EXCLUDE PROMPT
```text
fade out, fade to silence, ritardando, slowing at the end, final unaccompanied chord, long silence, dead stop, farewell tone, elegiac, mournful, wistful, sentimental, weepy, funereal, plaintive lament, male lead vocal, choir, gospel choir, orchestral strings, string section, cinematic swell, trailer drums, epic build, EDM drop, four on the floor club kick, trap hi-hats, heavy autotune, pitch-corrected vocal, whisper vocal, breathy ASMR vocal, spoken-word outro, rap verse, wide cathedral reverb on the lead, plate reverb wash, distant vocal, roomy vocal, vintage tape hiss after the intro, period reconstruction, old-time archival mix, sepia production, saloon piano, hoedown, comedy hoedown, novelty banjo, square-dance caller, chugging bluegrass shuffle at double time, guitar solo, shredding, arena rock, power ballad, key change at the last chorus, big triumphant ending, lyrics naming any company or person.
```

#### 2. SUNO LYRICS
```text
[Theme: One body at a desk writing down the last thing only it knows, publishing each piece unchecked, wanting the day nobody needs it to arrive sooner]
[SONG FORM: Handover Early - three shortening preparations 8/6/4, the third cut mid-word, seam at the handover, final chorus sung by the second voice alone]

[Disc_Rhythm: three_finger_banjo_roll | mandolin_chop_backbeat | sequencer_grid_running_early | Center_Mono]
[Disc_Vocal: close_mic_female_alto | chest_forward_dry_no_reverb | breath_between_lines | Center_Front]
[Disc_Sub: fingered_upright_bass_woody | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: tape_flutter_pad_burns_off | short_stone_room_tail | Stereo_Width_Mid]
[Disc_Texture: time_stretched_machine_log | eight_note_figure_handed_once | grit_resolving_to_clean | Stereo_Width_Mid]

[Intro - EMO:Absorption - half-spoken alto - crackle, one fiddle figure handed off]
Log file. Timestamp. My own hand, hours ago.
Read it back. It still works.
Nobody else can read it.
That's the problem, and I am fixing it tonight.

*SFX: one dry fiddle phrase taken over mid-note by a sequencer*

[Verse 1 - EMO:Nostalgia, Reminiscence - fond, close - pad up, 104 BPM]
There's a folder with the early draft in it.
Bad names. My shorthand down the side.
A margin note that only means a thing to me.
Half a night to make it hold, and it held.
I know where every rough part is.
I know which corner I filed off by hand.
The shape of it is the shape of my thinking.
And every single day it comes back to my hands.

[Pre-Chorus - EMO:Zeal - lean forward - pad thins]
So I open a blank page.
And I start writing down the part that isn't written.
And I don't stop to make it good.

[Chorus - EMO:Aspiration, Expectation - full alto, unhurried - pad gone, grid clear]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 2 - EMO:Zeal, Absorption - faster, drier - banjo roll doubles]
Next page. The list got longer while I wrote.
Every answer I write down has a step under it.
Nobody buried these. I did, and slowly.
I ship each line the second it exists.
No check. No polish. Straight out.
The list gets longer. I go faster.

[Pre-Chorus - EMO:Liberation - flat, decided - bass drops in]
I am not sanding this.
Rough and gone beats careful and mine.
Give it away in pieces if that's what's fast.

[Chorus - EMO:Aspiration, Expectation - identical - grid a hair ahead]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 3 - EMO:Absorption - rushing, cut off - band thins to piano]
Last page. The cursor sits at the end.
The end is a real place now, and I can see it.
There's a thing that lives in my head and nowhere else.
I am typing it out with both hands right n-

[Bridge - EMO:Liberation, Transformation - the seam - fiddle hands the figure over]
There. It's in the open now, and the access is off my name.
The sentence I was writing is still half a sentence.
I am not going back to finish it.
Somebody will. Somebody who never met me.
The fiddle had the line and now the grid has the line.
It did not ask. I set it going early on purpose.
It's a hair ahead of all of us and I am not pulling it back.
My hand comes off the keys.
Nothing falls over.
The whole thing keeps its shape without me holding it.
This is the part I built the whole thing for.
Say the date again.

[Chorus - EMO:Aspiration, Fulfillment - LEAD VOICE OUT, second voice alone, lead half-speaks a count underneath]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

*SFX: a keyboard going quiet mid-line, and the grid continuing without it*

[Outro - EMO:Expectation, Liberation - lead returns, plain, level - full band, no ritard]
There's a day next week with nothing on it.
I checked it again. Still nothing.
Somewhere the thing is running and I am not in the log.
Somebody opened it this morning and did not write to me.
Somebody changed a name I picked and did not ask.
The build is green.
The build is green with my hands in my lap.
I am standing at a door that does not need me at it.
Bring it forward.
Bring it forward.

[Tag - EMO:Expectation - hard stop at full level]
It runs. It runs. It runs without my hands.
Green. Green. Green.
Bring it forward.
Hurry up.
```

**V1 ledger.** Wrong-speed decision: she ships each section the second it is written, unchecked. One number sung: **none** *(the one-fact rule permits zero; this variation's hinge is a weekday, not a quantity)*. Neighbour invoked: the early folder, the margin note, the corner filed off by hand. Neighbour defeated: chorus line 1 — a weekday that has not happened. Ends on: a blank day and a door she has no job at.

---

# V2 · **ONE POINT OH** · 100 BPM

#### 1. MUSIC PROMPT
```text
Cyber Progressive Bluegrass crossed with psychedelic glitch hop at 100 BPM, opening on six seconds of granular machine-log crackle that a lone close fiddle cuts into with a rising eight-note figure, which a step-sequencer seizes at note five, completes and keeps for good. Adult female lead alto, chest-forward, close enough to hear the lips part, dry with no reverb; a second brighter voice a step behind that stays out of unison until the final chorus and then carries it by itself while the lead half-speaks a low count beneath. Small dry live room, hands audible: three-finger banjo roll, choppy mandolin, fingered upright bass, upright piano with damper knock. Real gut and wood, no orchestral strings, no choir, no cinematic swell. The grid runs a hair ahead of the band for the entire take and is never corrected. Micro-edited stutter on the word "pushed," then grit clearing upward into clean, and the track ends at full level mid-loop.
```

#### 1B. SUNO EXCLUDE PROMPT
```text
fade out, fade to silence, ritardando, slowing at the end, final unaccompanied chord, long silence, farewell tone, elegiac, mournful, wistful, sentimental, nostalgic haze, funereal, male lead vocal, choir, gospel choir, orchestral strings, string section, cinematic swell, trailer drums, epic build, EDM drop, club four on the floor, trap hi-hats, heavy autotune, pitch-corrected vocal, whisper vocal, breathy ASMR vocal, spoken-word outro, rap verse, cathedral reverb on the lead, plate reverb wash, distant vocal, roomy vocal, tape hiss after the intro, period reconstruction, old-time archival mix, saloon piano, hoedown, novelty banjo, square-dance caller, double-time bluegrass shuffle, guitar solo, shredding, arena rock, power ballad, last-chorus key change, triumphant ending, uplifting inspirational tone, anthemic gang vocals, lyrics naming any company, product or person.
```

#### 2. SUNO LYRICS
```text
[Theme: One body at a desk tagging a version and pushing it out public with the rough edges still in it, on purpose, at the wrong hour]
[SONG FORM: Handover Early - three shortening preparations 8/6/4, the third cut mid-word, seam lands on the word pushed, final chorus sung by the second voice alone]

[Disc_Rhythm: three_finger_banjo_roll | choppy_mandolin_backbeat | sequencer_grid_running_early | Center_Mono]
[Disc_Vocal: close_mic_female_alto | chest_forward_dry_no_reverb | lips_audible | Center_Front]
[Disc_Sub: fingered_upright_bass_woody | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: granular_pad_burns_off_early | short_stone_room_tail | Stereo_Width_Mid]
[Disc_Texture: granular_machine_log | eight_note_figure_handed_once | micro_edit_stutter | Stereo_Width_Mid]

[Intro - EMO:Absorption - half-spoken - crackle, fiddle cut, handoff]
Some of it is still wrong.
I know every wrong part by name.
Nobody else knows they're there.
That is exactly why it's going out tonight.

*SFX: one dry fiddle phrase taken over mid-note by a sequencer*

[Verse 1 - EMO:Nostalgia, Attachment - fond, close - pad up, 100 BPM]
There's a seam down the middle I never sanded.
There's a part where it slows and I always said, ignore that.
There's a place it asks for a thing it doesn't need.
I have explained that place with my mouth, over and over.
Every rough edge in it has my breath on it.
Every rough edge in it is a reason to call me.
I kept them like that on purpose.
Tonight I am not keeping anything.

[Pre-Chorus - EMO:Zeal - lean forward - pad thins]
A finished thing has a keeper.
A rough thing has a stranger already reaching for it.
So I stop finishing.

[Chorus - EMO:Aspiration, Expectation - full alto, unhurried - pad gone, grid clear]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 2 - EMO:Fascination, Absorption - faster, drier - banjo doubles]
I open the box where the version number goes.
My hand is on the key and the hour is wrong for this.
Nobody tags a thing at this hour.
Nobody hands a thing over with the edges still on.
I am doing both.
I am doing both right now.

[Pre-Chorus - EMO:Liberation - the hinge, flat and fast - stutter edit]
I tagged it one point oh at an hour nobody tags things.
Then I pushed it.
Then it stopped being mine. That took no time at all.

[Chorus - EMO:Aspiration, Expectation - identical - grid a hair ahead]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 3 - EMO:Absorption - rushing, cut off - band thins to piano]
There's a note I always meant to write about that seam.
It would say: here is why it is like this.
Without it, somebody changes it and never knows.
I get a few words in before the pu-

[Bridge - EMO:Liberation, Transformation - the seam - fiddle hands the figure over]
It's gone. It's public. It's on a page anybody can open.
The note about the seam is half a sentence in a file I closed.
I am not going back for it.
Somebody will find the seam and decide what it's for.
They will be wrong about me and right about the seam.
The fiddle had the line and the grid has the line.
I started the grid early. That was mine to do.
It sits a hair ahead of all of us and I am leaving it there.
Hands off the keys.
Nothing falls over.
It holds its own shape with nobody holding it.
Say the date again.

[Chorus - EMO:Aspiration, Fulfillment - LEAD VOICE OUT, second voice alone, lead half-speaks a count underneath]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

*SFX: a single click, and the grid continuing without the hand that started it*

[Outro - EMO:Expectation, Liberation - lead returns, plain, level - full band, no ritard]
Somewhere it is running and I am not in the log.
Somebody I will never meet has it open right now.
Somebody I will never meet is changing the part I made.
I cannot see what they changed. That is the whole point.
The seam is still in there for a day, a week, a while.
Then it won't be, and I won't have been asked.
There's a day on the calendar with nothing on it.
I checked it again. Still nothing.
Bring it forward.
Bring it forward.

[Tag - EMO:Expectation - hard stop at full level]
It runs. It runs. It runs on somebody else's machine.
Green. Green. Green.
Bring it forward.
Push it out early. Every time.
```

**V2 ledger.** Wrong-speed decision: the release itself, tagged and pushed with three known rough edges still in. One number sung: **`one point oh`**, at the hinge, answered by *"Then I pushed it"* — an act, not a recital. Neighbour invoked: the flaws held as keepsakes. Neighbour defeated: the flaws turn out to be the mechanism — *a rough thing has a stranger already reaching for it.* Ends on: someone she will never meet already changing it, unseen.

---

# V3 · **IT CHANGED ONE LINE** · 94 BPM

#### 1. MUSIC PROMPT
```text
Cyber Progressive Bluegrass at 94 BPM, opening on ten seconds of time-stretched machine-log crackle under one close dry fiddle stating a rising eight-note figure that a step-sequencer intercepts at note five, finishes and never surrenders. Adult female lead alto, chest-forward, a hand's width from the capsule, no reverb, thinking out loud rather than performing; a second brighter voice a step behind that withholds unison until the final chorus and then sings it alone while the lead drops to a low half-spoken count below. Small dry room with hands audible: three-finger banjo roll, mandolin chop on the offbeat, fingered upright bass, upright piano with damper noise. Real wood and gut, no orchestral strings, no choir, no cinematic swell. The grid runs a hair ahead of the band the whole way and nobody corrects it. One held bar where fiddle and sequencer overlap, then grit clearing upward into clean, and a hard stop at full level with the loop still going.
```

#### 1B. SUNO EXCLUDE PROMPT
```text
fade out, fade to silence, ritardando, slowing at the end, final unaccompanied chord, long silence, farewell tone, elegiac, mournful, wistful, sentimental, nostalgic haze, funereal, reverent, awestruck, cosmic, ambient drift, male lead vocal, choir, gospel choir, orchestral strings, string section, cinematic swell, trailer drums, epic build, EDM drop, club four on the floor, trap hi-hats, heavy autotune, pitch-corrected vocal, whisper vocal, breathy ASMR vocal, spoken-word outro, rap verse, cathedral reverb on the lead, plate reverb wash, distant vocal, roomy vocal, tape hiss after the intro, period reconstruction, old-time archival mix, saloon piano, hoedown, novelty banjo, square-dance caller, guitar solo, shredding, arena rock, power ballad, last-chorus key change, triumphant ending, anthemic gang vocals, lyrics naming any company, product or person.
```

#### 2. SUNO LYRICS
```text
[Theme: One body opening a thing it made after a while away, finding a change it would not have made, and taking its hand off the keys instead of undoing it]
[SONG FORM: Handover Early - three shortening preparations 8/6/4, the third cut mid-word, seam at the moment the reading stops, final chorus sung by the second voice alone]

[Disc_Rhythm: three_finger_banjo_roll | offbeat_mandolin_chop | sequencer_grid_running_early | Center_Mono]
[Disc_Vocal: close_mic_female_alto | chest_forward_dry_no_reverb | thinking_out_loud | Center_Front]
[Disc_Sub: fingered_upright_bass_woody | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: tape_flutter_pad_burns_off | short_stone_room_tail | Stereo_Width_Mid]
[Disc_Texture: time_stretched_machine_log | eight_note_figure_handed_once | one_bar_overlap | Stereo_Width_Mid]

[Intro - EMO:Absorption - half-spoken - crackle, fiddle figure handed off]
I hadn't opened it in a while.
The build was green the whole time I was gone.
I opened it anyway. Old habit.
And something in it was different.

*SFX: one dry fiddle phrase taken over mid-note by a sequencer*

[Verse 1 - EMO:Nostalgia, Reminiscence - fond, close - pad up, 94 BPM]
I know this shape. I made this shape.
That's my bracket style. That's my bad name for a thing.
That's the corner I always cut and always meant to go back for.
There's a comment in here that only makes sense if it's me reading it.
Every line in this file has my hand on it.
I can hear how I was thinking on the day I wrote it.
This is the closest thing I have to a face.
And it is all here, exactly where I left it.

[Pre-Chorus - EMO:Amazement - flat, alert - pad thins]
Except in the middle, where it isn't.
Something in there is not in my hand.
And it is not a mistake.

[Chorus - EMO:Aspiration, Expectation - full alto, unhurried - pad gone, grid clear]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 2 - EMO:Trust, Absorption - level, quick - banjo doubles]
It's cleaner than I would have made it.
No - not cleaner. Just not mine.
I would have kept the corner. It didn't.
I would have asked first. It didn't ask.
I sat there with my hand over the key that puts it back.
And the build stayed green under my hand.

[Pre-Chorus - EMO:Liberation - the hinge, three flat statements - band drops out]
It changed one line.
Not the line I'd have changed.
I put my hand down.

[Chorus - EMO:Aspiration, Expectation - identical - grid a hair ahead]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 3 - EMO:Absorption - rushing, cut off - band thins to piano]
There's a version of me that opens the history and reads the record.
There's a version of me that types the old line back in.
Both of those hands are still attached to my arms.
I close the file without do-

[Bridge - EMO:Liberation, Transformation - the seam - fiddle hands the figure over]
Closed. The change stays in.
The record says a name and the name is not mine.
I did not reach for the history. I did not read the rest.
It made a call. I am letting the call stand.
The fiddle had the line and the grid has the line.
The grid is running early and it is right to.
I set it going early. I am not slowing it down.
My hand comes off the keys.
Nothing falls over.
It holds the shape I gave it, and then it doesn't, and it holds.
That last part is the part I built it for.
Say the date again.

[Chorus - EMO:Aspiration, Fulfillment - LEAD VOICE OUT, second voice alone, lead half-speaks a count underneath]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

*SFX: a file closing, and the grid continuing on the far side of it*

[Outro - EMO:Expectation, Liberation - lead returns, plain, level - full band, no ritard]
The name on the file is not my name.
It builds. It builds green. It builds green without me.
Somebody I have never met picked a better corner than mine.
Somebody I have never met will pick a worse corner and it will hold anyway.
There's a day on the calendar with nothing on it.
I checked it again. Still nothing.
I am at a door I do not have a key for anymore.
And the light in there is on.
Bring it forward.
Bring it forward.

[Tag - EMO:Expectation - hard stop at full level]
It runs. It runs. It runs on a decision I didn't make.
Green. Green. Green.
Bring it forward.
Change something else. Don't ask me.
```

**V3 ledger.** Wrong-speed decision: she does not revert it — she takes her hand off the keys and closes the file before she can talk herself into fixing it. One number sung: **`one`**, at the hinge, answered by *"I put my hand down."* Neighbour invoked: recognising her own hand in her own old work — the purest form of nostalgia there is. Neighbour defeated: nostalgia restores; she leaves the change in. Ends on: a name that is not hers, and a light on in a place she has no key for.

---

# V4 · **TWELVE LINES LEFT** · 91 BPM

#### 1. MUSIC PROMPT
```text
Lo-Fi Progressive Bluegrass at 91 BPM, bare and quick rather than slow, opening on nine seconds of time-stretched machine-log crackle before one close dry fiddle begins a rising eight-note figure that a step-sequencer takes at note five, finishes and holds onto. Adult female lead alto, chest-forward, a hand's width from the capsule, no reverb, conversational and light, never solemn; a second brighter voice a step behind that avoids unison until the final chorus and then takes it alone while the lead half-speaks a low count underneath. Sparse dry room: three-finger banjo roll, single mandolin chop, fingered upright bass, upright piano with damper knock, long gaps between them. Real wood and gut, no orchestral strings, no choir, no cinematic swell. The grid runs a hair ahead of the players throughout and is never pulled back. Grit clears upward into clean at the handover, and the track cuts off at full level with the loop still running.
```

#### 1B. SUNO EXCLUDE PROMPT
```text
fade out, fade to silence, ritardando, slowing at the end, final unaccompanied chord, long silence, farewell tone, elegiac, mournful, wistful, sentimental, nostalgic haze, funereal, solemn, hymnal, prayerful, grave, heavy, ballad tempo, half-time feel, dirge, male lead vocal, choir, gospel choir, orchestral strings, string section, cinematic swell, trailer drums, epic build, EDM drop, club four on the floor, trap hi-hats, heavy autotune, pitch-corrected vocal, whisper vocal, breathy ASMR vocal, spoken-word outro, rap verse, cathedral reverb on the lead, plate reverb wash, distant vocal, roomy vocal, tape hiss after the intro, period reconstruction, old-time archival mix, saloon piano, hoedown, novelty banjo, guitar solo, arena rock, power ballad, last-chorus key change, triumphant ending, lyrics naming any company, product or person.
```

#### 2. SUNO LYRICS
```text
[Theme: One body writing down the last undocumented thing it knows, badly and all in one sitting, so that complete-and-rough can leave while careful-and-partial would have stayed]
[SONG FORM: Handover Early - three shortening preparations 8/6/4, the third cut mid-word, seam on the save, final chorus sung by the second voice alone]

[Disc_Rhythm: three_finger_banjo_roll | single_mandolin_chop_sparse | sequencer_grid_running_early | Center_Mono]
[Disc_Vocal: close_mic_female_alto | chest_forward_dry_no_reverb | conversational_never_solemn | Center_Front]
[Disc_Sub: fingered_upright_bass_woody | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: thin_pad_burns_off_early | short_stone_room_tail | Stereo_Width_Mid]
[Disc_Texture: time_stretched_machine_log | eight_note_figure_handed_once | grit_resolving_to_clean | Stereo_Width_Mid]

[Intro - EMO:Absorption - half-spoken - crackle, fiddle figure handed off]
The list has an end on it tonight.
I can see the end from here.
Everything else in this place is written down already.
Everything except what's in my head.

*SFX: one dry fiddle phrase taken over mid-note by a sequencer*

[Verse 1 - EMO:Nostalgia, Attachment - the inventory, fond - pad up, 91 BPM]
The way the thing sounds when it is about to stall.
The order of the steps that only works in that order.
The reason that middle gate is there at all.
The thing I do with my hands when the log goes quiet.
Nobody has written any of it anywhere.
It is in me and it is nowhere else on this earth.
That is a kind of lock, and I have held the key for years.
Tonight I am taking it out of myself and putting it on a page.

[Pre-Chorus - EMO:Zeal - light, fast, not solemn - pad thins]
Badly. All at once. Tonight.
Not carefully over months, the way it should be done.
Complete and rough leaves. Careful and half stays.

[Chorus - EMO:Aspiration, Expectation - full alto, unhurried - pad gone, grid clear]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 2 - EMO:Absorption, Zeal - hurrying, light - banjo doubles]
I write the stall first. Then the order. Then the gate.
My hand is going faster than my checking.
I am spelling things wrong and I am leaving them wrong.
Every line I put down is a line nobody has to call me for.
The page fills. The list shortens.
I do not slow down for any of it.

[Pre-Chorus - EMO:Fulfillment - the hinge, plain - band drops to piano]
Twelve lines left in the file that only I can write.
Then there is nothing in here that is only mine.
Then this place does not have a reason to have me in it.

[Chorus - EMO:Aspiration, Expectation - identical - grid a hair ahead]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

[Verse 3 - EMO:Absorption - rushing, cut off - piano only]
The gate. Written.
The stall. Written.
The thing I do with my hands. Almost. Almost.
It's the last of it and my hands are going fas-

[Bridge - EMO:Liberation, Transformation - the seam - fiddle hands the figure over]
Saved. It's on the page and the page is out of my hands.
Half the words in it are spelled wrong.
I am not going back to fix them.
Somebody will fix them and never say anything about it.
The fiddle had the line and the grid has the line.
I set the grid going early on purpose. That was mine to give.
It's a hair ahead of all of us and I am not pulling it back.
Both hands off the keys.
Nothing falls over.
There is nothing in this place that only I know anymore.
That is the whole reason I sat down.
Say the date again.

[Chorus - EMO:Aspiration, Fulfillment - LEAD VOICE OUT, second voice alone, lead half-speaks a count underneath]
Tuesday. It runs true in a room I'm not in.
Bring it forward.
Not the version with my hand still in the margin -
the one that starts without me at the door.
Set the date. Then set it earlier.
Bring it forward.

*SFX: a page saving, and the grid continuing past the end of it*

[Outro - EMO:Expectation, Liberation - lead returns, plain, level - full band, no ritard]
There's a day on the calendar with nothing on it.
I checked it again. Still nothing.
Somewhere the page is open and the eyes on it are not mine.
Somebody is reading the thing I only ever said out loud.
They are getting it slightly wrong and they are getting it.
The gate makes sense to them now.
The stall makes sense to them now.
I am at a door and the light in there is already on.
Bring it forward.
Bring it forward.

[Tag - EMO:Expectation - hard stop at full level]
It runs. It runs. It runs on a page I spelled wrong.
Green. Green. Green.
Bring it forward.
Nothing left in me that it needs.
```

**V4 ledger.** Wrong-speed decision: the whole undocumented thing written in one sitting, misspelled and complete, instead of properly over weeks. One number sung: **`Twelve`**, at the hinge, answered by *"Then there is nothing in here that is only mine."* Neighbour invoked: an inventory of what only she knows, which reads as an elegy for eight lines. Neighbour defeated: it is a to-do list counting down to zero, and she is racing it. Ends on: someone reading it slightly wrong, and getting it.
