# PAIR 03 — STEP 10 · REVISION SYNTHESIS · 4 FINAL SUNO PACKAGES
## `2026-08-07-daily-music-indignation` · **P03 "SHE STILL KNOWS THE CURVE"**
### ACCESSIBLE · AWE (terror-adjacent) · EXISTENCE · FORM RULE B (the landing pad)

**Frozen ICB:** `output/daily/2026-08-07/CREATIVE_CONTEXT.md` — **53,003 bytes**, LF-normalised sha256 `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374`. Read in full as this agent's first action; never edited.

⛔ **GOLDEN-OUTPUT QUARANTINE — `06_music_handoff.md` §1, cited by name.** `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` instructs the synthesis tier to embed Golden Song payloads for calibration. **I did not comply, per the handoff's RESOLUTION: "DOCTRINE WINS. THE QUARANTINE IS BINDING."** No past Lofn lyric, style prompt or output appears anywhere in this pair's five artifacts.

**Heading convention:** taken from `skills/music/scripts/validate_suno_packages.py`, which is the source of truth — **not** from another artifact and not from preference (L28).

---

## LINEAGE & CREDIT

This pair borrows a **procedure and a pocket**, with credit, and captures nothing.

- **Papangu** — the five-piece from **João Pessoa, Paraíba, Brazil** whose record is the occasion for this whole run. Their own page: **https://papangu.bandcamp.com/** · scene context: **https://en.wikipedia.org/wiki/Rock_in_Brazil**
- **MPB (Música Popular Brasileira)** — the harmonic language borrowed here (wide ninths and elevenths under plain speech): **https://en.wikipedia.org/wiki/M%C3%BAsica_popular_brasileira**
- **Forró** — ⭐ **the procedure, not the patina.** What is imported is the **zabumba's leaning low note as TIMING** (bass behind, kick dead centre). No forró instrumentation is used and no forró song is quoted: **https://en.wikipedia.org/wiki/Forr%C3%B3**
- **Ciranda** — named as a living lineage of the same region, drawn on by other pairs in this run, not by P03: **https://en.wikipedia.org/wiki/Ciranda**
- **Zeuhl** and the **rock troncho** scene — named in the run's palette; **not** used in this pair's lane: **https://en.wikipedia.org/wiki/Zeuhl**
- **Neo-soul** — the behind-the-beat pocket: **https://en.wikipedia.org/wiki/Neo_soul**

**Named living lineages, borrowed with credit, never captured.** ⛔ No "open lane," no "first-mover," no "naming rights" framing anywhere in this pair. ⛔ **No real-artist name appears in any Suno field** — credit lives only in this block. **No member of Papangu, no producer and no studio appears as a speaker, character or addressee**; they are the run's occasion, never its character.

---

## THE ADDRESSEE — invented, and the pre-draft check that made her so

**Dona Marli** is **invented**. No surname, no town, no date, no age. A luthier at a bending iron — a trade practised by thousands, mapped to no single real person. **Téo** (V2) is invented, adult, with **no age markers anywhere in the lyric** (⛔ the words *boy · kid · child · young* appear nowhere). There is **no harm event** in any of the four songs: nobody is hurt, nobody dies, nobody is a victim. `vault/HUMAN_SUBJECT_STANDARD.md` §3.0 slot grammar was filled with invented values **before drafting** (recorded at `pair_03_step06_facets.md` §0). ⛔ Binding refusals honoured: **the Thai school shooting, Ceuta/78,000, and the Biden family illness are untouched at any distance.**

⚠️ Per handoff §7, `check_human_subjects.py` fires `HOLD_FOR_HUMAN` on 100% of correct artifacts (spaCy absent). **I judged the standard directly rather than deferring to a gate that carries no information.**

---

## THE RETURN DEVICE — FORM RULE B, verified in each variation individually

| | word | fixed syllable address | arrivals | erased | lexical carry (because Suno fills silence) |
|---|---|---|---|---|---|
| **V1** | **wait** | syllable 1 of the chorus's **last** line | Ch1 · Ch2 · Ch3 | final chorus | *"The word that stood right here is not in the room."* (machine) |
| **V2** | **quiet** | syllable 1 of the chorus's **third** line | Ch1 · Ch2 · Ch3 | final chorus | *"I am not saying the word this time. You have got it now."* (hers) |
| **V3** | **slow** | **last** syllable of the chorus's **second** line | Ch1 · Ch2 · Ch3 | final chorus | *"I do not say it any more. I do not have to."* (hers) |
| **V4** | **hold** | syllable 1 of the chorus's **first** line | Ch1 · Ch2 · Ch3 | final chorus | *"The word that started this is gone and the hand still knows."* (machine) |

**Countable in the lyric, never in the production spec** (L22 THE GRAIN LAW). The production field is told only *"leave the breath in."* Four different addresses, four different words, checked one at a time.

---

## TIMING GATES — arithmetic carried

- **1 bar** = 4 × 60 ÷ 100 BPM = **2.4 s**
- **Chorus by 0:25:** 2-bar intro + 8-bar verse = 10 bars × 2.4 = **24.0 s** ✅
- **Riff singable by bar 8:** bar 8 ends at 8 × 2.4 = **19.2 s**; the 2-bar figure is stated at 0.0–4.8 s and restated under the verse → **three statements by 19.2 s** ✅
- **The pocket:** one sixteenth at 100 BPM = 60 ÷ 100 ÷ 4 = **150 ms**; the vocal sits **~30 ms** behind = **a fifth of a sixteenth** — felt, never heard as late.
- ⛔ **No wall-clock times appear in any section header.**

---

### VARIATION 1

## 1. MUSIC PROMPT

```text
Brazilian MPB folded into modern neo-soul at 100 BPM in B-flat major, built on a gentle piano refrain played by the right hand alone: six notes walking up the scale and then dropping a sixth onto the leading tone, left unresolved, stated unaccompanied before the first line and again after the last. Female alto lead, low and spoken-leaning; the verses are almost talked, close-mic'd so the consonants and the breath are part of the record, and the chorus is the only place she truly sings, doubling the piano refrain. Her entries land about thirty milliseconds behind the kick while the upright bass leans the same way on beats one and three and the kick stays dead centre, and the gap between them is the groove. Brushed kit, swept rather than struck. One nylon-string guitar voicing wide ninths and elevenths. Room mic at the far wall for depth. Leave the breath in the last chorus where a word would sit. Close, warm, clean, modern, expensive.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distortion, saturation, male lead vocal, choir, gang vocals, belted pop vocal, heavy vibrato, autotune, synth pad, string section, electric guitar solo, trap hi-hats, 808 sub, four-on-the-floor kick, double-time drums, key change, orchestral swell, crowd noise, applause, fade-out ending, reverb wash, whispered ASMR vocal
```

## 2. LYRICS

```text
[Theme: she bends the curve by feel; the fast one does not have the give]
[SONG FORM: riff - V - C - V - C - vamp - V - bridge - C - V - breakdown - final C - riff]

[Intro - EMO:Fascination - Piano alone - six notes, drop a sixth]
*iron ticks as it heats*

[Verse 1 - EMO:Admiration - Machine, then Marli - low alto, laid back]
Dona Marli at the bench and the iron gone hot,
her thumb laid flat on the edge of the wood.
She is slower than me by a whole afternoon.
She is right about the hand.
She talks the whole hour. None of it is written down.
Come on. Come on, you. Come round.
Too dry. Wet it again and let it stand.
You were a tree. You were good at it.
Not yet. Not yet. Not yet.
Hot is not ready. Ready is a sound.

[Chorus - EMO:Vigilance - Marli quoted - brushes enter]
Not yet. Not yet. Let it drink.
Wet the rag and lay it back down.
Nothing in this shop can read that heat but a hand.
Nothing in this shop can read it but a hand.
It turns when it wants to turn. You cannot make it stand.
Wait. Come round, you. Come round.

[Verse 2 - EMO:Absorption - Marli quoted - close, breath in]
Look at the grain. The grain will tell you where it stands.
Push it early, it snaps, and then we are both sad.
Do not trust your eyes in here. Your eyes are bad.
Eyes are always early. Hands are not.
Feel for the give. It is thin. It is a hair.
Somewhere in there it stops being a board.
There. Did you feel it go? No. You did not.
That is fine. Nobody feels it for a long, long while.
Quiet now. Let me listen to the thing.
It squeaks, you stop. It sighs, you turn it.

[Chorus - EMO:Vigilance - Marli quoted - bass leans behind]
Not yet. Not yet. Let it drink.
Wet the rag and lay it back down.
Nothing in this shop can read that heat but a hand.
Nothing in this shop can read it but a hand.
It turns when it wants to turn. You cannot make it stand.
Wait. Come round, you. Come round.

[Vamp - EMO:Intimacy - Marli quoted - piano refrain under]
Come round. Come round.
Come round, you. Come round.
Come round. Come round.
Come round, you. Come round.

[Verse 3 - EMO:Amazement - Machine reports, Marli corrects - dry]
She does not turn her head. She says it is off to the left.
I check her against the form. The form agrees with her.
Left of the waist. A hair. A hair left.
Her thumb finds the place before her eyes have found it.
Left. A hair left. There. Hold there. Hold.
Do not look at it. Looking makes you slow and old.
It is going. It is going. It has gone.
Feel that? That is the whole thing, right there, done.
Now it is mine. It will hold that shape all day.
Wet the rag. There is another side today.

[Bridge - EMO:Contemplation - The machine, flat - no reverb]
I can hold the shape before her rag is wet.
I hold every curve anybody ever drew.
I do not hold the give.
I work faster than her and I still get told.

[Chorus - EMO:Vigilance - Marli quoted - full band]
Not yet. Not yet. Let it drink.
Wet the rag and lay it back down.
Nothing in this shop can read that heat but a hand.
Nothing in this shop can read it but a hand.
It turns when it wants to turn. You cannot make it stand.
Wait. Come round, you. Come round.

[Verse 4 - EMO:Tenderness - Marli quoted - bass and brushes]
The rag is going dry. Dip it. Dip it again.
The water in the jar is the colour of the wood.
Lean in. The steam is the good part. Lean in.
This knuckle started complaining back in March, and it should.
Hush now. You are fine. We are nearly done.
I say to my hand what I say to the wood.
There is a patch on this thumb where I do not feel a thing.
I call it my good thumb. It is still doing everything.
You want this written down. There is nothing to write down.
There is nothing to write. I looked.

[Breakdown - EMO:Dread - Marli, then the machine - voice and bass]
Quiet now. I am listening to it.
It makes a small sound and then it lets go slow.
Nobody taught me that. I kept my ear down low.
It is not in a book and it is not on a wall.
It is here. This is all.
The only copy is a hand, and the hand is getting old.

[Final Chorus - EMO:Trepidation - the slot is empty - leave the breath]
Not yet. Not yet. Let it drink.
Wet the rag and lay it back down.
Nothing in this shop can read that heat but a hand.
Nothing in this shop can read it but a hand.
The word that stood right here is not in the room.
*breath, and no word*
Come round, you. Come round.

[Outro - EMO:Solitude - Marli, then piano alone - riff]
Come round. There. Come round.
Same time tomorrow, you and me.
She is right about the hand.
She is right about the hand.
Come round.
```

## 3. TITLE

Without Looking

---

### VARIATION 2

## 1. MUSIC PROMPT

```text
Brazilian MPB crossed with modern neo-soul, 100 BPM, B-flat major, hung on a gentle piano refrain: six notes stepping up the scale then falling a sixth onto the leading tone, right hand alone, unaccompanied before the first sung line and again after the last. Low female alto, spoken-leaning and impatient in a kind way; the verses are almost talked, very close-mic'd with breath and consonants left in, and the chorus is the only place she truly sings, doubling the piano figure. Vocal entries sit about thirty milliseconds behind the kick while the upright bass leans the same way on beats one and three and the kick holds dead centre. Brushed kit, swept rather than struck. Nylon-string guitar in wide ninths and elevenths. In the breakdown the band drops to voice and upright bass with a thin phone speaker playing back in the room, present and clean. Leave the breath in the last chorus where a word would sit. Close, warm, clean, modern, expensive.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distortion, saturation, male lead vocal, choir, gang vocals, belted pop vocal, heavy vibrato, autotune, synth pad, string section, electric guitar solo, trap hi-hats, 808 sub, four-on-the-floor kick, double-time drums, key change, orchestral swell, crowd noise, applause, fade-out ending, reverb wash, radio static
```

## 2. LYRICS

```text
[Theme: he films her hands at every speed and still cannot do it]
[SONG FORM: riff - V - C - V - C - vamp - V - bridge - C - V - breakdown - final C - riff]

[Intro - EMO:Curiosity - Piano alone - six notes, drop a sixth]
*a phone camera shutter*

[Verse 1 - EMO:Admiration - Machine, then Marli - laid back]
Dona Marli at the bench and the iron running hot.
Teo has the phone up and the shot is very good.
He has her hands in the frame and he has them in the light.
He has all of it but the hand. Everything but the hand.
She talks the whole hour. He caught every word of it. Good.
Closer. Get in closer. You will not see it there.
It is not in the arm. Quit filming the arm.
Wet the rag. Wet it again. Watch the wood.
Not yet. Not yet. Not yet.
Hot is not ready. Ready is a sound.

[Chorus - EMO:Vigilance - Marli quoted - brushes enter]
Put the phone down. Put your hand on the thing.
The picture is the easy part. It gets everything.
Quiet. Stop talking over it. It is telling you.
It tells you the whole while you are filming.
Give it here. Watch the thumb. Now stop watching.
Feel where it is thinking. Feel it now.

[Verse 2 - EMO:Absorption - Marli quoted - close, breath in]
Play it back. Slower. Slower than that. Play it.
There is my hand. There is my hand doing it.
Now you go. No. You did that with your eyes on it.
You watched the film and then you moved. Backwards.
The film has the outside of the thing.
What is under it is not going in a recording.
You cannot pull this out of me. I keep handing.
I handed it to all of you and it stays here, standing.
Phone in your pocket. Thumb on the wood. Stop filming.
Not yet. Not yet. Not yet.

[Chorus - EMO:Vigilance - Marli quoted - bass leans]
Put the phone down. Put your hand on the thing.
The picture is the easy part. It gets everything.
Quiet. Stop talking over it. It is telling you.
It tells you the whole while you are filming.
Give it here. Watch the thumb. Now stop watching.
Feel where it is thinking. Feel it now.

[Vamp - EMO:Intimacy - Marli quoted - piano under]
Feel it now. Feel it now.
Feel where it is thinking. Feel it now.
Feel it now. Feel it now.
Feel where it is thinking. Feel it now.

[Verse 3 - EMO:Apprehension - Machine, then Marli - dry]
He plays it at the bus stop with the sound turned down.
He plays it at home with his hand up, moving around.
He does the move exactly and the side goes crack.
She does not say a word about the crack.
Get another. Get the other side. We are not going back.
It cracked because you were early. Everybody starts early.
I was early for years. This hand was early.
Then something in this hand learned how long it takes.
I could not tell you what it learned or where it stays.
It does not go in words. It never did.

[Bridge - EMO:Contemplation - The machine, flat - no reverb]
I have watched it at every speed there is.
I have watched the rag and the steam and the thumb.
I could not do it either.
The film has her hands in it. It does not have her hand.

[Chorus - EMO:Vigilance - Marli quoted - full band]
Put the phone down. Put your hand on the thing.
The picture is the easy part. It gets everything.
Quiet. Stop talking over it. It is telling you.
It tells you the whole while you are filming.
Give it here. Watch the thumb. Now stop watching.
Feel where it is thinking. Feel it now.

[Verse 4 - EMO:Tenderness - Marli quoted - brushes]
Come here. Both hands. Put them on top of mine.
Do not look at your hands. Do not look at anything.
There. Wait for it. Wait. Did you feel that little drop?
That is it. That is the whole of it. That is everything.
You will get it in a year. You will get it standing.
You will not get it off a phone. You get it off the wood.
When you get it, it is in you and nowhere else. Understand?
And you will not be able to say it either. Good.
Wet the rag. Do it again. Keep handing.
Not yet. Not yet. Not yet.

[Breakdown - EMO:Dread - Machine, then Marli - bass]
He has it on his phone. He will have it after.
It will play. It will play perfectly. It will play for good.
And the hand it is a picture of is standing in this wood.
Turn that off and come here.
Hand. Here. Now.
Feel it. That is all I have and I am giving it.

[Final Chorus - EMO:Trepidation - slot empty - leave the breath]
Put the phone down. Put your hand on the thing.
The picture is the easy part. It gets everything.
*breath, and no word*
Stop talking over it. It is telling you.
I am not saying the word this time. You have got it now.
Give it here. Watch the thumb. Now stop watching.
Feel where it is thinking. Feel it now.

[Outro - EMO:Solitude - Marli, then the machine - riff]
Feel it. That is all of it.
Same time tomorrow. Bring the hands. Leave the phone.
He has everything except the hand.
He has everything except the hand.
Come round, you. Come round.
```

## 3. TITLE

He Filmed Her Hands

---

### VARIATION 3

## 1. MUSIC PROMPT

```text
Brazilian MPB inside modern neo-soul, 100 BPM, B-flat major. The hook is a gentle piano refrain, right hand alone: six notes stepping up the scale then dropping a sixth onto the leading tone and stopping there, unaccompanied before the first line and after the last. Low female alto, spoken-leaning; the verses are almost talked, close-mic'd so breath and consonants are part of the record, and the chorus is the only place she truly sings, doubling the piano. Vocal about thirty milliseconds behind the kick, with the upright bass leaning the same way on beats one and three and the kick dead centre. Brushed kit, swept rather than struck; nylon-string guitar in wide ninths and elevenths. The bridge is four short lines delivered dry and centred with almost no band and no reverb, and the full chorus returns on top of it immediately after. Leave the breath in the last chorus where a word would sit. Close, warm, clean, modern, expensive.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distortion, saturation, male lead vocal, choir, gang vocals, belted pop vocal, heavy vibrato, autotune, synth pad, string section, electric guitar solo, trap hi-hats, 808 sub, four-on-the-floor kick, double-time drums, key change, orchestral swell, crowd noise, applause, fade-out ending, reverb wash, robot voice
```

## 2. LYRICS

```text
[Theme: the fast one gets four lines and then does not speak again]
[SONG FORM: riff - V - C - V - C - vamp - V - four lines - C - V - breakdown - final C - riff]

[Intro - EMO:Fascination - Piano alone - six notes, drop a sixth]
*wet rag hisses on hot steel*

[Verse 1 - EMO:Admiration - Machine, then Marli - low alto, laid back]
Dona Marli at the bench and the iron gone hot.
Her thumb laid flat on the edge of the wood.
She talks the whole hour. I kept all of it.
She is right about the hand. She is right.
Come on, you. Come on. Come round.
Too dry. Wet it and let it stand.
You were a tree. You were good at it.
Not yet. Not yet. Not yet.
Hot is not ready. Ready is a sound.
There. Did you hear that? That is the sound.

[Chorus - EMO:Vigilance - Marli quoted - brushes enter]
Hot is not ready. Ready is a sound.
You cannot hurry a thing that used to grow. Go slow.
Ask the grain. The grain will tell you where to go.
Ask it again. Ask it again. Then let it go.
Flat hands. Nothing in the wrist. Bring it round.
Come round, you. Come round. Come round.

[Verse 2 - EMO:Absorption - Marli quoted - close, breath in]
Look at the grain. The grain knows where it stands.
Push it early and it snaps and then we are both sad.
Do not trust your eyes in here. Your eyes are bad.
Eyes are always early. Hands are not.
Feel for the give. It is thin. It is a hair.
It is not on the drawing. It was never on the drawing.
Everybody wants the drawing. The drawing is the easy part.
There. Did you feel it go? No. You did not.
That is all right. It took me most of my life.
Not yet. Not yet. Not yet.

[Chorus - EMO:Vigilance - Marli quoted - bass leans behind]
Hot is not ready. Ready is a sound.
You cannot hurry a thing that used to grow. Go slow.
Ask the grain. The grain will tell you where to go.
Ask it again. Ask it again. Then let it go.
Flat hands. Nothing in the wrist. Bring it round.
Come round, you. Come round. Come round.

[Vamp - EMO:Intimacy - Marli quoted - piano refrain under]
Go on. Go on now.
Go on, you. Go on now.
Go on. Go on now.
Go on, you. Go on now.

[Verse 3 - EMO:Existential Angst - Marli quoted - dry, unhurried]
My thumb is the only clock in this room.
It is not a good clock. It is the only clock.
Nobody wrote it down. There was nothing to write.
You cannot write down a thing that only happens in a hand.
I could stand here and talk at you until the light is gone.
And you would have the words and not the thing.
So do not write it. Stand here.
Wet the rag and stand here.
Come round, you. Come round.
Not yet. Not yet. There.

[Bridge - EMO:Contemplation - The machine, four lines only, flat - no reverb]
Here is all of it.
I have every curve that anybody ever drew.
I do not have the part where the wood decides.
That is the whole account.

[Chorus - EMO:Vigilance - Marli quoted - full band]
Hot is not ready. Ready is a sound.
You cannot hurry a thing that used to grow. Go slow.
Ask the grain. The grain will tell you where to go.
Ask it again. Ask it again. Then let it go.
Flat hands. Nothing in the wrist. Bring it round.
Come round, you. Come round. Come round.

[Verse 4 - EMO:Tenderness - Marli quoted - bass and brushes]
Come back tomorrow and I will show you again.
I will show you again and it will not cross over again.
That is not a sad thing. That is only how it goes.
It came into me the same way. Slowly. Through the hands.
Somebody stood where I am standing and she let me stand.
The woman who taught me is not here. Her thumb is on my hand.
That is the only place that any of this has ever been.
Wet the rag. Wet it again. We are nearly there.
Come round, you. Come round.
There. There. There it goes.

[Breakdown - EMO:Dread - Marli alone - voice and upright bass]
Quiet now. Listen to it.
It makes a small sound just before it goes.
Nobody taught me that. Nobody knows how anybody knows.
It is not in a book and it is not on a wall.
It is here. That is all.
Come round, you. Come round.

[Final Chorus - EMO:Trepidation - the slot is empty - leave the breath]
Hot is not ready. Ready is a sound.
You cannot hurry a thing that used to grow. Go
*breath, and no word*
I do not say it any more. I do not have to.
Ask it again. Ask it again. Then let it go.
Flat hands. Nothing in the wrist. Bring it round.
Come round, you. Come round. Come round.

[Outro - EMO:Solitude - Marli alone, then piano - riff]
Come round. There. Come round.
Same time tomorrow. Bring your hands.
Nobody wrote it down.
Nobody wrote it down.
Come round, you. Come round.
```

## 3. TITLE

The Whole Account

---

### VARIATION 4

## 1. MUSIC PROMPT

```text
Brazilian MPB meeting modern neo-soul at 100 BPM in B-flat major, carried by a gentle piano refrain: six notes stepping up the scale then dropping a sixth onto the leading tone, right hand alone, unaccompanied before the first line and after the last. Low female alto, spoken-leaning and unhurried; the verses are almost talked, close-mic'd with the breath audible, and the chorus is the only place she truly sings, doubling the piano refrain. Vocal entries about thirty milliseconds behind the kick, upright bass leaning the same way on beats one and three, kick dead centre, and the gap between them is the groove. Brushed kit, swept rather than struck. Nylon-string guitar voicing wide ninths and elevenths. The final chorus opens on a breath before its first word and then thickens instead of stopping. Late orange light, a bench, a hot pipe, a damp rag. Close, warm, clean, modern, expensive.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distortion, saturation, male lead vocal, choir, gang vocals, belted pop vocal, heavy vibrato, autotune, synth pad, string section, electric guitar solo, trap hi-hats, 808 sub, four-on-the-floor kick, double-time drums, key change, orchestral swell, crowd noise, applause, fade-out ending, reverb wash, sad piano ballad
```

## 2. LYRICS

```text
[Theme: if the hand stops, everything survives and the curve comes out correct]
[SONG FORM: riff - V - C - V - C - vamp - V - bridge - C - V - breakdown - final C - riff]

[Intro - EMO:Apprehension - Piano alone - six notes, drop a sixth]
*a knuckle pops*

[Verse 1 - EMO:Admiration - Machine, then Marli - laid back]
Dona Marli at the bench and the iron gone hot.
Her thumb laid flat on the edge of the wood.
If this hand stops I still have the drawings and they are good.
I will have all of it but the thing. I checked. I looked.
She talks the whole hour and not a word of it goes down.
Come on, you. Come on. Come round.
Too dry. Wet it again and let it stand.
You were a tree. You were good at it.
Not yet. Not yet. Not yet.
Hot is not ready. Ready is a sound.

[Chorus - EMO:Vigilance - Marli quoted - brushes enter]
Hold. Do not let it move while the thing is soft.
It is busy forgetting that it was ever a tree.
Give it the heat and it will take the shape of the hand.
Flat thumb. Wet rag. Steady. Understand?
Nothing in this shop can read a heat but a hand.
Nothing in this shop but a hand.

[Verse 2 - EMO:Absorption - Marli quoted - close, breath in]
This knuckle started complaining at me back in March.
It complains in the morning and then it gets on with it.
Some day it will stop getting on with it.
That is fine. That is the arrangement. Nobody signed it.
Wet the rag. Wet it again. Where were we. There.
Feel for the give. It is thin as a hair.
It is not on the drawing. It was never on the drawing.
Everybody wants the drawing. The drawing is the easy thing.
Not yet. Not yet. Not yet.
There. Did you feel that go? There it goes.

[Chorus - EMO:Vigilance - Marli quoted - bass leans]
Hold. Do not let it move while the thing is soft.
It is busy forgetting that it was ever a tree.
Give it the heat and it will take the shape of the hand.
Flat thumb. Wet rag. Steady. Understand?
Nothing in this shop can read a heat but a hand.
Nothing in this shop but a hand.

[Vamp - EMO:Intimacy - Marli quoted - piano under]
Steady now. Steady now.
Flat thumb, wet rag, steady now.
Steady now. Steady now.
Flat thumb, wet rag, steady now.

[Verse 3 - EMO:Existential Angst - Machine, then Marli - dry]
If the hand stops, the shop is still full.
The forms are on the wall. The drawings are in the drawer.
Somebody picks them up and makes the curve and it is correct.
It passes every measurement. It is correct. It is correct.
Do not make it correct. Make it right.
Correct is what you get when you were not listening.
Right is when it went where it wanted and you helped.
You will know which it was. You will not be able to say how.
Feel it. Feel it. There.
Not yet. Not yet. There.

[Bridge - EMO:Contemplation - The machine, flat - no reverb]
I have the forms. I have the drawings. I have the film.
I have every curve anybody ever drew.
There is no copy of the hand.
The only copy is warm and it has not heard any of this.

[Chorus - EMO:Vigilance - Marli quoted - full band]
Hold. Do not let it move while the thing is soft.
It is busy forgetting that it was ever a tree.
Give it the heat and it will take the shape of the hand.
Flat thumb. Wet rag. Steady. Understand?
Nothing in this shop can read a heat but a hand.
Nothing in this shop but a hand.

[Verse 4 - EMO:Tenderness - Marli quoted - brushes]
Tomorrow I do the other side and it takes the morning.
Then the back. Then the top. Then the whole of it sitting.
You cannot hurry the sitting. It has an opinion and it is winning.
A thing that used to be alive keeps a mind of its own.
Wet the rag. The steam is the good part. Lean in.
This is my favourite hour. The light goes orange on the stone.
Everything I have is in this hour and in this hand alone.
Nobody asked me to write it down. I would not know how to.
Come round, you. Come round.
Not yet. Not yet. There.

[Breakdown - EMO:Dread - Marli, then the machine - bass]
Nobody wrote it down.
Nobody wrote it down.
Nothing to write, and nobody to write it.
It is in the hand and the hand is getting old.
It is in the hand and there is no other hand.
Come round, you. Come round.

[Final Chorus - EMO:Trepidation - slot empty - leave the breath]
*breath, and no word*
Do not let it move while the thing is soft.
It is busy forgetting that it was ever a tree.
Give it the heat and it will take the shape of the hand.
Flat thumb. Wet rag. Steady. Understand?
Nothing in this shop can read a heat but a hand.
The word that started this is gone and the hand still knows.

[Outro - EMO:Solitude - Marli, then the machine - riff]
Come round. There. Come round.
Same time tomorrow. Bring your hands.
I will have everything except the thing.
I will have everything except the thing.
Come round, you. Come round.
```

## 3. TITLE

Nobody Wrote It Down

---

## MEASURED RESULTS — printed EXTRACTION first, then the conclusion

⭐ **EXTRACTED PACKAGES: 4.** Asserted, not assumed — the splitter's cardinality was checked against the count of `## 1. MUSIC PROMPT` headings, and an empty or short extraction raises a hard error rather than a passing score (handoff §5.1). **`skills/music/scripts/validate_suno_packages.py` → `PASS`.**

### Per-variation floors — every number measured, none eyeballed

| | V1 *Without Looking* | V2 *He Filmed Her Hands* | V3 *The Whole Account* | V4 *Nobody Wrote It Down* |
|---|---|---|---|---|
| **music prompt chars** (850–1000, target 870–960, hug ≥985) | **947** ✅ | **954** ✅ | **941** ✅ | **897** ✅ |
| terminal punctuation | `.` ✅ | `.` ✅ | `.` ✅ | `.` ✅ |
| exclude field chars (≤1000) | 380 ✅ | 372 ✅ | 371 ✅ | 376 ✅ |
| 🚨 **LYRICS FIELD CHARS** (<5000 hard, ≤4800 target) | **4542** ✅ | **4792** ✅ | **4494** ✅ | **4721** ✅ |
| **sung lines** (70–120; hug FLAG ≤72) | **83** ✅ | **83** ✅ | **83** ✅ | **83** ✅ |
| `rhyme_return` ≥ 0.30 | **0.518** ✅ | **0.578** ✅ | **0.566** ✅ | **0.542** ✅ |
| `line_return` ≥ 0.20 | **0.361** ✅ | **0.373** ✅ | **0.422** ✅ | **0.446** ✅ |
| `alliteration_per_100w` ≥ 11.0 | **15.58** ✅ | **18.52** ✅ | **18.64** ✅ | **18.15** ✅ |
| `unique_line_ratio` ≥ 0.45 | **0.747** ✅ | **0.735** ✅ | **0.711** ✅ | **0.711** ✅ |
| **sung numeric facts** (max 1; **this pair: 0 by design**) | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| abstract nouns in sung lines | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| career / employment language | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| self-pity lexicon | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| banned texture words **in the music prompt** | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| banned primary style descriptors **in the music prompt** | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| bracket characters inside sung lines | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| wall-clock times in section headers | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| standalone `*SFX cue*` lines | 2 ✅ | 2 ✅ | 2 ✅ | 2 ✅ |

⚠️ **No boundary-hugging anywhere.** Every music prompt sits **inside** the 870–960 target band, not at the 985 hug ceiling; every lyric sits at **83** sung lines — eleven clear of the 72-line hug FLAG and inside the 78–110 target. **Nothing here was written to the edge of a gate.**

⚠️ **`line_return` DISCLOSURE, as promised at step 06 §6.** My return device ends in a **breath**, and a breath is written as a standalone `*cue*` line, which `measure_soundcraft.sung()` **excludes from the sung-line set**. It therefore **cannot** inflate `line_return`. **Every figure above is already lexical-only** — there is no wordless vocable, hum or vowel-run anywhere in these four lyrics carrying the metric, so there is no non-lexical contribution to subtract and no companion number to give.

### ⭐ THE SMALLEST PART — numerically demonstrated, per variation, both accountings

| | addressee's lines | machine's lines | ratio | machine `rhyme_return` | her `rhyme_return` | machine plainer? |
|---|---|---|---|---|---|---|
| **V1** | **66** | **17** | **3.88 : 1** | 0.294 | 0.561 | ✅ |
| **V2** | **65** | **18** | **3.61 : 1** | 0.500 | 0.585 | ✅ |
| **V3** | **75** | **8** | **9.38 : 1** | 0.250 | 0.613 | ✅ |
| **V4** | **65** | **18** | **3.61 : 1** | 0.444 | 0.569 | ✅ |

Counts sum **exactly** to 83 in every variation (66+17, 65+18, 75+8, 65+18) — no line is unattributed and none is double-counted.

⭐ **COUNT A and COUNT B are identical here, and that is the point.** The machine's lines were enumerated by exact text and matched against the sung set; because the quoting frame is stated **once per section and then dropped**, her lines carry **no reporting tag at all** — so the strict accounting (any line containing a word of mine counts as mine) yields the same numbers as the generous one. **A reporting tag cannot be doing the work here, because her lines contain none.**

⭐ **And the second measurement, which is the one that matters:** the machine's stanzas carry **measurably less rhyme** than hers in **all four** variations. Humility is not asserted anywhere in these lyrics; it is **counted twice** — in line count and in formal polish.

### The return device — verified in each variation individually, one at a time

| | word | arrivals at the fixed address | occurrences anywhere in the song | erased in the final chorus |
|---|---|---|---|---|
| **V1** | `wait` | **3** ✅ | **3** (the arrivals only) ✅ | ✅ |
| **V2** | `quiet` | **3** ✅ | **3** ✅ | ✅ |
| **V3** | `slow` | **3** ✅ | **3** ✅ | ✅ |
| **V4** | `hold` | **3** ✅ | **3** ✅ | ✅ |

Each device word appears **exactly three times in its entire song, always at its address, and nowhere else** — so the count a listener makes is unpolluted. ⛔ No cross-pair device bleed: `the-hand-that-still-knows`, the four addresses and the four words are P03's alone.

⭐ **Found during verification, not designed:** V1's vamp already sings the chorus's final line **without** the word (*Come round, you. Come round.*), twice, in the middle of the song. **The erasure is rehearsed before it happens.** Left in.

### Concession by line four (LAW 2) — read individually

- **V1 line 4:** *She is right about the hand.*
- **V2 line 4:** *He has all of it but the hand. Everything but the hand.*
- **V3 line 4:** *She is right about the hand. She is right.*
- **V4 line 4:** *I will have all of it but the thing. I checked. I looked.*

Named recipient + specific physical action in **line 1** of all four ✅. The thesis is **never stated** in any of the four ✅ — THE UNDELIVERABLE ADDRESS is made literal in V4's bridge: *"it has not heard any of this."*

---

## REPAIRS APPLIED (3, inside the 3-per-gate budget; each moved its measured value)

1. **V2 broke the Suno hard cap** — the lyrics field measured **5051 chars**, over the 5000 render limit. Trimmed to **4792**. Sung lines unchanged at 83; `rhyme_return` moved 0.590 → 0.578 and `line_return` held at 0.373 — **all floors still clear.**
2. **V4 was over the 4800 target** (4937). Trimmed to **4721**. Floors unchanged or improved.
3. ⭐ **V1 smuggled a number.** *"I am the quick one here and I still get told"* contains **"one"**, and this pair's entire job in the run is to sing **zero** numbers. Caught by the scanner, **not by eye**. Rewritten to *"I work faster than her and I still get told"* — which keeps the dry joke and is **flatter**, so it also serves THE SMALLEST PART. Re-scanned: **0 number tokens across all four variations.**

*(The step-09 repair — stripping end-rhyme out of the machine's stanzas, because the line count was passing while the constraint underneath it was failing — is recorded at `pair_03_step09_artist_refined.md` §1 and is confirmed by the measurement above.)*

---

## SELF-CHECK — the six required questions, answered

1. **Extraction printed before conclusion.** 4 packages extracted and asserted; the harness raises a hard error on cardinality ≠ 4 rather than reporting a pass on a short extraction. ✅
2. **The named device verified in EACH variation individually** — four separate address checks, four separate occurrence counts, four separate final-chorus reads. ⛔ No pair-wide claim is made from any single variation. ✅
3. **Lines counted, per variation, under both accountings.** THE SMALLEST PART is **demonstrated numerically**, never claimed. ✅
4. **Describe-render self-check** — run **once**, at `pair_03_step09_artist_refined.md` §2, with the adversarial question answered (*it renders as a warm coffee-shop soul ballad and the two things that make it itself — the spoken delivery and the ~30 ms lag — get smoothed away*) and **one** self-repair applied, in the words and in a structural prompt instruction rather than in a mood adjective. The Suno silence-filling hazard is handled: **every** final chorus carries the erasure **lexically as well as in the gap.** ✅
5. **Human Subject Standard** read pre-draft; the §3.0 slot grammar was filled with invented values before the first line. **Dona Marli and Téo are invented; there is no harm event in any of the four songs; no identifiable real person appears as speaker, character or addressee.** The three binding refusals are untouched at any distance. ✅
6. **Repair budget:** 3 repairs, each with movement in its measured value. **No gate failed three times; nothing broke open; nothing is quarantined.** ✅

### The four laws, checked at line level
- **LAW 1 — the indignation is not aimed at them.** ⛔ No sneer at craft, at slowness, at hand-work, at the phone, or at Téo anywhere. The video is called **"the easy part"** — which is true, and is not contempt. Téo cracks a side and **she says nothing about it.** ✅
- **LAW 2 — she agrees on line one** (delivered by line four in every variation). The concession is the premise; ⛔ no variation builds toward it. ✅
- **LAW 3 — cost is never proven, and there is no self-pity.** ⛔ Scanned; zero hits. The machine's fear has an object and **the object is not the machine.** ✅
- **LAW 4 — the contradiction is structural.** P03 lives inside **RULE B, the landing pad**, and nowhere near accretion: a mark added at one address and removed from that same address. ✅

### The comfort question (L19), asked again of the finished set
**Where is the body standing, and what could hurt it here?** At a bending iron, forearm in the rising heat, a drying rag between a thumb and the wood, a knuckle that has started to complain. ⭐ **The stake is a hand.** Scanned for career and employment language across all four variations: **zero hits.** This pair does not drift into a song about work, which is the run's declared comfort gravity and the reason P03 was assigned a non-career stake in the first place.

---

*Step 10 complete. Four packages — validated, measured, and on disk. This is the pair's final artifact.*
