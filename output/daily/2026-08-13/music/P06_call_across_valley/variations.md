# VARIATIONS — P06 · THE CALL ACROSS THE VALLEY
**Four songs derived from THIS pair's concept. ⛔ No global template.**
Every package below is the exact shipped text, and every number is measured by
`scripts/measure_soundcraft.py → profile()` on that exact text.

---

## THE FOUR ANGLES — what actually differs

| | **V1 Kom Hem, Silja** | **V2 The Older Throat** | **V3 The Burr** | **V4 The Air Lies Low At Dusk** |
|---|---|---|---|---|
| **Whose body** | the caller, on her rock | the woman who answers, in her hut | a teacher and a student on cold rock | the valley itself, at the hour it starts carrying |
| ⭐ **Second eye — reflected medium** | **milk in a pail, ringing** | **a loose windowpane, buzzing** | **her breath, white in cold air** | **a line scratched in frost on a door** |
| **How it refuses to resolve** | neither substance can become the other | the reflection is touchable, the call is not | the breath dies at her collar, the call is already at the birches | the drawn line ends at her finger, the call ends at another woman |
| **The count** | by **name** of each cow | by **name** of every caller answered over the years | by **landmark** the attempt reached | by **condition** met before sound will carry |
| **Gang chant** | `Kom hem, Silja!` | `I hear you, Marit!` | `More rough, more road!` | `Lie low, cold air!` |
| **Register span** | 4 octaves (A5 / A1, 880 / 55 Hz) | 5 octaves (≈1100 Hz / D1 37 Hz) | 4 octaves (G5 / G1, 784 / 49 Hz) | 5 octaves (≈1300 Hz / E1 41 Hz) |
| **Tempo** | 66 BPM | 58 BPM | 72 BPM | 52 BPM |
| **SOURCE-2 verbatim used** | funky bass · slapping/popping · swagger | funky bass · new jack swing | slapping/popping · swagger | funky bass · slow jam |
| **Where "eighty years" sits** | eighty years off this same stone | eighty years of throats over this water | eighty years of women on this rock | eighty years of asking how it goes |

**Within-set distinctiveness, measured** (portfolio ceilings: lyric 0.42, prompt 0.58, 5-gram Jaccard 0.18):

| pair | lyric sim | prompt sim | 5-gram Jaccard |
|---|---|---|---|
| V1–V2 | 0.022 | 0.064 | 0.002 |
| V1–V3 | 0.011 | 0.072 | 0.002 |
| V1–V4 | 0.021 | 0.067 | 0.000 |
| V2–V3 | 0.018 | 0.081 | 0.004 |
| V2–V4 | 0.025 | 0.075 | 0.006 |
| V3–V4 | 0.016 | 0.127 | 0.004 |
| **WORST** | **0.025** | **0.127** | **0.006** |

⚠️ *Honest caveat: Python's `SequenceMatcher` applies autojunk on long strings and deflates the ratio, so
the two similarity columns read low by construction. **The 5-gram Jaccard is the trustworthy number** —
0.006 worst case against a 0.18 ceiling. These four share a bed and share nothing else.*

---
---

# V1 — **KOM HEM, SILJA**
*The caller. The direct statement of the figure, and the milk that shows it to her.*

### MUSIC PROMPT — **955 chars**, terminal punctuation ✅, SOURCE-2 verbatim ×3 ✅
```
A Swedish herding call held over a body-felt floor; the four-octave gap between them is the arrangement. Female vocalist, late twenties: open-throated kulning, unvibratoed head tone around eight hundred and eighty hertz, neutral third, a fourth pitched slightly high, free time, no metre, breath audible. Under her a superhumanly funky bass parked on the same note four octaves down at fifty-five hertz, sixty-six BPM felt not counted, furiously slapping and popping at the floor of hearing so it lands in the sternum instead of the ear, mono and centred. Signature 1: THE SCOOPED VALLEY, two hundred to six hundred hertz cut almost to silence so the mids are open ground the call has to cross, inhabited only when a distant second woman answers from the far side. Signature 2: THE MILK RING, a close-miked pail of milk trembling at forty hertz, the sub made visible. Gang chant with swagger, humor, and sheer danceability. No reverb tail reading as echo.
```

### EXCLUDE
```
male lead vocals, child vocals, vibrato, pop belting, trap hi-hats, EDM risers, four-on-the-floor kick, guitar, brass, orchestral strings, choir pads, glitch, bitcrush, chiptune, breakcore, distortion, autotune, whispered ASMR, long reverb tails, slapback echo on the lead, stereo-widened bass, untuned drone, sampled field recording of a real singer
```

### LYRIC TEXT — see `step08_lyrics_draft.md` §1 for the full shipped text (identical; kept there as the single source of truth, and readable by `measure_soundcraft.py` at that path).

### MEASURED
| Gate | Band | V1 | |
|---|---|---|---|
| lyric chars | ≤5000 | 3848 | PASS |
| sung lines | 70–120 (t 78–110) | **108** | PASS |
| end-rhyme | ≥0.30 | **0.843** | PASS |
| line return | ≥0.20 | **0.324** | PASS |
| words/line | ≤7.5 | **4.62** | PASS |
| allit/100w | ≥11.0 | **15.83** | PASS |
| sung numerals | `eighty` only | **`['eighty']`** | PASS |
| L41 screen | clean | **clean** | PASS |

---
---

# V2 — **THE OLDER THROAT**
*The far side. She answers short because short is enough — and the answering costs her nothing she is not
willing to spend. The reflection is a pane of glass she can put her thumb on.*

### MUSIC PROMPT — **954 chars**, terminal punctuation ✅, SOURCE-2 verbatim ×2 ✅
```
An old woman's answer, sent back across water, over a floor you feel in the ribs. Female vocalist, sixty-ish: kulning technique but rationed, short phrases, lower tessitura, no vibrato, a neutral third, breath taken audibly before each one. The younger caller she answers arrives from far off, thin and high near eleven hundred hertz, dry, carrying no reverb of her own. Beneath both a superhumanly funky bass on D one at thirty-seven hertz, mono and centred, five octaves under the far call, fifty-eight BPM. Its articulation carries a splash of vintage new jack swing, a sixteenth-note push and shuffle set against a voice that keeps no metre at all. Signature 1: THE BUZZING PANE, a loose window-glass resonance near ninety hertz rattling in sympathy each time the sub moves, close-miked and dry, the only near object in the mix. Signature 2: THE RATIONED ANSWER, every returning phrase shorter than the phrase it answers. Cut the mids to open ground.
```

### EXCLUDE
```
male lead vocals, child vocals, vibrato, pop belting, young pop tone, trap hi-hats, EDM risers, guitar, brass, strings, choir pads, glitch, bitcrush, breakcore, distortion, autotune, long reverb tails, slapback echo, stereo-widened bass, untuned drone, new age pad, sampled field recording of a real singer
```

## LYRICS — V2 "The Older Throat"
```
[Theme: An older woman on the far mountain answers a younger caller across the water. Her answer is short because short is enough.]
[SONG FORM: one long call in, one short answer out. The answer is never the length of the call.]

[Intro - The Window - EMO:Awe - Female, lower and older - Room tone, sub only, no call]
*a hut, a stove, a loose pane; wind*
Something is coming over the water.
(Over the water.)
The window knows before I do.
(Before I do.)

[Answer I - EMO:Awe - Female - Sub full, the far call high and thin]
She is calling and she is calling hard.
(Calling hard.)
I hear the rasp before I hear the tune.
(Before the tune.)
Marit, that is a rough thing. Good.
(That is good.)
Rough is what got it over here.
(Got it over.)
I will not spend what she has spent.
(Will not spend.)
Mine is older and mine is short.
(Mine is short.)

[Chorus - EMO:Awe - Female + gang - Sub forward, the answer sent]
I hear you, Marit!
(I hear you, Marit!)
I hear you, Marit!
(I hear you, Marit!)
Send it slant and send it strong.
(Send it strong!)
Short is not the same as wrong.
(I hear you, Marit!)

[Answer II - EMO:Awe - Female - The far call keeps coming; mine goes back shorter]
She sends a whole valley of it.
(A whole valley.)
I send back a breath with a bend in it.
(With a bend.)
That is the trade. That has always been the trade.
(Always the trade.)
She has the lungs and I have the years.
(I have the years.)

[The Pane - EMO:Awe - Female - THE CALL IS REMOVED; sub alone; the loose glass buzzing]
*the far voice stops; the pane*
I cannot see a voice come over water.
(Cannot see it.)
But the pane in the west wall is buzzing.
(The pane is buzzing.)
That is her, in glass, at my elbow.
(In glass.)
Same shape. Flatter. Pressed into a pane.
(Pressed into a pane.)
The buzzing stops at the putty.
(At the putty.)
Her call is still out over the water.
(Out over the water.)
I can put my thumb on the glass and feel her.
(Feel her.)
I cannot put my thumb on the valley.
(Not on the valley.)

[The Count - EMO:Awe to ache - Female - She counts the callers she has answered, by name]
Ingrid.
(Answered.)
Sigrid.
(Answered.)
Britta.
(Answered.)
Marit.
(Answering.)
Eighty years of throats over this water.
(Over this water.)
The how of it is still up the valley.
(Still up the valley.)
I do not need the how. I need her home.
(Need her home.)

[Chorus - EMO:Awe - Female + gang - Identical return, harder]
I hear you, Marit!
(I hear you, Marit!)
I hear you, Marit!
(I hear you, Marit!)
Send it slant and send it strong.
(Send it strong!)
Short is not the same as wrong.
(I hear you, Marit!)

[The Dirt - EMO:Awe - Female - Bare, sub swelling]
They tell the young to sing it sweet.
(Sing it sweet.)
Sweet sinks in the sedge like a stone.
(Like a stone.)
The burr is the boat. The burr is the bridge.
(The burr is the bridge.)
Send the dirt. The dirt is what swims.
(The dirt swims.)

[Loop Until It Hurts - EMO:Awe strained - Female + gang - Answers repeating and shortening]
I hear you, Marit!
(I hear you, Marit!)
I hear you, Marit!
(I hear you, Marit!)
Shorter. Shorter. Still enough.
(Still enough.)
I hear you, Marit!
(I hear you, Marit!)
I will answer until the answer is a breath.
(Until it is a breath.)
I hear you, Marit!
(I hear you, Marit!)

[Outro - EMO:Awe - Sub decaying; the pane settling]
*the glass goes still*
Her herd is turning. I can feel it in the wall.
(In the wall.)
She is a long way off and she is coming.
(She is coming.)
I hear you, Marit!
(I hear you, Marit!)
*one long breath, no note*
```

### MEASURED
| Gate | Band | V2 | |
|---|---|---|---|
| lyric chars | ≤5000 | 3514 | PASS |
| sung lines | 70–120 (t 78–110) | **96** | PASS |
| end-rhyme | ≥0.30 | **0.865** | PASS |
| line return | ≥0.20 | **0.302** | PASS |
| words/line | ≤7.5 | **4.94** | PASS |
| allit/100w | ≥11.0 | **14.56** | PASS |
| sung numerals | `eighty` only | **`['eighty']`** | PASS |
| L41 screen | clean | **clean** | PASS |

⭐ **Note on the second eye here:** the non-resolution is inverted relative to V1. *I can put my thumb on
the glass and feel her. / I cannot put my thumb on the valley.* The reflection is the only touchable
version of a person who is four hundred metres away — which is a harder and kinder statement of the same
refusal, and it is the sapphic reading the pair earns without ever declaring it.

---
---

# V3 — **THE BURR**
*The technique, taught. "Loop until it hurts" is not a production note here — it is the lesson. The
reflection is her own breath, the only part of the call she is allowed to watch.*

### MUSIC PROMPT — **957 chars**, terminal punctuation ✅, SOURCE-2 verbatim ×2 ✅
```
A lesson in a herding call, taught on cold rock, over a frequency felt in the sternum. Two female voices: an older teacher, low and dry, and a student in her twenties learning open-throated kulning, unvibratoed head tone near seven hundred and eighty hertz, neutral third, a fourth pitched high, free time. The student's tone starts thin and gets audibly rougher and louder across the song until the top of it frays. Under them a mono sub on G one at forty-nine hertz, seventy-two BPM, furiously slapping and popping at the floor of hearing so each attack lands as impact, not pitch. The four-syllable gang chant is delivered with swagger, humor, and sheer danceability. Signature 1: THE LANDMARK ANSWER, a bell or a bird placed at a measurable distance and reverb-matched to it, so each attempt returns from further away. Signature 2: THE FRAYING TOP, deliberate breakage at the peak of the final calls. Cut two hundred to six hundred hertz to open ground.
```

### EXCLUDE
```
male lead vocals, child vocals, choirboy tone, vibrato, pop belting, trap hi-hats, EDM risers, guitar, brass, strings, glitch, bitcrush, breakcore, distortion, autotune, pitch correction on the fraying notes, long reverb tails, slapback echo, stereo-widened bass, untuned drone, sampled field recording of a real singer
```

## LYRICS — V3 "The Burr"
```
[Theme: An old herder teaches a young woman the call on a cold rock. The lesson is: do not sand it smooth. The lesson costs a throat.]
[SONG FORM: teacher calls, student answers shorter; landmarks come back as the answers.]

[Intro - The Lesson - EMO:Awe - Two female voices - No bass, boots on rock]
*cold morning; wind; a bell far below*
Stand here. Put your feet apart.
(Feet apart.)
Now open. Do not be pretty.
(Not pretty.)

[Lesson I - EMO:Awe - Female - Sub enters at fifty-five hertz]
That went to the fence.
(To the fence.)
Rounder. Rougher. Ride the ragged part.
(Ride the ragged part.)
That went to the birches.
(To the birches.)
Do not save your throat. Spend your throat.
(Spend your throat.)
That went to the water.
(To the water.)
Now put the gravel in and go again.
(Go again.)

[Chorus - EMO:Awe - Female + gang - Full sub, gang enters]
More rough, more road!
(More rough, more road!)
More rough, more road!
(More rough, more road!)
Sweet stays home in the yard.
(In the yard!)
The scarred note travels hard.
(More rough, more road!)

[Lesson II - EMO:Awe - Female - Student calling, teacher answering shorter]
My mother made her living on this note.
(On this note.)
Her mother made a winter out of it.
(Made a winter.)
They did not have a pretty year between them.
(Not a pretty year.)
They had the herd home every single night.
(Home every night.)

[The Breath - EMO:Awe - Female - THE CALL IS REMOVED; sub alone; close-miked breathing in cold air]
*the calls stop; breathing*
You cannot watch a call go.
(Cannot watch it.)
Watch your breath instead.
(Watch your breath.)
There it is, white, at the end of your face.
(There it is.)
Same push. Slower. Made of water.
(Made of water.)
Your breath dies at your collar.
(At your collar.)
Your call is at the birches by then.
(At the birches.)
Breath is the part you get to see.
(You get to see.)
The rest of it is out there working.
(Out there working.)

[The Ridge - EMO:Awe to awe-terror - Female - Student calls; the hill answers]
Again.
(Again.)
That went to the fence.
(The fence.)
That went to the birches.
(The birches.)
That went to the water.
(The water.)
That went over the ridge.
*a bell, far, on the other side*
Eighty years of women on this rock.
(On this rock.)
The how of it is still up there somewhere.
(Still up there.)
It works. That is the whole of what we know.
(That is the whole.)
Your throat will hurt tonight. Good.
(Good.)

[Chorus - EMO:Awe - Female + gang - Identical return, harder]
More rough, more road!
(More rough, more road!)
More rough, more road!
(More rough, more road!)
Sweet stays home in the yard.
(In the yard!)
The scarred note travels hard.
(More rough, more road!)

[The Burr - EMO:Awe - Female - Bare, sub swelling]
They will tell you the burr is a fault.
(Is a fault.)
The burr is the reason the birds move.
(The birds move.)
Smooth is for singing in a room.
(In a room.)
We are not singing. We are reaching.
(We are reaching.)

[Loop Until It Hurts - EMO:Awe strained - Two female voices + gang - Repetition as training]
More rough, more road!
(More rough, more road!)
More rough, more road!
(More rough, more road!)
Again. Again. Do not sand it.
(Do not sand it.)
More rough, more road!
(More rough, more road!)
My throat is raw and my throat is right.
(Raw and right.)
More rough, more road!
(More rough, more road!)

[Outro - EMO:Awe - Sub decaying; one far bell]
*a bell across the valley, no closer*
They are turning.
(Turning.)
They are a long way down and they are turning.
(They are turning.)
More rough, more road!
(More rough, more road!)
```

### MEASURED
| Gate | Band | V3 | |
|---|---|---|---|
| lyric chars | ≤5000 | 3560 | PASS |
| sung lines | 70–120 (t 78–110) | **99** | PASS |
| end-rhyme | ≥0.30 | **0.848** | PASS |
| line return | ≥0.20 | **0.364** | PASS |
| words/line | ≤7.5 | **4.64** | PASS |
| allit/100w | ≥11.0 | **22.88** | PASS |
| sung numerals | `eighty` only | **`['eighty']`** | PASS |
| L41 screen | clean | **clean** | PASS |

⭐ **`More rough, more road!` is the set's most shoutable return** — four syllables, two alliterating
pairs, no language barrier, and it states the thesis rather than decorating it. It is also why V3's
alliteration measures 22.88: an alliterative chant pays back on every repeat.

---
---

# V4 — **THE AIR LIES LOW AT DUSK**
*The carrier itself. At dusk the cold settles under the warm and the seam between them bends a voice
downward — which is why the call goes further than it should. Slowest, sparsest, most somatic.*

### MUSIC PROMPT — **908 chars**, terminal punctuation ✅, SOURCE-2 verbatim ×2 ✅
```
Dusk in a mountain valley, one voice and a floor, everything slow. Female vocalist, thirties: open-throated kulning, unvibratoed, sparse, for most of the song only fragments, with a single full call near the end at thirteen hundred hertz. Under her a superhumanly funky bass on E one at forty-one hertz, mono and dead centre, five octaves down, moving with the unhurried physicality of a downright salacious slow jam at fifty-two BPM, beneath a voice that is working and not seducing. Signature 1: THE INVERSION LAYER, a slow filtered swell closing over the top of the mix like cold air settling at dusk, so the whole picture goes flatter and further away as the song continues. Signature 2: THE FROST LINE, a fingernail-on-ice texture, tiny and dry and very close, the only near thing in a mix that is otherwise all distance. Cut two hundred to six hundred hertz almost to silence and leave the gap audible.
```

### EXCLUDE
```
male lead vocals, child vocals, vibrato, pop belting, trap hi-hats, EDM risers, four-on-the-floor kick, guitar, brass, strings, choir pads, glitch, bitcrush, breakcore, distortion, autotune, long reverb tails, slapback echo, stereo-widened bass, untuned drone, ambient new age wash, sampled field recording of a real singer
```

## LYRICS — V4 "The Air Lies Low At Dusk"
```
[Theme: Dusk in a mountain valley. The cold air settles in layers, and the layers bend a voice downward, so a call carries further than it has any right to.]
[SONG FORM: conditions counted first, then the call. Everything is long except the answer.]

[Intro - EMO:Awe - Female - Sub alone; no call yet]
*evening; the grass goes wet; smoke lies flat*
Wait.
(Wait.)
It will not carry yet.
(Not yet.)

[Conditions - EMO:Awe - Female - Sub only, spoken-sung, close-miked]
When the grass goes wet.
(Goes wet.)
When the smoke lies flat along the byre.
(Lies flat.)
When the water stops moving.
(Stops moving.)
When the cold comes down and lies on the warm.
(Lies on the warm.)
Then the air is a floor.
(Is a floor.)
Then a voice can walk on it.
(Can walk.)

[Chorus - EMO:Awe - Female + gang - Sub full, gang enters]
Lie low, cold air!
(Lie low, cold air!)
Lie low, cold air!
(Lie low, cold air!)
Lie low and let it long.
(Let it long!)
Carry the crooked song.
(Lie low, cold air!)

[The Layers - EMO:Awe - Female - Sub articulating under a sparse call]
The warm goes up. The cold comes and covers it.
(Covers it.)
Between them there is a seam.
(There is a seam.)
Sound will not go straight through a seam.
(Not straight through.)
Sound leans along it like a hand on a wall.
(Like a hand.)
That lean is the whole of my reach.
(The whole reach.)

[The Door - EMO:Awe - Female - THE CALL IS REMOVED; sub alone; a fingernail on frost]
*frost on the byre door*
I cannot draw a sound.
(Cannot draw it.)
So I draw the valley.
(Draw the valley.)
Here is me. Here is her. Here is the gap.
(Here is the gap.)
Same shape. Shorter. Held in frost.
(Held in frost.)
My line stops where my finger stops.
(Where my finger stops.)
My call stops where she is standing.
(Where she is standing.)
The frost will be water by morning.
(Water by morning.)
The call will still be out there working.
(Still working.)

[The Count - EMO:Awe - Female - Conditions met, then the one call, then the number]
Grass, wet.
(Wet.)
Smoke, flat.
(Flat.)
Water, still.
(Still.)
Cold, down.
(Down.)
Now.
*a single long call, the only full call in the song*
Eighty years of asking how it goes.
(How it goes.)
The answer is out there walking home.
(Walking home.)
Meanwhile the herd hears me.
(The herd hears me.)
Meanwhile is enough.
(Enough.)

[Chorus - EMO:Awe - Female + gang - Identical return, wider]
Lie low, cold air!
(Lie low, cold air!)
Lie low, cold air!
(Lie low, cold air!)
Lie low and let it long.
(Let it long!)
Carry the crooked song.
(Lie low, cold air!)

[The Dirt In It - EMO:Awe - Female - Bare, sub swelling]
Dust in the light is how you see the light.
(How you see it.)
Grit in the air is how the air holds a note.
(How it holds.)
The clean part of me stayed in the yard.
(In the yard.)
The dirty part of me is over the water.
(Over the water.)

[Loop Until It Hurts - EMO:Awe - Female + gang - Chant slows and lengthens; sub swallows the room]
Lie low, cold air!
(Lie low, cold air!)
Lie low, cold air!
(Lie low, cold air!)
Stay down until she is in the yard.
(Until she is in the yard.)
Lie low, cold air!
(Lie low, cold air!)
Stay down until my throat gives out.
(Until it gives out.)
Lie low, cold air!
(Lie low, cold air!)

[Outro - EMO:Awe - Sub decaying under wind]
*the air warms a little; the carrying stops*
It is going.
(Going.)
The floor under the sound is going.
(Is going.)
She is over the water and she is coming.
(She is coming.)
Lie low, cold air!
(Lie low, cold air!)
```

### MEASURED
| Gate | Band | V4 | |
|---|---|---|---|
| lyric chars | ≤5000 | 3461 | PASS |
| sung lines | 70–120 (t 78–110) | **103** | PASS |
| end-rhyme | ≥0.30 | **0.864** | PASS |
| line return | ≥0.20 | **0.272** | PASS |
| words/line | ≤7.5 | **4.43** | PASS |
| allit/100w | ≥11.0 | **13.16** | PASS |
| sung numerals | `eighty` only | **`['eighty']`** | PASS |
| L41 screen | clean | **clean** | PASS |

⭐ **V4 carries the run's cleanest statement of the seed and of the assigned discharge in one breath:**
*Eighty years of asking how it goes. / The answer is out there walking home. / Meanwhile the herd hears
me. / Meanwhile is enough.* An unsolved problem restated as a distance, and a woman who does not need the
mechanism because the result already arrived.

---

## ⛔ WHOLE-SET BAN SWEEP

| Ban | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| `D4` invent the people | ✅ | ✅ | ✅ | ✅ |
| `D7` no AI song | ✅ | ✅ | ✅ | ✅ |
| `D9` bad thing not redeemed | ✅ debt unpaid | ✅ trade never fair | ✅ *your throat will hurt tonight* | ✅ the carrying stops |
| Qusra / Ryanair / USS Lincoln | absent | absent | absent | absent |
| Female vocals, no children | ✅ | ✅ | ✅ two adult women | ✅ |
| L41 `nobody`/`nothing` discharge | clean | clean | clean | clean |
| One sung numeral, in words | ✅ | ✅ | ✅ | ✅ |
| No credited name in a Suno field or lyric | ✅ | ✅ | ✅ | ✅ |
| Kulning = technique, no sacred repertoire | ✅ | ✅ | ✅ | ✅ |
