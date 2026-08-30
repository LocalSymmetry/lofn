# PAIR 02 — STEP 08 · RAW GENERATION
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P02 — THE TUB**

**Continuity Payload Used:** frozen ICB, LF-sha `9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a`, **142,900 B** · LOFN-PRIME DNA inlined **27,796 B** · 18 baseline seats, 3 Hyper-Skeptics at 6/12/18 · **15 Special Flairs** present.
**Step file:** `skills/music/steps/08_Generate_Music_Generation.md` · **Inputs:** `pair_02_step06_facets.md`, `pair_02_step07_guides.md`
**Binding override:** `06_music_handoff.md` §4. ⛔ The step file's stale `≤1500` / `≤1000` char caps and its bracketed `key:value` prompt spec are superseded by the **dense-paragraph 850–1000 mandate, mid-band target 870–960** (the step file says so itself in its own 2026-07 override header).

> ⚠️ **THESE ARE RAW DRAFTS AND ARE NOT GATE-BOUND.** The 850–1000 char band binds the **final** prompts at step 10. Raw prompt char counts below are **measured, not estimated** — an earlier draft of this file eyeballed them and three of the four were wrong by up to 52 chars. Corrected against `len()`. They are stated so the refinement at step 09 is auditable as a real operation rather than a relabelling.

**Cardinality reconciliation, declared:** the step file's phase map expects **six** prompts (one per guide). The run's fixed cardinality is **4 variations per pair** (`gates.yaml → total_prompts: 24` across 6 pairs). **The run wins.** The six guides are *rubrics applied to* four variations, not six songs: Guides 1/2/3/5 apply to all four; Guide 4 lands once per song at verse three; **Guide 6 is the wildcard and lands hardest on V3.**

---

## THE FOUR VARIATIONS — angles authored from THIS pair's concept

| | angle | the ONE object handled across four verses | title |
|---|---|---|---|
| **V1** | THE LAMINATED CARD | a laminated pass with a face on it | **The Face On The Card** |
| **V2** | THE KEY | a key on a cracked plastic fob | **A Key For A Sold Car** |
| **V3** | THE PHOTOGRAPH OF THE SAND | a photograph stuck to the tub's bottom ⭐ *strongest card* | **A Photograph Of Sand** |
| **V4** | THE LID | the tub itself; **nothing is taken out** 🛟 *safety variation* | **The Lid And The High Shelf** |

**Title law check (`00_research_brief.md` §4, measured):** all four name a **THING**, not an argument. No persona-brand prefix. Strong tokens present: `card`, `key`, `sand`, `lid`, `shelf`.

---

## 1. RAW MUSIC PROMPTS

**V1 — RAW (measured 730 chars).**
```
Composed and unhurried. Three-part close-harmony folk hymn sung by two women and one man with no lead voice at any point. A warm drawbar organ holds one chord underneath everything. A jangling triangle marks the offbeat. Claps and shakers arrive at the second verse. Tracked as a band in a kitchen with hard surfaces, one lamp and a fridge running. Seventy-two beats per minute in D major. One tempo, one key, no modulation. Four numbered verses with an identical refrain between each. The two inner voices sit a major second apart and beat against one another; in the third verse they swap parts mid-phrase and cross once, and neither line alters. The song grows by addition and ends on the refrain in an emptied room. No risers.
```

**V2 — RAW (measured 705 chars).**
```
Level and companionable. Three-part close-harmony folk hymn for two women and one man, all three carrying the tune together, no lead voice. A drawbar organ sits under the whole thing on one chord and never swells. Shakers and handclaps come in at the second verse and stay. A jangling triangle keeps the offbeat throughout. Recorded live in a kitchen with hard surfaces, a fridge and one lamp. Sixty-nine beats per minute in G major, one tempo, one key. Four numbered verses, identical refrain between each, no bridge. Two inner voices sit a major second apart and beat; at the third verse they swap parts mid-phrase and cross once without either line changing. Ends on the refrain, room empty. No risers.
```

**V3 — RAW (measured 669 chars).**
```
Plain and communal. Three-part close-harmony folk hymn, two women and one man singing as one thickened voice, no lead and no solo. A drawbar organ holds a single chord beneath the whole song. Handclaps and shakers enter at the second verse. A jangling triangle rings the offbeat. Tracked as a band in a kitchen with a fridge, a hard table and one lamp. Sixty-six beats per minute in F major, one tempo, one key, no modulation. Four numbered verses with an identical refrain between each. The two inner voices sit a major second apart and audibly beat; at the third verse they swap parts mid-phrase and cross once, and neither line moves. Ends on the refrain. No risers.
```

**V4 — RAW (measured 761 chars).**
```
Patient and level. Three-part close-harmony folk hymn for two women and one man singing together throughout, no lead voice. A drawbar organ holds one chord under everything. A jangling triangle marks the offbeat and claps and shakers join at the second verse. From the third verse an upright piano doubles the tune with a gentle piano refrain and stays to the end. Tracked as a band in a kitchen with hard surfaces, a fridge and one lamp. Sixty-three beats per minute in E-flat major, one tempo, one key. Four numbered verses and an identical refrain, no bridge. Two inner voices sit a major second apart and beat against each other; at the third verse they swap parts mid-phrase and cross once, neither line changing. Ends on the piano and the room. No risers.
```

**What is missing from the raw set, and is added at step 09:** the **one-room posture in the concrete form step 04 requires** (bleed named, entries by ear not by count, singers audible in each other's microphones), the **congregational size cue** (people standing close enough to hear each other breathe), and **Source 2's verbatim phrase** *"a joyful symphony of claps and shakers."* The raw prompts name the ingredients; they do not yet name the **room's behaviour**, which is D11 and is the single production decision carrying the most weight in this run.

---

## 2. DRAFT LYRICS

⛔ **Format contract applied to all four:** `[Theme: …]` first, `[SONG FORM: …]` second, EMO headers in the four-slot shape `[Section - EMO:<emotion> - <Role> - <cue>]` with the emotion drawn from `step00`'s 50-value `EMOTION_TAXONOMY`, ≥1 `*SFX*` cue of five words or fewer, **no digits in any sung line**, no real-artist name and no tradition label anywhere.

### V1 · **The Face On The Card**
```
[Theme: a plastic tub on a kitchen table after the good light has gone; one laminated card with a face on it is lifted, handled, and put back]
[SONG FORM: hymn. Refrain stated first, then four numbered verses, identical refrain between each and twice at the close. No bridge. No key change. One tempo. Three-part close harmony throughout, two women and one man, no lead voice at any point. In Verse 3 the two inner voices swap parts mid-phrase and cross once; neither line changes.]

[Intro - EMO:Composure - Organ And Room - one held chord, a fridge, no voices]

*fridge hum, one lamp*

[Refrain - EMO:Equanimity - Three-Part Close Harmony - flat, communal, unhurried]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Verse 1 - EMO:Composure - Three-Part Close Harmony - the hand goes in without looking]
Lamp is on. The good light's gone.
Hand goes in without a look.
Cards and cables, cold and laminate.
Fingers find it. Fingers hook.
Out it comes with someone's hair
stuck along the sticky seam.
Pick it off. It won't come off.
Pull it off against the beam.
Wipe it on my sleeve. Hold it
level with the kitchen lamp.
Plastic, and inside the plastic,
paper that has never been damp.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - flat, communal, unhurried]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Verse 2 - EMO:Irritation - Three-Part Close Harmony - claps and shakers enter, played close]
Corner's lifting. Seal has failed.
Thin grey air has got inside.
Press it with a thumb and hold it.
Air just moves. It goes to hide.
Ink beneath has gone to powder.
Dates along the edge are past.
Stop it went from has been moved.
Card is fine. The card will last.
Nothing here that says to bin it.
Nothing here that says to keep.
Plastic doesn't mind the difference.
Plastic's dry. The tub is deep.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - flat, communal, unhurried]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Verse 3 - EMO:Detachment - Two Inner Voices Swap Parts Mid-Phrase - they cross once on the second thumb, neither line changes]
Turn it over. There's a face.
Photo taken at a wall.
Someone in a collar, looking
slightly past the lens. That's all.
Thumb goes over it. Comes off.
Thumb goes over it again.
I don't know who this is at all.
Nobody to ask. And then
nothing. Just the fridge. Just me,
holding someone to the light.
Face is fine. The face is nobody.
Set it down. That's that. All right.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - flat, communal, unhurried]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Verse 4 - EMO:Composure - Three-Part Close Harmony - the lid, the shelf, the door]
Face down first, then face up, then
face down. Doesn't matter. In.
In it goes on top of the rest.
Lid goes on along the line.
Press the corners. One won't take.
Press it anyway. It's fine.
Tub goes up above the door
where the warm air is. That's mine
sorted. Autumn, do it properly.
Chair goes back against the wall.
Lamp goes off. The kitchen's dark.
Out, and down the hall.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - flat, communal, unhurried]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - the empty room keeps singing it]
Keep it. Keep it. Keep it flat.
Keep the plastic, keep it plain.
Laminate and everlasting.
Nothing in this tub gets rain.
Keep the corner. Keep the card.
Keep it. Keep it. Keep it plain.

[Outro - EMO:Equanimity - Organ And Room - the held chord alone, triangle stops last]
```

### V2 · **A Key For A Sold Car**
```
[Theme: a plastic tub on a kitchen table; one key on a plastic fob, to a car that was sold, is found by sound, pressed once, and put back]
[SONG FORM: hymn. Refrain stated first, then four numbered verses, identical refrain between each and twice at the close. No bridge. No key change. One tempo. Three-part close harmony throughout, two women and one man, no lead voice at any point. In Verse 3 the two inner voices swap parts mid-phrase and cross once; neither line changes.]

[Intro - EMO:Composure - Organ And Room - one held chord, a fridge, no voices]

*small key rattles once*

[Refrain - EMO:Equanimity - Three-Part Close Harmony - level, close, nobody in front]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Verse 1 - EMO:Composure - Three-Part Close Harmony - the hand goes in for something else]
Kettle's off. The lamp's enough.
Reaching for the pile of leads.
Something small and cold goes rattle
underneath the tangled threads.
Don't look down. Just feel for it.
Fingers close on something round.
Out it comes: a key, a keyring,
plastic fob, a rubber sound.
Set it on the table. Listen.
Table's cold and slightly damp.
Metal ticking as it warms up
level with the kitchen lamp.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - level, close, nobody in front]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Verse 2 - EMO:Absorption - Three-Part Close Harmony - claps and shakers enter, played close]
Button's worn down to the white
where a thumb went, years of thumb.
Seam has split along the side.
Something in it's gone to crumb.
Ring's been forced out of its ring.
Someone pulled a key off, leaving
half a coil that grips at nothing,
hooked on nothing, holding nothing.
Metal's brass, or near to brass.
Plastic's cracked. The plastic's old.
Key's still sharp along its cut edge.
Key is fine. The car was sold.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - level, close, nobody in front]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Verse 3 - EMO:Detachment - Two Inner Voices Swap Parts Mid-Phrase - they cross once on the second nothing, neither line changes]
Thumb goes on the button. Press.
Nothing. Press the button. Press.
Nothing coming. Nothing came.
Nothing's what it does. Regardless,
press it once more, out of habit.
Nothing. That is what it's for.
Dead for years, this little battery.
Car went out the door before.
Put it down. And pick it up.
Put it down. And that is that.
Not for throwing. Not for keeping.
Nothing to be done with that.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - level, close, nobody in front]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Verse 4 - EMO:Composure - Three-Part Close Harmony - the lid, the shelf, the stairs]
In it goes on top. It sits.
Something underneath it shifts.
Lid goes on. The corner lifts.
Press the corner down. It fits.
Tub goes up above the door,
high enough to need a chair.
Chair goes back against the table.
Warm air up there. Leave it there.
Autumn, then. I'll do it properly.
Autumn, when there's time and light.
Kettle's cold. The lamp goes off.
Out. The hall. The stairs. The night.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - level, close, nobody in front]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - the empty room keeps singing it]
Hold it. Hold it. Hold it still.
Hold the plastic, hold the ring.
Brass and button, past all wearing.
Nothing in this tub goes missing.
Hold the fob and hold the key.
Hold it. Hold it. Hold the thing.

[Outro - EMO:Equanimity - Organ And Room - the held chord alone, triangle stops last]
```

### V3 · **A Photograph Of Sand** ⭐
```
[Theme: a plastic tub on a kitchen table; one photograph, stuck to the bottom, of a heap of sand a stranger stopped to photograph; it is peeled up, handled, and put back]
[SONG FORM: hymn. Refrain stated first, then four numbered verses, identical refrain between each and twice at the close. No bridge. No key change. One tempo. Three-part close harmony throughout, two women and one man, no lead voice at any point. In Verse 3 the two inner voices swap parts mid-phrase and cross once; neither line changes.]

[Intro - EMO:Composure - Organ And Room - one held chord, a fridge, no voices]

*photograph peels off plastic*

[Refrain - EMO:Equanimity - Three-Part Close Harmony - plain, side by side, one thickened voice]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Verse 1 - EMO:Composure - Three-Part Close Harmony - peeling it off the bottom]
Bottom of the tub is tacky.
Something's stuck flat to the base.
Get a nail beneath the edge.
Peel it up. It leaves a crease.
Comes up slow and comes up sticky,
lifting, sticking, lifting, sticking.
Photograph. Somebody's photograph.
Wrong size. Wrong for anything.
Hold it by the edges. Careful.
Thumb marks on it. Now there's more.
Set it flat. It sticks again.
Table's got a mark. Ignore.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - plain, side by side, one thickened voice]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Verse 2 - EMO:Irritation - Three-Part Close Harmony - claps and shakers enter, played close]
Wrong size for a frame. Too tall.
Wrong size for a purse. Too wide.
Crease across the middle where
something heavy sat inside.
Gloss has gone off half the surface.
Half of it is shiny yet.
Someone's fingerprint below it,
older than the one I've set.
Doesn't fit a single pocket.
Doesn't fit the album spine.
Nothing in the house it goes with.
Nothing in the house. That's fine.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - plain, side by side, one thickened voice]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Verse 3 - EMO:Detachment - Two Inner Voices Swap Parts Mid-Phrase - they cross once on Someone, neither line changes]
Sand. It's sand. A hill of sand.
Somebody has built it wide.
Somebody has stood and made it.
Somebody has stood beside.
Then a stranger with a camera
stops and takes it. That is all.
Not a friend. Not anybody.
Stranger at a stranger's wall.
Somebody has bothered. Someone
stopped, and looked, and carried on.
That's the photo. That's the whole
of it. Some sand. Put it down.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - plain, side by side, one thickened voice]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Verse 4 - EMO:Composure - Three-Part Close Harmony - the lid, the cupboard, the door]
Face up. Face down. Face up again.
Doesn't matter. In it goes.
Something underneath it shifts.
Lid goes on. A corner shows.
Press it flat. It lifts. Press hard.
Press it down. It holds. It's in.
Tub goes up above the cupboard.
Chair goes back against the wall.
Autumn. Do the lot at once.
Autumn, when the days go dark.
Lamp goes off. The window's black.
Out. And leave the kitchen dark.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - plain, side by side, one thickened voice]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - the empty room keeps singing it]
Nothing in this tub shall want.
Nothing in this tub shall tear.
Plastic, paper, past all perishing.
Nothing here will disappear.
Keep the corner. Keep the crease.
Nothing in this tub shall tear.

[Outro - EMO:Equanimity - Organ And Room - the held chord alone, triangle stops last]
```

### V4 · **The Lid And The High Shelf** 🛟
```
[Theme: a plastic tub on a kitchen table; the lid is tested, the tub is lifted, and it goes on a high shelf. Nothing is taken out]
[SONG FORM: hymn. Refrain stated first, then four numbered verses, identical refrain between each and twice at the close. No bridge. No key change. One tempo. Three-part close harmony throughout, two women and one man, no lead voice at any point. A gentle piano refrain doubles the sung refrain from the third verse. In Verse 3 the two inner voices swap parts mid-phrase and cross once; neither line changes.]

[Intro - EMO:Composure - Organ And Room - one held chord, a fridge, no voices]

*plastic lid clicks shut*

[Refrain - EMO:Equanimity - Three-Part Close Harmony - patient, close, nobody leading]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Verse 1 - EMO:Composure - Three-Part Close Harmony - the lid, one corner]
Tub is on the kitchen table.
Lid is on. It's on. It's fine.
One long corner's stopped believing
in the groove along the line.
Press it with the heel of a hand.
Hear it click and hear it take.
One short click. And then it lifts.
Press it. Click. For its own sake.
Everything in there is quiet.
Nothing in there needs a thing.
Lid's the only part that argues.
Lid's the only moving thing.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - patient, close, nobody leading]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Verse 2 - EMO:Resignation - Three-Part Close Harmony - claps and shakers enter, played close]
Could go through it. Not tonight.
Light's gone. Kitchen bulb's too bright.
Can't tell paper from receipt.
Won't tell either by this light.
Wardrobe first. The wardrobe's simple.
Wardrobe empties in a day.
This does not. This needs deciding.
Deciding needs a Saturday.
And I'd have to be quite sure,
and I'm not, and that's the whole
reason, and it's a good reason.
Autumn. Autumn. Leave it whole.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - patient, close, nobody leading]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Verse 3 - EMO:Detachment - Two Inner Voices Swap Parts Mid-Phrase - they cross once on the second Something, neither line changes; gentle piano refrain enters]
Both hands under. Lift. It's heavier
than a tub of nothing much.
Something in there slides and settles.
Something in there answers touch.
Hold it steady. Something shifts.
Hold it still. It shifts. That's fine.
Whatever's doing that in there
will be doing it in time.
Not my business what it's doing.
Not tonight. Not on this shelf.
Lid stays on. And what's inside it
carries on all by itself.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - gentle piano doubles the refrain]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Verse 4 - EMO:Composure - Three-Part Close Harmony - the chair, the high shelf, the stairs]
Chair across. Stand on the chair.
Shelf above the door is clear.
Push it back against the wall.
Push it right back to the rear.
Down. And put the chair back straight.
Kitchen's how it was. It's fine.
Nothing's gone and nothing's opened.
Nothing crossed a single line.
Autumn, then. And bin bags. Right.
Autumn's got a whole Sunday.
Lamp goes off. The hall's not light.
Out. The stairs. And that's tonight.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - gentle piano doubles the refrain]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Refrain - EMO:Equanimity - Three-Part Close Harmony - the empty room keeps singing it]
Let it be. Let it abide.
Let the lid stay where it's laid.
Plastic, patient, past all wearing.
Nothing in this tub's afraid.
Let it be. Let it abide.
Let it bide until the autumn.

[Outro - EMO:Equanimity - Piano And Room - the gentle piano refrain alone, then the fridge]
```

---

## 3. DRAFT-STAGE MEASUREMENT (instrument: `scripts/measure_soundcraft.py → profile()`)

**Extraction proof printed before conclusion** (handoff §4: *"Print what was EXTRACTED before trusting what was CONCLUDED"*). Each block extracted cleanly: **1 block, 84 sung lines**, first line = the refrain's opening, last line = the refrain's closing.

| | end_rhyme ≥0.30 | line_return ≥0.20 | words/line ≤7.5 | allit/100w ≥11.0 | sung lines | field chars |
|---|---|---|---|---|---|---|
| **V1** | **0.476** | **0.429** | **5.95** | **18.80** | **84** | 4220 |
| **V2** | **0.595** | **0.429** | **6.19** | **19.04** | **84** | 4387 |
| **V3** | **0.464** | **0.429** | **5.65** | **21.89** | **84** | 4375 |
| **V4** | **0.595** | **0.429** | **5.89** | **15.35** | **84** | 4365 |

**All four pass every floor at draft stage. No repair attempt was needed on any gate.** Full enumeration, including the gates this table does not show, is in step 10.

⚠️ **The wordless-return caveat does not apply to this pair.** The return vehicle is a **byte-identical lexical refrain**, not a hum or a vocable — so `line_return 0.429` is entirely lexical and needs no companion measurement. Stripping every non-lexical element from these lyrics changes the number by **zero**, because there are no vocables in them.

**Self-critique.** The raw prompts are competent and slightly dead: they list ingredients and never say what the room *does*. That is exactly the failure step 04 warned about — *"a vague version renders as nothing."* Four prompts that say "recorded in a kitchen" will render as four songs recorded in a studio with a reverb preset called *kitchen*. Step 09's job is to replace the noun with the **behaviour** — bleed, entries by ear, breath distance — and if the renders come back sounding like a booth, the diff between these raw prompts and the step-10 finals is the experiment that will say whether that language did any work.
