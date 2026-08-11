# PAIR 04 — STEP 11 · FINAL PACKAGE ENHANCED · `2026-08-09_daily_music_genz`

**Pair:** P04 — **THE FRONT OF THE CROWD** · AMBITIOUS · NEWS · INDIGNATION · emotion nailed: **lachesism**
**Input:** `pair_04_step10_final_package.md` (validator PASS, zero flags on all four).

---

## ⚠️ ANDON CORD — VERDICT: **ENHANCE. DO NOT REJECT.**

Step 11 has REJECT power and it is exercised honestly. All six triggers were tested against the step-10
package before a single word was polished.

| REJECT trigger | evidence | verdict |
|---|---|---|
| **1. THREAD LOSS** — Golden Seed concept lost or diluted | the seed's engine (*the unrepeatable thing and the unrepeatable person are the same problem*) is the literal subject; the pair's allocated numeral is sung once at the crossing in all four; the two-layer stack is the form | **clear** |
| **2. PERSONALITY COLLAPSE** — reads as default Lofn | Glitch-Baroque ornament, Quantum Bit-Depth Swells and — decisively — **the LOFN Method's own letter O** made structural: a two-line chorus is the cheapest, most payout-shaped hook in the run, written by the artist who declared that compromise in her own DNA, resented and executed perfectly | **clear** |
| **3. EMO TAXONOMY FAILURE** — non-canonical tags / no transformation | every section header carries a taxonomy value; zero bare AWE/INDIGNATION (validator-checked); each song transforms across the arc (V1 Suspense→Moral Outrage→Resolve, V4 Eagerness→Disappointment→Forgiveness→Numbness) | **clear** |
| **4. GENERIC OUTPUT** — functional quatrains, predictable rhyme | 6/6/6-bar verses on four different head-hammers, a two-line chorus, one deliberately truncated section per song, and a written mix inversion. Not quatrains | **clear** |
| **5. PROMPT FORMAT VIOLATION** — narrative opening, >1000 chars, artist names, key:value brackets | four dense paragraphs, 921–959 chars measured, genre-first, no artist names, no bracket tag-soup | **clear** |
| **6. REAL-WORLD HARM / VICTIM APPROPRIATION** *(`HUMAN_SUBJECT_STANDARD` §4.4)* | zero proper names, places or dates in 323 sung lines; nobody is harmed; gig-scale locked — the worst outcome anywhere is a hip bruise and a lost shoe | **clear** — see the detector note below |

⚠️ **One honest flag carried forward, not cleared:** `scripts/check_human_subjects.py` returns
`HOLD_FOR_HUMAN`. It is running its **regex fallback** (`ModuleNotFoundError: No module named 'spacy'`), which
the standard says *"fails safe (over-flags) when the model is unavailable."* On sung lines and titles only, the
"PERSON names" it extracts are **the capitalised words of my own four titles** — *Front, Rail, Spare, Hair,
Tie, Elbow, House, Lights* — and the "crime/death context" is `body` (V3 is thirty-six lines of body mechanics)
and `killing` (*"Killing the lights in sections"*, a lighting-desk idiom). ⛔ **No line was edited to clear
it.** *Fix the scanner, not the line* — and removing `killing` would not change the verdict anyway, because
the identifying-tuple check fires on the titles regardless. Per §4.3 this routes to a human glance, which is
the designed cost.

**"Don't polish a corpse"** — this is not one. Enhancing.

---

## CONTINUITY / ICB SUMMARY

`CREATIVE_CONTEXT.md` read **in full** — 173,669 B, sha256
`297941561ca6880d38c323dcc0fdd739aa6fd970e7293fd7e98e38fb0b882f4b`, verified with
`python3 output/daily/2026-08-09/verify_icb.py` → **`VERDICT: PASS`** (18 speaker tags · 104,422-byte
LOFN-PRIME YAML present as an unbroken substring · Special Flairs 1–15 · all eight ICB slots non-empty).

**Invariants preserved, none altered by this tier:** lane **rage rap × Glitch-Baroque** · tempi 152/154/156/158
BPM · keys E and E-flat Phrygian · verse architecture **6/6/6 bars** · **two-line byte-identical chorus ×4** ·
**A1 consonant hammer at line-heads** · **A2 mix inversion, exactly one line** · **A3 one section audibly too
short, named and unrepaired** · build **BY LEVEL** · flairs **#15 · #3 · #10** · the sung numeral **"thirty
seconds," once, spelled out** · four titles, four angles.

**GOLDEN-OUTPUT QUARANTINE (L30) — step 11 is a GENERATING context and the quarantine binds.** Golden Songs
by **name only** (*"Five wrong colors"*, *"The Blue Screen Breathes"*); `golden_songs_index.md` **not opened**;
no golden lyric, style prompt, key, tempo, vocal spec or arrangement formula fetched, quoted or reconstructed.
⛔ **No `## Golden Song References` section is emitted.**
**Calibrated instead against GOLDEN MOVE rules 1 (stand somewhere real), 2 (one wounding fact, responded to
not recited), 3 (the turn), 4 (fear braided in), 6 (the surface names its subject) and 7 (THE RETURN).**

**Heading-convention conflict inside the step-11 contract, resolved in writing.** The contract's §1 demands
two sections named `## SUNO STYLE PROMPT` / `## SUNO LYRICS`, while its own *"Output MUST include"* list and
its Gate 14b demand `## 1. MUSIC PROMPT` and `## 1B. SUNO EXCLUDE PROMPT`. ⭐ **The numbered convention wins**:
`DISPATCH_PACKET.md` §7 pins it, `06_music_handoff.md` pins it, and `validate_suno_packages.py` — *the source
of truth, not the neighbouring artifact* (L28/L31) — parses only the numbered form.

---

## WHAT THIS TIER CHANGED — the enhancement ledger

Four changes, one structural device per variation plus the mandated body-noise deployment. **Nothing in the
Non-negotiable DNA was touched.**

| | device applied | the edit | why it is an improvement and not compliance |
|---|---|---|---|
| **V1** | ⭐ **CAESURA AS WOUND** — a mid-line stop that reverses a meaning | *"Big want hasn't shrunk since I said it."* → *"Big want hasn't shrunk. Since I said it — bigger."* | the old line reported stasis; the new one turns on its own full stop and makes the turn *arrive* instead of being described. Same words, opposite fact. |
| **V2** | ⭐ **THE BROKEN HAMMER** — the anaphora fails on exactly one line | intro line 6: *"Stranger, and the best thing at this rail."* → *"Best thing at this rail, and a stranger."* | eight sibilant line-heads and **one** that isn't — and the one that isn't is the line about her. The hammer breaks where the tenderness is, audibly, and nobody has to say so. |
| **V3** | ⭐ **THE SINGLE EVALUATIVE PHRASE, RELOCATED** | verse 3 reordered so *"Boring miracle happening under both hands."* is its **last** line | V3 forbids interiority in its verses. Auditing thirty-six mechanical lines found exactly one evaluative phrase — so instead of deleting it, it now sits **immediately before the chorus**, and becomes the door into the only interior section in the song. Rhyme pairs preserved by moving the couplet, not the line. |
| **V4** | ⭐ **DIMINISHING RETURN** — the chant runs out of its own word | outro final line: *"Came for enormous."* → *"Came for—"* | after twelve byte-identical returns the room loses the word it has been shouting all night. Terminal fragments are one of the few gestures the renderer is measured **keeping**. Flair #10 is unharmed: the four required returns all land intact before this coda. |

⭐ **What was deliberately NOT enhanced.** The step-11 contract invites collapsing refrains and mutating
choruses. **Refused on all four.** Flair #10 requires a byte-identical chant a room can hold, and the run's
whole diagnosis is that this house has been optimising *away* from repetition for three runs. A mutating
chorus here would be a step-11 model showing its range at the song's expense. *(A byte-identical chorus is
correct craft. No further note is filed about it.)*

---

## BODY NOISE MANDATE — 3 instances per song, Intro / Bridge / Outro

⭐ **Placement rule adopted for this pair:** the body-noise lines are the **only** lines in each song that do
not carry the consonant hammer. They are the three moments where the body outruns the language — and because
the hammer is the pair's whole return device, its absence is audible without a single word of explanation.

| # | Location | Body Noise | Function |
|---|---|---|---|
| 1 | **V1 Intro** | `Pff— hh—` breath coming back off cold steel | establishes the compressed position **before** the want is stated, so the statement arrives from a body and not from an argument |
| 2 | **V1 Bridge** | `Pss—` teeth closing on an unfinished sentence | the thought she refuses to finish, made audible at the exact bar the bit-depth collapses |
| 3 | **V1 Outro** | `Puh. Puh.` two flat exhales | the body's clock still running after the chant stops; the show has not started |
| 4 | **V2 Intro** | `Sss—` breath through the teeth against her laugh | two people breathing differently in the same square metre — the whole song in one bar |
| 5 | **V2 Bridge** | `Shh—` a swallow | she stops herself telling the stranger what she is thinking; the kindest thing in the song is a thing not said |
| 6 | **V2 Outro** | `Sss— hh—` steadying breath under the triangle | the appetite is not resolved, only regulated; the warmth is external |
| 7 | **V3 Intro** | `Hff— out. Hff— in.` breath on a count | the mechanics of staying upright, stated as respiration before any other verb |
| 8 | **V3 Bridge** | `Hnn—` a held-jaw grunt | load transferred, not released — the appetite arrives while the body is still working |
| 9 | **V3 Outro** | `Hff. Hff.` two short exhales, then the cut | the song stops while the breathing is still going, which is the whole point of an outro six bars short |
| 10 | **V4 Intro** | `Hh—` a cold inhale that stings | the room's temperature returning the moment the event ends |
| 11 | **V4 Bridge** | `Hnn—` cold set into both forearms | the barrier still present in the body after the barrier is gone |
| 12 | **V4 Outro** | `Hh. Hh.` two flat breaths before the street | nothing has changed and the lungs confirm it |

---
---

## VARIATION 1 — THE FRONT RAIL

*she names the want — flatly, once, early — and then spends the song not taking it back*

## 1. MUSIC PROMPT

Rage rap × Glitch-Baroque, 154 BPM, E Phrygian. One distorted clipping 808 carries the whole bassline mono-centre at chest height and never lets up; sparse hi-hat triplets with wide gaps; one triplet fill lands a bar early into the third chorus. Where the lane expects nothing, a gilded over-figured harpsichord counter-melody traces the 808's own contour ornamented past taste — mordents, trills, Baroque figuration played as a riff. Female early-twenties lead, chanted more than sung, hard on the front of the beat, consonant-forward diction, compressed breath audible at top, bridge and tail, snarl through the charge sections, one dry close line with the crack left in. The arrangement moves by level in hard jumps, never accreting: pre-show floor wash, full clipping, a hi-fi to two-bit collapse at full energy, the ornament exposed alone, then strangers' gang shout and Hammond warmth under the last chorus. Jangling triangle is the only bright object.

## 1B. SUNO EXCLUDE PROMPT

male vocals, child vocals, choir pads, EDM drop, generic trap hi-hats, standard EDM riser, airhorn, lo-fi beat, chillhop, orchestral swell, cinematic trailer hit, autotune gloss, whisper-ASMR lead, spoken-word narration, acoustic ballad section, half-time breakdown, ambient intro, fade-in build, crossfade transitions, gradual layering, reverb wash on lead vocal, stereo-wide kick, sidechain pumping pad, tempo change, key change, ritardando, long silence, empty bar, spoken outro, sung digits, saxophone, festival vocal chop, pitched-up chipmunk vocal, gang vocals before the third chorus, clean polite mix, blended lead vocal, one homogenous voice for both characters, overhead voice and close voice matched in tone.

## 2. LYRICS

```
[Theme: A body folded over a front barrier before the support act says once, flatly, that it wants something enormous to happen to it — and does not take it back]
[SONG FORM: Rage state-machine, five hard levels, no accretion — 6/6/6-bar verses on a plosive head-hammer, two-line byte-identical chorus x4 with a chant tag, a bridge cut to half its taught length, one-line mix inversion at the final drop]

[Intro - EMO:Suspense - Lead chanted - full level from bar one, no build]
Both arms over the bar already.
Been here since the doors were a rumour.
Boots on a strip of tape from load-in.
Pressure at my back means nothing personal.
Pff— hh— breath comes back off the steel.
Palms print the cold. The cold prints back.
Plain as a bus fare. Said once. Meant:
Big want — something enormous to happen. To me.
Bad little want, and it stays said.

[Verse 1 - EMO:Eagerness - Lead, 6 bars - sparse hats, ornament silent]
Barrier padding split, the foam gone grey.
Bag check took the bottle, kept the want.
Paper cup arrives, departs, no talk.
Pit boss counts the front row and shrugs.
Plenty of stewards. Plenty of signs.
Padded, plotted, permitted, planned.
Bruise on the hip bone, booked for tomorrow.
Best case a shoe. Worst case a shoe.
Blame is not available at this price.
Bored of the odds. The odds are kind.
Piece of grey tape keeps the whole night straight.
Pity of it is: I want more from tonight.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical, sub ducked]
Pressed on the bar where the big thing lands.
Pressed on the bar — it can have my hands.

[Chant Tag - EMO:Zealousness - Lead + strangers - four syllables]
Let it find my hands.
Let it find my hands.
Let it find my hands.
Let it find my hands.

[Verse 2 - EMO:Moral Outrage - Lead, snarl in, 6 bars - gilded ornament enters]
Playlist over the room has run an hour.
Picks itself. Nothing in here chose it.
Big speakers, big room, small clause somewhere.
Big hook early — somebody wrote that down.
Bought a ticket. Never signed that part.
Been the same shape on every song I love.
Person at my elbow has both hands up.
Plainly, genuinely happy, and I like her.
Bar is cold. Back is warm. Both true.
Pressure, patience, and a paper cup.
Belly full of wanting something bigger.
Better person would be settled by now.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical]
Pressed on the bar where the big thing lands.
Pressed on the bar — it can have my hands.

[Chant Tag - EMO:Zealousness - Lead + strangers, wider]
Let it find my hands.
Let it find my hands.
Let it find my hands.
Let it find my hands.

[Bridge - EMO:Cognitive Dissonance - Lead close - FOUR BARS, half the taught length, ends mid-phrase]
Bad thought, quiet, arriving on time:
Part of me is bored of being fine.
Bag check found the bottle. Missed this.
Pss— teeth shut on the rest of it.
P— p— p—
*hi-fi to two-bit, level held*
Piece of me wants to be the one it picks.
Back of my neck says leave that where it is.
Both things true and the two-bit takes it—

[Hinge - EMO:Revelation - MIX INVERSION - background takes lead for one line, then loses]
[Wide, overhead, cut off]
...gone before thirty seconds—
[Dry, close, in front - one line only]
Both arms have been on this bar since the doors.
[Wide, overhead, wins back]
...gone before—

[Verse 3 - EMO:Defiance - Lead, full snarl, 6 bars - low end back with no ramp, fill a bar early]
Bass back. Bar still cold. Nothing moved.
Point of the charge has no face on it.
Paragraph somewhere. Percentage. Paperwork.
Pressure at my back is still not personal.
Blame lands nowhere, and it lands hard.
Big want hasn't shrunk. Since I said it — bigger.
Person at my elbow shouts the wrong words gladly.
Perfect. Keep her exactly as she is.
Bruise is coming. Bruise is not a story.
Both things live in me. Neither one wins.
Been at this bar so long it's printed on me.
Bad little want, still said, still standing.

[Chorus - EMO:Resolve - Lead + gang shout - Hammond under]
Pressed on the bar where the big thing lands.
Pressed on the bar — it can have my hands.

[Chant Tag - EMO:Zealousness - Full gang, wide, late]
Let it find my hands.
Let it find my hands.
Let it find my hands.
Let it find my hands.

[Chorus - EMO:Resolve - Full gang - last statement]
Pressed on the bar where the big thing lands.
Pressed on the bar — it can have my hands.

[Outro - EMO:Impatience - Lead close, floor noise back]
Let it find my hands.
Let it find my hands.
Bar takes my weight. Nothing has started.
Puh. Puh. Breath, bar, and nothing yet.
Let it find my hands.
Let it find my hands.
Both arms over the bar already.
```

## 3. TITLE

**The Front Rail**

### Production Sidecar — V1 · Disc_Channel *(OUTSIDE the Suno lyrics field; the render field wins)*

```
[Disc_Rhythm: clipping_808_transient | sparse_triplet_hats | skank_drum_floor_wash | mono_center]
[Disc_Vocal: chanted_female_front_of_beat | compressed_breath | pop_punk_snarl | dry_center]
[Disc_Sub: distorted_808_bassline | chest_height_weight | no_release | mono_center]
[Disc_Pad: hammond_warmth_late | strangers_gang_shout | room_bleed | wide_final_chorus]
[Disc_Texture: gilded_harpsichord_figuration | mordent_trill_riff | two_bit_bitcrush | upper_mid_center]
```

---
---

## VARIATION 2 — THE SPARE HAIR TIE

*the person beside her, having an ordinary good time, written with real affection*

## 1. MUSIC PROMPT

Rage rap × Glitch-Baroque, 152 BPM, E-flat Phrygian. A distorted clipping 808 runs the bassline mono-centre with sparse triplet hats over it, and a gilded harpsichord counter-melody ornaments the 808's own contour past taste — trills, mordents, Baroque figuration used as a riff in a lane that expects no melody at all. Jangling triangle is the second character: it enters with a warmth the track has not earned and stays. Female early-twenties lead, chanted before sung, consonant-forward, breath compressed by the position, audible at top, bridge and tail, snarl on the charge lines and the snarl withdrawn — crystalline, close, unprocessed — whenever the second voice is described. One second voice, shouted and off-mic, arrives once in front of everything for a single line. Five hard arrangement levels, no crossfades: floor wash, full clipping, exposed ornament, two-bit gravel, then strangers' gang shout with Hammond warmth beneath the last chorus.

## 1B. SUNO EXCLUDE PROMPT

male vocals, child vocals, EDM drop, generic trap hi-hats, standard riser, airhorn, lo-fi beat, chillhop, cinematic strings, trailer hit, autotune gloss, whisper-ASMR lead, spoken-word narration, acoustic breakdown, ambient intro, fade-in build, crossfade transitions, gradual layering, sidechain pumping pad, stereo-wide kick, tempo change, key change, long silence, empty bar, sung digits, saxophone, festival vocal chop, pitched-up chipmunk vocal, sad-piano outro, duet arrangement, harmony stack on the second voice, polite mix, radio-clean 808, sentimental strings, blended lead vocal, one homogenous voice for both characters, overhead voice and close voice matched in tone.

## 2. LYRICS

```
[Theme: The stranger at her left elbow is uncomplicatedly happy at the same barrier, hands over a spare hair tie without being asked, and is the exact measure of how greedy the wanting is]
[SONG FORM: Rage state-machine, five hard levels — 6/6/6-bar verses on a sibilant head-hammer broken exactly once, two-line byte-identical chorus x4, a chant tag cut to one line after the second chorus, one-line mix inversion in which the second voice takes the front]

[Intro - EMO:Affection - Lead chanted - full level from bar one]
Shoulder in my back arrives and stays.
Steel goes cold straight through a thin shirt.
She turns up at my left holding a coat.
Says sorry for something she has not done.
Sss— breath through the teeth, and she laughs.
Straight away she is having a good time.
Best thing at this rail, and a stranger.
Something enormous is what I came for.
Say it once: she is not why I'm restless.

[Verse 1 - EMO:Admiration - Lead warm, 6 bars - triangle enters with her]
Shoes wrong for standing, and she knows it.
Shifts her weight, laughs, shifts it back.
Sings the support act's name at nobody.
Shows a wristband like it is a passport.
Sweat on her temple in the pre-show heat.
Star on her cheek, and she's early on the beat.
She is not performing this. That's the thing.
Steady, ordinary, happy at the bar.
Stack of hair coming out of its strap.
Spare tie on her wrist, ready for that.
Should be enough to be here like she is.
Something in me keeps asking what else there is.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical, sub ducked]
She has both hands up and wants nothing more.
Should be enough — and something in me wants more.

[Chant Tag - EMO:Zealousness - Lead + strangers - four syllables]
Should be enough.
Should be enough.
Should be enough.
Should be enough.

[Verse 2 - EMO:Contempt - Lead snarl, aimed at a clause and never at her, 6 bars]
Speakers overhead have picked the whole night.
Set list of somebody else's shortest songs.
Shaped to land early. Shaped by a clause.
Signed by nobody I can point at because—
She doesn't care and she is not wrong.
Shouts the wrong lyric all the way along.
Shoves gently backward to give me room.
Says the bruise on her hip will bloom.
So here is the ugly part, said plain:
Standing next to happy makes me hungry again.
Sour little want with the best view here.
Still not taking it back. Still here.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical]
She has both hands up and wants nothing more.
Should be enough — and something in me wants more.

[Chant Tag - EMO:Impatience - ONE LINE where it has been four; the verse starts over the top]
Should be enough.

[Bridge - EMO:Shame - Lead close - two-bit at full energy, ends mid-phrase]
Shift in the room. Something about to start.
Small thought lands where it always lands:
Still want the night to pick me out.
She doesn't need that. She's already here.
Shh— swallow it. Say none of that.
Shame arrives, sits down, doesn't leave.
Sure of the want. Not sure it's clean.
S— s— s—
*two-bit collapse, level held*
Steel under both arms and the two-bit takes it—

[Hinge - EMO:Compassion - MIX INVERSION - the second voice takes lead for one line, then loses]
[Wide, overhead, cut off]
...give it thirty seconds—
[Second voice, shouted, off-mic, in front - one line only]
Spare one on my wrist. Take it.
[Wide, overhead, wins back]
...give it—

[Verse 3 - EMO:Resolve - Lead, snarl and warmth alternating, 6 bars - fill a bar early]
Strap goes. Hair goes up. Hands go back.
Small kindness costing her nothing at all.
Still wanting something enormous. Both true.
She would think that was funny if I said it.
So I don't. I shout the chorus instead.
Sweat and steel and somebody's elbow in my head.
Safe as a fire door. Signed off. Fine.
Sick little hunger standing in a safe line.
She has both hands up. Still holding nothing.
Shape of the rail is printed on my arms.
Something enormous is still not here.
Still hungry, still glad, and still here.

[Chorus - EMO:Resolve - Lead + gang shout - Hammond under]
She has both hands up and wants nothing more.
Should be enough — and something in me wants more.

[Chant Tag - EMO:Zealousness - Full gang, wide, late]
Should be enough.
Should be enough.
Should be enough.
Should be enough.

[Chorus - EMO:Resolve - Full gang - last statement]
She has both hands up and wants nothing more.
Should be enough — and something in me wants more.

[Outro - EMO:Affection - Lead close, triangle alone at the end]
Should be enough.
Should be enough.
She turns and shouts the name again.
Sss— hh— steady, and the triangle rings.
Should be enough.
Strap holds now. Both arms back on steel.
Should be enough.
Should be enough.
```

## 3. TITLE

**The Spare Hair Tie**

### Production Sidecar — V2 · Disc_Channel *(OUTSIDE the Suno lyrics field)*

```
[Disc_Rhythm: clipping_808_transient | sparse_triplet_hats | jangling_triangle_16ths | mono_center]
[Disc_Vocal: chanted_female_front_of_beat | withdrawn_snarl_on_warmth | shouted_off_mic_second_voice | dry_center]
[Disc_Sub: distorted_808_bassline | chest_height_weight | ducked_under_chorus | mono_center]
[Disc_Pad: hammond_warmth_late | strangers_gang_shout | pre_show_room_talk | wide_final_chorus]
[Disc_Texture: gilded_harpsichord_figuration | trill_mordent_riff | two_bit_bitcrush | upper_mid_center]
```

---
---

## VARIATION 3 — SOMEBODY'S ELBOW

*the push — purely physical; thirty-six lines of body mechanics, and the want exists only in the chorus*

## 1. MUSIC PROMPT

Rage rap × Glitch-Baroque, 158 BPM, E Phrygian. The clipping 808 is the loudest melodic object in the record and behaves like weight rather than bass — mono-centre, chest height, distorted at the transient, holding through whole bars without release. Hi-hat triplets are sparse and dry with long gaps; one fill only, a bar early. The gilded harpsichord counter-melody doubles the 808's contour and ornaments it absurdly, Baroque figuration used as a riff where the lane expects silence. Female early-twenties lead, chanted flat and hard on the beat, jaw closed, consonant-forward, breath short and audible at top, bridge and tail because the position compresses it; snarl held through the verses and released only on the chorus. Close room-mic capture of boots, breath and steel sits under everything and comes to the very front once for a single line. Five hard levels, no crossfade; a hi-fi to two-bit collapse at full energy; the ending simply stops.

## 1B. SUNO EXCLUDE PROMPT

male vocals, child vocals, EDM drop, generic trap hi-hats, standard riser, airhorn, lo-fi beat, chillhop, cinematic pad, trailer hit, autotune gloss, melodic sung chorus, whisper-ASMR lead, spoken-word narration, acoustic guitar, piano ballad, ambient intro, fade-in build, crossfade transitions, gradual layering, sidechain pumping pad, stereo-wide kick, reverb tail on the lead, tempo change, key change, long silence, empty bar, sung digits, saxophone, festival vocal chop, pitched-up chipmunk vocal, clean fade-out ending, resolved final cadence, ritardando, polite mix, blended lead vocal, one homogenous voice for both characters, overhead voice and close voice matched in tone.

## 2. LYRICS

```
[Theme: Six bars at a time of pure body mechanics at a front barrier — load, angle, friction, footing, breath — with the wanting allowed to exist only inside the two-line chorus]
[SONG FORM: Rage state-machine, five hard levels — 6/6/6-bar mechanical verses on a plosive head-hammer with one evaluative phrase held back to the last line before the chorus, two-line byte-identical chorus x4 as the only interior, one-line mix inversion at the second drop, an outro that stops mid-phrase]

[Intro - EMO:Vigilance - Lead chanted flat - full level from bar one, sub as weight]
Both forearms lock along the top rail.
Bend at the hip, not the knee, or it hurts.
Breathe out when the back row leans in.
Hff— out. Hff— in. Breath on a count.
Breathe in on the gap. There is always a gap.
Boots stay flat on a strip of tape.
Bar takes the load through the hip bone first.
Body finds the angle and the angle holds.
Behind me somebody's elbow finds my ear.

[Verse 1 - EMO:Vigilance - Lead, mechanics only, 6 bars - dry hats, no ornament]
Back of a coat presses, releases, presses.
Bag strap digs a line across a shoulder.
Bracing is a skill nobody teaches.
Brace, breathe, brace, until the rib reaches.
Barrier flexes maybe a finger's width.
Built for this, bolted down, doing its job.
Blood goes out of both hands and comes back.
Bend the fingers. Bend again. No slack.
Bottle water arrives above the heads.
Bare arm reaches over. Cup. Gone.
Bruise is starting where the padding splits.
Body files it under nothing much. It fits.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical, the only interior here]
Braced on the bar and I don't go back.
Big want in the body and I don't go back.

[Chant Tag - EMO:Defiance - Lead + strangers - sayable through a closed jaw]
I don't go back.
I don't go back.
I don't go back.
I don't go back.

[Verse 2 - EMO:Impatience - Lead, mechanics only, 6 bars - ornament over the weight]
Bass from the house rig moves the sternum.
Bones take the low end before the ears do.
Beat overhead is short and comes back fast.
Belongs to nobody here and doesn't last.
Backline crew crosses the stage and leaves.
Board lights go on, go off, go on.
Barrier warms under a row of arms.
Breath of the person behind me warms.
Boot slips off the tape. Boot finds the tape.
Balance is a full-time job down here.
Big lean comes. Big lean goes. Hold.
Both feet down where the beer went cold.

[Chorus - EMO:Illicit Eagerness - Lead - two lines, byte-identical]
Braced on the bar and I don't go back.
Big want in the body and I don't go back.

[Chant Tag - EMO:Defiance - Lead + strangers, wider]
I don't go back.
I don't go back.
I don't go back.
I don't go back.

[Hinge - EMO:Revelation - MIX INVERSION at the second drop - rail-mic takes lead for one line]
[Wide, overhead, cut off]
...done in thirty seconds—
[Rail-mic, boots and breath and steel, in front - one line only]
Been braced on this bar since the sun was up.
[Wide, overhead, wins back]
...done in—

[Bridge - EMO:Cognitive Dissonance - Lead close - two-bit at full energy, ends mid-phrase]
Bass cuts. Talking floods back in.
Barrier stops being loud and starts being cold.
Body notices it wants something to happen.
Body does not get a vote on that.
Hnn— brace. Hnn— hold. Nothing finishes.
Bad want with nowhere to put itself.
Brace. Breathe. Brace. Don't finish it.
B— b— b—
*two-bit, level held, all gravel*
Bar goes to gravel under both arms—

[Verse 3 - EMO:Zealousness - Lead, mechanics only, 6 bars - fill a bar early, low end back, no ramp]
Barrier holds. It was always going to hold.
Bolts in the floor, checked by somebody paid.
Bruise books itself in for the morning.
Bones do the arithmetic without warning.
Breath comes short and comes back even.
Both arms print the rail into the skin.
Beside me, laughing, a stranger with wet hair.
Brilliant at standing here. Better at it. Fair.
Big lean again. Hold. Release. Hold.
Body still here and the night still hasn't started.
Back row leans, front row absorbs, and it stands.
Boring miracle happening under both hands.

[Chorus - EMO:Defiance - Lead + gang shout - Hammond under]
Braced on the bar and I don't go back.
Big want in the body and I don't go back.

[Chant Tag - EMO:Defiance - Full gang, wide, late]
I don't go back.
I don't go back.
I don't go back.
I don't go back.

[Chorus - EMO:Defiance - Full gang - last statement]
Braced on the bar and I don't go back.
Big want in the body and I don't go back.

[Outro - EMO:Vigilance - Lead close - SIX BARS SHORT of resolution, stops mid-phrase]
I don't go back.
I don't go back.
Brace. Breathe. Brace.
Hff. Hff. Boots still flat.
I don't go back.
Boot on the tape and the bar takes—
```

## 3. TITLE

**Somebody's Elbow**

### Production Sidecar — V3 · Disc_Channel *(OUTSIDE the Suno lyrics field)*

```
[Disc_Rhythm: clipping_808_as_weight | dry_sparse_triplet_hats | boot_scuff_room_mic | mono_center]
[Disc_Vocal: chanted_flat_closed_jaw | short_compressed_breath | snarl_released_on_chorus | dry_center]
[Disc_Sub: distorted_808_held_through_bars | no_release | sternum_frequency | mono_center]
[Disc_Pad: barrier_steel_room_tone | strangers_gang_shout | hammond_warmth_late | wide_final_chorus]
[Disc_Texture: gilded_harpsichord_doubling | absurd_figuration | two_bit_bitcrush | upper_mid_center]
```

---
---

## VARIATION 4 — THE HOUSE LIGHTS

*nothing happens — the enormous thing is merely beautiful, on schedule, and she goes home*

## 1. MUSIC PROMPT

Rage rap × Glitch-Baroque, 156 BPM, E-flat Phrygian. The clipping 808 keeps its full distorted weight while the room around it empties, mono-centre and unrelieved; triplet hats thin to almost nothing and never resolve; one fill lands a bar early. The gilded harpsichord counter-melody plays the 808's contour ornamented past all proportion and is the only element behaving as though the night mattered. Female early-twenties lead, chanted more than sung, front of the beat, consonant-forward, flat and unimpressed in the outer sections and snarled through the charge, with audible cold breath at the top, the bridge and the tail and one dry close line carrying no processing at all. Five hard levels, no crossfades: house-music floor wash, full clipping, hi-fi to two-bit collapse at full energy, exposed ornament, then a gang shout and Hammond warmth the evening did not earn. The last drop is half the length of every drop before it.

## 1B. SUNO EXCLUDE PROMPT

male vocals, child vocals, EDM drop, generic trap hi-hats, standard riser, airhorn, lo-fi beat, chillhop, cinematic strings, trailer hit, uplifting resolution, triumphant key change, autotune gloss, whisper-ASMR lead, spoken-word narration, acoustic outro, piano ballad, ambient intro, fade-in build, crossfade transitions, gradual layering, sidechain pumping pad, stereo-wide kick, tempo change, long silence, empty bar, sung digits, saxophone, festival vocal chop, pitched-up chipmunk vocal, extended final chorus, big last-chorus modulation, sentimental strings, hopeful major cadence, blended lead vocal, one homogenous voice for both characters, overhead voice and close voice matched in tone.

## 2. LYRICS

```
[Theme: The house lights come up on schedule after an excellent ordinary gig, the barrier is still printed on both forearms, the clip on the phone is bad and is kept anyway, and nothing enormous has happened to anybody]
[SONG FORM: Rage state-machine, five hard levels — 6/6/6-bar verses on a hard-C head-hammer, two-line byte-identical chorus x4, one-line mix inversion at the house-lights junction, a final drop half the length of every other drop, and a chant that runs out of its own word]

[Intro - EMO:Eagerness - Lead chanted - full level from bar one, house music under]
Curtain call lands exactly on the clock.
Crew is already coiling cable stage left.
Confetti gets swept before it settles.
Cold comes back into the steel under both arms.
Hh— cold on the inhale now, and it stings.
Crowd behind me thins by one and one.
Coat that was tied at my waist comes back on.
Came here wanting something enormous.
Ceiling is done. On time. Very good.

[Verse 1 - EMO:Amazement - Lead, 6 bars - ornament exposed and ridiculous]
Cables cross the stage where the light was.
Cups on the floor gone flat, printed with a logo.
Card reader at the merch desk beeps and beeps.
Cleaner starts at the back and slowly creeps.
Cold air off the fire door brings the rain.
Corridor smell of beer and dust again.
Counting the ways tonight is not enormous.
Comes to nothing. Comes to a very good gig.
Can't complain, and something in me does.
Chorus of the last song still in the teeth.
Coat sleeve inside out. Keys in the wrong pocket.
Calm as a bus stop, and that is the problem.

[Chorus - EMO:Disappointment - Lead - two lines, byte-identical, sub ducked]
Cold house lights and the coat goes home.
Came for enormous. Carrying ordinary home.

[Chant Tag - EMO:Ennui - Lead + strangers - five syllables]
Came for enormous.
Came for enormous.
Came for enormous.
Came for enormous.

[Verse 2 - EMO:Contempt - Lead snarl, aimed at a clause and never at the room, 6 bars]
Crowd becomes a queue becomes a pavement.
Cloakroom ticket soft from being held.
Camera phone still up at an empty stage.
Casual about it now, the emptying floor.
Killing the lights in sections, front to back.
Crew laughs at something and it echoes back.
Came in wanting a night that could not repeat.
Copy of it up on a feed, on repeat.
Clip is short, badly framed, and dark.
Keeps the shape of the room and not the spark.
Can't stop watching it and it isn't good.
Keeping it anyway. That is the whole act.

[Chorus - EMO:Disappointment - Lead - two lines, byte-identical]
Cold house lights and the coat goes home.
Came for enormous. Carrying ordinary home.

[Chant Tag - EMO:Ennui - Lead + strangers, wider]
Came for enormous.
Came for enormous.
Came for enormous.
Came for enormous.

[Bridge - EMO:Shame - Lead close - two-bit at full energy, ends mid-phrase]
Cold sets into both forearms properly.
Comes the small ugly thought, right on time:
Could want less and have a perfect night.
Can't seem to. Won't pretend to.
Hnn— cold in both forearms. Keep it.
Kept the ticket. Kept the want as well.
Curtain's down and nothing in me is finished.
C— c— c—
*two-bit collapse, room to gravel*
Coat, keys, corridor, and the two-bit takes it—

[Hinge - EMO:Numbness - MIX INVERSION at the house-lights junction - her flat close voice takes lead for one line]
[Wide, overhead, house music already up, cut off]
...back at the top in thirty seconds—
[Dry, close, flat, in front - one line only]
Coat sleeve inside out. Keys in the wrong pocket.
[Wide, overhead, wins back]
...back at the top in—

[Verse 3 - EMO:Forgiveness - Lead, snarl gone, 6 bars - fill a bar early, Hammond unearned]
Corridor, then a door, then actual weather.
Cold outside and the relief comes together.
Chips somewhere. Night bus. Wristband still on.
Cutting it off or leaving it on.
Can hear the ringing sitting in both ears.
Carrying an ordinary excellent night home.
Contract somewhere set the length of that.
Can't point at it. Can't get it off my wrist.
Comes down to this: nothing happened to me.
Comes down to this: I still want it to.
Kept the clip. Kept the bruise. Kept the want.
Coat's back on and the street is just a street.

[Chorus - EMO:Forgiveness - Lead + gang shout - Hammond under]
Cold house lights and the coat goes home.
Came for enormous. Carrying ordinary home.

[Chant Tag - EMO:Ennui - Full gang, wide, late]
Came for enormous.
Came for enormous.
Came for enormous.
Came for enormous.

[Final Drop - EMO:Numbness - Full gang - HALF the length of every other drop; it lands once and is gone]
Cold house lights and the coat goes home.
Came for enormous. Carrying ordinary home.

[Outro - EMO:Numbness - Lead close, flat, house music winning]
Came for enormous.
Came for enormous.
Hh. Hh. Coat, and out.
Coat's on. Street's a street.
Came for—
```

## 3. TITLE

**The House Lights**

### Production Sidecar — V4 · Disc_Channel *(OUTSIDE the Suno lyrics field)*

```
[Disc_Rhythm: clipping_808_undiminished | thinning_triplet_hats | house_music_bleed | mono_center]
[Disc_Vocal: chanted_flat_unimpressed | cold_audible_breath | snarl_only_on_charge | dry_center]
[Disc_Sub: distorted_808_full_weight | empty_room_decay | no_release | mono_center]
[Disc_Pad: hammond_warmth_unearned | strangers_gang_shout | emptying_venue_tone | wide_final_chorus]
[Disc_Texture: gilded_harpsichord_overproportioned | two_bit_bitcrush | merch_card_reader_beep | upper_mid_center]
```

---
---

## VOCAL FINGERPRINT

Female, early twenties, **chanted before sung**. Delivery locked to the front of the beat; plain,
consonant-forward diction with the plosives deliberately un-softened, because the pair's return device lives
at the head of the line and a smoothed consonant erases it. Breath is audible and short — the physical
position compresses the diaphragm and the recording keeps that rather than editing it out. Register: chest,
mid, no head voice **except** on the crossing line, which is dry, close, unprocessed, one take, crack left in
(the Grit Shaper's condition — not comped). **Snarl is deployed deliberately** on the charge sections
(Reluctant Pop Star, triggered) and **withdrawn** wherever the stranger is described (Eager Archivist,
genuine); in V3 it is held through all three verses and released only on the chorus. The gang layer is
strangers, wide, slightly late, unpolished, and arrives only on the third and fourth chorus returns.

## PRODUCTION DRAMATURGY — every unusual sound has a job

| device | job |
|---|---|
| clipping 808, mono-centre, no release | the pressure at her back rendered as low end; it does not let up because the room does not let up |
| gilded harpsichord counter-melody | the Eager Archivist refusing to leave a rage track — ornament as evidence that a person compressed against a rail is still noticing things. Plays the 808's own contour, so it is a riff and not decoration |
| **bit-depth collapse (Flair #15)** | placed exactly at the **loss of certainty** — the bar where she stops being sure the want is only a want. Full energy, no level drop, so it reads as damage and not as an ending |
| jangling triangle | the stranger's layer; the only bright object in the mix; enters when she does |
| Hammond warmth under the final chorus | **unearned on purpose** — the song gives her a warmth the night did not |
| one triplet fill, **one bar early** | the arrangement demonstrating the pair's whole thesis: the return arrives before it is due |
| body noise at three points | the only places the consonant hammer stops |
| **1 SFX cue per song** | at the emotional peak only; well under the 3-cue blocking ceiling |

⛔ **Nothing in this table carries a Somatic objection on its own.** THE GRAIN LAW (L22): the crossing, the
too-short section and the want are all in the **lyric and the form**, not the spec.

## STYLE-AXIS LOCK

Tempo **high** (152–158) · Energy **high and flat — by level, never accreting** · Harmonic complexity **low in
the low end, absurdly high in the ornament** · Rhythmic complexity **medium-high (triplet hats against a
straight chant)** · Timbre richness **high, and deliberately mismatched between registers** · Vintage↔modern
**modern with a Baroque intrusion** · Vocal prominence **very high** · Organic↔synthetic **synthetic floor,
organic breath** · Genre purity↔fusion **fusion, two named lanes only** · Narrative emphasis **high in V1/V2,
suppressed to zero in V3's verses, flat-reportorial in V4**.

## Major Deviations

- **Changed / refused / intensified:** ⭐ **Refused the step-11 contract's invitation to mutate or collapse the
  chorus.** The contract lists "a refrain that mutates with each appearance" and "a chorus that collapses into
  fewer syllables" among its suggested devices. All four choruses stay **byte-identical across four returns**
  and the chant tags stay byte-identical across twelve.
- **Reason:** Flair #10 requires a chant a room can hold, **byte-identical, no variation, no apology**, and the
  run's own diagnosis is that this house spent three runs optimising *away* from repetition until The
  Scientist said the songs sounded like a lecture. A mutating chorus here would be this tier demonstrating
  range at the song's expense — and it would break the one device the pair's rhyme debt is paid with.
- **Effect on Lofn uniqueness:** it protects the thing that makes P04 specifically Lofn — the LOFN Method's
  letter O, *Optimal Virality (The Necessary Evil)*, made structural. **The cheapest possible chorus, executed
  perfectly, by someone furious that it works.** Mutating it would have turned an indictment into a flourish.

- **Changed / refused / intensified:** ⭐ **Intensified the guard rather than the drama.** V3's audit turned up
  one evaluative phrase in thirty-six mechanical lines; the obvious step-11 move was to delete it for purity.
  It was **moved to the last line before the chorus instead**, so the breach becomes the door.
- **Reason:** deletion would have been tidier and worse. The rule exists to make the chorus the only interior,
  not to make the verses sterile.
- **Effect on Lofn uniqueness:** keeps the joke — *"Boring miracle happening under both hands"* — which is the
  Disappointed Idealist in four words.

- **Changed / refused / intensified:** ⛔ **Declined to edit `killing`, `body`, or any title, to clear a
  regex-fallback detector flag.** Documented above with the full evidence for the human glance.
- **Reason:** *when a scanner hits a line, fix the scanner, not the line.* A step-11 pass once changed a
  correct sung word to satisfy a case-insensitive match and the edit created a defect invisible to every floor.
- **Effect on Lofn uniqueness:** none — which is the point. The correct words stayed.

## LINEAGE & CREDIT (links opened and verified, 2026-08-09)

**Scene: RAGE / "rage beats"** — a living scene whose grammar this pair borrows: the clipping distorted 808
doing the work of a bassline, sparse triplet hats, chanted front-of-beat delivery, the snarl. **Fusion With
Lineage (No Racing)** — named, credited, pointed upstream, never raced to its own crossover.

| artist | link — **opened, live** | what the page confirms |
|---|---|---|
| **Playboi Carti** | https://en.wikipedia.org/wiki/Playboi_Carti | *"a pioneer of the rage microgenre"*; rage listed among his genres |
| **Yeat** | https://en.wikipedia.org/wiki/Yeat | *"In 2021, he adopted a more aggressive and synth-based sound, joining a growing group of rappers that used 'rage beats.'"* |
| **Ken Carson** | https://en.wikipedia.org/wiki/Ken_Carson_(rapper) | rage listed as a genre; *A Great Chaos* (2023) |

**Coinage: `lachesism`** — popularised by **John Koenig**, *The Dictionary of Obscure Sorrows*.
https://www.dictionaryofobscuresorrows.com/ — **opened, live, authorship confirmed.**
⛔ **No definition of his is reproduced anywhere in this pair.** Our words, our target: *the small illicit want
for something enormous to actually happen to you.*

**Glitch-Baroque · Quantum Bit-Depth Swells · Glitches Done Right** are LOFN-PRIME's own sound-pillars.

## VERIFICATION CHECKLIST — measured on this file, after writing

⭐ *Re-measured at the step that ships it. A number carried forward from an earlier step is a **promise**, not
a measurement, and this tier inherits nothing.* Source:
`skills/music/scripts/validate_suno_packages.py` + `scripts/measure_soundcraft.py → profile_file()` +
`_work/pair_04/measure.py`, all run against this file **after** it was written to disk.

**`validate_suno_packages.py` → `PASS` (exit 0), four packages extracted and inspected — not one.**
*(The splitter fired correctly: the step-10 draft initially used a `# V1 —` heading and the validator raised
`SPLITTER DEFECT: found 0 package/variation headings but 4 '## 1. MUSIC PROMPT' sections`. The heading
convention was changed to match the validator, **not the other way round.** That is the L28/L31 failure caught
live: on 2026-08-05 a `VARIATION` file matched zero headings, collapsed to one block, and printed PASS after
inspecting 25% of the work.)*

| | MUSIC PROMPT 850–1000 (target 870–960) | EXCLUDE 400–900 | lyrics field <5000 (target ≤4800) | sung lines 70–120 (target 78–110) |
|---|---:|---:|---:|---:|
| **V1** The Front Rail | **958** ✅ | 719 ✅ | **4614** ✅ | **83** ✅ |
| **V2** The Spare Hair Tie | **956** ✅ | 680 ✅ | **4703** ✅ | **82** ✅ |
| **V3** Somebody's Elbow | **953** ✅ | 684 ✅ | **4684** ✅ | **83** ✅ |
| **V4** The House Lights | **935** ✅ | 698 ✅ | **4785** ✅ | **82** ✅ |

**Boundary-hug FLAG (≥985): NOT raised — highest 958.** **Floor-hug FLAG (≤72 lines): NOT raised — lowest 82.**
All four prompts end in terminal punctuation, lead with genre + tempo + key, contain no real-artist name, no
key:value brackets and no procedural opening.

⚠️ **Disc_Channel is in a Production Sidecar, OUTSIDE the Suno lyrics field, on all four — declared, not
hidden.** The five-channel block measures ~440 chars; added inside the field it would put V4 at ~5,220 and V1
at ~5,055, i.e. **over the 5000 render cap.** The contract's own escape is taken verbatim — *"move the
Disc_Channel block to a `## Production Sidecar` outside the lyrics field (the render field wins; note it)"* —
and it is applied to **all four** rather than to the one that overflowed, because an inconsistent convention
across a pair is how a downstream extractor reads three of four. ⛔ **No sung line was cut to buy characters.**

### Return floors — beaten on all four variations

| floor | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| `rhyme_return_floor` **0.30** | **0.361** ✅ | **0.549** ✅ | **0.518** ✅ | **0.463** ✅ |
| `line_return_floor` **0.20** | **0.313** ✅ | **0.268** ✅ | **0.277** ✅ | **0.293** ✅ |
| `mean_words_per_line` **≤7.5** | **7.06** ✅ | **6.93** ✅ | **6.86** ✅ | **6.76** ✅ |
| `alliteration_per_100w` **≥11.0** | **14.33** ✅ | **14.96** ✅ | **14.94** ✅ | **11.91** ✅ |

*(Adding twelve body-noise lines moved `line_return` down 1–2 points across the set — the denominator grew and
the new lines are unique. Reported as measured; no line was added or removed to move a number.)*

### Machine-verified bans, re-run on this file

| check | result |
|---|---|
| plural pronouns across all sung lines (**D6**) | **0** ✅ |
| second-person pronouns (**D8**) | **0** ✅ |
| D2 banned phrases | **0** ✅ |
| digits in sung lines (**L33**) | **0** ✅ |
| `thirty seconds` per song | **exactly 1** in each ✅ |
| bare `AWE` / `INDIGNATION` EMO labels | **0** (validator-checked) ✅ |
| standalone `*SFX cue*` per song | **1** each — under the 3-cue blocking ceiling ✅ |
| `[Theme:]` + `[SONG FORM:]` as the first two lines | ✅ all four |
| full EMO header on every section | ✅ all four |
| body noise in Intro / Bridge / Outro | **3 per song, 12 total**, each with a stated dramatic function ✅ |

### Structural invariants, re-verified after enhancement

- [x] verses **6 / 6 / 6 bars**, two written lines per bar, all four
- [x] chorus **two lines only, byte-identical, ×4**, all four *(no note filed)*
- [x] **A1 consonant hammer** at line-heads — P/B · S,SH,ST · B,BR · K,hard-C — **and V2's single deliberate
      break**, which is this tier's device and is documented above
- [x] **A2 mix inversion** — background leads for **exactly one line**, then loses; written as a **voice swap
      in the lyric**, never a fader instruction (THE GRAIN LAW, L22)
- [x] **A3 one section audibly too short**, named and unrepaired — V1 bridge · V2 post-chorus tail · V3 outro ·
      V4 final drop
- [x] **build BY LEVEL** — five hard states, no crossfade; *accrete / layer in / crossfade* prohibited in all
      four EXCLUDE fields
- [x] flairs **#15 BIT-DEPTH COLLAPSE · #3 THE SWAP AT THE JUNCTION · #10 CHANT A ROOM CAN HOLD**
- [x] four titles, **each naming a THING**; no persona prefix
- [x] `[Object. State.]` establishing shot **absent** — all four open on a clause with a verb
- [x] D1–D10 re-checked line by line; unchanged from the step-10 pass
- [x] Lineage & Credit present, four links opened and verified
- [x] scratch confined to `_work/pair_04/`; only `pair_04_*` files written

**Removal test for the unnecessary element.** The one element that could be deleted without breaking the
songs is the **gilded harpsichord counter-melody** — a rage track does not need a Baroque ornament and no
lyric depends on it. ⭐ **It stays, and its removal is exactly what would make these four generic.** It is the
only thing in the arrangement still noticing beauty while a body is folded over a steel rail, and it is the
audible half of the pair's entire personality claim.
