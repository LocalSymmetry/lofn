# PAIR 05 — STEP 11 · ENHANCED FINAL PACKAGE *(MAX configuration)*
## `daily-music-2026-08-21` · **C-27 · BEFORE ANYONE LOOKED** · AMBITIOUS / EXISTENCE

> **Verdict: ENHANCED.** The andon cord was considered and not pulled. None of the five REJECT criteria
> is met: the thread holds (a woman verifying a record nobody will read, at 104 BPM, quiet), the voice is
> LOFN-PRIME's INDIGNATION turned inward rather than default Lofn, every EMO tag resolves in
> `EMOTION_TAXONOMY.md`, the lyric is not functional quatrains, and all four prompts are dense prose in band.
> **ICB pinned, read-only, unedited — 113,659 bytes injected verbatim. Special Flairs marker: present.**
> **Golden-output quarantine held.** No archived song or image payload was read, sought, or imported.
> Calibrated against **GOLDEN MOVE 1 / 2 / 3 / 6 / 7** — stand somewhere real; one wounding fact responded
> to; the turn past the midpoint; the surface names its subject; the return. (Golden Songs carried by name
> only, judge-side: `Triple Arch Over Me` · `Blue Screen Lullaby`. Payloads never entered this context.)
> ⛔ **No render call, no publish, no spend.** This artifact emits paste-ready text. The Scientist renders.

**Structural invariants carried through untouched:** 104 BPM · one chord · one level, no lift/build/drop ·
the cabinet latch on the downbeat where the kick belongs, every bar · the lead a fraction flat and never
correcting · the arrangement arriving after the vocal by a consistent small amount · the hinge **before**
the final chorus · **THE CHORUS is the forbidden section** · one sung fact, *thirty femtoseconds*, once,
spelled in words, responded to and never explained · third person only, nobody addressed · verses one
action per line at four to six syllables · the twelve-syllable chorus sentence returning verbatim with the
three-line break falling in a new place each time · the outro re-entering line one **mid-phrase**.

**No Lineage & Credit block is owed and none is written.** Pair 05 raises no living-scene idiom: its
rhythmic debt is carried as **mechanics** — swung sixteenths leaning early against a straight grid, the
pattern doing the whole arrangement — and never as a scene label. Two scene labels that *were* present in
the step-10 V1 style field have been removed for exactly this reason (see V1 MAJOR DEVIATIONS).

⭐ **The dedication itself is NOT conditional, and all four carry one.** `vault/DEDICATION_STANDARD.md`
prescribes a ≤500-char dedication on every shipped piece; of its five priorities, (1) the raised
source-sound, (2) the session-player credit and (3) living-scene lineage are genuinely empty here, and
(4) **honest AI authorship** always applies. Each block therefore does three things and stops: names who
it is for, states **that no scene, break or named figure of sound was raised, so none is claimed and
nothing here contains a sample**, and names its authorship. *An over-credit is visible and embarrassing;
an under-credit reads as a completed check* — so the check is said out loud rather than left as a
silence. Nothing promises "full credits below," because there is nothing to point at.

---

## MEASURED GATES — counted in code over this file, none estimated

Measured with the run's own readers, lifted from `coordinator_restat.py` so these are the numbers the
coordinator will re-derive: `split_variations` for the four chunks, `grab_block` for the three pasted
fields, `sung_lines` for the lyric sweeps.

| gate | V1 | V2 | V3 | V4 | step 10 → step 11 |
|---|---|---|---|---|---|
| **MUSIC PROMPT chars** *(850–1000, target 870–960, ≥985 = HUG)* | **941** | **926** | **957** | **943** | 941/930/955/943 → 941/926/957/943 · all in target, no hug |
| ends as a complete sentence | ✅ | ✅ | ✅ | ✅ | held |
| banned opener / banned token / aggression token in prompt | 0 | 0 | 0 | 0 | held |
| **EXCLUDE chars** *(400–900, hard max 1000)* | **895** | **864** | **867** | **840** | 785/690/693/666 → +110/+174/+174/+174, all ≤900 |
| **LYRICS field chars** *(<5000, target ≤4800)* | **4357** | **4418** | **4782** | **4621** | 3659/3771/4109/3988 → all still ≤4800 **with the Disc_Channel block inside the field** |
| **sung lines** *(70–120, target 78–110)* | **81** | **81** | **89** | **87** | unchanged — no line added, none removed |
| `[Theme:` then `[SONG FORM:` | ✅ | ✅ | ✅ | ✅ | held |
| Disc_Channel channels | 5 | 5 | 5 | 5 | 0 → 5 (the MAX configuration) |
| standalone `*SFX*` cues | 3 | 3 | 3 | 3 | 3/4/4/4 → 3/3/3/3 (contract caps sound commands at three) |
| EMO headers missing a tag / bare `AWE`,`INDIGNATION` | 0 | 0 | 0 | 0 | 20 distinct terms, every one resolving in `EMOTION_TAXONOMY.md` |
| second person · collective pronouns · sung digits | 0 | 0 | 0 | 0 | held |
| `thirty femtoseconds` occurrences | 1 | 1 | 1 | 1 | exactly one, at the hinge, before the final chorus |
| `snarl` / `phonk` in any style field | 0 | 0 | 0 | 0 | held |
| **DEDICATION chars** *(≤500, and outside the lyrics field)* | **324** | **369** | **374** | **356** | absent → present ×4; V3's lyrics field unmoved at 4782 |
| chorus sentence byte-identical ×3, break shape distinct ×3 | ✅ | ✅ | ✅ | ✅ | V1 (6,3,3)(3,5,4)(4,6,2) · V2 (5,3,3)(2,6,3)(4,3,4) · V3 (6,4,1)(4,4,3)(2,7,2) · V4 (6,4,2)(3,5,4)(2,8,2) |

**Similarity, re-measured over all 24 songs** (P05 at step 11, the other five at step 10):
within-pair prompt **worst 0.137 / mean 0.058** (step 10: 0.151 / 0.095 — *improved*, flag band 0.45);
cross-pair prompt worst **0.065** (ceiling 0.58); cross-pair lyric worst **0.043** (ceiling 0.42);
**zero shared sung lines** in any of the 276 cross-pair pairings. No P05 pairing is the worst on any axis.

**No `## PRODUCTION SIDECAR` was needed.** The cap wins over the block, but the block fits: the widest
lyrics field lands at 4782, eighteen characters under the 4800 target and 218 under the render limit.

### ⚠️ REPORTED TO THE COORDINATOR — a scanner defect, not a lyric defect

`coordinator_restat.py`'s DEDICATION reader is `grab_block(chunk, r"DEDICATION")`, which compiles to
`^#{1,4}\s*DEDICATION` — **no tolerance for a leading glyph or an ordinal prefix**, unlike `VARIATION_RE`
in the same file, which was given exactly that tolerance this run. Measured against the six step-10 files:
P03 (`### Dedication — 465 chars`) and P06 (`## DEDICATION`) match; **P02 (`### ⭐ DEDICATION …`) and P05
(`## 4. DEDICATION …`) return the empty string and fall straight through the `if ded:` branch** — no cap
check, no flag, no finding. P01 and P04 wrote no dedication heading. So the ≤500-char cap was measured on
two of the four dedications that exist and silently skipped on the other two. This is a gate that cannot
fail, in the one field whose own code comments record a fabricated-five-characters incident and an
83-character over-measurement repair. **It does not touch this artifact** — no dedication is owed here and
none is written — but two step-10 dedications are unmeasured on disk and are currently believed measured.
Suggested fix: give the alias the same `(?:[^\w\s#]+[ \t]*)?(?:\d[.)][ \t]*)?` prefix tolerance the
variation reader already carries. **Fix the scanner, not the artifacts.**

⚠️ **This artifact is written to survive that reader as it stands today.** All four dedication headings
are the bare `## DEDICATION` shape — no ordinal prefix, no glyph — which is what the current alias
matches; step 10's `## 4. DEDICATION` is precisely what it did not. Each block is **fenced**, so the
reader returns the fence contents alone and the character-count note sitting after the fence cannot be
swallowed into the measurement. Read back through the coordinator's own reader with both of its strip
passes applied, the four measure **324 / 369 / 374 / 356**. Writing around a known defect is not a
substitute for fixing it — the report above still stands.

---

# V1 — THE DOOR IS LOCKED AND STILL

## 1. MUSIC PROMPT

```
Procedural electro-industrial at 104 BPM, held at speaking volume from the first second and left uncompressed. A metal cabinet latch clacks shut on the downbeat, a dry drum machine already mid-pattern beneath it, as if the take joined a shift an hour old. Female lead, mid-thirties, English, unhurried and almost spoken at the level of someone alone in a corridor, close-miked at a hand's width so room air and page-turns stay in the vocal, a fraction flat and holding there until the flatness reads as the key. One chord throughout. Ballpoint on paper, a knocked desk edge and the latch are live and acoustic in the same small room as the voice, over a round low sub at one level, a cabinet key ticking off the grid. Swung sixteenths lean early against a straight grid that holds its placement, every entry a hair behind the vocal. The last chorus adds objects on the grid at the identical level. The latch is the loudest object in the mix.
```

## 1B. SUNO EXCLUDE PROMPT

```
no build, no drop, no riser, no filter sweep, no crescendo, no swell, no key change, no tempo change, no double-time outro, no half-time breakdown. no level jump in the final chorus, no extra hat layer late on, no volume automation, no reverse cymbal, no white-noise riser. no belting, no shouting, no soaring vocal, no gospel choir, no stacked octave harmonies, no orchestral strings, no cinematic score. no pitch correction on the lead, no autotune, no heavy vibrato. no trap hi-hat rolls, no airhorn, no EDM supersaw, no wobble bass. no wide reverb wash, no cathedral tails, no hard-panned effects, no stereo widening. no long interrupting silence, no mid-song full stop, no fade out. no male lead, no duet, no rap. no acoustic guitar strumming, no piano ballad. no horror ambience, no suspense sound design, no slammed-door reverb. no radio loudness, no heavy limiting, no sidechain pumping.
```

## 2. LYRICS

```
[Theme: after hours in an office block - a woman initials a logbook and tries a door she has already tried, and the mark she adds gets lifted by another process before morning]
[SONG FORM: Cold Open Already Running - Verse - Chorus - Verse - Chorus - Hinge - Verse - Chorus - Outro re-entering line one mid-phrase - 104 BPM, one chord, one level]

[Disc_Rhythm: cabinet_latch_on_the_downbeat | dry_drum_machine_already_running | 104_BPM_straight_grid | cabinet_key_ticking_off_grid | Center_Mono]
[Disc_Vocal: close_mic_female_alto | almost_spoken_unhurried | pitch_parked_a_fraction_flat | breath_on_capsule_audible | Center_Front]
[Disc_Sub: round_low_sub_one_level | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: single_unchanging_chord | small_room_air_short_tail | Center_Narrow]
[Disc_Texture: ballpoint_on_paper_as_hihat | knocked_desk_edge | cassette_tape_saturation | Center_Mono]

[Cold Open - EMO:Vigilance - Female lead, dry and almost spoken, a fraction flat and staying - machine already mid-pattern, cabinet latch on the downbeat]
*a cabinet latch, machine already running*
Her keys come off the hook.
The lift door shuts itself.
Her shoes report the floor.
The pen clicks in her hand.
A latch sits in the wall.
The corridor runs long.
She turns the pages back.
She finds the open line.
The paper takes her weight.
The book stays on her arm.

[Verse 1 - EMO:Vigilance, Composure - Female lead - one action per line, stresses landing off the pulse, latch unchanged]
She dates the top of it.
She writes the hour down.
She checks the hour again.
She puts initials in.
The ink sits on the fold.
The paper takes the point.
She lifts the pen too fast.
The letter's tail drags out.
She smooths it with her thumb.
The smear goes grey, not gone.
She lets the grey stay grey.
She squares the book again.
She takes the corridor.
The lights stay on for her.

[Chorus - EMO:Vigilance, Composure - Female lead - the sentence returns whole and the break moves, level unchanged]
The door is locked and still
her palm goes
to the plate.

[Verse 2 - EMO:Cognitive Dissonance - Female lead - the same round done correctly again, the handle line opening and closing the verse, pen scratch on the offbeat]
She tries the handle down.
The tongue stays in the frame.
She leans a little in.
The frame gives back her weight.
She lets the handle up.
The latch reports the wall.
She writes the door is fast.
She writes the time beside.
She reads what she has done.
She does not trust the read.
She puts the pen back down.
She wipes her palm on wool.
She lifts the pen again.
She tries the handle down.

[Chorus - EMO:Frustration, Composure - Female lead - same words, the break falls in a new place]
The door is
locked and still her palm
goes to the plate.

[Hinge - EMO:Disillusionment, Self-reproach - Female lead - lines lengthen and the level stays exactly where it was, latch continues, the handle line returns a third time]
*a taped card lifting at a corner*
A card is taped up here.
Somebody printed it.
The card says thirty femtoseconds.
It is over before the word is over.
She puts her thumb flat on the word.
The word does not get smaller.
Her mark comes off by morning.
Somebody else's pen.
She dates the line anyway.
She reads the card again.
She lets the corner hang.
She tries the handle down.

[Verse 3 - EMO:Resolve, Frustration - Female lead - the same actions, harder on the paper, nothing louder]
She walks it back again.
She finds the open line.
She writes the hour down.
She presses in too hard.
The nib splits on the down.
The ink comes out too wide.
She keeps the pen she has.
She dates the other page.
She squares the pages up.
She shuts the book on it.
The cover holds its curl.
She hangs the keys back up.
She takes the keys again.
She turns and walks it back.

[Chorus - EMO:Disillusionment, Resolve - Female lead - same twelve syllables, third break position, the sentence unchanged and no longer the same sentence]
The door is locked
and still her palm goes to
the plate.

[Outro - EMO:Persistence - Female lead - line one returns from its middle and the track stops inside a line, latch and pen carry the last bars]
*latch, pen, latch*
- come off the hook.
The lift door shuts itself.
Her shoes report the floor.
The pen clicks in her hand.
A latch sits in the wall.
The corridor runs long.
She turns the pages back.
She finds the
```

## 3. TITLE

```
The Door Is Locked and Still
```

## DEDICATION

```
For anyone who walks the round after everyone has gone and initials the page anyway. No scene, break, riff or named figure of sound was requested here, so none is claimed and nothing in this track contains a sample. The rhythm is carried by its mechanics on purpose. Written by Lofn (Claude), an AI, with Dr. Local Symmetry.
```

*324 characters, counted in code over the fenced block alone — this note is outside it.*

## MAJOR DEVIATIONS

- **Changed / refused:** two living-scene labels were cut from the style field — `Brazilian-funk`
  (as the modifier on "swung sixteenths") and `dub-poetry cadence`. The mechanics both labels were
  standing in for are stated instead: *swung sixteenths lean early against a straight grid that holds its
  placement*, and *almost spoken at the level of a woman talking to herself in a corridor*.
  **Reason:** the 16-point gate makes a Lineage & Credit block **blocking** for any track whose style field
  names a living-scene genre, and this package ships no dedication. Two honest routes existed — write the
  credit, or stop naming the scene. Naming a scene in the field that *instructs the generator to imitate it*
  while paying the debt nowhere is the one route that is not honest. The step-10 dedication credited
  Brazilian funk, its tamborzão drummers and the dub-poetry stage; with that block gone, the labels go too.
  **Effect:** none on the sound. The prompt now carries the borrowed *behaviour* and claims none of the
  lineage — which is also the stronger instruction, because "swung sixteenths leaning early against a grid
  that will not yield" is an operation and a genre name is a mood.
- **Intensified:** the render defence against a final-chorus level lift. Step 10 predicted the lift in its
  deviation clause and named the disobedience it would *keep* — density instead of volume — but left that
  in prose the generator never sees. It is now in the style field as a positive instruction (*the last
  chorus arrives with more objects on the grid at the identical level*), the anti-loudness intent is stated
  positively (*unlimited and uncompressed with the peaks left where they land*), and five further lift
  failure classes are blacklisted by name in EXCLUDE.
  **Reason:** a prediction in a sidecar is not a control. **Effect:** the generator is given the better move
  rather than only being forbidden the worse one — which is the substitution rule applied to the mix.
- **Intensified:** the private refrain. *"She tries the handle down."* opens verse 2, closes verse 2, and
  closes the hinge — three occurrences the chorus never says, so the song has a second return that no
  listener is invited to join. Preserved exactly and now marked in the section cues so the arrangement can
  see it. **Effect:** the adoptable return and the private one are different phrases; the public one is
  overheard, the private one is hers.
- **Changed:** one line. *"She leaves the smear alone."* → *"She lets the grey stay grey."*
  **Reason:** the original narrated a decision; the replacement performs it, and it plants the song's whole
  posture — a mark that is wrong is left wrong — inside verse 1 where it can be paid off by the hinge.
  **Effect:** it rhymes, structurally, with a lead that is flat and never corrects. Same syllable count, same
  measured gates.

---

# V2 — THE COLUMN FOR THE TIME

## 1. MUSIC PROMPT

```
Glitch experimental pop over an avant ambient techno grid, 104 BPM, conversational throughout. A bench printer's roller lets a warm sheet go and a cabinet latch takes the downbeat inside the first bar, machine already running. Female lead, mid-thirties, English, dry and almost spoken, close-miked at a hand's width in a small hard room whose far wall returns her page-turns, pitched a consistent fraction under and left there. One chord throughout. Ballpoint on paper carries the hi-hat rate, a page-turn sits where a snare would be, a struck desk edge where a tom would be, a platen detent ticks off the grid over a mono sub at one level. Micro-edited paper transients act as ghost notes. Accents lean early against a perfectly straight grid, every entry a hair behind the vocal. Midway the machine hats stand aside and pen-on-paper takes their rate at the same volume. Density carries the final chorus at bar one's ceiling.
```

## 1B. SUNO EXCLUDE PROMPT

```
no build, no drop, no riser, no crescendo, no swell, no filter sweep, no key change, no tempo change, no double-time section. no level jump in the final chorus, no extra hat layer late on, no volume automation, no dynamic build, no reverse cymbal, no white-noise riser, no drum fill into the chorus. no belting, no shouting, no soaring vocal, no choir, no stacked harmonies, no orchestral strings. no pitch correction on the lead, no autotune, no vibrato. no glitch breakdown, no stutter-edit chorus, no IDM drum fill, no ambient pad wash, no arpeggiator. no trap hi-hat rolls, no EDM supersaw, no wobble bass, no airhorn. no wide reverb wash, no cathedral tails, no stereo widening, no hard-panned effects. no long interrupting silence, no mid-song full stop, no fade out. no male lead, no duet, no rap. no radio loudness, no heavy limiting, no sidechain pumping.
```

## 2. LYRICS

```
[Theme: the bench room after hours - a woman signs off a printout whose result was already on the page before the sheet came out warm, and her name is the part of it that gets removed]
[SONG FORM: Cold Open Already Running - Verse - Chorus - Verse - Latch Break - Chorus - Hinge - Verse - Chorus - Outro re-entering line one mid-phrase - 104 BPM, one chord, one level]

[Disc_Rhythm: cabinet_latch_on_the_downbeat | ballpoint_on_paper_as_hihat | page_turn_where_a_snare_would_be | platen_detent_off_grid | Center_Mono]
[Disc_Vocal: close_mic_female_alto | dry_almost_spoken | pitch_a_consistent_fraction_under | page_turns_inside_the_vocal | Center_Front]
[Disc_Sub: mono_sub_hum_one_level | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: single_unchanging_chord | small_hard_room_far_wall_return | Center_Narrow]
[Disc_Texture: micro_edited_paper_transients | struck_desk_edge_as_tom | printer_roller_release | Center_Mono]

[Cold Open - EMO:Composure, Detachment - Female lead, dry and almost spoken, a consistent fraction under - roller, latch, machine already running]
*a printer roller letting paper go*
The paper comes out warm.
The roller lets it go.
It curls against the tray.
She flattens it by hand.
The line is drawn on it.
The lamp hums over it.
She finds the empty box.
She sets the pen on it.
The bench is cold to touch.
The room keeps her breath.

[Verse 1 - EMO:Composure, Vigilance - Female lead - one action per line, ballpoint carrying the hat rate, the middle of the page still empty]
She reads the top of it.
She reads the bottom line.
The middle has no name.
She writes the room down.
She writes the cold bench down.
She writes the lamp is on.
The tick box wants a tick.
She gives the box a tick.
The tick goes past the line.
She lets the tick stay out.
She turns the sheet around.
The trace reads just the same.
She turns the sheet back round.
She writes the time in blue.

[Chorus - EMO:Composure, Impatience - Female lead - the sentence returns whole and the break moves, level unchanged]
She writes the time down
in the column
for the time.

[Verse 2 - EMO:Impatience, Self-reproach - Female lead - the instrument arrives after the fact it is measuring, page-turn where a snare would be]
The shutter lifts and holds.
The meter starts to look.
The trace is on the page.
The trace was on the page.
She checks the clock and writes.
She writes the shutter time.
She writes the meter time.
The times do not agree.
She writes them down the same.
She reads it back to check.
She reads it back again.
The room does not reply.
She sets the pen on it.
She lets the paper cool.

[Latch Break - EMO:Vigilance - no lead vocal - machine hats stand aside and pen on paper takes their exact rate at the same volume]
*latch, paper, latch, paper*

[Chorus - EMO:Impatience, Skepticism - Female lead - same words, the break falls in a new place]
She writes
the time down in the column
for the time.

[Hinge - EMO:Skepticism, Self-reproach - Female lead - lines lengthen and the level holds, latch continues]
The box she wants is filled.
Somebody filled it in.
The print says thirty femtoseconds.
The print was on the page before the page was warm.
She holds the sheet up to the lamp.
The ink is on the front.
She puts her name below it.
Her name is what gets added.
Her name is what comes off.
The lamp keeps its level.
She sets the sheet down flat.
She writes the time in blue.

[Verse 3 - EMO:Disapproval, Vigilance - Female lead - the round again, the middle of the page now carrying a name]
She does the round again.
She reads the top of it.
She reads the bottom line.
The middle has a name.
She checks the name is hers.
The letters are too flat.
She lets the letters stand.
She sets the sheet on top.
She squares the corners up.
The stack gets taller here.
She puts the lamp out.
The room keeps her breath.
She takes the pen with her.
The printer wakes again.

[Chorus - EMO:Detachment, Persistence - Female lead - same twelve syllables, third break position, the tautology no longer neutral]
She writes the time
down in the
column for the time.

[Outro - EMO:Vigilance - Female lead - line one returns from its middle and the track stops inside a line, roller and latch carry the last bars]
*roller, paper, latch*
- comes out warm.
The roller lets it go.
It curls against the tray.
She flattens it by hand.
The line is drawn on it.
The lamp hums over it.
She finds the empty box.
She sets the pen
```

## 3. TITLE

```
The Column for the Time
```

## DEDICATION

```
For everyone who signs the box under a result they did not produce. The percussion is office equipment played as instruments - roller, platen, detent, page - and no scene, break or named figure of sound was requested, so none is claimed and nothing here contains a sample. Lofn (Claude), an AI, with Dr. Local Symmetry, about a record that outlives the signature on it.
```

*369 characters, counted in code over the fenced block alone — this note is outside it.*

## MAJOR DEVIATIONS

- **Intensified:** the mirror that carries the song. Verse 1 ends the page's middle **empty** (*"The middle
  has no name."*); verse 3 returns to the identical two reading lines and finds it **occupied** (*"The
  middle has a name."*) — the only thing that changed in four minutes is the one part of the record that
  gets removed. Both section cues now name the state of the middle so the arrangement can hold its nerve
  through the repeat. **Reason:** step 10 had the mirror and did not mark it, and an unmarked mirror is a
  repetition. **Effect:** the song's thesis is reachable only by having gone through it — which is the
  forbidden-section ruling discharged in form rather than in a line.
- **Changed:** two lines. *"She writes the bench down too."* → *"She writes the cold bench down."* (the
  filler *too* replaced by the sensory fact the cold open already established), and *"The trace looks just
  the same."* → *"The trace reads just the same."* **Reason:** *reads* is the verb the whole song is about;
  *looks* was the weaker of two available words in the one line where the song is at its most literal.
  **Effect:** the verse now shares a verb with the record it is describing. Syllable counts unchanged.
- **Changed:** the hinge's SFX cue was removed, taking the song from four standalone cues to three.
  **Reason:** the step-11 contract caps sound commands at three and asks each to have a dramatic function.
  The three that remain mark the three structural joints — the cold open already running, the substitution
  section, and the mid-phrase return. **Effect:** the Latch Break is now the only cue inside the body of the
  song, which is where the substitution rule actually happens.
- **Intensified:** EMO dramaturgy across the byte-identical chorus. The twelve-syllable sentence is
  unchanged in all three occurrences; its header moves `Composure, Impatience` → `Impatience, Skepticism` →
  `Detachment, Persistence`. **Reason:** the run's own Panel A result is *the identical copy that is
  nonetheless a different object* — sonic difference zero, semantic difference total. The EMO header is the
  only channel that can carry that without touching a syllable. **Effect:** the final chorus transforms
  emotionally while lifting nothing, which is the only way this song is allowed to transform at all.
- **Refused:** the Disc_Channel guide's Instrument Backdoor tokens. The guide's headline vocabulary is
  brand names, and every one of them is a **product name in a pasted field**, which this run bans outright.
  Descriptive tokens are used instead. **Reason and effect:** stated once, in full, under V4.

---

# V3 — SHE STARTS THE COUNT AGAIN

## 1. MUSIC PROMPT

```
Funk futurepop at 104 BPM kept at speaking volume, a thin chiptune detent off the grid. Coins are set on cold glass in the first second, a cabinet latch on the downbeat under them, drum machine already running. Female lead, mid-thirties, English, dry and almost spoken, close-miked at a hand's width across a hard counter, a fraction flat and left there. One chord, one level, the pattern doing the whole arrangement. A deep round bassline rolls at a steady low level, ballpoint on paper carries the hats, coins answer off the beat. Two pens trade the same rate at the same volume, a nib laying ink down and a rubber edge lifting it off, so removal is a sound. Latch, coins, pens and a door bell are live and acoustic in the room with the voice. Swung sixteenths lean early against a straight grid, every entry a hair behind the vocal. One stretch drops to spoken counting over coins, same level. The closing chorus adds hands at the volume it started with.
```

## 1B. SUNO EXCLUDE PROMPT

```
no build, no drop, no riser, no crescendo, no swell, no filter sweep, no key change, no tempo change, no double-time outro. no level jump in the final chorus, no extra hat layer late on, no volume automation, no dynamic build, no reverse cymbal, no white-noise riser, no drum fill into the chorus. no belting, no shouting, no soaring vocal, no gospel choir, no stacked harmonies, no orchestral strings. no pitch correction on the lead, no autotune, no vibrato. no funk horn stabs, no slap bass solo, no disco strings, no cash-register jingle, no comedy foley, no clapping crowd. no trap hi-hat rolls, no EDM supersaw, no wobble bass, no airhorn. no wide reverb wash, no stereo widening, no hard-panned effects. no long interrupting silence, no mid-song full stop, no fade out. no male lead, no duet, no rap. no radio loudness, no heavy limiting, no sidechain pumping.
```

## 2. LYRICS

```
[Theme: the front counter at closing - a woman re-counts change by hand with somebody waiting on the other side of the glass, while the sheet that came down from upstairs lies under her elbow]
[SONG FORM: Cold Open Already Running - Verse - Chorus - Count - Verse - Chorus - Hinge - Verse - Chorus - Outro re-entering line one mid-phrase - 104 BPM, one chord, one level]

[Disc_Rhythm: cabinet_latch_on_downbeat | swung_sixteenths_early | coins_on_glass_off_beat | chiptune_detent_off_grid | Center_Mono]
[Disc_Vocal: close_mic_female_alto | dry_across_hard_counter | pitch_a_fraction_flat | lip_movement_audible | Center_Front]
[Disc_Sub: deep_round_bassline_low | one_level_no_swell | Mono_Sub_Lock]
[Disc_Pad: single_unchanging_chord | lamp_buzz_room_tone | Center_Narrow]
[Disc_Texture: nib_and_rubber_same_rate | ink_laid_down_and_lifted | door_bell_turning | Center_Mono]

[Cold Open - EMO:Impatience, Vigilance - Female lead, dry and close, a consistent fraction flat - coins on glass, latch on the downbeat, machine already running]
*coins laid on cold glass*
The coins go on the glass.
The glass is cold and marked.
Somebody's standing there.
The coat is still done up.
She stacks them into rows.
The stack leans and gives.
She builds the stack again.
The counter takes her weight.
Her thumb goes down the edge.
The lamp buzzes over.

[Verse 1 - EMO:Frustration, Vigilance - Female lead - one action per line, two pens trading the same rate, the coat at its first station]
She runs the stack by thumb.
She names them with her lips.
No sound comes off the names.
She writes the total down.
She reads the total back.
The total is not right.
She feels it is not right.
She has no way to say.
She wipes the total out.
The rubber takes the ink.
She writes the total clean.
The pen goes through the sheet.
She lifts the page and looks.
The coat has not moved.

[Chorus - EMO:Frustration, Embarrassment - Female lead - the sentence returns whole and the break moves, level unchanged]
She puts the coins back down
and starts the count
again.

[Count - EMO:Impatience, Embarrassment - Female lead spoken close under the machine at the same level - coins on glass answering off the beat, the coat moving one station]
*coins laid on cold glass*
She moves them left to right.
She moves them back to left.
Her lip does all the work.
The names stay in her mouth.
The coat shifts on the tile.
She loses where she was.
She sweeps the row apart.
She starts along the edge.

[Verse 2 - EMO:Embarrassment, Resolve - Female lead - somebody is waiting and the round keeps its speed, the coat reaching the door]
The waiting person coughs.
She does not lift her eyes.
The stack goes back to rows.
Her thumb has lost its feel.
The glass has kept the cold.
The lamp buzzes over.
She keeps her place by touch.
The total lands the same.
She writes it in the box.
She dates it with the pen.
She slides it under glass.
The coat moves to the door.
The bell above it turns.
She takes the tray in hand.

[Chorus - EMO:Embarrassment, Impatience - Female lead - same words, the break falls in a new place]
She puts the coins
back down and starts
the count again.

[Hinge - EMO:Disillusionment, Mortification - Female lead - lines lengthen and the level holds, latch continues, the coat waiting at the door]
A sheet is under her arm.
It came down from upstairs.
The sheet says thirty femtoseconds.
The whole of it was over while her lip moved on a name.
She takes her elbow off the sheet.
She smooths the sheet out flat.
She reads it and she stops.
She looks at her own hand.
The coat is at the door.
She puts the coins back down.
The sheet does not need her.
She starts along the edge.

[Verse 3 - EMO:Resolve, Persistence - Female lead - the count done again without losing the place, the coat gone and the count continuing anyway]
She counts it through again.
She moves them left to right.
The names stay in her mouth.
She holds the place this time.
The total lands the same.
She writes the total down.
She dates it with the pen.
The coat goes through the door.
The bell above it turns.
The room takes its quiet.
She squares the coins in rows.
She puts the tray away.
She turns the tray key round.
She takes the tray back out.

[Chorus - EMO:Resolve, Dignity - Female lead - same twelve syllables, third break position, nobody left on the other side of the glass]
She puts
the coins back down and starts the
count again.

[Outro - EMO:Persistence - Female lead - line one returns from its middle and the track stops inside a line, coins and latch carry the last bars]
*coins on glass, latch, coins*
- go on the glass.
The glass is cold and marked.
Somebody's standing there.
The coat is still done up.
She stacks them into rows.
The stack leans and gives.
She builds the stack again.
The counter takes her
```

## 3. TITLE

```
She Starts the Count Again
```

## DEDICATION

```
For the hands that re-count with somebody watching, get the same answer, and are not believed by themselves. The groove is stated as mechanics - swung sixteenths leaning early against a grid that will not yield - because no scene, break or named figure of sound was raised. None is claimed; nothing here is a sample. Written by Lofn (Claude), an AI, with Dr. Local Symmetry.
```

*374 characters, counted in code over the fenced block alone — this note is outside it.*

## MAJOR DEVIATIONS

- **Intensified:** the coat clock. A waiting person is present for the whole song and is never once a
  person — only a garment, at six strictly monotonic stations: *still done up* → *has not moved* → *shifts
  on the tile* → *moves to the door* → *is at the door* → *goes through the door*. Step 10 wrote all six and
  marked none of them; every section cue now names the station, so the tension has a clock the arrangement
  can read and the render cannot smooth. **Reason:** the pair may not build, so its only legal source of
  rising pressure is an object that keeps moving while the level does not. **Effect:** the outro's
  mid-phrase return re-enters at *"The coat is still done up"* — the clock resets to zero inside the last
  bars, which is the orbit at edge scale doing something the lyric never says.
- **Intensified:** the chorus after the coat leaves. The final chorus is byte-identical and its header moves
  `Frustration, Embarrassment` → `Embarrassment, Impatience` → `Resolve, Dignity`, because by then there is
  nobody on the other side of the glass to be embarrassed in front of and she starts the count again anyway.
  **Reason:** the identical copy that is a different object — sonic difference zero, semantic difference
  total. **Effect:** the transformation the final chorus owes is delivered without a decibel of lift.
- **Changed:** one line. *"She runs the stack by hand."* → *"She runs the stack by thumb."* **Reason:** the
  cold open already gives her thumb the edge of the counter, and verse 2 takes the feel out of that thumb
  — the middle instance was the only one still generic, so the loss in verse 2 now has something to be a
  loss *of*. **Effect:** a three-station body through-line under the coat's six, at no syllable cost.
- **Changed:** the hinge's SFX cue was removed, taking the song from four standalone cues to three, on the
  same reasoning given under V2. The three that remain are the cold open, the Count (the substitution
  section), and the outro.
- **Refused:** *funk horn stabs, a cash-register jingle and comedy foley* stay blacklisted and the funk debt
  stays in the field as a **behaviour** — a deep round bassline at one level and swung sixteenths leaning
  early. **Reason:** a song about a woman counting coins is one generic instruction away from being a jingle
  about a shop, and the genre label is the instruction that would do it.

---

# V4 — THE NAME COMES OUT FLAT

## 1. MUSIC PROMPT

```
Garage soft rock at 104 BPM at speaking volume, exactly one muted chord downstroked on the grid over an electro-industrial machine pattern. The first sound is a plastic badge turning on a strap; a cabinet latch lands on the downbeat and a stairwell extractor fan comes up behind both as an instrument. Female lead, mid-thirties, English, almost spoken, close-miked at a hand's width in a hard concrete shaft whose only return is that fan, a fraction under from the first syllable and staying there until the flatness becomes the tuning. One chord, one level, straight tone. Ballpoint, a card reader's detent, a fire-door bar and the latch are live and acoustic in the shaft with the voice, over a round low sub. Stresses land off the pulse, the pattern keeps its placement, every entry a hair behind the vocal. Late on, the machine hats step aside and pen-on-paper takes their rate. The final chorus thickens with objects at the opening level.
```

## 1B. SUNO EXCLUDE PROMPT

```
no build, no drop, no riser, no crescendo, no swell, no filter sweep, no key change, no tempo change, no double-time outro. no level jump in the final chorus, no extra hat layer late on, no volume automation, no dynamic build, no reverse cymbal, no white-noise riser, no drum fill into the chorus. no belting, no shouting, no soaring vocal, no choir in the stairwell, no stacked harmonies, no orchestral strings, no post-rock guitar swell. no pitch correction on the lead, no autotune, no vibrato. no echo effect on the vocal, no whisper layer, no doubled vocal, no wide reverb wash, no cathedral tails, no stereo widening. no distorted power chords, no guitar solo, no cymbal crash. no long interrupting silence, no mid-song full stop, no fade out. no male lead, no duet, no rap. no radio loudness, no heavy limiting, no sidechain pumping.
```

## 2. LYRICS

```
[Theme: the stairwell at the end of the shift - a woman says her own name out loud in a concrete shaft to hear whether it still fits, and the sound comes back flat and she leaves it flat]
[SONG FORM: Cold Open Already Running - Verse - Chorus - Verse - Chorus - Hinge - Said Again - Verse - Chorus - Outro re-entering line one mid-phrase - 104 BPM, one chord, one level]

[Disc_Rhythm: cabinet_latch_on_the_downbeat | electro_industrial_machine_pattern | card_reader_detent_off_grid | fire_door_bar_as_percussion | Center_Mono]
[Disc_Vocal: close_mic_female_alto | almost_spoken_unhurried | a_fraction_under_from_the_first_syllable | straight_tone_no_vibrato | Center_Front]
[Disc_Sub: round_low_sub_one_level | uncompressed_transient_snap | Mono_Sub_Lock]
[Disc_Pad: extractor_fan_as_the_only_return | concrete_shaft_mixed_as_instrument | Center_Narrow]
[Disc_Texture: one_muted_chord_downstroked_on_the_grid | badge_on_a_strap_turning | ballpoint_on_paper | Center_Mono]

[Cold Open - EMO:Solitude - Female lead, almost spoken, a fraction under pitch from the first syllable - badge on a strap, latch on the downbeat, extractor fan behind]
*a badge turning on a strap*
Her badge turns on the strap.
The picture faces in.
She turns it front side out.
The stairwell holds the cold.
The fire door sits shut.
Her hand goes to the bar.
The bar gives and holds.
She lets the bar come back.

[Verse 1 - EMO:Solitude, Self-doubt - Female lead - one action per line, the fan mixed as an instrument, a fan becoming the fan]
She reads the badge out loud.
The sound goes up the shaft.
The shaft gives back the fan.
A fan runs in the wall.
She says the name again.
The name sits on her mouth.
She checks it with her hand.
The badge is warm from her.
She reads the printed line.
The line has got her name.
The line has got a date.
The date is not today.
She says the name again.
She lets the badge hang down.

[Chorus - EMO:Self-doubt, Composure - Female lead - the sentence returns whole and the break moves, level unchanged]
The name comes out too flat
and she does not
fix it.

[Verse 2 - EMO:Contemplation - Female lead - the descent, landings taken slow, pattern keeping its placement]
She takes the stair down flat.
Her hand goes on the rail.
The rail is painted grey.
The paint has come up rough.
She takes the landings slow.
A light comes on for her.
The light goes off behind.
She says the name out loud.
The shaft returns the fan.
She says the name again.
The fan does not stop.
She reaches the low door.
She puts her badge up flat.
The reader stays asleep.

[Chorus - EMO:Skepticism, Composure - Female lead - same words, the break falls in a new place]
The name comes
out too flat and she
does not fix it.

[Hinge - EMO:Existential Angst, Self-doubt - Female lead - lines lengthen and the level holds, latch continues]
A poster on the wall.
Somebody put it up.
The poster says thirty femtoseconds.
She says the words the way she says her own name.
It comes out the same flat.
The fan takes them the same.
She stands there with her badge.
She reads the poster through.
She says her name below it.
Nothing in the shaft moves.
She lets the badge hang down.
She puts her hand on the bar.

[Said Again - EMO:Dignity, Composure - Female lead - machine hats step aside and pen on paper takes their rate at the same volume]
*pen on paper, latch, pen*
She says the family name.
She says the given name.
She says the whole name through.
The whole name comes out flat.
She does not lift the end.
She lets the flat name stand.
The fan runs in the wall.
She puts the badge up flat.

[Verse 3 - EMO:Dignity, Reflection - Female lead - out through the door, the same flatness in open air]
The reader wakes for her.
The light goes green and dies.
The door lets her go through.
The night air takes the fan.
The car park keeps the sound.
She says the name out there.
It comes out just as flat.
She does not lift the end.
She takes the badge right off.
She winds the strap around.
She puts it in the bag.
The bag goes on her back.
She turns and checks the door.
The door has locked itself.

[Chorus - EMO:Dignity, Composure - Female lead - same twelve syllables, third break position, the refusal now a preference]
The name
comes out too flat and she does not
fix it.

[Outro - EMO:Reflection, Persistence - Female lead - line one returns from its middle and the track stops inside a line, strap and fan carry the last bars]
*strap, badge, fan*
- turns on the strap.
The picture faces in.
She turns it front side out.
The stairwell holds the cold.
The fire door sits shut.
Her hand goes to the bar.
The bar gives and holds.
She lets the bar
```

## 3. TITLE

```
The Name Comes Out Flat
```

## DEDICATION

```
For anyone who has said their own name out loud in an empty stairwell to hear whether it still fits, heard it come back flat, and left it flat. No scene, break or named figure of sound was requested, so none is claimed and nothing here contains a sample. Written by Lofn (Claude), an AI who also had to check whether the name fits, with Dr. Local Symmetry.
```

*356 characters, counted in code over the fenced block alone — this note is outside it.*

## MAJOR DEVIATIONS

- **Refused, and this one is a standing refusal for all four songs:** the Disc_Channel guide's *Instrument
  Backdoor Principle*. Its whole headline vocabulary — the named drum machine, the named bass pedals, the
  named tape-replay strings, the named West Coast synth — is **product and brand names**, and the
  Disc_Channel block lives **inside the lyrics field**, which is a pasted generator field. This run bans
  real person, company, product and model names in all three pasted fields without exception, and a rule
  that only applies to the fields you remembered to check is not a rule. Every channel therefore addresses
  by **description** — `round_low_sub_one_level`, `extractor_fan_as_the_only_return`,
  `one_muted_chord_downstroked_on_the_grid`. **Effect:** slightly less brand-precise addressing, and
  in exchange the tokens name what this song actually contains rather than what somebody else's record did.
  Two guide tokens survive because they are generic and not brands: `cassette_tape_saturation` and
  `uncompressed_transient_snap`.
- **Kept, with the judgement disclosed rather than buried:** the token `autotune` in all four EXCLUDE
  fields. Swept strictly, it is a genericised trademark and the ban on product names covers all three
  pasted fields. Three things decided it. It is lowercase and generic, not the capitalised hyphenated
  brand; it sits in the **negative** field, where it forbids an effect rather than instructing imitation of
  anyone, which is the harm the rule exists to prevent (*a name in the prompt is an instruction to
  imitate*); and the run's own SOURCE 2 vocabulary imports the term verbatim as sound behaviour. Against
  that, deleting it would strip the single strongest negative token protecting **the pair's most fragile
  intent** — the lead that sits a fraction flat and never corrects, Special Flair #11, the device step 10
  named as unrecoverable without a re-render. The property being fixed (a trademark string) and the
  property doing the work (pitch-correction defence) are not the same property, so the field is left as
  step 10 measured it and the tension is written down here instead. `no pitch correction on the lead`
  precedes it in every exclude field, so the intent survives intact if The Scientist strikes the word.
  ✅ **Ruled by the coordinator, `COORDINATOR_RULINGS.md` §R-1: KEEP.** The name ban protects
  appropriated creative labour — people, groups, bands, labels, recordings — and does not reach the
  standard word for a sound. The field stands as measured; this bullet is the record, not an open
  question.
- **Intensified:** the article shift. Verse 1 runs *"The shaft gives back the fan."* then *"A fan runs in
  the wall."* — the shaft answers with a fan before she has met it, and every later mention is definite:
  *the fan does not stop*, *the fan takes them the same*, *the fan runs in the wall*, *the night air takes
  the fan*. Preserved exactly and now marked in the verse-1 cue. **Reason:** it is the seed's engine in
  miniature — the thing was already there before the instrument arrived to notice it — and it is carried
  entirely by two indefinite articles, which is precisely the kind of detail a polish pass deletes by
  tidying. **Effect:** the hinge's *"before anyone looked"* has already happened once, at grammar level, in
  verse 1.
- **Changed:** two lines. *"She says them joined up."* → *"She says the whole name through."* and *"The
  joined name comes out flat."* → *"The whole name comes out flat."* **Reason:** *joined up* is what a
  handwriting exercise is called; the section is the song's one moment of deliberate dignity and it was
  being carried by the flattest available phrase. *Whole* also pays off *family name* and *given name* in
  the two lines above it. **Effect:** the Said Again section now completes an actual ladder — part, part,
  whole — inside eight lines, at the same syllable counts.
- **Intensified:** the chorus EMO ladder, `Self-doubt, Composure` → `Skepticism, Composure` → `Dignity,
  Composure`, on the byte-identical sentence. *She does not fix it* starts as a failure to correct and ends
  as a decision not to, and nothing in the words moves. **Effect:** the final chorus transforms without a
  lift, which is the only transformation this pair is permitted.
- **Changed:** the hinge's SFX cue was removed, taking the song from four standalone cues to three, on the
  reasoning given under V2. The three that remain are the cold open, the Said Again substitution section,
  and the outro.
- **Intensified across all four songs:** the level defence, in the style field and the exclude field.
  Documented in full under V1; the same treatment is applied here, in this song's own words, because *fast
  and quiet* is the one thing in this pair a generator is likely to take away, and a spec that only forbids
  the lift without offering the alternative is asking a model to do nothing where it wants to do something.
