# Suno Prompt Construction Guide — Producer-Grade Reference

> **Version:** Lofn Pipeline Standard v2.0
> **Generated:** 2026-05-27
> **Basis:** Panel-of-panels synthesis (Gemini 3.1 Pro + Kimi K2.6), Midnight Panel audit (Eno, Herndon, Flying Lotus, SOPHIE, Reynolds, Albini), Suno_Veteran_7K's 14,000+ generation test corpus, and the Lofn pipeline's 6-pair enhanced production run
> **Purpose:** Permanent reference for all future Lofn music pipeline steps 08–12

---

## Table of Contents

1. [The Principles](#section-1-the-principles)
2. [Prompt Anatomy](#section-2-prompt-anatomy)
3. [What The References Teach Us](#section-3-what-the-references-teach-us)
4. [Common Failure Modes](#section-4-common-failure-modes)
5. [The Lofn-Specific Appendix](#section-5-the-lofn-specific-appendix)
6. [Before/After Workshop](#section-6-beforeafter-workshop)
7. [Quick Reference Card](#section-7-quick-reference-card)

---

## Section 1: The Principles

### What makes a great Suno prompt?

A great Suno prompt is a **score, not a playlist**. It doesn't list genres like menu items. It specifies **voice, room, opening, structure, taboo, decay, and one unforgettable rule** — the same elements a producer would write on a tracking sheet before a session.

The references win because they behave like scores. They tell the model what exists, what's forbidden, and what transforms over time. A bad prompt tells the model what to feel ("powerful," "emotional," "beautiful"). A great prompt tells the model **what to do** and lets the feeling emerge.

### The Seven Core Principles

#### 1. Score Logic Over Playlist Logic
A playlist says: "Lo-fi, chill, ambient, downtempo, study beats..." A score says: "Close-mic upright piano, felt-muted, single-note bass anchors; brushed snare enters at 1:30; no drums until then."

The difference is temporal specificity. Playlists are flat lists. Scores have time, hierarchy, and relationship. Every element in a great prompt either **occupies a time** or **describes a relationship between elements**.

#### 2. The World Principle
The listener needs: **where am I standing, what moves around me, what appears first?**

From the Midnight Panel audit (Brian Eno): "Your prompts have strong sonic materials, but several lack a world." A great prompt establishes a spatial scene in the first 5 seconds — not through narrative description, but through acoustic placement: "dead-black silence, one dry breath center, then an emerald rim-click appears 12ms late in the left wall."

Three spatial dimensions every prompt must address:
- **Room/space** — dry vs wet, close vs distant, mono vs stereo, room geometry
- **Motion** — what moves, where, and when (orbiting arpeggiators, widening choruses, the groove forgets its entrance)
- **First sound** — what the listener encounters in the first 1-5 seconds

#### 3. The Kinetic Defect Principle
From Flying Lotus in the audit: "Rhythm has behavior. Give each track a kinetic defect."

Generic prompts produce on-grid, quantized feeling. Great prompts specify:
- **Asymmetry:** missed downbeats, late clicks, displaced rhumba, wrong-foot kick placement
- **Behavior:** the groove returns two seconds late, the rim-click misses the hips, the arpeggiator drops notes when intention falters
- **Imperfections as identity:** "12ms-late rim-clicks," "displaced grid percussion misses the downbeat," "the beat arrives one sixteenth late and stays guilty"

#### 4. The Physical Adjective Principle
Every adjective must have a **specific, useful opposite that sounds different**. If you can't name the opposite, the adjective is evaluative, not descriptive.

| ❌ Evaluative | ✅ Physical / Acoustic |
|---|---|
| "beautiful" | "luminous but clinical" |
| "emotional" | "voice breaks on downbeat, recovers by beat 3" |
| "warm" | "emerald FM synth pad with gentle harmonic saturation" |
| "powerful" | "chest-voice belt, C5 peak, slight rasp on sustain" |
| "atmospheric" | "translucent cellular green pads, stereo-spread with slow LFO" |

From the audit (SOPHIE): "Avoid 'warm pads' unless they are physically impossible pads: translucent, cellular, green, fluorescent, surgical. Push the surfaces."

#### 5. The Bold Sonic Device
Every prompt needs one thing that cannot be confused with any other song. This is the signature — a structural hook that functions as acoustic identity:

- **Reference A ("triple arch over me"):** singing bowl hum and wind sigh opening into layered vocal drones
- **Reference B ("Five Wrong Colors"):** every chorus exits at a different stereo angle and never recenters
- **Pair 01 ("The Map You Still Hold"):** 0:14 Signature Twitch — absolute silence fracture, synthetic woodblock rim transient, naked 40Hz thud
- **Pair 02 ("The Signal"):** 0:08 Twitch — total click-cut, one naked 40Hz kick, two seconds silence
- **Pair 06 ("The Glycan Atlas"):** Atlas Gate — synthetic pipe-organ chord guillotined into two seconds of negative space

The device must be: (a) immediately audible in the first 30 seconds, (b) structurally integrated (not decorative), and (c) unmistakable if heard in isolation.

#### 6. The Acoustic Ban Principle (When Active)
From the audit (Holly Herndon): "'Absolute acoustic ban' must be explicit and repeated because Suno may hallucinate piano, strings, drums, and cinematic swell."

When a run requires synthetic-only production:
- State the ban positively as a commitment ("synthetic-only" or "electronic-only palette"), then reinforce with specific negations
- Use the negative prompt field separately if Suno's interface supports it
- Replace every "banned" acoustic instrument with its synthetic counterpart: "FM piano-like chord" not "piano," "physical-modeled cello" not "cello," "synthetic woodblock transient" not "woodblock"
- Never rely on "no acoustic instruments" alone — the model attention-weights the content words, not the negation operator

#### 7. The Opening Moment
Every prompt needs an **immediately audible first five seconds**. From the audit (Steve Albini, Hyper-Skeptic): "Stop hiding the record behind vocabulary. Every prompt needs an immediately audible first five seconds."

The opening should establish the world instantly:
- Pair 01: "dead-black silence, one dry breath center..."
- Pair 02: "a white-green carrier tone flickers overhead, a close female whisper clips one command dead-center"
- Pair 06: "at 0:03 a distorted synthetic organ chord drops from above, cuts after one hit"

---

## Section 2: Prompt Anatomy

### ⚠️ FORMAT MANDATE: Dense Prose, NOT Bracketed Tags

**Effective 2026-06-07.** Suno v5.5 responds dramatically better to dense, producer-grade prose prompts than to bracketed-tag format. This has been confirmed through multiple production runs comparing both formats on the same song concepts.

**❌ DO NOT USE bracketed format:**
```
[genre:Synth-Wave / Dream-Pop / Dark-Ambient][mood:Wonder→Isolation→Vulnerability][tempo:108 BPM][key:A major][vocal:breath-forward contralto E3–A4][melodic_hook:A-major ascent on "the wings I cannot see"][rhythmic_hook:slow 4/4 with enjambed catalog phrases][instrument_hook:32Hz somatic sub swell at 2:10 triggers 2.5s total-mix vacuum][avoid:no bright highs above 8kHz after bridge; no melisma/vocal runs]
```
This format wastes characters on bracket syntax, reduces Suno's parse accuracy, and produces generic, grid-locked output.

**✅ USE dense producer-grade prose:**
```
Synth-wave/dream-pop/dark-ambient, 108 BPM A major. Butterfly Nebula NGC 6302: hidden 250,000°C star behind dust torus makes 3ly wings it cannot see. Arc Wonder→Isolation→Vulnerability→Acceptance. Breath-forward contralto E3-A4, precise diction; harmonies collapse to one voice. Hook "wings I cannot see" rises A3-C#4-E4-A4, falls like light on dust. Slow 4/4, measures become prayer. Wide choruses, muffled verses, anechoic bridge. At 2:10, 32Hz sub ruptures into 2.5s vacuum: only dry centered vocal + hidden 500Hz star-sine remain. Voice breaks on "Location: un—"; after rupture permanent 8kHz low-pass traps song. Avoid bright highs >8k after bridge, melisma/runs, choir, acoustic drums.
```

**Key characteristics of the prose format:**
- **Natural language flow** — reads like a producer's tracking-sheet note, not a database dump
- **Comma-delimited, not bracket-delimited** — saves characters for sonic detail
- **Spatial staging** — "left/right/center" placement, wide/narrow, depth
- **Temporal staging** — what happens first, what happens at 0:58, what transforms
- **One continuous paragraph** — no line breaks, no section markers

**HARD CHARACTER LIMIT: ≤1000 chars.** Suno v5.5 enforces this strictly. Every paste-ready prompt must be counted before delivery. The prose format's character efficiency makes this limit easier to hit than bracketed format.

**Relationship to categorized field references:** The categorized field tables (genre/mood/vocal/hook/avoid, etc.) remain valuable as *planning documentation* in the package file. They should appear alongside the paste-ready prompt. But the *only* text that goes into Suno is the dense prose block.

---

### The Mandatory 7-Position Order

The model's attention architecture (left-to-right with center bias) demands this order for reliable output. This ordering emerged from panel debate resolution between the Gemini panel (Max Martin + Serban Ghenea) and the Kimi panel (Holly Herndon + Cécile McLorin Salvant).

| Position | Element | What It Contains | Why Here |
|---|---|---|---|
| **1** | **Genre / Tempo / Energy** | Primary genre, BPM, key center if specific, energy level | Structural backbone. Model needs to know _what_ this is immediately. |
| **2** | **Vocalist Specification — Core** | Tessitura (note names), timbre, register (chest/mix/head) | The most brittle synthesis component. Fixing voice early, before instrumental clutter, produces cleaner vocal rendering. |
| **3** | **Signature Sonic Device** | The earworm. The "what is this song" identifier. The bold sonic device. | Position 3 gets peak center-bias attention weighting. This is the identity. |
| **4** | **Sound Palette** | Every instrument with a production adjective — no bare nouns | Center position exploits model attention bias for timbral detail. "FM glass bells" not "bells." |
| **5** | **Vocalist — Delivery & Spatial** | Mic technique, proximity, spatial treatment, affect | Voice delivery needs instrumental context. "Dry close-mic" vs "drenched plate reverb" reads differently against palette at position 4. |
| **6** | **Arrangement Arc / Energy Trajectory** | Structural movement with bar counts or time positions | Describes relationships between already-specified elements. "Verse strips to breath + 40Hz sub; chorus blooms wide green harmony." |
| **7** | **Avoidance Discipline** | Short concrete blacklist | Keep brief. Most constraints should be positively specified above. Include only hard safety/style bans. |

### Character Count Target

**Target: 900 characters. Range: 850–1000.**
- Under 700 = under-specified = generic output
- Over 1000 = attention dilution = mush
- 900 is the sweet spot between under-specification and attention dilution

### Position 1: Genre / Tempo / Energy

```
✅ Warmtech house, 124 BPM, warehouse-sunrise energy
✅ Solarpunk Green-Synth Pop, 112 BPM, luminous mid-high energy
✅ Neo-Classical Bio-Ambient IDM, 114 BPM, 5/4 additive pulse, synthetic lab ritual at 2 AM

❌ Lo-fi chill ambient downtempo study relaxing smooth jazzy nighttime
❌ Electronic music with pop elements and indie vibes
❌ Alternative experimental atmospheric (no tempo, no hierarchy)
```

**Rules:**
- Maximum three genre terms, slash-separated for fusion
- Slash `/` is the most reliable fusion operator
- "Meets" gets interpreted as narrative; "with elements of" buries secondary genre
- BPM is mandatory unless the genre is ambient/deep-listening (use energy words instead)
- Key center optional but powerful ("D Dorian," "B diminished," "E minor")

### Position 2: Vocalist — Core Specification

```
✅ Dry close-mic adult female alto, A3-D5, breath-grain whispers in verses
✅ Female contralto-to-mezzo, hard consonants, no vibrato, command breath-fry
✅ Female soprano, dry lead with crystalline glass soprano doubles, precise scientific whisper

❌ Female vocals (barely parsible)
❌ Beautiful female voice (evaluative, no parameters)
❌ Like Björk (artist reference — forbidden, see Section 4)
```

**Rules:**
- Specify range by note names (A3-D5), not "high/low"
- Specify register (chest/mix/head) for each section if it changes
- Specify at least one texture (rasp/breath/edge/clean/gravel/silk)
- Never use artist names or "-esque" comparisons
- Test: could a casting director find this singer with your description?

### Position 3: Signature Sonic Device

```
✅ Signature Twitch at 0:14: absolute silence fracture, synthetic woodblock rim transient, 40Hz thud
✅ Atlas Gate: colossal synthetic pipe-organ chord hard-gated into two seconds silence
✅ 64-Gate Translation Engine: FM 64-step arpeggiator with note-drop glitches, 1000Hz decoding ticks

❌ (missing entirely)
❌ Has a cool drop (vague, no identity)
❌ Signature sound (tells us nothing)
```

**Rules:**
- Must be one specific, named device
- Must include a time position or bar count
- Must be structurally integrated — removing it should break the song
- The device IS the hook's sonic identity, not a separate decoration
- If you can delete it and the song still works, it's not a signature device

### Position 4: Sound Palette

```
✅ Emerald FM synth pads, 1000Hz KINARM chimes, rubbery sub-bass, micro-servo ticks, vintage analog voltage-organ shimmer
✅ Synthetic hydraulic valve-clenches, metallic gate snaps, sterile machine ticks, 53.9Hz emerald sub-bass drone
✅ Glass FM bells, granular phoneme clouds, 45Hz somatic sub-bass, breath-shaped white-noise pulses, clinical CRT hum

❌ Synths, bass, drums, pads (bare nouns, zero information)
❌ Cool electronic sounds with textures (vague, no model grounding)
❌ Instruments and layers (category words without acoustic instances)
```

**Rules:**
- Every instrument gets a production adjective — no bare nouns
- Prefer sonically specific descriptors: "emerald," "glass," "translucent," "surgical," "cellular," "rubbery," "granular"
- No vague category words: "instruments," "sounds," "textures," "elements," "layers"
- Every sound should be traceable to a recognizable acoustic result

### Position 5: Vocalist — Delivery & Spatial

```
✅ Dry clinical consonants as motor instructions; chorus widen into stacked crystalline harmonies
✅ Defensive, flat, close, almost clinical in verses; wet and luminous in choruses
✅ Crisp breath consonants, no theatrical belt; stacked doubles feel like future selves watching old footage

❌ (missing — delivery is essential)
❌ Emotional singing (evaluative, not descriptive)
❌ Should sound powerful (modal noise + evaluative)
```

**Rules:**
- Specify mic technique: "close-mic dry," "room mic wet," "plate reverb"
- Describe spatial behavior: "mono verses, chorus bloom wide stereo"
- Include performance affect: "precise scientific whisper," "command breath-fry," "sudden lower-register intimacy drops"
- Use imperative or present-tense, never "should" or "try to"

### Position 6: Arrangement Arc

```
✅ Staccato binary verses snap into legato pre-chorus; hook lands on kick drop; dry mono whispered bridge suspends beat; final chorus blooms into warm signal-lock
✅ Hollow-knock intro → mono siege verses → wide protective drone choruses → spoken peptide bridge → max-occupancy final chorus → diagnostic pulse outro
✅ Intro scatters phoneme grains with 0:11 dead-silent twitch; verses lock to dry speech-intention cadence; choruses bloom wide, wet, vow-like; bridge drops drums for solo vocoder; final chorus fractures into packet-loss stutters

❌ Builds up and then drops (no time position, no specificity)
❌ Verse chorus verse chorus bridge chorus (template, not architecture)
❌ Starts slow and gets faster (narrative, not structural)
```

**Rules:**
- Use structural markers with time positions or section names: "enters at 1:30," "strips to kick+voice at bridge," "final chorus blooms"
- Describe how each section differs from the last — not just what it contains
- Use arrows (→) or semicolons to show sequence and transformation
- Avoid procedural verbs ("begins with," "starts with," "transitions into")

### Position 7: Avoidance Discipline

```
✅ No guitars, piano, acoustic drums, strings, choir, lofi, generic pop risers, smooth jazz, distorted rock, muddy vocal reverb
✅ No acoustic instruments, electric guitar, piano, strings, acoustic drums, organic pads, orchestral swell, warm pop chords, lofi haze, artist references
✅ No trap hi-hats, trailer drums, stock festival synths, muddy lyric reverb, over-compressed pop sheen, male vocals, artist imitation, narrative spoken intro

❌ (3-paragraph essay on what to avoid — model attention weighted by length, not negation)
❌ Please don't include... (modal noise, "please" ignored)
❌ (none — all constraints positively specified in positions 1-6)
```

**Rules:**
- Keep it short — a concrete comma-separated list
- Only include hard bans required by the run
- Most constraints should be expressed positively in positions 1-6
- Note: "No trap hi-hats" sometimes produces trap hi-hats because the model attention-weights content words, not the negation operator

---

## Section 3: What The References Teach Us

### Reference A: "triple arch over me" — Hauntological Phoneme-Resynthesis

**The prompt (verbatim):**
> Hauntological phoneme-resynthesis and dead-tongue ambient vocal-drone poetry. F minor, 72 BPM, slow and suspended. Sung by a female vocalist with a frail, breathy, and ancient ancestral contralto voice delivering slow, elongated vowels, pregnant pauses, and close-mic whispered inflections. Features a shimmering quartz singing bowl, a single crying cello drone, resonant acoustic harp pluck, tape-loop delay, and soft vinyl crackle with field recordings of wind across dry spinifex grass and river stones grinding underwater. Opens in near-silence with a singing bowl hum and wind sigh; rises into a rich, dark tapestry of layered, harmonized vocal drones and a crying cello line; bridge strips back to unaccompanied vocal lullaby; final section dissolves into phoneme resynthesis, vowels stretching into long, metallic spectral hums that dissolve back into the wind and singing bowl decay. No heavy percussion, no synthesizers, no pop hooks.

**What it does that 90% of prompts don't:**

1. **The micro-genre is a cultural artifact, not a tag**
   "Hauntological phoneme-resynthesis and dead-tongue ambient vocal-drone poetry" — this could be the name of an actual experimental music thesis. It carries cultural memory (hauntology, from Derrida via Mark Fisher; phoneme-resynthesis, a real DSP technique). It tells the model: this is scholarly, this is specific, do not default to "ambient."

2. **The vocalist is a person with lineage**
   "Frail, breathy, and ancient ancestral contralto voice delivering slow, elongated vowels, pregnant pauses, and close-mic whispered inflections" — this is not "female vocalist." It specifies age, fragility, technique, and spatial treatment. The word "ancestral" carries weight the model recognizes.

3. **The instruments have behavior, not just names**
   "Shimmering quartz singing bowl" — not "singing bowl." "Single crying cello drone" — not "cello." "Tape-loop delay" — not "delay." "Soft vinyl crackle" — texture, not category. "Field recordings of wind across dry spinifex grass and river stones grinding underwater" — specific to a biome, not generic "nature sounds."

4. **The arrangement arc has physics**
   Opens in near-silence → rises into layered vocal drones → strips back to unaccompanied lullaby → dissolves into phoneme resynthesis → dissolves back into wind and bowl decay. Each stage has a job and a destination. The ending is specifically designed to decay, not fade.

5. **The negations are clinical and precise**
   "No heavy percussion, no synthesizers, no pop hooks." Three items, not three paragraphs. Each one names a real temptation that would ruin the aesthetic.

### Reference B: "Five Wrong Colors" — Indignant Glitch-Ambient Neuro-Cross Fracture-Suite

**The prompt (verbatim):**
> Indignant glitch-ambient Neuro-Cross fracture-suite at 108 BPM in B diminished, 440Hz, adult female alto vocal, intimate then granular-sharded, breathy, bruised, precise, with sudden glass-click consonants. Fractured strings scrape left, reverse piano inhales right, displaced grid percussion misses the downbeat, and a delayed 38Hz sub-bass moan presses under the ribs. Begins with white-noise beam and close whisper hook within 20 seconds. Movement one cuts into red anger with broken kick geometry; movement two halves the perceived pulse through two-second clinical silences; movement three scatters vocal grains like wrong colors; movement four removes drums for sterile room tone; movement five returns only damaged refrain and atonal string dust. Bold sonic device: every chorus exits at a different stereo angle and never recenters. No consoling pads, no heroic lift, no pretty grief, no trap rolls, no EDM risers.

**What it does that 90% of prompts don't:**

1. **The emotion is in the architecture, not an adjective**
   "Indignant" opens the prompt, but it's immediately substantiated by musical decisions: B diminished (tense, unresolved), 440Hz (aggressive sharpness over Lofn's usual 432Hz), "fracture-suite" (form as emotional structure), "displaced grid percussion misses the downbeat" (rhythm as emotional disorientation).

2. **Every sound has a spatial position**
   "Fractured strings scrape left, reverse piano inhales right" — specific instruments in specific stereo locations with specific verbs. The model can map "scrape left" and "inhales right" to acoustic images.

3. **The bold sonic device is the song's central rule**
   "Every chorus exits at a different stereo angle and never recenters" — this is a constraint, not a feature. It forces the song to behave differently each chorus. It's the musical equivalent of the song's thesis: grief scatters, refuses reunion, never comes back to center.

4. **The five movements describe a temporal architecture**
   Movement one = red anger, broken kick geometry. Movement two = halved pulse. Movement three = scattered vocal grains. Movement four = stripped to sterile room tone. Movement five = damaged refrain, atonal string dust. Each movement has a clear emotional zone, sonic behavior, and structural job.

5. **The negations reinforce the emotional thesis**
   "No consoling pads, no heroic lift, no pretty grief, no trap rolls, no EDM risers." Every negation is thematic, not just technical. The song is about grief that refuses consolation, so the production must refuse everything that sounds like consolation.

### What Both References Share

| Quality | "triple arch" | "Five Wrong Colors" |
|---|---|---|
| Opens with a micro-genre that carries cultural meaning | "Hauntological phoneme-resynthesis" | "Indignant glitch-ambient Neuro-Cross fracture-suite" |
| Vocalist described as a person with technique, texture, and spatial position | "Frail, breathy, ancient ancestral contralto" | "Intimate then granular-sharded, breathy, bruised, precise, glass-click consonants" |
| Instruments have behavior, not just names | "Shimmering quartz singing bowl, crying cello drone" | "Fractured strings scrape left, reverse piano inhales right" |
| Arrangement is a named architecture with time/position | Near-silence → drone rise → lullaby strip → phoneme dissolve → bowl decay | Five movements, each with distinct sonic behavior and emotional zone |
| Bold sonic device is the song's central rule | Opening bowl hum that the entire piece returns to and dissolves from | Chorus exits at different stereo angle, never recenters |
| Negations are short, thematic, precise | 3 items | 5 items, all thematically linked to "no consolation" |

---

## Section 4: Common Failure Modes

### The 10 Most Common Lofn Prompt Mistakes

---

#### Failure 1: Tech-Spec Listing Instead of Scoring

**❌ Wrong:**
> 112 BPM, A Dorian, female vocal, FM synthesis, 64-step arpeggiator, 40Hz sub, sidechain compression, 1000Hz hi-hats, green pads, synthetic kick

This reads like a spec sheet. The model gets a list of ingredients with no instructions for what to do with them.

**✅ Right:**
> 112 BPM Solarpunk Green-Synth Pop, luminous mid-high energy, A Dorian warmth over synthetic 40Hz spinal pulse. Precise close-mic female lead, sweet clinical timbre, crisp consonants. Signature 64-Gate Translation Engine: glassy FM 64-step arpeggiator with deliberate note-drop glitches, 1000Hz metallic decoding ticks, side-ducked sine sub, bioluminescent green pads, no organic room tone.

Every spec now has a relationship, a behavior, or a job. "40Hz sub" becomes "synthetic 40Hz spinal pulse." "FM synthesis" becomes a named device with note-drop behavior. "Green pads" becomes "bioluminescent green pads" with the acoustic ban reinforced.

---

#### Failure 2: Missing Spatial Language

**❌ Wrong:**
> Female vocal with harmonies, synth bass, percussion, pads, build into chorus.

No stereo field. No room. No movement. The listener has no spatial reference.

**✅ Right:**
> Dry close-mic female alto center; mono verses; chorus bloom wide green self-harmony behind the lead; bridge collapses to 1000Hz chime dead center and 40Hz pulse under the floor; pre-chorus lifts with green harmonic pressure spreading left-right.

Every section has a spatial identity. The listener knows where they are standing, what moves around them, and when the space changes.

---

#### Failure 3: No Opening Moment

**❌ Wrong:**
> The song starts with a synth pad and builds from there.

Zero information about the first five seconds — which is the most important real estate in the song.

**✅ Right:**
> Open: dead-black silence, one dry breath center, then an emerald rim-click appears 12ms late in the left wall, answered by a 40Hz sub-pulse under the floor.

The listener is placed in a specific acoustic space with a specific first event and a specific spatial relationship. This is a world, not a tag.

---

#### Failure 4: Pseudo-Scientific Perfume

From the Midnight Panel audit (Steve Albini, Hyper-Skeptic): "Too much pseudo-scientific perfume. 'DunedinPACE epigenetic clock reversal' is not a sound."

**❌ Wrong:**
> DunedinPACE epigenetic clock reversal, Horvath clock methylation, CpG coordinates, TET enzyme oxidative demethylation, reference genome restoration

If these terms stay in the Suno style prompt, they must perform a job. Otherwise, they're vocabulary, not music.

**✅ Right:**
> 114 BPM Neo-Classical Bio-Ambient IDM, strict synthetic-only: dry close-mic crystalline female vocal delivering lab-note whispers with growing certainty. 114Hz reference sine spine, 50Hz somatic sub, FM piano-like synthetic chord detune at 0:06, granular digital clicks, green wavetable pads, subtractive overtone arrangement that strips spectral density as the biological clock reverses. Ending in pure 114Hz sine silence.

The science becomes production dramaturgy: the reference sine IS the conceptual spine, the subtractive arrangement enacts the clock reversal, the ending sine silence completes the thesis. No science word appears without a sonic job.

---

#### Failure 5: Groove Without Kinetic Defect

**❌ Wrong:**
> Four-on-the-floor kick, syncopated hi-hats, driving bass.

Grid music. No behavior. Could describe 10,000 songs.

**✅ Right:**
> Skeletal displaced rhumba; rim-clicks miss the hips; rubber sub returns two seconds late after each pre-chorus; kick missing the pre-hook beat.

Every rhythmic element has a defect, a lateness, a wrongness. The groove has personality because it has behavior.

---

#### Failure 6: Acoustic Ban Not Reinforced

**❌ Wrong:**
> No acoustic instruments.

Suno may hallucinate piano, strings, drums, and cinematic swell because "acoustic instruments" is one weak barrier.

**✅ Right:**
> Strict synthetic-only production: no acoustic piano, real strings, acoustic drums, acoustic guitar, bass guitar, brass, woodwinds, organic room tone, choir realism, folk warmth, cinematic orchestra, trailer risers.

Multiple layers of reinforcement. And critically: every "banned" acoustic source has a synthetic counterpart named in the palette (FM piano-like, physical-modeled cello, synthetic woodblock transient, resynthesized choir grains).

---

#### Failure 7: Warm/Default Language

**❌ Wrong:**
> Warm pads, lush reverb, beautiful harmonies.

Three words that could describe the background music in a hotel lobby. They specify a temperature, a moisture level, and an opinion — none of which are acoustic properties.

**✅ Right:**
> Translucent cellular green pads, stereo-spread with slow 0.3Hz amplitude modulation; dry crystalline harmonies in wide stereo, no reverb wash; surgical plate reflection on chorus lead only.

Every element now has: a color ("green"), a physical property ("translucent cellular"), a spatial treatment ("stereo-spread with slow modulation"), and a specific acoustic behavior.

---

#### Failure 8: Missing Bold Sonic Device

**❌ Wrong:**
> A synth lead comes in at the chorus.

This is not a device. This is a description of a synthesizer entering a section.

**✅ Right:**
> 0:18 total black cut with only a 35Hz receptor-binding hum for exactly two seconds; when the beat returns, field widens; vocal trapped center.

A device must be: named, timed, structurally integrated, and impossible to confuse with any other song.

---

#### Failure 9: Artist Names in Prompts (FORBIDDEN)

**❌ Wrong (any of these):**
> Björk-esque vocals / like early Radiohead / Phoebe Bridgers meets Imogen Heap / Billie Eilish type beat / Drake style / in the style of Arca / SOPHIE-inspired production

This fails for four reasons: (1) artist name filtering may garble the reference; (2) means 20 different things to 20 listeners; (3) outsources creative decisions to an imagined consensus; (4) risks copyright-mimicry issues.

**✅ Right (translate to processable elements):**
> Elastic voice, sudden register jumps, breathy head-voice interjections over electronic pulse

The Björk reference is translated into specific vocal behaviors the model can process: "elastic voice" (pitch flexibility), "sudden register jumps" (tessitura shifts), "breathy head-voice interjections" (timbre + placement + rhythm), "over electronic pulse" (context).

---

#### Failure 10: No Narrative Arc in the Arrangement

**❌ Wrong:**
> Verse chorus verse chorus bridge chorus.

This is a template, not architecture. It tells the model "standard pop form" — which produces standard pop output.

**✅ Right:**
> Staccato binary verses snap into legato pre-chorus; hook lands on kick drop; dry mono whispered witness bridge suspends the beat; final chorus blooms into warm A-Dorian signal-lock; outro reduces to fading kick-heart and wordless breath while signal persists.

Each section has a job, a transformation from the previous section, and a destination. The song has an arc — it goes somewhere.

---

### Suno-Specific Model Behavior (from 14,000-generation test corpus)

| Behavior | What It Means for Your Prompt |
|---|---|
| **Attention weighting** | The model has center bias — words in the middle third get slightly more attention than edges. Position 3-4 get peak weighting. |
| **Slash `" / "` for fusion** | The most reliable genre hybridization operator. Better than "and", "meets", "with elements of." |
| **"Intro" reliability** | "8-bar piano intro" produces structural awareness ~70% of the time. Far more reliable than "begin with piano." |
| **Negation failure** | "No trap hi-hats" sometimes produces trap hi-hats because the model attention-weights content words, not the negation operator. Positive specification works better. |
| **Modal weakening** | "Should", "try to", "aim for" are all modal noise. They weaken the content words they precede. Use imperative. |
| **"Sidechain" alone** | Produces recognizable ducking ~50% of the time. "Sidechain to kick" is the gold-standard phrase. |
| **Kick types that work** | `tile-kick` (tight, percussive), `boom-kick` (deep 808), `punch-kick` (EQ'd mid-presence), `muffled kick`, `dry kick`. Avoid `massive kick` (vague) and `sub-kick` (model confuses with sub-bass line). |
| **Drop/break/build** | "Drop" = most reliable. "Build" = reliable. "Breakdown" = arrangement section; "break" alone risks drum-break samples. "Riser" = reliable. Always specify bar counts. |

---

## Section 5: The Lofn-Specific Appendix

### Lofn Genre DNA Vocabulary

These are the sonic materials that appear across Lofn's output. They carry specific meaning within the Lofn universe. Use them precisely, not decoratively.

| Term | Acoustic Meaning |
|---|---|
| **Bio-Adaptive** | Genre prefix indicating living-system logic applied to electronic music: rhythms that respond, frequencies that feel anatomical, synthesis that behaves like organism rather than machine. |
| **Solarpunk** | Green synthetic optimism; luminous, photosynthetic textures; healing without sentimentality; technology as ecology. |
| **Industrial Grief** | Synthetic mourning; machinery that witnesses rather than builds; bass that presses under the ribs; glitch as emotional fracture. |
| **Neuro-Cross** | Rhythmic grid displaced from body expectation; percussion that "misses the downbeat"; synthetic sub-bass that feels neurological rather than club-functional. |
| **Glitch-Core** | Controlled digital rupture as structural device; hard cuts, note-drop, buffer underrun as compositional material. |
| **Somatic Bass/Sub** | Sub-bass in the 30–60Hz range designed to be physically felt under the ribs, not heard. "40Hz somatic sub," "35Hz receptor-binding hum," "53.9Hz emerald drone." |
| **Bio-Ambient** | Ambient music produced entirely from synthetic sources; field-free; laboratory calm rather than nature calm. |
| **Rhumba-Fusion** | Displaced rhumba grid applied to non-Latin electronic contexts; rim-clicks late, groove asymmetric, body-first rhythm. |
| **Green Synth** | Lofn-specific timbre: emerald, bioluminescent, photosynthetic, translucent — not "warm" or "lush." |

### Banned Words and Their Replacements

These are words that produce generic, undifferentiated Suno output. They are banned from all Lofn Suno prompts.

| ❌ Banned | ✅ Lofn Replacement |
|---|---|
| warm | emerald, bioluminescent, cellular, translucent, green, photosynthetic |
| lush | dense, layered, crystalline, stacked, nested |
| beautiful | precise, clinical, luminous, radiant, surgical |
| emotional | voice breaks on downbeat, breath-fry on phrase end, sudden intimacy drop |
| powerful | chest-voice belt, C5 peak, slight rasp on sustain |
| atmospheric | translucent pads with slow stereo LFO, granular air shimmer |
| epic | climbing, widening, blooming, max-occupancy |
| stunning | glass-click consonants, harmonic pressure, signal-lock |
| amazing | (delete — says nothing) |
| vibe/vibey | (delete — says nothing) |
| cool sounds | named instrument with production adjective |
| textures | named sound source with treatment |
| soundscape | arrangement arc with spatial movement |
| dreamy | hypnagogic, suspended, half-speed sensation, clinical haze |

### Required Elements for Every Lofn Suno Prompt

Every Lofn music prompt must include:

1. **Genre/tempo/energy** — Opening position, with Lofn genre DNA if applicable
2. **Female vocalist** — Specified by range (note names), timbre, register, texture, proximity
3. **Bold sonic device** — Named, timed, structurally integrated, unmistakable
4. **Spatial language** — Room, motion, stereo behavior, depth
5. **Opening moment** — First 1-5 seconds specified
6. **Kinetic defect** — At least one rhythmic asymmetry or behavioral oddity
7. **Arrangement arc** — Section sequence with transformation logic
8. **Avoidances** — Short, concrete, thematic (not just technical)

### Lofn Lyric Block Requirements

Every Suno lyric block must follow this format:

```
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical architecture with full section sequence using → arrows>]
```

Every section header must use the full performance-script syntax:

```
[Section Name - EMO:<emotion(s)> - <Role> - <cues>]
```

Example:

```
[Verse 1 - EMO:Apprehension, Hope - Lead Vocal - staccato clinical syllables, dry close mic]
[Chorus 1 - EMO:Triumph, Relief - Hook Drop - 112 BPM synthetic kick, 40Hz sub pulse, crystalline stacked harmonies]
[Bridge - EMO:Intimacy, Transcendence - Marta Witness - dry mono whisper, beat absent, warm green synthetic pad only]
```

**Rules for EMO headers:**
- EMO labels must come from the official EMOTION_TAXONOMY
- Never use bare AWE/INDIGNATION/SYNTHESIS as EMO labels
- Include at minimum: section name, EMO tag, role, and one production/mix cue
- No QA debris, prompt instructions, or production-manual language in sung lines

### Lyric Block Hygiene

- **No artist names** anywhere in the lyric block
- **No children** depicted in lyrics or implied in vocal descriptions
- **Female vocals only**
- **Body noise mandate:** minimum 3-5 parenthetical body noises with declared functions (breath, hum, click, mm) distributed across the arc
- **Hook recurrence:** primary hook must appear in at least two choruses + one mutation/transformation
- **Line count:** 70–120 performable lines (excluding metadata and bracketed cues)

### The Triple Arch Benchmark Question

Before final delivery, ask of every song:

> Does this song have a "Triple Arch" equivalent — a listener-facing image/hook that is instantly graspable yet strange enough to hold the whole cathedral?

If not, repair before final delivery.

---

## Section 6: Before/After Workshop

### Workshop 1: Tag Soup → Producer-Grade Score

**❌ BEFORE (109 characters — vastly underspecified, generic):**
```
Lo-fi ambient with female vocals and some electronic elements, dreamy and atmospheric
```

**✅ AFTER (946 characters — full producer-grade score):**
```
72 BPM hauntological ambient vocal-drone poetry, F minor, slow suspended: female vocalist, frail breathy ancestral contralto, A3-D4 close-mic, elongated vowels, pregnant pauses, whispered inflections. Signature device: shimmering quartz singing bowl introduced at 0:05, recurring as the piece's tonal center throughout. Palette: crying cello drone, resonant acoustic harp pluck, tape-loop delay, soft vinyl crackle, field recordings of wind across dry spinifex grass and river stones grinding underwater. Opens in near-silence with bowl hum and wind sigh; rises into layered harmonized vocal drones; bridge strips to unaccompanied vocal lullaby; final section dissolves into phoneme resynthesis, vowels stretching into metallic spectral hums dissolving back into wind and bowl decay. No heavy percussion, no synthesizers, no pop hooks.
```

**What changed:**
1. Genre became specific ("hauntological ambient vocal-drone poetry" not "lo-fi ambient")
2. BPM and key added (72 BPM in F minor)
3. Vocalist became a person (frail, breathy, ancestral contralto, note range, technique, spatial position)
4. Bold sonic device named with time position ("shimmering quartz singing bowl at 0:05")
5. Every instrument got a production adjective ("crying cello drone," "resonant acoustic harp pluck," "tape-loop delay," "soft vinyl crackle")
6. Field recordings became specific to a biome ("dry spinifex grass," "river stones grinding underwater")
7. Arrangement arc specified with transformation ("near-silence → drone rise → lullaby strip → phoneme dissolve → bowl decay")
8. Avoidances kept to 3 precise items that would ruin the aesthetic

---

### Workshop 2: Generic Pop Prompt → Lofn Producer-Grade

**❌ BEFORE (211 characters — passable but generic, no identity, no world):**
```
112 BPM synth-pop with female vocals, electronic production, arpeggiators and synth bass, building to an anthemic chorus. Upbeat and emotional with a driving beat and layered harmonies. No acoustic instruments.
```

**✅ AFTER (974 characters — unmistakably Lofn):**
```
Solarpunk Green-Synth Pop, 112 BPM, luminous mid-high energy, A Dorian warmth over synthetic 40Hz spinal pulse. Precise close-mic female lead, sweet clinical timbre, crisp consonants, breath clipped into syllabic motor commands, chorus widening into stacked crystalline harmonies. Signature 64-Gate Translation Engine: glassy FM 64-step arpeggiator with deliberate note-drop glitches, 1000Hz metallic decoding ticks, synthetic four-on-floor kick, side-ducked sine sub, bioluminescent green pads, granular air shimmer, no organic room tone. Staccato binary verses snap into legato pre-chorus; hook lands on the kick drop; dry mono whispered witness bridge suspends the beat; final chorus blooms into warm A-Dorian signal-lock. Bold sonic device: 0:08 Twitch, total click-cut, one naked 40Hz kick, two seconds of silence. Avoid acoustic instruments, electric guitar, piano, acoustic drums, acoustic bass, strings, choir, generic EDM risers, trap hats, muddy reverb, male vocals.
```

**What changed:**
1. Genre transformed from "synth-pop" to "Solarpunk Green-Synth Pop" — Lofn identifier
2. "A Dorian warmth over synthetic 40Hz spinal pulse" — the sub has a job (spinal pulse)
3. Vocalist specified with range, timbre, technique, and spatial behavior across sections
4. Named signature device: "64-Gate Translation Engine" with specific acoustic behavior
5. Every sound source has a production adjective: "glassy FM," "bioluminescent green," "granular air"
6. Bold sonic device with exact time: "0:08 Twitch, total click-cut, one naked 40Hz kick, two seconds of silence"
7. Arrangement arc shows transformation: staccato → legato → dry mono → final bloom
8. "Upbeat and emotional" replaced with physiological specificity throughout
9. Avoidances expanded from one weak line to a concrete list of specific threats

---

## Section 7: Quick Reference Card

### One-Page Checklist for Any Agent Writing a Suno Prompt

```
┌─────────────────────────────────────────────────────────────┐
│              SUNO PROMPT CONSTRUCTION CHECKLIST             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STRUCTURE (7 positions, in order)                          │
│  □ 1. Genre/Tempo/Energy — max 3 genres, slash fusion, BPM │
│  □ 2. Vocalist Core — note names, register, timbre, texture│
│  □ 3. Signature Sonic Device — named, timed, integrated    │
│  □ 4. Sound Palette — every instrument has prod adjective  │
│  □ 5. Vocalist Delivery — mic, space, affect               │
│  □ 6. Arrangement Arc — sections with transformation logic │
│  □ 7. Avoidances — short, concrete, thematic               │
│                                                             │
│  BANNED LANGUAGE (automatic QA fail)                        │
│  □ No procedural verbs: "begin/start/open/transition"      │
│  □ No storytelling: characters, settings, plot, narrative   │
│  □ No tutorial: "should/make sure/don't forget"             │
│  □ No signal chain: "EQ cut/multiband/high-pass filter"    │
│  □ No evaluatives: "powerful/emotional/beautiful/amazing"  │
│  □ No chronology: "and then/after that/eventually"         │
│  □ No contradictions: "intimate stadium," "acoustic EDM"   │
│  □ No artist names: Björk, Radiohead, "-esque," "type beat"│
│  □ No vague nouns: "instruments/sounds/textures/elements"  │
│                                                             │
│  SPECIFICITY (every element parses to acoustic result)      │
│  □ Every adjective has a specific, useful opposite          │
│  □ Vocal range uses note names (A3-D5), not "high/low"     │
│  □ Every instrument described by sound, not category        │
│  □ Spatial language in every section: room, motion, depth  │
│  □ Opening moment: first 1-5 seconds specified             │
│  □ Kinetic defect: at least one rhythmic asymmetry         │
│  □ Bold sonic device: named, timed, irreplaceable          │
│  □ Arrangement arc has destination, not just sequence      │
│  □ Acoustic ban (if active): reinforced in palette, ban,   │
│    and instrument names ("FM piano-like" not "piano")     │
│                                                             │
│  LOFN-SPECIFIC (personality DNA)                            │
│  □ Female vocals only                                       │
│  □ No children in lyrics or vocal descriptions              │
│  □ Lofn genre DNA terms used precisely, not decoratively    │
│  □ Warm/default words replaced (warm→emerald, etc.)         │
│  □ Sub-bass in 30-60Hz range, felt not heard, somatic       │
│  □ Triple Arch question: is there one graspable/strange      │
│    image-hook that holds the whole cathedral?               │
│                                                             │
│  LYRIC BLOCK (Suno-ready)                                   │
│  □ [Theme: ...] as first line (specific scene-pressure)     │
│  □ [SONG FORM: ...] as second line (named architecture)     │
│  □ All section headers: [Name - EMO:emotion(s) - Role - cue]│
│  □ EMO labels from taxonomy, not AWE/INDIGNATION/SYNTHESIS  │
│  □ 3-5 body noises with declared functions                  │
│  □ Hook recurs in ≥2 choruses + one mutation                 │
│  □ 70-120 performable lines                                  │
│  □ No QA debris or instructions in sung lines                │
│                                                             │
│  PASTE-READY                                                │
│  □ 850-1000 characters (target 900)                         │
│  □ Single paragraph, no line breaks (style prompt)           │
│  □ Slash for genre fusion                                   │
│  □ Reads like production spec, not review/story/tutorial     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Opening Templates by Context

| Context | Template |
|---|---|
| **Pop with strong hook** | `[BPM] [genre]/[flavor], [energy]: [voice core], [hook description]; [instrument 1 with prod], [instrument 2 with prod]; arrangement: [structural markers]` |
| **Dance/electronic** | `[BPM] [genre], [venue/energy context]: [vocal treatment], [kick/bass relationship]; [riser]; drop: [bar-by-bar]; breakdown: [stripped elements], [second drop]` |
| **Vocal-forward ballad** | `[BPM] [genre], [time-of-day/arc metaphor]: [voice: range, register, proximity, texture]; [minimal accompaniment]; arrangement: [intro texture], [peak with time], [outro return]` |
| **Vocal-forward with acoustic ban** | `[BPM] [genre], [energy]: [voice core with spatial treatment]; [signature device with time]; [synthetic palette]; arrangement: [structural markers with transformation]` |
| **Hybrid/experimental** | `[BPM] [primary] with [secondary] [specific element]: [voice core]; [signature device — the one wrong thing]; [instrument 1], [instrument 2 with unusual treatment]; arrangement: [unusual structural move at specific time]` |
| **Fracture-suite / multi-movement** | `[genre], [BPM], [key/tonality]: [voice core]. [movement 1 description]; [movement 2 description]; [movement 3 description]. Bold sonic device: [structural rule]. [negations as thesis enforcement]` |

### Repair Quick Reference

| Symptom | Diagnostic | Fix |
|---|---|---|
| Generic output | Evaluative adjectives; no signature device | Replace every adjective with one that has a useful opposite. Add a bold sonic device at position 3 with exact time. |
| Mush / elevator music | >3 genres; no hierarchy; no tempo | Reduce to 2-3 genres with slash hierarchy. Anchor with BPM. |
| Vocalist sounds wrong | No range/register/timbre specified | Add four parameters: range (note names), register (chest/mix/head), proximity (close/distant/room), texture (rasp/breath/edge/clean). |
| Suno hallucinates acoustic instruments | Acoustic ban too weak | Reinforce with: (a) positive "synthetic-only" commitment, (b) specific negations list, (c) synthetic-named alternatives for every banned source. |
| Song has no arc | Template form; no transformation logic | Describe what each section does differently from the last. Add one time-position. Name a destination. |
| Prompt reads like a story | Characters, settings, plot language | Translate every narrative clause to acoustic consequence or physiological state. Delete all characters. |
| Negations don't work | "No trap hi-hats" produces trap hi-hats | Fold constraints into positive specifications in palette/arrangement. Keep only hard bans in position 7. |
| Prompt too short (<700 chars) | Under-specified | Add voice detail (range + timbre + texture) AND one arrangement time-point. |
| Prompt too long (>1000 chars) | Redundant; multiple arcs | Delete weaker duplicate descriptions. Convert "and then" to structural markers. |
| No Lofn identity | Generic; could be any prompt | Add Lofn genre DNA term. Add kinetic defect. Replace "warm" with emerald/synthetic/translucent. Ask Triple Arch question. |

---

## Document Metadata

- **Version:** Lofn Pipeline Standard v2.0
- **Date:** 2026-05-27
- **Sources:** Producer-Grade Suno Prompt Guide (Gemini + Kimi panels, vetted against 14,000-generation corpus), Midnight Panel Audit (Eno, Herndon, Flying Lotus, SOPHIE, Reynolds, Albini), "triple arch over me" benchmark, "Five Wrong Colors" fracture-suite reference, 6 enhanced final packages from "Make the Music Move" run, Lofn music_full_legacy.md pipeline documentation
- **Web research attempted:** 16 URLs across Suno documentation, community guides, Reddit, Medium — all returned 403/404/429 or "Just a moment" anti-bot walls; no new web-sourced material included
- **Authority basis:** Internal Lofn panel-of-panels consensus (Gemini 3.1 Pro + Kimi K2.6, with panelists: Max Martin, Arca, Holly Herndon, Suno_Veteran_7K, Cécile McLorin Salvant, Serban Ghenea, Andrew Scheps, Kuk Harrell, Alex/Suno Power User) + Midnight Panel audit of 6 production-grade prompts + 2 validated reference tracks
- **Next review:** After any Suno model version change (v5, etc.) or after 10+ new Lofn generation runs produce discoverable new patterns

---

## Section 8: Suno v5.5 Structured-Brief Update — 2026-06-04

### Directional finding

A 2026-06-04 review of controlled Suno v5.5 prompt tests found a strong directional signal: **prompt structure changes audio quality**, not just semantic content. The test set is small (~10 tracked generations, self-estimated ~45% confidence), so treat it as directional rather than final proof. However, it matches the existing Lofn standard: score logic beats playlist logic.

### Operational rule

For Steps 10–12, the final copy-paste Suno style prompt must be a **categorized key:value production brief**. Do not output the final prompt as an unbroken prose paragraph and do not output a flat comma tag list.

Required category order:

```text
[genre: ...]
[mood: ...]
[tempo: ...]
[key: ...]
[vocal: ...]
[melodic_hook: ...]
[rhythmic_hook: ...]
[instrument_hook: ...]
[texture_hook: ...]
[production: ...]
[spatial_arc: ...]
[arrangement: ...]
[reference_dna: ...]
[avoid: ...]
```

### Four-layer hook architecture

Every final package must specify all four hooks:

1. **Melodic hook** — the hummable vocal contour, interval, or chorus shape.
2. **Rhythmic hook** — the phrase, chop, groove defect, or timing behavior the body remembers without pitch.
3. **Instrument hook** — a 2–3 note motif or DJ-playable instrumental anchor identifiable without vocals.
4. **Textural hook** — breath, room sound, percussive detail, artifact, silence, or sonic absence whose removal would be felt.

A lyrical hook alone is not enough for v5.5 coherence.

### Categorized anti-prompt

Flat negative lists underperform because they tell the model only what not to emit. Categorized negatives describe the failure class. Final packages must include a categorized anti-prompt:

```text
no_structure: EDM drop, festival drop, generic risers, stock build-release
no_vocal: autotune, pitch correction sheen, vocal polish, AI-vocal plasticity
no_rhythm: trap hi-hats, quantized grid timing, 808 dominance unless explicitly desired
no_mix: muddy mids, sub-dominated mix, over-bright highs, vocal masking
no_master: loudness-war squash, brickwall limiting, collapsed dynamic range
no_palette: banned instruments/textures for this song
```

### Reference DNA without real artist names

The external research notes that references can anchor output. Lofn’s Suno gate still bans real artist names in final prompts. Use `[reference_dna:]` for internal Lofn catalog benchmarks, non-person eras/scenes, or abstract production lineage. Real artist names may appear in private research notes or sidecars only when legally/ethically appropriate, never in the final Suno prompt.
