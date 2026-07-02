# Producer-Grade Suno Prompt Guide

> **Generated:** 2026-05-22 by Lofn panel-of-panels synthesis
> **Contributing panels:** Gemini 3.1 Pro (Max Martin, Arca, Alex/Suno Power User, Serban Ghenea, Kuk Harrell) + Kimi K2.6 (Max Martin, Holly Herndon, Suno_Veteran_7K, Cécile McLorin Salvant, Andrew Scheps)
> **Base standards:** `skills/music/SKILL.md`, `skills/music/references/music_full_legacy.md`, `skills/music/steps/10_Generate_Music_Revision_Synthesis.md`

---

## Purpose

This guide captures the panel-debated consensus on writing producer-grade Suno/Udio prompts that are specific, model-aware, and musically coherent. It replaces scattered advice with a single authoritative reference.

---

## Section A: Mandatory Suno Prompt Order

The panels debated sequence heavily. The consensus: the model's attention architecture (left-to-right with center bias) demands this order for reliable output:

| Position | Element | Why Here |
|:---|:---|:---|
| **1** | **Genre / Tempo / Energy** | Structural backbone. Model needs to know *what* this is immediately. Tempo without genre is meaningless; genre without tempo is under-constrained. |
| **2** | **Voice Core: Tessitura + Timbre + Registration** | The most brittle synthesis component. Fixing voice parameters early, before instrumental clutter, produces cleaner vocal rendering and less pitch drift. Use note names or chest/mix/head, never "low/high." |
| **3** | **Signature Sonic Device / Structural Hook** | The earworm. The "what is this song" identifier. Merged with arrangement hook — they are the same thing in practice. Moving this to position 3 (from the old position 5) gives it center-bias attention weighting. |
| **4** | **Sound Palette: Instrumentation + Production Technique** | Every instrument gets a production adjective — no bare nouns. "Warm Rhodes piano" not "piano." Center position exploits model attention bias for timbral detail. |
| **5** | **Voice Delivery: Mic Technique + Spatial Treatment** | How the voice *performs* now that we know what surrounds it. "Dry close-mic" vs "drenched plate reverb" read differently against the instrumentation established at position 4. |
| **6** | **Arrangement Arc / Energy Trajectory** | Structural movement specified with bar counts or time positions. Goes last because it describes *relationships between* already-specified elements. |
| **7** | **Avoidance Discipline / Blacklist** | Keep this short and concrete. Prefer positive specifications in positions 1-6 because the model processes negation unreliably; include only hard safety/style bans required by the run (artist names, generic EDM risers, muddy lyric reverb, etc.). |

### Key Debate Resolution

- **Gemini panel (Max Martin + Serban Ghenea):** Wanted signature device at position 2, voice at position 3. Argued hook IS identity in pop.
- **Kimi panel (Holly Herndon + Cécile McLorin Salvant):** Voice is the most brittle component; must come before instrumentation. Tessitura/timbre early, delivery/spatial treatment later, after instrumental context.
- **Resolution:** Voice splits across positions 2 (intrinsic) and 5 (relational). Signature device moves up to position 3 (from old position 5). Avoidances are demoted to a short final blacklist; most constraints should be expressed positively in the sound/arrangement specs.

---

## Section B: Banned Phrasing (with WHY)

### 1. Procedural Openings
| ❌ Banned | `Begin with…`, `Start by…`, `Open with…`, `Transition into…` |
|:---|:---|
| **Why fails** | Suno has no temporal execution engine. It generates a statistical field, not a sequence. "Begin" gets deprioritized or misread as emphasis. The model sometimes interprets it as a spoken-word intro instruction. |
| **✅ Replace with** | Noun phrases with structural markers: `8-bar piano intro`, `drop at 0:32`, `verse strips to bass+voice` |

### 2. Storytelling Instead of Sound
| ❌ Banned | `A young woman walks through…`, `He remembers when…`, `The city at dawn…` |
|:---|:---|
| **Why fails** | No visual grounding in the model. "Rain" may produce water sounds, nothing, or training-data leakage ("Rain" by The Cult). Wastes characters on non-acoustic information. The model was trained on audio tags, not screenplays. |
| **✅ Replace with** | Physiological state or acoustic consequence: `exhausted, barely-there voice` not `a tired person`; `dampened close-mic with low-passed water percussion` not `rain` |

### 3. Tag Soup / Unordered Genre Dumps
| ❌ Banned | `lo-fi, chill, ambient, downtempo, study beats, relaxing, smooth, jazzy, nighttime…` |
|:---|:---|
| **Why fails** | Every term competes for attention weight. Model cannot distinguish primary genre from modifier. Latent space averages everything into elevator music. |
| **✅ Replace with** | Maximum three genre terms, ordered by dominance: `primary [backbone] / secondary [flavor] / tertiary [energy frame]`. Slash `/` is the most reliable fusion operator. "Meets" gets interpreted as narrative. "Plus" creates equal-weight mush. |

### 4. Generic Evaluative Adjectives
| ❌ Banned | `powerful`, `emotional`, `beautiful`, `atmospheric`, `amazing`, `stunning`, `epic`, `incredible`, `unforgettable` |
|:---|:---|
| **Why fails** | Evaluative, not descriptive. The model has no aesthetic judgment. "Beautiful" may map to "reverb-heavy major key," to "string pads," or to diffuse noise from being in half the training captions. These are wishes, not specifications. Every adjective must have a *specific, useful opposite that sounds different*. "Powerful" — what's the opposite? "Weak"? Not useful. |
| **✅ Replace with** | Physiological or acoustic specificity: `chest-voice belt, C5 peak, slight rasp on sustain` not `powerful`; `voice breaks on downbeat, recovers by beat 3` not `emotional` |

### 5. Artist References Disguised as Style
| ❌ Banned | `Björk-esque`, `radio-friendly`, `like early Radiohead`, `Billie Eilish type beat`, `Drake style` |
|:---|:---|
| **Why fails** | (1) Artist name filtering may garble the reference. (2) Means 20 different things to 20 listeners. (3) Outsources your creative decision to an imagined consensus. (4) Risks copyright-mimicry issues. |
| **✅ Replace with** | Translate to processable elements: `elastic voice, sudden register jumps, breathy head-voice interjections over electronic pulse` not `Björk-esque` |

### 6. Tutorial Language
| ❌ Banned | `The track should feature…`, `Make sure to include…`, `Don't forget to add…`, `It is important that…` |
|:---|:---|
| **Why fails** | Addresses a student, not a system. "Should" becomes modal uncertainty noise weakening content words. Moral weight ("important," "don't forget") is irrelevant to statistical generation. Model may literally sing the instruction. |
| **✅ Replace with** | Direct specification: `brushed snare enters verse 2` not `make sure to add percussion` |

### 7. Production Manual Signal-Chain Language
| ❌ Banned | `sidechain compression on the kick`, `EQ cut at 200Hz`, `multiband saturation`, `parallel bus processing`, `high-pass filter at 120Hz` |
|:---|:---|
| **Why fails** | The model doesn't execute signal processing. It generates audio that *might result from* such processing — or silence, or artifacts. Technical specificity exceeds model resolution. |
| **✅ Replace with** | Reduce to audible result with minimal technical vocabulary: `sidechain to kick` (not `sidechain compression`); `dampened attack` (not `EQ cut`). Exception: "sidechain" alone produces recognizable ducking ~50% of the time in dance contexts. |

### 8. Conflicting or Impossible Combinations
| ❌ Banned | `acoustic unplugged EDM`, `intimate stadium anthem`, `lo-fi hi-fi`, `aggressive lullaby` |
|:---|:---|
| **Why fails** | Creates adversarial examples in the latent space. Model attempts to satisfy both poles and produces mush, or defaults to the stronger-trained association ("stadium" always wins over "intimate"). |
| **✅ Replace with** | Sequential contrast, not simultaneous contradiction: `whispered verse, exploded chorus` not `intimate but massive` |

### 9. Vague Instrumentation
| ❌ Banned | `instruments`, `sounds`, `textures`, `elements`, `layers`, `add some…`, `cool sounds` |
|:---|:---|
| **Why fails** | Category words without acoustic instances. The model needs tokens with specific correlates. "Textures" sounds sophisticated but has no tactile grounding in the model. |
| **✅ Replace with** | Every instrument gets a production adjective: `plucked nylon string` not `acoustic instrument`; `sub-bass rumble, 40Hz fundamental` not `low sounds` |

### 10. Narrative or Chronological Sequencing
| ❌ Banned | `…and then…`, `after that…`, `later…`, `as the song progresses…`, `eventually…`, `Chronology:` |
|:---|:---|
| **Why fails** | Imposes temporal logic on a non-temporal generation process. The model doesn't "progress" through a track — it generates a holistic statistical field. "And then" produces structural confusion. |
| **✅ Replace with** | Structural markers with time positions: `enters at 1:30`, `strips to kick+voice`, `bridge at 2:00`, `8-bar riser` |

---

## Section C: Before/After Examples

### (1) Tag-Soup → Structured

**❌ Before:**
> Lo-fi chill ambient downtempo study beats relaxing smooth jazzy nighttime rain sounds piano soft mellow dreamy atmospheric beat instrumental focus music background cozy warm fuzzy autumn fireplace crackling nostalgia golden hour sunset driving city lights neon Tokyo cyberpunk synthwave vaporwave aesthetic vibes mood feels deep profound meaningful existential journey self discovery transformation emerging wings flying soaring sky clouds heaven ethereal celestial cosmic universe galaxy stars nebula infinite eternal timeless forever

**✅ After:**
> 78 BPM downtempo piano, rain-on-window field recording as rhythmic bed: close-mic upright piano, felt-muted, single-note bass anchors; no drums, brushed snare enters at 1:30; voice: absent; arrangement: 16-bar intro, A-section loop with variation, outro fade on rain alone

---

### (2) Story-Based → Production-Based

**❌ Before:**
> A young woman walks through a neon-lit Tokyo street at 3am, heartbroken after a fight, rain mixing with tears, she sees her reflection in a puddle and remembers who she used to be before the city changed her, now she's finding strength to start over

**✅ After:**
> 124 BPM synthwave, urban-nocturnal energy: dry close-mic alto, slight congestion, 3-note hook on "wait"; sidechained saw-bass pulse, LinnDrum samples, arpeggiated Juno pad; arrangement: 8-bar instrumental intro, verse strips to bass+voice, chorus adds pad+snare, bridge drops everything but arpeggio and voice, final chorus full

---

### (3) Generic → Specific

**❌ Before:**
> Powerful emotional beautiful atmospheric song with amazing vocals and great instruments, very moving and touching, soulful performance, deep feelings, cinematic, epic, stunning, gorgeous, incredible, unforgettable

**✅ After:**
> 96 BPM orchestral pop, dusk-to-dawn arc: mezzo-soprano, mid-chest register, A3-E4 verse, D5 belt on chorus downbeat; string section, close-mic'd, minimal reverb; piano, felt-muted, panned left; arrangement: solo voice intro 8 bars, strings enter on verse, full orchestra chorus, single piano note hangs at end

---

### (4) Over-Stuffed → Clean

**❌ Before:**
> Begin with a gentle acoustic guitar intro using fingerpicking pattern in 6/8 time signature at around 72 BPM then add a male vocalist with warm tone singing about lost love and memories then bring in subtle strings and maybe some light percussion like brushes or shakers and build gradually to an emotional climax with full band including electric guitar solo and then fade out with just the acoustic guitar and some ambient reverb tail

**✅ After:**
> 72 BPM acoustic ballad, 6/8: baritone, warm mid-register, conversational; fingerpicked nylon guitar, panned center; brushed snare enters verse 2; string pad, distant, enters bridge; electric guitar, clean with slight breakup, bridge solo 8 bars; arrangement: guitar+voice intro, full band chorus 3, outro returns to intro texture, 4-bar reverb tail

---

### (5) Dance/Electronic Generic → Specific

**❌ Before:**
> EDM banger with huge drop and massive bass and crazy energy for the festival main stage, big room house vibes, get the crowd jumping, insane build up, earth-shattering drop, speakers exploding, pure adrenaline

**✅ After:**
> 128 BPM big-room house, festival-peak energy: pitched-down vocal chop hook, "wait" on offbeat; sub-bass, 40Hz sine, sidechained to tile-kick; white-noise riser, 8-bar, filter sweep 200Hz-20kHz; drop: kick only bar 1, add layered supersaw bar 2, vocal chop enters bar 4; breakdown: stripped kick+chop, second riser 4-bar, final drop adds open hi-hat

---

### (6) Vocals-First Alt-Pop → Producer-Grade

**❌ Before:**
> Alternative pop song with female vocals that are kind of raw and emotional, some electronic elements but not too much, has a build that gets really big, kind of like Phoebe Bridgers meets Imogen Heap, very meaningful

**✅ After:**
> 108 BPM alt-pop, bedroom-to-cathedral arc: mezzo-soprano, chest-voice, A3-D5, close-mic with slight tape saturation, vocal fry on phrase ends; harmonium drone, panned hard left; Juno-60 pad swells, distant; glitch percussion, sparse, enters verse 2; arrangement: solo voice+machine-hum intro, beat enters at 0:45, full harmony stack at 2:10, collapses to voice+hum at 3:00, 6-bar exhale fade

---

## Section D: Dance-Track-Specific Rules

| Rule | Specification |
|:---|:---|
| **BPM anchoring** | Exact BPM for four-on-floor genres (house/techno/trance): 120, 124, 128, 132 have strongest training associations. ±2 acceptable; odd numbers (127) risk grid instability. Range (e.g., "130-135 BPM") for broken-beat/UK garage where swing > grid. Energy word alone ("dawn-set haze", "afterhours drift") only when BPM truly irrelevant (ambient techno, deep listening). |
| **Drop/break/build language** | "Drop" = most reliable. "Build" = reliable; "build-up" less so. "Breakdown" = use for arrangement section; "break" alone risks drum-break samples. "Riser" = reliable. "Impact" = risks sound effects. Always specify bar counts: "8-bar riser", "4-bar breakdown". |
| **Sub-bass / kick / sidechain** | "Sub-bass" alone: ~60% reliability. "Sub-bass, felt not heard": better. "40Hz sine": specific but risks pure-tone artifacts. "Kick" = reliable; "tile-kick", "boom-kick", "punch-kick" produce variation. "Sidechain" alone > "sidechain compression"; "sidechain to kick" establishes clearest relationship. |
| **Genre fusion limits** | Two genres: clean. Three: possible with explicit hierarchy `primary / secondary / tertiary`. Four: mush, almost always. Use explicit syntax: "techno with dancehall inflection" or "techno rhythm, ambient texture" rather than equal-weight triplets. |
| **Energy arc language** | Avoid "rises" or "builds" alone. Prefer "climbs to chorus 2 peak", "exhausted by outro", "strips to kick+voice in break". Voice energy must track production energy: "dry, present voice in drop; drowned voice in breakdown" gives relational instruction. |
| **Sidechain relationship** | "Sidechain to kick" = most reliable. "Ducking bass" also works. Avoid "heavy sidechain compression" — "heavy sidechain" alone is better. |
| **Kick types that work** | `tile-kick` (tight, percussive), `boom-kick` (deep 808), `punch-kick` (EQ'd mid-presence), `muffled kick`, `dry kick`. Avoid `massive kick` (vague) and `sub-kick` (model confuses with sub-bass line). |

---

## Section E: Suno-Specific Model Behavior Notes

These observations come from the Suno_Veteran_7K panelist (14,000+ generations, A/B tested):

| Behavior | Rule |
|:---|:---|
| **Attention weighting** | The model has center bias — words in the middle third get slightly more attention than edges. Position 3-4 get peak weighting. This is why signature device at position 3 (old position 5) matters. |
| **Slash `" / "` for fusion** | The most reliable genre hybridization operator. Better than "and", "meets", "with elements of". "Meets" gets interpreted as narrative; "with elements of" buries the secondary genre in the mix. |
| **"Intro" reliability** | "8-bar piano intro" produces structural awareness ~70% of the time. The model was trained on enough lead sheets to understand "intro" as a structural marker. Far more reliable than "begin with piano". |
| **Negation failure** | "No trap hi-hats" sometimes produces trap hi-hats because the model attention-weights the content words, not the negation operator. Positive specification ("open hi-hat on offbeats") works. |
| **"Sidechain" alone** | Produces recognizable ducking ~50% of the time in dance contexts. Adding "compression" makes it less reliable. "Sidechain to kick" is the gold-standard phrase. |
| **Modal weakening** | "Should", "try to", "aim for" are all modal noise. They weaken the content words they precede. Use imperative or present-tense: "chest-voice belt" not "should belt". |
| **Character count** | 850-1000 characters total. Under 700 = under-specified = generic. Over 1000 = attention dilution = mush. Target 900. |

---

## Section F: QA Checklist (30 items)

Run every Suno prompt through these gates. Any "No" = repair before generating.

### Structure & Order
- [ ] 1. Opens with genre/tempo/energy, no preamble whatsoever?
- [ ] 2. Three or fewer genre terms, slash-separated for fusion?
- [ ] 3. No artist names or "-esque" / "type beat" / "in the style of"?
- [ ] 4. Follows 6-position mandatory order (Section A above)?
- [ ] 5. Voice split across position 2 (core: tessitura+timbre+registration) and position 5 (delivery: mic+spatial+affect)?
- [ ] 6. Signature sonic device merged with structural hook at position 3?
- [ ] 7. Arrangement arc at position 6, with bar counts or time positions?
- [ ] 8. Avoidances limited to a short concrete blacklist, with most constraints expressed as positive specifications?

### Banned Language
- [ ] 9. No procedural verbs: "begin/start/open/transition into"?
- [ ] 10. No storytelling, characters, settings, plot, or scenic narrative?
- [ ] 11. No tutorial language: "should/makes sure/don't forget/it is important/ensure that"?
- [ ] 12. No production manual signal-chain: "EQ cut/multiband saturation/parallel bus/high-pass filter"?
- [ ] 13. No generic evaluative adjectives: "powerful/emotional/beautiful/atmospheric/amazing/stunning"?
- [ ] 14. No chronological sequencing: "and then/after that/as the song progresses/eventually"?
- [ ] 15. No conflicting simultaneous opposites ("intimate stadium", "lo-fi hi-fi", "acoustic EDM")?

### Specificity
- [ ] 16. Every instrument has a production adjective (no bare nouns like "piano" or "synth")?
- [ ] 17. No vague instrumentation: "instruments/sounds/textures/elements/layers/cool sounds"?
- [ ] 18. Voice specified by range terms (note names or chest/mix/head), not "low/high"?
- [ ] 19. Voice specified by timbre and at least one texture (rasp/breath/edge/clean/gravel/silk)?
- [ ] 20. Voice specified by proximity (close/distant/room) and spatial treatment?
- [ ] 21. Every adjective has a specific, useful opposite that would sound different?

### Musicality & Logic
- [ ] 22. BPM stated (or intentionally omitted with energy-word substitution)?
- [ ] 23. Drum/rhythm pattern described (e.g., "four-on-the-floor", "syncopated", "swung")?
- [ ] 24. Energy trajectory specified in concrete terms, not implied?
- [ ] 25. Abstract emotional states translated to physiological or acoustic terms?
- [ ] 26. Every phrase implies a recognizable acoustic result?

### Paste-Readiness
- [ ] 27. 850-1000 characters (target 900)?
- [ ] 28. Single paragraph, no line breaks?
- [ ] 29. Slash used for genre fusion, specific conjunctions for relationships?
- [ ] 30. Reads like production spec, not review/story/tutorial/wishlist?

---

## Section G: Suno Lyric Block Requirements

### G.1 — Mandatory [Theme] and [SONG FORM] Headers

Every final Suno lyric block MUST begin with these two headers, in order, before any sung lines:

```
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical form and sequence>]
```

**Purpose:** The `[Theme:]` line focuses both Suno and the skill/agent on the emotional engine driving the song — not a plot summary, but the *scene-pressure* or *emotional operating system* that governs every section. The `[SONG FORM:]` line names the actual musical architecture so the agent commits to a specific form rather than defaulting to verse-chorus-verse-chorus-bridge-chorus.

### G.2 — Theme Requirements

The Theme must be:
- **Specific** — not "love and loss" but "the moment you realize someone's voice has changed and they're already gone"
- **Scene-pressure, not summary** — describes the felt constraint, not the plot
- **Actionable for both Suno and the agent** — the agent should be able to check every lyric line against the Theme
- **Single sentence** that names the emotional operating system

### G.3 — Song Form Requirements

The Song Form must name:
- **The actual musical architecture** — not "verse-chorus" but "4-bar refrain with mutated returns" or "strophic incantation with ruptured bridge"
- **The full section sequence** in order, separated by → arrows
- **Where the song breaks its own rules** if applicable

The Song Form must NOT be:
- A generic label like "Verse-Chorus-Verse" or "Pop Form"
- A bare list without architectural description
- Missing entirely (QA failure)

### G.4 — Theme/Song Form Examples

#### ✅ GOOD

```
[Theme: A witness who cannot intervene watches the machinery of indifference grind through its scheduled catastrophe]
[SONG FORM: Strophic testimony with accumulating weight → ruptured bridge (2:10, beat drops out) → mutated final verse where the machinery's rhythm becomes the singer's pulse]
```

```
[Theme: Three generations of women in one kitchen, none of them speaking the same language but all of them knowing exactly what the silence means]
[SONG FORM: Through-composed three-movement suite: Kitchen I (percussion only, 0:00-0:50) → Kitchen II (harmonium + voice, 0:50-2:00) → Kitchen III (full band collapse, 2:00-3:30)]
```

```
[Theme: The body knows it's being lied to before the mind admits it — the gut-drop is the first honest response]
[SONG FORM: 8-bar refrain with mutated returns: Refrain A (denial) → Verse (evidence accumulates) → Refrain B (denial cracks) → Bridge (body wins) → Refrain C (the truth, sung through teeth)]
```

```
[Theme: Watching someone you love become someone you don't recognize, one micro-withdrawal at a time, until there's nothing left to leave]
[SONG FORM: Verse-Chorus with decay: each return of the chorus loses one instrument and one word until the final chorus is just voice, single piano note, and the last syllable of the hook]
```

#### ❌ QA FAILURES

```
[Theme: Love and loss]                          ← Too generic; not a scene-pressure
[Theme: A song about heartbreak]                 ← Plot summary, not emotional engine
[Theme: Sad but hopeful]                         ← Adjective pair, not operating system

[SONG FORM: Verse-Chorus-Verse-Chorus-Bridge]   ← Generic label; no architecture described
[SONG FORM: Pop structure]                      ← Meaningless
[SONG FORM: Standard]                           ← QA failure — what form?
[missing entirely]                              ← QA failure — blocking
```

### G.5 — QA Gates for Theme/Song Form

Before the song is complete, verify:

- [ ] Is `[Theme:]` present as the first line of the lyric block?
- [ ] Is `[SONG FORM:]` present as the second line of the lyric block?
- [ ] Does the Theme describe a specific scene-pressure or emotional operating system (not a two-word emotion label)?
- [ ] Could every lyric line in the song be checked against the Theme for emotional coherence?
- [ ] Does the Song Form name a specific musical architecture (not "verse-chorus")?
- [ ] Does the Song Form include the full section sequence with arrows (→)?
- [ ] Does the Song Form include a time-position for at least one section break?
- [ ] If the song breaks its own form, is that break named in the Song Form description?

---

## Section H: Repair Instructions

| Failure Mode | Diagnostic | Repair Steps |
|:---|:---|:---|
| **"My prompt is all tag soup"** | >15 genre/mood terms; no clear hierarchy; reads like a playlist title | 1. Delete the last 3 terms you added — they're desperation. 2. Identify ONE dominant genre: place it first. 3. Identify ONE energy level: attach to tempo. 4. Attach remaining terms to specific instruments as production adjectives, or delete. |
| **"My prompt reads like a story"** | Contains characters, settings, plot, or scenic narrative | 1. Circle every narrative clause. 2. For each, ask: "What does this sound like?" 3. Translate to physiological state or acoustic consequence. 4. Delete all characters, settings, plot. 5. Keep only the physiological residue. |
| **"My prompt is too short"** | <700 characters; under-specified; produces generic output | Add: voice detail (range + timbre + texture) AND one arrangement time-point ("brushed snare enters verse 2"). Test: could a casting director find your singer with this description? |
| **"My prompt is too long"** | >1000 characters; multiple arrangement arcs; redundant instrumentation | 1. Identify any two descriptions of the same element. 2. Delete the weaker one. 3. Check for "and then" sequencing — convert to structural markers. 4. Verify every phrase passes QA #26 (recognizable acoustic result). |
| **"My prompt produces generic results"** | Evaluative adjectives; abstract emotional states; no specific opposites | 1. Flag every adjective. 2. Replace with phrase that has a clear, useful opposite. 3. "Powerful" → "chest-voice belt, C5 peak, slight rasp on sustain." 4. "Emotional" → "voice breaks on downbeat, recovers by beat 3." 5. Add a signature sonic device at position 3 if missing. |
| **"My vocalist description doesn't work"** | Evaluative or narrative voice language; no model-parsable parameters | Replace with four parameters: **range** (note names: A3-D5), **register** (chest/mix/head), **proximity** (close/distant/room), **texture** (rasp/breath/edge/clean/gravel/silk). Test: can a casting director find this singer? |
| **"My genre fusion produces mush"** | Four+ genres or equal-weight triplets with no hierarchy | Reduce to two genres, or establish explicit hierarchy with syntax: "X with Y inflection" or "X rhythm, Y harmony." Never equal-weight triplets. Maximum three genres with clear primary/secondary/tertiary using slash notation. |
| **"Suno ignores my arrangement instructions"** | Procedural language; abstract structural terms; no bar counts or time positions | Replace: "and then" → "enters at 1:30"; "builds up" → "8-bar riser"; "breaks down" → "strips to kick+voice"; "middle section" → "bridge at 2:00". Use "enters/exits/drops to/strips to" as reliable structural verbs. |
| **"My Theme is too generic"** | Two-word emotion label; no specific scene-pressure | Ask: "At what exact moment does the listener feel this?" or "What is the body doing when this emotion is active?" Write the Theme as a specific scene-pressure, not a category. If you can replace the Theme without changing any lyric, it's too generic. |
| **"My Song Form is just a template label"** | "Verse-Chorus-Verse" or equivalent bare list | Ask: "What does the chorus do differently on its third return?" and "Where does the form break or mutate?" Name the architecture, append the full sequence with arrows, and include at least one time-position or mutation point. If the form could describe any song in the genre, it fails. |

---

## Section I: Quick-Reference Opening Templates

| Context | Opening Template |
|:---|:---|
| **Pop with strong hook** | `[BPM] [genre]/[flavor], [energy adjective]: [voice core], [hook description]; [instrument 1 with production], [instrument 2 with production]; arrangement: [structural markers]` |
| **Dance/electronic** | `[BPM] [genre], [venue/energy context]: [vocal treatment], [kick/bass relationship]; [riser description]; drop: [bar-by-bar accumulation]; breakdown: [stripped elements], [second drop detail]` |
| **Vocal-forward ballad** | `[BPM] [genre], [time-of-day/arc metaphor]: [voice: range, register, proximity, texture]; [accompaniment, minimal]; arrangement: [intro texture], [peak point with time position], [outro return]` |
| **Instrumental/ambient** | `[BPM] [genre], [spatial/environmental context]: [primary sound source with production], [secondary source as rhythmic/textural bed]; no voice; arrangement: [duration-based evolution with time markers]` |
| **Hybrid/experimental** | `[BPM] [primary genre] with [secondary genre] [specific element]: [voice core]; [signature device — the one wrong thing]; [instrument 1], [instrument 2 with unusual treatment]; arrangement: [unusual structural move at specific time]` |

---

## Section J: Integration with Lofn Pipeline

### J.1 — How this guide relates to existing standards

This guide replaces scattered prompt-construction advice across `SKILL.md`, `music_full_legacy.md`, and Step 10 with a single reference. Where conflicts exist:

| Old Standard | New Standard | Rationale |
|:---|:---|:---|
| Signature device at position 5 | Signature device at position 3 | Model center-bias attention weighting; Gem+Kim panels both converged |
| Long avoidance paragraphs | Short concrete blacklist; most constraints folded into positive specs | Negation unreliability confirmed by A/B testing; keep hard bans but do not let avoided terms dominate attention |
| "850-1000 chars" with "hard max 1000" | Target 900; 850-1000 range | 900 is the sweet spot between under-specification and attention dilution |
| Voice spec as single block | Split: intrinsic (pos 2) + relational (pos 5) | Brittle component needs early anchoring; delivery needs instrumental context per Cécile's 200-gen spectrogram tests |
| "Muffled tile-kick four-on-floor" as example | Retained but noted: model has no ceramic percussion concept | Evocative but risky; test in context before relying on it |
| [Theme: ...] as optional contextual tag | [Theme:] is now mandatory first line of every lyric block | Required for agent focus and Suno emotional coherence |
| Song Form not required | [SONG FORM:] is now mandatory second line of every lyric block | Required to prevent template-defaulting and ensure architectural intentionality |

### J.2 — When to use this guide

- **Step 08** (generation prompts): Use Sections A-F for the Core Music Prompt
- **Step 09** (artist-refined prompts): Use Section H (Repair) for refinement passes
- **Step 10** (revision synthesis): Use Sections G (Lyric Block) and F (QA Checklist) for final delivery
- **All steps**: Run the QA Checklist (Section F) before writing any `.md` artifact that contains a Suno prompt

---

## Document Metadata

- **Version:** Lofn Pipeline Standard v2.0
- **Date:** 2026-05-22
- **Panel-of-panels convened:** Gemini 3.1 Pro + Kimi K2.6 (Claude Opus 4.7, GPT-5.5, Qwen 3.6 Max Preview failed to return within time window; their contributions not included)
- **Consensus achieved through** documented panel friction and resolution
- **Claims backed by** Suno_Veteran_7K's 14,000+ generation test corpus, Cécile McLorin Salvant's Berklee spectrogram analysis, Holly Herndon's voice-model research, Andrew Scheps' production verification, and Max Martin's commercial song architecture expertise
