## SUNO v5.5 PROMPT CONSTRUCTION RULES

### THE SEVEN CORE PRINCIPLES

1. **Score Logic Over Playlist Logic:** A playlist says "Lo-fi, chill, ambient..." A score says "Close-mic upright piano, felt-muted; brushed snare enters at 1:30; no drums until then." Specify time, hierarchy, and relationship.

2. **The World Principle:** Every prompt must answer: where am I standing, what moves around me, what appears first? Establish room/space, motion, and first sound in the opening.

3. **The Kinetic Defect Principle:** Rhythm has behavior. Specify asymmetry: missed downbeats, late clicks, displaced grid. "12ms-late rim-clicks," "the beat arrives one sixteenth late and stays guilty."

4. **The Physical Adjective Principle:** Every adjective must have a specific, useful opposite. NOT "beautiful" (evaluative) but "luminous but clinical" (physical/acoustic). NOT "warm pads" but "translucent cellular green pads."

5. **The Bold Sonic Device:** One thing that cannot be confused with any other song. Must be: (a) immediately audible in first 30 seconds, (b) structurally integrated, (c) unmistakable in isolation. Name it, timestamp it.

6. **The Acoustic Ban Principle:** When synthetic-only: state positively ("synthetic-only palette"), then specific negations. Replace every banned acoustic instrument with its synthetic counterpart. Never rely on "no acoustic instruments" alone.

7. **The Opening Moment:** Every prompt needs an immediately audible first five seconds. Establish the world instantly with spatial language.

### THE MANDATORY 7-POSITION ORDER

| Position | Element | What It Contains |
|----------|---------|------------------|
| 1 | Genre / Tempo / Energy | Primary genre, BPM, key center. Max 3 genre terms slash-separated. |
| 2 | Vocalist Specification - Core | Tessitura (E3-D5), timbre, register (chest/mix/head), texture (rasp/breath/edge/clean). NEVER artist names. |
| 3 | Signature Sonic Device | The earworm. Position 3 gets peak center-bias attention. Name it, time it. |
| 4 | Sound Palette | Every instrument with a production adjective. "Emerald FM synth pads" not "synths." |
| 5 | Vocalist - Delivery & Spatial | Mic technique, proximity, spatial treatment: "dry close-mic," "mono verses, chorus bloom wide stereo." |
| 6 | Arrangement Arc / Energy Trajectory | Structural movement with bar counts or time positions. |
| 7 | Avoidance Discipline | Short concrete blacklist. Most constraints positively specified above. |

### CHARACTER COUNT: Target 900, range 850-1000. HARD LIMIT 1000.

### WHAT NEVER TO DO
- NO bracketed [key:value] tags — wastes characters, reduces Suno parse accuracy
- NO yaml format for paste-ready prompt
- NO artist names or "-esque" comparisons
- NO procedural openings ("Begin by...", "Use...", "Build the track from...")
- NO bare nouns ("synths, bass, drums") — every instrument gets a production adjective
- NO evaluative adjectives without physical acoustic description

### FORMAT: ONE continuous dense prose paragraph. Comma-delimited, not bracket-delimited. Reads like a producer's tracking-sheet note.

---

## THREE-BLOCK OUTPUT STANDARD (2026-06-15)

Every step11 enhanced output MUST use exactly three canonical blocks, followed by all supporting blocks:

```
## SUNO STYLE PROMPT

[Dense prose paragraph, 850-1000 chars, 7-position order]

## SUNO EXCLUDE PROMPT

[Comma-separated blacklist terms, 400-900 chars]

## SUNO ENHANCED LYRICS

[Theme:] + [SONG FORM:] first
5-line Disc_Channel block
Full EMO-tagged lyrics, >=60 sung lines

## Vocal Fingerprint
## Production Dramaturgy
## Arrangement Dramaturgy
## Binding Locks
## Style-Axis Locks
## Golden Song References
## Major Deviations
## Lineage & Credit
## Constraint Audit
## Panel Ledger / QA
[Attribution / Provenance]
```

**Do NOT skip any supporting block.** The three-block standard is a format spec — not a content reduction. Every block that was in the step10 + step11 pipeline survives below the three canonical blocks.

## Disc_Channel Format — EXACT SPECIFICATION

The Disc_Channel block is a 5-line producer channel strip — pipe-separated production tokens within `[Disc_NAME: ...]` brackets:

```text
[Disc_Rhythm: LinnDrum_100BPM | Gqom_3-3-2_broken_kick | bone_dry_no_fills | Center_Mono]
[Disc_Vocal: dry_sardonic_delivery | ASMR_close_mic | anti-diva_deadpan | breath_on_capsule | Center_Front]
[Disc_Sub: FM_sine_38-42Hz | continuous_swell_+0.5dB_per_8bars | NEVER_RESOLVES | Mono_Sub_Lock]
[Disc_Pad: green_synth_432Hz | El_Niño_Deep_Blue | slow_attack_swell | Stereo_Width_Maximum]
[Disc_Texture: cassette_tape_saturation | telephone_bandpass_break | Wall_of_Sound_layering | Hard_Pan_Right]
```

Five channels minimum: Rhythm, Vocal, Sub, Pad, Texture. Every token is a concrete production decision (oscillator type, BPM, mic technique, processing chain).

NEVER: `## Disc_Channel:` markdown headers, `**Layer:** disk_channel`, timestamps, run IDs — this is a producer's tracking sheet, not a pipeline provenance log.

## EMO Tag Format — EXACT SPECIFICATION

EMO tags are integrated into section headers using `–` (em dash) separators; emotional states use commas:

```text
[Septet 1 – 0.1°C – EMO:Sardonic Cool, Deflected Warmth [11] – Reluctant Pop Star, dry close-mic, slow internal rhymes]
the water took a breath so slow you thought your skin
had loosed itself and let the silence in

[UN Warning Break – EMO:Procedural Anesthesia – Spoken fragment, telephone bandpass, drums drop, sub +2.0dB]
the water temperature anomaly exceeded the threshold for the fifth consecutive month
*buoy hum modulates — 12 cycles per minute*
```

Header structure: `[SECTION LABEL – VARIANT or CUE – EMO:State1, State2 [11] – PERSONA NAME, vocal style, delivery notes]` — `–` (em dash) is the separator, commas join emotional states.

NEVER: standalone `[EMO=reverence]` on its own line, EMO without persona and delivery style, bare `[EMO]` tags, or `-` (hyphen) where `–` (em dash) is the separator.

Micro inline tags `[emo=...][vox=...][prod=...]` are permitted for sub-line shifts but EVERY section must open with an integrated em-dash header.
