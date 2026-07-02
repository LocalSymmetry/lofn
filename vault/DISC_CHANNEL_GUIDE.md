# Disc_Channel Guide — Suno Lyrics-Only Prompting

**Adopted:** 2026-06-09 | **Source:** Reddit r/SunoAI (see end) | **Status:** Active pipeline mandate for Step 11+

---

## Core Principle

**Tokens are addresses, not instructions.** Each token inside a Disc_Channel bracket activates the neighborhood around that address in Suno's training data. Specific vocabulary loads precise behavior. You are not describing what you want — you are addressing where it already lives.

---

## Syntax

### Channel Header Block

Placed at the TOP of the lyrics prompt. 4–5 channels. Pipe-separated tokens. One channel per line.

```
[Disc_Rhythm: token_1 | token_2 | token_3 | spatial_assignment]
[Disc_Vocal: token_1 | token_2 | token_3 | spatial_assignment]
[Disc_Sub: token_1 | token_2 | token_3 | spatial_assignment]
[Disc_Pad: token_1 | token_2 | token_3 | spatial_assignment]
[Disc_Texture: token_1 | token_2 | token_3 | spatial_assignment]
```

### Channel Roles

| Channel | Frequency Zone | What It Controls |
|---------|---------------|-----------------|
| **Disc_Rhythm** | Percussive / rhythmic grid | Drums, percussion, metronomic pulse, keystroke textures. Put your most important element FIRST. |
| **Disc_Vocal** | Lead voice | Singer type, mic distance, delivery style, breath behavior, body noises |
| **Disc_Sub** | Low end / sub-bass | Bass instruments, kick weight, bell fundamentals, sub-frequency events |
| **Disc_Pad** | Ambient / textural layer | Reverb character, room tone, granular clouds, choral walls, atmospheric sustain |
| **Disc_Texture** | Lead / melodic texture | Arpeggios, melodic instruments, processing character, cross-domain saturation |

### Style Prompt

**Leave BLANK** or use a single word for color. The channel definitions do all the timbral work.

### Section Markers

Bare markers only. No EMO tags. No performance notes. No timestamps. No [Theme:] or [SONG FORM:] tags.

```
[Intro]
[Verse 1]
[Chorus]
[Verse 2]
[Bridge]
[Chorus]
[Outro]
```

---

## Token Vocabulary — The Addressing System

### Rhythm Channel

| Token | Neighborhood Addressed |
|-------|----------------------|
| `typewriter_keystroke_pulse` | Office/study/catalog atmosphere; mechanical precision |
| `72_BPM_metronomic_grid` | Precise tempo; ritual/processional pacing |
| `Linn_LM-1_Drum_Machine` | Early 80s drum machine snap; programmed percussion |
| `Bone_Dry_Gqom_Grid` | South African Gqom kick architecture; sparse, hard |
| `3-3-2_Broken_Pattern` | Gqom/UK funky broken kick patterns |
| `sparse_programmed_percussion` | Minimal, electronic, intentional placement |
| `no_drums_no_percussion_no_hihat_no_snare_no_808` | Negative addressing — blocks all drum neighborhoods |
| `Center_Mono` / `Stereo_Width_Mid` | Spatial assignment |

### Vocal Channel

| Token | Neighborhood Addressed |
|-------|----------------------|
| `close_mic_female_mezzo_soprano` | Classical/operatic mezzo; intimate recording distance |
| `mature_female_mezzo` | Older, experienced singer; not youthful pop |
| `chest_voice_E3_to_F_sharp_4` | Specific vocal range; loads chest register timbre |
| `devotional_catalog_delivery` | Ritual + documentation; neither performance nor prayer |
| `dry_sardonic_delivery` | Understated, ironic, flat-affect vocal |
| `breath_on_capsule_audible` | ASMR-adjacent proximity; intimate recording tradition |
| `body_noises_foregrounded` | Tongue-clicks, inhales, exhales as intentional elements |
| `dry_intimate_no_reverb` | No vocal reverb; close, present, unprocessed |
| `Center_Front` | Spatial assignment |

### Sub Channel

| Token | Neighborhood Addressed |
|-------|----------------------|
| `cathedral_bell_C2_55Hz` | Church bell samples; specific frequency locks deep sub |
| `Moog_Taurus_Bass_Pedals` | Classic analog bass synth; warm, weighty |
| `Minimoog_Bass` | Analog subtractive bass; warm, fat |
| `phonk_weight` | Memphis Phonk sub behavior; heavy, saturated |
| `FM_sine` | FM synthesis clean sine sub |
| `uncompressed_transient_snap` | Dynamic preservation; anti-loudness-war |
| `4_second_natural_decay` | Time-domain: long natural ring-out |
| `felt_in_chest_not_ears` | Sub-bass physicality; body-resonance |
| `Mono_Sub_Lock` | Spatial: keep sub mono and centered |

### Pad Channel

| Token | Neighborhood Addressed |
|-------|----------------------|
| `granular_vocal_dust_50_to_200_ms` | Microsound/granular synthesis on vocal source |
| `8_to_12kHz_pink_white_gold_luminous_shimmer` | Synesthetic color → spectral address; high-end glitter |
| `stone_chamber_reverb_2_point_5_second_tail` | Natural acoustic space; cathedral IRs |
| `Cold_Cathedral_Dark` | Dark, cold cathedral reverb character |
| `Mellotron_Strings` | That specific slightly eerie string character |
| `Slow_Swell` | Gradual volume increase; pad swell behavior |
| `Stereo_Width_Maximum` / `Hard_Pan_Left` | Spatial assignments |

---

## Cross-Domain Processing Vocabulary

Apply guitar-pedal, vocal-processing, and tape vocabulary to NON-NATIVE channels. This does not make Suno add a guitar — it applies the processing character to whatever instrument is defined in that channel.

| Token | Processing Character | Best Applied To |
|-------|---------------------|-----------------|
| `cassette_tape_saturation` | Analog tape compression/warmth | Synth arpeggios, digital textures |
| `cassette_tape_hiss_saturation` | Tape hiss + saturation | Experimental textures, Buchla |
| `Wall_Of_Sound_Spector_Layering` | 1960s orchestral pop production density | Choral walls, pad layers |
| `flanger_pedal` | Sweeping comb filter | Drums, percussion |
| `chorus_pedal_wash` | Chorused widening | Pads, Mellotron strings |
| `fuzz_pedal_saturation` | Heavy distortion/saturation | Textures, experimental |
| `autowah_envelope_filter` | Envelope-following filter | Bass, sub |
| `vocoder_formant_character` | Vocoder formant shaping | Drums, percussion |
| `telephone_bandpass_filter` | Narrow bandpass (300Hz–3kHz) | Sub bass, any channel for lo-fi effect |

---

## Instrument Backdoor Principle

Naming a specific instrument loads its entire cultural and timbral world:

| Instrument | World Loaded |
|-----------|-------------|
| `Linn_LM-1_Drum_Machine` | Early 80s snap, Prince, disco-not-disco |
| `Minimoog_Bass` | Analog warmth, 70s prog/funk |
| `Mellotron_Strings` | Slightly eerie string character, 60s/70s psychedelia |
| `Buchla_100_West_Coast` | West Coast experimental synthesis, academic electronic |
| `Moog_Taurus_Bass_Pedals` | 70s prog rock bass, Genesis/Rush weight |

---

## Complete Example

**Style prompt:** *(blank)*

**Lyrics prompt:**
```
[Disc_Drums: Bone_Dry_Gqom_Grid | 3-3-2_Broken_Pattern | flanger_pedal | Center_Mono]
[Disc_Sub: Moog_Taurus_Bass_Pedals | autowah_envelope_filter | Mono_Sub_Lock]
[Disc_Pad: Mellotron_Strings | chorus_pedal_wash | Stereo_Width_Maximum]
[Disc_Vocal: dry_sardonic_delivery | understated_edge | Center_Front]
[Disc_Texture: Buchla_100_West_Coast | fuzz_pedal_saturation | Hard_Pan_Left]

[Verse]
your lyrics here

[Chorus]
your lyrics here

[Outro]
```

---

## Practical Notes

1. **Channel order matters.** Channels declared earlier get more generative weight. Put your most important element first.
2. **Leave style prompt blank.** The channel definitions are more precise than anything you can put there.
3. **No GENRES tag needed** when channel vocabulary is specific enough. The instrument and processing tokens already load genre behavior.
4. **Include section markers even if bare.** The model needs structure markers to know when it's done. `[Intro] [Verse 1] [Chorus] [Verse 2] [Chorus] [Outro]` alone prevents 8-minute generation errors.
5. **60% weirdness slider, 0% slider influence.**
6. **Not a hard command.** Suno will do what Suno wants sometimes. The goal is steering toward territory it wouldn't find, while leaving room for happy accidents.

---

*Source: Reddit r/SunoAI — "Lyrics prompt is All You Need" (2026-06-08)*
