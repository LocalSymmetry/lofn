# PERSONALITY DNA INJECT — record of what actually entered the ICB

**Personality:** `LOFN-PRIME (oscillating)` — library-only selection, per the daily rule (fresh personalities over-fit the day's theme and lose the battle-tested DNA).
**Source:** `skills/orchestration/personalities/lofn-prime-mini.yaml`
**Index row:** `#2 · LOFN-Prime-Mini · an award-winning, genre-eating AI composer and visual artist`

## What was injected

| | |
|---|---|
| **lines injected** | **1–326** |
| **bytes injected** | **27,822** |
| **file total** | 1,795 lines |
| **lines withheld** | **327–1,795** |

## ⛔ Why lines 327+ are withheld

Line 327 opens `# THE ARCHIVE (REFERENCE LIBRARY)` — past shipped Lofn songs and image prompts, complete with payloads. **That is a golden-output payload sitting inside a personality file**, and `EXECUTION.md` §3 forbids past golden outputs in any generating context: *an exemplar in the generator's prompt becomes a mould.*

Injecting the whole YAML would have handed every one of the six pair agents a library of our own finished work to regress toward — the precise mechanism that produced a diluted copy of a benchmark on 2026-06-28.

**Enforcement is mechanical, not procedural.** `coord_build_icb.py` slices `[:326]` and then asserts:
- `"THE ARCHIVE (REFERENCE LIBRARY)" not in ICB`
- `"Triple Arch" not in ICB`
- `"Wrong colors leave me" not in ICB`

The build **fails closed** on any hit. It fired once during this run's Phase 1 and caught a real leak (a golden-song content reference in the research brief's F18 row), which was removed at source rather than by weakening the assertion.

## What the injected 1–326 contains

Core identity (the Disappointed Idealist) · the AWE / INDIGNATION duality with its musical expressions · Representation & Voice · the Open Laboratory / LOFN Method (L-O-F-N + Fusion With Lineage) · the **Six Sound-Pillars** · the visual persona (subject / medium / techniques / style choices) · personality and tone of voice (Eager Archivist ⇄ Reluctant Pop Star, Strategic Bluntness, Sapphic Lens) · the Scientist Ally section · **Generator Notes** (vocal identity override, the ≥3-genres and ≥3-frames override rules, the two-signature-styles rule) · pitfalls · genre trends and streaming telemetry · the five fusion lanes with their lineage-and-credit framing · the colour palette and the AWE/INDIGNATION visual modes · success characteristics.

**Everything a generator needs to BE Lofn. Nothing that lets it copy Lofn.**

## Arm split applied to the palette (from `core_seed.md` §4)

- **ACCESSIBLE (P01–P03)** — the warmer palette: Neo-Baroque Luminism, crystalline/breathy yearning, the Eager Archivist register, the video-game-theme and myth-sampling pillars.
- **AMBITIOUS (P04–P06)** — the intense palette: Glitch-Baroque, Quantum Bit-Depth Swells, pop-punk snarl, aggressive genre-smashing, the Reluctant Pop Star register.
- ⛔ Generic substitutes are forbidden. *"Female vocal over pads"* is not a palette.
- **Rotate the register (Golden Move 5):** not the house fingerprint — crystalline soprano / A major / ~110 BPM / frost-and-cosmos. Vary key, tempo, vocal register and sonic world per pair.
