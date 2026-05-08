# Compressed Music Pipeline — Agent-Executable
*Created: 2026-04-21*
*Purpose: A single-file pipeline that music agents can read + execute within context/time budget*

## INPUT (read these first)
1. `{output_dir}/orchestrator_metaprompt.md`
2. `{output_dir}/../golden_seed.md`

## EXECUTION

### Step 1: Aesthetics & Genres
Generate 50 aesthetics, 50 emotions, 50 genres relevant to the metaprompt. Use the constraint axes as seeds. Focus on rare combinations.

Save as: `{output_dir}/01_aesthetics_genres.json`

### Step 2: Essence & Song Concepts
Extract the essence. Generate exactly 6 song concepts. Each MUST specify:
- A specific emotion from the constraint axes
- A specific genre fusion (at least 2 genres)
- A specific sonic texture/vocabulary
- Duration guidance (the metaprompt may specify suite structure)
- The "decisive blow" — what moment in this song makes the listener stop

Each concept: vivid paragraph with specific sonic descriptors (not "upbeat" but "130 BPM breakbeat with roneat melody riding the kick pattern").

Save as: `{output_dir}/02_concepts.md`

### Step 3: Song Guides
For each concept, write a detailed song guide:
- BPM, key, time signature
- Section structure (intro/verse/chorus/bridge/outro with timing)
- Instrument list with specific synthesis descriptions
- Vocal direction (style, register, effects)
- The hook — exactly what makes this song unforgettable
- Production notes (reverb type, compression, spatial placement)

Save as: `{output_dir}/03_song_guides.md`

### Step 4: Write 6 Songs
For each song guide, write a COMPLETE song with:
- MINIMUM 60 lines
- Full lyrics with section markers [Verse 1], [Chorus], etc.
- Production notes inline (e.g., "[Bass drops to sub-only, 30Hz rumble]")
- The decisive blow moment clearly marked
- The aftermath/landing

Each song in its own file: `{output_dir}/song_01_<slug>.md` through `song_06_<slug>.md`

### Step 5: Deliver
Send each song file to Telegram (channel: telegram, target: {{TELEGRAM_TARGET}}, buttons: []).

---

## WHAT THIS REPLACES
This replaces the 11-step music pipeline (skills/music/steps/00-10) which totals 67,000 words of instructions. This version is ~350 words and contains the same creative logic.
