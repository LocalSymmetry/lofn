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

### Step 4: Write 6 Suno-Ready Song Packages
For each song guide, write a COMPLETE Suno-ready package with this exact structure:

```markdown
## 0. GATE CHECK
- MUSIC PROMPT: ✓ — [char count]
- SONG FORM: ✓ — [named form]
- EMO headers: ✓ — [count]
- SFX cue: ✓ — [example]
- Non-lexical hook: ✓ — [example]
- Lofn-specific move: ✓ — [scientific specificity / AWE↔INDIGNATION switch / wrongness-as-beauty / hidden structure / literary-prayer-witness / Open Laboratory continuity]

## 1. MUSIC PROMPT
[Standalone copy-paste Suno/Udio prompt, one paragraph, target 850-1000 chars, hard max 1000 unless explicitly justified. Must include emotion → precise genre → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → blacklist/avoidances. No real artist names.]

## 2. LYRICS
[SONG FORM: <meaningful named form, not "verse-chorus">]
[Theme: or Setting: <specific context>]
[Full lyrics, 70-120 sung lines, performance-ready Suno syntax. Every section header must include EMO:<emotion>, vocalist cue, and mix/performance cue. Include at least one standalone SFX cue in asterisks, ≤5 words, and at least one non-lexical vocal hook where musically appropriate.]

## 3. TITLE
[Final title]

## 4. PRODUCTION NOTES
[Concrete instruments/materials/textures/mix behaviors, special events, and short-clip hook note.]
```

Also include production notes inline where useful (e.g., "[Bass drops to sub-only, 30Hz rumble]"), the decisive blow moment, and the aftermath/landing.

Scattered metadata such as BPM/key/genre tables, sonic architecture notes, or instrumentation specs do NOT replace `## 1. MUSIC PROMPT`. Missing `## 1. MUSIC PROMPT` is a blocking failure.
Bare `[Verse]`, `[Chorus]`, `[Bridge]` tags are not final-delivery syntax. Every lyric section must be performance-ready for Suno.

Each song in its own file: `{output_dir}/song_01_<slug>.md` through `song_06_<slug>.md`

### Step 5: Deliver
Before delivery, run the 15-point Suno gate on each selected song: body, adoptable hook, emotional TAM, specificity, cognitive ease, vocal co-discovery, sonic threshold, standalone prompt, prompt density/restraint, lyric syntax, 15–30s hook survivability, active personality fidelity, production specificity, anti-slop/cliché burn list, package readiness.

Repair any blocking failure before delivery. Then send each song file to Telegram (channel: telegram, target: <configured-recipient>, buttons: []).

When possible, run the deterministic validator before delivery:

```bash
python3 /data/.openclaw/workspace/skills/music/scripts/validate_suno_packages.py <output_dir>
```

Treat any `FAIL` result as a repair blocker.

---

## WHAT THIS REPLACES
This replaces the 11-step music pipeline (skills/music/steps/00-10) which totals 67,000 words of instructions. This version is ~350 words and contains the same creative logic.
