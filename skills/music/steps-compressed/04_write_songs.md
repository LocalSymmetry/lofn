# Music Pipeline Step 4: Write Songs

Read:
1. `{metaprompt_path}`
2. `{output_dir}/03_song_guides.md`

For each song guide, write a COMPLETE Suno-ready package with:
- `## 0. GATE CHECK` — document: music prompt char count; named song form; EMO header count; one SFX cue; one non-lexical hook; one Lofn-specific move that survives in prompt + lyrics
- `## 1. MUSIC PROMPT` — REQUIRED standalone copy-paste Suno/Udio prompt, single paragraph, target 850-1000 chars, hard max 1000 unless explicitly justified. It must include: emotion → precise genre → vocalist spec → instrumentation/mix → chronological progression → bold sonic device → blacklist/avoidances. Scattered genre/tempo/key, sonic-world, metadata tables, or production notes do NOT satisfy this requirement.
- `## 2. LYRICS` — full lyrics, MINIMUM 70 sung lines, target 70-120 lines. Start with `[SONG FORM: <meaningful named form>]` and `[Theme:]` or `[Setting:]`.
- `## 3. TITLE` — final song title
- `## 4. PRODUCTION NOTES` — concrete instruments/materials/textures/mix behaviors, special events, and a short-clip hook note
- Full lyrics with section markers using the full emotional taxonomy: `[Verse 1 – EMO:tender grief – Vocalist – dry close]`
  - ⚠️ The `EMO:` prefix IS the emotion tag — use it to express the precise feeling for each section
  - Draw from `skills/lofn-core/refs/EMOTION_TAXONOMY.md` — choose the specific emotion or combination (e.g. `nostalgia + yearning`, `righteous fury`, `quiet dread`, `ecstatic release`)
  - NEVER use bare Lofn architectural states (`AWE:`, `INDIGNATION:`, `SYNTHESIS:`) as emotion labels — those are coarse categories, not section-level feeling
- Production notes inline (e.g., "[Bass drops to sub-only, 30Hz rumble]")
- At least one standalone SFX cue in asterisks, ≤5 words (e.g., `*phone buzz*`)
- At least one non-lexical vocal hook where musically appropriate (`mm`, `ooh`, `ah`, whispered echo, call-response fragment)
- The decisive blow moment clearly marked
- The aftermath/landing

Each song in its own file: `{output_dir}/song_01_<slug>.md` through `song_06_<slug>.md`

## Pre-completion gate — mandatory

Before writing each final file, verify and include/mentally enforce:
- [ ] `## 0. GATE CHECK` documents all required checks
- [ ] `## 1. MUSIC PROMPT` exists and is a standalone 850-1000 char prompt
- [ ] `## 2. LYRICS` exists with EMO-tagged performance headers
- [ ] `## 3. TITLE` exists
- [ ] `[SONG FORM: <named form>]` exists at top of lyrics block
- [ ] At least one SFX cue and one non-lexical hook exist
- [ ] `## 4. PRODUCTION NOTES` exists with concrete production specificity and short-clip hook note
- [ ] No real artist names in the prompt
- [ ] Prompt is not replaced by metadata, genre table, or sonic architecture notes

If any item fails, repair before delivery. Missing `## 1. MUSIC PROMPT` is a blocking failure even if lyrics and production notes are excellent.
