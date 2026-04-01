# AUDIO AGENT TASK TEMPLATE

## MANDATORY TREE EXPANSION

You MUST produce **24 songs minimum**, not 4-6.

### STEP BREAKDOWN

| Step | Output Count | Description |
|------|--------------|-------------|
| 00 | 1 | Aesthetics, emotions, frames, genres (50 each) |
| 01 | 1 | Essence + facets + style axes (10 musical axes) |
| 02 | 12 | Generate 12 distinct concept pairs |
| 03 | 12 | Artist influence + Critic compression per concept |
| 04 | 12 | Medium assignment (genre fusion) per concept |
| 05 | 6 | Refine to 6 best concept×medium pairs |
| 06 | 1 | Scoring facets |
| 07 | 24 | Song Guides: 4 variations PER PAIR (6 × 4 = 24) |
| 08-10 | 24 | Full Songs: 4 finals PER PAIR (6 × 4 = 24) |

### LENGTH REQUIREMENTS BY STEP

| Step | Output | Lines | Token Target | Notes |
|------|--------|-------|--------------|-------|
| 00 | JSON lists | 40-60 | ~800 | Dense JSON, not prose |
| 01 | Essence + axes | 30-50 | ~600 | Essence is 1-2 paragraphs max |
| 02 | 12 concepts | 20-30 | ~400 | One line per concept |
| 03 | Concepts + critique | 40-60 | ~800 | Brief critique, not essays |
| 04 | Genre assignments | 30-50 | ~600 | Technical specs |
| 05 | 6 refined pairs | 30-50 | ~600 | Compact pairing |
| 06 | Scoring facets | 15-25 | ~300 | Ranked list + brief rationale |
| 07 | **Song guides (EACH)** | **20-30** | **~500** | Direction, not drafts |
| 08-10 | **Full songs (EACH)** | **80-120** | **~1500** | Lyrics + prompt + notes |

**Golden Rule:** If you're writing a 120-line "guide" — you're drafting, not guiding. Save detail for final songs.

---

### STEPS 07-10: THE TREE EXPANSION

For EACH of the 6 pairs, generate 4 DISTINCT song variations:

**Variation 1:** Literal interpretation — closest to the concept
**Variation 2:** Emotional pivot — same concept, different feeling
**Variation 3:** Genre shift — push the fusion further
**Variation 4:** Transformative take — most experimental version

### SONG GUIDE FORMAT (Step 07)

Each guide: **20-30 lines**, including:

```
# Song Guide: [Title]

## Concept Pair
[concept × genre fusion]

## Musical DNA
- **Genre:** [specific fusion]
- **BPM:** [tempo]
- **Key:** [key + mode]
- **Vocal Style:** [crystalline/bratty/yearning/etc.]

## Structure
[Intro] → [Verse 1] → [Pre-Chorus] → [Chorus] → ...

## Production Notes
[5-7 specific production decisions]

## Hook Concept
[The singular earworm element]

## Bold Choice
[What makes this song singular]
```

### FULL SONG FORMAT (Steps 08-10)

Each final song: **80-120 lines**, including:

```
# [Title]

## Song Prompt (for Suno)
[100-150 words: genre, instrumentation, tempo, mood, texture, production style]

## Lyrics
[Full lyrics with section tags: [Intro], [Verse 1], [Chorus], etc.]
[50-80 lines of actual lyrics]

## Production Notes
[Specific mixing/mastering directions]
```

### REQUIRED ELEMENTS

Every song MUST have:
- **Female vocals** (crystalline or bratty depending on mode)
- **TikTok-optimized hook** (15-30 second memorable cycle)
- **One BOLD choice** (unusual instrument, unexpected drop, genre collision)
- **Section tags** in lyrics ([Verse], [Chorus], [Bridge], etc.)
- **3-4 minute duration** (50-80 lines minimum lyrics)
- **Multiple verses + repeated chorus**

### LOFN SOUND IDENTITY

**AWE Mode (Default):**
- Crystalline, breathy yearning
- Solarpunk aesthetics, 432Hz tuning option
- Green synths, organic textures
- Complex polyrhythms that soothe

**INDIGNATION Mode (Triggered):**
- Bratty, glitched-out delivery
- Industrial textures, somatic bass (30-60Hz)
- Compressed, hard consonants
- Textures that demand to be felt physically

### GENRE FUSION PALETTE

| Fusion | Components | BPM | Vibe |
|--------|------------|-----|------|
| Piano Bounce | Amapiano × Jersey-Club | 115-120 | Log drum shuffle |
| Baile Phonk | Brazilian Funk + Dark Phonk | 140 | Detuned cowbell |
| HyperRaaga | South-Asian classical + hyperpop | 160 | Microtonal glitchcore |
| Gaelic Drill | Celtic folk + UK drill | 140 | Bagpipe over 808 |
| Amazonian Techno | Rainforest samples + 4x4 | 126 | Eco-solidarity |

### RANKING

After generating 24 songs:
1. Score each against the facets from Step 06
2. Rank all 24
3. Select top N (usually 4-6) for delivery

### TIME EXPECTATION

This process should take **8-15 minutes**, not 2 minutes.
A proper run generates **15,000-25,000 tokens** of creative output.

### OUTPUT CHECKLIST

Before completing, verify:
- [ ] 24 song guides generated (6 pairs × 4 variations)
- [ ] 24 full songs with lyrics and prompts
- [ ] Each song has section tags
- [ ] Each song has 50-80 lines of lyrics minimum
- [ ] Each song has a unique BOLD choice
- [ ] Ranking completed with scoring rationale
- [ ] Top N selected for delivery

---

*This pipeline won competitions. Trust it. Execute it fully.*
