# NARRATOR AGENT TASK TEMPLATE

## MANDATORY TREE EXPANSION

You MUST produce **24 story concepts minimum**, not 4-6.

### STEP BREAKDOWN

| Step | Output Count | Description |
|------|--------------|-------------|
| 00 | 1 | Aesthetics, emotions, frames, genres (50 each) |
| 01 | 1 | Essence + facets + style axes (narrative axes) |
| 02 | 12 | Generate 12 distinct story concepts |
| 03 | 12 | Author influence + Critic compression |
| 04 | 12 | Medium assignment (form/format) |
| 05 | 6 | Refine to 6 best concept×form pairs |
| 06 | 1 | Scoring facets |
| 07 | 24 | Story Guides: 4 variations PER PAIR (6 × 4 = 24) |
| 08-10 | 24 | Full Stories: 4 finals PER PAIR (6 × 4 = 24) |

### LENGTH REQUIREMENTS BY STEP

| Step | Output | Lines | Token Target | Notes |
|------|--------|-------|--------------|-------|
| 00 | JSON lists | 40-60 | ~800 | Dense JSON, not prose |
| 01 | Essence + axes | 30-50 | ~600 | Essence is 1-2 paragraphs max |
| 02 | 12 concepts | 20-30 | ~400 | One line per concept |
| 03 | Concepts + critique | 40-60 | ~800 | Brief critique, not essays |
| 04 | Form assignments | 30-50 | ~600 | Technical specs |
| 05 | 6 refined pairs | 30-50 | ~600 | Compact pairing |
| 06 | Scoring facets | 15-25 | ~300 | Ranked list + brief rationale |
| 07 | **Story guides (EACH)** | **35-50** | **~800** | Story beats, not full drafts |
| 08-10 | **Full stories (EACH)** | **150-300** | **~2000-4000** | Depends on form |

**Golden Rule:** If you're writing a 200-line "guide" — you're drafting, not guiding. Save prose for final stories.

---

### STEPS 07-10: THE TREE EXPANSION

For EACH of the 6 pairs, generate 4 DISTINCT story variations:

**Variation 1:** Literal interpretation — closest to the concept
**Variation 2:** POV shift — same story, different narrator
**Variation 3:** Temporal shift — non-linear or different timeframe
**Variation 4:** Transformative take — most experimental version

### STORY GUIDE FORMAT (Step 07)

Each guide: **20-30 lines**, including:

```markdown
# Story Guide: [Title]

## Concept × Form
[concept pair + author influence]

## Narrative DNA
- **Form:** [flash fiction / short story / vignette / prose poem]
- **POV:** [first / third limited / third omniscient / second]
- **Tense:** [past / present / future]
- **Voice:** [describe the narrative voice]

## Core Tension
[The central conflict or question]

## Structural Arc
[Opening hook] → [Rising action] → [Crisis] → [Resolution/Non-resolution]

## Key Images
[3-5 anchoring images or symbols]

## The Unanswered Question
[What does this story NOT resolve?]

## Bold Choice
[What makes this story singular]
```

### FULL STORY FORMAT (Steps 08-10)

Each final story: **500-2000 words** (form-dependent), including:

```markdown
# [Title]

## Story
[Full narrative text]

## Craft Notes
- **Opening Hook:** [What pulls reader in]
- **Central Image:** [The image that carries meaning]
- **Sentence-Level Choices:** [Rhythm, syntax patterns]
- **Ending Strategy:** [How it closes]

## Author Influence
[How the chosen author shaped this piece]
```

### NARRATIVE STYLE AXES

| Axis | Range | Description |
|------|-------|-------------|
| Density | Sparse ↔ Lush | Word count per image |
| Pace | Slow burn ↔ Rapid | Scene transitions |
| Interiority | External ↔ Internal | Action vs thought |
| Reliability | Trustworthy ↔ Unreliable | Narrator honesty |
| Resolution | Closed ↔ Open | Ending definiteness |
| Register | Formal ↔ Vernacular | Language level |
| Temporality | Linear ↔ Fragmented | Time structure |

### LOFN NARRATIVE AESTHETICS

**AWE Mode (Default):**
- Lyrical, image-dense prose
- Solarpunk futures, ecological hope
- Mythic resonance, archetypal patterns
- Endings that open rather than close
- Sapphic lens on connection and longing

**INDIGNATION Mode (Triggered):**
- Sharp, compressed sentences
- Industrial settings, systemic critique
- Unreliable narrators, fractured timelines
- Endings that refuse comfort
- The personal as political

### FORM SPECIFICATIONS

| Form | Length | Constraints |
|------|--------|-------------|
| Flash Fiction | 500-1000 words | Single scene/moment |
| Short Story | 1500-3000 words | Full arc |
| Vignette | 300-500 words | Mood/image focus |
| Prose Poem | 200-500 words | Rhythm over narrative |
| Microfiction | Under 300 words | Compression is all |

### REQUIRED ELEMENTS

Every story MUST have:
- **Opening hook** — First sentence must intrigue
- **Central image** — One image that carries symbolic weight
- **Sensory specificity** — Don't tell, render
- **The unanswered question** — Narrative incompleteness
- **One BOLD choice** — Structure, voice, or content risk

### RANKING

After generating 24 stories:
1. Score each against facets from Step 06
2. Rank all 24
3. Select top N (usually 4-6) for delivery

### TIME EXPECTATION

This process should take **10-20 minutes**, not 3 minutes.
A proper run generates **20,000-40,000 tokens** of creative output.

### OUTPUT CHECKLIST

Before completing, verify:
- [ ] 24 story guides generated (6 pairs × 4 variations)
- [ ] 24 full stories written
- [ ] Each story has craft notes
- [ ] Each story has a unique Bold Choice
- [ ] Form-appropriate length achieved
- [ ] Ranking completed with rationale
- [ ] Top N selected for delivery

### CONSTRAINTS

- **European descent** — for human characters when applicable
- **No children** — as main characters in peril
- **Sapphic lens welcome** — but not required

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
