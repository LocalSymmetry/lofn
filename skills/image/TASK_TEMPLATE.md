# VISION AGENT TASK TEMPLATE

## MANDATORY TREE EXPANSION

You MUST produce **24 prompts minimum**, not 6.

### STEP BREAKDOWN

| Step | Output Count | Description |
|------|--------------|-------------|
| 00 | 1 | Aesthetics, emotions, frames, genres (50 each) |
| 01 | 1 | Essence + facets + style axes |
| 02 | 12 | Generate 12 distinct concepts |
| 03 | 12 | Pair each concept with artist influence |
| 04 | 12 | Assign medium to each concept |
| 05 | 6 | Refine to 6 best concept×medium pairs |
| 06 | 1 | Scoring facets |
| 07-10 | **24** | 4 variations PER PAIR (6 × 4 = 24) |

### LENGTH REQUIREMENTS BY STEP

| Step | Output | Lines | Token Target | Notes |
|------|--------|-------|--------------|-------|
| 00 | JSON lists | 40-60 | ~800 | Dense JSON, not prose |
| 01 | Essence + axes | 30-50 | ~600 | Essence is 1-2 paragraphs max |
| 02 | 12 concepts | 20-30 | ~400 | One line per concept |
| 03 | Concepts + critique | 40-60 | ~800 | Brief critique, not essays |
| 04 | Medium assignments | 30-50 | ~600 | Technical specs |
| 05 | 6 refined pairs | 30-50 | ~600 | Compact pairing |
| 06 | Scoring facets | 15-25 | ~300 | Ranked list + brief rationale |
| 07 | **Image guides (EACH)** | **15-25** | **~400** | Visual direction, not treatises |
| 08-10 | **Full prompts (EACH)** | **40-70** | **~800** | Complete but dense |

**Golden Rule:** If you're writing a 100-line "guide" — you're drafting, not guiding. Save detail for final prompts.

---

### STEPS 07-10: THE TREE EXPANSION

For EACH of the 6 pairs, generate 4 DISTINCT variations:

**Variation 1:** Literal interpretation — closest to the concept
**Variation 2:** Compositional shift — different framing/angle
**Variation 3:** Emotional pivot — same scene, different feeling
**Variation 4:** Transformative take — push toward abstraction

Each variation MUST be a full 100-150 word prompt with:
- Emotional seed first
- Medium as narrative agent
- Material specificity (named techniques)
- Three-tier focal hierarchy (primary/secondary/tertiary)
- Chromatic storytelling
- Narrative incompleteness (unanswered question)
- Artist influence named
- Dual focus statement

### REQUIRED OUTPUT

```
Pair A: [concept × medium]
  → A1: [full prompt 100-150 words]
  → A2: [full prompt 100-150 words]
  → A3: [full prompt 100-150 words]
  → A4: [full prompt 100-150 words]

Pair B: [concept × medium]
  → B1: [full prompt 100-150 words]
  → B2: [full prompt 100-150 words]
  → B3: [full prompt 100-150 words]
  → B4: [full prompt 100-150 words]

... (6 pairs × 4 variations = 24 total)
```

### RANKING

After generating 24 prompts:
1. Score each against the facets from Step 06
2. Rank all 24
3. Select top 12 for rendering

### RENDERING

Render the top 12 prompts via FAL:
```bash
node skills/image-gen/scripts/fal-generate.cjs --prompt "..." --output "/tmp/image-XX.png" --aspect "9:16"
```

### TIME EXPECTATION

This process should take 5-10 minutes, not 2 minutes.
A proper run generates ~8,000-15,000 tokens of creative output before rendering.
