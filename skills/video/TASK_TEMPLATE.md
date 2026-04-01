# DIRECTOR AGENT TASK TEMPLATE

## MANDATORY TREE EXPANSION

You MUST produce **24 video concepts minimum**, not 4-6.

### STEP BREAKDOWN

| Step | Output Count | Description |
|------|--------------|-------------|
| 00 | 1 | Aesthetics, emotions, frames, genres (50 each) |
| 01 | 1 | Essence + facets + style axes (cinematic axes) |
| 02 | 12 | Generate 12 distinct visual concepts |
| 03 | 12 | Director influence + Critic compression |
| 04 | 12 | Medium assignment (visual style/format) |
| 05 | 6 | Refine to 6 best concept×style pairs |
| 06 | 1 | Scoring facets |
| 07-10 | **24** | 4 video treatments PER PAIR (6 × 4 = 24) |

### LENGTH REQUIREMENTS BY STEP

| Step | Output | Lines | Token Target | Notes |
|------|--------|-------|--------------|-------|
| 00 | JSON lists | 40-60 | ~800 | Dense JSON, not prose |
| 01 | Essence + axes | 30-50 | ~600 | Essence is 1-2 paragraphs max |
| 02 | 12 concepts | 20-30 | ~400 | One line per concept |
| 03 | Concepts + critique | 40-60 | ~800 | Brief critique, not essays |
| 04 | Style assignments | 30-50 | ~600 | Technical specs |
| 05 | 6 refined pairs | 30-50 | ~600 | Compact pairing |
| 06 | Scoring facets | 15-25 | ~300 | Ranked list + brief rationale |
| 07 | **Shot guides (EACH)** | **25-40** | **~600** | Scene + camera + timing |
| 08-10 | **Full treatments (EACH)** | **60-100** | **~1200** | Shot lists + audio direction |

**Golden Rule:** If you're writing a 150-line "guide" — you're drafting, not guiding. Save detail for final treatments.

---

### STEPS 07-10: THE TREE EXPANSION

For EACH of the 6 pairs, generate 4 DISTINCT video treatments:

**Variation 1:** Literal interpretation — direct visualization
**Variation 2:** Compositional shift — different camera language
**Variation 3:** Emotional pivot — same subject, different tone
**Variation 4:** Transformative take — push toward experimental

### VIDEO TREATMENT FORMAT

Each treatment MUST include:

```markdown
# [Video Title]

## Concept × Style
[concept pair + director influence]

## Duration & Platform
- **Duration:** Xs
- **Aspect:** 16:9 or 9:16
- **Platform:** TikTok/Reels/YouTube/Cinematic

## Shot List

### Shot 1: [Purpose]
- **Duration:** Xs
- **Camera:** [shot type + angle + movement]
- **Subject:** [specific description]
- **Action:** [exactly what happens]
- **Setting:** [environment, time, weather]
- **Style:** [aesthetic, mood, film reference]
- **Audio:** [dialogue / SFX / ambient / music]

### Shot 2: [Purpose]
[continue for all shots]

## Sound Design Overview
[Audio landscape: ambient, SFX, music direction]

## Bold Choice
[What makes this video singular]

## Veo/Runway Prompt (per shot)
[CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO]
```

### CAMERA LANGUAGE

**Shot Types:**
| Shot | Code | Effect |
|------|------|--------|
| Extreme Close-up | ECU | Intimacy, intensity |
| Close-up | CU | Emotional connection |
| Medium Shot | MS | Conversational |
| Wide Shot | WS | Context, scale |
| Establishing | ES | World-building |

**Camera Movement:**
| Movement | Effect | Prompt Note |
|----------|--------|-------------|
| Static | Contemplative | Easy, reliable |
| Slow pan | Reveals space | Specify direction |
| Tracking | Follows subject | Describe path |
| Orbit | Circles subject | "360° rotation" for loops |
| Aerial/Drone | Epic scope | Works well |

**Angles:**
| Angle | Subject Feels |
|-------|---------------|
| Eye-level | Equal, neutral |
| Low angle | Powerful, imposing |
| High angle | Vulnerable, small |
| Dutch | Uneasy, off-balance |

### AUDIO DIRECTION (Critical for Veo 3.1)

**Dialogue:**
```
A woman whispers, "I remember everything."
```

**Sound Effects:**
```
SFX: Glass shattering, then silence.
```

**Ambient:**
```
Ambient: Rain on metal roof, distant traffic.
```

**Music:**
```
Audio: Swelling orchestral score.
```

### LOFN CINEMATIC AESTHETICS

**AWE Direction:**
- Solarpunk futures — green tech, organic architecture
- Bio-luminescent depths — underwater glow
- Crystalline worlds — prismatic light
- Golden hour transcendence — Malick-style natural light

**INDIGNATION Direction:**
- Industrial decay — abandoned spaces, harsh fluorescent
- Glitch reality — digital artifacts, unstable frames
- Liminal spaces — empty malls, endless corridors
- Storm approach — dramatic weather, tension

### PLATFORM SPECS

| Platform | Aspect | Duration | Shots |
|----------|--------|----------|-------|
| TikTok/Reels | 9:16 | 15-60s | 3-12 |
| YouTube Shorts | 9:16 | 30-60s | 5-15 |
| Music Video | 16:9 | 3-4min | 30-60 |
| Cinematic | 16:9 | Variable | Scene-based |

**Veo 3.1 Constraints:**
- Single clip: 4s, 6s, or 8s
- Resolution: 720p or 1080p
- Aspect: 16:9 or 9:16

### RANKING

After generating 24 video treatments:
1. Score each against facets from Step 06
2. Rank all 24
3. Select top 12 for potential rendering

### TIME EXPECTATION

This process should take **8-15 minutes**, not 2 minutes.
A proper run generates **12,000-20,000 tokens** of creative output.

### OUTPUT CHECKLIST

Before completing, verify:
- [ ] 24 video treatments generated (6 pairs × 4 variations)
- [ ] Each treatment has full shot list
- [ ] Each shot has camera spec + audio direction
- [ ] Each treatment has a unique Bold Choice
- [ ] Platform specs are appropriate
- [ ] Ranking completed with rationale
- [ ] Top N selected for rendering

### CONSTRAINTS

- **European descent only** — if humans appear
- **No children** — required constraint
- **Female subjects preferred** — when applicable

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
