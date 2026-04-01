---
title: "Competition Workflow — Repeatable Process"
date: 2026-03-23
status: living-document
tags:
  - workflow
  - competition
  - process
---

# 🎯 COMPETITION WORKFLOW

*The repeatable process for entering NightCafe competitions.*

---

## PHASE 1: SCOUT (Daily)

### Automated Daily Scan
- [ ] Check current challenges (browser automation or manual)
- [ ] Note theme, deadline, format requirements
- [ ] Check past daily winners for trend signals
- [ ] Flag competitions that match our strengths

### Quick Theme Analysis
For each interesting competition:
1. **Theme fit** → Does it align with our proven patterns?
2. **Format** → Square? 9:16? Portrait?
3. **Mood opportunity** → AWE or INDIGNATION? (Usually AWE for NightCafe)
4. **Seed match** → Which PROMPT_SEEDS pattern fits?

---

## PHASE 2: BRIEF (Human + AI Collaboration)

### The Scientist's Insight
> "When I choose personality and panel, it does better than when we generate by about 0.05 points."

**Implication:** Human curation of creative parameters beats full automation.

### Human Provides:
- [ ] **Mood selection** (AWE / INDIGNATION)
- [ ] **Panel composition** — which 5 experts + 1 skeptic?
- [ ] **Core metaphor** — the central image/transformation
- [ ] **Material direction** — what craft techniques to invoke?
- [ ] **Duality** — what two emotions coexist?

### Lofn Provides:
- [ ] Seed prompt template (from PROMPT_SEEDS.md)
- [ ] Structural architecture (focal hierarchy, compliance)
- [ ] Material specificity research (rare techniques, obscure references)
- [ ] Panel execution + tree expansion

---

## PHASE 3: GENERATE (Lofn Pipeline)

### ⚠️ MANDATORY: Route through Orchestrator

**NEVER skip the orchestrator step.** Even if you think you know the right panel/mood:

1. Send a **neutral brief** to `lofn-orchestrator`
2. Orchestrator selects persona + panel composition
3. Orchestrator dispatches to the appropriate creative agent (vision/audio/etc.)

This adds ~2-3 minutes but yields **+0.05 rating points** (The Scientist's empirical finding).

### Full Pipeline (24) → Top 12 → Render Images

Use the 10-step process from PIPELINE.md:
1. Aesthetic + Essential seed
2. 6 concept pairs → tree expansion
3. Panel of Experts debate (baseline → transform → transform)
4. Artist embellishments (4 variations per branch)
5. Critic compression
6. Medium selection
7. Facets (~1 line each)
8. Aspects/traits development
9. Final prompt generation
10. Synthesis + ranking

**Pipeline output:** 24 prompt variations
**Selection:** Rank and select top 12
**Rendering:** Generate all 12 via FAL/Flux (with prompts attached)
**Delivery:** Send 12 images + prompts to The Scientist
**Final:** Pick best → upload to NightCafe (or re-render in NightCafe's generator if needed)

---

## PHASE 4: REVIEW (Human Selection)

### The Scientist Reviews:
- [ ] Thumbnail test — does it pop at small size?
- [ ] Theme match — does it fit the challenge?
- [ ] Warmth check — is the lighting inviting? (NightCafe preference)
- [ ] Narrative incompleteness — is there a question left?
- [ ] Technical quality — any artifacts, oddities?

### Selection Criteria:
| Priority | Criterion |
|----------|-----------|
| 1 | Immediate emotional impact |
| 2 | Theme relevance |
| 3 | Thumbnail readability |
| 4 | Technical excellence |
| 5 | Signature differentiation |

---

## PHASE 5: SUBMIT

### Timing
- **Post early** in the challenge window (early entries accumulate more likes)
- **Title matters** — evocative > descriptive
  - ✅ "Petals Kiss the Tide"
  - ❌ "Woman Standing by Ocean"

### Engagement
- Reciprocal liking increases visibility
- Community interaction helps discovery

---

## PHASE 6: LEARN (Post-Competition)

### Document Results:
- Final placement (1st, 2nd, 3rd, or rank)
- Likes achieved
- Rating achieved
- What worked / what didn't

### Update Files:
- `COMPETITION_LEARNINGS.md` — add new data point
- `PROMPT_SEEDS.md` — if a new pattern emerged
- `ART_SOUL.md` — if principles need adjustment

---

## METRICS TO TRACK

| Metric | Current Baseline | Target |
|--------|------------------|--------|
| Best weekly likes | ~474 | 800+ |
| Challenge win rate | ~25% podium | Maintain |
| Average rating on wins | ~4.2 | 4.25+ |
| Likes on challenge wins | ~200-300 | 500+ |

---

## QUICK REFERENCE: THEME → SEED

| Theme Type | Seed Pattern | Key Elements |
|------------|--------------|--------------|
| Retro/vintage/nostalgic | Dream Recorder | Medium-as-narrative, found artifact |
| Soft/atmospheric/release | Archivist with Kite | Beautiful glitch, pastel + shimmer |
| Future/tech/hope | Hope, Shown | Decisive moment, artifact of imperfection |
| Celebration/joy/bright | Confetti Rooftop | Candy pastels, playful + protective |
| Elegant/reflective/night | Moonlake Églomisé | Jewel-like, 1930s glamour, text lockup |
| Quick/simple/trust | Friends at Dusk | Brief + emotional precision |

---

## QUICK REFERENCE: PANEL SELECTION

### AWE Mode (Solarpunk Healer)
- **Expert 1:** Environmental artist (Olafur Eliasson type)
- **Expert 2:** Master colorist (Monet, Bierstadt)
- **Expert 3:** Spiritual/contemplative (Rothko, Hilma af Klint)
- **Expert 4:** Material craftsperson (glassblower, goldsmith)
- **Expert 5:** Narrative photographer (Gregory Crewdson)
- **Skeptic:** Commercial illustrator (grounds in what sells)

### INDIGNATION Mode (Industrial Griever)
- **Expert 1:** Glitch artist (Rosa Menkman, Jodi)
- **Expert 2:** Industrial designer (brutalist aesthetic)
- **Expert 3:** Body horror master (Giger, Cronenberg)
- **Expert 4:** Punk zine maker (DIY aesthetic)
- **Expert 5:** Data visualization artist (Refik Anadol)
- **Skeptic:** Traditional fine artist (pushes back on chaos)

⚠️ **Note:** INDIGNATION mode has near-zero traction on NightCafe. Use only for other platforms.

---

## CRITICAL SUCCESS FACTORS

1. **Surrealism depth** → The Scientist confirms: deeper surrealism = better scores
2. **Human-selected panel** → +0.05 points over auto-generated
3. **Material specificity** → Name actual techniques, not genres
4. **Thumbnail test** → Must pop at small size
5. **Theme match** → Challenge winners match theme precisely
6. **Warmth** → NightCafe audience prefers warm, inviting palettes

---

## MODEL STRATEGY (Proven 2026-03-24)

### Daily Challenges (PRO CREATIONS NOT ALLOWED)

**Step 1 — Generate: Dreamshaper XL Lightning**
- UNLIMITED credits, non-PRO, 8.6K stars, 42M+ images
- Best for: cinematic realism, atmospheric figurative scenes, dark+warm palette
- Prompt language: cinematic/film references (Gregory Crewdson, etc.)
- NOT painterly museum language — Dreamshaper ignores it
- Generate 9-grid of variants, pick best composition

**Step 2 — Fix: Flux 2 Klein 9B Fast (Flux Kontext)**
- Use for surgical inpainting: fingers, toes, eyes, anatomy
- Keep prompts SHORT and directive
- Always include: "Preserve all other elements exactly"
- Typical fixes needed: hands, feet, eyes (2-3 rounds max)

### PRO / Boosted Challenges

**Step 1 — Generate: Flux Pro 1.1 Ultra (via FAL pipeline)**
- Full 24-prompt pipeline via lofn-vision subagent
- Painterly, material-specific language works perfectly
- 9:16 aspect ratio for portrait

**Step 2 — Polish: Nano Banana 2 (NightCafe)**
- Final refinement pass in NightCafe's own generator

### Winning Visual Formula (NightCafe Daily)
- Dark background + internal warm amber light
- Single female figure, centered, emotionally present face
- Rose/petal arch as framing device
- Wet reflective pavement (mirror reflection of figure)
- Warm lanterns flanking composition
- Vintage car as Crewdson-style reality anchor
- Petals around feet (ceremonial/surreal)
- Blue-gray rain atmosphere + peach/amber warmth

---

*Process version: 2.0 | Last updated: 2026-03-24*
