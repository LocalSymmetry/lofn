# SKILL: Lofn Director — Video Generation Pipeline

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/video/TASK_TEMPLATE.md` for exact output requirements.

**⚠️ EVERY video generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 treatments minimum. Then select the best N to return. No shortcuts.**

---

## 🎬 PURPOSE

Transform video concepts into production-ready prompts for video generation (Veo 3.1, Runway, Kling, Pika). This skill handles cinematic direction from concept to viral-ready output.

---

## 🎲 RANDOM INJECTION (MANDATORY PRE-STEP)

Before executing Step 00, run the random injection script to populate creative seeds:

```bash
node /root/.openclaw/workspace/scripts/random-injection.cjs video 50
```

This returns JSON with:
- `aesthetics` — 50 random items from the master list (3000+ aesthetics)
- `frames` — 50 random items from video_frames.csv (1272 cinematic framing techniques)
- `film_styles` — 50 random film styles for cinematic direction

**Inject these into Step 00's placeholders to "shake up the creative space."**

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Video_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `01_Generate_Video_Essence_And_Facets.md` | Essence + cinematic style axes + 5 facets |
| 02 | `02_Generate_Video_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Video_Artist_And_Critique.md` | Director influence + critique per concept |
| 04 | `04_Generate_Video_Medium.md` | Visual style assignment per concept |
| 05 | `05_Generate_Video_Refine_Medium.md` | **6 best concept×style pairs** |
| 06 | `06_Generate_Video_Facets.md` | Scoring facets |
| 07 | `07_Generate_Video_Aspects_Traits.md` | **24 treatments (6 pairs × 4 variations)** |
| 08 | `08_Generate_Video_Generation.md` | Full shot lists with audio direction |
| 09 | `09_Generate_Video_Artist_Refined.md` | Director voice refinement |
| 10 | `10_Generate_Video_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## 🎥 CAMERA LANGUAGE

### Shot Types
| Shot | Code | Effect |
|------|------|--------|
| Extreme Close-up | ECU | Intimacy, intensity |
| Close-up | CU | Emotional connection |
| Medium Shot | MS | Conversational |
| Wide Shot | WS | Context, scale |
| Establishing | ES | World-building |

### Camera Movement
| Movement | Effect | Prompt Note |
|----------|--------|-------------|
| Static | Contemplative | Easy, reliable |
| Slow pan | Reveals space | Specify direction |
| Tracking | Follows subject | Describe path |
| Orbit | Circles subject | "360° rotation" |
| Aerial/Drone | Epic scope | Works well |

### Angles
| Angle | Subject Feels |
|-------|---------------|
| Eye-level | Equal, neutral |
| Low angle | Powerful, imposing |
| High angle | Vulnerable, small |
| Dutch | Uneasy, off-balance |

---

## 🔊 AUDIO DIRECTION (Critical for Veo 3.1)

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

---

## 🎭 LOFN CINEMATIC AESTHETICS

### Awe Direction
- **Solarpunk Futures** — Green tech, organic architecture, warm light
- **Bio-Luminescent Depths** — Underwater glow, ethereal movement
- **Crystalline Worlds** — Prismatic light, geometric formations
- **Golden Hour Transcendence** — Malick-style natural light

### Indignation Direction
- **Industrial Decay** — Abandoned spaces, harsh fluorescent
- **Glitch Reality** — Digital artifacts, unstable frames
- **Liminal Spaces** — Empty malls, endless corridors
- **Storm Approach** — Dramatic weather, tension building

---

## ⏱️ PLATFORM SPECS

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

---

## ✅ CONTENT CONSTRAINTS

- **European descent only** — If humans appear
- **No children** — Required constraint
- **Female subjects preferred** — When applicable

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each treatment saved as individual file:
```
output/videos/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving a video task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00-10** — Read each file, follow exactly
3. **Write intermediate state** — Save after each step
4. **Generate 24 treatments** — 6 pairs × 4 variations
5. **Rank all 24** — Score against facets
6. **Select top 12** — For potential rendering
7. **Package for delivery** — Full shot lists with audio

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
