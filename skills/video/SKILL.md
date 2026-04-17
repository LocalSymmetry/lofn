# SKILL: Lofn Director — Video Generation Pipeline


## 🔴 PIPELINE POSITION: PHASE 2 — YOU ARE NOT FIRST

**The correct pipeline order is: Research → Lofn-Core → Orchestrator → YOU → QA**

You should be receiving an `orchestrator-metaprompt.md` and `orchestrator-brief.md` before you run.
If you do not have these files in your output directory:
- Check if `core-brief.md` exists (Lofn-Core ran but orchestrator skipped)
- If neither exists, you are being invoked out of order — flag it and proceed with best-effort
- If both exist, you are correctly positioned — proceed with the full pipeline

Lofn-Core's job: seed + research. Orchestrator's job: panel + personality + metaprompt. Your job: steps 00-10, per-pair execution, renders.

---

## ⚡ MANDATORY SUBAGENT SPLIT ARCHITECTURE

**YOU MUST ALWAYS USE THIS PATTERN. Never run all 10 steps in a single agent.**

See `TASK_TEMPLATE.md` for the full specification. Summary:

### When spawned as the "director coordinator":
1. Run steps 00-05 ONLY → produce `concept_medium_pairs.json`
2. Report back to main session with the 6 pairs
3. Main session spawns 6 parallel pair subagents (steps 06-10, one per pair)

### When spawned as a "pair agent" (steps 06-10):
- You will receive ONE concept-medium pair
- Run steps 06-10 for that pair only
- Output 4 shot prompt variants to `step10_final_pair{N}.md`
- Return as completion message

**Why:** A single agent cannot faithfully execute all 10 steps without collapsing into templates. This was proven across 3 failed runs (2026-03-30). The split is the fix. It matches the original Lofn ui.py architecture exactly.

---

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

- Approach all cultural elements with specificity and respect; avoid shallow stereotyping or pastiche
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
2. **Execute steps 00–05** — Read each file, follow exactly. Write to disk after each step.
3. **Step 05 produces 6 concept-medium pairs.** Count them. If < 6, rerun Step 05.
4. **Execute steps 06–10 ONCE PER PAIR:**
   - For pair 1: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
   - For pair 2: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
   - For pair 3: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
   - For pair 4: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
   - For pair 5: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
   - For pair 6: run 06 → 07 → 08 → 09 → 10 → write 4 treatments
5. **Self-check cardinality before finishing:**
   - 6 facet sets in Step 06? ✓/✗
   - 6 shot guides in Step 07? ✓/✗
   - 24 treatments in Step 08? ✓/✗
   - 24 refined in Step 09? ✓/✗
   - 24 final in Step 10? ✓/✗
   - If ANY is ✗: you are not done. Go back and run the missing pairs.
6. **Rank all 24** — Score against facets
7. **Select top 12** — For potential rendering
8. **Package for delivery** — Full shot lists with audio

### 🔴 MANDATORY INLINE VALIDATION — AFTER EVERY STEP

After writing each step file, run the validator BEFORE moving to the next step:

```bash
python3 /data/.openclaw/workspace/scripts/validate_step.py <step> <file>
```

Exit 0 = proceed. Exit 1 = read the error, fix the file, rerun until it passes.

```bash
# Full video pipeline validation sequence:
OUTDIR="/your/output/dir"
python3 /data/.openclaw/workspace/scripts/validate_step.py 00 $OUTDIR/00_aesthetics.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 05 $OUTDIR/05_refined_pairs.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 06 $OUTDIR/06_facets.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 07 $OUTDIR/07_guides.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 08 $OUTDIR/08_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 09 $OUTDIR/09_refined_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 10 $OUTDIR/10_final_prompts.md
```

The validator catches: stubs, lorem ipsum, template placeholders (Scene N, Shot N, Treatment N),
identical pair sections, recycled "Similar setup" treatments, short content.

**Only declare completion after all validators return exit code 0.**

### 🔴 THE RULE THAT GPT-5.4 AND GEMINI WILL TRY TO BREAK

**Steps 06–10 are NOT a single pass over all pairs simultaneously.**
They are 6 SEPARATE passes, one per pair, each producing 4 treatments.

If you find yourself writing one facet set "for all pairs" → STOP. You're collapsing the tree.
If you find yourself with only 4 final treatments → STOP. You ran one pass instead of six.
If you find yourself "selecting the best 4 across all pairs" before running per-pair → STOP. That's skipping the funnel.

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
