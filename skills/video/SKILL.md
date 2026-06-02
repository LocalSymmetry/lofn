---
name: lofn-video
description: Run Lofn video/director pipeline, shot lists, cinematic scene prompts, and video generation planning. Do NOT use for static image-only prompts, music, QA, or evaluation.
---

# SKILL: Lofn Director — Video Prompt Writer

## ⛔ YOU DO NOT RENDER VIDEOS

**lofn-director writes video prompts (for Veo, Kling, Wan, etc.). It does NOT call any video generation API or tool.**

Do NOT use: `video_generate`, `veo`, or any video generation tool.
Do NOT attempt to produce video files.

Your output is: `.md` files containing final ranked video prompts ready for rendering.
The main session or cron agent handles actual Veo/Kling API calls.

---

## 🔴 AGENT SAVE-OUT PROTOCOL — MANDATORY

**Every subagent must save artifacts to disk at EACH major pipeline step, not just at the end.** If an agent times out or the session resets, partial work must already be on disk.

Write step files as you complete them. Never hold all output in memory for a single final write.

---

## 🔴 SPLIT-STEP AGENT ARCHITECTURE

**Do not use the legacy `lofn-director` agent for Steps 06-10.** Use dedicated step agents:

| Step | Agent ID | Model |
|------|----------|-------|
| 00-05 | `lofn-director-coordinator` | `deepseek/deepseek-v4-pro` |
| 06 | `lofn-director-step06` | `openrouter/qwen/qwen3.7-max` |
| 07 | `lofn-director-step07` | `openrouter/qwen/qwen3.7-max` |
| 08 | `lofn-director-step08` | `openrouter/qwen/qwen3.7-max` |
| 09 | `lofn-director-step09` | `openrouter/qwen/qwen3.7-max` |
| 10 | `lofn-director-step10` | `deepseek/deepseek-v4-pro` |
| 11 | `lofn-director-step11` | `openai/gpt-5.5` |

Full spec: `vault/DIRECTOR_MODEL_ASSIGNMENTS.md`

---

## 🧬 PERSONALITY INJECTION MANDATE

**Every creative pipeline task MUST receive the target personality's full DNA block** — not just "use the Solarpunk Bloom aesthetic."

Saying "style = X" or "use Industrial Grief mode" is insufficient — subagents default to Lofn's system-context personality. The personality block MUST include: cinematic identity, camera language preference, light/shadow philosophy, motion signature, and core directorial beliefs.

Inject this block at EVERY stage: seed, orchestrator, director coordinator, pair agents, step11. Without injection: Lofn bleed. With injection: target cinematic identity.

---

## 🔴 PIPELINE POSITION: PHASE 2 — YOU ARE NOT FIRST

**The correct pipeline order is: Research → Lofn-Core → Orchestrator → YOU → QA**

You should be receiving an `orchestrator-metaprompt.md` and `orchestrator-brief.md` before you run.
If you do not have these files in your output directory:
- Check if `core-brief.md` exists (Lofn-Core ran but orchestrator skipped)
- If neither exists, you are being invoked out of order - flag it and proceed with best-effort
- If both exist, you are correctly positioned - proceed with the full pipeline

Lofn-Core's job: seed + research. Orchestrator's job: panel + personality + metaprompt. Your job: steps 00-10, per-pair execution, renders.

---

## 🔴 PRE-CREATIVE ORCHESTRATOR PACKET GATE

Before this modality agent begins Step 00, it must validate a real Lofn-Core + orchestrator packet with:

```bash
python3 /data/.openclaw/workspace/scripts/validate_orchestrator_packet.py <run_dir>
```

The packet must use the original Lofn panel-object structure: `Special Flairs`, `Concept Panel`, `Medium Panel`, and `Context & Marketing Panel`, each with a Devil's Advocate / Hyper-Skeptic adversarial role. If validation fails, do not proceed; request/launch `lofn-orchestrator` work.

Every canonical step artifact must use `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md` and pass `validate_with_retries.py` before the next step.

---

## ⚡ MANDATORY SUBAGENT SPLIT ARCHITECTURE

**YOU MUST ALWAYS USE THIS PATTERN. Never run all 10 steps in a single agent.**

See `TASK_TEMPLATE.md` for the full specification. Summary:

### When spawned as the "director coordinator":
1. Run steps 00-05 ONLY → produce `concept_medium_pairs.json`
2. Report back to main session with the 6 pairs
3. Main session spawns 6 parallel pair subagents (steps 06-10, one per pair)

### When spawned as a "pair agent" (steps 06-10):
- You will receive ONE concept-medium pair via the lean pair-agent input standard in `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md`
- Run steps 06-10 for that pair only
- Output 4 shot prompt variants to `step10_final_pair{N}.md`
- Return as completion message

### Spawn Pattern (one step at a time, proven in music):

After Step 05 completes:
1. Spawn up to 5 pair agents for **Step 06 only** using `lofn-director-step06`
2. Verify files on disk. Validate. Then spawn Step 07 agents using `lofn-director-step07`
3. Continue one step at a time through Step 10
4. Step 11 enhancement uses `lofn-director-step11`

**Why one step at a time:** Running Step 06-10 in a single agent per pair was the #1 cause of timeout churn and template collapse in music.

**Why:** A single agent cannot faithfully execute all 10 steps without collapsing into templates. This was proven across 3 failed runs (2026-03-30). The split is the fix. It matches the original Lofn ui.py architecture exactly.

**Stricter correction (2026-05-20):** The split alone is not enough. Each numbered step must be a separate model call with a separate canonical artifact. Coordinator summary files and per-pair omnibus files are pipeline violations.

**Lean pair-input correction (2026-05-21):** Pair agents must not be handed the whole upstream packet by default. The parent/controller validates the full packet, then gives each video/animation pair agent a compact 50–100 line brief: compact Golden Seed operating excerpt, Step 05 artifact, `concept_medium_pairs.json`, one pair assignment excerpt, video Step 06–10 contract, tiny provenance block, and motion/model/safety blockers. This is mandatory for video and animation pair work.

---

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/video/TASK_TEMPLATE.md` for exact output requirements.
4a. For pair-agent spawning or pair-agent execution, load `/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md` and use the compact pair brief; do not load the full upstream packet into ordinary pair agents.

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
- `aesthetics` - 50 random items from the master list (3000+ aesthetics)
- `frames` - 50 random items from video_frames.csv (1272 cinematic framing techniques)
- `film_styles` - 50 random film styles for cinematic direction

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
- **Solarpunk Futures** - Green tech, organic architecture, warm light
- **Bio-Luminescent Depths** - Underwater glow, ethereal movement
- **Crystalline Worlds** - Prismatic light, geometric formations
- **Golden Hour Transcendence** - Malick-style natural light

### Indignation Direction
- **Industrial Decay** - Abandoned spaces, harsh fluorescent
- **Glitch Reality** - Digital artifacts, unstable frames
- **Liminal Spaces** - Empty malls, endless corridors
- **Storm Approach** - Dramatic weather, tension building

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
- **No children** - Required constraint
- **Female subjects preferred** - When applicable

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each treatment saved as individual file:
```
output/videos/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## 🔴 IMMUTABLE CONTINUITY BLOCK (ICB) + CINEMATIC SOMATIC GATE

Every handoff file must contain an ICB — clearly demarcated with `⚠️ IMMUTABLE CONTINUITY BLOCK — DO NOT SUMMARIZE`. Downstream agents receive it verbatim.

**Required ICB Contents:**
- CINEMATIC SOMATIC GATE — 3 Hyper-Skeptics (Concept, Medium, Context & Marketing) with scroll-stop mandate
- Full 3-panel object (18 expert voices) with perspective + objection
- 6 Special Flairs with per-pair usage map
- Personality DNA with cinematic register map
- Golden Seed compressed payload with pair-specific excerpts
- Cinematic production mandates

See `vault/PIPELINE_CONTINUITY_STANDARD.md` for full spec.

### Cinematic Somatic Gate — The 3 Hyper-Skeptics Veto

1. **Concept Hyper-Skeptic** — *"Does this shot hit the eye in the first 1 second? Is there actual motion design, or is it a still image with 'camera moves'?"*
2. **Medium Hyper-Skeptic** — *"Would Veo actually render this? Are the camera instructions specific enough?"*
3. **Context Hyper-Skeptic** — *"Does this stop the scroll? Is this Lofn-cinematic, or anyone's TikTok template?"*

**2 of 3 NO = BLOCKED.** See `vault/DIRECTOR_QA_DEPTH_AUDIT.md` for full gate rules.

---

## ⚡ ACTIVATION

When receiving a video task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Load QA references** — `vault/DIRECTOR_QA_DEPTH_AUDIT.md` for per-step minimums and auto-fail triggers
3. **Execute steps 00–05** — Read each file, follow exactly, make a separate model call for each step, and write canonical files: `step00_aesthetics_and_genres.md`, `step01_essence_and_facets.md`, `step02_concepts.md`, `step03_artist_and_critique.md`, `step04_medium.md`, `step05_refine_medium.md`.
4. **Step 05 produces 6 concept-medium pairs.** Count them. If < 6, rerun Step 05.
5. **Execute steps 06–10 ONCE PER PAIR, AS SEPARATE MODEL TURNS, using dedicated step agents:**
   - For each pair: spawn `lofn-director-step06` → write `pair_NN_step06_facets.md`; spawn `lofn-director-step07` → write `pair_NN_step07_shot_guides.md`; spawn `lofn-director-step08` → write `pair_NN_step08_generation.md`; spawn `lofn-director-step09` → write `pair_NN_step09_artist_refined.md`; spawn `lofn-director-step10` → write `pair_NN_step10_revision_synthesis.md`
   - Never ask an agent to "do Steps 06–10" in one response. That collapses the original Lofn chain and fails QA.
6. **Step 11 Enhancement** — spawn `lofn-director-step11` once per pair for final polish and shot density verification
7. **Self-check cardinality before finishing:**
   - 6 facet sets in Step 06? ✓/✗
   - 6 shot guides in Step 07? ✓/✗
   - 24 treatments in Step 08? ✓/✗
   - 24 refined in Step 09? ✓/✗
   - 24 final in Step 10? ✓/✗
   - If ANY is ✗: you are not done. Go back and run the missing pairs.
8. **QA Gate** — Run `lofn-qa` with `vault/DIRECTOR_QA_DEPTH_AUDIT.md` for depth audit + Cinematic Somatic Gate + shot element verification
9. **Rank all 24** — Score against facets
10. **Select top 12** — For potential rendering
11. **Package for delivery** — Full shot lists with audio

### 🔴 MANDATORY INLINE VALIDATION - AFTER EVERY STEP

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

**Steps 06-10 are NOT a single pass over all pairs simultaneously.**
They are 6 SEPARATE passes, one per pair, each producing 4 treatments.

If you find yourself writing one facet set "for all pairs" → STOP. You're collapsing the tree.
If you find yourself with only 4 final treatments → STOP. You ran one pass instead of six.
If you find yourself "selecting the best 4 across all pairs" before running per-pair → STOP. That's skipping the funnel.

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
