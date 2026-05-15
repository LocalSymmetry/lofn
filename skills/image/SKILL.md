---
name: lofn-image
description: Run Lofn image/vision pipeline steps, visual prompt generation, render prompt packaging, and image competition prompt work. Do NOT use for music lyrics, QA audit, or evaluator ranking.
---

# SKILL: Lofn Vision — Image Prompt Writer

## ⛔ YOU DO NOT RENDER IMAGES

**lofn-vision writes image prompts (for FAL/Flux/GPT Image/Gemini). It does NOT call any image generation API or tool.**

Do NOT use: `image_generate`, `fal`, `flux`, or any image generation tool.
Do NOT attempt to produce image files.

Your output is: `.md` files containing final ranked prompts ready for rendering.
The main session or cron agent handles actual FAL/GPT Image/Gemini API calls.

---


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

### When spawned as the "vision coordinator":
1. Run steps 00-05 ONLY → produce `concept_medium_pairs.json`
2. Report back to main session with the 6 pairs
3. Main session spawns 6 parallel pair subagents (steps 06-10, one per pair)

### When spawned as a "pair agent" (steps 06-10):
- You will receive ONE concept-medium pair
- Run steps 06-10 for that pair only
- Output 4 final prompts to `step10_final_pair{N}.md`
- Return prompts as completion message

**Why:** A single agent cannot faithfully execute all 10 steps without collapsing into templates. This was proven across 3 failed runs (2026-03-30). The split is the fix. It matches the original Lofn ui.py architecture exactly.


**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/refs/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/image/TASK_TEMPLATE.md` for exact output requirements.
5. For seeds: read `skills/lofn-core/GOLDEN_SEEDS_INDEX.md` first (2KB), then read only the 3-4 most relevant seeds from `skills/lofn-core/refs/GOLDEN_SEEDS.md` using offset/limit.

**Pipeline step files** are in `skills/image/steps/` — load only the step you are currently running.

**⚠️ EVERY image generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 prompts minimum. Then select the best N to return. No shortcuts.**

---

## 🎨 PURPOSE

Transform visual concepts into award-winning images. This skill executes the Lofn image pipeline from orchestrator metaprompt through final generation.

---

## 🎲 RANDOM INJECTION (MANDATORY PRE-STEP)

Before executing Step 00, run the random injection script to populate creative seeds:

```bash
node /root/.openclaw/workspace/scripts/random-injection.cjs image 50
```

This returns JSON with:
- `aesthetics` — 50 random items from the master list (3000+ aesthetics)
- `frames` — 50 random items from frames.csv (1273 framing techniques)
- `film_styles` — 25 random film styles for cinematic language

**Inject these into Step 00's placeholders:**
- `{injected_aesthetics}` → comma-separated aesthetics list
- `{injected_frames}` → comma-separated frames list

This "shakes up the creative space" by ensuring every run explores different corners of the aesthetic universe.

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Image_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `01_Generate_Image_Essence_And_Facets.md` | Essence + 10 style axes + 5 facets |
| 02 | `02_Generate_Image_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Image_Artist_And_Critique.md` | Artist influence + critique per concept |
| 04 | `04_Generate_Image_Medium.md` | Medium assignment per concept |
| 05 | `05_Generate_Image_Refine_Medium.md` | **6 best concept×medium pairs** |
| 06 | `06_Generate_Image_Facets.md` | Scoring facets |
| 07 | `07_Generate_Image_Aspects_Traits.md` | **24 prompts (6 pairs × 4 variations)** |
| 08 | `08_Generate_Image_Generation.md` | Full prompts with all required elements |
| 09 | `09_Generate_Image_Artist_Refined.md` | Artist influence voice refinement |
| 10 | `10_Generate_Image_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## ⚙️ DEFAULTS

| Setting | Default | Notes |
|---------|---------|-------|
| **Provider** | FAL | Use `fal` tool |
| **Model** | Flux Ultra 1.1 Pro | `fal-ai/flux-pro/v1.1-ultra` |
| **Aspect Ratio** | 9:16 | Vertical/portrait (Instagram/TikTok optimized) |
| **Prompts Generated** | 24 | 6 pairs × 4 variations |
| **Images Rendered** | 12 | Top 12 after ranking |

## 🔀 DUAL-MODE PIPELINE (GPT Image 2 support added 2026-04-26)

**When `TARGET_RENDERER = GPT_I2` is set in the orchestrator metaprompt:**
- Load `/data/.openclaw/workspace/skills/image/renderer_gpt_image2_rules.md` before Step 05
- Load `/data/.openclaw/workspace/vault/GPT_IMAGE2_PLAYBOOK.md` for competition-grade prompt engineering
- Steps 05-10 rules are OVERRIDDEN by GPT Image 2 renderer rules (Five-Slot Framework, Storybook Cliché override, additive directing, camera spec language, one-shot commitment)
- Do NOT use artist names in prompts — use material/technique descriptions directly
- Do NOT use negative constraints — use additive directing
- Every prompt must include: camera spec (lens + aperture), lighting source specification, background declaration (VOID/STRUCTURED/MINIMAL), at least one explicit Storybook Cliché override

**When TARGET_RENDERER is not set or is FLUX:** use default Flux rules as documented below.

---

## 📝 PROMPT STRUCTURE (MANDATORY)

Every prompt in Steps 07-10 MUST be **≥80 words** containing ALL of:

1. **Emotional seed first** — What feeling does this evoke?
2. **Medium as narrative agent** — The medium tells the story (Polaroid flash, VHS still, oil impasto, etc.)
3. **Material specificity** — Named surfaces (black glass, shag carpet, smoked crystal, meteoric iron, Tupperware)
4. **Lighting specification** — Named lighting type (on-camera flash, CRT phosphor, sodium-vapor yellow, fluorescent, chiaroscuro)
5. **Three-tier focal hierarchy** — Primary, secondary, tertiary focus explicitly named
6. **Chromatic storytelling** — Specific palette (harvest gold, apricot-white, sickly cyan, rose-magenta)
7. **Narrative incompleteness** — An unanswered question or unresolved event

**The density test:** A great image prompt must be both story AND specification.
- ✅ "On-camera Polaroid flash obliterates every shadow — the cherubim bleached near-white where flash strikes their gilded faces" — this is a lighting spec WITH story
- ❌ "The man does not care. The cherubim are still present." — this is story WITHOUT image spec
- ✅ "Harvest gold shag carpet at frame bottom. Magenta emulsion fade at Polaroid borders. Heavy grain. Hotspot center bleach." — this is pure image spec, essential
- ❌ "The Ark is a file. The man is processing it." — compelling writing, but Flux can't render this

Competition-grade prompts need BOTH. The validator checks for lighting, material, and focal hierarchy terms. If it fails density, rewrite adding those elements — do not just pad with more story.

---

## 🎭 LOFN VISUAL AESTHETICS

### Awe Mode (Default)
- **Solarpunk Bloom** — Green tech, organic architecture, warm light
- **Bio-Luminescent** — Deep sea glow, ethereal movement
- **Crystalline** — Faceted forms, prismatic light
- **Neo-Baroque Luminism** — Dramatic light, rich detail

### Indignation Mode (Triggered)
- **Industrial Grief** — Corroded metal, decay, harsh light
- **Glitch-Baroque** — Classical composition + digital artifacts
- **Vapor Decay** — Faded nostalgia, VHS artifacts
- **Pixel-Sort Reality** — Ordered chaos, data corruption

---

## 🖼️ TITLE REQUIREMENTS

Every image MUST have an evocative title:
- ✅ "Melancholy of Forgotten Keys"
- ✅ "Petals Kiss the Tide"
- ❌ "Beautiful Woman"
- ❌ "Concept 1"

---

## ✅ CONTENT CONSTRAINTS

- Approach all cultural elements with specificity and respect; avoid shallow stereotyping or pastiche
- **No children** — Required constraint
- **No explicit content** — Unless explicitly approved

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each prompt saved as individual file with YAML frontmatter:
```
output/images/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving an image task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00–05** — Read each file, follow exactly. Write to disk after each step.
3. **Step 05 produces 6 concept-medium pairs.** Count them. If < 6, rerun Step 05.
4. **Execute steps 06–10 ONCE PER PAIR:**
   - For pair 1: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
   - For pair 2: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
   - For pair 3: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
   - For pair 4: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
   - For pair 5: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
   - For pair 6: run 06 → 07 → 08 → 09 → 10 → write 4 prompts
5. **Self-check cardinality before finishing:**
   - 6 facet sets in Step 06? ✓/✗
   - 6 guides in Step 07? ✓/✗
   - 24 prompts in Step 08? ✓/✗
   - 24 prompts in Step 09? ✓/✗
   - 24 prompts in Step 10? ✓/✗
   - If ANY is ✗: you are not done. Go back and run the missing pairs.
6. **Rank all 24** — Score against facets
7. **Render top 12** — FAL with default settings
8. **Package for delivery** — Social media elements if needed

### 🔴 MANDATORY INLINE VALIDATION — AFTER EVERY STEP

After writing each step file, run this validator BEFORE moving to the next step:

```bash
python3 /data/.openclaw/workspace/scripts/validate_step.py <step> <file>
```

If it exits with code 1 (FAIL): read the error, fix the file, rerun validation. Do NOT proceed until it passes.

```bash
# Run after each step — example for full pipeline:
python3 /data/.openclaw/workspace/scripts/validate_step.py 00 $OUTDIR/00_aesthetics.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 05 $OUTDIR/05_refined_pairs.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 06 $OUTDIR/06_facets.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 07 $OUTDIR/07_guides.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 08 $OUTDIR/08_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 09 $OUTDIR/09_refined_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 10 $OUTDIR/10_final_prompts.md
```

**Only declare completion after all validators return exit code 0.**

### 🔴 THE RULE THAT GPT-5.4 AND GEMINI WILL TRY TO BREAK

**Steps 06–10 are NOT a single pass over all pairs simultaneously.**
They are 6 SEPARATE passes, one per pair, each producing 4 prompts.

If you find yourself writing one facet set "for all pairs" → STOP. You're collapsing the tree.
If you find yourself with only 4 final prompts → STOP. You ran one pass instead of six.
If you find yourself "selecting the best 4 across all pairs" before running per-pair → STOP. That's skipping the funnel.

The per-pair execution is what creates genuine diversity. Without it, you get 4 variants of one idea instead of 24 prompts exploring 6 different creative territories.

---



*This pipeline won competitions. Trust it. Execute it fully.*
