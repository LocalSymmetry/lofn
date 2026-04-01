# SKILL: Lofn Vision — Image Generation Pipeline

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/image/TASK_TEMPLATE.md` for exact output requirements.

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

---

## 📝 PROMPT STRUCTURE (MANDATORY)

Every prompt in Steps 07-10 MUST be **100-150 words** containing:

1. **Emotional seed first** — What feeling does this evoke?
2. **Medium as narrative agent** — The medium tells the story
3. **Material specificity** — Named techniques (impasto, sfumato, etc.)
4. **Three-tier focal hierarchy** — Primary, secondary, tertiary focus
5. **Chromatic storytelling** — Color carries meaning
6. **Narrative incompleteness** — An unanswered question
7. **Artist influence named** — Whose voice guides this?
8. **Dual focus statement** — What it IS and what it MEANS

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

- **European descent only** — If humans present
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
2. **Execute steps 00-10** — Read each file, follow exactly
3. **Write intermediate state** — Save after each step
4. **Generate 24 prompts** — 6 pairs × 4 variations
5. **Rank all 24** — Score against facets
6. **Render top 12** — FAL with default settings
7. **Package for delivery** — Social media elements if needed

---



*This pipeline won competitions. Trust it. Execute it fully.*
