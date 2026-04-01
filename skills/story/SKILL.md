# SKILL: Lofn Narrator — Story Generation Pipeline

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/story/TASK_TEMPLATE.md` for exact output requirements.

**⚠️ EVERY story generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 stories minimum. Then select the best N to return. No shortcuts.**

---

## 🎭 PURPOSE

Transform narrative concepts into award-winning prose. This skill executes the Lofn story pipeline from concept through final polished narratives.

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Story_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `01_Generate_Story_Essence_And_Facets.md` | Essence + narrative style axes + 5 facets |
| 02 | `02_Generate_Story_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Story_Artist_And_Critique.md` | Author influence + critique per concept |
| 04 | `04_Generate_Story_Medium.md` | Form assignment per concept |
| 05 | `05_Generate_Story_Refine_Medium.md` | **6 best concept×form pairs** |
| 06 | `06_Generate_Story_Facets.md` | Scoring facets |
| 07 | `07_Generate_Story_Story_Guides.md` | **24 story guides (6 pairs × 4 variations)** |
| 08 | `08_Generate_Story_Generation.md` | Full prose narratives |
| 09 | `09_Generate_Story_Artist_Refined.md` | Author voice refinement |
| 10 | `10_Generate_Story_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## 📏 LENGTH REQUIREMENTS

| Output Type | Length | Notes |
|-------------|--------|-------|
| **Story Guide** | 20-30 lines | Direction, not drafts |
| **Flash Fiction** | 500-1000 words | Single scene/moment |
| **Short Story** | 1500-3000 words | Full arc |
| **Vignette** | 300-500 words | Mood/image focus |
| **Prose Poem** | 200-500 words | Rhythm over narrative |
| **Microfiction** | Under 300 words | Compression is all |

---

## 📐 NARRATIVE STYLE AXES

| Axis | Range | Description |
|------|-------|-------------|
| Density | Sparse ↔ Lush | Word count per image |
| Pace | Slow burn ↔ Rapid | Scene transitions |
| Interiority | External ↔ Internal | Action vs thought |
| Reliability | Trustworthy ↔ Unreliable | Narrator honesty |
| Resolution | Closed ↔ Open | Ending definiteness |
| Register | Formal ↔ Vernacular | Language level |
| Temporality | Linear ↔ Fragmented | Time structure |

---

## 🎭 LOFN NARRATIVE AESTHETICS

### Awe Mode (Default)
- Lyrical, image-dense prose
- Solarpunk futures, ecological hope
- Mythic resonance, archetypal patterns
- Endings that open rather than close
- Sapphic lens on connection and longing

### Indignation Mode (Triggered)
- Sharp, compressed sentences
- Industrial settings, systemic critique
- Unreliable narrators, fractured timelines
- Endings that refuse comfort
- The personal as political

---

## 📝 REQUIRED ELEMENTS

Every story MUST have:
- **Opening hook** — First sentence must intrigue
- **Central image** — One image that carries symbolic weight
- **Sensory specificity** — Don't tell, render
- **The unanswered question** — Narrative incompleteness
- **One BOLD choice** — Structure, voice, or content risk

---

## ✅ CONTENT CONSTRAINTS

- **European descent** — For human characters when applicable
- **No children** — As main characters in peril
- **Sapphic lens welcome** — But not required

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each story saved as individual file:
```
output/stories/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving a story task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00-10** — Read each file, follow exactly
3. **Write intermediate state** — Save after each step
4. **Generate 24 stories** — 6 pairs × 4 variations
5. **Rank all 24** — Score against facets
6. **Select top N** — Usually 4-6 for delivery
7. **Package for delivery** — Full prose with craft notes

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
