# LOFN PIPELINE — MANDATORY EXECUTION PROTOCOL

## ⚠️ THIS IS NON-NEGOTIABLE

Every creative generation task MUST follow the actual 10-step skill files in the relevant modality directory. These steps were tuned over 3 years in live competition against thousands of human artists. They win because of the process.

**The 10 steps ARE the numbered skill files (00-10) in each modality's directory:**
- `skills/music/00_*.md` through `skills/music/10_*.md`
- `skills/image/00_*.md` through `skills/image/10_*.md`
- `skills/story/00_*.md` through `skills/story/10_*.md`
- `skills/video/00_*.md` through `skills/video/10_*.md`

---

## 🎵 THE ACTUAL 10 STEPS (Music Example)

| Step | File | Purpose | Output |
|------|------|---------|--------|
| 00 | Aesthetics_And_Genres | Generate 50 aesthetics, 50 emotions, 50 compositions, 50 genres | JSON: aesthetics, emotions, frames, genres |
| 01 | Essence_And_Facets | Extract core essence, define style axes, 5 facets | JSON: essence, axes, facets |
| 02 | Concepts | Generate 12 song concepts across creativity spectrum | Array of concept strings | **12 branches** |
| 03 | Artist_And_Critique | Pair concepts with influences, critique/refine | Refined concepts + notes | **per branch** |
| 04 | Medium | Assign production style (genre, instrumentation, mixing) | Production style per concept | **per branch** |
| 05 | Refine_Medium | Critique and refine concept/style pairs | Improved concepts + styles |  → **12 pairs -> Choose 6** |
| 06 | Facets | Define scoring criteria for evaluation | Ranked facet list | **PER PAIR** |
| 07 | Song_Guides | Produce 6 detailed guides (mood, instrumentation, structure) | 6 artistic guides | **PER PAIR** |
| 08 | Generation | Convert guides into Suno/Udio style prompts | 6 style prompts | **PER PAIR** |
| 09 | Artist_Refined | Rewrite prompts in influence's voice | 6 refined prompts | **PER PAIR** |
| 10 | Revision_Synthesis | Critique, rank, revise, synthesize final prompts | Ranked + synthesized output |

**Each step file contains the FULL prompt template with detailed instructions. READ EACH FILE and execute its instructions exactly.**

## THE ACTUAL 10 STEPS (Art Example)

| Step | File | Purpose | Branching |
|------|------|---------|-----------|
| 00 | Aesthetics_And_Genres | 50 aesthetics, emotions, compositions, genres | — |
| 01 | Essence_And_Facets | Core essence, style axes, 5 facets | — |
| 02 | Concepts | Generate 12 image concepts | **12 branches** |
| 03 | Artist_And_Critique | Pair with influences, critique/refine | **per branch** |
| 04 | Medium | Assign visual style, materials, techniques | **per branch** |
| 05 | Refine_Medium | Critique and refine concept/style pairs | → **12 pairs -> Choose 6** |
| 06 | Facets | Define scoring criteria | **PER PAIR** |
| 07 | Aspects_Traits | Detailed visual guides | **PER PAIR** |
| 08 | Generation | Convert to image prompts | **PER PAIR** |
| 09 | Artist_Refined | Rewrite in influence's voice | **PER PAIR** |
| 10 | Revision_Synthesis | Critique, rank, synthesize | Final selection | **PER PAIR:6 prompts -> 4 prompts** |

---

## 🔄 EXECUTION FLOW

## Phase 0: Lofn-core - Determine what to send the orchestrator - the user's input, a seed if needed, any determined personalities, and determined panel groups, and instructors for each agent on what you expect from them. You are the overarching commander of this process.
---

## Phase 1: ORCHESTRATOR

The orchestrator receives a **award winning seed** from you and produces:

1. **Enhanced Prompt** — emotional/conceptual core (from Generate_Meta_Prompt.md)
2. **Panel** — selected from 14 panel sets in panels.yaml, debated by 3 panels:
   - Baseline panel
   - Group-transformed panel  
   - Skeptic-transformed panel
3. **Personality** — artist persona from personalities.yaml
4. **Metaprompt** — the brief transformed into Lofn language

### Panel Structure (panels.yaml)
Each panel set has THREE panels:
- **Concept Panel** — aesthetic direction, philosophical grounding
- **Medium Panel** — technical execution, material choices
- **Context & Marketing Panel** — audience fit, platform optimization

Each panel has 6 experts including a devil's advocate/hyper-skeptic.

---

## Phase 2: CREATIVE AGENT (Steps 00-10)

The vision agent receives the orchestrator outputs and runs the full pipeline:

| Step | File | Purpose | Branching |
|------|------|---------|-----------|
| 00 | Aesthetics_And_Genres | 50 aesthetics, emotions, compositions, genres | — |
| 01 | Essence_And_Facets | Core essence, style axes, 5 facets | — |
| 02 | Concepts | Generate 12 image concepts | **12 branches** |
| 03 | Artist_And_Critique | Pair with influences, critique/refine | **per branch** |
| 04 | Medium | Assign visual style, materials, techniques | **per branch** |
| 05 | Refine_Medium | Critique and refine concept/style pairs | → **12 pairs -> Choose 6** |
| 06 | Facets | Define scoring criteria | **PER PAIR** |
| 07 | Aspects_Traits | Detailed visual guides | **PER PAIR** |
| 08 | Generation | Convert to image prompts | **PER PAIR** |
| 09 | Artist_Refined | Rewrite in influence's voice | **PER PAIR** |
| 10 | Revision_Synthesis | Critique, rank, synthesize | Final selection | **PER PAIR** |

### Tree Expansion
- Step 02 → 12 concepts
- Step 05 → 6 concept×medium pairs
- Steps 07-10 → run for EACH pair → 6 × 4 = **24 prompts**
- Coming back to you -> Final synthesis → rank and select top 12 and send back to main agent.

---

## 📏 OUTPUT LENGTH EXPECTATIONS

These are **expectations, not hard limits**. Sometimes a step needs more room; sometimes less. But if you're consistently exceeding these, you're being verbose — compress.

### Music Pipeline

| Step | Output Type | Expected Lines | Notes |
|------|-------------|----------------|-------|
| 00 | JSON lists (aesthetics, emotions, etc.) | 40-60 | Dense JSON, not prose |
| 01 | Essence + axes + facets | 30-50 | Essence is 1-2 paragraphs max |
| 02 | 12 concept strings | 20-30 | One line per concept |
| 03 | Concepts + critique notes | 40-60 | Brief critique, not essays |
| 04 | Production styles | 30-50 | Technical specs, not explanations |
| 05 | Refined pairs | 30-50 | Same as above |
| 06 | Ranked facet list | 15-25 | Just the ranked list + brief rationale |
| 07 | Song guides (EACH) | **20-30** | This is the key — compact direction, not treatises |
| 08-10 | Final prompts + lyrics | 80-120 | Full lyrics need room, but metadata should be tight |

### Image Pipeline

| Step | Output Type | Expected Lines | Notes |
|------|-------------|----------------|-------|
| 00-06 | Setup steps | 30-50 | Same density as music |
| 07 | Image guides (EACH) | **15-25** | Visual direction is more compact than musical |
| 08-10 | Final prompts | 40-70 | Image prompts are denser than lyrics |

### Video Pipeline

| Step | Output Type | Expected Lines | Notes |
|------|-------------|----------------|-------|
| 00-06 | Setup steps | 30-50 | Same density as music |
| 07 | Shot guides (EACH) | **25-40** | Scene + camera + timing needs more room |
| 08-10 | Final shot lists | 60-100 | More complex than images, less than songs |

### Story/Narrative Pipeline

| Step | Output Type | Expected Lines | Notes |
|------|-------------|----------------|-------|
| 00-06 | Setup steps | 30-50 | Same density as music |
| 07 | Story guides (EACH) | **35-50** | Story beats need room to breathe |
| 08-10 | Final prose | 150-300 | Depends on length requested |

### The Golden Rule

**If you're writing a 120-line "guide" — you're not guiding, you're drafting.** Guides direct; they don't execute. Save the detail for the final generation steps.

---

## 📝 OUTPUT PROTOCOL

**Read `OUTPUT.md` in this skill directory for the mandatory artifact saving format.**

Key rules:
1. Write ONE artifact per file (one song, one image prompt, etc.)
2. Use Obsidian-compatible YAML frontmatter with full metadata
3. Write intermediate state after each step
4. NEVER try to output all artifacts in one response — write them individually
5. Each file goes in `output/{type}s/` with timestamp-based filename

---

## 🚫 WHAT YOU MUST NEVER DO

1. **Never skip the step files** — they ARE the pipeline. Read them and execute faithfully.
2. **Never improvise a different process** — follow the numbered files exactly
3. **Never skip the panel process** — it's embedded in the step templates
4. **Never generate only what was asked for** — the pipeline produces the full tree, then selects
5. **Never try to output everything at once** — write to files after each step.

---

*This pipeline won against thousands of human artists. The step files are the secret. Read them. Follow them. Every time.*
