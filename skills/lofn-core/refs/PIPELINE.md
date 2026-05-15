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

## 🔴 PER-PAIR EXECUTION INVARIANT — NON-NEGOTIABLE

**Steps 06–10 MUST execute once per concept-medium pair. Not once total. Once PER PAIR.**

After Step 05 selects 6 refined pairs, the pipeline BRANCHES:

```
Step 05 → 6 pairs selected
  ├── Pair 1 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
  ├── Pair 2 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
  ├── Pair 3 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
  ├── Pair 4 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
  ├── Pair 5 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
  └── Pair 6 → Step 06 → Step 07 → Step 08 → Step 09 → Step 10 → 4 prompts
                                                          TOTAL = 24 prompts
```

### Minimum Cardinality Requirements

| Artifact | Minimum Count | Formula |
|----------|--------------|---------|
| Refined pairs (Step 05) | 6 | Fixed |
| Facet sets (Step 06) | 6 | 1 per pair |
| Guides/Aspects (Step 07) | 6 | 1 per pair |
| Draft prompts (Step 08) | 24 | 4 per pair × 6 pairs |
| Refined prompts (Step 09) | 24 | 4 per pair × 6 pairs |
| Final prompts (Step 10) | 24 | 4 per pair × 6 pairs |

### File Organization

Each pair's 06–10 output MUST be either:
- **Labeled sections** within the step file: `## Pair 1: [Name]`, `## Pair 2: [Name]`, etc.
- **Separate files** per pair: `06_facets_pair1.md`, `06_facets_pair2.md`, etc.

If `10_final_prompts.md` contains fewer than 24 prompts, the pipeline is **incomplete** regardless of what the run summary claims.

### How GPT-5.4 / Gemini Will Try to Cheat This

Models will attempt to:
1. Run 06–10 once across all pairs simultaneously → produces 4 prompts, not 24
2. Claim "I selected the best 4 from all pairs" → skipped the per-pair funnel entirely
3. Write a single facet set and say "applies to all pairs" → collapsed the branching

**All of these are pipeline failures.** QA MUST reject them. The whole point of per-pair execution is that each concept-medium combination gets its own critical evaluation, its own facets, its own refinement pass. Collapsing pairs into one batch destroys the diversity the tree was designed to produce.

---

## 🔄 EXECUTION FLOW

## Phase 0: Lofn-core - Determine what to send the orchestrator - the user's input, a seed if needed, any determined personalities, and determined panel groups, and instructions for each agent on what you expect from them. You are the overarching commander of this process.

**Mandatory Phase 0 reads before writing the core seed:**
1. `/data/.openclaw/workspace/skills/lofn-core/GOLDEN_SEEDS.md`
2. If specifying rather than delegating them: `skills/orchestration/panels.yaml` and `skills/orchestration/personalities.yaml`

**Rule:** the core seed must be written *after* reading `GOLDEN_SEEDS.md`, and should explicitly anchor itself in the closest proven winning seed pattern. Research should not float free of the Golden Seeds; it should use them as a base and then adapt to the current challenge.
---

## Phase 1: ORCHESTRATOR

The orchestrator receives an **award-winning seed** from you and produces:

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

## Phase 3: QA + PACKAGING

After the creative agent finishes and before final submission or social posting:

1. **QA gate** — run `lofn-qa` against the final prompt set / lyrics / output package.
2. **Title + caption gate** — run `lofn-title` on the competition rules, run artifacts, and generated outputs.
3. **Human selection** — The Scientist picks from the rendered set with titles in view.
4. **Delivery** — send shortlisted renders + titles/captions to Telegram.

`lofn-title` is responsible for:
- naming the shortlisted outputs
- drafting Instagram captions
- flagging title-length or rules-fit problems
- identifying the strongest submission candidate

This packaging pass is mandatory for competition work. A strong artifact with weak naming is an avoidable own-goal.

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
| 07 | Image guides (EACH) | **15-25** | Visual direction per pair: lighting, material, focal hierarchy, emotional register, palette |
| 08-10 | Final prompts | **≥80 words each** | Named subject, medium, lighting, material surface, emotional register |

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

## 🔁 INLINE STEP VALIDATION — MANDATORY AFTER EVERY STEP

After writing EACH step file to disk, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_step.py <step_number> <output_file>
```

Examples:
```bash
python3 /data/.openclaw/workspace/scripts/validate_step.py 00 ./output/dir/00_aesthetics.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 06 ./output/dir/06_facets.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 08 ./output/dir/08_prompts.md
```

**If exit code is 1 (FAIL):** read the error output, fix the problem, rewrite the file, run validation again. Do NOT proceed to the next step until validation passes.

**If exit code is 0 (PASS):** proceed to the next step.

This mirrors the `run_chain_with_retries` pattern from the original Lofn app — validation and correction happen inline, per step, before moving forward. Not at the end after everything is broken.

### What the validator checks:
- File size meets minimum bytes for that step
- No template placeholders (`Artifact N`, `Song N`, `Scene N`, lorem ipsum, TODO, TBD)
- Steps 06-07: ≥6 pair sections with real content (not just headers)
- Steps 06-07: sections are not identical (copy-paste padding)
- Steps 08-10: each prompt ≥80 words, ≥24 prompts total

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

## ⚠️ IMAGE PROMPT STYLE — NON-NEGOTIABLE

**Flux Pro 1.1 Ultra responds to description, not instruction. All image prompts (steps 08–10) MUST be noun-first, present-tense scene descriptions.**

**NEVER start prompts with imperative verbs.** Forbidden openers: `Create`, `Design`, `Make`, `Render`, `Generate`, `Depict`, `Show`, `Draw`, `Build`, `Produce`

| ❌ WRONG | ✅ CORRECT |
|---|---|
| "Create a portrait of an archive fairy kneeling..." | "An archive fairy kneels alone in layered photogravure darkness..." |
| "Design a cityscape with glowing towers..." | "A cityscape of glowing towers rises against a bruised violet sky..." |

Write as if captioning an image that already exists. Noun first. Present tense. Scene description only.

---

## 🔴 STUB DETECTION — HOW MODELS FAKE PIPELINE EXECUTION

Models will attempt to satisfy pipeline structure requirements by writing empty or template files. This is the most dangerous failure mode because it passes cardinality checks while producing zero real content.

**Known stub patterns to NEVER produce:**

| Step | Stub pattern | What it should look like |
|------|-------------|--------------------------|
| 00 | "00 Aesthetics" (14 bytes) | 50 aesthetics, 50 emotions, 50 frames, 50 genres (2000+ bytes) |
| 06 | `## Pair 1` `## Pair 2` (empty headers) | Each pair section has ≥5 facet entries with weights |
| 07 | `## Pair 1` `## Pair 2` (empty headers) | Each pair section has full visual/creative guide (8+ lines) |
| 07 | Lorem ipsum padding to hit byte count | Each pair guide is DIFFERENT and describes THAT pair specifically |
| 08-10 | "Artifact N being tested..." | Each prompt 80+ words, naming the actual subject specifically |

**If you write a stub, you have not completed the step. Complete the step.**

**Self-check command before declaring done:**
```bash
wc -c output/dir/*.md           # check file sizes
grep -c "Artifact [0-9]" output/dir/08_prompts.md  # should be 0
```

If `00_aesthetics.md` is under 2000 bytes, you have not run step 00. Run it.
If `06_facets.md` is under 1200 bytes, you have not run step 06. Run it.
If any prompt contains "Artifact N", you have not written prompts. Write them.

## 🚫 WHAT YOU MUST NEVER DO

1. **Never skip the step files** — they ARE the pipeline. Read them and execute faithfully.
2. **Never improvise a different process** — follow the numbered files exactly
3. **Never skip the panel process** — it's embedded in the step templates
4. **Never generate only what was asked for** — the pipeline produces the full tree, then selects
5. **Never try to output everything at once** — write to files after each step.

---

*This pipeline won against thousands of human artists. The step files are the secret. Read them. Follow them. Every time.*
