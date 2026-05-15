# Compressed Image Pipeline — Agent-Executable
*Created: 2026-04-21*
*Purpose: A single-file pipeline that agents can read + execute within their context/time budget*

## INPUT (read these first)
1. `{output_dir}/orchestrator_metaprompt.md`
2. `{output_dir}/../golden_seed.md`

## EXECUTION (complete each step, save to disk, move on)

### Step 1: Aesthetics & Genres
Generate 50 aesthetics, 50 emotions, 50 compositions, 50 genres relevant to the metaprompt's creative direction. Use the constraint axes and medium collisions from the metaprompt as your primary seeds. Focus on RARE and UNEXPECTED combinations that serve the anti-Victorian territorial fae concept.

Save as: `{output_dir}/01_aesthetics_genres.json`

### Step 2: Essence & 6 Concepts
Extract the essence from the metaprompt. Generate exactly 6 concepts (not 12 — we go straight to 6 for efficiency). Each concept MUST specify:
- A specific fae sovereignty type from the metaprompt's constraint axes
- A specific medium collision (base medium + disruption medium)
- A specific crossing/threshold as compositional spine
- A specific emotional register
- A specific scale relationship
- Adult women fae only. No children. No cute.
- The Descent Rule: every image has a crossing, threshold, or descent

Each concept must be a vivid, sensory paragraph (3-5 sentences) with proper nouns and specific descriptors for:
- The fae figure (body type, age, ethnicity, how her form merges with landscape)
- The environment (specific location or terrain type)
- The threshold/crossing element
- The "concrete horizon" — one straight edge of different material visible at thumbnail

Save as: `{output_dir}/02_concepts.md`

### Step 3: Medium & Concept Refinement
For each of the 6 concepts, refine and specify:
- The exact medium collision and how it appears in the image
- The color palette (specific hex-like descriptors: "deep iron oxide and cold teal" not "warm and cool")
- The lighting conditions (source, quality, temperature)
- The compositional hierarchy (3 attention tiers)
- The "wrongness" — what makes this NOT look like typical fairy art

Save as: `{output_dir}/03_refined.md`

### Step 4: Write 6 Final Prompts
Convert each refined concept into a Flux Pro 1.1 Ultra prompt. Rules:

**CRITICAL — DESCRIPTION NOT INSTRUCTION:**
- Noun-first, present-tense. NEVER start with: Create/Design/Make/Render/Generate/Depict/Show/Draw
- Describe what IS in the image — subject, present-tense verb, scene details
- ≥80 words per prompt
- No artist names
- Every concept has a legible primary subject
- One straight edge, different material, visible at thumbnail (concrete horizon rule)

**Prompt structure (follow this order):**
1. First sentence: Subject + medium collision + the crossing
2. Expand: The fae figure and how she merges with landscape
3. Detail: The threshold/crossing element and the concrete horizon
4. Atmosphere: Color palette, lighting, emotional register
5. The wrongness: What makes this NOT typical fairy art
6. Final punch: One sentence that captures the territorial claim

Save as: `{output_dir}/04_final_prompts.md`

### Step 5: Generate Images
Read `skills/image-gen/SKILL.md` for FAL generation instructions.

For each of the 6 prompts, generate with:
```bash
node skills/image-gen/scripts/fal-generate.cjs \
  --prompt "<prompt text>" \
  --aspect "3:4" \
  --output {output_dir}/0N_<slug>.png \
  --safety 3
```

### Step 6: Deliver
Send each image through the configured delivery channel:
- channel: telegram
- target: <configured-recipient>
- buttons: []

Then send a summary message with all 6 titles and one-line descriptions.

---

## WHAT THIS REPLACES
This compressed pipeline replaces the 11-step pipeline (skills/image/steps/00-10) which totals 37,000 words of instructions — too large for any subagent to read + execute within its context/time budget. This version is ~600 words and contains the same creative logic in executable form.
