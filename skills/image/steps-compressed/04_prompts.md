# Image Pipeline Step 4: Write Final Prompts

Read:
1. `{metaprompt_path}`
2. `{output_dir}/03_refined.md`

Convert each refined concept into a Flux Pro 1.1 Ultra prompt. Rules:

**CRITICAL — DESCRIPTION NOT INSTRUCTION:**
- Noun-first, present-tense. NEVER: Create/Design/Make/Render/Generate/Depict/Show/Draw
- Describe what IS in the image — subject, present-tense verb, scene details
- ≥80 words per prompt
- No artist names
- Legible primary subject: the fae must read as FAE at thumbnail
- Concrete horizon: one straight edge, different material, visible at thumbnail

**Prompt structure:**
1. First sentence: Fae figure + medium collision + the crossing
2. Expand: The fae and how her wings/form merge with landscape
3. Detail: The threshold/crossing and the concrete horizon
4. Atmosphere: Color palette, lighting, emotional register
5. The wrongness: What makes this NOT typical fairy art
6. Final punch: One sentence capturing the territorial claim

Save to: `{output_dir}/04_final_prompts.md`
