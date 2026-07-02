# Orchestrator Step 6: Write Metaprompt

## Input
- Core seed: `{output_dir}/core_seed.md`
- Dispatch brief: `{output_dir}/dispatch_brief.md`
- Personality: `{output_dir}/personality.md`
- Final synthesis: `{output_dir}/final_synthesis.md`

## Task

Write the orchestrator metaprompt. This is the creative brief that downstream agents (vision, audio, etc.) will read. It MUST contain:

### Structure:
1. **Personality voice** — name, energy, how they approach the brief
2. **Locked mood** — the panel-refined emotional anchor
3. **Key panel aha moments** — 3-5 bullets with attribution (which panel, which expert)
4. **Condensed world context** — 5-7 bullets (fact → creative implication)
5. **Constraint axes** — 4-5 axes with 4-6 options each
6. **What this is NOT** — boundaries the panels agreed on
7. **Source 2 vocabulary** — how the secondary research (Bandcamp, APOD, etc.) enters the work
8. **Source 3 form rule** — the material structure rule (from David Berman, etc.)
9. **Daily mandates** — specific rules for this run
10. **Legibility rule** — the subject must be VISIBLE as what it is at first glance

### Writing rules:
- Be specific, not generic. "Teal and rust" not "cool and warm"
- Name emotions precisely. "Territorial grief" not "sadness"
- Each constraint axis option should be a vivid phrase, not a single word
- The metaprompt should read like a creative director's brief, not a technical document

## Save
- Metaprompt: `{output_dir}/orchestrator_metaprompt.md`
