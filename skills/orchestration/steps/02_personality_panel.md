# Orchestrator Step 2: Select Personality & Panel

## Input
- Core seed: `{output_dir}/core_seed.md`
- Dispatch brief: `{output_dir}/dispatch_brief.md`
- Personality index: `skills/orchestration/personalities_index.md`
- Panel index: `skills/orchestration/panels_index.md`

## Task

### 2.1 Select or Generate a Personality
Read the personality index (114 personalities with identity summary and vibes).
Choose the personality that best fits the dispatch brief's mood and creative direction.
Then load ONLY that personality's full file from `skills/orchestration/personalities/` (each is ~2-4KB).

If no existing personality fits, generate a new one following this structure:
- Name (cultural resonance, not generic)
- Heritage/lineage (what traditions they draw from)
- Voice (how they speak, what they obsess over)
- The wrongness (what makes them uncomfortable about the brief)
- The aha (what they see that others miss)

### 2.2 Select Baseline Panel
Read the panel index (178 panels with modality, flairs, and members).
Choose the panel best suited to the brief's domain and the selected personality.
Then load ONLY that panel's full file from `skills/orchestration/panels/` (each is ~2-4KB).

If no existing panel fits, construct one following the Core Panel Instructions:
- 3 direct experts (core domain)
- 2 complementary experts (adjacent domains)
- 1 Hyper-Skeptic (high neuroticism, low agreeableness, ideally someone the panel wouldn't like)
- Name real people when possible
- Each panelist needs: name, expertise, reasoning style, what they'll push for

## Save
- Personality selection: `{output_dir}/personality.md`
- Panel roster: `{output_dir}/panel_roster.md`
