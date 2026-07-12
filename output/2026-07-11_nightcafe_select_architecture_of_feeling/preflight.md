# Competition Pipeline Preflight

- Run ID: `2026-07-11_nightcafe_select_architecture_of_feeling`
- Modality: image, followed by competition sequence packaging.
- Target: 24 candidates from 6 pairs × 4 variations; one selected engine expanded into an 8-frame Body of Work.
- Renderer planning: Flux-style exploration; NightCafe Consistent Character and Nano Banana Pro continuity refinement after human render selection.
- Model availability: current Codex session responding; no paid render tool invoked.
- Output directory: exists and is writable.
- Concurrency: four total agent slots are available, so the controller may run at most three child agents concurrently. Pair plan is staggered 3 + 3.
- Research: `00_research_brief.md` is substantive and source-grounded.
- Rules: `00_competition_rules.md` is saved.
- Golden Seed: `core_seed.md` and `01_seed_lineage.md` exist; primary anchor is SEED 15.
- Barbell: 3 ACCESSIBLE + 3 AMBITIOUS. The machine-readable preflight route is `accessible` because the final entry must clear the first-glance surface; the ambitious half remains mandatory.
- Human subject: invented adult protagonist, no identifiable real person, no children.
- Golden-output quarantine: prior winning images remain judge-side; only structural moves are distilled.
- Competition timeout policy: 120-minute hard cap per uninterrupted production run; pair agents 20 minutes; QA 12 minutes. Persistent disk checkpoints make continuation safe.

## Canonical path resolution

The `.agents/skills/lofn/SKILL.md` canonical paths and all image Steps 00–10 resolve. The legacy `skills/orchestration/SKILL.md` mention of `skills/orchestration/TASK_TEMPLATE.md` is stale and has no tracked file; its required three-panel contract is resolved to the higher-authority `resources/panel-of-experts.md`, the orchestration skill itself, and `skills/orchestration/steps/06_metaprompt.md`. The image output contract resolves to `skills/image/TASK_TEMPLATE.md`.

