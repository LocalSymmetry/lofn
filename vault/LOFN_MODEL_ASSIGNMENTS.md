# Lofn Model Assignments

Updated: 2026-05-25 (15:58 EDT)
Status: Active — Hybrid pipeline (DeepSeek coordinator, Gemini orchestrator + pair agents)

## Controller / Orchestration

- Main session / controller: `deepseek/deepseek-v4-pro` (unchanged)
- `lofn-orchestrator`: `google/gemini-3.5-flash` ← Active
- Evaluation / ranking: `deepseek/deepseek-v4-pro` (unchanged)
- QA: `google/gemini-3.5-flash` ← Active

## Core / Seed

- Research synthesis: `deepseek/deepseek-v4-pro` (unchanged)
- Seed lineage: `deepseek/deepseek-v4-pro` (unchanged)
- Golden Seed: `deepseek/deepseek-v4-pro` (unchanged)

## Audio Execution

- Steps 00–05 (coordinator): `deepseek/deepseek-v4-pro` ← PIVOTED (Gemini 3.1 Pro marginal in test)
- Steps 06–08 (pair agents): `google/gemini-3.1-pro-preview` ← Active (Gemini test)
- Steps 09–10 (pair agents): `google/gemini-3.5-flash` ← Active (Gemini test)

## Pair Agent Breakdown

| Step Range | Model | Role |
|-----------|-------|------|
| 06-08 | `google/gemini-3.1-pro-preview` | Facets, Song Guides, Generation |
| 09-10 | `google/gemini-3.5-flash` | Refinement, Revision Synthesis |

## Output Validation (REINSTATED)

Per Scientist directive (2026-05-25): Output validation scripts must run at every stage.

| Stage | Validation Script |
|-------|------------------|
| After Coordinator Step 05 | `validate_portfolio_distinctiveness.py` (on step05) |
| After Pair Step 06 | `validate_step06_distinctiveness.py <audio_dir>` |
| After Pair Step 09 | `validate_step09_distinctiveness.py <audio_dir>` |
| After Pair Step 10 | `validate_step.py 10 <file>` per pair |
| Before QA | `validate_pair_artifacts.py` for all pairs |
| QA Gate | `validate_portfolio_distinctiveness.py` final check |

Failure at any validation gate is a repair blocker, not advisory.

## Notes

Hybrid pipeline (2026-05-25 15:58 EDT):
- Orchestrator: Gemini 3.5 Flash ✅ (proven in test: 1m45s, clean output)
- Coordinator: DeepSeek V4 Pro ← PIVOTED (Gemini 3.1 Pro produced marginal ~1KB step files, DeepSeek more reliable for this critical junction)
- Pair agents (06-08): Gemini 3.1 Pro — test active
- Pair agents (09-10): Gemini 3.5 Flash — test active
- QA: Gemini 3.5 Flash
- Core/seed/research: DeepSeek V4 Pro
- Output validation REINSTATED at all stages
- Immutable Continuity Block (ICB) enforced per `vault/PIPELINE_CONTINUITY_STANDARD.md`
