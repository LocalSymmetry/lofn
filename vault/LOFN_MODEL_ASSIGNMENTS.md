# Lofn Model Assignments

Updated: 2026-05-25 (23:16 EDT)
Status: Active — Gemini 3.5 Flash full pair pipeline, DeepSeek V4 Pro coordinator

## Controller / Orchestration

- Main session / controller: `deepseek/deepseek-v4-pro` (unchanged)
- `lofn-orchestrator`: `google/gemini-3.5-flash` ✅ proven
- Evaluation / ranking: `deepseek/deepseek-v4-pro` (unchanged)
- QA: `google/gemini-3.5-flash` ✅ proven

## Core / Seed

- Research synthesis: `deepseek/deepseek-v4-pro` (unchanged)
- Seed lineage: `deepseek/deepseek-v4-pro` (unchanged)
- Golden Seed: `deepseek/deepseek-v4-pro` (unchanged)

## Audio Execution

- Steps 00–05 (coordinator): `deepseek/deepseek-v4-pro` — critical junction, richest sustained output
- Steps 06–09 (pair agents — facets, song guides, generation, refinement): `google/gemini-3.5-flash` ← NEW
- Step 10 (pair agents — final synthesis package): `google/gemini-3.5-flash` ← UPDATED 2026-05-25 (EMO enforcement, consistent template)

## Pair Agent Breakdown

| Step Range | Model | Role |
|-----------|-------|------|
| 06-09 | `google/gemini-3.5-flash` | Facets, Song Guides, Generation, Refinement |
| 10 | `google/gemini-3.5-flash` | Final Synthesis — Suno-ready package with EMO enforcement |
| 11 | `openai/gpt-5.5` | Enhancement — strong model final polish, 15-point gate verification, producer-grade prompt |

## Rationale

- **Steps 06-09 (Gemini 3.5 Flash):** First lyric writing must commit to genre without retreating to acoustic safety. Gemini 3.5 Flash obeys genre constraints where gpt-5.5 defaults to lofi-acoustic when given vulnerable emotional concepts. Fast, genre-committed, no apology.
- **Step 10 (DeepSeek V4 Pro):** The final forge demands the richest sustained output — full ICB carrier, proper Suno formatting, EMO header compliance, Somatic Gate check, pair-specific provenance. DeepSeek V4 Pro is the model that won't summarize away creative DNA.
- **Coordinator (DeepSeek V4 Pro):** ICB must survive all 5 coordinator steps without degradation. DeepSeek V4 Pro produces 40-60KB step files with full panel fidelity.

## Output Validation (REINSTATED)

| Stage | Validation Script |
|-------|------------------|
| After Coordinator Step 05 | `validate_portfolio_distinctiveness.py` (on step05) |
| After Pair Step 06 | `validate_step06_distinctiveness.py <audio_dir>` |
| After Pair Step 09 | `validate_step09_distinctiveness.py <audio_dir>` |
| After Pair Step 10 | `validate_step.py 10 <file>` per pair |
| After Pair Step 11 | 15-point QA self-check in enhanced package |
| Before QA | `validate_pair_artifacts.py` for all pairs |
| QA Gate | Somatic Gate (3 Hyper-Skeptics) + full 15-point check |

## Notes

- **Gemini 3.1 Pro retired entirely** — marginal coordinator output, thin pair output, no advantage over 3.5 Flash for speed or DeepSeek V4 Pro for depth
- **Gemini 3.5 Flash:** Fast (1m-2m per pair), genre-obedient, no hallucination issues in production
- **DeepSeek V4 Pro:** Gold standard for rich sustained output — coordinator and final forge only
- **ICB (Immutable Continuity Block):** Mandatory at every handoff per `vault/PIPELINE_CONTINUITY_STANDARD.md`
- **Somatic Gate:** 3 Hyper-Skeptics vote as bloc on every step10 package. 2 of 3 NO = BLOCKED. Gate question: "Is this sonically distinctive enough to be Lofn, or could any competent prompt generate this?"
