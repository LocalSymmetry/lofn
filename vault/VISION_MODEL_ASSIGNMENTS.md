# Vision (Image/Art) Model Assignments

Created: 2026-06-01 | Status: ACTIVE
Parent: `vault/LOFN_MODEL_ASSIGNMENTS.md` — inherits all music pipeline learnings

---

## Core Principle

**Vision is an LLM chain, not one giant worker.** The same architecture that made music runs reliable applies to image/art: split-step agents, per-step model assignments, dedicated step11 enhancement, QA gate, personality injection, ICB preservation.

Research → Core Seed → Orchestrator → Coordinator Steps 00-05 → Pair Step 06 → Pair Step 07 → Pair Step 08 → Pair Step 09 → Pair Step 10 → Step 11 Enhancement → QA

Do **not** assign one `lofn-vision` subagent to do Steps 06-10 in a single run. Use dedicated configured step agents. Per-spawn `model` overrides are unreliable — agents may silently fall back to their configured default.

---

## Vision Execution — Split-Step Chain

| Stage | Agent ID | Model | Role |
|---|---|---|---|
| Steps 00-05 | `lofn-vision-coordinator` | `deepseek/deepseek-v4-pro` | Coordinator: aesthetics, essence, concepts, mediums, pair-task packaging; preserves ICB |
| Step 06 | `lofn-vision-step06` | `openrouter/qwen/qwen3.7-max` | Facets: reliable structured writing, variation separation |
| Step 07 | `lofn-vision-step07` | `openrouter/qwen/qwen3.7-max` | Aspects/Traits: visual guides, emotional continuity, variation shaping |
| Step 08 | `lofn-vision-step08` | `openrouter/qwen/qwen3.7-max` | Generation: prompts with lighting, material, focal hierarchy, chromatic storytelling |
| Step 09 | `lofn-vision-step09` | `openrouter/qwen/qwen3.7-max` | Artist refinement: voice/style repair, compositor polish |
| Step 10 | `lofn-vision-step10` | `deepseek/deepseek-v4-pro` | Final synthesis: full Step 10 package, render prompts, provenance, self-check |
| Step 11 | `lofn-vision-step11` | `openai/gpt-5.5` | Enhancement pass: strong model polish, density validation, render-ready verification |

### Legacy agents

The legacy `lofn-vision` agent remains available (`deepseek/deepseek-v4-pro`) but **do not use it for Steps 06-10**. Use the dedicated step agents above.

---

## Controller / Orchestration

- Main session / controller: `deepseek/deepseek-v4-pro`
- `lofn-orchestrator`: `google/gemini-3.5-flash` ✅ proven
- Evaluation / ranking: `deepseek/deepseek-v4-pro`
- QA: `google/gemini-3.5-flash` ✅ proven

## Core / Seed

- Research synthesis: `deepseek/deepseek-v4-pro`
- Seed lineage: `deepseek/deepseek-v4-pro`
- Golden Seed: `deepseek/deepseek-v4-pro`

---

## Why These Models

### DeepSeek V4 Pro — Coordinator / Seed / Research / Step 10
Best at large sustained synthesis and preserving ICB. Use where long context must be compressed into durable artifacts. For Step 10, produces rich full synthesis packages with render prompts, visual notes, and provenance.

### Qwen3.7 Max — Steps 06, 07, 08, 09
The workhorse of the split chain. Reliable artifact writer. Battle-tested in music pipeline. Use for structural reasoning, variation separation, visual-guide continuity, prompt generation, and artist refinement.

### GPT-5.5 (direct OpenAI) — Step 11
Enhancement pass. Strong model polish, density verification, render-ready verification. Proven in music step11.

---

## Prompt Density Requirements (Image-Specific)

Every final image prompt (Steps 08-11) MUST contain ALL of:

1. **Emotional seed first** — feeling the image evokes
2. **Medium as narrative agent** — Polaroid flash, VHS still, oil impasto, etc.
3. **Material specificity** — named surfaces (black glass, shag carpet, smoked crystal, meteoric iron)
4. **Lighting specification** — named lighting type (on-camera flash, CRT phosphor, sodium-vapor yellow, chiaroscuro)
5. **Three-tier focal hierarchy** — Primary, secondary, tertiary focus explicitly named
6. **Chromatic storytelling** — specific palette (harvest gold, apricot-white, sickly cyan, rose-magenta)
7. **Narrative incompleteness** — an unanswered question or unresolved event

**Minimum 80 words per prompt.**

---

## Spawn Pattern for Vision Runs

After Step 05 completes:

1. Spawn up to 5 pair agents for **Step 06 only** using `lofn-vision-step06`.
2. Verify files on disk. Validate. Then spawn Step 07 agents using `lofn-vision-step07`.
3. Continue one step at a time through Step 10.
4. Step 11 enhancement uses `lofn-vision-step11`.
5. QA gate uses `lofn-qa`.

### File expectations per pair

- `pair_XX_step06_facets.md`
- `pair_XX_step07_aspects_traits.md`
- `pair_XX_step08_generation.md`
- `pair_XX_step09_artist_refined.md`
- `pair_XX_step10_revision_synthesis.md`
- `pair_XX_step11_enhanced.md` (Step 11 output)

---

## Dual-Mode Renderer Support

| Renderer | Rule File | Key Differences |
|----------|-----------|-----------------|
| **Flux Pro 1.1 Ultra** (default) | `skills/image/renderer_flux_rules.md` | Noun-first, present-tense, description not instruction |
| **GPT Image 2** | `skills/image/renderer_gpt_image2_rules.md` + `vault/GPT_IMAGE2_PLAYBOOK.md` | Five-Slot Framework, additive directing, no artist names |

When `TARGET_RENDERER = GPT_I2`: load renderer-specific rules before Step 05.

---

## Output Validation

| Stage | Validation |
|---|---|
| After Coordinator Step 05 | `validate_orchestrator_packet.py` on run dir |
| After Pair Step 10 | `validate_step.py 10 <file>` per pair |
| After Pair Step 11 | Density self-check (7 required elements + ≥80 words) |
| Before QA | Full pair artifacts present |
| QA Gate | Depth audit + Somatic Gate + density verification |

---

## Standing Warnings

- Legacy `lofn-vision` is NOT the split-step pair model. Use dedicated step agents.
- Spawn model overrides are unreliable — prefer dedicated configured step agents.
- If any subagent times out, inspect disk artifacts first; resume from last completed step.
- Every subagent must write artifacts as it completes each major step.
- ICB must survive every handoff.
- Disk is the only authority.
- MiMo V2.5 Pro and MiniMax M2.7 should not be used for artifact-writing steps (proven unreliable in music pipeline).
