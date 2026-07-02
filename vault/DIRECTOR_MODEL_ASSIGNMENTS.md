# Director (Video/Animation) Model Assignments

Created: 2026-06-01 | Status: ACTIVE
Parent: `vault/LOFN_MODEL_ASSIGNMENTS.md` — inherits all music pipeline learnings

---

## Core Principle

**Video/animation is an LLM chain, not one giant worker.** The same architecture that made music runs reliable applies to video: split-step agents, per-step model assignments, dedicated step11 enhancement, QA gate, personality injection, ICB preservation.

Research → Core Seed → Orchestrator → Coordinator Steps 00-05 → Pair Step 06 → Pair Step 07 → Pair Step 08 → Pair Step 09 → Pair Step 10 → Step 11 Enhancement → QA

Do **not** assign one `lofn-director` subagent to do Steps 06-10 in a single run. Use dedicated configured step agents.

---

## Director Execution — Split-Step Chain

| Stage | Agent ID | Model | Role |
|---|---|---|---|
| Steps 00-05 | `lofn-director-coordinator` | `deepseek/deepseek-v4-pro` | Coordinator: aesthetics, essence, 12 concepts, 6 medium pairs; preserves ICB |
| Step 06 | `lofn-director-step06` | `openrouter/qwen/qwen3.7-max` | Facets: reliable structured writing, variation separation |
| Step 07 | `lofn-director-step07` | `openrouter/qwen/qwen3.7-max` | Shot Guides: camera language, blocking, timing, audio direction |
| Step 08 | `lofn-director-step08` | `openrouter/qwen/qwen3.7-max` | Generation: full shot lists with [CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO] |
| Step 09 | `lofn-director-step09` | `openrouter/qwen/qwen3.7-max` | Artist refinement: director voice, shot flow repair, timing polish |
| Step 10 | `lofn-director-step10` | `deepseek/deepseek-v4-pro` | Final synthesis: full Step 10 package, Veo/Kling prompts, provenance, self-check |
| Step 11 | `lofn-director-step11` | `openai/gpt-5.5` | Enhancement pass: strong model polish, shot density verification, render-ready verification |

### Legacy agent

The legacy `lofn-director` agent remains available (`deepseek/deepseek-v4-pro`) but **do not use it for Steps 06-10**. Use the dedicated step agents above.

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
Best at large sustained synthesis and preserving ICB. Produces rich full synthesis packages with shot lists, Veo/Kling prompts, and provenance.

### Qwen3.7 Max — Steps 06, 07, 08, 09
The workhorse of the split chain. Reliable artifact writer. Battle-tested in music pipeline.

### GPT-5.5 (direct OpenAI) — Step 11
Enhancement pass. Strong model polish. Proven in music step11.

---

## Shot Prompt Requirements (Video-Specific)

Every final shot/treatment (Steps 08-11) MUST contain:

```
[CAMERA] — shot type + angle + movement (e.g., "Low-angle tracking shot")
[SUBJECT] — specific characteristics, not generic
[ACTION] — exactly what's happening, temporal progression
[SETTING] — environment, time, weather, lighting conditions
[STYLE & AUDIO] — aesthetic reference + sound design (dialogue, SFX, ambient, music)
```

### Camera Elements Required
- Shot type: ECU, CU, MS, WS, or ES
- Angle: eye-level, low, high, overhead, or Dutch
- Movement: static, slow pan, tracking, dolly, crane, orbit, aerial

### Audio Elements Required
- Dialogue (with quotation marks)
- SFX (with `SFX:` prefix)
- Ambient (soundscape description)
- Music direction where applicable

### Platform Constraints
| Platform | Aspect | Duration | Loop |
|----------|--------|----------|------|
| TikTok/Reels | 9:16 | 4-8s | Yes |
| YouTube Shorts | 9:16 | 8s | Can extend |
| Cinematic | 16:9 | Variable | Scene-based |

---

## Spawn Pattern for Director Runs

After Step 05 completes:

1. Spawn up to 5 pair agents for **Step 06 only** using `lofn-director-step06`.
2. Verify files on disk. Validate. Then spawn Step 07 agents using `lofn-director-step07`.
3. Continue one step at a time through Step 10.
4. Step 11 enhancement uses `lofn-director-step11`.
5. QA gate uses `lofn-qa`.

### File expectations per pair

- `pair_XX_step06_facets.md`
- `pair_XX_step07_shot_guides.md`
- `pair_XX_step08_generation.md`
- `pair_XX_step09_artist_refined.md`
- `pair_XX_step10_revision_synthesis.md`
- `pair_XX_step11_enhanced.md` (Step 11 output)

---

## Output Validation

| Stage | Validation |
|---|---|
| After Coordinator Step 05 | `validate_orchestrator_packet.py` on run dir |
| After Pair Step 10 | `validate_step.py 10 <file>` per pair |
| After Pair Step 11 | Shot density self-check (all 5 elements present) |
| Before QA | Full pair artifacts present |
| QA Gate | Depth audit + Somatic Gate + shot density verification |

---

## Standing Warnings

- Legacy `lofn-director` is NOT the split-step pair model.
- Spawn model overrides are unreliable — prefer dedicated configured step agents.
- Every subagent must write artifacts as it completes each major step.
- ICB must survive every handoff.
- Disk is the only authority.
- Veo 3.1 constraints: single clip 4s/6s/8s, 720p/1080p, 16:9 or 9:16.
