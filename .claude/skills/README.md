# Lofn — Claude-native skills

Claude Code ports of the Lofn creative pipeline. These run the same award-winning, 3-phase Lofn process — **but with Claude as the engine for every step**, no OpenClaw required. They are **invokable in this repo** (cwd = repo root) as `/lofn`, `/lofn-music`, etc.

## The set

| Skill | Role |
|-------|------|
| **`/lofn`** | Front door. Phase 0 (Golden Seed) + Phase 1 (3-panel orchestrator, metaprompt, pair assignments, ICB), then dispatches to a modality + QA. **Start here.** |
| **`/lofn-music`** | Music pipeline 00–11 → Suno two-field packages, EMO lyrics. |
| **`/lofn-image`** | Image pipeline 00–10 → Flux/GPT-Image render-ready prompts. |
| **`/lofn-video`** | Video **and animation** pipeline 00–10 → Veo 3.1 shot lists / loops. |
| **`/lofn-story`** | Story pipeline 00–10 → panel-driven prose. |
| **`/lofn-qa`** | Strict adversarial gate → SHIP / REPAIR / FAIL. |
| **`/lofn-daily`** | The daily run — fetch real-world facts, then generate the day's music + images through the pipeline (tri-source, dual 3+3, emotional duality). Down-scalable for a quick test. |
| `lofn/EXECUTION.md` | Shared Claude-native execution protocol (subagent spawning + self-check gates that replace the Python validators). |

## Quick start

```
/lofn  make a solarpunk song about healing after collapse, full pipeline
```
`/lofn` runs the seed + 3-panel debate inline, then fans the 6 pairs out to parallel Claude subagents, then QA. You can also call a modality directly if a Phase-0/1 packet already exists in the run dir.

## How these relate to `skills/` (the OpenClaw originals)

- **Non-destructive.** The originals under `skills/**` are untouched and still work for OpenClaw deployment.
- **Content is reused, not duplicated.** These skills point at the original step files, the 114-personality / 178-panel libraries, the Golden Seeds, and the QA references — the "3 years of tuning."
- **Only the execution layer is replaced:** OpenClaw `sessions_spawn`/`openclaw agent run` → Claude **Agent tool** subagents; DeepSeek/GPT-5.5/Gemini model-tiering → **Claude**; `python3 validate_*.py` → **Claude-native self-check gates** (`lofn/EXECUTION.md` §4); `/data/.openclaw/workspace/` paths → **repo-relative**.
- **Corrected paths.** Several legacy SKILLs cite `skills/lofn-core/GOLDEN_SEEDS.md` / `PIPELINE.md`; those actually live in `skills/lofn-core/refs/`. The Claude skills use the real locations.

Output lands under `output/<run>/` (working artifacts) and `output/<type>s/` (final per-artifact files), per `skills/lofn-core/OUTPUT.md`.
