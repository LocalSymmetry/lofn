---
name: lofn-step11-packager
description: Build complete Step 11 refinement prompt bundles (the "fable packager") for a Lofn music run — full personality YAML + Suno v5.5 rules + golden songs + full run context + step-10 (and any prior step-11 draft) + the canonical Manual Refinement Instructions, as paste-ready 120-200KB prompt files for Codex-Fable/Opus refinement. Use when the user asks to "package step 11", build fable/opus refinement prompts, or assemble the full-context refinement bundle for a finished run's pairs. Do NOT use to run the inline pipeline (that's lofn-music) or to render audio.
---

# Lofn Step 11 Packager — Codex-backed fable/opus refinement-prompt builder

> **⚖️ AUTHORITY (2026-07-01):** the `.claude/skills/` twin of this skill is the CANONICAL policy source; this Codex mirror binds to it and to `.agents/skills/lofn/EXECUTION.md` §8 (Policy Deltas — golden-output quarantine, no-skip/NON-CANONICAL, itemized packet, per-pair variation angles, judge separation, the publish bar, gate mid-bands). On any disagreement, the `.claude` file wins.

Assembles the **full** context bundle for a Lofn music run into one paste-ready prompt per pair, so a high-capability model (Codex‑Fable 5 / Opus) can re-produce the Step 11 enhanced Suno package with complete creative authority. Each prompt embeds: the entire archive personality YAML (~50KB), the Suno v5.5 construction rules, the golden songs index, the full run context (seed → panel debate → metaprompt → pair assignments → creative context/ICB), the production mandates, the **Step 10** package (and any prior inline **Step 11** draft), and — as the final block — the canonical **Manual Refinement Instructions** (the executable task).

This is the Codex-native port of the legacy `skills/step11-packager/`. It is **self-contained**: the packager-specific references (Suno v5.5 rules, golden songs index, the default LOFN-PRIME YAML) are vendored under `references/` so the skill runs even outside the repo. The repo archive is only a fallback for **named** personalities not vendored here. This skill swaps the **execution layer**: vendored/repo-relative paths, the Codex-native run layout, and cross-platform (Windows, UTF-8) I/O. The legacy `skills/step11-packager/` is left untouched. See `.agents/skills/lofn/EXECUTION.md`.

## When to use vs. lofn-music
- **lofn-music** runs the pipeline 00→11 inline (it already produces `pair_NN_step10_final_package_enhanced.md`).
- **This skill** is the *manual-refinement packager*: it does not generate songs. It wraps a finished (or partway) run's per-pair Step 10 outputs into the heavy refinement prompt so the smart model can rebuild Step 11 from scratch with all context in one paste — for a deeper polish pass, a Fable/Opus rerun, or a head-to-head against the inline step-11 draft.

## Primary workflow — scripted (recommended)

`scripts/build_fable_prompt.py` auto-discovers everything from the run directory. On this machine the interpreter is `python` (Python 3.14).

```bash
# Build all pairs for a run (auto-detect repo root, pairs, personalities, context):
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19

# Opus-named variant:
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19 --mode opus

# Specific pairs only:
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19 --pairs 1,3,5

# Override personality (archive filename or absolute path) / custom output / explicit repo root:
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19 --personality lofn-prime-mini.yaml
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19 -o /tmp/prompts
python .agents/skills/lofn-step11-packager/scripts/build_fable_prompt.py output/daily/2026-06-19 --repo-root E:/git/lofn
```

**What it does:**
1. Auto-detects the repo root (from the script location, then the run dir) — no hardcoded workspace path.
2. Discovers per-pair Step 10 files: `<run>/music/pair_NN_step10_package.md` (Codex-native) or `<run>/audio/pair_XX_step10_production_wrap.md` (legacy) — backslash-safe.
3. Resolves each pair's personality from `05_pair_assignments.md` (LOFN-PRIME AWE/INDIGNATION mode by default; a named personality maps to its archive YAML; otherwise LOFN-PRIME).
4. Embeds the full personality YAML, the Suno rules, and the golden songs index — **vendored copies in this skill's `references/` are used first**; for a named personality not vendored, it falls back to the repo archive `skills/orchestration/personalities/` if present.
5. Embeds the full run context (research brief, golden seed, panel debate, metaprompt, pair assignments, creative context/ICB) and the production mandates.
6. Embeds the full Step 10 package and, if present, the prior inline Step 11 draft (`pair_NN_step10_final_package_enhanced.md`) as "improve on this" (suppress with `--no-enhanced`).
7. Appends the canonical Manual Refinement Instructions (three-block output spec, Disc_Channel + EMO format, <5000-char lyrics cap) as the FINAL block.
8. Writes `<run>/music/pair_NN_step11_{fable|opus}_prompt.md` per pair, in UTF-8.

## Then: execute the refinement (Codex-native)
The packager only builds the prompt — it does not call any model. To produce the enhanced package, hand each `pair_NN_step11_*_prompt.md` to the refinement engine:
- **Subagent (default):** spawn one Agent per pair (model `fable` for the Fable-5 ceremony tier, or inherit Opus), feeding it the prompt file as its task. It returns the three-block enhanced package.
- **Manual:** paste a file into a Codex‑Fable/Opus chat for a hand pass.

The model's output MUST be the three canonical blocks (`## SUNO STYLE PROMPT` 850–1000 chars · `## SUNO EXCLUDE PROMPT` 400–900 chars · `## SUNO ENHANCED LYRICS` with `[Theme:]`+`[SONG FORM:]`, 5-line Disc_Channel strip, em-dash EMO section headers, entire lyrics field **< 5000 chars**) plus every supporting block. Save as `pair_NN_step11_enhanced_suno.md`, then route through **lofn-qa**.

## Fallback workflow — manual build
If the script can't run, assemble by hand per the legacy spec in `skills/step11-packager/SKILL.md` §"Fallback Workflow" — same mandatory inclusions, but read context from the Codex-native filenames (`core_seed.md`, `03_panel_debate.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `CREATIVE_CONTEXT.md`) and Step 10 from `<run>/music/pair_NN_step10_package.md`.

## Mandatory inclusions (every prompt file)
1. Context header (run, run ID, pair, personality, mode)
2. **Full personality YAML — the ENTIRE file, zero bytes trimmed** (LOFN-PRIME = `lofn-prime-mini.yaml`, ~1800 lines)
3. Full Suno v5.5 construction rules (7 principles + 7-position order + limits + format)
4. Full run context — never rely on the reviewer opening repo files
5. Golden Song References index (two examples with payloads)
6. Production mandates — full text, not summaries
7. **Full Step 10 output** — all sections, not just music prompt + lyrics
8. Prior Step 11 draft if it exists (improve, don't echo)
9. **Manual Refinement Instructions as the FINAL block** — the executable task

## Anti-patterns — never do these
- ❌ Abbreviated personality block instead of the full YAML file
- ❌ Extracting only "music prompt + lyrics" from Step 10
- ❌ Truncating files with "[… truncated …]" markers
- ❌ Building prompts under 50KB (the script warns) — the YAML alone is ~50KB; a real bundle is **120–200KB**

## ⛔ FUSION IS PERMANENTLY BANNED
OpenRouter Fusion (`openrouter/fusion`) drained all OpenRouter credits. Never invoke, mention, offer, or build Fusion-mode prompts. There is no `--mode fusion`. The refinement engine is Codex (Fable/Opus) only.

## References
**Vendored (self-contained, used first):**
- `references/suno_rules_condensed.md` — Suno v5.5 rules (embedded by the script)
- `references/golden_songs_index.md` — golden song payloads
- `references/personalities/lofn-prime-mini.yaml` — the default LOFN-PRIME DNA (daily/fallback persona)

**Repo fallback (for named personalities not vendored, non-destructive):**
- `skills/orchestration/personalities/*.yaml` — full archive personality DNA. To make a named-personality run self-contained too, copy its YAML into `references/personalities/`.
- `skills/step11-packager/SKILL.md` — the legacy spec this port mirrors
