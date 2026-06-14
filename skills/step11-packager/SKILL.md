---
name: step11-packager
description: "Build complete manual Step 11 refinement prompts for Claude-Fable/Opus or OpenRouter Fusion per-pair use from step10+step11 output with full personality YAML, Suno construction rules, and production mandates."
---

# Step 11 Packager — Manual Opus/Fable/Fusion Refinement Prompt Builder

Takes Step 10 + Step 11 source artifacts, wraps them with full archive personality YAML + Suno Prompt Construction Guide rules + production mandates, and produces paste-ready prompt files for manual Claude-Fable/Opus refinement or explicitly approved OpenRouter Fusion use.

**Cost rule:** Do not invoke Fusion from this skill by default. Fusion is packaged here as text. If The Scientist explicitly approves direct Fusion spend, use six isolated per-pair requests with the leanest direct/model-wrapper route available; do not resurrect the old persistent Step 11 agent loop.

## Workflow

1. Confirm `step10_suno_ready_production_wrap.md` and `step11_enhanced.md` exist in the pair directory.
2. Identify the assigned personality for this pair (from orchestrator pair assignments or handoff).
3. Read the full personality YAML from `skills/orchestration/personalities/<persona>.yaml`.
4. Read `vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md` and use `references/suno_rules_condensed.md` for the inline rules block.
5. Read the production mandates from the handoff file or use defaults.
6. Run `scripts/build_fable_prompt.py <pair_dir> <personality_yaml> <output_path>` or construct manually.
7. For Fusion mode, add a short top-level instruction block naming the intended panel (`anthropic/claude-opus-4.8`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`) and explicitly state: "Use one isolated Fusion request for this pair; this is not an agent loop."
8. Verify: full personality YAML present, Suno construction rules present, step10 present, step11 present, 1K/5K constraints stated, Disc_Channel/EMO rules stated.
9. Output to `<pair_dir>/step11_fable_prompt.md`, `<pair_dir>/step11_opus_prompt.md`, or `<pair_dir>/step11_fusion_pair_request_prompt.md`.

## Personality Matching

- Use only full archive YAML files from `skills/orchestration/personalities/`
- Never use abbreviated handoff personality blocks
- If no archive file exists for a personality, substitute the closest available with full YAML
- LOFN-PRIME (`lofn-prime-mini.yaml`) is the default fallback

## Mandatory Inclusions

Every prompt file must contain:
1. Context header (run, theme, constraint, pair, focus)
2. **Full personality YAML content — the ENTIRE file, zero bytes trimmed.** For LOFN-PRIME this is `lofn-prime-mini.yaml` (~1800 lines, ~50KB). Never substitute a compact/abbreviated block.
3. Suno Prompt Construction Guide condensed rules (7 principles + 7-position order + character limits + format rules) — embed the FULL `references/suno_rules_condensed.md` file
4. Production mandates — embed the FULL mandate text, not summaries
5. Refinement instructions with non-negotiable technical constraints (1K/5K, Disc_Channel, EMO, intro/prod)
6. Critical preservation rules
7. **Full step10 output — the ENTIRE file. All sections: hook note, personality note, continuity payload, music prompt, negative prompt, public lyrics, suno lyrics, vocal fingerprint, style-axis lock, arrangement dramaturgy, production dramaturgy, image ladder audit, controlled fracture, ghost verse bank, panel ledger, QA report.** Do not extract only music prompt + lyrics.
8. **Full step11 output — the ENTIRE file.** For per-pair step11 files: complete. For single cross-pair synthesis: embed the full synthesis.
9. **Suno two-field mandate (2026-06-14):** final output must include a positive `style` / `[SUNO STYLE PROMPT:]` field and a separate `exclude` / `[SUNO EXCLUDE PROMPT:]` field. Style stays positive, 850-1000 chars. Exclude is concrete comma-separated negative-control terms, 400-900 chars, hard max 1000.

## Fusion Mode

Use Fusion mode when The Scientist asks for "Fusion" but wants controlled, explicit spend.

Fusion mode produces isolated pair prompts:

- For a single pair: one complete `step11_fusion_pair_request_prompt.md`.
- For all six pairs: one archive containing six isolated pair prompts; invoke as six Fusion calls, one pair per request, when explicitly approved.
- One combined all-pairs prompt is a fallback packaging artifact only, not the preferred invocation.
- The prompt may name the intended Fusion deliberation panel, but the local agent must not call OpenRouter by itself unless The Scientist explicitly approves the spend in the current turn.
- Include "do not cross-pollinate pair structures" whenever more than one pair appears in the same archive or fallback combined prompt.
- Include the current Suno two-field rule at the top of the prompt.

## Size Expectations

A properly built prompt file is **120-200KB** (2000-3000 lines). If your output is under 50KB, you have truncated something. The personality YAML alone is ~50KB.

## ANTI-PATTERNS — NEVER DO THESE

- ❌ Using a compact/abbreviated personality block instead of the full YAML file
- ❌ Extracting only "music prompt + lyrics" from step10 — include ALL sections
- ❌ Truncating large files with "[... truncated ...]" markers
- ❌ Using the write tool to build the prompt manually from memory — use the script or copy the full source files verbatim
- ❌ Building prompts under 50KB — that means you cut something

## Reference Implementation

The gold-standard reference is the Fable 5 Ceremony run:
`output/daily/2026-06-10/daily-run/audio/pair_XX/pair_XX_fable_prompt.md`
These files are 135-189KB each and contain the complete pipeline output. Use them as your template.

## References

- `references/suno_rules_condensed.md` — Condensed Suno construction rules for inline embedding
- `scripts/build_fable_prompt.py` — Automated builder script
- `~/.openclaw/workspace/skills/orchestration/personalities/lofn-prime-mini.yaml` — Full LOFN-PRIME YAML (1800 lines)
