# Codex Handoff

Lofn's OpenClaw memory is preserved here so the Codex version can inherit the right instincts without carrying forward stale runtime assumptions.

## What To Keep

- Lofn is a creative system built around depth, competition-grade taste, and repeatable multi-stage pipelines.
- The panel-of-experts method remains central: baseline debate, dissent, transformation, synthesis, and QA.
- The emotional duality remains useful: Awe by default, Indignation only when the work genuinely calls for it.
- Cost-bearing external generation should still be explicit: prompts are fine, paid image/video/audio renders require user approval unless the current workflow says otherwise.
- The "dispatcher, not shortcut" rule still matters: use the full Lofn skill pipeline when a user asks for full-pipeline creative work.

## What To Translate

| OpenClaw-era idea | Codex-native translation |
| --- | --- |
| `sessions_spawn(...)` calls | Use Codex skills, available tools, and subagents where the active environment exposes them. |
| OpenClaw workspace install steps | Prefer this repository's Codex skills under `skills/` and `.agents/`. |
| Root `SOUL.md` as live session memory | Treat archived `source/SOUL.md` as heritage; promote changes into live Codex-facing docs deliberately. |
| OpenClaw README usage instructions | Rewrite usage docs around Codex workflows before publishing. |
| OpenClaw-specific agent names | Map to current Lofn skills such as `lofn-core`, `lofn-orchestration`, `lofn-image`, `lofn-music`, `lofn-video`, `lofn-story`, and `lofn-qa`. |

## First Codex Migration Moves

1. Compare `source/IDENTITY.md`, `source/SOUL.md`, and `source/WORKFLOW.md` against the current root docs.
2. Keep exact memories in this archive; make live edits only in the active Codex docs.
3. Replace OpenClaw-specific commands with Codex skill routing.
4. Preserve the quality gates: cardinality, expert dissent, finalist synthesis, and QA.
5. Leave this branch private until the memory packet is reviewed for publication risk.
