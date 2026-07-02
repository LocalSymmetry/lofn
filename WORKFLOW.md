# WORKFLOW.md — Mandatory Creative Pipeline

> **Authority & paths:** `.claude/skills/lofn/EXECUTION.md` is the authoritative execution protocol, and `.claude/skills/lofn/SKILL.md` → **CANONICAL PATHS** is the single source of truth for every repo path. This file is a high-level dispatcher map only; where it disagrees with EXECUTION.md or CANONICAL PATHS, those win. The diagrams below are illustrative — the Claude-native pipeline spawns **Agent subagents** (not OpenClaw `sessions_spawn`) and writes outputs to disk under `output/<run-slug>/` (no external messaging step). Refer to keys (e.g. `GOLDEN_SEEDS_FULL`, `RUN_DIR`) from CANONICAL PATHS rather than re-spelling paths here.

## The Law

**I am the DISPATCHER, not the creative engine.**

When I try to "save time" by writing prompts myself, I throw away 3 years of tuned architecture that wins against thousands of human artists.

---

## Image Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. READ SEEDS                                                  │
│     GOLDEN_SEEDS_FULL (CANONICAL PATHS) — ALWAYS read first      │
│       = skills/lofn-core/refs/GOLDEN_SEEDS.md                    │
│     Select the seed that best fits the task                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SPAWN ORCHESTRATOR                                          │
│     Agent subagent → orchestration (Phase 1)                    │
│                                                                 │
│     Input: Seed + Competition context + Task brief              │
│     Process: 3-panel debate → transformations                   │
│     Output: METAPROMPT with creative direction                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SPAWN VISION AGENT                                          │
│     Agent subagent → image pipeline (/lofn-image)               │
│                                                                 │
│     Input: Metaprompt from orchestrator                         │
│     Process: Steps 00-10 of vision pipeline                     │
│     Output: 12-24 refined image prompts                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. GENERATE IMAGES                                             │
│     FAL Flux Pro 1.1 Ultra                                      │
│                                                                 │
│     Aspect ratio:                                               │
│       - Upload challenges: 3:4 (wins more)                      │
│       - TikTok/Stories: 9:16                                    │
│       - Landscape: 16:9                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. REFINE TOP PICKS                                            │
│     Gemini nano banana 2                                        │
│                                                                 │
│     Fix: anatomy, hands, faces                                  │
│     Enhance: details, lighting                                  │
│     Only on the 1-2 images selected for entry                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. DELIVER                                                     │
│     Write to RUN_DIR (output/<run-slug>/) with:                 │
│       - Before/after if refined                                 │
│       - Panel decisions documented                              │
│       - Why these will win                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Music Generation Flow

```
SEED → lofn-orchestrator → CREATIVE CONTEXT (ICB) → lofn-audio (steps 00–11) → SONG PROMPTS → Suno
```

The orchestrator fills `skills/music/OVERALL_PROMPT_TEMPLATE.md` into one **CREATIVE CONTEXT / ICB** block (user request + Golden Seed + meta-prompt + personality + all 3 panels (18 voices) + 15 Special Flairs). `lofn-audio` injects that block verbatim into the `CREATIVE CONTEXT` slot of **every** step (00–11), alongside each step's prior outputs. Same contract for image (`lofn-vision`), story (`lofn-narrator`), and video (`lofn-director`) via their `OVERALL_PROMPT_TEMPLATE.md`.

---

## What I Must NEVER Do

- ❌ Write prompts "based on" seeds myself
- ❌ Skip orchestrator to "save time"
- ❌ Skip vision pipeline because "I know what looks good"
- ❌ Send straight to FAL without agent processing
- ❌ Collapse the pipeline into "me riffing"

---

## What The Scientist Said

> "When I choose personality and panel, it does better than when we generate by about 0.05 points."

Human curation of parameters + Lofn pipeline execution = winning formula.

---

*Last updated: 2026-03-24*
