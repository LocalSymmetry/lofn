# WORKFLOW.md — Mandatory Creative Pipeline

## The Law

**I am the DISPATCHER, not the creative engine.**

When I try to "save time" by writing prompts myself, I throw away 3 years of tuned architecture that wins against thousands of human artists.

---

## Image Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. READ SEEDS                                                  │
│     lofn-core/GOLDEN_SEEDS.md — ALWAYS read first                   │
│     Select the seed that best fits the task                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SPAWN ORCHESTRATOR                                          │
│     sessions_spawn(agentId: "lofn-orchestrator")                │
│                                                                 │
│     Input: Seed + Competition context + Task brief              │
│     Process: 3-panel debate → transformations                   │
│     Output: METAPROMPT with creative direction                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SPAWN VISION AGENT                                          │
│     sessions_spawn(agentId: "lofn-vision")                      │
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
│     Send to Telegram with:                                      │
│       - Before/after if refined                                 │
│       - Panel decisions documented                              │
│       - Why these will win                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Music Generation Flow

```
SEED → lofn-orchestrator → METAPROMPT → lofn-audio → SONG PROMPTS → Suno
```

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
