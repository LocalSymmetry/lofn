# Step 05 — Refine Medium into Panel-Pressed Song Seeds

Read first:
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/golden_seed_alloy.md`
- `skills/music/references/triple_arch_benchmark_excerpt.md`
- `skills/music/references/EMOTION_TAXONOMY.md`

## Purpose

Step 05 converts Step 04 concept/arrangement material into **4–7 high-pressure song seeds**.

It must not write final lyrics, full Suno prompts, full QA reports, or final title locks.

The goal is not simplification. The goal is focused depth: recognizable surface, strange cathedral underneath.

## Inputs

### 🎯 CREATIVE CONTEXT — full upstream context (Full Context Always)
Source: `04_orchestrator_metaprompt.md` + `06_audio_handoff.md` (**Panel Ledger**). Carry ALL of it into this step:
- User's original request / research brief
- Golden Seed
- Meta-Prompt (the creative directive to follow)
- Personality / Persona (the CORE voice; if not an AI, do not write like one)
- Concept Panel, Medium Panel, Context & Marketing Panel — the 3 orchestrator panels (18 voices)
- 15 Special Flairs

Use also:
- Step 04 concept/arrangement output
- compact Golden Seed operating excerpt
- orchestrator pair assignment when available
- user request / target platform constraints
- Triple Arch benchmark standard

## Panel requirement

**Use the orchestrator's supplied Panel Ledger** (the Concept / Medium / Context & Marketing panels — 18 named voices + 15 Special Flairs from `06_audio_handoff.md`). Embody those exact panelists; do NOT invent a new panel. Each panel's Devil's Advocate / Hyper-Skeptic must genuinely dissent, and every disagreement must visibly alter the seed.

If (and only if) no Panel Ledger was supplied, fall back to concise internal panel pressure:
- Hit Topliner
- Experimental Producer
- Lyric Dramaturg
- EMO Taxonomist
- Lofn Pipeline Architect
- Hostile Hyper-Skeptic

Output only:
- main disagreement
- what changed
- before/after revision

Every disagreement must visibly alter the seed.

## Required output per seed

Each seed must include:

1. **Singer**
   - who is singing
   - physical situation
   - unsaid ache

2. **Surface**
   - dominant body image
   - plain-language ache
   - 3–5 candidate hooks
   - provisional panel-lean hook

3. **Hook craft scan**
   - stress pattern
   - vowel/breath shape
   - singback potential
   - title clarity
   - emotional adoptability
   - risk

4. **Golden Seed Alloy**
   - source/lineage
   - primary medium
   - contrast medium
   - mythic/material escalation
   - one exact sensory/scientific/historical/craft detail if useful
   - forbidden cliché
   - anti-genre contaminant
   - controlled fracture
   - routing destination: hook, verse, bridge, production, sidecar, ghost bank

5. **Triple-Arch ritual paragraph**
   - ordinary image
   - specific image
   - impossible/mythic transformation
   - human trace
   - micro-textures
   - sonic light/shadow
   - afterimage

6. **EMO dramaturgy preview**
   - section-level emotional future plan using the taxonomy
   - no final lyric lines

7. **Production dramaturgy**
   - cradle
   - haunt
   - rupture
   - afterglow
   - one concrete sonic gesture for each

8. **Attention ladder**
   - first 5 seconds
   - first hook memory
   - second-listen reward
   - deep-lore reward

9. **Complexity routing**
   - hook
   - verse
   - bridge
   - production only
   - sidecar only
   - ghost bank

10. **Forced differentiation**
    Each seed differs from siblings on at least three of: medium, fracture, body trace, EMO arc, genre driver, production rupture, mythic transformation, hook rhythm.

## JSON contract

Step 05 has TWO required outputs:

1. The human-readable canonical artifact: `step05_refine_medium.md`.
2. The machine-readable handoff file beside it: `concept_medium_pairs.json`.

Do not bury JSON only inside the markdown artifact. The parent/controller reads the standalone JSON file to spawn lean pair agents. Missing `concept_medium_pairs.json` is a coordinator handoff failure even when `step05_refine_medium.md` is creatively strong.

The standalone JSON must be parseable and contain 4–7 entries. For music, use this normalized schema so pair agents do not have to infer fields from prose:

```json
[
  {
    "pair_num": 1,
    "pair_id": "01",
    "seed_title": "working title",
    "concept": "full refined song seed / concept text",
    "medium": "full production / genre / arrangement medium text",
    "artist_influence": "optional influence or panel pressure; omit artist names from final Suno prompts",
    "singer": {"who": "", "physical_situation": "", "unsaid_ache": ""},
    "surface": {"dominant_body_image": "", "plain_language_ache": "", "candidate_hooks": [], "panel_lean_hook": ""},
    "hook_craft_scan": {},
    "golden_seed_alloy": {},
    "triple_arch_ritual_paragraph": "",
    "emo_dramaturgy_preview": {},
    "production_dramaturgy": {},
    "attention_ladder": {},
    "complexity_routing": {},
    "forced_differentiation": []
  }
]
```

## Hard gates

- 4–7 seeds unless caller specifies otherwise.
- No final lyrics.
- No full Suno prompt.
- Every seed has panel-forced revision.
- Every seed has mythic image pressure, not just an ordinary object.
- Every seed is recognizable but still strange.
