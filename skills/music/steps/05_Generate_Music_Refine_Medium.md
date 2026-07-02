# Step 05 — Refine Medium into Panel-Pressed Song Seeds

Read first:
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/golden_seed_alloy.md`
- `skills/lofn-core/refs/EMOTION_TAXONOMY.md`

> Past golden outputs are judge-side only (GOLDEN-OUTPUT QUARANTINE, `.claude/skills/lofn/EXECUTION.md` §3); you get the **GOLDEN MOVE** block (`.claude/skills/lofn-music/SKILL.md`), not the mold.

## Purpose

Step 05 converts Step 04 concept/arrangement material into **6 high-pressure song seeds — one per Phase-1 pair slot**.

It must not write final lyrics, full Suno prompts, full QA reports, or final title locks.

The goal is not simplification. The goal is focused depth: recognizable surface, strange cathedral underneath.

## Slot reconciliation — who owns what at 05 (binding)

Step 05 does not invent its own cardinality: it selects concepts **INTO the six Phase-1 pair slots**. Phase 1 owns each slot's **arm / genre / verse-structure** — those assignments are fixed before any concept exists. Step 05 owns **which of the 12 step-02 concepts fills each slot**, and must record the **runner-up rationale** per slot: which concept filled it and why it beat the alternatives for THAT slot. Cardinality is pinned at **exactly 6** — only The Scientist may downsize, explicitly (`.claude/skills/lofn/EXECUTION.md` §4, Steps 02/05). The 12→6 cut is comparative judgment, not relabeling: the six winners must be genuinely distinct, and the six losers go to the cut ledger below, not the void.

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
- the GOLDEN MOVE block (five rules; golden-song payloads are judge-side only per the GOLDEN-OUTPUT QUARANTINE)

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

11. **Slot record**
    Which Phase-1 pair slot this seed fills (arm / genre / verse-structure as assigned by Phase 1), which step-02 concept won it, and the runner-up rationale: why the winner beat the alternatives for THIS slot.

## Cut ledger (required output — the reserve bench)

After the six slots are filled, list every losing step-02 concept — **one line each: why it lost + one organ worth harvesting** (a device, an image system, a structural trick). Example: `CUT: "Rust Choir" — lost slot 3 to a sharper body-image; harvest the antiphonal call-and-rust-response verse device.`

This ledger is not a courtesy note. It is the **reserve bench that §7.3's REPLACE and REDIRECT routes draw from** (`.claude/skills/lofn/EXECUTION.md`): when a pair breaks open downstream, the coordinator may promote a reserve concept from this ledger into the empty slot — the promoted reserve inherits the slot's Phase-1 arm/genre/verse-structure unchanged. A killed branch with no ledger line is unrecoverable; a missing cut ledger is a gate failure, not a style choice.

## JSON contract

Step 05 has TWO required outputs:

1. The human-readable canonical artifact: `step05_refine_medium.md`.
2. The machine-readable handoff file beside it: `concept_medium_pairs.json`.

Do not bury JSON only inside the markdown artifact. The parent/controller reads the standalone JSON file to spawn lean pair agents. Missing `concept_medium_pairs.json` is a coordinator handoff failure even when `step05_refine_medium.md` is creatively strong.

The standalone JSON must be parseable and contain **exactly 6 entries — one per Phase-1 pair slot** (fewer only when The Scientist explicitly downsized the run). For music, use this normalized schema so pair agents do not have to infer fields from prose:

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
    "forced_differentiation": [],
    "phase1_slot": {"arm": "", "genre": "", "verse_structure": ""},
    "slot_rationale": "which step-02 concept won this slot and why it beat the alternatives"
  }
]
```

## Hard gates

- Exactly 6 seeds — one per Phase-1 pair slot (arm/genre/verse-structure fixed by Phase 1; concept content chosen here). Only The Scientist may downsize, explicitly.
- Every seed records its slot + runner-up rationale (which concept won and why it beat the alternatives).
- The cut ledger is present: one line per losing concept — why it lost + one organ worth harvesting.
- No final lyrics.
- No full Suno prompt.
- Every seed has panel-forced revision.
- Every seed has mythic image pressure, not just an ordinary object.
- Every seed is recognizable but still strange.
