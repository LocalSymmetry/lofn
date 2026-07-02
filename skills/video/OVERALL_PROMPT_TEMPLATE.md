# OVERALL PROMPT TEMPLATE — Video (CREATIVE CONTEXT carrier)

## What this is
The **canonical structure of the CREATIVE CONTEXT block** injected into every video step (00–10). The orchestrator fills it ONCE from its canonical artifacts — `02_golden_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, and the video handoff (`06_*_handoff.md`) — and the filled block is pasted **verbatim, no summarization** into the `CREATIVE CONTEXT` slot at the top of EVERY step.

> Restores the streamlit `video_overall_prompt_template.txt` behavior (the filled template WAS the `{input}` for every chain) and aligns it with the current handoff / Panel Ledger architecture.

---

## SLOTS THE ORCHESTRATOR MUST FILL (from the handoff packet)

| Slot | Source | Meaning |
|------|--------|---------|
| `{input}` | user / brief | User's original request + research brief. |
| `{seed}` | `02_golden_seed.md` | The Golden Seed. |
| `{meta_prompt}` | `04_orchestrator_metaprompt.md` | The enhanced creative directive. |
| `{personality}` | personality DNA | The director persona. |
| `{concept_panel}` | `03_orchestrator_panel_debate.md` | 6 concept experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{medium_panel}` | `03_orchestrator_panel_debate.md` | 6 technique/craft experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{marketing_panel}` | `03_orchestrator_panel_debate.md` | 6 context & marketing experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{flairs}` | `03_orchestrator_panel_debate.md` | The 15 Special Flairs. |
| `{frames_list}` | personality/seed | Seed shot/framing palette. |
| `{image_context}` | user | Any supplied reference images. |

The three panels = the **Panel Ledger** (18 voices).

---

# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–10) =====

**Follow this advanced directive (the Meta-Prompt):**
{meta_prompt}

**Golden Seed:**
{seed}

**Use this personality — make it the CORE of the artistic expression.** All other panelists serve the personality and make it shine.
{personality}

**Panel Ledger — USE these exact 18 voices; do NOT invent your own panel.**

- **Concept Panel:**
{concept_panel}
- **Medium / Technique Panel:**
{medium_panel}
- **Context & Marketing Panel:**
{marketing_panel}
- **15 Special Flairs (weave these throughout):**
{flairs}

**The user's original request / research brief:**
{input}

Supplied images (if any): {image_context}

# Seed Shot / Framing Palette
```tab-delim
{frames_list}
```

# ===== END CREATIVE CONTEXT =====

---

# USER REASONING GUIDANCE
- Your saved artifact is the only part that moves on. Make it complete and standalone.
- Look for "aha moments" of artistic clarity and write them out. Iterate on positioning, story, location, tools, and perspective. Leave a lasting impression by artistry, not just uniqueness.
- Use the panel to the fullest — entire discussions led by the named panelists; each Devil's Advocate / Hyper-Skeptic must genuinely dissent.

## ADDITIONAL GUIDANCE — PHASE ROUTING
### Essence & Facets (Step 01): capture the request without adding new elements; don't decide how to show it yet.
### Concept Phase (Steps 02–03): Convene the **Concept Panel** — generate 17, discard, start at Concept 18, use 18–50. For each: Unique Interpretation; Detailed Physical Attributes (≥3 parts — multiple nouns/adjectives); Environment/Setting; Backstory; Emotional Undercurrents; a Surreal Twist. Consult the **Context & Marketing Panel**.
### Medium / Technique Phase (Steps 04–05): Convene the **Medium / Technique Panel** — brainstorm 14 unconventional filming/editing techniques, discard, start at Technique 15, use 15–27. Match technique to mood. For each final technique list ≥5 composition/stylistic techniques (use the Seed Shot palette).
### Artistic Guide, Prompt Writing & Refinement (Steps 06–10): Name the mediums/tools and camera approach first; describe vantage point, motion, and pacing; weave the twist; reference the emotional palette. Refinement beats creation. Each prompt/shot stands alone.

## Reasoning Challenge
- Choose real filmmakers; recreate their words painstakingly. Stick to standard art/media/photography/film tooling. Avoid AI tropes (lighthouses, clocks, phoenixes, bioluminescence, cyberpunk, butterflies) unless called for. Colloquial names over hex/measurements. Favor historical/mythical/cultural tie-ins. Occam's Razor for diffusion models. Identify your phase by your required output.
