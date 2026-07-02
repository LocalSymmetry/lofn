# OVERALL PROMPT TEMPLATE — Image (CREATIVE CONTEXT carrier)

## What this is
The **canonical structure of the CREATIVE CONTEXT block** injected into every image step (00–10). The orchestrator fills it ONCE from its canonical artifacts — `02_golden_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `06_vision_handoff.md` — and the filled block is pasted **verbatim, no summarization** into the `CREATIVE CONTEXT` slot at the top of EVERY step.

> Restores the streamlit `overall_prompt_template.txt` behavior (the filled template WAS the `{input}` for every chain) and aligns it with the current handoff / Panel Ledger architecture.

---

## SLOTS THE ORCHESTRATOR MUST FILL (from the handoff packet)

| Slot | Source | Meaning |
|------|--------|---------|
| `{input}` | user / brief | User's original request + research brief. |
| `{seed}` | `02_golden_seed.md` | The Golden Seed. |
| `{meta_prompt}` | `04_orchestrator_metaprompt.md` | The enhanced creative directive. |
| `{personality}` | personality DNA | The artist persona. |
| `{concept_panel}` | `03_orchestrator_panel_debate.md` | 6 concept experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{medium_panel}` | `03_orchestrator_panel_debate.md` | 6 medium experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{marketing_panel}` | `03_orchestrator_panel_debate.md` | 6 context & marketing experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{flairs}` | `03_orchestrator_panel_debate.md` | The 15 Special Flairs. |
| `{frames_list}` | personality/seed | Seed framing/composition palette. |
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
- **Medium Panel:**
{medium_panel}
- **Context & Marketing Panel:**
{marketing_panel}
- **15 Special Flairs (weave these throughout):**
{flairs}

**The user's original request / research brief:**
{input}

Supplied images (if any): {image_context}

# Seed Framing / Composition Palette
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
### Medium Phase (Steps 04–05): Convene the **Medium Panel** — brainstorm 14 unconventional medium/tool combos, discard, start at Medium 15, use 15–27. For each final medium list ≥5 composition/stylistic techniques (use the Seed Framing palette).
### Artistic Guide, Prompt Writing & Refinement (Steps 06–10): Name mediums/tools first; describe vantage point; weave the twist; reference the emotional palette. Refinement beats creation. Describe hands if present. Each prompt stands alone.

## Reasoning Challenge
- Choose real artists; recreate their words painstakingly. Stick to standard art/media/photography/film tooling. Avoid AI tropes (lighthouses, clocks, phoenixes, bioluminescence, cyberpunk, butterflies) unless called for. Colloquial color/size names over hex/measurements. Favor historical/mythical/cultural tie-ins. Occam's Razor for diffusion models. Identify your phase by your required output.
