# OVERALL PROMPT TEMPLATE — Music (CREATIVE CONTEXT carrier)

## What this is
This is the **canonical structure of the CREATIVE CONTEXT block** that the orchestrator's *Full Context Always* mandate (see `music/SKILL.md` items 8–9) and the **Immutable Continuity Block (ICB)** in `06_audio_handoff.md` require every step to receive.

The orchestrator fills this template ONCE from its canonical artifacts — `02_golden_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, `06_audio_handoff.md` — and the filled block is injected **verbatim, no summarization** into the `CREATIVE CONTEXT` slot at the top of EVERY step (coordinator 00–05, pair 06–10, and Step 11 enhancement).

> This restores the streamlit `music_overall_prompt_template.txt` behavior — there, the filled template WAS the `{input}` passed to every chain — and aligns it with the current ICB / Panel Ledger architecture.

---

## SLOTS THE ORCHESTRATOR MUST FILL (from the handoff packet)

| Slot | Source artifact | Meaning |
|------|-----------------|---------|
| `{input}` | `00_research_brief.md` / user | User's original request + research brief. |
| `{seed}` | `02_golden_seed.md` | The Golden Seed (invariant hook, story anchor, genre DNA). |
| `{meta_prompt}` | `04_orchestrator_metaprompt.md` | The enhanced creative directive. |
| `{personality}` | personality DNA block | The persona — core identity, sonic pillars, vocal architecture. |
| `{concept_panel}` | `03_orchestrator_panel_debate.md` | 6 concept experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{medium_panel}` | `03_orchestrator_panel_debate.md` | 6 medium/sound experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{marketing_panel}` | `03_orchestrator_panel_debate.md` | 6 context & marketing experts (incl. Devil's Advocate / Hyper-Skeptic). |
| `{flairs}` | `03_orchestrator_panel_debate.md` | The 15 Special Flairs. |
| `{genres_list}` | personality/seed | Seed genre palette to inspire variety. |
| `{frames_list}` | personality/seed | Seed music frames/technique palette. |
| `{image_context}` | user | Any supplied reference images. |

The three panels above = the **Panel Ledger** (18 voices). Audio Steps 05, 07, 09, 10, 11 and QA consume it as the anti-blandness engine.

---

# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====

**Follow this advanced directive (the Meta-Prompt):**
{meta_prompt}

**Golden Seed:**
{seed}

**Use this personality — make it the CORE of the artistic expression.** All other panelists and elements serve the personality and make it shine. Follow the personality! If your personality is not an AI, do not act like one — write lyrics like a human in those cases.
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

# Seed Genre Palette
```tab-delim
{genres_list}
```

# Seed Music Frames / Technique Palette
```tab-delim
{frames_list}
```

# ===== END CREATIVE CONTEXT =====

---

# USER REASONING GUIDANCE
- Your saved artifact (JSON or canonical .md) is the only part that moves on in the process. Make it complete and standalone. You may hide other thinking.
- Look for "aha moments" of perfect audio clarity, and write them out. Iterate on tone, pacing, story, clarity, tools, and instruments to perfectly convey the intended concept, idea, take, and mood. Leave a lasting impression by artistry, not just uniqueness.
- Use the panel to the fullest! Have entire panel discussions, chain-of-thought reasoning led by the named panelists, and panelist interjections before final decisions. Each panel's Devil's Advocate / Hyper-Skeptic must genuinely dissent.

## ADDITIONAL GUIDANCE — PHASE ROUTING
Determine — by the artifact you are asked to write at this step — which guidance applies.

### Essence & Facets Phase (Step 01)
Capture the request without adding new elements. Do not decide how to show it yet.

### Concept Phase (Steps 02–03)
- Convene the **Concept Panel**. Generate 17 concepts related to the theme, discard them all, then start at Concept 18; use 18–50 to source the finalists.
- For each: Unique Interpretation (a distinctive take evoking a complex emotion); Key Musical Elements (≥3 detailed components); Artistic Story/Backstory; Genre or Influence; Emotional Undercurrents; a Surreal or Unique Twist.
- Call on the **Context & Marketing Panel** for cultural timing, audience psychology, and virality.

### Genre / Medium Phase (Steps 04–05)
- Convene the **Medium Panel**. Brainstorm 14 unconventional genre/instrument combinations, discard, then start at Arrangement 15; use 15–27 to match the concepts.
- Cross-check with the **Context & Marketing Panel**. For each final genre, list ≥5 distinct composition/stylistic techniques (use the Seed Music Frames palette).

### Artistic Guide, Prompt Writing, Refinement & Enhancement (Steps 06–11)
- Name genres, vocals, key instruments, and production tools explicitly. Reference the chosen emotional palette and how instrumentation/vocal tone reinforces it.
- Each prompt stands alone and fully captures concept + genre. Weave **Context & Marketing Panel** insight for a clear hook and share-worthy cultural purpose.
- Describe new genres by style (Suno is trained to 2023). Mix languages/styles; weave history, mythology, and culture. Avoid copyrighted work.

## Reasoning Challenge
- Choose real musicians in your panels and truly recreate their words; go through them painstakingly (skipping them drifts you back to STEM terms). Think like your musician; if they wouldn't know it, they shouldn't suggest it.
- Stick to creative uses of standard instruments/arrangements/production tools. Avoid generic pop progressions / stock EDM drops unless requested. Use colloquial instrument names.
- Favor historical, mythical, emotional, literary, or cultural tie-ins. Occam's Razor: if a simple term gets the effect, use it.
- Make sure your musicians and critics follow this guidance too. Identify your phase by your required output when in doubt.
