---
name: lofn-image
description: Run the Lofn image/visual pipeline (steps 00–10) backed by Claude — contest-grade, render-ready image prompts (Flux noun-first by default, GPT-Image-2 directive mode optional). Use for images, pictures, artwork, portraits, visual concepts, or "make an image with the full pipeline". Expects a Phase-0/1 orchestrator packet from the `lofn` skill; if none exists, run `lofn` first. Do NOT use for music, video, story prose, or QA-only audits.
---

# Lofn Image — Claude-backed vision pipeline

Produces render-ready image prompts at Lofn competition grade (the catalog behind 11 first-place finishes). Depth lives in `skills/image/`; this skill runs it with Claude as the engine (hybrid execution per `.claude/skills/lofn/EXECUTION.md`).

## Before you start
1. Confirm a Phase-0/1 packet exists (`core_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `06_vision_handoff.md` with the **ICB / Panel Ledger**, filled `CREATIVE_CONTEXT.md`). **No packet → run `lofn` first.**
2. **Pick the renderer mode** (the metaprompt/dispatch may set `TARGET_RENDERER`):
   - **FLUX / unset** (daily & community challenges) → read `skills/image/renderer_flux_rules.md`. Description-style, noun-first, 80–150 words.
   - **GPT_I2** (PRO) → read `skills/image/renderer_gpt_image2_rules.md`. Five-slot directive, camera-spec, 250–400 words; the orchestrator also swaps in the Typography Structuralist / Physics Epistemologist / Storybook-Assassin panel slots (see `skills/orchestration/SKILL.md`).
3. Visual quality bar + density checklist: `vault/VISION_QA_DEPTH_AUDIT.md` (the Visual Somatic Gate).

## Execution (hybrid)
Coordinator **00–05 inline**, then **6 pairs as parallel subagents** for 06–10. Inject the full `CREATIVE_CONTEXT.md` everywhere. Default cardinality: **6 pairs × 4 = 24 prompts → rank → top picks.**

### Coordinator steps (inline)
| Step | File | Artifact |
|------|------|----------|
| 00 | `skills/image/steps/00_Generate_Image_Aesthetics_And_Genres.md` | `step00_aesthetics_and_genres.md` |
| 01 | `skills/image/steps/01_Generate_Image_Essence_And_Facets.md` | `step01_essence_and_facets.md` |
| 02 | `skills/image/steps/02_Generate_Image_Concepts.md` | `step02_concepts.md` (12 concepts) |
| 03 | `skills/image/steps/03_Generate_Image_Artist_And_Critique.md` | `step03_artist_and_critique.md` |
| 04 | `skills/image/steps/04_Generate_Image_Medium.md` | `step04_medium.md` |
| 05 | `skills/image/steps/05_Generate_Image_Refine_Medium.md` | `step05_refine_medium.md` → **6 pairs** |

### Per-pair steps (parallel subagents, one chain per pair)
| Step | File | Per-pair artifact |
|------|------|-------------------|
| 06 | `skills/image/steps/06_Generate_Image_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/image/steps/07_Generate_Image_Aspects_Traits.md` | `pair_{NN}_step07_aspects_traits.md` |
| 08 | `skills/image/steps/08_Generate_Image_Generation.md` | `pair_{NN}_step08_generation.md` (4 prompts) |
| 09 | `skills/image/steps/09_Generate_Image_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/image/steps/10_Generate_Image_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |

**Describe-render self-check (one capped pass, reuses the existing max-3-attempt loop — `EXECUTION.md` §4).** Before a pair returns its step-10 prompts, it predicts in 2–3 sentences what its Flux/GPT-Image-2 prompt would actually **PRODUCE** — the literal frame a renderer would emit from this exact text (the composition, the dominant material, the light, what reads at thumbnail size) — then diffs that predicted frame against the Golden Seed. Phrase it adversarially: **"name the one way this would render generic"** (the stock-portrait centering, the material named but not foregrounded, the emotion stated in words the renderer can't draw, the Storybook-Assassin softness creeping back in). If the predicted frame drifts from the seed or names a generic outcome, **self-repair ONCE** through the same repair loop, then move on. This is one inline pass by the pair itself — **no dedicated render-verifier subagent, no recursion, no new tier.** It governs fidelity only; the noun-first / word-count / banned-opener contract below is unchanged.

## The image prompt contract (hard gate — non-waivable)
Lead with seed/scene, end with the checklist.

**FLUX mode (default):**
- **Noun-first, present-tense.** Write as if captioning an image that already exists. ❌ Forbidden imperative openers: Create, Design, Make, Render, Generate, Depict, Show, Draw, Build, Produce. ✅ "An archive fairy kneels in layered photogravure darkness…"
- **80–150 words.** Dense, not long — weaker models lose coherence past ~150. Every word earns its place.
- **Medium/material named in the first third** ("Corroded mirror glass and torn washi frame…"); name concrete materials (cracked egg tempera, indigo cloth, dull bronze) — these are the hooks Flux renders.
- **Emotion SHOWN, not named** until a possible final reveal ("one eye carries borrowed strain", not "defiant tenderness").
- **No camera specs, no Kelvin numbers** — use mood words ("cold institutional light", "face-dominant, soft background blur"). Palette as color words.
- **Storybook Assassin ban (all modes):** no "ethereal, dreamlike, whimsical, gentle light, soft glow, magical, delicate, floating". **No living-artist names** — reference techniques ("like a hand-tinted ambrotype"), not artists. Hands mentioned simply.

**GPT_I2 mode:** five-slot directive, front-loaded camera spec (85mm, f/1.8), lighting source+angle+Kelvin+shadow behavior, background declared (VOID/STRUCTURED/MINIMAL), 3+ named devices with visible behavior; apply the mandatory anti-default substitutions ("warm rim light" → "hard axial light from below"; "centered" → "lower third with aggressive negative space"; etc.). 250–400 words.

**Both modes:** subject **legible at first glance** (thumbnail impact); material specificity; emotional legibility in pose/expression. Aspect ratio: upload challenges **3:4**, TikTok/Stories **9:16**, landscape **16:9** (default 9:16 unless told otherwise).

### ⭐ ONE SEAM — N = 1 (hard gate, `COMPETITION_LEARNINGS` L38)
**A piece is made OF a material, never ABOUT materials. The medium is the SUBSTANCE, never the SUBJECT.**
- **One dominant material** = the world. **At most one** intruding material = the wound. **One seam**, placed at the junction that carries the meaning — and **the subject must be DOING something across it.**
- ⛔ **Multi-style collision engines are RETIRED** (`six-ways-v2` / Faerie Convocation / Impossible Bouquet lineage: N art movements sharing one frame at named collision devices). They produce no hero, N spectacle layers, and seams that vanish at 128 px. **A catalogue has no verb.** Variety belongs **ACROSS a set** — one obscure medium per image, differentiated against the field — never inside one frame.
- ✅ **KEEP the historical-medium anchor.** Without a named lineage the renderer defaults to "pretty AI portrait". The anchor is the good half; the collision is the bad half.
- **Pair self-check (state it in step 10):** count the distinct material systems in the frame. **If the count is >2, cut until it isn't.** Name the single seam and what the subject does across it.

### THE VINCENT TEST — post-render, pre-submission (`COMPETITION_LEARNINGS` L39)
NightCafe's auto-generated **"Creation Summary by Vincent"** has never seen the prompt — a **free blind first-read**. **If its headline names an EVENT, the piece works; if it names a STYLE, TECHNIQUE or MEDIUM, the picture is about its own craft — repair or hold.** This is THE BLIND RULE for images; it costs nothing, so it is never skipped.

## QA & delivery
Run **lofn-qa** with the **Visual Somatic Gate** (`vault/VISION_QA_DEPTH_AUDIT.md`, 7-element density checklist) + the renderer pre-gen check from the rules file. Rank by visual concreteness → thumbnail impact → material specificity → emotional legibility → model fit; select top picks. Save each selected prompt per `skills/lofn-core/OUTPUT.md` (note `image_model` + `aspect_ratio` in frontmatter); INDEX last. **Do not call render tools** — emit paste-ready prompt text; the user renders (FAL Flux / GPT Image 2). Refinement (anatomy/hands) is a post-render inpainting pass on the 1–2 chosen images only.
