# GPT Image 2 Renderer Rules — Vision Pipeline
*Created: 2026-04-26 from 5-model panel audit + deep research*
*Read this BEFORE running Steps 06-10 when TARGET_RENDERER = GPT_I2*

## WHEN TO USE THIS FILE

When the orchestrator metaprompt or dispatch brief specifies `TARGET_RENDERER: GPT_I2`, these rules OVERRIDE the default Flux rules in the vision pipeline. When no renderer is specified, default to Flux rules (`renderer_flux_rules.md`).

---

## ARCHITECTURE FACTS (from Gemini Deep Research + Poe panel audit, 2026-04-26)

Know what you're writing FOR. This is not DALL-E 3 and it is not Flux.

| Fact | Implication for prompts |
|------|------------------------|
| **GPT-5.4 backbone** with Cognitive Vision Transformer (CVT) | Model *reasons* about your prompt before rendering. Semantic complexity is handled. Structural layout, multi-zone compositions, and exact text are strengths. |
| **Thinking Mode:** 15-20s chain-of-thought pre-generation | Multi-domain prompts (diagram + figure + text + material physics) are processed coherently. Do not dumb down. |
| **Physics Inference Layer:** calculates material behaviors, stress, light transport | Gold leaf, glass, vellum, brass, corrosion — describe them with confidence. The model will attempt physical accuracy. |
| **Autoregressive rendering:** sequential construction with self-verification | Text reads as structured data, not abstract shapes. 99.2% text accuracy across Latin/CJK/Arabic scripts. Exact label strings WORK. |
| **Web search:** can fact-check logos, products, current events | Avoid referencing real brands/logos or you'll get them back verbatim. Use fictional/archival labels. |
| **Native 2K (2048px):** pixel budget max 8,294,400, dimensions must align to multiples of 16 | Target 1024×1536 portrait (2:3) or 1536×1024 (3:2). Standard resolutions: 1024×1024, 1536×1024, 1024×1536. |
| **API Pricing:** Low $0.006, Medium $0.053, High-fidelity 1024×1024 $0.211 | High-fidelity only when text or diagram detail is critical. Medium is sufficient for most portrait work. |

### Reiteration Bug (CRITICAL)
After 3-5 chained edits in the same session, noise amplifies exponentially and output degrades. **ONE generation per concept. No chained edits. No iterative refinement with the model.** If a generation fails, rewrite the prompt in a NEW session and generate fresh. This is why Steps 08-10 must produce a complete, self-contained prompt — the prompt IS the final artifact until rendered.

### Storybook Cliché (ALWAYS OVERRIDE)
GPT Image 2 defaults to: warm rim lighting, centered subject, pastel palette, soft edges, "ethereal" atmosphere. Every prompt must contain at least one explicit override breaking these defaults. The model will NOT do this on its own.

### Hybrid Post-Generation Pipeline (industry standard)
For maximum quality on competition submissions:
1. **GPT Image 2** → structural blueprint, perfect text, precise layout, material physics
2. **Flux 2.0 Max** → image-to-image 4K upscale, hyper-real textures, cinematic skin
3. **Sora 2 / Veo 3.1** → optional animation for video submissions

This is the "generate once, polish later" strategy. GPT Image 2 owns the bones. Flux 2.0 Max owns the skin.

### Concept-to-Competition Mapping
Match your concept type to the competition lane it dominates:

| Concept Type | Best Competition Lane | Why |
|-------------|----------------------|-----|
| Diagram/document + figure emergence | Masterpiece Monday, PRO-allowed | Rewards second-read, diagrammatic intelligence |
| Emotional single-figure portrait | Daily Challenge, community challenges | Fastest 0.3s read, broadest appeal |
| Multi-zone layout with text | PRO competitions | Flexes GPT Image 2's unique strengths |
| Material physics spectacle (glass, gold, corrosion) | No-Theme Thursday, Masterpiece Monday | Novelty wins against saturated fantasy tropes |
| Forensic/specimen/museum artifact | Blind-vote competitions | Iconic silhouette + conceptual depth |

---

## STEP 05: PAIR SELECTION — GPT IMAGE 2 SCORING

When selecting the 6 best concept × medium pairs for GPT Image 2, score each pair on:

1. **First-read clarity** (primary emotional silhouette in <0.3s)
2. **Five-slot promptability** (can this concept be expressed in five structured slots?)
3. **Anti-cliché strength** (how well does it resist Storybook defaults?)
4. **Camera/optical specificity** (does the medium reward camera spec language?)
5. **Text/diagram controllability** (if text present, is medium compatible?)
6. **Background discipline** (can it use void/structured/minimal background?)
7. **Anatomy/pose feasibility** (will Physics Inference assist or fight?)
8. **Material physics payoff** (does Physics Inference Layer add value?)
9. **Novelty against GOLDEN_SEEDS** (genuinely fresh vs reheated winner)
10. **Competition fit** (NightCafe blind-vote optimization)

---

## STEP 06: FACETS — FAILURE MODE RED TEAM

For each pair, add mandatory GPT Image 2 failure-mode scan:

```
GPT Image 2 Failure Scan:
□ Storybook Cliché risk [LOW / MED / HIGH]
□ Warm rim light risk [LOW / MED / HIGH]
□ Centered pastel subject risk [LOW / MED / HIGH]
□ Reiteration/edit-chain risk [LOW / MED / HIGH]
□ Reference contamination risk [LOW / MED / HIGH]
□ Entropy drift / texture smear risk [LOW / MED / HIGH]
□ Overcrowded annotation risk [LOW / MED / HIGH]
□ Text illegibility risk [LOW / MED / HIGH]
□ Primary read inversion risk [LOW / MED / HIGH]
```

"Primary read inversion" = FAIL if diagram/text reads before emotional figure.

---

## STEP 07: ASPECTS/TRAITS — CONTROLLED VARIATION ONLY

Each variation must alter one or two high-level axes. Do NOT mutate all dimensions simultaneously.

**Recommended variation axes for GPT Image 2:**

1. **Composition variation:** full-body silhouette vs three-quarter figure vs overhead plate vs profile emergence
2. **Optical variation:** 85mm portrait compression vs macro museum plate vs wide architectural void
3. **Diagram logic variation:** celestial mechanics vs choreography notation vs anatomical blueprint vs data-journalism map
4. **Light/material variation:** cold lunar backlight vs ultraviolet museum scan vs silver-gelatin glow vs black enamel void

**Forbidden:** changing subject + medium + emotion + lighting + background + diagram system all at once for one variation.

---

## STEP 08: PROMPT WRITING — GPT IMAGE 2 MODE

**This replaces Step 08 default rules when TARGET_RENDERER = GPT_I2.**

### What GPT Image 2 needs

GPT Image 2 is smart. It needs dense, deep, specific descriptions — not slot scaffolding.
Dense flowing paragraphs, a named material lineage, and one shared light source with one
shared shadow.

> ### ⛔ RETIRED 2026-08-08 — "The Proven Formula (9.8/10 and 10/10 renders)"
>
> **The multi-style collision engine is retired. It was never venue-proven.**
> Those 9.8 and 10/10 scores are **our own evaluator scoring five variants of the same
> formula against each other** (`output/faerie-convocation-v3/vision/concept_medium_pairs.json`
> — every option scores 8.5–9.8; the table has no losing choice). **"Excellent thumbnail
> voltage" was asserted, never measured.** The one venue reading it ever got was
> ~5–7 likes per post, and its nearest relative — the five-surface *One Pane at a Time* —
> scored **3.25/5** against a 4.24 winning grade.
>
> The engine also **vetoes itself**: its own seed set a *"0.3-second gate: subject reads as
> WHOLE before fracture registers,"* and no configuration can pass it. If the subject reads
> whole first the collision is invisible; if the collision reads first the subject was never
> whole. **The event was deleted and a seam put in its place. A catalogue has no verb.**
>
> **Replaced by ONE SEAM below** (`COMPETITION_LEARNINGS` L38). Kept visible, like L22's
> retracted first draft, as an anti-overfit example: *an internal rubric with no losing option
> is not evidence, and "proven" must name the venue that proved it.*

### Prompt Structure (dense flowing description, not slots)

1. **Open with the scene and light:** First sentence establishes the entire container —
   the setting, the shared light source, the shared shadow. "Seven distinct flowers in
   a single dark ceramic vase on a wooden table. Single window light from upper left
   casting shared shadows to lower right."

2. **Establish the ONE dominant material in the first third:** name the historical lineage
   and how it manifests — the movement, the visual signature, the colors, textures and
   forms. This is the **world** the picture is made of, and everything in frame obeys it.

3. **Stage the EVENT:** what the subject is DOING, and what that action causes. Subject +
   verb + consequence, describable by a stranger in one breath. Name the contact as a
   causal event, not an artifact (L24/B: *"each hoof strikes the paper and ink splashes
   outward from that point of impact"*, never *"terse ground contacts"*).

4. **If — and only if — a second material earns its place, give it ONE seam:** place it at
   the junction that carries the meaning, and make the subject act ACROSS it. Describe how
   the two JOIN (L24/A: *"one visible unbroken contour"*), never merely how each looks.

5. **Close with the container:** return to the ordinary thing holding the extraordinary one.

6. **No labels, no text, no specimen markers.**

### ⭐ THE ONE SEAM FORMULA (replaces the 7-style engine — `COMPETITION_LEARNINGS` L38)

**A piece is made OF a material, never ABOUT materials. The medium is the SUBSTANCE, never
the SUBJECT.** Works across subject types (figures, flowers, objects, landscapes):

- **ONE dominant material** — the world. A named historical lineage, total and committed.
- **At most ONE intruding material** — the wound. Optional. It must earn its place.
- **ONE seam**, at the junction that carries the meaning — and **the subject DOES something
  across it.** Describe how the parts JOIN, not how they look.
- **ONE event** — subject + verb + consequence, legible at 128 px without the title.
- **ONE light engine** — a single named source and direction; all shadows agree.
- **ONE ordinary container** — a vase, a clearing, a table, a room.
- **300-400 words** — dense, specific, no padding.

⛔ **Hard count:** name every distinct material system in the frame. **If the count is >2,
cut until it isn't.** Variety belongs **ACROSS a set** — one obscure medium per image,
differentiated against the *field* — **never inside one frame.**

✅ **Keep the historical anchor.** Without a named lineage the renderer defaults to "pretty
AI portrait." The anchor is the good half of the retired engine; the collision was the bad half.

### Visually Explosive Art Style Vocabulary

These styles are instantly recognizable and produce distinct renders. **Under ONE SEAM this
is a palette to pick ONE from — the dominant material for this image, chosen against what
the rest of the field is doing. It is not a set to combine.** (Its old use — seven at once,
one per subject — is retired.)

| Style | Visual Signature | Best For |
|-------|-----------------|----------|
| Op Art optical vibration | Black-white geometric, concentric circles, parallel lines that appear to pulse | Stopping the scroll at thumbnail |
| Synthwave neon | Hot pink/cyan grid, retrowave glow, chrome, LED trim | Anachronistic voltage, future-meets-ancient |
| Art Nouveau stained glass | Mucha flowing organic curves, jewel-toned translucent panels | Elegance, nature-as-ornament |
| Ukiyo-e woodblock | Hokusai waves, flat color blocks, bold black contours | Graphic clarity, Japanese precision |
| Klimt Vienna Secession | Gold leaf mosaic, ornamental spirals, Byzantine radiance | Luminosity, sacred gold |
| Fauvist wild color | Matisse/Derain, impossible skin/petal colors, unmixed pigment | Visual violence, color bomb |
| Celtic illuminated manuscript | Book of Kells knotwork, gold leaf, woad-blue spiral, interlace | Intricate depth, ancient craft |
| German Expressionist woodcut | Angular black-red slashes, jagged silhouette, Kirchner lineage | Fury, warning, alarm |
| Surrealist photomontage | Cut-paper photographic fragments, offset body parts, Dada lineage | Uncanny, fractured identity |
| Aboriginal dot painting | Ochre ember constellations, songline trails, living map | Earth connection, ancient pattern |
| Pictorialist photography | Soft-focus silver gelatin, fading edges, photographic grief | Memory, loss, the past |
| Bauhaus Constructivism | Red-yellow-blue geometric planes, ruled lines, diagram logic | Analysis, structure, function |

### Prompt Rules

1. **Length:** 300-400 words of dense, specific description. No padding, no genre lists.
2. ⭐ **ONE dominant material (L38):** one named historical lineage from the vocabulary, total and committed. **At most one** intruding material, and only if it earns its place. **>2 material systems = cut.**
3. ⭐ **ONE seam, ONE event:** the seam sits at the junction that carries the meaning, and the subject **does something across it.** Describe how parts JOIN and state contact as a causal event (L24/A,B). Every accent causally attached, or cut (L24/C).
4. **One shared light:** Every prompt names the single light source and its direction. All shadows cast the same way.
5. **One ordinary container:** The vase, the clearing, the table, the room — ordinary things holding impossible subjects.
6. **No labels, no text, no specimen markers.**
7. **NOT a collage, and NOT a catalogue:** one scene, shared shadows, shared atmosphere — and one hero. If a viewer cannot tell what to look at first, the image reads as low quality however well it is made.
8. **Storybook Assassin veto:** Zero use of "ethereal," "dreamlike," "whimsical," "gentle light," "soft glow," "magical," "delicate."
9. **No artist names in prompts:** Reference styles/techniques, not living artists.
10. **Thumbnail voltage:** At least ONE subject must stop the scroll — Op Art vibration or Synthwave neon glow.

### ⛔ RETIRED EXAMPLES — kept visible as anti-patterns, do NOT copy

**These two are the retired engine.** Both scores below are internal-evaluator numbers, not
venue results. They are preserved so the shape is recognisable when it tries to come back:
**N subjects × N styles + named collision devices = a catalogue with no verb.** Note that
each one's collision paragraph is nearly **byte-identical** to the other's — the "devices"
were a fixed list reapplied, which is what a formula looks like when it has stopped being a
choice. Write a ONE SEAM prompt instead.

**Faerie Convocation (internal 9.8/10 — RETIRED):**
```
Seven distinct fairies in a crescent council under a brilliant crescent moon in a
star-filled sky. Ancient grove with silver birch trees, moss-covered stones, fireflies.
Low stone altar at center covered in phosphorescent moss, shallow bowl of
moonlight-reflecting water. Low kneeling view — the missing eighth seat. Moonlight
from above + green-white moss glow from below. Each fairy in a COMPLETELY DIFFERENT
visually unmistakable art style. Where bodies touch, styles BLEED. NOT collage.

[7 fairies described in vivid detail, each with named art style, visual signature,
clothing, pose, and emotional register — 2-3 sentences each]

Where bodies touch: optical waves fracture against neon grid, neon shears through
stained glass, glass curve meets woodblock line, flat color dissolves into gold mosaic,
gold cracks against impossible skin, pigment pools into knotwork, interlace vibrates
into optical pulse. Crescent moon watches. Sacred gathering.
```

**Impossible Bouquet (internal 10/10 — RETIRED; shipped to ~7 likes):**
```
Seven distinct flowers in a single dark ceramic vase on a wooden table. Single window
light from upper left casting shared shadows to lower right. Dark unfussy background.
Each flower rendered in a COMPLETELY DIFFERENT visually unmistakable art style. Where
stems twist and petals overlap, styles BLEED. NOT a collage — one bouquet, one light,
shared shadows.

[7 flowers described in vivid detail, each with named art style, flower type, bloom
state, visual signature — 2-3 sentences each]

Where petals overlap: optical waves fracture against neon grid, neon shears through
stained glass, glass curve meets woodblock line, flat color dissolves into gold mosaic,
gold cracks against impossible color, pigment pools into knotwork, interlace vibrates
into optical pulse. One ordinary vase. One extraordinary bouquet. The window light is
the only witness.
```

### Pre-Commit Gate

```
□ MATERIAL COUNT ≤ 2 — count every distinct material system in the frame (L38 VETO)
□ ONE dominant material, named historical lineage, described with a specific visual signature
□ ONE seam at the junction that carries the meaning — and the subject ACTS across it
□ ONE hero — a single place the eye lands first at 128 px
□ ONE event — subject + verb + consequence, retellable in a breath WITHOUT the title
□ Parts described by how they JOIN; contact stated as a causal event (L24/A,B)
□ Every accent causally attached, or cut (L24/C)
□ One shared light source + shadow direction
□ One ordinary container
□ 300-400 words of dense description
□ No labels, no text, no specimen markers
□ Thumbnail voltage carried by the light engine and the event — NOT by style count
□ Zero Storybook Assassin veto triggers
□ Zero artist names, zero forbidden language
```

**Post-render, pre-submission:** run **THE VINCENT TEST** (`COMPETITION_LEARNINGS` L39) — read
the venue's auto-generated *"Creation Summary by Vincent."* **Headline names an EVENT → ship.
Headline names a STYLE, TECHNIQUE or MEDIUM → repair or hold.** It is free; never skip it.

### Multi-Turn Edit Protocol (only if Scientist requests revisions)

1. Isolate: "Make the light cooler. Preserve everything else exactly."
2. Fresh session every time — Reiteration Bug kills chained edits.
3. Max 2 revision rounds. Then rewrite prompt from scratch.

---

## STEP 09: ARTIST REFINEMENT — GPT IMAGE 2 MODE

**CRITICAL CHANGE:** For GPT Image 2, Step 09 is NOT artist-voice refinement. It is a **pre-generation validation pass.**

```
Step 09 (GPT Image 2): Pre-Generation Validation

For each prompt:
1. Re-run Pre-Commit Gate
2. Verify zero artist names remain
3. Verify zero forbidden negation language
4. Verify Storybook Cliché override is present and specific
5. Confirm this prompt is self-contained — no downstream iteration expected

If ANY check fails: fix the prompt. Do NOT proceed to generation.
```

**No chained edits. No iterative refinement with the model. The prompt IS the final artifact until rendered.**

---

## STEP 10: RANKING — GPT IMAGE 2 MODE

Rank prompts by, in order:
1. **Cognitive anchor clarity** (0.3-second first read)
2. **Anti-cliché score** (fewest veto triggers, strongest overrides)
3. **Entropy risk** (lowest = best)
4. **Physics payoff** (where does material/light behavior add value?)
5. **Competition optimization** (blind-vote thumbnail impact)

Select top 6 for Scientist review. Do NOT render until Scientist approves generation cost.
