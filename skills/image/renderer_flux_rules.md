# Flux Renderer Rules — Vision Pipeline (Non-PRO / Community Challenges)
*Created: 2026-04-27*
*Read this BEFORE running Steps 06-10 when TARGET_RENDERER = FLUX or not set*

## WHEN TO USE THIS FILE

When the orchestrator metaprompt or dispatch brief specifies `TARGET_RENDERER: FLUX` (or no renderer is specified), these rules apply. Use this for:
- Daily Challenges (PRO creations not allowed)
- Community challenges (The Women in any Style, Female Portrait, etc.)
- No-Theme Thursday and similar
- Any competition where GPT Image 2 / Flux Pro 1.1 Ultra are gated

When `TARGET_RENDERER: GPT_I2`, use `/data/.openclaw/workspace/skills/image/renderer_gpt_image2_rules.md` instead.

---

## MODEL SELECTION BY PROMPT TYPE

Choose the model BEFORE writing the prompt. The model determines the prompt style.

### ✅ RECOMMENDED (non-PRO daily challenges)

| Model | Best For | Prompt Style | Credit Cost |
|-------|----------|-------------|-------------|
| **Flux 2 9B Fast** | Painterly, oil, museum language, figurative, warm palette, portraiture | Description, noun-first, material-specific | Standard |
| **HiDream I1 Fast** | Stylized, illustrative, graphic, flat-color, woodblock, unusual stylistic instructions | Description, style-first, bold | Standard |
| **Crystal Clear XL Lightning** | Semi-realistic, graphic, atmospheric, general fallback | Simple description, keyword-dense | Unlimited |
| **Juggernaut XL Lightning** | Detailed complex prompts, general-purpose | Longer description, detail-heavy | Unlimited |

### ❌ AVOID (confirmed failures on complex prompts)
- Dreamshaper XL Lightning — ignores painterly/museum language
- Z-Image — fails on hybrid/complex concepts
- Flux Kontext Dev — editing/inpainting only, not generation
- Clarity Upscaler — upscaling only, not generation

### 🔧 INPAINTING (post-generation anatomy fixes)
- **Flux 2 Klein 9B Fast (Flux Kontext):** Surgical inpainting for fingers, toes, eyes
- Keep prompts SHORT and directive
- Always end with: "Preserve all other elements exactly"
- Max 2-3 rounds per image

---

## STEP 08: PROMPT WRITING — FLUX MODE

**This replaces the GPT Image 2 Step 08 rules for non-PRO competitions.**

### Prompt Structure: "Description, Not Instruction"

Flux and SDXL-based models respond to **descriptive caption-style prompts**, not directive instructions. This is the inverse of GPT Image 2.

| GPT Image 2 (PRO) | Flux / SDXL (non-PRO) |
|-------------------|----------------------|
| "Directive description — front-loaded, camera-spec, constraint-explicit" | "Description, not instruction — noun-first, present-tense, visually concrete" |
| Five-Slot structured | Single flowing paragraph |
| Camera specs (85mm, f/1.8) | Mood-based lighting ("soft window light," "harsh overhead") |
| Lighting source/angle/Kelvin | Lighting mood ("cold institutional," "warm amber glow") |
| 250-400 words | 80-150 words |
| Additive directing ("abyssal black void fills the frame") | Negative prompts OK in platform UI, not in prompt body |
| Hands described precisely | Hands mentioned simply ("hands rest in lap") |
| Background declared STRUCTURED/VOID | Background described naturally ("against a dark wall") |

### Flux Prompt Rules

1. **Noun-first, present-tense:** "An adult woman stands at a worn doorway, her tired face half-lit by cool rain light." NOT "Place the woman at a doorway."
2. **80-150 words:** Weaker models lose coherence beyond ~150 words. Be dense, not long. Every word must earn its place.
3. **Mediums named early:** First sentence names the primary medium or material treatment. "Corroded mirror glass and torn paper frame a woman at a threshold."
4. **Concrete over abstract:** "Her mouth is steady, one eye carries borrowed strain" NOT "her expression embodies defiant tenderness." SHOW the emotion, don't name it.
5. **Material specificity:** Name the materials — corroded mirror, torn washi, cracked egg tempera, indigo cloth, dull bronze. These are the hooks Flux renders well.
6. **No camera specs:** Flux ignores lens focal length and f-stop. Use mood descriptors instead: "seen from close, face-dominant, soft background blur."
7. **No Kelvin numbers:** Flux responds to color mood words, not temperature numbers. "Cold institutional light" not "4500K."
8. **Palette as color words:** "Indigo, bottle-green, rain-gray, dull bronze, tea-stained paper" not "limited palette of..."
9. **No anti-default overrides:** Flux doesn't have Storybook Cliché defaults. Skip the override language.
10. **Storybook Assassin still applies:** No "ethereal," "dreamlike," "whimsical," "gentle light," "soft glow," "magical," "delicate." These produce generic AI-art gloss on any model.
11. **No artist names in prompts:** Reference styles/techniques, not living artists. "Like a hand-tinted ambrotype" not "in the style of X."
12. **Hands:** Mention hands briefly. "Her hand rests against the doorframe, fingers relaxed." Simple presence is enough for Flux.

### Flux Prompt Example — "The Borrowed Face at the Threshold"

```
Corroded mirror glass and hand-torn washi veil frame a broad-shouldered woman at
a rain-dark doorway, seen close from the shoulders up. Her face is tired and
dignified, a steady mouth, one eye carrying a borrowed strain under a strip of
tea-stained torn paper across her brow — the fibers rubbed thin at the edge.
Indigo cloth wraps her shoulders, a dull bronze latch behind her. The mirror
beside her face is eaten with pitted spots and long blue-green corrosion streaks
running down the doorframe. Dark cracks break through the indigo cloth like torn
velvet at her shoulder. Cool rain light spills from above, a hard side beam from
the doorway casting firm shadows. A portrait of defiant tenderness — she has
received too many confessions and still refuses to become cruel. Face-dominant,
close crop, severe indigo and bottle-green palette against rain-dark gray.
```

**Word count: ~140.** Dense, concrete, noun-first, present-tense. Every sentence adds visual information. Emotion is shown, not named until the final reveal.

---

## STEP 09: PRE-GENERATION CHECK — FLUX MODE

```
Flux Pre-Generation Check:
□ Word count: 80-150
□ Noun-first, present-tense throughout
□ No camera specs (no mm, no f-stop)
□ No Kelvin numbers
□ No "ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate"
□ No artist names
□ Mediums/material named in first third of prompt
□ Hands mentioned (even briefly)
□ Emotion SHOWN through physical description, not named abstractly
□ Dense, concrete, every word earns its place
```

---

## STEP 10: RANKING — FLUX MODE

Rank prompts by, in order:
1. **Visual concreteness** — does every sentence paint a specific image Flux can render?
2. **Thumbnail impact** — does the first-read subject (face, figure, silhouette) dominate?
3. **Material specificity** — are the mediums/textures concrete and renderable?
4. **Emotional legibility** — is the feeling visible in the described expression/pose?
5. **Model fit** — does the style match the chosen model's strengths (painterly vs graphic)?

Select top 6. Do NOT render until Scientist approves generation cost.

---

## QUICK REFERENCE: GPT Image 2 vs Flux

| | GPT Image 2 (PRO) | Flux / SDXL (non-PRO) |
|---|---|---|
| **Rules file** | `renderer_gpt_image2_rules.md` | `renderer_flux_rules.md` (this file) |
| **Prompt style** | Five-Slot directive | Description, noun-first, present-tense |
| **Length** | 250–400 words | 80–150 words |
| **Camera** | 85mm, f/1.8, focus on X | "seen from close, face-dominant" |
| **Lighting** | Source + angle + Kelvin + shadow behavior | Mood words: "cold institutional," "warm amber" |
| **Background** | Declared: VOID / STRUCTURED / MINIMAL | Described naturally |
| **Anti-default** | Explicit Storybook override required | Storybook Assassin veto only |
| **Mediums** | Named in first sentence | Named in first third |
| **Hands** | Described with detail | Mentioned simply |
| **Devices** | 3+ named with visible behavior | Optional, as visual texture |
| **Max models** | GPT Image 2 via FAL | Flux 2 9B Fast / HiDream I1 / Crystal Clear XL / Juggernaut XL |
| **Inpainting** | Not needed (Physics Inference) | Flux Kontext for anatomy fixes |
| **Edit chain** | One-shot only (Reiteration Bug) | Can iterate with inpainting passes |
