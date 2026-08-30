# GPT Image 2 Playbook — Competition-Grade Prompt Engineering
*Created: 2026-04-26 based on multi-stream deep research + 5-model panel audit*
*Status: Living document — update as model behavior changes*

---

## ARCHITECTURE FACTS

- **Backbone:** GPT-5.4, Cognitive Vision Transformer (CVT)
- **Rendering:** Autoregressive (not diffusion) — sequential construction with self-verification
- **Thinking Mode:** 15-20s chain-of-thought pre-generation with web search + Physics Inference Layer
- **Text accuracy:** 99.2% across Latin/CJK/Arabic scripts
- **Physics Inference:** Calculates material behaviors, refractive indices, stress points. "AI hands" largely solved.
- **Resolution:** Native 2K (2048px), 4K upscaling. Pixel budget max 8,294,400, dimensions multiple of 16.
- **Pricing:** Low $0.006, Medium $0.053, High-fidelity $0.211 per 1024x1024.

---

## FIVE-SLOT PROMPT FRAMEWORK

**This replaces the Flux "description, not instruction" rule for GPT Image 2.**

```
Slot 1: SCENE (highest weight — first 30% of tokens)
Slot 2: SUBJECT (figure, position, material, light interaction)
Slot 3: IMPORTANT DETAILS (material specificity, diagram, impossible element, text)
Slot 4: USE CASE (implied purpose / emotional register / viewer relationship)
Slot 5: CONSTRAINTS (camera spec, technical, anti-default overrides — ADDITIVE ONLY)
```

---

## CRITICAL RULE INVERSIONS (from Flux mode)

| Flux Rule | GPT Image 2 Rule |
|-----------|-----------------|
| "Description, not instruction" | "Directive description — front-loaded, camera-spec, constraint-explicit" |
| Start with noun-first, present-tense | Start with Scene anchor (where/light/void), then Subject |
| "Narrative incompleteness" | "Cognitive incompleteness" — first read must resolve; second read must not |
| "Artist voice refinement" | "Material grammar" — no artist names; describe medium/technique directly |
| 80+ words | Structured completeness, not arbitrary length; first 60 tokens critical |
| Random injection of aesthetics | Strategic omission — avoid injecting incompatible textures |
| Negative prompts ("no clutter") | Additive directing ("abyssal black void fills the frame") |

---

## MANDATORY PROMPT ELEMENTS (every GPT Image 2 prompt)

1. **Scene anchor** in first 30% of tokens: environment, void, structure, light source
2. **Camera specification**: lens (mm), aperture (f/stop), focus point, film/sensor look
3. **Material specificity with physics claim**: surface, aging state, light interaction behavior
4. **Lighting specification as source + direction + temperature**: NOT mood ("warm"), NOT vague ("dramatic")
5. **Additive background**: "abyssal black void fills the frame" — never "no clutter" or "clean background"
6. **Explicit Storybook Cliché override**: reject warm rim light, centered framing, pastel palette, soft edges
7. **Typography spec** (if text present): typeface category, weight, point size, tracking, color, rendering condition

---

## STORYBOOK CLICHÉ — MANDATORY OVERRIDE

GPT Image 2 defaults to: warm rim lighting, centered subject, pastel palette, soft edges, "ethereal" atmosphere.

**Every prompt must override with at least one of:**

| Default | Override Example |
|---------|-----------------|
| Warm rim light | "Single hard source from below at 15° elevation, 3200K, casting sharp shadow" |
| Centered framing | "Figure placed at lower third, decentered left by 40%" |
| Pastel palette | "Saturated complementary pair with near-achromatic ground" |
| Soft edges | "Hard material boundary with internal luminosity" |
| Storybook warmth | "Cold institutional distance, forensic lighting, temporal suspension" |

**Storybook Assassin veto list** — presence of ANY of these in a GPT Image 2 prompt = auto-fail:
"ethereal," "dreamlike," "whimsical," "gentle light," "soft glow," "magical," "delicate"

---

## ADDITIVE DIRECTING (mandatory)

**Never use negation.** Every constraint must describe what IS present:

| ❌ Forbidden | ✅ Required |
|-------------|------------|
| "No clutter" | "Abyssal black void fills the frame. No objects. No texture." |
| "Not centered" | "Figure placed at lower third, decentered left by 40%" |
| "Avoid warm light" | "Single hard source from below, 3200K or colder" |
| "Without text" | "Annotations are non-linguistic glyphic marks" or no mention of text |

---

## TYPOGRAPHY SPECIFICATION (when text is present)

Any text element requires: typeface category, weight, approximate point size, tracking, color, rendering condition.

```
Example: "9pt condensed serif, tight tracking, oxidized black ink on cream ground, 
slight impression into paper surface, museum-plate positioning in lower left"
```

**Text mediums that preserve fidelity:**
- Letterpress on laid paper, architectural blueprint, technical diagram, scientific plate, engraving/etching

**Text mediums that degrade fidelity (avoid when text is critical):**
- Oil painting, watercolor wash, impasto, wet media with bleeding

---

## FAILURE MODES — PREVENTION PROTOCOL

| Failure | Mechanism | Prevention |
|---------|-----------|------------|
| **Reiteration Bug** | Noise amplifies after 3-5 chained edits in same session | One generation per session. Fresh session every time. |
| **Reference Contamination** | Web search bakes source artifacts/watermarks | No artist names in prompts. Describe techniques, not references. |
| **Entropy Drift** | Organic textures tile and smear | Structured/minimal/void backgrounds only. Avoid grass, foliage, crowds. |
| **Storybook Cliché** | Model defaults to warm rim light, centered, pastel | Explicit override required in every prompt. |
| **Primary Read Inversion** | Diagram/text reads before emotional figure | Front-load figure silhouette; verify 0.3-second first read. |

---

## COMPETITION-GRADE TITLE RULES

- Evocative narrative titling improves blind-vote scores (e.g., "Love Lost" 4.26, "The Girl Who Borrowed Wings from Silence")
- Title should prime emotional interpretation before full cognitive processing
- Avoid descriptive prompt-segment titles ("Woman in Rain")
- Format: poetic, mysterious, 3-7 words ideal
- Titles can reframe surreal/anomalous elements as deliberate avant-garde

---

## NIGHTCAFE COMPETITION RULES

- Masterpiece Monday = blind peer voting, not Explore-feed engagement
- First-read impact + second-read conceptual reward
- Avoid saturated clichés: generic fantasy portraits, Rembrandt lighting, Pixar 3D, lazy B&W, miniature diorama
- Underserved opportunity: typographical/diagrammatic storytelling
- Genre hybridization / temporal collision = proven high-engagement formula
- 1:1 ratio = uninspired; 9:16/2:3 = intimate/monumental; 16:9 = cinematic

---

## RENDERING PROTOCOL

1. ONE generation per session — Reiteration Bug prevention
2. No chained edits
3. If refinement needed: new session, rewrite prompt, fresh generation
4. Pre-commit validation gate before generation (see below)

### Pre-Commit Gate (run before every GPT Image 2 generation)

```
□ Scene anchor in first 30% of tokens
□ Camera specification present with lens + aperture
□ Zero Storybook Assassin veto triggers
□ Zero negative/forbidden constraint language
□ Background type declared (VOID / STRUCTURED / MINIMAL)
□ Text (if any) has complete typography specification
□ Storybook Cliché override confirmed
□ Single-shot commitment — no downstream iteration expected
```

---

## HYBRID STRATEGY (industry recommendation)

When maximum quality is required:
1. GPT Image 2 → structural blueprint + perfect text + precise layout
2. Flux 2.0 Max → image-to-image 4K upscale + hyper-real textures
3. Optional: Sora 2 / Veo 3.1 → animation

---

*This playbook was synthesized from Gemini Deep Research, Poe panel analysis, OpenRouter Perplexity Deep Research, OpenAI developer docs, and a 5-model panel audit of the Lofn creative pipeline.*
