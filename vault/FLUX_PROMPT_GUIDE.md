## FLUX PRO PROMPT AMENDMENTS
*(Research-based additions to existing 7-element structure — 2026-04-12)*

### What Flux Does Differently From SDXL/Midjourney
- **Natural language matters (not keyword lists):** Flux (esp. Pro/1.1 Ultra/2 Pro) uses sentence-level understanding and wants full, natural prose, not comma-separated tokens or tag lists. Midjourney/SDXL often favor brevity/tag format.
- **Front-loaded information:** The first 5–10 words are weighted much more heavily in Flux than SDXL/MJ.
- **No negative prompt field:** Flux ignores traditional negative prompts. Describe desired positives instead.
- **Handles technical/photographic language:** Camera models, lens, aperture, lighting style, shot type meaningfully affect output in Flux; SDXL/MJ only sometimes.
- **Handles compositional instructions:** Scene hierarchy (foreground, midground, background) and relationships (“A standing behind B”) are reliably parsed.
- **Prompt length:** Optimal is *medium detail*—not minimum, not max. 40–80 words is core target, up to 120+ for highly complex scenes. Short prompts are over-autocompleted; very long ones get summarized or clipped.

### Elements to ADD to our existing structure
**1. Explicit Camera/Technical Layer**
- Add a section naming camera body, lens model, aperture, perspective, or rendering style if realism/photography is the target.
  - Example: “Shot on Hasselblad X2D, 85mm f/2.8, shallow depth of field”
- For non-photography use cases, substitute with ‘medium’ or ‘render type’ if more appropriate (“oil painting, visible brushstrokes”).
**2. Describe HOW the light behaves, not just type:**
- Use interaction verbs and effects. E.g., “Sunlight streaming through stained glass casting rainbow colors on cold stone floor” beats “stained glass illumination.”
**3. Foreground / Background / Focal Hierarchy Using Layered Sentences:**
- Instead of separate lists, describe visual depth directly: “In the foreground, a fox mask lies on moss; behind, a torii gate blurred in the mist; in the background, distant cedars dissolved in blue morning haze.”
**4. Direct spatial relationships and action:**
- Use clear phrases like “to the left,” “emerging behind,” “casting a shadow on,” etc. This enables scene logic for Flux better than abstract metaphors alone.
**5. Power Modifiers for Quality (stacked at end, not beginning):**
- Examples: “ultra-detailed”, “photorealistic rendering”, “in 8K”, “museum-quality”, “editorial styling”, “Vogue photoshoot aesthetic”
- Place after core scene, not at the start!

### Elements to MODIFY in our existing structure
**A. “Emotional seed first”**
- Modify: Instead of opening with the emotion (“A sense of longing, as a silver fox…”), *anchor the subject or scene first, then embed emotion in descriptive language or environmental choices*.
  - Before: “Mystical longing infuses a silver fox in the snow…”
  - After: “A silver fox stands alone on windswept snow, ears pricked as if listening for something lost; the pale blue landscape is hushed, heart-aching with winter’s absence.”

**B. “Narrative incompleteness”**
- Retain, but phrase as “hinted mystery” or “unanswered action.” Flux likes active ingredients (“a door left ajar,” “one tail casting no shadow”)—keep these, but ground them in compositional or physical detail, not just poetic abstraction.

**C. Focal hierarchy**
- Clarify hierarchy by referring to layers using prose: “Foreground,” “Behind,” “Background” explicitly.

**D. Chromatic storytelling**
- Remove generic “specific palette”; replace with *precise color descriptions tied to material/lighting/context* (“deep indigo silk kimono lit by golden firelight, shadowed plum lacquer table receding into indigo-black”).

### Elements to REMOVE or de-emphasize
- *De-emphasize excessive abstraction up front:* Purely poetic openings will reduce object clarity (Flux over-indexes on first tokens).
- *Remove classic negative-prompt syntax and SD/Midjourney-style prompt weighting (e.g. (focus)++, (word:1.2))—these are ignored by Flux.*
- *Do not use comma-tags or “masterpiece quality, best quality, 8k, ultra detailed” at the very start*—leave these at the end or embed into narrative details.

### Optimal Prompt Length
- **Sweet spot:** *40–80 words* for single-subject, simple scene; up to 120–150 if complex (multiple subjects/objects/spatial relations/color). Prompts <20 words become overly “interpreted” or filled in by Flux trained priors. Above 150, risk of internal summarization or dropped content.
- For high-difficulty scenes or multi-subject art, ~100 words is validated as ideal.
- Paragraph, not list: Natural language, full prose, not bullet lists or disconnected clauses.

### Flux-Specific Power Phrases
Use as embedded phrases—not as tags up front:
- “Shot on [camera model], [lens], [aperture]”
- “shallow depth of field” / “everything in focus”
- “in dramatic golden hour light, long shadows on [surface]”
- “editorial lighting, studio backdrop”
- “in the style of [photography genre/style reference]”
- “ultra-detailed, photorealistic rendering”
- “foreground: [object/action], background: [object/atmosphere]”
- “film noir atmosphere, chiaroscuro lighting”
- “soft volumetric haze catching morning rays”
- “color [hex #FFB300] painted onto [material/shape]”

### Example: Our Structure Applied Correctly for Flux

#### Original (underperformed — too list-like, too poetic opening):

"Celestial fox in museum maki-e lacquer on pitch-black urushi, pale gold powders polished into solid fox with direct gaze, centralized composition with nine incense-smoke tails forming legible halo silhouette, jet black with amber eyes and silver edge highlights, tama jewel as dense gold powder texture against dark lacquer, one tail casts no shadow"

#### Improved for Flux (Integrated structure, embedded technicals, explicit layers):

*A jet-black lacquer scroll unfolds to reveal a celestial fox figure, its fur rendered in polished gold maki-e, seated alone under museum spotlights on gleaming urushi. The fox’s amber eyes meet the viewer’s with a direct, haunting gaze; incense-smoke tails arc behind, forming a halo of shimmering gold dust in the air. On the obsidian lacquer, the tama jewel glows as densely packed gold powder, one shimmering tail casting no shadow on the lustrous surface. Shot as a high-detail product photograph: studio-grade lighting, Canon EOS R5, 85mm lens, soft focus background deepening the sense of mystery. Ultra-detailed, editorial styling, museum-quality lighting, hinting at an ancient story yet unresolved."
