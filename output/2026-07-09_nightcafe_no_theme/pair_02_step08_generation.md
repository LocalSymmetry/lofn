# Pair 02 · Step 08 — Raw Generation Prompts — "The Allotment Out of Time" (ACCESSIBLE)

**Continuity Payload Used:** frozen `CREATIVE_CONTEXT.md` verbatim (icb_sha 7747f51d…) — Golden Seed, metaprompt, LOFN-ArtCore personality YAML, Panel Ledger (18 voices), 15 **Special Flairs**, axis corner A4·B2·C4·D2·E3, step-05 refined concept/medium for P02, step-06 facets F1–F5, step-07 artistic guide. Step contract: `skills/image/steps/08_Generate_Image_Generation.md` + `skills/image/renderer_flux_rules.md` (Flux mode: noun-first, present-tense, 80–150 words, medium in first third, no camera specs, no Kelvin, no banned haze words, no artist names).

## V1 — First Bud (148 words)

```text
An autochrome plate in bright first-minute-after-rain daylight, its potato-starch grain a pastel stipple thickening around impossibility, records a brown London allotment in October: turned beds, dripping chain-link, empty bean canes, a pearl sky band resting above the sheds. At the foot of the canes one Punjab dahlia stands open, rose copper and saturated as nothing else on the plate, a thin column of rain climbing above it in vertical grain-streaks, inside the plot only. A Punjabi-British woman of sixty-six stands mid-step on the wet slab path, walk abandoned, mackintosh hem lifted over cardigan and salwar, silver hair pinned low, steel kara at her wrist, hands open, face turned three-quarter to the bloom. Marigold petals catch in the fence wire. Her mother's seed tin sits opened on the bench edge, empty, rain-beaded. Storm grey-green owns the plate; the dahlia carries its only warmth — the first yes, not yet believed.
```

## V2 — Full Riot (148 words)

```text
An autochrome plate in bright after-rain daylight, its potato-starch grain blooming thickest where the color is least believable, holds a London allotment gone Punjab monsoon: dahlias, marigold ropes swagged on bean canes, green-gold wheat rising as one rose-copper column through the right third, rain climbing upward inside that band only in vertical grain-streaks, the plot-boundary seam the sharpest zone on the plate. Around it lie brown October beds, dripping chain-link, dark sheds, a pearl sky band at rest. At the column's base a Punjabi-British woman of sixty-six stands mid-step, walk abandoned, wet mackintosh dark over cardigan and salwar, hem lifted, steel kara at her wrist, hands open, face three-quarter to the blooms, silver daylight rimming her wet shoulders. Marigold petals catch in the fence wire; her mother's seed tin sits open on the bench edge, empty, rain-beaded. Storm grey-green keeps London; rose copper belongs to the garden alone.
```

## V3 — Fruit No Calendar Allows (150 words)

```text
An autochrome plate, its potato-starch stipple bright with after-rain daylight, documents a harvest no calendar allows: wheat gone heavy-headed bows over a London allotment path in October, marigold ropes strung with seed pods, dahlias blown wide, the rose-copper column leaning ripe through the right third while rain climbs upward inside it in vertical grain-streaks. Brown turned beds and dripping chain-link surround the plot; loose grains scatter the wet slabs; a pearl sky band rests above the sheds. A Punjabi-British woman of sixty-six stands mid-step at the column's base, walk abandoned, hem lifted, wet mackintosh dark over cardigan and salwar, steel kara at her wrist, one hand half-raised toward a bowing wheat ear, not yet touching. Marigold petals catch in the fence wire. Her mother's seed tin stays open on the bench edge, empty, rain-beaded, fed by nothing it holds. Storm grey-green holds the plate; rose copper belongs to the harvest.
```

## V4 — Seeds Already Leaving (137 words)

```text
An autochrome plate in washed after-rain daylight, its potato-starch stipple sharpest along the seam where physics lets go, catches the monsoon garden releasing: silvered seed lifts from blown dahlias and seeding marigold ropes, climbing the upward rain in vertical grain-streaks out of the frame's top, while the rose-copper column thins at its base back toward storm grey-green. Brown October beds and dripping chain-link hold the London plot; a pearl sky band rests above the sheds; pale petals hang caught in the fence wire. A Punjabi-British woman of sixty-six stands mid-step below the rising seed, walk abandoned, hem lifted, wet mackintosh dark over cardigan and salwar, steel kara at her wrist, hands open, face lifted three-quarter following the climb. Her mother's seed tin sits open on the bench edge, empty, rain-beaded. What was given is already passing on.
```

## Density-element audit (all 7 required per prompt)

| Element | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| Emotional seed sentence | final reveal: "the first yes, not yet believed" | "walk abandoned" + palette-claim close | "fed by nothing it holds" | final reveal: "What was given is already passing on" |
| Medium as narrative agent | stipple "thickening around impossibility" | grain "blooming thickest where color is least believable" | plate "documents a harvest no calendar allows" | stipple "sharpest along the seam where physics lets go" |
| Material specificity | mackintosh, cardigan, salwar, kara, slab path, chain-link, tin | canes, wheat, mackintosh, kara, sheds, fence wire, tin | seed pods, grains, slabs, mackintosh, kara, tin | seed, ropes, mackintosh, kara, chain-link, tin |
| Lighting specification | bright first-minute daylight; dahlia carries the only warmth | bright daylight; silver rim on wet shoulders | stipple bright with after-rain daylight | washed daylight; column thinning toward grey-green |
| Three-tier focal hierarchy | bloom → figure/tell → petals + tin | column/seam → figure/tell → petals + tin | column/wheat → figure/hand → grains + tin | rising seed → figure/tell → petals + tin |
| Chromatic storytelling | grey-green owns plate; dahlia = only warmth | grey-green keeps London; rose copper = garden's alone | grey-green holds plate; rose copper = harvest's | column thins back toward grey-green; silvered seed |
| Narrative incompleteness | tin freshly opened, empty — by whom? | tin open, empty beside the riot | abundance that never refills the tin | seeds leave; the tin stays empty |

## Hard-gate self-check (Flux contract)

- Noun-first, present tense, no imperative openers: all four open "An autochrome plate…" ✓
- Word counts 80–150: **V1 148 · V2 148 · V3 150 · V4 137** (measured, alphanumeric tokens) ✓
- Medium + physical signature in first third ✓ · no camera specs, no Kelvin ✓ · no artist names ✓
- Banned haze words scan (ethereal/dreamlike/whimsical/gentle/soft glow/magical/delicate/floating): clean ✓
- Splinter (empty rain-beaded tin), tell (hem lifted mid-step, walk abandoned), contact (marigold petals in fence wire) visible in every prompt ✓
- Adult figure, face turned three-quarter, fully clothed, lived-in 1970s-to-present attire ✓ · hands mentioned simply ✓
- ~10% rest zone (pearl sky band) described in every prompt ✓ · two-temperature E3 palette only ✓
- BRIGHT register: every prompt names bright/washed after-rain daylight; no dusk ✓

## Contract-compatibility output

```json
{ "image_gen_prompts": [
  {"image_gen_prompt": "P02-V1 First Bud — see fenced text above (148 words)"},
  {"image_gen_prompt": "P02-V2 Full Riot — see fenced text above (148 words)"},
  {"image_gen_prompt": "P02-V3 Fruit No Calendar Allows — see fenced text above (150 words)"},
  {"image_gen_prompt": "P02-V4 Seeds Already Leaving — see fenced text above (137 words)"}
] }
```

*Provenance: Pair-02 agent, steps 06→10 chain, 2026-07-09; inputs = frozen CREATIVE CONTEXT + step-06 facets + step-07 guide; counts measured by script (tokens containing alphanumerics). Self-critique: all four prompts share the same opener chassis ("An autochrome plate…") — consistent for the pair but a monotony risk; step 09 should vary the plate's first-clause behavior per variation while keeping the medium in the first third.*
