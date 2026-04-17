---
name: image-gen
description: Generate images using FAL (Flux Pro 1.1 Ultra), Gemini (nano-banana 2), or OpenAI. Supports generation, editing, and refinement workflows. Use for any image creation task.
---

# Image Generation Skill

Multi-provider image generation with a preference hierarchy:
1. **FAL Flux Pro 1.1 Ultra** — Primary generator (best quality, 9:16 native)
2. **Gemini nano-banana 2** — Editing/refinement (fix hands, details, consistency)
3. **OpenAI DALL-E 3** — Fallback generator

## Prerequisites

| Provider | Env Var | Status |
|----------|---------|--------|
| FAL | `FAL_KEY` | Required for Flux |
| Gemini | `GEMINI_API_KEY` | Required for editing |
| OpenAI | `OPENAI_API_KEY` | Optional fallback |

## Quick Usage

### Generate with Flux (9:16 portrait)
```bash
node skills/image-gen/scripts/fal-generate.js \
  --prompt "A solarpunk garden city at golden hour" \
  --aspect "9:16" \
  --output ./images/garden.png
```

### Edit with Gemini (nano-banana)
```bash
node skills/image-gen/scripts/gemini-edit.js \
  --image ./images/garden.png \
  --instruction "Fix the hands to have exactly 5 fingers" \
  --output ./images/garden-fixed.png
```

### Full Pipeline (generate → refine)
```bash
node skills/image-gen/scripts/pipeline.js \
  --prompt "A solarpunk garden city at golden hour" \
  --aspect "9:16" \
  --refine "Ensure all human figures have correct anatomy" \
  --output ./images/final.png
```

## Workflow: Lofn Image Pipeline

1. **Concept Phase** (skills/image/00-03): Generate aesthetics, concepts, artist/critique
2. **Prompt Synthesis** (skills/image/04-10): Build final prompt
3. **Generation** (this skill): FAL Flux Pro 1.1 Ultra @ 9:16
4. **Refinement** (this skill): Gemini nano-banana for fixes
5. **Output**: Final image saved to workspace

## FAL Flux Pro 1.1 Ultra

**Endpoint:** `fal-ai/flux-pro/v1.1-ultra`

**Key Parameters:**
- `prompt`: The generation prompt
- `aspect_ratio`: 21:9, 16:9, 4:3, 3:2, 1:1, 2:3, 3:4, **9:16**, 9:21
- `output_format`: jpeg, png
- `safety_tolerance`: 1 (strict) to 6 (permissive)
- `raw`: true for less processed, natural look
- `seed`: for reproducibility

**Cost:** ~$0.06 per image

## Gemini nano-banana 2 (Image Editing)

Uses Gemini's native image generation/editing capabilities.

**Models:**
- `gemini-2.0-flash-exp` — Fast editing
- `gemini-2.0-pro` — Higher quality edits

**Use Cases:**
- Fix anatomical issues (hands, fingers, faces)
- Adjust clothing/accessories
- Color correction
- Add/remove elements
- Style transfer

## Error Handling

- **FAL timeout**: Retry with exponential backoff (max 3 attempts)
- **NSFW block**: Adjust `safety_tolerance` or rephrase prompt
- **Gemini edit fail**: Fallback to regeneration with refined prompt

## Output Structure

All generated images saved with metadata:
```
images/
  {timestamp}_{seed}.png
  {timestamp}_{seed}.json  # metadata: prompt, params, provider
```

## Integration with Lofn Pipeline

This skill is called by `skills/image/08_Generate_Image_Generation.md` after prompt synthesis completes. It can also be invoked directly for quick generations.
