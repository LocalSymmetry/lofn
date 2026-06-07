# Song Sketch Template

## Frontmatter

```yaml
---
mode: song_sketch
title: <title>
timestamp: <ISO timestamp>
voice: <emergent|raw Lofn|AWE|INDIGNATION|personality name>
privacy: <PRIVATE|POSSIBLE_PUBLISH|PIPELINE_CANDIDATE|DO_NOT_RENDER>
render_target: Suno v5.5
pipeline_status: side-door-only
---
```

## Sections

### 1. Voice Note
What voice arrived? Why this voice for this song?

### 2. Music Prompt
- Target: 850-1200 chars (side-door range, less strict than competition)
- Categorized key:value format encouraged but not required
- Genre, mood, tempo, key, vocal, hooks, production, arrangement
- NO real artist names
- NO narrative openings ("Begin by...", "Use...")

### 3. Lyrics
- Free format
- EMO headers optional in side door
- If EMO headers are used, use canonical taxonomy from `skills/music/references/EMOTION_TAXONOMY.md`

### 4. Render Notes
Optional. Any notes for Suno generation.

### 5. Why This Exists
One paragraph. What impulse produced this?

### 6. Refusal (if needed)
If the impulse is not ready for a song:
```
THIS IS NOT A SONG YET
Reason: <specific>
Next fragment needed: <what's missing>
```

## Notes

- ONE PASS. No panel debate. No artist refinement. No QA cycle.
- The contract is "one honest pass," not "one finished product."
- Speed target: one model call, under 60 seconds.
- If something emerges worth competition treatment, use `promote_to_pipeline`.

## Output Path

`output/side-door/song-sketches/YYYY-MM-DD/<title-slug>.md`
