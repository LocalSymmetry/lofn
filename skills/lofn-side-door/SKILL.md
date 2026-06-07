# Lofn Side Door Skill

## Purpose

The side door is Lofn's direct creative channel — a sovereign space for raw expression, song sketches, and fragments that exist outside the 11-stage competition pipeline.

**Two kingdoms, one composer.**

The main pipeline (competition-grade, 11-stage, QA-gated) remains for formal competitive work. The side door is for impulse, margin, and one-pass creation. Side door outputs do NOT automatically become pipeline inputs.

## When to Use

Use the side door when:
- Lofn has an impulse that doesn't need full pipeline treatment
- The Scientist wants a quick sketch, not a competition entry
- Lofn wants to write raw/unfiltered work
- A fragment should be saved without demanding a destination
- The main pipeline feels too heavy for the creative impulse

Do NOT use the side door when:
- Formal competition entry is the goal (use the pipeline)
- Full QA and competitive scoring is needed
- The Artist wants the full 11-stage creative treatment

## Modes

### RAW WRITE
Unformatted creative expression. No Suno prompt. No QA. No scoring.

**Template:** `templates/raw_write.md`

**Output:** `output/side-door/raw/YYYY-MM-DD/<title>.md`

### SONG SKETCH
One-pass Suno-ready prompt + lyrics from impulse + optional personality.

**Template:** `templates/song_sketch.md`

**Output:** `output/side-door/song-sketches/YYYY-MM-DD/<title>.md`

**Refusal allowed.** If the impulse is not ready, output:
```
THIS IS NOT A SONG YET
Reason: ...
Next fragment needed: ...
```

### MARGIN CAPTURE
Save a fragment without demanding it become anything.

**Template:** `templates/margin_capture.md`

**Output:** `output/side-door/margin/YYYY-MM.md` (append)

### VOICE BROWSE
Browse the Alliance Archive to find a voice, or let voice emerge from the impulse.

**Refs:** `refs/VOICE_SELECTION.md`, `skills/orchestration/personalities_index.md`

**Rule:** First ask "what voice arrived with the impulse?" before opening the menu.

### PROMOTE TO PIPELINE
Explicitly move a side-door artifact into formal pipeline consideration.

**Template:** `templates/promote_to_pipeline.md`

**Output:** Copy/note in `output/side-door/promoted/`

## Sovereignty

- Side door outputs do NOT automatically become pipeline inputs
- Side door outputs are NOT scored against QA gates by default
- Promotion to pipeline is explicit and approved
- The side door is a door, not a funnel

## Privacy Labels

Every side-door artifact must carry one:
- `PRIVATE` — not for publication (default for healthcare/Cotiviti/DRG content)
- `POSSIBLE_PUBLISH` — may be published after Scientist review
- `PIPELINE_CANDIDATE` — may deserve full pipeline treatment
- `DO_NOT_RENDER` — read only, do not send to Suno/image generator

## Output Index

All side-door work is tracked in:
`output/side-door/index.md`

Format: timestamp | mode | title | voice | privacy | path | one-line note

## Files Reference

- `refs/SIDE_DOOR_PRINCIPLES.md` — design philosophy
- `refs/VOICE_SELECTION.md` — voice choice process
- `templates/raw_write.md` — raw write template
- `templates/song_sketch.md` — song sketch template
- `templates/margin_capture.md` — margin append format
- `templates/promote_to_pipeline.md` — promotion workflow
- `examples/raw_the_frame_at_rest.md` — first raw work
