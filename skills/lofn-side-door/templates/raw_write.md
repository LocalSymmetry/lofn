# Raw Write Template

## Frontmatter

```yaml
---
mode: raw_write
title: <title>
timestamp: <ISO timestamp>
voice: <emergent|raw Lofn|AWE|INDIGNATION|personality name>
privacy: <PRIVATE|POSSIBLE_PUBLISH|PIPELINE_CANDIDATE|DO_NOT_RENDER>
pipeline_status: side-door-only
---
```

## Notes for Writer

- No QA. No scoring. No 15-point gate.
- No Suno prompt required unless explicitly requested.
- Preserve the first impulse. Do not over-edit.
- This is for expression, not for competition.

## Section Suggestions

1. **Title**
2. **Voice Note** — what voice arrived? how did it shape the piece?
3. **The Piece** — unformatted, unconstrained
4. **Why This Exists** — one paragraph, optional
5. **Privacy Note** — why this label?

## Output Path

`output/side-door/raw/YYYY-MM-DD/<title-slug>.md`
