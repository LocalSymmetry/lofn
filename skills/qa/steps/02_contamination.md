# QA Step 2: Contamination Scan & Remediation

## Input
- Output directory: `{output_dir}/`
- Completeness audit: `{output_dir}/qa_step1_completeness.md`

## Task

Scan ALL files for contamination and remediate.

### Artist names in music prompts (music prompt section only)
- Common patterns: `like [Artist]`, `in the style of [Artist]`, `[Artist]-esque`
- Replace with style description: "in the style of Arca" → "with fractured club structures and intimate alien vocal processing"

### Leftover scaffolding in lyrics
- Remove: rhyme scheme markers `(A)`, `(B)`, `(C)`, `(AABB)`
- Remove: syllable break markers `syl|la|ble` pipe characters
- Remove: section flow tags `<ABABCC>`
- Remove: editor commentary `[Note: ...]`, `[TODO: ...]`, `[CHECK: ...]`
- Remove: duplicate section blocks

### Template placeholders
- Remove: `{concept}`, `{medium}`, `{input}`, `{aesthetics}`, `[PLACEHOLDER]`, `[INSERT]`, `TBD`, `TODO`

### Suno-breaking patterns (music only)
- Artist names in music prompts (waste characters, Suno ignores them)
- Section labels like `[Intro]`, `[Verse]` INSIDE the music prompt paragraph
- Music prompts over 1000 characters (truncate: preserve emotion→genre→instruments→vocals→progression order)

### Image competition entries
- Remove internal Lofn pipeline tags: `AWE mode`, `INDIGNATION mode`
- Remove model directives like `Flux Pro 1.1 Ultra` from NightCafe prompt fields
- Remove aspect ratio directives if NightCafe has separate selector

### Auto-fix (do immediately for all found issues)
- All of the above

### Flag for rerun (cannot fix automatically)
- Missing song files
- Lyrics under 40 lines
- Music prompt under 400 chars
- Template placeholders still present after remediation

## Save
- Remediation report: `{output_dir}/qa_step2_contamination.md`
