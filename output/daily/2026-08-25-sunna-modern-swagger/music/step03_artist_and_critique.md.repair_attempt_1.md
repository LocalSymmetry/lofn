# Lofn Step Repair Required

Artifact: `output\daily\2026-08-25-sunna-modern-swagger\music\step03_artist_and_critique.md`
Step: `03`
Attempt: 1/3

## Validator failure

```
FAIL: artifact describes what the step would do instead of containing complete step output
```

## Repair instructions

Revise the artifact in place. Preserve valid creative content. Fix only the failing structural/content requirements. Then rerun validate_with_retries.py with the next attempt number and the same --gates path, if supplied.

If this is Step 08 or Step 10 music, verify: standalone `## 1. MUSIC PROMPT`, `## 2. LYRICS`, bracketed `[SONG FORM: ...]`, full `[Section - EMO:... - Voice - Cue]` headers, SFX cue, non-lexical hook, and no bare `[EMO:...]`, `EMO HEADER:`, or plain `SONG FORM:` lines.
