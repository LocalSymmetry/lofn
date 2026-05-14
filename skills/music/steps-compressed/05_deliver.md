# Music Pipeline Step 5: Deliver

Before delivery, run the final package gate on every song file:

- [ ] Contains `## 0. GATE CHECK` with prompt char count, song form, EMO count, SFX cue, non-lexical hook, and Lofn-specific move
- [ ] Contains `## 1. MUSIC PROMPT` with standalone copy-paste prompt, 850-1000 chars, no artist names
- [ ] Contains `## 2. LYRICS` with `[SONG FORM: <named form>]`, `[Theme:]`/`[Setting:]`, and EMO-tagged Suno performance headers
- [ ] Contains `## 3. TITLE`
- [ ] Contains `## 4. PRODUCTION NOTES` with concrete production specificity, special events, and short-clip hook note
- [ ] Lyrics are 70-120 sung lines unless explicitly justified
- [ ] Lyrics include at least one standalone SFX cue in asterisks and one non-lexical vocal hook where musically appropriate
- [ ] Prompt is not replaced by metadata, sonic architecture, genre/key/tempo table, or production notes
- [ ] The song passes the Suno 15-point QA delivery core: standalone prompt, prompt density/restraint, performance-ready syntax, hook survivability, personality fidelity, production specificity, anti-slop, package readiness

If any item fails, repair before delivery. Missing `## 1. MUSIC PROMPT` is a blocking failure.

When possible, run the deterministic validator before delivery:

```bash
python3 skills/music/scripts/validate_suno_packages.py <output_dir>
```

Treat any `FAIL` result as a repair blocker.

Send each repaired song file to Telegram:
- channel: telegram
- target: {{TELEGRAM_TARGET}}
- buttons: []

Then send a summary message with all 6 song titles and one-line descriptions.
