# Suno v6: lyrics length and packaging

- Aim for **370–430 sung words** (about 400) for a roughly **3:30** song. Line count is a form choice. Text counts cannot guarantee duration: arrangement, phrasing, and generated takes vary.
- New runs default to the word gate in [`vault/gates.yaml`](../vault/gates.yaml). The default advisory band is 340–460 words, and the default hard band is 300–520. Declare a deliberate form exception in the run-local gates before drafting. Existing run-local gates that specify a line rule, or explicitly use `length_gate: lines`, retain legacy line-mode validation.
- Keep separate style and exclude fields. Put `[Disc_*]` and other non-pasted production notes in a `## Production Sidecar`. Keep the pasteable lyrics box under 5,000 characters (soft ceiling 3,800); count headers and cues in the box but not as sung words.
- A validation pass establishes package shape, not render length, creative quality, or readiness to publish. For an actual timing test, record all generated takes including rejects and early stops.

From the repository root, run the synthetic regression tests:

```sh
python -m unittest scripts.test_check_brief_line_budget scripts.test_validate_step_music_cardinality scripts.test_validate_suno_packages
```

To check a package:

```sh
python skills/music/scripts/validate_suno_packages.py path/to/package.md
```
