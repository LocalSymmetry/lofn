# Pipeline Repair — Sunna R4.1 Completion Band

## Finding

Sunna R4.1 explicitly defines a complete song as **36–56 sung lines** and
forbids padding to satisfy Lofn's generic long-form exam. The generation and
package validators nevertheless hard-coded `<60` as a repair trigger. That
made the personality correction non-executable: a compliant Sunna song would
be rejected and expanded until it resembled the older run.

The prior Sunna R4 standalone packages measured 70, 74, 74, 70, 73, and 72
probable sung lines. This is evidence of the pressure the stale gate applied;
it is not a quality verdict on those songs.

## Repair

- `scripts/validate_step.py` now reads a configurable
  `sung_lines_hard_floor` and optional `sung_lines_hard_ceiling` from the
  selected gates file.
- `scripts/validate_with_retries.py` now accepts `--gates PATH` and passes it
  through to the canonical validator.
- `skills/music/scripts/validate_suno_packages.py` now accepts optional
  `--min-sung-lines` and `--max-sung-lines` arguments.
- `vault/gates.yaml` records the unchanged canonical hard floor of 60.
- This run's `gates_sunna_r41.yaml` alone enforces a hard 36–56 band.

Canonical Lofn behavior is unchanged unless a controller deliberately supplies
a run-local gate file.

## Verification

- Python compilation: PASS for all three modified validators.
- Skill mirror: PASS, 28 compared and in sync.
- Canonical-package default path: PASS on the prior standalone package.
- Run-local ceiling negative control: the prior 76-line package fails at
  `>56`, as expected.
- Run-local Step validator negative control: the prior 74-line Step 10 package
  fails at `>56`, as expected.

## Additional reader repairs

- The standalone package splitter recognized `# V1` but not the frozen run's
  zero-padded `# V01` convention. Its direct-V pattern now accepts both, and it
  proves four extracted packages instead of inspecting one or refusing the
  file.
- Structural PASS still allowed score language to become sung lyric (for
  example, “the guitar hits” and “the organ answers”) and allowed the old
  corrective thesis to return as “don't fix me.” The run-local
  `audit_sunna_r41_surface.py` reads only sung lines and flags production
  notation, corrective abstraction, hedge-analysis, body-noise leakage,
  missing direct address, and missing future coordinates.
- Wave-one Step 08 initially triggered those findings. Root-routed repairs
  moved production back into headers/style fields and replaced explanations
  with literal object, time, reply, and action language.
- A root close-read then found a second class that structural readers still
  cannot settle: exact choruses that ignored the adult's counter-choice, and a
  flirt surface that exposed AI incapacity or announced an anti-speech rule.
  Pair 04 was repaired so the 8:10 answer returns as “just past eight,” the
  accepted pink ribbon returns through a color-neutral hook, and the rooftop
  song stays wholly inside the affirmative lime-and-red-chair event.
- All 24 Step-08 songs now pass the canonical validator, the 36–56 package
  validator, the run-local surface audit, and root lyric-level review.

The new Step 08, Step 10, and standalone-package gates must all use the run-
local band. No agent may repair a Sunna song by padding it past 56 lines.
