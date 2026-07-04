# Lofn Prompt Explorer — Ground Truth (WS0)

Resolved 2026-07-03 by four parallel repo investigations + an empirical byte-stability
probe. This is the evidence base the manifest (`tools/explorer/pipeline_map.yaml`) and
the backend encode. **The one item needing your confirmation is §1** (per the plan's
risk register: "WS0's canonical-source call must be user-confirmed before WS5 builds on it").

## 1. Canonical library source — CONFIRM ME

**Personalities and panels are edited as PER-ITEM files. The monolith + indexes are DERIVED.**

| Collection | Canonical (edit here) | Derived (regen-only) |
|---|---|---|
| Personalities | `skills/orchestration/personalities/*.yaml` | `personalities.yaml`, `personalities_index.json`, `personalities_index.md` |
| Panels | `skills/orchestration/panels/*.yaml` | `panels.yaml`, `panels_index.json`, `panels_index.md` |

Evidence (what the pipeline actually reads): `skills/orchestration/SKILL.md` and
`.claude/skills/lofn/SKILL.md` both instruct "scan the index, then load
`personalities/<name>.yaml` / `panels/<name>.yaml` for the full DNA"; `steps/02_personality_panel.md`
loads "that personality's full file from `skills/orchestration/personalities/`". The monolith
is a byte-concatenation; the indexes are navigation aids. **No regen script exists** — this
build writes `scripts/regen_library.py` as the canonical regen path.

## 2. Counts are GLOBBED, never hardcoded — the indexes are already drifted

| | canonical files (non-`-1`) | `-1` variant twins | index rows |
|---|---|---|---|
| personalities | 95 | 19 | 113 |
| panels | 158 | 20 | 178 |

The `_index.md` row counts match neither the file counts nor each other's convention
(personalities' index omits variants; panels' index appears to include them). This is
**expected drift** and is exactly what the Library sync-banner surfaces. The `-1.yaml`
twins are orphaned, un-indexed, shorter alternates — the UI **flags them, never hides or
deletes them** (deletion is a human decision outside the app).

## 3. Byte-stability (acceptance gate #1) — empirically proven

Probe: read bytes → detect BOM + dominant EOL → ruamel round-trip (per-file sequence-indent
detection, `width=2^20`, `preserve_quotes`) → re-apply EOL/BOM → compare.

- **gates.yaml (heavy comments): byte-identical.** Comments, key order, and lists survive.
- **250 / 254 canonical library files: byte-identical.**
- **4 files fall back to raw-text mode** (irregular block-scalar indentation or chomping;
  or a strict-parse failure): `lofn-prime-mini.yaml`, `emotion-architects.yaml`,
  `holographic-heritage.yaml`, `masters-of-the-dark-room.yaml`. Raw mode is still fully
  editable and trivially byte-safe. This is gate-1's designed escape hatch.

**Mechanism baked into `server/yamlio.py`:** a load→dump→reload self-check per file; if it
isn't byte-identical, the file is served in raw-text mode (no structured card).

## 4. Per-file format facts that drive the editors

- Per-item personality/panel files are a **1-element YAML sequence** (`- name:` / `  prompt: |`),
  dash at column 0. Nested lists in `gates.yaml` sit at column 2. → per-file indent detection.
- EOL varies per file: personality/panel/gates/CSVs are **CRLF**; `film_styles.txt` and
  `aesthetics.txt` are **LF**. No BOMs observed. Preserve per-file bytes.
- `vault/aesthetics.txt` is a **single comma-separated line** (not newline-delimited) → the
  Aesthetics list editor splits on commas. `genres.txt` (1200) and `film_styles.txt` (681)
  are newline-delimited.
- The three frames CSVs share header `Category,Technique,Description`, comma-delimited,
  **unquoted** (a literal comma in a Description would break naive parsing — the CSV editor
  must use a real csv reader/writer). `music_frames.csv` has a **leading blank line**.
- Panel `prompt` blocks contain `## Special Flairs`, `## Concept Panel`, `## Medium Panel`,
  `## Context & Marketing Panel` (18 voices + 15 flairs) → the panel card parser.
- Step files share the `##` outline: Description · Trigger Conditions · Required Inputs ·
  Creative Context Inputs · Execution Instructions.
- CREATIVE CONTEXT slots (Step Editor presence-check): `{input} {seed} {meta_prompt}
  {personality} {concept_panel} {medium_panel} {marketing_panel} {flairs} {genres_list}
  {frames_list} {image_context}`.

## 5. Two artifact-naming schemes (Run Inspector concern, v1.5)

The Atlas nodes are the STABLE step **prompt files**. The run **artifacts** they emit are
named differently by the two execution layers — Claude-native `.claude/skills/lofn`
(`core_seed.md`, `personality.md`, `orchestrator_metaprompt.md`, `CREATIVE_CONTEXT.md`, …)
vs. the OpenClaw/validator scheme (`02_golden_seed.md`, `03_orchestrator_panel_debate.md`,
`04_orchestrator_metaprompt.md`, `06_audio_handoff.md`, …). v1 visualizes prompt files and
logical edges; reconciling run-artifact names is deferred to the v1.5 Run Inspector.

## 6. Gate topology (drives where chips render)

- **Step-level** (chips on edges): `validate_step.py` (+ `check_human_subjects.py`,
  `validate_pair_artifacts.py`, `validate_with_retries.py`).
- **Run-level** (Gate Center, not edges): `validate_orchestrator_packet.py`,
  `validate_portfolio_distinctiveness.py`, `audit_lofn_pipeline_artifacts.py`,
  `check_and_repair_lofn_run.py`, `rebuild_manifest.py`, `--meta-check`.
- **Orchestration-phase**: `validate_preflight.py`, `validate_spawn_manifest.py`,
  `validate_phase_gate.py`, `validate_checkpoint.py`.
- `vault/gates.yaml` is the single source of numeric thresholds (fail-open); the app reads
  them live and surfaces prose-vs-YAML drift via `--meta-check`.
