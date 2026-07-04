# Lofn Prompt Explorer — Design & Build Plan

**Status:** ✅ **BUILT & VERIFIED 2026-07-03** (designed via expert panel, then implemented). Lives in [`tools/explorer/`](../tools/explorer/); see its [README](../tools/explorer/README.md). Launch: `pwsh -File tools/explorer/run.ps1` → http://127.0.0.1:8765.
**Builder:** implemented directly (WS0 ground-truth via 4 parallel agents → WS1 backend → WS2–5 frontend → WS7 hardening), verified in-browser + 516 passing tests. The v1.5 Run Inspector (WS6) remains deferred by design.

### Build status by workstream
| WS | Status | Evidence |
|---|---|---|
| WS0 ground truth + manifest | ✅ | `pipeline_map.yaml` verifies clean; `docs/explorer-ground-truth.md`; canonical source = per-item YAML (confirm §10.? if desired) |
| WS1 backend | ✅ | 22 routes; no-op save byte-identical; 409 conflict + 403 jail proven |
| WS2 Atlas | ✅ | 5 rails, 51 nodes, gate chips on outgoing edges, hue-coded, toggles |
| WS3 Step Editor | ✅ | Monaco + gates context + 11 slot presence-checks + prev/next + diff/revert |
| WS4 Gate Center | ✅ | thresholds (9 sections live from gates.yaml), drift dashboard (live meta-check), 13 validators runnable, raw editor |
| WS5 Library | ✅ | 8 collections, cards, sync-drift banner + regen, `-1` variants flagged |
| WS6 Run Inspector | ⏸ deferred (v1.5) | as planned |
| WS7 hardening | ✅ | 516/516 tests incl. whole-surface byte-stability sweep; run.ps1; README |


**One-line pitch:** a local, file-truthful atlas of the Lofn pipeline — see every agent from orchestrator through step 11, see every gate between them and how it validates, and edit any prompt, personality, panel, or supplemental with a 30-second edit→verify loop.

---

## 1. Problem, goals, non-goals

The Lofn pipeline is ~60 prompt files (agents), a deterministic gate layer, and ~300 library items spread across `skills/`, `vault/`, and `.claude/skills/`. Reviewing or editing any of it today means knowing the layout by heart. The user (solo developer / creative director) wants fast review-and-edit cycles.

**Goals (v1):**
- G1. Visualize the full pipeline: orchestration steps 01–06 (Phase 0/1) → dispatch → four modality rails (music 00–11, image/video/story 00–10), with the gate checks drawn **on the edges** between steps.
- G2. Open and edit any step prompt, router `SKILL.md`, `OVERALL_PROMPT_TEMPLATE.md`, `TASK_TEMPLATE.md`, or `.claude/skills/lofn*` file in a real editor, with git-diff awareness.
- G3. Make validation *visible*: which gate guards which step, its thresholds from `vault/gates.yaml`, its severity (hard fail / flag / prose), and live meta-check drift (prose numbers that contradict gates.yaml).
- G4. Edit all supplementals: 114 personalities, 178 panels, aesthetics/genres/film-styles lists, the three frames CSVs — with search, structured card views, and safe round-trip saves.
- G5. Never lie about the files: derived indexes are regenerated, never hand-edited; canonical-vs-derived drift is surfaced, not hidden.

**Non-goals (v1):**
- Running the creative pipeline itself (that stays in `/lofn` skills).
- Rendering (Suno/Flux/Veo), publishing, or any network calls beyond localhost.
- Editing pipeline *outputs* (run artifacts) — deferred to the v1.5 Run Inspector (§5.5).
- Multi-user, auth, database, docker. Single local user, files are the only state.

---

## 2. Ground truth — what the Explorer reads and writes

### 2.1 Agents (prompt files) — the nodes
| Rail | Files | Notes |
|---|---|---|
| Orchestrator | `skills/orchestration/steps/01_lofn_core.md` … `06_metaprompt.md` | Phase 0/1: core → personality/panel → baseline debate → group transform → skeptic synthesis → metaprompt |
| Music | `skills/music/steps/00..11_*.md` | 13 files; **two step-11 variants** (`11_Generate_Music_Enhancement.md`, `11_Generate_Music_GPT55_Enhancement.md`) |
| Image / Video / Story | `skills/<m>/steps/00..10_*.md` | 11 files each |
| Compressed editions | `skills/<m>/steps-compressed/*.md` | Parallel *edition* of the same rail (music: 5 files), not separate nodes |
| Routers & templates | `skills/<m>/SKILL.md`, `OVERALL_PROMPT_TEMPLATE.md`, `TASK_TEMPLATE.md`, `COMPRESSED_PIPELINE.md` | The CREATIVE CONTEXT carrier lives in the template |
| Execution layer | `.claude/skills/lofn*/SKILL.md`, `.claude/skills/lofn/EXECUTION.md` | Claude-native port; EXECUTION.md §4 is the authoritative gate *prose* |

Step files share a structure the editor exploits: `## Description`, `## Trigger Conditions`, `## Required Inputs`, `## Creative Context Inputs`, `## Execution Instructions` (+ output-format sections).

### 2.2 Gates — the edges
| Artifact | Role |
|---|---|
| `vault/gates.yaml` | **Single source of numeric thresholds** (fail-open contract; comments are load-bearing documentation) |
| `scripts/validate_step.py` | Deterministic per-step validator; emits `GATE_REPORT.json` rows; `--meta-check` scans skill files for restated numbers that disagree with gates.yaml |
| `scripts/validate_orchestrator_packet.py`, `validate_pair_artifacts.py`, `validate_portfolio_distinctiveness.py`, `validate_with_retries.py`, `check_and_repair_lofn_run.py`, `audit_lofn_pipeline_artifacts.py`, `rebuild_manifest.py`, `check_human_subjects.py` | Run-level validators/auditors (Run Inspector surface) |
| `skills/orchestration/scripts/validate_checkpoint.py`, `validate_phase_gate.py`, `validate_preflight.py`, `validate_spawn_manifest.py` | Orchestration-phase gates |
| `.claude/skills/lofn/EXECUTION.md` §4 | Prose gate authority quoted in the UI next to the YAML numbers |
| `skills/qa/`, `.claude/skills/lofn-qa/` | Taste-layer QA (linked, browsable, not executed by the app) |

### 2.3 Libraries (supplementals)
| Collection | Canonical? (WS0 resolves) | Derived |
|---|---|---|
| Personalities (114) | `skills/orchestration/personalities/*.yaml` **or** monolith `personalities.yaml` — *three representations exist; WS0 must determine which is read by the pipeline and declare it canonical* | `personalities_index.json`, `personalities_index.md`, the non-canonical form |
| Panels (178) | same question: `panels/*.yaml` vs `panels.yaml` | `panels_index.json`, `panels_index.md`; note `-1` duplicate files (e.g. `acid-n-jazz-house-1.yaml`) — surface as dupes, never auto-delete |
| Aesthetics / Genres / Film styles | `vault/aesthetics.txt`, `vault/genres.txt`, `vault/film_styles.txt` | — (plain lists) |
| Frames | `vault/frames.csv`, `vault/music_frames.csv`, `vault/video_frames.csv` | — (CSV tables) |
| References | `skills/<m>/references/`, `skills/orchestration/refs/` | browsable/editable as plain files |

Library item format: `{name, prompt}` where `prompt` is a large markdown block scalar (panels embed `## Special Flairs`, `## Concept Panel`, `## Medium Panel`, … sections). **Raw text is the edit surface; parsed sections are display-only.**

---

## 3. Design decisions (with panel rationale)

1. **Files are the only truth; the app is a lens.** No database, no app-side state beyond a config file. Every save is an atomic in-place write designed to produce a clean, minimal git diff. (Kleppmann seat; unanimous.)
2. **A declarative manifest is the app's ONE new concept.** `tools/explorer/pipeline_map.yaml` declares nodes, edges, gates, editions, and libraries with their canonical/derived paths and regen commands. The UI renders *from the manifest*; nothing about the repo layout is hardcoded in app code. `scripts/verify_pipeline_map.py` checks the manifest against the real tree (run at app startup + CI-able) so the map can never silently rot. (Brooks seat's conceptual-integrity demand.)
3. **Stack: FastAPI backend + Vite/React/TypeScript frontend + Monaco editor, single-command launch, localhost only.** Streamlit was seriously debated (Python-native, precedent in `origin/streamlit`) and rejected: its rerun model fights a multi-pane editor, and 200KB YAML block scalars + a clickable pipeline atlas exceed what it does well. The Python advantage is kept by putting all validator/YAML logic in the FastAPI layer, which **imports or subprocesses the existing repo scripts rather than reimplementing them**. (Resolution of the Victor-vs-Huyen stack fight.)
4. **Dumb backend doctrine.** The backend does file CRUD (path-jailed), YAML round-trip, subprocess running of existing validators, index regeneration, and git status/diff. Zero domain logic is reimplemented: thresholds come from gates.yaml via the *existing* loader, drift rows come from `validate_step.py --meta-check` verbatim. If the app and CLI ever disagree, it's a bug by definition. (Huyen + Brooks.)
5. **Raw text is always the edit surface.** Structured views (gate tables, personality cards, frames grids) are projections; editing giant prompt blocks happens in Monaco with markdown highlighting. Exception: plain lists (txt) and CSVs get structured editors, and `gates.yaml` gets a two-mode editor (table + raw) since it's genuinely tabular. (Prevents destructive parse-reserialize of creative prose.)
6. **YAML round-trip via `ruamel.yaml` with a byte-stability acceptance gate.** A no-op save of any file must be byte-identical (comments, key order, block style, line endings preserved). Files that can't round-trip cleanly automatically fall back to raw-text editing mode. (Kleppmann; non-waivable, §8.)
7. **The Atlas is a swimlane rail, not a force-directed graph.** The pipeline order is known and linear-with-fanout; draw five horizontal rails (orchestrator + four modalities) with gate chips sitting on the edges. Editions (compressed, GPT55 step-11, `.claude` execution layer) render as toggleable layers on the same nodes. (Tufte seat killed the force-graph.)
8. **Derived files are regenerated, never edited.** Index files show a "derived" badge; the only affordance is *Regenerate*. WS0 locates the existing index generator or, if none exists, the fleet writes one and it becomes the canonical regen path (also usable by the pipeline itself). Canonical-vs-derived drift shows as a sync banner with a diff.
9. **Git stays in the user's hands.** The app shows per-file dirty state, inline diff vs HEAD, and offers *Revert file* — no commit/push UI in v1. (Catmull's "don't bureaucratize the studio" + Brooks scope cut.)
10. **Edit→verify loop under 30 seconds.** Saving a skill file triggers a debounced, scoped meta-check; the affected gate chips update in place. This is the "immediate connection" the tool exists for. (Victor + Catmull convergence.)

---

## 4. Architecture

```
tools/explorer/
  pipeline_map.yaml          # THE manifest (§6.1)
  server/                    # FastAPI app (Python ≥3.11, uvicorn)
    main.py                  # app factory, startup manifest verification
    manifest.py              # load/verify pipeline_map.yaml
    files.py                 # path-jailed read/write, atomic saves, encoding/EOL preservation
    yamlio.py                # ruamel round-trip helpers + byte-stability self-test
    libraries.py             # collection listing/search, canonical-sync check, index regen
    gates.py                 # gates.yaml model, meta-check runner + parser
    validators.py            # subprocess runner for scripts/* (timeout, structured capture)
    gitio.py                 # status / diff / revert via git CLI
  web/                       # Vite + React + TS + Monaco; built assets served by FastAPI
scripts/verify_pipeline_map.py   # manifest-vs-tree checker (startup + CI)
```

- **Launch:** `python -m tools.explorer` → starts uvicorn on `127.0.0.1:<port>`, opens the browser. Windows-first (the daily driver), POSIX-compatible.
- **Write safety:** allowlist = `skills/**`, `vault/**`, `.claude/skills/**`, `tools/explorer/pipeline_map.yaml`. Everything else read-only. `output/**` read-only in v1 (writable only inside the v1.5 Run Inspector's repair mode). Writes are temp-file + atomic replace, UTF-8, per-file EOL style preserved byte-wise.
- **No network, no auth, no telemetry.**

---

## 5. The five views

### 5.1 Atlas (home)
Five swimlane rails: **Orchestrator** (01–06, plus Phase-0/1 artifact slots: golden seed → panel debate → metaprompt → pair assignments → handoff/ICB) fanning out to **Music / Image / Video / Story** rails. Per node: title, file badge(s), dirty-dot if git-modified. Per edge: **gate chips** colored by severity (red = hard fail, amber = flag, grey = prose-only), sourced from the manifest. Toggleable layers: `steps-compressed` edition, step-11 GPT55 variant, `.claude/skills` execution layer, QA/taste layer (link-out). Click node → Step Editor. Click chip → Gate Center, filtered. Search-anything box (nodes, gates, library items) with keyboard palette.

### 5.2 Step Editor
Three panes. **Left:** outline built from the file's `##` headings + rail navigation (prev/next step). **Center:** Monaco, markdown, the prompt is the star. **Right, collapsible:** (a) *Creative Context Inputs* — the threaded slots (`{input} {seed} {meta_prompt} {personality} {concept_panel} {medium_panel} {marketing_panel} {flairs}` + lists) with presence-check against the file; (b) *Gates guarding this step* — thresholds live from gates.yaml; (c) *Meta-check for this file* — restated numbers vs gates.yaml, updated on save; (d) *Diff vs HEAD* toggle + Revert. Same editor serves routers, templates, EXECUTION.md, references, and `.claude` skill files.

### 5.3 Gate Center
- **Thresholds:** `gates.yaml` in a two-mode editor (structured table grouped by the file's own comment sections ↔ raw Monaco). Comments always survive (round-trip gate).
- **Drift dashboard:** every `--meta-check` warning as a row — file, line, restated value, gates.yaml value, jump-to-editor link. One-click re-run.
- **Gate map:** the manifest's edge→gate table: which validator, which gates.yaml keys, severity, and which EXECUTION.md §4 clause quotes it.
- **Validator browser:** the scripts in §2.2 with their docstrings and CLI usage, read-only, jump-to-source.

### 5.4 Library
Collection browser for personalities (114), panels (178), aesthetics, genres, film styles, frames ×3.
- List + full-text search + name filter; counts always visible.
- **Item view:** parsed card (panels: Flairs / Concept Panel / Medium Panel / Marketing sections; personalities: prompt outline) — display-only — plus the raw Monaco editor as the true edit surface.
- **Sync banner:** canonical vs monolith vs index drift per item (content hash compare); *Regenerate derived* button runs the canonical regen script. Dupe detector surfaces `-1` twins side-by-side with a diff (user decides; no auto-delete).
- Lists (txt): line editor with dedupe + count. Frames (csv): table editor preserving column order/quoting.
- New-item flow: create from blank or *duplicate existing* (the realistic authoring path), writes canonical form + regenerates derived.

### 5.5 Run Inspector — **v1.5, behind a flag**
Pick an `output/<run>/`; show `rebuild_manifest.py` state (disk is authority), GATE_REPORT.json chips per pair/step, re-run any validator against artifacts, open artifacts in the editor (repair mode makes `output/**` writable). Deliberately deferred so v1 ships the review/edit core first; every backend primitive it needs (validator runner, file service) already exists by then.

---

## 6. Contracts (frozen after review)

### 6.1 `pipeline_map.yaml` schema (abridged)
```yaml
version: 1
rails:
  - id: music
    title: Music
    nodes:
      - id: music.07
        title: Song Guides
        path: skills/music/steps/07_Generate_Music_Song_Guides.md
        editions:
          compressed: skills/music/steps-compressed/03_song_guides.md
        variants: []          # e.g. music.11 lists the GPT55 file here
edges:
  - from: music.07
    to: music.08
    artifact: song_guides
    gates: [g.sung_lines, g.music_prompt_chars]
gates:
  - id: g.music_prompt_chars
    keys: [music_prompt_chars, music_prompt_chars_target, music_prompt_hug_ceiling]
    validator: scripts/validate_step.py
    severity: mixed           # hard | flag | prose | mixed
    prose: ".claude/skills/lofn/EXECUTION.md#§4"
libraries:
  - id: personalities
    kind: yaml-collection
    canonical: skills/orchestration/personalities/   # WS0 confirms
    derived: [skills/orchestration/personalities.yaml,
              skills/orchestration/personalities_index.json,
              skills/orchestration/personalities_index.md]
    regen: <located-or-written-by-WS0>
```

### 6.2 API surface (backend, JSON; OpenAPI is the source of truth)
`GET /api/manifest` · `GET/PUT /api/file?path=` (PUT returns new git status + byte-stability report) · `GET /api/git/status|diff?path=` · `POST /api/git/revert` · `GET /api/libraries/:id` (+ `?q=` search) · `GET/PUT /api/libraries/:id/items/:name` · `POST /api/libraries/:id/regen` · `GET /api/gates` · `POST /api/gates/metacheck` (scoped by path optional) · `POST /api/validators/:name/run` · v1.5: `GET /api/runs`, `POST /api/runs/:id/manifest`, `GET /api/runs/:id/gatereport`.

### 6.3 Fleet coding doctrine
Reuse repo scripts by import/subprocess — **never** copy a threshold or re-derive a check in app code. TypeScript types generated from OpenAPI. No new dependencies beyond: fastapi, uvicorn, ruamel.yaml, watchfiles (optional live-reload) / react, vite, monaco-editor, zustand or equivalent-lightweight state.

---

## 7. Build plan — workstreams for the Opus fleet

Dependencies: **WS0 → WS1 → {WS2, WS3, WS4, WS5 in parallel} → WS7**; WS6 optional after WS1.

| WS | Deliverable | Key tasks | Done when |
|---|---|---|---|
| **WS0 — Ground truth & manifest** (blocking) | `pipeline_map.yaml` + `scripts/verify_pipeline_map.py` + `docs/explorer-ground-truth.md` | Scan the real tree; enumerate every node/edge/gate/library; **resolve the canonical-source question** for personalities & panels (which representation does the pipeline actually read — check `skills/orchestration/SKILL.md`, steps 02, and `.claude/skills` — and does an index/monolith regen script exist, e.g. under `skills/orchestration/scripts/` or repo `scripts/`? If none: spec one); map each gates.yaml key → validator → edge → EXECUTION.md §4 clause; catalogue the `-1` dupes | verify script passes; a human can answer "what guards step X?" from the manifest alone |
| **WS1 — Backend core** (blocking) | FastAPI app, OpenAPI frozen | files.py (path jail, atomic write, EOL/encoding preservation), yamlio.py (round-trip + self-test harness across ALL repo YAML), gitio.py, validators.py subprocess runner, manifest.py + startup verification | no-op PUT is byte-identical for **every** file under skills/ vault/ .claude/skills/; OpenAPI published; unit tests green on Windows |
| **WS2 — Shell & Atlas** | App shell, routing, command palette, Atlas view | swimlane renderer from manifest, gate chips, edition layers, dirty-dots via git status polling | click-through from every node/chip to the right view; Atlas cold-load < 2s |
| **WS3 — Step Editor** | Editor view | Monaco integration, outline pane, context panel (slots presence, gates, per-file meta-check, diff/revert), debounced save-verify loop | edit→save→chip-refresh < 5s on a step file; slot presence-check matches `OVERALL_PROMPT_TEMPLATE.md` |
| **WS4 — Gate Center** | Gates view | gates.yaml table↔raw editor, drift dashboard wired to `--meta-check`, gate map table, validator browser | drift rows byte-equal to CLI output; comment-preserving save proven on gates.yaml |
| **WS5 — Library** | Library view | collection endpoints + search, card parsers (display-only), raw editors, sync banners + regen, dupe detector, txt/csv editors, new-item/duplicate flow | all 114+178 items open/search/edit; regen produces indexes identical to the canonical generator's output; no-op saves byte-stable |
| **WS6 — Run Inspector** (v1.5, flagged) | Runs view | run listing, rebuild_manifest integration, GATE_REPORT chips, validator re-run, repair-mode writes | can audit `output/daily/2026-07-02/` end-to-end |
| **WS7 — Hardening & acceptance** | Test suite + release | byte-stability sweep as a pytest over the whole repo, Windows CRLF/UTF-8 property tests, Playwright smoke (open→edit→save→revert each view), perf pass, `README` + launch script | §8 gates all green; user runs it with one command |

Suggested fleet shape: WS0 one agent (highest-judgment task); WS1 one agent; WS2–WS5 four parallel agents against the frozen OpenAPI + manifest; WS7 an adversarial agent that did not build the thing it tests.

## 8. Acceptance gates (non-waivable)

1. **Byte-stability:** no-op save byte-identical for every editable file (comments, key order, EOL, encoding). Files failing round-trip are auto-locked to raw-text mode — never silently rewritten.
2. **Path jail:** the server cannot write outside the §4 allowlist (test with traversal attempts).
3. **No reimplementation:** every threshold/drift/validation result shown in the UI is produced by the existing repo scripts; grep-level audit confirms no duplicated constants (the meta-check ethos applied to the app itself).
4. **Derived-file doctrine:** indexes/monolith non-canonical forms are regen-only in the UI.
5. **Single-command launch on Windows**; no docker/db/auth/telemetry; localhost binding only.
6. **Manifest honesty:** startup verify fails loudly (banner) if pipeline_map.yaml disagrees with the tree.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| ruamel can't round-trip an odd YAML byte-identically | WS1 self-test harness sweeps the repo up front; failures → raw-mode lock (gate 1) |
| Canonical-source call in WS0 is wrong | WS0 must cite the pipeline files that *read* each representation; user confirms in review of WS0 output before WS5 builds on it |
| Monaco perf on 200KB block scalars | per-item extraction for library editing (never load the whole monolith into one editor) |
| Concurrent pipeline run mutates files while the app is open | git-status polling + file-mtime conflict check on save (reject stale writes with a diff prompt) |
| Mixed CRLF/LF across the repo | preserve per-file EOL bytes; property tests in WS7 |
| The `-1` panel dupes get "cleaned up" by an eager agent | explicitly read-only surfaced; deletion is a user decision outside the app |

## 10. Open decisions for user review

1. **Stack confirmation** — FastAPI + React/Monaco (panel's pick) vs Streamlit (faster to build, worse daily driver). §3.3 records the rationale.
2. **v1 scope** — ship WS2–WS5 first and hold WS6 (Run Inspector) for v1.5, or include it in the first fleet pass?
3. **Git actions** — v1 has status/diff/revert only. Add a commit button, or does git stay in the terminal?
4. **Placement** — `tools/explorer/` in-repo (assumed) vs a sibling repo.
5. **Editable surface** — `.claude/skills/**` included in the edit allowlist (assumed yes); anything you want locked read-only (e.g. `vault/COMPETITION_LEARNINGS.md`)?
