# Lofn Prompt Explorer

A local, file-truthful atlas of the Lofn pipeline. See every agent from the
orchestrator through step 11, see the gate checks that ride the edges between them
and how each validates, and edit any prompt, personality, panel, or supplemental
with a fast review→edit→verify loop.

Built to the design in [`docs/lofn-prompt-explorer-plan.md`](../../docs/lofn-prompt-explorer-plan.md);
ground truth in [`docs/explorer-ground-truth.md`](../../docs/explorer-ground-truth.md).

## Run it

**Windows:**

```powershell
pwsh -File tools/explorer/run.ps1
```

**macOS / Linux:**

```bash
./tools/explorer/run.sh
```

First run creates the venv, installs deps, builds the web UI, and serves everything
from one process at **http://127.0.0.1:8765**. Re-run with `--build` after changing
the frontend. Localhost only — no cloud, no auth, no telemetry.

### Dev mode (hot reload)

```powershell
# terminal 1 — backend
tools/explorer/server/.venv/Scripts/python -m tools.explorer.server
# terminal 2 — vite dev server (proxies /api to :8765)
cd tools/explorer/web ; npm run dev      # http://127.0.0.1:5990
```

## The four views

- **Atlas** — five swimlane rails (orchestrator + music/image/video/story), one hue
  per rail. Gate chips ride each step's outgoing edge, colored by severity
  (red=hard · magenta=mixed · amber=flag · grey=prose). Toggle editions / execution
  layer. Click a node to edit; click a chip to jump to the gate.
- **Editor** (`⌘K` or click a node) — Monaco with the prompt front and center; a
  context panel showing the gates guarding this step, CREATIVE-CONTEXT slot presence,
  and the heading outline; Save (`⌘S`) with conflict detection, Diff vs HEAD, Revert.
- **Gate Center** — thresholds live from `vault/gates.yaml` (grouped by its own
  comment sections), the prose-vs-YAML **drift** dashboard (`--meta-check`), the gate
  map, a runnable validator browser, and raw `gates.yaml` editing.
- **Library** — 8 collections (personalities, panels, aesthetics, genres, film styles,
  3 frame CSVs). Per-item cards (display-only) beside the raw editor; a **sync banner**
  showing canonical-vs-derived drift with one-click **Regenerate derived**; `-1`
  variant twins flagged, never hidden or deleted.

## Architecture

Dumb backend, file-truthful. No database, no app state beyond a config file.

```
tools/explorer/
  pipeline_map.yaml     # THE manifest — the app's one new concept; UI renders from it
  server/               # FastAPI: path-jailed file CRUD, YAML round-trip, git, validators
    files.py            # byte-preserving read/write (no-op save is byte-identical)
    yamlio.py           # ruamel round-trip + can_roundtrip (raw-mode fallback)
    gates.py            # gates.yaml model + meta-check reader
    validators.py       # subprocesses the repo's OWN scripts (no reimplementation)
    libraries.py        # collections, cards, sync-drift, regen
    tests/              # the byte-stability acceptance sweep
  web/                  # Vite + React + TS + Monaco (bundled offline)
scripts/verify_pipeline_map.py   # manifest-vs-tree check (startup + CI)
scripts/regen_library.py         # canonical regen: per-item YAML -> monolith + indexes
```

**Doctrine:** every threshold, drift result, and validation shown in the UI is
produced by the repo's own scripts — the app never restates a number. Every write is
raw bytes echoed from the editor (cards/tables are read-only projections), so a no-op
save is byte-identical; files that can't round-trip cleanly fall back to raw-text mode.

## Acceptance gates (all green)

1. **Byte-stability** — a no-op save is byte-identical for every editable file
   (`server/tests/test_byte_stability.py`, 516 checks incl. mixed-EOL + BOM).
2. **Path jail** — the server writes only under `skills/**`, `vault/**`,
   `.claude/skills/**`, and the manifest; traversal + non-UTF-8 writes refused.
3. **No reimplementation** — thresholds/drift/validation come from the repo scripts.
4. **Derived-file doctrine** — monolith + indexes are regen-only in the UI.
5. **One-command launch on Windows**; localhost only; no db/auth/telemetry.
6. **Manifest honesty** — startup verify fails loudly if the map disagrees with disk.

## Tests

```powershell
tools/explorer/server/.venv/Scripts/python -m pytest tools/explorer/server/tests -q
tools/explorer/server/.venv/Scripts/python scripts/verify_pipeline_map.py
```

## Creative Studio (v2)

The v1 app is a read/edit lens on canon. The **Creative Studio** turns it into an
instrument: edit the **flow** of steps and the gates between them, turn the magic
numbers into **knobs**, press **Run** against a real provider API, watch the
Suno/Flux packages stream out, **compare** two runs (side-by-side or blind), and
**promote** what wins back to canon through the v1 byte-safe write path.

It is additive and separate from canon (the Two-Truths doctrine): canon is prose
read by Claude; the studio is the **engine** interpreting a **FlowSpec**. Experiments
never mutate `skills/**`, `vault/**`, or `.claude/**` — acceptance gate **S1**
sha-sweeps them before and after every run and asserts byte-identity. `vault/gates.yaml`
gains no new key; run-local knobs (e.g. `pair_count_band`) live only in the
per-run `gates.yaml` the engine writes under the run directory.

### The four surfaces

- **Flow Lab** (`g s f`, `/studio/flow`) — the pipeline rail for a modality as an
  editable FlowSpec: insert/remove/reorder/rewire steps and the gates on their edges,
  edit any step's prompt via an overlay (canon untouched), and turn every magic number
  into a **knob** with range + invariants. A zero-cost **Compile** preview updates
  instantly; presets ("classic 6×4", "wide 10×2") snap knob sets.
- **Run Bench** (`g s r`, `/studio/run`) — launch a `flow@version` + knob-preset +
  personality/panel/seed against a provider; watch live per-step artifact streaming
  with gate chips, a running **cost meter**, and the **budget cap**; cancel/resume
  (disk is authority). History lists prior runs with full provenance.
- **Compare** (`g s c`, `/studio/compare`) — pick two runs; diff their resolved
  prompts and final packages, or run a **blind judge** (payload-only, fails-closed).
- **Promote** — write a winning knob/threshold back to `vault/gates.yaml` (byte-safe,
  diff-previewed, meta-checked); structural flow changes emit a **Promotion Brief**
  to apply to canon prose deliberately.

### Launch

Same process, same origin as v1 — a fresh build includes the studio views:

```powershell
pwsh -File tools/explorer/run.ps1 --build     # Windows      -> http://127.0.0.1:8765
```
```bash
./tools/explorer/run.sh --build               # macOS/Linux  -> http://127.0.0.1:8765
```

The studio UI calls `/api/studio/*` on the same origin (the vite dev proxy forwards
`/api` in dev mode). Localhost only — no cloud, no auth, no telemetry.

### Keys — `.env.studio`

Provider keys are read from the process environment first, then from a **gitignored**
`.env.studio` at the repo root (a tiny `KEY=VALUE` dotenv; never served by the file
API, never written to `os.environ`). Keys **never** reach the frontend: `GET
/api/studio/providers` returns only `{provider, configured, model_list?}`, and a
redaction filter guards every text/SSE path (gate **S2**). Accepted names:

| Provider | Env var (legacy alt) |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` (`ANTHROPIC_API`) |
| `openai` | `OPENAI_API_KEY` (`OPENAI_API`) |
| `openrouter` | `OPENROUTER_API_KEY` (`OPEN_ROUTER_API_KEY`) |
| `poe` | `POE_API_KEY` (`POE_API`) |
| `gemini` | `GEMINI_API_KEY` (`GOOGLE_API_KEY`) |
| `mock` | *(keyless — deterministic fixtures; the default for tests/demos)* |

### Budget caps (mandatory)

Every run carries a hard **budget cap in USD**. The engine preflights an estimate and
meters actual usage per call; crossing the cap **cancels remaining calls loudly** and
records `budget_exceeded` in the run log. There is no uncapped run.

### Prompts only — no media rendering

The product is the **prompt package** (Suno-ready style/lyrics, Flux/GPT-Image
prompts, shot lists, prose). The studio makes **no** Suno/Flux/Veo render calls —
render handoff stays manual/external. Media rendering is a non-goal by design.

## Not in v1 (deferred to v1.5, per the plan)

The **Run Inspector** — auditing `output/<run>/` artifacts with `rebuild_manifest.py`
and GATE_REPORT chips. Every backend primitive it needs already exists; it's gated
behind scope, not capability.
