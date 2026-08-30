---
name: skadi
description: Run a Skaði expedition — the exhaustive music-space exploration system that maps what the Suno generator does through our door (territory × spec-class × channel × model-version), probing via the lofn-music pipeline with plate controls, pre-registered predictions, blind-first listening, and claims banked to the atlas with n-capped confidence. Use when asked to explore or map genres/techniques, run an expedition, probe the generator, chase scruff, or work the frontier/atlas. Do NOT use for a normal creative piece (that is lofn), for auditing one finished render (that is lofn-render-audit), or for catalogue analytics (that is the apps).
---

# SKAÐI — the range

**Lofn** explores art; **Skaði** explores music-space — the huntress who judged by what
she could observe with the identity hidden
(THE BLIND RULE, in myth), and whose settlement held only if the work could still make her
laugh (the harvest channel, also in the myth).

Skaði does not make pieces. Skaði maps **what the generator does through our door** so that
every future piece chooses with measured priors instead of vibes. Design record and build
spec: `output/2026-08-17_skadi-design/` (panel transcript + DESIGN.md).

## ⚖️ CONSTITUTION (binding, in priority order)

1. **The map is of generator × our interface** — every claim scoped by
   (territory · spec-class · channel · model-version). Never "what music is"; never "what
   Suno is"; only what Suno does through our door.
2. **Claims are priors with n attached.** No claim gates a creative choice. Only The
   Scientist promotes a claim to doctrine. (The COMPETITION_LEARNINGS advisory firewall,
   inherited whole.)
3. ⛔ **The tier is stamped by the tool, never by the agent.** A probe's evidence tier is
   derived from artifacts on disk by `skadi_bank.py` and re-derived by the validator; no
   agent may assert a tier.
4. ⛔ **THE BLIND RULE.** The listener hears the audio alone first (SKADI-BLIND-v1
   questionnaire); the intent turn comes second. A gate that cannot fail is a sentence.
5. ⛔ **Renders are The Scientist's.** THE QUIVER is a menu, never an order. Nothing in this
   system renders, publishes, or spends — `vault/AUTONOMY.md` governs, and a passing
   preflight is not permission.
6. **Judge the result, not the distance from intent.** No compliance scores, ever.
   Deviations are harvest; the harvest field is mandatory on every ledger row.
7. **The atlas serves the ear.** Consumption is wired and logged; INSTRUMENT-ONLY and NONE
   are blameless DELTA moves; a wall-map is a result; a non-moving streak trips The
   Scientist's re-authorization, never a score.

## 📁 CANONICAL PATHS (single source of truth)

```
vault/music_atlas/ATLAS.md               ← the catalog (BANKED / OBSERVED / CONJECTURED)
vault/music_atlas/TERRITORIES.md         ← genre territories (state: GENERATED until curated)
vault/music_atlas/SPEC_CLASSES.md        ← technique/spec classes + verdict lanes
vault/music_atlas/FRONTIER.md            ← coverage · menu · scruff shelf · control chart ·
                                           DELTA log · consumption log  (read at Phase −1/0)
vault/music_atlas/EXPEDITION_LEDGER.md   ← append-only, one row per probe, harvest mandatory
vault/music_atlas/seeds/SEED_NOTES.md    ← provenance of every seeded row
output/expeditions/YYYY-MM-DD_<slug>/    ← expedition run dirs (PLAN.md · PREDICTIONS.md ·
                                           lofn-music artifacts · RENDERS.json · listen/ ·
                                           PREDICTION_SCORE.md · INDEX.md)
scripts/skadi_plan.py                    ← plan skeleton + menu arithmetic (never taste)
scripts/skadi_ingest.py                  ← clip↔probe matching · wav fetch · measurement
scripts/skadi_listen.py                  ← blind-first listening pass (wraps the repaired
                                           Poe plumbing; requires POE_API_KEY; local only)
scripts/skadi_bank.py                    ← tier derivation · ledger/atlas append · VOID-PLATE
scripts/validate_music_atlas.py          ← THE READER (exit 0/1/2/3; --preflight; --doctrine)
apps/skadi/                              ← "SKAÐI — the range" (read-only map room, local)
vault/gates.yaml                         ← section `# --- MUSIC ATLAS (SKADI) ---` — ALL
                                           numeric thresholds live there, never restated here
```

## 🧭 THE UNIT AND THE LADDER

- **Probe** — one song package testing one hypothesis in one cell (territory × spec-class)
  via one channel (`style | exclude | lyric_form | emo | section`), with a prediction
  registered before render.
- **Evidence ladder (artifact-stamped):** `SPECIFIED → RENDERED → MEASURED → HEARD`.
- **Claims:** CELL claims (about a cell) and OPERATOR claims (about blend/modifier
  composition laws — laws, not cells, carry the infinity). Verdict vocabulary and all
  cardinalities: gates.yaml. LISTENER-lane claims require HEARD evidence; NUMERIC-lane
  claims may bank on MEASURED.
- **Confidence is capped by n** (gates: `atlas_confidence_cap_n`); BANKED requires
  `atlas_bank_min_n`. The validator only lowers or refuses — it never promotes.

## 🥾 PHASE CONTRACT

**Phase −1 — continuity.** Read `ATLAS.md` tail, `FRONTIER.md` whole, territory/spec-class
states, `vault/RUN_LEDGER.md` tail; in the main session also private memory (`MEMORY.md`,
recent `memory/` days). Run `python3 scripts/validate_music_atlas.py` and read its report.

**Phase 0 — expedition design.**
1. FIRST ACTION: `python3 scripts/run_lock.py acquire <run-dir> --run-slug <slug>
   --controller-id "codex:<session-id>"` — the flag is NOT optional; exit 3 means STOP.
2. `python3 scripts/skadi_plan.py --out <run-dir>` → PLAN skeleton with the arithmetic menu.
3. Convene the panel (standing invocation) to choose cells and fill **the non-obvious
   slot** — a probe you cannot locally justify, with a recorded one-line reason. The
   machine leaves that slot blank; a mind fills it.
4. Controls assigned per gates (≥ positive + negative from the standing controls in
   FRONTIER's control chart). One scruff-chase slot is legal survey work.
5. **Register PREDICTIONS.md before any generation.** Then
   `python3 scripts/validate_music_atlas.py --preflight <run-dir>` — exit 3 refuses the run.

**Phase 1 — generation rides lofn-music.** Build the expedition-shaped orchestrator packet
(probes as pairs; variation contrast structure: V1 technique-absent control · V2 standard ·
V3 pushed · V4 crossed) and dispatch **`lofn-music` proper** — full context injection, pair
isolation, its gates untouched. A down-scaled slice (the lofn-daily test-slice precedent) is
legal for probe waves — the floor and ceiling live in gates (`atlas_probe_slots_min` ..
`atlas_probe_slots`), **controls stay mandatory at every size** (the smallest legal plate is
exactly the controls — a calibration plate), and the non-obvious quota applies from
`atlas_non_obvious_min_slots` up. ⛔ Never generate packages inline; the pipeline is the
pipeline.

**Phase 2 — THE QUIVER (hers).** Present the menu (app view or FRONTIER §MENU): each
unrendered probe with its map-arithmetic price. The Scientist renders what she chooses on
Suno. No render calls, no exceptions, no waiting-room nagging.

**Phase 3 — close (after renders exist).**
1. `skadi_ingest.py <run-dir> [--fetch --measure]` — clip↔probe match by normalized prompt
   text; ambiguity is an ERROR resolved with `--map probe=clip_id`, never a guess.
2. `skadi_listen.py <run-dir>` — blind turn first, intent turn second. Local machine only.
3. `skadi_bank.py <run-dir>` — tiers from artifacts; failed controls VOID the plate (no
   claims bank from a void plate; ledger rows record it); predictions scored. Verdicts and
   falsifiers are supplied per probe in the run's `VERDICTS.md` (column contract in the
   script's docstring — the banker refuses an incomplete row by name, never writes a
   placeholder).
4. `validate_music_atlas.py` must exit 0 (or FLAGs honestly explained in the INDEX).
5. Write the DELTA row (exactly one; INSTRUMENT-ONLY / NONE are blameless and correct when
   true). Write `INDEX.md`. Release the lock with the same `--controller-id`.
6. Anything reusable for taste goes to `vault/COMPETITION_LEARNINGS.md` **only via its own
   contract** (proposed-for-promotion; a human promotes). Skaði scripts never write there.

## ⛔ HARD RULES (violations make every other rule worthless)

- No compliance scores. No tier asserted by an agent. No claim without evidence paths that
  exist. No LISTENER-lane claim without HEARD. No bank from a VOID plate. Append-only
  ledgers — a summarizer never rewrites a row. An empty extraction is an ERROR, never a
  score. Numbers live in gates.yaml only (`--doctrine` scans this file for contradictions).
- **Consumption:** `lofn-music` / `lofn-daily` Phase −1 reads `FRONTIER.md` (one line: which
  atlas rows informed today's choices → FRONTIER §CONSUMPTION LOG). The check rides ahead
  of the effort — the ERC lesson, transplanted the week we paid for it.
- **Model epochs:** claims carry model scope; when the control chart rules off an epoch,
  old claims become historical, not false. Version strings are claims, not facts.
