# Lofn Creative Studio — Design & Build Plan (v2 of the Prompt Explorer)

**Status:** ✅ **APPROVED 2026-07-03; Opus fleet build in progress.** Open decisions resolved in §13. Legacy provider integrations vendored for the fleet at [`docs/reference/lofn-legacy-llm/`](reference/lofn-legacy-llm/) (the streamlit-era `llm_integration.py` is the engine ancestor).
**Builds on:** the shipped v1 Explorer ([`tools/explorer/`](../tools/explorer/), [design](lofn-prompt-explorer-plan.md), [ground truth](explorer-ground-truth.md)). v1's non-waivable gates (byte-stability, path jail, dumb backend, manifest honesty) remain in force; this plan only *adds*.
**One-line pitch:** turn the read/edit lens into an instrument — edit the **flow** of steps and the gates between them, turn the **magic numbers** into knobs, press **Run** against a real API (Anthropic / OpenAI / OpenRouter / Poe / Gemini), watch the final Suno/Flux packages come out, compare, and promote what wins back to canon.

---

## 0. The Two-Truths Doctrine (the load-bearing decision)

There are **two executors** of the Lofn pipeline and this plan keeps them separate on purpose:

| | **Canon** (production) | **Studio** (experiment bench) |
|---|---|---|
| What runs it | Claude reading prose (`/lofn` skills, `EXECUTION.md`, `skills/**/steps/*.md`) | The studio **engine** interpreting a **FlowSpec** via provider APIs |
| Source of truth | `skills/**` + `vault/gates.yaml` + `.claude/skills/**` | `tools/explorer/flows/*.flow.yaml` + `knobs` + `overlays/` |
| Mutability | **Never mutated by experiments** (acceptance gate S1) | Freely editable; versioned; copy-on-write |
| How they connect | **Import**: a compiler reads canon → generates the baseline FlowSpec | **Promote**: an explicit, diff-previewed write-back (knobs/thresholds automatic; structure via a generated brief) |

Why: `pipeline_map.yaml` is *descriptive* — editing it changes nothing in production, because canon's executor is prose read by Claude, not the map. Making the map executable would make it lie about the repo (violating v1 gate 6). So the studio gets its **own executable document** (FlowSpec), imported *from* canon, promoted *back* explicitly. The v1 manifest keeps its job (describing canon); the Atlas gains a version switcher: **canon** (from pipeline_map) vs any **flow@version**.

Heritage: this restores what `origin/streamlit` did three years ago — `lofn/llm_integration.py` running `{slot}`-template chains against provider APIs — with v1's provenance, gates, and byte-discipline layered on. WS0 studies that fossil for chain-fidelity notes; it is a reference, never a dependency.

---

## 1. Goals / non-goals

**Goals (v2.0):**
- G1. **Flow editing** — insert/remove/reorder/rewire steps and the gates between them, per modality, as versioned FlowSpecs; edit any step's prompt via overlays without touching canon.
- G2. **Knobs** — every magic number (§3 census) becomes a parameter with range, derivation, and invariants; presets ("classic 6×4", "wide 10×2"); changing a knob updates a zero-cost **Compile** preview instantly.
- G3. **Execution** — run any flow@version + knob-preset + personality/panel/seed against Anthropic, OpenAI, OpenRouter, Poe, or Gemini; live per-step artifact streaming with gate chips; per-step and total **cost tracking** with a hard budget cap; cancel/resume (disk is authority); the same deterministic gates canon uses (`validate_step.py` et al.) guard every edge.
- G4. **Experiments** — full provenance per run; side-by-side compare of two runs; **blind judging** (payload-only, fails-closed extraction per `build_blind_set.py` doctrine); verdicts recorded.
- G5. **Promotion** — winning knob/threshold values write back to `vault/gates.yaml`/prose via the v1 byte-safe path with diff preview + meta-check; structural flow changes emit a **Promotion Brief** for deliberate application to canon prose.

**Non-goals (v2.0):**
- Replacing the `/lofn` Claude-skill path (the studio is the lab, not the factory).
- Rendering media (no Suno/Flux/Veo API calls — prompts are the product; render handoff stays manual/external).
- Arbitrary DAGs (Lofn's shape is linear-with-one-fanout; see §4).
- Multi-user, cloud, telemetry. LLM-taste-QA as a required stage (available as an *optional experimental* flow step running the lofn-qa prompts).

---

## 2. Architecture

```
tools/explorer/
  pipeline_map.yaml            # v1 — still descriptive-of-canon only
  flows/                       # NEW — studio-owned, versioned, git-tracked
    lofn-music@1.flow.yaml     #   baseline imported from canon (immutable once run)
    lofn-music@2-wide.flow.yaml
    presets/classic-6x4.knobs.yaml
    overlays/music-05-ten-pairs.patch.md
  server/
    studio/                    # NEW backend package (same dumb-backend doctrine)
      flowspec.py              #   load/validate/version FlowSpecs; canon importer
      knobs.py                 #   knob schema, derivation, invariants, presets
      resolver.py              #   THE single prompt-resolution path (compile == execute)
      providers.py             #   ProviderClient: anthropic | openai-compat | mock
      pricing.py               #   pricing.yaml loader; estimates; budget meter
      engine.py                #   async run executor: state machine, fanout, gates, retry
      runstore.py              #   output/studio/<run-id>/ + run.json + runlog.jsonl
      judge.py                 #   compare + blind-judge (build_blind_set extraction rules)
      promote.py               #   knob/threshold write-back + Promotion Brief generator
  web/src/views/               # NEW views: FlowLab, Knobs, RunBench, Compare
output/studio/<run-id>/        # runs (artifacts mirror canonical naming) — engine-owned
.env.studio                    # API keys — gitignored, chmod'd, never served/logged
```

- **Path jail extension:** `write_allowlist` += `tools/explorer/flows/**`; the **engine** (not the file API) owns `output/studio/**`. `.env.studio` is readable by the server process only — never served by any endpoint, never written by the file API.
- **Dumb-backend doctrine extended:** the engine never reimplements a gate — it shells out to the same `scripts/validate_*.py` canon uses. The single new canon-side change: expose `load_gates(path)`'s existing parameter as a `--gates PATH` CLI flag on `validate_step.py` (backward-compatible, fail-open unchanged) so runs can use knob-adjusted thresholds; plus teach `--meta-check` to skip `tools/explorer/flows/**` (knob files legitimately restate numbers).

---

## 3. Knobs — the census and the schema

### 3.1 Census (grounded; WS0 completes it with file:line for every entry)

| Knob | Default | Where the number lives today |
|---|---|---|
| `n_pairs` | 6 | `.claude/skills/lofn/SKILL.md` ("6 pair assignments", per-pair invariant), `lofn-music/SKILL.md` ("**exactly 6** — only the Scientist downsizes"), `EXECUTION.md` ("one wave of 6 pairs"), step 05 prose |
| `n_variations_per_pair` | 4 | SKILL.md ("6 pairs × 4 variations = 24"), `gates.yaml total_prompts: 24` |
| `barbell_accessible` / `ambitious` | 3 / 3 | SKILL.md ("3 ACCESSIBLE + 3 AMBITIOUS"), `lofn-daily` ("pairs 1–3 / 4–6"; "best 3 from each arm") |
| `n_concepts` (step 02) | 12 | modality SKILL.md step tables ("(12 concepts)") |
| `panel_count` × `voices_per_panel` | 3 × 6 = 18 | SKILL.md ("THREE panels of 6 voices = 18"), OVERALL_PROMPT_TEMPLATE |
| `n_flairs` | 15 | SKILL.md, template |
| `taxonomy_cardinality` | 50 | `gates.yaml` |
| `music_prompt_chars` / target / hug | [850,1000] / [870,960] / 985 | `gates.yaml` |
| `sung_lines` / target / floor-hug | [70,120] / [78,110] / 72 | `gates.yaml` |
| `suno_lyrics_field_max` / target | 5000 / 4800 | `gates.yaml` |
| `image_min_words` | 80 | `gates.yaml` |
| `concept_warmup_discard` | 17 (concepts), 14 (arrangements) | OVERALL_PROMPT_TEMPLATE ("generate 17, discard, start at 18") |
| `daily_funnel` | 24 → 12 → 6 | `lofn-daily/SKILL.md` |
| `max_concurrency` | 6 (12 → cap-and-stagger) | `EXECUTION.md` §concurrency, `lofn-daily` |
| `retry_max_attempts` | 3 | `validate_with_retries.py` |

### 3.2 Schema (`*.knobs.yaml`)

```yaml
knobs:
  n_pairs:            {value: 6, range: [2, 12], desc: "concept×medium pairs selected at step 05"}
  n_variations_per_pair: {value: 4, range: [1, 8]}
  n_concepts:         {value: 12, min_expr: "2 * n_pairs", desc: "step-02 divergence pool"}
  barbell_accessible: {expr: "ceil(n_pairs / 2)", override: null}
  barbell_ambitious:  {expr: "n_pairs - barbell_accessible"}
  total_prompts:      {expr: "n_pairs * n_variations_per_pair", gates_key: total_prompts}
  music_prompt_chars: {value: [850, 1000], gates_key: music_prompt_chars}
  # ... every gates.yaml numeric key gets a gates_key binding
invariants:
  - "n_concepts >= 2 * n_pairs"
  - "barbell_accessible + barbell_ambitious == n_pairs"
  - "sung_lines[0] < sung_lines_target[0] < sung_lines_target[1] < sung_lines[1]"
```

- `expr` values are evaluated by a **tiny safe evaluator** (int arithmetic + ceil/floor/min/max only — no eval()).
- Invariant violations render inline in the Knobs UI and **block Run** (not Compile — you can preview a broken preset, you can't spend money on one).

### 3.3 How knob values reach the model (canon untouched)

1. **RUN PARAMETERS block** — the resolver appends a studio-owned section to the CREATIVE CONTEXT injection: an explicit, authoritative statement of the run's numbers ("Select **exactly {n_pairs}** pairs — {barbell_accessible} ACCESSIBLE + {barbell_ambitious} AMBITIOUS. Produce **{n_variations_per_pair}** variations per pair…") with an instruction that these values override any different number remembered from the step prose. Cheap, universal, reversible.
2. **Overlays** — for steps where a number is load-bearing mid-instruction, a per-step patch file (`flows/overlays/*.patch.md`: ordered exact-match find→replace hunks; apply-or-error, never fuzzy) rewrites the resolved prompt **in memory**. Canon files are never modified.
3. **Run-local gates** — the engine materializes `output/studio/<run>/gates.yaml` with knob-resolved thresholds and passes `--gates` to the validators, so deterministic checks enforce the *experiment's* numbers.

---

## 4. FlowSpec — the executable document

Lofn's real shape is **linear coordinator → one fan-out over pairs → join → run-level gates**, so FlowSpec v1 models exactly that (not a general DAG): a `stages` list where each stage is `steps: [...]` (sequential) or `foreach: pair` (parallel over the pair list with a concurrency cap).

```yaml
# tools/explorer/flows/lofn-music@1.flow.yaml  (generated by the canon importer)
flow: lofn-music
version: 1
modality: music
base: canon@<git-sha-at-import>          # provenance: what this was imported from
knobs_defaults: presets/classic-6x4.knobs.yaml
context:
  template: skills/music/OVERALL_PROMPT_TEMPLATE.md   # read from canon at resolve time
  slots: [input, seed, meta_prompt, personality, concept_panel, medium_panel,
          marketing_panel, flairs, genres_list, frames_list, image_context]
model_tiers:            # heritage: vault/LOFN_MODEL_ASSIGNMENTS.md — now real again
  coordinator: {provider: anthropic, model: claude-opus-4-8, max_tokens: 16000}
  pair:        {provider: anthropic, model: claude-opus-4-8, max_tokens: 32000}
  enhance:     {provider: anthropic, model: claude-opus-4-8, max_tokens: 32000}
stages:
  - id: coordinator
    steps:
      - id: "00"
        title: Aesthetics & Genres
        prompt: skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md
        overlay: null
        tier: coordinator
        emits: step00_aesthetics_genres.md
        consumes: []                    # + CREATIVE CONTEXT always
        gates: [g.step00]
        on_fail: {policy: retry, max_attempts: 3}   # block | flag | retry
      # ... 01–05; step 05 emits pair_assignments (n_pairs entries)
  - id: pairs
    foreach: pair                       # fan out over step-05's assignments
    concurrency: "{max_concurrency}"
    steps:
      - id: "06" ... - id: "10"
      - id: "11"
        prompt: skills/music/steps/11_Generate_Music_Enhancement.md
        gates: [g.music_prompt, g.music_lyrics, g.andon]
        on_fail: {policy: quarantine_pair}          # the Andon Cord
  - id: portfolio
    steps:
      - id: distinctiveness
        kind: validator                 # a step can be a pure validator, no LLM call
        run: scripts/validate_portfolio_distinctiveness.py
        gates: [g.portfolio]
```

**Editing semantics:** flows are files; the Flow Lab edits them structurally (insert/remove/reorder step, change gates/tier/on_fail, add overlay) or as raw YAML with schema validation. A flow that has ever produced a run is **immutable** — edits copy-on-write to `@N+1` (provenance never dangles). The **canon importer** (WS0) generates `@1` per modality from `pipeline_map.yaml` + the step files' Required Inputs, and its output is verified against the v1 manifest (same node set, same gate set) — drift here is a build failure.

---

## 5. Engine — executing a run

**State machine.** `run: pending → running → complete | failed | cancelled`. Per step-instance: `pending → resolving → calling → validating → done | flagged | retrying(n) | failed`. Per pair: `active | quarantined | done`.

**The loop per step:** resolve prompt (resolver §6) → call provider (streaming; usage captured) → write artifact to `output/studio/<run>/` **using canonical artifact names** (`pair_03_step08_generation.md`, `step02_concepts.md`) so every existing validator, `rebuild_manifest.py`, and even `lofn-qa` work on studio runs unchanged → run the step's gates (`validate_step.py --gates <run-local>` + `check_human_subjects.py` where wired) → route: pass → next; FLAG → record chip amber, continue; hard fail → `on_fail` policy — retry re-calls with a repair prompt built from the validator's stderr (exactly `validate_with_retries.py` semantics, honoring its `.repair_attempt_N.md` convention), exhausted retries → **quarantine the pair** and emit the doctrine line: *"N of {n_pairs} pairs broke open at step X (gate: <name>)"* — loudly, before any portfolio stage, never a silent 5-shipped-as-6.

**Safety rails (each is an acceptance gate, §10):**
- **Human-subject:** `check_human_subjects.py` runs on every lyric/scene-bearing artifact (music 08–11, image/video/story 08–10); `HOLD_FOR_HUMAN` pauses the pair with a blocking banner — a human clicks proceed/kill. REAL GRIEF IS NOT RAW MATERIAL is enforced in the lab too.
- **Budget:** pre-run estimate (Anthropic: `count_tokens` on resolved prompts; others: chars/4 heuristic, labeled as such) × `pricing.yaml`; live meter accumulates actual `usage`; crossing the cap **cancels remaining calls loudly**, finishes writing in-flight artifacts, leaves the run resumable.
- **Canon immutability:** the engine holds no write handle outside `output/studio/**`; WS7 verifies a full run leaves `git status` clean.

**Resume.** Disk is authority (`rebuild_manifest.py` doctrine): on resume, re-scan the run dir, trust valid artifacts, restart at the first missing/failed step per lane. The append-only `runlog.jsonl` (every state transition + usage) is what SSE tails live and what the UI replays for a finished run — one event stream, two consumers.

**Concurrency.** Pair fan-out via semaphore, default `max_concurrency: 6`; a second concurrent run shares a global cap (cap-and-stagger, per `EXECUTION.md`/daily doctrine). Provider rate-limit errors back off per SDK defaults; 429s never burn retry-attempts (they're transport, not gate failures).

**`run.json` (provenance, written incrementally):** run id/title/created; `flow` (name, version, file sha); `knobs` (fully resolved incl. derived); `overlays` (paths + shas); `canon_sha` (git HEAD at start) + dirty-flag; `context` (seed/personality/panel refs + shas); `model_tiers` as executed; per-step: provider/model, tokens in/out, cost, duration, attempts, gate results, artifact sha; totals; budget cap & spend; status; judge verdicts (appended later). **Keys never appear anywhere in it.**

---

## 6. Resolver — one path, two consumers

`resolver.resolve(flow, knobs, context, step, pair?) -> ResolvedPrompt{system?, user, meta}`:

1. Read the step's canon prompt file (read-only) → 2. apply overlay hunks (exact-match or error) → 3. fill CREATIVE CONTEXT slots from the run's context artifacts (verbatim-injection rule: full ICB, never name-references — the anti-personality-collapse doctrine) → 4. append the RUN PARAMETERS block from knobs → 5. return with token estimate.

The **Compile view** and the **engine** call this same function — acceptance gate S3 asserts byte-equality between what Compile displayed and what the engine sent (recorded in runlog). Turning a knob in the UI re-resolves affected steps and shows a diff — the "feel the knob" loop, at zero API cost.

---

## 7. Providers, pricing, keys

**Heritage reference (WS1 ports these):** [`docs/reference/lofn-legacy-llm/`](reference/lofn-legacy-llm/) vendors the streamlit-era provider code — `llm_integration.py` (`OpenRouterLLM`, `PoeLLM`, `GeminiLLM`, OpenAI Responses, the retry-with-correction loop, the per-model max_tokens table), `parsing.py` (the tolerant JSON loader to reuse for step-output parsing), `o1_integration.py`, `model_defaults.yaml` (modality×role model tiers with `Poe-`/`OR-` prefixes). See that dir's README.

**Three dialects cover all five providers (+ mock):**

| Provider | Dialect | Base URL / notes |
|---|---|---|
| Anthropic | `anthropic` (native SDK) | default `claude-opus-4-8`; adaptive thinking `{type:"adaptive"}` on 4.6+; streaming always; `count_tokens` for estimates; per-model params gated by a capability table (e.g. no `temperature` on 4.7+/Fable — the dialect strips unsupported params rather than 400ing). Follow current `claude-api` conventions, NOT the legacy `budget_tokens`. |
| OpenAI | `openai-compat` | `https://api.openai.com/v1` (native OpenAI SDK). |
| OpenRouter | `openai-compat` | `https://openrouter.ai/api/v1`; Bearer auth + `HTTP-Referer`/`X-Title` headers; `fetch_openrouter_models()` for context length; **prefer the per-request cost OpenRouter returns**. `OR-` model prefix. |
| Poe | **`poe` (fastapi_poe)** | NOT OpenAI-compatible. `fp.get_bot_response(bot_name=<model minus "Poe-">, api_key=…)`, collect `PartialResponse.text`; **no stop sequences**. Ported from the legacy `PoeLLM`. |
| Gemini | **`google-genai` (native) primary; `openai-compat` optional** | Native `google.genai` `generate_content` (proven in the legacy `GeminiLLM`, incl. 2.5 thinking budget). WS1 verifies the OpenAI-compat endpoint `…/v1beta/openai/` and may use it if simpler. |
| Mock | `mock` | record/replay fixtures under `tools/explorer/server/studio/fixtures/`; a `record` mode captures one cheap live run into fixtures. **All engine tests run on mock — CI spends $0.** |

**Interface:** `ProviderClient.complete(request: {model, system, messages, max_tokens, stream, params}) -> streamed chunks + final {text, usage{in,out}, stop_reason}`. No tool use, no structured outputs in v2.0 — Lofn steps are single-shot prompt→text.

**Pricing** (`tools/explorer/pricing.yaml`, user-editable — prices drift; seeded from current published rates, per-MTok in/out): `claude-opus-4-8: 5/25`, `claude-sonnet-5: 3/15 (intro 2/10 through 2026-08-31)`, `claude-haiku-4-5: 1/5`, `claude-fable-5: 10/50`, plus OpenAI/OpenRouter/Gemini entries at WS1 (OpenRouter returns per-request cost in its response — prefer actuals when the provider reports them). Unknown model → cost shows "unpriced" and the budget meter counts tokens only, loudly.

**Keys:** resolved from environment first, then `.env.studio` (repo root, gitignored, `chmod 600`-equivalent). Named per provider: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `POE_API_KEY`, `GEMINI_API_KEY` — **and** the legacy `config.yaml` aliases the operator already uses (`ANTHROPIC_API`, `OPENAI_API`, `OPEN_ROUTER_API_KEY`, `POE_API`, `GOOGLE_API_KEY`); the resolver accepts either. `/api/studio/providers` reports only `{provider, configured: bool, model_list?}` — never key material. A **redaction filter** scrubs anything matching known key shapes from runlog, SSE, error messages, and run.json; WS7 tests it with canary keys. Keys never leave the machine except to their own provider's endpoint.

---

## 8. UI — four new surfaces (v1 register: EXPERT_OPERATOR cockpit, keyboard-first, dense, calm)

1. **Flow Lab** (`g f`) — flow picker + version list; the Atlas rail rendered *from the FlowSpec* in edit mode: insert/remove/reorder steps (buttons, not drag-physics), per-step drawer (prompt source + overlay editor, tier, gates multi-select, on_fail policy), fan-out badge showing `foreach: pair × {n_pairs}`; raw YAML tab (Monaco, schema-validated on save); "New version from this" (copy-on-write); diff vs any other version.
2. **Knobs** (panel within Flow Lab + standalone) — grouped numeric inputs with range sliders, derived values live-computed and badged `ƒ`, invariant violations inline (blocks Run, not Compile), preset save/load/duplicate; every change updates the Compile diff and the cost estimate in place. *The signature interaction: drag `n_pairs` 6→10 and watch step-05's resolved prompt, the derived barbell, total_prompts, the run-local gate values, and the estimated cost all move together.*
3. **Run Bench** (`g r`) — launch form (flow@ver · preset · seed/brief input · personality & panel pickers from v1 Library · model-tier map with provider status lights · budget cap · dry-run toggle); live view: one lane per pair + coordinator lane, step cards fill as artifacts stream (tail of `runlog.jsonl` over SSE), gate chips light green/amber/red, cost ticker vs budget bar, quarantine banners, HOLD_FOR_HUMAN modal; artifact click → Monaco viewer; Cancel / Resume buttons; run history table (provenance columns, cost, verdicts).
4. **Compare & Judge** (`g c`) — pick runs A/B: aligned per-artifact diffs, gate-report delta table, cost/token delta, knob diff; **Blind Judge** mode: payload-only cards (extraction per `build_blind_set.py` rules — provenance-stripped, fails closed if a payload can't be cleanly extracted), randomized sides, pick winner per matchup + optional note; verdicts append to both run.json files. The OEC is the operator's taste; the studio's job is to make the comparison honest, not to fake statistics at n=1.

---

## 9. API surface (added to the v1 FastAPI app under `/api/studio/*`)

`GET/POST /flows` · `GET/PUT /flows/{name}@{v}` (immutable-once-run enforced) · `POST /flows/import` (canon importer) · `GET/POST /presets` · `POST /compile` `{flow, knobs, context, step?}` → resolved prompts + diffs + token/cost estimates · `GET /providers` · `POST /providers/{p}/test` (1-token ping) · `POST /runs` (launch; `dry_run: true` = compile-all only) · `GET /runs`, `GET /runs/{id}` · `GET /runs/{id}/events` (SSE tail of runlog) · `POST /runs/{id}/cancel|resume` · `POST /runs/{id}/decision` (HOLD_FOR_HUMAN proceed/kill) · `POST /judge` `{run_a, run_b}` → blind set · `POST /judge/verdict` · `POST /promote/preview` → diffs (gates.yaml + prose patches) · `POST /promote/apply` (knobs/thresholds only; via v1 FileService; runs meta-check after) · `POST /promote/brief` → `PROMOTION_BRIEF.md` for structural changes.

---

## 10. Acceptance gates (non-waivable, additive to v1's)

- **S1 — Canon immutability:** a full mock run + a live smoke run leave `git status` byte-clean except `output/studio/**` and explicitly saved `flows/**` files. Tested by hash-sweeping `skills/ vault/ .claude/` before/after.
- **S2 — Key hygiene:** canary keys planted in env/`.env.studio` never appear in run.json, runlog.jsonl, SSE frames, API responses, or UI state dumps.
- **S3 — Compile ≡ execute:** the resolver output shown in Compile is byte-identical to what the engine sent (asserted against runlog for every step of a mock run).
- **S4 — Budget hard-stop:** with a cap set below projected spend, the run halts at/below cap, states why, and resumes cleanly after the cap is raised.
- **S5 — Resume idempotence:** kill the engine mid-run (chaos test), resume → final artifact set byte-identical to an uninterrupted mock run.
- **S6 — Quarantine is loud:** a forced hard-fail exhausting retries produces the "N of {n_pairs} pairs broke open at step X (gate: g.…)" banner before any portfolio/QA stage; the run completes as N-of-M with the shortfall named everywhere the run is displayed.
- **S7 — Human-subject gate wired:** a fixture lyric with a name+crime tuple triggers HOLD_FOR_HUMAN and blocks the pair until decided.
- **S8 — Provenance replay:** `run.json` + fixtures reproduce a mock run byte-identically; a live run re-launches with identical resolved prompts (model outputs may differ — that's the experiment).
- **S9 — Gate parity:** studio gate results for an artifact equal the CLI's results for the same artifact + same gates file.

---

## 11. Build plan — workstreams for the Opus fleet

Dependencies: **WS0 → WS1 → WS2 → WS3 → {WS4, WS5, WS6 in parallel} → WS7**. (WS4 can start against WS2's frozen API while WS3 is in flight.)

| WS | Deliverable | Key tasks | Done when |
|---|---|---|---|
| **WS0 — Contracts & importer** (blocking; senior agent) | FlowSpec/knobs/run.json JSON-schemas frozen; `flows/lofn-{music,image,video,story}@1.flow.yaml`; `presets/classic-*.knobs.yaml`; `docs/KNOB_CENSUS.md` (every number, file:line); canon-side micro-patch (`validate_step.py --gates` flag wired to existing `load_gates(path)`; meta-check skiplist for `flows/**`); streamlit-fossil study note (chain order & slot-filling fidelity vs `lofn/llm_integration.py`) | schemas validate; importer output cross-checks against `pipeline_map.yaml` (same nodes/gates or explicit delta); `--gates` proves backward-compat (no-flag behavior byte-identical) |
| **WS1 — Providers & pricing** | `providers.py` (anthropic native + openai-compat + mock w/ record/replay), `pricing.py` + `pricing.yaml`, key resolution + redaction filter, `/providers` endpoints; **verify Poe & Gemini compat endpoints live**, document or drop | mock replay deterministic; 1-token live ping per configured provider; redaction unit tests pass; cost math matches hand-computed fixtures |
| **WS2 — Resolver & knobs** | `knobs.py` (safe evaluator, derivations, invariants), `resolver.py` (overlay hunks, slot filling, RUN PARAMETERS block), `/compile`, run-local gates materializer | compile of canon-baseline flow with classic preset reproduces (modulo the appended RUN PARAMETERS block) the prompts the v1 pipeline would assemble; knob change → correct diff; property tests on evaluator |
| **WS3 — Engine & run store** | `engine.py` state machine (fanout, semaphore, retry/repair, quarantine, HOLD_FOR_HUMAN, budget abort), `runstore.py` (run.json incremental, runlog.jsonl, SSE), resume, `/runs*` endpoints | S3–S7 pass on mock; a $≤2 live smoke run (haiku-tier, `n_pairs:2 × n_variations:1`) completes end-to-end with real gate chips |
| **WS4 — Flow Lab & Knobs UI** | FlowSpec-driven editable rail, step drawer, overlay editor, YAML tab, version mgmt; knobs panel wired to `/compile` | edit→version→compile loop under 10s; invariant violations render; immutability enforced in UI |
| **WS5 — Run Bench UI** | launch form, live lanes + gate chips + cost ticker via SSE, quarantine/HOLD banners, artifact viewer, cancel/resume, history | mock run renders live end-to-end; replay of a finished run identical to its live rendering |
| **WS6 — Compare & Judge UI** | A/B diffs, gate-report deltas, blind-judge flow (fails-closed extraction), verdict recording | blind cards contain zero provenance strings (automated sweep); verdicts land in both run.json files |
| **WS7 — Hardening & acceptance** (adversarial agent — did not build what it tests) | S1–S9 test suite; chaos resume; canary-key sweep; canon hash sweep; docs (`tools/explorer/README.md` studio section, runbook) | all gates green on mock; live smoke documented with actual cost |

Fleet shape: WS0 solo senior; WS1–WS3 one backend agent each (sequential contracts, parallel tail-work); WS4–WS6 three UI agents in parallel against frozen OpenAPI; WS7 adversarial. Estimated new code: ~4–5k backend, ~4k frontend.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Knob prose-injection isn't obeyed by weaker models (model writes 6 pairs anyway) | Deterministic gates enforce the knob values (run-local gates.yaml) → violations fail/flag loudly; overlays escalate stubborn steps; census marks which steps needed overlays |
| Poe/Gemini OpenAI-compat gaps (streaming quirks, param support) | WS1 verifies live before UI depends on it; dialect param-capability table strips unsupported params; drop Poe if the account lacks API access |
| Cost surprises on full 6×4 runs | Dry-run default ON for first run of any new flow; budget cap required (no unlimited default); haiku-tier smoke preset ships in-box |
| A studio run mutates canon via a mis-scoped path | S1 hash sweep in CI + engine has no FileService write handle at all — it constructs paths only under its run dir |
| Flow immutability circumvented by editing the YAML on disk | run.json stores the flow file's sha; a mismatch renders a tamper warning on the run and disables promote-from-run |
| `--meta-check` starts warning about flows/presets restating numbers | WS0 skiplist patch; flows declare `gates_key` bindings so drift detection can even *extend* to them later |
| Blind judge accidentally leaks provenance | reuse `build_blind_set.py` extraction + its fail-closed doctrine; automated no-provenance sweep in WS6 done-criteria |

---

## 13. Decisions (resolved 2026-07-03 by the operator)

1. **Poe — KEEP.** Poe is not OpenAI-compatible; it uses `fastapi_poe` (`fp.get_bot_response`, `bot_name` = model minus the `Poe-` prefix, no stop sequences). The historical integration (`PoeLLM`) is vendored at [`docs/reference/lofn-legacy-llm/llm_integration.py`](reference/lofn-legacy-llm/llm_integration.py) — WS1 ports it as a dedicated `poe` dialect (§7).
2. **LLM-judge QA step — INCLUDE behind an `experimental` flag** — a flow step that runs the `lofn-qa` prompts via API; off by default; WS3 stretch.
3. **Structural promotion — AUTO-APPLY for patches.** All knob/threshold values and simple prose patches auto-apply via the v1 byte-safe FileService with diff preview + a `--meta-check` after; a Promotion Brief still accompanies the change for the record.
4. **Budget — CONFIRMED.** No run starts without an explicit cap; smoke preset `$2`, standard `$25`.
5. **Modalities — ALL FOUR at import (music / image / video / story), PROMPTS ONLY.** The studio produces prompt packages (Suno / Flux / Veo / story prompts) and **does not call any media-generation API** — rendering is the operator's to do from the finished prompts ("we let the user take it from here"). Reinforces the §1 non-goal.
6. **Keys — `.env.studio`** (Windows Credential Manager deferred). Resolver accepts both standard `*_API_KEY` names and the legacy `config.yaml` aliases (§7).
