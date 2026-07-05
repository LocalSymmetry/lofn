# Run Ledger — Operational Memory (infra/process only)

> The pipeline's **operational** memory: what BROKE and how it was fixed, run over run. Ported concept from
> the OpenClaw private build's `MEMORY.md` operational ledger (dated mandates + infra lessons that made run
> N's fix law by run N+1). Distinct from `vault/COMPETITION_LEARNINGS.md`, which is the **aesthetic** ledger.

## ⛔ HARD FIREWALL — this file records INFRA / PROCESS facts ONLY

No aesthetic or taste claim may ever be filed here. "A pair dropped at concurrency 6," "the research
subagent couldn't fetch," "a gate false-failed a valid prompt" — yes. "INDIGNATION underperforms," "warm
palette wins," "this motif is tired" — **NO**; those are venue-taste and belong in
`vault/COMPETITION_LEARNINGS.md` under its advisory contract (and triggered-INDIGNATION is exempt there).
The fixed schema below has **no free-text aesthetic slot** by design — if a lesson won't fit the columns, it
does not belong in this file.

## Contract
- **Written by** `lofn-qa` / daily Phase 3 on any run that hit an *operational* failure (dropped/quarantined
  pair, stale path, quota/API error, timeout, gate false-fail, resume/manifest disagreement).
- **Read at Phase −1** (continuity load): scan the tail for `open` / `watch` hazards so the next run doesn't
  re-hit a known plumbing failure. An open hazard is a **heads-up, never a creative constraint**.
- **Advisory only.** This never gates SHIP/FAIL and never enters the ICB.
- **Hard-capped (~30 rows).** Prune `fixed`-and-stale entries first; keep `open`/`watch`.

## Schema
`{ date · run_id · what_broke · root_cause · infra_fix · status }`  — `status ∈ open | watch | fixed`

| date | run_id | what_broke | root_cause | infra_fix | status |
|------|--------|------------|------------|-----------|--------|
| 2026-04-20 | daily | research brief ~50% fabricated (fake challenge, wrong APOD, invented album) | the research SUBAGENT had no web-fetch tool — it could only hallucinate | Phase 1 fetch is done by the CONTROLLER session; subagents expand verified facts only, never fetch | fixed |
| 2026-04-xx | daily | a fan-out pair was dropped when 6 spawned at once | provider cap of 5 concurrent children | cap-and-stagger: spawn 5, hold the 6th in a pending slot (staggered 5+1) | fixed |
| 2026-06-19 | migration | subagents read dead `/data/.openclaw/` and `lofn-core/GOLDEN_SEEDS.md` paths | OpenClaw absolute paths in satellite docs after the Claude-native port | repo-relative paths; single CANONICAL PATHS block; `verify_pipeline_map.py` | fixed |
| 2026-06-28 | daily | a MUSIC PROMPT shipped truncated mid-phrase yet self-check PASSed | the check counted chars, not sentence-completeness | `gates.yaml music_prompt_terminal_punctuation` sense-floor (hard fail) | fixed |
| 2026-07-04 | daily-lofn-prime-test | `validate_step06_distinctiveness.py` FALSE-FAILED pair_06 (counted 3 facets, needs >=8) | the facet-count regex only recognized `###`/numbered/`facet N` line-starts, not bold `**Facet N**` markers; pair_06 wrote 10 real facets as bold (10 why-it-matters + 10 failure-mode) | broadened the regex to count bold facets + added a `why it matters` format-agnostic fallback (`max` of signals) in `scripts/validate_step06_distinctiveness.py`; re-ran -> PASS, no fixture regression | fixed |
| 2026-07-04 | daily-lofn-prime-test | two independent daily runs on the same APOD produced identical song TITLES (airbag-hymn / lander-learned-to-bounce / house-cat-signal / dust-brown-coronation) | no cross-RUN title dedup exists; both runs derived titles from the same strong shared fact (content similarity was 0.002-0.004, i.e. NOT contamination) | none needed (content is distinct); noted as a future enhancement — a cross-run title-collision check would complement the cross-PAIR distinctiveness validators | watch |

> Add a row the moment an operational failure is diagnosed; update `status` to `fixed` when the fix lands and
> is verified. New entries go at the bottom; prune from the top when over cap.
