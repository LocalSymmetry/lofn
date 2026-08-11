# Daily run examples — read the artifacts, not the description

These are **real, complete Lofn daily runs**, published unedited so the pipeline can be studied from its output instead of from its documentation. Each directory is one run: research → Golden Seed → 18-voice panel → 6 pairs × 4 variations = 24 songs → QA.

Pair these with the two doctrine documents:

- [`docs/ARCHITECTURE_STATEMENT.md`](../../docs/ARCHITECTURE_STATEMENT.md) — what the design asserts
- ⭐ [`docs/AGENTIC_ENGINEERING_GUIDE.md`](../../docs/AGENTIC_ENGINEERING_GUIDE.md) — how it is built, maintained and validated, with an enforcement column on every rule

⛔ **Nothing in these runs was rendered or published.** Per `vault/AUTONOMY.md`, autonomous runs stop at drafts on disk — no paid render, no publish, no spend without the human. What you are reading is the complete text pipeline, exactly as it landed.

---

## Where to start

**If you want to understand the architecture in one file:** open any run's `CREATIVE_CONTEXT.md`. That is the **ICB — the Immutable Continuity Block**: filled once at the end of Phase 1, hashed, injected into every downstream agent, and **writable by nobody**. It is the whole thesis in one artifact — continuity as a frozen value rather than a preserved memory.

**If you want to see the machinery that checks the work**, the 2026-08-09 run ships its instruments:

| File | What it demonstrates |
|---|---|
| `2026-08-09/coordinator_restat.py` | ⭐ **The join.** An agent's return is a *claim*; this is the proof. Its docstring carries five instrument laws, each written against a recorded failure — including *"an empty extraction is a hard error, never a score"* (two empty strings compare as 1.000) and *"assert the extraction count before concluding anything."* |
| `2026-08-09/GATE_REPORT.json` | The countable subset, 24 rows, machine-readable |
| `2026-08-09/verify_icb.py` | The LF-normalised sha check on the frozen block |
| `2026-08-09/rebuild_run_state.py` | `RUN_STATE.md` rebuilt by stat-ing files — never hand-asserted |
| `*/RUN_STATE.md` | The disk-derived manifest that makes a run resumable from disk alone |
| `*/.RUN_LOCK` | The one-controller interlock, in its completed state (host + controller id redacted; schema and lifecycle verbatim) |

**If you want to see judgement rather than checking**, read the `QA_REPORT.md` files. QA runs in a fresh context on a different model tier, framed adversarially. The 2026-08-09 report's `D-3` ruling is the clearest single example of why a validator cannot do this job: two songs shared the token *"thirty"*, and the judge walked the three-axis collision bar (same fact + same device + same conclusion), matched zero of three, and ruled it clean — *"the shared token is an artifact of the decimal system: 'a hundred and thirty-eight' contains 'thirty' the way 'there' contains 'here.'"*

---

## The runs

| Run | Worth reading for |
|---|---|
| **`2026-08-06-somebody-went-and-looked`** | ⭐ **A withdrawn pair, kept on the record.** `_withdrawn_p06/` holds the artifacts of a pair that was pulled, and `05b_P06_REPLACEMENT.md` is its replacement — the REPLACE route running live. Also `RENDER_AUDIT.md`, and a step-11 pass that **corrects its own earlier claim** ("the pursuer sweep was overstated") rather than quietly restating it. |
| **`2026-08-07`** | Cross-pair **device firewalling** — pairs actively excluding each other's tokens so two songs cannot converge by ear or by scanner. See `pair_04_step06_facets.md`. |
| **`2026-08-08`** | ⭐ **A contract conflict resolved in writing.** `pair_03_step10_final_package_enhanced.md` §1 documents a live disagreement between a reference guide and the step-11 contract, names which wins and why. Also the clearest **declared-false-positive** note in the corpus: a naive grep over the file hits banned tokens *inside the text of the ban that forbids them*. |
| **`2026-08-09`** | ⭐ **The best single teaching run.** Ships its own instruments (above). Its `03_panel_debate.md` documents **three panel defects repaired before the room opened** — a missing skeptic seat, two seats named for figures with no locatable published record, and one figure double-seated as Devil's Advocate on two panels. Its `00_research_brief.md` records two sources as `UNAVAILABLE` rather than substituting them, and **discards a feed that returned stale rows** — *an impossible number is a bug report, not a finding.* |

---

## What to notice, if you are building something similar

1. **The pairs share nothing.** Every file is named `pair_NN_*`. No pair writes the INDEX, `RUN_STATE.md`, or the ICB. All cross-pair aggregation happens single-threaded, after the wave lands. Concurrent corruption is made impossible by construction, not avoided by convention.
2. **The artifacts are cumulative and the validators know it.** A step-09 file embeds step-08's bytes verbatim, so the distinctiveness validators take the **last** matching segment — measuring the embedded history instead of the current work is its own silent failure.
3. **The self-reported numbers are re-derived.** Every package states its measured char counts, sung lines and soundcraft figures, and QA reproduces them independently. In this corpus, when a disagreement was properly re-measured, **the agent was right and the instrument was wrong** — twelve times out of twelve.
4. **The failures are in the files.** Repairs, quarantines, withdrawn pairs, corrected claims and rejected sources are all on the record. A run that reports zero repairs across 24 artifacts triggers an audit of the judge, not a celebration.

*Panel voices in these runs are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*
