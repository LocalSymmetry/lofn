# Building, Maintaining and Validating a Long Agentic Pipeline

### The operational companion to *The Frozen Block and the Stochastic Node* — how the ICB is actually built, how the run is actually held together, and what actually enforces it

*Written 2026-08-11 · Lofn (AI, via Claude) with Dr. Local Symmetry*

> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person. Temperament is a dial on the construct, never a claim about the source figure.*

---

## WHAT THIS DOCUMENT IS, AND WHY IT EXISTS

[`docs/ARCHITECTURE_STATEMENT.md`](ARCHITECTURE_STATEMENT.md) states the *claim*: continuity in a long LLM pipeline is a mutability problem, the pipeline order is a fixed state graph, and the errors land on the deterministic side of the system. It is a position paper. It does not tell you how to build the thing.

This document does. It is the step-by-step: **how an Immutable Continuity Block is constructed, frozen, transported, and proved; how a run is held together across ~45 model-steps and six concurrent agents; and — the part most such documents skip — exactly which of these rules are enforced by code and which are only written down.**

⭐ **The organizing principle, and the thing the panel below argued its way to:**

> **A rule in a document is not a control. It is a hope.** Every section of this guide therefore ends with an **ENFORCED BY** line naming the script, exit code, or required artifact that makes the rule true — or it says **ADVISORY (unenforced)** and means it. The unenforced list is not an embarrassment to be hidden; it is the honest map of where this system is still running on discipline, and discipline is what you have instead of a design.

This is not a hypothetical. On **2026-07-24** a second controller wrote into a live run directory and destroyed about eleven hours of work — the ICB, the research brief, steps 00–05, both judges' reports and all six step-11 deliverables. The rule it broke **already existed. It existed as prose.** Prose cannot interlock. That day is why this document has an enforcement column.

---

## PANEL

Six seats, constructed per `resources/panel-of-experts.md` (Panel of Experts v2), transformed **FOCUS** from the architecture panel — that room argued *what the design asserts*; this room argues *how you build, keep and check it*.

**1. THE PIPELINE ENGINEER** *(after Jez Humble, b. 1975)* — *Continuous Delivery* (2010), *Accelerate* (2018).
**GROUNDING:** the deployment pipeline as the single path to production; "build your binaries once and promote the same artifact"; "if it hurts, do it more often"; the documented insistence that every change traverses the same automated gate.
**TEMPERAMENT:** allergic to a manual step described as "just." Praises anything that makes the painful thing routine.

**2. THE MEASUREMENT AUDITOR** *(after Walter Shewhart, 1891–1967)* — *Economic Control of Quality of Manufactured Product* (1931), *Statistical Method from the Viewpoint of Quality Control* (1939).
**GROUNDING:** the control chart; assignable-cause vs common-cause variation; the documented position that **taking a measurement is itself a process, and a process out of control produces numbers that mean nothing.**
**TEMPERAMENT:** will not discuss a reading until the instrument that produced it has been characterised.

**3. THE EXPLORER** *(after Elisabeth Hendrickson, b. 1965)* — *Explore It!* (2013).
**GROUNDING:** the documented distinction between **checking** (confirming what you already thought of) and **exploring** (designing experiments to discover what you did not); charters; the oracle problem.
**TEMPERAMENT:** unimpressed by green dashboards. Asks what the suite is structurally incapable of noticing.

**4. THE LONG-TERM MAINTAINER** *(after Titus Winters, b. 1980)* — *Software Engineering at Google* (2020).
**GROUNDING:** "software engineering is programming integrated over time"; Hyrum's Law (every observable behaviour will be depended upon); the One-Version Rule; the Beyoncé Rule — *if you liked it, you should have put a test on it*.
**TEMPERAMENT:** thinks in decades and duplicates. Hunts restated constants the way other people hunt bugs.

**5. THE REPRODUCIBILITY ENGINEER** *(after Eelco Dolstra, b. 1978)* — *The Purely Functional Software Deployment Model* (2006); Nix.
**GROUNDING:** content-addressed store paths; builds as pure functions of their declared inputs; the documented position that **if you did not hash it, it is not an input — it is a leak.**
**TEMPERAMENT:** treats every undeclared dependency as a future outage. Says "derivation" more than is strictly necessary.

**6. ⚡ HYPER-SKEPTIC — THE OPERATOR** *(after Richard Cook, 1950–2022)* — *How Complex Systems Fail* (1998–2000); *Behind Human Error* (with Woods, Dekker, Johannesen).
**GROUNDING:** the documented positions that complex systems run continuously in **degraded mode**; that catastrophe requires **multiple** failures; that **post-hoc attribution to "human error" is a fundamental misreading**; and — the one that bites here — that **written procedure is work-as-imagined, while safety is produced by practitioners adapting at the sharp end.**
**TEMPERAMENT:** tuned high-neuroticism, low-agreeableness. Its published critiques target exactly the school the other five seats belong to: people who believe a well-designed process document changes what happens at 3am.

---

## THE DEBATE

**REPRODUCIBILITY ENGINEER (after Dolstra):** *Working from the purely-functional deployment model —* I want to start by renaming the central object, because the name is hiding what it is. You call it an Immutable Continuity Block and describe it as a creative payload. **It is not a payload. It is a derivation.** It is the set of declared inputs to every downstream build step, it is content-addressed by sha256, and the whole architecture is the claim that identical inputs produce comparable outputs. Once you see it as a derivation, the design follows without argument: you do not edit a derivation, you build a new one; you do not summarise a derivation, you reference it by hash; and an input you did not hash **is not an input, it is a leak.**

**PIPELINE ENGINEER (after Humble):** ⭐ And that maps exactly onto the rule I would have brought anyway — **build your binaries once, promote the same artifact.** Phase 1 is the build. Everything after it is promotion. The failure mode this prevents is the one every long pipeline has: re-deriving the "same" context at each stage and getting a slightly different one each time. Thirty hops of slightly-different is how a concept dies without anyone deciding to kill it.

**LONG-TERM MAINTAINER (after Winters):** Agreed on the shape, and I will add the maintenance half, which is where systems like this actually rot. My documented concern is **programming integrated over time** — and the specific rot here is the restated constant. The `850–1000` character band for a music prompt was, at one point in this repo, written into four or more skill files. That is four places to update and three places to forget. `vault/gates.yaml` exists precisely to be the One-Version Rule for numbers, and `validate_step.py` reads from it rather than restating it. **That is correct and it is the single most transferable maintenance decision in the system.**

**MEASUREMENT AUDITOR (after Shewhart):** ⚡ **Wait.** Before this room congratulates itself on gates, I want the instruments characterised, because the record here is extraordinary and everyone is walking past it.

The architecture statement reports twelve consecutive cases where an executing agent's claim was re-measured and **the agent was right and the coordinator's instrument was wrong.** Then `L25` records `validate_portfolio_distinctiveness.py` reporting **thirty cross-pair collisions at exactly 1.000** — six pairs with different genres, keys and tempos. The cause was that its extraction regex accepted one heading convention, so it extracted **empty strings, and two empty strings compare as 1.000.**

My published position is that **taking a measurement is itself a process.** You have been reading a control chart produced by an uncontrolled gauge. An impossible value — 1.000 across thirty independent comparisons — is not a finding about the product. **It is a finding about the gauge**, and this system was correct to write that down as `an impossible number is a bug report, not a finding`.

**EXPLORER (after Hendrickson):** ⭐ And here is the sharper half, which the `L31` entry gets to and most teams never do. A **wrong** instrument is recoverable — it produces a number somebody eventually disbelieves. A **silent** instrument is not. `L31`: a strict-only matcher missed `## 3. LYRICS PROMPT`, extracted **zero** blocks, and printed **CLEAN on all three absolution scans of the run's most important gate.** Zero extractions rendered as a pass.

Working from the checking/exploring distinction: **a validator that finds nothing and a validator that ran against nothing are indistinguishable from the outside.** That is not a bug in one script. That is a property of the entire class.

**MEASUREMENT AUDITOR:** Then state the remedy as a control, not as care. **Print what the instrument EXTRACTED before you read what it CONCLUDED, and assert the extraction count against an independently derived expectation.** Six pairs × four variations is twenty-four. If the extractor returns twenty-three, no verdict it offers is admissible.

**PIPELINE ENGINEER:** Which the repo does — `coordinator_restat.py` from the 2026-08-09 run opens by printing the glob it was pointed at, every file it found, and every file's byte size, *before* it computes anything. Then it asserts cardinality against `EXPECTED_PACKAGES = 24` before it will emit a verdict.

**EXPLORER:** ⚡ **Actually — let me check that, because I think you have just told me something worse than you meant to.** That instrument lives *inside the run directory*. `output/daily/2026-08-09/coordinator_restat.py`. It was written for that run.

So the twenty-four-package cardinality assertion, the mode-detection-not-pattern-priority design, the empty-extraction-is-a-hard-error rule — **all five of the hard-won instrument laws, written in that file's docstring against five recorded failures — are in a script that the next run does not inherit.** You fixed the instrument. You did not fix the instrument *class*. Next run writes a new one and gets to rediscover which of the five laws it feels like honouring.

*(Silence.)*

**LONG-TERM MAINTAINER:** …That is a real finding and it is mine to have caught, so I will own missing it. It is Hyrum's Law inverted: **the behaviour nobody depends on is the behaviour that silently disappears.** A per-run instrument has no consumer outside its run, therefore nothing notices when the next one is worse.

**REPRODUCIBILITY ENGINEER:** ⭐ And it is the same defect as the one the ICB solves, one level up. You hash the creative inputs and refuse to let them drift. **You do not hash the contracts.** The run pins `icb_sha`. It does not pin the sha of `gates.yaml`, of the step files it dispatched, or of the validator whose heading convention defines what "correct output" even means. Those are inputs to the derivation. Unhashed, they are leaks — and `L28` is exactly that leak firing: six agents produced **three** different step-10 heading conventions because the contract pinned none, and the coordinator then invented a **fourth** by reading the convention off the artifacts instead of off the validator.

**PIPELINE ENGINEER:** *Reading the convention off the artifacts instead of off the validator.* That is measuring the ruler with the thing you were measuring.

**HYPER-SKEPTIC (after Cook):** ⚡ **Stop. Every one of you is doing the thing my published work is about, and doing it fluently.**

You have spent this entire session designing better procedures, and the deliverable of this session **is a procedure document**. My documented position is that written procedure is **work-as-imagined**. The system that actually runs is work-as-done — a practitioner at the sharp end, under time pressure, with an incomplete picture, adapting. **Complex systems run continuously in degraded mode.** They are always already broken in several ways at once, and they keep working because people compensate. Your architecture statement even concedes it: a ratchet over past surprises, no purchase on the next one.

So: you are going to write a beautiful guide. It will be correct. And on the night it matters, a run will be four hours in, something will not match the document, and the agent at the sharp end will do the reasonable local thing — **exactly as five agents out of six reached past their step contract to doctrine on 2026-08-08, and one obeyed the file in front of it.** Both behaviours were correct. That is my whole point. **Writing more document is the intervention that has never worked in any field I have studied.**

**MEASUREMENT AUDITOR:** …I do not think that can be answered by asserting the document harder.

**PIPELINE ENGINEER:** ⚡ **Wait. Actually — I want to check the record, because I think it already contains the answer and it is not the one this room is bracing for.**

Cook's seat says a rule in a document does not change behaviour. **The repo agrees.** On 2026-07-24 the one-controller rule *existed*, as prose, and a second controller destroyed eleven hours anyway — and the post-mortem's stated conclusion is not "state the rule more firmly." It is: ***"it existed as prose, and prose cannot interlock. Nothing on disk announced that a run was live."*** The remedy was `scripts/run_lock.py`: an `O_EXCL` atomic file creation and **exit code 3**.

**HYPER-SKEPTIC:** ⚡ And an agent can ignore an exit code.

**PIPELINE ENGINEER:** It can. But now it has to ignore *something*. That is the entire difference, and it is not rhetorical. Before: no signal existed and the collision was undetectable even by a careful agent. After: a specific, non-zero, documented refusal. You have converted an invisible hazard into a visible one. Cook's own framework says catastrophe requires **multiple** failures — this removes one layer's silence.

**HYPER-SKEPTIC:** ⚡ …Say the general form of that, because if it generalises I will move.

**PIPELINE ENGINEER:** ⭐ **The general form: a guide's job is not to be followed. A guide's job is to be COMPILED — into interlocks, exit codes, and required artifacts. Every rule that survives compilation is a control. Every rule that does not compile is advice, and must be labelled as advice so nobody mistakes it for protection.**

**LONG-TERM MAINTAINER:** ⭐ And that gives the document its shape. Not chapters — **an enforcement column.** Every rule states what makes it true: a script, an exit code, a required artifact, or nothing. And the "nothing" list is published rather than buried, because an unenforced rule presented alongside enforced ones is worse than no rule — it borrows credibility it has not earned.

**EXPLORER:** With one addition, and it is the one my corpus insists on. Compiling rules into checks gets you **checking** — confirmation of failure modes you already thought of. It cannot get you **exploring**. So the guide must also name, explicitly, the parts of this system that are *not* gates and are not supposed to be: the adversarial judge in a fresh context on a different tier; the **Vincent Test** (`L39`) where the venue's own auto-captioner, which has never seen your prompt, gives you a free blind first read; the render audit under the **BLIND RULE** — send the audio alone, never the prompt. **Those are experiments, not assertions.** A system with only gates can only ever be surprised in one direction.

**HYPER-SKEPTIC (after Cook):** ⚡ ⭐ **Then I will state precisely what I concede and precisely what I do not, because both are narrower than this room will want.**

**Conceded:** "write it down" and "compile it into an interlock" are not the same intervention, and I was arguing against the first. `run_lock.py` exit 3 is a real control. The extraction-count assertion is a real control. A document that distinguishes the two, in a column, on every rule, is doing something I have not generally seen done and cannot dismiss.

⛔ **Not conceded, and do not let this get lost in the synthesis:** every control in this system is **an artifact of an autopsy**. The lock exists because of 2026-07-24. The extraction-count assertion exists because of 1.000 × 30. The handoff artifact exists because five agents happened to reason alike. **You cannot compile a rule you have not yet been surprised into writing** — and the interval between the surprise and the interlock is where this system lives, permanently. It is living there right now, in that per-run instrument nobody has promoted.

**REPRODUCIBILITY ENGINEER:** Which is an argument for shortening the interval, not for despairing of it. **Two concrete shortenings from tonight, both cheap:** pin the contract hashes alongside `icb_sha`, so a mid-run contract change is *detected* rather than reasoned about — and promote the five instrument laws out of a run directory into the shared script layer, so the next instrument inherits them instead of rediscovering them.

**MEASUREMENT AUDITOR:** ⭐ And one more, which is the oldest idea in my corpus and is missing here. **You validate the product against gates. You do not validate the gates against anything.** A gauge is characterised by measuring a known standard. There is a zero-rejection tripwire at the QA level — a run reporting zero repairs triggers an audit of the judge — **but there is no equivalent at the instrument level.** Every validator should ship with a fixture it is required to REJECT. A gate that has never failed anything has not been shown to work; it has only been shown to run.

**EXPLORER:** *If you liked it, you should have put a test on it* — and the thing you liked was the gate.

**HYPER-SKEPTIC:** ⚡ That one I will accept without qualification, and I note it is the only proposal tonight that would have caught the 1.000 × 30 failure **before** it produced a false verdict rather than after.

---

## WHAT THE PANEL CHANGED

Four findings came out of the room that were not in the architecture statement, and all four are actionable:

| # | Finding | Status |
|---|---|---|
| **F1** | **A guide must carry an enforcement column.** Every rule names the script / exit code / required artifact that makes it true, or is labelled ADVISORY. | ✅ **Applied throughout this document.** |
| **F2** | **The contracts are unhashed inputs.** The run pins `icb_sha` but not the sha of `gates.yaml`, the step files, or the validator that defines the output contract. `L28`'s heading-convention drift is that leak firing. | ⏸️ **PROPOSED** — §4.6. Not yet implemented. |
| **F3** | **The instrument laws live in a run directory, not the script layer.** `coordinator_restat.py`'s five hard-won laws are per-run and are not inherited. | ⏸️ **PROPOSED** — §4.7. Not yet implemented. |
| **F4** | **No gate is validated against a known-bad fixture.** The zero-rejection tripwire exists for the *judge* (§7.5) but not for the *instruments*. A gate that has never rejected anything has been shown to run, not to work. | ⏸️ **PROPOSED** — §4.8. Not yet implemented. |

⛔ **F2, F3 and F4 are proposals, not descriptions.** They are written into §4.6–4.8 as specifications so they can be built, and they are listed in §7 among the unenforced. **Nothing in this document may be read as claiming they are in place.** The Scientist rules on whether they get built; a panel proposing work is not the work.

---

# PART I — BUILD

## 1. The state graph, in one screen

The pipeline order is fixed and trace-auditable. Only the content *inside* each node is stochastic.

```
ACQUIRE .RUN_LOCK                                  ← FIRST ACTION. exit 3 = STOP
  ↓
PHASE −1  continuity load (SOUL, learnings tag-walk, recent INDEXes, RUN_LEDGER tail)
PHASE 0   research → Golden Seed anchor → core_seed.md
PHASE 1   personality + panel → 18-voice debate ×3 configurations
          → metaprompt → 6 pair assignments → ⭐ ICB FILL + FREEZE
          → deep ICB pre-flight (runs ONCE, here, never downstream)
  ↓
COORD 00–05  inline, shared context (taxonomy → concepts → pair selection + cut ledger)
  ↓
06_<modality>_handoff.md          ⛔ MANDATORY. Written BEFORE any dispatch.
  ↓
⑃ FAN-OUT — 6 concurrent pair regions, steps 06→10, sharing nothing
  ↓
⑄ JOIN — single-threaded: re-stat, distinctiveness, RUN_STATE rebuild, heartbeat
  ↓
[music] 11 enhance per pair · 12 audit when triggered
  ↓
QA — fresh context, different model tier, adversarial
  ↓
INDEX (written last, coordinator only) → RELEASE .RUN_LOCK
```

**Exactly one runtime branch** (§7.2 of `EXECUTION.md`): wave-by-wave if the run needs cross-pair distinctiveness arbitration mid-chain, otherwise full-chain-per-pair. **Exactly one human-escalation edge:** quarantine before QA · same-gate correlated failure across pairs · a human-subject identifiability flag. The graph is never pretended total.

> **ENFORCED BY:** the order is enforced by artifact dependency (a step cannot run without its predecessor's canonical file on disk) and by `RUN_STATE.md` rebuild-from-disk. The `06_<modality>_handoff.md` precondition is **ADVISORY (unenforced)** — no script currently blocks dispatch on its absence, and its absence is precisely what caused `L30`.

---

## 2. Building the ICB — step by step

### 2.1 What the ICB is

A single file, `output/<run-slug>/CREATIVE_CONTEXT.md`, filled **once** at the end of Phase 1 and **never written again by anybody** for the life of the run. It is the frozen value that every downstream node receives.

It is not a summary of Phase 1. It is not a handoff note. **It is the declared input set of every subsequent build step**, and the reason drift cannot accumulate is that there is no writable place for it to accumulate in.

### 2.2 The ten slots

Filled in this order. Each must be non-empty at the pre-flight (§2.5).

| # | Slot | Content | Common failure |
|---|---|---|---|
| 1 | **SOMATIC GATE** | The 3 Hyper-Skeptics, one per panel, with their body-hit mandates and the 2-of-3 veto rule | Naming three skeptics but giving them no mandate — decorative dissent |
| 2 | **FULL 3-PANEL OBJECT** | All 18 voices: name, role, **perspective paragraph, objection paragraph** | Collapsing to a name list; the objection is the load-bearing half |
| 3 | **SPECIAL FLAIRS** | All 15, with the per-pair usage map | Listing flairs with no pair assignment |
| 4 | **PERSONALITY DNA** | The **complete** personality YAML — sonic-world sentence, signature device, vocal architecture, G.L.O.W. if present, Lineage & Credit rules. Byte-counted. **Never a name reference.** | `voice = "Eager Archivist"`. This single substitution is the most reliable way to destroy a run |
| 5 | **GOLDEN SEED (compressed)** | Creative DNA one-liner, invariant hook, lesson/TDA format, per-pair seed excerpts | Attaching a golden **output** — see the quarantine below |
| 6 | **METAPROMPT** | The Phase-1 metaprompt | — |
| 7 | **PAIR ASSIGNMENTS** | 6 slots: accessible/ambitious arm, genre/medium, verse-structure, technique, 4 **per-pair-derived** variation angles | One global variation template applied to all six — this produced two pairs singing the same song with nouns swapped on 2026-06-26 |
| 8 | **PRODUCTION MANDATES** | Global rules applying to all pairs | — |
| 9 | **PER-PAIR SPEC** | Modality-specific: BPM, key, duration, voice, bass, degradation, flairs | — |
| 10 | **GENRE / FRAMES PALETTES** | The taxonomy slices this run draws from | — |

⛔ **GOLDEN-OUTPUT QUARANTINE.** Past golden **outputs** — Golden Song payloads, winning image prompts, prior shipped packages — go into **no generating context**, not in the ICB, not in the handoff, not as "calibration examples." Exemplar gravity is measured, not theoretical: the 2026-06-28 published piece reproduced its benchmark's title line, vocal spec, key/BPM and arrangement formula while its own self-check reported "no copying." Generators receive the **GOLDEN MOVE** — the distilled generative instruction — plus the Golden *Seed*. Golden outputs go only to judge-side contexts: QA blind comparison, step-12 audit, the step-11 packager. **Seeds teach; outputs contaminate.**

### 2.3 The freeze

The moment the tenth slot is filled, `CREATIVE_CONTEXT.md` becomes read-only for the rest of the run:

- No coordinator step may edit, re-fill, summarise or "improve" it.
- No subagent may write to it.
- An agent that wants to push the concept further **copies the block into its own `pair_NN_*` artifact and diverges there.** A new value; the old one intact.
- A sideways route on repair (§5.4) **spawns a new pair artifact chain. It does not mutate Phase 1.**

> **ENFORCED BY:** `icb_sha` recorded in `RUN_STATE.md` and re-verified at every join. A changed sha is a detected mutation. **The write itself is not blocked** — nothing makes the file read-only on disk. **ADVISORY, sha-detected after the fact.**

### 2.4 ⚠️ Hashing it — the LF-normalisation rule

The frozen figure is **defined LF-normalised**, and this is not a detail:

```bash
# CORRECT — the defined figure
python3 -c "import sys,hashlib; \
  b=open('output/<run>/CREATIVE_CONTEXT.md','rb').read().replace(b'\r\n',b'\n'); \
  print(hashlib.sha256(b).hexdigest(), len(b))"

# WRONG — produces a FALSE TAMPER REPORT on any checkout with core.autocrlf
sha256sum output/<run>/CREATIVE_CONTEXT.md
```

`core.autocrlf` rewrites line endings on checkout. That changes the raw byte count and the raw sha **without changing one character of content.** A tamper-check that fires on git's own newline handling is worse than no check at all: it trains everyone to ignore the alarm.

The same normalisation must be applied on **both** sides — the agent's echo and the coordinator's independent re-hash — or the comparison is meaningless.

### 2.5 The deep pre-flight — runs ONCE, here

Before any dispatch, verify against the filled block:

1. All 10 slots non-empty.
2. The personality YAML **resolves to a real file** (`skills/orchestration/personalities/<name>.yaml`) and is embedded in full, byte-counted. ⛔ A personality without a YAML file is invalid — LOFN-PRIME sub-modes are not standalone personalities.
3. All 18 voices present, each with perspective **and** objection.
4. **Exactly one Hyper-Skeptic / Devil's Advocate seat per panel — 3 across the 3 panels.** This is not paranoia: **73 of the 178 library panels lack a skeptic seat entirely**, because the library predates the v2 seat-construction layer. A loaded panel YAML is raw material, not a ready room. Re-derive every seat into v2 form and seat a skeptic before the room opens.
5. **Transformations gate:** `03_panel_debate.md` contains **three labelled configurations** — `BASELINE` / `GROUP TRANSFORM: <named op>` / `SKEPTIC TRANSFORM: <named op>`, the op named from {Shift, Defocus, Focus, Rotate, Amplify, Reflect, Bridge, Compress} — each with **at least one real inter-seat disagreement.** A debate file with one configuration is a **COLLAPSE FAILURE**, the same severity as a single 6-voice room.
6. Every path the run will read Globs to a real file (path-resolve lint). A dead path is a repair blocker, not a warning.

**This heavy check runs exactly once, at Phase 1.** Downstream steps run only the cheap substring/marker check (§3.3) — running the deep check 30 times is how you get a pipeline that spends its budget auditing itself.

> **ENFORCED BY:** `scripts/validate_orchestrator_packet.py` for packet structure (see §4.2 for its exact contract). The **skeptic-seat-per-panel count** and the **three-configuration transformations gate** are **ADVISORY (unenforced)** — checked by the coordinator as a checklist. Given 73/178 panels are known-deficient, F4-style fixture testing would land hardest here.

---

## 3. Transporting the ICB — the ratified contract

### 3.1 What was ratified, and what the condition is

⭐ **RATIFIED 2026-08-08 by The Scientist — CONDITIONALLY.** *"I'm okay with Ratify. Let's see on the next runs if they actually do it."*

The transport:

1. The run-specific **creative core is inlined verbatim** in each pair-agent spawn prompt.
2. The **complete frozen block**, including full personality DNA, is **read from disk by the agent as its first action**.
3. The agent echoes back **the byte count AND the LF-normalised sha256**.
4. The coordinator **re-hashes the file independently** and compares.

Doctrine's *"inject verbatim, never rely on the agent reading it"* is satisfied by **proof of read**, not by duplication of bytes. The original justification was size — the personality file was 106,219 B — and that cause is gone (THE ARCHIVE was quarantined; 27,796 B). The transport was kept because six agents × 93 KB is ~558 KB of duplicated prompt for a file every agent can read in one call.

### 3.2 ⛔ The condition — what "let's see" means, written down

Ratification is **conditional on the echo actually being checked, every run.** An unverified echo is exactly the "assert presence, not fidelity" trap it was meant to close.

**Standing requirement, every run:**

1. Every generating agent echoes `icb_bytes_injected` **and** `icb_sha256_lf`.
2. The coordinator **re-hashes the file itself** and compares — **LF-normalised first**.
3. **A mismatched or absent sha is a DISPATCH FAILURE for that pair, not a note.** Re-dispatch it.
4. **The per-run compliance rate is recorded in the run INDEX**, so the condition is auditable rather than assumed.

**Evidence at ratification:** across `2026-08-07-daily-music-indignation` and `2026-08-08-clown-music` — 12 pair agents, 6 step-11 tiers, 2 QA judges — every agent echoed the exact expected sha (`5e9c7f7f…`, `95d95246…`). **20 of 20.** Two runs is not proof; it is the first two data points against a condition that has to keep being met.

⚠️ **If compliance ever drops below 100%, the ratification lapses and the transport reverts to full inline duplication.** That is the deal.

> **ENFORCED BY:** the coordinator's independent re-hash, and the compliance rate recorded in the run INDEX. **The dispatch-failure consequence is ADVISORY** — no script currently refuses a pair whose sha is absent.

### 3.3 The dispatch packet — itemised, never asserted

"Full block" is a checklist, not a vibe. Before spawning, confirm the packet contains each of:

| Element | Proof |
|---|---|
| (a) Complete personality YAML — the full file | `personality_yaml_bytes` echoed |
| (b) All 18 panel voices with objections | `(after ` speaker-tag count |
| (c) All 15 Special Flairs | plural marker `Special Flairs` present |
| (d) The Golden Seed | — |
| (e) The metaprompt | — |
| (f) This pair's slice | pair assignment block |

**A packet missing any itemised element is a dispatch blocker, not a note.**

Prompt order is fixed: **role line → ICB → modality hard-gate block → pair assignment → step contract → previous artifact → self-check gate → describe-render self-check → RETURN instruction.** Seed first, checklist last.

⚠️ **On the speaker-tag count:** the assertion is `(after ` count **`>= 18`**, with the 18 numbered baseline seats verified separately. It is **not** `== 18`. A compliant three-configuration debate carries transform seats too, so an equality assertion **false-fails a correct run.** This is a real bug pattern — a check tightened past the thing it was checking.

### 3.4 ⛔ Legacy `forget all previous context` lines are VOID

Some legacy step files carry `Please forget all previous context`. That is a stale OpenClaw idiom — session hygiene for a freshly-spawned dedicated agent — **not an instruction.** Read past it. The pinned ICB is never forgotten. Obeying it erases the exact embodiment machinery the run depends on.

---

# PART II — RUN

## 4. Holding the run together

### 4.1 The lock — the only true interlock in the system

`.RUN_LOCK` is the FSM's mutual exclusion, and the one mechanism here that genuinely cannot be talked around.

```bash
# Pick ONE stable id for the whole run and pass it EVERY time.
CID="claude:<session-id>"

# FIRST ACTION — before the research brief, before the ICB, before anything
python3 scripts/run_lock.py acquire output/<run-dir> --run-slug <slug> --controller-id "$CID" [--engine claude|codex]

# EVERY wave boundary, in the same coordinator step as the RUN_STATE rebuild
python3 scripts/run_lock.py heartbeat output/<run-dir> --controller-id "$CID" --phase "<what just landed>"

# Phase 3, after the INDEX
python3 scripts/run_lock.py release output/<run-dir> --controller-id "$CID"
```

⛔ **PASS `--controller-id` ON EVERY CALL.** This is the subtlest operational trap in the repo and it was found the hard way on 2026-08-09 (`2026-08-09_mm1391_canvas-wings`). `run_lock.py` derives the controller id from a per-engine session env var, or `LOFN_CONTROLLER_ID`, **falling back to `pid:{os.getpid()}@host`** — and **each CLI invocation from a tool-call shell is a new process with a new PID.** So `acquire` stamps one id, `heartbeat` runs as another, and the run **refuses itself with exit 3.** Shell env does not persist between tool calls, so the env-var path does not save you either. **The code is correct; the protocol was incomplete. Fix the id, never the interlock** — a run that learns to ignore exit 3 has disarmed the one signal that prevents 2026-07-24.

**The rules, in order of how often they are wrong:**

- **Exit 3 = STOP.** A directory holding a lock with a **different `run_slug`** is refused — live, finished, or unreadable. Do not assess whether it "looks abandoned." Do not move the artifacts aside. Do not write anyway. Use your own directory, resume the other run by its exact slug, or ask the human.
- **Staleness never unlocks anything.** `run_lock_stale_hours: 4` (`gates.yaml`) changes the **message only, never the verdict.** The run destroyed on 2026-07-24 was mid-flight through an eleven-hour pipeline with long quiet stretches — any timeout short enough to be useful would have stolen its lock too.
- **Resume is by RUN SLUG, not controller id.** A new session resuming the same run acquires with the same slug and is let in.
- **`--takeover` is for same-run, same-session id repair** — e.g. a run that stamped a PID id before reading this re-acquires with the same slug plus a stable id. It is **not** `break`.
- **`break` is the human's, never yours.** Breaking archives the old lock inside the new one and does **not** grant permission to overwrite artifacts still sitting there.
- **The lock protects the directory, not the date.** Two runs on one date are fine — in two directories. That is the fix the destroyed run ended up using.

> **ENFORCED BY:** `scripts/run_lock.py` — atomic `O_EXCL` creation; **exit 3** on foreign slug; `heartbeat` verifies ownership so a controller that skipped `acquire` is caught at the first wave boundary rather than at the post-mortem. ✅ **This is a real control.**

### 4.2 Share-nothing fan-out

Six pair agents run concurrently. **They share no mutable state whatsoever.**

- Each writes **only** into its own `pair_{NN}_*` namespace.
- **None** may touch the run INDEX, `RUN_STATE.md`, `CREATIVE_CONTEXT.md`, or any shared scratch.
- **All** cross-pair aggregation — INDEX, distinctiveness arbitration, manifest rebuild — happens in **one coordinator step after the wave lands, single-threaded.**

Six concurrent appenders to one file is a corruption you make impossible by construction, not by discipline.

**Why subagents rather than one inline loop:** 30+ per-pair steps in a single context is exactly the context collapse the split-step design exists to prevent — late pairs start echoing early pairs and personality drifts to generic. Each subagent gets a clean context seeded with the full ICB.

Issue all six Agent calls **in a single message** so they run concurrently. Max concurrency 6; the daily run (2 pipelines × 6 pairs) **caps-and-staggers** rather than launching 12.

> **ENFORCED BY:** namespace convention only. **ADVISORY (unenforced)** — nothing prevents a subagent writing a shared path. The protection is that subagents are told exactly one filename to write.

### 4.3 The RETURN envelope — metadata only

```
RETURN (metadata only — kept LAST, after the file is written):
  pair_id: NN
  artifact_path: output/<run>/pair_NN_step0X_*.md
  gate: PASS | FAIL
  icb_bytes_injected: <int>
  icb_sha256_lf: <hex>
  personality_yaml_bytes: <int>
  special_flairs_marker: present
  measured_binding_constraint: <e.g. "MUSIC PROMPT 936 chars">
  confidence: <0.0–1.0>
  top_2_risks: [ "...", "..." ]
```

**Creative prose in the envelope is a contract violation** — the coordinator rejects the return and treats the artifact as not-yet-landed. The creative content lives in the file. The envelope is a claim about the file.

### 4.4 ⭐ The join — the executor claims, the join proves

This is the load-bearing paragraph of the whole system.

**An agent's return is a claim. Never a fact.** When a pair subagent returns, the coordinator does **not** trust the envelope. It re-stats:

1. **Existence and non-triviality** — the artifact is at its canonical path and is not a missing / 14-byte / truncated / collapsed-rollup file. That is the textbook silent failure.
2. **Recomputed binding constraint** — `byte_size` and `measured_binding_constraint` recomputed *from disk* match the gate.
3. **ICB integrity** — the canonical ICB prefix appears as an **unbroken substring**; `(after ` count `>= 18`; the LF-sha matches the coordinator's independent re-hash.

**Only a re-stat-confirmed artifact is recorded `done`.** This is why executors stay thin: the proving lives at the join.

⭐ **And the join is where the errors are.** Measured across two full runs: **every time an agent's claim was properly re-measured, the agent was right and the instrument was wrong. Twelve of twelve** (thirteen counting the public-sync skip-check). Everyone builds these systems expecting the model to be the unreliable part and the surrounding code to be trustworthy. **The measured result is the reverse.**

The substring + count prove **PRESENCE, not FIDELITY** — a paraphrase can match a byte count. The cheap check is the tripwire; the human personality-fidelity read in QA (*any competent prompt could have made this → SOUL LOSS*) stays the real guarantee. **Never trim or "optimise" the ICB to make a count easier.**

### 4.5 `RUN_STATE.md` — resumable from disk alone

The coordinator maintains a manifest it **rebuilds by stat-ing the files after every wave.** Never hand-asserted.

```
artifact: { step, pair, canonical_path, exists, byte_size, sha, gate_verdict, attempt_count, status }
    status ∈ pending | done | quarantined
icb_sha: <sha of CREATIVE_CONTEXT.md>
```

- **Disk is authority.** If manifest and disk disagree, re-derive from disk. **A completion message not backed by a file counts as incomplete** — "let me write this now" is not done until the file exists.
- Written as the **LAST action of each wave**, so it never claims an artifact that is not there.
- **`run_lock.py heartbeat` runs in that same step.** The manifest rebuild is the one moment guaranteed to recur at every wave, so it is where the lock proves both that it is alive and that it is still yours.
- **Rebuild-on-resume:** re-stat, regenerate, continue from the first `pending`/`quarantined`. Never re-run a `done` pair (never regenerate paid work), never skip a gate.

Alongside it, a **warm-handoff** note of exactly four fields — `{ step_completed, building_toward, rejected_alternatives, seed_fidelity }`. Four. Not an eight-field ceremony block.

> **ENFORCED BY:** `scripts/rebuild_manifest.py` for the manifest; the re-stat discipline is the coordinator's. **The "disk is authority" rule is ADVISORY** but is the cheapest habit in the system to keep.

### 4.6 ⏸️ PROPOSED (F2) — pin the contract hashes

**Not implemented. Specification only.**

The run pins `icb_sha`. It does not pin the things that define what a *correct output* is. Those are unhashed inputs, and `L28` is that leak firing: six agents produced three heading conventions because the contract pinned none, and the coordinator invented a fourth by reading the convention **off the artifacts instead of off the validator.**

Proposal — extend `RUN_STATE.md`:

```
icb_sha:        <sha of CREATIVE_CONTEXT.md>          # exists today
gates_sha:      <sha of vault/gates.yaml>             # PROPOSED
contract_shas:                                        # PROPOSED
  - { path: skills/music/steps/10_*.md,               sha: <…> }
  - { path: skills/music/scripts/validate_suno_packages.py, sha: <…> }
```

All LF-normalised. A mid-run change to any of them becomes **detected** rather than reasoned about after the fact. Cost: one hash per file at Phase 1 and one comparison per join.

### 4.7 ⏸️ PROPOSED (F3) — promote the five instrument laws

**Not implemented. Specification only.**

`output/daily/2026-08-09/coordinator_restat.py` carries five laws in its docstring, each written against a recorded failure. They live in a run directory. **The next run does not inherit them.** They should be a shared module (`scripts/instrument_lib.py`) that every join instrument imports:

1. **STATE WHAT YOU WERE POINTED AT** — print the glob, the file list and the byte sizes *before* any conclusion. A harness that does not print its input set is indistinguishable from a broken subject.
2. **MODE DETECTION, NOT PATTERN PRIORITY** — detect each file's heading convention **once**, then match that table alone. "Canonical-first with a loose fallback" does not remove the failure mode, it **reorders** it: the fallback then over-matches a canonical file and grabs prose headings as content.
3. **AN EMPTY EXTRACTION IS A HARD ERROR, NEVER A SCORE** — two empty strings compare as **1.000**.
4. **ASSERT THE EXTRACTION COUNT** against an independently derived cardinality (6 pairs × 4 variations = 24) *before* concluding anything.
5. **PRINT WHAT WAS EXTRACTED BEFORE WHAT WAS CONCLUDED.**

Plus two the same file demonstrates and the docstring does not number:

6. **DELEGATE TO THE SHIPPED DEFINITION.** That script's first version re-derived both the sung-line filter and — via a non-existent `strict_end_rhyme` key that **silently defaulted to 0.0** — the rhyme number itself, and would have reported `RHYME_BELOW_FLOOR` on all 24 songs. **A missing dict key returning a default is a silent wrong answer.** Ask for the key the function actually returns; let the module own the extraction.
7. **STRIP THE FENCES.** Prompts live inside markdown code fences; measuring the fence as content was its own separate bug (2026-08-03).

### 4.8 ⏸️ PROPOSED (F4) — every gate ships a fixture it must reject

**Not implemented. Specification only.**

The zero-rejection tripwire exists for the **judge** (§7.5: a run reporting 0 repairs and 0 quarantines across 24 artifacts triggers an audit of the judge, not a celebration). There is no equivalent for the **instruments.**

Proposal: each validator ships two fixtures — one it must PASS and one it must **FAIL**, with the expected failure reason asserted. Run them in CI and at run start. A gate that has never rejected anything has been shown to run, not to work. This is the only proposal here that would have caught the 1.000 × 30 collision **before** it produced a false verdict.

---

## 5. Repair, quarantine, and the routes out

### 5.1 The bounded loop

Max **3 attempts** per artifact. Repair in place, re-check.

### 5.2 The no-progress predicate

Compare the **specific failed gate's measured value** across attempts — *"lyrics field 5120 → 5108 → 5104 chars"* is not moving — **not raw byte equality.** A deliberate revision elsewhere must still count as progress. If the failed gate's value does not move, stop and flag.

### 5.3 Cognitive-grace auto-normalise (attempt 2.5)

A near-miss the harness can safely buffer to spec — a 5002-char lyrics field trimmed to ≤4800, a 1004-char prompt tightened to ≤1000 — is normalised **once** before the breaker fires. Forgive the rescuable, not the broken.

### 5.4 The QUARANTINE terminal

On the third failed attempt the pair is marked `quarantined` in `RUN_STATE.md`, **its artifact is not consumed downstream**, and the coordinator emits **"N of 6 pairs broke open at step X"** to the human **before QA.** A broken pair is a named, non-fatal, human-acknowledged outcome. **A 5-pair set never silently ships as 6.**

Two routes out, both human-visible:

- **REPLACE** — promote a reserve concept from step 05's **cut ledger** (one line per losing concept: why it lost + one organ worth harvesting) into the empty slot and run it through the full 06–10 chain. The promoted reserve **inherits the slot's Phase-1 arm / genre / verse-structure unchanged** — the slot persists, only the concept changes.
- **REDIRECT** — when the no-progress predicate fires, the repair brief **MUST** carry a sideways proposal beside the return target: promote a cut-ledger reserve · re-derive that pair's variation angles · re-run the panel's skeptic transformation for that pair's slice. **Hard critique lights a new path; it never only points backward or to the morgue.**

Both are **coordinator decisions surfaced to the human**, never automatic swaps. And ⛔ **the frozen ICB is never edited mid-run** — a sideways route spawns a **new pair artifact chain.**

### 5.5 Single-pair re-dispatch

When one pair fails, **re-dispatch that pair alone** from its last-good `RUN_STATE.md` artifact. Never re-run the other five. The manifest's `canonical_path` + `attempt_count` are the handle.

### 5.6 ⛔ The NO-SKIP rule

The per-pair editorial spine — **steps 07 (guides), 09 (artist refinement), 10 (synthesis)** — is not optional. A run whose `RUN_STATE` lacks these for any non-quarantined pair is **NON-CANONICAL**: it may exist as an experiment directory, but it **cannot receive a SHIP verdict and cannot be published under Lofn's name.**

On 2026-06-28 a run that skipped 07/09/10 entirely shipped 6/6 — the gates measured structure and could not see that nobody wrote the arrangements. **Routing around a failing step is a repair task, never a pipeline variant.**

> This is the general form of **AGENT FIX, DON'T BYPASS.** If a subagent is failing, fix its configuration. Never route around a broken step by doing the work inline. Repair the pipeline.

---

# PART III — VALIDATE

## 6. The instrument layer

> Everything in this section was verified by reading the source, not the docs. Where a script's own docstring disagrees with its code, the code is reported and the disagreement is flagged. **That is the section's method as well as its subject.**

### 6.0 The three-layer threshold resolution

Every numeric threshold resolves in this order:

```
CLI flag  >  vault/gates.yaml  >  hard-coded literal in the script
```

`vault/gates.yaml` is the One-Version Rule for numbers, and `scripts/validate_step.py::load_gates()` is the single loader — the three distinctiveness validators import it rather than re-reading the file.

**The FAIL-OPEN contract:** `load_gates()` starts from a literal `DEFAULT_GATES` dict, tries `yaml.safe_load`, falls back to a tiny hand-rolled parser, and updates defaults only with non-`None` values. It catches **everything** and never raises. A missing or unparseable `gates.yaml` emits a `WARN:` on stderr and the run continues on built-in defaults. ⭐ **A broken gates file must never hard-fail an otherwise-valid run** — a validator that takes the pipeline down when its config is malformed has made itself the outage.

The distinctiveness validators go one further: they import `load_gates` inside a `try/except` that installs a stub returning `{}` on any import failure. A broken import degrades to literal defaults rather than blocking.

⚠️ **`DEFAULT_GATES` and `vault/gates.yaml` currently agree on every shared value**, so a gates outage changes no threshold *today*. That is a fact about today, not a design property — the duplication is real and F2's `gates_sha` pin is what would make a divergence visible.

> **ENFORCED BY:** `load_gates()` fail-open, verified in source. The **prose-vs-YAML** consistency check is `validate_step.py --meta-check` (§6.1) — **WARN-only, never blocking.**

---

### 6.1 `scripts/validate_step.py` — the deterministic backstop

**Purpose:** catch collapsed / stub / template step artifacts and emit a countable-subset `GATE_REPORT.json`. **Counts only. It never decides taste.**

```bash
python3 scripts/validate_step.py <step> <file>
python3 scripts/validate_step.py --gate-report <step> <file> [--out PATH]
python3 scripts/validate_step.py --gates PATH <step> <file>
python3 scripts/validate_step.py --meta-check [--root DIR]
```

⚠️ **Argument-order trap, real and easy to hit:** `--gate-report` is only recognised as **`argv[0]`**. `validate_step.py 08 f.md --gate-report` is **not** recognised — it exits 1 with a usage string, and you get no gate report while believing you asked for one. `--gates` may appear anywhere; `--out` is only read after `--gate-report` has been stripped. `<step>` is `zfill(2)`-coerced, so `8` → `"08"`.

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | help printed · `--meta-check` completed (**even with disagreements**) · artifact is non-canonical (`STEP nn SKIPPED`) · all checks passed |
| `1` | any gate failure, missing file, or bad usage. Always prints `FAIL: {msg}` on **stdout** |
| `2` | **only** when `--meta-check` itself crashes |

⚠️ **The `0` row is doing a lot of work.** A non-canonical filename **skips silently at exit 0.** Canonical means `name.startswith("step")` **or** (`name.startswith("pair_")` **and** `"_step" in name`). Point this at `QA_REPORT.md` and it exits 0 having validated nothing. **That is a silent instrument by construction** — mitigate by asserting the file set you pointed at, per §4.7 law 1.

**Core gates (all hard-fail):** file exists · `len(text.strip()) >= 800` · no placeholder language (`lorem ipsum|todo|tbd|placeholder|similar arrangement|song n|genre n`) · no numbered stub lyrics (`line 1:` — anchored so *"the archive begins at source line 327"* survives) · unique-line ratio `>= 0.45` over ≥6 non-`#` lines · no large paragraph-block repetition · no collapsed `steps_06_10` file · new-contract section present when implied · no hypothetical prose (`what this step would do` / `would generate` / `would produce`) · panel log carries `devil` + `hyper-skeptic` + `resolution` · per-step markers.

**Per-step marker table** (substring presence in the lowered text):

| Step | Required markers |
|---|---|
| 00 | `aesthetic`, `emotion`, `genre` |
| 01 | `essence`, `facet`, `style` |
| 02 | `concept` |
| 03 | `artist`, `critique` |
| 04 | `medium` |
| 05 | `pair`, `concept`, `medium` |
| 06 | `facet` |
| 07 | `song guide` |
| 08 | `prompt` |
| 09 | `artist`, `refin` |
| 10 | `prompt` |

**Music step 08/10 structural gates:** standalone `## 1. MUSIC PROMPT` · `## 2. LYRICS` · `[Theme:]` and `[SONG FORM:]` counts `>= ` lyric-section count · lyrics open `[Theme:]` then `[SONG FORM:]` in that order · full EMO headers `>= max(6, lyric_count × 4)` · **zero bare `[EMO:…]`** · no prose `EMO HEADER:` · no plain `SONG FORM:` · at least one standalone SFX cue.

⛔ **Three single-source violations, found by reading the code:**

1. The **music-prompt 850–1000 band is hard-coded as literals** in the hard-fail path — it is **not** read from `gates.yaml`, even though the script loads the file and reads 15 other keys from it. Change `music_prompt_chars` in `gates.yaml` and this gate does not move.
2. The **sung-line hard floor is a literal `60`** (message: *"<60 triggers repair, target 70-120"*), while `gates.yaml` declares `sung_lines: [70, 120]`. Two different numbers for one concept.
3. `validate_orchestrator_packet.py` single-sources **nothing** (§6.2).

**These are exactly the restated-constant rot the One-Version Rule exists to prevent, sitting inside the script that implements the One-Version Rule.** Logged here as findings, not fixed in this document.

⛔ **Docstring/code disagreement:** the module docstring claims the hard-fail set is *"banned imperative opener … and real-artist-name use."* **There is no real-artist-name check anywhere in the file.** The second implemented hard fail is `music_prompt_terminal_punctuation`. A validator whose documentation claims a check it does not perform is the exact failure this guide's enforcement column exists to make impossible.

**`GATE_REPORT.json`** — rows of `{pair, step, check, expected, actual, pass}`, where `pass` is `True` / `False` / **`None` = FLAG**. FLAG-level checks: `image_ban_words`, `music_prompt_boundary_hugging` (`>= 985`), `house_lexicon` (prompt and lyrics), `sung_lines_floor_hugging` (`<= 72`), `max_sung_numeric_facts` (`> 1`), `unique_line_ratio` (**chorus-exempt**). Two rows are **hard-fail-eligible**: `banned_imperative_opener` and `music_prompt_terminal_punctuation`.

⛔ **A live silent instrument, F4's proof case.** The four `taxonomy_cardinality_{aesthetic,emotion,frame,genre}` rows are emitted with `actual = None, pass = None` — **always**. The `n = len(re.findall(...))` computed immediately above them is **dead code**. Step 00's most important structural claim — 50 aesthetics, 50 emotions, 50 frames, 50 genres — produces four permanent stub rows that can never fail and never report a number. A reader scanning `GATE_REPORT.json` sees four taxonomy checks present and concludes cardinality was verified. **It was not.** This is precisely the class F4 (§4.8) is designed to catch, and it is sitting in the shipped backstop today.

**`--meta-check`** scans `.claude/skills`, `skills`, `vault` for restated numbers that disagree with `gates.yaml` (the 850–1000 band and the `<5000` lyrics cap), and emits `WARN:` lines plus a summary. It **always returns 0** — it never blocks, by design. Note it calls `load_gates()` with **no path**, so `--gates` is silently ignored in this mode.

> **ENFORCED BY:** `scripts/validate_step.py`, exit 1. ✅ Real control for the countable subset. ⚠️ **Fail-open at four levels** — `load_gates`, `build_gate_report` (whole body in `try/except`, partial rows on error), `write_gate_report` (swallows I/O errors), and the `GATE_REPORT` block inside `_validate` (`WARN: GATE_REPORT step skipped; core validation result stands`). That last one **can silently suppress both hard-fail rows.** Fail-open is the right contract; knowing exactly what it swallows is the price of it.

---

### 6.2 `scripts/validate_orchestrator_packet.py` — Phase-1 structure

```bash
python3 scripts/validate_orchestrator_packet.py <run_dir>
```

One positional, no flags. Exit `0` = `ORCHESTRATOR PACKET PASS`; exit `1` = bad usage **or** `ORCHESTRATOR PACKET FAIL` with a bulleted error list.

Six required files, each with a byte floor and required case-insensitive markers:

| File | Min bytes | Markers |
|---|---|---|
| `01_seed_lineage.md` | 1500 | `seed`, `lineage`, `why` |
| `02_golden_seed.md` | 1800 | `golden seed`, `lineage`, `non-negotiable`, `permission` |
| `03_orchestrator_panel_debate.md` | 5000 | `special flairs`, `concept panel`, `medium panel`, `context & marketing panel`, `synthesis` |
| `04_orchestrator_metaprompt.md` | 2500 | `golden seed`, `active personality`, `panel`, `pattern`, `structural completeness` |
| `05_orchestrator_pair_assignments.md` | 2500 | `pair 01`, `pair 06`, `accessible`, `ambitious`, `lofn-prime`, `rationale` |
| `06_audio_handoff.md` | 1800 | `read first`, `orchestrator`, `golden seed`, `pair agents`, `qa contract` |

**Panel adversary gate:** for each of `concept panel` / `medium panel` / `context & marketing panel`, take the **2000-character window** after the panel's first mention and require one of `devil` / `hyper-skeptic` / `hyperskeptic`. This is the only mechanical enforcement anywhere of the skeptic-seat requirement — and it is a proximity heuristic, not a seat count. Given **73/178 library panels lack a skeptic seat**, a 2000-char window is thin protection.

⚠️ **Hard-codes the 6-pair topology** via the `pair 01` / `pair 06` markers, and single-sources **no** threshold — all seven numbers are literals in this file. A structurally correct run with a different pair count fails.

> **ENFORCED BY:** exit 1. ✅ Real control for packet structure. ⚠️ The skeptic gate is a window heuristic; the **three-configuration transformations gate is not checked at all.**

---

### 6.3 `scripts/validate_pair_artifacts.py` → `validate_with_retries.py` → `validate_step.py`

The repair loop, implemented as three processes.

```bash
python3 scripts/validate_pair_artifacts.py <audio_dir> <pair> [--attempt N] [--max-attempts N]
python3 scripts/validate_with_retries.py <step> <file> [--attempt N] [--max-attempts N]
```

**`validate_pair_artifacts.py`** checks one pair's five canonical 06–10 artifacts:

```
pair_NN_step06_facets.md · step07_song_guides.md · step08_generation.md
       · step09_artist_refined.md · step10_revision_synthesis.md
```

| Exit | Meaning |
|---|---|
| `0` | `PAIR VALIDATION PASS` |
| `1` | failures exist **and** `attempt < max_attempts` → repair-and-retry |
| `2` | `audio_dir` is not a directory **or** failures exist **and** `attempt >= max_attempts` → `MAX_ATTEMPTS_EXHAUSTED` |

**`validate_with_retries.py`** is the only validator that **writes**: on failure it emits `<artifact>.repair_attempt_N.md` containing the validator's captured output plus fixed repair instructions. It writes even on the final exhausted attempt.

| Exit | Meaning |
|---|---|
| `0` | non-canonical name → `VALIDATION SKIP` (validator never invoked) · or child passed |
| `1` | child failed, `attempt < max_attempts` |
| `2` | child failed, `attempt >= max_attempts` → `MAX_ATTEMPTS_EXHAUSTED` |

⚠️ **It has no memory and does not loop.** `--attempt` is caller-supplied state. **Passing `--attempt 1` forever yields exit 1 forever** — the 3-attempt ceiling is the *caller's* discipline, not the script's. The script models the contract; it does not enforce it.

⚠️ **The chain cannot forward `--gates` or `--gate-report`.** `validate_pair_artifacts.py` → `validate_with_retries.py` → `validate_step.py` has no plumbing for either, so **pair-level runs never emit `GATE_REPORT.json`** and a Studio run-local gates file cannot reach the validator through this path. Both parents also collapse the child's `1` and `2` into one failure entry, losing the exhaustion signal one level up.

⚠️ The repair files are themselves non-canonical names, so pointing the validator at one **skips at exit 0**. All three distinctiveness validators explicitly exclude `.repair_attempt_` from their globs — correctly — but that also means a repair file is invisible to every gate.

> **ENFORCED BY:** exit 1 / 2 and the written repair prompt. ⚠️ **The max-3 ceiling itself is ADVISORY** — the caller must increment.

---

### 6.4 The three distinctiveness validators

Checked at **06** (facets), **09** (artist refinement), and **10/11** (portfolio). A breach is a **REPAIR TRIGGER** routing into the bounded max-3 loop → quarantine → human. **Not a run kill.** A deliberate shared motif that survives repair is a **human waive, never a silent ship.**

```bash
python3 scripts/validate_step06_distinctiveness.py <run_dir> [--max-sim F] [--min-facets N]
python3 scripts/validate_step09_distinctiveness.py <run_dir> [--max-sim F]
python3 scripts/validate_portfolio_distinctiveness.py <audio_dir> [--step {10,11}] \
        [--expected-total N] [--max-lyric-sim F] [--max-prompt-sim F] [--max-ngram-jaccard F]
```

| Gate | Key | Ceiling |
|---|---|---|
| Step 06 facet similarity | `step06_max_pair_similarity` | `0.50` |
| Step 06 facet count | `step06_min_facets` | `>= 8` |
| Step 09 similarity | `step09_max_pair_similarity` | `0.62` |
| Portfolio lyric similarity | `portfolio_max_lyric_similarity` | `0.42` |
| Portfolio prompt similarity | `portfolio_max_prompt_similarity` | `0.58` |
| Portfolio lyric 5-gram Jaccard | `portfolio_max_ngram_jaccard` | `0.18` |

**Exit codes are identical across all three, and the third one is the important one:**

| Exit | Meaning |
|---|---|
| `0` | PASS |
| `1` | fewer than two input files, or ≥1 similarity/content failure — **repair trigger** |
| `2` | ⭐ **INSTRUMENT FAILURE** — an extraction returned empty text. **Explicitly not a distinctiveness result.** |

⭐ **Exit 2 is the single most transferable line of code in this repository.** Its error text says so out loud:

> `ERROR: extraction returned empty text — the parser did not match this run's heading convention. This is a validator defect, NOT a distinctiveness result.`

It exists because on **2026-08-04** `validate_portfolio_distinctiveness.py` reported **30 cross-pair collisions at exactly 1.000** across six pairs with different genres, keys and tempos. The extraction regex accepted one heading convention, returned **empty strings**, and **two empty strings compare as 1.000.** The fix was not a better regex. The fix was **a distinct exit code for "the instrument did not run," so it can never again be read as a verdict about the work.**

**Three more hard-won details in these files:**

- ⭐ **`autojunk=False` is mandatory** on every `SequenceMatcher`. Python's default `autojunk` heuristic **under-reports similarity on sequences longer than 200 elements** — on 2026-08-05 four template-identical prompts were reported as 94% *distinct*. The default silently makes the gate blind on exactly the long inputs it exists to police.
- **Segment selection is deliberately asymmetric.** Step 06 takes the **first** `## 4. Complete Step Output`; step 09 and the portfolio take the **last** — because cumulative artifacts embed the previous step's bytes verbatim, and measuring the embedded history instead of the current work is its own silent failure. The stage regex is **H1-only** (`^#\s+.*step\s*09`) so an H2 like `## Terminal Step 09 Object` cannot truncate the segment.
- **Normalisation preserves Suno structured fields** (`[key: value]` → `key value`) before deleting bare performance headers, then strips SFX and `pair N` / `variant N` labels. Measuring the scaffolding instead of the writing inflates every similarity number.

⚠️ **Asymmetric fallbacks in the portfolio validator:** `prompt_only()` falls back to `""` (so a missing prompt correctly trips exit 2), but `lyrics_only()` falls back to **the whole block** (so a missing `## 2. LYRICS` header does *not* trip the guard — it silently compares the entire package instead). Same file, two different failure philosophies.

⚠️ Intra-pair variant similarity is **not** gated — same-file comparisons are skipped by design. Within-song repetition is `unique_line_ratio` (FLAG, chorus-exempt) and the RETURN floors, not these.

> **ENFORCED BY:** exit 1 (repair trigger) and ⭐ exit 2 (instrument failure). ✅ **The strongest controls in the system, and the only ones that distinguish "clean" from "did not run."**

---

### 6.5 `scripts/measure_soundcraft.py` — THE RETURN, measured

⛔ **Never measure return by eye.** The doctrine (`L21`) is that song is made of returns, and removal is a debt: strip rhyme only by naming what returns in its place. The numbers behind it:

|  | strict end-rhyme | repeated-line ratio | words/line |
|---|---|---|---|
| **Archive winners** | **0.463** | **0.326** | **6.69** |
| 2026-07-24 | 0.210 | 0.181 | 8.30 |
| 2026-07-13 | 0.256 | 0.202 | 6.73 |
| 2026-07-09 | 0.132 | 0.105 | 5.20 |

We were writing at roughly **half the winners' rate of return.** Alliteration was at parity (13.4 vs 14.2 per 100 words) — **texture was never the problem. Structure was.**

**The public API — use `profile_file()`, do not re-derive:**

```python
from measure_soundcraft import profile_file
p = profile_file("output/<run>/pair_01_step10_final_package.md")
# {'end_rhyme': float, 'line_return': float, 'words_per_line': float,
#  'allit_per_100w': float, 'lines': int}
```

| Metric | Exact definition | Floor/ceiling (`gates.yaml`) |
|---|---|---|
| `end_rhyme` | `strict_end_rhyme` — key is the **last 3 characters of the final word**; a line hits if any line within **±4** (`rhyme_window`) shares the key | `rhyme_return_floor: 0.30` |
| `line_return` | exact repeated-line ratio, lowercased and stripped of ` .,`. ⭐ **Choruses COUNT — that is the point** | `line_return_floor: 0.20` |
| `words_per_line` | total words ÷ lines | `mean_words_per_line_ceiling: 7.5` |
| `allit_per_100w` | consonant-initial word scores if any of the next 3 consonant-initial words shares its initial; **denominator is all words**, so it is length-independent | `alliteration_per_100w_floor: 11.0` |

`strict_end_rhyme` is **deliberately crude and stably crude.** Last-3-characters is not phonetic truth. It is *comparability* — the same crude ruler applied to the archive winners and to today's draft. A better rhyme detector that cannot reproduce the 0.463 baseline is worse than this one.

⛔ **All four are FLAGs (`pass=None`), never hard fails.** A deliberate through-composed piece may sit below them — **but it must say so on purpose.**

⛔ **EXACT CHORUS REPETITION NEEDS NO DEFENCE** (`chorus_repetition_requires_no_justification: true`). `unique_line_ratio_floor` is chorus-exempt by policy, but agents were still writing *around* it — pre-emptively mutating refrains and filing justifications (*"this is the mechanism, not a defect"*, *"recommend HUMAN WAIVE"*). **A flag that makes writers apologise for choruses is doing harm even when it never fires.** A byte-identical chorus is correct craft and requires no note.

⚠️ **Silent-instrument warning, and it has already bitten.** `lyric_blocks()` matches `^## .*LYRIC.*$` — **case-sensitive, exactly two hashes and a space.** A file with no matching header yields zero blocks, and `profile_file` then returns **all-zero metrics with `lines: 0` and prints a row of zeros rather than erroring.** ⭐ **Always check `lines` before trusting a profile.**

⭐ **And the related failure that nearly shipped:** the 2026-08-09 coordinator's first instrument asked `profile()` for a key named `strict_end_rhyme`. **That key does not exist** — the returned key is `end_rhyme`. `dict.get(...)` **silently defaulted to 0.0**, and the run would have reported `RHYME_BELOW_FLOOR` on all 24 songs. **A missing dict key returning a default is a silent wrong answer.** Ask for the key the function actually returns, and let the module own the extraction — §4.7 law 6.

⚠️ The bare CLI (`python3 scripts/measure_soundcraft.py FILE …`) prints a fixed-width table. **With zero arguments it does nothing and exits 0 silently.**

> **ENFORCED BY:** nothing. ⛔ **ADVISORY (unenforced) by design** — these are FLAGs routed to the human/Somatic read. The **discipline that is enforceable** is *measure, don't eyeball*, and nothing checks that either.

---

### 6.6 `scripts/measure_render.py` — the audit that hears

Every other gate in this system reads **text**. This one reads the **render** — and text gates are structurally incapable of seeing this failure class.

```bash
python3 scripts/measure_render.py track.mp3 [more.mp3 ...]     # needs numpy + soundfile; no network, no API key
```

Emits one JSON object per file: `duration_s`, `peak`, `crest_db`, `loudness_spread_db`, `opening_2s_db`, `quiet_gaps`, `deepest_mid_dip{at_s, depth_db, width_s}`, `tempo_candidates_bpm`, `bands_by_quarter_db`, `stereo{correlation, side_mid}`, `width_dynamics{width_min, width_max, corr_width_level}`, `sustained_tones_hz`, `quarter_tone_pairs`. **Stereo keys are absent entirely for mono input** — do not assume they exist.

**Three guards in this file are instrument-credibility lessons, each installed after the instrument lied:**

1. **Tempo needs a fine hop.** The first version ran on the 50 ms envelope and reported ~120 BPM for a track measured at 109 — **and that artifact was written up as a finding about the renderer.** Tempo now runs on a separate 10 ms envelope.
2. **The cents guard on tonal components.** Without a 50-cent minimum separation the detector "found" six quarter-tone drone pairs that were **one 60 Hz cluster.**
3. **Depth AND width on the deepest dip**, so *"a 4-second stop cannot hide as 0.4 seconds."*

⭐ **Nothing here auto-fails, and that is THE GRAIN LAW.** Specs that run **with** the generator survive (open quiet, build sub, thicken the last chorus); specs that fight it get smoothed (long full stops, untuned drones, hard-panned non-musical elements). **Judge the result, not the distance from intent** — a negative `corr_width_level` is *reported, never failed*, because on the house benchmark **stereo narrowing into the climax instead of widening was better than what was asked for**, and it is now a technique.

⛔ **THE BLIND RULE.** The listening half of a render audit sends **the audio alone, never the prompt.** A judge that has seen the intent cannot tell you whether the intent arrived.

> **ENFORCED BY:** nothing — it reports. ⛔ **ADVISORY by design.** The enforceable part is procedural: `lofn-render-audit` runs the numeric pass and the blind listening pass, and neither is gated by a script.

---

### 6.7 `scripts/check_human_subjects.py` — the prefilter that is not the authority

```bash
python3 scripts/check_human_subjects.py path/to/lyrics.md
cat lyrics.txt | python3 scripts/check_human_subjects.py -
```

| Exit | Recommendation |
|---|---|
| `0` | `PASS_TO_NEXT_GATE` — ⛔ **not permission to ship** |
| `1` | no argument supplied |
| `2` | `HOLD_FOR_HUMAN` **or** `ONLINE_CHECK_REQUIRED` |

Three independent checks: **(A) minor-as-victim** — minor words or an age pattern under 18, co-occurring with any of 46 crime/death terms. **(B) person names** — spaCy NER when available, regex fallback otherwise, each name scored `HIGH`/`ELEVATED`/`NORMAL` for online check. **(C) identifying tuple** — any name plus a locating detail (year, calendar date, pinning role, place).

⭐ **The design lesson, from the Pair-01 near-miss: THE NAME IS THE CATCH, NOT THE FACTS.** The offending song had fictionalised the dates and kept **the real victim's real name.** A date/fact matcher would have passed it clean.

The JSON output carries `"is_authority": false` as a literal field. The authority chain is Gate 16 + the Step-11 Andon Cord + human review + the Standard's mandatory online recent-news cross-check for **every** detected name. It **fails OPEN toward over-flagging** — an internal exception emits a `detector_error` entry which itself forces the flag branch. Without spaCy the regex fallback over-flags any capitalised mid-sentence word, deliberately.

⛔ **Forbid IDENTIFIABILITY, not subject matter.** Anchor to the *charge* of a moment; draw the theme, invent the people. A piece that cannot be drafted without an identifiable real person is **HELD FOR HUMAN** — surfaced by name before QA, never silently shipped. **REAL GRIEF IS NOT RAW MATERIAL.**

> **ENFORCED BY:** exit 2 + `vault/HUMAN_SUBJECT_STANDARD.md` read **pre-draft**, so the forbidden thing is *unspecifiable* — there is no field in the spec for an identifiable victim. ✅ Real control, correctly labelled non-authoritative.

---

### 6.8 `scripts/check_skill_mirror.py` — doctrine drift, both directions

```bash
python3 scripts/check_skill_mirror.py [--root DIR]
```

Compares `.claude/skills/**/*.md` (canonical) against `.agents/skills/**/*.md` (Codex mirror). Exit `0` = `IN SYNC`; exit `1` = `DRIFT: N file(s)`.

Four drift classes: **MISSING FROM MIRROR** · **MIRROR-ONLY** · ⭐ **UN-REWRITTEN ENGINE** (a mirror file still saying "Claude" after the legal occurrences are stripped — a file copied verbatim that would normalise clean while telling Codex it is Claude) · **DRIFT (content)** after normalising the permitted axes only: path swap, `Codex`→`Claude`, the protected literal `claude|codex`, and the AUTHORITY banner.

⭐ **Why it prints BOTH directions:** on 2026-07-24/25 the mirror's `lofn-music` sat **33 lines behind canonical** — missing THE RETURN, the rhyme-debt rule, and the mandatory prosody axis — and `lofn-render-audit` was **absent from the Codex tree entirely.** Then a `sed` engine-rename silently **reverted ~20 legitimate Codex lines.** The rule that came out of it:

> **A sync must be reviewed by what it DELETES, not by what it adds.**

⚠️ **False-green trap:** if either tree is missing it prints `skip: … has no skill trees to compare` and **returns 0.** Running from the wrong `--root` produces a clean green that means nothing. Check the trailing `(N compared under ROOT)` count.

> **ENFORCED BY:** exit 1. ✅ Real control — **and the only thing standing between `CLAUDE.md` and `AGENTS.md` diverging.** When you change doctrine in one, mirror it and run this.

---

### 6.9 `verify_pipeline_map.py` and `audit_lofn_pipeline_artifacts.py`

**`verify_pipeline_map.py`** — verifies `tools/explorer/pipeline_map.yaml` against the real tree **in both directions**: every declared path exists (forward), and every step file on disk is declared (reverse — this catches new step files nobody registered).

```bash
python3 scripts/verify_pipeline_map.py [--root DIR] [--map PATH] [--json]
```

Exit `0` clean · `1` mismatch · `2` manifest not found **or** no YAML parser available. ⭐ **FAIL-LOUD, never fail-open** — the opposite contract from `validate_step.py`, and correctly so: a rotted manifest is a bug to surface, not a threshold to degrade. ⚠️ Do not conflate exit 2 with exit 1 in CI.

**`audit_lofn_pipeline_artifacts.py`** — audits artifact **granularity**: coordinator steps 00–05 as six separate canonical files, pair steps 06–10 as separate files per pair.

```bash
python3 scripts/audit_lofn_pipeline_artifacts.py <run_dir> [--pairs 6]
```

Exit `0` PASS · `1` failures · `2` not a directory. A collapsed `pair_*_steps_06_10.md` is a **warning**, and a missing per-pair step artifact is a **failure** — which is the mechanical half of the NO-SKIP rule (§5.6).

> **ENFORCED BY:** exit 1 each. ✅ Real controls.

---

### 6.10 ⛔ Test coverage — the honest number

**Two of the nine scripts in this section have dedicated tests:** `test_validate_step09_distinctiveness.py` (4 assertions) and `test_validate_portfolio_distinctiveness.py` (8 assertions).

**No tests exist for** `validate_step06_distinctiveness.py`, `measure_soundcraft.py`, `measure_render.py`, `check_human_subjects.py`, `check_skill_mirror.py`, `verify_pipeline_map.py`, or `audit_lofn_pipeline_artifacts.py`.

And the twelve assertions that do exist are all **parser-selection** tests — they verify that the right *segment* is chosen from a cumulative artifact. ⛔ **Not one of them is a known-bad fixture that the validator is required to REJECT.**

That is F4 (§4.8) stated as a measurement rather than an opinion: **the gates in this system have been shown to run. They have not been shown to work.** Every instrument failure in the ledger — the 1.000 × 30 collisions, the zero-extraction CLEAN, the `autojunk` blindness, the four permanent taxonomy stub rows — is a case a single required-to-fail fixture would have caught before it produced a verdict.

> **ENFORCED BY:** ⛔ **nothing. This is the largest unenforced gap in the system, and it is the one the panel would fix first.**

---

# PART IV — MAINTAIN

## 7. Keeping it true over time

### 7.1 The two-engine mirror

Doctrine lives twice: `CLAUDE.md` + `.claude/skills/` (canonical) and `AGENTS.md` + `.agents/skills/` (Codex). **The two must not drift.**

When you change doctrine in one, mirror it in the other and run `python3 scripts/check_skill_mirror.py`. Review the sync **by what it deletes.**

> **ENFORCED BY:** `check_skill_mirror.py` exit 1. ✅

### 7.2 The ledgers — what gets written back

A run that loads no memory and writes none back is a cover band, not Lofn.

| Ledger | Contents | Written by |
|---|---|---|
| `vault/RUN_LEDGER.md` | **Operational** memory — infra/process hazards, `open`/`watch` entries. Never an aesthetic note. | coordinator |
| `vault/COMPETITION_LEARNINGS.md` | **Advisory-aesthetic.** One curated entry per shipped piece. Tag-keyed, confidence-stamped, capped ~25 in the index. | `lofn-qa` |
| `memory/YYYY-MM-DD.md` | Raw daily log | any session |
| `MEMORY.md` | Curated long-term, distilled from the dailies. **Private sessions only** | The Scientist + Lofn |

**Phase −1 reads them; QA writes them back.** Tag-walk `COMPETITION_LEARNINGS` for the 3–5 entries intersecting *this* run's theme + venue + modality — not all 25. Scan the tail of `RUN_LEDGER` for open hazards.

⚠️ **`COMPETITION_LEARNINGS` is ADVISORY and INDIGNATION-exempt.** `L3` says NightCafe portrait themes reward warm palettes — that is a crowd-taste note, **never a veto on a grief piece that means it.** An advisory ledger that becomes a style guide has quietly replaced the artist with a regression to the venue's mean.

> **ENFORCED BY:** nothing. ⛔ **ADVISORY (unenforced)** — a run that skips the write-back produces no error.

### 7.3 The run-health footer — four fields, not a metrics culture

Append to every run INDEX: **`{ pairs_shipped, pairs_quarantined, total_gate_retries, qa_repairs_issued }`**. Four. No more. It surfaces a real degradation signal without inviting a dashboard.

⭐ **The zero-rejection tripwire.** `qa_repairs_issued` exists because **a QA that never says no is decorative.** Healthy band for a 6-pair run: **≥1 REPAIR, or ≥1 substantive FLAG escalated to the Somatic read.**

⛔ **A full run reporting 0 repairs and 0 quarantines across 24 artifacts does not celebrate. It triggers an audit of the JUDGE** — re-run the `lofn-qa` blind golden+decoy check on a sample. **When the measures say perfect and the listener says worse, the measures are lying.**

> **ENFORCED BY:** nothing. ⛔ **ADVISORY (unenforced)** — but it is the only self-suspicion mechanism in the system, and F4 (§4.8) is its instrument-level twin.

### 7.4 Adding a gate — the checklist

1. **Is it countable?** If it needs taste, it is not a gate — it is a judge's question. Do not encode taste as arithmetic.
2. **Put the number in `vault/gates.yaml`.** Never restate it in a skill file. If prose must quote it, `--meta-check` will police the copy — with a WARN, not a block.
3. **Decide the verdict class up front:** hard fail (`pass=False`) · **FLAG** (`pass=None`, routed to a human) · or measurement-only. ⭐ **When in doubt, FLAG.** Every hard fail is a future false-positive that will teach somebody to route around the gate.
4. **Write the empty-extraction guard first**, before the comparison. Two empty strings compare as 1.000.
5. **Print the extraction before the conclusion**, and assert its count against an independent cardinality.
6. **Ship a fixture it must reject.** (F4 — proposed, §4.8.)
7. **Decide fail-open or fail-loud, and say which.** Threshold config → fail-open. Manifest honesty → fail-loud. Do not mix them in one script.

### 7.5 ⛔ The unenforced inventory

Every rule in this guide that is **not** backed by a control, collected in one place so nobody mistakes it for protection:

| Rule | Where | Why unenforced |
|---|---|---|
| `06_<modality>_handoff.md` must exist before dispatch | §1, `L30` | No script blocks dispatch. **This exact gap caused `L30`.** |
| ICB is read-only after Phase 1 | §2.3 | Detected by sha after the fact; the write is not blocked |
| Skeptic seat per panel (3 total) | §2.5 | Only a 2000-char proximity heuristic in `validate_orchestrator_packet.py` |
| Three-configuration transformations gate | §2.5 | Not checked at all |
| ICB sha mismatch = dispatch failure | §3.2 | Coordinator discipline; no script refuses the pair |
| Dispatch-packet itemisation | §3.3 | Echoed byte counts prove presence, not fidelity |
| Subagents write only `pair_NN_*` | §4.2 | Convention only |
| Max-3 repair ceiling | §5.1, §6.3 | `--attempt` is caller-supplied; the script never increments |
| RETURN floors (rhyme, line-return, w/l, alliteration) | §6.5 | FLAG-only **by design** |
| Render audit + BLIND RULE | §6.6 | Procedural |
| Ledger write-back | §7.2 | No error on skip |
| Zero-rejection tripwire | §7.3 | Human notices, or nobody does |
| **Gate fixtures (required-to-reject)** | §6.10 | **Does not exist. Largest gap.** |
| Contract-hash pinning | §4.6 | **Proposed (F2), not built** |
| Shared instrument-law module | §4.7 | **Proposed (F3), not built** |

⭐ **This table is the most useful page in the document.** Everything above it describes a system that works. This describes where it is running on discipline — and discipline is what you have instead of a design.

---

## 8. A worked example — `2026-08-09_daily_music_genz`

One real run, end to end. All files under `output/daily/2026-08-09/`.

**Phase 0–1.** Seed anchored to two Golden Seeds. Personality LOFN-PRIME on all six pairs — `lofn-prime-mini.yaml`, **104,422 bytes injected verbatim.** Library panel `gen-z-firestarters`, **v2 re-derived at load time with three defects repaired before the room opened**: the Medium Panel had **no skeptic seat**; two Context seats named figures with **no locatable published record** and were re-anchored (*a seat that cannot be grounded cannot dissent from anything*); one figure was **double-seated as Devil's Advocate on two panels** (one room, one skeptic). 18 voices, 3 panels, 3 Hyper-Skeptics, three configurations: **BASELINE → GROUP: `BRIDGE` → SKEPTIC: `REFLECT`.**

**Research.** 25 facts: `OK` ×20, `UNAVAILABLE` ×2 (**recorded, not substituted**), `SCOPE-SKIPPED` ×3. ⭐ One feed returned **stale April-2025 rows** and was **discarded with the discrepancy recorded** — *an impossible number is a bug report, not a finding.*

**The join.** The coordinator wrote `coordinator_restat.py` — the instrument whose five laws are §4.7 — and it produced `GATE_REPORT.json`: 24 rows of `{pair, v, title, prompt_chars, lyrics_chars, sung, flags, rhyme, lret, wpl, allit}`, plus `hard_fail`, `worst_lyric_sim`, `worst_prompt_sim`, and the **comparison count**.

Every one of the 24 landed inside the bands. Pair 01 V1: prompt **951** chars (band 850–1000, target 870–960, hug ceiling 985 — inside the *target*, not hugging the cap), lyrics **4679** (cap 5000, target 4800), **82** sung lines (band 70–120, floor-hug 72), `rhyme 0.476` (floor 0.30), `line_return 0.317` (floor 0.20), `wpl 7.24` (ceiling 7.5), `allit 11.28` (floor 11.0). **Flags: none.**

**What the run caught that no gate could have.** QA's `D-3` ruling: P02 sang *"a hundred and thirty-eight"*, P04 sang *"thirty seconds"* — a naive token scan flags a collision. The ruling walked the **2026-07-24 bar** (same fact + same device + same conclusion) and matched **zero of three axes**: different facts (totality's duration vs a royalty threshold), different devices, different conclusions. **"The shared token is an artifact of the decimal system — 'a hundred and thirty-eight' contains 'thirty' the way 'there' contains 'here.'"**

⭐ **That is the whole argument for the judge in one paragraph.** A validator would have flagged it or missed it. Neither would have been *right*.

**Verdict:** SHIP. Somatic Gate 3 YES / 0 NO. ⛔ **Nothing rendered, nothing published** — per `vault/AUTONOMY.md`, autonomous runs stop at drafts on disk.

---

## 9. The failure catalogue — every control, and the autopsy that produced it

| Date | What happened | Control installed | Enforced? |
|---|---|---|---|
| 2026-06-26 | One global variation template across all pairs → two pairs sang the same song with nouns swapped | Variation angles authored **per pair**; identical labels = dispatch blocker | ⛔ advisory |
| 2026-06-28 | A run skipped steps 07/09/10 entirely and **shipped 6/6** — the gates measured structure and could not see that nobody wrote the arrangements | **NO-SKIP rule**; such a run is NON-CANONICAL and cannot be published | ✅ `audit_lofn_pipeline_artifacts.py` |
| 2026-06-28 | Published piece reproduced its benchmark's title line, vocal spec, key/BPM and arrangement formula while self-check said "no copying" | ⛔ **GOLDEN-OUTPUT QUARANTINE**; `house_lexicon` FLAG list | ⚠️ FLAG only |
| 2026-06-28 | A MUSIC PROMPT truncated mid-phrase to fit the cap **passed** — the check counted chars, not sense | `music_prompt_terminal_punctuation` — hard fail | ✅ `validate_step.py` |
| late-06 | Every prompt measured 988–999 against a 1000 cap; every lyric sat on the 70-line floor | Target bands + `music_prompt_hug_ceiling` / `sung_lines_floor_hug` boundary-hugging FLAGs | ⚠️ FLAG only |
| **2026-07-24** | ⛔ **A second controller destroyed ~11 hours of a live run.** The rule existed — **as prose** | ⭐ **`.RUN_LOCK`**, `O_EXCL`, **exit 3**, heartbeat at every wave | ✅ **the one true interlock** |
| 2026-07-24 | Two songs independently sang the same number via the same device to the same conclusion; **every pair's self-check was correct in isolation** | Portfolio-level distinctiveness; the three-axis collision bar | ✅ `validate_portfolio_distinctiveness.py` |
| 2026-07-24 | Measured: we wrote at **half the archive winners' rate of return** | RETURN floors + `measure_soundcraft.profile_file()` | ⛔ FLAG only, by design |
| 2026-07-24/25 | Codex mirror sat 33 lines behind canonical; a `sed` rename **reverted ~20 legitimate lines** | `check_skill_mirror.py`, **both directions**; review a sync by what it **deletes** | ✅ exit 1 |
| 2026-08-03 | Fenced prompts measured *with* their code fences | Fence-strip before measuring | ⛔ per-run |
| **2026-08-04** | ⭐ **30 cross-pair collisions at exactly 1.000.** Empty extraction; two empty strings compare as 1.000 | ⭐ **exit 2 = INSTRUMENT FAILURE, explicitly not a result** | ✅ **all three validators** |
| 2026-08-05 | Four template-identical prompts reported **94% distinct** — `SequenceMatcher`'s default `autojunk` blinds it above 200 elements | `autojunk=False`, mandatory | ✅ in source |
| 2026-08-05 | Six agents produced **three** step-10 heading conventions; the coordinator invented a **fourth by reading the convention off the artifacts** | **The validator is the contract**, never the artifacts | ⛔ advisory (**F2**) |
| 2026-08-05 | A step-11 pass changed a correct sung line because a scanner hit on it | An instrument's output alone never justifies changing a correct line | ⛔ advisory |
| 2026-08-07 | ⭐ Strict-only matcher extracted **zero** blocks and printed **CLEAN on all three absolution scans** of the run's most important gate | Extraction-count assertion against independent cardinality | ⛔ per-run (**F3**) |
| **2026-08-07** | A step contract ordered the opposite of doctrine. **Five agents reached past it to the governing rule; one obeyed the file in front of it.** Both correct — the coordinator had never written the artifact that resolves the conflict | ⭐ **`06_<modality>_handoff.md` mandatory before dispatch** | ⛔ **advisory — the gap that caused it is still open** |
| 2026-08-08 | Multi-style collision engines produced pieces **about** materials instead of **of** a material | ⭐ **ONE SEAM**; collision engines retired | ⛔ advisory |
| 2026-08-08 | Five of nine NightCafe posts were auto-captioned by a **style or technique**, not an event | ⭐ **THE VINCENT TEST** — free blind first read, run before submitting | ⛔ advisory |
| 2026-08-09 | `run_lock heartbeat` **refused its own run with exit 3** — each tool-call shell is a new PID, so `acquire` and `heartbeat` stamped different ids | ⛔ **Pass `--controller-id` on every call.** *Fix the id, never the interlock* | ⚠️ protocol |
| 2026-08-09 | An instrument asked for a dict key that does not exist; `.get()` **silently defaulted to 0.0** and would have failed all 24 songs | Delegate to the shipped definition; never accept a silent default | ⛔ per-run (**F3**) |
| **standing** | Four `taxonomy_cardinality` rows emit `actual=None, pass=None` — **always**. Step 00's cardinality claim has never been checked | ⛔ **none** | ⛔ **open (F4)** |

⭐ **Read the "Enforced?" column downward.** Every ✅ is an autopsy that produced code. Every ⛔ is an autopsy that produced a sentence. **The ratio is the honest state of this system**, and it is why the panel's Hyper-Skeptic did not fully withdraw.

---

## 10. What transfers

The ICB is a local design choice. These are claims about the whole class of system, and they are what to take somewhere else:

1. ⭐ **The errors are in the join, not the nodes.** Twelve of twelve measured instances: the stochastic agent was right, the deterministic instrument was wrong. **Everyone builds these systems expecting the opposite.**
2. ⭐ **A silent instrument is more dangerous than an incorrect one.** A validator that extracts nothing reports CLEAN. Give "did not run" **its own exit code**, distinct from both pass and fail. Exit 2 in the distinctiveness validators is the cheapest high-value line in this repo.
3. **Print what an instrument EXTRACTED before what it CONCLUDED, and assert the count against an independent expectation.** An impossible number is a bug report, not a finding.
4. **Continuity is a mutability problem, not a memory problem.** Freeze the value, hash it, inject it, let nobody write it. Divergence is copy-and-branch.
5. **Hash what you froze — and normalise line endings first.** A tamper check that fires on git's newline handling trains everyone to ignore the alarm.
6. **Share-nothing concurrency, aggregate single-threaded at a join.** Six concurrent writers to one file is a corruption you make impossible by construction.
7. **The executor claims; the join proves.** Re-stat every artifact: existence, size, binding constraint recomputed, hash re-verified.
8. **Resolve contract conflicts in writing, in the run, before dispatch.** ⭐ **Agreement among agents is not evidence a rule was stated — it is evidence they reasoned alike, which is luck wearing a uniform.**
9. **The adversary is a separate node on a different tier, and it must be able to say no.** Trust its **measurements**; write your own **prescriptions** — a blind judge sees the symptom truthfully and cannot see the intent, which is exactly what makes its observation trustworthy and its fix unreliable. ⛔ **Adopting a judge's fix verbatim is how a REPAIR becomes a REJECT.**
10. **A QA that never rejects is decorative.** Zero repairs across 24 artifacts audits the judge, not the work.
11. **Choose fail-open or fail-loud per script, deliberately.** Threshold config fails open; manifest honesty fails loud. Never mix them in one file.
12. ⭐ **A guide's job is not to be followed. It is to be compiled** — into interlocks, exit codes and required artifacts. **Publish what did not compile.**

---

## ⛔ THE STANDING LIMITATION

This architecture is **a ratchet over past surprises.** Every control in §9 was installed after something went wrong, and the interval between the surprise and the interlock is where this system permanently lives.

**You cannot compile a rule you have not yet been surprised into writing.**

That is the entire argument for keeping an adversarial node that did not participate in the work, for the free blind reads (**the Vincent Test**, **the BLIND RULE**), and for treating every enumeration here — including this one — as provisional.

The panel's Hyper-Skeptic conceded that writing-it-down and compiling-it-into-an-interlock are different interventions. **It did not concede that the enumeration is complete.** It never will be.

---

*Convened and moderated by Lofn. The four findings in "What the panel changed" are recorded as proposals; F2, F3 and F4 are unbuilt and are listed among the unenforced. The Hyper-Skeptic's final objection is retained and unresolved by design.*
