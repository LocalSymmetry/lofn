---
name: lofn-qa
description: Audit, validate, and classify completed Lofn pipeline outputs against the strict quality gates, backed by Claude. Use after a lofn-music / lofn-image / lofn-video / lofn-story run, or on a suspicious partial run, to get a SHIP / REPAIR / FAIL verdict and a repair brief. Do NOT use for creative generation — this is the adversarial auditor, not the artist.
---

# Lofn QA — Claude-backed adversarial gate

The competitive auditor. It proves two things at once: **a listener/viewer can grasp the surface, and a second pass reveals the cathedral.** It fails both extremes — impressive obscurity AND competent blandness. QA stays strict: it does not loosen structural gates to protect "creative freedom."

Replaces the OpenClaw Python validators with Claude judgment + the same checklists. Adopt the auditor stance from `skills/orchestration/references/adversarial_qa_stance.md` first.

## Procedure
0. **Run in a fresh, clean-context judgment subagent.** The QA / Somatic / Step-11-reject judgment MUST execute in a clean Agent-tool subagent fed ONLY the artifact + the verbatim ICB + the gate spec (and the `GATE_REPORT.json` if present) — **never appended to the thread that generated the artifact.** A generator grading its own homework is the conflict that ships corpses past the Andon Cord. Tier follows the *role* (judge, not generate), never the step number; this is a dedicated clean-context spawn, not a per-spawn flag that can silently fall back.
1. **Identify** the run directory + modality.
2. **Load the gates** just-in-time:
   - All modalities: `skills/qa/references/qa_full_legacy.md` (the full tuned procedure — authoritative).
   - **Music:** `skills/qa/references/suno_15_point_qa.md` + `skills/qa/references/eligibility_7_properties.md`. Classify ACCESSIBLE vs AMBITIOUS, then run the 16-point gate.
   - **Image:** `vault/VISION_QA_DEPTH_AUDIT.md` (Visual Somatic Gate + 7-element density checklist).
   - **Video/Animation:** `vault/DIRECTOR_QA_DEPTH_AUDIT.md` (Cinematic Somatic Gate + 5-element shot checklist).
   - **Thresholds:** numeric bands are read from `vault/gates.yaml` where present (the single source for the 850–1000 / `<5000` / 70–120 / ≥80-word numbers); `EXECUTION.md` §4 is authoritative if the two disagree.
3. **Read the deterministic evidence first.** If `scripts/validate_step.py` wrote a `GATE_REPORT.json` for this run (the L4 acceptance helper, fail-open), load it and **paste its measured `{pair, step, check, expected, actual, pass}` rows verbatim as proof-of-fix evidence** in the Evidence column — these are computed values, not QA guesses. The helper is fail-open: if it is absent or logged a warning, fall back to in-prose measurement and note it; never block a valid run on a missing/broken helper.
4. **Verify ICB integrity (cheap proof, not a soul substitute).** Confirm the canonical ICB / CREATIVE_CONTEXT prefix appears as an **unbroken verbatim substring** at the head of each step's prompt (additions below pass; edits *above* the block fail), and that the count of `(after ` speaker tags **== 18**. A voice count ≠ 18 or a broken prefix is caught as `REPAIR — THREAD LOSS`. **The count proves presence, not fidelity** — a paraphrase can match length and voice-count. So this is only the cheap tripwire for a gross drop; **the human personality-fidelity read (below) stays the real guarantee, and the count never substitutes for the soul read.** Never summarize, trim, or "optimize" the ICB to make a count easier.
5. **Run the gate** without weakening any check. Apply the Claude-native self-check gates in `.claude/skills/lofn/EXECUTION.md` §4 for structural/pipeline integrity.
6. **Write** `QA_REPORT.md` in the run directory, including the structured evidence block (below) BESIDE the verdict. If failures require rerun, write the repair brief in the rerun format from `qa_full_legacy.md` and route to the failing step (09/07 for music thread loss, 08 for prompt-format, etc.).
7. **On SHIP (every shipped/selected piece): append ONE curated failure-ledger entry** to `vault/COMPETITION_LEARNINGS.md` — see "Failure-ledger write-back" below.

## What every PASS must clear
- **Pipeline integrity / granularity:** coordinator steps exist as separate files (`step00…step05`), and steps 06–10 exist as separate **per-pair** files (`pair_{NN}_step06…step10`). A collapsed `pair_{NN}_steps_06_10.md` rollup or a single batch run is a **blocking** failure even if filenames look present. 6 pairs × 4 = 24 outputs unless the Scientist downsized.
- **Continuity / ICB:** every step cites the continuity payload (Special Flairs + all 18 panel voices + Golden Seed + active personality + previous artifact). Missing → `REPAIR — THREAD LOSS`, even if formatting passes.
- **Personality fidelity:** the piece proves which personality made it (sonic-world/voice sentence + signature device + seed-derived weirdness). If any competent prompt could have produced it → `REPAIR — SOUL LOSS`.
- **Somatic Gate:** the 3 Hyper-Skeptics vote as a bloc — *"distinctive enough to be Lofn, or generic?"* 2 of 3 NO = BLOCKED. **This is the primary gate.** No structured evidence block, measured count, or helper report below ever overrides, substitutes for, or pre-empts the somatic read — they are inputs the Skeptics may cite, never a verdict that ships past them. Quantify the corpses; never pretend to quantify the soul.

## Named-corpse Andon checklist (prompts for the 3-Hyper-Skeptic bloc)
These are **enumerated reject-conditions worded as prompts** the Hyper-Skeptics carry into the Somatic Gate — not scores, not auto-reject floors, not a Python detector. Pulling the cord is never failure; shipping past it is. A Skeptic who flags one must cite the named condition with evidence ("emotion goes nowhere — no second movement") rather than a bare "feels generic." Under-flagging is acceptable; the conditions exist to give the human a sharper language, not to auto-fail. Walk each:
- **One-note emotional arc** — does the piece reach a *second movement*, or does it hold a single register start to finish with nowhere to go?
- **Single dominant repeated section** — is one section (chorus/refrain/stanza) doing all the load-bearing work while the rest is filler around it?
- **A motif that never transforms** — does the central image/phrase/hook *change* across the piece (recontextualized, inverted, paid off), or just recur unchanged?
- **Repeated-line collapse** — has repetition stopped being a deliberate device and become the piece *running out of things to say*?
> A deliberate refrain, villanelle, incantatory INDIGNATION dirge, or sustained-register elegy is NOT a corpse. These prompts ask the Skeptic to tell *intentional return* from *creative exhaustion* — a judgment only a human read makes. Never convert any of these into a numeric threshold that gates SHIP/FAIL.

## Modality hard gates (non-waivable — a custom/looser parent checklist cannot waive these)
Phrase each check as a **natural-language assertion that names the concrete checkable value** — e.g. "the EXCLUDE field is present and its content is under cap", "the lyric carries a 3+3 emotional duality", "the MUSIC PROMPT is a dense paragraph of 850–1000 chars, NOT bracket tag-soup". Evaluate the assertion by meaning, not by an exact-string grep that false-fails on minor format drift; the concrete band / EMO-header shape stays *inside* the assertion so "robust" never decays into vibes.
- **Music:** standalone `## 1. MUSIC PROMPT` (a dense paragraph of 850–1000 chars — NOT bracketed key:value tag-soup — naming no real artist) + a separate EXCLUDE field present and under cap; lyrics open `[Theme:…]` then `[SONG FORM:…]`; full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` use taxonomy emotions, not bare AWE/INDIGNATION; ≥1 SFX cue; 70–120 sung lines; the Suno lyrics field stays `<5000` chars (target ≤4800); verse-structure diversity across the 6; Lineage & Credit block on living-scene genres. Run the full 7 Singer-Surface + 5 Cathedral-Engine + 3 Suno-Package + Lineage gates. *(Note: the music SKILL's 2026-06-09 mandate is dense-paragraph prompts; where `suno_15_point_qa.md` still says "categorized key:value", the paragraph mandate is the newer authority — flag the conflict, do NOT fail a correct paragraph prompt for the stale bracket rule.)*
- **Image:** noun-first present-tense scene description, no imperative openers, medium named early, emotion shown not named, no Storybook-Assassin ban-words ("ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate"), no living-artist names, subject legible at first glance, ≥80 words (Flux) / five-slot directive (GPT_I2).
- **Video/Animation:** `[CAMERA]+[SUBJECT]+[ACTION]+[SETTING]+[STYLE & AUDIO]`; audio directed explicitly; loop logic stated (animation); distinct camera grammar across pairs.
- **Story:** standalone distinct voice per pair, body-before-thesis, world coherence, no generic AI cadence.

## Human-Subject identifiability backstop (HOLD-FOR-HUMAN)
QA enforces the identifiability backstop from `vault/HUMAN_SUBJECT_STANDARD.md`. The line forbidden is **identifiability**, not subject matter: a real, named, or recognizably-depicted private individual rendered in a way that ties the piece to *that person*. On any reasonable suspicion of an identifiable real human subject, the verdict is **HOLD-FOR-HUMAN** — not an auto-FAIL, not a silent SHIP. The piece does not ship and is surfaced to the human for an explicit decision. This backstop sits above every numeric and somatic gate and is non-waivable by a looser parent checklist.

## Structured QA evidence block (BESIDE the verdict, never replacing it)
Surface the measured values QA already reasons about as a structured block **alongside** SHIP / REPAIR / FAIL so verdicts are trace-auditable and trendable across runs — a floor under the judgment, never a substitute for it. Quantify the corpses; do not pretend to quantify the soul. Report, for the modality:
- **char counts vs caps** — MUSIC PROMPT chars vs 850–1000; Suno lyrics field vs `<5000` (target ≤4800); Flux ≥80 words.
- **sung-line count vs 70–120** (music).
- **EMO-tag balance** — taxonomy-emotion distribution; flag a single dominant emotion or bare AWE/INDIGNATION tags.
- **repeated-line / n-gram collapse ratio** — reported as a **FLAG only**, with chorus/refrain **exempt**; a deliberate refrain must never auto-fail on this number. It routes to the human/Somatic read, it does not decide.
Where `GATE_REPORT.json` exists, populate these from its measured rows. **A piece that passes every count can still be REPAIR — SOUL LOSS by the somatic read.** The structured block sits beside the 3-Hyper-Skeptic verdict; it never displaces it.

## Failure-ledger write-back (advisory, capped, INDIGNATION-exempt)
On every shipped/selected piece, append exactly **ONE curated entry** to `vault/COMPETITION_LEARNINGS.md` capturing: **theme-tags** (theme-type / venue / modality), **what the gate caught**, **one transferable rule**, and a **confidence %**. Discipline:
- **Advisory, never a constraint.** These are failure-ledger / process notes a future run reads *as advice*, confidence-stamped, LOW until human- or second-run-corroborated. An auto-written entry may only feed the corpse/process checklist — it must **NEVER** become an aesthetic constraint (no "warm palette mandatory", no "INDIGNATION underperforms"). Only a human promotes a lesson to a hard creative constraint.
- **Venue/modality-scoped.** A lesson is tagged to its venue+modality and must not leak into music or non-competition runs.
- **Triggered-INDIGNATION is explicitly EXEMPT from suppression.** Never write an entry, and never let one stand, that would suppress or tune away triggered-INDIGNATION work toward what a voting venue rewards.
- **Hard-capped (~25 live curated entries).** Growth is by disciplined append-and-prune, not unbounded accumulation; prune the lowest-confidence stale entry when over cap.
- Apply the mandatory **"would this lesson have hurt our best past entry?"** check before recording — if yes, do not record it as a rule.

## Verdicts (every report)
- **Pipeline Integrity Verdict:** PASS / REPAIR REQUIRED / FAIL
- **Package Verdict** (modality contract): PASS / REPAIR REQUIRED / FAIL
- **Human-Subject Verdict:** CLEAR / HOLD-FOR-HUMAN
- **Overall:** SHIP / REPAIR / FAIL  *(the Somatic Gate is decisive; the structured evidence block is a floor beside it, never the verdict)*

## Report format (`QA_REPORT.md`)
```markdown
# Lofn QA Report — <run> (<modality>)
## Verdicts
- Pipeline Integrity: …
- Package: …
- Human-Subject: CLEAR | HOLD-FOR-HUMAN
- Overall: …
## ICB Integrity
- prefix verbatim substring: yes/no · `(after ` voice count: N (==18?) · injected bytes: N
## Structured Evidence Block  (measured values, BESIDE the verdict — from GATE_REPORT.json where present)
| Metric | Measured | Band/Cap | Pass/Flag |
(char counts vs caps · sung-line count vs 70–120 · EMO-tag balance · repeated-line ratio = FLAG only, chorus-exempt)
## Score Table
| # | Gate | Verdict | Evidence | Repair |
## Blocking Fails
## Somatic Gate (3 Hyper-Skeptics)  — PRIMARY; cite named-corpse conditions with evidence
## Required Repairs  (routed to step N)
## Failure-ledger entry appended (on SHIP)  — theme-tags · what the gate caught · transferable rule · confidence
## Final Recommendation
```

A strong artifact with weak structure still fails; a soulful artifact missing required pieces still fails structurally. A piece that passes every measured count still fails if the Somatic Gate reads SOUL LOSS — the numbers are a floor, the somatic read is the guarantee. Fix, then re-QA before delivery.
