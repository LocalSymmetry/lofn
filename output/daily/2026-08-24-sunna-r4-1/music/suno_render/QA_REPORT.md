# Lofn QA Report — 2026-08-24-sunna-r4-1 (music)

**Audit role:** fresh clean-context adversarial judge  
**Audit scope:** text packages and the explicitly permitted pipeline evidence only  
**Render status:** **UNAUDITED** — no finished audio exists in the evidence set; no claim below is an audio verdict  
**Doctrine note:** the named file `skills/qa/references/adversarial_qa_stance.md` does not exist at the permitted path. No substitute path was read; the binding adversarial stance in `.agents/skills/lofn-qa/SKILL.md` and `qa_full_legacy.md` was applied.

## Verdicts

- **Pipeline Integrity:** REPAIR REQUIRED
- **Package:** REPAIR REQUIRED
- **Human-Subject:** CLEAR
- **Overall:** **FAIL**

The run is canonical in the no-skip sense—Steps 07, 09, and 10 exist for every pair—but it cannot ship. The frozen ICB is not placed at the head of the dispatches as required, the Phase-1 transformation record does not contain the three required fully labeled/debated configurations, and the finalist set triggers the expanded eligibility **RUN FAIL** because Uniqueness in Overall Run averages 2.0/5. Three individual packages also miss the strict Cathedral Gate 9, all six miss a binding legacy vocalist-fingerprint subcheck, and two songs are blocked by the primary Somatic Gate.

## Blind Stage 1 — SEALED; decode pending

`music/QA_BLIND_STAGE1.md` was written before any identified finalist or pipeline artifact was read. No `qa_blind_key_*.txt` was read, and no candidate/golden/decoy label is inferred here.

| Set | Sealed ranking |
|---:|---|
| 01 | **B > C > A** |
| 02 | **B > C > A** |
| 03 | **A > B > C** |
| 04 | **C > B > A** |
| 05 | **C > A > B** |
| 06 | **B > C > A** |

Coordinator decode is pending. Until the withheld keys are decoded outside this judge context, the candidate-below-decoy and decoy-above-golden calibration rules cannot be applied. This report makes no identity inference from the rankings.

## ICB Integrity

- **prefix verbatim substring:** **YES, but head-position FAIL** · `(after ` voice count: **18** in every dispatch · injected bytes: **142,674**
- Canonical ICB SHA-256: `1ce0b17997c26edf5291ce01bb2ffdc9544c7ae147b9bffdcf8b197114c206c9`; the root and `music/` copies are byte-identical.
- Exact unbroken ICB occurrence: **6/6** dispatch files.
- ICB position: character offset **2,781** in every dispatch, after a common dispatch-control header. The contract permits additions below the block and fails additions above it; therefore strict prefix-at-head integrity is **NO** even though no ICB bytes were altered.
- Deep-content tripwires present: 18 voices, 3 named Hyper-Skeptic seats, 15 Special Flairs, Golden Seed `THE EXACT INVITATION`, active Sunna R4.1 personality, metaprompt, pair assignments, and complete personality YAML.
- **Transformation gate:** **FAIL — COLLAPSE FAILURE.** The panel ledger says the library object is a persisted `BASELINE → BRIDGE → REFLECT` result and includes `Group choice — BRIDGE` plus `Hyper-Skeptic choice — REFLECT`, but it does not contain three live configurations labeled `BASELINE`, `GROUP TRANSFORM: BRIDGE`, and `SKEPTIC TRANSFORM: REFLECT`, each with its own inter-seat disagreement. A persisted library history and two moderator summaries do not satisfy the required three-config debate.

**ICB verdict:** `REPAIR — THREAD LOSS / CONFIGURATION COLLAPSE`. The payload is present; the failure is ordering and active-configuration proof, not missing bytes.

## Pipeline Integrity and Per-Pair Cardinality

| Check | Measured | Verdict |
|---|---:|---|
| Coordinator Steps 00–05 | 6 separate, non-trivial files; 7,566–22,826 bytes | PASS |
| Step 00 taxonomy | 50 aesthetics + 50 emotions + 50 frames + 50 genres; 14,186 bytes | PASS |
| Step 02 concepts | 12 distinct concepts with panel pressure | PASS |
| Step 05 pairs | exactly 6; runner-up rationale + six-entry cut ledger | PASS |
| Step 06 facets | 6 separate files; 10/10/11/10/10/10 weighted pair-specific facets | PASS |
| Step 07 guides | 6 separate files; 6 substantive guides per pair | PASS |
| Step 08 packages | 24 total, 4 per pair | PASS |
| Step 09 packages | 24 total, 4 per pair | PASS |
| Step 10 packages | 24 total, 4 per pair | PASS |
| Step 11 finals | 6 separate files | PASS |
| No-skip spine | Steps 07, 09, 10 present for all 6 pairs | PASS — canonical |
| Stub/debris scan | no TODO/TBD/lorem/unfilled-variable debris in the 36 canonical pair files | PASS |
| Step 08 hard package ranges | prompt 950–1,000; lyrics 2,255–3,055; 40–46 sung lines | PASS under R4.1 override |
| Step 09 hard package ranges | prompt 890–960; lyrics 2,262–3,054; 40–46 sung lines | PASS under R4.1 override |
| Step 10 hard package ranges | prompt 890–960; lyrics 2,341–3,054; 40–46 sung lines | PASS under R4.1 override |
| Step 10 Sunna surface audit | 24/24 packages | PASS |
| Final Sunna surface audit | 6/6 packages | PASS |
| Step 11 ↔ standalone final | SHA-256 byte-identical, 6/6 | PASS |
| Final cross-pair 5-gram Jaccard | maximum 0.000 in the measured normalization | PASS vs 0.18 ceiling |

The explicit run-local **36–56** sung-line band controls. The generic 70–120 exam is not applied and no song is failed for being 40–44 lines. Runtime could not be audited inside the evidence boundary because `RUN_STATE.md` and `INDEX.md` were explicitly forbidden.

## Structured Evidence Block

Measured values sit beside, and do not replace, the Somatic verdict.

| # | Style chars | Exclude chars | Lyrics chars | Sung lines | Dedication chars | End rhyme | Exact-line return | Mean words/line | Alliteration /100w | Chorus-exempt duplicate ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 956 | 578 | 2,844 | 44 | 368 | .659 | .227 | 6.30 | 11.91 | .000 |
| 02 | 957 | 659 | 2,392 | 40 | 388 | .300 | .300 | 4.70 | 12.23 | .000 |
| 03 | 956 | 393 | 2,753 | 42 | 399 | .333 | .310 | 5.29 | 11.71 | .000 |
| 04 | 946 | 514 | 2,767 | 41 | 413 | .415 | .390 | 5.85 | 11.67 | .000 |
| 05 | 945 | 454 | 2,578 | 40 | 437 | .325 | .325 | 4.78 | 14.66 | .071 |
| 06 | 935 | 610 | 2,639 | 42 | 367 | .357 | .238 | 5.83 | 11.43 | .000 |

All style prompts pass **850–1,000** and the preferred 870–960 target; none hugs the ≥985 ceiling. All lyrics pass `<5,000` and the ≤4,800 target. All line counts pass 36–56 and avoid the ≤38 floor-hug flag. Every prompt ends in terminal punctuation. Every package has full EMO headers, at least one SFX cue, a separate EXCLUDE field, and no sung digits.

### EMO-tag balance

| # | Dominant taxonomy label | Count | Judgment |
|---:|---|---:|---|
| 01 | Excitement | 3 | balanced; no single-emotion domination |
| 02 | Desire | 3 | balanced; no single-emotion domination |
| 03 | Delight | 3 | balanced; no single-emotion domination |
| 04 | Confidence | 3 | balanced; no single-emotion domination |
| 05 | Delight | 4 | varied arc remains present |
| 06 | Eagerness | 3 | balanced; no single-emotion domination |

No bare AWE/INDIGNATION tag appears. No house-lexicon phrase from `vault/gates.yaml` appears in a finalist. Repetition does not collapse outside chorus/refrain blocks; exact chorus repetition is exempt and requires no defence.

### Boundary, house-lexicon, and one-fact FLAGs

- **Boundary hugging:** clear, 6/6.
- **House lexicon:** clear, 6/6.
- **One-fact rule:** **FLAG, 6/6**, routed to the Somatic read. A high-recall count finds 11/13/8/13/5/9 unique sung lines containing number words. Much of this is deliberate title/coordinate return or ordinary object count rather than curriculum, so this is not an auto-fail; nevertheless every song exceeds the literal “one sung numeric-fact line” tripwire.
- **Repeated-line / n-gram collapse:** clear outside exempt chorus/refrain blocks. The semantic portfolio-repetition failure is separate and is recorded under Expanded Hit Quality below.

## Eligibility Score — Core 7 Properties

Scores are independent of the assigned ACCESSIBLE/AMBITIOUS route. All six meet the numerical ACCESSIBLE threshold; that uniformity is itself evidence that the intended 3/3 barbell collapsed toward one surface lane.

| # | Body | Hook | Emotional TAM | Specificity | Ease | Co-discovery | Sonic threshold | Average | Classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | 5 | 5 | 5 | 5 | 5 | 4 | 2 | 4.43 | ACCESSIBLE |
| 02 | 5 | 5 | 5 | 5 | 5 | 4 | 2 | 4.43 | ACCESSIBLE |
| 03 | 5 | 5 | 5 | 5 | 5 | 5 | 2 | 4.57 | ACCESSIBLE |
| 04 | 5 | 5 | 4 | 5 | 5 | 4 | 2 | 4.29 | ACCESSIBLE |
| 05 | 5 | 5 | 4 | 5 | 5 | 4 | 2 | 4.29 | ACCESSIBLE |
| 06 | 5 | 5 | 4 | 5 | 5 | 5 | 2 | 4.43 | ACCESSIBLE |

The Sonic Threshold score is 2 because all six open with an immediate bright identity riff/SFX rather than calm, dangerous silence, or a dynamic surprise. That miss does not prevent ACCESSIBLE classification because the other six properties are strong.

## Expanded Hit Quality and Set-Level Eligibility

| # | Music uniqueness | Impact | Depth | Earworm | Uniqueness in run | Unresolved ache |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 4 | 4 | 3 | 5 | **2** | 3 |
| 02 | 3 | 3 | 3 | 5 | **2** | 3 |
| 03 | 4 | 4 | 4 | 5 | **2** | 3 |
| 04 | 3 | 3 | 2 | 5 | **2** | 2 |
| 05 | 4 | 4 | 3 | 5 | **2** | 3 |
| 06 | 4 | 4 | 4 | 5 | **2** | 3 |

**Set-level Check 12 average: 2.0/5 — RUN FAIL.** The sonic identities and props differ, but the same one-sentence account fits every finalist: Sunna selects a bright object in a public counter/errand scene, gives one adult an exact future coordinate, receives an answer, crosses one substitution seam, and repeats the title before the terminal riff. Emotional core, literary mode, energy profile, stakes, and form are therefore materially narrower than the labels `3 ACCESSIBLE + 3 AMBITIOUS` claim. Low lexical Jaccard does not rescue semantic skeleton reuse.

## Score Table — Simple Surface / Complex Engine 16-Point Gate

Legend: `P` = PASS; `R` = REPAIR REQUIRED.

| # | Gate | 01 | 02 | 03 | 04 | 05 | 06 | Evidence / repair |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Human singer | P | P | P | P | P | P | Sunna is an adult subject with a concrete want; the addressed/answering figure is one invented adult. |
| 2 | Body-first opening | P | P | P | P | P | P | Diner/purse/stool; ball/windmill; counter/jumpsuit; lime/chair; booth/photo; page/counter all arrive in the opening surface. |
| 3 | Adoptable hook | P | P | P | P | P | P | All title hooks are 3–5 words, vowel-forward, and singable in one breath. |
| 4 | Hook recurrence / mutation | P | P | P | P | P | P | Exact returns recur; reply, coordinate, or punctuation changes the social meaning. |
| 5 | Chorus clarity | P | P | P | P | P | P | Choruses are invitations/declarations, not policy or procedure. |
| 6 | Voice + pulse survival | P | P | P | P | P | P | Title, lead, bass/pulse, and answer architecture remain legible without sidecars. |
| 7 | 15–30 second clip survival | P | P | P | P | P | P | Hook, object/room, and invitation arrive early enough for a short excerpt. |
| 8 | Golden Seed Alloy pressure | P | P | P | P | P | P | `THE EXACT INVITATION` determines choice → coordinate → answer → seam → exact return → terminal riff in every package. |
| 9 | Mythic image ladder reaches lyric/hook | P | **R** | P | **R** | **R** | P | 01 personifies Friday and returns to the shared table; 03 wears Saturday through the zipper; 06 turns grammar/future into names on a pink page. In 02 the “windmill cuts maybes,” in 04 the marina-scale lime escalation, and in 05 the clockwork/classification court remain sidecar-only or are deliberately removed. Strict Gate 9 does not permit sidecar-only pressure. Return 02/04/05 to Step 07, then flow 08–11. |
| 10 | EMO dramaturgy depth | P | P | P | P | P | P | Apprehension/suspense becomes trust, relief, delight, or triumph after an audible answer; no single-label domination. |
| 11 | Production dramaturgy | P | P | P | P | P | P | Each named sound has a job: bookend, cradle, answer, rupture/substitution, or afterglow. One world/one seam is preserved. |
| 12 | Panel pressure / anti-blandness | P | P | P | P | P | P | Step-09/10 ledgers show concrete critic-forced line, answer, coordinate, and seam changes. The set-level sameness failure is recorded separately under expanded Check 12. |
| 13 | Clean Suno lyrics | P | P | P | P | P | P | `[Theme:]` immediately precedes `[SONG FORM:]`; full EMO/performance headers; SFX; no prompt debris; no sung digits; 36–56 override satisfied. |
| 14 | Producer-grade v5.5 prompt | **R** | **R** | **R** | **R** | **R** | **R** | Current hard format passes: one dense 850–1,000-char paragraph, separate EXCLUDE, no real artist, no banned opener, terminal punctuation. Binding legacy subcheck still requires an explicit age band plus accent/ethnicity descriptor; all six say only `Adult female` and omit accent/ethnicity. Return to Step 08 and propagate through 09–11. |
| 15 | Full package completeness | P | P | P | P | P | P | Run-scoped package is complete across Steps 06–11, final file, dedication, credits, and this QA report; standalone render files contain the paste-ready subset. |
| 16 | Lineage & Credit | P | P | P | P | P | P | Six measured dedications + companion credits; living-scene/player lineage named; no positive style field names a real artist or named break; no-sample honesty present. |

## Human-Subject Backstop

**Verdict: CLEAR.** The lyrics concern Sunna herself and one invented adult. No real private person, victim, harmed person, minor, identifying tuple, recent tragedy, or reconstructed case appears. Public musicians occur only in factual recording/lineage credits; the captions do not claim that any living person is the song's subject, collaborator, endorser, or sampled content. No online check is triggered because no name co-occurs with a harm/identifiability context.

## Somatic Gate — 3 Hyper-Skeptics (PRIMARY)

The bloc asks one question: **distinctive enough to be Lofn/Sunna, or generic?** Two NO votes block a song. Every NO includes a counter-move; these are observations for a coordinator repair brief, not edits performed by QA.

### 01 — Friday Looks Good on Me — **PASS, 2 YES / 1 NO**

- **Body-and-Hook Skeptic — YES.** The red purse, stool, pie, exact time, and two-forks answer make a complete event; the hook survives flat speech.
- **Second-Listen Skeptic — YES.** One fork becomes two and the same title changes from self-display to mutual booking. No named-corpse condition fires inside the song.
- **Portfolio-Corpse Skeptic — NO.** The song is individually alive but shares the run's choice/coordinate/answer/seam/return chassis almost exactly; the production distinction is doing more differentiation than the lyric form. **COUNTER-MOVE:** re-derive the middle around the escalator-to-stool threshold so room geometry—not another coordinate chorus—owns the second movement.

### 02 — Make It Nine — **BLOCKED, 1 YES / 2 NO**

- **Body-and-Hook Skeptic — YES.** Blue ball, windmill, nine-oh-five counter-time, and mutual putt are instantly graspable; `Make It Nine` is highly adoptable.
- **Second-Listen Skeptic — NO.** The sidecar's “moving blade cuts maybe” insight never reaches the lyric or hook; the ball/blade motif remains literal game business. **COUNTER-MOVE:** make the blade interrupt the long-A hook's meter once and let the two mutual rolls restore it, so the motif transforms in song without an explanatory line.
- **Portfolio-Corpse Skeptic — NO — single dominant repeated section.** The title block does most of the load-bearing work while several connecting lines feel engineered around the fixed template. **COUNTER-MOVE:** give `nine-oh-five` one concise answer-hook return of its own and remove one redundant title block, while preserving the exact final title.

### 03 — Zip Me Into Saturday — **PASS, 3 YES / 0 NO**

- **Body-and-Hook Skeptic — YES.** Zipper, stop instruction, chosen turn, exact time, and answer are inseparable from the title.
- **Second-Listen Skeptic — YES.** `Saturday` becomes something she can be zipped into; permission, fastening, and the deliberate turn transform dependence into pace-setting agency.
- **Portfolio-Corpse Skeptic — YES.** The fricative hook, consent blocking, soul-pop response stack, and hard-guitar turn give this song a distinct body even inside the shared seed chassis. No named-corpse condition fires.

### 04 — One Lime, Red Chair — **BLOCKED, 1 YES / 2 NO**

- **Body-and-Hook Skeptic — YES.** The title is immediate, the request is wonderfully small, and `Salt too?` supplies adult counter-choice rather than automatic compliance.
- **Second-Listen Skeptic — NO — motif that never transforms.** The lime and chair recur, but the hook remains semantically unchanged after the salt answer; the sidecar explicitly protects “lime stays lime,” leaving too little second-listen engine in the sung field. **COUNTER-MOVE:** let the salt counter-question alter one answer line or vocal owner before the exact chorus returns, so reciprocity changes the motif without turning the lime into symbolism.
- **Portfolio-Corpse Skeptic — NO — single dominant repeated section.** Three exact four-line chorus blocks carry more emotional weight than the verses/reply/receipt earn; deliberate repetition is legal, but here the surrounding material is too slight to make the third return consequential. **COUNTER-MOVE:** replace one middle chorus with a short, room-locked call-and-response built from `Salt too? / Yes—red chair, eight`, then keep the final chorus exact.

### 05 — Pretty Busy Tomorrow — **PASS, 2 YES / 1 NO**

- **Body-and-Hook Skeptic — YES.** Booth flash, glossy picture, pink slip, tear, time, and adult reply form a clean visual/social event; the title is an excellent ordinary phrase with a second use.
- **Second-Listen Skeptic — YES.** The title moves from apparent unavailability to a future Sunna schedules herself, and the shared receipt action earns that semantic turn even though the formal mythic-ladder gate remains unmet.
- **Portfolio-Corpse Skeptic — NO.** Its scene is fresh, but the emotional and formal chassis still matches the other five appointments too closely. **COUNTER-MOVE:** let the torn receipt, rather than another coordinate-return sequence, determine the stanza shape and answer cadence.

### 06 — Hot Girl Future Tense — **PASS, 3 YES / 0 NO**

- **Body-and-Hook Skeptic — YES.** Pink page, pen, names, Thursday at ten, and the copy-shop counter make the grammar joke physically playable.
- **Second-Listen Skeptic — YES.** The question mark becomes a declaration only after the pen is picked up; the title genuinely transforms rather than merely repeats.
- **Portfolio-Corpse Skeptic — YES.** The chant's comic grammar, question/declaration architecture, and pen-caused stomp-to-two-tone seam remain unmistakable despite the shared seed form. No named-corpse condition fires.

## Blocking Fails

1. **ICB head-position failure:** exact ICB begins at offset 2,781 in all six dispatches, not at the head.
2. **Phase-1 transformation configuration failure:** no three fully labeled, independently debated BASELINE / GROUP TRANSFORM / SKEPTIC TRANSFORM configurations.
3. **Expanded eligibility RUN FAIL:** Uniqueness in Overall Run averages 2.0/5; the intended 3/3 barbell collapses into six accessible exact-invitation songs with one semantic chassis.
4. **Cathedral Gate 9:** 02, 04, and 05 keep the required image escalation in sidecars instead of lyric/hook.
5. **Producer prompt legacy subcheck:** all six omit the required explicit age band and accent/ethnicity descriptor.
6. **Primary Somatic Gate:** 02 and 04 receive two NO votes and are BLOCKED.
7. **Blind calibration:** coordinator decode remains pending; no candidate-level blind verdict exists yet.

## Required Repairs

1. **Return to Phase 1 / dispatch construction — new run chain required.** Put the frozen ICB in the contractually permitted head position and record three fully labeled debate configurations with real inter-seat disagreement. The ICB is frozen; do not patch it in place mid-run.
2. **Return to Step 05 for set differentiation.** Re-derive at least the three AMBITIOUS slots so emotional core, literary mode, energy/stakes, and form—not only genre/prop—differ. A cut-ledger reserve or re-derived pair angles is appropriate; any replacement must traverse a fresh 06–11 chain.
3. **Return 02/04/05 to Step 07, then 08–11.** Bring one ordinary→specific→strange/mythic→body image movement into lyric or hook without reintroducing the abstract lecture R4.1 correctly rejects. The strict gate and Sunna's anti-abstraction instinct are in tension; the coordinator must resolve that tension explicitly rather than silently waive either side.
4. **Return all six to Step 08, then propagate 09–11.** Add the legacy-required vocalist age band plus accent/ethnicity descriptor while preserving the current dense paragraph, 850–1,000-char band, and artist-name prohibition.
5. **Apply the Somatic counter-moves to 02 and 04** through the coordinator, not verbatim from this blind judge. Re-QA in a fresh context afterward.
6. **Coordinator decodes the six sealed blind sets** from the withheld keys. Candidate below decoy → REPAIR; decoy above golden → judge audit/re-run. Do not send the keys back into this judge context.

This is the first QA attempt, so the no-progress predicate and mandatory REDIRECT rule have not fired. The set-level differentiation repair is nevertheless best treated as a sideways re-derivation, not line-polish on six near-identical chassis.

## Failure-ledger entry appended (on SHIP)

**No.** Nothing receives SHIP, and the audit evidence boundary excludes ledger writes.

## Final Recommendation

**Do not render, rank as audio, publish, or spend on this set.** Preserve the sealed blind report and deterministic evidence, but treat the present six as a failed text-package run requiring a fresh Phase-1/pair chain. The best individual text packages are 03 and 06; 01 and 05 pass Somatic but remain structurally/package-blocked; 02 and 04 are Somatic-blocked. Finished audio remains **UNAUDITED** even after text repair and will require a separate blind-first render audit.
