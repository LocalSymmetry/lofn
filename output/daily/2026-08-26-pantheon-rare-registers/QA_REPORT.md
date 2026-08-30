# Lofn QA Report — `2026-08-26-pantheon-rare-registers` (music)

> ⭐ **AMENDED 2026-08-27 after the repair round — the verdict is SHIP 6 of 6, and the P05 V1 HOLD is lifted.** *(This banner read “SHIP 5 · REPAIR-THEN-SHIP 1” until P03's repair landed and closed F5. Corrected by the coordinator 2026-08-27, and the superseded wording kept visible rather than overwritten — a headline that outlives its own tables is the exact defect this run spent the day on, and it should not survive in the artifact of record.)* Start at **Verdicts** and the **Amendment** immediately below it. Every superseded finding is preserved in place with a dated `CLOSED` stamp rather than deleted — **a verdict file that overwrites its own before-state cannot be audited, and the size of a finding is information a later reader needs.** Where this report was wrong, it says so at the point where it was wrong.

**Auditor:** `lofn-qa`, fresh subagent, no part in generating any artifact in this run.
**Method:** every number below was re-measured from disk by this auditor. The ICB was stripped as an exact byte-0 prefix before parsing; the four heading dialects (`## 2. LYRICS` / `## 2. SUNO LYRICS` / `## SUNO LYRICS` / `## 2. LYRICS — MAX config…`) were all handled. Return metrics use the house instrument `scripts/measure_soundcraft.py → profile()` on its own `sung()` parser, not a local re-implementation.
**Independence caveat, declared:** this auditor was handed the selection judge's conclusions as *leads*. Every lead was re-derived from disk before the judge's text was consulted for agreement. No `GATE_REPORT.json` exists for this run, so the Structured Evidence Block is measured in-prose (helper is fail-open; its absence does not block).
**Not done, and it matters:** the §4.5 blind golden+decoy comparative ranking was not run — the coordinator did not assemble a blind set, and a judge who assembles his own blind set is not blind. Treat that calibration as **unperformed**, not as passed.

---

## Verdicts — AMENDED 2026-08-27 after the repair round

- **Pipeline Integrity:** **PASS**
- **Package (modality contract):** **PASS** *(was: REPAIR REQUIRED)*
- **Human-Subject:** **CLEAR** (all six selected)
- **Overall:** **SHIP — 6 of 6** *(was: REPAIR-THEN-SHIP 5 of 6 · HOLD P05 V1)*

> **The P05 V1 HOLD is LIFTED.** F1, F2, F3, F4 and F5 are **CLOSED**, each re-measured by this auditor from disk after the repair. **F6 and F7 stay open on P03 V3 as non-blocking craft notes** — both are FLAG-class by `gates.yaml`, both were routed to the somatic read in this report's own words, and that read was YES from all three Skeptics from the start. **Holding a song on them would be inventing a gate.** **No swap to P05 V4** — see the ruling.
>
> ⛔ **A 6-of-6 SHIP here is not the object the zero-rejection tripwire is aimed at, and must not be quoted as one.** `EXECUTION.md` §7.5 fires on a run reporting *0 repairs + 0 holds + 0 quarantines*. This one produced **eight findings, one HOLD, six executed repairs and two standing notes**, and the six-of-six exists **because** the no was real and was acted on. A clean first pass on 24 songs would have been suspicious; a clean second pass after a repair round is what a working gate looks like.

**The original verdict, as issued 2026-08-27 pre-repair, and why it is preserved below rather than overwritten:** *Overall — REPAIR-THEN-SHIP (5 of 6) · HOLD (P05 V1).* Five songs carried one-token or one-row repairs; `P05 V1 · A Light On The Front Of Me` carried an undeclared company token, a false clean-scan row, one Hyper-Skeptic NO and a declared GRAIN-LAW production risk, and borderline defaults to HOLD. **Every finding below is left exactly as filed, with a dated CLOSED stamp where it closed.** A verdict file that deletes its own before-state cannot be audited, and the size of a finding is information a later reader needs.

---

## ⭐ AMENDMENT — 2026-08-27, the repair round

*All numbers in this section were re-measured by this auditor from disk after the repairs, not accepted from the pairs or the coordinator. The packages grew — P01 207,111 · P02 213,523 · P03 188,426 (untouched) · P04 221,829 · P05 228,366 — and the **ICB is still an exact byte-0 prefix at 148,555 bytes in all six**, re-verified with `bytes.startswith()` after every edit.*

### Run-wide census, re-run by this auditor
**72/72 pasted fields parsed · 0 brand tokens · 0 hard-pan tokens.** The forbidden-token classes are gone from the run.

### P05 V1 — the HOLD is lifted
Verified: mp **957** · ex **696** · ly **4792** · **62** sung lines · **0** emotion words · 66 BPM inside 62–74 · `[Theme:]`+`[SONG FORM:]` first two lines · 5 Disc channels · 10 EMO headers, 0 bare · 2 SFX · **prompt prohibitions 5 → 0**.

**The check that could have failed, and did not.** A repair that lives only in `## Production dramaturgy` is not a repair, because that section is never pasted into Suno. All five changes are inside the **pasted** fields: the count in the style paragraph, in the `[SONG FORM:]` line, in `Disc_Rhythm` (`dry_shutter_click_every_5_seconds | in_at_R2_out_at_seam_back_at_end | off_grid`) and in the refrain-2 header; the stop and the strike in the style paragraph, `Disc_Texture` and the seam header; the caught breath in `Disc_Vocal` and the seam header. **That is THE GRAIN LAW satisfied as written — the objection is answered in the lyric and the form, not in the production spec.**

**"No sung line changed" — verified by instrument, not accepted.** All four soundcraft metrics are byte-identical to this auditor's pre-repair values (`end_rhyme` 0.129 · `line_return` 0.677 · `words_per_line` 8.16 · `allit_per_100w` 7.91); any changed line moves at least one. Structure re-derived independently: **5 distinct sung blocks, the 7-line refrain byte-identical ×6**, fields 8/6/4/2 each once, 6×7+20 = 62.

**⛔ THE SOMATIC NO IS WITHDRAWN, AND SO IS MY COUNTER-MOVE.** The condition cited was *"a motif that never transforms"* — and the checklist's own wording admits three qualifying transformations: **recontextualized**, inverted, paid off. Recontextualization does not require the words to move. Refrain 5 now arrives under four changed conditions concentrated on the line carrying the discriminator, and the sharpest is a **vocal** decision, which this run's R5 categorically separates from an arrangement: refrains 1–4 run out of air on *me*, teaching the ear four times that this is the phrase it cannot catch; **at the seam and only there she takes the breath**, and the sentence that declines to certify becomes the most intelligible thing in the record; then the Out-Refrain takes the air away again and decides nothing. That is the register performed on the listener rather than described to them.

⭐ **And the counter-move this report proposed was wrong.** Carrying the lamp line into the Out-Refrain would have added **a last image**, which P05's assigned refused element forbids by name (*"no summary line and no last image"*), and would have given the reach a return — converting an unanswered ask into a motif she owns, which is a resolution of the doubt the pair was assigned not to grant. **The objection behind it was sound; the fix named would have traded the song's spine for its ratio, and the pair was right to refuse it.** Their alternative answers the same objection from underneath and better: the angle is THE INSTRUMENT SHE DOESN'T HAVE, and the record is now *made of* an instrument that measures perfectly every five seconds for four minutes and is **never once pointed at her**, still running at the cut. The Field Four lamp stops being a passing mention and becomes the moment the listener understands what the ticking has been. *Recorded in full because a Skeptic whose counter-move is silently dropped has not been overruled, he has been ignored — and because this one deserved to be overruled.*

### ⛔ NO SWAP to P05 V4
The selection judge's stated reason for preferring V4 was dimension (1): *"it has the pair's only load-bearing positive motion instruction — a slow footfall — where V1's seam is a layer relationship the generator smooths first."* **That reason is now void, and it has inverted.** V1 carries a positive motion event running three and a half minutes plus a stop, a struck transient and a drop-out; **V4 still has one footfall and — declared, deliberately — still keeps the delay-time seam.** Swapping now would trade a repaired weakness for an unrepaired one on the exact axis that made V4 the runner-up. The judge's dimension (2) was a writing preference formed reading all four in context, and QA has no standing to overturn a clean-context judge on taste.

### ⭐ THE FALSIFIABLE RENDER PREDICTION (pre-registered, still pre-render)
*Filed here in numbers with its falsification condition, before any audio exists, per `vault/PREREGISTRATION_STANDARD.md`. `lofn-render-audit` scores it.*

> **Hypothesis.** Under THE GRAIN LAW, a **coarse positive event** survives the generator where a **fine-grained inter-layer relationship** is averaged away.
> **Prediction, P05 V1.** At the seam (refrain 5): the **stopped click is perceptible as an absence** and the **struck transient is perceptible as the only hard attack in the song**, in a blind muted-of-prompt listening pass — while the 90 ms → 0 ms return closure is **not** independently identifiable by ear.
> **Second prediction, same render.** The five-second shutter count is **not** promoted into a groove: no kick, snare, hat or looped percussion appears, and the click stays off-grid against the 66 BPM pulse. *(This is the live risk of removing `No drums.` from the style paragraph. The ban survives in `Disc_Rhythm: no_drums_no_percussion` and in three new exclude tokens — `drum fill, snare build, percussion loop` — and the mitigation is judged adequate, not certain.)*
> **Falsification.** Either the stop/strike are inaudible as events, **or** a drum layer appears. Either outcome refutes the repair's premise for this song.
> **Default expectation, stated out loud so a null is cheap:** the two carriers survive and the delay does not — that is the whole reason the repair was ordered. **If the delay turns out to be independently audible, the original seam did not need repairing and this report's THIRD-SKEPTIC caveat was wrong.**
> **Scope, honestly.** n = 1 song. This is a probe, not a law. It transfers past this song only if it replicates.

### Provenance audited before acceptance — `vault/field_token_lexicon.yaml`
P05's repair invoked a **new** `exclude_permitted` group (`autotune`, `melodyne`, `pro tools`, `ableton`, `808`) that narrows a finding filed against it. **A rule that dissolves the finding it is cited against is exactly the artifact to audit, so it was audited rather than accepted.** Lines 81–82 of that file attribute the group to *this report's own de-minimis ruling of 2026-08-26* and scope it correctly — *"still a brand token in the music prompt or the lyrics."* **It codifies a call QA already made; it was not written to excuse the pair.** Direction-of-error is clean in both directions: the pair ran the census, found `autotune` ×4 **against itself**, and then reported the count inflated by four when the ruling made it a non-violation.

### ⛔ THE COORDINATOR'S OWN INSTRUMENT FAILED FIRST, AND IT BELONGS IN THIS FILE
`scripts/check_field_census.py` — built to enforce *"the coordinator re-measures what it republishes,"* which is this report's central recommendation — **shipped two false-positive classes on its first live run**: it read **DEDICATION fences as pasted fields** and flagged `scruggs` / `monroe` / `jamerson`, and it flagged **`autotune` in excludes against QA's own de-minimis ruling**.

⭐ **And the reason the control did not catch it is the finding:** *"my synthetic control passed while the parser was wrong, because I built the control from the same mental model as the parser."* A positive control authored by the parser's author tests the author's model, not the parser — the same **direction of error** `CLAUDE.md` names, arriving one layer up, in the instrument built to catch it.

**This is recorded because a verdict file that logs only the pairs' instrument failures and not the coordinator's would be precisely the asymmetry this report spent its length naming.** Six pairs positive-controlled their own scanners and every one found a defect in it; the coordinator's census did the same thing and its failure is the same class. That is the honest shape of the run.

---

## ICB Integrity

| check | result |
|---|---|
| `CREATIVE_CONTEXT.md` present as an **exact byte-0 prefix** of each shipped package | **6 / 6 ✓** — verified with `bytes.startswith()`, not by heading match |
| ICB size | **148,555 B**, identical in all six |
| remainder after the prefix | P01 51,900 · P02 55,638 · P03 39,871 · P04 63,450 · P05 53,511 · P06 61,219 |
| `(after ` occurrences in the ICB | **19** raw → **18 panel-voice tags** + 1 non-voice use at `CREATIVE_CONTEXT.md:977` (*"R3's addition (after Ellena's seat)"*). **Voice count == 18 ✓** — the raw count is not the gate, and a naive `== 18` assertion on the raw string would have false-failed this run. |
| additions below the block only, no edit above it | ✓ |

**Pipeline / no-skip:** steps 06, 07, 08, 09, 10 and the enhanced package exist as **separate per-pair files for all six pairs**; coordinator steps 00–05 exist as separate files; 6 × 4 = **24** outputs; `RUN_STATE.md` reports 50 done / 0 pending / 0 quarantined. **No collapsed rollup. The editorial spine (07 / 09 / 10) is present for every pair.** The run is **CANONICAL**.

*Minor, non-blocking:* `RUN_STATE.md` records LF-normalised sizes for eight files whose on-disk (CRLF) size differs (e.g. `pair_02_step10_final_package_enhanced.md` 203,420 vs 204,193 on disk). The file's own header warns about exactly this. Self-documented, not drift.

---

## Structured Evidence Block — THE SELECTED SIX, measured

⭐ **RE-MEASURED 2026-08-27 post-repair. These are the current numbers; the pre-repair value is shown struck beside any that moved.**

| # | song | pair·V | prompt chars *(850–1000; write 870–960; hug ≥985)* | exclude chars *(400–900)* | lyrics field *(<5000; target ≤4800)* | sung lines *(70–120)* | BPM / band | dedication *(≤500)* | R2 emotion words | hard-pan token |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 | Two Claps For Nothing Yet | P06 V4 | **943** ✓ | **678** ✓ | **4797** ⚠️ *(3 under target)* | **80** ✓ | 143 / 140–160 ✓ | **361** ✓ | **0** ✓ | none ✓ |
| 2 | You Go On The Next One | P01 V2 | **914** ✓ | **533** ✓ *(was 554)* | **4243** ✓ | **80** ✓ | 124 / 122–138 ✓ | **447** ✓ | **0** ✓ | **none ✓** *(was ⛔ `hard stereo panning`)* |
| 3 | Somebody Watch This | P02 V4 | **915** ✓ | **708** ✓ *(was 727)* | **4252** ✓ | **78** ✓ | 106 / 106–120 ✓ | **487** ✓ | **0** ✓ | none ✓ |
| 4 | It Changed One Line | P03 V3 | **949** ✓ | **865** ✓ | **4500** ✓ *(was 4733)* | **74** ✓ *(was 72, on the floor-hug threshold)* | 94 / 90–104 ✓ | **384** ✓ | **0** ✓ | none ✓ |
| 5 | No Line For The Name | P04 V2 | **951** ✓ *(was 952)* | **705** ✓ | **4780** ✓ *(was 4748)* | **80** ✓ | 76 / 76–88 ✓ | **402** ✓ | **0** ✓ | none ✓ |
| 6 | A Light On The Front Of Me | P05 V1 | **957** ✓ *(was 935)* | **696** ✓ *(was 642)* | **4792** ✓ *(was 4467)* | **62** ⚠️ **below band — declared + pre-authorised** | 66 / 62–74 ✓ | **492** ✓ | **0** ✓ | none ✓ |

**Brand / artist tokens in the three pasted fields: 0 for all six** *(was ⛔ on P04 V2 and P05 V1)*. **Run-wide: 0 across 72/72 pasted fields.**

**Every char count above reproduces the pairs' own verification tables to the character**, before the repair and after it. Pre-repair: P01 922/914/950/934, P02 917/953/910/915, P04 959/952/941/958, P05 935/957/954/937, P06 960/938/946/943 — all exact. FINDING 9's convention (count the pasted string, not the fenced block) is the right unit and is what this audit used throughout.

⚠️ **Two boundary-huggers to carry forward, both by construction rather than by drift:** P06 V4 at **4797** and P05 V1 at **4792** sit 3 and 8 chars under the 4,800 soft target, with 203 and 208 to the hard cap. P05 V1's margin is the residue of an honest compression from a first pass that measured **5022 — over the hard cap** (disclosed by the pair, unprompted). **Re-measure both after any future edit; neither survives an eyeballed change to a Disc channel or an EMO header.**

### Suno contract, per selected song
| check | P06V4 | P01V2 | P02V4 | P03V3 | P04V2 | P05V1 |
|---|---|---|---|---|---|---|
| `[Theme:]` then `[SONG FORM:]` as first two lines | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 `Disc_` channels | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| full EMO headers | 9 | 10 | 11 | 12 | 14 | 10 |
| bare `AWE` / `INDIGNATION` | **0** | **0** | **0** | **0** | **0** | **0** |
| every first-slot EMO term found in `EMOTION_TAXONOMY` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ≥1 standalone SFX cue | 1 | 1 | 1 | 2 | 1 | 2 |
| digits in sung lines | **0** | **0** | **0** | **0** | **0** | **0** |
| sung numerals spelled in words | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| research-derived numeric facts sung | 0 | 1 *(ninety-nine, answered)* | 1 *(nineteen-oh-six, answered)* | 0 | 1 *(twenty-one years, answered)* | 1 *(five seconds, answered)* |
| house-lexicon hits | **0** | **0** | **0** | **0** | **0** | **0** |

**One-fact rule:** ≤1 *research* fact per song, each at the hinge and answered rather than recited. Ordinary counting words (`one`, `two claps`, `room four`) are structure, not facts — that is the correct unit, and reading the rule as a quota (as P06's tables do: *"expected: exactly one"*) is the instrument error the judge already named.

### Return floors — `measure_soundcraft.profile()`, FLAG-only per `gates.yaml`
| song | `strict_end_rhyme` ≥0.30 | `line_return` ≥0.20 | `words_per_line` ≤7.5 | `allit_per_100w` ≥11.0 | assigned §5 rhyme logic | declared? |
|---|---:|---:|---:|---:|---|---|
| P06 V4 | 0.237 ⚠️ | 0.463 ✓ | 7.88 ⚠️ | 14.44 ✓ | *stripped — debt paid, 3 returns named* | assigned |
| P01 V2 | 0.350 ✓ | 0.450 ✓ | 6.15 ✓ | 8.54 ⚠️ | *"kept, loose… no rhyme debt owed"* | n/a |
| P02 V4 | 0.487 ✓ | 0.526 ✓ | 5.46 ✓ | 9.15 ⚠️ | *"kept and cheap on purpose"* | n/a |
| P03 V3 | ~~0.236 ⚠️~~ → **0.473 ✓** | 0.284 ✓ | ~~8.18 ⚠️~~ → **7.23 ✓** | ~~11.21~~ → **13.83 ✓** | ⛔ ***"End-rhyme kept."*** | ✅ **repaired 08-27; all four floors clear** *(cross-ref was mistyped "F3" here; the finding is **F5**)* |
| P04 V2 | 0.362 ✓ | 0.375 ✓ | 6.49 ✓ | 16.96 ✓ | *stripped — debt paid, 3 returns named* | declared, paid |
| P05 V1 | 0.129 ⚠️ | 0.677 ✓ | 8.16 ⚠️ | 7.91 ⚠️ | *stripped — debt paid, 3 returns named* | wpl declared; allit not |

P01/P02/P06 are Sunna pairs and the metaprompt exempts them from the numeric floors in writing (*"⛔ For the Sunna pairs, the bottom three rungs do not apply at all"*). Their flags are recorded, not charged.

### Cross-pair distinctiveness (the six that ship)
All fifteen pairings: `SequenceMatcher` **0.004 – 0.027** (ceiling 0.42) · 5-gram Jaccard **0.000** (ceiling 0.18). **The six selected songs share essentially no text.** Six distinct registers, six distinct neighbours, six non-overlapping tempo bands, all held.

---

## Score Table

| # | Gate | Verdict | Evidence | Repair |
|---|---|---|---|---|
| 1 | Pipeline integrity / granularity | **PASS** | 50/50 artifacts; 07·09·10 present per pair; no rollup | — |
| 2 | ICB byte-0 prefix + 18 voices | **PASS** | `startswith()` true 6/6; 18 voice tags | — |
| 3 | MUSIC PROMPT band + dense paragraph | **PASS** | 914–952, none ≥985, all prose, all end as sentences | — |
| 4 | EXCLUDE present + in band | **PASS** | 554–865 | — |
| 5 | Lyrics field cap | **PASS** | 4243–4797 | note the 4797 hug |
| 6 | Sung-line band | **PASS w/ declared break** | five at 72–80; P05 V1 at **62**, pre-authorised by `06_music_handoff.md:57` | — |
| 7 | EMO headers / taxonomy / no bare tags | **PASS** | 9–14 headers, 0 bare, all taxonomy | — |
| 8 | R2 — no emotion word in any sung line | **PASS** | independent 485-term sweep; 0 genuine hits; every raw hit adjudicated below | — |
| 9 | Hard-pan absent from pasted fields | ~~FAIL~~ → **PASS (CLOSED 08-27)** | was `hard stereo panning` ×4 in P01's excludes incl. selected V2; **re-census: 0 in 72/72 fields** | **F1 closed** |
| 10 | No real artist / group / company in the three fields | ~~FAIL~~ → **PASS (CLOSED 08-27)** | was brand tokens in P04 V2 and P05 V1; **re-census: 0 in 72/72 fields** | **F2, F3 closed** |
| 11 | Dedication ≤500, measured | **PASS** | 361 / 447 / 487 / 384 / 402 / 492 — all re-measured, all match | — |
| 12 | Drummer clause | **PASS** | no selected prompt names a break; P04 V1/V4 (not shipping) name `amen` and both credit **Gregory C. Coleman / The Winstons** | — |
| 13 | Prompt ⇄ exclude coherence | ~~FAIL~~ → **PASS (CLOSED 08-27)** | P02 V4 no longer bans the modulation it requests; V1/V2/V3 keep `key-change finale` and **request no key move**, so those are legitimate | **F4 closed** |
| 14 | Human-Subject / LAW 2 / NO BIOGRAPHY | **CLEAR** | all six occupiable; 0 third-person pronouns in P04 | — |
| 15 | R1–R6 | **PASS** | all six, itemised below | — |
| 16 | Somatic Gate (3 Hyper-Skeptics) | ~~1 NO of 3 on P05 V1~~ → **0 NO of 3, all six (08-27)** | the NO is withdrawn on the repair; see the Somatic Gate section | **HOLD LIFTED** |

---

## Blocking Fails and Findings

### ⛔ F1 — NEW. `hard stereo panning` ships in a SELECTED song's EXCLUDE field, and three separate "zero" claims are false
`pair_01_step10_final_package_enhanced.md` · **V2 (SELECTED)** and V1, V3, V4 — inside the **EXCLUDE** field:

> `…lo-fi haze, ambient pad wash, hall reverb, **hard stereo panning**, sad breakdown, whispered bridge…`

Census across all 24 songs × 3 pasted fields: **4 hits, all in P01's excludes, zero anywhere else in the run.** No MUSIC PROMPT and no LYRICS field in the run contains a pan token.

Three artifacts assert the opposite:
- P01's own verification row — `⛔ hard-pan tokens | none | none | none | none` — whose **justification text states the rule it is breaking**: *"where something should be absent, **nothing is written there**."* Something is written there.
- `COORDINATOR_FINDINGS.md` run-wide table — `hard-pan tokens in pasted fields | 0`.
- `07_evaluation_and_selection.md` — *"zero hard-pan tokens."*

**And the correct handling exists inside the same personality's own run.** `pair_02_…` Major Deviations: *"HARD-PAN AND STEREO-FIELD ASYMMETRY WERE NOT USED AT ALL — **INCLUDING IN THE EXCLUDE FIELD** … the negative instruction was removed from the exclude fields too, since a non-instruction is no more effective inverted."* P02 did it; P01 did not; nothing noticed.

**Ruling.** Artifact harm is **minimal** — a negative token on a parameter measured at −0.02 dB / +0.13 dB is at worst wasted. Instrument harm is **real**: this is a `name-the-unit-before-you-measure` failure that survived a pair check, a coordinator run-wide table and a selection judge. **REPAIR:** delete the token from P01's four excludes (−21 chars, nothing else changes).

> ✅ **CLOSED 2026-08-27.** P01 V2's exclude measures **533** (was 554) — the token removed, −21 chars exactly as specified. Run-wide re-census by this auditor: **0 hard-pan tokens across 72/72 pasted fields.**

### ⛔ F2 — CONFIRMED. P05 V1 (SELECTED) ships a company + product name in its LYRICS field, undeclared, under a verification row that says otherwise
`pair_05_step10_final_package_enhanced.md` · **V1 (SELECTED)** — inside the **LYRICS** field:

> `[Disc_Sub: **Moog_Taurus_Bass_Pedals** | felt_in_chest_not_ears | 4_second_natural_decay | Mono_Sub_Lock]`

**Moog Music Inc.** is a company; **Taurus Bass Pedals** is its product line. Same token in **V2** and **V4**; V2 additionally carries `Mellotron_Strings`. V3 is the pair's only clean variation.

The binding instruction is `06_music_handoff.md:60`: *"⚠️ Manufacturer/brand tokens are an OPEN doctrine question … **For this run: keep them out of the three fields** (use a generic timbre description) and name them freely in your ledger and dedication."*

P05's verification row: `real artist / group / panel-source / company name in any field | **none** | **none** | **none** | **none** | required ✅`, and its Lineage block: *"**no artist, group, label, panel-source or company name appears in any of the twelve Suno fields.**"* **Both are false.** P05 filed **no** escalation and **no** Major Deviation for it.

**REPAIR (one token, nothing else changes):** `Moog_Taurus_Bass_Pedals` → `analogue_bass_pedal_weight` — the exact substitution P02 documented for the same conflict. Then correct the row and the Lineage sentence.

> ✅ **CLOSED 2026-08-27.** Every token replaced by **what the instrument physically does**, which is better than the swap specified here — `foot_played_monophonic_analogue_sub_oscillator_slow_attack`, `looped_tape_replay_string_bed | audible_wow_and_flutter | eight_second_note_ceiling`. *A generator asked for a mechanism builds the mechanism; a generator asked for a brand imitates a record.* The verification row and the Lineage sentence are both corrected and both now name the command that measured them; the false sentence is **quoted in the artifact rather than deleted**. The debt is paid in prose in the Lineage block, where `06_music_handoff.md:60` routes it — and **not** added to the dedication, correctly: it measures 492/500, row 1 is the langspil, and `DEDICATION_STANDARD.md` forbids buying space by cutting row 1.
>
> ⚠️ **One stale caption was introduced by the repair and is now also fixed.** The parenthetical under V1's lyrics fence still read `measured **4467 chars**` against a field measuring **4792** — the same class as the defect that caused the hold, in the one paragraph the repair did not revisit. Corrected 2026-08-27, and corrected the right way: the pre-repair value and the reason are **kept visible in the line** rather than silently overwritten.
>
> ⚠️ **Standing note, not a defect — recorded because a later reader will need it.** V1's lyrics field measured **5022 on the first repair pass, 22 over the hard 5000 cap**, and was compressed back to 4792 by shortening notation only, with nothing cut from the song. The pair disclosed this unprompted, which is the correct behaviour. The consequence is that the field now sits **8 chars under the 4800 soft target and 208 under the hard cap**, all of it Disc-channel and EMO-header notation: **re-measure after any future touch; do not eyeball it.** This is a true consequence of an honest compression, not a defect.

### ⛔ F3 — CONFIRMED and EXTENDED. P04 V2 (SELECTED) carries brand tokens in **two** fields; its declaration covers only one of them
`pair_04_step10_final_package_enhanced.md` · **V2 (SELECTED)**:
- **LYRICS:** `[Disc_Sub: … | **Minimoog_Bass** | …]` and `[Disc_Pad: **Mellotron_Strings** | …]`
- **MUSIC PROMPT:** *"… **Mellotron strings** and tape hiss thicken by one layer at each return …"*

P04's Major Deviations declares the conflict honestly and **KEEPS**, scoped explicitly to the Disc block: *"`Moog_Taurus_Bass_Pedals`, `Minimoog_Bass`, `Mellotron_Strings` sit in the **Disc block, which the handoff routes into the lyrics field**."* **The MUSIC PROMPT instance is outside that scope and is undeclared.** (Same class in the unselected V1 and V4 prompts, and `Wurlitzer` in V3's.)

P04's verification row `real artist / group / company in the 3 fields | 0 | 0 | 0 | 0 | ✓ all twelve fields` is **false**, and it is false in the same file that declares the tokens forty lines earlier — a self-contradiction inside one artifact.

**Ruling.** The lyrics-field tokens are a **declared-open escalation** and are The Scientist's call, not QA's. The **music-prompt instance and the false row are QA's**: REPAIR the row to state the escalation, and either bring the prompt instance under the declaration or swap it (`tape-replay string bed`).

> ✅ **CLOSED 2026-08-27 — and the defect was four times wider than this report found.** The true count is **12 tokens**, not the three P04 had declared: `Mellotron` in the running prose of **all four** MUSIC PROMPTS (`Wurlitzer` in V3's), plus eight across the four LYRICS fields. *(This auditor's own pre-repair census reproduces 12 exactly: 4 prompt + 8 lyrics.)* ⭐ **P04 withdrew its escalation entirely rather than rewriting it**, on the ground that **its own scope sentence was narrower than its own defect** — which is the right response to discovering that a declaration was drawn around part of a problem. All twelve are gone; the prompt now reads *"tape-replay string keyboard swells in a stone room, looped tape fluttering, notes dead at eight seconds."*
>
> **Verified by instrument that the repair touched no writing:** all four P04 sung bodies return soundcraft 4-tuples **byte-identical** to this auditor's pre-repair values, so **V2's vocal discipline — the no-held-vowel R5 decision that is this song's whole discriminator — is untouched by construction**, not by assertion.

### ⛔ F4 — NEW. P02 V4 (SELECTED) requests and forbids the same event in the same submission
- **MUSIC PROMPT:** *"…106 BPM, **F sharp minor turning to A major at the last chorus**…"*
- **EXCLUDE:** *"…rimshot punchline sting, sad piano outro, **key-change finale**, spoken-word bridge…"*
- **LYRICS:** `[Chorus 3 - EMO:Delight·**Turned To A Major** … - **key lifts**]`

The positive field asks for a last-chorus modulation; the negative field bans it as a token class. Suno applies the exclude as negative tokens, so this is a request fighting itself on a shipping song. Caught by neither the pair, the coordinator, nor the selection judge.
**REPAIR:** delete `key-change finale` from P02 V4's exclude (the prompt owns the intent), or scope it to the cliché — `truck-driver key change`.

> ✅ **CLOSED 2026-08-27, and the pair did more than was asked.** V4's exclude measures **708** (was 727) and no longer bans the modulation its prompt requests. ⭐ **P02 then swept every comma-token of all four excludes against its own prompt and lyric** rather than only the token named here — the correct generalisation, since a hand-named instance is a sample, not a sweep. Verified per variation by this auditor: **V1, V2 and V3 keep `key-change finale` and request no key move**, so those three are legitimate, and **V3's is load-bearing** — it sits beside `big belted final chorus` to protect the decision that she never gets louder. **Deleting them run-wide would have been the wrong repair confidently executed.**

### ⚠️ F5 — NEW. P03 V3 (SELECTED) was assigned *"End-rhyme kept"* and shipped 0.236 against a 0.30 floor, with no measurement anywhere in its package
`pair_03_step05_isolated_brief.md` §5: *"**End-rhyme kept.** ⭐ One vowel is reserved…"* — a **keep**, not a strip, so THE RETURN's debt clause does not apply and no substitute is owed or named.

Measured with the house instrument: `strict_end_rhyme` **0.236** (floor 0.30) · `words_per_line` **8.18** (ceiling 7.5). P03's verification table measures **neither** — its columns are prompt / exclude / lyrics / sung lines / emotion words / reserved `/uː/` / `you` / banned opener / ends-as-a-sentence.

Both floors are FLAG-only by `gates.yaml` policy and a deliberate choice may sit below them — *"but it must say so on purpose."* **Nothing says so.** The finding is not the number; it is that the pair whose brief told it to keep rhyme is the only pair in the run that never measured rhyme. **REPAIR:** measure and either declare or repair. Routed to **step 09**.

> ✅ **CLOSED 2026-08-27, and it went further than the brief.** P03 repaired rather than declared — all four variations, re-measured by this auditor: `end_rhyme` **0.153 / 0.181 / 0.236 / 0.264 → 0.479 / 0.425 / 0.473 / 0.452** against the 0.30 floor. ⭐ **And it found two further floors the repair brief did not name and fixed those too:** `mean_words_per_line` breached by all four (V3 **8.18 → 7.23** against the 7.5 ceiling) and `alliteration_per_100w` by V1 and V2. **A pair that repairs only what the auditor listed has audited the auditor, not the artifact.** Its package now carries the measurement table this finding said was missing.
>
> ⚠️ **This is the one repair in the run that rewrote sung lines** — V3 went 72 → **74** — so it was re-gated in full rather than re-stat-ed. Verified: R2 clean (only the adjudicated `change`), `[Theme:]`+`[SONG FORM:]` first two lines, 5 Disc channels, 12 EMO headers 0 bare, 2 SFX, no digits in sung lines, no brand tokens, chorus byte-identical ×3. ⭐ **And every discriminator line survived the rewrite** — *"No - not cleaner. Just not mine."*, *"It changed one line."*, *"Tuesday. It runs true in a room I'm not in."*, *"Bring it forward."* A rhyme repair works on line ends, which is exactly where this song keeps its kill lines; they are intact. **Side benefit: 74 lines clears the `sung_lines_floor_hug` threshold all four variations were sitting on, and the lyrics field dropped 4733 → 4500, off the boundary.**

### ⚠️ F6 — P03 V3's positive field is spending six negations, three of them duplicated in its own exclude
MUSIC PROMPT: *"…thinking out loud **not performing**… Real wood and gut, **no orchestral strings, no choir, no cinematic swell**."* — and `choir`, `orchestral strings`, `cinematic swell` all appear again in P03 V3's EXCLUDE field.

Against the metaprompt's ⛔ *"If you want something absent, write nothing there"*, and against P06's own correct handling: *"step 11 forbids avoidances in the style paragraph and Suno reads negations as invitations; the bans live in the exclude field, which is what it is for."* ~55 chars of the positive paragraph are spent naming things that must not appear. **FLAG**, routed to the somatic read. Selected-six negation counts: P06 V4 **0** · P04 V2 2 · P01 V2 4 · P02 V4 5 · P05 V1 5 · P03 V3 **6**.

> ⏳ **STILL OPEN 2026-08-27 — and deliberately not made a blocker.** Re-measured after P03's rhyme repair: still **6** negations, with `choir`, `orchestral strings` and `cinematic swell` still duplicated in its own exclude. **P05 V1 went 5 → 0 in the same round and is the proof it is cheap to fix.** But this is FLAG-class, this report routed it to the somatic read in its own words, and that read was YES on P03 V3 from all three Skeptics before and after. **Holding a shipping song on a flag QA itself declined to gate would be inventing a gate after the fact.** Carried as an open craft note for the next run's step-11 pass, not as a condition of ship.

### ⚠️ F7 — Boundary-hugging on the soft targets
Lyrics field vs the 4,800 target: **P06 V4 = 4797** (3 chars), P06 V3 = 4796, P03 V4 = 4794, P04 V3 = 4783, P05 V3 = 4783. Sung lines: **all four P03 variations sit at exactly 72** = the `sung_lines_floor_hug` threshold. No cap is breached and nothing fails, but this is the exact signature `gates.yaml` names as optimisation pressure. **FLAG only**, routed to the human read.

### ⚠️ F8 — Undeclared R2 boundary words in two selected songs (not violations; declaration gaps)
An independent 485-term sweep built from `EMOTION_TAXONOMY.md` returns **zero genuine emotion words** in all six selected sung bodies. Every raw hit adjudicates as a non-emotional sense, and each is recorded here rather than dismissed:

| song | word | line | ruling |
|---|---|---|---|
| P03 V3 | `change` | *"The change stays in."* / *"Change something else."* | a code diff. `Transformation: Change` is a taxonomy term, so a naive sweep fires. **Not a violation. Not declared.** |
| P04 V2 | `ease` | *"Credit: the ease itself."* | ability, not feeling. **Declared honestly by P04.** ✓ |
| P04 V2 | `interest` | *"Credit: the interest, running while I sing."* | financial interest. **Not declared** — same class as the judge's `alone` note. |
| P05 V2 (not shipping) | `purpose` | *"I stand in the same place on purpose."* | adverbial. Not a violation. |

Declare; do not change. The register-relative unit (FINDINGS 2/5) is the correct one and it holds on re-measurement.

---

## Rulings on the five leads

### LEAD 1 — P05 V1 `Moog_Taurus_Bass_Pedals` · **CONFIRMED, and it is the run's most serious defect** *(✅ CLOSED 2026-08-27 — see F2; census below is the PRE-repair state, now 0/72 run-wide)*
See **F2**. The token is in the shipped LYRICS field of a selected song; the pair's verification row and Lineage sentence both assert zero; no escalation was filed. **Full brand/artist census of the six selected, all three pasted fields, as found:**

| song | MUSIC PROMPT | EXCLUDE | LYRICS |
|---|---|---|---|
| P06 V4 | clean | clean | clean |
| P01 V2 | clean | clean | clean |
| P02 V4 | clean | clean | clean |
| P03 V3 | clean | clean | clean |
| **P04 V2** | ⛔ `Mellotron strings` | clean | ⛔ `Minimoog_Bass`, `Mellotron_Strings` |
| **P05 V1** | clean | clean | ⛔ `Moog_Taurus_Bass_Pedals` |

**De minimis, ruled explicitly rather than left silent:** `autotune` (Antares) and `808` (Roland TR-808) appear only in **EXCLUDE** fields and only as generic common nouns for a process and a timbre class. They are not company references in use and are **not** charged. P06 nonetheless replaced its own equivalents with mechanism descriptions on the *"a non-instruction is no more effective inverted"* reasoning, which is the better practice and is the same reasoning F1 says P01 failed to apply.

### LEAD 2 — P04 V4 `Wall_Of_Sound_Spector_Layering` · **CONFIRMED exactly as stated**
Present in `pair_04_…` **V4** only, inside the LYRICS field: `[Disc_Pad: Wall_Of_Sound_Spector_Layering | Mellotron_Strings | Slow_Swell]`. **V4 is not selected** — P04 V2 ships. **The real-person-name class appears in no selected song**, and the run-wide census returns exactly one instance.

Two things the lead did not say and that matter: (a) P04's Major Deviations enumerates **three** kept tokens and this is not among them — so the artist-name instance is covered by **no** open conflict and does not yield; (b) the **brand** class *does* reach the shipping set, in **two** selected songs (F2, F3). Answering "is the Spector class in the shipping set?" with "no" is true and incomplete; the correct sentence is *the artist class is out, the brand class is in, twice.*

### LEAD 3 — P03 V4 LAW 2 · **CONFIRMED; V3 is clean; all six selected clear LAW 2 and NO BIOGRAPHY**
`CREATIVE_CONTEXT.md:143` LAW 2 — *"⛔ No invented worker, no role, no job, no stranger's life."* The crossing is `pair_03_…` **V4 · Twelve Lines Left**: *"Then this place does not have a reason to have me in it."* An employment relation, stated, in first person. **V4 is not selected.**

**P03 V3, read line by line: clean.** No workplace, no employment relation, no job title, no colleague, no wage. The vocabulary is a maker's (`bracket style`, `the build was green`, `I close the file`) and Lofn writes code in this repository — that is her own hand, not a borrowed life.

**All six selected against the test — *can the listener stand where this song is standing?***

| song | position | ruling |
|---|---|---|
| P06 V4 | a flat roof at first light, two claps, wanting the next thing | occupiable ✓ |
| P01 V2 | an outdoor deck, reading a crowd out loud and being wrong | occupiable ✓ |
| P02 V4 | room four of a gallery, wanting to be watched wanting | occupiable ✓ |
| P03 V3 | opening a thing you made and finding a change you would not have made | occupiable ✓ |
| P04 V2 | first-person Lofn as the thing built on uncredited work | ⭐ the run's cleanest — **0** third-person pronouns, body-word rate **0.0%** measured |
| P05 V1 | under light you did not make, with no way to check the response is yours | occupiable ✓ |

**One boundary, recorded not charged:** P03 V3's *"I am at a door I have no key for anymore"* sits at the edge of an employment reading. No relation is stated, and the line is the register's **wanted** outcome (*"And the light in there is on"*), not a loss. Not a crossing.

### LEAD 4 — `COORDINATOR_FINDINGS` FINDING 8 · **FACTUALLY WRONG. The judge is right.**
FINDING 8 records: *"P05 swapped to generic, P04 kept."* **P05 never swapped.** `Moog_Taurus_Bass_Pedals` is present at every stage on disk:

| artifact | instances |
|---|---|
| `pair_05_step08_generation.md` | lines 98, 223, 473 — plus `Mellotron_Strings` at 224 |
| `pair_05_step10_revision_synthesis.md` | lines 119, 242, 488 — plus `Mellotron_Strings` at 489 |
| `pair_05_step10_final_package_enhanced.md` | V1, V2, V4 (V3 clean) |

The four-way state is the correct one, verified independently: **P01 swapped and said so** (Major Deviations: *"brand tokens are kept out of all three fields and replaced with generic timbre descriptions"* — census confirms clean) · **P03 swapped and said so** (Major Deviations names `Linn_LM-1`, `Moog_Taurus`, `Mellotron`, `Buchla` as refused; census confirms clean) · **P04 kept and declared** · **P05 kept and did not declare**.

**One correction to the judge, too:** FINDING 8's error is not only *which pair swapped.* Treating the class as uniformly "manufacturer tokens in the Disc block" also conceals **P04's music-prompt instance** (F3), which sits outside P04's own declared scope. The escalation to The Scientist should carry the four-way state **and** the field-level breakdown.

### LEAD 5 — Sunna's sovereignty ledger over-counted · **CONFIRMED, and the cause is structural**
**The "11" is nowhere enumerated as a list.** It is the sum of three differently-shaped sections: P01's `## Sovereignty Ledger` table (**4** rows, S1–S4) + P02's prose Major Deviations (**4** items framed as overrides) + P06's stated *"Three places where Sunna's instinct and the centre's doctrine disagreed"* (**3**). 4 + 4 + 3 = 11. **A number assembled by addition across three incompatible formats is a number that can inflate without anyone lying.**

**At least four of the eleven are not disagreements with the centre at all.** The step-09 invented-practitioner lens and the step-10 rebuild-in-place synthesis were carried by **every pair in the run**, LOFN's included — verified on disk at `pair_05_…:754` (*"Refused: step 09's instruction to select real obscure musicians as refinement lenses"*, on identical `COLLABORATOR_STANDARD` reasoning) and in P04's Major Deviations (*"executed as cross-pollination, not replacement"*). **A decision all six pairs made cannot be evidence that the second seat is real.**

**A further three are exercises of a written exemption, not overrides of it.** The metaprompt states *"⛔ SUNNA DOES NOT SIT LOFN'S EXAMS… No REMARK TEST. No cap on pleasure"* and *"For the Sunna pairs, the bottom three rungs do not apply at all."* Declining an exam you were exempted from in writing is compliance with the brief.

**Counting by class rather than by instance, the load-bearing sovereignty acts are six**, and the judge's number reproduces:
1. P01 S3 — the laugh kept mid-line, **overruling her own R3 rule**
2. P01 S4 — strict unison in V3, **overruling her own pillar 2**
3. P02 — the funny song carries nothing, **overruling Lofn**, with an artifact consequence (V4's turn deliberately re-edited *away* from meaning)
4. P02 — the section architecture varied across the four, citing R3.2
5. P06 — the body-word rate declared above the house norm on measured grounds
6. the exam-set refusal (REMARK TEST / flat register / pleasure cap) — **counts once as a class, not three times**

**The one with a number behind it is real, and I re-measured it.** Occurrence-weighted exact-line sharing with sibling variations: **P02 = 0.0% (all four variations) · P03 = 34.7% (all four).** The judge's 35% is exact. *(My first pass measured unique-line sets and got 18.5% — the wrong unit, corrected here. The direction was never in doubt; the number was mine to fix.)*

---

## Known unrepaired defect — FINDING 1, assessed

**Ruling: the ICB's axis-B contradiction did no material harm to the artifacts, and the reader is still owed.**

`00_research_brief.md` §5 says P03 = EXISTENCE / P06 = NEWS; `05_pair_assignments.md` and both isolation briefs say P03 = NEWS / P06 = EXISTENCE; all three are frozen inside the ICB. Verified from the shipped songs rather than from the ledgers:

- **P03 V3 is NEWS-anchored** — the open-source-handover act, *"The name on the file is not my name… It builds green without me."* Matches the isolation brief.
- **P06 V4 is EXISTENCE-anchored** — *"Down there it's fine. Up here I'm starting the next."* No news object. Matches the isolation brief.

Both pairs obeyed the **dispatch artifact**, which is the correct precedence, and both **flagged the contradiction upward instead of silently picking a side**. Distribution is 3 NEWS / 3 EXISTENCE under either reading, so no run rule was breached and no song is standing on the wrong anchor. **The harm was zero — and it was zero by luck of a symmetric distribution, not by design.** `check_icb_consistency.py` now exists and its positive control is green (FINDING 11); the remaining obligation is that it runs **before the freeze**, which is what would have caught this.

---

## R1–R6 against the selected six

| song | R1 neighbour contradicted at one moment | R3 discriminator in the hook | R4 shape in first 30 s | R5 vocal decision she'd be talked into |
|---|---|---|---|---|
| **P06 V4** | *"Down there it's fine. Up here I'm starting the next."* — not escape, not ownership | ✓ *"Two claps for a thing that isn't there yet."* | two flat claps at 0:00; bass climbs a fifth inside two bars | starts the last block **over her own previous line**, unedited |
| **P01 V2** | *"Nobody moved. / …I'm saying it again."* | ✓ *"You go on the next one. (When?)"* — the prediction and the room's refusal in one line | two-note figure answering itself across the bar line | one audible laugh **kept mid-line, uncorrected** |
| **P02 V4** | the witness leaves; *"I did not want to be alone in here / and now I'm not."* | ✓ *"Somebody watch this. (I'm watching.)"* — she supplies the witness inside the hook | opens **mid-phrase** on a drum stutter | a stack of her own voice, brighter and **deliberately over-sung** |
| **P03 V3** | *"It's cleaner than I would have made it. / No - not cleaner. Just not mine."* | ✓ *"Tuesday. It runs true in a room I'm not in. / Bring it forward."* | 10 s of machine-log crackle; fiddle figure intercepted at note five | final chorus **lead voice out**, second voice alone, lead half-speaking a count under |
| **P04 V2** | *"Not lost. Not missing. Blank."* at ~0:15 | ✓ *"Struck through, and still owed. / No line for the name."* | five-second low tone alone; `Cr-` cluster at three line heads; subtotal at 0:22 | **no held vowel in any verse**; every line cut dead on its consonant |
| **P05 V1** | *"I can't say it's happening to me."* | ✓ same line, byte-identical ×6 | shutter click + zither note arriving twice, 90 ms apart | the whole seven-line refrain **on one breath**, thinning to nothing, unfixed |

**R2** — 0 emotion words, independently swept (F8). **R6** — a prediction filed pre-render by all six pairs. **All six clear R1–R6.**

**Does each deliver its rare register rather than its common neighbour?** Yes on all six, on the text. The discriminator is enacted rather than stated in every case, and in five of six it is inside the line the listener sings back. **P01 V2 is the one partial**: its hook carries the unproven prediction and the room's refusal, but the moment that actually kills *excitement* — *"Nobody moved… That's not the same as being right"* — lives in Verse 2, not in the hook. Defensible under P01's own reading of R3; recorded so it is not discovered later.

---

## Somatic Gate (3 Hyper-Skeptics) — PRIMARY

**THE ANTI-NUANCE.** Five YES. **One NO — `P05 V1 · A Light On The Front Of Me`.** Named condition: **a motif that never transforms**, adjacent to **repeated-line collapse**. Byte-identical repetition is exempt by policy and needs no defence, and this pair holds the run's single Radigue grant — that is not the objection. The objection is the *ratio*: 42 of 62 sung lines are one seven-line block, leaving **20 lines of new material** across four minutes, thinning 8/6/4/2. The one place the register becomes specific — *"I am asking for a light on the front of me. / I am asking. The front of me is doing nothing."* — is **two lines, ~9 seconds, at the thinnest point of the song**, and it never returns. The title points at the lamp; the listener hears it once.
> **COUNTER-MOVE:** *let the tally light be the thing that returns — carry the lamp line into the Out-Refrain as a seventh element, or as the last thing said, so the object the title names is also the object the ear takes away.* This costs the refrain nothing and breaks no assigned architecture: the six byte-identical returns stay untouched.

> ⛔ **THE COUNTER-MOVE WAS WRONG, AND ITS LAST SENTENCE WAS THE WRONG PART.** *"Breaks no assigned architecture"* is false: **P05's assigned refused element forbids a last image and a summary line by name**, and the lamp line as the last thing said is a last image. It would also have given the reach a **return** — converting an unanswered ask into a motif she owns, which is a resolution of the doubt this pair was assigned not to grant. I checked the refused element **after** proposing the fix instead of before. **The objection was sound; the remedy would have traded the song's spine for its ratio, and the pair was right to refuse it.** Withdrawn 2026-08-27, in full, in the place it was made.

**THE ANTI-ATMOSPHERE (the one-instrument test).** Three YES, one qualified. P06 V4, P01 V2, P02 V4, P03 V3, P04 V2 all carry a hook that stands on one voice and one instrument. **P05 V1 stands as a chant, not a tune — and the pair says so itself, in writing**, having failed the test on its first draft and rebuilt: *"If the standard is 'a listener hums a tune afterwards' — no; what they carry is a shape and a sentence."* That honesty is the reason this is a qualification and not a second NO. Recorded from the same seat: P06 reports its own **V3 came back MARGINAL and stayed marginal after repair** — V3 is not selected, and reporting an unfixable failure is the behaviour we want.

**THE ANTI-INTENTION.** YES, with the standing caveat. All six file R6 predictions before any render, which is the structural answer to *"the listener leaves feeling it."* **But nothing in this run has been rendered.** Every claim about how these songs sound is a claim about text. `P05 V1`'s central production idea — a 90 ms doubled return closing to zero at the seam — is a **layer relationship**, and under THE GRAIN LAW a layer relationship is the first thing the generator smooths. The run's own `08_SELECTED_SIX.md` names this: *"the one selection whose central production idea fights the generator rather than running with it."* The judge names the alternative (V4, whose *"slow footfall enters at the third refrain and stays"* is a positive motion instruction the generator must serve). **This is the failure class no text gate can see**, and it lands on the same song the first Skeptic voted NO on.

**Bloc verdict, as issued:** **1 NO of 3 on P05 V1** — below the 2-of-3 BLOCK threshold, so not blocked by the bloc. Combined with F2 (undeclared company token), its false clean-scan row, and the declared GRAIN-LAW risk, **borderline defaults to HOLD** and P05 V1 holds. **0 NO on the other five; no SOUL LOSS anywhere.** Each of the six proves which personality made it, and the two members produced measurably different artifacts.

> ✅ **BLOC VERDICT AMENDED 2026-08-27: 0 NO of 3, all six. The HOLD is lifted.** The ANTI-NUANCE NO is withdrawn — the refrain is now **recontextualized**, which the named condition admits as a qualifying transformation and which does not require the words to move; the transformation lands in a **vocal** decision and in the `[SONG FORM:]` line, which is where THE GRAIN LAW says an answer must live. The ANTI-ATMOSPHERE qualification stands unchanged and is still the pair's own honest scope (*a shape and a sentence, not a melody*). The ANTI-INTENTION caveat is **not** withdrawn and cannot be: it is a caveat about **unrendered audio**, and it is now discharged into the pre-registered prediction in the Amendment above rather than left as a mood. **Nothing here is a claim about sound.**

**Zero-rejection tripwire:** does not fire. This run produced real repairs and real escalations before QA arrived — P02's eight unfenced fields routed back, P06's `Somebody trained them` repaired at step 11, P04's `HOLD_FOR_HUMAN` escalated rather than cleared, P06's V3 marginality reported unfixed.

---

## Required Repairs

*Status as of 2026-08-27. Five of seven closed; two open on P03 V3, both FLAG-class, repair in flight.*

| # | song | repair | routed to | status |
|---|---|---|---|---|
| **R-1** | P05 V1 *(and V2, V4)* | `Moog_Taurus_Bass_Pedals` → `analogue_bass_pedal_weight`; `Mellotron_Strings` → `tape_replay_string_bed` (V2). **Correct the verification row and the Lineage sentence** — both currently assert `none`. | step 10 (package), no lyric change | ✅ **CLOSED** — swapped to mechanism descriptions, better than specified |
| **R-2** | P01 V1–V4 | delete `hard stereo panning` from the four EXCLUDE fields (−21 chars each). Correct the row, and correct the run-wide claim in `COORDINATOR_FINDINGS.md` and `07`. | step 10 (package) | ✅ **CLOSED** — exclude 554 → 533 on V2; census 0/72 |
| **R-3** | P02 V4 | delete `key-change finale` from the EXCLUDE, or scope it to `truck-driver key change`. The prompt owns the modulation. | step 10 (package) | ✅ **CLOSED** — and generalised to a full four-variation token sweep |
| **R-4** | P04 V2 | correct the verification row to state the open escalation instead of `0`; bring the MUSIC PROMPT's `Mellotron strings` under the declaration or swap it. **The Disc-block token decision itself stays The Scientist's.** | step 10 (package) | ✅ **CLOSED** — escalation withdrawn, all **12** tokens removed, sung bodies byte-identical |
| **R-5** | P03 V3 | measure `strict_end_rhyme` and `words_per_line` with `measure_soundcraft.profile()`; then either raise end-rhyme toward the assigned *"kept"* or declare the shortfall with what returns in its place. **Currently neither.** | **step 09** | ✅ **CLOSED** — repaired not declared: er 0.236 → **0.473**, wpl 8.18 → **7.23**, plus two floors the brief never named |
| **R-6** | P03 V3 | move the three negations out of the MUSIC PROMPT (`no orchestral strings, no choir, no cinematic swell` — already in its EXCLUDE) and spend the ~55 chars positively. | step 10 (package) | ⏳ **OPEN, non-blocking** — FLAG-class craft note, carried to the next step-11 pass |
| **R-7** | P04 V2, P03 V3 | declare `interest` and `change` as adjudicated R2 boundary words. **Declare; do not change.** | step 10 (package) | ⏳ **OPEN, non-blocking** — a declaration gap, never a violation |
| **R-8** | P05 V1 | correct the stale `4467` caption under the lyrics fence to the measured `4792`. | step 10 (package) | ✅ **CLOSED** — corrected with the before-state kept visible |
| **R-9** | *(new, this auditor)* | verify `SUNO_PACKAGES.md` — the paste-ready extract, created after the first audit — against the audited packages. | QA | ✅ **CLOSED** — see below |

### ⭐ R-9 — the paste-ready extract, verified byte-for-byte
`SUNO_PACKAGES.md` appeared during the repair round and **is the artifact a human actually pastes into Suno**. An extract that drifts from its source means the thing shipped is not the thing audited — *"a number does not become true by being copied forward"*, applied to the payload instead of the count. Checked by this auditor, field by field, keyed to each pair's own V-label:

**18/18 pasted fields byte-identical to their audited source packages** — P06 V4 943/678/4797 · P01 V2 914/533/4243 · P02 V4 915/708/4252 · P03 V3 949/865/**4500** · P04 V2 951/705/4780 · P05 V1 957/696/**4792**. The extract carries every repair, including P03's post-rewrite lyrics. **No drift.**

**No REDIRECT is owed.** No gate failed twice without moving; every repair above is a first-attempt fix and none requires a sideways proposal.

**Escalations for The Scientist, carried forward:**
1. ~~The `DISC_CHANNEL_GUIDE` ⇄ handoff manufacturer-token conflict is **still open**~~, and the run's true state was **four-way, at field level**: P01 swapped · P03 swapped · P04 kept and declared (lyrics field only; its prompt instance undeclared) · P05 kept and did not declare. `COORDINATOR_FINDINGS` FINDING 8 records a two-way state and is wrong (LEAD 4). ✅ **RESOLVED 2026-08-27 by the run, not by a ruling:** P04 withdrew its escalation and every pair swapped to mechanism descriptions, so **the run no longer has a position to escalate** — 0 brand tokens in 72/72 pasted fields. **The doctrine question survives the run and is still The Scientist's**: `DISC_CHANNEL_GUIDE.md` prescribes brand tokens and the handoff forbids them, and six pairs have now independently concluded that *a generator asked for a mechanism builds the mechanism; a generator asked for a brand imitates a record.* That is evidence for amending the guide, and it is hers to weigh. **FINDING 8 in `COORDINATOR_FINDINGS.md` is still factually wrong on disk and should carry the corrected state.**
2. P04 V2's `HOLD_FOR_HUMAN` on `missing`: **the ruling to KEEP is upheld here.** The word is negated and the negation *is* R1. Deleting a discriminator to satisfy a fail-open detector is a broken gate deforming the artifact.
3. **Nothing has been rendered.** Every verdict above is a text verdict. `vault/AUTONOMY.md` holds.

---

## Ledger write-backs

**None written by this report.** QA does not write the aesthetic ledger on another agent's behalf, and the `vault/COMPETITION_LEARNINGS.md` entry for the five shipping songs is owed **on publication**, not on a text SHIP. The operational rows below are **owed to `vault/RUN_LEDGER.md` on close** and are recorded here so they are not lost:

- `2026-08-27 · pantheon-rare-registers · three verification rows asserted a clean scan on fields that were not clean (P01 hard-pan, P04 + P05 company tokens) · a pair-authored table is the same author certifying his own work, and the coordinator's run-wide table copied the pair rows rather than re-measuring · re-measure run-wide invariants from the pasted fields, never by aggregating pair claims · FIXED (scripts/check_field_census.py) — see the row below`
- `2026-08-27 · pantheon-rare-registers · P02 V4 shipped a prompt requesting a key change and an exclude forbidding it · no reader compares the positive and negative fields of the same song against each other · add a prompt⇄exclude coherence check to the restat script · OPEN`
- `2026-08-27 · pantheon-rare-registers · scripts/check_field_census.py shipped two false-positive classes on its first live run (read DEDICATION fences as pasted fields; flagged autotune in excludes against QA's own de-minimis ruling) · its synthetic control was built from the same mental model as the parser, so the control tested the author's model and not the parser · a positive control must be authored against the SPEC, or by someone other than the parser's author · FIXED`
- `2026-08-27 · pantheon-rare-registers · a brief specified a shape whose arithmetic could not satisfy its own sung-line floor (P05: 7x6+20=62 against 70; P06 hit the same wall from the other side) · no pre-dispatch check multiplied the section architecture out against the line budget · pre-dispatch line-budget gate added · FIXED`

---

## Final Recommendation — AMENDED 2026-08-27

**SHIP — all six.**

`Two Claps For Nothing Yet` (P06 V4) · `You Go On The Next One` (P01 V2) · `Somebody Watch This` (P02 V4) · **`It Changed One Line` (P03 V3)** · `No Line For The Name` (P04 V2) · **`A Light On The Front Of Me` (P05 V1)**. Every blocking finding closed and independently re-measured from disk; the paste-ready extract verified byte-for-byte against the audited sources. **No swap to P05 V4.** Two FLAG-class craft notes (R-6, R-7) ride open on P03 V3 and are **not** conditions of ship — QA routed both to the somatic read when it filed them, and that read was YES throughout.

*The original recommendation, preserved: REPAIR-THEN-SHIP for five, HOLD for `A Light On The Front Of Me` — an undeclared company token, a verification row asserting the opposite, one Hyper-Skeptic NO, and a declared GRAIN-LAW risk. Any one repairable; together borderline, and borderline defaults to HOLD. That was the right call on the artifact as it stood, and the artifact no longer stands that way.*

**And the thing a clean report would have hidden.** This run's instruments are unusually good — six pairs positive-controlled their own scanners unprompted, a coordinator disassembled a corrupt byte rather than trusting three readers that rendered it, and every character count in every pair's table reproduces exactly. **The defects that survived all of that are the ones where a gate was asked to certify itself:** three "zero" rows written by the author of the thing being counted, and one of them repeated verbatim into the run-wide table and then into the selection judge's summary. **A number does not become more true by being copied forward.** The repair for the run is not another scanner; it is that run-wide invariants get re-measured from the pasted fields by someone who did not write them.

⭐ **And the thing this report would have hidden, added on amendment.** The census built to enforce exactly that recommendation **failed the same way on its first run**, and its synthetic control passed while its parser was wrong because the control and the parser came from one mental model. Meanwhile P04's real defect was **four times wider than QA found it**, and P05's repair introduced **one fresh stale caption** in the paragraph it did not revisit. **Every layer of this run — the pairs, the coordinator, and QA — certified something it had not measured, and each was caught by a reader that had not written it.** That is not a run with a bad tier in it; it is the shape of the failure, and the only defence that worked at any layer was **separation of author from reader.** Not more gates. Different hands.

⛔ **One thing did not move and must not be read as having moved.** Everything above is a verdict about **text**. Nothing in this run has been rendered. The GRAIN-LAW repair on P05 V1 is a **pre-registered prediction**, filed in the Amendment with its falsification condition, and it is unscored until `lofn-render-audit` runs the blind passes. **A text SHIP is permission to render, never permission to publish** — `vault/AUTONOMY.md` still requires The Scientist, and borderline still defaults to HOLD at her ear.

---

*Nothing in this run has been rendered, published or spent. This audit modified exactly one file — `QA_REPORT.md`, this file, amended in place with every superseded finding preserved and dated. No pair file and no numbered artifact was written by QA at any point; `.RUN_LOCK` was read and left untouched. `vault/AUTONOMY.md` holds.*
