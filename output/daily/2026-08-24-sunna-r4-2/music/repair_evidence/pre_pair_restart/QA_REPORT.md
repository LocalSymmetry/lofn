# Lofn Music QA Report — Sunna R4.2

## Verdict: REPAIR

The six selected songs are musically viable, distinct, sound-return compliant, human-subject clear, and textually Suno-ready after small field cleanup. The run as a whole is **not certifiable as a current-ICB Step 00–11 chain**: coordinator Steps 00–05 attest a different, older immutable context than the canonical ICB carried by the pair dispatches, and current control artifacts retain obsolete predecessor fields. This is a pipeline-integrity blocker, not a rejection of the six songs.

This is a text-package verdict only. No audio render exists in the scoped run, so no claim is made about rendered performance, mix, or intent survival. Nothing was rendered, published, uploaded, or spent during this audit.

## Blocking findings

### B1 — Coordinator Steps 00–05 fail current-ICB continuity

- Current canonical `CREATIVE_CONTEXT.md`: **154,765 bytes**, SHA-256 `6ed856fb2fe4974030f8b040caaffaed718aff22b7d87554a4cfab91f886c0a0`.
- All six files in `music/dispatch/pair_01_dispatch.md` through `pair_06_dispatch.md` contain those exact 154,765 bytes as a byte-for-byte prefix at offset 0. Each contains all **18** `(after …)` panel-voice markers. This part passes.
- `music/step00_aesthetics_and_genres.md` instead attests a **142,674-byte** immutable block with SHA-256 `1ce0b17997c26edf5291ce01bb2ffdc9544c7ae147b9bffdcf8b197114c206c9`.
- `music/step01_essence_and_facets.md`, `step02_concepts.md`, `step03_artist_and_critique.md`, `step04_medium.md`, and `step05_refine_medium.md` all cite the same obsolete `1ce0b179…` hash.

Under the ICB hard gate, downstream quality cannot retroactively prove that Steps 00–05 used the current immutable context. This is **THREAD LOSS / continuity failure** and requires restart from the last valid artifact.

Evidence: `CREATIVE_CONTEXT.md`, `PHASE1_PREFLIGHT.md`, `music/step00_aesthetics_and_genres.md` through `music/step05_refine_medium.md`, and `music/dispatch/pair_01_dispatch.md` through `pair_06_dispatch.md`.

### B2 — Stale contracts remain in current control artifacts

- `music/concept_medium_pairs.json` identifies P02 as `MAKE IT NINE`, but P02 `hook_craft_scan.stress_pattern` still says `KISS the MAY-be OFF`, and `attention_ladder.first_hook_memory` still says `Kiss the Maybe Off.`
- The same file identifies P04 as `ONE LIME, RED CHAIR`, but its hook scan still says `BRING LIME at EIGHT`, and its first-hook memory still says `Bring Lime at Eight.`
- `06_audio_handoff.md` names the controller as `...:sunna-r4-1` inside the R4.2 run.
- `RUN_STATE.md` says P01–P03 are running and P04–P06, selection, Step 11, standalone files, and QA are pending, although all those artifacts exist.

These are not harmless archival notes: the obsolete P02/P04 values sit inside the pair objects copied into dispatches. `RUN_STATE.md` also disagrees materially with disk state.

Evidence: `music/concept_medium_pairs.json`, `06_audio_handoff.md`, `RUN_STATE.md`.

### B3 — Negative instructions remain in positive Style fields

The two-field contract assigns desired sound to Style and failure modes to Exclude. The following positive fields retain prohibitions that are already, or should be, represented in Exclude:

- P01: `no acoustic guitar`
- P02: `no rasp`
- P03: `no upspeak on choices`
- P05: `Acoustic guitar is absent`
- P06: `no acoustic guitar`

P04 is clean. These are bounded standalone-package repairs; deleting or positively restating them leaves every Style field safely above the 850-character floor.

Evidence: `music/suno_render/01_friday_looks_good_on_me.md`, `02_make_it_nine.md`, `03_turn_when_i_lower.md`, `05_pretty_busy_tomorrow.md`, `06_hot_girl_future_tense.md`.

## Gates that pass

### Phase 0/1 and live disagreement

- `00_research_brief.md` and `02_golden_seed.md` explicitly freeze the exact **3 ACCESSIBLE + 3 AMBITIOUS** split and **2 NEWS + 4 EXISTENCE** axis.
- `03_orchestrator_panel_debate.md` contains three live labeled configurations—**BASELINE**, **BRIDGE**, and **REFLECT**—with actual disagreement and changed decisions, not merely three summaries. It supplies 18 seats, three Hyper-Skeptics, and the plural 15-item Special Flairs palette.
- Golden outputs remain judge-side and quarantined from generation.
- The pair dispatches pass current ICB byte-zero and 18-voice presence, as measured above.

### Exact selected-set allocation and action/form engines

| Pair | Arm | Axis | Selected action/form engine | Result |
|---|---|---|---|---|
| P01 | ACCESSIBLE | EXISTENCE | reserve / linear reservation whose title changes from offered to confirmed | PASS |
| P02 | ACCESSIBLE | NEWS | challenge / scorecard rounds, blade-cut bar, counter-wager restoration | PASS |
| P03 | ACCESSIBLE | EXISTENCE | permit / palm-up stop and palm-down restart antiphony | PASS |
| P04 | AMBITIOUS | EXISTENCE | season / cumulative call-response changed by salt ownership and handoff | PASS |
| P05 | AMBITIOUS | NEWS | select / torn photo-strip routing and choice-caused rondeau return | PASS |
| P06 | AMBITIOUS | EXISTENCE | declare / pen-caused question-to-declaration contour | PASS |

The set is exactly 3+3 and 2+4. All six forms are causally different; this is not one appointment scaffold with nouns exchanged.

Evidence: `00_research_brief.md`, `05_orchestrator_pair_assignments.md`, `music/TOP_6_SELECTION.md`, and the six final lyric fields.

### Canonical step completeness and portfolio uniqueness

- Step 06: six separate files, each with at least eight substantive pair-local facets, why-it-matters language, and failure modes. Independent `scripts/validate_step06_distinctiveness.py` run: **PASS**.
- Step 07: six separate files, exactly six guides per pair, including the required body-position and hurt/stakes questions.
- Steps 08, 09, and 10: each has exactly four complete packages per pair—24/24 Style, Exclude, Lyrics, and Title fields at every step.
- Independent `scripts/validate_step09_distinctiveness.py` run: **PASS**.
- Independent `scripts/validate_portfolio_distinctiveness.py ... --step 10 --expected-total 24` run: **PASS**, 24 packages extracted.
- The same validator on the six standalone Step-11 deliveries: **PASS**, six packages extracted.

Evidence: `music/pair_01_step06_facets.md` through `pair_06_step10_revision_synthesis.md`, plus `music/suno_render/`.

### Manual Step 11 materiality

Every selected candidate received a real manual edit rather than a mechanical promotion. Independent comparison of Step-10 V01 to the standalone final found unchanged core Style/Exclude intent but material lyric changes:

- P01: 15 added / 15 removed lines; purse movement and confirmed-seat causality sharpened.
- P02: 21 added / 17 removed lines; written counter-time, `Beat mine`, circled next round, and restoration clarified.
- P03: 13 added / 12 removed lines; Sunna’s hand owns restart and the return language is more performable.
- P04: 32 added / 25 removed lines; seasoning and chair-opening jobs separated, adult salt ownership preserved, then a later blind-stage alliteration repair reduced metric-writing artifacts.
- P05: 9 added / 3 removed lines; physical palm transfer and exact rondeau return restored.
- P06: 8 added / 8 removed lines; adult answer and positive sound-command language improved while preserving the punctuation hinge.

Evidence: `music/pair_01_step11_enhancement.md` through `pair_06_step11_enhancement.md`, compared with the matching Step-10 files and final standalone packages.

### Deterministic standalone and sound-return measurements

The current Sunna counterweight uses the run-local **36–56 sung-line band** recorded in `gates_sunna_r41.yaml`. This intentionally supersedes the generic 70–120 floor for this personality run; the disagreement is recorded rather than hidden. All other canonical sound-return floors were remeasured with `scripts/measure_soundcraft.py -> profile_file()`.

| Final | Style chars | Exclude chars | Lyrics chars | Sung lines | Rhyme return | Exact-line return | Mean words/line | Alliteration/100w | Dedication chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 Friday Looks Good on Me | 929 | 513 | 2,477 | 40 | .400 | .250 | 5.28 | 12.32 | 368 |
| P02 Make It Nine | 959 | 708 | 2,235 | 44 | .636 | .227 | 4.61 | 12.81 | 388 |
| P03 Turn When I Lower | 954 | 624 | 1,864 | 38 | .500 | .342 | 3.53 | 13.43 | 399 |
| P04 One Lime, Red Chair | 931 | 638 | 2,284 | 47 | .617 | .362 | 3.96 | 11.83 | 413 |
| P05 Pretty Busy Tomorrow | 947 | 608 | 2,908 | 51 | .333 | .216 | 4.78 | 11.89 | 437 |
| P06 Hot Girl Future Tense | 953 | 671 | 2,201 | 42 | .310 | .262 | 4.86 | 11.27 | 391 |

All six pass: Style 850–1000; Exclude 400–900; Lyrics under 5,000 and under the 4,800 target; sung lines 36–56; rhyme return at least .30; exact-line return at least .20; mean words per line at most 7.5; alliteration at least 11/100 words; dedication at most 500 characters. All have Theme first, SONG FORM second, section-level EMO/performance headers, an SFX cue, no sung digits, terminal punctuation in Style, and no credited artist name in any generator-facing field.

### Vocalist fingerprint and Suno readiness

Every Style field specifies an adult late-20s/early-30s female, a light Scandinavian English accent, at least two positive tone/timbre traits, and at least two phrasing traits. Each package also names a pre-vocal identity figure, a matching post-vocal receipt, placement/depth, one dominant world, and one seam. After B3’s field cleanup, each file is standalone and paste-ready with exactly the four required deliverables.

### Eligibility, Gate 9, and Somatic Gate

| Final | Seven-property eligibility | Gate 9 ladder in lyric/hook | Hyper-Skeptic Somatic vote | Result |
|---|---|---|---|---|
| P01 | 7/7 | door/purse/forks → pink pie machine and exact coordinate → Friday personified → finger/hand/seat | YES / YES / YES | PASS |
| P02 | 7/7 | ball/card → blade and counter-time → two small worlds → hands collide/retrieve | YES / YES / YES | PASS |
| P03 | 7/7 | zipper/palm → yellow teeth → bright ladder/room → thumb, hand, seam | YES / YES / YES | PASS |
| P04 | 7/7 | lime/chair/salt → dock and low tide → green-white tide line → sting, thumb, palm | YES / YES / YES | PASS |
| P05 | 7/7 | booth strip → chosen third frame → clockwork court → torn warm paper in hands | YES / YES / NO | PASS 2–1 |
| P06 | 7/7 | flyer/page/pen → Thursday coordinate → strange grammatical title/declaration → pen, ink, two names | YES / YES / NO | PASS 2–1 |

P05 dissent: the tiny-court bridge briefly risks becoming formal wit admired from outside the flirt. Counter-move if later listening confirms it: shorten the court by two lines and let the selected frame reorder the next sung couplet more immediately.

P06 dissent: the grammar conceit could read as cleverness before it reads as appetite. Counter-move if rendered delivery flattens the hinge: delete one pre-hinge title question and give the pen-touch answer a longer melodic ownership phrase. Neither dissent blocks the text package.

All six have a one-sentence retellable surface, an unexpected but causal action, emotional consequence, a memorable return, and enough formal inevitability for 7/7 eligibility. The strange/mythic enlargement occurs in the sung lyric or hook and returns to handled scale; none relies on a production sidecar to answer Gate 9.

### Blind plate and calibration

- Original six-set blind plate: candidate > designated decoy **6/6**; golden > designated decoy **6/6**. Five candidates ranked first; original P04 ranked second to its golden.
- P04 was repaired after Stage 1. Three independent repair reblinds all ranked the repaired candidate first.
- In all three new P04 reblinds, however, the designated decoy ranked above the golden. Therefore the P04 mini-set’s calibration is **unstable**. It is disclosed here and is not cherry-picked as a clean calibration pass. The only defensible claim from that mini-set is the repeated candidate-first result; the original six-set plate supplies the healthy control result.

Evidence: `music/QA_BLIND_STAGE1.md`, `music/QA_BLIND_DECODE.md`, and the P04 repair/reblind artifacts under `music/qa_blind/`.

### Dedication, lineage, and honesty

All six dedication bodies independently measure at 368–437 characters, below the 500-character ceiling. `music/DEDICATIONS.md` and `music/CREDITS.md` name the musical lineage and salient players; the generator fields contain no artist names. Each dedication states AI authorship and says no sample is present. No prompt names a borrowed break, so no uncredited named-break debt is triggered. Source links and extended credits live in `music/CREDITS.md`.

### Human subjects and autonomy boundary

The songs are Sunna-first wants addressed to invented, non-identifiable adults. No lyric uses a real harmed person, identifiable victim tuple, child-as-victim, or real biography. The deterministic human-subject prefilter overflags performance-header capitals and P05’s `clockwork court` as name/place-like tokens; manual Gate 16 review finds no real person in the title or lyric. Real musicians appear only in dedication/credits lineage, not as subjects or generator instructions.

No audio, image, or video render is present in this run; the artifact tree remains text-only. There is no evidence of publish, upload, or spend activity. Human ear approval remains required before any outward release.

## Per-final verdicts

| Final | Verdict | Concise reason |
|---|---|---|
| P01 — Friday Looks Good on Me | **REPAIR** | Strongest immediately shareable title return; purse/fork action makes confirmation audible. Delete `no acoustic guitar` from Style. |
| P02 — Make It Nine | **REPAIR** | Distinct competitive scorecard form, best rhyme return, equal adult counter-wager. Replace `no rasp` with positive vocal wording. |
| P03 — Turn When I Lower | **REPAIR** | Cleanest form/action identity; stop and restart belong to Sunna’s palm and the zipper riff returns. Positively restate `no upspeak on choices`. |
| P04 — One Lime, Red Chair | **SHIP as a standalone text package** | Repaired candidate wins all three reblinds; cumulative seasoning/handoff form, sound floors, and Gate 9 pass. Mini-set calibration remains unstable and cannot support broader claims. |
| P05 — Pretty Busy Tomorrow | **REPAIR** | The chosen frame causes the rondeau and the physical tear has consequence. Remove `Acoustic guitar is absent` from Style; Somatic dissent is advisory. |
| P06 — Hot Girl Future Tense | **REPAIR** | Pen and second name physically convert question into declaration; hook is adoptable. Delete `no acoustic guitar` from Style; Somatic dissent is advisory. |

These per-final verdicts address the current standalone files. The overall run remains REPAIR even for P04 because B1/B2 prevent full-pipeline certification.

## Exact repair route

1. Treat the frozen current `CREATIVE_CONTEXT.md` (`6ed856fb…`, 154,765 bytes) as the last valid continuity root.
2. Restart coordinator music work at Step 00 and regenerate Steps 00–05 under that exact ICB. Every replacement artifact must cite the current hash; do not re-attest the old outputs after the fact.
3. Rebuild `music/concept_medium_pairs.json`. Remove the obsolete P02 `KISS THE MAYBE OFF` fields and P04 `BRING LIME AT EIGHT` fields; make every nested hook, attention, action, form, and danger field agree with the selected current seed.
4. Correct the controller identity in `06_audio_handoff.md`, rebuild all six dispatches with the current ICB at byte zero and all 18 voice markers, and rebuild `RUN_STATE.md` from disk.
5. Because the corrected pair objects are generating inputs, restart the affected pair chain at Step 06 and carry it through Step 10, selection, and manual Step 11. Re-run the Step 06, Step 09, 24-package, soundcraft, eligibility, Gate 9, Somatic, dedication, human-subject, and standalone validators. Do not assume today’s packages survive unchanged.
6. If any final package bytes change, rebuild the blind sets and repeat blind-first evaluation before decoding. Preserve the original healthy six-set calibration and disclose any unstable mini-set; never select only the favorable control ordering.
7. In the five affected standalone Styles, remove or positively restate the negative clauses listed in B3 and keep prohibitions in Exclude. Remeasure field lengths afterward.
8. Run a fresh clean-context QA against the repaired chain. No render or publication is authorized by a repaired text verdict.

## Final disposition

**REPAIR, not FAIL.** The creative result is strong and the six-package set passes the substantive musical gates; the defect is a recoverable but non-waivable continuity break plus bounded two-field cleanup. **Do not label the run SHIP until Steps 00–11 can be reopened as one current-ICB chain and the repaired artifacts receive a fresh audit.**
