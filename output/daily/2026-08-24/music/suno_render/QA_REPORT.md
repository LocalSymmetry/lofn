# Lofn QA Report — 2026-08-24 Suno Render Packages

Audit basis: the frozen current bytes of the six numbered files in this directory, the frozen `05_orchestrator_pair_assignments.md`, and `music/TOP_6_FINAL_SELECTION.md`. No song file was edited.

## Verdicts

- Pipeline Integrity: **PASS**. Exactly six canonical numbered render packages are present. The six selected pair/variant/title rows match the frozen selection. The non-skip spine is present for P01–P06 at Steps 07, 09, and 10, and `RUN_STATE.md` records each as `done`.
- Suno Package: **PASS**. `validate_suno_packages.py` returns PASS for all six; independent current-byte measurements below apply the stricter canonical 70–120 line, 400–900 exclude, five-channel, and integrated-header checks.
- Human Subject: **CLEAR** by direct current-byte review. Every depicted person and place is explicitly invented; no lyric or caption identifies a real harmed/private person or minor. The deterministic prefilter produced false HOLDs; that instrument defect is segregated below.
- Somatic Gate: **PASS, 3–0 for every song**. The P03 70-line boundary flag was escalated to this read and does not evidence collapse.
- Overall: **SHIP — TEXT PACKAGE ONLY**.

There has been **no audio render, no listening pass, no render audit, no upload, no publication, no distribution, and no spend** in this audit. SHIP applies only to the frozen text packages. The Scientist's ear remains the render/publication gate.

## Frozen Canonical Inventory and SHA-256

| # | Canonical file | Bytes | SHA-256 |
|---:|---|---:|---|
| 1 | `01_still_in_orbit.md` | 21,511 | `6e7f4e1be4a4a665664edcb29433cc2af2d0774cab0b204f45fb2917d9009a24` |
| 2 | `02_no_significant_events.md` | 19,072 | `1ddab274feb84ba1c2a04cf2c1dc1728068763a100cc33fc65143ac20b763ed7` |
| 3 | `03_leave_the_line_open.md` | 19,212 | `89882b52f17e7b0a11cdeae4374a24af316de42ed6110f763ebd7be741ec9b8c` |
| 4 | `04_the_last_verb_fails.md` | 21,544 | `ba851c6d7f199c44be3ebc860cf06ae1839c706311fc927e081f873591fccc03` |
| 5 | `05_short_of_the_groove.md` | 18,141 | `435807ae2aa94bde0fdfbb36e569514f62dabc5aea1619b7b340dc8229318563` |
| 6 | `06_backwash_before_wave.md` | 22,373 | `d10c1d2c90cc9ed9b031089570432b9ac87a95bcafaff139569fdc4bd330a86c` |

Inventory check: **6 expected / 6 found / 0 missing / 0 extra numbered files**.

## Exact Generator-Field Evidence

Character counts exclude Markdown fence-marker lines and include all current text inside the field. Lyrics counts include Theme, SONG FORM, all five Disc channels, every EMO header, SFX, and sung lines.

| File | Canonical headings Style/Exclude/Lyrics/Title | Style chars | Exclude chars | Lyrics chars | Sung lines | Disc channels | EMO headers integrated/total | `[11]` or pipeline tokens in lyrics | Dedication chars | Links in Lineage block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 | 1 / 1 / 1 / 1 | 976 | 832 | 4,203 | 89 | 5 | 9 / 9 | 0 | 385 | 0, N/A scene |
| P02 | 1 / 1 / 1 / 1 | 956 | 785 | 4,723 | 102 | 5 | 10 / 10 | 0 | 459 | 2 |
| P03 | 1 / 1 / 1 / 1 | 979 | 623 | 3,801 | 70 | 5 | 9 / 9 | 0 | 484 | 2 |
| P04 | 1 / 1 / 1 / 1 | 983 | 798 | 4,113 | 93 | 5 | 11 / 11 | 0 | 415 | 2 |
| P05 | 1 / 1 / 1 / 1 | 943 | 754 | 3,839 | 94 | 5 | 9 / 9 | 0 | 481 | 0, historical/non-scene source |
| P06 | 1 / 1 / 1 / 1 | 955 | 824 | 4,348 | 100 | 5 | 9 / 9 | 0 | 422 | 2 |

All six meet Style 850–1,000, Exclude 400–900, Lyrics <5,000 and <=4,800 target, and 70–120 sung lines. Each opens lyrics with `[Theme:]` immediately followed by `[SONG FORM:]`; each has exactly the five required Disc channels: Rhythm, Vocal, Sub, Pad, Texture. Every EMO-bearing section header uses an integrated em-dash form with a specific emotional state, persona/role, and delivery cue. No bare AWE/INDIGNATION EMO label occurs.

P03 sits exactly at the 70-line hard floor and therefore triggers the canonical boundary-hugging **FLAG** (<=72), not a failure. Its exact three-return chorus, debit/credit/care tally, care overrun, alarm seam, stair turn, and dry trace show intentional economy rather than repeated-line exhaustion. Somatic verdict remains PASS.

## Deterministic Instrument Results

### Suno validator

`python skills/music/scripts/validate_suno_packages.py <six files>`: **PASS 6 / FAIL 0**, exit 0.

### Soundcraft

| File | Strict end rhyme | Exact line return | Mean words/line | Alliteration/100w | Sung lines | Verdict against 0.30 / 0.20 / <=7.5 / 11.0 floors |
|---|---:|---:|---:|---:|---:|---|
| P01 | 0.461 | 0.202 | 4.82 | 11.42 | 89 | PASS |
| P02 | 0.627 | 0.265 | 4.94 | 11.31 | 102 | PASS |
| P03 | 0.429 | 0.314 | 5.20 | 23.90 | 70 | PASS |
| P04 | 0.409 | 0.280 | 3.84 | 16.25 | 93 | PASS |
| P05 | 0.362 | 0.330 | 3.77 | 14.41 | 94 | PASS |
| P06 | 0.350 | 0.200 | 4.20 | 15.48 | 100 | PASS; line return is exactly at floor |

### Step-11 portfolio distinctiveness

`validate_portfolio_distinctiveness.py ... --step 11 --expected-total 6`: **PASS**, exit 0; six packages extracted from six files.

- Maximum prompt similarity: **0.420375**, P01 vs P05, below 0.58.
- Maximum lyric similarity: **0.316214**, P02 vs P04, below 0.42.
- Maximum lyric 5-gram Jaccard: **0.007064**, P05 vs P06, below 0.18.

These are the validator's normalized comparison measures, not raw field character counts; raw counts are reported separately above.

## Step-11 Quarantine and Supporting-Block Audit

Each file contains exactly one `## Golden Move Calibration` block. Across the six files there are:

- zero Golden Song titles (`Triple Arch Over Me`, `Five wrong colors`, `I Will Stop the Almost`, `The Blue Screen Breathes`);
- zero Golden Song URLs/IDs and zero `suno.com/@localsymmetry` profile URL;
- zero `Golden Song` payload markers;
- zero house-lexicon payload hits from `vault/gates.yaml`.

The generated outputs retain distilled moves only. Every file also contains all required supporting blocks: Vocal Fingerprint, Production Dramaturgy, Arrangement Dramaturgy, Binding Locks, Style-Axis Locks, Golden Move Calibration, Major Deviations, Lineage & Credit, Constraint Audit, Panel Ledger / QA, and Attribution / Provenance. P02 contains the specifically required `## Body Noise Map`.

## Frozen-Assignment Fidelity

| Pair | Frozen route / mode | Current package metadata | Return and one seam | Verdict |
|---|---|---|---|---|
| P01 | Accessible / AWE | LOFN-PRIME AWE; 2:48 | Exact `Still in orbit`; cloth burr + one PA withdrawal + missing bass floor at 2:04 | PASS |
| P02 | Accessible / AWE | LOFN-PRIME accessible AWE; 3:06 | Exact `No significant events`; one shelf impact deletes the learned N and kick/knee repay it | PASS |
| P03 | Accessible / AWE | LOFN-PRIME AWE; 3:24 | Exact `Leave the line open`; one sustained alarm harmonic joins ledger, fire-plan line, wrist, and breath | PASS |
| P04 | Ambitious / INDIGNATION | LOFN-PRIME INDIGNATION; 3:37 | `Put it in my hand` loses one cumulative terminal word; fixed vacancy pays the debt; one click at 3:08 | PASS |
| P05 | Ambitious / INDIGNATION | LOFN-PRIME INDIGNATION; 4:04 | Exact `My hand stops short` / late hiss; hiss withheld once at the single chisel pulse | PASS |
| P06 | Ambitious / INDIGNATION | LOFN-PRIME INDIGNATION; 4:40 | Exact `Backwash before wave`; B→A returns earn one A→B completion at the sole wrench/water seam | PASS |

Metadata reconciliation is exact: **3 Accessible/AWE + 3 Ambitious/INDIGNATION**. The seven-property eligibility re-read agrees with the frozen barbell: P01–P03 clear the accessible threshold; P04–P06 deliberately fall on the ambitious side through immediate pressure, reduced cognitive ease, monody/deletion/reverse-canon mechanisms, and unresolved endings. AMBITIOUS is a route classification, not a defect.

P02's duration arithmetic is present and correct: `8 + 24 + 18 + 30 + 18 + 48 + 8 + 12 + 8 + 12 = 186 seconds (3:06)`. A post-seam `Final Denial Hook` occurs before the twelve-second afterimage, so the assigned final return is present.

P03's positive Style field contains the exact required phrase **`airy, spacious beats with heavy bag drums`**. Its Lineage block links Fused Arrow Records and the Bandcamp Daily label profile, identifies the phrase as vocabulary/lineage rather than audio, and the Dedication repeats that distinction.

## Generator-Field Names and Human Subjects

Direct inspection of all six Style, Exclude, and Lyrics fields finds **zero real artist/person names**. Real names and organizations occur only in lineage/dedication/provenance below the generator fields. Each song depicts one invented adult in an invented or unnamed place, with body-first action and no identifiable harmed/private person, child, victim biography, or real grief.

The captions and their linked Full Credits/overflow blocks were audited as one shipped paratext surface. They make no claim about a living person's life, motives, standing, collaboration, endorsement, or presence in a generated recording. No possessive or “taken/lifted from” grammar implies that credited source audio is contained in a render.

## Lineage, Dedications, and No-Sample Honesty

- All six have one measured dedication <=500 characters: **385 / 459 / 484 / 415 / 481 / 422**.
- P02 carries the sharp session debt in both caption and overflow: the future adult alto and shelf-impact/foley performer whose breath and single blow realize the score.
- P05 carries the sharp session-player priority in both surfaces: future contrabass performer first, then the future adult low-mezzo vocalist.
- P02, P03, P04, and P06 each have exactly **two linked sources** in `Lineage & Credit`. P01 correctly marks living-scene lineage N/A; P05 treats a historical inscription/conservation record without inventing a living-scene claim.
- Every caption/overflow pair contains honest generated/new-performance wording and denies sampling, source-audio presence, affiliation, endorsement, or collaboration as applicable.
- P06 keeps the required conditional distinction exactly: tim.kahn recorded `Atlantic Ocean Waves` for Freesound under CC BY 4.0; the current prompt requests a generated evocation and **does not use or sample** that recording; if the licensed file is later imported, title, recordist, platform, and license must travel with it. The real person's name appears only as a factual recordist credit, never as the song's subject.

## Singer Surface, Cathedral Engine, and Somatic Gate

| Pair | First-listen event | Second-pass engine / named-corpse read | Skeptic bloc |
|---|---|---|---|
| P01 | Cleaner works an icy exposed rail after a designation changes | Fixed hook changes under three harmonic grounds; the burr causes the sole rupture; danger survives the ending | 3 YES / 0 NO |
| P02 | Archive worker braces a moving shelf while a field reports nothing significant | N-headed canon loses one onset and repays it; double retraction keeps both field and body bounded | 3 YES / 0 NO |
| P03 | Bookkeeper leaves a care-bearing ledger line open as smoke changes the room | Exact chorus, care overrun, wrist lift, and alarm seam transform `open`; 70-line flag is deliberate economy | 3 YES / 0 NO |
| P04 | Worker follows a correct smoke route to an unusable release | Cumulative word loss is transformed at every return and paid by one fixed vacancy; click refuses victory | 3 YES / 0 NO |
| P05 | Conservator stops her hand short above rising water | Monody is intentional; repeated command/hiss transforms when breath is withheld once and the hand releases farther | 3 YES / 0 NO |
| P06 | Marine technician services a moving-deck instrument whose field reads M.M. | Consequence-before-cause canon earns one ordered return; sole stereo water breach remains bounded and conditional | 3 YES / 0 NO |

All named-corpse prompts clear: every piece reaches a second emotional movement; no repeated section does all the work; each central motif transforms or is recontextualized; exact refrain repetition is deliberate return, not creative exhaustion. All six pass the 16-point Singer Surface / Cathedral Engine / Suno Package / Lineage gate on current text.

The frozen selection records that each candidate beat its decoy and that the judge produced zero decoy-over-golden failures. This stable-byte audit did not re-inject Golden Song payloads; its independent adversarial and Somatic reads are reported above.

## Tool Defects and Coverage Notes — Not Package Findings

1. `scripts/check_human_subjects.py` returned `HOLD_FOR_HUMAN` for all six under its `regex-fallback`. This is a false-positive instrument failure: the extractor prepends Markdown H1/H2 headings and retains EMO headers, then the fallback classifies title-cased structural words such as `Eager`, `Archivist`, `Dramaturgy`, `Return`, and emotion names as PERSON entities. It also manufactured a P01 place from the cross-line text `near\nStill`, while generic `body`/`missing` vocabulary triggered sensitive context. Those strings are not people or identifying tuples. Direct title/lyrics/paratext review is CLEAR. The faulty prefilter result is not used as evidence against the songs.
2. `validate_suno_packages.py` is a structural floor, not complete canonical proof: its code enforces >=60 probable sung lines and Exclude <=1,000, whereas the current binding bands are 70–120 and 400–900; it also does not by itself prove five exact Disc channels or fully integrated headers. The independent measurements in this report apply the current gates and all six pass.
3. The distinctiveness tool prints normalized prompt/lyric lengths for comparison. Those values are not raw Suno field lengths and are not reported as such here.

No song, skill, manifest, ledger, memory, or selection artifact was edited. The explicit instruction not to append a SHIP entry to the failure ledger governs this stable-byte audit.

## Per-Song and Final Recommendation

| Pair | Song | Package | Human subject | Somatic | Overall |
|---|---|---|---|---|---|
| P01 | Still in Orbit | PASS | CLEAR | PASS | **SHIP — text only** |
| P02 | No Significant Events | PASS | CLEAR | PASS | **SHIP — text only** |
| P03 | Leave the Line Open | PASS with escalated 70-line boundary FLAG | CLEAR | PASS | **SHIP — text only** |
| P04 | The Last Verb Fails | PASS | CLEAR | PASS | **SHIP — text only** |
| P05 | Short of the Groove | PASS | CLEAR | PASS | **SHIP — text only** |
| P06 | Backwash Before Wave | PASS | CLEAR | PASS | **SHIP — text only** |

**FINAL: SHIP all six frozen text render packages. No render or listening judgment has occurred, and nothing is authorized for publication by this report.**
