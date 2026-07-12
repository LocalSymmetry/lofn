# Lofn QA Report — 2026-07-11 NightCafe Select “UNBRACING” (image)

**Audit scope:** prompt-only pre-render QA. `SHIP` would mean permission to advance the selected engine into final eight-frame packaging, not that unrendered images are submission-ready.

## Verdicts

- **Pipeline Integrity:** REPAIR REQUIRED
- **Package:** PASS
- **Human-Subject:** CLEAR
- **Overall:** REPAIR
- **Engine decision:** **UPHOLD P06 — Failure Becomes Mesh** after the coordinator manifest repair. Do not redirect to P04 at prompt level. Keep **P04 — Negative Cast** as the automatic render-evidence fallback.

The only blocking repair is traceability, not a creative rewrite: rebuild `RUN_STATE.md` from current disk state as one row per expected artifact, with exact path, byte size, SHA, validator verdict, attempt count, and status. The existing pair rows aggregate five files and carry stale byte ranges (for example, Pair 01 reports `10694–20227`, while the current disk range is `11625–20227`) and no per-artifact SHA. This violates `EXECUTION.md` §6’s disk-derived manifest contract. Re-run the structural delta after that coordinator repair; do not regenerate prompts.

## Early-Exit Completeness and Depth Audit

No pipeline-stub early exit fired.

| Layer | Expected | Measured | Verdict |
|---|---:|---:|---|
| Coordinator Steps 00–05 | 6 separate substantive files | 6 | PASS |
| Pair Steps 06–10 | 6 pairs × 5 separate files | 30 | PASS |
| No-skip spine | Step 07/09/10 for every pair | 18/18 present | PASS |
| Step-08 candidates | 4 per pair | 24 total | PASS |
| Step-09 refinements | 4 per pair | 24 total | PASS |
| Step-10 finals | 4 per pair, valid 2+2 JSON | 24 total; 6/6 parse as 2 revised + 2 synthesized | PASS |

### Depth by step

| Step | Measured lines | Required minimum | Verdict |
|---|---:|---:|---|
| 00 | 227 | 50 | PASS |
| 01 | 83 | 50 | PASS |
| 02 | 151 | 50 | PASS |
| 03 | 180 | 80 | PASS |
| 04 | 164 | 50 | PASS |
| 05 | 127 | 60 | PASS |
| Pair 06 | 68–144 | 25 | PASS |
| Pair 07 | 174–218 | 40 | PASS |
| Pair 08 | 72–108 | 60 | PASS |
| Pair 09 | 74–121 | 60 | PASS |
| Pair 10 | 110–158 | 60 | PASS |

All coordinator Step-00–05 validators and all 30 pair Step-06–10 validators were re-run from current disk and passed. Step-06 cross-pair distinctiveness passed with the documented current-image-contract override `--min-facets 5`; Step-09 cross-pair distinctiveness passed.

### Wave-1 authority repair

The first wave initially followed the legacy six-candidate Step-08/09 wording. The recorded repair correctly restored current authority before advancement: six Step-07 explorations remain, four are explicitly advanced, two are explicitly cut, Step 08 and Step 09 each contain four prompts, and Step 10 contains valid 2+2 JSON. The repaired cardinality is present on disk for Pairs 01–03 and validated.

## ICB Integrity and Continuity

| Check | Measured | Verdict |
|---|---|---|
| Canonical `CREATIVE_CONTEXT.md` bytes | 24,512 | PASS |
| SHA-256 | `9e04ca4ca84f15120acb12c7bbbbaadd3afd7f02880f003b202326fd10ab8fd6` | PASS |
| BEGIN marker | 1 | PASS |
| END marker | 1 | PASS |
| `(after ` voices | 18 | PASS |
| Frozen Hyper-Skeptic seats | Interpretation Skeptic; Weight Prosecutor; Catharsis Auditor | PASS |

All 30 pair artifacts cite the frozen `CREATIVE_CONTEXT.md`; Steps 07–10 cite their immediately previous pair artifact or prior pair-artifact set. Their material decisions also demonstrate a successful soul read rather than generic compliance: P06 retains charred brace → broken fragment, differentiated wire gauges, celadon eyelets, brocade collars, stress foil, visible anchors, and the ordinary-hands aftertask; P04 retains the exact negative stair counterform, casein/gesso against lacquer, oak feet, copper contact, handprints, and floor deflection.

Disk provenance has a limit: the actual model-call prompts/envelopes were not persisted, so the saved files cannot independently prove that the 24,512-byte ICB was an unbroken prefix of every unrecorded call. The artifacts consistently assert that injection and show strong personality/seed fidelity, so no thread-loss contradiction is found, but byte counts are not treated as proof of an unrecorded prefix. Future packaging calls should persist an injection receipt rather than relying only on prose claims.

## Structured Deterministic Evidence

No `GATE_REPORT.json` is expected or present; measurements are reported in prose as the fail-open QA path requires.

| Metric | Measured | Contract | Verdict |
|---|---:|---:|---|
| Final Flux prompt count | 24 | 24 | PASS |
| Final prompt words | 118–146 | 80–150 | PASS 24/24 |
| Noun-first/non-imperative opener | 24/24 | 24/24 | PASS |
| Present-tense scene description | 24/24 | 24/24 | PASS |
| Material/medium in first third | 24/24 | 24/24 | PASS |
| Camera/Kelvin/parameter syntax | 0 hits | 0 | PASS |
| Living-artist names | 0 hits | 0 | PASS |
| Storybook words, including `floating` | 0 hits | 0 | PASS |
| Placeholder/scaffold debris | 0 hits | 0 | PASS |
| Attachment paths/prior-winner payload | 0 hits | 0 | PASS |
| Step-10 cross-pair individual-prompt similarity | max 0.437 (P04V3–P05V2) | ≤0.58 | PASS |
| Step-10 cross-pair word 5-gram Jaccard | max 0.004 (P02V2–P04V2) | ≤0.18 | PASS |

The repository’s `validate_portfolio_distinctiveness.py` is music-shaped: it searches for `## 1. MUSIC PROMPT`, obtains empty strings for image files, and therefore reports false `1.000` prompt similarity. It is not used as image evidence. The independent prompt-only JSON extraction above parsed all 24 final image prompts and measured their actual text.

### Word counts by pair

| Pair | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| P01 | 140 | 134 | 135 | 140 |
| P02 | 127 | 133 | 129 | 135 |
| P03 | 127 | 138 | 141 | 136 |
| P04 | 123 | 120 | 130 | 135 |
| P05 | 118 | 125 | 120 | 126 |
| P06 | 121 | 126 | 125 | 146 |

## Seven-Element Density Audit — All 24

Codes: **E** emotional seed in the opening sentence; **M** medium as narrative agent; **S** material specificity; **L** lighting; **F** explicit primary/secondary/tertiary hierarchy; **C** chromatic story; **N** narrative incompleteness. A single opening-seed flag does not fire the auto-fail rule; missing two elements would.

| Prompt | Words | Density | Verdict / evidence |
|---|---:|---:|---|
| P01V1 | 140 | 7/7 | Shoulder/ceiling dependency; tempera/linen/gold/copper; hard rake; three tiers; severe palette; polishing continues under load. |
| P01V2 | 134 | 7/7 | Open hand at raw hole; material cross-section; cold side light; first clearance; hand remains high. |
| P01V3 | 135 | 7/7 | Only clear interval under descending courses; hard lateral shadows; gold hardens at contact; repetition remains unresolved. |
| P01V4 | 140 | 7/7 | Folding task beside independent oak support; daylight stops at unfinished joint; shoulder still rises. |
| P02V1 | 127 | 7/7 | Body visibly bears brocade lintel; singular coral route; hard side rake; lowest course still settles. |
| P02V2 | 133 | 7/7 | Loaded open net and tool-shaped hands; bent anchors; daylight bands; highest pin remains straight. |
| P02V3 | 129 | 7/7 | Controlled pull rotates anchors and lifts lintel; dry crosslight; one anchor has not turned. |
| P02V4 | 135 | 6/7 | **FLAG E:** opening sentence leads with the arch before the human state; all other six elements pass and the mismatched hands restore residue. |
| P03V1 | 127 | 7/7 | Descending wrist and grounded tile; hard shadow; side-pier compression; gaze still measures upper course. |
| P03V2 | 138 | 7/7 | Conservator as load point; six impacts return to two piers; roof light; one contour remains unjoined. |
| P03V3 | 141 | 7/7 | Pressure-creased palm; matte tile/grout/lime; hard rake; second hand hovers at former height. |
| P03V4 | 136 | 7/7 | Self-supporting wall around adult; bounded light; uneven hands; unfinished contour. |
| P04V1 | 123 | 7/7 | Pulling hand during half-landed support transfer; casein/lacquer/oak/copper; raked matte light; shoulder rehearses load. |
| P04V2 | 120 | 7/7 | Decentered woman beside exact counterform; dry side light; handprints/arm hollows; eyes keep counting joint. |
| P04V3 | 130 | 7/7 | One-body-width interval carries aftershock; chalk bands align with prints; no bright exit; gaze remains under stair. |
| P04V4 | 135 | 7/7 | Floor curve retains knee history; exact fit and copper bearing; dry lateral light; one shoulder listens. |
| P05V1 | 118 | 6/7 | **FLAG E:** first sentence is architecture-only and the adult arrives later; other six elements pass, including old bow/new straight edge and final impact-ready stance. |
| P05V2 | 125 | 7/7 | Shoulder held too high; wedge takes bearing; copper straightens; hovering hand and tracking eyes remain. |
| P05V3 | 120 | 7/7 | Elbow stays beneath empty air; glass notches take courses; overhead band; measuring hand persists. |
| P05V4 | 126 | 7/7 | Chosen step crosses one support path; buckled shim and flattened copper; head looks back. |
| P06V1 | 121 | 7/7 | Quiet broken brace above paused hands; four anchored routes lift ceiling seam; shoulders remain raised. |
| P06V2 | 126 | 7/7 | Ordinary fold beneath a visibly loaded shell; restrained daylight; checking finger remains on ceramic. |
| P06V3 | 125 | 7/7 | Adult stands in open center; perimeter loops and three fans terminate in anchors; hovering hand waits for proof. |
| P06V4 | 146 | 7/7 | Body-fitted compression and active wrapped joint; hard rake; widening interval; central brace still bears weight. |

No prompt misses two density elements. P02V4 and P05V1 are substantive portfolio flags, not selected-engine blockers.

## Human-Subject Backstop

**CLEAR.** Every protagonist is explicitly invented and adult (ages 44–52 where specified). `Nera Vale` is declared invented. There are no children, real-person likenesses, identifying real-world tuples, victims, or recent-tragedy reconstruction.

## Official Competition Criteria and Story Test

| Criterion | P06 assessment | P04 fallback assessment |
|---|---|---|
| Concept | Excellent: one rigid support fails into distributed relations; the body reacts late. | Excellent: missing stair volume becomes exact independent support. |
| Craft | Prompt-level PASS; topology is the main render risk. | More render-stable solid mass, exact contact, oak feet, copper edge, floor bend. |
| Cohesion | Strong material bible; identity and eyelet-route locking must be formalized in packaging. | Strong repeated identity and material bible; reverse chronology and repeated cast/stair geometry risk middle-frame redundancy. |
| Intent | Clear agency and anti-cure ending through ordinary hands plus one check. | Clear set-down agency and rough handprint/shoulder residue. |

P06 is capable of one story rather than a mood board: body-fitted compression → repeated over-repair → gauge disparity → quiet break → strands catch → load spreads to edge anchors → permeable shell settles → ordinary hands with one remaining check. Final packaging must make those eight state changes explicit; merely alternating woman-and-wire portraits would fail Cohesion and Intent.

## Adversarial Rejection Case

QA found concrete reasons to reject or stop the package; this is not decorative approval.

1. **Blocking repair — disk manifest is not a disk-derived per-artifact manifest.** Spec: `EXECUTION.md` §6 requires one exact row per expected artifact with `canonical_path`, `byte_size`, `sha`, verdict, attempts, and status. Evidence: `RUN_STATE.md` collapses each pair into entries such as ``pair_01_step06..10 | 10694–20227``; current disk is 11,625–20,227 bytes and no per-artifact SHA is recorded. Harm: a resume or join cannot prove which revision passed and can accept stale state. **Route:** coordinator / `RUN_STATE.md`, re-stat only; do not touch creative files.

2. **Substantive flag — P06 may render as decorative copper enclosure instead of architecture.** Spec: Image hard gate and Weight Prosecutor require visible weight, anchors, and first-glance subject legibility. Evidence: P06V1 asks for “four large wire routes” plus “many fine strands,” and P06V3 asks for “heavy … perimeter loops” plus “three fans of finer wire.” Harm: Flux can merge routes into a uniform web, crop endpoints, turn eyelets into jewelry, and erase the changed load path. **Counter-move:** package a topology bible with only four named anchor families, oversized eyelets, heavy-perimeter/fine-interior gauge separation, and visible endpoints in every structural frame.

3. **Repair flag for packaging — P06 identity detail is uneven across its four finals.** Spec: Cohesion requires one stable adult protagonist. Evidence: V1 carries “long angular face” and V2 carries “salt-and-pepper curls,” while V3/V4 retain age and skin but omit parts of the facial/hair anchor. Harm: an eight-frame render can drift into different women even if materials stay consistent. **Route:** final eight-frame packaging character bible, then NightCafe identity locking; do not rewrite the frozen exploration artifacts.

4. **Substantive portfolio flag — two prompts delay the human seed behind architecture.** Spec: the seven-element density checklist asks the first sentence to carry the emotional seed. Evidence: P02V4 opens “A spare bone-plaster room surrounds an open-warp brocade arch…,” and P05V1 opens “A crisp black-glass shadow-column occupies the central bearing line…”. Harm: these may read as accomplished architectural studies before they read as embodied feeling. Neither is the selected P06 engine, so the flag does not block engine advancement after the manifest repair.

5. **Provenance limitation — saved assertions cannot prove unrecorded prompt prefixes.** Spec: ICB integrity calls for an unbroken verbatim prefix. Evidence: artifacts state the full 24,512-byte ICB was injected, but the actual call envelopes are absent. Harm: future auditors cannot distinguish genuine injection from a good prose claim. This does not become thread-loss here because the canonical ICB is intact and the soul read is unusually specific, but final packaging should persist dispatch receipts.

## Named-Corpse Andon Walk — P06

| Reject condition | Finding | Evidence |
|---|---|---|
| One-note emotional arc | NOT A CORPSE | Compression, compulsive repair, quiet failure, redistributed support, and awkward aftertask create a second and third movement. |
| Single dominant repeated frame | GUARDED PASS | Plan/profile, break aftermath, wide shell, and hand task differ; eight-frame packaging must not repeat the centered woman-and-wire composition. |
| Motif never transforms | NOT A CORPSE | Wire changes from repair wrapping to distributed support; the brace changes from central support to broken witness; hands change from maintenance to modest chosen task. |
| Repeated-line / visual collapse | NOT A CORPSE | Step-06 and Step-09 distinctiveness pass; cross-pair max similarity 0.437 and word 5-gram Jaccard 0.004. Shared identity/material phrases serve continuity rather than exhaustion. |

## Three-Hyper-Skeptic Bloc — P06 Primary Gate

| Exact frozen Hyper-Skeptic | Vote | Finding / required counter-move for NO |
|---|---|---|
| INTERPRETATION SKEPTIC (after Susan Sontag) | **YES** | The broken black member, paused hands, ceiling lift, and anchored routes deliver sensory force before the “distributed relations” interpretation. |
| WEIGHT PROSECUTOR (after Richard Serra) | **NO** | The prose proves force, but Flux may turn the eyelet/selvage network into a tasteful mobile. **Counter-move:** lock four anchor families and reject any cover in which fewer than three endpoints, the charred base, and ceiling response are simultaneously readable. |
| CATHARSIS AUDITOR (after Sianne Ngai) | **YES** | `Ordinary Hands` keeps one incomplete fold, unequal shoulders, a checking finger, and the loaded shell; release creates an awkward task instead of instant healing. |

Bloc result: **2 YES / 1 NO — P06 is not blocked.** QA upholds P06 at prompt level. The Weight Prosecutor’s NO becomes the mandatory pre-render topology test, not a silent waiver.

## Evaluation Integrity

- Run-specific facet rows: 24/24.
- Final competition-ranking rows: 24 unique ranks, 1–24.
- Eligibility rows: 24/24.
- Competition totals recalculate correctly under conventional round-half-up display; no ranking-changing arithmetic error was found.
- Engine table: 6/6 rows; each displayed `/50` total equals its five components.
- Top-six barbell: exactly 3 ACCESSIBLE + 3 AMBITIOUS.
- The distinction between top single candidate (P04V1, 9.26) and selected story engine (P06, 48.5/50) is explicit and logically supported.

The evaluation is trustworthy as prompt-level evidence. It correctly retains P04 as runner-up rather than forcing the top single frame to become the best eight-frame engine.

## Required Repair Brief

**Return target:** coordinator state join / `EXECUTION.md` §6, not a creative step.

**Failed gate:** disk-derived manifest integrity.

**Expected:** one row per expected artifact with exact current `canonical_path`, existence, byte size, SHA, gate verdict, attempt count, and status; the ICB hash remains frozen.

**Actual:** pair chains are aggregated into ranges, several ranges no longer match current files, and pair-file SHAs are absent.

**Repair:** rebuild `RUN_STATE.md` by stat-ing the current directory after all authority repairs. Preserve all existing creative artifacts byte-for-byte. Record all 6 coordinator files and all 30 pair files separately, plus evaluation and QA artifacts. Re-run only manifest consistency and the structural QA delta.

**Sideways proposal:** none; the failed value is provenance bookkeeping and has not entered a repeated no-progress loop. No cut-ledger concept promotion, variation re-derivation, or skeptic re-transformation is warranted.

## Mandatory Pre-Render and Post-Render Checks

### Before rendering / during eight-frame packaging

1. **P06 identity lock:** one character bible in every frame — invented 46-year-old brown-skinned craftswoman, broad shoulders, close-cropped salt-and-pepper curls, long angular face, plain bone-linen work dress, one black-and-celadon brocade cuff. Lock only after choosing the best hero render.
2. **Topology lock:** one charred central brace/base; four named anchor families; palm-sized celadon eyelets; black-and-celadon selvage collars; heavy perimeter gauge versus fine interior gauge; every structural route terminates visibly in a wall plate or floor anchor.
3. **State map:** assign exactly one of the eight causal states to each frame; no frame may exist only for beauty. Preserve one broken black fragment after the turn.
4. **Cover thumbnail test:** at phone-grid size and in grayscale, P06V1 must reveal within one second (a) one adult, (b) one broken black member, (c) at least three readable anchor routes/endpoints, and (d) a ceiling response. Failure triggers one controlled composition correction, not prompt accretion.
5. **P04 fallback packet:** keep P04’s exact cast/stair geometry, adult identity, oak feet, copper edge, handprints, and floor deflection ready without changing its engine logic.

### After pilot renders and after the full eight

1. Verify the same adult identity in all eight; no age, face, hair, skin-tone, clothing, or hand-count drift.
2. Verify anchor readability and load destination in every structural frame. Wire cannot terminate off-frame without a visible receiving support.
3. Verify eyelet/selvage topology: eyelets stay ceramic and palm-sized; collars stay structural textile, not jewelry; perimeter/interior gauges remain distinct; no chandelier, cocoon, or airborne-web reading.
4. Verify the cover at phone-grid size, color and grayscale. If the adult and changed support cannot be retold in one breath, it is not the cover.
5. Verify the final eight tell one causal story: compression → over-repair → quiet break → catch → transfer → settled mesh → ordinary task → structural rhyme. Reject duplicate poses, unchanged motifs, or light-only progression.
6. **Automatic P04 trigger:** after one controlled correction, redirect the engine to P04 if the P06 cover still fails the one-second test, or if two or more pilot/full-sequence frames lose identity, anchor endpoints, eyelet/selvage topology, or visible ceiling response. This is a render-evidence switch, not a creative rescore.

## Failure-Ledger Status

No entry appended to `vault/COMPETITION_LEARNINGS.md` because the Overall verdict is **REPAIR**, not SHIP.

## Final Recommendation

Repair the coordinator manifest, re-run the structural delta, then advance **P06 — Failure Becomes Mesh** into eight-frame packaging under the mandatory identity/topology/thumbnail gates above. P04 remains the fallback only if actual render evidence crosses the stated trigger. No creative artifact should be rewritten during the current repair.
