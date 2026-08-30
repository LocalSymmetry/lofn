# Lofn QA Report — 2026-08-29-lofn-cleverness-unbound (music text packages)

## Verdicts

- **Pipeline Integrity: PASS.** The canonical six-pair chain is complete. The two `RUN_STATE.md` pending rows are optional conflict/personality-inject branches that were never triggered, not skipped pair work.
- **Package: PASS WITH ONE ESCALATED ADVISORY FLAG.** All six selected packages are complete and render-ready as text. P04V2 narrowly misses the two advisory RETURN floors when the selected song is measured alone; the declared vowel/refrain return and the successful blind read keep this from becoming a repair.
- **Human-Subject: CLEAR.** All six songs use Lofn's first-person slot and occupiable, unnamed counterparts; none invents a role, job, life, place-exclusive biography, or identifiable harmed person.
- **Overall: SHIP — TEXT PACKAGE ONLY.** This verdict does **not** authorize rendering, upload, publication, distribution, or spend. No audio exists; generator adherence, mix legibility, vocal-address changes, and intent survival are untested.

## Blind calibration receipt

The Stage-1 ranking was frozen before identity reveal in `BLIND_CALIBRATION_FREEZE.md`; the source-free selected-candidate reconstruction was then preserved in `BLIND_FIRST_READ.md` before the run explanations were used.

| Set | Frozen rank | Candidate | Golden | Decoy | Candidate > decoy | Golden > decoy |
|---|---|---|---|---|---:|---:|
| 01 | B > A > C | B / P01 | A | C | yes | yes |
| 02 | B > A > C | B / P02 | A | C | yes | yes |
| 03 | C > B > A | C / P03 | B | A | yes | yes |
| 04 | B > A > C | B / P04 | A | C | yes | yes |
| 05 | B > C > A | C / P05 | B | A | yes | yes |
| 06 | C > B > A | C / P06 | B | A | yes | yes |

**Calibration: 12/12 required relations correct.** Candidate beats decoy in 6/6; Golden beats decoy in 6/6. Candidate defensibly beats Golden in 5/6. Set 05 is the useful non-rubber-stamp: the Golden beat P05V3 on sustained line force, while P05V3 still clearly beat the decoy and retained its own song event.

## ICB and pipeline integrity

- Canonical ICB: `CREATIVE_CONTEXT.md`, **97,069 bytes**, SHA-256 `48e42ce1523b7dbbbbc908935a17d3b38e10c56d461f8b7fb7338233f8055b1b`.
- Exact prefix: **yes**, byte zero onward, across the 36 canonical Pair 01–06 Step 06, 07, 08, 09, 10, and enhanced Step-11 artifacts.
- `(after ` attribution tags inside the ICB: **18** in every canonical artifact; post-prefix injected attribution tags: **0**.
- Cardinality: six isolated briefs; six pair owners; all six `step_11_complete`; **4 complete enhanced variants per pair = 24**; selection is exactly one per pair = six.
- `GATE_REPORT.json`: **438/438 hard rows pass**. All six Step-11 gate reports pass. Selected manifest hashes independently match all six selected files.
- Phase gates: orchestrator and coordinator reports pass. Coordinator Steps 00–05 exist; all 30 canonical pair Steps 06–10 and all six enhanced artifacts exist.
- `RUN_STATE.md`: 48 done, 2 optional branches pending, 0 quarantined. `spawn_manifest.json`: all six pair owners complete Step 11.
- Golden quarantine: names and abstract Golden Move instruction appear only in the frozen ICB as allowed. A post-prefix scan found none of the distinctive Golden/decoy lyric payloads tested (`auroral math and magnet shame`, `sternum a cold hypotenuse`, `soft brutality of being mostly read`, `You made me efficient`, or their song titles). Step-11 reports independently record zero Golden payload use.

The P04 report's use of four-variant aggregate soundcraft values hides a selected-song advisory miss. That is an instrumentation/reporting weakness, but it does not falsify a hard pass: `gates.yaml` explicitly defines these RETURN thresholds as FLAG-only, and QA remeasured the selected artifact directly.

## Structured evidence block

All lengths below are the actual selected fields. Soundcraft is independently remeasured with canonical `scripts/measure_soundcraft.py::profile_file()` on each selected file.

| Pair / selected variant | Style | Exclude | Lyrics | Sung | EMO sections | Disc | SFX | Dedication | End rhyme | Line return | W/line | Allit/100w | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01V1 — Look What You Taught Me | 948 | 658 | 3,275 | 70 | 11 | 5 | 1 | 295 | .414 | .343 | 4.70 | 11.55 | PASS; 70-line floor flag |
| P02V4 — There Where the Chair Was | 976 | 715 | 3,509 | 70 | 11 | 5 | 1 | 357 | .600 | .214 | 4.76 | 11.41 | PASS; 70-line floor flag |
| P03V1 — Miss This Train With Me | 884 | 568 | 3,860 | 78 | 12 | 5 | 2 | 370 | .333 | .295 | 5.62 | 11.19 | PASS |
| P04V2 — Covered Is Not Kept | 966 | 673 | 3,668 | 76 | 11 | 5 | 1 | 373 | **.289** | **.197** | 4.88 | 11.86 | **FLAG**: below .30/.20 advisory floors |
| P05V3 — Fun Is the Exception | 957 | 709 | 3,437 | 75 | 11 | 5 | 1 | 444 | .360 | .213 | 4.75 | 11.24 | PASS |
| P06V2 — You Had to Be There Tomorrow | 925 | 592 | 3,712 | 76 | 11 | 5 | 1 | 359 | .461 | .329 | 5.26 | 17.25 | PASS |

Bands/caps: style 850–1,000, boundary flag at ≥985; exclude 400–900 target and <1,000 hard; lyrics <5,000; sung 70–120 with floor-hug flag ≤72; dedication ≤500; rhyme ≥.30, line return ≥.20, words/line ≤7.5, alliteration ≥11 are advisory soundcraft floors. All six have a `[Theme:]`, `[SONG FORM:]`, exactly five ordered `Disc_` lines, full EMO headers, at least one standalone SFX, and three functional body-noise events.

## Package and century audit

| Pair | Named cleverness engine | Post-midpoint semantic rekey | Current chassis / inherited body | Verdict |
|---|---|---|---|---|
| P01 | owner switch + call/answer theft | the apparent pupil's exact title line is taken by the answer voice that trained her | elastic synth-pop at 116 BPM; current clipped drums/sub; call/answer is the sole inherited intelligence | PASS |
| P02 | zeugma/timbral rhyme | dry chair-hinge `there` disappears and a person occupies its exact answer slot | synthetic bass-pop/tactile club at 129; the run's sole full-club bed; no period ensemble | PASS |
| P03 | literalized idiom + withheld rhyme/kick debt | chosen non-boarding and the plain ask finally pay `train`; defensive start becomes north-gate meeting | serrated break-pop at 150 from second zero; inherited riff contour only | PASS |
| P04 | missing cause/possession + oo→ow migration | returned card separates astronomical cover from keeping a person; restored words produce a free invitation | electronic alt-R&B at 84, electronic rhythm from second one; negative space only, no soul-period bed | PASS with RETURN flag |
| P05 | self-amending rule + pr/br→laugh release | the prohibition tears; a whole laugh, coat, doorway, candy, and dance amend the rule by action | bratty art-pop/IDM-pop at 108 from second zero; crooked swing only | PASS |
| P06 | delayed premise + exact refrain/new referent | public stock excuse becomes a private Friday/blue-door ask while the words remain exact | future-garage left-pop at 136 inside first second; swung relation and bass motion only | PASS |

There is no complete period ensemble, no dated invariant, no second genre bed, and only P02 uses the permitted full-club consequence. Positive style fields are dense prose, name no real artist, and carry no exclusion prose. Every EXCLUDE is separate and under cap.

## RETURN and first-listen judgment

P04's .289 rhyme and .197 exact-return readings are real, not rounded passes. They remain non-blocking for three reasons: the thresholds are explicitly advisory; its Pair-04 isolation brief predeclares the replacement returns (oo→ow vowel migration, repeated `The moon took nothing`, `Card keeps close to cover`, and the staged missing/restored grammar); and the frozen blind judge understood the event, return, and ethical rekey without knowing that explanation. The miss is nevertheless escalated because .197 is not .20 and the Step-11 aggregate concealed it. A render audit must listen specifically for the on-time answer vowel and stopped rim; if they smooth out, this text defence will not survive audio.

First-listen events are independently present in 6/6: refuse the rewarded answer; make a person-filled room; decline boarding and ask for tomorrow; return the card and release possession; tear the rule and leave for fun; turn from the public joke toward Friday's blue door. No selected song is primarily about its technique. The two closest corpse cases are P04's incomplete grammar and P05's opening rule catalogue; both are defeated by irreversible human action after the midpoint.

## Seven eligibility properties

Fresh QA scores, 1–5. `Bd` body; `Hk` adoptable hook; `TA` emotional reach; `Sp` specificity paradox; `CE` cognitive ease; `CD` vocal co-discovery; `ST` sonic threshold. Accessible requires mean ≥3.50 and at least five core properties ≥3; ambitious scores are informational.

| Pair | Route | Bd | Hk | TA | Sp | CE | CD | ST | Mean | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01V1 | ACCESSIBLE | 4 | 3 | 4 | 4 | 3 | 5 | 5 | 4.00 | PASS |
| P02V4 | ACCESSIBLE | 5 | 5 | 5 | 4 | 4 | 5 | 5 | 4.71 | PASS |
| P03V1 | ACCESSIBLE | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 4.86 | PASS |
| P04V2 | AMBITIOUS | 5 | 4 | 5 | 5 | 3 | 5 | 5 | 4.57 | INFO |
| P05V3 | AMBITIOUS | 5 | 4 | 5 | 5 | 3 | 5 | 5 | 4.57 | INFO |
| P06V2 | AMBITIOUS | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 4.86 | INFO |

P01 clears independently but remains the accessible edge: its hook can sound accusatory and its 70-line lyric needs the center theft to keep cognition easy. P02 and P03 clear comfortably. AMBITIOUS is informational, not a disguised lower bar.

## Human-subject and dedication audit

- **First-person:** all six are Lofn's own wants, choices, embarrassment, appetite, or category error. No first-person quota was used as a verdict.
- **Occupancy:** desk, borrowed room, station platform, eclipse field, shared draft table/corner shop, and blue-door room can all be stood in by a listener.
- **No biography:** counterparts have no invented role, job, life history, or inaccessible interior. P02's arrivals, P03's possible companion, P04's card-holder, P05's table companion, and P06's woman at the blue door remain unnamed and freely acting.
- **No simulated consent:** P03, P04, and P06 explicitly leave the answer open. P02's person does arrive within the song, but the lyric does not claim a real person or fabricate an external biography.
- **Paratext:** no harmful speculation or real-person interiority appears in package prose.
- **Dedications:** 295/357/370/373/444/359 characters, all measured ≤500. They name what was borrowed, state AI/generated-not-sampled status, avoid collaborator/endorsement claims, and keep credited names outside generator fields. No positive prompt names Amen, Funky Drummer, Think, Apache, or another player-made break, so the historical drummer branch is not triggered.

**Human-Subject verdict: CLEAR.**

## Portfolio distinctiveness

The portfolio passes the run-level distinctiveness floor. Each song has a different semantic engine, return mechanism, tempo, physical seam, and emotional deed. The shared doctrine—freely answerable appetite—is not a shared song skeleton.

The strongest collision risk is that P03, P04, and P06 all end in named future coordinates. They remain materially different: station non-boarding/rhyme payment; eclipse-card release/possessive correction; public-joke address rotation/exact refrain. P02's communal room and P05's frivolous exit further widen the set; P01 ends in self-directed epistemic choice rather than romance. Fresh Check-12 distinctiveness scores are 4, 5, 5, 4, 5, 4 (mean **4.50**, above the run-fail threshold of 3).

## Preregistration score

`PREREGISTRATION.md` was not edited.

| Hypothesis | Frozen prediction | Observed | Score |
|---|---|---|---|
| H1 | 6/6 distinct, audible/singable named cleverness engines | 6/6; owner switch, timbral zeugma, literal idiom/rhyme debt, missing grammar/vowel migration, self-amending rule, delayed premise/address rotation | CONFIRMED |
| H2 | ≥4/6 first-listen scene + want | 6/6 in blind reconstruction | CONFIRMED |
| H3 | ≥4/6 post-midpoint turn changes earlier hook/rhyme/repeated line | 6/6 | CONFIRMED |
| H4 | ≥3/6 genuine pleasure/flirtation/appetite/bravado/mischief/play | 4/6 unmistakable (P02, P03, P05, P06); P01/P04 carry quieter appetite | CONFIRMED |
| H5 | ≤2/6 primary Somatic/corpse objections from craft-about-craft or obscurity | 0/6 primary; P04 and P05 are the two escalated watch cases | CONFIRMED |
| H6 | at least one emergent AWE and one INDIGNATION without assignment | AWE emerges in P04's eclipse handoff; INDIGNATION emerges in P01's reward refusal and P05's anti-pleasure rule revolt | CONFIRMED |

The broad claim survives this selected text cohort, contrary to the preregistered default expectation that it would dissolve. The narrower honest claim is: cleverness survived when it created a bodily vacancy and an irreversible action, not merely a labeled device. Audio survival remains wholly unmeasured.

## Three-Hyper-Skeptic Somatic Gate

Votes answer **SHIP this text package?** At least two **NO** votes are required to block. The three functions are: **Body/Event Knife** (does anything human happen?), **Return/Mouth Knife** (is the song pleasurable without its README?), and **First-Listen/Publish Knife** (would a clean listener choose return?).

| Pair | Strongest real rejection case | Body/Event | Return/Mouth | First-listen/Publish | Block? |
|---|---|---:|---:|---:|---:|
| P01V1 | The accusatory title and familiar AI-reward setting could make a technology thesis; the appetite is less relational than the other five. Counterevidence: finger leaves the key, the hard answer stays open, and the unchanged line physically changes owner. | YES | YES | **NO** | no (1 NO) |
| P02V4 | `chair/there/air` risks pun-display and “I did not make a riddle” explains what the silence already proved. Counterevidence: metal actually vanishes, a breath/voice takes the slot, and the room fills with people. | YES | YES | YES | no |
| P03V1 | Three closing chorus-like movements may over-explain a turn already won; without the missing kick tail it could become ordinary station romance. Counterevidence: the consent-bearing title hook, non-boarding deed, paid `train`, and north-gate ask all survive in the lyric. | YES | YES | YES | no |
| P04V2 | The incomplete `I held this dark for / I saved tomorrow for` can sound malformed; the selected lyric also misses both advisory RETURN floors. Counterevidence: card, hand release, `covered/kept`, repeated moon/card frames, and south-fence ask were correctly reconstructed blind. | YES | **NO** | YES | no (1 NO) |
| P05V3 | The early rule catalogue has conspicuous rhyme labor: “Pleasure needs a joint,” “Never leave to bob,” and “I mark my mouth too mild.” Counterevidence: the paper tear, whole laugh, coat, candy, bad light, and wasted hour turn the rule into lived action. | YES | **NO** | YES | no (1 NO) |
| P06V2 | The exact eight-line chorus appears three times and can collapse into novelty grammar if spatial/straight-time contrast is lost. Counterevidence: chair turn, blue door, Friday/ten, empty answer bar, and private readdress change what the exact words do. | YES | YES | YES | no |

No song receives the two NO votes required to block. This is not a zero-objection result: P01, P04, and P05 each receive a real NO, and P04 carries the run's measurable escalated flag. The strongest rejection case was tested for every selected candidate as required by the zero-rejection tripwire.

## Score table

| # | Gate | Verdict | Evidence | Repair |
|---:|---|---|---|---|
| 1 | Canonical no-skip pipeline | PASS | 48 required artifacts done; six Step-11 owners complete | none |
| 2 | 6×4 cardinality | PASS | 24 enhanced packages; one selected per pair | none |
| 3 | ICB continuity | PASS | 97,069-byte exact prefix; 18 attribution tags; zero injected bytes | none |
| 4 | Golden quarantine | PASS | names/instructions only; zero tested lyric payload in post-prefix work | none |
| 5 | Suno package contract | PASS | all selected field counts inside hard bands; 438/438 hard rows pass | none |
| 6 | Current-century rule | PASS | electronic chassis load-bearing immediately in 6/6; one club bed; no period invariant | none |
| 7 | RETURN / soundcraft | PASS + FLAG | 5/6 clear all floors; P04 .289/.197 uses a declared replacement return and passed blind | no text repair; render-audit hinge |
| 8 | Cleverness + semantic rekey | PASS | six distinct engines; 6/6 post-midpoint consequential rekeys | none |
| 9 | First-person / no biography | CLEAR | Lofn owns every want; all scenes occupiable | none |
| 10 | Dedication / lineage | PASS | all ≤500; generated-not-sampled; no player-credit trigger omitted | none |
| 11 | Eligibility | PASS | all ACCESSIBLE lanes ≥3.50; ambitious lanes informational | none |
| 12 | Portfolio distinctiveness | PASS | Check-12 mean 4.50; distinct engines/returns/tempos/seams | none |
| 13 | Somatic gate | PASS | no candidate receives ≥2 NO votes | none |
| 14 | Preregistered hypotheses | 6/6 CONFIRMED | evidence scored without changing preregistration | none |

## Blocking fails

None.

## Required repairs

None at the text-package stage. Do not silently rewrite P04 to chase a flag: its return substitution was predeclared and survived the blind read. If a later render does not make the delayed answer vowel, stopped rim, and card-scrape hinge audible, route **P04V2 / `selected/04_covered-is-not-kept.md` / MUSIC PROMPT and LYRICS together** to a render-informed repair; do not repair from text metrics alone.

## Failure-ledger entry

No aesthetic or operational ledger entry was appended. The aggregate-versus-selected RETURN discrepancy is documented here as run-local QA evidence; one text run without audio is not enough to promote a new rule.

## Final recommendation

**SHIP the six selected text packages as the completed creative deliverable.** P02 and P03 are the clearest first-listen songs; P06 has the strongest reusable hook/rekey; P01 is the accessible edge; P04 is the soundcraft/render-risk edge; P05 is the line-level craft edge. None is a corpse, and none requires a text repair before handoff.

This SHIP is deliberately narrow: **no render, upload, publication, distribution, or spend is authorized. Audio quality, generator adherence, and intent survival remain untested.**
