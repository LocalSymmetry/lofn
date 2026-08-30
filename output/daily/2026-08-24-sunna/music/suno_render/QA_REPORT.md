# Lofn QA Report — SUNNA R4 Top Six (music text packages)

**Audit date:** 2026-08-24  
**Scope:** the six numbered Markdown packages in this directory, their current manifest/source identity, the required prior blind reports, and the six binding soundcraft-repair reports.  
**Render status:** **NO AUDIO EXISTS IN THIS AUDIT. No render or listening pass occurred.** Every verdict below is for the text package only; none is a listening, publication, or spend authorization.

## Verdicts

- **Pipeline Integrity:** PASS within the requested final-package scope. Coordinator Steps 00–05 and separate per-pair Steps 06–11 are present; no no-skip/non-canonical condition fired; the prior blind judge calibrated correctly (golden above decoy), and the repaired P01 candidate was independently re-ranked above both controls.
- **Suno Package:** **PASS — 6/6.** P04's sung craft-narration defect and the later R4 HUM-test omission are both repaired; its fresh independent recheck is `P04_FINAL_RECHECK.md`.
- **Human Subject:** CLEAR — all lyric subjects are invented/anonymous adults in public social events. Public musicians and recordists appear only in credit/debt text, not as story subjects or implied collaborators.
- **Overall:** **SHIP — six text packages, zero open repairs.** The menu was rebuilt from the repaired canonical P04 source and all six standalone packages revalidate. This remains text-only: rendering still requires The Scientist's authorization and every render remains subject to her ear.

No `GATE_REPORT.json` exists in scope, so the deterministic evidence below comes from fresh validator runs, `measure_soundcraft.py`, and a read-only rebuild of the manifest extractor.

## Fresh deterministic evidence

`python skills/music/scripts/validate_suno_packages.py <six numbered files>` returned **PASS 6/6** with exit `0`.

`python scripts/measure_soundcraft.py <six numbered files>` returned exit `0`:

| Candidate | Prompt chars | Exclude chars | Lyrics chars | Sung lines | End rhyme | Exact line return | Words/line | Alliteration/100w | Structural result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01V01 | 963 | 666 | 3423 | 76 | .355 | .447 | 4.32 | 12.80 | PASS |
| P02V03 | 945 | 790 | 3411 | 74 | .351 | .338 | 4.88 | 13.30 | PASS |
| P03V01 | 950 | 603 | 3404 | 74 | .473 | .243 | 4.66 | 18.55 | PASS |
| P04V04 | 984 | 702 | 3044 | 70 | .371 | .243 | 4.10 | 15.33 | PASS + sung-line floor FLAG |
| P05V04 | 932 | 618 | 3512 | 73 | .507 | .274 | 4.22 | 12.34 | PASS |
| P06V04 | 957 | 625 | 3727 | 72 | .417 | .250 | 5.29 | 13.12 | PASS |

All prompts are complete dense paragraphs in the 850–1000 band and end in terminal punctuation. All lyrics fields are below 5000 characters; all section emotions resolve to the taxonomy; each package has at least one SFX cue and direct adult `you` address. Exact returns are event-owned rather than filler. P04 is now `984` prompt characters, below the `>=985` prompt-boundary flag; its `70` sung lines retain the floor FLAG only, which the fresh Somatic read cleared as intentional short-cell return rather than padding or truncation.

No prompt contains a word-boundary hit for `amen`, `funky drummer`, `think break`, or `apache break`. No real artist is named in a generator-facing prompt.

## Manifest and source-field integrity

The current source-block SHA-256 and all four manifest field counts were recomputed with the run's read-only extraction functions. All six current hashes match `build_manifest.json`; prompt, exclude, lyrics, title, dedication, and lineage fields in every numbered file are byte-equal to the current canonical source extraction.

| Candidate | Current source SHA-256 | Manifest/count result |
|---|---|---|
| P01V01 | `5334eaa297cbf8bdd3e77b984424c259194ae989f5df1d42df21a2f86eaf5ee5` | MATCH |
| P02V03 | `ca4be5e80f33826b84b97219e18eb98b1861ff2e1b4b8c629b92aa21f598e87c` | MATCH |
| P03V01 | `8b03f585141ca1c9463b74fee3314dad198c40192c140777166771dcd64e72fe` | MATCH |
| P04V04 | `ca5c8b5d3353eded81dbe13043d762b83c69c1ec066b839088abe5ba0ae3571d` | MATCH |
| P05V04 | `be0563368962c5b0fda0ec7ff511c33f501cfa9ad415d523ed898501324aa53b` | MATCH |
| P06V04 | `08ee797f373acbfa0c96d32137f74088192a2a054dd98bb0f19e8ccbef2bf8d5` | MATCH |

**P04 final provenance:** the focused fresh report `P04_FINAL_RECHECK.md` reproduces the repaired Step-09/10/11 selected-block hashes and stamps the canonical LF Step-11 source `ca5c8b5d…` SHIP. The controller updated the builder's frozen source lock to that hash, rebuilt the render menu, and the rebuilt manifest now records `prompt 984 / exclude 702 / dedication 320`. The menu package is current, not the pre-repair `c744e67d…` or intermediate `6067b2d7…` source.

## Portfolio contract

| Candidate | Frozen arm | Axis B / pole | Unique event | One seam | Result |
|---|---|---|---|---|---|
| P01V01 | ACCESSIBLE | EXISTENCE | closing roller-rink lap; shoulder cue opens the inside line | four-note brass replaces the clipped third guitar chop | PASS |
| P02V03 | ACCESSIBLE | NEWS | damp laundromat claim ticket; blue square washes while the crescent stays | one swooning seventh stack replaces the brass center | PASS |
| P03V01 | ACCESSIBLE | EXISTENCE | arcade compact mishap; she spends every ticket on the enormous shark | one rocksteady pocket at the prize-choice crossing | PASS |
| P04V04 | AMBITIOUS | NEWS / AWE | concession cup moves; the chosen adult occupies the gap before credits | one low-dembow hinge beneath continuous ska bass; defining horn riff returns after final vocal | PASS |
| P05V04 | AMBITIOUS | EXISTENCE / INDIGNATION | backstage wrong scan becomes a shared password and chosen door | hand percussion substitutes for glam toms | PASS |
| P06V04 | AMBITIOUS | EXISTENCE | ferry-deck kiss; phone is pocketed and she takes the gangway first | compressed downstrokes/claps replace dub air | PASS |

The frozen portfolio is exactly **3 ACCESSIBLE + 3 AMBITIOUS**, **2 NEWS + 4 EXISTENCE**, with AWE carried by P04 and INDIGNATION carried explicitly by P05 (and secondarily by P02). The six rooms, objects, causal actions, returns, and seams are distinct. These are frozen Sunna arm assignments; they are not post-hoc relabeled merely because an ambitious song also has an easy surface.

## File verdicts

| File | Text-package verdict | Concrete adversarial finding | Human-escalation outcome |
|---|---|---|---|
| `01_look_with_your_eyes.md` | **SHIP** | The exact second chorus changes truth-value only after stripe-hold → shoulder cue → crossing → adjacent-wheel consequence. High shine, direct address, and boundary contraband remain first-listen flirtation. | No text escalation. The Scientist's ear remains mandatory after any render. |
| `02_leave_it_crooked.md` | **SHIP** | Damp ticket, blue-square removal, retained crescent, `fold it` → `keep it`, and missing-center return form one causal event. `Leave one note out` is craft-aware but still functions as an addressed, singable tease rather than an explanation. The file also carries duplicated source-side subheads after Lineage; they are non-generator-facing packaging clutter, not lyric leakage. | No human override needed at text stage. Keep the final-paste boundary at Sections 1/1B/2/3 explicit. |
| `03_pretty_on_purpose.md` | **SHIP** | The repaired returns remain owned by the admirer (`Keep talking`, `Hold this shark`, `Bring the shark, bring you`); the giant-prize action completes the beauty/intelligence contraband without a disclaimer. The dense local rhyme is conspicuous but still reads as candy-pop craft, not credentials. | No text escalation. |
| `04_again_before_credits.md` | **SHIP** | Fresh `P04_FINAL_RECHECK.md` confirms both repair layers: all sung bass/horn self-explanation is gone; cup → step → straw → grin → shoulder → choice → `AGAIN` completes the human event even if the hinge pitch is smoothed. The R4 HUM TEST now closes explicitly: the defining two-note horn riff returns after final `Ah—again`, before the lid pop. Prompt `984`; dedication `320`; source `ca5c8b5d…`; soundcraft `.371 / .243 / 4.10 / 15.33`, 70 lines. | No text escalation. The rebuilt standalone package passes; The Scientist's ear remains mandatory after any authorized render. |
| `05_scan_me_twice.md` | **SHIP** | The earlier access-suite concern remains visible in headers, but the sung surface now reads as a flirt game: tangerine wristband, door, shoulder, shared password, question/permission mirror. The screen loses authority without a market/platform lecture. `The bass walks through` is one isolated craft-aware phrase, not a controlling pattern. | No text escalation. Render audit should check that the access-level form feels offhand rather than procedural. |
| `06_high_tide_high_gloss.md` | **SHIP** | `No invoice on this kiss` remains the only edge. On the page it is a six-word detonation joke, immediately returned to empty hand, kiss, chipped nail, ropes, and gangway action; the lyric never explains the metaphor. The text therefore passes contraband/no-lecture. | **Mandatory listening escalation after render:** The Scientist must hear this line. If delivery becomes instructional, slogan-like, or solemn, HOLD the render; if it stays tossed-off and sexy, the text verdict stands. |

## Dedication gate — Start With One

| Candidate | Characters | Borrowed debt / who is owed | Player/source and no-sample honesty | Verdict |
|---|---:|---|---|---|
| P01V01 | 318 | Jamaican ska, two-tone, power-pop return; Lyn Taitt named | no melody/recording sampled or imitated | PASS |
| P02V03 | 296 | Jamaican practice, Panamanian reggae en español, Puerto Rican underground builders | new hook/brass named; no sample claimed | PASS |
| P03V01 | 342 | Jamaican rocksteady and sound-system communities; session singers/electric players | generated music, not a sample; no likeness request | PASS |
| P04V04 | 320 | Lloyd Brevett, Jamaican/Panamanian/Puerto Rican carriers, Comet 220P observers | grammar/fact rather than recordings; AI named; `full credits below` is truthful | PASS |
| P05V04 | 335 | rhythm sections plus Jamaican/Panamanian/Puerto Rican lineages | generated track contains no sample; no likeness claim | PASS |
| P06V04 | 299 | Jamaican rocksteady/dub communities and tim.kahn's Atlantic recording | explicitly generated sounds; no sample claim | PASS |

All canonical dedications are measured below 500 characters. No positive style field raises a named break or historical sound figure that lacks its player credit. No dedication claims a living person as the lyric subject, collaborator, endorser, or source of contained audio.

## Somatic Gate — three adversarial reads

| Candidate | Singer-Surface Skeptic | Remark-Test Skeptic | Generator-Survival Skeptic | Bloc result |
|---|---|---|---|---|
| P01V01 | YES | YES | YES | PASS |
| P02V03 | YES | YES | YES | PASS |
| P03V01 | YES | YES | YES | PASS |
| P04V04 | YES | YES — human event owns hinge/tail; production mechanics are not sung | YES — event survives hinge smoothing; prompt and terminal SFX specify the post-vocal riff return | PASS |
| P05V04 | YES | YES, with access-suite watch | YES | PASS |
| P06V04 | YES | YES on text, delivery watch | YES | PASS |

P04's two old counter-moves are now satisfied: the relational action carries the meaning, and the repaired post-vocal horn return completes the R4 instrumental-hook debt without restoring sung craft narration. Its fresh independent Somatic bloc is `YES / YES / YES`.

Named-corpse walk: none of the six SHIP packages has a one-note emotional arc, single filler-supported dominant section, untransformed motif, or exhausted repeated-line pattern. P04's 70-line floor and exact returns are purposeful contraction/return, not collapse.

## Resolved P04 repair chain

P04 returned alone through Steps 09–11. The first repair removed the sung arrangement narration, completed the human/object hinge, and corrected `full credits below`. The independent fresh recheck then caught the pre-existing R4 HUM-test omission; the pair author added one explicit post-vocal return of the defining two-note horn riff to the positive prompt and terminal SFX without changing a sung line. `P04_FINAL_RECHECK.md` stamps the repaired canonical source SHIP. The controller locked `ca5c8b5d…`, rebuilt the menu, and all six standalone package validations now PASS.

No sideways redirect, reserve substitution, or cross-arm change was needed.

## Four-field daily status

**pairs shipped / pairs quarantined / total gate-retries / historical QA repairs+holds issued: `6 / 0 / 10 / 1 resolved`**

Basis: final selected-pair dispositions after the focused P04 closure. The ten cumulative validator retries visible on disk are Step 05 (`2`), P01/P02/P03/P04/P05 Step 06 (`5`), P03 Step 08 (`2`), and P03 Step 10 (`1`). Final QA historically issued one REPAIR and zero HOLDs; that repair is now closed, so current open repairs/holds are `0 / 0`.

## Final recommendation

**All six text packages SHIP; zero text repairs remain.** The repaired menu is ready for The Scientist to decide whether to authorize rendering. No audio has been rendered or heard, and no text verdict substitutes for her ear. No failure-ledger or operational-ledger writeback was performed because this closure was explicitly restricted to the two requested reports.
