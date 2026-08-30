# Lofn QA Report — 2026-08-25 Sunna Modern Swagger (Music)

**Audit scope:** fresh independent, blind-first review of written text packages. No audio was rendered or heard. No network, external model, render, upload, publication, or spend was used. The live run lock was inspected read-only and was not touched or released.

## Verdicts

- **Pipeline Integrity:** **REPAIR REQUIRED** — full 00–10 granularity and 6×4 cardinality exist, but the mandatory transformed-panel configurations are absent and `RUN_STATE.md` is stale.
- **Package:** **REPAIR REQUIRED** — the six selected Suno fields pass numeric/package checks, but P05 contains a non-taxonomy EMO term; P02/P03/P06 fail the living-lineage completeness gate; P01 has an internal body-noise receipt mismatch.
- **Human-Subject:** **CLEAR** — no identifiable real private person, harmed real person, or minor appears in any title/lyric; Sunna is the permitted first-person subject and all counterpart adults are invented.
- **Overall:** **REPAIR** — do not render, upload, publish, spend, or call these canonical until the repairs below land and fresh QA is rerun.

## Blind-First Evidence

The frozen listener-first read is [BLIND_FIRST_READ.md](BLIND_FIRST_READ.md). It was written before opening the selection ledger, research, Golden Seed, panels, pair map, canonical chain, shared dedications, or credits.

Blind result: no song was a first-read corpse. All six communicated a concrete event, current-vs-tribute-act position, first-person stake, and hook/return. The substantive set-level flag was a repeated house mechanism: glossy machine/object setting → missing beat/note/slot → adult answer or object-noise substitution → bass/drop consequence → exact return → abrupt receipt.

## Blocking Fails and Required Repairs

### 1. Mandatory panel transformations are absent

`03_orchestrator_panel_debate.md` contains 18 credited voices and three Hyper-Skeptics across Concept, Medium, and Context panels, but contains **zero** `BASELINE`, `GROUP TRANSFORM: <op>`, or `SKEPTIC TRANSFORM: <op>` configurations. A repo-wide search of this run found none. `EXECUTION.md §4` defines this as a mandatory transformation-gate collapse failure.

**Repair route:** return to Phase 1. Run and save the baseline, named group transformation, and named skeptic transformation with actual inter-seat disagreement. Because that changes the frozen ICB, rebuild `CREATIVE_CONTEXT.md` and re-run downstream coordinator/pair work from the new ICB; do not append a retrospective panel receipt to artifacts generated without that pressure.

### 2. Disk-derived run state is stale

`RUN_STATE.md` still says coordinator Steps 00–05, the ICB, pair chains, selection, Step 11, and QA are pending, even though those artifacts exist. It is a 617-byte narrative note, not the required per-artifact disk-derived manifest.

**Repair route:** coordinator rebuilds `RUN_STATE.md` from disk with one row per expected artifact, byte size/SHA/gate status/attempt/status, then heartbeats with the existing controller ID. This auditor did not touch the lock.

### 3. P05 uses a non-taxonomy EMO term

`The Version After This` contains `[Cross-Out Round - EMO:Suspense + Comic Nerve - …]`. `Comic Nerve` does not occur in `EMOTION_TAXONOMY.md`; all other selected EMO terms resolve.

**Repair route:** selected-enhancement/package cleanup. Replace the descriptive phrase with the intended canonical taxonomy emotion (for example the existing `Amusement`/`Playfulness` family), preserve the section's comic function, and re-run exact taxonomy plus Suno-package validation. Do not alter the lyric event or seam.

### 4. Living-lineage blocks are incomplete on P02, P03, and P06

The 16-point gate requires a living-scene track to name the scene/region, cite 2–3 source artists or labels with links, state Borrowed/Made boundaries, and deny endorsement/sample claims.

- **P02 — Beat Me to the Line:** names new jack swing and R&B, but gives no 2–3 source artists/labels or links.
- **P03 — Future, Dance Back:** names Jamaican dancehall/Caribbean sound-system lineage, but gives no 2–3 source artists/labels or links.
- **P06 — Don't Put Me in Past Tense:** names glam and electroclash, but gives no 2–3 source artists/labels or links.

P01 and P04 satisfy this form; P05 invokes historical/common-practice sequence rather than a living-scene source. No prompt names an amen/Funky Drummer/Think/Apache figure, so no missing session-player debt was detected.

**Repair route:** credits/Lineage & Credit only. Add source-grounded 2–3 artist/label citations and explicit Borrowed/Made sentences for P02/P03/P06; do not invent citations, imply collaboration, or put names in generator fields. Re-measure the ≤500-character dedication only if its text changes.

### 5. P01 body-noise receipt disagrees with the lyric

`Stay Through the Blue` lists `Hah—look at you` in the Body-Noise Dramaturgy table, while the actual lyric is `Hah—you stayed right here`.

**Repair route:** selected-enhancement package receipt only. Make the table quote the actual lyric or deliberately revise both together; do not leave the sidecar claiming a phrase the renderer will not receive.

### 6. P05 source pointer is internally inconsistent

`concept_medium_pairs.json` labels the Sarah Davachi interview as `F11`, but `00_research_brief.md` defines F11 as the PDB histone-ubiquitination item. The selected package and `CREDITS.md` contain a direct Davachi/Bandcamp link, so this is a traceability defect, not evidence that the source was fabricated.

**Repair route:** correct the pair-map source pointer or add the interview as its own research-ledger row; never relabel the PDB item silently.

### 7. Judge calibration plate is unproven

No coordinator-built blind candidate + Golden + known-mediocre decoy plates were supplied in this run. This audit therefore does **not** claim the `lofn-qa` golden+decoy calibration gate passed.

**Repair route:** after the canonical rerun/repairs, the coordinator—not this judge—assembles shuffled blind plates, then a fresh judge verifies candidate/Golden/decoy order before final SHIP.

## Pipeline Integrity and Cardinality Evidence

- Deterministic artifact audit: coordinator Steps 00–05 **6/6 present**; per-pair Steps 06, 07, 08, 09, 10 **30/30 present**; selected enhanced artifacts **6/6 present**.
- No-skip editorial spine: Steps 07, 09, and 10 exist separately for every pair. **NON-CANONICAL no-skip rule did not fire.**
- Step validators: coordinator Steps 00–05 **6/6 PASS**; raw Step-10 validators **6/6 PASS**.
- Candidate extraction: **24 complete packages** = 6 pairs × 4 variations. All 24 pass the standalone structural Suno validator at 70–120 lines.
- Raw Step-10 candidate ranges: style prompt **916–999 chars**; Suno lyrics field **2,611–3,244 chars**; sung lines **72–78**.
- Cross-pair Step-10 distinctiveness, 240 comparisons: max prompt similarity **0.465 < 0.58**; max lyric similarity **0.302 < 0.42**; max lyric 5-gram Jaccard **0.0043 < 0.18**. No deterministic collision.
- Runtime evidence: lock acquired 00:54:04Z; final selection index written 01:46:49Z, well above the five-minute suspicion threshold.

## ICB Integrity

- Canonical ICB: **91,745 bytes**; SHA-256 `dd408016355b5541a47d2ff5346894d1d7a15d4cd71b57f2eecb52234b1a28f8`.
- `(after ` speaker-tripwire count: **18 == 18**.
- Exact byte-zero ICB prefix: **54/54 canonical prompt/artifact files PASS** — 6 coordinator, 6 isolated pair briefs, 30 pair Steps 06–10, 6 selected-enhanced, and 6 Step-11 dispatch packets.
- Seven small `repair_attempt` notes were excluded because they are validator receipts, not generated step prompts.
- Enhanced canonical generator fields match the six delivery files exactly: style, exclude, lyric field, and title all **24/24 field comparisons PASS**.

The exact-prefix tripwire proves continuity presence, not personality fidelity; the Somatic read below remains decisive.

## Portfolio Balance

- Selected arm balance: **3 ACCESSIBLE / 3 AMBITIOUS**.
- Selected axis balance: **4 EXISTENCE / 2 NEWS** — EXISTENCE ≥3 and NEWS ≤3.
- Emotional duality: P01/P04 carry AWE/wonder with threat; P06 carries Moral Outrage + Defiance + Triumph rather than a forbidden bare `INDIGNATION` tag.
- Selected first-person subject rate: **6/6**. Each lyric contains Sunna's own explicit want and a consequential answering act.
- Club-bed cap: P04 is the sole designated club-derived chassis.

## Structured Evidence Block — Selected Six

`Lyrics chars` reports the conservative deterministic helper count, including markdown fence overhead; the actual paste-field body is 14 characters lower. All remain far below the `<5000` hard cap and `≤4800` target.

| Pair / title | Prompt chars | Lyrics chars | Sung lines | Soundcraft: rhyme / line-return / words-line / allit-100w | Unique-line ratio, chorus-exempt | 4-gram collapse | EMO | Boundary / house / numeric flags |
|---|---:|---:|---:|---|---:|---:|---|---|
| P01 · Stay Through the Blue | 938 | 3,494 | 82 | .476 / .244 / 4.60 / 13.26 | .980 | .0043 | 11 headers; all taxonomy-valid | none / none / 0 |
| P02 · Beat Me to the Line | 978 | 3,971 | 87 | .506 / .264 / 4.70 / 15.40 | .944 | .0333 | 12 headers; all taxonomy-valid | none / none / 0 |
| P03 · Future, Dance Back | 972 | 3,612 | 88 | .545 / .364 / 3.47 / 16.07 | .943 | .0054 | 10 headers; all taxonomy-valid | none / none / 0 |
| P04 · Late Looks Good on You | 978 | 3,640 | 90 | .389 / .233 / 4.19 / 11.94 | 1.000 | .0000 | 12 headers; all taxonomy-valid | none / none / 0 |
| P05 · The Version After This | 956 | 3,941 | 87 | .310 / .322 / 4.32 / 13.03 | .968 | .0112 | 12 headers; **`Comic Nerve` invalid** | none / none / 0 |
| P06 · Don't Put Me in Past Tense | 976 | 3,775 | 77 | .325 / .234 / 4.96 / 12.04 | 1.000 | .0000 | 10 headers; all taxonomy-valid | none / none / 0 |

All six clear the soundcraft floors: rhyme ≥.30, line return ≥.20, words/line ≤7.5, and alliteration ≥11/100 words. Exact chorus return is deliberate and exempt from collapse failure. No house-lexicon phrase or sung numeric fact appears.

## Deterministic Current-Chassis / Seam / Field Checks

- Selected style prompts: dense prose **6/6**, terminal punctuation **6/6**, separate exclude fields **6/6**, at least one SFX cue **6/6**, Theme→SONG FORM opening **6/6**.
- Real artist/person names in positive generator fields or lyrics: **0** across the known source/credit names used by this run.
- Named source-sound break figures in positive style fields (`amen`, `Funky Drummer`, `Think`, `Apache` with word boundaries): **0**.
- Current chassis: **6/6** specify current electronic drum/low-end/vocal/edit/form behavior that survives muting the inherited invariant.
- Now-marker: **6/6** causal and audible on the written spec (answer/object becomes sub, transient, sidechain key, released note, or spectrum/drop restart).
- Swagger gap: **6/6** specify one absent beat, clap, gate, downbeat, chord, or high note.
- Tribute-act falsifier: **6/6** explicitly removes the period ensemble/mix while retaining a complete current record.
- One seam: **6/6** name one substitution junction and keep later widening/return as consequence rather than a second material world.

These are text-spec checks. They do not claim the generator would obey them; only a later render audit can answer that.

## Dedication, Credit, and Human-Subject Gates

### Dedication counts

P01–P06 measure exactly **363 / 303 / 334 / 321 / 344 / 321 characters**, matching the declarations and all ≤500. Each states generated/no-sample honesty. The shared credits file states no rendered audio, sample, stem, real voice, or named performance is contained.

### Human subject

Manual title+lyric review found no real private person, harmed real person, identifying tuple, or minor. The offline `check_human_subjects.py` regex fallback was unusable on these rich markdown packages: it classified section labels such as `Adult`, `Female`, `Lead`, and emotion names as people, treated words such as `body`, `missing`, and `dead stop` as crime/death context, and even parsed `Title at Zero` as a place. Those are concrete detector false positives, not reasonable identifiability suspicions. Manual Gate 16 result remains **CLEAR**.

## Eligibility Classification — Core 7

Scores judge the written package only: Body / Hook / emotional TAM / Specificity / Cognitive Ease / Co-discovery / Sonic Threshold.

| Pair | Scores | Average | Independent classification |
|---|---|---:|---|
| P01 | 5 / 5 / 4 / 4 / 3 / 4 / 3 | 4.00 | ACCESSIBLE |
| P02 | 5 / 5 / 4 / 4 / 3 / 5 / 3 | 4.14 | ACCESSIBLE |
| P03 | 4 / 5 / 4 / 3 / 3 / 5 / 3 | 3.86 | ACCESSIBLE |
| P04 | 4 / 5 / 4 / 3 / 2 / 4 / 2 | 3.43 | AMBITIOUS |
| P05 | 4 / 4 / 4 / 4 / 2 / 4 / 2 | 3.43 | AMBITIOUS |
| P06 | 5 / 4 / 4 / 4 / 2 / 3 / 2 | 3.43 | AMBITIOUS |

This independently reproduces the assigned 3+3 barbell. High lyric counts and immediate/high-energy openings lower the ambitious side's Cognitive Ease/Sonic Threshold; that is classification, not failure.

## Award-Potential Text Scores

Scores: Emotional Precision / Hook Inevitability / Sonic Singularity / Lofn Lineage / Suno Readiness. Package defects reduce readiness but do not erase the creative object.

| Pair | Scores | Average |
|---|---|---:|
| P01 | 4.4 / 4.5 / 4.6 / 4.4 / 4.2 | 4.42 |
| P02 | 4.5 / 4.6 / 4.5 / 4.4 / 3.5 | 4.30 |
| P03 | 4.2 / 4.6 / 4.7 / 4.3 / 3.5 | 4.26 |
| P04 | 4.5 / 4.4 / 4.7 / 4.4 / 4.7 | 4.54 |
| P05 | 4.4 / 4.6 / 4.7 / 4.4 / 3.0 | 4.22 |
| P06 | 4.6 / 4.8 / 4.7 / 4.6 / 3.5 | 4.44 |

No creative package falls below the 4.0 competitive-text average or has a category ≤2. The current REPAIR verdict is caused by canonical process/package evidence, not by a compliant-but-dead set.

## Somatic Gate — Primary 3-Hyper-Skeptic Read

Seats: **Surface Skeptic** (first-listen event), **Corpse-Pattern Skeptic** (arc/return transformation), **Personality Skeptic** (Sunna specificity and run interchangeability). `YES` means distinctive enough on the written artifact. Two NO votes would block.

| Pair | Surface | Corpse pattern | Personality | Bloc |
|---|---|---|---|---|
| P01 | YES — rooftop/arc/promise is immediately visible | YES — partial arcs complete after the promise | YES — fear turns into amused chosen appetite | PASS 3–0 |
| P02 | YES — copy-shop question race is legible | YES — commands become questions and the rival earns the hook | YES — interruption, not admiration, is a specific Sunna want | PASS 3–0 |
| P03 | YES — `it → me` is a clean event | **NO — mechanism-reporting lines briefly explain rather than embody** | YES — future appetite and reciprocity remain unmistakable | PASS 2–1 |
| P04 | YES — four lamps/false arrival/AGAIN reads cleanly | YES — delay changes from threat to chosen game | YES — suspense stays hot, funny, and unsentimental | PASS 3–0 |
| P05 | YES — crossed-out labels and Friday signature are concrete | YES — labels contract into grains and the missing note is paid | **NO — receipt/ruby/printer grammar overlaps the set's house wrapper** | PASS 2–1 |
| P06 | YES — placard flip/read/restart is unmistakable | YES — indignation becomes invitation before exact return | YES — this is Sunna's own refusal and future want | PASS 3–0 |

Required counter-moves for the NO votes:

- **P03:** replace the most arrangement-reporting couplet (`Bass bends up / … Three notes rise / I rise`) with one bodily/social consequence of the returned `me`; keep the one-word seam and hook intact.
- **P05:** rotate non-hook imagery away from the shared ruby/receipt/printer wrapper while preserving the label-crossout, exposed Friday coordinate, and missing-note signature.

Named-corpse audit: no one-note emotional arc, no single section doing all load-bearing work, no untransformed motif, and no repeated-line collapse. Exact choruses are intentional returns. At set level the vote is **2 YES / 1 NO**: events and transformations differ, but the repeated missing-slot→answer→bass/drop grammar is a substantive portfolio flag. If Phase 1 is rerun for the mandatory transform repair, the sideways proposal is to **re-derive at least two pair variation angles so the answering action changes lyric/form without becoming another bass/drop control signal**.

## Per-Song Package Findings

- **P01 — Stay Through the Blue:** creative PASS; numeric, currentness, one-seam, lineage, dedication, and human-subject gates pass. Repair the Body-Noise table mismatch.
- **P02 — Beat Me to the Line:** creative PASS; currentness and equal-rival action are strong. Repair the incomplete living-lineage citations.
- **P03 — Future, Dance Back:** creative PASS with one nonblocking Somatic NO on mechanism-reporting lines. Repair the incomplete living-lineage citations.
- **P04 — Late Looks Good on You:** selected package PASS. The one-use rave chord is bounded, credited as generated, and not load-bearing after the mute test.
- **P05 — The Version After This:** creative PASS; package REPAIR for `Comic Nerve` and source-ledger mismatch. P05's selected theme/form/lineage otherwise pass.
- **P06 — Don't Put Me in Past Tense:** creative PASS; present-tense event and hope payoff are the set's clearest personality proof. Repair the incomplete living-lineage citations.

## External-Action Audit

- Media/render files under the run: **0** (`wav`, `mp3`, `flac`, `m4a`, `mp4`, `mov`, `png`, `jpg`, `jpeg`, `webp`).
- Selected frontmatter: **6/6 `text-only; not rendered`**.
- `CREDITS.md`, `TOP_6_FINAL.md`, preflight, and audio handoff all state no render/upload/publication/spend authority.
- Disk evidence supports **no render, upload, publication, or spend**. No claim is made about audio quality.

## Final Recommendation

**REPAIR.** The six creative objects are viable and the 24-candidate generation/cardinality evidence is real, but the run cannot receive canonical SHIP while the Phase-1 transform gate is absent, run state contradicts disk, and selected package/lineage defects remain. Complete the routed repairs, rebuild from the correct ICB where required, assemble the judge-calibration plates, then run a new blind-first QA. No learning-ledger entry is appended because this is not SHIP; no operational ledger was modified because this audit was authorized to write only the two QA evidence files.
