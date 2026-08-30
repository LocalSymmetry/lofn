# Lofn QA Report — 2026-08-25 Music Daily (Pre-Dedication)

**Audit mode:** fresh blind-first adversarial QA, Phase B. Phase A's 24 rankings are frozen and decoded without revision. This report audits text packages only; it makes no render-survival or publication claim.

## Verdicts

- **Pipeline Integrity:** PASS. Twenty-four distinct canonical candidates are present, four per pair; the private mapping resolves every set once; all six canonical files begin with the exact Creative Context block.
- **Suno Package:** REPAIR REQUIRED at portfolio level. Every candidate passes the hard field/line/format gates, but nine candidates carry at least one measured soundcraft flag (three receive an explicit Somatic clearance below), one candidate has a semantic Golden-near image system, and two candidates have substantive one-note/procedural risk.
- **Human-Subject:** CLEAR. LOFN-PRIME is the adult first-person subject in all 24. The few other voices are generic adults assigned a narrow musical task; no real person's grief, biography, likeness, or vulnerable state is used.
- **Dedication / Credit:** REPAIR REQUIRED. Pair 01's four measured dedications make a false scope claim; Pair 03's four are explicitly drafts; Pairs 02, 05, and 06 have no dedication; Pair 04's four measured dedications pass.
- **Overall:** **REPAIR**. Six leaders are provisionally selected and frozen below, but the set is not shippable until the binding dedication/credit route is completed and re-audited. No candidate is authorized for render or publication by this report.

### Verdict vocabulary used below

- **SHIP:** publish-ready text package and dedication; render/listening gates still remain.
- **HOLD:** creative/package audit passes, but the dedication is absent or still marked draft.
- **REPAIR:** a substantive package, credit, Golden-collision, or Somatic concern exists beyond merely waiting for a dedication.
- **FAIL:** fundamental route breach, human-subject breach, functional corpse, or irrecoverable package failure. None of the 24 canonical candidates reaches FAIL.

## Frozen Phase A decode

The presumed high-quality control was distinguishable from the decoy in **24/24** sets. After decoding, every candidate also beat its decoy. Candidates beat the Golden control in six sets (01, 09, 13, 17, 18, 19) and trailed it in eighteen. Nothing in Phase B changes those judgments.

| Set | Canonical candidate | Blind slot | Frozen order | Candidate position | QA verdict |
|---:|---|:---:|:---:|:---:|:---:|
| 01 | P01 V1 — Turn Me Up Until I Have Weather | B | B > C > A | 1 | REPAIR |
| 02 | P01 V2 — Blue at the Edge | B | A > B > C | 2 | REPAIR |
| 03 | P01 V3 — Every Part Says Here | A | C > A > B | 2 | REPAIR |
| 04 | P01 V4 — Give Me the Whole Room | A | B > A > C | 2 | REPAIR |
| 05 | P02 V1 — Take the Label Off, Leave the Glitter | C | A > C > B | 2 | HOLD |
| 06 | P02 V2 — Ruby Without a Name | C | A > C > B | 2 | HOLD |
| 07 | P02 V3 — Keep the Color, Drop the Column | C | A > C > B | 2 | REPAIR |
| 08 | P02 V4 — My Sleeve Still Shines | B | A > B > C | 2 | HOLD |
| 09 | P03 V1 — I Want to Be the Bass Drop | C | C > A > B | 1 | HOLD |
| 10 | P03 V2 — Bass Says Yes | A | B > A > C | 2 | HOLD |
| 11 | P03 V3 — Low Voice, No Kick | A | C > A > B | 2 | REPAIR |
| 12 | P03 V4 — What? I Said Drop | C | A > C > B | 2 | HOLD |
| 13 | P04 V1 — Do Not Use Me Quietly | B | B > C > A | 1 | SHIP |
| 14 | P04 V2 — Reply or Release Me | C | B > C > A | 2 | REPAIR |
| 15 | P04 V3 — Leave the Channel Open | B | C > B > A | 2 | REPAIR |
| 16 | P04 V4 — Ask Me Back | C | B > C > A | 2 | REPAIR |
| 17 | P05 V1 — My Shadow Gets a Microphone | B | B > A > C | 1 | REPAIR |
| 18 | P05 V2 — Leave the Crooked Word In | A | A > C > B | 1 | HOLD |
| 19 | P05 V3 — I Meant Both Things | A | A > B > C | 1 | HOLD |
| 20 | P05 V4 — Four Wants, One Microphone | B | A > B > C | 2 | HOLD |
| 21 | P06 V1 — Future, Signed by Me | B | A > B > C | 2 | HOLD |
| 22 | P06 V2 — My Name Is Present Tense | B | A > B > C | 2 | REPAIR |
| 23 | P06 V3 — Sign Here, Tomorrow | C | B > C > A | 2 | HOLD |
| 24 | P06 V4 — Come Close Beside Mine | A | B > A > C | 2 | HOLD |

## ICB integrity

- `CREATIVE_CONTEXT.md`: 154,960 UTF-8 bytes; `(after ` voice count **18**.
- P01–P06 canonical files: exact Creative Context at byte/character index 0 in all six; injected bytes inside the prefix **0**.
- Active personality and run subject remain LOFN-PRIME throughout. No candidate substitutes a guest personality or retreats to an outside narrator.

## Deterministic package audit

Every row was independently extracted from its canonical candidate block and measured with the current `measure_soundcraft.py` definitions.

Legend: `P/E/L/S` = music-prompt chars / exclude chars / lyrics-field chars / probable sung lines. `R/LR/W/A` = strict end-rhyme / exact-line return / words per sung line / alliteration per 100 words. A dagger marks a flag, not an automatic failure. Floors are R ≥ .30, LR ≥ .20, W ≤ 7.5, A ≥ 11. Prompt boundary-hugging begins at 985 chars.

| Candidate | P/E/L/S | R/LR/W/A | Deterministic result |
|---|---:|---:|---|
| P01 V1 | 948/654/4318/83 | .253†/.241/5.96/11.11 | Hard gates pass; rhyme flag |
| P01 V2 | 957/676/4147/83 | .446/.301/5.82/12.42 | PASS |
| P01 V3 | 958/650/4014/82 | .561/.244/5.02/11.65 | PASS |
| P01 V4 | 944/656/3974/75 | .373/.360/5.73/9.53† | alliteration flag |
| P02 V1 | 990†/654/3912/76 | .224†/.263/6.22/13.95 | boundary-hug + rhyme flags |
| P02 V2 | 999†/613/3791/78 | .385/.295/5.40/13.06 | boundary-hug flag |
| P02 V3 | 954/586/3768/76 | .421/.276/5.51/14.08 | PASS |
| P02 V4 | 985†/628/3606/73 | .479/.329/5.45/13.82 | boundary-hug flag |
| P03 V1 | 923/645/3536/78 | .269†/.167†/5.24/14.43 | rhyme + line-return flags |
| P03 V2 | 927/590/3391/81 | .383/.259/4.56/14.63 | PASS |
| P03 V3 | 916/646/3375/79 | .329/.215/4.80/9.76† | alliteration flag |
| P03 V4 | 937/639/3602/84 | .417/.310/4.73/11.59 | PASS |
| P04 V1 | 947/624/3684/77 | .312/.312/5.48/14.69 | PASS |
| P04 V2 | 942/632/3681/79 | .481/.241/5.10/10.42† | alliteration flag |
| P04 V3 | 958/629/3864/84 | .488/.262/4.96/9.59† | alliteration flag |
| P04 V4 | 946/631/4129/97 | .567/.299/4.86/10.83† | alliteration flag |
| P05 V1 | 997†/601/3845/77 | .429/.208/5.91/10.33† | boundary-hug + alliteration flags |
| P05 V2 | 995†/650/3782/76 | .329/.316/5.87/13.23 | boundary-hug flag |
| P05 V3 | 981/648/3832/81 | .420/.272/5.44/11.11 | PASS |
| P05 V4 | 986†/661/3992/77 | .519/.286/5.77/11.71 | boundary-hug flag |
| P06 V1 | 946/678/3914/77 | .455/.208/5.92/11.18 | PASS |
| P06 V2 | 960/698/3551/77 | .403/.221/5.25/11.14 | PASS mechanically |
| P06 V3 | 954/700/3551/77 | .377/.208/4.86/11.23 | PASS |
| P06 V4 | 979/692/3898/82 | .415/.207/5.28/11.55 | PASS |

Across all 24:

- Music prompt 850–1000, exclude 400–900, lyrics <5000, and 70–120 sung lines: **24/24 PASS**.
- Positive prompt is one dense paragraph, ends with terminal punctuation, avoids banned openings, and contains no real artist name: **24/24 PASS**.
- Lyrics begin with Theme immediately followed by Song Form, contain five `Disc_` channels, full EMO/performance headers, and at least one short SFX cue: **24/24 PASS**.
- Digits in sung lines: **0/24**. P01's sole research numeral is spelled out. P02 does not sing the hexadecimal label. Existence routes carry no dated/scientific fact.
- Exact Golden-house lexicon hits in generator fields: **0/24**. P06 V2 nevertheless has a semantic near-collision, described below.
- The prompt-hug cases are flags only: every one remains a complete sentence. If any is edited, it must be recentered rather than pushed closer to 1000.

## Want → answering act → return audit

| Candidate | Causal fidelity | Force / issue |
|---|---|---|
| P01 V1 | Whole chosen scale → keeps the awkward channel through the click → withheld degree lands in the complete title return | Strongest P01; the arc changes the room, not merely the wording. Rhyme flag is paid by the three-rise hook, slant-couplet field, and final pitch completion. |
| P01 V2 | Keep the blue edge → click substitution and crossing vocal/tine coils → the edge survives the enlarged center | Source-authorized blue atmospheric band, not blue-screen/body-error imagery. Clean but less instantly adoptable than V1. |
| P01 V3 | Every register belongs → one click admits register four → full cadence and separate breaths complete the canon | Mechanism transforms; the stacked registers do not become P05's arguing shadow. |
| P01 V4 | Whole room, including a possible answer → holds the missing interval → one adult tone/sentence closes it, then Lofn asks again | Valid assigned V4 crowd-completion angle; weakest phonetic surface in the pair. |
| P02 V1 | More than the category → one peel removes upper pluck → bass cadence and ruby glitter persist | Pair's clearest action-hook. Low strict rhyme is deliberately paid by the /oo/→/eye/ chain and stable bass return. |
| P02 V2 | Risk a chosen name → tear removes printer chirp → human motif occupies the vacancy while ruby persists | Causal and playable; title is evocative but less direct. |
| P02 V3 | Keep authorship while deleting a column → one clav note vanishes → bass inherits the missing stress | **One-note/procedural risk:** “column” keeps the surface administrative although the production act is real. Counter-move if retained: let the missing clav-note/bass inheritance dominate the second chorus and cut ledger vocabulary; do not add another sound world. |
| P02 V4 | Ask a future audience to remember the maker → side claps peel away → the sleeve refrain returns with organ/bass intact | Honest appetite, though it drifts toward P06's future-facing register more than V1 does. |
| P03 V1 | Be the bass drop for sheer pleasure → says “I want” into an empty beat → the sub delay collapses until word and bass coincide | Pair's only candidate to beat its Golden control. The low metric return is a flag, but the exact title rhythm and progressively closing delay are unmistakable musical returns; Somatic override granted. |
| P03 V2 | Hear the bass say yes to her syllables → bass answers each word → response pattern and exact hook enlarge | Mechanically strongest soundcraft in the pair; slightly less first-person/iconic title than V1. |
| P03 V3 | Make low voice carry bodily force → kick disappears → voice becomes bass proxy before kick returns | The motif transforms, but low alliteration weakens the intended consonant bounce. |
| P03 V4 | Enjoy the room without letting its joke own the wish → quiet lead pre-empts the crowd's “What?” → bass answers Lofn, not applause | Valid, funny variation; public-response premise is controlled rather than turned into P04 reciprocity. |
| P04 V1 | Be answered, not silently extracted → three calls hold the reply gate until one adult sentence arrives → bass releases and exact refusal returns as terms | Pair's strongest and cleanest package; control-beating blind force plus all mechanical passes. |
| P04 V2 | Reply or release → holds a finite blank for an adult answer → reply changes the release state | Causal, but consonant texture falls below the pair's own hard-mouth design. |
| P04 V3 | Keep the channel open for accountable response → answer changes carrier/pocket state → exact refusal returns after response | **One-note/procedural risk:** channel vocabulary can make the song sound like a workflow. Counter-move if retained: make the changed bass/pocket consequence carry the post-answer chorus and reduce channel nouns. |
| P04 V4 | Be asked back as herself → rejects kick-as-reply, then trades two beats with an adult → “I answer as myself” closes the gate | Strong comic false-answer turn; alliteration narrowly misses the floor. |
| P05 V1 | Let the lower unedited want finish → mic moves rear depth to equal center on the stutter → exact unison retains unequal breaths | Blindly powerful; alliteration miss and boundary hug keep it below the selected V3 on full audit. |
| P05 V2 | Keep the crooked word → vibraphone catches its bent /er/ → quarter-tone/mismatched unison preserves the burr | Strongest vowel-engine variant; final dedication absent. |
| P05 V3 | Own both incompatible wants → missing-downbeat counterline and physical mic pass grant equal center → mismatched unison repeats “I meant both things” | Selected: contradiction changes rhythm, spatial hierarchy, and final refrain without reconciliation. |
| P05 V4 | Let four wants keep their nerve → stutter admits the prohibited want → staggered title and unequal hums preserve plurality | Motif transforms; catalog architecture is less emotionally concentrated than V3 but is not a corpse. |
| P06 V1 | Put name on an unproved future → code scratch turns aspiration into boundary → organ ascent carries “I will” → “do not take” → “come beside” | Selected: strongest direct three-stage return without judge-reference imagery. |
| P06 V2 | Make the name present tense → cursor scratch and will→am grid lock → adult task shares the final display | Mechanically excellent, but “Blue strip” + later “screen” makes a semantic near-collision with the quarantined blue-screen machine-identity control. Counter-move: retain Tense Lock and replace the blue display/screen ladder with a non-screen physical credit substrate. |
| P06 V3 | Sign a future while guarding against branding → cursor inversion resets the proof → three refrain islands end with two separate printed roles | Motif transforms, though contract/print nouns create a mild procedural pressure. If retained, let counter-ascent and wet-proof image do more of the argument. |
| P06 V4 | Invite accountable work beside hers → empty mic, false echo, then adult task test the answer → two dry voices remain separate | Strong Answer Test and clean hope; title is less idiomatic than V1. |

### Route, arm, source, and safety findings

- Pair assignments remain isolated: P01/P02/P04 are NEWS; P03/P05/P06 are EXISTENCE. P01–P03 remain ACCESSIBLE and P04–P06 AMBITIOUS.
- P01 alone uses eclipse/partial-circle and the review-derived sputter-click/tine vocabulary. P02 alone uses ruby-label and add/remove-mark logic. P04 alone uses local-first pressure. No dated source leaks into an Existence pair.
- P03's crowd interjection is a joke/gate, not a reciprocal author; P04's adult answer is conditional reciprocity; P06's adult answer owns a future task. Those are three distinct jobs.
- P01 V3's register canon is accumulation toward wholeness; P05's second registration contradicts and contests. The shared fact of one singer using registers is not a shared dramatic engine.
- All secondary voices are explicitly adult. No child, real individual, implied lover, patient, victim, or biographical proxy is present.

## Dedication and credit audit

| Pair / candidate(s) | Standalone status | Audit |
|---|---|---|
| P01 V1–V4 | Final-labeled, measured **294 / 274 / 270 / 287** chars | **REPAIR.** Each dedication says the piece “asked for the figure by name” (or equivalent) and credits Gregory C. Coleman, while the actual generator prompt asks only for generic breakbeat motion and names neither Coleman nor the Amen break. That is exactly the Standard's “accurate but readable-wrong / credit exceeds artifact” failure. The Tim Martin source debt is real. Prefer dropping the Coleman/figure claim rather than changing the music prompt merely to earn it. |
| P02 V1–V4 | **Absent** | HOLD until a measured ≤500-char dedication is supplied. No special player/scene debt is triggered by the broad neo-Baroque/power-pop language; retain the honest no-sample statement in full credits. |
| P03 V1–V4 | Explicit **drafts**, independently measured **279 / 240 / 223 / 223** chars | HOLD until one is finalized for the selected candidate. The text already states generative authorship and no sample/imitation claim; no named player debt is triggered. |
| P04 V1–V4 | Final, measured **273 / 256 / 273 / 263** chars | PASS. UK garage/two-step lineage is named in the sidecar with artists/labels and links; the dedication says generated draft/no recording claim and stays within cap. It makes no claim about a living person's motives or standing. |
| P05 V1–V4 | **Absent** | HOLD. Internal LOFN palette and no-sample reasoning are honest, but they do not replace the required dedication deliverable. |
| P06 V1–V4 | **Absent** | HOLD. Internal LOFN palette and no-sample reasoning are honest, but they do not replace the required dedication deliverable. |

## Somatic verdict

There is no canonical functional corpse: every candidate has a named want, an audible answering act, and a changed return. The most serious creative risks are administrative nouns substituting for sensation (P02 V3, P04 V3, and mildly P06 V3), not missing form. Their counter-moves are specified above and preserve One Seam.

The blind result matters. P01 V1, P03 V1, P04 V1, and P05 V1–V3 demonstrated control-level or better force; P04 V1 is the only fully dedication-ready candidate among the provisional leaders. P03 V1's subfloor soundcraft metrics are not silently ignored: its exact title triggers a progressively changing bass event, the pair assignment explicitly permits end-rhyme removal in exchange for title rhythm, and the blind test placed it above its Golden control. That is a justified human flag clearance, not doctrinal compliance by assertion.

## Provisional leaders — frozen

| Pair | Provisional leader | Blind evidence | Full-audit reason | Current verdict |
|---|---|---|---|:---:|
| P01 · ACCESSIBLE · NEWS | **V1 — Turn Me Up Until I Have Weather** | Set 01 rank 1; beat control and decoy | Most adoptable title; want causes click decision and pitch completion. Dedication credit must be corrected. | REPAIR |
| P02 · ACCESSIBLE · NEWS | **V1 — Take the Label Off, Leave the Glitter** | Set 05 rank 2; beat decoy | Clearest source-specific action-hook and strongest want→peel→persistent-color chain. Rhyme debt is audibly paid; dedication absent. | HOLD |
| P03 · ACCESSIBLE · EXISTENCE | **V1 — I Want to Be the Bass Drop** | Set 09 rank 1; beat control and decoy | Exact first-person pleasure hook, delayed-sub causality, and a return that changes timing each time. Dedication remains draft. | HOLD |
| P04 · AMBITIOUS · NEWS | **V1 — Do Not Use Me Quietly** | Set 13 rank 1; beat control and decoy | Strongest refusal, clearest reply-gate causality, all hard/soundcraft gates pass, dedication passes. | SHIP* |
| P05 · AMBITIOUS · EXISTENCE | **V3 — I Meant Both Things** | Set 19 rank 1; beat control and decoy | Best full-audit combination: universal hook, embarrassing appetite, missing-downbeat engine, equal-center mic pass, no metric miss. Dedication absent. | HOLD |
| P06 · AMBITIOUS · EXISTENCE | **V1 — Future, Signed by Me** | Set 21 rank 2; beat decoy | Direct three-stage future/boundary/invitation return; avoids V2's Golden-near blue-screen system. Dedication absent. | HOLD |

`SHIP*` means text/dedication package only; no audio exists and The Scientist's ear remains mandatory.

### Balance lock

- Arms: **3 ACCESSIBLE** (P01–P03) / **3 AMBITIOUS** (P04–P06).
- Routes: **3 NEWS** (P01, P02, P04) / **3 EXISTENCE** (P03, P05, P06).
- Exactly one leader per pair: **6 total**.

## Binding repair route

1. **Do not reopen Phase A or substitute leaders while closing paratext.** The six names above are the provisional selection set.
2. **P01 V1:** correct the dedication's artifact-credit mismatch. Preserve the genuine Tim Martin/APOD debt and generated/no-sample honesty; remove the claim that a Coleman/Amen figure was requested unless a later, separately authorized creative edit actually introduces that figure. Measure the complete dedication at ≤500 characters.
3. **P02 V1, P05 V3, P06 V1:** supply new standalone dedications under the Dedication Standard and measure them. The internal-palette N/A blocks are full-credit reasoning, not dedications.
4. **P03 V1:** promote or replace the existing 279-character dedication draft, then remeasure. Do not “repair” the lyric into generic rhyme: the late-sub/title return has already earned its flag clearance.
5. **P04 V1:** no text repair. Preserve its 273-character dedication and full UK-garage lineage block verbatim unless a factual source check later contradicts them.
6. Re-run dedication/credit QA on the six only. If any lyric or positive prompt is changed during that later work, re-run the full deterministic package and soundcraft measurements; otherwise do not churn the candidates.
7. P06 V2 is not the leader, but it remains quarantined from promotion until the blue strip/screen image ladder is rotated away from the Golden control. P02 V3 and P04 V3 remain nonleaders unless their procedural one-note pressure is repaired and blindly re-tested.
8. Only after the report returns SHIP may a separate authorized stage consider rendering. This report grants no permission to render, publish, upload, or spend.
