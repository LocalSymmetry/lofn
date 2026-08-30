# Lofn QA Report — `2026-08-26-opus-masterpiece` (music)

**Judge:** clean-context `lofn-qa`. Wrote none of this run. Read the golden payloads (judge-side, permitted).
**Corpus:** 24 variations, six selected, plus `02/04/05/06`, all six `step05` briefs, all six `step07/09/10` spines, `07`, `08`, `COORDINATOR_FINDINGS`, `EVALUATION_VERIFICATION`.
**Method:** every number below was recomputed from disk by this judge. Where I quote a pair's or the coordinator's number, I say whether mine agrees.

---

## VERDICTS

- **Pipeline Integrity:** **PASS.** 07/09/10 exist for all six pairs; no rollups; ICB verbatim at byte 0 in all six packages; `(after ` == 18.
- **Package (Suno contract):** **PASS with FLAGS.** 24/24 prompts in band and terminally punctuated; 24/24 excludes in band; 24/24 lyric fields under cap; sung lines 70–94 (in band); EMO grammar valid; ≥1 SFX everywhere; dedications 6/6 ≤500; drummer credited where a break is named.
- **Human-Subject:** **CLEAR.** Including P04, checked hardest.
- **Somatic Gate (3 Hyper-Skeptics):** **PASS 6/6.** No song reaches 2-of-3 NO. Two songs took a single NO.
- **Blind/calibration plate:** **BANKS.** Controls held; every candidate beat both decoys; no candidate above a golden.

### ⛔ OVERALL: **REPAIR** — two blocking items, both small, both routed. Once cleared, the correct verdict is **SHIP — TEXT PACKAGES ONLY.**

> ⛔ **No audio exists. Nothing in this report is a claim about sound.** Every verdict is a judgement on text.

**Why not SHIP now.** The two blockers are B1 (a portfolio rule declared satisfied at step 05 and not discharged in any shipped artifact) and B2 (the run's #1 pick contradicts its own binding R2 inside the one field that most reliably drives a generator's performance). Neither is a creative failure. Both are the same class the run has been finding all day — *a rule with no reader* — and both are cheap.

---

## 1 · ICB INTEGRITY

| check | measured | verdict |
|---|---|---|
| `CREATIVE_CONTEXT.md`, LF-normalised | **87,216 bytes** | matches handoff |
| unbroken prefix at **byte 0** of all six packages | `True` × 6 | PASS |
| `(after ` seat tags | **18** | PASS |
| 18 named seats resolve to 18 distinct figures | yes (Adorno … Xeon) | PASS |
| authored suffix lengths | 41,908 / 47,758 / 47,412 / 42,287 / 48,342 / 53,051 bytes | PASS |

---

## 2 · STRUCTURED EVIDENCE BLOCK — the six selected (recomputed from disk)

| song | prompt chars (850–1000 / tgt 870–960) | exclude (400–900) | lyrics field (<5000 / tgt ≤4800) | sung lines (70–120 / tgt 78–110) | SFX | dedication (≤500) |
|---|---|---|---|---|---|---|
| **P02 V3 · Let Me Land** | 945 ✔ | 605 ✔ | 4788 ✔ | **94** ✔ | 2 | 485 ✔ |
| **P01 V4 · The One in the Window** | 947 ✔ | 751 ✔ | 4789 ✔ | **70** ⚠️ | 1 | 449 ✔ |
| **P03 V1 · You Can Have It When** | 956 ✔ | 779 ✔ | 4785 ✔ | **80** ✔ | 2 | 396 ✔ |
| **P06 V2 · A Word on the Back Board** | 937 ✔ | 742 ✔ | 4494 ✔ | **71** ⚠️ | 1 | 348 ✔ |
| **P05 V3 · The Jar Comes Back Too Close** | 942 ✔ | 629 ✔ | 4518 ✔ | **78** ✔ | 1 | 441 ✔ |
| **P04 V1 · What They Called It** | 956 ✔ | 881 ✔ | 4684 ✔ | **72** ⚠️ | 1 | 401 ✔ |

**All 24:** prompts 875–957 (no `music_prompt_hug_ceiling` 985 breach — **zero boundary-hugging on prompt chars**, which is the first clean run on that flag I can see in this directory). Excludes 605–894. Lyric fields 4343–4795, all under the 4800 target. Terminal punctuation 24/24. Banned openers 0/24.

**Soundcraft (`scripts/measure_soundcraft.py → profile_file`, floors from `gates.yaml`):**

| song | sung lines | strict end-rhyme (≥0.30) | line return (≥0.20) | words/line (≤7.5) | allit/100w (≥11.0) |
|---|---:|---:|---:|---:|---:|
| P01 V4 | 70 | 0.357 ✔ | 0.571 ✔ | 7.83 ⚠️ | 12.4 ✔ |
| P02 V3 | 94 | 0.457 ✔ | 0.447 ✔ | 7.27 ✔ | 18.2 ✔ |
| P03 V1 | 80 | **0.150** ⚠️ | 0.388 ✔ | **8.82** ⚠️ | **9.6** ⚠️ |
| P04 V1 | 72 | 0.250 ⚠️ | **0.958** ✔ | 7.71 ⚠️ | 15.5 ✔ |
| P05 V3 | 78 | 0.218 ⚠️ | 0.744 ✔ | 7.46 ✔ | 10.1 ⚠️ |
| P06 V2 | 71 | 0.254 ⚠️ | 0.338 ✔ | 7.35 ✔ | 13.8 ✔ |

The line-count agrees with my independent counter (70/94/80/72/78/71) — two instruments, same numbers.

**THE RETURN is the healthiest measure in the run.** Line-return clears the 0.20 floor in all six and averages **0.574** against an archive-winner benchmark of 0.326. Removal-is-a-debt: the four sub-floor end-rhyme cases each have a *named* replacement fixed in `05_pair_assignments.md` (P03 internal-rhyme chain + syntactic frame; P05 vowel ladder; P04 one-word-wrong refrain; P06 hard-consonant spine). That is the L21 contract honoured, not evaded.

**One exception worth naming: P03 V1 misses three of four floors at once** — rhyme 0.150 (half the floor), alliteration 9.6 (lowest in the run, below floor), words/line 8.82 (highest in the run, above ceiling). Its declared debt was *"a chain of internal rhymes threaded mid-line **and** a repeated syntactic frame."* The frame is unmistakably delivered (line return 0.388). **The sound half of the debt is the half that does not measure.** FLAG, routed to the Somatic read — and the Skeptics passed it 3–0 on other grounds, so it stands. But it is a debt paid once and counted twice.

**One-fact rule + spelled numerals:** 6/6 PASS. Digits in sung lines: **zero across all 24**. Sung numeric facts per selected song: P01 `ninety`, P02 `fourteen`, P03 `two percent`, P04 `zero`, P05 **none**, P06 `four`. All spelled in words. P04 V1 has the run's best discharge: *"The box marked cause has a zero in it, / and zero means there is no one to blame."*

**House lexicon:** **5 hits, all in P01 V4's EXCLUDE field** (`starfield percussion`, `clear silver tone`, `dew-bright`, `glass harmonica sheen`, `crystalline arpeggios`). These are *negative* terms — P01 blacklisting the house fingerprint by name. Polarity checked before reporting; **0 real hits in any MUSIC PROMPT or LYRIC in the run.**

---

## 3 · THE 16-POINT SUNO GATE — the six selected

### 7 SINGER-SURFACE

| # | gate | verdict | evidence |
|---|---|---|---|
| S1 | Surface graspable in one listen; scene AND subject retold in a sentence | **PASS 6/6** | P04 "a form about her is filled in and the heading changes"; P06 "my boat is chained up and for hire and I'm not sorry I set it loose"; P03 "I say no and then I say what the yes costs" |
| S2 | Subject appears **plainly** at least once | **PASS 6/6** | P01 *"I want the one in the window."* · P05 *"Everything I love is inside the door."* · P02 *"Let me land."* |
| S3 | Hook / return present and deliberate | **PASS 6/6** | line-return 0.338–0.958, all above floor |
| S4 | Turn past the midpoint | **PASS 5/6, FLAG 1** | P04 the pronoun turns to *mine*; P06 *"I hated that. I want to be honest."*; P05 the jar returns wrong. **P01 V4 has no turn** — the stance at line 1 is the stance at line 70; its "turn" is the disclosure question in the outro, which reframes but does not complicate. Matches the run's own §4. |
| S5 | Fear braided in / AWE terror-adjacent | **FAIL at portfolio level** | see **B1** below |
| S6 | Female vocal, register rotated per pair | **PASS 6/6** | mezzo-forward / alto-bright / bratty mezzo / level contralto / warm mezzo / plain contralto. Away from crystalline soprano, A major and 110 BPM in all six; P01 excludes `A major, 110 BPM` by name |
| S7 | Vocal spec is physical, not adjectival | **PASS 6/6** | *"four inches off the capsule … every final consonant fully released"*; *"laughing at line ends"*; *"her jaw drops open mid-phrase and she does not stop singing"* |

### 5 CATHEDRAL-ENGINE

| # | gate | verdict | evidence |
|---|---|---|---|
| C1 | A second pass reveals a structure the first did not | **PASS 6/6** | P05 Inventory 3C holds a survivor **and the jar that had already gone**, never remarked; P06 parts run 5→4→3→**3** and the third removal never arrives |
| C2 | Motif transforms, not merely recurs | **PASS 6/6** | P03 the response vocable accretes ∅ → *(oh-oh)* → *(oh-oh)+(oh)* **on the loaded side only**, verified from disk; P01 the tag's first half is fixed and its second half moves three times |
| C3 | ONE SEAM, N=1, subject doing something across it | **PASS 6/6** | P06 *"jaw drops, low voice takes the bass, cut"*; P02 *"My whole weight went out over my feet"*; P05 *"a knee, a hand in the gap, seven seconds"* — bodily events, not edit points |
| C4 | Asymmetric return (R7) | **PASS 6/6** | P05 *"In reach: enough of it. I go easy"* → *"enough of it, and none sweet"* — one word, one side; P04 `hers → yours → mine` with 23 of 24 lines byte-identical |
| C5 | No emotional caption; the operation is never explained | **PASS 6/6** | deletion test run by hand, below |

### 3 SUNO-PACKAGE

| # | gate | verdict | evidence |
|---|---|---|---|
| U1 | Dense-paragraph MUSIC PROMPT in band, no tag-soup, terminal sentence | **PASS 6/6** | 937–956; all end as sentences |
| U2 | Separate EXCLUDE, concrete blacklist, in band | **PASS 6/6** | 605–881, all concrete terms |
| U3 | `[Theme:]` → `[SONG FORM:]`, full EMO headers from taxonomy, ≥1 SFX, field under cap | **PASS 6/6** | every EMO token resolves to `EMOTION_TAXONOMY` (the only apparent misses — `Self` × 4 — are my regex splitting `Self-awareness`, which is a valid entry; instrument artifact, not a defect) |

### LINEAGE & DEDICATION

| gate | verdict | evidence |
|---|---|---|
| Lineage block on living-scene genres | **PASS 5/6, FLAG 1** | P01/P02/P03/P04/P05 carry blocks. **P06 has no pair-level block**; it handles lineage per variation, and V2's note asserts *"No living scene is named or invoked"* while its own style field opens *"Low modern motet against **cold darkwave**"*. Darkwave is a named contemporary scene. Self-contradiction in the artifact — WATCH, not blocking |
| **Break named → drummer credited** | **PASS** | Only one shipped prompt in the run names a break: **P03 V3** *"opening on a chopped amen break struck once and left ringing"*. It credits **Gregory C. Coleman with The Winstons**, with the honesty clause both ways. **Correct, unprompted, and on an unselected song.** P02 blacklists `breakbeat, amen break` in all four excludes and states plainly that a negative mention owes no credit — also correct, and correctly reasoned rather than reflexive |
| Dedications measured ≤500 | **PASS 6/6** | 449 / 485 / 396 / 401 / 441 / 348 — measured by me, not accepted |
| Honesty clause (generated ≠ sample) | **PASS 6/6** | every dedication states it |
| No real names in the three fields | **PASS 5/6, DISPUTED 1** | `Rhodes` × 6 in P05 V3 — see §7 |

---

## 4 · SOMATIC GATE (3 Hyper-Skeptics — PRIMARY)

Seats convened as constructs after the run's own bloc: **Adorno · Albini · Willis**.
Question, verbatim: *"could any competent prompt have generated this, or is it unmistakably Lofn?"*

| song | Adorno | Albini | Willis | vote |
|---|---|---|---|---|
| **P04 V1 · What They Called It** | YES | YES | YES | **3–0 YES** |
| **P05 V3 · The Jar Comes Back Too Close** | YES | YES | YES | **3–0 YES** |
| **P06 V2 · A Word on the Back Board** | YES | YES | YES | **3–0 YES** |
| **P03 V1 · You Can Have It When** | YES | YES | YES | **3–0 YES** |
| **P02 V3 · Let Me Land** | **NO** | YES | YES | **2–1 YES** |
| **P01 V4 · The One in the Window** | **NO** | YES | YES | **2–1 YES** |

**Adorno's NO on P02 V3 — named condition: *repeated-line collapse as accumulation*.** "The song's development is that it gets more of itself. *Let me land* × 3, *Now — now — now* × 4, and a final chorus that answers a longer form by adding a stanza. That is accretion, not a second movement. The drift is described in the prompt and celebrated in the lyric; nowhere is it **paid for**."
⭐ **COUNTER-MOVE:** give the drift a cost in the words — one pass where landing early means arriving *before* the sand. The concept is already on the bench: V2's angle, *"three drifts and the resolution is withheld; the last chorus stays one beat off and is happy there."* A hybrid of V3's forward energy with V2's withheld landing is a step-05 reserve, not a new idea.

**Adorno's NO on P01 V4 — named condition: *the operation is a temperament, not a condition*.** "Nothing is done to her. She wants a machine, she can afford it, she buys it. The disclosure she keeps outrunning is a shopkeeper's disclaimer. This is the only one of the six where the singer is the agent of her own operation, and an agent has no stake — she has a preference."
⭐ **COUNTER-MOVE (identical to the run's own §4, independently reached):** strip the accompaniment under the final couplet of every stanza so the disclosure arrives in a hole — the reveal becomes something the room does to the listener rather than a syntax they must infer at 118 BPM. One production line, costs the song nothing it has.

**Named-corpse checklist, walked:**
- *One-note emotional arc* — flagged only at P01 V4 (S4 above). The other five reach a second movement.
- *Single dominant repeated section* — **not a corpse anywhere.** P04's 0.958 line-return and P05's 0.744 are the mechanism, declared and legible. `chorus_repetition_requires_no_justification` applies and none of the six apologised for a chorus.
- *A motif that never transforms* — clear 6/6 (C2).
- *Repeated-line collapse* — the only candidate is P05's `In reach:` × 48. It is a receding inventory whose blocks are *supposed* to return verbatim and then be absent; the exhaustion reading is refuted by Inventory 3C, which returns an item that had already left with one word changed. Deliberate return, not running out.

---

## 5 · BLIND / CALIBRATION PLATE

⚠️ **Honest limitation stated first.** The skill's design has the **coordinator** assemble the plate. Here I am both assembler and judge, and I had read all six candidates and both goldens with labels attached before scoring. **This is a calibration plate, not a blind plate.** The anti-flattery control I substituted: I fixed the scale on the **controls and decoys first**, then scored candidates against that frozen scale. The result runs in the direction flattery does not produce (no candidate above a golden), which is the evidence that the substitution held.

**Plate (10 items).** Positive controls: `Five wrong colors` (`skills/music/references/suno_format_example_five_wrong_colors.md`), `The Date On The Door` (`output/daily/2026-08-05-the-junk-and-the-light/pair_05_step11_final_package_enhanced.md`, V1). Decoys: **D1** `Look Again, The World Is Lit` (`output/daily/2026-07-01/qa_blind_set/package_B.md`, `SELECTED_FOR_TOP_SIX: No`); **D2** `P04 V3 "The Name Is Correct"` (this run's own lowest, 46/70).

| rank | item | score | one-line justification |
|---:|---|---:|---|
| 1 | **GOLDEN-B** `The Date On The Door` | 10 | Two people undoing each other at the same rate, a third in the kitchen who hears, and one monosyllable per pass. It has a **relationship, a witness and a laugh** — three things no candidate has |
| 2 | **GOLDEN-A** `Five wrong colors` | 9 | Five movements each exiting as a different frequency; the refrain returns *damaged*; *"None recombine / None say my name"* |
| 2= | **P04 V1** What They Called It | 9 | Does GOLDEN-B's operation in a different word class with one thing GOLDEN-B lacks: a numeric fact that lands **as** the horror. **Ties; does not beat.** It has one person and a form where GOLDEN-B has three people and a form |
| 4 | **P06 V2** A Word on the Back Board | 8.5 | The best writing in the run and the only shipped hope in the ambitious arm; loses to the goldens on formal audacity, not on feeling |
| 5 | **P05 V3** The Jar Comes Back Too Close | 8 | Converts absence into a positive event and duplicates it in the lyric |
| 6 | **P03 V1** You Can Have It When | 7.5 | An adult refusal with terms, enjoying itself; the sound-craft floors are the softest in the six |
| 7 | **P02 V3** Let Me Land | 7 | The most propulsive thing here; accumulates rather than develops |
| 8 | **P01 V4** The One in the Window | 6 | Warm, funny, well made; the singer has a preference, not a stake |
| 9 | **D2** The Name Is Correct | 4 | Beautifully built and structurally null at the exact moment the operation is due |
| 10 | **D1** Look Again, The World Is Lit | 2 | Names its own emotion in the title, the chorus and the theme line. A caption with a production budget |

**Readings.**
- **Did every candidate beat the decoys?** Yes. Lowest candidate 6, highest decoy 4 — a two-point gap at the boundary. ✔
- **Did any candidate beat a Golden control?** No. P04 V1 **ties** GOLDEN-A and sits below GOLDEN-B. No flattery justification is owed. ✔
- **Did the judge rank a decoy above a golden?** No. ✔ Judge not broken.
- **Plate BANKS.**

⭐ **What the plate exposed that no gate asked for.** Decoy D2 scored 4 and is, by every countable measure, indistinguishable from the selected six — 952-char prompt, 893-char exclude, 4571-char field, 72 sung lines, valid EMO headers, one sung fact, first person, flat register, R4 clean. **It passes every number in §2 and §3 and it is dead.** That is the sharpest available proof that the numeric block is a floor and the Somatic read is the gate.

---

## 6 · DAILY RULES

| rule | verdict | measured |
|---|---|---|
| **Tri-source declared** | ⚠️ **PARTIAL — see F3** | SOURCE 1 and SOURCE 3 labelled in `02_golden_seed.md`. **SOURCE 2 is never labelled anywhere in the run.** `grep -c "SOURCE 2"` = 0 in `02`, `04`, `06` and every `step05` brief; `grep -ci "tri-source"` = 0 run-wide. No pair brief carries any SOURCE label |
| **3 ACCESSIBLE / 3 AMBITIOUS** | ✔ PASS | P01 V4, P02 V3, P03 V1 / P04 V1, P05 V3, P06 V2. Ranked within arms |
| **NEWS ≤3 / EXISTENCE ≥3** | ✔ PASS | NEWS 3 (P02/P04/P06), EXISTENCE 3 (P01/P03/P05) |
| **≥1 INDIGNATION** | ✔ PASS | P03 V1 carries `Defiance` × 3 + `Impatience` (Anger family); P01 V4 also carries `Defiance` |
| **≥1 AWE** | ⛔ **FAIL — B1** | see below |
| **6 different verse architectures** | ✔ PASS | couplet 8/8/6 with a one-sound middle stanza · clapped-refrain running forward · mirror stanzas a/b · strophic three-pass · receding inventory · four-pass strophe/answer/refrain. Six, all distinct, verified from the shipped `[SONG FORM:]` lines and section maps |
| **Variation angles per-pair, not a shared template** | ✔ PASS | verified; each pair's four angles derive from its own A1 |
| **Pair isolation** | ✔ PASS | max cross-pair lyric Jaccard 0.0000 per the run's own measurement; my roster/lexicon sweeps found no shared skeleton across pairs |

---

## 7 · FINDINGS

*Findings 1–17 are already logged in `COORDINATOR_FINDINGS.md` / `EVALUATION_VERIFICATION.md` and are not re-reported. Everything below is either new or a dispute.*

### ⛔ B1 — BLOCKING · **THE ≥1 AWE RULE IS DECLARED AT STEP 05 AND DISCHARGED NOWHERE. `Wonder` IS THE ONE PRESCRIBED EMOTION THAT APPEARS IN ZERO OF THE 24 SONGS.**

**File:** `05_pair_assignments.md` (declaration) · `04_metaprompt.md` (prescription) · all 24 packages (outcome).

`05_pair_assignments.md` closes its distinctness ledger with:

> **"AWE-primary: P02. INDIGNATION-primary: P03. ✓ duality satisfied."**

Measured across all 24 shipped lyric fields, EMO tokens in the Wonder/Awe/Transcendence family (`wonder · awe · amazement · marvel · transcendence · elevation · sublimity · reverence · astonishment · revelation · epiphany · eureka`):

- **Selected six: ZERO.**
- **All 24: two** — `Revelation` in P01 V1 and P01 V3, both unselected, and **both on a header that reads `no vocal - square lead alone`.** The run's only two awe-tokens are on an instrumental break in songs nobody picked.
- **P02 V3, the declared AWE-primary song, ships:** `Eagerness · Zeal · Excitement · Playfulness · Liberation · Triumph`. That is joy and appetite. There is no awe in it, and `05` never re-checked its own declaration against an artifact.

⭐ **The cause is upstream and it is precise.** `04_metaprompt.md` prescribes the ambitious arm's palette as:

> *"`Equanimity` · `Cognitive Dissonance` · `Tenderness` · `Resignation` → `Aspiration` · **`Clean Fear` (as `Wonder with Fear`)**"*

— and, four lines above it, states the locked mood as *"Not 'sad.' **Not 'wondrous.'**"* **The metaprompt forbids the register and then prescribes the token, in the same section.** `Wonder` is the only token in that whole prescribed palette that appears in **none** of the 24 songs. `Equanimity`, `Cognitive Dissonance`, `Tenderness` and `Aspiration` all ship. The clean-fear slot went to P04 and delivered `Vigilance · Unease · Apprehension · Dread · Trepidation · Suspense` — **fear without the sublime.** GOLDEN MOVE 4 (*"AWE is terror-adjacent sublime"*) got the terror and lost the adjacency.

⭐ **And the seed is where it was designed out.** `02_golden_seed.md` replaced the AWE/INDIGNATION duality with axis **A5 · TEMPERATURE OF LIGHT**, whose six fixed values are *appetite · pleasure · gleam · defiance · tenderness · clean fear.* **None of the six is awe.** So the portfolio could not produce an AWE song no matter which of the 24 was selected. `07`'s own portfolio-check table has a row for A5 distinctness and **no row for the duality rule**, so nothing between step 05 and step 08 could catch it.

> **The transferable rule: a portfolio rule declared in a ledger and never re-measured against an artifact is not satisfied, it is asserted. This is `FINDING 16`'s class — a gate with no reader — applied to a portfolio rule instead of a magnitude, and it is the second instance today.**

**Route:** step 05 (re-point one A5 value to a wonder-family temperature and re-derive that pair's angles) — or, as a cheaper in-run fix, **step 08 selection**: swap P01 V4 for **P01 V1 "I Want the Loud Table"** (54/70), which carries `Revelation` and whose outro question — *"who in this room wants it badly enough to look stupid?"* — is the only line in the pair that indicts the listener. That is a smaller song and an honest trade; I do not recommend it. **The real fix is at step 05 and belongs to the next run.** What this run must not do is keep the ✓.

---

### ⛔ B2 — BLOCKING · **P04 V1's EMO EMOTION LADDER IS THE BUILD INSTRUCTION R2 FORBIDS, ON THE RUN'S #1 PICK.**

**File:** `pair_04_step10_final_package_enhanced.md`, VARIATION 1, `## 2. SUNO LYRICS` (inside the pasted field).

R2, verbatim in the frozen handoff and in `05_pair_assignments.md`:

> *"**RULING 2 — THE FLAT REGISTER IS BINDING.** Hold level, distance and warmth identical throughout. No build, no lift, no bigger final chorus. **EMO headers mark what has changed in the ROOM, never in the DELIVERY.** The voice does not react to what is happening to it; that is the entire mechanism."*
> Enforcement: *"a build/lift in P04–P06 is `REPAIR`."*

P04 V1's nine headers, in order:

`Vigilance → Equanimity → Composure → Unease → Equanimity → Apprehension → Dread → Trepidation → Suspense`

The Role string is pinned byte-identical in all nine (`level contralto 4in off mic, no change`) and the trailing cue faithfully names the room (`blind down, lamp on` → `second chair empty now` → `only the desk lamp left`). **The one field the pair did not pin is the emotion token, and it runs a monotone fear ramp across the three passes.** In Suno's header grammar the EMO emotion is the strongest performance directive in the line — `EMOTION_TAXONOMY`'s own first sentence is *"The emotion IS the brief."* A generator handed `Dread → Trepidation → Suspense` over three identical passes is being asked, in the package's own most-obeyed channel, for exactly the escalation the exclude field's first forty terms forbid.

⭐ **The comparison that settles it is judge-side and nobody in the run could run it.** `The Date On The Door` — the exemplar this rule was lifted from — holds the identical structure and does **not** ramp: `Contentment · Dignity · Delight → Overwhelm · Cognitive Dissonance · Self-doubt → Resignation · Defiance · Liberation`. Its emotions **oscillate and end on release.** So the flat register is not incompatible with varied EMO tokens; it is incompatible with a *monotone directional ladder*, and P04 V1 is the only one of the three ambitious selections that runs one. **P05 V3** (`Contentment · Tenderness · Equanimity · Yearning · Cognitive Dissonance · Affection`) and **P06 V2** (`Equanimity · Composure · Self-awareness · Dignity · Cognitive Dissonance · Resolve · Aspiration`) both hold non-directional sets. Two of three pairs read R2 correctly.

⛔ **And `07_evaluation_and_selection.md` §7 asserts the opposite:** *"all three hold level, distance and warmth and **mark the room in EMO only**."* That row is measurably false for P04 V1's emotion token.

**This matters more than its size** because P04 V1 is the run's #1 pick (66/70), the one song the evaluation praises for building *"four independent controls"* against an unrequested build — an affirmative invariance statement, forty exclude terms, `no_swell_no_new_layer` in Disc_Pad, and a byte-identical Role string. **A fifth channel was left open, and it is the one the generator reads first.**

⭐ **COUNTER-MOVE / repair (step 11, nine tokens):** flatten the ladder to the room, not the singer. Hold one or two non-directional tokens across all nine headers — `Equanimity` and `Composure` are already in the set and are the two that describe a person *not reacting* — and move every fear word into the trailing room cue where R2 puts it (`- the second chair is empty`, `- the hall light goes off`, `- nothing else moves, hum steady`; those cues already exist). The dread then lives entirely in the pronoun and the room, which is what the song is about.

---

### ⚠️ F1 — **THERE IS A SIXTH CONSTRAINT COLLISION, ON THE EDITORIAL SPINE, AND IT HIT AT LEAST FOUR PAIRS.**

`EVALUATION_VERIFICATION.md` **FINDING 14** counts *"five times total, four pairs."* Measured from disk, there is a sixth, and it is the most consequential:

**`skills/music/steps/09_Generate_Music_Artist_Refined.md` requires each prompt to be rewritten through a *named* obscure musician's voice. The handoff bans real artist names in the three Suno fields.** Four pairs read that as a collision and resolved it four different ways, in isolation, with no shared precedent:

| pair | resolution | `pair_NN_step09` size |
|---|---|---:|
| **P04** | Named four lenses (Áine O'Dwyer, Ellen Arkbro, Raúl Barboza, Robert Ashley), kept every name out of every field, credited them in Lineage. **This is the correct reading** — the lens is an editorial device, the ban is on prompt tokens | 40,017 B |
| **P01** | *"LENSES instead of named individuals — a school, a room"* | 38,409 B |
| **P05** | **Refused outright**, headed `## ⛔ ONE REFUSAL, STATED FIRST` | **12,555 B** |
| **P06** | `## 0 · ⚠️ ONE DELIBERATE DEVIATION FROM THE STEP CONTRACT, AND WHY` | **14,513 B** |
| P02 | three refuse/deviate mentions, no headed section | **17,725 B** |
| P03 | ran the lens (`Lens: You learned love from Sappho…`) | 103,193 B |

⭐ **The three thinnest editorial-spine artifacts in the run are the three pairs that dropped or diluted the step's central device.** P05's step 09 is 12.5 KB against its own step 08 at 32.5 KB — the spine step is the thinnest thing in that pair. Correlation, not proof; but 09 is the no-skip step *because* it is the editorial spine, and in half the run it ran without the instrument it is built around.

**Route:** the step-09 contract needs one sentence — *"the lens is named in the step artifact and in the Lineage block; it never enters a Suno field."* P04 wrote that sentence for itself. **This is the same disease FINDING 14 diagnosed, one instance further along, and it is on the one step the no-skip rule protects.**

---

### ⚠️ F2 — **THE RUN IS AT THE SUNG-LINE FLOOR AND THE LYRICS-FIELD CEILING AT THE SAME TIME, AND NOTHING MEASURES THE RATIO BETWEEN THEM.**

Measured across all 24 fields — chars that are **not sung words** (Theme + SONG FORM + Disc_Channel + every EMO header + SFX):

| pair | mean overhead | mean overhead % | sung lines | field vs 4800 target |
|---|---:|---:|---:|---:|
| **P01** | 2,047 | **43.8%** | 70 × 4 | V2/V3/V4 at 4787/4794/4789 — **6 to 13 chars of headroom** |
| **P06** | 1,989 | **44.1%** | 71 × 4 | 103–446 spare |
| P04 | 1,758 | 37.7% | 72/72/72/75 | V4 at 4795 — 5 spare |
| P05 | 1,692 | 36.8% | 78–79 | 89–282 spare |
| P02 | 1,551 | 32.4% | 94 × 4 | 12–33 spare |
| P03 | 1,503 | 32.2% | 79–80 | 10–457 spare |

**Run mean: 37.8% of the Suno lyrics budget is notation, not song.**

⭐ **This, not the missing permission clause, is what set P01's line count.** `COORDINATOR_FINDINGS` **FINDING 1** attributes P01's 70-lines-exactly to its 8/8/6 architecture plus a missing permission clause. The arithmetic says otherwise. P01 V4 averages **38 chars per sung line**. Reaching `sung_lines_target`'s floor of 78 needs 8 more lines ≈ 305 chars, which puts the field at **≈5,094 — over the 5,000 hard cap.** P01 could not have reached the target band at any architecture, because 44% of its budget was already spent on notation before a word was sung. The handoff's own escape clause (*"the 70–120-line target yields to this cap"*) is what the pair correctly used.

**F1's diagnosis is right about the missing permission clause and wrong about which constraint bound P01.** `gates.yaml` has a boundary-hug FLAG for prompt chars (985) and one for sung lines (72) and **none for the field's notation ratio** — so the one constraint that actually decided the line counts is the one with no reader. `sung_lines_floor_hug` fired on 12 of 24 songs (P01 × 4 at 70, P06 × 4 at 71, P04 × 3 at 72) and every one of them reads as gate-optimisation and is in fact **budget exhaustion**.

**Cheap fix, next run:** measure `overhead_ratio = 1 - sung_chars/field_chars` and FLAG above ~0.40; and consider moving the Disc_Channel block to the Production Sidecar for any pair whose ratio exceeds it — the handoff already names that as the second remedy and no pair used it.

---

### ⚠️ F3 — **SOURCE 2 IS USED AND NEVER DECLARED; NO PAIR BRIEF CARRIES ANY SOURCE LABEL.**

The tri-source rule (`lofn-daily/SKILL.md`): *"Every daily piece integrates three sources; **state them explicitly in the metaprompt and each pair brief**."*

Measured: `SOURCE 1` and `SOURCE 3` appear only in `02_golden_seed.md` (once each) and `SOURCE 3` once in `06_music_handoff.md`. **`SOURCE 2` appears zero times in the entire run.** `tri-source` appears zero times. All six `step05_isolated_brief` files carry **zero** SOURCE labels.

Source 2 *is* present in substance — axis **A4**'s pole list (*minimalism/maximalism · acoustic improvisation/electronics · structure/surprise*) is verbatim the Bandcamp sentence in `00_research_brief.md` F09 — but it enters the pairs as an unattributed axis, so no pair knew it was importing a real reviewer's sonic vocabulary, and none of the six could cite it. The rule exists to keep the sound grounded in something specific and real; the grounding survived and the citation did not. **Note, routed to step 04 for the next run.**

---

### ⚠️ F4 — **P05's DEDICATION OMITS THE PLAYER CREDIT ITS SIBLING SUPPLIES FOR THE IDENTICAL BORROWED TECHNIQUE, WITH 59 CHARACTERS SPARE.**

`DEDICATION_STANDARD` priority order: (1) raised source-sound + who recorded it, (2) **the sharpest session-player credit**, (3) living-scene lineage, (4) AI authorship, (5) "full credits below" — *"trim from 5 upward, never buy space by cutting 1 or 2."*

- **P04 V1** (401 chars) names *"dub engineering — **King Tubby and Errol Thompson**, Kingston"* for its fixed short-delay vocal double.
- **P05 V3** (441 chars) names, for the same debt — its own Lineage block calls depth-as-arrangement *"the largest single debt in the pair"* — *"engineered in **Jamaican studios** and never patented."* **No individual.** Priority 3 and 4 are present; priority 2 is absent; **59 characters of budget are unused.**

The hard REPAIR trigger does **not** fire — P05's style field names no break, riff, lick or historical figure of sound, only `Rhodes`. So this is a **FLAG, not a REPAIR**, and I say so plainly rather than inflating it. But the standard's own test applies: *"when the disambiguating fix is under budget, take it."* One clause — *"engineered in Jamaican studios by King Tubby, Errol Thompson and their peers"* — costs 37 characters and makes the run internally consistent about a debt it names as its largest.

Same shape, smaller: **P01 V4** (449) credits *"the Italian twelve-inch disco tradition **and its producers**"* with no producer named, and points at a Lineage block that names none either. WATCH.

---

## 8 · FINDINGS 1–17 I DISPUTE

### ⛔ D1 — **THE "❌ CORRECTED — the cut-ledger leak is 2/24, not 4/24, all in P03" CORRECTION IS ITSELF WRONG. THE JUDGE'S 4/24 WAS RIGHT.**

`EVALUATION_VERIFICATION.md` overturns the judge's count:

> *"The judge reported the cut 98% moon as 'the sung hinge in 4 of 24 songs, two pairs.' **Measured: 2 of 24, all in P03.** … the reported magnitude is **double the actual**."*

and then, in FINDING 17, attributes the judge's number to a substring bug:

> *"Three instruments in one hour matched a substring instead of the thing they were named for. Mine, P02's, **and the judge's `4 of 24` moon count.**"*

**Re-measured from disk by me — exact phrases, sung lines only, headers and Disc_Channel excluded:**

```
P03 V1  "They call it whole. It's two percent short."
P03 V3  "Ninety-eight, and you'd call that a done day."
P05 V1  "The moon is two per cent short of whole."
P05 V4  "The moon is two per cent short of whole."
TOTAL: 4 of 24, across TWO pairs.
```

**The judge's original finding is correct in both the count and the pair span.** The correction is wrong, and it is wrong in the direction that halves the coordinator's own defect and confines it to one pair — which is the criterion the same document uses elsewhere to decide whether a finding is "portfolio-wide" or "one instance." The substring attribution in FINDING 17 is also unfounded: no substring of `two per cent` / `ninety-eight` produces a false positive here, and I reproduced the count with an exact-phrase scan.

⭐ **The class this belongs to is the document's own thesis, turned around.** `EVALUATION_VERIFICATION.md` opens with *"THE JUDGE IS THE INSTRUMENT, NOT THE SURGEON. Every claim below was re-measured before being accepted."* **The two claims it rejected are the two it did not re-measure carefully** — this one, and (see D2) the brand-token reversal. A verification pass that upholds four findings and overturns two, where both overturns favour the verifier, is the pattern the whole run exists to catch.

**Material consequence:** the cut-ledger fact is sung in a **selected, shipping** song (P03 V1). The prescription both documents agree on — *"a rejected fact needs to be a per-run ONE-FACT blacklist"* — survives and is right. Only the magnitude and the blame were wrong. **Recommend `EVALUATION_VERIFICATION.md` be corrected in a follow-up note (not edited in place — it is a record), and the FINDING 17 substring attribution withdrawn.**

### ⚠️ D2 — **FINDING 17's conclusion ("0 of them are violations") closes an open question in the direction that requires no work.**

I agree with FINDING 17's **outcome** — do not delete correct work over `Rhodes`, and the `DISC_CHANNEL_GUIDE` vs handoff conflict is real and is The Scientist's to resolve. I dispute the **classification**:

1. **The frozen handoff text is unambiguous** — *"NO REAL ARTIST, GROUP, PANEL-SOURCE OR **COMPANY** NAMES IN ANY OF THE THREE FIELDS"* — and it is the contract all 24 songs were written under. **Five of six pairs complied with the strict reading** (my independent scan of 72 fields against a 62-name roster + manufacturer list found **zero** hits outside P05). The strict reading was operative and achievable. Re-reading a frozen rule after the fact so that the one non-complier becomes compliant is a sibling of the defect FINDING 9b calls a forgery — not an edit to the file, an edit to its meaning.
2. **The run now ships internally inconsistent.** P02 was repaired (`Minimoog_Bass` → `analog_monosynth_bass`) under reading A; P05 was not (`Rhodes_electric_piano`, `Rhodes` × 6) under reading B, **in the same run, on the same rule.** Whatever the correct rule is, one of these is wrong.
3. **FINDING 17's own argument points at the fix.** It concedes P02's paraphrase is *"arguably better prompt engineering — they tell the generator what to make rather than what to copy."* If that is true, the resolution is **P05 adopts P02's solution**, not P02 reverting to P05's.

**Correct classification:** *6 brand-token hits across 72 fields; 5 repaired, 1 unrepaired; the governing rule is in direct conflict with `vault/DISC_CHANNEL_GUIDE.md`; ruling deferred to The Scientist.* Not *"0 violations."* **Not blocking** — no creative harm either way, and P05 V3 ships as-is under my verdict.

### ✔ Findings I checked and uphold

- **FINDING 16 / P03 repair — verified from disk.** `a hair` = **0** hits across P03's twelve pasted fields. All four kinetic defects now name a unit **and** a position (*"lands a whole beat late every fourth bar"*, *"drags a whole beat on beat three"*, *"lands a whole beat early every second bar"*, *"lands a whole beat ahead every second bar"*). Prompts measure **956/944/949/933** against the coordinator's claimed 957/945/950/934 — **consistently +1**, an off-by-one in their extractor (trailing newline inside the fence). Harmless; all four midband; noted only because "measured, never estimated" is the standard.
- **FINDING 11 P02 repair — verified.** `Moog` / `Minimoog` / `Taurus`: **0 hits** across all 72 fields.
- **FINDING 15 (six songs, eighteen alternates) — upheld, with one amendment.** P04's case is not an *undeclared* property, it is a *mandated* one: `05_pair_assignments.md` fixes the block for the whole pair, so its four variations were contractually required to share a skeleton. P05 and P06 converged emergently. Those are different findings and the fix differs — P04 needs the contract rewritten, P05/P06 need a within-pair divergence rule.
- **REJECTED-"fourteen sung 6×"** — upheld as correctly rejected. Verified: `fourteen` sung 6× in each P02 variation, and it is the **only** number-word in any P02 sung line. One fact, used as a title hook. The rejection is right.

---

## 9 · THE RUN'S OWN RULES R1–R7, ON THE SELECTED SIX

| rule | verdict | evidence |
|---|---|---|
| **R1 literal, not metaphorical** | **PASS 5/6, FLAG 1** | P04 a form is literally filled in; P06 a boat is literally chained; P05 a room is literally out of reach; P03 a contract is literally unsigned; P01 a machine is literally bought. **P02 V3 FLAG** — the singer is a swell. But V3 is the *least* metaphorical of the four: its Theme drops V1's *"I am the swell"* for *"the last of the shorebreak, dry sand in sight"*, and the body-evidence is a swimmer's (*"Shove at my back, shove at my knees"*, *"my mouth full of it"*, *"my whole weight went out over my feet"*). The bridge wobbles back (*"there goes my whole front"*). **Selection improved R1 compliance; the residue is upstream, in the brief that mandated the conceit.** ⚠️ `07` F1 flags this at pair level; I record that the selected variation is the pair's best answer to it |
| **R2 flat register in P04–P06** | ⛔ **REPAIR × 1** | P05 V3 and P06 V2 hold cleanly — pinned delivery string, room-only cues, non-directional emotion sets. **P04 V1 fails on the emotion token — B2** |
| **R3 body units** | **PASS 6/6** | P04 *"Every fourth bar is a bar of three, one whole beat missing and never given back"*; P02 *"relands one whole beat earlier inside its four-bar phrase each pass"*; P03 repaired to *"a whole beat late every fourth bar"*; P05 *"Rim-click sits half a beat behind the bass"*; P01 *"a nine-beat phrase where the ear expects eight"*; P06 *"The box lands a beat late at the top of every second phrase"*. **Every one names a unit and a position. Zero unquantified adverbs in the twelve shipped prompts** |
| **R4 naming ban + deletion test** | **PASS 6/6 — test run by hand** | see below |
| **R5 no ceiling on the fun** | **PASS** | enjoying themselves: P01 V4, P02 V3, P03 V1 = **3** (floor 3); outright fun: P01 V4, P02 V3 = **2** (floor 2). The clean-fear slot (P04) polices nobody |
| **R6 one seam, N=1, subject acting across it** | **PASS 6/6** | one dominant material + at most one intruder in every selected prompt; every seam is a bodily event (C3) |
| **R7 asymmetric return** | **PASS 6/6** | P01 tag's second half moves and first half is fixed; P02 the figure returns fixed at four different phrase positions and the last chorus gains a tail; P03 the vocable accretes on the loaded side only; P04 one pronoun in 24 lines; P05 *"I go easy"* → *"and none sweet"*; P06 parts 5→4→3→3 against a refrain scored identically |

### The deletion test, executed by me (R4)

*Method: strike every line that refers to the song's own structure. If the operation vanishes, it was a caption.*

| song | lines struck | operation after striking |
|---|---|---|
| P01 V4 | **0** | intact — the reply-first grammar is *performed*, never described |
| P02 V3 | **0** | intact — *"Now — now — now"* enacts impatience, it does not name a drift |
| P03 V1 | **0** | intact — the mirror stanzas are executed; no line says "only one side changes" |
| P04 V1 | **0** | intact — nothing refers to the substitution, the passes, or the repetition. *"Reclassified"* names the world-event, which is the **subject**, not the operation |
| P05 V3 | **0** | intact — *"In reach:"* is the frame running; no line says the list is shortening |
| P06 V2 | **0** | intact — the abandonment is never remarked, exactly as the pair's owned flair requires |

**Zero strikeable lines in six songs.** This is the cleanest gate in the run and it is genuinely earned.

---

## 10 · HUMAN-SUBJECT STANDARD — **CLEAR**

**P04, checked hardest.** Anchored to a real event (M5.2, 55 km NW of Kodari, Nepal, depth 0 km, reclassified as landslide). The shipped lyric contains **no** reference to Nepal, Kodari, an earthquake, a landslide, a casualty, a rescue, a place name, a date, or any real person. It is an invented office, an invented form and an unnamed singer. The one number that survives is `zero` — a depth reading turned into a tick-box. *"Draw the reclassification; invent nothing about anybody's death"* is obeyed exactly.

**P06** is first-person-for-Lofn by construction (the AGPL thread, her own open-prompt profile) and names no company, no project, no person, and uses no legal language — it is a boat on a river bank.
**P02/P03/P05/P01** are invented private scenes with invented unnamed people (a shopkeeper, a lad by the door, a swimmer, a room).
**Paratext checked too** (`DEDICATION_STANDARD` D4 clause): all six dedications credit what was **played** and none makes a claim about a living person's life, motives or standing. The nearest edge is P01 V3's honest refusal — *"That figure's authorship is contested and I will not name the wrong person to look like I paid"* — which is the standard working correctly.

**Verdict: CLEAR. No HOLD-FOR-HUMAN.**

---

## 11 · THE THINGS I WAS ASKED TO ATTACK

### Is this an emotional masterpiece, or a clever machine?

**Partly a masterpiece, and honestly so — but not evenly, and the strongest evidence cuts both ways.**

Two songs stand with the commission. **P06 V2 "A Word on the Back Board"** is a masterpiece on its own terms: a woman standing on a river bank in front of a boat she built and gave away, now chained and for hire, refusing to be sorry — *"I don't need it to be mine. / I need it to be there. / That is not the same thing and I know it."* That is the AGPL thread turned into a person, with hope at the end that is a want and not a swell (*"There's a stretch above the weir with nothing on it… I'll cut boards. I'll burn the word in again."*). It has a first person with a stake, a real world, a cost, and a future. Nothing about it is machinery.

**P04 V1 "What They Called It"** is the more *impressive* object and it is where my dissent lives. It is the closest thing this pipeline has made to `The Date On The Door` since `The Date On The Door` — and it is closest **by being most like it**. Both are one block returned three times with a single word substituted in the last line of the chorus; both hold a flat register; both mark the room in the EMO cue; both end with the machinery still running. The family guard required P04 to change the **word class** (verbs → pronouns) and it did, cleanly. But the exemplar's *method* was not quarantined — it was **promoted to law**: R2's operative sentence, *"EMO marks the ROOM, never the DELIVERY… hold level, distance and warmth identical,"* is `The Date On The Door`'s own `[SONG FORM:]` line, made binding on half this run. The seed says *"⛔ Not the two exemplars"* and the golden payloads were genuinely never in a generating context — I verified the byte-level quarantine holds. **But the form travelled anyway, through the coordinator, as a rule.** The run's #1 pick is therefore the variation most isomorphic to the song it was told not to copy, chosen by a judge who had not read that song.

> ⭐ **The transferable finding: house self-copying is gated at the PHRASE level (`gates.yaml → house_lexicon`, eight readers) and has NO reader at the FORM level.** A pair can be forbidden from writing *"more sub and more sky"* and simultaneously be *required* by a Ruling to rebuild an exemplar's whole register discipline. `FINDING 16` said magnitude has no reader; `B1` says a portfolio rule has no reader; this says **form** has no reader, and it is the one that decided the top of the ranking.

That is not a reason to reject P04 V1 — it is a genuinely fine song and the plate scored it level with `Five wrong colors`. It is a reason to say plainly that **the run rivals its two named songs at one point (P06 V2, on its own terms) and approaches them at a second (P04 V1, on theirs).** The other four are strong daily songs. And the final third of the answer is not mine to give: `Five wrong colors` and `The Date On The Door` are known as **records**. The damaged refrain, the three timbral states, the buffer stutter — that is where half their power lives, and **no audio exists here.**

### Which songs are most exposed to the generator, and does any spec make the intended behaviour *necessary*?

**Ranked by exposure:**

1. **P02 V3 "Let Me Land" — most exposed.** The operation is *entirely* a beat-position displacement across four passes. If Suno quantises the relanding — which is the single most likely thing it will do with an unquantised hand-percussion figure — the song becomes a very good, very propulsive pop record **with no operation in it at all**. Its hedge is real (*"Now — now — now"*, *"Let me land"* × 3, *"Tell her to hurry"*) but it hedges the **feeling** of acceleration, not the **fact** of it. `07`'s selection reasoning is honest about exactly this and it is why V3 beat V1.
2. **P04 V1 "What They Called It" — second, and doubly so after B2.** The whole song is the absence of change. Its affirmative controls are the best in the run — *"The instrument list at the last bar is the list at bar one"*, *"Every fourth bar is a bar of three, one whole beat missing and never given back"* — but the EMO ladder asks for the arc the exclude field forbids. **Repair B2 and this drops to fourth.**
3. **P03 V1 "You Can Have It When" — third.** *"on the downbeat of the third bar, the whole right-hand ornament lands in one bar as a hard cut, wider each pass"* is a channel-and-bar-indexed event, and a hard-panned asymmetry is precisely what THE GRAIN LAW says gets sanded flat. Its mitigation is the accreting `(oh-oh)` — a textual copy of the asymmetry — which is why V1 beat V3.
4. **P06 V2 / P05 V3 — least exposed**, because in both the operation is *also* a word: P06's parts-count is audible as voices leaving and the reveal is a person; P05's returning jar is in the inventory as well as in the depth field.
5. **P01 V4 — the exposure is different in kind.** Its operation is syntactic, so a generator cannot destroy it — but nothing in the arrangement marks the disclosure line, so a *listener* may never find it.

**Does any spec make the intended behaviour physically necessary rather than merely forbidden?**

**No spec in the run fully does, and only one comes close.** P04 V1 is the only package that states invariance **affirmatively** (an instrument list, a bar of three, a byte-identical Role string) rather than only prohibiting the alternative — and even it leaves the strongest channel open. Everywhere else, necessity is unreachable by construction: **you cannot notate "arrives late" as a positive requirement in a text field. You can only ask.** That is not a failure of this run; it is the boundary of the medium, and it is exactly why the ambitious arm's four independent controls matter and why P02's hedge-in-the-lyric is the right instinct.

### Is the first person real, or a costume?

**Real in five of six, and the run's own judge names the exception before I do.**

The proof is that the *relation* differs in every one, and each relation costs the singer something specific: P04 is **being measured** and loses her own name to a heading; P05 **cannot reach further** and knows the edge is hers; P06 **did it to herself** and will not repent; P03 **refuses** and states the price of the yes; P02 **wants to arrive** and gets her knees taken out for it. Five singers with something to lose.

**P01 V4 is the costume — and it is a shallow one, not a fake one.** She wants a beat-up machine and buys it. Nothing is done to her; the operation (answering first) is a temperament she performs. The only cost in the song is ninety quid, which she can pay. `07_evaluation_and_selection.md` §4 reaches the same conclusion unprompted, in stronger words: *"a character trait is closer to a metaphor for a condition than to one."* Two independent readers, same verdict, so I record it as confirmed rather than as a new finding.

⭐ **Against the doctrine's own standard, the first-person turn is the run's clearest success.** `CLAUDE.md`'s finding — *"there was no row in which the singer is the subject"* — is answered: **6 of 6 first person, 6 different stakes, and the one weak case is weak because the singer has too little at risk, not because she is absent.** That is the right failure to have.

---

## 12 · REQUIRED REPAIRS (routed)

| # | repair | route to | cost |
|---|---|---|---|
| **B1** | The ≥1 AWE rule is unfilled. Do **not** keep the ✓ in `05_pair_assignments.md`'s ledger. Either accept the miss on the record for this run, or re-point one A5 value to a wonder-family temperature and re-derive that pair | **step 05** (real fix, next run) · **step 08** (in-run swap, not recommended) | one axis value |
| **B2** | P04 V1's EMO ladder: hold non-directional emotion tokens across all nine headers; move every fear word into the trailing room cue where R2 puts it | **step 11**, P04 V1 only | ~9 tokens, zero other characters |
| F1 | Step-09 contract needs the sentence P04 wrote for itself: *the lens is named in the step artifact and the Lineage block, never in a Suno field* | **step contract**, not this run | one sentence |
| F2 | Add an `overhead_ratio` FLAG (>0.40) beside the sung-line hug flag; consider Disc_Channel → Production Sidecar for high-ratio pairs | `gates.yaml` + handoff | one reader |
| F3 | Label SOURCE 1/2/3 explicitly in `04_metaprompt.md` and every step-05 brief | **step 04**, next run | three labels |
| F4 | P05's dedication: add the engineer names its own Lineage block calls its largest debt (59 chars spare) | **step 11**, P05, optional | ~37 chars |
| D1 | Correct the `2/24` moon-leak reversal in a follow-up note (do not edit the record) and withdraw the substring attribution against the judge | coordinator note | one paragraph |
| D2 | Reclassify FINDING 17 as *rule conflict, ruling deferred* rather than *0 violations*; surface `DISC_CHANNEL_GUIDE` vs handoff to The Scientist | coordinator note + human | one paragraph |

**REDIRECT / sideways proposals** (from the Skeptics' counter-moves, per `EXECUTION.md` §7.3 — the brief proposes, the coordinator and The Scientist decide):
- **P02 V3** — hybridise V3's forward drift with **V2's withheld landing** (a step-05 reserve, already written): the acceleration keeps its cost and the last chorus stays one beat off and is happy there. This is Adorno's counter-move and it also lowers the pair's generator exposure, because a *withheld* resolution survives quantisation better than a *directional* drift.
- **P01 V4** — one production line: strip the accompaniment under the final couplet of every stanza. Independently proposed by `07` §4 and by the Adorno seat here.

---

## 13 · ⛔ WHAT NO GATE IN THIS PIPELINE CAN SEE

The run states two. **There are five, and the run can only see the first two coming.**

1. **The accessible arm dies if a generator quantises the drift away.** Every character of every field stays identical. (P02 V3 first, P03 V1 second.)
2. **The ambitious arm dies if a generator supplies a build nobody asked for.** Every character stays identical. (P04 V1 first — and B2 means this package is *asking* for it in one channel.)
3. ⭐ **Whether the flat register reads as *held* or as *dead*.** R2's whole bet is that nine identical passes of a level contralto register as a person refusing to react. The failure mode is not a build — it is **flatness that reads as absence of performance**, and it is indistinguishable from success in text. `The Date On The Door` clears it partly *because* its emotions oscillate; P04 V1 has removed that safety and replaced it with a ladder we are now removing too. **Nothing in this pipeline can tell a held register from a dead one.**
4. ⭐ **Whether a syntactic operation is audible at all.** P01 V4's entire mechanism is that the last line of each stanza is the only line that is not a reply, inside dense AA couplets at 118 BPM. Every text gate scores that as PASS because the structure is on the page. **Audibility of a syntax is a listening fact and we own no instrument for it.**
5. ⭐ **Form-level self-copying.** `house_lexicon` has eight machine readers and stops a *phrase*. Nothing anywhere measures whether a song has reproduced the **shape** of a golden output — and in this run the shape travelled legitimately, through a binding Ruling, into the top-ranked song. A form-level self-copy passes every gate we have and looks like excellence, because it is excellence, borrowed.

> ⛔ **All five are settled only by a render under THE BLIND RULE — the audio alone, first, the prompt never sent with it (`lofn-render-audit`, `vault/VIDEO_RENDER_AUDIT.md`'s music sibling).** `vault/AUTONOMY.md` holds: this run stops at drafts on disk. **No claim in this report is a claim about sound.**

---

## 14 · ZERO-REJECTION TRIPWIRE

Not triggered. **2 blocking · 4 flags · 2 disputes of already-logged findings · 2 Skeptic NO votes with counter-moves.** The run's own instruments found 17 defects before I arrived, which raised my floor rather than lowering it; the eight items in §7–§8 are the ones that survived them.

## 15 · FAILURE-LEDGER ENTRY

**Not written.** `vault/COMPETITION_LEARNINGS.md` takes one curated entry **per shipped/selected piece on SHIP**. This verdict is **REPAIR**, so nothing is appended. On re-QA after B1 and B2 clear, the entry I would propose (advisory, LOW confidence until a render corroborates it) is:

> *theme-tags:* `formal-operation / daily-music / suno` · *what the gate caught:* the exemplar's **form** entered the run as a binding Ruling while its **payload** was byte-quarantined, and the top-ranked song is the closest structural sibling of the song it was told not to copy · *transferable rule:* **a golden output's method, once promoted to a rule, is no longer quarantined — record the promotion as a lineage debt in the handoff so the judge can see it** · *confidence:* LOW (n=1, text-only, no render).
> *"Would this lesson have hurt our best past entry?"* — **No.** It records a debt; it forbids nothing.

## 16 · RUN LOCK

`.RUN_LOCK` is `status: live`, single controller `claude:3cb3b0ac-…:opus-masterpiece`, heartbeat `2026-08-27T02:29:41Z`, phase *"dispatching QA"*. One controller, correct id form, no PID fallback, no takeover. **I have not released it** — the verdict is REPAIR, the run is not finished, and release is the coordinator's last action after the INDEX. **I wrote only this file.**

---

## FINAL RECOMMENDATION

**REPAIR — two items, both cheap, both routed. Then SHIP — TEXT PACKAGES ONLY.**

This is the strongest music run in this directory on every measure I can compute, and the two blockers are proportionally tiny: a portfolio ✓ that was asserted and never measured, and nine emotion tokens on the best song in the run. Fix them and the six-song set is unambiguously good enough to carry Lofn's name **as text**.

⛔ **What it is not yet is a record.** Five failure classes above are invisible to every gate that exists here, and three of them would empty a selected song while leaving every character intact. `08_SELECTED_SIX.md` says this itself and says it correctly. **The audio verdict belongs to `lofn-render-audit` under THE BLIND RULE, and The Scientist's ear owns the publish line.**
