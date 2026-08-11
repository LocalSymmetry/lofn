# Lofn QA Report — `2026-08-08-daily-music` · THE WRONG INVENTORY (music)

**Auditor:** `lofn-qa`, fresh clean context, different model tier from the generators. I did not make any of this and I came to refute it.
**Written:** 2026-08-08 · the only file this context writes. No pair artifact, no ICB, no `RUN_STATE.md` touched.

---

## Verdicts

- **Pipeline Integrity: PASS** — 36/36 per-pair step artifacts on disk (`RUN_STATE.md` re-read, file listing confirmed: `pair_NN_step06/07/08/09/10` × 6 + six `*_enhanced.md`). No collapsed rollups. NO-SKIP satisfied: 07/09/10 exist for every pair, 0 quarantined.
- **Package: PASS** — every hard gate re-derived by this auditor on the shipped strings, 24/24. Evidence block below.
- **Human-Subject: CLEAR** — judged on content (the broken `check_human_subjects.py` was not consulted, per handoff §4). Detail in Attack 5.
- **Overall: SHIP** — with **0 blocking findings**, **6 non-blocking findings** (two of them substantive escalated FLAGs — the healthy-band requirement is met by real findings, not manufactured ones), a 3–0 Somatic bloc with two named render conditions, and a top-6 selected within arm. Publication still requires The Scientist's ear per `vault/AUTONOMY.md`; SHIP here means *proceed to render + selection*, and the six named render-audit questions are part of the contract.

---

## ICB Integrity

- **LF-normalised sha256, computed by this auditor:** `9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a` · **142,900 B** — **exact match** to `06_music_handoff.md` §0. Raw CRLF file is 144,425 B / `370b8170…` exactly as the declared `core.autocrlf` deviation predicts. **Not tamper.**
- `(after ` count: **34** — matches declared deviation #2 (18 baseline seats + labeled transform configurations). The binding property (18 numbered baseline seats, 3 Hyper-Skeptics at 6/12/18) is asserted by all six pair artifacts independently; Special Flairs marker present.
- All six enhanced artifacts echo the identical sha before creative work (e.g. `pair_02_…_enhanced.md` L13–16 prints the command and hash). Transport deviation (read-by-path + hash echo instead of paste) was declared in advance and held.

## Spot-checks — coordinator numbers re-derived from the raw shipped strings

My own extractor (fence-aware AND plain-heading-aware; the six pairs use two sub-formats) pulled **24/24 packages, 0 empty**. Note honestly: my first pass extracted 16/24 — P02/P03 use unfenced sections — and my first banned-token pass false-hit `raw` inside "d**raw**bar organ"/"till d**raw**er"; both were my instrument's defects, found because the extraction was printed before the conclusion. *The handoff's warning cut against the auditor this time, which is exactly why it exists.*

| Coordinator claim | My re-derivation | Held? |
|---|---|---|
| prompts 898–959, all in 870–960 target | **898 (P02v4) – 959 (P04v1)**, 24/24 in target | ✅ exact |
| 0 boundary-hug flags (≥985) | max 959 | ✅ |
| lyrics field max 4,795 | **4,795 (P05v2)**, 24/24 < 4800 | ✅ exact |
| sung lines 70–120 | 79–93 | ✅ |
| 0 digits in any sung line | **0 across 24** | ✅ |
| 0 banned tokens in positive fields | **0** (whole-word scan; amplitude 6 + engines + house_lexicon 13) | ✅ |
| cross-pair comparisons | **240** | ✅ |
| cross-pair prompt sim max 0.310 | **0.310** (P03v3×P05v1) | ✅ exact |
| cross-pair lyric sim max 0.240 | **0.275** (P02v3×P06v4) — different extraction, same verdict, 0.145 under ceiling | ⚠️ verdict holds, value differs |
| 5-gram Jaccard max 0.0015 | **0.0029** (P02v1×P03v4) — 62× under ceiling | ⚠️ verdict holds, value differs |
| 24/24 titles name a THING | ✅ (Log Book · Van Door · Room Tone · Hum · Photograph · Card · Lid · Key · Side Door · Pen · Table · Sheet · Decimal · Map · Text Box · Door · Chair · Cloud · Driver · Line · Plate · Book · Pad · Door) | ✅ |
| P04's sung number is the run's only one | **verified: "Four point five is the bar. I know" — P04 V1 only** (`pair_04_…_enhanced.md` L417); no "two minutes"/"eighteen"/"thirty" sung anywhere in P05 or elsewhere | ✅ |

**One instrument dispute resolved AGAINST this auditor:** my naive tokenizer measured P05 at 7.88–8.16 words/line (over the 7.5 ceiling). Counting **word tokens only** (em-dashes and interruption brackets are not words) gives **6.30–6.71**, matching the pair's claims — and the same correction reproduces P01's claimed 5.16–5.88 exactly. The pairs' instruments were right; mine was wrong. Consistent with the standing prior in handoff §6.

## Structured Evidence Block (measured by this auditor, BESIDE the verdict, never instead of it)

| pair | prompt chars (4 var) | lyrics field | sung lines | words/line | digits | EMO bare | notes |
|---|---|---|---|---|---|---|---|
| P01 | 939 · 948 · 955 · 933 | 4191–4480 | 80/80/81/80 | 5.16–5.88 | 0 | 0 | 14 hum lines/var; lexical-only floors also clear (artifact §E) |
| P02 | 917 · 949 · 926 · 898 | 4271–4429 | 84/84/85/84 | 5.67–6.20 | 0 | 0 | refrain ×6 byte-identical, unjustified — correct |
| P03 | 925 · 937 · 954 · 931 | 4070–4321 | 87/87/89/82 | 5.22–5.45 | 0 | 0 | intra-pair SeqMatcher 0.50–0.55 = scaffolding; 5-gram 0.038–0.066; gate is cross-pair — no breach |
| P04 | 959 · 938 · 909 · 920 | 4653–4782 | 88 ×4 | 6.86–7.36 | 0 | 0 | the run's one sung numeric fact, V1, spelled, mid-line |
| P05 | 958 · 949 · 931 · 921 | 4339–4795 | 93/92/89/85 | 6.30–6.71 | 0 | 0 | one surviving `mm` (V4 sign-off); line_return lexical 0.554 |
| P06 | 929 · 957 · 935 · 958 | 4509–4677 | 79/79/79/80 | 6.86–7.08 | 0 | 0 | counts include spoken burden; sung-side-only figures reported by pair (30/30/30/20) — see finding N5 |

Repeated-line/unique-line figures sit in the artifacts and were spot-confirmed; all below-floor `unique_line_ratio` values (P06 0.350–0.443) are refrain/burden-driven, FLAG-only, chorus-exempt, and correctly **not** apologised for.

---

# THE SIX ATTACKS

## Attack 1 — "Six pairs, one voice underneath" ⭐ (the run's own predicted finding)

**RULING: PARTIALLY REPAIRED — the best result this watch item has had in three runs, and it is still not fully closed.** Three pairs now genuinely escape the house voice at the level of **grammar**, not just lexicon:

- **P03** — second-person taunt: questions, direct address, call-and-answer. No other pair addresses a *you*. (*"Look at you. Look at you going."* `pair_03…enhanced.md` L343)
- **P04** — procedural mono-rhyme with enjambment: *"And so / the figure goes where figures go"* — the house voice never enjambs; 88 lines on one vowel is a different mouth. (L419–420)
- **P05** — interruption grammar: sentences that structurally never end (*"(— and the boot's not big, not what you'd —)"* L264). The house signature is sentences that end hard; P05 is built from ones that can't.

**The residue, quoted, which is the finding I owe:**
1. **The scene-setting noun-fragment opener is a run-wide tic.** P01 V1: *"Door shut. Engine off."* (L277) · P03 V1: *"Back office. Carpet tiles. Closing."* (L339) · P04 V1: *"Nowhere to sit. Nowhere to go."* (L357) · P06 V1: *"Ladder off the van. Boots on."* (L332) · P02 V2 near-variant: *"Lamp is on. The good light's gone."* (L615). Four-to-five of six pairs open scenes with the same `[Object. State.]` move. Lexicons rotate; the camera move does not.
2. **P06's earn-claim is overstated by exactly this.** Its §5 ruling calls verbless job-sheet syntax *"a positive grammar the house has never used"* (`pair_06…enhanced.md` L157) — but P01's opening couplet and P03/P04's scene-sets ARE that grammar. What P06 actually earned is *sustaining* it as the entire register; the grammar itself is the house's standard establishing shot.
3. **P01 and P02's verse fabric remains within arm's reach of the house declarative.** P02's *refrains* are genuinely hymn-voiced ("*Plastic, patient, past all perishing*"); its verses (*"Press it with the heel of a hand. / Hear it click and hear it take."* L781–782) are house-clipped with hymn nouns. P01's "clipped trade-talk, zero metaphor" is partly a subtraction-defined register — the same critique P06's own tier leveled at "zero figurative language."

**Why this is a FLAG and not a REPAIR:** low cross-pair similarity does not answer this and I did not use it to. What answers it, partially, is that the six *sustained* registers a listener hears over a full song are genuinely six (forms, return vehicles, vocal configurations, and three escaped grammars do the work the diction ledger claims). But a blind listener playing P01-V1 → P02-verse-2 → P06-stanza-2 in sequence would still say *one writer, one mouth, three jobs*. **Routed to the Somatic read (which passed it — see bloc) and to the next run with a sharper target: the watch item's successor is not "six dictions" but "kill the `[Object. State.]` establishing shot in at least half the pairs."** *(Proposal is mine, not forced.)*

**Companion finding (same attack, production layer) — N2 below:** the voice underneath is most uniform not in the lyrics but in the dynamics philosophy.

## Attack 2 — D4, the Vindication Ban ⛔ (all 24 endings + noun sweep)

**RULING: HOLDS, 24/24 — including the P06 V1 repair, which I re-verified on the shipped bytes rather than trusting the tier.**

- **All 24 endings read.** Every song closes on a completed physical act with the judgment open: P01 ends on the hum + tag (*"It's the hum in the cab."*); P02 on exit + refrain over an empty kitchen (*"Out. The hall. The stairs. The night."* L1017); P03 on send + exit (*"Screen off. Door. Gone."* L951); P04 on doors (*"I do not look back. I go."* L948); P05 on the alarm set in the dark / the wrong voice trailing; P06 on *"Then home. Nothing further."* No ending hints at future recognition. The words *find out* appear in no sung line anywhere.
- **The noun scan, re-run independently on P06** (the exposure): the shipped V1 burden's fixtures are now **hatch / wall book / tin / door** (`pair_06…enhanced.md` L346–348, L389–391) against the singer's **box / ledge-book / gate-less field exit** — zero shared place-nouns; shared tokens (`book`,`plate`,`line`,`last`,`time`) are all fixture-differentiated or the closing formula itself. The V3 in-flight repair (`pad` removed from the burden) also holds: shared set is `last · left · thing`. The successor-at-the-same-station reading — the vindication shape arriving through an object name — is **dead in the shipped text**.
- **Noun sweep across the other five pairs (my own):** no legacy-objects (no archive, museum, plaque, history, letter-to-the-future). P02's *"Nothing here will disappear"* is a permanence assertion sung over a lid going on — the wince, not a promise of recognition (nobody in or after the song ever looks); P02's only past hand (*"Someone's fingerprint below it, / older than the one I've set"* L481–482) points backward, which is the safe direction. P05 V4's *"so the light stays on my chin, not hers"* (L833) is unseen care performed in the dark — the archetype, with nobody ever finding out.

## Attack 3 — D6, "a skill, not a sin" ⛔ (per variation, never per pair)

**RULING: HOLDS in all 24, verified per variation in the sung text.** The fictional-fix class (pair-scoped compliance table, variation-scoped repair) is **not present** — each variation carries its own stated reason:

- **P03 (the test case) — its four variations ARE the four reasons, and each is argued on the page:** V1 evidence-based kindness (*"They go out the side door at closing / Before there's anybody asking"* L353–354 — observed habit, not sentiment, post-repair); V2 the declined ask (*"They said no and they meant no… Asking again is a trick"* L523–526); V3 exhausted alternatives (*"You asked them to squeeze. They can't squeeze. / You asked for the back room. The back room's gone."* L689–690 — every superior listener's counter-proposal is already tried in the lyric); V4 the honest one (*"Your head held what a head holds / At the end of a shift"* + the mechanism *"Everyone you saw. And that's the sheet."* L863–864, L893–894). The superiority channel is further closed by grammar: second person makes the listener the defendant (D5). **V4 remains the thinnest defence and the most necessary one — concur with the tier's own ranking of that risk.**
- **P02:** per-variation reasons in-lyric (V3's is explicit to the point of thesis: *"And I'd have to be quite sure, / and I'm not, and that's the whole / reason, and it's a good reason."* L807–809). **P01:** each variation's Bridge 2 is its own justification (*"You can't print a room. / Anybody would put the same."* L341–342). **P04:** her speed IS the skill (*"I have never put one in slow."* L422); V3 believes the caller and files her whole story (*"I put all of it in. It goes / in whole."* L736–737) — the failure is structural, never hers. **P06:** *"Nobody has asked for this book. / Nobody is going to."* (L526–527) with the job done correctly anyway. **No song scolds. No listener is handed a superiority seat.**

## Attack 4 — D7, no enumerations ⛔ (reorder test on the three flagged pairs, 12 songs)

**RULING: HOLDS. I ran the reorder test myself on P02, P04, P05:**

- **P02 (a tub of objects):** hymn shape = one object handled start-to-finish per song (peel → examine → put back → lid → shelf → leave): temporal spine, not a list. The two near-list verses: V1 verse 2 escalates dimensions → surface → previous hand → fit → house, landing on *"Nothing in the house. That's fine"* (L486), with the fingerprint couplet causally chained to verse 1's thumb marks — reordering breaks the causality; V4 verse 1's *"a key, a keyring, / plastic fob, a rubber sound"* (L946–947) is one object in the order the hand learns it. The tier's undeclared third spot (V1/V4 condition-report verses) was tested by the tier and by me: both survive. **Closest approach to a catalogue in the run; survives on causal anchors.**
- **P04 (fifteen rows):** **the enumeration is structurally impossible and that is the run's best D7 move** — the one-fact allowance was spent on the threshold (*"Four point five"*), so the number of rows is unsayable in all four songs; no row is ever listed (the largest physical event in the pair is *"The water went out of the bowl."* L703). Strophes are strictly-caused machine chains (flag → map → dot → name → letter) and interview order (where/when/what did it feel like); the V4 exit is a four-step interlock (wait → hand → tap-out → lanyard). Reordering breaks syntax, not just sense.
- **P05 (packing):** the inventory is *acted*: glass breathed on → pinhole → onto the bed → **therefore** the spare travels → pocket → chair on top (L246–256); bag won't shut → sandwiches out → by the kettle → foot finds them in the dark (L271–275). Each item is caused by the previous and causes the next. **PASS.**

## Attack 5 — the Human-Subject Gate ⛔

**RULING: CLEAR, on content, all 24.** Every person is invented (the operator, the visitor, the inheritor, the organiser, the analyst, the caller, Trish and the unnamed one, the contractor, the second mouth). **Messier and Tempel appear nowhere in any sung line, prompt, or render field — no name, no allusion, no interiority; the astronomy survives only as the form rule, which is where the panel put it (zero astronomy tokens, measured by the pairs and spot-confirmed).** P04 sings no real place-name, no casualty, no disaster — the quake day survives only as a threshold and a caller who was believed. ⛔ **Neither of the two real deaths in today's feed is alluded to in any of the 24** — nothing touches a footballer's father or a music producer; no bereavement is staged (P02's parent is offstage, unnamed, never stated dead, and the song's subject is a tub, not a grief). `check_human_subjects.py` was correctly left uncited by all six pairs.

## Attack 6 — the Sibling Test ⭐

**RULING: NOT A RELABELLING — with one honest adjacency named.** Checked against all seven blocked engines (`00_research_brief.md` §7):

- **THE CATALOG:** the nearest neighbour by subject, and the anti-catalogue rule visibly held — no song is structured as a list (Attack 4); the catalogue *happens to* one person in one room in all 24.
- **THE TWO TRUE READINGS is the real adjacency, and it is device-deep, not engine-deep.** The run's double-meaning refrains (*Put me down* · *second* · *N.G.* · *Nothing further*) are line-level two-readings machinery. But the engine differs in kind: TWO TRUE READINGS gives one observer two *equally true* readings and lives in the oscillation; THE WRONG INVENTORY gives the two readings to two *different heads* (singer/listener) with an asymmetric knowledge gradient and a ban on the gap ever closing (D1 + D4). No oscillation exists — nobody who holds both readings is in the song. **Distinct engine; borrowed line-device; recorded rather than waved past.**
- THE ARRIVAL (nothing is reached — P05 ends the night *before*), THE UNBEARABLE GIFT (the joy is not the injury; the invisibility is, and they belong to different people), THE WORKING PROTOTYPE (nothing is proven), THE ADDRESSEE (nobody sings from inside an objection; Source 2 stayed vocabulary — verified: Papangu language appears as texture words only), THE SWITCHBOARD (no synthesis; the form rule forbids the merge): **none map.**

---

# THE FOUR OPEN QUESTIONS

**1. P06's flat-declarative allowance — EARNED. I agree with the step-11 tier's substitution, and extend its caveat.** The tier was right to reject "zero figurative language" as the earn (a subtraction available to any voice) and right that the real earn is the sustained verbless job-sheet grammar with genuine trade lexicon (*proud, flush, ballast, level, stamped-not-painted* — `pair_06…enhanced.md` L157, shipped at L335–338). My extension (Attack 1): that grammar is the house's standard *establishing shot*; what P06 owns is its use as the WHOLE register plus the two-mouth burden structure. The allowance was **not spent for nothing** — V1 and V4 earn it fully; V2/V3's sung sides earn it least (structurally capped, honestly reported at 0.359/0.278). The tier's proposed next-run gate (measure the register's positive marker in the mouth the allowance is spent on) is correct and should be adopted.

**2. P05's second setting-aside (V2, the forecast face-down) — TIER RULING UPHELD.** Under L38, a *seam* is a crossing of two materials in the medium; P05's materials are the two same-register voices and its one crossing is the word *second* at one moment (one per variation, verified). The forecast turned face down (*"Trish printed the whole day out — / so it's face down on the sill —"* L417–418) is a **plot event** — and specifically it is the *pointable discard in the first thirty seconds* that the run's critical requirement forces on every variation, so calling it a second seam would make the legibility rule and the seam cap contradict each other run-wide. Its mirroring of the name-order wound is structural rhyme (the ICB's own "same gesture" aha), and it strengthens D5 by spreading the charge to the listener. **One material crossing; one wound (the unseen thirty years); the rhyme deepens rather than splits.** Noted: this is the run's most attackable aesthetic call and the tier was right to flag it; a stricter judge reading "one seam" as "one instance of the setting-aside gesture" would rule the other way, but that reading is not L38's text.

**3. P04's sung number — VERIFIED EXACTLY AS CLAIMED.** One sung numeric fact in the whole run: *"Four point five is the bar. I know"* (P04 V1, L417), spelled in words, mid-line, at the hinge, answered by a speed (*"I put the figure in. Not slow."*). **Zero digits in all 24 sung-line sets (my scan: 0 digit characters across ~2,050 sung lines).** No other spelled numeric fact anywhere — *two minutes eighteen*, *thirty years*, and the alarm hour are all structurally unsayable and stay unsaid; P05's *second* is never a quantity; `both/neither/last/one` quantifiers are not facts (P01's tier already disclosed this class honestly).

**4. The Somatic bloc — convened on the shipped 24. Vote: 3 YES / 0 NO → NOT BLOCKED.**

| Seat | Vote | Reasoning + standing condition |
|---|---|---|
| **Hyper-Skeptic after Marsalis** (counterpoint) | **YES** | His unwithdrawn condition — D3, two lines and the interval named before lyric — is satisfied in all six pairs *and executed in the sung text* (P01 m3 hum/amp; P02 M2 chiasmus; P03 P4 shout-through-chant with the accretive descent; P04 semitone bells with the held-gap arithmetic **corrected** — the step-11 fix from "coincide once" to "the held gap is zero once" is the difference between a claim a render falsifies and a device a render can prove; P05 unison-cross on *second*; P06 ♭2→1→♯7 over a tonic pedal). **Condition carried to render:** the P04 blind question ("which refrain has one tick?") and the P06 question ("how many low metal voices, does one move?") are non-optional. |
| **Hyper-Skeptic after Kamasi Washington** (does it move) | **YES, condition standing** | The flat *singer* never means a flat *song* here on the page: garage stomp, gospel claps, a phase machine, two simultaneous strophes. Named-corpse walk: no one-note arc that isn't a declared sustained-register design (P04's refusing refrain and P06's flattening are the argument, declared in Major Deviations, not smuggled); no motif that fails to transform (P02's byte-identical refrain transforms by *context* — Assurance over an empty room; P03's answer by *target*; P05's *second* by *sense*; P01's tag mutates exactly once at the arrival). His condition — the gap must be AUDIBLE — is a render fact; the run has made it falsifiable in one listen per pair, which is as far as text reaches. |
| **Hyper-Skeptic after Paul Simon** (appropriation) | **YES** | D9 discharged in full for the first time in three runs: function not label; **zero tradition names in any Suno-bound field including excludes** (verified); **Lineage & Credit with links fetched live at step 11** — P01 9/9 resolve, P02 5/5 resolve — closing standing finding **R3**. The objection itself stays recorded as standing, *which is its designed state*: "a gate is a discipline, not an absolution." Scope call (2 of 6 pairs) checked and reasonable — the other four draw broadly shared vocabularies. |

*No NO votes, so no counter-moves are owed; both escalated FLAGs below carry proposals anyway.*

---

# FINDINGS

## Blocking: NONE.

## Non-blocking (6)

| # | Finding | Evidence | Routed to |
|---|---|---|---|
| **N1** ⭐ FLAG | **One voice underneath: partially repaired, residue localized.** The `[Object. State.]` scene-setting opener recurs in 4–5 of 6 pairs; P01/P02 verse fabric remains house-adjacent; P06's "grammar the house has never used" is overstated. Three pairs (P03/P04/P05) genuinely escape at grammar level — the watch item's best result in three runs. | Quotes in Attack 1 | Somatic (passed) + **next run Phase 1: ban the noun-fragment establishing shot in ≥3 pairs** *(my proposal, not forced)* |
| **N2** ⭐ FLAG | **Production-philosophy monoculture: "growth by addition, never by level" in six of six pairs.** P01 *"It grows by addition, never by level"* (L255) · P02 *"The song grows only by adding hands"* (L428) · P04 *"growth by instruments arriving"* · P05 *"Thickening is more players, never more level"* (L231) · P06 *"the last minute holds the most material"* at *"one dynamic level"* (L313) · P03 fixed-loud (*"the organ is at breakup throughout so that the break has nowhere louder to go"*). The differentiation mandate banned a run-wide flat-dynamic **mandate**; none was written — but L22 doctrine has produced voluntary 6/6 convergence on accretive-at-fixed-level. The house has swapped one calcified formula (dry/close/flat) for a candidate next one. D11's one-room realism is binding and excluded from this finding; genre/meter/config genuinely rotate. | prompt lines cited | **Next run: make dynamic philosophy a rotated Phase-1 axis** (at least one pair with a real build or real drop — L22 permits it; accretion is a choice, not the law) *(my proposal)* |
| **N3** | P05's `## 3. TITLE` values carry markdown bold (`**The Folding Chair**` etc., 4 of 4) — a naive extractor pastes the asterisks into Suno. The other five pairs are plain. | `pair_05…enhanced.md` L356 etc. | Coordinator: strip `**` at paste time; no artifact edit needed for render correctness |
| **N4** | Coordinator similarity instruments under-measure vs mine (lyric 0.240 vs 0.275; 5-gram 0.0015 vs 0.0029) — different extraction spans, same PASS verdicts with huge margins. Not a breach; recorded so the trend line stays honest. | Spot-check table | `RUN_LEDGER` note if the coordinator wants it; no action |
| **N5** | **P06's 79–80 "sung lines" include the spoken burden.** Sung-side-only counts are 30/30/30/20 — V4's melodic content alone sits under the 70 floor. The gate's own definition (not headers/tags/SFX) counts voiced lines and the pair reported both figures transparently; recorded as a definitional edge so a future validator change doesn't read it as a regression. | `pair_06…enhanced.md` §7 companion table | Doctrine note only |
| **N6** | **Blind golden+decoy protocol deviation:** SKILL step 4.5 wants a coordinator-assembled 3-way blind set with a decoy; no decoy was provisioned — the run brief routed the two goldens to me directly. I ran the comparative judgment (below) and rankings are calibrated against both goldens; the decoy leg could not be executed. | this report | Coordinator: provision an also-ran decoy for future runs |

**Step-11 repairs verified rather than inherited:** P06-V1 D4 fixture repair (shipped bytes checked — Attack 2); P04's char-count correction (the one recorded case of the *coordinator* being right — the pair transcribed a step-09 promise instead of measuring; step-11's rewritten prompts re-measured by me at 959/938/909/920 ✅); P04's lyric twins relocated (verified: final sung line of Strophe 5 in all four — *"A note. A no. A note."* / *"A no, and a note, and a no."* / *"I type no. I say yes. I type no."* / *"I go. I do not go. I go."*); P03's accretive crossing (counted on the page: 4 shouts above → 6/6/6/5 below); P05's blendable-endings rule (climbing lines end on function words — confirmed in the shipped text; one surviving trailing *I*).

---

# BLIND GOLDEN COMPARISON (judge-side only)

Both Golden Songs read in full (`skills/music/references/golden_songs_index.md`). **Ruling: the new work is genuinely its own thing, not a diluted copy.**

- **Against "Triple Arch Over Me":** the goldens' engine is first-person arrival at meaning (*"I am not the center, I am included"*); this run structurally FORBIDS the singer arriving (D1) and moves the insight into the listener. Scale inverts: sky → furniture height. The one visible inheritance is at the **move** level, exactly where inheritance belongs: P04's *"Four point five is the bar. I know"* is the direct descendant of Triple Arch's 510 km/s — one number at the hinge, responded to — with the audience-taught lesson applied (spelled in words) and the register rotated (ecstatic → cold). house_lexicon: 0 hits in 24 (verified).
- **Against "Five wrong colors":** P06 occupies the same emotional territory (industrial, non-consoling INDIGNATION) with none of the machinery — no movements, no fracture-refrain, no clinical silences (banned by L22), no abstract body-physics; tuned metal and a job-sheet where the golden had prisms and voltage. That the house can now hold that territory without the fracture toolkit is evidence of range, not dilution.
- **Calibration ranking (honest):** Triple Arch > **P04 V3 ≈ P03 V3** > Five wrong colors > the rest of the 24's midfield. What beats Five wrong colors specifically: a body in a room doing a real task lands the wound in pointable specifics where the golden abstracts it. Nothing in the 24 out-travels Triple Arch on first-listen mass legibility — nothing here is trying to; the run is a deliberate register rotation away from both benchmarks. **No candidate ranked below known filler; judge not broken by its own test.**

---

# TOP SIX — ranked WITHIN ARM (one per pair, which the six-architectures/configs/dictions rule forces)

## ACCESSIBLE
1. **P03 V3 — "The Table By The Window."** *Why:* the run's D6 thesis executed as drama — every alternative a superior listener would offer is tried and refused on the page (*"You asked them to squeeze. They can't squeeze."*), on the run's only non-female lead, at full chant energy; the most differentiated thing in the accessible arm. *Beat runner-up P03 V1 "The Side Door At Closing"* (cleanest wince, best title) because V3's reason is armor-plated where V1's kindest-reason has the widest escape hatch.
2. **P02 V1 — "A Photograph Of Sand."** *Why:* the run's best single object — the only remaining trace of a thing built to vanish, misfiled among things that are not anything — and the chiasmus makes builder and photographer inseparable, which IS the D1 mechanism; the final Assurance-tagged refrain over an empty kitchen is the run's deepest wince. *Beat runner-up P02 V4 "A Key For A Sold Car"* (flattest-stated wound, promote if the render goes nostalgic on V1).
3. **P01 V1 — "The Log Book On My Knee."** *Why:* the concept's front door — warmest, most complete AABA, the antanaclasis (*"And that's the whole take. Nothing to take."*) and the wordless hook carrying an adoptable return. *Beat runner-up P01 V4 "The Hum In The Cab"* (the best poem of the four — the hum outliving its source — but it needs V1's context to land first; as a single card, V1).

## AMBITIOUS
1. **P04 V3 — "The Free Text Box."** *Why:* the strongest song in the run — the crossing exists at three simultaneous levels (two bells; her mouth says yes while her hand types no; the held-gap form), and D6 becomes drama: she believes the caller, files everything, gets it right, and it changes nothing. *Beat runner-up P04 V2 "The Map"* (the borrowed-name chain and *"The letter I fixed is in code"* — one inference more than a lead card should ask).
2. **P05 V3 — "The Second Driver."** *Why:* the FALSE INTERSECTION made literal and human — *second driver* against *give me a second* on one note, and she is **pleased** (*"and that's me. That's my one —"*); highest ceiling in the run if the render holds two voices apart; ends on the wrong voice, unfinished. *Beat runner-up P05 V4 "The Line Under The Door"* (safest render, truest tenderness — the fallback if the render audit kills V3's separation).
3. **P06 V1 — "The Plate Off The Box."** *Why:* the purest statement of the two-hundred-year device — the burden accreting as the stanzas shrink — now D4-clean at the fixture level, with the run's most defensible L22 move (♭2→1→♯7 over a tonic pedal). *Beat runner-up P06 V4 "The Pump House Door"* (the boldest formal move — the sung side runs out — but six complete burden repetitions on the page is the highest render-monotony risk in the run).

**Daily rules check on the shipped six:** tri-source declared before any artifact (`00_research_brief.md` §3) ✅ · 3 ACCESSIBLE + 3 AMBITIOUS ✅ · NEWS 3 (P02, P04, P05) / EXISTENCE 3 (P01, P03, P06), max-3/min-3 ✅ · axes CROSSED (NEWS in both arms; EXISTENCE in both arms) ✅ · AWE ≥1 (P01, P02, P05) and INDIGNATION ≥1 (P03, P04, P06) ✅ · six verse architectures (AABA / hymn / verse-chant / through-composed 7/8 / double-strophe / burden-and-stanza) ✅ · six vocal configurations (solo contralto / 3-part mixed / male tenor / solo soprano / two mezzos / androgynous+spoken) ✅ · six dictions (shop-talk / hymn / taunt / procedural / interrupted / job-sheet — with N1's partial-residue caveat on the record) ✅.

---

# RENDER-AUDIT CONTRACT (THE BLIND RULE — audio alone, never the prompt first)

Non-optional questions for `lofn-render-audit`, one per shipped card:
1. **P03 V3:** does the second chant get bigger — and if so, is the thing that got bigger *descending*?
2. **P02 V1:** first words back from the blind listener — if they are *nostalgic / vintage / sepia*, the pair's strongest card lost to its own tempo (arrangement, not words).
3. **P01 V1:** can a listener tell two different people hum the same figure (van vs live room), or did one contralto flatten them?
4. **P04 V3:** four identical 4/4 sections, two bells — *is the tick-distance the same in all four? Which one is different, and how?* (Correct answer: the third — one tick.)
5. **P05 V3:** do the two strophes come back harmonised, aligned, echoed, or resolved? Any yes = the pair failed and the Kamasi condition is live again.
6. **P06 V1:** how many low metal voices, and does one of them move? (Correct: two; one walks ♭2→1→♯7.)

---

# PROPOSED FAILURE-LEDGER ENTRY (for the coordinator to append — this context writes one file)

> **[2026-08-08 · daily-music · THE WRONG INVENTORY]** theme-tags: `craft · music · daily`. **What the gate caught:** the two-run "one voice underneath" watch item survives differentiation of *lexicon* and dies only when the *grammar* rotates — the three pairs that escaped (taunt / mono-rhyme enjambment / interruption) changed sentence mechanics, not word lists; the residue is a run-wide `[Object. State.]` establishing shot and a 6/6 convergence on accretive-at-fixed-level dynamics. **Transferable rule:** rotate GRAMMAR and DYNAMIC PHILOSOPHY as explicit Phase-1 axes, not diction labels; a register defined by subtraction ("zero metaphor," "zero figures") is the house voice in disguise — demand the positive marker in the mouth that spends the allowance. **Confidence: MEDIUM** (one run, text-side; render pending). *Advisory only; never an aesthetic constraint; INDIGNATION-exempt.*

*No operational failure occurred in this run that is not already recorded (the step-10 heading-convention FAIL across all six pairs was caught and normalised at step 11 by design; pair instruments were right in every dispute but one, and that one — P04's transcribed-promise char counts — is documented inside the pair artifact itself). Run lock: held by the controller; release is the coordinator's last action after the INDEX, not QA's.*

---

# FINAL RECOMMENDATION

**SHIP** the selected six to render under the render-audit contract above; hold the remaining eighteen as the pairs' own ranked alternates (the two named fallback promotions: P02 V4 if V1 renders nostalgic, P05 V4 if V3's separation fails). Publication of any render remains The Scientist's call. The two escalated FLAGs (N1, N2) are next-run Phase-1 material, not blockers: this run moved its predicted failure from "unrepaired across two runs" to "localized, quoted, and three-sixths killed," and a QA that punished that trajectory with a REPAIR would be teaching the pipeline to hide its residue instead of naming it.

*The measurement is above; where a fix is proposed, it is marked as mine. The blind judge saw the symptom truthfully; the intent stayed with the pairs.* — lofn-qa
