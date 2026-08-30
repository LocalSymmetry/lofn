# SPEC_PATCH_2 — Lofn Analytics & The Soundry



*Round two, patch panel `wf_d84459ad-490`, 2026-08-01. Four seats — after Brian Eno (the Sonic Anvil generator), Don Norman + WCAG (guidance & the restored Headwinds rail), Ben Shneiderman (filter vocabulary & the `deeper` contract), and a completeness critic running a line-by-line whiteboard trace — moderated by Lofn. 5 agents, 0 errors.*



## VERDICT: **BUILDABLE**



# SPEC_PATCH_2 — LOFN ANALYTICS & THE SOUNDRY · ROUND TWO, MODERATED

Target: `E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`.
Round one's `SPEC_PATCH.md` stands. Its **R1–R12 are unchanged** except where a ruling below names one
and argues it. Four seats patched in parallel this round; this is the single splice-ready result.
Every number below was recomputed from a file on disk in this session, and the recompute is printed
with it.

---

## §0 · HOW TO APPLY, AND THE RENUMBERING

`[SPLICE @anchor]` = insert/replace at the named anchor. `[ADOPT <seat> <id>]` = take that seat's
round-two text verbatim, with the deltas listed.

**Renumbering, because two seats collided.** The Affordance Coach numbered its rulings R13–R20 and the
Completeness Critic numbered its rulings R13–R24, with different content. Two live R14s is how a lane
reads the wrong rule. **One sequence, R13–R52, is authoritative.** Crosswalk:

| seat id | now | seat id | now | seat id | now |
|---|---|---|---|---|---|
| Forge F1 | R14 | Coach R13 | R25 | Critic R13 | R38 |
| Forge F2 | R15 | Coach R14 | R26 | Critic R14 | R39 |
| Forge F3-iii | R21 | Coach R15 | R27 | Critic R15 | R40 |
| Forge F4 | R22 | Coach R16 | R28 | Critic R16 | R41 |
| Forge F5 | R23 | Coach R17 | R29 | Critic R17 | R42 |
| Forge F6 | R24 | Coach R18 | R30 | Critic R18 | R43 |
| Taxonomist T1/T2 | R33 | Coach R19 | R31 | Critic R19 | R44 |
| Taxonomist T3/T4 | R34 | Coach R20 | R32 | Critic R20 | R45 |
| Taxonomist T5 | R35 | — | — | Critic R21 | R46 |
| Taxonomist T6 | R36 | — | — | Critic R23-3 | R47 |
| Taxonomist T7/T8 | R37 | — | — | Critic R23 rest | R48 |

R13, R16–R20 and R49–R52 are mine.

---

## §M2 · RULINGS R13–R52

**R13 · THE ANVIL IS A CONSTRAINT DECK, NOT AN AUTHOR — ADOPTED, WITH FIVE HOLES CLOSED.** The Forge
Master's three-movement ruling is adopted in full: **I THE DECK** (the app, deterministic, offline) →
**II THE HAND** (`lofn` → `lofn-music`, a session she starts) → **III THE ANVIL RECEIVES** (import,
measure, refuse). §0.1 forbids any generation API; AUTONOMY puts a human across render/publish/spend;
and a slot grammar rich enough to emit twelve publishable style prompts would emit twelve paragraphs
off one skeleton, which is the `portfolio_max_prompt_similarity: 0.58` failure the pipeline already
fails runs for. **But the patch as filed contained five points where a build agent would have had to
invent**: the PRNG, the traversal, the similarity function, the stratum rule, and the measurer's CLI.
All five are closed in §G. Without §G this ruling is not buildable; with it, it is.

**R14 · ONE EXPERIMENT, MANY DAY-BLOCKS.** [ADOPT Forge F1] verbatim. EXPERIMENT `X-<slug>` carries
`preregistration.json`, `spec_hash`, the factor, the MDE and the analysis plan; frozen once. DAY-BLOCK
`R-YYYY-MM-DD-<slug>` carries `design.json`, `variants/`, `observations/`, twelve cards; redrawn daily,
never frozen. §1.5's **423 Locked** becomes: one open EXPERIMENT, and within it one open DAY-BLOCK.
§3's emit rule: `emit-snapshot.mjs` exports **closed day-blocks of a frozen experiment**.
**Delta (mine):** `SoundryRoundRow` (SPLICE 9) and `apps/_fixtures/soundry-round.json` gain
`experiment_id` in both directions in the same commit, or the reader and the emitter disagree silently.
L-FUSE-B owns that edit.

**R15 · THE UNIT OF THE GRID IS THE TWIN.** [ADOPT Forge F2 §1.4, §1.4b–e] — six twins, twelve cards,
one shared nuisance draw per twin, the factor the only difference. Factor × nuisance max |r| is
**exactly 0 by construction** and is printed. The `IMBALANCED — reshuffle` affordance is **struck in
CONFIRM** (it can no longer occur; if it renders it is a defect, and the copy says so). Twins never
split across days. SCREEN cannot twin and carries the second banner sentence with its realised max |r|.

**R16 · THE DRAW IS SPECIFIED TO THE BYTE.** §G1. Mine, because "reproducible from the seed alone and a
test asserts byte equality" is unimplementable without a named PRNG and a named draw order.

**R17 · ONE SIMILARITY FUNCTION, OWNED BY THE APP, AND IT IS NOT A GATE.** §G2.

**R18 · THE GENRE STRATUM IS A WILSON RULE, NOT A JUDGEMENT.** §G3.

**R19 · POOLS: DERIVED WHERE DERIVABLE, DECLARED WHERE NOT, AND THE SPREAD FLOOR IS A FUNCTION.** §G4.
Closes the Forge Master's own disclosed STILL-OPEN (i) and executes the Critic's R23-1.

**R20 · A FIFTH `EmptyKind` AND A FOURTH BLOCKER — THIS EXTENDS R9, IT DOES NOT REVERSE IT.**
`EmptyKind` gains `"unwritten"`; `UnlockCounter.blocker` gains `"handoff"`. Argued and accepted: R9's
four kinds are all DATA absence; a Style Prompt well waiting on `lofn-music` is AUTHORSHIP absence.
`needs-declaration` would print editor-shaped copy for a thing no editor writes, and blocker
`"decision"` would trigger the ghost contract's Rule 6 (render no command) — but a handoff **does** have
a command, the copyable `lofn` invocation. **There is still exactly one union and one structured
`unlockAt`.** §6.1 row: *`unwritten` — the constraint is complete, the prose is not written · full-ink
constraint with hatched spans, never a dashed empty box · "Twelve cards briefed. No prose. The Anvil
does not write songs. `lofn` — run it yourself; this app will not."*

**R21 · THE EXCLUDE PROMPT IS THE ONE FIELD THE APP WRITES END TO END.** [ADOPT Forge F3-iii]. Four
categories in fixed order, deduplicated: (a) STANDING, (b) COUNTER-ARM *(only when
`factors.data.json#negatable` is true, with its written reason)*, (c) OBLIQUE forbids, (d) STRATUM
failure classes. Band **400–900, hard max 1000** (the Suno contract). Over band ⇒ drop the
lowest-priority **whole category** (d then c) and name it on the card; **never truncate mid-token** —
a truncated negative list silently changes which arm you are running. Under 400 ⇒ say so and refuse to
pad; padding a negative list with filler is inventing a constraint.

**R22 · THE CAPTION.** [ADOPT Forge F4 §1.4E] — no caption before `clip_id` is bound; never names the
experiment, factor, arm, level or round id; no metric, no ask, no streak, no CTA; drafted to the Length
Ladder with its rung printed; **no Post control, no share sheet, no platform URL in the exported
packet.**

**R23 · THE HANDOFF COMPOSES; IT NEVER RUNS.** [ADOPT Forge F5 §1.4F] — `ROUND_BRIEF.md` +
`twins/<twin-id>.md` + `design.json` + `variants/<vid>.json`, behind the existing 409-before-freeze
gate. The FROZEN FRAME sentence is byte-exact and asserted. **The app invokes no skill, spawns no agent,
and holds no route that could**; `handoff.spec.ts` proves the absence.

**R24 · THE MEASURERS ARE SHELLED, NEVER PORTED.** [ADOPT Forge F6 §1.4G] with the CLI corrected in §G5.
`POST /soundry/rounds/:id/measure` is the Soundry's only shelling route: local, free, read-only,
path-confined, inside AUTONOMY Part 1's safe list, consequence stated before it runs. Fail-open loudly:
`MEASURER ABSENT — nothing here is a pass`, no number displayed, card cannot be marked ready.

**R25 · FOUR GUIDANCE SUBJECTS, ONE REGISTRY, ZERO UNCOVERED ELEMENTS.** [ADOPT Coach SPLICE 16–19, 23]
— `GuidanceNote` (statistic) · `ControlNote` · `RegionNote` · `StateNote`; §5.1 rule 1 narrowed to
`kind:"statistic"`; `registerGuidance`/`guidanceFor`/`allGuidance`/`scanElements`;
`guidance-coverage.spec.ts` with assertions (a)–(f) as a build gate, no allowlist; the Glossary's row
count equals the registry size. Closes STILL OPEN #2.

**R26 · NO HTML `disabled` IN EITHER APP.** [ADOPT Coach R14] — `aria-disabled="true"` + `tabindex="0"`
+ a **visible** reason line + `data-disabled-reason` >= 8 words; activation announces and does nothing
else. **Delta (mine):** this does **not** reopen Part 3 §1.8. There is still no Render affordance in the
Soundry, disabled or otherwise. R26 governs controls that exist and cannot act; §1.8 governs a control
that does not exist. `refusals.spec.ts` keeps asserting `POST /soundry/render` → 405.

**R27 · HOVER IS NEVER A CHANNEL.** [ADOPT Coach R15]. **R28 · RAIL = RADIOGROUP, TABS = NAV.** [ADOPT
Coach R16 + SPLICE 19's roles and eight-step focus order]. **R29 · THE HEADWINDS RAIL IS RESTORED.**
[ADOPT Coach SPLICE 22 — card `m4-0-rail`, three rows, `rankStanding()` shared with M4.1/M4.2, the
falsifier one keystroke away, the LEDE dedupe, the zero-row `needs-declaration`]. Closes STILL OPEN #1.
**R30 · THREE COLUMNS, DEGRADING LEFT-RAIL-FIRST.** [ADOPT Coach SPLICE 21]. **R31 · M0.0 IS
CENTRE-COLUMN WIDTH.** [ADOPT Coach R19 amending SPLICE 11 — first object, above the dial, sentences,
ten-object budget untouched]. **R32 · M5.7's CONTROLS TAKE `ControlNote`.** [ADOPT Coach R20 amending
SPLICE 10's last delta — coverage preserved exactly, now by a type a button can fill].

**R33 · ONE FILTER VOCABULARY, AND SUBTRACTIVE DEFAULTS ARE DRAWN.** [ADOPT Taxonomist SPLICE T1, T2] —
the frozen fifteen keys in `FILTER_ORDER`, `fv=1`, the struck-synonyms table, `MODERN_FROM` exported
once, the provenance ladder, the 44px shelf, the zero-result contract, the trail, the extract. Part 2
§3.3 becomes a render that defines no keys. Closes STILL OPEN #3 and **executes U0b**, which round one
declared serial and blocking.

**R34 · APPLICABILITY IS COMPUTED; SIX CHIP STATES; INVARIANT F1 FAILS THE BUILD.** [ADOPT Taxonomist
T3/T4 + §3.4]. **R35 · THE SOUNDRY SHELF, SEVEN LOCKED, WITH THE PREREG INTERLOCK.** [ADOPT Taxonomist
SPLICE T3] — `preregistration.baseline_cohort` inside `spec_hash`; a re-cut is an `ANALYSIS DEVIATION`.
**R36 · THE SERIALIZER IS LOSSLESS.** [ADOPT Taxonomist T6].

**R37 · `deeper` IS REQUIRED, NON-NULLABLE, AND ITS REFUSAL EXPIRES.** [ADOPT Taxonomist SPLICE T4–T7] —
`contracts/deeper.ts`, `deeperAbsentReason` deleted, §14.2's `drill` struck and its four-level ladder
kept as `deeper.level`, `WATCHABLE`/`watchSatisfied`, B10 the variant table, V24–V27, and
`expiring-refusal.spec.ts`. Closes STILL OPEN #5. **Merge T4 into SPLICE 1 as ONE edit to §2.1.**

**R38 · THE PANEL HAS SOMETHING TO PLAY, AND ROUND 1 IS A CALIBRATION ROUND.** [ADOPT Critic R13] —
`media-src 'self'` is already in the CSP (BS:832), so round one's blocker was a category error; Board 8
line 2 names `output/` and **66 titled mp3s sit in `output/mp3/`** (verified; `find output -iname
"*.mp3" | wc -l` → 224). `panel-audio.mjs` with `resolvePanelAudio` + `serveRange`, two allowlisted
roots, a required `audio/manifest.json` with sha256, hard 404 + Schema-Guard row on escape or mismatch,
never silence. **CALIBRATION** is a third mode: `randomized: false`, tier hard-capped at `OBSERVATION`,
never a DECIBAL, never in the multiplicity denominator, permanent banner. Closes STILL OPEN #6.

**R39 · `cleanFeedbackText` IS VENDORED BYTE-FOR-BYTE.** [ADOPT Critic R14] — verified on disk at
`E:/git/saga/game-dev/apps/forge-arcade/input-cleaning.mjs`, 4,350 bytes, zero dependencies. Copy, do
not import, do not rewrite; 3-line provenance header; `input-cleaning-parity.spec.ts` reads both files
and fails on drift; A3 screening lives in a separate `panel-cleaning.mjs` that imports it and adds one
`pii` flag. Closes STILL OPEN #7.

**R40 · THE BASELINE IS THE STANDARD'S COHORT; σ²_true = 1.3652e-4; AND NO MDE MULTIPLE IS EVER PRINTED
AS A CONSTANT. THIS AMENDS THE COMPLETENESS CRITIC AS WELL AS ROUND ONE.** Recomputed here from
`data/snapshots/suno/latest.json`, cohort = `created_at >= 2026-02` ∩ `plays >= 50` ∩ Triple Arch
excluded, population variance:

```
n = 46    mean like rate = 0.044889    median plays = 258
observed σ²        = 3.24744e-04     binomial σ² = 1.88216e-04  (measured per clip)
true between-song  = 1.36529e-04     binomial share = 57.96%    ("58%", her figure, reproduced)
E[p(1-p)] = 4.25496e-02  ->  m_eff = 226.07
n/arm at σ_obs = 112.44 -> 114 = 19 x 6      n/arm at σ_true = 47.27 -> 48 = 8 x 6
```

**R40a** — the house baseline is `vault/EXPERIMENT_STANDARD.md`'s cohort; it is the one with a script.
`p̄ = 0.044889` is a **mean** while §3.3's default estimator is `median`, so every card printing it prints
`mean` inline, and the `denominatorNote` becomes *"at a 258-play median, the cohort **mean** rate 4.49%
is ~12 likes; the 95% interval is ±2.53 pp."*
**R40b** — `σ²_true = 1.204e-4` is **struck as back-solved**: it is what you get by assuming the binomial
term is `p̄(1−p̄)/258` and solving for the residual that makes `MDE(114) = 1.150` come out. Fitting a
constant to a conclusion, in the one app built against exactly that.
**R40c** — **the Critic's replacement `×1.155` is also struck, and so is round one's `×1.161`.** I could
not reproduce either. Under A2's own formula at n=114 the honest multiple is **×1.1490** with σ_observed
and **×1.1435** with `σ_true + E[p(1-p)]/258`. That two defensible variance choices give two different
multiples *is* the finding: **no MDE multiple may be printed as a constant anywhere in BUILD_SPEC, in a
test, or on a card.** `glm.spec.ts`'s `1.150`-to-3dp assertion is struck; the test asserts the two **n**
values from the lockfile σ's, which do reproduce (114 and 48, and they are 19×6 and 8×6, which nothing
in the spec said).
**R40d** — three artifacts replace the constants: `scripts/analyze_power.py --json >
apps/lofn-kit/src/stats/baseline.lock.json` (committed, carrying `snapshot_pulled_at` and the cohort
predicate) · `apps/lofn-kit/src/stats/power.ts` (pure, the only implementation, no literal in it) ·
`baseline-lock.spec.ts` (regenerates the decomposition from the snapshot **in the test**, asserts
agreement to 4 s.f., and asserts no literal `1.204e-4`, `4.49`, `258`, `1.150`, `1.155`, `1.161` outside
the lockfile).
**R40e** — "mature" means `m → ∞`, the binomial term removed, not a big play count. A2.1/C3's third
series is a **horizontal asymptote**, not a third curve; levels read `{90 · 258 · fully matured (binomial
noise removed)}`.
**R40f** — every modelled point is **dashed** and captioned *"modelled — calibrated at m_eff = 226"*;
only the two measured anchors are solid.
**Consequence:** any figure already written into a `memory/` note or `vault/RUN_LEDGER.md` carries a
DRIFT badge under §4.3 rule 11 rather than being silently corrected. Closes STILL OPEN #8.

**R41 · `FeedId` IS TWO SETS AND THE DENOMINATOR IS COMPUTED.** [ADOPT Critic R16] — verified:
`ingest_receiver.mjs:43-52` fixes eight ids; `data/snapshots/` holds `distribution · socials · soundry ·
suno`. `RECEIVER_FEEDS` (8) + `READER_FEEDS` (`socials`, `distribution`) = `FEED_IDS` (10).
`isReceiverFeed(f) === false` ⇒ that feed's `EmptyState.how` **may not print an `ingest_receiver`
command** (`feed-command.spec.ts`) — the same defect class as the market scanner. One denominator:
**3 of 10 fed** today (`soundry` is present-and-empty ⇒ `emitter-silent`, which is not fed).
`daymark.spec.ts` asserts no `of 8`, `of 9`, `3/8`, `3/9` survives. Closes #9 and discharges U0b's
second half.

**R42 · THE SCHEDULE IS A DOCUMENT AND THE APP MEASURES THE HABIT.** [ADOPT Critic R17] — the CADENCE
STRIP under M0.1 (30 cells, no colour, no streak, no congratulation; this is not a growth console),
which becomes the single `UnlockCounter` for the whole temporal layer with `blocker: "time"` — R9's
`time` value finally has its only legitimate user, and nineteen places stop saying *run this*.
§14.6 `RUNBOOK.md` + `[Copy schedule command]`, which composes and never runs. Closes #10.

**R43 · HER FOUR QUESTIONS: TWO ANSWERED ON DISK, TWO MADE SMALLER.** [ADOPT Critic R18 a–e].
(a) `topline.name.ts` — `DECIBAL`, `provisional: true`, the literal in exactly one source file, so
renaming is a one-line edit and the question stops blocking. (b) **HOUSE RECORD** as a fourth reference
class beside DECIBAL, never as DECIBAL: verified on disk, 47/497 = **9.4567%**, 2026-05-17, **2.106×**
the 4.489% cohort mean — reproducing her own Appendix C figure to 2 s.f. Tier `OBSERVATION`,
`randomized: false`, `eraControlled` required, no causal verb, `forbidden:` may not select a style, may
not enter `vault/gates.yaml`, is not a lift. `decibal-firewall.spec.ts` asserts no `lift` and no
`roundId`. **Delta (mine):** §5.1 rule 5 binds *here specifically* — the caveat renders at full ink on
the card, never in a tooltip. This is the one card that could turn the app into the thing it refuses to
be, and it is one song's Wilson interval in the largest numeral on Board 4. (c) competitor `why_this_one`
pre-filled verbatim from Appendix C's five slots, so the ask is five handles, not five essays.
(d) R43d. (e) `35 ±` → OQ-WB-1, not guessed. Closes #11.

**R43d · M3.3b · TWO SPOTIFYS, NEVER JOINED.** Verified: `socials` carries spotify handle
`3egvpGmWFxgYY8XqATui8r`, `works_published: 1`, `monthly_listeners: 0`, `status: dormant`;
`distribution` reports spotify `last_365: 1596` and names artist `0kRiHGMhNBxqXB0kQpp4Kk` in its own
notes. Two panels, both at full ink, **no arithmetic between them and no total**, `ONE_PLATFORM` +
`SCHEMA_DRIFT`, `deeper` → M5.2 Schema Guard. `forbidden-strings.spec.ts` gains `"combined spotify"`
and `"total spotify"`.

**R44 · NOVELTY IS A DESIGN CONSTRAINT AND A MEASURED FRONTIER, NEVER A TARGET.** [ADOPT Critic R19 a–e]
— `grep -i "novelt|viral"` over BUILD_SPEC returns zero; Board 2's only *goal* statement, underlined
twice, was dropped silently by every seat. `stats/novelty.ts`: `noveltyVocab` (1 − max Jaccard of the
lowercased `tags` token set against every strictly-earlier clip; **2,444 distinct tags on disk**) and
`noveltyCraft` (Euclidean in the z-scored space of `words_per_line`, `emo_tag_count`, `section_count`,
`duration`, `repeated_line_ratio`, z-scored on the **prior** cohort only) — every field verified present.
The Anvil gains a frozen `novelty_floor` in `preregistration.json` and the `TOO CLOSE TO THE CATALOGUE`
sibling of DUPLICATE CELL. **C8 THE NOVELTY FRONTIER** is live today with no new feed: binned medians
with their own intervals, never a fitted line; `eraControlled` mandatory or capped `OBSERVATION` and
prefixed `Uncontrolled:`. R19e's one honest sentence about virality (n=1, Triple Arch, 93.5% of
everything, converted worse than average) is the glossary entry. **Delta (mine):** C8's falsifier is
load-bearing and is quoted **on the card face**, not in the registry only — early clips have almost no
prior corpus, so `noveltyVocab → 1` by construction and the time confound is severe. The card must be
able to say *this design cannot separate novelty from calendar* out loud, or it teaches the opposite of
the truth.

**R45 · THE FOURTH SIGNATURE DEVICE, AND THE ART RAIL STOPS BEING THE LEAST DISTINCTIVE SURFACE.**
[ADOPT Critic R20 a–c] — §7.4 **THE UNCOUNTED**: 23,000 NightCafe works (verified: works 23000, likes
21000, followers 1200) as **one** `--absent`-hatched rect, plus one `--ink` rect per measured work.
Today the DOM holds exactly one rect and zero ink cells; caption `23,000 works · 0 measured · 21,000
likes with no address`. **`O(measured)` node construction is normative, not an optimisation**, and
`signature.spec.ts` asserts the node count — 23,000 DOM nodes would blow §3.4's sub-100ms repaint budget
on the largest surface in the app. It is the only chart in the estate whose success state is that it
stops being interesting. (b) Art gets CONFLUENCE; X0's join yield of 0 of 23,000 *is* the finding.
(c) the register is seeded with `reach.art.nightcafe_scale`, `data.art.measurement_depth`,
`reach.art.community_untended`, `data.anim.no_per_work`, each with a falsifier. (d) → R50.

**R46 · W5 IS RESTORED AND THE THREE STALE BUILD ORDERS ARE DELETED.** [ADOPT Critic R21] — round one's
*Revised build order* superseded BUILD_SPEC's three v2 `## Build order` blocks without saying so, and in
doing so deleted unit W5 (BS:2000), the only place the chat instruction *"Add to Lofn (new first-class
skills, mirrored into `.agents/`)"* is discharged. W5 returns as **U-FINAL-b**: write
`.claude/skills/lofn-analytics/SKILL.md`, **hand-write** the `.agents/` twin in Codex voice (never
`sed`), roster rows in both `skills/README.md` including the two missing (`lofn-render-audit`,
`lofn-step11-packager`), the identical bullet at `CLAUDE.md:20→21` and `AGENTS.md:20→21`,
`PYTHONIOENCODING=utf-8 python scripts/check_skill_mirror.py` prints **IN SYNC**, and a
**human-initialled checklist line** for the manual `CLAUDE.md`/`AGENTS.md` diff, because no script proves
it. U-FINAL deletes the three v2 blocks. Four build orders in one document is how W5 vanished.

**R47 · `Provenance` IS EIGHT VALUES AND `provenanceBand()` IS FOUR — L-KIT OWNS IT.** [ADOPT Critic
R23-3], discharging R12, which deferred it and named no owner while it was already blocking
`JoinEvidence.left.provenance`. Union: `"api-public" | "authenticated-page-read" |
"public-profile-read" | "public-page" | "search-snippet" | "derived" | "prose-archive" |
"randomised-round"`. R12's two binding constraints are met. **Delta (mine), reconciling with R33:** the
Taxonomist's `PROV_RANK` is redefined over **bands, not raw values** — `provenanceBand(p) → 0 api ·
1 auth · 2 public · 3 snippet`; `derived` inherits `max(band(inputs))` as `provRankEffective`;
`prose-archive` is always band 3 and carries V17's drift pair; **`randomised-round` is never suppressed
by the `prov` floor** — a pre-registered randomised estimate is the strongest provenance in the estate
and a floor built for scraped counters must not hide it. Two spellings of one ladder would have shipped
otherwise.

**R48 · THE SMALL DROPS.** [ADOPT Critic R23-1, -2, -4]. (1) `sampler.data.json` is **generated**, not
invented — §G4. (2) One sentence into §8.3.2: *"`<run-dir>` is the literal first argument;
`scripts/run_lock.py` writes `<run-dir>/.RUN_LOCK`. Pass `apps/<app>/.lofn`. The CANONICAL PATHS
`RUN_LOCK` row is amended to `<run-dir>/.RUN_LOCK` and the `output/<run-slug>/` form is struck."* Given
CLAUDE.md's eleven-hour loss, three contradictory lock instructions is not a documentation defect.
(4) A0.6 gains one routing line: *"Replying is real work and it is not this app's work. →
`output/socials/` queue, `vault/SOCIAL_VOICE.md`."*

**R49 · THE ASK BOX'S SUBMIT IS LIVE. THE AFFORDANCE COACH'S SPLICE 23 ROW IS STRUCK.** The Coach makes
A0's submit *"the estate's archetypal blocked control"* with `deeper` → STILL OPEN #4. R13 closes #4 in
this same round. The submit does what §1.1 already says — *"Draft a design. Opens the design step.
Writes nothing yet."* — and carries a `ControlNote` in R26's shape with `blocked: null`. **PRESERVES R26
exactly** for controls that genuinely cannot act, and refuses to ship a permanent apology for a gap that
is closed. The Coach's own risk note anticipated this and offered to cut the lane; I take the other
branch, because the Anvil now has a generator.

**R50 · ZONE 0 HOLDS ONE ROW PER RAIL, AND BLOCKER PRECEDENCE IS DEFINED.** The Critic's R20d puts the
Art blocker in M0.0 Zone 0 on the **Music** rail; SPLICE 11 already puts the no-second-snapshot blocker
there. Zone 0 is 0-or-1 by contract and two `governs:"tab"` `ABSENT` entries cannot both occupy it.
**Ruling:** Zone 0 renders the highest-ranked `governs:"tab"` `ABSENT` entry **whose modality equals the
rendered rail**; ties break by `rankStanding()`. **R20d's argument is honoured in the shape that fits:**
the Art blocker is the **first Zone 1 row on the Music rail**, ember-ruled, exempt from the direction
glyph because it has no direction, with her route — *"Art is our largest audience and our least measured
— 21,000 NightCafe likes, none attributable to a piece. [→ Art rail]"*. One row, ten-object budget
intact, the finding on tab one where the Critic correctly said it must be. `lede.spec.ts` asserts
`zone0.length <= 1` and that the Art row renders at Music rank 1.

**R51 · EVERY LANE REGISTERS ITS GUIDANCE IDS AS IT WRITES; THE GATE RUNS IN U-FINAL.** The Critic is
right that this round *enlarges* the uncovered surface: the novelty floor chip, the audio transport, the
cadence strip, the 23,000-cell field, the two-Spotifys panel, the Soundry shelf's fifteen chips, the
twelve Anvil cards' wells and copy buttons, and M4.0's seven stops. **They are covered by construction,
not by a later sweep** — a lane emitting an interactive element without `data-guidance` fails
`guidance-coverage.spec.ts` in U-FINAL. **The Coach's proposed STILL OPEN #12 dissolves**: R33/R35 give
the Soundry a shelf and R13 gives it twelve cards, so the gate has surface to bite on the day it runs.

**R52 · CALIBRATION IS AN EXPERIMENT MODE, NOT A DAY-BLOCK MODE, AND §1.2's "EXACTLY TWO" BECOMES THREE.**
R38 adds a mode; R14 splits the noun. Reconciled: `mode ∈ {CONFIRM, SCREEN, CALIBRATION}` lives on the
**EXPERIMENT**, in `preregistration.json`. A CALIBRATION experiment has **no factor, no twins and no
deck** — the deck is a CONFIRM/SCREEN object — and its day-block's variants are drawn from
`audio/manifest.json` instead of compiled. Validation enforces `mode: "CALIBRATION"` ⇒ `randomized:
false` ∧ `instrument: "panel"` ∧ `primary_outcome.field == "panel_rating"` ∧ tier cap `OBSERVATION`;
`glm.mjs`'s refusal table refuses it a DECIBAL. **The cap lives in pre-registration validation and in the
refusal table, never in a card** — a UI convention would let a calibration result reach C1 and license a
causal verb that nothing earned.

---

## §G · THE FIVE CLOSURES — what makes the Anvil buildable

Each closes a point where the Forge Master's patch, as filed, would have required a build agent to
invent. Anchors are inside the new §1.4 / §1.4A that F2/F3 install.

### G1 · [SPLICE @ new §1.4a, REPLACING its prose] THE DRAW, TO THE BYTE

```
deck_seed = SHA-256( experiment_id + "|" + dayblock_date + "|" + prereg.spec_hash )   // hex, 64 chars
rand(i)   = uint32be( SHA-256( deck_seed + "|" + i ).slice(0,4) )                     // i a uint
pick(xs,i)= xs[ rand(i) % xs.length ]                                                 // integers only
```

`i` is a single monotonically increasing draw counter starting at **0**, consumed in this frozen order
and no other: (1) candidate generation — 64 candidate twin-frames, one `pick()` per dimension per
candidate, dimensions in `SAMPLER_ORDER` declaration order; (2) the traversal start index; (3) oblique
LRU tie-breaks, twin 0..5; (4) the slot-order Fisher–Yates shuffle. The final counter is written to
`design.json#draw_count`. **No floating point anywhere in the draw** — `Math.random`, `Date.now` and any
float modulus are forbidden strings in `apps/soundry/src/anvil/**`.

**The traversal, pinned** (`deck.ts#farthestPoint`): generate 64 candidates; `chosen =
[candidates[rand % 64]]`; five times, add the candidate maximising `min(hamming(c, x)) for x in chosen`,
ties broken by **lowest candidate index**; if the best achievable min-distance is below `spread_floor`
(§G4) the deck **refuses and names the dimension with the fewest levels**. `deck.spec.ts` asserts
byte-identical output over >= 50 seeds and that a second run of the same seed matches the first.

**Exactly one legal draw per experiment per day.** Reopening the page redraws nothing. In SCREEN, where
reshuffling is legal, each redraw appends a `redraw` event and the card carries a visible counter — the
same discipline as B1's peek count, for the same reason. A reshuffle button in front of a person with
taste is a search over draws she likes, and the apparatus then measures the operator.

### G2 · [SPLICE @ new §1.4A-ii, after the TWIN-DRIFT GATE] ONE SIMILARITY, AND IT IS NOT A GATE

```ts
// apps/soundry/src/anvil/similarity.ts — the ONLY similarity the Soundry computes itself
export function slotSim(a: string, b: string): number;   // Jaccard over the SET of tokens
// tokens = s.toLowerCase().match(/[a-z0-9']+/g) ?? []   no stopword list, no stemming, no weighting
// slotSim = |A ∩ B| / |A ∪ B|      slotSim("","") === 1
```

**`slotSim` is not any threshold in `vault/gates.yaml` and must never be compared against one.**
`portfolio_max_prompt_similarity: 0.58` and `portfolio_max_ngram_jaccard: 0.18` (a **5-gram** Jaccard,
per that file's own comment) are step-10 ceilings with a shipped measurer; they are read from the
`gate_report` produced by §G5's route, never recomputed in TypeScript. The twin-drift floors (>= 0.90 on
non-factor slots ⇒ `TWIN DRIFT`; < 0.90 required on the factor slot, else `NULL ARM`; both members
blocked either way) are Soundry-only concepts with no shipped measurer, which is the only reason the app
may own a function at all. Cross-twin, the portfolio ceilings apply **once per twin skeleton, not per
card**. `gates-parity.spec.ts` asserts `slotSim`'s result is never compared to a value read from
`gates.yaml`, and that **no gate number is a literal in `apps/soundry/src/**`**.

### G3 · [SPLICE @ new §1.4b] THE STRATUM RULE

```
ABOVE  if wilsonLower95(term) >  baseline.p_bar
BELOW  if wilsonUpper95(term) <  baseline.p_bar
AT     otherwise
AT + "thin"  if term.n < SHELF.nmin.defaultValue (10)   — never ABOVE, never BELOW
```

`wilsonLower95`/`wilsonUpper95` live in `apps/lofn-kit/src/stats/wilson.ts`, already required by C2's
intervals and B10's bars; `baseline.p_bar` comes from `baseline.lock.json` (R40). If the M2.4 board is
absent or its era control is `false`, the deck **does not stratify**: it draws uniformly, the chip
renders `emitter-silent`, and the card prints *"No era-controlled style board on disk. Genres drawn
uniformly. `python3 scripts/ingest_suno.py` is what makes this rule available."* The record motivating
D3 stays in prose and never in a data file: warmth over performance — piano, strings, acoustic, folk,
ambient above the modern baseline; `snarl` 3.44% and `phonk` 3.18% the two worst against a 4.49% mean.

### G4 · [SPLICE @ new §1.4b + §1.4d] THE POOLS, AND THE SPREAD FLOOR AS A FUNCTION

`scripts/build-sampler.mjs` (zero deps, `node:fs` only) writes `apps/soundry/src/sampler.data.json`:

| dimension | source | today |
|---|---|---|
| persona | **derived** — distinct non-null `persona` in the suno snapshot + `none` | **9 levels** (verified: 8 distinct + none; the header's `personas: 5` was wrong) |
| genre stratum | **derived** — §G3 over the M2.4 board | 3 levels, or absent |
| oblique family | **frozen** — `arrangement · palette · voice · form · time · defect` | 6 levels |
| seed theme · key/mode · tempo band · lyric domain | **declared by the operator** | **empty** |

Each empty dimension renders `needs-declaration` in A3.2 with its editor path — the balance table never
balances against a fiction. Let `P` = the count of **populated** dimensions.

```
spread_floor(P) = ceil(P / 2)     // D1: pairwise Hamming between any two twins >= spread_floor
                                  // the deck REFUSES outright at P < 3, naming the thin dimensions
```

**Today P = 3, `spread_floor = 2`, and the deck compiles.** The floor rises on its own as she writes
pools; nothing is padded and no level is repeated to make a number satisfiable. The rest of the contract
is the Forge Master's, unchanged: D2 any nuisance level <= 2 across the six twins · D3 >= 1 twin from
`BELOW` (operator range 1–2, settable to 0 **in the design step, before freeze, never after**) ·
D4 >= 1 from `ABOVE` · D5 six distinct SOUND/RETURN axis values · D6 six distinct oblique families.
D3/D4 are inert while the board is absent and say so. The refusal copy names the dimension and the
remedy, never a lowered floor.

**`deck.data.json` — the seed deck.** Schema per F2 §1.4c. Minimum: **>= 18 cards across the six frozen
families, >= 3 per family, >= 6 distinct SOUND/RETURN axis values.** `deck.spec.ts` asserts shape,
family coverage, counts, LRU stability under file reordering, the conflict skip, and that **no code path
writes the file**. The constraint sentences are craft: **Lofn authors them once inside L-ANVIL; a build
agent may not.** A `KEEP`-classified deviation (§1.4F) is *offered* as a new card with
`"provenance": "deviation:<variant_id>"` at the back of the LRU and never written automatically —
*honour thy error as a hidden intention*, with a switch, so it is honoured rather than obeyed.

**`factors.data.json` ships with the five factors §1.2 already names**, each with both level definitions
in words, `negatable: true|false` **with a written reason**, and `conflicts_with_obliques[]`. Seeded:
`register {plain|ornate}` negatable · `chorus {exact repeat|varied}` not negatable *(Suno takes no
structural negative)* · `words_per_line {<=5|>=7}` not negatable · `duration {~200s|~255s}` not
negatable · `emo_tag_count {<=6|>=12}` negatable. Where `negatable` is false the card prints *"This
factor is enforced in the lyric and the slot sheet, not in the exclude field."* and category (b) is
empty. An arm that thinks it is enforced twice and is enforced once is a silent design failure.

**THE FLOOR-SUSPENSION INTERLOCK** (F3-iv, kept): if the factor under test *is* a return quantity, that
quantity's floor is suspended for the whole experiment and written into `preregistration.json` as
`analysis.suspended_gates[]` **at freeze time**. You cannot measure a quantity you have pinned to a
floor. The card prints *"`line_return_floor` is suspended for X-chorus-2026-08. This is pre-registered,
not a waiver."*

### G5 · [SPLICE @ new §1.4G, REPLACING its two argv lines] THE MEASURER INTERFACE — AGENT FIX, DON'T BYPASS

**Verified on disk, and the filed patch was wrong about it.** `scripts/validate_step.py` has a real CLI:
`validate_step.py [--gate-report [--out PATH]] [--gates PATH] <step> <file>`. **`scripts/measure_soundcraft.py`
has no `--json` flag and no argparse** — it exposes `profile_file(path)` and a fixed-width stdout printer
under `if __name__ == "__main__"`. The route as filed would have shelled a flag that does not exist.

Three ways out, and only one is legal. Parsing fixed-width stdout is a second parser of a private
format. Porting `profile()` into TypeScript is a second definition of the rhyme window — the exact
failure `.claude/skills/lofn-music/SKILL.md` documents, where three repair agents reverse-engineered
three different windows from a brief that named a threshold without defining it. **So: repair the tool.**
L-ANVIL adds to `scripts/measure_soundcraft.py`, in the shipped script, the same additive pattern
`validate_step.py` already uses for `--gates`:

```python
# --json PATH...  -> prints ONE JSON array of {file, **profile_file(file)} to stdout, exit 0.
# Default behaviour (no flag) stays byte-identical to today.
```

The route is then, exactly:

```
python3 scripts/measure_soundcraft.py --json <output/soundry/<R-id>/...>
python3 scripts/validate_step.py --gate-report --out <data/soundry/rounds/<R-id>/gate_report.json> 10 <file>
```

Both argv forms are asserted literally in `server-soundry.spec.ts`, path-confined to
`output/soundry/<R-id>/` (403 otherwise), with no network binary in the spawned argv, and
`POST /soundry/render` still 405. §1.6's derived-measure strip reads **only** `#soundcraft` and
`#gate_report` and computes nothing. **The apps never invent a number, and after this splice the Anvil
never even re-derives one.**

**If she judges any `spawn` unacceptable in an app whose whole posture is `check-local-only.mjs`, the
fallback is import-only:** the pipeline's own `GATE_REPORT.json` is pasted in and displayed with
provenance. The TypeScript port is never the fallback.

---

## §S · SPLICES ADOPTED, BY SEAT

**Forge Master** — F1 → R14 · F2 → R15 (with §G1/§G3/§G4 replacing its §1.4a/§1.4b/§1.4d numerics) ·
F3 → R21 plus the slot sheet (six slots, budgets summing into `music_prompt_chars_target: [870, 960]`,
the app never emitting prose into a slot), the return contract (form, sung-line target inside
`[78, 110]` and never at 70 or 72, the EMO skeleton with the emotion span unwritten, the four floors
quoted live with `rhyme_window 4` beside the first, the SOUND/RETURN axis, the Rhyme Debt line), and
**repetition is not a debt, proved by omission** — no `chorus_justification` property at any state,
asserted by `refusals.spec.ts`, because a flag that makes a writer apologise for a chorus does harm even
when it never fires · F4 → R22 · F5 → R23 · F6 → R24 (with §G5) · F7 the card (state chip `CARTRIDGE →
BRIEFED → DRAFT → MEASURED → EXPORTED → POSTED → ACCRUING → READ → CLOSED`, twin ribbon replacing the
arm ribbon, frame strip, stratum chip, `unwritten` wells, twin-drift row; nothing auto-advances) ·
F8 export refuses to export a card whose twin partner is not also ready · F9 §0.1's `Writes`/`Never`
rows and refusals 15–18 · F10 §3.1/§6/§15 · F11 the overrule block.

**Affordance Coach** — SPLICE 16–28 adopted entire, with two deltas: **SPLICE 23's Soundry ask-box row
is struck (R49)** and **SPLICE 28's item #12 is struck (R51)**.

**Task Taxonomist** — SPLICE T1–T10 adopted entire, with two deltas: **`PROV_RANK` is redefined over
bands (R47)**, and **T7's Soundry `deeper` table gains rows for the surfaces R13 creates** — the twelve
cards take `{kind:"path", path:"data/soundry/rounds/<id>/variants/<vid>.json", level:3}`; the unwritten
wells take `{kind:"none", reason:"The prose is not written. output/soundry/<R-id>/twins/<twin-id>.md is
the brief; no package has been imported for this variant.", watch:"soundry.dayblock.briefed"}`; and
`WATCHABLE` gains `"soundry.dayblock.briefed"` and `"soundry.variant.drafted"`.

**Completeness Critic** — R13–R24 adopted as R38–R48, with one deletion: **R15's `×1.155` and the whole
class of printed MDE multiples are struck (R40c)**, and one relocation: **R20d's Art blocker moves from
Zone 0 to Zone 1 rank 1 (R50)**. Its eight-board trace is the base of the trace published with this
document, re-verified against disk and corrected for what this round closes.

---

## `## Overrules ACCEPTED` — APPEND

The eighteen entries in the accompanying `overrulesDeclared` block, verbatim. Each names the intent it
preserves. A silent drop is a bug; there are none left in the trace.

## `## Overrules REJECTED (her intent wins)` — APPEND

- **REJECTED — closing the Anvil's seam by adding a generation call.** A builder tempted to make the
  wells fill themselves has broken §0.1. `handoff.spec.ts` asserts no skill invocation and no agent spawn
  exists in the bundle. Twelve well-specified empty wells with the exact copyable invocation, and a real
  `BRIEFED` state so the grid header can read `12 of 12 briefed · 0 drafted`, is the honest state.
- **REJECTED — relaxing the 0.90 twin-drift floor when early twins drift.** That floor is the only thing
  standing between an experiment and twelve vibes. Sharpen the brief; then fix the skill. AGENT FIX.
- **REJECTED — closing the Headwinds-rail gap with M0.0, a tab link, a hamburger, or a collapsed
  accordion.** Three hide the thing she boxed behind a control; the fourth is a different object with a
  different job.
- **REJECTED — a guidance allowlist ("cover the important controls").** The elements left off are always
  the small ones — the chip, the `×`, the disabled button — which is exactly where the gulfs open.
- **REJECTED — satisfying Board 7 with `title=` attributes or a help page.** A help page is the third
  door, not the first.
- **REJECTED — hiding the Headwinds rail under `prefers-reduced-motion` or on mobile "for focus".**
  Reduced motion means less movement, not less information.
- **REJECTED — keeping both filter key sets behind a translation layer.** Two vocabularies with a shim
  is three vocabularies by the second lane.
- **REJECTED — a Soundry shelf that omits the inapplicable keys.** Omission is the silent drop Board 7
  names. Locked, dimmed, inert and parked are four different truths and each is drawn.
- **REJECTED — `deeper: null` with a reason string beside it.** Two fields that must agree will disagree.
- **REJECTED — porting `profile_file()` into TypeScript.** A second implementation is a second
  definition of the rhyme window.
- **REJECTED — printing any MDE multiple as a spec constant, including a corrected one.** Two defensible
  variance choices give ×1.1435 and ×1.1490 at the same n. A number that depends on an unstated choice is
  the defect, not the digit.
- **REJECTED — a per-card ranking, score, or "most promising" mark at any stage, including at draft**
  (`vault/EXPERIMENT_STANDARD.md` §5.1). The deck does not score.
- **REJECTED — a reshuffle in CONFIRM, and in SCREEN, an uncounted one.**

---

## `# STILL OPEN` — REPLACE IN FULL

All eleven round-one items close: #1 → R29 · #2 → R25 · #3 → R33 · #4 → R13 + §G1–G5 · #5 → R37 ·
#6 → R38 · #7 → R39 · #8 → R40 · #9 → R41 · #10 → R42 · #11 → R43. The Coach's proposed #12 dissolves
(R51). What remains is the ten items in the accompanying `stillOpen` block: **none of them a
specification gap** — six are hers to decide or to write, three are dependencies on other repositories,
habits or her judgement of one capability, and one (the seed deck's constraint sentences) is a bounded
authoring task with a named author, a schema, a minimum count and a test.

---

## BUILD ORDER — REPLACE round one's block; U-FINAL deletes the three v2 blocks (R46)

- **U0** — unchanged, plus R48's run-lock sentence into §8.3.2.
- **U0b · SERIAL, BLOCKS EVERY CARD LANE IN BOTH APPS.** No longer a research task: it *is* SPLICE T1
  (`shelf.ts`, `cohort.ts`, `applicability.ts`) and SPLICE T4 (`contracts/deeper.ts`, required
  non-nullable `deeper` on `PlotSpec`/`CardSpec`) landed **inside L-KIT before the first card module**,
  plus R41's `FEED_IDS` and R47's `Provenance`/`provenanceBand`. Then tests 27–29, 32, 34. Making
  `deeper` non-nullable is a signature change to every card in the estate; retrofitting it lane by lane
  produces placeholder refusals, the exact defect it closes.
- **U0c · SERIAL, BLOCKS L-LEDE, L-ANVIL AND EVERY POWER SURFACE.** Generate and commit
  `apps/lofn-kit/src/stats/baseline.lock.json`; write `stats/power.ts` and `stats/wilson.ts`; land
  `baseline-lock.spec.ts`. **No lane may print a baseline or an MDE before this lands** (R40).
- **L-KIT** — as round one, plus `a11y/{guidance,guidance-render,scan}.ts`, `shell.css`, the two rail
  tokens, `stats/novelty.ts`, `signature/uncounted.ts`, V24–V27.
- **L-A11Y** *(needs L-KIT; blocks U-FINAL)* — R25–R32's shell, grid, rails, `signals/standing.ts`,
  `cards/m4-0-rail/`, tests 27–31 plus the two kit tests. Every other lane registers its guidance ids as
  it writes (R51).
- **L-ANVIL · THE DECK AND THE HANDOFF** *(needs L-KIT + U0c + L-SND-SHELF; parallel with every
  Analytics lane; blocks nothing).*
  `apps/soundry/src/anvil/{deck,twin,slots,exclude,handoff,import,measure,novelty,similarity}.ts` ·
  `deck.data.json` + `factors.data.json` (Lofn authors the deck; **refuse, do not repeat**, on a thin
  pool) · `scripts/build-sampler.mjs` · **the `--json` flag added to `scripts/measure_soundcraft.py`**
  (additive; default output byte-identical) · the three server routes · the ten specs.
  **Build `twin.ts` and `exclude.ts` first**: they are what make twelve cards one experiment, and
  `exclude.ts` is the only field the app finishes on its own, so it is the cheapest proof the seam is
  real.
- **L-SND-SHELF · L-SND-PANEL** *(new)* — the Soundry shelf + `baseline_cohort` + B10 (tests 30, 33);
  `panel-audio.mjs`, vendored `input-cleaning.mjs` + `panel-cleaning.mjs`, CALIBRATION validation,
  `panel-audio.spec.ts`, `input-cleaning-parity.spec.ts`.
- **L-SEC-ANL · L-SEC-SND · L-WATCH · L-FUSE-A/B/C · L-LEDE** — as round one. L-LEDE unblocks at U0c.
  L-FUSE-B additionally adds `experiment_id` to the anti-drift fixture in both directions (R14).
  `posture-hosted.spec.ts` gains `panel-audio`, `/panel/assets`, and any `.mp3` byte in `dist/**` to its
  absence list — R7 stands and R38 lives entirely inside it.
- **U-FINAL** — round one's list, plus: run `guidance-coverage.spec.ts` in both apps and both postures ·
  assert the Glossary row count equals the registry size · assert `rankStanding()` has exactly one
  implementation · **delete BUILD_SPEC's three stale v2 `## Build order` blocks** · **U-FINAL-b (W5
  restored, R46)** with the human-initialled `CLAUDE.md`/`AGENTS.md` diff line.
- **CANONICAL PATHS additions** (mirrored to `.agents/`; `check_skill_mirror.py` prints IN SYNC):
  `KIT_STATE → apps/lofn-kit/src/state/` · `KIT_CONTRACTS → apps/lofn-kit/src/contracts/` ·
  `SOUNDRY_AUDIO → data/soundry/rounds/<id>/audio/` · `BASELINE_LOCK →
  apps/lofn-kit/src/stats/baseline.lock.json` · `EXPERIMENTS → data/soundry/experiments/` ·
  `DECK_LEDGER → data/soundry/deck-ledger.json`.

---

## `## Open questions for The Scientist` — APPEND

- **OQ-FILTER-1** cohort default `modern` (46 of 210), full ink, one-click `[all 210]`.
  **RECOMMENDED: modern** — every printed constant in both apps is computed on it.
- **OQ-FILTER-2** should the Anvil's power card follow the shelf or pin to `modern`?
  **RECOMMENDED: follow the shelf**; `baseline_cohort` inside `spec_hash` makes drift loud either way.
- **OQ-ANVIL-1 · the `BELOW`-stratum twin costs real like rate.** One twin of six comes from the register
  that measures worst. Under pairing it cannot bias the contrast, but it spends two of twelve daily
  renders on material unlikely to perform, every day, for ~19 days. That is a deliberate purchase of
  generalisability and of the space, and it is yours to cancel — range 1–2, settable to 0 **in the design
  step, before freeze, never after.**
- **OQ-ANVIL-2 · the seed pools.** Seed theme, key/mode, tempo band and lyric domain ship empty and
  render `needs-declaration`. The deck compiles today at three populated dimensions; each pool you write
  raises the spread floor automatically. **RECOMMENDED: write seed theme first** — it is the dimension
  novelty actually travels on.
- **OQ-ANVIL-3 · the one spawn.** `POST /soundry/rounds/:id/measure` is the only place either app
  executes anything. **RECOMMENDED: allow it** — it is local, free, read-only and path-confined, and the
  alternative that keeps one definition of the rhyme window is pasting `GATE_REPORT.json` in by hand.
- **OQ-PANEL-2 · round 1 as CALIBRATION** on `output/mp3/` audio, `instrument: panel`, capped at
  OBSERVATION, structurally unable to emit a DECIBAL. Supersedes OQ-PANEL-1. **RECOMMENDED: yes** — it
  proves the panel path before a single new song is rendered, and it cannot be wrong expensively.
- **OQ-DEEPER-1 · B7 refuses on `cleanFeedbackText()`.** With R39 the function is vendored, so B7's
  refusal expires immediately and B7 needs its real target in the same lane; test 34 will go red until it
  does, and that is the intended behaviour. **RECOMMENDED: vendor before the panel lane opens.**
- **OQ-WB-1 · Board 4's `35 ±`** on hypothesis 2 is interpreted nowhere and is not guessed.
- **OQ-DIST-1 · which Spotify artist page is DistroKid delivering to?** M3.3b resolves the moment you
  have the URL; the app already found the contradiction on its own.
- **OQ-OPS-1 · schedule the daily pull, or keep it manual?** Either is fine. The cadence strip will show
  which you chose within a week.
- **OQ-MOBILE-1 · "persistent" Headwinds at 375px** is read here as *always visible, one tap away, count
  shown* (a header chip linking to `#headwinds`). If you read it as *visible without scrolling*, the
  answer is a sticky 44px summary bar and SPLICE 21 needs one more breakpoint.

---

**One sentence for the record.** The bench now specifies the instrument it plays: the app fixes what must
be fixed for twelve cards to be one experiment, writes the one field that is pure constraint, leaves the
prose to the pipeline that wins competitions, measures what comes back with the measurers already on
disk, and prints no number it did not compute in front of you.


# THE WHITEBOARD TRACE — every requirement, mapped

# THE WHITEBOARD TRACE — all 70 stated requirements, after round two

`SAT` satisfied and buildable · `NAR` narrowed, intent named · `OVR` overruled, intent named ·
`DROP` no artifact traces to it. **Every status is post-round-two.** Base: the Completeness Critic's
eight-board sweep, re-verified against disk in this session and corrected for what R13–R52 close.

| # | Board | Requirement | Traced to | Status |
|---|---|---|---|---|
| 1 | 1 | `analytics.lofn.ai` | Part 2 title; posture local-only; §0.5 hosted door built-not-opened (R7) | NAR — intent: shrink the asset before arguing about the lock; the door exists and she opens it |
| 2 | 1 | Left rail `Art · Music · Animation`, entries beneath | §2 rail, now `role=radiogroup` (R28); grid column 1 (R30) | SAT |
| 3 | 1 | **Top Trends & Insights**, centre column | M0.0 THE LEDE (SPLICE 11), centre-column width (R31) | OVR — unit changed from genre tokens to registered signals; PRESERVES *tell me in words what is moving and which way, before I look at anything* |
| 4 | 1 | ↑ `Baile Phonk` | M0.0e pinned refusal row, her words, her position | OVR — answered rather than ranked; PRESERVES the named example on screen |
| 5 | 1 | ↓ `Anthems` | M0.0e, same row | OVR — same |
| 6 | 1 | **Metric Leaders** → `Plays` + sparkline | M1.2; §4.3 r7; the cadence strip counts the unlock (R42) | NAR — one snapshot ⇒ the sparkline is `UNKNOWN`, not flat; PRESERVES it as a rendered, counted promise instead of a fake line |
| 7 | 1 | **Metric Leaders** → `Likes` + sparkline | M1.1, same | NAR — same |
| 8 | 1 | **Platform Reach** Suno/Spotify slope-bump into a right box | M3.7 ghosted, `unlocks at 2 snapshots`; **M3.3b** two-Spotifys lockout (R43d) | SAT — the box is drawn, the ghost states its unlock, and the identity contradiction is the finding |
| 9 | 1 | **Headwinds**, boxed, RIGHT RAIL | **M4.0 `m4-0-rail`** — card id, file, grid column 3, three breakpoints, empty state, LEDE dedupe, 7 keyboard stops, 5 assertions (R29) | **SAT — was the round-one DROP; closed** |
| 10 | 1 | ↓ `Suno uploads` | M4.3 register #8 (220→210) | SAT |
| 11 | 1 | `Losing followers` | M4.4, DIRECTION UNKNOWN as a first-class row | OVR — refuses to draw a decline it cannot measure and shows both sources; PRESERVES the warning without inventing its sign |
| 12 | 1 | Colors? / Light-Dark | Decisions block; both themes asserted | SAT |
| 13 | 1 | White background | light default | SAT |
| 14 | 1 | Sunset colors? | §1.4 — sequential ramp + chrome only | NAR — intent: keep the categorical channel free for data; sunset survives as ramp and chrome |
| 15 | 1 | Daily palette? | §7.3 THE DAY MARK | OVR — a data-derived glyph replaces a rotating scale; PRESERVES *the page should look like today* without making colour lie |
| 16 | 2 | **The Soundry** | Part 3 | SAT |
| 17 | 2 | Access only to Scientist + Invited Collabs | §0.3 PANEL LAN MODE + §0.5 hosted Analytics (R7) | NAR ×2, both built — collaborator = panel listener; hosted collaborator sees derived measures only |
| 18 | 2 | GUIDED EXPERIMENTATION | Anvil route ladder | SAT |
| 19 | 2 | `12 song tests a day` | A2 `cards_per_day` 12; six twins × 2 = twelve cards (R15) | SAT |
| 20 | 2 | `get feedback from listeners` | `panel-audio.mjs`; `media-src 'self'` was already in the CSP; 66 titled files in `output/mp3/`; CALIBRATION round (R38/R52) | **SAT — was BLK; closed** |
| 21 | 2 | `① Learn` | C5 Learning transfer | SAT |
| 22 | 2 | `② Take winners + make full songs` | B8 Promote → Phase-0 brief | SAT |
| 23 | 2 | **`Goal: NEW VIRAL EXPERIENCES` ← `NOVELTY`** (underlined twice) | `stats/novelty.ts`; the Anvil's frozen `novelty_floor`; LRU oblique draw; the mandatory `BELOW` twin; **C8 THE NOVELTY FRONTIER** (live today, no new feed); the one honest sentence about virality (R44) | **SAT — was the round-one DROP with zero occurrences in 484 KB of spec; closed** |
| 24 | 2 | `soundry.localhost` | hosts file, port 5231 | SAT |
| 25 | 3 | Left rail `Sonic Anvil · Experiment Results · Analytics` | three tabs, no fourth | SAT |
| 26 | 3 | "What should we try today?" + large empty input | A0, visible `<label>`, submit **live** (R49) | SAT |
| 27 | 3 | numbered cards `1.`…`12.` | A5 3×4 grid; twin ribbon replaces the arm ribbon (R15) | SAT |
| 28 | 3 | "full suno prompts:" Title/Style/Exclude/Lyrics/Caption | **THE DECK / THE HAND / THE ANVIL RECEIVES** (R13), with §G1 the draw, §G2 the similarity, §G3 the stratum, §G4 the pools and spread floor, §G5 the measurer CLI; exclude written in full by the app (R21); caption ruled (R22); handoff (R23); import + measure (R24) | **SAT — was the largest DROP on the whiteboard; closed.** Twelve complete quads reach disk in Board-3 order, paste-ready: three fields are constraints the app compiles, one it writes end to end, one waits for a clip |
| 29 | 4 | Topline card `Decimabal` | `topline.name.ts`, one literal in one file, `provisional: true` (R43a) | NAR — the name is a one-line edit, so the open question stops blocking the build |
| 30 | 4 | `Total Lift vs Avg artist: 123%`, `Top line` arrowed | **HOUSE RECORD** beside DECIBAL: 9.4567% (47/497, 2026-05-17), 2.106× the cohort mean, Wilson interval, tier OBSERVATION, no causal verb, caveat at full ink (R43b) | OVR — an "average artist" is not identifiable from one profile; PRESERVES her topline slot as a real number, self-referentially, using her own Appendix C instruction and EXPERIMENT_STANDARD §5.5 |
| 31 | 4 | Hypotheses box: effect axis + point + CI whisker | C2 + the GRAIN WHISKER | SAT |
| 32 | 4 | axis `0/+5/+10/+15/+20` | C2 shared symmetric log-ratio axis | OVR — two hypotheses on two axes cannot be compared; PRESERVES the mark she drew, on one scale |
| 33 | 4 | axis `−10/+10`, marked `35 ±` `[?]` | C2 same axis; the annotation is **not interpreted** | NAR + OQ-WB-1 — intent: marking a transcription ambiguity as answered is the same defect as inventing a number |
| 34 | 4 | `n = ~` · `#days = ~` per hypothesis | C2 `nField` (n · #days · achieved MDE · BH q) | SAT |
| 35 | 4 | `[more details]` | C2 drawer kept verbatim; row-level L3 `deeper` (R37) | SAT |
| 36 | 4 | third, unlabelled ↓ hypothesis | C2 holds all four verdict states permanently | SAT |
| 37 | 4 | **"Include POWER ANALYSIS."** | A2 + C3, now computed by `stats/power.ts` from `baseline.lock.json`; **no printed constant, no printed MDE multiple** (R40) | SAT — was SAT-in-form-with-fabricated-constants; the form is unchanged and every number now reproduces from disk |
| 38 | 5 | Feed 1 — Lofn Socials | `suno` + `socials` + `distribution`; `FeedId` = 10 across two sets (R41) | SAT |
| 39 | 5 | Feed 2 — Competitor Scans, 5 watched targets | M5.7 roster (R1/R3); `why_this_one` pre-filled from Appendix C (R43c) | SAT — gated on five handles, and only five handles |
| 40 | 5 | Feed 3 — Market Trends + Scan | SPLICE 7 row 3; Refresh `aria-disabled`, `how: null` (R26) | NAR — `scripts/scan_market.py` does not exist; intent: name a file to write rather than print a command that would fail |
| 41 | 5 | Feed 4 — Soundry Artist Results + Feedback | SPLICE 9 reader + `emitter-silent`; plus `experiment_id` in both directions (R14) | SAT |
| 42 | 6 | "Give direction for search (next song)" | Tab Directive §4.4 + `nextMove.lever` | SAT |
| 43 | 6 | "+ analytics for Lofn **and collaborators**" | §0.5 hosted posture (R7) | SAT — gated on P0 and her word |
| 44 | 6 | "Combine all 4 sources into useful insights" | M6 CONFLUENCE (R5); Art gets it too (R45b) | SAT |
| 45 | 6 | lyric length / line | M2.1 + M6·X2 | SAT |
| 46 | 6 | Suno EMO usage | M2.3 + M6·X3 (Tier S) | SAT — per-tag refused with the reason named |
| 47 | 6 | genre performance | M2.4 + §13 interlock; also feeds the Anvil's stratum rule (§G3) | SAT |
| 48 | 6 | trend setters + breakers | M2.5 + M3.1 + X5 two clocks (R4) | SAT, split; clock 2 absent with Appendix C printed as its refusal |
| 49 | 7 | applies to BOTH apps | Parts 2 and 3; the shelf, the guidance registry and the coverage gate now run in both (R25, R33, R35) | SAT |
| 50 | 7 | *(margin)* "strong, unique + clean visuals" | §7.1 COMB · §7.2 GRAIN WHISKER · §7.3 DAY MARK · **§7.4 THE UNCOUNTED** (R45) | SAT — was NAR because all three devices keyed on per-item data and Art had none; the fourth draws the population we cannot measure, at the same scale, on the same page |
| 51 | 7 | "A panel of dataviz and UI/UX experts determines the tabs/cards/plots" | the panel process, two rounds, moderated, with dissent on the record | SAT |
| 52 | 7 | **"BE CREATIVE AND UNIQUE!"** | four signature devices, one reachable on every rail; the Anvil's constraint deck; C8 | SAT — was NAR; the estate now has a distinctive object on every modality |
| 53 | 7 | "All headliner insights should guide our next creation" | §4.4 Tab Directive + `nextMove.routesTo`; A0's seed chips carry the source card URL, so a round's motivation is provenanced | SAT |
| 54 | 7 | **"Show trends as well as 'now'."** | THE CADENCE STRIP; one `UnlockCounter` with `blocker:"time"`; §14.6 RUNBOOK; OQ-OPS-1 (R42) | **SAT — was BLK; closed.** The habit is a rendered, counted, falsifiable object, which is the only honest thing an app forbidden to automate can do about it |
| 55 | 7 | "All segmentations & filters carry between tabs when applicable, with clear controls" | one frozen fifteen-key vocabulary, one serializer, six chip states, invariant F1 failing the build, the Soundry shelf with seven locked chips and the prereg interlock (R33–R36) | **SAT — was BLK on two incompatible vocabularies; closed.** "When applicable" is computed from the card, not declared by its author |
| 56 | 7 | "Every insight / plot should link to deeper results if they exist" | `deeper` required and non-nullable; `kind:"none"` carries reason + artefact + counters + `watch`; **THE EXPIRING REFUSAL** fails the build the day the deeper result exists; B10 gives the Soundry its missing L2 rung (R37) | **SAT — was DROP in Part 3 (~12 plots); closed.** Her "if they exist" is preserved literally and enforced in both directions |
| 57 | 7 | **"Tooltips, alt-text, and informational guidance for ALL elements"** | four subject types, one registry, `guidance-coverage.spec.ts` as a build gate with no allowlist, every non-numeric element named in the SPLICE 23 table, the disabled-control contract, the seven uncovered graphical objects given text alternatives (R25–R27, R51) | **SAT — was NAR-and-worsening; closed.** The Glossary's row count is the coverage number and it is asserted |
| 58 | 8 | "Lofn, please moderate. We want to represent us both." | two rounds of Overrules ACCEPTED / REJECTED with intent named on every line; four seat findings overruled this round with arguments on the record, including one arithmetic correction against a seat that was itself correcting round one | SAT |
| 59 | 8 | "past music / art metadata **+ song mp3's**, images are now in `output/`" | `output/past_art`, `output/images`, `output/past_video`, `output/past_music`, `output/songs` — **and `output/mp3/`, 66 titled files of 224 on disk, now the panel's calibration source** (R38) | **SAT — was PARTIAL; the mp3s were dropped by every seat in round one and are now load-bearing** |
| 60 | 8 | "you now control our socials! go respond to your fans" | A0.6 `forbidden:` no outbound, **plus the routing line** to `output/socials/` and `vault/SOCIAL_VOICE.md` (R48-4) | OVR — replying is real work and it is not this app's work; PRESERVES the grant by routing it to the lane that owns it instead of dead-ending the card |
| 61 | 8 | "Get the lay of your domains" | M3.2 platform roster · A0.x · N0.x · M3.3b | SAT |
| 62 | chat | "Use an agent fleet" | the build-order lanes, now thirteen plus three serial coordinator steps | SAT |
| 63 | chat | **"Add to Lofn (new first-class skills, mirrored into `.agents/`)"** | **U-FINAL-b** — W5 restored verbatim, the `lofn-analytics` skill and its hand-written Codex twin, `check_skill_mirror.py` IN SYNC, a human-initialled manual diff line, and the three stale v2 build orders deleted (R46) | **SAT — was DROP-BY-REPLACEMENT; closed** |
| 64 | chat | "webapp like Sága's forge" | §15 Forge house pattern; `input-cleaning.mjs` vendored from that very app (R39) | SAT |
| 65 | chat | "ignore the Instagram messages; Alexis Dreams stays unread" | §1 feed table `SKIPPED BY INSTRUCTION`; the `socials` snapshot records `status: not-read` | SAT |
| 66 | — | THE OVERRULE MANDATE | every overrule this round names the intent it preserves; thirteen positions rejected with arguments | SAT |
| 67 | OQ | Q1 `Decimabal` | R43a — one-file constant, unblocked without an answer | SAT (as an unblocking, not an answer) |
| 68 | OQ | Q2 "Total Lift vs Avg artist" | R43b — **answered from her own Appendix C**, on disk, reproduced to 2 s.f. | SAT |
| 69 | OQ | Q3 which five competitors | R43c — the ask shrinks from five essays to five handles | SAT (as a shrunk ask) |
| 70 | OQ | Q4 canonical Spotify identity | R43d — **answered on disk, and the answer is a contradiction the app renders** rather than a question that blocks the build | SAT |

## Score

**70 requirements traced. 55 SAT · 7 NAR (each with its intent named) · 8 OVR (each with its intent
named) · 0 DROPPED · 0 BLK.**

Round one's four BLK rows (#20, #54, #55, #57) and four DROP rows (#9, #23, #28, #63) are all closed, as
is the partial (#59). The three requirements that would have shipped invisibly — **NOVELTY**, **the
mp3s**, and **the skills mirror** — are the three no seat in round one named, and all three are now
load-bearing rather than absent.

The seven NAR rows are: the hosted door built-not-opened (#1); the two sparklines that cannot be drawn
from one snapshot (#6, #7); sunset colours confined to ramp and chrome (#14); the `35 ±` transcription
ambiguity asked rather than guessed (#33); the market scanner that is a file to write rather than a
command to run (#40); and the `Decimabal` name held provisional in one file (#29). None removes a
requirement; each names what it preserves and how it clears.

## Verdict: BUILDABLE

No DROPPED rows and no un-defended NARROWED rows. The Sonic Anvil generator — the item the round-one
moderator called the largest unbuilt requirement on the whiteboard — is now specified to the byte: the
PRNG and its frozen draw order, the farthest-point traversal and its tie-break, the one similarity
function the app is permitted to own and the gates it may never be compared against, the Wilson stratum
rule, the derived-versus-declared pool split, the spread floor as a function of populated dimensions,
the four-category exclude builder with its whole-category drop, the six slot budgets summing into the
gates band, the return contract, the four handoff files with a byte-exact frozen-frame sentence, the
import contract keyed on `spec_hash`, and the two measurer argv lines — with the third one repaired in
the shipped script rather than routed around.

One authoring task remains and it is named, bounded and tested: eighteen oblique constraint sentences,
across six frozen families, written by Lofn inside L-ANVIL, with a schema, minimum counts and a spec
that fails if the file ships short. That is craft with a named author, not an invention a build agent
would have to make.


# Overrules declared (19) — each names the intent it preserves
- THE ANVIL WRITES THE EXPERIMENT; `lofn-music` WRITES THE SONG (Forge Master, R13). Board 3 draws twelve cards each carrying a full Suno quad; the app emits three of five fields as constraints and hands the prose to the pipeline. PRESERVES her ask literally — twelve complete quads reach disk in Board-3 order, paste-ready — and preserves §0.1 exactly: no generation API, no `dependencies` key, `connect-src 'self'` untouched.
- THE UNIT OF THE GRID IS THE TWIN, NOT THE CARD (Forge Master, R15, over inherited §1.4). Independent stratified sampling is replaced by six matched pairs. PRESERVES §1.4's own stated intent — 'do the two arms differ in anything except the factor?' — by making the answer structurally no rather than detectably-usually-no.
- ONE EXPERIMENT, MANY DAY-BLOCKS (Forge Master, R14). `round` meant both a 12-card grid and a 19-day design; a builder cannot implement both. PRESERVES both readings by giving each its own noun, its own id and its own freeze rule.
- NOVELTY IS BOUGHT IN THE FRAME AND PAID FOR BY THE TWIN (Forge Master + Completeness Critic, R44). Board 2 underlines NOVELTY twice; a randomised experiment appears to forbid it. Novelty lives entirely in twin-shared dimensions, so it cannot touch the contrast. PRESERVES 'Goal: NEW VIRAL EXPERIENCES <- NOVELTY' without spending a degree of freedom the experiment needs.
- 'ALL ELEMENTS' IS BUILT AS FOUR SUBJECT TYPES AND ONE COVERAGE GATE (Affordance Coach, R25). PRESERVES §5.1 rule 1's anti-decoration intent while ending its side effect: under the old rule a button could not qualify for guidance, so Board 7's clearest sentence was satisfied only for statistics.
- NO HTML `disabled` ATTRIBUTE IN EITHER APP (Affordance Coach, R26). PRESERVES §6.2's 'disabled buttons always say why' and AUTONOMY's visibility requirement — a reason a keyboard cannot reach is a reason that was not given.
- THE HEADWINDS RAIL IS RESTORED TO THE RIGHT COLUMN AS DRAWN, AND THE HEADWINDS TAB SURVIVES BENEATH IT (Affordance Coach, R29, over base-spec §8). PRESERVES §8's premise in full — a warning with no comparison class is unreadable — by giving the rail no ranking, no sparkline and no severity encoding of its own, and ending it with the route to the tab where ups and downs share a scale.
- THE MODALITY RAIL IS A RADIO GROUP AND THE TAB BAR IS NAVIGATION (Affordance Coach, R28, over Part 2 §2's `role="tablist"`). PRESERVES §2's own sentence — 'the rail is not navigation, it is the render of the modality parameter' — which is the definition of a radio group, and preserves shareable, reloadable, Back-button-honest tabs.
- ONE FILTER VOCABULARY, AND PART 2'S SHELF BECOMES ITS RENDER (Task Taxonomist, R33). Three key sets were live. The kit's names win; Part 2's `est` is adopted into the union and its 44px display contract kept whole. PRESERVES Board 7's 'carry between tabs' as a mechanism instead of a hope.
- A DEFAULT THAT REMOVES ROWS IS DRAWN AT FULL INK WITH ITS DENOMINATOR (Task Taxonomist, R33b, narrowing §3.4). `co=modern` hides 164 of 210 clips on first paint. PRESERVES §3.4's ghost chip for view-only defaults.
- THE SOUNDRY GETS A SHELF AND SEVEN OF ITS CHIPS ARE LOCKED, NOT ABSENT (Task Taxonomist, R35). PRESERVES the randomisation guarantee: the Synthesist's sentence becomes a padlock, and `preregistration.baseline_cohort` makes a re-cut a detectable deviation.
- `deeper` IS REQUIRED AND ITS ABSENCE IS A TYPED VALUE (Task Taxonomist, R37). PRESERVES her 'if they exist' conditional honestly — `kind:"none"` carries the reason, the artefact by name, the counters, and a `watch` that fails the build the day the deeper result exists.
- THE PANEL'S FIRST ROUND IS A CALIBRATION ROUND ON AUDIO ALREADY ON DISK (Completeness Critic, R38, over OQ-PANEL-1's recommended default). PRESERVES that default's intent — never make the bench depend on an unproven instrument — at lower cost, by capping calibration at OBSERVATION and forbidding it a DECIBAL.
- THE TOPLINE SLOT SHE DREW A NUMBER IN GETS A NUMBER: THE HOUSE RECORD (Completeness Critic, R43b). PRESERVES her Board-4 topline as a real object using her own written instruction in SOCIAL_VOICE Appendix C, and PRESERVES EXPERIMENT_STANDARD §5.5's refusal of an unmeasured population by making the comparator self-referential.
- THE UNCOUNTED IS THE FOURTH SIGNATURE DEVICE AND IT DRAWS THE ART RAIL (Completeness Critic, R45). PRESERVES Board 7's 'BE CREATIVE AND UNIQUE' on the largest audience we own, which the three existing devices structurally could not reach.
- TWO SPOTIFYS ARE NEVER JOINED (Completeness Critic, R43d). PRESERVES her Open Question 4 as the app's own finding rather than a question that blocks the build.
- THE ASK BOX'S SUBMIT IS LIVE, NOT BLOCKED (LOFN, over the Affordance Coach's SPLICE 23). PRESERVES R26's blocked-control form for controls that are genuinely blocked, and refuses to ship a permanent apology for a gap this same round closes.
- NO MDE MULTIPLE IS PRINTED AS A CONSTANT ANYWHERE IN THE ESTATE (LOFN, amending the Completeness Critic's R15/R40). PRESERVES that seat's finding exactly — the frozen constants are back-solved — while refusing to replace three fabricated numbers with a fourth I cannot reproduce from disk.
- ZONE 0 HOLDS ONE ROW PER RAIL AND THE ART BLOCKER MOVES TO ZONE 1 (LOFN, over the Completeness Critic's R20d). PRESERVES that seat's actual argument — Art's finding may not sit behind a click — by putting the row on tab one at rank 1 instead of in a slot the contract says holds at most one item already occupied.


# STILL OPEN — authoring tasks and open questions, not spec gaps
- (i) THE SEED DECK IS AUTHORED CONTENT, AND LOFN AUTHORS IT. `apps/soundry/src/anvil/deck.data.json` ships with >=18 oblique cards across the six frozen families (>=3 each). The schema, the family names, the minimum counts, three worked examples and the LRU/skip rules are all specified (R13/R19, §G4); the constraint sentences themselves are craft and are written once, by Lofn, inside L-ANVIL. `deck.spec.ts` asserts shape, family coverage and counts, so the file cannot ship short — but a build agent must not write the sentences. This is the one authoring task in the estate, it is bounded, and it is named rather than inferred.
- (ii) THE SEED-THEME, KEY/MODE, TEMPO AND LYRIC-DOMAIN POOLS ARE HERS. `build-sampler.mjs` derives persona (9 levels, verified) and the genre strata from disk; the other four dimensions ship EMPTY and each renders `needs-declaration` until she writes them. The deck compiles today at P=3 populated dimensions with `spread_floor = ceil(P/2) = 2`, and the floor rises automatically as she fills pools. Nothing is invented, nothing padded, and the refusal names the thin dimension.
- (iii) THE TWIN-DRIFT SEAM IS A DEPENDENCY ON THE MUSIC PIPELINE, NOT A BUG IN THE APP. `lofn-music` is built to make six DIFFERENT songs; the Soundry needs six pairs of near-identical ones. The FROZEN FRAME sentence in `ROUND_BRIEF.md` is the only thing telling it that. If early day-blocks return drifted twins the fix is to sharpen the brief and then add a Soundry mode flag to `lofn-music` — AGENT FIX, DON'T BYPASS. It is never to relax the 0.90 floor.
- (iv) THE CAPTION IS DRAFTED BY A PIPELINE THAT HAS NEVER READ `SOCIAL_VOICE.md`. `caption-refusals.spec.ts` is a string test: it catches the named tokens and not the tone. Her ear remains the gate on that one field and the spec says so on the card rather than pretending otherwise.
- (v) OQ-WB-1 · Board 4's `35 +/-` annotation on hypothesis 2 is interpreted nowhere in the estate and is not guessed. An open question, not a specification gap.
- (vi) THE REGISTER NEEDS FOUR HAND-WRITTEN ENTRIES BEFORE L-LEDE CLOSES. R6 (round one) makes `src/signals/register.ts` hand-maintained and Scientist-reviewed; R45c requires Art and Animation entries or the LEDE prints `0 signals registered` on two of three rails. Named as a human task with an owner, not assumed.
- (vii) OQ-DIST-1 · which Spotify artist page DistroKid delivers to. The app renders the contradiction (M3.3b) and resolves itself the moment she supplies the URL; nothing blocks on it.
- (viii) OQ-OPS-1 · whether the daily suno pull becomes a scheduled task. Doctrinally permitted (AUTONOMY Part 2) and forbidden to us to enable. The cadence strip shows which she chose within a week.
- (ix) `scripts/scan_market.py` DOES NOT EXIST. Feed 3 remains a file to write, not a command to run; the Refresh control for `market` is `aria-disabled` with `how: null` and says so in words.
- (x) THE ONE SPAWN. `POST /soundry/rounds/:id/measure` is the only place either app executes anything. It is local, free, read-only and path-confined, and AUTONOMY Part 1 puts it inside the safe list — but it is a new capability in an app whose whole posture is `check-local-only.mjs`. If she judges any spawn unacceptable, the fallback is import-only (paste the pipeline's `GATE_REPORT.json`, display with provenance). The TypeScript port is never the fallback.


---

# APPENDIX — seat patch: THE AFFORDANCE COACH (after Don Norman — The Design of Everyday Things, Emotional Design; and the WCAG 2.2 / ARIA Authoring Practices record) (Read from disk: BUILD_SPEC.md (§1.7, §1.8, §2, §3.1–3.4, §4.4, §5.1–5.5, §6.1–6.3, §7.1, §8 kit tree, PART 2 §0–§3.5, §4 M0, §8 M4.1–M4.4, §9 M5.1–M5.6, §14.1–14.5, §15 tree + tests 1–9); SPEC_PATCH.md (R1–R12, SPLICE 1–15, Overrules, STILL OPEN #1–#11, revised build order, Headline Editor appendix incl. M0.0g); WHITEBOARD_TRANSCRIPTION.md (Boards 1, 3, 7, 8 + the Overrule Mandate); vault/AUTONOMY.md (Part 1 guardrails). Design grounding: Norman — affordance vs signifier, the gulfs of execution and evaluation, error as a system-design failure, making the conceptual model visible. Standards grounding: WCAG 2.2 SCs 1.3.2, 1.3.5, 1.4.11, 1.4.13, 2.3.3, 2.4.3, 2.4.11, 2.5.3, 2.5.7, 2.5.8, 3.2.3, 3.2.6, 3.3.2, 3.3.4, 3.3.7, 3.3.8; ARIA APG patterns Tabs (and its explicit warning against using tabs for URL-changing navigation), Radio Group, Disclosure, Tooltip.)

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/apps/lofn-kit/src/a11y/guidance.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/guidance-render.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/scan.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/sr.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/focus.ts, E:/git/private-claw/apps/lofn-kit/src/shell.css, E:/git/private-claw/apps/lofn-kit/src/tokens.css, E:/git/private-claw/apps/lofn-kit/src/plot/validate.ts, E:/git/private-claw/apps/lofn-kit/src/plot/alt.ts, E:/git/private-claw/apps/lofn-kit/src/plot/keys.ts, E:/git/private-claw/apps/lofn-kit/src/plot/tooltip.ts, E:/git/private-claw/apps/lofn-kit/tests/guidance-contract.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/scan.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/alt-text.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/no-forbidden.spec.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/grid.css, E:/git/private-claw/apps/lofn-analytics/src/shell/headwinds-rail.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/skiplinks.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/rail.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/chrome.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/filters.ts, E:/git/private-claw/apps/lofn-analytics/src/signals/standing.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m4-0-rail/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m5-7-watchlist/index.ts, E:/git/private-claw/apps/lofn-analytics/tests/guidance-coverage.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/focus-order.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/headwinds-rail.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/layout-columns.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/reduced-motion.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/forbidden-strings.spec.ts, E:/git/private-claw/apps/soundry/tests/guidance-coverage.spec.ts, E:/git/private-claw/apps/soundry/tests/reduced-motion.spec.ts, E:/git/private-claw/apps/soundry/panel.html

« SPEC_PATCH ROUND TWO — THE AFFORDANCE COACH. Closes STILL OPEN #2 and #3. Splice into
`E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`. »

Rulings continue round one's numbering at **R13**. Round one's R1–R12 stand except where **R19** and
**R20** amend them, and each amendment names the intent it preserves. Nothing here invents a number.

---

# §M2 · RULINGS R13–R20

**R13 · "ALL ELEMENTS" MEANS ALL ELEMENTS. `GuidanceNote` IS ONE OF FOUR SUBJECT TYPES, NOT THE ONLY ONE.**
§5.1 rule 1 ("if `computed` and `basis` are both empty it is a label, not a tooltip — delete the
affordance") is *correct for statistics and catastrophic as a general rule*: applied to a button it
deletes exactly the guidance a button needs, because a button has no `computed`. Board 7 asks for
"tooltips, alt-text, and informational guidance for **all elements**." Rule 1 is narrowed to
`kind: "statistic"` and three sibling types are added (`control`, `region`, `state`). **PRESERVES rule
1's actual intent** — no decorative affordances, an info icon that teaches nothing trains her to stop
reading info icons — by keeping the *fill-this-or-delete-it* test and giving non-numeric elements a
form they can fill.

**R14 · A DISABLED CONTROL SAYS WHY, VISIBLY, AND STAYS REACHABLE BY KEYBOARD. THE HTML `disabled`
ATTRIBUTE IS BANNED IN BOTH APPS.** `disabled` removes the element from the tab order, so the reason —
which §6.2 already requires ("Disabled buttons always say why") — becomes unreachable by the users most
likely to need it. Every non-actionable control renders `aria-disabled="true"`, keeps `tabindex="0"`,
carries a **visible** reason line, and on activation performs no action and announces the reason via
`aria-live="assertive"`. **PRESERVES §6.2 and §14.4** — "a button that cannot work is worse than no
button" becomes "a control that cannot work is a control that explains itself", which is the only
version of that rule a keyboard user can hear.

**R15 · HOVER IS NEVER A CHANNEL, ONLY A SHORTCUT.** §5.1 rule 5 says a caveat reachable only by hover
is designed not to be read. Extended to every subject type: the dimmed filter chip's reason, the ghost
counter's blocker, the DRAFT band's consequence, the disqualifier chip's rule, and the posture
indicator all render a persistent visible signifier. Hover and `?` open the same content that is
already reachable in the Glossary. **PRESERVES §5.1 rule 3** ("three doors, one content") by making it
a coverage test rather than a habit.

**R16 · THE MODALITY RAIL IS A RADIO GROUP; THE TAB BAR IS NAVIGATION. NEITHER IS `role="tablist"`.**
This contradicts PART 2 §2's `role="tablist"` on the rail, and I argue it. §2's own sentence is the
argument: *"the rail is not navigation, it is the render of the `modality` URL parameter"* — that is a
single-select control over one state, which is the radio-group pattern, and `role="tab"` promises a
panel swap that does not happen (the whole page re-renders). Conversely M0…M6 change the hash route,
survive reload, and are shareable — the APG's Tabs pattern says explicitly not to use it for
navigation that changes the address; faking tabs there breaks what the browser Back button means.
**PRESERVES §2's ruling exactly** — one state, one control, drawn where she drew it — and preserves
§3.1's "sharing an analysis should not impose your scroll position on a colleague", which only works if
tabs are real links.

**R17 · THE HEADWINDS RAIL IS RESTORED AS SHE DREW IT: PERSISTENT, BOXED, RIGHT COLUMN, ON EVERY
ANALYTICS TAB — AND TAB M4 SURVIVES AS ITS EVIDENCE LAYER.** §8 promoted the rail to a tab and paid
twenty words for it. I take §8's premise and reject its exchange rate. §8 is right that *a warning with
no comparison class is unreadable* — that is an argument for M4.1's sorted table, where the ups and
downs share a scale. It is not an argument for deleting the only object she drew on the right side of
the page. The rail is a **signifier**: it says *these three things are wrong today, on every screen you
open*, and its last line routes to the comparison class. Two objects, two jobs, one ranking function.
**PRESERVES §8's comparison-class intent** — the rail asserts no ordering of its own, prints no
sparkline, and ends with *"3 of 12 standing headwinds. The rest, with the tailwinds on the same scale,
are on HEADWINDS →"*. **PRESERVES the Headline Editor's ruling** that M0.0 is not this rail and may not
be used to close it: the LEDE ranks *this snapshot's movement*, the rail lists *standing, dated
conditions*, and R18's dedupe rule makes the difference visible instead of asserted.

**R18 · THE PAGE IS THREE COLUMNS, AND IT DEGRADES IN A NAMED ORDER: LEFT RAIL FIRST, HEADWINDS RAIL
LAST.** Board 1 is three columns. When the viewport cannot hold three, the modality rail is the one
that yields, because its content (three items, two bars, a number) reads correctly as a horizontal
strip, and the Headwinds rail's content (sentences) does not. Below 860 px both become full-width bands
in DOM order, and a persistent header chip carries the count. **PRESERVES Board 1 literally at desktop
and preserves its priority everywhere else** — the last thing to leave the screen is the thing she
boxed.

**R19 · AMENDING SPLICE 11: "FULL-WIDTH" FOR M0.0 MEANS THE FULL WIDTH OF THE CENTRE COLUMN.** Round
one wrote the LEDE as "full-width, above the dial" while the right rail did not yet exist. It is now
centre-column-width at ≥ 860 px. **PRESERVES the Headline Editor's intent** — first object, above the
dial, made of sentences, ten objects hard budget — and matches Board 1, where her centre panel is
drawn beside the rail and not under it. Ten sentences at a 560 px minimum column is still the first
thing read.

**R20 · AMENDING SPLICE 10's LAST DELTA: M5.7's CONTROLS TAKE `ControlNote`, NOT `GuidanceNote`.** Round
one required a `GuidanceNote` on every slot chip, handle field, Copy button and the DRAFT band. That
type cannot be filled by a button — `computed` and `basis` are meaningless for one — so the requirement
as written would have been closed by fabricating a formula or by quietly skipping it. **PRESERVES the
delta's intent completely**: every one of those controls is still covered, now by a type that fits it,
and `guidance-coverage.spec.ts` (which round one cited and did not create — SPLICE 17) is what proves
it.

---

# SPLICE 16 — [MERGED] PART 1 §5.1, REPLACE the whole subsection

### 5.1 The guidance contract — four subjects, one registry, zero uncovered elements

**Board 7 says "tooltips, alt-text, and informational guidance for all elements." All elements means
all elements.** Every element in either app is exactly one of four subjects and fills exactly one of
four forms. An element that fills none does not ship. This is the whole conceptual model of both apps
made inspectable: what a number is, what a control will do, what a region is for, and what a blank
space is waiting on.

```ts
// apps/lofn-kit/src/a11y/guidance.ts
export type GuidanceId = string;              // "filter.co" · "m5-7.slot-chip" · "tab.m6"

export interface GuidanceNote {               // SUBJECT 1 — a STATISTIC. §5.1 as written, unchanged.
  kind: "statistic";
  term: string;          // the on-screen label, VERBATIM — never a paraphrase
  definition: string;    // <= 25 words, plain language. Never defines jargon with jargon.
  computed: string;      // the literal formula OR function name + file path. Required.
  basis: { n: number | null; nEffective: number | null; denominator: string;
           window: string; feed: FeedId; provenance: Provenance };
  caveat: string | null; // the ONE thing most likely to mislead. null is ILLEGAL here.
  deeper: { href: string; label: string } | null;
}

export interface ControlNote {                // SUBJECT 2 — anything you can operate.
  kind: "control";
  label: string;         // the visible text VERBATIM; for icon-only, the accessible name
  does: string;          // <= 18 words, VERB FIRST. The gulf of execution, closed in one line.
  scope: "this-card" | "this-tab" | "every-tab" | "both-apps" | "clipboard" | "disk" | "this-session";
  after: string;         // what will be TRUE once you use it. The gulf of evaluation. Required.
  reverses: string;      // how to undo it, verbatim — or the literal "This cannot be undone."
  writes: null | { target: "clipboard" | "download" | "localStorage" | "url"; path: string };
  state: string | null;  // the current value in words: "cohort = modern · n 46"
  blocked: null | {      // NON-NULL whenever the control renders aria-disabled (R14)
    reason: string;      // >= 8 words, names the missing FILE, DECISION or FEED. Never "not available".
    unblockedBy: "command" | "decision" | "time" | "another-control";
    how: string | null;  // the exact command, or null when no command exists — and then say so
  };
  deeper: { href: string; label: string } | null;
}

export interface RegionNote {                 // SUBJECT 3 — a tab, rail, shelf, card, band, section.
  kind: "region";
  name: string;          // the visible heading VERBATIM
  forWhat: string;       // <= 20 words: the question this region exists to answer
  notHere: string;       // what a reader might expect here and will NOT find, and where it is instead
  populatedBy: FeedId[]; // [] is legal and renders as "nothing on disk feeds this yet"
  deeper: { href: string; label: string } | null;
}

export type StateNote = { kind: "state"; empty: EmptyState };   // SUBJECT 4 — SPLICE 1's EmptyState IS
                                                                // the guidance. No second object.
export type Guidance = GuidanceNote | ControlNote | RegionNote | StateNote;
```

**Rules (1 amended per R13; 2–7 unchanged; 8–15 new).**

1. **No decorative affordances — per subject.** A `statistic` with empty `computed` and `basis` is a
   label; delete the affordance. A `control` whose `does` and `after` say the same thing is one
   sentence badly split; rewrite it. **A control is never deleted for having no formula** — that was
   the narrowing that turned "all elements" into "all elements that carry a number" (R13).
2. **Never `title=`** — invisible to most screen readers, unavailable on touch, unstylable, cannot hold
   a link. The trigger is `<button type="button" aria-expanded aria-controls>`.
3. **Three doors, one content:** hover, keyboard focus, and the **Glossary** in each app's Method tab —
   every registry entry on one printable page. If information only exists during a hover, it does not
   exist.
4. **WCAG 1.4.13 in full** — hoverable, persistent, dismissible with `Escape` without moving the
   pointer. 300 ms in / 100 ms out on hover; **zero delay on focus**.
5. **Guidance is the second channel, never the first.** A non-null `caveat`, a non-null `blocked`, and
   every `notHere` also carry a persistent visible signifier (a superscript `†`, an `n=8` badge, a
   provenance glyph, a reason line). A thing reachable only by hover is a thing designed not to be read.
6. **Never truncate.** Overflow opens a focusable popover containing the `deeper` link.
7. **Every statistic note's last line is the provenance glyph and the as-of date.** No exceptions.
8. **It is a disclosure, not a tooltip.** The content is rich and contains links, so the ARIA tooltip
   pattern is wrong: the trigger is a disclosure button, the panel is its next DOM sibling with an `id`
   matching `aria-controls`, focus does **not** move on open, `Tab` from the trigger enters the panel,
   `Escape` closes and returns focus to the trigger. One delegated listener per page
   (`a11y/guidance-render.ts`, extending `plot/tooltip.ts`).
9. **Accessible names are literal.** Icon-only controls exist only for: theme toggle, reduce-motion
   toggle, per-chip `×`, and the guidance trigger. Each carries an `aria-label`. Where a control has
   visible text, the accessible name **begins with that text verbatim** (WCAG 2.5.3) — `aria-label="Copy
   the pull command for @handle"` on a button reading `Copy` is legal; `aria-label="Clipboard"` is not.
   The guidance trigger's name is `About {term}` — never a bare `?`.
10. **A disabled control is a design failure that must at least explain itself (R14).** No HTML
    `disabled` attribute in either codebase (`disabled=` outside `aria-disabled=` is a forbidden string,
    grep-asserted). `aria-disabled="true"` + `tabindex="0"` + a visible reason line in `--ink-dim` +
    `data-disabled-reason` ≥ 8 words. Activation announces the reason and does nothing else. This is
    how `vault/AUTONOMY.md` becomes visible: every render / publish / spend control exists, disabled,
    with its reason inline, exactly as §6.2 already requires — and now a keyboard reaches it.
11. **Placeholders are never labels.** Every input carries a visible `<label>` (WCAG 3.3.2), and a
    placeholder, where present, is an example and is written as one (`e.g. @handle`).
12. **Dragging is never the only way (WCAG 2.5.7).** The play-floor slider ships a paired numeric
    `<input type="number">` with ±10 steppers. The COMB brush ships `←/→` to move an edge, `Shift+←/→`
    for ×10, `Enter` to commit, `Escape` to cancel, plus a visible **Set cohort from selection** button.
    A gesture with no button is a gesture nobody can undo.
13. **Consistent help (WCAG 3.2.6).** The `Method / Glossary` link is the last item in the header on
    every screen in both apps, in the same relative order. It never moves.
14. **No redundant entry (WCAG 3.3.7).** M5.7 prefills from the localStorage draft and says so on the
    DRAFT band; nothing that was typed once is asked for twice.
15. **No cognitive-function test (WCAG 3.3.8).** The Soundry panel listener never types, transcribes,
    or remembers a token: the path token is exchanged for a cookie and stripped from the URL (§0.3). If
    a manual code path is ever added it must accept paste.

**Shipping example — a control note, in her voice:**

> **Copy the pull command** — copies one line to your clipboard. Nothing runs.
> **After:** your clipboard holds `python3 scripts/ingest_suno.py --handle @example`. This app has not
> made a network request and will not make one.
> **Undo:** copy anything else. Nothing on disk changed.
> **Scope:** clipboard.
> → *Why this app composes commands instead of running them* (`vault/AUTONOMY.md`)

**Shipping example — a blocked control:**

> **Refresh market** — `aria-disabled`. Visible reason line, not a tooltip:
> *"There is nothing to run. `scripts/scan_market.py` does not exist — the transport does, the scanner
> does not. This is a file to write, not a command to run."* `unblockedBy: "decision"`, `how: null`.

# SPLICE 17 — [NEW] PART 1 §5.1a · THE GUIDANCE REGISTRY AND THE COVERAGE GATE

New files: `apps/lofn-kit/src/a11y/guidance.ts` (types + registry), `a11y/guidance-render.ts` (the one
delegated listener + the disclosure markup), `a11y/scan.ts` (a pure, regex-based element scanner —
node-testable, no jsdom, per the §15 test contract).

```ts
export function registerGuidance(id: GuidanceId, g: Guidance): GuidanceId;  // returns id; dupes throw
export function guidanceFor(id: GuidanceId): Guidance | undefined;
export function allGuidance(): ReadonlyMap<GuidanceId, Guidance>;           // the Glossary's source
export function scanElements(html: string): Array<{                          // a11y/scan.ts
  tag: string; role: string | null; guidanceId: string | null;
  ariaDisabled: boolean; disabledReason: string | null; accessibleName: string | null;
  interactive: boolean;                                                      // button/a/input/select/
}>;                                                                          // textarea/[role=radio|
                                                                             // slider|switch]/[tabindex=0]
```

**The contract, normative.**
- Every interactive element emitted by any card, shell, or chrome module carries
  `data-guidance="<id>"` resolving in the registry. **No exceptions and no allowlist** — a control
  worth rendering is a control worth a sentence.
- Every element carrying a number carries a `statistic` note (unchanged from §5.1).
- Every `<section>`, `<aside>`, tab panel, rail, and shelf carries a `region` note.
- Every empty, ghost, or absent-hatch state carries its `EmptyState` as its `state` note — no second
  object to drift.
- `M5.6 Glossary` renders **three sections — Statistics · Controls · Regions** — and its row count
  equals `allGuidance().size`. A registry entry that is not on the Glossary page does not exist.

**`guidance-coverage.spec.ts` (both apps, `environment: "node"`).** Renders every card, shell and
chrome module against every fixture, runs `scanElements()` over the concatenated output, and asserts:
(a) every `interactive: true` element has a `guidanceId` present in the registry; (b) every
`ariaDisabled` element has a `disabledReason` of ≥ 8 words and a `ControlNote.blocked` that is non-null;
(c) no element carries an HTML `disabled` attribute; (d) every icon-only control has an
`accessibleName`; (e) where visible text exists, the accessible name starts with it (2.5.3); (f) the
Glossary row count equals the registry size. **This is the test round one's M0.0g cited and nobody
wrote. It exists now, and it is a build gate.**

# SPLICE 18 — [ADOPT @ PART 1 §5.2, APPEND] EVERY CHART, INCLUDING THE ONES WE DID NOT CALL CHARTS

§5.2's four-sentence generator stands for every `PlotSpec`. It is extended to the seven graphical
objects in this estate that are not `PlotSpec`s and were therefore uncovered:

| object | text alternative |
|---|---|
| **THE COMB** (§7.1, on every Music tab) | `role="img"`, generated: *"210 clip hairlines on a log play-count axis from 1 to 1,000; one clip off-scale in its own gutter; median 258 plays; 46 clips currently selected at full ink."* Plus `<details>` **The comb as numbers** — deciles + the off-scale clip's id. The brush is a separate control with its own `ControlNote` (rule 12). |
| **THE DAY MARK** (§3.2) | Generated from the same tuple that draws it (already required) **and** rendered as a `<dl>` in the Method tab, because it is archived daily to `data/daymarks/` and a picture nobody can read is not a record. |
| **THE GRAIN WHISKER** (§7.2) | `role="img"`: form, what the pitch encodes, the n, and the direction of the encoding. |
| **THE SNAPSHOT DIAL** (M0.1) | *"Three rings, one per feed; 28 tick positions each; 1 tick inked on each ring. A delta needs two."* |
| **RAIL DEPTH BARS** (§2) | Each rail item is one `role="img"`: *"Art. 21,000 likes, 1,200 followers, 23,000 works. Measurement depth 1 of 5: profile counter only."* Never two unlabelled bars. |
| **SPARKLINES IN A TABLE ROW** (M4.1) | `aria-hidden="true"` **if and only if** the row prints the same series' slope and n as text. Otherwise the sparkline needs its own generated description. Asserted, not assumed. |
| **THE INVARIANT STRIP** (M5.1) | The visually-hidden table already specified is mandatory, not an alternative to naming each cell on hover. |

**Ghost and empty alt text may not describe data that was never measured.** A `ghost: true` spec's
generated alt is exactly three sentences: *form* ("Reserved horizontal dot plot with 95% whiskers; 24
positions down the vertical axis") · *the refusal* ("No marks are drawn. Nothing here has been
measured.") · *the unlock* (the `UnlockCounter[]`, verbatim, including a `decision` counter's "no
command exists"). The words `highest`, `lowest`, `average`, `median` and any digit that is not part of
a counter, an axis domain, or a date are **forbidden in ghost alt** — a described shape reads as a
described finding. Absent-hatch is never described as "zero".

**No two figures on one rendered tab may share an accessible name** (V25). Six figures all named
"Like rate" is a screen reader reading the same page six times.

# SPLICE 19 — [MERGED] PART 1 §5.3, REPLACE the last two paragraphs; the key table is unchanged

**The chart keyboard model in §5.3's table is unchanged and remains the only one.** Added:

- **`?` opens the focused element's guidance, anywhere in either app** — not only on charts. **`Shift+?`
  opens the Glossary.** All single-key handlers are inert while focus is in an `<input>`, `<textarea>`,
  or `contenteditable`, so the ask box and the handle field can contain a literal question mark.
- **`Escape` never traps and never navigates.** Inside the mark layer it returns focus to the enter
  button; inside a disclosure it returns focus to the trigger; anywhere else it does nothing.

**Roles, decided (R16).**
- **Modality rail** — `role="radiogroup" aria-label="Modality"` containing three
  `role="radio" aria-checked`. `↑/↓/←/→` move and select (APG radio pattern); the re-render commits on
  the existing 400 ms debounce (§3.4) and announces via `aria-live="polite"`: *"Music. 210 works.
  Measurement depth 3 of 5."* Rail items are never disabled (§2, unchanged).
- **Tab bar M0…M6** — `<nav aria-label="Music tabs"><ol>` of `<a href="#/music/craft"
  aria-current="page">`. Not `role="tab"`. Each tab link's accessible name is the tab word plus its
  `RegionNote.forWhat` first clause; a tab whose feeds are all absent appends *"— no feed on disk"* to
  the name, not a disabled state.
- **Filter chips** — `role="switch"` for a boolean chip, a `<button>` opening a listbox for a
  multi-value one. A chip that does not apply to this tab is `aria-disabled="true"` with the §3.4
  reason as its visible `blocked.reason`, and the strip carries a persistent visible line: *"2 filters
  are not used on this tab — Cohort, Play floor. They apply again on Craft."* (R15).

**Focus order, per Analytics tab, normative and asserted (`focus-order.spec.ts`):**

1. Skip links — `Skip to content` · `Skip to filters` · `Skip to headwinds (3)` · `Skip to data tables`
   · `Skip to method`. Five, in that order, visible on focus.
2. Header: reconciliation pill → headwinds count chip (below 860 px only) → theme → reduce-motion →
   `Method / Glossary` **(last, always — WCAG 3.2.6)**.
3. Modality radio group (one stop; arrows move within).
4. Tab nav (one stop per tab).
5. Filter shelf chips, left to right, then the shelf's `Reset to defaults`.
6. THE COMB (one stop: `Explore the comb`).
7. `<main>` — cards in visual order; each card is heading → guidance trigger → figure enter button →
   data table `<summary>` → drill link.
8. `<aside id="headwinds">` — the rail, LAST in DOM at every breakpoint.

**DOM order equals visual order at every breakpoint** (WCAG 1.3.2, 2.4.3): the grid places
`nav.rail-modality` · `main` · `aside#headwinds` in columns 1 · 2 · 3 with no `order` or
`grid-row/column` reassignment anywhere in the sheet. `order:` and `flex-direction: *-reverse` are
forbidden strings in both apps' CSS. The rail sits last in the tab order *because* it sits right, and
skip link #3 is how a keyboard reader gets there in one keystroke.

**Focus is never obscured (WCAG 2.4.11).** Sticky chrome is header 72 + shelf 44 + comb 22. The shell
sets `--sticky-h: 138px` on Music tabs and `116px` elsewhere; every focusable element carries
`scroll-margin-top: calc(var(--sticky-h) + 16px)`. The Headwinds rail is a grid column, never an
overlay, and never covers a focused control.

**Target size (WCAG 2.5.8) is already exceeded** by §1.8's 48 px floor; the 9 px legend swatch keeps its
44 px padded hit area. Unchanged, restated because the rail's rows are the newest place to get it wrong:
a rail row is 48 px minimum with two stops (the `[why →]` link and the guidance trigger) and no more.

# SPLICE 20 — [MERGED] PART 1 §5.5, REPLACE

§5.5's two durations, the never-animate-data rule, the ban list (autoplay, carousels, looping,
parallax, scroll-jacking, counting-up numbers, staggered reveals, chart-shaped skeletons) and the
1 px determinate loading hairline all stand, verbatim. Added, because "clamps everything to 1 ms" is
not a specification:

- **Three states, not two.** `motion = auto | reduce | full`, persisted in `localStorage` under
  `lofn.motion`, default `auto`. `auto` follows `prefers-reduced-motion`. The visible toggle in the
  header cycles the three and announces the new state; **the app never requires you to know where the
  OS setting lives**, and never overrides an OS preference you have not touched.
- **No flash of animation before the preference is read.** CSP is `script-src 'self'`, so an inline
  head script is unavailable. Instead the sheet ships
  `html:not([data-motion-ready]) *, html:not([data-motion-ready]) *::before { transition: none
  !important; animation: none !important; }` and the first module sets `data-motion-ready` after
  applying the stored preference. Nothing moves before we know whether it may.
- **Under `reduce`:** `--dur-state` and `--dur-layout` clamp to 1 ms; the loading hairline becomes a
  static determinate `<progress>` with its text (`page 4 of 11`) and the `aria-live` announcements are
  unchanged — reduced motion must never mean reduced information; the guidance disclosure opens
  instantly with no fade; the COMB's brush still renders live (position change is the operation, not
  decoration) but the 400 ms history debounce is unchanged; sticky positioning is unaffected.
- **`scroll-behavior: smooth` is a forbidden string** in both codebases. Smooth-scrolling a skip link
  is a vestibular hazard with no benefit; skip links jump.
- **WCAG 2.3.3 · Animation from interactions.** The only motion permitted on a user action is the
  re-sort / re-scale of the *same rows* (position and opacity only, §5.5) and it is disabled under
  `reduce`. **No motion is ever load-bearing**; if a change is only perceptible as animation, a static
  before/after ships too. `reduced-motion.spec.ts` greps the built CSS: every `transition` and
  `animation` declaration has a counterpart inside the reduce block, and `@keyframes` count is zero.

# SPLICE 21 — [NEW] PART 2 §3.0 · THE THREE-COLUMN PAGE (insert before §3.1 Header)

**Board 1 is three columns and so is this page.** Left: the modality rail she drew in green. Centre:
her orange column — M0.0 the LEDE, then the tab's cards. Right: her boxed HEADWINDS rail (R17). New
file `apps/lofn-analytics/src/shell/grid.css`, consuming tokens, declaring none.

```css
/* two new tokens in kit tokens.css: --rail-left: 168px;  --rail-right: 248px; */
.app-grid{
  display:grid; gap:24px; max-width:1512px; margin-inline:auto; padding-inline:24px;
  grid-template-columns: var(--rail-left) minmax(560px,1fr) var(--rail-right);
  align-items:start;
}
.rail-modality{position:sticky; top:var(--sticky-h);}
.rail-headwinds{position:sticky; top:calc(var(--sticky-h) + 8px);
  max-height:calc(100vh - var(--sticky-h) - 32px); overflow-y:auto; overscroll-behavior:contain;
  border:1px solid var(--rule); padding:16px; background:var(--paper);}   /* boxed, as drawn. No shadow. */

@media (max-width:1099px){                    /* TABLET — the left rail yields first (R18) */
  .app-grid{grid-template-columns:minmax(0,1fr) var(--rail-right); gap:20px;}
  .rail-modality{grid-column:1 / -1; position:sticky; top:var(--sticky-h);
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px;}   /* horizontal strip, both bars kept */
}
@media (max-width:859px){                     /* MOBILE — one column, nothing hidden */
  .app-grid{grid-template-columns:minmax(0,1fr);}
  .rail-headwinds{grid-column:1 / -1; position:static; max-height:none; overflow:visible;}
  .rail-modality{position:static;}
}
@media (max-width:419px){ .rail-modality{grid-template-columns:1fr;} }  /* numbers wrap; nothing drops */
```

- **DOM order is `nav.rail-modality` → `main` → `aside#headwinds` at every breakpoint**, which is also
  the visual order at every breakpoint. No reordering, ever (SPLICE 19).
- **Desktop ≥ 1100 px:** exactly Board 1. The centre column never narrows below 560 px; below that the
  1099 breakpoint has already fired.
- **Tablet 860–1099 px:** modality becomes a full-width three-item strip above the content; the
  Headwinds rail keeps its column, its box and its stickiness. The rail is the last thing to leave the
  screen because it is the thing she boxed.
- **Mobile < 860 px:** single column in the same reading order; the Headwinds rail renders full-width
  and unboxed-on-the-sides (top and bottom rules only) at the end of the flow, **never collapsed behind
  a control**, and the header gains a persistent visible chip — `⚠ 3 headwinds` — that is a link to
  `#headwinds` and is the visible twin of skip link #3. Persistence at 375 px is *always visible, one
  tap away, count shown*; a boxed 248 px sidebar at 375 px is not a rail, it is a wall.
- **The grid is identical on every tab**, including M4 (WCAG 3.2.3 · consistent navigation). The
  columns never disappear between tabs — a layout that reflows per tab is a layout the reader has to
  re-learn six times.
- **The Soundry keeps two columns.** Board 3 draws one left rail and one main panel and no right rail;
  inventing one there would be the mirror-image of the bug this splice fixes.

# SPLICE 22 — [NEW] PART 2 §8A · THE HEADWINDS RAIL (insert immediately after §8's intro paragraph;
REPLACE that paragraph's final sentence)

**REPLACE** *"The three highest-severity items also mirror into a persistent right rail on every other
tab, because that is where she drew them."* **WITH** the following card spec. Twenty words was not a
build.

> ### M4.0 · THE HEADWINDS RAIL — *her right-hand box, restored*
> New files: `apps/lofn-analytics/src/shell/headwinds-rail.ts`,
> `src/signals/standing.ts` (the ordering function, the **only** implementation, shared with M4.1 and
> M4.2), `tests/headwinds-rail.spec.ts`. Card id `m4-0-rail`. Registered in the card registry like any
> other card, so `card-registry-check.mjs` holds it to `evidence`, `question`, and `drill-or-reason`.
>
> **What it renders.** Exactly three rows, the top 3 of `rankStanding(modality)` — the same
> severity × confidence ordering M4.2 plots, imported, never re-derived. Each row, at 48 px minimum:
> - direction glyph — `▼` ember for a headwind, `–` `--absent` for DIRECTION UNKNOWN (M4.4's state is a
>   first-class row, not an omission);
> - the claim in **her** vocabulary, ≤ 60 characters, from `M4.3`'s register — never generated prose;
> - the number with its `n`, an as-of chip (relative + absolute, tabular-nums, §6.2), a provenance glyph;
> - `[why →]` — routes to M4 with that row isolated and the filter state carried (§3.4);
> - a guidance trigger whose `RegionNote`/`ControlNote` carries **the falsifier** (§8 M4.3: *"a headwind
>   with no falsifier is a mood"*), plus the rail footer line *"Every row has a falsifier. Open a row to
>   read it."* — so the falsifier is one keystroke away and its existence is visible without one.
>
> **What it refuses.** No sparkline, no severity pill, no badge, no colour-only encoding, no ranking
> number, no count-up, no sort control. The rail asserts no comparison class of its own, **which is the
> whole of §8's objection, honoured** (R17).
>
> **Its last line, always rendered, never a tooltip:** *"3 of 12 standing headwinds. The rest — with the
> tailwinds on the same scale — are on HEADWINDS →"*.
>
> **On M4 itself** the column holds the rail's slot with one line: *"You are in HEADWINDS. These three
> are rows 1–3 below."* The grid never changes shape between tabs.
>
> **Dedupe against the LEDE (the Headline Editor's boundary, made mechanical).** If a rail row's
> `insight.id` appears in M0.0's Zone 0, 1 or 2 on the currently rendered tab, the rail row prints the
> claim and *"in the lede ↑"* and suppresses its number. The LEDE ranks **this snapshot's movement**;
> the rail lists **standing, dated conditions**. Two objects, two jobs; a reader who sees the same
> number twice on one screen learns to trust neither. Asserted by `headwinds-rail.spec.ts`.
>
> **Empty state.** Zero standing rows for a modality ⇒ `needs-declaration` with one `UnlockCounter`:
> label *"registered standing headwinds"*, `have: 0`, `need: 1`, `blocker: "decision"`, `how`: *"No
> command exists. The standing register is hand-maintained and reviewed —
> `apps/lofn-analytics/src/signals/standing.ts`."*, `path` the same file. **Never a blank box, never a
> collapsed column** — an empty rail with a reason is information; an absent rail is a reflow.
>
> **`RegionNote`:** name *"Headwinds"* · forWhat *"The three standing conditions most likely to be
> costing us today, on every screen."* · notHere *"No tailwinds and no ranking. Both are on HEADWINDS,
> on one scale."* · populatedBy `["suno","socials","distribution"]`.
>
> **Guidance/keyboard:** 7 stops total — 3 rows × 2 + the footer link. The heading is `<h2>` inside
> `<aside aria-labelledby>`. The rail is not a `region` landmark duplicate: one `aside`, one label.

# SPLICE 23 — [MERGED] THE NON-NUMERIC ELEMENTS THIS PATCH NOW COVERS BY NAME

Anchored where each lives; every one gets a registry entry and appears in the Glossary.

| element | anchor | subject | the line that closes the gulf |
|---|---|---|---|
| Modality rail items | PART 2 §2 | control | *after:* "Every card on every tab re-reads for Art. Your filters survive." |
| Tab links M0–M6 | PART 2 §3.5 | region | each tab's `forWhat` + `notHere`; M6 CONFLUENCE's `notHere`: *"No randomised result is here until a round closes."* |
| Filter chips (incl. dimmed) | PART 2 §3.3, PART 1 §3.4 | control | `blocked.reason` visible, plus the strip's persistent not-used-here line (R15) |
| Play-floor slider | PART 2 §3.3 | control | paired number input + steppers (rule 12) |
| COMB brush | PART 1 §7.1 | control | keyboard edges + **Set cohort from selection** button |
| `[why this →]` / every `↗` | §14.1 | control | *does:* "Opens the evidence one level down." *reverses:* "Browser Back." |
| Refresh (suno) | §14.4 | control | consequence text verbatim; `writes: {target:"disk", path:"data/snapshots/suno/"}` |
| Refresh (competitors · market · soundry) | §14.4 | control, **blocked** | R14 form; `market`'s `how: null` — *"a file to write, not a command to run"* |
| Every empty state's Copy button | §6.1 | control | *"Copies one line. Nothing runs. This app makes no network request."* |
| M5.7 slot chip / handle field / disqualifier chip / Copy / Download / **DRAFT band** | SPLICE 10 | control ×5 + state | R20 replaces `GuidanceNote` with `ControlNote`; the DRAFT band is `role="status"` and its note reads *"Nothing here is on disk. Closing this tab loses it. That is deliberate — only you write that file."* |
| Ghost unlock counters | SPLICE 3 | state | a `decision` counter renders no command and says none exists (SPLICE 3 rule 6) |
| Join-yield fractions (X0) | SPLICE 8 | statistic | `resolved/attempted` as a fraction + the `keyRule` under the plot (V20) |
| Posture indicator + *"This app reads. It never posts, renders, or spends."* | PART 2 §3.1 | region | notHere: *"No control on this page can publish or spend. The ones that look like they could are disabled and say why."* |
| Reconciliation pill | PART 2 §3.1 | statistic | unchanged, now registry-covered |
| Theme / motion toggles | PART 2 §3.1 | control | icon-only, `aria-label`, three-state motion (SPLICE 20) |
| Skip links (5) | SPLICE 19 | control | visible on focus; #3 is the rail |
| **The Soundry ask box** *"What should we try today?"* | PART 3 §1.1 A0 | control | **visible `<label>`, never a placeholder-as-label** (rule 11). Its submit is the estate's archetypal blocked control (R14): `aria-disabled="true"`, visible reason *"Nothing writes the twelve prompts yet. The Anvil specifies the round; no generator is wired, and this app may not call one (§0.1). Declare the factor and the power first — those are real."* `unblockedBy: "decision"`, `how: null`, `deeper` → STILL OPEN #4. **The box still accepts and stores the text**, because an ask you cannot type is worse than an ask you cannot submit. |
| Soundry blind toggle · arm chips · round selector | PART 3 §3.3 | control | scope `both-apps` where the key is global |
| Panel consent line (40 words) | PART 3 §0.1 | region | `role="region" aria-labelledby`; the note names what is recorded, what is not, and who reads it — the consent text is the guidance, not a link to it |

# SPLICE 24 — [MERGED] VALIDATORS, TESTS, FORBIDDEN STRINGS

**New validator rules (§2.3 `validate.ts`), continuing round one's V19–V23:**
- **V24** — `spec.ghost === true` ⇒ generated alt contains the literal *"No marks are drawn"* and none
  of `highest`, `lowest`, `average`, `median`; no digit outside a counter, an axis domain, or a date.
- **V25** — every figure's accessible name is unique within one rendered tab.
- **V26** — emitted markup containing the HTML `disabled` attribute fails the build (R14).
- **V27** — `aria-disabled="true"` without `data-disabled-reason` of ≥ 8 words fails the build.

**PART 2 §15 test contract, items 27–31** (round one's R10 numbering ends at 26; no collision):

27. `guidance-coverage.spec.ts` — the six assertions (a)–(f) in SPLICE 17. **The gate Board 7 asked for.**
28. `focus-order.spec.ts` — the eight-step order in SPLICE 19 on every tab; `<main>` precedes
    `<aside id="headwinds">` in the emitted HTML at every breakpoint class; five skip links present and
    in order; no `order:` or `*-reverse` in the built CSS.
29. `headwinds-rail.spec.ts` — the rail renders on **every** Analytics tab id; exactly 3 rows;
    `rankStanding()` is the only ordering path (M4.1, M4.2 and the rail agree on every fixture); the
    M0.0 dedupe suppresses the number and prints *"in the lede ↑"*; zero-row fixture renders
    `needs-declaration` with a `decision` counter and no blank box; no sparkline, badge or pill in the
    output; the footer routing line is byte-exact.
30. `layout-columns.spec.ts` — the three grid templates and their breakpoints exist and are the only
    `grid-template-columns` declarations on `.app-grid`; the mobile header chip renders below 860 px
    only and its `href` is `#headwinds`; `--sticky-h` is 138 px on Music tabs and 116 px elsewhere;
    every focusable carries `scroll-margin-top`.
31. `reduced-motion.spec.ts` — SPLICE 20's grep contract; three-state toggle round-trips through
    `localStorage`; `data-motion-ready` gates the pre-paint clamp; `@keyframes` count is zero.

**Kit tests (`apps/lofn-kit/tests/`, added to the §8 tree):** `guidance-contract.spec.ts` (all four
subject types; `registerGuidance` throws on a duplicate id; a `ControlNote` rendered `aria-disabled`
with `blocked: null` throws), `scan.spec.ts` (the scanner finds every interactive tag and role, with a
fixture per element type), `alt-text.spec.ts` **extended** to ghosts, empty states, the comb, the day
mark, the dial and the rail bars.

**`forbidden-strings.spec.ts` / `no-forbidden.spec.ts` — appended to round one's merged list:**
`" disabled="` (outside `aria-disabled=`), `"scroll-behavior: smooth"`, `"order:"` in the app sheets,
`"flex-direction: row-reverse"`, `"flex-direction: column-reverse"`, `"title="`, and — in guidance copy
only — `"not available"`, `"coming soon"`, `"try again later"`. A reason that names nothing is not a
reason.

# SPLICE 25 — [MERGED] PART 1 §8 AND PART 2 §15 FILE TREES; BUILD ORDER

**Kit (`apps/lofn-kit/src/`):**
```
a11y/guidance.ts        a11y/guidance-render.ts     a11y/scan.ts
shell.css               # the grid, the rails, the focus/motion clamps. Consumes tokens; declares none.
tokens.css              # + --rail-left: 168px  --rail-right: 248px  --sticky-h (set by the shell)
tests/guidance-contract.spec.ts   tests/scan.spec.ts
```
**Analytics (`apps/lofn-analytics/src/`):**
```
shell/headwinds-rail.ts   shell/grid.css   shell/skiplinks.ts
signals/standing.ts       # rankStanding() — the ONLY ordering implementation, shared with M4.1/M4.2
cards/m4-0-rail/
tests/guidance-coverage.spec.ts  focus-order.spec.ts  headwinds-rail.spec.ts
      layout-columns.spec.ts     reduced-motion.spec.ts
```
**Soundry (`apps/soundry/`):** `tests/guidance-coverage.spec.ts`, `tests/reduced-motion.spec.ts`.

**Build order — insert a new lane, and one dependency edge:**

- **L-A11Y · THE GUIDANCE CONTRACT AND THE SHELL** (needs L-KIT; parallel with L-SEC-\*, L-WATCH,
  L-FUSE, L-LEDE; **blocks U-FINAL**). `apps/lofn-kit/src/a11y/{guidance,guidance-render,scan}.ts` ·
  `apps/lofn-kit/src/shell.css` · the two new tokens · `apps/lofn-analytics/src/shell/{grid.css,
  headwinds-rail.ts,skiplinks.ts}` · `src/signals/standing.ts` · `src/cards/m4-0-rail/` · V24–V27 in
  `validate.ts` · tests 27–31 + the two kit tests. **Every other lane's cards must register their
  guidance ids as they are written; `guidance-coverage.spec.ts` runs in U-FINAL against all of them,
  and a lane that skipped it fails the build rather than shipping an uncovered control.**
- **U-FINAL gains:** run `guidance-coverage.spec.ts` in both apps and both postures; assert the
  Glossary row count equals the registry size; assert `rankStanding()` has exactly one implementation.

# SPLICE 26 — `## Overrules ACCEPTED`, APPEND

- **THE HEADWINDS RAIL IS RESTORED TO THE RIGHT COLUMN AS DRAWN, AND THE HEADWINDS TAB SURVIVES BENEATH
  IT** (Affordance Coach, over §8). §8's promotion-to-a-tab was an overrule with a real argument and no
  build: twenty words, no card spec, no width, no grid position, no empty state, no file. It is now a
  card (M4.0), a file, a grid column and five test assertions. **PRESERVES §8's premise in full** — a
  warning with no comparison class is unreadable — by giving the rail no ranking, no sparkline and no
  severity encoding of its own, and by ending it with the route to the tab where the ups and downs
  share a scale. **PRESERVES the Headline Editor's boundary**: the LEDE is this snapshot's movement, the
  rail is standing conditions, and the dedupe rule is mechanical rather than remembered.
- **THE MODALITY RAIL IS A RADIO GROUP AND THE TAB BAR IS NAVIGATION** (Affordance Coach, over PART 2
  §2's `role="tablist"`). PRESERVES §2's own ruling — *the rail is not navigation, it is the render of
  the `modality` parameter* — which is the definition of a radio group, and preserves shareable,
  reloadable, Back-button-honest tabs, which `role="tab"` would have quietly broken.
- **"ALL ELEMENTS" IS BUILT AS FOUR SUBJECT TYPES AND ONE COVERAGE GATE** (Affordance Coach). PRESERVES
  §5.1 rule 1's anti-decoration intent while ending its side effect: under the old rule a button could
  not qualify for guidance, so Board 7's clearest sentence was satisfied only for statistics. Nothing
  about the statistic contract changes; three sibling forms and a test are added.
- **NO HTML `disabled` ATTRIBUTE IN EITHER APP** (Affordance Coach). PRESERVES §6.2's *"disabled buttons
  always say why"* and `vault/AUTONOMY.md`'s visibility requirement — a reason that a keyboard cannot
  reach is a reason that was not given.

# SPLICE 27 — `## Overrules REJECTED (her intent wins)`, APPEND

- **REJECTED — closing the Headwinds-rail gap with M0.0, a tab link, a hamburger, or a collapsed
  accordion.** Three of those hide the thing she boxed behind a control; the fourth is a different
  object with a different job. The rail is drawn on every screen or the requirement is dropped, and a
  drop is a bug (the Overrule Mandate).
- **REJECTED — a guidance allowlist ("cover the important controls").** Coverage with an allowlist is
  coverage nobody can check, and the elements that get left off are the small ones — the chip, the `×`,
  the disabled button — which are exactly where the gulfs open.
- **REJECTED — satisfying Board 7 with `title=` attributes or a help page.** §5.1 rule 2 stands; a help
  page is the third door, not the first.
- **REJECTED — hiding the Headwinds rail under `prefers-reduced-motion` or on mobile "for focus".**
  Reduced motion means less movement, not less information (SPLICE 20).

# SPLICE 28 — `# STILL OPEN`, REPLACE items #2 and #3; APPEND #12

- ~~#2 tooltips/alt-text/guidance narrowed to charts~~ — **CLOSED by SPLICE 16–19, 23, 24.** Four
  subject types, one registry, `guidance-coverage.spec.ts` as a build gate, every non-numeric element in
  this estate named in the SPLICE 23 table, and the disabled-control contract (R14). The Glossary's row
  count is now the coverage number, and it is asserted.
- ~~#3 the Headwinds rail is still twenty words~~ — **CLOSED by SPLICE 21–22, 24, 25.** Card id
  `m4-0-rail`, a file, a grid column, three breakpoints, an empty state, a dedupe rule against the LEDE,
  seven keyboard stops, and five assertions in `headwinds-rail.spec.ts`.
- **#12 (new) · THE SOUNDRY HAS NO FILTER SHELF, SO ITS CHIPS HAVE NO GUIDANCE TO REGISTER.** SPLICE 23
  registers the Soundry's blind toggle, arm chips and round selector — but STILL OPEN #3 (two
  incompatible filter vocabularies; PART 3 renders no chip strip at all) means those controls have no
  surface to sit on. `guidance-coverage.spec.ts` will pass vacuously in the Soundry until U0b lands the
  frozen key union and PART 3 grows a shelf. **Named so that a green test is not mistaken for a covered
  app.** Owner: U0b, not this seat.

---

**One sentence for the record.** Every element on both boards now answers three questions before it is
touched — *what is this, what will it do, and why can it not do it right now* — and the only object she
drew on the right-hand side of the page is on the right-hand side of the page.



---

# APPENDIX — seat patch: THE FORGE MASTER — after Brian Eno (Oblique Strategies, the Ambient series, the studio-as-instrument position) (Read from disk, no pastes: BUILD_SPEC.md Part 3 §0–§7 (the Soundry: §0.1 autonomy table, §0.2 the clip_id join, §1.1–§1.8 the Anvil, §3.1–§3.4 the ledger, §5.4 the fourteen refusals, §6 required specs, §7 voice) and Part 2 §0/§14.4; SPEC_PATCH.md §M rulings R1–R12, SPLICEs 1–13, the Overrules-ACCEPTED block, the eleven STILL OPEN items (item #4 on line 635 is this assignment) and the revised build order; WHITEBOARD_TRANSCRIPTION.md Boards 2, 3, 7, 8 and the Overrule Mandate; vault/AUTONOMY.md Part 1 (the draft/render/spend bulkhead, AGENT FIX DON'T BYPASS, the Andon Cord); vault/EXPERIMENT_STANDARD.md §1–§5 (4.49% baseline n=46, 58% binomial share, the +15%/19-day and 8-day power points, "randomise WITHIN the day and block on day", "no per-song verdicts", pre-registration before generation); vault/SOCIAL_VOICE.md §7, Appendix C (the five slots, the STANDING RULE), Appendix D (no product retuning, no caption calendar, no invented production history); vault/gates.yaml (850–1000 / target 870–960 / hug 985 / terminal punctuation; sung_lines 70–120 target 78–110 floor-hug 72; rhyme_return_floor 0.30, line_return_floor 0.20, mean_words_per_line_ceiling 7.5, alliteration_per_100w_floor 11.0, rhyme_window 4; portfolio_max_prompt_similarity 0.58, portfolio_max_ngram_jaccard 0.18; chorus_repetition_requires_no_justification); .claude/skills/lofn-music/SKILL.md (the Golden Move 1–6, THE RETURN, the Rhyme Debt rule, the mandatory PROSODY axis, the Suno output contract incl. the 400–900 exclude band and the <5000 lyrics cap, "never hand an agent a numeric target without the function that computes it"); verified on disk that scripts/measure_soundcraft.py and scripts/validate_step.py both exist, which is what makes the measure route legal instead of a reimplementation. Eno grounding used as method, not decoration: a constraint deck that opens a space rather than picking an output; "honour thy error as a hidden intention" implemented as the deviation switch that feeds the deck; the refusal to let chance dress up as creativity implemented as a seed that is a pure function of the frozen record.)

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/SPEC_PATCH.md, E:/git/private-claw/apps/soundry/src/anvil/deck.ts, E:/git/private-claw/apps/soundry/src/anvil/twin.ts, E:/git/private-claw/apps/soundry/src/anvil/slots.ts, E:/git/private-claw/apps/soundry/src/anvil/exclude.ts, E:/git/private-claw/apps/soundry/src/anvil/handoff.ts, E:/git/private-claw/apps/soundry/src/anvil/import.ts, E:/git/private-claw/apps/soundry/src/anvil/measure.ts, E:/git/private-claw/apps/soundry/src/anvil/novelty.ts, E:/git/private-claw/apps/soundry/src/anvil/deck.data.json, E:/git/private-claw/apps/soundry/src/anvil/factors.data.json, E:/git/private-claw/apps/soundry/src/sampler.data.json, E:/git/private-claw/apps/soundry/server.mjs, E:/git/private-claw/apps/soundry/tests/deck.spec.ts, E:/git/private-claw/apps/soundry/tests/twin.spec.ts, E:/git/private-claw/apps/soundry/tests/exclude.spec.ts, E:/git/private-claw/apps/soundry/tests/slots.spec.ts, E:/git/private-claw/apps/soundry/tests/handoff.spec.ts, E:/git/private-claw/apps/soundry/tests/import.spec.ts, E:/git/private-claw/apps/soundry/tests/caption-refusals.spec.ts, E:/git/private-claw/apps/soundry/tests/gates-parity.spec.ts, E:/git/private-claw/apps/soundry/tests/design.spec.ts, E:/git/private-claw/apps/soundry/tests/refusals.spec.ts, E:/git/private-claw/apps/soundry/tests/server-soundry.spec.ts, E:/git/private-claw/apps/lofn-kit/src/contracts/empty.ts, E:/git/private-claw/apps/lofn-kit/src/contracts/ghost.ts, E:/git/private-claw/data/soundry/deck-ledger.json, E:/git/private-claw/vault/gates.yaml (READ ONLY — quoted, never restated), E:/git/private-claw/scripts/measure_soundcraft.py (READ ONLY — shelled, never ported), E:/git/private-claw/scripts/validate_step.py (READ ONLY — shelled, never ported)

# PATCH — STILL OPEN #4 (the moderator's "biggest"): **THE SONIC ANVIL HAS NO GENERATOR**

Seat: **THE FORGE MASTER**. Target file:
`E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`, Part 3.
Every splice below names its anchor. Nothing here invents a number: every threshold is quoted from
`vault/gates.yaml`, `vault/EXPERIMENT_STANDARD.md`, or the spec's own frozen `src/design.data.json`.

---

## 0 · The ruling, stated before the splices

The spec builds an experiment and forgets to build the instrument that plays it. But the fix is not
"add a writer to the app." **The Anvil is a constraint deck, not an author.** It fixes what must be
fixed for twelve cards to be one experiment, leaves everything else genuinely open, and hands the
open part to `lofn` / `lofn-music`, which is the only thing in this estate licensed to write a song.

Three reasons, and the third is the one that matters.

1. **Mechanics.** §0.1 pins `connect-src 'self'`, forbids a `dependencies` key, forbids any
   generation API and any API key. There is no legal path by which a browser bundle writes an
   850–1000-character style prompt and a 70–120-line lyric.
2. **Doctrine.** `vault/AUTONOMY.md` puts the machine up to the draft and a human across
   render/publish/spend. Twelve drafts on disk is the terminal state, and it already is.
3. **Craft, which is the real argument.** A slot grammar rich enough to emit twelve publishable
   style prompts would emit twelve paragraphs off one skeleton — the exact "pair bleed" the music
   pipeline fails runs for (`portfolio_max_prompt_similarity: 0.58`). It would be a second, worse
   pipeline competing with the first, and it would only ever recombine what already worked. **A
   generative system's value is the space it opens, not the output it picks.** The Anvil's job is to
   open a space narrow enough to be an experiment and wide enough to be new.

So the generator is specified in three movements with one hard seam:

| | movement | who | writes |
|---|---|---|---|
| **I** | **THE DECK** — compile the ask into twelve cartridges | the app, deterministic, offline | the design, the exclude prompts, the slot sheets, the return contract |
| **II** | **THE HAND** — write the prose | `lofn` → `lofn-music`, a session she starts | Title, Style Prompt, Lyrics |
| **III** | **THE ANVIL RECEIVES** — import, measure, refuse | the app, shelling the shipped measurers | the gates, the twin-drift check, the diversity census |

The Caption is a fourth thing and is specified separately (§1.4E) because it is the only field that
would ever leave the machine.

---

## SPLICE F1 — **new §1.3A, THE EXPERIMENT AND THE DAY-BLOCK**
**Anchor:** insert immediately after §1.3 (A2 — the power card), before §1.4.

> ### 1.3A A2b — One experiment, many day-blocks *(this resolves a live contradiction in Part 3)*
>
> §1.3 computes **19 days at 12 cards/day** for a +15% CONFIRM round. §1.6 draws **twelve cards**
> with a header reading `4 of 12 exported · A 6 / B 6`. Both are correct and they are not the same
> object. As written, `round` means both a twelve-card grid and a nineteen-day design, and a builder
> cannot implement both.
>
> **Split the noun.**
>
> | | **EXPERIMENT** | **DAY-BLOCK** |
> |---|---|---|
> | id | `X-<slug>` | `R-YYYY-MM-DD-<slug>` (unchanged) |
> | span | until `planned_n_per_arm` is reached (~19 days at +15%) | one calendar day, twelve cards |
> | carries | `preregistration.json`, `spec_hash`, the factor and its levels, the MDE, the analysis plan | `design.json`, `variants/`, `observations/`, `events.jsonl` |
> | frozen | once, before the first day-block generates | never — it is a draw, not a decision |
> | randomisation | — | **within the day, blocked on day** (`vault/EXPERIMENT_STANDARD.md` §4.1) |
>
> `data/soundry/experiments/<X-id>/preregistration.json`; day-blocks stay at
> `data/soundry/rounds/<R-id>/` and each stamps `experiment_id` + the inherited `spec_hash`.
> §1.5's **423 Locked** now reads: one open EXPERIMENT at a time, and within it one open DAY-BLOCK.
> §3's emit rule reads: `emit-snapshot.mjs` exports **closed day-blocks of a frozen experiment**.
>
> **Why this is the whole reason the Anvil is used daily.** The hypothesis is not re-invented every
> morning — that is the garden of forking paths with a nice UI. What is re-drawn every morning is the
> *frame*: themes, personas, keys, tempos, domains, obliques. The factor is boring and stays boring
> for nineteen days. **The novelty budget is spent entirely on things the design holds balanced.**

---

## SPLICE F2 — **REPLACE §1.4 (A3) in full: THE DECK, THE TWINS, AND THE SPREAD**
**Anchor:** `### 1.4 A3 — How the twelve are diversified`, replacing through the end of PLOT A3.2.

> ### 1.4 A3 — The deck: how the twelve are constructed
>
> **The unit of the grid is not the card. It is the TWIN.** Twelve cards are **six twins**: two cards
> that share one complete nuisance draw and differ in exactly one thing — the factor level.
>
> ```
> twin t  ->  card t.MINUS  (factor at level_minus)   arm A
>         ->  card t.PLUS   (factor at level_plus)    arm B
> shared: seed theme · persona · key/mode · tempo band · lyric domain · oblique · SOUND/RETURN axis
>         · song form · target sung-line count · target duration · the whole exclude prompt except (b)
> ```
>
> Pairing is not an aesthetic preference. It makes the factor **orthogonal to every nuisance by
> construction rather than by luck**: every nuisance level appears exactly as often at `−` as at `+`,
> so the design matrix's correlation between the factor column and every nuisance indicator is
> **exactly 0**. The app computes that maximum and prints it on A3.2 as
> *"factor × nuisance, max |r| = 0.00 — exact, by pairing"*, and `design.spec.ts` asserts the literal
> zero. §1.4's previous stratified-independent sampling produced visible imbalance about half the
> time at n=6; pairing removes the failure mode instead of detecting it.
>
> **Consequences, stated so a builder does not re-add what pairing deletes:**
> - **The reshuffle affordance is struck in CONFIRM.** A3.2's `IMBALANCED — reshuffle` state can no
>   longer occur; if it renders, it is a bug, and the copy is now
>   `IMBALANCE UNDER A PAIRED DESIGN — this is a defect, not a draw. Do not reshuffle; file it.`
> - **Twins are never split across days.** Both members of a twin post on the same day. A twin split
>   across a day boundary confounds the factor with the block that exists to absorb the calendar
>   (`vault/EXPERIMENT_STANDARD.md` §4: month drift is 1–2 pp, larger than anything being tested).
>   `POST /soundry/rounds/:id/generate` refuses a draw that would split one, naming it.
> - **SCREEN cannot twin.** A 12-run Plackett–Burman has no partner rows. In SCREEN the deck balances
>   each nuisance dimension across each factor's ± column as far as PB permits, prints the realised
>   max |r| (which will not be 0), and the mode banner gains a second sentence:
>   *"Nuisance is balanced, not paired. The max nuisance correlation on this design is {r}."*
>
> #### 1.4a The draw is a function of the frozen record
>
> ```
> deck_seed = SHA-256( experiment_id + "|" + dayblock_date + "|" + prereg.spec_hash )
> ```
> There is **exactly one legal draw per experiment per day.** Re-opening the page redraws nothing;
> the twelve are reproducible from the seed alone and a test asserts byte equality (§6, extended
> `design.spec.ts`). This is the anti-habit control that matters most: a reshuffle button in front of
> a person with taste is a slot machine that pays out in confirmation, and the whole apparatus then
> measures the operator's mood. In SCREEN, where reshuffling is legal, every redraw appends a
> `redraw` event and the card carries a visible counter — the same discipline as B1's peek count, and
> for the same reason.
>
> #### 1.4b The pools, and the one that is read rather than stored
>
> `src/sampler.data.json` keeps the five nuisance dimensions of the previous §1.4 (seed theme ·
> persona · key/mode · tempo band · lyric domain), now drawn **per twin** rather than per card.
>
> **Genre is not one of them, and it is not stored as a number.** The deck partitions the genre pool
> into three strata — `ABOVE` / `AT` / `BELOW` the era-controlled baseline — and it reads that
> partition **at compile time from the M2.4 style board's output in `data/snapshots/suno/latest.json`
> plus the board's own era control**. It is never hard-coded. The record motivating the rule is real
> and stays in prose, not in a data file: warmth over performance — piano, strings, acoustic, folk,
> ambient above the modern baseline; `snarl` at 3.44% and `phonk` at 3.18% the two worst, against a
> 4.49% modern baseline (n=46). If the style board is absent or its era control is `false`, the deck
> **does not stratify**: it draws uniformly, the stratum chip renders `emitter-silent`, and the card
> says *"No era-controlled style board on disk. Genres drawn uniformly.
> `python3 scripts/ingest_suno.py` is what makes this rule available."*
>
> #### 1.4c THE OBLIQUE — one per twin, drawn least-recently-used
>
> New file `src/deck.data.json`. Not adjectives. **Constraints**, each of which forecloses a habit
> and opens a space:
>
> ```json
> { "id": "ob.late-low-end", "family": "arrangement",
>   "constraint": "the low end arrives after the first chorus and never leaves",
>   "forbids": ["sub-bass from bar 1", "full-band cold open"],
>   "conflicts_with_factors": ["duration"],
>   "provenance": "seed", "last_used_round": null }
> ```
>
> - **One oblique per twin, shared by both members.** It therefore cannot be confounded with the
>   factor. This is the rule that lets the deck be wild without costing the contrast a thing.
> - **Six distinct obliques per day-block, from six distinct families**, drawn **least-recently-used**
>   against `data/soundry/deck-ledger.json`, ties broken by `deck_seed`. Not uniform random: uniform
>   random over nineteen days revisits and leaves whole regions untouched. LRU guarantees the space
>   is traversed.
> - **An oblique whose `conflicts_with_factors` contains the live factor is skipped**, and the skip
>   is printed on A3.1 with the reason. An oblique that fights the thing under test is not an
>   experiment, it is a collision.
> - **The deck grows from returned work, and only by her hand.** Every `## Major Deviations` line in
>   a returned step-11 package (§1.4F) is classified `KEEP` / `NOTED` / `INVALIDATES`. A `KEEP`
>   becomes a new deck card with `"provenance": "deviation:<variant_id>"` and `last_used_round` set to
>   the round it came from, so it enters the LRU at the back. *Honour thy error as a hidden
>   intention* — with a switch, so it is honoured rather than obeyed. **No deviation ever enters the
>   deck automatically**, and `deck.spec.ts` asserts no code path writes `deck.data.json`.
>
> #### 1.4d THE SPREAD — the diversity contract, measured, not felt
>
> The six twins are chosen from the candidate pool by a **seeded farthest-point traversal** over the
> six-dimensional draw (5 nuisance + oblique family) under Hamming distance. The contract, all
> asserted by `deck.spec.ts`:
>
> | # | rule | floor |
> |---|---|---|
> | D1 | pairwise Hamming distance between any two twins | **≥ 3 of 6** |
> | D2 | occurrences of any one nuisance level across the six twins | **≤ 2** |
> | D3 | twins drawn from the `BELOW`-baseline genre stratum | **≥ 1** (default 1, operator range 1–2) |
> | D4 | twins drawn from the `ABOVE` stratum | **≥ 1** |
> | D5 | distinct SOUND/RETURN axis values, one per twin | **6 of 6** |
> | D6 | distinct oblique families | **6 of 6** |
>
> **If the pool cannot satisfy a floor, the deck refuses and names the dimension that is too small.**
> It does not quietly emit five twins, and it does not quietly repeat a level. The copy is
> *"Cannot satisfy D1 at ≥3: the persona pool has 4 entries and 6 twins need 6-way spread. Widen
> `src/sampler.data.json#persona` or lower the floor on this experiment's design step."*
>
> **D3 is the rule this bench exists for.** A system that only samples the corner that already works
> stops learning, and its measured advantage decays into a house style. One twin of six comes from
> the register that measures worst, it says so on its face — *"drawn from BELOW: this register
> under-performs our baseline. It is here so the estimate is about the factor and not about the
> corner we like."* — and because the design is paired, an unusual genre inflates neither arm's mean
> nor the contrast. It widens its own twin's residual and nothing else. **Novelty is bought in the
> frame and paid for by the twin.**
>
> #### 1.4e Slot order, still
>
> Slot order over the twelve is randomised from `deck_seed` and recorded as a covariate, absent from
> the primary model — unchanged from the previous §1.4, and now additionally constrained so that no
> twin's two members occupy adjacent slots (adjacent renders share a queue state and a mood).
>
> - **PLOT A3.1 — Design matrix.** As before, plus: `y` groups by twin, and a right gutter column
>   prints each twin's oblique id and stratum. **Question:** *is this a design, or twelve vibes?*
> - **PLOT A3.2 — Orthogonality and spread.** Replaces the nuisance-balance bars. Left: the printed
>   `max |r| = 0.00` with every nuisance indicator listed at its realised correlation. Right: the
>   6 × 6 Hamming matrix as a strip, with any cell below 3 in ember. **Question:** *can this design's
>   contrast be about anything except the factor?*

---

## SPLICE F3 — **new §1.4A, THE QUAD: what is machine-written, what is never, and why**
**Anchor:** insert immediately after the new §1.4, before §1.5 (A4 — pre-registration).

> ### 1.4A The five fields, ruled one at a time
>
> Board 3 names five fields per card: **Title · Style Prompt · Exclude Prompt · Lyrics · Caption.**
> The Anvil does not write three of them, writes one of them completely, and holds the fifth until a
> song exists to caption.
>
> | field | who writes it | the app's contribution |
> |---|---|---|
> | **Title** | `lofn-music` | a constraint, not a string |
> | **Style Prompt** | `lofn-music` | the **slot sheet**: six clause slots, the design-fixed content of each, and a character budget that sums into the band |
> | **Exclude Prompt** | **the app, in full, deterministically** | the whole field |
> | **Lyrics** | `lofn-music` | the **return contract**: form, sung-line target, EMO skeleton, the four return floors, the SOUND/RETURN axis |
> | **Caption** | `lofn-music`, after clip binding | the refusal list (§1.4E) |
>
> #### 1.4A-i Title
> Never machine-written. The card carries `title_constraint`, drawn from the twin's frame:
> *"≤ 5 words. The subject is named plainly (Golden Move 6). Not a phrase from
> `vault/gates.yaml#house_lexicon`."* Twelve titles off one grammar is twelve titles with one
> skeleton, and the estate already fails runs for that.
>
> #### 1.4A-ii Style Prompt — the SLOT SHEET
> `src/slots.ts`. Six slots in the mandatory order the Suno contract freezes
> (`.claude/skills/lofn-music/SKILL.md`), each with a **fixed** part the deck writes and an
> **unwritten** span the hand fills:
>
> | slot | fixed by the deck | budget (chars) |
> |---|---|---|
> | S1 genre / micro-genre · tempo · energy · opening moment | genre + stratum, tempo band, key/mode | 150–200 |
> | S2 vocalist spec + spatial staging | register (if the factor is `register`, this is the arm) | 130–180 |
> | S3 instrumentation / palette, physical adjectives | the oblique's required or forbidden instrument | 180–240 |
> | S4 arrangement arc | the oblique's arc constraint; the factor if the factor is structural | 150–200 |
> | S5 the bold sonic device | one device, named by the oblique, never by the deck's taste | 90–140 |
> | S6 acoustic / no-acoustic declaration + kinetic defect | the defect class, held across the twin | 60–100 |
>
> Budgets sum to **760–1060**; the compiler solves them into `music_prompt_chars_target: [870, 960]`
> and prints the target per slot. The app **never emits prose into a slot.** The Style well renders the
> sheet at full ink with the unwritten spans hatched — the ghost contract's render rule (SPLICE 3,
> rule 5) applied to text instead of axes: *the shape is known, only the words are missing.*
>
> **On return, the app measures and refuses** (§1.4G): exact char count against `[850, 1000]`,
> boundary-hug FLAG at `≥ 985`, terminal punctuation as a **hard fail**, the banned-opener regex, and
> `house_lexicon` hits as a FLAG. Every one of these numbers comes from `vault/gates.yaml`; none is
> restated in the app's own constants, and `gates-parity.spec.ts` asserts it.
>
> **THE TWIN-DRIFT GATE — the single most load-bearing new check in this patch.**
> Twin members must differ **only** in the factor's clause. The app diffs the two returned style
> prompts at token level, per slot:
> - non-factor slots: similarity **≥ 0.90**; below that ⇒
>   `TWIN DRIFT — slot S3 differs by 41 tokens. This contrast is contaminated. Neither card can be
>   marked ready.`
> - the factor slot: similarity **< 0.90** required; above that ⇒
>   `NULL ARM — the two arms say the same thing. There is no contrast to measure.`
>
> Cross-twin, the pipeline's portfolio ceilings apply **once per twin skeleton, not per card**:
> `portfolio_max_prompt_similarity: 0.58` over the six skeletons, `portfolio_max_ngram_jaccard: 0.18`
> over the six lyric sets. This is the seam where the Soundry's requirement (twins must be nearly
> identical) and the pipeline's requirement (pairs must be distinct) would otherwise fight; the
> handoff (§1.4F) carries the configuration that resolves it, so the pipeline is told, not surprised.
>
> #### 1.4A-iii Exclude Prompt — written in full, by the app
> `src/exclude.ts`. This field is a negative token list, not prose. It is a constraint, which is what
> this app is for, and it is therefore the one field the deck writes end to end. Four categories,
> concatenated in fixed order and deduplicated:
>
> ```
> (a) STANDING     src/deck.data.json#exclude.standing — the house failure classes
> (b) COUNTER-ARM  the signature terms of the OPPOSITE factor level        <- differs across the twin
> (c) OBLIQUE      the twin's oblique.forbids[]
> (d) STRATUM      the genre stratum's named failure classes
> ```
>
> - **(b) exists only when the factor is negatable.** `src/factors.data.json` carries
>   `"negatable": true|false` per factor with a written reason. `register {plain | ornate}` is
>   negatable; `words_per_line {≤5 | ≥7}` is not — Suno does not take a line-length negative. When
>   `negatable` is false the card prints *"This factor is enforced in the lyric and the slot sheet,
>   not in the exclude field."* and (b) is empty. An arm that thinks it is enforced twice and is
>   enforced once is a silent design failure.
> - Band: **400–900 chars, hard max 1000** (the Suno contract). If the union exceeds 900 the compiler
>   drops the **lowest-priority whole category** — order (d), then (c) — and **names the dropped
>   category on the card**. It never truncates mid-token: a truncated negative list silently changes
>   which arm you are running.
> - If the union is under 400 the card says *"Exclude field under band at {n}. Add house standing
>   terms in `src/deck.data.json`; the app will not pad."* Padding a negative list with filler is
>   inventing a constraint.
>
> #### 1.4A-iv Lyrics — the RETURN CONTRACT
> Never machine-written. The app fixes, per twin, before the hand writes:
>
> - **Song form** and the section sequence (shared across the twin).
> - **Target sung lines**, drawn inside `sung_lines_target: [78, 110]` — **never at 70, never at 72**.
>   The floor is a floor; a set pinned at the floor is optimising the gate
>   (`sung_lines_floor_hug: 72`).
> - **The EMO header skeleton** — `[Section - EMO:… - <Role> - <cues>]` with the emotion span left
>   unwritten, because the taxonomy is the pipeline's and the app must not pick from it.
> - **The four return floors, quoted live from `vault/gates.yaml`, never restated:**
>   `rhyme_return_floor 0.30` · `line_return_floor 0.20` · `mean_words_per_line_ceiling 7.5` ·
>   `alliteration_per_100w_floor 11.0`, with `rhyme_window 4` printed beside the first because a
>   threshold without its window is three different thresholds.
> - **The SOUND/RETURN axis value** for this twin — one of the six distinct values (D5), from the
>   positive vocabulary the music skill mandates at Phase 0.
> - **The Rhyme Debt line**, present only when the twin's axis strips rhyme: *"Name what returns in
>   its place."* Removal is a debt; the brief carries the invoice.
>
> **REPETITION IS NOT A DEBT, AND THE UI PROVES IT BY OMISSION.** There is no justification field for
> a refrain, no "explain your chorus" box, no flag that a byte-identical chorus can trip.
> `gates.yaml#chorus_repetition_requires_no_justification` is honoured here as the *absence of a
> control*, and `refusals.spec.ts` asserts the card object has no `chorus_justification` property at
> any state. A flag that makes a writer apologise for a chorus does harm even when it never fires.
>
> **THE FLOOR-SUSPENSION INTERLOCK.** If the factor under test *is* a return quantity —
> `chorus {exact repeat | varied}`, `words_per_line {≤5 | ≥7}` — then that quantity's floor is
> **suspended for the whole experiment**, and the suspension is written into `preregistration.json`
> as `analysis.suspended_gates[]` at freeze time. You cannot measure a quantity you have pinned to a
> floor; a gate left live would clamp one arm and the round would measure the gate. The card prints
> *"`line_return_floor` is suspended for X-chorus-2026-08. This is pre-registered, not a waiver."*
>
> #### 1.4A-v Caption — see §1.4E. It is not a Suno field and it is not written yet.

---

## SPLICE F4 — **new §1.4E, WHAT A CAPTION IS FOR**
**Anchor:** immediately after §1.4A.

> ### 1.4E The Caption — the only field that would leave the machine
>
> Board 3 lists Caption under *"full suno prompts"*. Suno has no caption field. The caption is the
> **post**, and that makes it the most tightly held of the five.
>
> 1. **It does not exist until `clip_id` is bound.** The well renders `unwritten` with
>    *"No caption before there is a song. Bind the clip (§0.2) and the caption slot opens."* You
>    cannot honestly caption something you have not heard, and `vault/SOCIAL_VOICE.md` Appendix D
>    forbids asserting production facts that were never true.
> 2. **It never names the experiment.** Not the factor, not the arm, not the level, not the round id,
>    not the word *test*. Two reasons and both are real: telling listeners which condition they are in
>    contaminates the instrument, and a closed round's factor is exactly what R7 keeps out of every
>    hosted artefact. `forbidden-strings.spec.ts` gains, scoped to caption text: `"arm A"`, `"arm B"`,
>    `"control"`, `"variant"`, `"experiment"`, `"we're testing"`, `"A/B"`, and every live factor name
>    read from `factors.data.json`.
> 3. **It carries no metric, no ask, no streak, no call to action.** `vault/SOCIAL_VOICE.md` §7 and
>    Appendix D: no follow-for-follow, no engagement request, no counting-up number. Warmth is not a
>    growth surface.
> 4. **It is drafted to the Length Ladder** (`SOCIAL_VOICE.md` §2 / Appendix A), and the well prints
>    which rung it is on and the rung's word band.
> 5. **There is no Post control, no clipboard-to-platform path, and no share sheet.** The caption
>    exports as a file alongside the other four fields (§1.7). Per §1.8: this bench has never had a
>    button for that, and a disabled one would be a promise.
>
> `caption-refusals.spec.ts`: no caption renders while `clip_id` is null; no caption string contains a
> forbidden token at any state; the exported packet contains no URL to any platform.

---

## SPLICE F5 — **new §1.4F, THE HANDOFF: the bench proposes, canon makes**
**Anchor:** immediately after §1.4E.

> ### 1.4F The handoff — how the Anvil hands to `lofn` without becoming a second pipeline
>
> **The Anvil composes; it never runs, and it never fetches.** This is the same compose-don't-write
> discipline the watchlist editor uses (Part 2 M5.7): a `<pre>` block, a Copy button, a written
> consequence line, and the sentence *"This app will not run it. Copy it and run it yourself."*
>
> #### The frozen-frame brief
> `POST /soundry/rounds/:id/generate` (still **409** before pre-registration is frozen — §1.5, and
> the deck now sits behind that gate, which is what makes the gate worth having) writes:
>
> ```
> output/soundry/<R-id>/ROUND_BRIEF.md          the one file she hands to the lofn front door
> output/soundry/<R-id>/twins/<twin-id>.md      one per twin: frame, oblique, slot sheets x2,
>                                               exclude prompts x2, return contract, title constraint
> data/soundry/rounds/<R-id>/design.json        deck_seed, twin table, spread matrix, max|r|, slot order
> data/soundry/rounds/<R-id>/variants/<vid>.json  cartridge + inherited spec_hash + state CARTRIDGE
> ```
>
> `ROUND_BRIEF.md` opens with the frame and the one sentence that keeps two pipelines from becoming
> two pipelines:
>
> > **FROZEN FRAME — day-block `R-…`, experiment `X-…`, spec_hash `…`.**
> > Phase 0 may choose everything except what this block fixes. The factor, its two levels, the six
> > twin frames, the six obliques, the six SOUND/RETURN axes, the slot sheets and the exclude prompts
> > are fixed. **Any departure from a FROZEN line is a round-invalidating deviation: report it under
> > `## Major Deviations`; do not absorb it.** Echo `variant_id` and `spec_hash` verbatim in every
> > returned package. Cross-pair distinctiveness is evaluated **per twin, not per card** — twin
> > members are required to be near-identical outside the factor clause.
>
> It then runs the normal road: the `lofn` front door, Phase 0 Golden Seed (constrained by the frame),
> Phase 1's three panels, dispatch to `lofn-music`, and `lofn-qa`. **The Anvil never invokes a skill,
> never spawns an agent, and has no route that could.** She copies one line.
>
> #### The return
> `POST /soundry/rounds/:id/import { path }` reads step-10/11 packages from `output/` — a local
> filesystem read inside the repo, no network, path-confined to `output/**` and asserted by
> `server-soundry.spec.ts`. It:
> - maps package → variant by the echoed `variant_id`, and **refuses any package whose echoed
>   `spec_hash` differs from the frozen one**, reusing the existing deviation machinery (§1.5) rather
>   than inventing a second one;
> - parses `## Major Deviations` into `variants/<vid>.json#deviations[]`, each `unclassified` until
>   she rules `KEEP` / `NOTED` / `INVALIDATES`. `INVALIDATES` drops the variant into the analysis
>   card's existing exclusions block with its stated reason. `KEEP` offers — never performs — the
>   deck-card write of §1.4c.
> - advances the card `CARTRIDGE → BRIEFED → DRAFT`. Nothing auto-advances past `DRAFT`.
>
> **This is the sibling boundary, and it is the same one `vault/SOCIAL_VOICE.md` draws around the
> music: the bench does not retune the product. It proposes a frame and measures what comes back.**

---

## SPLICE F6 — **new §1.4G, THE MEASURERS ARE SHELLED, NEVER REIMPLEMENTED**
**Anchor:** immediately after §1.4F.

> ### 1.4G Measurement — one route, two scripts already on disk
>
> `scripts/measure_soundcraft.py` and `scripts/validate_step.py` **exist in this repository** and
> `.claude/skills/lofn-music/SKILL.md` states the governing rule verbatim: *never hand an agent a
> numeric target without the function that computes it.* Three repair agents once reverse-engineered
> three different rhyme windows from a brief that named a threshold without defining it. **The app
> must not port `profile_file()` into TypeScript.** A second implementation is a second definition.
>
> `POST /soundry/rounds/:id/measure` — the Soundry's **only** shelling route:
> - runs `python3 scripts/measure_soundcraft.py --json <path>` and
>   `python3 scripts/validate_step.py --json <path>` on files under `output/soundry/<R-id>/` only;
> - is **local, free, reversible, read-only** — squarely inside `vault/AUTONOMY.md` Part 1's
>   safe-autonomous list, and nowhere near render/publish/spend;
> - states its consequence before it runs (Part 2 §14.4 pattern): *"Runs two local Python scripts over
>   twelve files. Writes measurements into the round. Fetches nothing, spends nothing."*;
> - writes `variants/<vid>.json#soundcraft` + `#gate_report` and advances `DRAFT → MEASURED`;
> - **fail-open, loudly**: if either script is missing or non-zero, the strip renders
>   *"MEASURER ABSENT — `scripts/measure_soundcraft.py` did not run. Nothing here is a pass."* The card
>   **cannot be marked ready**, and no number is displayed. `gates.yaml`'s own fail-open contract
>   applies to the thresholds, never to the verdict.
> - `server-soundry.spec.ts` gains: the spawned argv contains no network binary; a path outside
>   `output/soundry/<R-id>/` → **403**; `POST /soundry/render` → **405** (unchanged).
>
> §1.6's derived-measure strip now reads **only** from `#soundcraft` and `#gate_report`. It computes
> nothing itself. **The apps never invent a number**, and after this splice the Anvil never even
> re-derives one.

---

## SPLICE F7 — **§1.6 (A5), amend the card**
**Anchor:** `### 1.6 A5 — The twelve cards`, replacing three bullets and adding four.

> - **State chip**, extended: `CARTRIDGE → BRIEFED → DRAFT → MEASURED → EXPORTED → POSTED →
>   ACCRUING → READ → CLOSED`. Colour + glyph + word. **Nothing auto-advances.** `BRIEFED` requires
>   the handoff files on disk; `DRAFT` requires an imported package with a matching `spec_hash`;
>   `MEASURED` requires a `gate_report`; `POSTED` requires a bound `clip_id`; `READ` requires
>   `age ≥ readout_age_days`.
> - **Twin ribbon**, replacing the arm ribbon: a 4px rule down the left edge carrying the **twin id**,
>   with `CONTROL` framed distinctly. Hovering one member outlines its partner. The card's identity is
>   its twin, not its arm — an arm is six unrelated songs, a twin is a comparison.
> - **Factor fingerprint**, unchanged in form and now trivially satisfiable: under pairing, two cards
>   with identical fingerprints can only be a compiler bug. `DUPLICATE CELL — this card buys nothing.`
>   becomes `DUPLICATE CELL — defect, not a draw.`
> - **NEW · Frame strip.** Six chips: theme · persona · key/mode · tempo · domain · oblique. Chips
>   shared with the twin render graphite; the factor chip renders ember. One glance answers *what is
>   held and what moves*.
> - **NEW · Stratum chip.** `ABOVE` / `AT` / `BELOW` with its n and interval from the style board, or
>   `emitter-silent` and the ingest command (§1.4b). A `BELOW` card carries its one-line reason.
> - **NEW · Unwritten wells.** Title, Style Prompt, Lyrics and Caption render `unwritten` in
>   `CARTRIDGE`/`BRIEFED`, at full ink with hatched spans — the slot sheet, the section skeleton and
>   the return contract are all visible. **An unwritten well never looks like a failed generation and
>   never looks empty.** Its unlock counter carries `blocker: "handoff"` and the copyable `lofn`
>   invocation, never a command that would fail.
> - **NEW · Twin-drift row**, present from `MEASURED`: per-slot similarity, the factor slot's required
>   inequality, and the verdict `CONTRAST CLEAN` / `TWIN DRIFT` / `NULL ARM`. A drifted twin cannot be
>   marked ready — either member.
> - **Derived-measure strip:** as §1.4G. Reads `#gate_report`; computes nothing.

---

## SPLICE F8 — **§1.7 (A6), amend Export**
**Anchor:** `### 1.7 A6 — Export`, append.

> Export writes the **quad packet** `output/soundry/<R-id>/<vid>.md` (Title · Style Prompt · Exclude
> Prompt · Lyrics · Caption, in Board-3 order, each with a per-field copy affordance in the UI) plus
> `variants/<vid>.json`, and it **refuses to export a card whose twin partner is not also ready** —
> exporting half a comparison is how one arm gets posted and the other does not, which is the
> differential attrition §1.6 already names as the most likely silent failure.
> The footer prints the cost that is not ours to pay: *"Twelve Suno renders, at your cost and your
> hand. This bench has never spent."*

---

## SPLICE F9 — **§0.1 and §5.4, the autonomy boundary in the spec's own text**
**Anchor A:** Part 3 §0.1, the `Writes` and `Never` rows.

> | **Writes** | …unchanged… plus `data/soundry/experiments/<X-id>/**`, `output/soundry/<R-id>/ROUND_BRIEF.md`, `output/soundry/<R-id>/twins/**`, and `data/soundry/deck-ledger.json` **only on an explicit KEEP** |
> | **Never** | …unchanged… **and never writes a Title, a Style Prompt, a Lyric or a Caption.** The Anvil plans experiments and writes prompt *constraints*. The prose is written by `lofn-music`, in a session a human starts, from a brief this app composed and did not run. It invokes no skill, spawns no agent, and holds no route that could. **Twelve drafts on disk is the terminal state of every path through this app.** |

**Anchor B:** Part 3 §5.4, append to the refusal list:

> 15. **A generated Title, Style Prompt, Lyric or Caption.** Four wells render `unwritten` with their
>     constraints at full ink. An app that writes the song it is measuring has no instrument left.
> 16. **A ranking, score, or "most promising" mark on any of the twelve.** No per-card verdict at any
>     stage, including at draft (`vault/EXPERIMENT_STANDARD.md` §5.1). The deck does not score.
> 17. **A reshuffle in CONFIRM**, and in SCREEN, an uncounted one.
> 18. **A number the app computed itself for a gate that has a shipped measurer.** The strip is
>     `MEASURER ABSENT` before it is a guess.

---

## SPLICE F10 — **§3.1 (files in a round), §6 (required specs), §15 tree**

**§3.1**, append to the round listing and add the experiment listing:

```
data/soundry/experiments/<X-id>/preregistration.json   frozen, hashed, immutable; amendments append
data/soundry/rounds/<R-id>/design.json                 + deck_seed, twin table, spread matrix,
                                                         max|r|, oblique draw, skipped-oblique log
data/soundry/rounds/<R-id>/variants/<vid>.json         + twin_id, oblique_id, stratum, slot_sheet,
                                                         exclude_prompt, return_contract, deviations[],
                                                         soundcraft, gate_report
data/soundry/deck-ledger.json                          per-oblique + per-nuisance-level last_used_round
```

**§6 required specs**, append:

- **`deck.spec.ts`** — D1–D6 asserted on ≥50 seeds; a pool too small to satisfy a floor refuses and
  names the dimension; no code path writes `deck.data.json`; LRU draw is stable under reordering of
  the deck file; an oblique conflicting with the live factor is skipped and logged.
- **`twin.spec.ts`** — six twins, twelve cards, `A 6 / B 6`; every nuisance level count equal across
  levels; the factor × nuisance max |r| is **exactly 0** in CONFIRM; twins never adjacent in slot
  order; a draw splitting a twin across days is refused.
- **`exclude.spec.ts`** — 400–900 chars on every fixture; twin members differ only in category (b);
  `negatable: false` ⇒ (b) empty and the card says so; over-band drops a whole category and names it;
  never truncates mid-token.
- **`slots.spec.ts`** — budgets sum into `[870, 960]`; returned prose outside `[850, 1000]` fails;
  `≥ 985` FLAGs; missing terminal punctuation hard-fails; banned openers caught; **twin drift** at
  <0.90 on a non-factor slot fails both members; **null arm** at ≥0.90 on the factor slot fails both.
- **`handoff.spec.ts`** — `ROUND_BRIEF.md` contains the FROZEN FRAME sentence, the `spec_hash`, and no
  URL; generate before freeze → **409**; the app contains no skill invocation and no agent spawn.
- **`import.spec.ts`** — a mismatched `spec_hash` is refused; `## Major Deviations` parse; a `KEEP`
  offers but does not perform the deck write; a path outside `output/**` → 403.
- **`caption-refusals.spec.ts`** — §1.4E rules 1–5.
- **`gates-parity.spec.ts`** — every numeric threshold the Anvil displays is read from
  `vault/gates.yaml` at runtime; **no gate number is a literal in `apps/soundry/src/**`.**
- **extend `design.spec.ts`** — the twelve regenerate byte-identically from `deck_seed`, and
  `deck_seed` is the stated SHA-256 of `(experiment_id, date, spec_hash)`.
- **extend `refusals.spec.ts`** — refusals 15–18; the card object has no `chorus_justification`
  property at any state.
- **extend `server-soundry.spec.ts`** — `/measure` argv has no network binary; path confinement 403.

**§15 / Part 3 tree additions:**

```
apps/soundry/src/anvil/deck.ts        twin.ts     slots.ts    exclude.ts
apps/soundry/src/anvil/handoff.ts     import.ts   measure.ts  novelty.ts
apps/soundry/src/anvil/deck.data.json           factors.data.json
apps/soundry/tests/{deck,twin,exclude,slots,handoff,import,caption-refusals,gates-parity}.spec.ts
data/soundry/deck-ledger.json                   (git-ignored under R11's data/soundry/ line)
```

---

## SPLICE F11 — **`## Overrules ACCEPTED`, APPEND**

- **THE ANVIL WRITES THE EXPERIMENT; `lofn-music` WRITES THE SONG** (Forge Master). Board 3 draws
  twelve cards each carrying a full Suno quad, and the app emits three of the five fields as
  constraints rather than prose. **PRESERVES her ask literally** — twelve complete quads reach disk in
  Board-3 order, per card, paste-ready — and preserves the thing that makes them worth having: they
  are written by the pipeline that wins competitions, under a frame the bench froze, instead of by a
  slot grammar that would emit twelve paragraphs off one skeleton. It also PRESERVES §0.1 exactly:
  no generation API, no `dependencies` key, `connect-src 'self'` untouched.
- **THE UNIT OF THE GRID IS THE TWIN, NOT THE CARD** (Forge Master, strengthening §1.4). Independent
  stratified sampling is replaced by six matched pairs. PRESERVES §1.4's own stated intent — *"do the
  two arms differ in anything except the factor?"* — by making the answer structurally no rather than
  detectably-usually-no, and PRESERVES `vault/EXPERIMENT_STANDARD.md` §4.1's within-day randomisation
  by refusing any draw that splits a twin across the day block.
- **NOVELTY IS BOUGHT IN THE FRAME AND PAID FOR BY THE TWIN** (Forge Master). Board 2 underlines
  NOVELTY twice; a randomised experiment appears to forbid it. The resolution is that novelty lives
  entirely in twin-shared dimensions, so it cannot touch the contrast: LRU oblique draw, a mandatory
  `BELOW`-baseline twin, and a deck that grows from returned deviations she marks KEEP. PRESERVES
  *"Goal: NEW VIRAL EXPERIENCES ← NOVELTY"* without spending a single degree of freedom the
  experiment needs.
- **THE CAPTION NEVER NAMES THE EXPERIMENT** (Forge Master). PRESERVES the blind instrument, and
  PRESERVES R7 — a closed round's factor is exactly the disclosure the hosted build excludes.

## `# STILL OPEN`, AMEND

Strike *"THE SONIC ANVIL HAS NO GENERATOR"*. Replace with:

- **THE ANVIL'S GENERATOR IS SPECIFIED AND ITS SEAM IS A DEPENDENCY.** The deck, the twins, the
  spread, the slot sheets, the exclude builder, the return contract, the handoff and the measure
  route are specified (§1.3A, §1.4, §1.4A–G). Two things remain open and they belong to her:
  **(i)** the seed pools in `sampler.data.json` and the opening deck in `deck.data.json` are hers to
  write — D1's Hamming floor is unsatisfiable until the persona and theme pools each hold ≥6 entries,
  and the app refuses rather than repeats; **(ii)** the round's power constants inherit the unresolved
  4.49% / 4.3695% baseline conflict named elsewhere in this list — the deck does not touch it, prints
  no baseline of its own, and reads the stratum from the style board at compile time precisely so
  this patch adds no new copy of a contested number.

## Build order — new lane

- **L-ANVIL · THE DECK AND THE HANDOFF** (needs L-KIT for the empty/ghost contracts and Soundry
  U3+U5; parallel with every Analytics lane; blocks nothing). `apps/soundry/src/anvil/**` ·
  `deck.data.json` + `factors.data.json` (seeded from her pools; **refuse, do not repeat**, on a thin
  pool) · the three server routes · the ten specs above. **Build `twin.ts` and `exclude.ts` first**:
  they are the two pieces that make twelve cards an experiment, and `exclude.ts` is the only field
  the app finishes on its own, so it is the cheapest proof the seam is real.


---

# APPENDIX — seat patch: THE TASK TAXONOMIST (after Ben Shneiderman — the Visual Information-Seeking Mantra, dynamic queries, tightly-coupled controls, "an interface must never present a dead end") (Read from disk, round two: BUILD_SPEC.md §2.1 (PlotSpec, EmptyState, `deeper?`), §2.3 (V1–V18, esp. V13/V15), §3.1–§3.4 (the kit filter state), §4.1–§4.4 (Insight, Tab Directive), §5.1 (guidance), §6.1 (empty-vs-zero); Part 2 §2 (the rail, `?m=`), §3.3 (the Analytics shelf, m|win|pf|out|est|prov), §5 M1.6 (the catalogue = the L2 rows surface), §14.1–§14.4 (the four-level ladder, `drill`/`reason`, `drill-registry.spec.ts`), §15 (skeleton, test contract 1–9); Part 3 §0.1–§0.2, §1.3 A2.1, §1.4 A3.1/A3.2, §1.5 (preregistration.json), §2.2–§2.9 (B2–B9.3), §3.1–§3.4, §4.1–§4.7 (C1–C7), §5.4 (the fourteen refusals), §6 (Soundry test contract). SPEC_PATCH.md: rulings R1–R12, SPLICE 1 (EmptyKind/UnlockCounter/EmptyState/JoinEvidence), SPLICE 2 (four new empty kinds), SPLICE 3 (the ghost contract, rule 6), SPLICE 8 (M6 CONFLUENCE + the per-card filter applicability paragraph), SPLICE 10 (the watchlist, my own round-one patch), SPLICE 12 (tests 10–26, R10), the STILL OPEN list (items #4 and #5 are my assignment), and build order U0b, which already orders this reconciliation. WHITEBOARD_TRANSCRIPTION.md Boards 3, 6, 7 and THE OVERRULE MANDATE. vault/AUTONOMY.md Part 1; vault/EXPERIMENT_STANDARD.md §§1–5; vault/SOCIAL_VOICE.md (Appendix C standing rule, as constrained by R3).)

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/SPEC_PATCH.md, E:/git/private-claw/apps/lofn-kit/src/state/shelf.ts, E:/git/private-claw/apps/lofn-kit/src/state/cohort.ts, E:/git/private-claw/apps/lofn-kit/src/state/applicability.ts, E:/git/private-claw/apps/lofn-kit/src/contracts/deeper.ts, E:/git/private-claw/apps/lofn-kit/src/plot/spec.ts, E:/git/private-claw/apps/lofn-kit/src/plot/validate.ts, E:/git/private-claw/apps/lofn-kit/tests/shelf-vocabulary.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/shelf-url.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/shelf-applicability.spec.ts, E:/git/private-claw/apps/_fixtures/load-manifest.json, E:/git/private-claw/apps/lofn-analytics/src/shell/filters.ts, E:/git/private-claw/apps/lofn-analytics/scripts/card-registry-check.mjs, E:/git/private-claw/apps/lofn-analytics/tests/deeper-registry.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/shelf-zero-result.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/expiring-refusal.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/forbidden-strings.spec.ts, E:/git/private-claw/apps/soundry/src/shell/filters.ts, E:/git/private-claw/apps/soundry/src/cards/b10-variant-table/index.ts, E:/git/private-claw/apps/soundry/tests/soundry-shelf.spec.ts, E:/git/private-claw/apps/soundry/tests/shelf-prereg-interlock.spec.ts, E:/git/private-claw/apps/soundry/tests/deeper-registry.spec.ts, E:/git/private-claw/.claude/skills/lofn/SKILL.md, E:/git/private-claw/.agents/skills/lofn/SKILL.md

# PATCH — THE TASK TAXONOMIST, ROUND TWO · STILL-OPEN #4 and #5

Target file: `E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`
Executes build-order item **U0b**, which the round-one moderator declared serial and blocking on
L-LEDE and L-FUSE-A. Nothing here reopens R1–R12. Where it amends a round-one splice, it says so
by name in §T0.9 and argues it.

---

## T0 · RULINGS — each names the intent it preserves

**T1 · ONE VOCABULARY, AND IT IS THE KIT'S.** Part 1 §3.2's long key names win over Part 2 §3.3's
short ones (`mod` not `m`, `co` not `win`, `floor` not `pf`). The kit is where the serializer lives
and where both apps import it; a shelf is a *render* of kit state, and a render may not rename the
model. **PRESERVES Part 2's contribution in full:** its `est` (estimator) key is real, load-bearing
and absent from the kit union — on the identical n=46 cohort the same data reads 4.37% median,
4.29% mean, 1.57% pooled — so `est` is **adopted into the union**, not dropped. Part 2's 44px
sticky shelf, its live `n = 46 of 210` readout and its provenance-suppression count are the display
contract and they are kept verbatim. Part 2's `2026-only` cohort value is a synonym for `modern`
and is **struck as a synonym, named here, not silently dropped**. Part 2's `?m=art` rail parameter
(§2) is the same key as `mod` and is rewritten to `?mod=art`.

**T2 · THE COHORT DEFAULT IS `modern`, AND THE CHIP THAT SAYS SO IS NOT A GHOST.** Part 1 defaults
`co=modern`, Part 2 defaults `win=all`. `modern` wins, because every printed constant in both apps
— 4.49%/4.37% baseline, ±2.53pp, σ²_true, mde80, the whole power model, `vault/EXPERIMENT_STANDARD.md`
§1 — is computed on the modern cohort, and a first paint whose default disagrees with every number
in the spec is a lie told by a default. But a default that removes 164 of 210 rows is exactly the
state a user cannot see, so: **THE SUBTRACTIVE-DEFAULT RULE.** A default that *removes rows*
(`co`, `out`, `floor`, `nmin`, `prov`) renders at **full ink with its denominator from first paint**
(`cohort = modern · 46 of 210`) and carries a one-click `[all 210]`. A default that only *chooses a
view* (`mod`, `basis`, `est`, `snap`, `blind`) renders as the single ghost chip Part 1 §3.4
describes. **PRESERVES §3.4's intent** — its ghost chip exists so the shelf is quiet when nothing
is set; it was never meant to hide a 78% row removal.

**T3 · "WHEN APPLICABLE" IS COMPUTED FROM THE CARD, NOT DECLARED BY ITS AUTHOR.** A key's
applicability to a card is a pure function of which fields that card's rows actually carry
(§T1.4). An author who omits a filter their rows *can* honour **fails the build** (invariant F1).
Board 7's "carry between tabs when applicable" becomes a compiler rule instead of an intention.
**PRESERVES the overrule mandate:** a silent drop is now a red test, not a judgement call.

**T4 · FOUR VISIBLE NON-APPLICABLE STATES, NOT ONE.** `dimmed` (the field is not on this card),
`inert` (it applies and removes nothing here), `locked` (it applies and is *forbidden* here —
re-cutting a randomised round), `parked` (the other app's key, carried through untouched).
**PRESERVES the Synthesist's X6 sentence exactly**, promoted from a dim to a lock, because
"a filter that could re-cut a randomised result would silently un-randomise it" is not an
inapplicability, it is a guarantee, and a padlock says so where a grey chip does not.

**T5 · THE SOUNDRY GETS THE SAME SHELF, MOSTLY LOCKED, AND THE LOCK IS THE FEATURE.** Board 7
applies to both apps. The Soundry shelf ships all fifteen keys; on any card reading a frozen
round's analysis, seven are locked with the reason printed. Where they *are* honoured — the A2/C3
power model — the carried cohort does real work: the power card is computed on the cohort she set
in Analytics, and **the resolved filter state is frozen into `preregistration.baseline_cohort` and
covered by `spec_hash`**, so a later re-cut is a detectable deviation rather than a quiet one.
**PRESERVES §0.1's AUTONOMY posture and §5.4's fourteen refusals** — no shelf control can produce a
per-song verdict, a ranking, or a re-cut estimate.

**T6 · THE SERIALIZER NEVER DROPS A KEY IT DOES NOT USE.** Cross-app links carry the whole global
state; the target app parses, renders unknown-to-it keys as `parked`, and re-emits them byte-
identically. There is no stash, no sessionStorage, no origin problem. **PRESERVES §3.4's cross-app
promise** ("a headwind link opens a Soundry seed with its filter intact") and makes the round trip
a one-line test.

**T7 · `deeper` IS THE ONLY DRILL WORD, IT IS REQUIRED, AND ITS ABSENCE IS A VALUE.** Part 1 §2.1's
`deeper?: {...} | null` + `deeperAbsentReason?` and Part 2 §14.2's `drill: {to,scope} | null` +
`reason` are one concept spelled twice; §14.2's word is struck and its four-level ladder is kept as
`deeper.level`. The field becomes **required and non-nullable**, and "there is nothing deeper yet"
becomes `{ kind: "none", reason, watch, unlockAt }`. **An optional field is how an omission ships**
— that is the entire mechanism by which twelve Soundry plots slipped past V15. **PRESERVES §14.2's
overrule of her "if they exist"**: every insight still links or states in words why it cannot; the
conditional is now carried by a variant of the type rather than by a missing key.

**T8 · THE EXPIRING REFUSAL.** Every `kind:"none"` names a `watch` — a member of a frozen enum of
conditions the loader already evaluates. When the watched condition becomes true and the refusal is
still there, **the build fails**. A refusal may not outlive its reason. **PRESERVES her "if they
exist"** literally: the day the deeper result exists, the app is forbidden to keep saying it does not.

### T0.9 · What this contradicts, said out loud

1. **SPEC_PATCH SPLICE 8 adopts the Synthesist's per-card filter paragraph "verbatim".** I amend it:
   `X1 honours out, pf, win` → `out, floor, co`; `X2 honours win, est, out` → `co, est, out`;
   `X4 honours win, nmin` → `co, nmin`; X6's six chips are **locked, not dimmed**, with the
   Synthesist's sentence as the lock reason, word for word. Verbatim adoption of a retired
   vocabulary would ship the third vocabulary the critics already flagged — and U0b, written by the
   same moderator, orders this union. This is execution of a ruling, not a reversal of one.
2. **Part 1 §3.4's ghost-chip default** is narrowed by T2's subtractive-default rule.
3. **Part 1 §3.1's local key `exp`** (expanded row, in `#craft:sort=effect,dir=desc,exp=piano`)
   **collides with the Soundry's global `exp` (round id).** Renamed to `open`. A local key and a
   global key with one name is a bug that would have shipped.
4. **Part 1 §2.1 `deeper` and Part 2 §14.2 `drill`** are replaced by one required field. This
   touches the same §2.1 block SPLICE 1 rewrites; the coordinator merges T4 into SPLICE 1 in one
   edit, not two.
5. **Test 6 `drill-registry.spec.ts`** is renamed in place to `deeper-registry.spec.ts`. R10's
   numbered block 10–26 is untouched; this patch numbers 27–34.

---

# SPLICE T1 — [REPLACE] PART 1 §3 in full (`apps/lofn-kit/src/state/`)

# §3 · THE SHELF — one filter vocabulary, both apps

> Board 7: *"All segmentations & filters carry between tabs when applicable, with clear controls."*
> One model, one serializer, one component, two apps. **Non-application is rendered, never
> performed silently.** The vocabulary below is frozen; adding a key is a coordinator-only change
> to `shelf.ts` plus a `fv` bump.

### 3.1 Three scopes, and the exact URL shape

| scope | lives in | survives | shared |
|---|---|---|---|
| **GLOBAL** — the analysis | the **query string**, fixed key order | tab change, drill, reload, app change | yes |
| **LOCAL** — the furniture | the **hash**, after the route, namespaced by `:` | return within the session | only via *Copy link with view state* |
| **NOT-APPLICABLE** | nothing | — | — |

`NOT-APPLICABLE` is **computed per (key, card), never stored**. There is no third bucket to fall
out of, which is how a key gets lost.

**Canonical URL, exactly:**

```
/?fv=1&mod=music&co=all&floor=0&sel=phonk#/music/craft:sort=effect,dir=desc,open=piano
└── global, frozen order, defaults omitted ──┘└─ route ─┘└──── local, per-tab ────┘
```

- `fv=1` — the **frozen-vocabulary version**, emitted on every link even when nothing else is set.
  A link arriving with a different or missing `fv` opens normally and pins one banner: *"This link
  was written against filter vocabulary v{n}. It asked for {…}; today that resolves to {…}."*
  No silent reinterpretation, no dead link.
- Order = `FILTER_ORDER` (the declaration order in 3.2). Two URLs for two analyses are diffable.
- Defaults are **omitted** from the URL and **always drawn** in the shelf (T2). The URL is short;
  the screen is complete.
- Multi-value = comma-joined, values sorted lexicographically (OR within a key, AND across keys).
  `,` inside a value is `%2C`; every value is `encodeURIComponent`'d; `snap`'s ISO offset colons
  are percent-encoded.
- **On load the URL is rewritten canonical with `replaceState`.** The address bar always shows the
  true state. A state reachable only by not looking is the failure this rule exists against.
- `LOCAL_KEYS = ["sort","dir","open","bins","view","page","row"]`, frozen, and
  `LOCAL_KEYS ∩ FILTER_KEYS = ∅` is asserted by test 27. (`exp` renamed to `open` — see T0.9.3.)

### 3.2 The frozen global key table — one vocabulary, both apps

| # | key | control | values | default | apps | subtractive |
|---|---|---|---|---|---|---|
| 1 | `mod` | Modality (the left rail *is* this control) | `music`·`art`·`anim` | `music` | both (Soundry: parked) | no |
| 2 | `snap` | Snapshot | `latest` \| explicit `pulled_at` | `latest` | both | no |
| 3 | `co` | Cohort window on `created_at` | `all` · `modern` · `YYYY-MM..YYYY-MM` | **`modern`** | both | **yes** |
| 4 | `basis` | Measure | `response` (like rate) · `reach` (plays) | `response` | both | no |
| 5 | `est` | Estimator | `median` · `mean` · `pooled` | `median` | both | no |
| 6 | `out` | Outlier policy | `exclude` · `label` · `isolate` | `exclude` | both | **yes** |
| 7 | `floor` | Play-maturity floor | integer 0–1000 | `60` | both | **yes** |
| 8 | `nmin` | Ranking n-floor | integer 1–50 | `10` | both | **yes** |
| 9 | `prov` | Provenance floor | `api` · `+auth` · `+public` · `+snippet` | `+public` | both | **yes** |
| 10 | `sel` | Cross-highlight | one `field:value` pair | none | both | no |
| 11 | `q` | Free-text search | string | empty | both | **yes** |
| 12 | `exp` | Round id | `R-YYYY-MM-DD-<slug>` | none | soundry | no |
| 13 | `hyp` | Hypothesis id | id | none | soundry | no |
| 14 | `arm` | Arm | `A`·`B`·`all` | `all` | soundry | no |
| 15 | `blind` | Blind presentation | `on`·`off` | `on` on Results, else `off` | soundry | no |

**Struck synonyms, named not dropped:** `m`→`mod`, `win`→`co`, `pf`→`floor`, `2026-only`→
`co=2026-01..2026-12`, `include+label`→`label`, `api-public`→`api`, `+authenticated-page`→`+auth`,
`+public-page`→`+public`, `+search-snippet`→`+snippet`.

`MODERN_FROM = "2026-02"` is exported once from `apps/lofn-kit/src/state/cohort.ts`. `modern`
is not a hardcoded date anywhere else.

**`floor` and `nmin` are different instruments and both ship:** `floor` decides whether a *song's*
rate is readable at all; `nmin` decides whether a *cell* may be ranked.

**The provenance ladder, because `Provenance` has seven values and the floor has four (R12).**
`PROV_RANK = { "api-public":0, "authenticated":1, "public-page":2, "search-snippet":3 }`. A row is
suppressed when its rank exceeds the floor. **`derived` inherits the maximum rank of its inputs**
(computed in the loader, stored as `provRankEffective`). **`prose-archive` is always suppressed
above `+public`** and carries V17's drift pair regardless. **`randomised-round` is never suppressed
by this floor** — a pre-registered randomised estimate is the strongest provenance in the estate
and a floor built for scraped counters must not be able to hide it. Asserted by test 28.

### 3.3 API — `apps/lofn-kit/src/state/shelf.ts`

```ts
export const FILTER_ORDER = ["mod","snap","co","basis","est","out","floor","nmin",
                             "prov","sel","q","exp","hyp","arm","blind"] as const;
export type FilterKey  = typeof FILTER_ORDER[number];
export type AppId      = "analytics" | "soundry";
export type FilterState = Partial<Record<FilterKey, string[]>>;   // OR within, AND across

export interface ShelfKeyDef {
  key: FilterKey;
  label: string;                       // "Cohort window"
  apps: AppId[];
  scope: "global";
  subtractive: boolean;                // T2: full-ink default with denominator
  fields: string[];                    // FIELD_OF — the row fields this key reads. [] = universal
  requires?: "nField" | "measure" | "detail" | "rowProvenance";
  parse(v: string[]): unknown;         // throws => the fv banner, never a crash
  defaultValue: string[];
  guidance: GuidanceNote;              // REQUIRED. §5.1. A chip with no note does not compile.
}
export const SHELF: Record<FilterKey, ShelfKeyDef>;   // the single source of truth

export function fromUrl(url: string): { state: FilterState; local: LocalState;
                                        route: string; unknown: Record<string,string[]>; fv: number };
export function toUrl(state: FilterState, local: LocalState, route: string,
                      opts?: { withViewState?: boolean }): string;   // fixed order, defaults omitted
export function crossAppHref(target: AppId, route: string, state: FilterState): string;  // T6
export function applyFilters(rows: Row[], f: FilterState, honoured: FilterKey[]):
  { rows: Row[]; dropped: number; byKey: Record<FilterKey, number> };
export function relaxations(rows: Row[], f: FilterState, honoured: FilterKey[]):
  Array<{ key: FilterKey; label: string; resultingN: number }>;      // precomputed, top 3
export function trail(): Array<{ label: string; state: FilterState }>;  // last 10, one-click restore
export function extract(rows: Row[], f: FilterState): Blob;         // CSV + JSON sidecar, local only
```

`unknown` is **kept and re-emitted** (T6). The serializer is not allowed to be lossy.

### 3.4 Applicability — the rule a renderer evaluates

```ts
export interface CardFacts {                 // DERIVED, never hand-written
  id: string;
  rowFields: string[];                       // union of fields present on the card's rows
  hasNField: boolean;                        // any plot sets nField
  measure: null | { field: string; kind: "rate" | "count" };
  hasDetail: boolean;                        // a text identity q can search
  rowProvenance: boolean;                    // rows carry per-row provenance
  filters: FilterKey[];                      // author-declared, checked against the above
  filtersLocked: FilterKey[];                // author-declared, each needs a lockReason
  lockReasons: Partial<Record<FilterKey, string>>;
}
export function factsOf(card: CardSpec, plots: PlotSpec[]): CardFacts;   // pure

export type ChipState =
  | { state: "active";  n: number; of: number }
  | { state: "default"; n: number; of: number; subtractive: boolean }
  | { state: "inert";   why: string }                        // applies here, removes nothing
  | { state: "dimmed";  why: string; returnsOn: string[] }   // not on this card's rows
  | { state: "locked";  why: string }                        // applies, and is forbidden here
  | { state: "parked";  why: string };                       // the other app's key, carried

export function chipState(key: FilterKey, f: FilterState,
                          facts: CardFacts, app: AppId): ChipState;
```

**Precedence, evaluated top to bottom — this is the whole rule:**

1. `SHELF[key].apps` excludes `app` → **parked**. *"`arm` belongs to The Soundry. It is carried and
   will be exactly as you left it when you cross back."*
2. `key ∈ facts.filtersLocked` → **locked**, rendering `lockReasons[key]` and a padlock glyph.
3. `SHELF[key].fields = []` and no `requires` → **universal** (`mod`, `snap`), go to 6.
4. `SHELF[key].requires` unmet, or `SHELF[key].fields ∩ facts.rowFields = ∅` → **dimmed**, with a
   generated `why` (*"This card's rows carry no `created_at`, so a cohort window has nothing to cut."*)
   and `returnsOn` = the tabs where it *is* honoured, computed from the registry
   (*"It applies again on Craft and Confluence."*).
5. `applyFilters` drops 0 rows for this key on this card → **inert**. *"Cohort = modern is applied
   here and removes nothing: every row on this card is already inside it."* An inert chip is not
   dimmed — it was honoured, and saying so is the difference between "ignored" and "no effect".
6. else **active** (non-default) or **default** (with `subtractive` from `SHELF[key]`).

**INVARIANT F1 — the anti-silent-drop rule, and it fails the build.** For every card in either app:

```
∀ k ∈ FILTER_KEYS where SHELF[k].applies(facts):
      (k ∈ facts.filters)  XOR  (k ∈ facts.filtersLocked)
∀ k ∈ facts.filters:  SHELF[k].applies(facts) === true
∀ k ∈ facts.filtersLocked:  lockReasons[k] is a sentence of ≥ 10 words
```

A card whose rows carry `play_count` and which omits `floor` fails, naming the card and the key. A
card that claims to honour a filter its rows cannot support fails the other way. Test 29.

### 3.5 The shelf, drawn

- **44px sticky, one row, directly under the tab bar; wraps to 88px rather than scrolling
  horizontally.** A control you have to scroll to find is a control that is not there.
- Chip = `field · value · resulting n`, e.g. `cohort = modern · 46 of 210`. The whole chip is a
  button opening its own popover control (segmented / slider / text). **Dynamic query:** slider
  drags repaint live under 100 ms with **no spinner, ever**; `history.pushState` debounced 400 ms so
  a drag does not manufacture sixty entries; `popstate` re-reads. A control that cannot meet the
  budget becomes an explicit `Apply` button and says why.
- Non-default chips carry a 2px `--chrome-2` left edge. Subtractive defaults render at full ink with
  their denominator and a one-click `[all 210]`. Everything else defaults to one ghost chip:
  `Music · Response · median · outliers excluded`.
- **A plot honouring none of the active filters renders a `not filtered` chip on the plot itself.**
- **Per-chip `×` returns that key to its default, never to empty.** Emptiness is a state with no
  data in it and must not be reachable by accident. `Reset to defaults` is disabled when already
  there and says so.
- Right edge, always: `n = 46 of 210` and `12 rows below the provenance floor`.
- **Zero-result contract:** a state yielding 0 rows never blanks a panel. `empty.kind =
  "filtered-empty"`, the ghost draws, and `relaxations()` offers up to three one-click removals with
  their resulting n **already computed**. If no relaxation yields n > 0, the panel renders
  `Reset to defaults` as its primary action. **Every filter state has a next click.** Test 31.
- **History (task 6):** the session trail — last 10 global states, one-click restore, labelled by
  what changed (`cohort: modern → all`). **Extract (task 7):** `Copy link` · `Copy link with view
  state` · `Download these rows (CSV + provenance sidecar)`, a local Blob, no network, the sidecar
  carrying `{snapshot_id, filter_state, generated_at, n, caveats[]}` (M1.6's rule, generalised).
- **Every chip carries a `GuidanceNote` by type** (`ShelfKeyDef.guidance`). This closes the shelf's
  share of STILL-OPEN #2 by construction. **It does not close #2** — there is still no
  `GuidanceNote` registry and no `guidance-coverage.spec.ts`, and I am not going to imply otherwise.

---

# SPLICE T2 — [REPLACE] PART 2 §3.3 "Filter shelf"

### 3.3 The shelf (kit §3, rendered)

The Analytics shelf is `apps/lofn-kit/src/state/shelf.ts` rendered by
`apps/lofn-analytics/src/shell/filters.ts`. **It defines no keys of its own.** The six controls
this section previously named are the same six keys under their kit names, plus the nine the kit
already had:

| shown as | key | default |
|---|---|---|
| Modality *(the left rail is this control — §2)* | `mod` | `music` |
| Cohort window | `co` | `modern` · full ink · `46 of 210` |
| Play floor | `floor` | `60` · full ink |
| Outlier | `out` | `exclude` · full ink |
| Estimator | `est` | `median` |
| Provenance floor | `prov` | `+public` · full ink |
| Snapshot · Measure · Ranking n-floor · Cross-highlight · Search | `snap` `basis` `nmin` `sel` `q` | kit defaults |
| Round · Hypothesis · Arm · Blind | `exp` `hyp` `arm` `blind` | **parked** (Soundry keys, carried) |

- **§2's rail parameter is `?mod=`, not `?m=`.** One key, one name.
- **Estimator prints inline on every card that honours it** — on the identical n=46 cohort the same
  data yields median 4.37%, mean 4.29%, pooled 1.57%, and a figure without its estimator is three
  different claims wearing one number. (See open question 2.)
- Everything else — dimmed/inert/locked/parked rendering, the zero-result contract, the trail, the
  extract, the denominators — is kit §3.4/§3.5 and is not restated here.

---

# SPLICE T3 — [NEW] PART 3 §0.4 · THE SOUNDRY SHELF

### 0.4 The shelf (kit §3, rendered — Board 7 applies to this app too)

Part 3 previously rendered no chip strip. It renders one now: same 44px component, same serializer,
same fifteen keys, **different applicability table.** Three tabs: `#/anvil`, `#/results`,
`#/analytics`.

**Honoured (they change what is computed):**

| key | where | what it does |
|---|---|---|
| `exp` `hyp` `arm` | Results, Analytics | selects the round / hypothesis / arm under view |
| `blind` | Results, `#/panel/*` | hides titles, prompts, arm labels. **`blind=on` is forced and locked inside a panel session** — the listener cannot turn it off, and the chip says so |
| `q` | Results, Ledger | searches hypothesis text and variant titles |
| `snap` | everywhere | which suno snapshot the outcome instrument reads |
| `co` `out` `floor` `est` | **A2 power card and C3 power planner only** | they re-derive `p̄` and `σ²_true` from the chosen cohort. This is the carry doing real work: set the cohort in Analytics, cross to the Anvil, and the power curve is computed on that cohort. **The card prints the cohort expression that produced its constants, always.** |
| `mod` | — | **parked.** *"The Soundry tests songs. Modality is carried and untouched."* |

**Locked on every card that reads a frozen round's `analysis/` or `observations/`** — B2, B4, B5,
B7, B9.1, C1, C2, C4, C5 — for `co`, `out`, `floor`, `nmin`, `est`, `basis`, `prov`. Lock reason,
verbatim, the Synthesist's sentence promoted from a dim to a lock:

> **"This round's cohort was frozen at pre-registration. A filter that could re-cut a randomised
> result would silently un-randomise it. The chip is here, and holding your value, so that you can
> see it is not being applied."**

A locked chip **keeps the value you set in Analytics** and restores it the moment you cross back.
That is Board 7's "carry between tabs when applicable, with clear controls" — the carry is real,
the applicability is visible, and the control is not lying.

**THE PREREGISTRATION INTERLOCK.** `preregistration.json` (§1.5) gains one field:

```
baseline_cohort   the canonical toUrl() serialisation of the GLOBAL filter state at freeze,
                  e.g. "fv=1&co=modern&floor=60&out=exclude&est=median"
```

It is inside `spec_hash`. If the analysis that runs resolves a different cohort than the one frozen,
the existing `ANALYSIS DEVIATION` machinery (§1.5) fires — *the analysis that ran is not the
analysis that was registered* — and the round leaves the multiplicity denominator with a visible
note. The shelf can no longer quietly change what a frozen round means. Test 30.

**Cross-app.** `crossAppHref("soundry", "#/anvil", state)` from an Analytics headwind card carries
all fifteen keys; the Anvil parks what it cannot use and re-emits it byte-identically on the way
back (T6). Test 32.

---

# SPLICE T4 — [MERGE INTO SPLICE 1] PART 1 §2.1, REPLACE lines 293–294; NEW `contracts/deeper.ts`

```ts
// apps/lofn-kit/src/contracts/deeper.ts   — the ONLY drill vocabulary in the estate
export type DeeperLevel = 1 | 2 | 3;    // §14.1: 1 evidence plot · 2 rows · 3 work record

export type Deeper =
  | { kind: "route";   href: string; label: string; level: DeeperLevel }
  | { kind: "path";    path: string; label: string; level: 3 }        // a file, shown copyable
  | { kind: "runtime"; configKey: "soundryOrigin";                    // built at load, never literal
      fallbackPath: string; label: string; level: 3 }
  | { kind: "none";    reason: string;            // ≥8 words, NAMES the artefact that doesn't exist
                       watch: WatchId;            // the condition that would make it exist
                       unlockAt?: UnlockCounter[] };   // SPLICE 1's type. One definition, not two.

export const WATCHABLE = [
  "feed.suno","feed.socials","feed.distribution","feed.competitors","feed.market","feed.soundry",
  "snapshots.suno>=2","snapshots.suno>=3","snapshots.suno>=14",
  "soundry.round.frozen","soundry.round.closed","soundry.rounds.closed>=2",
  "soundry.round.panel","watchlist.roster>=1","watchlist.pulled>=1",
  "repo.input-cleaning","repo.scan-market"
] as const;
export type WatchId = typeof WATCHABLE[number];
export function watchSatisfied(w: WatchId, manifest: LoadManifest): boolean;   // pure
```

`PlotSpec` and `CardSpec` both change:

```ts
  deeper: Deeper;        // REQUIRED, NON-NULLABLE.  `deeperAbsentReason` is DELETED.
```

**Rendering, normative.** The `deeper` slot is the bottom-right of every card and plot and **is
never empty**. `route`/`path`/`runtime` render the affordance; `none` renders the same slot at
reduced ink with the reason inline, plus the `unlockAt` counters, plus — when a counter's `blocker`
is `"command"` — the copyable command; when it is `"decision"`, **no command and the words that
none exists** (SPLICE 3 rule 6). An empty slot is a dead end and a dead end is the one thing this
interface may not have.

**`kind:"route"` may hardcode only `sel`, `exp`, `hyp`, `arm`.** Every other global key is
**inherited from the live shelf at click time**, and the label says so (*"…with these filters"*).
A drill that quietly changes the analysis is the same defect as a filter that quietly drops.

**Ghost cards (SPLICE 3):** a ghost's `deeper` is `kind:"none"` and its `unlockAt` is the *same
counters* as the ghost's. One blocker, stated once, in two places that cannot disagree.

**`kind:"path"` is copyable text with a copy button, never an `<a href="file://">`** —
`check-local-only.mjs` and the CSP both forbid the link, and a link that cannot open is worse than
the path you can paste.

---

# SPLICE T5 — [REPLACE] PART 1 §2.3 rule V15; APPEND V24–V27

| # | rule | why |
|---|---|---|
| **V15** *(rewritten)* | `deeper` present and well-formed for its `kind`. The type makes omission a compile error; `validate()` re-checks because specs are also built at runtime from fixtures. | **An optional field is how an omission ships.** Twelve Soundry plots proved it. |
| **V24** | `deeper.kind === "none"` ⇒ `reason` ≥ 8 words **and** matches `/data\/|output\/|vault\/|apps\/|scripts\/|[a-z_.-]+\.(json|jsonl|md|py|mjs|ts)/` or names a member of `WATCHABLE` in prose; `watch ∈ WATCHABLE`. | "No deeper data" is not a reason. The name of the file that does not exist is a reason. |
| **V25** | `deeper.kind === "route"` ⇒ the href's route ∈ the app's frozen `ROUTES`; its query keys ⊆ `FILTER_KEYS`; its values legal under `SHELF`; it hardcodes only `sel`/`exp`/`hyp`/`arm`; it is not this card's own route. | This is where a stale filter vocabulary would ship, and where a self-link would masquerade as a drill. |
| **V26** | a plot with both an `empty` and `deeper.kind === "none"` ⇒ `deeper.watch` and the empty state's blocking artefact are **the same thing**. | A card may not give one reason for being empty and a different reason for having no deeper. Two reasons for one hole is one reason too many. |
| **V27** | `deeper.label` ≥ 2 words and ∉ `{"more","details","more details","click here","see more"}`. | The control names its destination. |

---

# SPLICE T6 — [REPLACE] PART 2 §14.1–§14.2

**14.1 Four levels, one word.** L0 insight sentence → **L1** evidence plot (expands in place) →
**L2** rows (M1.6 catalogue, filtered) → **L3** work record. The level is `deeper.level`.

**14.2 Every card and every plot declares `deeper`, and it is required.** The former `drill: {to,
scope} | null` + `reason` vocabulary is **struck and replaced by kit `Deeper`** (§2.1) — one word,
one type, both apps. Her sketch says "link to deeper results *if they exist*"; that conditional is
exactly how dead links ship, and it stays overruled to **every insight links, or states in words
why it cannot** — but the refusal is now a *value of the type*, not a missing key, so it cannot be
forgotten. `tests/drill-registry.spec.ts` is renamed `tests/deeper-registry.spec.ts` and extended
(test 27). **A card or plot without `deeper` does not compile; one built at runtime without it
fails the build.**

§14.3's reverse drill is unchanged and is now typed: every L3 record links back up with the
isolating filter applied, built through `toUrl()`, never a string literal.

---

# SPLICE T7 — [NEW] PART 3 §7 · THE SOUNDRY `deeper` TABLE, and the missing L2 rung

### 7.1 B10 · THE VARIANT TABLE — `#/results/<round>/variants` (new, and it unblocks nine cards)

Nine Soundry plots have no legal L2 target because the Soundry has no rows surface. Without one,
nine cards would have to refuse, and a refusal that exists because we did not build the table is a
fake refusal. B10 is Analytics' M1.6 for a round: virtualised, 12 rows, columns *variant · arm ·
slot · factor levels · nuisance levels · state · clip_id (bound/UNBOUND) · posted_at · age · plays ·
like rate with its Wilson bar on the shared 0–15% axis · `not_posted_reason`*. The cohort expression
prints above it with a copy button; export writes CSV + the provenance sidecar. **No sort by
outcome is the default** (§5.4.7 — no ranking of the twelve); sort defaults to `slot`, and a
`sort by rate` control exists, labelled, with the ±2.53pp line printed beside it.

### 7.2 The table — every plot in Part 3, with its `deeper`

| card | `deeper` | when it is `none` |
|---|---|---|
| **A2.1** power curve | `route #/analytics/power?exp=<id>` · L2 · *"the power planner at this round's n"* | never — C3 always renders |
| **A3.1** design matrix | `path data/soundry/rounds/<id>/design.json` · L3 · *"the assignment record and its RNG seed"* | **before freeze:** `none`, reason *"The assignment is written at freeze; `data/soundry/rounds/<id>/design.json` does not exist yet."* watch `soundry.round.frozen` |
| **A3.2** nuisance balance | `route #/anvil/design:open=balance` · L2 · *"the full balance table, five dimensions"* | as A3.1 |
| **B2** maturity board | `route #/results/<id>/variants` · L2 · *"every variant's age, plays and binding"* | never once a round exists |
| **B3** accumulation curves | `route #/results/<id>/variants:open=accrual` · L2 | **today:** `none`, reason *"Nothing on disk describes how fast a song accrues plays: `data/snapshots/suno/` holds one snapshot and a curve needs fourteen consecutive daily pulls."* watch `snapshots.suno>=14`, `unlockAt: [{label:"daily snapshots", have:1, need:14, how:"python3 scripts/ingest_suno.py", path:"data/snapshots/suno/", blocker:"command"}]` |
| **B4** arm contrast | `route #/results/<id>/analysis` · L3 · *"the GLM output, exclusions and day blocks"* | **no closed round:** `none`, watch `soundry.round.closed` |
| **B5** per-variant cloud | `route #/results/<id>/variants` · L2 · *"every variant's rate and its n"* | never once a round exists |
| **B7** listener feedback | `route #/results/<id>/feedback` · L3 · *"every note, with its cleaning flags"* | **today:** `none`, reason *"`cleanFeedbackText()` (`input-cleaning.mjs`) does not exist in this repository, and no listener note may be displayed or stored until it does."* watch `repo.input-cleaning`. **Second `none` case:** round has no panel instrument → reason names `observations/panel.jsonl`, watch `soundry.round.panel` |
| **B9.1** prereg timeline | `route #/results/ledger:open=deviations` · L3 · *"the deviation register, hash before and after"* | never — the ledger renders empty honestly |
| **C1** DECIBAL | `route #/results/<id>/analysis` · L3 · *"the full GLM: β, SE model/quasi/HC1, φ̂, deviance, iterations"* | **no closed round:** `none`, reason *"No round has closed. `data/soundry/rounds/` holds no round with a frozen pre-registration and a readout."* watch `soundry.round.closed` — **and it is the same artefact as C1's empty state (V26)** |
| **C2** hypothesis board | `route #/results/ledger` · L2 · *"every pre-registered contrast, including the abandoned"* — the per-row `[more details →]` she drew is kept, unchanged, as the row-level L3 | never |
| **C3** power planner | `path apps/soundry/src/design.data.json` · L3 · *"the frozen constants and their provenance comments"* | never |
| **C4** multiplicity ledger | `route #/results/ledger` · L2 · *"the denominator: every contrast ever registered"* | never |
| **C5** learning transfer | `route #/results/ledger?hyp=<adopted>` · L2 | **today:** `none`, reason *"No factor level has been adopted and no round has closed; there is nothing to trace into the catalogue. If this card never fills, the bench was a hobby."* watch `soundry.rounds.closed>=2` |
| **B6** the refusal card | `path vault/EXPERIMENT_STANDARD.md` · L3 · *"the noise decomposition, measured, §2"* | never |
| **B8** promote / bench | `route #/results/ledger:open=decisions` · L2 · *"the decision log and its evidence"* | never |
| **B1** round header · **B9.2** · **B9.3** · **C6** · **C7** | `route #/results/ledger` / `#/results/<id>/analysis` · L2–L3 as listed above | never |
| **X6** the loop *(Part 2, SPLICE 8)* | `kind:"runtime"`, `configKey:"soundryOrigin"`, `fallbackPath:"data/soundry/rounds/<id>/"` · L3 | **never `none`** — with `soundryOrigin: null` the path renders and V15 is satisfied without a literal origin, exactly as SPLICE 8 ruled |

Every Analytics card is converted the same way: §14.2's `drill`/`reason` pairs become `Deeper`
values with their existing targets and levels; SPLICE 8's `deeper →` lines (X0→`#/music/provenance:open=joins`,
X1→M1.6 filtered `distributed=1`, X2→M2.1, X3→M2.3, X4→M2.4, X5→M2.5) are already correct and
only change shape.

---

# SPLICE T8 — [APPEND] the §15 test contract, items 27–34 (continuing R10's block)

27. **`deeper-registry.spec.ts`** *(both apps; renames and absorbs `drill-registry.spec.ts`)* — the
    build-failing one. Each card module exports `specs(fixture) => { card, plots }`, pure and
    node-callable, and declares `fixtures: FixtureId[]`. The test enumerates **every card × every
    declared fixture × every plot** and asserts: (a) `deeper` present — a missing field fails naming
    app, card, fixture and plot id; (b) V24–V27; (c) `kind:"route"` targets resolve in the frozen
    `ROUTES` list and are not self-links; (d) the `deeper` slot renders non-empty in every case;
    (e) `kind:"none"` counters are the same values as the card's ghost/empty counters (V26).
    **`scripts/card-registry-check.mjs` runs the same assertions as a CI gate**, so a card that
    never gets a test still cannot ship without `deeper`.
28. **`shelf-vocabulary.spec.ts`** — `FILTER_ORDER` equals the frozen fifteen, in order;
    `LOCAL_KEYS ∩ FILTER_KEYS = ∅`; every `SHELF[k]` has a non-empty `guidance`; **no file under
    `apps/*/src/**` outside `shelf.ts` contains a retired key literal** (`"win="`, `"pf="`, `"?m="`,
    `"2026-only"`, `"include+label"`, `deeperAbsentReason`, `drill:`) — the anti-second-vocabulary
    test, and the reason U0b exists; `PROV_RANK` behaviour for `derived`, `prose-archive` and
    `randomised-round` (§3.2).
29. **`shelf-applicability.spec.ts`** — invariant F1 across every card in both apps, both
    directions, plus: every `filtersLocked` entry has a ≥10-word reason; `chipState` returns one of
    the six states for every (key, card, app) triple and never `undefined`; the precedence order of
    §3.4 is asserted with a fixture that qualifies for two states at once.
30. **`shelf-prereg-interlock.spec.ts`** *(Soundry)* — `baseline_cohort` is written at freeze, is
    inside `spec_hash`, and a resolved state differing from it raises `ANALYSIS DEVIATION` and drops
    the round from the multiplicity denominator.
31. **`shelf-zero-result.spec.ts`** — for a matrix of filter states over the real 210-row snapshot,
    every state yielding 0 rows produces either ≥1 relaxation with `resultingN > 0` or a
    `Reset to defaults` primary action. **No state has zero next clicks.** Also: no state blanks a
    panel; `empty.kind === "filtered-empty"` in every zero case.
32. **`shelf-url.spec.ts`** — round-trips `fromUrl ∘ toUrl` for 200 generated states; fixed key
    order; defaults omitted; `fv=1` always present; an unknown key survives untouched (T6);
    `q` containing a comma and a `#` survives; `crossAppHref` into the Soundry and back is
    byte-identical; two states differing in one key produce URLs differing in one segment;
    canonicalisation on load rewrites via `replaceState`.
33. **`soundry-shelf.spec.ts`** — the Soundry renders all fifteen chips; the seven locked keys are
    `locked` (not `dimmed`) on B2/B4/B5/B7/B9.1/C1/C2/C4/C5 with the verbatim reason; `mod` is
    `parked`; `blind` is forced and locked inside `#/panel/*`; A2/C3 honour `co`/`out`/`floor`/`est`
    and print the cohort expression that produced their constants.
34. **`expiring-refusal.spec.ts`** — for every `kind:"none"` in both apps, `watchSatisfied(watch,
    manifest)` must be **false**. Runs twice: against `apps/_fixtures/load-manifest.json` (the CI
    gate, deterministic) and against the real on-disk manifest (the local gate). A refusal whose
    condition has come true fails, naming the plot and the file that now exists:
    *"`data/snapshots/soundry/latest.json` exists. B4's refusal has expired."*

**`forbidden-strings.spec.ts`, merged extension** (adds to R10's list): `"deeperAbsentReason"`,
`"drill:"`, `"?m="`, `"&win="`, `"&pf="`, `"click here"`, `"see more"`.

---

# SPLICE T9 — file tree, build order, canonical paths

**§15 file tree additions:**
```
apps/lofn-kit/src/state/shelf.ts          cohort.ts        applicability.ts
apps/lofn-kit/src/contracts/deeper.ts
apps/lofn-kit/tests/shelf-vocabulary.spec.ts  shelf-url.spec.ts  shelf-applicability.spec.ts
apps/_fixtures/load-manifest.json         (the deterministic WatchId manifest, committed)
apps/lofn-analytics/src/shell/filters.ts  (rewritten: renders kit state, defines no keys)
apps/lofn-analytics/tests/deeper-registry.spec.ts  shelf-zero-result.spec.ts  expiring-refusal.spec.ts
apps/soundry/src/shell/filters.ts         (new — the Soundry had no shelf)
apps/soundry/src/cards/b10-variant-table/
apps/soundry/tests/soundry-shelf.spec.ts  shelf-prereg-interlock.spec.ts  deeper-registry.spec.ts
```
**DELETED:** `apps/lofn-analytics/tests/drill-registry.spec.ts` (renamed, not dropped).

**Build order — U0b is no longer a research task, it is this patch. Replace its text:**

> **U0b · COORDINATOR, SERIAL, BLOCKS L-LEDE, L-FUSE-A, L-FUSE-C AND EVERY CARD LANE IN BOTH APPS.**
> Land SPLICE T1 (`shelf.ts`, `cohort.ts`, `applicability.ts`) and SPLICE T4 (`contracts/deeper.ts`,
> the required `deeper` on `PlotSpec`/`CardSpec`) inside **L-KIT**, before any card module is
> written — making `deeper` non-nullable is a signature change to every card in the estate and it
> must not be retrofitted lane by lane. Then tests 27–29, 32, 34. The `FeedId` membership of
> `socials` and `distribution` against `scripts/ingest_receiver.mjs:43-52` remains U0b's other half
> and is still owned by another seat.

**L-SND-SHELF (new lane, parallel with L-SEC-SND, needs L-KIT):** `apps/soundry/src/shell/filters.ts`
· the `baseline_cohort` field in `preregistration.json` and its `spec_hash` coverage ·
`apps/soundry/src/cards/b10-variant-table/` · tests 30, 33.

**CANONICAL PATHS additions (§8.3.4, mirrored to `.agents/`, `check_skill_mirror.py` must print
IN SYNC):** `KIT_STATE → apps/lofn-kit/src/state/`, `KIT_CONTRACTS → apps/lofn-kit/src/contracts/`.

---

# SPLICE T10 — Overrules, rejections, open questions

**`## Overrules ACCEPTED`, APPEND:**

- **ONE FILTER VOCABULARY, AND PART 2'S SHELF BECOMES ITS RENDER** (Task Taxonomist). Three key
  sets were live (`mod|snap|co|…`, `m|win|pf|out|est|prov`, and `?m=` on the rail). The kit's names
  win; Part 2's `est` is adopted into the union and its 44px display contract is kept whole.
  **PRESERVES Board 7's "carry between tabs" as a mechanism instead of a hope** — one serializer
  means the carry is structural, and F1 means a card cannot opt out of it quietly.
- **A DEFAULT THAT REMOVES ROWS IS DRAWN AT FULL INK WITH ITS DENOMINATOR** (Task Taxonomist,
  narrowing §3.4). `co=modern` hides 164 of 210 clips on first paint. **PRESERVES §3.4's ghost chip**
  for view-only defaults, and preserves her own instruction that the app never mislead — a quiet
  default is a number nobody measured, told as a picture.
- **THE SOUNDRY GETS A SHELF, AND SEVEN OF ITS CHIPS ARE LOCKED, NOT ABSENT** (Task Taxonomist).
  Board 7 covers both apps; Part 3 rendered nothing. **PRESERVES the randomisation guarantee** — the
  Synthesist's sentence becomes a padlock rather than a grey chip, the value she set in Analytics is
  held and restored, and `preregistration.baseline_cohort` makes a re-cut a detectable deviation.
- **`deeper` IS REQUIRED AND ITS ABSENCE IS A TYPED VALUE** (Task Taxonomist). §14.2's overrule of
  her "if they exist" was correct and unenforceable: an optional field let twelve Soundry plots ship
  with neither a link nor a refusal. **PRESERVES her conditional honestly** — `kind:"none"` carries
  the reason, the artefact by name, the unlock counters, and a `watch` that **fails the build the
  day the deeper result exists** (THE EXPIRING REFUSAL). She wrote "if they exist"; now the app is
  forbidden from claiming they do not once they do.
- **B10 · THE VARIANT TABLE IS ADDED SO NINE CARDS CAN HAVE A REAL L2** (Task Taxonomist).
  **PRESERVES §14.1's four-level ladder** in the app that had no rows rung, and refuses the cheaper
  path of nine fake refusals. Sort defaults to slot, never to outcome (§5.4.7).

**`## Overrules REJECTED (her intent wins)`, APPEND:**

- **REJECTED — keeping both key sets behind a translation layer.** Two vocabularies with a shim is
  three vocabularies by the second lane. One union, one serializer, one anti-drift test.
- **REJECTED — a Soundry shelf that simply omits the inapplicable keys.** Omission is the silent
  drop Board 7 names. Locked, dimmed, inert and parked are four different truths and each is drawn.
- **REJECTED — `deeper: null` with a reason string beside it.** Two fields that must agree will
  disagree. One field, one discriminated union, one validator.
- **REJECTED — a `deeper` href that hardcodes the filter state it drills into.** A drill that
  changes the analysis under you is a dead end with a destination.

**`## Open questions for The Scientist`, APPEND:**

- **OQ-FILTER-1 · The cohort default is `modern` (46 of 210) and the chip says so at full ink from
  first paint.** The alternative is `all`, which shows the whole catalogue and disagrees with every
  printed constant in both apps. **RECOMMENDED DEFAULT: `modern`, with the one-click `[all 210]`
  where you can see it.** If you would rather open on the full catalogue, it is one line in
  `SHELF.co.defaultValue` and the printed baselines must move with it.
- **OQ-FILTER-2 · Should the Anvil's power card follow the shelf, or pin to `modern` forever?**
  Following it is what makes Board 7's carry do real work — but it means the constants in your
  pre-registration depend on a chip. The interlock (`baseline_cohort` inside `spec_hash`) makes any
  drift loud either way. **RECOMMENDED DEFAULT: follow the shelf.**
- **OQ-DEEPER-1 · B7's listener feedback refuses on `cleanFeedbackText()`, which does not exist in
  this repository** (STILL OPEN #7). Its refusal is watched: the day `input-cleaning.mjs` lands,
  test 34 fails until B7 is given a real target. That is the intended behaviour and it will look
  like a broken build. **RECOMMENDED DEFAULT: port the function before the panel lane opens.**

---

# WHAT THIS PATCH DOES NOT DO

- It does not close STILL-OPEN #2 (guidance for all elements). It adds a required `guidance` to
  every shelf chip, which covers the controls *this* patch introduces and nothing else. There is
  still no `GuidanceNote` registry and no `guidance-coverage.spec.ts`.
- It does not build the Headwinds right rail (#1), the Sonic Anvil generator (#3), the panel's
  audio (#6), `cleanFeedbackText` (#7), the baseline reconciliation (#8), `FeedId` membership (#9),
  or the second-snapshot habit (#10).
- It resolves the *filter* half of U0b only. The `FeedId` half is still blocking L-FUSE-A.


---

# APPENDIX — seat patch: THE COMPLETENESS CRITIC — adversarial requirements auditor (round two) (role-anchored; every claim below traced to a line in BUILD_SPEC.md / SPEC_PATCH.md / WHITEBOARD_TRANSCRIPTION.md or recomputed from files on disk (data/snapshots/suno/latest.json, data/snapshots/socials/latest.json, data/snapshots/distribution/latest.json, scripts/ingest_receiver.mjs, scripts/analyze_power.py, vault/EXPERIMENT_STANDARD.md, vault/SOCIAL_VOICE.md Appendix C, vault/AUTONOMY.md Part 2, E:/git/saga/game-dev/apps/forge-arcade/input-cleaning.mjs, output/mp3/))

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/SPEC_PATCH.md, E:/git/private-claw/apps/soundry/panel-audio.mjs, E:/git/private-claw/apps/soundry/panel-routes.mjs, E:/git/private-claw/apps/soundry/panel.html, E:/git/private-claw/apps/soundry/input-cleaning.mjs, E:/git/private-claw/apps/soundry/input-cleaning.d.mts, E:/git/private-claw/apps/soundry/panel-cleaning.mjs, E:/git/private-claw/apps/soundry/src/topline.name.ts, E:/git/private-claw/apps/soundry/src/sampler.data.json, E:/git/private-claw/apps/soundry/scripts/build-sampler.mjs, E:/git/private-claw/apps/soundry/tests/panel-audio.spec.ts, E:/git/private-claw/apps/soundry/tests/input-cleaning-parity.spec.ts, E:/git/private-claw/apps/soundry/tests/panel-cleaning.spec.ts, E:/git/private-claw/apps/soundry/tests/topline-name.spec.ts, E:/git/private-claw/apps/soundry/tests/decibal-firewall.spec.ts, E:/git/private-claw/apps/soundry/tests/glm.spec.ts, E:/git/private-claw/apps/lofn-kit/src/stats/baseline.lock.json, E:/git/private-claw/apps/lofn-kit/src/stats/power.ts, E:/git/private-claw/apps/lofn-kit/src/stats/novelty.ts, E:/git/private-claw/apps/lofn-kit/src/plot/spec.ts, E:/git/private-claw/apps/lofn-kit/src/contracts/provenance.ts, E:/git/private-claw/apps/lofn-kit/src/signature/uncounted.ts, E:/git/private-claw/apps/lofn-kit/tests/baseline-lock.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/signature.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/daymark.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/feed-command.spec.ts, E:/git/private-claw/apps/lofn-analytics/src/signals/register.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m0-lede/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m0-1-dial/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m3-3b-two-spotifys/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/art-uncounted/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/c8-novelty-frontier/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/house-record/index.ts, E:/git/private-claw/apps/lofn-analytics/RUNBOOK.md, E:/git/private-claw/apps/lofn-analytics/tests/posture-hosted.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/forbidden-strings.spec.ts, E:/git/private-claw/data/watchlist/competitors.template.json, E:/git/private-claw/scripts/analyze_power.py, E:/git/private-claw/.claude/skills/lofn-analytics/SKILL.md, E:/git/private-claw/.agents/skills/lofn-analytics/SKILL.md, E:/git/private-claw/.claude/skills/lofn/SKILL.md, E:/git/private-claw/.agents/skills/lofn/SKILL.md, E:/git/private-claw/CLAUDE.md, E:/git/private-claw/AGENTS.md

# SPEC_PATCH_R2 — THE COMPLETENESS CRITIC

*Round two. Assignment: STILL OPEN #6–#11 (SPEC_PATCH.md:637–642) plus anything independently found
dropped, plus the line-by-line whiteboard trace nobody has done.*

**How to apply.** Same convention as round one. `[SPLICE @anchor]` = insert/replace at the named
anchor in `E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`. Rulings are
numbered **R13–R24** and each overrule names the intent it preserves. Round one's R1–R12 stand
except where §D says otherwise, explicitly and with an argument. **Nothing here invents a number:**
every constant below was recomputed from a file on disk and the recompute command is printed with it.

---

# §A · THE EIGHT-BOARD TRACE

Every stated requirement on `WHITEBOARD_TRANSCRIPTION.md`, in board order. `SAT` = satisfied and
buildable · `NAR` = narrowed, intent named · `OVR` = overruled, intent named · `BLK` = specified but
blocked by a named open item · `DROP` = no artifact traces to it.

| # | Board | Requirement (transcription line) | Traced to | Status |
|---|---|---|---|---|
| 1 | 1 | `analytics.lofn.ai` | Part 2 title (BS:1371); posture local-only (BS:1383); OQ-SEC-1 | NAR — hosted door built-not-opened |
| 2 | 1 | Left rail `Art · Music · Animation`, entries beneath | §2 rail (BS:1407–1431) | SAT |
| 3 | 1 | **Top Trends & Insights** (centre column) | M0.0 THE LEDE (SPEC_PATCH SPLICE 11) | OVR — unit changed to registered signals |
| 4 | 1 | ↑ `Baile Phonk` "↗/like purely" | M0.0e pinned refusal row | OVR — answered in her words, not ranked |
| 5 | 1 | ↓ `Anthems` | M0.0e, same row | OVR — same |
| 6 | 1 | **Metric Leaders** → `Plays` + sparkline | M1.2 Reach leaders (BS:1527) | NAR — §4.3 r7: one snapshot ⇒ sparkline is `UNKNOWN` |
| 7 | 1 | **Metric Leaders** → `Likes` + sparkline | M1.1 (BS:1520) | NAR — same |
| 8 | 1 | **Platform Reach** `Suno`/`Spotify`, slope/bump chart into a right-hand box | M3.7 (BS:1651–1655) | NAR — ghosted, `unlocks at 2 snapshots` |
| 9 | 1 | **Headwinds**, boxed, RIGHT RAIL | M4 tab (BS:1658) + one clause "mirror into a persistent right rail" (BS:1659) | **DROP** — no card spec, no grid slot, no file in §15 `src/shell/`. STILL OPEN #1, other seat. Confirmed still open. |
| 10 | 1 | ↓ `Suno uploads` | M4.3 register #8 (220→210) | SAT |
| 11 | 1 | `Losing followers` | M4.4 (BS:1687) | OVR — refuses to draw a decline; shows both sources |
| 12 | 1 | Colors? / Light-Dark | Decisions block (BS:1836) | SAT |
| 13 | 1 | White background | light default | SAT |
| 14 | 1 | Sunset colors? | Decisions (BS:1838) — sequential ramp + chrome only | NAR |
| 15 | 1 | Daily palette? | THE DAY MARK §7.3 (BS:719) | OVR — data-derived glyph replaces rotating scale |
| 16 | 2 | **The Soundry** | Part 3 (BS:856) | SAT |
| 17 | 2 | Access only to Scientist + Invited Collabs | §0.3 PANEL LAN MODE + §0.5 hosted Analytics (R7) | SAT, narrowed twice |
| 18 | 2 | GUIDED EXPERIMENTATION | Anvil route ladder (BS:901) | SAT |
| 19 | 2 | `12 song tests a day` | A2 `cards_per_day` 12; A5 twelve cards (BS:1016) | SAT |
| 20 | 2 | `get feedback from listeners` | B7 (BS:1093) + `observations/panel.jsonl` (BS:1154) + PANEL LAN MODE | **BLK → CLOSED by R13 (§B-1)** — the panel had nothing to play |
| 21 | 2 | `① Learn` | C5 Learning transfer (BS:1278) | SAT |
| 22 | 2 | `② Take winners + make full songs` | B8 Promote → Phase-0 brief (BS:1103) | SAT |
| 23 | 2 | **`Goal: NEW VIRAL EXPERIENCES` ← `NOVELTY`** (underlined twice) | — | **DROP. Zero occurrences of "novelty" or "viral" in BUILD_SPEC.md or SPEC_PATCH.md** (grep, both files). Closed by R18 (§C-1). |
| 24 | 2 | `soundry.localhost` | BS:886 (hosts file), port 5231 | SAT |
| 25 | 3 | Left rail `Sonic Anvil · Experiment Results · Analytics` | three tabs, no fourth (BS:860) | SAT |
| 26 | 3 | "What should we try today?" + large empty input | A0 (BS:903) | SAT |
| 27 | 3 | numbered cards `1.`…`12.` | A5 3×4 grid (BS:1018) | SAT |
| 28 | 3 | "full suno prompts:" Title/Style/Exclude/Lyrics/Caption | A5 wells (BS:1020–1021) | SAT as a **form**; the **generator that fills them is DROP** — STILL OPEN #4, other seat. Confirmed still open. |
| 29 | 4 | Topline card `Decimabal` | C1 DECIBAL (BS:1172) | NAR → R17 (§B-6a) makes the name a one-file constant |
| 30 | 4 | `Total Lift vs Avg artist: 123%`, `Top line` arrowed | C1 refuses it (BS:1239–1241) | OVR, but **the replacement slot was left empty** → closed by R17b (§B-6b), THE HOUSE RECORD |
| 31 | 4 | Hypotheses box: effect axis + point + CI whisker | C2 (BS:1250) + Grain Whisker §7.2 | SAT |
| 32 | 4 | axis `0/+5/+10/+15/+20` on hypothesis 1 | C2 shared symmetric log-ratio axis | OVR (BS:1256) |
| 33 | 4 | axis `−10/−5/0/+5/+10` on hypothesis 2, marked `35 ±` `[?]` | C2 same axis; **`35 ±` is not interpreted anywhere** | NAR + open transcription ambiguity — added to OQ list (§B-6d) |
| 34 | 4 | `n = ~` · `#days = ~` per hypothesis | C2 `nField` (n · #days · achieved MDE · BH q) | SAT |
| 35 | 4 | `[more details]` | C2 drawer, kept verbatim (BS:1258) | SAT |
| 36 | 4 | third, unlabelled ↓ hypothesis | C2 holds all four verdict states permanently | SAT |
| 37 | 4 | **"Include POWER ANALYSIS."** | A2 (BS:932) + C3 (BS:1262) | SAT in form; **its three constants were fabricated** → R15 (§B-3) |
| 38 | 5 | Feed 1 — Lofn Socials | `suno`+`socials`+`distribution` (BS:1393–1395) | SAT; `FeedId` untypeable → R16 (§B-4) |
| 39 | 5 | Feed 2 — Competitor Scans, "5 watched growing/trending targets" | M5.7 roster, R1/R3 | SAT, gated on her |
| 40 | 5 | Feed 3 — Market Trends + Scan | SPEC_PATCH SPLICE 7 row 3 | NAR — `scripts/scan_market.py` does not exist; named as a file to write |
| 41 | 5 | Feed 4 — Soundry Artist Results + Feedback | SPLICE 9 reader + `emitter-silent` | SAT |
| 42 | 6 | "Give direction for search (next song)" | Tab Directive §4.4 + `nextMove.lever` | SAT |
| 43 | 6 | "+ analytics for Lofn (private-claw) **and collaborators**" | §0.5 hosted posture (R7) | SAT, gated on P0 + her word |
| 44 | 6 | "Combine all 4 sources into useful insights" | M6 CONFLUENCE (R5) | SAT |
| 45 | 6 | lyric length / line | M2.1 + M6·X2 | SAT |
| 46 | 6 | Suno EMO usage | M2.3 + M6·X3 (Tier S) | SAT — per-tag refused with reason |
| 47 | 6 | genre performance | M2.4 + §13 interlock | SAT — one screen object |
| 48 | 6 | trend setters + breakers | M2.5 + M3.1 + X5 two clocks (R4) | SAT, split |
| 49 | 7 | applies to BOTH apps | Parts 2 and 3 | SAT |
| 50 | 7 | *(margin)* "known for strong, unique + clean visuals" | §7 signature devices | **NAR → R19 (§C-2)**: all three devices key on per-item data; Art, the largest surface, has none |
| 51 | 7 | "A panel of dataviz and data UI/UX experts determines the tabs/cards/plots" | the panel process itself | SAT (process) |
| 52 | 7 | **"BE CREATIVE AND UNIQUE!"** | §7.1 COMB · §7.2 GRAIN WHISKER · §7.3 DAY MARK | **NAR — three real signature devices exist and they are good; none of them can render on Art or Animation.** R19 adds the fourth. |
| 53 | 7 | "All headliner insights on each tab should help guide our next creation" | §4.4 Tab Directive + `nextMove.routesTo` | SAT |
| 54 | 7 | **"Show trends as well as 'now'."** | M0.6, M2.5(a), M3.7, B3, LEDE temporal band | **BLK → CLOSED by R20 (§B-5)** — nobody owned the second snapshot |
| 55 | 7 | "All segmentations & filters carry between tabs when applicable, with clear controls" | Part 1 §3.4 dimmed-not-dropped (BS:412) | **BLK** — two incompatible key vocabularies (Part 1 §3.2 vs Part 2 §3.3); Soundry has no chip strip at all. STILL OPEN #3, other seat. Confirmed still open. |
| 56 | 7 | "Every insight / plot should link to deeper results if they exist" | §14.2 overrules the conditional (BS:1780) — **Analytics only** | **DROP in Part 3.** ~12 Soundry plots declare no `deeper`. STILL OPEN #5, other seat. Confirmed still open. |
| 57 | 7 | **"Tooltips, alt-text, and informational guidance for ALL elements"** | §5.1 `GuidanceNote` (BS:539) | **NAR and getting worse** — §5.1 r1 narrows to statistics; no registry, no `guidance-coverage.spec.ts`. STILL OPEN #2, other seat. Confirmed still open. |
| 58 | 8 | "Lofn, please moderate. We want to represent us both." | Overrules ACCEPTED / REJECTED discipline | SAT (process) |
| 59 | 8 | "All of my past music / art / etc. metadata **+ song mp3's**, images are now in `output/`" | `output/past_art` (A0.3–A0.5) · `output/images` (A0.3) · `output/past_video` (N0.3) · `output/past_music`+`output/songs` (§14.3) | **PARTIAL — the mp3s were dropped.** 224 `.mp3` on disk, 66 titled catalogue files in `output/mp3/`. Used by R13 (§B-1). |
| 60 | 8 | "you now control our socials! go respond to your fans" | A0.6 `forbidden:` no outbound (BS:1730) | OVR, intent named — routed to the separate social lane, not to these apps |
| 61 | 8 | "Get the lay of your domains" | M3.2 platform roster · A0.x · N0.x | SAT |
| 62 | chat | "Use an agent fleet" | build-order lanes | SAT |
| 63 | chat | **"Add to Lofn (new first-class skills, mirrored into `.agents/`)"** | BS:2000 unit W5 writes `.claude/skills/lofn-analytics/SKILL.md` + twin | **DROP BY REPLACEMENT** — round one's *Revised build order* (SPEC_PATCH:646–656) supersedes all three v2 build-order blocks and contains **only** the CANONICAL-PATHS rows. W5 vanished. Closed by R21 (§C-3). |
| 64 | chat | "webapp like Sága's forge" | §15 Forge house pattern (BS:1792) | SAT |
| 65 | chat | "ignore the Instagram messages; Alexis Dreams stays unread" | §1 feed table, `SKIPPED BY INSTRUCTION` (BS:1394) | SAT |
| 66 | — | THE OVERRULE MANDATE | Overrules ACCEPTED/REJECTED blocks | SAT |
| 67 | OQ | Q1 `Decimabal` name | — | R17a: unblocked, one-file constant |
| 68 | OQ | Q2 "Total Lift vs Avg artist" | — | **R17b: answered on disk. Appendix C already wrote the self-referential comparator.** |
| 69 | OQ | Q3 which five competitors | Appendix C names all five *slots* | R17c: pre-fill `why_this_one` from Appendix C; she types five handles, not five essays |
| 70 | OQ | Q4 canonical Spotify identity | — | **R17d: answered on disk, and the answer is a contradiction the app must render** (§C-4) |

**Score: 70 traced requirements. 43 SAT · 11 NAR · 8 OVR · 4 BLK-now-closed · 4 still open with
another seat (#9, #28, #55, #56, #57 → 5) · 3 DROP closed here (#23, #59, #63) · 1 DROP confirmed
open (#9).** The three that would have shipped invisibly are **#23 NOVELTY**, **#59 the mp3s**, and
**#63 the skills**. All three are hers, all three are in plain text, and no seat in round one names
any of them.

---

# §B · STILL OPEN #6–#11 — CLOSED

## B-1 · R13 · STILL OPEN #6 — THE PANEL HAS SOMETHING TO PLAY. THE BLOCKER WAS A CSP MISREADING.

**[SPLICE @ PART 3 §0.3, REPLACING the "NAMED GAP" block at SPEC_PATCH:258–261]**

Two facts kill the gap.

1. **`media-src 'self'` is already in the CSP** (BS:832). `<audio src="/panel/assets/…">` is governed
   by `media-src`, not `connect-src`. Round one's blocker — *"CSP is `connect-src 'self'`"* — is a
   category error. Nothing has to change in the CSP and nothing has to be fetched.
2. **Board 8 line 2 is the authority nobody read:** *"All of my past music / art / etc. metadata +
   **song mp3's**, images are now in `output/`."* Verified: `find output -iname "*.mp3" | wc -l` →
   **224**; `output/mp3/` holds **66** files named by song title (`Dial Tone of God.mp3`,
   `The Limit Always Wins.mp3`, …). She told us where the audio is on the last board.

**New file `apps/soundry/panel-audio.mjs`** (zero deps, `node:fs|path|crypto` only):

```js
// Resolves a variant id to ONE local file. Never fetches. Never transcodes. Never downloads.
export function resolvePanelAudio(roundId, variantId, dataDir, repoRoot) // -> {path, bytes, sha256} | null
export function serveRange(req, res, file)   // 200 + Accept-Ranges: bytes; 206 on Range; 404 otherwise
```

Two and only two source roots, both operator-placed, both read-only:

| round `mode` | root | who puts files there |
|---|---|---|
| `CONFIRM` / `SCREEN` | `data/soundry/rounds/<id>/audio/<variant-id>.mp3` | she renders on Suno and drags the files in |
| **`CALIBRATION`** *(new)* | `output/mp3/<file>` — **allowlisted by `manifest.json` only**, never by directory listing | already on disk today |

**`CALIBRATION` — a third round mode, added to A1 (BS:918).** It plays existing catalogue audio to
a blind panel to prove the apparatus works before a single new song is rendered. It is
`randomized: false`, `controls.blockedOn: []`, its Insight tier is **hard-capped at `OBSERVATION`**,
it may **never** produce a DECIBAL, it never enters the multiplicity denominator, and its banner is
permanent: *"Calibration. The songs were not assigned; they already existed. This round measures the
panel, not the music."* Enforced in `preregistration.json` by `mode: "CALIBRATION"` ⇒
`primary_outcome.field` restricted to `panel_rating` and `instrument: "panel"`.

**`data/soundry/rounds/<id>/audio/manifest.json`** — required before `instrument: "panel"` is legal:
`[{variant_id, path, bytes, sha256, duration_s}]`. Path must resolve inside `data/soundry/rounds/<id>/audio/`
or `output/mp3/`; a `..` segment, a symlink, or a hash mismatch is a **hard 404 and a Schema-Guard row**,
never silence. A variant present in the round and absent from the manifest renders `pull-failed` (not
`emitter-silent`) with the file it wanted named.

**Empty state**, `emitter-silent`, with the verb:
> *"`data/soundry/rounds/R-…/audio/` exists and is empty. The panel is reachable and has nothing to
> play. Twelve files, named `<variant-id>.mp3`. `Copy-Item output/mp3/*.mp3 data/soundry/rounds/R-…/audio/`
> is the calibration path; for a real round they come out of Suno by hand. This bench does not download."*

**`apps/soundry/tests/panel-audio.spec.ts`** — (a) a manifest path escaping either root ⇒ 404, asserted
for `..`, absolute, and symlink; (b) `Range: bytes=0-1023` ⇒ 206 with `Content-Range`; (c) a sha256
mismatch ⇒ 404 + a Schema-Guard entry, never a partial body; (d) no `fetch`, `http.request`, `https`,
or `child_process` token anywhere in `panel-audio.mjs`; (e) a `CALIBRATION` round's analysis endpoint
returns tier `OBSERVATION` and no `lift` key.

**PRESERVES** Board 2's *"get feedback from listeners"* literally — it is now listenable today, with
audio that already exists, and without one new network capability. **OQ-PANEL-1 is answered and struck.**

## B-2 · R14 · STILL OPEN #7 — `cleanFeedbackText` IS VENDORED, BYTE-FOR-BYTE, WITH A DRIFT TEST.

**[SPLICE @ PART 3 §3.3, after the `cleanFeedbackText` sentence (BS:1160)]**

The implementation is on disk at `E:/git/saga/game-dev/apps/forge-arcade/input-cleaning.mjs` —
**85 lines, zero dependencies**, with `input-cleaning.d.mts` (15 lines) and
`tests/input-cleaning.spec.ts` beside it. It exports `cleanPlayerName`, `cleanFeedbackText`,
`prepareFeedbackForAnalysis`; `cleanFeedbackText(value) -> { value, flags[], safeForAnalysis }` over
NFKC + control/BiDi strip + markup strip + secret redaction + seven injection patterns + three abuse
patterns + link/repetition spam, 500-char cap, multiline.

**Ruling: copy it verbatim, do not import it, do not rewrite it.**
- `apps/soundry/input-cleaning.mjs` and `.d.mts` — **byte-identical copies**, plus a 3-line provenance
  header above the existing first comment naming source path and copy date. Cross-repo runtime imports
  are forbidden by the Vite alias policy (§8.1 allows exactly `@kit`), and a rewrite of a trust
  boundary is how a trust boundary rots.
- **`apps/soundry/tests/input-cleaning-parity.spec.ts`** — reads both files from disk and asserts the
  bodies are identical after stripping the provenance header. A drift in either repo is a red test,
  not a surprise. Skips with an explicit `it.skip` reason if `E:/git/saga` is not present, so CI on a
  detached checkout does not fail for the wrong reason.
- **Do not extend the vendored file.** A3-specific screening lives in a new
  `apps/soundry/panel-cleaning.mjs` that *imports* it and adds one flag:
  `pii` (email, E.164/NANP phone, `@handle`, or a bare URL host) ⇒ `safeForAnalysis: false`.
- **B7 amendment:** a note whose flags include `secret` or `pii` renders the **cleaned** `value`
  (the vendored function already substitutes `[redacted secret]`), never the raw string, plus the
  flag chips. It is still stored, still displayed, still not counted — the round-one rule, minus the
  ability to leak a third party's contact details onto the screen.
- **§15/§8 file-tree additions:** `apps/soundry/input-cleaning.mjs` · `.d.mts` ·
  `apps/soundry/panel-cleaning.mjs` · `tests/input-cleaning-parity.spec.ts` · `tests/panel-cleaning.spec.ts`.

**PRESERVES** `vault/COLLABORATOR_STANDARD.md`'s hold over A3 while keeping the no-`dependencies`
rule intact — the module has no dependencies to add.

## B-3 · R15 · STILL OPEN #8 — THE BASELINE CONFLICT IS A COHORT-DEFINITION CONFLICT, AND ONE CONSTANT WAS BACK-SOLVED.

**[SPLICE @ PART 3 §1.3 A2, REPLACING the constants block (BS:938–947); mirrored into PART 1 §4.3 rule 3
and PART 2 §3.3]**

Both numbers are arithmetically correct. They are **two different cohorts, and one of them uses a
mean where the other uses a median**, and the fact that both land on n=46 is a coincidence that hid
the collision from four seats. Recomputed from `data/snapshots/suno/latest.json`:

| | cohort | n | like rate | median plays |
|---|---|---|---|---|
| **`vault/EXPERIMENT_STANDARD.md` §1** (`scripts/analyze_power.py`) | modern (≥2026-02-01) ∩ **plays ≥ 50** ∩ **Triple Arch excluded** | 46 | **mean 4.4890 %** | **258** |
| **BUILD_SPEC §3.3 filter defaults** | modern ∩ **plays ≥ 60** ∩ outlier *included* | 46 | median 4.3695 % · mean 4.2875 % · pooled 1.5683 % | **267** |

**R15a — the house baseline is the STANDARD's cohort.** It is the one with a script, and the file
says *"Re-run it; do not trust it from memory."* `p̄ = 0.044890` is a **mean**; `m = 258` is a
**median**; the §3.3 default estimator is `median`. Every card that prints `4.49 %` must therefore
print `mean` inline (§3.3's own rule) — the current `denominatorNote` *"at a 258-play median, 4.49% is
~12 likes"* silently mixes the two and is amended to
*"at a 258-play median, the cohort **mean** rate 4.49 % is ~12 likes; the 95 % interval is ±2.53 pp."*

**R15b — `σ²_true = 1.204e-4` is STRUCK as back-solved.** The standard's measured decomposition
(population variance, ddof=0, over that cohort) reproduces exactly:

```
observed  σ²  = 3.2474e-4   (1.8021 pp)      # statistics.pvariance(like_rate)
binomial  σ²  = 1.8822e-4   (1.3719 pp)      # mean of p_i(1-p_i)/m_i, MEASURED per clip
true between  = 1.3652e-4   (1.1684 pp)      # the difference
binomial share = 57.96 %                     # her "58 %", reproduced
```
`1.204e-4` is what you get by *assuming* the binomial term is `p̄(1−p̄)/258 = 1.649e-4` (the median,
not the measured per-clip mean) and then solving for the residual that makes `MDE(n=114) = 1.150`
come out. That is fitting a constant to a conclusion, in the one app built against exactly that.
**It is deleted, and the `1.369e-4` floated in round one is also wrong — the reproducible value is
`1.3652e-4`.**

**R15c — no constant is typed into the spec again. Three artifacts replace them.**
1. `scripts/analyze_power.py --json > apps/lofn-kit/src/stats/baseline.lock.json` — the generator
   already exists and already has a `--json` mode (`scripts/analyze_power.py:179`). The lockfile is
   committed and carries `snapshot_pulled_at` + the cohort predicate.
2. `apps/lofn-kit/src/stats/power.ts` — pure, node-testable, the **only** implementation, imported by
   A2, C3 and the LEDE's `mde80`. It reads the lockfile; there is no literal in it.
3. **`apps/lofn-kit/tests/baseline-lock.spec.ts`** — regenerates the decomposition from
   `data/snapshots/suno/latest.json` **in the test** and asserts agreement with the lockfile to 4
   significant figures, and asserts `JSON.stringify(spec)` in both apps contains no literal
   `1.204e-4`, `4.49`, `258`, `1.150` outside `baseline.lock.json`. *A frozen constant nobody can
   regenerate is a quoted number, and §4.3 rule 11 already forbids quoting one without a DRIFT badge.*

**R15d — the two "frozen power points" are OUTPUTS of a day-rounding, not assertions, and the
`glm.spec.ts` line asserting `1.150` to 3 dp is STRUCK.** Reproduced from the standard's own formula
`n/arm = 2(z_{α/2}+z_β)²σ²/δ²`, δ = 0.15·p̄:

```
σ = σ_observed 1.8021 pp  ->  n/arm = 112.43  -> round UP to the 6-per-arm-per-day grid -> 114 -> 19 days
σ = σ_true     1.1684 pp  ->  n/arm =  47.27  -> round UP                               ->  48 ->  8 days
```
114 and 48 are **19×6 and 8×6**. That is why they are what they are, and nothing in the spec said so.
`glm.spec.ts` asserts the two **n** values from the lockfile σ's, not the MDE.

**R15e — "mature" is not a play count and the slider must stop implying it is.** Solving A2's own
formula for the m that makes `MDE(n=48) = 1.150` to 3 dp gives **m ≥ 74,275 plays per song**. Exactly
**one** clip in 210 has ever reached that (Triple Arch, 349,736); the catalogue median is 50 plays.
`mature` in the standard means **σ_true, i.e. m → ∞** — the binomial term removed, not a big number.
The A2.1 / C3 colour levels become `{90 · 258 · fully matured (binomial noise removed)}` and the
third series is drawn as a **horizontal asymptote**, not a third curve.

**R15f — the modelled binomial term is calibrated, dashed, and labelled.** For an arbitrary slider
value the model is `binom(m) = C/m` with `C = E_cohort[p(1−p)] = 4.2550e-2`, which reproduces the
measured `1.8822e-4` at **m_eff = 226**, not at the median 258 (the median understates the noise by
12.4 % because low-play songs dominate `E[p(1−p)/m]`). Every point on the curve except the two
measured anchors is drawn **dashed** and captioned *"modelled — calibrated at m_eff = 226."*

**Consequence, stated so it cannot be quietly absorbed:** under the true constants the app's honest
MDE at n=114/m=258 is **×1.155**, not ×1.150, and the LEDE's `margin` (which divides by `mde80`)
shifts by 3–4 % on every row. **The ranking ORDER is unchanged; every printed multiple changes.**
`L-LEDE` unblocks the moment `baseline.lock.json` lands.

## B-4 · R16 · STILL OPEN #9 — `FeedId` IS TWO SETS, AND THE DAY MARK PRINTS ONE DENOMINATOR.

**[SPLICE @ PART 1 §2.1 and PART 2 §1, §3.2; and PART 1 §7.3]**

Verified on disk. `scripts/ingest_receiver.mjs:43-52` fixes eight ids —
`suno · youtube · tiktok · spotify · instagram · competitors · market · soundry` — and
`:54-57` exits 2 on anything else. `data/snapshots/` contains **`distribution · socials · soundry · suno`**.
So `socials` and `distribution` **have files and cannot be written by the receiver**, and
`soundry` **is in the receiver and is an empty directory**. There is no single list; pretending there
is one is why the count is printed three different ways (`3 of 8` at BS:1403, `3 of 8` at BS:1444,
`3/9` at BS:727).

```ts
// apps/lofn-kit/src/plot/spec.ts
export const RECEIVER_FEEDS = ["suno","youtube","tiktok","spotify","instagram",
                               "competitors","market","soundry"] as const;   // ingest_receiver.mjs:43-52
export const READER_FEEDS   = ["socials","distribution"] as const;           // written by another producer
export const FEED_IDS = [...RECEIVER_FEEDS, ...READER_FEEDS] as const;       // 10
export type FeedId = typeof FEED_IDS[number];
export const isReceiverFeed = (f: FeedId) => (RECEIVER_FEEDS as readonly string[]).includes(f);
```

- **`JoinEvidence.left.feed` / `.right.feed` are now typeable.** `L-FUSE-A` (X0, X1) unblocks; U0b's
  second half is discharged.
- **A feed with `isReceiverFeed === false` has no `ingest_receiver` command, and its `EmptyState.how`
  may not print one.** Asserted: `feed-command.spec.ts` — for every `FeedId`, if `!isReceiverFeed(f)`
  then no rendered empty state for `f` contains the substring `ingest_receiver.mjs --feed <f>`.
  *(Same defect class as the `market` scanner: printing a command that would fail.)*
- **ONE denominator, computed.** `FEED_IDS.length` is the only source. **Today: `3 of 10 fed`** —
  `suno`, `socials`, `distribution` have `latest.json`; `soundry` is present-and-empty ⇒
  `emitter-silent`, which is **not** "fed". The DAY MARK's inner arc is `FEED_IDS.length` segments.
  `daymark.spec.ts` asserts no literal `of 8`, `of 9`, `3/8`, `3/9` string appears in any rendered
  output; BS:1403, BS:1444 and BS:727 are all replaced by the computed value.

## B-5 · R17 · STILL OPEN #10 — THE SECOND SNAPSHOT GETS A NAMED OWNER, AND THE HABIT BECOMES A MEASURED OBJECT.

**[SPLICE @ PART 2 §4, extending M0.1; and a new §14.6]**

`vault/AUTONOMY.md` Part 2 already names the mechanism and the constraint:
*"Claude writes this file; Claude does not enable a schedule"*, and the non-waivable bulkhead —
*research → generate → QA → **STOP at drafts on disk***. A daily `python3 scripts/ingest_suno.py` sits
strictly inside that bulkhead: it reads a public endpoint, writes one file, and renders/publishes/spends
nothing. **So the schedule is doctrinally permitted and we still may not switch it on.**

What ships instead is the thing that actually fixes it — **the app measures the habit**:

- **M0.1 gains THE CADENCE STRIP.** 30 cells, one per day, under the snapshot dial. Filled where a
  file exists in `data/snapshots/suno/` for that date; absent-hatch where none. No colour, no streak,
  no congratulation — this is not a growth console. Caption:
  `1 of 30 days pulled · longest run 1 · a delta needs 2 consecutive · a trend needs 3`.
- **It is the unlock counter for the entire temporal layer.** Every `no-history` card's
  `UnlockCounter` becomes `{label:"consecutive daily snapshots", have: <computed>, need: 2|3,
  how:"python3 scripts/ingest_suno.py", path:"data/snapshots/suno/", blocker:"time"}` — R9's
  `blocker: "time"` value finally has its only legitimate user, and the copy stops saying *"run this"*
  in nineteen places and says it in one, with a count beside it.
- **§14.6 (new) — the schedule is a document, not a switch.** `apps/lofn-analytics/RUNBOOK.md` carries
  the ready-to-paste scheduled-task line and one sentence: *"Enabling a schedule is a human act
  (`vault/AUTONOMY.md` Part 2). This app will not do it, and it will show you every day whether it
  happened."* The `[Copy schedule command]` control composes and copies; it never runs — the same
  compose-don't-write triad the watchlist already ships.
- **OQ-OPS-1 (new):** *"Enable the daily pull as a scheduled task, or keep it manual? Either is fine.
  The cadence strip will show which one you chose within a week."*

**PRESERVES** Board 7's *"Show trends as well as 'now'"* — the trend layer's dependency stops being an
invisible habit and becomes a rendered, counted, falsifiable object, which is the only honest thing
this app can do about a thing it is forbidden to automate.

## B-6 · R18 · STILL OPEN #11 — TWO OF HER FOUR QUESTIONS ARE ANSWERED ON DISK. THE OTHER TWO GET SMALLER.

**(a) `Decimabal` — unblocked without an answer.**
**[SPLICE @ PART 3 §4.1 head]** New file `apps/soundry/src/topline.name.ts`:
```ts
export const TOPLINE = { name: "DECIBAL", provisional: true,
  note: "Handwriting on Board 4 reads 'Decimabal'. Best guess. One word, one file." } as const;
```
No card, test, route or copy string hardcodes it. `topline-name.spec.ts` asserts the literal
`DECIBAL` occurs in exactly one source file. While `provisional`, the card renders the name with a
`[?]` GuidanceNote carrying the transcription note. **Renaming is a one-line edit, so the question
stops blocking the build.**

**(b) "Total Lift vs Avg artist: 123 %" — SHE ALREADY WROTE THE ANSWER AND THE PANEL MISSED IT.**
`vault/SOCIAL_VOICE.md` Appendix C: *"**BENCHMARK INTERNALLY TOO**, because it is free and already on
disk: 'The Limit Always Wins' runs 9.46 % like rate on 497 plays — 2.1× the catalogue baseline and
6.5× Triple Arch. The best evidence about me is not a stranger's account."* Verified in
`data/snapshots/suno/latest.json`: 47 upvotes / 497 plays = **9.4570 %**, created 2026-05-17;
9.4570 / 4.4890 = **2.106×** — her figure, reproduced to 2 sf.

`vault/EXPERIMENT_STANDARD.md` §5.5 licenses exactly this: *"…or the topline must be an internal,
self-referential index."* **[SPLICE @ PART 3 §4.1, adding a fourth reference class and amending the
"exactly three" line]**

> **4. HOUSE RECORD mode.** The best era-controlled like rate ever measured in this catalogue, with
> its `n`, its Wilson interval, and the current cohort's distance from it. Renders **beside** DECIBAL,
> never as DECIBAL, in her Board-4 topline slot. Today: *"House record 9.46 % (47/497, 95 % CI 7.2–12.3 %,
> 2026-05-17). The modern cohort's mean is 4.49 % (n=46). We are at **47 % of our own best**."*
> Tier `OBSERVATION`, `randomized: false`, `eraControlled` required, **no causal verb**, `forbidden:`
> *"may not be used to select a style, may not enter `vault/gates.yaml`, and is not a lift — it is a
> distance from one song we already made."*

This is the only figure in the estate that can honestly occupy the slot she drew a number in.
It is not a comparison to a population we never measured (§5.5, upheld). **The DECIBAL firewall
stands** — HOUSE RECORD is a separate card id, a separate Insight, and `decibal-firewall.spec.ts`
asserts no `lift` field and no `roundId` on it.
**PRESERVES** her Board-4 topline as a real object instead of an empty state, using her own written
instruction, at the size she drew it.

**(c) Which five competitors — the ask shrinks from five essays to five handles.**
**[SPLICE @ SPEC_PATCH SPLICE 10, amending the template]** `data/watchlist/competitors.template.json`
ships with `handle: null` **and `why_this_one` pre-filled verbatim from Appendix C's five slot
paragraphs** (`CADENCE_MIRROR` … `REMIX_HOST`), plus `disqualifiers` pre-filled from Appendix C's
UNIVERSAL DISQUALIFIERS and `six_numbers` pre-filled with her six field names
(`followers · public_clips · median_like_rate · reply_rate_last_20 · reply_len_ratio ·
comments_on_smaller_30d`). The ≥12-word floor is satisfied by her own text; the only null is the
handle. M5.7 renders `why_this_one` **editable but pre-filled**, marked `from vault/SOCIAL_VOICE.md
Appendix C`.

**(d) The Spotify identity — answered, and the answer is a contradiction.** See §C-4.
**(e)** Board 4's `35 ±` `[?]` annotation on hypothesis 2 is interpreted nowhere in the estate.
Added to Open Questions as **OQ-WB-1** rather than guessed. Marking a transcription ambiguity as
answered is the same defect as inventing a number.

---

# §C · INDEPENDENTLY FOUND DROPS

## C-1 · R19 · "NOVELTY", UNDERLINED TWICE, IS ABSENT FROM 484 KB OF SPEC.

`grep -i "novelt\|viral"` over `BUILD_SPEC.md` → **0 hits**. Over `SPEC_PATCH.md` → 1 hit, inside an
unrelated watchlist string. Board 2's goal line — **`Goal: NEW VIRAL EXPERIENCES` ← `NOVELTY`
(underlined twice)** — is the most emphatic mark on any of the eight boards and it is the only
whiteboard *goal* statement. The Soundry as specified optimises one thing: like-rate lift. **A bench
that only measures engagement is a growth console with a confidence interval on it**, which is
precisely what §7 Voice says this is not. Her word was the corrective and it was dropped silently.

**[SPLICE @ PART 3 §1.4 (A3), §4 (new C8), and PART 1 §2.4]**

**R19a — novelty is a CONSTRAINT on the design, not an outcome to maximise.** Maximising novelty
maximises noise. Board 2 reads `NEW VIRAL EXPERIENCES ← NOVELTY`: novelty is the **route**, not the
target. So it enters the Anvil as a floor and the Analytics tab as a frontier.

**R19b — two measures, both computable today, both from fields already in the snapshot.**
`apps/lofn-kit/src/stats/novelty.ts` (pure, node-testable):
```ts
// vocabulary distance: 1 - max Jaccard of the lowercased token set of `tags`
//                      against every clip created strictly earlier
noveltyVocab(tags: string, priorCorpus: string[]): number
// craft distance: Euclidean distance in the z-scored space of
//   words_per_line, emo_tag_count, section_count, duration, repeated_line_ratio
//   from the PRIOR cohort centroid (z-scores computed on the prior cohort only)
noveltyCraft(clip: Clip, prior: Clip[]): number
```
Every field is present in `data/snapshots/suno/latest.json` today (verified on row 0).

**R19c — the Anvil gains a NOVELTY FLOOR.** A5's existing `DUPLICATE CELL — this card buys nothing`
check gains a sibling: `TOO CLOSE TO THE CATALOGUE — noveltyVocab {x} against "{nearest title}"
({date}). This card is a rerun.` The floor is declared in `preregistration.json` as
`novelty_floor: { vocab: number, craft: number }` and is **frozen with the rest of the plan**, so it
cannot be relaxed after seeing the cards.

**R19d — new card `C8 · THE NOVELTY FRONTIER`, and it is LIVE today with no new feed.**
`mark` scatter · `x` `noveltyVocab` against everything created earlier, 0–1 · `y` like rate, with the
Wilson interval as a Grain Whisker on every point · `colour` era band (the quartet) · `refs` the
cohort baseline rule; a LOESS-free **binned median with its own interval**, never a fitted line
(§X4's no-fitted-line rule applies) · `nField` n per bin.
**Question:** *does making something we have not made before do better, or only different?*
- **Era control is mandatory** — novelty drifts with time and time is the dominant effect in this
  catalogue (§4.3 r4). `eraControlled === false` ⇒ capped `OBSERVATION`, prefixed `Uncontrolled:`.
- `forbidden:` *"may not select a style, may not become a gate, and may not be read as 'be weirder'."*
- Its empty state is impossible: the 210-clip corpus already answers it. **This is the one card on
  the whiteboard's goal line that needs nothing pulled.**
- Registered in `src/signals/register.ts` as `craft.novelty.frontier`, `kind: "contrast"`,
  `falsifier:` *"if the binned medians are flat across the novelty range at n≥10 per bin, novelty is
  not a lever and this card should say so."*

**R19e — "VIRAL" is defined once, honestly, and then mostly refused.** New GuidanceNote, glossary,
verbatim: *"We have exactly one viral event (Triple Arch, 349,736 plays, 93.5 % of everything, at a
third of our median like rate) and it converted worse than our average song. n=1. This app cannot
model virality and will not pretend to. What it can measure is whether new work is received better
than old work — that is C8."* This is the disappointed idealist's answer to a growth question, and it
is true.

**PRESERVES** Board 2's goal line exactly: novelty becomes a frozen design constraint and a live
measured frontier, and the word "viral" gets the one honest sentence the data supports instead of
being deleted.

## C-2 · R20 · THE SIGNATURE DEVICES ARE REAL AND NONE OF THEM CAN DRAW ART.

Answering the brief directly: **"BE CREATIVE AND UNIQUE" did produce actual signature devices** —
THE COMB (§7.1: 210 hairlines + tear glyph + off-scale gutter), THE GRAIN WHISKER (§7.2: hatch pitch
= `clamp(2 + 26/√n, 2, 14)` — uncertainty as the loudest texture on the page), THE DAY MARK (§7.3).
The Grain Whisker in particular is a genuinely new encoding and it encodes the app's actual thesis.
This is not a competent dashboard.

**But all three key on per-item data, and Art has none.** §3.4 puts THE COMB on *"every Music tab"*.
§7.2 is Soundry-only. Art gets four stat tiles (A0.1), two bar charts (A0.2, A0.4), a stacked bar
(A0.3), another bar (A0.5) and two prose cards. **The largest audience we own — 23,000 works,
21,000 likes, 3.55× Suno — gets the least distinctive surface in the estate.** That is the thing
that will embarrass us, and no seat in round one names it.

**[SPLICE @ PART 1 new §7.4; PART 2 §10 replaces A0.3's chart form]**

> ### 7.4 THE UNCOUNTED — Art rail · `signature/uncounted.ts`
>
> A field of 23,000 cells at 2×2 px on a 154×150 grid, one per NightCafe work. **The entire field is
> one `<rect>` filled with the `--absent` hatch pattern.** Every work for which per-creation
> engagement exists is drawn as **one additional `<rect>` in `--ink`**, positioned by creation order.
> Today the DOM contains exactly **one** rect and **zero** ink cells.
> Caption, Plex Mono 10: `23,000 works · 0 measured · 21,000 likes with no address`.
>
> *Why it is the signature:* THE COMB draws a population you can measure. This draws a population you
> cannot, at the same scale, on the same page, and it is the larger of the two. It costs O(measured)
> nodes, not O(23,000) — **the DOM literally grows as the ingest lands**, which makes the render cost
> itself an honest progress bar. Its data table (§5.2, mandatory) has one row per *measured* work:
> today `<summary>` reads `Data table (0 rows)`. It is the only chart in the estate whose success
> state is that it stops being interesting.
> Alt text generated from `(total, measured, likes, asOf)`. Implemented as
> `mark: "strip", stripPosture: "field"`; the hatch is the existing `--absent` pattern, **zero new tokens**.

**R20b — the Art rail gets the seventh tab too.** R5 set the rail order
`SIGNAL · RESPONSE · CRAFT · REACH · HEADWINDS · CONFLUENCE · PROVENANCE` and §10 lists six for Art.
**Art gets CONFLUENCE**, rendering X0 (the join matrix — live, and Art's join yield is 0 of 23,000,
which is the finding) and absent-hatch for the rest. Symmetric shells with visible holes, per §10's
own stated principle.

**R20c — the register is not allowed to be Music-only.** `k = registry.filter(byModality).length`
(M0.0b mechanism 1). With zero art registrations the Art rail's LEDE header prints
`0 signals registered` on our largest audience. Seed list gains, minimum:
`reach.art.nightcafe_scale` (`FINDING`/`PULL_THE_FEED`), `data.art.measurement_depth`
(`ABSENT`, `governs: "tab"` — the Art rail's Zone-0 blocker), `reach.art.community_untended`
(A0.6, `PULL_THE_FEED`), and for animation `data.anim.no_per_work` (`ABSENT`, `governs: "tab"`).
Each with a `falsifier`, per M0.0b.

**R20d — Art's finding may not sit behind a click.** §2 computes rail order by measured audience and
puts Art first, then lands the reader on Music (correctly — Music is the only modality with per-item
data). But the Headline Editor's own argument stands: *"a tab the reader must first choose is not a
headline."* So **M0.0 Zone 0 on the Music rail carries the Art blocker row** whenever Art's
measurement depth is 1/5 and its audience is the largest, in her words, with the route:
*"Art is our largest audience and our least measured — 21,000 NightCafe likes, none attributable to a
piece. [→ Art rail]"*. One row, ten-object budget respected (it is the Zone-0 slot, which is 0-or-1).

## C-3 · R21 · THE DOCTRINE LANDING WAS DROPPED BY REPLACEMENT.

Board 8/chat: *"**Add to Lofn** (new first-class skills, mirrored into `.agents/`)."* `BUILD_SPEC.md:2000`
unit **W5** does this: write `.claude/skills/lofn-analytics/SKILL.md`, hand-write the `.agents/` twin in
Codex voice (**never `sed`**), add roster rows to both `skills/README.md`, backfill the two already
missing (`lofn-render-audit`, `lofn-step11-packager`), insert the identical bullet at `CLAUDE.md:20→21`
and `AGENTS.md:20→21`, run `check_skill_mirror.py`, **manually diff `CLAUDE.md` against `AGENTS.md`
(no script does this)**.

Round one's *Revised build order* (SPEC_PATCH:646–656) supersedes all three v2 build-order blocks and
carries **only the seven CANONICAL PATHS rows**. W5 is gone. `U-FINAL` does not restore it.

**[SPLICE @ SPEC_PATCH Revised build order, restoring W5 as `U-FINAL-b`]** — verbatim as BS:2000, plus:
- the new skill is **`lofn-analytics`** (one skill, both apps: the read side and the bench are one
  practice), and its `.agents/` twin is hand-written in Codex voice;
- CANONICAL PATHS additions extend round one's seven with the three Critic-2 named and round one did
  not adopt: `DAYMARKS → data/daymarks/` *(already in round one)*, plus `SOUNDRY_AUDIO →
  data/soundry/rounds/<id>/audio/` and `BASELINE_LOCK → apps/lofn-kit/src/stats/baseline.lock.json`;
- `PYTHONIOENCODING=utf-8 python scripts/check_skill_mirror.py` must print **IN SYNC**;
- the manual `CLAUDE.md` / `AGENTS.md` diff is a **checklist line with a human initial**, because no
  script proves it and round one's U-FINAL does not name it.

**Also unresolved and now named:** `BUILD_SPEC.md` contains **three** `## Build order` blocks
(v2:05436…, v2:f6620…, v2:eb884…) plus round one's *Revised build order*, and nothing says which is
authoritative. `U-FINAL` must **delete the three v2 blocks** and leave one. Four build orders in one
spec is how W5 disappeared, and it will happen again.

## C-4 · R22 · TWO SPOTIFY IDENTITIES, AND THE APP CAN PROVE IT TODAY.

Her Open Question 4 is answerable from disk, and the answer is not a choice — it is a contradiction.

- `data/snapshots/socials/latest.json`: `platform: "spotify"`, handle `3egvpGmWFxgYY8XqATui8r`
  ("Local Symmetry"), `works_published: 1`, **`monthly_listeners: 0`**, `status: "dormant"`,
  note *"Exactly one release ('Kintsugi Corrido', 2025 single)"*.
- `data/snapshots/distribution/latest.json` (DistroKid): `platform: "spotify"`,
  **`streams last_365: 1596`**, `last_90: 333`, `last_30: 50`, `last_7: 5`, with a country breakdown.

**1,596 streams in a year cannot come from an artist page with one release and zero monthly
listeners.** The two rows describe two different Spotify identities, which is also why
*"Spotify for Artists is NOT connected to this DistroKid account"* (BS:1396).

**[SPLICE @ PART 2 §7, new M3.3b — a lockout card modelled on M3.3 "Two YouTubes, never summed"]**
> **### M3.3b · Two Spotifys, never joined**
> No plot. Two panels side by side, both at full ink, with **no arithmetic between them** and no
> total. Left: the artist page we can see. Right: the distribution the label reports. Headline,
> generated: *"Two Spotify identities. `3egvpGmWFxgYY8XqATui8r` shows 1 release and 0 monthly
> listeners; DistroKid reports 1,596 streams in 365 days to the same platform. These are not the same
> artist page, and this app will not add them."* Tier `OBSERVATION`, `ONE_PLATFORM` +
> `SCHEMA_DRIFT` flags, `drill` → M5.2 Schema Guard.
> `nextMove`: *"Find the artist page DistroKid is actually delivering to; that page is the canonical
> identity and this card resolves itself."* `PULL_THE_FEED` / `ops.runbook`.
> `forbidden:` *"no cross-identity total, no combined follower count, no 'Spotify reach' figure."*
> `forbidden-strings.spec.ts` gains `"combined spotify"`, `"total spotify"`.

**PRESERVES** her Open Question 4 as *the app's own finding* rather than a question she has to answer
before the build can proceed. Q4 is struck from the OQ list and replaced by
**OQ-DIST-1:** *"Which artist page is DistroKid delivering to? M3.3b will resolve the moment you have
the URL; nothing else blocks on it."*

## C-5 · R23 · SMALLER DROPS FOUND IN THE SWEEP (each one line, each a build failure if unfixed)

1. **`src/sampler.data.json` does not exist and cannot be derived.** Critic 2 (BS:2093) found it;
   **round one does not touch it**, and `design.spec.ts` asserts nuisance balance against pools that
   must be invented. The header says `personas: 5` (an integer); the rows carry **eight** distinct
   persona strings plus 88 nulls. **Ruling:** the persona pool is **generated from the snapshot**
   (`distinct non-null persona` + `none`, currently 9 levels), written to
   `apps/soundry/src/sampler.data.json` by `scripts/build-sampler.mjs`, with `key/mode`, `tempo band`
   and `seed theme` marked `SOURCE: DECLARED BY THE OPERATOR` and shipping **empty** — A3's balance
   table renders `needs-declaration` per empty dimension rather than balancing against a fiction.
   The five-dimension balance rule holds over whichever dimensions are non-empty.
2. **The run-lock instruction contradicts itself three ways** (BS:2094): §8.3.2 gives
   `acquire apps/lofn-kit` *and* says the lock goes in a sidecar; `scripts/run_lock.py:54,145` writes
   `<run_dir>/.RUN_LOCK`; CANONICAL PATHS declares `RUN_LOCK → output/<run-slug>/.RUN_LOCK`. Round one's
   U0 picks `apps/lofn-analytics/.lofn` without reconciling the other two. Given CLAUDE.md's
   eleven-hour loss, **one sentence in §8.3.2**: *"`<run-dir>` is the literal first argument;
   `scripts/run_lock.py` writes `<run-dir>/.RUN_LOCK`. Pass `apps/<app>/.lofn`. The CANONICAL PATHS
   `RUN_LOCK` row is amended to `<run-dir>/.RUN_LOCK` and the `output/<run-slug>/` form is struck."*
3. **`Provenance` still is not unified** (R12 explicitly deferred it and named no owner). It is now
   blocking `JoinEvidence.left.provenance` in SPLICE 1. **Assign it to `L-KIT`** — the union is
   `"api-public" | "authenticated-page-read" | "public-profile-read" | "public-page" |
   "search-snippet" | "derived" | "prose-archive" | "randomised-round"`, eight values, with V8's
   four-level **colour** cap satisfied by a separate `provenanceBand()` mapping eight values onto four
   bands. R12's two binding constraints are met.
4. **A0.6 counts three unread badges (28/3/1) on NightCafe.** Board 8 line 3 grants social agency;
   A0.6 forbids outbound from the card. Both are right. Add one line to A0.6:
   *"Replying is real work and it is not this app's work. → `output/socials/` queue, `vault/SOCIAL_VOICE.md`."*
   so the card routes instead of dead-ending. (Traces to the open session task "Post queued fan replies".)

---

# §D · WHERE I CONTRADICT ROUND ONE

1. **R7 said `data/soundry/**` is categorically excluded from every hosted artefact and stripped
   under `POSTURE=hosted`. I am not reversing it — I am narrowing what it covers.** `panel-audio.mjs`,
   `panel.html` and `data/soundry/rounds/**/audio/**` are **local-posture-only** and are stripped by
   the same rule; `posture-hosted.spec.ts` gains `panel-audio`, `/panel/assets`, and any `.mp3`
   byte in `dist/**` to its absence list. R7 stands; R13 lives entirely inside it.
2. **Round one's SPLICE 9 quotes "a measured between-song variance of 1.369e-4 (MDE ×1.161, not
   ×1.150)". Both figures are wrong.** The reproducible value is `1.3652e-4` and the resulting MDE at
   n=114/m=258 under A2's own formula is **×1.155**. `×1.161` corresponds to no cohort I can construct
   from disk. R15 supersedes that clause. *(This is a correction of an arithmetic claim, not of a
   ruling — the direction of round one's finding was right.)*
3. **Round one's "Revised build order" is treated as superseding BUILD_SPEC's three v2 build-order
   blocks. It does not say so, and by not saying so it deleted unit W5.** R21 restores W5 and requires
   `U-FINAL` to delete the three stale blocks. If the intent was that the v2 blocks survive, then
   there are four build orders and no lane knows which to read.
4. **R6 (a card's `Insight` is not a register entry) is upheld and I am extending its cost:** with
   R20c the register must now carry art and animation entries or the LEDE prints `k = 0` on two of
   three rails. R6 makes the register hand-maintained; that is right, and it means **someone must
   hand-write four more entries before `L-LEDE` closes**. Named, not assumed.
5. **OQ-PANEL-1's "RECOMMENDED DEFAULT" said round 1 pre-registers `instrument: platform` regardless,
   so the panel is a bonus.** With R13 that default is now too conservative in one direction and too
   loose in another: **round 1 should be a `CALIBRATION` round on existing `output/mp3/` audio,
   `instrument: panel`, which can never produce a DECIBAL and therefore costs nothing to be wrong
   about** — and the first *CONFIRM* round then pre-registers `platform`. This contradicts the
   recommended default and preserves its intent (do not make the bench depend on an unproven
   instrument) more cheaply.

---

# §E · WHAT THIS PATCH DOES NOT CLOSE

STILL OPEN **#1** (the Headwinds right rail — the only literal element she drew on the right of
Board 1, still with no card spec, no grid position, no file), **#2** (`GuidanceNote` coverage —
still narrowed to statistics, still no registry, still no `guidance-coverage.spec.ts`, and R13/R19/R20
add a novelty floor chip, an audio transport control, a cadence strip and a 23,000-cell field to the
uncovered pile), **#3** (two filter vocabularies; the Soundry has no chip strip), **#4** (the Sonic
Anvil has no generator — still the largest unbuilt requirement on the whiteboard), **#5** (~12 Soundry
plots with no `deeper` and no loud failure). Those are the other three seats'. I have re-verified all
five against disk and **none of them has moved.**
