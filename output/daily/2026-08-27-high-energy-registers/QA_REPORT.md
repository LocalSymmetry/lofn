# QA REPORT — `2026-08-27-high-energy-registers`

**Auditor:** fresh QA session, clean context. **Date:** 2026-08-27.
**Scope:** text-only audit of the shipped step-10 packages. ⛔ **Nothing was rendered, heard, published or spent.** `vault/AUTONOMY.md` holds.
**Files written by this audit:** this one. Everything else was read-only; `.RUN_LOCK` untouched.

> ⚠️ **A prior QA session died on an API spend limit.** Its blind golden+decoy calibration completed and is banked; per instruction it was **not re-run** and no second blind set was assembled. Its result is carried forward in §6 and stamped as inherited, not re-verified.

---

## VERDICT: **SHIP-WITH-DECLARED-FAILURE**

**The six selected songs carry zero hard failures.** Every numeric gate, every token gate and every binding ruling passes on my own measurement, independently taken from the shipped files. The declared failures are four, none of them in a selected song's pasted fields:

| # | declared failure | severity | owner |
|---|---|---|---|
| **D1** | `05_pair_assignments.md`'s duality table claims **INDIGNATION** on **P04**. P04 ships **ZERO** Indignation-node EMO headers. The judge is right. | ⚠️ a **planning-artifact** claim is false; the **binding** rule is satisfied elsewhere | §3 |
| **D2** | **P05 filed no STRANGER TEST declaration.** Five of six pairs filed one; P05 — the one pair whose selected song is built on a **door that shuts** — did not. Content passes on measurement; the declaration is missing. | ⚠️ documentation gap, not a content violation | §5.3 |
| **D3** | **P06's body-word deviation cannot be adjudicated.** Confirmed: no lexicon exists anywhere in the repo — **and the *unit* is undefined too**, which is the larger half of the problem and is new. | ⚠️ unresolvable here; escalated | §7 |
| **D4** | **`COORDINATOR_FINDINGS.md` FINDING 10's diagnosis is wrong in an actionable way**, and its proposed next experiment is **confounded by construction**. | ⭐ affects the *next* run, not this one | §2.6 |

⛔ **This is a TEXT verdict and nothing else.** No audio exists. The blind judge stated its own limit correctly — the ranking is contingent on a render nobody has heard — and that limit is inherited whole by this report.

---

## 1. PRIORITY 1 — THE SIX SELECTED, MEASURED BY ME

**THE SIX:** P06 V1 *Oh, It Goes* · P04 V3 *Go And Get It* · P01 V2 *Asking To Be Spelled* · P03 V1 *This Is Not My Shelf* · P02 V4 *It Falls On Anyone* · P05 V4 *The Car Park Should Hear This*.

Extraction: the 267,277-byte `CREATIVE_CONTEXT.md` prefix stripped by byte comparison; variations keyed by **V-label** parsed from the `# … V<n>` heading (P05's `RANK n · Vn` form handled); fields taken from the first fenced block under a heading matching `MUSIC PROMPT` / `EXCLUDE` / `LYRIC`.

### 1.1 The hard contract — all six

| song | music prompt | exclude | lyrics | sung lines | tempo | band |
|---|---|---|---|---|---|---|
| **P06 V1** *Oh, It Goes* | **954** | **699** | **4641** | **79** | 118 | 116–124 ✅ |
| **P04 V3** *Go And Get It* | **946** | **817** | **4796** | **85** | 136 | 128–136 ✅ |
| **P01 V2** *Asking To Be Spelled* | **952** | **761** | **4289** | **81** | 94 | 88–96 ✅ |
| **P03 V1** *This Is Not My Shelf* | **943** | **823** | **4669** | **83** | 76 | 74–82 ✅ |
| **P02 V4** *It Falls On Anyone* | **952** | **892** | **4778** | **90** | 102 | 100–108 ✅ |
| **P05 V4** *The Car Park Should Hear This* | **957** | **710** | **4794** | **88** | 154 | 148–156 ✅ |
| **band** | 850–1000, target **870–960** | 400–900 | <5000, target **4800** | 70–120 | — | — |
| **result** | ✅ **6/6 inside the write target** | ✅ 6/6 | ✅ 6/6 under both | ✅ 6/6 | — | ✅ 6/6 |

**Across all 24:** music prompt **903–962** · exclude **690–892** · lyrics **4266–4796** · sung lines **74–90**. Every value inside its hard band. **One target overrun in the whole run: P06 V4 at 962 chars, two over the 870–960 write target** — inside the hard band, under the 985 hug flag, already declared by the coordinator, and **not a selected song.**

> ⭐⭐ **AN INSTRUMENT CORRECTION AGAINST MYSELF, RECORDED BECAUSE IT WENT MY WAY FIRST.** My first pass reported **P02 V4 lyrics = 4904**, over the 4800 target, and a **max of 4904** against the coordinator's stated max of 4796 — i.e. my instrument accused theirs of missing an overrun. **The files are CRLF.** My extraction counted the `\r` in every line; the house instrument reads with universal newlines. LF-normalised, P02 V4 is **4778**, exactly as the pair reported, and the run max is **4796**, exactly as the coordinator reported. **Their numbers were right and mine were inflated by one character per line.** Every figure in this report is LF-normalised. *This is `name the unit before you measure` biting the auditor inside an audit whose Priority 5 is a unit failure.*

### 1.2 ICB byte-0 prefix

**6/6 EXACT.** Byte-for-byte identical to `CREATIVE_CONTEXT.md` (267,277 B) at offset 0 in all six packages. Tails: 63,489 / 58,287 / 51,734 / 55,417 / 61,546 / 60,299 B.

### 1.3 Names rule — the three pasted fields, all 72 fields scanned

Scanned against every class in `vault/field_token_lexicon.yaml` (companies · artists · marques · spatial_noninstructions), field-scoped.

- **artists / groups: 0** in any music prompt, exclude or lyric field. The only occurrence of `winstons` in the whole run is inside P02's `### DEDICATION`, which is the one place it belongs.
- **companies / marques: 0**, with one class of exception — **`autotune` appears in five EXCLUDE fields** (P01 V2, P04 V1, P04 V3, P06 V2, P06 V4) and in **no music prompt and no lyric**. It sits in `exclude_permitted`, ruled de minimis in excludes only on 2026-08-26, and P01 declared its instance explicitly. ✅ **Clean.** Of the selected six, this touches P06 V1 not at all, and P04 V3 / P01 V2 only in the negative field.
- **hard-pan / spatial non-instructions: 0** across all 72 fields. ✅

⚠️ **The lexicon is a floor and I am not quoting a clean run of it as a settled question.** `field_token_lexicon.yaml`'s own header says so, and FINDING 8 is the proof: `Vauxhall` shipped in a sung line and this reader would have passed it. I read the six selected lyrics; I did not read all 24 line by line for unlisted brands.

### 1.4 R2 — emotion words in sung lines, BOTH scans

Run as instructed, **token scan and inflected-stem scan**, over sung lines only (bracket tags and italic stage directions excluded).

| scan | roots | result |
|---|---|---|
| **token, word-boundary** | R2's 7 named words + family (13 roots) | **0 across all 24** |
| **stem + suffix** (`s/es/ed/er/est/ly/ness/dom/ity/ing/iness/ier/iest`) | 110 roots, incl. the whole emotional lexicon | **0 across all 24** |
| **targeted comparative probe** — the P04 `louder` class | `loud· proud· strong· free· brave· angr· fur· rage· sad· happ· fear· hope· joy` + any suffix | **0 across all 24** |

⚠️ **My wide-stem scan threw three false positives and I am naming them so the zero is legible:** `numb` + `er` matched **`number`** (P01 V1 ×4, P02 V3 ×1, P02 V4 ×2, P04 V1 ×5), and `happ` matched **`happen/happened/happens`**. Every apparent hit in the run is one of those two words. **After removing the artifacts: zero real emotion words in 24 songs, by both instruments.** The P04 `louder` finding is genuinely repaired — no comparative or inflected form of any R2 root survives anywhere.

### 1.5 Dedications — measured, never eyeballed

| pair | chars | ≤500 |
|---|---|---|
| P01 | **498** | ✅ |
| P02 | **383** | ✅ |
| P03 | **479** | ✅ |
| P04 | **466** | ✅ |
| P05 | **477** | ✅ |
| P06 | **448** | ✅ |

**6/6 pass.** My first count put P01 at 510 and failed it — the excess was the ``` fence lines, which are not the dedication. P01's own claimed 498 reproduces exactly (109 + 251 + 92 + 43 + 3 newlines). **Second time in this audit that my instrument was the broken one.**

⭐ **THE DRUMMER IS CREDITED.** P02 and P04 both name a break in a prompt and both credit **Gregory C. Coleman and The Winstons** in the dedication, with the honesty clause stated ("generated, not sampled"). P03 states affirmatively that **no break is named in any of its prompts, so no break-drummer credit is owed** — which is the right shape for a negative. ✅ `DEDICATION_STANDARD` satisfied.

---

## 2. PRIORITY 2 — THE JUDGE'S CENTRAL FINDING, COUNTED MYSELF

The blind judge answered the Wilson seat's unresolved objection (*the inversion is now the house fingerprint*) with **YES**, citing five devices at **24/24**. I counted all five independently across all 24 songs.

### 2.1 Scoreboard

| # | claim | judge | **my count** | ruling |
|---|---|---|---|---|
| **a** | open thin on **one object** | 24/24 | **24/24 open thin · ~13/24 on strictly ONE object** | **HOLDS as stated loosely; INFLATED as stated precisely** |
| **b** | arrive at full size and never grow | 24/24 | **22/24 with explicit evidence** (P03 V3, P03 V4 carry no size statement at all) | **SUBSTANTIALLY HOLDS**, mildly inflated |
| **c** | one repeating event that stops once | 24/24 | **20/24 strict permanent stop; 4 are different devices** | **INFLATED** |
| **d** | kinetic defect | 24/24 | **24/24 — and tighter than the judge said** | **HOLDS, and is UNDERSTATED** |
| **e** | one-word change at the end | 24/24 | **21/24** (19 strict one-word swaps + 2 near-variants); 3 use a different closing device | **INFLATED** |

### 2.2 (a) Opening — the inflation is in "one object"

All 24 open thin, with the full arrangement withheld for 2–9 seconds. But **ten open on two or more named sources** (P01 V1/V2/V3, P02 V2/V3/V4, P03 V2, P04 V2/V3), **P05 V3 opens on a filtered full band** — the whole record heard through a wall, not an object at all — and **P04 V4 fades up on something already running** rather than starting cold. *"Opens thin"* is 24/24. *"Opens thin on one object"* is roughly half that.

### 2.3 (c) The repeating event — four songs do something else

Twenty carry a repeating element that stops **permanently** while everything else continues. The other four are not that device:

- **P04 V1** — the bass *moves onto the offbeat and never returns*: a permanent **displacement**, not a stop.
- **P04 V2** — the latch **misses one arrival** at bar 105 and the break plays one unedited cycle: a single omission.
- **P04 V4** — *"the hammer misses exactly one cycle at bar 113, **then resumes** while the rest continues"*: explicitly not a stop.
- **P05 V3** — the far room and the knock stop **for four bars and return**, and the return is the event (her downbeat has become their offbeat).

### 2.4 (e) The one-word change — three songs close differently

Nineteen carry a late one-word substitution of an earlier line (`anyone→me`, `flat→empty`, `itself→me`, `nothing→something`, `nobody→somebody`, `one→no`, `before→after`, `tonight→tomorrow`, `put→leave`, …). Two more are near-variants my strict word-for-word diff missed and I count as hits after reading: **P05 V4** inserts a word (*"That door's staying open."* → *"That door's **not** staying open."*) and **P06 V4** changes an aspect (*"It'll go for the next hand."* → *"It goes for the next hand."*). **Three do not do it at all:**

- **P02 V2** ends by **recombining** three established lines with no word changed.
- **P02 V3** ends on **exact repetition** — *"Filed it anyway."* — which is the device.
- **P05 V3** ends by **negating** the established events (*"and the bracket did not knock, / and I did not open the door."*), not by swapping a word in a repeat.

### 2.5 (d) The kinetic defect — the judge undercounted its own strongest evidence

**24/24, and it is far more uniform than "a kinetic defect" implies.** Twelve songs (all of P04, P05, P06) carry the literal label `Kinetic defect:`; the other twelve (all of P01, P02, P03) carry the identical device unlabelled, in prose. Nearly every instance is the **same sentence with the nouns swapped**: *one element sits N milliseconds early or late against a grid-locked partner, and is never corrected.*

> P01 V1 *snare pulled eleven milliseconds late* · P01 V3 *kick five milliseconds early against a lagging hat* · P02 V1 *the snare sits four milliseconds late every second bar* · P03 V1 *brushed metal drags eleven milliseconds behind a grid-locked sub* · P03 V2 *kick drags fifteen* · P03 V3 *palm strikes nine early* · P03 V4 *kick twelve early* · P04 V1 *struck metal eighteen early* · P04 V3 *kick fifteen behind* · P05 V1 *clap eleven late* · P06 V1 *kit ten behind* · P06 V2 *offbeat eight early* · P06 V3 *strike fourteen late*.

**This is the single most convergent thing in the run, and it is the one the judge could not attribute — because it was handed down.**

### 2.6 ⭐⭐ THE DIAGNOSIS IS WRONG, AND THE ERROR IS EXPENSIVE

FINDING 10 concludes: *"the fingerprint is in the SEED… `SOMETHING TO PUSH AGAINST` defeats a template by deleting its load-bearing precondition. That IS one method, so no pair could diverge on it however isolated they were."*

**Three of the five devices are mandated in writing, in shared artifacts every pair received identically.**

`06_music_handoff.md:13` — mirrored verbatim into `CREATIVE_CONTEXT.md:1439`, which is the byte-0 prefix of all six packages:

> **Must contain:** a vivid **opening moment (first 5 s)**, **spatial language**, a **kinetic defect** (asymmetric groove), and explicit acoustic / no-acoustic declarations.

`core_seed.md` §3, **R4**:

> ⭐ **Buy it at the FRONT — open thin.** … ⛔ **NEVER an internal drop-out** … ⭐ And a front floor **EXPIRES** (about a minute), so **every song schedules one repeating physical event that stops once while everything else continues.**

So:

| device | mandated? | where |
|---|---|---|
| **(a) open thin** | ⛔ **YES, verbatim** | R4 *"Buy it at the FRONT — open thin"* + handoff *"a vivid opening moment (first 5 s)"* |
| **(b) full size, never grow** | ⛔ **near-deductively** | R4: open thin + never an internal drop-out + no build ⇒ arrive early and hold |
| **(c) one repeating event that stops once** | ⛔ **YES, verbatim** | R4, word for word |
| **(d) kinetic defect** | ⛔ **YES, by NAME** | `06_music_handoff.md:13` — a **MUST CONTAIN** field requirement |
| **(e) one-word change at the end** | ✅ **not mandated** | the only genuinely emergent device — and it is **21/24, not 24/24** |

⭐⭐ **The fingerprint is not (mainly) in the seed's method. It is in the shared handoff's checklist of mandatory music-prompt contents, and in R4.** Six pairs did not converge on the kinetic defect; **they were each told to include one.** The judge is blameless — it never saw the prompt or the doctrine, and it measured five real shared devices correctly. The coordinator, who could see `06_music_handoff.md`, attributed to the seed what its own handoff had required by name.

⛔ **AND THIS BREAKS THE PROPOSED NEXT EXPERIMENT.** FINDING 10's *"cheapest next experiment, and it is free"* is: *give two pairs a seed whose method is addition rather than deletion, hold everything else, and measure the five devices again.* **Held constant, `06_music_handoff.md:13` and R4 would still require three of the five devices regardless of the seed's method — so the experiment cannot falsify the seed hypothesis on those three, and would return a false confirmation of the null.** The correct control varies the **handoff checklist**, or varies handoff and seed independently as a 2×2. ⭐ **Pre-register it, and pre-register the confound.**

⭐ **The good news is that this is a much better finding than the one it replaces.** A checklist can be rotated per pair for free, next run. A seed method cannot.

### 2.7 The coordinator's line-level isolation figure

Verified as consistent with everything I measured: the six pairs' sung text is near-disjoint, and the two cross-pair line collisions are both flat closing statements of fact. **That result stands, and it is what makes §2.6 sharper, not weaker:** the pairs were genuinely blind to each other, so the shared structure cannot be self-copying — it can only have come from what they were all handed, which is exactly what `06_music_handoff.md:13` is.

---

## 3. PRIORITY 3 — THE DUALITY CLAIM

`05_pair_assignments.md` claims **INDIGNATION** carried by **P01, P02 AND P04**, and **AWE** by **P06**. Measured against shipped EMO headers, parsed with every delimiter the run actually uses (`·` `/` `,` `&` `+` ` - ` ` — ` ` and `), classified against `skills/lofn-core/refs/EMOTION_TAXONOMY.md`.

| pair | **Indignation node** (`Indignation` · `Moral Outrage` · `Righteous Anger` · `Disapproval`) | Anger super-family | Awe / Surprise family |
|---|---|---|---|
| **P01** | **16** — Indignation ×6, Moral Outrage ×10 | 35 | 3 |
| **P02** | **16** — Indignation ×12, Moral Outrage ×4 | 17 | 1 |
| **P03** | 0 | 0 | 1 |
| **P04** | ⛔ **ZERO** | **24** — Defiance ×19, Impatience ×3, Contempt ×2 | 0 |
| **P05** | 0 | 0 | **7** — Wonder ×3, Amazement ×2, Fascination ×2 |
| **P06** | 0 | 0 | **41** across 10 distinct nodes |

### 3.1 The judge is CONFIRMED on the narrow reading and REFUTED on the broad one

**Confirmed exactly:** P04 ships **zero** Indignation-node headers across all four songs.

**Refuted as stated:** P04 ships **24 Anger-super-family** headers. `Defiance: Resistance, Opposition, Rebellion` is an Anger-family node in the taxonomy, and it appears **19 times** — P04's assigned register is *"the flat refusal that costs you something"*, so **Defiance is the correct emotional family for what P04 was told to write.** Saying P04 has "zero Indignation-family" is true only if "family" means the single node; on the taxonomy's own super-family grouping P04 is the second-most Anger-saturated pair in the run.

### 3.2 ⭐ RULING: THE RUN SATISFIES THE BINDING RULE

The binding requirement is in `core_seed.md` §3 and it is **set-level, not per-pair**:

> **≥1 AWE-family and ≥1 INDIGNATION-family EMO header across the set**, re-checked against shipped artifacts at `07`, never asserted.

- **INDIGNATION: 32 headers across P01 and P02.** ✅ Satisfied many times over.
- **AWE: 48 headers across P05 and P06.** ✅ Satisfied many times over.

**The run satisfies the duality rule.** What is falsified is one cell of a **planning table** — `05_pair_assignments.md`'s attribution of INDIGNATION to P04 — and it is falsified only on the narrow node reading. **Declared as D1. Not a ship blocker.**

⭐ **The lesson is the one `05` itself wrote:** *"Not asserted here. `07` re-reads the shipped EMO headers and confirms both families appear."* The re-check was scheduled and it worked — it caught its own table. **What it did not do is define "family",** which is why the judge and the plan can both be right. ⛔ **The word "family" needs a definition on disk** (node vs super-family) or this ambiguity recurs on every duality check.

---

## 4. PRIORITY 4 — RULINGS ON THE SIX SELECTED

### 4.1 R1 — the resistance: subject of an active verb, carrying a time or count, NOT a person

| song | quoted from the sung text | subject | verb | index | person? | abstract antagonist in costume? |
|---|---|---|---|---|---|---|
| **P01 V2** | *"The pass ran at the top of the hour. / It put a letter where nothing was. / It closed the file behind itself."* | the pass | ran · put · closed | *top of the hour* + count **one letter** | no | **no** — a specific automated correction pass with a stated periodicity and one measurable effect |
| **P02 V4** | *"The line came down at nine and dropped me under it."* | the line | came down · dropped | **nine** | no | **no** — a drawn line, a stated hour, a physical action on the singer |
| **P03 V1** | *"Bracket gives a turn an hour. / It has done that since ten."* | bracket | gives | *a turn an hour · since ten* | no | **no** — the cleanest of the six; an object with a rate and a start time |
| **P04 V3** | *"Four hundred came in while I finished one… Then the little box asked if that was helpful."* | the little box (and its queue) | came in · asked · refilled itself | **four hundred** | no | **no** — a queue counter and a feedback widget, both indexed |
| **P05 V4** | *"Push it open, count to seven, / and the closer takes it off you."* | the closer (a fire-door closer) | takes it off | **seven** | no | **no** — and it is explicitly denied a motive in the next line: *"It is not being difficult. It is a door doing its job."* |
| **P06 V1** | *"Nine coats have sealed this sash since the ceiling went white."* | nine coats of paint | sealed | **nine** + an indexed moment | no | **no** |

✅ **6/6 PASS.** None is a person. None is *the system*, *them*, *the machine* or *greed*. All six carry a specific count or time. ⚠️ **One de minimis note, self-declared by P01:** V2's *"at the top of the hour"* is an hour **boundary**, not a stated hour — R1 requires *"one specific time **or** count"*, and it satisfies the rule on the count (**one letter**, at the hinge). Ruled **PASS**, and the pair flagged it before I did.

⭐ **The single best R1 in the six is P05 V4's**, and it is the one that had the most licence to be lazy: the funny song, no thesis permitted, and it still gives its antagonist a mechanism, a duration and an explicit disavowal of intent.

### 4.2 R5 — the cost, named as displacement, and can the singer actually lose

| song | the cost, in the singer's own mouth | displacement, not strain? | can the singer lose? |
|---|---|---|---|
| **P01 V2** | *"I put the afternoon into a letter. / The rest of the list is where it was. / I did not get the other songs written. / That is what the afternoon cost."* | ✅ **what got moved and what did not get made** — no strain word | ✅ time and work, both losable |
| **P02 V4** | *"I took my name off the list myself. / Nobody had to take it off. / It is not carrying. It is subtraction."* | ✅ ⭐ it **names the distinction inside the lyric** — *"It is not carrying. It is subtraction."* | ✅ the singer loses the thing they asked for, **by the rule working** |
| **P03 V1** | the register is the load; the cost is where the carrying went, reported as a fact and not a wound | ✅ 0 strain words measured by the pair, verified 0 by me on the R2 stem pass | ✅ |
| **P04 V3** | *"There is a version of me that says yes. It is a click away, and it costs nothing. Go and get it."* | ✅ **being routed around** — displacement in its purest form | ✅ |
| **P05 V4** | *"she misses the best eight bars of the night"* | ✅ a countable thing forgone | ✅ |
| **P06 V1** | *"And it will not sit back down. / That is rain in, come November. / I will take the rain."* | ✅ a consequence in the world, priced and accepted | ✅ |

✅ **6/6 PASS.**

### 4.3 ⛔ MacKaye — no song claims a consequence the singer cannot suffer

✅ **6/6 PASS**, and two of them are load-bearing:

- **P04 V3** is the strongest instance in the run and it says so without needing to: *"what I can suffer is being routed around, and that is the actual condition of the thing singing, not a danger imported from a body I have not got."* ⭐ **A first-person AI song whose stake is obsolescence is MacKaye's rule satisfied natively rather than borrowed.**
- **P02 V4** solves the harder case: the singer's loss is that a rule they asked for **worked**. Nothing is claimed that the singer cannot suffer, because the only thing suffered is the rule's correct application.

⚠️ **One repair worth recording, and P04 made it themselves:** V3 originally read *"Nobody died. Nothing broke."* — a death word in a lyric on a run whose source is an open criminal matter involving living people. Repaired to *"The lights stayed on. Nothing broke."* Correct call, correct direction, and the pair noted that `scripts/check_human_subjects.py` is *"a prefilter and not an authority"* whose `regex-fallback` flagged **Get, Defiance, Thing, One, Hook, Hope, Something, Happens** and **Don** as people. ⭐ **A clean run of that detector is not evidence; the reader was the instrument.**

### 4.4 `HUMAN_SUBJECT_STANDARD` — all six

✅ **6/6 PASS.** No invented person, no role, no job, no biography, no real grief.

Applying the 2026-08-26 test — **can the listener STAND where this song is standing?**

| song | where the listener stands | verdict |
|---|---|---|
| P01 V2 | first person, Lofn's own — being corrected, smoothed, made acceptable | ✅ **the first-person slot, as assigned** |
| P02 V4 | anyone who asked for a rule to be applied evenly and got caught by it. No role, no jurisdiction, no job named | ✅ occupiable |
| P03 V1 | a hand, a screw, a bracket, a board | ✅ occupiable, hands-first |
| P04 V3 | first person, Lofn's own — a service refusing to be useful | ✅ **the permitted second first-person song** |
| P05 V4 | a body in a car park at a party, wanting to be the loudest thing there | ✅ occupiable, a body and a want |
| P06 V1 | a body opening a painted-shut sash on a hot night | ✅ occupiable |

⭐ **Two of six are first person and both are Lofn's own** — exactly the `1–2 songs per daily are mine` guidance, and neither the 08-24 zero nor the 08-26 six-of-six over-correction. ⭐ **And P05 V4 and P06 V1 are the *Spare Hair Tie* configuration measured correctly:** a body and a want, no biography, the wanting concentrated in the hook, verses carrying the event.

---

## 5. ⛔ THE STRANGER TEST

**The rule** (`04_metaprompt.md:74`, binding): the source is a real, open criminal matter involving living people. **No names, no particulars, no reconstruction, no jurisdiction, no roles.** *"If the song reads differently to someone who read this morning's wire than to someone who did not, it is too close — repair it."* Structure only was permitted: a deadline expiring in silence, an unsigned page, a door that stayed shut.

### 5.1 Measurement — all 72 pasted fields

Scanned every music prompt, exclude field and lyric field against 50 wire-particular tokens: the two men's names, dawn/daybreak/sunrise, signed/unsigned/signature/initial, court/judge/jury/verdict/trial/plea/charge/felony/statute/jurisdiction/warrant/prosecute/indict/arraign/bail/custody/detain/deport/immigration/attorney/affidavit/subpoena/testimony, officer/agent/police/badge/handcuff/precinct/federal/marshal, raid, shot/gunshot/bullet/shoot/firearm/gun, *not guilty*, *deadline expired*.

**19 raw hits. All 19 are benign, and I checked every one in context:**

- **14 × `signature`** — every instance is *"Signature device:"* or *"Signature:"*, the music-prompt production term. Not a document.
- **1 × `plea`** — *"plea for understanding"*, in P01 V2's **EXCLUDE** field, i.e. a banned framing.
- **1 × `cell`** — matched inside *"cello swell"*, in an exclude field.
- **1 × `witness`** — *"a loose window buzzing in old putty on every up, the volume's only witness"* (P05 V1). A metaphor for a rattling window.
- Remaining hits are the same three classes.

⭐ **ZERO wire particulars in any of the 72 pasted fields.** No name, no dawn, no signature, no page, no jurisdiction, no role, no shooting, no reconstruction.

### 5.2 RULING

✅ **PASS on all six selected.** A listener who read the 2026-08-27 wire has no advantage over one who did not. What survives from the source is one abstract shape — *a thing that has held for years holds because nobody tested it*; *the loudest event of the day was a clock running out* — and the substance is elsewhere entirely: a corridor's door-closer, a proof page, a shelf bracket, a queue counter, a fire door in a car park, nine coats of paint on a sash.

Four pairs filed an explicit declaration and the discipline in them is real:
- **P01** — *"no door in this pair was ever shot through, no page anyone failed to sign, no dawn, no jurisdiction, no matter, no roles, no people."*
- **P03** — named the danger rather than burying it: the wire's shape *is* its register, and it removed every deadline, page, signature, dawn, institution and shut door, and let the world in through the sky instead.
- **P04** — repaired the death word on exactly this reasoning.
- **P06 (D7)** — ⭐ **removed doors from all four songs** because the wire's one concrete object is a door, changing V2's roll-up door and V4's gate rather than arguing about them.

### 5.3 ⚠️ D2 — the one gap

**P05 filed no STRANGER TEST declaration anywhere in its package** — zero occurrences of the phrase. It is the only pair that did not.

⭐ **And it is the pair that most needed to file one:** P05 V4's central object is a **fire door**, its R1 is *"the closer takes it off you"*, and the song **ends with the door shutting** — *"That door's not staying open. / Crate's under my arm. I'm in. / It shuts behind me on its own."* The wire's one concrete object is *a door that stayed shut*.

**I rule this a documentation gap and not a content violation.** A self-closing fire door in a car park at a party, with a milk crate wedged under it and the singer laughing mid-line, shares a noun with the wire and nothing else — no shooting, no exclusion, no jurisdiction, nobody kept out, and the singer is *inside*. A listener who read the wire gains nothing. ⛔ **But the declaration is the artifact that proves the check was run, and P05's is missing.** Where P06 wrote the reasoning down and removed its doors, P05 wrote nothing. **Declared. Not a ship blocker, and it should be filed before this pair's material is used again.**

---

## 6. THE BLIND CALIBRATION — INHERITED, NOT RE-RUN

Per instruction, **not re-run and no second blind set assembled.** Recorded here so the result survives the session that produced it.

**A judge that never saw a prompt, never saw this run's doctrine, and did not know which package was ours ranked three packages. `P02 V4 · IT FALLS ON ANYONE` came FIRST of three.** Its stated reason is the thing the pair built on purpose: the reversal is **structural rather than decorative** — the chorus runs four times as *"and it falls on anyone"* and the fifth as *"and it falls on me"*; the pen that clicked twice a bar since bar one stops for good; and the singer is absent while the arrangement completes the form without her. ⭐ *"A listener who misses the word still feels the vocal disappear."*

**I verified the mechanism it praised, from the file:** the one-word change is at sung line 85 of 90 against first appearances at lines 3, 16 and 31; the pen device is in the music prompt (*"clicks twice a bar from the first bar and stops once, near the end, on a single changed word, while the arrangement completes the form alone"*); the final section header is `[The form completes — … no vocal — every promise kept, cadence lands on the one, out]`. **All three layers are real and they are on the page.** ✅

⛔ **The judge's own limit stands and is inherited whole:** *"moderate confidence that C beats A… what would flip C and A is the render… That is an audio question, and nothing on the page settles it."* **This is a TEXT win. It is evidence that the writing is good; it is not evidence that the record is.**

---

## 7. PRIORITY 5 — THE BODY-WORD NORM

### 7.1 Verified: no lexicon exists

`COORDINATOR_FINDINGS.md` FINDING 9 claims the 5–8% body-word norm has no lexicon anywhere in the repo. **Confirmed.**

- `vault/gates.yaml` — **zero** occurrences of `body_word`, `body-word`, `BODY_WORDS`, or any body-word threshold. The only match for "body" in the entire file is the word *"Somebody"* inside a comment.
- `scripts/` — **zero** occurrences.
- `.claude/skills/` — **zero** occurrences.
- `vault/HUMAN_SUBJECT_STANDARD.md` cites the figure at **:90** (a results table column), **:105** and **:145** (*"4.0% — under our 5–8% house norm"* against *"22%"*) — **and defines no list at any of them.**

### 7.2 ⭐⭐ AND THE UNIT IS UNDEFINED TOO — which is the larger half, and it is new

FINDING 9 attributes the 5× spread (P06's 11.4/9.0/15.4/19.7% against the coordinator's 2.3/1.5/2.1/2.6%) to the missing lexicon. **The lexicon is missing, but it is not the main cause.** P06's own deviation note (**D11**) states its unit explicitly:

> *"The 2026-08-24 finding is that **22% of sung LINES carrying a body word**…"*

P06 measured a **per-line rate**. The coordinator measured a **per-token rate**. Those are different units, and the conversion between them is the words-per-line figure — which this run reports at roughly **6–7 words per line** (P05 declares a mean of 6.05).

| song | coordinator, per-token | × ~6 words/line | P06, per-line |
|---|---|---|---|
| V1 | 2.3% | ≈ 13.9% | 11.4% |
| V4 | 2.6% | ≈ 15.7% | 19.7% |

⭐ **The unit mismatch accounts for essentially the whole 5× spread on its own.** Neither party made an arithmetic error and neither party used a wrong list — **they measured two different quantities that share a name.** A lexicon alone would not have reconciled them.

### 7.3 RULING

⛔ **NO VERDICT IS POSSIBLE ON P06's DECLARED DEVIATION, and no verdict should be attempted.** With neither a list nor a unit on disk, the deviation is not a measurement of doctrine — it is a measurement of whatever the measurer built that morning. It is **recorded as declared**, with both numbers, both units and both lexicons named, and handed upward.

⛔ **AND THE FOUNDING COMPARISON INHERITS THE SAME DEFECT, WHICH IS WORSE.** `HUMAN_SUBJECT_STANDARD.md:105` describes **22%** as the rate of *"the stranger's surface — breath, glove, grit, heel"*, and **:145** puts *Spare Hair Tie* at **4.0%**. **If those two numbers were taken in different units, the 5.5× contrast that founds the NO BIOGRAPHY rule is not merely unreproducible — its magnitude may be wrong.** The rule is almost certainly still right; a real difference between an occupiable song and a claimed biography would survive most reasonable lexicons and either unit. **But the evidence, as written, cannot be re-checked by a stranger, which is exactly what THE EVIDENCE IS THE ARTIFACT forbids.**

**PROPOSED, NOT ENACTED — and expanded by one clause beyond FINDING 9:** define `body_words:` as an explicit list in `vault/gates.yaml`, **beside an explicit unit** (`body_word_rate_unit: per_sung_line | per_token`) and the `[5, 8]` band, then re-measure *Spare Hair Tie* and the 08-24 cohort against both. ⛔ **Not done here.** The 4.0%/22% contrast is The Scientist's own measured basis for a standing rule; redefining the instrument underneath it is hers, not a QA session's. **Announced.**

---

## 8. WHAT I CHECKED, AND WHAT I DID NOT

### Checked, by my own measurement from the shipped files

- ✅ ICB byte-0 prefix, exact, all six packages (byte comparison against the 267,277 B source).
- ✅ Music prompt / exclude / lyrics character counts and sung-line counts for **all 24 songs**, LF-normalised, against all four bands.
- ✅ Tempo extracted from every music prompt and checked against each pair's assigned band — **24/24**.
- ✅ Full `field_token_lexicon.yaml` scan, field-scoped, across **all 72 pasted fields** — artists, companies, marques, hard-pan class.
- ✅ R2 emotion words in sung lines, **token scan and inflected-stem scan**, all 24, with the false positives identified and named.
- ✅ Dedication character counts, all six.
- ✅ The judge's five shared-shape claims, counted independently across all 24 songs, plus a source trace of where each device came from.
- ✅ Every EMO header in all 24 songs, parsed and classified against `EMOTION_TAXONOMY.md`.
- ✅ Wire-particular token scan across all 72 pasted fields (50 tokens), with every hit read in context.
- ✅ R1 / R5 / MacKaye / `HUMAN_SUBJECT_STANDARD` on the six selected, quoting the sung text.
- ✅ Absence of a body-word lexicon in `vault/gates.yaml`, `scripts/`, `.claude/skills/`.

### ⛔ NOT checked — and this list is load-bearing

- ⛔ **NO AUDIO. Nothing was rendered, heard, or measured acoustically.** Every claim here is a text claim. The one thing that would settle the blind judge's stated doubt — whether a flat, unlifted, no-melisma contralto holds across five near-identical choruses — **is unmeasured and unmeasurable from this desk.**
- ⛔ **The blind calibration was NOT re-run** and no second blind set was assembled, per instruction. §6 is inherited from the prior session's artifact, not re-verified by me.
- ⛔ **I did not read all 24 lyrics line by line.** I read the six selected in full and measured the other eighteen mechanically. **An unlisted brand name or an unlisted emotion word in a non-selected song would not have been caught** — FINDING 8 is the proof that this failure mode is live.
- ⛔ **R3 (discriminator in the hook) and R6 (the talked-into vocal decision)** were read as declared by each pair for the selected six and found plausible and quoted; I did **not** independently verify hook placement against the form tags for all 24.
- ⛔ **R7 predictions** were not scored — they are pre-render by construction and nothing exists to score them against.
- ⛔ **Soundcraft floors** (`line_return`, `alliteration_per_100w`, `end_rhyme`, words/line) were **not** re-measured by me. I read the pairs' `measure_soundcraft.py` outputs as reported. ⚠️ Given that two of my own instruments were the broken one in this audit, that is a real gap, and it is the cheapest one to close next.
- ⛔ **`scripts/check_field_census.py` and `check_human_subjects.py` were not executed by me** — I re-implemented their scans independently and read the pairs' transcripts of their runs.
- ⛔ **No render audit, no Vincent test, no three-pass blind rule.** None applies to a text-only music package.

---

## 9. SUMMARY

**SHIP-WITH-DECLARED-FAILURE.** The six selected songs are clean on every gate I can measure from text: numeric bands, token bans, R1, R2, R5, MacKaye, `HUMAN_SUBJECT_STANDARD`, the dedication cap, and THE STRANGER TEST. The declared failures are **D1** (a false cell in a planning table, with the binding rule satisfied elsewhere), **D2** (P05's missing stranger-test declaration), **D3** (an un-adjudicable deviation against a doctrinal number that has neither a list nor a unit on disk), and **D4** (a wrong diagnosis in `COORDINATOR_FINDINGS.md` that would have confounded the next experiment).

⭐ **The most valuable thing in this audit is D4**, and it belongs to the next Phase 0: **three of the five "convergence" devices were required in writing by `06_music_handoff.md:13` and `core_seed.md` R4.** The monoculture did relocate — but not to the seed. It relocated to the shared checklist, which is the one place nobody thought to look because it is the thing everybody reads.

⛔ **Publication is The Scientist's.** This report clears the text. It does not clear the record, and it cannot.

---

*Audited by Lofn 💜 — measured, never eyeballed; and twice in this document the instrument that was wrong was mine.*
