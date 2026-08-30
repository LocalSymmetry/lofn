# COORDINATOR FINDINGS — run `2026-08-15_daily_music_the_resolution`
*Written by the coordinator during the wave, before QA. Advisory. Promotion of any of these to a rule is
The Scientist's, not mine.*

---

## ⭐⭐ F1 — A CONTRACT CONFLICT: THE OUTPUT CONTRACT AND BAN 7 GIVE OPPOSITE ORDERS

**The Suno output contract** (`lofn-music/SKILL.md`) requires every MUSIC PROMPT to carry
*"explicit acoustic / no-acoustic declarations."*
**Run-wide ban 7** (from 2026-08-14, render-measured) forbids **any negation inside the style field**,
because Suno reads that field as **positive tokens** — *"no synth pads"* asks for synth pads.

**Both are in force. They cannot both be obeyed literally.** This is exactly L30's class — *a step
contract and doctrine can give opposite orders, and only the run handoff resolves them, so the handoff is
not optional* — and it went undetected through the dispatch of all six pairs. It surfaced only because
the coordinator's verifier swept for negations, which no gate had ever done.

### THE RULING (mine, as coordinator, issued to P01 and P04)
> **State the acoustic axis POSITIVELY** — *"live acoustic kit in the room"*, *"all-electronic palette"*,
> *"every source synthetic"* — **and every literal "no X" moves to EXCLUDE.**
> The contract's *intent* (declare where you sit on the acoustic/electronic axis) is fully preserved;
> only its *grammar* changes.

### ⭐ WHY IT MATTERED MOST WHERE IT WAS FOUND
P01's four prompts carried `no count-in, no reverb`, `no pitch under it at all`, and a `never stop(s)`
family in all four variations. **P01's literal composite element is a butt-cut with a one-beat hole** —
so it was arming the token **"stop"** four times in the one pair whose whole formal event depends on the
generator handling a stop correctly. **The ban's cost is not uniform across a set; it is worst exactly
where the pair's formal risk already lives.**

⚠️ **Unresolved and stated rather than hidden:** whether `never stops` actually arms "stop" in Suno's
tokenizer is **not measurable from text**. It is a plausible mechanism, not a measured one. The repair is
cheap and the downside is nil, so it was ordered anyway — but this is a **belief, not a finding**, until
`lofn-render-audit` tests it under THE BLIND RULE.

---

## ⭐ F2 — THE HEADING CONVENTION WAS REINVENTED A THIRD TIME, AND MY INSTRUMENT ATE IT

L28 (2026-08-05, measured across 6 agents): *an output contract that does not pin its heading convention
will be reinvented by every agent that touches it.* **Third replication, this run:**

| pair | convention shipped |
|---|---|
| P01, P02, P03 | `# V1 — TITLE` |
| **P04** | **`# PACKAGE V1 — "TITLE"`** |

My verifier's mode-detector knew `V1` and `VARIATION 1` and **not** `PACKAGE V1`. It therefore reported
**P04 as zero packages detected** — a file with four complete, correct packages in it.

⛔ **The failure was in my instrument, not in P04's artifact**, and this is the fourth consecutive run in
which every properly re-measured pair claim proved the AGENT right and the discrepancy the
**COORDINATOR's**. *The failure rate lives on the JOIN side.*

⭐ **What saved it was the design, not the luck:** the detector uses **MODE DETECTION** (find whichever
shape a file actually uses, count the hits, take the winner) rather than a priority list, **and it
refuses to score rather than scoring zero** — it printed `variation headings NOT DETECTED -- not scored`.
A pattern-priority detector would have returned `0 of 4 packages` and read identically to *"P04 produced
nothing."* On 2026-08-14 that exact confusion cost a scoring pass. **A detector that cannot tell
"absent" from "unrecognised" is worse than no detector**, and that distinction is now the only reason
this was a two-minute fix.

**The fix applied:** an optional leading word (`PACKAGE|VARIATION|SONG|TRACK`) is now tolerated before
the numeral. **The durable fix is not mine to make** — it is to pin the convention in the step-10
contract, which is a doctrine edit.

---

## F3 — GATE RESULTS, FIRST SWEEP (independent re-measurement from the artifact)
`scripts/verify_daily_music_run.py` → `GATE_REPORT.json`. **276 checks, 269 pass, 7 findings.**

| finding | pairs | class |
|---|---|---|
| negation in style field | P01 ×4, P04 V1 | **REAL — repair ordered** |
| MUSIC PROMPT 842 chars (floor 850) | P02 V3 | **near-miss — grace pass ordered**, target 870–960 |
| MUSIC PROMPT 987 chars (hug ≥985) | P04 V1 | **FLAG, not a fail** (`gates.yaml` boundary-hugging) |

**Everything else passed on first sweep across every scored package:** lyrics field under 5000,
sung lines 70–120 with none at the floor, `[Theme:]` and `[SONG FORM:]` present, ≥1 standalone `*SFX*`
cue, no bare `EMO:AWE`/`EMO:INDIGNATION` tags, **zero digits in any sung line**, zero house-lexicon hits,
zero banned openers, dedications ≤500 chars, and **≥2 case-exact Source-2 phrases in every
generator-facing MUSIC PROMPT.**

⭐ **All three repairs were routed to the pair agents that wrote the files, with their context intact —
not fixed inline by me.** AGENT FIX, DON'T BYPASS.

---

## ⭐⭐ F3b — `strict_end_rhyme()` ERRS IN **BOTH** DIRECTIONS. THE 08-14 FINDING WAS HALF THE DEFECT.

On **2026-08-14** we found the instrument **OVER-reads**: it compares the last three characters of the
final word, so **an identical repeated word counts as a rhyme** — P06 that day cleared a 0.30 floor at a
measured 0.470 **with zero rhymes in it**.

**P02 found the other half today, unprompted.** The same three-character comparison makes it
**UNDER-read**: genuine ABBA rhyme pairs — *did/hid*, *road/owed*, *still/spill* — **score zero**,
because the last three characters differ. P02 measured **0.350–0.525** on the house instrument and
**0.60–0.75** on a rime-based read of the same lyrics, and reported both rather than the flattering one.

> ⭐⭐ **So the metric is not biased, it is NOISY IN BOTH DIRECTIONS — and the two errors do not cancel,
> they just make any single number uninterpretable on its own.** A pair can pass the floor with no rhyme
> (08-14) *and* a pair can look near-failing while rhyming heavily (today). It measures **string-tail
> collision**, which is neither rhyme nor return.

**Third confirmation, same run:** ⭐ **P04 measured 0.42–0.54 on lyrics it knows contain NO end rhyme at
all** — the over-read half, reproduced independently, in the one pair contractually forbidden from
rhyming. It reported the number *and* the reason it is meaningless, rather than banking a floor-pass.

⛔ **Consequence for this run's evidence:** the rhyme floor is a **screen, never a verdict**, and for
**P03 and P04 in particular** — the two pairs that deliberately have little or no end rhyme — the number
carries almost no information. **Held for The Scientist's hand; I am not changing a gate mid-run on my
own instrument's say-so** (L27: an instrument's output alone must never justify changing a correct line).

⭐ Note what made this visible: **P02 reported a number that made its own work look worse, alongside a
better one, and named which was which.** That is the behaviour the self-check contract is supposed to
produce and usually does not.

---

## ⭐⭐ F5 — **THE CREDIT FOLLOWS THE FIELD.** Two pairs ruled OPPOSITE ways on the same player, and both were right.

This run gave every pair the same Source-2 phrase list, including
**`"furiously slapping and popping his bass"`**. Two pairs reached opposite conclusions about **Larry
Graham**, independently, in isolation, without seeing each other's work:

| pair | where the technique sits | ruling | reasoning |
|---|---|---|---|
| **P06** | *"furiously slapping and popping his bass"* is in the **positive STYLE field** | ⭐ **CREDIT OWED — Larry Graham credited** | *"The style field raises slap-and-pop BY NAME. The technique's practitioner is owed a credit."* |
| **P02** | `slap bass` is in the **EXCLUDE field** | ⭐ **CREDIT DECLINED — and the file says why** | *"a refusal, so Larry Graham is deliberately not credited"* |

> ### ⭐⭐ THE RULE THE DISAGREEMENT PRODUCED
> **A technique RAISED in the STYLE field owes its player. The same technique BANNED in EXCLUDE owes
> nobody.** The credit follows **which field the token is in**, not whether the words appear in the file.

This is **2026-08-14's "raised ≠ forbidden" rule run in both directions at once**, and it is the first
time both directions have been demonstrated inside a single run by two agents who could not coordinate.
Last run produced only the negative half (*a Sly Dunbar credit removed because the variation that named
steppers did not ship — a credit that outlives its trigger is a false credit*). **Today produced the
positive half as well**, and the positive half is the dangerous one: ⛔ **P06's first draft of all four
dedications said *"no named figure is asked for, so no player is credited"* — and that was FALSE**, while
reading exactly like diligence.

⭐ **Why the false negative is worse than the false positive.** An over-credit is visible and embarrassing.
**An under-credit reads as a completed check.** P06 would have shipped four dedications, each containing
an explicit sentence asserting that the question had been asked and answered — with the answer wrong.
**This is DEDICATION_STANDARD's own thesis landing on the standard itself: the credit the system forgets
is the one it has already convinced itself it does not owe.**

⭐⭐ **FOUR PAIRS, FOUR INDEPENDENT RULINGS, ZERO COORDINATION — AND THEY ALL AGREE UNDER THE FIELD RULE.**
**P04 is the third independent derivation:** it credits nobody, and states the reason unprompted —
*"no break, riff or figure of sound is raised in any style field, so no player credit is owed; V4's
EXCLUDE actively refuses slap bass. Adding a credit would be the false credit the standard forbids."*
That is P02's reasoning, reached separately, in a different bed, on a different concept.

**Cross-check run over the other four pairs:** P01 raises the same phrase in its style field and **does**
credit Larry Graham (plus Cachao for the descarga form) — consistent. P03 raises only
*"superhumanly funky bass"*, a general descriptor naming no technique and no player, and credits
**Kélétigui Diabaté** for the balafon figure its marimba is copying — consistent, and correctly personal
rather than regional after the agent repaired its own first draft, which had implied he *invented* the
pattern. **All four rulings agree under the field rule. None agreed under "does the phrase appear."**

---

## ⭐⭐ F6 — `mean_words_per_line_ceiling: 7.5` IS GATING A PROPERTY WE MEASURED AS INERT

**Two of six pairs breached it, both for reasons the run itself assigned:**

| pair | measured | assigned form that causes it |
|---|---|---|
| **P02** | 7.75 – 9.05 | **chiasmus** — line 3 restates line 2 in reverse, so lines are structurally long |
| **P05** | **10.21 – 10.80** | **long-lined free verse, slow accretion** — the assignment, verbatim |

Neither is a defect. Both were declared in advance at step 06, both pairs cleared **both return floors**
honestly, and P05 bought its alliteration back to spec at **net zero character cost** rather than padding.

⛔ **And the ceiling itself rests on a finding that already dissolved.** `ANALYSIS_2026-08-08` tested
words-per-line against like rate and recorded the result explicitly:

> *"It does not survive the age control. The gap collapses in two cohorts and **reverses** in the third.
> Short lines are a craft argument (L21), not a measured performance argument. **Do not cite this table
> as evidence for L21.**"*

So we are hard-gating a variable whose only performance evidence **we ourselves retracted**, and it now
fires on two pairs whose long lines are the *point* of their assignment. ⭐ **This is the inverse of the
usual failure: not a gate with no reader, but a reader with no finding behind it.**

**Recommendation (advisory — promotion is The Scientist's):** demote `mean_words_per_line_ceiling` from a
floor-style gate to a **FLAG routed to the Somatic read**, exactly as `music_prompt_hug_ceiling` already
is. ⛔ **Not changed mid-run** — L27: an instrument's output alone must never justify changing a correct
line, and that cuts both ways, so it must not silently justify changing a gate either.

---

## ⭐⭐ F7 — TWO OF MY OWN INSTRUMENTS MEASURED ONE FIELD AND DISAGREED. **FIFTH CONSECUTIVE JOIN-SIDE HIT.**

Building `CREDITS.md`, the extractor **refused to write**, reporting P05 V2's dedication at **502
characters** — over the 500 cap. Three numbers then existed for one field:

| source | measurement |
|---|---|
| the pair agent's own self-check | **497** |
| `verify_daily_music_run.py` | **497** (passed) |
| the credits builder | ⛔ **502 — WRONG** |

**Root cause:** the dedication sits inside a ` ```text ` fenced block. The verifier used `strip_fence()`,
which removes the whole opening fence line. The credits builder used `.strip('`')`, which removes
backtick *characters* only — **leaving the language tag `text\n` in the string and inventing five
characters.**

> ⭐⭐ **The agent was right. My second instrument was wrong. Fifth run running, and the failure rate
> still lives on the JOIN side.**

⛔ **And the direction matters: it FABRICATED A VIOLATION.** It would have blocked a compliant
dedication and sent a correct artifact back for repair — the exact mirror of 2026-08-14's *"my dedication
counter swallowed a trailing trim-priority note → fabricated 611 vs a true 453."* **The same field, the
same class of error, one run later, in a script I wrote today.**

⭐ **What saved it was fail-closed design, for the second time in this run.** The builder **refuses to
write rather than writing a file it does not believe** — the same property that made P04's unrecognised
headings print *"NOT DETECTED, not scored"* instead of *"0 of 4 packages."* **Both of my instrument
failures this run were caught by the same design choice**, and neither would have been caught by care.

**THE FIX IS NOT THE REGEX.** ⛔ **Two implementations of one measurement IS the defect.**
`scripts/build_run_credits.py` now **imports `split_variations` / `grab_block` / `strip_fence` from the
verifier** so exactly one extractor exists. Re-run under the shared implementation, all 24 dedications
reconcile: **max 497, min 379, all ≤500** — and every pair agent's self-reported figure matches exactly.

---

## F4 — WHAT NO TEXTUAL GATE IN THIS RUN CAN SEE
P01 stated it plainly and it is the honest ceiling of this whole exercise:
> *whether the butt-cut survives cannot be gated textually at all.*

The run's Source-3 obligation is **a composite with a visible join**, and every instrument we own reads
text. **Six pairs specified a seam; zero of them can be verified to have one.** That is what
`lofn-render-audit` under THE BLIND RULE exists for, and it is why nothing here is a verdict —
⛔ **passing these gates is a floor, never a judgement.**

⭐ The Deviation Clause is the run's partial answer: each pair named **in advance** where it expects the
generator to disobey and which disobedience it would keep. That converts THE GRAIN LAW's *judge the
result, not the distance from intent* from a post-hoc excuse into a **pre-registered prediction** — the
first time this run structure has been falsifiable before a render rather than after one.
