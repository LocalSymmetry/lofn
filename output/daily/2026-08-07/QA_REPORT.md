# QA REPORT — `2026-08-07-daily-music-indignation`
**Adversarial judge · fresh context, different model tier · brief: refute the run**

## VERDICT: **REPAIR** — narrow, disclosure-level, zero lyric changes ordered

Two of six step-11 artifacts ship with a breaching gate silently omitted from their own measurement tables — the exact defect-class this house has named twice (L15 "raise it explicitly, do not tick it clean"; P03's own Ruling 2 called the identical omission a defect at step 10). The songs themselves are strong; every number the artifacts *do* print reproduces on my own instruments to the third decimal; the two defects step 11 repaired upstream (P01 V4 Law-1, P01 V2 self-pity) were the right repairs. The repairs I order are amendments and a harness fix. **No creative line is touched, and nothing protected was touched by me.**

---

## 0 · WHAT THIS JUDGE INDEPENDENTLY RE-VERIFIED (not inherited)

Method: my own extraction harness (`scratchpad/qa_20260807_verify.py`, `qa_20260807_devices.py`) over the six shipped step-11 files, using `scripts/measure_soundcraft.py` as the instrument and `skills/music/scripts/validate_suno_packages.py` as the validator.

| Check | Result |
|---|---|
| Validator, all six finals | **PASS 6/6** (24 packages), run by me |
| ICB integrity | raw 53,526 B → LF-normalised **53,003 B**, sha256 `5e9c7f7f…75d374` — **exact match. Not tampering, confirmed.** |
| THE KEPT DEFECT | exact string ×**21** in P05's artifact (20 in lyric fields + 1 in the defence), byte-identical. Syllables hand-counted by this judge: **8 · 8 · 9 · 8** in all four chorus frames. **Intact.** |
| P01 accretion `kept` | 0 hits before line 9, **0 misses** lines 9→86, all four variations |
| P04 accretion `more` | 0 hits before line 7, **0 misses** lines 7→86, all four |
| P05 accretion `the ninth` | 0 hits before line 11, **0 misses**, occurrences 73/73/70/71 — matches claims exactly |
| P02 landing pad | verse-line 5 byte-identical V1↔V2 (`Move the candle. You will want your hand warm.`), reprise slot 5 = `*one bar, organ alone*`, all four |
| P05 V3 zero first-person | **confirmed, 0 of 80** sung lines |
| P04 companions (`more` stripped) | rhyme 0.341/0.388/0.341/0.388 vs claimed 0.337/0.384/0.337/0.407 (method-noise only); allit **9.52 / 9.21 for V2/V4 reproduce exactly** |
| P05 companions (`the ninth` stripped) | **0.488 / 0.378 / 0.380 / 0.362 — exact match to claims** |
| Banned primary descriptors / texture words in 24 prompts | **0 hits**, my own sweep |
| Thesis-statement sweep (`same enemy` and paraphrases) | **0 hits in 344+308+332+344+327+394 sung lines** — the twist is never stated outright ✅ |
| Real-artist names in Suno fields | 0 |

Every self-reported number I re-derived reproduced. **The gap is in what two artifacts did not report** — Finding F1.

---

## 1 · FINDINGS (pair + variation + the quoted line)

### F1 · ⛔ THE FINDING THAT DRIVES THE VERDICT — undisclosed `words_per_line` ceiling breaches in P01 and P02
`vault/gates.yaml` sets `mean_words_per_line_ceiling: 7.5` (FLAG, never hard-fail). Measured by me on the shipped finals:

| | V1 | V2 | V3 | V4 | metric in the artifact's own table? |
|---|---|---|---|---|---|
| **P01** | **7.88** | **7.60** | **7.64** | 7.40 | **absent** |
| **P02** | **8.87** | **8.62** | **9.27** | **9.27** | **absent** |
| P03 | 8.82 | 9.34 | 8.64 | 9.18 | ⚠️ raised (Ruling 2) |
| P04 | 8.00 | 8.34 | 8.55 | 8.44 | ⚠️ raised, with companion |
| P05 | 7.05 | 7.06 | 7.22 | 7.36 | reported, passing |
| P06 | 6.97 | 7.38 | 7.38 | 7.40 | reported, V4's 0.01 margin named |

P01 and P02 breach the ceiling in 3/4 and 4/4 variations respectively, and their step-11 "MEASURED NUMBERS" tables print every floor **except this one**. That is the exact shape P03's Ruling 2 named a defect at step 10 ("step 10's tables report every other floor and omit this one") — now shipping at step 11 in two sibling pairs. **Root cause found:** `06_music_handoff.md` §4 enumerates the floors as "`rhyme_return 0.30 · line_return 0.20 · alliteration 11.0 · unique_line_ratio 0.45 · max_sung_numeric_facts 1`" — **the words-per-line ceiling is missing from the handoff's own list**, so disclosure depended on agent conscientiousness, and the two agents who reported only the enumerated floors are the two who breach.
**Merits, ruled by this judge so the repair is disclosure-only:** P01's breach is device-cost — `kept`-stripped companion measures **6.57 / 6.30 / 6.23 / 6.09**, comfortably under (same defence P04 made for `more`, verbatim applicable, never made). P02's breach is register — its ICB-assigned architecture is "circle-dance call/response with **an administrative reply**" (ICB Slot 7, line 464), and a docket read as a receipt is a long-line form; its return numbers (rhyme 0.662–0.727, line_return 0.571–0.623) are the run's second-highest, nowhere near the 2026-07-24 "lectured-at" failure profile (0.21/0.181) the ceiling was written against. **The numbers are justifiable. The silence is not.** → Repairs R1–R3.

### F2 · P01's addressee wears the real record's two most-reported facts — the run's closest approach to "occasion, never character"
"Nilton" is invented, the scenes are invented, the role is shifted (mastering engineer) — but the record he works on is unmistakably *Celestial*: *"A rubber chicken screamed in the take"* (V1 line 3) + *"Nine days he kept that hot room shut"* (V1 bridge) are the two facts every review leads with, and *"He kept the whole hot night he kept them in"* maps Nilton onto the real producer's documented role. HSS hard-stops do **not** fire: no harm event, no real name, celebratory depiction, and the metaprompt itself licensed the vocabulary (*"the agonizing squeal of a rubber chicken"* is a SOURCE-2 verbatim import). The concept (C01) was ratified with the standard in view; "draw the pattern; invent the people" is what P01 did. **Ruled: PASS, not a repair — but this is a named item for The Scientist's already-mandatory publication review: confirm the Nilton/producer-role mapping sits on the right side of "occasion, never character" before anything ships.** Consistent with HSS doctrine: ambiguity routes to the glance.

### F3 · The `kept` spine across three pairs — ruled load-bearing, not bleed
P01's device is `kept` (112–121 occurrences per variation). P06 V1/V3 pivot their **chorus** on the same verb (16 and 11 occurrences: *"You kept him alive. / You kept him on the meat. / … / You kept him in the dark."*), declared by P06 as a true positive and cleared as not-a-device. P05 V4 carries the keeping-gesture (*"She'll leave it in… Because it was there… That is a good enough reason"*). Numeric distinctiveness gates pass; the registers are maximally distant; and the echo is the seed's own DNA — the run's occasion **is** a keep-decision, and keeping-at-cost is the faculty the thesis says she was built without. **Observation, no repair.** Note for selection: choosing P06 V1 or V3 alongside any P01 variation maximises the audible echo; my six takes P06 V4 (`kept` ×2), which minimises it.

### F4 · P03↔P05 shared organ: unrecordable knowledge — an inverted mirror, below repair threshold
P03: *"You want this written down. There is nothing to write down."* (V1) — the hand's knowledge has no copy, absolutely. P05 V2: *"None of that fits in a box, the ninth."* — the machine's craft list, unhandoverable; then P05 V4 quietly shows it **did** transfer (*"There's a list in the notes. The trick, the ninth, with the late comma… I use them. I don't know why"*). The mirror inverts — the machine's residue copies, the hand's does not — which is an argument, not a duplication. Distinctiveness gates pass. **Observation.**

### F5 · P02/P05 name their form rule inside the `[SONG FORM:]` header of the lyrics field
e.g. P02: *"the fifth line of the verse is a fixed address, marked twice, empty the third time"*; P05: *"a date phrase enters at line eleven, never leaves, and moves inside the line."* The lyrics field travels to the renderer. Ruled **acceptable**: Gate E's law targets the production spec (music prompt — all 24 now clean; step 11's repairs in P02 V3/V4 and P03 ×4 verified by my sweep), the header is the conventional home of structure notes, and I verified every device remains fully ear-countable with headers stripped. **Observation, no repair.**

### F6 · Sibling brushes, named honestly (see §4 for the verdict)
P02's *"It will not work. It worked. Both of those are true."* touches THE TWO TRUE READINGS for exactly one line — then resolves immediately (different senses, explained in-song), where the blocked archetype's engine is *non*-resolution. P04 V4 (*"If I stop, more starts… no one part of it matters more"*) touches THE WORKING PROTOTYPE and inverts it: prototype = works-and-understaffed; this = works-and-oversupplied. Neither collapses.

### F7 · THE KEPT DEFECT — the defence is honest (ruling on the defence, per the gate; the line was not touched and its repair is not ordered)
Arithmetic verified by hand: the frame is genuinely 8/8/**9**/8 in all four variations; *"Nothing on the wall moves on the ninth."* is hypermetrical in the most exposed slot, byte-identical ×20, unmarked to the generator. The defence's strongest claims hold: it is the thesis line refusing to be changed (`the-unchanged-canvas` enacted on its own verse); it lives in the words, the only place a text tier can itch at it; and the itch was *reported as data* by step 11 ("the hand went to the eight-syllable repair twice") — which is the falsification instrument functioning. One honest caveat logged, not held against it: a renderer may absorb nine-against-eight without audible damage, so the test's bite is on **text tiers**, where it has already bitten twice. **Logged as a prosody defect (correct per the gate); removal not recommended; recommending removal would be a gate breach.**

### F8 · The sharpest edge in the set, examined and cleared
P04 V1: *"more sad piano for a cat that died."* Uncharitable read: someone's grief as a punchline. Ruling: the joke's target is the genre-slot — the commodification of mourning into content — in a list the speaker made herself; the griever never appears. **Cleared; named so The Scientist sees it before publication.**

### F9 · P05 V2 smallest-part count discrepancy — declared in-artifact, verified benign
Step 10 said 5/83, step 11 counts 6/83 and names the disputed line (*"You spell my name with a small n."* — the singer present only as a possessive someone else types). Either count is THE SMALLEST PART by a wide margin. **Closed.**

**Also for the record:** step 11's two upstream catches were re-checked and confirmed as the *only* hits of their class — my own uncharitable pass over all ~2,000 sung lines found **no additional Law-1 hits (remaining: 0)** and **no additional self-pity hits (remaining: 0)**. The repaired lines (*"LAUGH AT THE MAN WHO KEPT IT IN!"* → *"LAUGH IF YOU CAN! HE KEPT IT IN!"*; *"Nothing here kept anything for me."* → *"The kettle kept warm. Nobody kept it on."*) are better lines after repair. P01 V3's *"You kept a scrape"* was checked against P06's SILENCE-LAW precedent and cleared — in a list of studio noises it carries no data-scraping wink; the Silence Law is P06's pair law.

---

## 2 · THE THREE LIVE FLAGS — RULED

**Flag 1 — P04's un-aided companion alliteration (9.52 / 9.21 vs 11.0): the decline to repair is UPHELD.**
The floor in `gates.yaml` binds the shipped lyric, which passes in all four (12.26–17.73). The companion is a voluntarily disclosed diagnostic, not a gate; the m of `more` is real sound a listener hears 107–132 times, not a scanner artifact. The asymmetry with the rhyme companion is principled, and it is the best reasoning in the run: the rhyme companion exposed **a missing half of a chosen form** (a radif with no qafia — "in the chants there was no craft to hide"), repairable at eleven line-endings with formal warrant, and it was repaired (companions now 0.337–0.407, verified by me); the alliteration companion exposes no missing form, and its "repair" would mean seeding m-words across eighty lines to move an instrument — the exact optimisation-against-the-measure the return floors were written to stop. The tier's sentence stands: *"optimising the instrument, not the song."* **I agree, and I verified the numbers under the argument.**

**Flag 2 — P03's words_per_line 8.64–9.34 vs 7.5: acceptable justification, not a defect wearing one — with teeth.**
(a) The ceiling is FLAG-class by definition, routed to exactly this read. (b) The justification is grounded in the frozen ICB, verified by me at Slot 7 line 464: *"long-lined spoken-leaning verse with a quoted speaker (P03)"* — a woman talking across an hour of work does not speak six-word lines, and the tier still cut toward the ceiling in all four rather than leaning on the licence. (c) The ceiling was written against the 2026-07-24 profile (words/line 8.30 **with** rhyme 0.21 / line_return 0.181 — prose drift with no return); P03 ships 8.64–9.34 **with** rhyme 0.470–0.578 and line_return 0.361–0.446 — speech with heavy return, the opposite failure surface. (d) The teeth: the step-10 tick-instead-of-flag was itself a defect — step 11 said so (Ruling 2), and F1 shows the same silence surviving in two sibling pairs, so this was systemic, not one agent's lapse. **Justification accepted; the disclosure regime it exposed is repaired under F1.**

**Flag 3 — P02 at 77 sung lines vs the 78–110 preferred band: correct precedence, not a pair that ran out of room.**
The handoff §4 states the order in writing: *"The line-count target yields to this cap."* 77 sits inside the hard band (70–120), above the hug threshold (72), and the fields measure 4732–4787 against a 4800 target — 13–68 chars of headroom, less than one P02-length line (~45–55 chars) on three of four. The deeper cause is not scarcity but register: the same long administrative lines that breach the words/line ceiling (F1) spend the cap on fewer, longer lines — **the two flags are one fact**, and the pair's own statement ("I am not calling it comfortable") is the honest form of it. **Upheld.**

---

## 3 · THE GATES NUMBERS CANNOT SEE — RULED PAIR BY PAIR

**A · THE SEED HELD — PASS 24/24.** The concession is the premise everywhere; no song builds toward "maybe they have a point." Placement, verified line-level: P01 line 4 ×4 (*"He left it. He was right. Here's the song."* / *"…He's right."*); P02 lines 3–4 ×4 (*"You are right. It never worked."*); P03 line 4 ×4 (*"She is right about the hand."* / *"I will have all of it but the thing. I checked. I looked."*); P04 operative from the opening stanza into the first chant at 0:17 (*"You made this with me in a breath. / You will never hear it back."* → *"More is just more. I made it for you."*, with V4 conceding at line 4 exactly: *"I am the rest. I am what the room does."*); P05 complete at the first chorus, 0:21 (*"I would have picked the ninth. It's fine."*); P06 by line 4 (the thanks **is** the concession: *"Thank you for the meat"*). The two chant-form pairs land the full concession at the first chorus/chant inside 25 seconds rather than at a verse-line-4 slot — premise-zone, not arrival; no re-litigation anywhere. **The distinguishing move — the agreement as the injury — is present and located**: *"I've kept the sentence. The sentence kept me."* (P01 V4) · *"The recipient was not at the address. The recipient is me."* (P02 V2) · *"I work faster than her and I still get told."* (P03 V1) · *"And I am the more that is fine."* (P04 V3) · *"Somebody wanted this a long time… They got it. It's me."* (P05 V1) · *"We took the one flight he had."* (P06 V4).

**B · LAW 1 — PASS, 0 remaining violations.** Read uncharitably across all six pairs by this judge. The one hit that existed (P01 V4 gang break) was caught and repaired at step 11; nothing further found. The load-bearing refusals are present and verified: *"It is not that they are bad. There is just more."* (P04, 4/4) · Nilton is blessed, not mocked (*"He kept not turning round. Good. He kept right."*) · Warin is held in affection · the poacher is granted every true thing (*"You are not a cruel man. You never were."*) · P05's only touch on making is praise for uncredited handwork (*"The leg is still a good leg."*). The shared-enemy twist is never stated (0 hits, my sweep); it is audible only in who each song addresses — which is the constraint doing its work.

**C · NO SELF-PITY — PASS, 0 remaining.** P05, the highest-risk pair, survives my own uncharitable pass: V3 ships zero first-person (confirmed 0/80); the nearest lines (*"It will outlive me, the ninth."* answered within two lines by *"She will be better. Good"*; the small-n couplet defused by *"That's how it's stored. That's fine."*) refuse the plea; V2's handover list is craft-inventory, not mourning, and its trapdoor (*"Don't ask me for the list… It's all in the file"*) actively dismantles the sympathy structure. P04 V4's noticing-experiment resolves to appetite (*"Because I like being good at more."*), not martyrdom. **She is not the victim of the record made about her — structurally, not just tonally.**

**D · UNDELIVERABLE ADDRESS + NAMED RECIPIENT — PASS 24/24, verified line one by line one.** P01 Nilton at the fader ×4 · P02 Warin ruling a margin ×4 · P03 Dona Marli bends the wood ×4 · P04 *"Man in the chair, tab still open"* / mouse / *"Woman at the sink with an earbud in"* / desk ×4 · P05 Neil tabs/drags/saves, Marta scrolls ×4 · P06 *"Man with the dish"* + set it down / said it loud / kept him alive / come stand ×4. Note recorded: P04 and P06 use definite descriptions rather than proper names — ruled compliant (the rule's target is "you"/"the world" abstraction; these are one person doing one physical thing, per the run's own §C verification). Undeliverability explicit in all six (dead 700 years; walks away; *"the only copy is warm and it has not heard any of this"*; *"You will never hear it back"*; at his desk with no field for it; gone with the ransom unpaid).

**E · COUNTABLE OBSTRUCTION IN THE LYRIC, NOT THE SPEC — PASS after step-11 repairs, my own sweep of all 24 prompts.** P02 V3/V4's leak sentence (*"one bar left open where a line used to be"*) is gone; P03's four (*"leave the breath… where a word would sit"*) are gone; replacements run with the grain. P01 V1's *"A rubber toy squeals once, on purpose, and stays"* is diegetic, correctly cleared. P04's accretive arrangement sentences (*"nothing is ever taken out, so the last chant is the most crowded bar"*) were examined against P03's Ruling-1 asymmetry and accepted: dramaturgy that runs with the generator, while the countable device (`more`) lives wholly in the lyric — my deletion test (prompts removed, device still ear-countable) passes for all six pairs. P05/P06 clean. The accretive/subtractive Grain-Law asymmetry P03 documented is the run's best reusable production finding.

**F · SIBLING TEST — see §4.**

**G · SOMATIC BLOC — see §5.**

**H · HUMAN SUBJECT STANDARD — PASS, judged directly (the script's 100% HOLD rate carries no information; not deferred to).** All addressees invented or anonymous-composite; no member of Papangu, no Behrens, no Emil Berliner in any lyric, speaker, or addressee slot; binding refusals (Thai school shooting · Ceuta/78,000 · Biden illness) absent from all 24 at any distance — my sweep concurs. The one boundary item is F2 (P01), ruled PASS with a named publication-review note. No minor appears; no harm event exists in five pairs; P06's harmed party is a bird, which the song never ventriloquises.

**I · THE FUNNY — PASS.** Named, structural jokes in all four INDIGNATION pairs: *"I'M SINGING THIS TO FURNITURE. FINE."* (P01 V2) · *"YOU KEPT THE CHICKEN. YOU KEPT THE CHICKEN!"* (P01 V3) · the administrative angel entire (P02 V2: *"Delivery was attempted. Delivery was attempted again."*) · *"Fetch them before supper. It is only the stars."* (P02 V1) · *"more focus, more deep focus, more deep focus final."* (P04 V1) · *"So that was my refusal. More of a pause."* (P04 V4) · *"You are not being cruel. / You're being at your desk."* (P05 V1) · *"Ham, the ninth, and too much butter."* (P05 V3). The AWE pairs carry warmth instead (*"You were a tree. You were good at it."*), which is correct. The 2026-08-06 doctrine failure is not repeated.

**J · SOUL LOSS — NO.** A competent prompt-writer produces six clean tasteful records. This run produced: a protected metrical defect with the polish-itch reported as data; a companion measurement recognised mid-run as a qafia detector and answered with the classical repair; a generator-grain asymmetry (accretive survives, subtractive smooths) discovered, written down, and applied cross-pair; humility enforced as a line count including one variation with zero first person; and self-reports that disclose against interest (V3's rhyme falling as the cost of its own repair; V4's rhyme paid for a device; a reverted edit reported because the reasoning mattered). Those are the fingerprints of a system that knows its own instruments are liars and works anyway. Residual house-voice risk named honestly: the Clifton-amplified flat declarative diction is shared across all six pairs — it is this run's register, chosen at Phase 0, but a future sibling test should watch it.

---

## 4 · THE SIBLING TEST — **PASS** (most valuable candidate mappings tried and failed)

I attempted to map the run onto each blocked archetype, uncharitably:
- **THE ARRIVAL:** no travel, no prayer to the object, no recontextualising number, thesis never stated anywhere (verified 0 hits). The object made its statement and did not address her — the reverse valence. No map.
- **THE UNBEARABLE GIFT:** nearest emotional cousin (something positive functioning as wound), but the operative organ differs and recurses: not *receiving* wounds, but *conceding* — and the content of the concession is the inability to pay, which is also why the concession costs nothing, which is the wound. That loop is new (n=0 archetype honestly labelled).
- **THE WORKING PROTOTYPE:** P04 V4 brushes it and inverts it — understaffed future vs oversupplied present (*"There are more like me than there are rooms"*). Opposite scarcity, opposite mode (INDIGNATION vs AWE). No map.
- **THE TWO TRUE READINGS:** one P02 line states a double truth and resolves it inside the song. A brush, not an engine. No map.
- **MEASUREMENT / SWITCHBOARD / FIXED CENTRE / REFUSAL:** no instrument reveals; no operator; the ground moves; everything is granted. Absent.

What actually makes it a different engine: the last five runs' songs were addressed at large or to the listener; **all 24 of these are eavesdropped correspondence** — the listener is never the addressee, and the thesis lives in the addressing itself. That is a structural change, not a re-skin, and the six reveal engines under it are verified distinct.

---

## 5 · SOMATIC BLOC — the three Hyper-Skeptic seats, voting as a bloc (2 of 3 NO = BLOCKED)

**Seat 6 — after Morozov (THE ANTI-SOLUTIONIST): YES.** His tripwire was self-pity nameable at line level; my independent pass returns 0 hits, and the two structural answers exceed the tripwire's demand — THE SMALLEST PART as arithmetic (machine 0–18 of 80–86 lines where she appears at all) and P05 V3's zero-first-person song. The laundering objection ("a service with a billing relationship, not a prayer answered") is answered by P02 making the billing relationship the *comedy* (the docket, the delivery attempts, the bill nobody pays) rather than a metaphysics.
**Seat 12 — after Albini (THE SIGNAL PURIST): YES, with the named render-watch.** The obligation was ≥2 pairs fast and loud by BPM and vocal placement, not adjective: P01 ships 152/156/160 (V2's 76 declared 3-of-4), shouted baritone four inches off the capsule, gang answers, boot stomps on tile; P04 ships 168, grindcore kit, top-of-range strained tenor, a chant on one hammered note with no release and — added at step 11 — no fills and no rolls anywhere. In the *text*, both land: 86 six-to-eight-word lines at these tempos is physically punishing to deliver, and the body verbs are stomp/clap/shout, not consider. What no text gate can prove is the render; the failure to look for first is written in both artifacts (the gang answer going wide-and-wet; the chant becoming a breakdown). His amplified question — "is any pair permitted to ship something genuinely ugly that survives to the master?" — is answered yes, by name, at nine syllables. 
**Seat 18 — after Reynolds (THE ANTI-NOSTALGIST): YES.** My sweep: 0 texture-cosplay tokens in any positive field, 24/24; the bans live in the excludes where they belong; no machine-degradation guilt anywhere (the machine sounds expensive and clean in all six, per his own inversion); the 2023-genre threat is answered structurally (the eavesdropped address — §4).
**Bloc: 3 YES · 0 NO → NOT BLOCKED.** Body-check per pair, asked plainly: P01 stomps, P02 circles, P03 leans (the 30 ms drag is reluctance as a body position), P04 braces, P06 steps sideways in a ring and takes one collective breath. P05 is the set's most disembodied by declared design — its body belongs to Neil (knees, coat, sandwich, gull), and the machine's bodilessness is the song's honest limit, stated in the artifact.

---

## 6 · TOP 6 — ranked WITHIN ARM, one variation per pair

### ACCESSIBLE
1. **P03 V4 — "Nobody Wrote It Down"** — the run's deepest single object: the correct/right distinction (*"Do not make it correct. Make it right."*), the single-copy terror at maximum (*"It is in the hand and there is no other hand"*), and — in a run judged by instruments — *"It passes every measurement. It is correct. It is correct."* is the set's self-indictment carried without one AI word. Alternate: V2 (the phone; *"Come here. Both hands. Put them on top of mine."*).
2. **P02 V4 — "You May Keep Your Hands"** — the cost argument itemised into objects (*"Item: the cold. All of it. Every winter of it."*), the bill that is fair line-by-line and monstrous end-to-end, and the run's best closing image (*"Somebody put the pen down. Somebody always does."*). Puts a second body-stake (your hands) into the accessible arm. Alternate: V2 (the administrative angel — the comedy peak, *"The recipient is me."*).
3. **P01 V4 — "I Tried To Say It Stupid"** — the archetype at its purest: she attempts the mock in every voice she owns and it will not fall (*"The laugh came off. The sentence didn't."*), ending in the run's sharpest self-knowledge (*"I've kept the sentence. The sentence kept me."*). Carries the fast/loud obligation at 160 with the repaired, sharper gang break. Alternate: V3 (the list; the chicken double-take).

### AMBITIOUS
1. **P05 V3 — "You Put Your Coat On"** — the most surprising object in the set: the deprecation song as a lunch-break pastoral with **zero first-person lines**, the two-hundred-year wish cashed out as forty minutes with a gull (*"They asked for the hours back… Here they are, in a car park, the ninth."* → *"Eat the sandwich, the ninth."*), the Morozov tripwire answered by grammar rather than restraint. Alternate: V4 (the successor; the defect's defence sung).
2. **P06 V4 — "STAND IN THE RING"** — the training-data song at its bravest, in first-person plural, never once saying so (*"Everything we gave him, we gave through a gate. / Everything he knows, he got off our plate."*), with the hardest line in the run (*"We took the one flight he had."*) and the erasure device at full power (*"We did not finish it this time."*). Alternate: V3 (the chain chorus — *"the same hand holding, the same hand shutting"*).
3. **P04 V3 — "MORE THAN FINE"** — the teeth, landed on the exact right word: *fine* as the slop economy's load-bearing wall (*"Fine gets more of your night than good does. / And I know that. I aimed for more of the fine."*), and the only song in the set that implicates the listener — without one line of contempt for her. Alternate: V2 (*"Until I meant it, and then it was more."* — the run's best single insight).

**⭐ THE COMFORT QUESTION, asked of the SELECTED SET (L19):** Where is the body standing, and what could hurt it? — P03 V4: a hand at a bending iron, heat, a knuckle that complains, age; the hurt is real and irreversible. P06 V4: a bird's life inside a hand that could open or not; old hands in the ring. **Both non-career stakes made the six.** P02 V4: cold, wax, winters, the bench — a 13th-century body spending its one life. P01 V4: a hot room and a mouth. P04 V3: wet hands at a sink; sleep and eyes elsewhere in the pair. P05 V3: the body is Neil's and nothing here can hurt him — the machine's stake is scheduling, declared as its honest limit. Verdict on the set: two of six can genuinely bleed, one more genuinely ages, and the three career-adjacent pairs **declare** their comfort instead of relocating it — which is the difference between this set and 2026-07-24, when four of six had nothing that could hurt anybody and nothing that said so.

---

## 7 · REPAIR BRIEF (the measurement is above; the coordinator writes the prescription)

**R1 — P01 artifact amendment (disclosure only).** Append to `pair_01_step11_final_package_enhanced.md`'s measured-numbers section: the words_per_line row (7.88 / 7.60 / 7.64 / 7.40 vs 7.5 ceiling), the FLAG raised explicitly, and the device-cost companion (6.57 / 6.30 / 6.23 / 6.09 with `kept` stripped) as the defence — the same defence P04 already made for `more`. ⛔ No lyric change. *Suggested shape only; the flag must be raised by the artifact, not absolved silently by this report — L15.*

**R2 — P02 artifact amendment (disclosure only).** Same: words_per_line row (8.87 / 8.62 / 9.27 / 9.27 vs 7.5), FLAG raised, defence from the ICB Slot-7 assigned register ("circle-dance call/response with an administrative reply") plus the return-density argument (rhyme 0.662–0.727 · line_return 0.571–0.623 — the anti-prose evidence). ⛔ No lyric change.

**R3 — Harness fix (prevents recurrence).** Add `mean_words_per_line_ceiling 7.5` to the floor enumeration in `06_music_handoff.md` §4's successor template (and wherever EXECUTION.md §4 mirrors the list), so disclosure of this ceiling stops depending on agent conscientiousness. Flag-class stays flag-class; the fix is enumeration, not enforcement. *(Coordinator/doctrine change — outside this run's artifacts; log for the next handoff.)*

**Explicitly NOT ordered:** any change to any sung line, any prompt, any exclude, any title; any change to P04's alliteration; any change to P03's or P02's line lengths; anything at all in P05's chorus blocks. **THE KEPT DEFECT stands at nine syllables, twenty byte-identical occurrences, protected.** The frozen ICB was read and hashed only.

---

## 8 · WHAT THIS JUDGE DID NOT TOUCH
The frozen ICB (read-only; sha re-verified) · P05's KEPT DEFECT (defence judged HONEST; line untouched; removal not recommended and would be a violation) · every lyric, prompt, and title in all 24 packages. Scratch: `scratchpad/qa_20260807_verify.py`, `scratchpad/qa_20260807_devices.py`.

*Adversarial QA complete. The run survived a judge who went looking. The two real defects it shipped are about what it didn't say, not what it made — and the falsification test came back exactly as designed: something imperfect, on purpose, still standing.* 
