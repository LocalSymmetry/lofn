# QA_REPORT — `daily-music-2026-08-22-sunna-offbeat`
## INDEPENDENT JUDGE · adversarial audit of 24 songs · written 2026-08-22

> I did not write any of this run and I am not here to like it. Every verdict below names a file, a
> line, or a script output a stranger could re-open and re-check. Where I could not check something,
> **§11 WHAT I COULD NOT SEE** says so in words.

**RUN VERDICT: REPAIR.** · **8 repair items issued · 4 items escalated to The Scientist as HOLD.**

⭐ **THE ZERO-REJECTION TRIPWIRE DID NOT FIRE.** This audit says no in four places, and the most
serious finding is one that **every instrument this run owns reports as clean**.

---

# 0. THE HEADLINE

> ⭐⭐ **RULING X — the ruling written to make the run's unmeasurable claims checkable — WAS SATISFIED
> FOUR TIMES BY THE SAME SENTENCE IN THE SAME SLOT.**
>
> Five of six pairs invented **one** sentence for naming their absence and wrote a variant of it in all
> four variations, in the **same section, at the same line index**. The run's own falsification tests
> all point ACROSS pairs (hole-swap, transplant, wolf-swap). **Not one of them asks whether the four
> variations of a single pair differ in the place the ruling put them.** The one instrument that does
> — `distinctiveness.py` test 7 — measures **exact** repeated lines and correctly returns 0, because
> the blocks are paraphrased, not copied.
>
> This is `R3.2` rule 3 in its purest form: *"if an assignment can be satisfied by filling in a blank,
> it is a grammar and it will be templated."* **RULING X is the blank.**

Evidence, all four P03 variations, first three lines of Verse 2
(`pair_03_step10_final_package_enhanced.md:73-75, 231-233, 385-387, 538-540`):

| V | line 1 | line 2 | line 3 |
|---|---|---|---|
| V1 | The **melodica** played the whole of that, and now it has gone. | It is not coming back. Fine. | The bass **leaves the hole open** where the tune used to come in. |
| V2 | The **saxophone** had the tune for the first bit and then it went. | It is not coming back in… | The bass **leaves the hole at the top** where the horn came down. |
| V3 | The **trombone** finishes the line it is on, and then it is gone. | It does not come back. I waited. | The bass **keeps the gap open** at the top of every time round. |
| V4 | The **flute** had the tune down the road, and then it stopped. | It has gone. It is not coming back on. | The bass goes round and **leaves the top of it empty** every time. |

⛔ **Same section index [3]. Same line indices 1–3. Same three moves. One noun swapped.**

---

# 1. THE THREE PREDICTED FALSIFICATION TESTS — MEASURED

## TEST 1 — THE HOLE-SWAP (15 unordered pairings, run on the BLOCK)

**Predicted** (`05_pair_assignments.md`, "TEST 1"): *14 of 15 break; **P02 × P04** at risk, mitigation
assigned.*

**MEASURED: 13 of 15 break. 2 fail. NEITHER failure is the predicted one. The predicted at-risk
pairing HELD.**

| pairing | result | why |
|---|---|---|
| P01×P02 | **BREAKS** | a jump-blues shuffle with no beat one is not a shuffle; the walking bass is defined by the pulse |
| P01×P03 | **BREAKS** | a whole departing lead would delete P01's horn-answer architecture (its verse form); and a one-drop vamp has no V to remove |
| P01×P04 | **BREAKS** | P01's hanging third line is answered by **horns**, not a voice; the swap deletes the verse architecture |
| P01×P05 | **BREAKS HARD** | P01 is *made of* names — Marek, Bev, Pam, Row, Joyce, Del, Vic, Cass, Nell, Tam, Ade, Ken, Nita, Gwen, Sal, Fen. A song whose engine is naming people cannot host an unsayable name. Strongest single result in the test. |
| P01×P06 | **BREAKS** | vacating the centre fights a one-room-one-band record; a modal Dorian art-pop has no functional V to delete |
| **P02×P03** | ⛔ **FAILS** | **P02's assigned absence IS P03's bed.** The one drop *deletes beat one* by definition (`00_research_brief.md` §2, T2). Putting P02's hole into P03 is a null operation. |
| P02×P04 | **BREAKS** *(predicted at-risk)* | mitigation held: P02 excludes *"no crowd chant filling the gap"* ×4; P04V3's lyric has *"the whole car park comes in exactly on time."* Filled vs unfilled, as assigned. |
| **P02×P05** | ⛔ **FAILS** | **P02V3 is already doing P05's hole.** Its whole device is a signatory who is never named and everyone else being a function (*the man in green · the biro woman · the driver · a bloke*). P02V3's title is **"Somebody Signed It"**; P05V2's is **"Whoever Signed It."** |
| P02×P06 | **BREAKS** | in art-pop the programmed sub *is* the downbeat; deleting it deletes the bed |
| P03×P04 | **BREAKS** | P04's chime carries the tune all record; if it left the song has no tune. P03 excludes *"no answering second singer, no duet."* |
| P03×P05 | **BREAKS** (weakly one way) | P05's chiming figure returns after the last vocal — permanent departure would break its HUM-TEST answer. Reverse is weak: P03V2/V3/V4 barely name anyone. |
| P03×P06 | **BREAKS** | P06 has no lead instrument to lose (brass is punctuation); P03 excludes *"no hard-panned effects"* ×4 |
| P04×P05 | **BREAKS HARD** | P05 excludes *"no second vocalist, no duet"* ×4; and P04's reveal is that Della **is** named. Exact opposites. |
| P04×P06 | **BREAKS** | P06's answers are brass stabs; P04 excludes *"no stereo widener on the vocal"* |
| P05×P06 | **BREAKS HARD** | P06's assigned verse architecture is *"list-stanzas of exactly five items, the fifth always a person"* — Perry, Ines, Oti, Marek, Nadia, Sami, Ruthie, Del, Fen |

⭐ **THE FINDING BEHIND THE FINDING.** `R3.1` says *a falsification test only protects the thing it
names.* The one pairing that was named got a mitigation and survived. **The two that failed are the two
nobody named** — and both have the same structural cause: `05_pair_assignments.md` asserts six absences
at six architectural levels but **never checks the six absences against the six BEDS, or against each
other's executed lyrics.** P02's level (rhythmic / the downbeat) was already occupied by P03's genre
before a word was written.

## TEST 2 — THE GENRE TRANSPLANT (RULING C2)

**Predicted:** *P01, P03, P05, P06 survive; **P02 and P04** exposed.*
**MEASURED: 5 of 6 survive. P02 confirmed exposed. ⛔ P04 FALSIFIED — it survives.**

| pair | verdict | evidence |
|---|---|---|
| P01 | **SURVIVES** | the seating plan, the named guests, the card for Nell and the refrain *"Here's where everybody sits"* are bed-independent |
| P02 | **EXPOSED** ✔ predicted | re-record as swing and the *words* survive intact, but the absence does not: there is no four-on-the-floor to delete. The exposure is exactly at the absence, as predicted. |
| P03 | **SURVIVES** | launderette, borrowed coat, appetite; a departing lead transplants to any bed |
| P04 | ⛔ **SURVIVES — PREDICTION FALSIFIED** | the prediction reasoned *"a skank-over-break absence is made of its bed"* — but P04's **assigned** absence is PERFORMER-level (the second voice), and call-and-response is older than every bed in this run. The prediction confused the bed with the hole. |
| P05 | **SURVIVES** | function-only naming is a lyric device; re-record as rocksteady and nothing is lost |
| P06 | **SURVIVES** | an office taken apart and a woman taking the night; the list survives any bed |

⭐ **Marsalis's gate is answered — 5 of 6 survive — and one of the two predictions written to
anticipate it was wrong.** Report both, per `05`.

## TEST 3 — R3's WOLF SWAP (4 wolves, 6 pairings)

**MEASURED: 5 of 6 lose something. 1 fails — P02 ↔ P05, the same collision as TEST 1.**

| pairing | result |
|---|---|
| P01↔P02 | **LOSES** — a private card in a public indictment shrinks the coach song to a grief song |
| P01↔P04 | **LOSES** — P04's missing thing is explicitly *not* a loss; a dead friend destroys its AWE |
| P01↔P05 | **LOSES** — but by collision (P05's absence is already a removed person), not by contrast |
| P02↔P04 | **LOSES** — the car park has no institution to sign anything |
| **P02↔P05** | ⛔ **FAILS** — *"Somebody signed it. Somebody's warm."* (P02V3) and *"Whoever signed it. Whoever countersigned."* (P05V2) are the same wolf at two scales |
| P04↔P05 | **LOSES** — a substitute singer is not a wound and cannot do P05's job |

⚠️ **SEPARATE FINDING — P04 IS NOT CARRYING A WOLF.** `05_pair_assignments.md` marks P04 **"YES —
event"** and counts *"wolves 4 of 6, at ceiling ✅"*. But P04's own concept brief says *"That is not a
scandal; it is what a song is for, and realising it is the AWE"*, and the lyrics agree:
*"That isn't a thing that went wrong. That's the job."* (V2) · *"Nobody stood in for anybody. Somebody
stood up."* (V4) · *"That was not an accident. That was somebody's afternoon."* (V3).
**The run has 3 wolves, not 4.** No rule is broken (ceiling 4, floor 2 wolf-free) — but the arithmetic
that licensed *"at ceiling"* is wrong, and `INDEX.md` should say 3.

---

# 2. THE SEVEN BINDING RULINGS — PER PAIR

| ruling | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|
| **A1** decision, not accident | ✅ | ✅ | ✅ | ✅✅ | ✅ | ✅✅ |
| **B1** state → remove → keep pointing | ✅ 0:38 | ✅ 0:30.5 | ✅ | ✅ | ⚠️ **PARTIAL** | ✅ 1:09 |
| **A2** two cycles · no hole < 0:30 | ✅ | ✅✅ | ✅ | ✅ | ⛔ **FAIL** | ✅ |
| **C1** unplayed, not absent | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **C2** genre transplant | ✅ | ⚠️ exposed *(predicted)* | ✅ | ✅ | ✅ | ✅ |
| **C3** credit the player | ✅ | ✅✅ | ✅✅ | ⛔ **class credit only** | ✅✅ | ✅ |
| **X** every claim cashes out | ⚠️ | ⚠️ | ⛔ | ⛔ | ⚠️ | ✅ |

## A1 — IT WAS A DECISION, NOT AN ACCIDENT · **FULL COMPLIANCE, best-executed of the seven**
Multiple pairs put the word in the lyric, unprompted:
`pair_04_…:449` *"Somebody decided that. Somebody said leave it in. / That was not an accident."* ·
`pair_04_…:128` *"That is a decision. Somebody made it."* ·
`pair_01_…` V2 *"Somebody chose it. Somebody sat and chose."* / *"She's not warming up. She's deciding."* ·
`pair_03_…` V1 *"They sat down and did that on purpose, at a table, at home."* ·
`pair_06_…` V4 *"Box tape tearing in time, which cannot be an accident."*
I swept for the banned framing (*"we got it wrong and it turned out beautiful"*) and found **zero**
instances. The nearest thing — P04V2's *"The guitar's top string is flat the whole way through / They
left it in"* — is a decision to keep, which is the ruling's own shape.

## B1 / A2 — checked in the **section maps**, not the step-10 headers, as instructed

| pair | figure stated | absence taken | source |
|---|---|---|---|
| P01 | turnaround **twice**, 8-bar intro | **0:38** (turnaround of Verse 1, bars 17–20) | `pair_01_step07_song_guides.md:95,105,164,223,283` |
| P02 | downbeat, bars 1–16 = **two full 8-bar cycles = 30.5 s** | **0:30.5** (bar 17, the Lift) | `pair_02_step07_song_guides.md:37-38,74,130,184,241` |
| P03 | instrument leads the **entire first section** (bass 2 cycles + instrument 2 cycles + kit) | end of first chorus | `pair_03_step07_song_guides.md:65,76,102` |
| P04 | both voices trade **two full rounds** — *"second full round complete"* at **1:38 / 1:19** | Verse 3, **1:52** | `pair_04_step07_song_guides.md:38-42,86-88` |
| **P05** | ⛔ name said **ONCE**, bar 9 = **0:19** | ~0:21 (first function-word) | `pair_05_step07_song_guides.md:275` |
| P06 | stab figure **stated twice**, 0:00–0:17; lead dead-centre through 1:09 | **1:09** (downbeat of Verse 2) | `pair_06_step07_song_guides.md:39-43,55,59` |

⛔ **P05 FAILS RULING A2 ON BOTH CLAUSES, AND THE DEFECT IS IN THE COORDINATOR, NOT THE PAIR.**
`05_pair_assignments.md` STATE-IT-FIRST cell for P05 reads: *"the name is said **once**, early, by
somebody else."* RULING A2 in the ICB reads: *"The figure is stated **at least twice** before the
absence is taken. ⛔ No hole in the first 30 seconds."* **The assignment table contradicts a binding
ruling and nobody noticed.** P05 complied with its slot brief.

P05's cross-check row (`:281`) answers A2's clock with *"hole after 0:30, twice, inconvenient | bars
76 · 101"* — **but bars 76 and 101 are an arrangement rest, not P05's assigned absence.** The assigned
absence is LYRIC-level. Crediting a different, unassigned hole against the clause is a substitution,
and it is the only place in the run where a ruling was answered by changing the subject.

## C1 — every pair answers *"how does the body still feel it"* in a named production behaviour
P01 *"the bass still climbs to the fifth degree at the top of every turnaround with nothing above it"* ·
P02 *"leaning a fat note onto the four-and"* · P03 *"the bass keeps that answer slot empty… with only
spring reverb in it"* · P04 *"her bars held empty and exactly in time by the skank"* · P05 *"the bass
answers the hook alone in the two-bar gaps of every chorus"* · P06 *"the sides never match."*
**6/6 PASS**, all in the MUSIC PROMPT (readable by the generator), not only in prose.

## C3 — CREDIT THE PLAYER
`credit_sweep.py` → **33 credited names, 0 leaks.** I did not accept that as sufficient, because
`credit_sweep.py` can only find a name that was **also credited** — a real musician named in a style
field and credited nowhere is invisible to it. **I ran an independent 81-probe sweep** (Sublime,
Nowell, Happoldt, Gershwin, Specials, Skatalites, Knibb, Carlton Barrett, King Tubby, Linkin Park,
Bennington, Delson, Bourdon, Noga Erez, Rousso, Neneh Cherry, TikTok, Carney, Trump, ElevenLabs,
Winstons, Studio One, 2 Tone, Doin' Time, Coventry, Kingston …) over all 4 generator-facing fields ×
24 variations: **0 hits.** *(reproducible: 81 literal probes, word-boundary matched, over the fence-inner text of all four generator fields.)*

⛔ **P04 IS THE ONE PAIR THAT CREDITS NO PLAYER.** Its four dedications credit **a class**, explicitly:
*"They are credited as a class because we would rather owe all of them than name none of them"* ·
*"Credited as a class, deliberately — no individual is named and none is meant."*
RULING C3 is *"CREDIT THE PLAYER, NOT THE SCENE… a scene credit is the easy one and nobody is owed by
a scene."* The reasoning in the artifact is honest and I do not think it was evasion — P04 did not ask
the generator for anybody's figure by name, so *"we asked for it by name, so we name him"* does not
apply. **But `00_research_brief.md` §2 T5 promised in writing:** *"His name and Michael Happoldt's
belong in `DEDICATIONS.md`."* **`DEDICATIONS.md` does not exist** (`RUN_STATE.md:20`, `pending`). Until
it does and carries them, the run's own written promise is unkept and the class credit is the only
payment made. → **HOLD 2.**

**Dedication length, measured not eyeballed:** all 24 ≤ 500 chars (range 420–494). ✅
**P05's dedications are the run's best** — each names the specific mechanism and the player who
authored it (Bourdon: *"a kick drum refusing to be interesting"*; Delson: *"a three-note guitar part
that behaves like a bell and refuses to become a riff"*; Bernard Edwards; Neneh Cherry). **P02 and P03
pay the two credits the seed is built on** — Lloyd Knibb (the upstroke) and Carlton Barrett (the one
drop). That is the credit the system is built to forget, and it was not forgotten.

---

# 3. RULING X — THE ONE THE RUN WAS MOST LIKELY TO HAVE FAKED

## X.1 The absence NAMED IN THE LYRIC, in words, once — **all 24 checked by hand**

**PRESENT in all 24.** Nothing was answered only in the production spec. Counts verified exactly once
per variation where the naming is a proper noun: `Nell` ×4 (`pair_01…:92, 228, 382, 513`);
`Halloran` ×4 (`pair_05…:43, 206, 369, 534`); `nobody answering` ×4 (`pair_04…:128, 273, 446, 603`).

⛔ **BUT THE NAMING IS A TEMPLATE.** Section-position of the naming line and of the sung fact, measured
across all 24 (positional map, §11 method note):

| pair | absence named at | sung fact at | verdict |
|---|---|---|---|
| **P03** | **Verse 2 [idx 3] lines 2–3 — ALL FOUR** | **Verse 2 [idx 3] — ALL FOUR** | ⛔ **TOTAL LOCK** |
| **P05** | **Intro [idx 0] line 1 — ALL FOUR** | **Verse 3 [idx 7] line 1 — ALL FOUR** | ⛔ **TOTAL LOCK** |
| **P02** | Verse 4 [8] L1 ×3, Verse 3 [6] ×1 | **Verse 3 [6] — ALL FOUR** | ⛔ near-lock |
| **P04** | Verse 4 [6] L1 ×3, Verse 5 ×1 | **Verse 3 [4] — ALL FOUR** | ⛔ near-lock |
| P01 | Verse 3, Verse 3, Bridge, Verse 2 | Verse 2 ×3, Verse 4 ×1 | ⚠️ partial |
| **P06** | Verse 2, Verse 2, The Wedge, Verse 1 | Turn ×2, Verse 2 ×2 | ✅ **varied** |

**The paraphrase instrument I built for this** (`output/daily/2026-08-22/blockdup.py` — line-level token-set
Jaccard ≥ 0.50, refrains excluded), **calibrated in both directions before I believed it:**

| corpus | matches / comparison |
|---|---|
| CROSS-pair, this run (unrelated songs — must be ~0) | **0.02** |
| P06 | **0.17** |
| P01 | 0.50 |
| P03 | 0.83 |
| P05 | 1.00 |
| **P02** | **1.83** |
| **NEGATIVE CONTROL — 2026-08-21 pair 04, independently ruled templated** | **1.83** |
| **P04** | ⛔ **2.17 — HIGHER THAN THE KNOWN-TEMPLATED CORPUS** |

**P04's block, all four:** *"There's nobody answering now."* / *"There's nobody answering me now."* /
*"There's nobody answering me now, and that's the whole record."* / *"There's nobody answering now.
Nobody's going to."* — followed in every case by *"The bar goes by in time…"* (0.67, 0.57, 0.50).

**P02's closing formula, all four outros** — *"And nothing hits the top of the bar. / And nothing does.
And then we stop."* · *"And nothing lands on top of the bar. / And nothing will. And that's the best."* ·
*"And nothing comes down on the top. / And we go home. And that's the night."* · *"And nothing lands
where the beat should land. / And nobody minds. And we don't stop."* **This is exactly what R3.2 rule 4
(the closing-block swap) exists to catch, and `distinctiveness.py` test 3 scored it 0.077 against a
0.50 ceiling.**

**A second instrument** (`output/daily/2026-08-22/skeleton.py`, positional function-word skeleton) fired 2 more real
hits on P03 (V2×V4 and V1×V4, Verse-2 line slots) — **but it is UNCALIBRATED and I report it as such:
its negative control returns 0, because that corpus's templating was lexical, not skeletal. Its hits
are evidence; its zeros are not.**

## X.2 The strut as a NAMED INSTRUMENT entering/leaving at a NAMED POINT — **24/24 PASS**
Every MUSIC PROMPT carries it, in the generator-readable field, never only in prose. Examples:
P01 *"Handclaps at the first chorus, stacked girl-group thirds at the second, tambourine at the fourth
verse"* · P04 *"break enters at bar three… guitar at bar five… bass at bar seven"* · P06 *"a small
speaker is lifted off the middle of the floor onto a window ledge and handclaps take the empty middle."*

## X.3 SOURCE-2 verbatim imports — measured

| pair | assigned | found verbatim in step 10 |
|---|---|---|
| P01 | *twirling acoustic guitars* · *quiet reverence* (inverted) | **7 · 5** ✅ |
| P02 | *a sonic assault* (refused) · *gnarled* | **5 · 6** ✅ |
| P03 | *lazily-played synth pads* · *detuned bliss* | **4 · 4** ✅ |
| **P04** | *detuned bliss* · *a chiming synth riff* | ⚠️ **0 · 0** |
| P05 | *howling guitars* · *Olympian guitar feats* | **4 · 4** ✅ |
| P06 | *transmogrifies those voices…sonic events* · *glassy textures* | **4 · 4** ✅ |

P04 carries the **behaviour** of both (*"a chiming three-note synth riff"*; *"tuned a few cents sharp
and left out of true"*) but is the only pair with **zero** case-exact imports. *"detuned bliss"* appears
once in `pair_04_step06_facets.md` and never reaches the prompt. **CONCERN, not a fail** — the brief's
requirement is translation into production behaviour and that is met.

## X.4 Bandcamp phrases inside a lyric — **0 across all 24.** ✅

---

# 4. THE DAILY RULES

| rule | verdict | evidence |
|---|---|---|
| Tri-source declared **before** writing | ✅ | mtimes: `00` 13:09 → `02` 13:11 → `03` 13:15 → `04` 13:17 → `05` 13:18 → ICB 13:20 → **earliest pair artifact 13:34**. 25 minutes clear. |
| 3 NEWS / 3 EXISTENCE | ✅ | NEWS P02/P05/P06 · EXISTENCE P01/P03/P04 |
| ≥1 AWE and ≥1 INDIGNATION | ✅ | AWE 4 · INDIGNATION 2 |
| Six distinct beds | ✅ | swing · ska · rocksteady · ska-punk×boom-bap · rap-rock · art-pop |
| Six distinct tempos, all 100–129 | ✅ | 102 / 126 / 108 / 118 / 100 / 112 — verified from the `[SONG FORM]` line of all 24 |
| Six distinct settings | ✅ | function room · hard shoulder · launderette · car park · stairwell · office. Domestic-warm **1 of 6**. Club beds **0**. Drive **1 of 6**. |
| Six distinct verse architectures | ✅ | verified by reading; P01 couplet+hanging third, P02 stressed-monosyllable quatrains, P03 unrhymed one-breath vowel-chain, P04 call-and-response with shrinking answer, P05 spoken tercets/sung quatrains, P06 five-item list-stanzas |
| Six distinct forbidden sections, thesis absent from each | ✅ | final chorus / bridge / last verse / opening verse / chorus / outro — all clear. ⚠️ **one soft concern:** P02V2's bridge (*"The paper sheet has not gone wrong… The fault light's been on since the spring. Nobody's come."*) is the clearest statement of P02's reveal engine in that variation, and the bridge is P02's forbidden section. Borderline; logged, not failed. |
| Six distinct reveal engines | ⚠️ | **declared** distinct; **executed**, P02V3's engine collides with P05's (see TEST 1) |
| **Exactly one sung numeric fact, spelled in words** | ✅ **with one concern** | **0 digits in any sung line or title, all 24** (verified twice). One ledger fact per song. ⚠️ **P05V3 sings 84 number-words** (fourteen ×12, twelve ×10, four ×6, ten ×4, eleven ×2 …) and P05V1 63. The imported fact (*the thirteenth floor*) is sung once and answered; the rest is a counting device, not facts — but the density dilutes the hinge the rule exists to protect. **CONCERN.** |
| Wolves ≤ 4 of 6 | ✅ *(with a correction)* | declared 4; **measured 3** — P04's is an AWE, not a wound (§1 TEST 3) |
| **P03 and P06 carry no dark line** | ✅ **verified line by line** | P03 states it four ways in the lyric (*"Nothing is wrong. Nothing has gone wrong. That is all the news."*). The only *"dark"* in P03 is literal and enjoyed: *"It makes a dark bit in the road and I like the dark bit."* P06 does not end sad: V1 *"turning the last lamp round"* · V3 *"whistling up the stairwell at us"* · V4 five people singing unaccompanied after the power cut. |
| Byte-identical chorus | ✅ 20/24 · **4 declared exceptions** | P01/P02/P03/P05/P06 = 1 distinct chorus text per song. **P04 = 2–3 distinct**, because its assigned absence *is* the disappearing response. The **calls** are byte-identical throughout; only the answer bars empty. Legitimate by design and self-declared in the `[SONG FORM]` line — **but it must be a named deviation in `INDEX.md`, since `02_golden_seed.md` §3 makes the byte-identical chorus "mandatory in all 24 songs."** |
| House-lexicon drift | ✅ | 4 incidental hits across 24 songs, 1 occurrence each: `tea` (P02V1), `chip shop` (P03V4), `chips` (P04V1), `kettle` (P06V4). The in-flight P06 repair landed — `kettle` is now 1 of 4 (`pair_06_…:571`), not 3 of 4. |
| Hard gates | ✅ | `measure_variations.py` 24/24, **0 hard fails, 1 flag** (P05V3 prompt 963, inside the 850–1000 band). `validate_step.py --gate-report 10` → **STEP 10 PASSED** on all six. |

---

# 5. THE HUMAN SUBJECT STANDARD — ⛔ NO BREACH

| check | verdict | evidence |
|---|---|---|
| Real harmed person named + located | ✅ **CLEAN** | independent 81-probe sweep: **0 hits** in any generator-facing field |
| Real musician / band / record / company / official / case in a lyric or style field | ✅ **CLEAN** | same sweep. Invented throughout: Nell, Marnie, Della, Halloran, Draymark, Kestle House, Ferrow Street, Perry, Ines, Oti, Marek, Nadia, Sami, Ruthie, Fen. |
| **No child anywhere in the run** | ✅ **CLEAN, with one note** | full regex sweep (child/kid/baby/infant/toddler/pupil/school/nursery/playground/pram/teen/boy/girl/son/daughter) over all 4 generator fields × 24. **6 of 7 hits are the production term *"girl-group thirds"*** — false positives. The seventh is `pair_05_…` V2: *"to a daughter in a corridor at six."* **No age is given and the context (a nurse telling a patient's relative) reads adult by default.** I do not rule this a breach. I note it because the rule is written absolutely and this is the only human in 24 songs who could be read as a minor. A second, weaker implication: P01V4's *"A cardigan on a chair with a name tape on. / Somebody's mother will want that back."* No child speaks or acts anywhere in the run. |
| **P04: no death, no real record** | ✅ **CLEAN — and deliberately so** | T5's real shape involves a musician who **died**. P04 removes the death in every variation and says so explicitly: *"when he came back they kept it again"* (V1) · *"when he got his voice back they kept hers"* (V2) · *"The one on the sleeve had shouted his voice away"*. **Nobody stood in for a dead man; somebody stood up for a tired one.** The band, the singer, the record, the coast and the town are invented; the dedications state this in words. |

⭐ This is the dimension the run handled best, and it handled it by **changing the fact, not by
hiding it**.

---

# 6. THE SOMATIC READ — the part no script can do

⭐ **THE FLAT TEST** applied to the strongest and weakest line of every song, said out loud, flat, to a
person's face.

### P01 — SWING · the seating plan
**Moves.** The praise is Sunna-specific throughout — *which* bar, *whose* hands. *"Pam brought her own
knife for the icing from home. / She works with her elbows out. Leave her alone."* survives flat and
gets funnier. *"Row parks the cars and Row parks them tight. / He gets one more in than fits. Every
night."* survives. The wolf lands as an **event with an object and a verb**: *"I set it down and square
it to the fork. / I face it out, the same as all the work. / Then I read the next table."* — she does
something, once, and the band does not slow down. **Exactly the A1 clause working.**
**Weakest:** V4's *"Then she says goodnight and means it."* — *"and means it"* is the narrator vouching
for a feeling instead of showing it. It dies flat. One line.
**Verdict: the strongest set in the run. It is the only pair where the praise is consistently a person
doing a specific thing rather than an inventory of nouns.**

### P02 — SKA · the coach
**Moves, hard.** *"He calls us folks. He means it, too."* survives. *"The pen has got no light at all. /
The pen's the only thing that's right."* survives and is the reveal engine in six words. *"Nobody's
shouting. That's the worst."* is the best flat line in the run — it says the whole indictment without
raising its voice, which is the ska brief's own instruction.
**Weakest:** the four outros. Read consecutively they are one machine (§3). Individually they are fine;
as a set of four they are a formula the listener would never hear but the *writer* should have.
**V4 is the standout of the pair** — the singing coach, the woman who does the low bit wrong and loud,
the two blokes who stand up and give her the seat. That is appetite in an INDIGNATION song, which is
harder than either alone.

### P03 — REGGAE · the launderette · **NO WOLF**
⭐ **PELLY'S CARRIED DISSENT IS ANSWERED, AND THIS IS THE PROOF.** *"appetite is a verb. She must want
something and take it."*
- V1: *"And I take one mint, and then I take another one, slow. / I am not telling her, I am not sorry,
  the foil goes back closed."* — a taking, unapologised, with a specific hand movement.
- V2: *"There is a pound in this coat and there is nothing left to wash."* → *"Nothing inside it. I am
  running it anyway."* — she spends money on heat she does not need, on purpose, and says so.
- V3: *"And I am up. I am up before I have decided to be up."*
- V4: *"And I stop. I actually stop. In the middle of the road."*
**All four are verbs. None is pleasantness. This is not wallpaper.**
**Weakest:** V1's *"and it is lovely"* and *"and I look well"* — Sunna's own rule is *never "this is
amazing."* Two evaluations doing a noun's job.
**But:** §0. The four Verse 2s are one machine. **The pair's best writing and its worst defect are in
different sections of the same songs.**

### P04 — SUBLIME LINEAGE · the car park
**The idea is the best in the run.** A hole in a hook that a hundred people fill without being asked,
and the singer standing in it deliberately: *"So I keep my mouth shut through the good bit / and let the
whole car park hand it back to me."* That survives flat and it is a whole song's worth of feeling in two
lines. *"That isn't a thing that went wrong. That's the job. / That is the entire job."* survives.
**Weakest:** *"There's nobody answering me now"* — because I read it four times. On its own it is
strong; as the pair's only answer to RULING X it is a formula (§3, 2.17/comparison).
**And a real craft note:** the ALL-CAPS answer lines are ~40% of the sung-line count. `P04V3` measures
95 sung lines against a 78–110 target — but roughly a third of them are two-to-four-syllable echoes.
The gate cannot tell the difference between a line and an echo of a line.

### P05 — RAP-ROCK · the stairwell · **the one drive slot**
**The anaphora works and then it does not stop working.** *"The one who reads the pump and not the
screen. / The one who counts the sharps and counts them twice."* is the Catalog archetype done right —
each item is a competence, not an adjective. *"The one who never made you feel a fool / for asking
twice. So I asked twice instead."* is the best two-line unit in the pair.
The refusal lands: *"They can file a name. That is what a name is for. / So the handover is in jobs.
All the way down."* Flat, plain, no hush — **and the band does not slow down**, which the exclude field
enforces four different ways.
**Weakest — and it is structural:** V3 (`Twelve, Fourteen`) is a **counting song inside a run with a
one-number rule**. 84 sung number-words. The hinge (*the thirteenth floor*) has to compete with twelve,
fourteen, four, six, ten, eleven, nine, eight, three and six-fifteen. Say V3's chorus flat — *"Count
them all the way up. / Twelve, then fourteen, and my two boots in between."* — and it survives, but the
song around it has spent its numeric budget before the hinge arrives.
⭐ **The drive clause is honoured better than I expected.** V2 and V4 use **no** guitar swell at all, and
V4's only distorted instrument is **the bass**. *"If the guitars carry this song, the slot has failed"*
— it did not fail.

### P06 — ART-POP · the office · **NO WOLF**
⭐ **The best-varied pair by every instrument I ran, and the one that most obviously moves.**
*"I want the whole of it and I am having it."* · *"I want tonight and I am paying for it in tonight."* ·
*"I came back in and nobody asked me to."* · *"I want it in my chest. Get it into my chest."* — four
different appetites, four different verbs, all survive flat.
The money is answered four genuinely different ways: *"Fine. Nothing changes. / Then the nothing is mine
till they take the door off."* · *"Right. Then the carpet is the same carpet. / And I love this carpet,
and I am dancing on this carpet."* · *"Not this one. This one is loud and it has my name in it."* ·
*"Fen spent nothing, and this room is not the same."* **This is what "responded to, never recited"
looks like when it is actually done four times.**
V4's last verse is the best writing in the run: *"The low end gone for two beats and the feet keeping
it. / Somebody laughing on the beat, which is harder than it sounds."*
**Weakest:** V3's *"Whatever this building keeps, it keeps me loud."* — clever in a way that trips THE
REMARK TEST; a listener may notice the construction before feeling it.

### ⚠️ THE ONE THING THE SETTING AUDIT DID NOT ASK
The audit in `04_metaprompt.md` proves the six **rooms** are distinguishable. It never asks whether the
six **worlds** are. All six songs are set in Britain, in the same class register, with the same
narrator's diction — *bloke, mate, love, get told, mind how you go, carry on*. The beds are Jamaican
(×2), American (×2), Anglo-Jamaican (×1) and Israeli (×1). **Six rooms; one world.** This is the honest
residue of Marsalis's objection and the run never names it. Not a rule breach — no rule exists — but if
the answer to *"six costumes?"* is *"no, because the songs survive transplant"*, the follow-up is
*"then what did the six idioms buy?"*, and the run's own answer would have to be **the arrangements
only**.

---

# 7. THE SIX CARRIED DISSENTS — RULED

| # | dissent | ruling |
|---|---|---|
| 1 | **Adorno:** the six choruses must be measured **against each other** | ⚠️ **PARTLY CAME TRUE.** 24 distinct hooks; max chorus-vs-chorus token Jaccard **0.267**, all function words. **But two collisions the measure is too coarse to rank:** ⛔ **P05V3 *"Count them all the way up"* and P06V4 *"All the way up"* — one hook is a superset of the other, both AMBITIOUS, both eligible for the same top-3.** And P02V3 *"Somebody Signed It"* × P05V2 *"Whoever Signed It"* at title level. |
| 2 | **Hein:** no pair's distinctiveness may rest on microtiming | ✅ **DID NOT COME TRUE.** Microtiming appears in all 24 prompts as a house constant (+8–14 ms), but every pair's identity is a room, a person and a lyric device. Nothing rests on it. |
| 3 | **Marsalis:** a dedication does not make you fluent | ⚠️ **HALF-ANSWERED.** The transplant test he asked for returns **5 of 6 survive** — his gate is passed. The residue is §6's last note: six idioms, one world. His objection relocates rather than dies. |
| 4 | **Pelly:** the two wolf-free songs must be *appetite*, a verb | ✅ **DID NOT COME TRUE — emphatically.** P03 and P06 are the run's clearest verbs and its two least-templated pairs. See §6. **This is the run's best result.** |
| 5 | **Eno:** a system that has become a style is worse than no system | ⛔ **CAME TRUE, IN THE SHAPE HE NAMED.** The system is not "six unwavering grooves" — it is **RULING X's compliance grammar**. The run built a rule to make the unmeasurable checkable and the rule became a template in five of six pairs (§0, §3). *"Check that you have not just found a very comfortable new place to stand."* |
| 6 | **Bradley:** do not claim a lineage without the ledger | ✅ **MOSTLY DID NOT COME TRUE.** P01 (Domino, Barbie Gaye, Pimentel), P02 (Knibb, Drummond, Alphonso, Mittoo, Brevett, Jah Jerry, Rico Rodriguez, Bradbury), P03 (Carlton Barrett, King Tubby, Augustus Pablo, Cedric Brooks), P05 and P06 all cite the **specific fact** from `00` §2, not the vibe. ⛔ **P04 is the exception and it is the pair whose lineage the brief most explicitly promised to pay.** |

---

# 8. VERDICTS

| | verdict | headline reason |
|---|---|---|
| **P01** | ✅ **SHIP** | strongest writing; A2 at 0:38 documented; absence named at 4 positions with 4 syntaxes; 0.50 dup/comparison |
| **P02** | 🔧 **REPAIR** | four-way naming block + four-way outro formula; 1.83 dup/comparison = the known-templated control |
| **P03** | 🔧 **REPAIR** | Verse 2 is one machine in all four (lines 73-75 / 231-233 / 385-387 / 538-540); otherwise the best answer to a carried dissent in the run |
| **P04** | 🔧 **REPAIR** | 2.17 dup/comparison — **above** the negative control; no player credit; 0 verbatim Source-2 |
| **P05** | 🔧 **REPAIR** | ⛔ RULING A2 fails on the assigned absence (defect originates in `05`); fact and name both position-locked ×4; V3 numeric saturation |
| **P06** | ✅ **SHIP** | cleanest by every instrument (0.17 dup/comparison); appetite is a verb; the fact answered four different ways |
| **RUN** | 🔧 **REPAIR** | nothing is unshippable — 0 hard gate fails, 0 human-subject breaches, 0 real-name leaks — but **four of six pairs carry one defect class and every instrument the run owns reports it as clean** |

---

# 9. REPAIR BRIEFS — routed to a pair and a step

### 🔧 R1 → **P02 · step 09** (artist refinement; do not touch step 07 or the beds)
Two blocks are one machine across all four variations. **(a)** The RULING-X naming quatrain
(`:100-103, 280-283, 451-454, 616-619`): all four open *"[We had / There was / The downbeat played] a
downbeat…"* then *"we removed it"* then *"it hasn't/won't come back"* then *"your body still does it."*
**(b)** All four outros close *"And nothing [verb] the top/bar/beat. / And [X]. And [Y]."*
**Fix:** keep **one** variation's version of each. The other three must name the missing downbeat with a
different **grammatical move** — not a synonym. One could refuse to name it and let a *listener* in the
lyric name it; one could name it as an instruction (*"don't wait for it"*); one could name it only as a
consequence with no mention of the downbeat at all. **Do not solve this by changing nouns.**

### 🔧 R2 → **P03 · step 09**
The first three lines of Verse 2 are identical in structure in all four
(`:73-75, 231-233, 385-387, 538-540`), and the sung fact and the *"nothing is wrong"* reveal are locked
to the same section. **Fix:** move the naming to a **different section in at least two variations**
(the Bridge/Chant is unused for it in three), and rewrite two of the four so the departure is named
**without** the three-move sequence [instrument left → not coming back → the bass holds the gap]. The
writing here is otherwise the best-argued in the run; **do not soften the appetite to make room.**

### 🔧 R3 → **P04 · step 09 (lyric) and step 10 (dedication)**
**(a)** *"There's nobody answering (me) now"* + *"The bar goes by in time / you can count it out"* is a
two-line block in all four (`:128, 273, 446, 603`). Highest duplication in the run and above the
negative control. Keep it in **one**. **(b)** The dedication credits a class in all four. Either name a
player whose figure the prompt actually asks for (the boom-bap break's drummer, or the dub side), or —
if the class credit is the honest answer, which it may be — **say so once in `DEDICATIONS.md` and let
that file carry Nowell and Happoldt as `00_research_brief.md` §2 promised.** **(c)** Optional: put
*"detuned bliss"* into the style field case-exact; the behaviour is already there.

### 🔧 R4 → **COORDINATOR · step 05, then P05 · step 07**
`05_pair_assignments.md` STATE-IT-FIRST for P05 (*"the name is said **once**, early"*) contradicts
RULING A2 in the ICB (*"stated at least twice… no hole in the first 30 seconds"*). P05 complied with
its slot and answered the clock with an **unassigned** arrangement rest (`step07:281`, bars 76 · 101).
**Rule first, repair second.** Either (i) the tannoy pages twice, the second past 0:30 — cheap, and
`P05V4`'s intro already has a rest built for it; or (ii) **A2 is declared waived for P05 in `INDEX.md`,
in writing, with the reason.** ⛔ What is not acceptable is the current state, where a binding ruling is
recorded as met by a hole it was not written about.

### 🔧 R5 → **COORDINATOR · step 05** (structural, for the next run)
The orthogonality table asserts six absences at six architectural levels but never checks them against
the six **beds** or against each other's **executed** lyrics. Two consequences measured today:
**P02's assigned absence (the deleted downbeat) IS P03's bed (the one drop)** — the brief's own §2 T2
says so; and **P02V3's executed device (an unnamed signatory, everyone a function) IS P05's assigned
absence.** Both showed up as HOLE-SWAP and WOLF-SWAP failures, and neither was predicted.
**Add one line to the next assignment table: for each absence, name the pair whose bed already contains
it, and say why that is not a collision.**

### 🔧 R6 → **SHARED TOOLING · `scripts/validate_step.py`** *(outside this run; see HOLD 1)*
The LYRICS branch (`~line 311`) does **not** strip markdown code fences, while the MUSIC PROMPT branch
(`~line 279`) does. Reproduced on all six packages:

```
P01: validate=82  fence-lines counted as sung=2  true=80
P02: validate=86  ...=2  true=84      P03: validate=82  ...=2  true=80
P04: validate=88  ...=2  true=86      P05: validate=94  ...=2  true=92
P06: validate=85  ...=2  true=83
```

The two ``` markers are counted as **sung lines**, and `suno_lyrics_field_max` over-counts by ~10 chars.
`sung_lines` is therefore biased **+2 high on every correctly-fenced package** — a true 71-line lyric
reports 73 and escapes the ≤72 floor-hug flag. **No verdict in this run flips** (all 24 sit at 78–95).
⭐ This is the *same* defect `INSTRUMENT_LOG.md` §2 declared repaired, **one function lower in the same
file** — the fence-strip idiom now exists in three places and is missing from a fourth. P06 detected the
symptom independently and reported both numbers (`pair_06_…:713`) without diagnosing the cause; credit
to P06.

### 🔧 R7 → **RUN · `DEDICATIONS.md`** (pending, `RUN_STATE.md:20`)
`00_research_brief.md` §2 states in writing that Bradley Nowell's and Michael Happoldt's names *"belong
in `DEDICATIONS.md`."* The file does not exist. **The run's own promise is unkept until it does**, and
P04's four class-credits are currently the only payment on the lineage the whole slot is built from.
Honesty clause both ways: *we asked for the shape, not the recording* — never *"contains a sample of."*

### 🔧 R8 → **SELECTION · `TOP6_PASTE_READY.md`**
⛔ **P05V3's hook is *"Count them all the way up."* P06V4's hook is *"All the way up."*** A room chanting
one is chanting the other. Both are AMBITIOUS and both are plausible top-3 picks. **If both are
selected, one hook must change** — Sunna's law L makes the hook the deliverable, and two hooks that
collapse into each other cost the set a slot.

---

# 10. ESCALATED TO THE SCIENTIST — HOLD

**HOLD 1 · `scripts/validate_step.py` is measuring fenced packages wrong, today, in the repo.**
R6 above. It is **shared tooling outside this run's directory**, it biases a live gate by +2 lines on
exactly the packages that follow house format, and it is a re-occurrence of the defect the same run's
instrument log declared closed. Surfaced explicitly rather than repaired in place by a judge.

**HOLD 2 · `DEDICATIONS.md` and the T5 debt.** R7. The run wrote a promise into its own research brief
and has not yet kept it. P04 is the pair that inherits the debt and it credits a class. This is your
call, not mine — the class credit is honestly argued and may be right.

**HOLD 3 · `05_pair_assignments.md` contradicts RULING A2 for P05.** R4. A coordinator artifact
overrode a binding ruling and the override was never declared. The pair is not at fault. This needs a
ruling before P05 is selected from, not after.

**HOLD 4 · The zero-rejection question, turned on this run's own instruments.**
`measure_variations.py` (0 hard fails), `credit_sweep.py` (0 leaks) and `distinctiveness.py` tests 1–3
and 7 (0 over ceiling, 0 exact repeats) **all report clean on a corpus in which four of six pairs
answered the run's hardest ruling with a template.** `distinctiveness.py` says so about itself — its
own §6 declares tests 1–3 uncalibrated and its own control failed. **Test 7 is sound for what it
measures and I verified its separation (control 2 leaks vs ours 0), but what it measures is EXACT
repetition, and this run's formula is paraphrase.** The instrument that catches it did not exist
until this audit. ⭐ **`blockdup.py` and `skeleton.py` are now in this run directory; recommend `blockdup.py` be run at every
future daily, with its two calibration controls kept, because a number without a control is a
sentence.**

---

# 11. ⭐ WHAT I COULD NOT SEE

**Every instrument I used is textual. Nothing in this run has been heard. Nothing has been rendered.**

## The three claims the run itself says are invisible — do I agree?

⭐ **THE BRIDGE SEAT (Eno) SAID: *"ALL THREE OF YOUR CENTRAL CLAIMS ARE INVISIBLE TO EVERY INSTRUMENT
YOU OWN."* I agree, and I will sharpen it: the repair the panel prescribed made the claims READABLE
without making them TRUE, and that gap is where this run's defect lives.**

| claim | the cash-out | what I could actually check | what remains invisible |
|---|---|---|---|
| **THE ABSENCE** | *named in the lyric, in words, once* | that a sentence naming it exists, where it sits, and that five of six pairs used one sentence four times | ⛔ **Whether a listener experiences an absence at all.** A named hole is a *claim about* a hole. Bregman's precondition (two full cycles) is a claim about a render nobody has heard. **The word "gone" in a lyric is not a gap in a bar.** |
| **THE FEEL / THE STRUT** | *a named instrument entering or leaving at a named point* | that all 24 prompts carry it, generator-readable | ⛔ **Whether Suno plays it.** Hein's carried dissent is right and unanswerable from text: microtiming is the most-smoothed parameter in generated audio. All 24 prompts say +8–14 ms. **No text gate can tell you whether one millisecond of that survives.** L22, THE GRAIN LAW, is render-measured, and this run has no render. |
| **IDIOMATIC DEPTH** | *survives genre transplant* | I ran the transplant on the **words**, which is what the ruling asks | ⛔ **The transplant test measures whether the LYRIC is bed-independent. It does not measure fluency in the bed.** A song can survive being moved and still be a costume in its original clothes. Marsalis's objection is answered on the axis he was offered, not the axis he raised. |

## Specifically, and by name, the things no instrument in this report can check

1. **Does any of it move?** The FLAT TEST in §6 is my reading, out loud, in my own head. It is a
   judgement, not a measurement, and I have marked it as one.
2. **Whether the hole is heard as a hole or as "the pattern being different."** Bregman's threshold is
   the load-bearing assumption under RULING A2 and it is untested on this material.
3. **Whether the generator returns the bass in front of the guitars in P05.** P05's own artifact says
   this and it is right (`pair_05_…:end`). The exclude field names the failure class in words; naming a
   failure class is not detecting it.
4. **Whether P06's centre actually vacates**, or whether Suno renders a centred vocal and the whole
   spatial absence — the pair's entire assigned device — simply does not happen.
5. **Whether P04's empty answer bars stay in tempo** or get filled with a delay throw, ad-lib or crowd,
   which four exclude fields forbid in words and no gate can verify.
6. **Whether P02's downbeat stays deleted after 0:30.** *"A generator's most likely well-meant repair is
   to put a clap where the kick was"* — P02 said that itself (`pair_02_…:178`) and it is the sharpest
   line about instrumentation in the run.
7. **Whether SWAG happens.** *"An unwavering groove with a parade of specific timbres entering over it"*
   is a claim about a render. I can confirm the timbres are named at named points in text; I cannot
   confirm one of them arrives.
8. **Whether six idioms make one record or six costumes** — the venue answers this, not a script.
9. **Whether "Draymark" (P06V1), "Kestle House" / "Ferrow Street" (P06V3) collide with a real company or
   address.** I checked them against my own knowledge and found nothing; **I did not check the world.**
   That is a one-search job and it has not been done.

## What would be needed to actually check the three claims

- **For the absence:** `lofn-render-audit` under **THE BLIND RULE** — send the audio alone, first, to an
  audio-capable model and ask *"what is missing?"* If the listener names the wrong thing or nothing,
  the hole is a lyric, not a rest. **This is cheap and it is the only test that would settle it.**
- **For the feel:** `scripts/measure_render.py` on a rendered take — the microtiming is measurable off a
  waveform, which is exactly what the Hein seat was seated to say.
- **For idiomatic depth:** nothing textual will do it. It needs an ear that knows the idiom, and it
  needs the render.
- **For the templating defect I did find:** it needed no render at all — only an instrument that
  measures paraphrase instead of copies, and a calibration control in both directions. **It existed
  nowhere in this repo before today.** That is the finding I would carry forward above all the others.

---

## 12. METHOD — how to re-run everything in this report

```
python3 output/daily/2026-08-22/measure_variations.py     # 24/24, 0 hard fails, 1 flag
python3 output/daily/2026-08-22/distinctiveness.py        # tests 1-3 UNCALIBRATED by its own §6; test 7 sound
python3 output/daily/2026-08-22/credit_sweep.py           # 33 names, 0 leaks (dedication-derived only)
python3 output/daily/2026-08-22/blockdup.py               # NEW - paraphrase duplication, 2 controls
python3 output/daily/2026-08-22/skeleton.py               # NEW - positional skeleton; UNCALIBRATED, hits only
python3 scripts/validate_step.py --gate-report 10 output/daily/2026-08-22/pair_0N_step10_final_package_enhanced.md
```

⚠️ `blockdup.py` prints its own two negative controls every run (cross-pair, must be ~0; and the
2026-08-21 pair-04 corpus, known templated). **Read them before believing the numbers above them.**
⚠️ `skeleton.py` scores 0 on its negative control. **Its hits are evidence; its zeros are not.**
Positional maps in §3 and the child / real-name / number-word sweeps were run as one-off scripts whose
predicates are stated inline in this report; each is a `grep`-reproducible claim against a named file
and line.

*Independent QA judge, 2026-08-22. Verdicts: 2 SHIP · 4 REPAIR · run REPAIR. 8 repair items · 4 holds.
Every instrument I wrote is in this directory and every one of them was calibrated against a control
before I believed it.* 💜
