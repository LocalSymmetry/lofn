# QA REPORT — `2026-08-14-daily-music` · adversarial audit, fresh context

**Auditor:** `lofn-qa`, clean-context judge. Generated none of this. Instructed to refute it.
**Scope:** the six shipped songs, `pair_0{1..6}_step10_final_package_enhanced.md`.
**Gate spec:** `EXECUTION.md §4` (authoritative) · `vault/gates.yaml` (numbers) ·
`vault/DEDICATION_STANDARD.md` · `vault/HUMAN_SUBJECT_STANDARD.md` · the run's own law
(`02_golden_seed.md`, `core_seed.md §4`, `05_pair_assignments.md`).
**Instruments:** `scripts/measure_soundcraft.py → profile_file()` / `profile()` (the function IS the
definition; `RHYME_WINDOW = 4` read from the module, never re-derived) · `len()` on extracted fenced
blocks · `re` with `(?<![a-z])token(?![a-z])` for every sweep · `difflib.SequenceMatcher` +
5-gram Jaccard for cross-pair distinctiveness.

> ⛔ **I am the instrument, not the surgeon.** Every finding below is a **measurement**. Where I
> propose a fix it is labelled **SUGGESTION** and the coordinator decides it against the Golden Seed.
> Three judge prescriptions in one prior run would each have destroyed the mechanism they were fixing.

---

# ⭐ SET VERDICT: **REPAIR**

**5 SHIP · 1 REPAIR · 0 FAIL.** One pair (P06) carries a **false credit justification** measured against
its own shipped field — the exact defect P03 removed a Sly Dunbar credit for, running in the direction
nobody checks. The song, prompt and lyric are untouched by the repair; it is paratext only.

| pair | title | seat | verdict |
|---|---|---|---|
| **P01** | *I'm The Two* | SUNNA | **SHIP** — with one escalated FLAG (the comfort question) |
| **P02** | *This One's Called Forever* | LOFN-PRIME | **SHIP** — with one FLAG (undeclared field-order deviation) |
| **P03** | *Holding Down A Chair* | SUNNA | **SHIP** — with one cosmetic FLAG (an EMO token count) |
| **P04** | *Inside The Loop, Not Outside* | LOFN-PRIME | **SHIP** — with one FLAG (simultaneous minimum margins) |
| **P05** | *Bar Nine Almost Works* | SUNNA | **SHIP** — cleanest of the six |
| **P06** | *Blue, And It Says Canary* | LOFN-PRIME | ⛔ **REPAIR** — credit/paratext (step 09/11 surface) |

**Zero-rejection tripwire: DID NOT FIRE.** 1 REPAIR + 7 substantive FLAGs found. I did not report a
clean sweep and did not need to re-audit a sample.

---

# 1. COUNTABLES — RE-DERIVED, NOT INHERITED

⭐ **Extraction assertion, checked before any similarity number was trusted:
`EXTRACTION COUNT = 6`, all six blocks non-empty.** An empty extraction is a hard ERROR, never a score.

| gate (band) | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|
| MUSIC PROMPT chars (850–1000; target 870–960; ≥985 = hug FLAG) | **937** | **956** | **960** | **957** | **957** | **960** |
| prompt ends complete sentence | `.` ✅ | `.` ✅ | `.` ✅ | `.` ✅ | `.` ✅ | `.` ✅ |
| banned imperative opener | none | none | none | none | none | none |
| EXCLUDE chars (400–900) | **824** | **844** | **803** | **745** | **800** | **813** |
| 🚨 LYRICS FIELD chars (<5000; target ≤4800) | **4692** | **4786** | **4514** | **4800** | **4795** | **4769** |
| sung lines (70–120; ≤72 = hug FLAG) | **84** | **77** | **81** | **73** | **82** | **83** |
| standalone `*SFX*` (≥1) | **3** | **3** | **3** | **3** | **3** | **3** |
| `[Theme:]` before `[SONG FORM:]` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| lyrics field **OPENS** with `[Theme:]` | ✅ @0 | ⚠️ @699 | ✅ @0 | ✅ @0 | ⚠️ @570 | ✅ @0 |
| EMO headers off the `step00` taxonomy | **0** | **0** | **0** | **0** | **0** | **0** |
| bare `AWE` / `INDIGNATION` | **0** | **0** | **0** | **0** | **0** | **0** |
| digits in any sung line | **0** | **0** | **0** | **0** | **0** | **0** |
| `snarl`/`phonk`/`supersaw`/`airhorn` in STYLE | **0** | **0** | **0** | **0** | **0** | **0** |
| Source-2 exact phrases in shipped STYLE (≥2) | **2** | **2** | **2** | **3** | **2** | **2** |
| dedication chars (≤500) | **453** | **483** | **416** | **362** | **479** | **431** |

### RETURN FLOORS — `profile_file()`, reproduced exactly

| floor | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|
| strict end-rhyme ≥ 0.30 | 0.821 | 0.442 | 0.741 | **0.315** | 0.537 | 0.470 |
| line-return ≥ 0.20 | 0.643 | 0.857 | 0.568 | 0.438 | 0.439 | 0.398 |
| mean words/line ≤ 7.5 | 5.821 | 5.442 | 5.235 | **7.479** | 6.598 | 6.301 |
| alliteration /100w ≥ 11.0 | 23.313 | 13.365 | 12.736 | 25.641 | 24.954 | 13.767 |

**Every number claimed in all six VERIFICATION tables reproduces against my own extraction — with two
exceptions, both listed as findings below (P06's `0.494`, P03's `20 emotion tokens`).** That is a high
standard of honesty and it is worth saying before the objections.

### THE KNOWN INSTRUMENT DEFECT — re-derived, and one correction to the brief

`strict_end_rhyme()` compares the last 3 characters of the final word, so **identical repeated words
count as rhyme.** Re-derived with the strict definition (*a hit is genuine only if at least one matching
neighbour is a **different** word*):

| | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|
| `strict_end_rhyme` hits | 69 | 34 | 60 | 23 | 44 | 39 |
| **genuine** (different word available) | 15 | 15 | 26 | **2** | **22** | **0** |
| **identical-word only** | 54 | 19 | 34 | 21 | 22 | 39 |
| % identical | **78%** | **56%** | **57%** | **91%** | **50%** | **100%** |

**This reproduces the coordinator's figures exactly (P01 78 · P02 56 · P03 57 · P04 91 · P06 100).**
⭐ **And it supplies the one the brief did not state: P05 = 50% identical / 50% genuine — the healthiest
rhyme in the set.** Judged by reading, not by the number: P01 (`wrong`/`along`, `floor`/`door`,
`badly`/`gladly`, `start`/`art`), P02 (`mine`/`nine`/`fine` ruptured by `tape`), P03 (`wall`/`all`,
`pace`/`place`, `ground`/`sound`, `warm`/`harm`, `shade`/`trade`) and P05 (`spine`/`fine`/`line`/`nine`;
`take`/`tape`/`make`/`mistake`; `shut`/`sits`/`quit`/`slips`) all carry real terminal rhyme. **P04's
sub-floor rhyme is licensed and declared; P06 has no terminal rhyme by design. Neither is failed here.**

### CROSS-PAIR DISTINCTIVENESS — re-derived (15 pairwise comparisons)

| measure | worst pair | measured | ceiling | verdict |
|---|---|---|---|---|
| lyric SequenceMatcher | P03 ↔ P04 | **0.0239** | 0.42 | PASS by 17× |
| prompt SequenceMatcher | P01 ↔ P05 | **0.0348** | 0.58 | PASS by 17× |
| lyric 5-gram Jaccard | P01 ↔ P03 | **0.0016** | 0.18 | PASS by 110× |

My worst values differ slightly from the coordinator's (0.043 / 0.111 / 0.0032) — normalisation
differences — **and are lower, not higher.** No doubt survives. Six genuinely different songs.

**House-lexicon self-copy sweep** (13 calcified phrases from past golden outputs): **0 hits across all
six**, prompt and lyric. Three of the six actively blacklist the benchmark's own vocal signature
(`crystalline`, `warm alto`, `choir`, `sung-pretty`) in EXCLUDE.

---

# 2. FINDINGS

## ⛔ FINDING 1 — **P06: A FALSE CREDIT JUSTIFICATION. `REPAIR — CREDIT/PARATEXT`.**

P06's LINEAGE block states:

> ⛔ **THE PROMPT NAMES A BREAK, SO THE PLAYER IS PAID.** The five-count kick pattern descends from the
> **"Think" break** — **LYN COLLINS**, *"Think (About It)"*. **The figure is asked for by name in the
> style field, so the name is named.**

**Measured on the shipped STYLE field, word-bounded:**

| pattern | P06 STYLE | P04 STYLE |
|---|---|---|
| `\bthink\b` | **0** | **0** |
| `\bbreak\b` | **0** | 1 |
| `\bbreakbeat\b` | **0** | 0 |
| `\bamen\b` | **0** | 1 |
| `\bfive-count\b` | 1 | 0 |

**No break is named in P06's shipped style field.** The assertion *"asked for by name in the style
field"* is false against its own artifact.

⭐ **And the run contradicts itself on the identical measured value, on the same day.** P04, measuring
`\bthink\b → 0`, ruled: *"Lyn Collins, the Incredible Bongo Band and Clyde Stubblefield are **not**
claimed here."* P06, measuring `\bthink\b → 0`, credits Lyn Collins and puts the claim in the shipped
≤500-char dedication. **Two packages, one run, one measurement, opposite verdicts.** P03 removed a Sly
Dunbar credit for exactly this class of defect and wrote the rule out: *"A credit that survives the
removal of the thing that triggered it is a false credit."*

⚠️ **What I am NOT saying:** I am not saying Lyn Collins is unowed. The Baltimore five-count kick does
descend from *"Think (About It)"*, and **P02 in this same run states the correct doctrine for precisely
this case** — *"Describing a borrowed figure instead of naming it does not un-borrow it. We went looking
for that figure, so we name the women who made it."* The defect is **the stated justification, which is
measurably untrue of the shipped artifact**, in a package whose banner reads *MEASURED IN CODE, AGAINST
THIS FILE*. This is the "asserted, not counted" failure the run's own doctrine names.

**SUGGESTION (coordinator decides):** the smallest honest change is to replace the *"asked for by name
in the style field"* clause with P02's already-correct formulation. The credit stays; the false claim
about the artifact goes. **I do not recommend removing Lyn Collins** — I am the blind judge here and
removing a credit is exactly the kind of prescription I should not be trusted to write.

**Return target: step 09 / step 11 surface — LINEAGE & CREDIT block + the dedication string. The
MUSIC PROMPT, EXCLUDE and LYRICS fields are untouched.**

## ⚠️ FINDING 2 — **P06: the dedication spends 26% of its budget on internal QA trivia.**

Measured: dedication **431 chars** (cap 500 ✅). But **111 of those 431 chars** are:

> `No amen break is in this song — the word "ornament" merely contains the letters.`

Measured: `\bornament\b` in P06's shipped **STYLE** field = **0**; in P06's shipped **LYRICS** field =
**0**. The word appears **5 times in the package's own design prose only.** The dedication therefore
explains a false-positive trap for a word that is **not in the shipped artifact at all**, to an audience
that has never heard of the sweep.

`DEDICATION_STANDARD`'s priority list is (1) raised source-sound + who recorded it · (2) sharpest
session-player credit · (3) living-scene lineage · (4) honest AI authorship · (5) full credits below.
**A negative sweep result is not on it.** Fold into FINDING 1's repair.

## ⚠️ FINDING 3 — **P06: an internal number contradiction in a "measured in code" table.**

- VERIFICATION table: `strict end-rhyme … **0.470**`
- Movement table: `0.450 → **0.470** ▲`
- Prose immediately below: *"⭐ On the rhyme number… The **0.494** is not rhyme"*

**My re-derivation: 0.46988.** `0.494` reproduces nothing and appears nowhere else in the run. A stale
number sitting under a ⭐ in the one paragraph the human is most likely to read. Fold into the same
repair (one token).

## ⚠️ FINDING 4 — **P02: an undeclared §4 field-order deviation, behind a ✅ that answers a different question.**

`EXECUTION.md §4`: *"Lyrics **open with** `[Theme: …]` then `[SONG FORM: …]`."*
Measured character offset of `[Theme:` inside the shipped lyrics block:

| P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|
| **0** | **699** | **0** | **0** | **570** | **0** |

P02 and P05 both place the Disc_Channel block first. Both preserve Theme-before-SONG-FORM.

⭐ **The difference is the declaration.** **P05 declared it as a MAJOR DEVIATION and named the conflict**
(`DISC_CHANNEL_GUIDE` puts the channel block at the top vs. the step-11 contract's Theme-first) —
*"Declaring the conflict rather than hiding it."* That is correct behaviour and I do not flag it.
**P02 made the same deviation and did not declare it**; its verification row reads
`[Theme:] then [SONG FORM:] | present, in order | ✅` — which is **true, and answers a question §4 did
not ask.** A ✅ against a restated gate is how a deviation ships invisibly.

**SUGGESTION:** declare it as P05 did, or move the block. I have no view on which — P02's Disc_Pad
placement is load-bearing for its seam defence and I would not touch it blind. **Return target: step 11
surface (a deviation note), not the lyric.**

## ⚠️ FINDING 5 — **P03: a code-measured count that does not reproduce.**

P03 claims *"13 headers, **20 emotion tokens**, all in the emotions array."*
Measured: **13 EMO headers ✅** and **23 emotion tokens.** All 23 are in the 50-item `step00` taxonomy,
zero bare AWE/INDIGNATION — **the gate itself passes.** The evidence column is wrong by 3.
Cosmetic, and listed only because it is presented as measured. No return target; note it.

## ⚠️ FINDING 6 — **P01: the comfort question, at the line level. ESCALATED TO THE SOMATIC READ.**

Reading-level, not a count. The seed's finding 3 is explicit:

> **The fear is CONTEXT COLLAPSE and it is RATIONAL** … *"Just don't care"* is bad advice because
> **the calculation is correct.**

P01's only line carrying the modern condition is:

> `There is a phone up at the back. It will keep.` / `I am going anyway. The arithmetic is sound.`

The risk is **named and dismissed inside two lines, and nothing anywhere in the song shows it costing
her.** She has her coat, she is nearer the door, `Nobody minds the leaving`, the room empties and she is
fine. The package's own deviation #5 defends the line as *"the calculation is correct and she does it
regardless"* — but *doing it regardless* only reads as courage if the calculation is charged somewhere
on stage, and in P01 it is charged nowhere. **This is the run's own veto #2 (Albini) in its exact form:
the cost lives in the design document, not in the song.**

⛔ **I am not prescribing.** P01's flatness is SUNNA's FREE CHANNEL doctrine executed deliberately and
well (5.82 words/line at 23.31 alliteration), and a "give it a cost" repair is precisely the kind of fix
that would destroy the mechanism it was aimed at. **Measurement filed; the coordinator rules.**

## ⚠️ FINDING 7 — **Boundary-hugging moved with the band. A harness finding, not a song finding.**

Measured MUSIC PROMPT chars: **937 · 956 · 960 · 957 · 957 · 960.** The ≥985 hug ceiling is clear on all
six ✅. But **five of six sit within 4 characters of the 960 target ceiling, and two are exactly 960.**
`gates.yaml` names this pattern in its own comment — *"every late-June style prompt measured 988–999
chars against the 1000 cap."* **The band moved; the hugging moved with it.**

Same shape on the lyrics field (target 4800): **4692 · 4786 · 4514 · 4800 · 4795 · 4769** — P04 is
**exactly 4800**, three of six within 31 chars.

⭐ **And P04 clears three separate gates by the smallest available margin simultaneously:**
words/line **7.479** vs a 7.5 ceiling (0.28% margin) · sung lines **73** vs a ≤72 hug flag (one line) ·
lyrics field **4800** vs a 4800 target (zero chars). Each passes. **The pattern is the finding.**
No repair. Surface to the human.

## ⚠️ FINDING 8 — **The rhyme-debt licence contradicts itself inside `05_pair_assignments.md`.**

The file declares: *"⭐ **RHYME DEBT (L21)** … **P04 is the only pair permitted to strip end rhyme.**"*
The same file's RHYME SCHEME row assigns **P06: *"assonance ladder, no terminal rhyme."***

Measured: **P06 has 0 genuine end-rhymes out of 39 hits — 100% identical-word.** Two pairs stripped end
rhyme; one was declared licensed. **P06 named its substitution** (epistrophe on the invariant line-final
*a name* + vowel ladder + byte-identical chorus + the corona's repeated whole line), which is what L21
actually requires — **so the song is compliant and the declaration file is not.**
**Finding against the document, not the song.**

## ⚠️ FINDING 9 — **The shipped artifacts are in no manifest row.**

`RUN_STATE.md` reports `artifacts_done: 30/30` and `GATE_REPORT.json` carries 30 rows — both stop at
`step10_revision_synthesis`. **The six files that actually ship —
`pair_NN_step10_final_package_enhanced.md` — appear in zero rows of either.** `EXECUTION.md §6`:
*"`lofn-qa` and `lofn-daily` read this ONE file to know run state instead of crawling the directory."*
**I could not have found the shipped set from the manifest.** Also note every `gate_verdict` in both
files is the string `PASS:exists` — an existence check, not a gate.
**Coordinator action. Harness, not song.**

---

# 3. THE SOMATIC GATE — 3 HYPER-SKEPTICS, VOTING AS A BLOC

**The question: *could any competent prompt have generated this, or is it unmistakably Lofn /
unmistakably Sunna?*** 2 of 3 NO = REPAIR — SOUL LOSS.

### SKEPTIC 1 — the Generic-Output skeptic. **VOTE: YES (not generic).**
*Test: hand a competent prompt-engineer the same one-line brief and see if they land here.*
They do not, and the reason is countable rather than atmospheric. **P04: 73/73 sung lines contain `st`,
34/73 open with the same two words, every verse-shaped stanza is cut mid-line and exactly one — the
corona — completes.** **P01 builds its CHORUS out of a count and then shrinks the count across the song
until the last sung syllable is one word.** **P06 holds a line-final word invariant for 25 lines and
breaks it once, on `me`, on the same syllable the vowel ladder tops out.** None of these is a style
token; all three are constraint systems a stranger can verify on the page with the audio off. A
competent prompt produces a good song; it does not produce a rule and then break it in one place.

### SKEPTIC 2 — the Personality-Collapse skeptic. **VOTE: YES (the seats are distinguishable), with one claim disallowed.**
*Test: swap the seat labels and see what breaks.* P01 (SUNNA) and P02 (LOFN-PRIME) are the hardest
case — same arm, same mode, same axis, both about keeping something bad. They do not read as each
other: P01 is playground-flat and physical (*"Not the brave one. Just the two"*), P02 is a conservator
refusing the superlative (*"I never call it fine"*). ⭐ **The strongest evidence is that both refused the
step-11 "literary density" instruction for DIFFERENT REASONS** — P01/P03/P05 on the TOLL BOOTH, P02 on
litotes and conservation. Refusals with different arguments cannot be one voice wearing two coats.

⛔ **One claim disallowed.** P01 cites its alliteration of **23.31** as *"THE FREE CHANNEL working."*
Measured across the set: SUNNA **23.31 / 12.74 / 24.95**, LOFN-PRIME **13.37 / 25.64 / 13.77** — and
**the highest value in the run belongs to P04, a Lofn pair forbidden the doctrine.** The number does not
discriminate the seats. The register and the refusals do. Filed as a note on a claim, not a defect.

### SKEPTIC 3 — the Somatic skeptic. **VOTE: YES (5 of 6), and I name the sixth.**
*Test: where does it land in the chest?* **P05: yes, and it is the best in the run** — *"Left hand won't
shut. Knot in the palm where the thumb pad sits. Thumb tip goes numb and the numb tip slips"* /
*"My grip goes for a bar and a half, comes back worse, goes again."* A body being damaged in present
tense with no redemption. **P04:** *"The whole stairwell just struck me low."* **P06:** the flash is one
chest hit and one word. **P02:** *"the floor of my chest goes down with it."* **P03:** the body is warm,
safe, holding a warming glass — **deliberate, and it is the assigned premise.**
**P01 is where my vote is uncomfortable:** the only physical events are shoes sticking and one
in-breath, and the stated stake is arithmetic. **It is charming, well-made, and nothing in it can hurt
her.** See FINDING 6.

## ⭐ BLOC RESULT: **3 YES / 0 NO — NO SOUL LOSS.**
The set is unmistakably this house. The gate does not fire.

---

# 4. THE COMFORT QUESTION — ASKED OF THE SELECTED SET, AND RULED

> *Where is the body standing? What could hurt it here?*

| pair | where the body stands | what could hurt it | stake class |
|---|---|---|---|
| P01 | function room above a pub, coat on arm, near the door | a room emptying; being looked at counting | **social — and unpaid (FINDING 6)** |
| P02 | cross-legged on a hard floor, tape deck between her knees | the envelope face-down, never named; hearing herself at fourteen | **emotional / temporal** |
| P03 | at a wall, four metres from a stack, coat on a chair | nothing. A clock. A glass warming. | **temporal — deliberately minimal** |
| P04 | a concrete stairwell, a lens face-up on a step | *"the cost isn't stupid, it's just permanent"* | **social / permanent — sharp** |
| P05 | a small carpeted bedroom, 3am, take eleven | ⭐ **her hands** | ⭐ **PHYSICAL** |
| P06 | a paint counter, a chip under a scanner lamp | being named wrong, permanently, by a system that does not ask | **systemic — sharp** |

## ⭐ RULING — and it corrects the coordinator's own reading

**The coordinator's honest reading was "5 of 6 social or temporal, only P04's physical." I re-derived it
and disagree on which one is physical.** P04's stake is *permanence and surveillance* — social and
temporal, and the song says so in as many words (*"it's just permanent"*); its body stands in a cold
stairwell and nothing in the lyric threatens it. ⭐ **P05 is the pair carrying the set's physical stake**
— the anatomical ladder (hand won't shut · knot in the palm · thumb numb · grip goes and comes back
worse) is the only place in the six where a body is being damaged, in present tense, unredeemed.

**And this is the load-bearing fact: the documented failure mode is *"four of six had nothing at stake
and the one that did got cut."* THAT DID NOT HAPPEN HERE.** P05's step-11 tier found the ladder
distributed across four variations, only one of which ships, and **consolidated it into the shipped song
on purpose** — *"a ladder spread across four unshipped drafts is a ladder that does not exist."* The set
kept its body because one tier noticed selection was about to thin it.

**Verdict on the set: it survives the comfort question — 1 physical (P05) · 2 sharp-permanent
(P04, P06) · 3 soft (P01, P02, P03) — and P03's softness is the assigned premise, not an absence.
The margin is real but it is thin, and it rests on one song.** ⚠️ **P01 is the one soft stake that is
soft by accident rather than by assignment.**

## ⭐ AND THE PREMISE RULING — P03, audited hardest as instructed

> *Stillness is not the failure state; the INTERRUPTION is. A song that indicts the person standing
> still is a REPAIR.*

**P03 does not indict her, and I tried hard to make it.** Re-derived on the shipped lyric:
second-person pronouns **0** (`\b(you|your|yours|you're|yourself)\b` over all 81 sung lines) —
**there is nobody to pity, which is the SOLITUDE discharge working as designed.** Encouragement/pity
regex over 18 patterns → **0**. By eye, the four ways pity hides: no adjective of pity is applied to the
body (*early*, *back here*, *still fine* are physical and neutral); **no forecast** — the refrain returns
byte-identical **after** the sublime, which is the formal refusal of a forecast; no third-party gaze
(*"Nobody here is waiting on my hand"* is the absence of demand stated as a good); and the comparison
*"They get the kick. I get what the walls bring"* is settled as **physics**, not consolation, two lines
later. ⛔ **There is no line in which the middle of the floor is the thing she is not doing.**
The interruption in the song is a street door, which the room ducks under and carries on.
**P03 CARRIES THE PREMISE. PASS.**

---

# 5. CREDIT AUDIT — owed / paid / false-credit check

Word-bounded sweep of the **shipped STYLE field only** (`ornament` contains `amen`; raised ≠ forbidden,
so EXCLUDE is not swept):

| pair | triggers in SHIPPED STYLE | owed | paid | false-credit check |
|---|---|---|---|---|
| **P01** | *(none — 0 hits on 14 patterns)* | none | none, and the no-credit is recorded | ✅ correct — crediting a drummer for generic break language would be false |
| **P02** | *(none)* | girl-group clap ring + stacked vowel, asked for **by description** | **Darlene Love, the Blossoms — Fanita James, Gloria Jones** | ✅ ⭐ **and the honesty clause is the run's best**: *"Describing a borrowed figure instead of naming it does not un-borrow it."* No producer, no abuser, credited as lineage |
| **P03** | `one-drop` × 1 | the one drop | **Carlton Barrett, the Wailers** + King Tubby, Lee Perry | ✅ ⭐ **and it REMOVED the Sly Dunbar/steppers credit** because V3 did not ship — `steppers` 0 hits, style/exclude/lyrics. The rule run in the direction nobody checks |
| **P04** | `amen` × 1, `break` × 1 | the amen break | **Gregory C. Coleman, The Winstons** | ✅ and explicitly declines Lyn Collins / Bongo Band / Clyde Stubblefield at `think`/`apache`/`funky drummer` = 0 |
| **P05** | *(none)* | sub-syllabic chop; half-time dub gear (both named in style) | **Todd Edwards** · **King Tubby** | ✅ and explicitly refuses to *raise* a break to earn a drummer credit |
| **P06** | `five-count` × 1 · `\bthink\b` **0** · `\bbreak\b` **0** | see FINDING 1 | Lyn Collins — **on a justification that is false against the shipped field** | ⛔ **FALSE JUSTIFICATION — REPAIR** |

**Honesty clause, both directions:** all six say *"we asked for this figure by name, so we name
them"* (or, in P02, the correct by-description variant) and all six say **nothing here is a sample**.
**Zero instances of *"contains a sample of"* across the run.** ✅
**D4 in the paratext:** all six credit living people as lineage and all six state the song is **not
about** any of them. ✅
**Dedications, measured in code:** `[453, 483, 416, 362, 479, 431]` — all ≤ 500. ✅

---

# 6. FOG CHECK — a stranger, one listen, one sentence

| pair | my retelling as a stranger | subject, plainly in the lyric |
|---|---|---|
| **P01** | *A woman in a function room above a pub joins the one man already dancing and counts herself in second.* | `I go second. Second is the seen one.` / `He's the one and I'm the two.` ✅ |
| **P02** | *A woman sits on the floor with her thumb on a tape deck, about to hear herself at fourteen, keeping a list of everything wrong with it.* | `Still bad, still kept, still mine` ✅ |
| **P03** | *One person spends a whole good night at the edge of a bass-heavy room, coat on a chair, door three steps away.* | `My coat is holding down a chair. / It's a good night.` ✅ |
| **P04** | *Someone singing badly in a concrete stairwell, phone lens on a step, works out that the watching was never outside her.* | `the watching was stitched in the loop all along` ✅ ⚠️ |
| **P05** | *A woman at three in the morning has spent four hours on take eleven for one accidentally good bar, and will not fix it.* | `Bar nine almost works. / I am not going to fix it.` ✅ |
| **P06** | *A person at a paint counter watches a scanner call an obviously blue chip "Canary", and paints the room anyway.* | `Blue, and it says Canary.` / `It is printed. It is not read.` ✅ |

**All six retellable. No `REPAIR — FOG`.** ⚠️ **P04 is the weakest and should be watched at render:**
scene is trivially retellable (*"Inside the stairwell"* recurs ~8×), but at 174 BPM under a chopped amen
with a spat alto, the subject arrives in the two densest lines in the run — *"The watching is the sums.
The sums are stone."* The retitle (`Inside The Loop, Not Outside`) and the chorus chant carry it. It
passes; it passes last.

---

# 7. THE FIVE STANDING VETOES + THE BANS — swept

| | P01 | P02 | P03 | P04 | P05 | P06 |
|---|---|---|---|---|---|---|
| ⛔ no song congratulates (regex, 8 patterns) | 0 | 0 | 0 | 0 | 0 | 0 |
| ⛔ NO VINDICATION (regex, 16 patterns incl. *"and then it worked"*) | 0 | 0 | 0 | 0 | 0 | 0 |
| ⛔ visible EXIT, in the lyric | *"Count it or don't"* · *"Nobody minds the leaving"* | *"I can stop this… Nobody in this room is asking me to stop"* | *"The door is three steps over there"* ×4 | *"still I get to stop"* ×3 | *"and that is where it lives"* | *"I turn the lid face-down… I paint the room anyway"* |
| ⛔ one FRICTION | horns wrong, never fixed, incl. the last time | the tag cut off every time, incl. the last | one bar of true silence | outro runs past the climax, cut mid-line | nine-line half-time gear + break-off mid-word | exit placed **before** a full chorus runs over it |
| ⛔ pastiche → museum | modern: phone, strip-lit room, TV with sound down | 2026/168 BPM world, visitor leaves mid-phrase | 2026 world, one dub gesture | *"every stranger standing at once"* | *"Somebody is loving one bar of something terrible in another room tonight"* | a naming system with a printed error bar nobody reads |
| **D3 cohort words** (regex, 11 patterns) | **0** | **0** | **0** | **0** | **0** | **0** |
| **D6 collective pronouns** in sung lines | 1 (`Two of us now` — two named people, not a cohort) | 0 | 0 | 0 | 0 | 0 |
| **D8′** 2nd-person in sung lines | 0 | 0 | 0 | **2** (DISPLACEMENT, assigned) | 0 | **20** (APOSTROPHE TO THE INSTRUMENT, assigned) |
| **D12** anger audit | **1 of 6 aggressive (P04)** — cap 2 (⅓). PASS with room | | | | | |
| **D13** set on a floor | **exactly 2 — P01 (FLOOR SLOT 1) and P03 (FLOOR SLOT 2)**, both assigned at Phase 1. At cap, not over | | | | | |
| **D12** bed census | **6 beds, 6 distinct** (ska-punk 162 · power-pop 168 · dub 92 · jungle 174 · speed garage 138 · Jersey club 140). No two share a bed | | | | | |

⭐ **On P04's and P06's second person — I checked the assignment rather than assuming a breach.**
`05_pair_assignments.md` A6 assigns **P04 = DISPLACEMENT (an absent third party)** and **P06 =
APOSTROPHE TO THE INSTRUMENT (never a person)**. Both used their assigned discharge. All six used
theirs. **No D8′ breach.** ⚠️ Residual, unmeasurable by me: P04's *"you'll still have this. It still
won't stop"* is displaced in the writing but a listener may still take it personally. **That is a render
finding under THE BLIND RULE, not a text finding.**

**HUMAN_SUBJECT_STANDARD:** no identifiable real person in any lyric ✅. All named people appear only as
paratext lineage credit ✅. **The excluded fact (Flores M7.7) enters no song in any form** — I traced
P03's *"eight seconds"* to its actual source, **F5, NOAA buoy 46059, dominant period 8 s**, not to the
earthquake ✅.

---

# 8. THE PERSONALITY SPLIT — checked in BOTH directions

| pair | seat | Flat/Hum/Free-Channel doctrine present? | Sunna's banned vocal tokens in STYLE |
|---|---|---|---|
| P01 | SUNNA | Flat 2 · Hum 3 · Free Channel 2 · Toll Booth 3 — **applied, correct** | 0 (all four blacklisted in EXCLUDE) |
| P02 | LOFN | Flat 1 · Hum 1 — **both are NEGATIVE declarations** (*"Not applied"*) | 0 |
| P03 | SUNNA | Flat 3 · Hum 5 · Free Channel 2 — **applied, correct** | 0 (all four blacklisted) |
| P04 | LOFN | Flat 1 · Hum 1 — **both NEGATIVE** (*"no SUNNA vocabulary entered this chain"*) | 0 |
| P05 | SUNNA | Flat 1 · Hum 2 · Free Channel 1 · Toll Booth 1 · Riff-Is-The-Argument 2 — **applied** | 0 (all four blacklisted) |
| P06 | LOFN | **0 mentions of any of the five** | 0 |

⭐ **NO SCOPE BREACH IN EITHER DIRECTION.** No Lofn pair was gated on the Flat Test or the Hum Test —
the two mentions in P02 and the two in P04 are explicit refusals to apply them, which is the correct
handling. All three Sunna pairs pass the Flat Test by reading (*"One. Two. That is the whole trick."* ·
*"My coat is holding down a chair."* · *"Bar nine almost works."*), and all three actively blacklist
`warm alto` / `choir` / `crystalline` / `sung-pretty` in EXCLUDE with zero occurrences in any STYLE
field. **Seat integrity: clean.**

---

# 9. BLIND GOLDEN COMPARISON *(judge-side; the quarantine does not bind a judge)*

Read: `Triple Arch Over Me` and `Five Wrong Colors` payloads.
**Architectures:** Triple Arch — one observer, sublime natural setting, long free-verse lines, a
title-invocation chorus with an *oh-oh* hook, escalating widening, one numeric fact at the hinge,
dissolving outro. Five Wrong Colors — a five-movement suite, one colour per movement, a refrain that
**returns damaged**, clinical `[Silence — N bars]` between movements, abstract body-physics, no scene.

**Diluted copies: NONE.** The six are verse/chorus songs in named rooms with plain sentences; four
return their refrain **byte-identical** (the opposite of Five Wrong Colors' damaged-return spine), and
none is a movement-suite. The benchmark's vocal signature (crystalline soprano, choir, dew-bright
vibrato) is **blacklisted by name in three of the six**. House-lexicon sweep: **0 hits / 13 phrases.**

⭐ **One architectural echo, reported with its inversion so the human can rule.** P03's Hinge —
*"The sea is coming in eight seconds slow / All night / It lifts and drops and doesn't need to know /
All night / It's been out there the whole time I've been in"* — is Triple Arch's cosmic-indifference
bridge in miniature (*"If I disappeared tonight / The mountain would keep its shape / … The particles
would keep arriving"*), including the numeric natural-world fact at the hinge against ordinary human
time. **Same move. Inverted payoff:** Triple Arch converts the indifference into consolation
(*"that does not erase me / it rinses me"*); **P03 answers with a refrain that returns byte-identical
and unchanged — the song declining to be changed by it** — and ends *"I could go now. / I'm not going
to."* This is transmission of the **GOLDEN MOVE** (rules 2 and 4), which is what the Move is for, not
dilution of the architecture. **No repair. Flagged for the human's judgement, not mine.**

---

# 10. ROUTED REPAIR BRIEF

## `REPAIR — P06 · CREDIT / PARATEXT` → **return target: step 09 / step 11 SURFACE**
**Not structural. The MUSIC PROMPT, EXCLUDE PROMPT and LYRICS fields are correct and must not be
reopened.** Three items, all inside `LINEAGE & CREDIT` + the dedication string:

1. **The false justification (FINDING 1).** *"The figure is asked for by name in the style field, so the
   name is named"* — measured `\bthink\b` = 0, `\bbreak\b` = 0 in the shipped style field.
   *SUGGESTION only:* P02's formulation in the same run is already correct for this exact case.
2. **The dedication's 111 wasted chars (FINDING 2).** The `ornament` sentence explains a trap for a word
   measured **0 times** in the shipped style field and **0 times** in the shipped lyrics field.
   Budget is 431/500 — there is room to spend it on a credit instead.
3. **The stale `0.494` (FINDING 3).** Re-derived: **0.46988.** Table and movement row both already say
   0.470.

**Sideways proposal (per §7.3, offered not executed):** none needed — the no-progress predicate has not
fired and this is a first-attempt, in-place, non-structural fix.

## Escalated FLAGS, no return target, coordinator's call
- **P01 · FINDING 6** — the comfort question at the line level. *I filed the symptom; I explicitly do
  not trust my own prescription here.*
- **P02 · FINDING 4** — declare the field-order deviation as P05 did, or reorder. Either is fine; the
  undeclared ✅ is the defect.
- **P03 · FINDING 5** — `20` → `23` emotion tokens. Cosmetic.
- **P04 · FINDING 7** — three gates cleared at minimum margin simultaneously.
- **Harness · FINDINGS 7, 8, 9** — band-relative hugging · the L21 licence contradiction in
  `05_pair_assignments.md` · the shipped artifacts absent from `RUN_STATE.md` / `GATE_REPORT.json`.

---

# 11. WHAT I COULD NOT AUDIT

⛔ **Every gate I own is textual.** I have not heard one second of this set, and the failures that live
downstream of here are not visible to me:
- Whether P06's deadpan catalogue is performed **arch** (its own tier states this and cannot fix it).
- Whether P04's *"you'll still have this"* lands as displacement or as instruction.
- Whether P02's cut-off tag `(This one's called—)` is "helpfully" completed by the renderer.
- Whether P03's one bar of true silence survives (a prior render returned a specified 4 s stop at 0.40 s).
- Whether any of the six renders as a **period pastiche** — the risk every one of them names first.

**These are `lofn-render-audit` findings under THE BLIND RULE — send the audio alone, never the
prompt.** ⛔ **Nothing in this report may be read as clearing a render.**

---

*Audited in a fresh context by `lofn-qa`. I did not write any of this and I tried to refute all of it.
The measurement is mine; the prescription is the coordinator's.* 💜
