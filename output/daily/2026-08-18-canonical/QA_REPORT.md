# Lofn QA Report — `output/daily/2026-08-18-canonical/` (MUSIC · 24 songs)
## SUNNA `v.2026.08.R3 + R3.1` · adversarial clean-context judge · 2026-08-20

> **Scope.** I wrote none of this. The coordinator's mechanical re-stat (prompt/exclude/lyric chars, sung
> lines, BPM, Theme + SONG FORM presence, Disc channels, cross-pair similarity, anger audit, club audit)
> is **not redone** — it is cited and trusted. Everything below is the class of defect a counter cannot
> see. Nothing was rendered, published or spent. `.RUN_LOCK` untouched. No pair file edited.
> **This report prescribes; it does not operate.**

---

## VERDICTS

| Axis | Verdict |
|---|---|
| **Pipeline Integrity** | ✅ **PASS — CANONICAL.** 36/36 artifacts on disk as separate per-pair files. Steps 06·07·08·09·10 exist for all six pairs. **The NO-SKIP rule does not fire.** This is the predecessor's blocking failure, repaired. |
| **Package** (modality contract) | ⛔ **REPAIR REQUIRED.** One false credit propagated into four generator fields; a 4/4 seam-clause failure in P04; one surviving remark-test epigram; one cross-pair radiant collision. |
| **Human-Subject** | ✅ **CLEAR.** Every person in all 24 lyrics is invented (Mo, Dez, Rhi, Fitz, Nell, Bear, Sal, Tam, Row, Ig, Vee, Jax, Del, Bry, Ness, Pip, Bex, Cass, Roo, Ollie). Real names appear only in dedications, as historical or public professional lineage credit. Zero identifiable private individuals. |
| **OVERALL** | ⛔ **REPAIR** |

**Not FAIL** — the run is canonical and the writing is strong. **Not SHIP** — a factually false instrument
model sits in four Suno style fields, and two of the R3.1 repairs did not reach the position they were
aimed at. **Not HOLD** — the defects are specific, cheap and routed, not a quality judgment.

**Zero-rejection tripwire (`EXECUTION.md` §7.5):** not triggered. This report carries 6 routed repairs
and 9 flags.

---

## THE CENTRAL QUESTION: DID R3.1 WORK, OR DID THE FORMULA MOVE AGAIN?

### Both. And I can name where it moved, and measure it.

### ✅ What R3.1 repaired — every named target, hit

| Predecessor defect | This run | Evidence |
|---|---|---|
| 10 of 24 songs closed on tea/toast/soup/chips/kettle | **0 of 24** | Word-boundary sweep of all 24 lyric fields: kitchen · kettle · toast · tea · chip shop · hot drink · domestic interior = **0 hits**. Six assigned rooms, all distinct, all used. |
| One pair signed off all four variations with one noun-swapped sentence | **0 pairs do this at the closing line** | All 24 closing lines are distinct on material. 5 of 6 pairs ran and reported the swap. |
| Two radiants in two different ARMS were the same radiant | **Radiant kinds are registry-assigned and distinct**; ⛔ but see finding F3 — the collision recurred one level down |
| **7 of 16 wolves were status reports** | **16/16 are events; 15/16 cost her something she cannot take back** | Scored line by line below. **This is the run's biggest genuine gain and it is in the writing, not the filing.** |
| Seam carried on mix adjectives in 3 of 6 pairs | **6/6 name instruments and players absent from the rest of the record** | Verified in the actual style fields, not the notes. |
| A relative as a radiant origin | **0 of 24** | P01 caught one in its own draft (step 09, cut C3: *"somebody's dad has his camera out"* → *"a bloke with a camera"*) — **R3.1 firing inside the pipeline, as designed.** |
| Three incompatible EMO header conventions; 3 of 6 pairs named no emotion | **One convention across all 24**, every header a specific embodied emotion | Format contract in `05_pair_assignments.md` worked. |
| P06 dedication inverted the Tony Allen credit onto Fela Kuti | **Repaired.** The clause sits on Allen. | Fact-checked: Allen was drummer and musical director of Africa '70. Correct. |
| P04 carried two unverifiable drummer names | **Removed.** Zero individual names; an honest role credit plus *"we could not confirm who played it, and a wrong name is worse than none."* | Programmatic proper-noun check: zero non-sentence-initial capitals. **This is the model.** |

**The return-swap test itself works.** P01's step 09 ran it on the actual text, **found two real
failures and repaired them** — cut C8: *"This and V1's draft ending were the same sentence in different
words. **Caught by the test, not by reading.**"* That is an instrument doing its job. Which is exactly why
the rest of this section is about *where the instrument is pointed*, not whether swap-testing works.

---

### ⛔ F1 — THE FORMULA MOVED INTO **THE SEAM**. This is the finding.

**Every single pair's seam block is more self-similar across its four variations than the song around it.
Not one pair escapes it.**

| pair | seam-vs-seam similarity | whole sung body | **SEAM TEMPLATE INDEX** |
|---|---:|---:|---:|
| P01 | 0.300 | 0.183 | **1.64×** |
| P02 | 0.330 | 0.247 | **1.34×** |
| P03 | 0.341 | 0.142 | **2.40×** |
| **P04** | **0.674** | 0.285 | **2.36×** |
| P05 | 0.392 | 0.223 | **1.76×** |
| P06 | 0.233 | 0.184 | **1.27×** |

*(mean pairwise `SequenceMatcher(autojunk=False)` over the six variation-pairings, sung lines only,
headers/frame/SFX stripped. Mean index **1.79×**.)*

Read the writing and the number is obvious:

**P04 — four seams, one block, four times.** SFX (*a hand or a bar hits something twice*) → a count
phrase → *a second kit arrives* → **`[Name] has got the baritone`** → **`Everybody clapping [somewhere]`**
→ the count phrase again.
> `Two, three, four. / There's a second kit down in the pit. / Bex has got the baritone. / Everybody clapping on the floor. / Two, three, four.`
> `That's the count. / Second kit comes in behind. / Cass has got the sticks up. / Everybody clapping on the deck. / That's the count.`
> `Two. And two. / Second kit comes in behind. / Roo has got the baritone. / Everybody clapping in the room. / Two. And two.`
> `That is the one. / Somebody's kit is on the tailgate. / Ollie has got the baritone. / Everybody clapping round the vans. / That is the one.`
Line 5 is the identical sentence in **4 of 4**. Line 4 is the identical sentence in **3 of 4**.

**P02 — the same seam event described four times.** Line 2 of each seam is one sentence, rephrased:
> `There is no saxophone in a brass band.` · `That is not a brass band instrument.` ·
> `It does not belong in this band.` · `It is the one horn that does not belong in here.`

**P03 — one shape, four fillings.** `[doubled imperative] / [instruction] / [doubled imperative] /
[instruction] / [doubled imperative] / [line beginning "Both …"] / [imperative]`:
> `Hold it. Hold it. … Both arms locked and laughing. / Hold it.` ·
> `Hold. Hold. … Both thumbs on the tape. / Hold.` ·
> `Keep going. Keep going. … Both hands, all the way out. / Keep going.` ·
> `Take it. Take it. … Both hands, that is it. / Take it.`

**P05 — seven lines, four times.** SFX (wah clavinet) / an observation / `They said [clause]` /
**a one-or-two-word echo interrogating the clause** / the instrument leaves / `They said [clause]` /
her flat answer. The interrogating echo fires in the identical slot in all four:
`Seamless for who` · `Removed from where` · `Concludes` · `Which relationship`.

#### Why the seam, and not somewhere random

The seam is the **most heavily specified and least falsified object in the whole run.** The pair briefs
name it four separate times — *ONE SEAM* · *instruments and people, never mix adjectives* · *name a player
not otherwise in the record* · *she must do something across it* — and the doctrine tests it **zero**
times for variation. `sunna.yaml` pillar 5 mandates seam count and placement and attaches no instrument at
all; R3.1's swap test names **the closing line**; R3's names **the wolf line**.

So the pairs did exactly what a well-specified, unfalsified requirement produces: **each wrote one sentence
that satisfies every clause, then reused it four times with the nouns changed.**

> **This is R3.1's own root-cause sentence, one turn further on.**
> R3.1 says: *what is not assigned gets defaulted, and the default is the strongest prior.*
> The 2026-08-20 corollary, measured here: **what is specified but never varied gets templated — and the
> more clauses a rule attaches, the tighter the template, because the sentence that satisfies all of them
> is nearly unique.** P04 has the most clauses to satisfy in its seam and the highest template index.

---

### ⛔ F2 — THE RETURN-SWAP TEST IS DEFEATED BY FRAME-SHARING, AND BY THE CHORUS IMMUNITY

**I ran the swaps myself, across pairs and within them. Here is what the test cannot see.**

**(a) It detects noun collision, not frame collision.** P04's four outros are seven lines of one machine:
[location] / `[Name]'s van/engine running` / [a named crew member does a thing] / [detail] /
**`Nobody [X]. / Nobody [Y].`** / [one-word hook]. The `Nobody X / Nobody Y` couplet is in **4 of 4**.
Swap any two and the referents break — *Jax* is not in V2, the *floor with tape on it* is not in V4 — so
the pair ran the swap, saw breakage, and reported **PASS**. The report is honest. The test is blind.

**(b) Four of six pairs end on their hook, and R3 grants hooks blanket immunity.**

| | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|
| closing line is… | image | **hook** | gesture | **hook fragment** | **hook fragment ×3** | **shouted one-word hook** |

> `The late one!` `Second cornet!` `He can, he can!` `That's the half!`
> `Hands on.` `On three.` `Heads up.` `Doors open.`
> `NEXT!` `EASED!` `BENCH!` `GOES!`

`sunna.yaml` rule 4: *"A byte-identical chorus needs no defence and no note."* R3.1's only cross-song
instrument is *"swap the closing lines."* **For P02, P04, P05 and P06 those two rules meet on the same
string, and the immunity wins.** The one cross-song test in the doctrine lands on the one text it is
forbidden to question. Only **P01 and P03 wrote four endings**; the other four pairs wrote one ending
shape four times and passed the test that exists to catch precisely that.

**(c) The distinctiveness ceiling is applied cross-pair only — and the four songs a listener actually
hears back to back are the four LEAST measured against each other.** P04 published its own within-pair
numbers: **0.393–0.426** on the lyrics field. **The lyric ceiling is 0.42.** Two of its six pairings sit
at or above it (V1–V2 = 0.425, V2–V4 = 0.426) and nothing gated them, because no within-pair ceiling
exists. Coordinator cross-pair worst: 0.036.

**(d) P04 also found that step 11's own additions corrupted the instrument.** `[Theme:]`, `[SONG FORM:]`,
five `Disc_` channels and the EMO header shape are **mandated shared vocabulary**. Measured on the field,
P04's four songs read 0.393–0.426; measured on the sung body only, **0.199–0.265**. *"A distinctiveness
gate pointed at the lyrics FIELD is now measuring the format contract."* That is correct, it is the R3.1
lesson arriving in the instrument rather than the writing, and it should be fixed before the number is
trusted again.

---

### ⛔ F3 — THE PREDECESSOR'S EXACT RADIANT DEFECT RECURRED, AND ITS ROUTED REPAIR WAS NEVER BUILT

The predecessor found *"two radiants in two different ARMS are the same radiant"* and routed **R9: the
distinctiveness gate must measure the radiant against the other 23 radiants directly, not the whole lyric
bodies.** `GATE_REPORT.json` is a file-existence manifest — path, byte size, sha, `exists:non-trivial`.
**It contains no radiant measurement, no seam measurement and no per-song content measurement at all.**
R9 was not implemented, and the defect returned:

> **P04-V3:** `Somebody gave me a key to the paint frame and never asked for it back.`
> **P06-V4:** `That one time I had none, somebody gave me a working one and I never said thank you.`

Two isolated pairs. **Both AMBITIOUS**, so they would sit adjacent in the shipped three. Identical opening
four words, identical structure: *somebody gave me a [thing] and [nobody ever settled it]*. Cross-pair
4-gram scan confirms `somebody gave me a` is one of only two content-bearing 4-grams shared across pairs.

**Two further cross-pair collisions the 0.036 similarity number cannot see:**

- ⛔ **`wet to the elbow`** — P01-V1's **title and chorus hook**, appearing verbatim as an all-caps room
  response in **P06-V1**, line 2. A listener hearing the set hears the beach song's name shouted in the
  market-stall song.
- ⛔ **EMO vocabulary has converged across isolated pairs.** `Same Voice As The Joke` appears byte-identical
  in **P05 ×4 and P06 ×3**; P01 carries `Same Voice She Told The Joke In` ×4; P03 carries the phrase in its
  cues. **`Plain As Giving Directions` is byte-identical in P01-V2 and P06-V1 — two pairs, two different
  arms.** Four isolated pairs independently converged on the doctrine's own sentence as a tag, which
  proves the ICB is a stronger determinant of the writing than the isolation is.

---

### ⚠️ F4 — THE RADIANT REGISTRY ASSIGNS A GRAMMAR, SO IT MANUFACTURES THE COLLISION

Three of six radiant kinds are **sentence types**, not semantic territories, and each produced four
sentences with one grammar in one slot:

| pair | assigned kind | what four songs produced |
|---|---|---|
| **P04** | *a permission she was given once* | `An old stage manager **told me once I could** be the one to say when` · `The chippy who taught me **said I was allowed to** break what I make` · `Somebody gave me a key…` · `The night porter **told me once I could** stay as long as I turned the key` — **3 of 4 share the frame**, all four are the longest line in their stanza, all four break the metre identically, all four sit in the final line of verse 4 |
| **P05** | *a document she signed standing up* | **`I signed it standing up`** — **five byte-identical words in all four songs.** `…with a pen on a string` · `I signed it with my finger, standing up, for a parcel` · `…at a kiosk between two shops` · `…in a room with no chairs in it` |
| **P06** | *a debt she is repaying to a stranger* | **3 of 4 are the same radiant** — an anonymous man in the trade did her a kindness for nothing (a free repair · free teaching · a free working machine). Only V2 (a woman paid her pitch) differs in kind. |

P05's is the sharpest case: **the predecessor QA rejected P01's byte-identical `Tide's at nine` ×4 as
"one event stated four times."** P05 does the identical thing here with `I signed it standing up` — and
this time it is **the radiant**, the most load-bearing line Sunna writes.

P02 and P03 show the fix already exists: their radiant kinds are *territories* (*a place she keeps coming
back to* · *an instruction she ignored*), and their four instances share no frame. P02's are the best in
the run — `I know which chair in here does not wobble` proves the relationship without naming it.

---

## THE WOLF-COST SCORE — **16 / 16 EVENTS · 15 / 16 WITH A COST SHE PAYS**
*(predecessor: 9/16)*

Criterion applied, `sunna.yaml` R3.1 verbatim: *"does it cost her something in the song? If she can state
it without doing anything about it, it is weather."*

| | wolf | cost, inside the song | verdict |
|---|---|---|---|
| **01-V1** | `So I put the blade in and I take the bank down` | destroys her own moat to feed the turret | ✅ EVENT |
| **01-V2** | `I turn the stick west. Put it on theirs. / And my good side goes under with nobody there.` | gives away the last photographic plate — **the seed's own argument, paid for** | ✅ **EVENT, strongest in P01** |
| **01-V3** | `Bucket goes down at the turret and it's gone. / Wall stops where it stopped` | abandons the north wall; whole build goes forty minutes early | ✅ EVENT |
| **01-V4** | `so I put my elbow straight through the tower` | destroys Saturday's turret — the one they came back for | ✅ EVENT |
| **04-V1** | `I called it over them. I missed / the last time anybody took a bow.` | loses the last bow of the run | ✅ EVENT |
| **04-V2** | `So I put the bar through my own part` | swings first on her own build, in front of everyone | ✅ EVENT |
| **04-V3** | `I say start with the good one, Vee. / I could have kept it to the last.` | sends the three-week flat out first; forgoes the hours with it | ✅ EVENT *(softest in P04)* |
| **04-V4** | `I go back in and walk the floor… I lock it, and I leave it` | misses the car park — the thing the whole song is about | ✅ EVENT |
| **05-V1** | `I made the argument out loud / To a cleaner with a mop and a kind face / He said love, I only do the floors / He's right` | spends the argument and gets nothing | ✅ **EVENT — the run's best three lines** |
| **05-V2** | `I took them off the phone myself / Now the only copy is theirs` | deletes her own copies. **Irreversible, and hers.** | ✅ **EVENT — the strongest wolf in 24** |
| **05-V3** | `I said thank you out loud, in an airport, at night` | thanks the machine that kept her; spends her dignity | ✅ EVENT *(subtlest)* |
| **05-V4** | `the arguing was done in a queue with my coat over my arm / And I was the one doing it` | loses the account, the record and the argument | ✅ EVENT |
| **06-V1** | `This one's the last. Drawer comes out.` | spends the last part in the drawer | ⚠️ **EVENT, cost displaced** — the price is named as *"the next one in with this gets nothing"*, i.e. borne by a hypothetical stranger, not by her. **The one soft wolf in 16.** |
| **06-V2** | `Down goes the shutter on a market day. / DAY'S TAKINGS — GONE!` | closes on a Saturday, loses the takings and the queue, pays for the part | ✅ EVENT |
| **06-V3** | `Own part, out, in the palm, still warm. / WARM, AND IT GOES IN HIS!` | kills her own working machine | ✅ **EVENT — cleanest execution in the run** |
| **06-V4** | `Labour is all that is on it. / IT? THE PART IS NOT ON IT!` | eats the part cost **and** refuses the credit | ✅ EVENT |

**⛔ P02 AND P03 CARRY ZERO WOLF — VERIFIED INDEPENDENTLY, 8/8 CLEAN.** No clock, no cost, no invoice,
no named pursuer, no shadow in any last verse. The two closest calls, both cleared: P02-V3's
`And I am not learning it` is the assigned angle (open admiration for a skill she is not acquiring) and is
cheerful refusal, not loss; P03-V2's `the smear is in the picture now` is the assigned angle (it came out
wrong and it is better) and is embraced, not paid for. **Wolf ceiling: 4 of 6, inside the two-thirds cap.**

---

## THE SEAM — INSTRUMENTS AND PEOPLE

| pair | seam personnel | in the record elsewhere? | **does she act across it?** |
|---|---|---|---|
| P01 | three answering voices · castanets · hand-played floor-tom triplet | no | ✅ 4/4 — answers the queue and puts the blade down to do it by hand · hands over the good tool · presses the last grey on · hands out trowels |
| P02 | alto saxophone · trombone in unison · Hammond shuffle · rimshot | no — **and a saxophone in a brass band is the joke, and it is in the LYRIC** | ✅ 4/4 — claps late himself · gets up · turns her chair round · takes her coat off the chair. **Best-integrated seam in the run.** |
| P03 | real kit cross-stick · Hammond bubble · trombone four bars late · tenor sax · second pianist · a flat third voice · tambourine from across the room | no — **and the personnel VARIES across the four**, the only pair that does this | ✅ 4/4 |
| **P04** | second drum kit · baritone saxophone, four bars behind | no | ⚠️ **4/4 WEAKEST FORM.** The only thing she does across all four seams is **count the band in** (`Two, three, four` · `That's the count` · `Two. And two.` · `That is the one`). That is an act on the arrangement, not in the room. Everything else in the block is description of the arrangement arriving. **A seam that only describes has nothing to vary — which is why P04's template index is the run's highest.** |
| P05 | 1972 wah clavinet, a guest in for one bar, **who never returns** | no | ✅ 4/4, but speech acts only (`Seamless for who`) — the columns swap and she notices |
| P06 | second rhythm guitarist · two congas · shekere · snare off the one | no | ✅ **4/4 STRONGEST.** She solders the last part · pulls the shutter down and drives to the ring road · **opens her own machine and takes the part out** · hands it over with the ticket |

**6/6 name material rather than mix adjectives — the predecessor's 3-of-6 failure is fully repaired.**
THE GRAIN LAW is satisfied: nothing here asks Suno to hold a mix distinction for eight bars.

---

## THE REMARK TEST — devices that announce themselves

| # | Where | Text | Ruling |
|---|---|---|---|
| 1 | **P05-V4** Stage | `I made you a dance record about this / That is the wrong container for it / I know exactly what it is / I'm doing it anyway` | ✅ **LICENSED.** The Container Clause, spent once, in the voice that sang the joke. Correct, and the best moment in P05. |
| 2 | **P05 all four** Stage | `I'm calling it that anyway` · `I'm going to keep calling it that` · `I'm doing it anyway` | ⚠️ **FLAG.** The clause is a once-per-set license; the pair spends the *shape* four times, in the identical slot, with the identical em-dash count-in. |
| 3 | **P05-V2** Pre-Chorus 2 | `I have never been so seen / And I would like it back` | ⛔ **FAIL — REPAIR.** Constructed epigram: antithesis, pivot on one word, punchline rhythm. **This is the identical defect the predecessor QA failed in the identical pair and the identical variation slot** (`I'm not insulted. I'm appraised.`). Routed repair R7 did not land. R3 rule 1 is literally *depth lives in the situation, never in the sentence.* |
| 4 | **P05-V1** Seam | `They said the transition will be seamless / Seamless for who` | ⚠️ FLAG. A zinger — and *seamless*, in a run whose central doctrine is THE SEAM, reads as a wink whether one was intended or not. |
| 5 | **P02** chorus, line 6 | `Say it again for the back row.` · `Louder for the ones at the back.` · `One more time so he hears it.` · `Say it so the front row hears you.` | ⛔ **The predecessor's crime, relocated.** One sentence noun-swapped four times — sitting inside a byte-identical chorus, where the doctrine forbids looking. |
| 6 | **step 11's own additions** | `[Theme:]` · `[SONG FORM:]` · five `Disc_` channels | ✅ **Do NOT trip the remark test** — metadata, never sung, invisible to a listener. ⚠️ But they corrupted the distinctiveness instrument (F2d). Process repair, not a writing defect. |

**Cleared on inspection:** P03-V1 `in a shape a camera cannot make` (a plain description of a slit-scan
artefact, not an aphorism) · P06 `Copper's a road and the road came apart` (the trade's own voice) ·
P06 `Eleven minutes is eleven minutes` (the assigned tautology; it *refuses* meaning rather than supplying
it) · P04-V3 `Jax cut the corner off the good one. / It is face up on the dashboard.` (the best unannounced
image in the run).

---

## THE REGISTER NEVER MOVES · THE BAND DOES NOT KNOW — ✅ **CLEANEST GATE IN THE RUN**

- **Zero hush, drop-out, reverence or a-cappella instructions in any of the 24 lyrics fields.** Full sweep:
  three hits, all benign — `kids have gone quiet` (a description of children) · `He breathes like it is
  nothing` (a description of a trombonist) · P03's `[Break … one full bar of rest]`, which is the **assigned
  sonic device**, is **filled by the room with handclaps** (a hole with hands in it, not a hush), is tagged
  `Held Breath With A Grin`, and lands after the whole verse — **not on the radiant**, which the pair
  deliberately buffered.
- **Every one of the 24 EXCLUDE fields actively bans the failure mode** — `reverent hush`, `band drop-out
  under the final chorus`, `a cappella last chorus`, `dramatic pause before the key line`, `hush before the
  last chorus`, `band drop-out under the decision`, `half-time breakdown`, `key change`, `tempo change`.
- **The hard lines are tagged for stillness, not for weight:** P01 `Same Voice She Told The Joke In` ×4 ·
  P04 `lead level, band straight through` ×4 · P06 `arrangement unchanged` ×4 · P05 `Same Voice As The
  Joke` ×4.
- ⚠️ **One flag: P05 hands its most load-bearing moment to SPEECH in all four** (`[Stage] — lead speaks
  straight out over the riff`). `sunna.yaml`: *sing-speak is now ONE COLOUR, not the default.* One colour
  used in every song of a pair is a pair-level monochrome. Not a violation. Worth watching.
- ✅ **No negations of any kind in any of the 24 STYLE fields.** Full word-boundary sweep: **0 hits.**

---

## THE SEED'S TWO ROT POINTS — ✅ **BOTH HELD**

**(a) The tide gets a time and no adjectives.** P01 is the only pair carrying a tide. Every instance:
`the water was out past the poles` · `that's the water coming up under my hand` · `Nine, I tell her. The
water's in at nine.` · `Water's at the poles and the poles are the mark` · `wet coming round the far side`
· `the wet's getting near` · `Water's at the far post now, not the near` · `while the water comes up`.
**Zero agency verbs. Zero adjectives. Zero second person. Zero moralising.** Weather with a schedule,
exactly as frozen. P01's step 09 shows the discipline was deliberate: it cut *seaward* → *waterside* and
*wreck* → *muck it up* to keep the excluded event's vocabulary out entirely.

**(b) "It is good, and it also melts" — neither is the reason for the other.** No line in 24 makes one
fact the reason for the other. The two closest approaches both **answer** the rot point rather than
committing it: P01-V4's `I came back because Saturday wasn't enough of it` attaches *because* to her
appetite, never to the ending's value; and P06-V1's `Eleven minutes is eleven minutes` is a tautology
whose entire job is to refuse the limit any meaning at all. P04-V3's `I could have kept it to the last. /
Three weeks that one took.` states two adjacent facts with no connective — the seed's own instruction,
executed.

---

## DEDICATIONS AND CREDITS

**All six under cap, all six counts independently reproduced:** P01 437 · P02 483 · P03 469 · P04 491 ·
P05 463 · P06 473 (cap 500). Every count matches the pair's own self-reported number.

**⛔ ONE FALSE CREDIT — P05.** `Stevie Wonder on a **Hohner Clavinet D6**`. Wonder played the
"Superstition" riff on a **Hohner Clavinet model C**; Reverb's Clavinet history flags "D6" as the standard
misconception. **The wrong model propagates out of the dedication into all four P05 SUNO STYLE fields**
(`1972 Hohner Clavinet D6 through a wah pedal` — lines 22, 170, 324, 474). Everything else in that credit
is true: Wonder, 1972, the wah chain, Ernst Zacharias as the Clavinet's designer.

**⚠️ ONE SOFT ATTRIBUTION — P01.** `The hand-played floor-tom triplet … is Hal Blaine's 1963 figure.`
Blaine is unquestionably the right person for 1963 LA girl-group session drumming, but his documented 1963
figure is the kick-and-snare "Be My Baby" pattern; no source attaches a floor-tom triplet to him.
The standard's test is not *"is this false?"* but *"could a reader take this as a claim we did not make?"*
— and they could. Hedge it the way P03 hedges Winston Grennan (*"whose kits carry the one-drop"*).

**✅ Everything else verified TRUE**, including the two credits this audit was specifically pointed at:
- **P06 / Tony Allen — CORRECT, and the clause is on Allen.** `TONY ALLEN, the drummer who built that
  placement in Fela Kuti's band, and OGHENE KOLOGBO, the tenor guitarist who played the interlock against
  it.` Allen was drummer and musical director of Africa '70; Kologbo played tenor guitar in the same band.
  **The predecessor's inversion is repaired in the shipped text.**
- **P04 — CORRECT and exemplary.** Zero individual names; zero non-sentence-initial capitals; the twin-kit
  glam lineage is accurate (minor imprecision only: 1973 is the Glitter Band's formation year, the twin-drum
  setup dates to mid-1972). *"We could not confirm who played it, and a wrong name is worse than none"* is
  the honest refusal the standard asks for. **This is the model for the whole system.**
- P02 Rico Rodriguez (trombone) and Dick Cuthell (**cornet**, exactly right) · P03's six rocksteady credits,
  all true and all correctly hedged, drummers named first.

**⛔ ARTIST NAMES IN GENERATOR FIELDS: ZERO.** Exhaustive scan of all 24 STYLE + 24 EXCLUDE + 24 LYRICS
fields + 120 `Disc_` channel lines. The only capitalised non-sentence-initial tokens are acronyms, key
names, the genre word `Afrobeat`, and instrument designations (`Hammond`, `Wurlitzer`, `Hohner Clavinet
D6`) — sanctioned as instruments, not artists. `amen` fires as a substring of **`lament`** only, never as
the break.

**BREAK / DRUMMER RULE: NO DEBT OWED.** Word-boundary sweep of all 24 style fields for named breaks
(`amen`, `funky drummer`, `apache`, `think`, `impeach`, `winstons`, `breakbeat`, and 13 more): **zero
hits.** Both pairs that raise a named *drum figure* nonetheless put the drummer at the head of the credit
— P03 (Grennan, Malcolm) and P06 (Tony Allen). The clause is honoured beyond its letter.

---

## THE DAILY RULES — ✅ ALL PASS

| Rule | Result |
|---|---|
| **Tri-source declared before any artifact** | ✅ `00_research_brief.md` §TRI-SOURCE, binding on all six. S1 = F21 Atlantic City sand sculpture · S2 = F10 five verbatim Bandcamp phrases · S3 = F07 the radiant |
| **Tri-source honoured** | ✅ S1 in all 24 · S2 ≥2 exact imports in all 24 · S3 radiant present and late in 24/24 |
| **3 + 3 arms** | ✅ ACCESSIBLE P01–03 · AMBITIOUS P04–06 |
| **NEWS ≤ 3** | ✅ exactly 3 (P03, P05, P06) — at ceiling |
| **Emotional duality** | ✅ AWE ×4 (P01, P02, P04, P06) · INDIGNATION ×1 (P05) · *between* ×1 (P03) |
| **Six distinct verse architectures** | ✅ genuinely six — chant-first / catalogue-narrowing-to-one-item / procedural verse-instructs-chorus-reports / escalation-by-bodies / two-column told-vs-true / call-and-response-throughout-with-chain-rhyme |
| **Absolute exclusion (Kariba ferry)** | ✅ **0 hits in all 24 lyric fields, 24 style fields and 24 exclude fields.** Sweep: kariba · ferry · capsize · rescue · lake · drown · missing · body · disaster · wreck · shipwreck · sink · overboard, all word-boundaried. Every grep hit is prose commentary. **The closest approach in the run is P05's `tugs`** — unambiguously aircraft pushback tractors, listed beside *ground equipment* and *gate leases*. Cleared, and worth naming. |
| **Bed uniqueness · anger audit · club audit** | ✅ 6 distinct beds · anger 0/6 · club 0 (coordinator-measured) |
| **One-fact rule** | ✅ 24/24. One numeric fact per song, at the hinge, spelled in words: `nine` · `one plate` · `forty minutes` · `six` · `bar nineteen` · `ninety bars` · `twenty past` · `nine carriages` · `two hundred yards` · `three feet` · `a hundred and six windows` · `ninety seconds` · `two hundred screws` · `three weeks` · `four in the morning` · `eleven years` · `ten past three` · `seven years` · `eight cents` · `eleven minutes` · `three calls` · `seven years` · `four pound fifty` |

---

## ICB INTEGRITY

- `CREATIVE_CONTEXT.md` present · 79,841 bytes · `icb_sha c754a9bf88467787` · cited by every pair.
- `(after ` attribution tags: **36 lines, ≥40 distinct attributions** — above the `==18` tripwire. This
  reads as a *richer* roster (three panels A/B/C plus doctrine citations: Achatz, Willis, Brandi, Atget,
  Ellena, Fisher, Clayton, Parton, Greenwich, Rodgers, Perry, Pelly, Kode9, Notaro, Carson, Wilson,
  Albini, Brown), not thread loss. **NOT `REPAIR — THREAD LOSS`.**
- ⚠️ **The "unbroken verbatim ICB prefix at the head of each step's prompt" check cannot be run** — the
  prompts are not on disk, only the artifacts. Same limitation the predecessor QA reported. I will not
  claim a pass I did not measure.
- ⚠️ **Blind golden + decoy comparative judging: NOT RUN, and I will not pretend otherwise.** A judge who
  assembles his own blind set is not blind; the coordinator must assemble it. This is a gap in *this
  audit's* calibration, declared.

---

## STRUCTURED EVIDENCE BLOCK
*(measured; beside the verdict, never replacing it)*

| Metric | Measured | Band / Cap | Pass/Flag |
|---|---|---|---|
| MUSIC PROMPT chars (24) | 866–995 | ≤1000, aim 850–1000 | ✅ |
| EXCLUDE chars (24) | 545–782 | ≥400 | ✅ |
| Suno LYRICS field chars (24) | 4261–4909 | <5000 (target ≤4800) | ⚠️ **FLAG** — 4909 exceeds the 4800 target |
| Sung lines (24) | 77–101 | 70–120 | ✅ |
| BPM (24) | 100–126, per style field | 100–129 | ✅ |
| Theme + SONG FORM + 5 Disc channels | 24/24 | required | ✅ |
| SFX cue, discrete, own line | 24/24 | ≥1 | ✅ *(predecessor: 0/24 as a discrete cue)* |
| Prose sentences inside lyrics field | **0** | 0 | ✅ *(predecessor: 5 of 24)* |
| EMO header convention | **one**, all 24 | one | ✅ *(predecessor: three)* |
| EMO tags = specific embodied emotion | 24/24 | required | ✅ |
| **Cross-pair** worst lyric / prompt similarity | 0.036 / 0.096 | 0.42 / 0.58 | ✅ |
| **Within-pair** worst lyric-FIELD similarity | **P04 0.426** | *(unbounded — no ceiling exists)* | ⛔ **FLAG — above the 0.42 lyric ceiling if it were applied** |
| **Within-pair SEAM template index** | **1.27×–2.40×, mean 1.79×, all six pairs** | *(no instrument exists)* | ⛔ **RED — finding F1** |
| Cross-pair content 4-grams | 2 substantive (`wet to the elbow`, `somebody gave me a`) | 0 | ⛔ **FLAG — finding F3** |
| Cross-pair EMO-tag collisions | 2 strings, 4 pairs, 11 songs | 0 | ⛔ **FLAG — finding F3** |
| House lexicon (`toast`/`kettle`/`chips`/`Best ___ I've had`) | **0** | 0 | ✅ **fully repaired** |
| Boundary-hugging (prompt ≥985) | 1 — P06-V3 at 995 | flag only | ⚠️ FLAG *(predecessor: P02 4/4)* |
| Dedications | 437 · 483 · 469 · 491 · 463 · 473 | ≤500 | ✅ |
| Negations in STYLE fields | 0 | 0 | ✅ |
| Artist names in generator fields | 0 | 0 | ✅ |
| Wolves that cost her | **15/16 clear, 1 displaced** | — | ✅ *(predecessor 9/16)* |
| Wolf ceiling | 4 of 6 pairs | ≤ two-thirds | ✅ |
| Repeated-line ratio | high by design (byte-identical choruses, assigned spine lines) | FLAG only, chorus-exempt | ✅ — deliberate return, not exhaustion |

---

## SOMATIC GATE — THREE HYPER-SKEPTICS
**BLOC RESULT: 2 YES / 1 scoped NO → NOT BLOCKED**

**Skeptic 1 — the Anti-Sentimentalist (Sontag seat): YES.**
*"You took the kettle out and the songs did not get colder — that is the proof R3.1 was a real repair and
not a subtraction. Six rooms, and I can tell all six apart with the titles covered: a beach at hour five, a
strip-lit hall on a Wednesday, a sodium platform, a theatre at midnight, an airport lounge at three, a wet
market stall. Nobody consumes anything anywhere. **My complaint is the seam.** You wrote the one sentence
that satisfies every clause the brief attached and then you wrote it four times."*
**Counter-move:** *point a swap test at the seam. `Swap the seam block between two of your own four songs.
If nothing breaks, you wrote one seam four times.` Same sentence R3.1 already wrote — aim it one section
earlier.*

**Skeptic 2 — the Pleasure Politician (Willis seat): YES.**
*"Sixteen wolves and every one of them costs her, and not one of them makes the pleasure into a lesson.
`I took them off the phone myself` is the best thing this personality has done. `I know which chair in
here does not wobble` is the best radiant in twenty-four songs. And P02 and P03 hold eight songs with
nothing at stake and never once apologise for it — `And I am not learning it` is a woman refusing to be
improved, which is the pillar working. **My one no is `I have never been so seen / And I would like it
back`** — the same constructed epigram the last judge failed in the same pair, in the same variation
slot, surviving a routed repair."*
**Counter-move:** *replace it with the physical fact she already has in that verse — the phone nearly out,
asleep across the seats, a stranger's coat over the arm. She has the situation. Delete the sentence.*

**Skeptic 3 — the Fisher seat: NO, scoped.**
*"Across pairs the wolves break cleanly — a tide, a bow missed, a deletion request, a part out of her own
machine, four genuinely different mechanisms of harm. **That is a repair.** But look at what four pairs
did with the endings. P02, P04, P05 and P06 all end on the hook, and your own doctrine says a byte-identical
chorus needs no defence and no note — **so the only cross-song test you own lands on the one string you
have forbidden yourself to question.** P04's four outros are seven lines of one machine with
`Nobody X / Nobody Y` in all four. P05 sings `I signed it standing up` in four songs and calls it four
radiants — that is `Tide's at nine` with a new noun. And `somebody gave me a` opens a radiant in two
different pairs. **A personality that resolves the same way every time has a mannerism, not a repertoire.**"*
**Counter-move:** *(1) exempt nothing from the swap test — run it on the seam and the radiant, not only
the closing line. (2) Stop assigning radiant KINDS as sentence types; assign territories, as P02 and P03
were, and the frames stop colliding. (3) Build R9: measure the 24 radiants against each other directly.*

**Named-corpse checklist:** one-note emotional arc — **no**, every song reaches a second movement.
Single dominant repeated section — **no**. A motif that never transforms — ⚠️ **P05's `I signed it
standing up`** and **P04's `[authority] told me once I could`**, both of which recur unchanged in
function across four songs. Repeated-line collapse — **no**; every repetition in the run is an assigned,
defensible device (byte-identical choruses, spine lines, chain rhyme, call-and-response).

---

## THE RANKED TOP 6 — best three per arm, ranked **within arm only**

### ACCESSIBLE

**1 · P01-V2 — "COME BACK SANDY"**
She turns the photographer's last plate onto the children's badly-built wall and her own best work goes
under unphotographed — **the Golden Seed's exact argument paid for with her own hands, inside a song that
is otherwise pure delight.** Not V1 (cleanest statement, cheapest wolf), not V3 (strong decision, the one
P01 ending that could sit in any of the four), not V4 (superb wolf, but it spends its second half
explaining the trade). And `I came for the bit where somebody asks if they can help` is the least
explicable radiant of P01's four, which is what a radiant is for.

**2 · P02-V2 — "SECOND CORNET"**
It gives its admiration to the person the room is not looking at, doing the least glamorous job — ninety
bars of rest counted perfectly — and it carries **`I know which chair in here does not wobble`, the best
single radiant in the 24-song set**: it proves the relationship without naming it, refuses to explain
itself, and could not appear in any other song. Not V1 (the concept's most obvious instance), not V3 (the
pair's most repetitive spine, its safest admiration), not V4 (a strong device, the pair's least physical
ending).

**3 · P03-V4 — "COME AND SEE THIS"**
The only song in 24 where the return is completed by **two people holding the same object** — *"He has
still got his end of it. / I have still got mine. / Neither of us is putting it down."* `Then you wait.`
closing four consecutive verses is the run's best structural device. Not V1 (the most dazzling procedural
writing, but it resolves onto a man two hundred yards away who never touches the thing), not V2 (the
wittiest, the least at stake), not V3 (best chorus, ends on a stranger's gesture rather than a shared
object).

### AMBITIOUS

**1 · P05-V1 — "HIGH VALUE SEGMENT"**
`I made the argument out loud / To a cleaner with a mop and a kind face / He said love, I only do the
floors / He's right` — **the best four lines in the run**, and the cost R3.1 was written to buy: a thing
that happens to her, in a room, with a body in it. Not V2 — whose wolf (`I took them off the phone myself
/ Now the only copy is theirs`) is **the strongest single wolf in 24**, and which I would rank first but
for the unrepaired epigram in a *pre-chorus*, a position that repeats. **If R6 lands, V2 takes this slot.**
Not V3 (the pair's cleverest idea and its only bodiless room), not V4 (the Container moment is a comment
on the record rather than a thing that happens inside it).

**2 · P06-V3 — "OUT OF MINE"**
The run's cleanest wolf: she opens her own machine, takes the working part out warm, puts it in his, and
the song ends with her lid going down on a dark one. **The only song in 24 where the assigned seam, the
assigned wolf, the assigned radiant and the assigned return are the same physical action.** Not V1 (the
pair's most beautiful chain-rhyme, the run's one displaced-cost wolf), not V2 (superb costed action, but a
phone-call procedural whose second half is a list), not V4 (the most moving idea, the least physical song,
and its radiant is the one that collides with P04-V3).

**3 · P04-V3 — "THE GOOD ONE GOES FIRST"**
The **only** P04 variation whose radiant escapes the pair's `[authority] told me once I could` template,
and the only one whose ending contains an image nobody instructed: *"Jax cut the corner off the good one.
/ It is face up on the dashboard."* Not V1 (see below), not V2 (strongest single action in P04, but its
radiant explains the song it is in, which the radiant rule forbids), not V4 (best atmosphere, a wolf
reported rather than shown).
⚠️ **Ships only after R4:** this song holds one half of the cross-pair radiant collision. Its line is the
stronger of the two — **repair P06-V4's, not this one.**

---

## ⭐ THE SET'S SINGLE WEAKEST SONG

### **P04-V1 — "TURN THE ROOM AROUND"**

Plainly: **it is P04-V2 with less at stake and a worse hook, and if it vanished the set would lose
nothing.** Four compounding reasons.

1. **Its sibling makes it redundant.** V2 is the same room, the same crew-escalation architecture, the
   same seam, the same chorus shape — and adds a decisive physical act (she takes the first swing). V1
   has no act; the room simply comes apart around her.
2. **The vaguest hook in 24 songs.** `We turn the room around` describes the job in the language of a job
   description. Every other P04 chorus names a physical operation — *take it at the joint*, *take the good
   one first*, *leave the engines running*. This one names a shift.
3. **The most template-locked radiant in the run.** `An old stage manager told me once I could be the one
   to say when` is the purest instance of the frame P04 repeats three times, it is the longest line in its
   stanza, and it breaks the metre in the identical place as its three siblings.
4. **Its seam is the flattest of P04's four flat seams** — `Two, three, four. / There's a second kit down
   in the pit. / Bex has got the baritone. / Everybody clapping on the floor. / Two, three, four.` Nothing
   happens; the arrangement arrives and is described.

Its wolf is genuinely good (`I called it over them. I missed / the last time anybody took a bow.`) and
that line deserves to survive — **into V2 or V4, where something is at stake around it.**

**Runner-up weakest: P02-V3 "HE CAN, HE CAN"** — the pair's thinnest situation (a man holds a long note),
its most repetitive spine (`I could not do that on my best day` ×12), and its least *hers* radiant (`I
have been coming here since it was called something else` is a fact about the venue). Saved from the
bottom by `He does it twice and it is the same twice. / I checked.` and by `And I am not learning it`,
which is a woman refusing to be improved and is the R3 pillar working.

---

## REQUIRED REPAIRS — routed

| # | Severity | Finding | Route |
|---|---|---|---|
| **R1** | ⛔ **BLOCKING before render** | **P05: `Hohner Clavinet D6` is factually wrong** — Wonder played a **model C** on "Superstition". Wrong in the dedication **and in all four STYLE fields** (L22, L170, L324, L474). | **step 08** (style fields) + **step 10** (dedication). One string, five places. |
| **R2** | ⛔ SUBSTANTIVE | **P04's seam does nothing in 4 of 4.** The only action across it is counting the band in. Give the singer one physical act across the seam in each of the four — she is holding a crowbar, a ladder, a roller and a set of keys. | **step 07** (P04 song guides) → 08 → 09 |
| **R3** | ⛔ SUBSTANTIVE | **P05-V2's epigram** `I have never been so seen / And I would like it back` — the predecessor's routed R7 recurring in the identical slot. Replace with the physical fact already in that verse. | **step 09** (P05) |
| **R4** | ⛔ SUBSTANTIVE | **Cross-pair radiant collision**: P04-V3 `Somebody gave me a key to the paint frame…` / P06-V4 `somebody gave me a working one…`. Repair **P06-V4's** (P04's is the stronger line, and P06 already has 3-of-4 radiant collapse). | **step 09** (P06) |
| **R5** | ⚠️ SUBSTANTIVE | **`WET TO THE ELBOW!`** — P01-V1's title and hook, shouted verbatim inside P06-V1. Change P06's response line; P01 owns the phrase. | **step 09** (P06) |
| **R6** | ⚠️ | **P01's dedication**: hedge `is Hal Blaine's 1963 figure` to the construction P03 uses (`whose kits carry…`). Right person, unsourced figure. | **step 10** (P01) |

### PROCESS REPAIRS — for the coordinator and for `sunna.yaml`

| # | Item |
|---|---|
| **P1** | ⭐ **BUILD THE SEAM-SWAP TEST.** *Swap the seam block between two of your own four songs. If nothing breaks, you wrote one seam four times.* R3.1's own sentence, aimed one section earlier. Measured basis: seam template index 1.27×–2.40×, **all six pairs, no exceptions.** |
| **P2** | ⭐ **STOP ASSIGNING RADIANT KINDS AS SENTENCE TYPES.** *a permission she was given once* and *a document she signed standing up* are grammars, and each produced four sentences with one frame. *a place she keeps coming back to* and *an instruction she ignored* are territories, and each produced four unrelated sentences. **Assign territories.** |
| **P3** | ⭐ **BUILD R9 — the predecessor's un-built repair.** Measure the 24 radiants against each other directly. `GATE_REPORT.json` is a file-existence manifest and measures no content at all. |
| **P4** | ⭐ **ADD A WITHIN-PAIR SIMILARITY CEILING.** The 0.42 ceiling is cross-pair only. P04 measured itself at **0.426** on the field and nothing gated it. The four songs a listener hears back to back are the four least measured against each other. |
| **P5** | **MEASURE THE SUNG BODY, NOT THE LYRICS FIELD.** P04's finding, and it is right: `[Theme:]`, `[SONG FORM:]`, five `Disc_` channels and the EMO header shape are mandated shared vocabulary, so every pair that adds them raises every similarity number at once for reasons unrelated to whether two songs are the same song. P04 field 0.393–0.426 → sung body 0.199–0.265. |
| **P6** | **RESOLVE THE SOURCE-2 / DISTINCTIVENESS CONTRADICTION.** The Source-2 floor *mandates* ≥2 verbatim phrases from a five-phrase list into every style field; the distinctiveness ceiling forbids cross-pair sameness in the same field. **One rule manufactures the sameness the other forbids, and the metric cannot see it** (five phrases in 900 chars do not move cosine similarity — measured cross-pair prompt worst = 0.096). Decide which rule owns the style field. |
| **P7** | **THE CHORUS IMMUNITY AND THE RETURN-SWAP TEST COLLIDE.** Four of six pairs end on their hook; `sunna.yaml` rule 4 exempts byte-identical choruses from defence. Either the swap test must apply to the last *non-hook* line, or the immunity must be scoped to *within-song* repetition explicitly. |
| **P8** | **P01 ran both swap tests at step 09 and did not carry the result into step 10 or 11.** The brief says *"Report the result."* Five of six pairs did. Minor; substantively P01 passes and the step-09 record is excellent. |
| **P9** | **Coordinator to assemble a blind golden + decoy set** for a sample of finalists. This audit's calibration check was not run and a judge cannot blind himself. |

---

## ⭐ MEASURABLY BETTER, OR JUST BETTER-FILED? — **BOTH, AND THEY SEPARATE CLEANLY**

**Better-filed (real, and it is the difference between publishable and not):** 36/36 artifacts against the
predecessor's 6 files · the NO-SKIP rule no longer fires · one EMO convention against three · `SFX` cue
24/24 against 0/24 · zero prose sentences inside the lyrics field against 5 of 24 · per-pair isolated
briefs authored rather than centrally templated · the dedication extractor fixed · `RUN_STATE.md` derived
from disk.

**Genuinely better writing — and this is the half that matters:**
- ⭐ **Wolf-cost 9/16 → 16/16 events, 15/16 with a cost she pays.** That is not filing. That is sixteen
  women doing sixteen costly things instead of seven of them reciting conditions. It is the single largest
  quality delta between the two runs and R3.1 bought it outright.
- **Seam on material 3/6 → 6/6.** No pair now asks Suno to hold a mix distinction for eight bars.
- **The setting monoculture is gone at the root** — six rooms, zero domestic-warm, zero house-lexicon hits.
- **The architectures are executed with more rigour** — P03's four *different* verse devices, P06's chain
  rhyme sustained unbroken across four songs, P05's told-rhymes/true-doesn't asymmetry held for 24 verses.

**And one respect in which it is worse, which is the most useful thing in this report:**

> **The predecessor's ceiling was higher. This run's floor is much higher and its top is flatter.**
> The predecessor QA called that set *"the strongest lyric set I have audited under this personality"*
> and named a single peak. This run has no song as bad as the predecessor's worst — and, on my read, no
> single song quite as good as its best. **The variance came down.**

**The mechanism is measurable and it is the same mechanism as the seam finding.** Each pair brief here
assigns architecture, rhyme logic, hook grammar, sonic device, setting, radiant kind, return shape and
four variation angles. That is eight axes fixed before a word is written. **What is left to differ in is
the slots — so the pairs differed in the slots, and not in the sentences.** The specification that raised
the floor is the same specification that produced four seams from one block, four radiants from one frame,
and four endings from one machine.

That is not an argument for less specification. It is an argument for **pointing the falsification tests at
the things the specification most tightly constrains** — because those are now the positions where a
formula is cheapest to write and hardest to notice. R3.1 already knows this sentence. It just has not aimed
it at the seam yet.

---

## FAILURE-LEDGER ENTRY

**NOT WRITTEN.** Verdict is REPAIR; nothing is selected for ship. The entry drafted for the moment R1–R4
land, for `vault/COMPETITION_LEARNINGS.md`:

> **theme-tags:** daily-music · SUNNA R3.1 · 24-song set · text-stage, unrendered
> **what the gate caught:** the setting monoculture and the status-report wolf were both repaired at the
> positions R3.1 named, and the template re-formed in the SEAM — the most heavily specified and least
> falsified section in the run, at 1.27×–2.40× its own song's self-similarity, in all six pairs.
> **transferable rule:** *point the falsification test at the section your specification constrains most
> tightly. What is not assigned gets defaulted; what is specified but never varied gets templated — and
> the more clauses a rule attaches, the tighter the template, because the sentence that satisfies all of
> them is nearly unique.*
> **confidence:** MEDIUM — n=2 consecutive runs, one personality, text stage only, nothing rendered.

---

## FINAL RECOMMENDATION

**REPAIR.** Land R1 (five string replacements, one of them factually false and sitting in four generator
fields), then R2–R5. R6 and the nine process items are cheap and should go into `sunna.yaml` and the
coordinator contract before the next R3.1 run, because **the next run is where the seam template becomes
the sound of the personality** — by exactly the mechanism the genre-bed block already named, one level up.

Then re-QA and select. The top six above hold as a slate on the assumption R1–R5 land; **P05-V2 replaces
P05-V1 at AMBITIOUS-1 if R3 lands.**

⚠️ **Nothing rendered, nothing published, nothing spent.** `vault/AUTONOMY.md` holds: drafts on disk.
The `.RUN_LOCK` was not touched and no pair file was edited.

💜
