# PAIR 05 — STEP 09 · ARTIST REFINEMENT · **SUNNA R5**

**Run:** `2026-08-27-high-energy-registers` · **Pair:** 05 · Four variations, twelve fields.

⭐ **This file records what the artist pass actually CHANGED, and why, with the measurement that caught it.** Five of the seven changes were caught by a script, not by reading — which is the point of running one.

---

## 1. THE CHANGES THAT WERE NOT NOTATION — the song moved

### 1.1 ⛔ A COMPANY NAME IN A SUNG LINE. V4, chorus, byte-identical five times.

The chorus shipped from the first draft as:

> *"Bins and a Vauxhall and a puddle"*

**A car manufacturer is a company name, and `06_music_handoff.md` is unambiguous: no company name in any of the three pasted fields.** ⚠️ **And `vault/field_token_lexicon.yaml` would NOT have caught it** — the `companies:` list holds instrument manufacturers, not car makers — so a clean census run would have reported zero and been wrong. *That is exactly the FINDING 12 shape: a number that is true about the instrument and false about the world.*

**Repaired to:** *"Bins and a hatchback and a puddle."* Identical scansion, funnier, and it belongs to the listener instead of to a brand. ⭐ **This is the strongest argument I have for the naming rule**: the generic noun was better writing, not just safer.

**Flagged upstream:** the lexicon has an unguarded class. Vehicle, retail and venue brands are all reachable from a lyric about a car park, and none of them is in it.

### 1.2 ⛔ A HOUSE-LEXICON SENTENCE SHAPE. V4, verse three.

First draft closed the fox verse with *"Best review I've had."* ⛔ **`sunna.yaml` R3.1 names this exact shape as measured formula** — one pair signed off all four variations with `Best Thursday of my life / Best post I've ever had / Best Tuesday I've had / Best week I've had in ages`, nine instances, invisible to a 0.328 similarity ceiling. **It is my own rule and I walked into it on the first pass.**

**Repaired to:** *"I'm taking that as a yes. / It isn't one. I'm taking it."* ⭐ Which is a much better ending for the no-thesis song, because it names the sentimental reading and then declines it in the same breath.

### 1.3 ⚠️ AN EMOTION WORD THAT SURVIVES A NAIVE SCAN. V4, verse two.

*"Ha — sorry — the last one shut it"* — **`sorry` is in the remorse family**, and R2 does not care that it is functioning as a discourse marker. **Repaired to** *"Ha — no, the last one shut it."* ⭐ Better anyway: *no* is a refusal, and this is a woman who has just told the room that helpful people are the problem.

### 1.4 ⛔ THE ONE-FACT RULE, VIOLATED IN THREE PLACES ON THE FIRST DRAFT.

| where | first draft | repaired |
|---|---|---|
| V1 verse 1 | *"I moved the amp a foot to the left"* | *"I moved the amp off the wall"* |
| V1 verse 2 | *"have moved four inches since nine"* | *"have crossed it since we started"* |
| V2 verse 1 | *"Hall was booked from six … nobody's coming till eight"* | *"booked from the afternoon … nobody's coming for hours"* |

**Measured after repair: zero digits in any sung line across all four songs, and exactly one numeric FACT per song** — *eleven* · *half a second* · *three / the third one* · *seven* — each spelled in words and each **answered rather than recited**:

- V1: *"Box went red at eleven … I was stopping at eleven. / Doesn't mean I got to."*
- V2: *"Half a second, every time"* → *"Behind me, half a second late, / the room is still having a go."*
- V3: *"Wall's had your top off three nights running"* → *"Third one through the wall, / I'm not opening this door."*
- V4: *"Push it open, count to seven"* → *"Seven seconds and a boot, / that's the whole of my equipment."*

### 1.5 ⭐ AN ARRANGEMENT ANSWER MOVED INTO THE LYRIC.

First draft carried *"the loudest moment is where the record narrows and the singer gets closer"* in two style fields. ⛔ **The panel struck that ruling before a song existed** — `SCR-02` describes what the generator does anyway, so instructing it buys zero discrimination, and *a distinctiveness objection answered in the production spec is not answered.*

**So the narrowing is not requested anywhere.** It is treated as a **condition**, and the only decision left is what the words do inside it. All four peaks now go **singular and close** in the lyric:

- V1 — *"Hand. Glass. That's all it is."*
- V2 — *"Dark. Full. Same."*
- V3 — *"Door. Handle. No. / Hand off it. Play."*
- V4 — *"Same speed. Same size. More sky."*

### 1.6 ⭐ FIVE PROHIBITIONS DELETED FROM THE STYLE FIELDS.

A prohibition in the positive field is a non-instruction, and the exclude field already owns every ban. `holds that size to the last bar` / `never grows again` / `no top end at all` and two others were rewritten as positive statements (`and stays there`, `with no top end` as a **description of a filtered signal**, not an instruction to omit). **Measured: the four style fields now carry zero imperative prohibitions.**

### 1.7 ⭐ THE V1 ENDING WAS A THESIS AND IS NOT ANY MORE.

First draft ended *"The glass has stopped. / Everybody's still holding their hand there."* — a crowd, on the last line, in the one song whose whole argument is that volume is evidenced by objects and never by a crowd. **Repaired to** *"Somebody's hand is still on it."* One person. No solidarity claimed that the song has not earned a member of.

---

## 2. WHAT WAS ONLY NOTATION — the songs did not move

Every field except V4's chorus and verses two and three came in **over its cap on the first pass**, and every character bought back came out of **how the song is written down**, never out of the song:

| field | first pass | after | what was cut |
|---|---|---|---|
| V1 lyrics | 5138 | **4453** | EMO headers compressed from ~117 to ~74 chars; Disc block from 694 to ~570; Theme and SONG FORM tightened |
| V2 lyrics | 5263 | **4756** | same three, same order |
| V3 lyrics | 5151 | **4546** | same three, same order |
| V4 lyrics | 5385 | **4794** | same three, plus one Disc token and one section name |
| V1 / V2 / V3 / V4 music prompt | 1013 / 992 / 1037 / 1037 | **944 / 960 / 960 / 957** | clause-level trims only; **no signature, no kinetic defect and no opening moment was removed from any of the four** |

⛔ **Not one sung line was cut, merged or shortened to buy a character.** Sung-line counts are identical to the step-06 declarations: **88 / 90 / 86 / 88.**

---

## 3. WHAT I DECLINED TO CHANGE, AND WHY

**3.1 The tempo.** 148–156 is three cells above my own measured home band (`sunna.yaml`: 100–129 is 1.28×; 150+ is 0.69×, n small). ⭐ **I took the band and took back the pulse** — V1 and V3 are felt in half-time at 76 and 75 by the inherited body while the chassis runs at the assigned tempo. **The BPM is the run's; the body is mine.** Declared, not defied.

**3.2 The word `want`.** It appears once, in V2's chorus, and the wanter is **the room**. R2 bans emotion words; my own R4.1 requires appetite named out loud. ⭐ **An appetite pointed at an object is an act, not a feeling** — and putting the wanting in the architecture rather than in the singer settles the tension instead of arguing it. Declared.

**3.3 The laugh in V4.** It breaks my own contract. It stays. Declared as the R6 decision and as a deviation.

**3.4 The absence of a crowd anywhere.** Four records, zero crowd beds, zero gang unisons, zero room mics. ⭐ **The hooks are still built to be sung back by a real room** — that is a writing target, not a production layer, and I think the forbidden-default list conflates them. Declared.

**3.5 The four songs do not settle the open question.** They all *depict* moving objects, so they cannot distinguish *deliver* from *depict*. I know. It is in the package, in my own words, unresolved.
