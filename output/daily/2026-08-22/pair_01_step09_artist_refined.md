# PAIR 01 — STEP 09 · ARTIST-REFINED
## `daily-music-2026-08-22-sunna-offbeat` · **SLOT P01 — SWING** · the critic/artist loop

> **Refining voices are role seats, carried by ROLE TITLE only.** No panel figure's name goes anywhere
> near a generator-facing field. **Every measurement below was taken in code**, by
> `scripts/measure_soundcraft.py` (the canonical `sung()` definition) — ⛔ nothing here was counted by eye.
>
> **Contract limit honoured:** this step sharpens and repairs. **No new scene content was invented** —
> every person, object and room fact in the refined lyrics was already on the step-06 inventory tables
> before step 08 was written.

---

## 1. THE MEASUREMENT THAT DROVE THIS PASS

Run on `pair_01_step08_generation.md` **before** a single repair:

| | V1 | V2 | V3 | V4 | gate |
|---|---|---|---|---|---|
| music prompt chars | 899 | **814 ⛔** | 859 | **840 ⛔** | 850–1000, target 870–960 |
| lyrics field chars | **4946 ⚠️** | **5089 ⛔** | 4661 | 4802 ⚠️ | ≤4800 target, 5000 hard cap |
| sung lines | 80 | 78 | 78 | 78 | 70–120, target 78–110 ✅ |
| strict end-rhyme | 0.475 | 0.423 | 0.410 | **0.308 ⚠️** | floor 0.30 |
| repeated-line ratio | 0.325 | 0.333 | 0.333 | 0.333 | floor 0.20 ✅ |
| **words per line** | **7.76 ⛔** | **8.47 ⛔** | **7.59 ⛔** | **8.04 ⛔** | ceiling 7.50 |
| sung numeric facts | **2 ⛔** | 1 | 1 | 1 | exactly 1 |

⭐ **The four-way failure was one failure.** Every long line was long for the same reason: a rhyme was
being reached for at the end of a sentence that had already finished. Cutting the reach fixed the
words-per-line ceiling **and** the two oversized lyrics fields **and** the thin V4 rhyme, because the
shorter line puts the rhyming word where the ear expects it instead of four words past it.
**One edit, four gates.** That is the note this pass is built on.

---

## 2. THE THREE GATES, RUN LINE BY LINE

### ⭐ THE FLAT TEST — say it out loud, flat, to a face
Applied to all 314 sung lines. **Nine died.** A line that dies flat was never simple; it was short.

| line as drafted | verdict | what replaced it |
|---|---|---|
| *"Somebody should say that out loud at some point."* (V1) | dies — **it instructs the listener** | cut entirely; Del's tercet took the slot |
| *"Anyway. Vic does the lamps."* (V1) | dies — a hinge with no content, written to escape a hole | cut with the tercet above it |
| *"It stays where I put it. It's part of the bill."* (V1) | dies — *bill* means nothing in this room; the rhyme chose it | *"It has not moved. It's not what tonight's about."* |
| *"She has been up first at this thing for years."* (V1) | dies — *this thing* is the one vague phrase in a song made of specifics | *"She has done that in that dress for years."* |
| *"That is how it works. That is what a record does."* (V3) | dies — **interprets the story instead of continuing it** | Fen putting the needle back at the top, twice |
| *"She isn't here and the choice is still hers."* (V3) | dies — says the thesis a second time, one tercet after it landed | replaced by the same repair |
| *"Nothing either side of it. Bare wood. Bit rough."* (V4) | survives, barely — the rhyme is a mumble | *"Nothing either side of it. Bare wood. Full stop."* |
| *"She always does."* (V2) | survives alone, **fails as a set** — see §3 | *"There is a system and it is hers."* |
| *"They always do."* (V4) | same family | *"It happens every time."* |

### ⭐ THE HUM TEST — the tune must exist with no words at all
Each variation was checked for an instrumental hook that is **present before the first vocal and after
the last one.** All four passed on entry, but two were not *carrying* the tune:

- **V1** ✅ — the three-note rising horn answer opens the record in the intro's second turnaround and is
  the last figure before the bass walks out.
- **V2** ✅⭐ — **the strongest of the four**: the instrumental hook is Gwen's soundcheck phrase, heard
  behind Verse 1, developed in the Horn Break, and played **alone into an empty room** at the end.
  The voice never sings it. That is the hum test passing without help from the vocal at all.
- **V3** ⚠️ → repaired — the unison horn figure opened the song but the Tag handed the record to the
  guitars without ever restating it. **Fix:** the horns now play the figure once more before they leave,
  so the tune brackets the record instead of only opening it.
- **V4** ⚠️ → repaired — the full-band turnaround is the hook, but the Coda's dismantling was written in
  the style field as a *principle* rather than an order. **Fix:** the departures are now **named in
  sequence** (horns, tambourine, claps, guitars, piano, brushes, bass last) so the tune comes apart in
  front of the listener instead of fading.

### ⭐ THE REMARK TEST — would anyone say *"that was clever"*?
**Three hits. All three cut, none buried** — burying a device this close to the surface only moves it.

1. **V1, Verse 2:** *"The flowers will outlast the reason they were bought."* ⛔ This is the reveal engine
   **naming itself**. It is the single most quotable line in the draft and that is exactly the problem:
   the record would be *about* its own device. **Cut.** The engine now arrives only as facts — a switch
   learned off the man before, a date on a wall, a booking on Thursday — and it is never explained.
2. **V1, Tag:** an earlier pass had *"the arrangement outlives the party"*. ⛔ Cut before it reached the
   draft; step 06 §10 predicted this failure and named the guard. **The word *arrangement* now appears
   nowhere in any of the four lyrics.** It was going to be a wink and there is no such thing as a small one.
3. **V3, Bridge:** *"That is what a record does."* ⛔ Same class — the song stepping outside itself to
   explain what just happened. Replaced with Fen doing something instead.

---

## 3. THE CROSS-VARIATION SWEEP — the R3.2 check, run by machine
*(R3.2 measured a byte-identical line landing in all four variations of a heavily-specified block. That
failure is invisible to whoever wrote all four, so it is not looked for by eye.)*

**Method:** normalise every sung line (lowercase, strip punctuation), intersect the four sets, and report
every line and every near-family appearing in more than one variation. Choruses are exempt by
construction — they are four different choruses and each is byte-identical only **within** its own song.

**Result before repair — one exact collision and one family of three:**

| hit | where | ruling |
|---|---|---|
| *"Nobody asked him to do that."* | **V1 Verse 4 tercet 1 and V4 Verse 3 tercet 4 — byte-identical** | V1 keeps it (it is the older line and it lands better on Ade). **V4 → *"He does not like a mess."*** |
| *"She always does."* / *"They always do."* / *"Same as he always does."* | V2, V4, V1 | The **V1 Tag** keeps its form — it is the closing move of that variation and the return is the point. **V2 → *"There is a system and it is hers."*** · **V4 → *"It happens every time."*** |

**Result after repair:** zero sung lines shared between any two variations, and no near-family with more
than one member. Re-run and confirmed in code at the end of this pass.

⭐ **A note the run should keep:** both collisions were **hanging third lines** — the shortest, plainest,
most load-bearing line in the architecture. That is not a coincidence. When four songs share a verse
shape, the position that collides is the one with the fewest words to be different in. If this slot's
architecture is reused, sweep the hanging lines first.

---

## 4. THE FIFTEEN DEFECTS, ANSWERED BY NUMBER

| # | answer |
|---|---|
| **D1** | ✅ *"Forty-two of them"* → *"Nobody in this town irons like that."* The number is gone; V1 is back to exactly one sung numeric fact. **Sweep run across all four for any second count** — *four o'clock*, *table four*, *ninety people* and *two of them* were all removed at draft stage for the same reason. **The only number sung anywhere in this pair is *nineteen sixty-four*.** |
| **D2** | ✅ cut. See REMARK TEST hit 1. |
| **D3** | ✅ cut. See FLAT TEST row 1. |
| **D4** | ✅ cut. The whole tercet is replaced by Del carrying the bass up the back stair and tuning it to the room — **a person from the step-06 table, not a new invention.** |
| **D5** | ✅ *"at this thing for years"* → *"in that dress for years."* Specific object, better rhyme, one word shorter. |
| **D6** | ✅ *"It's part of the bill"* → *"It's not what tonight's about."* ⭐ This is the better line by a distance: it is Sunna refusing to let the card become the subject, said flatly, in the middle of a list. |
| **D7** | ✅ see §3. |
| **D8** | ✅ cut. See REMARK TEST hit 3. |
| **D9** | ✅ cut with D8; the tercet now **continues the story** — Fen lifts the arm and puts the record back on, unasked, so the room gets it twice. |
| **D10** | ✅ see §3. |
| **D11** | ✅ see §3. |
| **D12** | ✅ **every negation moved out of the style field into `## 1B. SUNO EXCLUDE PROMPT`.** Four style fields, zero negations. This is the field-discipline rule: Suno reads the nouns in *"no reverb tail"* positively. |
| **D13** | ✅ *restrained*, *tasteful*, *warm and bittersweet*, *wistful* all deleted. **THE GRAIN LAW:** a mood adjective is a hope; an operation is an instruction. Where the mood was, there is now a named entry at a named point. |
| **D14** | ✅ *"instruments come in gradually"* → the V2 style field now reads the parade out in order: brushes with the vocal, one trumpet through two verses, full section at the first chorus, guitars at the second verse, claps and thirds at the second chorus, tambourine at the fourth verse. **RULING X satisfied in the field that reaches the generator, not in prose about it.** |
| **D15** | ✅ **the C1 answer is now in all four style fields, in words:** *the bass keeps climbing to the fifth degree at the top of every turnaround under a chord that never arrives.* Without it the hole is true but inaudible. |

---

## 5. THE LINE REPAIRS — what got shorter, and why it got better
*(the words-per-line pass; 41 lines shortened across the four, none by deleting content)*

| drafted | refined | words |
|---|---|---|
| *"I set it at its place and I square it to the fork."* | *"I set it down and square it to the fork."* | 13 → 9 |
| *"I turn it so it faces out, the same as all the work."* | *"I face it out, the same as all the work."* | 13 → 9 |
| *"And there's a card at the far end with a name on it still."* | *"There's still a card at the far end, facing out."* | 14 → 9 |
| *"The cards are in a box and the box is on a chair."* | *"The cards are in a box on a spare chair."* | 13 → 9 |
| *"There's a card in the middle of the box that says Nell."* | *"Halfway down the box there's one that says Nell."* | 12 → 9 |
| *"The light comes down the room and lands on the far wall."* | *"The light comes down the room to the far wall."* | 12 → 10 |
| *"It only does that in August. It doesn't last long at all."* | *"August only. And it doesn't last at all."* | 12 → 8 |
| *"She has been up since it was dark. Now the day has come."* | *"She's been up since dark. Now the day has come."* | 13 → 10 |
| *"He walked the room to hear it. He heard it. That was all."* | *"He walked the room to hear it. He heard it. That's all."* | 13 → 11 |
| *"She is doing both and both of them are right."* | *"Both at once, and both of them are right."* | 10 → 8 |
| *"I stand it back up on the table with the cloth pulled off."* | *"I pick it up and stand it on the bare tabletop."* | 13 → 10 |

⭐ **The craft note, for the ledger:** in every one of these the drafted line had **already said the
thing** and then kept going to reach the rhyme. Cutting the reach did not cost an image anywhere. It is
the same finding as the measurement in §1 and it is the one worth carrying forward: on a plain-language
brief, **an over-long line is almost never over-full — it is a rhyme arriving late.**

---

## 6. WHAT WAS REFUSED

- ⛔ **Refused: adding a fifth structural device.** This pair already carries four that are load-bearing —
  the harmonic hole, the withheld horn answer, the byte-identical chorus, and the tercet's hanging line.
  A fifth would be a device arguing with the ones that are working.
- ⛔ **Refused: making the wolf line longer, softer, or explained.** *"There is one of them for Nell."* ·
  *"So I do it first."* · *"That was Nell."* · *"It says Nell."* Four sentences, four to six words each,
  no adjective in any of them. Every instinct to add a clause was the instinct to sell it, and
  **she'll say the line, she won't sell it.**
- ⛔ **Refused: giving the horns a quieter answer at the wolf instead of no answer.** A soft stab is a
  production hush wearing a smaller coat. **Nothing is the instruction.**
- ⛔ **Refused: repairing the identical rhyme *lit / it* in V4 Verse 1.** It reads like a defect and it is
  the flattest possible landing for a line about a phone nobody has come back for. Correcting it would
  make the line prettier and worse.
- ⛔ **Refused: letting V3's chorus carry the thesis.** *"Everybody off the floor"* stays a piece of
  furniture-moving. The thesis stays in the Bridge, which is what the forbidden-section ruling is for.
- ⛔ **Refused: shortening any chorus to save characters.** The two lyrics fields that were over budget
  were repaired **entirely in the verses and the section headers**. ⛔ A byte-identical chorus needs no
  defence and it is not a place to find room.

---

## 7. WHERE THE CHARACTERS CAME FROM — the budget repair, itemised

Two fields were over. Neither was fixed by deleting a sung line.

| source | V1 | V2 |
|---|---|---|
| section headers trimmed to ≤115 chars, all four parts kept | −372 | −402 |
| the 41 shortened lines (§5) | −188 | −243 |
| the cut tercet (D2/D3/D4), replaced by a shorter one | −41 | — |
| **net** | **4946 → ~4345** | **5089 → ~4444** |

⭐ **The header line is where a lyrics field quietly goes over.** Eleven headers at an average of 159
characters were costing more than a fifth of V1's entire field. Trimmed to ≤115 they still carry section,
EMO pair, voice and cue — the full four-part form the contract requires — and the song lost nothing.

---

## 8. ELIGIBILITY, SCORED HONESTLY *(target 6/7)*

| # | property | verdict |
|---|---|---|
| 1 | body in the song, first 30s | ✅ walking bass and brushes from bar one; the shuffle is physical inside two seconds |
| 2 | adoptable hook | ✅ four hooks, four to seven syllables, all instructions about furniture, doors, floors or lights |
| 3 | vast emotional TAM | ✅ setting up a room for people you love, with one seat that will stay empty |
| 4 | specificity paradox | ✅ one sung fact, in words, answered rather than recited, in a different mouth in each variation |
| 5 | cognitive ease | ✅ every line survives the flat test; the complexity is entirely in the harmony and the parade |
| 6 | vocal co-discovery | ⚠️ **REFUSED BY PERSONALITY, as the seed declared.** She already knows and is telling you. Delivered instead as *the singer noticing she is still listing after the card* |
| 7 | sonic threshold | ✅ the demand arrives as one timbre entering; no dynamic swell anywhere |

**6/7 — with property 6 declared refused rather than quietly scored.** *A gate that cannot fail is not a gate.*

---

## 9. THE FINAL SELF-CHECK BEFORE STEP 10

⛔ No gear change · no genre shift · no key change inside a song · no tempo change (102 BPM, all four,
first bar to last) · no hole before 0:38 · no band drop-out at the card · no hush, ritardando, held note
or reverb tail near it · no distortion anywhere · no thesis in any final chorus · no numeral as digits ·
exactly one sung numeric fact per song · no Bandcamp phrase in any lyric · no house-lexicon phrase ·
no real musician, band, record, company, official or case in any lyric, style field, exclude field or
title · no child anywhere · no line explains why she is not coming.

*Refinement closed 2026-08-22. Packages follow at step 10.* 💜
