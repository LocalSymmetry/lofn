# PAIR 04 — STEP 09 · ARTIST-REFINED · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Register:** the flat refusal that costs you something, said anyway

---

## ⭐ THE REPAIR LOG — what the instrument returned against the step-08 strings, ALL of it

**Command run:** `python3 output/daily/2026-08-27-high-energy-registers/p04_measure.py output/daily/2026-08-27-high-energy-registers/pair_04_step08_generation.md`
*(scratch instrument; reuses `pasted_fields()`, `load_lexicon()` and `hits()` **imported from `scripts/check_field_census.py`** so the field-extraction unit and the token regex are the coordinator's, not mine — the one thing I must not re-implement is the thing being counted)*

| # | what it found | class | repair |
|---|---|---|---|
| **D1** | **ALL FOUR MUSIC PROMPTS OVER THE HARD CAP.** V1 **1148**, V2 **1051**, V3 **1071**, V4 **1032** against a band of 850–1000 and a target of 870–960. | ⛔ **HARD FAIL ×4** | rewritten to 955 / 951 / 946 / 949, each ending on a complete sentence, each still carrying opening moment · spatial staging · kinetic defect · acoustic declaration · signature device in the mandatory order |
| **D2** | **R2 breach, V2:** *"Out loud, in the rain, at a door."* — **`loud` is on R2's own named list.** Idiomatic, meaning *audibly*, and that is not a defence: the ruling names the token. | ⛔ **HARD FAIL** | → *"Out where the rain is, at a door."* |
| **D3** | **R2 breach, V3, twice** — *"I want it out loud and I want it now."* and, worse, **`free` inside the hook**: *"It is a click away, and it is free."* | ⛔ **HARD FAIL, and it was in the discriminator** | → *"I want it said, and I want it now."* and → ⭐ *"It is a click away, and it costs nothing."* **The repair is better than the original**: the hook now speaks in the currency the whole register is denominated in. |
| **D4** | **P2 FALSIFIED AT FIRST PASS.** Shared sung lines across the four measured **5.21%** against a pre-registered ceiling of **≤5.0%**. Two distinct shared strings: `No.` (all four) and `Everything else keeps going.` (V3 **and** V4). | ⚠️ **pre-registered prediction, falsified** | see the note below — **reported as falsified, and both numbers are given** |
| **D5** | V3 mean words/line **7.20**, V4 **7.36** against a ceiling of 7.5. Inside, but the two closest in the set. | ⚠️ FLAG, no repair | left alone; a longer line is the correct shape for a song about a demand that will not stop and a song about years |
| **D6** | Lexicon census across all twelve fenced fields: **0 hits** — companies 0, artists 0, spatial non-instructions 0. | ✅ | none |
| **D7** | Every song carries exactly one numeric **fact**; the other number-words the scanner returned (`one` in *the next one* / *this one* / *the other one* / *every one of them*) are **pronouns, not facts.** | ✅ *(with the distinction named, because a naive count would have called this four breaches)* | none |
| **D8** | Repeated-line ratio 0.405 / 0.417 / 0.282 / 0.372 against `line_return_floor` 0.20; ⛔ every chorus byte-identical, **and no defence is filed for that** (`gates.yaml: chorus_repetition_requires_no_justification`). | ✅ | none |

### ⭐ D4, HANDLED IN THE OPEN — because I am the author of the thing being counted

`MEMORY.md`: *a self-check that lowers your own defect count needs a control.* **Both numbers are reported and neither replaces the other.**

- **First pass, as written, before I touched anything: 5.21%.** ⭐ **The prediction P2 (≤5.0%) is FALSIFIED as filed.** That stands in the record whatever I do next.
- **One repair was then made, and it is a real defect independent of the number:** V3 and V4 both opened their stopped-event section with the identical line *"Everything else keeps going."* — **that is exactly the convergence I pre-registered as the failure I expected** (§R7: *four songs that each clear F1–F4 and are nevertheless four mixes of one*). Two songs had converged at the one bar the whole pair is engineered around. V4's line → *"The rest of it keeps going."*
- ⛔ **`No.` was NOT repaired and will not be.** It is the refusal itself, it is the register's own word, and inventing four different single-syllable refusals to move a number I set myself is **writing around a gate**. It stays in all four.
- **Post-repair number is reported separately and labelled**, in the step-10 verification table, with the command beside it. ⭐ **The direction of my error here is obvious and I am naming it: every repair I made today reduced a defect I owned.** That is why the pre-repair number is printed first and is the one that scores P2.

---

## THE FOUR, REFINED

*(The lyric fields below are step-08's, with D2/D3/D4 applied and nothing else changed. The music prompts are rewritten. The exclude fields are unchanged — they measured 805 / 796 / 817 / 817, all inside 400–900.)*

### V1 · TWO OFF THE TOP — music prompt, rewritten to 955

Changes: the pole is *struck with a spanner* rather than *struck once with a spanner*, the capsule clause and the hats clause are gone, and *while everything else continues* → *while the rest continues*. **Nothing load-bearing left:** opening five seconds, spatial staging, arrival bar, the permanent offbeat move at bar 97, the missed relay click at bar 113, the kinetic defect in milliseconds, the acoustic declaration, ⛔ the guitar refusal, and the signature device all survive.

### V2 · I SAID IT TO THE DOOR — music prompt, rewritten to 951

Changes: *never moves or walks* → *never moves*; *rain recorded close on sheet metal* → *rain close on sheet metal*; the *no keys* clause dropped from the acoustic declaration. ⛔ **The break is described by mechanism only** — *a six-second funk drum break chopped into sixteenths* — with **no trade name and no player name in any pasted field, and the drummer credited by name in the dedication.**

### V3 · GO AND GET IT — music prompt, rewritten to 946

Changes: the *thin polite pad* clause dropped, *no vibrato and no ornament* → *no vibrato*, *no drum room* dropped from the declaration. **The hook repair (D3) is the important edit in this song** and it is a lyric edit, not a production one.

### V4 · I SAID IT AT THE END — music prompt, rewritten to 949

Changes: *at a level that suggests* → *at a level that says*; *dry room with no tail* → *no tail*; *the hammer figure* → *the hammer*. The eroding-section device stays in the signature-device slot, where it belongs — it is the form, and **the form is the argument.**

---

## ⭐ WHAT I CHECKED THAT THE SCRIPT CANNOT SEE

| check | result |
|---|---|
| **R1 — resistance indexed, subject of an active verb, never a person** | four objects, four verbs, four indices; **none is a person, none is *the system*** |
| **R5 — the cost is displacement, not strain** | ⭐ **zero strain words in any of the four**: no *hard, heavy, tired, ache, weight, carry, bear, endure, struggle* applied to the singer. Every cost is a **noun that moved**: a place, a chair, a coat, a walk, keys, a phone, a bus, a tab, a version of me, mornings, a key, a room. |
| **MacKaye — no consequence the singer cannot suffer** | V1 loses a queue position and a lift home · V2 is outside in the rain without keys · V3 is replaced by something cheaper · V4 loses years it cannot recover. **No jail, no injury, no violence, no borrowed danger.** |
| **THE STRANGER TEST** | ⛔ zero occurrences of *page, sign, signature, dawn, shut door, deadline, governor, pen-stroke as clemency* in any sung line. V1 has a pen — **on a chain, by a slot, and it is put back**, which is a shop counter, not a desk. **The wire supplied nothing; the sky supplied *force and volume are different measurements*, which is the whole R4 architecture.** |
| **⛔ forbidden-default #8 (the 138–150 chug)** | **zero distorted-guitar tokens in any positive field** (P1 holds), four non-guitar chassis, no half-time breakdown anywhere, no clean-verse/screamed-chorus flip, no key change, no drop-out bar. |
| **THE LEGIBILITY RULE** | one sentence per song after one listen: *she says no at a counter and goes to the back of the list* · *he is locked out and talks to the door* · *I refuse to be useful and say what I want instead* · *she stops getting up after years of getting up.* **Subject appears plainly in every one.** |
| **THE RETURN** | every chorus byte-identical ×3 · the Count / the door's answer / the hook / the chorus all return · **removal debt: none — nothing was stripped** |
| **ONE SEAM, N=1** | one dominant material + one intruder + one seam per song; **variety across the four, never inside one** |

⛔ `vault/AUTONOMY.md`: drafts on disk. Nothing rendered, published, or spent.
