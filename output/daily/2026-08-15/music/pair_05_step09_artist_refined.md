# P05 — STEP 09 · ARTIST REFINEMENT PASS
**Run:** `2026-08-15_daily_music_the_resolution` · **ICB sha:** `7d4dd66f2a6e` · **Pair:** P05 · **Bed B4**
**Refiner:** LOFN-PRIME, AWE mode — terror-adjacent, warm. **Tone gate:** *would Delphine find this
condescending?*

---

## 1. THE FOUR CUTS THAT DID THE MOST WORK

### CUT 1 — every elegiac verb, replaced with a working verb
The step-08 drafts kept reaching for the wrong register. Each of these was found and killed:

| drafted | why it failed | shipped |
|---|---|---|
| *"and the sea will take you and there is nothing I can do"* | helplessness; invites pity | *"and the sea can have the rest of you in the morning"* |
| *"a whole day gone by morning"* | *gone* is elegy grammar | *"and it will be gone, and it was worth the morning"* |
| *"the tide is a thief and it comes for everyone"* | villain; and it is a thesis | *"and it is not a thief, it is the evening arriving"* |
| *"nobody will ever see this one"* | ⛔ the banned thesis, verbatim | **deleted, not repaired** |
| *"I give my whole day away for an hour"* | martyrdom | *"done for an hour is still done. That is the price."* |

⭐ **The general rule extracted:** the sculptor is never the *object* of a sentence about the sea. He is
always the **subject of a verb of work** — packing, wetting, pressing, leaving alone, standing up, walking
home. When he is the object of the tide's verb, the song is pitying him.

### CUT 2 — the second numeric fact, four separate times
`nine ridges deep` · `seven this morning` · `the four hundredth mane` · `six hours in your brow` ·
`sixty minutes` · `two hours` · `three faults`. **All cut on sight.** ONE-FACT RULE: only **eleven feet**
survives, spelled in words, at the hinge. Everything else that wanted to be a number became a **body
measure** instead — *a hand's width, two of you nose to tail, a boat length, the thickness of a fingernail* —
which is how the countdown in V2 gets its shape without breaking the gate.
⚠️ **FLAG raised here, not hidden:** V2's refrain **contains** the fact (*"Eleven feet and closing"*), so the
phrase measures **five times** in that variation — four refrain returns plus the hinge line *"Eleven feet was
an hour ago. Now it is a hand."* Considered and kept: it is **one fact**, and it is **answered in the same
breath every single time** (*"…and I am in no hurry"* / *"Now it is a hand"*). That is the definition of
*responded to, never recited*. If a human read disagrees, the repair is one line and the refrain still scans.
**V1, V3 and V4 measure it once each.**

### CUT 3 — negation, out of the style field
Three first-draft prompt clauses read *"brushes never leave the snare"*, *"the bed never fully stops"*, and
*"no ride cymbal"*. **Suno reads negation as a positive token.** All rewritten as positive instruction —
*"brushes stay on the snare head throughout"*, *"the bed continues under everything"* — and the refusals
moved into EXCLUDE where they belong. ⭐ This is also what makes the **DEVIATION CLAUSE** enforceable: the
EXCLUDE line is `swing ride cymbal ostinato`, which forbids the ride **as a bed** while permitting the ride
**as an arrival** — the one disobedience I said in advance I would keep.

### CUT 4 — the room
Two drafts had somebody watching *with* him — a boy holding the pail, a woman on the rail. **Both cut.**
P05's residue mechanism is **ADDRESS TO THE WORK**, and a second human on the sand gives the song a listener
who could answer. The only other people in the finished songs are **shoes on the planks** and **a man with a
box** — neither is addressed, neither is looked at, and the sculptor never turns around.

---

## 2. THE FORM, TIGHTENED

**The rhyme chain, finalised per variation.** Each variation gets its **own anchor family**, so the four are
not one song four times:

| variation | anchor family | anchor word | the wave |
|---|---|---|---|
| **V1** | `-ing` | **morning** | every chain bends toward *morning* as the refrain approaches |
| **V2** | mixed, resolving to `-ing` | **morning** (negated: *"never promised a morning"*) | the chain shortens as the distance shortens |
| **V3** | `-ice` | **twice** | one dominant family the whole song — the most wave-like of the four |
| **V4** | `-ight` | **tonight** | the light going off the lion is also the rhyme going out |

**The collapse, placed.** In all four, the long free-verse line breaks for the seam — *"Palm flat. Press.
Lift. Wet clap. Again."* — and it lands **at the moment the sea reaches the plinth**, then the long line
returns. That is the ONE JOIN, and it is in the LYRIC and the FORM, not only in the production spec.
⭐ GRAIN-LAW compliance restated: **a Somatic objection answered only in the PRODUCTION SPEC is not
answered.** The seam is audible because the words get short.

**The return, paid.** Rhyme density is 1-in-4 by assignment and will measure under
`rhyme_return_floor: 0.30`. The debt is discharged by a **byte-identical four-line refrain returning four
times** in every variation, plus the anchor line returning inside it. ⛔ No defence is offered for the exact
repetition and none is required (`chorus_repetition_requires_no_justification: true`).

---

## 3. THE SOURCE-2 WELD, RE-SEATED

Both case-exact phrases were initially placed decoratively in the middle of the instrumentation list. **Moved.**
- **`superhumanly funky bass`** now sits **immediately beside its own contradiction** — *"a woody upright
  bass, right of centre, felt in the chest more than heard, a superhumanly funky bass playing chamber-quiet"*.
  The phrase is doing violence to the instrument it modifies, in the same clause. That is the seam.
- **`downright salacious slow jam`** now modifies **the tenor's dusk phrasing over sacred modal changes** —
  the most against-the-grain placement available, because spiritual jazz is the one idiom that phrase cannot
  belong to.
⭐ **Both are in the generator-facing MUSIC PROMPT of all four packages, case-exact, never paraphrased.**

---

## 4. THE BED'S DEFECT, MADE UNSKIPPABLE
Draft prompts described the bed beautifully and then let it be beautiful. **Panel C's finding: without a
named defect the bed becomes the song.** Every prompt now carries all three defects in load-bearing
positions, and the room defect is attached to the **voice** (where it cannot be smoothed as an instrument
choice) rather than to the mix:
> *"…carrying a room return a half-second longer than the space around her would give."*
Plus the kinetic defect, which is the fourth-line rhyme translated into the rhythm section:
> *"…landing one note behind the beat every fourth bar so the groove limps warm."*

---

## 5. TITLES — CHECKED AGAINST EVERY BAN
| # | title | concrete noun / first-person act | banned nouns | `water`/`sky`/`dark`/`current` | persona prefix |
|---|---|---|---|---|---|
| V1 | **The Last Ridge of Your Mane** | noun: *ridge* | none | none | none |
| V2 | **Eleven Feet and Closing** | measure + act | none | none | none |
| V3 | **I Pack Your Paw Twice** | first-person act | none | none | none |
| V4 | **A Box Clicked Behind Me** | noun + act | none | none | none |
⭐ *Tide* and *sea* are P05's assigned tokens and both appear in the lyrics; neither is needed in a title,
so no title carries the collision risk at all.

---

## 6. WHAT I REFUSED TO REFINE
- **The absurd ears.** *"Goodnight, lion. Your ears are perfect and absurd."* Three passes wanted to cut it as
  tonally loose. **Kept.** It is the single line that proves he is not grieving — a man saying goodbye to
  something he loved does not make a joke about its ears. It is the whole tone gate in eight words.
- **The thin place behind the ear** (V4). A named fault in the work, kept on purpose, claimed out loud:
  *"it is a fault and it is mine."* That is competence with affection in it, which is what this pair is for.
- **The pot on the stove a few streets up** (V2). It reads as sentiment for about a second and then it is
  just supper — *"and I am the last one down here, and I am beat… and I will be back before the light, and I
  will eat."* He goes home and eats. That is the least tragic ending available and it was chosen.

---

## 7. FINAL MEASURED STATE HANDED TO STEP 10
| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| MUSIC PROMPT chars | 949 | 947 | 922 | 939 |
| EXCLUDE chars | 799 | 810 | 787 | 796 |
| LYRICS chars | 4668 | 4796 | 4796 | 4796 |
| sung lines | 76 | 76 | 76 | 76 |
| `strict_end_rhyme` | 0.250 ⚠️ | 0.303 | 0.263 ⚠️ | 0.342 |
| `line_return` | 0.211 | 0.211 | 0.211 | 0.211 |
| `words_per_line` | 10.21 ⚠️ | 10.54 ⚠️ | 10.64 ⚠️ | 10.80 ⚠️ |
| `allit_per_100w` | 12.24 | 11.24 | 13.10 | 11.21 |
| DEDICATION chars | 482 | 497 | 491 | 491 |

⭐ **V2's alliteration measured 10.24 on the first pass — below the 11.0 floor — and was repaired by six
targeted edits** (*slumps/set*, *smoke slides*, *simpler to stand*, *take the evening / take the walk*,
*beneath/begun*, *no fault*) at net zero character cost. It was found by running the instrument, not by
reading the page.
