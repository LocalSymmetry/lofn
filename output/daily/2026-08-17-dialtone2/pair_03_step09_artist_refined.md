# P03 — STEP 09 · ARTIST-REFINED
### `2026-08-17-dialtone2` · **DIAL TONE OF GOD, PART 2** · ACCESSIBLE · EXISTENCE · **AWE, terror-adjacent**

---

## 1. THE FIVE REFUSALS *(what was cut, and why the cut is the refinement)*

### R1 — ⛔ THE SONG KEPT TRYING TO EXPLAIN ITSELF
Three separate drafts grew a line whose only job was to tell the listener *why* she doesn't correct anybody — *"because a correction is a way of winning"*, *"because I would rather you kept it"*, *"because being right is not the same as being kind"*. Every one of them is a **defence**, and the seed forbids the defence.

**Cut, all three.** What remains is the bare statement: **"I am not going to correct you."** A decision reported, not argued. ⭐ The flatness is the point — a reason would make it generous, and it is not generosity, it is **acquisition**.

### R2 — ⛔ THE MISHEARING WAS TURNING INTO A PUNCHLINE
Early V1 and V3 both ran the minimal pair five or six times as a hook. That makes the song **about its own wordplay** — the listener starts admiring the trick instead of hearing the person.

**Refined to a fixed budget:** the pair appears **twice per song as a refrain couplet, once at the unison, once in the outro** — and the unison occurrence is the only one where both registers say it together. The joke never gets to be the subject; ⛔ **ONE SEAM, and the seam is where the two registers touch, not where the two words rhyme.**

### R3 — ⛔ THE TERROR KEPT ARRIVING AS WEATHER
First drafts of V2 and V3 reached for rain-as-mood and sky-as-vastness. That is the cheap AWE and it is also **another pair's territory** in this run.

**Cut. Replaced with structural facts of the actual site:** in V2 the hazard is *the rail is at my hip and the open side is four floors*; in V3 it is *the rail stops below my ribs and the deck moves when the lorries pass*. ⭐ **A drop is more frightening than a sky, because a body can fall down it.** The terror is now load-bearing geometry rather than atmosphere.

### R4 — ⛔ V4 WANTED TO END WARM, AND THAT WOULD HAVE BEEN A LIE
The sting variation kept resolving on *"and they slept"* — which is true, and is exactly half of the truth. A song that stops there is a greeting card about how mistakes are secretly wonderful.

**Refused.** V4 now ends on **the unlit street and the missed stop**, and the unison block states both halves flat and refuses to choose: *"Both of those are true at once. / I am not choosing one of them."* ⭐ **The error was better AND it cost.** That is the difference between the card's instruction and a slogan about it.

### R5 — ⛔ "SHE WAS RIGHT ALL ALONG" KEPT LEAKING IN
Several lines had her quietly winning: the accuser shown to be foolish, the fast answer vindicated. That is **spite in a soft voice**, and spite is part one's engine, not this one's.

**Cut.** The accusation is now conceded outright in every variation — *"I took the part where you think"* (V1), *"that is a real thing to fear"* (V2), *"an answer is a thing you use"* (V3), *"That is the fault. That is all of it."* (V4). ⭐ **She agrees with them and keeps going anyway.** That is the only shape available once contempt is off the table.

---

## 2. THE ONE ADDITION THAT MADE THIS A PAIR RATHER THAN FOUR SONGS

⭐ **Moving the mandated unison off the hook and onto the misheard word.**

Every earlier draft put the two registers together at the emotional peak, which is what every song does and which makes the unison a **production flourish**. Moving it to the error changes what the arrangement means: **the only thing both halves of her agree about is the mistake somebody else made about her.**

That single relocation is what turns the run's engine — *being blamed, answered with the error owned* — from an assertion into a **structure**. It is also what makes the pillar-4 requirement (*two registers, meeting exactly once*) thesis-bearing instead of decorative, and it is why the unison sits in a dedicated section marked `BOTH REGISTERS IN UNISON, the only time` in all four lyric sheets.

---

## 3. MEASURED REPAIRS *(before → after, canonical instrument)*

**Instrument:** `scripts/measure_soundcraft.py → profile()` for the return floors, plus a purpose-written parser for field bands, digit sweep, house-lexicon sweep and roster sweep. **Every number below was printed by code, not estimated.**

| | gate | **before** | **after** | floor / band |
|---|---|---|---|---|
| **V1** | MUSIC PROMPT chars | 993 ⚠️ *hug* | **949** | 850–1000, target 870–960, hug ≥985 |
| | sung lines | 83 ❌ | **91** | 88–110 |
| | allit / 100w | 11.57 | **14.26** | ≥11.0 |
| **V2** | MUSIC PROMPT chars | 987 ⚠️ *hug* | **957** | as above |
| | sung lines | 82 ❌ | **92** | 88–110 |
| | allit / 100w | 10.86 ❌ | **12.48** | ≥11.0 |
| **V3** | MUSIC PROMPT chars | 980 *(over target)* | **950** | as above |
| | sung lines | 83 ❌ | **90** | 88–110 |
| | allit / 100w | 8.09 ❌ | **12.71** | ≥11.0 |
| **V4** | MUSIC PROMPT chars | 983 *(over target)* | **951** | as above |
| | sung lines | 77 ❌ | **89** | 88–110 |
| | allit / 100w | 9.49 ❌ | **12.76** | ≥11.0 |

**How each repair was made — and it matters that none of them was cosmetic:**
- **Prompt chars.** Trimmed by removing *hedging adjectives*, never clauses: *"hazy and patient and never wavering"* → *"hazy, patient, unwavering"*; *"pulled a quarter-tone apart"* → *"a quarter-tone apart"*. ⭐ **Every one of the six named timbral entries survives in all four prompts, in order.** No entry was traded for characters — that is the trade the boundary-hugging flag exists to prevent.
- **Sung lines.** Added as **whole couplets inside existing sections**, never as a new section. The pair's form is *couplets over one unwavering groove*; adding a section would have made the song more episodic, which is the exact banned failure. V4 gained a *"Too fast to fix. Too fast to check."* run in Verse Two — which is the accusation, restated in the accused's own mouth.
- **Alliteration.** ⚠️ **The interesting one, and it must be reported honestly.** V3 measured **8.09**, well under the floor, and the diagnosis was real rather than instrumental: V3's diction had drifted toward *abstraction* (*answer · useful · trade · nobody*), which is vowel-heavy and consonant-poor. The repair was to swap abstractions for **objects** — *"You take it out of town"* → **"You take it to the tip"**; adding **"Rope and rust and the rock beneath"**, **"The deck drops. The deck drifts."**, **"The rail is low. The road is loud."** ⭐ **The floor caught a genuine craft failure, not a lexical one.** An anchor song was talking about anchoring instead of naming rope, rust and rock. This is exactly the ratchet the return floors were written for.

**Gates that passed unrepaired and were not touched:** `line_return` (0.330 / 0.337 / 0.344 / 0.382, floor 0.20) · `strict_end_rhyme` (0.615 / 0.543 / 0.422 / 0.449, floor 0.30) · words-per-line (6.55 / 7.14 / 6.56 / 6.96, ceiling 7.50) · lyrics field chars (4423 / 4645 / 4406 / 4488, target ≤4800) · exclude chars (750 / 750 / 753 / 759, band 400–900) · dedication chars (418 / 463 / 431 / 447, ≤500) · digits in sung lines (0 / 0 / 0 / 0) · house lexicon (0 hits, prompt and lyric, all thirteen phrases, all four songs).

---

## 4. ⚠️ THE TWO FLAGS THIS PAIR CARRIES INTO STEP 10 — RAISED, NOT HIDDEN

### FLAG A — NUMERAL TOKENS BEYOND THE ONE SUNG FACT
The one-fact rule is satisfied — **"forty-five"** is the only *numeric fact* sung, it sits at the hinge in all four, and it is **answered** (*"Your own chest does it all the time"*) rather than recited. But a validator counting numeral **tokens** will also see, per song, roughly ten instances of *one* (*"the fast one"*, *"one bulb out"*, *"the slow one"*), *four* (four floors, four in the morning), *half*, *first*, *second*.

**These are object counts and ordinals, not data**, and the great majority are the word *one* used as a pronoun. ⭐ Raised here so nobody has to discover it in QA and wonder whether it was noticed. **No repair taken** — replacing *"the fast one"* with a synonym would damage the lyric to satisfy a token counter, which is precisely the *fix-the-scanner-not-the-line* failure (L27).

### FLAG B — THE TENDER/FLAT DISTINCTION IS UNFALSIFIABLE UNTIL A RENDER EXISTS
⛔ **This is the pair's whole prosody axis and no text gate can see it.** Every instrument I own reads words; the subtraction between the tender pass and the flat pass is a *performance* event.

**The mitigation is structural and it is already in the lyric**, per THE GRAIN LAW: the tender pass floats in off the beat and the flat pass starts on the downbeat, so two different **rhythmic** events exist in the writing whether or not the generator obeys the vocal spec. **But that is a mitigation, not a measurement.** ⚠️ A render audit under THE BLIND RULE is the only instrument that can return NO here, and it does not exist yet. **Not claimed as passed.**

---

## 5. VOCAL AND ARRANGEMENT REFINEMENTS CARRIED INTO STEP 10
- **The tender register is not a whisper.** Specified as *close and full*, and the whole ASMR/breathy bloc sits in EXCLUDE — a whisper collapses two registers into one grey texture and kills the axis.
- **The celesta is never softened.** *Struck hard from the wrist, hammers audible before tone* in all four; *soft twinkling celesta, music box, glockenspiel sparkle* sit in EXCLUDE.
- **The half-time must drive.** Every prompt states the mechanism explicitly — *thirty-seconds that never land square, so the backbeat stays slow while the record drives* — because a generator told only *"half-time"* returns a sag.
- **Nothing leaves.** The parade accumulates; no instrument is removed for a build, because a removal read as a drop would be the banned dynamic event.
- **The ending is flat, and it is terminal.** The tender register is absent from every outro. *She keeps the wrong word; she does not get to be tender about it.* ⛔ **Terminal, never fading** — terminal silence survives the generator, a fade is smoothed.
