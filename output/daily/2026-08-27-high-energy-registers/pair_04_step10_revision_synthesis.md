# PAIR 04 — STEP 10 · REVISION SYNTHESIS · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Register:** the flat refusal that costs you something, said anyway
**⛔ Template defeated:** rebellion as a pose · **Refused element:** ⛔ no distorted guitar anywhere

---

## ⭐ THE THREE REVISION PASSES, AND WHAT EACH ONE COST

### PASS 1 — THE INSTRUMENT (step 08 → step 09)

**Command:** `python3 output/daily/2026-08-27-high-energy-registers/p04_measure.py .../pair_04_step08_generation.md`

**Eight defects found; four of them hard.**

- **⛔ All four music prompts over the 1000 cap** (1148 / 1051 / 1071 / 1032). ⭐ **The instructive part is that I had no idea.** I wrote four prose paragraphs that *felt* dense-but-reasonable and every one was 5–15% over a hard render limit. **This is why the handoff says measure exactly with a script and never estimate** — and it is the second time in this pair's own record that an eyeballed number was wrong (see PASS 3).
- **⛔ R2 breached three times, and once inside a discriminator.** *"Out loud, in the rain"* (V2) and *"I want it out loud"* + **`free` in V3's hook**. All three read as idiom rather than as feeling-words, and **that is not a defence: the ruling names the token.**
  ⭐ **The V3 repair is better than the original.** *"It is a click away, and it is free."* → *"It is a click away, and it costs nothing."* The hook now speaks in the **currency the whole register is denominated in** — a pose is an empty cost column, and now the line that names my replacement prices it. **A gate caught a word and handed back a better line.**
- **⚠️ P2 falsified at first pass**: 5.21% shared sung lines against my own pre-registered ≤5.0%. Handled in the open at step 09 §D4 and again in the final package's verification table; **both numbers reported, the first-pass one scores the prediction.**

### PASS 2 — THE EAR (step 09 → the body)

**Command:** `scripts/measure_soundcraft.py :: profile()` + `strict_end_rhyme()` over the four sung-line sets.

- **Strict end-rhyme 0.418 / 0.369 / 0.412 / 0.500** against the 0.30 floor — ✅ THE RETURN holds without special pleading, and **V4, the plainest song in the pair, has the highest.**
- **Line-return 0.405 / 0.417 / 0.282 / 0.372** against 0.20 — every chorus byte-identical ×3. ⛔ **No defence is filed for that and none is owed** (`gates.yaml: chorus_repetition_requires_no_justification`).
- **⚠️ Alliteration under floor in two songs**: V2 **9.27**, V4 **7.47** against 11.0.
  ⭐ **Four repairs made, all of them genuine improvements rather than decoration:** *the last bus **back*** (×4, and *back* is the right word — it is the bus home), *nobody **buys** them back for a word* (×3 — an economic verb in a song about a debt that cannot be settled), *whoever **gets** up next* (which ties the room's next occupant to the song's own verb), and *nothing in the room **noticed*** . **V2 → 10.06, V4 → 8.51.**
  ⛔ **Then I stopped.** Both are still under the floor and both are **declared as FLAGs in the verification table with the reason**, because `gates.yaml` makes this a flag and not a fail, and because **the reason is real: consonant play sounds like someone enjoying the sentence, and enjoying the sentence is the pose.** V2 is a flat address by a man alone at a door; V4 is an inventory of a decade of mornings. Decorating either into compliance would have cost the register to buy a number.

### PASS 3 — THE TABLE AGAINST ITSELF

⭐⭐ **AND HERE IS THE ONE WORTH WRITING DOWN.** After PASS 2 I wrote the verification table — and **five of its cells were wrong**, because I had transcribed them from an earlier run of the instrument rather than re-running it against the final strings: V1's music prompt was 954 and I wrote 955; V2's lyrics field 4266 and I wrote 4256; V3's 4787 and I wrote 4778; V4's 4543 and I wrote 4547; V2's words-per-line 6.04 and I wrote 6.02.

> ⛔ **Every one of those was a number that had been TRUE once and was carried forward past the edit that changed it.**

**This is FINDING 12 happening to me, at pair scale, in the same session in which I wrote a paragraph about FINDING 12.** `check_field_census.py`'s own header says it: *a number does not become true by being copied forward.* The numbers were not invented — they were **stale**, which is worse, because a stale number looks exactly like a measured one and carries the memory of having been checked.

⭐ **The fix is not vigilance. It is sequence: the table is generated from the file it describes, after the last edit to that file, or it is not written.** Every row in the shipped table was re-measured against the assembled body **after** the alliteration repairs, and the dedication was measured rather than eyeballed (**466** against a 500 cap).

### PASS 4 — READING WITH STEMS INSTEAD OF TOKENS

⭐⭐ **The last pass found the two things every automated check had passed.** I re-read all four lyrics against a **stem** list (forbidden-default lexis + R5 strain words + cliché-#9 abstractions) rather than the exact-token list every instrument in this pipeline uses.

- **⛔ `louder` in V1's peak.** *"Not louder. Just again."* — and `louder` **does not match** `(?<![a-z])loud(?![a-z])`, which is the exact regex `check_field_census.py` uses. **It cleared every scan in this run while sitting one suffix away from a ruling that names the word.**
  ⭐ **The repair is better than the line it replaced:** *"Not bigger. Just again."* now returns against *"It does not get bigger"* in the chorus and *"Everything in here is at its biggest"* two lines above it, so the record's largest bar **states R4's own thesis inside the lyric** — biggest, smallest, not bigger. **The gate did not catch it; reading did.**
- **⚠️ `chain` ×3 in V1**, from forbidden-default item 17. **Kept, and argued for in Major Deviations 8**: the cliché is *chains as a bondage metaphor*, and mine is a ballpoint pen chained to a counter whose load-bearing line is a **measurement** — *"The chain is shorter than my arm."* ⛔ V1's EXCLUDE token was moved `chains` → `cages` so the negative field still forbids the class without instructing the generator against the literal object in the lyric.

> ⭐ **THE LESSON, AND IT GENERALISES PAST THIS PAIR: a word-boundary token scan cannot see an inflected form, so a rule stated as a word list is only as strong as the morphology of the words in it.** *loud* was on the list; *louder* was not; the song said *louder* and four instruments said clean. **The stem pass costs one command and it is the only thing in this pair's whole audit trail that found a defect no script could.**

---

## ⭐ WHAT DID NOT CHANGE, AND WHY THAT MATTERS

| held through every pass | why it was never up for revision |
|---|---|
| **`No.` in all four songs** | It is the refusal itself. Inventing four different single-syllable refusals to move a number I set myself is writing around a gate. **The falsified prediction is reported as falsified and the line stays.** |
| ⛔ **no distorted guitar** | The refused element held under exactly the pressure it was chosen for: I wrote four high-energy songs in a 128–136 band next door to the chug, and **the measured token count in the positive fields is zero.** P1 holds. |
| **the four resistance KINDS** — position / barrier / demand / time | The anti-convergence control, chosen at step 06 before a line existed. **P6 holds:** no two songs share a kind. It is the reason the four inventories do not rhyme with each other. |
| **the R6 decision** — no raised voice anywhere | Survived four passes without a single "just this once at the last chorus." **The temptation to lift the final chorus is the template's last door and it stays shut.** |
| **V3's second half** — *what I want instead* | ⭐ The part that makes it mine rather than a stance. **A pose refuses in order to be seen refusing; this refuses in order to get something, and says what.** HOPE as the doctrine defines it: appetite pointed at the future, unembarrassed. |
| **V4's eroding stretches** (10 → 8 → 6 → 4) | The one formal idea in the song. V4 ships at 78 sung lines, the bottom of the target band, **because the fourth stretch is four lines — padding it to hit a nicer number would delete the argument.** Declared in Major Deviations. |

---

## ⭐ THE SYNTHESIS, IN ONE PARAGRAPH

The register said *the flat refusal that costs you something, said anyway*, and the template said *the rebel who risks nothing*. **A pose is not a lie about feeling — it is an empty cost column**, which means the defence is not attitude but arithmetic. So all four songs are **ledgers**: a list that drops a name, a door that keeps a coat, a box that has a cheaper version of me a click away, an alarm that took a thousand mornings and does not give them back. Each resistance is an object with a verb and an index; each cost is a **noun that moved**, never a strain; each consequence is one the body standing there can actually suffer. The sound goes the other way on purpose — **the prior loaded in the first four bars, the exit at a bar I name, and the largest moment in every record containing the smallest vocal event in it.** ⛔ **And no distorted guitar anywhere, in a band that sits next door to the chug** — because the only refusal with a hard edge is at the level of the instrument, and because a pair about refusing something you want should refuse the thing that would have made it easy.

⛔ `vault/AUTONOMY.md`: drafts on disk. Nothing rendered, published, or spent. **A text SHIP is permission to render, never permission to publish.**
