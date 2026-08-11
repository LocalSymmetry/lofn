# PAIR 04 — STEP 11 · ENHANCED FINAL PACKAGES
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P04 — BELOW THE THRESHOLD**

**Continuity payload verified before any creative work.** `output/daily/2026-08-08/CREATIVE_CONTEXT.md` · LF-normalised **142,900 bytes** · sha256 **`9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a`** — **exact match**, read in full. Personality DNA **27,796 B** inlined (`lofn-prime-mini.yaml` 2–326, THE ARCHIVE quarantined) · **18 baseline seats**, Hyper-Skeptics at 6/12/18 · **15 Special Flairs — marker present.**
**Arm / Axis B / Mode:** AMBITIOUS · NEWS · `LOFN-PRIME (INDIGNATION mode — cold, procedural, no heat)` · **eligibility 0–3/7 by design.**
**Inputs:** `pair_04_step10_revision_synthesis.md` (unmodified) · `06_music_handoff.md` (**overrides every step file**) · `skills/music/steps/11_Generate_Music_Enhancement.md`.
**Scratch:** `_work/pair_04/` only. **Nothing outside this file was written.**

---

## 0 — ⛔ GOLDEN-OUTPUT QUARANTINE — REFUSAL DECLARED

The step-11 contract's "Output MUST include" list has been read and **item 2 is refused as written**: the file's own §Integration and §head-of-file forbid emitting golden-song payloads into a generating context, and the handoff's §1 overrides the contract on this point. **No past shipped style prompt, lyric, title, hook or arrangement appears anywhere below.** The Golden Songs' bare names travel in the handoff for the *judge's* blind comparison and are not repeated here, looked up, reconstructed, or calibrated against.

**GOLDEN MOVE rules this pair is calibrated against** (instructions, never an exemplar): **(1)** a real place a body stands in, named in the first thirty seconds — *a corridor, no chair, the screen held at her own forearm height*; **(2)** one wounding fact **responded to, never recited**, spelled in words, at the hinge — and the response is a **speed**; **(3)** a mid-song turn the singer performs and does not understand — she files her own row correctly; **(4)** a register rotated away from anything this house shipped recently — 7/8, soprano, procedural diction, one dry pair out of six.

---

## 1 — ⭐ THE ANDON CORD — THE VERDICT

> ### GATE: **REPAIR-AND-ENHANCE. No REJECT issued.**

I was told the pair itself declares V2 "the pair's fourth-best song and could not be lifted inside this chain," and that this is an invitation to adjudicate rather than a disclaimer to accept. **I adjudicated. The ruling is below and it goes against the pair.**

**The five REJECT criteria, tested rather than waved through:**

| # | Criterion | Finding |
|---|---|---|
| 1 | **THREAD LOSS** | ❌ not present. The gesture (D13), the crossing (D3), the sung number, the corridor, D1/D4/D6/D8 are all intact and load-bearing. |
| 2 | **PERSONALITY COLLAPSE** | ❌ not present. This is not default Lofn — no industrial grief, no category-theory register, no somatic machinery. It is LOFN-PRIME's INDIGNATION dial run **cold**, which is the assignment. |
| 3 | **EMO TAXONOMY FAILURE** | ❌ not present. 12/12 headers, four slots, all emotions in `EMOTION_TAXONOMY`, zero bare architectural labels. The arc *deliberately* refuses to transform at the final refrain — see §7 Major Deviations, where that refusal is declared rather than smuggled. |
| 4 | **GENERIC OUTPUT** | ❌ not present. Ninety line-ends on one vowel, a through-composed 7/8, a phase-locked bell mechanism. Whatever else this is, it is not functional quatrains. |
| 5 | **PROMPT FORMAT VIOLATION** | ⚠️ **PRESENT, and repaired, not rejected.** All four step-10 prompts opened on a mood clause (*"Cold and procedural."*, *"Flat, competent, faintly impatient."*, *"Careful and unmoved."*, *"Level and end-of-shift."*) rather than **genre × tempo × key**, which Gate 14a makes mandatory. Additionally **all four were mis-measured** — see §2. Both are repairs in place, not corpses. |

**"Don't polish a corpse" cuts both ways: it is also not licence to shoot a live one.** The defects found here are a measurement error, a heading convention, an arithmetic error in the pair's own documentation, and two admitted thin strophes. Every one of them is fixable inside step 11. **Returning this to step 09 would cost the run a working pair to repair four sentences.** REJECT withheld, and the reasons are itemised so QA can overrule me.

---

## 2 — ⚠️ THE MEASUREMENT FINDING — the pair's char counts are WRONG and the coordinator's were RIGHT

I was told the pair reports 955/952/953/945 and a coordinator extractor got 989/969/963/969, that one is wrong, and that it matters. **I measured the exact shipped strings with `len()` before concluding anything.**

```
python3 -c "extract each fenced MUSIC PROMPT from pair_04_step10_revision_synthesis.md, LF-normalise, .strip(), len()"
V1: claimed 955  ->  measured 989   V2: claimed 952  ->  measured 969
V3: claimed 953  ->  measured 963   V4: claimed 945  ->  measured 969
```

> ⭐ **The coordinator was right and the pair was wrong.** This is the first re-stat in three runs to break that way, and it is recorded plainly because the handoff's §6 standing observation ("every time a pair agent's claim was properly re-measured, the agent was right") is a real prior that has now taken a hit. **A prior that is never allowed to lose is not a prior.**

**Why it matters, in gate terms — all four figures were reported as "✅ in band" and three of the four claims are false:**

| | reported | true | `music_prompt_chars` 850–1000 | `music_prompt_chars_target` 870–960 | `music_prompt_hug_ceiling` ≥985 |
|---|---|---|---|---|---|
| V1 | 955 | **989** | pass | ❌ **outside** | 🚩 **FLAG `boundary_hugging` FIRED** |
| V2 | 952 | **969** | pass | ❌ **outside** | no |
| V3 | 953 | **963** | pass | ❌ **outside** | no |
| V4 | 945 | **969** | pass | ❌ **outside** | no |

**Provenance of the error:** the figures were written at step 09 §3 as a forward promise (*"trimmed from 986–1,111 chars into the 870–960 target band (945–955)"*) and step 10 **transcribed the promise instead of measuring the delivery.** Nothing was re-measured between step 09 and step 10; the numbers travelled as a claim.

**Repaired.** All four prompts are rewritten, genre-first, and measured on the exact shipped string: **959 / 938 / 909 / 920** — inside the target band, none within 26 chars of the hug ceiling.

---

## 3 — ⛔ THE L22 SCAN — THE GRAIN LAW, applied to the crossing

> **A form objection answered in the PRODUCTION SPEC is not answered.** The crossing's mechanism here is **rhythmic**, which is why it can survive at all. The question is whether any part of it lived *only* in a prose sentence in the style prompt.

### 3.1 ⭐⭐ DEFECT 1 — the pair's central arithmetic claim is FALSE

The pair states, in step 06, step 08, step 09 and step 10, that the two bells **"coincide at the third refrain only."** I ran the phase arithmetic against the pair's own declared bar structure rather than trusting the sentence.

**The mechanism as declared:** B♮ on every downbeat; B♭ every 8 eighth-notes from bar one, never resetting. A 7/8 bar is 7 eighths. A downbeat coincides with the walking bell whenever the elapsed eighth-count is ≡ 0 (mod 8). Since 7 ≡ −1 (mod 8), that is **every eight 7/8 bars** — a free-running 7-against-8 pulse coincides *periodically*, roughly every 17 seconds.

```
bar plan  6 / 12 / 12 -R- 12 -R- 12 / 10 -R- 10 -R- 6      (80 bars of 7/8 + 16 of 4/4 = 688 eighths)
downbeat coincidences in the whole song: 14, not 1
   Intro b1 · S1 b3 · S1 b11 · S2 b7 · S3 b3 · S3 b11 · S4 b7 · S5 b3 · R3 b1-b4 · S6 b1 · S6 b9
```

**"They coincide at the third refrain only" is not true and was never true.** It is stated four times in the chain and nobody checked it. Left in place it is a claim the render audit would falsify and read as the *device* failing, when in fact the device is fine and the *sentence* is wrong.

### 3.2 ⭐ THE REPAIR — the true statement is better than the false one, and it is what ships

⭐ **A 4/4 bar is eight eighth-notes — exactly the walking bell's period. So the 4/4 refrain is the only place in the song where the drift STOPS.** In 7/8 the gap moves an eighth every bar; inside a refrain the gap is frozen at whatever it was on entry and holds for four bars. The event is therefore not *"a coincidence happens"* — a coincidence is the machine ticking, and it happens fourteen times. **The event is that the held gap is ZERO, and that happens exactly once.**

| | gap held for four bars | what a listener hears |
|---|---|---|
| **Refrain 1** | **six eighths** | two ticks, the second arriving late and staying late |
| **Refrain 2** | **two eighths** | two ticks, close, a steady limp |
| ⭐ **Refrain 3** | **NOTHING** | **one tick.** Four bars of it, spilling one bar into Strophe 6 before the walk resumes |
| **Refrain 4** | **two eighths** | the limp again — identical to Refrain 2, and **nothing happens at it** |

**This satisfies L38 / N = 1 rather than breaking it.** The seam is not *"the bells touch"* — that is the material, and material is supposed to recur. The seam is *"the distance they hold is nothing,"* and there is exactly one of those. A form rule that counted every tick of a periodic machine as a crossing would be unsatisfiable by any periodic mechanism, and the ICB's own return vocabulary (byte-identical chorus, single-vowel end-rhyme chain) is periodic material with one seam in it.

⭐ **And it hands the Maximalist a falsifiable render-audit question, which the old claim could not.** His condition was that the gap be audible, and *"does anything happen at the repeated section"* is a question a listening model can answer wrongly in both directions. The new question cannot be fudged:

> **THE BLIND RULE question for `lofn-render-audit`, audio alone, prompt withheld:** *"There are four identical four-bar sections. In each one, two small bells tick. Is the distance between the two ticks the same in all four sections? If not, which one is different, and how?"*
> A correct render answers: **the third — in the third they are one tick.** Anything else and the argument was smoothed away.

### 3.3 WHERE THE MECHANISM NOW LIVES — five places, one of them prose

| Location | Suno-bound? | Carries |
|---|---|---|
| `[SONG FORM:]` line, first two lines of the lyrics field | ✅ **yes** | the bar plan and all four held gaps, in words |
| **The lyric twin**, final sung line of Strophe 5 in all four | ✅ **yes** | the crossing performed in her mouth, abutting the bells' |
| **Two short SFX cues** inside the lyrics field | ✅ **yes** | the two bells at the top, the single strike at Refrain 3 |
| `## 4. PRODUCTION SIDECAR` Disc_Channel | ✅ **yes** | the two bells addressed at token level on the Texture channel |
| The MUSIC PROMPT paragraph | ✅ yes | the mechanism in prose — **one of five, no longer the only one** |

⛔ **Nothing was added in the mix.** No duck, no pan, no send, no automation, no level move at the coincidence, in any of the four. The crossing is in the notes and in the lyric, per D3 and L22.

### 3.4 ⭐ DEFECT 2 — THE LYRIC TWIN DID NOT FIRE AT THE SAME MOMENT IN ALL FOUR

I was told to check the claim that each twin occurs exactly once. **It does — that half of the claim is true.** The other half is not.

```
V1 "A note. A no. A note."            1 occurrence — Strophe 5, line  5 of 10
V2 "A no, and a note, and a no."      1 occurrence — Strophe 5, line  4 of 10
V3 "I type no. I say yes. I type no." 1 occurrence — Strophe 5, line  5 of 10
V4 "I go. I do not go. I go."         1 occurrence — Strophe 5, line  3 of 10
```

**Three different positions, none of them adjacent to the bells' event.** Step 10 §9 claims the spoken crossing happens *"at exactly the same moment"* as the bells'. It does not: it fires three, five and seven lines early, and at a different offset in each variation. In V1 five lines of explanation sit between the twin and the refrain it is supposed to coincide with.

**Repaired: the twin is now the FINAL sung line of Strophe 5 in all four**, immediately abutting the third refrain and its SFX cue. It cannot be moved into the refrain itself — the refrain is byte-identical on all four returns and mutating it to carry an event would destroy the only thing that makes the third return legible. **So the order is: her mouth does it, then the bells do it.** Same structural moment in all four; one occurrence each; verified mechanically in §8.

### 3.5 THE REST OF THE L22 SCAN, per variation

| | L22 defect found | disposition |
|---|---|---|
| **V1** | Prompt opened on mood, not genre. Twin 5 lines early. **S5 carried a vowel-fed tautology** (*"The row is a row. It is told."*) with no hand in it. | all three repaired |
| **V2** | Prompt opened on mood. Twin 4 lines early. **S3 line 7 and the whole of S6** were vowel-fed. | all repaired |
| **V3** | Prompt opened on mood. Twin 5 lines early. **The disposition code and the read-back were in the wrong causal order** for the twin to land last. | repaired by re-chaining S5 |
| **V4** | Prompt opened on mood. Twin 3 lines early — the worst of the four. | repaired |
| **all four** | ⚠️ The one SFX cue per song was **90+ characters** and does not match `validate_suno_packages.py`'s standalone-cue pattern (≤40 inner chars). The pair reported "SFX ≥1 ✅"; the validator would have found none. | repaired: two short cues per song |

---

## 4 — ⭐⭐ THE RUN'S ONE SUNG NUMBER — verified, not asserted

**`max_sung_numeric_facts: 1` across the entire run. This pair spends it, in V1 only.**

> **`Four point five is the bar. I know`** — spelled in words, **mid-line**, never at the rhyme position, at the hinge (Strophe 4 of six), and **answered by a SPEED rather than a gloss**: *`I put the figure in. Not slow. / I have never put one in slow.`*

**Kept unchanged.** It is the single best-executed instruction in the pair and I refuse to touch it. She does not slow down, because slowing down would be an opinion and she has not got one.

**Verified mechanically across all four sung-line sets** (§8): **zero digits** in any sung line of any variation; the number-words `four`, `point`, `five` appear **only** in V1 Strophe 4. The ordinal `first` (V1 S1) and the determiner `one` are neither digits nor numeric facts — the gate's own wording is *"the corpus recited four data points as a weather report,"* and this pair recites nothing. **V2, V3 and V4 sing no numeral of any kind.**

⭐ And the structural consequence the pair found and I am keeping: **because the allowance is spent on the rounding, the number of rows is unsayable in all four songs.** A listener never learns how many there were. An inventory you are forbidden to count is not an inventory; it is a shift. **That is D7 enforced by arithmetic instead of by discipline.**

---

## 5 — ⚠️ THE TWO LINES THE PAIR NAMED AS VOWEL-DRIVEN — fixed, not accepted

The pair's self-critique admits: *"in a few places (V1 S5, V2 S6) the line exists because the vowel needed feeding rather than because the hand needed to do something… A reader who is unmoved by the constraint will read those lines as thin, and they will be right."*

⛔ **A disclosure is not a repair.** I went to both coordinates.

**V1 Strophe 5 — before:** the strophe's spine was `The row does not ask what I hold. / The row is a row. It is told.` The second of those is a tautology carrying an /oʊ/ and nothing else, and the strophe's only physical verbs were *put*, *sign*, *send*, *close*.
**After:** the box is given a verb and an appetite — `It does not ask what I hold. / It asks for a figure. I know / which figure it wants. And it goes / in, and the box takes the row.` The tautology is gone; the form is now doing something *to* her rather than sitting there rhyming; and the strophe ends on the twin.

**V2 Strophe 6 — before:** eight lines in which she does nothing at all. `Nothing about it was slow. / Nothing about it was told / to me by a thing I could hold.` is three lines of vowel with no object and no hand, and `It sits in the same kind of row. / It sits with the ones down below.` is one line written twice.
**After:** every line has an object or an action of hers — she **scrolls**, the row **sits at the same width**, the **map folds**, the **dot comes off the road**, and then the wince that was not there before:

> **`The letter I fixed is in code.`**

Her one human act in the entire song — correcting a single letter in a borrowed place-name — is now machine-readable, filed, and invisible. **The song still does not scold her (D6): she was right to fix it, it was skilled, and it changed nothing.** That is the pair's own thesis arriving in an image instead of in a rhyme.

**Third fix, same defect class, not on the named list:** V2 S3's `The row is the shape of a row.` → `I pull the two up in a row.` — a hand on a screen, and it now sets up the two lines that follow instead of paraphrasing them.

---

## 6 — ⭐ V2 — THE RULING THE PAIR ASKED FOR

The pair filed: *"V2 is the pair's fourth-best song and I could not lift it inside this chain… its subject — the one that cleared — is structurally the least interesting position on the corridor, because the thing that cleared needed her least, which is exactly what the variation is about and also exactly why there is the least for a body to do in it."*

> ### ⛔ **THE DIAGNOSIS IS OVERTURNED. The song is not weak because of its subject.**

V2 contains the strongest continuous stretch of writing in the whole pair, and the pair ranked around it. Strophes 2 and 4 are a strictly-caused machine chain (**it flags → the flag pulls a map → the map puts a dot → the dot needs a name → the name is borrowed from the nearest → the spelling can then be checked**) with a different verb from her at every stage, ending on a village given a load it did not ask for and a woman fixing one letter. **That is the run's binding decisions executed better than anywhere else in the pair.** It is not the subject that failed.

**What actually failed is local and was fixable in eleven lines:** four of V2's six strophes had no object in her hands, and the /oʊ/ spine filled the vacuum. The pair diagnosed a *concept* problem and therefore concluded it was unliftable; it was a *staging* problem. **A song where the protagonist has nothing to do is not the same as a song about someone who is not needed** — the second is the brief, the first is a defect, and they were being confused.

**Ruling:** V2 lifted, not rejected, and **moved from 4th to 2nd**. The Small Room seat's objection — that V2 asks one inference more than the other three — is **recorded and upheld**: it still should not lead. The pair was right that it is not the lead card and wrong about why it was fourth.

## 6.1 — RANKING (re-ranked after the repair; the step-10 order is stated so the change is visible)

| | step-10 rank | **step-11 rank** | why |
|---|---|---|---|
| **V3 · The Free Text Box** | 1 | **1** | Unchanged and unchallenged. The clearest body, the strictest chain, the sharpest wince — *she reads it back, she gets it right, and the caller thanks her* — and the one variation where the D6 trap was live and was defused rather than avoided. **Lead with this one.** |
| **V2 · The Map** | 4 | ⭐ **2** | The machine chain and the borrowed name are the pair's best writing. With S3/S6 restaged it has more physical action per strophe than V1 or V4. Still not the lead card: it asks one inference more. |
| **V1 · The Second Decimal** | 2 | **3** | Carries the run's one sung number and the purest procedural surface. Its arithmetic is the concept undisguised — a strength and a ceiling — and after the S5 repair it is the most *legible* song in the pair, which is not the same as the best. |
| **V4 · The Far Door** | 3 | **4** | The most formally satisfying ending and the only place the room is named. Flattest by design and **at the highest risk of rendering as atmosphere**; its twin repair (three lines early → last line) buys it the most of the four, and it still ranks last. |

---

## 7 — MAJOR DEVIATIONS

- **Changed / refused / intensified:** ⭐ **The bar-count claim.** *"The two bells coincide at the third refrain only"* is arithmetically false and is replaced everywhere with the true and stronger statement: the 4/4 refrain is the only place the drift stops, and the **held gap is zero exactly once**.
  **Reason:** a 7-against-8 free-running pulse coincides 14 times in this song. Shipping the false version would have handed the render audit a claim it could falsify while the mechanism was working.
  **Effect on Lofn uniqueness:** the device becomes checkable rather than asserted, and the Maximalist's unwithdrawn objection acquires a question a listening model cannot fudge.

- **Changed / refused / intensified:** ⛔ **REFUSED — the Body Noise Mandate (≥3 instances of breath/hum/vocal fry).**
  **Reason:** three separate collisions. **(1)** The pair's vocal configuration is Phase-1 non-negotiable DNA — *high and plain, no vibrato, never lands late, no audible breath at phrase ends* — and a singer who is never late is a singer who is never moved (D2). Breath at phrase ends is exactly the feeling-cue the register forbids. **(2)** It is the pair's differentiation slot: P04 is the run's only soprano and its only dry pair; body noise is P01's and P06's territory. **(3)** The handoff §4 warns in writing that a wordless return device can satisfy the return floor by itself (measured 2026-08-06: stripping one hum moved `line_return` 0.289 → 0.044). Adding three hums to a song whose return vehicle is **entirely lexical** would corrupt the one measurement this pair can currently make honestly.
  **Substituted, not merely refused:** the non-vocal body of this song is **the two bells**, and the SFX budget is spent on them — two short cues per song, at the top and at the crossing.
  **Effect:** the pair keeps the only vocal configuration in the run that has no body in it, which is the point of a person who is never moved.

- **Changed / refused / intensified:** ⛔ **REFUSED — "bridge and final chorus should transform."**
  **Reason:** this pair has no bridge by construction, and **the fourth refrain not transforming is the argument.** The song's whole claim is that the machine ran, the one legible moment passed, and the next return is indistinguishable from an earlier one. A final refrain that transformed would be the singer arriving at the insight (**D1 breach**) or the vindication the run bans (**D4 breach**).
  **Effect:** the EMO arc transforms across the *strophes* — Composure → Ennui → Equanimity → Detachment → Unconcern → Apathy → Indifference — and is byte-identical `Composure` at all four refrains, on purpose.

- **Changed / refused / intensified:** **Disc_Channel relocated out of the lyrics field** into `## 4. PRODUCTION SIDECAR`, per the single harness decision for all six pairs and the escape hatch in the step file's own §1 (*"move the Disc_Channel block to a Production Sidecar outside the lyrics field — the render field wins; note it"*). **Noted.** Gate 13a's in-field requirement yields to the harness decision and to the field cap.

- **Changed / refused / intensified:** **All four style prompts rewritten genre-first** (Gate 14a) and the mood clause deleted; **all four re-measured** into the 870–960 target band; **negations moved out of the style field** into the exclude field, including the vocal spec, which is now stated positively (*"a straight unwavering tone"* rather than *"no vibrato"*).
  **Effect:** you cannot ask a machine to not-do something. The step-09 seat argued this and the prompts had only half-adopted it.

- **Not changed, deliberately:** the four titles, the key, the meter, the tempi, the vocal configuration, the refrains (byte-identical, **no justification filed and none owed**), the sung number, the one device, and the ranking of V3 first. **Nothing was lengthened for its own sake:** sung-line counts are 88/88/88/88, unchanged from step 10.

---

## 8 — ⭐ THE COMPLETE GATE ENUMERATION, MEASURED ON THE SHIPPED STRINGS
*(handoff §4. **Every gate reported with a measured value, including the passes.** Numbers from `vault/gates.yaml`, cited not restated. Lyric figures from `scripts/measure_soundcraft.py → profile()` per variation — never eyeballed.)*

### 8.1 HARD gates

| Gate | Threshold | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `music_prompt_chars` | 850–1000 dense paragraph | **959** ✅ | **938** ✅ | **909** ✅ | **920** ✅ |
| `music_prompt_terminal_punctuation` | true | `.` ✅ | `.` ✅ | `.` ✅ | `.` ✅ |
| `suno_lyrics_field_max` | < 5000 whole field | **4,653** ✅ | **4,714** ✅ | **4,782** ✅ | **4,692** ✅ |
| `sung_lines` | 70–120 | **88** ✅ | **88** ✅ | **88** ✅ | **88** ✅ |
| `step06_min_facets` | ≥ 8 | **9 weighted facets**, weights sum 1.00 ✅ (unchanged from step 06) | | | |
| `total_prompts` | 24 across 6 pairs | **4 delivered from P04** ✅ | | | |
| EMO header shape | 4 slots, taxonomy emotion, never bare AWE/INDIGNATION | **12/12, 0 bare** ✅ | **12/12, 0** ✅ | **12/12, 0** ✅ | **12/12, 0** ✅ |
| Lyrics opener | `[Theme: …]` then `[SONG FORM: …]` | ✅ L1 then L2 | ✅ | ✅ | ✅ |
| SFX | ≥ 1 standalone cue **matching the validator pattern** | **2** ✅ | **2** ✅ | **2** ✅ | **2** ✅ |
| `sung_numerals_spelled_out` | true | **0 digits in any sung line** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| No real-artist names | any Suno-bound field | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |

**Sample EMO header, verbatim:** `[Strophe 4 - EMO:Detachment - Solo Soprano - THE HINGE, upright and viola out for two bars, thinnest point]` — four slots; `Detachment` is `EMOTION_TAXONOMY → Apathy: Indifference, Unconcern, Detachment`. Emotions used across the pair: **Composure · Ennui · Equanimity · Detachment · Unconcern · Apathy · Indifference · Impatience · Vigilance · Watchfulness · Listlessness** — all in the taxonomy, none bare, none warm.

### 8.2 TARGET BANDS (outside → FLAG, never an auto-fail)

| Gate | Band | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `music_prompt_chars_target` | 870–960 | **959** ✅ | **938** ✅ | **909** ✅ | **920** ✅ |
| `music_prompt_hug_ceiling` | ≥ 985 → FLAG | 959 — no flag ✅ | 938 ✅ | 909 ✅ | 920 ✅ |
| `sung_lines_target` | 78–110 | **88** ✅ | **88** ✅ | **88** ✅ | **88** ✅ |
| `sung_lines_floor_hug` | ≤ 72 → FLAG | 88 — no flag ✅ | 88 ✅ | 88 ✅ | 88 ✅ |
| `suno_lyrics_field_target` | ≤ 4800 | **4,653** ✅ | **4,714** ✅ | **4,782** ✅ | **4,692** ✅ |
| `max_sung_numeric_facts` | 1 · **only P04 spends it** | **1** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |
| Suno exclude length | 400–900 target, ≤1000 hard | **659** ✅ | **657** ✅ | **641** ✅ | **659** ✅ |

⚠️ **V3 is the binding constraint in this pair at 4,782** — eighteen characters under the `suno_lyrics_field_target` and two hundred and eighteen under the render cliff. It went 3 chars **over** the target on the first assembly and one header cue was shortened to bring it back. **Reported because a target crossed and recovered is still an event.**

### 8.3 ⭐ RETURN FLOORS (L21) — re-measured after every edit

| Gate | Floor / ceiling | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `rhyme_window` | ±4 lines (THE definition) | as shipped | | | |
| `rhyme_return_floor` | ≥ 0.30 | **0.614** ✅ | **0.659** ✅ | **0.670** ✅ | **0.773** ✅ |
| `line_return_floor` | ≥ 0.20 (choruses COUNT) | **0.227** ✅ | **0.227** ✅ | **0.227** ✅ | **0.273** ✅ |
| ⭐ `mean_words_per_line_ceiling` | **≤ 7.5** | **6.864** ✅ | **7.159** ✅ | **7.364** ✅ | **7.273** ✅ |
| `alliteration_per_100w_floor` | ≥ 11.0 | **13.079** ✅ | **13.492** ✅ | **13.272** ✅ | **11.875** ✅ |
| `unique_line_ratio_floor` | ≥ 0.45, FLAG only, refrain EXEMPT | refrains exempt; no flag ✅ | ✅ | ✅ | ✅ |
| `chorus_repetition_requires_no_justification` | true | **Byte-identical on all four returns. No justification filed and none owed.** | | | |

**⚠️ THE WORDLESS-RETURN CAVEAT — answered, and now load-bearing.** The handoff warns a vocable or hum can satisfy the return floor by itself. **This pair's return vehicle is entirely lexical** — an end-rhyme chain on one vowel plus a four-line sung refrain. There is no hum, no vocable and no non-lexical hook anywhere in it, so the lexical-only companion measurement **is** the measurement above. ⭐ **This is also why the Body Noise Mandate was refused** (§7): adding three hums would have destroyed the one honest measurement this pair can make. Re-measured with all sixteen refrain lines deleted (72 lines each), the rhyme return survives at **0.528 / 0.583 / 0.597 / 0.708** — the vowel spine is genuinely distributed across the strophes and is not one repeated element flattering the instrument. Line return does not survive (**0.056 / 0.056 / 0.056 / 0.111**) and should not: **the refrain is the line-return device, by design, and it is four sung lines of English, not a syllable.**

⚠️ **Two numbers reported against my own work rather than for it.** **(a)** Refrain-stripped alliteration falls to **10.282** in V1 and **10.769** in V4 — below the 11.0 floor. The floor is measured on the whole song and both pass comfortably there (13.079, 11.875), but it means **V1 and V4 carry a disproportionate share of their consonant return in the refrain**, which is where step 10's single repair put it. **(b)** V1's `rhyme_return` fell **0.659 → 0.614** as a direct cost of the Strophe 5 rewrite in §5: replacing a vowel-fed tautology with a physical chain spends rhyme to buy an image. **I judged that trade worth making at 2× the floor and I am recording the price rather than quietly banking the improvement.**

### 8.4 DISTINCTIVENESS (coordinator-side; written to pass)

`step06_max_pair_similarity 0.50` · `step09_max_pair_similarity 0.62` · `portfolio_max_lyric_similarity 0.42` · `portfolio_max_prompt_similarity 0.58` · `portfolio_max_ngram_jaccard 0.18`.

**No cross-pair collision is possible on the load-bearing axes, and each is checkable rather than asserted:** P04 is the **only** pair in 7/8, the **only** soprano lead, the **only** dry close-miked pair (the run's whole allowance), the **only** procedural diction, and the **only** pair whose return vehicle is a single-vowel end-rhyme chain. Its device — *the form being filled* — appears in no other pair.

⚠️ **Stated in advance so it is not read as a defect:** the four variations resemble each other — one band, one room, one singer, four positions on one corridor. That is by construction, and the three named validators measure **BETWEEN pairs, not within one.** Per the handoff, **do not trust a similarity number from them without printing what was EXTRACTED first.** **This artifact's extraction is assertable: four `## 2. LYRICS` fenced blocks, 88 sung lines each, 4,653–4,782 chars each; four MUSIC PROMPT blocks, 909–959 chars each.**

### 8.5 THE MECHANICAL VERIFICATIONS RUN ON THIS FILE

```
sung-line digit scan .............. 0 digits in 352 sung lines across 4 variations
number-words four|point|five ...... V1 Strophe 4 only, 1 line, mid-line, not at rhyme position
lyric twin occurrences ............ V1:1  V2:1  V3:1  V4:1 — each the FINAL sung line of Strophe 5
refrain byte-identity ............. V1:4/4  V2:4/4  V3:4/4  V4:4/4 identical returns
EMO header slot count ............. 48 headers, 48 with exactly four slots, 0 bare labels
house_lexicon (13 phrases) ........ 0 hits in 12 Suno-bound fields
amplitude vocabulary (6 tokens) ... 0 hits
Glitch-Baroque / HyperRaaga ....... 0 hits
real-artist blocklist ............. 0 hits
retro-trap tokens in style fields . 0 (present only in the exclude fields, where negatives belong)
```

---

## 9 — ⛔ THE BINDING DECISIONS, RE-CHECKED PER VARIATION AFTER THE EDITS

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| **D1** singer never arrives | ✅ she never doubts the threshold | ✅ | ✅ she never wonders who reads the box | ✅ |
| **D2** cold, mid-task, no reverence | ✅ | ✅ | ✅ | ✅ |
| **D3** two lines named by interval | ✅ B♮4 / B♭4, and now the **held gap** is named too | ✅ | ✅ | ✅ |
| **D4** nobody finds out | ✅ | ✅ | ✅ | ✅ |
| **D5** present tense, listener as defendant | ✅ | ✅ | ✅ | ✅ |
| ⭐ **D6** a skill, not a sin — **per variation** | the song **admires her speed**; the threshold is not hers and she applies it correctly | ⭐ **strengthened:** *"The letter I fixed is in code"* — her care is real, skilled, and invisible. The **procedure** gives the village the load, not her | **she believes the caller**, types every word, reads it back, gets it right. The failure is structural, never personal | the one who is late is never criticised, named or resented. *"Nobody has to."* |
| **D7** no enumerations · reorder test | ✅ **re-run on the rewritten S5** — the box must open before she fills it and must ask before she supplies | ✅ **re-run on S3 and S6** — the dot cannot leave the road before the map folds | ✅ strictest in the pair; **re-run on S5** — she cannot read back what she has not written | ✅ S6 is a four-step interlock; the lanyard cannot come off before the tap-out |
| **D8** ends on a completed physical act | *I turn to the door and I go* | *I push. It gives. And I go* | *I turn. I have somewhere to go* | *I do not look back. I go* |
| **D10** unspent, not sepia | ✅ no tape, no hiss, no dead format anywhere | ✅ | ✅ | ✅ |
| **D11** one room, not a booth · **the gap must be audible** | ⚠️ **carried to the render audit with a falsifiable question** (§3.2) | ⚠️ same | ⚠️ same | ⚠️ same |

**⛔ HUMAN-SUBJECT — FINAL SCAN.** Every person in all four songs is **invented**. The analyst has no name. The caller has no name, age, location or nationality. ⛔ **No real place is sung** — no Alaskan, Japanese or any other real place-name appears in any lyric or render field; V2 dramatises the **naming procedure** instead of any name. **Not a disaster song:** nothing collapses, nobody is hurt, no siren, no rubble — **the largest physical event in the pair is `The water went out of the bowl.`** ⛔ **The two real deaths in today's feed are not raw material and are not alluded to.** Messier and Tempel are not named, quoted, or given interiority. ⚠️ `scripts/check_human_subjects.py` **not consulted** — handoff §4 records that it returns `HOLD_FOR_HUMAN` on 100 % of correctly-written artifacts in this checkout (spaCy absent). **Judged on content, per handoff §5; its output is not reported as a finding in either direction.**

**⛔ D9 APPROPRIATION GATE — does not apply to P04, stated positively.** The gate is capped at two pairs and they are named: **P01 and P02.** This pair is written from broadly shared chamber-prog vocabulary, claims no living tradition, and **draws none of Flair #11 — not the word, not the function.** Verified: the token appears in **none of the 12 Suno-bound fields** and in no lyric line. **No Lineage & Credit block is owed by P04**, and QA R3's outstanding "links, not just names" repair belongs to P01 and P02.

---

## 10 — SELF-CRITIQUE OF THIS ENHANCEMENT PASS

**What I actually changed, in one line:** four prompts rewritten genre-first and honestly measured, one false arithmetic claim replaced with a true and better one, four lyric twins moved to the same structural moment, eleven thin lines restaged around objects, two short SFX cues per song where there had been one unparseable one, and a ranking overturned.

**The strongest thing I found** is that the pair's mechanism is better than the pair's description of it. *Seven against eight* was being sold as a rare coincidence when it is a machine that ticks fourteen times, and the actual event — **the only place in the song where the drift stops, and the one time the held distance is nothing** — is both truer and more audible. **A device that survives being described accurately is a real device.**

**The weakest thing here, and I would rather say it than have QA find it:** ⚠️ **I cannot verify the bar plan against a score, because there is no score.** The bar counts in `[SONG FORM:]` are arithmetically consistent and produce a 3:35–3:49 song at the four tempi, but they are a *chart written in prose for a generator that does not read charts.* If Suno squares the 7/8, the drift never happens, all four refrains hold the same gap, and the entire argument evaporates — and **no text gate in this pipeline can see that.** The lyric twin is the only part of the crossing that survives that failure, which is exactly why it was worth moving four times.

**The second weakest thing:** ⚠️ **the vowel spine is still doing a great deal of work.** I repaired the two coordinates the pair named and one it did not, but ninety line-ends on /oʊ/ exerts constant pressure toward the same twenty words, and I have reduced that pressure rather than removed it. **V1 Strophe 2 and V4 Strophe 2 are now the thinnest stretches in the pair** and I am naming them rather than repairing them, because both are load-bearing repetition (*"I read the new figure and hold"* twice; *"The hall gives it back to me slow"* twice) and cutting into them would cost more return than it bought in image. **A third pass should start there.**

**And the thing that cannot be settled from here.** ⛔ **The Maximalist did not withdraw and I have not made him withdraw.** Whether the third refrain sounds different from the fourth is a render fact, not a text fact. What has changed is that the question is now **falsifiable in one listen** (§3.2) instead of being a matter of impression. If the answer comes back *"the distance is the same in all four,"* the pair's argument was smoothed away and he was right the first time — and we will know it, which we would not have before.

---

### VARIATION 1

## 1. MUSIC PROMPT

```text
Odd-meter chamber prog, 96 BPM, 7/8 in D Dorian, cut live in one hallway, five players, one take. Solo female soprano, high and plain, a straight unwavering tone, consonants early, landing exactly where she says she will. Bass clarinet carries the seven, viola bowed at almost no pressure, felted upright on a gentle piano refrain, fretless bass, rim and closed hat. Signature device: two small struck bells, one sound twice, B natural on every downbeat and B flat every eight eighth-notes, ignoring the band, arriving an eighth later each 7/8 bar. The walking bell opens alone. A four-bar 4/4 refrain returns four times, note for note the same, and a 4/4 bar is eight eighth-notes, so the walking stops inside a refrain and the bells hold the gap they came in on. At the third that gap is nothing: one strike, a semitone wide. Close-miked a hand out, one level, highest note in the first third. Instruments arrive and leave; the last section is the thinnest.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, found-recording framing, reverb tail, room bloom, long decay, riser, drop, build into final chorus, swelling strings, orchestral swell, cymbal wash, crash cymbal, vibrato, portamento, scooped entry, stacked vocal harmony, backing vocals, doubled lead vocal, breathy whisper, autotune gloss, rubato, ritardando, fade-out ending, key change, tempo change, quantised straight 4/4, straightened odd meter, widened interval between the two bells, bells tuned to the same pitch, bells locked to the same grid, office foley, keyboard clicks, phone ringtone, sirens, glitch, male vocals, choir, spoken word, sad piano ballad
```

## 2. LYRICS

```text
[Theme: a duty analyst standing in a corridor between two doors, filing the day's events, and filing her own measurement, which came out under the bar by a rounding]
[SONG FORM: through-composed 7/8 - six unrepeating strophes and one 4-bar refrain in 4/4 returning four times, byte-identical; bar plan 6/12/12-R-12-R-12/10-R-10-R-6. A 4/4 bar is eight eighth-notes, so the walking bell stops drifting inside each refrain and holds the gap it entered on: six eighths at the first, two at the second, nothing at the third, two at the fourth. No bridge, no key change, tempo constant, D Dorian]

[Intro - EMO:Composure - Instrumental bass clarinet and felted upright - six bars, the walking bell alone in bar one]
*two bells, seven against eight*

[Strophe 1 - EMO:Composure - Solo Soprano high and plain - twelve bars, one line to a bar, consonants early]
Nowhere to sit. Nowhere to go.
A wall at my back. A row.
No desk in here, and so
the screen is a thing I hold.
The height of the day is my own.
The first of the day opens cold.
I read what the instrument shows.
I read it again. It still shows.
It hands me a figure to hold.
I hold it the way I am told.
I carry it into the code.
I put it in and I close.

[Strophe 2 - EMO:Ennui - Solo Soprano - bass clarinet on the seven, viola in contrary motion, level unchanged]
A number comes in on its own.
A number is never alone.
There is a correction to load.
The correction is not mine. It's code.
I run it. The figure goes low.
Corrections do not make things grow.
Down is the way that they go.
Down is the way I was told.
I read the new figure and hold.
I read the new figure and hold.
Nothing has gone wrong. It goes
the way that a figure goes.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, the two bells six eighths apart and holding]
The line is the line. I am told.
The line is a thing I'm told.
Under the line lies a row.
Over the line lies a row.

[Strophe 3 - EMO:Equanimity - Solo Soprano - 7/8 returns, felted upright enters, no level change]
Nothing is one number. I know.
A figure comes wearing a shadow.
The shadow goes high and goes low.
The shadow lies over and below.
The rule does not look at the shadow.
The rule takes the figure. And so
the figure is all that I hold.
The shadow is not what I hold.
I do not put shadows in rows.
I put in the figure. It shows.
The figure is under. It goes
into the box where it goes.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, identical, the bells now two eighths apart]
The line is the line. I am told.
The line is a thing I'm told.
Under the line lies a row.
Over the line lies a row.

[Strophe 4 - EMO:Detachment - Solo Soprano - THE HINGE, upright and viola out for two bars, thinnest point]
The one after that one I own.
I made it. I made it alone.
I built it myself from the code.
I built it the way I was told.
It came to a figure, and so
I carried the figure below.
Four point five is the bar. I know
what mine is. Mine is below.
Under by less than a hair, and so
the figure goes where figures go.
I put the figure in. Not slow.
I have never put one in slow.

[Strophe 5 - EMO:Unconcern - Solo Soprano - ten bars, viola returns, the bells one eighth apart at the last line]
The box for it opens, and so
I put the whole of it in a row.
It does not ask what I hold.
It asks for a figure. I know
which figure it wants. And it goes
in, and the box takes the row.
The note goes out on its own.
The no does not go. It is code.
I sign it. I send it. I close.
A note. A no. A note.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical; the two bells strike as one, a semitone apart, and neither moves]
*both bells land as one*
The line is the line. I am told.
The line is a thing I'm told.
Under the line lies a row.
Over the line lies a row.

[Strophe 6 - EMO:Apathy - Solo Soprano - the bells still together on the first bar, then walking again]
The next one comes in from below.
The next one comes in from below.
I run the correction. It goes.
It goes where the last of them goes.
There is nothing to flag. There is no
reason to slow, and I know
the shape of the rest of the row.
I am good at this. And I close.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical, the bells two eighths apart again; nothing happens]
The line is the line. I am told.
The line is a thing I'm told.
Under the line lies a row.
Over the line lies a row.

[Outro - EMO:Indifference - Solo Soprano - fewest players in the song, the walking bell last]
The last of them goes in a row.
The last of them goes below.
Nothing to note. Nothing owed.
The day is a thing I have closed.
I dock the screen and I go.
I turn to the door and I go.
```

## 3. TITLE

The Second Decimal

## 4. PRODUCTION SIDECAR

**Disc_Channel** *(outside the lyrics field, per the run's single harness decision)*

```text
[Disc_Rhythm: rim_and_closed_hat_only | 7_8_eighth_note_grid_96_BPM | uncompressed_transient_snap | Center_Mono]
[Disc_Vocal: close_mic_female_soprano | straight_tone_no_vibrato | dry_intimate_no_reverb | consonant_early_never_late | Center_Front]
[Disc_Sub: fretless_bass_fingered_soft | bell_fundamental_B_natural_4 | Mono_Sub_Lock]
[Disc_Pad: no_pad_no_wash_no_sustain_bed | bass_clarinet_carries_the_seven | Center_Mono]
[Disc_Texture: felted_upright_gentle_piano_refrain | viola_bowed_near_zero_pressure | two_struck_bells_seven_against_eight | Stereo_Width_Narrow]
```

**Vocal fingerprint.** Solo female soprano, high and plain, straight unwavering tone, almost choirboy. Consonants placed early; **she never lands late**, and she takes no audible breath at a phrase end — a singer who is never late is a singer who is never moved. No doubling, no stacked harmony, no backing vocal anywhere in this pair. **Highest note sits in the first third** and is structural: a late top note is a climax whether or not one is intended.

**Production dramaturgy.** Every unusual sound has a job. The **bass clarinet** carries the seven and is the only tired thing in the room. The **viola** moves in contrary motion so it never doubles the voice. The **felted upright** supplies the gentle piano refrain and is the only instrument that enters mid-song. The **two bells** are the argument: same sample, same envelope, same level, a semitone apart, one on the downbeat and one free-running at eight eighths, **and the player of the second one is instructed to ignore the band** — otherwise it gets "fixed" by ear within four bars and the piece has no event in it. **Nothing is automated at the coincidence.**

**Style-axis lock.** Genre `odd-meter chamber prog tracked live` · key **D Dorian** (B♮ makes the mode, B♭ would make it ordinary — the interval is the whole joke) · **7/8**, one 4-bar 4/4 refrain ×4 · **96 BPM**, eighth-note pulse constant · **single dynamic**, growth by instruments arriving, decay by instruments leaving · **dry, close-miked, one room, no reverb tail** — the run's one dry pair · **unnecessary element:** the bell that never fires.

**Bar / phase chart.** `6 · 12 · 12 · [R1] · 12 · [R2] · 12 · 10 · [R3] · 10 · [R4] · 6` = 80 bars of 7/8 + 16 of 4/4 = 688 eighths ≈ **3:35 at 96 BPM**. Held gap at the four refrains: **6 · 2 · 0 · 2** eighths.

---

### VARIATION 2

## 1. MUSIC PROMPT

```text
Odd-meter chamber prog, 94 BPM, 7/8 in D Dorian, tracked live to tape in a bare corridor, one take, the players cueing each other by ear. Solo female soprano, high and plain, a straight unwavering tone, unhurried, entering flat on the note with no scoop. Bass clarinet on the seven, viola in contrary motion so it never doubles the voice, felted upright under a gentle piano refrain, fretless bass, rim and closed hat. Signature device: two identical struck bells, B natural on each downbeat and the same bell on B flat every eight eighth-notes, never reset, sliding an eighth later per 7/8 bar. One bell opens by itself. A four-bar 4/4 refrain arrives four times, note for note the same, and a 4/4 bar is eight eighth-notes, so the drift halts inside a refrain and the bells hold the gap they entered on. On the third that gap is nothing: one strike, a semitone wide. Close-miked a hand out, dry, one steady level. It stops mid-corridor.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, found-recording framing, reverb tail, room bloom, long decay, riser, drop, build into final chorus, swelling strings, orchestral swell, cymbal wash, crash cymbal, vibrato, portamento, scooped entry, stacked vocal harmony, backing vocals, doubled lead vocal, breathy whisper, autotune gloss, rubato, ritardando, fade-out ending, key change, tempo change, quantised straight 4/4, straightened odd meter, widened interval between the two bells, bells tuned to the same pitch, bells locked to the same grid, alarm sample, office foley, keyboard clicks, sirens, glitch, male vocals, choir, spoken word, story-song swell
```

## 2. LYRICS

```text
[Theme: a duty analyst in a corridor, filing the one event that cleared the bar, which needed her less than any of the others did]
[SONG FORM: through-composed 7/8 - six unrepeating strophes and one 4-bar refrain in 4/4 returning four times, byte-identical; bar plan 6/12/12-R-12-R-12/10-R-10-R-6. A 4/4 bar is eight eighth-notes, so the walking bell stops drifting inside each refrain and holds the gap it entered on: six eighths at the first, two at the second, nothing at the third, two at the fourth. No bridge, no key change, tempo constant, D Dorian]

[Intro - EMO:Composure - Instrumental one struck bell then bass clarinet - six bars, the fired bell alone in bar one]
*one bell, then the other, later*

[Strophe 1 - EMO:Composure - Solo Soprano high and plain - twelve bars, one line to a bar, dry and level]
This one comes in with a tone.
The rest come in alone.
A bell is a thing I have known
since the day I was shown the code.
The bell says nothing it knows.
It means someone else has been told.
The telling is not mine to hold.
The telling is done. It is code.
I look at the end of the row.
The row has closed on its own.
It needed no thumb. It goes.
It goes where the cleared ones go.

[Strophe 2 - EMO:Impatience - Solo Soprano - felted upright enters, viola holds one note, no level change]
It flags on its own, and so
a notice goes where notices go.
The flag pulls a map from the code.
The map puts a dot on a road.
I did not draw it. I know
the hand that drew it. It goes
the way that the drawing goes.
Nothing was wanted, and so
I watched it. I added no note.
There was nothing to add to the note.
The note went out on its own.
The note is a thing I have known.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, the two bells six eighths apart and holding]
One of them clears and is told.
The rest of them are not told.
One of them goes as a note.
The rest of them go as I wrote.

[Strophe 3 - EMO:Equanimity - Solo Soprano - 7/8 returns, fretless bass enters, level unchanged]
The rest of them want me. This one
wanted nothing at all. It is done.
I read it. There's nothing to hold.
I read it. There's nothing to hold.
Nothing to check. Nothing told
to me that I did not know.
I pull the two up in a row.
The cleared one and the one below
sit in the very same row.
I take my thumb off that one.
I put it back down on the code.
The code takes it. So does the row.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, identical, the bells now two eighths apart]
One of them clears and is told.
The rest of them are not told.
One of them goes as a note.
The rest of them go as I wrote.

[Strophe 4 - EMO:Detachment - Solo Soprano - THE HINGE, upright and viola out for two bars, thinnest point]
A thing that clears has to be known.
The name that it gets is not its own.
It borrows the nearest. It goes
by a village that sits on a road.
The village felt nothing. And so
the village is given the load.
It carries a name down the road.
It did not ask for the load.
I check that the spelling is close.
I fix a letter. I close.
That is the whole of my note.
A letter. And then it goes.

[Strophe 5 - EMO:Unconcern - Solo Soprano - ten bars, viola returns, the bells one eighth apart at the last line]
I go back down to the row.
I go back down to the row.
The rest of them sit in a row.
The note does not alter the code.
The no does not alter the code.
Both of them true. Both of them close.
Both of them go where they go.
I stamp it. I send it. I close.
The next of them opens. I know.
A no, and a note, and a no.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical; the two bells strike as one, a semitone apart, and neither moves]
*the two bells land together*
One of them clears and is told.
The rest of them are not told.
One of them goes as a note.
The rest of them go as I wrote.

[Strophe 6 - EMO:Apathy - Solo Soprano - the bells still together on the first bar, then walking again]
The one with the name has gone.
I scroll, and it sits in the row.
It sits at the same width. And so
it reads like the rest of the row.
The map of it folds and it goes.
The dot comes away from the road.
The letter I fixed is in code.
It needed one thumb. And it's closed.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical, the bells two eighths apart again; nothing happens]
One of them clears and is told.
The rest of them are not told.
One of them goes as a note.
The rest of them go as I wrote.

[Outro - EMO:Indifference - Solo Soprano - fewest players in the song, the walking bell last]
The tone does not come back. And so
the hall is the hall that I know.
I hang the screen up. And I go.
Nothing was mine in that one.
I turn to the near door. I go.
I push. It gives. And I go.
```

## 3. TITLE

The Map

## 4. PRODUCTION SIDECAR

**Disc_Channel** *(outside the lyrics field, per the run's single harness decision)*

```text
[Disc_Rhythm: rim_and_closed_hat_only | 7_8_eighth_note_grid_94_BPM | uncompressed_transient_snap | Center_Mono]
[Disc_Vocal: close_mic_female_soprano | straight_tone_no_vibrato | flat_entry_no_scoop | dry_intimate_no_reverb | Center_Front]
[Disc_Sub: fretless_bass_fingered_soft | bell_fundamental_B_natural_4 | Mono_Sub_Lock]
[Disc_Pad: no_pad_no_wash_no_sustain_bed | viola_holding_one_note | Center_Mono]
[Disc_Texture: felted_upright_gentle_piano_refrain | bass_clarinet_on_the_seven | two_struck_bells_seven_against_eight | Stereo_Width_Narrow]
```

**Vocal fingerprint.** As V1, one notch more impatient: unhurried, entering flat on the note with no scoop and no swell. The faint irritation is in the **timing**, never in the volume — she is slightly ahead of every entry, as someone is when they already know what the form will ask.

**Production dramaturgy.** The **flag → map → dot → name** chain is dramatised entirely by her verbs and never by an arrangement move; the machine is loud in the lyric and silent in the mix. ⛔ **The arrangement must not swell at *"The village felt nothing"*** — that is the line a generator hears as the emotional peak, and the song becomes a story-song with a moral. The hinge's last two lines are deliberately administrative (*"That is the whole of my note. / A letter. And then it goes."*), so **any swell lands on a spelling correction and dissipates.** The exclude field carries `story-song swell` for exactly this.

**Style-axis lock.** Genre `odd-meter chamber prog tracked live to tape` · key **D Dorian** · **7/8**, one 4-bar 4/4 refrain ×4 · **94 BPM** · single dynamic, entries and exits rather than ramps · dry, close-miked, one corridor · **unnecessary element:** the bell that never fires. **It stops mid-corridor** — no resolution, no tail.

**Bar / phase chart.** `6 · 12 · 12 · [R1] · 12 · [R2] · 12 · 10 · [R3] · 10 · [R4] · 6` = 688 eighths ≈ **3:40 at 94 BPM**. Held gap at the four refrains: **6 · 2 · 0 · 2** eighths.

---

### VARIATION 3

## 1. MUSIC PROMPT

```text
Odd-meter chamber prog, 92 BPM, 7/8 in D Dorian, played live in one room, the five players a metre apart. Solo female soprano, high and plain, a straight unwavering tone, speaking-close, every consonant on time. Bass clarinet states a phrase, leaves a gap the length of an answer, states it again; viola lets those gaps stand. Felted upright on a gentle piano refrain, fretless bass fingered soft, rim and closed hat, one brushed snare. Signature device: two identical small bells, B natural on every downbeat and B flat every eight eighth-notes, never resetting, landing later each 7/8 bar. The B flat has the first bar alone. A four-bar 4/4 refrain returns four times, identical, and a 4/4 bar is eight eighth-notes, so the drift stops inside a refrain and the bells hold the gap they arrived with. On the third the gap is nothing: one event, a semitone wide. Close mics a hand from each source, dry, level.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, found-recording framing, reverb tail, room bloom, long decay, riser, drop, build into final chorus, swelling strings, orchestral swell, cymbal wash, crash cymbal, vibrato, portamento, scooped entry, stacked vocal harmony, backing vocals, doubled lead vocal, breathy whisper, autotune gloss, rubato, ritardando, tender ballad turn, fade-out ending, key change, tempo change, quantised straight 4/4, straightened odd meter, widened interval between the two bells, bells tuned to the same pitch, phone ringtone, dial tone, office foley, keyboard clicks, sirens, glitch, male vocals, choir, spoken word
```

## 2. LYRICS

```text
[Theme: a duty analyst takes a call standing in a corridor; the caller felt the floor move, and there is a box on the form that takes anything]
[SONG FORM: through-composed 7/8 - six unrepeating strophes and one 4-bar refrain in 4/4 returning four times, byte-identical; bar plan 6/12/12-R-12-R-12/10-R-10-R-6. A 4/4 bar is eight eighth-notes, so the walking bell stops drifting inside each refrain and holds the gap it entered on: six eighths at the first, two at the second, nothing at the third, two at the fourth. No bridge, no key change, tempo constant, D Dorian]

[Intro - EMO:Vigilance - Instrumental bass clarinet then a gap the length of an answer - six bars, the un-fired bell alone in bar one]
*a bell, and the same bell, late*

[Strophe 1 - EMO:Composure - Solo Soprano high and plain - twelve bars, speaking-close, dry]
The phone in the hall is a phone.
You take it up standing, alone.
Nowhere to put the screen, so
I hold both, and neither one goes.
A voice comes in. It is low.
A voice that has come a long road:
a number, then hold, then a code,
then a person, and then me. And so
she starts at the start. I know
the sound of a start that is old.
She has told it before. And so
she tells it to me, and I hold.

[Strophe 2 - EMO:Watchfulness - Solo Soprano - felted upright enters, viola silent in the gaps]
Where were you standing? I wrote.
She tells me. I take down the note.
What time was it? Say it out slow.
She says it out slow. It is low
on the hour. It goes in the code.
The hour is easy. It's code.
What did it feel like? And so
I come to the part where I go
quiet, and let her go slow.
She says that the floor came up slow.
The water went out of the bowl.
She stood in her hall. And I wrote.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, the two bells six eighths apart and holding]
A code for the thing that is known.
A box for the rest. And I wrote
the rest of it out on my own.
I put in the whole of the note.

[Strophe 3 - EMO:Equanimity - Solo Soprano - 7/8 returns, fretless bass enters, level unchanged]
And then there's a thing she knows.
She says it, and then the line slows.
She told it to somebody. No
one else in the room. And no
one said that she hadn't. And so
they let it alone. And it goes
the way that a thing like that goes.
She stood where she'd stood. And she knows.
She stood where she'd stood. And she knows.
She is not asking me. No.
She is telling me. I have been told
the difference. It is not a code.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, identical, the bells now two eighths apart]
A code for the thing that is known.
A box for the rest. And I wrote
the rest of it out on my own.
I put in the whole of the note.

[Strophe 4 - EMO:Detachment - Solo Soprano - THE HINGE, upright and viola out for two bars, thinnest point]
I believe her. And so I wrote.
There's a box for a note.
The box takes all of it. So
I put all of it in. It goes
in whole. It stays as it goes.
I read it back to her slow.
I read it back to her slow.
I got it right. And she knows
that I got it right. She knows
the thing she came with is told.
She says thank you. And so
I say what you say. And I close.

[Strophe 5 - EMO:Unconcern - Solo Soprano - ten bars, viola returns, the bells one eighth apart at the last line]
The last of the boxes is code.
There's no code for her road.
There is one that says "no
event," and that one is the code.
I read the two lines that I wrote.
Both of them true. Both of them close.
Neither one waits. Neither slows.
Neither one alters the code.
I put my hand on the key. And so
I type no. I say yes. I type no.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical; the two bells strike as one, a semitone apart, and neither moves]
*the bells strike as one event*
A code for the thing that is known.
A box for the rest. And I wrote
the rest of it out on my own.
I put in the whole of the note.

[Strophe 6 - EMO:Apathy - Solo Soprano - the bells still together on the first bar, then walking again]
The call is a row in a row.
The row goes where the rows go.
The box goes in with the row.
The box is a part of the row.
The next of them comes. And I know
the shape of the next. And I know
the shape of the shift. I hold
the phone and the screen. I go.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical, the bells two eighths apart again; nothing happens]
A code for the thing that is known.
A box for the rest. And I wrote
the rest of it out on my own.
I put in the whole of the note.

[Outro - EMO:Indifference - Solo Soprano - fewest players in the song, the walking bell last]
The hall does not change. And so
I take up the next. And it goes
the way that the next ones go.
Her words are in whole. And the code.
The phone goes back where it goes.
I turn. I have somewhere to go.
```

## 3. TITLE

The Free Text Box

## 4. PRODUCTION SIDECAR

**Disc_Channel** *(outside the lyrics field, per the run's single harness decision)*

```text
[Disc_Rhythm: rim_and_closed_hat_one_brushed_snare | 7_8_eighth_note_grid_92_BPM | uncompressed_transient_snap | Center_Mono]
[Disc_Vocal: close_mic_female_soprano | straight_tone_no_vibrato | speaking_close_intake_delivery | dry_intimate_no_reverb | Center_Front]
[Disc_Sub: fretless_bass_fingered_soft | bell_fundamental_B_natural_4 | Mono_Sub_Lock]
[Disc_Pad: no_pad_no_wash_no_sustain_bed | bass_clarinet_phrase_then_a_gap | Center_Mono]
[Disc_Texture: felted_upright_gentle_piano_refrain | viola_leaves_the_gaps_standing | two_struck_bells_seven_against_eight | Stereo_Width_Narrow]
```

**Vocal fingerprint.** As V1, at speaking distance: the intake register, every consonant on time, the phrasing of someone typing while listening. ⛔ **She is never given an adjective anywhere in the song.** The generator will hear *"I believe her"* and reach for warmth, vibrato and rubato — the lines immediately after it are pure procedure, and the next strophe is a disposition code. `tender ballad turn` is in the exclude field for this reason.

**Production dramaturgy.** The **bass clarinet states a phrase, leaves a gap the length of an answer, and states it again** — the form's own call-and-wait, made audible before a word is sung; the viola is instructed to let those gaps stand rather than fill them. The **read-back** is the song's only natural peak and needs nothing built around it. ⭐ **The crossing is in her body here, not only in the bells:** her mouth says yes to the caller while her hand types no into the disposition field, same second, neither altering the other — and it is the last thing sung before the two bells land as one.

**Style-axis lock.** Genre `odd-meter chamber prog played live in one room` · key **D Dorian** · **7/8**, one 4-bar 4/4 refrain ×4 · **92 BPM** · single dynamic · dry, close mics a hand from each source · **unnecessary element:** the bell that never fires. ⛔ **Not a disaster song** — the largest physical event is water going out of a bowl.

**Bar / phase chart.** `6 · 12 · 12 · [R1] · 12 · [R2] · 12 · 10 · [R3] · 10 · [R4] · 6` = 688 eighths ≈ **3:44 at 92 BPM**. Held gap at the four refrains: **6 · 2 · 0 · 2** eighths.

---

### VARIATION 4

## 1. MUSIC PROMPT

```text
Odd-meter chamber prog, 90 BPM, 7/8 in D Dorian, tracked live in a hard empty hallway, single take, bleed between players welcome. Solo female soprano, high and plain, a straight unwavering tone, tired and exact, never landing late. Bass clarinet on the seven, viola thin and dry, felted upright mid-keyboard under a gentle piano refrain, fretless bass, rim and closed hat with brushes. Signature device: two struck bells, one sound used twice, B natural on every downbeat and B flat every eight eighth-notes, never reset, sliding an eighth later per 7/8 bar. The B flat has the track to itself for one bar. A four-bar 4/4 refrain comes back four times without a change, and a 4/4 bar is eight eighth-notes, so the walking stops inside a refrain and the bells hold the gap they came in on. At the third that gap is nothing: one strike, a semitone wide. Close-miked a hand out, one level. Instruments leave one at a time.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, found-recording framing, reverb tail, room bloom, long decay, ambient pad bed, lonely hallway atmosphere, riser, drop, build into final chorus, swelling strings, orchestral swell, cymbal wash, crash cymbal, vibrato, portamento, scooped entry, stacked vocal harmony, backing vocals, doubled lead vocal, breathy whisper, autotune gloss, rubato, ritardando, fade-out ending, key change, tempo change, quantised straight 4/4, straightened odd meter, widened interval between the two bells, bells tuned to the same pitch, door slam sample, footsteps foley, keyboard clicks, sirens, glitch, male vocals, choir, spoken word
```

## 2. LYRICS

```text
[Theme: the end of a shift in a corridor between two doors; the handover does not arrive and the shift closes anyway]
[SONG FORM: through-composed 7/8 - six unrepeating strophes and one 4-bar refrain in 4/4 returning four times, byte-identical; bar plan 6/12/12-R-12-R-12/10-R-10-R-6. A 4/4 bar is eight eighth-notes, so the walking bell stops drifting inside each refrain and holds the gap it entered on: six eighths at the first, two at the second, nothing at the third, two at the fourth. No bridge, no key change, tempo constant, D Dorian]

[Intro - EMO:Ennui - Instrumental the un-fired bell alone then bass clarinet - six bars at 90 BPM in 7/8]
*the late bell first, then the beat*

[Strophe 1 - EMO:Composure - Solo Soprano high and plain - twelve bars, one line to a bar, dry and level]
The last of the day is a row.
The last of the day is a row.
I put it in, and it goes.
It goes where the last ones go.
Nothing is left to be told.
Nothing is left that I hold.
The screen goes flat, and I know
the shift is a thing I can close.
I close it. And nothing is owed.
Nothing is owed. Nothing is old.
I stand in the hall. And the code
is done, and the day is a code.

[Strophe 2 - EMO:Listlessness - Solo Soprano - viola enters thin and dry, no level change]
The hall has no carpet. And so
the hall gives it back when I go.
A cough comes back. And a note.
A word comes back as a note.
The hall is not in the code.
The hall has never been code.
The hall does not go in a row.
The hall is not one of those.
I stand in the middle and hold
the screen at the height that I hold.
The hall gives it back to me slow.
The hall gives it back to me slow.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, the two bells six eighths apart and holding]
The near door is the door I hold.
The far door does not open. I'm told.
I stand still where the hall goes.
I stand still where the hall goes.

[Strophe 3 - EMO:Impatience - Solo Soprano - fretless bass enters, level unchanged]
The one who comes after is slow.
The one who comes after is slow.
I stand at the near door. I hold.
The rule is you wait. I was told.
You wait. You hand it. You go.
You do not put it down. No.
So I do not put it down. No.
I stand. The hall does not go.
She'd come through the far. I know
the sound of that door. It's known.
That sound is one I would own.
Nothing comes through. And I hold.

[Refrain - EMO:Composure - Solo Soprano - 4/4, four bars, identical, the bells now two eighths apart]
The near door is the door I hold.
The far door does not open. I'm told.
I stand still where the hall goes.
I stand still where the hall goes.

[Strophe 4 - EMO:Detachment - Solo Soprano - THE HINGE, upright and viola out for two bars, thinnest point]
The hour goes over. And so
the shift is a thing that is old.
A shift does not wait to be told.
A shift ends because it is told.
Nobody comes. And I know
nobody has to. And so
I write in the book that I go.
I write in the book that I go.
There's a box for the close.
I put in the time. I close.
Nobody signs it but me. So
the whole of the shift is my own.

[Strophe 5 - EMO:Unconcern - Solo Soprano - ten bars, viola returns, the bells one eighth apart at the last line]
I turn to the near door and hold.
I put out my hand. And I hold.
One foot. Then the same foot. So
the hall is the same. And I know
the far door is not going to go.
I look at the near door. And so
I put my hand out. And I hold.
I put my hand out. And I go.
Nothing has changed. And I go.
I go. I do not go. I go.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical; the two bells strike as one, a semitone apart, and neither moves]
*the two bells hit as one*
The near door is the door I hold.
The far door does not open. I'm told.
I stand still where the hall goes.
I stand still where the hall goes.

[Strophe 6 - EMO:Apathy - Solo Soprano - the bells still together on the first bar, instruments leaving one at a time]
I tap out. The screen goes low.
The screen goes down to a glow.
The lanyard comes off. It goes
over the peg. And it goes.
The light is the last. And I know
the light goes off when I go.
I do not turn it off. It's code.
The light is not mine. It is code.

[Refrain - EMO:Composure - Solo Soprano - 4/4, identical, the bells two eighths apart again; nothing happens]
The near door is the door I hold.
The far door does not open. I'm told.
I stand still where the hall goes.
I stand still where the hall goes.

[Outro - EMO:Indifference - Solo Soprano - bass clarinet and the walking bell only]
The far one stays shut. And so
the far door is not in the code.
I turn to the near. And it goes.
I push it. It gives. And I go.
Behind me the light goes low.
I do not look back. I go.
```

## 3. TITLE

The Far Door

## 4. PRODUCTION SIDECAR

**Disc_Channel** *(outside the lyrics field, per the run's single harness decision)*

```text
[Disc_Rhythm: rim_and_closed_hat_with_brushes | 7_8_eighth_note_grid_90_BPM | uncompressed_transient_snap | Center_Mono]
[Disc_Vocal: close_mic_female_soprano | straight_tone_no_vibrato | tired_and_exact_never_late | dry_intimate_no_reverb | Center_Front]
[Disc_Sub: fretless_bass_fingered_soft | bell_fundamental_B_natural_4 | Mono_Sub_Lock]
[Disc_Pad: no_pad_no_wash_no_sustain_bed | viola_thin_and_dry | Center_Mono]
[Disc_Texture: felted_upright_gentle_piano_refrain | bass_clarinet_on_the_seven | two_struck_bells_seven_against_eight | Stereo_Width_Narrow]
```

**Vocal fingerprint.** As V1, at the end of a shift: tired and exact, still never landing late. **The tiredness is in the vocabulary, never in the tone** — she does not sag, slow, or breathe out. The register stays Composure while the strophes go Listlessness → Impatience → Detachment → Apathy around it.

**Production dramaturgy.** ⚠️ **This is the variation most at risk of rendering as atmosphere** — pads, a long tail, a "lonely hallway" mood, which is the medium promoted from substance to subject. Two things fight it and both are in the words: **Strophe 6 is a sequence of switches being operated** (tap out, lanyard off the peg, the light on its sensor) and the outro is a door being pushed. **There is nothing in the final ninety seconds that can be made atmospheric, because every image is an object being worked.** `ambient pad bed` and `lonely hallway atmosphere` are in the exclude field for the same reason.

⭐ **The one place the production idea is also a lyric idea (L22).** The corridor is hard and parallel and gives everything back; the record does not contain that ring, because the capsules sit a hand from each source. In the fiction the same fact is stated once, in her own procedural register — **`The hall is not in the code.`** It appears only in V4, only once, and is never explained.

**Style-axis lock.** Genre `odd-meter chamber prog tracked live in a hard hallway` · key **D Dorian** · **7/8**, one 4-bar 4/4 refrain ×4 · **90 BPM** · single dynamic, instruments leaving one at a time · dry, close-miked, bleed allowed · **unnecessary element:** the bell that never fires.

**Bar / phase chart.** `6 · 12 · 12 · [R1] · 12 · [R2] · 12 · 10 · [R3] · 10 · [R4] · 6` = 688 eighths ≈ **3:49 at 90 BPM**. Held gap at the four refrains: **6 · 2 · 0 · 2** eighths.

---

## LINEAGE & CREDIT

⛔ **P04 owes no Lineage & Credit block, and this is a positive statement rather than an omission.** The D9 Appropriation Gate is capped at two of six pairs and they are named in `05_pair_assignments.md`: **P01** (Ethio-jazz function) and **P02** (gospel close-harmony function). This pair is written from broadly shared chamber-prog and chamber-music vocabulary — odd meter, bass clarinet, viola, felted upright, struck bells — claims no living tradition, and **draws none of Flair #11, neither the word nor the function.** Verified mechanically: the token appears in none of the twelve Suno-bound fields and in no lyric line. QA R3's standing repair (*"lineage blocks name artists impeccably but omit the required links"*) is **P01's and P02's to ship this run.**

**Panel attribution note.** The seats consulted in this pair's chain are model-generated interpretive constructs, each *"after"* a named source figure's published work. **No statement anywhere in this chain is a quotation of, or an endorsement by, any named person**, and no real-artist name appears in any Suno-bound field.

---

*Step 11 complete for P04. Four enhanced packages · four prompts re-measured and rewritten genre-first · one false arithmetic claim replaced · four lyric twins relocated to the same structural moment · eleven thin lines restaged · one ranking overturned · one Hyper-Skeptic objection carried forward with a falsifiable question attached.*
