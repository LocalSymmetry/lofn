# PAIR 03 — STEP 08 · GENERATION (RAW PASS)
## `daily-music-2026-08-21` · **C-22 · THE GUARD'S ROOM** · four variations drafted, measured, and critiqued

> ⛔ **ICB PINNED**, read-only. **GOLDEN-OUTPUT QUARANTINE HONOURED.**
> ⛔ This is the **raw** pass. Everything here is a first-state artifact plus the **measured** reasons it was
> not shippable. Step 09 refines it; step 10 is the contract output. **Nothing in this file is paste-ready.**
> ⭐ **L25 in its writing form: an unmeasured draft is not a draft, it is an opinion.** Every defect below
> carries the number that found it, produced by `scripts/measure_soundcraft.py → sung()` — the canonical
> definition the coordinator re-stats with — not by eye.

---

## 1. THE FOUR VARIATIONS AS DRAFTED

| | **V1** | **V2** | **V3** | **V4** |
|---|---|---|---|---|
| **Working title** | *He Works The Wall* | *The Room Keeps Its Hours* | *I Kept The Empty Room* | *What The Wall Said* |
| **Person / tense** | third, present | third, present habitual, **room as agent** | **first**, past | third + **quotation** |
| **Hook grammar** | anaphoric four-syllable chant | serial temporal cascade, one period | epanaleptic envelope | quotation + refusal to confirm |
| **Section count** | 16 | 15 | 12 | 13 |
| **Opening** | chair scrape + one Rhodes chord | keys on wood + radiator | **cold start**, door bolt + brushes | radiator + rope clip + guitar answer |
| **Where `two years` lands** | log book / a notice, answered by *like a number were a door* | a doorway, answered by *the way a person says a street name* | a corridor, answered by *the way somebody says a bus route* | a corridor, answered by *and another voice says at last* |
| **Flair emphasis** | #6 The Identical Copy | #9 The Room Around The Event | #6 + the longer decay of an empty room | #14 Reported Speech Of The Object |

**Shared across all four (the pair's form, deliberately identical):** verse economy **8 → 4 → 2**; the chorus
byte-identical three times, header included; the rupture as a **substitution** at the midpoint; the vowel
ladder resetting at each shortening; `two years` sung exactly once, inside a reply.

---

## 2. RAW MUSIC PROMPT — V1, first state *(kept for the record)*

> Progressive chamber pop leaning into slinky, almost-R&B neo-soul at eighty-four beats per minute, minor
> moving to its relative major, **unhurried and lit from one side**. First five seconds: a folding chair
> scrapes once on parquet, then one felted Rhodes chord blooms with the plate reverb already wide open,
> **drawing** a hall far deeper than the four players **standing** in it. Female lead, warm low alto,
> close-miked at a hand's distance, conversational, breath audible, dry and forward **while the room behind
> her stays enormous**. Palm-muted electric guitar clicking on the offbeats, **slow warm** Rhodes tremolo,
> brushed snare **in circles on** a coated head, upright bass buzzing **lightly** at the frets; melodic-jazz
> voicings **underneath**, neo-baroque suspensions **and passing thirds** thickening only when the chorus
> arrives. The brushed backbeat drags a sixteenth behind the Rhodes and never corrects, so the groove leans
> **the whole way through**. Acoustic instruments **played together** in one room. At the midpoint the Rhodes
> hands its exact figure to a dry upright piano recorded closer, same bar, same notes, no gap.

**MEASURED: 1,093 characters.** ⛔ **FAIL** — the gate is 850–1000 with a 870–960 target. Over the hard cap
by 93 and over the target ceiling by 133. *(V2 raw: 1,007 · V3 raw: 997 · V4 raw: 990 — all four failed, and
all four failed **upward**, which is the characteristic direction: the prompt reads better every time a
clause is added and the cap is the only thing that disagrees.)*

⭐ **The repair was clause-deletion, never truncation.** Bolded above are the eleven clauses removed. Every
one is an adjective doing work an adjacent noun already did. **A mid-phrase cut to fit the cap is a hard
fail** — the field must end as a complete sentence, and it does.

---

## 3. THE MEASURED DEFECT TABLE — raw pass, all four

| # | Defect | Where | Raw measured value | Gate | Fix applied |
|---|---|---|---|---|---|
| D1 | Music prompt over cap | all four | **1093 / 1007 / 997 / 990** | 850–1000, target 870–960 | eleven-to-three clause deletions each → **955 / 937 / 948 / 955** |
| D2 | Mean sung line drifting to prose | V1 verse 1 raw | **8.50 w/l** for the stanza | ceiling 7.5 overall | rewritten to short hardware nouns + caesura → stanza 6.50, song **6.65** |
| D3 | Same defect, V3 | V3 verse 1 raw | **8.38 w/l** | ceiling 7.5 | six lines trimmed → song **7.31** |
| D4 | Same defect, V4 | V4 verse 1 raw | **9.00 w/l** | ceiling 7.5 | three lines cut to fragments → song **7.27** |
| D5 | ⛔ **Cross-variation line collision** | V1 & V2 rupture | line `The room is fixed. The room is fine.` present in **both** | pair distinctiveness | V2's replaced with `It says the room is ready to be a room.` |
| D6 | ⛔ Collision, second class | V1/V2/V3 rupture | all three drafted `New glass. New hooks. A coat on the patch.` | pair distinctiveness | V2 → `It is up again, a little to the left.` · V3 → `It hangs a finger left of where it hung.` · V1 keeps it |
| D7 | ⛔ Collision, third class | V1 & V3 verse 1 | V3's raw verse 1 was V1's verse 1 with pronouns swapped (5 of 8 lines) | pair distinctiveness | V3 verse 1 rewritten from a different door — the father's job, the shoes, the shelf |
| D8 | ⛔ Collision, fourth class | V3 bridge & V4 chorus | both logged the shift as `quiet` | motif vs. skeleton | V3 → `I wrote: no incident.` |
| D9 | ⛔ **Valuation register** | V2 chorus | `there is anything here **worth** a rope` | Panel A: no song about a rare object's worth | → `there is anything **behind** the rope` |
| D10 | ⛔ **Second-person address** | V3 rupture raw | `You can see it if you stand at the back.` | overhearing, not addressing | → `It shows from the back of the room.` |
| D11 | ⛔ Numeral leak | V4 intro, V4 tag | `the **first** thing anybody would say`, `in a full **one**` | one sung number, `two years`, and no other | → `That is what anybody would say.` · `He has heard it with the wall bare / covered.` |
| D12 | ⛔ Numeral leak, second | V1 rupture raw, V4 verse 1 raw | `Four small holes`, `nobody has sat since the tenth` | same | → `Small filled holes.` · `The bench says nobody sat.` |
| D13 | Portable thesis climbing into the lyric | V1 & V3 Section B raw | `Nobody chose. It came by weight.` and `Somebody decided by not deciding.` | ⛔ ruling 6, the portable-thesis veto | replaced with the **procedure**: a form with a box for who decided, and the box is blank |
| D14 | Catalogue with no verb | V4 verse 3 raw | `A flat new clip. A stanchion that stands. / The vent. The bench. The ledge. The step.` | ⛔ "a catalogue has no verb" | verbs restored: `He snaps the new clip flat and stands. / He passes the vent, the bench, the step.` |

⭐ **D5–D8 are one finding, not four.** Four variations written by one mind against one brief converge on the
same lines, and every individual line was *good*, which is exactly why none of them announced itself. The
detector that found them is set arithmetic over `sung()` output across all four songs — **the pair-isolation
law needs an instrument at the variation level too, not only at the pair level.** *(Carried to step 10's
self-check and re-run there.)*

---

## 4. RAW CHORUS DRAFTS — before and after *(the hook grammars, first state)*

**V1 raw** — the anaphoric chant. Drafted line 5 as *"Hands on the key. Breath on the glass. Light on the
street."* ⛔ **`Light on the street` is a banned collision surface** — flair #15 is motif-only and no pair may
write a line about light. Repaired to `Hands on the latch, breath on the sleeve, feet on the street`, which
also completes a full `ah → eh → ee` ladder **inside one chorus line**, so the device survives for a listener
who never hears a verse twice. *(The repair is better than the draft. Recorded because that is not always
true and the ledger should say when it is.)*

**V2 raw** — the cascade was drafted with `The room` anaphora on all eight lines. ⛔ That is V1's grammar
wearing V2's nouns. Repaired to a **serial `then`-cascade with no repeated subject**, which is a genuinely
different sentence shape and also drops two words per line.

**V3 raw** — the envelope chorus drafted `Not for the picture. A wall is not a listener.` ⛔ `A wall is not a
listener` is V1's pre-chorus line. Repaired to `Nothing here can be told`, and the pair's four non-reception
formulas were then **allocated explicitly**, one each:

| | the establishment that must land before the first chorus |
|---|---|
| **V1** | *A wall is not a listener.* |
| **V2** | *Plaster takes paint. It takes nothing said.* |
| **V3** | *Nothing in here gets told a thing.* |
| **V4** | *Plaster has no ear in it.* |

**V4 raw** — drafted with the quotation processed through a filter to separate the speakers. ⛔ Cut: a
processed quotation is a *device*, and the flair is **testimony**. The separation is now a **performance**
choice (quoted lines flat and six inches closer, narration keeps its contour) and costs nothing in the mix.

---

## 5. PANEL REVIEW — the supplied eighteen voices, on the raw pass

*(The ICB's roster, used as given. The three Hyper-Skeptics dissent for real; two of the three changed the
draft.)*

### Concept room
**THE AMBIGUOUS PHOTOGRAPH (after Sebald):** The raw V3 narrator is certain. She says the return did not
land in her hands and she says it like a finding. Make her *wait* for it and report that it did not come —
the difference between a verdict and a measurement is one line of syntax.
→ **TAKEN.** `I waited for it to arrive in me. / It did not arrive.`

**THE GLITCH CODIFIER (after Menkman):** The identical copy is in the chorus and nowhere in the *object*.
Put it in the furniture: the thing that came back is the same picture in a different rectangle.
→ **TAKEN.** New glass, new hooks, a coat over the patch, and in V3 it hangs a finger left of where it hung.

**⚠️ THE HABITUAL-MEMORY SKEPTIC (after Chun) — dissent:** *"Every draft here is about a thing being gone.
State the mechanism or you are doing mood. The mechanism is that the room is maintained by somebody who has
no guarantee anyone will keep paying for it, and when the picture returns his maintenance is not renewed —
it is simply no longer noticed."* ⛔ **She is right and the raw pass did not have it.** → **TAKEN into
Section B in all four:** the form with a blank box, the drawing from an office, the room declared fine in
writing, and — the honest half — *it is in writing and it is not wrong.*

### Medium room
**THE RAIN-GAP PRODUCER (after Burial):** Do not let the rupture be a hole. Put a page-turn, a knock, a
breath exactly where the thing left.
→ Already the pair's assignment; reinforced as the SFX cue that names the hand-off.

**THE PSYCH-POP ELECTRONICIST (after Broadcast's work):** ⛔ **The strongest objection to the raw pass.**
*"Four treatments of a source and not one tune. The chorus has to be singable by somebody who has decoded
nothing."* → **TAKEN.** V1's chant is four syllables and unaccompanied on its first appearance; V2 gets a
two-line refrain on the rim; V3's envelope repeats only at the two ends; V4's hook is a quotation, which is
the most singable thing in the language because somebody else already said it.

**⚠️ THE PROCESS COMPOSER (after Glass) — dissent:** *"'The plate describes a room larger than the
arrangement justifies' is a mood. What does it DO?"* → **TAKEN, and it changed the style field:** the plate
is now stated as an operation with a constant — *one setting held through the substitution*, so the room is
proven unchanged while the furniture changes. That is a statable process and it is now measurable on the
render.

### Context room
**THE ANGST CARTOGRAPHER (after Rodrigo's work):** Name the object on the bench. Raw V2 had none.
→ **TAKEN.** The mat that takes the yard in; the clip gone soft; the path worn in the floor where the
looking was.

**THE PLATFORM-LABOR SCHOLAR (after Hesmondhalgh):** The romantic frame deletes the person who is paid.
→ **TAKEN, and it produced the line the pair may be remembered for:** *he is the part that is paid.*

**⚠️ THE COHORT ABOLITIONIST (after Cohen) — dissent:** *"Scan for `we`. Then scan for `you`."*
→ **RUN, MECHANICALLY, not by eye.** `we / us / our / everyone`: **zero occurrences** in all four sung
fields. `you / your`: **one** — V3's raw rupture — **removed**. *(D10.)*

---

## 6. THE FOUR REFINEMENT LOOPS — what each pass actually moved

| Loop | Fired on | Gate value before → after |
|---|---|---|
| **1 · length** | all four music prompts | 1093 / 1007 / 997 / 990 → **955 / 937 / 948 / 955** chars |
| **2 · prose drift** | V1, V3, V4 lyrics | 8.50 / 8.38 / 9.00 stanza w/l → song-level **6.65 / 7.31 / 7.27** |
| **3 · distinctiveness** | V1↔V2↔V3↔V4 | 8 shared sung lines across the set → **0 shared sung lines** |
| **4 · doctrine** | V2, V3, V4, V1 | valuation register, a second-person address, three numeral leaks, two portable theses → **all cleared, re-measured** |

⛔ **No loop fired more than once on the same gate.** The repair budget is three attempts per failing gate and
the maximum used was one.

---

## 7. RAW TITLE DRAFTS AND THE CUT

| Draft | Ruling |
|---|---|
| *The Guard's Room* | ⛔ cut — it is the concept label, not a title; three of the four could carry it |
| *Two Years Of Nothing Happening* | ⛔ cut — puts the sung fact in the title, which spends it before the hinge |
| *It Came Back On A Tuesday* | ⛔ cut — invents a specificity the lyric does not support |
| *I Was Here Both Times* | ⛔ cut — `both` is number-adjacent and the pair is scanning for exactly that |
| *What The Wall Said Once* | ⛔ trimmed — `once` is the same exposure; **→ *What The Wall Said*** |
| **He Works The Wall** | ✅ kept — four words, the chant, no thesis |
| **The Room Keeps Its Hours** | ✅ kept — the room as agent, matches V2's grammar |
| **I Kept The Empty Room** | ✅ kept — first person, and *keep* is the polyptoton the pair runs on |
| **What The Wall Said** | ✅ kept — names the flair without explaining it |

---

## 8. WHAT STEP 09 INHERITS

- Four refined field sets, **all inside gate** on the measurements above.
- Two open questions for the influence pass: **(a)** does V4's quotation want a *plainer* delivery than the
  narration or a *slower* one — flatness and slowness are different instructions and only one survives a
  generator; **(b)** V3 is the least repetitive song of the four (exact-repeated-line ratio **0.227** against
  V1's 0.393) and sits closest to the house return floor, so its envelope chorus is the element most likely
  to be judged thin. Both are step-09's to settle.
- ⛔ **Carried, unresolved by design:** the Steyerl-seat / Dean-seat dissent about whether the carrier is
  neutral. This pair sits with **Dean** — the object came back and the room did not, because the carrier was
  never the point and the *hours* were. Another pair takes the other side. That is the ICB's instruction and
  it is not this pair's to close.

---

## PROVENANCE
Step file: `skills/music/steps/08_Generate_Music_Generation.md` (raw prompt generation, panel critics'
review, refinement loops 1–4, draft title).
Inputs: the frozen ICB, `pair_03_step06_facets.md`, `pair_03_step07_song_guides.md`,
`pair_03_step05_isolated_brief.md`, `06_music_handoff.md`, `EMOTION_TAXONOMY.md`, `vault/gates.yaml`,
`scripts/measure_soundcraft.py` (the canonical `sung()` definition).
**Self-critique, one line:** the raw pass failed **fourteen** measured checks and thirteen of them were
invisible to reading — the one that a careful read would have caught (`Light on the street`) was caught by a
banned-phrase scan first, which says plainly that on this pair my eye is a worse instrument than a regex and
should be used second.
