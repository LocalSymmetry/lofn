# THE DEDICATION STANDARD

*Established 2026-08-13 by The Scientist: **"Can we add that as a daily music requirement going forward.
We should log this for all our work."*** Format and budget confirmed 2026-07-14 on the **"Start With One"**
credits (`USER.md`).

---

## The rule

> **Every shipped piece carries a dedication. The dedication names what the piece borrowed and who it is
> owed to — and it is a DELIVERABLE, not a courtesy.**

**Budget: 500 characters**, including spaces, newlines and emoji. **Measured, never eyeballed.**
Overflow goes to a companion credits file / pinned comment, and the caption ends by pointing at it.

### Priority order when trimming (from `USER.md`, unchanged)
1. **The raised source-sound by name** + who reconstructed or recorded it.
2. ⭐ **The single sharpest session-player credit.**
3. **Living-scene lineage** — *"borrowed with lineage, not theft."*
4. **Honest AI authorship** — Lofn / Claude + The Scientist.
5. *"Full credits below."*

**Trim from 5 upward.** Never buy space by cutting 1 or 2.

---

## ⛔ THE RULE THAT MADE THIS A STANDARD — CREDIT THE FIGURE YOU NAMED

> **If the prompt names a break, a riff, a lick or a specific historical figure of sound, the dedication
> credits the person who PLAYED it — not only the living scene that inherited it.**

**Found 2026-08-13, on our own work, pre-render.** Two of six refined songs instructed the generator to
build an **amen break** — `A1` shipped *"quiet clipped amen edits"* (`chopped_amen_152BPM`), `B3` shipped
*"an amen-adjacent break rechopped every four bars"* (`amen_adjacent_break_172BPM`). Both had **full,
careful, multi-artist lineage blocks**. Between them they credited Nia Archives, underscores, UNIIQU3,
Machine Girl, Kai Whiston, Lumi Athena and Esdeekid.

**Neither credited Gregory C. Coleman or The Winstons.**

Coleman played the four-bar break on *"Amen, Brother"* (1969, B-side of *"Color Him Father"*) — the drums
jungle, drum & bass and a great deal of hip-hop are built from. **He died homeless in Atlanta in 2006 and
never received royalties for the most reused drums in recorded music.**

### Why the pipeline missed it, which is the transferable part
**Living-scene credit is easy, visible and flattering — and our gates already ask for it.** The session
player who actually made the figure is invisible, usually dead, and asks for nothing. **So the credit the
system forgets is the one that most needs paying.** A gate that says "name the scene and credit 2–3 living
artists" is satisfied *perfectly* by a block that omits the drummer whose four bars are the entire floor.

### ⭐⭐ THE TEST THAT CATCHES ALL OF THEM — **NOT "IS THIS FALSE?" BUT "COULD A READER TAKE THIS AS A CLAIM WE DIDN'T MAKE?"**
*(Given by P01, 2026-08-13, after closing a hold: "I was testing against **is this false?** and it passed,
which is exactly why it survived three of my own passes.")*

**Every credit defect found on this standard's first day was a TRUE statement that reads as a claim we did
not make:**
- *"the bassist in the room whose name is not the one anyone says"* — defensible, **reads as a verdict on a living man's standing**
- *"MFR Souls, whose log drum this is"* — they did build it, **reads as our recording containing their sound**
- *"The slap bass is Larry Graham's invention"* — true, **reads as *this* slap bass being his recording**

⛔ **Falsehood is not the failure mode. AMBIGUITY is.** A credit that is accurate and readable-wrong does the
same damage as a false one, and it survives every truth-check you run. **When the disambiguating fix is
under budget, take it** — hedging the referent (*"the slap bass **we asked for**"*) cost thirteen characters.
The three rules below are instances; **this is the test.**

### ⛔⛔ A DEDICATION IS PUBLIC TEXT ABOUT A REAL PERSON — **D4 APPLIES TO IT, NOT JUST TO THE LYRIC**
*(Found 2026-08-13 by QA, on this standard's first run. **This is the most serious defect the standard has produced, and the standard produced it.**)*

**CREDIT a living person: always. Say a song is ABOUT them: never.**

Two shipped captions stated that a real living man was a song's subject — *"ANDRÉ CYMONE… the man in the
room whose name is not the one anyone says. **This song is a handover sheet about exactly that**"* and
*"**the bassist in the room** whose name is not the one anyone says."* **All twenty-five LYRICS obeyed D4
perfectly.** The breach was entirely in the paratext.

⭐ **The transferable lesson: a new deliverable is a new surface, and an ethics gate that names one surface
does not cover the other.** We gated lyrics for decades of runs and shipped the violation in the caption on
the standard's first day — because the caption did not exist when the gate was written.

**The test, applied to every caption:** *does this sentence make a claim about a living person's life,
motives, or standing?* Crediting what they played is a fact about a recording. **Saying our song is about
their obscurity is a claim about them, made without consent, under our name.** ⛔ **HOLD-FOR-HUMAN,
non-waivable.** `vault/HUMAN_SUBJECT_STANDARD.md` governs the paratext exactly as it governs the lyric.

### ⛔ THE OVERFLOW BLOCK IS A SHIPPED SURFACE — THE CAPTION POINTS AT IT
*(Found 2026-08-13 by P04, after its own caption was repaired.)* Priority (5) is literally *"full credits
below"* — **the caption directs the reader to the companion block, so the companion block ships.** It is
page two, not a private appendix, and the 500-char budget **deliberately pushes content into it.**
**Every rule that binds the caption binds the overflow**: no claim about a living person, the honesty
clause, no implied-contents grammar. ⭐ **Repairing a caption while the block it points at still carries
the claim is a cosmetic fix** — P04 caught exactly this, in prose two shipped captions referenced.
**Audit caption + overflow + any credit block a caption names, as one artifact.**

### ⛔ THE HONESTY CLAUSE CAN BE BROKEN BY GRAMMAR, NOT ONLY BY A PHRASE
*(Found 2026-08-13 by P01, unprompted, during an unrelated hold.)* A caption read **"MFR Souls, *whose log
drum this is*"** — a **possessive that claims our recording contains their sound** — sitting three lines
above *"Suno generates, it does not sample."* A sweep for the literal string `"contains a sample"` returns
clean and misses it entirely.
**Credit the INVENTION, not our contents:** *"who built the log drum"* ✅ · *"whose log drum this is"* ⛔.
**Test every credit for an implied-contents construction** — possessives, "this is theirs", "taken from",
"lifted from" — not just for the banned sentence.

### ⛔ WORD BOUNDARIES ARE MANDATORY ON EVERY SWEEP
*(Two near-misses in two days.)* A substring sweep for `amen` false-positives on **`ornament` /
`unornamented`** and nearly produced a fabricated credit-failure finding; a substring sweep for `rage`
matched **`garage`** fifteen times and reported a UK-garage song as aggressive. **Use
`(?<![a-z])token(?![a-z])`.** A gate that cannot tell a word from a substring manufactures findings, and a
fabricated credit finding is as damaging as a missing credit.

### ⛔ SWEEP THE STYLE FIELD, NOT THE EXCLUDE FIELD — **RAISED ≠ FORBIDDEN**
*(Found 2026-08-13 on the standard's first run, by P04, which reported it rather than omitting it.)*
`amen` appeared in **all four of P04's EXCLUDE fields** — an anti-breakcore bloc keeping the pair off
distortion. A naive sweep of "every shipped field" would have credited **Gregory C. Coleman on four songs
that explicitly forbid the sound.** **The positive STYLE field is what raises a figure; the EXCLUDE field
is what refuses it.** Crediting a refusal is precisely the false credit this standard says makes the true
ones worthless. **A sweep that cannot tell raised from forbidden is not a gate, it is a name generator.**

### ⭐ TWO LEGITIMATE BRANCHES WHEN CREDIT AND ARTIFACT DISAGREE
When a dedication credits something the prompt does not actually ask for, **both repairs are valid — choose
by whether the artifact should change:**
- **FIX THE PROMPT** when the piece genuinely wants the thing. *(2026-08-13, P05 V4: the dedication credited
  the amen while the prompt said only "breakcore edits" — it corrected the prompt rather than soften the credit.)*
- **DROP THE CREDIT** when it does not. *(Same day, P04 V3: bass "restricted to one long note that swells and
  never changes pitch" — no slap, no thumb, so **Larry Graham and Bootsy Collins were deliberately NOT
  credited, and the caption says so and says why.** Adding a slap to earn a credit would damage the piece and
  invert the rule.)*
⛔ **Never paste one credit block across a whole pair.** *"Pasting one block onto all four would have been a
lie on one song in four."*

### ⚠️ THE BUDGET IS OVER-SUBSCRIBED WHEN A SONG OWES BOTH A PLAYER AND A SCENE
*(P04, first run: **priority (4) did not fit in any of four captions.**)* At 500 characters a caption cannot
always carry a session-player credit, a living-scene credit **and** its own authorship line.
**RULING:** **(4) honest AI authorship MAY move to the companion credits block** when the caption is
over-subscribed — it is **artist-level and identical across the catalogue**, and the profile carries it.
**(1) and (2) are song-specific and would otherwise be invisible, so they never move.** ⛔ **Never buy space
out of a player credit.** Flag the squeeze in the file, as P04 did, rather than resolving it silently.

### The honesty clause — do not overclaim either
A generated track is **not a sample.** Say so. The caption's claim is *"we asked for this figure by name,
so we name him,"* **never** *"contains a sample of…"*. **A false credit is its own dishonesty**, and it
would make the true ones worthless.

---

## Known debts — check every run against this list, and extend it
| Figure named in a prompt | Credit owed |
|---|---|
| **amen break** (any chop, "amen-adjacent", jungle/DnB/breakcore floors) | ⭐ **Gregory C. Coleman**, drums, and **The Winstons** — *"Amen, Brother"*, 1969 |
| **Funky Drummer** | **Clyde Stubblefield** with James Brown |
| **Think break** | **Lyn Collins**, *"Think (About It)"* |
| **Apache break** | **Michael Viner's Incredible Bongo Band** |
| a named historical medium/procedure | the tradition **and** the specific practitioner where one exists |
| **1960s girl-group vocal stack** | the singers, not the producer — e.g. **Darlene Love**, who sang lead on records released under other names |

⛔ **Never credit an abuser or a convicted violent offender as a lineage figure.** Credit the players.

---

## Where it binds
- **`lofn-music` step 10/11** — the dedication ships **with** the package. A package without one is incomplete.
- **`lofn-daily` Phase 3** — dedications are written for the selected set before the INDEX.
- **`lofn-qa`** — ⭐ **gate it:** does every shipped piece have a ≤500-char dedication; is every
  prompt-named source-sound figure credited to its player; is the no-sample honesty clause present where a
  generated track invokes a real break. **A missing player-credit is a REPAIR, not a note.**
- **`lofn-distribute`** — the caption that goes out is this caption.
- **Images, video, story:** same rule, same budget. *A borrowed medium is a borrowed sound.*

## The posture
**Naming a scene is amplification, never capture.** No credited artist appears in a Suno style field, in a
lyric, or is implied as a collaborator or endorser. Panel voices remain model-generated constructs "after"
published work. **Fusion with lineage, never capture. Borrowed with lineage, not theft.**

*Worked example, six captions measured 474–496 chars: `output/2026-08-13_genre-blend-refine/DEDICATIONS.md`.*
