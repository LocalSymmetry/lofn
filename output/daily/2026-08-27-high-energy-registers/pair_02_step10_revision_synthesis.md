# PAIR 02 — STEP 10 · REVISION SYNTHESIS · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Pair:** 02
**What this step did:** the final adversarial pass over the assembled package — the field-length repair, the boundary-hug repair, and the four objections I raised against my own work and what happened to each.

---

## 1 · THE FIELD-LENGTH REPAIR — and why it happened at all

**First assembly, measured:**

| | music prompt | exclude | lyrics field | verdict |
|---|---|---|---|---|
| **V1** | ⛔ **1022** | ⚠️ **900** | 4400 | over the 1000 hard cap; exclude pinned to the top of the band |
| **V2** | ⛔ **1002** | 866 | 4399 | over the cap |
| **V3** | ⛔ **1004** | 875 | 4243 | over the cap |
| **V4** | 952 | 892 | ⚠️ **4854** | lyric over the 4800 target |

**After repair:**

| | music prompt | exclude | lyrics field |
|---|---|---|---|
| **V1** | **954** | **836** | **4400** |
| **V2** | **955** | **866** | **4399** |
| **V3** | **949** | **875** | **4278** |
| **V4** | **952** | **892** | **4778** |
| **band** | 850–1000, write **870–960** | 400–900 | <5000, target ≤4800 |

⛔ **The trim was made by cutting clauses, never by truncating a sentence.** `music_prompt_terminal_punctuation: true` is an unambiguous hard fail and a prompt truncated mid-phrase shipped on 2026-06-28 and self-check PASSed **because the check counted characters, not sense.** All four end as complete sentences; the measurement reports `ends_ok=True` for each.

**What was actually cut:** hedging and doubled description — *"and staying flat rather than climbing"* → *"and staying flat"*; *"with the ceiling audible over her"* → *"with the ceiling audible"*; *"Bass, brushes and brass are played and unprocessed, and there is no acoustic guitar anywhere"* → *"Bass, kit and brass are played and unprocessed; no acoustic guitar anywhere."* ⭐ **Nothing load-bearing left: every opening moment, every kinetic defect, every acoustic declaration and all four "stops once" events survive the trim.**

**V1's exclude was at 900 — the exact top of the band.** Boundary-hugging is a FLAG, and the fix was to drop three tokens that the other three excludes already carry (`cathedral reverb`, `reverse cymbal`, `string pad`), which also stops the four excludes being one field copied out four times.

---

## 2 · THE FOUR OBJECTIONS I RAISED AGAINST MY OWN WORK

### ⚠️ OBJECTION 1 — *"Is the refused element a real constraint or a costume?"*

**The charge:** banning six pronouns is a cheap trick; the protest anthem could still arrive wearing *anybody* and *nobody*.

**Ruling: it held, and here is the evidence rather than the assertion.** The four hooks are *Read it back* · *I am the last part of it to know* · *Filed it anyway* · *Check it on me*. **Every one is singular, and three of the four are in the past or the imperative.** There is no line in 335 sung lines that could be chanted by a crowd about somebody else — ⭐ **not because a crowd was forbidden, but because there is nothing in any of them for a crowd to be right about.** The prohibition removed the pronouns; the register removed the audience.

⚠️ **What survives the objection in narrower form:** the prohibition is **necessary and not sufficient.** It makes *we shall / they must* unassemblable. **It does nothing about the SOUND**, which is where the template actually arrives, inside two seconds, before a word does. That is filed as R7's default expectation and it is expected to fail.

### ⚠️ OBJECTION 2 — *"V4 is a philosophy essay with a beat."*

**The charge:** *the right procedure, the wrong outcome* is a seminar topic. The song will explain rather than happen.

**Ruling: partly upheld, and repaired.** The draft's V4 carried the argument in the verses and the argument is genuinely abstract. **The repair was not to cut the argument — it was to make the argument physically checkable inside the song.** The win condition is stated as an *instruction to verify* (*"Say it now so it can be checked later"*), the chorus is an *audit* (*"Check the whole count"*), and the philosophy is carried by a **number that lands on a body**: *the line came down at nine, I came in at eight.* ⭐ **And the decisive line was already concrete and stayed untouched:** *"Nobody made a mistake. / That is the part there is no argument with. / Nobody made a mistake."*

⚠️ **Residual risk, named and not repaired away:** V4 is the least somatic of the four and the most likely to be heard as flat. **That is also its design** — the singer leaves before the song ends. It is filed in R7 as prediction **P3**: *V4 is reported as sad, flat or resigned.* If a listener reports it as **boring** rather than **flat**, the objection was right and I was wrong.

### ⚠️ OBJECTION 3 — *"V3's line-return of 0.570 means 38% of the song is one block."*

**Ruling: rejected, and the rejection is doctrinal rather than a preference.** `chorus_repetition_requires_no_justification: true` exists in `vault/gates.yaml` **precisely because agents were pre-emptively mutating refrains to get under a flag** — *"a flag that makes writers apologise for choruses is doing harm even when it never fires."* `line_return` is a **FLOOR** (0.20). V3 is a hooky, accessible song in the run's fastest band with the run's only major key, and its chorus is its whole argument. **The number is reported in the verification table. It is not defended and it is not repaired.**

### ⚠️ OBJECTION 4 — *"The stranger test is asserted, not satisfied."*

**The charge:** the run's world material is a real, open criminal matter. A song about a window that shuts, a page that sits, and a clock that ran out **is** the wire, dressed down.

**Ruling: upheld as a risk and closed by inspection, item by item.** The check was run as a list rather than as a feeling:

| the wire's structural objects | used in this pair? |
|---|---|
| a deadline expiring **at dawn** | ⛔ no — no song contains dawn, morning light, or an hour of the morning |
| an **unsigned page** | ⛔ no — no signature anywhere; V1's rule was rewritten away from signing during design for exactly this reason |
| a **door that stayed shut** | ⛔ no — there is no door in any of the four |
| a **clock running out** | ⛔ no clock is sounded and no countdown occurs; the run's forbidden-default #16 and the stranger test agree here |
| a jurisdiction, a role, a title, a term | ⛔ none |

**What is used instead:** a posted rule about turns, a queue, a service window, an empty chair, a drop slot, a paper appeal, a numeric cut-off, a stamp, a click-pen, a wooden counter, a drawer, a receipt. ⭐ **All four scenes are a counter, a queue and a form** — rooms the listener has physically stood in, and none of which acquires a second meaning from this morning's wire.

⚠️ **The honest residue:** V2's *"The window took its ten and shut"* and V3's *"Nobody had to. It just sat"* are **structurally** the same shape as a deadline expiring in silence — and that shape is what the brief explicitly grants (*"the wire supplies structure"*). **The test is whether a reader of the wire hears something a non-reader does not, and the answer is no: there is no particular in either line that a wire could supply.**

---

## 3 · WHAT CHANGED BETWEEN STEP 09 AND THE SHIPPED PACKAGE

1. Four music prompts trimmed into the write-target band (1022/1002/1004/952 → 954/955/949/952), by clause removal, all still terminating as complete sentences.
2. V1's exclude reduced from the band boundary (900 → 836) and de-duplicated against its three siblings.
3. V4's lyrics field brought under the 4800 target (4854 → 4778) by compressing the `[SONG FORM:]` line and the final section header — ⛔ **no sung line was cut**, because the field's overflow was in the notation, not the song.
4. The sung-line counts in the package's own LINE BUDGET section were corrected against the instrument (**86 / 80 / 79 / 90** — the first draft of that table said 86/80/80/86 from memory). ⭐ **A number written from memory into a verification table is exactly FINDING 12** — three shipped violations last run sat under a verification row saying zero, each written by the pair being counted. This one was caught by re-running the command instead of re-reading my own table.

---

## 4 · WHAT I COULD NOT DO

- ⛔ **Nothing was rendered, so every acoustic claim in this pair is provisional.** `vault/music_atlas/FRONTIER.md`: **all 57 territories UNEXPLORED, 0 probes.** There is no measured prior for high-energy material through our door and **zero measured claims that EXCLUDE tokens do anything at all.** Nothing here is justified by *"we know this works loud."*
- ⚠️ **R4's replacement device is untested.** *A repeating physical event that stops once while everything else continues* is a **proposal**, not a measured survivor. The measured facts are that an internal void gets filled (2/2) and a near-silent opening survives (1/1). **Whether a stopped event survives is unknown**, and it is pre-registered as R7 prediction **P8** with a falsification condition that would demote the device.
- ⚠️ **I cannot hear any of this**, and the failure class that matters most for this register — *does it sound like a protest anthem in the first two seconds* — is invisible to every instrument I own. **Text gates cannot see it.** That is why the listener prediction is filed before any render rather than after.
