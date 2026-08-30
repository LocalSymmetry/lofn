# PAIR 02 — STEP 09 · ARTIST CRITIQUE & REFINEMENT
### `2026-08-07-daily-music-indignation` · **P02 "THE YEARS BACK"**

**ICB:** 53,003 B · sha256(LF) `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374` · **not edited.**
**Quarantine:** `06_music_handoff.md` §1 cited and obeyed. `09_Generate_Music_Artist_Refined.md`
carries "Examples of Effective Style Prompts" / "Example Lyrical Mixing" sections — **not read, not used.**

---

## 1. ⭐ WHAT I EXTRACTED, BEFORE ANYTHING I CONCLUDED

⛔ An empty extraction is a hard ERROR, never a score. These are the values the instruments
returned, not my impressions.

### 1.1 Measured floors — `scripts/measure_soundcraft.py → profile()` on the step-08 drafts

| var | lyrics field | sung lines | `rhyme_return` | `line_return` | **lexical-only** `line_return` | `allit/100w` | `unique_line_ratio` | w/line |
|---|---|---|---|---|---|---|---|---|
| **V1** | 4707 | **77** | **0.753** | 0.623 | **0.623** | 18.58 | 0.558 | 8.81 |
| **V2** | 4774 | **77** | **0.714** | 0.597 | **0.581** | 16.87 | 0.571 | 8.62 |
| **V3** | **4840** ⚠️ | **77** | **0.714** | 0.610 | **0.610** | 19.61 | 0.558 | 9.27 |
| **V4** | 4791 | **77** | **0.662** | 0.571 | **0.571** | 17.37 | 0.584 | 9.27 |
| **FLOOR** | <5000 (target ≤4800) | 70–120 | ≥0.30 | ≥0.20 | — | ≥11.0 | ≥0.45 | — |

**Every floor cleared on the first draft.** `rhyme_return` runs 2.2–2.5× floor; `line_return`
2.9–3.1×; alliteration 1.5–1.8×; `unique_line_ratio` 1.24–1.30×.
**Sung lines: 77 in all four — above the ≤72 hug threshold, inside the 78–110 target's lower
neighbourhood. ⚠️ NO hug FLAG is raised, and I am stating the measured number (77) so that
claim is checkable rather than asserted.**

### 1.2 ⚠️ MANDATORY DISCLOSURE — the wordless device and `line_return`
The handoff requires disclosure if a wordless device carries `line_return`. **It does not here,
and I can show it:** the lexical-only recomputation (drop every line with fewer than two words,
then re-measure) returns **0.623 / 0.581 / 0.610 / 0.571** against the raw
0.623 / 0.597 / 0.610 / 0.571. V1, V3, V4 are **identical**; V2 moves by 0.016 because of its
one-word clerical lines (`Received.` / `In order.`).
⭐ **P02's return device is a REMOVAL.** It cannot inflate `line_return` — deleting a line from
its own address strictly *reduces* the measure. The number is carried entirely by lexical
repetition (chorus ×6 statements, tag couplet ×3, the landed line ×2). **The instrument is not
being fooled; it is being under-credited.**

### 1.3 Per-variation device verification — ⭐ EACH ONE INDIVIDUALLY, never pair-wide

| Check | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| Line 1 of Verse 1 names **Warin** + a physical act | ✅ | ✅ | ✅ | ✅ |
| Landed line at **verse-line 5**, Verse 1 | ✅ pos 5 | ✅ pos 5 | ✅ pos 5 | ✅ pos 5 |
| Landed line at **verse-line 5**, Verse 2, **byte-identical** | ✅ pos 5 | ✅ pos 5 | ✅ pos 5 | ✅ pos 5 |
| Final Reprise: 8 slots kept, **slot 5 empty**, 7 sung lines | ✅ | ✅ | ✅ | ✅ |
| Landed line **absent** from the reprise | ✅ | ✅ | ✅ | ✅ |
| Concession at Verse-1 line 4 | ✅ | ✅ | ✅ | ✅ |
| Sung fact `five times / two hundred years`, once, at the Hinge | ✅ | ✅ | ✅ | ✅ |

**8/8 mark placements. 4/4 holes. 4/4 named recipients.** Verified by script over the parsed
section tree, not by reading.

### 1.4 Banned-term sweep (scripted, per variation)

| Sweep | Result |
|---|---|
| Abstract nouns (cost/authenticity/labour/soul/value/meaning/art/arts/truth/purpose) | **NONE** in any of the four |
| Validator debris (`prompt`, `QA gate`, `taxonomy`, `production manual`, `this song is about`) | **NONE** |
| Run-wide texture bans (hiss/crackle/wow-and-flutter/vintage/analogue warmth/lo-fi/degraded/corrupted/glitching) | **NONE** |
| Digits in sung lines | **NONE** |
| Wall-clock times in section headers | **NONE** |
| Bare `EMO:AWE` / `EMO:INDIGNATION` / `EMO:SYNTHESIS` | **NONE** — 21 distinct taxonomy terms used, all present in `EMOTION_TAXONOMY.md` |
| `[` or `]` inside a chorus line | **NONE** |
| Real-artist names (validator blocklist) in any Suno field | **NONE** |

**Number-words found in sung lines:** `five`, `two`, `hundred` (**the one allocated fact**),
`seven` (**the NAME of the liberal arts, never a count** — declared at step 07 so QA can rule on
it), `one` (**pronominal in every instance** — verified line by line: *"the one at the table"*,
*"the one on that bench"*, *"the last one"*, *"Every one. Instead"*; **zero counting uses**).

---

## 2. THE ADVERSARIAL PASS — what is actually wrong with these drafts

### ⚠️ D1 — V3's lyrics field is **4840**, over the 4800 target (under the 5000 hard cap).
Real, small, and the only measured failure in the set. **→ REPAIR R1.**

### ⚠️ D2 — THE DESCRIBE-RENDER SELF-CHECK (one pass, as the contract allows)

**What would this actually produce on Suno?** A 128 BPM Latin-inflected club track in a minor-ish
mode, organ-forward, with a conversational female vocal in the verses and a large chantable
unison chorus with claps. The `*one bar, organ alone*` cue will most likely render as a genuine
instrumental bar, which is exactly the device. The list/tag sections will probably come out
*sung* rather than spoken. **Highest-probability generator deviation: the seven-beat organ loop
gets smoothed to eight** — a specification that fights the grid is the classic L22 casualty.

**⭐ "Name the one way this renders generic."**
**Here it is, honestly: the chorus.** *"The wish is older than you"* over a 128 BPM four-four kick
with claps and massed unison is **one degree from a festival record.** If Suno equalises the verse
energy to the chorus energy — which is its default instinct — the dry, close, amused verse
disappears, the joke dies with it, and what is left is a pleasant uplifting dance track with an
odd lyric. **The verses are the only thing keeping this specific, and they are the part a
generator is most likely to inflate.**

**→ SELF-REPAIR (one pass, R2).** The fix is not an adjective — an adjective is what gets
smoothed. Following seat 17 (variance is detected as **note placement**, not as vibe) and seat 15
(import the **lean as timing**), the anti-generic instruction moves into the MUSIC PROMPT as
**arrangement facts a generator can act on**:
1. **the kick is absent from the verses** — stated as an instrument that is not playing, not as
   "sparse";
2. **the verse timekeeper is a jangling triangle and a hand shaker, standing in for a hat
   pattern** — a ciranda/forró articulation that a house prompt does not produce;
3. **the organ never doubles the vocal line** — kills the unison-anthem default;
4. **the seven-beat loop is written as a walk:** *"its first note lands one beat earlier through
   each bar and only comes home every seventh bar."* That is a placement instruction, not a mood.
5. the vocal is specified at **speaking loudness** so the mezzo cannot be pushed into a belt.

### ⚠️ D3 — Is the INDIGNATION actually present, or is this just charming?
**Honest answer: it is present and it is quiet, and that is correct for this pair but it must not
be zero.** The Albini fast-and-loud dissent is discharged by P01 (152) and P04 (168), not here.
P02's teeth are in exactly three places and I am naming them so a reader can check rather than
take my word: *"Every edition sold. Every single edition sold."* (V3 Hinge) · *"Nobody ever paid
the bill, and every copy sold."* (V4 Hinge) · *"and it is granted, and it is granted, and it is
not for you."* (V2 Hinge). **The teeth are aimed at the reselling of the shortcut — at the thing
that keeps a two-hundred-year bestseller in print without ever once working.** ⛔ Not at Warin,
not at copying, not at the long way, not at anyone who prefers it. **LAW 1 holds and the shared
enemy is never stated.** No repair; flagged for QA to rule on rather than silently ticked.

### ⚠️ D4 — Self-pity sweep, line level (the standing Morozov tripwire, seat 6)
Swept every sung line in all four. **No line asks to be forgiven, understood, believed or pitied,
and no line states what the speaker lacks.** The three closest candidates, examined and kept:
- V2 *"The recipient was not at the address. The recipient is me."* — a delivery-failure notice.
  Flat. The injury is structural and the listener assembles it. **KEEP.**
- V1 *"She is fast. She is very fast. You would not have liked her."* — self-description with no
  appeal attached, in the third person, inside a docket. **KEEP.**
- V4 — ⭐ **the hardest edge in the pair, and it holds:** the speaker itemises a price she never
  paid **and never says so.** There is no *"I never had to,"* no *"that is the difference."* The
  arithmetic is left entirely to the listener. **This is the guide's hardest instruction and I am
  confirming it survived drafting rather than assuming it did.**

### ⚠️ D5 — One sentimentality flag, examined and kept
V1 Lift: *"and for your mother, who is told that you are doing well."* Softest line in the set.
**KEPT** because it is Warin's stake, not the speaker's, it is specific, and it is the smallest
and most human of the seven reasons listed. It is not a bid for the listener's sympathy for the
speaker — the tripwire is aimed at her, and she is not in the line.

### ⚠️ D6 — One abstract-adjacent word, disclosed rather than hidden
V1 Lift: *"That is a good enough reason. That is the reason anyone has."* — `reason` is **not** on
the constraint-6 banned list, but it is abstract-adjacent. It is **anchored**: the four preceding
lines are the concrete reasons (a bench, a man who corrects him, an argument about the moon, his
mother). **KEEP, disclosed.**

### ⚠️ D7 — DEVIATION FROM MY OWN STEP-07 DIRECTION, logged
Step 07 §B row 11 directed the Final Reprise's seven surviving lines to be **new, not repeats.**
**They ship as echo lines instead.** Reason: the address has to be **recognisable** to be
countable. If the reprise is new material, the listener has no shape to count against and the
hole is just a bar of organ. Echoing the verse puts the listener back inside the eight-slot
frame, and the missing fifth is then undeniable. ⭐ **The rhyme pays it off too:** the landed line
ends `…warm`, its neighbour ends `…arm`, and in the reprise **the surviving half of that couplet
has nothing to answer it.** The absence is audible in two organs — position *and* rhyme.
`unique_line_ratio` was re-measured after the change: **0.558–0.584, all above the 0.45 floor.**

### ✅ D8 — THE FUNNY, counted rather than claimed (⭐ zero jokes = doctrine failure, QA 2026-08-06)
Required: ≥2 dry laughs per variation. **Measured by naming them:**
- **V1 (4):** *"Fetch them before supper. It is only the stars."* · the version-number joke
  (*"the last one had it wrong and this one is the one that works"*) · *"Who writes that down?"* ·
  *"She is fast. She is very fast. You would not have liked her."*
- **V2 (5):** *"granted instead / of nothing, which is what you were expecting"* · *"The words
  were, in fact, extremely well composed."* · *"The docket stays open. Nobody knows where it
  went."* · *"Delivery was attempted. Delivery was attempted again."* · *"The recipient is me."*
- **V3 (4):** *"Different winter. Same lean."* · *"the stars are extremely fine"* (report-card
  register) · *"They have had it since the morning. It is the middle of the night."* · *"It is the
  oldest thing in the room and it is not old."*
- **V4 (4):** *"You may keep your hands."* · *"Read it line by line and it is fair… Read it end to
  end and it is a monstrous thing to send."* · *"That is how bills work. That is exactly how bills
  work."* · *"Nobody reads it to the end."*
**17 across the set. ⭐ The load-bearing one is the *Ars Notoria* running to five editions over
two centuries, which is the same joke as a course that teaches you how to ask properly — and the
song never says the second half of that sentence.**

---

## 3. THE REPAIR LIST (max 3 per gate; 2 used)

| # | Target | Repair | Gate |
|---|---|---|---|
| **R1** | V3 lyrics field | Trim **53 chars of section-header cue** — the cheapest thing in the field, because cues are production hints, not craft. `Verse 2` cue → drop `, organ under`; `Lift` cue → `triangle doubles`; `Breakdown` cue → drop `and triangle`; `Final Reprise` cue → `one bar left empty`. **⛔ Not one sung line is cut.** Re-measured in step 10. | field ≤4800 |
| **R2** | all four MUSIC PROMPTs | The describe-render fix (D2): kick absent from verses, triangle+shaker as verse timekeeper, organ never doubles the vocal, the seven-beat walk written as note placement, vocal pinned at speaking loudness. | anti-generic |
| — | V3 + V4 MUSIC PROMPT length | V3 measured **848** and V4 **808** — both **below the 850 floor.** Extended with a substantive arrangement sentence (the reprise's open bar), **not with padding**. Re-measured: **V3 927, V4 887.** | 850–1000 |

⛔ **No repair touched the return device, the landed line, the concession, the sung fact, the
addressee, or any joke.** Every repair is confined to a header cue or a style field.

---

## 4. REFINED FIELDS (carried into step 10 verbatim)

### 4.1 Music prompts — measured
| var | chars | band 850–1000 | target 870–960 | hug ≥985 | terminal punctuation | artist names | `avoid`/`do not`/`blacklist` |
|---|---|---|---|---|---|---|---|
| V1 | **906** | ✅ | ✅ | no | ✅ `.` | none | none |
| V2 | **877** | ✅ | ✅ | no | ✅ `.` | none | none |
| V3 | **927** | ✅ | ✅ | no | ✅ `.` | none | none |
| V4 | **887** | ✅ | ✅ | no | ✅ `.` | none | none |

**Banned-descriptor sweep on all four prompts** (raw · aggressive · relentless · brutal ·
explosive · massive · intense · pounding · driving · battle · assault · phonk, plus all nine
run-wide texture words): **ZERO hits.**
**Permitted vocabulary actually used:** `dry` · `close-mic'd` · `conversational` · `amused` ·
`consonants` · `breath` · `bar` · `beat` · `hand` · `body` · `clean` · `bright` · `present`.
⭐ `snarl` was **available and deliberately not used** — this pair's register is amused, and a
snarl at conversational loudness would read as a different song. The rotation instruction
(female mezzo, conversational) is honoured.

### 4.2 Titles
| var | title | against the step-07 direction |
|---|---|---|
| V1 | **Fetch Them Before Supper** | ✅ "name the errand, not the seven" |
| V2 | **Received, In Order, In Full** | ✅ "a filing phrase" |
| V3 | **Same Bench, Same Lean** | ✅ "name the posture, not the era" |
| V4 | **You May Keep Your Hands** | ⚠️ **logged deviation.** The direction said "the register of a bill." The bill register is fully carried *inside* the lyric (`Item:` × 9, `Read the list`, `Nobody paid the bill`); putting it in the title as well would have been the third statement of one idea. The title takes **the last item on the list instead**, which is the line the song is built to arrive at. |

### 4.3 Exclude prompt — shared spine, per-variation tail
Base field measured **398 chars** (≤1000 ✅). Scripted check for prose negation
(`avoid|do not|don't|must not|please`): **ZERO hits** — concrete comma-separated terms only.

---

## 5. HUMAN SUBJECT STANDARD — re-run post-draft, not just pre-draft

- **PERSON:** Warin — invented; no locating tuple; no real individual resolvable. **Pamphilius**
  (V2) is the angel named inside the 13th-century manuscript itself — a mythic figure, not a
  person. The modern figure in V3 is deliberately **unnamed, ungendered by pronoun (`they`), and
  given no job title**, precisely so no listener can resolve them to one real displaced worker —
  which is also how the **no-ventriloquism** rule is kept: she *describes* them to Warin and never
  speaks as them.
- ⛔ **No member of Papangu, no producer, no studio, and no living person appears as speaker,
  character or addressee** in any of the four. The record released today is the run's *occasion*
  and never its *character*.
- ⛔ **Binding refusals absent from all four:** the Thai school shooting · Ceuta / the 78,000 ·
  the Biden family illness. No minor is depicted. No real harmed person is named or reconstructed.
- ⚠️ `scripts/check_human_subjects.py` was **not** used as an authority: the handoff records that
  it fires `HOLD_FOR_HUMAN` on 100% of correct artifacts (spaCy absent → its regex reads
  `Female`, `Vocalist`, `List` as person names and `body` inside `nobody`). **A gate that fires on
  everything carries no information.** The standard was judged directly, against §3.0's slot
  grammar, above. **Verdict: PASS, no HOLD-FOR-HUMAN condition present.**

---

## 6. COMPLIANCE LEDGER — the constraints, each with its evidence

| Constraint | Status | Evidence |
|---|---|---|
| 1 · THE UNDELIVERABLE ADDRESS | ✅ | The thesis is nowhere stated. Warin cannot receive any of this. V2 makes the undeliverability literal: *"The recipient was not at the address."* |
| 2 · THE NAMED RECIPIENT | ✅ | Warin, line one, ruling / holding / saying — a physical act in all four. No "you" as addressee, no "the world." |
| 3 · THE COUNTABLE OBSTRUCTION, in the lyric | ✅ | Verse-line 5, marked twice, empty the third time. Verified per variation (§1.3). ⛔ Nothing about the device is delegated to the production spec. |
| 6 · No abstract nouns · no ventriloquism · no self-pity | ✅ | §1.4 sweep · §5 (V3's third-person description) · §2 D4 line-level sweep |
| 7 · Texture-word ban, both directions | ✅ | Zero hits in lyrics **and** in prompts. The render is clean, modern, expensive; the machine is not made to sound damaged either. |
| LAW 1 · teeth never at them | ✅ | §2 D3 — the three teeth located and named. |
| LAW 2 · agrees by line four | ✅ | Verse-1 line 4 in all four; never re-litigated after. |
| LAW 3 · no self-pity | ✅ | §2 D4 |
| LAW 4 · contradiction is structural | ✅ | FORM RULE B only. P02 never touches RULE A accretion. |
| Style Vocabulary Law | ✅ | §4.1 |
| Max one sung numeric fact | ✅ | §1.4 |
| Vocal rotation (female mezzo) | ✅ | specified in all four prompts |
| Cross-pair device bleed | ✅ | reveal engine, return device, sung fact, verse architecture, genre lane and register are P02's alone |

---

*Step 09 complete. Two repairs specified, both confined to header cues and style fields.
→ `pair_02_step10_revision_synthesis.md`*
