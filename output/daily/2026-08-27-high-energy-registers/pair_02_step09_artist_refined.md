# PAIR 02 — STEP 09 · ARTIST-REFINED · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Pair:** 02
**What this step did:** repaired the three defects the step-08 instrument found, wrote the eight production fields (four music prompts, four excludes), and re-measured. **Every change below is a change I can point at, and every number is the instrument's, not mine.**

---

## 1 · THE LYRIC REPAIRS

### ⛔ DEFECT 1 — `loud` in two sung lines (R2)

| | before | after |
|---|---|---|
| V1 | *"I read it three times out loud."* | *"I read it three times **aloud**."* |
| V3 | *"I am saying that out loud on purpose."* | *"I am saying **so aloud**, on purpose."* |

⭐ Both are better lines. `aloud` is the exact word for what is happening and `out loud` was a reach for the vernacular.

### ⚠️ DEFECT 2 — V3's strict end rhyme at 0.291, under the 0.30 floor

The chorus was carrying only 3 of its 6 lines. **Repaired at the chorus, because the chorus repeats five times and is therefore the highest-leverage line in the song:**

| before | after |
|---|---|
| *Nobody has to read a word of it.* | ⭐ ***Nobody has to read it. Nobody will.*** |

**Why this is the single best edit in the pair.** It was made for a rhyme key (`ill` ↔ `will` in the next line, +2 hits at five repeats) and it produced the pair's sharpest couplet:

> **"Nobody has to read it. Nobody will. / I wrote it like somebody will."**

⭐⭐ The same word doing opposite jobs one line apart — **the passive injustice and the appetite, welded.** This is now V3's discriminator line, and it exists because a rhyme instrument complained. *The gate improved the art; that is not the usual direction and it is worth recording.*

### ⚠️ DEFECT 3 — all four under `alliteration_per_100w_floor: 11.0`

Repaired by consonant-clustering **lines that were already weak**, never by inserting texture. Twelve edits; the ones that changed meaning as well as sound:

| | before | after | what it bought |
|---|---|---|---|
| V1 | *Paper does not argue.* | ⭐ ***Paper does not push.*** *(kept, and `argue` follows it)* | **FORCE NEEDS SOMETHING TO PUSH AGAINST** — the seed, in the third line of the pair |
| V1 | *Do not stop where it goes good.* | ⭐ ***Do not stop where it stops suiting me.*** | the singer admits the self-serving reading **in the first verse** |
| V1 | *The one ahead of me went again.* | ***The one ahead of me went and went again.*** | the repetition is the offence, so the line repeats |
| V1 | *Read the line with me inside the line.* | ⭐ ***Read the line with me left inside the line.*** | *left* inside — stuck in it, not merely included |
| V2 | *It got here before I did.* | ⭐ ***It got here before I got here.*** | the answer arrived before the body did — the song's thesis, in the hook |
| V3 | *Filed it before the counter opened.* | ***Filed it first, before the counter opened.*** | she was first in the queue and it made no difference |
| V4 | *Nobody moved it. Nobody leaned on it.* | ***Nobody moved it. Nobody nudged it.*** | `nudged` is the smaller, more damning verb |
| V4 | *A line that bends for one / is a line that was never a line.* | ⭐ ***A line that bends for one / is a line that lets anybody lean.*** | a rule that bends once bends for everybody — **the register, in a couplet** |
| V4 | *(chorus, 6 lines)* | ⭐ **added *"Check the whole count."*** *(chorus now 7 lines)* | foreshadows THE COUNT four sections early, and makes the hook an audit |

⭐ **Every one of these was made to satisfy a mechanical floor and every one of them is a better line.** That is the argument for having the floors.

---

## 2 · THE EIGHT PRODUCTION FIELDS, WRITTEN AT THIS STEP AND NOT BEFORE

⭐ **THE GRAIN LAW: a Somatic or distinctiveness objection answered in the PRODUCTION SPEC is NOT ANSWERED.** The prompts were written *after* the lyrics were finished, so that the spec could not be asked to carry the discriminator.

**What each music prompt is required to contain, and where it is:**

| requirement | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| genre/micro-genre + tempo/energy/opening **first** | office-percussion funk breakbeat, 104 | dub-skeleton electronic soul, 100 | montuno-driven breakbeat soul, 108 | field-snare march into slow electronic soul, 102 |
| **vivid opening moment (first 5 s)** | one unaccompanied contralto, four flat words, one stamp | one dry kick at pulse rate, one chair leg | a hand landing twice on wood, one held horn | a pen clicked twice over one sub note |
| vocalist spec **with spatial staging** | dead centre, very close | very close, slightly off centre | close and centred | close and centred |
| instrumentation with **physical adjectives** | round, dry, brushed, papery, muted, struck | held, released, brushed, detuned, fluttering | short, round, brushed, pushing, snapped | tight, sticks audible, plucked, warm, muted |
| arrangement arc | at full size from bar one and staying flat | thin, then arrives at full size and never gets smaller | thin, full by the second hook, never bigger | full early and holding, every section landing where the last one promised |
| **kinetic defect (asymmetric groove)** | snare 4 ms late every second bar | the two early, the four late | the vocal rushes, the band does not catch up | the kit pushes, the pen never does |
| explicit **acoustic / no-acoustic** declaration | bass, kit, brass played and unprocessed; no acoustic guitar | bell, rimshot, kick struck and unprocessed; no acoustic guitar, no strings | piano, horns, shaker played and unprocessed; no acoustic guitar, no choir | snare, plucked strings, sub played and unprocessed; no acoustic guitar, no choir |
| **bold sonic device** | ⭐ the read-back double — one beat behind, same pitch | the arrival that never retreats | the singer who is always early | ⭐ the pen that stops on the changed word |
| **the thing that stops once** | stamp | chair leg | hand on wood | pen |

⛔ **Banned openers avoided in all four** (*Begin in/with…, Use…, Build the track from…, Chronology:, For an adult human singer…*).
⛔ **No company or artist name in any of the twelve fields.** V2's tape-replay keyboard is a **mechanism description** — *"a tape-replay keyboard with audible flutter and an eight-second note ceiling"* — never a trade name. `vault/DISC_CHANNEL_GUIDE.md` prescribes brand tokens and `06_music_handoff.md` forbids them; **the handoff governs, and the conflict is escalated, not resolved.**
⛔ **No level word, no width word, no `hard pan`, and no requested narrowing** in any prompt.

**The exclude fields** each carry the `00_research_brief.md` PART 3 forbidden-default list, rotated per song so the four are not one field copied out. ⚠️ **The `exclude` channel has ZERO measured claims** — we have never measured whether anything in an EXCLUDE field leaves. **These are a discipline for the writer, never a guarantee about the render, and no discriminator in this pair rests on one.**

---

## 3 · RE-MEASUREMENT AFTER REPAIR

**Command:** `python p02_measure.py p02_body.md`

| | field chars | sung lines | endRhy | lineRet | w/line | allit/100w |
|---|---|---|---|---|---|---|
| **V1** | 4400 | 86 | **0.593** | 0.477 | 5.80 | **11.62** |
| **V2** | 4399 | 80 | **0.438** | 0.475 | 6.54 | **11.28** |
| **V3** | 4278 | 79 | **0.405** | 0.570 | 6.04 | **11.53** |
| **V4** | 4778 | 90 | **0.356** | 0.422 | 6.21 | **11.27** |
| **floor / ceiling** | <5000, ≤4800 | 70–120 (78–110) | ≥0.30 | ≥0.20 | ≤7.5 | ≥11.0 |

**All four now clear every floor.** V3's rhyme moved 0.291 → 0.405; alliteration moved 7.24–9.11 → 11.27–11.62; `loud` is gone from all sung text; banned pronouns remained at 0 throughout.

⭐ **Cross-variation identical sung lines: 0 of 335. 0.0%.**

---

⛔ **Nothing rendered, published, or spent.** `vault/AUTONOMY.md` holds.
