# PAIR 02 — STEP 08 · GENERATION (FIRST FULL DRAFT) · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Pair:** 02
**Status of this file:** the record of the **first complete draft** of all four songs and **its measurement before any repair.** The repaired text ships in `pair_02_step10_final_package_enhanced.md`; this file exists so the repair is auditable rather than asserted.

---

## WHAT WAS DRAFTED

Four complete songs, written in one pass against the form declared at step 06 (**THE READ-BACK**), with the refused element (⛔ no `we / us / our / they / them / their`) enforced while writing rather than checked afterwards.

| | title | BPM · key | sung lines drafted |
|---|---|---|---|
| **V1** | WITH ME INSIDE IT | 104 · G Dorian | 85 |
| **V2** | NOBODY IN THE CHAIR | 100 · F minor | 80 |
| **V3** | THERE IS A FIFTH | 108 · E♭ major | 79 |
| **V4** | IT FALLS ON ANYONE | 102 · C minor | 85 |

Draft held at `scratchpad/p02_lyrics_draft.md`; measured with `scratchpad/p02_measure.py`, which imports `scripts/measure_soundcraft.py` and calls its own `profile()` on the fenced blocks.

---

## ⭐ THE FIRST-DRAFT MEASUREMENT — the numbers before any repair

**Command:** `python p02_measure.py p02_lyrics_draft.md`

| | field chars | sung lines | endRhy | lineRet | w/line | allit/100w | banned pronouns | emotion words |
|---|---|---|---|---|---|---|---|---|
| **V1** | 4305 | 85 | **0.600** | 0.482 | 5.64 | ⚠️ **7.52** | 0 | ⚠️ **1** (`loud`) |
| **V2** | 4360 | 80 | 0.412 | 0.475 | 6.45 | ⚠️ 9.11 | 0 | 0 |
| **V3** | 4238 | 79 | ⚠️ **0.291** | 0.544 | 6.05 | ⚠️ 9.00 | 0 | ⚠️ **1** (`loud`) |
| **V4** | 4622 | 85 | 0.412 | 0.388 | 6.18 | ⚠️ **7.24** | 0 | 0 |

**Floors, from `vault/gates.yaml`:** `rhyme_return_floor: 0.30` · `line_return_floor: 0.20` · `mean_words_per_line_ceiling: 7.5` · `alliteration_per_100w_floor: 11.0`.

### ⭐ THE THREE DEFECTS THE INSTRUMENT FOUND, AND ONE OF THEM I WOULD HAVE SHIPPED

1. ⛔ **`loud` in two sung lines** — *"I read it three times out loud"* (V1) and *"I am saying that out loud on purpose"* (V3). **R2 forbids `loud` by name.** ⚠️ **I would have argued this one:** *out loud* is idiomatic for *audibly* and is not naming a feeling. **That argument is exactly the shape of the error `a-self-check-that-lowers-your-own-defect-count-needs-a-control` describes** — a verifier who is also the author, reducing a defect they own. So it is repaired rather than defended, and `aloud` is better writing besides.
2. ⚠️ **V3's strict end rhyme sat at 0.291, under the 0.30 floor.** The chorus was carrying only 3 of 6 lines.
3. ⚠️ **All four sat under the alliteration floor**, V1 and V4 badly (7.52 and 7.24 against 11.0). **The archive winners measure 14.2 per 100 words and texture was never our problem before** — this draft's problem was that its lines are short, plain and procedural, which is right for the register and starves the consonant clustering.

⭐ **What the instrument did NOT find, and it matters:** **zero banned pronouns in 329 sung lines, first pass, no repair needed.** The refused element was enforceable at the point of writing, which is the whole argument for choosing a grammatical prohibition over a stylistic one.

---

## ⭐ CROSS-VARIATION DUPLICATION, MEASURED ON THE FIRST DRAFT

**0 sung lines shared between any two of the four. 0.0% across all six pairings.**

> Last run **P02 disobeyed the section spec and shipped 0.0%; P03 obeyed and shipped 34.7%** — 25 of 72 lines identical across all four songs, chorus included. `core_seed.md` §2: *the tighter the specification, the tighter the template.* This run's brief specified no form at all, and the first draft came out at 0.0% without a de-duplication pass.

---

## ⛔ WHAT WAS NOT DRAFTED HERE, DELIBERATELY

- **The music prompts and exclude fields were drafted at step 09**, after the lyrics were fixed. ⭐ **THE GRAIN LAW: a Somatic or distinctiveness objection answered in the PRODUCTION SPEC is not answered.** Writing the spec first invites the spec to carry the discriminator, and the discriminator has to live in the lyric and the form.
- **Nothing was rendered.** `vault/AUTONOMY.md` holds: drafts on disk, nothing rendered, published, or spent.
