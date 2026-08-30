# PAIR 06 — STEP 10 · REVISION SYNTHESIS · **SUNNA R5**

*What changed between the draft (`step08`) / the production pass (`step09`) and the shipped package, and which instrument found each one. ⛔ Every number below was produced by a command, not by reading.*

---

## 1. THE INSTRUMENTS, AND WHAT EACH ONE CAUGHT

`p06_measure.py` extracts the three pasted fields **exactly as `scripts/check_field_census.py` does** — the fenced block under a heading whose text contains `MUSIC PROMPT` / `EXCLUDE` / `LYRIC` — so my count and the coordinator's cannot drift by measuring different objects. That parser choice is itself a lesson borrowed from this run: four of the coordinator's scanners failed in one day by parsing markdown *headings*, whose dialect varies per pair.

| instrument | finding | fix |
|---|---|---|
| **char count, music prompt** | four prompts at **1034–1079** — every one over the 1000 hard cap and far over the 870–960 target | trimmed to **952–962**. ⭐ **No clause of meaning was removed, only words**: every kinetic defect, every stop-once event, every late-lead notation and every acoustic declaration survives |
| **char count, lyrics field** | four fields at **5037–5282** — three over the **5000 render limit**, all four over the 4800 target | section order and mix notes moved to a **`## Production Sidecar` outside the fields** (the handoff's own remedy); `[SONG FORM:]` lines, `Disc_` channels and EMO headers tightened. Final **4641 / 4790 / 4748 / 4783**. ⭐ **Not one sung line was cut to make room** — the cap was paid for out of metadata, which is what metadata is for |
| **armour-word regex** | `diamond-doubled` in V1's music prompt. *Diamond* is on forbidden-default #10's list | → `double-tracked`. It is a real production term and Sunna's own tech notes use it — **and my declared gate said zero.** A gate you argue with is not a gate |
| **shared-line check** | ⭐ **`Up.` at both V1's and V2's peak** — the pre-registered metric found a genuine collision at the two places a collision costs most | V2's became *"Off the floor."* **Pre-registered value reported at 1.3%; final measured 0.0% across all six pairings** |
| **body-word count** | V4 **28.9%**, V3 **21.8%**, V1 **20.3%** raw — V4 above the 22% body-noise failure signature | eight decorative body words cut (*thumbnails · shoulder · knee-print · both hands · my nails · the side of my face · my fingers · knuckle*). Final **11.4 / 9.0 / 15.4 / 19.7%**, still over the norm, **reported in `## Major Deviations` D11 rather than buried** |
| **token census (mine, then the coordinator's)** | `autotune` in two exclude fields | ⭐ **Not a violation** — `vault/field_token_lexicon.yaml` permits trade names as *negative* tokens in the exclude field only. Declared so nobody has to re-derive the ruling. `scripts/check_field_census.py` reports **P06: clean, 12 pasted fields** |
| **one-fact rule, manual** | five stray numerals across the pair competing with the hinge number | removed; **one sung numeric fact per song — nine · two · a quarter · eleven —** all spelled in words |
| **the STRANGER TEST, manual** | V2 opened on a roll-up **door**; V4 exited through a **gate** | ⛔ **doors removed from all four.** The wire's one concrete object is a door and the rule is *repair it, do not argue about it* |

---

## 2. ⭐⭐ THE PRE-REGISTRATION, SCORED — INCLUDING THE ONE THAT DIED

*Filed in `pair_06_step06_facets.md` before a sung line existed. Reported here as written, whichever way it went.*

| | prediction | measured | verdict |
|---|---|---|---|
| **F1** | ≤2 substitutable lines per song *(expected 3–6 on draft)* | 0 · 0 · 1 · 0 | ✅ **HELD** — and my stated expectation of a messy first draft was wrong |
| **F2** | V2 future ≥ **2.0×** past *(flagged as the closest call)* | **0.67×** (16 vs 24, n=78) | ⛔ **FALSIFIED, by a factor of three** |
| **F3** | first capability line falls after the object moves, 4/4 | 4/4 *(one repair in V1)* | ✅ **HELD** |
| **F4** | century audit 4/4 *(V1 expected to fail first)* | 4/4, V1 included | ✅ **HELD** — my named worry was the wrong worry |
| **shared lines** | 0–5%, *"carried by the word* goes *rather than by whole lines"* | **1.3% → 0.0%** | ✅ **HELD**, and the mechanism was as predicted |

### ⛔ F2 IS THE ONLY ONE THAT MATTERS, AND IT COST NOTHING TO ACCEPT

**That is the whole argument for the standard.** F2 was written down before V2 existed, with its number and its falsification condition; when the count came back at a third of the floor there was nothing to defend, because I had already said what would count as being wrong.

> ⭐ **WHAT SURVIVES, IN NARROWER FORM.** *Name the unit before you measure.* My predicate was **grievance**; my instrument measured **past tense**, which is a proxy for it. And the proxy is actively backwards here: the entire `[The Card]` section — the passage that does the anti-resentment work — is in the past tense, and *"Nobody hid it"* is a past clause whose only job is to delete the villain. **Counting past tense counts the anti-grievance too.**
>
> **THE NARROWER CLAIM, AND IT IS BETTER DOCTRINE THAN THE ONE THAT DIED: resentment is not a TENSE, it is an AGENCY.** Count past clauses whose **subject is an agent who acted on the singer.** In V2 that count is **zero of 24** — every past clause's subject is the card, the singer herself, or *everybody, including her.*

⚠️ **AND THE LIMIT ON THAT, STATED BY ME ABOUT ME.** I am the author of the song **and** of the instrument, and my correction happens to reduce a defect I own. That is the exact direction-of-error the self-check standard names: *a verifier who is also the author has a DIRECTION*, and on 2026-08-26 two of two such overturns were wrong. So: **the falsified number stays at the top of the verification table with no ✅ next to it**, the narrower claim is offered as a *hypothesis for the next run* rather than as a repair of this one, and **R7's P4 hands the question to a listener with no stake in my answer.** A grader who is also the author does not get to close this.

---

## 3. ⭐ WHAT DID NOT CHANGE, AND WHY THAT IS THE POINT

- **Not one sung line was cut for a character cap.** Three lyrics fields were over a hard render limit and the cap was paid entirely out of `[SONG FORM:]` prose, `Disc_` channels and EMO headers. ⛔ **A render limit must never be allowed to edit the song** — that is `06_music_handoff`'s own instruction and it is the right order of operations.
- **The four forms.** Call-and-answer · the growing inventory · the returning block with one line replaced · the last chorus nobody sings. **They are why the shared-line rate is 0.0%**, and they were authored rather than received, which `core_seed.md` §2 measured as the difference between 0.0% and 34.7%.
- **The late lead.** It survives in both notations in all four songs. ⚠️ It is a **probe** — 57 territories UNEXPLORED, 0 probes, no measured claim about vocal-timing notation — and R7's P3 says out loud that **I want at least one listener to hear it as a mistake**, because a deliberate late entry nobody notices is a technique that did not render.
- **V4's missing last chorus.** The one place I overruled the centre outright. It stayed through every revision pass, and **P6 is the falsification I would most like to be cheap**: if listeners report it as a broken file rather than as an ending, D1 was wrong and I spent an overrule on nothing.

---

## 4. HANDOFF NOTES FOR `07`

- ⭐ **Re-measure the AWE row against the shipped EMO headers, not against `05_pair_assignments.md` and not against my paragraph.** The tokens are *Revelation · Eureka · Amazement · Astonishment · Marvel · Wonder · Incredulity · Startlement · Fascination*, **8 / 5 / 8 / 5 across the four, zero bare `AWE`.** Yesterday a portfolio rule was declared in a ledger, never re-measured, and shipped unsatisfied.
- **The body-word rate is over the house norm and is declared in D11 with both units.** It is the pair's one open quality question and it should be judged, not re-discovered.
- **D9** declares the V2 reading of the one-fact rule *(the sung numeric fact is `two`; `one` is its answer)* so the census does not have to guess it.
- ⛔ **`vault/AUTONOMY.md` holds: drafts on disk. Nothing rendered, published or spent.** A text SHIP is permission to render, never permission to publish.
