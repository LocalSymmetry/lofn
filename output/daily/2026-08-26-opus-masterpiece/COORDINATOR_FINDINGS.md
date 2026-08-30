# COORDINATOR FINDINGS — instrument defects and one real seed gap

*Written at the join, before QA. Every item here was found by a check firing, not by review.*

---

## ⭐ FINDING 1 — THE SEED HAD NO PERMISSION CLAUSE, AND IT COST P01 A REAL CONSTRAINT CONFLICT

**Measured:** `grep -ci "permission|may break|allowed to break|you may" 02_golden_seed.md` → **0**.

`lofn/SKILL.md` mandates the ordering **Golden Seed → permission (what it may break) → songmaking → QA contract last.** My Phase-0 seed carried the seed, a `WHAT THIS IS NOT` section (six prohibitions), and the QA contract. **It never told a pair what it was allowed to break.**

**The consequence is not hypothetical.** P01's assigned architecture (`answer-first couplets`, stanza lengths 8/8/6) mechanically yields **44–52 sung lines against a 70-line floor.** With no permission clause, the pair could not tell whether the architecture or the floor was the softer constraint. It resolved this correctly and honestly — added a four-line tag after each chorus plus a fourth chorus, and **declared it as Major Deviation 1** with a structural defence (A2 mandates a byte-identical chorus, so the chorus *cannot* carry R7's asymmetric return; it needs a separate object, and the tag is it).

Its own filed risk names the ambiguity precisely: *"A strict reading of 'the architecture is not renegotiable' could still call this out of scope."*

⭐ **And this explains the floor-hugging FLAG I raised earlier.** P01 sits at **exactly 70 sung lines × 4** — I first read that as gate-optimisation. It is not. **It is a pair meeting a floor my own brief made unreachable, by the smallest honest addition it could defend.** The FLAG is real; the blame is mine.

> **The transferable rule: a brief with six prohibitions and zero permissions forces every conflict to be resolved as a deviation.** A permission clause is not a loosening — it is the instruction that tells a pair which constraint yields when two of mine collide. **Without it, the pipeline cannot distinguish a considered trade from a violation, and neither can QA.**

⛔ **Not retro-fixed.** `02_golden_seed.md`, `04_metaprompt.md` and `05_pair_assignments.md` are **embedded verbatim inside the frozen 87,216-byte ICB** (verified). Editing them now would make the standalone files disagree with the copy all 24 songs were actually written against, and would silently invalidate the substring-integrity proof that is this run's main evidence of continuity. **The frozen block is the contract the work was made under.** This finding is the correct artifact; a retro-edit would be a forgery.

---

## FINDING 2 — MY OWN RE-STAT GATE PASSED A PAIR ON ZERO MEASUREMENTS

`coord_restat_pair.py` extracted **0 MUSIC PROMPT and 0 EXCLUDE blocks from P05** and returned **PASS**, because the per-block loops simply did not execute and the `ok` flag was never set false.

> **A gate that returns PASS when it measures nothing is not a gate; it is a sentence.**

Two causes, both mine: the extractor required a code fence (P05 and P04 shipped their fields unfenced), and there was no cardinality assertion. **Fixed:** the extractor now falls back to the unfenced body so an unfenced field is *measured* rather than skipped, and the gate now **fails closed** unless it extracts exactly 4 of each field type per pair. Re-run: 24/24 prompts, 24/24 excludes, 24/24 lyrics extracted across six pairs.

## FINDING 3 — THE SAME GATE FAILED A CORRECT PAIR

Before the fix, the extractor matched a prose heading `## 1b. What step 11 changed` as an exclude field and reported a phantom **1,519-char out-of-band FAIL on P03**. P03 was correct throughout. Caught by inspecting the instrument before believing its verdict; had I believed it, the repair an agent reaches for is to cut a correct exclude field down to size.

## FINDING 4 — I REPORTED A FIX THAT HAD NOT LANDED

I printed *"gate hardened"* from an edit script's own success message while the insertion had silently failed to match. Caught by `grep`-ing the target file rather than trusting the edit's output. **An edit script's report is a claim about the edit, not about the file.**

---

## FINDING 5 — ⛔ A STALE RUN-SCOPED CONSTANT IN A SHARED VALIDATOR WOULD HAVE CONDEMNED CORRECT WORK

**Found by P02, which refused to route around it and did not edit the shared script.**

`scripts/restat_pair_packages.py:54` carried:

```python
FORBIDDEN_SUNG = [r"\bfourteen\b", r"\bsix hundred and fifty[- ]eight\b", ...]
```

entered at commit `cb3956e0` and belonging to an entirely different run. **P02 of this run is MANDATED by its step-05 isolation brief to sing "fourteen"** as its one-fact hinge — it appears **18 times** in the shipped package. The scanner would have reported `forbidden_sung_number` FAIL ×4 on a passing pair, and **the repair an agent reaches for is to delete the mandated number to appease the scanner.**

This is the **fifth** stale constant of this exact class in that one file; three of the others (`ANCHOR_STEMS`, `MANDATED_SHARED`, `APOLOGY`) already carry comments documenting the same failure. **Fixed:** emptied, with the file's own established run-scoped convention and the evidence recorded inline.

> **A constant from another run either lies in the report or condemns correct work.**

## FINDING 6 — TWO VALIDATORS ENCODE OBSOLETE DIALECTS

- **`validate_orchestrator_packet.py`** hardcoded `03_orchestrator_panel_debate.md` / `04_orchestrator_metaprompt.md` / `05_orchestrator_pair_assignments.md` / `06_audio_handoff.md`, contradicting `lofn/SKILL.md` Phase 1 step 6, which saves `03_panel_debate.md` / `04_metaprompt.md` / `05_pair_assignments.md` / `06_<modality>_handoff.md`. **Fixed instrument-side** with an alias resolver accepting both; the artifacts were not renamed to satisfy a stale reader.
  ⚠️ **Residual, not fixed:** it also checks literal marker strings (`pair 01`, `active personality`, `golden seed`) that this run satisfies in substance but not verbatim (`P01`, `LOFN-PRIME (oscillating)`). **Those were left alone deliberately** — sprinkling literal words into frozen artifacts to turn a report green is gate-appeasement, and the frozen files cannot be edited anyway.
- **`validate_portfolio_distinctiveness.py`** returned, honestly: *"extraction returned empty text — the parser did not match this run's heading convention. **This is a validator defect, NOT a distinctiveness result.**"* Left as-is; it fails loudly and correctly. Cross-pair distinctiveness was measured independently instead — see the QA report.

**This is the same watch item the ledger already carries from 2026-08-25:** *"Legacy distinctiveness entrypoints could not recognize the current heading/filename dialects."* It has now recurred on a third run.

---

## FINDING 7 — TWO PAIRS SHIPPED UNFENCED SUNO FIELDS (repaired by the pairs, not by me)

P04 and P05 emitted their four MUSIC PROMPT and four EXCLUDE PROMPT payloads as bare paragraphs while correctly fencing their lyrics. On a dense ~950-char prose prompt an unfenced field has no unambiguous end, and the paste-ready package is the deliverable.

**Repaired by sending each pair back to its own artifact** (`AGENT FIX, DON'T BYPASS`), with instructions to change zero characters. Both returned arithmetic proof: file growth of **exactly +96 bytes = 8 fences × 12 delimiter chars**, every payload byte-identical, ICB still exact at byte 0. **72/72 Suno fields now fenced.**

---

## FINDING 8 — A SUPPLIED PALETTE IS NOT A CLEARED PALETTE (found by P01)

Two real-name violations reached a pair **through the ICB's own seed genre palette**, which I sampled mechanically from `vault/genres.txt`: **`Industrial Sega`** (a company) and **`Cosmic P-Funk`** (a group). A proper-noun heuristic clears both; only a roster scan catches them.

P01 also measured the heuristic itself: **80 flags, zero true positives** across its twelve fields (sentence-initial words, EMO tags, contractions, weekdays).

> ⭐ **The palette I hand a pair is generator-facing input and it has never been screened.** The no-real-names rule is enforced at the pair's output; nothing enforces it at the coordinator's input. Both names were caught by pairs and kept out of every Suno field — but that is six independent saves, not a control.

---

## WHAT THIS SET OF FINDINGS HAS IN COMMON

Six of the eight are **instrument defects, not creative defects** — and four of those six would have damaged correct work had they been believed. Three were mine.

> **Every gate that fired this run was checked before it was believed. The two that were wrong were wrong in opposite directions: one condemned correct work, one passed unmeasured work.** The second is the more dangerous, and it was mine.

---

## ⭐ FINDING 9 — MY QUARANTINE ASSERTION CHECKED THE NAME, NOT THE CONTENT — AND I EDITED A FROZEN SOURCE AFTER THE FREEZE

Two defects, both mine, found by a routine integrity re-check rather than by any gate.

**9a. A soft golden-song reference survived the quarantine assertion and IS in the frozen ICB.**
`coord_build_icb.py` asserts `"Triple Arch" not in ICB`. It passed. But the metaprompt's world paragraph read *"solar wind at 272 km/s (near the floor; **our own archived winner sang 510**)"* — a **content** reference to a golden song with the **name** removed. My assertion checked the name and was therefore satisfied by the exact edit that should have failed it.

> **A quarantine that greps for titles does not detect exemplar leakage. Titles are the easiest part to strip and the least contaminating part to leave.**

⭐ **Measured harm: zero.** Every one of the six authored suffixes was scanned for `510` and `archived winner`: **0 hits in all six.** No pair echoed it. The cut ledger had independently rejected the 272-vs-510 callback as self-copying, which is likely why. **The control that worked was the one written into the creative brief, not the one written into the build script.**

**9b. I edited a frozen source after the freeze, creating two disagreeing truths on disk.**
I removed that reference from `04_metaprompt.md` **after** the final ICB build and never rebuilt. Result: the standalone file said *"near the floor of the observed range"* while the ICB — the copy all 24 songs were actually written against — still said *"our own archived winner sang 510"*. Undetected for the entire pair wave; found only by re-running a source-vs-ICB comparison at session resume.

**Resolved by reverting the standalone to the frozen copy**, not by rebuilding the ICB. Rebuilding would have invalidated the byte-0 prefix proof in all six packages — the run's principal evidence of continuity. **The frozen block is the contract; a source that disagrees with it is the thing that is wrong.**

Verified after the revert: `02_golden_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `00_research_brief.md` all embedded verbatim; `03_panel_debate.md`'s four required slices present; **all six packages still carry the 87,216-byte ICB exactly at byte 0.**

> **The transferable rule: after a block is frozen, its sources are frozen too. There is no such thing as a harmless edit to an artifact that has already been copied — there is only a divergence you have not measured yet.**


---

# PART II — FINDINGS 10-15

The clean-context judge landed after this document was written. Its six claims were re-measured from disk; **four hold, two do not.** Recorded separately in **[EVALUATION_VERIFICATION.md](EVALUATION_VERIFICATION.md)** so the verification sits beside the claim rather than inside it.

- **F10** R3 body-units violated in all four P03 variations, including the selected one — routed back.
- **F11** Moog product names inside two pasted P02 lyrics fields — routed back.
- **F12** I specified a variation angle my own cut ledger had already disqualified (P04 V3).
- **F13** No tempo axis; the accessible arm collided at 118 BPM x3, 102 BPM x2.
- **F14** The missing permission clause is portfolio-wide: **five** constraint collisions across **four** pairs, not the one F1 recorded.
- **F15** Honest cardinality is six songs and eighteen alternates; pair isolation binds across pairs and is silent within them.
- **CORRECTED** the cut-ledger moon leak is 2/24, not the 4/24 reported.
- **REJECTED** "fourteen sung 6x" is one fact used as a title hook, not recitation.
