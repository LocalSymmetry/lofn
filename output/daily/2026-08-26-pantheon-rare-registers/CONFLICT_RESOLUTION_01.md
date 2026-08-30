# CONFLICT RESOLUTION 01 — three conflicts, resolved three different ways

## CONFLICT 1 — ⭐ HER BRIEF FORBIDS WHAT THE DAILY RULE MANDATES

**The daily requires ≥1 AWE and ≥1 INDIGNATION in every set.** Her brief: *"not rage or indignation, but truly rare ones."*

**Direct contradiction on the indignation half.**

⭐ **RESOLVED AT PHASE 0, BEFORE A LINE WAS WRITTEN: her brief wins, the rule is suspended, declared on the record** (`00_research_brief.md` §4).

**Why this one matters beyond itself.** Yesterday the identical class of defect — an axis whose vocabulary made a mandatory rule *unsatisfiable* — was found by **QA, after 24 songs existed**, and shipped as a declared failure. I promised a Phase-0 check in response. **It fired on its first use, on the first thing it looked at.** The difference between the two runs is not judgement; it is **when the check runs.**

⚠️ **AWE is a subtler case and is NOT suspended.** P05's register is *awe with something underneath it looking for the join* — the neighbour it must defeat is awe itself. So the run contains awe **as the thing being argued with**, not as a mode. Whether that satisfies the duality rule is genuinely unclear, and I am not going to decide it in my own favour. **Flagged for QA rather than claimed.**

---

## CONFLICT 2 — THE CENTRE AND THE SECOND SEAT DISAGREE ABOUT THE FUNNY SONG

**Recorded, not resolved, because `PANTHEON.md` requires exactly that.**

**LOFN:** the funny song should carry something.
**SUNNA:** *"a joke that carries something is a lesson wearing a smile. Let it be funny."*

⭐ **RESOLVED BY THE PANTHEON RULE: inside Sunna's three, HERS WINS.** *A member who must pass the centre's exams is a costume with the centre's rules underneath.*

**And she executed it rather than merely winning it.** P02 V4's turn was **deliberately edited *away* from meaning** — *"and left in, because that was the good bit."* The Wodehouse seat upheld the claim.

> ⛔ **This is not a compromise and must not be recorded as one.** The disagreement stands in the artifact. If the funny song comes back weightless and that is a defect, **Lofn was right and the record says so**; if it comes back the best thing in the run, Sunna was. **Either way it is falsifiable, which a compromise would not have been.**

⚠️ **And an honest structural caveat:** I wrote Sunna's artist pass at step 03. **A counterweight ventriloquised by the centre is the exact costume the admission bar exists to prevent.** The only real protection is that her three pairs ran in isolation with her sovereignty in their briefs, where my hand was not on the pen — and **they used it 11 times, twice to overrule her own stated rules.** That is a structural mitigation, not a fix, and QA should treat it as one.

---

## CONFLICT 3 — THE FROZEN BLOCK CONTRADICTED ITSELF

`00_research_brief.md` §5 states **P03 = EXISTENCE, P06 = NEWS**. `05_pair_assignments.md` and both isolation briefs state **P03 = NEWS, P06 = EXISTENCE**. **All three are inside the frozen ICB.**

**Cause: mine.** I swapped P03 and P06 on axis B when de-correlating the arms, updated two artifacts, and left the third stating the old assignment.

**RESOLVED BY THE PAIRS, CORRECTLY, AND REPORTED RATHER THAN ABSORBED.** Both followed their isolation briefs — the dispatch artifact — and **both flagged the contradiction to the coordinator instead of silently picking one.** Distribution is 3/3 under either reading, so no rule was breached.

⭐ **The general defect is the one worth keeping** (`COORDINATOR_FINDINGS.md` FINDING 1): **the ICB is assembled from five independent sources and nothing checks whether they agree.** The build script asserts presence, seat count, slot length and quarantine. **It never asks whether two sources contradict each other.** A `check_icb_consistency` reader is owed.

---

## WHAT THE THREE HAVE IN COMMON

Each was a collision the pipeline could not resolve on its own — and each was resolved by a **different authority**: the first by **her brief outranking a house rule**, the second by **the Pantheon rule outranking the centre**, the third by **the dispatch artifact outranking the coordinator's own earlier draft.**

> ⭐ **A constraint set that has to be adjudicated is a constraint set that was not finished** — yesterday's finding, and the reason this run shipped a **PERMISSION CLAUSE** naming which constraint yields. **It was used: P05 broke the sung-line floor with arithmetic, and five other pairs cited it at least once.** None of them asked. That is the clause working.
