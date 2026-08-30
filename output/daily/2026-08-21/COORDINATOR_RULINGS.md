# COORDINATOR RULINGS — `daily-music-2026-08-21`
## Questions escalated by the step-11 tiers, resolved **in writing, in the run directory**

> **L30, HIGH:** *"When doctrine and a step file conflict, the conflict must be resolved IN WRITING in the
> run directory, not left to each agent's judgement. Unanimity among five agents is not evidence the rule
> was stated — it is evidence five agents happened to reason the same way."*
>
> Three tiers reached the same two questions independently and **all three escalated instead of guessing.**
> That is the behaviour the Andon Cord is for, and it means the rule was never stated. It is stated here.

---

## R-1 · ⭐ INSTRUMENT AND EFFECT NAMES ARE **TECHNIQUE**, NOT PEOPLE — they may enter a generator field

**Raised by:** Pair 05 (`autotune` in all four EXCLUDE fields), Pair 02 (same, plus it *removed* `Rhodes`
from a style field as a brand name). Both disclosed rather than buried. Both were right to ask.

### The ruling

> ⛔ **The ban is on names whose CREATIVE LABOUR would be imitated — people, groups, artists, and
> recordings.** It is **not** a ban on the standard word for a sound.
>
> ✅ **PERMITTED in a generator field:** `Rhodes` · `Wurlitzer` · `Hammond` · `Mellotron` · `Moog` ·
> `808` · `Auto-Tune` · `Juno` · `Leslie` — and every living-scene **idiom**: `gasba` · `funaná` · `raï` ·
> `tamborzão` · `canntaireachd` · `Memphis half-time`.
> ⛔ **FORBIDDEN in a generator field:** any real person, group, band, label, or recording — including the
> eighteen panel seats' source figures and every player named in a dedication.

### Why the line falls there

The doctrine states its own purpose: ***"a name in the prompt is an instruction to imitate; a name in the
dedication is a debt paid. Crediting a player honours them; feeding their name to the generator uses
them."*** The harm is **appropriated labour**. A tine piano has no labour to appropriate; *"Rhodes"* is
simply the word for that sound, and the template's own guidance says **"use colloquial instrument names."**
The nearest circumlocution (*"tine-based electric piano"*) is worse for the generator and **no more ethical
— it protects nobody.** ⭐ **A rule that forbids naming an instrument is not stricter, it is just less
accurate.**

`Auto-Tune` is the sharpest case and it resolves the same way twice over: it is genericised, it names an
**effect**, and ⭐ **Pair 06's assignment REQUIRES it** — its Source-2 vocabulary is verbatim *"vocals
slathered in it to the point where they sound almost alien"*, and the effect **is the character, not a
correction.** A reading that bans the word would delete a pair's core spec to protect a trademark holder
who is not the party this standard exists to protect.

### Consequences

- **Pair 05:** `autotune` **stays** in the EXCLUDE fields. Ruling: clean. No edit.
- **Pair 02:** removing `Rhodes` was **over-correction, but it is not a defect** — the tier substituted a
  mechanic and its gates measured clean. ⛔ **Leave it.** Moving a passing gate on a reversed ruling is the
  exact regression this tier is warned against. The ruling is recorded so its MAJOR DEVIATIONS reads true.
- **Pair 03:** keeps `Rhodes`, which is load-bearing — its rupture *is* a Rhodes handing its figure to an
  upright. ✅ Explicitly permitted.
- ⚠️ **The inconsistency across pairs is mine**, not theirs: two tiers reached opposite answers because the
  contract named "product" in a ban whose stated purpose is about people. **Recommend the handoff template
  drop "product/model" from that clause and say what it means: no person, group, band, label or recording.**

---

## R-2 · ⛔ THE CREDIT DEBT — I GOT IT WRONG TWICE, AND A TIER CAUGHT IT

**Raised by:** Pair 01's step-11 tier, which found `gasba` in all four of its style fields against a
dispatch that told it **no Lineage & Credit block was owed.** It kept the idiom, **refused to invent a
credit**, and escalated.

### The ruling

> ⭐ **A technique raised in the STYLE field owes its scene and its players. The same technique banned in
> EXCLUDE owes nobody.** My dispatch was **wrong**, and it was wrong in the precise way the standard names:
>
> *"P06's first draft of all four dedications said 'no named figure is asked for, so no player is
> credited' — **and that was false, while reading exactly like diligence.** An over-credit is visible and
> embarrassing; **an under-credit reads as a completed check.**"*

### The sweep, and the second instance

A **credit-debt sweep** now exists (`credit_debt_sweep.py`) — it did not before. The repo had a sweep for
**named players and named breaks** (`amen` → Gregory C. Coleman / The Winstons) but **none for the scene
case**, which is commoner and is the one a coordinator gets wrong by omission. Its first run found **two**
unpaid debts, both created by my dispatch:

| pair | idiom raised in style fields | dedication | disposition |
|---|---|---|---|
| **P01** | `gasba` (Western Algeria) — the pair's assigned **ONE SEAM**, structural | **none** | ⛔ **DEBT. Fix is the CREDIT, not the deletion** — the seam is fixed at Phase 1. Tier instructed to add four ≤500-char blocks; nothing else touched. |
| **P04** | `dub-poetry` | **none** | ⛔ **DEBT.** Ruling pending its tier's return. |
| P02 | `funaná` | yes | ✅ paid |
| P06 | `gasba` | yes | ✅ paid |
| P05 | (none — the tier *dropped* two scene labels it did not structurally need) | n/a | ✅ **the other legitimate resolution.** Drop the idiom, or pay the credit. Never keep it silently. |
| P03 | (none raised) | yes | ✅ over-credit, harmless — but checked that it is not a **false** credit |

⭐ **Pair 02 supplied the case nobody had a rule for: a credit that was TRUE OF NOTHING.** All four of its
step-10 dedications credited `funaná` and **not one style field had asked for it** — V4 had asked for a
psychedelic balkan-beat squeezebox, a different living scene entirely. **A dedication that credits a scene
the track never raises is a false credit, and a false credit makes the true ones worthless.** The tier took
the *fix-the-prompt* branch (correctly: the run's Source 2 **is** funaná) so the credit became true.

---

## R-3 · ⚠️ CONCURRENT SCRATCHPAD COLLISION — a real defect, and it is mine

**Reported by Pair 02:** its `measure.py` was **overwritten mid-run** by another tier's harness in the
shared scratchpad. ⛔ ***"An instrument that can be replaced under you while still returning numbers is the
'gate that cannot fail' class."*** It caught the swap only because the output **schema** did not match the
script it had written, then re-verified with a uniquely-named reader.

**Ruling: my dispatch defect.** `EXECUTION.md` §2's serialized-write discipline covers the **run directory**
(*"every fan-out subagent writes ONLY into its own `pair_{NN}_*` namespace"*) and says nothing about
**scratch space** — so six concurrent tiers wrote helper scripts into one shared directory under generic
names. **The namespace rule must extend to scratch files: `p{NN}_*` prefixes, mandatory.** Recommend
promotion into `EXECUTION.md` §2.

⭐ **The accident produced a genuine control:** two independently written readers, one of them a stranger's,
**agreed exactly on all four of Pair 02's packages.** That is a stronger cross-check than either would have
been alone — but it was luck, and luck is not a method.
