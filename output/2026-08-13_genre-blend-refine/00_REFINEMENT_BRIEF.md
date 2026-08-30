# 00 — GENRE-BLEND REFINEMENT BRIEF · 2026-08-13
## Second refinement pass on the top six of `2026-08-12_genz-worst-i-will-ever-be`

**Commissioned by The Scientist:** *"I love these, but I think we can go one step further. For example,
TAKE A HUNDRED is amazing but feels dated as we didn't modernize the 60s girl group. A ska/glitch/trap/flow
rap addition would hit hard. Can we try a second refinement run on these where we smartly genre-blend."*

**Source run:** `output/daily/2026-08-12/` — QA REPAIR→closed, 24/24 measured, 0 hard fails.
**Scope:** the **six selected songs only**, refined individually. The other eighteen are untouched.
**Lock:** `output/2026-08-13_genre-blend-refine/.RUN_LOCK`, `claude:3c56afcb-…`, one controller.

---

## 1. ⭐ THE DIAGNOSIS — sharper than "add more genres"

Her example is the tell. P06's assigned bed was **"power-pop / sixties girl-group over a jungle break"** —
a *historical* bed named as the **dominant** and the modern element demoted to a subordinate clause.

**THE GRAIN LAW says the generator smooths toward its strongest prior.** For "sixties girl group" that prior
is **straight pastiche**. The jungle break was supposed to modernize it and never became the spine, because
nothing in the prompt made it the frame. **The song is dated because the coordinator's bed assignment was
dated, not because the pair executed badly.**

### ⭐⭐ THE INVERSION — the rule for this pass

> **The MODERN genre is the WORLD. The HISTORICAL element is the INTRUSION. ONE SEAM.**

Girl-group thirds stop being the bed and become **the wound** — a stack of 1963 harmony arriving *inside* a
jungle frame, at one named moment. This is **L38's good half applied to music**: *keep the historical-medium
anchor, drop the collision.* It is the difference between **retro** and **now**.

**Operationally, in every prompt:**
1. **The modern genre is named FIRST and as the frame** — first clause, with tempo and drum programming.
2. **The historical element is named as an INTRUSION at a NAMED MOMENT** — "…and at the third chorus a
   stack of sixties girl-group thirds arrives on top of it, unaccompanied" — never as the bed.
3. **ONE seam.** Not four genres in a blender. **N = 1 world + 1 intrusion + 1 seam**, and the singer must be
   *doing something across it.*

---

## 2. WHY THIS IS DOCTRINALLY FREE (checked, not assumed)

- **L38 does not apply.** Its own scope line reads `⚠️ Scope: venue=NightCafe, modality=image`, and it calls
  a music parallel **"flagged, not claimed."** There is no conflict to manufacture.
- ⭐ **SUNNA's DNA MANDATES IT.** Pillar 3, **THE GEAR-CHANGE**: *"style-shifts **inside** a single song:
  punk into bounce into dub and back, with rolling drum transitions. **One song that is three songs.**"*
  This pass is her file being obeyed, not stretched.
- ⭐ **THE FREE CHANNEL makes it cost nothing.** *"Put the complexity in the MUSIC and the feeling in the
  WORDS."* Musical complexity is processed **below conscious effort**; the listener cannot help receiving it.
  **Genre-blending is spending in the free channel — the one kind of depth that charges no toll.**
- **The variety-across-a-set rule still holds:** every song gets **its own** blend, derived from its own
  material. ⛔ **No global recipe.** Her four (ska / glitch / trap / flow-rap) are the **palette, not the
  formula** — the same error as a shared V1–V4 template, which this pipeline already caught once this week.

---

## 3. ⛔ DEFECT FOUND WHILE PLANNING — and this pass fixes it

**KEPT, NOT PLAYED (P04V1) and ROOM TONE (P04V3) share a genre bed** (trip-hop revival × neo-baroque).
`sunna.yaml`'s Anger Audit rule: ***no two songs in a set share a bed.*** The final six violates it.
**Fixed here by diverging KEPT, NOT PLAYED onto pluggnb** rather than by dropping a song.

---

## 4. THE SIX ASSIGNMENTS — each derived from its own material

| # | Song | Voice | Current bed | ⭐ WORLD (modern frame, named first) | INTRUSION (historical, one named moment) | Flow/rap |
|---|---|---|---|---|---|---|
| **A1** | **THE FIXER** | LOFN · INDIGNATION | glitch-baroque × breakcore | **already correctly inverted** — breakcore world, baroque intrusion. Modernize the **DELIVERY**, not the bed: half-time trap flip at the hinge | the harpsichord passacaglia (unchanged) | ⭐ **YES — the flat procedural list delivered as FLOW.** "Step one… step four" is already a cadence; rap it |
| **A2** | **THE LIGHT KEEPS LIMPING BACK** | SUNNA | steppers dub | **UK garage / 2-step shuffle**, skipped hats, swung sixteenths | the dub half-time + spring echo, arriving as the drop-out | no |
| **A3** | **STILL HOLDING STILL** | SUNNA | ska-punk / two-tone | **jersey-club / trap bounce** — triplet kick, bed-squeak, 140 half-time | the two-tone upstroke skank + horn stabs, **pitched and glitched** | short — 8 bars max |
| **B1** | **ROOM TONE** | LOFN · AWE | trip-hop × neo-baroque | **granular / glitch-processed trip-hop** — the room tone itself granulated | the neo-baroque figure (unchanged); it is already the intrusion | no |
| **B2** | **KEPT, NOT PLAYED** | LOFN · AWE | ⚠️ **shares B1's bed** | ⭐ **pluggnb** — plugg production × R&B, detuned bell lead, sparse triplet hats | the byte-identical chorus, now sung over a bed that cannot repeat cleanly | no |
| **B3** | ⭐ **TAKE A HUNDRED** | SUNNA | girl-group over jungle | ⭐⭐ **JUNGLE / BREAKBEAT IS THE WORLD** — amen-adjacent break, chopped, 170ish, mono sub | ⭐ **the sixties girl-group stack is the WOUND** — 1963 thirds arriving *on top of* the break at one named moment, unaccompanied | ⭐ **YES — the take-count as an accelerating flow** |

**Her four, allocated:** ska → A3 · glitch → A3, B1, B3 · trap → A1, A3, B2 · flow-rap → A1, A3, B3.
**No song gets all four.** Each blend is one world + one intrusion + one seam.

---

## 5. ⛔ CONSTRAINTS — what may NOT move

**The refinement is a PRODUCTION and DELIVERY pass. The songs' meaning is finished.**

- ⛔ **Do not change the concept, the situation, the coined obscure emotion, or the radiant.**
- ⛔ **THE COMPOSITE form rule holds**: one figure recurs ≥3×, each a failed/partial version, **the last NOT
  fixed**; closed-loop bottom register; one radiant. **A genre change may not resolve what the song refuses
  to resolve.**
- ⛔ **The ten bans hold**, especially **D5 PRESENT TENSE** (the sentence is a DECISION, not a PREDICTION —
  spoken from the floor, mid-failure), **D8 OVERHEARING NOT ADDRESSING**, **D9 the tape is not redeemed**,
  **D6 the cohort gate**. QA measured **zero D5 drifts across all 24** — do not introduce the first one.
- ⭐ **A flow/rap section is a DELIVERY change, not a licence to re-write meaning.** Where one is added, it
  must carry lines the song already earns. ⛔ **A rap verse that addresses the listener breaks D8** — the
  flow is still overheard, still busy with its own fifteen seconds.
- ⛔ **L41 WATCH:** this run's confirmed residue is **"nobody"/"nothing"** discharging D8 by negation
  (256 hits, 20/24 songs). **Do not add more.** If a new flow line needs to refuse an audience, use **P05's
  displacement move** (*"Somebody put that sentence in my mouth"*) — an unnamed third party, not a negated one.

### The measured gates (`vault/gates.yaml`) — re-measure everything you touch
Style prompt **850–1000, target 870–960**, ≥985 flags hugging, **must end on terminal punctuation** ·
lyrics field **≤5000**, target 4800 · **sung lines 70–120**, target 78–110 (⚠️ **a rap section moves this —
re-count**) · **end-rhyme ≥0.30 · line-return ≥0.20 · words/line ≤7.5 · allit/100w ≥11.0** ·
**≥2 Source-2 phrases inside the MUSIC PROMPT itself** (P06 shipped zero last pass and had to be repaired —
⭐ **count the gate on the shipped artifact, never on the document that describes it**) ·
sung numerals **spelled out in words** · at most **one** sung numeric fact (A1 keeps "sixteen years",
B3 keeps "a hundred"; the rest sing none).

### ⚠️ Two cross-set gates the coordinator will check after the wave
- **THE ANGER AUDIT — aggression ≤ one third of the set.** Trap and glitch can tip this. Currently **A1 is
  the only aggressive song**; B3 and A3 must land as **fast and bright**, not heavy. **A delight goddess who
  arrives sounding furious is the personality failing.**
- **DISTINCTIVENESS.** ⭐ Last pass, folding identical mandatory language into four sibling prompts pushed
  similarity to **0.669 against a 0.58 ceiling** — the repair caused the breach. **Blend in your own words.**

**Token note (advisory, not a ban):** `phonk` and `snarl` are our two worst-performing style tokens
(3.44% / 3.18%). The ban was **not applied** — The Scientist's ruling, measured on a bad denominator — but
do not reach for them by default when a trap lane is in play.

---

## 6. THE REFINEMENT BUNDLES
Built by **`lofn-step11-packager`** — full personality YAML + Suno v5.5 rules + golden songs + full run
context + step 10 + the canonical Manual Refinement Instructions:
`output/daily/2026-08-12/music/pair_{01,02,03,04,06}_step11_fable_prompt.md` (96–175 KB each).

⚠️ **Packager finding:** it resolves personality from `05_pair_assignments.md`, which this run does not have
(the seed and orchestration were written as one combined file), so it **silently defaulted P06 to
LOFN-PRIME when P06 is SUNNA.** Caught and overridden with `--personality`; all five bundles now carry the
correct DNA. **A wrong-personality injection is exactly the "Lofn bleed" the injection mandate exists to
prevent, and it failed silently.** Logged for The Scientist.
