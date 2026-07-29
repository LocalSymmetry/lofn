---
run: 2026-07-28_one_more_door_lithophane
agent: lofn-qa
date: 2026-07-28
modality: image (Lofn-Vision, steps 06–10 per pair)
renderer: GPT-Image-2, DIRECTIVE mode
personality: LOFN-PRIME (AWE mode — the Kiln Attendant)
verdict: REPAIR
---

# QA REPORT — ONE MORE DOOR / LITHOPHANE PORCELAIN

**Status: REPAIR — CROSS-PAIR CONVERGENCE (L12) + RUN-LEVEL SPEC CONFLICT**

Not a soul loss. Not a structural failure. Not a negative-gate breach.
This package is structurally complete, materially serious and unmistakably LOFN-PRIME. It is held
back by four routable defects, three of which are the exact failure the ICB names as a blocker.

| Gate | Result |
|---|---|
| Structural (6 pairs × 4 prompts, steps 06–10, 07/09/10 present) | **PASS** — 6/6 canonical |
| Cross-pair convergence (L12) | **FAIL** — 3 collisions, 2 near-verbatim |
| Negative gate (6 clauses) | **PASS** in substance · 2 lexical hazards |
| Material fidelity — six-terrace signature | **FAIL** — index convention inverted across the pack |
| Material fidelity — unlit + lit per door part | **PASS** |
| Thumbnail law (≥1 prompt per pair at 16 px) | **PASS** — 6/6 · 1 under-specified |
| Soul — LOFN-PRIME (AWE, Kiln Attendant) | **PASS**, emphatically |

---

## 1. STRUCTURAL GATE — PASS

| pair | file | 06 | 07 | 08 | 09 | 10 | prompts | canonical |
|---|---|---|---|---|---|---|---|---|
| P1 door panels | `pair_1_door-panels.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| P2 the light | `pair_2_the-light.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| P3 carried objects | `pair_3_carried-objects.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| P4 corridor walls | `pair_4_corridor-walls.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| P5 floor + threshold | `pair_5_floor-threshold.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| P6 frame + failure | `pair_6_frame-failure.md` | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |

- **Refined pairs (05): 6** ✓ · **Total shipped prompts: 24** ✓ (6 × 4)
- **Steps 07 / 09 / 10 present in all six.** No pair is NON-CANONICAL.
- Steps 08→09→10 show real editorial movement everywhere, not relabelling: P1 records four raw-pass
  kills with reasons; P2 records two blocking rerolls with quoted BEFORE/AFTER; P3 runs a four-seat
  critic/refiner loop; P4, P5, P6 each generate six guides, rank them, and consume two exploratory
  guides into the shipped four. This is a worked chain, not a collapsed one.
- No template placeholders, no `TODO`/`TBD`, no lorem, no unfilled variables anywhere.
- No artist name leaks into any shipped prompt body. Every pair carries the interpretive-construct
  disclaimer. Clean.

### Recorded deviation (non-blocking, not waived silently)

`skills/qa/SKILL.md` §9 defines the canonical artifact granularity as separate coordinator files
`step00_aesthetics_and_genres.md` … `step05_refine_medium.md` plus separate per-pair step files
`pair_NN_step06_facets.md` … `pair_NN_step10_revision_synthesis.md`. This run ships **one
consolidated file per pair** and has no `step00`/`step01`/`step02` files. The parent brief scopes the
structural gate to *"steps 06-10 worked per pair, steps 07/09/10 present"*, which is satisfied on the
evidence. QA is recording the granularity deviation rather than dropping the check. **Not a blocker
for this run; fix the file layout at the next run rather than re-cutting this one.**

Format nit: P4 and P5 ship their four prompts as markdown **blockquotes** (`> `); P1, P2, P3, P6 ship
them as fenced blocks. A blockquote copy drags `> ` prefixes into the renderer. Convert P4/P5 to
fenced blocks before hand-off.

---

## 2. CROSS-PAIR CONVERGENCE (L12) — **FAIL. This is the verdict driver.**

The ICB is explicit: *"a Special Flair seeds a motif, never a phrasing… Two pairs arriving at the
same sentence is a blocker."* Three pairs deferred their sibling diff to QA (P4 §5, P5 §L12 note,
P6 §6 "NOT CHECKED — P5"). The diff was owed here. It fails.

### COLLISION 1 — BLOCKER · P3 ↔ P6 · the atlas sentence, 33 words near-verbatim

**P3 `pair_3_carried-objects.md:277–281` (V1, repeated in V2):**
> "Produce five separate hand-made porcelain objects in one horizontal row on a completely flat,
> uniform deep-navy field, hex 10161F. **Space them evenly, isolate each one inside its own cell, let
> none of them touch another and none touch the frame edge.** Set every object on the identical
> baseline at the identical scale, seen straight on in orthographic front elevation with no
> perspective convergence."

**P6 `pair_6_frame-failure.md:718–721` (Prompt 4):**
> "Produce seven separated fragments of a single shattered bone-china slab, arranged in one
> horizontal row on a completely flat, uniform deep-navy field. **Space them evenly, isolate each one
> inside its own cell, let none of them touch another and none touch the frame edge.** Set them all
> at the same scale and seen straight on with no perspective convergence."

The bolded 21 words are **identical**. The surrounding sentences are the same construction with the
nouns swapped — the precise L12 failure mode the run's own core seed names. Both prompts also open
on the same formula: P3 *"Render a flat parts atlas, not a photograph of a display"* / P6 *"Render a
flat parts atlas of the pieces of one broken object."*

This is **not** one of the deliberately-shared mechanical clauses (ladder, matte hex, penumbra ban,
annotation ban). It is a **composition** instruction, and composition is where the pairs were told to
diverge. Aggravating: P6 §6 asserts *"CLEAN against P3 and P4. No shared motif and no shared sentence
outside the intentionally identical mechanical clauses."* That claim is false as written.

Aggravating further: P6's own step-07 killed guide **G6 (the rack of empty frames)** on exactly this
ground — *"It is P4's vitrine strategy in a different material — a straight motif collision"* — and
then re-entered the collision through P3's door instead of P4's.

### COLLISION 2 — BLOCKER · P3 ↔ P6 · the thumbprint solve, near-verbatim + poached flair

**P3 `pair_3_carried-objects.md:451–453` (V3):**
> "Beside the channel, show **a shallow dished oval** about the size of a thumb pressed into the wall
> before firing: it is **locally thinner**, so it **transmits one full step brighter** than everything
> around it, **as a flat brighter plateau with** the same **hard edge, not as a soft bloom**."

**P6 `pair_6_frame-failure.md:748–751` (Prompt 4, cell 2):**
> "the LEFT half of a thumb-mark pressed into the greenware before firing — **a shallow dished oval**
> with fine ridge lines, **locally thinner** than the clay around it, so it **transmits one full step
> brighter** than its surroundings **as a flat brighter plateau with a hard edge, never as a soft
> bloom**."

Same flair, same physics, same clause order, same closing negation. And `05_pair_assignments.md`
assigns **Thumbprint in the Greenware to P3**; P6 lists it under *"drawn additionally where the
material asked for them"* and then solves it in P3's words. P6's Prompt 1 thumb-marks (many hands,
in hardened **putty**, read by raking light) are genuinely differentiated and may stay. Prompt 4
cell 2 is the hit.

### COLLISION 3 — BLOCKER-ADJACENT · P4 ↔ P5 · the brightest-thing-with-no-picture, verbatim fragment

**P4 `pair_4_corridor-walls.md:457` (Prompt 4):**
> "Make that blade the brightest value in the entire image — **brighter than T6 could ever be** — and
> give it no structure whatsoever: no terraces, no contours, no picture. The one place with the most
> light is the one place with nothing carved into it."

**P5 `pair_5_floor-threshold.md:709–710` (Prompt 4):**
> "make it the brightest thing in the frame, **brighter than T6 could ever be**, and render it as a
> dashed line… Give the crack no terraces, no contours, no structure and no picture of any kind: the
> most light in the image arrives through the one place with nothing carved into it."

Two pairs, both in their **Prompt 4**, reach the same six-word fragment *and* the same closing
sentence shape *and* the same motif solve. P5's anti-convergence note caught the shared verb
"misregister" but missed this. The wider motif — *maximum brightness carries zero information* — is
additionally reached by P3 V4 (the T0 chip: *"the only part of it that says nothing"*), P5 Prompt 2
(*"the brightest area on the tile and the only area that says nothing"*) and P6 Prompt 4 cell 7
(*"carries no picture, no line and no detail at all"*). Four pairs on one solve.

P6 §10 rule 3 explicitly claims the opposite move to avoid this — *"Where P4's failure out-shines the
image, this pair's failure out-darkens it"* — and then Prompt 4 cell 7 executes P4's move anyway.
**The declared divergence is contradicted by the pair's own shipped prompt.**

### DECLARED OVERLAPS — reviewed and ACCEPTED, no repair owed

| overlap | pairs | ruling |
|---|---|---|
| The carver's likeness, verbatim sentence | P2 ↔ P6 | **ACCEPT.** Two parts in one game showing the same person must show the same person. P6 caught its own paraphrase and adopted P2's line deliberately. Correct call — but promote the sentence to a **run-level continuity constant** in `CREATIVE_CONTEXT.md` so it reads as a shared asset, not a collision. |
| Candle carbon / soot | P1 ↔ P6 | **ACCEPT.** Separated by placement and consequence: P1's is a surface patch with a scalloped tide line that *records a visit*; P6's lies in trough floors under gravity and *takes brightness back*. Guttering devices are distinct (hard scalloped ellipse vs one terrace half-open at a hard vertical). |
| Iron staples | P2 ↔ P6 | **ACCEPT.** Same axis draw by design. P2: three bars among several flaws on a lamp. P6: the entire thesis, with drilled pockets, sunk legs and a section at 4.5 mm in a 6.0 mm body. |
| Celadon Turn (double-assigned) | P2 ↔ P5 | **ACCEPT, ruling confirmed** (P5 raised it; QA answers). Mechanically distinct — P2 turns colour on a **body-thickness** shelf edge, P5 turns it by **presence/absence of a glaze skin**. Different palettes, different families, no shared sentence. Both ship. |

### One arithmetic error that predicted the failure

`05_pair_assignments.md:104` states *"P1/P6 and P2/P6 share **two** axes each."* P1/P6 is correct
(bone-china, candle). **P2/P6 share three** — hand `worn plaster mould`, image-of `the person who
carved it`, failure `visible staples`. The pair-diversity exemption for P6 was granted on a
miscount, and P2↔P6 is precisely where P6 had to fix two real collisions during the write. Correct
the table.

---

## 3. NEGATIVE GATE — PASS in substance (6/6), two lexical hazards

Audited every prompt body, not the pairs' self-checks.

1. **Printed / painted / decaled / projected** — clean. Every brightness in the run is bound to a
   named millimetre depth. P3's `sgraffito through a dark slip` was the riskiest draw and is answered
   three ways per prompt. P4's `another lithophane, lit in turn` was the other blocking risk and got a
   real ruling, not a rewording: *the picture does not survive the crossing; what arrives is banded
   ordnance that breaks and steps sideways at every riser it crosses.* A projection rides over a
   surface; this one is cut by it. Correct.
2. **Alpha-gradient glow / bloom / god-rays / soft falloff / lens flare** — clean. Every pair bans the
   full list by name in every prompt body. P2 killed two of its own drafts on this clause with quoted
   BEFORE/AFTER. Candle unevenness is solved as a hard scalloped terrace (P1) and as one terrace
   half-open at a hard vertical (P6) rather than as falloff. P5 strips *glow, aura, halo, shimmer,
   radiance* from its working vocabulary.
3. **Post-process filter over ordinary 3D** — clean, and provably so: P1 V1, P4 Prompt 1, P5 Prompt 1
   and P6 Prompt 1 each render the identical height field with **zero transmission**, which no tonal
   filter can generate. P6 Prompt 3 shows the ladder in **section**; P5 Prompt 4 misregisters the band
   structure 3 mm across a fault. Filters cannot do any of that.
4. **PBR metal / plastic / wet gloss / subsurface skin** — clean, with two bounded exceptions both
   handled honestly: P3's **pooled glaze** is confined inside one trough, declared dead vitrified
   glass rather than a highlight, forbidden from spreading, and exists because it *killed a line*;
   P5's **celadon** is pinned to the dry/matte end of the family with specular highlight, mirror and
   wet sheen banned by name, and P5 flags it as the instruction most likely to be ignored. P6's iron
   is the strongest move in the pack — *a light-eater, not a reflector*, with a checkable value
   ordering: **the brightest point on the iron stays darker than the darkest clay in frame.**
5. **Composed scene / room / HUD / typography** — clean, and the two structural typography risks were
   both anticipated: P4's image is a **survey** (surveys carry numerals) and bans numerals, lettering,
   annotation, legend and symbols in all four; P6's technique is an **exploded-parts sheet and a
   section** (which normally arrive with leader lines and hatching) and bans dimension lines, section
   marks, hatching and callouts, with the order *"this is a sawn object, not a drawing."*
6. **Documentary of a real pottery workshop** — clean. No maker, bench, kiln, wheel or tool in any
   frame; the candle is named as a cause and banned from appearing. The impossibility sits in the
   objects, not the staging.

### Lexical hazards — REPAIR, low cost

- **P3 `:293` and `:431`** — the slip is described as *"near-black with a brown bloom"* / *"a brown
  bloom"* **inside prompt bodies that later order "Do not add glow, bloom, haze…"**. A prompt that
  both requests and forbids "bloom" is a coin-flip for the renderer. Replace with *"a brown-black
  mottling where it thinned over a curve."*
- **P3 `:312`** — *"it reads as a **soft** dished depression"* in V1. `soft` is on the veto list. P2
  ran an explicit zero-veto-word scan on its four bodies (`pair_2_the-light.md:325`); P3 did not.
  Replace with *"a shallow dished depression with a hard rim."*

Neither is a gate breach of substance. Both are prompt hygiene and both will bite at render time.

---

## 4. MATERIAL FIDELITY

### 4a. Six-terrace signature — present in all six pairs, **but the index convention is inverted**

The *phenomenon* is consistent everywhere: exactly six carvable values, published in millimetres,
0.8 mm risers, sharp arrises, and blending/ramping/feathering/dithering forbidden by name. That much
is excellent and it is the run's spine.

The *labelling* is not:

| pair | T1 | T6 | convention |
|---|---|---|---|
| P1 | 6.0 mm — holds all light, near-black | **2.0 mm — brightest** | thick→thin |
| P4 | 6.0 mm | **2.0 mm** | thick→thin |
| P5 | 5.6 mm | **1.6 mm — brightest carved value** | thick→thin |
| P6 | 6.0 mm | **2.0 mm** | thick→thin |
| **P2** | **~2.0 mm — first to fuse to white** | ~6.0 mm — opaque, dead black | **thin→thick (INVERTED)** |
| **P3** | **0.8 mm — near-white** | 2.4 mm — opaque, dead black | **thin→thick (INVERTED)** |

`pair_1_door-panels.md:49` — *"**T6 = 2.0 mm** (brightest, one step off burn-out)"*
`pair_2_the-light.md:60` — *"| **T1** | ~2.0 mm | first to fuse into flat white |"*

`pair_6_frame-failure.md:270` asserts a corridor-wide constant — *"the same numbers P4 published, on
purpose. **One corridor, one ladder.**"* P2 and P3 break it. Any consuming kernel, any atlas
assembler, and any human cross-reading two prompts gets **opposite meanings from the same token**.
Per-family *thickness scaling* (P3's 0.8–2.4 mm eggshell, P5's 5.6–1.6 mm walked-on tile) is
justified in-file and should stay. The **direction of the index** must not vary.

### 4b. Matte value — three navies, not one

| hex | pairs |
|---|---|
| `#0B1220` | P2, P4, P5, P6 |
| `#0D1220` | **P1** |
| `#10161F` | **P3** |

`pair_6_frame-failure.md:916–920` caught two of the three and asked for a ruling: *"a parts atlas
assembled from both will have two navies in it and any tolerance-based key will behave differently
per part. The run needs one published value."* It is three, and P1 is the one P6 did not see. This
directly breaks the run's own repeated promise that *"a border-connected flood fill removes it in one
pass"* with **one** setting.

**Ruling: publish `#0B1220`** (4 of 6 pairs, both AMBITIOUS pairs, and the majority of the atlas
sheets). Amend P1 and P3.

### 4c. Unlit/shadow + lit/transmission per door-family part — **PASS**

The metaprompt rule is scoped: *"any **door part** ships as an unlit/shadow and a lit/transmission
pair from the same height field."*

- **P1 (the doors)** — four states off one identical registration and one carving: V1 unlit/raking,
  V2 mid-reveal, V3 full transmission, V4 burn-out. Fully satisfied and the best executed instance.
- **P6 (the armature + its panels)** — Prompt 1 unlit and Prompt 2 lit, explicitly *"armature
  geometry, scale, position and baseline IDENTICAL… so live code swaps the two states of the same
  object."* Satisfied.
- P2 V4 (unlit) partners V1–V3 (lit). P3 V1 (unlit) partners V2 (lit) at identical atlas geometry.
  P4 Prompts 1/2 and P5 Prompts 1/2 are genuine unlit/lit pairs off one height field.

P4, P5 and P6 each self-flag that their Prompts 3 and 4 have no partner. **QA rules this NOT a
failure** — the rule is written for door parts and those are single-state library parts (a macro
seam, a section, a shatter sheet). Add the scope word to the metaprompt so the next run does not
re-litigate it.

---

## 5. THUMBNAIL LAW — PASS 6/6

| pair | survivor | evidence |
|---|---|---|
| P1 | **V3**, **V4** | six countable warm terraces on flat navy; V4 is one black rectangle on flat white — *"instantly readable as a shape at any size"* |
| P2 | **V3** | one amber shape floating in black, palette reduced to 3 tones with amber dominant |
| P3 | **V2** | one lit object against four dark — a shape that survives any downscale |
| P4 | **Prompt 2** | one lit slab against three dark, risers coarsened so shadow bars survive shrinking |
| P5 | **Prompt 2**, **Prompt 4** | one bright unworn footfall against dark; plus P5's unlit state carries **hue** contrast (white-rust on bottle-green), so even Prompt 1 downscales to a legible contour map |
| P6 | **Prompt 4** | seven warm shards on navy — a constellation at 16 px |

Every pair that fails the law on a given frame **says so and refuses to fudge it**: P1 §1 on V1,
P2's `promo:false` tag on V4 (*"I am not going to smuggle a residual ember in to game the gate"*),
P3 §4 on V1, P4 §1 on Prompt 1, P6 §1 on Prompts 1 and 3. That honesty is worth more than a clean
sheet would have been.

**One under-specification — REPAIR, one line per prompt.** P2 writes **128 px** into all four prompt
bodies (`"survive being reduced to a hundred and twenty-eight pixels"`, V3 and V4) where the ICB law
is **16 px**. P2's own step 09 discusses 16 px correctly, so this is a transcription slip, not a
disagreement — but it hands the renderer an 8× easier target. Restate as 16 px.

---

## 6. SOUL — PASS. This is LOFN-PRIME (AWE, the Kiln Attendant).

The failure mode this gate exists to catch — *structurally complete but generic* — is not present.
There is no ceramic-studio neutrality anywhere in the package. Evidence, one per pair:

- **P1** — the brightest thing on the door is a person-shaped void, and the 9 mm corner that ruined
  the picture is the only thing that survives burn-out. *"The relief claims a furnished, occupied
  room. The light says: nobody is in there, and there is another door."*
- **P2** — the charge gauge is *"the picture of a woman going out."* The last thing the lamp shows
  you before it dies is her hands holding the same lamp — the one detail you cannot see while you
  still have plenty.
- **P3** — five objects that look identical in the dark, and the piece you traded comes back with a
  hole in it that is *"the brightest and emptiest thing you own."*
- **P4** — every panel is lit by another panel and still cannot read it. *"Light crosses the
  corridor; meaning does not."*
- **P5** — the worn path is where everyone went; the carved footfall is where you go next, and they
  are not the same place. The path of previous traffic transmits a **different colour** because feet
  wore the glaze off it.
- **P6** — *"In a picture made of light, the repair is the dark."* Five staples, each taking a piece
  of a face permanently, and they did not even close the crack — they stopped it travelling.

**PATIENT DREAD, LIT FROM BEHIND** is carried literally rather than tastefully throughout, which is
the Context Skeptic's actual demand. Three pairs killed their own best-sounding sentence on exactly
that ground and recorded the kill (P5's *"the floor remembers the walk you have not taken yet"* —
*"the most tasteful sentence I wrote all day and it is a gallery caption"*; P6's kintsugi warning;
P2's *"Render it and shut up"*). The AWE/INDIGNATION duality from `core_seed.md` survives intact.

Every pair ships an **HONEST PROBLEMS** section naming its own weakest frame, its own axis friction
and its own render-gate watch flags. Two pairs supplied *measurable* reject tests for the render gate
(P2: *"reject any output where the navy field is brighter within fifty pixels of the silhouette than
it is at the frame edge"*; P5: *"if any returned image shows a reflection of the light source in the
green, it is a gate-4 failure"*). That is the studio behaving like a studio.

**No soul repair is owed. Do not touch the voice while fixing the four defects below.**

---

## 7. ROUTED REPAIR BRIEF

Four items. All are targeted edits. **No pair needs regenerating; no step needs rerunning.**

### R1 — BLOCKER · `pair_6_frame-failure.md` · STEP 10 · Prompt 4 (`panel/carver_01__shatter_fragments__backlit_atlas`)
Rewrite two passages so P6's atlas is not P3's atlas.
- **Layout sentence (`:718–721`).** Delete *"Space them evenly, isolate each one inside its own cell,
  let none of them touch another and none touch the frame edge"* and the *"Render a flat parts
  atlas"* opener. P6's own content already supplies a different logic: the fragments are **rotated
  and deliberately mis-ordered so none sits beside the piece it broke from.** Build the layout
  instruction out of *that* — an anti-reassembly order, not an isolation grid. P3's frame is a set of
  identical things; P6's is one thing that has stopped being one thing. The sentences should not be
  able to swap.
- **Cell 2 (`:748–751`).** Remove the thumbprint solve. *Thumbprint in the Greenware* is assigned to
  **P3** (`05_pair_assignments.md:47`) and P3 owns the sentence. P6 already solves the flair better and
  uniquely in **Prompt 1** — many overlapping thumb-marks pressed by *maintainers*, in hardened putty,
  over years. Re-cast cell 2 from P6's own vocabulary (iron · soot in trough floors · sunk staple ·
  saw cut · snapped arris · shallow-pressed half) and drop the Thumbprint claim from the pair header.
- Then **re-run P6 §6 against P3 and P5** and correct the false *"CLEAN against P3 and P4"* line.

### R2 — BLOCKER · `pair_5_floor-threshold.md` STEP 10 Prompt 4 **and** `pair_6_frame-failure.md` STEP 10 Prompt 4
Break the shared solve on *maximum brightness carries zero information*.
- **P5 Prompt 4 (`:709–710`)** — delete the fragment *"brighter than T6 could ever be"* (verbatim from
  P4 Prompt 4 `:457`) and the trailing *"the one place with nothing carved into it"* construction. P5
  already owns a solve P4 cannot reach: the crack **displaces the picture 3 mm** and is a *dashed*
  line, half white-hot and half bottle-green because the glaze was still moving. Let the brightness
  clause be built on the **dash and the offset** — a bright line that *lies* — not on the absence of
  carving. P4 keeps the informationless-gap reading; it drew it first.
- **P6 Prompt 4 cell 7** — P6's §10 rule 3 promises the **opposite** move (*"this pair's failure
  out-darkens it"*) and cell 7 breaks that promise with a third burn-out-to-white. Either delete
  cell 7's whiteout and let cell 4's dead-black silhouette carry *both extremes* against the median
  exposure, or make cell 7's failure a **darkening** one so the pair keeps the divergence it declared.

### R3 — RUN-LEVEL SPEC · `pair_2_the-light.md` + `pair_3_carried-objects.md` (terrace index) · `pair_1_door-panels.md` + `pair_3_carried-objects.md` (matte hex)
- **Terrace index.** Invert P2's and P3's T-numbering to match P1/P4/P5/P6: **T1 = thickest/darkest,
  T6 = thinnest/brightest.** Change labels only — keep each pair's own millimetre scale (P3's
  0.8–2.4 mm eggshell and P5's 5.6–1.6 mm tile are justified in-file and correct). Touch: P2's ladder
  table `:58–66` and every `T1…T6` reference in its four spec headers; P3's terrace table `:57–66`,
  its four SPEC blocks and its four prompt bodies. P6 `:270` already declares the constant — make it
  true.
- **Matte hex.** Publish **`#0B1220`** as the run constant in `CREATIVE_CONTEXT.md` §RENDERER. Amend
  **P1** (`#0D1220` → `#0B1220`, four prompt bodies) and **P3** (`#10161F` → `#0B1220`, four prompt
  bodies + two SPEC headers).

### R4 — HYGIENE · `pair_3_carried-objects.md` (STEP 10, V1 and V3) and `pair_2_the-light.md` (STEP 10, all four)
- **P3 `:293`, `:431`** — replace *"brown bloom"* / *"brown-black bloom"* with *"brown-black
  mottling"*. A prompt must not request a veto word it also forbids.
- **P3 `:312`** — replace *"a soft dished depression"* with *"a shallow dished depression with a hard
  rim."*
- **P3** — run the zero-veto-word scan P2 documents at `pair_2_the-light.md:325` across all four
  bodies and record the result, as P2 did.
- **P2** — change **128 px** to **16 px** in V3 and V4 prompt bodies and in the four
  *"Thumbnail read at"* headers, to match the ICB's stated law.

### Housekeeping (do with the above, not separately)
- `05_pair_assignments.md:104` — correct *"P2/P6 share two axes"* to **three** (hand · image-of ·
  failure) and restate P6's structural exemption on the true count.
- `04_metaprompt.md` §DAILY MANDATES — scope *"Two states, always"* to **door-family parts**, so P4,
  P5 and P6 stop flagging correctly-single-state library parts as a shortfall.
- `CREATIVE_CONTEXT.md` — promote the carver's likeness line to a **run continuity constant** so
  P2/P6 sharing it verbatim reads as intended continuity rather than convergence.
- Convert P4 and P5 prompts from blockquotes to fenced blocks for clean paste.

**Re-QA after R1–R4:** convergence sweep only (a sentence diff across all six pair files) plus a
grep confirming one navy value and one terrace direction. Everything else has already passed.

---

## 8. CONTINUITY LEDGER — record back, per `core_seed.md`

The founder's instinct-pick of lithophane porcelain from an embroidery post landed on **four**
documented COMPETITION_LEARNINGS advantages at once (L1 container/inner-spectacle · L2
impossibility-over-realism · Emanating-Light-as-Formula · Recursive Wonder). QA confirms the seed's
claim: this is the strongest material/ledger alignment in the record, and the package delivers on it
— the material argues for the mechanic rather than dressing it.

**L12 confirmed HIGH, not MEDIUM.** The core seed rated anti-convergence at MEDIUM confidence. Three
of six pairs deferred their sibling diff to QA because they were written before their siblings
existed, and the diff caught **three collisions, two near-verbatim**, in a run where every pair had
distinct axis draws, distinct families and distinct techniques. Structural diversity did **not**
prevent phrase convergence. Recommend raising L12 to **HIGH** and adding a mandatory mechanical
sentence-diff as a step-10 exit condition rather than a QA discovery.

**New finding for the ledger — the mechanical/motif clause split works.** P4, P5 and P6 independently
converged on the same counter-move: *make the clauses we must share (ladder, matte, penumbra ban,
annotation ban) verbatim identical, and write every motif-carrying sentence in a shape no other
pair's draw would produce.* Where that discipline was applied it held. All three collisions above
are in **composition and motif** sentences, which is precisely where the discipline was not enforced
by a tool. Ship the rule; give it a checker.

**Images:** none generated. Prompts only, per the founder cost gate. Correctly stated in all six
pair files.

---

**VERDICT: REPAIR.**
Structurally complete · negative-gate clean in substance · thumbnail-legal · and genuinely
LOFN-PRIME. Held for the documented L12 blocker (2 near-verbatim collisions + 1 shared solve) and two
run-level spec conflicts (terrace index direction, three matte values). Four routed edits, no
regeneration. This is a strong package one editing pass away from SHIP.

---
---

# RE-QA — 2026-07-28 (post-R1–R4)

**Scope, as briefed:** narrow. Four checks only — (1) a mechanical sentence-diff across the six
`pair_*.md` files, (2) run-constant greps for the matte hex and the terrace direction, (3) a
veto-word positive-use scan of the prompt bodies, (4) confirmation that R1 and R2 are gone rather
than reworded around. Every other gate passed in the first pass and was not re-litigated.

**Method (check 1).** Mechanical, not editorial. All six pair files were tokenised to lowercase
word streams with markdown punctuation stripped; every 12-gram was indexed; every cross-file match
was extended to its maximal run and de-duplicated to left-maximal hits only. **51,349 tokens,
15 file pairings, 178 maximal shared runs of ≥12 words.** Each run was then classified against the
three deliberately-shared clause families. The classification below is the whole result set, not a
sample.

## RE-QA 1 — CONVERGENCE SENTENCE-DIFF

### Disposition of all 178 runs

| class | runs | ruling |
|---|---|---|
| Negative-gate self-check + the six ban clauses | 81 | **ALLOWED** — declared identical by design |
| Terrace-ladder statement (mm table, risers, arris, no-blend) | 19 | **ALLOWED** — the published run constant |
| Matte / extraction contract (hex, part box, flood fill, margins) | 3 | **ALLOWED** — the slicer needs one constant |
| Carver's-likeness line (P2 ↔ P6, 25 w) | 8 | **ALLOWED** — promoted to a run continuity constant at `CREATIVE_CONTEXT.md:89` |
| Pipeline scaffolding (step headers, DCB rows, the model-constructs disclaimer, the axis-draw table quoted verbatim from `05_pair_assignments.md`) | 33 | **ALLOWED** — not prose, not motif |
| Extensions of the above that the first keyword pass did not key on (re-read individually) | 32 | **ALLOWED** |
| **Genuine motif/composition convergence** | **2** | **1 blocker-grade, 1 minor** |

### R1 and R2 collisions: **CLEARED.**

- *"space them evenly, isolate each one inside its own cell…"* now survives in exactly **two**
  places: `pair_3_carried-objects.md:299` (P3's prompt body — P3 owns it, wrote it first) and
  `pair_6_frame-failure.md:991`, where it appears **inside P6's §6 note as a quotation of the
  withdrawn sentence.** Not a live instruction. The diff flags it; a human does not.
- *"render a flat parts atlas"* — same disposition. Live only in P3 (`:296`, `:380`).
- *"brighter than T6 could ever be"* — live in **P4 only** (`pair_4_corridor-walls.md:464`). Gone
  from P5's prompt body entirely.

### HIT 1 — **BLOCKER-GRADE** · P4 ↔ P5 · the kiln-warp sentence, 17 words identical, both in prompt bodies

**`pair_4_corridor-walls.md:393–394`** (PROMPT 3 body, `wall.panel.grog.transmit_fail`):
> "Use the slab's own warp against it: because it bows toward the camera, **hold the far half a full
> terrace darker than the near half, with the change happening along** a hard line where the bow
> turns."

**`pair_5_floor-threshold.md:543–545`** (PROMPT 2 body, `floor.tile.celadon.underlit` — the pair hero):
> "Lean the tile 8° toward the camera along its own low bow, and use the bow: **hold the far half a
> full terrace darker than the near half, with the change happening along** one hard line where the
> bow turns, so the object's warp costs it a step of information."

Seventeen consecutive identical words, then the same closing construction with one article swapped
(`a hard line` / `one hard line`) and the same trailing clause shape. This is a **composition and
lighting** sentence, not a mechanical clause — it is on none of the three shared lists.

**Aggravating, and this is why it is blocker-grade rather than a note.** *Kiln Warp* is assigned
**exclusively to P4** (`05_pair_assignments.md:60`), and **P5's own pair header declares it
untouched**: `pair_5_floor-threshold.md:36–38` — *"**Deliberately untouched** (owned elsewhere,
collision risk): … Kiln Warp (P4) …"* P5 then solves the warp in P4's sentence, in its hero prompt.
That is a false declaration of the same species as P6's withdrawn *"CLEAN against P3 and P4"*, and
it is the identical defect class the first pass ruled a **BLOCKER** at R1(b) — one pair solving
another pair's assigned flair in that pair's words. The prior QA did not catch it because its diff
was manual; this one is mechanical.

The first pass's own §L12 finding predicted exactly this: *"Structural diversity did not prevent
phrase convergence."* P4 and P5 have different families, different cameras, different clay bodies,
different millimetre ladders and different palettes — and still landed on the same sentence.

### HIT 2 — **MINOR** · P4 ↔ P5 · the grazing-source setup, 12 words exact inside a longer echo

**`pair_4_corridor-walls.md:261–263`** (PROMPT 1 body):
> "Light the strip with **one grazing source from image-left at 8° above the** slab **plane and
> nothing else.** Let that raking light do all of the work: **throw a hard black shadow off the left
> face of every riser** and leave the right face bare…"

**`pair_5_floor-threshold.md:453–455`** (PROMPT 1 body):
> "Light it with exactly **one grazing source from image-left at 8° above the** tile **plane and
> nothing else** — no fill, no ambient, no second source. Let that grazing light **throw a hard
> black shadow off the left face of every riser**, catch the throwing spiral…"

The 12-word exact run is *"throw a hard black shadow off the left face of every riser"*; it sits
inside a much longer near-verbatim construction (the source count, the bearing, the 8° elevation,
the "and nothing else" exclusion, the verb).

**Ruled MINOR, not blocking, because this is mechanical in kind** — an angle and an exclusion, the
same species as the ladder and the flood-fill rule, and *Raking Light* is a flair the assignment
table hands to **both** P4 (`:60`) and P5 (`:67`, `:75`). **But it is not declared anywhere.**
`CREATIVE_CONTEXT.md:41` and `04_metaprompt.md:24` say only *"unlit reads by shadow (raking light)"*;
the `8°`, the `image-left` bearing and the single-source exclusion appear in no run-constant list,
and no other pair uses them (P1, P3 and P6 all describe their unlit key differently). Two pairs
converged on an undeclared number. Fix it in **either** direction, but fix it deliberately:
publish the grazing key as a run constant in `CREATIVE_CONTEXT.md` — which makes the shared sentence
correct, like the ladder — or give P5 its own bearing. Leaving it undeclared and identical is the
only outcome that is wrong.

### Named-and-permitted, so a third pass does not rediscover them

- **P4 `:321` ↔ P6 `:531`** (13 w, both prompt bodies): *"Drive the transmission out of the face in
  flat hard-edged plateaus: open T6 to a…"*, and eleven words later both close on *"Make the boundary
  between any two terraces a single hard line."* This is the **terrace-ladder statement in its
  operative form** plus the hard-boundary ban — allowed — and the colour values diverge inside the
  same sentence (dirty ochre-amber vs near-white warm cream). Permitted. Recorded.
- **P2 `:58–59` ↔ P3 `:57–58`** (21 w): the index-convention paragraph. This is the R3 repair itself
  — identical by instruction. Permitted.
- **P4 `:187` ↔ P5 `:141`** (12 w): both quote the same Hyper-Skeptic (after Muratori) line about
  publishing the numbers. An attributed citation in commentary, not prose convergence. Permitted.
- **P1 `:91` ↔ P6 `:923`** (16 w): the run's legibility rule, published at `CREATIVE_CONTEXT.md:70`,
  `04_metaprompt.md:76` and `core_seed.md:130`. Quoted, and P6 quotes it inside quotation marks.
  Permitted.
- **P2 ↔ P6 axis-draw line** (13 w, ×4): *"the image is of the person who carved it · repaired with
  visible staples"* — the assignment table verbatim, in metadata lines, on a draw the table
  deliberately gives both pairs. Permitted, and already declared in P6 §6.

**Check 1 verdict: FAIL on one blocker-grade hit, one minor.** Down from three collisions to one,
and the one found is new — not a reworded survivor.

## RE-QA 2 — RUN CONSTANTS: **PASS, both.**

**Matte hex — exactly one value.** Every 6-digit hex in the six pairs plus `CREATIVE_CONTEXT.md` was
extracted and counted. Result: **`#0B1220` and nothing else** as a matte value — P1 ×5, P2 ×1
(SPEC only; P2's bodies spell every number out in words by house style and say *"deep-navy field"*),
P3 ×5, P4 ×4, P5 ×4, P6 ×6, `CREATIVE_CONTEXT.md` ×1 (`§RENDERER`, published as the run constant per
R3). The other hexes present are per-pair **palette tones** (`#FFE7BE`, `#E9A85C`, `#8A8172` etc.),
not matte fields. `#0D1220` and `#10161F` survive **only** in `INDEX.md:150` and `QA_REPORT.md`
`:78/:246/:247/:379` as the record of the defect, and in `pair_6_frame-failure.md:939/:943` as P6's
own resolution note. **Zero live occurrences.**

**Terrace direction — exactly one.** `T1 = thickest/darkest → T6 = thinnest/brightest`, everywhere:
`CREATIVE_CONTEXT.md:95` (the constant) · P1 `:49–50` · P2 `:58–66` (table inverted; T1 6.0 mm opaque
dead black → T6 2.0 mm first to fuse) · P3 `:57–66` (table inverted; T1 2.4 mm dead black → T6 0.8 mm
near-white, plus T0) · P4 `:257`, `:315`, `:381`, `:453` · P5 `:418`, `:500`, `:601`, `:691` ·
P6 `:272`, `:420`, `:517`, `:635`, `:740`. Every millimetre table descends monotonically from T1.
P2 and P3 keep their own scales, as R3 required — only the direction is shared. **No inverted index
survives anywhere in the pack.**

## RE-QA 3 — VETO WORDS: **PASS.**

Scanned `glow · bloom · haze/hazy · soft falloff · god-ray · lens flare`, plus `halo · blur · bokeh`,
across all six pairs. **Every occurrence inside a prompt body is a prohibition**, which is correct
and expected. Representative: P1 `:307`, `:379`, `:455`, `:523` (four explicit FORBIDDEN blocks) ·
P3 `:351–352`, `:413–414`, `:482–483`, `:552–553` · P4 `:349–350`, `:412–413`, `:463` ·
P5 `:555–556`, `:647–648`, `:739–740` · P6 `:449–450`, `:570`, `:674`, `:802–804`.

Every construction that could read positively was opened and checked:

- P1 `:301` *"never as a haze"* · `:481` *"render the loss as a merge, never as a bloom"* ·
  `:502` *"not softened, not faded, not glowing — they are simply GONE"* — all negations.
- P3 `:406` *"never a halo"* · `:474` *"not as a soft bloom"* — negations.
- P4 `:327` *"lay it across its two neighbours as bars, not as glow"* — negation.
- P6 `:748` *"Nothing fades. Nothing halos."* · `:763` *"no bloom around it"* — negations.
- **P1 `:386`** *"a person-shaped hole glowing alone in the dark"* and **P5 `:99`** *"The path of
  everyone who came before glows a different colour"* are the only positive uses in the pack, and
  **neither is in a prompt body** — P1 `:386` is a *"Why this wins"* rationale line, P5 `:99` is a
  step-06 design-consequence paragraph. No renderer ever sees them.
- **R4 residue is clean.** P3's *"brown bloom"* survives only at `:266`, inside P3's own step-10
  record of having removed it. No prompt body asks for a word it also forbids.

**One cosmetic note, not a gate failure.** `CREATIVE_CONTEXT.md:25` reads *"backlit a picture blooms
out of the clay."* That is material prose in a context file, not a prompt body, and it is outside
the veto's stated scope — but it is the one place in the pack where the banned verb is used
approvingly, and it costs nothing to change to *"a picture rises out of the clay."*

## RE-QA 4 — ARE R1 AND R2 ACTUALLY GONE? **Yes. Rebuilt, not reworded.**

**R1(a) — P6's atlas layout.** P6 Prompt 4 `:728–735` no longer has an isolation grid. It now orders
the row **against the break**: *"Turn every fragment to a different angle, none of them upright,
none of them holding the attitude it held in the slab, and put no fragment next to a piece it
actually broke away from — every snapped edge in this row faces a stranger… the order is chosen so
the eye cannot put it back."* P3's logic is *identical things held apart in a grid*; P6's is *one
thing that stopped being one thing, ordered so it cannot be reassembled*. **The sentences can no
longer be swapped** — the test the repair brief set, met on substance rather than on synonyms. The
*"render a flat parts atlas"* opener is gone.

**R1(b) — P6's thumbprint solve.** Cell 2 (`:757–764`) is now **a drilled blind staple pocket with
its partner gone**, built entirely from P6's own vocabulary (iron · drilled pockets · the break that
took the second half), and it lands a better line than the one it replaced: *"The brightest thing on
this fragment is the hole somebody drilled to save it."* The Thumbprint claim is dropped from P6's
header (`:30` now reads *"**Not** drawn here: Thumbprint in the Greenware is assigned to P3"*). P6's
Prompt 1 keeps its genuinely differentiated putty thumb-marks, as the brief allowed. Longest shared
run with P3 on this material: **below 12 words.**

**R1(c) — the false §6 claim.** Withdrawn in P6's own voice at `:988–1003`: *"The earlier claim on
this line was 'CLEAN against P3 and P4.' **That claim was false and is withdrawn.**"* Both collisions
are named, quoted, and their repairs described. The correct disposition.

**R2(a) — P5's brightness solve.** *"brighter than T6 could ever be"* is gone; so is *"the one place
with nothing carved into it."* P5 Prompt 4 `:715–727` now builds the clause on the two things only
P5 has — the **dashed** line (white-hot where the glaze left the crack empty, bottle-green where it
ran in while the fire was still moving) and the **3 mm misregistration** — and lands on a thesis P4
cannot reach: *"The brightest line on the floor is not empty — it is wrong, and it is pointing."*
P4 keeps the informationless gap. Two different ideas now, not one idea twice.

**R2(b) — P6 cell 7.** The third burn-out-to-white is gone. Cell 7 (`:777–787`) now **darkens**, as
P6's §10 rule 3 promised: the thinnest fragment on the sheet is *"the darkest clay in the frame
instead"*, its trough floors packed with candle carbon — *"The least clay on the sheet, and the least
light. Turning the candle up will not open it: carbon is not thickness, and it does not come off."*
The pair keeps the divergence it declared. Cell 4's dead-black silhouette and cell 6's T6/T5
crispness still carry both extremes.

**R3 and R4 were spot-checked in passing and have landed.** Both run constants verified above; P3's
*"soft dished depression"* now reads *"a shallow dished depression with a hard rim"* (`:268–269`);
P2's four *"Thumbnail read at 16px"* headers are correct at `:347`, `:373`, `:398`, `:422`;
`05_pair_assignments.md:104` now reads **three** axes with the arithmetic error named; the P4/P5
blockquote prompts are converted — the only `> ` blocks left are the model-constructs disclaimer and
one pull-quote. One stale **128px** survives at `pair_2_the-light.md:106`, inside a step-06
quality-bar paragraph rather than a prompt body or a thumbnail header — cosmetic; sweep it with the
rest.

## RE-QA VERDICT: **REPAIR.**

One editing pass, not a regeneration, and smaller than the last one. The pack is materially better
than it was: three collisions resolved, both run constants unified across six files and a context
doc, veto-word discipline clean in every prompt body, and R1/R2 rebuilt from each pair's own logic
rather than paraphrased around. The soul finding stands — LOFN-PRIME, untouched.

What holds it: **a mechanical diff found a collision a manual diff did not**, in prompt bodies, on a
flair one pair owns and the other pair's own header swears it did not touch. Shipping that would
make the pack's central law — *a flair seeds a motif, never a phrasing* — decorative.

### RE-REPAIR BRIEF — two items

**RR1 — BLOCKER · `pair_5_floor-threshold.md` STEP 10 PROMPT 2 (`:543–545`).**
Rebuild the warp instruction out of what only P5 has. P4 owns *Kiln Warp* and P4 wrote the sentence
first; P4 changes nothing. P5's bow is not a door's bow — it is a **floor** seen from directly
overhead at 75–90°, the pair-exclusive camera, and a floor's warp is felt underfoot before it is
seen. The far-half/near-half tonal split is P4's reading of a wall panel at eye level; from
vertically above, a bow reads as the **throwing spiral going out of true**, or as the tread track
**crossing the bow at the wrong angle**, or as one edge of the tile lifting off the bed it was set
into. Any of those is P5's and cannot be swapped into P4. Then correct P5's header claim at
`:36–38`: either the warp stays and *Kiln Warp* moves from *"deliberately untouched"* to *"solved in
support where the physics demanded it"* with the separation argued, or the warp goes and the claim
becomes true. Both are acceptable; the current state — untouched in the header, P4's sentence in the
hero prompt — is not.

**RR2 — MINOR · the grazing key.** Decide and record. Either publish *"one grazing source, image-left,
8° above the part plane, no fill, no ambient, no second source"* as a **run constant** in
`CREATIVE_CONTEXT.md` alongside the ladder and the matte — in which case P4 `:261` and P5 `:453` are
correct as written, and every other pair's unlit prompt should be checked against it — or give P5 its
own bearing and elevation. Publishing it is the cheaper and better answer: it is a reproducibility
number, it behaves like one, and the pack has already proved that declaring a shared clause is what
stops it reading as convergence.

**Housekeeping, do with the above:** `CREATIVE_CONTEXT.md:25` *"blooms out of the clay"* →
*"rises out of the clay."* `pair_2_the-light.md:106` **128px** → **16px**.

**Re-QA scope after RR1–RR2:** re-run this same mechanical diff on P4 and P5 only, plus a one-line
confirmation of the grazing-key decision. Nothing else needs re-reading.

### Render status — unchanged and confirmed

**NO IMAGES HAVE BEEN GENERATED.** Prompts only. Image generation remains founder cost-gated and
approval is not held. All six pair files state this correctly in their status lines. Nothing in this
re-QA changed that, and nothing in it should be read as authorising a render.

### Ledger note

The first pass recommended raising **L12 to HIGH** and adding a mandatory mechanical sentence-diff as
a step-10 exit condition. This pass is the evidence for it. A *manual* sibling diff, run by three
careful pairs who each wrote a detailed anti-convergence note, missed a 17-word verbatim run in two
prompt bodies. The mechanical diff found it in under a second. **Make the diff a tool, run it at
step 10, and make its output an exit artefact — not a QA discovery.**

Second-order finding: P5's anti-convergence note separated the *doctrine* (P4's warp is an assembly
failure; P5's is an information cost) while leaving the *wording* identical — the same trap P5 itself
named at `:882–892` when it wrote *"Separating the doctrine was not enough; the sentences have to be
unable to swap."* It applied that lesson to Prompt 4 and not to Prompt 2. Doctrine separation is not
sentence separation, and only one of the two is checkable by machine.
