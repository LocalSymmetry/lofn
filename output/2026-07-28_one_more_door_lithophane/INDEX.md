---
type: pipeline_index
run_name: "2026-07-28_one_more_door_lithophane"
created: 2026-07-28T01:38:00Z
modality: image
renderer: "GPT-Image-2 (directive mode) — Sága standard"
personality: "LOFN-PRIME (AWE mode — the Kiln Attendant)"
golden_seed_anchor: "SEED 11 The Homebrew Reliquaries ⭐ + SEED 4 Moonlake Églomisé Garden ⭐"
client: "Sága Forge — ONE MORE DOOR (first PRODUCT of the studio)"
panel_transforms: ["Bridge", "Reflect"]
pairs_generated: 6
variations_per_pair: 4
total_generated: 24
total_rendered: 0
barbell: "3 ACCESSIBLE + 3 AMBITIOUS"
qa_verdict: "REPAIR"
images_generated: false
cost_gate: "founder approval not held — prompts only"
---

# Pipeline Run: ONE MORE DOOR — Lithophane Porcelain

A material bible and separable-part prompt pack for **ONE MORE DOOR**: a first-person game about
walking a corridor of doors and trading what you carry for what might be behind the next one.

**Founder commitment:** *LITHOPHANE PORCELAIN — the image only exists when lit.* Thickness **is** the
image. Unlit, a pale mute slab; backlit, a picture blooms out of apparently featureless clay. That
property is the game's **information system**, not its decoration — light is the finite thing the
player spends to learn what is behind a door before committing to it.

**Locked mood:** PATIENT DREAD, LIT FROM BEHIND.
**Signature:** six carved terraces — light steps, it never fades. Contour lines in the height field.

---

## Artifact inventory

### Phase 0 — core

| file | bytes | role |
|---|---|---|
| [`core_seed.md`](./core_seed.md) | 8,050 | Golden Seed anchor, continuity load (L1/L2/Emanating-Light/Recursive-Wonder), the emotional engine, the **five constraint axes**, the thumbnail test, neutral dispatch brief |

### Phase 1 — orchestration

| file | bytes | role |
|---|---|---|
| [`03_panel_debate.md`](./03_panel_debate.md) | 21,724 | 3 panels × 6 seats = **18 voices**, each panel carrying a Hyper-Skeptic (Concept/Medium/Context — the Visual Somatic Gate); baseline debate, **Transformation 1 BRIDGE**, **Transformation 2 REFLECT**, cross-panel synthesis, the **15 Special Flairs** |
| [`04_metaprompt.md`](./04_metaprompt.md) | 4,897 | Creative director's brief — voice, locked mood, the five attributed panel aha-moments, world context, WHAT THIS IS NOT (the negative gate), daily mandates, legibility rule, renderer |
| [`05_pair_assignments.md`](./05_pair_assignments.md) | 6,737 | The 6 concept-medium pairs, per-pair axis draws, techniques, flairs, four variation angles each, non-collision table |
| [`CREATIVE_CONTEXT.md`](./CREATIVE_CONTEXT.md) | 5,071 | **The ICB** — injected verbatim at the top of every pair; nine synthesis points, 15 flairs, the six-clause NEGATIVE GATE, legibility rule, output contract |

### Steps 06–10 — one worked file per pair (4 prompts each = 24)

| pair | file | bytes | arm | part family | steps | prompts |
|---|---|---|---|---|---|---|
| **P1** | [`pair_1_door-panels.md`](./pair_1_door-panels.md) | 41,019 | ACCESSIBLE | the door panels (hero part) | 06·07·08·09·10 | 4 |
| **P2** | [`pair_2_the-light.md`](./pair_2_the-light.md) | 43,290 | ACCESSIBLE | the carried light, in depletion | 06·07·08·09·10 | 4 |
| **P3** | [`pair_3_carried-objects.md`](./pair_3_carried-objects.md) | 40,004 | ACCESSIBLE | the toll pieces you trade | 06·07·08·09·10 | 4 |
| **P4** | [`pair_4_corridor-walls.md`](./pair_4_corridor-walls.md) | 39,036 | AMBITIOUS | tileable corridor walls | 06·07·08·09·10 | 4 |
| **P5** | [`pair_5_floor-threshold.md`](./pair_5_floor-threshold.md) | 67,211 | AMBITIOUS | floor tiles and thresholds | 06·07·08·09·10 | 4 |
| **P6** | [`pair_6_frame-failure.md`](./pair_6_frame-failure.md) | 68,664 | AMBITIOUS | armature, trim and failure states | 06·07·08·09·10 | 4 |

### QA

| file | bytes | role |
|---|---|---|
| [`QA_REPORT.md`](./QA_REPORT.md) | 29,104 | Visual Somatic Gate + structural gate + L12 convergence sweep + negative gate + material fidelity + thumbnail law + soul. **Verdict: REPAIR**, with a routed 4-item repair brief |

---

## Pairs generated

| # | pair | arm | axis draws (body · hand · light · image-of · failure) | technique | hero frame |
|---|---|---|---|---|---|
| 1 | **The Door Panels** | ACCESSIBLE | bone-china · loop-tool chatter · guttering candle · the room beyond · a corner 3 mm too thick | frontal orthographic — the reliquary portrait | **V3** (full ladder) · V4 (burn-out) as icon |
| 2 | **The Light** | ACCESSIBLE | parian · worn plaster mould · cold filament · the person who carved it · visible staples | the object held — three-quarter, close | **V3** (one rung left) |
| 3 | **The Carried Objects** | ACCESSIBLE | slip-cast eggshell · sgraffito through dark slip · distant indirect daylight · what you carry, from behind · glaze pooled and drowned a detail | parts atlas — the vitrine read | **V2** (one lit, four dark) |
| 4 | **The Corridor Walls** | AMBITIOUS | grogged stoneware · 3D-printed then badly hand-finished · another lithophane, lit in turn · a place that no longer exists · warped out of flush | the tiling study — four abutted panels | **Prompt 2** (neighbour's ordnance) |
| 5 | **Floor and Threshold** | AMBITIOUS | celadon-glazed · wheel-thrown then hand-gouged · something alive and dim, unidentified · the corridor one moment later · a firing crack | steep downward camera — the part underfoot | **Prompt 2** (thin parts first) |
| 6 | **Frame, Trim and Failure** | AMBITIOUS | bone-china as the broken material · worn plaster mould · guttering candle · the person who carved it · visible staples | exploded-parts sheet | **Prompt 4** (the scatter) |

---

## The 24 parts

| # | pair | part key | state |
|---|---|---|---|
| 1 | P1 | `door_panel_bonechina_A__unlit_raking` | unlit / shadow |
| 2 | P1 | `door_panel_bonechina_A__midreveal_thin_first` | mid-reveal |
| 3 | P1 | `door_panel_bonechina_A__lit_full_ladder` | lit, correct exposure |
| 4 | P1 | `door_panel_bonechina_A__overlit_burnout` | over-lit / burn-out |
| 5 | P2 | `light.handlamp.state_full` | lit, over-driven |
| 6 | P2 | `light.handlamp.state_half` | lit, correct exposure |
| 7 | P2 | `light.handlamp.state_ember` | lit, floor of output |
| 8 | P2 | `light.handlamp.state_dead` | unlit / raking · `promo:false` |
| 9 | P3 | `carried/toll_set__unlit_shadow__atlas_v1` | unlit / shadow |
| 10 | P3 | `carried/toll_set__daylight_transmission__atlas_v1` | lit, deliberately incomplete |
| 11 | P3 | `carried/toll_03__glaze_pool_macro` | lit, correct exposure |
| 12 | P3 | `carried/toll_02__traded_chipped` | mid-reveal, both lights |
| 13 | P4 | `wall.strip.grog.unlit` | unlit / shadow |
| 14 | P4 | `wall.strip.grog.neighbourlit` | one panel lit / mid-crossing |
| 15 | P4 | `wall.panel.grog.transmit_fail` | lit and failing |
| 16 | P4 | `wall.seam.grog.macro` | mid-reveal, edge-on |
| 17 | P5 | `floor.tile.celadon.unlit` | unlit / shadow + glaze |
| 18 | P5 | `floor.tile.celadon.underlit` | mid-reveal, thin parts first |
| 19 | P5 | `floor.threshold.celadon.joint` | lit on one side only |
| 20 | P5 | `floor.tile.celadon.crack_split` | lit and split |
| 21 | P6 | `frame/armature_01__empty__candle_shadow` | unlit / shadow |
| 22 | P6 | `frame/armature_01__stapled_panel__candle_transmission` | lit / transmission |
| 23 | P6 | `trim/profile_section__rebate_and_body` | mid — section |
| 24 | P6 | `panel/carver_01__shatter_fragments__backlit_atlas` | lit, all seven fragments |

---

## Panel process, briefly

Three panels of six — **Concept** (what the image is), **Medium** (how the material behaves),
**Context & Marketing** (whether anyone will ever see it) — each seeded with a Hyper-Skeptic, which
is the Visual Somatic Gate's veto bench. Two transformations were run: **BRIDGE** (group choice) and
**REFLECT** (skeptic choice).

The debate's five surviving aha-moments, carried into every pair:

1. **The picture is the missing clay** — bright is where material was taken away. Carve out, never
   paint on.
2. **Two images that need not agree** — unlit reads by shadow under raking light; lit reads by
   transmission. When they disagree, the player has learned something and it cost them.
3. **Light steps, it never fades — six values.** A 2–6 mm body gives ~6 carvable brightnesses, so
   transmission arrives in hard-edged terraces like a topographic map. **The signature is contour
   lines**, baked into the height field, never a post-process filter.
4. **Both extremes are blindness** — unlit says nothing; over-lit burns to white. There is a correct
   exposure per panel.
5. **Be literal, not tasteful** — *"you have three matches"* beats *"attention is the scarce
   resource."*

---

## QA summary

**Current verdict: REPAIR (2 items)** — R1–R4 are closed; a mechanical sentence-diff found one new
blocker-grade collision. See the **RE-QA** section at the foot of [`QA_REPORT.md`](./QA_REPORT.md).

### Gate table — as of the RE-QA pass

| gate | first pass | RE-QA (post-R1–R4) |
|---|---|---|
| Structural — 6 pairs × 4 prompts, steps 06–10, 07/09/10 present | PASS · 6/6 canonical | not re-run (passed) |
| Cross-pair convergence (L12) | **FAIL** · 3 collisions | **FAIL** · original 3 CLEARED; **1 new blocker-grade + 1 minor** found by mechanical diff |
| Negative gate — 6 clauses | PASS in substance · 2 lexical hazards | **PASS** · hazards cleared |
| Veto words used positively in a prompt body | (not separately gated) | **PASS** · zero, all six pairs |
| Six-terrace signature + index convention | **FAIL** (P2, P3 inverted) | **PASS** · one direction, T1 darkest → T6 brightest, all 6 pairs + `CREATIVE_CONTEXT.md` |
| Matte / extraction contract | **FAIL** · three navies | **PASS** · one value, `#0B1220`, zero live exceptions |
| Unlit + lit per door-family part | PASS | not re-run (passed) |
| Thumbnail law — ≥1 prompt per pair at 16 px | PASS · 6/6 (P2 under-specified) | **PASS** · P2's four headers now read 16px |
| Soul — LOFN-PRIME (AWE, the Kiln Attendant) | **PASS**, emphatically | unchanged |

### R1–R4: **closed.** Rebuilt from each pair's own logic, not paraphrased around.

- **R1** ✔ P6's Prompt 4 atlas is now ordered *against the break* (no fragment beside a piece it
  broke from); cell 2 is a drilled blind staple pocket in P6's own vocabulary; the false
  *"CLEAN against P3 and P4"* claim is quoted and withdrawn in P6 §6.
- **R2** ✔ *"brighter than T6 could ever be"* is live in **P4 only**. P5's Prompt 4 now builds its
  brightness on the dashed line and the 3 mm offset — *"not empty — it is wrong, and it is
  pointing."* P6's cell 7 out-darkens instead of whiting out, as its §10 rule 3 promised.
- **R3** ✔ One matte hex, one terrace direction. Verified by exhaustive grep.
- **R4** ✔ P3's *"brown bloom"* and *"soft dished depression"* are gone from the bodies; P2's
  thumbnail headers read 16px. Housekeeping items (axis count, mandate scoping, likeness constant,
  blockquote→fenced) all landed.

### RR1–RR2 — the two items now outstanding

- **RR1 · BLOCKER** — `pair_5_floor-threshold.md:543–545` (Prompt 2, the pair hero) shares **17
  identical words** with `pair_4_corridor-walls.md:393–394` (Prompt 3): *"hold the far half a full
  terrace darker than the near half, with the change happening along [a/one] hard line where the bow
  turns."* Both are prompt bodies. *Kiln Warp* is assigned **exclusively to P4**
  (`05_pair_assignments.md:60`) and P5's own header at `:36–38` lists it under **"deliberately
  untouched (owned elsewhere, collision risk)"** — so the collision comes with a false declaration
  attached. Rebuild P5's warp from its overhead camera; correct the header claim either way.
- **RR2 · MINOR** — P4 `:261` and P5 `:453` share the grazing-key setup (*one source, image-left, 8°
  above the plane, nothing else*), which is mechanical in kind but **declared nowhere**. Publish it
  as a run constant in `CREATIVE_CONTEXT.md` or give P5 its own bearing.
- **Cosmetic, sweep with the above:** `CREATIVE_CONTEXT.md:25` *"blooms out of the clay"* →
  *"rises out of the clay"*; `pair_2_the-light.md:106` stale **128px** → **16px**.

**Re-QA scope after RR1–RR2:** re-run the mechanical diff on P4 and P5 only, plus one line recording
the grazing-key decision. Every other gate has now passed.

**Method note for the ledger.** The RE-QA convergence check was mechanical, not editorial: 51,349
tokens across the six pair files, every cross-file 12-gram indexed and extended to its maximal run —
**178 shared runs of ≥12 words**, of which 176 are the deliberately-shared clause families
(negative gate, terrace ladder, matte/extraction contract, the carver's-likeness constant, pipeline
scaffolding) and **2 are genuine motif convergence**. The manual sibling diffs written by three
careful pairs missed the 17-word run entirely. **L12 → HIGH, and the diff should be a step-10 exit
artefact rather than a QA discovery.**

---

## Render status

**NO IMAGES HAVE BEEN GENERATED.** Prompts only — image generation is founder cost-gated and approval
is not held. All six pair files state this correctly, and the RE-QA pass re-confirmed it. Nothing in
either QA pass authorises a render.

Because the pack is **REPAIR**, it is **not yet at the founder cost-approval decision**. Close RR1
and RR2 first — two targeted edits in two files, no regeneration.

Two measurable render-gate tests were supplied by the pairs and should be run first when rendering
is approved:

- **P2** — reject any V1/V2/V3 output where the navy field is brighter within 50 px of the silhouette
  than it is at the frame edge (bloom detector).
- **P5** — reject any output showing a reflection of the light source in the celadon (gate-4, wet
  gloss).

---

*#lofn/image · #lofn/pipeline/2026-07-28_one_more_door_lithophane · #saga/forge/one-more-door*
