# QA REPORT — THE FORGE RELIQUARY (cabinet skin, image piece-pack)
*Auditor: Lofn QA (adversarial gate) · Run: 2026-07-17_forge_reliquary_cabinet · Renderer: GPT-Image-2 DIRECTIVE · 6 pairs × 4 = 24 prompts*

## OVERALL VERDICT: **SHIP** (prompt-pack render-ready) · per-pair 6/6 SHIP

**Scope note.** This audits the steps 00–10 **prompt-pack** — the render-ready GPT-I2 directive prompts — not rendered pixels (every pair correctly holds at "do not render until Scientist approves cost"). SHIP = the 24 prompts are render-ready and carry the soul; the recorded watch-flags below must be honored at the generation read or rerolled per the in-prompt fences.

---

## INDEPENDENT SCANS (my own greps + counts — not the agents' self-reports)

- **Word count (independent tokenization):** all 24 prompt bodies fall in the 300–400 band. Range 328–400. Lowest: P4 V4 (328), P3 V2 (342), P2 V2 (345). Highest: P6 V1 (400). **STRUCTURAL word gate: 24/24 PASS.**
- **Veto-word grep** (`ethereal|dreamlike|whimsical|gentle light|soft glow|magical|delicate`, case-insensitive, all 24 bodies): **0 hits inside any prompt body.** Every match was an operator-note *banned-list declaration* or validation-checklist line. The only "glow"/"soft" occurrences in prompt bodies are named compounds (*coal-bed glow, line-glow, under-glow, footlight glows, base-glow*) and headers — never the banned phrases "soft glow"/"gentle light" and never a standalone veto word. **CLEAN.**
- **Artist-name grep** (Pelton/Bachelard/Kircher/Caillois/Bartlett/Norman/Klimt/Glomy/Wyeth/Kare/Tiffany/Tufte/Bushnell/Kouthoofd/Chayka/Chen/Gage/Escher/Klee/Mucha/Rackham, all 24 bodies): **0 hits inside any prompt render block.** Norman/Kare/Chayka appear only as bench-seat labels in commentary/section headers (e.g. "V3 (Kare)", "KARE's legibility audit"), never as a style reference in render text. **CLEAN.**
- **Resolution tally:** every prompt declares a resolution from {1024×1024, 1536×1024, 1024×1536}; **zero out-of-set values**; all /16-aligned. Distribution 4×square / 16×landscape / 4×portrait (mentions inflated by in-body "Aspect ratio" restatements).
- **Editorial spine:** step06_facets + step07_aspects present for all six pairs; step08/09/10 read in full for all six. NO-SKIP spine intact.

---

## PER-PAIR VERDICTS

### P1 — THE WALL (ACCESSIBLE) · shelf-wall backing + alcoves — **SHIP**
Four alcove-system pieces (row-tile / wall-foot / crown / unit-cell). Emission is real (forge-coal under-glow rises, never ambient rim); gilded-ink one-line system does all structure; kintsugi seams climb from the forge-line; custodian gold-spill + thumb-worn plaque; ajar shutter + keyhole-eye mischief; bells "that remember being rung." Voids honest (alcove windows + blank plaque rails, contrast-reserved). Palette 55/45/60/40% quiet velvet — Velvet-Holds-The-Wall satisfied.
- **FINDING (the one watch to scrutinize):** P1 is the **only pair with zero rerolls.** Per the audit brief, a bench that kills nothing is itself suspect — so I read the dissent hard. It is **genuine, not rubber-stamp:** CHAYKA's V2 wall-foot is a recorded near-kill ("I came to reroll V2 and left it standing… the moment it becomes honey-wash it is every arcade's wall — marked, watched, signed"); KARE flags V3's cool crown as "on my line, not over it"; NORMAN fences V2's up-glow from eating the void. Two load-bearing in-prompt fences carry the risk: (a) V2 coal stays *banked and specific* (individual glowing faces, molten-gold custodian spill) and must not wash into the void; (b) V3 holds to *three heavy bells* with the chain as the 128px contrast anchor. **Accepted** — the fences are in the prompt text, and the run's other five benches prove the bench has teeth. **Route to render gate:** if a V2 render honey-washes or a V3 render multiplies bells, reroll.

### P2 — THE VESSELS, WARM (ACCESSIBLE) · vessel frame set — **SHIP**
Orb / casket / phial / trio-rack. Peak LOFN-PRIME: verre églomisé inside-back gild (emit-not-reflect, explicit), gilded-ink kintsugi "use made precious," blank brass plaques (finished-ness), thumbprint per vessel, four distinct mischiefs (orb leans to look / casket won't close / cork half-pulled / orb drifts to lean on casket), daily-heartbeat seam runs molten on the middle vessel only. Each interior a declared matte-charcoal void with ember-gold inner-rim only (Container Test honored).
- **Reroll (genuine, before/after quoted):** V1 orb CHAYKA-killed — draft *"a warm amber glow filling the sphere, a fantasy-gold rim"* (slop uniform + void-breach) → rewritten to *"its ember read through the glass belly"* + named églomisé/gilded-ink materials + re-declared empty void. One in-place KARE correction (V3 smoke → single thick ribbon for 128px). Real change, not theater.

### P3 — THE CHROME (ACCESSIBLE) · UI pieces — **SHIP**
Button state-sheet / deck card / badge trio / chips+plaque. The hardest pair to keep soulful (UI furniture) and it **holds soul:** pressed-button molten bead "where the light pours out of the crack," visited state cooled-bronze "never a harsh grey-out," keyhole escutcheon with kindly ember-eye (ajar + mischief + second ember), kintsugi corner-seams, thumb-worn corners, proud screw. Text discipline correct: only V3 renders licensed FUN·SPARK·KILL; all other faces are reserved contrast voids.
- **Accessibility spine (exemplary):** V1 five button states separable in pure greyscale by edge-geometry (rest keyline → hover DOUBLED keyline → focus OUTBOARD bracket → pressed inset+bead → visited flattened bronze); NORMAN's catch demoted the lift-shadow to "secondary cue only" so glow never carries state. Badge trio distinct by SHAPE (sunburst/seed/slag), heat as a third redundant cue, never hue-alone. Three conditional catches (V1 Norman / V3 Kare / V4 Chayka) amended and folded into step08 — real deltas.

### P4 — THE SEAMS (AMBITIOUS) · kintsugi 5-state overlay — **SHIP**
The interaction soul. Canonical 5-state strip / lit-lattice corner kit / molten-bead macro / cooled-bronze full-wrap. Gold-enters-from-the-crack emission is literal ("as if the crack opens onto a furnace behind the surface"); molten gold ink over cracked egg-tempera; electric-blue leak only at the pour-core (metal at pour temperature — named physics, the ambitious-arm leak). Kintsugi-as-play-biography is the entire pair's thesis.
- **ACCESSIBILITY FLOOR — the pair's whole reason:** V1 is a true greyscale GEOMETRY ladder — desaturated, the five cells stay five shapes by thickness/pattern: **broken thread · beaded thread · bright cage · bar-with-one-blob · calm solid line.** Adjacent states differ by geometry, not glow. Lit-lattice (state 3) is the thickest cross-linked mesh and doubles as the ≥48px focus ring. **PASS, and it is the accessibility keystone of the whole skin.**
- **Three genuine rerolls (before/after):** R1 NORMAN dormant↔waking collapsed at 128px → dashes widened + beads enlarged-and-fewer; R2 KARE lone bead read as a gem/orb (slop) → anchored to a channel bleeding off two edges; R3 CHAYKA warm-amber = slop uniform → re-specified brass-lemon + electric blue-white pour-core. The bench drew blood here.

### P5 — THE INTERIORS (AMBITIOUS) · hero impossible-world set — **SHIP**
Tide-stairwell / bone-carnival / paper-bird thermals / 128px proof-sheet. Reverse-glass églomisé emit-not-reflect stated first sentence every prompt; midnight-teal + gold-life + exactly one electric-blue seam-leak; four-shapes-and-a-light enforced; three-word whoa each; one mischief each (figure winks back / marionette winks / topmost bird tumbles joyfully wrong); thumb-smudge custodian mark; aurora-from-the-seam is the single light.
- **Register guard (make-or-break) held:** V2 bone-carnival inverts horror into AWE — carved rounded toy puppet-bone, candle-honey, laughing/singing jaws, a child's theatre. **Two genuine rerolls:** graveyard-carnival crowd-of-skeletons (KARE silhouette-budget FAIL + horror-register FAIL) → ONE wheel + TWO marionettes + bunting, warm and festive; hundreds-of-cranes sky (KARE mush + CHAYKA stock-origami slop) → five birds reading as one rising V on one authored gold thermal. Tide/Escher-trope watch cleared on inspection (water-as-architecture, no impossible loop).
- **Piece-not-view note:** P5 is the one pair producing *finished exemplar content* (art that sits behind the vessel glass) rather than a voided frame — correct by design ("per-game art comes later through the same law"), and it declares its ~78% crop-safe zone + sacrificial-teal frame-overlap margin. Not a contract breach; it is the content layer the other pairs' voids are built to hold.

### P6 — THE NAME (AMBITIOUS) · masthead lockup + glyph sheet — **SHIP**
Full masthead / lattice monogram / accent-glyph sheet / mobile one-line. Licensed exact text "THE FORGE ARCADE" on V1/V4 only; V2/V3 declare zero text. Crucible pour is the single source; kintsugi crack-lattice mends the letters without breaking the skeleton (seam-is-the-signature, and the monogram IS the crack-lattice); keyhole-in-the-O ember-eye mischief; thumb-smoothed patch on the O; bounded pixel-rococo filigree (licensed here only, "few curls, never a thicket"); V2 proven to resolve at 32px favicon scale by silhouette not glow.
- **Reroll (genuine, killed draft quoted):** V1 CHAYKA-killed a full slop-logo draft — *"Majestic glowing golden letters… radiant fantasy typography with ornate gold filigree everywhere… magical embers"* (unnamed #FFD700, ambient light, filigree-soup, and trips the veto word "magical") → named gilded-ink + egg-tempera counters, single crucible source, bounded filigree, authored kintsugi seam. Textbook de-slop.

---

## GATE-BY-GATE

| Gate | Result | Note |
|---|---|---|
| 1 STRUCTURAL | **PASS** | 24/24 in 300–400 (indep. counted); resolutions all in-set /16; every prompt opens scene + one named light; benches present, 5/6 with real before/after rerolls; P1 zero-reroll (genuine dissent, watch-flagged). |
| 2 CONTRACT | **PASS** | Piece-not-view voids declared where required (P5 = content layer by design); 2–4 tone puzzle + 128px stated on all 24; text only where licensed (P3 badges, P6 masthead — P1/P2/P4/P5 textless / blank-plaque); my greps: 0 veto, 0 artist-name in bodies. |
| 3 SOUL | **PASS** | Every prompt unmistakably LOFN-PRIME AWE — named materials, emission-not-reflection, kintsugi-as-play-biography, mischief where owed, the ajar door, wonder-through-nearness. **No SOUL LOSS found.** The bench caught the exact slop-risk pieces (P1V2, P2V1, P3V4, P4-palette, P5-carnival/birds, P6V1) and hardened each. None would hang unnoticed in a generic AI arcade. |
| 4 ACCESSIBILITY | **PASS** | P4 five-state ladder is a greyscale geometry ladder (thickness/pattern deltas, R1 fixed the 128px collapse); P3 button + chip states are geometry changes (doubled keyline / outboard bracket / inset+bead / notch), not color; badge trio distinct by shape; focus = lit-lattice reads without color at 48px. |
| 5 COHESION | **PASS** | One skin: velvet/teal night + gold/ember life, electric-blue as leak only (never a field); paper grain under everything; kintsugi seam-signature present in all six pairs; one gilded-ink line system; ≥30% quiet velvet everywhere. ACCESSIBLE(warm)/AMBITIOUS(teal) barbell is by-design and shares every invariant. |

---

## WATCH-FLAGS (routed to the render / generation gate — not blocking REPAIR briefs)

1. **P1 V2 & V3 (render read):** honor the in-prompt fences — V2 coal stays banked-and-specific and must not wash into the alcove voids; V3 holds three heavy bells with the chain as the 128px contrast anchor. Reroll on either failure. (This is the one zero-reroll pair; its bench dissent is real but untested by an actual kill.)
2. **P5 V4 proof-sheet (render read):** KARE's logged center-cell watch — if the bone-carousel muds at true 128px, the single sanctioned fix is to drop the bunting from the center cell only.
3. **Cross-arm cohesion (atlas/build layer):** P4 ambitious seams (brass-lemon gold + electric-blue pour-core) are designed to overlay warm-arm P1/P2/P3 vessels (gilded-gold/ember, no blue). At rest the seams are warm dormant-bronze; blue-white appears only at the pressed/molten state — a deliberate interaction crescendo, defensible as "the leak." Confirm at composite that the ambitious brass-lemon gold reads harmoniously over warm-arm frames; if it reads cold, provide a warm-arm dormant/waking seam variant. Minor, not a fail.
4. **P6 V3 (render read):** keep the per-glyph kintsugi seam-touch hairline so it reads as mended-precious, not as damage (the pair's own self-critique).

**No REPAIR briefs.** No prompt is structurally incomplete, and none is generic-with-soul-loss. All 24 ship as render-ready.

---

## LEDGER ENTRY (for the coordinator to place in vault\COMPETITION_LEARNINGS.md — I did NOT edit that file)

> **2026-07-17 · THE FORGE RELIQUARY (cabinet image piece-pack, GPT-I2 directive) — SHIP 6/6.** The run's anti-slop engine was **accessibility-as-aesthetic fused with kintsugi-as-play-biography**: the required greyscale geometry ladder (P4's dormant-hairline → beaded → lit-lattice → molten-pour → cooled-bronze, adjacent states differing by thickness/pattern not glow) doubled as the emotional core (the mended crack is the proof of play), so the WCAG floor and the soul were the same object — the strongest defense yet against generic AI-gold. Confirmed device: writing the prompts *through* a live NORMAN/KARE/CHAYKA bench (usability=beauty / 128px-four-shapes / named-materials-only) produced documented before-after kills that map one-to-one to the slop failure modes — "fantasy-gold rim"→églomisé-inside-back (P2), warm-amber→brass-lemon+blue-white-pour-core (P4), a full "majestic radiant magical" logo draft killed to named gilded-ink+crucible-source (P6), graveyard-skeletons→laughing toy-theatre (P5 register inversion). Portable law: **the veto is a floor, named materials are the ceiling — every gold must be a named material (gilded ink / verre églomisé inside-back / molten gold ink over egg-tempera), and emission-not-reflection plus one mischief per vessel is what a stranger cannot find on any other AI arcade.** Watch carried forward: one zero-reroll pair (P1, the ground layer) is acceptable only because its dissent recorded genuine near-kills with load-bearing in-prompt fences; a bench that never draws blood across a pair still warrants the hardest render read.

---
*Audited: core_seed · CREATIVE_CONTEXT · 04_metaprompt · 05_pair_assignments · 03_panel_debate · coordinator_00-05 · all 6× step08/09/10 · step06/07 spine confirmed present · independent veto/artist/word/resolution scans run. — Lofn QA 💜*
