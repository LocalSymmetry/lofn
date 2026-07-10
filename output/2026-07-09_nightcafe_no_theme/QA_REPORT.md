# QA REPORT — "The Answered Room" (NightCafe #1365, No-Theme Thursday, 2026-07-09)

**Judge:** Lofn adversarial Vision QA, clean context (did not generate this work).
**Payload under judgment:** 6 pairs × 4 variations = 24 final Flux prompts (`pair_0[1-6]_step10_revision_synthesis.md`).
**Standard:** frozen `CREATIVE_CONTEXT.md` (icb_sha 7747f51d…, 21,814 B) · `06_vision_handoff.md` §QA CONTRACT · `vault/VISION_QA_DEPTH_AUDIT.md` · `vault/gates.yaml`.
**Framing:** default-to-REPAIR; a structurally complete but generic piece is SOUL LOSS. Every measurement below was re-run by the judge from disk (word counts, string gates, n-gram overlap) — self-reported numbers were NOT trusted.

## OVERALL VERDICT: **SHIP-WITH-REPAIRS**

The set is structurally immaculate (24/24 pass word-count, opener, haze, medium, splinter, tell, contact, palette, rest-zone, axis-corner, and register gates), both coordinator flags resolve in the work's favor (D1 **EARNED**, D2 **COHERENCE**), the Somatic bloc passes all six pairs, and there is a strong, low-render-risk venue entry. It is **not** a clean SHIP: Pair 05 is missing a mandated deliverable (its ≤5-word title candidates — dropped in a step-07→step-10 handoff regression), and two advisory repairs improve render-robustness and set-curation. The repairs do **not** touch the entry pick, which lives in Pair 04 and is fully clean.

---

## Depth Audit (Vision)

| Step | File | Lines | Min Req | Status |
|------|------|-------|---------|--------|
| 10 | pair_01_step10 | 98 | 60 | PASS |
| 10 | pair_02_step10 | 83 | 60 | PASS |
| 10 | pair_03_step10 | 284 (incl. embedded ICB) | 60 | PASS |
| 10 | pair_04_step10 | 88 | 60 | PASS |
| 10 | pair_05_step10 | 74 | 60 | PASS |
| 10 | pair_06_step10 | 75 | 60 | PASS |

Upstream chain spot-checked for provenance: `pair_05_step0[6-9]` all present (07 carries the title candidates that step-10 later dropped — see Repair R1).

## Cardinality Audit

| Artifact | Count | Required | Status |
|----------|-------|----------|--------|
| Final pairs (Step 10) | 6 | 6 | PASS |
| Final prompts (Step 10) | 24 | 24 | PASS |
| ACCESSIBLE / AMBITIOUS barbell | 3 / 3 | 3 / 3 | PASS |

---

## A. STRUCTURAL GATE AUDIT (judge-recounted, all 24)

Word counts re-measured two ways: **naive** = whitespace-split tokens; **words** = tokens containing ≥1 alphanumeric (word-processor convention; standalone em-dashes excluded). String gates (haze substrings incl. "delicately"/"floating"; imperative openers; camera-mm/f-stop/Kelvin) scripted across each prompt body.

| Prompt | words | naive | 80–150 | noun-first / present | haze | imperative | camera/Kelvin | artist | medium 1st-third | emotion shown |
|---|---|---|---|---|---|---|---|---|---|---|
| P1-V1 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ gum bichromate | ✓ |
| P1-V4 | 148 | 150 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P1-V2 | 150 | 150 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P1-V3 | 148 | 150 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P2-V2 | 150 | 150 | ✓ | ✓ | none | none | none | none | ✓ autochrome | ✓ |
| P2-V4 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P2-V1 | 150 | **151** | ✓* | ✓ | none | none | none | none | ✓ | ✓ |
| P2-V3 | 150 | 150 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P3-V1 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ cyanotype-over-platinum | ✓ |
| P3-V3 | 145 | 145 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P3-V4 | 147 | 147 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P3-V2 | 148 | 148 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P4-V4 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ mezzotint | ✓ |
| P4-V1 | 147 | 147 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P4-V2 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P4-V3 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P5-V2 | 150 | 150 | ✓ | ✓ | none | none | none | none | ✓ verre églomisé | ✓ |
| P5-V4 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P5-V3 | 149 | 149 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P5-V1 | 150 | 150 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P6-V1 | 149 | 150 | ✓ | ✓ | none | none | none | none | ✓ urushi + maki-e | ✓ |
| P6-V2 | 141 | 141 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P6-V4 | 135 | 135 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |
| P6-V3 | 148 | 148 | ✓ | ✓ | none | none | none | none | ✓ | ✓ |

**Structural result: 24/24 PASS.** No prompt exceeds 150 real words; none is below 80; no haze substrings (verified incl. "delicately"/"floating"); every opener is a noun phrase in present tense; no camera specs, no Kelvin, no artist names in any prompt body; medium named word-1 in all 24; no children and no named emotions anywhere in prompt bodies (the only "child"/"girl"/emotion hits sit in negative-prompt notes, the embedded ICB in pair 03, and pair 04's "what-to-avoid" adversarial note — never in a prompt).

**\* Boundary FLAG (non-blocking):** P2-V1 = **151** tokens by naive whitespace split but **150** real words (one standalone " — " dash token). Its self-check reported 150, which matches the word convention. If any downstream validator uses a naive splitter it will report 151 and trip the 150 cap. Recommend trimming one word from P2-V1 to clear the boundary under either tokenizer (e.g., drop "silver hair pinned low" → "hair pinned low"). Not a failure — the real count is compliant.

---

## B. SEED GATE AUDIT (per pair)

| Gate | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| Splinter visible (named, unrepaired) | ✓ dark attic window | ✓ empty seed tin | ✓ dusty 2nd mug | ✓ dry harbor / keeled boats | ✓ empty chair + half-scraped paint | ✓ shuttered stall / empty hook |
| Un-posable tell (D-corner) | ✓ hand off rail (D3) | ✓ hem mid-step (D2) | ✓ exhale at last (D4) | ✓ punch mid-fall (D1) | ✓ letter loosening (D6) | ✓ shadow-against-light (D5) |
| Contact point (guest touches world) | ✓ wheat to doorsill | ✓ petals in fence wire | ✓ **moon dents the towel** | ✓ brine wets cobbles | ✓ foam line at sill | ✓ **pole bows under real weight** |
| Two-temperature palette only (E-corner) | ✓ aubergine/gold + pearl | ✓ grey-green/rose-copper | ✓ umber/verdigris + pearl | ✓ prussian/saffron | ✓ teal/amber | ✓ slate/persimmon |
| ~10% rest zone described | ✓ open wheat/land | ✓ pearl sky band | ✓ plaster / cupboard | ✓ silver sky band | ✓ storm-held sky | ✓ lacquer-dark upper band |
| Two-mass silhouette @128px | ✓ (figure lighter mass) | ✓ | ✓ | ✓ **strongest (V4)** | ✓ | ✓ |
| Three-beat read traceable | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Title candidates ≤5 words | ✓ 4 | ✓ 3 | ✓ 3 | ✓ 3 | **✗ MISSING** | ✓ 3 |
| One-breath retell stated | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Seed result: PASS for all except P5's title-candidate deliverable** (see Repair R1). Contact points are unusually strong in P3 (a celestial body *denting* a folded towel) and P6 (the pole *bowing under real weight*, the figure's hands steadying it) — these convert the impossibility from painted-on to load-bearing, exactly the Estrangement Officer's demand. The only soft two-mass balance is P1, where the guest (the glowing house) outmasses the gate-figure; the pair itself flagged this and welded the figure tighter in V2 ("in the doorway's reach").

---

## C. SET GATE AUDIT

**Axis Latin square — honored in content, not just labeled:**

| Pair | A hour | B guest | C medium | D tell | E palette | Content match |
|---|---|---|---|---|---|---|
| P1 | A2 golden-hr | B4 ancestral interior | C1 gum bichromate | D3 released hand | E2 aubergine/gold/pearl | ✓ (reads dusk-leaning — see hour note) |
| P2 | A4 after-rain | B2 garden out of place | C4 autochrome | D2 hem mid-step | E3 grey-green/rose-copper | ✓ the mandated BRIGHT pair |
| P3 | A5 lamp-meets-day | B3 celestial domestic | C6 cyanotype/platinum | D4 exhale | E5 umber/verdigris/pearl | ✓ |
| P4 | A3 mid-eclipse | B1 forbidden migration | C3 mezzotint | D1 object mid-fall | E4 prussian/saffron | ✓ |
| P5 | A6 pre-storm | B6 sea where none belongs | C2 verre églomisé | D6 letter loosening | E1 teal/amber | ✓ |
| P6 | A1 blue hour | B5 light as creature | C5 urushi+maki-e | D5 shadow against light | E6 slate/persimmon | ✓ |

- **No axis option reused across pairs:** every axis A–E is a perfect permutation of {1..6}. **PASS.**
- **Variation-angle lenses UNIQUE per pair — the strongest anti-template signal in the run:** P1 **spatial** (rooms of the house) · P2 **temporal** (bud→riot→fruit→release) · P3 **proximity** (touch→yard→sky→faith) · P4 **population** (crest→mass→straggler→pause) · P5 **depth** (approach→arrival→interior→totality) · P6 **behavioral** (landing→gift→invitation→staying). Six different generative logics, no shared global frame-set. **PASS.**
- **Hour spread:** golden-dusk · after-rain-day · lamp-dawn · eclipse-silver · pre-storm-night · blue-hour. ≥1 genuinely bright pair = **P2 (PASS)**, P3 bright-adjacent. *Observation (non-blocking):* the set skews dark — the entire AMBITIOUS arm (P4/P5/P6) is nocturnal/eclipse and P1 leans dusk, so only P2 is unambiguously bright and P1's A2 "last three minutes of golden hour" renders closer to blue-dusk than golden glow ("violet-toned dusk," "darkening"). Passes the letter of the gate; the warm-light venue preference is carried mainly by P2/P3 and by two-temperature accents elsewhere.
- **Register purity — PASS.** ACCESSIBLE (P1/P2/P3) contain **no** holographic-foil/pixel-sort language (verified by scan; P2 and P3 explicitly declare foil/glitch "banned and absent"). AMBITIOUS carry intense-register devices: P4 "pixel-sorted filigree" + "refractive star-fields"; P5 "holographic sheen" + "suspended galaxies" + gold/palladium verso; P6 structural maki-e gold filigree + refractive scale-detail + "one radiant mass." *Note:* P6 is the softest-intensity AMBITIOUS — it deliberately bans bloom/glow and reads warm/intimate rather than cosmic; it qualifies via the "baroque abundance / one radiant mass" clause, and its gentler register is a distinctiveness asset (it does not look like P4/P5), not a violation.
- **ArtCore provable per pair — PASS.** All six carry a specific (non-generic) visual-world sentence + signature device + seed-derived weirdness section; each names a real medium-specific physical signature (brush borders / starch grain / platinum steps / rocker burr / verso leaf / togidashi edges).
- **Cross-pair distinctiveness — PASS (both flagged twin-risks cleared by measurement):**
  - **P1 vs P3 (domestic-interior):** **zero** shared 4-grams (judge-measured). Different guest (ancestral house vs moon), medium, hour, vantage (exterior-of-glowing-house vs interior-kitchen), figure (Halyna 63 farm woman vs Rosa 34 nurse). Only a **motif echo** remains: both trade on "home was not empty / waited up" (P1 "the house"; P3 "waited up," "empty house"). Not a noun-swap twin; a set-curation note (see Repair R3).
  - **P4 vs P5 (water-drama):** 5 shared 4-grams, and **all five are the figure-grammar scaffold** ("small in the lower third, face three-quarter") — the mandated house skeleton, not concept-twinning. Different guest (whale pod migrating vs the Pacific standing still), physics, medium, palette, figure, reveal-engine ("your street is the sea's road home" vs "the sea stands at your sill and asks first"). Not a twin.

---

## Prompt Density Spot-Check (7-element checklist, ≥3/pair)

| Pair | Prompt | words | 7 elements present | Status |
|---|---|---|---|---|
| P1 | V1 | 149 | emotional seed · medium-agent · material · lighting · 3-tier focal · palette · incompleteness — all 7 | PASS |
| P1 | V4 | 148 | all 7 | PASS |
| P2 | V2 | 150 | all 7 | PASS |
| P2 | V1 | 150 | all 7 (rain-cue render note, R2) | PASS |
| P3 | V1 | 149 | all 7 | PASS |
| P3 | V3 | 145 | all 7 | PASS |
| P4 | V4 | 149 | all 7 | PASS |
| P4 | V2 | 149 | all 7 | PASS |
| P5 | V2 | 150 | all 7 | PASS |
| P5 | V4 | 149 | all 7 | PASS |
| P6 | V1 | 149 | all 7 | PASS |
| P6 | V4 | 135 | all 7 | PASS |

---

## D. COORDINATOR FLAG ADJUDICATIONS

### D1 — CEILING-HUG: **EARNED** (not padded)

**Measurement:** by judge word-count, **22 of 24** prompts sit at ≥145 words (slightly higher than the coordinator's flagged 21); only P6-V2 (141) and P6-V4 (135) sit lower. No prompt exceeds 150.

**Sampled 6+ across pairs, ruled per prompt with evidence:**

- **P1-V1 — EARNED.** Every clause maps to a distinct doctrine element: "*the lit rooms receding doorway after doorway, deeper than the house is wide, each jamb found sharp while the dusk loses its edges*" is guest-impossibility + zoom-reward + sharpest-where-least-believable in one breath. No epithet floats free.
- **P2-V2 — EARNED.** "*rain beads lifting off the soil in vertical grain-streak threads inside that band only*" carries the impossibility, the medium (grain), and the containment (band-only) simultaneously; the three botanicals ("dahlias, marigold ropes swagged, wheat rising") are the *Full Riot* concept, not decoration.
- **P3-V1 — EARNED.** "*the full moon rests small as a mixing bowl, denting the folded dish towel, maria and craters etching-crisp, the sharpest zone in frame*" is scale-poetry + contact + craft-law + splinter-adjacency with zero slack.
- **P4-V4 — EARNED (exemplary).** Even the reveal does double duty: "*Fare accepted*" names her trade (ticket conductor) and the sea's granted passage. "*one barnacle cluster holds a pinpoint refractive star-field, the sharpest detail in the print*" is the intense-register zoom-reward earning its words.
- **P5-V2 — EARNED.** The triple-lock "*stands calm and a hand higher than the land … stopped ruler-straight at the sill, not one drop crossing*" is three non-redundant impossibility anchors (height, straightness, non-crossing); "*One white line of foam … does the work of the whole wave*" is deliberate economy, the opposite of padding.
- **P6-V1 — EARNED.** "*The pole bows under its real weight; her hands steady it*" spends words making light physically heavy — the load-bearing contact point, not filler.

**Corroboration that the cap-hug is deliberate, not inability:** the same authors wrote **P6-V2 at 141 and P6-V4 at 135** when the concept (an intimate flake-on-sleeve close-up; a quiet sleeping-creature register) needed less — proof they trim when the frame asks, and pack to 149–150 only where the world rewards density.

**Residual (carried, not a padding failure):** the ceiling-hug is a *render-execution* risk — reveal lines sit last and die first if Flux truncates a tail clause. Every pair self-flagged this. The render session must verify final-line survival (see venue render strategy).

### D2 — HOUSE-SKELETON ECHO: **COHERENCE** (one portfolio voice, six worlds — not one template)

The shared skeleton is **real and measurable**: ¾ face in all six; a final-line reveal in all 24; the "sharpest zone in frame" craft law in all six; the figure-grammar `[Name], [age], [ethnicity] [occupation] in [attire]` in 5 of 6; medium-first opening in all six. The P4↔P5 4-gram overlap is *exactly* this scaffold ("small in the lower third, face three-quarter").

**Ruling: this is the mandated voice, and the worlds transcend it.** The ICB's ANSWERED-ROOM form rule *prescribes* the skeleton (noun-first, medium-first, one tell/contact/splinter, ¾ face is the Gesture Vitalist's law, the reveal is the Title Smith's). The test is whether the six read as one picture six times, and they do not: six distinct guests, six mediums with genuinely different deployed signatures, six figures/cultures, six Latin-square palettes, and — decisively — **six different within-pair derivation lenses** (spatial/temporal/proximity/population/depth/behavioral), so the quartets aren't a reused frame-set. A voter at 128px sees six unmistakably different images (dusk wheat-house · rain-bright monsoon allotment · night-kitchen moon · eclipse whale-street · desert sea-door · lacquer night-market carp) that share a sensibility. **Coherence, not collapse.**

**Two soft spots (polish, not repair):** (1) the figure-grammar *syntax* is templated enough that 5/6 introduce the human with the same sentence-shape — a refinement could vary the syntax so the scaffold is less visible; (2) P6 alone omits the name+age (anonymous "tea-seller"), the single break in the grammar — mildly less individuated than the set, though culturally grounded via pasar-malam/apron/clogs.

---

## E. VISUAL SOMATIC GATE (3 Hyper-Skeptics vote per pair; 2-of-3 NO = BLOCKED)

| Hyper-Skeptic | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| Concept HS (Estrangement) | YES | YES | YES | YES | YES | YES |
| Medium HS (Straight-Print) | YES | YES | YES | YES | YES | YES |
| Marketing HS (Spectacle) | YES | YES | YES* | YES | YES | YES* |
| **VERDICT** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |

All six clear the bloc (no pair draws 2 NOs). Margins are not equal:

- **P1 — clean PASS.** Splinter (dark attic) survives as emotional proof; gum/impasto fusion is thumbnail-distinctive; the retell is a gut-punch; warmth earned by "thirty years demolished." Concept-HS caution: the impossibility is subtle at 128px (could read "a lit house in a field") — hence the depth/no-path repair, which holds.
- **P2 — PASS.** The only daylight/botanical entry; starch-grain does real work; empty tin = debt visible. Render caution: the upward-rain impossibility is the subtlest of the six (see R2).
- **P3 — PASS, narrow Marketing margin\*.** Distinctive via the dented towel + milk-in-bowl + cyanotype duotone + night-nurse specificity — but "moon on a windowsill" is the most *familiar* surreal trope in the set; the Marketing-HS's "could any competent MJ writer make this?" bites hardest here. The differentiators elevate it; render must hold the moon *inside* the sash (repair applied).
- **P4 — clean PASS (strongest).** Unmissable novel impossibility (whales at chimney height → dry harbor), mezzotint burr thumbnail-distinctive, "Fare accepted" earns the warmth against cold ground.
- **P5 — PASS.** Powerful novel impossibility (Pacific standing at a desert sill); verre-églomisé verso split is distinctive; widow-splinter is proof. Concept-HS caution: **highest render-variance** in the set ("a hand higher" / vertical-calm water can fail to a postcard door or a tsunami).
- **P6 — PASS, narrow Marketing margin\*.** A glowing koi-spirit is arguably NightCafe's single most common fantasy default; the prompt's separation from it (built maki-e gold *not* glow, pole bowing under real weight, wrong-way shadow, empty hook) is strong **on paper** but hinges entirely on the render holding the "built, not lit" line. If a test render shows bloom-halo koi, P6 flips to BLOCKED at the render gate — the negative-prompt note and craft tokens are the only guard.

---

## F. VENUE FITNESS — RANKED FOR ONE WINNING ENTRY

Scored for #1365: 128px fast-vote, warmth + narrative-moment rewarded, impossibility beats realism, **one-second arrest → 20-second hold → retell transmissibility → title gift → render-risk** (probability Flux produces the described frame).

### TOP 5

1. **P4-V4 · "The One That Stops" (title: *Fare accepted* / *Passage Granted*) — THE ENTRY.** Elite one-second arrest: a whale calf and a conductor **eye to eye**, fused into the set's strongest two-mass with a single saffron rim and a bright punch caught mid-fall. Deepest hold (the eye contact, the dry harbor behind), the most astonishing retell in the run, a gifting title, and — critically — the most *render-robust* of the high-arrest options: a calf's head at shoulder height beside a figure is very achievable, and the two-mass survives even if the mezzotint softens. Hits **both** proven venue archetypes at once (the "storm-at-sea" dramatic-natural-force lane *and* the intimate narrative moment).
2. **P1-V1 · "The House Waited Up" — THE BACKUP.** The warm golden-hour register that matches the venue's *winner* archetype, the cleanest warm two-mass, a devastating retell. Named backup because it **de-correlates risk** from the entry: different pair, medium, and register — the hedge if render sessions show the dark eclipse palette underperforming the warm-light preference at thumbnail. Its own risk: the impossibility collapses to "cozy cottage" if the wheat-to-doorsill / no-path cue fails.
3. **P4-V1 · "The Vanguard."** The **lowest render-risk in the entire run** — one whale cresting a dark roofline under a silver ring is trivially renderable — same world and retell as the entry. The same-concept safety net if V4's eye-to-eye composition wobbles (note: shares the entry's dark-register and medium risk, so it is a *safety*, not a *hedge*).
4. **P3-V3 · "Over the Roofline."** The cleanest 128px poster of the warm arm — two round lights (moon + lamp reflection) negotiating across one pane — warm, inviting, low-moderate render-risk, strong retell. The best warm alternative if a second warm finalist is wanted.
5. **P5-V4 · "Horizon at the Threshold."** The single highest *ceiling* for one-second arrest (a towering teal sea-wall over a tiny amber figure) but the highest render-variance in the run; include only if a render session can afford to roll for the calm-vertical-water frame.

### ENTRY RECOMMENDATION
- **THE ENTRY: Pair 04 · V4 · "The One That Stops" (*Fare accepted*).**
- **BACKUP: Pair 01 · V1 · "The House Waited Up."**

**Strategic note on warmth:** the venue rewards warm light, and the entry is eclipse-cold. I still name P4-V4 because at 128px fast-vote the *unmissable* impossibility + retell win the one-second game where P1's warmer-but-subtler "house that shouldn't be there" is the most render-fragile of the warm options; the bronze "storm at sea" precedent proves dramatic-cold natural-force images podium here, and P4 carries earned warmth as its two-temperature accent (saffron rim, two lamplit windows). The backup deliberately swings warm to hedge exactly this tension.

### Per-pick render strategy (verify BEFORE spending credits)
- **P4-V4 (entry):** Flux 2 9B Fast · 4:5 · negative field: *photograph, smooth digital painting, sunset glow, fog, cartoon, extra figures, children.* Verify: (1) calf head at shoulder height beside the figure (not a distant whale); (2) mezzotint grain/velvet-black survives (not smooth digital paint); (3) brass punch visibly mid-fall between them; (4) two-mass fuses at 128px with saffron as the only warm accent; (5) dry harbor + keeled boats read at street's end. **Re-roll, don't re-prompt**, if the medium smooths (pair's repair budget is spent).
- **P1-V1 (backup):** Flux 2 9B Fast · 4:5 · negative: *modern signage, vehicles, extra figures, children, lawn, path, footpath.* Verify: house stands in **unbroken wheat with no path**; lit doorways recede deeper than the house is wide; dark attic survives; gold interior vs aubergine field two-mass at 128px. Re-roll if a lawn/path appears (collapses to cottage).
- **P4-V1 (safety):** Flux 2 9B Fast · 4:5. Confirm single whale crests the roofline and the silver ring/sky band reads. Lowest-risk roll in the run.
- **P3-V3:** Flux 2 9B Fast · 4:5. Confirm the lamp-reflection disc survives (if lost, prefer P3-V1); two round lights across the pane at 128px.
- **P5-V4:** Flux 2 9B Fast · 4:5. Highest variance — verify water stands **vertical and calm** (no crest/spray/tsunami) and the horizon sits high inside the doorframe; budget extra credits, commit only on a clean roll.

---

## PER-PAIR VERDICTS

- **Pair 01 — SHIP.** All gates pass; density EARNED; Somatic clean PASS; houses the venue backup. Only soft note: guest outmasses the gate-figure (self-mitigated).
- **Pair 02 — SHIP (advisory repair R2).** All hard gates pass; the mandated BRIGHT pair. Advisory: propagate the V2/V4 bead-behavior rain fix into V1/V3, whose pre-repair "rain climbing upward" is the run's subtlest impossibility.
- **Pair 03 — SHIP.** All gates pass; Somatic PASS. Carries the highest "familiar trope" (moon-on-sill) margin, held above generic by the dented towel + duotone + nurse specificity.
- **Pair 04 — SHIP.** Strongest pair; all gates pass; houses THE ENTRY. Within-pair clause reuse (5 signature clauses verbatim across all four) is the declared compliance backbone — intentional, and the whale configurations still differ cleanly.
- **Pair 05 — REPAIR-with-brief (R1).** The four prompts are ship-grade (Somatic PASS, all structural/seed gates pass, highest arrest-ceiling in V4), **but the mandated ≤5-word title candidates are absent** from the terminal artifact — a step-07→step-10 handoff regression. Blocking for pair-05 deliverable completeness; trivial to fix.
- **Pair 06 — SHIP (render-dependency flag).** All gates pass; Somatic PASS on the narrowest Marketing margin. The prompt-as-written is distinctively Lofn, but it is the pair most likely to flip to BLOCKED at the actual render gate if the "built gold, not glow" discipline fails. Minor notes: anonymous figure (no name/age) and a pan-Asian medium/setting blend (Japanese lacquer craft + Malay/Indonesian pasar-malam) — defensible (the lacquer is the counter-medium, the axis assigned it), worth render-time awareness.

---

## ROUTED REPAIR BRIEFS (qa_repairs_issued = 3)

**R1 — Pair 05 · missing ≤5-word title candidates · BLOCKING (deliverable completeness).**
- *Defect:* `pair_05_step10` states its retell but the word "title" never appears; its only title, "The Sea at the Second Door," is **6 words** (>5-word gate). The candidates existed at `pair_05_step07` (lines 52–55: *The Ocean Asked Permission* · *The Sea Kept Its Promise* · *Six Hundred Miles, Arrived*) and were dropped through steps 08→09→10.
- *Return target:* re-enter Pair 05 at the **step-10 write only**; port a "Title candidates (≤5 words)" block forward. No prompt text changes — the four prompts ship as-is.
- *Sideways proposal (REDIRECT):* rather than restoring the upstream list, **harvest the title from the winning variation** — V2's own final line "*The sea asked first*" (4 words) gifts the meaning more sharply than any step-07 candidate and is already load-bearing in the prompt. Lead with it; keep "Six Hundred Miles, Arrived" as the alternate.

**R2 — Pair 02 · propagate the rain-direction fix to V1/V3 · ADVISORY (render-robustness).**
- *Defect:* V2/V4 were repaired from abstract "rain climbing upward" to renderable bead-behavior ("*rain beads lifting off the soil in vertical grain-streak threads inside that band only*"); V1/V3 kept the pre-repair phrasing, leaving the run's subtlest impossibility on the two lowest-ranked variations.
- *Return target:* light edit to V1/V3 replacing "rain climbing upward … vertical grain-streaks" with the proven bead-behavior clause. Non-blocking — the entered picks (V2/V4) are already fixed.
- *Sideways proposal (REDIRECT):* skip the edit entirely and simply **drop V1/V3 from the render shortlist** (rank them out). Their only role is set-completeness; V2/V4 carry the pair, so the cheapest fix is to not render the weaker cue rather than to repair it.

**R3 — Pairs 01 ↔ 03 · motif/title echo · ADVISORY (set curation).**
- *Defect:* the two ACCESSIBLE domestic-homecoming pairs share a "home waited up / house wasn't empty" motif and echoing titles ("The House Waited Up" / "The House Wasn't Empty" / "Someone Waited Up"). Zero text-twinning (0 shared 4-grams), so the images are distinct — but the titles self-echo.
- *Return target:* differentiate P3's lead title away from the "house/waited" family — steer P3 to "*Milk for the Moon*," leaving "The House Waited Up" uniquely to P1.
- *Sideways proposal (REDIRECT):* leave both title sets alone and enforce the separation **at the venue-selection layer** — treat P1 and P3 as one "warm-domestic-homecoming" slot and never shortlist both as finalists. This resolves the echo without editing either pair (and is moot for the current entry pick, which is P4).

---

## RUN-HEALTH FOOTER

```
{
  pairs_shipped: 5 (P1, P2, P3, P4, P6),
  pairs_repair:  1 (P5 — title-candidate deliverable),
  pairs_quarantined: 0,
  total_gate_retries: 10 (2 in-chain repairs reported by pairs 01+06 + 8 describe-render repairs
                          [P1×2, P2×2, P3×1, P4×1, P5×1, P6×1] — count verified against the six step-10 artifacts),
  qa_repairs_issued: 3 (R1 blocking-deliverable · R2 advisory render · R3 advisory curation)
}
```

## TWO BIGGEST RESIDUAL DOUBTS

1. **P6's "built gold, not glow" survival at render is unproven and load-bearing.** The whole pair's distinctiveness — and its Somatic PASS — rests on Flux rendering the carp as *burnished maki-e metal* rather than the bloom-halo spirit-koi that is NightCafe's most common fantasy default. The prompt guards this only with craft tokens and a platform negative-prompt note; no render exists yet. If it drifts, the run's most tender concept becomes wallpaper #4301. (This does not touch the entry pick, which is P4.)
2. **The set skews dark against a venue that rewards warm light, and the entry is the coldest strong image.** I ranked P4-V4 first on arrest + impossibility + render-robustness, but the podium precedent here is *golden-hour warmth*; if the actual render batch shows the eclipse palette reading grim/muddy at 128px rather than dramatic, the correct move is to promote the warm backup (P1-V1) or P3-V3 — a decision only the render session can settle, and one the entered credits must be prepared to make.
