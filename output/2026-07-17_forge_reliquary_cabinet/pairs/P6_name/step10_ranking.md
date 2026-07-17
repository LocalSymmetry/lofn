# PAIR P6 · STEP 10 — Ranking · "THE NAME" (AMBITIOUS)

*Run:* `output/2026-07-17_forge_reliquary_cabinet` · *Pair:* P6 · *Renderer:* GPT-Image-2 DIRECTIVE · *Chain:* 06→10.
*Note:* these four are not competing alternatives — they are a **complete piece-pack** (hero, mark, furniture, mobile) that the cabinet consumes together; **all four ship**. The rank below is priority / confidence / render-order for the coordinator, scored on the renderer's step-10 axes (0.3s anchor clarity · anti-cliché · entropy risk · physics payoff · consumption fit).

## Ranked (one line each)

**RANK 1 · V1 — FULL MASTHEAD LOCKUP** (1536×1024, 400w) — the hero that signs the cabinet: strongest single "whoa" and highest material payoff (crucible pour, one molten bead, églomisé emission), best 0.3s read; carries the most entropy of the set (text + crucible + bounded filigree + three authored details) but each is mitigated, and the hero is worth the spend — facet total 97, the piece the run is judged by.

**RANK 2 · V2 — THE LATTICE MONOGRAM, ALONE** (1024×1024, 394w) — the keystone brand mark and favicon: highest anti-cliché of the four (the kintsugi crack-lattice IS the signature — a device nothing generic carries) and the lowest entropy (two-tone, one emblem, 32px-proven); the most reliable render and the most reused asset, ranked second only because the masthead owns the primary public read.

**RANK 3 · V3 — ACCENT GLYPH SHEET** (1536×1024, 395w) — the cabinet's small furniture: five emblems distinct by SHAPE not colour (bell/wick/crucible/thumbprint/ajar-door), each four-shapes-and-a-light on a declared spacing grid; essential utility with moderate spectacle, its one live risk (a glyph-seam reading as damage) flagged for QA.

**RANK 4 · V4 — REDUCED ONE-LINE MOBILE MASTHEAD** (1536×1024, 373w) — the necessary responsive variant: deliberately the least ornamented (dormant hairline seam, filigree removed by the lead-line law), so the least striking as a standalone; ranked last on novelty because it is V1's string pared down, yet ship-critical for phone headers and the cleanest proof that the wordmark survives reduction.

## Set logic
Render order follows the rank: lock the hero (V1) and the mark (V2) first, since the whole cabinet is signed off them; the glyph sheet (V3) and mobile reduction (V4) inherit their locked palette, seam-line system and crucible source. Seam-state slides molten (V1) → static-mark (V2) → per-icon touch (V3) → dormant hairline (V4); filigree slides bounded → none. One name, four scales of attention, one hand.

## Describe-render self-check (top 2, per renderer discipline)

**V1 prediction:** GPT-Image-2 emits a centered two-line gold wordmark, THE FORGE ARCADE, on a near-black velvet field, a molten crucible glow above and behind the caps, thin gold crack-veins threading the letters, a few tight filigree curls at the serifs, one bright bead on the word FORGE. At 128px: a gold two-line block on black under a warm pour — reads as the name instantly.
*The one way it renders generic:* the model reverts to a default fantasy-logo glow and floods ambient amber, drowning the crucible source and the crack-lattice.
*Guard already in the prompt:* light is declared as a single crucible pour with "no ambient rim, no halo, no drop shadow, no bevel; the letters emit," plus the explicit Storybook override — drift risk acceptably low. If a render still floods amber, isolate one edit ("remove all ambient glow; keep only the crucible pour and the seam-leak") in a fresh session, never a chained edit.

**V2 prediction:** GPT-Image-2 emits one centered gold emblem on a black square — a cracked crucible-drop drawn in three or four thick gold strokes with a small ember heart and one blue node — generous dark margins. Downscaled to 32px it holds as a gold cracked droplet with one bright node.
*The one way it renders generic:* the crack detail proliferates into a busy filigree ball that dies at favicon scale.
*Guard already in the prompt:* "three or four confident strokes … thick, deliberate and few," plus the explicit 32px resolve statement and "no filigree at that size" — primary-read (silhouette) is protected before the crack detail. Verdict: HOLDS. Residual note: if the 32px test fails on first render, drop the branch-crack, keep the outer fracture + diagonal seam + node.

## Render plan (recommended)
| Rank | Piece | Resolution | Fidelity | Note |
|---|---|---|---|---|
| 1 | V1 masthead | 1536×1024 | High | exact multi-word text is critical |
| 2 | V2 monogram | 1024×1024 | Medium→High | step to High only if 32px test wobbles |
| 3 | V3 glyph sheet | 1536×1024 | Medium→High | High if any glyph-seam smears |
| 4 | V4 mobile | 1536×1024 | High | one-line exact text, small target |

*Provenance: pair P6 agent, step 10 of chain 06→10, 2026-07-17; inputs = step08_prompts.md + step09_validation.md + `skills/image/steps/10_Generate_Image_Revision_Synthesis.md` + renderer step-10 override. Self-critique: ranking the hero (V1) above the keystone mark (V2) trades a little render-reliability for public impact — defensible for a masthead pair whose ICB job is to SIGN the cabinet, but if the coordinator wants the single safest asset to lead the brand, V2 is the pick.*
