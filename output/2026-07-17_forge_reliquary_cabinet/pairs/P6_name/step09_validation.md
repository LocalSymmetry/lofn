# PAIR P6 · STEP 09 — Pre-Generation Validation + Verification Bench · "THE NAME" (AMBITIOUS)

*Run:* `output/2026-07-17_forge_reliquary_cabinet` · *Pair:* P6 · *Renderer:* GPT-Image-2 DIRECTIVE · *Chain:* 06→10.
*Per the renderer override, step 09 is NOT artist-voice refinement — it is a pre-generation validation pass. No chained edits; the prompt IS the final artifact until rendered.*

## A. GPT-Image-2 pre-generation checklist (per prompt)
*The renderer's default pre-commit gate is written for the 7-style integrated-scene formula; this pair is a text/UI piece-pack, so the gate is adapted to the P6 contract (exact-text lockup, named materials, one crucible source, seam-as-signature). Adaptation recorded here for QA.*

| Check | V1 masthead | V2 monogram | V3 glyph sheet | V4 mobile |
|---|---|---|---|---|
| Word count in 300–400 band | ✓ 400 | ✓ 394 | ✓ 395 | ✓ 373 |
| Exact render string stated once, correctly spelled | ✓ THE FORGE ARCADE | ✓ zero-text declared | ✓ zero-text declared | ✓ THE FORGE ARCADE |
| "ONLY that text / no other marks" declared | ✓ | ✓ (no text of any kind) | ✓ (no labels) | ✓ |
| One named light source (crucible pour) | ✓ pour behind | ✓ one ember core | ✓ one implied source below row | ✓ underline glow |
| Gold-enters-from-the-crack / no ambient rim | ✓ | ✓ through the fracture | ✓ light at every seam | ✓ no rim/halo |
| Named materials only (no default-gold) | ✓ ink/tempera/filigree | ✓ gilded ink + leaf | ✓ engraved-brass line | ✓ ink/tempera |
| Velvet-black knockout ground declared | ✓ | ✓ | ✓ | ✓ |
| 2–4 tone palette puzzle stated | ✓ 4-tone | ✓ 2-tone favicon | ✓ 3-tone | ✓ 3-tone |
| Storybook-Cliché override present + specific | ✓ closing line | ✓ closing line | ✓ closing line | ✓ closing line |
| 128px read stated in-prompt (V2: +32px) | ✓ | ✓ 32px + 128px | ✓ per-glyph 128px | ✓ header-height read |
| Zero artist names | ✓ | ✓ | ✓ | ✓ |
| Zero veto words (ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate) | ✓ | ✓ | ✓ | ✓ |
| Self-contained (no downstream iteration) | ✓ | ✓ | ✓ | ✓ |
| Reference-contamination guard (fictional archival mark, no real logo/QR/UI) | ✓ | ✓ | ✓ | ✓ |

All four pass all rows. Veto-word and artist-name checks were run as substring scans over the ```text blocks only (machine-verified): CLEAN on both across V1–V4.

## B. Facet re-score (from step 06; ship threshold ≥85; F1 & F2 are hard gates)

| Facet (weight) | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| F1 exact wordmark reads first, clean (25) | 25 | 24 (n/a text → scored as "zero-text declared cleanly") | 24 | 24 |
| F2 seam is the signature / 32px monogram (25) | 23 | 25 | 20 (seam-touch per glyph, not the mark itself) | 22 |
| F3 crucible single source / gold from crack (20) | 20 | 19 | 19 | 18 |
| F4 named materials, filigree bounded (18) | 17 | 18 | 18 | 18 |
| F5 finished, thumbprint, one mischief (12) | 12 | 10 | 12 | 10 |
| **Total /100** | **97** | **96** | **93** | **92** |

No facet falls below half-weight; both hard gates (F1, F2) clear on every prompt. All four ship-eligible.

## C. Verification bench — NORMAN / KARE / CHAYKA (every prompt, before it ships)

**NORMAN (the beautiful state is the usable/readable state):**
- V1: the seam is declared hairline, crossing letters "without breaking a letter's skeleton" — legibility protected; readable at 128px. **PASS.**
- V2: the monogram is required to resolve at 32px to 3–4 strokes by geometry not glow — usability-first favicon. **PASS.**
- V3: five glyphs distinguished by SHAPE (bell/wick/crucible/thumbprint/door), each with one point of light — never color-alone; declared gutters keep touch-target spacing. **PASS.**
- V4: seam thinned to a dormant hairline so it "never fractures a letter" at header height; declared status-bar/menu voids respect the real UI. **PASS.**

**KARE (128px, four shapes, light source named):**
- V1: four-shapes read stated (two text bars, crucible, seam-thread); source named (crucible). **PASS.**
- V2: 32px read is 3–4 gold strokes + one node; enrichment vanishes gracefully; source named (ember core). **PASS.**
- V3: each glyph composed as four-shapes-and-a-light and stated per glyph; single implied source below row. **PASS.**
- V4: header-height read stated; source named (underline glow). **PASS.**

**CHAYKA (named materials only; would this hang in any other AI arcade? if yes, reroll):**
- V1: gilded ink + cracked egg-tempera + bounded pixel-rococo + crucible-source + kintsugi-as-play-biography — authored, committed, would NOT hang elsewhere. **PASS** (after one recorded reroll, §D).
- V2: the mark IS the kintsugi crack-lattice — the single most un-generic possible brand device; not a glowing orb. **PASS.**
- V3: engraved-brass icon line with one seam touch each; the thumbprint glyph + the bell's echoing wear are authored specifics. **PASS.**
- V4: pared gilded wordmark, filigree deliberately removed, single ember underline — restraint reads as authored, not defaulted. **PASS.**

## D. Recorded reroll (CHAYKA kill — logged per the bench's no-default-gold clause)

**Where:** V1, first drafting pass. **Verdict: FAIL → rerolled.**

*Failed draft (killed on sight):*
> "Majestic glowing golden letters spelling THE FORGE ARCADE, radiant fantasy typography with ornate gold filigree everywhere, dramatic warm light, epic game logo, dark background with magical embers."

*CHAYKA's dissent (verbatim to the bench):* "This is the slop uniform. 'Radiant fantasy typography,' 'epic game logo,' 'magical embers,' filigree *everywhere* — every AI arcade on the internet ships this exact amber glow. The gold is unnamed (#FFD700 by default), the light is ambient not sourced, and it would hang in any other lobby without changing a pixel. It also trips a veto word ('magical'). Reroll with a named material or it does not ship." (NORMAN concurred: filigree-everywhere taxes legibility; KARE concurred: no stated 128px read.)

*Reroll applied (the shipped V1):*
1. Named the gold — "gilded ink" letterforms + "cracked egg-tempera" counters (kills the default-amber read).
2. Replaced ambient "dramatic warm light" with ONE sourced crucible pour + gold-enters-from-the-crack emission (kills the warm-rim halo).
3. Bounded the filigree by the lead-line law — "a few confident curls at cap-serifs and the crucible lip, never a dense thicket" (kills filigree soup).
4. Made the seam authored — kintsugi crack-lattice that mends the letters AND doubles as the monogram (the seam-is-the-signature; a device no generic logo carries).
5. Removed the veto word "magical" and the clichés "radiant / epic / majestic"; added the explicit Storybook override.
6. Stated the four-tone palette puzzle and the 128px read.

*Post-reroll: PASS on all three bench seats. Reroll count for P6 = 1 (V1). No other prompt required a reroll.*

## E. Pre-generation validation verdict
Zero artist names remain; zero veto/forbidden language remains; every Storybook override is present and specific; every prompt is self-contained (one generation, no chained edits — Reiteration Bug avoided). **All four prompts are cleared for generation.** Recommended fidelity: High-fidelity for V1 and V4 (exact multi-word text is critical); Medium sufficient for V2 (single emblem) and V3 (icon line-work), stepping to High if a first render shows any glyph-seam smear.

*Provenance: pair P6 agent, step 09 of chain 06→10, 2026-07-17; inputs = step08_prompts.md + step06 facets + `skills/image/steps/09_Generate_Image_Artist_Refined.md` + renderer step-09 override (validation pass, not refinement). Self-critique: F2 scores lowest on V3 (20/25) because the glyph sheet carries the seam as a per-icon touch rather than as the mark itself — accepted, since V2 is the piece that owns the monogram; QA should confirm the glyph seams stay hairline and do not read as damage.*
