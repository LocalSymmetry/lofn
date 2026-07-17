# STEP 09 — PRE-GENERATION VALIDATION + VERIFICATION BENCH
## PAIR P3 — THE CHROME
*Run: 2026-07-17_forge_reliquary_cabinet · Voice: LOFN-PRIME, AWE mode 💜 · TARGET_RENDERER: GPT_I2*
*Per renderer rules §09: this is a PRE-GENERATION validation pass, not artist-voice refinement. No chained edits. Fix the prompt before any render.*

---

## PRE-COMMIT GATE (adapted for asset-pack pieces — all four must pass every box)

| Gate check | V1 buttons | V2 card | V3 badges | V4 chips+plaque |
|---|:--:|:--:|:--:|:--:|
| Scene + single named light (wick-warm edge, one direction) | ✅ | ✅ | ✅ | ✅ |
| Named materials only (engraved brass / lacquer inlay / gilded ink) — no default gold | ✅ | ✅ | ✅ | ✅ |
| Storybook override present + specific (orthographic/flat-plate, no rim, no vignette) | ✅ | ✅ | ✅ | ✅ |
| Zero veto words (ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate) | ✅ | ✅ | ✅ | ✅ |
| Zero artist names in prompt body | ✅ | ✅ | ✅ | ✅ |
| Exact text only where required + strings stated | n/a (voids) | n/a (voids) | ✅ FUN·SPARK·KILL | n/a (voids) |
| Declared live-content voids present | ✅ label+state | ✅ thumb+cartouche | ✅ (banners=text) | ✅ chip+name |
| 2–4 tone puzzle stated | ✅ 4 | ✅ 3+reserve | ✅ 3+reserve | ✅ 4 |
| 128px read stated | ✅ | ✅ | ✅ | ✅ |
| Geometry-not-color / greyscale-legible / 48px targets | ✅ | ✅ (frame) | ✅ shapes | ✅ notch |
| Resolution set, multiple of 16 | ✅ 1536×1024 | ✅ 1024×1536 | ✅ 1536×1024 | ✅ 1536×1024 |
| Self-contained, one generation, no downstream iteration | ✅ | ✅ | ✅ | ✅ |

Word counts (300–400 target): **V1 355 · V2 347 · V3 356 · V4 368.** All in band.

---

## VERIFICATION BENCH — NORMAN / KARE / CHAYKA (this pair lives or dies on Norman)

### NORMAN (usability = the beautiful state; geometry deltas, targets, contrast voids)
- **V1 — CONDITIONAL → AMENDED → PASS.** *Initial catch:* "Your hover risked signaling with a lift-shadow — that's light, not geometry. Reject as drafted." *Fix folded into step 08:* hover's PRIMARY delta is now the DOUBLED keyline, shadow demoted to "secondary cue only"; focus is an OUTBOARD bracket-lattice "you could feel by shape alone." *Re-read:* all five states separable in greyscale by silhouette. **The beautiful state is the usable state. PASS.**
- **V2 — PASS.** Big thumbnail void + verdict cartouche are keyline-framed contrast voids; the keyhole ember is decoration at the FOOT, never the content signal. Card face is not itself a click target ambiguity — it is a backing. Clean.
- **V3 — PASS, with a demand honored:** "The badge must not require its word to be understood." Shape reads first, word confirms, heat is a third redundant cue. A color-blind or greyscale user still gets burst/seed/lump. **PASS.**
- **V4 — CONDITIONAL → AMENDED → PASS.** *Catch:* "A selected chip that only brightens is a color-only state — fail." *Fix:* selected chip now carries a DOUBLED keyline + NOTCHED left bracket (geometry). Plaque name is a reserved void with a baseline rule, meets contrast. Targets ≥48px stated. **PASS.**

### KARE (128px, four shapes, geometry not glow, no filigree)
- **V1 — PASS.** Five plates, escalating edge weight, one bracketed, one inset+bead, one flattened — reads as a legible strip at thumbnail. Few confident leads, no filigree.
- **V2 — PASS.** Four-shape read: card silhouette, gold frame, thumbnail void, one ember dot. Confident border, not filigree soup.
- **V3 — CONDITIONAL → AMENDED → PASS.** *Catch:* "A forty-ray sunburst is mush beside a seed at 128px. Reroll the ray count." *Fix:* sunburst = "a FEW bold confident rays"; seed = "exactly ONE tail"; slag = "no rays, no tail." Three shapes now call across the room with the color off. **PASS — the pair's sharpest 128px win.**
- **V4 — PASS.** Row-of-pills-over-nameplate reads at thumbnail; notch is a large-enough geometry event to survive scale.

### CHAYKA (named materials only; would this hang in any other AI arcade?)
- **V1 — PASS.** Engraved-brass keylines, de-bossed lacquer, molten-bead-from-seam, thumb-worn corner — authored, not generic. No #FFD700 gradient.
- **V2 — PASS.** Lacquer-black card + gilded-ink double rule + kintsugi corner-seams + keyhole escutcheon = committed and specific. Parchment correctly RETIRED.
- **V3 — PASS.** Sunburst/seed/slag is an authored verdict vocabulary, not stock check/star/X icons; engraved exergue banners are museum-specimen, not sticker badges.
- **V4 — CONDITIONAL → AMENDED → PASS.** *Catch (dissent):* "Chips are where AI arcades all look alike — a pill with a fill. As drafted this could hang anywhere." *Fix:* selected chip gets a NOTCHED bracket + a mended kintsugi seam + tool-chatter in the line — named engraved brass, "never a plain colored pill." **PASS.**

---

## RESULT

**Bench outcome: 4 / 4 PASS.** Zero kills. **Three conditional catches, all amended and folded into step 08** (V1 Norman: doubled-keyline-primary; V3 Kare: few-rays/one-tail/lumpen; V4 Chayka+Norman: notched-bracket selected chip). No prompt proceeds to render with an open dissent. All four are self-contained and generation-ready pending Scientist cost approval.

*Reduce-motion note logged: every state is a static truth; the pressed "molten pour" is a static inset+bead, requiring no animation to read (Tufte approved).*
