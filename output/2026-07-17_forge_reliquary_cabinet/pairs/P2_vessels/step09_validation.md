# P2 — THE VESSELS, WARM · STEP 09 — Pre-Generation Validation + Bench
*Run: 2026-07-17_forge_reliquary_cabinet · Pair P2 · renderer_gpt_image2_rules §09 (GPT-I2 mode: Step 09 is a PRE-GENERATION VALIDATION pass, not artist-voice refinement). Bench: NORMAN / KARE / CHAYKA — reroll on a CHAYKA fail and record it.*

---

## PRE-COMMIT GATE (per prompt)

Adapted for a **single-medium UI piece-pack** (verre églomisé + gilded ink + brass). The renderer's "7 visually explosive art styles, no repeats" item is the *integrated-scene* formula (many subjects, many styles in one vase) and is **N/A** here by design — a vessel FRAME must be one coherent material family, not a style-collision. That item is intentionally not applied; every other gate check is enforced and passes.

| Gate check | V1 orb | V2 casket | V3 phial | V4 trio |
|---|---|---|---|---|
| 300–400 words, dense, no padding | ✅ 386 | ✅ 352 | ✅ 365 | ✅ 368 |
| Opens with scene + single named light + shared shadow direction | ✅ wick, behind, low → LR | ✅ seam-leak → LR | ✅ wick, left → LR | ✅ shared wick, UL → LR |
| Named materials, physical confidence (églomisé / gilded ink / brass / cork) | ✅ | ✅ | ✅ + cork | ✅ |
| One container / declared **empty void** (matte charcoal, inner-rim wick-light) | ✅ orb belly | ✅ casket mouth | ✅ phial column | ✅ three windows |
| Finished-ness: lid/latch/cradle + **BLANK** brass plaque (text-void guarded) | ✅ | ✅ | ✅ | ✅ ×3 |
| One thumbprint · one mischief · one second-ember | ✅ claw / lean / rear-claw spark | ✅ hasp / thrown latch / escutcheon-eye | ✅ shoulder / half-cork / cork-base bead | ✅ rail / orb-leans-on-casket / daily-seam |
| Storybook-Cliché override present + specific | ✅ | ✅ | ✅ | ✅ |
| Zero veto words (ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate) | ✅ | ✅ | ✅ | ✅ |
| Zero artist names in prompt | ✅ | ✅ | ✅ | ✅ |
| Palette cap 3–4 warm tones on velvet-black | ✅ 4 | ✅ 4 | ✅ 4 | ✅ 4 |
| 128px read stated | ✅ | ✅ | ✅ | ✅ |
| Self-contained, one-shot (no downstream iteration expected) | ✅ | ✅ | ✅ | ✅ |
| Resolution assigned from {1024², 1536×1024, 1024×1536} | ✅ 1024² | ✅ 1536×1024 | ✅ 1024×1536 | ✅ 1536×1024 |

*(Word counts machine-verified against step08_prompts.md. Veto-word + artist-name greps ran clean against the prompt blocks — the only matches were in the operator-note's own banned-list and the step-pointer footer, not inside any render prompt.)*

---

## THE BENCH (NORMAN / KARE / CHAYKA)

### V1 — THE ORB
- **NORMAN (usability = beauty):** the cradle, blank plaque, and the lean are **geometric** landmarks (not color); the reserved matte void holds a contrast floor for content. **PASS.**
- **KARE (128px / four shapes / named light):** a sphere is one shape; one gold rim + one dark eye; wick named and sourced. Cheapest, cleanest silhouette on the sheet. **PASS.**
- **CHAYKA (named materials only):** verre églomisé inside-back gild + gilded-ink kintsugi + burnished brass; no #FFD700; the lean + thumbprint are authored. **PASS (after reroll — see below).**

### V2 — THE CASKET
- **NORMAN:** the ajar lid + thrown latch are **non-color affordance cues** (a state you can read in greyscale); escutcheon + corners are geometric. **PASS.**
- **KARE:** rectangular box + tilted lid + one blade of seam-light = four shapes and a light; reads at 128px as "box breathing light." **PASS.**
- **CHAYKA:** the seam-leak (gold-enters-from-the-crack) is the house brand device, not stock amber; escutcheon-eye is specific. **PASS.**

### V3 — THE PHIAL
- **NORMAN:** footed + collared + blank plaque = geometric finish; the half-pulled cork is a legible non-color state. **PASS.**
- **KARE:** tall bottle + tipped cork + one smoke-ribbon = four shapes and a light. **CONDITIONAL → PASS:** a *thin* smoke wisp would smear/vanish at 128px (entropy). Fixed in the prompt to **one thick-based confident ribbon**, not filigree. Re-checked: reads at thumbnail. **PASS.**
- **CHAYKA:** watched from step 07 (genie-lamp adjacency). Held to a single sourced ember-smoke ribbon on a named églomisé phial with a named wick — authored, not a genie plume. **PASS.**

### V4 — THE TRIO
- **NORMAN:** the sheet's whole job is a usability spec — it declares spacing rhythm, relative window sizes, and the daily-heartbeat state so the atlas/CSS layer inherits geometry, not vibes. **PASS.**
- **KARE:** three distinct silhouettes, one rail, one shared shadow — the 128px legibility audit made visible; this sheet IS the proof. **PASS.**
- **CHAYKA:** the daily-seam-runs-warmer + orb-leaning-on-casket are committed authored details no generic amber shelf carries. **PASS.**

---

## RECORDED CHAYKA REROLL (the clause bit — honest log)

**Piece:** V1 — THE ORB. **First-pass draft language (rejected):** *"a warm amber glow filling the sphere, a fantasy-gold rim catching the light."*

**CHAYKA fail (two counts):**
1. **Slop uniform** — "fantasy-gold" and "amber glow filling the sphere" is exactly the default-amber that hangs on every AI arcade on the internet; unnamed material; would pass anonymously in a hundred other shelves. Violates the no-#FFD700 clause.
2. **Doctrine breach** — "glow *filling* the sphere" fills the reserved void, contradicting the Container Test (the interior must be a declared EMPTY window, inner-rim only).

**Reroll (accepted, now in step 08):** replaced with **"its ember read *through* the glass belly"** + explicitly named materials (**verre églomisé inside-back gilding; gilded-ink kintsugi; leaf-catching-the-wick as one thin ring**) + the core re-declared as **flat matte charcoal-black reserved void with only an ember-gold inner-rim line.** Result: named-material sobriety restored AND the void doctrine honored in one fix.

**Secondary watch (no reroll, logged):** V3's escaping smoke — KARE flagged entropy/smear at 128px; corrected in-place to one thick-based ribbon rather than rerolled. CHAYKA's step-07 genie-lamp watch cleared on the sourced-and-named grounds above.

**Bench verdict:** 4/4 prompts PASS after 1 recorded CHAYKA reroll (V1) + 1 in-place KARE correction (V3). No kills. Cleared to rank.

*Next: step10 — ranking.*
