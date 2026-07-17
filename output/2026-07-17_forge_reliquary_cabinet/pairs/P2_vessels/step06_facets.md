# P2 — THE VESSELS, WARM · STEP 06 — Facets + GPT-Image-2 Failure Scan
*Run: 2026-07-17_forge_reliquary_cabinet · Pair P2 (ACCESSIBLE arm) · Vessel frame set A: ember-glass orb · hinged casket (ajar) · corked phial · Medium: verre églomisé + gilded ink · Light: single wick / seam-leak · TARGET_RENDERER: GPT_I2 (§06–§10 overrides bind) · Voice: LOFN-PRIME, AWE — the Eager Archivist at the forge. 💜*

> Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person. Ledger of 18 (03_panel_debate.md) — no new panel invented.

---

## THE CONTAINER READING (Container Test — the binding gate for this pair)

These vessels are **CONTAINERS**, and this pair renders the *frames* — not the worlds. That inverts the usual "amplify the interior" instruction into a harder problem, and the panel named it plainly:

- **Outer silhouette clean and dominant** — the vessel FRAME fills **70–80%+** of the plate; the velvet-black ground is the compressed support. *(after Kircher / the Cabinet Keeper: "variety under one order — the silhouette must win the wall before the interior earns a second look.")*
- **The interior window is a DECLARED EMPTY VOID** — a clean, matte, vacant plate reserved for live game art. No scene, no relic, no figure is rendered inside. *(Doctrine: "Nothing Finished, Everything Ready" — Flair 11. The Compression Auditor, after Tufte, dissents-approvingly: "Good — an empty void is nearly free bytes; do not let a single gilded pixel wander into the reserved zone.")*
- **The empty window must still feel INHABITED** — the one craft trick that carries the whole pair: **rim-light from within the void's own edge.** The wick or the seam-leak catches only the *inner rim* of the opening, so the vacancy reads as *a lantern waiting to be lit*, an inhabited absence — never a blank hole, never a filled picture. *(after Pelton / the Lit Window: "a window at night pulls because the light is evidence of life inside. Trace the rim, not the middle, and the emptiness will lean toward the viewer.")*
- **Reconciliation note (deliberate):** Flair 8's "≥30% quiet velvet" is a *wall/backing* law (P1's job). For a single-vessel museum plate the quiet is carried jointly by the velvet margin **and** the reserved matte void; the vessel still dominates 70–80%. Recorded so it reads as a choice, not a miss.

The emotional engine to hit, at 128px, in 0.3s: **the pull of a lit window at night** — approach-warmth, certainty of welcome — with **custodial pride** as the second lens (someone tends these nightly, and the thumbprint proves it).

---

## FIVE JUDGING FACETS (score any proposed P2 prompt against these)

1. **Threshold-warmth at a glance.** At 128px in 0.3s, does the vessel read as a lit window you want to lean toward — one *named, directional* warmth (single wick / seam-leak) throwing one hard shadow, never an ambient amber wash? If it reads as backlight-LED or a vending machine, it fails. *(after Pelton × Chen — warmth legible to a stranger, no lore gate.)*

2. **Inhabited emptiness.** Is the central window an unmistakably **reserved** void — flat, vacant, content-ready — that nonetheless feels tenanted because its inner rim alone catches the light? An absence that says "waiting to be lit," never a blank cut-out and never a rendered scene. *(after Bachelard / intimate immensity + Flair 11.)*

3. **Finished-ness you could hold.** Does the vessel carry its complete furniture — lid, latch or cradle, and a small **blank** brass plaque — with physical confidence (verre églomisé depth, gilded-ink seams, brass with real weight and a cast shadow), so it reads as an object tended nightly and not a sticker? *(after Yu / per-vessel finished-ness; Kouthoofd / premium restraint.)*

4. **Biography in the gold + one authored intimacy.** Do the hairline kintsugi seams read as *mended-from-play* (use made precious, warm — never damage), and does the **one thumbprint + one mischief** per vessel land as authored tenderness (a lean, a swung latch, a half-pulled cork) rather than decoration? *(after Bartlett / seam-biography; Caillois / the mischief that keeps it a toy.)*

5. **Named-material sobriety (anti-slop floor).** Is every gold a *named* material — gilded-ink gold, leaf-on-glass églomisé, burnished brass — the palette a disciplined 3–4 warm tones on velvet-black, with nothing that could hang on any other AI arcade? *(after Chayka / the Slop Sentry: "no #FFD700, or reroll.")*

```json
{ "facets": [
  "Threshold-warmth at a glance: at 128px in 0.3s the vessel reads as a lit window to lean toward, lit by one named directional source (single wick or seam-leak) casting one hard shadow — never an ambient amber wash or backlight-LED vending-machine read.",
  "Inhabited emptiness: the central window is an unmistakably reserved, flat, vacant, content-ready void whose inner rim alone catches the light, so it feels tenanted — a lantern waiting to be lit, never a blank cut-out and never a rendered scene.",
  "Finished-ness you could hold: complete furniture — lid, latch or cradle, and a small BLANK brass plaque — carried with physical confidence (verre eglomise depth, gilded-ink seams, weighted brass with a cast shadow), an object tended nightly, not a sticker.",
  "Biography in the gold plus one authored intimacy: hairline kintsugi seams read as mended-from-play (use made precious, warm, never damage), and exactly one thumbprint and one mischief per vessel land as authored tenderness, not decoration.",
  "Named-material sobriety: every gold is a named material (gilded-ink, leaf-on-glass eglomise, burnished brass), the palette a disciplined 3-4 warm tones on velvet-black, nothing that could hang on any other AI arcade."
] }
```

---

## GPT IMAGE 2 FAILURE SCAN (mandatory red team — renderer_gpt_image2_rules §06)

| # | Failure mode | Risk | Why / mitigation |
|---|---|---|---|
| 1 | **Storybook Cliché** | **MED** | A centered warm museum plate IS the cliché zone. Every prompt carries an explicit per-prompt override: high-contrast, saturated ember/gold/bone on true velvet-black, hard engraved brass edges + dry craquelure tooth — NOT pastel, NOT soft-edged, NOT centered-and-floating (grounded in cradle/rail with a cast shadow). |
| 2 | **Warm rim light** | **HIGH** | This is, by design, a warm-light pair — so the danger is the *unnamed ambient halo*. Mitigation: every light has a NAMED source (single wick / seam-leak) and a directional hard shadow; rim-light appears ONLY as the intentional inner-void-edge inhabitation, never as an ambient glow around the whole vessel. |
| 3 | **Centered pastel subject** | **MED** | Subject is centered on purpose (museum plate), but the palette is saturated, not pastel; velvet-black ground; the vessel is footed/cradled and casts one hard shadow — no floating. |
| 4 | **Reiteration / edit-chain** | **LOW** | One-shot self-contained prompts; no chained edits (renderer Reiteration Bug law). If a render fails, rewrite fresh in a new session. |
| 5 | **Reference contamination** | **LOW** | No real brands/logos; the plaque is declared BLANK (its face guarded so the model letters nothing); fictional archival furniture; zero artist names inside prompts. |
| 6 | **Entropy drift / texture smear** | **MED** | Églomisé + craquelure + brass filigree can smear at the seams. Mitigation (after Tiffany / few confident lead-lines + Wyeth / reserve the crack-map): lead-lines few and confident; craquelure reserved to the seam-runs only; the reserved void kept clean (no texture inside it); V3's escaping smoke rendered as ONE thick-based ribbon, not filigree. |
| 7 | **Overcrowded annotation** | **LOW** | No annotations, no specimen markers. Single vessel per plate; V4 is the only multi-object piece and it is a clean rail of three. |
| 8 | **Text illegibility** | **LOW / guarded** | No text is rendered anywhere. The plaque is the one place text could sneak in — every prompt declares it "burnished utterly blank: no letters, no numerals, a reserved zone." |
| 9 | **Primary read inversion** | **LOW** | No diagram/text to invert. Guard: the reserved void must not out-read the frame — keep it matte and quiet so the vessel *silhouette* wins the 0.3s read (the void is the second read, the frame is the first). |

**"Primary read inversion" (the pair-specific version):** FAIL if the empty void reads before the vessel silhouette. The frame is the hero; the window is the guest.

---

## SKEPTIC DISSENTS (each genuinely objects — the bench has teeth)

- **THE SHELF CYNIC (after Norman) — dissents:** *"A vessel a mother finds beautiful is worthless if she cannot tell it is a button. Where is the usability state?"* → Answered by the finished-ness furniture: lid/latch/cradle/plaque are **geometric** landmarks (not color), the ajar lid and thrown latch are non-color affordance cues, and the reserved void preserves a contrast floor for whatever content lands in it. The lit-lattice focus state itself is P4's overlay; P2 frames are built to receive it (dormant hairline seams sit where the focus lattice will thicken).
- **THE COMPRESSION AUDITOR (after Tufte) — dissents:** *"You are asking for glass AND gilding AND brass AND craquelure on every plate. Half of that is chartjunk kilobytes."* → Answered by the 3–4 tone cap, the velvet-black near-free ground, the clean matte void, and craquelure reserved to seams only. Each prompt states its tone budget explicitly (step 09 gate).
- **THE SLOP SENTRY (after Chayka) — dissents hardest:** *"Every AI arcade on the internet glows amber. Prove this one is authored."* → Answered by named materials only (verre églomisé inside-back gilding; gilded-ink kintsugi; burnished brass — never #FFD700), the kintsugi-as-play-biography fiction, the custodian's thumbprint, and the reserved-void doctrine that no other amber shelf bothers to build. **One CHAYKA reroll is recorded in step 09** to prove the clause bites.

*Next: step07 — four controlled variations (1–2 axes each).*
