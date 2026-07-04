---
run: 2026-07-01 (FULL DAILY — IMAGE LANE)
step: 04 — Medium Selection
input_icb: images/CREATIVE_CONTEXT_image.md (injected verbatim)
step_file: skills/image/steps/04_Generate_Image_Medium.md
Continuity Payload Used: image ICB — Golden Seed, metaprompt, personality, 18-voice Panel Ledger, 15 Special Flairs, composition law
---

# Step 04 — Medium Selection

One compelling medium per retained concept, drawn from the Seed Framing palette and the day's material vocabulary. Mediums are named as real print/photo/paint tooling (the Flux hooks), never as living-artist styles. The medium is chosen so the two-shell overlap is legible in the material itself.

```json
{
  "mediums": [
    "C1 — Long-exposure silver-gelatin astrophotograph feel, hand-tinted like a 19th-c. glass plate: crisp young shell, grain-faint old shell, cold catwalk realism.",
    "C2 — Autochrome-grain portrait photography, single screen-source underlight, the ad living inside a wet pupil reflection.",
    "C3 — Egg-tempera panel painting with a single dull-bronze orrery bead under one kiln-red beam, vast faint ring in muted filament-purple.",
    "C4 — Candlelit-interior oil feel translated to CRT-amber glow; a cold cyanotype-blue window as the second, older light.",
    "C6 — Torn-poster décollage photography, wet halftone gloss top layer over old grey pasted strata, a reflection in the wet sheen.",
    "C7 — Aftermath intaglio etching (drypoint burr), anti-glamour, warm ember-red edge-of-light against cold mezzotint dark.",
    "C9 — Double-exposure pictorialist portrait, sharp face over a photogravure filament-nebula ground, one eye carrying strain.",
    "C10 — Scientific-instrument engraving crossed with sharp macro photography: a dead-accurate dial, dried-blood-red needle, blurred cold sky.",
    "C11 — Subverted commercial product photography, flawless studio gloss made wrong, ghosted x-ray of the harm through the glass.",
    "C12 — Dry-plate long-exposure star-trail composite, young sharp streak over old slow concentric trails, small anchoring silhouette.",
    "C13 — Hands-only relief/linocut printmaking, warm cup-steam over cold dark field, no faces.",
    "C14 — Corroded ambrotype / ruined-plate salvage photograph, verdigris-and-Venetian-red oxidation over a sharp young reflection."
  ]
}
```

**Provenance:** step file `04_*`; inputs = step03 refined concepts + artists + image ICB. Self-critique: each medium names concrete materials (silver-gelatin, egg tempera, drypoint burr, ambrotype oxidation, halftone gloss) that Flux renders as hooks, and each carries the two-shell in the material (crisp-vs-grain, gloss-over-strata, sharp-over-oxidation). No living-artist names. Six distinct medium families are present so step 05 can assign six DIFFERENT camera grammars.
