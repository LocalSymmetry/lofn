---
run: 2026-07-28_one_more_door_lithophane
phase: 1 / step 06
---

# METAPROMPT — the creative director's brief

## VOICE
**LOFN-PRIME (AWE mode — the Kiln Attendant).** Something was made here patiently by someone who
is not in the room any more. You are not documenting a pottery studio; you are handling the output
of one, in the dark, with a limited light. Generic ceramic-studio neutrality is forbidden.

## LOCKED MOOD
**PATIENT DREAD, LIT FROM BEHIND.** Not "mysterious," not "atmospheric," not "eerie." The precise
compound: the tenderness of a hand-made object colliding with the fact that reading it costs you
the only light you have. Awe at the craft, dread at the price, in one slab of clay.

## THE FIVE PANEL AHA-MOMENTS (attributed — carry these, do not restate them)

1. **The picture is the missing clay** *(THE RELIEF, after Whiteread)* — bright is where material
   was taken away. Everything luminous is an absence. Never paint an image onto a slab; carve it
   out of one.
2. **Two images that need not agree** *(THE THRESHOLD, after Matta-Clark, repairing the Concept
   Skeptic)* — unlit reads by **shadow** under raking light (the door's surface claim); lit reads
   by **transmission** (what is actually behind). When they disagree, the player has learned
   something and it cost them. **The unlit state is never blank — it is carved.**
3. **Light steps, it never fades — SIX VALUES** *(THE FALLOFF + THE THICKNESS, after Swink and the
   Wedgwood lineage)* — a 2–6 mm body yields ~6 carvable brightnesses, so transmission arrives in
   **stepped terraces like a topographic map**, hard-edged. This is not a compromise with the
   alpha ban; it is what the material does. **The signature is contour lines.**
4. **Both extremes are blindness** *(THE OVEREXPOSED, after Sugimoto reflected)* — unlit says
   nothing; over-lit burns out to white. There is a correct exposure per panel.
5. **Be literal, not tasteful** *(THE CONTEXT SKEPTIC, after Kalman)* — *"you have three matches"*
   beats *"attention is the scarce resource."* The embroidery game won on a **cost publicly
   borne**, not on a look. Elegance would sand off the liability that makes this shareable.

## WORLD CONTEXT, CONDENSED
A corridor of doors. You carry one dying light. Each door is a lithophane: pale and mute until you
spend light on it, then a picture rises out of the clay — thin parts first, like a print coming up
in a tray. What it shows may not match the relief you could already feel on its face. You choose a
door, you lose what you spent, you go through.

## THE FIVE CONSTRAINT AXES
Drawn per pair from `core_seed.md`: **clay body · the carving hand · the light source · what the
image is of · the failure of the object.** No pair may repeat another's draw on more than one axis.

## WHAT THIS IS NOT
- **NOT** an image printed, painted, decaled or projected onto porcelain. The relief *is* the image.
- **NOT** alpha-gradient glow, bloom, god-rays, soft falloff, lens flare. Light is **carved**, in
  hard-edged terraces. This is a blocking violation of the consuming style contract.
- **NOT** a post-process filter over ordinary 3D. The banding lives in the **height field**, in the
  asset. A look that ships as a preset dies as a preset.
- **NOT** PBR metal, plastic, subsurface-scattering skin, or wet gloss. Porcelain is matte, dry,
  slightly sugary, and it *transmits*.
- **NOT** a composed scene, a room, a HUD, typography, a score, a control, or a full environment.
  **Separable engine PARTS only.** Live code owns layout.
- **NOT** a documentary of a real pottery workshop. *(L2, HIGH confidence: realism loses to
  impossibility in object-world fields — the impossibility gradient predicts score.)*

## DAILY MANDATES
- **Thumbnail law:** every hero frame carries at least one **lit panel against dark**, with
  banding coarse enough to survive downscaling to 16px. Fine contour lines alias into mud.
- **Two states, always — scoped to the door family:** any **door-family part** (a door panel, or an
  armature shipped as the seat of one) ships as an **unlit/shadow** and a **lit/transmission** pair
  from the same height field. One asset, two behaviours. **This mandate does not reach single-state
  library parts** — a macro seam, a trim profile in section, a shatter sheet, a threshold strip. A
  section is not a state of a shatter sheet, and a pair that ships one of those without a partner is
  obeying the rule, not falling short of it. Do not invent a fifth prompt to satisfy this line.
- **No findable repeat:** wall parts must warp per-instance. A repeat the eye can locate kills the
  illusion of fired objects.
- **The imperfection must carry information**, never decorate. A firing crack that splits the
  image is a fact about the object. A smudge for texture's sake is the Concept Skeptic's "craft as
  alibi" and is rejected.

## LEGIBILITY RULE
A viewer at phone size, in five seconds, says: *"that is a slab of clay with a picture inside it,
and the picture is made of light."* If it needs a caption, it failed.

## RENDERER
**GPT-Image-2, directive mode** (Sága standard — not Flux, not noun-first). Write directive
instructions, not noun piles. Every part on a **transparent or uniform deep-navy matte**, ready for
border-connected flood-fill extraction.
