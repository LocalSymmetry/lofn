---
run: 2026-07-01 (FULL DAILY — IMAGE LANE)
phase: 1 — Orchestrator, step 6 (modality handoff)
modality: image
renderer: FLUX (noun-first, present-tense, 80–150 words)
icb_path: output/daily/2026-07-01/images/CREATIVE_CONTEXT_image.md
icb_bytes: 142203
shares_phase_0_1_with: output/daily/2026-07-01/CREATIVE_CONTEXT.md (music lane) — same personality, panel, flairs, seed, metaprompt
---

# 06 — Vision Handoff (Image)

## Inject instruction (the one rule downstream steps must obey first)

**Every image step 00–10 receives `output/daily/2026-07-01/images/CREATIVE_CONTEXT_image.md` VERBATIM at the head of its prompt** — the filled image ICB, frozen after Phase 1, injected as an unbroken substring, never summarized, never paged out. It carries: the metaprompt, the Golden Seed, the complete LOFN-Prime-Mini personality YAML, the 18-voice Panel Ledger with objections, the 15 Special Flairs, the image genre/frames palettes, and the three standing image laws (composition law, human-subject discipline, thumbnail-test). A step or pair subagent that receives only "voice = Lofn" or a palette name instead of the full block is tainted and its output collapses to generic. This is the single most important rule (EXECUTION.md §3).

## Packet contents (itemized, per EXECUTION.md §3)

- (a) Complete personality YAML: LOFN-Prime-Mini, embedded verbatim in `CREATIVE_CONTEXT_image.md` (sourced from `skills/orchestration/personalities/lofn-prime-mini.yaml`, 106,219 bytes), byte-identical to the music lane's frozen copy.
- (b) All 18 panel voices with objections: Concept Panel (6, incl. Structural Skeptic after Cage), Medium Panel (6, incl. Raw-Production Skeptic after Rubin), Context & Marketing Panel (6, incl. Silence Skeptic after Cage) — the SAME panel record used for music, reused not re-derived. The Bridge transform's two domain seats (Deep-Time Semiotician after the Sandia Nuclear Semiotics task force; Boomerang Auditor after the harm-reduction PSA literature) carry into the image lane and are especially load-bearing here: the Deep-Time Semiotician governs how the wounding fact is PLACED in the frame (where the eye flinches, un-captioned) and the Boomerang Auditor governs the glossy-machine pair's named mechanism (production value reads as care reads as desirability).
- (c) All 15 Special Flairs, re-bound to visual devices in the image steps (the music bindings in `03_panel_debate.md` are the source; steps 00–05 re-anchor each flair to a composition/medium device).
- (d) The Golden Seed — `core_seed.md`, embedded in the image ICB.
- (e) The metaprompt — `04_metaprompt.md`, embedded in the image ICB.
- (f) The pair slice — `images/05_pair_assignments.md` (6 pairs, FULL-DAILY cardinality; written after coordinator step 05).

## The visual equivalents of the daily rules (the image laws, standing)

These are the music daily-mandates translated to the still frame. They govern all six pairs.

1. **Tri-source method (visual).**
   - *Source 1 — CONTENT/stakes:* the sibling supernova (a survivor form lit by the shell of the twin whose death illuminates it) is the EXISTENCE anchor; the AI anti-drug video that made harm gorgeous (the slop machine aestheticizing what it claims to warn against) is the NEWS anchor, triggering INDIGNATION. Waning gibbous 97%, Venetian Red #770011, the crappy-forums/small-warm-room warmth are available atmosphere.
   - *Source 2 — TEXTURE VOCABULARY:* the exact Bandcamp sonic phrases become MATERIAL/SURFACE instructions — "rave-sunrise-bliss vs bluesy-soul-sadness" → a warm-dawn glow bruising into a cold blue at one edge; "alien-transmission dub, bubbling fertile life" → a glossy, wet, over-processed surface with life squirming under it; "demented-joy breakcore, slithering pitch-shift" → smeared, torn, joyfully-degraded edges. Import the texture, not a genre label.
   - *Source 3 — MATERIAL STRUCTURE → COMPOSITION LAW:* **two superimposed structures of different ages must share the frame — a YOUNG bright form over an OLD slow form**, overlapping continuously in one picture plane (not two panels, not before/after). The old form is faint, diffuse, ancient, vast-and-behind; the young form is loud, sharp, recent, high-chroma, forward-or-central. This is mandatory and pair-specific — each pair realizes it with its OWN two forms.

2. **Human-subject discipline (visual).** Draw the charge, invent every particular. No identifiable real person, no name, country, "day six," or real institution/video-title/masthead/logo. Forbid IDENTIFIABILITY, not subject matter. Prefer hands, objects, backs-of-heads, invented figures. NO children as subjects. A concept undrawable without an identifiable real person is HELD FOR HUMAN. REAL GRIEF IS NOT RAW MATERIAL.

3. **Thumbnail-test legibility (the standing law).** Subject legible at first glance at thumbnail size — a stranger retells scene AND subject in one sentence after a one-second look. The two-shell structure must read as two forms, not mush. Depth lives in the second read; the first read is unmistakable. An un-nameable subject is REPAIR-FOG (the Scientist's test-slice verdict, carried into the image lane).

4. **One wounding fact, shown not captioned.** At most one numeric/scientific fact enters an image, placed where the eye already flinches (a scale reading, a horizon line, a hand) — never lettered across the frame, never explained.

5. **AWE stays terror-adjacent.** Awe frames name what could hurt the body in the exact scene, not just what is beautiful about it.

6. **Rotate the register / six different camera grammars.** No house fingerprint (frost-and-cosmos, crystalline blue, centered-hero). Six DIFFERENT compositional grammars across the six pairs; 4 variation angles per pair derived from that pair's OWN concept, never a shared template.

## Renderer & cardinality

FLUX default (TARGET_RENDERER unset). FULL DAILY: coordinator 00–05 inline → 6 pairs selected at 05 → per-pair 06–10 fan out as parallel subagents → 24 prompts (6×4) → rank → top 12 → top 6. Aspect ratio 9:16 default. No render tools called — paste-ready prompt text only.

## Advisory lessons consulted (non-competition — whispers only)

Tag-walk of `vault/COMPETITION_LEARNINGS.md` LESSONS INDEX. This is a non-competition run (NightCafe F1–F3 unavailable), so L1–L8 are advisory whispers only, applied where genuinely transferable, never as ICB constraints:
- **L6 (story legible at thumbnail speed, MEDIUM-HIGH):** transferable — adopted as the standing thumbnail-test law above (a voter/viewer must retell it in one sentence). Genuinely load-bearing given the Scientist's subject-fog verdict.
- **L7 (eco/nature → intimate reliquary > landscape, MEDIUM):** partially transferable — the small-warm-room / crappy-forums anchor and the survivor-star intimacy lean reliquary-scale (a body, a hand, a screen) over cosmic-landscape documentation, so the supernova pair frames the sky through a human-scale foreground rather than as wallpaper.
- **L9 (anti-overfit, HIGH) + the advisory contract:** honored — no lesson is hardened into the ICB; INDIGNATION-exemption respected (the glossy-machine pair is never tuned toward venue taste). L1–L5, L8 (container/portrait/fashion/branded) did not intersect this run's themes and were not applied.
