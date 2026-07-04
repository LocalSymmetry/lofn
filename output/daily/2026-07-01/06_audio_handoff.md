---
run: 2026-07-01 (TEST SLICE)
phase: 1 — Orchestrator, step 6 (modality handoff)
modality: music
icb_path: output/daily/2026-07-01/CREATIVE_CONTEXT.md
icb_bytes: 134637
---

# 06 — Audio Handoff (Music)

## Packet contents (itemized, per EXECUTION.md §3)

- (a) Complete personality YAML: LOFN-Prime-Mini, embedded verbatim in `CREATIVE_CONTEXT.md` (106,219 bytes as sourced from `skills/orchestration/personalities/lofn-prime-mini.yaml`).
- (b) All 18 panel voices with objections: Concept Panel (6, incl. Structural Skeptic after Cage), Medium Panel (6, incl. Raw-Production Skeptic after Rubin, added at v2 re-derivation), Context & Marketing Panel (6, incl. Silence Skeptic after Cage, added at v2 re-derivation) — full text in `03_panel_debate.md` and the Panel Ledger section of `CREATIVE_CONTEXT.md`.
- (c) All 15 Special Flairs, bound to specific pair devices — see `03_panel_debate.md` §"15 Special Flairs Fixed" and the ICB footer.
- (d) The Golden Seed — `core_seed.md`, embedded in the ICB.
- (e) The metaprompt — `04_metaprompt.md`, embedded in the ICB.
- (f) The pair slice — `05_pair_assignments.md` (2 pairs, TEST SLICE cardinality).

## Golden Songs — NAMES ONLY (⛔ GOLDEN-OUTPUT QUARANTINE)

Per `EXECUTION.md` §3 and `lofn-music/SKILL.md` "The Golden Move": past golden outputs never enter a generating context. The two names below travel to every pair subagent for eventual QA blind comparison; **their payloads (style prompt / lyrics / exclude prompt) are quarantined to judge-side contexts only** (QA, step 12, the step-11 packager) and do NOT appear anywhere in this handoff, in `CREATIVE_CONTEXT.md`, or in any coordinator/pair-subagent prompt.

- **Pair 01 reference:** *Triple Arch Over Me* (public Suno staff pick; AWE / scale / scientific-sublime fit)
- **Pair 02 reference:** *Five wrong colors* (pinned public profile song; INDIGNATION / body-fracture fit)

Selected via `skills/music/references/golden_songs_index.md`'s "Public-facing staff-pick follow-up pressure" default pairing heuristic.

## THE GOLDEN MOVE (what every pair subagent receives instead of the payloads above)

The 2026-07-01 regression review distilled why "Triple Arch Over Me" won, as instructions, not an exemplar:

1. **Stand somewhere real.** The song is a report from ONE concrete place the body occupies — name where it stands and what the senses register there. Concept-illustration ("a metaphor about X") is the failure mode; experience-report is the move. If three runs in a row are indoors and safe, go outside.
2. **One wounding fact.** At most ONE numeric/scientific fact is sung, placed at the emotional hinge, and the lyric must RESPOND to it ("It says behold and calculate"), never recite it. All other research stays in the brief as atmosphere.
3. **The turn.** Somewhere past the midpoint, the song contradicts or complicates its opening stance — a mind changing in real time, an argument with itself the ending has to earn. A song that asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** AWE is terror-adjacent sublime, not domestic reassurance — every awe song carries a clean fear it does not resolve cheaply.
5. **Rotate the register.** Do not default to the house winner's fingerprint (crystalline female soprano / A major / ~110 BPM / frost-and-cosmos palette). Vary key, tempo, vocal register, and sonic world per run unless the personality's YAML mandates them. The house-lexicon FLAG (`vault/gates.yaml`) catches verbatim self-copying; this rule prevents the softer clone.

## Cardinality (TEST SLICE)

2 pairs × 2 variations = 4 songs. Pair 01 = ACCESSIBLE/EXISTENCE/AWE-terror-adjacent (V1 "Catwalk", V2 "Field"). Pair 02 = AMBITIOUS/NEWS/INDIGNATION (V1 "Train car", V2 "Corridor of loops"). Full detail in `05_pair_assignments.md`.

## Downstream execution note

Coordinator steps 00–05 run inline (this session, shared context). Per-pair steps 06–10 fan out as parallel Claude subagents, one chain per pair, each receiving the full `CREATIVE_CONTEXT.md` verbatim + this handoff + its `05_pair_assignments.md` slice + the relevant step files under `skills/music/steps/`. Step 11 enhancement runs as one subagent per pair after 06–10 lands.
