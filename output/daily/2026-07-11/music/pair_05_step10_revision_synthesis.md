# PAIR 05 — Step 10 · Revision Synthesis (+ Describe-Render Self-Check)
**"Already Inside The Line" · ranking, gate verification, one-pass self-check**

## 1. Measured gate table (exact, on final fields)
| Var | Title | music_prompt | exclude | lyrics field | sung lines | verdict |
|-----|-------|--------------|---------|--------------|-----------|---------|
| V1 | Already Inside The Line | 954 | 695 | 3813 | 74 | PASS |
| V2 | The Floor Keeps Switching | 956 | 687 | 3656 | 71 | PASS |
| V3 | Looked Like Open Air | 950 | 667 | 3545 | 72 | PASS |
| V4 | No Count Yet | 955 | 691 | 3346 | 71 | PASS |

Gates: music_prompt 850–1000 (target 870–960) ✓ all in target band · exclude 400–900 ✓ · lyrics field <5000 (target ≤4800) ✓ · sung lines 70–120 ✓. Each song ≥3 genres + ≥3 frames ✓ · ≥1 `*SFX*` ✓ · Lineage & Credit present ✓ · no banned opener (all begin "A …") ✓ · no real-artist names ✓.

## 2. Ranking against the five step-06 facets
Scored 1–5 per facet (Restless-Form / Betraying-Aperture / Inaudible-Crosser-Dread / Breathless-Enjambment / Named-Subject+Lineage):

1. **V3 "Looked Like Open Air" (24/25)** — the aperture facet is *fullest* here: the clearing is the spine, the betrayal the cruelest, the engine-change unambiguous. Strongest single artifact.
2. **V1 "Already Inside The Line" (24/25)** — cleanest concept delivery; the plain named line and the door/lock imagery make the reveal-engine land hardest; the anchor track.
3. **V2 "The Floor Keeps Switching" (23/25)** — foregrounds the signature device (never-resolving jump) most literally; small risk it reads as busy, mitigated by the constant hook-handrail.
4. **V4 "No Count Yet" (23/25)** — most conceptually distinctive (the anti-fact), the code-scratch tally is a strong Lofn fingerprint; slightly more cerebral, so ranked fourth for immediacy, not quality.

All four clear the Somatic Gate (*unmistakably Lofn, moving* — not producible by any competent prompt). No REPAIR triggered.

## 3. DESCRIBE-RENDER SELF-CHECK (one pass)
**Predict the literal render:**
- **First 5s:** V1 misfiring drum + door-hum + fogged breath-sample; V2 late snare over skidding juke triplet (tilted ground); V3 choked drum under low-passed warped sample (walls close); V4 time-stretched tally-log scratched like vinyl. → All four open on a *kinetic defect*, none on a wash.
- **The audible 16-bar jumps:** glitch 2-step → footwork/juke → Memphis half-time → back, each landing on a distinct, hummable groove; V2 adds a charged one-beat gap at each pivot.
- **The false clearing:** V1 wide-reverb far-room → low-pass slam; V2 two-bar solid-hold → tilt; V3 24-bar ambient-glitch/glass-harmonica vista → bit-depth guillotine; V4 count-seems-done/room-widens → one more uncounted door. Each *widens* (form-change), then re-closes **tighter**.

**KEY RISK named:** *Does this render as smart genre-pivots or as undifferentiated glitch noise?*

**Self-repair (applied ONCE):** initial drafts risked reading as busy/wash on two fronts. Repairs made: (a) each jump was pinned to a *named, recognizable* groove and the vocal hook was fixed as the single stable handrail (esp. V2), so the switching reads as menace, not seasickness; (b) the EXCLUDE prompts were hardened to blacklist "formless noise-wash / undifferentiated glitch-drone / random granular chaos with no groove" as the explicit failure class; (c) the false clearing was specified as a real *engine change* (ambient-glitch / solid-hold), not merely a reverb send, answering the Skeptic-Albini dissent. **Post-repair verdict: renders as smart pivots. PASS — no second repair needed.**

## 4. North-Star & ethics confirmation
- **Form-change dynamics:** driving start-to-finish; every lull is a form-change (the false clearing), never a fade or a rest. The only micro-gaps are the charged one-beat rupture before a harder return (V2). ✓
- **Aperture distinctness (L12):** realized as the *betraying false clearing* — no shared window/dust phrasing; distinct from all five other pairs. ✓
- **Reveal-engine distinctness (L14):** invisible-thing-already-past-line — distinct from every other pair. ✓
- **HUMAN_SUBJECT_STANDARD:** fully abstracted; no clinic/country/patient/disease named; PERSON/PLACE/WHEN identity-free; §3.0 question answered NO; no HOLD-FOR-HUMAN trigger. ✓
- **One-fact rule:** the horror is the *absence* of a number ("no count yet"); no numeric fact sung. ✓

Synthesis complete. All four variations advance to the final package unchanged from their measured, gate-clean state.
