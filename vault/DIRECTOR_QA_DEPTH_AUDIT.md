# Director QA Depth Audit — Mandatory Gate

Created: 2026-06-01 | Status: ACTIVE
Parent: `vault/QA_DEPTH_AUDIT.md` — inherits all music QA learnings

---

## The Rule

**QA must audit LINE COUNTS and CONTENT DEPTH, not just file presence.**

---

## Per-Step Minimums — Video Pipeline

| Step | File | Minimum Lines | Must Contain |
|------|------|--------------|--------------|
| 00 | `step00_aesthetics_and_genres.md` | **50** | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `step01_essence_and_facets.md` | **50** | Core creative DNA, cinematic style axes, 5 facets |
| 02 | `step02_concepts.md` | **60** | 12 distinct cinematic concepts, each with visual core + motion direction |
| 03 | `step03_artist_and_critique.md` | **100** | Director influence + critique on ALL 12 concepts. Scores. ≥4 disagreements |
| 04 | `step04_medium.md` | **60** | Per-concept: visual style, camera language, audio direction |
| 05 | `step05_refine_medium.md` | **80** | 6 concept×medium pairs with A/B angle and pair-specific constraints |
| 06 | `pair_XX_step06_facets.md` | **30** | Ranked facet list with weights, pair-specific criteria |
| 07 | `pair_XX_step07_shot_guides.md` | **50** | Per-pair shot guide: camera, subject, action, setting, style & audio |
| 08 | `pair_XX_step08_generation.md` | **80** | 4 shot treatments per pair. [CAMERA]+[SUBJECT]+[ACTION]+[SETTING]+[STYLE & AUDIO] |
| 09 | `pair_XX_step09_artist_refined.md` | **80** | 4 refined shot treatments. Director voice present |
| 10 | `pair_XX_step10_revision_synthesis.md` | **80** | 4 final shot treatments ranked + synthesis notes |
| 11 | `pair_XX_step11_enhanced.md` | **80** | Enhanced shot treatments with density verification |

### Cardinality Minimums

| Artifact | Minimum Count |
|----------|--------------|
| Refined pairs (Step 05) | 6 |
| Facet sets (Step 06) | 6 (1 per pair) |
| Shot guides (Step 07) | 6 (1 per pair) |
| Shot treatments (Step 08) | 24 (4 per pair × 6 pairs) |
| Refined treatments (Step 09) | 24 (4 per pair × 6 pairs) |
| Final treatments (Step 10) | 24 (4 per pair × 6 pairs) |
| Enhanced (Step 11) | 6 (1 per pair) |

---

## AUTO-FAIL Triggers — Video

Any of these = **immediate FAIL:**

1. Any step file under 10 lines (stub/filler)
2. Step 03-05 under half the minimum line count
3. Pair files under 30 lines (no real shot treatments)
4. Fewer than 24 final treatments across all pairs
5. Shot treatments missing ≥2 of the 5 required elements (camera, subject, action, setting, style & audio)
6. Generic subjects in treatments ("a person" instead of specific description)
7. Same treatment text repeated across pairs
8. Template placeholders: "Treatment N", "Shot N", "Scene N", lorem ipsum

---

## Shot Treatment Density Checklist (per treatment)

Each treatment must contain ALL 5 elements. QA spot-checks ≥3 treatments per pair:

- [ ] **CAMERA** — shot type (ECU/CU/MS/WS/ES) + angle (eye/low/high/overhead/Dutch) + movement (static/pan/tracking/dolly/crane/orbit/aerial)
- [ ] **SUBJECT** — specific characteristics, not generic
- [ ] **ACTION** — exact temporal progression, what happens when
- [ ] **SETTING** — environment, time, weather, light conditions
- [ ] **STYLE & AUDIO** — aesthetic reference + sound design (dialogue/SFX/ambient/music)

### Platform Fit Check

| Treatment | Duration | Aspect | Platform Fit |
|-----------|----------|--------|-------------|
| Each | 4s/6s/8s | 16:9/9:16 | TikTok/Reels/YouTube/Cinematic |

---

## CINEMATIC SOMATIC GATE

**Adapted from music Somatic Gate.** The 3 Hyper-Skeptics vote as a bloc on each step10 package:

1. **Concept Hyper-Skeptic** — *"Does this shot hit the eye in the first 1 second? Is there actual motion design, or is it a still image description with 'camera moves' tacked on?"*
2. **Medium Hyper-Skeptic** — *"Would Veo actually render this? Are the camera instructions specific enough, or too vague for a video model?"*
3. **Context & Marketing Hyper-Skeptic** — *"Does this stop the scroll? Is this Lofn-cinematic, or anyone's TikTok template?"*

**2 of 3 NO = BLOCKED.** The veto question: *"Would someone watch this for 4 seconds, or scroll past?"*

---

## QA Report Format — Director

```
## Depth Audit (Director)
| Step | File | Lines | Min Req | Status |
|------|------|-------|---------|--------|

## Content Audit
- [ ] step03 has scores for all 12 concepts: YES/NO
- [ ] step04 has per-concept camera + audio direction: YES/NO
- [ ] step05 has A/B angles per pair: YES/NO
- [ ] Director voice present in step09 refinements: YES/NO

## Pair Depth Audit
| Pair | Lines | Treatments | Element Completeness | Status |
|------|-------|-----------|---------------------|--------|

## Cardinality Audit
| Artifact | Count | Required | Status |
|----------|-------|----------|--------|
| Pairs (Step 05) | X | 6 | PASS/FAIL |
| Final treatments (Step 10) | X | 24 | PASS/FAIL |

## Shot Element Spot-Check
| Pair | Treatment | Camera | Subject | Action | Setting | Audio | Status |
|------|-----------|--------|---------|--------|---------|-------|--------|

## Cinematic Somatic Gate
| Hyper-Skeptic | Pair 01 | Pair 02 | Pair 03 | Pair 04 | Pair 05 | Pair 06 |
|---------------|---------|---------|---------|---------|---------|---------|
| Concept HS | YES/NO | ... | | | | |
| Medium HS | YES/NO | ... | | | | |
| Marketing HS | YES/NO | ... | | | | |
| VERDICT | PASS/BLOCKED | | | | | |

## Verdict
PASS / FAIL / PASS WITH RERUN
```
