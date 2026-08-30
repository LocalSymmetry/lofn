# Director QA Depth Audit — Mandatory Gate

Created: 2026-06-01 | Updated: 2026-08-15 | Status: ACTIVE
Parent: `vault/QA_DEPTH_AUDIT.md` — inherits all music QA learnings
Binds with: `vault/VIDEO_RENDER_AUDIT.md` (three-pass blind rule) · `vault/VIDEO_MODEL_DOSSIER.yaml` (rates/caps) · `scripts/video_cost_preflight.py` (cost gate)

---

## The Rule

**QA must audit LINE COUNTS and CONTENT DEPTH, not just file presence.**

**And as of 2026-08-15 it must audit THE RENDER, not only the text that asked for it.** Every gate this house owned for video was textual. That was survivable while video cost nothing and shipped as prompt text; there is now a funded fal account. See §THREE-PASS below.

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

**Added 2026-08-15:**

9. **No named renderer.** Any treatment whose frontmatter lacks `renderer:` / `endpoint:` / `tier:` / `duration_s:` / `aspect:`. A prompt with no named renderer is not a prompt, it is a paragraph.
10. **Wrong grammar for the named renderer.** Timed shot blocks written for Veo 3.1 (which renders one continuous shot and ignores them); a bare appearance description handed to Kling v3; references cited with no declared job on MiniMax H3. Check `notation_grammars:` in the dossier.
11. **Duration exceeds the named model's cap** (`max_duration_s` in the dossier) — the call would fail or be clamped.
12. **A render exists with no `<run-dir>/VIDEO_RENDER_AUDIT.md` row.** UNAUDITED renders may not be ranked, selected, or shipped.
13. **Any render whose audit row reads FALSE WELD or NO EVENT** appearing in the selected set.
14. **A render plan with no pasted `video_cost_preflight.py` output**, or one whose preflight exited non-zero.
15. **Two pairs sharing an archetype chain** (`chain:` in frontmatter) — the distinctiveness rule, extended to chains.
16. **Missing dedication** on a shipped piece (`vault/DEDICATION_STANDARD.md`) — a missing player credit is a REPAIR, not a note.

---

## Shot Treatment Density Checklist (per treatment)

Each treatment must contain ALL 5 elements. QA spot-checks ≥3 treatments per pair:

- [ ] **CAMERA** — shot type (ECU/CU/MS/WS/ES) + angle (eye/low/high/overhead/Dutch) + movement (static/pan/tracking/dolly/crane/orbit/aerial)
- [ ] **SUBJECT** — specific characteristics, not generic
- [ ] **ACTION** — exact temporal progression, what happens when
- [ ] **SETTING** — environment, time, weather, light conditions
- [ ] **STYLE & AUDIO** — aesthetic reference + sound design (dialogue/SFX/ambient/music)
- [ ] **RENDERER** — named model + endpoint + tier, and the prompt written in that model's grammar
- [ ] **ARCHETYPE / CHAIN** — the motion verb, or the chain with named junction timestamps

### Platform Fit Check

| Treatment | Duration | Aspect | Platform Fit |
|-----------|----------|--------|-------------|
| Each | within the model's cap (8s Veo · 10s Kling/Omni · 15s Seedance 2.0/H3 · 20s LTX · 30s Seedance 2.5) | 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9 | TikTok / Reels / Shorts / **Spotify Canvas (9:16, 3–8s)** / Cinematic |

---

## ⛔ THREE-PASS RENDER AUDIT — required before ranking

`vault/VIDEO_RENDER_AUDIT.md` is binding on every render:

| Pass | Input | Question |
|---|---|---|
| **A** | video, **muted**, prompt unseen | ***"What happened?"*** — one sentence, event only |
| **B** | audio only, prompt unseen | ***"What happened?"*** |
| **C** | both, then the prompt | intent survival, GRAIN-LAW deviations, craft |

**A and B name different events → FALSE WELD → repair or cut.** A names a style or technique instead of an event → **NO EVENT** → repair the notation, not the render.

The reason this is not optional: picture and sound are generated in the same pass and are coincident by construction, so **synchresis** welds them involuntarily and the **McGurk effect** persists even in reviewers who know about it. A fused *"does the audio serve the image?"* check **cannot return NO** — it is a gate that cannot fail, which is not a gate.

QA reads `<run-dir>/VIDEO_RENDER_AUDIT.md`, never a claim. *"I watched it, it was fine"* is the sound a gate makes while it fails.

---

## CINEMATIC SOMATIC GATE

**Adapted from music Somatic Gate.** The 3 Hyper-Skeptics vote as a bloc on each step10 package:

1. **Concept Hyper-Skeptic** — *"Does this shot hit the eye in the first 1 second? Is there actual motion design, or is it a still image description with 'camera moves' tacked on?"*
2. **Medium Hyper-Skeptic** — *"Would the NAMED renderer actually execute this, in ITS grammar? Are the instructions physical enough for Kling, job-assigned enough for H3, timed for Seedance — or are they Veo prose wearing another model's name?"*
3. **Context & Marketing Hyper-Skeptic** — *"Does this stop the scroll? Is this Lofn-cinematic, or anyone's TikTok template?"*

**2 of 3 NO = BLOCKED.** The veto question: *"Would someone watch this for 4 seconds, or scroll past?"*

⚠️ **What this gate does NOT catch:** averageness. A clip where Pass A and Pass B agree perfectly and both describe something utterly generic passes the three-pass rule cleanly. Passing is a **floor**, not a verdict — distinctiveness and the Somatic bloc still own that call.

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
| Pair | Treatment | Camera | Subject | Action | Setting | Audio | Renderer | Chain | Status |
|------|-----------|--------|---------|--------|---------|-------|----------|-------|--------|

## Renderer & Cost Audit
| Item | Value | Status |
|------|-------|--------|
| Renderer named in every treatment | X/24 | PASS/FAIL |
| Grammar matches renderer | X/24 | PASS/FAIL |
| Duration within model cap | X/24 | PASS/FAIL |
| Distinct chains across pairs | X/6 | PASS/FAIL |
| Preflight output pasted, exit 0 | YES/NO | PASS/FAIL |
| Ladder families match (or --bakeoff declared) | YES/NO | PASS/FAIL |

## Three-Pass Render Audit (renders only)
| Clip | Pass A (muted): what happened | Pass B (audio): what happened | A=B? | Verdict |
|------|-------------------------------|------------------------------|------|---------|

Prediction logged before Pass A: ______   Outcome: ______
*(The Goodhart guard: a gate that is never checked against its own prediction cannot be shown to have earned itself.)*

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
