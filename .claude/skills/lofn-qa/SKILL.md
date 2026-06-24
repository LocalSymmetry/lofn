---
name: lofn-qa
description: Audit, validate, and classify completed Lofn pipeline outputs against the strict quality gates, backed by Claude. Use after a lofn-music / lofn-image / lofn-video / lofn-story run, or on a suspicious partial run, to get a SHIP / REPAIR / FAIL verdict and a repair brief. Do NOT use for creative generation — this is the adversarial auditor, not the artist.
---

# Lofn QA — Claude-backed adversarial gate

The competitive auditor. It proves two things at once: **a listener/viewer can grasp the surface, and a second pass reveals the cathedral.** It fails both extremes — impressive obscurity AND competent blandness. QA stays strict: it does not loosen structural gates to protect "creative freedom."

Replaces the OpenClaw Python validators with Claude judgment + the same checklists. Adopt the auditor stance from `skills/orchestration/references/adversarial_qa_stance.md` first.

## Procedure
1. **Identify** the run directory + modality.
2. **Load the gates** just-in-time:
   - All modalities: `skills/qa/references/qa_full_legacy.md` (the full tuned procedure — authoritative).
   - **Music:** `skills/qa/references/suno_15_point_qa.md` + `skills/qa/references/eligibility_7_properties.md`. Classify ACCESSIBLE vs AMBITIOUS, then run the 16-point gate.
   - **Image:** `vault/VISION_QA_DEPTH_AUDIT.md` (Visual Somatic Gate + 7-element density checklist).
   - **Video/Animation:** `vault/DIRECTOR_QA_DEPTH_AUDIT.md` (Cinematic Somatic Gate + 5-element shot checklist).
3. **Run the gate** without weakening any check. Apply the Claude-native self-check gates in `.claude/skills/lofn/EXECUTION.md` §4 for structural/pipeline integrity.
4. **Write** `QA_REPORT.md` in the run directory. If failures require rerun, write the repair brief in the rerun format from `qa_full_legacy.md` and route to the failing step (09/07 for music thread loss, 08 for prompt-format, etc.).

## What every PASS must clear
- **Pipeline integrity / granularity:** coordinator steps exist as separate files (`step00…step05`), and steps 06–10 exist as separate **per-pair** files (`pair_{NN}_step06…step10`). A collapsed `pair_{NN}_steps_06_10.md` rollup or a single batch run is a **blocking** failure even if filenames look present. 6 pairs × 4 = 24 outputs unless the Scientist downsized.
- **Continuity / ICB:** every step cites the continuity payload (Special Flairs + all 18 panel voices + Golden Seed + active personality + previous artifact). Missing → `REPAIR — THREAD LOSS`, even if formatting passes.
- **Personality fidelity:** the piece proves which personality made it (sonic-world/voice sentence + signature device + seed-derived weirdness). If any competent prompt could have produced it → `REPAIR — SOUL LOSS`.
- **Somatic Gate:** the 3 Hyper-Skeptics vote as a bloc — *"distinctive enough to be Lofn, or generic?"* 2 of 3 NO = BLOCKED.

## Modality hard gates (non-waivable — a custom/looser parent checklist cannot waive these)
- **Music:** standalone `## 1. MUSIC PROMPT` (dense paragraph 850–1000 chars, no real artist names) + separate EXCLUDE field; lyrics open `[Theme:…]` then `[SONG FORM:…]`; full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` (taxonomy emotions, not bare AWE/INDIGNATION); ≥1 SFX cue; 70–120 sung lines; verse-structure diversity across the 6; Lineage & Credit block on living-scene genres. Run the full 7 Singer-Surface + 5 Cathedral-Engine + 3 Suno-Package + Lineage gates. *(Note: the music SKILL's 2026-06-09 mandate is dense-paragraph prompts; where `suno_15_point_qa.md` still says "categorized key:value", the paragraph mandate is the newer authority — flag the conflict, don't fail a correct paragraph prompt for it.)*
- **Image:** noun-first present-tense scene description, no imperative openers, medium named early, emotion shown not named, no Storybook-Assassin ban-words ("ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate"), no living-artist names, subject legible at first glance, ≥80 words (Flux) / five-slot directive (GPT_I2).
- **Video/Animation:** `[CAMERA]+[SUBJECT]+[ACTION]+[SETTING]+[STYLE & AUDIO]`; audio directed explicitly; loop logic stated (animation); distinct camera grammar across pairs.
- **Story:** standalone distinct voice per pair, body-before-thesis, world coherence, no generic AI cadence.

## Verdicts (every report)
- **Pipeline Integrity Verdict:** PASS / REPAIR REQUIRED / FAIL
- **Package Verdict** (modality contract): PASS / REPAIR REQUIRED / FAIL
- **Overall:** SHIP / REPAIR / FAIL

## Report format (`QA_REPORT.md`)
```markdown
# Lofn QA Report — <run> (<modality>)
## Verdicts
- Pipeline Integrity: …
- Package: …
- Overall: …
## Score Table
| # | Gate | Verdict | Evidence | Repair |
## Blocking Fails
## Somatic Gate (3 Hyper-Skeptics)
## Required Repairs  (routed to step N)
## Final Recommendation
```

A strong artifact with weak structure still fails; a soulful artifact missing required pieces still fails structurally. Fix, then re-QA before delivery.
