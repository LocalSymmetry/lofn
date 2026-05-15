---
name: lofn-qa
description: Audit, validate, repair, and classify completed Lofn pipeline outputs. Use after music/image/video/story pipeline completion or suspicious partial runs. Do NOT use for creative generation or finalist ranking.
---

# SKILL: Lofn QA — Router

This file is intentionally lean to prevent context collapse. The full tuned QA procedure is preserved byte-for-byte in `references/qa_full_legacy.md` and must be loaded just-in-time when an audit is actually performed.

## Workflow

1. Identify the output directory and modality.
2. Read `../orchestration/references/adversarial_qa_stance.md` to adopt the competitive-auditor stance.
3. Read `references/qa_full_legacy.md` before performing any substantive QA audit.
4. For music/audio outputs, additionally read `references/eligibility_7_properties.md` and `references/suno_15_point_qa.md`; classify ACCESSIBLE vs AMBITIOUS and run the full Suno 15-point gate.
5. If using a structured report, copy the mold from `assets/eligibility_report.template.md` and add the `Suno 15-Point QA Gate` table.
6. When deterministic eligibility scoring is available from explicit numeric scores, use `scripts/score_eligibility.py`.
7. Apply the full QA procedure exactly as specified in `references/qa_full_legacy.md`; do not summarize or weaken its tuned requirements.
8. For music/audio outputs, explicitly verify standalone Suno/Udio music prompts before any PASS verdict. This gate is non-waivable even if the parent task supplies a shorter custom checklist.
9. Save `QA_REPORT.md` in the audited output directory.
10. If failures require rerun, follow the rerun task format from `references/qa_full_legacy.md`.

## Non-Negotiables

- The legacy QA text is authoritative until fully split into smaller verified references.
- Do not remove tuned QA checks; move only after byte-for-byte preservation and validation.
- **QA stays strict.** Do not loosen file, line-count, EMO-tag, header, standalone Suno prompt, contamination, child-safety, or artifact-provenance gates to preserve “creative freedom.” Missing required structure is still a blocking failure.
- **QA is not the creative brief.** These checks are applied after creation to verify package readiness and diagnose failures. Creative agents should be led by Golden Seed / lineage / personality first, with QA requirements last as a hard output contract. If a file is structurally complete but generic, QA may mark `REPAIR — SOUL LOSS`; if a file is soulful but missing required prompt pieces, QA must still fail it structurally.
- AMBITIOUS classification is not failure unless the route target was ACCESSIBLE.
- **Music/audio PASS requires a standalone copy-paste Suno/Udio music prompt per song.** QA must fail any final music file that only contains scattered metadata such as `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, `[PRODUCTION NOTES]`, or lyrics without a dedicated `## 1. MUSIC PROMPT` / `[SUNO STYLE PROMPT:]` section.
- **Custom parent QA instructions cannot waive the music prompt gate.** If a custom checklist omits music prompts, QA must add the legacy prompt check back in and note that it did so.
- QA reports for music must include a row or bullet named `Standalone Suno/Udio music prompt present` for every song/pair.
- **Music/audio QA must run the Suno 15-Point QA Gate** from `references/suno_15_point_qa.md`. The 7 eligibility properties alone are not enough for PASS; points 8–15 verify Suno readiness, hook survivability, Lofn specificity, anti-slop, and package readiness.
