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
4. For music/audio outputs, additionally read `references/eligibility_7_properties.md` and classify ACCESSIBLE vs AMBITIOUS.
5. If using a structured report, copy the mold from `assets/eligibility_report.template.md`.
6. When deterministic eligibility scoring is available from explicit numeric scores, use `scripts/score_eligibility.py`.
7. Apply the full QA procedure exactly as specified in `references/qa_full_legacy.md`; do not summarize or weaken its tuned requirements.
8. Save `QA_REPORT.md` in the audited output directory.
9. If failures require rerun, follow the rerun task format from `references/qa_full_legacy.md`.

## Non-Negotiables

- The legacy QA text is authoritative until fully split into smaller verified references.
- Do not remove tuned QA checks; move only after byte-for-byte preservation and validation.
- **QA stays strict.** Do not loosen file, line-count, EMO-tag, header, standalone Suno prompt, contamination, child-safety, or artifact-provenance gates to preserve “creative freedom.” Missing required structure is still a blocking failure.
- **QA is not the creative brief.** These checks are applied after creation to verify package readiness and diagnose failures. Creative agents should be led by Golden Seed / lineage / personality first, with QA requirements last as a hard output contract. If a file is structurally complete but generic, QA may mark `REPAIR — SOUL LOSS`; if a file is soulful but missing required prompt pieces, QA must still fail it structurally.
- AMBITIOUS classification is not failure unless the route target was ACCESSIBLE.
