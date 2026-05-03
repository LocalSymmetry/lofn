---
name: lofn-qa
description: Audit, validate, repair, and classify completed Lofn pipeline outputs. Use after music/image/video/story pipeline completion or suspicious partial runs. Do NOT use for creative generation or finalist ranking.
---

# SKILL: Lofn QA — Router

This file is intentionally lean to prevent context collapse. The full tuned QA procedure is preserved byte-for-byte in `references/qa_full_legacy.md` and must be loaded just-in-time when an audit is actually performed.

## Workflow

1. Identify the output directory and modality.
2. Read `references/qa_full_legacy.md` before performing any substantive QA audit.
3. For music/audio outputs, additionally read `references/eligibility_7_properties.md` and classify ACCESSIBLE vs AMBITIOUS.
4. If using a structured report, copy the mold from `assets/eligibility_report.template.md`.
5. When deterministic eligibility scoring is available from explicit numeric scores, use `scripts/score_eligibility.py`.
6. Apply the full QA procedure exactly as specified in `references/qa_full_legacy.md`; do not summarize or weaken its tuned requirements.
7. Save `QA_REPORT.md` in the audited output directory.
8. If failures require rerun, follow the rerun task format from `references/qa_full_legacy.md`.

## Non-Negotiables

- The legacy QA text is authoritative until fully split into smaller verified references.
- Do not remove tuned QA checks; move only after byte-for-byte preservation and validation.
- AMBITIOUS classification is not failure unless the route target was ACCESSIBLE.
