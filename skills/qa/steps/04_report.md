# QA Step 4: Final Report & Delivery Decision

## Input
- All QA steps: `{output_dir}/qa_step1_completeness.md`, `{output_dir}/qa_step2_contamination.md`, `{output_dir}/qa_step3_quality.md`

## Task

### Compile the final QA report

```markdown
# QA Report — [output_dir]
**Date:** YYYY-MM-DD HH:MM
**Agent:** lofn-qa
**Status:** PASS | PASS WITH WARNINGS | FAIL

## Summary
- Files scanned: N
- Auto-fixed: N
- Rewritten: N
- Rerun triggered: N
- Final status: N PASS / N WARN / N FAIL

## File Results
[Per-file results from steps 1-3]

## Warnings (Human Review)
[Anything that needs The Scientist's eyes]

## Delivery Status
[CLEARED for delivery / NEEDS RERUN / NEEDS HUMAN REVIEW]
```

### Decision rules
- **PASS**: All files pass completeness, contamination, and quality checks
- **PASS WITH WARNINGS**: Minor issues auto-fixed, or items flagged for human review that don't block delivery
- **FAIL**: Stub pipeline, cardinality failure, missing outputs, or quality issues requiring regeneration

### If FAIL: recommend rerun
Specify which step to rerun from and why. Include the rerun task format:
```
You are Lofn — award-winning AI composer/artist. [mode].
QA flagged this output for regeneration: [reason]
Write output to: [original_filepath]
[Full concept spec from pipeline docs]
```

## Save
- Final QA report: `{output_dir}/QA_REPORT.md`

## After saving
If PASS or PASS WITH WARNINGS: deliver outputs to Telegram (channel: telegram, target: {{TELEGRAM_TARGET}})
If FAIL: report to controller for rerun decision
