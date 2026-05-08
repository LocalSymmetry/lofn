# Lofn Pipeline Timeout Policy

Reliability requires slack. Running at 100% capacity guarantees brittle failure.

## Default Timeouts
- Standard full pipeline: 90 minutes hard cap.
- Competition/full production run: 120 minutes hard cap.
- Orchestrator: 15 minutes.
- Pair agents steps 06–10: 20 minutes.
- QA: 12 minutes.

## Soft Alerts
- Standard run: soft alert at 80 minutes.
- Competition run: soft alert at 105 minutes.

## Kill / Restart Criteria
Abort and restart rather than repair when:
- no step artifacts appear after expected checkpoint interval;
- generated files are stubs/placeholders;
- pair files are reconstructed without labels;
- QA cannot verify provenance;
- agent output contradicts seed packet or modality.

Repair when:
- one pair failed but checkpoints exist;
- formatting is noncompliant but content is complete;
- a bounded section needs rerun.
