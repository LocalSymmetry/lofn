# Warm Handoff Checkpoint

Agents must write this block after every major step and especially after pair steps 06–10. It preserves creative intent across timeout/restart.

```markdown
---CREATIVE_CHECKPOINT---
step_completed: [step id]
decisions_made:
  - chose: [specific creative choice]
    because: [reason rooted in seed / orchestrator direction]
rejected_alternatives:
  - rejected: [appealing alternative]
    because: [why it weakens the work]
building_toward: [what the next step should accomplish]
seed_fidelity_note: [how this preserves seed lineage]
eligibility_note: [which eligibility properties this step strengthens]
---END_CHECKPOINT---
```

Frame this as survival, not paperwork: if the agent times out, the next agent continues from the checkpoint instead of reconstructing cold.
