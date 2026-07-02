# Lofn Execution Policy — Concurrency + Timeouts

This policy makes agent spawning predictable and prevents max-children failures.

## Concurrency

Known constraint: `maxChildrenPerAgent` may be 5. Treat 5 as the default safe cap unless the controller has verified a higher limit.

### Pair-Agent Spawning Rule
- Standard run has 6 pairs.
- Spawn pairs 1–5 first.
- Hold pair 6 in `PENDING_CONCURRENCY_SLOT` state.
- When any one pair lands or fails, spawn pair 6.
- Never attempt a 6th simultaneous child unless the verified limit is >=6.

### Spawn Manifest
Every multi-agent run writes `spawn_manifest.json` before spawning:

```json
{
  "run_id": "YYYY-MM-DD-slug",
  "max_children": 5,
  "spawn_strategy": "staggered_5_plus_1",
  "agents": [
    {"pair": 1, "status": "spawned"},
    {"pair": 2, "status": "spawned"},
    {"pair": 3, "status": "spawned"},
    {"pair": 4, "status": "spawned"},
    {"pair": 5, "status": "spawned"},
    {"pair": 6, "status": "pending_concurrency_slot"}
  ]
}
```

## Timeouts

Use slack by default. Running at 100% capacity causes brittle failure.

| Agent / phase | Default timeout |
|---|---:|
| Orchestrator | 15 minutes |
| Coordinator steps 00–05 | 20 minutes |
| Pair agent steps 06–10 | 20 minutes |
| QA | 12 minutes |
| Standard full run | 90 minutes |
| Competition / full production run | 120 minutes |

## Soft Alerts
- Standard run: soft alert at 80 minutes.
- Competition run: soft alert at 105 minutes.

## Repair vs Restart
Restart when provenance is unverifiable, core files are missing/stubbed, or seed/modal constraints were violated.
Repair when a bounded file/section failed and warm checkpoints exist.
