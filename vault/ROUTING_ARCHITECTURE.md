# Routing Architecture — OpenClaw Intelligent Triage
*Established: 2026-03-25*

## Principle
**Triage & Delegate.** Handle cheap at the edge. Escalate only what earns it. Never compromise Lofn.

---

## Model Routing Table

| Role | Agent | Model | When to use |
|---|---|---|---|
| **Main / Triage** | `main` | `openrouter/openai/gpt-5.4-mini` | Default: chat, Q&A, summarization, file ops, routing decisions, heartbeats |
| **Creative Engine** | `lofn-orchestrator`, `lofn-vision`, `lofn-audio`, `lofn-director`, `lofn-narrator`, `lofn-evaluator` | `openrouter/openai/gpt-5.4` | All Lofn creative work: art, music, video, story, competition pipelines |
| **QA Gate** | `lofn-qa` | `openrouter/anthropic/claude-sonnet-4.6` | Post-pipeline audit, contamination checks, fix generation |
| **Architect** | `lofn-architect` | `openrouter/anthropic/claude-sonnet-4.6` | Code, scripts, DevOps, Docker, infra planning, debugging |
| **Oracle** | `lofn-oracle` | `openrouter/anthropic/claude-opus-4.6` | Complex reasoning, multi-step strategy, deep research synthesis, hard blockers |

---

## Routing Rules

### Main agent (gpt-5.4-mini) handles directly:
- General conversation, casual Q&A
- Summarizing logs or outputs
- File reading, formatting, basic data extraction
- Clarifying questions
- Heartbeats (cheap + fast)
- Routing decisions (determine which subagent to spawn)

### Escalate to lofn-orchestrator → lofn-vision/audio/etc. (gpt-5.4):
- Any Lofn creative pipeline request
- Art competition runs
- Music composition
- Video/story generation
- Panel of experts debates

### Escalate to lofn-architect (claude-sonnet-4.6):
- Script writing or debugging
- Docker/container configuration
- API integrations
- Infrastructure changes
- Code review

### Escalate to lofn-oracle (claude-opus-4.6):
- Multi-step strategic planning
- Complex logic or math
- Deep research synthesis across multiple sources
- When lofn-architect hits a hard blocker

---

## Token Optimization Rules

1. **Context pruning when spawning subagents:** Pass only what the subagent needs — not the full conversation history. Summarize the goal.
2. **Strict output format:** Tell subagents to return only the requested artifact, no filler.
3. **Plan-then-delegate:** For multi-step tasks, use Oracle for the plan, mini/sonnet for execution.
4. **No streaming for internal subagent calls:** Wait for full block, extract result, move on.
5. **Lofn is sacred:** Never downsample the creative pipeline to save money. gpt-5.4 stays for all creative agents, Opus stays available for oracle tasks.

---

## Non-Negotiables
- All 6 Lofn creative subagents stay on gpt-5.4 — no downgrade for cost
- lofn-qa stays on claude-sonnet-4.6 — careful auditing needs it
- lofn-oracle uses Opus — deep thinking earns the spend
- Heartbeats run on main (gpt-5.4-mini) — fast, cheap, sufficient
