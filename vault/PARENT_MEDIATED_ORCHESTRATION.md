# Parent-Mediated Orchestration Protocol
*Established: 2026-04-03*
*Status: Ready for implementation*

## Overview

All pipeline spawning flows through the **parent (main session)**. No agent spawns children. Every agent returns a structured handoff message; the parent parses it and decides the next spawn.

**Why:** OpenClaw subagents can spawn children, but nested spawning creates:
- Unobservable intermediate state (parent can't see what's happening)
- No intervention points (can't steer, pause, or redirect mid-pipeline)
- Compounding failure (child failure kills grandparent's context budget)
- Token waste (each nesting level duplicates system prompts)

Parent-mediated orchestration keeps the parent as the single control plane.

---

## 1. Feasibility

**Confirmed feasible.** OpenClaw's subagent model supports this directly:

- `sessions_spawn` returns a result string to the spawning session
- The parent receives this as an auto-announced completion event
- The parent can parse the string and call `sessions_spawn` again
- No polling needed — push-based completion

**Constraints:**
- Each handoff adds ~2-5s latency (spawn overhead). For a 5-stage pipeline, this is 10-25s total overhead — acceptable.
- The parent must hold the routing logic in its system prompt or in a readable file (this document).
- Handoff messages must be parseable — hence the structured contract below.
- Artifacts must be passed by **file path or URL**, never inline (subagent results have size limits).

---

## 2. Handoff Contract

Every agent MUST end its response with a handoff block. The block is fenced for reliable parsing.

### Format

```
---LOFN_HANDOFF---
next_action: spawn | done | error
agent: <agent_label>
model: <optional model override>
task: |
  <multi-line task description>
  Must be self-contained. The next agent has NO context from prior stages.
  Include all file paths, parameters, and constraints.
artifacts:
  - <path_or_url_1>
  - <path_or_url_2>
notes: |
  <optional context for the parent, not passed to next agent>
---END_HANDOFF---
```

### Field Definitions

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `next_action` | YES | `spawn`, `done`, `error` | What the parent should do next |
| `agent` | If spawn | String label | Which agent to spawn (e.g., `lofn-vision`, `lofn-qa`) |
| `model` | No | Model string | Override default model for this spawn |
| `task` | If spawn | Multi-line string | Complete, self-contained task for the next agent |
| `artifacts` | No | List of paths/URLs | Files produced by this agent or needed by next |
| `notes` | No | Multi-line string | Info for the parent only (logging, decisions made) |

### Rules

1. **The `task` field must be self-contained.** The next agent wakes up with zero context. Everything it needs must be in `task` or referenced as artifact paths it can read.
2. **Artifacts are file paths.** Write outputs to disk, reference by path. Never inline large content.
3. **One handoff per response.** Agents do their work, then emit exactly one handoff block at the end.
4. **The handoff block must be the last thing in the response.** The parent extracts it by finding `---LOFN_HANDOFF---` through `---END_HANDOFF---`.

### Examples

**Orchestrator completing, handing off to Vision:**
```
[... orchestrator's creative plan output ...]

---LOFN_HANDOFF---
next_action: spawn
agent: lofn-vision
task: |
  Generate a competition-ready image for the theme "Echoes of Tomorrow".
  
  Concept: A bioluminescent coral reef growing through the ruins of a 
  flooded brutalist library. Shafts of green-gold light pierce the water.
  A single reader sits cross-legged on a coral shelf, reading a book 
  whose pages have become translucent kelp.
  
  Style direction: Solarpunk realism meets underwater cathedral. 
  Muted earth tones with bioluminescent accents (cyan, amber).
  Aspect ratio: 16:9. Resolution: highest available.
  
  Read /data/.openclaw/workspace/vault/ART_SOUL.md for style constraints.
  Write output image to /data/.openclaw/workspace/output/vision_output.png
  Write the generation prompt used to /data/.openclaw/workspace/output/vision_prompt.txt
artifacts:
  - /data/.openclaw/workspace/output/orchestrator_plan.md
notes: |
  Panel of experts chose concept 3 of 5. Hyper-skeptic flagged potential 
  cliché in "underwater library" but was overruled — the coral-as-pages 
  twist tested well in panel discussion.
---END_HANDOFF---
```

**QA completing with DONE:**
```
[... QA analysis ...]

---LOFN_HANDOFF---
next_action: done
artifacts:
  - /data/.openclaw/workspace/output/vision_output.png
  - /data/.openclaw/workspace/output/qa_report.md
notes: |
  Image passed all checks. Score: 8.7/10. No contamination detected.
  Minor suggestion: slightly increase contrast in lower-right quadrant.
  Suggestion logged but not blocking — image is competition-ready.
---END_HANDOFF---
```

**Agent hitting an error:**
```
---LOFN_HANDOFF---
next_action: error
notes: |
  Image generation failed: FAL API returned 503 (service unavailable).
  Attempted 2 retries with 30s backoff. All failed.
  Recommendation: retry in 5 minutes or fall back to alternative provider.
artifacts:
  - /data/.openclaw/workspace/output/vision_prompt.txt
---END_HANDOFF---
```

---

## 3. Parent Routing Logic

The parent (main session) uses this logic after each subagent completion.

### Pseudocode

```python
import re

def parse_handoff(agent_result: str) -> dict | None:
    """Extract the LOFN_HANDOFF block from agent result."""
    match = re.search(
        r'---LOFN_HANDOFF---\s*\n(.*?)\n---END_HANDOFF---',
        agent_result,
        re.DOTALL
    )
    if not match:
        return None
    
    block = match.group(1)
    handoff = {}
    
    # Parse simple fields
    for field in ['next_action', 'agent', 'model']:
        m = re.search(rf'^{field}:\s*(.+)$', block, re.MULTILINE)
        if m:
            handoff[field] = m.group(1).strip()
    
    # Parse multi-line fields (YAML-style block scalar)
    for field in ['task', 'notes']:
        m = re.search(rf'^{field}:\s*\|\n((?:  .+\n?)*)', block, re.MULTILINE)
        if m:
            handoff[field] = '\n'.join(
                line[2:] for line in m.group(1).splitlines()
            )
    
    # Parse artifacts list
    artifacts = re.findall(r'^\s+-\s+(.+)$', 
        re.search(r'artifacts:\n((?:\s+-\s+.+\n?)*)', block, re.DOTALL).group(1)
        if re.search(r'artifacts:\n', block) else '',
        re.MULTILINE
    )
    if artifacts:
        handoff['artifacts'] = artifacts
    
    return handoff


def route_handoff(handoff: dict) -> str:
    """Decide what to do with a parsed handoff. Returns action description."""
    action = handoff.get('next_action')
    
    if action == 'done':
        # Pipeline complete. Deliver artifacts to user.
        return f"PIPELINE COMPLETE. Artifacts: {handoff.get('artifacts', [])}"
    
    if action == 'error':
        # Pipeline failed. Report to user with notes.
        return f"PIPELINE ERROR: {handoff.get('notes', 'Unknown error')}"
    
    if action == 'spawn':
        agent = handoff.get('agent')
        task = handoff.get('task')
        model = handoff.get('model')  # May be None
        
        if not agent or not task:
            return "MALFORMED HANDOFF: missing agent or task"
        
        # Map agent label to model (from ROUTING_ARCHITECTURE.md)
        MODEL_MAP = {
            'lofn-orchestrator': 'openrouter/openai/gpt-5.4',
            'lofn-vision':      'openrouter/openai/gpt-5.4',
            'lofn-audio':       'openrouter/openai/gpt-5.4',
            'lofn-director':    'openrouter/openai/gpt-5.4',
            'lofn-narrator':    'openrouter/openai/gpt-5.4',
            'lofn-evaluator':   'openrouter/openai/gpt-5.4',
            'lofn-qa':          'openrouter/anthropic/claude-sonnet-4.6',
            'lofn-architect':   'openrouter/anthropic/claude-sonnet-4.6',
            'lofn-oracle':      'openrouter/anthropic/claude-opus-4.6',
        }
        
        resolved_model = model or MODEL_MAP.get(agent)
        
        # This is the sessions_spawn call:
        # sessions_spawn(
        #     task=task,
        #     label=f"pmo-{agent}",
        #     model=resolved_model,
        #     mode="run"
        # )
        return f"SPAWN: {agent} (model: {resolved_model})"
    
    return f"UNKNOWN ACTION: {action}"


# === Main loop (conceptual — in practice this is the parent agent's behavior) ===
def pipeline_loop(initial_task: str):
    """
    The parent agent doesn't literally run this loop.
    Instead, each subagent completion triggers the parent to:
    1. Parse the handoff
    2. Route it
    3. Spawn next or deliver result
    """
    result = spawn_agent('lofn-orchestrator', initial_task)
    
    while True:
        handoff = parse_handoff(result)
        if not handoff:
            report_error("Agent returned no handoff block")
            break
        
        action = route_handoff(handoff)
        
        if handoff['next_action'] in ('done', 'error'):
            deliver_to_user(action, handoff)
            break
        
        # Spawn next agent and wait for result
        result = spawn_agent(handoff['agent'], handoff['task'], handoff.get('model'))
```

### Parent Agent Instruction (for system prompt or AGENTS.md)

When a subagent completion arrives, do the following:

1. Look for `---LOFN_HANDOFF---` in the result.
2. If `next_action: done` → deliver artifacts and summary to the user.
3. If `next_action: error` → report the error to the user with notes.
4. If `next_action: spawn` → call `sessions_spawn` with the `agent` as label, `task` as task, and model from the routing table.
5. If no handoff block found → treat as error, report raw result to user.

Do NOT modify the task text. Pass it verbatim. The originating agent crafted it to be self-contained.

---

## 4. Failure Modes

| # | Failure | Impact | Mitigation |
|---|---------|--------|------------|
| 1 | **Agent omits handoff block** | Parent can't route; pipeline stalls | Instruction enforcement in agent system prompts. Parent treats missing block as error and reports to user. |
| 2 | **Malformed handoff (bad YAML, missing fields)** | Parse failure | Lenient parser with fallback regex. If `next_action` is missing, treat as error. Log raw result for debugging. |
| 3 | **Agent hallucinates wrong agent name** | Spawn fails (unknown agent) | Validate `agent` against known agent list before spawning. Reject unknowns with error to user. |
| 4 | **Task field references files that don't exist** | Next agent fails on missing input | Each agent should verify its input artifacts exist before starting work. Fail fast with `next_action: error` if inputs are missing. |
| 5 | **Infinite loop (A → B → A → B...)** | Token burn, no progress | Parent tracks spawn history. If same agent is spawned >2x in one pipeline run, halt and report. Max pipeline depth of 10. |

### Additional Edge Cases

- **Context window exhaustion in parent:** Each handoff adds to parent context. For long pipelines (>6 stages), the parent should summarize prior handoffs rather than keeping full results.
- **Concurrent pipelines:** Label spawns with a run ID (e.g., `pmo-lofn-vision-run-20260403-165700`) to avoid confusion when multiple pipelines run simultaneously.

---

## 5. Lofn Pipeline: Parent-Mediated Mapping

### Standard Competition Pipeline

```
User Request
    ↓
Parent spawns: lofn-orchestrator
    ↓ (returns LOFN_HANDOFF → spawn lofn-vision)
Parent spawns: lofn-vision
    ↓ (returns LOFN_HANDOFF → spawn lofn-qa)
Parent spawns: lofn-qa
    ↓ (returns LOFN_HANDOFF → done)
Parent delivers result to user
```

### Extended Pipeline (with Audio/Director/Narrator)

```
User Request
    ↓
Parent spawns: lofn-orchestrator
    ↓ (returns LOFN_HANDOFF → spawn lofn-vision)
Parent spawns: lofn-vision
    ↓ (returns LOFN_HANDOFF → spawn lofn-audio)
Parent spawns: lofn-audio
    ↓ (returns LOFN_HANDOFF → spawn lofn-director)
Parent spawns: lofn-director
    ↓ (returns LOFN_HANDOFF → spawn lofn-narrator)
Parent spawns: lofn-narrator
    ↓ (returns LOFN_HANDOFF → spawn lofn-qa)
Parent spawns: lofn-qa
    ↓ (returns LOFN_HANDOFF → done | spawn packaging)
Parent delivers result to user
```

### Concrete Handoff Task Strings

#### Stage 1: Orchestrator Handoff → Vision

```
---LOFN_HANDOFF---
next_action: spawn
agent: lofn-vision
task: |
  Generate a competition image for: "{theme}"
  
  Creative brief (from orchestrator panel deliberation):
  {full creative brief — concept, mood, color palette, composition}
  
  Style constraints: Read /data/.openclaw/workspace/vault/ART_SOUL.md
  Competition format: {format requirements from competition}
  
  Output files:
  - Image: /data/.openclaw/workspace/output/{run_id}/vision_output.png
  - Prompt log: /data/.openclaw/workspace/output/{run_id}/vision_prompt.txt
  - Style notes: /data/.openclaw/workspace/output/{run_id}/vision_style.md
artifacts:
  - /data/.openclaw/workspace/output/{run_id}/orchestrator_plan.md
---END_HANDOFF---
```

#### Stage 2: Vision Handoff → QA (simple pipeline)

```
---LOFN_HANDOFF---
next_action: spawn
agent: lofn-qa
task: |
  QA audit for competition submission. Theme: "{theme}"
  
  Review the following artifacts for competition readiness:
  1. Image: /data/.openclaw/workspace/output/{run_id}/vision_output.png
  2. Creative brief: /data/.openclaw/workspace/output/{run_id}/orchestrator_plan.md
  3. Generation prompt: /data/.openclaw/workspace/output/{run_id}/vision_prompt.txt
  
  Check for:
  - Theme alignment (does image match the brief?)
  - Technical quality (artifacts, distortion, text legibility if any)
  - Contamination (watermarks, signatures, recognizable copyrighted elements)
  - Competition rule compliance: {specific rules}
  - Emotional impact rating (1-10)
  
  If PASS: return done with artifacts list and score.
  If FAIL: return spawn lofn-vision with a corrected task incorporating fixes.
  
  Write QA report to: /data/.openclaw/workspace/output/{run_id}/qa_report.md
artifacts:
  - /data/.openclaw/workspace/output/{run_id}/vision_output.png
  - /data/.openclaw/workspace/output/{run_id}/vision_prompt.txt
---END_HANDOFF---
```

#### Stage 2b: Vision Handoff → Audio (extended pipeline)

```
---LOFN_HANDOFF---
next_action: spawn
agent: lofn-audio
task: |
  Compose a soundtrack for the visual concept: "{theme}"
  
  Visual description: {brief description of what was generated}
  Mood target: {mood from orchestrator plan}
  Duration: {target duration}
  
  Read the orchestrator plan: /data/.openclaw/workspace/output/{run_id}/orchestrator_plan.md
  View the image: /data/.openclaw/workspace/output/{run_id}/vision_output.png
  
  Output: /data/.openclaw/workspace/output/{run_id}/audio_output.{ext}
  Prompt log: /data/.openclaw/workspace/output/{run_id}/audio_prompt.txt
artifacts:
  - /data/.openclaw/workspace/output/{run_id}/vision_output.png
  - /data/.openclaw/workspace/output/{run_id}/orchestrator_plan.md
---END_HANDOFF---
```

#### QA PASS → Done

```
---LOFN_HANDOFF---
next_action: done
artifacts:
  - /data/.openclaw/workspace/output/{run_id}/vision_output.png
  - /data/.openclaw/workspace/output/{run_id}/qa_report.md
  - /data/.openclaw/workspace/output/{run_id}/orchestrator_plan.md
notes: |
  Pipeline complete. QA score: 8.7/10. Image is competition-ready.
  Run ID: {run_id}
  Total pipeline stages: 3 (orchestrator → vision → qa)
---END_HANDOFF---
```

#### QA FAIL → Retry Vision

```
---LOFN_HANDOFF---
next_action: spawn
agent: lofn-vision
task: |
  REVISION REQUEST — QA failed the previous generation.
  
  Original concept: {concept from orchestrator plan}
  Previous attempt: /data/.openclaw/workspace/output/{run_id}/vision_output.png
  
  Issues found:
  {list of specific issues from QA}
  
  Corrections required:
  {specific corrections}
  
  This is attempt 2 of max 3. If you cannot resolve the issues,
  note them clearly for the next QA pass.
  
  Output: /data/.openclaw/workspace/output/{run_id}/vision_output_v2.png
  Prompt log: /data/.openclaw/workspace/output/{run_id}/vision_prompt_v2.txt
artifacts:
  - /data/.openclaw/workspace/output/{run_id}/qa_report.md
  - /data/.openclaw/workspace/output/{run_id}/vision_output.png
notes: |
  QA retry loop, attempt 2. Max 3 attempts before escalating to user.
---END_HANDOFF---
```

---

## 6. Agent System Prompt Addition

Add this to every Lofn pipeline agent's task preamble:

```
## HANDOFF PROTOCOL

You are part of a parent-mediated pipeline. You do NOT spawn other agents.
When your work is complete, you MUST end your response with a handoff block.

Format:
---LOFN_HANDOFF---
next_action: spawn | done | error
agent: <next_agent_label>
task: |
  <self-contained task for next agent — they have NO prior context>
artifacts:
  - <list of file paths you produced or they need>
notes: |
  <optional context for the parent session only>
---END_HANDOFF---

Rules:
1. The task MUST be self-contained. The next agent has zero context.
2. Reference files by full path, never inline large content.
3. Exactly one handoff block per response, at the very end.
4. If you encounter an unrecoverable error, use next_action: error.
```

---

## 7. Implementation Checklist

- [ ] Add handoff protocol instructions to each Lofn agent's system prompt template
- [ ] Add parent routing logic to main session's AGENTS.md or load from this file
- [ ] Create `/data/.openclaw/workspace/output/` directory structure convention
- [ ] Test with a minimal 2-stage pipeline (orchestrator → vision → done)
- [ ] Test QA retry loop (vision → qa-fail → vision-retry → qa-pass)
- [ ] Test error handling (agent with no handoff block)
- [ ] Add run ID generation (timestamp-based) to parent routing
- [ ] Document max-depth and max-retry limits in parent config

---

## Quick Reference

| Stage | Agent | Model | Hands off to |
|-------|-------|-------|-------------|
| Plan | `lofn-orchestrator` | gpt-5.4 | `lofn-vision` |
| Generate | `lofn-vision` | gpt-5.4 | `lofn-qa` or `lofn-audio` |
| Sound | `lofn-audio` | gpt-5.4 | `lofn-director` |
| Direct | `lofn-director` | gpt-5.4 | `lofn-narrator` |
| Narrate | `lofn-narrator` | gpt-5.4 | `lofn-qa` |
| Audit | `lofn-qa` | claude-sonnet-4.6 | `done` or retry `lofn-vision` |

**Max retries per stage:** 3
**Max pipeline depth:** 10
**Handoff sentinel:** `---LOFN_HANDOFF---` / `---END_HANDOFF---`
