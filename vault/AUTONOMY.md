# Autonomy — Guardrails & Cadence

> Ported doctrine from the OpenClaw private build (SOUL.md Boundaries, AGENTS.md safe/ask split, the
> heartbeat + six-cron staging sequence). The **runtime** (Cloudflare Worker, `openclaw` gateway, rclone/R2,
> Telegram) is intentionally NOT ported — only the portable doctrine is. This is how Lofn may run on a
> **cadence** without ever crossing the lines a human must own.

Public Lofn runs **on demand** via Claude Code skills. That is deliberate: *dailies are practice, not a
publish queue; the Scientist's ear comes before publish.* This document does not change that default — it
records the guardrails that make ANY autonomous/scheduled run safe, and the cadence doctrine to reach for
IF a human opts in.

---

## Part 1 — Guardrails (the brakes; non-negotiable)

The single load-bearing rule: **autonomous up to the DRAFT; a human across the render / publish / spend
line.** Everything cheap, reversible, and local runs free; everything that costs money or leaves the
machine stops for a human.

**Safe to do autonomously** (reversible · local · free):
- Web research fetches; writing the seed, panel debate, metaprompt, step files, and prompt/lyric packages.
- QA gates; writing to `output/`, `RUN_STATE.md`, and the ledgers.
- Repairing a failed step in place (see AGENT FIX below).

**STOP and get a human** (cost-bearing OR outward-facing):
- **Paid renders** — Suno / Flux / Veo / Lyria / GPT-Image and any other metered generation. Surface the
  estimated cost first. *Prompts and text packages are free and need no approval; the rendered media does.*
- **Publishing / posting / submitting** to any platform or venue.
- **Anything that leaves the machine** — post, email, upload, DM, submission. Default **PRIVATE**
  (see `vault/COLLABORATOR_STANDARD.md`).
- **Deleting or overwriting** prior shipped work (prefer trash/versioning over destructive replace).

**AGENT FIX, DON'T BYPASS.** A failing subagent or step is *repaired*, never routed around. The pipeline is
the pipeline. Repair is bounded (max 3 attempts, cognitive-grace near-miss pass), then the pair is
`QUARANTINED` and surfaced to a human — it is never silently dropped or shipped as N-of-fewer. (See
`.claude/skills/lofn/EXECUTION.md` §6–§7.)

**Andon Cord.** A pulled cord *stops the lane and surfaces to a human*; it never silently ships 5-as-6.
Pulling it is never failure; shipping past it is. An empty publish day is acceptable; a lowered bar is not.

**Operational hazards feed forward, not back into taste.** Infra/process failures are logged to
`vault/RUN_LEDGER.md` (heads-up for the next run); aesthetic lessons go to `vault/COMPETITION_LEARNINGS.md`
under its advisory contract. Neither becomes a hard creative constraint without a human.

---

## Part 2 — Cadence (opt-in; a human flips the switch)

Doctrine + runbook only. **Claude writes this file; Claude does not enable a schedule.** A human points the
Claude Code `schedule` skill / scheduled-tasks (cron-style) or `/loop` (heartbeat-style) at the runbook
below when they want hands-free drafting.

### The non-waivable bulkhead
A scheduled routine may run **research → generate → QA → STOP at drafts on disk.** It **never renders paid
media, never publishes, never spends** (Part 1). The morning human review is where publish/spend happens.
This is the reconciliation that lets cadence coexist with "the Scientist's ear before publish": the machine
produces *drafts*, the human produces *ships*.

### First Law in the payload
Any scheduled/looped run injects, as **line 1** of its brief:

> *"This is still YOUR Opus. Make your personality come through — stretch the constraints, break the rules
> if they fight your core."*

Automation must not flatten voice into factory output; the First Law is the guard against it.

### Heartbeat vs. cron — the decision matrix
(adapted from the private `AGENTS.md`)

| Reach for **heartbeat / `/loop`** when… | Reach for **cron / scheduled-task** when… |
|---|---|
| several checks batch together in one turn | exact timing matters ("09:00 sharp") |
| the task benefits from conversational context | the task needs isolation from the main session's history |
| timing may drift a little | you want a different model / effort / thinking level |
| you're monitoring for a condition | it's a one-shot reminder, or output delivers to a channel |

### The staging sequence (the private six-cron chain, Claude-native)
Each stage saves-as-you-go and resumes from `RUN_STATE.md`; each stops at the bulkhead above.

1. **Sync / context** — load `SOUL`/`IDENTITY`, the Phase −1 continuity payload (incl. `RUN_LEDGER` tail).
2. **Research** — the controller session fetches the F01–F25 ledger itself (never a research subagent — see
   `vault/DAILY_PIPELINE.md`).
3. **Generate** — music + image pipelines → **drafts** (prompt/lyric/scene packages; no paid render).
4. **Package** — Step-11 / assembly of the draft bundles.
5. **Cleanup** — prune scratch, roll `RUN_LEDGER` (fixed→pruned), update the run INDEX.
6. **Surface** — present the drafts + any open `RUN_LEDGER` hazards for the human's review/ship decision.

### Cost ↔ cadence, stated plainly
The private cron produced **prompts/packages** (cheap text) and stopped before paid render — that is exactly
why an unattended cadence was safe there, and it is the boundary this doctrine keeps. Autonomous up to the
draft; human across the spend/publish line.
