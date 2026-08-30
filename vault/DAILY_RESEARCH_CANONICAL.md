# Daily Research Canonical Brief Contract
*Established: 2026-05-09*

## Principle

Daily research runs **once per day** as its own cron job. It produces the canonical daily research substrate for all creative work that day.

Creative pipelines do **not** rerun full Daily Research v2 by default. They consume the canonical brief.

## Canonical Path

For local New York date `YYYY-MM-DD`:

```text
output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md
```

Optional compatibility copy:

```text
output/daily/YYYY-MM-DD/daily-run/00_research_brief.md
```

The canonical file is the source of truth. Run-specific `00_research_brief.md` files may symlink/copy it or contain a short note pointing to it plus run-specific appendices.

## Daily Research Cron Responsibilities

The daily research cron must:

1. Read:
   - `vault/DAILY_RESEARCH_V2.md`
   - `vault/DAILY_RESEARCH_SOURCES.md`
   - `vault/COMPETITIVE_RESEARCH.md`
   - `memory/research-ledger.json` when present
   - at least one Lofn memory / continuity artifact
2. Run the full seven-lens Daily Research v2 stack.
3. Run NightCafe daily challenge lookup first as the competitive pulse input.
4. Produce a substantive canonical brief at `output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md`.
5. Update `memory/research-ledger.json` with primary lens, secondary lens, sensory lens, metaphor, emotional register, selected theme, banned repeats, matter statement, and routing.
6. Write a short status/manifest file:

```text
output/daily/YYYY-MM-DD/research/RESEARCH_STATUS.json
```

with fields:

```json
{
  "date": "YYYY-MM-DD",
  "status": "complete",
  "createdAt": "ISO timestamp",
  "canonicalBrief": "output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md",
  "selectedTheme": "...",
  "nightCafe": "...",
  "routing": "..."
}
```

## Creative Run Preflight Rule

Before any creative run writes seed lineage or spawns Lofn-Core/orchestrator/modality agents:

1. Compute today's local date in America/New_York.
2. Check for `output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md`.
3. Verify the file is substantive (default minimum: 8KB or explicit approved exception).
4. Verify `RESEARCH_STATUS.json` says `status: complete`.
5. If valid: copy/symlink/reference it as the run's `00_research_brief.md` and proceed.
6. If missing/stale/stubbed: **pause creative creation**, launch the daily research job first, and do not continue until the canonical brief exists.

## Targeted Creation Search Addendum

Every new creative run gets one small targeted search addendum after canonical daily research and before seed lineage.

Path:

```text
output/daily/YYYY-MM-DD/<run-id>/00_targeted_search_addendum.md
```

Purpose: answer **how this specific creation should be aimed**, not **what is worth making today**. The canonical daily brief already answers the second question.

### Scope Selection

The controller chooses the search target from the strongest available trigger:

1. **Competition / venue trigger** — if a NightCafe challenge, community contest, Suno release lane, platform prompt, deadline, or ruleset exists, search that specific competition/venue.
2. **User request trigger** — if The Scientist gives a direct theme, mood, lyric, modality, constraint, or desired feeling, search that request's cultural/technical/commercial context.
3. **Lofn preference trigger** — if no external target exists, search based on Lofn's creative preference for the run: medium, genre lane, archival reference, aesthetic movement, instrument, material process, or platform fit.

### Search Budget

Default: 10–15 minutes maximum, 3–6 targeted queries, 2–5 sources. This is deliberately small.

Do not rerun the full seven-lens Daily Research v2 stack.

### Required Addendum Fields

```markdown
# Targeted Creation Search Addendum — YYYY-MM-DD / <run-id>

## Trigger
- Competition / User request / Lofn preference:
- Why this target matters for this run:

## Queries Run
1. ...
2. ...

## Findings
- Venue / audience / platform signal:
- Saturation or cliché risk:
- Technical / model / production constraint:
- Opportunity or differentiator:

## Application to Seed
- Keep from canonical daily brief:
- Change or emphasize for this creation:
- Banned move:
- Concrete formal requirement:

## Sources
- URL / note
```

### Use Cases

Use targeted addenda for:
- competition mechanics, prior winners, challenge-specific tropes, or host guidance
- direct Scientist prompts that shift emotion, modality, or target audience
- Suno genre lanes, prompt format, release packaging, or current musical saturation
- image renderer constraints or platform-specific composition strategy
- Lofn's own preference when the run is freeform: a medium, genre, historical reference, or technical frontier worth leaning into

### Failure / No-Result Behavior

If targeted search fails or returns little:

- write the addendum anyway;
- state `TARGETED SEARCH LIMITED`;
- proceed from the canonical daily brief plus the known user/competition constraints;
- do not block the run unless the missing information is a hard venue rule or safety constraint.

## Run-Specific Addenda

Run-specific addenda are still allowed, but the targeted search addendum is now the default run-specific research artifact. Additional addenda should be rare and narrow.

Do not duplicate the full seven-lens scan in addenda.

## Failure Behavior

If research cannot complete:

- Write `RESEARCH_STATUS.json` with `status: failed` and a clear blocker.
- Notify The Scientist.
- Creative runs must remain paused or proceed only with explicit Scientist override.

## Rationale

Research answers: **what is worth making today?**

That answer should be stable, shared, and reusable across music, vision, story, and video runs. Re-running research per creative pipeline creates drift, duplicated cost, contradictory themes, and timeout pressure.
